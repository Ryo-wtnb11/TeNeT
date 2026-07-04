//! Topology-keyed contraction-plan cache for the `tensor!` /
//! [`Network::contract`] path.
//!
//! The cache key is the network *topology*: per-operand label lists, conj
//! flags, codomain ranks and written `;` splits, plus the output labels and
//! the [`Optimizer`] choice. Leg dimensions are deliberately NOT part of the
//! key: a pairwise contraction order is correct for any dimensions, and
//! truncation drifts bond dimensions every sweep — an exact-dims key would
//! miss every iteration. Each entry stores the dimensions it was planned
//! under and its estimated cost; the [`ReplanPolicy`] decides when drift has
//! grown enough to re-plan.
//!
//! Storage is thread-local. The preferred owner is the user-layer `Runtime`
//! (one cache per runtime, configured on `Runtime::builder()`), but the plan
//! types live in this crate and `tenet` cannot depend on it; a thread-local
//! in this crate gives the same behavior for the user layer's documented
//! single-threaded driving model. Moving ownership onto the `Runtime`
//! builder is a follow-up once a type-erased cache slot lands there.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tenet::prelude::{Error, Tensor};

use crate::labels::TemporaryLabel;
use crate::network::{Network, PlannedNetwork};
use crate::optimizer::GreedyDenseOptimizer;

/// Which contraction-order search to run: a hashable value type (usable as
/// a cache-key component and as a process-wide default) rather than a trait
/// object. `#[non_exhaustive]` so future external searches (e.g. a
/// cotengrust adapter variant carrying its config) slot in without a
/// breaking change.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Optimizer {
    /// Greedy pairwise search ([`GreedyDenseOptimizer`]); the default.
    #[default]
    Greedy,
    /// Exhaustive optimal search (opt_einsum `"optimal"`; small networks
    /// only).
    #[cfg(feature = "opt-path")]
    Optimal,
}

/// When to re-plan a topology-matched cache entry whose leg dimensions have
/// drifted from the snapshot it was planned under. Reusing is always
/// *correct* (a pairwise order is dimension-independent); re-planning only
/// restores *optimality*.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReplanPolicy {
    /// Always reuse the cached order, whatever the current dimensions.
    AlwaysReuse,
    /// Re-plan when any leg dimension differs from the snapshot by more
    /// than this factor (as a ratio, in either direction).
    DriftFactor(f64),
}

/// Default [`ReplanPolicy::DriftFactor`].
///
/// Rationale: truncation moves bond dimensions by a few percent per sweep,
/// which never changes which pairwise order wins, so those calls must hit.
/// Once some leg has grown or shrunk past 2x its planning-time value the
/// network's cost balance has changed qualitatively and a fresh (cheap for
/// greedy, expensive for exhaustive — which is exactly when hits matter)
/// search is worth it.
pub const DEFAULT_REPLAN_DRIFT_FACTOR: f64 = 2.0;

impl Default for ReplanPolicy {
    fn default() -> Self {
        Self::DriftFactor(DEFAULT_REPLAN_DRIFT_FACTOR)
    }
}

/// Default maximum number of cached plans (per thread).
///
/// Rationale: an entry is plan metadata only (label lists, a step list and
/// a dims snapshot — well under a kilobyte for realistic networks), so 256
/// bounds the cache to a few hundred kilobytes while covering drivers that
/// cycle through many distinct expressions (e.g. every bond of a large
/// unit cell) without eviction thrash.
pub const DEFAULT_PLAN_CACHE_CAPACITY: usize = 256;

/// Plan-cache behavior; set with [`configure_plan_cache`].
#[derive(Clone, Debug)]
pub struct PlanCacheConfig {
    /// Master switch; `false` makes every [`Network::contract`] plan fresh.
    pub enabled: bool,
    /// Maximum cached entries before eviction.
    pub capacity: usize,
    /// When to re-plan on dimension drift.
    pub replan: ReplanPolicy,
    /// Default optimizer for [`Network::contract`] (the `tensor!` path).
    pub optimizer: Optimizer,
}

impl Default for PlanCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capacity: DEFAULT_PLAN_CACHE_CAPACITY,
            replan: ReplanPolicy::default(),
            optimizer: Optimizer::default(),
        }
    }
}

/// Counters for tests and diagnostics; see [`plan_cache_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanCacheStats {
    /// Topology hits that reused the cached order.
    pub hits: u64,
    /// Topology misses (planned and inserted fresh).
    pub misses: u64,
    /// Topology hits re-planned because dimension drift exceeded the
    /// [`ReplanPolicy`].
    pub replans: u64,
    /// Current number of cached plans.
    pub entries: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct OperandTopology {
    labels: Vec<TemporaryLabel>,
    conj: bool,
    /// Codomain rank of the operand tensor: it fixes the conj label
    /// rotation, so it is structural even though it is not a label.
    codomain_rank: usize,
    written_split: Option<usize>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct NetworkTopology {
    operands: Vec<OperandTopology>,
    output: Vec<TemporaryLabel>,
    output_codomain_rank: Option<usize>,
    optimizer: Optimizer,
}

struct CacheEntry {
    planned: Arc<PlannedNetwork>,
    /// Flat leg dims per operand at plan time (written leg order).
    dims_snapshot: Vec<Vec<usize>>,
    /// Estimated total plan cost at plan time (kept for diagnostics and
    /// future cost-aware policies).
    #[allow(dead_code)]
    cost: usize,
}

#[derive(Default)]
struct PlanCache {
    config: PlanCacheConfig,
    hits: u64,
    misses: u64,
    replans: u64,
    map: HashMap<NetworkTopology, CacheEntry>,
    /// Least-recently-used key first; same mechanism as the resolution
    /// cache in `tenet-tensors` (whose helpers are `pub(crate)` there, so
    /// the two small operations are replicated below).
    lru_order: VecDeque<NetworkTopology>,
}

/// Move `key` to the most-recently-used end of `order`.
fn touch_lru_key(order: &mut VecDeque<NetworkTopology>, key: &NetworkTopology) {
    if let Some(position) = order.iter().position(|stored| stored == key) {
        order.remove(position);
    }
    order.push_back(key.clone());
}

thread_local! {
    static PLAN_CACHE: RefCell<PlanCache> = RefCell::new(PlanCache::default());
}

/// Replaces the plan-cache configuration (thread-local, like the cache).
pub fn configure_plan_cache(config: PlanCacheConfig) {
    PLAN_CACHE.with(|cache| cache.borrow_mut().config = config);
}

/// The current plan-cache configuration.
pub fn plan_cache_config() -> PlanCacheConfig {
    PLAN_CACHE.with(|cache| cache.borrow().config.clone())
}

/// Hit/miss/re-plan counters and the current entry count.
pub fn plan_cache_stats() -> PlanCacheStats {
    PLAN_CACHE.with(|cache| {
        let cache = cache.borrow();
        PlanCacheStats {
            hits: cache.hits,
            misses: cache.misses,
            replans: cache.replans,
            entries: cache.map.len(),
        }
    })
}

/// Drops every cached plan and resets the counters (not the configuration).
pub fn clear_plan_cache() {
    PLAN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.map.clear();
        cache.lru_order.clear();
        cache.hits = 0;
        cache.misses = 0;
        cache.replans = 0;
    });
}

fn drifted(policy: ReplanPolicy, snapshot: &[Vec<usize>], current: &[Vec<usize>]) -> bool {
    match policy {
        ReplanPolicy::AlwaysReuse => false,
        ReplanPolicy::DriftFactor(factor) => snapshot
            .iter()
            .flatten()
            .zip(current.iter().flatten())
            .any(|(&snap, &cur)| {
                if snap == cur {
                    return false;
                }
                if snap == 0 || cur == 0 {
                    return true;
                }
                let ratio = snap.max(cur) as f64 / snap.min(cur) as f64;
                ratio > factor
            }),
    }
}

fn plan_fresh(
    network: &Network,
    tensors: &[&Tensor],
    optimizer: &Optimizer,
) -> Result<PlannedNetwork, Error> {
    match optimizer {
        Optimizer::Greedy => network.plan(tensors, &GreedyDenseOptimizer),
        #[cfg(feature = "opt-path")]
        Optimizer::Optimal => network.plan(
            tensors,
            &crate::pathopt::OptEinsumPathOptimizer::new(crate::pathopt::PathStrategy::Optimal),
        ),
    }
}

/// Cache-aware planning for [`Network::contract`]: reuse a topology-matched
/// plan (subject to the drift policy), otherwise plan fresh and cache.
pub(crate) fn get_or_plan(
    network: &Network,
    tensors: &[&Tensor],
    optimizer: &Optimizer,
) -> Result<Arc<PlannedNetwork>, Error> {
    if !PLAN_CACHE.with(|cache| cache.borrow().config.enabled) {
        return Ok(Arc::new(plan_fresh(network, tensors, optimizer)?));
    }

    let dims: Vec<Vec<usize>> = tensors
        .iter()
        .map(|tensor| tensor.leg_dims())
        .collect::<Result<_, _>>()?;
    let topology = NetworkTopology {
        operands: network
            .inputs
            .iter()
            .zip(&network.conj)
            .zip(&network.codomain_splits)
            .zip(tensors)
            .map(
                |(((labels, &conj), &written_split), tensor)| OperandTopology {
                    labels: labels.clone(),
                    conj,
                    codomain_rank: tensor.codomain_rank(),
                    written_split,
                },
            )
            .collect(),
        output: network.output.clone(),
        output_codomain_rank: network.output_codomain_rank,
        optimizer: optimizer.clone(),
    };

    enum Outcome {
        Hit(Arc<PlannedNetwork>),
        Replan,
        Miss,
    }
    let outcome = PLAN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let policy = cache.config.replan;
        match cache.map.get(&topology) {
            Some(entry) if !drifted(policy, &entry.dims_snapshot, &dims) => {
                let planned = Arc::clone(&entry.planned);
                cache.hits += 1;
                touch_lru_key(&mut cache.lru_order, &topology);
                Outcome::Hit(planned)
            }
            Some(_) => Outcome::Replan,
            None => Outcome::Miss,
        }
    });
    if let Outcome::Hit(planned) = outcome {
        return Ok(planned);
    }

    let planned = Arc::new(plan_fresh(network, tensors, optimizer)?);
    let cost = planned.plan().total_cost();
    PLAN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        match outcome {
            Outcome::Replan => cache.replans += 1,
            _ => cache.misses += 1,
        }
        cache.map.insert(
            topology.clone(),
            CacheEntry {
                planned: Arc::clone(&planned),
                dims_snapshot: dims,
                cost,
            },
        );
        touch_lru_key(&mut cache.lru_order, &topology);
        while cache.map.len() > cache.config.capacity {
            let Some(oldest) = cache.lru_order.pop_front() else {
                break;
            };
            cache.map.remove(&oldest);
        }
    });
    Ok(planned)
}
