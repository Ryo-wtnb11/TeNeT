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
//! grown enough to re-plan. Eviction is LRU (same mechanism as the
//! resolution cache in `tenet-tensors`).
//!
//! Storage is per-[`Runtime`]: the configuration value types live in
//! `tenet::plancache` (set them on `Runtime::builder()` or with
//! [`configure_plan_cache`]), and the cache state sits in the runtime's
//! type-erased plan-cache slot, claimed and downcast by this crate. The
//! operands' runtime is resolved per call, so different runtimes never share
//! plans or counters.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tenet::prelude::{Error, Runtime, Tensor};

pub use tenet::plancache::{
    Optimizer, PlanCacheConfig, PlanCacheStats, ReplanPolicy, DEFAULT_PLAN_CACHE_CAPACITY,
    DEFAULT_REPLAN_DRIFT_FACTOR,
};

use crate::labels::TemporaryLabel;
use crate::network::{Network, PlannedNetwork};
use crate::optimizer::GreedyDenseOptimizer;

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
    hits: u64,
    misses: u64,
    replans: u64,
    map: HashMap<NetworkTopology, CacheEntry>,
    /// Least-recently-used key first; same mechanism as the resolution
    /// cache in `tenet-tensors` (whose helpers are `pub(crate)` there, so
    /// the two small operations are replicated below).
    lru_order: VecDeque<NetworkTopology>,
}

// The cache lives in the runtime's `dyn Any + Send` slot; plans are
// step lists + label vectors, so this holds by construction.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<PlanCache>();
};

/// Move `key` to the most-recently-used end of `order`.
fn touch_lru_key(order: &mut VecDeque<NetworkTopology>, key: &NetworkTopology) {
    if let Some(position) = order.iter().position(|stored| stored == key) {
        order.remove(position);
    }
    order.push_back(key.clone());
}

/// The runtime slot's cache, claimed (created) on first use.
fn cache_mut(slot: &mut Option<Box<dyn Any + Send>>) -> &mut PlanCache {
    slot.get_or_insert_with(|| Box::new(PlanCache::default()))
        .downcast_mut::<PlanCache>()
        .expect("runtime plan-cache slot claimed by another type")
}

/// Replaces the runtime's plan-cache configuration (the builder-time
/// equivalent is `Runtime::builder().plan_cache(config)`).
pub fn configure_plan_cache(runtime: &Runtime, config: PlanCacheConfig) {
    runtime.set_plan_cache_config(config);
}

/// The runtime's current plan-cache configuration.
pub fn plan_cache_config(runtime: &Runtime) -> PlanCacheConfig {
    runtime.plan_cache_config()
}

/// Hit/miss/re-plan counters and the current entry count.
pub fn plan_cache_stats(runtime: &Runtime) -> PlanCacheStats {
    runtime.with_extension_slot(|slot| {
        let cache = cache_mut(slot);
        PlanCacheStats {
            hits: cache.hits,
            misses: cache.misses,
            replans: cache.replans,
            entries: cache.map.len(),
        }
    })
}

/// Drops every cached plan and resets the counters (not the configuration).
pub fn clear_plan_cache(runtime: &Runtime) {
    runtime.with_extension_slot(|slot| {
        let cache = cache_mut(slot);
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
        // `Optimizer` is #[non_exhaustive] and defined in `tenet`; variants
        // this build has no search for (e.g. Optimal without `opt-path`)
        // are an explicit error rather than a silent greedy fallback.
        #[allow(unreachable_patterns)]
        other => Err(Error::InvalidArgument(format!(
            "optimizer {other:?} is not available in this build \
             (is the `opt-path` feature enabled?)"
        ))),
    }
}

/// Cache-aware planning for [`Network::contract`]: reuse a topology-matched
/// plan from the operands' runtime (subject to the drift policy), otherwise
/// plan fresh and cache.
pub(crate) fn get_or_plan(
    network: &Network,
    tensors: &[&Tensor],
    optimizer: &Optimizer,
) -> Result<Arc<PlannedNetwork>, Error> {
    // The cache lives on the operands' runtime; step execution would reject
    // mixed-runtime operands anyway, so the first operand's runtime is it.
    let Some(runtime) = tensors.first().map(|tensor| tensor.runtime()) else {
        return Ok(Arc::new(plan_fresh(network, tensors, optimizer)?));
    };
    let config = runtime.plan_cache_config();
    if !config.enabled {
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
    let outcome = runtime.with_extension_slot(|slot| {
        let cache = cache_mut(slot);
        match cache.map.get(&topology) {
            Some(entry) if !drifted(config.replan, &entry.dims_snapshot, &dims) => {
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
    runtime.with_extension_slot(|slot| {
        let cache = cache_mut(slot);
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
        while cache.map.len() > config.capacity {
            let Some(oldest) = cache.lru_order.pop_front() else {
                break;
            };
            cache.map.remove(&oldest);
        }
    });
    Ok(planned)
}
