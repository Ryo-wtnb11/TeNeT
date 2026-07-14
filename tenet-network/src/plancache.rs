//! Topology-keyed contraction-plan cache for the `tensor!` /
//! [`Network::contract`] path.
//!
//! The cache key is the network *topology*: per-operand label lists, conj
//! flags, codomain ranks and written `;` splits, plus the output labels and
//! the [`Optimizer`] choice. Leg dimensions are deliberately NOT part of the
//! key: a pairwise contraction order is correct for any dimensions, and
//! truncation drifts bond dimensions every sweep — an exact-dims key would
//! miss every iteration. Each entry stores the dimensions it was planned
//! under; the [`ReplanPolicy`] decides whether a dimension change forces a
//! re-plan. The default ([`ReplanPolicy::BakeOnce`]) finds the order once at
//! real dims and reuses it for any later dims — the standard "search once,
//! reuse the path regardless of rank" design (cotengra's reusable
//! `ContractionTree`, `@tensoropt`'s compile-time bake) — so the
//! (χ-dependent) order search is paid at most once per topology, not per χ.
//! Eviction is LRU (same mechanism as the resolution cache in
//! `tenet-tensors`).
//!
//! Storage is per-[`Runtime`]: the configuration value types live in
//! `tenet::plancache` (set them on `Runtime::builder()` or with
//! [`configure_plan_cache`]), and the cache state sits in the runtime's
//! type-erased plan-cache slot, claimed and downcast by this crate. The
//! operands' runtime is resolved per call, so different runtimes never share
//! plans or counters.

use std::any::Any;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use lru::LruCache;
use tenet::prelude::{Error, Runtime, Tensor};

pub use tenet::plancache::{
    Optimizer, PlanCacheConfig, PlanCacheStats, ReplanPolicy, DEFAULT_PLAN_CACHE_CAPACITY,
    DEFAULT_REPLAN_DRIFT_FACTOR,
};

use crate::labels::TemporaryLabel;
use crate::network::{Network, NetworkExecutionWorkspace, PlannedNetwork, StaticTopologySpec};
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
    workspaces: Arc<WorkspacePool>,
    /// Flat leg dims per operand at plan time (written leg order).
    dims_snapshot: Vec<Vec<usize>>,
}

#[derive(Default)]
struct WorkspacePool {
    available: Mutex<Vec<NetworkExecutionWorkspace>>,
    created: AtomicU64,
    reused: AtomicU64,
    slot_grows: AtomicU64,
}

const MAX_IDLE_WORKSPACES_PER_PLAN: usize = 2;

struct WorkspaceLease {
    pool: Arc<WorkspacePool>,
    workspace: Option<NetworkExecutionWorkspace>,
}

impl WorkspacePool {
    fn lease(self: &Arc<Self>) -> WorkspaceLease {
        let workspace = self
            .available
            .lock()
            .expect("network workspace pool poisoned")
            .pop();
        let workspace = match workspace {
            Some(workspace) => {
                self.reused.fetch_add(1, Ordering::Relaxed);
                workspace
            }
            None => {
                self.created.fetch_add(1, Ordering::Relaxed);
                NetworkExecutionWorkspace::default()
            }
        };
        WorkspaceLease {
            pool: Arc::clone(self),
            workspace: Some(workspace),
        }
    }
}

impl WorkspaceLease {
    fn workspace(&mut self) -> &mut NetworkExecutionWorkspace {
        self.workspace
            .as_mut()
            .expect("workspace lease always owns a workspace")
    }
}

impl Drop for WorkspaceLease {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.workspace.take();
            return;
        }
        if let Some(mut workspace) = self.workspace.take() {
            workspace.clear();
            let mut available = self
                .pool
                .available
                .lock()
                .expect("network workspace pool poisoned");
            if available.len() < MAX_IDLE_WORKSPACES_PER_PLAN {
                available.push(workspace);
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct CachedPlan {
    planned: Arc<PlannedNetwork>,
    workspaces: Arc<WorkspacePool>,
}

impl CachedPlan {
    pub(crate) fn execute(&self, tensors: &[&Tensor]) -> Result<Tensor, Error> {
        let mut lease = self.workspaces.lease();
        let previous_capacity = lease.workspace().slot_capacity();
        let result = self
            .planned
            .execute_with_workspace(tensors, lease.workspace());
        if lease.workspace().slot_capacity() > previous_capacity {
            self.workspaces.slot_grows.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
}

struct PlanCache {
    hits: u64,
    misses: u64,
    replans: u64,
    topology_materializations: u64,
    /// O(1) LRU (HashMap + intrusive linked list): touch-on-hit, evict-LRU-on-
    /// insert, all O(1) — the Rust analog of TensorKit's `LRUCache.jl`-backed
    /// `GlobalLRUCache`. Capacity tracks `PlanCacheConfig::capacity`, resized on
    /// insert if the configured capacity changed.
    map: LruCache<Arc<NetworkTopology>, CacheEntry>,
    static_aliases: LruCache<StaticTopologyKey, Vec<StaticAlias>>,
    dynamic_aliases: LruCache<DynamicTopologyKey, Vec<StaticAlias>>,
    /// Persisted contraction orders keyed by stable topology text (see
    /// [`topology_text`]), populated by [`load_plan_cache`] and grown on
    /// every fresh search. A disk hit skips the (cold) optimal-order search
    /// entirely — the plancache analog of `@tensoropt`'s compile-time bake.
    disk: HashMap<String, crate::plan::ContractionPlan>,
    /// Whether cross-process persistence is in use. Set by [`load_plan_cache`]
    /// (the application's opt-in) and only then is [`disk`] consulted/grown.
    /// Off by default so the in-memory replan behavior is byte-identical when
    /// persistence is not used: a persisted order recorded from an early
    /// non-degenerate search must not silently replace a later drift-replan's
    /// fresh search, which the truncation basis (hence energy) depends on.
    persist: bool,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct StaticTopologyKey {
    spec: &'static StaticTopologySpec,
    optimizer: Optimizer,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct DynamicTopologyKey {
    network_id: u64,
    optimizer: Optimizer,
}

struct StaticAlias {
    codomain_ranks: Vec<usize>,
    dims_snapshot: Vec<Vec<usize>>,
    topology: Arc<NetworkTopology>,
    planned: Weak<PlannedNetwork>,
    workspaces: Weak<WorkspacePool>,
}

impl StaticAlias {
    fn cached(&self) -> Option<CachedPlan> {
        Some(CachedPlan {
            planned: self.planned.upgrade()?,
            workspaces: self.workspaces.upgrade()?,
        })
    }
}

/// Clamp a configured capacity to a non-zero LRU capacity (0 would disable
/// caching, which the search-once design never wants — treat it as 1).
fn lru_capacity(capacity: usize) -> NonZeroUsize {
    NonZeroUsize::new(capacity.max(1)).expect("capacity.max(1) is non-zero")
}

impl Default for PlanCache {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            replans: 0,
            topology_materializations: 0,
            map: LruCache::new(lru_capacity(DEFAULT_PLAN_CACHE_CAPACITY)),
            static_aliases: LruCache::new(lru_capacity(DEFAULT_PLAN_CACHE_CAPACITY)),
            dynamic_aliases: LruCache::new(lru_capacity(DEFAULT_PLAN_CACHE_CAPACITY)),
            disk: HashMap::new(),
            persist: false,
        }
    }
}

/// Serialized-plan-cache format version. Bumped whenever the cost model or an
/// optimizer's order search changes so that a stale on-disk file (which would
/// otherwise replay a now-suboptimal order and silently drift truncation) is
/// rejected on load rather than trusted.
const PLAN_CACHE_FILE_VERSION: &str = "TENET_PLANCACHE 1";

/// Stable one-line text key for a network topology: optimizer, output split
/// and labels, then each operand's conj / codomain rank / written split /
/// labels. Labels are `tensor!` identifiers (no separators), so the packed
/// form round-trips by construction and is stable across processes.
fn topology_text(topology: &NetworkTopology) -> String {
    let mut text = format!("{:?}|", topology.optimizer);
    match topology.output_codomain_rank {
        Some(rank) => text.push_str(&rank.to_string()),
        None => text.push('-'),
    }
    text.push('|');
    for (i, label) in topology.output.iter().enumerate() {
        if i > 0 {
            text.push(',');
        }
        text.push_str(label.as_str());
    }
    for operand in &topology.operands {
        text.push('|');
        text.push(if operand.conj { '1' } else { '0' });
        text.push(':');
        text.push_str(&operand.codomain_rank.to_string());
        text.push(':');
        match operand.written_split {
            Some(split) => text.push_str(&split.to_string()),
            None => text.push('-'),
        }
        text.push(':');
        for (i, label) in operand.labels.iter().enumerate() {
            if i > 0 {
                text.push(',');
            }
            text.push_str(label.as_str());
        }
    }
    text
}

// The cache lives in the runtime's `dyn Any + Send` slot; plans are
// step lists + label vectors, so this holds by construction.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<PlanCache>();
};

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
        let (workspaces_created, workspace_reuses, workspace_slot_grows) =
            cache
                .map
                .iter()
                .fold((0, 0, 0), |(created, reused, grows), (_, entry)| {
                    (
                        created + entry.workspaces.created.load(Ordering::Relaxed),
                        reused + entry.workspaces.reused.load(Ordering::Relaxed),
                        grows + entry.workspaces.slot_grows.load(Ordering::Relaxed),
                    )
                });
        let idle_workspaces = cache
            .map
            .iter()
            .map(|(_, entry)| {
                entry
                    .workspaces
                    .available
                    .lock()
                    .expect("network workspace pool poisoned")
                    .len()
            })
            .sum();
        let dynamic_aliases = cache
            .dynamic_aliases
            .iter()
            .map(|(_, aliases)| aliases.len())
            .sum();
        PlanCacheStats {
            hits: cache.hits,
            misses: cache.misses,
            replans: cache.replans,
            entries: cache.map.len(),
            workspaces_created,
            workspace_reuses,
            workspace_slot_grows,
            topology_materializations: cache.topology_materializations,
            idle_workspaces,
            dynamic_aliases,
        }
    })
}

/// Drops every cached plan and resets the counters (not the configuration).
pub fn clear_plan_cache(runtime: &Runtime) {
    runtime.with_extension_slot(|slot| {
        let cache = cache_mut(slot);
        cache.map.clear();
        cache.static_aliases.clear();
        cache.dynamic_aliases.clear();
        cache.hits = 0;
        cache.misses = 0;
        cache.replans = 0;
        cache.topology_materializations = 0;
    });
}

/// Serialize the persisted contraction orders (topology text + plan) to a
/// versioned text blob for the application to write to a cache file. Restore
/// it in a later process with [`load_plan_cache`] before the first contraction
/// to skip the cold optimal-order search. The order is topology-only and thus
/// dimension-independent, so one saved file serves every χ.
pub fn save_plan_cache(runtime: &Runtime) -> String {
    runtime.with_extension_slot(|slot| {
        let cache = cache_mut(slot);
        let mut text = String::from(PLAN_CACHE_FILE_VERSION);
        text.push('\n');
        // Sort at save rather than switching `disk` to a BTreeMap: saving is
        // cold (once per process, at shutdown/checkpoint) while `disk` is
        // read on every cache miss, so paying the sort here keeps the hot
        // lookup path's HashMap unchanged. Without this, iterating the
        // HashMap's std RandomState order made the saved bytes vary run to
        // run for identical content, breaking reproducible builds and
        // content-addressed/git-diffed cache blobs (issue #151).
        let mut entries: Vec<(&String, &crate::plan::ContractionPlan)> =
            cache.disk.iter().collect();
        entries.sort_by_key(|(topo, _)| topo.as_str());
        for (topo, plan) in entries {
            let plan_text = plan.to_text();
            text.push_str("TOPO ");
            text.push_str(topo);
            text.push('\n');
            text.push_str(&format!("PLAN {}\n", plan_text.lines().count()));
            text.push_str(&plan_text);
            if !plan_text.ends_with('\n') {
                text.push('\n');
            }
        }
        text
    })
}

/// Restore orders saved by [`save_plan_cache`]. A blob whose version header
/// does not match this build is ignored (returns 0): a stale file would
/// replay now-suboptimal orders and silently drift truncation, so it is
/// dropped rather than trusted. Returns the number of orders loaded.
pub fn load_plan_cache(runtime: &Runtime, text: &str) -> usize {
    let mut lines = text.lines();
    // An empty blob is a fresh persistence file (first run): activate
    // persistence and load nothing. A non-empty blob with a mismatched version
    // header is stale/foreign and is ignored WITHOUT activating persistence, so
    // it neither replays bad orders nor perturbs in-memory replan numerics.
    let header = lines.next();
    if header.is_some() && header != Some(PLAN_CACHE_FILE_VERSION) {
        return 0;
    }
    runtime.with_extension_slot(|slot| {
        let cache = cache_mut(slot);
        // The application opted into persistence: from now on record and reuse
        // orders through the disk map (even if this file was empty).
        cache.persist = true;
        let mut loaded = 0;
        while let Some(topo_line) = lines.next() {
            let Some(topo) = topo_line.strip_prefix("TOPO ") else {
                continue;
            };
            let Some(count) = lines
                .next()
                .and_then(|l| l.strip_prefix("PLAN "))
                .and_then(|n| n.trim().parse::<usize>().ok())
            else {
                break;
            };
            let plan_text: String =
                (0..count)
                    .filter_map(|_| lines.next())
                    .fold(String::new(), |mut acc, l| {
                        acc.push_str(l);
                        acc.push('\n');
                        acc
                    });
            if let Ok(plan) = crate::plan::ContractionPlan::from_text(&plan_text) {
                cache.disk.insert(topo.to_string(), plan);
                loaded += 1;
            }
        }
        loaded
    })
}

/// A plan made while some leg was trivial (dim ≤ 1) can encode a degenerate,
/// outer-product-heavy order that fits the real state poorly (reusing it is
/// catastrophically slow — that is what [`ReplanPolicy::BakeOnce`] guards
/// against). Once planned at non-degenerate dims the order is frozen.
fn snapshot_is_degenerate(snapshot: &[Vec<usize>]) -> bool {
    snapshot.iter().flatten().any(|&d| d <= 1)
}

/// Whether a topology-matched cache entry must be re-planned given how its
/// leg dims have drifted, per the [`ReplanPolicy`].
fn needs_replan(policy: ReplanPolicy, snapshot: &[Vec<usize>], current: &[Vec<usize>]) -> bool {
    match policy {
        ReplanPolicy::AlwaysReuse => false,
        // Reuse the once-found path for any real dims (cotengra/@tensoropt
        // style); only replace a plan seeded at degenerate dims, and only
        // once the dims have actually moved off that seed.
        ReplanPolicy::BakeOnce => {
            snapshot_is_degenerate(snapshot)
                && snapshot.iter().flatten().ne(current.iter().flatten())
        }
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

fn needs_replan_tensors(
    policy: ReplanPolicy,
    snapshot: &[Vec<usize>],
    tensors: &[&Tensor],
) -> Result<bool, Error> {
    // The per-operand rank guard must run for every policy: a cache hit can
    // arrive via `static_alias_matches`, which compares codomain rank only, so a
    // tensor whose full rank differs from the snapshot reaches here and must
    // force a replan. Hoisting an early-out above this loop would reuse a plan
    // built for a different rank — the reason a naive top-level early-out is
    // unsafe.
    for (operand, dims) in snapshot.iter().enumerate() {
        if tensors[operand].rank() != dims.len() {
            return Ok(true);
        }
    }

    // Past the rank guard the per-axis `leg_dim` scan is dead work for the
    // policies whose result never depends on it: `AlwaysReuse` never replans on
    // drift, and a non-degenerate `BakeOnce` snapshot is frozen for any real
    // dims. Skipping the scan drops `leg_dim(axis)?`, but that call errors only
    // on `axis >= rank` (see `Tensor::leg_dim`), which the guard above already
    // precludes — so no error side effect is lost. `DriftFactor` (and a
    // degenerate `BakeOnce` seed) still need the full comparison.
    match policy {
        ReplanPolicy::AlwaysReuse => return Ok(false),
        ReplanPolicy::BakeOnce if !snapshot_is_degenerate(snapshot) => return Ok(false),
        _ => {}
    }

    let mut changed = false;
    let mut exceeds_factor = false;
    for (operand, dims) in snapshot.iter().enumerate() {
        for (axis, &snap) in dims.iter().enumerate() {
            let current = tensors[operand].leg_dim(axis)?;
            changed |= snap != current;
            if snap != current {
                exceeds_factor |= match policy {
                    ReplanPolicy::DriftFactor(factor) if snap != 0 && current != 0 => {
                        snap.max(current) as f64 / snap.min(current) as f64 > factor
                    }
                    ReplanPolicy::DriftFactor(_) => true,
                    _ => false,
                };
            }
        }
    }
    Ok(match policy {
        ReplanPolicy::AlwaysReuse => false,
        ReplanPolicy::BakeOnce => snapshot_is_degenerate(snapshot) && changed,
        ReplanPolicy::DriftFactor(_) => exceeds_factor,
    })
}

fn static_alias_matches(alias: &StaticAlias, tensors: &[&Tensor]) -> bool {
    alias.codomain_ranks.len() == tensors.len()
        && alias
            .codomain_ranks
            .iter()
            .zip(tensors)
            .all(|(&rank, tensor)| rank == tensor.codomain_rank())
}

fn dynamic_alias_matches(alias: &StaticAlias, network: &Network, tensors: &[&Tensor]) -> bool {
    static_alias_matches(alias, tensors)
        && alias.topology.operands.len() == network.inputs.len()
        && alias
            .topology
            .operands
            .iter()
            .zip(&network.inputs)
            .zip(&network.conj)
            .zip(&network.codomain_splits)
            .all(|(((cached, labels), &conj), &split)| {
                cached.labels == *labels && cached.conj == conj && cached.written_split == split
            })
        && alias.topology.output == network.output
        && alias.topology.output_codomain_rank == network.output_codomain_rank
}

/// Outcome of a warm-path alias lookup done under the single plan-cache lock
/// (shared by the static and dynamic paths; `Disabled` fallbacks differ).
enum Lookup {
    /// Caching is off; execute uncached.
    Disabled,
    Hit(CachedPlan),
    Miss,
}

/// Confirms the alias still points at the resident cache entry AND promotes it
/// to most-recently-used in ONE `LruCache::get` (#155): `get` moves the entry to
/// MRU and hands it back, so the residency check (Arc identity) and the LRU
/// touch share a single `NetworkTopology` hash instead of a `peek` (residency)
/// followed by a `promote`. Counts the hit only on a confirmed match. A miss
/// here (evicted, or the topology now maps to a different plan) is harmless: the
/// caller replans, and promoting whatever currently holds the key — or nothing —
/// changes only LRU order.
fn promote_if_resident(
    cache: &mut PlanCache,
    topology: &Arc<NetworkTopology>,
    cached: CachedPlan,
) -> Option<CachedPlan> {
    let resident = match cache.map.get(topology) {
        Some(entry) => {
            Arc::ptr_eq(&entry.planned, &cached.planned)
                && Arc::ptr_eq(&entry.workspaces, &cached.workspaces)
        }
        None => false,
    };
    if resident {
        cache.hits += 1;
        Some(cached)
    } else {
        None
    }
}

pub(crate) fn execute_static(
    spec: &'static StaticTopologySpec,
    tensors: &[&Tensor],
    optimizer: &Optimizer,
) -> Result<Tensor, Error> {
    let Some(runtime) = tensors.first().map(|tensor| tensor.runtime()) else {
        return spec.network()?.contract_with(tensors, optimizer);
    };
    let key = StaticTopologyKey {
        spec,
        optimizer: topology_optimizer(optimizer),
    };
    // Warm hit path under ONE plan-cache lock (#155): resolve enable/replan
    // policy and do the alias lookup + LRU touch together instead of a separate
    // `plan_cache_config()` acquisition followed by a `with_extension_slot` one.
    // `needs_replan_tensors` stays inside — for the default `BakeOnce` policy it
    // is a rank-only guard (no leg-dim scan, no lock), so it is not the hot
    // cost; the two `NetworkTopology` hashes were, and residency + promote now
    // fold into one `LruCache::get`.
    let lookup = runtime.with_plan_cache(|config, slot| -> Result<Lookup, Error> {
        if !config.enabled {
            return Ok(Lookup::Disabled);
        }
        let cache = cache_mut(slot);
        let Some(aliases) = cache.static_aliases.get(&key) else {
            return Ok(Lookup::Miss);
        };
        let Some(alias) = aliases
            .iter()
            .find(|alias| static_alias_matches(alias, tensors))
        else {
            return Ok(Lookup::Miss);
        };
        if needs_replan_tensors(config.replan, &alias.dims_snapshot, tensors)? {
            return Ok(Lookup::Miss);
        }
        let Some(cached) = alias.cached() else {
            return Ok(Lookup::Miss);
        };
        let topology = alias.topology.clone();
        Ok(match promote_if_resident(cache, &topology, cached) {
            Some(cached) => Lookup::Hit(cached),
            None => Lookup::Miss,
        })
    })?;
    match lookup {
        Lookup::Disabled => return spec.network()?.contract_with(tensors, optimizer),
        Lookup::Hit(cached) => return cached.execute(tensors),
        Lookup::Miss => {}
    }

    let network = spec.network()?;
    let cached = get_or_plan_internal(&network, tensors, optimizer, false)?;
    let topology = topology_for(&network, tensors, optimizer);
    let codomain_ranks = tensors
        .iter()
        .map(|tensor| tensor.codomain_rank())
        .collect();
    let dims_snapshot = tensors
        .iter()
        .map(|tensor| tensor.leg_dims())
        .collect::<Result<_, _>>()?;
    runtime.with_plan_cache(|config, slot| {
        let cache = cache_mut(slot);
        let capacity = lru_capacity(config.capacity);
        if cache.static_aliases.cap() != capacity {
            cache.static_aliases.resize(capacity);
        }
        if let Some(aliases) = cache.static_aliases.get_mut(&key) {
            if let Some(alias) = aliases
                .iter_mut()
                .find(|alias| static_alias_matches(alias, tensors))
            {
                *alias = StaticAlias {
                    codomain_ranks,
                    dims_snapshot,
                    topology,
                    planned: Arc::downgrade(&cached.planned),
                    workspaces: Arc::downgrade(&cached.workspaces),
                };
            } else {
                aliases.push(StaticAlias {
                    codomain_ranks,
                    dims_snapshot,
                    topology,
                    planned: Arc::downgrade(&cached.planned),
                    workspaces: Arc::downgrade(&cached.workspaces),
                });
            }
        } else {
            cache.static_aliases.put(
                key,
                vec![StaticAlias {
                    codomain_ranks,
                    dims_snapshot,
                    topology,
                    planned: Arc::downgrade(&cached.planned),
                    workspaces: Arc::downgrade(&cached.workspaces),
                }],
            );
        }
    });
    cached.execute(tensors)
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
        // Dynamic programming yields the optimal order in polynomial time for
        // TeNeT's small networks — the `@tensoropt` analog without exhaustive
        // search cost. Upstream `dp` errors on all-dim-1 networks (the same
        // degenerate case `auto-hq` trips on), where the order is irrelevant
        // anyway, so fall back to greedy there.
        #[cfg(feature = "opt-path")]
        Optimizer::DynamicProgramming => {
            use crate::pathopt::{OptEinsumPathOptimizer, PathStrategy};
            match network.plan(
                tensors,
                &OptEinsumPathOptimizer::new(PathStrategy::DynamicProgramming),
            ) {
                Ok(plan) => Ok(plan),
                Err(_) => network.plan(tensors, &GreedyDenseOptimizer),
            }
        }
        // Legacy `default_dense_plan` fallback chain: auto-hq -> auto -> dp
        // -> greedy. Upstream `opt-einsum-path` errors on some all-dim-1
        // networks, so each failed driver falls through to the next.
        #[cfg(feature = "opt-path")]
        Optimizer::AutoHq => {
            use crate::pathopt::{OptEinsumPathOptimizer, PathStrategy};
            let mut last_error = None;
            for strategy in [
                PathStrategy::AutoHq,
                PathStrategy::Auto,
                PathStrategy::DynamicProgramming,
            ] {
                match network.plan(tensors, &OptEinsumPathOptimizer::new(strategy)) {
                    Ok(plan) => return Ok(plan),
                    Err(err) => last_error = Some(err),
                }
            }
            let _ = last_error;
            network.plan(tensors, &GreedyDenseOptimizer)
        }
        #[cfg(feature = "cotengra-python")]
        Optimizer::CotengraPython(config) => network.plan(
            tensors,
            &crate::cotengra_python::CotengraPythonOptimizer::new(config.clone()),
        ),
        // `Optimizer` is #[non_exhaustive] and defined in `tenet`; variants
        // this build has no search for (e.g. Optimal without `opt-path`)
        // are an explicit error rather than a silent greedy fallback.
        #[allow(unreachable_patterns)]
        other => Err(Error::InvalidArgument(format!(
            "optimizer {other:?} is not available in this build \
             (is the matching planner feature enabled?)"
        ))),
    }
}

fn topology_optimizer(optimizer: &Optimizer) -> Optimizer {
    #[cfg(feature = "cotengra-python")]
    if let Optimizer::CotengraPython(config) = optimizer {
        let mut config = config.clone();
        // Normal cached contractions are path-only. `optimize_sliced` consumes
        // slicing explicitly and does not go through this cache, so slicing
        // policy must not fragment ordinary plan-cache entries.
        config.slicing = tenet::plancache::CotengraSlicingConfig::None;
        return Optimizer::CotengraPython(config);
    }
    optimizer.clone()
}

fn topology_for(
    network: &Network,
    tensors: &[&Tensor],
    optimizer: &Optimizer,
) -> Arc<NetworkTopology> {
    Arc::new(NetworkTopology {
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
        optimizer: topology_optimizer(optimizer),
    })
}

/// Cache-aware planning for [`Network::contract`]: reuse a topology-matched
/// plan from the operands' runtime (subject to the drift policy), otherwise
/// plan fresh and cache.
pub(crate) fn get_or_plan(
    network: &Network,
    tensors: &[&Tensor],
    optimizer: &Optimizer,
) -> Result<CachedPlan, Error> {
    get_or_plan_internal(network, tensors, optimizer, true)
}

fn get_or_plan_internal(
    network: &Network,
    tensors: &[&Tensor],
    optimizer: &Optimizer,
    register_dynamic_alias: bool,
) -> Result<CachedPlan, Error> {
    // The cache lives on the operands' runtime; step execution would reject
    // mixed-runtime operands anyway, so the first operand's runtime is it.
    let Some(runtime) = tensors.first().map(|tensor| tensor.runtime()) else {
        return Ok(CachedPlan {
            planned: Arc::new(plan_fresh(network, tensors, optimizer)?),
            workspaces: Arc::new(WorkspacePool::default()),
        });
    };
    let dynamic_key = register_dynamic_alias.then(|| DynamicTopologyKey {
        network_id: network.cache_id(),
        optimizer: topology_optimizer(optimizer),
    });
    // Warm alias hit path under ONE plan-cache lock (#155): same shrink as
    // `execute_static` — config + alias lookup + LRU touch in one acquisition,
    // residency + promote folded into one `LruCache::get`.
    let lookup = runtime.with_plan_cache(|config, slot| -> Result<Lookup, Error> {
        if !config.enabled {
            return Ok(Lookup::Disabled);
        }
        let Some(key) = dynamic_key.as_ref() else {
            return Ok(Lookup::Miss);
        };
        let cache = cache_mut(slot);
        let Some(aliases) = cache.dynamic_aliases.get(key) else {
            return Ok(Lookup::Miss);
        };
        let Some(alias) = aliases
            .iter()
            .find(|alias| dynamic_alias_matches(alias, network, tensors))
        else {
            return Ok(Lookup::Miss);
        };
        if needs_replan_tensors(config.replan, &alias.dims_snapshot, tensors)? {
            return Ok(Lookup::Miss);
        }
        let Some(cached) = alias.cached() else {
            return Ok(Lookup::Miss);
        };
        let topology = alias.topology.clone();
        Ok(match promote_if_resident(cache, &topology, cached) {
            Some(cached) => Lookup::Hit(cached),
            None => Lookup::Miss,
        })
    })?;
    match lookup {
        Lookup::Disabled => {
            return Ok(CachedPlan {
                planned: Arc::new(plan_fresh(network, tensors, optimizer)?),
                workspaces: Arc::new(WorkspacePool::default()),
            })
        }
        Lookup::Hit(cached) => return Ok(cached),
        Lookup::Miss => {}
    }

    runtime.with_extension_slot(|slot| cache_mut(slot).topology_materializations += 1);

    let dims: Vec<Vec<usize>> = tensors
        .iter()
        .map(|tensor| tensor.leg_dims())
        .collect::<Result<_, _>>()?;
    let topology = topology_for(network, tensors, optimizer);

    enum Outcome {
        Hit(CachedPlan),
        Replan,
        Miss,
    }
    let outcome = runtime.with_plan_cache(|config, slot| {
        let cache = cache_mut(slot);
        // `peek` inspects without touching LRU order, so a stale entry that will
        // be replanned does not count as a use; a genuine hit is promoted to
        // most-recently-used with an O(1) `promote`.
        match cache.map.peek(&topology) {
            Some(entry) if !needs_replan(config.replan, &entry.dims_snapshot, &dims) => {
                let planned = CachedPlan {
                    planned: Arc::clone(&entry.planned),
                    workspaces: Arc::clone(&entry.workspaces),
                };
                cache.hits += 1;
                cache.map.promote(&topology);
                Outcome::Hit(planned)
            }
            Some(_) => Outcome::Replan,
            None => Outcome::Miss,
        }
    });
    if let Outcome::Hit(planned) = outcome {
        let alias = StaticAlias {
            codomain_ranks: tensors
                .iter()
                .map(|tensor| tensor.codomain_rank())
                .collect(),
            dims_snapshot: dims,
            topology,
            planned: Arc::downgrade(&planned.planned),
            workspaces: Arc::downgrade(&planned.workspaces),
        };
        if let Some(dynamic_key) = dynamic_key {
            runtime.with_plan_cache(|config, slot| {
                let cache = cache_mut(slot);
                let capacity = lru_capacity(config.capacity);
                if cache.dynamic_aliases.cap() != capacity {
                    cache.dynamic_aliases.resize(capacity);
                }
                cache.dynamic_aliases.put(dynamic_key, vec![alias]);
            });
        }
        return Ok(planned);
    }

    // With persistence in use, consult the persisted orders before paying for
    // a fresh search — on a miss AND on a drift-replan (a degenerate seed
    // reused at real dims still pays the full search otherwise). Disk plans are
    // only ever recorded from non-degenerate searches, so a disk hit wraps that
    // good order via `plan_with`, skipping the cold optimal-order search. When
    // persistence is off the disk map is never touched, keeping in-memory
    // replan numerics byte-identical.
    let topo_key = topology_text(&topology);
    let disk_plan = runtime.with_extension_slot(|slot| {
        let cache = cache_mut(slot);
        cache
            .persist
            .then(|| cache.disk.get(&topo_key).cloned())
            .flatten()
    });
    let planned = match disk_plan {
        Some(plan) => Arc::new(network.plan_with(tensors, plan)?),
        None => {
            let fresh = Arc::new(plan_fresh(network, tensors, optimizer)?);
            // Record the freshly searched order so a later process reusing
            // this cache file skips the search — but only under persistence and
            // only when searched at non-degenerate dims. A degenerate seed
            // (dim ≤ 1) yields the outer-product-heavy order `BakeOnce` exists
            // to reject; persisting it would replay that bad order on reuse.
            if !snapshot_is_degenerate(&dims) {
                let plan_copy = fresh.plan().clone();
                runtime.with_extension_slot(|slot| {
                    let cache = cache_mut(slot);
                    if cache.persist {
                        cache.disk.insert(topo_key, plan_copy);
                    }
                });
            }
            fresh
        }
    };
    let workspaces = Arc::new(WorkspacePool::default());
    let alias = StaticAlias {
        codomain_ranks: tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect(),
        dims_snapshot: dims.clone(),
        topology: topology.clone(),
        planned: Arc::downgrade(&planned),
        workspaces: Arc::downgrade(&workspaces),
    };
    runtime.with_plan_cache(|config, slot| {
        let cache = cache_mut(slot);
        match outcome {
            Outcome::Replan => cache.replans += 1,
            _ => cache.misses += 1,
        }
        // Track the configured capacity (it may change between calls), then
        // `put`, which inserts as most-recently-used and evicts the LRU entry
        // in O(1) when at capacity.
        let capacity = lru_capacity(config.capacity);
        if cache.map.cap() != capacity {
            cache.map.resize(capacity);
        }
        if register_dynamic_alias && cache.dynamic_aliases.cap() != capacity {
            cache.dynamic_aliases.resize(capacity);
        }
        cache.map.put(
            topology.clone(),
            CacheEntry {
                planned: Arc::clone(&planned),
                workspaces: Arc::clone(&workspaces),
                dims_snapshot: dims,
            },
        );
        if let Some(dynamic_key) = dynamic_key {
            if let Some(aliases) = cache.dynamic_aliases.get_mut(&dynamic_key) {
                if let Some(existing) = aliases
                    .iter_mut()
                    .find(|existing| existing.codomain_ranks == alias.codomain_ranks)
                {
                    *existing = alias;
                } else {
                    aliases.push(alias);
                }
            } else {
                cache.dynamic_aliases.put(dynamic_key, vec![alias]);
            }
        }
    });
    Ok(CachedPlan {
        planned,
        workspaces,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        configure_plan_cache, get_or_plan, needs_replan_tensors, plan_cache_stats, Optimizer,
        PlanCacheConfig, ReplanPolicy, WorkspacePool, MAX_IDLE_WORKSPACES_PER_PLAN,
    };
    use crate::{Network, TemporaryLabel};
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use tenet::prelude::{Dtype, Runtime, Space, Tensor};

    /// A concurrent burst gets independent leases, while only the explicit
    /// idle cap and its bounded slot capacity survive after return.
    #[test]
    fn nested_workspace_leases_bound_idle_retention() {
        let pool = Arc::new(WorkspacePool::default());
        let mut burst = (0..6).map(|_| pool.lease()).collect::<Vec<_>>();
        for lease in &mut burst {
            lease.workspace().reserve_slots(64);
        }
        assert_eq!(pool.created.load(Ordering::Relaxed), 6);
        drop(burst);

        let available = pool.available.lock().unwrap();
        assert_eq!(available.len(), MAX_IDLE_WORKSPACES_PER_PLAN);
        assert!(available.iter().all(|workspace| workspace.slot_len() == 0));
        let maximum = available
            .iter()
            .map(|workspace| workspace.slot_capacity())
            .max()
            .unwrap();
        let retained: usize = available
            .iter()
            .map(|workspace| workspace.slot_capacity())
            .sum();
        assert!(retained <= MAX_IDLE_WORKSPACES_PER_PLAN * maximum);
        drop(available);

        let first = pool.lease();
        let second = pool.lease();
        let third = pool.lease();
        assert_eq!(pool.reused.load(Ordering::Relaxed), 2);
        assert_eq!(pool.created.load(Ordering::Relaxed), 7);
        drop((first, second, third));
    }

    /// Unwinding quarantines the whole workspace because backend context and
    /// arena contents may have been partially mutated before the panic.
    #[test]
    fn injected_backend_panic_quarantines_workspace() {
        let pool = Arc::new(WorkspacePool::default());
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 2)]);
        let tensor =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 12421).unwrap();
        let retained_before = tensor.storage_strong_count();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let pool = Arc::clone(&pool);
            let tensor = tensor.clone();
            move || {
                let mut lease = pool.lease();
                lease.workspace().retain_tensor(tensor);
                let injected_backend = || panic!("injected backend panic fixture");
                injected_backend();
            }
        }));
        assert!(result.is_err());
        assert_eq!(tensor.storage_strong_count(), retained_before);
        assert!(pool.available.lock().unwrap().is_empty());
        let mut replacement = pool.lease();
        assert_eq!(pool.created.load(Ordering::Relaxed), 2);
        assert_eq!(replacement.workspace().slot_len(), 0);
        replacement.workspace().retain_tensor(tensor.clone());
        drop(replacement);
        assert_eq!(pool.available.lock().unwrap().len(), 1);
    }

    /// An in-flight strong reference cannot make a weak alias valid after its
    /// owning LRU entry has been evicted.
    #[test]
    fn alias_does_not_resurrect_evicted_in_flight_plan() {
        let runtime = Runtime::builder().build().unwrap();
        configure_plan_cache(
            &runtime,
            PlanCacheConfig {
                capacity: 1,
                replan: ReplanPolicy::AlwaysReuse,
                ..PlanCacheConfig::default()
            },
        );
        let space = Space::u1([(0, 2)]);
        let a = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 12411).unwrap();
        let b = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 12412).unwrap();
        let labels = |names: &[&str]| {
            names
                .iter()
                .map(|name| TemporaryLabel::from(*name))
                .collect::<Vec<_>>()
        };
        let first = Network::new(
            vec![labels(&["i", "j"]), labels(&["j", "k"])],
            vec![false; 2],
            vec![None; 2],
            labels(&["i", "k"]),
            None,
        )
        .unwrap();
        let second = Network::new(
            vec![labels(&["a", "b"]), labels(&["b", "c"])],
            vec![false; 2],
            vec![None; 2],
            labels(&["c", "a"]),
            None,
        )
        .unwrap();
        let tensors = [&a, &b];

        let in_flight = get_or_plan(&first, &tensors, &Optimizer::Greedy).unwrap();
        let _replacement = get_or_plan(&second, &tensors, &Optimizer::Greedy).unwrap();
        let _replanned = get_or_plan(&first, &tensors, &Optimizer::Greedy).unwrap();
        drop(in_flight);

        let stats = plan_cache_stats(&runtime);
        assert_eq!((stats.hits, stats.misses, stats.entries), (0, 3, 1));
    }

    /// The rank guard in `needs_replan_tensors` fires for *every* policy,
    /// including the ones whose dim-drift result would otherwise early-out to
    /// `Ok(false)`. This pins the reachability the issue #149 fix depends on: a
    /// static-alias hit compares codomain rank only, so a full-rank mismatch
    /// must still force a replan and the early-out must sit below the guard.
    #[test]
    fn rank_mismatch_forces_replan_for_all_policies() {
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 2)]);
        // rank-2 tensor (one codomain, one domain leg).
        let t = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 1491).unwrap();
        let tensors = [&t];
        // Snapshot claims rank 3, non-degenerate dims — the BakeOnce/AlwaysReuse
        // early-out would say "reuse" if it ran before the rank guard.
        let snapshot = vec![vec![2, 2, 2]];
        for policy in [
            ReplanPolicy::AlwaysReuse,
            ReplanPolicy::BakeOnce,
            ReplanPolicy::DriftFactor(2.0),
        ] {
            assert!(
                needs_replan_tensors(policy, &snapshot, &tensors).unwrap(),
                "rank mismatch must force replan for {policy:?}"
            );
        }
    }

    /// Observable dim-drift behavior is identical before and after the #149
    /// early-out for all three policies.
    #[test]
    fn needs_replan_tensors_matches_policy_semantics() {
        let runtime = Runtime::builder().build().unwrap();
        // rank-2 tensor with both legs dim 2.
        let d2 = Space::u1([(0, 2)]);
        let t = Tensor::rand_with_seed(&runtime, Dtype::F64, [&d2], [&d2], 1492).unwrap();
        let tensors = [&t];

        // AlwaysReuse: never replans on drift (matching rank).
        assert!(
            !needs_replan_tensors(ReplanPolicy::AlwaysReuse, &vec![vec![8, 8]], &tensors).unwrap()
        );

        // BakeOnce, non-degenerate snapshot: frozen even when dims drift (the
        // early-out path).
        assert!(
            !needs_replan_tensors(ReplanPolicy::BakeOnce, &vec![vec![8, 8]], &tensors).unwrap()
        );

        // BakeOnce, degenerate seed: replans once dims move off the seed...
        assert!(needs_replan_tensors(ReplanPolicy::BakeOnce, &vec![vec![1, 2]], &tensors).unwrap());
        // ...but a degenerate seed that still matches current dims stays put.
        let d1 = Space::u1([(0, 1)]);
        let deg = Tensor::rand_with_seed(&runtime, Dtype::F64, [&d1], [&d1], 1493).unwrap();
        assert!(!needs_replan_tensors(ReplanPolicy::BakeOnce, &vec![vec![1, 1]], &[&deg]).unwrap());

        // DriftFactor: replans past the factor, holds within it. Current dims
        // are 2, snapshot 8 → ratio 4.
        assert!(
            needs_replan_tensors(ReplanPolicy::DriftFactor(2.0), &vec![vec![8, 8]], &tensors)
                .unwrap()
        );
        assert!(
            !needs_replan_tensors(ReplanPolicy::DriftFactor(8.0), &vec![vec![8, 8]], &tensors)
                .unwrap()
        );
    }
}
