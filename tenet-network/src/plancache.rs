//! Topology-keyed contraction-plan cache behind [`tenet_macros::tensor`].
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
//! Eviction is LRU; ordinary pairwise contraction routes in `tenet-tensors`
//! are resolved eagerly instead.
//!
//! Storage is per-[`Runtime`]: the configuration value types live in
//! `tenet::plancache` (set them on `Runtime::builder()` or with
//! [`configure_plan_cache`]), and the cache state sits in the runtime's
//! type-erased plan-cache slot, claimed and downcast by this crate. The
//! operands' runtime is resolved per call, so different runtimes never share
//! plans or counters.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use lru::LruCache;
#[cfg(feature = "cuda")]
use tenet::core::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use tenet::core::{TensorStorage, TypedSectorAdmission};
use tenet::prelude::{Error, Runtime, TensorScalar};
#[cfg(feature = "cuda")]
use tenet::typed::CudaStorage;
use tenet::typed::TensorMap;

pub use tenet::plancache::{
    Optimizer, PlanCacheConfig, PlanCacheStats, ReplanPolicy, DEFAULT_PLAN_CACHE_CAPACITY,
    DEFAULT_REPLAN_DRIFT_FACTOR, DEFAULT_WORKSPACE_BUDGET_BYTES,
};

use crate::labels::TemporaryLabel;
use crate::network::{
    HostNetworkError, HostNetworkModeDispatch, Network, NetworkExecutionWorkspace, PlannedNetwork,
    StaticTopologySpec,
};
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
    workspaces: Arc<WorkspacePools>,
    /// Flat leg dims per operand at plan time (written leg order).
    dims_snapshot: Vec<Vec<usize>>,
}

impl Drop for CacheEntry {
    fn drop(&mut self) {
        // Why: an executing CachedPlan can outlive map eviction/clear. Marking
        // its pools inactive makes those late lease returns quarantine their
        // buffers instead of repopulating an orphaned cache entry.
        self.workspaces.deactivate_all();
    }
}

#[derive(Default)]
struct WorkspacePoolCounters {
    created: AtomicU64,
    reused: AtomicU64,
    slot_grows: AtomicU64,
    idle: AtomicU64,
}

#[derive(Default)]
struct WorkspaceBudgetState {
    limit: usize,
    retained: usize,
    peak: usize,
    admissions: u64,
    rejections: u64,
    evictions: u64,
}

struct WorkspaceBudget {
    state: Mutex<WorkspaceBudgetState>,
}

// The byte budget covers dimension-dependent storage retained for reuse. Plan
// metadata, the TypeId registry, pool mutexes, and at most two idle-workspace
// shells per plan remain under the existing plan/idle count bounds; they are
// deliberately not mixed into this numerical-storage budget.

impl WorkspaceBudget {
    fn new(limit: usize) -> Self {
        Self {
            state: Mutex::new(WorkspaceBudgetState {
                limit,
                ..WorkspaceBudgetState::default()
            }),
        }
    }

    fn reserve(&self, bytes: usize) -> bool {
        let mut state = self
            .state
            .lock()
            .expect("network workspace budget poisoned");
        let Some(next) = state.retained.checked_add(bytes) else {
            state.rejections = state.rejections.saturating_add(1);
            return false;
        };
        if state.limit == 0 || next > state.limit {
            state.rejections = state.rejections.saturating_add(1);
            return false;
        }
        state.retained = next;
        state.peak = state.peak.max(next);
        state.admissions = state.admissions.saturating_add(1);
        true
    }

    fn release(&self, bytes: usize, evicted: bool) {
        let mut state = self
            .state
            .lock()
            .expect("network workspace budget poisoned");
        state.retained = state
            .retained
            .checked_sub(bytes)
            .expect("workspace budget release exceeds retained bytes");
        if evicted {
            state.evictions = state.evictions.saturating_add(1);
        }
    }

    fn set_limit(&self, limit: usize) {
        self.state
            .lock()
            .expect("network workspace budget poisoned")
            .limit = limit;
    }

    fn reset_statistics(&self) {
        let mut state = self
            .state
            .lock()
            .expect("network workspace budget poisoned");
        debug_assert_eq!(state.retained, 0);
        state.retained = 0;
        state.peak = 0;
        state.admissions = 0;
        state.rejections = 0;
        state.evictions = 0;
    }

    fn snapshot(&self) -> (usize, usize, u64, u64, u64) {
        let state = self
            .state
            .lock()
            .expect("network workspace budget poisoned");
        (
            state.retained,
            state.peak,
            state.admissions,
            state.rejections,
            state.evictions,
        )
    }
}

struct WorkspacePools {
    pools: Mutex<LruCache<TypeId, Arc<dyn ErasedWorkspacePool>>>,
    counters: Arc<WorkspacePoolCounters>,
    budget: Arc<WorkspaceBudget>,
    accepting: AtomicBool,
}

struct WorkspacePool<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
{
    available: Mutex<Vec<IdleWorkspace<R, D>>>,
    counters: Arc<WorkspacePoolCounters>,
    budget: Arc<WorkspaceBudget>,
    registered: AtomicBool,
}

struct IdleWorkspace<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
{
    workspace: NetworkExecutionWorkspace<R, D>,
    charge: usize,
}

const MAX_IDLE_WORKSPACES_PER_PLAN: usize = 2;
// Empty typed pool shells have no reuse value beyond the plan-wide number of
// buffers that can remain idle, so the same bound caps the TypeId registry.

trait ErasedWorkspacePool: Any + Send + Sync {
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;
    fn deactivate(&self);
}

struct WorkspaceLease<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
{
    pool: Arc<WorkspacePool<R, D>>,
    workspace: Option<NetworkExecutionWorkspace<R, D>>,
    recyclable: bool,
}

impl Default for WorkspacePools {
    fn default() -> Self {
        Self::new(Arc::new(WorkspaceBudget::new(
            DEFAULT_WORKSPACE_BUDGET_BYTES,
        )))
    }
}

impl WorkspacePools {
    fn new(budget: Arc<WorkspaceBudget>) -> Self {
        Self {
            pools: Mutex::new(LruCache::new(lru_capacity(MAX_IDLE_WORKSPACES_PER_PLAN))),
            counters: Arc::new(WorkspacePoolCounters::default()),
            budget,
            accepting: AtomicBool::new(true),
        }
    }

    fn unpooled() -> Self {
        Self::new(Arc::new(WorkspaceBudget::new(0)))
    }

    fn deactivate_all(&self) {
        self.accepting.store(false, Ordering::SeqCst);
        let mut pools = self.pools.lock().expect("network pool registry poisoned");
        while let Some((_, pool)) = pools.pop_lru() {
            pool.deactivate();
        }
    }

    fn rotate_all(&self) {
        let mut pools = self.pools.lock().expect("network pool registry poisoned");
        while let Some((_, pool)) = pools.pop_lru() {
            pool.deactivate();
        }
    }

    fn host_pool<R, D>(&self) -> Arc<WorkspacePool<R, D>>
    where
        R: TypedSectorAdmission + Send + Sync,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar + Send + Sync + 'static,
    {
        if !self.accepting.load(Ordering::SeqCst) {
            return Arc::new(WorkspacePool {
                available: Mutex::new(Vec::new()),
                counters: Arc::clone(&self.counters),
                budget: Arc::clone(&self.budget),
                registered: AtomicBool::new(false),
            });
        }
        let key = TypeId::of::<(R, D)>();
        let mut pools = self.pools.lock().expect("network pool registry poisoned");
        if !self.accepting.load(Ordering::SeqCst) {
            return Arc::new(WorkspacePool {
                available: Mutex::new(Vec::new()),
                counters: Arc::clone(&self.counters),
                budget: Arc::clone(&self.budget),
                registered: AtomicBool::new(false),
            });
        }
        if let Some(pool) = pools.get(&key) {
            return Arc::clone(pool)
                .as_any_arc()
                .downcast::<WorkspacePool<R, D>>()
                .expect("workspace TypeId mapped to the wrong pool type");
        }
        let pool = Arc::new(WorkspacePool {
            available: Mutex::new(Vec::new()),
            counters: Arc::clone(&self.counters),
            budget: Arc::clone(&self.budget),
            registered: AtomicBool::new(true),
        });
        if let Some((_, evicted)) =
            pools.push(key, Arc::clone(&pool) as Arc<dyn ErasedWorkspacePool>)
        {
            evicted.deactivate();
        }
        pool
    }
}

impl<R, D> ErasedWorkspacePool for WorkspacePool<R, D>
where
    R: TypedSectorAdmission + Send + Sync,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar + Send + Sync + 'static,
{
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn deactivate(&self) {
        self.registered.store(false, Ordering::SeqCst);
        let mut available = self
            .available
            .lock()
            .expect("network workspace pool poisoned");
        let removed = available.len() as u64;
        for idle in available.drain(..) {
            self.budget.release(idle.charge, true);
        }
        self.counters.idle.fetch_sub(removed, Ordering::SeqCst);
    }
}

impl<R, D> WorkspacePool<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
{
    fn lease(self: &Arc<Self>) -> WorkspaceLease<R, D> {
        let workspace = {
            let mut available = self
                .available
                .lock()
                .expect("network workspace pool poisoned");
            let idle = available.pop();
            if let Some(idle) = idle {
                self.budget.release(idle.charge, false);
                self.counters.idle.fetch_sub(1, Ordering::SeqCst);
                Some(idle.workspace)
            } else {
                None
            }
        };
        let workspace = match workspace {
            Some(workspace) => {
                self.counters.reused.fetch_add(1, Ordering::Relaxed);
                workspace
            }
            None => {
                self.counters.created.fetch_add(1, Ordering::Relaxed);
                NetworkExecutionWorkspace::default()
            }
        };
        WorkspaceLease {
            pool: Arc::clone(self),
            workspace: Some(workspace),
            recyclable: false,
        }
    }
}

impl<R, D> WorkspaceLease<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
{
    fn workspace(&mut self) -> &mut NetworkExecutionWorkspace<R, D> {
        self.workspace
            .as_mut()
            .expect("workspace lease always owns a workspace")
    }

    fn commit_recycling(&mut self) {
        self.recyclable = true;
    }
}

impl<R, D> Drop for WorkspaceLease<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
{
    fn drop(&mut self) {
        if std::thread::panicking() || !self.recyclable {
            self.workspace.take();
            return;
        }
        if let Some(mut workspace) = self.workspace.take() {
            workspace.clear_slots();
            <R::Mode as HostNetworkModeDispatch<R, D>>::park_workspace(&mut workspace);
            let mut available = self
                .pool
                .available
                .lock()
                .expect("network workspace pool poisoned");
            if !self.pool.registered.load(Ordering::SeqCst) {
                return;
            }
            let reserved = self
                .pool
                .counters
                .idle
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |idle| {
                    (idle < MAX_IDLE_WORKSPACES_PER_PLAN as u64).then_some(idle + 1)
                })
                .is_ok();
            if reserved {
                let charge = workspace.retained_idle_bytes();
                if self.pool.budget.reserve(charge) {
                    available.push(IdleWorkspace { workspace, charge });
                } else {
                    self.pool.counters.idle.fetch_sub(1, Ordering::SeqCst);
                }
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct CachedPlan {
    planned: Arc<PlannedNetwork>,
    workspaces: Arc<WorkspacePools>,
}

impl CachedPlan {
    pub(crate) fn execute_host<R, D>(
        &self,
        tensors: &[&TensorMap<R, D>],
    ) -> Result<TensorMap<R, D>, HostNetworkError<R>>
    where
        R: TypedSectorAdmission + Send + Sync,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar + Send + Sync + 'static,
    {
        let pool = self.workspaces.host_pool::<R, D>();
        let mut lease = pool.lease();
        let previous_capacity = lease.workspace().slot_capacity();
        let result = self
            .planned
            .execute_with_workspace(tensors, lease.workspace());
        if lease.workspace().slot_capacity() > previous_capacity {
            self.workspaces
                .counters
                .slot_grows
                .fetch_add(1, Ordering::Relaxed);
        }
        if result.is_ok() {
            lease.commit_recycling();
        }
        result
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn execute_cuda<R>(
        &self,
        tensors: &[&TensorMap<R, f64, CudaStorage>],
    ) -> Result<TensorMap<R, f64, CudaStorage>, Error>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    {
        self.planned.execute_cuda(tensors)
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
    /// One Runtime-wide owner shared by every resident typed workspace pool.
    workspace_budget: Arc<WorkspaceBudget>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct StaticTopologyKey {
    spec: &'static StaticTopologySpec,
    optimizer: Optimizer,
}

struct StaticAlias {
    codomain_ranks: Vec<usize>,
    dims_snapshot: Vec<Vec<usize>>,
    topology: Arc<NetworkTopology>,
    planned: Weak<PlannedNetwork>,
    workspaces: Weak<WorkspacePools>,
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
        Self::new(DEFAULT_WORKSPACE_BUDGET_BYTES)
    }
}

impl PlanCache {
    fn new(workspace_budget_bytes: usize) -> Self {
        Self {
            hits: 0,
            misses: 0,
            replans: 0,
            topology_materializations: 0,
            map: LruCache::new(lru_capacity(DEFAULT_PLAN_CACHE_CAPACITY)),
            static_aliases: LruCache::new(lru_capacity(DEFAULT_PLAN_CACHE_CAPACITY)),
            disk: HashMap::new(),
            persist: false,
            workspace_budget: Arc::new(WorkspaceBudget::new(workspace_budget_bytes)),
        }
    }
}

/// Serialized-plan-cache format version. Bumped whenever the cost model or an
/// optimizer's order search changes so that a stale on-disk file (which would
/// otherwise replay a now-suboptimal order and silently drift truncation) is
/// rejected on load rather than trusted.
const PLAN_CACHE_FILE_VERSION: &str = "TENET_PLANCACHE 2";

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
fn cache_mut(
    slot: &mut Option<Box<dyn Any + Send>>,
    workspace_budget_bytes: usize,
) -> &mut PlanCache {
    slot.get_or_insert_with(|| Box::new(PlanCache::new(workspace_budget_bytes)))
        .downcast_mut::<PlanCache>()
        .expect("runtime plan-cache slot claimed by another type")
}

/// Read/write access that preserves an unclaimed runtime extension slot.
/// Rejected device plans must not make an otherwise-cold runtime own an empty
/// cache merely by probing it.
fn existing_cache_mut(slot: &mut Option<Box<dyn Any + Send>>) -> Option<&mut PlanCache> {
    slot.as_deref_mut().map(|cache| {
        cache
            .downcast_mut::<PlanCache>()
            .expect("runtime plan-cache slot claimed by another type")
    })
}

/// Replaces the runtime's plan-cache configuration (the builder-time
/// equivalent is `Runtime::builder().plan_cache(config)`).
pub fn configure_plan_cache(runtime: &Runtime, config: PlanCacheConfig) {
    runtime.replace_plan_cache_config(config, |previous, next, slot| {
        let Some(cache) = existing_cache_mut(slot) else {
            return;
        };

        if previous.enabled && !next.enabled {
            cache.map.clear();
            cache.static_aliases.clear();
        } else {
            if next.workspace_budget_bytes < previous.workspace_budget_bytes {
                cache
                    .workspace_budget
                    .set_limit(next.workspace_budget_bytes);
                for (_, entry) in cache.map.iter() {
                    entry.workspaces.rotate_all();
                }
            } else if next.workspace_budget_bytes > previous.workspace_budget_bytes {
                cache
                    .workspace_budget
                    .set_limit(next.workspace_budget_bytes);
            }

            if next.capacity < previous.capacity {
                let capacity = lru_capacity(next.capacity);
                cache.map.resize(capacity);
                cache.static_aliases.resize(capacity);
            }
        }
        cache
            .workspace_budget
            .set_limit(next.workspace_budget_bytes);
    });
}

/// The runtime's current plan-cache configuration.
pub fn plan_cache_config(runtime: &Runtime) -> PlanCacheConfig {
    runtime.plan_cache_config()
}

/// Hit/miss/re-plan counters and the current entry count.
#[allow(deprecated)]
pub fn plan_cache_stats(runtime: &Runtime) -> PlanCacheStats {
    runtime.with_plan_cache(|config, slot| {
        let cache = cache_mut(slot, config.workspace_budget_bytes);
        let (workspaces_created, workspace_reuses, workspace_slot_grows) =
            cache
                .map
                .iter()
                .fold((0, 0, 0), |(created, reused, grows), (_, entry)| {
                    (
                        created + entry.workspaces.counters.created.load(Ordering::Relaxed),
                        reused + entry.workspaces.counters.reused.load(Ordering::Relaxed),
                        grows + entry.workspaces.counters.slot_grows.load(Ordering::Relaxed),
                    )
                });
        let idle_workspaces = cache
            .map
            .iter()
            .map(|(_, entry)| entry.workspaces.counters.idle.load(Ordering::Relaxed) as usize)
            .sum();
        let (
            retained_workspace_bytes,
            peak_retained_workspace_bytes,
            workspace_byte_admissions,
            workspace_byte_rejections,
            workspace_byte_evictions,
        ) = cache.workspace_budget.snapshot();
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
            retained_workspace_bytes,
            peak_retained_workspace_bytes,
            workspace_byte_admissions,
            workspace_byte_rejections,
            workspace_byte_evictions,
            dynamic_aliases: 0,
        }
    })
}

/// Drops every cached plan and resets the counters (not the configuration).
pub fn clear_plan_cache(runtime: &Runtime) {
    runtime.with_plan_cache(|config, slot| {
        let cache = cache_mut(slot, config.workspace_budget_bytes);
        cache.map.clear();
        cache.static_aliases.clear();
        cache.hits = 0;
        cache.misses = 0;
        cache.replans = 0;
        cache.topology_materializations = 0;
        cache.workspace_budget.reset_statistics();
    });
}

/// Serialize the persisted contraction orders (topology text + plan) to a
/// versioned text blob for the application to write to a cache file. Restore
/// it in a later process with [`load_plan_cache`] before the first contraction
/// to skip the cold optimal-order search. The order is topology-only and thus
/// dimension-independent, so one saved file serves every χ.
pub fn save_plan_cache(runtime: &Runtime) -> String {
    runtime.with_plan_cache(|config, slot| {
        let cache = cache_mut(slot, config.workspace_budget_bytes);
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
    runtime.with_plan_cache(|config, slot| {
        let cache = cache_mut(slot, config.workspace_budget_bytes);
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

fn needs_replan_tensors<R, D, S>(
    policy: ReplanPolicy,
    snapshot: &[Vec<usize>],
    tensors: &[&TensorMap<R, D, S>],
) -> Result<bool, HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
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
    // on `axis >= rank` (see `TensorMap::leg_dim`), which the guard above already
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
        let current_dims = <R::Mode as HostNetworkModeDispatch<R, D>>::leg_dims(tensors[operand])?;
        for (&snap, current) in dims.iter().zip(current_dims) {
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

fn static_alias_matches(alias: &StaticAlias, codomain_ranks: &[usize]) -> bool {
    alias.codomain_ranks == codomain_ranks
}

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

fn plan_fresh<R, D, S>(
    network: &Network,
    tensors: &[&TensorMap<R, D, S>],
    optimizer: &Optimizer,
) -> Result<PlannedNetwork, HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
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
        ))
        .into()),
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

fn topology_for<R, D, S>(
    network: &Network,
    tensors: &[&TensorMap<R, D, S>],
    optimizer: &Optimizer,
) -> Arc<NetworkTopology>
where
    R: TypedSectorAdmission,
    S: TensorStorage<D>,
{
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

fn install_static_alias(
    cache: &mut PlanCache,
    key: StaticTopologyKey,
    codomain_ranks: Vec<usize>,
    dims_snapshot: Vec<Vec<usize>>,
    topology: Arc<NetworkTopology>,
    cached: &CachedPlan,
    capacity: NonZeroUsize,
) {
    if cache.static_aliases.cap() != capacity {
        cache.static_aliases.resize(capacity);
    }
    let alias = StaticAlias {
        codomain_ranks: codomain_ranks.clone(),
        dims_snapshot,
        topology,
        planned: Arc::downgrade(&cached.planned),
        workspaces: Arc::downgrade(&cached.workspaces),
    };
    if let Some(aliases) = cache.static_aliases.get_mut(&key) {
        if let Some(existing) = aliases
            .iter_mut()
            .find(|existing| static_alias_matches(existing, &codomain_ranks))
        {
            *existing = alias;
        } else {
            aliases.push(alias);
        }
    } else {
        cache.static_aliases.put(key, vec![alias]);
    }
}

/// Static macro planning over reduced typed operands. `codomain_ranks` are
/// from the original expressions and guard trace/conj lowering; dimensions
/// and the structural plan come from `tensors` after call-local trace lowering.
pub(crate) fn get_or_plan_static<R, D, S>(
    spec: &'static StaticTopologySpec,
    tensors: &[&TensorMap<R, D, S>],
    codomain_ranks: &[usize],
    optimizer: &Optimizer,
    validate_plan: impl Fn(&PlannedNetwork) -> Result<(), Error>,
    make_network: impl FnOnce() -> Result<Network, Error>,
) -> Result<CachedPlan, HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    let Some(runtime) = tensors.first().map(|tensor| tensor.runtime()) else {
        return Err(Error::InvalidArgument(
            "network execution requires at least one operand".to_string(),
        )
        .into());
    };
    let key = StaticTopologyKey {
        spec,
        optimizer: topology_optimizer(optimizer),
    };
    let lookup =
        runtime.with_plan_cache(|config, slot| -> Result<Lookup, HostNetworkError<R>> {
            if !config.enabled {
                return Ok(Lookup::Disabled);
            }
            let Some(cache) = existing_cache_mut(slot) else {
                return Ok(Lookup::Miss);
            };
            let Some(aliases) = cache.static_aliases.peek(&key) else {
                return Ok(Lookup::Miss);
            };
            let Some(alias) = aliases
                .iter()
                .find(|alias| static_alias_matches(alias, codomain_ranks))
            else {
                return Ok(Lookup::Miss);
            };
            if needs_replan_tensors(config.replan, &alias.dims_snapshot, tensors)? {
                return Ok(Lookup::Miss);
            }
            let Some(cached) = alias.cached() else {
                return Ok(Lookup::Miss);
            };
            validate_plan(&cached.planned)?;
            let topology = alias.topology.clone();
            cache.static_aliases.promote(&key);
            Ok(match promote_if_resident(cache, &topology, cached) {
                Some(cached) => Lookup::Hit(cached),
                None => Lookup::Miss,
            })
        })?;
    if let Lookup::Hit(cached) = lookup {
        return Ok(cached);
    }

    let network = make_network().map_err(HostNetworkError::<R>::from)?;
    if matches!(lookup, Lookup::Disabled) {
        let planned = Arc::new(plan_fresh(&network, tensors, optimizer)?);
        validate_plan(&planned).map_err(HostNetworkError::<R>::from)?;
        return Ok(CachedPlan {
            planned,
            workspaces: Arc::new(WorkspacePools::unpooled()),
        });
    }

    let dims: Vec<Vec<usize>> = tensors
        .iter()
        .map(|tensor| <R::Mode as HostNetworkModeDispatch<R, D>>::leg_dims(tensor))
        .collect::<Result<_, _>>()?;
    let topology = topology_for(&network, tensors, optimizer);

    #[derive(Clone)]
    enum Outcome {
        Hit(CachedPlan),
        Replan,
        Miss,
    }
    let outcome =
        runtime.with_plan_cache(|config, slot| -> Result<Outcome, HostNetworkError<R>> {
            let Some(cache) = existing_cache_mut(slot) else {
                return Ok(Outcome::Miss);
            };
            // `peek` inspects without touching LRU order, so a stale entry that will
            // be replanned does not count as a use; a genuine hit is promoted to
            // most-recently-used with an O(1) `promote`.
            match cache.map.peek(&topology) {
                Some(entry) if !needs_replan(config.replan, &entry.dims_snapshot, &dims) => {
                    validate_plan(&entry.planned).map_err(HostNetworkError::<R>::from)?;
                    let planned = CachedPlan {
                        planned: Arc::clone(&entry.planned),
                        workspaces: Arc::clone(&entry.workspaces),
                    };
                    cache.hits += 1;
                    cache.map.promote(&topology);
                    Ok(Outcome::Hit(planned))
                }
                Some(_) => Ok(Outcome::Replan),
                None => Ok(Outcome::Miss),
            }
        })?;
    if let Outcome::Hit(planned) = outcome.clone() {
        runtime.with_plan_cache(|config, slot| {
            install_static_alias(
                cache_mut(slot, config.workspace_budget_bytes),
                key,
                codomain_ranks.to_vec(),
                dims,
                topology,
                &planned,
                lru_capacity(config.capacity),
            )
        });
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
        existing_cache_mut(slot).and_then(|cache| {
            cache
                .persist
                .then(|| cache.disk.get(&topo_key).cloned())
                .flatten()
        })
    });
    let (planned, fresh_plan_copy) = match disk_plan {
        Some(plan) => (Arc::new(network.plan_with(tensors, plan)?), None),
        None => {
            let fresh = Arc::new(plan_fresh(&network, tensors, optimizer)?);
            // Record the freshly searched order so a later process reusing
            // this cache file skips the search — but only under persistence and
            // only when searched at non-degenerate dims. A degenerate seed
            // (dim ≤ 1) yields the outer-product-heavy order `BakeOnce` exists
            // to reject; persisting it would replay that bad order on reuse.
            let plan_copy = (!snapshot_is_degenerate(&dims)).then(|| fresh.plan().clone());
            (fresh, plan_copy)
        }
    };
    validate_plan(&planned).map_err(HostNetworkError::<R>::from)?;
    let workspace_budget = runtime.with_plan_cache(|config, slot| {
        Arc::clone(&cache_mut(slot, config.workspace_budget_bytes).workspace_budget)
    });
    let candidate = CachedPlan {
        planned,
        workspaces: Arc::new(WorkspacePools::new(workspace_budget)),
    };
    let winner =
        runtime.with_plan_cache(|config, slot| -> Result<CachedPlan, HostNetworkError<R>> {
            let cache = cache_mut(slot, config.workspace_budget_bytes);
            if !config.enabled {
                candidate.workspaces.deactivate_all();
                return Ok(candidate.clone());
            }
            let capacity = lru_capacity(config.capacity);
            if cache.map.cap() != capacity {
                cache.map.resize(capacity);
            }
            let cached = match cache.map.peek(&topology) {
                Some(entry) if !needs_replan(config.replan, &entry.dims_snapshot, &dims) => {
                    validate_plan(&entry.planned).map_err(HostNetworkError::<R>::from)?;
                    cache.hits += 1;
                    CachedPlan {
                        planned: Arc::clone(&entry.planned),
                        workspaces: Arc::clone(&entry.workspaces),
                    }
                }
                _ => {
                    cache.topology_materializations += 1;
                    if cache.persist {
                        if let Some(plan_copy) = &fresh_plan_copy {
                            cache.disk.insert(topo_key.clone(), plan_copy.clone());
                        }
                    }
                    match outcome {
                        Outcome::Replan => cache.replans += 1,
                        _ => cache.misses += 1,
                    }
                    let cached = candidate.clone();
                    cache.map.put(
                        topology.clone(),
                        CacheEntry {
                            planned: Arc::clone(&cached.planned),
                            workspaces: Arc::clone(&cached.workspaces),
                            dims_snapshot: dims.clone(),
                        },
                    );
                    cached
                }
            };
            cache.map.promote(&topology);
            install_static_alias(
                cache,
                key,
                codomain_ranks.to_vec(),
                dims.clone(),
                topology.clone(),
                &cached,
                capacity,
            );
            Ok(cached)
        })?;
    Ok(winner)
}

#[cfg(test)]
mod tests {
    use super::{needs_replan, ReplanPolicy, WorkspaceBudget, WorkspacePools};
    use crate::network::NetworkExecutionWorkspace;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Barrier};
    use tenet::core::{SU2FusionRule, U1FusionRule};
    use tenet::prelude::Complex64;

    #[test]
    fn replan_policy_matches_dimension_drift_semantics() {
        let snapshot = vec![vec![2, 3]];
        assert!(!needs_replan(
            ReplanPolicy::AlwaysReuse,
            &snapshot,
            &[vec![2]],
        ));
        assert!(!needs_replan(
            ReplanPolicy::AlwaysReuse,
            &snapshot,
            &[vec![9, 11]],
        ));
        assert!(needs_replan(
            ReplanPolicy::DriftFactor(2.0),
            &snapshot,
            &[vec![5, 3]],
        ));
    }

    #[test]
    fn panic_quarantines_typed_workspace_lease() {
        let pools = WorkspacePools::default();
        let pool = pools.host_pool::<U1FusionRule, f64>();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let pool = pool.clone();
            move || {
                let _lease = pool.lease();
                panic!("injected workspace panic");
            }
        }));
        assert!(result.is_err());
        assert!(pool.available.lock().unwrap().is_empty());
        let mut lease = pool.lease();
        lease.commit_recycling();
        drop(lease);
        assert_eq!(
            pools
                .counters
                .created
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }

    #[test]
    fn active_lease_return_after_typed_pool_eviction_is_not_retained() {
        let pools = WorkspacePools::default();
        let first = pools.host_pool::<U1FusionRule, f64>();
        let mut lease = first.lease();
        lease.commit_recycling();
        drop(pools.host_pool::<U1FusionRule, Complex64>());
        drop(pools.host_pool::<SU2FusionRule, f64>());
        assert!(!first.registered.load(Ordering::SeqCst));
        drop(lease);
        assert!(first.available.lock().unwrap().is_empty());
        assert_eq!(pools.counters.idle.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn active_lease_return_after_plan_clear_is_not_retained() {
        let pools = WorkspacePools::default();
        let pool = pools.host_pool::<U1FusionRule, f64>();
        let mut lease = pool.lease();
        lease.commit_recycling();
        pools.deactivate_all();
        drop(lease);
        assert!(pool.available.lock().unwrap().is_empty());
        assert_eq!(pools.counters.idle.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn deactivated_registry_rejects_late_typed_pool_creation() {
        let pools = WorkspacePools::default();
        pools.deactivate_all();
        let pool = pools.host_pool::<U1FusionRule, f64>();
        assert!(!pool.registered.load(Ordering::SeqCst));
        let mut lease = pool.lease();
        lease.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(16));
        lease.commit_recycling();
        drop(lease);
        assert!(pool.available.lock().unwrap().is_empty());
        assert_eq!(pools.budget.snapshot(), (0, 0, 0, 0, 0));
    }

    #[test]
    fn budget_rotation_rejects_old_return_and_accepts_new_pool() {
        let budget = Arc::new(WorkspaceBudget::new(usize::MAX));
        let pools = WorkspacePools::new(Arc::clone(&budget));
        let old = pools.host_pool::<U1FusionRule, f64>();
        let mut stale = old.lease();
        stale.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(16));
        stale.commit_recycling();
        pools.rotate_all();
        drop(stale);
        assert!(old.available.lock().unwrap().is_empty());

        let current = pools.host_pool::<U1FusionRule, f64>();
        assert!(current.registered.load(Ordering::SeqCst));
        let mut lease = current.lease();
        lease.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(16));
        lease.commit_recycling();
        drop(lease);
        assert_eq!(current.available.lock().unwrap().len(), 1);
        let (retained, _, admissions, rejections, _) = budget.snapshot();
        assert!(retained > 0);
        assert_eq!((admissions, rejections), (1, 0));
    }

    #[test]
    fn concurrent_whole_returns_race_for_one_final_budget_slot() {
        let probe = NetworkExecutionWorkspace::<U1FusionRule, f64>::with_test_slot_capacity(32);
        let charge = probe.retained_idle_bytes();
        assert!(charge > 0);
        let budget = Arc::new(WorkspaceBudget::new(charge));
        let pools = WorkspacePools::new(Arc::clone(&budget));
        let pool = pools.host_pool::<U1FusionRule, f64>();
        let mut first = pool.lease();
        let mut second = pool.lease();
        first.workspace = Some(probe);
        second.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(32));
        first.commit_recycling();
        second.commit_recycling();
        let barrier = Arc::new(Barrier::new(2));
        std::thread::scope(|scope| {
            let first_barrier = Arc::clone(&barrier);
            let second_barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                first_barrier.wait();
                drop(first);
            });
            scope.spawn(move || {
                second_barrier.wait();
                drop(second);
            });
        });
        assert_eq!(pool.available.lock().unwrap().len(), 1);
        let (retained, peak, admissions, rejections, _) = budget.snapshot();
        assert_eq!((retained, peak), (charge, charge));
        assert_eq!((admissions, rejections), (1, 1));
    }

    #[test]
    fn typed_pools_share_one_plan_wide_byte_owner() {
        let f64_probe = NetworkExecutionWorkspace::<U1FusionRule, f64>::with_test_slot_capacity(8);
        let complex_probe =
            NetworkExecutionWorkspace::<U1FusionRule, Complex64>::with_test_slot_capacity(8);
        let f64_charge = f64_probe.retained_idle_bytes();
        let complex_charge = complex_probe.retained_idle_bytes();
        let budget = Arc::new(WorkspaceBudget::new(f64_charge + complex_charge - 1));
        let pools = WorkspacePools::new(Arc::clone(&budget));

        let f64_pool = pools.host_pool::<U1FusionRule, f64>();
        let complex_pool = pools.host_pool::<U1FusionRule, Complex64>();
        let mut f64_lease = f64_pool.lease();
        let mut complex_lease = complex_pool.lease();
        f64_lease.workspace = Some(f64_probe);
        complex_lease.workspace = Some(complex_probe);
        f64_lease.commit_recycling();
        complex_lease.commit_recycling();
        drop(f64_lease);
        drop(complex_lease);

        assert_eq!(f64_pool.available.lock().unwrap().len(), 1);
        assert!(complex_pool.available.lock().unwrap().is_empty());
        assert_eq!(budget.snapshot().0, f64_charge);
        assert_eq!(budget.snapshot().2, 1);
        assert_eq!(budget.snapshot().3, 1);
    }

    #[test]
    fn lifo_small_return_keeps_one_large_bottom_workspace_charged() {
        let budget = Arc::new(WorkspaceBudget::new(usize::MAX));
        let pools = WorkspacePools::new(Arc::clone(&budget));
        let pool = pools.host_pool::<U1FusionRule, f64>();
        let mut first = pool.lease();
        let mut second = pool.lease();
        first.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(64));
        second.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(64));
        let large = first.workspace.as_ref().unwrap().retained_idle_bytes();
        first.commit_recycling();
        second.commit_recycling();
        drop(first);
        drop(second);

        let mut top = pool.lease();
        top.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(1));
        let small = top.workspace.as_ref().unwrap().retained_idle_bytes();
        top.commit_recycling();
        drop(top);
        assert_eq!(pool.available.lock().unwrap().len(), 2);
        assert_eq!(budget.snapshot().0, large + small);
    }
}
