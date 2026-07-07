//! Configuration of the topology-keyed contraction-plan cache owned by
//! [`Runtime`](crate::prelude::Runtime).
//!
//! The cache itself (keys and plan entries) lives in `tenet-network`, which
//! depends on this crate; the runtime stores it behind a type-erased slot
//! (see `Runtime::with_extension_slot`) and owns only the configuration
//! value types defined here. Set the configuration on
//! [`RuntimeBuilder`](crate::prelude::RuntimeBuilder) via
//! `plan_cache`/`optimizer`, or later through `tenet-network`'s
//! `configure_plan_cache`.
//!
//! Naming and placement of this module are subject to a later API pass.

/// Which contraction-order search to run: a hashable value type (usable as
/// a cache-key component and as a runtime-wide default) rather than a trait
/// object. `#[non_exhaustive]` so future external searches (e.g. a
/// cotengrust adapter variant carrying its config) slot in without a
/// breaking change.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Optimizer {
    /// Greedy pairwise search (`tenet-network`'s `GreedyDenseOptimizer`);
    /// the default.
    #[default]
    Greedy,
    /// Exhaustive optimal search (opt_einsum `"optimal"`; small networks
    /// only). Requires `tenet-network`'s `opt-path` feature at execution.
    #[cfg(feature = "opt-path")]
    Optimal,
    /// Dynamic-programming search (opt_einsum `"dp"`): the SAME optimal
    /// pairwise order as `Optimal` for the small networks TeNeT contracts,
    /// but polynomial-time instead of exhaustive `O(n!)`. This is the
    /// `@tensoropt` analog — optimal order without the branch-and-bound
    /// search cost that dominates the first (cold) contraction of each
    /// topology. Requires `tenet-network`'s `opt-path` feature at execution.
    #[cfg(feature = "opt-path")]
    DynamicProgramming,
    /// The legacy `EinsumPlan::compile` default: opt_einsum `"auto-hq"`
    /// with fallback to `"auto"`, then `"dp"`, then greedy when a driver
    /// errors (upstream `opt-einsum-path` rejects some all-dim-1 networks).
    /// Near-optimal orders for the large gram / environment-body networks
    /// where plain greedy picks memory-exploding orders. Requires
    /// `tenet-network`'s `opt-path` feature at execution.
    #[cfg(feature = "opt-path")]
    AutoHq,
}

/// When to re-plan a topology-matched cache entry whose leg dimensions have
/// drifted from the snapshot it was planned under. Reusing is always
/// *correct* (a pairwise order is dimension-independent); re-planning only
/// restores *optimality*.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReplanPolicy {
    /// Always reuse the cached order, whatever the current dimensions.
    AlwaysReuse,
    /// Find the order once at real (non-degenerate) dims, then freeze and
    /// reuse it for any later dimensions — the standard "search once, reuse
    /// the path regardless of rank" design (cotengra's reusable
    /// `ContractionTree`, `@tensoropt`'s compile-time bake). Unlike
    /// [`AlwaysReuse`](Self::AlwaysReuse) it *does* replace a plan that was
    /// seeded at degenerate dims (some leg trivial, dim ≤ 1), whose order can
    /// be a poor outer-product-heavy fit for the real state. This is the
    /// default: re-searching a drifted topology (see
    /// [`DriftFactor`](Self::DriftFactor)) buys no measured speedup on
    /// TeNeT's networks while paying the (χ-dependent) search cost each time.
    BakeOnce,
    /// Re-plan when any leg dimension differs from the snapshot by more
    /// than this factor (as a ratio, in either direction). Chases the
    /// per-shape-optimal order; only worth it when a network's winning order
    /// genuinely flips between dimension regimes.
    DriftFactor(f64),
}

/// Drift ratio for [`ReplanPolicy::DriftFactor`] when that (non-default)
/// policy is selected: re-plan once a leg has grown or shrunk past 2x its
/// planning-time value, on the theory that the cost balance has changed
/// qualitatively. Not the default — see [`ReplanPolicy::BakeOnce`].
pub const DEFAULT_REPLAN_DRIFT_FACTOR: f64 = 2.0;

impl Default for ReplanPolicy {
    fn default() -> Self {
        Self::BakeOnce
    }
}

/// Default maximum number of cached plans (per runtime).
///
/// Rationale: an entry is plan metadata only (label lists, a step list and
/// a dims snapshot — well under a kilobyte for realistic networks), so 256
/// bounds the cache to a few hundred kilobytes while covering drivers that
/// cycle through many distinct expressions (e.g. every bond of a large
/// unit cell) without eviction thrash.
pub const DEFAULT_PLAN_CACHE_CAPACITY: usize = 256;

/// Plan-cache behavior; set on
/// [`RuntimeBuilder`](crate::prelude::RuntimeBuilder) or with
/// `tenet-network`'s `configure_plan_cache`.
#[derive(Clone, Debug)]
pub struct PlanCacheConfig {
    /// Master switch; `false` makes every network contraction plan fresh.
    pub enabled: bool,
    /// Maximum cached entries before LRU eviction.
    pub capacity: usize,
    /// When to re-plan on dimension drift.
    pub replan: ReplanPolicy,
    /// Default optimizer for network contraction (the `tensor!` path).
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

/// Counters for tests and diagnostics; see `tenet-network`'s
/// `plan_cache_stats`.
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
