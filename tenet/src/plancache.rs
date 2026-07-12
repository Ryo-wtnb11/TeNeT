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
/// object. `#[non_exhaustive]` so future external searches (e.g. another
/// external-planner adapter carrying its config) slot in without a
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
    /// External Python `cotengra` path search. Requires `tenet-network`'s
    /// `cotengra-python` feature at execution and an importable Python
    /// `cotengra` installation. This is intentionally a cold-path planner:
    /// the returned pairwise order is cached and warm execution stays in Rust.
    #[cfg(feature = "cotengra-python")]
    CotengraPython(CotengraPythonConfig),
}

/// Search family used by the Python `cotengra` backend.
#[cfg(feature = "cotengra-python")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CotengraPythonMethod {
    /// `cotengra` preset string `"auto"`.
    Auto,
    /// `cotengra` preset string `"auto-hq"`.
    #[default]
    AutoHq,
    /// `cotengra.GreedyOptimizer`.
    Greedy,
    /// `cotengra.OptimalOptimizer`.
    Optimal,
    /// `cotengra.RandomGreedyOptimizer`.
    RandomGreedy,
    /// `cotengra.HyperOptimizer`.
    Hyper,
}

/// Objective string passed to `cotengra` optimizers that accept `minimize`.
#[cfg(feature = "cotengra-python")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CotengraMinimize {
    /// Minimize estimated contraction FLOPs.
    #[default]
    Flops,
    /// Minimize largest intermediate tensor.
    Size,
    /// Minimize total tensor write volume.
    Write,
    /// Minimize cotengra's combined FLOPs/write objective.
    Combo,
    /// Minimize cotengra's memory-limit-oriented combined objective.
    Limit,
    /// Pass a custom objective name through unchanged.
    Custom(String),
}

/// Configuration for the optional Python `cotengra` planner.
#[cfg(feature = "cotengra-python")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CotengraPythonConfig {
    /// Which cotengra optimizer family to instantiate.
    pub method: CotengraPythonMethod,
    /// Objective for optimizers that expose `minimize`.
    pub minimize: CotengraMinimize,
    /// Trial count for `RandomGreedyOptimizer` and `HyperOptimizer`.
    pub max_repeats: usize,
    /// Optional RNG seed for randomized optimizers.
    pub seed: Option<u64>,
    /// Whether cotengra may parallelize the planner. Defaults to `false` so
    /// path search is reproducible and does not compete with TeNeT execution.
    pub parallel: bool,
    /// Optional slicing policy. `None` keeps this as a path-only planner.
    pub slicing: CotengraSlicingConfig,
    /// Python executable. `None` means `$TENET_COTENGRA_PYTHON`,
    /// `$TENET_COTENGRA_UV_PROJECT`, or `python3`.
    pub python: Option<String>,
    /// Arguments inserted between the Python executable and `-c <planner>`.
    /// This supports launchers such as `uv run --project <dir> python`.
    pub python_args: Vec<String>,
}

/// Optional post-processing to ask cotengra to slice a searched tree.
///
/// This only affects explicit sliced-plan searches. Normal
/// `Optimizer::CotengraPython` contraction planning remains path-only and
/// ignores slicing, because TeNeT's ordinary executor cannot consume slices.
#[cfg(feature = "cotengra-python")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CotengraSlicingConfig {
    /// Do not slice.
    #[default]
    None,
    /// Call `ContractionTree.slice`.
    Slice {
        /// Target maximum intermediate size in scalar elements.
        target_size: usize,
        /// Number of repeated slicing searches.
        max_repeats: usize,
        /// Whether cotengra may slice output/open indices.
        allow_outer: bool,
    },
    /// Call `ContractionTree.slice_and_reconfigure`.
    Reconfigure {
        /// Target maximum intermediate size in scalar elements.
        target_size: usize,
        /// Minimum slicing progress before each reconfiguration round.
        step_size: usize,
        /// Number of repeated slicing searches per round.
        max_repeats: usize,
        /// Whether cotengra may slice output/open indices.
        allow_outer: bool,
        /// Use cotengra's forested subtree reconfiguration option.
        forested: bool,
    },
    /// Call `ContractionTree.slice_and_reconfigure_forest`.
    ForestReconfigure {
        /// Target maximum intermediate size in scalar elements.
        target_size: usize,
        /// Size reduction factor per forest round.
        step_size: usize,
        /// Number of candidate trees in the forest.
        num_trees: usize,
        /// Number of repeated slicing searches per candidate.
        max_repeats: usize,
        /// Whether cotengra may slice output/open indices.
        allow_outer: bool,
    },
}

#[cfg(feature = "cotengra-python")]
impl Default for CotengraPythonConfig {
    fn default() -> Self {
        Self {
            method: CotengraPythonMethod::AutoHq,
            minimize: CotengraMinimize::Flops,
            max_repeats: 128,
            seed: Some(0),
            parallel: false,
            slicing: CotengraSlicingConfig::None,
            python: None,
            python_args: Vec::new(),
        }
    }
}

#[cfg(feature = "cotengra-python")]
impl CotengraPythonConfig {
    /// Default config launched through `uv run --project <project> python`.
    pub fn with_uv_project(project: impl Into<String>) -> Self {
        Self::default().uv_project(project)
    }

    /// Launch this config through `uv run --project <project> python`.
    pub fn uv_project(mut self, project: impl Into<String>) -> Self {
        let project = resolve_cotengra_uv_project(project.into());
        self.python = Some("uv".to_string());
        self.python_args = vec![
            "run".to_string(),
            "--project".to_string(),
            project,
            "python".to_string(),
        ];
        self
    }

    /// Launch this config through a specific Python executable.
    pub fn python(mut self, python: impl Into<String>) -> Self {
        self.python = Some(python.into());
        self.python_args.clear();
        self
    }
}

#[cfg(feature = "cotengra-python")]
fn resolve_cotengra_uv_project(project: String) -> String {
    let path = std::path::Path::new(&project);
    if path.is_absolute() || path.exists() {
        return project;
    }

    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(workspace) = manifest.parent() {
        let workspace_path = workspace.join(&project);
        if workspace_path.exists() {
            return workspace_path.to_string_lossy().into_owned();
        }
    }

    project
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
    /// Execution workspaces allocated because no cached lease was available.
    pub workspaces_created: u64,
    /// Execution workspace leases served from the per-plan pool.
    pub workspace_reuses: u64,
    /// Slot-table capacity growth events across cached workspaces.
    pub workspace_slot_grows: u64,
    /// Owned topology materializations performed by the static macro path.
    pub topology_materializations: u64,
}
