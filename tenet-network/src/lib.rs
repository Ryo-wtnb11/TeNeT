#![forbid(unsafe_code)]

//! # tenet-network — contraction planning and execution for the TeNeT user layer
//!
//! The planner half (labels, [`NetworkIR`], cost models, optimizer trait,
//! [`ContractionPlan`], slicing) is ported nearly verbatim from the legacy
//! `tenet-contract` crate: it is **pure structure** over labels and leg
//! dimensions and never touches tensor data. Explicit execution uses homogeneous
//! typed Host [`tenet::typed::TensorMap`] operands and a caller-owned typed
//! workspace; device operands run the same schedule on CUDA when enabled.
//!
//! ## Pipeline
//!
//! ```text
//! Network::new(labels) ->  Network (label lists + conj markers + output)
//!   -> NetworkIR + DenseCostModel        (per-label dimension map)
//!   -> DenseContractionOptimizer         (greedy by default)
//!   -> ContractionPlan                   (reusable, serializable)
//!   -> PlannedNetwork::execute(&[&TensorMap<R, D>], &mut workspace) -> TensorMap<R, D>
//! ```
//!
//! [`Network::contract`] runs the same pipeline eagerly through the
//! Runtime's topology-keyed plan cache.
//!
//! There is **no public einsum-string parser**: labels are per-operand
//! identifier lists given to [`Network::new`] and lower directly to
//! [`NetworkIR`].
//!
//! ## Follow-ups (intentionally not in this round)
//!
//! - **cotengra external path search**: the optional `cotengra-python`
//!   feature calls the installed Python `cotengra` package for path search
//!   while keeping execution in Rust. The optional `opt-path` feature wraps
//!   the `opt-einsum-path` crate for optimal / dp / branch-and-bound searches.
//! - **Sliced execution**: the slicing *decision* types ([`SlicePlan`],
//!   [`greedy_slice`]) are ported, and [`Network::lower_symmetric_sliced_plan`]
//!   with [`Network::execute_symmetric_sliced`] runs a lowered plan under a
//!   measured payload ceiling.

#[cfg(test)]
extern crate self as tenet_network;

#[cfg(test)]
#[path = "../../tenet/tests/braiding_probe/mod.rs"]
mod braiding_probe;

mod cost;
#[cfg(feature = "cotengra-python")]
mod cotengra_python;
mod error;
mod ir;
mod labels;
mod network;
mod optimizer;
#[cfg(test)]
pub(crate) mod parse;
#[cfg(feature = "opt-path")]
mod pathopt;
mod plan;
mod plancache;
mod slice;
mod tree;

pub use cost::{DenseCostModel, DenseTensorInfo};
#[cfg(feature = "cotengra-python")]
pub use cotengra_python::CotengraPythonOptimizer;
pub use error::{
    ContractError, Result, SliceError, SliceResult, SymmetricSliceExecutionError,
    SymmetricSliceLowerError,
};
pub use ir::{HyperEdge, NetworkIR, TensorNode};
pub use labels::{LabelOccurrence, TemporaryLabel, TensorAxis, TensorId};
pub use network::{
    Network, NetworkExecutionWorkspace, NetworkOperand, PlannedNetwork, SymmetricSliceStats,
};
pub use optimizer::{
    ContractionStep, DenseContractionOptimizer, DensePlanCostReport, GreedyDenseOptimizer,
    LabelOrderDenseOptimizer,
};
#[cfg(feature = "opt-path")]
pub use pathopt::{
    BranchLevel, DpObjective, OptEinsumPathOptimizer, PathMemoryLimit, PathStrategy,
};
pub use plan::{
    active_pair_path_from_steps, active_pair_path_from_tree, dense_steps_from_active_pair_path,
    ActivePair, ContractionPlan,
};
pub use plancache::{
    clear_plan_cache, configure_plan_cache, load_plan_cache, plan_cache_stats, save_plan_cache,
    Optimizer, PlanCacheConfig, PlanCacheStats, ReplanPolicy, DEFAULT_PLAN_CACHE_CAPACITY,
    DEFAULT_REPLAN_DRIFT_FACTOR,
};
pub use slice::{
    best_next_internal_index, best_next_slice_index, contraction_width, greedy_slice,
    slice_plan_for, DegeneracyRange, SectorSlice, SliceKind, SliceLabels, SlicePlan, SlicedPlan,
    SymmetricIndexSlice, SymmetricSlicePlan, SymmetricSliceSpec, SymmetricSlicedPlan,
};
#[cfg(feature = "cotengra-python")]
pub use tenet::plancache::{
    CotengraMinimize, CotengraPythonConfig, CotengraPythonMethod, CotengraSlicingConfig,
};
pub use tree::ContractionTree;

/// The workspace tolerance rule for arithmetic test comparisons
/// (`docs/testing_numerics.md`).
#[cfg(test)]
#[path = "../../tests/support"]
mod test_numerics {
    use tenet::typed::{Complex32, Complex64};
    pub(crate) mod numerics;
}
