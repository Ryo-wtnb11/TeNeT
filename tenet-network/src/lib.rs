#![forbid(unsafe_code)]

//! # tenet-network — contraction planning and execution for the TeNeT user layer
//!
//! The planner half (labels, [`NetworkIR`], cost models, optimizer trait,
//! [`ContractionPlan`], slicing) is ported nearly verbatim from the legacy
//! `tenet-contract` crate: it is **pure structure** over labels and leg
//! dimensions and never touches tensor data. The execution half is a thin
//! new loop over the user-layer [`tenet::prelude::Tensor`]
//! (`contract` + `permute` per planned pairwise step); the legacy old-core
//! executor was not ported.
//!
//! ## Pipeline
//!
//! ```text
//! tensor!(...) labels  ->  Network (label lists + conj markers + output)
//!   -> NetworkIR + DenseCostModel        (per-label dimension map)
//!   -> DenseContractionOptimizer         (greedy by default)
//!   -> ContractionPlan                   (reusable, serializable)
//!   -> PlannedNetwork::execute(&[&Tensor]) -> Tensor
//! ```
//!
//! There is **no public einsum-string parser** (decision 4 in
//! `docs/user_api_design.md`): labels are identifiers supplied by the
//! [`tensor!`] macro and lower directly to [`NetworkIR`].
//!
//! ## Follow-ups (intentionally not in this round)
//!
//! - **cotengra external path search**: the optional `cotengra-python`
//!   feature calls the installed Python `cotengra` package for path search
//!   while keeping execution in Rust. The optional `opt-path` feature wraps
//!   the `opt-einsum-path` crate for optimal / dp / branch-and-bound searches.
//! - **Sliced execution**: the slicing *decision* types ([`SlicePlan`],
//!   [`greedy_slice`]) are ported; a memory-bounded sliced executor over
//!   `Tensor` needs `select_index` on the user layer first.

#[cfg(feature = "opt-path")]
mod bitset_dp;
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

pub use cost::{
    BlockInfo, BlockLabelInfo, BlockSparseCostModel, BlockSparseTensorInfo, DenseCostModel,
    DenseTensorInfo,
};
#[cfg(feature = "cotengra-python")]
pub use cotengra_python::CotengraPythonOptimizer;
pub use error::{ContractError, Result};
pub use ir::{HyperEdge, NetworkIR, TensorNode};
pub use labels::{LabelOccurrence, TemporaryLabel, TensorAxis, TensorId};
pub use network::{contract_network, NetOperand, Network, PlannedNetwork};
pub use optimizer::{
    block_sparse_order_from_labels, greedy_order, greedy_order_block_sparse,
    BlockSparseContractionOptimizer, ContractionStep, DenseContractionOptimizer,
    DensePlanCostReport, GreedyBlockSparseOptimizer, GreedyDenseOptimizer,
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
    clear_plan_cache, configure_plan_cache, load_plan_cache, plan_cache_config, plan_cache_stats,
    save_plan_cache, Optimizer, PlanCacheConfig, PlanCacheStats, ReplanPolicy,
    DEFAULT_PLAN_CACHE_CAPACITY, DEFAULT_REPLAN_DRIFT_FACTOR,
};
pub use slice::{
    best_next_internal_index, best_next_slice_index, contraction_width, greedy_slice,
    greedy_slice_with_output, slice_plan_for, SliceKind, SlicePlan, SlicedPlan,
};
#[cfg(feature = "cotengra-python")]
pub use tenet::plancache::{
    CotengraMinimize, CotengraPythonConfig, CotengraPythonMethod, CotengraSlicingConfig,
};
pub use tree::ContractionTree;

/// The `tensor!` @tensor-style contraction macro (from `tenet-macros`).
pub use tenet_macros::tensor;
