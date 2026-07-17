#![forbid(unsafe_code)]

//! Symmetric-tensor execution for TeNeT (the operational half of what
//! TensorKit.jl does in Julia): fusion-tree transforms (F/R recoupling,
//! braids, permutes), symmetric contraction routes, and the basic tensor
//! operations, lowered onto strided-rs kernels and the dense executor the
//! same way TensorKit lowers onto Strided.jl and dense backends. The
//! symmetry-agnostic einsum layer (TensorOperations.jl's role) lives below
//! this crate in strided-rs / tenferro; the structural data layer lives in
//! `tenet-core`.

mod adjoint;
mod backend_trace;
mod cache;
mod contract;
mod facade;
mod lowering;
mod tensortrace;
#[cfg(test)]
mod test_support;
mod tree_context;
mod tree_transform;

pub use adjoint::{
    adjoint, adjoint_bound_dyn, adjoint_bound_dyn_generic, adjoint_bound_dyn_lowered,
    adjoint_bound_space_dyn, adjoint_bound_space_dyn_generic, adjoint_bound_space_dyn_lowered,
};
pub use backend_trace::TensorTraceOperationsBackend;
pub use cache::{
    operation_cache_reset_epoch, registered_operation_cache, reset_global_operation_caches,
    BlockStructureCacheBlockKey, BlockStructureCacheKey, OperationCachePolicy,
    TensorContractStructureCache, TensorContractStructureCacheKey, TreeTransformStructureCache,
    TreeTransformStructureCacheKey,
};
#[cfg(test)]
pub(crate) use contract::{
    contracted_fusion_tree_basis_matches, TensorContractDenseRouteKind,
    EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CORE_DST,
};
pub use contract::{
    prepare_tensorcontract_fusion_plan, prepare_tensorcontract_fusion_plan_dyn,
    tensorcontract_execute_with, tensorcontract_fusion_block_specs, tensorcontract_fusion_into,
    tensorcontract_fusion_into_with, tensorcontract_fusion_into_with_backends,
    tensorcontract_fusion_prepared_into, tensorcontract_fusion_prepared_into_core_dst,
    tensorcontract_fusion_prepared_into_core_dst_with, tensorcontract_fusion_prepared_into_with,
    tensorcontract_fusion_structure, tensorcontract_fusion_structure_dyn,
    tensorcontract_fusion_via_tree_pair_transforms_into, tensorcontract_into,
    tensorcontract_into_with, tensorcontract_into_with_context, tensorcontract_structure,
    tensorproduct_fusion_into, tensorproduct_fusion_into_with_conjugation, tensorproduct_into,
    tensorproduct_into_with_conjugation, FusionContractPlan, HostTensorContractBackend,
    HostTensorContractWorkspace, HostTreeFusionExecutionContext, PreparedTensorContractFusion,
    TensorContractBackend, TensorContractBlockSpec, TensorContractCache, TensorContractCacheStats,
    TensorContractExecutionContext, TensorContractFusionExecutionContext,
    TensorContractFusionProfile, TensorContractFusionRoute, TensorContractPlanKey,
    TensorContractStructure, TensorContractStructureTerm, TensorContractWorkspace,
};
pub use contract::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand, ValidatedDynamicFusionLayout,
};
pub use facade::{
    braid_into, braid_into_with, braid_into_with_context, permute_into, permute_into_with,
    permute_into_with_context, transpose_into, transpose_into_with, transpose_into_with_context,
};
// Stage B3a: Generic-fusion (outer-multiplicity) facade siblings.
pub use facade::{
    braid_into_generic, permute_into_generic, transpose_into_generic, tree_transform_into_generic,
    tree_transform_into_with_generic, tree_transform_structure_generic,
};
pub use facade::{
    copy_into, scaled_add_into, scaled_assign_into, tensoradd_add_into, tensoradd_assign_into,
    tensoradd_execute_with, tensoradd_fusion_into, tensoradd_fusion_into_with,
    tensoradd_fusion_into_with_context, tensoradd_into, tensoradd_into_with,
    tensoradd_into_with_backend_and_conjugation, tensoradd_into_with_conjugation, tensorcopy_into,
    tensorcopy_into_with, tensortrace_execute_with, tensortrace_fusion_execute_with,
    tensortrace_fusion_into, tensortrace_fusion_into_with, tensortrace_into, tensortrace_into_with,
    tree_transform_execute_with, tree_transform_into, tree_transform_into_with,
    tree_transform_into_with_context, tree_transform_overwrite_execute_with,
    tree_transform_overwrite_into, tree_transform_overwrite_into_with,
    tree_transform_overwrite_into_with_context, tree_transform_structure,
};
/// CUDA storage and GEMM seams (flat device buffers, never host-readable).
#[cfg(feature = "cuda")]
pub use tenet_operations::cuda;
pub use tenet_operations::OperationError;
pub use tenet_operations::ReportsPlacement;
pub use tenet_operations::TreeTransformReplayProfile;
pub use tenet_operations::TreeTransformStructure;
pub(crate) use tenet_operations::{host_scratch, storage_scratch, strided};
pub use tenet_operations::{
    tensoradd_structure, tensoradd_structure_with_conjugation, TensorAddStructure,
    TensorAddStructureTerm,
};
pub(crate) use tenet_operations::{
    tensortrace_raw_strided_kernel, tensortrace_raw_strided_kernel_add_with_coefficient,
};
pub use tenet_operations::{
    ConjugateValue, DenseBlockScalar, DenseRecouplingScalar, RealStructuralCoefficient,
    RecouplingCoefficientAction, TreeTransformScalar,
};
pub use tenet_operations::{
    DenseTreeTransformOperations, HostAllocator, HostTensorOperations, HostTensorOperationsBackend,
    HostTensorOperationsWorkspace, HostTreeTransformBackend, TensorOperationsBackend,
    TreeTransformBackend,
};
pub(crate) use tenet_operations::{HostKernelAdapter, StridedHostKernelAdapter};
pub use tenet_operations::{HostTreeTransformWorkspace, TransposeBackend, TreeTransformWorkspace};
pub use tenet_operations::{OutputAxisOrder, TensorContractSpec, TensorTraceAxisSpec};
pub use tensortrace::{
    tensortrace_fusion_dyn_into, tensortrace_fusion_structure, tensortrace_structure,
    TensorTraceFusionStructure, TensorTraceFusionStructureTerm, TensorTraceStructure,
    TensorTraceStructureTerm,
};
pub use tree_context::TreeTransformExecutionContext;
pub use tree_transform::{
    build_all_codomain_tree_transform_group_plan, build_generic_tree_pair_transform_group_plan,
    build_tree_pair_transform_group_plan, build_tree_transform_group_plan, TreePairTransformCache,
    TreeTransformBlockSpec, TreeTransformBuiltinRuleCacheKey, TreeTransformCache,
    TreeTransformCacheStats, TreeTransformGroupBlockSpec, TreeTransformGroupPlan,
    TreeTransformKeyBlockSpec, TreeTransformOperation, TreeTransformPlanScope,
    TreeTransformProductRuleCacheKey, TreeTransformRuleCacheKey, TreeTransformSectorPlanKey,
    TreeTransformSourceGroupKey, TreeTransformSu3RuleCacheKey,
};
#[cfg(test)]
pub(crate) use tree_transform::{
    build_unique_all_codomain_tree_transform_group_plan,
    build_unique_tree_pair_transform_group_plan, build_unique_tree_transform_group_plan,
    TreeTransformGroupPlanCache, TreeTransformGroupPlanKey,
};

#[cfg(test)]
mod tests;
