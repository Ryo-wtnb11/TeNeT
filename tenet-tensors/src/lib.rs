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
mod axis;
mod backend;
mod cache;
mod contract;
mod facade;
mod host_kernels;
mod lowering;
mod structure_identity;
mod tensoradd;
mod tensortrace;
mod tree_context;
mod tree_structure;
mod tree_transform;

pub use adjoint::adjoint;
pub use axis::{
    AxisPermutation, OwnedTensorContractAxisSpec, TensorContractAxisSpec, TensorTraceAxisSpec,
};
pub use backend::{
    DenseTreeTransformOperations, HostAllocator, HostTensorOperations, HostTensorOperationsBackend,
    HostTensorOperationsWorkspace, HostTreeTransformBackend, TensorOperationsBackend,
    TreeTransformBackend,
};
pub use cache::{
    BlockStructureCacheBlockKey, BlockStructureCacheKey, OperationCachePolicy,
    TensorContractStructureCache, TensorContractStructureCacheKey, TreeTransformStructureCache,
    TreeTransformStructureCacheKey,
};
#[cfg(test)]
pub(crate) use contract::{
    contracted_fusion_tree_basis_matches, TensorContractDenseRouteKind,
    EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
};
pub use contract::{
    tensorcontract_execute_with, tensorcontract_fusion_block_specs,
    tensorcontract_fusion_explicit_plan, tensorcontract_fusion_explicit_plan_into,
    tensorcontract_fusion_explicit_plan_into_canonical_dst,
    tensorcontract_fusion_explicit_plan_into_canonical_dst_with,
    tensorcontract_fusion_explicit_plan_into_with, tensorcontract_fusion_into,
    tensorcontract_fusion_into_with, tensorcontract_fusion_into_with_backends,
    tensorcontract_fusion_structure, tensorcontract_fusion_via_tree_pair_transforms_into,
    tensorcontract_into, tensorcontract_into_with, tensorcontract_into_with_context,
    tensorcontract_structure, tensorproduct_fusion_into,
    tensorproduct_fusion_into_with_conjugation, tensorproduct_into,
    tensorproduct_into_with_conjugation, HostTensorContractBackend, HostTensorContractWorkspace,
    HostTreeFusionExecutionContext, TensorContractBackend, TensorContractBlockPlanKey,
    TensorContractBlockPlanTerm, TensorContractBlockSpec, TensorContractCache,
    TensorContractCacheStats, TensorContractExecutionContext, TensorContractFusionExecutionContext,
    TensorContractFusionExplicitPlan, TensorContractFusionProfile, TensorContractFusionRoute,
    TensorContractPlanKey, TensorContractStructure, TensorContractStructureTerm,
    TensorContractWorkspace,
};
pub use facade::{
    all_codomain_tree_transform_into_with_context, copy_into, scaled_add_into, scaled_assign_into,
    tensoradd_add_into, tensoradd_assign_into, tensoradd_execute_with, tensoradd_fusion_into,
    tensoradd_fusion_into_with, tensoradd_fusion_into_with_context, tensoradd_into,
    tensoradd_into_with, tensoradd_into_with_backend_and_conjugation,
    tensoradd_into_with_conjugation, tensorcopy_into, tensorcopy_into_with,
    tensortrace_execute_with, tensortrace_fusion_execute_with, tensortrace_fusion_into,
    tensortrace_fusion_into_with, tensortrace_into, tensortrace_into_with,
    tree_pair_transform_into, tree_pair_transform_into_with, tree_pair_transform_into_with_context,
    tree_pair_transform_structure, tree_transform_execute_with,
};
pub(crate) use host_kernels::{
    tensoradd_structure_with_strided_kernel, tree_transform_structure_with_strided_kernel,
    tree_transform_structure_with_strided_kernel_raw,
    tree_transform_structure_with_structural_recoupling,
    tree_transform_structure_with_structural_recoupling_raw,
    tree_transform_structure_with_structural_recoupling_raw_profiled,
};
pub use host_kernels::{HostTreeTransformWorkspace, TreeTransformWorkspace};
pub use tenet_operations::OperationError;
pub use tenet_operations::ReportsPlacement;
pub use tenet_operations::TreeTransformReplayProfile;
pub(crate) use tenet_operations::{
    copy_block_with_strided_kernel, tensoradd_raw_strided_kernel,
    tensoradd_raw_strided_kernel_trusted, tensortrace_raw_strided_kernel,
    tensortrace_raw_strided_kernel_add_with_coefficient,
};
pub(crate) use tenet_operations::{host_scratch, storage_scratch, strided};
pub use tenet_operations::{
    ConjugateValue, DenseBlockScalar, DenseRecouplingScalar, RealStructuralCoefficient,
    RecouplingCoefficientAction, TreeTransformScalar,
};
pub(crate) use tenet_operations::{HostKernelAdapter, StridedHostKernelAdapter};
pub use tensoradd::{
    tensoradd_structure, tensoradd_structure_with_conjugation, TensorAddStructure,
    TensorAddStructureTerm,
};
pub(crate) use tensortrace::tensortrace_fusion_structure_with_strided_kernel;
pub(crate) use tensortrace::tensortrace_structure_with_strided_kernel;
pub use tensortrace::{
    tensortrace_fusion_structure, tensortrace_structure, TensorTraceFusionStructure,
    TensorTraceFusionStructureTerm, TensorTraceStructure, TensorTraceStructureTerm,
};
pub use tree_context::TreeTransformExecutionContext;
pub use tree_structure::TreeTransformStructure;
pub(crate) use tree_structure::{
    TreeTransformBlock, TreeTransformLayout, TreeTransformLayoutTable,
};
pub use tree_transform::{
    build_all_codomain_tree_transform_group_plan, build_tree_pair_transform_group_plan,
    build_tree_transform_group_plan, TreePairTransformCache, TreeTransformBlockSpec,
    TreeTransformBuiltinRuleCacheKey, TreeTransformCache, TreeTransformCacheStats,
    TreeTransformGroupBlockSpec, TreeTransformGroupPlan, TreeTransformKeyBlockSpec,
    TreeTransformOperationKey, TreeTransformPlanScope, TreeTransformProductRuleCacheKey,
    TreeTransformRuleCacheKey, TreeTransformSectorPlanKey, TreeTransformSourceGroupKey,
};
#[cfg(test)]
pub(crate) use tree_transform::{
    build_unique_all_codomain_tree_transform_group_plan,
    build_unique_tree_pair_transform_group_plan, build_unique_tree_transform_group_plan,
    TreeTransformGroupPlanCache, TreeTransformGroupPlanKey,
};

#[cfg(test)]
mod tests;
