#![forbid(unsafe_code)]

//! TensorOperations-style lowering for TeNeT.
//!
//! Public/core tensor code talks in terms of TeNeT-owned block views. This crate
//! lowers those views to strided-rs kernels at the same granularity that
//! TensorKit uses Strided.jl/StridedViews.jl internally.

mod axis;
mod backend;
mod cache;
mod contract;
mod error;
mod facade;
mod host_kernels;
mod host_scalar_kernels;
mod lowering;
mod placement;
mod scalar;
mod strided;
mod structure_identity;
mod tensoradd;
mod tensortrace;
mod tree_context;
mod tree_profile;
mod tree_structure;
mod tree_transform;

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
pub use error::OperationError;
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
pub(crate) use host_scalar_kernels::{
    axpby_raw_strided_kernel_trusted, copy_block_with_strided_kernel,
    copy_scale_raw_strided_kernel_trusted, copy_scale_raw_strided_kernel_with_conjugate_trusted,
    scale_raw_strided_kernel_trusted, tensoradd_raw_strided_kernel,
    tensoradd_raw_strided_kernel_profiled, tensoradd_raw_strided_kernel_trusted,
    tensortrace_raw_strided_kernel, tensortrace_raw_strided_kernel_add_with_coefficient,
};
pub use placement::ReportsPlacement;
pub use scalar::{
    ConjugateValue, DenseBlockScalar, DenseRecouplingScalar, RealStructuralCoefficient,
    RecouplingCoefficientAction, TreeTransformScalar,
};
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
pub use tree_profile::TreeTransformReplayProfile;
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
