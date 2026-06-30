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
mod scalar;
mod strided;
mod structure_identity;
mod tensoradd;
mod tree_context;
mod tree_structure;
mod tree_transform;

pub use axis::{AxisPermutation, OwnedTensorContractAxisSpec, TensorContractAxisSpec};
pub use backend::{
    DenseTreeTransformOperations, HostAllocator, HostTensorOperations, TensorOperationsBackend,
    TreeTransformBackend,
};
pub use cache::{
    BlockStructureCacheBlockKey, BlockStructureCacheKey, TensorContractStructureCache,
    TensorContractStructureCacheKey, TreeTransformStructureCache, TreeTransformStructureCacheKey,
};
#[cfg(test)]
pub(crate) use contract::{
    contracted_fusion_tree_basis_matches, EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
};
pub use contract::{
    tensorcontract_execute_with, tensorcontract_fusion_block_specs,
    tensorcontract_fusion_explicit_plan, tensorcontract_fusion_explicit_plan_into,
    tensorcontract_fusion_explicit_plan_into_canonical_dst,
    tensorcontract_fusion_explicit_plan_into_canonical_dst_with,
    tensorcontract_fusion_explicit_plan_into_with, tensorcontract_fusion_into,
    tensorcontract_fusion_into_with, tensorcontract_fusion_structure,
    tensorcontract_fusion_via_tree_pair_transforms_into, tensorcontract_into,
    tensorcontract_into_with, tensorcontract_into_with_context, tensorcontract_structure,
    TensorContractBackend, TensorContractBlockPlanKey, TensorContractBlockPlanTerm,
    TensorContractBlockSpec, TensorContractCache, TensorContractCacheStats,
    TensorContractExecutionContext, TensorContractFusionExecutionContext,
    TensorContractFusionExplicitPlan, TensorContractPlanKey, TensorContractStructure,
    TensorContractStructureTerm, TensorContractWorkspace,
};
pub use error::OperationError;
pub use facade::{
    all_codomain_tree_transform_into_with_context, copy_into, scaled_add_into, scaled_assign_into,
    tensoradd_add_into, tensoradd_assign_into, tensoradd_execute_with, tensoradd_into,
    tensoradd_into_with, tensorcopy_into, tensorcopy_into_with, tree_pair_transform_into,
    tree_pair_transform_into_with, tree_pair_transform_into_with_context,
    tree_pair_transform_structure, tree_transform_execute_with,
};
pub use host_kernels::TreeTransformWorkspace;
pub(crate) use host_kernels::{
    copy_block_with_strided_kernel, tensoradd_raw_strided_kernel,
    tensoradd_structure_with_strided_kernel, tree_transform_structure_with_dense_recoupling,
    tree_transform_structure_with_strided_kernel,
};
pub use scalar::{
    DenseBlockScalar, DenseRecouplingScalar, RecouplingCoefficientAction, TreeTransformScalar,
};
pub use tensoradd::{tensoradd_structure, TensorAddStructure, TensorAddStructureTerm};
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
