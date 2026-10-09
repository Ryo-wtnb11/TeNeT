mod api;
mod backend;
mod checked_generic;
#[doc(hidden)]
pub use checked_generic::{
    tensorcompose_owned_checked_generic_in_context, tensorcontract_owned_checked_generic_in_context,
};
mod context;
mod dynamic;
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use dynamic::cuda::CudaContractScratch;
#[cfg(test)]
pub(crate) use dynamic::{
    execute_dynamic_tree_execution_artifact_for_test,
    execute_dynamic_tree_execution_artifact_profile_pair_for_test,
    execute_prelowered_dynamic_tree_execution_artifact_for_test, profiled_artifact_compile_phases,
    reset_profiled_artifact_compile_phases, reset_source_layout_homspace_id_comparisons,
    source_layout_homspace_id_comparisons, tensorcontract_fusion_dynamic_plan_into_with,
};
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use route_cuda::{
    execute_storage_contract_members_cuda, execute_storage_contract_resolution_on_cuda,
    CudaContractMembersWorkspace,
};
#[doc(hidden)]
pub use route_host::HostContractMembersWorkspace;
mod dynamic_space;
mod fusion;
#[cfg(test)]
pub(crate) use fusion::contracted_axis_order_candidates;
#[doc(hidden)]
pub use fusion::FusionContractOrientation;
#[cfg(test)]
pub(crate) use fusion::{
    candidate_score_calls, prepare_tensorcontract_fusion_candidate_facts_dyn_raw,
    prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation,
    reset_candidate_score_calls, FusionContractCandidateFacts,
};
mod fusion_block;
mod resolution;
#[cfg(feature = "cuda")]
mod route_cuda;
mod route_host;
#[doc(hidden)]
pub use resolution::{
    copy_c_output_transform, zero_copy_contract_order_for_output_permute, CopyCRoute, CoreMiss,
    CoreRoute, DirectCoreExecutor, ExecCaps, HostEagerExecutor, StorageContractResolution,
};
#[cfg(test)]
mod candidate_core_tests;
mod scratch;
#[cfg(test)]
mod storage_contract_tests;
mod structure;

pub use api::{
    tensorcontract_execute_with, tensorcontract_into, tensorcontract_into_with, tensorproduct_into,
    tensorproduct_into_with_conjugation,
};
#[cfg(test)]
pub(crate) use backend::tensorcontract_structure_with_dense_executor_raw;
pub use backend::{
    HostTensorContractBackend, HostTensorContractWorkspace, TensorContractBackend,
    TensorContractWorkspace,
};
#[cfg(test)]
pub use context::tensorcontract_fusion_dyn_prelowered_direct_on_storage;
pub use context::{
    compile_direct_composition_plan, tensorcompose_fusion_dyn_prelowered_direct_on_storage,
    tensorcontract_into_with_context, try_compile_storage_contract_core_route,
    HostTreeFusionExecutionContext, PreparedTensorContractFusion, TensorContractCache,
    TensorContractCacheStats, TensorContractExecutionContext, TensorContractFusionExecutionContext,
    TensorContractPlanKey,
};
#[cfg(test)]
pub(crate) use dynamic_space::{
    checked_layout_primer, checked_metadata_dispatcher, fusion_operand_projection_prepares,
    reset_fusion_operand_projection_prepares, reset_scratch_publication_observations,
    scratch_publication_observations, MetadataOutput, MetadataRequest,
};
pub(crate) use dynamic_space::{
    dispatch_prepare, tree_transform_operation_axes, FusionOperandLayout, LayoutKeyBuilder,
};
pub use dynamic_space::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand,
    PreparedCheckedGenericDynamicSpace, ValidatedDynamicFusionLayout,
};
pub(crate) use fusion::rhs_contract_twist_factor_oriented;
#[cfg(test)]
pub(crate) use fusion::{
    contracted_fusion_tree_basis_matches, EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CORE_DST,
};
pub use fusion::{
    prepare_tensorcontract_fusion_plan, prepare_tensorcontract_fusion_plan_dyn,
    tensorcontract_fusion_block_specs, FusionContractPlan,
};
#[cfg(test)]
pub(crate) use structure::TensorContractDenseRouteKind;
pub use structure::{
    tensorcontract_structure, TensorContractBlockSpec, TensorContractStructure,
    TensorContractStructureTerm,
};
pub use tenet_operations::TensorContractFusionProfile;
#[cfg(test)]
pub use tenet_operations::TensorContractFusionRoute;

#[cfg(test)]
pub(crate) use dynamic_space::encoded_layout_primer;
