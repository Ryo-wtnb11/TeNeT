mod api;
mod backend;
mod context;
mod dynamic;
#[cfg(test)]
pub(crate) use dynamic::{
    reset_source_layout_homspace_id_comparisons, source_layout_homspace_id_comparisons,
    tensorcontract_fusion_dynamic_plan_into_with,
};
mod dynamic_space;
mod fusion;
#[cfg(test)]
pub(crate) use fusion::contracted_axis_order_candidates;
#[cfg(test)]
pub(crate) use fusion::{
    candidate_score_calls, prepare_tensorcontract_fusion_candidate_facts_dyn_raw,
    prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation,
    reset_candidate_score_calls, FusionContractCandidateFacts, FusionContractOrientation,
};
mod fusion_block;
mod resolution;
mod scratch;
mod structure;

pub use api::{
    tensorcontract_execute_with, tensorcontract_fusion_into, tensorcontract_fusion_into_with,
    tensorcontract_fusion_into_with_backends, tensorcontract_fusion_prepared_into,
    tensorcontract_fusion_prepared_into_core_dst,
    tensorcontract_fusion_prepared_into_core_dst_with, tensorcontract_fusion_prepared_into_with,
    tensorcontract_fusion_via_tree_pair_transforms_into, tensorcontract_into,
    tensorcontract_into_with, tensorproduct_fusion_into,
    tensorproduct_fusion_into_with_conjugation, tensorproduct_into,
    tensorproduct_into_with_conjugation,
};
#[cfg(test)]
pub(crate) use backend::{
    tensorcontract_structure_with_dense_executor_raw,
    tensorcontract_structure_with_storage_workspace_dense_executor,
};
pub use backend::{
    HostTensorContractBackend, HostTensorContractWorkspace, TensorContractBackend,
    TensorContractWorkspace,
};
pub use context::{
    tensorcontract_into_with_context, HostTreeFusionExecutionContext, PreparedTensorContractFusion,
    TensorContractCache, TensorContractCacheStats, TensorContractExecutionContext,
    TensorContractFusionExecutionContext, TensorContractPlanKey,
};
pub(crate) use dynamic_space::{dispatch_prepare, LayoutKeyBuilder};
#[cfg(test)]
pub(crate) use dynamic_space::{
    encoded_layout_primer, lowered_layout_primer, lowered_metadata_dispatcher,
    reset_scratch_publication_observations, scratch_publication_observations, MetadataOutput,
    MetadataRequest,
};
pub use dynamic_space::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand, ValidatedDynamicFusionLayout,
};
#[cfg(test)]
pub(crate) use fusion::{
    contracted_fusion_tree_basis_matches, EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CORE_DST,
};
pub use fusion::{
    prepare_tensorcontract_fusion_plan, prepare_tensorcontract_fusion_plan_dyn,
    tensorcontract_fusion_block_specs, tensorcontract_fusion_structure,
    tensorcontract_fusion_structure_dyn, FusionContractPlan,
};
#[cfg(test)]
pub(crate) use structure::TensorContractDenseRouteKind;
pub use structure::{
    tensorcontract_structure, TensorContractBlockSpec, TensorContractStructure,
    TensorContractStructureTerm,
};
pub use tenet_operations::{TensorContractFusionProfile, TensorContractFusionRoute};
