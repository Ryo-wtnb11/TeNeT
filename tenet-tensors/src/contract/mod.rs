mod api;
mod backend;
mod context;
mod dynamic;
mod dynamic_space;
mod fusion;
#[cfg(test)]
pub(crate) use fusion::contracted_axis_order_candidates;
#[cfg(test)]
pub(crate) use fusion::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order;
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
pub use backend::{
    HostTensorContractBackend, HostTensorContractWorkspace, TensorContractBackend,
    TensorContractWorkspace,
};
pub use context::{
    tensorcontract_into_with_context, HostTreeFusionExecutionContext, PreparedTensorContractFusion,
    TensorContractBlockPlanKey, TensorContractBlockPlanTerm, TensorContractCache,
    TensorContractCacheStats, TensorContractExecutionContext, TensorContractFusionExecutionContext,
    TensorContractPlanKey,
};
pub(crate) use dynamic_space::LayoutKeyBuilder;
#[cfg(test)]
pub(crate) use dynamic_space::{encoded_layout_primer, lowered_layout_primer};
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
