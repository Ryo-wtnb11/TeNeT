mod api;
mod backend;
mod context;
mod dynamic;
mod dynamic_space;
mod fusion;
mod fusion_block;
mod profile;
mod scratch;
mod structure;

pub use api::{
    tensorcontract_execute_with, tensorcontract_fusion_explicit_plan_into,
    tensorcontract_fusion_explicit_plan_into_canonical_dst,
    tensorcontract_fusion_explicit_plan_into_canonical_dst_with,
    tensorcontract_fusion_explicit_plan_into_with, tensorcontract_fusion_into,
    tensorcontract_fusion_into_with, tensorcontract_fusion_into_with_backends,
    tensorcontract_fusion_via_tree_pair_transforms_into, tensorcontract_into,
    tensorcontract_into_with, tensorproduct_fusion_into,
    tensorproduct_fusion_into_with_conjugation, tensorproduct_into,
    tensorproduct_into_with_conjugation,
};
pub use backend::{TensorContractBackend, TensorContractWorkspace};
pub use context::{
    tensorcontract_into_with_context, HostTreeFusionExecutionContext, TensorContractBlockPlanKey,
    TensorContractBlockPlanTerm, TensorContractCache, TensorContractCacheStats,
    TensorContractExecutionContext, TensorContractFusionExecutionContext, TensorContractPlanKey,
};
#[cfg(test)]
pub(crate) use fusion::{
    contracted_fusion_tree_basis_matches, EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
};
pub use fusion::{
    tensorcontract_fusion_block_specs, tensorcontract_fusion_explicit_plan,
    tensorcontract_fusion_structure, TensorContractFusionExplicitPlan,
};
pub use profile::{TensorContractFusionProfile, TensorContractFusionRoute};
#[cfg(test)]
pub(crate) use structure::TensorContractDenseRouteKind;
pub use structure::{
    tensorcontract_structure, TensorContractBlockSpec, TensorContractStructure,
    TensorContractStructureTerm,
};
