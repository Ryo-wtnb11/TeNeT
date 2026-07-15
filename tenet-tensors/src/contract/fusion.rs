mod block_specs;
mod plan;
#[cfg(test)]
pub(crate) use plan::contracted_axis_order_candidates;

#[cfg(test)]
pub(crate) use block_specs::contracted_fusion_tree_basis_matches;
pub(crate) use block_specs::{external_axis_is_dual, rhs_contract_twist_factor};
pub(crate) use block_specs::{
    reject_fusion_contract_conjugation, tensorcontract_fusion_structure_dyn_raw,
    EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CORE_DST, SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
};
pub use block_specs::{
    tensorcontract_fusion_block_specs, tensorcontract_fusion_structure,
    tensorcontract_fusion_structure_dyn,
};
pub(crate) use plan::prepare_tensorcontract_fusion_plan_dyn_raw;
pub use plan::{
    prepare_tensorcontract_fusion_plan, prepare_tensorcontract_fusion_plan_dyn, FusionContractPlan,
};
