mod block_specs;
mod plan;

pub(crate) use block_specs::{
    contracted_fusion_tree_basis_matches, reject_fusion_contract_conjugation,
    EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST, SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
};
pub(crate) use block_specs::{external_axis_is_dual, rhs_contract_twist_factor};
pub use block_specs::{tensorcontract_fusion_block_specs, tensorcontract_fusion_structure};
pub use plan::{tensorcontract_fusion_explicit_plan, TensorContractFusionExplicitPlan};
