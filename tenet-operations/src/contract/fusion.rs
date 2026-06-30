mod block_specs;
mod plan;

pub(crate) use block_specs::{
    contracted_fusion_tree_basis_matches, EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
    SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
};
pub use block_specs::{tensorcontract_fusion_block_specs, tensorcontract_fusion_structure};
pub use plan::{tensorcontract_fusion_explicit_plan, TensorContractFusionExplicitPlan};
