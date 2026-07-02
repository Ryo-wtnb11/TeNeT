use std::time::Duration;

use crate::TreeTransformReplayProfile;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TensorContractFusionRoute {
    #[default]
    Unset,
    CanonicalFusionBlocks,
    DenseFusionStructure,
    DenseConjugateStructure,
    DynamicTreeCanonical,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TensorContractFusionProfile {
    pub route: TensorContractFusionRoute,
    pub total: Duration,
    pub typed_space_setup: Duration,
    pub canonical_route_check: Duration,
    pub dense_block_specs: Duration,
    pub dense_structure_lookup: Duration,
    pub dense_contract: Duration,
    pub explicit_plan: Duration,
    pub source_space_lookup: Duration,
    pub lhs_scratch_prepare: Duration,
    pub rhs_scratch_prepare: Duration,
    pub lhs_transform: Duration,
    pub rhs_transform: Duration,
    pub canonical_dst_space_lookup: Duration,
    pub dst_scratch_prepare: Duration,
    pub fusion_block_plan_lookup: Duration,
    pub canonical_contract_total: Duration,
    pub canonical_validate: Duration,
    pub canonical_scale: Duration,
    pub canonical_workspace_prepare: Duration,
    pub canonical_pack_lhs: Duration,
    pub canonical_pack_rhs: Duration,
    pub canonical_matmul: Duration,
    pub canonical_scatter: Duration,
    pub output_transform: Duration,
    pub lhs_transform_calls: usize,
    pub rhs_transform_calls: usize,
    pub output_transform_calls: usize,
    pub canonical_contract_groups: usize,
    pub canonical_direct_pack_skips: usize,
    pub canonical_direct_gemm_groups: usize,
    pub tree_replay: TreeTransformReplayProfile,
}

impl TensorContractFusionProfile {
    #[inline]
    pub fn accumulate(&mut self, other: &Self) {
        self.total += other.total;
        self.typed_space_setup += other.typed_space_setup;
        self.canonical_route_check += other.canonical_route_check;
        self.dense_block_specs += other.dense_block_specs;
        self.dense_structure_lookup += other.dense_structure_lookup;
        self.dense_contract += other.dense_contract;
        self.explicit_plan += other.explicit_plan;
        self.source_space_lookup += other.source_space_lookup;
        self.lhs_scratch_prepare += other.lhs_scratch_prepare;
        self.rhs_scratch_prepare += other.rhs_scratch_prepare;
        self.lhs_transform += other.lhs_transform;
        self.rhs_transform += other.rhs_transform;
        self.canonical_dst_space_lookup += other.canonical_dst_space_lookup;
        self.dst_scratch_prepare += other.dst_scratch_prepare;
        self.fusion_block_plan_lookup += other.fusion_block_plan_lookup;
        self.canonical_contract_total += other.canonical_contract_total;
        self.canonical_validate += other.canonical_validate;
        self.canonical_scale += other.canonical_scale;
        self.canonical_workspace_prepare += other.canonical_workspace_prepare;
        self.canonical_pack_lhs += other.canonical_pack_lhs;
        self.canonical_pack_rhs += other.canonical_pack_rhs;
        self.canonical_matmul += other.canonical_matmul;
        self.canonical_scatter += other.canonical_scatter;
        self.output_transform += other.output_transform;
        self.lhs_transform_calls += other.lhs_transform_calls;
        self.rhs_transform_calls += other.rhs_transform_calls;
        self.output_transform_calls += other.output_transform_calls;
        self.canonical_contract_groups += other.canonical_contract_groups;
        self.canonical_direct_pack_skips += other.canonical_direct_pack_skips;
        self.canonical_direct_gemm_groups += other.canonical_direct_gemm_groups;
        self.tree_replay.accumulate(&other.tree_replay);
        if self.route == TensorContractFusionRoute::Unset {
            self.route = other.route;
        }
    }
}
