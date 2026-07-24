use std::time::Duration;

use crate::TreeTransformReplayProfile;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TensorContractFusionRoute {
    #[default]
    Unset,
    CoreFusionBlocks,
    DenseFusionStructure,
    DenseConjugateStructure,
    DynamicTreeCore,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TensorContractFusionProfile {
    pub route: TensorContractFusionRoute,
    pub total: Duration,
    pub typed_space_setup: Duration,
    pub core_route_check: Duration,
    pub dense_block_specs: Duration,
    pub dense_structure_lookup: Duration,
    pub dense_contract: Duration,
    /// Reserved for source compatibility; ordinary contraction no longer
    /// looks up or publishes complete execution artifacts.
    pub prepared_plan: Duration,
    /// Route preflight and candidate selection for one ordinary eager call.
    pub resolution_preflight: Duration,
    /// DynamicTree plan construction after route selection.
    pub dynamic_tree_plan_build: Duration,
    pub source_space_lookup: Duration,
    pub lhs_scratch_prepare: Duration,
    pub rhs_scratch_prepare: Duration,
    pub lhs_transform: Duration,
    pub rhs_transform: Duration,
    pub core_dst_space_lookup: Duration,
    pub dst_scratch_prepare: Duration,
    /// DynamicTree artifact assembly excluding separately attributed structure
    /// lookups and core block-plan construction.
    pub dynamic_tree_artifact_prepare: Duration,
    /// Fresh coupled reduced-block plan construction.
    pub core_block_plan_build: Duration,
    pub core_contract_total: Duration,
    pub core_validate: Duration,
    pub core_scale: Duration,
    pub core_workspace_prepare: Duration,
    pub core_pack_lhs: Duration,
    pub core_pack_rhs: Duration,
    pub core_matmul: Duration,
    pub core_scatter: Duration,
    pub output_transform: Duration,
    pub lhs_transform_calls: usize,
    pub rhs_transform_calls: usize,
    pub output_transform_calls: usize,
    pub core_contract_groups: usize,
    pub core_direct_pack_skips: usize,
    pub core_direct_gemm_groups: usize,
    pub tree_replay: TreeTransformReplayProfile,
}

impl TensorContractFusionProfile {
    #[inline]
    pub fn accumulate(&mut self, other: &Self) {
        self.total += other.total;
        self.typed_space_setup += other.typed_space_setup;
        self.core_route_check += other.core_route_check;
        self.dense_block_specs += other.dense_block_specs;
        self.dense_structure_lookup += other.dense_structure_lookup;
        self.dense_contract += other.dense_contract;
        self.prepared_plan += other.prepared_plan;
        self.resolution_preflight += other.resolution_preflight;
        self.dynamic_tree_plan_build += other.dynamic_tree_plan_build;
        self.source_space_lookup += other.source_space_lookup;
        self.lhs_scratch_prepare += other.lhs_scratch_prepare;
        self.rhs_scratch_prepare += other.rhs_scratch_prepare;
        self.lhs_transform += other.lhs_transform;
        self.rhs_transform += other.rhs_transform;
        self.core_dst_space_lookup += other.core_dst_space_lookup;
        self.dst_scratch_prepare += other.dst_scratch_prepare;
        self.dynamic_tree_artifact_prepare += other.dynamic_tree_artifact_prepare;
        self.core_block_plan_build += other.core_block_plan_build;
        self.core_contract_total += other.core_contract_total;
        self.core_validate += other.core_validate;
        self.core_scale += other.core_scale;
        self.core_workspace_prepare += other.core_workspace_prepare;
        self.core_pack_lhs += other.core_pack_lhs;
        self.core_pack_rhs += other.core_pack_rhs;
        self.core_matmul += other.core_matmul;
        self.core_scatter += other.core_scatter;
        self.output_transform += other.output_transform;
        self.lhs_transform_calls += other.lhs_transform_calls;
        self.rhs_transform_calls += other.rhs_transform_calls;
        self.output_transform_calls += other.output_transform_calls;
        self.core_contract_groups += other.core_contract_groups;
        self.core_direct_pack_skips += other.core_direct_pack_skips;
        self.core_direct_gemm_groups += other.core_direct_gemm_groups;
        self.tree_replay.accumulate(&other.tree_replay);
        if self.route == TensorContractFusionRoute::Unset {
            self.route = other.route;
        }
    }
}
