mod cache;
mod helpers;
mod operation;
mod plan;

pub use cache::{
    TreePairTransformCache, TreeTransformCache, TreeTransformCacheStats, TreeTransformPlanScope,
    TreeTransformSectorPlanKey, TreeTransformSourceGroupKey,
};
pub use operation::{
    TreeTransformBuiltinRuleCacheKey, TreeTransformOperation, TreeTransformProductRuleCacheKey,
    TreeTransformRuleCacheKey, TreeTransformSu3RuleCacheKey,
};
pub use plan::{
    build_all_codomain_tree_transform_group_plan, build_generic_tree_pair_transform_group_plan,
    build_tree_pair_transform_group_plan, build_tree_transform_group_plan, TreeTransformBlockSpec,
    TreeTransformGroupBlockSpec, TreeTransformGroupPlan, TreeTransformKeyBlockSpec,
};
pub(crate) use plan::{
    build_generic_tree_pair_transform_group_plan_validated,
    build_tree_pair_transform_group_plan_validated, validate_generic_tree_pair_preflight,
    validate_multiplicity_free_tree_pair_preflight,
};

#[cfg(test)]
pub(crate) use cache::{TreeTransformGroupPlanCache, TreeTransformGroupPlanKey};
#[cfg(test)]
pub(crate) use plan::{
    build_all_codomain_tree_transform_group_plan_validated_with_threads,
    build_multiplicity_free_all_codomain_tree_transform_group_plan,
    build_multiplicity_free_tree_pair_transform_group_plan,
    build_tree_pair_transform_group_plan_validated_with_threads,
    build_unique_all_codomain_tree_transform_group_plan,
    build_unique_tree_pair_transform_group_plan, build_unique_tree_transform_group_plan,
    partition_staged_groups_for_test, reset_tree_pair_lowering_calls,
    reset_tree_pair_operation_preparations, tree_pair_lowering_calls,
    tree_pair_operation_preparations, validate_multiplicity_free_all_codomain_preflight,
};
