mod cache;
mod operation;
mod plan;

pub use cache::{
    RuntimeTreeTransformCacheInfo, RuntimeTreeTransformStore, TreePairTransformCache,
    TreeTransformCache, TreeTransformCacheStats,
};
pub use operation::{
    TreeTransformOperation, TreeTransformOperationKind, TreeTransformRuleCacheKey,
};
pub use plan::{
    build_all_codomain_tree_transform_group_plan,
    build_checked_generic_tree_pair_transform_group_plan,
    build_generic_tree_pair_transform_group_plan, build_tree_pair_transform_group_plan,
    build_tree_transform_group_plan, CheckedGenericPlanError, TreeTransformBlockSpec,
    TreeTransformGroupBlockSpec, TreeTransformGroupPlan, TreeTransformKeyBlockSpec,
};
#[cfg(test)]
pub(crate) use plan::{
    build_all_codomain_tree_transform_group_plan_validated_with_threads,
    build_multiplicity_free_all_codomain_tree_transform_group_plan,
    build_multiplicity_free_tree_pair_transform_group_plan,
    build_tree_pair_transform_group_plan_validated_with_threads,
    build_unique_all_codomain_tree_transform_group_plan,
    build_unique_tree_pair_transform_group_plan, build_unique_tree_transform_group_plan,
    multiplicity_free_capability_validations, partition_staged_groups_for_test,
    reset_multiplicity_free_capability_validations, reset_tree_pair_lowering_calls,
    reset_tree_pair_operation_preparations, tree_pair_lowering_calls,
    tree_pair_operation_preparations, validate_multiplicity_free_all_codomain_preflight,
    validate_multiplicity_free_tree_pair_preflight,
};
pub(crate) use plan::{
    build_checked_generic_tree_pair_transform_group_plan_validated,
    build_generic_tree_pair_transform_group_plan_validated,
    compile_multiplicity_free_tree_pair_structure,
    validate_checked_generic_tree_pair_plan_preflight, validate_generic_tree_pair_preflight,
};
