mod cache;
mod operation;
mod plan;

pub use cache::{
    admit_exact_tree_pair_layout, exact_layout_tree_pair_hit, RuntimeTreeTransformCacheInfo,
    TreeTransformOperationView,
};
#[cfg(test)]
pub(crate) use cache::{
    before_next_completed_publication, take_coefficient_group_activity,
    take_completed_transformer_activity, take_oriented_tree_pair_compiles,
    take_trace_column_activity, CoefficientGroupActivity, CompletedActivity,
};
pub(crate) use cache::{
    lookup_bound, publish_committed, publishable, CheckedPendingCoefficients,
    CoefficientGroupReuse, CompletedTransformerKey, OrientedBasisOrder, PendingCoefficientGroups,
    TraceColumnReuse, TransformerMode, TreeTransformPlanning, TreeTransformScope,
};
pub use operation::{
    TreeTransformOperation, TreeTransformOperationKind, TreeTransformRuleCacheKey,
};
pub use plan::{
    build_all_codomain_tree_transform_group_plan, build_tree_pair_transform_group_plan,
    CheckedGenericPlanError, TreeTransformBlockSpec, TreeTransformGroupBlockSpec,
    TreeTransformGroupPlan, TreeTransformKeyBlockSpec,
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
#[cfg(test)]
pub(crate) use plan::{
    build_checked_generic_tree_pair_transform_group_plan,
    build_oriented_tree_pair_transform_group_plan_capability_validated,
};
pub(crate) use plan::{
    build_checked_generic_tree_pair_transform_group_plan_validated,
    compile_multiplicity_free_tree_pair_structure,
    validate_checked_generic_tree_pair_plan_preflight,
};
#[cfg(test)]
pub use plan::{build_generic_tree_pair_transform_group_plan, build_tree_transform_group_plan};
#[cfg(any(test, feature = "testing"))]
pub(crate) use plan::{
    build_generic_tree_pair_transform_group_plan_validated, validate_generic_tree_pair_preflight,
};
