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
    TreeTransformRuleCacheKey,
};
pub use plan::{
    build_all_codomain_tree_transform_group_plan, build_tree_pair_transform_group_plan,
    build_tree_transform_group_plan, TreeTransformBlockSpec, TreeTransformGroupBlockSpec,
    TreeTransformGroupPlan, TreeTransformKeyBlockSpec,
};

#[cfg(test)]
pub(crate) use cache::{TreeTransformGroupPlanCache, TreeTransformGroupPlanKey};
#[cfg(test)]
pub(crate) use plan::{
    build_unique_all_codomain_tree_transform_group_plan,
    build_unique_tree_pair_transform_group_plan, build_unique_tree_transform_group_plan,
};
