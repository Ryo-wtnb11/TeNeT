use core::ops::{Add, Mul};
#[cfg(test)]
use rustc_hash::FxHashMap;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    BlockStructure, FusionTreeBlockGroup, FusionTreeGroupKey, FusionTreePairKey,
    GenericBraidScalar, GenericRigidSymbols, LocallyValidatedFusionTreeBlockStructure,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, TensorMap, TensorStorage,
};

use crate::cache::{
    local_lru, local_lru_capacity, OperationCachePolicy, TreeTransformStructureCacheKey,
};
use crate::{OperationError, TreeTransformStructure, TreeTransformStructureCache};

use super::helpers::{duplicate_fusion_tree_pair_index, fusion_tree_group_block_keys};
use super::operation::{TreeTransformOperation, TreeTransformRuleCacheKey};
#[cfg(test)]
use super::plan::TreeTransformGroupBlockSpec;
use super::plan::{
    build_all_codomain_tree_transform_group_plan_validated,
    build_all_codomain_tree_transform_group_plan_validated_with_threads,
    build_generic_tree_pair_transform_group_plan_validated,
    build_tree_pair_transform_group_plan_validated,
    build_tree_pair_transform_group_plan_validated_with_threads,
    validate_all_codomain_namespace_before_cache, validate_generic_tree_pair_preflight,
    validate_multiplicity_free_all_codomain_preflight_after_capability,
    validate_multiplicity_free_tree_pair_preflight,
    validate_multiplicity_free_tree_pair_preflight_after_capability,
    validate_multiplicity_free_tree_transform_capability,
    validate_tree_pair_namespace_before_cache, LocallyValidatedAllCodomainFusionTreeBlockStructure,
    TreeTransformGroupPlan,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformPlanScope {
    AllCodomain,
    TreePair,
}

// Why not copy TensorKit's 10^4-entry global cache: TeNeT's explicit execution
// contexts have shorter owner lifetimes and retain larger compiled artifacts.
// Bound plans and structures at that owner without changing the separate
// lifecycle contracts of contraction and derived-space caches.
pub(crate) const DEFAULT_TREE_TRANSFORM_CACHE_ENTRIES: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformSectorPlanKey<RuleKey> {
    rule: RuleKey,
    scope: TreeTransformPlanScope,
    operation: TreeTransformOperation,
    source_groups: Vec<TreeTransformSourceGroupKey>,
}

#[cfg(test)]
thread_local! {
    static TREE_TRANSFORM_SECTOR_PLAN_KEY_CONSTRUCTIONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_tree_transform_sector_plan_key_constructions() {
    TREE_TRANSFORM_SECTOR_PLAN_KEY_CONSTRUCTIONS.set(0);
}

#[cfg(test)]
pub(crate) fn tree_transform_sector_plan_key_constructions() -> usize {
    TREE_TRANSFORM_SECTOR_PLAN_KEY_CONSTRUCTIONS.get()
}

impl<RuleKey> TreeTransformSectorPlanKey<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    pub fn tree_pair<R>(
        rule: &R,
        operation: TreeTransformOperation,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        Self::new(
            rule.tree_transform_rule_cache_key(),
            TreeTransformPlanScope::TreePair,
            operation,
            src_structure,
        )
    }

    pub fn all_codomain<R>(
        rule: &R,
        operation: TreeTransformOperation,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        Self::new(
            rule.tree_transform_rule_cache_key(),
            TreeTransformPlanScope::AllCodomain,
            operation,
            src_structure,
        )
    }

    fn new(
        rule: RuleKey,
        scope: TreeTransformPlanScope,
        operation: TreeTransformOperation,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        #[cfg(test)]
        TREE_TRANSFORM_SECTOR_PLAN_KEY_CONSTRUCTIONS
            .set(TREE_TRANSFORM_SECTOR_PLAN_KEY_CONSTRUCTIONS.get() + 1);
        let source_groups = src_structure
            .fusion_tree_group_slice()
            .into_iter()
            .map(|group| TreeTransformSourceGroupKey::from_group(src_structure, group))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            rule,
            scope,
            operation,
            source_groups,
        })
    }

    #[inline]
    pub fn rule(&self) -> &RuleKey {
        &self.rule
    }

    #[inline]
    pub fn scope(&self) -> TreeTransformPlanScope {
        self.scope
    }

    #[inline]
    pub fn operation(&self) -> &TreeTransformOperation {
        &self.operation
    }

    #[inline]
    pub fn source_groups(&self) -> &[TreeTransformSourceGroupKey] {
        &self.source_groups
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformSourceGroupKey {
    group_key: FusionTreeGroupKey,
    src_keys: Vec<FusionTreePairKey>,
}

impl TreeTransformSourceGroupKey {
    fn from_group(
        structure: &BlockStructure,
        group: &FusionTreeBlockGroup,
    ) -> Result<Self, OperationError> {
        let src_keys = fusion_tree_group_block_keys(structure, group, "src")?;
        let Some(first) = src_keys.first() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        if let Some(index) = duplicate_fusion_tree_pair_index(&src_keys) {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "src",
                index,
            });
        }
        let group_key = first.group_key();
        Ok(Self {
            group_key,
            src_keys,
        })
    }

    #[inline]
    pub fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }

    #[inline]
    pub fn src_keys(&self) -> &[FusionTreePairKey] {
        &self.src_keys
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformGroupPlanKey {
    operation: TreeTransformOperation,
    groups: Vec<TreeTransformCachedGroupKey>,
}

#[cfg(test)]
impl TreeTransformGroupPlanKey {
    pub fn new<Groups>(operation: TreeTransformOperation, groups: Groups) -> Self
    where
        Groups: IntoIterator<Item = TreeTransformCachedGroupKey>,
    {
        Self {
            operation,
            groups: groups.into_iter().collect(),
        }
    }

    pub fn from_plan<T>(
        operation: TreeTransformOperation,
        plan: &TreeTransformGroupPlan<T>,
    ) -> Self {
        Self::new(
            operation,
            plan.specs()
                .iter()
                .map(TreeTransformCachedGroupKey::from_spec),
        )
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformCachedGroupKey {
    group_key: FusionTreeGroupKey,
    dst_keys: Vec<FusionTreePairKey>,
    src_keys: Vec<FusionTreePairKey>,
}

#[cfg(test)]
impl TreeTransformCachedGroupKey {
    pub fn from_spec<T>(spec: &TreeTransformGroupBlockSpec<T>) -> Self {
        Self {
            group_key: spec.group_key().clone(),
            dst_keys: spec.dst_keys().to_vec(),
            src_keys: spec.src_keys().to_vec(),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct TreeTransformGroupPlanCache<T> {
    plans: FxHashMap<TreeTransformGroupPlanKey, TreeTransformGroupPlan<T>>,
}

#[cfg(test)]
impl<T> Default for TreeTransformGroupPlanCache<T> {
    fn default() -> Self {
        Self {
            plans: FxHashMap::default(),
        }
    }
}

#[cfg(test)]
impl<T> TreeTransformGroupPlanCache<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn get(&self, key: &TreeTransformGroupPlanKey) -> Option<&TreeTransformGroupPlan<T>> {
        self.plans.get(key)
    }

    pub fn insert(
        &mut self,
        key: TreeTransformGroupPlanKey,
        plan: TreeTransformGroupPlan<T>,
    ) -> Option<TreeTransformGroupPlan<T>> {
        self.plans.insert(key, plan)
    }
}

pub struct TreeTransformCache<T, RuleKey> {
    plans: lru::LruCache<
        TreeTransformSectorPlanKey<RuleKey>,
        Arc<TreeTransformGroupPlan<T>>,
        rustc_hash::FxBuildHasher,
    >,
    structures: TreeTransformStructureCache<T, TreeTransformSectorPlanKey<RuleKey>>,
    last_structure: Option<TreeTransformLastStructure<T, RuleKey>>,
    policy: OperationCachePolicy,
    stats: TreeTransformCacheStats,
    // Why not retain source-column rows beside plans: an exact plan hit already
    // owns the complete transform, while partial-row replay repeated group
    // lowering and was slower than rebuilding the ordered whole block. This
    // keeps the persistent boundary at the TensorKit-style complete transform.
    // Why not expose a second compile knob: the execution context propagates
    // the backend's `recoupling_threads` to whole-group transform + assembly,
    // keeping one setting for replay and compile.
    recoupling_threads: usize,
}

impl<T, RuleKey> Clone for TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    fn clone(&self) -> Self {
        let mut plans = local_lru(self.policy);
        for (key, plan) in self.plans.iter().rev() {
            plans.put(key.clone(), Arc::clone(plan));
        }
        Self {
            plans,
            structures: self.structures.clone(),
            last_structure: self
                .last_structure
                .as_ref()
                .map(|last| TreeTransformLastStructure {
                    rule: last.rule.clone(),
                    scope: last.scope,
                    operation: last.operation.clone(),
                    dst_ptr: last.dst_ptr,
                    src_ptr: last.src_ptr,
                    dst_content_id: last.dst_content_id,
                    src_content_id: last.src_content_id,
                    storage_conjugate: last.storage_conjugate,
                    structure: Arc::clone(&last.structure),
                }),
            policy: self.policy,
            stats: self.stats,
            recoupling_threads: self.recoupling_threads,
        }
    }
}

impl<T, RuleKey> fmt::Debug for TreeTransformCache<T, RuleKey> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TreeTransformCache")
            .field("policy", &self.policy)
            .field("stats", &self.stats)
            .field("recoupling_threads", &self.recoupling_threads)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_group_key_rejects_repeated_public_group_indices() {
        let key = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap();
        let structure =
            crate::tests::packed_fixture_structure(2, [(key.clone(), vec![1, 1])]).unwrap();
        let repeated = FusionTreeBlockGroup::new(key.group_key(), vec![0, 0]);

        let err = TreeTransformSourceGroupKey::from_group(&structure, &repeated).unwrap_err();

        // What: a caller-built group cannot create a non-canonical cache key by
        // repeating one valid source basis index.
        assert_eq!(
            err,
            OperationError::DuplicateTreeTransformKey {
                tensor: "src",
                index: 1,
            }
        );
    }
}

pub type TreePairTransformCache<T, RuleKey> = TreeTransformCache<T, RuleKey>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TreeTransformCacheStats {
    plan_hits: usize,
    plan_misses: usize,
    structure_hits: usize,
    structure_misses: usize,
}

impl TreeTransformCacheStats {
    #[inline]
    pub fn plan_hits(self) -> usize {
        self.plan_hits
    }

    /// Always zero: retained source-row memoization was removed.
    ///
    /// Exact transform reuse is reported by [`Self::plan_hits`].
    #[deprecated(
        since = "0.1.0",
        note = "source rows are no longer cached; use plan_hits for exact transform reuse"
    )]
    #[inline]
    pub fn tree_row_hits(self) -> usize {
        0
    }

    /// Always zero: retained source-row memoization was removed.
    #[deprecated(
        since = "0.1.0",
        note = "source rows are no longer cached; use plan_misses for transform compilation"
    )]
    #[inline]
    pub fn tree_row_misses(self) -> usize {
        0
    }

    #[inline]
    pub fn plan_misses(self) -> usize {
        self.plan_misses
    }

    #[inline]
    pub fn structure_hits(self) -> usize {
        self.structure_hits
    }

    #[inline]
    pub fn structure_misses(self) -> usize {
        self.structure_misses
    }
}

#[derive(Clone, Debug)]
struct TreeTransformLastStructure<T, RuleKey> {
    rule: RuleKey,
    scope: TreeTransformPlanScope,
    operation: TreeTransformOperation,
    // `dst_ptr`/`src_ptr` are raw-pointer keys, sound only because the payload
    // `structure: Arc<TreeTransformStructure<T>>` transitively pins the dst/src
    // structures it was built from (it owns `dst_structure`/`src_structure`
    // Arcs — see transform_structure.rs). While this entry lives, those
    // addresses cannot be recycled, so a pointer match is a true identity match.
    // This safety depends on that payload pinning: if `TreeTransformStructure`
    // ever stopped holding those Arcs, these keys would become unsound (ABA).
    dst_ptr: usize,
    src_ptr: usize,
    dst_content_id: usize,
    src_content_id: usize,
    storage_conjugate: bool,
    structure: Arc<TreeTransformStructure<T>>,
}

/// Defaults to context-local LRUs bounded independently at 256 plan entries
/// and 256 compiled-structure entries. These are entry bounds, not byte bounds.
/// Use [`Self::with_policy`] or
/// [`TreeTransformExecutionContext::set_cache_policy`](crate::TreeTransformExecutionContext::set_cache_policy)
/// to select no retention, unbounded context-local retention, or another cap.
impl<T, RuleKey> Default for TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    fn default() -> Self {
        let policy = OperationCachePolicy::task_local_lru(DEFAULT_TREE_TRANSFORM_CACHE_ENTRIES);
        Self {
            plans: local_lru(policy),
            structures: TreeTransformStructureCache::with_policy(policy),
            last_structure: None,
            policy,
            stats: TreeTransformCacheStats::default(),
            recoupling_threads: 1,
        }
    }
}

impl<T, RuleKey> TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    /// Creates the bounded context-local cache described by [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a context-local cache with an explicit retention policy.
    pub fn with_policy(policy: OperationCachePolicy) -> Self {
        Self {
            plans: local_lru(policy),
            structures: TreeTransformStructureCache::with_policy(policy),
            last_structure: None,
            policy,
            stats: TreeTransformCacheStats::default(),
            recoupling_threads: 1,
        }
    }

    #[inline]
    pub fn recoupling_threads(&self) -> usize {
        self.recoupling_threads
    }

    /// Plan-compile worker count; the execution context keeps this in sync
    /// with the backend's `recoupling_threads`, so the one configured knob
    /// drives both replay and compile parallelism. Serial and parallel
    /// compilation use the same staged group algorithm.
    pub fn set_recoupling_threads(&mut self, threads: usize) {
        self.recoupling_threads = threads.max(1);
    }

    #[inline]
    pub fn policy(&self) -> OperationCachePolicy {
        self.policy
    }

    pub fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.structures.set_policy(policy);
        self.last_structure = None;
        if !policy.stores_entries() {
            self.plans.clear();
        }
        self.plans.resize(local_lru_capacity(policy));
    }

    #[inline]
    pub fn plan_len(&self) -> usize {
        self.plans.len()
    }

    #[inline]
    pub fn structure_len(&self) -> usize {
        self.structures.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty() && self.structures.is_empty()
    }

    #[inline]
    pub fn stats(&self) -> TreeTransformCacheStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = TreeTransformCacheStats::default();
    }

    fn fast_structure(
        &mut self,
        rule_key: &RuleKey,
        scope: TreeTransformPlanScope,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Option<Arc<TreeTransformStructure<T>>> {
        if !self.policy.stores_entries() {
            return None;
        }
        let last = self.last_structure.as_ref()?;
        if &last.rule == rule_key
            && last.scope == scope
            && &last.operation == operation
            && ((last.dst_ptr == Arc::as_ptr(dst_structure) as usize
                && last.src_ptr == Arc::as_ptr(src_structure) as usize)
                || (last.dst_content_id == dst_structure.content_id()
                    && last.src_content_id == src_structure.content_id()))
            && last.storage_conjugate == storage_conjugate
        {
            let structure = Arc::clone(&last.structure);
            self.stats.plan_hits += 1;
            self.stats.structure_hits += 1;
            // Why not promote through both LRUs: every ordinary path publishes
            // the entry it just promoted, so this front is already MRU. The
            // prelowered path clears the front before touching either owner.
            Some(structure)
        } else {
            None
        }
    }

    fn remember_structure(
        &mut self,
        rule: RuleKey,
        scope: TreeTransformPlanScope,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
        structure: Arc<TreeTransformStructure<T>>,
    ) {
        self.last_structure = Some(TreeTransformLastStructure {
            rule,
            scope,
            operation,
            dst_ptr: Arc::as_ptr(dst_structure) as usize,
            src_ptr: Arc::as_ptr(src_structure) as usize,
            dst_content_id: dst_structure.content_id(),
            src_content_id: src_structure.content_id(),
            storage_conjugate,
            structure,
        });
    }

    fn begin_lru_activity(&mut self) {
        // Why not retain the exact front across deep-cache activity: a later
        // structure compile can fail after promoting only its plan, leaving
        // the published front out of sync with LRU recency.
        self.last_structure = None;
    }

    fn touch_plan(&mut self, key: &TreeTransformSectorPlanKey<RuleKey>) {
        let _ = self.plans.get(key);
    }

    fn insert_plan_arc(
        &mut self,
        key: TreeTransformSectorPlanKey<RuleKey>,
        plan: Arc<TreeTransformGroupPlan<T>>,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.plans.put(key, plan);
    }

    fn compile_tree_pair_plan<R>(
        &mut self,
        source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
        operation: TreeTransformOperation,
    ) -> Result<Arc<TreeTransformGroupPlan<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T>,
        T: 'static + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    {
        build_tree_pair_transform_group_plan_validated_with_threads(
            source_proof,
            operation,
            self.recoupling_threads,
        )
        .map(Arc::new)
    }

    fn compile_all_codomain_plan<R>(
        &mut self,
        source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
        operation: TreeTransformOperation,
    ) -> Result<Arc<TreeTransformGroupPlan<T>>, OperationError>
    where
        R: MultiplicityFreeFusionSymbols<Scalar = T> + Sync,
        T: 'static + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    {
        build_all_codomain_tree_transform_group_plan_validated_with_threads(
            source_proof,
            operation,
            self.recoupling_threads,
        )
        .map(Arc::new)
    }

    /// Resolve an exact tree-pair replay structure.
    ///
    /// Raw block keys follow
    /// [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain precondition.
    pub fn get_or_compile_tree_pair<
        R,
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        validate_multiplicity_free_tree_transform_capability(rule, &operation)?;
        validate_tree_pair_namespace_before_cache(&operation, src.structure())?;
        let rule_key = rule.tree_transform_rule_cache_key();
        if let Some(structure) = self.fast_structure(
            &rule_key,
            TreeTransformPlanScope::TreePair,
            &operation,
            dst.structure(),
            src.structure(),
            false,
        ) {
            return Ok(structure);
        }
        let source_proof = validate_multiplicity_free_tree_pair_preflight_after_capability(
            rule,
            &operation,
            src.structure(),
        )?;
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
            .map_err(OperationError::from_core_preserving_context)?;
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan =
                build_tree_pair_transform_group_plan_validated(&source_proof, operation.clone())?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::TreePair,
            operation.clone(),
            src.structure(),
        )?;
        self.begin_lru_activity();
        if self.plans.contains(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.compile_tree_pair_plan(&source_proof, operation.clone())?;
            self.insert_plan_arc(plan_key.clone(), plan);
        }
        self.get_or_compile_structure(
            rule_key,
            TreeTransformPlanScope::TreePair,
            operation,
            plan_key,
            dst,
            src,
        )
    }

    /// Structure-only variant of [`Self::get_or_compile_tree_pair`], with the
    /// same provider-domain precondition.
    pub fn get_or_compile_tree_pair_structures_with_storage_conjugation<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
    {
        self.get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            rule,
            &operation,
            dst_structure,
            src_structure,
            storage_conjugate,
        )
    }

    /// Borrowed-operation variant of
    /// [`Self::get_or_compile_tree_pair_structures_with_storage_conjugation`],
    /// with the same provider-domain precondition.
    pub fn get_or_compile_tree_pair_structures_with_storage_conjugation_ref<R>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
    {
        validate_multiplicity_free_tree_transform_capability(rule, operation)?;
        validate_tree_pair_namespace_before_cache(operation, src_structure)?;
        let rule_key = rule.tree_transform_rule_cache_key();
        if let Some(structure) = self.fast_structure(
            &rule_key,
            TreeTransformPlanScope::TreePair,
            operation,
            dst_structure,
            src_structure,
            storage_conjugate,
        ) {
            return Ok(structure);
        }
        let source_proof = validate_multiplicity_free_tree_pair_preflight_after_capability(
            rule,
            operation,
            src_structure,
        )?;
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
            .map_err(OperationError::from_core_preserving_context)?;
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan =
                build_tree_pair_transform_group_plan_validated(&source_proof, operation.clone())?;
            return Ok(Arc::new(
                plan.compile_shared_structures_with_storage_conjugation(
                    Arc::clone(dst_structure),
                    Arc::clone(src_structure),
                    storage_conjugate,
                )?,
            ));
        }
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::TreePair,
            operation.clone(),
            src_structure,
        )?;
        self.begin_lru_activity();
        if self.plans.contains(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.compile_tree_pair_plan(&source_proof, operation.clone())?;
            self.insert_plan_arc(plan_key.clone(), plan);
        }
        self.get_or_compile_structure_from_structures_with_storage_conjugation(
            rule_key,
            TreeTransformPlanScope::TreePair,
            operation.clone(),
            plan_key,
            dst_structure,
            src_structure,
            storage_conjugate,
        )
    }

    pub(crate) fn get_or_compile_tree_pair_prelowered<R, FBlock, FAxis>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        logical_src_structure: &Arc<BlockStructure>,
        storage_src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
        logical_to_storage_block: FBlock,
        logical_to_storage_axis: FAxis,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        FBlock: Fn(usize) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        let source_proof =
            validate_multiplicity_free_tree_pair_preflight(rule, operation, logical_src_structure)?;
        let logical_source_id = logical_src_structure.content_id();
        let storage_source_id = storage_src_structure.content_id();
        let destination_id = dst_structure.content_id();
        if storage_source_id != logical_source_id {
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, storage_src_structure)
                .map_err(OperationError::from_core_preserving_context)?;
        }
        if destination_id != logical_source_id && destination_id != storage_source_id {
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
                .map_err(OperationError::from_core_preserving_context)?;
        }
        let rule_key = rule.tree_transform_rule_cache_key();
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan =
                build_tree_pair_transform_group_plan_validated(&source_proof, operation.clone())?;
            return Ok(Arc::new(
                plan.compile_shared_structures_with_storage_mapping(
                    Arc::clone(dst_structure),
                    logical_src_structure,
                    Arc::clone(storage_src_structure),
                    logical_to_storage_block,
                    logical_to_storage_axis,
                    storage_conjugate,
                )?,
            ));
        }
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::TreePair,
            operation.clone(),
            logical_src_structure,
        )?;
        self.begin_lru_activity();
        if self.plans.contains(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.compile_tree_pair_plan(&source_proof, operation.clone())?;
            self.insert_plan_arc(plan_key.clone(), plan);
        }
        let structure_key =
            TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
                plan_key.clone(),
                dst_structure,
                storage_src_structure,
                storage_conjugate,
            )?;
        if let Some(structure) = self.structures.get_arc(&structure_key) {
            self.stats.structure_hits += 1;
            self.structures.touch(&structure_key);
            return Ok(structure);
        }
        self.stats.structure_misses += 1;
        let plan = self
            .plans
            .get(&plan_key)
            .expect("tree transform plan inserted before prelowered structure compile");
        let structure = Arc::new(plan.compile_shared_structures_with_storage_mapping(
            Arc::clone(dst_structure),
            logical_src_structure,
            Arc::clone(storage_src_structure),
            logical_to_storage_block,
            logical_to_storage_axis,
            storage_conjugate,
        )?);
        self.structures
            .insert_arc(structure_key, Arc::clone(&structure));
        Ok(structure)
    }

    /// Generic-fusion (outer-multiplicity, e.g. SU(3)) sibling of
    /// [`Self::get_or_compile_tree_pair`].
    ///
    /// ponytail: NON-MEMOIZED to start — it rebuilds and compiles the generic
    /// plan on every call. Generic (SU(3)) recoupling is not yet on any hot
    /// path, and correctness-before-perf is the Stage B rule. The
    /// multiplicity-free sibling caches a complete context-owned plan and uses
    /// compile-local ordered whole-block lowering on misses; extending that
    /// boundary is deferred until a real Generic workload measures the
    /// recompile cost (the B3c / perf handoff). The
    /// `TreeTransformRuleCacheKey` bound is still required: the Su3 `Key` embeds
    /// the table's provenance hash, so once complete-plan caching lands a
    /// swapped table can never reuse another table's plans.
    ///
    /// Raw block keys follow
    /// [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain precondition.
    pub fn get_or_compile_tree_pair_generic<
        R,
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: GenericRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        R::Scalar: GenericBraidScalar,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        let source_proof = validate_generic_tree_pair_preflight(rule, &operation, src.structure())?;
        let _destination_proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
                .map_err(OperationError::from_core_preserving_context)?;
        self.stats.plan_misses += 1;
        self.stats.structure_misses += 1;
        let plan =
            build_generic_tree_pair_transform_group_plan_validated(&source_proof, operation)?;
        Ok(Arc::new(plan.compile(dst, src)?))
    }

    /// Structure-only generic sibling for the dynamic-rank (raw-slice) path —
    /// the top-level `tenet::Tensor` SU(3) `permute`/`braid`/`transpose` route.
    /// Same non-memoized rationale as [`Self::get_or_compile_tree_pair_generic`].
    /// It has the same provider-domain precondition.
    pub fn get_or_compile_tree_pair_structures_generic<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: GenericRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        R::Scalar: GenericBraidScalar,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
    {
        let source_proof = validate_generic_tree_pair_preflight(rule, &operation, src_structure)?;
        let _destination_proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
                .map_err(OperationError::from_core_preserving_context)?;
        self.stats.plan_misses += 1;
        self.stats.structure_misses += 1;
        let plan =
            build_generic_tree_pair_transform_group_plan_validated(&source_proof, operation)?;
        Ok(Arc::new(
            plan.compile_shared_structures_with_storage_conjugation(
                Arc::clone(dst_structure),
                Arc::clone(src_structure),
                false,
            )?,
        ))
    }

    /// Resolve an exact all-codomain replay structure.
    ///
    /// Raw block keys follow
    /// [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain precondition.
    pub fn get_or_compile_all_codomain<
        R,
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeFusionSymbols<Scalar = T>
            + TreeTransformRuleCacheKey<Key = RuleKey>
            + Sync,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        validate_multiplicity_free_tree_transform_capability(rule, &operation)?;
        validate_all_codomain_namespace_before_cache(&operation, src.structure())?;
        let rule_key = rule.tree_transform_rule_cache_key();
        if let Some(structure) = self.fast_structure(
            &rule_key,
            TreeTransformPlanScope::AllCodomain,
            &operation,
            dst.structure(),
            src.structure(),
            false,
        ) {
            return Ok(structure);
        }
        let source_proof = validate_multiplicity_free_all_codomain_preflight_after_capability(
            rule,
            &operation,
            src.structure(),
        )?;
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
            .map_err(OperationError::from_core_preserving_context)?;
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan = build_all_codomain_tree_transform_group_plan_validated(
                &source_proof,
                operation.clone(),
            )?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::AllCodomain,
            operation.clone(),
            src.structure(),
        )?;
        self.begin_lru_activity();
        if self.plans.contains(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.compile_all_codomain_plan(&source_proof, operation.clone())?;
            self.insert_plan_arc(plan_key.clone(), plan);
        }
        self.get_or_compile_structure(
            rule_key,
            TreeTransformPlanScope::AllCodomain,
            operation,
            plan_key,
            dst,
            src,
        )
    }

    fn get_or_compile_structure<
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule_key: RuleKey,
        scope: TreeTransformPlanScope,
        operation: TreeTransformOperation,
        plan_key: TreeTransformSectorPlanKey<RuleKey>,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        T: 'static + Copy + Send + Sync,
        RuleKey: 'static + Send + Sync,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        let structure_key = TreeTransformStructureCacheKey::from_structures(
            plan_key.clone(),
            dst.structure(),
            src.structure(),
        )?;
        if self.structures.get(&structure_key).is_some() {
            self.stats.structure_hits += 1;
            self.structures.touch(&structure_key);
        } else {
            self.stats.structure_misses += 1;
            let plan = self
                .plans
                .get(&plan_key)
                .expect("tree transform plan inserted before structure compile");
            let structure = Arc::new(plan.compile(dst, src)?);
            self.structures.insert_arc(structure_key.clone(), structure);
        }
        let structure = self
            .structures
            .get_arc(&structure_key)
            .expect("tree transform structure inserted before return");
        self.remember_structure(
            rule_key,
            scope,
            operation,
            dst.structure(),
            src.structure(),
            false,
            Arc::clone(&structure),
        );
        Ok(structure)
    }

    fn get_or_compile_structure_from_structures_with_storage_conjugation(
        &mut self,
        rule_key: RuleKey,
        scope: TreeTransformPlanScope,
        operation: TreeTransformOperation,
        plan_key: TreeTransformSectorPlanKey<RuleKey>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        T: 'static + Copy + Send + Sync,
        RuleKey: 'static + Send + Sync,
    {
        let structure_key = TreeTransformStructureCacheKey::from_structures(
            plan_key.clone(),
            dst_structure,
            src_structure,
        )?;
        let structure_key = if storage_conjugate {
            TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
                plan_key.clone(),
                dst_structure,
                src_structure,
                true,
            )?
        } else {
            structure_key
        };
        if self.structures.get(&structure_key).is_some() {
            self.stats.structure_hits += 1;
            self.structures.touch(&structure_key);
        } else {
            self.stats.structure_misses += 1;
            let plan = self
                .plans
                .get(&plan_key)
                .expect("tree transform plan inserted before structure compile");
            let structure = Arc::new(plan.compile_shared_structures_with_storage_conjugation(
                Arc::clone(dst_structure),
                Arc::clone(src_structure),
                storage_conjugate,
            )?);
            self.structures.insert_arc(structure_key.clone(), structure);
        }
        let structure = self
            .structures
            .get_arc(&structure_key)
            .expect("tree transform structure inserted before return");
        self.remember_structure(
            rule_key,
            scope,
            operation,
            dst_structure,
            src_structure,
            storage_conjugate,
            Arc::clone(&structure),
        );
        Ok(structure)
    }
}
