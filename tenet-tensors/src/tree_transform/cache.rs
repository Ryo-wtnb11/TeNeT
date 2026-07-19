use core::ops::{Add, Mul};
use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    BlockStructure, FusionTreeBlockGroup, FusionTreeGroupKey, FusionTreePairKey,
    GenericBraidScalar, GenericRigidSymbols, LocallyValidatedFusionTreeBlockStructure,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, TensorMap, TensorStorage,
};

use crate::cache::{
    rebuild_lru_order_from_keys, touch_lru_key, OperationCachePolicy,
    TreeTransformStructureCacheKey,
};
use crate::{OperationError, TreeTransformStructure, TreeTransformStructureCache};

use super::helpers::{duplicate_fusion_tree_pair_index, fusion_tree_group_block_keys};
use super::operation::{TreeTransformOperation, TreeTransformRuleCacheKey};
#[cfg(test)]
use super::plan::TreeTransformGroupBlockSpec;
use super::plan::{
    build_all_codomain_tree_transform_group_plan_validated,
    build_generic_tree_pair_transform_group_plan_validated,
    build_tree_pair_transform_group_plan_validated, validate_all_codomain_namespace_before_cache,
    validate_generic_tree_pair_preflight,
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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformSectorPlanKey<RuleKey> {
    rule: RuleKey,
    scope: TreeTransformPlanScope,
    operation: TreeTransformOperation,
    source_groups: Vec<TreeTransformSourceGroupKey>,
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
        let source_groups = src_structure
            .fusion_tree_groups()
            .into_iter()
            .map(|group| TreeTransformSourceGroupKey::from_group(src_structure, &group))
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

#[derive(Clone, Debug)]
pub struct TreeTransformCache<T, RuleKey> {
    plans: FxHashMap<TreeTransformSectorPlanKey<RuleKey>, Arc<TreeTransformGroupPlan<T>>>,
    plan_lru_order: VecDeque<TreeTransformSectorPlanKey<RuleKey>>,
    structures: TreeTransformStructureCache<T, TreeTransformSectorPlanKey<RuleKey>>,
    last_structure: Option<TreeTransformLastStructure<T, RuleKey>>,
    fast_structures:
        FxHashMap<TreeTransformFastStructureKey<RuleKey>, Arc<TreeTransformStructure<T>>>,
    policy: OperationCachePolicy,
    stats: TreeTransformCacheStats,
    // Why not store recoupling rows inside each plan: context-local rows are
    // independent of degeneracy shape and remain reusable across plan-key misses.
    tree_rows: crate::tree_transform::plan::TreePairRowMemo<T, RuleKey>,
    // Why not key all-codomain rows by a full tree pair: this scope leaves the
    // domain unchanged, so the codomain tree contains every deciding input.
    all_codomain_rows: crate::tree_transform::plan::AllCodomainRowMemo<T, RuleKey>,
    // Why not expose a second compile knob: the execution context propagates
    // the backend's `recoupling_threads` to whole-group transform + assembly,
    // keeping one setting for replay and compile.
    recoupling_threads: usize,
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
    tree_row_hits: usize,
    tree_row_misses: usize,
}

impl TreeTransformCacheStats {
    #[inline]
    pub fn plan_hits(self) -> usize {
        self.plan_hits
    }

    /// Shape-independent context-local recoupling-row memo hits.
    #[inline]
    pub fn tree_row_hits(self) -> usize {
        self.tree_row_hits
    }

    #[inline]
    pub fn tree_row_misses(self) -> usize {
        self.tree_row_misses
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
    plan_key: TreeTransformSectorPlanKey<RuleKey>,
    structure_key: TreeTransformStructureCacheKey<TreeTransformSectorPlanKey<RuleKey>>,
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
    storage_conjugate: bool,
    structure: Arc<TreeTransformStructure<T>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TreeTransformFastStructureKey<RuleKey> {
    rule: RuleKey,
    scope: TreeTransformPlanScope,
    operation: TreeTransformOperation,
    dst_structure_id: usize,
    src_structure_id: usize,
    storage_conjugate: bool,
}

impl<T, RuleKey> Default for TreeTransformCache<T, RuleKey> {
    fn default() -> Self {
        Self {
            plans: FxHashMap::default(),
            plan_lru_order: VecDeque::new(),
            structures: TreeTransformStructureCache::default(),
            last_structure: None,
            fast_structures: FxHashMap::default(),
            policy: OperationCachePolicy::default(),
            stats: TreeTransformCacheStats::default(),
            tree_rows: crate::tree_transform::plan::TreePairRowMemo::default(),
            all_codomain_rows: crate::tree_transform::plan::AllCodomainRowMemo::default(),
            recoupling_threads: 1,
        }
    }
}

impl<T, RuleKey> TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_policy(policy: OperationCachePolicy) -> Self {
        Self {
            plans: FxHashMap::default(),
            plan_lru_order: VecDeque::new(),
            structures: TreeTransformStructureCache::with_policy(policy),
            last_structure: None,
            fast_structures: FxHashMap::default(),
            policy,
            stats: TreeTransformCacheStats::default(),
            tree_rows: crate::tree_transform::plan::TreePairRowMemo::default(),
            all_codomain_rows: crate::tree_transform::plan::AllCodomainRowMemo::default(),
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
        self.fast_structures.clear();
        if !policy.stores_entries() {
            self.plans.clear();
            self.plan_lru_order.clear();
            self.tree_rows.clear();
            self.all_codomain_rows.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            rebuild_lru_order_from_keys(&self.plans, &mut self.plan_lru_order);
            self.enforce_plan_lru_limit(max_entries);
        }
    }

    #[inline]
    pub fn plan_len(&self) -> usize {
        self.plans.len()
    }

    #[inline]
    pub fn structure_len(&self) -> usize {
        self.structures.len()
    }

    #[cfg(test)]
    pub(crate) fn tree_row_len(&self) -> usize {
        self.tree_rows.len()
    }

    #[cfg(test)]
    pub(crate) fn all_codomain_row_len(&self) -> usize {
        self.all_codomain_rows.len()
    }

    #[cfg(test)]
    pub(crate) fn fast_structure_len(&self) -> usize {
        self.fast_structures.len()
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
            && last.dst_ptr == Arc::as_ptr(dst_structure) as usize
            && last.src_ptr == Arc::as_ptr(src_structure) as usize
            && last.storage_conjugate == storage_conjugate
        {
            let structure = Arc::clone(&last.structure);
            self.stats.plan_hits += 1;
            self.stats.structure_hits += 1;
            if self.policy.max_entries().is_some() {
                let plan_key = last.plan_key.clone();
                let structure_key = last.structure_key.clone();
                self.touch_plan(&plan_key);
                self.structures.touch(&structure_key);
            }
            Some(structure)
        } else {
            if self.policy.max_entries().is_some() {
                return None;
            }
            let key = TreeTransformFastStructureKey {
                rule: rule_key.clone(),
                scope,
                operation: operation.clone(),
                dst_structure_id: dst_structure.content_id(),
                src_structure_id: src_structure.content_id(),
                storage_conjugate,
            };
            let structure = self.fast_structures.get(&key)?;
            self.stats.plan_hits += 1;
            self.stats.structure_hits += 1;
            Some(Arc::clone(structure))
        }
    }

    fn remember_structure(
        &mut self,
        plan_key: TreeTransformSectorPlanKey<RuleKey>,
        structure_key: TreeTransformStructureCacheKey<TreeTransformSectorPlanKey<RuleKey>>,
        rule: RuleKey,
        scope: TreeTransformPlanScope,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
        structure: Arc<TreeTransformStructure<T>>,
    ) {
        if self.policy.stores_entries() && self.policy.max_entries().is_none() {
            self.fast_structures.insert(
                TreeTransformFastStructureKey {
                    rule: rule.clone(),
                    scope,
                    operation: operation.clone(),
                    dst_structure_id: dst_structure.content_id(),
                    src_structure_id: src_structure.content_id(),
                    storage_conjugate,
                },
                Arc::clone(&structure),
            );
        }
        self.last_structure = Some(TreeTransformLastStructure {
            plan_key,
            structure_key,
            rule,
            scope,
            operation,
            dst_ptr: Arc::as_ptr(dst_structure) as usize,
            src_ptr: Arc::as_ptr(src_structure) as usize,
            storage_conjugate,
            structure,
        });
    }

    fn touch_plan(&mut self, key: &TreeTransformSectorPlanKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.plans.contains_key(key) {
            touch_lru_key(&mut self.plan_lru_order, key);
        }
    }

    fn insert_plan_arc(
        &mut self,
        key: TreeTransformSectorPlanKey<RuleKey>,
        plan: Arc<TreeTransformGroupPlan<T>>,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.plans.insert(key.clone(), plan);
        if self.policy.max_entries().is_some() {
            self.touch_plan(&key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_plan_lru_limit(max_entries);
        }
    }

    fn enforce_plan_lru_limit(&mut self, max_entries: usize) {
        while self.plans.len() > max_entries {
            let Some(oldest) = self.plan_lru_order.pop_front() else {
                break;
            };
            self.plans.remove(&oldest);
        }
    }

    fn compile_tree_pair_plan<R>(
        &mut self,
        source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
        rule_key: &RuleKey,
        operation: TreeTransformOperation,
    ) -> Result<Arc<TreeTransformGroupPlan<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T>,
        T: 'static + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    {
        let plan =
            crate::tree_transform::plan::build_multiplicity_free_tree_pair_transform_group_plan_memoized_validated(
            source_proof,
            rule_key,
            operation,
            &mut self.tree_rows,
            &mut self.stats.tree_row_hits,
            &mut self.stats.tree_row_misses,
            self.recoupling_threads,
        )?;
        Ok(Arc::new(plan))
    }

    fn compile_all_codomain_plan<R>(
        &mut self,
        source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
        rule_key: &RuleKey,
        operation: TreeTransformOperation,
    ) -> Result<Arc<TreeTransformGroupPlan<T>>, OperationError>
    where
        R: MultiplicityFreeFusionSymbols<Scalar = T> + Sync,
        T: 'static + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    {
        let plan =
            crate::tree_transform::plan::build_multiplicity_free_all_codomain_tree_transform_group_plan_memoized_validated(
            source_proof,
            rule_key,
            operation,
            &mut self.all_codomain_rows,
            &mut self.stats.tree_row_hits,
            &mut self.stats.tree_row_misses,
            self.recoupling_threads,
        )?;
        Ok(Arc::new(plan))
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
        if rule.fusion_style() == tenet_core::FusionStyleKind::Unique {
            let source_proof = validate_multiplicity_free_tree_pair_preflight_after_capability(
                rule,
                &operation,
                src.structure(),
            )?;
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
                .map_err(OperationError::from_core_preserving_context)?;
            // Why-not cache Unique plans: Unique removes fusion multiplicity,
            // but does not imply a single total destination. Retaining
            // plan/row state costs more than direct compilation here and
            // risks process-lifetime growth for cheap keys.
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan =
                build_tree_pair_transform_group_plan_validated(&source_proof, operation.clone())?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
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
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::TreePair,
            operation.clone(),
            src.structure(),
        )?;
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan =
                build_tree_pair_transform_group_plan_validated(&source_proof, operation.clone())?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.compile_tree_pair_plan(&source_proof, &rule_key, operation.clone())?;
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
        if rule.fusion_style() == tenet_core::FusionStyleKind::Unique {
            let source_proof = validate_multiplicity_free_tree_pair_preflight_after_capability(
                rule,
                operation,
                src_structure,
            )?;
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
                .map_err(OperationError::from_core_preserving_context)?;
            // Why-not cache Unique plans: Unique removes fusion multiplicity,
            // but does not imply a single total destination. The storage-only
            // form still has cheap, layout-specific entries, so retaining a
            // second structural cache would duplicate state and permit
            // unbounded entries.
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
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::TreePair,
            operation.clone(),
            src_structure,
        )?;
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
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.compile_tree_pair_plan(&source_proof, &rule_key, operation.clone())?;
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
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::TreePair,
            operation.clone(),
            logical_src_structure,
        )?;
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
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.compile_tree_pair_plan(&source_proof, &rule_key, operation.clone())?;
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
    /// path, and correctness-before-perf is the Stage B rule; the memoized
    /// builder (the generic analogue of `TreePairRowMemo` / the plan/row cache
    /// the mult-free sibling above uses) is deferred until a real Generic
    /// workload measures the recompile cost (the B3c / perf handoff). The
    /// `TreeTransformRuleCacheKey` bound is still required: the Su3 `Key` embeds
    /// the table's provenance hash, so once memoization lands a swapped table
    /// can never reuse another table's plans.
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
        if rule.fusion_style() == tenet_core::FusionStyleKind::Unique {
            let source_proof = validate_multiplicity_free_all_codomain_preflight_after_capability(
                rule,
                &operation,
                src.structure(),
            )?;
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
                .map_err(OperationError::from_core_preserving_context)?;
            // Why-not cache Unique all-codomain transforms: with no fusion
            // multiplicity this lowering is a direct tree relabeling.  A
            // process/context cache only retains layout descriptors for a
            // cheap, non-reusable key and defeats the eager Unique path used
            // by tree-pair transforms.
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan = build_all_codomain_tree_transform_group_plan_validated(
                &source_proof,
                operation.clone(),
            )?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
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
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::AllCodomain,
            operation.clone(),
            src.structure(),
        )?;
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan = build_all_codomain_tree_transform_group_plan_validated(
                &source_proof,
                operation.clone(),
            )?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan =
                self.compile_all_codomain_plan(&source_proof, &rule_key, operation.clone())?;
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
            plan_key,
            structure_key,
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
            plan_key,
            structure_key,
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
