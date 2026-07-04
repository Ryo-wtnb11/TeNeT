use core::ops::{Add, Mul};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    BlockKey, BlockStructure, FusionTreeBlockGroup, FusionTreeGroupKey,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, TensorMap, TensorStorage,
};

use crate::cache::{
    rebuild_lru_order_from_keys, touch_lru_key, OperationCachePolicy,
    TreeTransformStructureCacheKey,
};
use crate::{OperationError, TreeTransformStructure, TreeTransformStructureCache};

use super::helpers::fusion_tree_group_block_keys;
use super::operation::{TreeTransformOperation, TreeTransformRuleCacheKey};
#[cfg(test)]
use super::plan::TreeTransformGroupBlockSpec;
use super::plan::{
    build_all_codomain_tree_transform_group_plan, build_tree_pair_transform_group_plan,
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
    src_keys: Vec<BlockKey>,
}

impl TreeTransformSourceGroupKey {
    fn from_group(
        structure: &BlockStructure,
        group: &FusionTreeBlockGroup,
    ) -> Result<Self, OperationError> {
        Ok(Self {
            group_key: group.group_key().clone(),
            src_keys: fusion_tree_group_block_keys(structure, group, "src")?,
        })
    }

    #[inline]
    pub fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }

    #[inline]
    pub fn src_keys(&self) -> &[BlockKey] {
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
    dst_keys: Vec<BlockKey>,
    src_keys: Vec<BlockKey>,
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
    plans: HashMap<TreeTransformGroupPlanKey, TreeTransformGroupPlan<T>>,
}

#[cfg(test)]
impl<T> Default for TreeTransformGroupPlanCache<T> {
    fn default() -> Self {
        Self {
            plans: HashMap::new(),
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
    plans: HashMap<TreeTransformSectorPlanKey<RuleKey>, TreeTransformGroupPlan<T>>,
    plan_lru_order: VecDeque<TreeTransformSectorPlanKey<RuleKey>>,
    structures: TreeTransformStructureCache<T, TreeTransformSectorPlanKey<RuleKey>>,
    last_structure: Option<TreeTransformLastStructure<T, RuleKey>>,
    policy: OperationCachePolicy,
    stats: TreeTransformCacheStats,
    // Shape-independent recoupling rows per (rule, operation, source tree):
    // survives degeneracy changes, so chi sweeps recompile plans without
    // recomputing F/R-symbol contractions (TensorKit @cached fstranspose/fsbraid).
    tree_rows: crate::tree_transform::plan::TreePairRowMemo<T, RuleKey>,
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

    /// Shape-independent recoupling-row memo hits (TensorKit
    /// fstranspose/fsbraid @cached analog): rows reused across degeneracy changes.
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
    dst_ptr: *const BlockStructure,
    src_ptr: *const BlockStructure,
    storage_conjugate: bool,
    structure: Arc<TreeTransformStructure<T>>,
}

impl<T, RuleKey> Default for TreeTransformCache<T, RuleKey> {
    fn default() -> Self {
        Self {
            plans: HashMap::new(),
            plan_lru_order: VecDeque::new(),
            structures: TreeTransformStructureCache::default(),
            last_structure: None,
            policy: OperationCachePolicy::default(),
            stats: TreeTransformCacheStats::default(),
            tree_rows: crate::tree_transform::plan::TreePairRowMemo::default(),
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
            plans: HashMap::new(),
            plan_lru_order: VecDeque::new(),
            structures: TreeTransformStructureCache::with_policy(policy),
            last_structure: None,
            policy,
            stats: TreeTransformCacheStats::default(),
            tree_rows: crate::tree_transform::plan::TreePairRowMemo::default(),
        }
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
            self.plan_lru_order.clear();
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
            && last.dst_ptr == Arc::as_ptr(dst_structure)
            && last.src_ptr == Arc::as_ptr(src_structure)
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
            None
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
        self.last_structure = Some(TreeTransformLastStructure {
            plan_key,
            structure_key,
            rule,
            scope,
            operation,
            dst_ptr: Arc::as_ptr(dst_structure),
            src_ptr: Arc::as_ptr(src_structure),
            storage_conjugate,
            structure,
        });
    }

    fn touch_plan(&mut self, key: &TreeTransformSectorPlanKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.plans.contains_key(key) {
            touch_lru_key(&mut self.plan_lru_order, key);
        }
    }

    fn insert_plan(
        &mut self,
        key: TreeTransformSectorPlanKey<RuleKey>,
        plan: TreeTransformGroupPlan<T>,
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
        T: Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
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
                build_tree_pair_transform_group_plan(rule, operation.clone(), src.structure())?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan =
                crate::tree_transform::plan::build_multiplicity_free_tree_pair_transform_group_plan_memoized(
                rule,
                &rule_key,
                operation.clone(),
                src.structure(),
                &mut self.tree_rows,
                &mut self.stats.tree_row_hits,
                &mut self.stats.tree_row_misses,
            )?;
            self.insert_plan(plan_key.clone(), plan);
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

    pub fn get_or_compile_tree_pair_structures<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        if let Some(structure) = self.fast_structure(
            &rule_key,
            TreeTransformPlanScope::TreePair,
            &operation,
            dst_structure,
            src_structure,
            false,
        ) {
            return Ok(structure);
        }
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
                build_tree_pair_transform_group_plan(rule, operation.clone(), src_structure)?;
            return Ok(Arc::new(
                plan.compile_shared_structures_with_storage_conjugation(
                    Arc::clone(dst_structure),
                    Arc::clone(src_structure),
                    false,
                )?,
            ));
        }
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan =
                crate::tree_transform::plan::build_multiplicity_free_tree_pair_transform_group_plan_memoized(
                rule,
                &rule_key,
                operation.clone(),
                src_structure,
                &mut self.tree_rows,
                &mut self.stats.tree_row_hits,
                &mut self.stats.tree_row_misses,
            )?;
            self.insert_plan(plan_key.clone(), plan);
        }
        self.get_or_compile_structure_from_structures(
            rule_key,
            TreeTransformPlanScope::TreePair,
            operation,
            plan_key,
            dst_structure,
            src_structure,
        )
    }

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
        T: Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        if let Some(structure) = self.fast_structure(
            &rule_key,
            TreeTransformPlanScope::TreePair,
            &operation,
            dst_structure,
            src_structure,
            storage_conjugate,
        ) {
            return Ok(structure);
        }
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
                build_tree_pair_transform_group_plan(rule, operation.clone(), src_structure)?;
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
            let plan =
                crate::tree_transform::plan::build_multiplicity_free_tree_pair_transform_group_plan_memoized(
                rule,
                &rule_key,
                operation.clone(),
                src_structure,
                &mut self.tree_rows,
                &mut self.stats.tree_row_hits,
                &mut self.stats.tree_row_misses,
            )?;
            self.insert_plan(plan_key.clone(), plan);
        }
        self.get_or_compile_structure_from_structures_with_storage_conjugation(
            rule_key,
            TreeTransformPlanScope::TreePair,
            operation,
            plan_key,
            dst_structure,
            src_structure,
            storage_conjugate,
        )
    }

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
        R: MultiplicityFreeFusionSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
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
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::AllCodomain,
            operation.clone(),
            src.structure(),
        )?;
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan = build_all_codomain_tree_transform_group_plan(
                rule,
                operation.clone(),
                src.structure(),
            )?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = build_all_codomain_tree_transform_group_plan(
                rule,
                operation.clone(),
                src.structure(),
            )?;
            self.insert_plan(plan_key.clone(), plan);
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
        T: Copy,
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
            let structure = plan.compile(dst, src)?;
            self.structures.insert(structure_key.clone(), structure);
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

    fn get_or_compile_structure_from_structures(
        &mut self,
        rule_key: RuleKey,
        scope: TreeTransformPlanScope,
        operation: TreeTransformOperation,
        plan_key: TreeTransformSectorPlanKey<RuleKey>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        T: Copy,
    {
        self.get_or_compile_structure_from_structures_with_storage_conjugation(
            rule_key,
            scope,
            operation,
            plan_key,
            dst_structure,
            src_structure,
            false,
        )
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
        T: Copy,
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
            let structure = plan.compile_shared_structures_with_storage_conjugation(
                Arc::clone(dst_structure),
                Arc::clone(src_structure),
                storage_conjugate,
            )?;
            self.structures.insert(structure_key.clone(), structure);
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
