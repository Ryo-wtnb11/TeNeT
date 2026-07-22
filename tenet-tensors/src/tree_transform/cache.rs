use core::ops::{Add, Mul};
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    BlockStructure, GenericBraidScalar, GenericRigidSymbols,
    LocallyValidatedFusionTreeBlockStructure, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, TensorMap, TensorStorage,
};

use crate::cache::{OperationCachePolicy, TreeTransformStructureCacheKey};
use crate::{OperationError, TreeTransformStructure, TreeTransformStructureCache};

use super::operation::{TreeTransformOperation, TreeTransformRuleCacheKey};
use super::plan::{
    build_all_codomain_tree_transform_group_plan_validated_with_threads,
    build_generic_tree_pair_transform_group_plan_validated,
    build_tree_pair_transform_group_plan_validated_with_threads,
    compile_multiplicity_free_tree_pair_structure_with_threads,
    validate_all_codomain_namespace_before_cache, validate_generic_tree_pair_preflight,
    validate_multiplicity_free_all_codomain_preflight_after_capability,
    validate_multiplicity_free_tree_pair_preflight,
    validate_multiplicity_free_tree_transform_capability,
    validate_tree_pair_namespace_before_cache,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum TreeTransformScope {
    AllCodomain,
    TreePair,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TreeTransformStructureOperationKey<RuleKey> {
    rule: RuleKey,
    scope: TreeTransformScope,
    operation: TreeTransformOperation,
}

pub(crate) const DEFAULT_TREE_TRANSFORM_CACHE_ENTRIES: usize = 256;

/// Context-local retention for completed immutable tree-transform structures.
///
/// Ordinary multiplicity-free and all-codomain operations may retain completed
/// structures according to [`OperationCachePolicy`]. Generic and prelowered
/// callback paths compile eagerly and are not retained here.
pub struct TreeTransformCache<T, RuleKey> {
    structures: TreeTransformStructureCache<T, TreeTransformStructureOperationKey<RuleKey>>,
    policy: OperationCachePolicy,
    stats: TreeTransformCacheStats,
    recoupling_threads: usize,
}

impl<T, RuleKey> Clone for TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    fn clone(&self) -> Self {
        Self {
            structures: self.structures.clone(),
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

pub type TreePairTransformCache<T, RuleKey> = TreeTransformCache<T, RuleKey>;

/// Observable completed-structure cache activity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TreeTransformCacheStats {
    structure_hits: usize,
    structure_misses: usize,
}

impl TreeTransformCacheStats {
    #[inline]
    pub fn structure_hits(self) -> usize {
        self.structure_hits
    }

    #[inline]
    pub fn structure_misses(self) -> usize {
        self.structure_misses
    }
}

/// Defaults to a context-local LRU of completed tree-transform structures.
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
            structures: TreeTransformStructureCache::with_policy(policy),
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_policy(policy: OperationCachePolicy) -> Self {
        Self {
            structures: TreeTransformStructureCache::with_policy(policy),
            policy,
            stats: TreeTransformCacheStats::default(),
            recoupling_threads: 1,
        }
    }

    #[inline]
    pub fn recoupling_threads(&self) -> usize {
        self.recoupling_threads
    }

    /// Sets the worker count used by whole-group categorical compilation.
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
    }

    #[inline]
    pub fn structure_len(&self) -> usize {
        self.structures.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty()
    }

    #[inline]
    pub fn stats(&self) -> TreeTransformCacheStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = TreeTransformCacheStats::default();
    }

    fn structure_key(
        rule: RuleKey,
        scope: TreeTransformScope,
        operation: TreeTransformOperation,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        storage_conjugate: bool,
    ) -> Result<
        TreeTransformStructureCacheKey<TreeTransformStructureOperationKey<RuleKey>>,
        OperationError,
    > {
        TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
            TreeTransformStructureOperationKey {
                rule,
                scope,
                operation,
            },
            dst_structure,
            src_structure,
            storage_conjugate,
        )
    }

    fn cached_structure(
        &mut self,
        key: &TreeTransformStructureCacheKey<TreeTransformStructureOperationKey<RuleKey>>,
    ) -> Option<Arc<TreeTransformStructure<T>>> {
        let structure = self.structures.get_arc(key)?;
        self.stats.structure_hits += 1;
        self.structures.touch(key);
        Some(structure)
    }

    fn retain_structure(
        &mut self,
        key: TreeTransformStructureCacheKey<TreeTransformStructureOperationKey<RuleKey>>,
        structure: Arc<TreeTransformStructure<T>>,
    ) {
        self.structures.insert_arc(key, structure);
    }

    /// Resolve an exact tree-pair replay structure.
    ///
    /// Fusion-tree block keys in `dst` and `src` follow
    /// [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain
    /// precondition.
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
        self.get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            rule,
            &operation,
            dst.structure(),
            src.structure(),
            false,
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
    /// [`Self::get_or_compile_tree_pair_structures_with_storage_conjugation`].
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
        if !self.policy.stores_entries() {
            self.stats.structure_misses += 1;
            return compile_multiplicity_free_tree_pair_structure_with_threads(
                rule,
                operation,
                Arc::clone(dst_structure),
                Arc::clone(src_structure),
                storage_conjugate,
                self.recoupling_threads,
            )
            .map(Arc::new);
        }

        validate_multiplicity_free_tree_transform_capability(rule, operation)?;
        validate_tree_pair_namespace_before_cache(operation, src_structure)?;
        let key = Self::structure_key(
            rule.tree_transform_rule_cache_key(),
            TreeTransformScope::TreePair,
            operation.clone(),
            dst_structure,
            src_structure,
            storage_conjugate,
        )?;
        if let Some(structure) = self.cached_structure(&key) {
            return Ok(structure);
        }

        self.stats.structure_misses += 1;
        let structure = Arc::new(compile_multiplicity_free_tree_pair_structure_with_threads(
            rule,
            operation,
            Arc::clone(dst_structure),
            Arc::clone(src_structure),
            storage_conjugate,
            self.recoupling_threads,
        )?);
        self.retain_structure(key, Arc::clone(&structure));
        Ok(structure)
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
        self.stats.structure_misses += 1;
        let plan = build_tree_pair_transform_group_plan_validated_with_threads(
            &source_proof,
            operation.clone(),
            self.recoupling_threads,
        )?;
        Ok(Arc::new(
            plan.compile_shared_structures_with_storage_mapping(
                Arc::clone(dst_structure),
                logical_src_structure,
                Arc::clone(storage_src_structure),
                logical_to_storage_block,
                logical_to_storage_axis,
                storage_conjugate,
            )?,
        ))
    }

    /// Generic-fusion sibling of [`Self::get_or_compile_tree_pair`].
    ///
    /// This remains eager because completed-transformer retention for Generic
    /// fusion needs its own measured key and ownership contract. The rule key
    /// bound still preserves provider provenance for that future boundary.
    /// Fusion-tree block keys follow
    /// [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain
    /// precondition.
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
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
            .map_err(OperationError::from_core_preserving_context)?;
        self.stats.structure_misses += 1;
        let plan =
            build_generic_tree_pair_transform_group_plan_validated(&source_proof, operation)?;
        Ok(Arc::new(plan.compile(dst, src)?))
    }

    /// Structure-only Generic sibling. It has the same eager ownership and
    /// provider-domain contracts as [`Self::get_or_compile_tree_pair_generic`].
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
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
            .map_err(OperationError::from_core_preserving_context)?;
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
    /// Fusion-tree block keys in `dst` and `src` follow
    /// [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain
    /// precondition.
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

        let key = if self.policy.stores_entries() {
            let key = Self::structure_key(
                rule.tree_transform_rule_cache_key(),
                TreeTransformScope::AllCodomain,
                operation.clone(),
                dst.structure(),
                src.structure(),
                false,
            )?;
            if let Some(structure) = self.cached_structure(&key) {
                return Ok(structure);
            }
            Some(key)
        } else {
            None
        };

        let source_proof = validate_multiplicity_free_all_codomain_preflight_after_capability(
            rule,
            &operation,
            src.structure(),
        )?;
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
            .map_err(OperationError::from_core_preserving_context)?;
        self.stats.structure_misses += 1;
        let plan = build_all_codomain_tree_transform_group_plan_validated_with_threads(
            &source_proof,
            operation,
            self.recoupling_threads,
        )?;
        let structure = Arc::new(plan.compile_shared_structures_with_storage_conjugation(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            false,
        )?);
        if let Some(key) = key {
            self.retain_structure(key, Arc::clone(&structure));
        }
        Ok(structure)
    }
}
