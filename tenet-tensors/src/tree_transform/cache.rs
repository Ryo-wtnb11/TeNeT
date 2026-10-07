use core::ops::{Add, Mul};
use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, Weak};

use num_traits::Zero;
use rustc_hash::FxHashMap;
use tenet_core::{
    BlockStructure, BlockStructureContent, FusionTreeGroupKey, FusionTreePairKey,
    FusionTreePairOrientation, HomSpaceId, LocallyValidatedFusionTreeBlockStructure,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, RuleIdentity, TensorMap,
    TensorStorage, WeakHomSpaceId,
};
#[cfg(any(test, feature = "testing"))]
use tenet_core::{CategoricalScalar, GenericRigidSymbols};

use crate::cache::{BlockStructureCacheKey, OperationCachePolicy, TreeTransformStructureCacheKey};
use crate::{
    OperationError, TreeTransformGroupBlockSpec, TreeTransformGroupPlan, TreeTransformStructure,
    TreeTransformStructureCache,
};

use super::operation::{TreeTransformOperation, TreeTransformRuleCacheKey};
use super::plan::{
    build_all_codomain_tree_transform_group_plan_validated_with_threads,
    build_multiplicity_free_tree_pair_plan_after_capability_with_threads,
    build_oriented_tree_pair_transform_group_plan_capability_validated,
    compile_multiplicity_free_tree_pair_structure_after_capability_with_threads,
    compile_multiplicity_free_tree_pair_structure_with_threads,
    validate_all_codomain_namespace_before_cache,
    validate_multiplicity_free_all_codomain_preflight_after_capability,
    validate_multiplicity_free_tree_transform_capability,
    validate_tree_pair_namespace_before_cache,
};
#[cfg(any(test, feature = "testing"))]
use super::plan::{
    build_generic_tree_pair_transform_group_plan_validated, validate_generic_tree_pair_preflight,
};
#[cfg(test)]
use super::plan::{
    build_tree_pair_transform_group_plan_validated_with_threads,
    validate_multiplicity_free_tree_pair_preflight,
};

mod runtime_store;
#[cfg(test)]
mod runtime_store_tests;
pub use runtime_store::*;

fn oriented_source_projection<'a>(
    logical_keys: &'a [FusionTreePairKey],
    storage_indices: &[usize],
    storage_src_structure: &BlockStructure,
) -> Result<FxHashMap<&'a FusionTreePairKey, usize>, OperationError> {
    let mut projection =
        FxHashMap::with_capacity_and_hasher(logical_keys.len(), rustc_hash::FxBuildHasher);
    // Why not track storage-index uniqueness here: FusionOperandLayout
    // already proves this projection is a bijection onto parent blocks.
    for (position, (key, &storage_index)) in logical_keys.iter().zip(storage_indices).enumerate() {
        if storage_index >= storage_src_structure.block_count() {
            return Err(OperationError::BlockIndexOutOfBounds {
                tensor: "oriented src",
                index: storage_index,
                count: storage_src_structure.block_count(),
            });
        }
        if projection.insert(key, storage_index).is_some() {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "src",
                index: position,
            });
        }
    }
    Ok(projection)
}

#[allow(clippy::too_many_arguments)]
fn compile_oriented_tree_pair_structure<R, FAxis>(
    rule: &R,
    operation: &TreeTransformOperation,
    dst_structure: &Arc<BlockStructure>,
    logical_keys: &[FusionTreePairKey],
    storage_src_structure: &Arc<BlockStructure>,
    orientation: FusionTreePairOrientation,
    basis_order: OrientedBasisOrder,
    logical_rank: usize,
    projection: &FxHashMap<&FusionTreePairKey, usize>,
    logical_to_storage_axis: FAxis,
    threads: usize,
    plans: Option<&RuntimeTreeTransformStore<R::Scalar>>,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
    FAxis: Fn(usize) -> Result<usize, OperationError>,
{
    #[cfg(test)]
    ORIENTED_TREE_PAIR_COMPILES.set(ORIENTED_TREE_PAIR_COMPILES.get() + 1);
    let build = |reuse: Option<&GroupSpecReuse<'_, R::Scalar>>| {
        build_oriented_tree_pair_transform_group_plan_capability_validated(
            rule,
            operation.clone(),
            logical_keys,
            storage_src_structure,
            orientation,
            basis_order,
            logical_rank,
            projection,
            threads,
            reuse,
        )
    };
    // Why the storage source's sectors suffice: the logical keys, projection,
    // rank and axis map all derive from the parent structure's keys and split
    // (see `TreeTransformCache::get_or_compile_tree_pair_oriented`).
    let plan = match plans {
        Some(store) => store.get_or_build_plan(
            CategoricalTransformKey::new(
                rule.rule_identity(),
                operation,
                orientation,
                basis_order,
                orientation == FusionTreePairOrientation::Adjoint,
                dst_structure,
                storage_src_structure,
                None,
            ),
            |reuse| build(Some(reuse)),
        )?,
        None => Arc::new(build(None)?),
    };
    let source_index = |key: &FusionTreePairKey| {
        projection
            .get(key)
            .copied()
            .ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(tenet_core::BlockKey::FusionTree(key.clone())),
            })
    };
    plan.compile_shared_structures_with_source_projection(
        Arc::clone(dst_structure),
        Arc::clone(storage_src_structure),
        logical_rank,
        source_index,
        logical_to_storage_axis,
        orientation == FusionTreePairOrientation::Adjoint,
    )
}

#[cfg(test)]
std::thread_local! {
    static ORIENTED_TREE_PAIR_COMPILES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Oriented tree-pair plans compiled on this thread since the last reset.
#[cfg(test)]
pub(crate) fn take_oriented_tree_pair_compiles() -> usize {
    ORIENTED_TREE_PAIR_COMPILES.replace(0)
}

pub(crate) const DEFAULT_TREE_TRANSFORM_CACHE_ENTRIES: usize = 256;

/// Context-local retention for completed immutable tree-transform structures.
///
/// Standalone expert contexts may retain ordinary multiplicity-free and
/// all-codomain structures according to [`OperationCachePolicy`]. Runtime-bound
/// ordinary, adjoint-oriented, and checked Generic tree-pair operations use
/// their Runtime-owned store instead. Prelowered callback paths compile eagerly
/// and are not retained here.
pub struct TreeTransformCache<T, RuleKey> {
    structures: TreeTransformStructureCache<T, TreeTransformStructureOperationKey<RuleKey>>,
    runtime_store: Option<Weak<RuntimeTreeTransformStore<T>>>,
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
            runtime_store: self.runtime_store.clone(),
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
            .field("runtime_bound", &self.runtime_store.is_some())
            .field("stats", &self.stats)
            .field("recoupling_threads", &self.recoupling_threads)
            .finish()
    }
}

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
            runtime_store: None,
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
            runtime_store: None,
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
    /// Returns context-local activity only; Runtime-owned stores report through
    /// the user Runtime API.
    pub fn stats(&self) -> TreeTransformCacheStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = TreeTransformCacheStats::default();
    }

    pub(crate) fn runtime_store(&self) -> Option<Arc<RuntimeTreeTransformStore<T>>> {
        self.runtime_store.as_ref().and_then(Weak::upgrade)
    }

    fn structure_key(
        rule: RuleKey,
        scope: TreeTransformScope,
        operation: TreeTransformOperation,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        storage_conjugate: bool,
        orientation: FusionTreePairOrientation,
        basis_order: OrientedBasisOrder,
    ) -> Result<
        TreeTransformStructureCacheKey<TreeTransformStructureOperationKey<RuleKey>>,
        OperationError,
    > {
        TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
            TreeTransformStructureOperationKey {
                rule,
                scope,
                operation,
                orientation,
                basis_order,
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
        if let Some(runtime_store) = &self.runtime_store {
            let Some(store) = runtime_store.upgrade() else {
                return compile_multiplicity_free_tree_pair_structure_with_threads(
                    rule,
                    operation,
                    Arc::clone(dst_structure),
                    Arc::clone(src_structure),
                    storage_conjugate,
                    self.recoupling_threads,
                )
                .map(Arc::new);
            };
            validate_multiplicity_free_tree_transform_capability(rule, operation)?;
            validate_tree_pair_namespace_before_cache(operation, src_structure)?;
            let key = TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
                RuntimeTreeTransformOperationKey {
                    rule: rule.rule_identity(),
                    operation: operation.clone(),
                    orientation: FusionTreePairOrientation::Direct,
                    basis_order: OrientedBasisOrder::Canonical,
                    logical_source: None,
                },
                dst_structure,
                src_structure,
                storage_conjugate,
            )?;
            let threads = self.recoupling_threads;
            return store.get_or_compile(key, || {
                let plan_key = CategoricalTransformKey::new(
                    rule.rule_identity(),
                    operation,
                    FusionTreePairOrientation::Direct,
                    OrientedBasisOrder::Canonical,
                    storage_conjugate,
                    dst_structure,
                    src_structure,
                    None,
                );
                store
                    .get_or_build_plan(plan_key, |reuse| {
                        build_multiplicity_free_tree_pair_plan_after_capability_with_threads(
                            rule,
                            operation,
                            dst_structure,
                            src_structure,
                            threads,
                            Some(reuse),
                        )
                    })?
                    .compile_shared_structures_with_storage_conjugation(
                        Arc::clone(dst_structure),
                        Arc::clone(src_structure),
                        storage_conjugate,
                    )
                    .map(Arc::new)
            });
        }

        if !self.policy.stores_entries() {
            let structure = compile_multiplicity_free_tree_pair_structure_with_threads(
                rule,
                operation,
                Arc::clone(dst_structure),
                Arc::clone(src_structure),
                storage_conjugate,
                self.recoupling_threads,
            )
            .map(Arc::new)?;
            self.stats.structure_misses += 1;
            return Ok(structure);
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
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
        )?;
        if let Some(structure) = self.cached_structure(&key) {
            return Ok(structure);
        }

        let structure = Arc::new(
            compile_multiplicity_free_tree_pair_structure_after_capability_with_threads(
                rule,
                operation,
                Arc::clone(dst_structure),
                Arc::clone(src_structure),
                storage_conjugate,
                self.recoupling_threads,
            )?,
        );
        self.stats.structure_misses += 1;
        self.retain_structure(key, Arc::clone(&structure));
        Ok(structure)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_compile_tree_pair_oriented<'p, R, FIndices, FAxis>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        logical_keys: &[FusionTreePairKey],
        storage_indices: FIndices,
        storage_src_structure: &Arc<BlockStructure>,
        orientation: FusionTreePairOrientation,
        basis_order: OrientedBasisOrder,
        logical_rank: usize,
        logical_to_storage_axis: FAxis,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        FIndices: FnOnce() -> Result<&'p [usize], OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        validate_multiplicity_free_tree_transform_capability(rule, operation)?;
        validate_tree_pair_namespace_before_cache(operation, storage_src_structure)?;
        // Why lazy: only a store miss reads the storage map, so a warm call
        // derives no per-block projection.
        let projection = || {
            let storage_indices = storage_indices()?;
            if logical_keys.len() != storage_indices.len() {
                return Err(OperationError::StructureMismatch {
                    tensor: "oriented source projection",
                });
            }
            oriented_source_projection(logical_keys, storage_indices, storage_src_structure)
        };
        let storage_conjugate = orientation == FusionTreePairOrientation::Adjoint;
        let threads = self.recoupling_threads;
        let compile = |projection: &FxHashMap<&FusionTreePairKey, usize>,
                       plans: Option<&RuntimeTreeTransformStore<T>>| {
            compile_oriented_tree_pair_structure(
                rule,
                operation,
                dst_structure,
                logical_keys,
                storage_src_structure,
                orientation,
                basis_order,
                logical_rank,
                projection,
                logical_to_storage_axis,
                threads,
                plans,
            )
        };
        // Why only the adjoint orientation: a direct oriented key would equal
        // the ordinary tree-pair key while its plan follows the caller's
        // logical key order, and no production caller compiles it.
        let runtime_store = storage_conjugate.then(|| self.runtime_store()).flatten();
        if let Some(store) = runtime_store {
            // Why not key the logical keys, projection, rank, or axis map: the
            // two operand preparations derive them from the parent structure
            // and the keyed basis order: canonical enumeration for Dynamic,
            // parent block order for static views. The axis map reads the
            // parent split; a blockless parent compiles no spec.
            let key = TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
                RuntimeTreeTransformOperationKey {
                    rule: rule.rule_identity(),
                    operation: operation.clone(),
                    logical_source: None,
                    orientation,
                    basis_order,
                },
                dst_structure,
                storage_src_structure,
                storage_conjugate,
            )?;
            return store
                .get_or_compile(key, || compile(&projection()?, Some(&store)).map(Arc::new));
        }
        let key = if self.runtime_store.is_none() && self.policy.stores_entries() {
            let key = Self::structure_key(
                rule.tree_transform_rule_cache_key(),
                TreeTransformScope::TreePair,
                operation.clone(),
                dst_structure,
                storage_src_structure,
                storage_conjugate,
                orientation,
                basis_order,
            )?;
            if let Some(structure) = self.cached_structure(&key) {
                return Ok(structure);
            }
            Some(key)
        } else {
            None
        };
        let projection = projection()?;
        self.stats.structure_misses += 1;
        let structure = compile(&projection, None).map(Arc::new)?;
        if let Some(key) = key {
            self.retain_structure(key, Arc::clone(&structure));
        }
        Ok(structure)
    }

    /// Structure-only Generic-fusion sibling of [`Self::get_or_compile_tree_pair`].
    ///
    /// This remains eager because completed-transformer retention for Generic
    /// fusion needs its own measured key and ownership contract. Why not retain
    /// a rule key here: provider-domain validation is the eager boundary, and
    /// this path retains no key or completed structure.
    #[cfg(any(test, feature = "testing"))]
    pub fn get_or_compile_tree_pair_structures_generic<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: GenericRigidSymbols<Scalar = T>,
        R::Scalar: CategoricalScalar,
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
                FusionTreePairOrientation::Direct,
                OrientedBasisOrder::Canonical,
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

    /// Binds this cache lane to one Runtime-owned completed-structure store.
    #[doc(hidden)]
    pub fn bind_runtime_store(&mut self, store: Weak<RuntimeTreeTransformStore<T>>) {
        self.structures.set_policy(OperationCachePolicy::NoCache);
        self.policy = OperationCachePolicy::NoCache;
        self.runtime_store = Some(store);
    }
}
