use core::ops::{Add, Mul};
use std::fmt;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, Weak};

use num_traits::Zero;
use rustc_hash::FxHashMap;
use tenet_core::{
    BlockStructure, FusionTreePairKey, FusionTreePairOrientation, GenericBraidScalar,
    GenericRigidSymbols, LocallyValidatedFusionTreeBlockStructure, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, RuleIdentity, TensorMap, TensorStorage,
};

use crate::cache::{OperationCachePolicy, TreeTransformStructureCacheKey};
use crate::{OperationError, TreeTransformStructure, TreeTransformStructureCache};

use super::operation::{TreeTransformOperation, TreeTransformRuleCacheKey};
use super::plan::{
    build_all_codomain_tree_transform_group_plan_validated_with_threads,
    build_generic_tree_pair_transform_group_plan_validated,
    build_oriented_tree_pair_transform_group_plan_with_threads,
    compile_multiplicity_free_tree_pair_structure_after_capability_with_threads,
    compile_multiplicity_free_tree_pair_structure_with_threads,
    validate_all_codomain_namespace_before_cache, validate_generic_tree_pair_preflight,
    validate_multiplicity_free_all_codomain_preflight_after_capability,
    validate_multiplicity_free_tree_transform_capability,
    validate_tree_pair_namespace_before_cache,
};
#[cfg(test)]
use super::plan::{
    build_tree_pair_transform_group_plan_validated_with_threads,
    validate_multiplicity_free_tree_pair_preflight,
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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct RuntimeTreeTransformOperationKey {
    rule: RuleIdentity,
    operation: TreeTransformOperation,
}

type RuntimeTreeTransformKey = TreeTransformStructureCacheKey<RuntimeTreeTransformOperationKey>;

#[derive(Clone)]
struct RuntimeTreeTransformStoreEntry<T> {
    structure: Arc<TreeTransformStructure<T>>,
    charged_bytes: usize,
}

struct RuntimeTreeTransformStoreState<T> {
    entries: lru::LruCache<
        RuntimeTreeTransformKey,
        RuntimeTreeTransformStoreEntry<T>,
        rustc_hash::FxBuildHasher,
    >,
    entry_capacity: usize,
    byte_budget: usize,
    max_entry_bytes: usize,
    charged_payload_bytes: usize,
    generation: u64,
    hits: usize,
    misses: usize,
    evictions: usize,
    admission_bypasses: usize,
}

const DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_ENTRIES: usize = 256;
const DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_MAX_ENTRY_BYTES: usize = 8 * 1024 * 1024;
const RUNTIME_TREE_TRANSFORM_LRU_NODE_ALLOWANCE: usize = 8 * core::mem::size_of::<usize>();

/// One Runtime-owned store for completed multiplicity-free tree-pair structures.
#[doc(hidden)]
pub struct RuntimeTreeTransformStore<T> {
    state: Mutex<RuntimeTreeTransformStoreState<T>>,
}

/// Snapshot of one Runtime's completed tree-transform cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeTreeTransformCacheInfo {
    entries: usize,
    entry_capacity: usize,
    charged_payload_bytes: usize,
    byte_budget: usize,
    hits: usize,
    misses: usize,
    evictions: usize,
    admission_bypasses: usize,
}

impl RuntimeTreeTransformCacheInfo {
    pub fn entries(self) -> usize {
        self.entries
    }

    pub fn entry_capacity(self) -> usize {
        self.entry_capacity
    }

    /// Conservative cache-owned payload charge, not resident-memory usage.
    pub fn charged_payload_bytes(self) -> usize {
        self.charged_payload_bytes
    }

    pub fn byte_budget(self) -> usize {
        self.byte_budget
    }

    pub fn hits(self) -> usize {
        self.hits
    }

    pub fn misses(self) -> usize {
        self.misses
    }

    pub fn evictions(self) -> usize {
        self.evictions
    }

    pub fn admission_bypasses(self) -> usize {
        self.admission_bypasses
    }
}

impl<T> RuntimeTreeTransformStoreState<T> {
    fn new(entry_capacity: usize, byte_budget: usize, max_entry_bytes: usize) -> Self {
        Self {
            entries: lru::LruCache::with_hasher(
                NonZeroUsize::new(entry_capacity)
                    .expect("tree-transform cache capacity is nonzero"),
                rustc_hash::FxBuildHasher,
            ),
            entry_capacity,
            byte_budget,
            max_entry_bytes,
            charged_payload_bytes: 0,
            generation: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            admission_bypasses: 0,
        }
    }

    fn info(&self) -> RuntimeTreeTransformCacheInfo {
        RuntimeTreeTransformCacheInfo {
            entries: self.entries.len(),
            entry_capacity: self.entry_capacity,
            charged_payload_bytes: self.charged_payload_bytes,
            byte_budget: self.byte_budget,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            admission_bypasses: self.admission_bypasses,
        }
    }
}

impl<T> RuntimeTreeTransformStore<T> {
    #[doc(hidden)]
    pub const DEFAULT_BYTE_BUDGET: usize = 64 * 1024 * 1024;

    pub fn new(byte_budget: usize) -> Self {
        Self::with_limits(
            DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_ENTRIES,
            byte_budget,
            DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_MAX_ENTRY_BYTES,
        )
    }

    fn with_limits(entry_capacity: usize, byte_budget: usize, max_entry_bytes: usize) -> Self {
        Self {
            state: Mutex::new(RuntimeTreeTransformStoreState::new(
                entry_capacity,
                byte_budget,
                max_entry_bytes,
            )),
        }
    }

    pub fn info(&self) -> RuntimeTreeTransformCacheInfo {
        self.state
            .lock()
            .expect("runtime tree-transform store poisoned")
            .info()
    }

    pub fn clear(&self) {
        let mut state = self
            .state
            .lock()
            .expect("runtime tree-transform store poisoned");
        state.generation = state.generation.wrapping_add(1);
        state.entries.clear();
        state.charged_payload_bytes = 0;
        state.hits = 0;
        state.misses = 0;
        state.evictions = 0;
        state.admission_bypasses = 0;
    }

    fn charged_entry_bytes(
        key: &RuntimeTreeTransformKey,
        structure: &TreeTransformStructure<T>,
    ) -> usize {
        const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();

        core::mem::size_of::<RuntimeTreeTransformKey>()
            .saturating_add(core::mem::size_of::<Arc<TreeTransformStructure<T>>>())
            .saturating_add(key.plan().rule.charged_retained_bytes())
            .saturating_add(key.plan().operation.charged_retained_bytes())
            .saturating_add(structure.charged_payload_bytes())
            .saturating_add(ARC_CONTROL_BYTES)
            .saturating_add(RUNTIME_TREE_TRANSFORM_LRU_NODE_ALLOWANCE)
    }

    fn get_or_compile<E>(
        &self,
        key: RuntimeTreeTransformKey,
        compile: impl FnOnce() -> Result<Arc<TreeTransformStructure<T>>, E>,
    ) -> Result<Arc<TreeTransformStructure<T>>, E> {
        let generation = {
            let mut state = self
                .state
                .lock()
                .expect("runtime tree-transform store poisoned");
            if let Some(entry) = state.entries.get(&key) {
                let structure = Arc::clone(&entry.structure);
                state.hits = state.hits.saturating_add(1);
                return Ok(structure);
            }
            state.misses = state.misses.saturating_add(1);
            state.generation
        };

        let structure = compile()?;
        let charged_bytes = Self::charged_entry_bytes(&key, &structure);
        let mut state = self
            .state
            .lock()
            .expect("runtime tree-transform store poisoned");
        if let Some(entry) = state.entries.get(&key) {
            return Ok(Arc::clone(&entry.structure));
        }
        if state.generation != generation {
            return Ok(structure);
        }
        if charged_bytes > state.max_entry_bytes || charged_bytes > state.byte_budget {
            state.admission_bypasses = state.admission_bypasses.saturating_add(1);
            return Ok(structure);
        }
        while state.entries.len() == state.entry_capacity
            || state.charged_payload_bytes.saturating_add(charged_bytes) > state.byte_budget
        {
            let Some((_, evicted)) = state.entries.pop_lru() else {
                break;
            };
            state.charged_payload_bytes = state
                .charged_payload_bytes
                .saturating_sub(evicted.charged_bytes);
            state.evictions = state.evictions.saturating_add(1);
        }
        state.charged_payload_bytes = state.charged_payload_bytes.saturating_add(charged_bytes);
        state.entries.put(
            key,
            RuntimeTreeTransformStoreEntry {
                structure: Arc::clone(&structure),
                charged_bytes,
            },
        );
        Ok(structure)
    }
}

impl<T> Default for RuntimeTreeTransformStore<T> {
    fn default() -> Self {
        Self::new(Self::DEFAULT_BYTE_BUDGET)
    }
}

pub(crate) const DEFAULT_TREE_TRANSFORM_CACHE_ENTRIES: usize = 256;

/// Context-local retention for completed immutable tree-transform structures.
///
/// Standalone expert contexts may retain ordinary multiplicity-free and
/// all-codomain structures according to [`OperationCachePolicy`]. Runtime-bound
/// ordinary tree-pair operations use their Runtime-owned store instead. Generic
/// and prelowered callback paths compile eagerly and are not retained here.
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
                },
                dst_structure,
                src_structure,
                storage_conjugate,
            )?;
            return store.get_or_compile(key, || {
                compile_multiplicity_free_tree_pair_structure_after_capability_with_threads(
                    rule,
                    operation,
                    Arc::clone(dst_structure),
                    Arc::clone(src_structure),
                    storage_conjugate,
                    self.recoupling_threads,
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
    pub(crate) fn get_or_compile_tree_pair_oriented<R, FAxis>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        logical_keys: &[FusionTreePairKey],
        storage_indices: &[usize],
        storage_src_structure: &Arc<BlockStructure>,
        orientation: FusionTreePairOrientation,
        logical_rank: usize,
        logical_to_storage_axis: FAxis,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        if logical_keys.len() != storage_indices.len() {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented source projection",
            });
        }
        let mut projection =
            FxHashMap::with_capacity_and_hasher(logical_keys.len(), rustc_hash::FxBuildHasher);
        // Why not track storage-index uniqueness here: FusionOperandLayout
        // already proves this projection is a bijection onto parent blocks.
        for (position, (key, &storage_index)) in
            logical_keys.iter().zip(storage_indices).enumerate()
        {
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
        self.stats.structure_misses += 1;
        let plan = build_oriented_tree_pair_transform_group_plan_with_threads(
            rule,
            operation.clone(),
            logical_keys,
            storage_src_structure,
            orientation,
            logical_rank,
            &projection,
            self.recoupling_threads,
        )?;
        let source_index = |key: &FusionTreePairKey| {
            projection
                .get(key)
                .copied()
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: Box::new(tenet_core::BlockKey::FusionTree(key.clone())),
                })
        };
        Ok(Arc::new(
            plan.compile_shared_structures_with_source_projection(
                Arc::clone(dst_structure),
                Arc::clone(storage_src_structure),
                logical_rank,
                source_index,
                logical_to_storage_axis,
                orientation == FusionTreePairOrientation::Adjoint,
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

#[cfg(test)]
mod runtime_store_tests {
    use std::convert::Infallible;
    use std::sync::{Arc, Barrier};

    use tenet_core::{BlockKey, BlockSpec, BlockStructure, RuleIdentity};

    use super::{
        RuntimeTreeTransformKey, RuntimeTreeTransformOperationKey, RuntimeTreeTransformStore,
    };
    use crate::{
        TreeTransformBlockSpec, TreeTransformOperation, TreeTransformStructure,
        TreeTransformStructureCacheKey,
    };

    struct TestRuleIdentity;

    fn fixture(tag: usize) -> (RuntimeTreeTransformKey, Arc<TreeTransformStructure<f64>>) {
        let block = BlockSpec::with_key(BlockKey::ordinal(tag), vec![1], vec![1], 0).unwrap();
        let structure = BlockStructure::from_blocks_with_rank(1, vec![block]).unwrap();
        let compiled = Arc::new(
            TreeTransformStructure::compile_structures(
                &structure,
                &structure,
                &[TreeTransformBlockSpec::single(0, 0, 1.0)],
            )
            .unwrap(),
        );
        let key = TreeTransformStructureCacheKey::from_structures(
            RuntimeTreeTransformOperationKey {
                rule: RuleIdentity::of_type::<TestRuleIdentity>(),
                operation: TreeTransformOperation::permute([tag], []),
            },
            &structure,
            &structure,
        )
        .unwrap();
        (key, compiled)
    }

    #[test]
    fn runtime_store_enforces_resources_and_clear_keeps_returned_arcs_valid() {
        // What: entry and byte pressure evict, oversized entries bypass, and
        // clear resets accounting without invalidating caller-owned payloads.
        let (key0, structure0) = fixture(0);
        let (key1, structure1) = fixture(1);
        let charge0 = RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&key0, &structure0);
        let charge1 = RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&key1, &structure1);

        let entry_limited = RuntimeTreeTransformStore::with_limits(1, usize::MAX, usize::MAX);
        let active = entry_limited
            .get_or_compile(key0.clone(), || {
                Ok::<_, Infallible>(Arc::clone(&structure0))
            })
            .unwrap();
        entry_limited
            .get_or_compile(key1.clone(), || {
                Ok::<_, Infallible>(Arc::clone(&structure1))
            })
            .unwrap();
        assert_eq!(entry_limited.info().entries(), 1);
        assert_eq!(entry_limited.info().evictions(), 1);
        assert_eq!(active.block_count(), 1);

        let byte_limited = RuntimeTreeTransformStore::with_limits(
            2,
            charge0.saturating_add(charge1).saturating_sub(1),
            usize::MAX,
        );
        byte_limited
            .get_or_compile(key0.clone(), || {
                Ok::<_, Infallible>(Arc::clone(&structure0))
            })
            .unwrap();
        byte_limited
            .get_or_compile(key1, || Ok::<_, Infallible>(Arc::clone(&structure1)))
            .unwrap();
        assert_eq!(byte_limited.info().entries(), 1);
        assert_eq!(byte_limited.info().evictions(), 1);

        let oversized =
            RuntimeTreeTransformStore::with_limits(2, usize::MAX, charge0.saturating_sub(1));
        oversized
            .get_or_compile(key0.clone(), || {
                Ok::<_, Infallible>(Arc::clone(&structure0))
            })
            .unwrap();
        assert_eq!(oversized.info().entries(), 0);
        assert_eq!(oversized.info().admission_bypasses(), 1);

        let disabled = RuntimeTreeTransformStore::new(0);
        disabled
            .get_or_compile(key0, || Ok::<_, Infallible>(Arc::clone(&structure0)))
            .unwrap();
        assert_eq!(disabled.info().entries(), 0);
        assert_eq!(disabled.info().byte_budget(), 0);
        assert_eq!(disabled.info().admission_bypasses(), 1);

        entry_limited.clear();
        let cleared = entry_limited.info();
        assert_eq!(cleared.entries(), 0);
        assert_eq!(cleared.entry_capacity(), 1);
        assert_eq!(cleared.charged_payload_bytes(), 0);
        assert_eq!(cleared.byte_budget(), usize::MAX);
        assert_eq!(cleared.hits(), 0);
        assert_eq!(cleared.misses(), 0);
        assert_eq!(cleared.evictions(), 0);
        assert_eq!(cleared.admission_bypasses(), 0);
        assert_eq!(active.block_count(), 1);
    }

    #[test]
    fn clear_prevents_a_racing_old_generation_from_reinserting() {
        // What: a compiler that began before clear may finish for its caller but
        // cannot publish into the cleared Runtime generation.
        let store = Arc::new(RuntimeTreeTransformStore::with_limits(
            2,
            usize::MAX,
            usize::MAX,
        ));
        let (key, structure) = fixture(2);
        let next_key = key.clone();
        let next_structure = Arc::clone(&structure);
        let started = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        let worker_store = Arc::clone(&store);
        let worker_started = Arc::clone(&started);
        let worker_resume = Arc::clone(&resume);
        let worker = std::thread::spawn(move || {
            worker_store
                .get_or_compile(key, || {
                    worker_started.wait();
                    worker_resume.wait();
                    Ok::<_, Infallible>(structure)
                })
                .unwrap()
        });

        started.wait();
        store.clear();
        resume.wait();
        let returned = worker.join().unwrap();

        assert_eq!(returned.block_count(), 1);
        let cleared = store.info();
        assert_eq!(cleared.entries(), 0);
        assert_eq!(cleared.misses(), 0);
        store
            .get_or_compile(next_key, || Ok::<_, Infallible>(next_structure))
            .unwrap();
        let admitted = store.info();
        assert_eq!(admitted.entries(), 1);
        assert_eq!(admitted.misses(), 1);
        assert_eq!(admitted.hits(), 0);
    }
}

impl<T, RuleKey> TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    /// Binds this cache lane to one Runtime-owned completed-structure store.
    #[doc(hidden)]
    pub fn bind_runtime_store(&mut self, store: Weak<RuntimeTreeTransformStore<T>>) {
        self.structures.set_policy(OperationCachePolicy::NoCache);
        self.policy = OperationCachePolicy::NoCache;
        self.runtime_store = Some(store);
    }
}
