use std::collections::VecDeque;
use std::fmt;
use std::num::NonZeroUsize;

use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tenet_core::{BlockStructure, BlockStructureContent, BlockStructureContentBlock};

use crate::{OperationError, TensorContractStructure, TreeTransformStructure};

/// Clears the tenet-core intern tables.
///
/// Tree-transform execution plans are not persisted to disk, so this function
/// has no filesystem ownership or cross-process reset effect.
///
/// Why not rename it: public compatibility; no operation-result cache remains,
/// and it now resets only global tenet-core intern, layout, and algebra state.
pub fn reset_global_operation_caches() {
    tenet_core::reset_core_intern_tables();
}

/// Cache policy for reusable algebra and tree-transform components.
///
/// Ordinary contraction routes resolve eagerly; complete replay is retained
/// only by an explicit prepared handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperationCachePolicy {
    NoCache,
    TaskLocal,
    TaskLocalLru { max_entries: usize },
}

// Why-not use `OperationCachePolicy::default()`: that intentionally remains the
// explicit unbounded task-local policy for callers that ask for it, while default
// execution contexts should not grow without a cap.
pub(crate) const DEFAULT_CONTRACT_CONTEXT_CACHE_ENTRIES: usize = 256;

impl Default for OperationCachePolicy {
    fn default() -> Self {
        Self::TaskLocal
    }
}

impl OperationCachePolicy {
    #[inline]
    pub const fn no_cache() -> Self {
        Self::NoCache
    }

    #[inline]
    pub const fn task_local() -> Self {
        Self::TaskLocal
    }

    #[inline]
    pub const fn task_local_lru(max_entries: usize) -> Self {
        Self::TaskLocalLru { max_entries }
    }

    #[inline]
    pub(crate) const fn stores_entries(self) -> bool {
        !matches!(self, Self::NoCache | Self::TaskLocalLru { max_entries: 0 })
    }

    #[inline]
    pub(crate) const fn max_entries(self) -> Option<usize> {
        match self {
            Self::NoCache | Self::TaskLocal => None,
            Self::TaskLocalLru { max_entries } => Some(max_entries),
        }
    }
}

pub(crate) fn touch_lru_key<K>(order: &mut VecDeque<K>, key: &K)
where
    K: Clone + Eq,
{
    if let Some(position) = order.iter().position(|stored| stored == key) {
        order.remove(position);
    }
    order.push_back(key.clone());
}

pub(crate) fn enforce_lru_limit<K, V>(
    map: &mut FxHashMap<K, V>,
    order: &mut VecDeque<K>,
    max_entries: usize,
) where
    K: Clone + Eq + Hash,
{
    while map.len() > max_entries {
        let Some(oldest) = order.pop_front() else {
            break;
        };
        map.remove(&oldest);
    }
}

pub(crate) fn rebuild_lru_order_from_keys<K, V>(map: &FxHashMap<K, V>, order: &mut VecDeque<K>)
where
    K: Clone,
{
    order.clear();
    order.extend(map.keys().cloned());
}

#[derive(Clone, Debug)]
pub struct BlockStructureCacheKey {
    content: Arc<BlockStructureContent>,
}

impl BlockStructureCacheKey {
    pub fn from_structure(structure: &BlockStructure) -> Result<Self, OperationError> {
        Ok(Self {
            content: structure.content_key(),
        })
    }

    #[inline]
    pub fn id(&self) -> usize {
        self.content.id()
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.content.rank()
    }

    #[inline]
    pub fn blocks(&self) -> &[BlockStructureCacheBlockKey] {
        self.content.blocks()
    }
}

impl PartialEq for BlockStructureCacheKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.content.id() == other.content.id()
    }
}

impl Eq for BlockStructureCacheKey {}

impl Hash for BlockStructureCacheKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.content.id().hash(state);
    }
}

pub type BlockStructureCacheBlockKey = BlockStructureContentBlock;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformStructureCacheKey<PlanKey> {
    plan: PlanKey,
    dst: BlockStructureCacheKey,
    src: BlockStructureCacheKey,
    storage_conjugate: bool,
}

impl<PlanKey> TreeTransformStructureCacheKey<PlanKey>
where
    PlanKey: Clone,
{
    pub fn from_structures(
        plan: PlanKey,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        Self::from_structures_with_storage_conjugation(plan, dst_structure, src_structure, false)
    }

    pub fn from_structures_with_storage_conjugation(
        plan: PlanKey,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        Ok(Self {
            plan,
            dst: BlockStructureCacheKey::from_structure(dst_structure)?,
            src: BlockStructureCacheKey::from_structure(src_structure)?,
            storage_conjugate,
        })
    }

    #[inline]
    pub fn plan(&self) -> &PlanKey {
        &self.plan
    }

    #[inline]
    pub fn dst(&self) -> &BlockStructureCacheKey {
        &self.dst
    }

    #[inline]
    pub fn src(&self) -> &BlockStructureCacheKey {
        &self.src
    }

    #[inline]
    pub fn storage_conjugate(&self) -> bool {
        self.storage_conjugate
    }
}

pub struct TreeTransformStructureCache<T, PlanKey> {
    structures: lru::LruCache<
        TreeTransformStructureCacheKey<PlanKey>,
        Arc<TreeTransformStructure<T>>,
        rustc_hash::FxBuildHasher,
    >,
    policy: OperationCachePolicy,
}

impl<T, PlanKey> Clone for TreeTransformStructureCache<T, PlanKey>
where
    PlanKey: Clone + Eq + Hash,
{
    fn clone(&self) -> Self {
        let mut cloned = Self::with_policy(self.policy);
        for (key, structure) in self.structures.iter().rev() {
            cloned.structures.put(key.clone(), Arc::clone(structure));
        }
        cloned
    }
}

impl<T, PlanKey> fmt::Debug for TreeTransformStructureCache<T, PlanKey> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TreeTransformStructureCache")
            .field("policy", &self.policy)
            .finish()
    }
}

impl<T, PlanKey> Default for TreeTransformStructureCache<T, PlanKey>
where
    PlanKey: Clone + Eq + Hash,
{
    fn default() -> Self {
        Self::with_policy(OperationCachePolicy::default())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TensorContractStructureCacheKey<PlanKey> {
    plan: PlanKey,
    dst: BlockStructureCacheKey,
    lhs: BlockStructureCacheKey,
    rhs: BlockStructureCacheKey,
}

#[cfg(test)]
thread_local! {
    static TENSOR_CONTRACT_STRUCTURE_CACHE_KEY_BUILDS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_tensor_contract_structure_cache_key_build_count() {
    TENSOR_CONTRACT_STRUCTURE_CACHE_KEY_BUILDS.set(0);
}

#[cfg(test)]
pub(crate) fn tensor_contract_structure_cache_key_build_count() -> usize {
    TENSOR_CONTRACT_STRUCTURE_CACHE_KEY_BUILDS.get()
}

impl<PlanKey> TensorContractStructureCacheKey<PlanKey>
where
    PlanKey: Clone,
{
    pub fn from_structures(
        plan: PlanKey,
        dst_structure: &BlockStructure,
        lhs_structure: &BlockStructure,
        rhs_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        #[cfg(test)]
        TENSOR_CONTRACT_STRUCTURE_CACHE_KEY_BUILDS
            .set(TENSOR_CONTRACT_STRUCTURE_CACHE_KEY_BUILDS.get() + 1);
        Ok(Self {
            plan,
            dst: BlockStructureCacheKey::from_structure(dst_structure)?,
            lhs: BlockStructureCacheKey::from_structure(lhs_structure)?,
            rhs: BlockStructureCacheKey::from_structure(rhs_structure)?,
        })
    }

    #[inline]
    pub fn plan(&self) -> &PlanKey {
        &self.plan
    }

    #[inline]
    pub fn dst(&self) -> &BlockStructureCacheKey {
        &self.dst
    }

    #[inline]
    pub fn lhs(&self) -> &BlockStructureCacheKey {
        &self.lhs
    }

    #[inline]
    pub fn rhs(&self) -> &BlockStructureCacheKey {
        &self.rhs
    }
}

#[derive(Clone, Debug)]
pub struct TensorContractStructureCache<C, PlanKey> {
    structures:
        FxHashMap<TensorContractStructureCacheKey<PlanKey>, Arc<TensorContractStructure<C>>>,
    lru_order: VecDeque<TensorContractStructureCacheKey<PlanKey>>,
    policy: OperationCachePolicy,
}

impl<C, PlanKey> Default for TensorContractStructureCache<C, PlanKey> {
    fn default() -> Self {
        Self {
            structures: FxHashMap::default(),
            lru_order: VecDeque::new(),
            policy: OperationCachePolicy::task_local_lru(DEFAULT_CONTRACT_CONTEXT_CACHE_ENTRIES),
        }
    }
}

impl<C, PlanKey> TensorContractStructureCache<C, PlanKey>
where
    PlanKey: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_policy(policy: OperationCachePolicy) -> Self {
        Self {
            structures: FxHashMap::default(),
            lru_order: VecDeque::new(),
            policy,
        }
    }

    #[inline]
    pub fn policy(&self) -> OperationCachePolicy {
        self.policy
    }

    pub fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        if !policy.stores_entries() {
            self.structures.clear();
            self.lru_order.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            rebuild_lru_order_from_keys(&self.structures, &mut self.lru_order);
            enforce_lru_limit(&mut self.structures, &mut self.lru_order, max_entries);
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.structures.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty()
    }

    pub fn get(
        &self,
        key: &TensorContractStructureCacheKey<PlanKey>,
    ) -> Option<&TensorContractStructure<C>> {
        self.structures.get(key).map(Arc::as_ref)
    }

    pub fn get_arc(
        &self,
        key: &TensorContractStructureCacheKey<PlanKey>,
    ) -> Option<Arc<TensorContractStructure<C>>> {
        self.structures.get(key).map(Arc::clone)
    }

    pub fn touch(&mut self, key: &TensorContractStructureCacheKey<PlanKey>) {
        if self.policy.max_entries().is_some() && self.structures.contains_key(key) {
            touch_lru_key(&mut self.lru_order, key);
        }
    }

    pub fn insert(
        &mut self,
        key: TensorContractStructureCacheKey<PlanKey>,
        structure: TensorContractStructure<C>,
    ) -> Option<Arc<TensorContractStructure<C>>> {
        self.insert_arc(key, Arc::new(structure))
    }

    pub fn insert_arc(
        &mut self,
        key: TensorContractStructureCacheKey<PlanKey>,
        structure: Arc<TensorContractStructure<C>>,
    ) -> Option<Arc<TensorContractStructure<C>>> {
        if !self.policy.stores_entries() {
            return None;
        }
        let old = self.structures.insert(key.clone(), structure);
        if self.policy.max_entries().is_some() {
            touch_lru_key(&mut self.lru_order, &key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            enforce_lru_limit(&mut self.structures, &mut self.lru_order, max_entries);
        }
        old
    }
}

impl<T, PlanKey> TreeTransformStructureCache<T, PlanKey>
where
    PlanKey: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_policy(policy: OperationCachePolicy) -> Self {
        Self {
            structures: local_lru(policy),
            policy,
        }
    }

    #[inline]
    pub fn policy(&self) -> OperationCachePolicy {
        self.policy
    }

    pub fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        if !policy.stores_entries() {
            self.structures.clear();
        }
        self.structures.resize(local_lru_capacity(policy));
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.structures.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty()
    }

    pub fn get(
        &self,
        key: &TreeTransformStructureCacheKey<PlanKey>,
    ) -> Option<&TreeTransformStructure<T>> {
        self.structures.peek(key).map(Arc::as_ref)
    }

    pub fn get_arc(
        &self,
        key: &TreeTransformStructureCacheKey<PlanKey>,
    ) -> Option<Arc<TreeTransformStructure<T>>> {
        self.structures.peek(key).map(Arc::clone)
    }

    pub fn touch(&mut self, key: &TreeTransformStructureCacheKey<PlanKey>) {
        let _ = self.structures.get(key);
    }

    pub fn insert(
        &mut self,
        key: TreeTransformStructureCacheKey<PlanKey>,
        structure: TreeTransformStructure<T>,
    ) -> Option<Arc<TreeTransformStructure<T>>> {
        self.insert_arc(key, Arc::new(structure))
    }

    pub fn insert_arc(
        &mut self,
        key: TreeTransformStructureCacheKey<PlanKey>,
        structure: Arc<TreeTransformStructure<T>>,
    ) -> Option<Arc<TreeTransformStructure<T>>> {
        if !self.policy.stores_entries() {
            return None;
        }
        self.structures.put(key, structure)
    }
}

pub(crate) fn local_lru_capacity(policy: OperationCachePolicy) -> NonZeroUsize {
    NonZeroUsize::new(policy.max_entries().unwrap_or(usize::MAX).max(1))
        .expect("local LRU capacity is at least one")
}

pub(crate) fn local_lru<K, V>(
    policy: OperationCachePolicy,
) -> lru::LruCache<K, V, rustc_hash::FxBuildHasher>
where
    K: Eq + Hash,
{
    let mut cache = lru::LruCache::unbounded_with_hasher(rustc_hash::FxBuildHasher);
    cache.resize(local_lru_capacity(policy));
    cache
}

#[cfg(test)]
mod tests {
    use super::{
        reset_global_operation_caches, OperationCachePolicy, TensorContractStructureCache,
        DEFAULT_CONTRACT_CONTEXT_CACHE_ENTRIES,
    };
    use crate::test_support::CACHE_TEST_LOCK;
    use tenet_core::BlockStructure;

    #[test]
    fn tensor_contract_structure_cache_default_is_bounded() {
        let cache = TensorContractStructureCache::<f64, usize>::default();

        assert_eq!(
            cache.policy(),
            OperationCachePolicy::task_local_lru(DEFAULT_CONTRACT_CONTEXT_CACHE_ENTRIES)
        );
    }

    #[test]
    fn tensor_contract_structure_cache_explicit_task_local_stays_unbounded() {
        let mut cache = TensorContractStructureCache::<f64, usize>::default();

        cache.set_policy(OperationCachePolicy::TaskLocal);

        assert_eq!(cache.policy(), OperationCachePolicy::TaskLocal);
    }

    #[test]
    fn reset_global_operation_caches_chains_core_intern_reset_without_id_reuse() {
        // What: this public reset facade races any test that assumes the shared
        // tenet-core intern table stays stable across
        // two builds (e.g. `dynamic_fusion_fast_space_key_uses_structure_content_identity`
        // in `contract::dynamic`) — a reset landing between such a test's two
        // interning builds would evict the first entry and hand the second a
        // fresh id. See `test_support` for why this is a shared lock.
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Cross-layer coherence: the tensors-level reset must chain into the
        // tenet-core intern tables. If it did not, the identical content would
        // stay interned and re-yield the same id (a stale key could then alias).
        // The monotonic counter + cleared table give a strictly greater id.
        let base = 700_000_000usize;
        let before = BlockStructure::trivial(&[base]).unwrap();
        let id_before = before.content_id();
        reset_global_operation_caches();
        assert_eq!(before.required_len().unwrap(), base);
        let id_after = BlockStructure::trivial(&[base]).unwrap().content_id();
        assert!(
            id_after > id_before,
            "reset chain must not reuse content ids, got before={id_before} after={id_after}"
        );
    }
}
