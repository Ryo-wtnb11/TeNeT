use std::any::{Any, TypeId};
use std::collections::VecDeque;
use std::fmt;
use std::num::NonZeroUsize;

use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use tenet_core::{BlockStructure, BlockStructureContent, BlockStructureContentBlock};

use crate::{OperationError, TensorContractStructure, TreeTransformStructure};

type ErasedGlobalCache = Arc<dyn Any + Send + Sync>;
type GlobalCacheRegistry = RwLock<FxHashMap<TypeId, ErasedGlobalCache>>;

static OPERATION_CACHE_RESET_EPOCH: AtomicU64 = AtomicU64::new(0);

fn operation_cache_reset_lock() -> &'static Mutex<()> {
    static RESET_LOCK: Mutex<()> = Mutex::new(());
    &RESET_LOCK
}

fn operation_global_registry() -> &'static GlobalCacheRegistry {
    static REGISTRY: OnceLock<GlobalCacheRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(FxHashMap::default()))
}

#[doc(hidden)]
/// Registers one typed downstream cache in TeNeT's global reset lifecycle.
///
/// Why not expose the erased registry: returning the generation together with
/// a typed `Arc` prevents downstream crates from inventing incompatible keys.
/// A call overlapping reset may finish with the old generation; a lookup that
/// starts after reset returns sees the new generation.
///
/// `T::default()` may run more than once under contention or across reset, so
/// implementations must not rely on construction having a unique side effect.
pub fn registered_operation_cache<T>() -> (u64, Arc<T>)
where
    T: 'static + Default + Send + Sync,
{
    let registry = operation_global_registry();
    let key = TypeId::of::<T>();
    loop {
        {
            let caches = registry.read().expect("global cache registry poisoned");
            if let Some(cache) = caches.get(&key) {
                let epoch = OPERATION_CACHE_RESET_EPOCH.load(Ordering::Relaxed);
                return (
                    epoch,
                    Arc::downcast::<T>(Arc::clone(cache))
                        .expect("global cache registry type id collision"),
                );
            }
        }

        let build_epoch = OPERATION_CACHE_RESET_EPOCH.load(Ordering::Acquire);
        // Why not construct under the registry write lock: a downstream Default
        // may itself register another typed cache. Building a losing candidate
        // twice is preferable to making that valid composition self-deadlock.
        let cache = Arc::new(T::default());
        let mut caches = registry.write().expect("global cache registry poisoned");
        if let Some(existing) = caches.get(&key) {
            let epoch = OPERATION_CACHE_RESET_EPOCH.load(Ordering::Relaxed);
            return (
                epoch,
                Arc::downcast::<T>(Arc::clone(existing))
                    .expect("global cache registry type id collision"),
            );
        }
        let epoch = OPERATION_CACHE_RESET_EPOCH.load(Ordering::Relaxed);
        if epoch != build_epoch {
            // Why not publish a candidate built across reset: its Default may
            // retain another registered payload from the discarded generation.
            drop(caches);
            continue;
        }
        caches.insert(key, Arc::clone(&cache) as ErasedGlobalCache);
        return (epoch, cache);
    }
}

#[doc(hidden)]
/// Returns the generation used for lock-free downstream front-cache validation.
///
/// Why not treat this as a cross-layer quiescence barrier: reset also chains
/// core intern state, while overlapping immutable operations
/// may finish against their previously acquired generation.
pub fn operation_cache_reset_epoch() -> u64 {
    OPERATION_CACHE_RESET_EPOCH.load(Ordering::Acquire)
}

#[cfg(test)]
pub(crate) fn typed_global_map<K, V>() -> Arc<RwLock<FxHashMap<K, V>>>
where
    K: 'static + Eq + Hash + Send + Sync,
    V: 'static + Send + Sync,
{
    registered_operation_cache::<RwLock<FxHashMap<K, V>>>().1
}

/// Clears process-resident operation caches and core intern tables.
///
/// Tree-transform execution plans are not persisted to disk, so this function
/// has no filesystem ownership or cross-process reset effect.
pub fn reset_global_operation_caches() {
    let _reset_guard = operation_cache_reset_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let registry = operation_global_registry();
    let mut caches = registry.write().expect("global cache registry poisoned");
    let base_epoch = OPERATION_CACHE_RESET_EPOCH.load(Ordering::Relaxed);
    let final_epoch = base_epoch
        .checked_add(2)
        .expect("operation cache reset epoch exhausted");
    OPERATION_CACHE_RESET_EPOCH.store(base_epoch + 1, Ordering::Release);
    caches.clear();
    drop(caches);

    // Why not hold the registry lock while resetting lower layers: arbitrary
    // downstream cache initialization may acquire those locks before entering
    // this registry, creating the inverse lock order. The final phase removes
    // every cache registered during this deliberately unlocked interval.
    // Chain the tenet-core intern tables. Safe to run after the registry clear:
    // core content ids are monotonic and never reset, so no id survives here to
    // alias a fresh structure (see `reset_core_intern_tables`).
    tenet_core::reset_core_intern_tables();

    let mut caches = registry.write().expect("global cache registry poisoned");
    OPERATION_CACHE_RESET_EPOCH.store(final_epoch, Ordering::Release);
    caches.clear();
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
            policy: OperationCachePolicy::default(),
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
    use super::reset_global_operation_caches;
    use crate::test_support::CACHE_TEST_LOCK;
    use rustc_hash::FxHashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, OnceLock, RwLock};

    #[derive(Default)]
    struct DownstreamCacheProbe;

    #[test]
    fn downstream_cache_registration_tracks_global_reset_epoch() {
        // What: downstream cache payloads are shared before reset and replaced afterward.
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before_epoch = super::operation_cache_reset_epoch();
        let (first_epoch, first) = super::registered_operation_cache::<DownstreamCacheProbe>();
        let (same_epoch, same) = super::registered_operation_cache::<DownstreamCacheProbe>();
        assert_eq!(first_epoch, before_epoch);
        assert_eq!(same_epoch, before_epoch);
        assert!(Arc::ptr_eq(&first, &same));

        reset_global_operation_caches();

        let after_epoch = super::operation_cache_reset_epoch();
        let (replaced_epoch, replaced) =
            super::registered_operation_cache::<DownstreamCacheProbe>();
        assert!(after_epoch > before_epoch);
        assert_eq!(replaced_epoch, after_epoch);
        assert!(!Arc::ptr_eq(&first, &replaced));
    }

    #[test]
    fn typed_map_is_the_registered_typed_map_entry() {
        // What: the map helper and public registration API resolve one typed entry.
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        reset_global_operation_caches();
        let map = super::typed_global_map::<u8, u16>();
        let (_, registered) = super::registered_operation_cache::<RwLock<FxHashMap<u8, u16>>>();

        assert!(Arc::ptr_eq(&map, &registered));
    }

    #[derive(Default)]
    struct NestedCacheProbe;

    struct RegisteringCacheProbe {
        nested: Arc<NestedCacheProbe>,
    }

    impl Default for RegisteringCacheProbe {
        fn default() -> Self {
            Self {
                nested: super::registered_operation_cache::<NestedCacheProbe>().1,
            }
        }
    }

    #[test]
    fn registered_cache_default_may_register_another_cache() {
        // What: composing registered caches during initialization completes without deadlock.
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        reset_global_operation_caches();
        let (_, outer) = super::registered_operation_cache::<RegisteringCacheProbe>();
        let (_, nested) = super::registered_operation_cache::<NestedCacheProbe>();

        assert!(Arc::ptr_eq(&outer.nested, &nested));
    }

    static CROSS_RESET_DEFAULT_CALLS: AtomicUsize = AtomicUsize::new(0);
    static CROSS_RESET_DEFAULT_STARTED: OnceLock<Barrier> = OnceLock::new();
    static CROSS_RESET_DEFAULT_RESUME: OnceLock<Barrier> = OnceLock::new();

    #[derive(Default)]
    struct CrossResetNestedProbe;

    struct CrossResetCacheProbe {
        initialized_epoch: u64,
        nested: Arc<CrossResetNestedProbe>,
    }

    impl Default for CrossResetCacheProbe {
        fn default() -> Self {
            let initialized_epoch = super::operation_cache_reset_epoch();
            let nested = super::registered_operation_cache::<CrossResetNestedProbe>().1;
            if CROSS_RESET_DEFAULT_CALLS.fetch_add(1, Ordering::SeqCst) == 0 {
                CROSS_RESET_DEFAULT_STARTED
                    .get_or_init(|| Barrier::new(2))
                    .wait();
                CROSS_RESET_DEFAULT_RESUME
                    .get_or_init(|| Barrier::new(2))
                    .wait();
            }
            Self {
                initialized_epoch,
                nested,
            }
        }
    }

    #[test]
    fn cache_built_across_reset_is_rebuilt_in_the_final_generation() {
        // What: a candidate spanning reset cannot retain dependencies from the old generation.
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        reset_global_operation_caches();
        let worker =
            std::thread::spawn(|| super::registered_operation_cache::<CrossResetCacheProbe>());
        CROSS_RESET_DEFAULT_STARTED
            .get_or_init(|| Barrier::new(2))
            .wait();

        reset_global_operation_caches();
        CROSS_RESET_DEFAULT_RESUME
            .get_or_init(|| Barrier::new(2))
            .wait();

        let (epoch, cache) = worker.join().expect("registration worker panicked");
        let (_, current_nested) = super::registered_operation_cache::<CrossResetNestedProbe>();
        assert_eq!(cache.initialized_epoch, epoch);
        assert!(Arc::ptr_eq(&cache.nested, &current_nested));
        assert!(CROSS_RESET_DEFAULT_CALLS.load(Ordering::SeqCst) >= 2);
    }

    #[derive(Default)]
    struct ConcurrentCacheProbe;

    #[test]
    fn concurrent_registration_observes_a_stable_final_reset_generation() {
        // What: registrations racing both reset phases cannot survive the completed reset.
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        reset_global_operation_caches();
        let start_epoch = super::operation_cache_reset_epoch();
        let barrier = Arc::new(Barrier::new(6));
        let worker_barrier = Arc::clone(&barrier);
        let worker = std::thread::spawn(move || {
            worker_barrier.wait();
            let mut observed = Vec::with_capacity(2_048);
            for _ in 0..2_048 {
                observed.push(super::registered_operation_cache::<ConcurrentCacheProbe>());
                std::thread::yield_now();
            }
            observed
        });
        let reset_workers = (0..4)
            .map(|_| {
                let reset_barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    reset_barrier.wait();
                    for _ in 0..8 {
                        reset_global_operation_caches();
                        std::thread::yield_now();
                    }
                })
            })
            .collect::<Vec<_>>();

        barrier.wait();
        for reset_worker in reset_workers {
            reset_worker.join().expect("reset worker panicked");
        }
        let observed = worker.join().expect("registration worker panicked");
        let (final_epoch, final_cache) =
            super::registered_operation_cache::<ConcurrentCacheProbe>();
        let (same_epoch, same_cache) = super::registered_operation_cache::<ConcurrentCacheProbe>();

        assert_eq!(final_epoch, start_epoch + 64);
        assert_eq!(same_epoch, final_epoch);
        assert!(Arc::ptr_eq(&final_cache, &same_cache));
        for (epoch, cache) in observed {
            if epoch == final_epoch {
                assert!(Arc::ptr_eq(&cache, &final_cache));
            }
        }
    }
    use tenet_core::BlockStructure;

    #[test]
    fn reset_global_operation_caches_chains_core_intern_reset_without_id_reuse() {
        // What: this resets the process-global caches, which races any test
        // that assumes the shared tenet-core intern table stays stable across
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
        let id_before = BlockStructure::trivial(&[base]).unwrap().content_id();
        reset_global_operation_caches();
        let id_after = BlockStructure::trivial(&[base]).unwrap().content_id();
        assert!(
            id_after > id_before,
            "reset chain must not reuse content ids, got before={id_before} after={id_after}"
        );
    }
}
