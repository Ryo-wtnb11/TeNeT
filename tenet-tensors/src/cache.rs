use std::any::{Any, TypeId};
use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock, RwLock};

use tenet_core::{BlockStructure, BlockStructureContent, BlockStructureContentBlock};

use crate::{OperationError, TensorContractStructure, TreeTransformStructure};

type ErasedGlobalCache = Arc<dyn Any + Send + Sync>;
type GlobalCacheRegistry = RwLock<FxHashMap<TypeId, ErasedGlobalCache>>;

pub(crate) fn operation_global_registry() -> &'static GlobalCacheRegistry {
    static REGISTRY: OnceLock<GlobalCacheRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(FxHashMap::default()))
}

pub(crate) fn typed_global_map<K, V>(
    registry: &'static GlobalCacheRegistry,
) -> Arc<RwLock<FxHashMap<K, V>>>
where
    K: 'static + Eq + Hash + Send + Sync,
    V: 'static + Send + Sync,
{
    let type_id = TypeId::of::<(K, V)>();
    if let Some(cache) = registry
        .read()
        .expect("global cache registry poisoned")
        .get(&type_id)
    {
        return Arc::downcast::<RwLock<FxHashMap<K, V>>>(Arc::clone(cache))
            .expect("global cache registry type id collision");
    }

    let mut caches = registry.write().expect("global cache registry poisoned");
    if let Some(cache) = caches.get(&type_id) {
        return Arc::downcast::<RwLock<FxHashMap<K, V>>>(Arc::clone(cache))
            .expect("global cache registry type id collision");
    }
    let cache = Arc::new(RwLock::new(FxHashMap::<K, V>::default()));
    caches.insert(type_id, Arc::clone(&cache) as ErasedGlobalCache);
    cache
}

pub fn reset_global_operation_caches() {
    operation_global_registry()
        .write()
        .expect("global cache registry poisoned")
        .clear();
    crate::tree_transform::reset_tree_transform_persistent_cache_state();
}

/// Cache policy for TensorKit-style replay caches.
///
/// `TaskLocal` means the cache is owned by an explicit execution context. Keeping
/// one context per task mirrors TensorKit's task-local cache; sharing the same
/// context/cache handle from a process-level owner gives the corresponding
/// global cache without hiding synchronization in ordinary tensor operations.
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

#[derive(Clone, Debug)]
pub struct TreeTransformStructureCache<T, PlanKey> {
    structures: FxHashMap<TreeTransformStructureCacheKey<PlanKey>, Arc<TreeTransformStructure<T>>>,
    lru_order: VecDeque<TreeTransformStructureCacheKey<PlanKey>>,
    policy: OperationCachePolicy,
}

impl<T, PlanKey> Default for TreeTransformStructureCache<T, PlanKey> {
    fn default() -> Self {
        Self {
            structures: FxHashMap::default(),
            lru_order: VecDeque::new(),
            policy: OperationCachePolicy::default(),
        }
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
        key: &TreeTransformStructureCacheKey<PlanKey>,
    ) -> Option<&TreeTransformStructure<T>> {
        self.structures.get(key).map(Arc::as_ref)
    }

    pub fn get_arc(
        &self,
        key: &TreeTransformStructureCacheKey<PlanKey>,
    ) -> Option<Arc<TreeTransformStructure<T>>> {
        self.structures.get(key).map(Arc::clone)
    }

    pub fn touch(&mut self, key: &TreeTransformStructureCacheKey<PlanKey>) {
        if self.policy.max_entries().is_some() && self.structures.contains_key(key) {
            touch_lru_key(&mut self.lru_order, key);
        }
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
