use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

use tenet_core::{BlockKey, BlockStructure};

use crate::{OperationError, TensorContractStructure, TreeTransformStructure};

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
    map: &mut HashMap<K, V>,
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

pub(crate) fn rebuild_lru_order_from_keys<K, V>(map: &HashMap<K, V>, order: &mut VecDeque<K>)
where
    K: Clone,
{
    order.clear();
    order.extend(map.keys().cloned());
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct BlockStructureCacheKey {
    rank: usize,
    blocks: Vec<BlockStructureCacheBlockKey>,
}

impl BlockStructureCacheKey {
    pub fn from_structure(structure: &BlockStructure) -> Result<Self, OperationError> {
        let mut blocks = Vec::with_capacity(structure.block_count());
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            blocks.push(BlockStructureCacheBlockKey {
                key: block.key().clone(),
                shape: block.shape().to_vec(),
                strides: block.strides().to_vec(),
                offset: block.offset(),
            });
        }
        Ok(Self {
            rank: structure.rank(),
            blocks,
        })
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn blocks(&self) -> &[BlockStructureCacheBlockKey] {
        &self.blocks
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct BlockStructureCacheBlockKey {
    key: BlockKey,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl BlockStructureCacheBlockKey {
    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }
}

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
    structures: HashMap<TreeTransformStructureCacheKey<PlanKey>, Arc<TreeTransformStructure<T>>>,
    lru_order: VecDeque<TreeTransformStructureCacheKey<PlanKey>>,
    policy: OperationCachePolicy,
}

impl<T, PlanKey> Default for TreeTransformStructureCache<T, PlanKey> {
    fn default() -> Self {
        Self {
            structures: HashMap::new(),
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
    structures: HashMap<TensorContractStructureCacheKey<PlanKey>, TensorContractStructure<C>>,
    lru_order: VecDeque<TensorContractStructureCacheKey<PlanKey>>,
    policy: OperationCachePolicy,
}

impl<C, PlanKey> Default for TensorContractStructureCache<C, PlanKey> {
    fn default() -> Self {
        Self {
            structures: HashMap::new(),
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
            structures: HashMap::new(),
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
        self.structures.get(key)
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
    ) -> Option<TensorContractStructure<C>> {
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
            structures: HashMap::new(),
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
        if !self.policy.stores_entries() {
            return None;
        }
        let old = self.structures.insert(key.clone(), Arc::new(structure));
        if self.policy.max_entries().is_some() {
            touch_lru_key(&mut self.lru_order, &key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            enforce_lru_limit(&mut self.structures, &mut self.lru_order, max_entries);
        }
        old
    }
}
