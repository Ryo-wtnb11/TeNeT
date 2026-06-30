use std::collections::HashMap;
use std::hash::Hash;

use tenet_core::{BlockKey, BlockStructure};

use crate::{OperationError, TreeTransformStructure};

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
        Ok(Self {
            plan,
            dst: BlockStructureCacheKey::from_structure(dst_structure)?,
            src: BlockStructureCacheKey::from_structure(src_structure)?,
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
}

#[derive(Clone, Debug)]
pub struct TreeTransformStructureCache<T, PlanKey> {
    structures: HashMap<TreeTransformStructureCacheKey<PlanKey>, TreeTransformStructure<T>>,
}

impl<T, PlanKey> Default for TreeTransformStructureCache<T, PlanKey> {
    fn default() -> Self {
        Self {
            structures: HashMap::new(),
        }
    }
}

impl<T, PlanKey> TreeTransformStructureCache<T, PlanKey>
where
    PlanKey: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
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
        self.structures.get(key)
    }

    pub fn insert(
        &mut self,
        key: TreeTransformStructureCacheKey<PlanKey>,
        structure: TreeTransformStructure<T>,
    ) -> Option<TreeTransformStructure<T>> {
        self.structures.insert(key, structure)
    }
}
