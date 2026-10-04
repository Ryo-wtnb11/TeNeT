use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DegeneracyBlock {
    pub(super) shape: DimVec,
    pub(super) strides: DimVec,
    offset: usize,
}

impl DegeneracyBlock {
    pub fn new(shape: DimVec, strides: DimVec, offset: usize) -> Result<Self, CoreError> {
        if shape.len() != strides.len() {
            return Err(CoreError::RankMismatch {
                shape: shape.len(),
                strides: strides.len(),
            });
        }
        storage_end_exclusive(&shape, &strides, offset)?;
        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    pub fn column_major(shape: Vec<usize>, offset: usize) -> Result<Self, CoreError> {
        let strides = column_major_strides(&shape)?;
        Self::new(
            shape.into_iter().collect(),
            strides.into_iter().collect(),
            offset,
        )
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

    pub fn element_count(&self) -> Result<usize, CoreError> {
        checked_product(&self.shape)
    }

    pub fn storage_end_exclusive(&self) -> Result<usize, CoreError> {
        storage_end_exclusive(&self.shape, &self.strides, self.offset)
    }
}

/// One block's degeneracy layout, borrowed from a [`DegeneracyStructure`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DegeneracyBlockRef<'a> {
    shape: &'a [usize],
    strides: &'a [usize],
    offset: usize,
}

impl<'a> DegeneracyBlockRef<'a> {
    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    #[inline]
    pub fn strides(&self) -> &'a [usize] {
        self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn element_count(&self) -> Result<usize, CoreError> {
        checked_product(self.shape)
    }

    pub fn storage_end_exclusive(&self) -> Result<usize, CoreError> {
        storage_end_exclusive(self.shape, self.strides, self.offset)
    }
}

/// The degeneracy layouts of a block structure's blocks.
///
/// Why one flat arena rather than a [`DegeneracyBlock`] per block (#1998):
/// each block's two inline `DimVec`s cost 144 bytes at any rank and spill to
/// the heap above rank 8, while a block's shape and strides are `2 * rank`
/// words. TensorKit's `DegeneracyStructure` likewise stores one
/// `StridedStructure` tuple per subblock.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DegeneracyStructure {
    pub(super) rank: usize,
    /// Per block, its shape then its strides.
    pub(super) dims: Vec<usize>,
    pub(super) offsets: Vec<usize>,
}

impl DegeneracyStructure {
    pub fn packed_column_major<I>(rank: usize, shapes: I) -> Result<Self, CoreError>
    where
        I: IntoIterator,
        I::Item: Into<Vec<usize>>,
    {
        let mut offset = 0usize;
        let mut blocks = Vec::new();
        for shape in shapes {
            let shape = shape.into();
            if shape.len() != rank {
                return Err(CoreError::StructureRankMismatch {
                    expected: rank,
                    actual: shape.len(),
                });
            }
            let block = DegeneracyBlock::column_major(shape, offset)?;
            offset = block.storage_end_exclusive()?;
            blocks.push(block);
        }
        Self::from_blocks_with_rank(rank, blocks)
    }

    pub fn from_blocks_with_rank(
        rank: usize,
        blocks: Vec<DegeneracyBlock>,
    ) -> Result<Self, CoreError> {
        let mut dims = Vec::with_capacity(blocks.len().saturating_mul(2 * rank));
        let mut offsets = Vec::with_capacity(blocks.len());
        for block in &blocks {
            if block.shape().len() != rank {
                return Err(CoreError::StructureRankMismatch {
                    expected: rank,
                    actual: block.shape().len(),
                });
            }
            block.storage_end_exclusive()?;
            dims.extend_from_slice(block.shape());
            dims.extend_from_slice(block.strides());
            offsets.push(block.offset());
        }
        Ok(Self {
            rank,
            dims,
            offsets,
        })
    }

    pub(super) fn empty(rank: usize) -> Self {
        Self {
            rank,
            dims: Vec::new(),
            offsets: Vec::new(),
        }
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.offsets.len()
    }

    #[inline]
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = DegeneracyBlockRef<'_>> + '_ {
        (0..self.block_count()).map(|index| self.block_unchecked(index))
    }

    #[inline]
    fn block_unchecked(&self, index: usize) -> DegeneracyBlockRef<'_> {
        let start = index * 2 * self.rank;
        let (shape, strides) = self.dims[start..start + 2 * self.rank].split_at(self.rank);
        DegeneracyBlockRef {
            shape,
            strides,
            offset: self.offsets[index],
        }
    }

    pub fn block(&self, index: usize) -> Result<DegeneracyBlockRef<'_>, CoreError> {
        if index >= self.block_count() {
            return Err(CoreError::BlockIndexOutOfBounds {
                index,
                count: self.block_count(),
            });
        }
        Ok(self.block_unchecked(index))
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.blocks().try_fold(0usize, |required, block| {
            Ok(required.max(block.storage_end_exclusive()?))
        })
    }

    /// Heap bytes of the layout arenas.
    pub(super) fn charged_heap_bytes(&self) -> usize {
        self.dims
            .capacity()
            .saturating_mul(std::mem::size_of::<usize>())
            .saturating_add(
                self.offsets
                    .capacity()
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
    }
}
