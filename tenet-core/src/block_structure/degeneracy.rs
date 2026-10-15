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
        let shapes = shapes.into_iter();
        let mut structure = Self::with_block_capacity(rank, shapes.size_hint().0);
        let mut offset = 0usize;
        for shape in shapes {
            let shape = shape.into();
            if shape.len() != rank {
                return Err(CoreError::StructureRankMismatch {
                    expected: rank,
                    actual: shape.len(),
                });
            }
            offset = structure.push_block_with(|dims| {
                dims.extend_from_slice(&shape);
                extend_column_major_strides(&shape, dims)?;
                Ok(offset)
            })?;
        }
        Ok(structure)
    }

    pub fn from_blocks_with_rank(
        rank: usize,
        blocks: Vec<DegeneracyBlock>,
    ) -> Result<Self, CoreError> {
        let mut structure = Self::with_block_capacity(rank, blocks.len());
        for block in &blocks {
            structure.push_block(block.shape(), block.strides(), block.offset())?;
        }
        Ok(structure)
    }

    /// An empty structure whose arenas hold exactly `blocks` blocks.
    pub(crate) fn with_block_capacity(rank: usize, blocks: usize) -> Self {
        Self {
            rank,
            dims: Vec::with_capacity(blocks.saturating_mul(2 * rank)),
            offsets: Vec::with_capacity(blocks),
        }
    }

    /// Appends one block, copying its shape and strides into the arena.
    pub(crate) fn push_block(
        &mut self,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Result<usize, CoreError> {
        if shape.len() != self.rank {
            return Err(CoreError::StructureRankMismatch {
                expected: self.rank,
                actual: shape.len(),
            });
        }
        if strides.len() != shape.len() {
            return Err(CoreError::RankMismatch {
                shape: shape.len(),
                strides: strides.len(),
            });
        }
        self.push_block_with(|dims| {
            dims.extend_from_slice(shape);
            dims.extend_from_slice(strides);
            Ok(offset)
        })
    }

    /// Appends one block that `write` emits in place: it pushes the block's
    /// shape then its strides onto the arena and returns the block's offset.
    /// Returns the block's exclusive storage end.
    ///
    /// Why the builder writes into the arena rather than handing over a
    /// [`DegeneracyBlock`]: that block owns two rank-sized `DimVec`s, which
    /// spill to the heap per block above rank 8 (#2012).
    pub(crate) fn push_block_with(
        &mut self,
        write: impl FnOnce(&mut Vec<usize>) -> Result<usize, CoreError>,
    ) -> Result<usize, CoreError> {
        let start = self.dims.len();
        let offset = write(&mut self.dims)?;
        let block = &self.dims[start..];
        if block.len() != 2 * self.rank {
            return Err(CoreError::RankMismatch {
                shape: self.rank,
                strides: block.len().saturating_sub(self.rank),
            });
        }
        let (shape, strides) = block.split_at(self.rank);
        let end = storage_end_exclusive(shape, strides, offset)?;
        self.offsets.push(offset);
        Ok(end)
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
