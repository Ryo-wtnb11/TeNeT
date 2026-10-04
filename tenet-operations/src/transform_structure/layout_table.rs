use super::*;

/// One prebaked fused layout's location in the arena (issue #232).
///
/// `rank == 0` is the "absent" sentinel: normalized layouts always have
/// rank >= 1, so a zero-rank slot means the entry was never baked (a
/// Single-block source layout, looked up only via its destination twin, or an
/// inactive destination). The compact 32-bit fields match QSpace's runtime-rank
/// metadata without making a small-rank execution split.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct FusedSlot {
    start: u32,
    rank: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TreeTransformLayoutTable {
    pub(super) entries: Vec<TreeTransformLayout>,
    pub(super) shapes: Vec<usize>,
    pub(super) strides: Vec<isize>,
    pub(super) packed_strides: Vec<isize>,
    // Baked fused loop layouts (issue #232): SoA arena mirroring shapes/strides
    // above, indexed per (entry, role) through `fused_slots`. Populated once at
    // compile time so replay skips layout normalization.
    pub(super) fused_dims: Vec<usize>,
    pub(super) fused_dst_strides: Vec<isize>,
    pub(super) fused_src_strides: Vec<isize>,
    pub(super) fused_slots: Vec<FusedSlot>,
    pub(super) max_fused_rank: usize,
}

impl TreeTransformLayoutTable {
    pub(super) fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Prebaked fused layout for `entry_index`, or `None` when the entry was
    /// not baked (see [`FusedSlot`]) and the caller must recompute. Returned
    /// slices are the exact normalized `(dims, dst_strides, src_strides)` that
    /// the production normalizer produced for that entry's role.
    pub(crate) fn fused_baked(&self, entry_index: usize) -> Option<BakedFusedLayout<'_>> {
        let slot = self.fused_slots.get(entry_index).copied()?;
        if slot.rank == 0 {
            return None;
        }
        let start = slot.start as usize;
        let end = start + slot.rank as usize;
        Some(BakedFusedLayout::from_compiled_normalized_slices(
            &self.fused_dims[start..end],
            &self.fused_dst_strides[start..end],
            &self.fused_src_strides[start..end],
        ))
    }

    /// Heap bytes of the base layout metadata. Test/diagnostic API.
    #[doc(hidden)]
    pub fn layout_table_bytes(&self) -> usize {
        self.entries.len() * core::mem::size_of::<TreeTransformLayout>()
            + self.shapes.len() * core::mem::size_of::<usize>()
            + (self.strides.len() + self.packed_strides.len()) * core::mem::size_of::<isize>()
    }

    /// Heap bytes of the compiled fused-layout arena. Test/diagnostic API.
    #[doc(hidden)]
    pub fn baked_arena_bytes(&self) -> usize {
        self.fused_dims.len() * core::mem::size_of::<usize>()
            + (self.fused_dst_strides.len() + self.fused_src_strides.len())
                * core::mem::size_of::<isize>()
            + self.fused_slots.len() * core::mem::size_of::<FusedSlot>()
    }

    pub(crate) fn max_fused_rank(&self) -> usize {
        self.max_fused_rank
    }

    /// Bakes the fused layout of `entry_index` for the `dst_strides`/`src_strides`
    /// pair of its role (single/pack/scatter).
    fn bake_entry(
        &mut self,
        entry_index: usize,
        dst_is_packed: bool,
        src_is_packed: bool,
        scratch: &mut FusedLayoutScratch,
    ) -> Result<(), OperationError> {
        let layout = &self.entries[entry_index];
        let range = layout.range();
        let dst_strides = if dst_is_packed {
            self.packed_strides(layout)
        } else {
            &self.strides[range.clone()]
        };
        let src_strides = if src_is_packed {
            self.packed_strides(layout)
        } else {
            &self.strides[range.clone()]
        };
        normalize_fused_layout(&self.shapes[range], dst_strides, src_strides, scratch)?;
        self.push_baked(entry_index, scratch)
    }

    /// Bakes a Single block's fused layout at its destination entry, combining
    /// the destination entry's strides with the source entry's strides (both
    /// share the same shape, validated at compile). Looked up via the
    /// destination index only.
    fn bake_single(
        &mut self,
        dst_entry: usize,
        src_entry: usize,
        scratch: &mut FusedLayoutScratch,
    ) -> Result<(), OperationError> {
        let dst = &self.entries[dst_entry];
        let (ds, dr) = (dst.layout_start, dst.rank as usize);
        let ss = self.entries[src_entry].layout_start;
        normalize_fused_layout(
            &self.shapes[ds..ds + dr],
            &self.strides[ds..ds + dr],
            &self.strides[ss..ss + dr],
            scratch,
        )?;
        self.push_baked(dst_entry, scratch)
    }

    fn push_baked(
        &mut self,
        entry_index: usize,
        fused: &FusedLayoutScratch,
    ) -> Result<(), OperationError> {
        BakedFusedLayout::try_from_normalized_slices(
            fused.dims(),
            fused.dst_strides(),
            fused.src_strides(),
        )?;
        let start = u32::try_from(self.fused_dims.len())
            .map_err(|_| OperationError::ElementCountOverflow)?;
        let rank =
            u32::try_from(fused.dims().len()).map_err(|_| OperationError::ElementCountOverflow)?;
        start
            .checked_add(rank)
            .ok_or(OperationError::ElementCountOverflow)?;
        self.max_fused_rank = self.max_fused_rank.max(rank as usize);
        self.fused_dims.extend_from_slice(fused.dims());
        self.fused_dst_strides
            .extend_from_slice(fused.dst_strides());
        self.fused_src_strides
            .extend_from_slice(fused.src_strides());
        if entry_index >= self.fused_slots.len() {
            self.fused_slots
                .resize(entry_index + 1, FusedSlot::default());
        }
        self.fused_slots[entry_index] = FusedSlot { start, rank };
        Ok(())
    }

    /// Differential self-check: every baked role equals the single production
    /// normalizer applied to that role's stride pair.
    #[doc(hidden)]
    pub fn baked_matches_recomputed(&self, blocks: &[TreeTransformBlock]) -> bool {
        let matches = |entry_index: usize, shape: &[usize], dst: &[isize], src: &[isize]| {
            let Some(baked) = self.fused_baked(entry_index) else {
                return false;
            };
            let mut scratch = FusedLayoutScratch::default();
            if normalize_fused_layout(shape, dst, src, &mut scratch).is_err() {
                return false;
            }
            baked.dims() == scratch.dims()
                && baked.dst_strides() == scratch.dst_strides()
                && baked.src_strides() == scratch.src_strides()
        };
        for block in blocks {
            match *block {
                TreeTransformBlock::Single {
                    dst_layout,
                    src_layout,
                    ..
                } => {
                    let dst = self.entry(dst_layout);
                    let src = self.entry(src_layout);
                    if !matches(
                        dst_layout,
                        self.shape(dst),
                        self.strides(dst),
                        self.strides(src),
                    ) {
                        return false;
                    }
                }
                TreeTransformBlock::Multi {
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    ..
                } => {
                    for index in src_layout_start..src_layout_start + src_count {
                        let entry = self.entry(index);
                        if !matches(
                            index,
                            self.shape(entry),
                            self.packed_strides(entry),
                            self.strides(entry),
                        ) {
                            return false;
                        }
                    }
                    for index in dst_layout_start..dst_layout_start + dst_count {
                        let entry = self.entry(index);
                        if !matches(
                            index,
                            self.shape(entry),
                            self.strides(entry),
                            self.packed_strides(entry),
                        ) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    pub fn entry(&self, index: usize) -> &TreeTransformLayout {
        &self.entries[index]
    }

    pub fn shape(&self, layout: &TreeTransformLayout) -> &[usize] {
        &self.shapes[layout.range()]
    }

    pub fn strides(&self, layout: &TreeTransformLayout) -> &[isize] {
        &self.strides[layout.range()]
    }

    /// Column-major strides of the packed column of a `Multi` block's entry.
    ///
    /// # Panics
    ///
    /// For an entry no `Multi` block packs or scatters (a `Single` block's or
    /// an inactive destination's); no replay path reads those.
    pub fn packed_strides(&self, layout: &TreeTransformLayout) -> &[isize] {
        let start = layout.packed_start as usize;
        &self.packed_strides[start..start + layout.rank as usize]
    }

    /// Reserves exact capacity for `entries` layouts of `rank` axes, of which
    /// `packed_entries` belong to `Multi` blocks. Why exact: a retaining cache
    /// charges capacity, and amortized growth left up to half of it unused.
    pub(super) fn reserve_exact(&mut self, entries: usize, rank: usize, packed_entries: usize) {
        self.entries.reserve_exact(entries);
        self.shapes.reserve_exact(entries.saturating_mul(rank));
        self.strides.reserve_exact(entries.saturating_mul(rank));
        self.packed_strides
            .reserve_exact(packed_entries.saturating_mul(rank));
    }

    /// Populates the baked arena for every replayed (entry, role): each Single
    /// block's fused single layout at its destination entry, each Multi source
    /// entry's pack layout (dst = packed column, src = block strides), each Multi
    /// destination entry's scatter layout (dst = block strides, src = packed
    /// column). Inactive destinations and Single source entries are intentionally
    /// unbaked — the former never fuse-copy from a source, the latter are only
    /// reached through their destination twin. Block order after the replay sort
    /// is irrelevant: baking is keyed by stable entry index.
    pub(super) fn bake_fused_layouts(
        &mut self,
        blocks: &[TreeTransformBlock],
    ) -> Result<(), OperationError> {
        // Reserve the arena once so baking adds a bounded, block-count-independent
        // number of allocations rather than growing per push. Fused rank never
        // exceeds an entry's rank, except that a rank-0 entry fuses to one unit
        // axis, so the baked entries' ranks (at least 1) bound each arena;
        // a Single block's source entry and an inactive destination are never
        // baked, so they reserve nothing (#1998). One slot per entry covers
        // `fused_slots`.
        let baked_ranks = blocks.iter().fold(0usize, |total, block| {
            total.saturating_add(match *block {
                TreeTransformBlock::Single { dst_layout, .. } => {
                    (self.entries[dst_layout].rank as usize).max(1)
                }
                TreeTransformBlock::Multi {
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    ..
                } => self.entries[src_layout_start..src_layout_start + src_count]
                    .iter()
                    .chain(&self.entries[dst_layout_start..dst_layout_start + dst_count])
                    .map(|entry| (entry.rank as usize).max(1))
                    .sum(),
            })
        });
        self.fused_slots
            .resize(self.entries.len(), FusedSlot::default());
        self.fused_dims.reserve_exact(baked_ranks);
        self.fused_dst_strides.reserve_exact(baked_ranks);
        self.fused_src_strides.reserve_exact(baked_ranks);
        let mut scratch = FusedLayoutScratch::default();
        for block in blocks {
            match *block {
                TreeTransformBlock::Single {
                    dst_layout,
                    src_layout,
                    ..
                } => self.bake_single(dst_layout, src_layout, &mut scratch)?,
                TreeTransformBlock::Multi {
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    ..
                } => {
                    for index in src_layout_start..src_layout_start + src_count {
                        self.bake_entry(index, true, false, &mut scratch)?;
                    }
                    for index in dst_layout_start..dst_layout_start + dst_count {
                        self.bake_entry(index, false, true, &mut scratch)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Appends one layout entry; `packed` gives it packed strides, for an entry
    /// of a `Multi` block.
    pub(super) fn push_block(
        &mut self,
        rank: usize,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        packed: bool,
    ) -> Result<usize, OperationError> {
        self.push_block_mapped(rank, shape, strides, offset, None, packed)
    }

    fn push_block_mapped(
        &mut self,
        rank: usize,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        axes: Option<&[usize]>,
        packed: bool,
    ) -> Result<usize, OperationError> {
        if shape.len() != rank {
            return Err(OperationError::RankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
        if strides.len() != rank {
            return Err(OperationError::RankMismatch {
                expected: rank,
                actual: strides.len(),
            });
        }
        let axis = |index: usize| axes.map_or(index, |axes| axes[index]);
        let element_count = (0..rank).try_fold(1usize, |count, index| {
            count
                .checked_mul(shape[axis(index)])
                .ok_or(OperationError::ElementCountOverflow)
        })?;
        let mut packed_stride = 1usize;
        for index in 0..rank {
            isize::try_from(packed_stride).map_err(|_| OperationError::StrideOverflow {
                value: packed_stride,
            })?;
            packed_stride = packed_stride
                .checked_mul(shape[axis(index)])
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        for index in 0..rank {
            let stride = strides[axis(index)];
            isize::try_from(stride)
                .map_err(|_| OperationError::StrideOverflow { value: stride })?;
        }
        let offset = offset_to_isize(offset)?;
        let overflow = |_| OperationError::ElementCountOverflow;
        let rank_u32 = u32::try_from(rank).map_err(overflow)?;
        let packed_start = if packed {
            let start = u32::try_from(self.packed_strides.len()).map_err(overflow)?;
            if start == NOT_PACKED {
                return Err(OperationError::ElementCountOverflow);
            }
            start
        } else {
            NOT_PACKED
        };

        let layout_start = self.shapes.len();
        let mut packed_stride = 1usize;
        for index in 0..rank {
            let axis = axis(index);
            self.shapes.push(shape[axis]);
            self.strides.push(strides[axis] as isize);
            if packed {
                self.packed_strides.push(packed_stride as isize);
            }
            packed_stride *= shape[axis];
        }
        self.entries.push(TreeTransformLayout {
            layout_start,
            rank: rank_u32,
            packed_start,
            offset,
            element_count,
        });
        Ok(element_count)
    }

    pub(super) fn push_block_with_axes(
        &mut self,
        rank: usize,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        axes: Option<&[usize]>,
        packed: bool,
    ) -> Result<usize, OperationError> {
        let Some(axes) = axes else {
            return self.push_block(rank, shape, strides, offset, packed);
        };
        crate::axis::validate_permutation(axes, rank)?;
        self.push_block_mapped(rank, shape, strides, offset, Some(axes), packed)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformLayout {
    pub(super) layout_start: usize,
    pub(super) rank: u32,
    /// Start of this entry's packed strides, or [`NOT_PACKED`]. Why only
    /// `Multi` entries carry them (#1998): only pack and scatter read packed
    /// strides, and only a `Multi` block packs or scatters.
    pub(super) packed_start: u32,
    pub offset: isize,
    pub element_count: usize,
}

/// [`TreeTransformLayout::packed_start`] of an entry no `Multi` block packs.
pub(super) const NOT_PACKED: u32 = u32::MAX;

impl TreeTransformLayout {
    #[inline]
    fn range(&self) -> Range<usize> {
        self.layout_start..self.layout_start + self.rank as usize
    }
}
