use std::ops::Range;
use std::sync::Arc;

use tenet_core::{
    validate_block_storage_injective, BlockStructure, CoreError, TensorMap, TensorStorage,
};
use tenet_dense::DenseGemmBatchJob;

use crate::kernel_adapter::{normalize_fused_layout, BakedFusedLayout, FusedLayoutScratch};
use crate::strided::offset_to_isize;
use crate::structure_identity::validate_structure_identity;
use crate::transform_plan::{
    ResolvedTreeTransformBlockSpec, TreeTransformBlockSpec, TreeTransformKeyBlockSpec,
};
use crate::OperationError;

mod coefficients;
mod compile;
mod layout_table;
mod schedule;

pub(crate) use self::coefficients::*;
pub use self::compile::*;
pub use self::layout_table::*;
pub use self::schedule::*;

/// Replay-ready tree-transform descriptor.
///
/// This is the TensorKit-style transformer-build boundary: construction resolves
/// tree keys, block layouts, offsets, and pack/scatter descriptors against
/// concrete source and destination structures. Matrix-valued categorical
/// coefficients may be shared across compatible layout bindings. Hot paths
/// should build this once and replay it with `tree_transform_execute_with`
/// while reusing a backend and workspace.
///
/// Why not expose mutable compiled fields: the recoupling plan, converted
/// coefficient cache, and threaded replay schedule all derive from the same
/// descriptors. Read them through [`Self::blocks`], [`Self::layouts`], and
/// [`Self::block_coefficients`] so those derived plans cannot go stale after
/// compilation.
///
/// Coefficient layout: `TreeTransformBlock::Single::coefficient` and
/// `TreeTransformBlock::Multi::coefficient_start` index one logical payload
/// holding every Single scalar (spec order) followed by every Multi matrix
/// (spec order). Only the scalars are stored contiguously; each Multi block
/// references its spec's shared matrix, and every binding of one categorical
/// plan shares that plan's payload, so binding never copies a coefficient.
///
/// Migration: code that previously read the public `blocks` or `layouts`
/// fields must use the same-named accessor methods. The flat
/// `recoupling_coefficients_dst_src` view no longer exists: read one block
/// with [`Self::block_coefficients`], or copy the whole payload explicitly
/// with [`Self::gather_recoupling_coefficients_into`]. Post-compilation
/// mutation is no longer supported.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformStructure<T> {
    rank: usize,
    storage_conjugate: bool,
    identity: Arc<()>,
    blocks: Vec<TreeTransformBlock>,
    layouts: TreeTransformLayoutTable,
    coefficients: Arc<TreeTransformCoefficients<T>>,
    inactive_dst_layouts: Vec<usize>,
    physical_overwrite_len: Option<usize>,
    recoupling_plan: TreeTransformRecouplingPlan,
    parallel_schedule: TreeTransformParallelSchedule,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
}

impl<T> TreeTransformStructure<T> {
    /// Conservative retained-byte charge for this immutable replay payload.
    ///
    /// This counts owned capacities and the structure identity control block.
    /// The source and destination structures are separate canonical owners, so
    /// only their inline `Arc` handles are included through `size_of::<Self>()`.
    /// Shared categorical recoupling matrices are conservatively charged in
    /// full for every bound structure; aggregate unique-byte accounting belongs
    /// with a future retention policy, not this ownership split.
    #[doc(hidden)]
    pub fn charged_payload_bytes(&self) -> usize {
        const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();

        let vector_bytes =
            |capacity: usize, element_size: usize| capacity.saturating_mul(element_size);
        core::mem::size_of::<Self>()
            .saturating_add(ARC_CONTROL_BYTES)
            .saturating_add(vector_bytes(
                self.blocks.capacity(),
                core::mem::size_of::<TreeTransformBlock>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.entries.capacity(),
                core::mem::size_of::<TreeTransformLayout>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.shapes.capacity(),
                core::mem::size_of::<usize>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.strides.capacity(),
                core::mem::size_of::<isize>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.packed_strides.capacity(),
                core::mem::size_of::<isize>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.fused_dims.capacity(),
                core::mem::size_of::<usize>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.fused_dst_strides.capacity(),
                core::mem::size_of::<isize>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.fused_src_strides.capacity(),
                core::mem::size_of::<isize>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.fused_slots.capacity(),
                core::mem::size_of::<FusedSlot>(),
            ))
            .saturating_add(self.coefficients.charged_bytes())
            .saturating_add(vector_bytes(
                self.inactive_dst_layouts.capacity(),
                core::mem::size_of::<usize>(),
            ))
            .saturating_add(vector_bytes(
                self.recoupling_plan.block_indices.capacity(),
                core::mem::size_of::<usize>(),
            ))
            .saturating_add(vector_bytes(
                self.recoupling_plan.jobs.capacity(),
                core::mem::size_of::<DenseGemmBatchJob>(),
            ))
            .saturating_add(vector_bytes(
                self.parallel_schedule.singles.capacity(),
                core::mem::size_of::<TreeTransformSingleReplay>(),
            ))
            .saturating_add(vector_bytes(
                self.parallel_schedule.pack_columns.capacity(),
                core::mem::size_of::<TreeTransformPackReplay>(),
            ))
            .saturating_add(vector_bytes(
                self.parallel_schedule.scatter_columns.capacity(),
                core::mem::size_of::<TreeTransformScatterReplay>(),
            ))
            .saturating_add(vector_bytes(
                self.parallel_schedule.scatter_groups.capacity(),
                core::mem::size_of::<TreeTransformScatterGroupReplay>(),
            ))
    }
}

impl<T: Copy> TreeTransformStructure<T> {
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Immutable compiled block descriptors.
    #[inline]
    pub fn blocks(&self) -> &[TreeTransformBlock] {
        &self.blocks
    }

    /// Immutable compiled layout table.
    #[inline]
    pub fn layouts(&self) -> &TreeTransformLayoutTable {
        &self.layouts
    }

    /// Differential self-check: every baked fused layout matches a fresh run of
    /// the production normalizer for its (block, role) stride pair.
    /// Test-only.
    #[doc(hidden)]
    pub fn baked_layouts_match_recomputed(&self) -> bool {
        self.layouts.baked_matches_recomputed(&self.blocks)
    }

    /// Block `block_index`'s `U[dst, src]` coefficients, read in place: a
    /// Single block's scalar, or the Multi block's shared matrix.
    #[inline]
    pub fn block_coefficients(&self, block_index: usize) -> Option<&[T]> {
        match *self.blocks.get(block_index)? {
            TreeTransformBlock::Single { coefficient, .. } => {
                self.coefficients.singles.get(coefficient..=coefficient)
            }
            TreeTransformBlock::Multi { .. } => self.block_matrix(block_index),
        }
    }

    #[inline]
    pub(crate) fn block_matrix(&self, block_index: usize) -> Option<&[T]> {
        match *self.blocks.get(block_index)? {
            TreeTransformBlock::Single { .. } => None,
            TreeTransformBlock::Multi {
                dst_layout_start, ..
            } => self.coefficients.matrix(dst_layout_start),
        }
    }

    /// Scalars of the Single blocks, indexed by their `coefficient`.
    #[inline]
    pub(crate) fn single_coefficients(&self) -> &[T] {
        &self.coefficients.singles
    }

    /// Length of the logical coefficient payload the block offsets index.
    #[inline]
    pub fn coefficient_len(&self) -> usize {
        self.coefficients.len
    }

    /// Copies the logical destination-by-source coefficient payload into
    /// `out`, replacing its contents.
    ///
    /// This is an O([`Self::coefficient_len`]) copy; replay reads
    /// [`Self::block_coefficients`] in place instead.
    pub fn gather_recoupling_coefficients_into(&self, out: &mut Vec<T>)
    where
        T: Copy,
    {
        self.gather_converted_coefficients_into(out, |coefficient| coefficient);
    }

    /// The logical payload, converted: the scalars, then the Multi matrices in
    /// `coefficient_start` (= spec) order. The CUDA executor calls this once
    /// per device structure upload.
    pub(crate) fn gather_converted_coefficients_into<U>(
        &self,
        out: &mut Vec<U>,
        mut convert: impl FnMut(T) -> U,
    ) where
        T: Copy,
    {
        out.clear();
        out.reserve(self.coefficients.len);
        out.extend(
            self.coefficients
                .singles
                .iter()
                .map(|&value| convert(value)),
        );
        for matrix in &self.coefficients.matrices {
            out.extend(matrix.iter().map(|&value| convert(value)));
        }
    }

    pub fn workspace_lens(&self) -> (usize, usize) {
        self.blocks
            .iter()
            .fold((0, 0), |(max_src, max_dst), block| match block {
                TreeTransformBlock::Single { .. } => (max_src, max_dst),
                TreeTransformBlock::Multi {
                    dst_count,
                    src_count,
                    element_count,
                    ..
                } => (
                    max_src.max(element_count.saturating_mul(*src_count)),
                    max_dst.max(element_count.saturating_mul(*dst_count)),
                ),
            })
    }

    pub fn workspace_len(&self) -> usize {
        let (source, destination) = self.workspace_lens();
        source.max(destination)
    }

    pub fn has_pack_gemm_scatter_blocks(&self) -> bool {
        !self.recoupling_plan.is_empty()
    }

    #[inline]
    pub(crate) fn identity_marker(&self) -> &Arc<()> {
        &self.identity
    }

    #[cfg(test)]
    pub(crate) fn gathered_coefficients(&self) -> Vec<T>
    where
        T: Copy,
    {
        let mut coefficients = Vec::new();
        self.gather_recoupling_coefficients_into(&mut coefficients);
        coefficients
    }

    #[cfg(test)]
    pub(crate) fn shares_recoupling_matrices_with(&self, other: &Self) -> bool {
        !self.coefficients.matrices.is_empty()
            && Arc::ptr_eq(&self.coefficients, &other.coefficients)
    }

    #[inline]
    pub fn recoupling_plan(&self) -> &TreeTransformRecouplingPlan {
        &self.recoupling_plan
    }

    /// Test/diagnostic helper: per-block replay weights.
    #[doc(hidden)]
    pub fn replay_weights(&self) -> Vec<usize> {
        self.blocks
            .iter()
            .map(|block| tree_transform_block_weight(block, &self.layouts))
            .collect()
    }

    #[inline]
    pub fn storage_conjugate(&self) -> bool {
        self.storage_conjugate
    }

    /// Scalar of a [`TreeTransformBlock::Single`] block: `index` is that
    /// block's `coefficient`. `None` when `index` is not a Single scalar
    /// (Multi matrices are read with [`Self::block_coefficients`]).
    #[inline]
    pub fn single_coefficient(&self, index: usize) -> Option<T> {
        self.coefficients.singles.get(index).copied()
    }

    #[inline]
    pub(crate) fn inactive_destination_layouts(&self) -> &[usize] {
        &self.inactive_dst_layouts
    }

    #[inline]
    pub(crate) fn physical_overwrite_len(&self) -> Option<usize> {
        self.physical_overwrite_len
    }

    #[inline]
    pub(crate) fn parallel_schedule(&self) -> &TreeTransformParallelSchedule {
        &self.parallel_schedule
    }

    pub fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("src", &self.src_structure, src_structure)
    }

    pub(crate) fn replay_storage_lens(&self) -> Result<(usize, usize), OperationError> {
        Ok((
            self.src_structure.required_len()?,
            self.dst_structure.required_len()?,
        ))
    }

    /// Rebinds an already compiled descriptor to content-equal canonical
    /// structures without rebuilding its replay payload.
    #[doc(hidden)]
    pub fn with_canonical_structures(
        mut self,
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
    ) -> Result<Self, OperationError> {
        validate_structure_identity("dst", &self.dst_structure, &dst_structure)?;
        validate_structure_identity("src", &self.src_structure, &src_structure)?;
        self.dst_structure = dst_structure;
        self.src_structure = src_structure;
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TreeTransformBlock {
    Single {
        dst_layout: usize,
        src_layout: usize,
        coefficient: usize,
    },
    Multi {
        dst_layout_start: usize,
        dst_count: usize,
        src_layout_start: usize,
        src_count: usize,
        coefficient_start: usize,
        element_count: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenet_core::{BlockKey, BlockSpec};

    #[test]
    fn compiled_fused_layouts_match_the_production_normalizer() {
        // What: a runtime-rank layout is admitted and bakes the same normalized
        // layout that eager replay derives from its shapes and strides.
        let shape = vec![2; 9];
        let strides = (0..9).map(|axis| 1usize << axis).collect::<Vec<_>>();
        let block = BlockSpec::with_key(BlockKey::ordinal(0), shape, strides, 0).unwrap();
        let structure = BlockStructure::from_blocks_with_rank(9, vec![block]).unwrap();
        let specs =
            vec![TreeTransformBlockSpec::single(0, 0, 1.0_f64).with_source_axes((0..9).rev())];
        let compiled =
            TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
        assert!(!compiled.has_pack_gemm_scatter_blocks());
        assert!(compiled.layouts().fused_baked(0).is_some());
        assert_eq!(compiled.layouts().max_fused_rank(), 9);
        assert!(compiled.baked_layouts_match_recomputed());
    }

    #[test]
    fn retained_payload_charge_tracks_capacity_not_diagnostic_lengths() {
        // What: admission accounting sees retained spare capacity that the
        // existing len-based layout diagnostics intentionally omit.
        let block = BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![1], 0).unwrap();
        let structure = BlockStructure::from_blocks_with_rank(1, vec![block]).unwrap();
        let mut compiled = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0_f64)],
        )
        .unwrap();
        let diagnostic = compiled.layouts().layout_table_bytes();
        let charged = compiled.charged_payload_bytes();

        compiled.layouts.shapes.reserve_exact(128);

        assert_eq!(compiled.layouts().layout_table_bytes(), diagnostic);
        assert!(compiled.charged_payload_bytes() > charged);
    }

    #[test]
    fn layout_normalization_overflow_returns_a_typed_compile_error() {
        // What: legal layout metadata whose signed span cannot be represented
        // is rejected as an overflow instead of panicking during normalization.
        let max = isize::MAX as usize;
        let block =
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![max - 1, max], 0).unwrap();
        let structure = BlockStructure::from_blocks_with_rank(2, vec![block]).unwrap();
        let error = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0_f64)],
        )
        .unwrap_err();

        assert!(matches!(error, OperationError::ElementCountOverflow));
    }

    #[test]
    fn layout_range_covers_negative_stride_rank_zero_and_zero_extent() {
        let layouts = TreeTransformLayoutTable {
            entries: vec![
                TreeTransformLayout {
                    layout_start: 0,
                    rank: 1,
                    offset: 5,
                    element_count: 3,
                },
                TreeTransformLayout {
                    layout_start: 1,
                    rank: 0,
                    offset: 7,
                    element_count: 1,
                },
                TreeTransformLayout {
                    layout_start: 1,
                    rank: 1,
                    offset: 0,
                    element_count: 0,
                },
            ],
            shapes: vec![3],
            strides: vec![-2],
            packed_strides: vec![1],
            fused_dims: Vec::new(),
            fused_dst_strides: Vec::new(),
            fused_src_strides: Vec::new(),
            fused_slots: Vec::new(),
            max_fused_rank: 0,
        };

        assert_eq!(layout_index_range(&layouts, 0).unwrap(), Some((1, 5)));
        assert_eq!(layout_index_range(&layouts, 1).unwrap(), Some((7, 7)));
        assert_eq!(layout_index_range(&layouts, 2).unwrap(), None);
    }

    #[test]
    fn single_only_schedule_allocates_no_scatter_group_metadata() {
        let structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        let schedule = transform.parallel_schedule();

        // What: an empty recoupling plan owns no Multi replay metadata.
        assert!(schedule.pack_columns.is_empty());
        assert!(schedule.scatter_columns.is_empty());
        assert!(schedule.scatter_groups.is_empty());
        assert_eq!(schedule.scatter_groups.capacity(), 0);
    }

    #[test]
    fn compiled_parallel_schedule_is_stable_for_equivalent_structures() {
        let block = |sector, offset| {
            BlockSpec::with_key(BlockKey::ordinal(sector), vec![2], vec![1], offset).unwrap()
        };
        let dst = BlockStructure::from_blocks_with_rank(1, vec![block(0, 0), block(1, 2)]).unwrap();
        let src = BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap();
        let specs = [TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 0.0, 0.0, 1.0],
        )];

        let first = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
        let second = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();

        assert_eq!(first.parallel_schedule(), second.parallel_schedule());
        assert_eq!(first.parallel_schedule().pack_columns.len(), 2);
        assert_eq!(first.parallel_schedule().scatter_columns.len(), 2);
        assert_eq!(first.blocks(), second.blocks());
        assert_eq!(first.layouts(), second.layouts());
        assert_eq!(
            first.gathered_coefficients(),
            second.gathered_coefficients()
        );
    }

    #[test]
    fn mapped_layout_append_preserves_axis_order_and_column_major_metadata() {
        let mut layouts = TreeTransformLayoutTable::default();

        let count = layouts
            .push_block_with_axes(3, &[2, 3, 4], &[1, 2, 6], 7, Some(&[2, 0, 1]))
            .unwrap();

        // What: source-axis permutation changes stored shape and source
        // strides in the requested order while packed strides describe that
        // same final shape.
        let layout = layouts.entry(0);
        assert_eq!(count, 24);
        assert_eq!(layouts.shape(layout), &[4, 2, 3]);
        assert_eq!(layouts.strides(layout), &[6, 1, 2]);
        assert_eq!(layouts.packed_strides(layout), &[1, 4, 8]);
        assert_eq!(layout.offset, 7);
    }

    #[test]
    fn mapped_layout_validation_is_atomic_and_keeps_permuted_zero_extent_order() {
        let mut layouts = TreeTransformLayoutTable::default();
        let empty = layouts.clone();

        let error = layouts
            .push_block_with_axes(3, &[2, 3, 4], &[1, 2, 6], 0, Some(&[0, 0, 2]))
            .unwrap_err();
        assert_eq!(
            error,
            OperationError::InvalidPermutation {
                axes: vec![0, 0, 2],
                rank: 3,
            }
        );
        assert_eq!(layouts, empty);

        let count = layouts
            .push_block_with_axes(3, &[usize::MAX, 2, 0], &[1, 1, 1], 0, Some(&[2, 0, 1]))
            .unwrap();
        // What: validation follows the materialized permutation's order, so a
        // leading zero extent keeps the same non-overflowing element count.
        assert_eq!(count, 0);
        assert_eq!(layouts.shape(layouts.entry(0)), &[0, usize::MAX, 2]);
    }

    #[test]
    fn high_rank_axis_validation_accepts_reverse_and_rejects_late_duplicate() {
        let rank = 257;
        let reverse = (0..rank).rev().collect::<Vec<_>>();
        let mut duplicate = reverse.clone();
        duplicate[rank - 1] = duplicate[rank - 2];

        // What: validation remains correct when rank exceeds inline metadata
        // capacity, including a duplicate discovered only at the final axis.
        assert_eq!(crate::axis::validate_permutation(&reverse, rank), Ok(()));
        assert_eq!(
            crate::axis::validate_permutation(&duplicate, rank),
            Err(OperationError::InvalidPermutation {
                axes: duplicate,
                rank,
            })
        );
    }

    #[test]
    fn physical_overwrite_coverage_includes_active_and_inactive_layouts() {
        // What: the compiled overwrite proof covers each destination byte once,
        // including blocks with no numerical source contribution.
        let dst = BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap();
        let src = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
        let transform = TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();

        assert_eq!(transform.physical_overwrite_len(), Some(4));
    }

    #[test]
    fn physical_overwrite_coverage_rejects_holes() {
        // What: storage padding that belongs to no destination block cannot be
        // exposed as initialized owned tensor data.
        let block = |sector, offset| {
            BlockSpec::with_key(BlockKey::ordinal(sector), vec![2], vec![1], offset).unwrap()
        };
        let dst = BlockStructure::from_blocks_with_rank(1, vec![block(0, 0), block(1, 3)]).unwrap();
        let src = BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap();
        let transform = TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[
                TreeTransformBlockSpec::single(0, 0, 1.0),
                TreeTransformBlockSpec::single(1, 1, 1.0),
            ],
        )
        .unwrap();

        assert_eq!(transform.physical_overwrite_len(), None);
    }

    #[test]
    fn physical_overwrite_coverage_handles_rank_zero_and_zero_extent() {
        // What: scalar blocks each own one physical slot, while an empty
        // destination proves the empty range without manufacturing a write.
        let scalar =
            BlockStructure::packed_column_major(0, [Vec::<usize>::new(), Vec::<usize>::new()])
                .unwrap();
        let scalar_transform = TreeTransformStructure::compile_structures(
            &scalar,
            &scalar,
            &[
                TreeTransformBlockSpec::single(0, 0, 1.0),
                TreeTransformBlockSpec::single(1, 1, 1.0),
            ],
        )
        .unwrap();
        assert_eq!(scalar_transform.physical_overwrite_len(), Some(2));

        let empty = BlockStructure::packed_column_major(1, [vec![0]]).unwrap();
        let empty_transform = TreeTransformStructure::compile_structures(
            &empty,
            &empty,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        assert_eq!(empty_transform.physical_overwrite_len(), Some(0));
    }
}
