use std::ops::Range;
use std::sync::Arc;

use smallvec::SmallVec;
use tenet_core::{
    validate_block_storage_injective, BlockStructure, CoreError, TensorMap, TensorStorage,
};
use tenet_dense::{strided_batch_runs, DenseGemmBatchJob};

use crate::kernel_adapter::{normalize_fused_layout, BakedFusedLayout, FusedLayoutScratch};
use crate::strided::offset_to_isize;
use crate::structure_identity::validate_structure_identity;
use crate::transform_plan::{
    ResolvedTreeTransformBlockSpec, TreeTransformBlockSpec, TreeTransformGroupBlockSpec,
    TreeTransformKeyBlockSpec,
};
use crate::OperationError;

trait TreeTransformCompileSpec<T> {
    fn dst_blocks(&self) -> &[usize];
    fn src_blocks(&self) -> &[usize];
    fn coefficients(&self) -> &[T];
    fn source_axes(&self) -> Option<&[usize]>;
}

impl<T> TreeTransformCompileSpec<T> for TreeTransformBlockSpec<T> {
    fn dst_blocks(&self) -> &[usize] {
        self.dst_blocks()
    }

    fn src_blocks(&self) -> &[usize] {
        self.src_blocks()
    }

    fn coefficients(&self) -> &[T] {
        self.recoupling_coefficients_dst_src()
    }

    fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes()
    }
}

impl<T> TreeTransformCompileSpec<T> for ResolvedTreeTransformBlockSpec<'_, T> {
    fn dst_blocks(&self) -> &[usize] {
        self.dst_blocks()
    }

    fn src_blocks(&self) -> &[usize] {
        self.src_blocks()
    }

    fn coefficients(&self) -> &[T] {
        self.coefficients()
    }

    fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes()
    }
}

/// Replay-ready tree-transform descriptor.
///
/// This is the TensorKit-style transformer-build boundary: construction resolves
/// tree keys, coefficients, block layouts, offsets, and pack/scatter descriptors
/// against concrete source and destination structures. Hot paths should build
/// this once and replay it with `tree_transform_execute_with` while reusing a
/// backend and workspace.
///
/// Why not expose mutable compiled fields: the recoupling plan, converted
/// coefficient cache, and threaded replay schedule all derive from the same
/// descriptors. Read them through [`Self::blocks`], [`Self::layouts`], and
/// [`Self::recoupling_coefficients_dst_src`] so those derived plans cannot go
/// stale after compilation.
///
/// Migration: code that previously read the public `blocks`, `layouts`, or
/// `recoupling_coefficients_dst_src` fields must use the same-named accessor
/// methods. Post-compilation mutation is no longer supported.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformStructure<T> {
    rank: usize,
    storage_conjugate: bool,
    identity: Arc<()>,
    blocks: Vec<TreeTransformBlock>,
    layouts: TreeTransformLayoutTable,
    recoupling_coefficients_dst_src: Vec<T>,
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
            .saturating_add(vector_bytes(
                self.recoupling_coefficients_dst_src.capacity(),
                core::mem::size_of::<T>(),
            ))
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
                self.recoupling_plan.runs.capacity(),
                core::mem::size_of::<usize>(),
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TreeTransformRecouplingPlan {
    source_len: usize,
    destination_len: usize,
    coefficient_len: usize,
    block_indices: Vec<usize>,
    jobs: Vec<DenseGemmBatchJob>,
    // Plan-time run partition of `jobs` (see issue #103): the dense backend
    // reads it to route each run without recomputing the partition per replay.
    runs: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformSingleReplay {
    pub dst_layout: usize,
    pub src_layout: usize,
    pub coefficient: usize,
    pub dst_lo: isize,
    pub dst_hi: isize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformPackReplay {
    pub src_layout: usize,
    pub packed_offset: usize,
    pub packed_hi: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformScatterReplay {
    pub dst_layout: usize,
    pub packed_offset: usize,
    pub dst_lo: isize,
    pub dst_hi: isize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformScatterGroupReplay {
    pub columns: Range<usize>,
    pub slice_disjoint: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TreeTransformParallelSchedule {
    pub singles: Vec<TreeTransformSingleReplay>,
    pub pack_columns: Vec<TreeTransformPackReplay>,
    pub scatter_columns: Vec<TreeTransformScatterReplay>,
    pub scatter_groups: Vec<TreeTransformScatterGroupReplay>,
    pub single_block_count: usize,
    pub packed_column_count: usize,
    pub scattered_column_count: usize,
    pub singles_slice_disjoint: bool,
}

impl TreeTransformRecouplingPlan {
    #[inline]
    pub fn source_len(&self) -> usize {
        self.source_len
    }

    #[inline]
    pub fn destination_len(&self) -> usize {
        self.destination_len
    }

    #[inline]
    pub fn coefficient_len(&self) -> usize {
        self.coefficient_len
    }

    #[inline]
    pub fn jobs(&self) -> &[DenseGemmBatchJob] {
        &self.jobs
    }

    /// Plan-time run partition of [`Self::jobs`]; handed to the backend so it
    /// routes runs without recomputing the partition (see issue #103).
    #[inline]
    pub fn runs(&self) -> &[usize] {
        &self.runs
    }

    #[inline]
    pub fn block_indices(&self) -> &[usize] {
        &self.block_indices
    }

    #[inline]
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (usize, &DenseGemmBatchJob)> + '_ {
        self.block_indices.iter().copied().zip(self.jobs.iter())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }
}

impl<T: Copy> TreeTransformStructure<T> {
    pub fn compile<
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
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        specs: &[TreeTransformBlockSpec<T>],
    ) -> Result<Self, OperationError>
    where
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        Self::compile_shared_structures(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            specs,
            false,
        )
    }

    pub fn compile_structures(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        specs: &[TreeTransformBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        Self::compile_structures_with_storage_conjugation(
            dst_structure,
            src_structure,
            specs,
            false,
        )
    }

    pub fn compile_structures_with_storage_conjugation(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        specs: &[TreeTransformBlockSpec<T>],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            specs,
            storage_conjugate,
        )
    }

    pub(crate) fn compile_resolved_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[ResolvedTreeTransformBlockSpec<'_, T>],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(dst_structure, src_structure, specs, storage_conjugate)
    }

    pub fn compile_keyed<
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
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        specs: &[TreeTransformKeyBlockSpec<T>],
    ) -> Result<Self, OperationError>
    where
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        Self::compile_keyed_shared_structures(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            specs,
            false,
        )
    }

    pub fn compile_keyed_structures(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        specs: &[TreeTransformKeyBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        Self::compile_keyed_structures_with_storage_conjugation(
            dst_structure,
            src_structure,
            specs,
            false,
        )
    }

    pub fn compile_keyed_structures_with_storage_conjugation(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        specs: &[TreeTransformKeyBlockSpec<T>],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        Self::compile_keyed_shared_structures(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            specs,
            storage_conjugate,
        )
    }

    pub fn compile_grouped<
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
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        specs: &[TreeTransformGroupBlockSpec<T>],
    ) -> Result<Self, OperationError>
    where
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        Self::compile_grouped_shared_structures(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            specs,
            false,
        )
    }

    pub fn compile_grouped_structures(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        specs: &[TreeTransformGroupBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        Self::compile_grouped_structures_with_storage_conjugation(
            dst_structure,
            src_structure,
            specs,
            false,
        )
    }

    pub fn compile_grouped_structures_with_storage_conjugation(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        specs: &[TreeTransformGroupBlockSpec<T>],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        Self::compile_grouped_shared_structures(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            specs,
            storage_conjugate,
        )
    }

    pub fn compile_grouped_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[TreeTransformGroupBlockSpec<T>],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        let mut resolved_specs = Vec::with_capacity(specs.len());
        // Why not compile spec-by-spec: grouped and keyed entry points must
        // resolve every key before rank/count/layout validation.
        for spec in specs {
            resolved_specs.push(spec.resolve(&dst_structure, &src_structure)?);
        }
        Self::compile_shared_structures(
            dst_structure,
            src_structure,
            &resolved_specs,
            storage_conjugate,
        )
    }

    fn compile_keyed_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[TreeTransformKeyBlockSpec<T>],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        let mut resolved_specs = Vec::with_capacity(specs.len());
        // Why not compile spec-by-spec: grouped and keyed entry points must
        // resolve every key before rank/count/layout validation.
        for spec in specs {
            resolved_specs.push(spec.resolve(&dst_structure, &src_structure)?);
        }
        Self::compile_shared_structures(
            dst_structure,
            src_structure,
            &resolved_specs,
            storage_conjugate,
        )
    }

    fn compile_shared_structures<S>(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[S],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError>
    where
        S: TreeTransformCompileSpec<T>,
    {
        let rank = dst_structure.rank();
        if src_structure.rank() != rank {
            return Err(OperationError::StructureRankMismatch {
                expected: rank,
                actual: src_structure.rank(),
            });
        }
        validate_destination_layouts_injective(
            &dst_structure,
            "tree transform destination layouts overlap",
        )?;

        let mut layouts = TreeTransformLayoutTable::default();
        let mut blocks = Vec::with_capacity(specs.len());
        let mut recoupling_coefficients_dst_src = Vec::new();
        let mut touched_dst_blocks = vec![false; dst_structure.block_count()];

        for spec in specs {
            let dst_blocks = spec.dst_blocks();
            let src_blocks = spec.src_blocks();
            let coefficients = spec.coefficients();
            if dst_blocks.is_empty() || src_blocks.is_empty() {
                return Err(OperationError::EmptyTransformBlock);
            }
            let src_count = src_blocks.len();
            let dst_count = dst_blocks.len();
            let expected_coefficients = src_count
                .checked_mul(dst_count)
                .ok_or(OperationError::ElementCountOverflow)?;
            if coefficients.len() != expected_coefficients {
                return Err(OperationError::CoefficientCountMismatch {
                    expected: expected_coefficients,
                    actual: coefficients.len(),
                });
            }

            for &dst_block in dst_blocks {
                let touched = touched_dst_blocks.get_mut(dst_block).ok_or(
                    OperationError::BlockIndexOutOfBounds {
                        tensor: "dst",
                        index: dst_block,
                        count: dst_structure.block_count(),
                    },
                )?;
                if *touched {
                    return Err(OperationError::DuplicateTransformDestination { dst_block });
                }
                *touched = true;
            }

            let dst_layout_start = layouts.entry_count();
            let mut element_count = None;
            for &dst_block in dst_blocks {
                let block = dst_structure.block(dst_block)?;
                let layout_element_count =
                    layouts.push_block(rank, block.shape(), block.strides(), block.offset())?;
                match element_count {
                    Some(expected) if expected != layout_element_count => {
                        return Err(OperationError::ElementCountMismatch {
                            expected,
                            actual: layout_element_count,
                        });
                    }
                    Some(_) => {}
                    None => element_count = Some(layout_element_count),
                }
            }

            let src_layout_start = layouts.entry_count();
            for &src_block in src_blocks {
                let block = src_structure.block(src_block)?;
                let layout_element_count = layouts.push_block_with_axes(
                    rank,
                    block.shape(),
                    block.strides(),
                    block.offset(),
                    spec.source_axes(),
                )?;
                match element_count {
                    Some(expected) if expected != layout_element_count => {
                        return Err(OperationError::ElementCountMismatch {
                            expected,
                            actual: layout_element_count,
                        });
                    }
                    Some(_) => {}
                    None => element_count = Some(layout_element_count),
                }
            }
            let element_count = element_count.expect("validated non-empty block");
            let coefficient_start = recoupling_coefficients_dst_src.len();
            recoupling_coefficients_dst_src.extend_from_slice(coefficients);

            if src_count == 1 && dst_count == 1 {
                let dst_layout = layouts.entry(dst_layout_start);
                let src_layout = layouts.entry(src_layout_start);
                if layouts.shape(dst_layout) != layouts.shape(src_layout) {
                    return Err(OperationError::ShapeMismatch {
                        dst: layouts.shape(dst_layout).to_vec(),
                        src: layouts.shape(src_layout).to_vec(),
                    });
                }
                blocks.push(TreeTransformBlock::Single {
                    dst_layout: dst_layout_start,
                    src_layout: src_layout_start,
                    coefficient: coefficient_start,
                });
            } else {
                blocks.push(TreeTransformBlock::Multi {
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    coefficient_start,
                    element_count,
                });
            }
        }
        let mut inactive_dst_layouts = Vec::new();
        for (dst_block, touched) in touched_dst_blocks.into_iter().enumerate() {
            if touched {
                continue;
            }
            let block = dst_structure.block(dst_block)?;
            inactive_dst_layouts.push(layouts.entry_count());
            layouts.push_block(rank, block.shape(), block.strides(), block.offset())?;
        }
        blocks.sort_by(|lhs, rhs| {
            tree_transform_block_weight(rhs, &layouts)
                .cmp(&tree_transform_block_weight(lhs, &layouts))
        });
        layouts.bake_fused_layouts(&blocks)?;
        let recoupling_plan = compile_recoupling_plan(&blocks)?;
        let parallel_schedule = compile_parallel_schedule(&blocks, &layouts, &recoupling_plan)?;
        let physical_overwrite_len = compile_physical_overwrite_coverage(
            &blocks,
            &inactive_dst_layouts,
            &layouts,
            &recoupling_plan,
            dst_structure.block_count(),
            dst_structure.required_len()?,
        )?;

        Ok(Self {
            rank,
            storage_conjugate,
            identity: Arc::new(()),
            blocks,
            layouts,
            recoupling_coefficients_dst_src,
            inactive_dst_layouts,
            physical_overwrite_len,
            recoupling_plan,
            parallel_schedule,
            dst_structure,
            src_structure,
        })
    }

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

    /// Immutable destination-by-source recoupling coefficients.
    #[inline]
    pub fn recoupling_coefficients_dst_src(&self) -> &[T] {
        &self.recoupling_coefficients_dst_src
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

    pub fn coefficient(&self, index: usize) -> T {
        self.recoupling_coefficients_dst_src[index]
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
}

#[doc(hidden)]
pub fn validate_destination_layouts_injective(
    dst_structure: &BlockStructure,
    overlap_message: &'static str,
) -> Result<(), OperationError> {
    // Why-not deduplicate only the beta scale: aliased logical destination
    // blocks can also require distinct alpha contributions, so no replay order
    // can represent their outputs in one physical element.
    match validate_block_storage_injective(dst_structure) {
        Ok(()) => Ok(()),
        Err(CoreError::OverlappingBlockStorage { .. }) => {
            // Why not expose block/offset details in a new variant:
            // OperationError is a public exhaustive enum, so that would
            // break downstream matches for a validation-only diagnostic.
            Err(OperationError::InvalidArgument {
                message: overlap_message,
            })
        }
        Err(CoreError::ElementCountOverflow) => Err(OperationError::ElementCountOverflow),
        Err(error) => Err(OperationError::Core(error)),
    }
}

fn compile_parallel_schedule(
    blocks: &[TreeTransformBlock],
    layouts: &TreeTransformLayoutTable,
    recoupling_plan: &TreeTransformRecouplingPlan,
) -> Result<TreeTransformParallelSchedule, OperationError> {
    let single_block_count = blocks
        .iter()
        .filter(|block| matches!(block, TreeTransformBlock::Single { .. }))
        .count();
    let mut packed_column_count = 0usize;
    let mut scattered_column_count = 0usize;
    let mut singles = Vec::new();
    for block in blocks {
        let TreeTransformBlock::Single {
            dst_layout,
            src_layout,
            coefficient,
        } = *block
        else {
            continue;
        };
        let Some((dst_lo, dst_hi)) = layout_index_range(layouts, dst_layout)? else {
            continue;
        };
        singles.push(TreeTransformSingleReplay {
            dst_layout,
            src_layout,
            coefficient,
            dst_lo,
            dst_hi,
        });
    }
    singles.sort_unstable_by_key(|item| item.dst_lo);

    let mut pack_columns = Vec::new();
    let mut scatter_columns = Vec::new();
    let mut scatter_groups = Vec::new();
    for (block_index, job) in recoupling_plan.entries() {
        let TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            src_layout_start,
            src_count,
            element_count,
            ..
        } = blocks[block_index]
        else {
            return Err(OperationError::InvalidArgument {
                message: "tree transform recoupling plan references a single block",
            });
        };
        packed_column_count = packed_column_count
            .checked_add(src_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        scattered_column_count = scattered_column_count
            .checked_add(dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let group_start = scatter_columns.len();
        if element_count == 0 {
            scatter_groups.push(TreeTransformScatterGroupReplay {
                columns: group_start..group_start,
                slice_disjoint: true,
            });
            continue;
        }
        for src_index in 0..src_count {
            let packed_offset = job
                .lhs_offset
                .checked_add(
                    src_index
                        .checked_mul(element_count)
                        .ok_or(OperationError::ElementCountOverflow)?,
                )
                .ok_or(OperationError::ElementCountOverflow)?;
            let packed_hi = packed_offset
                .checked_add(element_count - 1)
                .ok_or(OperationError::ElementCountOverflow)?;
            pack_columns.push(TreeTransformPackReplay {
                src_layout: src_layout_start + src_index,
                packed_offset,
                packed_hi,
            });
        }
        for dst_index in 0..dst_count {
            let dst_layout = dst_layout_start + dst_index;
            let Some((dst_lo, dst_hi)) = layout_index_range(layouts, dst_layout)? else {
                continue;
            };
            let packed_offset = job
                .dst_offset
                .checked_add(
                    dst_index
                        .checked_mul(element_count)
                        .ok_or(OperationError::ElementCountOverflow)?,
                )
                .ok_or(OperationError::ElementCountOverflow)?;
            scatter_columns.push(TreeTransformScatterReplay {
                dst_layout,
                packed_offset,
                dst_lo,
                dst_hi,
            });
        }
        scatter_columns[group_start..].sort_unstable_by_key(|item| item.dst_lo);
        scatter_groups.push(TreeTransformScatterGroupReplay {
            columns: group_start..scatter_columns.len(),
            slice_disjoint: destination_ranges_are_slice_disjoint(
                scatter_columns[group_start..]
                    .iter()
                    .map(|item| (item.dst_lo, item.dst_hi)),
            ),
        });
    }
    pack_columns.sort_unstable_by_key(|item| item.packed_offset);

    let pack_disjoint = pack_columns
        .windows(2)
        .all(|pair| pair[0].packed_hi < pair[1].packed_offset);
    if !pack_disjoint
        || pack_columns
            .last()
            .is_some_and(|item| item.packed_hi >= recoupling_plan.source_len())
    {
        return Err(OperationError::InvalidArgument {
            message: "tree transform packed source schedule is invalid",
        });
    }

    Ok(TreeTransformParallelSchedule {
        singles_slice_disjoint: destination_ranges_are_slice_disjoint(
            singles.iter().map(|item| (item.dst_lo, item.dst_hi)),
        ),
        single_block_count,
        packed_column_count,
        scattered_column_count,
        singles,
        pack_columns,
        scatter_columns,
        scatter_groups,
    })
}

fn destination_ranges_are_slice_disjoint(ranges: impl IntoIterator<Item = (isize, isize)>) -> bool {
    let mut previous_hi = None;
    for (lo, hi) in ranges {
        if lo < 0 || hi < lo || previous_hi.is_some_and(|previous| previous >= lo) {
            return false;
        }
        previous_hi = Some(hi);
    }
    true
}

fn layout_index_range(
    layouts: &TreeTransformLayoutTable,
    layout_index: usize,
) -> Result<Option<(isize, isize)>, OperationError> {
    let layout = layouts.entry(layout_index);
    if layout.element_count == 0 {
        return Ok(None);
    }
    let mut lo = layout.offset;
    let mut hi = layout.offset;
    for (&extent, &stride) in layouts.shape(layout).iter().zip(layouts.strides(layout)) {
        let extent = isize::try_from(extent.saturating_sub(1))
            .map_err(|_| OperationError::ElementCountOverflow)?;
        let span = extent
            .checked_mul(stride)
            .ok_or(OperationError::ElementCountOverflow)?;
        if span < 0 {
            lo = lo
                .checked_add(span)
                .ok_or(OperationError::ElementCountOverflow)?;
        } else {
            hi = hi
                .checked_add(span)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
    }
    Ok(Some((lo, hi)))
}

fn compile_recoupling_plan(
    blocks: &[TreeTransformBlock],
) -> Result<TreeTransformRecouplingPlan, OperationError> {
    #[derive(Clone, Copy)]
    struct MultiEntry {
        block_index: usize,
        element_count: usize,
        src_count: usize,
        dst_count: usize,
    }

    let mut entries = Vec::new();
    for (block_index, block) in blocks.iter().enumerate() {
        if let TreeTransformBlock::Multi {
            dst_count,
            src_count,
            element_count,
            ..
        } = *block
        {
            entries.push(MultiEntry {
                block_index,
                element_count,
                src_count,
                dst_count,
            });
        }
    }
    entries.sort_by_key(|entry| {
        (
            entry.element_count,
            entry.src_count,
            entry.dst_count,
            entry.block_index,
        )
    });

    let mut source_len = 0usize;
    let mut destination_len = 0usize;
    let mut coefficient_len = 0usize;
    let mut block_indices = Vec::with_capacity(entries.len());
    let mut jobs = Vec::with_capacity(entries.len());
    for entry in entries {
        let block_source_len = entry
            .element_count
            .checked_mul(entry.src_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let block_destination_len = entry
            .element_count
            .checked_mul(entry.dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let block_coefficient_len = entry
            .src_count
            .checked_mul(entry.dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        block_indices.push(entry.block_index);
        jobs.push(DenseGemmBatchJob {
            dst_offset: destination_len,
            lhs_offset: source_len,
            rhs_offset: coefficient_len,
            rows: entry.element_count,
            contracted: entry.src_count,
            cols: entry.dst_count,
        });
        source_len = source_len
            .checked_add(block_source_len)
            .ok_or(OperationError::ElementCountOverflow)?;
        destination_len = destination_len
            .checked_add(block_destination_len)
            .ok_or(OperationError::ElementCountOverflow)?;
        coefficient_len = coefficient_len
            .checked_add(block_coefficient_len)
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    let runs = strided_batch_runs(&jobs);
    Ok(TreeTransformRecouplingPlan {
        source_len,
        destination_len,
        coefficient_len,
        block_indices,
        jobs,
        runs,
    })
}

fn tree_transform_block_weight(
    block: &TreeTransformBlock,
    layouts: &TreeTransformLayoutTable,
) -> usize {
    match *block {
        TreeTransformBlock::Single { dst_layout, .. } => layouts.entry(dst_layout).element_count,
        TreeTransformBlock::Multi {
            dst_count,
            src_count,
            element_count,
            ..
        } => dst_count
            .saturating_mul(src_count)
            .saturating_mul(element_count),
    }
}

fn compile_physical_overwrite_coverage(
    blocks: &[TreeTransformBlock],
    inactive_dst_layouts: &[usize],
    layouts: &TreeTransformLayoutTable,
    recoupling_plan: &TreeTransformRecouplingPlan,
    destination_block_count: usize,
    required_len: usize,
) -> Result<Option<usize>, OperationError> {
    let multi_count = blocks
        .iter()
        .filter(|block| matches!(block, TreeTransformBlock::Multi { .. }))
        .count();
    let mut scheduled = vec![false; blocks.len()];
    if recoupling_plan.block_indices().len() != multi_count
        || recoupling_plan.block_indices().iter().any(|&index| {
            let Some(slot) = scheduled.get_mut(index) else {
                return true;
            };
            if *slot || !matches!(blocks[index], TreeTransformBlock::Multi { .. }) {
                return true;
            }
            *slot = true;
            false
        })
    {
        return Ok(None);
    }
    let active_layout_count = blocks.iter().try_fold(0usize, |count, block| {
        let destination_count = match *block {
            TreeTransformBlock::Single { .. } => 1,
            TreeTransformBlock::Multi { dst_count, .. } => dst_count,
        };
        count
            .checked_add(destination_count)
            .ok_or(OperationError::ElementCountOverflow)
    })?;
    let covered_layout_count = active_layout_count
        .checked_add(inactive_dst_layouts.len())
        .ok_or(OperationError::ElementCountOverflow)?;

    if covered_layout_count != destination_block_count {
        return Ok(None);
    }

    let mut intervals = Vec::with_capacity(covered_layout_count);
    let mut record_layout = |layout_index: usize| -> Result<bool, OperationError> {
        let layout = layouts.entry(layout_index);
        let Some((lo, hi)) = layout_index_range(layouts, layout_index)? else {
            return Ok(true);
        };
        let Ok(start) = usize::try_from(lo) else {
            return Ok(false);
        };
        let Ok(hi) = usize::try_from(hi) else {
            return Ok(false);
        };
        let Some(end) = hi.checked_add(1) else {
            return Ok(false);
        };
        if end > required_len || end - start != layout.element_count {
            return Ok(false);
        }
        intervals.push((start, end));
        Ok(true)
    };
    for block in blocks {
        match *block {
            TreeTransformBlock::Single { dst_layout, .. } => {
                if !record_layout(dst_layout)? {
                    return Ok(None);
                }
            }
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                ..
            } => {
                for layout in dst_layout_start..dst_layout_start + dst_count {
                    if !record_layout(layout)? {
                        return Ok(None);
                    }
                }
            }
        }
    }
    for &layout in inactive_dst_layouts {
        if !record_layout(layout)? {
            return Ok(None);
        }
    }
    intervals.sort_unstable_by_key(|&(start, _)| start);
    let mut next = 0usize;
    for (start, end) in intervals {
        if start != next {
            return Ok(None);
        }
        next = end;
    }

    // Why not enumerate physical scalar offsets: compile_shared_structures has
    // already rejected aliased destination layouts and duplicate block owners.
    // Contiguous layout intervals prove exact cover using only block metadata;
    // PhysicalOverwriteProof additionally requires canonical coupled-sector
    // regions to partition the same 0..required_len range before unsafe replay.
    Ok((next == required_len).then_some(required_len))
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

/// One prebaked fused layout's location in the arena (issue #232).
///
/// `rank == 0` is the "absent" sentinel: normalized layouts always have
/// rank >= 1, so a zero-rank slot means the entry was never baked (a
/// Single-block source layout, looked up only via its destination twin, or an
/// inactive destination). The compact 32-bit fields match QSpace's runtime-rank
/// metadata without making a small-rank execution split.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct FusedSlot {
    start: u32,
    rank: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TreeTransformLayoutTable {
    entries: Vec<TreeTransformLayout>,
    shapes: Vec<usize>,
    strides: Vec<isize>,
    packed_strides: Vec<isize>,
    // Baked fused loop layouts (issue #232): SoA arena mirroring shapes/strides
    // above, indexed per (entry, role) through `fused_slots`. Populated once at
    // compile time so replay skips layout normalization.
    fused_dims: Vec<usize>,
    fused_dst_strides: Vec<isize>,
    fused_src_strides: Vec<isize>,
    fused_slots: Vec<FusedSlot>,
    max_fused_rank: usize,
}

impl TreeTransformLayoutTable {
    fn entry_count(&self) -> usize {
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
        let (start, rank) = (layout.layout_start, layout.rank);
        let range = start..start + rank;
        let dst_strides = if dst_is_packed {
            &self.packed_strides[range.clone()]
        } else {
            &self.strides[range.clone()]
        };
        let src_strides = if src_is_packed {
            &self.packed_strides[range.clone()]
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
        let (ds, dr) = (dst.layout_start, dst.rank);
        let src = &self.entries[src_entry];
        let (ss, _sr) = (src.layout_start, src.rank);
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
        &self.shapes[layout.layout_start..layout.layout_start + layout.rank]
    }

    pub fn strides(&self, layout: &TreeTransformLayout) -> &[isize] {
        &self.strides[layout.layout_start..layout.layout_start + layout.rank]
    }

    pub fn packed_strides(&self, layout: &TreeTransformLayout) -> &[isize] {
        &self.packed_strides[layout.layout_start..layout.layout_start + layout.rank]
    }

    /// Populates the baked arena for every replayed (entry, role): each Single
    /// block's fused single layout at its destination entry, each Multi source
    /// entry's pack layout (dst = packed column, src = block strides), each Multi
    /// destination entry's scatter layout (dst = block strides, src = packed
    /// column). Inactive destinations and Single source entries are intentionally
    /// unbaked — the former never fuse-copy from a source, the latter are only
    /// reached through their destination twin. Block order after the replay sort
    /// is irrelevant: baking is keyed by stable entry index.
    fn bake_fused_layouts(&mut self, blocks: &[TreeTransformBlock]) -> Result<(), OperationError> {
        // Reserve the arena once so baking adds a bounded, block-count-independent
        // number of allocations rather than growing per push: fused rank never
        // exceeds an entry's rank, so `shapes.len()` upper-bounds each arena, and
        // one slot per entry covers `fused_slots`.
        self.fused_slots
            .resize(self.entries.len(), FusedSlot::default());
        self.fused_dims.reserve(self.shapes.len());
        self.fused_dst_strides.reserve(self.shapes.len());
        self.fused_src_strides.reserve(self.shapes.len());
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

    fn push_block(
        &mut self,
        rank: usize,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Result<usize, OperationError> {
        self.push_block_mapped(rank, shape, strides, offset, None)
    }

    fn push_block_mapped(
        &mut self,
        rank: usize,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        axes: Option<&[usize]>,
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

        let layout_start = self.shapes.len();
        self.shapes.reserve(rank);
        self.strides.reserve(rank);
        self.packed_strides.reserve(rank);
        let mut packed_stride = 1usize;
        for index in 0..rank {
            let axis = axis(index);
            self.shapes.push(shape[axis]);
            self.strides.push(strides[axis] as isize);
            self.packed_strides.push(packed_stride as isize);
            packed_stride *= shape[axis];
        }
        self.entries.push(TreeTransformLayout {
            layout_start,
            rank,
            offset,
            element_count,
        });
        Ok(element_count)
    }

    fn push_block_with_axes(
        &mut self,
        rank: usize,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        axes: Option<&[usize]>,
    ) -> Result<usize, OperationError> {
        let Some(axes) = axes else {
            return self.push_block(rank, shape, strides, offset);
        };
        validate_axis_permutation(axes, rank)?;
        self.push_block_mapped(rank, shape, strides, offset, Some(axes))
    }
}

fn validate_axis_permutation(axes: &[usize], rank: usize) -> Result<(), OperationError> {
    if axes.len() != rank {
        return Err(OperationError::InvalidPermutation {
            axes: axes.to_vec(),
            rank,
        });
    }
    let mut seen = SmallVec::<[bool; 16]>::new();
    seen.resize(rank, false);
    for &axis in axes {
        if axis >= rank || seen[axis] {
            return Err(OperationError::InvalidPermutation {
                axes: axes.to_vec(),
                rank,
            });
        }
        seen[axis] = true;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformLayout {
    layout_start: usize,
    rank: usize,
    pub offset: isize,
    pub element_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenet_core::{BlockKey, BlockSpec};

    fn multi(element_count: usize, src_count: usize, dst_count: usize) -> TreeTransformBlock {
        TreeTransformBlock::Multi {
            dst_layout_start: 0,
            dst_count,
            src_layout_start: 0,
            src_count,
            coefficient_start: 0,
            element_count,
        }
    }

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
    fn recoupling_plan_bakes_run_partition() {
        // Two same-shape Multi blocks fold into one length-2 constant-stride
        // run; a third differently-shaped block is a singleton. The compiled
        // plan stores that partition (issue #103) so the backend routes it
        // without recomputing, and it always covers every job.
        let blocks = vec![multi(2, 2, 2), multi(2, 2, 2), multi(3, 1, 1)];
        let plan = compile_recoupling_plan(&blocks).unwrap();
        assert_eq!(plan.jobs().len(), 3);
        assert_eq!(plan.runs(), &[2, 1]);
        assert_eq!(plan.runs(), strided_batch_runs(plan.jobs()));
        assert_eq!(
            plan.runs().iter().sum::<usize>(),
            plan.jobs().len(),
            "run partition must cover all jobs"
        );
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
            first.recoupling_coefficients_dst_src(),
            second.recoupling_coefficients_dst_src()
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
        assert_eq!(validate_axis_permutation(&reverse, rank), Ok(()));
        assert_eq!(
            validate_axis_permutation(&duplicate, rank),
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
