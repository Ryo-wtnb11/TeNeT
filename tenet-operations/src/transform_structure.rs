use std::sync::Arc;

use tenet_core::{BlockStructure, TensorMap, TensorStorage};
use tenet_dense::{strided_batch_runs, DenseGemmBatchJob};

use crate::strided::{column_major_strides_isize, element_count, offset_to_isize};
use crate::structure_identity::validate_structure_identity;
use crate::transform_plan::{
    TreeTransformBlockSpec, TreeTransformGroupBlockSpec, TreeTransformKeyBlockSpec,
};
use crate::OperationError;

/// Replay-ready tree-transform descriptor.
///
/// This is the TensorKit-style transformer-build boundary: construction resolves
/// tree keys, coefficients, block layouts, offsets, and pack/scatter descriptors
/// against concrete source and destination structures. Hot paths should build
/// this once and replay it with `tree_transform_execute_with` while reusing a
/// backend and workspace.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformStructure<T> {
    rank: usize,
    storage_conjugate: bool,
    identity: Arc<()>,
    pub blocks: Vec<TreeTransformBlock>,
    pub layouts: TreeTransformLayoutTable,
    pub recoupling_coefficients_dst_src: Vec<T>,
    inactive_dst_layouts: Vec<usize>,
    recoupling_plan: TreeTransformRecouplingPlan,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
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
        let indexed_specs = specs
            .iter()
            .map(|spec| spec.to_indexed_spec(&dst_structure, &src_structure))
            .collect::<Result<Vec<_>, _>>()?;
        Self::compile_shared_structures(
            dst_structure,
            src_structure,
            &indexed_specs,
            storage_conjugate,
        )
    }

    fn compile_keyed_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[TreeTransformKeyBlockSpec<T>],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        let indexed_specs = specs
            .iter()
            .map(|spec| spec.to_indexed_spec(&dst_structure, &src_structure))
            .collect::<Result<Vec<_>, _>>()?;
        Self::compile_shared_structures(
            dst_structure,
            src_structure,
            &indexed_specs,
            storage_conjugate,
        )
    }

    fn compile_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[TreeTransformBlockSpec<T>],
        storage_conjugate: bool,
    ) -> Result<Self, OperationError> {
        let rank = dst_structure.rank();
        if src_structure.rank() != rank {
            return Err(OperationError::StructureRankMismatch {
                expected: rank,
                actual: src_structure.rank(),
            });
        }

        let mut layouts = TreeTransformLayoutTable::default();
        let mut blocks = Vec::with_capacity(specs.len());
        let mut recoupling_coefficients_dst_src = Vec::new();
        let mut touched_dst_blocks = vec![false; dst_structure.block_count()];

        for spec in specs {
            if spec.dst_blocks.is_empty() || spec.src_blocks.is_empty() {
                return Err(OperationError::EmptyTransformBlock);
            }
            let src_count = spec.src_blocks.len();
            let dst_count = spec.dst_blocks.len();
            let expected_coefficients = src_count
                .checked_mul(dst_count)
                .ok_or(OperationError::ElementCountOverflow)?;
            if spec.recoupling_coefficients_dst_src.len() != expected_coefficients {
                return Err(OperationError::CoefficientCountMismatch {
                    expected: expected_coefficients,
                    actual: spec.recoupling_coefficients_dst_src.len(),
                });
            }

            for &dst_block in &spec.dst_blocks {
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
            for &dst_block in &spec.dst_blocks {
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
            for &src_block in &spec.src_blocks {
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
            recoupling_coefficients_dst_src
                .extend_from_slice(&spec.recoupling_coefficients_dst_src);

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
        let recoupling_plan = compile_recoupling_plan(&blocks)?;

        Ok(Self {
            rank,
            storage_conjugate,
            identity: Arc::new(()),
            blocks,
            layouts,
            recoupling_coefficients_dst_src,
            inactive_dst_layouts,
            recoupling_plan,
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

    pub fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("src", &self.src_structure, src_structure)
    }
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TreeTransformLayoutTable {
    entries: Vec<TreeTransformLayout>,
    shapes: Vec<usize>,
    strides: Vec<isize>,
    packed_strides: Vec<isize>,
}

impl TreeTransformLayoutTable {
    fn entry_count(&self) -> usize {
        self.entries.len()
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

    fn push_block(
        &mut self,
        rank: usize,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
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
        let element_count = element_count(shape)?;
        let layout_start = self.shapes.len();
        let packed_strides = column_major_strides_isize(shape)?;
        self.shapes.extend_from_slice(shape);
        self.strides.extend(
            strides
                .iter()
                .map(|&stride| {
                    isize::try_from(stride)
                        .map_err(|_| OperationError::StrideOverflow { value: stride })
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        self.packed_strides.extend_from_slice(&packed_strides);
        self.entries.push(TreeTransformLayout {
            layout_start,
            rank,
            offset: offset_to_isize(offset)?,
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
        let shape = axes.iter().map(|&axis| shape[axis]).collect::<Vec<_>>();
        let strides = axes.iter().map(|&axis| strides[axis]).collect::<Vec<_>>();
        self.push_block(rank, &shape, &strides, offset)
    }
}

fn validate_axis_permutation(axes: &[usize], rank: usize) -> Result<(), OperationError> {
    if axes.len() != rank {
        return Err(OperationError::InvalidPermutation {
            axes: axes.to_vec(),
            rank,
        });
    }
    let mut seen = vec![false; rank];
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
}
