use core::ops::{Add, Mul};
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{
    BlockStructure, BlockView, BlockViewMut, HostReadableStorage, HostWritableStorage, Placement,
    ScratchStorage, SimilarStorage, TensorMap,
};
use tenet_dense::{DenseExecutor, DenseGemmBatchJob};

use crate::host_scratch::HostScratchBuffer;
use crate::storage_scratch::{StorageTreeTransformWorkspace, TreeTransformScratchBuffers};
use crate::strided::offset_to_isize;
use crate::tensoradd::{TensorAddDescriptor, TensorAddDescriptorTerm};
use crate::{
    tensoradd_raw_strided_kernel, tensoradd_raw_strided_kernel_trusted, ConjugateValue,
    DenseRecouplingScalar, HostAllocator, HostKernelAdapter, OperationError,
    RecouplingCoefficientAction, ReportsPlacement, TensorAddStructure, TreeTransformBlock,
    TreeTransformLayout, TreeTransformLayoutTable, TreeTransformReplayProfile,
    TreeTransformStructure,
};

/// Host scratch/replay workspace backed by `Vec<T>`.
///
/// Raw replay methods using this workspace operate on host slices. Device
/// execution should use a separate device workspace instead of hiding device
/// storage behind this type.
#[derive(Clone, Debug)]
pub struct HostTreeTransformWorkspace<T> {
    zero_strides: Vec<isize>,
    packed: TreeTransformScratchBuffers<HostScratchBuffer<T>, HostScratchBuffer<T>>,
    // Recoupling matrices converted into the data scalar type for the GEMM
    // application (TensorKit's basistransform buffer); replay packs every
    // Multi block's matrix into this one buffer so the recoupling GEMMs
    // submit as a single batch.
    coefficient_scratch: Vec<T>,
    // Reused job list for the batched recoupling GEMM.
    batch_jobs: Vec<DenseGemmBatchJob>,
}

pub type TreeTransformWorkspace<T> = HostTreeTransformWorkspace<T>;

impl<T> Default for HostTreeTransformWorkspace<T> {
    fn default() -> Self {
        Self {
            zero_strides: Vec::new(),
            packed: TreeTransformScratchBuffers::default(),
            coefficient_scratch: Vec::new(),
            batch_jobs: Vec::new(),
        }
    }
}

impl<T> HostTreeTransformWorkspace<T> {
    #[inline]
    pub fn placement(&self) -> Placement {
        Placement::Host
    }

    #[inline]
    pub fn is_host_workspace(&self) -> bool {
        self.placement() == Placement::Host
    }

    pub fn source_len(&self) -> usize {
        self.packed.source().len()
    }

    pub fn destination_len(&self) -> usize {
        self.packed.destination().len()
    }

    fn prepare_packed_buffers(&mut self, source_len: usize, destination_len: usize, zero: T)
    where
        T: Clone,
    {
        self.packed
            .source_mut()
            .resize_filled(source_len, zero.clone());
        self.packed
            .destination_mut()
            .resize_filled(destination_len, zero);
    }
}

impl<T> ReportsPlacement for HostTreeTransformWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

pub fn tensoradd_structure_with_strided_kernel<
    T,
    const NOUT: usize,
    const NIN: usize,
    S,
    DDst,
    DSrc,
>(
    allocator: &mut HostAllocator,
    structure: &TensorAddStructure,
    dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
    src: &TensorMap<T, NOUT, NIN, S, DSrc>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
    DDst: HostWritableStorage<T>,
    DSrc: HostReadableStorage<T>,
{
    let descriptor = structure.descriptor();
    structure.validate_replay_structures(dst.structure(), src.structure())?;
    if dst.structure().block_count() != descriptor.terms().len() {
        return Err(OperationError::BlockCountMismatch {
            dst: dst.structure().block_count(),
            src: descriptor.terms().len(),
        });
    }
    if src.structure().block_count() != descriptor.terms().len() {
        return Err(OperationError::BlockCountMismatch {
            dst: descriptor.terms().len(),
            src: src.structure().block_count(),
        });
    }

    let zero_strides = &mut allocator.zero_strides;
    let dst_data = dst.data_mut();
    let src_data = src.data();
    for term in descriptor.terms() {
        tensoradd_prepared_block_with_strided_kernel(
            zero_strides,
            descriptor,
            term,
            dst_data,
            src_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

pub fn tree_transform_structure_with_strided_kernel<
    A,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    tree_transform_structure_with_strided_kernel_raw(
        kernels,
        workspace,
        structure,
        &dst_structure,
        &src_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
    )
}

pub fn tree_transform_structure_with_storage_workspace_strided_kernel<
    A,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    kernels: &mut A,
    workspace: &mut StorageTreeTransformWorkspace<DSrc::Similar, DDst::Similar>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
    DDst: HostWritableStorage<D> + SimilarStorage<D>,
    DSrc: HostReadableStorage<D> + SimilarStorage<D>,
    DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    DSrc::Similar: HostWritableStorage<D> + ScratchStorage<D>,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    structure.validate_replay_structures(&dst_structure, &src_structure)?;
    validate_replay_storage_len(&dst_structure, dst.storage().len())?;
    validate_replay_storage_len(&src_structure, src.storage().len())?;

    let src_data = src.data();
    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                kernels,
                workspace.zero_strides_mut(),
                &structure.layouts,
                structure.layouts.entry(dst_layout),
                structure.layouts.entry(src_layout),
                structure.coefficient(coefficient),
                structure.storage_conjugate(),
                dst.data_mut(),
                src_data,
                alpha,
                beta,
            )?,
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => {
                let source_len = element_count
                    .checked_mul(src_count)
                    .ok_or(OperationError::ElementCountOverflow)?;
                let destination_len = element_count
                    .checked_mul(dst_count)
                    .ok_or(OperationError::ElementCountOverflow)?;
                workspace.prepare_from_storages(
                    src.storage(),
                    dst.storage(),
                    source_len,
                    destination_len,
                    D::zero(),
                );
                let (zero_strides, scratch) = workspace.replay_parts_mut();
                tree_transform_multi_with_scratch_buffers(
                    kernels,
                    zero_strides,
                    scratch,
                    &structure.layouts,
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    coefficient_start,
                    element_count,
                    &structure.recoupling_coefficients_dst_src,
                    structure.storage_conjugate(),
                    dst.data_mut(),
                    src_data,
                    alpha,
                    beta,
                )?;
            }
        }
    }
    Ok(())
}

/// Replays a prepared tree-transform structure on host slices.
pub fn tree_transform_structure_with_strided_kernel_raw<A, D, C>(
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                kernels,
                &mut workspace.zero_strides,
                &structure.layouts,
                structure.layouts.entry(dst_layout),
                structure.layouts.entry(src_layout),
                structure.coefficient(coefficient),
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                beta,
            )?,
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => tree_transform_multi_with_pack_gemm_scatter(
                kernels,
                workspace,
                &structure.layouts,
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
                &structure.recoupling_coefficients_dst_src,
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                beta,
            )?,
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn tree_transform_structure_with_structural_recoupling<
    A,
    E,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    tree_transform_structure_with_structural_recoupling_raw(
        kernels,
        dense,
        workspace,
        structure,
        &dst_structure,
        &src_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
        threads,
    )
}

/// Replays a prepared structural-recoupling tree transform on host slices.
///
/// `threads` selects the replay parallelism (a property of the executing
/// backend, not of the cached structure): `<= 1` runs the existing serial
/// path unchanged; `> 1` runs Single applies, Multi pack columns and Multi
/// scatter columns as independent work items over up to `threads`
/// work-stealing workers, with the batched recoupling GEMM staying a single
/// serial call between the two parallel phases.
#[allow(clippy::too_many_arguments)]
pub fn tree_transform_structure_with_structural_recoupling_raw<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
    if threads > 1 {
        return tree_transform_blocks_with_batched_recoupling_parallel(
            kernels, dense, workspace, structure, dst_data, src_data, alpha, beta, threads, None,
        );
    }
    tree_transform_blocks_with_batched_recoupling(
        kernels, dense, workspace, structure, dst_data, src_data, alpha, beta, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn tree_transform_structure_with_structural_recoupling_raw_profiled<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    threads: usize,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let total_start = std::time::Instant::now();

    let start = std::time::Instant::now();
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
    profile.validate += start.elapsed();

    if threads > 1 {
        tree_transform_blocks_with_batched_recoupling_parallel(
            kernels,
            dense,
            workspace,
            structure,
            dst_data,
            src_data,
            alpha,
            beta,
            threads,
            Some(profile),
        )?;
    } else {
        tree_transform_blocks_with_batched_recoupling(
            kernels,
            dense,
            workspace,
            structure,
            dst_data,
            src_data,
            alpha,
            beta,
            Some(profile),
        )?;
    }

    profile.total += total_start.elapsed();
    Ok(())
}

/// Executes a validated tree-transform block list against a dense executor:
/// Single blocks apply directly through the strided kernel, and every Multi
/// block packs into one shared source/destination scratch pair so all the
/// recoupling GEMMs (`destination = source * U^T` per block) submit as a
/// single batched call — small transform groups then pay the dense executor's
/// per-call dispatch cost once per replay instead of once per block.
///
/// Inlined into both the plain and profiled entry points so the
/// `Option<&mut profile>` checks constant-fold away in the unprofiled copy.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tree_transform_blocks_with_batched_recoupling<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    let layouts = &structure.layouts;

    // All-Single structures (abelian recoupling is diagonal) skip the batch
    // machinery entirely: no pack scratch, no job list, no scatter pass.
    if !structure
        .blocks
        .iter()
        .any(|block| matches!(block, TreeTransformBlock::Multi { .. }))
    {
        for block in &structure.blocks {
            let TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } = *block
            else {
                unreachable!("checked above: no Multi blocks");
            };
            let start = profile.as_ref().map(|_| std::time::Instant::now());
            tree_transform_single_with_strided_kernel(
                kernels,
                &mut workspace.zero_strides,
                layouts,
                layouts.entry(dst_layout),
                layouts.entry(src_layout),
                structure.coefficient(coefficient),
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                beta,
            )?;
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                let elapsed = start.elapsed();
                profile.single_blocks += 1;
                profile.single_total += elapsed;
                profile.strided_kernel += elapsed;
            }
        }
        return Ok(());
    }

    // Size the shared pack scratch over every Multi block.
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let mut total_source_len = 0usize;
    let mut total_destination_len = 0usize;
    for block in &structure.blocks {
        if let TreeTransformBlock::Multi {
            dst_count,
            src_count,
            element_count,
            ..
        } = *block
        {
            let source_len = element_count
                .checked_mul(src_count)
                .ok_or(OperationError::ElementCountOverflow)?;
            let destination_len = element_count
                .checked_mul(dst_count)
                .ok_or(OperationError::ElementCountOverflow)?;
            total_source_len = total_source_len
                .checked_add(source_len)
                .ok_or(OperationError::ElementCountOverflow)?;
            total_destination_len = total_destination_len
                .checked_add(destination_len)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
    }
    workspace.prepare_packed_buffers(total_source_len, total_destination_len, D::zero());
    workspace.coefficient_scratch.clear();
    let mut jobs = std::mem::take(&mut workspace.batch_jobs);
    jobs.clear();
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        profile.multi_workspace_prepare += start.elapsed();
    }

    // Singles apply directly; Multi blocks pack their source columns and
    // convert their recoupling matrix into the shared coefficient buffer.
    let mut source_base = 0usize;
    let mut destination_base = 0usize;
    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => {
                // Timestamps only under profiling: the per-block clock reads
                // are measurable against microsecond replays.
                let start = profile.as_ref().map(|_| std::time::Instant::now());
                tree_transform_single_with_strided_kernel(
                    kernels,
                    &mut workspace.zero_strides,
                    layouts,
                    layouts.entry(dst_layout),
                    layouts.entry(src_layout),
                    structure.coefficient(coefficient),
                    structure.storage_conjugate(),
                    dst_data,
                    src_data,
                    alpha,
                    beta,
                )?;
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                    let elapsed = start.elapsed();
                    profile.single_blocks += 1;
                    profile.single_total += elapsed;
                    profile.strided_kernel += elapsed;
                }
            }
            TreeTransformBlock::Multi {
                dst_layout_start: _,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => {
                let start = profile.as_ref().map(|_| std::time::Instant::now());
                for src_index in 0..src_count {
                    let layout = layouts.entry(src_layout_start + src_index);
                    pack_layout_into_column(
                        kernels,
                        layouts,
                        layout,
                        src_data,
                        workspace.packed.source_mut().as_mut_slice(),
                        source_base + src_index * element_count,
                        structure.storage_conjugate(),
                    )?;
                }
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                    profile.multi_blocks += 1;
                    profile.packed_columns += src_count;
                    profile.multi_pack += start.elapsed();
                }

                let start = profile.as_ref().map(|_| std::time::Instant::now());
                let coefficient_len = src_count
                    .checked_mul(dst_count)
                    .ok_or(OperationError::ElementCountOverflow)?;
                let coefficient_end = coefficient_start
                    .checked_add(coefficient_len)
                    .ok_or(OperationError::ElementCountOverflow)?;
                let coefficients = structure
                    .recoupling_coefficients_dst_src
                    .get(coefficient_start..coefficient_end)
                    .ok_or(OperationError::CoefficientCountMismatch {
                        expected: coefficient_end,
                        actual: structure.recoupling_coefficients_dst_src.len(),
                    })?;
                let rhs_offset = workspace.coefficient_scratch.len();
                workspace.coefficient_scratch.extend(
                    coefficients
                        .iter()
                        .map(|&coefficient| D::coefficient_as_data(coefficient)),
                );
                jobs.push(DenseGemmBatchJob {
                    dst_offset: destination_base,
                    lhs_offset: source_base,
                    rhs_offset,
                    rows: element_count,
                    contracted: src_count,
                    cols: dst_count,
                });
                source_base += element_count * src_count;
                destination_base += element_count * dst_count;
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                    profile.multi_coefficient_prepare += start.elapsed();
                }
            }
        }
    }

    // One batched recoupling GEMM across all Multi blocks (TensorKit's
    // `_add_transform_multi!` `mul!` step, grouped).
    if !jobs.is_empty() {
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let (source, destination) = workspace.packed.source_and_destination_mut();
        recoupling_gemm_batch(
            dense,
            destination.as_mut_slice(),
            source.as_slice(),
            &workspace.coefficient_scratch,
            &jobs,
        )?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.multi_scalar_recoupling += elapsed;
            profile.multi_matmul_total += elapsed;
        }
    }
    workspace.batch_jobs = jobs;

    // Scatter each Multi block's destination columns back out.
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let mut destination_base = 0usize;
    let mut scattered_columns = 0usize;
    for block in &structure.blocks {
        if let TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            element_count,
            ..
        } = *block
        {
            for dst_index in 0..dst_count {
                let layout = layouts.entry(dst_layout_start + dst_index);
                scatter_column_into_layout(
                    kernels,
                    &mut workspace.zero_strides,
                    layouts,
                    layout,
                    workspace.packed.destination().as_slice(),
                    destination_base + dst_index * element_count,
                    dst_data,
                    alpha,
                    beta,
                )?;
            }
            scattered_columns += dst_count;
            destination_base += element_count * dst_count;
        }
    }
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        profile.scattered_columns += scattered_columns;
        profile.multi_scatter += start.elapsed();
    }
    Ok(())
}

/// Inclusive index range `[lo, hi]` touched by a layout's strided walk over
/// `shape` from `offset` (negative strides walk downward).
fn layout_index_range(
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
) -> (isize, isize) {
    let mut lo = layout.offset;
    let mut hi = layout.offset;
    for (&extent, &stride) in layouts
        .shape(layout)
        .iter()
        .zip(layouts.strides(layout).iter())
    {
        let span = (extent as isize - 1) * stride;
        if span < 0 {
            lo += span;
        } else {
            hi += span;
        }
    }
    (lo, hi)
}

/// Splits `data` into one disjoint `&mut` region per item (items sorted by
/// `lo`, each `(payload, lo, hi)` an inclusive touched range); each result
/// carries the region and its absolute start index so layout offsets can be
/// rebased. Returns `None` when regions overlap or run out of bounds — valid
/// packed transform structures never do (compile rejects duplicate
/// destination blocks), so `None` only guards degenerate stride patterns and
/// sends the caller down the serial path.
#[allow(clippy::type_complexity)]
fn split_regions<'a, T, P: Copy>(
    data: &'a mut [T],
    items: &[(P, isize, isize)],
) -> Option<Vec<(P, &'a mut [T], isize)>> {
    let mut regions = Vec::with_capacity(items.len());
    let mut rest = data;
    // Absolute index where `rest` begins.
    let mut consumed = 0isize;
    for &(payload, lo, hi) in items {
        if lo < consumed || hi < lo {
            return None;
        }
        let skip = (lo - consumed) as usize;
        let len = (hi - lo + 1) as usize;
        if skip.checked_add(len)? > rest.len() {
            return None;
        }
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(skip);
        let (region, tail) = tail.split_at_mut(len);
        regions.push((payload, region, lo));
        rest = tail;
        consumed = hi + 1;
    }
    Some(regions)
}

/// Threaded variant of [`tree_transform_blocks_with_batched_recoupling`]
/// (TensorKit `_add_abelian_kernel_threaded!` / `_add_general_kernel_threaded!`
/// precedent, indexmanipulations.jl:520-738):
///
/// - Phase A packs every Multi source column and applies every Single block
///   in parallel across tree pairs; phase B scatters every Multi destination
///   column in parallel. Work items are independent because the compile step
///   rejects duplicate destination blocks
///   (`OperationError::DuplicateTransformDestination`) and pack columns are
///   disjoint scratch ranges by construction; the workspace forbids `unsafe`,
///   so disjointness is realized structurally by pre-splitting the buffers
///   into per-item `&mut` regions (`split_at_mut`) and rebasing offsets,
///   instead of TensorKit-style shared writes.
/// - The batched recoupling GEMM stays ONE serial grouped call between the
///   two parallel phases — the dense executor owns its own parallelism and
///   no nesting arises because the batch submits outside both regions.
///
/// Scheduling: rayon parallel iterators on the global pool (the same pool
/// strided-kernel's threaded kernels use), with `with_min_len` bounding the
/// split count to the configured `threads` — the moral equivalent of
/// TensorKit's `min(ntasks, nblocks)` spawned workers.
///
/// Per-task state is one cloned kernel adapter (a ZST for the strided
/// adapter) and one `Vec::new()` zero-strides scratch (no allocation until a
/// kernel actually needs it); the pack/destination scratch itself stays the
/// reused workspace buffer, so the `ScratchStorage` reuse contract is
/// untouched — a deliberate deviation from TensorKit, which allocates pack
/// buffers inside every spawned task.
///
/// Profiling attribution is coarser than the serial path: phase A lands in
/// `multi_pack` (Singles included) and phase B in `multi_scatter`; per-item
/// clocks across workers would measure contention, not work.
#[allow(clippy::too_many_arguments)]
fn tree_transform_blocks_with_batched_recoupling_parallel<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    threads: usize,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    use rayon::prelude::*;

    let layouts = &structure.layouts;

    // Build phase (serial): size the pack scratch, convert coefficients,
    // build GEMM jobs, and collect the parallel work items.
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let mut total_source_len = 0usize;
    let mut total_destination_len = 0usize;
    for block in &structure.blocks {
        if let TreeTransformBlock::Multi {
            dst_count,
            src_count,
            element_count,
            ..
        } = *block
        {
            let source_len = element_count
                .checked_mul(src_count)
                .ok_or(OperationError::ElementCountOverflow)?;
            let destination_len = element_count
                .checked_mul(dst_count)
                .ok_or(OperationError::ElementCountOverflow)?;
            total_source_len = total_source_len
                .checked_add(source_len)
                .ok_or(OperationError::ElementCountOverflow)?;
            total_destination_len = total_destination_len
                .checked_add(destination_len)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
    }
    workspace.prepare_packed_buffers(total_source_len, total_destination_len, D::zero());
    workspace.coefficient_scratch.clear();
    let mut jobs = std::mem::take(&mut workspace.batch_jobs);
    jobs.clear();

    // (dst_layout, src_layout, coefficient index) per Single block.
    let mut singles: Vec<(usize, usize, usize)> = Vec::new();
    // (source layout, column length) per Multi pack column, in scratch order.
    let mut pack_columns: Vec<(usize, usize)> = Vec::new();
    // (dst layout, packed destination offset) per Multi scatter column.
    let mut scatter_columns: Vec<(usize, usize)> = Vec::new();

    let mut source_base = 0usize;
    let mut destination_base = 0usize;
    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => singles.push((dst_layout, src_layout, coefficient)),
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => {
                for src_index in 0..src_count {
                    pack_columns.push((src_layout_start + src_index, element_count));
                }
                for dst_index in 0..dst_count {
                    scatter_columns.push((
                        dst_layout_start + dst_index,
                        destination_base + dst_index * element_count,
                    ));
                }
                let coefficient_len = src_count
                    .checked_mul(dst_count)
                    .ok_or(OperationError::ElementCountOverflow)?;
                let coefficient_end = coefficient_start
                    .checked_add(coefficient_len)
                    .ok_or(OperationError::ElementCountOverflow)?;
                let coefficients = structure
                    .recoupling_coefficients_dst_src
                    .get(coefficient_start..coefficient_end)
                    .ok_or(OperationError::CoefficientCountMismatch {
                        expected: coefficient_end,
                        actual: structure.recoupling_coefficients_dst_src.len(),
                    })?;
                let rhs_offset = workspace.coefficient_scratch.len();
                workspace.coefficient_scratch.extend(
                    coefficients
                        .iter()
                        .map(|&coefficient| D::coefficient_as_data(coefficient)),
                );
                jobs.push(DenseGemmBatchJob {
                    dst_offset: destination_base,
                    lhs_offset: source_base,
                    rhs_offset,
                    rows: element_count,
                    contracted: src_count,
                    cols: dst_count,
                });
                source_base += element_count * src_count;
                destination_base += element_count * dst_count;
            }
        }
    }
    let single_count = singles.len();
    let multi_count = jobs.len();
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        profile.multi_workspace_prepare += start.elapsed();
        profile.single_blocks += single_count;
        profile.multi_blocks += multi_count;
    }

    // At most `threads` parallel chunks per phase (TensorKit's
    // `min(ntasks, nblocks)` worker bound) on rayon's global pool.
    let min_len = |items: usize| items.div_ceil(threads).max(1);
    let storage_conjugate = structure.storage_conjugate();

    // Phase A: pack columns and Single applies in parallel.
    {
        let start = profile.as_ref().map(|_| std::time::Instant::now());

        // Pack columns are contiguous consecutive ranges of the source
        // scratch, so the buffer splits sequentially into per-column &mut
        // regions (pack offsets rebase to 0).
        let mut column_regions: Vec<(usize, &mut [D])> = Vec::with_capacity(pack_columns.len());
        let mut rest = workspace.packed.source_mut().as_mut_slice();
        for &(layout, len) in &pack_columns {
            let (column, tail) = std::mem::take(&mut rest).split_at_mut(len);
            column_regions.push((layout, column));
            rest = tail;
        }
        let pack_chunk = min_len(column_regions.len());
        column_regions
            .into_par_iter()
            .with_min_len(pack_chunk)
            .try_for_each_init(
                || kernels.clone(),
                |kernels, (layout, column)| {
                    pack_layout_into_column(
                        kernels,
                        layouts,
                        layouts.entry(layout),
                        src_data,
                        column,
                        0,
                        storage_conjugate,
                    )
                },
            )?;

        // Single blocks write disjoint destination subblocks: split dst_data
        // into per-item regions and rebase the destination offsets.
        let mut items: Vec<((usize, usize, usize), isize, isize)> = singles
            .iter()
            .map(|&item| {
                let (lo, hi) = layout_index_range(layouts, layouts.entry(item.0));
                (item, lo, hi)
            })
            .collect();
        items.sort_unstable_by_key(|&(_, lo, _)| lo);
        match split_regions(dst_data, &items) {
            Some(regions) => {
                let single_chunk = min_len(regions.len());
                regions
                    .into_par_iter()
                    .with_min_len(single_chunk)
                    .try_for_each_init(
                        || (kernels.clone(), Vec::new()),
                        |(kernels, zero_strides),
                         ((dst_layout, src_layout, coefficient), region, region_start)| {
                            let dst_layout = layouts.entry(dst_layout);
                            let src_layout = layouts.entry(src_layout);
                            let scale =
                                alpha.scale_by_coefficient(structure.coefficient(coefficient));
                            kernels.add_strided(
                                zero_strides,
                                region,
                                src_data,
                                layouts.shape(dst_layout),
                                layouts.strides(dst_layout),
                                layouts.strides(src_layout),
                                dst_layout.offset - region_start,
                                src_layout.offset,
                                storage_conjugate,
                                scale,
                                beta,
                            )
                        },
                    )?;
            }
            // Degenerate (overlapping) regions: fall back to the serial
            // Single loop; valid packed structures never reach this.
            None => {
                let mut zero_strides = Vec::new();
                for &(dst_layout, src_layout, coefficient) in &singles {
                    tree_transform_single_with_strided_kernel(
                        kernels,
                        &mut zero_strides,
                        layouts,
                        layouts.entry(dst_layout),
                        layouts.entry(src_layout),
                        structure.coefficient(coefficient),
                        storage_conjugate,
                        dst_data,
                        src_data,
                        alpha,
                        beta,
                    )?;
                }
            }
        }

        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.packed_columns += pack_columns.len();
            profile.multi_pack += start.elapsed();
        }
    }

    // One batched recoupling GEMM across all Multi blocks, outside both
    // parallel regions (the dense executor owns its parallelism).
    if !jobs.is_empty() {
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let (source, destination) = workspace.packed.source_and_destination_mut();
        recoupling_gemm_batch(
            dense,
            destination.as_mut_slice(),
            source.as_slice(),
            &workspace.coefficient_scratch,
            &jobs,
        )?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.multi_scalar_recoupling += elapsed;
            profile.multi_matmul_total += elapsed;
        }
    }
    workspace.batch_jobs = jobs;

    // Phase B: scatter destination columns in parallel (disjoint destination
    // subblocks, same compile guarantee as the Singles).
    {
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let packed_destination = workspace.packed.destination().as_slice();
        let mut items: Vec<((usize, usize), isize, isize)> = scatter_columns
            .iter()
            .map(|&item| {
                let (lo, hi) = layout_index_range(layouts, layouts.entry(item.0));
                (item, lo, hi)
            })
            .collect();
        items.sort_unstable_by_key(|&(_, lo, _)| lo);
        match split_regions(dst_data, &items) {
            Some(regions) => {
                let scatter_chunk = min_len(regions.len());
                regions
                    .into_par_iter()
                    .with_min_len(scatter_chunk)
                    .try_for_each_init(
                        || kernels.clone(),
                        |kernels, ((layout, packed_offset), region, region_start)| {
                            let layout = layouts.entry(layout);
                            kernels.axpby_strided(
                                region,
                                packed_destination,
                                layouts.shape(layout),
                                layouts.strides(layout),
                                layouts.packed_strides(layout),
                                layout.offset - region_start,
                                offset_to_isize(packed_offset)?,
                                alpha,
                                beta,
                            )
                        },
                    )?;
            }
            None => {
                let mut zero_strides = Vec::new();
                for &(layout, packed_offset) in &scatter_columns {
                    scatter_column_into_layout(
                        kernels,
                        &mut zero_strides,
                        layouts,
                        layouts.entry(layout),
                        packed_destination,
                        packed_offset,
                        dst_data,
                        alpha,
                        beta,
                    )?;
                }
            }
        }
        if let (Some(profile), Some(start)) = (profile, start) {
            profile.scattered_columns += scatter_columns.len();
            profile.multi_scatter += start.elapsed();
        }
    }
    Ok(())
}

pub fn tensoradd_block_with_strided_kernel<T>(
    allocator: &mut HostAllocator,
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    let dst_shape = dst.shape().to_vec();
    let dst_strides = crate::strided::strides_to_isize(dst.strides())?;
    let dst_offset = offset_to_isize(dst.offset())?;
    let (dst_data, _) = dst.into_parts();
    let src_shape = src.shape().to_vec();
    let src_strides = crate::strided::strides_to_isize(src.strides())?;
    let src_offset = offset_to_isize(src.offset())?;
    let src_data = src.data();

    if dst_shape != src_shape {
        return Err(OperationError::ShapeMismatch {
            dst: dst_shape,
            src: src_shape,
        });
    }

    tensoradd_raw_strided_kernel(
        &mut allocator.zero_strides,
        dst_data,
        src_data,
        &dst_shape,
        &dst_strides,
        &src_strides,
        dst_offset,
        src_offset,
        false,
        alpha,
        beta,
    )
}

fn tensoradd_prepared_block_with_strided_kernel<T>(
    zero_strides: &mut Vec<isize>,
    descriptor: &TensorAddDescriptor,
    term: &TensorAddDescriptorTerm,
    dst_data: &mut [T],
    src_data: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    tensoradd_raw_strided_kernel_trusted(
        zero_strides,
        dst_data,
        src_data,
        descriptor.shape(term),
        descriptor.dst_strides(term),
        descriptor.src_strides(term),
        term.dst_offset,
        term.src_offset,
        descriptor.source_conjugate(),
        alpha,
        beta,
    )
}

fn validate_replay_storage_len(
    structure: &BlockStructure,
    actual_len: usize,
) -> Result<(), OperationError> {
    let expected = structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    if actual_len != expected {
        return Err(OperationError::ElementCountMismatch {
            expected,
            actual: actual_len,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_single_with_strided_kernel<A, D, C>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    dst_layout: &TreeTransformLayout,
    src_layout: &TreeTransformLayout,
    coefficient: C,
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let shape = layouts.shape(dst_layout);
    let scale = alpha.scale_by_coefficient(coefficient);
    kernels.add_strided(
        zero_strides,
        dst_data,
        src_data,
        shape,
        layouts.strides(dst_layout),
        layouts.strides(src_layout),
        dst_layout.offset,
        src_layout.offset,
        source_conjugate,
        scale,
        beta,
    )
}

/// Applies every Multi block's recoupling matrix in one batched GEMM over
/// shared flat scratch buffers: per job, the column-major
/// (element_count x dst_count) destination block receives `source_block *
/// U^T`, with `recoupling_coefficients_dst_src` (row-major `U[dst, src]`)
/// reinterpreted as the column-major (src_count x dst_count) matrix `U^T`.
/// This is TensorKit's `_add_transform_multi!` `mul!` step submitted as one
/// grouped call; the naive per-element loop in the kernel adapter remains
/// only for adapters without a dense executor. Job offsets are constructed by
/// the replay against scratch sized to their exact totals, matching the
/// plan-compile validation contract of the trusted views.
fn recoupling_gemm_batch<E, D>(
    dense: &mut E,
    destination: &mut [D],
    source: &[D],
    coefficients: &[D],
    jobs: &[DenseGemmBatchJob],
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar,
{
    let dst_shape = [destination.len()];
    let lhs_shape = [source.len()];
    let rhs_shape = [coefficients.len()];
    let flat_strides = [1];
    let lhs = D::dense_read(tenet_dense::DenseView::new_trusted(
        source,
        &lhs_shape,
        &flat_strides,
        0,
    ));
    let rhs = D::dense_read(tenet_dense::DenseView::new_trusted(
        coefficients,
        &rhs_shape,
        &flat_strides,
        0,
    ));
    let output = D::dense_write(tenet_dense::DenseViewMut::new_trusted(
        destination,
        &dst_shape,
        &flat_strides,
        0,
    ));
    dense
        .matmul_batch_axpby_into(
            output,
            lhs,
            rhs,
            jobs,
            D::one().dense_scalar(),
            D::zero().dense_scalar(),
        )
        .map_err(OperationError::Dense)
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_pack_gemm_scatter<A, D, C>(
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    recoupling_coefficients_dst_src: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + Zero + One + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    workspace.prepare_packed_buffers(source_len, destination_len, D::zero());
    tree_transform_multi_with_scratch_buffers(
        kernels,
        &mut workspace.zero_strides,
        &mut workspace.packed,
        layouts,
        dst_layout_start,
        dst_count,
        src_layout_start,
        src_count,
        coefficient_start,
        element_count,
        recoupling_coefficients_dst_src,
        source_conjugate,
        dst_data,
        src_data,
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_scratch_buffers<A, D, C, SourceScratch, DestinationScratch>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    scratch: &mut TreeTransformScratchBuffers<SourceScratch, DestinationScratch>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    recoupling_coefficients_dst_src: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + One + RecouplingCoefficientAction<C>,
    C: Copy,
    SourceScratch: HostWritableStorage<D>,
    DestinationScratch: HostWritableStorage<D>,
{
    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column(
            kernels,
            layouts,
            layout,
            src_data,
            scratch.source_mut().as_mut_slice(),
            src_index * element_count,
            source_conjugate,
        )?;
    }

    {
        let (source, destination) = scratch.source_and_destination_mut();
        kernels.recoupling_src_times_u_transpose(
            destination.as_mut_slice(),
            source.as_slice(),
            recoupling_coefficients_dst_src,
            coefficient_start,
            element_count,
            src_count,
            dst_count,
        )?;
    }

    for dst_index in 0..dst_count {
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout(
            kernels,
            zero_strides,
            layouts,
            layout,
            scratch.destination().as_slice(),
            dst_index * element_count,
            dst_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

fn pack_layout_into_column<A, T>(
    kernels: &mut A,
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    src_data: &[T],
    packed: &mut [T],
    packed_offset: usize,
    source_conjugate: bool,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + One,
{
    let shape = layouts.shape(layout);
    let packed_offset = offset_to_isize(packed_offset)?;
    kernels.copy_scale_strided(
        packed,
        src_data,
        shape,
        layouts.packed_strides(layout),
        layouts.strides(layout),
        packed_offset,
        layout.offset,
        source_conjugate,
        T::one(),
    )
}

#[allow(clippy::too_many_arguments)]
fn scatter_column_into_layout<A, T>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    packed: &[T],
    packed_offset: usize,
    dst_data: &mut [T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy,
{
    let shape = layouts.shape(layout);
    zero_strides.clear();
    kernels.axpby_strided(
        dst_data,
        packed,
        shape,
        layouts.strides(layout),
        layouts.packed_strides(layout),
        layout.offset,
        offset_to_isize(packed_offset)?,
        alpha,
        beta,
    )
}
