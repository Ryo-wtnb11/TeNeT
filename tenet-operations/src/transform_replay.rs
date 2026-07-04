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
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
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
    )
}

/// Replays a prepared structural-recoupling tree transform on host slices.
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
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
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
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    let total_start = std::time::Instant::now();

    let start = std::time::Instant::now();
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
    profile.validate += start.elapsed();

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
