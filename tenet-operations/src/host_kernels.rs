use core::ops::{Add, Mul};
use std::sync::Arc;

use num_traits::{One, Zero};
#[cfg(test)]
use tenet_core::SimilarStorage;
use tenet_core::{
    BlockStructure, BlockView, BlockViewMut, HostReadableStorage, HostWritableStorage, Placement,
    TensorMap,
};
use tenet_dense::DenseExecutor;

use crate::host_scratch::HostScratchBuffer;
use crate::storage_scratch::TreeTransformScratchBuffers;
use crate::strided::offset_to_isize;
use crate::tensoradd::{TensorAddDescriptor, TensorAddDescriptorTerm};
use crate::{
    axpby_raw_strided_kernel_trusted, copy_scale_raw_strided_kernel_with_conjugate_trusted,
    tensoradd_raw_strided_kernel, tensoradd_raw_strided_kernel_profiled,
    tensoradd_raw_strided_kernel_trusted, ConjugateValue, DenseRecouplingScalar, HostAllocator,
    OperationError, RecouplingCoefficientAction, ReportsPlacement, TensorAddStructure,
    TreeTransformBlock, TreeTransformLayout, TreeTransformLayoutTable, TreeTransformReplayProfile,
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
}

pub type TreeTransformWorkspace<T> = HostTreeTransformWorkspace<T>;

impl<T> Default for HostTreeTransformWorkspace<T> {
    fn default() -> Self {
        Self {
            zero_strides: Vec::new(),
            packed: TreeTransformScratchBuffers::default(),
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

pub(crate) fn tensoradd_structure_with_strided_kernel<
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

pub(crate) fn tree_transform_structure_with_strided_kernel<
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
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
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

#[cfg(test)]
pub(crate) fn tree_transform_structure_with_storage_scratch_strided_kernel<
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
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
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
    DDst::Similar: HostWritableStorage<D>,
    DSrc::Similar: HostWritableStorage<D>,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    structure.validate_replay_structures(&dst_structure, &src_structure)?;
    validate_replay_storage_len(&dst_structure, dst.storage().len())?;
    validate_replay_storage_len(&src_structure, src.storage().len())?;

    let mut zero_strides = Vec::new();
    let src_data = src.data();
    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                &mut zero_strides,
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
                let mut scratch = TreeTransformScratchBuffers::from_parts(
                    src.similar_storage_filled(source_len, D::zero()),
                    dst.similar_storage_filled(destination_len, D::zero()),
                );

                for src_index in 0..src_count {
                    let layout = structure.layouts.entry(src_layout_start + src_index);
                    pack_layout_into_column(
                        &structure.layouts,
                        layout,
                        src_data,
                        scratch.source_mut().as_mut_slice(),
                        src_index * element_count,
                        structure.storage_conjugate(),
                    )?;
                }

                {
                    let (source, destination) = scratch.source_and_destination_mut();
                    apply_recoupling_matrix_src_times_u_transpose(
                        destination.as_mut_slice(),
                        source.as_slice(),
                        &structure.coefficients_src_by_dst,
                        coefficient_start,
                        element_count,
                        src_count,
                        dst_count,
                    )?;
                }

                for dst_index in 0..dst_count {
                    let layout = structure.layouts.entry(dst_layout_start + dst_index);
                    scatter_column_into_layout(
                        &mut zero_strides,
                        &structure.layouts,
                        layout,
                        scratch.destination().as_slice(),
                        dst_index * element_count,
                        dst.data_mut(),
                        alpha,
                        beta,
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Replays a prepared tree-transform structure on host slices.
pub(crate) fn tree_transform_structure_with_strided_kernel_raw<D, C>(
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
                workspace,
                &structure.layouts,
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
                &structure.coefficients_src_by_dst,
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

pub(crate) fn tree_transform_structure_with_structural_recoupling<
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
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    tree_transform_structure_with_structural_recoupling_raw(
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
pub(crate) fn tree_transform_structure_with_structural_recoupling_raw<E, D, C>(
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
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
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
            } => tree_transform_multi_with_structural_recoupling(
                dense,
                workspace,
                &structure.layouts,
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
                &structure.coefficients_src_by_dst,
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
pub(crate) fn tree_transform_structure_with_structural_recoupling_raw_profiled<E, D, C>(
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

    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => {
                profile.single_blocks += 1;
                let start = std::time::Instant::now();
                tree_transform_single_with_strided_kernel_profiled(
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
                    profile,
                )?;
                profile.single_total += start.elapsed();
            }
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => {
                profile.multi_blocks += 1;
                tree_transform_multi_with_structural_recoupling_profiled(
                    dense,
                    workspace,
                    &structure.layouts,
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    coefficient_start,
                    element_count,
                    &structure.coefficients_src_by_dst,
                    structure.storage_conjugate(),
                    dst_data,
                    src_data,
                    alpha,
                    beta,
                    profile,
                )?;
            }
        }
    }

    profile.total += total_start.elapsed();
    Ok(())
}

pub(crate) fn tensoradd_block_with_strided_kernel<T>(
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

fn tree_transform_single_with_strided_kernel<D, C>(
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
    let shape = layouts.shape(dst_layout);
    let scale = alpha.scale_by_coefficient(coefficient);
    tensoradd_raw_strided_kernel_trusted(
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

#[allow(clippy::too_many_arguments)]
fn tree_transform_single_with_strided_kernel_profiled<D, C>(
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
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
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
    let shape = layouts.shape(dst_layout);
    let scale = alpha.scale_by_coefficient(coefficient);
    tensoradd_raw_strided_kernel_profiled(
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
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_pack_gemm_scatter<D, C>(
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    coefficients_src_by_dst: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
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
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    workspace.prepare_packed_buffers(source_len, destination_len, D::zero());

    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column(
            layouts,
            layout,
            src_data,
            workspace.packed.source_mut().as_mut_slice(),
            src_index * element_count,
            source_conjugate,
        )?;
    }

    {
        let (source, destination) = workspace.packed.source_and_destination_mut();
        apply_recoupling_matrix_src_times_u_transpose(
            destination.as_mut_slice(),
            source.as_slice(),
            coefficients_src_by_dst,
            coefficient_start,
            element_count,
            src_count,
            dst_count,
        )?;
    }

    for dst_index in 0..dst_count {
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout(
            &mut workspace.zero_strides,
            layouts,
            layout,
            workspace.packed.destination().as_slice(),
            dst_index * element_count,
            dst_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_structural_recoupling<E, D, C>(
    _dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    coefficients_src_by_dst: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    workspace.prepare_packed_buffers(source_len, destination_len, D::zero());

    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column(
            layouts,
            layout,
            src_data,
            workspace.packed.source_mut().as_mut_slice(),
            src_index * element_count,
            source_conjugate,
        )?;
    }

    {
        let (source, destination) = workspace.packed.source_and_destination_mut();
        apply_recoupling_matrix_src_times_u_transpose(
            destination.as_mut_slice(),
            source.as_slice(),
            coefficients_src_by_dst,
            coefficient_start,
            element_count,
            src_count,
            dst_count,
        )?;
    }

    for dst_index in 0..dst_count {
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout(
            &mut workspace.zero_strides,
            layouts,
            layout,
            workspace.packed.destination().as_slice(),
            dst_index * element_count,
            dst_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_structural_recoupling_profiled<E, D, C>(
    _dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    coefficients_src_by_dst: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;

    let start = std::time::Instant::now();
    workspace.prepare_packed_buffers(source_len, destination_len, D::zero());
    profile.multi_workspace_prepare += start.elapsed();

    let start = std::time::Instant::now();
    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column_profiled(
            layouts,
            layout,
            src_data,
            workspace.packed.source_mut().as_mut_slice(),
            src_index * element_count,
            source_conjugate,
            profile,
        )?;
        profile.packed_columns += 1;
    }
    profile.multi_pack += start.elapsed();

    let start = std::time::Instant::now();
    {
        let (source, destination) = workspace.packed.source_and_destination_mut();
        apply_recoupling_matrix_src_times_u_transpose(
            destination.as_mut_slice(),
            source.as_slice(),
            coefficients_src_by_dst,
            coefficient_start,
            element_count,
            src_count,
            dst_count,
        )?;
    }
    let elapsed = start.elapsed();
    profile.multi_scalar_recoupling += elapsed;
    profile.multi_matmul_total += elapsed;

    let start = std::time::Instant::now();
    for dst_index in 0..dst_count {
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout_profiled(
            &mut workspace.zero_strides,
            layouts,
            layout,
            workspace.packed.destination().as_slice(),
            dst_index * element_count,
            dst_data,
            alpha,
            beta,
            profile,
        )?;
        profile.scattered_columns += 1;
    }
    profile.multi_scatter += start.elapsed();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_recoupling_matrix_src_times_u_transpose<D, C>(
    destination: &mut [D],
    source: &[D],
    coefficients_src_by_dst: &[C],
    coefficient_start: usize,
    element_count: usize,
    src_count: usize,
    dst_count: usize,
) -> Result<(), OperationError>
where
    D: Copy + Add<D, Output = D> + Zero + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_end = coefficient_start
        .checked_add(coefficient_count)
        .ok_or(OperationError::ElementCountOverflow)?;

    if source.len() != source_len {
        return Err(OperationError::ElementCountMismatch {
            expected: source_len,
            actual: source.len(),
        });
    }
    if destination.len() != destination_len {
        return Err(OperationError::ElementCountMismatch {
            expected: destination_len,
            actual: destination.len(),
        });
    }
    if coefficients_src_by_dst.len() < coefficient_end {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_end,
            actual: coefficients_src_by_dst.len(),
        });
    }

    // TensorKit's dense-vector GenericTreeTransformer uses `U[dst, src]` and
    // computes `buffer_dst = buffer_src * transpose(U)` after packing source
    // trees as columns. Keep this as the backend-replaceable boundary.
    for dst_index in 0..dst_count {
        let dst_column_start = dst_index * element_count;
        let coefficient_row_start = coefficient_start + dst_index * src_count;
        for element in 0..element_count {
            let mut sum = D::zero();
            for src_index in 0..src_count {
                let coeff = coefficients_src_by_dst[coefficient_row_start + src_index];
                let src_value = source[element + src_index * element_count];
                sum = sum + src_value.scale_by_coefficient(coeff);
            }
            destination[dst_column_start + element] = sum;
        }
    }
    Ok(())
}

fn pack_layout_into_column<T>(
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    src_data: &[T],
    packed: &mut [T],
    packed_offset: usize,
    source_conjugate: bool,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    let shape = layouts.shape(layout);
    let packed_offset = offset_to_isize(packed_offset)?;
    copy_scale_raw_strided_kernel_with_conjugate_trusted(
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

fn pack_layout_into_column_profiled<T>(
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    src_data: &[T],
    packed: &mut [T],
    packed_offset: usize,
    source_conjugate: bool,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    let shape = layouts.shape(layout);
    let start = std::time::Instant::now();
    let packed_offset = offset_to_isize(packed_offset)?;
    let result = copy_scale_raw_strided_kernel_with_conjugate_trusted(
        packed,
        src_data,
        shape,
        layouts.packed_strides(layout),
        layouts.strides(layout),
        packed_offset,
        layout.offset,
        source_conjugate,
        T::one(),
    );
    profile.strided_kernel += start.elapsed();
    result
}

fn scatter_column_into_layout<T>(
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
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    let shape = layouts.shape(layout);
    zero_strides.clear();
    axpby_raw_strided_kernel_trusted(
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

#[allow(clippy::too_many_arguments)]
fn scatter_column_into_layout_profiled<T>(
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    packed: &[T],
    packed_offset: usize,
    dst_data: &mut [T],
    alpha: T,
    beta: T,
    profile: &mut TreeTransformReplayProfile,
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
    let shape = layouts.shape(layout);
    let start = std::time::Instant::now();
    let packed_offset = offset_to_isize(packed_offset)?;
    let result = axpby_raw_strided_kernel_trusted(
        dst_data,
        packed,
        shape,
        layouts.strides(layout),
        layouts.packed_strides(layout),
        layout.offset,
        packed_offset,
        alpha,
        beta,
    );
    profile.strided_kernel += start.elapsed();
    zero_strides.clear();
    result
}
