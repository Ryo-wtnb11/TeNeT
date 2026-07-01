use core::ops::{Add, Mul};
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{BlockStructure, BlockView, BlockViewMut, TensorMap};
use tenet_dense::DenseExecutor;

use crate::strided::{
    error as strided_error, offset_to_isize, read as strided_read, write as strided_write,
};
use crate::tensoradd::{TensorAddDescriptor, TensorAddDescriptorTerm};
use crate::{
    ConjugateValue, DenseRecouplingScalar, HostAllocator, OperationError,
    RecouplingCoefficientAction, TensorAddStructure, TreeTransformBlock, TreeTransformLayout,
    TreeTransformLayoutTable, TreeTransformReplayProfile, TreeTransformStructure,
};

#[derive(Clone, Debug)]
pub struct TreeTransformWorkspace<T> {
    zero_strides: Vec<isize>,
    source: Vec<T>,
    destination: Vec<T>,
}

impl<T> Default for TreeTransformWorkspace<T> {
    fn default() -> Self {
        Self {
            zero_strides: Vec::new(),
            source: Vec::new(),
            destination: Vec::new(),
        }
    }
}

impl<T> TreeTransformWorkspace<T> {
    pub fn source_len(&self) -> usize {
        self.source.len()
    }

    pub fn destination_len(&self) -> usize {
        self.destination.len()
    }
}

pub(crate) fn copy_block_with_strided_kernel<T>(
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
) -> Result<(), OperationError>
where
    T: Copy + strided_kernel::MaybeSendSync,
{
    let mut dst = strided_write(dst)?;
    let src = strided_read(src)?;
    strided_kernel::copy_into(&mut dst, &src).map_err(strided_error)
}

pub(crate) fn tensoradd_structure_with_strided_kernel<T, const NOUT: usize, const NIN: usize, S>(
    allocator: &mut HostAllocator,
    structure: &TensorAddStructure,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
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
>(
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
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
>(
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
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
    tensoradd_raw_strided_kernel(
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensoradd_raw_strided_kernel<T>(
    zero_strides: &mut Vec<isize>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
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
    if source_conjugate {
        return tensoradd_raw_strided_conjugating_kernel(
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
        );
    }
    zero_strides.clear();
    axpby_raw_strided_kernel(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
fn tensoradd_raw_strided_kernel_profiled<T>(
    zero_strides: &mut Vec<isize>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
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
    if source_conjugate {
        let start = std::time::Instant::now();
        let result = tensoradd_raw_strided_conjugating_kernel(
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
        );
        profile.strided_kernel += start.elapsed();
        return result;
    }

    let start = std::time::Instant::now();
    let result = axpby_raw_strided_kernel(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        alpha,
        beta,
    );
    profile.strided_kernel += start.elapsed();
    zero_strides.clear();
    result
}

#[allow(clippy::too_many_arguments)]
fn tensoradd_raw_strided_conjugating_kernel<T>(
    zero_strides: &mut Vec<isize>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + PartialEq + Zero + One + ConjugateValue,
{
    validate_raw_strided_views(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
    )?;
    raw_strided_combine_loop(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        true,
        raw_strided_action(alpha, beta),
    )?;
    zero_strides.clear();
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum RawStridedAction<T> {
    CopyScale { alpha: T },
    Axpy { alpha: T },
    Axpby { alpha: T, beta: T },
}

fn raw_strided_action<T>(alpha: T, beta: T) -> RawStridedAction<T>
where
    T: Copy + PartialEq + Zero + One,
{
    if beta.is_zero() {
        RawStridedAction::CopyScale { alpha }
    } else if beta.is_one() {
        RawStridedAction::Axpy { alpha }
    } else {
        RawStridedAction::Axpby { alpha, beta }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn axpby_raw_strided_kernel<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
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
    validate_raw_strided_views(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
    )?;
    raw_strided_combine_loop(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        false,
        raw_strided_action(alpha, beta),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_scale_raw_strided_kernel<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + ConjugateValue,
{
    validate_raw_strided_views(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
    )?;
    raw_strided_combine_loop(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        false,
        RawStridedAction::CopyScale { alpha },
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_raw_strided_views<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
) -> Result<(), OperationError> {
    validate_raw_strided_bounds(dst_data.len(), shape, dst_strides, dst_offset)?;
    validate_raw_strided_bounds(src_data.len(), shape, src_strides, src_offset)?;
    Ok(())
}

fn validate_raw_strided_bounds(
    len: usize,
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> Result<(), OperationError> {
    if shape.len() != strides.len() {
        return Err(OperationError::RankMismatch {
            expected: shape.len(),
            actual: strides.len(),
        });
    }
    if shape.iter().any(|&dim| dim == 0) {
        return Ok(());
    }

    let mut min_offset = offset;
    let mut max_offset = offset;
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        if dim <= 1 {
            continue;
        }
        let dim = isize::try_from(dim - 1).map_err(|_| OperationError::ElementCountOverflow)?;
        let end = stride
            .checked_mul(dim)
            .ok_or(OperationError::ElementCountOverflow)?;
        if end >= 0 {
            max_offset = max_offset
                .checked_add(end)
                .ok_or(OperationError::ElementCountOverflow)?;
        } else {
            min_offset = min_offset
                .checked_add(end)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
    }
    if min_offset < 0 {
        return Err(OperationError::OffsetOverflow { value: usize::MAX });
    }
    let max_offset = checked_offset_to_index(max_offset)?;
    if max_offset >= len {
        return Err(OperationError::OffsetOverflow { value: max_offset });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn raw_strided_combine_loop<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    action: RawStridedAction<T>,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + ConjugateValue,
{
    let len = crate::strided::element_count(shape)?;
    if len == 0 {
        return Ok(());
    }
    if shape.is_empty() {
        let dst_index = checked_offset_to_index(dst_offset)?;
        let src_index = checked_offset_to_index(src_offset)?;
        apply_raw_strided_action(
            &mut dst_data[dst_index],
            src_data[src_index].maybe_conj(source_conjugate),
            action,
        );
        return Ok(());
    }
    if is_column_major_contiguous(shape, dst_strides)?
        && is_column_major_contiguous(shape, src_strides)?
    {
        let dst_start = checked_offset_to_index(dst_offset)?;
        let src_start = checked_offset_to_index(src_offset)?;
        let dst_end = dst_start
            .checked_add(len)
            .ok_or(OperationError::ElementCountOverflow)?;
        let src_end = src_start
            .checked_add(len)
            .ok_or(OperationError::ElementCountOverflow)?;
        let dst = dst_data
            .get_mut(dst_start..dst_end)
            .ok_or(OperationError::OffsetOverflow { value: dst_end })?;
        let src = src_data
            .get(src_start..src_end)
            .ok_or(OperationError::OffsetOverflow { value: src_end })?;
        for (dst_value, src_value) in dst.iter_mut().zip(src.iter().copied()) {
            apply_raw_strided_action(dst_value, src_value.maybe_conj(source_conjugate), action);
        }
        return Ok(());
    }

    raw_strided_combine_recurse(
        shape.len() - 1,
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        source_conjugate,
        action,
    )
}

#[allow(clippy::too_many_arguments)]
fn raw_strided_combine_recurse<T>(
    axis: usize,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_base: isize,
    src_base: isize,
    source_conjugate: bool,
    action: RawStridedAction<T>,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + ConjugateValue,
{
    if axis == 0 {
        for index in 0..shape[0] {
            let dst_index =
                checked_offset_to_index(checked_strided_offset(dst_base, index, dst_strides[0])?)?;
            let src_index =
                checked_offset_to_index(checked_strided_offset(src_base, index, src_strides[0])?)?;
            apply_raw_strided_action(
                &mut dst_data[dst_index],
                src_data[src_index].maybe_conj(source_conjugate),
                action,
            );
        }
        return Ok(());
    }

    for index in 0..shape[axis] {
        raw_strided_combine_recurse(
            axis - 1,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            checked_strided_offset(dst_base, index, dst_strides[axis])?,
            checked_strided_offset(src_base, index, src_strides[axis])?,
            source_conjugate,
            action,
        )?;
    }
    Ok(())
}

fn apply_raw_strided_action<T>(dst: &mut T, src: T, action: RawStridedAction<T>)
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
{
    *dst = match action {
        RawStridedAction::CopyScale { alpha } => alpha * src,
        RawStridedAction::Axpy { alpha } => *dst + alpha * src,
        RawStridedAction::Axpby { alpha, beta } => beta * *dst + alpha * src,
    };
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
    tensoradd_raw_strided_kernel(
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
    workspace.source.resize(source_len, D::zero());
    workspace.destination.resize(destination_len, D::zero());

    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column(
            layouts,
            layout,
            src_data,
            &mut workspace.source,
            src_index * element_count,
            source_conjugate,
        )?;
    }

    apply_recoupling_matrix_src_times_u_transpose(
        &mut workspace.destination,
        &workspace.source,
        coefficients_src_by_dst,
        coefficient_start,
        element_count,
        src_count,
        dst_count,
    )?;

    for dst_index in 0..dst_count {
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout(
            &mut workspace.zero_strides,
            layouts,
            layout,
            &workspace.destination,
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
    workspace.source.resize(source_len, D::zero());
    workspace.destination.resize(destination_len, D::zero());

    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column(
            layouts,
            layout,
            src_data,
            &mut workspace.source,
            src_index * element_count,
            source_conjugate,
        )?;
    }

    apply_recoupling_matrix_src_times_u_transpose(
        &mut workspace.destination,
        &workspace.source,
        coefficients_src_by_dst,
        coefficient_start,
        element_count,
        src_count,
        dst_count,
    )?;

    for dst_index in 0..dst_count {
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout(
            &mut workspace.zero_strides,
            layouts,
            layout,
            &workspace.destination,
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
    workspace.source.resize(source_len, D::zero());
    workspace.destination.resize(destination_len, D::zero());
    profile.multi_workspace_prepare += start.elapsed();

    let start = std::time::Instant::now();
    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column_profiled(
            layouts,
            layout,
            src_data,
            &mut workspace.source,
            src_index * element_count,
            source_conjugate,
            profile,
        )?;
        profile.packed_columns += 1;
    }
    profile.multi_pack += start.elapsed();

    let start = std::time::Instant::now();
    apply_recoupling_matrix_src_times_u_transpose(
        &mut workspace.destination,
        &workspace.source,
        coefficients_src_by_dst,
        coefficient_start,
        element_count,
        src_count,
        dst_count,
    )?;
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
            &workspace.destination,
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
    validate_raw_strided_views(
        packed,
        src_data,
        shape,
        layouts.packed_strides(layout),
        layouts.strides(layout),
        packed_offset,
        layout.offset,
    )?;
    raw_strided_combine_loop(
        packed,
        src_data,
        shape,
        layouts.packed_strides(layout),
        layouts.strides(layout),
        packed_offset,
        layout.offset,
        source_conjugate,
        RawStridedAction::CopyScale { alpha: T::one() },
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
    let result = validate_raw_strided_views(
        packed,
        src_data,
        shape,
        layouts.packed_strides(layout),
        layouts.strides(layout),
        packed_offset,
        layout.offset,
    )
    .and_then(|()| {
        raw_strided_combine_loop(
            packed,
            src_data,
            shape,
            layouts.packed_strides(layout),
            layouts.strides(layout),
            packed_offset,
            layout.offset,
            source_conjugate,
            RawStridedAction::CopyScale { alpha: T::one() },
        )
    });
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
    axpby_raw_strided_kernel(
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
    let result = axpby_raw_strided_kernel(
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

fn checked_strided_offset(
    base: isize,
    index: usize,
    stride: isize,
) -> Result<isize, OperationError> {
    let index = isize::try_from(index).map_err(|_| OperationError::ElementCountOverflow)?;
    base.checked_add(
        index
            .checked_mul(stride)
            .ok_or(OperationError::ElementCountOverflow)?,
    )
    .ok_or(OperationError::ElementCountOverflow)
}

fn checked_offset_to_index(offset: isize) -> Result<usize, OperationError> {
    usize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })
}

fn is_column_major_contiguous(shape: &[usize], strides: &[isize]) -> Result<bool, OperationError> {
    let mut expected = 1isize;
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        if dim > 1 && stride != expected {
            return Ok(false);
        }
        let dim = isize::try_from(dim).map_err(|_| OperationError::ElementCountOverflow)?;
        expected = expected
            .checked_mul(dim)
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(true)
}
