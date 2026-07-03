use core::ops::{Add, Mul};

use num_traits::{One, Zero};
use tenet_core::{BlockView, BlockViewMut};

use crate::strided::{error as strided_error, read as strided_read, write as strided_write};
use crate::{ConjugateValue, OperationError};

/// Host scalar strided kernel boundary.
///
/// This module owns the current host-slice scalar kernels used by tensoradd,
/// pack, scatter, and scale replay. Higher-level tree/fusion algorithms should
/// call these primitives instead of embedding raw strided loops directly.
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
pub(crate) fn tensoradd_raw_strided_kernel_trusted<T>(
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
        return tensoradd_raw_strided_conjugating_kernel_trusted(
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
    axpby_raw_strided_kernel_trusted(
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

#[allow(clippy::too_many_arguments)]
fn tensoradd_raw_strided_conjugating_kernel_trusted<T>(
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
    #[cfg(debug_assertions)]
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
pub(crate) fn axpby_raw_strided_kernel_trusted<T>(
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
    #[cfg(debug_assertions)]
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

pub(crate) fn scale_raw_strided_kernel_trusted<T>(
    dst_data: &mut [T],
    shape: &[usize],
    dst_strides: &[isize],
    dst_offset: isize,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Mul<T, Output = T>,
{
    #[cfg(debug_assertions)]
    validate_raw_strided_bounds(dst_data.len(), shape, dst_strides, dst_offset)?;
    raw_strided_scale_loop(dst_data, shape, dst_strides, dst_offset, beta)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensortrace_raw_strided_kernel<T>(
    dst_data: &mut [T],
    src_data: &[T],
    output_shape: &[usize],
    trace_shape: &[usize],
    dst_strides: &[isize],
    src_output_strides: &[isize],
    src_trace_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + PartialEq + Zero + One + ConjugateValue,
{
    let output_len = crate::strided::element_count(output_shape)?;
    let trace_len = crate::strided::element_count(trace_shape)?;
    for output_linear in 0..output_len {
        let dst_index =
            strided_linear_offset(output_linear, output_shape, dst_strides, dst_offset)?;
        let src_base =
            strided_linear_offset(output_linear, output_shape, src_output_strides, src_offset)?;
        let src_base = isize::try_from(src_base)
            .map_err(|_| OperationError::OffsetOverflow { value: src_base })?;
        let mut sum = T::zero();
        for trace_linear in 0..trace_len {
            let src_index =
                strided_linear_offset(trace_linear, trace_shape, src_trace_strides, src_base)?;
            sum = sum + src_data[src_index].maybe_conj(source_conjugate);
        }
        let value = alpha * sum;
        dst_data[dst_index] = if beta.is_zero() {
            value
        } else {
            beta * dst_data[dst_index] + value
        };
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensortrace_raw_strided_kernel_add_with_coefficient<T, C>(
    dst_data: &mut [T],
    src_data: &[T],
    output_shape: &[usize],
    trace_shape: &[usize],
    dst_strides: &[isize],
    src_output_strides: &[isize],
    src_trace_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    alpha: T,
    coefficient: C,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + ConjugateValue
        + crate::RecouplingCoefficientAction<C>,
    C: Copy,
{
    let output_len = crate::strided::element_count(output_shape)?;
    let trace_len = crate::strided::element_count(trace_shape)?;
    for output_linear in 0..output_len {
        let dst_index =
            strided_linear_offset(output_linear, output_shape, dst_strides, dst_offset)?;
        let src_base =
            strided_linear_offset(output_linear, output_shape, src_output_strides, src_offset)?;
        let src_base = isize::try_from(src_base)
            .map_err(|_| OperationError::OffsetOverflow { value: src_base })?;
        let mut sum = T::zero();
        for trace_linear in 0..trace_len {
            let src_index =
                strided_linear_offset(trace_linear, trace_shape, src_trace_strides, src_base)?;
            sum = sum + src_data[src_index].maybe_conj(source_conjugate);
        }
        let value = (alpha * sum).scale_by_coefficient(coefficient);
        dst_data[dst_index] = dst_data[dst_index] + value;
    }
    Ok(())
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

fn raw_strided_scale_loop<T>(
    dst_data: &mut [T],
    shape: &[usize],
    dst_strides: &[isize],
    dst_offset: isize,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Mul<T, Output = T>,
{
    let len = crate::strided::element_count(shape)?;
    if len == 0 {
        return Ok(());
    }
    if shape.is_empty() {
        let dst_index = checked_offset_to_index(dst_offset)?;
        dst_data[dst_index] = beta * dst_data[dst_index];
        return Ok(());
    }
    if is_column_major_contiguous(shape, dst_strides)? {
        let dst_start = checked_offset_to_index(dst_offset)?;
        let dst_end = dst_start
            .checked_add(len)
            .ok_or(OperationError::ElementCountOverflow)?;
        let dst = dst_data
            .get_mut(dst_start..dst_end)
            .ok_or(OperationError::OffsetOverflow { value: dst_end })?;
        for dst_value in dst.iter_mut() {
            *dst_value = beta * *dst_value;
        }
        return Ok(());
    }

    raw_strided_scale_recurse(
        shape.len() - 1,
        dst_data,
        shape,
        dst_strides,
        dst_offset,
        beta,
    )
}

fn raw_strided_scale_recurse<T>(
    axis: usize,
    dst_data: &mut [T],
    shape: &[usize],
    dst_strides: &[isize],
    dst_base: isize,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Mul<T, Output = T>,
{
    if axis == 0 {
        for index in 0..shape[0] {
            let dst_index =
                checked_offset_to_index(checked_strided_offset(dst_base, index, dst_strides[0])?)?;
            dst_data[dst_index] = beta * dst_data[dst_index];
        }
        return Ok(());
    }

    for index in 0..shape[axis] {
        raw_strided_scale_recurse(
            axis - 1,
            dst_data,
            shape,
            dst_strides,
            checked_strided_offset(dst_base, index, dst_strides[axis])?,
            beta,
        )?;
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

fn strided_linear_offset(
    mut linear: usize,
    shape: &[usize],
    strides: &[isize],
    base: isize,
) -> Result<usize, OperationError> {
    let mut offset = base;
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        let coord = if dim == 0 { 0 } else { linear % dim };
        if dim != 0 {
            linear /= dim;
        }
        let coord = isize::try_from(coord).map_err(|_| OperationError::ElementCountOverflow)?;
        offset = offset
            .checked_add(
                coord
                    .checked_mul(stride)
                    .ok_or(OperationError::ElementCountOverflow)?,
            )
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    usize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })
}
