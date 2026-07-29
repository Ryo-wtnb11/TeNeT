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
pub fn copy_block_with_strided_kernel<T>(
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
pub fn tensoradd_raw_strided_kernel<T>(
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
pub fn tensoradd_raw_strided_kernel_trusted<T>(
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

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensoradd_raw_strided_kernel_mapped<T, D, S>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_stride: D,
    src_stride: S,
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + PartialEq + Zero + One + ConjugateValue,
    D: Copy + Fn(usize) -> Result<isize, OperationError>,
    S: Copy + Fn(usize) -> Result<isize, OperationError>,
{
    validate_raw_strided_bounds_mapped(dst_data.len(), shape, dst_stride, dst_offset)?;
    validate_raw_strided_bounds_mapped(src_data.len(), shape, src_stride, src_offset)?;
    raw_strided_combine_loop_mapped(
        dst_data,
        src_data,
        shape,
        dst_stride,
        src_stride,
        dst_offset,
        src_offset,
        source_conjugate,
        raw_strided_action(alpha, beta),
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn bilinear_raw_strided_kernel_mapped<T, L, R>(
    lhs_data: &[T],
    rhs_data: &[T],
    shape: &[usize],
    lhs_stride: L,
    rhs_stride: R,
    lhs_offset: isize,
    rhs_offset: isize,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
) -> Result<T, OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero + ConjugateValue,
    L: Copy + Fn(usize) -> Result<isize, OperationError>,
    R: Copy + Fn(usize) -> Result<isize, OperationError>,
{
    validate_raw_strided_bounds_mapped(lhs_data.len(), shape, lhs_stride, lhs_offset)?;
    validate_raw_strided_bounds_mapped(rhs_data.len(), shape, rhs_stride, rhs_offset)?;
    let len = crate::strided::element_count(shape)?;
    if len == 0 {
        return Ok(T::zero());
    }
    if shape.is_empty() {
        return Ok(
            lhs_data[checked_offset_to_index(lhs_offset)?].maybe_conj(lhs_conjugate)
                * rhs_data[checked_offset_to_index(rhs_offset)?].maybe_conj(rhs_conjugate),
        );
    }
    bilinear_raw_strided_recurse_mapped(
        shape.len() - 1,
        lhs_data,
        rhs_data,
        shape,
        lhs_stride,
        rhs_stride,
        lhs_offset,
        rhs_offset,
        lhs_conjugate,
        rhs_conjugate,
    )
}

fn validate_raw_strided_bounds_mapped<F>(
    len: usize,
    shape: &[usize],
    stride: F,
    offset: isize,
) -> Result<(), OperationError>
where
    F: Fn(usize) -> Result<isize, OperationError>,
{
    if shape.contains(&0) {
        return Ok(());
    }
    let mut min_offset = offset;
    let mut max_offset = offset;
    for (axis, &dim) in shape.iter().enumerate() {
        if dim <= 1 {
            continue;
        }
        let dim = isize::try_from(dim - 1).map_err(|_| OperationError::ElementCountOverflow)?;
        let end = stride(axis)?
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
fn bilinear_raw_strided_recurse_mapped<T, L, R>(
    axis: usize,
    lhs_data: &[T],
    rhs_data: &[T],
    shape: &[usize],
    lhs_stride: L,
    rhs_stride: R,
    lhs_base: isize,
    rhs_base: isize,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
) -> Result<T, OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero + ConjugateValue,
    L: Copy + Fn(usize) -> Result<isize, OperationError>,
    R: Copy + Fn(usize) -> Result<isize, OperationError>,
{
    let mut sum = T::zero();
    if axis == 0 {
        let lhs_stride = lhs_stride(0)?;
        let rhs_stride = rhs_stride(0)?;
        for index in 0..shape[0] {
            let lhs_index =
                checked_offset_to_index(checked_strided_offset(lhs_base, index, lhs_stride)?)?;
            let rhs_index =
                checked_offset_to_index(checked_strided_offset(rhs_base, index, rhs_stride)?)?;
            sum = sum
                + lhs_data[lhs_index].maybe_conj(lhs_conjugate)
                    * rhs_data[rhs_index].maybe_conj(rhs_conjugate);
        }
        return Ok(sum);
    }
    let lhs_axis_stride = lhs_stride(axis)?;
    let rhs_axis_stride = rhs_stride(axis)?;
    for index in 0..shape[axis] {
        sum = sum
            + bilinear_raw_strided_recurse_mapped(
                axis - 1,
                lhs_data,
                rhs_data,
                shape,
                lhs_stride,
                rhs_stride,
                checked_strided_offset(lhs_base, index, lhs_axis_stride)?,
                checked_strided_offset(rhs_base, index, rhs_axis_stride)?,
                lhs_conjugate,
                rhs_conjugate,
            )?;
    }
    Ok(sum)
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
pub fn axpby_raw_strided_kernel<T>(
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
pub fn axpby_raw_strided_kernel_trusted<T>(
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

pub fn scale_raw_strided_kernel_trusted<T>(
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
pub fn tensortrace_raw_strided_kernel<T>(
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
pub fn tensortrace_raw_strided_kernel_add_with_coefficient<T, C>(
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

pub(crate) fn validate_raw_strided_bounds(
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

    raw_strided_combine_recurse_mapped(
        shape.len() - 1,
        dst_data,
        src_data,
        shape,
        |axis| Ok(dst_strides[axis]),
        |axis| Ok(src_strides[axis]),
        dst_offset,
        src_offset,
        source_conjugate,
        action,
    )
}

#[allow(clippy::too_many_arguments)]
fn raw_strided_combine_loop_mapped<T, D, S>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_stride: D,
    src_stride: S,
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    action: RawStridedAction<T>,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + ConjugateValue,
    D: Copy + Fn(usize) -> Result<isize, OperationError>,
    S: Copy + Fn(usize) -> Result<isize, OperationError>,
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
    raw_strided_combine_recurse_mapped(
        shape.len() - 1,
        dst_data,
        src_data,
        shape,
        dst_stride,
        src_stride,
        dst_offset,
        src_offset,
        source_conjugate,
        action,
    )
}

#[allow(clippy::too_many_arguments)]
fn raw_strided_combine_recurse_mapped<T, D, S>(
    axis: usize,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_stride: D,
    src_stride: S,
    dst_base: isize,
    src_base: isize,
    source_conjugate: bool,
    action: RawStridedAction<T>,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + ConjugateValue,
    D: Copy + Fn(usize) -> Result<isize, OperationError>,
    S: Copy + Fn(usize) -> Result<isize, OperationError>,
{
    if axis == 0 {
        let dst_stride = dst_stride(0)?;
        let src_stride = src_stride(0)?;
        for index in 0..shape[0] {
            let dst_index =
                checked_offset_to_index(checked_strided_offset(dst_base, index, dst_stride)?)?;
            let src_index =
                checked_offset_to_index(checked_strided_offset(src_base, index, src_stride)?)?;
            apply_raw_strided_action(
                &mut dst_data[dst_index],
                src_data[src_index].maybe_conj(source_conjugate),
                action,
            );
        }
        return Ok(());
    }

    let dst_axis_stride = dst_stride(axis)?;
    let src_axis_stride = src_stride(axis)?;
    for index in 0..shape[axis] {
        raw_strided_combine_recurse_mapped(
            axis - 1,
            dst_data,
            src_data,
            shape,
            dst_stride,
            src_stride,
            checked_strided_offset(dst_base, index, dst_axis_stride)?,
            checked_strided_offset(src_base, index, src_axis_stride)?,
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

/// Overflow signal for the per-element strided offset helpers.
///
/// Fieldless on purpose: these helpers run once per element on the
/// non-contiguous combine/scale recurse path, where returning
/// `Result<_, OperationError>` (536 bytes) forced a per-element sret move +
/// drop that dominated the profile (issue #230). This ZST-sized error keeps
/// the hot `Result` pointer-small.
///
/// Why-not shrink `OperationError` itself: that is issue #231 (parked) and
/// would ripple through every operation call site; a local error confined to
/// this helper family fixes the hot path without touching the shared enum.
#[derive(Clone, Copy, Debug)]
enum OffsetError {
    /// index/stride `isize` arithmetic overflowed.
    ElementCount,
    /// signed offset did not fit back into `usize`.
    Offset,
}

impl From<OffsetError> for OperationError {
    fn from(err: OffsetError) -> Self {
        // Map to the exact variants/messages these helpers emitted before #230
        // so every `?` call site stays observably identical.
        match err {
            OffsetError::ElementCount => OperationError::ElementCountOverflow,
            OffsetError::Offset => OperationError::OffsetOverflow { value: usize::MAX },
        }
    }
}

fn checked_strided_offset(base: isize, index: usize, stride: isize) -> Result<isize, OffsetError> {
    let index = isize::try_from(index).map_err(|_| OffsetError::ElementCount)?;
    base.checked_add(index.checked_mul(stride).ok_or(OffsetError::ElementCount)?)
        .ok_or(OffsetError::ElementCount)
}

fn checked_offset_to_index(offset: isize) -> Result<usize, OffsetError> {
    usize::try_from(offset).map_err(|_| OffsetError::Offset)
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

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    // What: the offset helper family's Result must stay pointer-small so the
    // per-element non-contiguous combine/scale loop pays no 536-byte sret move
    // + drop (issue #230). Fails if OffsetError grows a field or OperationError
    // is routed back onto these hot helpers.
    #[test]
    fn offset_error_result_stays_small() {
        assert!(std::mem::size_of::<Result<usize, OffsetError>>() <= 16);
    }

    // What: the offset helpers still emit the exact OperationError variants they
    // did before #230, so `?` call sites are observably unchanged.
    #[test]
    fn offset_helpers_map_to_original_operation_errors() {
        // isize::MAX * 2 overflows the checked_mul -> ElementCountOverflow.
        let err: OperationError = checked_strided_offset(0, 2, isize::MAX).unwrap_err().into();
        assert_eq!(err, OperationError::ElementCountOverflow);

        // A negative offset cannot be a usize index -> OffsetOverflow{usize::MAX}.
        let err: OperationError = checked_offset_to_index(-1).unwrap_err().into();
        assert_eq!(err, OperationError::OffsetOverflow { value: usize::MAX });
    }

    #[test]
    fn bilinear_kernel_handles_independent_conjugation_and_padding() {
        let lhs = [
            Complex64::new(99.0, 0.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(98.0, 0.0),
            Complex64::new(-2.0, 0.5),
            Complex64::new(4.0, 3.0),
        ];
        let rhs = [
            Complex64::new(5.0, -1.0),
            Complex64::new(97.0, 0.0),
            Complex64::new(2.0, 4.0),
            Complex64::new(-3.0, 2.0),
            Complex64::new(96.0, 0.0),
            Complex64::new(1.0, -2.0),
        ];
        let shape = [2, 2];
        let lhs_strides = [1, 3];
        let rhs_strides = [2, 3];
        let lhs_values = [lhs[1], lhs[2], lhs[4], lhs[5]];
        let rhs_values = [rhs[0], rhs[2], rhs[3], rhs[5]];
        for (lhs_conjugate, rhs_conjugate) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let expected = lhs_values.iter().copied().zip(rhs_values).fold(
                Complex64::new(0.0, 0.0),
                |sum, (lhs, rhs)| {
                    sum + lhs.maybe_conj(lhs_conjugate) * rhs.maybe_conj(rhs_conjugate)
                },
            );
            assert_eq!(
                bilinear_raw_strided_kernel_mapped(
                    &lhs,
                    &rhs,
                    &shape,
                    |axis| Ok(lhs_strides[axis]),
                    |axis| Ok(rhs_strides[axis]),
                    1,
                    0,
                    lhs_conjugate,
                    rhs_conjugate,
                )
                .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn bilinear_kernel_validates_each_view_before_reading() {
        assert_eq!(
            bilinear_raw_strided_kernel_mapped(
                &[1.0],
                &[2.0],
                &[2],
                |_| Ok(1),
                |_| Ok(1),
                0,
                0,
                false,
                false,
            )
            .unwrap_err(),
            OperationError::OffsetOverflow { value: 1 }
        );
    }

    #[test]
    fn bilinear_kernel_accepts_checked_negative_strides() {
        assert_eq!(
            bilinear_raw_strided_kernel_mapped(
                &[1.0, 2.0, 3.0],
                &[4.0, 5.0, 6.0],
                &[3],
                |_| Ok(-1),
                |_| Ok(1),
                2,
                0,
                false,
                false,
            )
            .unwrap(),
            3.0 * 4.0 + 2.0 * 5.0 + 1.0 * 6.0
        );
    }
}
