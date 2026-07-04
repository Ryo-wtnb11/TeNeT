use core::ops::{Add, Mul};

use num_traits::{One, Zero};

use crate::strided::error as strided_error;
use crate::{
    axpby_raw_strided_kernel_trusted, scale_raw_strided_kernel_trusted,
    tensoradd_raw_strided_kernel_trusted, ConjugateValue, OperationError,
    RecouplingCoefficientAction,
};

fn read_view<'a, T>(
    data: &'a [T],
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> Result<strided_kernel::StridedView<'a, T>, OperationError>
where
    T: Copy,
{
    strided_kernel::StridedView::new(data, shape, strides, offset).map_err(strided_error)
}

fn write_view<'a, T>(
    data: &'a mut [T],
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> Result<strided_kernel::StridedViewMut<'a, T>, OperationError>
where
    T: Copy,
{
    strided_kernel::StridedViewMut::new(data, shape, strides, offset).map_err(strided_error)
}

const FUSED_RANK_LIMIT: usize = 8;

/// Allocation-free fused loop layout for one (destination, source) view pair.
///
/// Axes with extent 1 are dropped, the rest are ordered by destination stride
/// and adjacent axes are fused when both stride patterns are contiguous, so
/// small replay copies avoid per-call heap allocation and plan building.
#[derive(Clone, Copy, Debug)]
struct FusedPairLayout {
    rank: usize,
    dims: [usize; FUSED_RANK_LIMIT],
    dst_strides: [isize; FUSED_RANK_LIMIT],
    src_strides: [isize; FUSED_RANK_LIMIT],
}

fn fuse_pair_layout(
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
) -> Option<FusedPairLayout> {
    if shape.len() > FUSED_RANK_LIMIT {
        return None;
    }
    let mut layout = FusedPairLayout {
        rank: 0,
        dims: [1; FUSED_RANK_LIMIT],
        dst_strides: [0; FUSED_RANK_LIMIT],
        src_strides: [0; FUSED_RANK_LIMIT],
    };
    for axis in 0..shape.len() {
        if shape[axis] == 1 {
            continue;
        }
        if shape[axis] == 0 {
            return Some(FusedPairLayout {
                rank: 1,
                dims: [0; FUSED_RANK_LIMIT],
                dst_strides: [0; FUSED_RANK_LIMIT],
                src_strides: [0; FUSED_RANK_LIMIT],
            });
        }
        let mut position = layout.rank;
        while position > 0 && layout.dst_strides[position - 1] > dst_strides[axis] {
            layout.dims[position] = layout.dims[position - 1];
            layout.dst_strides[position] = layout.dst_strides[position - 1];
            layout.src_strides[position] = layout.src_strides[position - 1];
            position -= 1;
        }
        layout.dims[position] = shape[axis];
        layout.dst_strides[position] = dst_strides[axis];
        layout.src_strides[position] = src_strides[axis];
        layout.rank += 1;
    }
    if layout.rank == 0 {
        layout.rank = 1;
        layout.dims[0] = 1;
    }
    let mut fused = 0usize;
    for axis in 1..layout.rank {
        let extent = layout.dims[fused] as isize;
        if layout.dst_strides[fused] * extent == layout.dst_strides[axis]
            && layout.src_strides[fused] * extent == layout.src_strides[axis]
        {
            layout.dims[fused] *= layout.dims[axis];
        } else {
            fused += 1;
            layout.dims[fused] = layout.dims[axis];
            layout.dst_strides[fused] = layout.dst_strides[axis];
            layout.src_strides[fused] = layout.src_strides[axis];
        }
    }
    layout.rank = fused + 1;
    Some(layout)
}

/// Applies `dst = combine(dst, alpha * op(src))` over a fused layout with a
/// plain loop nest; safe indexing keeps out-of-bounds layouts a panic rather
/// than undefined behavior.
#[allow(clippy::too_many_arguments)]
fn apply_fused_pair<T, Apply, ElementOp>(
    dst_data: &mut [T],
    src_data: &[T],
    layout: &FusedPairLayout,
    dst_offset: isize,
    src_offset: isize,
    apply: Apply,
    op: ElementOp,
) where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    if layout.dims[..layout.rank].iter().any(|&dim| dim == 0) {
        return;
    }
    let inner_len = layout.dims[0];
    let inner_dst = layout.dst_strides[0];
    let inner_src = layout.src_strides[0];
    let mut index = [0usize; FUSED_RANK_LIMIT];
    let mut dst_base = dst_offset;
    let mut src_base = src_offset;
    loop {
        if inner_dst == 1 && inner_src == 1 {
            let dst_start = dst_base as usize;
            let src_start = src_base as usize;
            let dst = &mut dst_data[dst_start..dst_start + inner_len];
            let src = &src_data[src_start..src_start + inner_len];
            for position in 0..inner_len {
                apply(&mut dst[position], op(src[position]));
            }
        } else {
            for position in 0..inner_len {
                let dst_position = (dst_base + position as isize * inner_dst) as usize;
                let src_position = (src_base + position as isize * inner_src) as usize;
                apply(&mut dst_data[dst_position], op(src_data[src_position]));
            }
        }
        let mut axis = 1;
        loop {
            if axis >= layout.rank {
                return;
            }
            index[axis] += 1;
            dst_base += layout.dst_strides[axis];
            src_base += layout.src_strides[axis];
            if index[axis] < layout.dims[axis] {
                break;
            }
            dst_base -= layout.dims[axis] as isize * layout.dst_strides[axis];
            src_base -= layout.dims[axis] as isize * layout.src_strides[axis];
            index[axis] = 0;
            axis += 1;
        }
    }
}

/// Backend-neutral low-level kernel adapter for host-slice replay.
///
/// Replay drivers (tree-transform pack/recoupling/scatter, fusion-block
/// pack/scatter/scale) call these primitives instead of concrete kernel
/// functions, so the low-level execution backend (scalar loops, strided-rs,
/// BLAS, future C++ kernels) is replaceable behind one boundary.
///
/// The data contract is host slices. Device replay needs a separate
/// storage-aware adapter; device storage must not be hidden behind this trait.
pub trait HostKernelAdapter<T> {
    /// `dst = alpha * op(src) + beta * dst` over strided views, where `op` is
    /// conjugation when `source_conjugate` is set (tensoradd / single-block
    /// tree replay primitive).
    #[allow(clippy::too_many_arguments)]
    fn add_strided(
        &mut self,
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
    ) -> Result<(), OperationError>;

    /// `dst = alpha * src + beta * dst` over strided views without
    /// conjugation (scatter primitive).
    #[allow(clippy::too_many_arguments)]
    fn axpby_strided(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>;

    /// `dst = alpha * op(src)` over strided views (pack primitive).
    #[allow(clippy::too_many_arguments)]
    fn copy_scale_strided(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
    ) -> Result<(), OperationError>;

    /// `dst = beta * dst` over a strided block (inactive-block scale
    /// primitive).
    fn scale_strided(
        &mut self,
        dst_data: &mut [T],
        shape: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
        beta: T,
    ) -> Result<(), OperationError>;

    /// `destination[:, dst] = Σ_src coefficient[dst, src] * source[:, src]`
    /// over packed tree columns.
    ///
    /// TensorKit's dense-vector GenericTreeTransformer uses `U[dst, src]` and
    /// computes `buffer_dst = buffer_src * transpose(U)` after packing source
    /// trees as columns. This is the BLAS/GEMM replacement point for the
    /// recoupling matrix application.
    #[allow(clippy::too_many_arguments)]
    fn recoupling_src_times_u_transpose<C>(
        &mut self,
        destination: &mut [T],
        source: &[T],
        recoupling_coefficients_dst_src: &[C],
        coefficient_start: usize,
        element_count: usize,
        src_count: usize,
        dst_count: usize,
    ) -> Result<(), OperationError>
    where
        C: Copy,
        T: RecouplingCoefficientAction<C>;
}

/// Default host kernel adapter backed by the strided-rs style raw kernels.
///
/// The recoupling matrix application is currently a scalar loop; swapping it
/// for a BLAS/GEMM call happens by replacing this adapter, not by editing the
/// replay drivers.
#[derive(Clone, Copy, Debug, Default)]
pub struct StridedHostKernelAdapter;

impl<T> HostKernelAdapter<T> for StridedHostKernelAdapter
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
    fn add_strided(
        &mut self,
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
    ) -> Result<(), OperationError> {
        if beta.is_zero() || beta.is_one() {
            if let Some(layout) = fuse_pair_layout(shape, dst_strides, src_strides) {
                let assign = beta.is_zero();
                apply_fused_pair(
                    dst_data,
                    src_data,
                    &layout,
                    dst_offset,
                    src_offset,
                    |dst, value| {
                        if assign {
                            *dst = value;
                        } else {
                            *dst = *dst + value;
                        }
                    },
                    |value: T| alpha * value.maybe_conj(source_conjugate),
                );
                zero_strides.clear();
                return Ok(());
            }
            let src = read_view(src_data, shape, src_strides, src_offset)?;
            let mut dst = write_view(dst_data, shape, dst_strides, dst_offset)?;
            let result = match (beta.is_zero(), source_conjugate) {
                (true, false) => strided_kernel::copy_scale(&mut dst, &src, alpha),
                (true, true) => strided_kernel::copy_scale(&mut dst, &src.conj(), alpha),
                (false, false) => strided_kernel::axpy(&mut dst, &src, alpha),
                (false, true) => strided_kernel::axpy(&mut dst, &src.conj(), alpha),
            };
            zero_strides.clear();
            return result.map_err(strided_error);
        }
        tensoradd_raw_strided_kernel_trusted(
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            beta,
        )
    }

    fn axpby_strided(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError> {
        if beta.is_zero() || beta.is_one() {
            if let Some(layout) = fuse_pair_layout(shape, dst_strides, src_strides) {
                let assign = beta.is_zero();
                apply_fused_pair(
                    dst_data,
                    src_data,
                    &layout,
                    dst_offset,
                    src_offset,
                    |dst, value| {
                        if assign {
                            *dst = value;
                        } else {
                            *dst = *dst + value;
                        }
                    },
                    |value: T| alpha * value,
                );
                return Ok(());
            }
            let src = read_view(src_data, shape, src_strides, src_offset)?;
            let mut dst = write_view(dst_data, shape, dst_strides, dst_offset)?;
            let result = if beta.is_zero() {
                strided_kernel::copy_scale(&mut dst, &src, alpha)
            } else {
                strided_kernel::axpy(&mut dst, &src, alpha)
            };
            return result.map_err(strided_error);
        }
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

    fn copy_scale_strided(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
    ) -> Result<(), OperationError> {
        if let Some(layout) = fuse_pair_layout(shape, dst_strides, src_strides) {
            apply_fused_pair(
                dst_data,
                src_data,
                &layout,
                dst_offset,
                src_offset,
                |dst, value| *dst = value,
                |value: T| alpha * value.maybe_conj(source_conjugate),
            );
            return Ok(());
        }
        let src = read_view(src_data, shape, src_strides, src_offset)?;
        let mut dst = write_view(dst_data, shape, dst_strides, dst_offset)?;
        if source_conjugate {
            strided_kernel::copy_scale(&mut dst, &src.conj(), alpha).map_err(strided_error)
        } else {
            strided_kernel::copy_scale(&mut dst, &src, alpha).map_err(strided_error)
        }
    }

    fn scale_strided(
        &mut self,
        dst_data: &mut [T],
        shape: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
        beta: T,
    ) -> Result<(), OperationError> {
        scale_raw_strided_kernel_trusted(dst_data, shape, dst_strides, dst_offset, beta)
    }

    fn recoupling_src_times_u_transpose<C>(
        &mut self,
        destination: &mut [T],
        source: &[T],
        recoupling_coefficients_dst_src: &[C],
        coefficient_start: usize,
        element_count: usize,
        src_count: usize,
        dst_count: usize,
    ) -> Result<(), OperationError>
    where
        C: Copy,
        T: RecouplingCoefficientAction<C>,
    {
        validate_recoupling_lens(
            destination.len(),
            source.len(),
            recoupling_coefficients_dst_src.len(),
            coefficient_start,
            element_count,
            src_count,
            dst_count,
        )?;
        for dst_index in 0..dst_count {
            let dst_column_start = dst_index * element_count;
            let coefficient_row_start = coefficient_start + dst_index * src_count;
            for element in 0..element_count {
                let mut sum = T::zero();
                for src_index in 0..src_count {
                    let coeff = recoupling_coefficients_dst_src[coefficient_row_start + src_index];
                    let src_value = source[element + src_index * element_count];
                    sum = sum + src_value.scale_by_coefficient(coeff);
                }
                destination[dst_column_start + element] = sum;
            }
        }
        Ok(())
    }
}

/// Shared dimension validation for recoupling matrix application.
///
/// All adapter implementations should validate against the same packed-column
/// layout before touching data.
pub fn validate_recoupling_lens(
    destination_len: usize,
    source_len: usize,
    coefficient_len: usize,
    coefficient_start: usize,
    element_count: usize,
    src_count: usize,
    dst_count: usize,
) -> Result<(), OperationError> {
    let expected_source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let expected_destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_end = coefficient_start
        .checked_add(coefficient_count)
        .ok_or(OperationError::ElementCountOverflow)?;

    if source_len != expected_source_len {
        return Err(OperationError::ElementCountMismatch {
            expected: expected_source_len,
            actual: source_len,
        });
    }
    if destination_len != expected_destination_len {
        return Err(OperationError::ElementCountMismatch {
            expected: expected_destination_len,
            actual: destination_len,
        });
    }
    if coefficient_len < coefficient_end {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_end,
            actual: coefficient_len,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_host_adapter_add_strided_matches_axpby_semantics() {
        let mut adapter = StridedHostKernelAdapter;
        let mut zero_strides = Vec::new();
        let mut dst = [10.0_f64, 20.0];
        let src = [2.0_f64, 3.0];

        adapter
            .add_strided(
                &mut zero_strides,
                &mut dst,
                &src,
                &[2],
                &[1],
                &[1],
                0,
                0,
                false,
                2.0,
                3.0,
            )
            .unwrap();

        assert_eq!(dst, [34.0, 66.0]);
    }

    #[test]
    fn strided_host_adapter_recoupling_applies_u_transpose() {
        let mut adapter = StridedHostKernelAdapter;
        // Two source columns of two elements, two destination columns:
        // destination[:, d] = sum_s U[d, s] * source[:, s].
        let source = [1.0_f64, 2.0, 10.0, 20.0];
        let mut destination = [0.0_f64; 4];
        let coefficients = [1.0_f64, 0.5, -1.0, 2.0];

        adapter
            .recoupling_src_times_u_transpose(&mut destination, &source, &coefficients, 0, 2, 2, 2)
            .unwrap();

        assert_eq!(destination, [6.0, 12.0, 19.0, 38.0]);
    }

    #[test]
    fn recoupling_len_validation_rejects_mismatched_columns() {
        let err = validate_recoupling_lens(4, 3, 4, 0, 2, 2, 2).unwrap_err();
        assert_eq!(
            err,
            OperationError::ElementCountMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }
}
