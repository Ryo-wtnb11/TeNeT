use core::ops::{Add, Mul};
use std::cell::RefCell;

use num_traits::{One, Zero};

use crate::{
    axpby_raw_strided_kernel_trusted, scale_raw_strided_kernel_trusted,
    tensoradd_raw_strided_kernel_trusted, ConjugateValue, OperationError,
    RecouplingCoefficientAction,
};

thread_local! {
    /// Reused fused-loop scratch so replay copies allocate nothing after warmup,
    /// for ANY rank. The previous stack-array fast path capped rank at 8 and fell
    /// back to an allocating strided-view kernel (StridedView::new + plan build)
    /// for the rank>8 contraction intermediates that dominate warm replay.
    /// Reusing one buffer per thread mirrors TensorOperations.jl reusing a
    /// temporaries allocator instead of allocating per contraction.
    static FUSE_SCRATCH: RefCell<FuseScratch> = const { RefCell::new(FuseScratch::new()) };
}

#[derive(Default)]
struct FuseScratch {
    dims: Vec<usize>,
    dst_strides: Vec<isize>,
    src_strides: Vec<isize>,
    index: Vec<usize>,
}

impl FuseScratch {
    const fn new() -> Self {
        Self {
            dims: Vec::new(),
            dst_strides: Vec::new(),
            src_strides: Vec::new(),
            index: Vec::new(),
        }
    }
}

/// Runs `dst = apply(dst, op(src))` over one (destination, source) strided view
/// pair with a plain loop nest and NO per-call allocation, for any rank.
///
/// The fused layout (extent-1 axes dropped, remaining axes ordered by
/// destination stride, adjacent contiguous axes merged) is built into the
/// thread-local scratch whose backing buffers are retained across calls, so the
/// hot replay path never touches the heap after warmup. Safe indexing keeps an
/// out-of-bounds layout a panic rather than undefined behavior.
#[allow(clippy::too_many_arguments)]
fn fused_pair<T, Apply, ElementOp>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    apply: Apply,
    op: ElementOp,
) where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    if shape.iter().any(|&dim| dim == 0) {
        return;
    }
    FUSE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let FuseScratch {
            dims,
            dst_strides: fdst,
            src_strides: fsrc,
            index,
        } = &mut *scratch;
        dims.clear();
        fdst.clear();
        fsrc.clear();
        // insertion sort by destination stride, dropping extent-1 axes
        for axis in 0..shape.len() {
            let extent = shape[axis];
            if extent == 1 {
                continue;
            }
            let d = dst_strides[axis];
            let s = src_strides[axis];
            let mut position = dims.len();
            while position > 0 && fdst[position - 1] > d {
                position -= 1;
            }
            dims.insert(position, extent);
            fdst.insert(position, d);
            fsrc.insert(position, s);
        }
        if dims.is_empty() {
            dims.push(1);
            fdst.push(0);
            fsrc.push(0);
        }
        // fuse adjacent contiguous axes
        let mut fused = 0usize;
        for axis in 1..dims.len() {
            let extent = dims[fused] as isize;
            if fdst[fused] * extent == fdst[axis] && fsrc[fused] * extent == fsrc[axis] {
                dims[fused] *= dims[axis];
            } else {
                fused += 1;
                dims[fused] = dims[axis];
                fdst[fused] = fdst[axis];
                fsrc[fused] = fsrc[axis];
            }
        }
        let rank = fused + 1;
        dims.truncate(rank);
        fdst.truncate(rank);
        fsrc.truncate(rank);
        index.clear();
        index.resize(rank, 0);

        let inner_len = dims[0];
        let inner_dst = fdst[0];
        let inner_src = fsrc[0];
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
                if axis >= rank {
                    return;
                }
                index[axis] += 1;
                dst_base += fdst[axis];
                src_base += fsrc[axis];
                if index[axis] < dims[axis] {
                    break;
                }
                dst_base -= dims[axis] as isize * fdst[axis];
                src_base -= dims[axis] as isize * fsrc[axis];
                index[axis] = 0;
                axis += 1;
            }
        }
    });
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
            let assign = beta.is_zero();
            fused_pair(
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                move |dst, value| {
                    if assign {
                        *dst = value;
                    } else {
                        *dst = *dst + value;
                    }
                },
                move |value: T| alpha * value.maybe_conj(source_conjugate),
            );
            zero_strides.clear();
            return Ok(());
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
            let assign = beta.is_zero();
            fused_pair(
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                move |dst, value| {
                    if assign {
                        *dst = value;
                    } else {
                        *dst = *dst + value;
                    }
                },
                move |value: T| alpha * value,
            );
            return Ok(());
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
        fused_pair(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            |dst, value| *dst = value,
            move |value: T| alpha * value.maybe_conj(source_conjugate),
        );
        Ok(())
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
