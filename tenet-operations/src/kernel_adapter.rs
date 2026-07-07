use core::ops::{Add, Mul};
use std::cell::RefCell;

use num_traits::{One, Zero};

use crate::{
    axpby_raw_strided_kernel_trusted, scale_raw_strided_kernel_trusted,
    tensoradd_raw_strided_kernel_trusted, ConjugateValue, OperationError,
    RecouplingCoefficientAction,
};

/// Whether to route pure permuted copies (pack / assign-scatter) through the
/// HPTT-style blocked micro-kernel transpose (`strided_perm::copy_into` via
/// `strided_kernel::copy_into_col_major`) instead of the naive fused loop.
///
/// Env-gated (`TENET_HPTT=1`) so the win can be A/B measured: HPTT's blocking
/// pays off for large blocks but its per-call plan build can lose against the
/// zero-alloc fused loop on the many tiny blocks of small-D SU(2) replay.
#[inline]
fn hptt_enabled() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("TENET_HPTT").as_deref() == Ok("1"))
}

/// HPTT-backed permuted copy `dst = src` over strided views (no scale, no
/// conjugate). Returns `Ok(true)` when HPTT handled the copy, `Ok(false)` when
/// the layout is outside HPTT's supported class and the caller must fall back
/// to the fused loop.
///
/// HPTT's transpose micro-kernel assumes each side has a genuine stride-1
/// axis (the classic row-major↔transpose case). A pack that gathers a strided
/// *slice* of a larger storage can have no stride-1 axis once extent-1 axes
/// are dropped (its contiguous axis was a singleton) — HPTT would then treat
/// the smallest-stride axis as if it were stride-1 and silently corrupt. We
/// detect that and decline.
#[allow(clippy::too_many_arguments)]
fn hptt_permuted_copy<T: Copy + strided_kernel::MaybeSendSync>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
) -> Result<bool, OperationError> {
    // Drop extent-1 axes (their strides are irrelevant, and the HPTT planner is
    // not robust to extent-1 axes carrying colliding strides). `fused_pair`
    // does the same normalization.
    use smallvec::SmallVec;
    let mut rshape: SmallVec<[usize; 8]> = SmallVec::new();
    let mut rdst: SmallVec<[isize; 8]> = SmallVec::new();
    let mut rsrc: SmallVec<[isize; 8]> = SmallVec::new();
    for axis in 0..shape.len() {
        if shape[axis] != 1 {
            rshape.push(shape[axis]);
            rdst.push(dst_strides[axis]);
            rsrc.push(src_strides[axis]);
        }
    }
    if rshape.is_empty() {
        // single element (all axes extent 1)
        dst_data[dst_offset as usize] = src_data[src_offset as usize];
        return Ok(true);
    }
    // Eligibility: each side needs a real stride-1 axis for the micro-kernel.
    if !rsrc.contains(&1) || !rdst.contains(&1) {
        return Ok(false);
    }
    // View construction validates the same bounding box fused_pair accesses
    // (max index = offset + Σ(dim-1)·stride < len), so with the positive-stride
    // guard it only errors on a genuine out-of-bounds layout — an upstream bug.
    // Propagating that as a clean error beats declining into a fused panic.
    let src = strided_kernel::StridedView::new(src_data, &rshape, &rsrc, src_offset)
        .map_err(crate::strided::error)?;
    let mut dst = strided_kernel::StridedViewMut::new(dst_data, &rshape, &rdst, dst_offset)
        .map_err(crate::strided::error)?;
    strided_kernel::copy_into_col_major(&mut dst, &src).map_err(crate::strided::error)?;
    Ok(true)
}

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
        // Assign-scatter (beta=0, alpha=1, positive strides) is a pure permuted
        // copy: route through the HPTT blocked transpose when enabled and the
        // layout is HPTT-eligible; otherwise fall through to the fused loop.
        if hptt_enabled()
            && beta.is_zero()
            && alpha.is_one()
            && src_strides.iter().all(|&s| s >= 0)
            && dst_strides.iter().all(|&s| s >= 0)
            && hptt_permuted_copy(
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
            )?
        {
            return Ok(());
        }
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
        // Pack is a pure permuted copy (alpha=1, no conjugate, positive
        // strides): route it through the HPTT blocked transpose when enabled
        // and HPTT-eligible; otherwise fall through to the fused loop.
        if hptt_enabled()
            && alpha.is_one()
            && !source_conjugate
            && src_strides.iter().all(|&s| s >= 0)
            && dst_strides.iter().all(|&s| s >= 0)
            && hptt_permuted_copy(
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
            )?
        {
            return Ok(());
        }
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

#[cfg(test)]
mod hptt_probe {
    use super::{fused_pair, hptt_permuted_copy};

    /// Inclusive max linear index a positive-stride layout reaches.
    fn span(shape: &[usize], strides: &[isize], offset: isize) -> usize {
        let mut hi = offset;
        for (&d, &s) in shape.iter().zip(strides) {
            hi += (d as isize - 1).max(0) * s;
        }
        (hi + 1) as usize
    }

    /// HPTT-path output must equal the fused-loop output for every layout HPTT
    /// accepts (the routing contract: routing to HPTT never changes results).
    fn assert_hptt_matches_fused(
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_off: isize,
        src_off: isize,
    ) {
        let src: Vec<f64> = (0..span(shape, src_strides, src_off))
            .map(|i| (i as f64) * 0.5 - 3.0)
            .collect();
        let dlen = span(shape, dst_strides, dst_off);
        let mut hptt = vec![0.0f64; dlen];
        let mut fused = vec![0.0f64; dlen];
        let handled = hptt_permuted_copy(
            &mut hptt,
            &src,
            shape,
            dst_strides,
            src_strides,
            dst_off,
            src_off,
        )
        .unwrap();
        assert!(
            handled,
            "expected HPTT to handle shape={shape:?} ds={dst_strides:?} ss={src_strides:?}"
        );
        fused_pair(
            &mut fused,
            &src,
            shape,
            dst_strides,
            src_strides,
            dst_off,
            src_off,
            |d, v| *d = v,
            |v: f64| v,
        );
        assert_eq!(
            hptt, fused,
            "HPTT != fused for shape={shape:?} ds={dst_strides:?} ss={src_strides:?} \
             doff={dst_off} soff={src_off}"
        );
    }

    #[test]
    fn hptt_matches_fused_across_layouts() {
        // (shape, dst_strides, src_strides, dst_off, src_off) — every case has a
        // genuine stride-1 axis on each side and positive strides.
        let cases: &[(&[usize], &[isize], &[isize], isize, isize)] = &[
            (&[4, 4], &[1, 4], &[4, 1], 0, 0), // 2D square transpose
            (&[3, 5], &[1, 3], &[5, 1], 0, 0), // 2D rectangular transpose
            (&[3, 3], &[1, 3], &[3, 1], 5, 7), // 2D with offsets
            (&[3, 4, 2], &[1, 3, 12], &[1, 10, 40], 0, 0), // 3D gapped sub-block, stride-1 axis0
            (&[3, 4, 2], &[1, 3, 12], &[4, 1, 16], 0, 0), // 3D, src stride-1 on axis1
            (&[2, 3, 2, 2], &[1, 2, 6, 12], &[24, 8, 1, 4], 0, 0), // 4D permuted
            (&[1, 4, 1, 4], &[1, 1, 4, 4], &[9, 1, 9, 8], 0, 0), // extent-1 mixed, reduces to [4,4]
            (&[6, 6], &[1, 6], &[6, 1], 0, 0), // larger transpose (hits blocking)
        ];
        for &(shape, ds, ss, doff, soff) in cases {
            assert_hptt_matches_fused(shape, ds, ss, doff, soff);
        }
    }

    #[test]
    fn declines_strided_slice_without_stride1_axis() {
        // Strided slice of a larger buffer: reduced strides [168,42] have no
        // stride-1 axis (its contiguous axis was a singleton). HPTT cannot do
        // this correctly, so the wrapper must decline (return false) and leave
        // dst untouched. This is the exact pattern that corrupted the energy
        // (-1.803 vs -1.772) before the guard existed.
        let src = vec![0.0f64; 4096];
        let shape = [4usize, 4];
        let mut dst = [7.0f64; 16];
        let handled =
            hptt_permuted_copy(&mut dst, &src, &shape, &[1, 4], &[168, 42], 0, 0).unwrap();
        assert!(!handled, "must decline the no-stride-1 strided slice");
        assert!(
            dst.iter().all(|&x| x == 7.0),
            "dst must be untouched when declined"
        );
    }

    #[test]
    fn declines_when_only_dst_has_stride1() {
        // dst has a stride-1 axis but src does not (both sides are required).
        let src = vec![1.0f64; 4096];
        let handled =
            hptt_permuted_copy(&mut [0.0f64; 16], &src, &[4, 4], &[1, 4], &[10, 40], 0, 0).unwrap();
        assert!(!handled, "must decline when src lacks a stride-1 axis");
    }

    #[test]
    fn all_extent1_copies_single_element() {
        let src = [42.0f64, 99.0];
        let mut dst = [0.0f64, 0.0];
        let handled = hptt_permuted_copy(&mut dst, &src, &[1, 1], &[1, 1], &[1, 1], 1, 1).unwrap();
        assert!(handled);
        assert_eq!(
            dst[1], src[1],
            "single-element copy must move src[off] to dst[off]"
        );
    }

    #[test]
    fn transposes_genuine_case_correctly() {
        // Genuine transpose against a hand-computed reference (not just fused):
        // src row-major [3,1], dst col-major [1,2]. dst[i+2j] == src[3i+j].
        let src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = [0.0f64; 6];
        let handled = hptt_permuted_copy(&mut dst, &src, &[2, 3], &[1, 2], &[3, 1], 0, 0).unwrap();
        assert!(handled, "genuine transpose must be handled by HPTT");
        let mut expect = [0.0f64; 6];
        for i in 0..2 {
            for j in 0..3 {
                expect[i + 2 * j] = src[3 * i + j];
            }
        }
        assert_eq!(dst, expect, "HPTT transpose disagrees with manual copy");
    }

    // Crossover micro-bench (ignored; run explicitly):
    //   cargo test --release -p tenet-operations -- --ignored --nocapture hptt_crossover
    // Prints fused vs HPTT per-call ns over square transposes of growing N so
    // the block size where HPTT's blocked micro-kernel overtakes the fused loop
    // (amortizing its per-call plan build) is visible. Confirms the insertion
    // point is correct and identifies when enabling HPTT pays off.
    #[test]
    #[ignore]
    fn hptt_crossover_bench() {
        use std::time::Instant;
        println!("\n  N     fused(ns)      hptt(ns)   speedup");
        for &n in &[4usize, 8, 16, 32, 64, 128, 256, 512] {
            let src: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
            let shape = [n, n];
            let ss = [n as isize, 1]; // src row-major (stride-1 axis1)
            let ds = [1, n as isize]; // dst col-major (stride-1 axis0) => transpose
            let mut dst = vec![0.0f64; n * n];
            let iters = (1usize << 24 >> (2 * (n as f64).log2() as usize)).max(50);
            // warm caches
            fused_pair(
                &mut dst,
                &src,
                &shape,
                &ds,
                &ss,
                0,
                0,
                |d, v| *d = v,
                |v: f64| v,
            );
            let _ = hptt_permuted_copy(&mut dst, &src, &shape, &ds, &ss, 0, 0);
            let t = Instant::now();
            for _ in 0..iters {
                fused_pair(
                    &mut dst,
                    &src,
                    &shape,
                    &ds,
                    &ss,
                    0,
                    0,
                    |d, v| *d = v,
                    |v: f64| v,
                );
            }
            let fused_ns = t.elapsed().as_nanos() as f64 / iters as f64;
            let t = Instant::now();
            for _ in 0..iters {
                let _ = hptt_permuted_copy(&mut dst, &src, &shape, &ds, &ss, 0, 0);
            }
            let hptt_ns = t.elapsed().as_nanos() as f64 / iters as f64;
            println!(
                "{n:4}  {fused_ns:11.1}  {hptt_ns:11.1}   {:.2}x {}",
                fused_ns / hptt_ns,
                if hptt_ns < fused_ns {
                    "<- HPTT wins"
                } else {
                    ""
                }
            );
        }
    }
}
