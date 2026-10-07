use super::*;

/// Uploads a rank-0 operand in the payload's **real** lane.
///
/// The lane is not a symmetry: Tenferro rejects an `F64` rank-0 operand
/// against an `F32` or `C32` tensor as a dtype mismatch, and rejects a
/// *payload*-typed complex operand as a shape mismatch, so "complex tensor
/// with a real rank-0 scalar" is the one production shape (probe
/// `cuda-single-precision-probe-2026-09-20.md`, finding 3).
fn upload_scalar<R: CudaRealScalar>(
    ctx: &CudaDenseContext,
    value: f64,
) -> Result<Tensor, DenseError> {
    let host = R::into_tensor(vec![], vec![R::narrow(value)])
        .map_err(|err| cuda_error("cuda_hermitian", err))?;
    let tensor = upload_tensor(ctx.backend.runtime(), &host)
        .map_err(|err| cuda_error("cuda_hermitian", err))?;
    record_h2d(std::mem::size_of::<R>());
    Ok(tensor)
}
/// The rank-0 lane operand through which [`scale_by_power_of_two`] multiplies
/// a payload by the exact power of two `normalizer`.
fn power_of_two_operand<D: CudaScalar>(
    ctx: &CudaDenseContext,
    normalizer: f64,
) -> Result<Tensor, DenseError> {
    let operand = if D::IS_COMPLEX {
        normalizer
    } else {
        // Exact: the reciprocal of a power of two inside the lane's range.
        normalizer.recip()
    };
    upload_scalar::<D::Real>(ctx, operand)
}
/// Multiplies a payload tensor by the power of two that `operand` encodes.
fn scale_by_power_of_two<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    op: &'static str,
    tensor: &Tensor,
    operand: &Tensor,
) -> Result<Tensor, DenseError> {
    if D::IS_COMPLEX {
        ctx.backend.mul(tensor, operand)
    } else {
        // Why not `mul`: Tenferro 0.7.1's real `mul` does not broadcast a
        // rank-0 operand, while its real `div` does and, unlike the
        // complex-by-real one, never squares the divisor.
        ctx.backend.div(tensor, operand)
    }
    .map_err(|err| cuda_error(op, err))
}
/// Real magnitudes of a device tensor, so the real-only sum-of-squares
/// reduction sees `|z|^2` for a complex payload. `abs` leaves an `f64` tensor
/// numerically unchanged under that reduction, so the extra pass is skipped
/// there on the dtype invariant.
fn magnitudes_for_sum_squares<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    op: &'static str,
    tensor: &Tensor,
) -> Result<Option<Tensor>, DenseError> {
    if !D::IS_COMPLEX {
        return Ok(None);
    }
    ctx.backend
        .abs(tensor)
        .map(Some)
        .map_err(|err| cuda_error(op, err))
}
/// Tests packed square CUDA regions `(offset, n)` of `src` with the host EIGH
/// rule `||(A - A^H)/2||_F <= relative_tolerance * ||A||_F`, returning one
/// decision per region. The residual uses the *conjugate* transpose, so a
/// complex-symmetric non-Hermitian block is rejected.
///
/// `relative_tolerance` is the caller's resolved eigh admission tolerance
/// (TeNeT's `HermitianTol` at the payload's real-lane epsilon), the same one
/// the host twin `normwise_hermitian` applies.
/// The caller must supply a finite, non-negative value.
///
/// A complex entry whose modulus overflows the lane is rejected here (the
/// `hypot`-based `abs` gives infinity) but admitted by the component-scaled
/// host twin. That disagreement is on the safe side and predates the power
/// of two normalizer; it is why the device tests' scale window stops at
/// `2^(MAX_EXP - 4)`.
///
/// Each region's rule needs three dependent scalar stages: its maximum picks
/// the input normalizer, the residual maximum picks the residual normalizer,
/// and the residual sum of squares decides. Every stage runs over all
/// still-undecided regions before its scalars are gathered with one
/// `concatenate` and downloaded together, so a call makes at most three
/// downloads whatever the region count. Why not per region: each download
/// blocks the host (a D2H plus a CubeCL flush), which made admission the
/// largest TeNeT-owned cost of `eigh` (#1483). The price is that the
/// materialized input, then residual, of every undecided region is live
/// across a stage boundary: one extra copy of the regions' payload, the same
/// order as the eigenvector factor `eigh` allocates anyway.
///
/// Only scalar norm metadata is downloaded; no region is copied to the host.
#[doc(hidden)]
pub fn cuda_hermitian_regions<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    regions: &[(usize, usize)],
    relative_tolerance: f64,
) -> Result<Vec<bool>, DenseError> {
    const OP: &str = "cuda_hermitian";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    let max_exp = <D::Real as CudaRealScalar>::MAX_EXP;
    let mut accepted = vec![true; regions.len()];

    // Stage 1: max |A|.
    let mut inputs = Vec::new();
    let mut maxima = Vec::new();
    for (index, &(offset, n)) in regions.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let normal = contiguous_square::<D>(ctx, src, n, [1, n], offset)?;
        let input_abs = ctx
            .backend
            .abs(&normal)
            .map_err(|err| cuda_error(OP, err))?;
        maxima.push(
            ctx.backend
                .reduce_max(&input_abs, &[0, 1])
                .map_err(|err| cuda_error(OP, err))?,
        );
        inputs.push((index, normal));
    }
    let input_scales = download_gathered::<D::Real>(ctx, &maxima, OP)?;
    drop(maxima);

    // Stage 2: sum |A/s|^2 and max |(A - A^H)/s|.
    let mut residuals = Vec::new();
    let mut input_sums = Vec::new();
    let mut residual_maxima = Vec::new();
    for ((index, normal), input_scale) in inputs.into_iter().zip(input_scales) {
        // Pinned Tenferro's CUDA reduce_max propagates NaN. Keep this check
        // before the zero fast path so an otherwise-zero matrix containing NaN
        // is rejected.
        if !input_scale.is_finite() {
            accepted[index] = false;
            continue;
        }
        if input_scale == 0.0 {
            continue;
        }
        let (offset, n) = regions[index];
        let transpose = contiguous_square::<D>(ctx, src, n, [n, 1], offset)?;
        // Conjugation is a backend op on the transposed copy, never a TeNeT loop.
        let transpose = if D::IS_COMPLEX {
            ctx.backend
                .conj(&transpose)
                .map_err(|err| cuda_error(OP, err))?
        } else {
            transpose
        };
        let normalizer =
            power_of_two_operand::<D>(ctx, power_of_two_normalizer(input_scale, max_exp))?;
        let normal_scaled = scale_by_power_of_two::<D>(ctx, OP, &normal, &normalizer)?;
        drop(normal);
        let transpose_scaled = scale_by_power_of_two::<D>(ctx, OP, &transpose, &normalizer)?;
        input_sums.push(sum_of_squares::<D>(ctx, OP, &normal_scaled)?);
        let residual = ctx
            .backend
            .sub(&normal_scaled, &transpose_scaled)
            .map_err(|err| cuda_error(OP, err))?;
        let residual_abs = ctx
            .backend
            .abs(&residual)
            .map_err(|err| cuda_error(OP, err))?;
        residual_maxima.push(
            ctx.backend
                .reduce_max(&residual_abs, &[0, 1])
                .map_err(|err| cuda_error(OP, err))?,
        );
        residuals.push((index, residual));
    }
    let count = residuals.len();
    input_sums.append(&mut residual_maxima);
    let stage2 = download_gathered::<D::Real>(ctx, &input_sums, OP)?;
    drop(input_sums);
    let (input_ss, residual_scales) = stage2.split_at(count);

    // Stage 3: sum |R/r|^2.
    let mut undecided = Vec::new();
    let mut residual_sums = Vec::new();
    for (((index, residual), &input_ss), &residual_scale) in
        residuals.into_iter().zip(input_ss).zip(residual_scales)
    {
        if !residual_scale.is_finite() {
            accepted[index] = false;
            continue;
        }
        if residual_scale == 0.0 {
            accepted[index] = input_ss.is_finite() && input_ss >= 0.0;
            continue;
        }
        let residual_normalizer = power_of_two_normalizer(residual_scale, max_exp);
        let residual_normalizer_tensor = power_of_two_operand::<D>(ctx, residual_normalizer)?;
        let residual_normalized =
            scale_by_power_of_two::<D>(ctx, OP, &residual, &residual_normalizer_tensor)?;
        residual_sums.push(sum_of_squares::<D>(ctx, OP, &residual_normalized)?);
        undecided.push((index, input_ss, residual_normalizer));
    }
    let residual_ss = download_gathered::<D::Real>(ctx, &residual_sums, OP)?;
    for ((index, input_ss, residual_normalizer), residual_ss) in
        undecided.into_iter().zip(residual_ss)
    {
        accepted[index] = scaled_hermitian_residual_accepts(
            input_ss,
            // Exact: the reciprocal of a normal power of two.
            residual_normalizer.recip(),
            residual_ss,
            relative_tolerance,
        );
    }
    Ok(accepted)
}
/// [`cuda_hermitian_regions`] over `members` stacked copies of the same
/// regions, member `b` of region `(offset, n)` at `offset + b *
/// member_stride`. Returns one decision per (member, region), member-major:
/// `decisions[b * regions.len() + r]`.
/// The caller must supply a finite, non-negative `relative_tolerance`.
///
/// Each (member, region) pair is decided by exactly the rule and the three
/// scalar stages of [`cuda_hermitian_regions`], with its own power-of-two
/// normalizers: members of one batch may differ by many orders of magnitude,
/// and a normalizer shared across members would change the acceptance of the
/// small ones. The member axis is a trailing mode of every device tensor, so
/// every stage still ends in one download (at most three per call) and the
/// submissions per region do not depend on `members`. The normalizers of a
/// region are one `[members]` upload per stage, broadcast along the matrix
/// modes.
///
/// Real payloads only: a complex payload returns [`DenseError::Unsupported`].
/// Why: Tenferro 0.7.1 scales a complex tensor by a real operand only when the
/// operand is rank 0, so a per-member real normalizer has no complex form yet.
#[doc(hidden)]
pub fn cuda_hermitian_regions_batched<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    regions: &[(usize, usize)],
    members: usize,
    member_stride: usize,
    relative_tolerance: f64,
) -> Result<Vec<bool>, DenseError> {
    const OP: &str = "cuda_hermitian_batched";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    if D::IS_COMPLEX {
        return Err(DenseError::Unsupported {
            op: OP,
            message: "per-member normalizers of a complex payload".into(),
        });
    }
    let max_exp = <D::Real as CudaRealScalar>::MAX_EXP;
    let count = regions.len();
    let mut accepted = vec![true; count * members];
    let slot = |member: usize, region: usize| member * count + region;

    // Stage 1: max |A| per member.
    let mut inputs = Vec::new();
    let mut maxima = Vec::new();
    for (index, &(offset, n)) in regions.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let normal = contiguous_stack::<D>(ctx, src, n, [1, n], offset, members, member_stride)?;
        let input_abs = ctx
            .backend
            .abs(&normal)
            .map_err(|err| cuda_error(OP, err))?;
        maxima.push(
            ctx.backend
                .reduce_max(&input_abs, &[0, 1])
                .map_err(|err| cuda_error(OP, err))?,
        );
        inputs.push((index, normal));
    }
    let input_scales = download_concatenated::<D::Real>(ctx, &maxima, members, OP)?;
    drop(maxima);

    // Stage 2: sum |A/s|^2 and max |(A - A^H)/s| per member.
    let mut residuals = Vec::new();
    let mut input_sums = Vec::new();
    let mut residual_maxima = Vec::new();
    for ((index, normal), scales) in inputs.into_iter().zip(input_scales.chunks(members)) {
        let mut undecided = false;
        let divisors: Vec<f64> = scales
            .iter()
            .enumerate()
            .map(|(member, &scale)| {
                // As the single-region rule: NaN rejects before the zero
                // fast path, and a decided member divides by one.
                if !scale.is_finite() {
                    accepted[slot(member, index)] = false;
                    1.0
                } else if scale == 0.0 {
                    1.0
                } else {
                    undecided = true;
                    // Exact: the reciprocal of a power of two inside the lane.
                    power_of_two_normalizer(scale, max_exp).recip()
                }
            })
            .collect();
        if !undecided {
            continue;
        }
        let (offset, n) = regions[index];
        let transpose = contiguous_stack::<D>(ctx, src, n, [n, 1], offset, members, member_stride)?;
        let divisor = broadcast_member_divisors::<D>(ctx, &divisors, n, OP)?;
        let normal_scaled = ctx
            .backend
            .div(&normal, &divisor)
            .map_err(|err| cuda_error(OP, err))?;
        drop(normal);
        let transpose_scaled = ctx
            .backend
            .div(&transpose, &divisor)
            .map_err(|err| cuda_error(OP, err))?;
        input_sums.push(sum_of_squares::<D>(ctx, OP, &normal_scaled)?);
        let residual = ctx
            .backend
            .sub(&normal_scaled, &transpose_scaled)
            .map_err(|err| cuda_error(OP, err))?;
        let residual_abs = ctx
            .backend
            .abs(&residual)
            .map_err(|err| cuda_error(OP, err))?;
        residual_maxima.push(
            ctx.backend
                .reduce_max(&residual_abs, &[0, 1])
                .map_err(|err| cuda_error(OP, err))?,
        );
        residuals.push((index, residual, scales.to_vec()));
    }
    let stage2_regions = residuals.len();
    input_sums.append(&mut residual_maxima);
    let stage2 = download_concatenated::<D::Real>(ctx, &input_sums, members, OP)?;
    drop(input_sums);
    let (input_ss, residual_scales) = stage2.split_at(stage2_regions * members);

    // Stage 3: sum |R/r|^2 per member.
    let mut undecided = Vec::new();
    let mut residual_sums = Vec::new();
    for (((index, residual, scales), input_ss), residual_scales) in residuals
        .into_iter()
        .zip(input_ss.chunks(members))
        .zip(residual_scales.chunks(members))
    {
        let mut pending = Vec::new();
        let divisors: Vec<f64> = (0..members)
            .map(|member| {
                let scale = scales[member];
                let residual_scale = residual_scales[member];
                if !scale.is_finite() || scale == 0.0 {
                    1.0
                } else if !residual_scale.is_finite() {
                    accepted[slot(member, index)] = false;
                    1.0
                } else if residual_scale == 0.0 {
                    let input_ss = input_ss[member];
                    accepted[slot(member, index)] = input_ss.is_finite() && input_ss >= 0.0;
                    1.0
                } else {
                    let normalizer = power_of_two_normalizer(residual_scale, max_exp);
                    pending.push((member, input_ss[member], normalizer));
                    normalizer.recip()
                }
            })
            .collect();
        if pending.is_empty() {
            continue;
        }
        let (_, n) = regions[index];
        let divisor = broadcast_member_divisors::<D>(ctx, &divisors, n, OP)?;
        let residual_normalized = ctx
            .backend
            .div(&residual, &divisor)
            .map_err(|err| cuda_error(OP, err))?;
        residual_sums.push(sum_of_squares::<D>(ctx, OP, &residual_normalized)?);
        undecided.push((index, pending));
    }
    let residual_ss = download_concatenated::<D::Real>(ctx, &residual_sums, members, OP)?;
    for ((index, pending), residual_ss) in undecided.into_iter().zip(residual_ss.chunks(members)) {
        for (member, input_ss, residual_normalizer) in pending {
            accepted[slot(member, index)] = scaled_hermitian_residual_accepts(
                input_ss,
                // Exact: the reciprocal of a normal power of two.
                residual_normalizer.recip(),
                residual_ss[member],
                relative_tolerance,
            );
        }
    }
    Ok(accepted)
}
/// Materializes `members` stacked `n x n` regions (element strides `strides`
/// within a member, `member_stride` between members) as one compact
/// `[n, n, members]` device tensor.
fn contiguous_stack<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    n: usize,
    strides: [usize; 2],
    offset: usize,
    members: usize,
    member_stride: usize,
) -> Result<Tensor, DenseError> {
    const OP: &str = "cuda_hermitian_batched";
    let region = CudaRegion::new(
        vec![n, n, members],
        vec![strides[0], strides[1], member_stride],
        offset,
    )?;
    validate_region(&region, src.len)?;
    let view = src.region_view_nd::<D>(
        region.dims(),
        &isize_strides(region.strides())?,
        region.offset_isize()?,
    )?;
    ctx.backend
        .to_contiguous_read(TensorRead::from_view(view))
        .map_err(|err| cuda_error(OP, err))
}
/// Uploads one real divisor per member and broadcasts it over the matrix
/// modes of an `[n, n, members]` payload, so one elementwise `div` scales
/// every member by its own power of two.
///
/// Why broadcast rather than a rank-0 operand per member: Tenferro 0.7.1's
/// elementwise ops take equal shapes or a rank-0 operand, so a per-member
/// divisor needs the full shape; the broadcast is one device kernel, and the
/// upload is `members` real values.
fn broadcast_member_divisors<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    divisors: &[f64],
    n: usize,
    op: &'static str,
) -> Result<Tensor, DenseError> {
    let values: Vec<D::Real> = divisors
        .iter()
        .map(|&value| <D::Real as CudaRealScalar>::narrow(value))
        .collect();
    let host =
        D::Real::into_tensor(vec![divisors.len()], values).map_err(|err| cuda_error(op, err))?;
    let tensor = upload_tensor(ctx.backend.runtime(), &host).map_err(|err| cuda_error(op, err))?;
    record_h2d(divisors.len() * std::mem::size_of::<D::Real>());
    ctx.backend
        .broadcast_in_dim(&tensor, &[n, n, divisors.len()], &[2])
        .map_err(|err| cuda_error(op, err))
}
/// Downloads rank-1 real reductions of `members` values each with one
/// transfer (concatenated on device first). No parts means no transfer.
fn download_concatenated<R: CudaRealScalar>(
    ctx: &mut CudaDenseContext,
    parts: &[Tensor],
    members: usize,
    op: &'static str,
) -> Result<Vec<f64>, DenseError> {
    if parts.is_empty() {
        return Ok(Vec::new());
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    let gathered = ctx
        .backend
        .concatenate(&refs, 0)
        .map_err(|err| cuda_error(op, err))?;
    let values = download_values::<R>(ctx, &gathered)?;
    if values.len() != parts.len() * members {
        return Err(cuda_error(
            op,
            format!(
                "device reductions returned {} values; expected {}",
                values.len(),
                parts.len() * members
            ),
        ));
    }
    Ok(values)
}
/// Materializes the `n x n` region at `offset` with element strides
/// `strides` as a compact `[n, n, 1]` device tensor.
///
/// Why the trailing unit axis: a reduction over axes `[0, 1]` then yields a
/// rank-1 `[1]` tensor, which Tenferro's `concatenate` accepts, whereas the
/// rank-0 result of a rank-2 input would need a reshape copy per scalar.
fn contiguous_square<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    n: usize,
    strides: [usize; 2],
    offset: usize,
) -> Result<Tensor, DenseError> {
    const OP: &str = "cuda_hermitian";
    src.check_matrix_bound([n, n], strides, offset)?;
    let to_isize =
        |value: usize| isize::try_from(value).map_err(|_| cuda_error(OP, "index exceeds isize"));
    let view = src.region_view_nd::<D>(
        &[n, n, 1],
        &[to_isize(strides[0])?, to_isize(strides[1])?, 1],
        to_isize(offset)?,
    )?;
    ctx.backend
        .to_contiguous_read(TensorRead::from_view(view))
        .map_err(|err| cuda_error(OP, err))
}
/// `sum |x|^2` over axes `[0, 1]` of a `[n, n, 1]` payload tensor, in its
/// real lane.
fn sum_of_squares<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    op: &'static str,
    tensor: &Tensor,
) -> Result<Tensor, DenseError> {
    let magnitudes = magnitudes_for_sum_squares::<D>(ctx, op, tensor)?;
    ctx.backend
        .reduce_sum_squares_read(
            TensorRead::from_tensor(magnitudes.as_ref().unwrap_or(tensor)),
            &[0, 1],
        )
        .map_err(|err| cuda_error(op, err))
}
/// Downloads rank-1 real reductions of lane `R` with one transfer: they are
/// concatenated on device first. No parts means no transfer.
fn download_gathered<R: CudaRealScalar>(
    ctx: &mut CudaDenseContext,
    parts: &[Tensor],
    op: &'static str,
) -> Result<Vec<f64>, DenseError> {
    if parts.is_empty() {
        return Ok(Vec::new());
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    let gathered = ctx
        .backend
        .concatenate(&refs, 0)
        .map_err(|err| cuda_error(op, err))?;
    let values = download_values::<R>(ctx, &gathered)?;
    if values.len() != parts.len() {
        return Err(cuda_error(
            op,
            format!(
                "device reductions returned {} values; expected {}",
                values.len(),
                parts.len()
            ),
        ));
    }
    Ok(values)
}
