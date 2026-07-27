//! Matrix functions of fusion tensors, built from spectral factorizations,
//! coupled-sector linear solves, or a blockwise polynomial approximant at the
//! dense boundary.

use std::hash::Hash;

use tenet_core::MultiplicityFreeRigidSymbols;
use tenet_dense::{DenseExecutor, DenseView, DenseViewMut};
use tenet_tensors::{
    OperationError, TensorContractBackend, TensorContractFusionExecutionContext,
    TreeTransformBackend, TreeTransformRuleCacheKey,
};

use crate::compose::compose_bound_dyn;
use crate::factorize::{
    adjoint_bound_factor, eigh_full_dyn, inverse_by_sector_dyn, is_hermitian_endomorphism_dyn,
    map_square_sectors_dyn, scale_axis_by_spectrum, svd_compact_factors_dyn,
    typed_from_bound_factor, BoundDynFactor, BoundDynamicTensorRef, BoundTensorMap,
    BoundTensorMapRef, FactorScalar, SectorSpectrum, SvdFactorsDyn,
};

/// Matrix exponential of any endomorphism (TensorKit `exp!`, which checks only
/// `domain == codomain`). Hermitian input takes the spectral route
/// `exp(t) = V exp(D) V^H`; everything else takes blockwise scaling-and-squaring
/// Padé [13/13] (Higham 2005) around LAPACK `gebal('B')` balancing, the
/// algorithm behind the `LinearAlgebra.exp!` TensorKit calls.
pub fn exp<E, RuleKey, BT, BC, R, D, const N: usize>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundTensorMapRef<'_, R, D, N, N>,
) -> Result<BoundTensorMap<R, D, N, N>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let out = exp_dyn(dense, context, &input.dynamic())?;
    typed_from_bound_factor(out)
}

/// Dynamic-rank [`exp`].
pub fn exp_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    // TensorKit's `exp!` (`linalg.jl:420-428`) checks only that the map is an
    // endomorphism and then runs a general per-block exponential. Ask that
    // question here, at the one point both routes pass through, instead of
    // letting it fall to whichever helper notices first: the dispatch below
    // reaches `is_hermitian_endomorphism_dyn`, whose refusal names `eigh` —
    // not the function the caller called.
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "exp requires an endomorphism (codomain == domain)",
        });
    }
    // The Hermitian spectral route is kept for Hermitian input — it is exact,
    // it is what the published values of this function have always been, and it
    // costs one eigendecomposition instead of ~7 GEMMs and a solve — with
    // blockwise Padé behind it for everything else (issue #577). The predicate
    // is asked directly rather than inferred from a failed EIGH, so a backend
    // failure is never mistaken for non-hermiticity.
    if is_hermitian_endomorphism_dyn(input)? {
        return spectral_function_dyn(dense, context, input, &f64::exp);
    }
    exp_pade13_by_sector_dyn(dense, input)
}

/// Higham's `theta_13`: the largest `||A||_1` for which the [13/13] Padé
/// approximant of `exp(A)` has backward error below the double-precision unit
/// roundoff (Higham 2005, Table 2.3). Julia's `LinearAlgebra.exp!` rounds this
/// to `5.4`; the exact constant is used here because nothing downstream depends
/// on matching Julia's choice of squaring count, only on the accuracy it buys.
const PADE13_THETA: f64 = 5.371_920_351_148_152;

/// Padé [13/13] numerator/denominator coefficients `b_0 .. b_13`, the same
/// table Julia's `LinearAlgebra.exp!` carries. `exp(A) ≈ (V - U)^-1 (V + U)`
/// with `U = A (b13 A^6 + ... )` odd and `V` even in `A`.
const PADE13_B: [f64; 14] = [
    64_764_752_532_480_000.0,
    32_382_376_266_240_000.0,
    7_771_770_303_897_600.0,
    1_187_353_796_428_800.0,
    129_060_195_264_000.0,
    10_559_470_521_600.0,
    670_442_572_800.0,
    33_522_128_640.0,
    1_323_241_920.0,
    40_840_800.0,
    960_960.0,
    16_380.0,
    182.0,
    1.0,
];

/// Scratch for the Padé evaluation, sized once to the largest coupled sector.
///
/// Every matrix buffer is `max_c n_c²` and `balance` is `max_c n_c`; the sector
/// loop borrows them and allocates nothing. `image` and `square` are swapped
/// rather than copied during the squaring phase.
struct Pade13Workspace<D> {
    scaled: Vec<D>,
    power2: Vec<D>,
    power4: Vec<D>,
    power6: Vec<D>,
    odd: Vec<D>,
    even: Vec<D>,
    inner: Vec<D>,
    accumulator: Vec<D>,
    image: Vec<D>,
    square: Vec<D>,
    /// LAPACK `gebal`'s `scale` output for the sector in flight.
    balance: Vec<f64>,
}

impl<D: FactorScalar> Pade13Workspace<D> {
    fn new(order: usize) -> Result<Self, OperationError> {
        let elements = order
            .checked_mul(order)
            .ok_or(OperationError::ElementCountOverflow)?;
        let buffer = || vec![D::zero(); elements];
        Ok(Self {
            scaled: buffer(),
            power2: buffer(),
            power4: buffer(),
            power6: buffer(),
            odd: buffer(),
            even: buffer(),
            inner: buffer(),
            accumulator: buffer(),
            image: buffer(),
            square: buffer(),
            balance: vec![0.0; order],
        })
    }
}

/// Blockwise matrix exponential by scaling-and-squaring Padé [13/13].
///
/// The general-endomorphism arm of [`exp_dyn`], and TeNeT's port of what
/// TensorKit's `exp!` gets from `LinearAlgebra.exp!`: N. J. Higham, "The
/// Scaling and Squaring Method for the Matrix Exponential Revisited", SIAM J.
/// Matrix Anal. Appl. 26(4), 2005.
///
/// Per coupled sector the block is first balanced — LAPACK `gebal('B')`, run
/// where Julia's `exp!` runs it and undone where Julia undoes it, see
/// [`balance_in_place`] — and then, with
/// `s = max(0, ceil(log2(||A||_1 / theta_13)))` over the *balanced* norm and
/// `B = A / 2^s`:
///
/// ```text
/// U = B (B^6 (b13 B^6 + b11 B^4 + b9 B^2) + b7 B^6 + b5 B^4 + b3 B^2 + b1 I)
/// V =    B^6 (b12 B^6 + b10 B^4 + b8 B^2) + b6 B^6 + b4 B^4 + b2 B^2 + b0 I
/// exp(B) = (V - U)^-1 (V + U),   exp(A) = exp(B)^(2^s)
/// ```
///
/// Why a single [13/13] degree while Julia switches to degree 3/5/7/9 below
/// `||A||_1 = 2.1`: the low-degree branch is a speed optimization at equal
/// accuracy, and one branch is one thing to get wrong. Values therefore agree
/// with Julia to approximant error, not bit for bit.
///
/// Complexity: `O(Σ_c n_c³)` time — six GEMMs, one solve and `s` squarings per
/// sector — and `O(Σ_c n_c²)` result, with no cross-sector coupling and no
/// allocation inside the sector loop. The Padé workspace is `O(max_c n_c²)`,
/// sized once to the largest sector; on the canonical direct-region layout that
/// is the whole of the scratch, while the packed fallback in
/// [`map_square_sectors_dyn`] matricizes every sector up front and so adds
/// `O(Σ_c n_c²)` of its own.
///
/// # Errors
///
/// - [`OperationError::InvalidArgument`] for a nonfinite block, which has no
///   exponential and would otherwise reach the backend as a silent NaN, or for
///   a block whose balanced column 1-norm overflows to infinity even though
///   every entry is finite — the scaling count derived from it is not
///   representable;
/// - [`OperationError::Dense`] from the backend, including
///   [`tenet_dense::DenseError::Unsupported`] when the selected executor has no
///   dense solve. Nothing is published unless every sector succeeded.
fn exp_pade13_by_sector_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    map_square_sectors_dyn(
        input,
        Pade13Workspace::<D>::new,
        |workspace, source, order, output, output_leading| {
            exp_pade13_sector(dense, workspace, source, order, output, output_leading)
        },
    )
}

/// One coupled sector of [`exp_pade13_by_sector_dyn`].
fn exp_pade13_sector<E, D>(
    dense: &mut E,
    workspace: &mut Pade13Workspace<D>,
    source: &[D],
    order: usize,
    output: &mut [D],
    output_leading: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let Pade13Workspace {
        scaled,
        power2,
        power4,
        power6,
        odd,
        even,
        inner,
        accumulator,
        image,
        square,
        balance,
    } = workspace;
    let elements = order * order;

    // The finiteness gate, before anything reads the block: a nonfinite entry
    // would otherwise pass through balancing and the whole approximant as a NaN
    // and come back as an opaque backend failure or a silently wrong tensor.
    for &value in &source[..elements] {
        let value = value.widen_complex();
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(OperationError::InvalidArgument {
                message: "exp requires finite coupled-sector blocks",
            });
        }
    }

    // Balance first, exactly where Julia's `exp!` does it (stdlib v1.11
    // `dense.jl:684`), and undo it after the squaring phase: the 1-norm the
    // squaring count comes from is the *balanced* norm, so a block that is
    // merely badly scaled — `[0 1e16; 1e-16 0]`, whose exponential is a
    // hyperbolic rotation — costs no squarings instead of fifty-one, and does
    // not lose the digits fifty-one squarings of an approximant cost.
    scaled[..elements].copy_from_slice(&source[..elements]);
    let (ilo, ihi) = balance_in_place(&mut scaled[..elements], order, &mut balance[..order]);

    let mut norm1 = 0.0_f64;
    for column in 0..order {
        let mut column_sum = 0.0_f64;
        for row in 0..order {
            column_sum += scaled[row + order * column].widen_complex().norm();
        }
        norm1 = norm1.max(column_sum);
    }
    // Finite entries do not imply a finite norm: two entries near `f64::MAX` in
    // one column sum to infinity. The squaring count below is a saturating cast
    // — `ceil(log2(inf / theta_13)) as u32` is `u32::MAX`, whose `i32` reading
    // is -1 — so an infinite norm would scale the block *up* and then square it
    // ~4.3e9 times, turning a finite input into a hang. TeNeT already refuses
    // nonfinite blocks by policy; a norm that is not representable is the same
    // refusal one derivation later.
    if !norm1.is_finite() {
        return Err(OperationError::InvalidArgument {
            message: "exp requires coupled-sector blocks with a finite 1-norm",
        });
    }

    let squarings = if norm1 > PADE13_THETA {
        (norm1 / PADE13_THETA).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    // An exact power of two, so the scaling is exact and the squaring phase
    // undoes it exactly.
    let scale = D::from_real(2.0_f64.powi(-(squarings as i32)));
    for value in &mut scaled[..elements] {
        *value = *value * scale;
    }

    gemm(dense, power2, scaled, scaled, order)?;
    gemm(dense, power4, power2, power2, order)?;
    gemm(dense, power6, power2, power4, order)?;

    let b = &PADE13_B;
    combine(
        inner,
        &[
            (b[13], &power6[..]),
            (b[11], &power4[..]),
            (b[9], &power2[..]),
        ],
        order,
    );
    gemm(dense, accumulator, power6, inner, order)?;
    add_terms(
        accumulator,
        &[
            (b[7], &power6[..]),
            (b[5], &power4[..]),
            (b[3], &power2[..]),
        ],
        b[1],
        order,
    );
    gemm(dense, odd, scaled, accumulator, order)?;

    combine(
        inner,
        &[
            (b[12], &power6[..]),
            (b[10], &power4[..]),
            (b[8], &power2[..]),
        ],
        order,
    );
    gemm(dense, even, power6, inner, order)?;
    add_terms(
        even,
        &[
            (b[6], &power6[..]),
            (b[4], &power4[..]),
            (b[2], &power2[..]),
        ],
        b[0],
        order,
    );

    // `inner` = V + U (right-hand side), `accumulator` = V - U (system matrix).
    // `D` carries no `Sub`, so the negation rides a real scalar multiply.
    let minus_one = D::from_real(-1.0);
    for index in 0..elements {
        inner[index] = even[index] + odd[index];
        accumulator[index] = even[index] + minus_one * odd[index];
    }
    solve(dense, accumulator, inner, image, order)?;

    for _ in 0..squarings {
        gemm(dense, square, image, image, order)?;
        std::mem::swap(image, square);
    }

    unbalance_in_place(&mut image[..elements], order, &balance[..order], ilo, ihi);

    for column in 0..order {
        let source_start = order * column;
        let destination_start = output_leading * column;
        output[destination_start..destination_start + order]
            .copy_from_slice(&image[source_start..source_start + order]);
    }
    Ok(())
}

/// `|Re z| + |Im z|`, LAPACK's `CABS1` — the magnitude `gebal` sums and
/// compares, and plain `abs` on a real block.
fn abs1<D: FactorScalar>(value: D) -> f64 {
    let value = value.widen_complex();
    value.re.abs() + value.im.abs()
}

/// `IDAMAX`/`IZAMAX` followed by `ABS` of the element it picked, which is how
/// `gebal` forms `CA`/`RA` (`dgebal.f:343-346`, `zgebal.f:348-351`).
///
/// The two magnitudes are deliberately different and must stay so: `IZAMAX`
/// *selects* by `CABS1 = |Re| + |Im|`, but `ABS` of the selected element is its
/// **modulus**. Collapsing both onto `abs1` would inflate every complex `CA`
/// and `RA` by up to a factor of `sqrt(2)`, which is a factor of the radix in
/// the loop that consumes them.
///
/// `IDAMAX` compares strictly, so ties keep the earliest element; starting the
/// incumbent key at `-inf` reproduces that, including LAPACK's zero for an
/// all-zero span.
fn iamax_modulus<D: FactorScalar>(values: impl Iterator<Item = D>) -> f64 {
    let mut key = f64::NEG_INFINITY;
    let mut modulus = 0.0;
    for value in values {
        let candidate = abs1(value);
        if candidate > key {
            key = candidate;
            modulus = value.widen_complex().norm();
        }
    }
    modulus
}

/// `DNRM2`/`DZNRM2`: the Euclidean norm, by Blue's three-accumulator scaling
/// (`dnrm2.f90:139-197`, `dznrm2.f90` identically over the interleaved real and
/// imaginary parts), so that a span containing entries near the overflow or
/// underflow threshold still gets an exact-to-rounding answer instead of `inf`
/// or `0`.
///
/// This is the norm `gebal` measures its rows and columns with
/// (`dgebal.f:341-342`), not the `abs1` sum: they order pairs of vectors
/// differently, so the `abs1` sum reaches a different scale vector — on
/// `[0 4 0; 1 0 1; 1 1 0]`, `[1, 1/2, 1]` instead of LAPACK's `[2, 1, 1]`.
///
/// The accumulation is `f64` for every `D`, one step better than the reference
/// (which works in the component type), because the bounds that gate the
/// balancing loop already come from the component type via
/// [`FactorScalar::safe_minimum`]: it is the *factor* that has to be
/// representable in `D`, not the norms it was derived from.
fn nrm2<D: FactorScalar>(values: impl Iterator<Item = D>) -> f64 {
    // Blue's constants for `f64`: `radix^ceiling((minexponent - 1) / 2)` and
    // friends, `dnrm2.f90:103-110` evaluated at `wp = real64`.
    let tsml = 2.0_f64.powi(-511);
    let tbig = 2.0_f64.powi(486);
    let ssml = 2.0_f64.powi(537);
    let sbig = 2.0_f64.powi(-538);

    let mut notbig = true;
    let mut asml = 0.0_f64;
    let mut amed = 0.0_f64;
    let mut abig = 0.0_f64;
    for value in values {
        let value = value.widen_complex();
        for ax in [value.re.abs(), value.im.abs()] {
            if ax > tbig {
                abig += (ax * sbig) * (ax * sbig);
                notbig = false;
            } else if ax < tsml {
                if notbig {
                    asml += (ax * ssml) * (ax * ssml);
                }
            } else {
                amed += ax * ax;
            }
        }
    }

    let (scale, sum_of_squares) = if abig > 0.0 {
        if amed > 0.0 || amed.is_nan() {
            abig += (amed * sbig) * sbig;
        }
        (1.0 / sbig, abig)
    } else if asml > 0.0 {
        if amed > 0.0 || amed.is_nan() {
            let amed = amed.sqrt();
            let asml = asml.sqrt() / ssml;
            let (smaller, larger) = if asml > amed {
                (amed, asml)
            } else {
                (asml, amed)
            };
            (1.0, larger * larger * (1.0 + (smaller / larger).powi(2)))
        } else {
            (1.0 / ssml, asml)
        }
    } else {
        (1.0, amed)
    };
    scale * sum_of_squares.sqrt()
}

/// LAPACK `dgebal`/`zgebal` with `job = 'B'`, in place on a column-major
/// `order x order` block: the permutation that pushes already-isolated
/// eigenvalues out of the active window, then the radix-2 diagonal similarity
/// that equalizes row and column norms inside it.
///
/// This is the balancing Julia's `LinearAlgebra.exp!` runs before its Padé
/// evaluation and undoes after it (stdlib v1.11 `dense.jl:684` and `769-780`),
/// which is what TensorKit's `exp!` inherits per block, so TeNeT's general arm
/// runs it too. Both halves of `'B'` are ported because both halves are what
/// `exp!` asks for and what it undoes.
///
/// Returns the **0-based, inclusive** active window `(ilo, ihi)`. `scale` is
/// LAPACK's dual-purpose output: inside the window, the diagonal factor applied
/// to that index; outside it, the **1-based** index the position was exchanged
/// with. The 1-based encoding is LAPACK's and is kept because
/// [`unbalance_in_place`] reads both meanings out of the one array — including
/// the corner where a fully isolated block leaves `ilo = ihi = 0` and the lone
/// window entry is the self-exchange `1`, which the undo then reads as the
/// harmless scaling factor `1.0`, exactly as Julia does.
pub(crate) fn balance_in_place<D: FactorScalar>(
    matrix: &mut [D],
    order: usize,
    scale: &mut [f64],
) -> (usize, usize) {
    if order == 0 {
        return (0, 0);
    }
    scale.fill(1.0);
    let mut k = 0usize;
    let mut l = order - 1;

    // Rows that isolate an eigenvalue, pushed down; then columns that do,
    // pushed left. The exchanges are partial, as in LAPACK: a column swap
    // touches rows `0..=l` and a row swap columns `k..`, leaving the already
    // isolated borders alone.
    loop {
        let isolated = (0..=l)
            .rev()
            .find(|&j| (0..=l).all(|i| i == j || abs1(matrix[j + order * i]) == 0.0));
        let Some(j) = isolated else { break };
        scale[l] = (j + 1) as f64;
        exchange(matrix, order, j, l, l, k);
        if l == 0 {
            return (0, 0);
        }
        l -= 1;
    }
    loop {
        let isolated =
            (k..=l).find(|&j| (k..=l).all(|i| i == j || abs1(matrix[i + order * j]) == 0.0));
        let Some(j) = isolated else { break };
        scale[k] = (j + 1) as f64;
        exchange(matrix, order, j, k, l, k);
        k += 1;
    }

    // The iterative scaling, transcribed from `dgebal.f`: at each index, walk
    // the row and column norms towards each other in factors of the radix while
    // neither the scaled quantities nor the accumulated factor can overflow or
    // underflow, and keep the step only if it shrinks their sum by more than
    // `FACTOR`.
    const RADIX: f64 = 2.0;
    const FACTOR: f64 = 0.95;
    // `xLAMCH('S') / xLAMCH('P')` of the *component* type (`dgebal.f:330-333`,
    // `sgebal.f:330-333`), not of `f64` unconditionally: these bounds are what
    // stops the radix loop, so they decide whether the factor it produces is
    // representable in `D`.
    let sfmin1 = D::safe_minimum() / D::epsilon();
    let sfmax1 = 1.0 / sfmin1;
    let sfmin2 = sfmin1 * RADIX;
    let sfmax2 = 1.0 / sfmin2;
    let mut converged = false;
    while !converged {
        converged = true;
        for i in k..=l {
            // `dgebal.f:341-346`. The norms span the whole window including the
            // diagonal — `DNRM2(L-K+1, A(K,I), 1)` is contiguous — while `CA`
            // and `RA` reach outside it, down every row and along every column
            // the similarity will touch.
            let mut c = nrm2((k..=l).map(|row| matrix[row + order * i]));
            let mut r = nrm2((k..=l).map(|column| matrix[i + order * column]));
            let mut ca = iamax_modulus((0..=l).map(|row| matrix[row + order * i]));
            let mut ra = iamax_modulus((k..order).map(|column| matrix[i + order * column]));
            // Guard against a row or column norm that underflowed to zero.
            if c == 0.0 || r == 0.0 {
                continue;
            }
            // `dgebal.f:354-358` bails out on a NaN to avoid an infinite loop.
            // This signature has no error channel to bail through, so the index
            // is skipped instead — same effect on termination, since only a
            // scaling that fired clears `converged`.
            if (c + ca + r + ra).is_nan() {
                continue;
            }
            let mut g = r / RADIX;
            let mut f = 1.0_f64;
            let s = c + r;
            while c < g && f.max(c).max(ca) < sfmax2 && r.min(g).min(ra) > sfmin2 {
                f *= RADIX;
                c *= RADIX;
                ca *= RADIX;
                r /= RADIX;
                g /= RADIX;
                ra /= RADIX;
            }
            g = c / RADIX;
            while g >= r && r.max(ra) < sfmax2 && f.min(c).min(g).min(ca) > sfmin2 {
                f /= RADIX;
                c /= RADIX;
                g /= RADIX;
                ca /= RADIX;
                r *= RADIX;
                ra *= RADIX;
            }
            if c + r >= FACTOR * s {
                continue;
            }
            if f < 1.0 && scale[i] < 1.0 && f * scale[i] <= sfmin1 {
                continue;
            }
            if f > 1.0 && scale[i] > 1.0 && scale[i] >= sfmax1 / f {
                continue;
            }
            scale[i] *= f;
            converged = false;
            // `f` is a power of the radix, so both factors are exact and the
            // similarity introduces no rounding of its own.
            let row_factor = D::from_real(1.0 / f);
            let column_factor = D::from_real(f);
            for column in k..order {
                matrix[i + order * column] = matrix[i + order * column] * row_factor;
            }
            for row in 0..=l {
                matrix[row + order * i] = matrix[row + order * i] * column_factor;
            }
        }
    }
    (k, l)
}

/// LAPACK's partial row/column exchange of `j` and `m`: columns over rows
/// `0..=rows`, rows over columns `first_column..`.
fn exchange<D: Copy>(
    matrix: &mut [D],
    order: usize,
    j: usize,
    m: usize,
    rows: usize,
    first_column: usize,
) {
    if j == m {
        return;
    }
    for row in 0..=rows {
        matrix.swap(row + order * j, row + order * m);
    }
    for column in first_column..order {
        matrix.swap(j + order * column, m + order * column);
    }
}

/// Undoes [`balance_in_place`] on an exponentiated block, in the order Julia's
/// `exp!` does it (stdlib v1.11 `dense.jl:769-780`): the diagonal similarity
/// inside the window first, then the exchanges below it in reverse order and
/// the ones above it in forward order — each the reverse of the order they were
/// made in.
fn unbalance_in_place<D: FactorScalar>(
    matrix: &mut [D],
    order: usize,
    scale: &[f64],
    ilo: usize,
    ihi: usize,
) {
    for j in ilo..=ihi {
        // Powers of the radix again, so `1 / s` is exact.
        let row_factor = D::from_real(scale[j]);
        let column_factor = D::from_real(1.0 / scale[j]);
        for i in 0..order {
            matrix[j + order * i] = matrix[j + order * i] * row_factor;
        }
        for i in 0..order {
            matrix[i + order * j] = matrix[i + order * j] * column_factor;
        }
    }
    for (j, &partner) in scale[..ilo].iter().enumerate().rev() {
        row_column_swap(matrix, order, j, partner as usize - 1);
    }
    for (j, &partner) in scale.iter().enumerate().skip(ihi + 1) {
        row_column_swap(matrix, order, j, partner as usize - 1);
    }
}

/// Julia's `rcswap!`: swap rows `i` and `j` and columns `i` and `j`.
fn row_column_swap<D: Copy>(matrix: &mut [D], order: usize, i: usize, j: usize) {
    if i == j {
        return;
    }
    for k in 0..order {
        matrix.swap(k + order * i, k + order * j);
    }
    for k in 0..order {
        matrix.swap(i + order * k, j + order * k);
    }
}

/// `destination = Σ_k coefficient_k * term_k` over `order x order` blocks.
fn combine<D: FactorScalar>(destination: &mut [D], terms: &[(f64, &[D])], order: usize) {
    let elements = order * order;
    for index in 0..elements {
        let mut value = D::zero();
        for (coefficient, term) in terms {
            value = value + D::from_real(*coefficient) * term[index];
        }
        destination[index] = value;
    }
}

/// `destination += Σ_k coefficient_k * term_k + diagonal * I`.
fn add_terms<D: FactorScalar>(
    destination: &mut [D],
    terms: &[(f64, &[D])],
    diagonal: f64,
    order: usize,
) {
    let elements = order * order;
    for index in 0..elements {
        let mut value = destination[index];
        for (coefficient, term) in terms {
            value = value + D::from_real(*coefficient) * term[index];
        }
        destination[index] = value;
    }
    let diagonal = D::from_real(diagonal);
    for index in 0..order {
        destination[index + order * index] = destination[index + order * index] + diagonal;
    }
}

/// `destination = lhs * rhs` on column-major `order x order` blocks.
fn gemm<E, D>(
    dense: &mut E,
    destination: &mut [D],
    lhs: &[D],
    rhs: &[D],
    order: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let elements = order * order;
    let shape = [order, order];
    let strides = [1usize, order];
    let lhs =
        DenseView::new(&lhs[..elements], &shape, &strides, 0).map_err(OperationError::Dense)?;
    let rhs =
        DenseView::new(&rhs[..elements], &shape, &strides, 0).map_err(OperationError::Dense)?;
    let destination = DenseViewMut::new(&mut destination[..elements], &shape, &strides, 0)
        .map_err(OperationError::Dense)?;
    dense
        .matmul_into(
            D::dense_write(destination),
            D::dense_read(lhs),
            D::dense_read(rhs),
        )
        .map_err(OperationError::Dense)
}

/// Solves `matrix * solution = rhs` on column-major `order x order` blocks.
fn solve<E, D>(
    dense: &mut E,
    matrix: &[D],
    rhs: &[D],
    solution: &mut [D],
    order: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let elements = order * order;
    let shape = [order, order];
    let strides = [1usize, order];
    let matrix =
        DenseView::new(&matrix[..elements], &shape, &strides, 0).map_err(OperationError::Dense)?;
    let rhs =
        DenseView::new(&rhs[..elements], &shape, &strides, 0).map_err(OperationError::Dense)?;
    let solution = DenseViewMut::new(&mut solution[..elements], &shape, &strides, 0)
        .map_err(OperationError::Dense)?;
    dense
        .solve_into(
            D::dense_read(matrix),
            D::dense_read(rhs),
            D::dense_write(solution),
        )
        .map_err(OperationError::Dense)
}

/// Applies a scalar function to a Hermitian endomorphism through its
/// eigendecomposition: `f(t) = V f(D) V^H`.
///
/// Visible to the crate so that the dispatch test can build `exp`'s reference
/// on whichever backend is running it, instead of pinning constants that are a
/// few ULP different on another platform's LAPACK.
pub(crate) fn spectral_function_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
    function: &dyn Fn(f64) -> f64,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let (v, eigenvalues) = eigh_full_dyn(dense, input)?.into_parts();
    let mapped: Vec<SectorSpectrum> = eigenvalues
        .iter()
        .map(|entry| SectorSpectrum {
            sector: entry.sector,
            values: entry.values.iter().map(|&value| function(value)).collect(),
        })
        .collect();
    // f(t) = V f(D) V^H. Fold the diagonal f(D) into a column scaling of V
    // (bond = trailing axis) rather than materializing it and running an extra
    // GEMM (issue #46); V^H is built before V is scaled.
    let vh = adjoint_bound_factor(&v)?;
    let mut vd = v;
    let (space, data) = vd.raw_space_and_data_mut();
    scale_axis_by_spectrum(space, data, None, &mapped)?;
    compose_bound_dyn(context, &vd, &vh)
}

/// Moore-Penrose pseudo-inverse via the compact SVD with an
/// `rcond * sigma_max` cutoff: `t^+ = V S^+ U^H`.
pub fn pinv<E, RuleKey, BT, BC, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
    rcond: f64,
) -> Result<BoundTensorMap<R, D, NIN, NOUT>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let out = pinv_dyn(dense, context, &input.dynamic(), rcond)?;
    typed_from_bound_factor(out)
}

/// Dynamic-rank [`pinv`].
pub fn pinv_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
    rcond: f64,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    if !rcond.is_finite() || rcond < 0.0 {
        return Err(OperationError::InvalidArgument {
            message: "pinv rcond must be finite and non-negative",
        });
    }
    // Only the factors and the spectrum are needed — S^+ is folded into a
    // scaling below — so skip materializing the dense diagonal S.
    let factors = svd_compact_factors_dyn(dense, input)?;
    pinv_from_factors(context, factors, rcond)
}

/// Applies the public `pinv` cutoff to compact SVD factors before the
/// factor-recomposition step shared with `inv_dyn`.
fn pinv_from_factors<RuleKey, BT, BC, R, D>(
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    factors: SvdFactorsDyn<R, D>,
    rcond: f64,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let (u, vh, singular_values) = factors;
    let sigma_max = singular_values
        .iter()
        .flat_map(|entry| entry.values.iter().copied())
        .fold(0.0_f64, f64::max);
    let cutoff = rcond * sigma_max;
    let inverted: Vec<SectorSpectrum> = singular_values
        .iter()
        .map(|entry| SectorSpectrum {
            sector: entry.sector,
            values: entry
                .values
                .iter()
                .map(|&sigma| if sigma > cutoff { 1.0 / sigma } else { 0.0 })
                .collect(),
        })
        .collect();
    inverse_from_factors(context, u, vh, &inverted)
}

fn inverse_from_factors<RuleKey, BT, BC, R, D>(
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    u: BoundDynFactor<R, D>,
    vh: BoundDynFactor<R, D>,
    inverted: &[SectorSpectrum],
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    // t^+ = V S^+ U^H. Fold S^+ into a column scaling of V (bond = trailing
    // axis) instead of building the dense diagonal and running an extra GEMM
    // (issue #46).
    let mut v = adjoint_bound_factor(&vh)?;
    let uh = adjoint_bound_factor(&u)?;
    let v_space = v.space().space().clone();
    scale_axis_by_spectrum(&v_space, v.data_mut(), None, inverted)?;
    compose_bound_dyn(context, &v, &uh)
}

/// True inverse of a nonsingular map between isomorphic spaces.
///
/// The context parameter is retained for source compatibility. Inverse itself
/// is context-free and performs one dense solve per nonempty coupled sector.
pub fn inv<E, RuleKey, BT, BC, R, D, const N: usize>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundTensorMapRef<'_, R, D, N, N>,
) -> Result<BoundTensorMap<R, D, N, N>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let out = inv_dyn(dense, context, &input.dynamic())?;
    typed_from_bound_factor(out)
}

/// Dynamic-rank [`inv`].
pub fn inv_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    _context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    inv_direct_dyn(dense, input)
}

/// Context-free dynamic-rank inverse used by the user layer.
#[doc(hidden)]
pub fn inv_direct_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    // Why not reuse pinv's SVD/recomposition path: ordinary inverse has no
    // truncation policy, so factor tensors and a recoupling contraction are
    // avoidable work.
    inverse_by_sector_dyn(dense, input)
}
