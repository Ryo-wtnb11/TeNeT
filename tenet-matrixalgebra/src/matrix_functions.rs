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

/// Matrix exponential of a Hermitian endomorphism: `exp(t) = V exp(D) V^H`.
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
/// Every buffer is `max_c n_c²`; the sector loop borrows them and allocates
/// nothing. `image` and `square` are swapped rather than copied during the
/// squaring phase.
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
/// Per coupled sector, with `s = max(0, ceil(log2(||A||_1 / theta_13)))` and
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
/// sector — `O(Σ_c n_c²)` result and `O(max_c n_c²)` scratch, with no
/// cross-sector coupling and no allocation inside the loop.
///
/// # Errors
///
/// - [`OperationError::InvalidArgument`] for a nonfinite block, which has no
///   exponential and would otherwise reach the backend as a silent NaN, or for
///   a block whose column 1-norm overflows to infinity even though every entry
///   is finite — the scaling count derived from it is not representable;
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
    } = workspace;
    let elements = order * order;

    // One pass for both the finiteness gate and the matrix 1-norm: a nonfinite
    // entry would otherwise pass through the whole approximant as a NaN and
    // come back as an opaque backend failure or a silently wrong tensor.
    let mut norm1 = 0.0_f64;
    for column in 0..order {
        let mut column_sum = 0.0_f64;
        for row in 0..order {
            let value = source[row + order * column].widen_complex();
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(OperationError::InvalidArgument {
                    message: "exp requires finite coupled-sector blocks",
                });
            }
            column_sum += value.norm();
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
    for (destination, &value) in scaled[..elements].iter_mut().zip(source) {
        *destination = value * scale;
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

    for column in 0..order {
        let source_start = order * column;
        let destination_start = output_leading * column;
        output[destination_start..destination_start + order]
            .copy_from_slice(&image[source_start..source_start + order]);
    }
    Ok(())
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
fn spectral_function_dyn<E, RuleKey, BT, BC, R, D>(
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
