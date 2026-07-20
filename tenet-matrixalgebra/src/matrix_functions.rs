//! Matrix functions of fusion tensors, built from the spectral
//! factorizations: factorize on the device boundary, transform the spectrum
//! on the host, recompose through contraction.

use std::hash::Hash;

use tenet_core::MultiplicityFreeRigidSymbols;
use tenet_dense::DenseExecutor;
use tenet_tensors::{
    OperationError, TensorContractBackend, TensorContractFusionExecutionContext,
    TreeTransformBackend, TreeTransformRuleCacheKey,
};

use crate::compose::compose_bound_dyn;
use crate::factorize::{
    adjoint_bound_factor, eigh_full_dyn, scale_axis_by_spectrum, svd_compact_factors_dyn,
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
    spectral_function_dyn(dense, context, input, &f64::exp)
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

/// True inverse of a full-rank map between isomorphic spaces via the compact
/// SVD; fails when any block is rank-deficient at working precision.
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
    if !input.space().codomain_isomorphic_to_domain()? {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inv requires isomorphic codomain and domain",
        });
    }
    // Why not block LU like TensorKit: DenseExecutor does not yet expose an
    // LU/solve capability, so retain one compact SVD until that backend seam
    // exists.
    let (u, vh, mut singular_values) = svd_compact_factors_dyn(dense, input)?;
    let has_rank_deficient_sector = singular_values.iter().any(|entry| {
        let sigma_max = entry.values.first().copied().unwrap_or(0.0);
        let tolerance = D::epsilon() * entry.values.len() as f64 * sigma_max;
        entry
            .values
            .iter()
            .any(|&sigma| !sigma.is_finite() || sigma <= tolerance)
    });
    if has_rank_deficient_sector {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inv requires full-rank blocks",
        });
    }
    for entry in &mut singular_values {
        for sigma in &mut entry.values {
            *sigma = sigma.recip();
        }
    }
    inverse_from_factors(context, u, vh, &singular_values)
}
