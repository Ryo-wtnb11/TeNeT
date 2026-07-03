//! Matrix functions of fusion tensors, built from the spectral
//! factorizations: factorize on the device boundary, transform the spectrum
//! on the host, recompose through contraction.

use std::hash::Hash;

use tenet_core::{CoreError, MultiplicityFreeRigidSymbols, TensorMap};
use tenet_dense::DenseExecutor;
use tenet_tensors::{
    adjoint, OperationError, TensorContractBackend, TensorContractFusionExecutionContext,
    TreeTransformBackend, TreeTransformRuleCacheKey,
};

use crate::compose::compose;
use crate::factorize::{eigh_full, svd_compact, FactorScalar, SectorSpectrum};

/// Matrix exponential of a Hermitian endomorphism: `exp(t) = V exp(D) V^H`.
pub fn exp<E, RuleKey, BT, BC, R, D, const N: usize>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    tensor: &TensorMap<D, N, N>,
) -> Result<TensorMap<D, N, N>, OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + Hash,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    spectral_function(dense, context, rule, tensor, &f64::exp)
}

/// Applies a scalar function to a Hermitian endomorphism through its
/// eigendecomposition: `f(t) = V f(D) V^H`.
fn spectral_function<E, RuleKey, BT, BC, R, D, const N: usize>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    tensor: &TensorMap<D, N, N>,
    function: &dyn Fn(f64) -> f64,
) -> Result<TensorMap<D, N, N>, OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + Hash,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let eigh = eigh_full(dense, rule, tensor)?;
    let mapped: Vec<SectorSpectrum> = eigh
        .eigenvalues
        .iter()
        .map(|entry| SectorSpectrum {
            sector: entry.sector,
            values: entry.values.iter().map(|&value| function(value)).collect(),
        })
        .collect();
    let d_tensor = crate::factorize::diagonal_bond_tensor(rule, &mapped, &D::from_real)?;
    let vh = adjoint(rule, &eigh.v)?;
    let vd = compose(context, rule, &eigh.v, &d_tensor)?;
    compose(context, rule, &vd, &vh)
}

/// Moore-Penrose pseudo-inverse via the compact SVD with an
/// `rcond * sigma_max` cutoff: `t^+ = V S^+ U^H`.
pub fn pinv<E, RuleKey, BT, BC, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
    rcond: f64,
) -> Result<TensorMap<D, NIN, NOUT>, OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + Hash,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let svd = svd_compact(dense, rule, tensor)?;
    let sigma_max = svd
        .singular_values
        .iter()
        .flat_map(|entry| entry.values.iter().copied())
        .fold(0.0_f64, f64::max);
    let cutoff = rcond * sigma_max;
    let inverted: Vec<SectorSpectrum> = svd
        .singular_values
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
    let s_plus = crate::factorize::diagonal_bond_tensor(rule, &inverted, &D::from_real)?;
    let v = adjoint(rule, &svd.vh)?;
    let uh = adjoint(rule, &svd.u)?;
    let vs = compose(context, rule, &v, &s_plus)?;
    compose(context, rule, &vs, &uh)
}

/// True inverse of a full-rank endomorphism via the compact SVD; fails when
/// any block is rank-deficient at working precision.
pub fn inv<E, RuleKey, BT, BC, R, D, const N: usize>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    tensor: &TensorMap<D, N, N>,
) -> Result<TensorMap<D, N, N>, OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + Hash,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    if fusion_space.homspace().codomain() != fusion_space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inv requires an endomorphism (codomain == domain)",
        });
    }
    let svd = svd_compact(dense, rule, tensor)?;
    let sigma_max = svd
        .singular_values
        .iter()
        .flat_map(|entry| entry.values.iter().copied())
        .fold(0.0_f64, f64::max);
    let tolerance = sigma_max * 1e-13;
    if svd
        .singular_values
        .iter()
        .any(|entry| entry.values.iter().any(|&sigma| sigma <= tolerance))
    {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inv requires full-rank blocks",
        });
    }
    pinv(dense, context, rule, tensor, 1e-14)
}
