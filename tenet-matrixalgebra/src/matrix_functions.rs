//! Matrix functions of fusion tensors, built from the spectral
//! factorizations: factorize on the device boundary, transform the spectrum
//! on the host, recompose through contraction.

use std::hash::Hash;

use tenet_core::{MultiplicityFreeRigidSymbols, TensorMap};
use tenet_dense::DenseExecutor;
use tenet_tensors::{
    adjoint_dyn, DynamicFusionMapSpace, OperationError, TensorContractBackend,
    TensorContractFusionExecutionContext, TreeTransformBackend, TreeTransformRuleCacheKey,
};

use crate::compose::compose_dyn;
use crate::factorize::{
    diagonal_bond_tensor_dyn, dyn_space_of, eigh_full_dyn, svd_compact_dyn, typed_from_dyn,
    DynFactor, FactorScalar, SectorSpectrum,
};

/// Matrix exponential of a Hermitian endomorphism: `exp(t) = V exp(D) V^H`.
pub fn exp<E, RuleKey, BT, BC, R, D, const N: usize>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    tensor: &TensorMap<D, N, N>,
) -> Result<TensorMap<D, N, N>, OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let out = exp_dyn(dense, context, rule, &dyn_space_of(tensor)?, tensor.data())?;
    typed_from_dyn(rule, out)
}

/// Dynamic-rank [`exp`].
pub fn exp_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<DynFactor<D>, OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    spectral_function_dyn(dense, context, rule, space, data, &f64::exp)
}

/// Applies a scalar function to a Hermitian endomorphism through its
/// eigendecomposition: `f(t) = V f(D) V^H`.
fn spectral_function_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
    function: &dyn Fn(f64) -> f64,
) -> Result<DynFactor<D>, OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let eigh = eigh_full_dyn(dense, rule, space, data)?;
    let mapped: Vec<SectorSpectrum> = eigh
        .eigenvalues
        .iter()
        .map(|entry| SectorSpectrum {
            sector: entry.sector,
            values: entry.values.iter().map(|&value| function(value)).collect(),
        })
        .collect();
    let d_factor = diagonal_bond_tensor_dyn(rule, &mapped, &D::from_real)?;
    let vh = adjoint_dyn(rule, &eigh.v.0, &eigh.v.1)?;
    let vd = compose_dyn(
        context,
        rule,
        (&eigh.v.0, &eigh.v.1),
        (&d_factor.0, &d_factor.1),
    )?;
    compose_dyn(context, rule, (&vd.0, &vd.1), (&vh.0, &vh.1))
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
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let out = pinv_dyn(
        dense,
        context,
        rule,
        &dyn_space_of(tensor)?,
        tensor.data(),
        rcond,
    )?;
    typed_from_dyn(rule, out)
}

/// Dynamic-rank [`pinv`].
pub fn pinv_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
    rcond: f64,
) -> Result<DynFactor<D>, OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let svd = svd_compact_dyn(dense, rule, space, data)?;
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
    let s_plus = diagonal_bond_tensor_dyn(rule, &inverted, &D::from_real)?;
    let v = adjoint_dyn(rule, &svd.vh.0, &svd.vh.1)?;
    let uh = adjoint_dyn(rule, &svd.u.0, &svd.u.1)?;
    let vs = compose_dyn(context, rule, (&v.0, &v.1), (&s_plus.0, &s_plus.1))?;
    compose_dyn(context, rule, (&vs.0, &vs.1), (&uh.0, &uh.1))
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
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let out = inv_dyn(dense, context, rule, &dyn_space_of(tensor)?, tensor.data())?;
    typed_from_dyn(rule, out)
}

/// Dynamic-rank [`inv`].
pub fn inv_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<DynFactor<D>, OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inv requires an endomorphism (codomain == domain)",
        });
    }
    let svd = svd_compact_dyn(dense, rule, space, data)?;
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
    pinv_dyn(dense, context, rule, space, data, 1e-14)
}
