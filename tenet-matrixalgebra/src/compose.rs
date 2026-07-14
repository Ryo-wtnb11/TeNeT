//! Composition of factor maps with automatic destination allocation.

use std::hash::Hash;
#[cfg(test)]
use std::sync::Arc;

use tenet_core::MultiplicityFreeRigidSymbols;
use tenet_tensors::{
    OperationError, TensorContractFusionExecutionContext, TensorContractSpec,
    TreeTransformRuleCacheKey,
};

use crate::factorize::{BoundDynFactor, FactorScalar};

/// Recomposition of provider-bound factors. The destination is created from
/// the operands' shared provider and remains bound to that exact allocation.
pub(crate) fn compose_bound_dyn<RuleKey, BT, BC, R, D>(
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    lhs: &BoundDynFactor<R, D>,
    rhs: &BoundDynFactor<R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let lhs_space = lhs.space();
    let rhs_space = rhs.space();
    if lhs_space.space().nin() != rhs_space.space().nout() {
        return Err(OperationError::RankMismatch {
            expected: lhs_space.space().nin(),
            actual: rhs_space.space().nout(),
        });
    }
    let lhs_axes: Vec<usize> = (lhs_space.space().nout()..lhs_space.space().rank()).collect();
    let rhs_axes: Vec<usize> = (0..rhs_space.space().nout()).collect();
    let dst_space = tenet_tensors::BoundDynamicFusionMapSpace::contracted_multiplicity_free(
        lhs_space, rhs_space, &lhs_axes, &rhs_axes,
    )?;
    let mut dst_data = vec![
        D::zero();
        dst_space
            .space()
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?
    ];
    context.tensorcontract_fusion_dyn_into(
        &dst_space,
        &mut dst_data,
        lhs_space,
        lhs.data(),
        rhs_space,
        rhs.data(),
        TensorContractSpec::with_default_output_order(&lhs_axes, &rhs_axes),
        D::one(),
        D::zero(),
    )?;
    let nout = dst_space.space().nout();
    let nin = dst_space.space().nin();
    BoundDynFactor::from_bound(dst_space, dst_data, nout, nin)
}

/// Typed composition over the full interface (test-suite convenience; the
/// production paths use [`compose_bound_dyn`] directly).
#[cfg(test)]
pub(crate) fn compose<RuleKey, BT, BC, R, D, const A: usize, const B: usize, const C: usize>(
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    lhs: &tenet_core::TensorMap<D, A, B>,
    rhs: &tenet_core::TensorMap<D, B, C>,
) -> Result<tenet_core::TensorMap<D, A, C>, OperationError>
where
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: Clone
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let provider = Arc::new(rule.clone());
    let lhs = crate::factorize::BoundTensorMap::try_new(Arc::clone(&provider), lhs.clone())?;
    let rhs = crate::factorize::BoundTensorMap::try_new(Arc::clone(&provider), rhs.clone())?;
    let lhs = BoundDynFactor::from_bound(lhs.space().clone(), lhs.data().to_vec(), A, B)?;
    let rhs = BoundDynFactor::from_bound(rhs.space().clone(), rhs.data().to_vec(), B, C)?;
    let out = compose_bound_dyn(context, &lhs, &rhs)?;
    Ok(
        crate::factorize::typed_from_bound_factor::<R, D, A, C>(out)?
            .into_parts()
            .1,
    )
}
