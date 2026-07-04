//! Composition of factor maps with automatic destination allocation.

use std::hash::Hash;

use tenet_core::MultiplicityFreeRigidSymbols;
use tenet_tensors::{
    DynamicFusionMapSpace, OperationError, TensorContractFusionExecutionContext,
    TensorContractSpec, TreeTransformRuleCacheKey,
};

use crate::factorize::{DynFactor, FactorScalar};

/// `lhs . rhs` over the full domain/codomain interface, allocating the
/// destination in the coupled layout. The recomposition step of the derived
/// operations (`V f(D) V^H`, `U Vh`, ...).
pub(crate) fn compose_dyn<RuleKey, BT, BC, R, D>(
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    (lhs_space, lhs_data): (&DynamicFusionMapSpace, &[D]),
    (rhs_space, rhs_data): (&DynamicFusionMapSpace, &[D]),
) -> Result<DynFactor<D>, OperationError>
where
    RuleKey: Clone + Eq + Hash,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    if lhs_space.nin() != rhs_space.nout() {
        return Err(OperationError::RankMismatch {
            expected: lhs_space.nin(),
            actual: rhs_space.nout(),
        });
    }
    let lhs_axes: Vec<usize> = (lhs_space.nout()..lhs_space.rank()).collect();
    let rhs_axes: Vec<usize> = (0..rhs_space.nout()).collect();
    let dst_space =
        DynamicFusionMapSpace::contracted(rule, lhs_space, rhs_space, &lhs_axes, &rhs_axes)?;
    let mut dst_data = vec![
        D::zero();
        dst_space
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?
    ];
    context.tensorcontract_fusion_dyn_into(
        rule,
        &dst_space,
        &mut dst_data,
        lhs_space,
        lhs_data,
        rhs_space,
        rhs_data,
        TensorContractSpec::with_default_output_order(&lhs_axes, &rhs_axes),
        D::one(),
        D::zero(),
    )?;
    Ok((dst_space, dst_data))
}

/// Typed composition over the full interface (test-suite convenience; the
/// production paths use [`compose_dyn`] directly).
#[cfg(test)]
pub(crate) fn compose<RuleKey, BT, BC, R, D, const A: usize, const B: usize, const C: usize>(
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    lhs: &tenet_core::TensorMap<D, A, B>,
    rhs: &tenet_core::TensorMap<D, B, C>,
) -> Result<tenet_core::TensorMap<D, A, C>, OperationError>
where
    RuleKey: Clone + Eq + Hash,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let lhs_space = crate::factorize::dyn_space_of(lhs)?;
    let rhs_space = crate::factorize::dyn_space_of(rhs)?;
    let out = compose_dyn(
        context,
        rule,
        (&lhs_space, lhs.data()),
        (&rhs_space, rhs.data()),
    )?;
    crate::factorize::typed_from_dyn(rule, out)
}
