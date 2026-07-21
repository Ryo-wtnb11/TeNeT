//! Eager contraction route compilation.
//!
//! Ordinary calls resolve per operation, as TensorKit and QSpace do. Explicit
//! prepared handles own the returned [`Resolution`] and any complete dynamic
//! execution artifact required for lookup-free replay.

use std::sync::Arc;

use tenet_core::{FusionTreeHomSpace, MultiplicityFreeRigidSymbols};

use super::structure::TensorContractStructure;
use crate::OperationError;
use tenet_operations::axis::TensorContractSpec;
use tenet_operations::fusion_replay::{FusionBlockContractPlan, MatrixOp};

use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::{external_axis_is_dual, FusionContractPlan};
use super::fusion_block::{
    compile_fusion_block_contract_plan_prelowered, compile_fusion_block_contract_plan_validated,
    is_core_form_fusion_block_contract, validate_fusion_contract_rule,
};
use super::structure::TensorContractAxisPlan;

/// Resolved execution artifact for one contraction key: the route decision
/// and its compiled plan are one value, never cached separately.
#[derive(Clone, Debug)]
pub(crate) enum Resolution {
    /// Coupled-sector direct GEMM (TensorKit `mul!` shape).
    Core(Arc<FusionBlockContractPlan>),
    /// Source/output tree transforms around a core contraction
    /// (TensorKit `@tensor` shape).
    DynamicTree(Arc<FusionContractPlan>),
    /// Dense one-shot structure for conjugated operands (TeNeT optimization
    /// over the faithful transform-then-contract path).
    Structure(Arc<TensorContractStructure<f64>>),
}

/// Compiles the route and plan for one ordinary contraction.
pub(crate) fn compile_resolution<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce()
        -> Result<Option<Arc<TensorContractStructure<f64>>>, OperationError>,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
) -> Result<Resolution, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    validate_fusion_contract_rule(rule, dst, lhs, rhs)?;
    TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), dst.rank(), axes)?;
    if !axes.lhs_conjugate() && !axes.rhs_conjugate() {
        if is_core_form_fusion_block_contract(rule, dst, lhs, rhs, axes)?
            && !rhs_contract_requires_twist(rule, rhs, axes)?
        {
            let plan = compile_fusion_block_contract_plan_validated(rule, dst, lhs, rhs, axes)?;
            return Ok(Resolution::Core(Arc::new(plan)));
        }
        return Ok(Resolution::DynamicTree(compile_dynamic()?));
    }
    if let Some(structure) = compile_structure()? {
        return Ok(Resolution::Structure(structure));
    }
    Ok(Resolution::DynamicTree(compile_dynamic()?))
}

/// Compiles one contraction whose logical and storage spaces are already
/// separated by a validated lazy-adjoint boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_prelowered_resolution<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs_logical: &DynamicFusionMapSpace,
    lhs_storage: &DynamicFusionMapSpace,
    rhs_logical: &DynamicFusionMapSpace,
    rhs_storage: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce()
        -> Result<Option<Arc<TensorContractStructure<f64>>>, OperationError>,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
) -> Result<Resolution, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    lhs_storage.validate_rule(rule)?;
    rhs_storage.validate_rule(rule)?;
    validate_fusion_contract_rule(rule, dst, lhs_logical, rhs_logical)?;
    TensorContractAxisPlan::compile(lhs_logical.rank(), rhs_logical.rank(), dst.rank(), axes)?;
    let logical_axes = TensorContractSpec::new(
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axes.output_permutation(),
    );
    if is_core_form_fusion_block_contract(rule, dst, lhs_logical, rhs_logical, logical_axes)?
        && !rhs_contract_requires_twist(rule, rhs_logical, logical_axes)?
    {
        let plan = compile_fusion_block_contract_plan_prelowered(
            rule,
            dst,
            lhs_logical,
            lhs_storage,
            rhs_logical,
            rhs_storage,
            logical_axes,
            if axes.lhs_conjugate() {
                MatrixOp::Adjoint
            } else {
                MatrixOp::Identity
            },
            if axes.rhs_conjugate() {
                MatrixOp::Adjoint
            } else {
                MatrixOp::Identity
            },
        )?;
        return Ok(Resolution::Core(Arc::new(plan)));
    }
    if let Some(structure) = compile_structure()? {
        return Ok(Resolution::Structure(structure));
    }
    Ok(Resolution::DynamicTree(compile_dynamic()?))
}

/// Compiles the coupled block plan for already-materialized core operands.
pub(crate) fn compile_core_plan<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<Arc<FusionBlockContractPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    validate_fusion_contract_rule(rule, dst, lhs, rhs)?;
    TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), dst.rank(), axes)?;
    compile_fusion_block_contract_plan_validated(rule, dst, lhs, rhs, axes).map(Arc::new)
}

/// Compiles TensorKit `mul!` composition without inserting a fermionic
/// supertrace twist.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_composition_plan<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs_logical: &DynamicFusionMapSpace,
    lhs_storage: &DynamicFusionMapSpace,
    rhs_logical: &DynamicFusionMapSpace,
    rhs_storage: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<Arc<FusionBlockContractPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    lhs_storage.validate_rule(rule)?;
    rhs_storage.validate_rule(rule)?;
    validate_fusion_contract_rule(rule, dst, lhs_logical, rhs_logical)?;
    TensorContractAxisPlan::compile(lhs_logical.rank(), rhs_logical.rank(), dst.rank(), axes)?;
    let logical_axes = TensorContractSpec::new(
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axes.output_permutation(),
    );
    compile_fusion_block_contract_plan_prelowered(
        rule,
        dst,
        lhs_logical,
        lhs_storage,
        rhs_logical,
        rhs_storage,
        logical_axes,
        if axes.lhs_conjugate() {
            MatrixOp::Adjoint
        } else {
            MatrixOp::Identity
        },
        if axes.rhs_conjugate() {
            MatrixOp::Adjoint
        } else {
            MatrixOp::Identity
        },
    )
    .map(Arc::new)
}

/// True when the fermionic supertrace twist can be nontrivial: such
/// contractions take the dynamic route, where the twist is applied during
/// rhs materialization; the core direct-GEMM route stays
/// coefficient-free (TensorKit mul! parity).
pub(crate) fn rhs_contract_requires_twist<R>(
    rule: &R,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    rhs_contract_homspace_requires_twist(rule, rhs.homspace(), axes)
}

pub(crate) fn rhs_contract_homspace_requires_twist<R>(
    rule: &R,
    rhs: &FusionTreeHomSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if rule.braiding_style() != tenet_core::BraidingStyleKind::Fermionic {
        return Ok(false);
    }
    for &axis in axes.rhs_contracting_axes() {
        if external_axis_is_dual(rhs, axis)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenet_core::{
        FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace, ProductFusionRuleExt,
        SU2FusionRule, SU2Irrep, SectorId, SectorLeg, U1FusionRule, U1Irrep,
    };

    fn single_sector_matrix_space<R>(
        rule: &R,
        sector: SectorId,
        codomain_dual: bool,
        domain_dual: bool,
    ) -> DynamicFusionMapSpace
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], codomain_dual)]),
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], domain_dual)]),
        );
        let count = hom.fusion_tree_keys(rule).len();
        DynamicFusionMapSpace::from_degeneracy_shapes(rule, hom, vec![vec![1, 1]; count]).unwrap()
    }

    #[test]
    fn rhs_twist_requirement_uses_external_domain_dual_and_product_parity() {
        let fermion = FermionParityFusionRule;
        let odd = SectorId::new(1);
        let cases = [
            (false, false, 0, false),
            (true, false, 0, true),
            (false, false, 1, true),
            (false, true, 1, false),
        ];
        for (codomain_dual, domain_dual, rhs_axis, expected) in cases {
            let rhs = single_sector_matrix_space(&fermion, odd, codomain_dual, domain_dual);
            let rhs_axes = [rhs_axis];
            let axes = TensorContractSpec::with_default_output_order(&[0], &rhs_axes);
            // What: codomain uses its stored dual flag, while domain external
            // duality is the inverse of its stored flag.
            assert_eq!(
                rhs_contract_requires_twist(&fermion, &rhs, axes).unwrap(),
                expected
            );
        }

        let fp_u1 = FermionParityFusionRule.product(U1FusionRule);
        let odd_charge = fp_u1.encode_sector(odd, U1Irrep::new(0).sector_id());
        let product = fp_u1.product(SU2FusionRule);
        let odd_product =
            product.encode_sector(odd_charge, SU2Irrep::from_twice_spin(0).sector_id());
        let rhs = single_sector_matrix_space(&product, odd_product, true, false);
        // What: a bosonic U(1) x SU(2) component does not erase the odd fZ2
        // twist on an externally dual product-sector axis.
        assert!(rhs_contract_requires_twist(
            &product,
            &rhs,
            TensorContractSpec::with_default_output_order(&[0], &[0]),
        )
        .unwrap());
    }
}
