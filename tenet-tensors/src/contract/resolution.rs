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
use tenet_operations::fusion_replay::FusionBlockContractPlan;

use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::{external_axis_is_dual, FusionContractPlan};
use super::fusion_block::{
    compile_fusion_block_contract_plan_prelowered_validated,
    compile_fusion_block_contract_plan_validated, CoreContractPreflight, ValidatedCoreContract,
};

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
    let preflight = CoreContractPreflight::compile(rule, dst, lhs, rhs, axes)?;
    if !preflight.has_conjugation() {
        if let Some(validated) = preflight.validate_core_geometry()? {
            if !rhs_contract_requires_twist(&validated)? {
                let plan = compile_fusion_block_contract_plan_validated(validated)?;
                return Ok(Resolution::Core(Arc::new(plan)));
            }
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
    let preflight = CoreContractPreflight::compile(rule, dst, lhs_logical, rhs_logical, axes)?;
    if let Some(validated) = preflight.validate_core_geometry()? {
        if !rhs_contract_requires_twist(&validated)? {
            let (lhs_op, rhs_op) = validated.storage_ops();
            let plan = compile_fusion_block_contract_plan_prelowered_validated(
                validated,
                lhs_storage,
                rhs_storage,
                lhs_op,
                rhs_op,
            )?;
            return Ok(Resolution::Core(Arc::new(plan)));
        }
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
    let preflight = CoreContractPreflight::compile(rule, dst, lhs, rhs, axes)?;
    super::fusion::reject_fusion_contract_conjugation(axes)?;
    let validated = preflight.require_core_geometry()?;
    compile_fusion_block_contract_plan_validated(validated).map(Arc::new)
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
    let validated = CoreContractPreflight::compile(rule, dst, lhs_logical, rhs_logical, axes)?
        .require_core_geometry()?;
    let (lhs_op, rhs_op) = validated.storage_ops();
    compile_fusion_block_contract_plan_prelowered_validated(
        validated,
        lhs_storage,
        rhs_storage,
        lhs_op,
        rhs_op,
    )
    .map(Arc::new)
}

/// True when the fermionic supertrace twist can be nontrivial: such
/// contractions take the dynamic route, where the twist is applied during
/// rhs materialization; the core direct-GEMM route stays
/// coefficient-free (TensorKit mul! parity).
pub(crate) fn rhs_contract_requires_twist<R>(
    validated: &ValidatedCoreContract<'_, R>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    rhs_contract_homspace_requires_twist(
        validated.rule(),
        validated.rhs().homspace(),
        validated.axis_plan().rhs_contracting_axes.as_slice(),
    )
}

pub(crate) fn rhs_contract_homspace_requires_twist<R>(
    rule: &R,
    rhs: &FusionTreeHomSpace,
    rhs_contracting_axes: &[usize],
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if rule.braiding_style() != tenet_core::BraidingStyleKind::Fermionic {
        return Ok(false);
    }
    for &axis in rhs_contracting_axes {
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
                rhs_contract_homspace_requires_twist(
                    &fermion,
                    rhs.homspace(),
                    axes.rhs_contracting_axes(),
                )
                .unwrap(),
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
        assert!(rhs_contract_homspace_requires_twist(
            &product,
            rhs.homspace(),
            TensorContractSpec::with_default_output_order(&[0], &[0]).rhs_contracting_axes(),
        )
        .unwrap());
    }

    #[test]
    fn core_resolution_derives_axis_plan_and_expected_homspace_once() {
        let rule = U1FusionRule;
        let zero = U1Irrep::new(0).sector_id();
        let lhs = single_sector_matrix_space(&rule, zero, false, false);
        let rhs = single_sector_matrix_space(&rule, zero, false, false);
        let dst = single_sector_matrix_space(&rule, zero, false, false);
        super::super::fusion_block::reset_core_contract_derivations();

        let resolution = compile_resolution(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            || panic!("core contraction must not compile a dense structure"),
            || panic!("core contraction must not compile tree transforms"),
        )
        .unwrap();

        // What: one core execution compile derives each structural authority
        // exactly once before block-layout compilation.
        assert!(matches!(resolution, Resolution::Core(_)));
        assert_eq!(
            super::super::fusion_block::core_contract_derivations(),
            (1, 1)
        );
    }
}
