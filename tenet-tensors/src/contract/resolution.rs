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
use tenet_operations::TensorContractFusionProfile;

use super::dynamic_space::{DynamicFusionMapSpace, FusionOperand, FusionOperandLayout};
use super::fusion::{external_axis_is_dual, FusionContractPlan};
use super::fusion_block::{
    compile_fusion_block_contract_plan_prelowered_validated,
    compile_fusion_block_contract_plan_validated, try_compile_oriented_canonical_core_plan,
    CoreContractPreflight, ValidatedCoreContract,
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
    compile_resolution_with_profile::<R, false>(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        compile_structure,
        compile_dynamic,
        None,
    )
}

/// Compiles and attributes the ordinary eager route without introducing a
/// reusable execution artifact.
pub(crate) fn compile_resolution_profiled<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce()
        -> Result<Option<Arc<TensorContractStructure<f64>>>, OperationError>,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
    profile: &mut TensorContractFusionProfile,
) -> Result<Resolution, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    compile_resolution_with_profile::<R, true>(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        compile_structure,
        compile_dynamic,
        Some(profile),
    )
}

#[allow(clippy::too_many_arguments)]
fn compile_resolution_with_profile<R, const PROFILED: bool>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce()
        -> Result<Option<Arc<TensorContractStructure<f64>>>, OperationError>,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
    mut profile: Option<&mut TensorContractFusionProfile>,
) -> Result<Resolution, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let preflight_start = PROFILED.then(std::time::Instant::now);
    let preflight = CoreContractPreflight::compile(rule, dst, lhs, rhs, axes)?;
    if !preflight.has_conjugation() {
        if let Some(validated) = preflight.validate_core_geometry()? {
            if !validated_rhs_contract_requires_twist(&validated)? {
                record_resolution_preflight(&mut profile, preflight_start);
                let block_plan_start = profile.as_ref().map(|_| std::time::Instant::now());
                let plan = compile_fusion_block_contract_plan_validated(validated, dst, lhs, rhs)?;
                if let Some(start) = block_plan_start {
                    profile
                        .as_deref_mut()
                        .expect("profiled route compilation carries a profile")
                        .core_block_plan_build += start.elapsed();
                }
                return Ok(Resolution::Core(Arc::new(plan)));
            }
        }
        record_resolution_preflight(&mut profile, preflight_start);
        return compile_dynamic_tree_plan::<PROFILED>(compile_dynamic, &mut profile);
    }
    if let Some(structure) = compile_structure()? {
        record_resolution_preflight(&mut profile, preflight_start);
        return Ok(Resolution::Structure(structure));
    }
    record_resolution_preflight(&mut profile, preflight_start);
    compile_dynamic_tree_plan::<PROFILED>(compile_dynamic, &mut profile)
}

fn record_resolution_preflight(
    profile: &mut Option<&mut TensorContractFusionProfile>,
    start: Option<std::time::Instant>,
) {
    if let Some(start) = start {
        profile
            .as_deref_mut()
            .expect("profiled route compilation carries a profile")
            .resolution_preflight += start.elapsed();
    }
}

fn compile_dynamic_tree_plan<const PROFILED: bool>(
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
    profile: &mut Option<&mut TensorContractFusionProfile>,
) -> Result<Resolution, OperationError> {
    let start = PROFILED.then(std::time::Instant::now);
    let plan = compile_dynamic()?;
    if let Some(start) = start {
        profile
            .as_deref_mut()
            .expect("profiled route compilation carries a profile")
            .dynamic_tree_plan_build += start.elapsed();
    }
    Ok(Resolution::DynamicTree(plan))
}

/// Compiles one contraction whose logical and storage spaces are already
/// separated by a validated lazy-adjoint boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_prelowered_resolution<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce()
        -> Result<Option<Arc<TensorContractStructure<f64>>>, OperationError>,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
) -> Result<Resolution, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let preflight = CoreContractPreflight::compile_oriented(
        rule,
        dst.homspace(),
        lhs.oriented_homspace(),
        rhs.oriented_homspace(),
        axes,
    )?;
    if let Some(validated) = preflight.validate_core_geometry()? {
        if !validated_rhs_contract_requires_twist(&validated)? {
            let plan =
                compile_fusion_block_contract_plan_prelowered_validated(validated, dst, lhs, rhs)?;
            return Ok(Resolution::Core(Arc::new(plan)));
        }
    }
    if let Some(structure) = compile_structure()? {
        return Ok(Resolution::Structure(structure));
    }
    Ok(Resolution::DynamicTree(compile_dynamic()?))
}

/// Tries the parent-owned coupled-region route before an adjoint operand
/// derives logical block keys. A miss is not an error: the exact projection is
/// then prepared by the general prelowered lowering path.
pub(crate) fn try_compile_oriented_canonical_core_resolution<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
) -> Result<Option<Resolution>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let preflight = CoreContractPreflight::compile_oriented(
        rule,
        dst.homspace(),
        lhs.oriented_homspace(),
        rhs.oriented_homspace(),
        axes,
    )?;
    let Some(validated) = preflight.validate_core_geometry()? else {
        return Ok(None);
    };
    if validated_rhs_contract_requires_twist(&validated)? {
        return Ok(None);
    }
    let plan = try_compile_oriented_canonical_core_plan(
        &validated,
        dst,
        lhs.storage_space(),
        rhs.storage_space(),
    )?;
    Ok(plan.map(|plan| Resolution::Core(Arc::new(plan))))
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
    compile_fusion_block_contract_plan_validated(validated, dst, lhs, rhs).map(Arc::new)
}

/// Compiles TensorKit `mul!` composition without inserting a fermionic
/// supertrace twist.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_composition_plan<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
    axes: TensorContractSpec<'_>,
) -> Result<Arc<FusionBlockContractPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let validated = CoreContractPreflight::compile_oriented(
        rule,
        dst.homspace(),
        lhs.oriented_homspace(),
        rhs.oriented_homspace(),
        axes,
    )?
    .require_core_geometry()?;
    compile_fusion_block_contract_plan_prelowered_validated(validated, dst, lhs, rhs).map(Arc::new)
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

fn validated_rhs_contract_requires_twist<R>(
    validated: &ValidatedCoreContract<'_, R>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if validated.rule().braiding_style() != tenet_core::BraidingStyleKind::Fermionic {
        return Ok(false);
    }
    Ok(validated
        .rhs_contracting_axes()
        .iter()
        .copied()
        .any(|axis| {
            validated
                .rhs_homspace()
                .external_axis_is_dual(axis)
                .expect("core preflight validated every rhs contraction axis")
        }))
}

pub(crate) fn rhs_contract_homspace_requires_twist<R>(
    rule: &R,
    rhs: &FusionTreeHomSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    rhs_contract_axes_require_twist(rule, rhs, axes.rhs_contracting_axes())
}

fn rhs_contract_axes_require_twist<R>(
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

    #[test]
    fn core_resolution_derives_geometry_once() {
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

        // What: one core compiler invocation derives each geometry authority once.
        assert!(matches!(resolution, Resolution::Core(_)));
        assert_eq!(
            super::super::fusion_block::core_contract_derivations(),
            (1, 1)
        );
    }
}
