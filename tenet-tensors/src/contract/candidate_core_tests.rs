//! Route pins for the zero-copy TensorKit candidates (#1468): a contraction
//! whose sources are the identity only after the paired-axis sort or the
//! operand swap, including through a lazy adjoint, resolves to the direct
//! Core GEMM and never projects or transforms an operand. Independent value
//! oracles live in `tenet/tests/contract_candidate_core.rs`.

use std::sync::Arc;

use tenet_core::{
    FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
    ProductFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep, SectorId, SectorLeg,
    U1FusionRule, U1Irrep,
};
use tenet_operations::{OutputAxisOrder, TensorContractSpec};

use crate::contract::fusion::FusionContractOrientation;
use crate::{BoundDynamicFusionMapSpace, FusionOperand, RuleIdentity};

type Context = crate::TensorContractFusionExecutionContext<f64, RuleIdentity>;
type FermionU1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

fn square<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    codomain: &SectorLeg,
    domain: &SectorLeg,
) -> BoundDynamicFusionMapSpace<R> {
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([codomain.clone(), codomain.clone()]),
            FusionProductSpace::new([domain.clone(), domain.clone()]),
        ),
    )
    .unwrap()
}

/// Non-self-dual: `-1` and `1` carry different degeneracies.
fn u1_leg() -> SectorLeg {
    SectorLeg::new(
        [
            (U1Irrep::new(-1).sector_id(), 2),
            (U1Irrep::new(0).sector_id(), 1),
            (U1Irrep::new(1).sector_id(), 3),
        ],
        false,
    )
}

fn su2_leg() -> SectorLeg {
    SectorLeg::new(
        [
            (SU2Irrep::from_twice_spin(0).sector_id(), 2),
            (SU2Irrep::from_twice_spin(1).sector_id(), 1),
            (SU2Irrep::from_twice_spin(2).sector_id(), 1),
        ],
        false,
    )
}

fn fermion_u1_leg(rule: &FermionU1, dual: bool) -> SectorLeg {
    let (even, odd) = (SectorId::new(0), SectorId::new(1));
    let charge = |q: i32| U1Irrep::new(if dual { -q } else { q }).sector_id();
    SectorLeg::new(
        [
            (rule.encode_sector(even, charge(0)), 2),
            (rule.encode_sector(odd, charge(1)), 2),
            (rule.encode_sector(odd, charge(-1)), 1),
            (rule.encode_sector(even, charge(1)), 1),
        ],
        dual,
    )
}

fn data(len: usize, salt: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i * 37 + salt * 11) % 17) as f64 / 8.0 - 1.0)
        .collect()
}

/// `(name, lhs lazy, rhs lazy, lhs axes, rhs axes, output, orientation)`:
/// the audit probes C1, C2, L3, L5, L7 plus the literal controls C0, L4.
type Probe = (
    &'static str,
    bool,
    bool,
    &'static [usize],
    &'static [usize],
    &'static [usize],
    FusionContractOrientation,
);

const PROBES: [Probe; 7] = [
    ("C0", false, false, &[2, 3], &[0, 1], &[0, 1, 2, 3], LHS_RHS),
    ("C1", false, false, &[3, 2], &[1, 0], &[0, 1, 2, 3], LHS_RHS),
    ("C2", false, false, &[0, 1], &[2, 3], &[2, 3, 0, 1], RHS_LHS),
    ("L4", true, false, &[2, 3], &[0, 1], &[0, 1, 2, 3], LHS_RHS),
    ("L3", true, false, &[3, 2], &[1, 0], &[0, 1, 2, 3], LHS_RHS),
    ("L5", false, true, &[0, 1], &[2, 3], &[2, 3, 0, 1], RHS_LHS),
    ("L7", false, true, &[3, 2], &[1, 0], &[0, 1, 2, 3], LHS_RHS),
];
const LHS_RHS: FusionContractOrientation = FusionContractOrientation::LhsRhs;
const RHS_LHS: FusionContractOrientation = FusionContractOrientation::RhsLhs;

fn operand(space: &BoundDynamicFusionMapSpace<impl Sized>, lazy: bool) -> FusionOperand<'_> {
    if lazy {
        FusionOperand::adjoint(space.space())
    } else {
        FusionOperand::direct(space.space())
    }
}

/// Every probe on `V⊗V ← V⊗V` operands resolves to Core in its TensorKit
/// orientation, with no logical projection or adjoint view of a lazy
/// operand, on the Host and on the device compile route.
fn assert_probes_resolve_to_core<R>(provider: &Arc<R>, leg: &SectorLeg)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let space = square(provider, leg, leg);
    let len = space.space().required_len().unwrap();
    let (lhs_data, rhs_data) = (data(len, 1), data(len, 2));
    let mut context = Context::default();
    for (name, lhs_lazy, rhs_lazy, lhs_axes, rhs_axes, output, orientation) in PROBES {
        let axes = TensorContractSpec::new_with_conjugation(
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::from_axes(output),
            lhs_lazy,
            rhs_lazy,
        );
        let mut dst = vec![0.0; len];
        crate::lowering::reset_adjoint_view_build_count();
        crate::contract::reset_fusion_operand_projection_prepares();
        if lhs_lazy || rhs_lazy {
            context
                .tensorcontract_fusion_dyn_prelowered_into(
                    &space,
                    &mut dst,
                    operand(&space, lhs_lazy),
                    &lhs_data,
                    operand(&space, rhs_lazy),
                    &rhs_data,
                    axes,
                    1.0,
                    0.0,
                )
                .unwrap();
        } else {
            context
                .tensorcontract_fusion_dyn_into(
                    &space, &mut dst, &space, &lhs_data, &space, &rhs_data, axes, 1.0, 0.0,
                )
                .unwrap();
        }
        // What: the selected zero-copy candidate runs the direct Core GEMM.
        assert!(context.last_resolution_is_core(), "{name}");
        let swapped = (orientation == RHS_LHS).then_some(RHS_LHS);
        assert_eq!(context.last_resolution_orientation(), swapped, "{name}");
        // What: no logical key projection or adjoint view of a lazy operand.
        assert_eq!(
            crate::contract::fusion_operand_projection_prepares(),
            0,
            "{name}"
        );
        assert_eq!(crate::lowering::adjoint_view_build_count(), 0, "{name}");
        assert!(dst.iter().any(|&x| x != 0.0), "{name}: empty fixture");

        // What: the device compile route takes the same Core candidate.
        let resolution = crate::try_compile_storage_contract_core_route(
            &space,
            operand(&space, lhs_lazy),
            operand(&space, rhs_lazy),
            axes,
        )
        .unwrap()
        .unwrap_or_else(|| panic!("{name}: device route declined Core"));
        assert!(!resolution.is_dynamic_tree(), "{name}");
    }
}

#[test]
fn zero_copy_candidates_resolve_to_core_on_non_self_dual_u1() {
    assert_probes_resolve_to_core(&Arc::new(U1FusionRule), &u1_leg());
}

#[test]
fn zero_copy_candidates_resolve_to_core_on_su2() {
    assert_probes_resolve_to_core(&Arc::new(SU2FusionRule), &su2_leg());
}

#[test]
fn zero_copy_candidates_resolve_to_core_on_fermion_u1() {
    let rule = FermionParityFusionRule.product(U1FusionRule);
    let leg = fermion_u1_leg(&rule, false);
    assert_probes_resolve_to_core(&Arc::new(rule), &leg);
}

/// `A: V⊗V ← V*⊗V*` against `B: V*⊗V* ← V⊗V` sorted into `mul!` form puts
/// dual legs on B's contracted codomain: TensorKit `blas_contract!` copies
/// one operand to twist it, so the Host must not select a Core candidate.
#[test]
fn a_fermionic_twist_keeps_the_sorted_candidate_off_the_host_core_route() {
    let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let (v, v_dual) = (fermion_u1_leg(&rule, false), fermion_u1_leg(&rule, true));
    let lhs = square(&rule, &v, &v_dual);
    let rhs = square(&rule, &v_dual, &v);
    let dst = square(&rule, &v, &v);
    let axes = TensorContractSpec::new(&[3, 2], &[1, 0], OutputAxisOrder::identity());
    let mut out = vec![0.0; dst.space().required_len().unwrap()];
    let mut context = Context::default();
    context
        .tensorcontract_fusion_dyn_into(
            &dst,
            &mut out,
            &lhs,
            &data(lhs.space().required_len().unwrap(), 3),
            &rhs,
            &data(rhs.space().required_len().unwrap(), 4),
            axes,
            1.0,
            0.0,
        )
        .unwrap();
    assert!(!context.last_resolution_is_core());
}
