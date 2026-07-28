use tenet_core::{
    CU1FusionRule, CU1Irrep, FermionParityFusionRule, FibonacciFusionRule,
    LoweredMultiplicityFreeAlgebra, MultiplicityFreePivotalSymbols, SU2FusionRule, U1FusionRule,
    Z2FusionRule, CU1_MAX_TWICE_CHARGE,
};

fn assert_lowered<T: LoweredMultiplicityFreeAlgebra>() {}

fn assert_pivotal<T: MultiplicityFreePivotalSymbols>() {}

#[test]
fn built_in_providers_implement_core_rule_bridge_traits() {
    assert_lowered::<Z2FusionRule>();
    assert_lowered::<FermionParityFusionRule>();
    assert_lowered::<U1FusionRule>();
    assert_lowered::<SU2FusionRule>();
    assert_lowered::<CU1FusionRule>();
    assert_pivotal::<Z2FusionRule>();
    assert_pivotal::<FermionParityFusionRule>();
    assert_pivotal::<FibonacciFusionRule>();
}

#[test]
fn cu1_lowered_bridge_keeps_complete_ordered_channels() {
    let rule = CU1FusionRule;
    let q = CU1Irrep::from_twice_charge(1);
    let mut channels = Vec::new();
    rule.try_for_each_lowered_channel(q, q, &mut |sector| {
        channels.push(sector);
        Ok(())
    })
    .unwrap();
    assert_eq!(
        channels,
        [
            CU1Irrep::VACUUM,
            CU1Irrep::PSEUDOSCALAR,
            CU1Irrep::from_twice_charge(2),
        ]
    );
    let edge = CU1Irrep::from_twice_charge(CU1_MAX_TWICE_CHARGE);
    let mut emitted = false;
    let error = rule
        .try_for_each_lowered_channel(edge, edge, &mut |_| {
            emitted = true;
            Ok(())
        })
        .unwrap_err();
    assert!(!emitted);
    assert!(matches!(
        error.into_checked_fusion_algebra(),
        tenet_core::FusionAlgebraError::FusionNotRepresentable { .. }
    ));
}
