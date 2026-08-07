use tenet_core::{
    CU1FusionRule, CU1Irrep, FermionParityFusionRule, LoweredMultiplicityFreeAlgebra,
    SU2FusionRule, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep, ZNFusionRule,
    CU1_MAX_TWICE_CHARGE,
};

fn assert_lowered<T: LoweredMultiplicityFreeAlgebra>() {}

fn assert_lowered_roundtrip<R>(
    rule: &R,
    left: R::Sector,
    right: R::Sector,
    expected_channels: &[R::Sector],
) where
    R: LoweredMultiplicityFreeAlgebra,
    R::Sector: std::fmt::Debug,
{
    for sector in [left, right, rule.try_lowered_vacuum().unwrap()] {
        let encoded = rule.try_encode_lowered(sector).unwrap();
        assert_eq!(rule.try_decode_lowered(encoded).unwrap(), sector);
    }
    let dual = rule.try_lowered_dual(left).unwrap();
    assert_eq!(rule.try_lowered_dual(dual).unwrap(), left);

    let mut channels = Vec::new();
    rule.try_for_each_lowered_channel(left, right, &mut |sector| {
        channels.push(sector);
        Ok(())
    })
    .unwrap();
    assert_eq!(channels, expected_channels);
    for &coupled in expected_channels {
        assert_eq!(rule.try_lowered_nsymbol(left, right, coupled).unwrap(), 1);
    }
}

#[test]
fn built_in_providers_implement_core_rule_bridge_traits() {
    assert_lowered::<Z2FusionRule>();
    assert_lowered::<FermionParityFusionRule>();
    assert_lowered::<U1FusionRule>();
    assert_lowered::<SU2FusionRule>();
    assert_lowered::<CU1FusionRule>();
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

#[test]
fn built_in_lowered_bridges_roundtrip_and_enumerate_channels() {
    assert_lowered_roundtrip(&Z2FusionRule, Z2Irrep::ODD, Z2Irrep::ODD, &[Z2Irrep::EVEN]);
    assert_lowered_roundtrip(
        &FermionParityFusionRule,
        Z2Irrep::ODD,
        Z2Irrep::ODD,
        &[Z2Irrep::EVEN],
    );
    assert_lowered_roundtrip(
        &U1FusionRule,
        U1Irrep::new(-2),
        U1Irrep::new(3),
        &[U1Irrep::new(1)],
    );
    assert_lowered_roundtrip(
        &SU2FusionRule,
        tenet_core::SU2Irrep::from_twice_spin(1),
        tenet_core::SU2Irrep::from_twice_spin(1),
        &[
            tenet_core::SU2Irrep::from_twice_spin(0),
            tenet_core::SU2Irrep::from_twice_spin(2),
        ],
    );

    let zn = ZNFusionRule::new(5).unwrap();
    assert_lowered_roundtrip(&zn, zn.irrep(4), zn.irrep(3), &[zn.irrep(2)]);
}
