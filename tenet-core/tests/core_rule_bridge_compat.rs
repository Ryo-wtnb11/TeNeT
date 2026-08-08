//! Channel-order and representability contracts the layout enumerator relies
//! on. These were written against the built-in lowered bridge; the bridge is
//! gone (issue #977) and the same properties are asserted through the checked
//! algebra every provider implements, which is now the only enumeration path.

use tenet_core::{
    CU1FusionRule, CU1Irrep, CheckedFusionAlgebra, FermionParityFusionRule, FusionAlgebraError,
    SU2FusionRule, SU2Irrep, SectorCodec, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
    ZNFusionRule, CU1_MAX_TWICE_CHARGE,
};

fn assert_codec_roundtrip_and_channels<R>(
    rule: &R,
    left: R::Sector,
    right: R::Sector,
    expected_channels: &[R::Sector],
) where
    R: SectorCodec + CheckedFusionAlgebra,
    R::Sector: std::fmt::Debug,
{
    for sector in [&left, &right] {
        let encoded = rule.encode_sector(sector).unwrap();
        assert_eq!(&rule.decode_sector(encoded).unwrap(), sector);
    }
    let left_id = rule.encode_sector(&left).unwrap();
    let right_id = rule.encode_sector(&right).unwrap();

    let dual = rule.try_dual_sector(left_id).unwrap();
    assert_eq!(rule.try_dual_sector(dual).unwrap(), left_id);

    let channels = rule
        .try_fusion_channels(left_id, right_id)
        .unwrap()
        .into_iter()
        .map(|id| rule.decode_sector(id).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(channels, expected_channels);
    for coupled in expected_channels {
        let coupled_id = rule.encode_sector(coupled).unwrap();
        assert_eq!(rule.try_nsymbol(left_id, right_id, coupled_id).unwrap(), 1);
    }
}

#[test]
fn cu1_keeps_complete_ordered_channels() {
    // What: CU(1) enumerates its three channels in the documented order and
    // reports an unrepresentable product rather than truncating it.
    let rule = CU1FusionRule;
    let q = rule.encode_sector(&CU1Irrep::from_twice_charge(1)).unwrap();
    let channels = rule
        .try_fusion_channels(q, q)
        .unwrap()
        .into_iter()
        .map(|id| rule.decode_sector(id).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        channels,
        [
            CU1Irrep::VACUUM,
            CU1Irrep::PSEUDOSCALAR,
            CU1Irrep::from_twice_charge(2),
        ]
    );

    let edge = rule
        .encode_sector(&CU1Irrep::from_twice_charge(CU1_MAX_TWICE_CHARGE))
        .unwrap();
    assert!(matches!(
        rule.try_fusion_channels(edge, edge).unwrap_err(),
        FusionAlgebraError::FusionNotRepresentable { .. }
    ));
}

#[test]
fn built_in_providers_roundtrip_and_enumerate_channels() {
    assert_codec_roundtrip_and_channels(
        &Z2FusionRule,
        Z2Irrep::ODD,
        Z2Irrep::ODD,
        &[Z2Irrep::EVEN],
    );
    assert_codec_roundtrip_and_channels(
        &FermionParityFusionRule,
        Z2Irrep::ODD,
        Z2Irrep::EVEN,
        &[Z2Irrep::ODD],
    );
    assert_codec_roundtrip_and_channels(
        &U1FusionRule,
        U1Irrep::new(2),
        U1Irrep::new(-5),
        &[U1Irrep::new(-3)],
    );
    assert_codec_roundtrip_and_channels(
        &SU2FusionRule,
        SU2Irrep::from_twice_spin(1),
        SU2Irrep::from_twice_spin(1),
        &[SU2Irrep::from_twice_spin(0), SU2Irrep::from_twice_spin(2)],
    );

    let zn = ZNFusionRule::new(5).unwrap();
    assert_codec_roundtrip_and_channels(&zn, zn.irrep(4), zn.irrep(3), &[zn.irrep(2)]);
}
