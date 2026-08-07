//! Issue #971 regression: every shipped multiplicity-free provider states its
//! fusion algebra once on [`CheckedFusionAlgebra`]; `FusionRule::{dual,
//! fusion_channels, nsymbol}` must be exact one-line forwards, so they panic
//! on precisely the sectors the checked entry point rejects — never more
//! permissive (a silent wrong answer on an out-of-domain id) and never less
//! (an unnecessary panic on a representable one).
//!
//! Two providers are deliberately absent. `ProductFusionRule` keeps two
//! independent bodies on purpose: its `FusionRule` impl accepts components
//! that are not `CheckedFusionAlgebra`, so it cannot forward without
//! restricting which providers compose — its infallible `nsymbol` validates
//! indirectly, through whichever components do forward. `tenet-category-data`'s
//! Fibonacci provider is converted but not swept here, because this test lives
//! in `tenet-sectors` and that crate is not a dependency; a reintroduced
//! recursion there would overflow the stack in that crate's own tests.

use std::panic::{self, AssertUnwindSafe};

use tenet_sectors::{
    CU1FusionRule, CU1Irrep, CheckedFusionAlgebra, FermionParityFusionRule, FibonacciFusionRule,
    FusionRule, SU2FusionRule, SectorId, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
    ZNFusionRule, CU1_MAX_TWICE_CHARGE,
};

fn assert_dual_forwards<R: FusionRule + CheckedFusionAlgebra>(rule: &R, ids: &[SectorId]) {
    for &id in ids {
        let checked = rule.try_dual_sector(id);
        let infallible = panic::catch_unwind(AssertUnwindSafe(|| rule.dual(id)));
        match checked {
            Ok(expected) => assert_eq!(
                infallible.expect("dual must not panic on a representable sector"),
                expected,
                "dual({id:?}) disagreed with try_dual_sector on its Ok value"
            ),
            Err(_) => assert!(
                infallible.is_err(),
                "dual({id:?}) must panic exactly where try_dual_sector errs"
            ),
        }
    }
}

fn assert_fusion_channels_forwards<R: FusionRule + CheckedFusionAlgebra>(
    rule: &R,
    pairs: &[(SectorId, SectorId)],
) {
    for &(left, right) in pairs {
        let checked = rule.try_fusion_channels(left, right);
        let infallible =
            panic::catch_unwind(AssertUnwindSafe(|| rule.fusion_channels(left, right)));
        match checked {
            Ok(expected) => assert_eq!(
                infallible.expect("fusion_channels must not panic on representable inputs"),
                expected,
                "fusion_channels({left:?}, {right:?}) disagreed with try_fusion_channels"
            ),
            Err(_) => assert!(
                infallible.is_err(),
                "fusion_channels({left:?}, {right:?}) must panic exactly where try_fusion_channels errs"
            ),
        }
    }
}

fn assert_nsymbol_forwards<R: FusionRule + CheckedFusionAlgebra>(
    rule: &R,
    triples: &[(SectorId, SectorId, SectorId)],
) {
    for &(left, right, coupled) in triples {
        let checked = rule.try_nsymbol(left, right, coupled);
        let infallible =
            panic::catch_unwind(AssertUnwindSafe(|| rule.nsymbol(left, right, coupled)));
        match checked {
            Ok(expected) => assert_eq!(
                infallible.expect("nsymbol must not panic on representable inputs"),
                expected,
                "nsymbol({left:?}, {right:?}, {coupled:?}) disagreed with try_nsymbol"
            ),
            Err(_) => assert!(
                infallible.is_err(),
                "nsymbol({left:?}, {right:?}, {coupled:?}) must panic exactly where try_nsymbol errs"
            ),
        }
    }
}

fn all_pairs(ids: &[SectorId]) -> Vec<(SectorId, SectorId)> {
    ids.iter()
        .flat_map(|&l| ids.iter().map(move |&r| (l, r)))
        .collect()
}

fn all_triples(ids: &[SectorId]) -> Vec<(SectorId, SectorId, SectorId)> {
    ids.iter()
        .flat_map(|&l| {
            ids.iter()
                .flat_map(move |&r| ids.iter().map(move |&c| (l, r, c)))
        })
        .collect()
}

#[test]
fn zn_infallible_entry_points_match_checked_domain() {
    let rule = ZNFusionRule::new(5).unwrap();
    let valid: Vec<SectorId> = (0..5i64).map(|c| rule.irrep(c).into()).collect();
    let invalid = [
        SectorId::new(5),
        SectorId::new(6),
        SectorId::new(usize::MAX),
    ];
    let ids: Vec<SectorId> = valid.iter().copied().chain(invalid).collect();

    assert_dual_forwards(&rule, &ids);
    assert_fusion_channels_forwards(&rule, &all_pairs(&ids));
    assert_nsymbol_forwards(&rule, &all_triples(&ids));
}

#[test]
fn z2_infallible_entry_points_match_checked_domain() {
    let rule = Z2FusionRule;
    let ids = [
        Z2Irrep::EVEN.sector_id(),
        Z2Irrep::ODD.sector_id(),
        SectorId::new(2),
        SectorId::new(3),
        SectorId::new(usize::MAX),
    ];

    assert_dual_forwards(&rule, &ids);
    assert_fusion_channels_forwards(&rule, &all_pairs(&ids));
    assert_nsymbol_forwards(&rule, &all_triples(&ids));
}

#[test]
fn fermion_parity_infallible_entry_points_match_checked_domain() {
    let rule = FermionParityFusionRule;
    let ids = [
        Z2Irrep::EVEN.sector_id(),
        Z2Irrep::ODD.sector_id(),
        SectorId::new(2),
        SectorId::new(usize::MAX),
    ];

    assert_dual_forwards(&rule, &ids);
    assert_fusion_channels_forwards(&rule, &all_pairs(&ids));
    assert_nsymbol_forwards(&rule, &all_triples(&ids));
}

#[test]
fn u1_infallible_entry_points_match_checked_domain() {
    let rule = U1FusionRule;
    // Representable domain, including the two charges whose *generated*
    // dual/fusion output overflows i32 (checked errs on an otherwise-valid
    // input, not merely an invalid id).
    let valid: Vec<SectorId> = [-17, -1, 0, 1, 17, i32::MAX, i32::MIN]
        .into_iter()
        .map(|c| U1Irrep::new(c).sector_id())
        .collect();
    // Only representable when usize is wider than u32 (true for every CI
    // target); the zigzag codec's domain is exactly 0..=u32::MAX.
    let invalid = [
        SectorId::new(u32::MAX as usize + 1),
        SectorId::new(usize::MAX),
    ];
    let ids: Vec<SectorId> = valid.iter().copied().chain(invalid).collect();

    assert_dual_forwards(&rule, &ids);
    assert_fusion_channels_forwards(&rule, &all_pairs(&ids));
    assert_nsymbol_forwards(&rule, &all_triples(&ids));
}

#[test]
fn su2_infallible_entry_points_match_checked_domain() {
    let rule = SU2FusionRule;
    // A sample of the representable doubled-spin domain, including the pair
    // (128, 127) whose fusion *closure* escapes the 254 catalog bound even
    // though both inputs are individually representable.
    let valid: Vec<SectorId> = [0, 1, 2, 127, 128, 253, 254]
        .into_iter()
        .map(SectorId::new)
        .collect();
    let invalid = [
        SectorId::new(255),
        SectorId::new(256),
        SectorId::new(usize::MAX),
    ];
    let ids: Vec<SectorId> = valid.iter().copied().chain(invalid).collect();

    assert_dual_forwards(&rule, &ids);
    assert_fusion_channels_forwards(&rule, &all_pairs(&ids));
    // Full triples over `ids` are O(10^3) racah calls; keep it, this suite is
    // not a hot path.
    assert_nsymbol_forwards(&rule, &all_triples(&ids));
}

#[test]
fn cu1_infallible_entry_points_match_checked_domain() {
    let rule = CU1FusionRule;
    let big = CU1Irrep::from_twice_charge(CU1_MAX_TWICE_CHARGE);
    let valid: Vec<SectorId> = [
        CU1Irrep::VACUUM,
        CU1Irrep::PSEUDOSCALAR,
        CU1Irrep::from_twice_charge(1),
        CU1Irrep::from_twice_charge(2),
        CU1Irrep::from_twice_charge(3),
        big,
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    // `big` fused with itself doubles past `CU1_MAX_TWICE_CHARGE`: both
    // inputs representable, generated output is not.
    let invalid = [
        SectorId::new(CU1_MAX_TWICE_CHARGE as usize + 2),
        SectorId::new(usize::MAX),
    ];
    let ids: Vec<SectorId> = valid.iter().copied().chain(invalid).collect();

    assert_dual_forwards(&rule, &ids);
    assert_fusion_channels_forwards(&rule, &all_pairs(&ids));
    assert_nsymbol_forwards(&rule, &all_triples(&ids));
}

#[test]
fn fibonacci_infallible_entry_points_match_checked_domain() {
    let rule = FibonacciFusionRule;
    let ids = [
        SectorId::new(0),
        SectorId::new(1),
        SectorId::new(2),
        SectorId::new(usize::MAX),
    ];

    assert_dual_forwards(&rule, &ids);
    assert_fusion_channels_forwards(&rule, &all_pairs(&ids));
    assert_nsymbol_forwards(&rule, &all_triples(&ids));
}
