//! Exact cross-sector ties keep the sector TensorKit keeps (#1381).
//!
//! TensorKit 0.17.1 `findtruncated(::SectorVector, ...)`
//! (`src/factorizations/truncation.jl`) sorts the flat `parent(values)` with
//! `sortperm`/`partialsortperm`, whose ties go to the lower flat index. The
//! flat order is the `SectorVector`'s block order, which is sorted by
//! TensorKitSectors `isless`. So `truncrank` keeps the `isless`-earlier sector
//! first and `truncerror` discards it first.
//!
//! Every expected count below was produced by running TensorKit 0.17.1 /
//! TensorKitSectors 0.3.9 `MAK.findtruncated` (and `svd_trunc`/`eigh_trunc`
//! for the end-to-end cases) on the same fixture. It also follows by hand from
//! the `isless` order quoted beside each case. No expectation is read back from
//! TeNeT.

use std::sync::Arc;

use tenet::core::{
    product_sector, CheckedGenericFusion, FermionParityFusionRule, FusionRule, InfallibleGeneric,
    ProductFusionRuleExt, SU2FusionRule, SU2Irrep, SectorOrderKey, U1FusionRule, U1Irrep,
    Z2FusionRule, Z2Irrep, ZNFusionRule,
};
use tenet::prelude::{Runtime, TensorMap, Truncation};
use tenet::typed::{GradedSpace, SectorSpectrum};

/// Kept count per entry, in entry order, from `GradedSpace::find_truncated`.
macro_rules! kept {
    ($rule:expr, [$(($sector:expr, [$($value:expr),*])),* $(,)?], $truncation:expr) => {{
        let entries = vec![$(($sector, vec![$($value),*])),*];
        let leg = GradedSpace::try_new_with_arc(
            Arc::new($rule),
            entries.iter().map(|(sector, values)| (sector.clone(), values.len())),
        )
        .unwrap();
        let spectra: Vec<SectorSpectrum<_>> = entries
            .iter()
            .map(|(sector, values)| SectorSpectrum {
                sector: sector.clone(),
                values: values.clone(),
            })
            .collect();
        let found = leg.find_truncated(&spectra, &$truncation).unwrap();
        entries
            .iter()
            .map(|(sector, _)| found.selection.subspace().degeneracy(sector).unwrap())
            .collect::<Vec<usize>>()
    }};
}

fn u1(charge: i32) -> U1Irrep {
    U1Irrep::new(charge)
}

/// TeNeT's `relative_error(rtol)` budget is `(rtol * norm)^2`; this `rtol`
/// gives the fixture's budget `0.12^2`, TensorKit's `truncerror(; atol = 0.12)`.
/// One `0.1` fits in it, two do not.
fn one_small_value_budget() -> Truncation {
    Truncation::relative_error(0.12 / 1.02_f64.sqrt()).unwrap()
}

#[test]
fn u1_opposite_charge_tie_keeps_the_positive_charge_under_rank() {
    // isless: 0 < +1 < -1. TeNeT ids are 0, 2, 1, so id order kept -1.
    let kept = kept!(
        U1FusionRule,
        [(u1(0), [3.0]), (u1(1), [2.0]), (u1(-1), [2.0])],
        Truncation::rank(2)
    );
    assert_eq!(kept, [1, 1, 0]);
}

#[test]
fn u1_opposite_charge_tie_discards_the_positive_charge_under_discard_weight() {
    // Ascending stable sort discards the isless-earlier +1 first.
    let kept = kept!(
        U1FusionRule,
        [(u1(0), [1.0]), (u1(1), [0.1]), (u1(-1), [0.1])],
        one_small_value_budget()
    );
    assert_eq!(kept, [1, 0, 1]);
}

#[test]
fn u1_tie_inside_and_across_sectors_follows_the_flat_order() {
    // Sorted: 3 (0), 2 (+1), 2 (-1), 1 (+1), 1 (-1).
    let kept = kept!(
        U1FusionRule,
        [(u1(0), [3.0]), (u1(1), [2.0, 1.0]), (u1(-1), [2.0, 1.0])],
        Truncation::rank(4)
    );
    assert_eq!(kept, [1, 2, 1]);
}

#[test]
fn u1_x_z2_opposite_charge_tie_follows_the_product_order() {
    // Product isless over findindex - 1: (+1, 0) -> (3, 0), (-1, 0) -> (4, 0).
    let rule = || U1FusionRule.product(Z2FusionRule);
    let sector = |charge| product_sector(u1(charge), Z2Irrep::EVEN);
    let rank = kept!(
        rule(),
        [(sector(0), [3.0]), (sector(1), [2.0]), (sector(-1), [2.0])],
        Truncation::rank(2)
    );
    assert_eq!(rank, [1, 1, 0]);
    let error = kept!(
        rule(),
        [(sector(0), [1.0]), (sector(1), [0.1]), (sector(-1), [0.1])],
        one_small_value_budget()
    );
    assert_eq!(error, [1, 0, 1]);
}

#[test]
fn u1_x_u1_tie_uses_tensorkit_positions_not_integer_ranks() {
    // TensorKit's U(1) positions interleave half-integers: +1 -> 3, -1 -> 4.
    // (-1, 0) -> degree 4 precedes (+1, +1) -> degree 6. Ranks over integer
    // charges alone (+1 -> 1, -1 -> 2) would tie the degrees at 2 and put
    // (+1, +1) first.
    let kept = kept!(
        U1FusionRule.product(U1FusionRule),
        [
            (product_sector(u1(1), u1(1)), [1.0]),
            (product_sector(u1(-1), u1(0)), [1.0]),
        ],
        Truncation::rank(1)
    );
    assert_eq!(kept, [0, 1]);
}

#[test]
fn nested_product_tie_follows_the_flattened_tensorkit_order() {
    // TensorKit flattens fP ⊠ SU(2) ⊠ fP: (even, 1, even) -> (0, 2, 0) and
    // (odd, 0, odd) -> (1, 0, 1) both have degree 2, and (0, 2, 0) is
    // lexicographically first. Comparing the nested pair ((fP, SU2), fP) would
    // put (odd, 0, odd) first. The spin-1 sector weighs 3, so rank 3 keeps
    // exactly the sector that comes first.
    let rule = FermionParityFusionRule
        .product(SU2FusionRule)
        .product(FermionParityFusionRule);
    let kept = kept!(
        rule,
        [
            (
                product_sector(
                    product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                    Z2Irrep::EVEN
                ),
                [1.0]
            ),
            (
                product_sector(
                    product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(0)),
                    Z2Irrep::ODD
                ),
                [1.0]
            ),
        ],
        Truncation::rank(3)
    );
    assert_eq!(kept, [1, 0]);
}

#[test]
fn su2_tie_keeps_the_smaller_spin_first() {
    // isless: 1/2 < 1. Weights 2 and 3: rank 3 fits only the first.
    let kept = kept!(
        SU2FusionRule,
        [
            (SU2Irrep::from_twice_spin(1), [1.0]),
            (SU2Irrep::from_twice_spin(2), [1.0]),
        ],
        Truncation::rank(3)
    );
    assert_eq!(kept, [1, 0]);
}

#[test]
fn zn_tie_keeps_the_smaller_charge_first() {
    let rule = ZNFusionRule::new(3).unwrap();
    let kept = kept!(
        rule.clone(),
        [
            (rule.irrep(0), [3.0]),
            (rule.irrep(1), [2.0]),
            (rule.irrep(2), [2.0]),
        ],
        Truncation::rank(2)
    );
    assert_eq!(kept, [1, 1, 0]);
}

#[test]
fn untied_truncation_is_decided_by_magnitude_alone() {
    let kept = kept!(
        U1FusionRule,
        [
            (u1(0), [3.0]),
            (u1(1), [1.5]),
            (u1(-1), [2.0]),
            (u1(2), [1.0]),
        ],
        Truncation::rank(2)
    );
    assert_eq!(kept, [1, 0, 1, 0]);
}

#[test]
fn svd_and_eigh_trunc_keep_tensorkits_sector_at_a_tie() {
    // TensorKit: svd_trunc/eigh_trunc(id(Rep[U1](0 => 1, 1 => 1, -1 => 1)),
    // truncrank(2)) both return Rep[U1](0 => 1, 1 => 1).
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(u1(0), 1), (u1(1), 1), (u1(-1), 1)],
    )
    .unwrap();
    let source: TensorMap<U1FusionRule, f64> = TensorMap::id(&runtime, [&leg]).unwrap();
    let expected =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(u1(0), 1), (u1(1), 1)]).unwrap();
    let svd = source.svd_trunc(&Truncation::rank(2)).unwrap();
    assert_eq!(svd.u.domain()[0], expected);
    let eigh = source.eigh_trunc(&Truncation::rank(2)).unwrap();
    assert_eq!(eigh.d.domain()[0], expected);
}

#[test]
fn u1_order_keys_are_tensorkit_findindex_positions() {
    let keys: Vec<SectorOrderKey> = [0, 1, -1, 2, -2]
        .map(|charge| U1FusionRule.sector_order_key(u1(charge).into()))
        .into();
    let expected: Vec<SectorOrderKey> = [0, 3, 4, 7, 8].map(SectorOrderKey::position).into();
    assert_eq!(keys, expected);
    // The checked-Generic truncation path reads the key through this adapter.
    assert_eq!(
        CheckedGenericFusion::sector_order_key(
            &InfallibleGeneric::new(&U1FusionRule),
            u1(-1).into()
        ),
        SectorOrderKey::position(4)
    );
}
