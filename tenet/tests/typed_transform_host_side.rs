//! Host-side gates of the device structural-transform leaf (#1322) that need
//! no CUDA device — and, deliberately, no `cuda` feature either, so ordinary
//! CI runs them.
//!
//! Two things live here:
//!
//! 1. the pin that makes the dense physical-basis oracle *independent*
//!    evidence. `typed_cuda_transform.rs` compares a device permute against
//!    `common::permute_dense` of the source's `to_physical_dense()`; that is
//!    only an oracle if the same function is first shown to agree with the
//!    Host for the providers it is used on;
//! 2. the Host half of the Runtime device-state contract: a Runtime built
//!    without a device reports no device transform state and still clears its
//!    Host transform store.

mod common;

use std::sync::Arc;

use common::permute_dense;
use tenet::core::{ProductFusionRuleExt, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::typed::{GradedSpace, Runtime, TensorMap};

fn assert_close(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (left - right).abs() <= 1e-12 * (1.0 + right.abs()),
            "{what}: element {index} is {left}, expected {right}"
        );
    }
}

/// The fill `typed_cuda_transform.rs` uses, so the pin and the device gate
/// exercise the same values.
fn real_fill<S: std::fmt::Debug>(_trees: &tenet::typed::BlockFusionTrees<S>, idx: &[usize]) -> f64 {
    let mut value = 0.37;
    for (axis, &index) in idx.iter().enumerate() {
        value = value * 1.7 + (index as f64 + 1.0) * (axis as f64 + 2.0);
    }
    value
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

#[test]
fn the_dense_permute_oracle_agrees_with_the_host_for_u1_and_su2() {
    let runtime = Runtime::builder().build().unwrap();

    let u1 = u1_leg();
    let u1_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    let source = u1_tensor.to_physical_dense().unwrap();
    let permuted = u1_tensor.permute(&[1, 0], &[3, 2]).unwrap();
    let (shape, data) = permute_dense(&source.shape, &source.data, &[1, 0, 3, 2]);
    let actual = permuted.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape, "U(1) oracle shape");
    assert_close(&actual.data, &data, "U(1) oracle");

    // SU(2) is the case that matters: the reduced-block replay must recouple
    // to produce what is, physically, a bare axis permutation.
    let su2 = su2_leg();
    let su2_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&su2, &su2], [&su2, &su2], real_fill).unwrap();
    let source = su2_tensor.to_physical_dense().unwrap();
    let permuted = su2_tensor.permute(&[1, 0], &[3, 2]).unwrap();
    let (shape, data) = permute_dense(&source.shape, &source.data, &[1, 0, 3, 2]);
    let actual = permuted.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape, "SU(2) oracle shape");
    assert_close(&actual.data, &data, "SU(2) oracle");

    // Non-vacuity: the reduced payload is not a reordering of the source, so
    // the oracle really did survive a recoupling.
    let sorted = |data: &[f64]| {
        let mut values = data.to_vec();
        values.sort_by(|left, right| left.partial_cmp(right).unwrap());
        values
    };
    assert_ne!(
        sorted(su2_tensor.data()),
        sorted(permuted.data()),
        "SU(2) permute must apply coefficients other than 1"
    );
}

#[test]
fn a_runtime_without_a_device_reports_no_device_transform_state_and_still_clears() {
    // The device half of `clear_tree_transform_cache` is reached only through
    // the device lease, so a device-less Runtime must clear its Host store and
    // do nothing else — including when the `cuda` feature is compiled in.
    let runtime = Runtime::builder().build().unwrap();
    #[cfg(feature = "cuda")]
    assert!(runtime.cuda_tree_transform_stats().is_none());

    let v = u1_leg();
    let tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&v, &v], [&v, &v], real_fill).unwrap();
    let _ = tensor.permute(&[1, 0], &[3, 2]).unwrap();
    assert!(runtime.tree_transform_cache_info().entries() > 0);

    runtime.clear_tree_transform_cache();

    assert_eq!(runtime.tree_transform_cache_info().entries(), 0);
    #[cfg(feature = "cuda")]
    assert!(runtime.cuda_tree_transform_stats().is_none());
}

// ---------------------------------------------------------------------------
// The Host contract the device `*_overwrite_into` mirrors (issue #1329)
// ---------------------------------------------------------------------------

/// The positions that hold a NaN, so a NaN *pattern* can be compared rather
/// than merely "some NaN survived".
fn nan_positions(data: &[f64]) -> Vec<usize> {
    data.iter()
        .enumerate()
        .filter(|(_, value)| value.is_nan())
        .map(|(index, _)| index)
        .collect()
}

/// The Host precondition order and wording that `typed_cuda_transform_contracts
/// .rs` mirrors on the device. Pinned here, without a device and without the
/// `cuda` feature, so ordinary CI catches a Host drift that would silently
/// make the device mirror wrong rather than failing it.
///
/// Only the two storage nouns differ on device ("host" becomes "CUDA"), which
/// is why this test spells them out.
#[test]
fn the_host_overwrite_into_preconditions_have_a_fixed_order_and_wording() {
    let runtime = Runtime::builder().build().unwrap();
    let v = u1_leg();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&v, &v], [&v, &v], real_fill).unwrap();
    // Fresh, uniquely owned destinations: a `clone()` shares the payload `Arc`
    // and would trip the unique-ownership check instead.
    let destination = || source.permute(&[2, 0], &[1, 3]).unwrap();

    let message = |error: tenet::prelude::Error| error.to_string();

    // Runtime mismatch precedes a rule mismatch.
    let other = Runtime::builder().build().unwrap();
    let z2 = Arc::new(tenet::core::ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(tenet::core::ZNFusionRule::new(3).unwrap());
    let z2_leg = GradedSpace::try_new_with_arc(Arc::clone(&z2), [(z2.irrep(0), 2)]).unwrap();
    let z3_leg = GradedSpace::try_new_with_arc(Arc::clone(&z3), [(z3.irrep(0), 2)]).unwrap();
    let zn_fill = |_: &_, indices: &[usize]| indices.iter().map(|&i| i as f64 + 1.0).sum::<f64>();
    let z2_source = TensorMap::from_block_fn(&runtime, [&z2_leg], [&z2_leg], zn_fill).unwrap();
    let mut foreign =
        TensorMap::from_block_fn(&other, [&z3_leg], [&z3_leg], |_, _| f64::NAN).unwrap();
    assert_eq!(
        z2_source
            .permute_overwrite_into(&mut foreign, &[0], &[1], 1.0)
            .unwrap_err(),
        tenet::prelude::Error::RuntimeMismatch
    );

    // Rule mismatch precedes a lazy-adjoint source.
    let mut z3_destination =
        TensorMap::from_block_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| f64::NAN).unwrap();
    assert_eq!(
        z2_source
            .adjoint()
            .unwrap()
            .permute_overwrite_into(&mut z3_destination, &[0], &[1], 1.0)
            .unwrap_err(),
        tenet::prelude::Error::RuleMismatch
    );

    // A lazy-adjoint source is rejected — not lowered onto its parent — and
    // is reported before a lazy-adjoint destination.
    let lazy_source = source.adjoint().unwrap();
    let mut lazy_destination = destination().adjoint().unwrap();
    assert_eq!(
        message(
            lazy_source
                .permute_overwrite_into(&mut lazy_destination, &[2, 0], &[1, 3], 1.0)
                .unwrap_err()
        ),
        "invalid argument: typed destination tree transform requires an ordinary \
         dense host source"
    );
    assert_eq!(
        message(
            source
                .permute_overwrite_into(&mut lazy_destination, &[2, 0], &[1, 3], 1.0)
                .unwrap_err()
        ),
        "invalid argument: destination must use ordinary dense host storage"
    );

    // The alias check precedes the operation build, so malformed axes do not
    // mask it.
    let mut alias = source.clone();
    assert_eq!(
        message(
            source
                .permute_overwrite_into(&mut alias, &[0, 0], &[1, 3], 1.0)
                .unwrap_err()
        ),
        "invalid argument: destination storage must not alias an input"
    );

    // The space check precedes the unique-ownership check.
    let mut wrong_space = source.transpose().unwrap();
    let wrong_space_handle = wrong_space.clone();
    assert_eq!(
        message(
            source
                .permute_overwrite_into(&mut wrong_space, &[2, 0], &[1, 3], 1.0)
                .unwrap_err()
        ),
        "invalid argument: destination fusion space or block layout does not match \
         the operation result"
    );
    drop(wrong_space_handle);

    let mut shared = destination();
    let shared_handle = shared.clone();
    assert_eq!(
        message(
            source
                .permute_overwrite_into(&mut shared, &[2, 0], &[1, 3], 1.0)
                .unwrap_err()
        ),
        "invalid argument: destination storage must be uniquely owned"
    );
    drop(shared_handle);

    // `repartition_overwrite_into` onto a destination of another rank.
    let mut rank_three =
        TensorMap::from_block_fn(&runtime, [&v, &v], [&v], |_, _| f64::NAN).unwrap();
    assert_eq!(
        message(
            source
                .repartition_overwrite_into(&mut rank_three, 1.0)
                .unwrap_err()
        ),
        "invalid argument: repartition destination rank 3 does not match source rank 4"
    );
}

#[test]
fn host_overwrite_into_clears_a_poisoned_destination_and_zero_scales_to_zeros() {
    // The three Host destination semantics: Overwrite clears whatever the
    // destination held, `alpha == 0` writes zeros whatever the source holds
    // (#1438; the device still computes `0 * src` until its leaf aligns it),
    // and an identity axis list is still written rather than short circuited.
    let runtime = Runtime::builder().build().unwrap();
    let v = u1_leg();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&v, &v], [&v, &v], real_fill).unwrap();

    let mut poisoned = source.permute(&[2, 0], &[1, 3]).unwrap().scale(f64::NAN);
    assert!(poisoned.data().iter().all(|value| value.is_nan()));
    source
        .permute_overwrite_into(&mut poisoned, &[2, 0], &[1, 3], 1.0)
        .unwrap();
    assert!(
        poisoned.data().iter().all(|value| value.is_finite()),
        "Overwrite mode must clear every destination layout, inactive ones included"
    );
    assert_eq!(
        poisoned.data(),
        source.permute(&[2, 0], &[1, 3]).unwrap().data()
    );

    // No identity short circuit: `alpha * self` is written.
    let mut identity = source.scale(f64::NAN);
    source
        .permute_overwrite_into(&mut identity, &[0, 1], &[2, 3], -2.5)
        .unwrap();
    assert_eq!(identity.data(), source.scale(-2.5).data());
    let mut same_split = source.scale(f64::NAN);
    source
        .repartition_overwrite_into(&mut same_split, 2.0)
        .unwrap();
    assert_eq!(same_split.data(), source.scale(2.0).data());

    // `alpha == 0`, `-0.0` included, writes VectorInterface's
    // `scale(x, 0) = zero(x) * 0` (#1438): an exact zero at every position,
    // NaN sources included, as TensorKit's `permute!(tdst, tsrc, p, 0.0,
    // Zero())` does (observed: `0.0 + 0.0im` over a `NaN + 1.0im`
    // destination and an `Inf + 1.0im` source).
    let nan_source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&v, &v], [&v, &v], |trees, idx| {
            if idx.iter().sum::<usize>() % 3 == 0 {
                f64::NAN
            } else {
                real_fill(trees, idx)
            }
        })
        .unwrap();
    let permuted_source = nan_source.permute(&[1, 2], &[3, 0]).unwrap();
    assert!(
        !nan_positions(permuted_source.data()).is_empty(),
        "the fixture must carry NaNs through the permute"
    );
    for alpha in [0.0, -0.0] {
        let mut destination = nan_source.permute(&[1, 2], &[3, 0]).unwrap();
        nan_source
            .permute_overwrite_into(&mut destination, &[1, 2], &[3, 0], alpha)
            .unwrap();
        assert!(
            destination.data().iter().all(|value| *value == 0.0),
            "alpha = {alpha}: every position must be an exact zero"
        );
    }
}

// ---------------------------------------------------------------------------
// Device twist (#1330, G2b-t): the Host-side decisions its lowering rests on
// ---------------------------------------------------------------------------

/// Every twist of `$tensor` over `$cases` keeps an entry or negates it, and
/// does both at least once — the `±1` domain, written once so each provider
/// is checked identically. A macro rather than a generic function: the typed
/// `twist` bound names the admission mode, which no test needs to spell out.
macro_rules! assert_signs_only_over {
    ($what:expr, $tensor:expr, $cases:expr) => {
        for legs in $cases {
            for twisted in [
                $tensor.twist(legs).unwrap(),
                $tensor.twist_inverse(legs).unwrap(),
            ] {
                let mut kept = 0usize;
                let mut negated = 0usize;
                for (&before, &after) in $tensor.data().iter().zip(twisted.data()) {
                    if after == before {
                        kept += 1;
                    } else {
                        assert_eq!(
                            after, -before,
                            "{} twist {legs:?} scaled by something else",
                            $what
                        );
                        negated += 1;
                    }
                }
                assert!(negated > 0, "{} twist {legs:?} changed nothing", $what);
                assert!(
                    kept > 0,
                    "{} twist {legs:?} scaled every entry, so no unscaled block is covered",
                    $what
                );
            }
        }
    };
}

/// The device `twist` puts the per-block factor on the contraction
/// descriptor's own scale instead of uploading a factor table. That is only
/// sound because, for every provider the device impl admits (`Scalar = f64`),
/// a ribbon twist is a real sign and never zero — a zero would be rejected by
/// `cuda_region_axpby`, and would also let CUDA skip the source read.
///
/// This pins the value domain observationally, without a device: a Host twist
/// may keep an entry or negate it, and nothing else. It also pins that some
/// blocks really do keep their factor of one, which is why the device has to
/// move every block rather than only the scaled ones.
#[test]
fn a_fermionic_twist_only_ever_keeps_or_negates_an_entry() {
    let runtime = Runtime::builder().build().unwrap();

    // The fixtures are the device gate's own: fZ2 with a dual leg, and the two
    // product providers, whose factors are products of the two rules' own.
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::FermionParityFusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let dual = leg.try_dual().unwrap();
    let tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &dual], [&leg, &leg], real_fill).unwrap();
    assert_signs_only_over!(
        "fZ2",
        tensor,
        [
            &[0usize][..],
            &[1][..],
            &[3][..],
            &[0, 3][..],
            &[1, 2, 3][..]
        ]
    );

    // fZ2 x U(1).
    let rule = Arc::new(tenet::core::FermionParityFusionRule.product(U1FusionRule));
    let leg_u1 = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (
                tenet::core::product_sector(tenet::core::Z2Irrep::EVEN, U1Irrep::new(0)),
                2,
            ),
            (
                tenet::core::product_sector(tenet::core::Z2Irrep::ODD, U1Irrep::new(1)),
                1,
            ),
            (
                tenet::core::product_sector(tenet::core::Z2Irrep::EVEN, U1Irrep::new(2)),
                1,
            ),
        ],
    )
    .unwrap();
    let product_u1: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg_u1, &leg_u1], [&leg_u1, &leg_u1], real_fill)
            .unwrap();
    assert_signs_only_over!(
        "fZ2 x U(1)",
        product_u1,
        [&[1usize][..], &[2][..], &[0, 2][..]]
    );

    // fZ2 (x) SU(2): the same sign domain with non-Abelian degeneracies.
    let rule = Arc::new(tenet::core::FermionParityFusionRule.product(SU2FusionRule));
    let leg_su2 = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (
                tenet::core::product_sector(
                    tenet::core::Z2Irrep::EVEN,
                    SU2Irrep::from_twice_spin(0),
                ),
                2,
            ),
            (
                tenet::core::product_sector(
                    tenet::core::Z2Irrep::ODD,
                    SU2Irrep::from_twice_spin(1),
                ),
                1,
            ),
            (
                tenet::core::product_sector(
                    tenet::core::Z2Irrep::EVEN,
                    SU2Irrep::from_twice_spin(2),
                ),
                1,
            ),
        ],
    )
    .unwrap();
    let product_su2: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&leg_su2, &leg_su2],
        [&leg_su2, &leg_su2],
        real_fill,
    )
    .unwrap();
    assert_signs_only_over!(
        "fZ2 (x) SU(2)",
        product_su2,
        [&[0usize][..], &[3][..], &[1, 3][..]]
    );

    // Twisting *every* leg is the identity in value on a parity-conserving
    // block: the factors multiply to the block's total parity, which is even.
    // It is not a short circuit — the detection tests each leg's own factor,
    // not the product — so the device gate expects the ordinary per-block
    // work here, and this pins the value it must produce.
    for legs in [&[0usize, 1, 2, 3][..], &[0, 1, 2, 3, 0, 1, 2, 3][..]] {
        assert_eq!(tensor.twist(legs).unwrap().data(), tensor.data());
        assert_eq!(tensor.twist_inverse(legs).unwrap().data(), tensor.data());
    }

    // An inverse twist undoes a twist exactly, which is what lets the device
    // gate compare the two runs for equality rather than to a tolerance.
    let round_trip = tensor
        .twist(&[0, 2])
        .unwrap()
        .twist_inverse(&[0, 2])
        .unwrap();
    assert_eq!(round_trip.data(), tensor.data());
}

/// The zero-block fixture of the device gate. A space with no coupled sector
/// has no block, so `twist_is_identity_over_blocks` is vacuously true and the
/// call short-circuits to a clone on both Host and device: it proves the
/// degenerate space is handled, not that the zero-length upload path runs.
/// Reaching that path needs blocks with a zero extent, which is a device-only
/// distinction and is left untested.
#[test]
fn a_twist_of_a_space_with_no_coupled_sector_is_the_identity_short_circuit() {
    let runtime = Runtime::builder().build().unwrap();
    let even = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::FermionParityFusionRule),
        [(tenet::core::Z2Irrep::EVEN, 2)],
    )
    .unwrap();
    let odd = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::FermionParityFusionRule),
        [(tenet::core::Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&even], [&odd], real_fill).unwrap();
    assert!(empty.data().is_empty(), "the fixture must carry no element");
    assert!(empty.twist(&[0, 1]).unwrap().data().is_empty());
}
