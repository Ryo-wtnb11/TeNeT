//! Real-device gates for `twist` / `twist_inverse` on CUDA tensors
//! (issue #1330, G2b-t).
//!
//! A twist is not a tree transform: the Host builds a per-block ribbon-twist
//! factor at call time and scales a copy, so the evidence chain is not the
//! transform executor's. It is:
//!
//! 1. the value domain of the factor — `±1` for every provider this device
//!    impl admits — is gated without a device in `typed_transform_host_side.rs`,
//!    which ordinary CI runs;
//! 2. the lowering is gated here, by running the same public call on the Host
//!    and on the device; the Host side carries its own TensorKit and
//!    hand-computed oracles (`typed_facade.rs`, `semantic_suite.rs`,
//!    `checked_generic_twist.rs`);
//! 3. the transfer, allocation and rejection contracts are in
//!    `typed_cuda_transform_contracts.rs`, which owns the process-wide
//!    counters.
//!
//! Every fermionic fixture asserts non-vacuity: at least one block really is
//! negated, so a device path that copied its source would fail.
//!
//! Run with `cargo test -p tenet-rs --features cuda,cpu-faer --test \
//! typed_cuda_twist -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use num_complex::Complex64;
use tenet::core::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::prelude::{Runtime, TensorScalar};
use tenet::typed::{GradedSpace, TensorMap};

trait Payload: TensorScalar + Copy + PartialEq + std::fmt::Debug {
    fn negated(self) -> Self;
}

impl Payload for f64 {
    fn negated(self) -> Self {
        -self
    }
}

impl Payload for Complex64 {
    fn negated(self) -> Self {
        -self
    }
}

/// Runs one twist on the Host and on the device and asserts the device answer
/// is the Host answer exactly: every admitted factor is `±1`, so a device
/// result that differs in a single bit differs semantically.
macro_rules! device_matches_host {
    ($what:expr, $host:expr, |$t:ident| $body:expr) => {{
        let host_source = &$host;
        let device_source = host_source.to_cuda().unwrap();
        let expected = {
            let $t = host_source;
            $body
        }
        .unwrap();
        let actual = {
            let $t = &device_source;
            $body
        }
        .unwrap()
        .to_host()
        .unwrap();
        let what: &str = $what;
        assert_eq!(actual.data(), expected.data(), "{what}: payload");
        assert_eq!(actual.rank(), expected.rank(), "{what}: rank");
        assert_eq!(
            actual.codomain_rank(),
            expected.codomain_rank(),
            "{what}: split"
        );
        actual
    }};
}

/// Non-vacuity: the twist negated at least one entry, and negated *nothing
/// else* — the `±1` value domain the descriptor-scale lowering relies on.
fn assert_signs_only<D: Payload>(source: &[D], twisted: &[D], what: &str) {
    assert_eq!(source.len(), twisted.len(), "{what}: length");
    let mut negated = 0usize;
    for (index, (&before, &after)) in source.iter().zip(twisted).enumerate() {
        if after == before {
            continue;
        }
        assert_eq!(
            after,
            before.negated(),
            "{what}: element {index} is neither kept nor negated"
        );
        negated += 1;
    }
    assert!(
        negated > 0,
        "{what}: no entry changed sign, so the fixture proves nothing"
    );
}

fn runtime() -> Runtime {
    Runtime::builder().cuda(0).build().unwrap()
}

fn real_fill(_trees: &tenet::typed::BlockFusionTrees<impl std::fmt::Debug>, idx: &[usize]) -> f64 {
    let mut value = 0.37;
    for (axis, &index) in idx.iter().enumerate() {
        value = value * 1.7 + (index as f64 + 1.0) * (axis as f64 + 2.0);
    }
    value
}

fn complex_fill(
    trees: &tenet::typed::BlockFusionTrees<impl std::fmt::Debug>,
    idx: &[usize],
) -> Complex64 {
    let re = real_fill(trees, idx);
    Complex64::new(re, -0.5 * re + 0.25)
}

fn fz2_leg(dual: bool) -> GradedSpace<FermionParityFusionRule> {
    let space = GradedSpace::try_new_with_arc(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    if dual {
        space.try_dual().unwrap()
    } else {
        space
    }
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

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_matches_the_host_on_fermion_parity() {
    let runtime = runtime();
    let leg = fz2_leg(false);
    let dual = fz2_leg(true);
    // A dual leg in the codomain and a plain one in the domain, so the leg
    // indices below reach both sides and both dualities.
    let real: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &dual], [&leg, &leg], real_fill).unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &dual], [&leg, &leg], complex_fill).unwrap();
    assert!(real.block_count() >= 2, "multi-block fixture");

    // Single codomain leg, single dual codomain leg, single domain leg.
    for legs in [&[0usize][..], &[1][..], &[3][..]] {
        let what = format!("fZ2/f64 twist {legs:?}");
        let twisted = device_matches_host!(&what, real, |t| t.twist(legs));
        assert_signs_only(real.data(), twisted.data(), &what);
    }
    // Multi-axis, across the split and repeating both sides.
    for legs in [&[0usize, 3][..], &[1, 2, 3][..], &[0, 1, 1, 3][..]] {
        let what = format!("fZ2/f64 twist {legs:?}");
        let twisted = device_matches_host!(&what, real, |t| t.twist(legs));
        assert_signs_only(real.data(), twisted.data(), &what);
    }
    let what = "fZ2/c64 twist [1, 2]";
    let twisted = device_matches_host!(what, complex, |t| t.twist(&[1, 2]));
    assert_signs_only(complex.data(), twisted.data(), what);

    // `twist_inverse` is the conjugate factor, which is the same `±1` here —
    // gated against the Host rather than assumed.
    let what = "fZ2/f64 twist_inverse [0, 3]";
    let inverted = device_matches_host!(what, real, |t| t.twist_inverse(&[0, 3]));
    assert_signs_only(real.data(), inverted.data(), what);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_matches_the_host_for_product_providers() {
    let runtime = runtime();

    // fZ2 x U(1): fermionic signs on top of a charge grading.
    let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1),
        ],
    )
    .unwrap();
    let host: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], real_fill).unwrap();
    for legs in [&[1usize][..], &[2][..], &[0, 2][..]] {
        let what = format!("fZ2xU1 twist {legs:?}");
        let twisted = device_matches_host!(&what, host, |t| t.twist(legs));
        assert_signs_only(host.data(), twisted.data(), &what);
    }

    // fZ2 (x) SU(2): fermionic signs with non-Abelian degeneracies.
    let rule = Arc::new(FermionParityFusionRule.product(SU2FusionRule));
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                2,
            ),
            (
                product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(1)),
                1,
            ),
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                1,
            ),
        ],
    )
    .unwrap();
    let host: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], complex_fill).unwrap();
    for legs in [&[0usize][..], &[3][..], &[1, 3][..]] {
        let what = format!("fZ2xSU2 twist {legs:?}");
        let twisted = device_matches_host!(&what, host, |t| t.twist(legs));
        assert_signs_only(host.data(), twisted.data(), &what);
    }
    let what = "fZ2xSU2 twist_inverse [1, 3]";
    let inverted = device_matches_host!(what, host, |t| t.twist_inverse(&[1, 3]));
    assert_signs_only(host.data(), inverted.data(), what);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_round_trips_bitwise_and_short_circuits() {
    let runtime = runtime();
    let leg = fz2_leg(false);
    let host: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], complex_fill).unwrap();
    let device = host.to_cuda().unwrap();

    // twist ∘ twist_inverse == id, bitwise: both factors are exactly ±1.
    let round_trip = device
        .twist(&[0, 2, 3])
        .unwrap()
        .twist_inverse(&[0, 2, 3])
        .unwrap()
        .to_host()
        .unwrap();
    assert_eq!(round_trip.data(), host.data(), "twist round trip is exact");
    let twisted = device.twist(&[0, 2, 3]).unwrap().to_host().unwrap();
    assert_signs_only(host.data(), twisted.data(), "twist [0, 2, 3]");

    // An empty leg list is a clone on both sides.
    let empty = device.twist(&[]).unwrap().to_host().unwrap();
    assert_eq!(empty.data(), host.data(), "empty twist is a clone");

    // A bosonic provider twists by 1 on every block: a clone, never a scale.
    let u1 = u1_leg();
    let bosonic: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    let bosonic_device = bosonic.to_cuda().unwrap();
    for legs in [&[0usize][..], &[1, 3][..]] {
        assert_eq!(
            bosonic_device
                .twist(legs)
                .unwrap()
                .to_host()
                .unwrap()
                .data(),
            bosonic.data(),
            "bosonic twist {legs:?} must be a clone"
        );
        assert_eq!(
            bosonic_device
                .twist_inverse(legs)
                .unwrap()
                .to_host()
                .unwrap()
                .data(),
            bosonic.data(),
            "bosonic twist_inverse {legs:?} must be a clone"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_on_a_lazy_adjoint_matches_the_host() {
    let runtime = runtime();
    let leg = fz2_leg(false);
    let dual = fz2_leg(true);
    let host: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &dual], [&leg], complex_fill).unwrap();
    let device = host.to_cuda().unwrap();

    for legs in [&[0usize][..], &[2][..], &[0, 1, 2][..]] {
        let expected = host.adjoint().unwrap().twist(legs).unwrap();
        let actual = device
            .adjoint()
            .unwrap()
            .twist(legs)
            .unwrap()
            .to_host()
            .unwrap();
        assert_eq!(actual.data(), expected.data(), "adjoint twist {legs:?}");
        assert_eq!(
            actual.rank(),
            expected.rank(),
            "adjoint twist {legs:?} rank"
        );
        let inverse_expected = host.adjoint().unwrap().twist_inverse(legs).unwrap();
        let inverse_actual = device
            .adjoint()
            .unwrap()
            .twist_inverse(legs)
            .unwrap()
            .to_host()
            .unwrap();
        assert_eq!(
            inverse_actual.data(),
            inverse_expected.data(),
            "adjoint twist_inverse {legs:?}"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_handles_an_empty_tensor_and_rejects_a_leg_past_the_rank() {
    let runtime = runtime();
    // No coupled sector: zero blocks, zero elements, and still a real call.
    let even =
        GradedSpace::try_new_with_arc(Arc::new(FermionParityFusionRule), [(Z2Irrep::EVEN, 2)])
            .unwrap();
    let odd = GradedSpace::try_new_with_arc(Arc::new(FermionParityFusionRule), [(Z2Irrep::ODD, 1)])
        .unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&even], [&odd], real_fill).unwrap();
    assert!(empty.data().is_empty(), "empty fixture");
    let device = empty.to_cuda().unwrap();
    assert!(device
        .twist(&[0, 1])
        .unwrap()
        .to_host()
        .unwrap()
        .data()
        .is_empty());

    // The range check precedes everything, on both sides, with the Host text.
    let leg = fz2_leg(false);
    let host: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], real_fill).unwrap();
    let device = host.to_cuda().unwrap();
    assert_eq!(
        device.twist(&[2]).unwrap_err().to_string(),
        host.twist(&[2]).unwrap_err().to_string(),
    );
    assert_eq!(
        device.twist_inverse(&[7]).unwrap_err().to_string(),
        host.twist_inverse(&[7]).unwrap_err().to_string(),
    );
}
