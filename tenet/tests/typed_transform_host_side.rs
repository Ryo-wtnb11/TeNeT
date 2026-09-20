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
use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
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
