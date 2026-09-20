//! CPU-only half of the device tree-transform evidence (issue #1304).
//!
//! The device suite (`cuda_tree_transform.rs`, `#[ignore]`) asserts that the
//! device executor reproduces the explicit-index oracle in `common`. That is
//! only evidence if the oracle itself is right, so this file — which needs no
//! GPU and therefore runs in CI — pins the oracle against the *host* executor
//! replaying the same compiled structures. The two statements are independent:
//! the oracle walks the fixture's own block metadata, the host executor walks
//! the structure's baked fused layouts.

mod common;

use common::{
    all_fixtures, inactive_destination_layouts, interleaved_multi_block, many_distinct_signatures,
    Fixture, TestScalar,
};
use num_complex::Complex64;
use tenet_operations::{
    tree_transform_structure_overwrite_with_strided_kernel_raw,
    tree_transform_structure_with_strided_kernel_raw, StridedHostKernelAdapter,
    TreeTransformWorkspace,
};

/// Host replay of `fixture` in either destination mode, on host slices.
fn host_replay<T>(fixture: &Fixture, source: &[T], destination: &[T], overwrite: bool) -> Vec<T>
where
    T: TestScalar
        + tenet_operations::TreeTransformScalar
        + tenet_operations::RecouplingCoefficientAction<f64>
        + tenet_operations::DenseBlockScalar,
{
    let structure = fixture.compile();
    let dst_structure = fixture.dst_structure();
    let src_structure = fixture.src_structure();
    let mut kernels = StridedHostKernelAdapter::default();
    let mut workspace = TreeTransformWorkspace::<T>::default();
    let mut data = destination.to_vec();
    let one = T::from_parts(1.0, 0.0);
    if overwrite {
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut kernels,
            &mut workspace,
            &structure,
            &dst_structure,
            &src_structure,
            &mut data,
            source,
            one,
        )
        .unwrap();
    } else {
        tree_transform_structure_with_strided_kernel_raw(
            &mut kernels,
            &mut workspace,
            &structure,
            &dst_structure,
            &src_structure,
            &mut data,
            source,
            one,
            one,
        )
        .unwrap();
    }
    data
}

fn assert_close<T: TestScalar>(actual: &[T], expected: &[T], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            left.distance(*right) <= 1e-12,
            "{what}: element {index} is {left:?}, expected {right:?}"
        );
    }
}

fn check_fixture<T>(fixture: &Fixture)
where
    T: TestScalar
        + tenet_operations::TreeTransformScalar
        + tenet_operations::RecouplingCoefficientAction<f64>
        + tenet_operations::DenseBlockScalar,
{
    let source = fixture.source::<T>();
    let destination: Vec<T> = (0..fixture.dst_len())
        .map(|index| T::from_parts(-3.0 - index as f64, 0.5))
        .collect();
    for overwrite in [true, false] {
        let expected = fixture.expected(&source, &destination, overwrite);
        let host = host_replay(fixture, &source, &destination, overwrite);
        assert_close(
            &host,
            &expected,
            &format!("{} / {} / overwrite = {overwrite}", fixture.name, T::NAME,),
        );
    }
}

#[test]
fn the_explicit_index_oracle_agrees_with_host_replay_for_every_fixture() {
    // What: every fixture the device suite replays means, element for element,
    // what the host executor does with the same compiled structure — in both
    // destination modes and both payload dtypes.
    for fixture in all_fixtures() {
        check_fixture::<f64>(&fixture);
        check_fixture::<Complex64>(&fixture);
    }
    check_fixture::<f64>(&many_distinct_signatures(70));
}

#[test]
fn overwrite_cleans_inactive_layouts_and_accumulate_leaves_them_alone() {
    // What: the two destination modes really differ on the fixture the device
    // suite uses for the poisoned-destination contract, so that test is not
    // passing by accident.
    let fixture = inactive_destination_layouts();
    let source = fixture.source::<f64>();
    let poisoned: Vec<f64> = (0..fixture.dst_len()).map(|_| f64::NAN).collect();

    let overwritten = host_replay(&fixture, &source, &poisoned, true);
    assert!(
        overwritten.iter().all(|value| !value.is_nan()),
        "overwrite must clean a NaN destination: {overwritten:?}"
    );
    assert_close(
        &overwritten,
        &fixture.expected(&source, &poisoned, true),
        "inactive_destination_layouts / overwrite",
    );

    let accumulated = host_replay(&fixture, &source, &poisoned, false);
    assert!(
        accumulated.iter().any(|value| value.is_nan()),
        "accumulation into NaN stays NaN: {accumulated:?}"
    );
}

#[test]
fn the_oracle_is_sensitive_to_the_coefficient_and_to_the_permutation() {
    // Negative control: an oracle that ignored the coefficient or the axis map
    // would still pass the agreement test above, so prove it does neither.
    let fixture = interleaved_multi_block();
    let source = fixture.source::<f64>();
    let destination = vec![0.0; fixture.dst_len()];
    let expected = fixture.expected(&source, &destination, true);

    let mut flipped = fixture.clone();
    flipped.pairs[0].coefficient = 1.0;
    assert_ne!(
        flipped.expected(&source, &destination, true),
        expected,
        "the oracle must depend on the coefficient"
    );

    let mut unpermuted = fixture.clone();
    unpermuted.pairs[0].axes = vec![0, 1];
    assert_ne!(
        unpermuted.expected(&source, &destination, true),
        expected,
        "the oracle must depend on the axis map"
    );
}
