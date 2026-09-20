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
    all_fixtures, expert_interleaved_destination, expert_interleaved_recoupling_destination,
    inactive_destination_layouts, interleaved_multi_block, many_distinct_signatures,
    mixed_single_and_multi, recoupling_fixtures, recoupling_non_symmetric_u,
    unit_coefficient_fixtures, Fixture, TestScalar,
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
    host_replay_scaled(
        fixture,
        source,
        destination,
        overwrite,
        T::from_parts(1.0, 0.0),
    )
}

/// The caller scales the device executor must reproduce: one, a scale that is
/// neither 1 nor -1, both signed zeros, and a genuinely complex one (which is
/// its real part on `f64`).
fn alphas<T: TestScalar>() -> Vec<T> {
    vec![
        T::from_parts(1.0, 0.0),
        T::from_parts(-2.5, 0.0),
        T::from_parts(0.0, 0.0),
        T::from_parts(-0.0, -0.0),
        T::from_parts(0.5, -1.25),
    ]
}

/// Host replay of `fixture` with the caller scale `alpha`.
fn host_replay_scaled<T>(
    fixture: &Fixture,
    source: &[T],
    destination: &[T],
    overwrite: bool,
    alpha: T,
) -> Vec<T>
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
            alpha,
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
            alpha,
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
        for alpha in alphas::<T>() {
            let expected = fixture.expected_scaled(&source, &destination, overwrite, alpha);
            let host = host_replay_scaled(fixture, &source, &destination, overwrite, alpha);
            assert_close(
                &host,
                &expected,
                &format!(
                    "{} / {} / overwrite = {overwrite} / alpha = {alpha:?}",
                    fixture.name,
                    T::NAME,
                ),
            );
        }
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
fn the_host_replays_the_layout_the_device_reports_as_unsupported() {
    // What: the fixture the device rejection test uses is a *legal* transform
    // the host executes, so that rejection is a device capability boundary and
    // not a malformed fixture.
    let fixture = expert_interleaved_destination();
    let source = fixture.source::<f64>();
    let destination = vec![0.0; fixture.dst_len()];

    let host = host_replay(&fixture, &source, &destination, true);

    assert_close(
        &host,
        &fixture.expected(&source, &destination, true),
        "expert_interleaved_destination / overwrite",
    );

    // The same, for the recoupling fixture whose *scatter* destination is the
    // inexpressible layout.
    let fixture = expert_interleaved_recoupling_destination();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];

    assert_close(
        &host_replay(&fixture, &source, &destination, true),
        &fixture.expected(&source, &destination, true),
        "expert_interleaved_recoupling_destination / overwrite",
    );
}

#[test]
fn unit_coefficient_fixtures_exist_and_are_exact_on_host() {
    // What: the device suite asserts bit-exactness for these, so there must be
    // some, and the host side must itself be exact.
    let fixtures = unit_coefficient_fixtures();
    assert!(!fixtures.is_empty(), "no unit-coefficient fixture");
    for fixture in fixtures {
        let source = fixture.source::<f64>();
        let destination = vec![0.0; fixture.dst_len()];
        assert_eq!(
            host_replay(&fixture, &source, &destination, true),
            fixture.expected(&source, &destination, true),
            "{}",
            fixture.name
        );
    }
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
fn the_recoupling_oracle_agrees_with_host_replay() {
    // What: the pack -> U^T -> scatter pipeline the device lowers means, element
    // for element, the plain weighted sum `dst[d] = sum_s U[d][s] * src[s]` the
    // oracle computes — in both destination modes and both payload dtypes, for
    // a non-symmetric square U, a rectangular U, a conjugated source, and a
    // structure mixing Single and Multi blocks.
    for fixture in recoupling_fixtures() {
        check_fixture::<f64>(&fixture);
        check_fixture::<Complex64>(&fixture);
    }
}

#[test]
fn the_recoupling_oracle_distinguishes_u_from_its_transpose() {
    // Negative control for the GEMM orientation: the host applies `U^T` on the
    // right (`mul!(dst, src, transpose(U))`). If the oracle's `U[d][s]` and the
    // host's agreed only up to a transpose, every agreement test above would
    // pass with the orientation reversed — so prove the fixture's own U is
    // asymmetric enough to tell the two apart, and that the *host* picks the
    // one the oracle does.
    let fixture = recoupling_non_symmetric_u();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let expected = fixture.expected(&source, &destination, true);

    let mut transposed = fixture.clone();
    let group = &mut transposed.groups[0];
    let rows = group.dst_blocks.len();
    let columns = group.src_blocks.len();
    let original = group.u.clone();
    for row in 0..rows {
        for column in 0..columns {
            group.u[row * columns + column] = original[column * rows + row];
        }
    }
    assert_ne!(
        transposed.expected(&source, &destination, true),
        expected,
        "the fixture's U must not be symmetric"
    );

    assert_close(
        &host_replay(&fixture, &source, &destination, true),
        &expected,
        "recoupling_non_symmetric_u / overwrite",
    );
    // And the host replaying the *transposed* fixture is the transposed result,
    // not the original one: the orientation is carried end to end.
    assert_close(
        &host_replay(&transposed, &source, &destination, true),
        &transposed.expected(&source, &destination, true),
        "transposed U / overwrite",
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

#[test]
fn the_caller_scale_multiplies_the_scatter_once_and_not_the_pack() {
    // Negative control for where alpha enters a recoupling group: the host
    // applies it at the scatter alone (`alpha * (U x)`). An implementation that
    // also scaled the packed columns would compute `alpha^2 * (U x)`, which the
    // oracle below expresses by folding alpha into U as well. The two must
    // disagree, and the host must land on the single-application one.
    let fixture = mixed_single_and_multi();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let alpha = -2.5_f64;

    let mut doubled = fixture.clone();
    for group in &mut doubled.groups {
        for entry in &mut group.u {
            *entry *= alpha;
        }
    }
    for pair in &mut doubled.pairs {
        pair.coefficient *= alpha;
    }

    for overwrite in [true, false] {
        let once = fixture.expected_scaled(&source, &destination, overwrite, alpha);
        let twice = doubled.expected_scaled(&source, &destination, overwrite, alpha);
        assert_ne!(
            once, twice,
            "alpha applied twice must change the result / overwrite = {overwrite}"
        );
        assert_close(
            &host_replay_scaled(&fixture, &source, &destination, overwrite, alpha),
            &once,
            &format!("mixed_single_and_multi / alpha once / overwrite = {overwrite}"),
        );
    }
}

#[test]
fn a_zero_caller_scale_multiplies_rather_than_skipping_the_source() {
    // What: the host has no alpha short circuit, so a zero scale propagates a
    // NaN source into every written element and leaves the zero fills of
    // Overwrite exact. This is the contract the device reproduces with a zero
    // 1x1 operand instead of a zero descriptor scale; here it is pinned on the
    // host, which is what the device is compared against.
    let fixture = mixed_single_and_multi();
    let poisoned = vec![f64::NAN; fixture.src_len()];
    let destination = vec![0.0_f64; fixture.dst_len()];

    for alpha in [0.0_f64, -0.0_f64] {
        let host = host_replay_scaled(&fixture, &poisoned, &destination, true, alpha);
        assert!(
            host.iter().any(|value| value.is_nan()),
            "a zero scale must still multiply the source: {host:?}"
        );
        let oracle = fixture.expected_scaled(&poisoned, &destination, true, alpha);
        for (index, (left, right)) in host.iter().zip(&oracle).enumerate() {
            assert_eq!(
                left.is_nan(),
                right.is_nan(),
                "element {index}: host {left} vs oracle {right}"
            );
            assert!(left.is_nan() || left == right, "element {index}");
        }
        // The inactive destination layout is an exact zero whatever the scale.
        let clean = host_replay_scaled(
            &fixture,
            &fixture.source::<f64>(),
            &destination,
            true,
            alpha,
        );
        assert!(
            clean.iter().all(|value| *value == 0.0),
            "a zero scale over a finite source writes zeros: {clean:?}"
        );
    }
}
