//! Where the reductions accumulate, and what that is allowed to change (#1315).
//!
//! `inner`, `norm`, `tr` and the compact full trace sum over a whole coupled
//! region, so they accumulate in `WideScalar::Wide` — the payload type itself
//! for `f64`/`Complex64`, double precision for `f32`/`Complex32`. Two things
//! need pinning, and neither is reachable through the widened-input oracle in
//! `single_precision_base.rs`, whose tolerance is far larger than the
//! difference between a wide and a narrow accumulator.
//!
//! 1. **Double precision is bit-for-bit unchanged.** The reference bit
//!    patterns below were captured by running this file's `f64`/`Complex64`
//!    cases against `origin/main` at `89b1cde6` — before the accumulators were
//!    widened — in a separate worktree, and they are asserted here as exact
//!    `to_bits()` equalities. Every storage form is covered, because the four
//!    accumulators live on four different paths: dense (`coupled_region_inner`
//!    / `weighted_trace`), compact diagonal (`compact_inner`,
//!    `tr_multiplicity_free`, `trace_pairs_multiplicity_free`) and lazy
//!    adjoint (`tenet_tensors::oriented_fusion_inner`).
//!
//! 2. **Single precision really does accumulate wide.** The fixture is one
//!    entry of `8192.0f32` and the rest `1.0f32`. `8192^2` is `2^26`, whose
//!    `f32` spacing is 8, so a narrow accumulator swallows every `1.0` that
//!    follows and returns exactly `2^26`, while a double accumulator returns
//!    `2^26 + (n - 1)`. With `n - 1 > 8` the two differ *after* narrowing to
//!    `f32`, so even `inner`, which returns the payload type, tells them
//!    apart. Each assertion is paired with `assert_ne!` against the value a
//!    narrow accumulator would produce, computed in this file, so the test
//!    cannot pass by both paths agreeing.
//!
//! A third case pins range rather than precision: entries of `1e20f32` square
//! to `1e40`, far past the `f32` maximum of about `3.4e38`. `norm` must be
//! finite in every storage form.

use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{Complex32, Complex64, GradedSpace, Runtime, SectorSpectrum, TensorMap};

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

/// One U(1) charge with degeneracy 4: a single 4x4 dense block, a 4-entry
/// compact spectrum, and `dim(c) == 1` so the weight cannot mask a difference.
fn leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap()
}

/// Degeneracy 32: 1024 dense entries and a 32-entry compact spectrum, both
/// long enough for the swallowed `1.0`s below to add up past one `f32` step of
/// `2^26`.
fn wide_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 32)]).unwrap()
}

// --- 1. double precision is bit-for-bit unchanged ----------------------------

/// Reference bits captured on `origin/main` 89b1cde6 (pre-widening).
const F64_REFERENCE: [(&str, u64); 6] = [
    ("dense norm", 0x4043_56cd_ebc9_b5e2),
    ("dense inner", 0x4065_8000_0000_0000),
    ("dense tr", 0x4008_0000_0000_0000),
    ("compact norm", 0x4014_7e70_54af_0989),
    ("compact tr", 0x4021_0000_0000_0000),
    ("lazy adjoint inner", 0x4065_8000_0000_0000),
];

fn f64_measurements() -> Vec<(&'static str, u64)> {
    let runtime = runtime();
    let leg = leg();
    let dense: TensorMap<U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            (index[0] * 4 + index[1] + 1) as f64
        })
        .unwrap();
    let other: TensorMap<U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            (index[0] as f64) - (index[1] as f64) * 0.5
        })
        .unwrap();
    let compact: TensorMap<U1FusionRule, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![1.0, 3.0, 0.5, 4.0],
        }],
    )
    .unwrap();

    vec![
        ("dense norm", dense.norm().unwrap().to_bits()),
        ("dense inner", dense.inner(&other).unwrap().to_bits()),
        ("dense tr", other.tr().unwrap().to_bits()),
        ("compact norm", compact.norm().unwrap().to_bits()),
        ("compact tr", compact.tr().unwrap().to_bits()),
        (
            "lazy adjoint inner",
            dense
                .adjoint()
                .unwrap()
                .inner(&other.adjoint().unwrap())
                .unwrap()
                .to_bits(),
        ),
    ]
}

/// Reference bits captured on `origin/main` 89b1cde6 (pre-widening), as
/// `(real, imaginary)`.
const C64_REFERENCE: [(&str, u64, u64); 4] = [
    ("dense norm", 0x4044_b02b_4f7c_0a88, 0),
    ("dense inner", 0x4070_0000_0000_0000, 0x406f_8000_0000_0000),
    ("dense tr", 0x4008_0000_0000_0000, 0x4018_0000_0000_0000),
    (
        "lazy adjoint inner",
        0x4070_0000_0000_0000,
        0xc06f_8000_0000_0000,
    ),
];

fn c64_measurements() -> Vec<(&'static str, u64, u64)> {
    let runtime = runtime();
    let leg = leg();
    let dense: TensorMap<U1FusionRule, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            Complex64::new((index[0] * 4 + index[1] + 1) as f64, (index[1] + 2) as f64)
        })
        .unwrap();
    let other: TensorMap<U1FusionRule, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            Complex64::new(index[0] as f64 - index[1] as f64 * 0.5, index[0] as f64)
        })
        .unwrap();

    let bits = |value: Complex64| (value.re.to_bits(), value.im.to_bits());
    let (inner_re, inner_im) = bits(dense.inner(&other).unwrap());
    let (tr_re, tr_im) = bits(other.tr().unwrap());
    let (lazy_re, lazy_im) = bits(
        dense
            .adjoint()
            .unwrap()
            .inner(&other.adjoint().unwrap())
            .unwrap(),
    );
    vec![
        ("dense norm", dense.norm().unwrap().to_bits(), 0),
        ("dense inner", inner_re, inner_im),
        ("dense tr", tr_re, tr_im),
        ("lazy adjoint inner", lazy_re, lazy_im),
    ]
}

#[test]
fn double_precision_reductions_are_bit_for_bit_unchanged() {
    for ((what, got), (expected_what, expected)) in f64_measurements().iter().zip(&F64_REFERENCE) {
        assert_eq!(what, expected_what);
        assert_eq!(
            got, expected,
            "{what}: {got:#018x} against the pre-widening reference {expected:#018x}"
        );
    }
    for ((what, re, im), (expected_what, expected_re, expected_im)) in
        c64_measurements().iter().zip(&C64_REFERENCE)
    {
        assert_eq!(what, expected_what);
        assert_eq!(
            (re, im),
            (expected_re, expected_im),
            "{what}: ({re:#018x}, {im:#018x}) against the pre-widening reference"
        );
    }
}

// --- 2. single precision accumulates wide ------------------------------------

/// The one large entry; `8192^2 == 2^26`, where the `f32` spacing is 8.
const PEAK: f32 = 8192.0;

/// `sum |x|^2` as a narrow accumulator computes it: `2^26` first, then `n - 1`
/// additions of `1.0` that each round straight back to `2^26`.
fn narrow_sum(n: usize) -> f32 {
    let mut sum = PEAK * PEAK;
    for _ in 1..n {
        sum += 1.0f32 * 1.0f32;
    }
    sum
}

/// `sum |x|^2` as a double accumulator computes it: exact.
fn wide_sum(n: usize) -> f64 {
    f64::from(PEAK) * f64::from(PEAK) + (n - 1) as f64
}

#[test]
fn single_precision_reductions_accumulate_in_double() {
    let runtime = runtime();
    let leg = wide_leg();
    let dense_entries = 32 * 32;
    let dense: TensorMap<U1FusionRule, f32> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            if index == [0, 0] {
                PEAK
            } else {
                1.0
            }
        })
        .unwrap();
    assert_eq!(dense.data().len(), dense_entries);

    let expected = wide_sum(dense_entries);
    assert_ne!(
        expected as f32,
        narrow_sum(dense_entries),
        "the fixture must distinguish a wide from a narrow accumulator in f32"
    );

    // `inner(self, self)` is `sum |x|^2` unweighted (dim(c) == 1).
    assert_eq!(dense.inner(&dense).unwrap(), expected as f32, "dense inner");
    // `norm` returns f64, so it shows the accumulator without narrowing.
    assert_eq!(dense.norm().unwrap(), expected.sqrt(), "dense norm");
    assert_ne!(
        dense.norm().unwrap(),
        f64::from(narrow_sum(dense_entries)).sqrt()
    );

    // The lazy-adjoint operands run `oriented_fusion_inner`, a different
    // kernel with its own per-block and cross-block accumulators.
    let lazy = dense.adjoint().unwrap();
    assert_eq!(
        lazy.inner(&lazy).unwrap(),
        expected as f32,
        "lazy adjoint inner must accumulate like the dense form"
    );

    // Compact diagonal storage: `compact_inner`, a third accumulator.
    let mut values = vec![1.0f32; 32];
    values[0] = PEAK;
    let compact: TensorMap<U1FusionRule, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values,
        }],
    )
    .unwrap();
    let compact_expected = wide_sum(32);
    assert_ne!(compact_expected as f32, narrow_sum(32));
    assert_eq!(
        compact.inner(&compact).unwrap(),
        compact_expected as f32,
        "compact inner"
    );
    assert_eq!(
        compact.norm().unwrap(),
        compact_expected.sqrt(),
        "compact norm"
    );

    // Compact `tr` and the compact full trace sum the values themselves, which
    // are exact here; what they must not do is disagree with each other.
    let trace = PEAK + 31.0;
    assert_eq!(compact.tr().unwrap(), trace, "compact tr");
    assert_eq!(
        compact.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap(),
        trace,
        "compact full trace"
    );
}

#[test]
fn complex32_reductions_accumulate_in_double() {
    let runtime = runtime();
    let leg = wide_leg();
    let dense_entries = 32 * 32;
    let dense: TensorMap<U1FusionRule, Complex32> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            Complex32::new(if index == [0, 0] { PEAK } else { 1.0 }, 0.0)
        })
        .unwrap();
    let expected = Complex32::new(wide_sum(dense_entries) as f32, 0.0);
    assert_ne!(expected.re, narrow_sum(dense_entries));
    assert_eq!(dense.inner(&dense).unwrap(), expected, "dense inner");
    assert_eq!(
        dense.norm().unwrap(),
        wide_sum(dense_entries).sqrt(),
        "dense norm"
    );
    let lazy = dense.adjoint().unwrap();
    assert_eq!(lazy.inner(&lazy).unwrap(), expected, "lazy adjoint inner");
}

#[test]
fn single_precision_norm_stays_finite_where_a_narrow_accumulator_overflows() {
    // 1e20^2 = 1e40, past the f32 maximum of ~3.4e38: a narrow accumulator
    // saturates to +inf and the storage form would decide the answer.
    let runtime = runtime();
    let leg = leg();
    let value = 1e20f32;
    assert!(
        (value * value).is_infinite(),
        "the fixture must overflow an f32 accumulator"
    );

    let dense: TensorMap<U1FusionRule, f32> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| value).unwrap();
    let compact: TensorMap<U1FusionRule, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![value; 4],
        }],
    )
    .unwrap();

    for (what, norm) in [
        ("dense", dense.norm().unwrap()),
        ("compact", compact.norm().unwrap()),
        ("lazy adjoint", dense.adjoint().unwrap().norm().unwrap()),
    ] {
        assert!(norm.is_finite(), "{what} norm overflowed: {norm}");
    }
    // `inner` returns the payload type, and `4 * 4 * 1e40` genuinely exceeds
    // the `f32` range, so `+inf` there is the correct narrowing of a finite
    // wide accumulator rather than an accumulator that saturated. `norm`
    // above is what distinguishes the two, since it returns `f64`.
    let lazy = dense.adjoint().unwrap();
    for (what, value) in [
        ("dense inner", dense.inner(&dense).unwrap()),
        ("lazy adjoint inner", lazy.inner(&lazy).unwrap()),
    ] {
        assert!(value.is_infinite(), "{what}: {value}");
    }
}
