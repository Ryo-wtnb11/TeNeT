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
//!    patterns below were captured by running the `f64`/`Complex64` half of
//!    this file against `origin/main` at `89b1cde6` — before the accumulators
//!    were widened — in a detached worktree with its own target directory,
//!    and they are asserted here as exact `to_bits()` equalities.
//!
//!    The fixtures are deliberately **not** exactly representable. Dyadic
//!    entries make every partial sum exact, which would let a reordered or
//!    re-associated accumulation pass unnoticed; the values below come from
//!    `0.1`-based sequences, so the result depends on the summation order and
//!    the pins fail if it changes. They cover all three storage forms (dense,
//!    compact diagonal, lazy adjoint), the three reduction kinds (`norm`,
//!    `inner`, `tr`/full trace), an abelian provider with `dim(c) == 1` and a
//!    non-abelian one where `dim(c) == 2`, so the quantum-dimension weight is
//!    a real factor rather than a no-op. `checked_generic` adds the
//!    Checked-Generic admission mode behind `racah-generated`.
//!
//!    That the fixtures bite is visible in the reference table itself:
//!    `f64 su2 dense inner` and `f64 su2 lazy adjoint inner` differ in their
//!    last bit, because the two paths sum the same products in a different
//!    order. Dyadic entries would have made them identical.
//!
//!    One micro-difference is *not* observable here and is worth recording:
//!    applying the weight componentwise (`re * w`, `im * w`) rather than as
//!    the full complex product with `w + 0i` differs only when the real part
//!    of a block's contribution is a signed zero or non-finite, and a signed
//!    zero is then flattened by the `+0.0` the accumulator starts from — no
//!    public input reaches it. The implementation keeps the full complex
//!    product anyway, because it is the expression the loop used before the
//!    accumulator was widened. The `negative zero` fixtures pin what *is*
//!    observable: `-0.0` components entering the lazy-adjoint kernel, with
//!    the other component non-dyadic so the result is nonzero and the
//!    conjugation direction and multiplication order show in its bits.
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

use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
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

/// Three U(1) charges with unequal degeneracies: several coupled blocks of
/// different sizes, every `dim(c) == 1`.
fn u1_multi_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
}

/// Spin 0 and spin 1/2, so the reductions weight one block by `dim(c) == 1`
/// and the other by `dim(c) == 2`.
fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap()
}

/// A non-dyadic real: no finite binary expansion, so sums of these depend on
/// the order they are added in.
fn real_at(step: usize) -> f64 {
    (step as f64) * 0.1 - 0.35 + 0.7 / (step as f64 + 1.3)
}

/// A second, independent non-dyadic sequence for the imaginary part and for
/// the right-hand operand.
fn other_at(step: usize) -> f64 {
    0.9 / (step as f64 + 2.1) - (step as f64) * 0.03
}

/// Counts fill calls so the value depends on position in the payload, which is
/// what makes a reordered accumulation visible.
fn stepper() -> impl FnMut() -> usize {
    let mut step = 0usize;
    move || {
        step += 1;
        step
    }
}

fn bits(value: Complex64) -> (u64, u64) {
    (value.re.to_bits(), value.im.to_bits())
}

macro_rules! dense_reductions {
    ($out:expr, $prefix:literal, $runtime:expr, $leg:expr, $dtype:ty, $wide:expr) => {{
        let mut left = stepper();
        let a: TensorMap<_, $dtype> = TensorMap::from_block_fn($runtime, [$leg], [$leg], |_, _| {
            let step = left();
            <$dtype>::from_parts(real_at(step), other_at(step))
        })
        .unwrap();
        let mut right = stepper();
        let b: TensorMap<_, $dtype> = TensorMap::from_block_fn($runtime, [$leg], [$leg], |_, _| {
            let step = right();
            <$dtype>::from_parts(other_at(step + 5), real_at(step + 2))
        })
        .unwrap();

        $out.push((
            concat!($prefix, " dense norm"),
            a.norm().unwrap().to_bits(),
            0,
        ));
        let value = $wide(a.inner(&b).unwrap());
        $out.push((
            concat!($prefix, " dense inner"),
            bits(value).0,
            bits(value).1,
        ));
        let value = $wide(a.tr().unwrap());
        $out.push((concat!($prefix, " dense tr"), bits(value).0, bits(value).1));
        // Both operands lazy, then one lazy and one owned: the oriented kernel
        // is reached with a different conjugation pattern each way.
        let value = $wide(a.adjoint().unwrap().inner(&b.adjoint().unwrap()).unwrap());
        $out.push((
            concat!($prefix, " lazy adjoint inner"),
            bits(value).0,
            bits(value).1,
        ));
        let value = $wide(a.adjoint().unwrap().inner(&b).unwrap());
        $out.push((
            concat!($prefix, " lazy adjoint mixed inner"),
            bits(value).0,
            bits(value).1,
        ));
    }};
}

/// Compact diagonal storage has its own reductions. The Checked-Generic mode
/// rejects compact payloads, so this is separate from [`dense_reductions`].
macro_rules! compact_reductions {
    ($out:expr, $prefix:literal, $runtime:expr, $leg:expr, $dtype:ty, $wide:expr) => {{
        let mut diagonal = stepper();
        let mut spectrum = || {
            $leg.sectors()
                .unwrap()
                .into_iter()
                .zip($leg.degeneracies())
                .map(|(sector, &degeneracy)| SectorSpectrum {
                    sector,
                    values: (0..degeneracy)
                        .map(|_| {
                            let step = diagonal();
                            <$dtype>::from_parts(real_at(step), other_at(step))
                        })
                        .collect(),
                })
                .collect::<Vec<_>>()
        };
        let compact: TensorMap<_, $dtype> =
            TensorMap::diagonal($runtime, $leg, spectrum()).unwrap();
        let other: TensorMap<_, $dtype> = TensorMap::diagonal($runtime, $leg, spectrum()).unwrap();

        $out.push((
            concat!($prefix, " compact norm"),
            compact.norm().unwrap().to_bits(),
            0,
        ));
        let value = $wide(compact.inner(&other).unwrap());
        $out.push((
            concat!($prefix, " compact inner"),
            bits(value).0,
            bits(value).1,
        ));
        let value = $wide(compact.tr().unwrap());
        $out.push((
            concat!($prefix, " compact tr"),
            bits(value).0,
            bits(value).1,
        ));
        let value = $wide(compact.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap());
        $out.push((
            concat!($prefix, " compact full trace"),
            bits(value).0,
            bits(value).1,
        ));
    }};
}

/// Bridges the two payload dtypes into one `Complex64` for the bit dump.
trait FromParts: Copy {
    fn from_parts(re: f64, im: f64) -> Self;
}

impl FromParts for f64 {
    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }
}

impl FromParts for Complex64 {
    fn from_parts(re: f64, im: f64) -> Self {
        Self::new(re, im)
    }
}

fn double_precision_measurements() -> Vec<(&'static str, u64, u64)> {
    let runtime = runtime();
    let u1 = u1_multi_leg();
    let su2 = su2_leg();
    let real = |v: f64| Complex64::new(v, 0.0);
    let complex = |v| v;
    let mut out = Vec::new();
    dense_reductions!(out, "f64 u1", &runtime, &u1, f64, real);
    compact_reductions!(out, "f64 u1", &runtime, &u1, f64, real);
    dense_reductions!(out, "f64 su2", &runtime, &su2, f64, real);
    compact_reductions!(out, "f64 su2", &runtime, &su2, f64, real);
    dense_reductions!(out, "c64 u1", &runtime, &u1, Complex64, complex);
    compact_reductions!(out, "c64 u1", &runtime, &u1, Complex64, complex);
    dense_reductions!(out, "c64 su2", &runtime, &su2, Complex64, complex);
    compact_reductions!(out, "c64 su2", &runtime, &su2, Complex64, complex);

    // Negative zeros in the payload, carried through the lazy-adjoint kernel.
    // Each operand has `-0.0` in one component and a non-dyadic value in the
    // other, so the conjugation the oriented inner applies and the order it
    // multiplies in both reach the sign bits of a *nonzero* result rather
    // than a zero the accumulator would flatten.
    let mut step = stepper();
    let signed: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&u1], [&u1], |_, _| {
            Complex64::new(real_at(step()), -0.0)
        })
        .unwrap();
    let mut step = stepper();
    let probe: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&u1], [&u1], |_, _| {
            Complex64::new(-0.0, other_at(step()))
        })
        .unwrap();
    let (re, im) = bits(
        signed
            .adjoint()
            .unwrap()
            .inner(&probe.adjoint().unwrap())
            .unwrap(),
    );
    out.push(("c64 negative zero lazy adjoint inner", re, im));
    let (re, im) = bits(signed.adjoint().unwrap().inner(&probe).unwrap());
    out.push(("c64 negative zero lazy adjoint mixed inner", re, im));
    let (re, im) = bits(signed.tr().unwrap());
    out.push(("c64 negative zero tr", re, im));
    out
}

/// Reference bits measured on `origin/main` 89b1cde6, before the accumulators
/// were widened. Regenerated by the procedure in the module docs.
const DOUBLE_PRECISION_REFERENCE: &[(&str, u64, u64)] = &[
    ("f64 u1 dense norm", 0x4009a46ee0a3c378, 0x0000000000000000),
    ("f64 u1 dense inner", 0xc0149cf932c30ad2, 0x0000000000000000),
    ("f64 u1 dense tr", 0x401483669828948a, 0x0000000000000000),
    (
        "f64 u1 lazy adjoint inner",
        0xc0149cf932c30ad2,
        0x0000000000000000,
    ),
    (
        "f64 u1 lazy adjoint mixed inner",
        0xc0145084f921ba54,
        0x0000000000000000,
    ),
    (
        "f64 u1 compact norm",
        0x3fe4f6912c49514b,
        0x0000000000000000,
    ),
    (
        "f64 u1 compact inner",
        0x3ff59ee8758233ba,
        0x0000000000000000,
    ),
    ("f64 u1 compact tr", 0x3ff73e272cd4267c, 0x0000000000000000),
    (
        "f64 u1 compact full trace",
        0x3ff73e272cd4267c,
        0x0000000000000000,
    ),
    ("f64 su2 dense norm", 0x3ff2a43660ae06e6, 0x0000000000000000),
    (
        "f64 su2 dense inner",
        0xbfeee16b65101eac,
        0x0000000000000000,
    ),
    ("f64 su2 dense tr", 0x3ffcf26a090b0bd0, 0x0000000000000000),
    (
        "f64 su2 lazy adjoint inner",
        0xbfeee16b65101ead,
        0x0000000000000000,
    ),
    (
        "f64 su2 lazy adjoint mixed inner",
        0xbfee9e5e3e5ae3ca,
        0x0000000000000000,
    ),
    (
        "f64 su2 compact norm",
        0x3fd4179d71dbbad3,
        0x0000000000000000,
    ),
    (
        "f64 su2 compact inner",
        0x3fd4cbb5ef75eed6,
        0x0000000000000000,
    ),
    ("f64 su2 compact tr", 0x3fe69933a14cf00a, 0x0000000000000000),
    (
        "f64 su2 compact full trace",
        0x3fe69933a14cf00a,
        0x0000000000000000,
    ),
    ("c64 u1 dense norm", 0x400b1436751d0e19, 0x0000000000000000),
    ("c64 u1 dense inner", 0xc02207dea03e6db5, 0x402590c2dfbbc519),
    ("c64 u1 dense tr", 0x401483669828948a, 0xbff55b78b74c4d06),
    (
        "c64 u1 lazy adjoint inner",
        0xc02207dea03e6db6,
        0xc02590c2dfbbc519,
    ),
    (
        "c64 u1 lazy adjoint mixed inner",
        0xbff54a15a15d673e,
        0x402b6b92f903d113,
    ),
    (
        "c64 u1 compact norm",
        0x3fe7b1918ea8db75,
        0x0000000000000000,
    ),
    (
        "c64 u1 compact inner",
        0x3ff5344daa642833,
        0xbfe19bed10a31666,
    ),
    ("c64 u1 compact tr", 0x3ff73e272cd4267c, 0x3fd528dd1a088b94),
    (
        "c64 u1 compact full trace",
        0x3ff73e272cd4267c,
        0x3fd528dd1a088b94,
    ),
    ("c64 su2 dense norm", 0x3ff3da78a03ff248, 0x0000000000000000),
    (
        "c64 su2 dense inner",
        0xbff51b8e64afc627,
        0x3ffd3a26c6f0024d,
    ),
    ("c64 su2 dense tr", 0x3ffcf26a090b0bd0, 0xbfaeec6b609de2c8),
    (
        "c64 su2 lazy adjoint inner",
        0xbff51b8e64afc627,
        0xbffd3a26c6f0024d,
    ),
    (
        "c64 su2 lazy adjoint mixed inner",
        0xbfe3b811e22b3459,
        0x4000eee20408f566,
    ),
    (
        "c64 su2 compact norm",
        0x3fdd349baaf1174f,
        0x0000000000000000,
    ),
    (
        "c64 su2 compact inner",
        0x3fd1f195d239fe45,
        0xbfd40407a2450844,
    ),
    ("c64 su2 compact tr", 0x3fe69933a14cf00a, 0x3fe4bb40880fde0c),
    (
        "c64 su2 compact full trace",
        0x3fe69933a14cf00a,
        0x3fe4bb40880fde0c,
    ),
    (
        "c64 negative zero lazy adjoint inner",
        0x0000000000000000,
        0x400a334586b46571,
    ),
    (
        "c64 negative zero lazy adjoint mixed inner",
        0x0000000000000000,
        0xc00965365c1a5b8a,
    ),
    (
        "c64 negative zero tr",
        0x401483669828948a,
        0x0000000000000000,
    ),
];

#[test]
fn double_precision_reductions_are_bit_for_bit_unchanged() {
    let measured = double_precision_measurements();
    assert_eq!(
        measured.len(),
        DOUBLE_PRECISION_REFERENCE.len(),
        "the reference list must cover every measurement"
    );
    for ((what, re, im), (expected_what, expected_re, expected_im)) in
        measured.iter().zip(DOUBLE_PRECISION_REFERENCE)
    {
        assert_eq!(what, expected_what);
        assert_eq!(
            (re, im),
            (expected_re, expected_im),
            "{what}: ({re:#018x}, {im:#018x}) against the pre-widening reference \
             ({expected_re:#018x}, {expected_im:#018x})"
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

/// The Checked-Generic admission mode reaches the same reductions through its
/// own dispatch and its own fallible weight lookup, so it needs its own pins.
/// SU(3) with the adjoint irrep also carries outer multiplicity.
#[cfg(feature = "racah-generated")]
mod checked_generic {
    use super::*;
    use tenet::typed::SUNFusionRule;

    fn measurements() -> Vec<(&'static str, u64, u64)> {
        let runtime = runtime();
        let provider = Arc::new(SUNFusionRule::new(3).unwrap());
        let adjoint = vec![2i64, 2];
        let leg =
            GradedSpace::try_new_with_arc(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
        // Compact payloads are rejected by this admission mode
        // ("checked Generic reductions require dense payloads"), so only the
        // dense and lazy-adjoint reductions exist to pin here.
        let real = |v: f64| Complex64::new(v, 0.0);
        let complex = |v| v;
        let mut out = Vec::new();
        dense_reductions!(out, "f64 su3", &runtime, &leg, f64, real);
        dense_reductions!(out, "c64 su3", &runtime, &leg, Complex64, complex);
        out
    }

    /// Measured on `origin/main` 89b1cde6 with `--features racah-generated`.
    const REFERENCE: &[(&str, u64, u64)] = &[
        ("f64 su3 dense norm", 0x3ff315470a2c4936, 0x0000000000000000),
        (
            "f64 su3 dense inner",
            0xbffacf946aab6d12,
            0x0000000000000000,
        ),
        ("f64 su3 dense tr", 0x401988a19f4fedde, 0x0000000000000000),
        (
            "f64 su3 lazy adjoint inner",
            0xbffacf946aab6d12,
            0x0000000000000000,
        ),
        (
            "f64 su3 lazy adjoint mixed inner",
            0xbff9f09942323c09,
            0x0000000000000000,
        ),
        ("c64 su3 dense norm", 0x400051ec764bf090, 0x0000000000000000),
        (
            "c64 su3 dense inner",
            0x3fe89429f51145f8,
            0x4011b00fa69112af,
        ),
        ("c64 su3 dense tr", 0x401988a19f4fedde, 0x401f16da112a7b51),
        (
            "c64 su3 lazy adjoint inner",
            0x3fe89429f51145f8,
            0xc011b00fa69112af,
        ),
        (
            "c64 su3 lazy adjoint mixed inner",
            0xc010e22c8877339d,
            0x3ff530dcff4fa754,
        ),
    ];

    #[test]
    fn checked_generic_reductions_are_bit_for_bit_unchanged() {
        let measured = measurements();
        assert_eq!(measured.len(), REFERENCE.len());
        for ((what, re, im), (expected_what, expected_re, expected_im)) in
            measured.iter().zip(REFERENCE)
        {
            assert_eq!(what, expected_what);
            assert_eq!(
                (re, im),
                (expected_re, expected_im),
                "{what}: ({re:#018x}, {im:#018x}) against the pre-widening reference"
            );
        }
    }
}
