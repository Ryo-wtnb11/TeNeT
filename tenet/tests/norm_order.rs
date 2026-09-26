//! `TensorMap::norm(p)` is the single norm entry (#1544).
//!
//! It replaced `norm()`, `norm_p(p)` and `norm_inf()`. The golden bit patterns
//! below were emitted by those three methods at `origin/main` cf19fa17, so
//! each arm is pinned bit-for-bit to the reduction it replaced:
//! `norm(2.0)` to the old `norm()` (overflow rescaling included, see `huge`
//! and `tiny`), `norm(f64::INFINITY)` to the old `norm_inf()`, and every other
//! exponent to the old `norm_p(p)`. TensorKit agreement of the same values is
//! asserted in `tk_norm_p.rs` and `tk_norm_range.rs`.

use std::sync::Arc;

use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::prelude::{Complex32, Complex64, Error, Runtime};
use tenet::typed::{GradedSpace, SectorSpectrum, TensorMap};

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

fn real_fill(indices: &[usize]) -> f64 {
    1.0 + 0.5 * indices[0] as f64 - 0.25 * indices[1] as f64
}

fn complex_fill(indices: &[usize]) -> Complex64 {
    Complex64::new(
        real_fill(indices),
        0.5 + 0.125 * indices[0] as f64 + 0.375 * indices[1] as f64,
    )
}

fn u1() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let rule = Arc::new(U1FusionRule);
    let pairs = |entries: [(i32, usize); 3]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&rule),
            entries.map(|(charge, deg)| (U1Irrep::new(charge), deg)),
        )
        .unwrap()
    };
    (
        pairs([(-1, 2), (0, 3), (1, 4)]),
        pairs([(-1, 5), (0, 1), (1, 2)]),
    )
}

fn su2() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let rule = Arc::new(SU2FusionRule);
    let pairs = |entries: [(usize, usize); 3]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&rule),
            entries.map(|(twice_spin, deg)| (SU2Irrep::from_twice_spin(twice_spin), deg)),
        )
        .unwrap()
    };
    (
        pairs([(0, 2), (1, 3), (2, 4)]),
        pairs([(0, 5), (1, 1), (2, 2)]),
    )
}

fn compact_su2(rt: &Runtime) -> TensorMap<SU2FusionRule, f64> {
    let (bond, _) = su2();
    let spectrum = |twice_spin, values: &[f64]| SectorSpectrum {
        sector: SU2Irrep::from_twice_spin(twice_spin),
        values: values.to_vec(),
    };
    TensorMap::diagonal(
        rt,
        &bond,
        [
            spectrum(0, &[3.25, -0.5]),
            spectrum(1, &[1.75, 0.0, -2.125]),
            spectrum(2, &[0.375, 4.5, -1.0, 0.625]),
        ],
    )
    .unwrap()
}

/// Exponents of the golden rows, in column order.
const EXPONENTS: [f64; 5] = [2.0, f64::INFINITY, 1.0, 3.0, 0.5];

/// `[norm(), norm_inf(), norm_p(1), norm_p(3), norm_p(0.5)]` at cf19fa17.
#[rustfmt::skip]
const GOLDEN: [(&str, [u64; 5]); 9] = [
    ("u1_f64", [0x4018d1c0be7f20ac, 0x4004000000000000, 0x4039000000000000, 0x401052557b4bcdc2, 0x407d4f0c460c8e65]),
    ("u1_c64", [0x40204a535d95fbc8, 0x4005308af161f4a5, 0x40420b48518ebb90, 0x40143d951de00fc4, 0x4087458a2847fef7]),
    ("su2_f64", [0x4023502e64db5456, 0x4004000000000000, 0x404bc00000000000, 0x40165ae90b39d16f, 0x40a010e47e6135e6]),
    ("su2_c64", [0x40274630b96b8bd8, 0x4005308af161f4a5, 0x4051b59da4e7f12a, 0x401a0e633b5258e8, 0x40a5b1b21a04a538]),
    ("su2_c32", [0x40274630b96b8bd8, 0x4005308af161f4a5, 0x4051b59da4e7f12a, 0x401a0e633b5258e8, 0x40a5b1b21a04a538]),
    ("huge", [0x69a93b2d98b14ad9, 0x698a20df0dcd3af0, 0x69d220678b2cc74a, 0x699d3486888a73ef, 0x6a24fd2a72ec208c]),
    ("tiny", [0x1698f05b18cca0af, 0x1680383813b5fbff, 0x16bb9faa12d422d0, 0x168efc7d7000c5fe, 0x1701d033aafa02c9]),
    ("compact", [0x40231c8c3cc94cf6, 0x4012000000000000, 0x403f000000000000, 0x401bf66fac709513, 0x407d4754f90d31c9]),
    ("adjoint", [0x40274630b96b8bd8, 0x4005308af161f4a5, 0x4051b59da4e7f12a, 0x401a0e633b5258e8, 0x40a5b1b21a04a538]),
];

fn assert_golden(name: &str, norm: impl Fn(f64) -> Result<f64, Error>) {
    let (_, bits) = GOLDEN.iter().find(|(row, _)| *row == name).unwrap();
    for (&p, &expected) in EXPONENTS.iter().zip(bits) {
        let actual = norm(p).unwrap();
        assert_eq!(
            actual.to_bits(),
            expected,
            "{name} norm({p}) = {actual:e}, expected {:e}",
            f64::from_bits(expected)
        );
    }
}

#[test]
fn every_arm_is_bit_identical_to_the_method_it_replaced() {
    let rt = runtime();
    let (uv, uw) = u1();
    let (sv, sw) = su2();
    let u1_f64: TensorMap<U1FusionRule, f64> =
        TensorMap::from_block_fn(&rt, [&uv], [&uw], |_, i| real_fill(i)).unwrap();
    let u1_c64: TensorMap<U1FusionRule, Complex64> =
        TensorMap::from_block_fn(&rt, [&uv], [&uw], |_, i| complex_fill(i)).unwrap();
    let su2_f64: TensorMap<SU2FusionRule, f64> =
        TensorMap::from_block_fn(&rt, [&sv], [&sw], |_, i| real_fill(i)).unwrap();
    let su2_c64: TensorMap<SU2FusionRule, Complex64> =
        TensorMap::from_block_fn(&rt, [&sv], [&sw], |_, i| complex_fill(i)).unwrap();
    let su2_c32: TensorMap<SU2FusionRule, Complex32> =
        TensorMap::from_block_fn(&rt, [&sv], [&sw], |_, i| {
            let z = complex_fill(i);
            Complex32::new(z.re as f32, z.im as f32)
        })
        .unwrap();
    // Squares leave the f64 range in both directions, so `p = 2` and the
    // power sums take their rescaling passes.
    let huge: TensorMap<SU2FusionRule, f64> =
        TensorMap::from_block_fn(&rt, [&sv], [&sw], |_, i| 1e200 * real_fill(i)).unwrap();
    let tiny: TensorMap<U1FusionRule, Complex64> =
        TensorMap::from_block_fn(&rt, [&uv], [&uw], |_, i| 1e-200 * complex_fill(i)).unwrap();
    let compact = compact_su2(&rt);
    let adjoint = su2_c64.adjoint().unwrap();

    assert_golden("u1_f64", |p| u1_f64.norm(p));
    assert_golden("u1_c64", |p| u1_c64.norm(p));
    assert_golden("su2_f64", |p| su2_f64.norm(p));
    assert_golden("su2_c64", |p| su2_c64.norm(p));
    assert_golden("su2_c32", |p| su2_c32.norm(p));
    assert_golden("huge", |p| huge.norm(p));
    assert_golden("tiny", |p| tiny.norm(p));
    assert_golden("compact", |p| compact.norm(p));
    assert_golden("adjoint", |p| adjoint.norm(p));
}

#[test]
fn infinity_arm_propagates_nan_and_is_positive_zero_without_entries() {
    let rt = runtime();
    let (uv, uw) = u1();
    let poisoned: TensorMap<U1FusionRule, Complex64> =
        TensorMap::from_block_fn(&rt, [&uv], [&uw], |trees, i| {
            if *trees.coupled() == U1Irrep::new(1) && i == [0, 0] {
                Complex64::new(1.0, f64::NAN)
            } else {
                complex_fill(i)
            }
        })
        .unwrap();
    assert!(poisoned.norm(f64::INFINITY).unwrap().is_nan());
    assert!(poisoned
        .adjoint()
        .unwrap()
        .norm(f64::INFINITY)
        .unwrap()
        .is_nan());

    let only_zero =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let only_one =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(1), 3)]).unwrap();
    let empty: TensorMap<U1FusionRule, f64> =
        TensorMap::zeros(&rt, [&only_zero], [&only_one]).unwrap();
    let zeros: TensorMap<U1FusionRule, f64> = TensorMap::zeros(&rt, [&uv], [&uw]).unwrap();
    for (what, tensor) in [("empty", &empty), ("all-zero", &zeros)] {
        let value = tensor.norm(f64::INFINITY).unwrap();
        assert_eq!(
            value.to_bits(),
            0.0_f64.to_bits(),
            "{what}: {value:e} is not +0.0"
        );
    }
}

#[test]
fn invalid_exponents_are_typed_errors() {
    let rt = runtime();
    let (uv, uw) = u1();
    let tensor: TensorMap<U1FusionRule, f64> =
        TensorMap::from_block_fn(&rt, [&uv], [&uw], |_, i| real_fill(i)).unwrap();
    for p in [f64::NAN, f64::NEG_INFINITY, 0.0, -0.0, -1.0] {
        assert!(
            matches!(tensor.norm(p), Err(Error::InvalidArgument(_))),
            "norm({p})"
        );
    }
}

/// The device reduction exists only for `p = 2`. Its value is the square root
/// of the device self inner product, exactly the body of the removed
/// argument-free device `norm()`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_norm_is_the_frobenius_arm_and_rejects_other_exponents() {
    let rt = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let (uv, uw) = u1();
    let (sv, sw) = su2();
    let u1_f64: TensorMap<U1FusionRule, f64> =
        TensorMap::from_block_fn(&rt, [&uv], [&uw], |_, i| real_fill(i)).unwrap();
    let su2_c64: TensorMap<SU2FusionRule, Complex64> =
        TensorMap::from_block_fn(&rt, [&sv], [&sw], |_, i| complex_fill(i)).unwrap();

    let real = u1_f64.to_cuda().unwrap();
    let complex = su2_c64.to_cuda().unwrap();
    let real_norm = real.norm(2.0).unwrap();
    let complex_norm = complex.norm(2.0).unwrap();
    assert_eq!(
        real_norm.to_bits(),
        real.inner(&real).unwrap().sqrt().to_bits()
    );
    assert_eq!(
        complex_norm.to_bits(),
        complex.inner(&complex).unwrap().re.sqrt().to_bits()
    );
    assert_eq!(
        complex.adjoint().unwrap().norm(2.0).unwrap().to_bits(),
        complex_norm.to_bits()
    );
    let host = f64::from_bits(GOLDEN[0].1[0]);
    assert!((real_norm - host).abs() <= 1e-13 * host);
    let host = f64::from_bits(GOLDEN[3].1[0]);
    assert!((complex_norm - host).abs() <= 1e-13 * host);

    for p in [f64::INFINITY, 1.0, 3.0, 0.5] {
        assert!(
            matches!(real.norm(p), Err(Error::UnsupportedOnDevice(_))),
            "device norm({p})"
        );
    }
    for p in [f64::NAN, f64::NEG_INFINITY, 0.0, -1.0] {
        assert!(
            matches!(real.norm(p), Err(Error::InvalidArgument(_))),
            "device norm({p})"
        );
    }
}
