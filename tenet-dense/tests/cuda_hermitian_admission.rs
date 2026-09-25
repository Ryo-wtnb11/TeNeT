//! Batched device Hermitian admission (#1483): one call over many regions
//! makes the same per-region decisions as the per-region rule, and its
//! downloads do not grow with the region count.
//!
//! The decisions are asserted against hand-known truth values (exactly
//! Hermitian, asymmetric by a whole unit, zero, empty, non-finite, and
//! residuals just inside and far outside `64 * eps(real(D))`), at both ends
//! of each lane's normal range. This file holds a single test because it reads
//! the process-wide [`cuda_transfer_stats`] counters.
//!
//! Run with `cargo test -p tenet-dense --no-default-features --features \
//! cuda,cpu-faer --test cuda_hermitian_admission -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use num_complex::{Complex32, Complex64};
use tenet_dense::{
    cuda_hermitian_regions, cuda_is_hermitian_region, cuda_transfer_stats, CudaDenseContext,
    CudaDenseStorage, CudaScalar,
};

trait Payload: CudaScalar + Copy {
    const EPSILON: f64;
    /// The lane's smallest normal magnitude and `2^(MAX_EXP - 4)`.
    const TINY: f64;
    const HUGE: f64;
    fn narrow(value: Complex64) -> Self;
}

impl Payload for f32 {
    const EPSILON: f64 = f32::EPSILON as f64;
    const TINY: f64 = f32::MIN_POSITIVE as f64;
    const HUGE: f64 = 2.126_764_793_255_87e37;
    fn narrow(value: Complex64) -> Self {
        value.re as f32
    }
}

impl Payload for f64 {
    const EPSILON: f64 = f64::EPSILON;
    const TINY: f64 = f64::MIN_POSITIVE;
    const HUGE: f64 = 1.117_902_744_918_257e307;
    fn narrow(value: Complex64) -> Self {
        value.re
    }
}

impl Payload for Complex32 {
    const EPSILON: f64 = f32::EPSILON as f64;
    const TINY: f64 = f32::MIN_POSITIVE as f64;
    const HUGE: f64 = 2.126_764_793_255_87e37;
    fn narrow(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
}

impl Payload for Complex64 {
    const EPSILON: f64 = f64::EPSILON;
    const TINY: f64 = f64::MIN_POSITIVE;
    const HUGE: f64 = 1.117_902_744_918_257e307;
    fn narrow(value: Complex64) -> Self {
        value
    }
}

/// Hermitian by construction: `(row, col)` and `(col, row)` share a
/// magnitude and negate the imaginary part, scaled by an exact power of two.
fn hermitian(n: usize, scale: f64) -> Vec<Complex64> {
    (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            let (low, high) = (row.min(col), row.max(col));
            let re = 1.0 + 0.5 * ((low % 7) as f64) + 0.25 * ((high % 5) as f64);
            let im = match row.cmp(&col) {
                std::cmp::Ordering::Equal => 0.0,
                std::cmp::Ordering::Less => 0.5 + (low % 3) as f64,
                std::cmp::Ordering::Greater => -(0.5 + (low % 3) as f64),
            };
            Complex64::new(re * scale, im * scale)
        })
        .collect()
}

fn asymmetric(n: usize, scale: f64) -> Vec<Complex64> {
    let mut block = hermitian(n, scale);
    block[1] += Complex64::new(scale, 0.0);
    block
}

/// `[[1, delta], [0, 1]]` with `delta = epsilons * eps(real(D))`: admitted
/// below `128 * eps`, rejected above.
fn skewed<D: Payload>(epsilons: f64) -> Vec<Complex64> {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    vec![one, zero, Complex64::new(epsilons * D::EPSILON, 0.0), one]
}

fn poisoned(bad: f64) -> Vec<Complex64> {
    let mut block = hermitian(3, 1.0);
    block[0] = Complex64::new(bad, 0.0);
    block
}

/// The mixed fixture: every early return and every stage of the rule.
fn cases<D: Payload>() -> Vec<(Vec<Complex64>, bool)> {
    vec![
        (hermitian(3, 1.0), true),
        (asymmetric(4, 1.0), false),
        (vec![Complex64::new(0.0, 0.0); 4], true),
        (Vec::new(), true),
        (hermitian(4, D::HUGE), true),
        (asymmetric(4, D::HUGE), false),
        (hermitian(4, D::TINY), true),
        (asymmetric(4, D::TINY), false),
        (poisoned(f64::NAN), false),
        (poisoned(f64::INFINITY), false),
        (skewed::<D>(120.0), true),
        (skewed::<D>(4096.0), false),
        (hermitian(33, 1.0), true),
    ]
}

/// Packs `copies` repetitions of the fixture into one buffer, returning it,
/// the `(offset, n)` regions and the expected decisions.
fn packed<D: Payload>(copies: usize) -> (Vec<D>, Vec<(usize, usize)>, Vec<bool>) {
    let mut data = Vec::new();
    let mut regions = Vec::new();
    let mut expected = Vec::new();
    for _ in 0..copies {
        for (block, hermitian) in cases::<D>() {
            let n = (block.len() as f64).sqrt() as usize;
            regions.push((data.len(), n));
            data.extend(block.into_iter().map(D::narrow));
            expected.push(hermitian);
        }
    }
    (data, regions, expected)
}

fn case<D: Payload>(ctx: &mut CudaDenseContext, name: &str) {
    let mut downloads = Vec::new();
    for copies in [1, 3] {
        let (data, regions, expected) = packed::<D>(copies);
        let src = CudaDenseStorage::upload::<D>(ctx, &data).expect("upload");

        let before = cuda_transfer_stats().d2h_calls;
        let batched = cuda_hermitian_regions::<D>(ctx, &src, &regions).expect("batched");
        downloads.push(cuda_transfer_stats().d2h_calls - before);

        assert_eq!(batched, expected, "{name}: batched decisions x{copies}");
        for (&(offset, n), &decision) in regions.iter().zip(&batched) {
            assert_eq!(
                cuda_is_hermitian_region::<D>(ctx, &src, offset, n).expect("single"),
                decision,
                "{name}: region ({offset}, {n}) alone"
            );
        }
    }
    assert!(
        downloads[0] <= 3,
        "{name}: at most one download per stage, got {downloads:?}"
    );
    assert_eq!(
        downloads[0], downloads[1],
        "{name}: downloads must not grow with the region count"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn batched_admission_matches_the_per_region_rule_with_bounded_downloads() {
    let mut ctx = CudaDenseContext::new(0).expect("CUDA device 0");
    case::<f32>(&mut ctx, "f32");
    case::<f64>(&mut ctx, "f64");
    case::<Complex32>(&mut ctx, "Complex32");
    case::<Complex64>(&mut ctx, "Complex64");

    let empty = CudaDenseStorage::upload::<f64>(&ctx, &[0.0]).expect("upload");
    let before = cuda_transfer_stats().d2h_calls;
    assert_eq!(
        cuda_hermitian_regions::<f64>(&mut ctx, &empty, &[]).expect("no regions"),
        Vec::<bool>::new()
    );
    assert_eq!(
        cuda_transfer_stats().d2h_calls,
        before,
        "no regions, no downloads"
    );
}
