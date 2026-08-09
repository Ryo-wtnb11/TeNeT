//! Timing for the spectral matrix functions and polar decomposition
//! (issues #46 / #51).
//!
//! `pinv` / `inv` / `exp` / `left_polar` / `right_polar` fold the (inverted /
//! mapped / singular) spectrum into a block-local scaling of a factor's bond
//! axis (TensorKit's `DiagonalTensorMap` `rmul!`) instead of materializing a
//! dense `rank x rank` diagonal and recomposing `V * D * U^H` through a full
//! block GEMM. This example times those user entry points on a representative
//! symmetric endomorphism (iTEBD-sized bond); the win grows with the per-block
//! bond rank, where the removed diagonal GEMM is O(rank^2 * deg).
//!
//! Run: `cargo run -p tenet --release --example matrix_fn_timing`

use std::hint::black_box;
use std::time::Instant;

use tenet::prelude::*;

fn time_it(label: &str, iters: usize, mut f: impl FnMut()) {
    // warm up (plan caches, first-touch allocations)
    f();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    println!(
        "  {label:<28} {iters:>6} iters  {:>10.3?}  ({:>8.1} us/iter)",
        elapsed,
        elapsed.as_secs_f64() * 1e6 / iters as f64
    );
}

macro_rules! bench {
    ($name:literal, $rule:ty, $v:expr, $iters:expr $(,)?) => {{
        let v = $v;
        let rt = Runtime::builder().build().unwrap();
        // Hermitian positive endomorphism t = a^H a so exp/inv/pinv are valid.
        let a = TensorMap::<$rule, f64>::rand_with_seed(&rt, [&v, &v], [&v, &v], 0xA11CE).unwrap();
        let t = a.adjoint().unwrap().compose(&a).unwrap();
        println!("{} (space dim {}):", $name, v.dim().unwrap());
        time_it("pinv(1e-12)", $iters, || {
            black_box(black_box(&t).pinv(1e-12).unwrap());
        });
        time_it("inv", $iters, || {
            black_box(black_box(&t).inv().unwrap());
        });
        time_it("exp", $iters, || {
            black_box(black_box(&t).exp().unwrap());
        });
        time_it("left_polar", $iters, || {
            black_box(black_box(&t).left_polar().unwrap());
        });
        time_it("right_polar", $iters, || {
            black_box(black_box(&t).right_polar().unwrap());
        });
    }};
}

fn main() {
    println!("matrix-function timing (release) — issue #46 baseline\n");
    // U(1): several charge sectors, degeneracy 8 -> per-block rank up to 8.
    bench!(
        "U1",
        U1FusionRule,
        GradedSpace::try_new(
            U1FusionRule,
            [
                (U1Irrep::new(-2), 4),
                (U1Irrep::new(-1), 8),
                (U1Irrep::new(0), 8),
                (U1Irrep::new(1), 8),
                (U1Irrep::new(2), 4),
            ]
        )
        .unwrap(),
        2_000,
    );
    // SU(2): quantum-dimension weighting exercises the recompose norm path.
    bench!(
        "SU2",
        SU2FusionRule,
        GradedSpace::try_new(
            SU2FusionRule,
            [
                (SU2Irrep::from_twice_spin(0), 6),
                (SU2Irrep::from_twice_spin(1), 6),
                (SU2Irrep::from_twice_spin(2), 4),
            ]
        )
        .unwrap(),
        2_000
    );
}
