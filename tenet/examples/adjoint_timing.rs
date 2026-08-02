//! Timing harness for the lazy adjoint (`TensorMap::adjoint`) and its contraction
//! fold. Run with `cargo run --release --example adjoint_timing`.
//!
//! Measures three things per dtype:
//!   * `adjoint()` construction alone — lazy, so this should be ~free (O(blocks)
//!     metadata, no data copy);
//!   * `adjoint().compose(b)` — the headline: the adjoint folds into the GEMM
//!     with no materialized conjugate-transpose, for both f64 (a pure transpose
//!     via `adjoint_view`) and c64 (a conjugate-transpose folded via the seam's
//!     conjugate flag, BLAS `op='C'`);
//!   * a plain `compose(b)` baseline, so the adjoint overhead is visible as the
//!     gap above it.
//!
//! To see the fold's effect as a before/after, run this example at this commit
//! (fold) and again with `tensor.rs` reverted to the materialize-only commit;
//! the `adjoint().compose` row is the one that moves.

use std::time::Instant;

use tenet::prelude::*;

fn bench(label: &str, iters: usize, mut f: impl FnMut()) {
    // Warm up (plan caches, first-touch allocations).
    for _ in 0..iters.min(50) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per = elapsed.as_secs_f64() * 1e6 / iters as f64;
    println!("  {label:<34} {iters} iters  {elapsed:>10.3?}  ({per:>9.2} us/iter)");
}

fn main() {
    let rt = Runtime::builder().build().unwrap();
    // A few sectors with non-trivial degeneracies so the coupled blocks are big
    // enough for the conjugate-transpose copy to matter.
    let v = GradedSpace::try_new_owned(
        U1FusionRule,
        [
            (U1Irrep::new(-1), 6),
            (U1Irrep::new(0), 10),
            (U1Irrep::new(1), 8),
            (U1Irrep::new(2), 4),
        ],
        false,
    )
    .unwrap();
    let iters = 4000;

    macro_rules! bench_dtype {
        ($dtype:ty, $label:literal) => {{
            println!("{} (endomorphism [v,v] <- [v,v], v dims 6/10/8/4):", $label);
            let a = TensorMap::<U1FusionRule, $dtype>::rand_with_seed(&rt, [&v, &v], [&v, &v], 1)
                .unwrap();
            let b = TensorMap::<U1FusionRule, $dtype>::rand_with_seed(&rt, [&v, &v], [&v, &v], 2)
                .unwrap();

            bench("adjoint() only (lazy)", iters, || {
                let _ = a.adjoint().unwrap();
            });
            bench("compose (baseline, no adjoint)", iters, || {
                let _ = a.compose(&b).unwrap();
            });
            bench("adjoint().compose(b)", iters, || {
                let _ = a.adjoint().unwrap().compose(&b).unwrap();
            });
            println!();
        }};
    }

    bench_dtype!(f64, "f64");
    bench_dtype!(Complex64, "c64");
}
