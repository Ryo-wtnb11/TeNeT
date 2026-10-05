//! Standalone eager control for #1662. This file also compiles unchanged at
//! TeNeT main e887fb14. Run with `cargo test -p tenet-rs --release
//! --no-default-features --features cuda,blas-openblas --test
//! typed_cuda_eager_batch_baseline -- --ignored --nocapture --test-threads=1`.
//! Host elapsed times do not include an explicit device synchronization.
#![cfg(feature = "cuda")]

mod common;
#[path = "../../tests/support"]
mod support {
    use num_complex::{Complex32, Complex64};
    pub mod numerics;
}
use support::numerics;
#[macro_use]
#[allow(unused_macros)]
mod contract_cases;

use contract_cases::{candidate_core_probes, su2};
use std::hint::black_box;
use std::time::Instant;
use tenet::expert::cuda_transfer_stats;
use tenet::typed::{LinalgBackend, Runtime};

#[test]
#[ignore = "release-only A100 measurement"]
fn eager_c1_c2_per_member_baseline() {
    let runtime = Runtime::builder()
        .cuda(0)
        .dense_threads(1)
        .gemm_backend(LinalgBackend::Blas)
        .linalg_backend(LinalgBackend::Blas)
        .build()
        .unwrap();
    for (case, _) in candidate_core_probes::<_, f64>(&runtime, &su2()) {
        if !matches!(case.name, "C1" | "C2") {
            continue;
        }
        for count in [1, 2, 17] {
            let inputs: Vec<_> = (0..count)
                .map(|i| {
                    (
                        case.lhs.scale(1.0 + i as f64 / 8.0).to_cuda().unwrap(),
                        case.rhs.scale(1.0 - i as f64 / 32.0).to_cuda().unwrap(),
                    )
                })
                .collect();
            let run = || {
                for (lhs, rhs) in &inputs {
                    black_box(lhs.contract(rhs, &case.spec()).unwrap());
                }
            };
            let before = cuda_transfer_stats();
            let start = Instant::now();
            run();
            let cold = (start.elapsed(), before, cuda_transfer_stats());
            let mut warm = Vec::new();
            for _ in 0..11 {
                let before = cuda_transfer_stats();
                let start = Instant::now();
                run();
                warm.push((start.elapsed(), before, cuda_transfer_stats()));
            }
            warm.sort_by_key(|sample| sample.0);
            let median = warm[5];
            let cache = runtime.cuda_plan_cache_stats().unwrap().unwrap();
            let scalar_bytes = runtime
                .cuda_tree_transform_stats()
                .unwrap()
                .context_scalar_operand_bytes;
            for (phase, (elapsed, before, after)) in [("cold", cold), ("warm_median", median)] {
                eprintln!("{} B={count} f64 CUDA0 Faer1 eager_members {phase}: host_elapsed={elapsed:?}; h2d={}/{}B d2h={}/{}B device_allocs={} gemm_submissions={} copy_submissions={} plan_ledger={} plan_cache_bytes={} runtime_scalar_template_bytes={scalar_bytes}",
                    case.name, after.h2d_calls-before.h2d_calls, after.h2d_bytes-before.h2d_bytes,
                    after.d2h_calls-before.d2h_calls, after.d2h_bytes-before.d2h_bytes,
                    after.device_allocs-before.device_allocs, after.gemm_calls-before.gemm_calls,
                    after.copy_calls-before.copy_calls, cache.reserved_entries, cache.retained_bytes);
            }
        }
    }
}
