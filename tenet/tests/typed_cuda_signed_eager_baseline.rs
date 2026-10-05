//! Standalone eager signed control for #1664. Compiles unchanged at
//! f418b97a. Host elapsed is unsynchronized submission time.
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

use contract_cases::{fermion_u1, fill, Case};
use std::{hint::black_box, time::Instant};
use tenet::expert::cuda_transfer_stats;
use tenet::typed::{LinalgBackend, Runtime, TensorMap};

#[test]
#[ignore = "release-only A100 measurement"]
fn signed_eager_per_member_baseline() {
    let runtime = Runtime::builder()
        .cuda(0)
        .dense_threads(1)
        .gemm_backend(LinalgBackend::Blas)
        .linalg_backend(LinalgBackend::Blas)
        .build()
        .unwrap();
    let space = fermion_u1();
    let dual = space.try_dual().unwrap();
    let a = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&space], [&dual], fill(201)).unwrap();
    let b = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&dual], [&space], fill(202)).unwrap();
    for swapped in [false, true] {
        let case = Case {
            name: if swapped {
                "signed swapped"
            } else {
                "signed direct"
            },
            lhs: if swapped { b.clone() } else { a.clone() },
            rhs: if swapped { a.clone() } else { b.clone() },
            lhs_axes: vec![usize::from(!swapped)],
            rhs_axes: vec![usize::from(swapped)],
            output_axes: if swapped { vec![1, 0] } else { vec![0, 1] },
            dense: false,
        };
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
            for (phase, (elapsed, before, after)) in [("cold", cold), ("warm_median", warm[5])] {
                eprintln!("{} B={count} f64 CUDA0 Faer1 eager {phase}: unsynchronized_host_elapsed={elapsed:?}; h2d={}/{}B d2h={}/{}B device_allocs={} gemm_submissions={} copy_submissions={}",
                    case.name, after.h2d_calls-before.h2d_calls, after.h2d_bytes-before.h2d_bytes,
                    after.d2h_calls-before.d2h_calls, after.d2h_bytes-before.d2h_bytes,
                    after.device_allocs-before.device_allocs, after.gemm_calls-before.gemm_calls,
                    after.copy_calls-before.copy_calls);
            }
        }
    }
}
