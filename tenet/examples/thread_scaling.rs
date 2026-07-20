//! Thread-scaling benchmark for warm cached contractions (issue #155).
//!
//! **What it measures.** Outer-thread throughput of a representative warm
//! rank-4 SU(2) `contract`, when N independent driver threads each churn on
//! THEIR OWN tensors (independent data, same rule + dtype). Three arms expose
//! where the ceiling is:
//!   (a) `shared`  — one `Runtime` handle cloned into every thread, standalone
//!       `Tensor::contract`. Every op takes the runtime's single coarse Mutex
//!       for the whole computation, so different threads (even different data)
//!       serialize. This is the suspected FLAT arm.
//!   (b) `perthread` — each thread builds its own `Runtime`. No shared lock;
//!       the expected ~linear arm, the throughput ceiling to compare against.
//!   (c) `network` — shared `Runtime`, but the `tensor!` cached-plan path,
//!       which clones a per-call workspace and holds the lock only briefly.
//!       The claimed already-parallel path.
//!
//! **Why a fixed CPU budget.** Scaling numbers only mean something if the
//! outer threads are the ONLY source of parallelism — otherwise inner BLAS /
//! rayon threads confound the arithmetic. So the harness pins the dense
//! backend to 1 thread (`.dense_threads(1)`, which also caps the global rayon
//! pool at 1) and expects `RAYON_NUM_THREADS=1` in the environment. Backend is
//! the default single-threaded faer GEMM (no `blas-*` feature). Every core the
//! run uses comes from an outer thread, nothing else.
//!
//! Output: CSV `arm,N,d,evals_per_sec,speedup_vs_N1` (median of REPEATS).
//!
//! Env knobs: `TENET_SCALING_NS` (default `1,2,4,8`), `TENET_SCALING_DS`
//! (default `8,16`), `TENET_SCALING_M` (iters/thread, default `400`),
//! `TENET_SCALING_REPEATS` (default `3`).
//!
//! Run: `RAYON_NUM_THREADS=1 cargo run --release --example thread_scaling`.

use std::hint::black_box;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use tenet::prelude::{Dtype, Runtime, Space, Tensor};
use tenet_network::tensor;

fn env_usizes(key: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(key) {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect(),
        Err(_) => default.to_vec(),
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// SU(2) sectors {0, 1/2, 1} (twice-spin 0,1,2) each with degeneracy `d`.
fn space(d: usize) -> Space {
    Space::su2([(0, d), (1, d), (2, d)]).unwrap()
}

/// Rank-4 endomorphism `[v,v] <- [v,v]` with a deterministic per-thread seed.
fn make_pair(rt: &Runtime, d: usize, seed: u64) -> (Tensor, Tensor) {
    let v = space(d);
    let a = Tensor::rand_with_seed(rt, Dtype::F64, [&v, &v], [&v, &v], seed).expect("lhs");
    let b = Tensor::rand_with_seed(rt, Dtype::F64, [&v, &v], [&v, &v], seed + 1).expect("rhs");
    (a, b)
}

/// Compose-form contract: `a[i,j;k,l] * b[k,l;m,n]` -> rank-4. lhs domain
/// axes (2,3) with rhs codomain axes (0,1).
fn contract_once(a: &Tensor, b: &Tensor) -> Tensor {
    a.contract(b, &[2, 3], &[0, 1]).expect("contract")
}

fn build_runtime() -> Runtime {
    // dense_threads(1): single-threaded faer GEMM AND caps the global rayon
    // pool at 1 (best-effort, once per process) — the fixed CPU budget.
    Runtime::builder()
        .dense_threads(1)
        .build()
        .expect("runtime")
}

/// Warm caches (plan cache, tree-transform replays, first-touch allocs) on the
/// given runtime, single-threaded, before any timed region.
fn warm(rt: &Runtime, d: usize) {
    let (a, b) = make_pair(rt, d, 1);
    for _ in 0..8 {
        black_box(contract_once(&a, &b));
        black_box(tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).expect("warm net"));
    }
}

/// Run `m` contractions per thread across `n` threads; returns evals/sec.
/// `body(t, m)` does its own (untimed) setup — building this thread's runtime
/// and tensors — then times ONLY its `m`-iteration compute loop and returns
/// that Duration. Throughput uses the slowest worker's loop time (the parallel
/// region's wall), so per-thread setup never pollutes the number and mutex
/// waiting in the loop (arm a) is captured exactly.
fn measure(
    n: usize,
    m: usize,
    body: impl Fn(usize, usize) -> Duration + Send + Sync + 'static,
) -> f64 {
    let body = Arc::new(body);
    let handles: Vec<_> = (0..n)
        .map(|t| {
            let body = Arc::clone(&body);
            thread::spawn(move || body(t, m))
        })
        .collect();
    let max = handles
        .into_iter()
        .map(|h| h.join().expect("thread"))
        .max()
        .unwrap()
        .as_secs_f64();
    (n * m) as f64 / max
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    let ns = env_usizes("TENET_SCALING_NS", &[1, 2, 4, 8]);
    let ds = env_usizes("TENET_SCALING_DS", &[8, 16]);
    let m = env_usize("TENET_SCALING_M", 400);
    let repeats = env_usize("TENET_SCALING_REPEATS", 3);

    eprintln!(
        "thread_scaling: NS={ns:?} DS={ds:?} M={m} repeats={repeats} \
         RAYON_NUM_THREADS={:?}",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(unset)".into())
    );
    println!("arm,N,d,evals_per_sec,speedup_vs_N1");

    // Arm (a): shared Runtime, standalone Tensor::contract.
    // Arm (c): shared Runtime, cached network path.
    let shared = build_runtime();
    for &d in &ds {
        warm(&shared, d);
    }

    for (arm, uses_shared) in [("shared", true), ("perthread", false), ("network", true)] {
        for &d in &ds {
            let mut base = 0.0f64;
            for (i, &n) in ns.iter().enumerate() {
                let reps: Vec<f64> = (0..repeats)
                    .map(|_| {
                        let shared_rt = shared.clone();
                        measure(n, m, move |t, m| {
                            // Untimed setup: own runtime (perthread) or the
                            // shared handle; own tensors (distinct seeds) either
                            // way. perthread warms its fresh runtime here too.
                            let rt = if uses_shared {
                                shared_rt.clone()
                            } else {
                                build_runtime()
                            };
                            let seed = 1000 + (t as u64) * 2;
                            let (a, b) = make_pair(&rt, d, seed);
                            if !uses_shared {
                                warm(&rt, d);
                            }
                            let start = Instant::now();
                            for _ in 0..m {
                                if arm == "network" {
                                    black_box(
                                        tensor!([i, j; p, q] = a[i, j; k, l] * b[k, l; p, q])
                                            .expect("net"),
                                    );
                                } else {
                                    black_box(contract_once(&a, &b));
                                }
                            }
                            start.elapsed()
                        })
                    })
                    .collect();
                let eps = median(reps);
                if i == 0 {
                    base = eps;
                }
                println!("{arm},{n},{d},{eps:.1},{:.3}", eps / base);
            }
        }
    }
}
