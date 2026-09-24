//! #1291 victim-latency probe, kept here uncompiled: copy it to
//! `tenet-dense/examples/gemm_partition_victim.rs` of the revision under test
//! and build with `cargo build --release -p tenet-dense --example
//! gemm_partition_victim`.
//!
//! An aggressor thread issues one op-bearing (`Adjoint`, `Identity`) c64 batch
//! GEMM phase with a fixed run partition back to back. The main thread, the
//! victim, issues a short unrelated Tenferro call (a 4x4 f64 `matmul_into` on
//! its own executor) at spaced intervals and records each call's latency.
//! Both use default threads. Output: one CSV row per (pattern, n) with the
//! victim's latency quantiles alone and under the aggressor, the aggressor's
//! phase time alone and under the victim, and the TeNeT session/admission
//! counts of one phase.
//!
//! Usage: `gemm_partition_victim [samples]` (default 4000).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use num_complex::Complex64;
use tenet_dense::{
    cpu_session_stats, DefaultDenseExecutor, DenseExecutor, DenseGemmBatchJob, DenseRead,
    DenseScalar, DenseView, DenseViewMut, DenseWrite, MatrixOp,
};

struct Phase {
    jobs: Vec<DenseGemmBatchJob>,
    runs: Vec<usize>,
    lhs: Vec<Complex64>,
    rhs: Vec<Complex64>,
    dst: Vec<Complex64>,
}

/// A run of length >= 2 is `len` equal-shape, constant-stride jobs; a run of
/// length 1 is a singleton whose shape differs from its neighbours.
fn phase(pattern: &[usize], n: usize) -> Phase {
    let mut jobs = Vec::new();
    let (mut l, mut r, mut d) = (0usize, 0usize, 0usize);
    for (index, &len) in pattern.iter().enumerate() {
        let (rows, contracted, cols) = if len >= 2 {
            (n, n, n)
        } else {
            (n + 1 + index % 3, n - 1, n + 2)
        };
        for _ in 0..len {
            jobs.push(DenseGemmBatchJob {
                dst_offset: d,
                lhs_offset: l,
                rhs_offset: r,
                rows,
                contracted,
                cols,
            });
            l += rows * contracted;
            r += contracted * cols;
            d += rows * cols;
        }
    }
    let value = |i: usize| Complex64::new(((i * 7 + 3) % 13) as f64 * 0.1, ((i * 5) % 7) as f64 * 0.1);
    Phase {
        jobs,
        runs: pattern.to_vec(),
        lhs: (0..l).map(value).collect(),
        rhs: (0..r).map(|i| value(i + 5)).collect(),
        dst: vec![Complex64::new(0.0, 0.0); d],
    }
}

fn run_phase(executor: &mut DefaultDenseExecutor, p: &mut Phase) {
    let s = [1usize];
    let (ll, rl, dl) = (p.lhs.len(), p.rhs.len(), p.dst.len());
    executor
        .matmul_batch_axpby_with_ops_into(
            DenseWrite::C64(DenseViewMut::new(&mut p.dst, &[dl], &s, 0).unwrap()),
            DenseRead::C64(DenseView::new(&p.lhs, &[ll], &s, 0).unwrap()),
            DenseRead::C64(DenseView::new(&p.rhs, &[rl], &s, 0).unwrap()),
            &p.jobs,
            &p.runs,
            MatrixOp::Adjoint,
            MatrixOp::Identity,
            DenseScalar::C64(Complex64::new(1.0, 0.0)),
            DenseScalar::C64(Complex64::new(0.0, 0.0)),
        )
        .unwrap();
}

fn victim_call(executor: &mut DefaultDenseExecutor, a: &[f64], out: &mut [f64]) -> Duration {
    let s = [1usize, 4];
    let start = Instant::now();
    executor
        .matmul_into(
            DenseWrite::F64(DenseViewMut::new(out, &[4, 4], &s, 0).unwrap()),
            DenseRead::F64(DenseView::new(a, &[4, 4], &s, 0).unwrap()),
            DenseRead::F64(DenseView::new(a, &[4, 4], &s, 0).unwrap()),
        )
        .unwrap();
    start.elapsed()
}

fn victim_samples(samples: usize) -> Vec<f64> {
    let mut executor = DefaultDenseExecutor::new();
    let a = (0..16).map(|i| i as f64 * 0.25).collect::<Vec<_>>();
    let mut out = vec![0.0; 16];
    for _ in 0..200 {
        victim_call(&mut executor, &a, &mut out);
    }
    let mut lat = Vec::with_capacity(samples);
    for i in 0..samples {
        // Spread the arrivals over the aggressor's phase boundaries.
        std::thread::sleep(Duration::from_micros(50 + (i * 37) as u64 % 150));
        lat.push(victim_call(&mut executor, &a, &mut out).as_secs_f64() * 1e6);
    }
    lat.sort_by(f64::total_cmp);
    lat
}

fn q(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let samples = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(4000);
    println!(
        "pattern,n,phase_sessions,phase_admissions,solo_phase_us,loaded_phase_us,\
         victim_alone_p50,victim_alone_p99,victim_p50,victim_p90,victim_p99,victim_max"
    );
    let alone = victim_samples(samples);
    for pattern in [&[4usize, 1, 1, 1][..], &[1, 4, 1, 4][..]] {
        for n in [8usize, 48] {
            let mut executor = DefaultDenseExecutor::new();
            let mut p = phase(pattern, n);
            for _ in 0..50 {
                run_phase(&mut executor, &mut p);
            }
            let before = cpu_session_stats();
            run_phase(&mut executor, &mut p);
            let after = cpu_session_stats();
            let iters = 2000;
            let start = Instant::now();
            for _ in 0..iters {
                run_phase(&mut executor, &mut p);
            }
            let solo = start.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let stop = Arc::new(AtomicBool::new(false));
            let count = Arc::new(AtomicU64::new(0));
            let aggressor = {
                let (stop, count) = (Arc::clone(&stop), Arc::clone(&count));
                std::thread::spawn(move || {
                    let start = Instant::now();
                    while !stop.load(Ordering::Relaxed) {
                        run_phase(&mut executor, &mut p);
                        count.fetch_add(1, Ordering::Relaxed);
                    }
                    start.elapsed()
                })
            };
            std::thread::sleep(Duration::from_millis(20));
            let lat = victim_samples(samples);
            stop.store(true, Ordering::Relaxed);
            let elapsed = aggressor.join().unwrap();
            let loaded = elapsed.as_secs_f64() * 1e6 / count.load(Ordering::Relaxed) as f64;
            println!(
                "{},{n},{},{},{solo:.2},{loaded:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
                pattern.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("-"),
                after.sessions_opened - before.sessions_opened,
                after.admissions - before.admissions,
                q(&alone, 0.5),
                q(&alone, 0.99),
                q(&lat, 0.5),
                q(&lat, 0.9),
                q(&lat, 0.99),
                lat[lat.len() - 1],
            );
        }
    }
}
