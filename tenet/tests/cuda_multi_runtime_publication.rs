//! Public-API probe for #1384: a device output returned by one `Runtime` can
//! be read early by a thread that also uses a second `Runtime` on the same
//! device.
//!
//! `tenet-dense/tests/cuda_multi_context_publication.rs` forces the exact
//! interleaving at the adapter seam. Here the same steps run through
//! `Runtime`, where step 2 cannot be placed inside `compose` directly, so the
//! thread-B delay is swept across the late part of the call instead:
//!
//! 1. thread A uploads a tiny `T` through R2, then runs `compose` through R1
//!    (zero upload binds `O`, then one GEMM per sector block writes it);
//! 2. thread B, after the swept delay, reads `T` through R2; if that lands
//!    between `O`'s bind and a later GEMM, B's stream records itself synced to
//!    A's past `O`'s bind cursor (tensor4all/cubecl#16);
//! 3. B reads `O` through R1 and compares with the hand oracle: all-ones
//!    operands, so every element equals the sector degeneracy.
//!
//! The one-Runtime control reads `T` through R1, whose lease orders it after
//! `compose`, and must always pass. The test asserts no stale read, so it fails
//! while the hazard exists and guards the TeNeT-side fix.
//!
//! Run with `cargo test --release -p tenet-rs --no-default-features --features \
//! cuda,cpu-faer --test cuda_multi_runtime_publication -- --ignored \
//! --nocapture --test-threads=1` on a CUDA host. A debug build also trips an
//! unrelated `u32` overflow in CubeCL's staged-upload counter
//! (`drop_queue/policy.rs:36`) once 4 GiB have been uploaded.

#![cfg(feature = "cuda")]

use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::typed::{CudaStorage, GradedSpace, Runtime, TensorMap};

const SECTORS: i32 = 2;
const DEG: usize = 4096;
const ITERS: usize = 60;

type Host = TensorMap<U1FusionRule, f64>;
type Device = TensorMap<U1FusionRule, f64, CudaStorage<f64>>;

fn ones(runtime: &Runtime, sectors: i32, deg: usize) -> Host {
    let space = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        (0..sectors).map(|q| (U1Irrep::new(q), deg)),
    )
    .unwrap();
    TensorMap::from_block_fn(runtime, [&space], [&space], |_, _| 1.0).unwrap()
}

/// Host wall time of one warm `compose`, which B's delay sweep spans.
fn compose_time(runtime: &Runtime) -> Duration {
    let lhs = ones(runtime, SECTORS, DEG).to_cuda().unwrap();
    lhs.compose(&lhs).unwrap().to_host().unwrap();
    let start = Instant::now();
    let o = lhs.compose(&lhs).unwrap();
    let elapsed = start.elapsed();
    o.to_host().unwrap();
    elapsed
}

/// B's delay for `iter`: the window sits late in `compose` (after the zero
/// upload), so the sweep covers its last 30%.
fn delay(iter: usize, compose: Duration) -> Duration {
    compose.mul_f64(0.7 + 0.3 * iter as f64 / ITERS as f64)
}

fn run(r1: Runtime, r2: Runtime) -> usize {
    let compose = compose_time(&r1);
    println!("compose takes {compose:?}");
    let (t_tx, t_rx) = mpsc::channel::<Device>();
    let (o_tx, o_rx) = mpsc::channel::<Device>();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let a = std::thread::spawn(move || {
        let lhs = ones(&r1, SECTORS, DEG).to_cuda().unwrap();
        let tiny = ones(&r2, 1, 1);
        for _ in 0..ITERS {
            t_tx.send(tiny.to_cuda().unwrap()).unwrap();
            o_tx.send(lhs.compose(&lhs).unwrap()).unwrap();
            done_rx.recv().unwrap();
        }
    });
    let mut stale = 0;
    for iter in 0..ITERS {
        let t = t_rx.recv().unwrap();
        std::thread::sleep(delay(iter, compose));
        t.to_host().unwrap();
        let host = o_rx.recv().unwrap().to_host().unwrap();
        let data = host.data();
        let wrong = data.iter().filter(|&&x| x != DEG as f64).count();
        if wrong > 0 {
            stale += 1;
            let first = data.iter().position(|&x| x != DEG as f64).unwrap();
            println!(
                "iter {iter} (delay {:?}): {wrong}/{} stale, first [{first}] = {}",
                delay(iter, compose),
                data.len(),
                data[first]
            );
        }
        done_tx.send(()).unwrap();
    }
    a.join().unwrap();
    stale
}

#[test]
#[ignore = "requires a CUDA device"]
fn second_runtime_on_one_device_reads_only_completed_outputs() {
    let r1 = Runtime::builder().cuda(0).build().unwrap();
    let r2 = Runtime::builder().cuda(0).build().unwrap();
    let stale = run(r1, r2);
    println!("two runtimes: {stale}/{ITERS} iterations read a stale output");
    assert_eq!(stale, 0, "#1384 reproduced: {stale}/{ITERS} stale reads");
}

#[test]
#[ignore = "requires a CUDA device"]
fn single_runtime_control_reads_only_completed_outputs() {
    let r = Runtime::builder().cuda(0).build().unwrap();
    let stale = run(r.clone(), r);
    println!("one runtime: {stale}/{ITERS} iterations read a stale output");
    assert_eq!(stale, 0);
}
