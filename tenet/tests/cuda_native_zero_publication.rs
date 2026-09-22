//! M3 (#740) gate: a returning device `contract`/`compose` output is zeroed on
//! the device (Tenferro `alloc_zero_output`: bind, flush, `cuMemsetD8Async`),
//! and every element no GEMM writes must read back as exact `+0.0`, across
//! threads and across two `Runtime`s on one device, for all four payloads.
//!
//! The memset writes after the allocation's bind, which CubeCL does not
//! publish (tensor4all/cubecl#16); it is ordered before every later use only
//! because all device work runs on one CubeCL stream (#1391). Several threads
//! over two Runtimes interleave poison, contraction, cross-Runtime reads and
//! readback, so a memset overtaken by a later write, or a read overtaking the
//! memset, shows up as a wrong or non-`+0.0` element.
//!
//! Fixture: `lhs: b -> a`, `rhs: a -> b` with `a = {0, 1}` and `b = {0}`. The
//! output `a -> a` stores a sector-1 block that no GEMM writes (`b` has no
//! sector 1), so it holds exactly what the zero initialisation left. Every
//! entry has positive parts, so each active output element is nonzero and
//! "expected == 0" identifies the inactive block exactly. Values are small
//! integers, exact in every lane, so the comparison with Host is exact.
//!
//! Pool poison: right before each operation the thread uploads and drops a
//! NaN tensor of exactly the output's size, so CubeCL's pool is likely to hand
//! that buffer to the output. Reuse is not observable through the public API,
//! so a pass does not prove reuse happened; a stale NaN or a `-0.0` fails.
//!
//! Run with `cargo test --release -p tenet-rs --no-default-features --features \
//! cuda,cpu-faer --test cuda_native_zero_publication -- --ignored --nocapture \
//! --test-threads=1` on a CUDA host.

#![cfg(feature = "cuda")]

mod common;

use std::sync::{Arc, Barrier};

use common::DevicePayload;
use num_complex::{Complex32, Complex64};
use tenet::core::{U1FusionRule, U1Irrep};
use tenet::typed::{GradedSpace, Runtime, TensorMap};

const THREADS: usize = 6;
const ROUNDS: usize = 12;
const DEG: usize = 96;

fn space(sectors: i32) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        (0..sectors).map(|q| (U1Irrep::new(q), DEG)),
    )
    .unwrap()
}

fn operands<D: DevicePayload>(
    runtime: &Runtime,
    seed: usize,
) -> (TensorMap<U1FusionRule, D>, TensorMap<U1FusionRule, D>) {
    let (a, b) = (space(2), space(1));
    let entry = move |index: &[usize]| {
        D::entry(
            1.0 + ((index[0] + 2 * index[1] + seed) % 3) as f64,
            1.0 + ((index[0] + index[1] + seed) % 2) as f64,
        )
    };
    let lhs = TensorMap::from_block_fn(runtime, [&a], [&b], move |_, index| entry(index)).unwrap();
    let rhs = TensorMap::from_block_fn(runtime, [&b], [&a], move |_, index| entry(index)).unwrap();
    (lhs, rhs)
}

/// Uploads and drops a NaN tensor the size of an `a -> a` output.
fn poison<D: DevicePayload>(runtime: &Runtime) {
    let a = space(2);
    let nan = D::entry(f64::NAN, f64::NAN);
    drop(
        TensorMap::<U1FusionRule, D>::from_block_fn(runtime, [&a], [&a], move |_, _| nan)
            .unwrap()
            .to_cuda()
            .unwrap(),
    );
}

/// Exact comparison with Host; inactive elements must be `+0.0` bits.
fn check<D: DevicePayload>(actual: &[D], expected: &[D], what: &str) -> usize {
    assert_eq!(actual.len(), expected.len(), "{what} [{}]", D::NAME);
    let zero = D::entry(0.0, 0.0);
    let mut inactive = 0;
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if want == zero {
            inactive += 1;
            let (re, im) = got.parts();
            assert!(
                re.to_bits() == 0 && im.to_bits() == 0,
                "{what} [{}]: inactive element {index} is {got:?}, not +0.0",
                D::NAME
            );
        } else {
            assert_eq!(got, want, "{what} [{}]: element {index}", D::NAME);
        }
    }
    inactive
}

fn round<D: DevicePayload>(own: &Runtime, other: &Runtime, seed: usize) {
    let (lhs, rhs) = operands::<D>(own, seed);
    let composed = lhs.compose(&rhs).unwrap();
    let contracted = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    let (lhs, rhs) = (lhs.to_cuda().unwrap(), rhs.to_cuda().unwrap());

    poison::<D>(own);
    let device_composed = lhs.compose(&rhs).unwrap();
    poison::<D>(own);
    let device_contracted = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    // Touch the other Runtime between the write and the readback.
    poison::<D>(other);
    operands::<D>(other, seed)
        .0
        .to_cuda()
        .unwrap()
        .to_host()
        .unwrap();

    let inactive = check(
        device_composed.to_host().unwrap().data(),
        composed.data(),
        "compose",
    ) + check(
        device_contracted.to_host().unwrap().data(),
        contracted.data(),
        "contract",
    );
    assert_eq!(
        inactive,
        2 * DEG * DEG,
        "the fixture must keep one inactive block per output"
    );
}

fn stress(r1: &Runtime, r2: &Runtime) {
    let barrier = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let (own, other) = if t.is_multiple_of(2) {
                (r1.clone(), r2.clone())
            } else {
                (r2.clone(), r1.clone())
            };
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for r in 0..ROUNDS {
                    let seed = t * ROUNDS + r;
                    round::<f64>(&own, &other, seed);
                    round::<f32>(&own, &other, seed);
                    round::<Complex64>(&own, &other, seed);
                    round::<Complex32>(&own, &other, seed);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn device_zeroed_outputs_match_host_across_threads_and_two_runtimes() {
    let r1 = Runtime::builder().cuda(0).build().unwrap();
    let r2 = Runtime::builder().cuda(0).build().unwrap();
    stress(&r1, &r2);
}

#[test]
#[ignore = "requires a CUDA device"]
fn device_zeroed_outputs_match_host_across_threads_on_one_runtime() {
    let r = Runtime::builder().cuda(0).build().unwrap();
    stress(&r, &r);
}
