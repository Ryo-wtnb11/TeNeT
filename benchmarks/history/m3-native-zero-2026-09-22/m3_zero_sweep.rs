//! M3 (#740) small-output sweep, run as `tenet-network/examples/m3_zero_sweep.rs` from this directory
//! (copied in for the measurement, not built by the workspace).
//!
//! Warm `compose` / `contract` of `lhs: [k] -> [m]`, `rhs: [n] -> [k]` over one
//! U(1) sector with a small inner extent `k = 4`, so output initialisation is a
//! large share of the call. Per row: 5 warm-ups, then 51 samples of
//! - `submit`: the call alone (no completion; TeNeT has no public sync);
//! - `e2e`: the call followed by a 1-element `to_host`, which drains the one
//!   CubeCL stream, minus nothing (the `sync` row reports that readback alone).
//!
//! Output: `dtype,op,bytes,submit_ns,e2e_ns`, medians.

#[cfg(not(feature = "cuda"))]
fn main() {}

#[cfg(feature = "cuda")]
fn main() {
    use std::hint::black_box;
    use std::sync::Arc;
    use std::time::Instant;
    use tenet::core::{U1FusionRule, U1Irrep};
    use tenet::prelude::Complex64;
    use tenet::typed::{CudaPayload, GradedSpace, Runtime, TensorMap};

    const SAMPLES: usize = 51;
    const K: usize = 4;

    fn leg(d: usize) -> GradedSpace<U1FusionRule> {
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), d)]).unwrap()
    }

    fn median(mut v: Vec<u128>) -> u128 {
        v.sort_unstable();
        v[v.len() / 2]
    }

    fn row<D: CudaPayload>(rt: &Runtime, name: &str, one: D, m: usize, n: usize) {
        let (lm, lk, ln) = (leg(m), leg(K), leg(n));
        let lhs = TensorMap::<U1FusionRule, D>::from_block_fn(rt, [&lm], [&lk], |_, _| one)
            .unwrap()
            .to_cuda()
            .unwrap();
        let rhs = TensorMap::<U1FusionRule, D>::from_block_fn(rt, [&lk], [&ln], |_, _| one)
            .unwrap()
            .to_cuda()
            .unwrap();
        let tiny = TensorMap::<U1FusionRule, D>::from_block_fn(rt, [&leg(1)], [&leg(1)], |_, _| one)
            .unwrap()
            .to_cuda()
            .unwrap();
        let bytes = m * n * std::mem::size_of::<D>();
        let ops: [(&str, &dyn Fn() -> TensorMap<U1FusionRule, D, tenet::typed::CudaStorage<D>>); 2] = [
            ("compose", &|| lhs.compose(&rhs).unwrap()),
            ("contract", &|| lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap()),
        ];
        for (op, call) in ops {
            for _ in 0..5 {
                drop(call());
                tiny.to_host().unwrap();
            }
            let (mut submit, mut e2e) = (Vec::new(), Vec::new());
            for _ in 0..SAMPLES {
                tiny.to_host().unwrap();
                let start = Instant::now();
                let out = call();
                submit.push(start.elapsed().as_nanos());
                black_box(tiny.to_host().unwrap());
                e2e.push(start.elapsed().as_nanos());
                drop(out);
            }
            println!("{name},{op},{bytes},{},{}", median(submit), median(e2e));
        }
        let mut sync = Vec::new();
        for _ in 0..SAMPLES {
            let start = Instant::now();
            black_box(tiny.to_host().unwrap());
            sync.push(start.elapsed().as_nanos());
        }
        println!("{name},sync,{bytes},0,{}", median(sync));
    }

    let rt = Runtime::builder().cuda(0).build().unwrap();
    println!("dtype,op,bytes,submit_ns,e2e_ns");
    // m x n outputs of 8 f64 / 4 c64 elements (64 B) up to 16 MiB.
    for (m, n) in [(2, 4), (16, 32), (128, 256), (1024, 2048)] {
        row::<f64>(&rt, "f64", 1.0, m, n);
    }
    for (m, n) in [(2, 2), (16, 16), (128, 128), (1024, 1024)] {
        row::<Complex64>(&rt, "c64", Complex64::new(1.0, 0.5), m, n);
    }
}
