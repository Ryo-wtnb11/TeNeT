//! #41 perf probe (ignored; run explicitly):
//!   cargo test --release -p tenet-operations --test bench_41_probe -- --ignored --nocapture
//!
//! Small-block repeated copy at the SU(2) replay regime (21 rank-4 d=4
//! transposed blocks): the pre-#41 hand-rolled kernel loop (reached through the
//! public adapter's #232 baked route, `apply_fused_pair_slices`, NO per-call
//! bounds check) vs the #41-delegated strided-rs #140 route (`copy_scale_raw`
//! behind the non-baked adapter path; per-call fuse + 2x validate_bounds) vs
//! #142 `CopyPlan` (compile once, execute many; still validates per execute).
//!
//! Documents the ACCEPTED constant-factor cost of #41: the delegated non-baked
//! path pays the per-call `RawStrided::new` bounds validation (~+25% on this
//! regime). Same complexity order; the warm hot replay path is unaffected (it
//! takes the baked route, which is the baseline here). Reaching the baseline on
//! the non-baked path needs a validation-free prepared execute in strided-rs;
//! `new_unchecked` is blocked by tenet-operations' #![deny(unsafe_code)].

use std::time::Instant;

use strided_kernel::{CopyPlan, RawStridedMut, RawStridedRef};
use tenet_operations::{BakedFusedLayout, HostKernelAdapter, StridedHostKernelAdapter};

#[test]
#[ignore]
fn bench_41_fused_vs_raw_vs_plan() {
    const BLOCKS: usize = 21;
    let dims = [4usize, 4, 4, 4];
    let src_strides = [1isize, 4, 16, 64]; // column-major src
    let dst_strides = [64isize, 16, 4, 1]; // transposed dst (non-contiguous)
    let elems = 256usize;
    let src: Vec<f64> = (0..elems).map(|i| i as f64 * 0.5 - 3.0).collect();
    let mut dst = vec![0.0f64; elems];
    let iters = 20_000usize;
    let mut adapter = StridedHostKernelAdapter::default();

    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    // (a) pre-#41 hand-rolled loop via the baked route: the slices below are
    // the fuse_pair_layout normalization of this stride pair (ordered by
    // destination stride; no adjacent pair fuses), computed once — exactly what
    // the #232 layout table bakes at compile time.
    let baked_dims = [4usize, 4, 4, 4];
    let baked_dst = [1isize, 4, 16, 64];
    let baked_src = [64isize, 16, 4, 1];
    let baked = BakedFusedLayout {
        dims: &baked_dims,
        dst_strides: &baked_dst,
        src_strides: &baked_src,
    };
    let mut fused_ns = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..iters {
            for _ in 0..BLOCKS {
                adapter
                    .copy_scale_strided_baked(
                        &mut dst,
                        &src,
                        &dims,
                        &dst_strides,
                        &src_strides,
                        0,
                        0,
                        false,
                        1.0,
                        Some(baked),
                    )
                    .unwrap();
            }
        }
        fused_ns.push(t.elapsed().as_nanos() as f64 / (iters * BLOCKS) as f64);
    }

    // (b) #41 delegated route: non-baked adapter path -> strided-rs #140
    // copy_scale_raw (per-call RawStrided::new + fuse).
    let mut raw_ns = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..iters {
            for _ in 0..BLOCKS {
                adapter
                    .copy_scale_strided(
                        &mut dst,
                        &src,
                        &dims,
                        &dst_strides,
                        &src_strides,
                        0,
                        0,
                        false,
                        1.0,
                    )
                    .unwrap();
            }
        }
        raw_ns.push(t.elapsed().as_nanos() as f64 / (iters * BLOCKS) as f64);
    }

    // (c) #142 CopyPlan, compile once then execute_scale per block.
    let plan = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();
    let compile_t = Instant::now();
    for _ in 0..iters {
        let _ = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();
    }
    let compile_ns = compile_t.elapsed().as_nanos() as f64 / iters as f64;
    let mut plan_ns = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..iters {
            for _ in 0..BLOCKS {
                let s = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
                let mut d = RawStridedMut::new(&mut dst, &dims, &dst_strides, 0).unwrap();
                plan.execute_scale(&mut d, &s, 1.0).unwrap();
            }
        }
        plan_ns.push(t.elapsed().as_nanos() as f64 / (iters * BLOCKS) as f64);
    }

    println!("\n#41 small-block (21x rank4 d4 transposed) per-block ns, median-of-5:");
    println!("  baked hand-rolled loop:   {:.1}", median(fused_ns));
    println!("  copy_scale_raw (#140):    {:.1}", median(raw_ns));
    println!(
        "  CopyPlan.execute (#142):  {:.1}  (compile once: {:.1} ns)",
        median(plan_ns),
        compile_ns
    );
}
