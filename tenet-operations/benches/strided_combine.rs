//! Regression microbench for issue #230: the non-contiguous strided-combine
//! recurse path in `host_scalar_kernels`. This is the regime a rank-4 SU(2)
//! d=4 contract feeds — a recoupling/permute block combine whose source is not
//! column-major-contiguous, so `raw_strided_combine_recurse` walks it element
//! by element through `checked_strided_offset` + `checked_offset_to_index`.
//!
//! No timing assertion here: shared CI runners are too noisy for a pass/fail
//! latency gate (same rationale as benches/fusion_tree_key.rs in tenet-core).
//! The gate is this bench compiling and running, plus the `size_of` canary
//! `offset_error_result_stays_small` in host_scalar_kernels.rs.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tenet_operations::tensoradd_raw_strided_kernel;

// Rank-4, d=4 block: 256 elements, the SU(2) recoupling block size.
const D: usize = 4;
const RANK: usize = 4;
const LEN: usize = D * D * D * D;

fn bench_noncontiguous_combine(c: &mut Criterion) {
    let shape = [D; RANK];
    // Destination is column-major-contiguous; source reads as a transposed
    // (row-major) layout. Mismatched contiguity forces the per-element recurse
    // path instead of the fast contiguous slice copy — exactly the recoupling
    // access pattern that regressed in #230.
    let dst_strides: [isize; RANK] = [1, 4, 16, 64];
    let src_strides: [isize; RANK] = [64, 16, 4, 1];

    let mut dst = vec![0.0f64; LEN];
    let src = vec![1.0f64; LEN];
    let mut zero_strides: Vec<isize> = Vec::new();

    c.bench_function("strided_combine_noncontiguous_rank4_d4", |bencher| {
        bencher.iter(|| {
            tensoradd_raw_strided_kernel(
                &mut zero_strides,
                black_box(&mut dst),
                black_box(&src),
                &shape,
                &dst_strides,
                &src_strides,
                0,
                0,
                false,
                black_box(1.0f64),
                black_box(1.0f64),
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, bench_noncontiguous_combine);
criterion_main!(benches);
