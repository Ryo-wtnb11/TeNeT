//! Allocation gate for the host raw combine actions routed through Strided's
//! raw kernels: `Copy`, `CopyScale` and `Axpy` stay allocation-free at every
//! rank up to `RAW_FUSED_RANK_LIMIT`, and above it TeNeT's own loop keeps
//! them allocation-free.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use num_complex::Complex64;
use tenet_operations::tensoradd_raw_strided_kernel;

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn raw_combine_actions_allocate_nothing_at_any_rank() {
    // What: for ranks 1..=10 (two above the fused limit) and a transposed
    // source, conjugating copy, scale and axpy allocate nothing on the
    // caller thread. 2^10 elements stay below Strided's parallel threshold.
    let (alpha, zero, one) = (
        Complex64::new(0.5, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    );
    let mut zero_strides = Vec::new();
    for rank in 1..=10usize {
        let shape = vec![2usize; rank];
        let dst_strides: Vec<isize> = (0..rank).map(|axis| 1 << axis).collect();
        let src_strides: Vec<isize> = dst_strides.iter().rev().copied().collect();
        let len = 1 << rank;
        let src: Vec<Complex64> = (0..len)
            .map(|i| Complex64::new(i as f64, -(i as f64)))
            .collect();
        let mut dst = vec![zero; len];
        for conjugate in [false, true] {
            for (a, b) in [(one, zero), (alpha, zero), (alpha, one)] {
                let mut run = || {
                    tensoradd_raw_strided_kernel(
                        &mut zero_strides,
                        black_box(&mut dst),
                        black_box(&src),
                        &shape,
                        &dst_strides,
                        &src_strides,
                        0,
                        0,
                        conjugate,
                        a,
                        b,
                    )
                    .unwrap()
                };
                run();
                ALLOCATIONS.set(0);
                COUNTING.set(true);
                run();
                COUNTING.set(false);
                assert_eq!(ALLOCATIONS.get(), 0, "rank {rank} alpha {a} beta {b}");
            }
        }
    }
}

#[test]
fn parallel_copy_matches_oracle_without_caller_allocations() {
    // What: a transposed 256 x 512 copy (131072 elements, above Strided's
    // 32768-element parallel threshold), plain and conjugating, run inside an
    // explicit 4-thread Rayon pool so `CopyPlan` fans out. Every element
    // equals the transposed (conjugated) source bit for bit, and the thread
    // running the call allocates nothing once warm.
    let (rows, cols) = (256usize, 512usize);
    let len = rows * cols;
    let shape = [rows, cols];
    let dst_strides = [1isize, rows as isize];
    let src_strides = [cols as isize, 1];
    let src: Vec<Complex64> = (0..len)
        .map(|i| Complex64::new(i as f64 + 0.25, -(i as f64) - 0.5))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for conjugate in [false, true] {
        let (dst, allocations) = pool.install(|| {
            assert_eq!(rayon::current_num_threads(), 4);
            let mut dst = vec![Complex64::new(0.0, 0.0); len];
            let mut zero_strides = Vec::new();
            let mut run = |dst: &mut [Complex64]| {
                tensoradd_raw_strided_kernel(
                    &mut zero_strides,
                    black_box(dst),
                    black_box(&src),
                    &shape,
                    &dst_strides,
                    &src_strides,
                    0,
                    0,
                    conjugate,
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                )
                .unwrap()
            };
            run(&mut dst);
            dst.fill(Complex64::new(0.0, 0.0));
            ALLOCATIONS.set(0);
            COUNTING.set(true);
            run(&mut dst);
            COUNTING.set(false);
            (dst, ALLOCATIONS.get())
        });
        assert_eq!(allocations, 0, "conj {conjugate}");
        for col in 0..cols {
            for row in 0..rows {
                let source = src[row * cols + col];
                let want = if conjugate { source.conj() } else { source };
                let got = dst[row + col * rows];
                assert_eq!(
                    (got.re.to_bits(), got.im.to_bits()),
                    (want.re.to_bits(), want.im.to_bits()),
                    "conj {conjugate} ({row}, {col})"
                );
            }
        }
    }
}
