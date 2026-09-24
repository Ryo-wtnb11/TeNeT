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
