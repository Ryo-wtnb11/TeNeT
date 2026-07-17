use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenet::prelude::*;

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

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let value = operation();
    COUNTING.set(false);
    (value, ALLOCATIONS.get())
}

#[test]
fn warmed_non_abelian_inner_and_norm_do_not_allocate() {
    // What: cold region compilation is observable, while explicit warm-up
    // leaves both public reductions allocation-free on the caller thread.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::fz2_u1_su2([
        ((0, -2, 0), 4),
        ((0, 1, 2), 3),
        ((1, -1, 1), 4),
        ((1, 2, 3), 2),
    ])
    .unwrap();
    let lhs =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space, &space], [&space], 282_401).unwrap();
    let rhs =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space, &space], [&space], 282_402).unwrap();

    let (cold, cold_allocations) = measured(|| lhs.inner(&rhs).unwrap());
    eprintln!("cold coupled-region initialization: {cold_allocations} allocations");
    black_box(cold);
    black_box(lhs.norm().unwrap());

    let (inner, inner_allocations) = measured(|| lhs.inner(&rhs).unwrap());
    let (norm, norm_allocations) = measured(|| lhs.norm().unwrap());
    black_box((inner, norm));

    assert_eq!(inner_allocations, 0);
    assert_eq!(norm_allocations, 0);
}
