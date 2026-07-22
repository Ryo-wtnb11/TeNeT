use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenet::prelude::*;

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATED: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATED.set(ALLOCATED.get() + layout.size() as u64);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATED.set(ALLOCATED.get() + new_size as u64);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measure_allocated_bytes(f: impl FnOnce()) -> u64 {
    ALLOCATED.set(0);
    ENABLED.set(true);
    f();
    ENABLED.set(false);
    ALLOCATED.get()
}

#[test]
fn identity_output_order_adds_no_axis_validation_allocation() {
    let runtime = Runtime::builder()
        .dense_threads(1)
        .recoupling_threads(1)
        .build()
        .unwrap();
    let space = Space::su2([(0, 2), (1, 3), (2, 2)]).unwrap();
    let lhs =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space, &space], [&space], 224_701).unwrap();
    let rhs =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space, &space], 224_702).unwrap();

    black_box(lhs.contract(&rhs, &[2], &[0]).unwrap());
    black_box(
        lhs.contract_ordered(&rhs, &[2], &[0], &[0, 1, 2, 3])
            .unwrap(),
    );
    let direct = measure_allocated_bytes(|| {
        black_box(lhs.contract(&rhs, &[2], &[0]).unwrap());
    });
    let identity = measure_allocated_bytes(|| {
        black_box(
            lhs.contract_ordered(&rhs, &[2], &[0], &[0, 1, 2, 3])
                .unwrap(),
        );
    });

    // What: the identity pAB wrapper performs the same owned contraction and
    // does not allocate two temporary open-axis validation vectors first.
    assert_eq!(identity, direct);
}
