use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenet::prelude::*;

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn repartition_to_current_split_does_not_allocate() {
    // What: a repartition which leaves the boundary unchanged only clones Arc
    // handles and performs no heap allocation.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::su2([(0, 1), (1, 2)]).unwrap();
    let source =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space, &space], [&space], 191).unwrap();

    black_box(source.repartition(source.codomain_rank()).unwrap());
    ALLOCATIONS.set(0);
    ENABLED.set(true);
    let output = black_box(source.repartition(source.codomain_rank()).unwrap());
    ENABLED.set(false);
    black_box(output);

    assert_eq!(ALLOCATIONS.get(), 0);
}
