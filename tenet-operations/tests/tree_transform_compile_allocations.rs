use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tenet_core::BlockStructure;
use tenet_operations::{TreeTransformBlockSpec, TreeTransformStructure};

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

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(ptr, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn permuted_layout_compile_avoids_per_block_metadata_scratch() {
    let shapes = (0..8).map(|_| vec![2, 2, 1, 1]).collect::<Vec<_>>();
    let structure = BlockStructure::packed_column_major(4, shapes).unwrap();
    let specs = (0..8)
        .map(|block| {
            TreeTransformBlockSpec::single(block, block, 1.0_f64).with_source_axes([1, 0, 2, 3])
        })
        .collect::<Vec<_>>();
    let _ = TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let compiled =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
    COUNTING.set(false);

    // What: compiling eight permuted rank-four layouts stores metadata in the
    // final table without allocating shape/stride/seen scratch per block.
    assert_eq!(compiled.block_count(), 8);
    assert!(ALLOCATIONS.get() <= 40, "allocations={}", ALLOCATIONS.get());
}
