use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tenet_core::BlockStructure;
use tenet_operations::{TreeTransformBlockSpec, TreeTransformStructure};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            ALLOCATED_BYTES.set(ALLOCATED_BYTES.get() + layout.size());
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
            ALLOCATED_BYTES.set(ALLOCATED_BYTES.get() + new_size);
        }
        pointer
    }
}

#[test]
fn overwrite_proof_allocation_is_bounded_by_layout_metadata() {
    let logical_elements = 1_000_000;
    let structure = BlockStructure::packed_column_major(1, [vec![logical_elements]]).unwrap();
    let specs = [TreeTransformBlockSpec::single(0, 0, 1.0_f64)];

    ALLOCATIONS.set(0);
    ALLOCATED_BYTES.set(0);
    COUNTING.set(true);
    let compiled = TreeTransformStructure::compile_structures(&structure, &structure, &specs);
    COUNTING.set(false);

    // What: compile-time overwrite proof memory scales with block/layout
    // metadata, not with the number of physical scalar destinations.
    assert_eq!(compiled.unwrap().block_count(), 1);
    assert!(
        ALLOCATED_BYTES.get() < 64 * 1024,
        "bytes={}",
        ALLOCATED_BYTES.get()
    );
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn permuted_layout_compile_avoids_per_block_metadata_scratch() {
    let fixture = |blocks| {
        let shapes = (0..blocks).map(|_| vec![2, 2, 1, 1]).collect::<Vec<_>>();
        let structure = BlockStructure::packed_column_major(4, shapes).unwrap();
        let specs = (0..blocks)
            .map(|block| {
                TreeTransformBlockSpec::single(block, block, 1.0_f64).with_source_axes([1, 0, 2, 3])
            })
            .collect::<Vec<_>>();
        (structure, specs)
    };
    let (small_structure, small_specs) = fixture(8);
    let (large_structure, large_specs) = fixture(64);
    let _ = TreeTransformStructure::compile_structures(
        &large_structure,
        &large_structure,
        &large_specs,
    )
    .unwrap();

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let small = TreeTransformStructure::compile_structures(
        &small_structure,
        &small_structure,
        &small_specs,
    )
    .unwrap();
    COUNTING.set(false);
    let small_allocations = ALLOCATIONS.get();

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let large = TreeTransformStructure::compile_structures(
        &large_structure,
        &large_structure,
        &large_specs,
    )
    .unwrap();
    COUNTING.set(false);
    let large_allocations = ALLOCATIONS.get();

    // What: increasing block count eightfold grows final metadata but does not
    // introduce one scratch allocation per block.
    assert_eq!(small.block_count(), 8);
    assert_eq!(large.block_count(), 64);
    assert!(
        large_allocations < small_allocations * 2,
        "small={small_allocations} large={large_allocations}"
    );
}
