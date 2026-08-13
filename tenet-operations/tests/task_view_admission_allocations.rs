use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet_core::BlockStructure;
use tenet_operations::{
    DenseTreeTransformOperations, TreeTransformBackend, TreeTransformReplayProfile,
    TreeTransformStructure, TreeTransformWorkspace,
};

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
        unsafe { System.dealloc(pointer, layout) }
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
fn warmed_public_host_admission_without_tasks_is_allocation_free() {
    // An empty completed transform takes the normal profiled Host admission
    // seam but has no scale, block, coefficient, or dense numerical work.
    let structure =
        Arc::new(BlockStructure::packed_column_major(1, std::iter::empty::<Vec<usize>>()).unwrap());
    let transform =
        TreeTransformStructure::<f64>::compile_structures(&structure, &structure, &[]).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    let mut destination = [];
    let source = [];

    let mut replay = |profile: &mut TreeTransformReplayProfile| {
        backend
            .tree_transform_structure_overwrite_into_raw_profiled(
                &mut workspace,
                &transform,
                &structure,
                &structure,
                &mut destination,
                &source,
                1.0,
                profile,
            )
            .unwrap();
    };
    replay(&mut TreeTransformReplayProfile::default());

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let mut profile = TreeTransformReplayProfile::default();
    for _ in 0..100 {
        replay(&mut profile);
    }
    COUNTING.set(false);

    assert_eq!(ALLOCATIONS.get(), 0);
    assert_eq!(profile.single_blocks, 0);
    assert_eq!(profile.multi_blocks, 0);
    assert_eq!(profile.multi_coefficient_prepare, std::time::Duration::ZERO);
}
