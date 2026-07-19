use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet_core::{BlockKey, BlockSpec, BlockStructure, FusionTreePairKey};
use tenet_dense::DefaultDenseExecutor;
use tenet_operations::{
    try_tree_transform_structure_overwrite_owned_raw, TransposeBackend, TreeTransformBlockSpec,
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

fn canonical_structure() -> Arc<BlockStructure> {
    let key = BlockKey::from(FusionTreePairKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [false],
        [],
        [],
        [],
        [],
    ));
    Arc::new(
        BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::with_key(key, vec![8, 8], vec![1, 8], 0).unwrap()],
        )
        .unwrap(),
    )
}

#[test]
fn warm_owned_transform_allocates_only_the_output_payload() {
    // What: after plan/region caches are warm, the uninitialized writer makes
    // one allocation for the returned Vec and no pre-zero scratch allocation.
    let structure = canonical_structure();
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::single(0, 0, 1.0)],
    )
    .unwrap();
    let source = vec![2.0; 64];
    let mut dense = DefaultDenseExecutor::new();
    let mut workspace = TreeTransformWorkspace::default();
    let warm = try_tree_transform_structure_overwrite_owned_raw(
        &mut dense,
        &mut workspace,
        TransposeBackend::FusedLoops,
        &transform,
        &structure,
        &structure,
        1,
        &source,
        1.0,
    )
    .unwrap()
    .unwrap();
    drop(warm);

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let output = try_tree_transform_structure_overwrite_owned_raw(
        &mut dense,
        &mut workspace,
        TransposeBackend::FusedLoops,
        &transform,
        &structure,
        &structure,
        1,
        &source,
        1.0,
    )
    .unwrap()
    .unwrap();
    COUNTING.set(false);

    assert_eq!(ALLOCATIONS.get(), 1);
    assert_eq!(output, source);
}
