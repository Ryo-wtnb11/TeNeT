use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tenet_core::{unique_permute_tree, FusionTreeKey, Z2FusionRule, Z2Irrep};

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
fn unique_identity_permute_does_not_allocate() {
    // What: the identity Unique permutation returns its inline tree key
    // without constructing permutation-level scratch.
    let tree = FusionTreeKey::try_new_for_rule(
        &Z2FusionRule,
        [Z2Irrep::ODD.sector_id()],
        Some(Z2Irrep::ODD.sector_id()),
        [false],
        [],
        [],
    )
    .unwrap();
    let _ = unique_permute_tree(&Z2FusionRule, &tree, &[0]).unwrap();

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let transformed = unique_permute_tree(&Z2FusionRule, &tree, &[0]).unwrap();
    COUNTING.set(false);

    assert_eq!(transformed.0, tree);
    assert_eq!(transformed.1, 1.0);
    assert_eq!(ALLOCATIONS.get(), 0);
}
