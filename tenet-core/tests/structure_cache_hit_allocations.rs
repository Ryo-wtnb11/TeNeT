use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;

use tenet_core::{FusionTreeHomSpace, SU2FusionRule, SU2Irrep};

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
fn complete_structure_cache_hit_returns_canonical_arc_without_allocating() {
    // What: while a wrapper is live, a complete-structure lookup returns the
    // canonical Arc itself; the key is built on the stack and no wrapper,
    // region state, or key Arc is constructed. Own test binary so no other
    // test can evict the entry from the bounded cache between lookups.
    let spin = |twice: usize| SU2Irrep::from_twice_spin(twice).sector_id();
    let hom = FusionTreeHomSpace::from_sectors(
        [(spin(1), 2), (spin(3), 1)],
        [(spin(1), 3), (spin(2), 2)],
    );
    let first = hom
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let second = black_box(&hom)
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();
    COUNTING.set(false);

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(ALLOCATIONS.get(), 0);
}

#[test]
fn warm_layout_cache_hit_builds_its_lookup_key_without_allocating() {
    // What: a warm fusion-tree layout lookup constructs its cache key from the
    // HomSpace content in place, so neither the read-side lookup nor the commit
    // allocates. Uses a HomSpace of its own so the shared bounded layout cache
    // holds the warmed entry regardless of what other tests admit.
    let spin = |twice: usize| SU2Irrep::from_twice_spin(twice).sector_id();
    let hom = FusionTreeHomSpace::from_sectors(
        [(spin(2), 2), (spin(4), 3)],
        [(spin(2), 1), (spin(3), 2)],
    );
    let first = hom.fusion_tree_keys(&SU2FusionRule);

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let second = black_box(&hom).fusion_tree_keys(&SU2FusionRule);
    COUNTING.set(false);

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(ALLOCATIONS.get(), 0);
}
