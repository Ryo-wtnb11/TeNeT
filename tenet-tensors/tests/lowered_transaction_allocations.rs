use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet_core::{FusionProductSpace, FusionTreeHomSpace, U1FusionRule};
use tenet_tensors::{reset_global_operation_caches, BoundDynamicFusionMapSpace};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            ALLOCATIONS.with(|count| count.set(count.get() + 1));
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[test]
fn lowered_scratch_hit_matches_encoded_hit_allocation_and_identity() {
    // What: in this single-test process, the transactional lowered warm hit
    // keeps encoded allocation cost and exact retained structure identity.
    reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    let provider = Arc::new(U1FusionRule);
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let cold = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::clone(&provider),
        homspace,
        [Vec::<usize>::new()],
    )
    .unwrap();

    let lowered_homspace = cold.space().homspace().clone();
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let lowered = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::clone(&provider),
        lowered_homspace,
        [Vec::<usize>::new()],
    )
    .unwrap();
    COUNTING.set(false);
    let lowered_allocations = ALLOCATIONS.get();

    let encoded_homspace = cold.space().homspace().clone();
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let encoded = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
        provider,
        encoded_homspace,
        [Vec::<usize>::new()],
    )
    .unwrap();
    COUNTING.set(false);
    let encoded_allocations = ALLOCATIONS.get();

    assert_eq!(lowered_allocations, encoded_allocations);
    assert!(Arc::ptr_eq(
        cold.space().structure(),
        lowered.space().structure()
    ));
    assert!(Arc::ptr_eq(
        cold.space().structure(),
        encoded.space().structure()
    ));
    assert_eq!(lowered.space(), encoded.space());
}
