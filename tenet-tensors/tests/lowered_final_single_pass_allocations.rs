use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep};
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

fn homspace(sector_count: i32) -> FusionTreeHomSpace {
    let sectors = (0..sector_count)
        .map(|charge| (U1Irrep::new(charge).sector_id(), 2))
        .collect::<Vec<_>>();
    FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(sectors.clone(), false)]),
        FusionProductSpace::new([SectorLeg::new(sectors, false)]),
    )
}

fn cold_lowered_allocations(sector_count: i32) -> usize {
    reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let result = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_lowered(
        Arc::new(U1FusionRule),
        homspace(sector_count),
    )
    .unwrap();
    COUNTING.set(false);
    assert_eq!(
        result.space().structure().block_count(),
        sector_count as usize
    );
    ALLOCATIONS.get()
}

fn cold_encoded_allocations(sector_count: i32) -> usize {
    reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let result = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(U1FusionRule),
        homspace(sector_count),
    )
    .unwrap();
    COUNTING.set(false);
    assert_eq!(
        result.space().structure().block_count(),
        sector_count as usize
    );
    ALLOCATIONS.get()
}

#[test]
fn lowered_final_cold_build_has_no_per_tree_shape_allocation_slope() {
    // What: growing one-sector final storage to sixteen U1 sectors does not
    // restore one heap-allocated shape vector per fusion-tree key.
    let one = cold_lowered_allocations(1);
    let sixteen = cold_lowered_allocations(16);
    let encoded_one = cold_encoded_allocations(1);
    let encoded_sixteen = cold_encoded_allocations(16);

    // The encoded builder is the established #256 single-pass baseline. The
    // checked enumeration may change the constant term, but must not add a
    // steeper allocation slope as the tree count grows.
    assert!(sixteen - one <= encoded_sixteen - encoded_one);
}
