use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tenet_core::{
    BlockKey, BlockStructure, DegeneracyStructure, FusionTreeBlockKey, FusionTreeKey,
    SU2FusionRule, SectorId, SectorStructure,
};
use tenet_tensors::{
    build_all_codomain_tree_transform_group_plan, build_tree_pair_transform_group_plan,
    TreeTransformOperation,
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

fn su2_f_move_structure() -> BlockStructure {
    let keys = [[0, 1], [2, 1]].map(|inner| {
        BlockKey::from(FusionTreeBlockKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [SectorId::new(1); 4],
                Some(SectorId::new(0)),
                [false; 4],
                inner.map(SectorId::new),
                [SectorId::new(1); 3],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&SU2FusionRule, [], Some(SectorId::new(0)), [], [], [])
                .unwrap(),
        ))
    });
    BlockStructure::from_parts(
        SectorStructure::from_keys(4, keys).unwrap(),
        DegeneracyStructure::packed_column_major(4, [vec![1; 4], vec![1; 4]]).unwrap(),
    )
    .unwrap()
}

#[test]
fn su2_f_move_compile_has_no_per_destination_coefficient_rows() {
    let structure = su2_f_move_structure();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let _ =
        build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation.clone(), &structure)
            .unwrap();

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let plan = build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation, &structure)
        .unwrap();
    COUNTING.set(false);

    // What: compiling a two-channel SU(2) F move owns one final coefficient
    // matrix without allocating one temporary coefficient Vec per destination.
    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src().len(), 4);
    assert!(ALLOCATIONS.get() <= 32, "allocations={}", ALLOCATIONS.get());
}

#[test]
fn su2_tree_pair_f_move_compile_has_no_per_destination_coefficient_rows() {
    let structure = su2_f_move_structure();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let _ = build_tree_pair_transform_group_plan(&SU2FusionRule, operation.clone(), &structure)
        .unwrap();

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let plan = build_tree_pair_transform_group_plan(&SU2FusionRule, operation, &structure).unwrap();
    COUNTING.set(false);

    // What: the tree-pair assembler owns one row-major coefficient matrix for
    // the same two-channel F move, independently of the all-codomain builder.
    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src().len(), 4);
    assert!(ALLOCATIONS.get() <= 52, "allocations={}", ALLOCATIONS.get());
}
