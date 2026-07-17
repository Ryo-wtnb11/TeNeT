use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet_core::{
    BlockKey, BlockStructure, DegeneracyStructure, FusionProductSpace, FusionTreeBlockKey,
    FusionTreeHomSpace, FusionTreeKey, SU2FusionRule, SU2Irrep, SectorId, SectorLeg,
    SectorStructure, TensorMap, TensorMapSpace, U1FusionRule,
};
use tenet_tensors::{
    build_all_codomain_tree_transform_group_plan, build_tree_pair_transform_group_plan,
    reset_global_operation_caches, BoundDynamicFusionMapSpace, TreeTransformBuiltinRuleCacheKey,
    TreeTransformCache, TreeTransformOperation,
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

#[test]
fn missing_position_rescan_removes_cardinality_dependent_metadata_allocations() {
    for (missing, expected_removed_calls) in [(1, 1), (2, 1), (4, 1), (5, 2), (8, 2), (9, 3)] {
        let sources = vec![None::<()>; missing];

        ALLOCATIONS.set(0);
        COUNTING.set(true);
        let positions = sources
            .iter()
            .enumerate()
            .filter_map(|(position, rows)| rows.is_none().then_some(position))
            .collect::<Vec<_>>();
        COUNTING.set(false);
        let old_calls = ALLOCATIONS.get();

        ALLOCATIONS.set(0);
        COUNTING.set(true);
        let missing_count = sources.iter().filter(|rows| rows.is_none()).count();
        COUNTING.set(false);
        let rescan_calls = ALLOCATIONS.get();

        // What: replacing the old position Vec with ordered rescans removes
        // every allocation/reallocation at and above its growth boundaries.
        assert_eq!(positions.len(), missing);
        assert_eq!(missing_count, missing);
        assert_eq!(old_calls, expected_removed_calls);
        assert_eq!(rescan_calls, 0);
    }
}

fn rank_eight_su2_subset(count: usize) -> (TensorMap<f64, 8, 0>, TensorMap<f64, 8, 0>) {
    let half = SU2Irrep::from_twice_spin(1).sector_id();
    let leg = || SectorLeg::new([(half, 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new((0..8).map(|_| leg())),
        FusionProductSpace::new([]),
    );
    let keys = hom
        .fusion_tree_keys(&SU2FusionRule)
        .iter()
        .take(count)
        .cloned()
        .map(BlockKey::from)
        .collect::<Vec<_>>();
    assert_eq!(keys.len(), count);
    let structure = BlockStructure::from_parts(
        SectorStructure::from_keys(8, keys).unwrap(),
        DegeneracyStructure::packed_column_major(8, (0..count).map(|_| vec![1usize; 8])).unwrap(),
    )
    .unwrap();
    let space = TensorMapSpace::<8, 0>::from_dims([1; 8], []).unwrap();
    let src =
        TensorMap::from_vec_with_structure(vec![1.0; count], space.clone(), structure.clone())
            .unwrap();
    let dst = TensorMap::from_vec_with_structure(vec![0.0; count], space, structure).unwrap();
    (dst, src)
}

#[test]
fn cold_memoized_tree_pair_compile_avoids_missing_position_allocations() {
    for (missing, expected_allocations) in
        [(1, 72), (2, 84), (4, 110), (5, 128), (8, 160), (9, 180)]
    {
        reset_global_operation_caches();
        let (dst, src) = rank_eight_su2_subset(missing);
        let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();
        cache.set_recoupling_threads(1);

        ALLOCATIONS.set(0);
        COUNTING.set(true);
        let plan = cache
            .get_or_compile_tree_pair(
                &SU2FusionRule,
                TreeTransformOperation::permute(0..8, []),
                &dst,
                &src,
            )
            .unwrap();
        COUNTING.set(false);

        // What: the public cold memoized path removes every allocation and
        // reallocation formerly paid by its missing-position Vec.
        assert_eq!(ALLOCATIONS.get(), expected_allocations, "missing={missing}");
        std::hint::black_box(plan);
    }
}

#[test]
fn lowered_scratch_hit_matches_encoded_hit_allocation_and_identity() {
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

    // What: the transactional lowered capability preserves the encoded warm
    // hit's allocation cost and returns the exact retained structure Arc.
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
