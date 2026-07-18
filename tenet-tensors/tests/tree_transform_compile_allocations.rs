use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tenet_core::{
    BlockKey, BlockStructure, DegeneracyStructure, FusionProductSpace, FusionTreeBlockKey,
    FusionTreeHomSpace, FusionTreeKey, SU2FusionRule, SU2Irrep, SectorId, SectorLeg,
    SectorStructure, TensorMap, TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet_tensors::{
    build_all_codomain_tree_transform_group_plan, build_tree_pair_transform_group_plan,
    reset_global_operation_caches, TreeTransformBlockSpec, TreeTransformBuiltinRuleCacheKey,
    TreeTransformCache, TreeTransformOperation, TreeTransformStructure,
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
    for (missing, expected_allocations) in [(1, 64), (2, 68), (4, 78), (5, 87), (8, 95), (9, 106)] {
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

fn rank_one_u1_pair_structure(count: usize) -> BlockStructure {
    let keys = (0..count).map(|charge| {
        let sector = U1Irrep::new(charge as i32).sector_id();
        BlockKey::from(FusionTreeBlockKey::pair(
            FusionTreeKey::try_new_for_rule(&U1FusionRule, [sector], Some(sector), [false], [], [])
                .unwrap(),
            FusionTreeKey::try_new_for_rule(&U1FusionRule, [sector], Some(sector), [false], [], [])
                .unwrap(),
        ))
    });
    BlockStructure::from_parts(
        SectorStructure::from_keys(2, keys).unwrap(),
        DegeneracyStructure::packed_column_major(2, (0..count).map(|_| vec![1, 1])).unwrap(),
    )
    .unwrap()
}

#[test]
fn unique_rank_one_u1_plan_allocations_do_not_scale_with_source_blocks() {
    for count in [1, 2, 4, 8, 16] {
        let structure = rank_one_u1_pair_structure(count);
        let operation = TreeTransformOperation::permute([0], [1]);

        ALLOCATIONS.set(0);
        COUNTING.set(true);
        let plan =
            build_tree_pair_transform_group_plan(&U1FusionRule, operation, &structure).unwrap();
        COUNTING.set(false);

        // What: every Unique source block is an inline Single and shares the
        // operation axis map, so plan construction owns only the outer specs
        // allocation and the shared axis allocation at every cardinality.
        assert_eq!(ALLOCATIONS.get(), 2, "source_blocks={count}");
        assert_eq!(plan.specs().len(), count);

        let indexed_specs = (0..count)
            .map(|index| TreeTransformBlockSpec::single(index, index, 1.0).with_source_axes([0, 1]))
            .collect::<Vec<_>>();

        ALLOCATIONS.set(0);
        COUNTING.set(true);
        let direct =
            TreeTransformStructure::compile_structures(&structure, &structure, &indexed_specs)
                .unwrap();
        COUNTING.set(false);
        let direct_allocations = ALLOCATIONS.get();

        ALLOCATIONS.set(0);
        COUNTING.set(true);
        let grouped = plan.compile_structures(&structure, &structure).unwrap();
        COUNTING.set(false);
        let grouped_allocations = ALLOCATIONS.get();

        // What: grouped lowering adds exactly its preallocated indexed-spec
        // vector. Key ownership in the structures and the final replay arena
        // are paid equally by both compilation paths.
        assert_eq!(
            grouped_allocations,
            direct_allocations + 1,
            "source_blocks={count}"
        );
        assert_eq!(
            grouped.recoupling_coefficients_dst_src(),
            direct.recoupling_coefficients_dst_src()
        );
    }
}
