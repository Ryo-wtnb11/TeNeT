use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use tenet_core::{
    BlockKey, BlockStructure, BraidingStyleKind, DegeneracyStructure, FusionProductSpace,
    FusionRule, FusionStyleKind, FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    MultiplicityIndex, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, SectorStructure, SectorVec,
    TensorMap, TensorMapSpace, U1FusionRule, U1Irrep,
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

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// Why not rely on thread-local allocation counters alone: categorical plan
// compilation and reset still mutate shared process-global cache state.
static GLOBAL_CACHE_RESET_LOCK: Mutex<()> = Mutex::new(());

fn su2_f_move_structure() -> BlockStructure {
    let keys = [[0, 1], [2, 1]].map(|inner| {
        BlockKey::from(FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [SectorId::new(1); 4],
                SectorId::new(0),
                [false; 4],
                inner.map(SectorId::new),
                [MultiplicityIndex::ONE; 3],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&SU2FusionRule, [], SectorId::new(0), [], [], [])
                .unwrap(),
        ))
    });
    BlockStructure::from_parts(
        SectorStructure::from_keys(4, keys).unwrap(),
        DegeneracyStructure::packed_column_major(4, [vec![1; 4], vec![1; 4]]).unwrap(),
    )
    .unwrap()
}

fn rank_nine_same_split_su2_groups() -> BlockStructure {
    let vacuum = SU2FusionRule.vacuum();
    let keys = [0usize, 1, 2].map(|twice_spin| {
        let mut uncoupled = [vacuum; 9];
        uncoupled[0] = SectorId::new(twice_spin);
        uncoupled[1] = SectorId::new(twice_spin);
        BlockKey::from(FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                uncoupled,
                vacuum,
                [false; 9],
                [vacuum; 7],
                [MultiplicityIndex::ONE; 8],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&SU2FusionRule, [], vacuum, [], [], []).unwrap(),
        ))
    });
    BlockStructure::from_parts(
        SectorStructure::from_keys(9, keys).unwrap(),
        DegeneracyStructure::packed_column_major(9, std::array::from_fn::<_, 3, _>(|_| vec![1; 9]))
            .unwrap(),
    )
    .unwrap()
}

#[derive(Clone)]
struct AdmissionCountingSu2Rule {
    nsymbol_calls: Arc<AtomicUsize>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AdmissionCountingSu2CacheKey;

impl FusionRule for AdmissionCountingSu2Rule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        SU2FusionRule.rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        SU2FusionRule.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        SU2FusionRule.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        SU2FusionRule.vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        SU2FusionRule.dual(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        SU2FusionRule.fusion_channels(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.nsymbol_calls.fetch_add(1, Ordering::Relaxed);
        SU2FusionRule.nsymbol(left, right, coupled)
    }
}

impl MultiplicityFreeFusionRule for AdmissionCountingSu2Rule {}

impl MultiplicityFreeFusionSymbols for AdmissionCountingSu2Rule {
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        SU2FusionRule.scalar_one()
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        SU2FusionRule.scalar_conj(value)
    }

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        SU2FusionRule.f_symbol_scalar(left, middle, right, coupled, left_coupled, right_coupled)
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        SU2FusionRule.r_symbol_scalar(left, right, coupled)
    }
}

impl MultiplicityFreeRigidSymbols for AdmissionCountingSu2Rule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.dim_scalar(sector)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.inv_dim_scalar(sector)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.sqrt_dim_scalar(sector)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.inv_sqrt_dim_scalar(sector)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.twist_scalar(sector)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.frobenius_schur_phase_scalar(sector)
    }
}

impl tenet_tensors::TreeTransformRuleCacheKey for AdmissionCountingSu2Rule {
    type Key = AdmissionCountingSu2CacheKey;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        AdmissionCountingSu2CacheKey
    }
}

fn rank_129_su2_vacuum_structure() -> Arc<BlockStructure> {
    const RANK: usize = 129;

    let vacuum = SectorId::new(0);
    let key = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            vec![vacuum; RANK],
            vacuum,
            vec![false; RANK],
            vec![vacuum; RANK - 2],
            vec![MultiplicityIndex::ONE; RANK - 1],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&SU2FusionRule, [], vacuum, [], [], []).unwrap(),
    );
    Arc::new(
        BlockStructure::from_parts(
            SectorStructure::from_keys(RANK, [BlockKey::from(key)]).unwrap(),
            DegeneracyStructure::packed_column_major(RANK, [vec![1; RANK]]).unwrap(),
        )
        .unwrap(),
    )
}

#[test]
fn rank_129_second_exact_warm_structure_hit_is_allocation_and_provider_free() {
    let _global_cache_guard = GLOBAL_CACHE_RESET_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_global_operation_caches();

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
    };
    let structure = rank_129_su2_vacuum_structure();
    let operation = TreeTransformOperation::permute(0..129, []);
    let mut cache = TreeTransformCache::<f64, AdmissionCountingSu2CacheKey>::default();
    let cold = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &rule, &operation, &structure, &structure, false,
        )
        .unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);

    calls.store(0, Ordering::Relaxed);
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let warm = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &rule, &operation, &structure, &structure, false,
        )
        .unwrap();
    COUNTING.set(false);

    // What: a dynamic-rank categorical replay reuses the exact structural
    // admission and compiled plan without rank-dependent scratch or providers.
    assert!(Arc::ptr_eq(&cold, &warm));
    assert_eq!(ALLOCATIONS.get(), 0);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
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
fn cold_ordered_tree_pair_compile_has_stable_allocation_counts() {
    let _global_cache_guard = GLOBAL_CACHE_RESET_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // What: exact counts cover missing-position plan compilation after registry
    // capacity exists, independently of unrelated typed-cache test order.
    reset_global_operation_caches();
    let (dst, src) = rank_eight_su2_subset(1);
    TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new()
        .get_or_compile_tree_pair(
            &SU2FusionRule,
            TreeTransformOperation::permute(0..8, []),
            &dst,
            &src,
        )
        .unwrap();
    reset_global_operation_caches();

    for (source_count, expected_allocations) in
        [(1, 41), (2, 46), (4, 54), (5, 62), (8, 68), (9, 76)]
    {
        reset_global_operation_caches();
        let (dst, src) = rank_eight_su2_subset(source_count);
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

        // What: the ordered whole-block compiler retains its measured cold
        // allocation envelope without a source-column memo or position list.
        assert_eq!(
            ALLOCATIONS.get(),
            expected_allocations,
            "source_count={source_count}"
        );
        std::hint::black_box(plan);
    }
}

#[test]
fn rank_nine_same_split_groups_do_not_clone_prepared_spill_storage() {
    let structure = Arc::new(rank_nine_same_split_su2_groups());
    let operation = TreeTransformOperation::braid([1, 0, 2, 3, 4, 5, 6, 7, 8], [], 0..9, []);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();
    cache.set_recoupling_threads(1);

    ALLOCATIONS.set(0);
    ALLOCATED_BYTES.set(0);
    COUNTING.set(true);
    let compiled = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    COUNTING.set(false);

    // What: three same-split groups reuse one rank-nine prepared operation;
    // cloning spilled prepared storage exceeds both allocation envelopes.
    assert_eq!(structure.fusion_tree_groups().len(), 3);
    assert_eq!(ALLOCATIONS.get(), 217);
    assert!(ALLOCATED_BYTES.get() < 56_500);
    std::hint::black_box(compiled);
}

fn rank_one_u1_pair_structure(count: usize) -> BlockStructure {
    let keys = (0..count).map(|charge| {
        let sector = U1Irrep::new(charge as i32).sector_id();
        BlockKey::from(FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(&U1FusionRule, [sector], sector, [false], [], [])
                .unwrap(),
            FusionTreeKey::try_new_for_rule(&U1FusionRule, [sector], sector, [false], [], [])
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

        // What: resolving grouped Single entries owns one descriptor arena but
        // borrows coefficients and shared source axes from the plan.
        assert_eq!(
            grouped_allocations,
            direct_allocations + 1,
            "source_blocks={count}"
        );
        assert_eq!(grouped, direct);
    }
}

#[test]
fn grouped_multi_compile_borrows_plan_coefficient_matrix() {
    const BLOCKS: usize = 2;
    const COEFFICIENT_BYTES: usize = 64 * 1024;

    let structure = su2_f_move_structure();
    let keys = (0..BLOCKS)
        .map(|block| {
            structure
                .block(block)
                .unwrap()
                .key()
                .as_fusion_tree_pair()
                .unwrap()
                .clone()
        })
        .collect::<Vec<_>>();
    let coefficients = vec![[1_u8; COEFFICIENT_BYTES]; BLOCKS * BLOCKS];
    let grouped_spec = tenet_tensors::TreeTransformGroupBlockSpec::try_multi(
        keys.clone(),
        keys,
        coefficients.clone(),
    )
    .unwrap();
    let grouped_plan = tenet_tensors::TreeTransformGroupPlan::new(vec![grouped_spec]);
    let direct_specs = [TreeTransformBlockSpec::multi(
        (0..BLOCKS).collect(),
        (0..BLOCKS).collect(),
        coefficients,
    )];

    let _ = grouped_plan
        .compile_structures(&structure, &structure)
        .unwrap();
    let _ =
        TreeTransformStructure::compile_structures(&structure, &structure, &direct_specs).unwrap();

    ALLOCATIONS.set(0);
    ALLOCATED_BYTES.set(0);
    COUNTING.set(true);
    let direct =
        TreeTransformStructure::compile_structures(&structure, &structure, &direct_specs).unwrap();
    COUNTING.set(false);
    let direct_allocations = ALLOCATIONS.get();
    let direct_bytes = ALLOCATED_BYTES.get();

    ALLOCATIONS.set(0);
    ALLOCATED_BYTES.set(0);
    COUNTING.set(true);
    let grouped = grouped_plan
        .compile_structures(&structure, &structure)
        .unwrap();
    COUNTING.set(false);
    let grouped_allocations = ALLOCATIONS.get();
    let grouped_bytes = ALLOCATED_BYTES.get();

    // What: grouped key resolution may own index/descriptor scratch, but it
    // does not allocate the 256 KiB coefficient matrix a second time.
    assert!(
        grouped_allocations <= direct_allocations + 16,
        "grouped_allocations={grouped_allocations}, direct_allocations={direct_allocations}"
    );
    assert!(
        grouped_bytes <= direct_bytes + 32 * 1024,
        "grouped_bytes={grouped_bytes}, direct_bytes={direct_bytes}"
    );
    assert_eq!(
        grouped.recoupling_coefficients_dst_src(),
        direct.recoupling_coefficients_dst_src()
    );
}
