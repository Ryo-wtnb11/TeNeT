use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
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
    reset_global_operation_caches, RuleIdentity, TreeTransformBlockSpec, TreeTransformCache,
    TreeTransformOperation, TreeTransformStructure,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

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
fn rank_129_second_exact_warm_structure_hit_has_no_operation_key_allocation_or_provider_work() {
    let _global_cache_guard = counting_alloc::serial();
    reset_global_operation_caches();

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
    };
    let structure = rank_129_su2_vacuum_structure();
    let operation = TreeTransformOperation::permute(0..129, []);
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::default();
    let cold = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &rule, &operation, &structure, &structure, false,
        )
        .unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);

    calls.store(0, Ordering::Relaxed);
    counting_alloc::start();
    let warm = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &rule, &operation, &structure, &structure, false,
        )
        .unwrap();
    let allocs = counting_alloc::stop();

    // What: cloning the runtime-rank operation into the completed-structure
    // lookup key performs no allocation or provider work on an exact warm hit.
    assert!(Arc::ptr_eq(&cold, &warm));
    assert_eq!(allocs.calls, 0);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn su2_f_move_compile_has_no_per_destination_coefficient_rows() {
    let structure = su2_f_move_structure();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let _ =
        build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation.clone(), &structure)
            .unwrap();
    counting_alloc::start();
    let plan = build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation, &structure)
        .unwrap();
    let allocs = counting_alloc::stop();

    // What: compiling a two-channel SU(2) F move owns one final coefficient
    // matrix without allocating one temporary coefficient Vec per destination.
    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src().len(), 4);
    assert!(allocs.calls <= 32, "allocations={}", allocs.calls);
}

#[test]
fn su2_tree_pair_f_move_compile_has_no_per_destination_coefficient_rows() {
    let structure = su2_f_move_structure();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let _ = build_tree_pair_transform_group_plan(&SU2FusionRule, operation.clone(), &structure)
        .unwrap();
    counting_alloc::start();
    let plan = build_tree_pair_transform_group_plan(&SU2FusionRule, operation, &structure).unwrap();
    let allocs = counting_alloc::stop();

    // What: the tree-pair assembler owns one row-major coefficient matrix for
    // the same two-channel F move, independently of the all-codomain builder.
    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src().len(), 4);
    assert!(allocs.calls <= 52, "allocations={}", allocs.calls);
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
fn cold_ordered_tree_pair_compile_stays_within_allocation_envelopes() {
    let _global_cache_guard = counting_alloc::serial();

    // What: exact counts cover missing-position plan compilation after registry
    // capacity exists, independently of unrelated typed-cache test order.
    reset_global_operation_caches();
    let (dst, src) = rank_eight_su2_subset(1);
    TreeTransformCache::<f64, RuleIdentity>::new()
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
        let mut cache = TreeTransformCache::<f64, RuleIdentity>::new();
        cache.set_recoupling_threads(1);
        counting_alloc::start();
        let plan = cache
            .get_or_compile_tree_pair(
                &SU2FusionRule,
                TreeTransformOperation::permute(0..8, []),
                &dst,
                &src,
            )
            .unwrap();
        let allocs = counting_alloc::stop();

        // What: the ordered whole-block compiler stays within its prior cold
        // envelope plus one bounded all-rank normalization workspace.
        assert!(
            (allocs.calls as usize) <= expected_allocations + 3,
            "source_count={source_count}, allocations={}",
            allocs.calls
        );
        std::hint::black_box(plan);
    }
}

#[test]
fn rank_nine_same_split_groups_do_not_clone_prepared_spill_storage() {
    let structure = Arc::new(rank_nine_same_split_su2_groups());
    let operation = TreeTransformOperation::braid([1, 0, 2, 3, 4, 5, 6, 7, 8], [], 0..9, []);
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::new();
    cache.set_recoupling_threads(1);
    counting_alloc::start();
    let compiled = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    let allocs = counting_alloc::stop();

    // What: three same-split groups do not regress beyond the prior compiled
    // allocation and byte envelopes.
    assert_eq!(structure.fusion_tree_groups().len(), 3);
    assert!(allocs.calls <= 217);
    assert!(allocs.bytes < 56_500);
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
        counting_alloc::start();
        let plan =
            build_tree_pair_transform_group_plan(&U1FusionRule, operation, &structure).unwrap();
        let allocs = counting_alloc::stop();

        // What: every Unique source block is an inline Single and shares the
        // operation axis map, so plan construction owns only the outer specs
        // allocation and the shared axis allocation at every cardinality.
        assert_eq!(allocs.calls, 2, "source_blocks={count}");
        assert_eq!(plan.specs().len(), count);

        let indexed_specs = (0..count)
            .map(|index| TreeTransformBlockSpec::single(index, index, 1.0).with_source_axes([0, 1]))
            .collect::<Vec<_>>();
        counting_alloc::start();
        let direct =
            TreeTransformStructure::compile_structures(&structure, &structure, &indexed_specs)
                .unwrap();
        let allocs = counting_alloc::stop();
        let direct_allocations = allocs.calls as usize;
        counting_alloc::start();
        let grouped = plan.compile_structures(&structure, &structure).unwrap();
        let allocs = counting_alloc::stop();
        let grouped_allocations = allocs.calls as usize;

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
    counting_alloc::start();
    let _ = grouped_plan
        .compile_structures(&structure, &structure)
        .unwrap();
    let allocs = counting_alloc::stop();
    let cold_grouped_allocations = allocs.calls as usize;
    let cold_grouped_bytes = allocs.bytes as usize;
    let _ =
        TreeTransformStructure::compile_structures(&structure, &structure, &direct_specs).unwrap();
    counting_alloc::start();
    let direct =
        TreeTransformStructure::compile_structures(&structure, &structure, &direct_specs).unwrap();
    let allocs = counting_alloc::stop();
    let direct_allocations = allocs.calls as usize;
    let direct_bytes = allocs.bytes as usize;
    counting_alloc::start();
    let grouped = grouped_plan
        .compile_structures(&structure, &structure)
        .unwrap();
    let allocs = counting_alloc::stop();
    let grouped_allocations = allocs.calls as usize;
    let grouped_bytes = allocs.bytes as usize;

    // What: the first categorical binding references the spec's matrix and
    // allocates no coefficient buffer beyond what a direct binding does.
    assert!(
        cold_grouped_allocations <= direct_allocations + 16,
        "cold_grouped_allocations={cold_grouped_allocations}, direct_allocations={direct_allocations}"
    );
    assert!(
        cold_grouped_bytes <= direct_bytes + 32 * 1024,
        "cold_grouped_bytes={cold_grouped_bytes}, direct_bytes={direct_bytes}"
    );

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
        grouped.gathered_coefficients(),
        direct.gathered_coefficients()
    );
}

/// `[s, s, s] <- [d]` with `s = {1/2, 1}` and the domain leg's sectors `d`.
fn su2_three_to_one_structure(domain_twice_spins: &[usize]) -> Arc<BlockStructure> {
    let leg = |twice_spins: &[usize]| {
        SectorLeg::new(
            twice_spins
                .iter()
                .map(|&j| (SU2Irrep::from_twice_spin(j).sector_id(), 1)),
            false,
        )
    };
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(&[1, 2]), leg(&[1, 2]), leg(&[1, 2])]),
        FusionProductSpace::new([leg(domain_twice_spins)]),
    );
    let keys = hom
        .fusion_tree_keys(&SU2FusionRule)
        .iter()
        .cloned()
        .map(BlockKey::from)
        .collect::<Vec<_>>();
    let count = keys.len();
    Arc::new(
        BlockStructure::from_parts(
            SectorStructure::from_keys(4, keys).unwrap(),
            DegeneracyStructure::packed_column_major(4, (0..count).map(|_| vec![1usize; 4]))
                .unwrap(),
        )
        .unwrap(),
    )
}

fn convert_spec<T: Copy>(
    spec: &tenet_tensors::TreeTransformGroupBlockSpec<f64>,
    convert: fn(f64) -> T,
) -> tenet_tensors::TreeTransformGroupBlockSpec<T> {
    let coefficients = spec
        .recoupling_coefficients_dst_src()
        .iter()
        .map(|&value| convert(value))
        .collect::<Vec<_>>();
    let converted = if coefficients.len() == 1 {
        tenet_tensors::TreeTransformGroupBlockSpec::single(
            spec.dst_keys()[0].clone(),
            spec.src_keys()[0].clone(),
            coefficients[0],
        )
    } else {
        tenet_tensors::TreeTransformGroupBlockSpec::try_multi(
            spec.dst_keys().iter().cloned(),
            spec.src_keys().iter().cloned(),
            coefficients,
        )
        .unwrap()
    };
    match spec.source_axes() {
        Some(axes) => converted.with_source_axes(axes.iter().copied()),
        None => converted,
    }
}

/// The `c` plan a warm Runtime assembles after a sector change: every group
/// `a` already built is the cached spec handle (`plan.rs` clones group specs),
/// and only the added groups are new. Returns the plan, the coefficient count
/// of the added groups, and the number of Single (1x1) specs.
fn reused_sector_change_plan<T: Copy>(
    a: &tenet_tensors::TreeTransformGroupPlan<f64>,
    c: &tenet_tensors::TreeTransformGroupPlan<f64>,
    convert: fn(f64) -> T,
) -> (tenet_tensors::TreeTransformGroupPlan<T>, usize, usize) {
    let cached = a
        .specs()
        .iter()
        .map(|spec| convert_spec(spec, convert))
        .collect::<Vec<_>>();
    let mut changed_coefficients = 0;
    let mut singles = 0;
    let specs = c
        .specs()
        .iter()
        .map(|spec| {
            if spec.recoupling_coefficients_dst_src().len() == 1 {
                singles += 1;
            }
            cached
                .iter()
                .find(|hit| {
                    hit.group_key() == spec.group_key()
                        && hit.src_keys() == spec.src_keys()
                        && hit.dst_keys() == spec.dst_keys()
                })
                .cloned()
                .unwrap_or_else(|| {
                    changed_coefficients += spec.recoupling_coefficients_dst_src().len();
                    convert_spec(spec, convert)
                })
        })
        .collect();
    (
        tenet_tensors::TreeTransformGroupPlan::new(specs),
        changed_coefficients,
        singles,
    )
}

#[test]
fn sector_change_binding_copies_coefficients_of_changed_groups_only() {
    // Why only operations that keep the space: the binding needs the
    // destination structure, and these map `[s, s, s] <- [d]` onto itself.
    for operation in [
        TreeTransformOperation::permute([0, 2, 1], [3]),
        TreeTransformOperation::braid([0, 2, 1], [3], [0, 1, 2], [3]),
    ] {
        assert_sector_change_binding_copies_only_changed_groups(operation);
    }
}

fn assert_sector_change_binding_copies_only_changed_groups(operation: TreeTransformOperation) {
    let a = su2_three_to_one_structure(&[1, 2]);
    let c = su2_three_to_one_structure(&[1, 2, 3]);
    let plan_a =
        build_tree_pair_transform_group_plan(&SU2FusionRule, operation.clone(), &a).unwrap();
    let plan_c = build_tree_pair_transform_group_plan(&SU2FusionRule, operation, &c).unwrap();

    let bind_bytes = |bind: &dyn Fn()| {
        counting_alloc::start();
        bind();
        let allocs = counting_alloc::stop();
        allocs.bytes as usize
    };
    // Why compare two scalar widths: the bound plans have identical keys,
    // blocks and layouts, so their binding allocations differ only by the
    // coefficient values the binding stores, 8 extra bytes per copied value.
    let (real, changed, singles) = reused_sector_change_plan(&plan_a, &plan_c, |value| value);
    let (complex, _, _) = reused_sector_change_plan(&plan_a, &plan_c, |value| {
        num_complex::Complex64::new(value, 0.0)
    });
    let total = plan_c
        .specs()
        .iter()
        .map(|spec| spec.recoupling_coefficients_dst_src().len())
        .sum::<usize>();
    assert!(
        changed > 0 && total > changed + singles,
        "fixture must reuse matrix groups"
    );

    let bind_real = || {
        std::hint::black_box(
            real.compile_shared_structures_with_storage_conjugation(
                Arc::clone(&c),
                Arc::clone(&c),
                false,
            )
            .unwrap(),
        );
    };
    let bind_complex = || {
        std::hint::black_box(
            complex
                .compile_shared_structures_with_storage_conjugation(
                    Arc::clone(&c),
                    Arc::clone(&c),
                    false,
                )
                .unwrap(),
        );
    };
    let real_bytes = bind_bytes(&bind_real);
    let complex_bytes = bind_bytes(&bind_complex);
    let copied = complex_bytes.saturating_sub(real_bytes) / 8;

    // What: a plan miss copies coefficients of the groups it built plus the
    // Single scalars, never the reused groups' recoupling matrices.
    assert!(
        copied <= changed + singles,
        "copied={copied} coefficients, changed={changed}, singles={singles}, total={total}"
    );

    // What: binding the already-bound plan again (a structure-tier miss on a
    // plan-tier hit) shares the plan's payload and copies no coefficient.
    let rebound_real = bind_bytes(&bind_real);
    let rebound_complex = bind_bytes(&bind_complex);
    assert_eq!(
        rebound_complex, rebound_real,
        "a re-binding must not store any coefficient"
    );
}

/// Test view of the logical coefficient payload (an explicit copy).
trait GatheredCoefficients<T> {
    fn gathered_coefficients(&self) -> Vec<T>;
}

impl<T: Copy> GatheredCoefficients<T> for tenet_tensors::TreeTransformStructure<T> {
    fn gathered_coefficients(&self) -> Vec<T> {
        let mut coefficients = Vec::new();
        self.gather_recoupling_coefficients_into(&mut coefficients);
        coefficients
    }
}
