use std::hint::black_box;
use std::time::{Duration, Instant};

use tenet_core::{
    product_fusion_rule, BlockKey, BlockStructure, FermionParityFusionRule, FusionProductSpace,
    FusionTensorMapSpace, FusionTreeBlockKey, FusionTreeHomSpace, FusionTreeKey,
    ProductFusionRuleExt, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, TensorMap, TensorMapSpace,
    U1FusionRule, U1Irrep,
};
use tenet_operations::{
    tree_pair_transform_into, tree_pair_transform_into_with, tree_pair_transform_into_with_context,
    tree_pair_transform_structure, tree_transform_execute_with, DenseTreeTransformOperations,
    TreeTransformBuiltinRuleCacheKey, TreeTransformExecutionContext, TreeTransformOperationKey,
    TreeTransformRuleCacheKey, TreeTransformWorkspace,
};

fn main() {
    const BUILD_ITERS: usize = 100_000;
    const REPLAY_ITERS: usize = 1_000_000;
    const CONTEXT_ITERS: usize = 1_000_000;
    const REBUILD_EXECUTE_ITERS: usize = 100_000;
    const ONESHOT_ITERS: usize = 10_000;

    let product = bench_product(
        BUILD_ITERS,
        REPLAY_ITERS,
        CONTEXT_ITERS,
        REBUILD_EXECUTE_ITERS,
        ONESHOT_ITERS,
    );
    let su2 = bench_su2(
        BUILD_ITERS,
        REPLAY_ITERS,
        CONTEXT_ITERS,
        REBUILD_EXECUTE_ITERS,
        ONESHOT_ITERS,
    );

    println!("tree transform replay timing (release)");
    print_fixture(
        "product",
        product,
        BUILD_ITERS,
        REPLAY_ITERS,
        CONTEXT_ITERS,
        REBUILD_EXECUTE_ITERS,
        ONESHOT_ITERS,
    );
    print_fixture(
        "su2",
        su2,
        BUILD_ITERS,
        REPLAY_ITERS,
        CONTEXT_ITERS,
        REBUILD_EXECUTE_ITERS,
        ONESHOT_ITERS,
    );
}

#[derive(Clone, Copy)]
struct Timings {
    compile: Duration,
    replay: Duration,
    context_hit_execute: Duration,
    rebuild_execute: Duration,
    oneshot: Duration,
}

fn nanos_per(duration: Duration, iterations: usize) -> f64 {
    duration.as_secs_f64() * 1.0e9 / iterations as f64
}

fn print_fixture(
    name: &str,
    timings: Timings,
    build_iters: usize,
    replay_iters: usize,
    context_iters: usize,
    rebuild_execute_iters: usize,
    oneshot_iters: usize,
) {
    let compile_ns = nanos_per(timings.compile, build_iters);
    let replay_ns = nanos_per(timings.replay, replay_iters);
    let context_hit_execute_ns = nanos_per(timings.context_hit_execute, context_iters);
    let rebuild_execute_ns = nanos_per(timings.rebuild_execute, rebuild_execute_iters);
    let oneshot_ns = nanos_per(timings.oneshot, oneshot_iters);
    println!("{name} compile avg: {:>10.3} ns", compile_ns);
    println!("{name} replay  avg: {:>10.3} ns", replay_ns);
    println!(
        "{name} context cache-hit+execute avg: {:>10.3} ns",
        context_hit_execute_ns
    );
    println!(
        "{name} rebuild+execute avg: {:>10.3} ns",
        rebuild_execute_ns
    );
    println!("{name} one-shot avg: {:>10.3} ns", oneshot_ns);
    println!(
        "{name} compile/replay ratio: {:>8.2}x",
        compile_ns / replay_ns
    );
    println!(
        "{name} context/replay ratio: {:>8.2}x",
        context_hit_execute_ns / replay_ns
    );
    println!(
        "{name} rebuild+execute/replay ratio: {:>8.2}x",
        rebuild_execute_ns / replay_ns
    );
}

fn bench_product(
    build_iters: usize,
    replay_iters: usize,
    context_iters: usize,
    rebuild_execute_iters: usize,
    oneshot_iters: usize,
) -> Timings {
    let (rule, src_space, dst_space) = product_fixture();
    let operation = TreeTransformOperationKey::permute([1, 0], [2]);
    let mut src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();

    let compile = time_loop(build_iters, || {
        let structure =
            tree_pair_transform_structure(&rule, operation.clone(), &dst, &src).unwrap();
        black_box(structure);
    });

    let structure = tree_pair_transform_structure(&rule, operation.clone(), &dst, &src).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    let replay = time_loop(replay_iters, || {
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            black_box(1.0),
            black_box(0.0),
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    let mut rebuild_backend = DenseTreeTransformOperations::default();
    let mut rebuild_workspace = TreeTransformWorkspace::default();
    let rebuild_execute = time_loop(rebuild_execute_iters, || {
        src.data_mut().copy_from_slice(&[10.0, 20.0]);
        tree_pair_transform_into_with(
            &mut rebuild_backend,
            &mut rebuild_workspace,
            &rule,
            operation.clone(),
            &mut dst,
            &src,
            black_box(1.0),
            black_box(0.0),
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    type ProductRuleKey = <ProductRule as TreeTransformRuleCacheKey>::Key;
    let mut context = TreeTransformExecutionContext::<f64, ProductRuleKey>::default();
    tree_pair_transform_into_with_context(
        &mut context,
        &rule,
        operation.clone(),
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    let context_hit_execute = time_loop(context_iters, || {
        src.data_mut().copy_from_slice(&[10.0, 20.0]);
        tree_pair_transform_into_with_context(
            &mut context,
            &rule,
            operation.clone(),
            &mut dst,
            &src,
            black_box(1.0),
            black_box(0.0),
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    let oneshot = time_loop(oneshot_iters, || {
        src.data_mut().copy_from_slice(&[10.0, 20.0]);
        tree_pair_transform_into(
            &rule,
            operation.clone(),
            &mut dst,
            &src,
            black_box(1.0),
            black_box(0.0),
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    Timings {
        compile,
        replay,
        context_hit_execute,
        rebuild_execute,
        oneshot,
    }
}

fn bench_su2(
    build_iters: usize,
    replay_iters: usize,
    context_iters: usize,
    rebuild_execute_iters: usize,
    oneshot_iters: usize,
) -> Timings {
    let (structure, src_space, dst_space) = su2_recoupling_fixture();
    let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![0.0, 0.0],
        dst_space.clone(),
        structure.clone(),
    )
    .unwrap();

    let compile = time_loop(build_iters, || {
        let structure =
            tree_pair_transform_structure(&SU2FusionRule, operation.clone(), &dst, &src).unwrap();
        black_box(structure);
    });

    let compiled =
        tree_pair_transform_structure(&SU2FusionRule, operation.clone(), &dst, &src).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    let replay = time_loop(replay_iters, || {
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &compiled,
            &mut dst,
            &src,
            black_box(1.0),
            black_box(0.0),
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    let mut rebuild_backend = DenseTreeTransformOperations::default();
    let mut rebuild_workspace = TreeTransformWorkspace::default();
    let rebuild_execute = time_loop(rebuild_execute_iters, || {
        src.data_mut().copy_from_slice(&[10.0, 20.0]);
        tree_pair_transform_into_with(
            &mut rebuild_backend,
            &mut rebuild_workspace,
            &SU2FusionRule,
            operation.clone(),
            &mut dst,
            &src,
            black_box(1.0),
            black_box(0.0),
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    let mut context =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    tree_pair_transform_into_with_context(
        &mut context,
        &SU2FusionRule,
        operation.clone(),
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    let context_hit_execute = time_loop(context_iters, || {
        src.data_mut().copy_from_slice(&[10.0, 20.0]);
        tree_pair_transform_into_with_context(
            &mut context,
            &SU2FusionRule,
            operation.clone(),
            &mut dst,
            &src,
            black_box(1.0),
            black_box(0.0),
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    let oneshot = time_loop(oneshot_iters, || {
        src.data_mut().copy_from_slice(&[10.0, 20.0]);
        tree_pair_transform_into(
            &SU2FusionRule,
            operation.clone(),
            &mut dst,
            &src,
            black_box(1.0),
            black_box(0.0),
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    Timings {
        compile,
        replay,
        context_hit_execute,
        rebuild_execute,
        oneshot,
    }
}

fn time_loop(iterations: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..1_000 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed()
}

type FpU1Rule = tenet_core::ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
type ProductRule = tenet_core::ProductFusionRule<FpU1Rule, SU2FusionRule>;

fn product_fixture() -> (
    ProductRule,
    FusionTensorMapSpace<2, 1>,
    FusionTensorMapSpace<2, 1>,
) {
    let left_rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let rule = FermionParityFusionRule
        .product(U1FusionRule)
        .product(SU2FusionRule);
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let left_sector =
        |parity, charge| left_rule.encode_sector(parity, U1Irrep::new(charge).sector_id());
    let sector = |parity, charge, twice_spin| {
        rule.encode_sector(
            left_sector(parity, charge),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };

    let a = sector(odd, 1, 1);
    let b = sector(odd, -1, 1);
    let c0 = sector(even, 0, 0);
    let c1 = sector(even, 0, 2);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([a], false), SectorLeg::new([b], false)]),
        FusionProductSpace::new([SectorLeg::new([c0, c1], false)]),
    );
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([b], false), SectorLeg::new([a], false)]),
        FusionProductSpace::new([SectorLeg::new([c0, c1], false)]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
        src_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
        dst_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    (rule, src_space, dst_space)
}

fn su2_recoupling_fixture() -> (BlockStructure, TensorMapSpace<4, 0>, TensorMapSpace<4, 0>) {
    let src_key0 = all_codomain_fusion_tree_key([0, 1]);
    let src_key1 = all_codomain_fusion_tree_key([2, 1]);
    let structure = BlockStructure::packed_column_major_with_keys(
        4,
        [(src_key0, vec![1, 1, 1, 1]), (src_key1, vec![1, 1, 1, 1])],
    )
    .unwrap();
    (
        structure,
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
    )
}

fn all_codomain_fusion_tree_key(innerlines: [usize; 2]) -> BlockKey {
    BlockKey::from(FusionTreeBlockKey::pair(
        FusionTreeKey::from_sector_ids(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            innerlines,
            [1, 1, 1],
        ),
        FusionTreeKey::new(
            Vec::<SectorId>::new(),
            None,
            Vec::<bool>::new(),
            Vec::<SectorId>::new(),
            Vec::<SectorId>::new(),
        ),
    ))
}
