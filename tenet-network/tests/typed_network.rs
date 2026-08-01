use std::fmt::Debug;
use std::sync::Arc;

use tenet::core::{
    product_sector, CheckedFusionAlgebra, FermionParityFusionRule, FusionAlgebraError,
    MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, SectorCodec, TensorStorage, TypedSectorAdmission, U1FusionRule,
    U1Irrep, Z2Irrep,
};
use tenet::prelude::{Complex64, TensorScalar};
use tenet::typed::{GradedSpace, Runtime, TensorMap};
use tenet_network::{
    GreedyDenseOptimizer, LabelOrderDenseOptimizer, Network, NetworkExecutionWorkspace,
    TemporaryLabel, TensorId,
};

fn labels(names: &[&str]) -> Vec<TemporaryLabel> {
    names.iter().copied().map(TemporaryLabel::from).collect()
}

fn pair_network() -> Network {
    Network::new(
        vec![labels(&["a", "k"]), labels(&["k", "b"])],
        vec![false, false],
        vec![Some(1), Some(1)],
        labels(&["a", "b"]),
        Some(1),
    )
    .unwrap()
}

fn plan_accepts_storage<R, D, S>(network: &Network, tensors: &[&TensorMap<R, D, S>])
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    network.plan(tensors, &GreedyDenseOptimizer).unwrap();
}

fn pair_case<R, D>(space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar + PartialEq + Debug,
{
    let runtime = Runtime::builder().build().unwrap();
    let lhs = TensorMap::<R, D>::rand_with_seed(&runtime, [space], [space], 1).unwrap();
    let rhs = TensorMap::<R, D>::rand_with_seed(&runtime, [space], [space], 2).unwrap();
    let tensors = [&lhs, &rhs];
    let planned = pair_network()
        .plan(&tensors, &GreedyDenseOptimizer)
        .unwrap();
    let expected = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for _ in 0..2 {
        let actual = planned
            .execute_with_workspace(&tensors, &mut workspace)
            .unwrap();
        assert_eq!(actual.data(), expected.data());
        assert_eq!(actual.codomain(), expected.codomain());
        assert_eq!(actual.domain(), expected.domain());
    }
}

fn assert_same<R, D>(actual: &TensorMap<R, D>, expected: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar + PartialEq + Debug,
{
    assert_eq!(actual.data(), expected.data());
    assert_eq!(actual.codomain(), expected.codomain());
    assert_eq!(actual.domain(), expected.domain());
}

#[test]
fn provider_and_dtype_matrix_matches_direct_contract() {
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        false,
    )
    .unwrap();
    pair_case::<_, f64>(&u1);
    pair_case::<_, Complex64>(&u1);

    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
        false,
    )
    .unwrap();
    pair_case::<_, f64>(&su2);
    pair_case::<_, Complex64>(&su2);

    let fz2 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
        false,
    )
    .unwrap();
    pair_case::<_, f64>(&fz2);
    pair_case::<_, Complex64>(&fz2);

    let product_rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let product = GradedSpace::try_new(
        product_rule,
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 1),
        ],
        false,
    )
    .unwrap();
    pair_case::<_, f64>(&product);
    pair_case::<_, Complex64>(&product);
}

#[test]
fn planning_conjugation_uses_checked_effective_duals_without_reading_storage() {
    fn run<R>(
        runtime: &Runtime,
        x0: &GradedSpace<R>,
        x1: &GradedSpace<R>,
        y: &GradedSpace<R>,
        z0: &GradedSpace<R>,
        z1: &GradedSpace<R>,
        seed: u64,
    ) where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
    {
        let a = TensorMap::<R, f64>::rand_with_seed(runtime, [x0, x1], [y], seed).unwrap();
        let b = TensorMap::rand_with_seed(runtime, [x1], [z0, z1], seed + 1).unwrap();
        let adjoint = a.adjoint().unwrap();
        assert_eq!(adjoint.codomain(), vec![y.clone()]);
        assert_eq!(adjoint.domain(), vec![x0.clone(), x1.clone()]);
        let network = Network::new(
            vec![labels(&["i0", "i1", "j"]), labels(&["i1", "k0", "k1"])],
            vec![true, false],
            vec![Some(2), Some(1)],
            labels(&["j", "i0", "k0", "k1"]),
            Some(2),
        )
        .unwrap();
        let refs = [&a, &b];
        plan_accepts_storage(&network, &refs);
        let actual = network
            .plan(&refs, &GreedyDenseOptimizer)
            .unwrap()
            .execute(&refs)
            .unwrap();
        let expected = adjoint.contract(&b, &[2], &[0], &[0, 1, 2, 3]).unwrap();
        assert_eq!(actual.codomain(), vec![y.clone(), x0.try_dual().unwrap()]);
        assert_eq!(actual.domain(), vec![z0.clone(), z1.clone()]);
        assert_same(&actual, &expected);
    }

    let runtime = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let x0 = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(2), 2)], false).unwrap();
    let x1_base = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(-1), 1)], false).unwrap();
    let x1 = x1_base.try_dual().unwrap();
    let y = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(1), 3)], false).unwrap();
    let z0 = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(-2), 2)], false).unwrap();
    let z1 = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 1)], false).unwrap();
    run(&runtime, &x0, &x1, &y, &z0, &z1, 748_001);

    let product_rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let product_x0 = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1)],
        false,
    )
    .unwrap();
    let product_x1 = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 2)],
        false,
    )
    .unwrap()
    .try_dual()
    .unwrap();
    let product_y = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2)],
        false,
    )
    .unwrap();
    let product_z0 = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::EVEN, U1Irrep::new(-2)), 1)],
        false,
    )
    .unwrap();
    let product_z1 = GradedSpace::try_new(
        product_rule,
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 1)],
        false,
    )
    .unwrap();
    run(
        &runtime,
        &product_x0,
        &product_x1,
        &product_y,
        &product_z0,
        &product_z1,
        748_003,
    );
}

#[test]
fn single_scalar_split_and_heterogeneous_final_permutation() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = |degeneracy| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), degeneracy)],
            false,
        )
        .unwrap()
    };
    let v = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        false,
    )
    .unwrap();
    let tensor = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 10).unwrap();

    let single = Network::new(
        vec![labels(&["i", "j"])],
        vec![false],
        vec![Some(1)],
        labels(&["j", "i"]),
        Some(1),
    )
    .unwrap();
    let single_plan = single.plan(&[&tensor], &GreedyDenseOptimizer).unwrap();
    assert_same(
        &single_plan.execute(&[&tensor]).unwrap(),
        &tensor.permute(&[1], &[0]).unwrap(),
    );

    let scalar = Network::new(
        vec![labels(&["i", "j"]), labels(&["i", "j"])],
        vec![true, false],
        vec![Some(1), Some(1)],
        vec![],
        Some(0),
    )
    .unwrap();
    let scalar_plan = scalar
        .plan(&[&tensor, &tensor], &GreedyDenseOptimizer)
        .unwrap();
    let value = scalar_plan
        .execute(&[&tensor, &tensor])
        .unwrap()
        .scalar()
        .unwrap();
    let norm = tensor.norm().unwrap();
    assert!((value - norm * norm).abs() <= 1e-12 * (1.0 + norm * norm));
    let other_provider = Arc::new(U1FusionRule);
    let other_v = GradedSpace::try_new(
        other_provider,
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        false,
    )
    .unwrap();
    let other = TensorMap::rand_with_seed(&runtime, [&other_v], [&other_v], 13).unwrap();
    let scalar_from_lhs = scalar_plan.execute(&[&tensor, &other]).unwrap();
    assert!(std::ptr::eq(scalar_from_lhs.provider(), tensor.provider()));
    let scalar_from_other_lhs = scalar_plan.execute(&[&other, &tensor]).unwrap();
    assert!(std::ptr::eq(
        scalar_from_other_lhs.provider(),
        other.provider()
    ));

    let (a, b, bond, c, d) = (space(2), space(3), space(4), space(5), space(6));
    let lhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&a, &b], [&bond], 11).unwrap();
    let rhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&bond], [&c, &d], 12).unwrap();
    let crossed = Network::new(
        vec![labels(&["a", "b", "k"]), labels(&["k", "c", "d"])],
        vec![false, false],
        vec![Some(2), Some(1)],
        labels(&["d", "a", "b", "c"]),
        Some(2),
    )
    .unwrap();
    let refs = [&lhs, &rhs];
    let plan = crossed.plan(&refs, &GreedyDenseOptimizer).unwrap();
    let expected = lhs
        .contract(&rhs, &[2], &[0], &[0, 1, 2, 3])
        .unwrap()
        .permute(&[3, 0], &[1, 2])
        .unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    assert_same(
        &plan.execute_with_workspace(&refs, &mut workspace).unwrap(),
        &expected,
    );
    assert_same(
        &plan.execute_with_workspace(&refs, &mut workspace).unwrap(),
        &expected,
    );
}

#[test]
fn compact_and_lazy_representation_replay_stays_semantic() {
    let runtime = Runtime::builder().build().unwrap();
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)], false).unwrap();
    let dense = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 20).unwrap();
    let identity = Network::new(
        vec![labels(&["i", "j"])],
        vec![false],
        vec![Some(1)],
        labels(&["i", "j"]),
        Some(1),
    )
    .unwrap();
    let plan = identity.plan(&[&dense], &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    assert_same(
        &plan
            .execute_with_workspace(&[&dense], &mut workspace)
            .unwrap(),
        &dense,
    );
    let lazy = dense.adjoint().unwrap();
    assert_same(
        &plan
            .execute_with_workspace(&[&lazy], &mut workspace)
            .unwrap(),
        &lazy,
    );
    let conjugate = Network::new(
        vec![labels(&["i", "j"])],
        vec![true],
        vec![Some(1)],
        labels(&["j", "i"]),
        Some(1),
    )
    .unwrap();
    let lazy_conj_plan = conjugate.plan(&[&lazy], &GreedyDenseOptimizer).unwrap();
    for _ in 0..2 {
        assert_same(
            &lazy_conj_plan
                .execute_with_workspace(&[&lazy], &mut workspace)
                .unwrap(),
            &dense,
        );
    }

    let (_, compact, _) = dense.svd_compact().unwrap();
    let compact_plan = identity.plan(&[&compact], &GreedyDenseOptimizer).unwrap();
    let scaled = compact.scale(2.0);
    assert_same(
        &compact_plan
            .execute_with_workspace(&[&compact], &mut workspace)
            .unwrap(),
        &compact,
    );
    assert_same(
        &compact_plan
            .execute_with_workspace(&[&scaled], &mut workspace)
            .unwrap(),
        &scaled,
    );
    let compact_conj_plan = conjugate.plan(&[&compact], &GreedyDenseOptimizer).unwrap();
    let compact_adjoint = compact.adjoint().unwrap();
    for _ in 0..2 {
        assert_eq!(
            compact_conj_plan
                .execute_with_workspace(&[&compact], &mut workspace)
                .unwrap()
                .diagonal_spectrum()
                .unwrap(),
            compact_adjoint.diagonal_spectrum().unwrap()
        );
    }
}

fn u1_space(provider: &Arc<U1FusionRule>, degeneracy: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(Arc::clone(provider), [(U1Irrep::new(0), degeneracy)], false).unwrap()
}

#[test]
fn workspace_drift_and_provider_allocation_changes_do_not_leave_stale_results() {
    let runtime1 = Runtime::builder().build().unwrap();
    let runtime2 = Runtime::builder().build().unwrap();
    let provider1 = Arc::new(U1FusionRule);
    let provider2 = Arc::new(U1FusionRule);
    let v1 = u1_space(&provider1, 2);
    let v2 = u1_space(&provider2, 2);
    let larger = u1_space(&provider1, 3);
    let make = |runtime: &Runtime, space: &GradedSpace<U1FusionRule>, seed| {
        TensorMap::<U1FusionRule, f64>::rand_with_seed(runtime, [space], [space], seed).unwrap()
    };
    let (a1, b1) = (make(&runtime1, &v1, 30), make(&runtime1, &v1, 31));
    let (a2, b2) = (make(&runtime1, &v2, 32), make(&runtime1, &v2, 33));
    let (wide_a, wide_b) = (make(&runtime1, &larger, 34), make(&runtime1, &larger, 35));
    let (foreign_a, foreign_b) = (make(&runtime2, &v1, 36), make(&runtime2, &v1, 37));
    let network = pair_network();
    let plan = network.plan(&[&a1, &b1], &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for operands in [
        [&a1, &b1],
        [&a2, &b1],
        [&a1, &b2],
        [&wide_a, &wide_b],
        [&foreign_a, &foreign_b],
    ] {
        let actual = plan
            .execute_with_workspace(&operands, &mut workspace)
            .unwrap();
        let expected = operands[0]
            .contract(operands[1], &[1], &[0], &[0, 1])
            .unwrap();
        assert_same(&actual, &expected);
    }

    assert!(plan
        .execute_with_workspace(&[&a1, &foreign_b], &mut workspace)
        .is_err());
    assert_same(
        &plan
            .execute_with_workspace(&[&a1, &b1], &mut workspace)
            .unwrap(),
        &a1.contract(&b1, &[1], &[0], &[0, 1]).unwrap(),
    );

    let incompatible = u1_space(&provider1, 4);
    let bad = make(&runtime1, &incompatible, 38);
    assert!(plan
        .execute_with_workspace(&[&a1, &bad], &mut workspace)
        .is_err());
    assert!(plan.execute_with_workspace(&[&a1], &mut workspace).is_err());

    let second_plan = network.plan(&[&a2, &b2], &GreedyDenseOptimizer).unwrap();
    assert_same(
        &second_plan
            .execute_with_workspace(&[&a2, &b2], &mut workspace)
            .unwrap(),
        &a2.contract(&b2, &[1], &[0], &[0, 1]).unwrap(),
    );
}

#[test]
fn one_plan_replays_concurrently_with_distinct_workspaces() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let v = u1_space(&provider, 3);
    let lhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 40).unwrap();
    let rhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 41).unwrap();
    let plan = pair_network()
        .plan(&[&lhs, &rhs], &GreedyDenseOptimizer)
        .unwrap();
    let expected = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    std::thread::scope(|scope| {
        let handles = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let mut workspace = NetworkExecutionWorkspace::default();
                    plan.execute_with_workspace(&[&lhs, &rhs], &mut workspace)
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            assert_same(&handle.join().unwrap(), &expected);
        }
    });
}

#[test]
fn greedy_order_and_four_site_ring_match_manual_typed_oracles() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = |degeneracy| u1_space(&provider, degeneracy);
    let (va, vb, vc, vd, ve) = (space(4), space(8), space(4), space(2), space(2));
    let chain = [
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&va], [&vb], 51).unwrap(),
        TensorMap::rand_with_seed(&runtime, [&vb], [&vc], 52).unwrap(),
        TensorMap::rand_with_seed(&runtime, [&vc], [&vd], 53).unwrap(),
        TensorMap::rand_with_seed(&runtime, [&vd], [&ve], 54).unwrap(),
    ];
    let chain_refs = [&chain[0], &chain[1], &chain[2], &chain[3]];
    let chain_network = Network::new(
        vec![
            labels(&["a", "b"]),
            labels(&["b", "c"]),
            labels(&["c", "d"]),
            labels(&["d", "e"]),
        ],
        vec![false; 4],
        vec![Some(1); 4],
        labels(&["a", "e"]),
        Some(1),
    )
    .unwrap();
    let greedy = chain_network
        .plan(&chain_refs, &GreedyDenseOptimizer)
        .unwrap();
    let naive = chain_network
        .plan(
            &chain_refs,
            &LabelOrderDenseOptimizer::new(labels(&["b", "c", "d"])),
        )
        .unwrap();
    assert_eq!(
        (
            greedy.plan().steps()[0].lhs(),
            greedy.plan().steps()[0].rhs()
        ),
        (TensorId::new(2), TensorId::new(3))
    );
    assert!(greedy.plan().total_cost() < naive.plan().total_cost());
    let greedy_result = greedy.execute(&chain_refs).unwrap();
    let naive_result = naive.execute(&chain_refs).unwrap();
    assert_eq!(greedy_result.codomain(), naive_result.codomain());
    assert_eq!(greedy_result.domain(), naive_result.domain());
    assert!(greedy_result
        .data()
        .iter()
        .zip(naive_result.data())
        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));

    let v = space(3);
    let ring = std::array::from_fn::<_, 4, _>(|index| {
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 60 + index as u64)
            .unwrap()
    });
    let ring_refs = [&ring[0], &ring[1], &ring[2], &ring[3]];
    let ring_network = Network::new(
        vec![
            labels(&["a", "b"]),
            labels(&["b", "c"]),
            labels(&["c", "d"]),
            labels(&["d", "a"]),
        ],
        vec![false; 4],
        vec![Some(1); 4],
        vec![],
        Some(0),
    )
    .unwrap();
    let planned = ring_network
        .plan(&ring_refs, &GreedyDenseOptimizer)
        .unwrap();
    let manual = ring[0]
        .contract(&ring[1], &[1], &[0], &[0, 1])
        .unwrap()
        .contract(&ring[2], &[1], &[0], &[0, 1])
        .unwrap()
        .contract(&ring[3], &[1, 0], &[0, 1], &[])
        .unwrap();
    let actual = planned.execute(&ring_refs).unwrap();
    assert_eq!(actual.codomain(), manual.codomain());
    assert_eq!(actual.domain(), manual.domain());
    assert!(actual
        .data()
        .iter()
        .zip(manual.data())
        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
}
