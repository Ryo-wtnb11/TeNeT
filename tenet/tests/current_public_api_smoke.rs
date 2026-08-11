//! Independent Phase A smoke program for the canonical typed public API.

use tenet::core::TypedSectorAdmission;
use tenet::prelude::{
    product_sector, FermionParityFusionRule, GradedSpace, ProductFusionRuleExt, Runtime,
    SU2FusionRule, SU2Irrep, TensorMap, Truncation, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::typed::{TensorScalar, TypedTensorRootDispatch};
use tenet_network::{GreedyDenseOptimizer, Network, NetworkExecutionWorkspace, TemporaryLabel};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-11 * (1.0 + expected.abs()));
    }
}

fn labels(names: &[&str]) -> Vec<TemporaryLabel> {
    names.iter().copied().map(TemporaryLabel::from).collect()
}

fn inspect_with_the_existing_root_bound<R, D>(tensor: &TensorMap<R, D>)
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: TensorScalar,
{
    assert_eq!(tensor.blocks().unwrap().count(), tensor.block_count());
    if tensor.block_count() != 0 {
        let _ = tensor.block_fusion_trees(0).unwrap();
    }
}

#[test]
fn constructs_u1_su2_and_product_tensors_from_provider_labels() {
    let runtime = Runtime::builder().build().unwrap();

    let u1 =
        GradedSpace::try_new(U1FusionRule, [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2)]).unwrap();
    let u1_tensor = TensorMap::<U1FusionRule, f64>::zeros(&runtime, [&u1], [&u1]).unwrap();
    inspect_with_the_existing_root_bound(&u1_tensor);
    let coupled: Vec<_> = (0..u1_tensor.block_count())
        .map(|block| *u1_tensor.block_fusion_trees(block).unwrap().coupled())
        .collect();
    assert!(coupled.contains(&U1Irrep::new(-1)));
    assert!(coupled.contains(&U1Irrep::new(0)));

    let su2 = GradedSpace::try_new(
        SU2FusionRule,
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_tensor = TensorMap::<SU2FusionRule, f64>::zeros(&runtime, [&su2], [&su2]).unwrap();
    assert!(su2_tensor.block_count() >= 2);

    let product = GradedSpace::try_new(
        FermionParityFusionRule.product(U1FusionRule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 1),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
        ],
    )
    .unwrap();
    let product_tensor = TensorMap::<_, f64>::zeros(&runtime, [&product], [&product]).unwrap();
    assert_eq!(product_tensor.block_count(), 2);
}

#[test]
fn u1_index_contraction_trace_and_decomposition_paths_are_executable() {
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new(U1FusionRule, [(U1Irrep::new(0), 2)]).unwrap();
    let tensor =
        TensorMap::<U1FusionRule, f64>::from_block_fn(&runtime, [&space], [&space], |_, index| {
            match index {
                [0, 0] => 3.0,
                [1, 1] => 2.0,
                _ => 1.0,
            }
        })
        .unwrap();

    let identity = TensorMap::<U1FusionRule, f64>::id(&runtime, [&space]).unwrap();
    assert_close(
        identity
            .contract(&tensor, &[1], &[0], &[0, 1])
            .unwrap()
            .data(),
        tensor.data(),
    );
    assert_close(
        tensor.adjoint().unwrap().adjoint().unwrap().data(),
        tensor.data(),
    );
    assert_eq!(identity.tr().unwrap(), 2.0);

    let rank_three = TensorMap::<U1FusionRule, f64>::from_block_fn(
        &runtime,
        [&space, &space],
        [&space],
        |_, i| (1 + i[0] + 2 * i[1] + 4 * i[2]) as f64,
    )
    .unwrap();
    let roundtrip = rank_three
        .permute(&[1, 0], &[2])
        .unwrap()
        .permute(&[1, 0], &[2])
        .unwrap();
    assert_eq!(roundtrip.data(), rank_three.data());

    let svd = tensor.svd_trunc(&Truncation::rank(2)).unwrap();
    let reconstructed = svd.u.compose(&svd.s).unwrap().compose(&svd.vh).unwrap();
    assert_close(reconstructed.data(), tensor.data());
}

#[test]
fn planned_network_replays_through_the_current_public_api() {
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new(U1FusionRule, [(U1Irrep::new(0), 2)]).unwrap();
    let lhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&space], [&space], 9_001)
        .unwrap();
    let rhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&space], [&space], 9_002)
        .unwrap();
    let network = Network::new(
        vec![labels(&["a", "k"]), labels(&["k", "b"])],
        vec![false, false],
        vec![Some(1), Some(1)],
        labels(&["a", "b"]),
        Some(1),
    )
    .unwrap();
    let inputs = [&lhs, &rhs];
    let plan = network.plan(&inputs, &GreedyDenseOptimizer).unwrap();
    let expected = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();

    for _ in 0..2 {
        let actual = plan
            .execute_with_workspace(&inputs, &mut workspace)
            .unwrap();
        assert_close(actual.data(), expected.data());
    }
}

// No "trivial symmetry" smoke is fabricated here: current main exposes no
// canonical no-symmetry provider. The operation matrix classifies that public
// path as unsupported rather than treating a vacuum-only U(1) fixture as one.
