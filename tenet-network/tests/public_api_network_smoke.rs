use tenet::prelude::{GradedSpace, Runtime, TensorMap, U1FusionRule, U1Irrep};
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
