use tenet::sector::{U1FusionRule, U1Irrep};
use tenet::typed::ContractSpec;
use tenet::typed::{GradedSpace, Runtime, TensorMap};
use tenet_network::{GreedyDenseOptimizer, Network, NetworkExecutionWorkspace};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-11 * (1.0 + expected.abs()));
    }
}

#[test]
fn planned_network_replays_through_the_current_public_api() {
    let runtime = Runtime::builder().build().unwrap();
    let space =
        GradedSpace::try_new(std::sync::Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let lhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&space], [&space], 9_001)
        .unwrap();
    let rhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&space], [&space], 9_002)
        .unwrap();
    let network = Network::new(
        vec![vec!["a", "k"], vec!["k", "b"]],
        vec![false, false],
        vec![1, 1],
        vec!["a", "b"],
        Some(1),
    )
    .unwrap();
    let inputs = [&lhs, &rhs];
    let plan = network.plan(&inputs, &GreedyDenseOptimizer).unwrap();
    let expected = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();

    for _ in 0..2 {
        let actual = plan.execute(&inputs, &mut workspace).unwrap();
        assert_close(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    }
}
