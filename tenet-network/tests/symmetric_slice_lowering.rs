use std::sync::Arc;

use tenet::core::{SectorId, U1FusionRule, U1Irrep};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, TensorMap};
use tenet_network::{
    slice_plan_for, ContractionPlan, DenseCostModel, DenseTensorInfo, Network, NetworkIR,
    SlicedPlan, TemporaryLabel,
};

fn label(name: &str) -> TemporaryLabel {
    TemporaryLabel::from(name)
}

#[test]
fn common_lowering_is_atomic_and_reconstructs_after_dense_roundtrip() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let two = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let vacuum = GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(0), 1)]).unwrap();
    let tensor = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        &runtime,
        [&two, &two, &vacuum],
        std::iter::empty::<&GradedSpace<U1FusionRule>>(),
        1028,
    )
    .unwrap();
    let labels = vec![label("a"), label("b"), label("c")];
    let ir = NetworkIR::from_labels(vec![labels.clone()], labels.clone()).unwrap();
    let order = ContractionPlan::new(1, labels.clone(), Vec::new()).unwrap();
    let cost = DenseCostModel::from_network(&ir, &[DenseTensorInfo::new(vec![2, 2, 1])]).unwrap();
    let dense = SlicedPlan::new(
        order,
        slice_plan_for(
            &ir,
            &ContractionPlan::new(1, labels.clone(), Vec::new()).unwrap(),
            &cost,
            &labels,
        ),
    );
    let network = Network::new(
        vec![labels.clone()],
        vec![false],
        vec![Some(3)],
        labels,
        Some(3),
    )
    .unwrap();

    let lowered = network
        .lower_symmetric_sliced_plan(&[&tensor], dense.clone())
        .unwrap();
    let restored = SlicedPlan::from_text(&dense.to_text()).unwrap();
    let reconstructed = network
        .lower_symmetric_sliced_plan(&[&tensor], restored)
        .unwrap();

    assert_eq!(lowered, reconstructed);
    assert_eq!(lowered.slices().nslices(), 4);
    assert!(lowered.slices().indices().iter().all(|index| {
        index
            .pieces()
            .iter()
            .all(|piece| piece.range().end() - piece.range().start() == 1)
    }));
    let q1 = SectorId::from(U1Irrep::new(1));
    let q0 = SectorId::from(U1Irrep::new(0));
    assert!(lowered.slices().combinations().any(|combination| {
        combination
            .iter()
            .map(|piece| piece.sector())
            .eq([q1, q1, q0])
    }));
}

#[test]
fn adjoint_lowering_uses_the_actual_non_self_dual_effective_axis() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let codomain =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(2), 1)]).unwrap();
    let domain = GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(-1), 2)]).unwrap();
    let tensor =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&codomain], [&domain], 1029)
            .unwrap();
    let written = vec![label("p"), label("x")];
    let effective = vec![label("x"), label("p")];
    let ir = NetworkIR::from_labels(vec![effective.clone()], effective.clone()).unwrap();
    let order = ContractionPlan::new(1, effective.clone(), Vec::new()).unwrap();
    let cost = DenseCostModel::from_network(&ir, &[DenseTensorInfo::new(vec![2, 1])]).unwrap();
    let dense = SlicedPlan::new(
        order,
        slice_plan_for(
            &ir,
            &ContractionPlan::new(1, effective.clone(), Vec::new()).unwrap(),
            &cost,
            &[label("x")],
        ),
    );
    let network =
        Network::new(vec![written], vec![true], vec![Some(1)], effective, Some(1)).unwrap();

    let lowered = network
        .lower_symmetric_sliced_plan(&[&tensor], dense)
        .unwrap();
    let authority = lowered.slices().indices()[0].authority_leg();
    assert_eq!(authority.sectors(), &[SectorId::from(U1Irrep::new(-1))]);
    assert!(!authority.is_dual());
}
