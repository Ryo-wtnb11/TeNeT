#![cfg(feature = "cuda")]

use std::sync::Arc;

use tenet::core::{
    product_sector, CheckedFusionAlgebra, FermionParityFusionRule, FusionAlgebraError,
    MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, SectorCodec, TypedSectorAdmission, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::typed::{CudaStorage, GradedSpace, Runtime, TensorMap};
use tenet_network::{
    ContractionPlan, ContractionStep, GreedyDenseOptimizer, Network, TemporaryLabel, TensorId,
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

fn cuda_pair<R>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let lhs = TensorMap::<R, f64>::rand_with_seed(runtime, [space], [space], seed).unwrap();
    let rhs = TensorMap::<R, f64>::rand_with_seed(runtime, [space], [space], seed + 1).unwrap();
    let host_plan = pair_network()
        .plan(&[&lhs, &rhs], &GreedyDenseOptimizer)
        .unwrap();
    let lhs_cuda = lhs.to_cuda().unwrap();
    let rhs_cuda = rhs.to_cuda().unwrap();
    let cuda_refs: [&TensorMap<R, f64, CudaStorage>; 2] = [&lhs_cuda, &rhs_cuda];
    let cuda_plan = pair_network()
        .plan(&cuda_refs, &GreedyDenseOptimizer)
        .unwrap();
    assert_eq!(host_plan.plan().steps(), cuda_plan.plan().steps());
    let actual = cuda_plan.execute_cuda(&cuda_refs).unwrap();
    let manual = lhs_cuda.contract(&rhs_cuda, &[1], &[0], &[0, 1]).unwrap();
    let host_oracle = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    assert_eq!(actual.placement(), lhs_cuda.placement());
    assert!(std::ptr::eq(actual.provider(), lhs_cuda.provider()));
    assert_eq!(actual.codomain(), host_oracle.codomain());
    assert_eq!(actual.domain(), host_oracle.domain());
    let actual_host = actual.to_host().unwrap();
    assert_eq!(actual_host.data(), manual.to_host().unwrap().data());
    assert_eq!(actual_host.data(), host_oracle.data());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn canonical_cuda_network_provider_matrix_chain_and_lazy_conj() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)], false).unwrap();
    cuda_pair(&runtime, &u1, 748_100);
    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [(SU2Irrep::from_twice_spin(0), 2)],
        false,
    )
    .unwrap();
    cuda_pair(&runtime, &su2, 748_110);
    let fz2 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::ODD, 2)],
        false,
    )
    .unwrap();
    cuda_pair(&runtime, &fz2, 748_120);
    let product = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
        false,
    )
    .unwrap();
    cuda_pair(&runtime, &product, 748_130);

    let tensors = (0..3)
        .map(|index| {
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&u1], [&u1], 748_140 + index)
                .unwrap()
                .to_cuda()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let chain = Network::new(
        vec![
            labels(&["a", "b"]),
            labels(&["b", "c"]),
            labels(&["c", "d"]),
        ],
        vec![false; 3],
        vec![Some(1); 3],
        labels(&["a", "d"]),
        Some(1),
    )
    .unwrap();
    let refs = [&tensors[0], &tensors[1], &tensors[2]];
    let chain_order = ContractionPlan::new(
        3,
        labels(&["a", "d"]),
        vec![
            ContractionStep::new(
                TensorId::new(0),
                TensorId::new(1),
                TensorId::new(3),
                0,
                labels(&["a", "c"]),
            ),
            ContractionStep::new(
                TensorId::new(3),
                TensorId::new(2),
                TensorId::new(4),
                0,
                labels(&["a", "d"]),
            ),
        ],
    )
    .unwrap();
    let planned = chain.plan_with(&refs, chain_order).unwrap();
    let actual = planned.execute_cuda(&refs).unwrap();
    let manual = tensors[0]
        .contract(&tensors[1], &[1], &[0], &[0, 1])
        .unwrap()
        .contract(&tensors[2], &[1], &[0], &[0, 1])
        .unwrap();
    assert_eq!(actual.codomain(), manual.codomain());
    assert_eq!(actual.domain(), manual.domain());
    assert_eq!(
        actual.to_host().unwrap().data(),
        manual.to_host().unwrap().data()
    );

    let conj_network = Network::new(
        vec![labels(&["k", "i"]), labels(&["k", "j"])],
        vec![true, false],
        vec![Some(1), Some(1)],
        labels(&["i", "j"]),
        Some(1),
    )
    .unwrap();
    let conj_refs = [&tensors[0], &tensors[1]];
    let conj_actual = conj_network
        .plan(&conj_refs, &GreedyDenseOptimizer)
        .unwrap()
        .execute_cuda(&conj_refs)
        .unwrap();
    let conj_manual = tensors[0]
        .adjoint()
        .unwrap()
        .contract(&tensors[1], &[1], &[0], &[0, 1])
        .unwrap();
    assert_eq!(conj_actual.codomain(), conj_manual.codomain());
    assert_eq!(conj_actual.domain(), conj_manual.domain());
    assert_eq!(
        conj_actual.to_host().unwrap().data(),
        conj_manual.to_host().unwrap().data()
    );

    let single = Network::new(
        vec![labels(&["k", "i"])],
        vec![true],
        vec![Some(1)],
        labels(&["i", "k"]),
        Some(1),
    )
    .unwrap();
    let single_actual = single
        .plan(&[&tensors[0]], &GreedyDenseOptimizer)
        .unwrap()
        .execute_cuda(&[&tensors[0]])
        .unwrap();
    let single_expected = tensors[0].adjoint().unwrap();
    assert!(std::ptr::eq(
        single_actual.provider(),
        tensors[0].provider()
    ));
    assert_eq!(single_actual.codomain(), single_expected.codomain());
    assert_eq!(single_actual.domain(), single_expected.domain());
    assert_eq!(
        single_actual.to_host().unwrap().data(),
        single_expected.to_host().unwrap().data()
    );

    let ket = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        &runtime,
        std::iter::empty::<&GradedSpace<U1FusionRule>>(),
        [&u1],
        748_150,
    )
    .unwrap()
    .to_cuda()
    .unwrap();
    let bra = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        &runtime,
        [&u1],
        std::iter::empty::<&GradedSpace<U1FusionRule>>(),
        748_151,
    )
    .unwrap()
    .to_cuda()
    .unwrap();
    let scalar_network = Network::new(
        vec![labels(&["k"]), labels(&["k"])],
        vec![false; 2],
        vec![Some(0), Some(1)],
        vec![],
        Some(0),
    )
    .unwrap();
    let scalar = scalar_network
        .plan(&[&ket, &bra], &GreedyDenseOptimizer)
        .unwrap()
        .execute_cuda(&[&ket, &bra])
        .unwrap();
    assert_eq!(scalar.rank(), 0);
    assert!(scalar.codomain().is_empty());
    assert!(scalar.domain().is_empty());
    assert!(std::ptr::eq(scalar.provider(), ket.provider()));
}
