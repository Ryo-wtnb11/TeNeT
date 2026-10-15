use std::sync::Arc;

use tenet_core::{
    FermionParityFusionRule, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace,
    MultiplicityFreeFusionRule, MultiplicityFreeRigidSymbols, ProductFusionRule,
    ProductSectorCodec, SU2Irrep, SectorId, SectorLeg, TensorKitProductCodec, TensorMap,
    TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet_operations::ContractDestinationInit;
use tenet_tensors::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand, HostEagerExecutor,
    OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
    TreeTransformRuleCacheKey,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

const WORKLOADS: [([usize; 2], [usize; 2], [usize; 4]); 3] = [
    ([2, 3], [0, 1], [0, 1, 2, 3]),
    ([3, 2], [0, 1], [0, 1, 2, 3]),
    ([3, 2], [0, 1], [1, 0, 2, 3]),
];

fn u1_sectors() -> Vec<SectorId> {
    [-1, 0, 1]
        .into_iter()
        .map(|charge| U1Irrep::new(charge).sector_id())
        .collect()
}

fn product_sectors() -> Vec<SectorId> {
    [(-1, 1), (0, 0), (1, 1)]
        .into_iter()
        .map(|(charge, parity)| {
            TensorKitProductCodec::try_encode(
                U1Irrep::new(charge).sector_id(),
                SectorId::new(parity),
            )
            .unwrap()
        })
        .collect()
}

fn assert_replay_allocates_nothing<R>(rule: &R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + MultiplicityFreeFusionRule
        + TreeTransformRuleCacheKey
        + Clone,
    R::Key: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
{
    let provider = Arc::new(rule.clone());
    let bind = |tensor: &TensorMap<f64, 2, 2>| {
        BoundDynamicFusionMapSpace::bind_multiplicity_free(
            DynamicFusionMapSpace::from_typed(tensor.fusion_space().unwrap()),
            Arc::clone(&provider),
        )
        .unwrap()
    };
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, 1)), false);
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space = |hom: FusionTreeHomSpace| {
        let key_count = hom.fusion_tree_keys(rule).len();
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims(
                [sectors.len(), sectors.len()],
                [sectors.len(), sectors.len()],
            )
            .unwrap(),
            hom,
            rule,
            vec![vec![1; 4]; key_count],
        )
        .unwrap()
    };
    let lhs_space = space(homspace());
    let rhs_space = space(homspace());
    let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_space.required_len().unwrap())
            .map(|index| index as f64 * 0.25 - 2.0)
            .collect(),
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..rhs_space.required_len().unwrap())
            .map(|index| index as f64 * 0.5 - 3.0)
            .collect(),
        rhs_space,
    )
    .unwrap();

    for (lhs_axes, rhs_axes, output_axes) in WORKLOADS {
        let axes = || {
            TensorContractSpec::new(
                &lhs_axes,
                &rhs_axes,
                OutputAxisOrder::from_axes(&output_axes),
            )
        };
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs.fusion_space().unwrap().homspace(),
            rhs.fusion_space().unwrap().homspace(),
            &lhs_axes,
            &rhs_axes,
            &output_axes,
            2,
        )
        .unwrap();
        let dst_space = space(dst_hom);
        let mut expected = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            vec![0.0; dst_space.required_len().unwrap()],
            dst_space,
        )
        .unwrap();
        let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
        let (dst_bound, lhs_bound, rhs_bound) = (bind(&expected), bind(&lhs), bind(&rhs));
        let dst_structure = Arc::clone(expected.structure());
        context
            .tensorcontract_fusion_dyn_into(
                &dst_bound,
                expected.data_mut(),
                &lhs_bound,
                lhs.data(),
                &rhs_bound,
                rhs.data(),
                axes(),
                1.0,
                0.0,
            )
            .unwrap();
        let mut actual = expected.clone();
        let prepared = context
            .plan_contract::<HostEagerExecutor, R>(
                &dst_bound,
                FusionOperand::direct(lhs_bound.space()),
                FusionOperand::direct(rhs_bound.space()),
                &lhs_axes,
                &rhs_axes,
                &output_axes,
            )
            .unwrap();
        let replay = |context: &mut TensorContractFusionExecutionContext<f64, R::Key>,
                      dst: &mut [f64]| {
            context.execute_contract_resolution_host(
                &prepared,
                (&dst_structure, dst),
                (lhs.structure(), lhs.data()),
                (rhs.structure(), rhs.data()),
                1.0,
                ContractDestinationInit::Axpby(0.0),
            )
        };
        for _ in 0..2 {
            replay(&mut context, actual.data_mut()).unwrap();
        }
        counting_alloc::start();
        let result = replay(&mut context, actual.data_mut());
        let allocs = counting_alloc::stop();
        result.unwrap();

        assert_eq!(actual.data(), expected.data());
        assert_eq!(allocs.calls, 0, "axes={lhs_axes:?}/{output_axes:?}");
    }
}

#[test]
fn warmed_prepared_rank4_fusion_replay_allocates_nothing() {
    assert_replay_allocates_nothing(&U1FusionRule, &u1_sectors());
    assert_replay_allocates_nothing(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
    );
    assert_replay_allocates_nothing(
        &tenet_core::SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
            SU2Irrep::from_twice_spin(2).sector_id(),
        ],
    );
    assert_replay_allocates_nothing(
        &ProductFusionRule::<U1FusionRule, FermionParityFusionRule>::new(
            U1FusionRule,
            FermionParityFusionRule,
        ),
        &product_sectors(),
    );
}
