//! Real-device gates for provider-neutral typed ownership transfer.
//!
//! Run with `cargo test -p tenet --features cuda,cpu-faer --test \
//! typed_cuda_transfer -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use tenet::core::{
    product_sector, CheckedFusionAlgebra, FermionParityFusionRule, MultiplicityFreeRigidSymbols,
    ProductFusionRuleExt, SU2FusionRule, SU2Irrep, SectorCodec, U1FusionRule, U1Irrep, Z2Irrep,
    ZNFusionRule,
};
use tenet::typed::{BlockFusionTrees, GradedSpace, Runtime, TensorMap};

#[derive(Debug, Eq, PartialEq)]
struct LegSnapshot<S> {
    sectors: Vec<S>,
    degeneracies: Vec<usize>,
    is_dual: bool,
}

#[derive(Debug, Eq, PartialEq)]
struct BlockSnapshot<S> {
    key: tenet::core::BlockKey,
    fusion_trees: BlockFusionTrees<S>,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

#[derive(Debug, Eq, PartialEq)]
struct StructuralSnapshot<S> {
    codomain: Vec<LegSnapshot<S>>,
    domain: Vec<LegSnapshot<S>>,
    blocks: Vec<BlockSnapshot<S>>,
}

fn structural_snapshot<R>(tensor: &TensorMap<R, f64>) -> StructuralSnapshot<R::Sector>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let leg_snapshot = |leg: GradedSpace<R>| LegSnapshot {
        sectors: leg.sectors().unwrap(),
        degeneracies: leg.degeneracies().to_vec(),
        is_dual: leg.is_dual(),
    };
    StructuralSnapshot {
        codomain: tensor
            .codomain_spaces()
            .into_iter()
            .map(&leg_snapshot)
            .collect(),
        domain: tensor
            .domain_spaces()
            .into_iter()
            .map(leg_snapshot)
            .collect(),
        blocks: (0..tensor.block_count())
            .map(|index| {
                let block = tensor.block(index).unwrap();
                BlockSnapshot {
                    key: block.key().clone(),
                    fusion_trees: tensor.block_fusion_trees(index).unwrap(),
                    offset: block.offset(),
                    shape: block.shape().to_vec(),
                    strides: block.strides().to_vec(),
                }
            })
            .collect(),
    }
}

fn assert_roundtrip<R>(source: TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = source.runtime().identity();
    let structure = structural_snapshot(&source);
    let expected = source.data().to_vec();

    let device = source.to_cuda().unwrap();
    assert_eq!(device.placement(), tenet::core::Placement::Cuda(0));
    let device_clone = device.clone();
    let restored = device_clone.to_host().unwrap();

    assert!(std::ptr::eq(restored.provider(), provider));
    assert!(runtime.matches(restored.runtime()));
    assert_eq!(restored.data(), expected);
    assert_eq!(structural_snapshot(&restored), structure);
}

fn assert_direct_contract_and_compose<R>(lhs: &TensorMap<R, f64>, rhs: &TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let lhs_axes: Vec<_> = (lhs.codomain_rank()..lhs.rank()).collect();
    let rhs_axes: Vec<_> = (0..rhs.codomain_rank()).collect();
    let output_axes: Vec<_> = (0..lhs.codomain_rank() + rhs.domain_rank()).collect();
    let expected_contract = lhs
        .contract(rhs, &lhs_axes, &rhs_axes, &output_axes)
        .unwrap();
    let expected_compose = lhs.compose(rhs).unwrap();
    let provider = lhs.provider() as *const R;
    let runtime = lhs.runtime().identity();
    let lhs_device = lhs.to_cuda().unwrap();
    let rhs_device = rhs.to_cuda().unwrap();

    let contract = lhs_device
        .contract(&rhs_device, &lhs_axes, &rhs_axes, &output_axes)
        .unwrap()
        .to_host()
        .unwrap();
    let ordered = lhs_device
        .contract_ordered(&rhs_device, &lhs_axes, &rhs_axes, &output_axes)
        .unwrap()
        .to_host()
        .unwrap();
    let compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();

    for (actual, expected) in [
        (&contract, &expected_contract),
        (&ordered, &expected_contract),
        (&compose, &expected_compose),
    ] {
        assert!(std::ptr::eq(actual.provider(), provider));
        assert!(runtime.matches(actual.runtime()));
        assert_eq!(actual.data(), expected.data());
        assert_eq!(structural_snapshot(actual), structural_snapshot(expected));
    }
}

#[test]
#[ignore]
fn builtin_and_simple_product_providers_share_one_transfer_path() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();

    let u1_rule = Arc::new(U1FusionRule);
    let u1 = GradedSpace::try_new(
        Arc::clone(&u1_rule),
        [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2)],
        false,
    )
    .unwrap();
    assert_roundtrip(
        TensorMap::from_block_fn(&runtime, [&u1], [&u1], |_, indices| indices[0] as f64 + 1.0)
            .unwrap(),
    );

    let fz2_rule = Arc::new(FermionParityFusionRule);
    let fz2 = GradedSpace::try_new(
        Arc::clone(&fz2_rule),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)],
        false,
    )
    .unwrap();
    assert_roundtrip(
        TensorMap::from_block_fn(&runtime, [&fz2], [&fz2], |_, indices| {
            indices[0] as f64 + 2.0
        })
        .unwrap(),
    );

    let su2_rule = Arc::new(SU2FusionRule);
    let su2 = GradedSpace::try_new(
        Arc::clone(&su2_rule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
        false,
    )
    .unwrap();
    assert_roundtrip(
        TensorMap::from_block_fn(&runtime, [&su2], [&su2], |_, indices| {
            indices[0] as f64 + 3.0
        })
        .unwrap(),
    );

    let product_rule = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 1),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 2),
        ],
        false,
    )
    .unwrap();
    assert_roundtrip(
        TensorMap::from_block_fn(&runtime, [&product], [&product], |_, indices| {
            indices[0] as f64 + 4.0
        })
        .unwrap(),
    );
}

#[test]
#[ignore]
fn typed_cuda_direct_execution_matches_host_providers_and_structure() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();

    let u1_rule = Arc::new(U1FusionRule);
    let u1 = GradedSpace::try_new(
        Arc::clone(&u1_rule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        false,
    )
    .unwrap();
    let u1_lhs = TensorMap::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let u1_rhs = TensorMap::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 2.0
    })
    .unwrap();
    assert_direct_contract_and_compose(&u1_lhs, &u1_rhs);

    let su2_rule = Arc::new(SU2FusionRule);
    let su2 = GradedSpace::try_new(
        Arc::clone(&su2_rule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
        false,
    )
    .unwrap();
    let su2_lhs =
        TensorMap::from_block_fn(&runtime, [&su2, &su2, &su2], [&su2, &su2], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let su2_rhs =
        TensorMap::from_block_fn(&runtime, [&su2, &su2], [&su2, &su2, &su2], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 3.0
        })
        .unwrap();
    assert_direct_contract_and_compose(&su2_lhs, &su2_rhs);

    let product_rule = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
        false,
    )
    .unwrap();
    let product_lhs = TensorMap::from_block_fn(&runtime, [&product], [&product], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let product_rhs = TensorMap::from_block_fn(&runtime, [&product], [&product], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 4.0
    })
    .unwrap();
    assert_direct_contract_and_compose(&product_lhs, &product_rhs);

    let product_dual = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
        true,
    )
    .unwrap();
    let product_multileg_lhs = TensorMap::from_block_fn(
        &runtime,
        [&product],
        [&product_dual, &product_dual],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    let product_multileg_rhs = TensorMap::from_block_fn(
        &runtime,
        [&product_dual, &product_dual],
        [&product],
        |_, indices| indices.iter().sum::<usize>() as f64 + 2.0,
    )
    .unwrap();
    assert_direct_contract_and_compose(&product_multileg_lhs, &product_multileg_rhs);
}

#[test]
#[ignore]
fn typed_cuda_fermionic_contract_is_minus_six_and_compose_stays_plus_six() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let lhs_codomain =
        GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)], false).unwrap();
    let lhs_domain =
        GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)], true).unwrap();
    let rhs_codomain =
        GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)], true).unwrap();
    let rhs_domain =
        GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)], false).unwrap();
    let lhs =
        TensorMap::from_block_fn(&runtime, [&lhs_codomain], [&lhs_domain], |_, _| 2.0).unwrap();
    let rhs =
        TensorMap::from_block_fn(&runtime, [&rhs_codomain], [&rhs_domain], |_, _| 3.0).unwrap();
    assert_eq!(
        lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap().data(),
        [-6.0]
    );
    assert_eq!(lhs.compose(&rhs).unwrap().data(), [6.0]);

    let lhs_device = lhs.to_cuda().unwrap();
    let rhs_device = rhs.to_cuda().unwrap();
    assert_eq!(
        lhs_device
            .compose(&rhs_device)
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        [6.0]
    );
    assert_eq!(
        lhs_device
            .contract(&rhs_device, &[1], &[0], &[0, 1])
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        [-6.0]
    );
}

#[test]
#[ignore]
fn typed_cuda_direct_supports_canonical_lazy_and_rejects_other_scopes_before_mutation() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let other_runtime = Runtime::builder().cuda(0).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)], false).unwrap();
    let host = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let other_host = TensorMap::from_block_fn(&other_runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
    let expected = host.data().to_vec();
    let device = host.to_cuda().unwrap();
    let other_device = other_host.to_cuda().unwrap();

    assert_eq!(
        device.compose(&other_device).unwrap_err(),
        tenet::typed::Error::RuntimeMismatch
    );
    assert!(matches!(
        device.contract(&device, &[0], &[0], &[0, 1]),
        Err(tenet::typed::Error::Operation(error))
            if matches!(*error, tenet::operations::OperationError::UnsupportedTensorContractScope { .. })
    ));
    assert!(matches!(
        device.contract(&device, &[1], &[0], &[1, 0]),
        Err(tenet::typed::Error::Operation(error))
            if matches!(*error, tenet::operations::OperationError::UnsupportedTensorContractScope { .. })
    ));
    let lazy_host = host.adjoint().unwrap();
    let expected_lazy_compose = lazy_host.compose(&host).unwrap();
    let lazy = lazy_host.to_cuda().unwrap();
    let lazy_compose = lazy.compose(&device).unwrap().to_host().unwrap();
    assert!(std::ptr::eq(lazy_compose.provider(), host.provider()));
    assert!(runtime.identity().matches(lazy_compose.runtime()));
    assert_eq!(lazy_compose.data(), expected_lazy_compose.data());
    assert_eq!(
        structural_snapshot(&lazy_compose),
        structural_snapshot(&expected_lazy_compose)
    );
    assert!(matches!(
        lazy.contract(&device, &[1], &[0], &[1, 0]),
        Err(tenet::typed::Error::Operation(error))
            if matches!(*error, tenet::operations::OperationError::UnsupportedTensorContractScope { .. })
    ));
    assert_eq!(device.to_host().unwrap().data(), expected);

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let zn4 = Arc::new(ZNFusionRule::new(4).unwrap());
    let zn3_leg = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 1)], false).unwrap();
    let zn4_leg = GradedSpace::try_new(Arc::clone(&zn4), [(zn4.irrep(0), 1)], false).unwrap();
    let zn3_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&zn3_leg], [&zn3_leg], |_, _| 1.0).unwrap();
    let zn4_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&zn4_leg], [&zn4_leg], |_, _| 1.0).unwrap();
    let zn3_device = zn3_tensor.to_cuda().unwrap();
    let zn4_device = zn4_tensor.to_cuda().unwrap();
    assert!(zn3_device.compose(&zn4_device).is_err());
    assert_eq!(zn3_device.to_host().unwrap().data(), [1.0]);
    assert_eq!(zn4_device.to_host().unwrap().data(), [1.0]);

    let left_open = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 1)], false).unwrap();
    let seam = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(1), 1)], false).unwrap();
    let right_open = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(2), 1)], false).unwrap();
    let zero_lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&left_open], [&seam], |_, _| 1.0).unwrap();
    let zero_rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&seam], [&right_open], |_, _| 1.0).unwrap();
    assert_eq!(zero_lhs.block_count(), 0);
    assert_eq!(zero_rhs.block_count(), 0);
    let zero_output = zero_lhs
        .to_cuda()
        .unwrap()
        .compose(&zero_rhs.to_cuda().unwrap())
        .unwrap()
        .to_host()
        .unwrap();
    assert_eq!(zero_output.block_count(), 0);
    assert!(zero_output.data().is_empty());
}

#[test]
#[ignore]
fn typed_cuda_mixed_active_and_inactive_destination_blocks_stay_ordered() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let q0 = U1Irrep::new(0);
    let q1 = U1Irrep::new(1);
    let open = GradedSpace::try_new(Arc::clone(&provider), [(q0, 1), (q1, 1)], false).unwrap();
    let seam = GradedSpace::try_new(Arc::clone(&provider), [(q0, 1)], false).unwrap();
    let lhs = TensorMap::from_block_fn(&runtime, [&open], [&seam], |_, _| 2.0).unwrap();
    let rhs = TensorMap::from_block_fn(&runtime, [&seam], [&open], |_, _| 3.0).unwrap();
    assert_eq!(lhs.block_count(), 1);
    assert_eq!(rhs.block_count(), 1);

    let lhs_axes = [1];
    let rhs_axes = [0];
    let output_axes = [0, 1];
    let host_contract = lhs
        .contract(&rhs, &lhs_axes, &rhs_axes, &output_axes)
        .unwrap();
    let host_ordered = lhs
        .contract_ordered(&rhs, &lhs_axes, &rhs_axes, &output_axes)
        .unwrap();
    let host_compose = lhs.compose(&rhs).unwrap();
    let lhs_device = lhs.to_cuda().unwrap();
    let rhs_device = rhs.to_cuda().unwrap();
    let device_contract = lhs_device
        .contract(&rhs_device, &lhs_axes, &rhs_axes, &output_axes)
        .unwrap()
        .to_host()
        .unwrap();
    let device_ordered = lhs_device
        .contract_ordered(&rhs_device, &lhs_axes, &rhs_axes, &output_axes)
        .unwrap()
        .to_host()
        .unwrap();
    let device_compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();

    for (actual, expected) in [
        (&device_contract, &host_contract),
        (&device_ordered, &host_ordered),
        (&device_compose, &host_compose),
    ] {
        assert_eq!(structural_snapshot(actual), structural_snapshot(expected));
        assert_eq!(actual.data(), expected.data());
        assert_eq!(actual.block_count(), 2);
        for (sector, expected_value) in [(q0, 6.0), (q1, 0.0)] {
            let index = (0..actual.block_count())
                .find(|&index| actual.block_fusion_trees(index).unwrap().coupled() == &sector)
                .unwrap();
            let block = actual.block(index).unwrap();
            let len = block.shape().iter().product::<usize>();
            let values = &actual.data()[block.offset()..block.offset() + len];
            assert!(!values.is_empty());
            if sector == q1 {
                assert!(values.iter().all(|value| value.to_bits() == 0));
            } else {
                assert!(values.iter().all(|&value| value == expected_value));
            }
        }
    }
}
