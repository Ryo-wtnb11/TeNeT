//! Real-device gates for provider-neutral typed ownership transfer.
//!
//! Run with `cargo test -p tenet --features cuda,cpu-faer --test \
//! typed_cuda_transfer -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use tenet::core::{
    product_sector, CheckedFusionAlgebra, FermionParityFusionRule, MultiplicityFreeRigidSymbols,
    ProductFusionRuleExt, SU2FusionRule, SU2Irrep, SectorCodec, U1FusionRule, U1Irrep, Z2Irrep,
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
