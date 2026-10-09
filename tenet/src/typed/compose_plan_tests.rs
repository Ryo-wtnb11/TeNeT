//! `ComposePlan` pins that need private stack fields (#1775): the errors of
//! corrupted stacks, and the CUDA zero regions of the inactive destination
//! blocks against a hand derivation (#1980).

#[cfg(feature = "cuda")]
use std::collections::HashSet;
use std::sync::Arc;

use super::*;
use crate::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use crate::typed::GradedSpace;

fn u1_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let q = U1Irrep::new;
    let rule = Arc::new(U1FusionRule);
    (
        GradedSpace::try_new(Arc::clone(&rule), [(q(-1), 2), (q(0), 1), (q(1), 3)]).unwrap(),
        GradedSpace::try_new(rule, [(q(0), 2), (q(1), 1)]).unwrap(),
    )
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn su2_legs() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let j = SU2Irrep::from_twice_spin;
    let rule = Arc::new(SU2FusionRule);
    (
        GradedSpace::try_new(Arc::clone(&rule), [(j(0), 2), (j(1), 2), (j(2), 1)]).unwrap(),
        GradedSpace::try_new(rule, [(j(0), 1), (j(2), 2)]).unwrap(),
    )
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn fz2u1_legs() -> (
    GradedSpace<
        impl MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    >,
    GradedSpace<
        impl MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    >,
) {
    let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let even = |charge| product_sector(Z2Irrep::EVEN, U1Irrep::new(charge));
    let odd = |charge| product_sector(Z2Irrep::ODD, U1Irrep::new(charge));
    (
        GradedSpace::try_new(Arc::clone(&rule), [(even(0), 2), (odd(1), 1), (odd(-1), 2)]).unwrap(),
        GradedSpace::try_new(rule, [(even(0), 1), (odd(1), 2)]).unwrap(),
    )
}

#[test]
fn corrupted_stack_errors_are_pinned() {
    // What: a stack whose payload disagrees with its structure (reachable
    // only through these private fields) is rejected with a typed error, and
    // a destination likewise; the observed errors are pinned row by row.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (v, w) = u1_legs();
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w], 5).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&v], 6).unwrap();
    let lhs = StackedTensorMap::pack(&[&a, &a]).unwrap();
    let rhs = StackedTensorMap::pack(&[&b, &b]).unwrap();
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let mut workspace = plan.workspace().unwrap();
    plan.execute(&lhs, &rhs, &mut workspace).unwrap();
    let mut dst = workspace.take_output().unwrap();

    // Length and structure shrunk together: only the executor sees it.
    let mut consistent = StackedTensorMap::pack(&[&a, &a]).unwrap();
    consistent.member_len -= 1;
    consistent
        .storage
        .truncate(consistent.member_len * consistent.members);
    // Payload one entry short of its structure.
    let mut short = StackedTensorMap::pack(&[&a, &a]).unwrap();
    short.storage.pop();
    let mut short_dst = StackedTensorMap::pack(&[&a, &a]).unwrap();
    short_dst.space = dst.space.clone();
    short_dst.signature = dst.signature.clone();
    short_dst.member_len = dst.member_len;
    short_dst.storage = vec![1.0; dst.storage.len() - 1];

    let show = |result: Result<(), Error>| format!("{:?}", result.err());
    let observed = [
        (
            "execute: consistent short lhs",
            show(plan.execute(&consistent, &rhs, &mut workspace).map(|_| ())),
        ),
        (
            "execute: payload shorter than structure",
            show(plan.execute(&short, &rhs, &mut workspace).map(|_| ())),
        ),
        (
            "execute_into: payload shorter than structure",
            show(plan.execute_into(&short, &rhs, &mut dst, &mut workspace)),
        ),
        (
            "execute_into: destination shorter than structure",
            show(plan.execute_into(&lhs, &rhs, &mut short_dst, &mut workspace)),
        ),
    ];
    for (row, error) in &observed {
        eprintln!("    ({row:?}, {error:?}),");
    }
    assert!(short_dst.storage.iter().all(|&value| value == 1.0));
    assert_eq!(
        observed.map(|(row, error)| (row, error)),
        CORRUPTED.map(|(row, error)| (row, error.to_string()))
    );
}

#[rustfmt::skip]
const CORRUPTED: [(&str, &str); 4] = [
    ("execute: consistent short lhs", "Some(Operation(InvalidArgument { message: \"stacked operand does not match the replay's members or layout\" }))"),
    ("execute: payload shorter than structure", "Some(Operation(ElementCountMismatch { expected: 64, actual: 63 }))"),
    ("execute_into: payload shorter than structure", "Some(Operation(ElementCountMismatch { expected: 64, actual: 63 }))"),
    ("execute_into: destination shorter than structure", "Some(Operation(ElementCountMismatch { expected: 78, actual: 77 }))"),
];

/// The inactive destination blocks of `a · b` as device zero regions over a
/// `members`-member stack, derived from the public block API alone: a block
/// of the eager result whose coupled sector is not a coupled sector of both
/// operands receives no GEMM; its region is the block's shape, strides and
/// offset with a trailing member axis of stride `member_len`.
#[cfg(feature = "cuda")]
fn hand_regions<R, D>(
    a: &TensorMap<R, D>,
    b: &TensorMap<R, D>,
    members: usize,
) -> Vec<(Vec<usize>, Vec<usize>, usize)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let coupled = |t: &TensorMap<R, D>| {
        (0..t.subblock_count())
            .map(|i| format!("{:?}", t.subblock_fusion_trees(i).unwrap().coupled()))
            .collect::<HashSet<_>>()
    };
    let active: HashSet<_> = coupled(a).intersection(&coupled(b)).cloned().collect();
    let c = a.compose(b).unwrap();
    let member_len = c.dense_data().unwrap().len();
    let mut regions: Vec<_> = (0..c.subblock_count())
        .filter(|&i| {
            !active.contains(&format!(
                "{:?}",
                c.subblock_fusion_trees(i).unwrap().coupled()
            ))
        })
        .map(|i| {
            let block = c.subblock(i).unwrap();
            let mut dims = block.shape().to_vec();
            dims.push(members);
            let mut strides = block.strides().to_vec();
            strides.push(member_len);
            (dims, strides, block.offset())
        })
        .collect();
    regions.sort();
    assert!(!regions.is_empty(), "the fixture has inactive blocks");
    regions
}

#[cfg(feature = "cuda")]
fn regions_match_hand_derivation<R>(runtime: &Runtime, (v, w): (GradedSpace<R>, GradedSpace<R>))
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let a = TensorMap::<_, f64>::rand_with_seed(runtime, [&v, &v], [&w], 11).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(runtime, [&w], [&v], 12).unwrap();
    for members in [1, 2, 17] {
        let lhs = StackedTensorMap::pack(&vec![&a; members])
            .unwrap()
            .to_cuda()
            .unwrap();
        let rhs = StackedTensorMap::pack(&vec![&b; members])
            .unwrap()
            .to_cuda()
            .unwrap();
        let plan = ComposePlan::new(&lhs, &rhs).unwrap();
        let mut workspace = plan.workspace().unwrap();
        let mut lease = runtime.lease_cuda().unwrap();
        plan.prepare_zero_regions(&mut workspace, &mut lease, members)
            .unwrap();
        let mut observed: Vec<_> = workspace
            .device
            .zero_regions
            .iter()
            .map(|region| {
                (
                    region.dims().to_vec(),
                    region.strides().to_vec(),
                    region.offset(),
                )
            })
            .collect();
        observed.sort();
        assert_eq!(observed, hand_regions(&a, &b, members), "B={members}");
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_zero_regions_match_the_hand_derivation() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    regions_match_hand_derivation(&runtime, u1_legs());
    regions_match_hand_derivation(&runtime, su2_legs());
    regions_match_hand_derivation(&runtime, fz2u1_legs());
}
