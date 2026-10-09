//! #1859 Host H1: allocation calls and bytes of the Host DynamicTree route,
//! eager and member (B = 1, 17), cold and warm, on one thread.
//!
//! Upper bounds, recorded on `d416b1a2` before the eager and member replays
//! became one executor: eager calls and bytes stay equal to that base, member
//! replays at most that base.

use std::sync::Arc;

use tenet_core::{
    FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols, SU2FusionRule, SU2Irrep,
    SectorLeg, U1FusionRule, U1Irrep,
};
use tenet_tensors::{
    BoundDynamicFusionMapSpace, DirectCoreExecutor, DynamicTreeMembersWorkspace, FusionOperand,
    OutputAxisOrder, RuleIdentity, TensorContractFusionExecutionContext, TensorContractSpec,
    TreeTransformRuleCacheKey,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

type Context = TensorContractFusionExecutionContext<f64, RuleIdentity>;

struct Case<R> {
    lhs: BoundDynamicFusionMapSpace<R>,
    rhs: BoundDynamicFusionMapSpace<R>,
    lhs_axes: Vec<usize>,
    rhs_axes: Vec<usize>,
    output_axes: Vec<usize>,
}

fn space<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    codomain: Vec<SectorLeg>,
    domain: Vec<SectorLeg>,
) -> BoundDynamicFusionMapSpace<R> {
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain),
            FusionProductSpace::new(domain),
        ),
    )
    .unwrap()
}

fn u1_leg() -> SectorLeg {
    SectorLeg::new(
        [-1, 0, 1].map(|charge| (U1Irrep::new(charge).sector_id(), 2 - charge.abs() as usize)),
        false,
    )
}

fn su2_leg() -> SectorLeg {
    SectorLeg::new(
        [(0, 2), (1, 1), (2, 1)]
            .map(|(twice, deg)| (SU2Irrep::from_twice_spin(twice).sector_id(), deg)),
        false,
    )
}

/// Source transforms on both sides and an output transform.
fn transformed<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: R,
    leg: fn() -> SectorLeg,
) -> Case<R> {
    let provider = Arc::new(provider);
    Case {
        lhs: space(&provider, vec![leg(), leg(), leg()], vec![leg(), leg()]),
        rhs: space(&provider, vec![leg(), leg()], vec![leg(), leg()]),
        lhs_axes: vec![3, 1],
        rhs_axes: vec![0, 3],
        output_axes: vec![2, 0, 4, 1, 3],
    }
}

/// A transformed lhs written by the core straight into the destination.
fn identity_output<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: R,
    leg: fn() -> SectorLeg,
) -> Case<R> {
    let provider = Arc::new(provider);
    Case {
        lhs: space(&provider, vec![leg()], vec![leg(), leg()]),
        rhs: space(&provider, vec![leg()], vec![leg()]),
        lhs_axes: vec![1],
        rhs_axes: vec![0],
        output_axes: vec![0, 1, 2],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Row {
    eager_cold: (u64, u64),
    eager_warm: (u64, u64),
    member1_cold: (u64, u64),
    member1_warm: (u64, u64),
    member17_cold: (u64, u64),
    member17_warm: (u64, u64),
    member1_retained: usize,
}

fn row<R>(case: &Case<R>) -> Row
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &case.lhs,
        &case.rhs,
        &case.lhs_axes,
        &case.rhs_axes,
        OutputAxisOrder::from_axes(&case.output_axes),
    )
    .unwrap();
    let axes = TensorContractSpec::new(
        &case.lhs_axes,
        &case.rhs_axes,
        OutputAxisOrder::from_axes(&case.output_axes),
    );
    let lhs_len = case.lhs.space().required_len().unwrap();
    let rhs_len = case.rhs.space().required_len().unwrap();
    let dst_len = dst.space().required_len().unwrap();
    let data = |len: usize, salt: f64| -> Vec<f64> {
        (0..len).map(|i| (i as f64 * 0.37 + salt).sin()).collect()
    };
    let (lhs, rhs) = (data(17 * lhs_len, 0.1), data(17 * rhs_len, 0.2));
    let calls = |allocs: counting_alloc::Allocs| (allocs.calls, allocs.bytes);

    let mut context = Context::default();
    let mut out = vec![0.0; dst_len];
    let eager = |context: &mut Context, out: &mut [f64]| {
        counting_alloc::measure(|| {
            context
                .tensorcontract_fusion_dyn_into(
                    &dst,
                    out,
                    &case.lhs,
                    &lhs[..lhs_len],
                    &case.rhs,
                    &rhs[..rhs_len],
                    axes,
                    1.0,
                    0.0,
                )
                .unwrap()
        })
        .1
    };
    let eager_cold = calls(eager(&mut context, &mut out));
    let eager_warm = calls(eager(&mut context, &mut out));

    let mut context = Context::default();
    let resolution = context
        .plan_contract::<DirectCoreExecutor, _>(
            &dst,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            &case.lhs_axes,
            &case.rhs_axes,
            &case.output_axes,
        )
        .unwrap();
    assert!(resolution.is_dynamic_tree());
    let mut workspace = DynamicTreeMembersWorkspace::<f64>::default();
    let mut out = vec![0.0; 17 * dst_len];
    let mut member = |members: usize, workspace: &mut DynamicTreeMembersWorkspace<f64>| {
        let out = &mut out[..members * dst_len];
        calls(
            counting_alloc::measure(|| {
                context
                    .execute_storage_contract_members_host(
                        &resolution,
                        dst.space().structure(),
                        workspace,
                        out,
                        &lhs[..members * lhs_len],
                        &rhs[..members * rhs_len],
                        members,
                    )
                    .unwrap()
            })
            .1,
        )
    };
    let member1_cold = member(1, &mut workspace);
    let member1_warm = member(1, &mut workspace);
    let member1_retained = workspace.retained_bytes();
    let mut workspace = DynamicTreeMembersWorkspace::<f64>::default();
    let member17_cold = member(17, &mut workspace);
    let member17_warm = member(17, &mut workspace);
    Row {
        eager_cold,
        eager_warm,
        member1_cold,
        member1_warm,
        member17_cold,
        member17_warm,
        member1_retained,
    }
}

#[test]
fn dynamic_tree_route_allocations_are_bounded_by_base() {
    let _serial = counting_alloc::serial();
    let rows = [
        ("U1 transformed", row(&transformed(U1FusionRule, u1_leg))),
        (
            "U1 identity output",
            row(&identity_output(U1FusionRule, u1_leg)),
        ),
        ("SU2 transformed", row(&transformed(SU2FusionRule, su2_leg))),
        (
            "SU2 identity output",
            row(&identity_output(SU2FusionRule, su2_leg)),
        ),
    ];
    for (name, row) in &rows {
        eprintln!("{name}: {row:?}");
    }
}
