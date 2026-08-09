//! Public network conformance for representative multiplicity-free providers.
//!
//! The direct typed operations are the oracle: this file checks that the
//! macro and an explicitly planned `Network` preserve their provider and
//! payload, including a static intra-operand trace.

use std::fmt::Debug;
use std::sync::Arc;

use tenet::core::{
    product_sector, CU1FusionRule, CU1Irrep, CheckedFusionAlgebra, FermionParityFusionRule,
    FusionAlgebraError, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols,
    ProductFusionRuleExt, SU2FusionRule, SU2Irrep, SectorCodec, TypedSectorAdmission, U1FusionRule,
    U1Irrep, Z2FusionRule, Z2Irrep, ZNFusionRule,
};
use tenet::prelude::TensorScalar;
use tenet::typed::{GradedSpace, Runtime, TensorMap};
use tenet_network::{
    plan_cache_stats, tensor, GreedyDenseOptimizer, Network, NetworkExecutionWorkspace,
    TemporaryLabel,
};

fn labels(names: &[&str]) -> Vec<TemporaryLabel> {
    names.iter().copied().map(TemporaryLabel::from).collect()
}

fn assert_same<R, D>(actual: &TensorMap<R, D>, expected: &TensorMap<R, D>, provider: &R)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
    D: TensorScalar + PartialEq + Debug,
{
    assert!(std::ptr::eq(actual.provider(), provider));
    assert!(std::ptr::eq(expected.provider(), provider));
    assert!(actual
        .codomain()
        .iter()
        .all(|leg| std::ptr::eq(leg.provider(), provider)));
    assert!(actual
        .domain()
        .iter()
        .all(|leg| std::ptr::eq(leg.provider(), provider)));
    assert_eq!(actual.codomain(), expected.codomain());
    assert_eq!(actual.domain(), expected.domain());
    assert_eq!(actual.data(), expected.data());
}

fn ordinary_network_and_workspace_reuse<R>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let lhs = TensorMap::<R, f64>::rand_with_seed(runtime, [space], [space], seed).unwrap();
    let rhs = TensorMap::<R, f64>::rand_with_seed(runtime, [space], [space], seed + 1).unwrap();
    let expected = lhs
        .contract(&rhs, &[1], &[0], &[0, 1])
        .unwrap()
        .permute(&[1], &[0])
        .unwrap();
    let network = Network::new(
        vec![labels(&["i", "j"]), labels(&["j", "k"])],
        vec![false, false],
        vec![Some(1), Some(1)],
        labels(&["k", "i"]),
        Some(1),
    )
    .unwrap();
    let tensors = [&lhs, &rhs];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    let first = planned
        .execute_with_workspace(&tensors, &mut workspace)
        .unwrap();
    let second = planned
        .execute_with_workspace(&tensors, &mut workspace)
        .unwrap();
    assert_same(&first, &expected, space.provider());
    assert_same(&second, &expected, space.provider());

    // This is workspace reuse only.  The public contract intentionally makes
    // no promise that output payload allocations themselves are reused.
    let cold = plan_cache_stats(runtime);
    let macro_first = tensor!([k; i] = lhs[i; j] * rhs[j; k]).unwrap();
    let macro_second = tensor!([k; i] = lhs[i; j] * rhs[j; k]).unwrap();
    assert_same(&macro_first, &expected, space.provider());
    assert_same(&macro_second, &expected, space.provider());
    assert!(plan_cache_stats(runtime).workspace_reuses > cold.workspace_reuses);
}

fn static_trace_matches_typed_oracle<R>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let dual = space.try_dual().unwrap();
    let source =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, &dual], [space], seed).unwrap();
    let expected = source.trace_pairs(&[(0, 1)]).unwrap();
    let actual = tensor!([; out] = source[i, i; out]).unwrap();
    assert_same(&actual, &expected, space.provider());
}

#[test]
fn multiplicity_free_public_network_path_matches_typed_oracles() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();

    let z2 = GradedSpace::try_new_with_shared_provider(
        Arc::new(Z2FusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(&runtime, &z2, 1002);
    static_trace_matches_typed_oracle(&runtime, &z2, 1003);

    let z3_provider = Arc::new(ZNFusionRule::new(3).unwrap());
    let z3 = GradedSpace::try_new_with_shared_provider(
        Arc::clone(&z3_provider),
        [
            (z3_provider.irrep(0), 1),
            (z3_provider.irrep(1), 2),
            (z3_provider.irrep(2), 1),
        ],
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(&runtime, &z3, 1010);
    static_trace_matches_typed_oracle(&runtime, &z3, 1011);

    // A charged CU(1) leg exercises the nontrivial pseudo-scalar provider,
    // rather than a vacuum-only dense block.
    let cu1 = GradedSpace::try_new_with_shared_provider(
        Arc::new(CU1FusionRule),
        [
            (CU1Irrep::VACUUM, 1),
            (CU1Irrep::PSEUDOSCALAR, 2),
            (CU1Irrep::from_twice_charge(1), 1),
        ],
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(&runtime, &cu1, 1004);
    static_trace_matches_typed_oracle(&runtime, &cu1, 1005);

    let product_rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let product = GradedSpace::try_new_with_shared_provider(
        Arc::clone(&product_rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
        ],
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(&runtime, &product, 1006);
    static_trace_matches_typed_oracle(&runtime, &product, 1007);
    // The odd fZ2 sector contributes with the supertrace sign even after the
    // U(1) factor is attached: (2 + 3) - 7 = -2.  This is intentionally a
    // hand oracle, not another call to `trace_pairs`.
    let product_diagonal =
        TensorMap::from_block_fn(&runtime, [&product], [&product], |trees, i| {
            if i[0] != i[1] {
                0.0
            } else if *trees.coupled() == product_sector(Z2Irrep::EVEN, U1Irrep::new(0)) {
                2.0 + i[0] as f64
            } else {
                7.0
            }
        })
        .unwrap();
    assert_eq!(
        tensor!([] = product_diagonal[i; i])
            .unwrap()
            .scalar()
            .unwrap(),
        -2.0
    );

    let nested_rule = Arc::new(
        FermionParityFusionRule
            .product(U1FusionRule)
            .product(SU2FusionRule),
    );
    let nested = GradedSpace::try_new_with_shared_provider(
        nested_rule,
        [
            (
                product_sector(
                    product_sector(Z2Irrep::EVEN, U1Irrep::new(0)),
                    SU2Irrep::from_twice_spin(0),
                ),
                2,
            ),
            (
                product_sector(
                    product_sector(Z2Irrep::ODD, U1Irrep::new(1)),
                    SU2Irrep::from_twice_spin(1),
                ),
                1,
            ),
        ],
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(&runtime, &nested, 1008);
    static_trace_matches_typed_oracle(&runtime, &nested, 1009);
}
