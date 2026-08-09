//! Public network conformance for representative multiplicity-free providers.
//!
//! The direct typed operations are the oracle: this file checks that the
//! macro and an explicitly planned `Network` preserve their provider and
//! payload, including a static intra-operand trace.

use std::fmt::Debug;
use std::sync::Arc;

use tenet::core::{
    product_sector, BraidingStyleKind, CU1FusionRule, CU1Irrep, CheckedFusionAlgebra,
    FermionParityFusionRule, FusionAlgebraError, FusionRule, FusionStyleKind,
    MultiplicityFreeAdmissionMode, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, ProductFusionRuleExt, RuleIdentity, SectorCodec, SectorId,
    SectorVec, TypedSectorAdmission, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
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

/// An external Z3 provider, deliberately not a TeNeT built-in.  Charge one
/// and two are distinct duals, so codec and dual handling cannot collapse to
/// the Z2/self-dual case.
#[derive(Clone, Copy)]
struct ExternalZ3;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Z3Charge(u8);

impl FusionRule for ExternalZ3 {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(0x5a33_0000_0000_0000, Arc::<[u8]>::from([]))
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        SectorId::new((3 - sector.id() % 3) % 3)
    }
    fn fusion_channels(&self, lhs: SectorId, rhs: SectorId) -> SectorVec {
        core::iter::once(SectorId::new((lhs.id() + rhs.id()) % 3)).collect()
    }
}
impl MultiplicityFreeFusionRule for ExternalZ3 {}
impl MultiplicityFreeFusionSymbols for ExternalZ3 {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> f64 {
        1.0
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        1.0
    }
}
impl MultiplicityFreeRigidSymbols for ExternalZ3 {
    fn dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn twist_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
}
impl CheckedFusionAlgebra for ExternalZ3 {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(self.dual(sector))
    }
    fn try_fusion_channels(
        &self,
        lhs: SectorId,
        rhs: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(self.fusion_channels(lhs, rhs))
    }
    fn try_nsymbol(
        &self,
        lhs: SectorId,
        rhs: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(self.nsymbol(lhs, rhs, coupled))
    }
}
impl SectorCodec for ExternalZ3 {
    type Sector = Z3Charge;
    fn encode_sector(&self, value: &Z3Charge) -> Result<SectorId, FusionAlgebraError> {
        if value.0 < 3 {
            Ok(SectorId::new(value.0.into()))
        } else {
            Err(FusionAlgebraError::InvalidSector {
                sector: SectorId::new(value.0.into()),
            })
        }
    }
    fn decode_sector(&self, sector: SectorId) -> Result<Z3Charge, FusionAlgebraError> {
        u8::try_from(sector.id())
            .ok()
            .filter(|charge| *charge < 3)
            .map(Z3Charge)
            .ok_or(FusionAlgebraError::InvalidSector { sector })
    }
}

fn assert_same<R, D>(actual: &TensorMap<R, D>, expected: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
    D: TensorScalar + PartialEq + Debug,
{
    assert!(std::ptr::eq(actual.provider(), expected.provider()));
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
    let expected = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    let network = Network::new(
        vec![labels(&["i", "j"]), labels(&["j", "k"])],
        vec![false, false],
        vec![Some(1), Some(1)],
        labels(&["i", "k"]),
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
    assert_same(&first, &expected);
    assert_same(&second, &expected);

    // This is workspace reuse only.  The public contract intentionally makes
    // no promise that output payload allocations themselves are reused.
    let cold = plan_cache_stats(runtime);
    let macro_first = tensor!([i; k] = lhs[i; j] * rhs[j; k]).unwrap();
    let macro_second = tensor!([i; k] = lhs[i; j] * rhs[j; k]).unwrap();
    assert_same(&macro_first, &expected);
    assert_same(&macro_second, &expected);
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
    assert_same(&actual, &expected);
}

#[test]
fn multiplicity_free_public_network_path_matches_typed_oracles() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();

    let z2 = GradedSpace::try_new(
        Arc::new(Z2FusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
        false,
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(&runtime, &z2, 1002);
    static_trace_matches_typed_oracle(&runtime, &z2, 1003);

    let z3_provider = Arc::new(ExternalZ3);
    let z3 = GradedSpace::try_new(
        z3_provider,
        [(Z3Charge(0), 1), (Z3Charge(1), 2), (Z3Charge(2), 1)],
        false,
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(&runtime, &z3, 1010);
    static_trace_matches_typed_oracle(&runtime, &z3, 1011);

    // A charged CU(1) leg exercises the nontrivial pseudo-scalar provider,
    // rather than a vacuum-only dense block.
    let cu1 = GradedSpace::try_new(
        Arc::new(CU1FusionRule),
        [
            (CU1Irrep::from_twice_charge(1), 1),
            (CU1Irrep::from_twice_charge(3), 2),
        ],
        false,
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(&runtime, &cu1, 1004);
    static_trace_matches_typed_oracle(&runtime, &cu1, 1005);

    let product_rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let product = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
        ],
        false,
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
            .product(Z2FusionRule),
    );
    let nested = GradedSpace::try_new(
        nested_rule,
        [
            (
                product_sector(
                    product_sector(Z2Irrep::EVEN, U1Irrep::new(0)),
                    Z2Irrep::EVEN,
                ),
                2,
            ),
            (
                product_sector(product_sector(Z2Irrep::ODD, U1Irrep::new(1)), Z2Irrep::ODD),
                1,
            ),
        ],
        false,
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(&runtime, &nested, 1008);
    static_trace_matches_typed_oracle(&runtime, &nested, 1009);
}
