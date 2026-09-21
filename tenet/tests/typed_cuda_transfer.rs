//! Real-device gates for provider-neutral typed ownership transfer.
//!
//! Run with `cargo test -p tenet --features cuda,cpu-faer --test \
//! typed_cuda_transfer -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use num_complex::Complex64;

use tenet::core::{
    product_sector, BraidingStyleKind, CheckedFusionAlgebra, FermionParityFusionRule,
    FusionAlgebraError, FusionRule, FusionStyleKind, MultiplicityFreeFusionRule,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, ProductFusionRuleExt,
    RuleIdentity, SU2FusionRule, SU2Irrep, SectorCodec, SectorId, SectorVec, U1FusionRule, U1Irrep,
    Z2Irrep, ZNFusionRule,
};
use tenet::prelude::TensorScalar;
use tenet::typed::{BlockFusionTrees, CudaStorage, GradedSpace, Runtime, TensorMap, Truncation};

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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ProbeSector;

/// Tiny device operation for the canary below: `to_cuda` takes the Runtime's
/// device lease, on a provider that never calls back into the Runtime.
fn device_lease_probe(runtime: &Runtime) {
    let leg = GradedSpace::try_new(U1FusionRule, [(U1Irrep::new(0), 1)]).unwrap();
    let probe: TensorMap<_, f64> = TensorMap::zeros(runtime, [&leg], [&leg]).unwrap();
    probe.to_cuda().unwrap();
}

/// One-sector provider whose dimension callback re-enters the Runtime's device
/// lease. A reduction deadlocks here if it calls provider code under that
/// lease. (Before #1281 the callback re-entered the coarse state lock through
/// `cuda_device_ordinal`, which no longer locks anything.)
struct ReentrantDimensionRule {
    runtime: Runtime,
    calls: Arc<AtomicUsize>,
}

impl FusionRule for ReentrantDimensionRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(0x7520_0000_0000_0001, Arc::<[u8]>::from([]))
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

    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        core::iter::once(SectorId::new(0)).collect()
    }
}

impl MultiplicityFreeFusionRule for ReentrantDimensionRule {}

impl MultiplicityFreeFusionSymbols for ReentrantDimensionRule {
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

impl MultiplicityFreeRigidSymbols for ReentrantDimensionRule {
    fn dim_scalar(&self, _: SectorId) -> f64 {
        assert_eq!(self.runtime.cuda_device_ordinal(), Some(0));
        device_lease_probe(&self.runtime);
        self.calls.fetch_add(1, Ordering::SeqCst);
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

impl CheckedFusionAlgebra for ReentrantDimensionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(self.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(self.nsymbol(left, right, coupled))
    }
}

impl SectorCodec for ReentrantDimensionRule {
    type Sector = ProbeSector;

    fn encode_sector(&self, _: &ProbeSector) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(0))
    }

    fn decode_sector(&self, sector: SectorId) -> Result<ProbeSector, FusionAlgebraError> {
        if sector == SectorId::new(0) {
            Ok(ProbeSector)
        } else {
            Err(FusionAlgebraError::InvalidSector { sector })
        }
    }
}

fn structural_snapshot<R, D>(tensor: &TensorMap<R, D>) -> StructuralSnapshot<R::Sector>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let leg_snapshot = |leg: GradedSpace<R>| LegSnapshot {
        sectors: leg.sectors().unwrap(),
        degeneracies: leg.degeneracies().to_vec(),
        is_dual: leg.is_dual(),
    };
    StructuralSnapshot {
        codomain: tensor.codomain().into_iter().map(&leg_snapshot).collect(),
        domain: tensor.domain().into_iter().map(leg_snapshot).collect(),
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

#[allow(deprecated)]
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

#[allow(deprecated)]
fn assert_reduction_parity<R>(lhs: &TensorMap<R, f64>, rhs: &TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let expected_inner = lhs.inner(rhs).unwrap();
    let expected_norm = lhs.norm().unwrap();
    let lhs_device = lhs.to_cuda().unwrap();
    let rhs_device = rhs.to_cuda().unwrap();
    let inner = lhs_device.inner(&rhs_device).unwrap();
    let norm = lhs_device.norm().unwrap();
    let tolerance = 1e-12 * (1.0 + expected_inner.abs().max(expected_norm));

    assert!((inner - expected_inner).abs() <= tolerance);
    assert_eq!(lhs_device.dot(&rhs_device).unwrap(), inner);
    assert!((norm - expected_norm).abs() <= tolerance);
    let self_inner = lhs_device.inner(&lhs_device).unwrap();
    assert!((self_inner - norm.powi(2)).abs() <= 1e-12 * (1.0 + self_inner.abs()));
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "actual {actual:?}, expected {expected:?}"
        );
    }
}

fn assert_typed_cuda_qr_matches_host<R>(source: &TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = source.runtime().identity();
    let source_bits: Vec<_> = source.data().iter().map(|value| value.to_bits()).collect();
    let (expected_left, expected_right) = source.qr_compact().unwrap();
    let source_device = source.to_cuda().unwrap();
    let (left_device, right_device) = source_device.qr_compact().unwrap();

    for factor in [&left_device, &right_device] {
        assert!(std::ptr::eq(factor.provider(), provider));
        assert!(runtime.matches(factor.runtime()));
        assert_eq!(factor.placement(), tenet::core::Placement::Cuda(0));
    }
    let left = left_device.to_host().unwrap();
    let right = right_device.to_host().unwrap();
    assert_close(left.data(), expected_left.data(), 1e-10);
    assert_close(right.data(), expected_right.data(), 1e-10);
    assert_eq!(
        structural_snapshot(&left),
        structural_snapshot(&expected_left)
    );
    assert_eq!(
        structural_snapshot(&right),
        structural_snapshot(&expected_right)
    );
    let rebuilt = left.compose(&right).unwrap();
    assert_close(rebuilt.data(), source.data(), 1e-10);
    assert_eq!(
        source_device
            .to_host()
            .unwrap()
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        source_bits
    );
}

fn assert_typed_cuda_svd_matches_host<R>(source: &TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let source_data = source.data().to_vec();
    let source_device = source.to_cuda().unwrap();
    let expected = source.svd_compact().unwrap();
    assert_cuda_svd_result(source, &expected, source_device.svd_compact().unwrap());
    assert_eq!(source_device.to_host().unwrap().data(), source_data);
}

/// Host and device `(u, s, vh)` triples, as `svd_compact` returns them.
type HostSvdFactors<R, D> = (TensorMap<R, D>, TensorMap<R, D>, TensorMap<R, D>);
type DeviceSvdFactors<R, D> = (
    TensorMap<R, D, CudaStorage<D>>,
    TensorMap<R, D, CudaStorage<D>>,
    TensorMap<R, D, CudaStorage<D>>,
);

fn assert_cuda_svd_result<R>(
    source: &TensorMap<R, f64>,
    expected: &HostSvdFactors<R, f64>,
    factors: DeviceSvdFactors<R, f64>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = source.runtime().identity();
    for factor in [&factors.0, &factors.1, &factors.2] {
        assert!(std::ptr::eq(factor.provider(), provider));
        assert!(runtime.matches(factor.runtime()));
        assert_eq!(factor.placement(), tenet::core::Placement::Cuda(0));
    }
    let actual = (
        factors.0.to_host().unwrap(),
        factors.1.to_host().unwrap(),
        factors.2.to_host().unwrap(),
    );
    assert_close(actual.1.data(), expected.1.data(), 1e-10);
    assert_eq!(
        structural_snapshot(&actual.0),
        structural_snapshot(&expected.0)
    );
    assert_eq!(
        structural_snapshot(&actual.1),
        structural_snapshot(&expected.1)
    );
    assert_eq!(
        structural_snapshot(&actual.2),
        structural_snapshot(&expected.2)
    );
    assert!(actual.0.is_isometric(1e-10).unwrap());
    assert!(actual.2.adjoint().unwrap().is_isometric(1e-10).unwrap());
    let rebuilt = actual
        .0
        .compose(&actual.1)
        .unwrap()
        .compose(&actual.2)
        .unwrap();
    assert_close(rebuilt.data(), source.data(), 1e-10);
}

/// The device replacement for `svd_trunc`: compact SVD on the device, one
/// download per factor, then the Host truncation primitives. It must reproduce
/// Host `svd_trunc` in the kept bond space and in the kept spectrum, and
/// reconstruct `u s vh` to dtype tolerance. The device factors keep the raw
/// cuSOLVER gauge, so `u`/`vh` payloads are never compared bit for bit.
fn assert_typed_cuda_svd_trunc_composition_matches_host<R>(
    source: &TensorMap<R, f64>,
    truncation: &Truncation,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = source.runtime().identity();
    let source_bits: Vec<_> = source.data().iter().map(|value| value.to_bits()).collect();
    let expected = source.svd_trunc(truncation).unwrap();
    let source_device = source.to_cuda().unwrap();

    let (u_device, s_device, vh_device) = source_device.svd_compact().unwrap();
    for factor in [&u_device, &s_device, &vh_device] {
        assert!(std::ptr::eq(factor.provider(), provider));
        assert!(runtime.matches(factor.runtime()));
        assert_eq!(factor.placement(), tenet::core::Placement::Cuda(0));
    }
    let u = u_device.to_host().unwrap();
    let s = s_device.to_host().unwrap();
    let vh = vh_device.to_host().unwrap();

    let found = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), truncation)
        .unwrap();
    let selection = &found.selection;
    let u = u.restrict_leg(u.codomain_rank(), selection).unwrap();
    let s = s.restrict_diagonal(selection).unwrap();
    let vh = vh.restrict_leg(0, selection).unwrap();

    // The kept bond space is exactly Host's, including the empty bond of a
    // discard-all policy.
    assert_eq!(*selection.subspace(), expected.s.domain()[0]);
    for (actual, expected) in [(&u, &expected.u), (&s, &expected.s), (&vh, &expected.vh)] {
        assert_eq!(structural_snapshot(actual), structural_snapshot(expected));
    }
    // Per sector and matched by provider label: `diagview` orders by encoded
    // `SectorId`, Host `singular_values` by decoded label, and those two
    // orders are not the same contract.
    let mut kept = s.diagview().unwrap();
    kept.sort_by(|left, right| left.sector.cmp(&right.sector));
    assert_eq!(kept.len(), expected.singular_values.len());
    for (actual, expected) in kept.iter().zip(&expected.singular_values) {
        assert_eq!(actual.sector, expected.sector);
        assert_close(&actual.values, &expected.values, 1e-10);
    }
    assert!((found.error - expected.error).abs() <= 1e-10 * (1.0 + expected.error));

    assert!(u.is_isometric(1e-10).unwrap());
    assert!(vh.adjoint().unwrap().is_isometric(1e-10).unwrap());
    let actual_rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    let expected_rebuilt = expected
        .u
        .compose(&expected.s)
        .unwrap()
        .compose(&expected.vh)
        .unwrap();
    assert_close(actual_rebuilt.data(), expected_rebuilt.data(), 1e-10);
    assert_eq!(
        source_device
            .to_host()
            .unwrap()
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        source_bits
    );
}

fn assert_finite_r_diagonal_nonnegative<R>(right: &TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    for block_index in 0..right.block_count() {
        let block = right.block(block_index).unwrap();
        let diagonal_len = block.shape()[0].min(block.shape()[1]);
        for index in 0..diagonal_len {
            let value = right.data()
                [block.offset() + index * block.strides()[0] + index * block.strides()[1]];
            if value.is_finite() {
                assert!(value >= 0.0, "negative finite R diagonal {value:?}");
            }
        }
    }
}

#[test]
#[ignore]
fn builtin_and_simple_product_providers_share_one_transfer_path() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();

    let u1_rule = Arc::new(U1FusionRule);
    let u1 = GradedSpace::try_new_with_arc(
        Arc::clone(&u1_rule),
        [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2)],
    )
    .unwrap();
    assert_roundtrip(
        TensorMap::from_block_fn(&runtime, [&u1], [&u1], |_, indices| indices[0] as f64 + 1.0)
            .unwrap(),
    );

    let fz2_rule = Arc::new(FermionParityFusionRule);
    let fz2 = GradedSpace::try_new_with_arc(
        Arc::clone(&fz2_rule),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)],
    )
    .unwrap();
    assert_roundtrip(
        TensorMap::from_block_fn(&runtime, [&fz2], [&fz2], |_, indices| {
            indices[0] as f64 + 2.0
        })
        .unwrap(),
    );

    let su2_rule = Arc::new(SU2FusionRule);
    let su2 = GradedSpace::try_new_with_arc(
        Arc::clone(&su2_rule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    assert_roundtrip(
        TensorMap::from_block_fn(&runtime, [&su2], [&su2], |_, indices| {
            indices[0] as f64 + 3.0
        })
        .unwrap(),
    );

    let product_rule = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product = GradedSpace::try_new_with_arc(
        Arc::clone(&product_rule),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 1),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 2),
        ],
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
    let u1 = GradedSpace::try_new_with_arc(
        Arc::clone(&u1_rule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
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
    let su2 = GradedSpace::try_new_with_arc(
        Arc::clone(&su2_rule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
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
    let product = GradedSpace::try_new_with_arc(
        Arc::clone(&product_rule),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
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

    let product_dual = GradedSpace::try_new_with_arc(
        Arc::clone(&product_rule),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .and_then(|space| space.try_dual())
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
#[ignore = "requires a real CUDA device"]
fn typed_cuda_reductions_cover_weights_providers_lazy_and_preflight() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let u1_provider = Arc::new(U1FusionRule);
    let u1_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&u1_provider),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let u1_lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&u1_leg], [&u1_leg], |trees, _| {
            if trees.coupled() == &U1Irrep::new(0) {
                1.0
            } else {
                2.0
            }
        })
        .unwrap();
    let u1_rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&u1_leg], [&u1_leg], |trees, _| {
            if trees.coupled() == &U1Irrep::new(0) {
                3.0
            } else {
                4.0
            }
        })
        .unwrap();
    assert_eq!(u1_lhs.inner(&u1_rhs).unwrap(), 11.0);
    assert_reduction_parity(&u1_lhs, &u1_rhs);

    let u1_device = u1_lhs.to_cuda().unwrap();
    let u1_norm = u1_device.norm().unwrap();
    let lazy = u1_device.adjoint().unwrap();
    assert!((lazy.norm().unwrap() - u1_norm).abs() < 1e-12);
    assert!(matches!(
        lazy.inner(&lazy),
        Err(tenet::typed::Error::UnsupportedOnDevice(_))
    ));
    let u1_rhs_device = u1_rhs.to_cuda().unwrap();
    let repeated = u1_device.inner(&u1_rhs_device).unwrap();
    for _ in 0..3 {
        assert_eq!(u1_device.inner(&u1_rhs_device).unwrap(), repeated);
    }
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| u1_device.inner(&u1_rhs_device).unwrap()))
            .collect();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), repeated);
        }
    });

    let su2_provider = Arc::new(SU2FusionRule);
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let su2_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&su2_provider), [(spin0, 1), (spin_half, 1)])
            .unwrap();
    let su2_lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&su2_leg], [&su2_leg], |trees, _| {
            if trees.coupled() == &spin0 {
                1.0
            } else {
                2.0
            }
        })
        .unwrap();
    let su2_rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&su2_leg], [&su2_leg], |trees, _| {
            if trees.coupled() == &spin0 {
                3.0
            } else {
                4.0
            }
        })
        .unwrap();
    // dim(j=0) * 1 * 3 + dim(j=1/2) * 2 * 4 = 1 * 3 + 2 * 8.
    assert_eq!(su2_lhs.inner(&su2_rhs).unwrap(), 19.0);
    assert_reduction_parity(&su2_lhs, &su2_rhs);

    let su2_rank5_lhs = TensorMap::from_block_fn(
        &runtime,
        [&su2_leg, &su2_leg, &su2_leg],
        [&su2_leg, &su2_leg],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    let su2_rank5_rhs = TensorMap::from_block_fn(
        &runtime,
        [&su2_leg, &su2_leg, &su2_leg],
        [&su2_leg, &su2_leg],
        |_, indices| indices.iter().sum::<usize>() as f64 + 3.0,
    )
    .unwrap();
    assert_reduction_parity(&su2_rank5_lhs, &su2_rank5_rhs);

    let fz2_provider = Arc::new(FermionParityFusionRule);
    let fz2_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&fz2_provider),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let fz2_lhs = TensorMap::from_block_fn(&runtime, [&fz2_leg], [&fz2_leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let fz2_rhs = TensorMap::from_block_fn(&runtime, [&fz2_leg], [&fz2_leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 3.0
    })
    .unwrap();
    assert_reduction_parity(&fz2_lhs, &fz2_rhs);

    let product_provider = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&product_provider),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product_lhs =
        TensorMap::from_block_fn(&runtime, [&product_leg], [&product_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 2.0
        })
        .unwrap();
    let product_rhs =
        TensorMap::from_block_fn(&runtime, [&product_leg], [&product_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 5.0
        })
        .unwrap();
    assert_reduction_parity(&product_lhs, &product_rhs);

    let callback_count = Arc::new(AtomicUsize::new(0));
    let probe_provider = Arc::new(ReentrantDimensionRule {
        runtime: runtime.clone(),
        calls: Arc::clone(&callback_count),
    });
    let probe_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&probe_provider), [(ProbeSector, 1)]).unwrap();
    let probe_lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&probe_leg], [&probe_leg], |_, _| 2.0).unwrap();
    let probe_rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&probe_leg], [&probe_leg], |_, _| 3.0).unwrap();
    let calls_before_reduction = callback_count.load(Ordering::SeqCst);
    assert_eq!(
        probe_lhs
            .to_cuda()
            .unwrap()
            .inner(&probe_rhs.to_cuda().unwrap())
            .unwrap(),
        6.0
    );
    assert!(callback_count.load(Ordering::SeqCst) > calls_before_reduction);

    // Same canary for a device structural transform (#1322). This provider's
    // `dim_scalar` re-enters `lease_cuda`, so any provider callback made while
    // the device lease is held deadlocks on a non-re-entrant mutex. Today the
    // transform's compile needs F and R symbols but no dimension for this
    // one-sector rule, so the counter is not asserted to move — what is
    // guarded is that a future edit which does need a dimension cannot make
    // that call under the lease.
    let probe_device = probe_lhs.to_cuda().unwrap();
    assert_eq!(
        probe_device
            .permute(&[1], &[0])
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        probe_lhs.permute(&[1], &[0]).unwrap().data()
    );

    let other_runtime = Runtime::builder().cuda(0).build().unwrap();
    let foreign = TensorMap::from_block_fn(&other_runtime, [&u1_leg], [&u1_leg], |_, _| 1.0)
        .unwrap()
        .to_cuda()
        .unwrap();
    assert_eq!(
        u1_device.inner(&foreign).unwrap_err(),
        tenet::typed::Error::RuntimeMismatch
    );

    let wider = GradedSpace::try_new_with_arc(
        Arc::clone(&u1_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let mismatched = TensorMap::from_block_fn(&runtime, [&wider], [&wider], |_, _| 1.0)
        .unwrap()
        .to_cuda()
        .unwrap();
    assert!(matches!(
        u1_device.inner(&mismatched),
        Err(tenet::typed::Error::InvalidArgument(_))
    ));
    assert_eq!(u1_device.to_host().unwrap().data(), u1_lhs.data());

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge0 = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(0), 1)]).unwrap();
    let charge1 = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(1), 1)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&charge0], [&charge1], |_, _| 1.0).unwrap();
    assert!(empty.data().is_empty());
    let empty_device = empty.to_cuda().unwrap();
    assert_eq!(empty_device.norm().unwrap(), 0.0);
    assert_eq!(empty_device.inner(&empty_device).unwrap(), 0.0);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_qr_compact_streams_multiplicity_free_f64_factors() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let u1 = Arc::new(U1FusionRule);
    let tall = GradedSpace::try_new_with_arc(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let wide = GradedSpace::try_new_with_arc(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let mixed = TensorMap::from_block_fn(&runtime, [&tall], [&wide], |_, indices| {
        if indices[0] == indices[1] {
            6.0 + indices[0] as f64
        } else {
            (1 + indices[0] + 2 * indices[1]) as f64
        }
    })
    .unwrap();
    assert_typed_cuda_qr_matches_host(&mixed);

    let charge_zero_only =
        GradedSpace::try_new_with_arc(Arc::clone(&u1), [(U1Irrep::new(0), 2)]).unwrap();
    let unmatched =
        TensorMap::from_block_fn(&runtime, [&tall], [&charge_zero_only], |_, indices| {
            if indices[0] == indices[1] {
                4.0 + indices[0] as f64
            } else {
                1.0
            }
        })
        .unwrap();
    assert_typed_cuda_qr_matches_host(&unmatched);

    let square = TensorMap::from_block_fn(&runtime, [&wide], [&wide], |_, indices| {
        if indices[0] == indices[1] {
            8.0 + indices[0] as f64
        } else {
            (1 + indices[0] + indices[1]) as f64
        }
    })
    .unwrap();
    assert_typed_cuda_qr_matches_host(&square);

    let su2 = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&su2),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_tensor = TensorMap::from_block_fn(&runtime, [&su2_leg], [&su2_leg], |_, indices| {
        if indices[0] == indices[1] {
            7.0 + indices[0] as f64
        } else {
            1.0
        }
    })
    .unwrap();
    assert_typed_cuda_qr_matches_host(&su2_tensor);

    let product = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&product),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product_tensor =
        TensorMap::from_block_fn(&runtime, [&product_leg], [&product_leg], |_, indices| {
            if indices[0] == indices[1] {
                5.0 + indices[0] as f64
            } else {
                1.0
            }
        })
        .unwrap();
    assert_typed_cuda_qr_matches_host(&product_tensor);

    let multi_tree =
        TensorMap::from_block_fn(&runtime, [&su2_leg, &su2_leg], [&su2_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let (expected_multi_left, expected_multi_right) = multi_tree.qr_compact().unwrap();
    let multi_tree_device = multi_tree.to_cuda().unwrap();
    let (left, right) = multi_tree_device.qr_compact().unwrap();
    let left = left.to_host().unwrap();
    let right = right.to_host().unwrap();
    assert_eq!(
        structural_snapshot(&left),
        structural_snapshot(&expected_multi_left)
    );
    assert_eq!(
        structural_snapshot(&right),
        structural_snapshot(&expected_multi_right)
    );
    let rebuilt = left.compose(&right).unwrap();
    assert_close(rebuilt.data(), multi_tree.data(), 1e-10);

    let zero = TensorMap::from_block_fn(&runtime, [&wide], [&wide], |_, _| 0.0).unwrap();
    let zero_device = zero.to_cuda().unwrap();
    let (zero_left, zero_right) = zero_device.qr_compact().unwrap();
    let zero_left = zero_left.to_host().unwrap();
    let zero_right = zero_right.to_host().unwrap();
    let zero_rebuilt = zero_left.compose(&zero_right).unwrap();
    assert_close(zero_rebuilt.data(), zero.data(), 1e-10);
    assert_finite_r_diagonal_nonnegative(&zero_right);

    let rank_deficient_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&u1), [(U1Irrep::new(0), 3)]).unwrap();
    let rank_deficient = TensorMap::from_block_fn(
        &runtime,
        [&rank_deficient_leg],
        [&rank_deficient_leg],
        |_, indices| (indices[0] + 1) as f64 * (indices[1] + 1) as f64,
    )
    .unwrap();
    let (rank_left, rank_right) = rank_deficient.to_cuda().unwrap().qr_compact().unwrap();
    let rank_left = rank_left.to_host().unwrap();
    let rank_right = rank_right.to_host().unwrap();
    assert!(rank_left.is_isometric(1e-10).unwrap());
    assert_close(
        rank_left.compose(&rank_right).unwrap().data(),
        rank_deficient.data(),
        1e-10,
    );
    assert_finite_r_diagonal_nonnegative(&rank_right);

    let tiny_negative = TensorMap::from_block_fn(
        &runtime,
        [&charge_zero_only],
        [&charge_zero_only],
        |_, indices| {
            if indices[0] == indices[1] {
                -1.0e-300 * (indices[0] + 1) as f64
            } else {
                0.0
            }
        },
    )
    .unwrap();
    let (tiny_left, tiny_right) = tiny_negative.to_cuda().unwrap().qr_compact().unwrap();
    let tiny_left = tiny_left.to_host().unwrap();
    let tiny_right = tiny_right.to_host().unwrap();
    assert_close(
        tiny_left.compose(&tiny_right).unwrap().data(),
        tiny_negative.data(),
        1e-10,
    );
    assert_finite_r_diagonal_nonnegative(&tiny_right);

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge0 = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(0), 2)]).unwrap();
    let charge1 = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(1), 3)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&charge0], [&charge1], |_, _| 1.0).unwrap();
    assert!(empty.data().is_empty());
    assert_typed_cuda_qr_matches_host(&empty);

    let device = mixed.to_cuda().unwrap();
    let lazy = device.adjoint().unwrap();
    assert!(matches!(
        lazy.qr_compact(),
        Err(tenet::typed::Error::UnsupportedOnDevice(_))
    ));
    let expected = device.qr_compact().unwrap();
    let expected_left = expected.0.to_host().unwrap();
    let expected_right = expected.1.to_host().unwrap();
    for _ in 0..3 {
        let actual = device.qr_compact().unwrap();
        assert_close(
            actual.0.to_host().unwrap().data(),
            expected_left.data(),
            1e-10,
        );
        assert_close(
            actual.1.to_host().unwrap().data(),
            expected_right.data(),
            1e-10,
        );
    }
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| device.qr_compact().unwrap()))
            .collect();
        for worker in workers {
            let actual = worker.join().unwrap();
            assert_close(
                actual.0.to_host().unwrap().data(),
                expected_left.data(),
                1e-10,
            );
            assert_close(
                actual.1.to_host().unwrap().data(),
                expected_right.data(),
                1e-10,
            );
        }
    });
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_compact_streams_dense_multiplicity_free_f64_factors() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = Arc::new(U1FusionRule);
    let tall = GradedSpace::try_new_with_arc(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let wide = GradedSpace::try_new_with_arc(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let rectangular = TensorMap::from_block_fn(&runtime, [&tall], [&wide], |_, indices| {
        if indices[0] == indices[1] {
            6.0 + indices[0] as f64
        } else {
            (1 + indices[0] + 2 * indices[1]) as f64
        }
    })
    .unwrap();
    assert_typed_cuda_svd_matches_host(&rectangular);

    let rank_deficient = TensorMap::from_block_fn(&runtime, [&tall], [&tall], |_, indices| {
        (indices[0] + 1) as f64 * (indices[1] + 1) as f64
    })
    .unwrap();
    assert_typed_cuda_svd_matches_host(&rank_deficient);

    let su2 = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&su2),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let multi_tree =
        TensorMap::from_block_fn(&runtime, [&su2_leg, &su2_leg], [&su2_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let multi_tree_snapshot = structural_snapshot(&multi_tree);
    assert!(multi_tree_snapshot
        .blocks
        .iter()
        .enumerate()
        .any(|(index, left)| {
            multi_tree_snapshot.blocks[index + 1..].iter().any(|right| {
                left.fusion_trees.coupled() == right.fusion_trees.coupled()
                    && left.fusion_trees != right.fusion_trees
            })
        }));
    assert_typed_cuda_svd_matches_host(&multi_tree);

    let all_zero: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&wide], [&wide]).unwrap();
    assert!(all_zero.data().iter().all(|value| *value == 0.0));
    assert_typed_cuda_svd_matches_host(&all_zero);

    let fermion = Arc::new(FermionParityFusionRule);
    let fermion_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&fermion),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let fermion_tensor =
        TensorMap::from_block_fn(&runtime, [&fermion_leg], [&fermion_leg], |_, indices| {
            (1 + indices[0] + 3 * indices[1]) as f64
        })
        .unwrap();
    assert_typed_cuda_svd_matches_host(&fermion_tensor);

    let product = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&product),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product_tensor =
        TensorMap::from_block_fn(&runtime, [&product_leg], [&product_leg], |_, indices| {
            (2 + indices[0] + indices[1]) as f64
        })
        .unwrap();
    assert_typed_cuda_svd_matches_host(&product_tensor);

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge0 = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(0), 2)]).unwrap();
    let charge1 = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(1), 3)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&charge0], [&charge1], |_, _| 1.0).unwrap();
    assert_typed_cuda_svd_matches_host(&empty);

    let device = rectangular.to_cuda().unwrap();
    let source_bits: Vec<_> = device
        .to_host()
        .unwrap()
        .data()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert!(matches!(
        device.adjoint().unwrap().svd_compact(),
        Err(tenet::typed::Error::UnsupportedOnDevice(_))
    ));
    let expected = rectangular.svd_compact().unwrap();
    for _ in 0..3 {
        assert_cuda_svd_result(&rectangular, &expected, device.svd_compact().unwrap());
    }
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..2)
            .map(|_| scope.spawn(|| device.svd_compact().unwrap()))
            .collect();
        for worker in workers {
            assert_cuda_svd_result(&rectangular, &expected, worker.join().unwrap());
        }
    });
    assert_eq!(
        device
            .to_host()
            .unwrap()
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        source_bits
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_compact_handles_large_wide_sector() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let rows =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(0), 8)]).unwrap();
    let cols =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(0), 1025)]).unwrap();
    let source = TensorMap::from_block_fn(&runtime, [&rows], [&cols], |_, indices| {
        let row = indices[0];
        let col = indices[1];
        ((row * 17 + col * 13 + 3) % 31) as f64 / 31.0 - 0.5 + if row == col { 2.0 } else { 0.0 }
    })
    .unwrap();

    assert_typed_cuda_svd_matches_host(&source);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_trunc_composition_handles_large_wide_sector() {
    // What: the 8 x 1025 cuSOLVER path composes with a kept prefix on the Host.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let rows =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(0), 8)]).unwrap();
    let cols =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(0), 1025)]).unwrap();
    let source = TensorMap::from_block_fn(&runtime, [&rows], [&cols], |_, indices| {
        let row = indices[0];
        let col = indices[1];
        ((row * 17 + col * 13 + 3) % 31) as f64 / 31.0 - 0.5 + if row == col { 2.0 } else { 0.0 }
    })
    .unwrap();

    assert_typed_cuda_svd_trunc_composition_matches_host(&source, &Truncation::rank(4));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_trunc_composition_matches_host_policies_structure_and_ownership() {
    // What: all truncation policies and supported provider families match the
    // Host semantic oracle without comparing backend-dependent U/Vh gauges.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let rows = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let cols = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(2), 1)],
    )
    .unwrap();
    let mixed = TensorMap::from_block_fn(&runtime, [&rows], [&cols], |_, indices| {
        if indices[0] == indices[1] {
            5.0 + indices[0] as f64
        } else {
            (1 + indices[0] + 2 * indices[1]) as f64
        }
    })
    .unwrap();
    let same_identity_space = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let policies = [
        Truncation::Full,
        Truncation::rank(0),
        Truncation::rank(2),
        Truncation::rank(usize::MAX),
        Truncation::absolute_cutoff(0.25).unwrap(),
        Truncation::relative_inf_cutoff(0.2).unwrap(),
        Truncation::relative_error(0.1).unwrap(),
        Truncation::space(same_identity_space.truncspace()),
        Truncation::rank(3).and(Truncation::absolute_cutoff(0.1).unwrap()),
    ];
    for policy in &policies {
        assert_typed_cuda_svd_trunc_composition_matches_host(&mixed, policy);
    }

    let dimension_calls = Arc::new(AtomicUsize::new(0));
    let reentrant = Arc::new(ReentrantDimensionRule {
        runtime: runtime.clone(),
        calls: Arc::clone(&dimension_calls),
    });
    let reentrant_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&reentrant), [(ProbeSector, 2)]).unwrap();
    let reentrant_source = TensorMap::from_block_fn(
        &runtime,
        [&reentrant_leg],
        [&reentrant_leg],
        |_, indices| (1 + indices[0] + indices[1]) as f64,
    )
    .unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(&reentrant_source, &Truncation::rank(1));
    assert!(dimension_calls.load(Ordering::SeqCst) > 0);

    // `[3, 2] <- [3, 2]` factors put a 2x2 block at element offset 9, the
    // unaligned destination that used to fault the device factor copy
    // (#1320); these two fixtures cover that layout through the composition.
    let rank_deficient = TensorMap::from_block_fn(&runtime, [&rows], [&rows], |_, indices| {
        (indices[0] + 1) as f64 * (indices[1] + 1) as f64
    })
    .unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(&rank_deficient, &Truncation::rank(1));
    let all_zero: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&rows], [&rows]).unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(
        &all_zero,
        &Truncation::relative_error(0.0).unwrap(),
    );

    let no_intersection_rows =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(4), 2)]).unwrap();
    let no_intersection: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&no_intersection_rows], [&cols], |_, _| 1.0).unwrap();
    assert!(no_intersection.data().is_empty());
    assert_typed_cuda_svd_trunc_composition_matches_host(&no_intersection, &Truncation::Full);

    let su2 = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&su2),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let multi_tree =
        TensorMap::from_block_fn(&runtime, [&su2_leg, &su2_leg], [&su2_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(&multi_tree, &Truncation::rank(3));

    let fermion = Arc::new(FermionParityFusionRule);
    let fermion_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&fermion),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let fermion_tensor =
        TensorMap::from_block_fn(&runtime, [&fermion_leg], [&fermion_leg], |_, indices| {
            (1 + indices[0] + 3 * indices[1]) as f64
        })
        .unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(&fermion_tensor, &Truncation::rank(2));

    let product = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&product),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product_tensor =
        TensorMap::from_block_fn(&runtime, [&product_leg], [&product_leg], |_, indices| {
            (2 + indices[0] + indices[1]) as f64
        })
        .unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(&product_tensor, &Truncation::rank(2));

    // Repetition and concurrency now bear on the device half of the
    // composition, which is `svd_compact`.
    let device = mixed.to_cuda().unwrap();
    let expected = device.svd_compact().unwrap().1.to_host().unwrap();
    let expected_spectrum = expected.diagview().unwrap();
    for _ in 0..2 {
        let actual = device.svd_compact().unwrap().1.to_host().unwrap();
        assert_eq!(actual.diagview().unwrap(), expected_spectrum);
    }
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..2)
            .map(|_| scope.spawn(|| device.svd_compact().unwrap().1.to_host().unwrap()))
            .collect();
        for worker in workers {
            assert_eq!(
                worker.join().unwrap().diagview().unwrap(),
                expected_spectrum
            );
        }
    });

    // The truncated entry points are an explicit capability boundary that
    // precedes every device action, including on a dense device receiver.
    let before = tenet::dense::cuda_transfer_stats();
    for (operation, error) in [
        ("svd_trunc", device.svd_trunc(&Truncation::rank(2)).err()),
        ("eigh_trunc", device.eigh_trunc(&Truncation::rank(2)).err()),
    ] {
        assert!(
            matches!(error, Some(tenet::prelude::Error::UnsupportedOnDevice(ref message))
                if message.contains(operation) && message.contains("find_truncated")),
            "{operation}: {error:?}"
        );
    }
    // No lease, no plan, no allocation, no transfer: every device counter is
    // exactly where it was before the two rejected calls. `cuda_transfer_stats`
    // is process-wide, so this equality only holds because the device suite
    // runs with `--test-threads=1`; the operand-independent evidence is the
    // non-device unit test in `typed.rs`.
    assert_eq!(tenet::dense::cuda_transfer_stats(), before);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_arithmetic_matches_host_lazy_ownership_and_concurrency() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let rhs_provider = Arc::new(U1FusionRule);
    let rhs_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&rhs_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&rhs_leg], [&rhs_leg], |_, indices| {
            2.0 * indices.iter().sum::<usize>() as f64 - 1.0
        })
        .unwrap();
    let lhs_data = lhs.data().to_vec();
    let rhs_data = rhs.data().to_vec();
    let lhs_provider = lhs.provider() as *const U1FusionRule;
    let rhs_provider = rhs.provider() as *const U1FusionRule;
    let runtime_id = runtime.identity();
    let lhs_device = lhs.to_cuda().unwrap();
    let rhs_device = rhs.to_cuda().unwrap();

    for factor in [2.5, -1.75] {
        let expected = lhs.scale(factor);
        let actual = lhs_device.scale(factor).unwrap().to_host().unwrap();
        assert_eq!(actual.data(), expected.data());
        assert!(std::ptr::eq(actual.provider(), lhs_provider));
        assert!(runtime_id.matches(actual.runtime()));
        assert_eq!(structural_snapshot(&actual), structural_snapshot(&lhs));
    }

    let (alpha, beta) = (2.25, -3.5);
    let expected_add = lhs.add(&rhs, alpha, beta).unwrap();
    let actual_add = lhs_device
        .add(&rhs_device, alpha, beta)
        .unwrap()
        .to_host()
        .unwrap();
    assert_eq!(actual_add.data(), expected_add.data());
    assert!(std::ptr::eq(actual_add.provider(), lhs_provider));
    assert!(!std::ptr::eq(actual_add.provider(), rhs_provider));
    assert!(runtime_id.matches(actual_add.runtime()));
    assert_eq!(structural_snapshot(&actual_add), structural_snapshot(&lhs));
    assert_eq!(lhs_device.to_host().unwrap().data(), lhs_data);
    assert_eq!(rhs_device.to_host().unwrap().data(), rhs_data);
    for _ in 0..3 {
        assert_eq!(
            lhs_device
                .add(&rhs_device, alpha, beta)
                .unwrap()
                .to_host()
                .unwrap()
                .data(),
            expected_add.data()
        );
    }

    let nonfinite_values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0];
    let nonfinite_index = std::cell::Cell::new(0usize);
    let nonfinite = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| {
        let index = nonfinite_index.get();
        nonfinite_index.set(index + 1);
        nonfinite_values[index % nonfinite_values.len()]
    })
    .unwrap();
    let nonfinite_bits: Vec<_> = nonfinite
        .data()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let nonfinite_device = nonfinite.to_cuda().unwrap();
    let exact_zero = nonfinite_device.zeros_like().unwrap().to_host().unwrap();
    assert!(exact_zero.data().iter().all(|value| value.to_bits() == 0));

    let finite_values = [0.0, -0.0, 2.0, -3.0, 1.0];
    let finite_index = std::cell::Cell::new(0usize);
    let finite = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| {
        let index = finite_index.get();
        finite_index.set(index + 1);
        finite_values[index % finite_values.len()]
    })
    .unwrap();
    let finite_device = finite.to_cuda().unwrap();
    let assert_nonfinite_numeric_parity = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan(), "expected NaN, got {actual:?}");
            } else {
                assert_eq!(actual, expected);
            }
        }
    };

    for zero_factor in [0.0, -0.0] {
        let expected_scale_zero = nonfinite.scale(zero_factor);
        let actual_scale_zero = nonfinite_device
            .scale(zero_factor)
            .unwrap()
            .to_host()
            .unwrap();
        assert_nonfinite_numeric_parity(actual_scale_zero.data(), expected_scale_zero.data());

        let expected_zero_alpha = nonfinite.add(&finite, zero_factor, 1.0).unwrap();
        let actual_zero_alpha = nonfinite_device
            .add(&finite_device, zero_factor, 1.0)
            .unwrap()
            .to_host()
            .unwrap();
        assert_nonfinite_numeric_parity(actual_zero_alpha.data(), expected_zero_alpha.data());

        let expected_zero_beta = finite.add(&nonfinite, 1.0, zero_factor).unwrap();
        let actual_zero_beta = finite_device
            .add(&nonfinite_device, 1.0, zero_factor)
            .unwrap()
            .to_host()
            .unwrap();
        assert_nonfinite_numeric_parity(actual_zero_beta.data(), expected_zero_beta.data());
    }
    assert_eq!(
        nonfinite_device
            .to_host()
            .unwrap()
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        nonfinite_bits
    );

    let lhs_lazy = lhs_device.adjoint().unwrap();
    let rhs_lazy = rhs_device.adjoint().unwrap();
    let lazy_scale = lhs_lazy.scale(alpha).unwrap().to_host().unwrap();
    let lazy_add = lhs_lazy
        .add(&rhs_lazy, alpha, beta)
        .unwrap()
        .to_host()
        .unwrap();
    let lazy_zero = lhs_lazy.zeros_like().unwrap().to_host().unwrap();
    assert_eq!(
        lazy_scale.data(),
        lhs.adjoint().unwrap().scale(alpha).data()
    );
    assert_eq!(
        lazy_add.data(),
        lhs.adjoint()
            .unwrap()
            .add(&rhs.adjoint().unwrap(), alpha, beta)
            .unwrap()
            .data()
    );
    assert!(std::ptr::eq(lazy_add.provider(), lhs_provider));
    assert!(!std::ptr::eq(lazy_add.provider(), rhs_provider));
    assert!(lazy_zero.data().iter().all(|value| value.to_bits() == 0));
    assert!(matches!(
        lhs_lazy.add(&rhs_device, alpha, beta),
        Err(tenet::typed::Error::UnsupportedOnDevice(_))
    ));

    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    lhs_device
                        .add(&rhs_device, alpha, beta)
                        .unwrap()
                        .to_host()
                        .unwrap()
                        .data()
                        .to_vec()
                })
            })
            .collect();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), expected_add.data());
        }
    });

    let su2_provider = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&su2_provider),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let su2: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&su2_leg], [&su2_leg], |trees, _| {
            if trees.coupled() == &SU2Irrep::from_twice_spin(0) {
                1.0
            } else {
                2.0
            }
        })
        .unwrap();
    let normalized = su2.to_cuda().unwrap().normalize().unwrap();
    assert!((normalized.norm().unwrap() - 1.0).abs() < 1e-12);
    let expected_normalized = su2.normalize().unwrap();
    let normalized_host = normalized.to_host().unwrap();
    for (&actual, &expected) in normalized_host
        .data()
        .iter()
        .zip(expected_normalized.data())
    {
        assert!((actual - expected).abs() < 1e-12);
    }

    let zero = lhs.zeros_like().to_cuda().unwrap().normalize().unwrap();
    assert!(zero
        .to_host()
        .unwrap()
        .data()
        .iter()
        .all(|value| !value.is_finite()));

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge0 = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(0), 1)]).unwrap();
    let charge1 = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(1), 1)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&charge0], [&charge1], |_, _| f64::NAN).unwrap();
    let empty_device = empty.to_cuda().unwrap();
    assert!(empty_device
        .scale(2.0)
        .unwrap()
        .to_host()
        .unwrap()
        .data()
        .is_empty());
    assert!(empty_device
        .add(&empty_device, alpha, beta)
        .unwrap()
        .to_host()
        .unwrap()
        .data()
        .is_empty());
    assert!(empty_device
        .zeros_like()
        .unwrap()
        .to_host()
        .unwrap()
        .data()
        .is_empty());
    assert!(empty_device
        .normalize()
        .unwrap()
        .to_host()
        .unwrap()
        .data()
        .is_empty());

    let other_runtime = Runtime::builder().cuda(0).build().unwrap();
    let other_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(0), 1)]).unwrap();
    let foreign = TensorMap::from_block_fn(&other_runtime, [&other_leg], [&other_leg], |_, _| 1.0)
        .unwrap()
        .to_cuda()
        .unwrap();
    assert_eq!(
        lhs_device.add(&foreign, alpha, beta).unwrap_err(),
        tenet::typed::Error::RuntimeMismatch
    );
    let mismatched = TensorMap::from_block_fn(&runtime, [&other_leg], [&other_leg], |_, _| 1.0)
        .unwrap()
        .to_cuda()
        .unwrap();
    assert!(matches!(
        lhs_device.add(&mismatched, alpha, beta),
        Err(tenet::typed::Error::InvalidArgument(_))
    ));
    assert_eq!(lhs_device.to_host().unwrap().data(), lhs_data);
}

#[test]
#[ignore]
fn typed_cuda_fermionic_contract_is_minus_six_and_compose_stays_plus_six() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let lhs_codomain =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z2Irrep::ODD, 1)]).unwrap();
    let lhs_domain = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z2Irrep::ODD, 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let rhs_codomain = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z2Irrep::ODD, 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let rhs_domain =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z2Irrep::ODD, 1)]).unwrap();
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
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
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
    // General axes are admitted since G2c-1a (#1345): a mismatched pairing is
    // the Host's own error, and a permuted output is the Host's result.
    assert_eq!(
        device
            .contract(&device, &[0], &[0], &[0, 1])
            .unwrap_err()
            .to_string(),
        host.contract(&host, &[0], &[0], &[0, 1])
            .unwrap_err()
            .to_string()
    );
    let close = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-12 * (1.0 + expected.abs()));
        }
    };
    close(
        device
            .contract(&device, &[1], &[0], &[1, 0])
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        host.contract(&host, &[1], &[0], &[1, 0]).unwrap().data(),
    );
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
    let lazy_general = lazy.contract(&device, &[1], &[0], &[1, 0]).unwrap();
    let expected_lazy_general = lazy_host.contract(&host, &[1], &[0], &[1, 0]).unwrap();
    close(
        lazy_general.to_host().unwrap().data(),
        expected_lazy_general.data(),
    );
    assert_eq!(device.to_host().unwrap().data(), expected);

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let zn4 = Arc::new(ZNFusionRule::new(4).unwrap());
    let zn3_leg = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(0), 1)]).unwrap();
    let zn4_leg = GradedSpace::try_new_with_arc(Arc::clone(&zn4), [(zn4.irrep(0), 1)]).unwrap();
    let zn3_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&zn3_leg], [&zn3_leg], |_, _| 1.0).unwrap();
    let zn4_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&zn4_leg], [&zn4_leg], |_, _| 1.0).unwrap();
    let zn3_device = zn3_tensor.to_cuda().unwrap();
    let zn4_device = zn4_tensor.to_cuda().unwrap();
    assert!(zn3_device.compose(&zn4_device).is_err());
    assert_eq!(zn3_device.to_host().unwrap().data(), [1.0]);
    assert_eq!(zn4_device.to_host().unwrap().data(), [1.0]);

    let left_open = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(0), 1)]).unwrap();
    let seam = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(1), 1)]).unwrap();
    let right_open = GradedSpace::try_new_with_arc(Arc::clone(&zn3), [(zn3.irrep(2), 1)]).unwrap();
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

// ---------------------------------------------------------------------------
// Complex64 device payload (#1268, G1a). Every fixture below has nonzero
// imaginary parts in every stored entry: a real value cast to `Complex64`
// cannot distinguish a conjugation defect from a transpose.
// ---------------------------------------------------------------------------

/// Genuinely complex fill: `im` is never zero for any admitted index tuple.
fn complex_entry(indices: &[usize], seed: f64) -> Complex64 {
    let ramp = indices.iter().map(|&index| index as f64).sum::<f64>();
    Complex64::new(ramp + seed, -(ramp + seed + 0.75))
}

fn assert_close_c64(actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).norm() <= tolerance * (1.0 + expected.norm()),
            "actual {actual:?}, expected {expected:?}"
        );
    }
}

fn assert_c64_roundtrip<R>(source: TensorMap<R, Complex64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = source.runtime().identity();
    let structure = structural_snapshot(&source);
    let expected = source.data().to_vec();
    assert!(
        expected.iter().all(|value| value.im != 0.0),
        "fixture must have nonzero imaginary parts"
    );

    let device = source.to_cuda().unwrap();
    assert_eq!(device.placement(), tenet::core::Placement::Cuda(0));
    let restored = device.clone().to_host().unwrap();

    assert!(std::ptr::eq(restored.provider(), provider));
    assert!(runtime.matches(restored.runtime()));
    assert_eq!(restored.data(), expected);
    assert_eq!(structural_snapshot(&restored), structure);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_roundtrip_is_bit_exact_across_providers() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();

    let u1 = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2)],
    )
    .unwrap();
    assert_c64_roundtrip(
        TensorMap::<_, Complex64>::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
            complex_entry(indices, 1.0)
        })
        .unwrap(),
    );

    let fz2 = GradedSpace::try_new_with_arc(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)],
    )
    .unwrap();
    assert_c64_roundtrip(
        TensorMap::<_, Complex64>::from_block_fn(&runtime, [&fz2], [&fz2], |_, indices| {
            complex_entry(indices, 2.0)
        })
        .unwrap(),
    );

    let su2 = GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    assert_c64_roundtrip(
        TensorMap::<_, Complex64>::from_block_fn(
            &runtime,
            [&su2, &su2, &su2],
            [&su2, &su2],
            |_, indices| complex_entry(indices, 3.0),
        )
        .unwrap(),
    );

    let product = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 1),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 2),
        ],
    )
    .unwrap();
    assert_c64_roundtrip(
        TensorMap::<_, Complex64>::from_block_fn(&runtime, [&product], [&product], |_, indices| {
            complex_entry(indices, 4.0)
        })
        .unwrap(),
    );

    // The real path is unchanged by the generic payload.
    assert_roundtrip(
        TensorMap::from_block_fn(&runtime, [&u1], [&u1], |_, indices| indices[0] as f64 + 1.0)
            .unwrap(),
    );
}

#[allow(deprecated)]
fn assert_c64_contract_and_compose<R>(lhs: &TensorMap<R, Complex64>, rhs: &TensorMap<R, Complex64>)
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
        assert_close_c64(actual.data(), expected.data(), 1e-12);
        assert_eq!(structural_snapshot(actual), structural_snapshot(expected));
    }

    // Lazy conjugate-transpose operands on either side must route the
    // conjugation through the GEMM flag, never a materialized buffer.
    // `A^H . A` and `B . B^H` are composable for every split.
    let host_left = lhs.adjoint().unwrap().compose(lhs).unwrap();
    let device_left = lhs_device
        .adjoint()
        .unwrap()
        .compose(&lhs_device)
        .unwrap()
        .to_host()
        .unwrap();
    assert_close_c64(device_left.data(), host_left.data(), 1e-12);
    let host_right = rhs.compose(&rhs.adjoint().unwrap()).unwrap();
    let device_right = rhs_device
        .compose(&rhs_device.adjoint().unwrap())
        .unwrap()
        .to_host()
        .unwrap();
    assert_close_c64(device_right.data(), host_right.data(), 1e-12);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_contract_and_compose_match_host() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let u1 = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    assert_c64_contract_and_compose(
        &TensorMap::<_, Complex64>::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
            complex_entry(indices, 1.0)
        })
        .unwrap(),
        &TensorMap::<_, Complex64>::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
            complex_entry(indices, 2.0)
        })
        .unwrap(),
    );

    let su2 = GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    assert_c64_contract_and_compose(
        &TensorMap::<_, Complex64>::from_block_fn(
            &runtime,
            [&su2, &su2, &su2],
            [&su2, &su2],
            |_, indices| complex_entry(indices, 1.0),
        )
        .unwrap(),
        &TensorMap::<_, Complex64>::from_block_fn(
            &runtime,
            [&su2, &su2],
            [&su2, &su2, &su2],
            |_, indices| complex_entry(indices, 3.0),
        )
        .unwrap(),
    );

    // Fermionic provider: contract carries the twist, compose does not, so
    // the two results stay distinct for complex payloads too.
    let fz2 = GradedSpace::try_new_with_arc(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let fz2_dual = GradedSpace::try_new_with_arc(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    assert_c64_contract_and_compose(
        &TensorMap::<_, Complex64>::from_block_fn(
            &runtime,
            [&fz2],
            [&fz2_dual, &fz2_dual],
            |_, indices| complex_entry(indices, 1.0),
        )
        .unwrap(),
        &TensorMap::<_, Complex64>::from_block_fn(
            &runtime,
            [&fz2_dual, &fz2_dual],
            [&fz2],
            |_, indices| complex_entry(indices, 2.0),
        )
        .unwrap(),
    );

    // General axes are admitted since G2c-1a (#1345): the formerly rejected
    // scopes now agree with the Host.
    let host_lhs =
        TensorMap::<_, Complex64>::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
            complex_entry(indices, 5.0)
        })
        .unwrap();
    let host_rhs =
        TensorMap::<_, Complex64>::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
            complex_entry(indices, 6.0)
        })
        .unwrap();
    let lhs = host_lhs.to_cuda().unwrap();
    let rhs = host_rhs.to_cuda().unwrap();
    for (lhs_axes, rhs_axes, output) in [(&[0], &[1], &[0, 1]), (&[1], &[0], &[1, 0])] {
        let expected = host_lhs
            .contract(&host_rhs, lhs_axes, rhs_axes, output)
            .unwrap();
        let actual = lhs
            .contract(&rhs, lhs_axes, rhs_axes, output)
            .unwrap()
            .to_host()
            .unwrap();
        assert_eq!(actual.data().len(), expected.data().len());
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - expected).norm() <= 1e-12 * (1.0 + expected.norm()));
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_inner_is_conjugate_linear_in_the_first_argument() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let i = Complex64::new(0.0, 1.0);

    for (sectors, seed) in [(0_usize, 1.0_f64), (1, 7.0)] {
        let su2 = GradedSpace::try_new_with_arc(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 1 + sectors),
                (SU2Irrep::from_twice_spin(1), 2),
            ],
        )
        .unwrap();
        let a = TensorMap::<_, Complex64>::from_block_fn(&runtime, [&su2], [&su2], |_, indices| {
            complex_entry(indices, seed)
        })
        .unwrap();
        let b = TensorMap::<_, Complex64>::from_block_fn(&runtime, [&su2], [&su2], |_, indices| {
            complex_entry(indices, seed + 2.0)
        })
        .unwrap();
        let host_inner = a.inner(&b).unwrap();
        let host_norm = a.norm().unwrap();
        assert!(
            host_inner.im != 0.0,
            "the fixture must exercise a complex inner product"
        );

        let a_device = a.to_cuda().unwrap();
        let b_device = b.to_cuda().unwrap();
        let inner = a_device.inner(&b_device).unwrap();
        let tolerance = 1e-12 * (1.0 + host_inner.norm());
        assert!((inner - host_inner).norm() <= tolerance);
        assert!((b_device.inner(&a_device).unwrap() - inner.conj()).norm() <= tolerance);
        let scaled = a_device.scale(i).unwrap();
        assert_close_c64(scaled.to_host().unwrap().data(), a.scale(i).data(), 1e-12);
        assert!((scaled.inner(&b_device).unwrap() - (-i) * inner).norm() <= tolerance);
        assert!(
            (a_device.inner(&b_device.scale(i).unwrap()).unwrap() - i * inner).norm() <= tolerance
        );

        let norm = a_device.norm().unwrap();
        assert!((norm - host_norm).abs() <= 1e-12 * (1.0 + host_norm));
        let self_inner = a_device.inner(&a_device).unwrap();
        assert!(self_inner.im.abs() <= 1e-12 * (1.0 + self_inner.norm()));
        assert!((self_inner.re - norm * norm).abs() <= 1e-12 * (1.0 + self_inner.norm()));

        #[allow(deprecated)]
        {
            assert_eq!(a_device.dot(&b_device).unwrap(), inner);
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_scale_and_add_match_host_including_the_lazy_fold() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let alpha = Complex64::new(2.0, -3.0);
    let beta = Complex64::new(-0.5, 1.25);

    let a = TensorMap::<_, Complex64>::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
        complex_entry(indices, 1.0)
    })
    .unwrap();
    let b = TensorMap::<_, Complex64>::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
        complex_entry(indices, 4.0)
    })
    .unwrap();
    let a_device = a.to_cuda().unwrap();
    let b_device = b.to_cuda().unwrap();

    assert_close_c64(
        a_device.scale(alpha).unwrap().to_host().unwrap().data(),
        a.scale(alpha).data(),
        1e-12,
    );
    assert_close_c64(
        a_device
            .add(&b_device, alpha, beta)
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        a.add(&b, alpha, beta).unwrap().data(),
        1e-12,
    );
    assert_close_c64(
        a_device.zeros_like().unwrap().to_host().unwrap().data(),
        &vec![Complex64::new(0.0, 0.0); a.data().len()],
        0.0,
    );
    assert_close_c64(
        a_device.normalize().unwrap().to_host().unwrap().data(),
        a.normalize().unwrap().data(),
        1e-12,
    );

    // `(alpha A^H + beta B^H) == (conj(alpha) A + conj(beta) B)^H`.
    let lazy_a = a_device.adjoint().unwrap();
    let lazy_b = b_device.adjoint().unwrap();
    let host_fold = a
        .adjoint()
        .unwrap()
        .add(&b.adjoint().unwrap(), alpha, beta)
        .unwrap();
    assert_close_c64(
        lazy_a
            .add(&lazy_b, alpha, beta)
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        host_fold.data(),
        1e-12,
    );
    for factor in [Complex64::new(0.0, 1.0), Complex64::new(1.0, 2.0), alpha] {
        assert_close_c64(
            lazy_a.scale(factor).unwrap().to_host().unwrap().data(),
            a.adjoint().unwrap().scale(factor).data(),
            1e-12,
        );
    }
    assert!(matches!(
        lazy_a.add(&b_device, alpha, beta),
        Err(tenet::typed::Error::UnsupportedOnDevice(_))
    ));
}

/// Independent oracle for the conjugation flag itself.
///
/// Host and device both lower operand conjugation to the same Tenferro
/// `DotGeneralAccumulation` flags, so a Host-vs-device comparison alone cannot
/// catch a shared misreading of those flags. This computes `A^H . B` for a
/// one-sector U(1) fixture by explicit conjugate-transpose loops over the
/// reduced block and compares both the Host and the device result against it.
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_lazy_adjoint_contract_matches_a_hand_expansion() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let a = TensorMap::<_, Complex64>::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
        complex_entry(indices, 1.0) * Complex64::new(1.0, indices[1] as f64 + 1.0)
    })
    .unwrap();
    let b = TensorMap::<_, Complex64>::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
        complex_entry(indices, 3.0) * Complex64::new(indices[0] as f64 + 1.0, -1.0)
    })
    .unwrap();
    assert_eq!(a.block_count(), 1);

    fn reduced_block(
        tensor: &TensorMap<U1FusionRule, Complex64>,
    ) -> impl Fn(usize, usize) -> Complex64 {
        let block = tensor.block(0).unwrap();
        assert_eq!(block.shape(), [2, 2]);
        let offset = block.offset();
        let strides = block.strides().to_vec();
        let data = tensor.data().to_vec();
        move |row: usize, col: usize| data[offset + row * strides[0] + col * strides[1]]
    }
    let a_block = reduced_block(&a);
    let b_block = reduced_block(&b);

    // (A^H B)[i, j] = sum_k conj(A[k, i]) * B[k, j].
    let mut expected = [[Complex64::new(0.0, 0.0); 2]; 2];
    for (i, row) in expected.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            for k in 0..2 {
                *entry += a_block(k, i).conj() * b_block(k, j);
            }
        }
    }

    let a_device = a.to_cuda().unwrap();
    let b_device = b.to_cuda().unwrap();
    // Both device entry points: `compose` (twist-free compiler) and `contract`
    // (which lowers the lazy operand through `TensorContractSpec::
    // new_with_conjugation`). U(1) is bosonic, so both must equal the same
    // conjugate-transposed product.
    let results = [
        (
            "compose",
            a.adjoint().unwrap().compose(&b).unwrap(),
            a_device
                .adjoint()
                .unwrap()
                .compose(&b_device)
                .unwrap()
                .to_host()
                .unwrap(),
        ),
        (
            "contract",
            a.adjoint()
                .unwrap()
                .contract(&b, &[1], &[0], &[0, 1])
                .unwrap(),
            a_device
                .adjoint()
                .unwrap()
                .contract(&b_device, &[1], &[0], &[0, 1])
                .unwrap()
                .to_host()
                .unwrap(),
        ),
    ];
    for (label, host, device) in &results {
        let result_block = host.block(0).unwrap();
        let (offset, strides) = (result_block.offset(), result_block.strides().to_vec());
        for (i, row) in expected.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let index = offset + i * strides[0] + j * strides[1];
                assert!(
                    (host.data()[index] - value).norm() <= 1e-12 * (1.0 + value.norm()),
                    "host {label} {:?} != hand {value:?}",
                    host.data()[index]
                );
                assert!(
                    (device.data()[index] - value).norm() <= 1e-12 * (1.0 + value.norm()),
                    "device {label} {:?} != hand {value:?}",
                    device.data()[index]
                );
            }
        }
    }
    assert!(
        expected.iter().flatten().any(|value| value.im != 0.0),
        "the oracle must distinguish conjugation from transposition"
    );
}

// ---------------------------------------------------------------------------
// Complex64 device factorizations
// ---------------------------------------------------------------------------

/// Deterministic, genuinely complex, generically full-rank fill.
///
/// A ramp fill such as [`complex_entry`] makes every coupled-sector block
/// rank two at most, which neither exercises the positive-diagonal gauge on a
/// full-rank sector nor lets Host factors be compared at all. Every element
/// gets its own hashed value instead.
fn pseudo_random_c64(ordinal: usize) -> Complex64 {
    fn mix(seed: u64) -> f64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        state ^= state >> 29;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 32;
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }
    let ordinal = ordinal as u64 + 1;
    Complex64::new(mix(ordinal), mix(ordinal.wrapping_add(0x0005_DEEC_E66D)))
}

/// Per-element counter fill; see [`pseudo_random_c64`].
fn distinct_c64_fill<S>() -> impl FnMut(&BlockFusionTrees<S>, &[usize]) -> Complex64 {
    let mut ordinal = 0usize;
    move |_, _| {
        ordinal += 1;
        pseudo_random_c64(ordinal)
    }
}

fn assert_device_factor_handles<R, const N: usize>(
    source: &TensorMap<R, Complex64>,
    factors: [&TensorMap<R, Complex64, CudaStorage<Complex64>>; N],
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = source.runtime().identity();
    for factor in factors {
        assert!(std::ptr::eq(factor.provider(), provider));
        assert!(runtime.matches(factor.runtime()));
        assert_eq!(factor.placement(), tenet::core::Placement::Cuda(0));
    }
}

fn assert_c64_svd_matches_host<R>(source: &TensorMap<R, Complex64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let source_data = source.data().to_vec();
    let expected = source.svd_compact().unwrap();
    let device = source.to_cuda().unwrap();
    let (u_device, s_device, vh_device) = device.svd_compact().unwrap();
    assert_device_factor_handles(source, [&u_device, &s_device, &vh_device]);

    let u = u_device.to_host().unwrap();
    let s = s_device.to_host().unwrap();
    let vh = vh_device.to_host().unwrap();
    for (actual, expected) in [(&u, &expected.0), (&s, &expected.1), (&vh, &expected.2)] {
        assert_eq!(structural_snapshot(actual), structural_snapshot(expected));
    }
    // `u`/`vh` keep the raw device gauge, so only gauge-invariant quantities
    // are compared: the spectrum, both orthonormality relations, and `u s vh`.
    assert_close_c64(s.data(), expected.1.data(), 1e-9);
    assert!(u.is_isometric(1e-10).unwrap(), "U^H U = I");
    assert!(
        vh.adjoint().unwrap().is_isometric(1e-10).unwrap(),
        "V^H V = I"
    );
    assert_close_c64(
        u.compose(&s).unwrap().compose(&vh).unwrap().data(),
        &source_data,
        1e-9,
    );
    assert_eq!(device.to_host().unwrap().data(), source_data);
}

fn assert_c64_svd_trunc_composition_matches_host<R>(
    source: &TensorMap<R, Complex64>,
    truncation: &Truncation,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let source_data = source.data().to_vec();
    let expected = source.svd_trunc(truncation).unwrap();
    let device = source.to_cuda().unwrap();

    let (u_device, s_device, vh_device) = device.svd_compact().unwrap();
    assert_device_factor_handles(source, [&u_device, &s_device, &vh_device]);
    let u = u_device.to_host().unwrap();
    let s = s_device.to_host().unwrap();
    let vh = vh_device.to_host().unwrap();

    let found = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), truncation)
        .unwrap();
    let selection = &found.selection;
    let u = u.restrict_leg(u.codomain_rank(), selection).unwrap();
    let s = s.restrict_diagonal(selection).unwrap();
    let vh = vh.restrict_leg(0, selection).unwrap();

    assert_eq!(*selection.subspace(), expected.s.domain()[0]);
    for (actual, expected) in [(&u, &expected.u), (&s, &expected.s), (&vh, &expected.vh)] {
        assert_eq!(structural_snapshot(actual), structural_snapshot(expected));
    }
    let mut kept = s.diagview().unwrap();
    kept.sort_by(|left, right| left.sector.cmp(&right.sector));
    assert_eq!(kept.len(), expected.singular_values.len());
    for (actual, expected) in kept.iter().zip(&expected.singular_values) {
        assert_eq!(actual.sector, expected.sector);
        let values: Vec<f64> = actual.values.iter().map(|value| value.re).collect();
        assert_close(&values, &expected.values, 1e-9);
    }
    assert!((found.error - expected.error).abs() <= 1e-9 * (1.0 + expected.error));

    assert!(u.is_isometric(1e-10).unwrap(), "U^H U = I");
    assert!(
        vh.adjoint().unwrap().is_isometric(1e-10).unwrap(),
        "V^H V = I"
    );
    let actual_rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    let expected_rebuilt = expected
        .u
        .compose(&expected.s)
        .unwrap()
        .compose(&expected.vh)
        .unwrap();
    assert_close_c64(actual_rebuilt.data(), expected_rebuilt.data(), 1e-9);
    assert_eq!(device.to_host().unwrap().data(), source_data);
}

/// `a + a^H` as a Hermitian fixture with nonzero imaginary off-diagonals: a
/// transpose-only admission rule would reject it.
fn hermitian_c64<R>(source: &TensorMap<R, Complex64>) -> TensorMap<R, Complex64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let hermitian = source
        .add(
            &source.adjoint().unwrap(),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        )
        .unwrap();
    assert!(
        hermitian.block_count() > 0
            && (0..hermitian.block_count()).any(|index| {
                let block = hermitian.block(index).unwrap();
                (0..block.shape()[0]).any(|row| {
                    (0..block.shape()[1]).any(|col| {
                        row != col
                            && hermitian.data()[block.offset()
                                + row * block.strides()[0]
                                + col * block.strides()[1]]
                                .im
                                .abs()
                                > 1e-6
                    })
                })
            }),
        "Hermitian fixture must have nonzero imaginary off-diagonal entries"
    );
    hermitian
}

fn assert_c64_eigh_trunc_composition_matches_host<R>(
    source: &TensorMap<R, Complex64>,
    truncation: &Truncation,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let source_data = source.data().to_vec();
    let expected = source.eigh_trunc(truncation).unwrap();
    let device = source.to_cuda().unwrap();

    let (d_device, v_device) = device.eigh_full().unwrap();
    assert_device_factor_handles(source, [&d_device, &v_device]);
    let d = d_device.to_host().unwrap();
    let v = v_device.to_host().unwrap();

    let found = d.domain()[0]
        .find_truncated(&d.diagview().unwrap(), truncation)
        .unwrap();
    let selection = &found.selection;
    let d = d.restrict_diagonal(selection).unwrap();
    let v = v.restrict_leg(v.codomain_rank(), selection).unwrap();

    assert_eq!(*selection.subspace(), expected.d.domain()[0]);
    assert_eq!(structural_snapshot(&d), structural_snapshot(&expected.d));
    assert_eq!(structural_snapshot(&v), structural_snapshot(&expected.v));
    // Hermitian eigenvalues are real for a complex payload too.
    assert!(
        d.data().iter().all(|value| value.im.abs() <= 1e-10),
        "eigenvalues must be real"
    );
    let mut kept = d.diagview().unwrap();
    kept.sort_by(|left, right| left.sector.cmp(&right.sector));
    assert_eq!(kept.len(), expected.eigenvalues.len());
    for (actual, expected) in kept.iter().zip(&expected.eigenvalues) {
        assert_eq!(actual.sector, expected.sector);
        let values: Vec<f64> = actual.values.iter().map(|value| value.re).collect();
        assert_close(&values, &expected.values, 1e-9);
    }
    assert!((found.error - expected.error).abs() <= 1e-9 * (1.0 + expected.error));

    assert!(v.is_isometric(1e-10).unwrap(), "V^H V = I");
    if matches!(truncation, Truncation::Full) {
        let rebuilt = v
            .compose(&d)
            .unwrap()
            .compose(&v.adjoint().unwrap())
            .unwrap();
        assert_close_c64(rebuilt.data(), &source_data, 1e-9);
    }
    assert_eq!(device.to_host().unwrap().data(), source_data);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_svd_matches_host_spectra_and_truncation_policies() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = Arc::new(U1FusionRule);
    let rows = GradedSpace::try_new_with_arc(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let cols = GradedSpace::try_new_with_arc(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let rectangular =
        TensorMap::<_, Complex64>::from_block_fn(&runtime, [&rows], [&cols], distinct_c64_fill())
            .unwrap();
    assert_c64_svd_matches_host(&rectangular);

    let su2 = GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_tensor = TensorMap::<_, Complex64>::from_block_fn(
        &runtime,
        [&su2, &su2, &su2],
        [&su2, &su2],
        distinct_c64_fill(),
    )
    .unwrap();
    assert_c64_svd_matches_host(&su2_tensor);

    let unmatched_space = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let policies = [
        // Full, discard-all, and mixed keep/drop decisions.
        Truncation::Full,
        Truncation::rank(0),
        Truncation::rank(2),
        Truncation::rank(usize::MAX),
        Truncation::absolute_cutoff(0.25).unwrap(),
        Truncation::relative_inf_cutoff(0.2).unwrap(),
        Truncation::relative_error(0.1).unwrap(),
        Truncation::space(unmatched_space.truncspace()),
        Truncation::rank(3).and(Truncation::absolute_cutoff(0.1).unwrap()),
    ];
    for policy in &policies {
        assert_c64_svd_trunc_composition_matches_host(&rectangular, policy);
    }
    assert_c64_svd_trunc_composition_matches_host(&su2_tensor, &Truncation::rank(2));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_eigh_admits_hermitian_and_rejects_complex_symmetric_input() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let hermitian = hermitian_c64(
        &TensorMap::<_, Complex64>::from_block_fn(&runtime, [&u1], [&u1], distinct_c64_fill())
            .unwrap(),
    );
    assert_c64_eigh_trunc_composition_matches_host(&hermitian, &Truncation::Full);
    assert_c64_eigh_trunc_composition_matches_host(&hermitian, &Truncation::rank(2));

    let (d, v) = hermitian.to_cuda().unwrap().eigh_full().unwrap();
    let d = d.to_host().unwrap();
    let v = v.to_host().unwrap();
    assert!(v.is_isometric(1e-10).unwrap(), "V^H V = I");
    assert_close_c64(
        v.compose(&d)
            .unwrap()
            .compose(&v.adjoint().unwrap())
            .unwrap()
            .data(),
        hermitian.data(),
        1e-9,
    );

    let su2 = GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_hermitian = hermitian_c64(
        &TensorMap::<_, Complex64>::from_block_fn(
            &runtime,
            [&su2, &su2],
            [&su2, &su2],
            distinct_c64_fill(),
        )
        .unwrap(),
    );
    assert_c64_eigh_trunc_composition_matches_host(&su2_hermitian, &Truncation::Full);

    // `a^T = a` but `a^H != a`: a transpose-only admission rule would accept.
    let complex_symmetric =
        TensorMap::<_, Complex64>::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
            let (row, col) = (indices[0].min(indices[1]), indices[0].max(indices[1]));
            Complex64::new(1.0 + row as f64, 1.0 + col as f64)
        })
        .unwrap();
    assert!(
        complex_symmetric.eigh_full().is_err(),
        "the Host oracle rejects a complex-symmetric non-Hermitian input"
    );
    let device_error = complex_symmetric
        .to_cuda()
        .unwrap()
        .eigh_full()
        .expect_err("device EIGH must reject a complex-symmetric non-Hermitian input");
    assert!(
        matches!(
            &device_error,
            tenet::typed::Error::Operation(error)
                if matches!(
                    **error,
                    tenet::operations::OperationError::UnsupportedTensorContractScope { .. }
                )
        ),
        "unexpected error: {device_error:?}"
    );

    // A 2x2 hand case: `[[1, i], [-i, 1]]` is Hermitian, `[[1, i], [i, 1]]` is
    // complex-symmetric and must be rejected.
    let leg =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let hand_hermitian =
        TensorMap::<_, Complex64>::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 1) => Complex64::new(0.0, 1.0),
                (1, 0) => Complex64::new(0.0, -1.0),
                _ => Complex64::new(1.0, 0.0),
            }
        })
        .unwrap();
    let (d, _) = hand_hermitian.to_cuda().unwrap().eigh_full().unwrap();
    let mut values: Vec<f64> = d.to_host().unwrap().data().iter().map(|z| z.re).collect();
    values.sort_by(f64::total_cmp);
    assert_close(&values, &[0.0, 0.0, 0.0, 2.0], 1e-10);

    let hand_symmetric =
        TensorMap::<_, Complex64>::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 1.0)
            }
        })
        .unwrap();
    assert!(hand_symmetric.to_cuda().unwrap().eigh_full().is_err());
}

/// G3c-2 (#1276): the device destination-overwrite entry writes exactly what
/// the returning device contraction returns, including the `+0.0` of every
/// destination block no GEMM reaches, and every rejection leaves the
/// destination's bytes untouched.
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_contract_overwrite_into_matches_the_returning_contraction() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    // `k` carries only charge 0, so the destination's charge-1 block has no
    // contributing GEMM and must come out as exactly `+0.0` from the zeroing
    // of the plan's inactive blocks.
    let outer = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let inner =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let lhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&outer], [&inner], 761_000)
        .unwrap()
        .to_cuda()
        .unwrap();
    let rhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&inner], [&outer], 761_001)
        .unwrap()
        .to_cuda()
        .unwrap();

    let expected = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    let expected_data = expected.to_host().unwrap().data().to_vec();
    assert!(
        expected_data.contains(&0.0),
        "fixture must contain a destination block no GEMM reaches"
    );

    let poisoned = || {
        TensorMap::<U1FusionRule, f64>::from_block_fn(&runtime, [&outer], [&outer], |_, _| 7.5)
            .unwrap()
            .to_cuda()
            .unwrap()
    };
    let mut destination = poisoned();
    lhs.contract_overwrite_into(&rhs, &mut destination, &[1], &[0], &[0, 1], 1.0)
        .unwrap();
    let written = destination.to_host().unwrap().data().to_vec();
    assert_eq!(written.len(), expected_data.len());
    for (index, (&actual, &expected)) in written.iter().zip(&expected_data).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "element {index}: {actual} != {expected}"
        );
        if expected == 0.0 {
            assert_eq!(
                actual.to_bits(),
                0,
                "unreachable element {index} must be +0.0"
            );
        }
    }

    // Every rejection is refused before a device write: the poisoned bytes
    // survive each one unchanged.
    let poison_data = poisoned().to_host().unwrap().data().to_vec();
    let assert_rejected =
        |label: &str,
         result: Result<(), tenet::prelude::Error>,
         dst: &TensorMap<U1FusionRule, f64, CudaStorage>| {
            assert!(result.is_err(), "{label} must be rejected");
            assert_eq!(
                dst.to_host().unwrap().data(),
                poison_data.as_slice(),
                "{label} wrote to the destination before rejecting"
            );
        };

    // Alias: the destination shares the lhs payload body, directly and through
    // a lazy adjoint view of the same parent.
    let lhs_data = lhs.to_host().unwrap().data().to_vec();
    let mut lhs_alias = lhs.clone();
    assert!(
        lhs.contract_overwrite_into(&rhs, &mut lhs_alias, &[1], &[0], &[0, 1], 1.0)
            .is_err(),
        "an lhs alias must be rejected"
    );
    assert_eq!(lhs_alias.to_host().unwrap().data(), lhs_data.as_slice());
    let mut adjoint_alias = lhs.clone();
    assert!(lhs
        .adjoint()
        .unwrap()
        .contract_overwrite_into(&rhs, &mut adjoint_alias, &[0], &[0], &[0, 1], 1.0)
        .is_err());
    assert_eq!(adjoint_alias.to_host().unwrap().data(), lhs_data.as_slice());

    // Same required length, different block charges: the space check must
    // catch what the length check cannot.
    let drifted = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(4), 3)],
    )
    .unwrap();
    let mut wrong_space =
        TensorMap::<U1FusionRule, f64>::from_block_fn(&runtime, [&drifted], [&drifted], |_, _| 7.5)
            .unwrap()
            .to_cuda()
            .unwrap();
    let wrong_space_before = wrong_space.to_host().unwrap().data().to_vec();
    assert_eq!(
        wrong_space_before.len(),
        expected_data.len(),
        "the drifted destination must have the same required length"
    );
    assert!(
        lhs.contract_overwrite_into(&rhs, &mut wrong_space, &[1], &[0], &[0, 1], 1.0)
            .is_err(),
        "a destination whose block layout differs must be rejected"
    );
    assert_eq!(
        wrong_space.to_host().unwrap().data(),
        wrong_space_before.as_slice()
    );

    let mut short =
        TensorMap::<U1FusionRule, f64>::from_block_fn(&runtime, [&inner], [&inner], |_, _| 7.5)
            .unwrap()
            .to_cuda()
            .unwrap();
    let short_before = short.to_host().unwrap().data().to_vec();
    assert!(
        lhs.contract_overwrite_into(&rhs, &mut short, &[1], &[0], &[0, 1], 1.0)
            .is_err(),
        "a destination of the wrong length must be rejected"
    );
    assert_eq!(short.to_host().unwrap().data(), short_before.as_slice());

    let mut destination = poisoned();
    let shared = destination.clone();
    assert_rejected(
        "shared ownership",
        lhs.contract_overwrite_into(&rhs, &mut destination, &[1], &[0], &[0, 1], 1.0),
        &destination,
    );
    drop(shared);
    assert_rejected(
        "alpha other than one",
        lhs.contract_overwrite_into(&rhs, &mut destination, &[1], &[0], &[0, 1], 2.0),
        &destination,
    );

    // A destination whose blocks all exist but none of which any GEMM reaches:
    // the inactive-block zeroing alone produces the whole result.
    let disjoint =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(7), 2)]).unwrap();
    let empty_lhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&outer], [&disjoint], 761_002)
            .unwrap()
            .to_cuda()
            .unwrap();
    let empty_rhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&disjoint], [&outer], 761_003)
            .unwrap()
            .to_cuda()
            .unwrap();
    let empty_expected = empty_lhs.contract(&empty_rhs, &[1], &[0], &[0, 1]).unwrap();
    let mut empty_destination = poisoned();
    empty_lhs
        .contract_overwrite_into(&empty_rhs, &mut empty_destination, &[1], &[0], &[0, 1], 1.0)
        .unwrap();
    assert_eq!(
        empty_destination.to_host().unwrap().data(),
        empty_expected.to_host().unwrap().data()
    );

    // A destination space with no coupled sector at all: `required_len` is 0,
    // so the replay writes nothing.
    let single =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let zero_lhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&single], [&single], 761_004)
            .unwrap()
            .to_cuda()
            .unwrap();
    let zero_rhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&single], [&disjoint], 761_005)
            .unwrap()
            .to_cuda()
            .unwrap();
    let mut zero_destination =
        TensorMap::<U1FusionRule, f64>::from_block_fn(&runtime, [&single], [&disjoint], |_, _| 7.5)
            .unwrap()
            .to_cuda()
            .unwrap();
    assert!(
        zero_destination.to_host().unwrap().data().is_empty(),
        "the fixture must have a zero-length destination"
    );
    zero_lhs
        .contract_overwrite_into(&zero_rhs, &mut zero_destination, &[1], &[0], &[0, 1], 1.0)
        .unwrap();
    assert_eq!(
        zero_destination.to_host().unwrap().data(),
        zero_lhs
            .contract(&zero_rhs, &[1], &[0], &[0, 1])
            .unwrap()
            .to_host()
            .unwrap()
            .data()
    );
}

/// #1320: a factor block that starts at an odd element offset of the flat
/// device buffer.
///
/// The aligned whole-factor route copies each block straight into its sector
/// region, and Tenferro 0.5.0 told cuTENSOR that an offset destination view is
/// 256-byte aligned when it is not, so the launch faulted with
/// `cudaErrorMisalignedAddress` at the next synchronizing call. Every
/// degeneracy pair here puts the second sector's block at an offset whose byte
/// product is not a multiple of 256: `(3, 2)` at element 9, `(5, 2)` at 25,
/// `(3, 3)` at 9. The `(4, 2)` and `(2, 2)` pairs are the controls that passed
/// before the fix.
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_factorizations_handle_blocks_at_unaligned_offsets() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = Arc::new(U1FusionRule);

    for (d0, d1) in [(3usize, 2usize), (5, 2), (3, 3), (4, 2), (2, 2)] {
        let leg = GradedSpace::try_new_with_arc(
            Arc::clone(&u1),
            [(U1Irrep::new(0), d0), (U1Irrep::new(1), d1)],
        )
        .unwrap();
        let square = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                4.0 + indices[0] as f64
            } else {
                (1 + indices[0] + 2 * indices[1]) as f64
            }
        })
        .unwrap();
        assert_typed_cuda_svd_matches_host(&square);
        assert_typed_cuda_qr_matches_host(&square);

        // Complex64 doubles the element size, so the same block offsets are a
        // different multiple of 256; device QR stays unsupported for c64
        // (#1271), so only SVD is exercised.
        let complex =
            TensorMap::<_, Complex64>::from_block_fn(&runtime, [&leg], [&leg], distinct_c64_fill())
                .unwrap();
        assert_c64_svd_matches_host(&complex);
    }
}
