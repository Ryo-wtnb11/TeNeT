//! Real-device gates for provider-neutral typed ownership transfer.
//!
//! Run with `cargo test -p tenet --features cuda,cpu-faer --test \
//! typed_cuda_transfer -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenet::core::{
    product_sector, BraidingStyleKind, CheckedFusionAlgebra, FermionParityFusionRule,
    FusionAlgebraError, FusionRule, FusionStyleKind, MultiplicityFreeFusionRule,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, ProductFusionRuleExt,
    RuleIdentity, SU2FusionRule, SU2Irrep, SectorCodec, SectorId, SectorVec, U1FusionRule, U1Irrep,
    Z2Irrep, ZNFusionRule,
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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ProbeSector;

/// One-sector provider whose dimension callback re-enters the public Runtime
/// lock. A reduction deadlocks here if it calls provider code under that lock.
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

    fn scalar_one(&self) -> f64 {
        1.0
    }

    fn scalar_conj(&self, value: f64) -> f64 {
        value
    }

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
#[ignore = "requires a real CUDA device"]
fn typed_cuda_reductions_cover_weights_providers_lazy_and_preflight() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let u1_provider = Arc::new(U1FusionRule);
    let u1_leg = GradedSpace::try_new(
        Arc::clone(&u1_provider),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
        false,
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
    assert!(matches!(
        lazy.dot(&lazy),
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
    let su2_leg = GradedSpace::try_new(
        Arc::clone(&su2_provider),
        [(spin0, 1), (spin_half, 1)],
        false,
    )
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
    let fz2_leg = GradedSpace::try_new(
        Arc::clone(&fz2_provider),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
        false,
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
    let product_leg = GradedSpace::try_new(
        Arc::clone(&product_provider),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
        false,
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
        GradedSpace::try_new(Arc::clone(&probe_provider), [(ProbeSector, 1)], false).unwrap();
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

    let other_runtime = Runtime::builder().cuda(0).build().unwrap();
    let foreign = TensorMap::from_block_fn(&other_runtime, [&u1_leg], [&u1_leg], |_, _| 1.0)
        .unwrap()
        .to_cuda()
        .unwrap();
    assert_eq!(
        u1_device.inner(&foreign).unwrap_err(),
        tenet::typed::Error::RuntimeMismatch
    );

    let wider = GradedSpace::try_new(
        Arc::clone(&u1_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        false,
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
    let charge0 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 1)], false).unwrap();
    let charge1 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(1), 1)], false).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&charge0], [&charge1], |_, _| 1.0).unwrap();
    assert!(empty.data().is_empty());
    let empty_device = empty.to_cuda().unwrap();
    assert_eq!(empty_device.norm().unwrap(), 0.0);
    assert_eq!(empty_device.inner(&empty_device).unwrap(), 0.0);
    assert_eq!(empty_device.dot(&empty_device).unwrap(), 0.0);
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
