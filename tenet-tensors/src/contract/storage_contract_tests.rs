//! The Host compile entry of the storage (device) contraction route, and —
//! with the `cuda` feature, on a real device — its executor replaying the
//! Host `DynamicTree` artifact (G2c-1a, #1345).
//!
//! The device tests force every orientation and axis-order candidate through
//! the test-only plan builder, which the typed API cannot reach, and compare
//! the device replay of each artifact with the Host replay of the same
//! artifact. Independent oracles for the typed lowering live in
//! `tenet/tests/typed_contract_host_oracle.rs` and `typed_cuda_contract.rs`.

use std::sync::Arc;

use tenet_core::{
    FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
    SU2FusionRule, SU2Irrep, SectorId, SectorLeg, U1FusionRule, U1Irrep,
};
use tenet_operations::{OutputAxisOrder, TensorContractSpec};

use crate::{BoundDynamicFusionMapSpace, FusionOperand, RuleIdentity};

type Context<D> = crate::TensorContractFusionExecutionContext<D, RuleIdentity>;

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

fn u1_leg(dual: bool) -> SectorLeg {
    SectorLeg::new(
        [
            (U1Irrep::new(-1).sector_id(), 2),
            (U1Irrep::new(0).sector_id(), 1),
            (U1Irrep::new(1).sector_id(), 2),
        ],
        dual,
    )
}

fn su2_leg() -> SectorLeg {
    SectorLeg::new(
        [
            (SU2Irrep::from_twice_spin(0).sector_id(), 2),
            (SU2Irrep::from_twice_spin(1).sector_id(), 1),
            (SU2Irrep::from_twice_spin(2).sector_id(), 1),
        ],
        false,
    )
}

/// A rank-5 × rank-4 U(1) contraction on a codomain leg against a domain leg
/// and a domain leg against a codomain leg, contracted out of order, with a
/// permuted output: source transforms on both sides and an output transform.
struct Case<R> {
    lhs: BoundDynamicFusionMapSpace<R>,
    rhs: BoundDynamicFusionMapSpace<R>,
    lhs_axes: Vec<usize>,
    rhs_axes: Vec<usize>,
    output_axes: Vec<usize>,
}

impl<R: MultiplicityFreeRigidSymbols<Scalar = f64>> Case<R> {
    fn dst(&self) -> BoundDynamicFusionMapSpace<R> {
        BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            &self.lhs,
            &self.rhs,
            &self.lhs_axes,
            &self.rhs_axes,
            OutputAxisOrder::from_axes(&self.output_axes),
        )
        .unwrap()
    }

    fn axes(&self) -> TensorContractSpec<'_> {
        TensorContractSpec::new(
            &self.lhs_axes,
            &self.rhs_axes,
            OutputAxisOrder::from_axes(&self.output_axes),
        )
    }
}

fn u1_case() -> Case<U1FusionRule> {
    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    Case {
        lhs: space(&provider, vec![v(), v(), v()], vec![v(), v()]),
        rhs: space(&provider, vec![v(), v()], vec![v(), v()]),
        lhs_axes: vec![3, 1],
        rhs_axes: vec![0, 3],
        output_axes: vec![2, 0, 4, 1, 3],
    }
}

fn su2_case() -> Case<SU2FusionRule> {
    let provider = Arc::new(SU2FusionRule);
    let s = su2_leg;
    Case {
        lhs: space(&provider, vec![s(), s()], vec![s(), s()]),
        rhs: space(&provider, vec![s(), s()], vec![s()]),
        lhs_axes: vec![3, 2],
        rhs_axes: vec![0, 1],
        output_axes: vec![1, 0, 2],
    }
}

/// `lhs` already has its contracted leg as its whole domain: under LhsRhs
/// its source transform is the identity and it is read in place; the output
/// is permuted, so an output transform remains.
fn lhs_identity_case() -> Case<U1FusionRule> {
    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    Case {
        lhs: space(&provider, vec![v(), v()], vec![v()]),
        rhs: space(&provider, vec![v(), v()], vec![v()]),
        lhs_axes: vec![2],
        rhs_axes: vec![1],
        output_axes: vec![3, 1, 0, 2],
    }
}

/// `rhs` already has its contracted leg as its whole codomain.
fn rhs_identity_case() -> Case<U1FusionRule> {
    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    Case {
        lhs: space(&provider, vec![v(), v()], vec![v()]),
        rhs: space(&provider, vec![u1_leg(true)], vec![v(), v()]),
        lhs_axes: vec![0],
        rhs_axes: vec![0],
        output_axes: vec![1, 0, 2, 3],
    }
}

#[test]
fn identity_source_fixtures_compile_a_dynamic_tree() {
    for case in [lhs_identity_case(), rhs_identity_case()] {
        let resolution = Context::<f64>::default()
            .compile_storage_contract_resolution(
                &case.dst(),
                FusionOperand::direct(case.lhs.space()),
                FusionOperand::direct(case.rhs.space()),
                case.axes(),
            )
            .unwrap();
        assert!(resolution.is_dynamic_tree());
    }
}

#[test]
fn canonical_axes_keep_the_direct_core_route_and_others_compile_one_dynamic_tree() {
    let case = u1_case();
    let dst = case.dst();
    let mut context = Context::<f64>::default();
    let general = context
        .compile_storage_contract_resolution(
            &dst,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap();
    // What: arbitrary axes take the Host DynamicTree artifact, bosonic so no
    // core-right twist.
    assert!(general.is_dynamic_tree());
    assert!(!general.requires_core_right_twist());
    let su2 = su2_case();
    let su2_resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(
            &su2.dst(),
            FusionOperand::direct(su2.lhs.space()),
            FusionOperand::direct(su2.rhs.space()),
            su2.axes(),
        )
        .unwrap();
    assert!(su2_resolution.is_dynamic_tree());

    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    let lhs = space(&provider, vec![v(), v()], vec![v()]);
    let rhs = space(&provider, vec![v()], vec![v(), v()]);
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lhs,
        &rhs,
        &[2],
        &[0],
        OutputAxisOrder::identity(),
    )
    .unwrap();
    let canonical = context
        .compile_storage_contract_resolution(
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::identity()),
        )
        .unwrap();
    // What: the TensorKit `mul!` form stays on the fully-direct core route.
    assert!(!canonical.is_dynamic_tree());

    // What: a lazy adjoint with arbitrary axes compiles the prelowered
    // artifact, never the Host's dense `Structure` route.
    let adjoint_bound = lhs.adjoint_view().unwrap();
    let rhs3 = space(&provider, vec![v(), v()], vec![v()]);
    let output = [3, 0, 1, 2];
    let lazy_dst = crate::BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &adjoint_bound,
        &rhs3,
        &[1],
        &[0],
        OutputAxisOrder::from_axes(&output),
    )
    .unwrap();
    let lazy = context
        .compile_storage_contract_resolution(
            &lazy_dst,
            FusionOperand::adjoint(lhs.space()),
            FusionOperand::direct(rhs3.space()),
            TensorContractSpec::new_with_conjugation(
                &[1],
                &[0],
                OutputAxisOrder::from_axes(&output),
                true,
                false,
            ),
        )
        .unwrap();
    assert!(lazy.is_dynamic_tree());
    assert!(!lazy.requires_core_right_twist());
}

#[test]
fn a_fermionic_dual_contracted_leg_on_a_transformed_operand_reports_the_twist() {
    let provider = Arc::new(FermionParityFusionRule);
    let odd_even = |dual| SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], dual);
    // `rhs` codomain leg 1 is dual: contracting it is a supertrace twist on
    // the core-right operand.
    let lhs = space(
        &provider,
        vec![odd_even(false), odd_even(false)],
        vec![odd_even(false)],
    );
    let rhs = space(
        &provider,
        vec![odd_even(false), odd_even(true)],
        vec![odd_even(false)],
    );
    let lhs_axes = [2, 0];
    let rhs_axes = [0, 1];
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lhs,
        &rhs,
        &lhs_axes,
        &rhs_axes,
        OutputAxisOrder::identity(),
    )
    .unwrap();
    let mut context = Context::<f64>::default();
    let resolution = context
        .compile_storage_contract_resolution(
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            TensorContractSpec::new(&lhs_axes, &rhs_axes, OutputAxisOrder::identity()),
        )
        .unwrap();
    // What: the classification the typed device boundary rejects on (G2c-2).
    assert!(resolution.is_dynamic_tree());
    assert!(resolution.requires_core_right_twist());
}

/// Why the device executor needs no zero fill for a core job with
/// `contracted == 0`: no admitted space has one. A zero degeneracy is dropped
/// from the leg, so every operand block has a non-empty contracted extent and
/// a coupled sector with nothing to contract has no job at all — its
/// destination block is then an inactive block, zeroed by the core zeroing
/// rule instead.
#[test]
fn a_zero_degeneracy_sector_never_yields_a_zero_extent_core_job() {
    let provider = Arc::new(U1FusionRule);
    let leg = || {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 1),
                (U1Irrep::new(1).sector_id(), 1),
            ],
            false,
        )
    };
    let empty = || {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 0),
                (U1Irrep::new(1).sector_id(), 2),
            ],
            false,
        )
    };
    let case = Case {
        lhs: space(&provider, vec![leg(), leg()], vec![empty()]),
        rhs: space(&provider, vec![empty()], vec![leg()]),
        lhs_axes: vec![2],
        rhs_axes: vec![0],
        output_axes: vec![1, 0, 2],
    };
    let no_zero_extent = |space: &crate::DynamicFusionMapSpace| {
        let structure = space.structure();
        (0..structure.block_count())
            .all(|index| !structure.block(index).unwrap().shape().contains(&0))
    };
    assert!(no_zero_extent(case.lhs.space()));
    assert!(no_zero_extent(case.rhs.space()));
    let dst = case.dst();
    let resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(
            &dst,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap();
    assert!(resolution.is_dynamic_tree());
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;

    use num_complex::Complex64;
    use tenet_dense::{CudaDenseContext, CudaScalar};
    use tenet_operations::cuda::CudaStorage;
    use tenet_operations::CudaTreeTransformExecutor;

    use crate::contract::dynamic::{
        compile_dynamic_tree_execution_artifact, execute_dynamic_tree_execution_artifact,
        DynamicFusionSpaceCache,
    };
    use crate::contract::fusion::FusionContractOrientation;
    use crate::contract::resolution::{StorageContractResolution, StorageContractRoute};
    use crate::contract::scratch::DynamicFusionScratchWorkspace;
    use crate::tree_context::TreeTransformExecutionContext;
    use crate::{
        CudaContractScratch, DenseRecouplingScalar, DenseTreeTransformOperations,
        RecouplingCoefficientAction,
    };

    trait Payload:
        CudaScalar + DenseRecouplingScalar + RecouplingCoefficientAction<f64> + std::fmt::Debug
    {
        fn fill(index: usize) -> Self;
        fn nan() -> Self;
        fn distance(self, other: Self) -> f64;
        fn magnitude(self) -> f64;
    }

    impl Payload for f64 {
        fn fill(index: usize) -> Self {
            (index as f64 * 0.37 + 0.1).sin()
        }
        fn nan() -> Self {
            f64::NAN
        }
        fn distance(self, other: Self) -> f64 {
            (self - other).abs()
        }
        fn magnitude(self) -> f64 {
            self.abs()
        }
    }

    impl Payload for Complex64 {
        fn fill(index: usize) -> Self {
            Complex64::new(
                (index as f64 * 0.37 + 0.1).sin(),
                (index as f64 * 0.23).cos(),
            )
        }
        fn nan() -> Self {
            Complex64::new(f64::NAN, f64::NAN)
        }
        fn distance(self, other: Self) -> f64 {
            (self - other).norm()
        }
        fn magnitude(self) -> f64 {
            self.norm()
        }
    }

    fn data<D: Payload>(space: &crate::DynamicFusionMapSpace, salt: usize) -> Vec<D> {
        (0..space.required_len().unwrap())
            .map(|index| D::fill(index * 7 + salt))
            .collect()
    }

    fn assert_close<D: Payload>(actual: &[D], expected: &[D], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        assert!(
            expected.iter().any(|value| value.magnitude() > 1e-3),
            "{what}: vacuous all-zero reference"
        );
        for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
            assert!(
                left.distance(right) <= 1e-12 * (1.0 + right.magnitude()),
                "{what}: element {index} is {left:?}, expected {right:?}"
            );
        }
    }

    struct Replay<D: Payload> {
        ctx: CudaDenseContext,
        transforms: CudaTreeTransformExecutor,
        scratch: CudaContractScratch,
        _payload: std::marker::PhantomData<D>,
    }

    impl<D: Payload> Replay<D> {
        fn new() -> Self {
            Self {
                ctx: CudaDenseContext::new(0).expect("a CUDA device"),
                transforms: CudaTreeTransformExecutor::default(),
                scratch: CudaContractScratch::default(),
                _payload: std::marker::PhantomData,
            }
        }

        /// Device and Host replays of one artifact for one forced candidate.
        fn run<R>(
            &mut self,
            case: &Case<R>,
            candidate: &crate::contract::fusion::ContractAxisOrderCandidate,
            orientation: FusionContractOrientation,
        ) -> (Vec<D>, Vec<D>, bool, bool)
        where
            R: MultiplicityFreeRigidSymbols<Scalar = f64>
                + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
        {
            let rule = case.lhs.provider();
            let dst = case.dst();
            let plan =
                crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
                    rule,
                    dst.space(),
                    case.lhs.space(),
                    case.rhs.space(),
                    case.axes(),
                    candidate,
                    orientation,
                )
                .unwrap();
            let mut tree_context = TreeTransformExecutionContext::new(
                DenseTreeTransformOperations::default_executor(),
            );
            let mut cache = DynamicFusionSpaceCache::default();
            let artifact = compile_dynamic_tree_execution_artifact::<_, _, _, D, _, false>(
                &mut tree_context,
                &mut cache,
                rule,
                crate::contract::encoded_layout_primer::<R>,
                &plan,
                dst.space(),
                case.lhs.space(),
                case.lhs.space().structure(),
                case.rhs.space(),
                case.rhs.space().structure(),
                None,
            )
            .unwrap();
            let lhs = data::<D>(case.lhs.space(), 1);
            let rhs = data::<D>(case.rhs.space(), 2);
            let mut host = vec![D::ZERO; dst.space().required_len().unwrap()];
            execute_dynamic_tree_execution_artifact(
                &mut tree_context,
                &mut DenseTreeTransformOperations::default(),
                &mut crate::contract::backend::TensorContractWorkspace::default(),
                &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
                &mut DynamicFusionScratchWorkspace::default(),
                &artifact,
                dst.space().structure(),
                &mut host,
                &lhs,
                &rhs,
                D::ONE,
                D::ZERO,
            )
            .unwrap();
            let borrowed = artifact.borrowed_sources();
            let resolution = StorageContractResolution {
                route: StorageContractRoute::DynamicTree(Arc::new(artifact)),
            };
            let device = self.execute(&resolution, &dst, &lhs, &rhs);
            (device, host, borrowed.0, borrowed.1)
        }

        fn execute<R>(
            &mut self,
            resolution: &StorageContractResolution<f64>,
            dst: &BoundDynamicFusionMapSpace<R>,
            lhs: &[D],
            rhs: &[D],
        ) -> Vec<D> {
            let ctx = &mut self.ctx;
            let lhs = CudaStorage::<D>::upload(ctx, lhs).unwrap();
            let rhs = CudaStorage::<D>::upload(ctx, rhs).unwrap();
            let mut out = CudaStorage::<D>::upload_owned(
                ctx,
                vec![D::ZERO; dst.space().required_len().unwrap()],
            )
            .unwrap();
            crate::execute_storage_contract_resolution_on_cuda(
                ctx,
                &mut self.transforms,
                &mut self.scratch,
                resolution,
                dst.space().structure(),
                &mut out,
                &lhs,
                &rhs,
            )
            .unwrap();
            out.download(ctx).unwrap()
        }
    }

    fn every_candidate_and_orientation<R, D>(case: &Case<R>, what: &str) -> (bool, bool)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
        D: Payload,
    {
        let mut replay = Replay::<D>::new();
        let mut any_borrow = (false, false);
        for candidate in
            crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes)
        {
            for orientation in [
                FusionContractOrientation::LhsRhs,
                FusionContractOrientation::RhsLhs,
            ] {
                let (device, host, lhs_borrowed, rhs_borrowed) =
                    replay.run(case, &candidate, orientation);
                any_borrow.0 |= lhs_borrowed;
                any_borrow.1 |= rhs_borrowed;
                assert_close(
                    &device,
                    &host,
                    &format!("{what} {candidate:?} {orientation:?}"),
                );
            }
        }
        any_borrow
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn every_forced_candidate_and_orientation_matches_the_host_replay() {
        // What: both orientations of every axis-order candidate, U(1) rank 5
        // and SU(2) recoupling, real and complex, device == Host of the same
        // artifact.
        every_candidate_and_orientation::<_, f64>(&u1_case(), "U(1) f64");
        every_candidate_and_orientation::<_, Complex64>(&u1_case(), "U(1) c64");
        every_candidate_and_orientation::<_, f64>(&su2_case(), "SU(2) f64");
        every_candidate_and_orientation::<_, Complex64>(&su2_case(), "SU(2) c64");
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn an_identity_source_transform_borrows_either_operand_on_device() {
        // `lhs` already has its contracted legs as its whole domain in
        // order, `rhs` its as its whole codomain: under LhsRhs the lhs is
        // borrowed, and the permuted output keeps an output transform.
        let lhs_canonical = lhs_identity_case();
        let borrowed = every_candidate_and_orientation::<_, f64>(&lhs_canonical, "lhs borrow");
        let rhs_canonical = rhs_identity_case();
        let rhs_borrowed = every_candidate_and_orientation::<_, f64>(&rhs_canonical, "rhs borrow");
        // What: non-vacuity — some forced artifact really borrowed each side.
        assert!(borrowed.0, "no artifact borrowed the lhs");
        assert!(rhs_borrowed.1, "no artifact borrowed the rhs");
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn a_nan_poisoned_core_destination_scratch_is_zeroed_on_its_inactive_blocks() {
        // `v` carries charge 2 that `w` lacks: the core destination has a
        // (2, 2)-coupled block no GEMM job writes, so a retained scratch that
        // still holds NaN leaks unless exactly those blocks are zeroed.
        let provider = Arc::new(U1FusionRule);
        let v = || {
            SectorLeg::new(
                [
                    (U1Irrep::new(0).sector_id(), 1),
                    (U1Irrep::new(1).sector_id(), 2),
                    (U1Irrep::new(2).sector_id(), 1),
                ],
                false,
            )
        };
        let w = || {
            SectorLeg::new(
                [
                    (U1Irrep::new(0).sector_id(), 2),
                    (U1Irrep::new(1).sector_id(), 1),
                ],
                false,
            )
        };
        let case = Case {
            lhs: space(&provider, vec![v(), v()], vec![w()]),
            rhs: space(&provider, vec![w()], vec![v()]),
            lhs_axes: vec![2],
            rhs_axes: vec![0],
            output_axes: vec![1, 0, 2],
        };
        let mut replay = Replay::<f64>::new();
        let candidate = crate::contract::contracted_axis_order_candidates(&[2], &[0]).remove(0);
        let (first, host, _, _) = replay.run(&case, &candidate, FusionContractOrientation::LhsRhs);
        assert_close(&first, &host, "cold");
        let poisoned = replay
            .scratch
            .poison_core_destination::<f64>(&replay.ctx, f64::nan());
        assert!(poisoned > 0, "the first replay retained a core destination");
        let (second, host, _, _) = replay.run(&case, &candidate, FusionContractOrientation::LhsRhs);
        assert!(second.iter().all(|value| value.is_finite()), "{second:?}");
        assert_close(&second, &host, "after poisoning");
    }
}
