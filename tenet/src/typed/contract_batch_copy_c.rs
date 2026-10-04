//! CopyC: one direct temporary contraction, then one completed output transform.

use super::*;
use tenet_operations::{
    admit_tree_transform_members_overwrite_raw, tree_transform_members_overwrite_raw,
    StridedHostKernelAdapter, TreeTransformWorkspace,
};
use tenet_tensors::CopyCRoute;

pub(super) struct CopyCWorkspace<D> {
    temporary: Vec<D>,
    transform: TreeTransformWorkspace<D>,
    #[cfg(test)]
    completed_transforms: usize,
}

impl<D> Default for CopyCWorkspace<D> {
    fn default() -> Self {
        Self {
            temporary: Vec::new(),
            transform: TreeTransformWorkspace::default(),
            #[cfg(test)]
            completed_transforms: 0,
        }
    }
}

impl<D> CopyCWorkspace<D> {
    pub(super) fn retained_bytes(&self) -> usize {
        self.temporary
            .capacity()
            .saturating_mul(std::mem::size_of::<D>())
            .saturating_add(self.transform.retained_bytes())
    }
}

/// Replays a planned `CopyC` route over Host stacks: the exact-sign direct
/// core into the workspace temporary, then the output transform into `dst`.
pub(super) fn run<R, D>(
    copy_route: &CopyCRoute<f64>,
    plan: &ContractPlan<R, D>,
    lhs: &StackedTensorMap<R, D>,
    rhs: &StackedTensorMap<R, D>,
    dst: &mut [D],
    members: usize,
    workspace: &mut ContractWorkspace<R, D>,
) -> Result<(), Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let copy = workspace
        .copy_c
        .as_mut()
        .ok_or_else(|| Error::InvalidArgument("copyC workspace belongs to another plan".into()))?;
    let temporary_len = copy_route.temporary_len();
    let total = plan.total_len(temporary_len, members)?;
    // Admission includes all transform views, coefficients and packed jobs.
    // It precedes even the temporary core submission.
    admit_tree_transform_members_overwrite_raw(
        &mut copy.transform,
        copy_route.transform(),
        plan.space.space().structure(),
        copy_route.temporary_structure(),
        dst.len(),
        total,
        members,
    )?;
    if workspace
        .replay
        .as_ref()
        .is_none_or(|(replay, _)| replay.members() != members)
    {
        workspace.replay = plan
            .resolution
            .direct_core()
            .map(|(core, swapped)| {
                StackedDirectReplay::new_signed(Arc::clone(core), members)
                    .map(|replay| (replay, swapped))
            })
            .transpose()?;
    }
    let (replay, swapped) = workspace
        .replay
        .as_ref()
        .ok_or_else(|| Error::InvalidArgument("copyC temporary is not a direct core".into()))?;
    let (left, right) = if *swapped { (rhs, lhs) } else { (lhs, rhs) };
    let left =
        StackedStorageView::new::<D>(&left.storage, left.member_len, members, left.member_len)?;
    let right =
        StackedStorageView::new::<D>(&right.storage, right.member_len, members, right.member_len)?;
    copy.temporary.resize(total, D::from_real(0.0));
    let mut temporary = StackedStorageViewMut::new::<D>(
        &mut copy.temporary,
        temporary_len,
        members,
        temporary_len,
    )?;
    let mut lease = plan.runtime.lease_context()?;
    let lane = lease.context().multiplicity_free_lane::<D>()?;
    lane.execute_stacked_signed_direct_host(replay, &mut temporary, &left, &right, true)?;
    let backend = lane.tree_context_mut().backend_mut();
    let threads = backend.recoupling_threads();
    tree_transform_members_overwrite_raw(
        &mut StridedHostKernelAdapter::default(),
        backend.dense_mut(),
        &mut copy.transform,
        copy_route.transform(),
        plan.space.space().structure(),
        copy_route.temporary_structure(),
        dst,
        &copy.temporary,
        members,
        threads,
    )?;
    #[cfg(test)]
    {
        copy.completed_transforms += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
    use crate::typed::{ContractSpec, GradedSpace, TensorMap};
    use std::sync::Arc;
    use tenet_dense::{
        DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
        DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseWrite, MatrixOp,
    };
    use tenet_operations::Rank2Gemm;

    #[derive(Default)]
    struct CoreCount {
        calls: usize,
        jobs: usize,
    }

    impl Rank2Gemm<f64> for CoreCount {
        fn matmul_rank2(
            &mut self,
            _: &mut [f64],
            _: &[f64],
            _: &[f64],
            _: usize,
            _: usize,
            _: usize,
            _: f64,
            _: f64,
        ) -> Result<(), OperationError> {
            panic!("CopyC core must submit one batch")
        }

        fn matmul_rank2_batch(
            &mut self,
            _: &mut [f64],
            _: &[f64],
            _: &[f64],
            jobs: &[tenet_operations::fusion_replay::Rank2GemmBatchJob],
            _: &[usize],
            alpha: f64,
            beta: f64,
        ) -> Result<(), OperationError> {
            assert_eq!((alpha, beta), (1.0, 0.0));
            self.calls += 1;
            self.jobs += jobs.len();
            Ok(())
        }
    }

    include!("../../tests/common/spy_executor.rs");

    #[test]
    fn poisoned_temporary_and_output_are_overwritten_on_warm_replay() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let v = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(0), 1),
                (U1Irrep::new(1), 2),
                (U1Irrep::new(2), 1),
            ],
        )
        .unwrap();
        let w = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        )
        .unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w], 55).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&v], 56).unwrap();
        let spec = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[1, 0],
            domain: &[2],
        };
        let expected = a.contract(&b, &spec).unwrap();
        let left = StackedTensorMap::pack(&[&a, &a]).unwrap();
        let right = StackedTensorMap::pack(&[&b, &b]).unwrap();
        let plan = ContractPlan::new(&left, &right, &spec).unwrap();
        assert!(plan.copy_c().is_some());
        let mut workspace = plan.workspace();
        plan.execute(&left, &right, &mut workspace).unwrap();
        let copy = workspace.copy_c.as_mut().unwrap();
        assert!(copy.temporary.contains(&0.0));
        copy.temporary.fill(f64::NAN);
        let poisoned = expected.scale(f64::NAN);
        let mut dst = StackedTensorMap::pack(&[&poisoned, &poisoned]).unwrap();
        plan.execute_into(&left, &right, &mut dst, &mut workspace)
            .unwrap();
        for i in 0..2 {
            for (&actual, &reference) in dst
                .member(i)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .zip(expected.dense_data().unwrap())
            {
                let tolerance = 128.0 * 64.0_f64.sqrt() * f64::EPSILON * reference.abs().max(1.0);
                assert!((actual - reference).abs() <= tolerance);
            }
        }
        assert!(workspace
            .copy_c
            .as_ref()
            .unwrap()
            .temporary
            .iter()
            .all(|x| x.is_finite()));
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_copy_c_zeros_poisoned_inactive_temporary_on_warm_replay() {
        let runtime = Runtime::builder().cuda(0).build().unwrap();
        let v = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(0), 1),
                (U1Irrep::new(1), 2),
                (U1Irrep::new(2), 1),
            ],
        )
        .unwrap();
        let w = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        )
        .unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w], 55).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&v], 56).unwrap();
        let spec = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[1, 0],
            domain: &[2],
        };
        let expected = a.contract(&b, &spec).unwrap();
        let lhs = StackedTensorMap::pack(&[&a, &a])
            .unwrap()
            .to_cuda()
            .unwrap();
        let rhs = StackedTensorMap::pack(&[&b, &b])
            .unwrap()
            .to_cuda()
            .unwrap();
        let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
        assert!(plan.copy_c().is_some());
        assert_eq!(plan.core().0.inactive_destination_regions().len(), 1);
        assert_eq!(
            tenet_operations::cuda_transform::CudaSingleMemberRegions::admit(
                plan.copy_c().unwrap().transform(),
            )
            .unwrap(),
            0
        );
        let mut workspace = plan.workspace().unwrap();
        plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        let (temporary, members) = workspace.copy_c_temporary.as_mut().unwrap();
        let member_len = plan.copy_c().unwrap().temporary_len();
        let lease = runtime.lease_cuda().unwrap();
        *temporary = CudaStorage::upload_members(
            &lease,
            vec![f64::NAN; *members * member_len],
            member_len,
            *members,
        )
        .unwrap();
        drop(lease);
        let poisoned = expected.scale(f64::NAN);
        let mut dst = StackedTensorMap::pack(&[&poisoned, &poisoned])
            .unwrap()
            .to_cuda()
            .unwrap();
        plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
            .unwrap();
        let actual = dst.to_host().unwrap();
        for i in 0..2 {
            for (&value, &reference) in actual
                .member(i)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .zip(expected.dense_data().unwrap())
            {
                assert!(value.is_finite());
                let tolerance = 128.0 * 64.0_f64.sqrt() * f64::EPSILON * reference.abs().max(1.0);
                assert!((value - reference).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn labeled_copy_c_routes_are_admitted_at_every_payload_precision() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        macro_rules! cases {
            ($v:expr, $c:expr, $dtype:ty) => {{
                let v = $v;
                let c = $c;
                let tensor = |out: [&GradedSpace<_>; 2], input: [&GradedSpace<_>; 2], seed| {
                    TensorMap::<_, $dtype>::rand_with_seed(&runtime, out, input, seed).unwrap()
                };
                let a = tensor([v, v], [v, v], 61);
                let b = tensor([v, v], [v, v], 62);
                let cases = [
                    ("C1p", a.clone(), b.clone(), [3, 2], [1, 0], [1, 0, 3, 2]),
                    ("C2p", a, b, [0, 1], [2, 3], [3, 2, 1, 0]),
                    (
                        "S1",
                        tensor([c, c], [v, v], 101),
                        tensor([v, v], [c, c], 102),
                        [3, 2],
                        [1, 0],
                        [2, 3, 0, 1],
                    ),
                    (
                        "S2",
                        tensor([v, v], [c, c], 103),
                        tensor([c, c], [v, v], 104),
                        [0, 1],
                        [2, 3],
                        [3, 2, 1, 0],
                    ),
                ];
                for (name, lhs, rhs, lhs_axes, rhs_axes, output) in cases {
                    let spec = ContractSpec {
                        lhs: &lhs_axes,
                        rhs: &rhs_axes,
                        codomain: &output[..2],
                        domain: &output[2..],
                    };
                    for members in [1, 2, 17] {
                        let left = StackedTensorMap::pack(&vec![&lhs; members]).unwrap();
                        let right = StackedTensorMap::pack(&vec![&rhs; members]).unwrap();
                        let plan = ContractPlan::new(&left, &right, &spec).unwrap();
                        plan.copy_c()
                            .unwrap_or_else(|| panic!("{name} did not choose CopyC"));
                        let mut workspace = plan.workspace();
                        plan.execute(&left, &right, &mut workspace).unwrap();
                    }
                }
            }};
        }
        let u1_v = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(-1), 8),
                (U1Irrep::new(0), 8),
                (U1Irrep::new(1), 8),
            ],
        )
        .unwrap();
        let u1_c = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        )
        .unwrap();
        cases!(&u1_v, &u1_c, f64);
        cases!(&u1_v, &u1_c, crate::typed::Complex64);
        cases!(&u1_v, &u1_c, f32);
        cases!(&u1_v, &u1_c, crate::typed::Complex32);
        let su2_v = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
        )
        .unwrap();
        let su2_c =
            GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(0), 1)])
                .unwrap();
        cases!(&su2_v, &su2_c, f64);
        cases!(&su2_v, &su2_c, crate::typed::Complex64);
        cases!(&su2_v, &su2_c, f32);
        cases!(&su2_v, &su2_c, crate::typed::Complex32);
    }

    #[test]
    fn su2_copy_c_orientations_are_unconjugated_scaled_single_tasks() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let v = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
        )
        .unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 61).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 62).unwrap();
        for (lhs_axes, rhs_axes, codomain, domain) in [
            ([3, 2], [1, 0], [1, 0], [3, 2]),
            ([0, 1], [2, 3], [3, 2], [1, 0]),
        ] {
            let spec = ContractSpec {
                lhs: &lhs_axes,
                rhs: &rhs_axes,
                codomain: &codomain,
                domain: &domain,
            };
            let lhs = StackedTensorMap::pack(&[&a]).unwrap();
            let rhs = StackedTensorMap::pack(&[&b]).unwrap();
            let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
            let transform = plan.copy_c().expect("route must choose CopyC").transform();
            assert!(!transform.storage_conjugate());
            let coefficients: Vec<_> = transform
                .blocks()
                .iter()
                .map(|block| match *block {
                    tenet_operations::TreeTransformBlock::Single { coefficient, .. } => {
                        transform.single_coefficient(coefficient).unwrap()
                    }
                    tenet_operations::TreeTransformBlock::Multi { .. } => {
                        panic!("reached SU2 CopyC route must contain only Single moves")
                    }
                })
                .collect();
            assert!(coefficients.contains(&-1.0));
            assert!(coefficients
                .iter()
                .any(|&coefficient| coefficient != 1.0 && (coefficient - 1.0).abs() < 1e-12));
        }
    }

    #[test]
    fn compiled_copy_c_core_and_transform_submit_member_expanded_jobs() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let v = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
        )
        .unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 61).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 62).unwrap();
        let spec = ContractSpec {
            lhs: &[3, 2],
            rhs: &[1, 0],
            codomain: &[1, 0],
            domain: &[3, 2],
        };
        let mut core_jobs_per_member = 0;
        let mut transform_jobs_per_member = 0;
        for members in [1, 2, 17] {
            let lhs = StackedTensorMap::pack(&vec![&a; members]).unwrap();
            let rhs = StackedTensorMap::pack(&vec![&b; members]).unwrap();
            let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
            let copy = plan.copy_c().expect("C1p must choose CopyC");
            assert!(
                copy.transform().blocks().iter().any(|block| match *block {
                    tenet_operations::TreeTransformBlock::Single { coefficient, .. } =>
                        copy.transform().single_coefficient(coefficient).unwrap() != 1.0,
                    tenet_operations::TreeTransformBlock::Multi { .. } => true,
                }),
                "this selected SU2 CopyC route must contain a scaled Single move"
            );
            let mut workspace = plan.workspace();
            let expected = plan
                .execute(&lhs, &rhs, &mut workspace)
                .unwrap()
                .storage
                .clone();
            assert_eq!(workspace.copy_c.as_ref().unwrap().completed_transforms, 1);

            let (replay, _) = workspace.replay.as_ref().unwrap();
            let [dst_len, left_len, right_len] = replay.member_lens();
            let mut core_output = vec![f64::NAN; dst_len * members];
            let left = vec![1.0; left_len * members];
            let right = vec![1.0; right_len * members];
            let mut core = CoreCount::default();
            replay
                .execute_host(
                    &mut StridedHostKernelAdapter::default(),
                    &mut core,
                    &mut StackedStorageViewMut::new::<f64>(
                        &mut core_output,
                        dst_len,
                        members,
                        dst_len,
                    )
                    .unwrap(),
                    &StackedStorageView::new::<f64>(&left, left_len, members, left_len).unwrap(),
                    &StackedStorageView::new::<f64>(&right, right_len, members, right_len).unwrap(),
                    true,
                )
                .unwrap();
            assert_eq!(core.calls, 1);
            if members == 1 {
                core_jobs_per_member = core.jobs;
            }
            assert!(core_jobs_per_member > 0);
            assert_eq!(core.jobs, core_jobs_per_member * members);

            let mut output = vec![f64::NAN; expected.len()];
            let transform_counts = Arc::new(SpyCounts::default());
            let mut transform = SpyExecutor::counting(&transform_counts);
            tree_transform_members_overwrite_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut transform,
                &mut TreeTransformWorkspace::default(),
                copy.transform(),
                plan.space.space().structure(),
                copy.temporary_structure(),
                &mut output,
                &workspace.copy_c.as_ref().unwrap().temporary,
                members,
                1,
            )
            .unwrap();
            if members == 1 {
                transform_jobs_per_member = transform_counts.batch_jobs();
            }
            assert_eq!(
                transform_jobs_per_member,
                copy.transform().recoupling_plan().jobs().len()
            );
            assert_eq!(transform_jobs_per_member, 0);
            assert_eq!(
                (
                    transform_counts.of(&[Kernel::MatmulBatch, Kernel::MatmulBatchOps]),
                    transform_counts.batch_jobs()
                ),
                (
                    usize::from(transform_jobs_per_member > 0),
                    transform_jobs_per_member * members
                )
            );
            for (actual, reference) in output.iter().zip(&expected) {
                let tolerance = 128.0 * 64.0_f64.sqrt() * f64::EPSILON * reference.abs().max(1.0);
                assert!((actual - reference).abs() <= tolerance);
            }
        }
    }
}
