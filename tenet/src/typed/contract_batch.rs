//! Restricted Host binding for one ordinary contraction over owned-dense stacks.

use super::*;
use tenet_tensors::{
    DynamicTreeMembersWorkspace, OutputAxisOrder, StorageContractResolution, TensorContractSpec,
};

#[path = "contract_batch_copy_c.rs"]
mod copy_c;
use copy_c::{CopyCPlan, CopyCWorkspace};

/// Immutable Host contraction structure for owned-dense stacks.
///
/// This binding admits twist-free transformed-tree routes, fully direct
/// unit-alpha core routes, and CopyC when its temporary is such a
/// direct core followed by one completed output transform.
/// Direct composition is served by [`ComposePlan`]. The plan fixes structure
/// and axes, but not member count.
pub struct ContractPlan<R, D> {
    runtime: Runtime,
    lhs: StructureSignature,
    rhs: StructureSignature,
    output_signature: StructureSignature,
    space: BoundDynamicFusionMapSpace<R>,
    resolution: Arc<StorageContractResolution<f64>>,
    copy_c: Option<CopyCPlan<R>>,
    member_len: usize,
    _payload: PhantomData<D>,
}

/// Caller-owned output and replay scratch for a [`ContractPlan`].
pub struct ContractWorkspace<R, D> {
    binding: Arc<StorageContractResolution<f64>>,
    output: Option<StackedTensorMap<R, D>>,
    members: DynamicTreeMembersWorkspace<D>,
    replay: Option<(StackedDirectReplay, bool)>,
    copy_c: Option<CopyCWorkspace<D>>,
}

impl<R, D> ContractPlan<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// Fixes an ordinary contraction without reading operand payloads.
    /// Unsupported routes return `UnsupportedTensorContractScope`.
    pub fn new(
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        spec: &super::super::ContractSpec<'_>,
    ) -> Result<Self, Error> {
        if !lhs.runtime.same_runtime(&rhs.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if lhs.members != rhs.members {
            return Err(Error::InvalidArgument(
                "operand stacks have different member counts".into(),
            ));
        }
        super::super::checked_generic_contract::reject_non_symmetric_contraction(
            lhs.space.provider().braiding_style(),
        )?;
        let _pool = lhs.runtime.enter_host_pool();
        let output_axes = spec.output_axes();
        let order = OutputAxisOrder::from_axes(&output_axes);
        let space = BoundDynamicFusionMapSpace::contracted_multiplicity_free_partitioned(
            &lhs.space,
            &rhs.space,
            spec.lhs,
            spec.rhs,
            order,
            spec.codomain.len(),
        )?;
        let lhs_operand = FusionOperand::direct(lhs.space.space());
        let rhs_operand = FusionOperand::direct(rhs.space.space());
        let copy_c_order = tenet_tensors::zero_copy_contract_order_for_output_permute(
            space.provider(),
            space.space(),
            lhs_operand,
            rhs_operand,
            spec.lhs,
            spec.rhs,
            &output_axes,
        );
        let copy_c = copy_c_order
            .map(|orientation| {
                copy_c::CopyCGeometryBinding::new(lhs, rhs, spec, &output_axes, orientation)
            })
            .transpose()?;
        let (contract_space, first, second, first_axes, second_axes) =
            if let Some(binding) = &copy_c {
                let (first, second, first_axes, second_axes) =
                    binding.geometry.oriented(lhs, rhs, spec.lhs, spec.rhs);
                (
                    binding.temporary_space.clone(),
                    first,
                    second,
                    first_axes,
                    second_axes,
                )
            } else {
                (space.clone(), lhs, rhs, spec.lhs, spec.rhs)
            };
        let axes = TensorContractSpec::new(first_axes, second_axes, OutputAxisOrder::identity());
        let mut lease = lhs.runtime.lease_context()?;
        let lane = lease.context().multiplicity_free_lane::<D>()?;
        let resolution = lane.compile_storage_contract_resolution(
            &contract_space,
            FusionOperand::direct(first.space.space()),
            FusionOperand::direct(second.space.space()),
            if copy_c.is_some() {
                axes
            } else {
                TensorContractSpec::new(spec.lhs, spec.rhs, order)
            },
        )?;
        let direct = !resolution.is_dynamic_tree();
        if copy_c.is_some() && (!direct || resolution.admits_stacked_direct_host_replay().is_err())
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "Host copyC batch requires a unit-alpha direct temporary",
            }
            .into());
        }
        if resolution.requires_source_twist() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "Host batch contraction requires a twist-free transformed-tree or direct core route",
            }
            .into());
        }
        if direct {
            resolution.admits_stacked_direct_host_replay()?;
        }
        let copy_c = if let Some(binding) = copy_c {
            let transform = lane.tree_context_mut().compile_tree_pair_structure(
                space.provider(),
                &binding.geometry.operation,
                space.space().structure(),
                binding.temporary_space.space().structure(),
            )?;
            Some(CopyCPlan {
                temporary_space: binding.temporary_space,
                transform,
                input_swapped: binding.geometry.is_swapped(),
            })
        } else {
            None
        };
        let member_len = space.space().required_len()?;
        Ok(Self {
            runtime: lhs.runtime.clone(),
            lhs: lhs.signature.clone(),
            rhs: rhs.signature.clone(),
            output_signature: StructureSignature::of_space(&space, Placement::Host, &lhs.runtime),
            space,
            resolution: Arc::new(resolution),
            copy_c,
            member_len,
            _payload: PhantomData,
        })
    }

    /// Creates independent mutable state; member count can change later.
    pub fn workspace(&self) -> ContractWorkspace<R, D> {
        ContractWorkspace {
            binding: Arc::clone(&self.resolution),
            output: None,
            members: DynamicTreeMembersWorkspace::default(),
            replay: None,
            copy_c: self.copy_c.as_ref().map(|_| CopyCWorkspace::default()),
        }
    }

    /// Signature required of every output member.
    pub fn output_signature(&self) -> &StructureSignature {
        &self.output_signature
    }

    fn check(
        &self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        workspace: &ContractWorkspace<R, D>,
    ) -> Result<usize, Error> {
        if !Arc::ptr_eq(&self.resolution, &workspace.binding) {
            return Err(Error::InvalidArgument(
                "contract workspace belongs to another plan".into(),
            ));
        }
        for (actual, expected) in [(&lhs.signature, &self.lhs), (&rhs.signature, &self.rhs)] {
            if let Some(field) = expected.first_mismatch(actual) {
                return Err(Error::BatchSignatureMismatch {
                    member: None,
                    field,
                });
            }
        }
        if lhs.members != rhs.members {
            return Err(Error::InvalidArgument(
                "operand stacks have different member counts".into(),
            ));
        }
        for (len, actual) in [
            (lhs.member_len, lhs.storage.len()),
            (rhs.member_len, rhs.storage.len()),
        ] {
            if actual != self.total_len(len, lhs.members)? {
                return Err(Error::InvalidArgument(
                    "stacked payload length differs from its structure".into(),
                ));
            }
        }
        Ok(lhs.members)
    }

    fn total_len(&self, len: usize, members: usize) -> Result<usize, Error> {
        let total = len.checked_mul(members).ok_or_else(|| {
            Error::InvalidArgument("stacked payload length overflows usize".into())
        })?;
        std::alloc::Layout::array::<D>(total).map_err(|_| {
            Error::InvalidArgument("stacked payload allocation overflows usize".into())
        })?;
        Ok(total)
    }

    fn run(
        &self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        dst: &mut Vec<D>,
        members: usize,
        workspace: &mut ContractWorkspace<R, D>,
    ) -> Result<(), Error> {
        if let Some(copy_c) = &self.copy_c {
            return copy_c.run(self, lhs, rhs, dst, members, workspace);
        }
        if !self.resolution.is_dynamic_tree() {
            if workspace
                .replay
                .as_ref()
                .is_none_or(|(replay, _)| replay.members() != members)
            {
                workspace.replay = self.resolution.stacked_direct_host_replay(members)?;
            }
            let (replay, swapped) = workspace.replay.as_ref().ok_or_else(|| {
                Error::InvalidArgument("direct Host replay is not prepared".into())
            })?;
            let (left, right) = if *swapped { (rhs, lhs) } else { (lhs, rhs) };
            let left = StackedStorageView::new::<D>(
                &left.storage,
                left.member_len,
                members,
                left.member_len,
            )?;
            let right = StackedStorageView::new::<D>(
                &right.storage,
                right.member_len,
                members,
                right.member_len,
            )?;
            let mut destination =
                StackedStorageViewMut::new::<D>(dst, self.member_len, members, self.member_len)?;
            let mut lease = self.runtime.lease_context()?;
            lease
                .context()
                .multiplicity_free_lane::<D>()?
                .execute_stacked_direct_host(replay, &mut destination, &left, &right, true)?;
            return Ok(());
        }
        let mut lease = self.runtime.lease_context()?;
        lease
            .context()
            .multiplicity_free_lane::<D>()?
            .execute_storage_contract_members_host(
                &self.resolution,
                self.space.space().structure(),
                &mut workspace.members,
                dst,
                &lhs.storage,
                &rhs.storage,
                members,
            )?;
        Ok(())
    }

    /// Overwrites the workspace-owned output and borrows it until the next call.
    /// Validation errors preserve any previous output. After a backend error,
    /// its payload may be partially overwritten while its metadata stays valid.
    pub fn execute<'a>(
        &self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        workspace: &'a mut ContractWorkspace<R, D>,
    ) -> Result<&'a StackedTensorMap<R, D>, Error> {
        let members = self.check(lhs, rhs, workspace)?;
        let total = self.total_len(self.member_len, members)?;
        let mut output = workspace
            .output
            .take()
            .filter(|output| output.members == members)
            .unwrap_or_else(|| StackedTensorMap {
                runtime: self.runtime.clone(),
                space: self.space.clone(),
                signature: self.output_signature.clone(),
                storage: zeroed_payload(total),
                members,
                member_len: self.member_len,
                _payload: PhantomData,
            });
        let result = self.run(lhs, rhs, &mut output.storage, members, workspace);
        let output = workspace.output.insert(output);
        result.map(|()| &*output)
    }

    /// Overwrites a checked caller destination. Validation and capability
    /// errors leave it unchanged. After a backend error, its payload may be
    /// partially overwritten while its metadata stays valid.
    pub fn execute_into(
        &self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        dst: &mut StackedTensorMap<R, D>,
        workspace: &mut ContractWorkspace<R, D>,
    ) -> Result<(), Error> {
        let members = self.check(lhs, rhs, workspace)?;
        if let Some(field) = self.output_signature.first_mismatch(&dst.signature) {
            return Err(Error::BatchSignatureMismatch {
                member: None,
                field,
            });
        }
        if dst.members != members {
            return Err(Error::InvalidArgument(
                "destination member count differs from operands".into(),
            ));
        }
        if dst.storage.len() != self.total_len(self.member_len, members)? {
            return Err(Error::InvalidArgument(
                "destination payload length differs from its structure".into(),
            ));
        }
        self.run(lhs, rhs, &mut dst.storage, members, workspace)
    }
}

impl<R, D> ContractWorkspace<R, D> {
    /// Moves the workspace-owned output to the caller.
    pub fn take_output(&mut self) -> Option<StackedTensorMap<R, D>> {
        self.output.take()
    }

    /// Retained payload and replay-scratch bytes, excluding Runtime resources.
    pub fn retained_bytes(&self) -> usize {
        self.members.retained_bytes()
            + self
                .copy_c
                .as_ref()
                .map_or(0, CopyCWorkspace::retained_bytes)
            + self
                .replay
                .as_ref()
                .map_or(0, |(replay, _)| replay.retained_bytes())
            + self.output.as_ref().map_or(0, |output| {
                output
                    .storage
                    .capacity()
                    .saturating_mul(std::mem::size_of::<D>())
            })
    }
}

#[cfg(test)]
mod fermionic_unit_tests {
    use super::*;
    use crate::sector::{
        product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
        U1FusionRule, U1Irrep, Z2Irrep,
    };
    use crate::typed::GradedSpace;
    use tenet_operations::{fusion_replay::Rank2GemmBatchJob, Rank2Gemm, StridedHostKernelAdapter};

    #[derive(Default)]
    struct Count {
        calls: usize,
        jobs: usize,
    }

    impl Rank2Gemm<f64> for Count {
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
            panic!("direct member replay must submit one batch")
        }

        fn matmul_rank2_batch(
            &mut self,
            _: &mut [f64],
            _: &[f64],
            _: &[f64],
            jobs: &[Rank2GemmBatchJob],
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

    #[test]
    fn fermionic_unit_routes_submit_one_member_expanded_core_batch() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let u1 = GradedSpace::try_new(
            Arc::new(FermionParityFusionRule.product(U1FusionRule)),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
            ],
        )
        .unwrap();
        let su2 = GradedSpace::try_new(
            Arc::new(FermionParityFusionRule.product(SU2FusionRule)),
            [
                (
                    product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                    2,
                ),
                (
                    product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(1)),
                    2,
                ),
                (
                    product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                    1,
                ),
            ],
        )
        .unwrap();
        check_routes(&runtime, &u1);
        check_routes(&runtime, &su2);
    }

    fn check_routes<R>(runtime: &Runtime, v: &GradedSpace<R>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    {
        let a = TensorMap::<_, f64>::rand_with_seed(runtime, [v, v], [v, v], 61).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(runtime, [v, v], [v, v], 62).unwrap();
        let cases = [
            ("C0", [2, 3], [0, 1], [0, 1, 2, 3], 2, false, false),
            ("C1", [3, 2], [1, 0], [0, 1, 2, 3], 2, false, false),
            ("C2", [0, 1], [2, 3], [2, 3, 0, 1], 2, false, true),
            ("C1p", [3, 2], [1, 0], [1, 0, 3, 2], 1, true, false),
            ("C2p", [0, 1], [2, 3], [3, 2, 1, 0], 1, true, false),
        ];
        for (name, lhs_axes, rhs_axes, output, split, copy_c, swapped_expected) in cases {
            let spec = super::super::super::ContractSpec {
                lhs: &lhs_axes,
                rhs: &rhs_axes,
                codomain: &output[..split],
                domain: &output[split..],
            };
            let left = StackedTensorMap::pack(&[&a]).unwrap();
            let right = StackedTensorMap::pack(&[&b]).unwrap();
            let plan = ContractPlan::new(&left, &right, &spec).unwrap();
            assert_eq!(plan.copy_c.is_some(), copy_c, "{name}");
            assert!(!plan.resolution.is_dynamic_tree(), "{name}");
            let mut jobs_per_member = 0;
            for members in [1, 2, 17] {
                let (replay, swapped) = plan
                    .resolution
                    .stacked_direct_host_replay(members)
                    .unwrap()
                    .expect("unit direct core");
                if !copy_c {
                    assert_eq!(swapped, swapped_expected, "{name}");
                }
                let [dst_len, lhs_len, rhs_len] = replay.member_lens();
                let mut dst = vec![f64::NAN; dst_len * members];
                let lhs = vec![1.0; lhs_len * members];
                let rhs = vec![1.0; rhs_len * members];
                let mut count = Count::default();
                replay
                    .execute_host(
                        &mut StridedHostKernelAdapter::default(),
                        &mut count,
                        &mut StackedStorageViewMut::new::<f64>(&mut dst, dst_len, members, dst_len)
                            .unwrap(),
                        &StackedStorageView::new::<f64>(&lhs, lhs_len, members, lhs_len).unwrap(),
                        &StackedStorageView::new::<f64>(&rhs, rhs_len, members, rhs_len).unwrap(),
                        true,
                    )
                    .unwrap();
                if members == 1 {
                    jobs_per_member = count.jobs;
                    assert!(jobs_per_member > 0, "{name}");
                }
                assert_eq!(
                    (count.calls, count.jobs),
                    (1, jobs_per_member * members),
                    "{name}"
                );
            }
        }
    }
}
