//! Restricted binding for one ordinary contraction over owned-dense stacks.

use super::*;
use tenet_tensors::{
    DynamicTreeMembersWorkspace, OutputAxisOrder, StorageContractResolution, TensorContractSpec,
};

#[path = "contract_batch_copy_c.rs"]
mod copy_c;
use copy_c::{CopyCPlan, CopyCWorkspace};

// Only the two owned-dense stack payloads have an execution implementation.
trait ContractBatchStorage<D>: TensorStorage<D> {}
impl<D> ContractBatchStorage<D> for Vec<D> {}
#[cfg(feature = "cuda")]
impl<D: CudaPayload> ContractBatchStorage<D> for CudaStorage<D> {}

/// Immutable contraction structure for owned-dense Host or CUDA stacks.
///
/// This binding admits owned-source transformed-tree routes, fully direct
/// Core/SwappedCore routes with exact +1/-1 coefficients, and CopyC when its
/// temporary is a unit-alpha direct core followed by one output transform.
/// CUDA admits only fully direct unit-alpha Core/SwappedCore routes. Direct
/// composition is served by [`ComposePlan`]. The plan fixes structure and
/// axes, but not member count.
pub struct ContractPlan<R, D, S = Vec<D>> {
    runtime: Runtime,
    lhs: StructureSignature,
    rhs: StructureSignature,
    output_signature: StructureSignature,
    space: BoundDynamicFusionMapSpace<R>,
    resolution: Arc<StorageContractResolution<f64>>,
    #[cfg(feature = "cuda")]
    device_plan: Option<(Arc<tenet_operations::FusionBlockContractPlan<f64>>, bool)>,
    copy_c: Option<CopyCPlan<R>>,
    member_len: usize,
    _payload: PhantomData<(D, S)>,
}

/// Caller-owned output and execution resources for a [`ContractPlan`].
pub struct ContractWorkspace<R, D, S = Vec<D>> {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    runtime: Runtime,
    binding: Arc<StorageContractResolution<f64>>,
    output: Option<StackedTensorMap<R, D, S>>,
    members: DynamicTreeMembersWorkspace<D>,
    replay: Option<(StackedDirectReplay, bool)>,
    copy_c: Option<CopyCWorkspace<D>>,
    #[cfg(feature = "cuda")]
    device: DeviceComposeState,
}

#[allow(private_bounds)]
impl<R, D, S> ContractPlan<R, D, S>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
    S: ContractBatchStorage<D>,
{
    /// Fixes an ordinary contraction without reading operand payloads.
    /// CUDA admits only unit-alpha direct Core/SwappedCore routes.
    pub fn new(
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
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
        let placement = lhs.signature.placement;
        if rhs.signature.placement != placement {
            return Err(Error::PlacementMismatch);
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
        if direct {
            if matches!(placement, Placement::Host) {
                resolution.admits_stacked_signed_direct_host_replay()?;
            } else {
                resolution.admits_stacked_direct_host_replay()?;
            }
        }
        if !matches!(placement, Placement::Host) && (!direct || copy_c.is_some()) {
            return Err(OperationError::UnsupportedTensorContractScope {
                message:
                    "CUDA contract batch requires a unit-alpha direct Core or SwappedCore route",
            }
            .into());
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
        #[cfg(feature = "cuda")]
        let device_plan = if matches!(placement, Placement::Cuda(_)) {
            resolution.unit_direct_core_plan()?
        } else {
            None
        };
        Ok(Self {
            runtime: lhs.runtime.clone(),
            lhs: lhs.signature.clone(),
            rhs: rhs.signature.clone(),
            output_signature: StructureSignature::of_space(&space, placement, &lhs.runtime),
            space,
            resolution: Arc::new(resolution),
            #[cfg(feature = "cuda")]
            device_plan,
            copy_c,
            member_len,
            _payload: PhantomData,
        })
    }

    /// Signature required of every output member.
    pub fn output_signature(&self) -> &StructureSignature {
        &self.output_signature
    }

    fn check(
        &self,
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
        workspace: &ContractWorkspace<R, D, S>,
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
}

impl<R, D> ContractPlan<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// Creates independent mutable state; member count can change later.
    pub fn workspace(&self) -> ContractWorkspace<R, D> {
        ContractWorkspace {
            runtime: self.runtime.clone(),
            binding: Arc::clone(&self.resolution),
            output: None,
            members: DynamicTreeMembersWorkspace::default(),
            replay: None,
            copy_c: self.copy_c.as_ref().map(|_| CopyCWorkspace::default()),
            #[cfg(feature = "cuda")]
            device: DeviceComposeState::default(),
        }
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
                workspace.replay = self.resolution.stacked_signed_direct_host_replay(members)?;
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
                .execute_stacked_signed_direct_host(
                    replay,
                    &mut destination,
                    &left,
                    &right,
                    true,
                )?;
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

impl<R, D, S> ContractWorkspace<R, D, S> {
    /// Moves the workspace-owned output to the caller.
    pub fn take_output(&mut self) -> Option<StackedTensorMap<R, D, S>> {
        self.output.take()
    }

    fn retained_scratch_bytes(&self) -> usize {
        self.members.retained_bytes()
            + self
                .copy_c
                .as_ref()
                .map_or(0, CopyCWorkspace::retained_bytes)
            + self
                .replay
                .as_ref()
                .map_or(0, |(replay, _)| replay.retained_bytes())
    }
}

impl<R, D> ContractWorkspace<R, D> {
    /// Retained Host payload capacity and replay scratch, excluding Runtime resources.
    pub fn retained_bytes(&self) -> usize {
        self.retained_scratch_bytes()
            + self.output.as_ref().map_or(0, |output| {
                output
                    .storage
                    .capacity()
                    .saturating_mul(std::mem::size_of::<D>())
            })
    }
}

#[cfg(feature = "cuda")]
impl<R, D: CudaPayload> ContractWorkspace<R, D, CudaStorage<D>> {
    /// Retained device payload and Host layout scratch, excluding Runtime resources.
    pub fn retained_bytes(&self) -> usize {
        self.retained_scratch_bytes()
            + self.output.as_ref().map_or(0, |output| {
                output
                    .members
                    .saturating_mul(output.member_len)
                    .saturating_mul(std::mem::size_of::<D>())
            })
            + self.device.zero_regions.capacity() * std::mem::size_of::<tenet_dense::CudaRegion>()
            + self
                .device
                .zero_regions
                .iter()
                .map(|region| 2 * region.dims().len() * std::mem::size_of::<usize>())
                .sum::<usize>()
    }
}

#[cfg(feature = "cuda")]
impl<R, D> ContractPlan<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    fn core(&self) -> &(Arc<tenet_operations::FusionBlockContractPlan<f64>>, bool) {
        self.device_plan
            .as_ref()
            .expect("admitted CUDA direct core")
    }

    /// Creates independent B-dependent output, zero-fill, and plan resources.
    pub fn workspace(&self) -> Result<ContractWorkspace<R, D, CudaStorage<D>>, Error> {
        let mut workspace = ContractWorkspace {
            runtime: self.runtime.clone(),
            binding: Arc::clone(&self.resolution),
            output: None,
            members: DynamicTreeMembersWorkspace::default(),
            replay: None,
            copy_c: None,
            device: DeviceComposeState::default(),
        };
        let plan = &self.core().0;
        let gemms = plan.distinct_direct_gemm_shapes();
        let fills = plan
            .inactive_destination_regions()
            .iter()
            .map(|layout| (&layout.block.shape, &layout.block.strides))
            .collect::<std::collections::HashSet<_>>()
            .len();
        workspace.device.reserved_plan_entries = self
            .runtime
            .lease_cuda()?
            .reserve_plan_entries(gemms + fills)
            .map_err(tenet_operations::OperationError::Dense)?;
        Ok(workspace)
    }

    fn prepare_zero_regions(
        &self,
        workspace: &mut ContractWorkspace<R, D, CudaStorage<D>>,
        ctx: &mut tenet_dense::CudaDenseContext,
        members: usize,
    ) -> Result<(), Error> {
        if workspace.device.members == members {
            return Ok(());
        }
        let mut regions = Vec::with_capacity(self.core().0.inactive_destination_regions().len());
        let mut largest = 0usize;
        for layout in self.core().0.inactive_destination_regions() {
            let block = &layout.block;
            let unsigned = |value: isize| {
                usize::try_from(value).map_err(|_| {
                    Error::InvalidArgument(
                        "inactive destination layout has a negative stride".into(),
                    )
                })
            };
            let mut dims = block.shape.clone();
            dims.push(members);
            let mut strides = block
                .strides
                .iter()
                .map(|&stride| unsigned(stride))
                .collect::<Result<Vec<_>, _>>()?;
            strides.push(self.member_len);
            let region = tenet_dense::CudaRegion::new(dims, strides, unsigned(block.offset)?)
                .map_err(tenet_operations::OperationError::Dense)?;
            region
                .validate_as_destination("prepared contract zero fill")
                .map_err(tenet_operations::OperationError::Dense)?;
            largest = largest.max(block.shape.iter().product::<usize>());
            regions.push(region);
        }
        ctx.reserve_zero_template::<D>(
            largest
                .checked_mul(members)
                .ok_or_else(|| Error::InvalidArgument("zero template length overflows".into()))?,
        )
        .map_err(tenet_operations::OperationError::Dense)?;
        workspace.device.zero_regions = regions;
        workspace.device.members = members;
        Ok(())
    }

    fn run_cuda(
        &self,
        workspace: &ContractWorkspace<R, D, CudaStorage<D>>,
        ctx: &mut tenet_dense::CudaDenseContext,
        lhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        rhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        dst: &mut CudaStorage<D>,
        zero_inactive: bool,
    ) -> Result<(), Error> {
        let (plan, swapped) = self.core();
        let (left, right) = if *swapped { (rhs, lhs) } else { (lhs, rhs) };
        let left = StackedStorageView::new::<D>(
            &left.storage,
            left.member_len,
            lhs.members,
            left.member_len,
        )?;
        let right = StackedStorageView::new::<D>(
            &right.storage,
            right.member_len,
            lhs.members,
            right.member_len,
        )?;
        let mut dst =
            StackedStorageViewMut::new::<D>(dst, self.member_len, lhs.members, self.member_len)?;
        if zero_inactive {
            for region in &workspace.device.zero_regions {
                tenet_dense::cuda_region_zero::<D>(ctx, &mut dst.storage_mut().0, region)
                    .map_err(tenet_operations::OperationError::Dense)?;
            }
        }
        plan.execute_direct_on_storage_prezeroed(
            &mut tenet_operations::cuda::CudaStackedStorageGemm::new(ctx),
            &mut dst,
            &left,
            &right,
        )?;
        Ok(())
    }

    /// Overwrites workspace-owned device output; validation leaves it unchanged.
    /// A new member count uploads one zeroed output buffer (#740); that
    /// buffer already has zero inactive regions. A warm call zeroes those
    /// regions in place and transfers no payload.
    pub fn execute<'a>(
        &self,
        lhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        rhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        workspace: &'a mut ContractWorkspace<R, D, CudaStorage<D>>,
    ) -> Result<&'a StackedTensorMap<R, D, CudaStorage<D>>, Error> {
        let members = self.check(lhs, rhs, workspace)?;
        let total = self.total_len(self.member_len, members)?;
        let fresh = workspace
            .output
            .as_ref()
            .is_none_or(|output| output.members != members);
        let runtime = self.runtime.clone();
        let mut lease = runtime.lease_cuda()?;
        self.prepare_zero_regions(workspace, &mut lease, members)?;
        let mut output = match workspace
            .output
            .take()
            .filter(|output| output.members == members)
        {
            Some(output) => output,
            None => StackedTensorMap {
                runtime: self.runtime.clone(),
                space: self.space.clone(),
                signature: self.output_signature.clone(),
                storage: CudaStorage::upload_members(
                    &lease,
                    zeroed_payload(total),
                    self.member_len,
                    members,
                )?,
                members,
                member_len: self.member_len,
                _payload: PhantomData,
            },
        };
        let result = self.run_cuda(workspace, &mut lease, lhs, rhs, &mut output.storage, !fresh);
        let output = workspace.output.insert(output);
        result.map(|()| &*output)
    }

    /// Overwrites a checked caller device destination. After the zero
    /// template has been reserved for this member count, replay transfers no
    /// payload and zeroes each inactive region across all members.
    pub fn execute_into(
        &self,
        lhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        rhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        dst: &mut StackedTensorMap<R, D, CudaStorage<D>>,
        workspace: &mut ContractWorkspace<R, D, CudaStorage<D>>,
    ) -> Result<(), Error> {
        let members = self.check(lhs, rhs, workspace)?;
        if let Some(field) = self.output_signature.first_mismatch(&dst.signature) {
            return Err(Error::BatchSignatureMismatch {
                member: None,
                field,
            });
        }
        if dst.members != members
            || dst.storage.len() != self.total_len(self.member_len, members)?
        {
            return Err(Error::InvalidArgument(
                "destination count or payload length differs from operands".into(),
            ));
        }
        let runtime = self.runtime.clone();
        let mut lease = runtime.lease_cuda()?;
        self.prepare_zero_regions(workspace, &mut lease, members)?;
        self.run_cuda(workspace, &mut lease, lhs, rhs, &mut dst.storage, true)
    }
}

impl<R, D, S> Drop for ContractWorkspace<R, D, S> {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if self.device.reserved_plan_entries > 0 {
            if let Some(mut lease) = self.runtime.lease_cuda_for_maintenance() {
                lease.release_plan_entries(self.device.reserved_plan_entries);
            }
        }
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
        classes: Vec<(f64, usize)>,
        fail_negative: bool,
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
            assert_eq!(beta, 0.0);
            self.calls += 1;
            self.jobs += jobs.len();
            self.classes.push((alpha, jobs.len()));
            if self.fail_negative && alpha < 0.0 {
                return Err(OperationError::InvalidArgument {
                    message: "recording negative-class backend failure",
                });
            }
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
                assert_eq!(count.classes, [(1.0, jobs_per_member * members)]);
            }
        }
    }

    #[test]
    fn signed_direct_submits_one_batch_per_present_sign_class() {
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
        let odd_only = GradedSpace::try_new(
            Arc::new(FermionParityFusionRule.product(U1FusionRule)),
            [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
        )
        .unwrap();
        for swapped in [false, true] {
            check_signed_classes(&runtime, &u1, 1, 2, swapped);
            check_signed_classes(&runtime, &su2, 2, 1, swapped);
            check_signed_classes(&runtime, &odd_only, 0, 1, swapped);
        }
    }

    fn check_signed_classes<R>(
        runtime: &Runtime,
        v: &GradedSpace<R>,
        positive: usize,
        negative: usize,
        swapped_expected: bool,
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    {
        let dual = v.try_dual().unwrap();
        let a = TensorMap::<_, f64>::from_subblock_fn(runtime, [v], [&dual], |_, _| 1.0).unwrap();
        let b = TensorMap::<_, f64>::from_subblock_fn(runtime, [&dual], [v], |_, _| 1.0).unwrap();
        let (lhs, rhs) = if swapped_expected { (b, a) } else { (a, b) };
        let left = StackedTensorMap::pack(&[&lhs]).unwrap();
        let right = StackedTensorMap::pack(&[&rhs]).unwrap();
        let lhs_axes = [usize::from(!swapped_expected)];
        let rhs_axes = [usize::from(swapped_expected)];
        let output = if swapped_expected { [1, 0] } else { [0, 1] };
        let spec = super::super::super::ContractSpec {
            lhs: &lhs_axes,
            rhs: &rhs_axes,
            codomain: &output[..1],
            domain: &output[1..],
        };
        let plan = ContractPlan::new(&left, &right, &spec).unwrap();
        assert!(plan.copy_c.is_none());
        assert!(!plan.resolution.is_dynamic_tree());
        assert_eq!(
            plan.resolution.admits_stacked_direct_host_replay().is_ok(),
            negative == 0
        );
        assert!(plan
            .resolution
            .admits_stacked_signed_direct_host_replay()
            .unwrap());
        for members in [1, 2, 17] {
            let (replay, swapped) = plan
                .resolution
                .stacked_signed_direct_host_replay(members)
                .unwrap()
                .unwrap();
            assert_eq!(swapped, swapped_expected);
            let [dst_len, lhs_len, rhs_len] = replay.member_lens();
            let mut dst = vec![f64::NAN; dst_len * members];
            let lhs = vec![1.0; lhs_len * members];
            let rhs = vec![1.0; rhs_len * members];
            let mut count = Count::default();
            if negative > 0 && members == 1 {
                assert!(replay
                    .execute_host(
                        &mut StridedHostKernelAdapter::default(),
                        &mut count,
                        &mut StackedStorageViewMut::new::<f64>(&mut dst, dst_len, members, dst_len)
                            .unwrap(),
                        &StackedStorageView::new::<f64>(&lhs, lhs_len, members, lhs_len).unwrap(),
                        &StackedStorageView::new::<f64>(&rhs, rhs_len, members, rhs_len).unwrap(),
                        true,
                    )
                    .is_err());
                assert!(dst.iter().all(|value| value.is_nan()));
                assert_eq!(count.calls, 0);
            }
            replay
                .execute_signed_host(
                    &mut StridedHostKernelAdapter::default(),
                    &mut count,
                    &mut StackedStorageViewMut::new::<f64>(&mut dst, dst_len, members, dst_len)
                        .unwrap(),
                    &StackedStorageView::new::<f64>(&lhs, lhs_len, members, lhs_len).unwrap(),
                    &StackedStorageView::new::<f64>(&rhs, rhs_len, members, rhs_len).unwrap(),
                    true,
                )
                .unwrap();
            let expected: Vec<_> = [(1.0, positive), (-1.0, negative)]
                .into_iter()
                .filter(|(_, jobs)| *jobs > 0)
                .map(|(alpha, jobs)| (alpha, jobs * members))
                .collect();
            assert_eq!(count.classes, expected);
            assert_eq!(
                count.calls,
                usize::from(positive > 0) + usize::from(negative > 0)
            );
            assert_eq!(count.jobs, (positive + negative) * members);
            if negative > 0 && members == 1 {
                count.fail_negative = true;
                let error = replay
                    .execute_signed_host(
                        &mut StridedHostKernelAdapter::default(),
                        &mut count,
                        &mut StackedStorageViewMut::new::<f64>(&mut dst, dst_len, members, dst_len)
                            .unwrap(),
                        &StackedStorageView::new::<f64>(&lhs, lhs_len, members, lhs_len).unwrap(),
                        &StackedStorageView::new::<f64>(&rhs, rhs_len, members, rhs_len).unwrap(),
                        true,
                    )
                    .unwrap_err();
                assert!(matches!(
                    error,
                    OperationError::InvalidArgument {
                        message: "recording negative-class backend failure"
                    }
                ));
            }
        }
    }
}
