//! Restricted binding for one ordinary contraction over owned-dense stacks.

use super::*;
use tenet_tensors::{DynamicTreeMembersWorkspace, OutputAxisOrder, StorageContractResolution};

#[path = "contract_batch_copy_c.rs"]
mod copy_c;
use copy_c::CopyCWorkspace;

// Only the two owned-dense stack payloads have an execution implementation.
trait ContractBatchStorage<D>: TensorStorage<D> {}
impl<D> ContractBatchStorage<D> for Vec<D> {}
#[cfg(feature = "cuda")]
impl<D: CudaPayload> ContractBatchStorage<D> for CudaStorage<D> {}

/// Immutable contraction structure for owned-dense Host or CUDA stacks.
///
/// The route is the contraction planner's (`plan_contract`), the one the
/// device [`TensorMap::contract`] replays. This binding admits Core, CopyC
/// and owned-source transformed-tree routes whose core is fully direct with
/// exact +1/-1 coefficients, checked once at construction so the admitted
/// set does not depend on the member count. CUDA admits CopyC when every
/// output move is an unconjugated nonzero Single task, and transformed-tree
/// routes whose source and output moves are all such tasks over a unit-alpha
/// direct core. Direct
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
    member_len: usize,
    _payload: PhantomData<(D, S)>,
}

/// Caller-owned output and execution resources for a [`ContractPlan`].
pub struct ContractWorkspace<R, D, S = Vec<D>> {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    runtime: Runtime,
    binding: Arc<StorageContractResolution<f64>>,
    output: OutputSlot<StackedTensorMap<R, D, S>>,
    members: DynamicTreeMembersWorkspace<D>,
    replay: Option<(StackedDirectReplay, bool)>,
    copy_c: Option<CopyCWorkspace<D>>,
    /// The CopyC temporary stack and its member count.
    #[cfg(feature = "cuda")]
    copy_c_temporary: Option<(S, usize)>,
    #[cfg(feature = "cuda")]
    dynamic: tenet_tensors::CudaDynamicTreeMembersWorkspace<S>,
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
    /// CUDA admits exact-sign direct Core/SwappedCore, nonzero Single CopyC,
    /// and transformed-tree routes whose transforms are nonzero Single moves.
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
        tenet_tensors::reject_non_symmetric_contraction(lhs.space.provider().braiding_style())?;
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
        let mut lease = lhs.runtime.lease_context()?;
        let lane = lease.context().multiplicity_free_lane::<D>()?;
        let resolution = lane.plan_contract::<tenet_tensors::DirectCoreExecutor, _>(
            &space,
            FusionOperand::direct(lhs.space.space()),
            FusionOperand::direct(rhs.space.space()),
            spec.lhs,
            spec.rhs,
            &output_axes,
        )?;
        // One admission for every route and member count: a plan accepted at
        // B = 1 replays at any B.
        resolution
            .core_plan()
            .require_identity_signed_direct_replay()?;
        #[cfg(feature = "cuda")]
        let direct_core = resolution.direct_core();
        #[cfg(feature = "cuda")]
        if matches!(placement, Placement::Cuda(_)) {
            if let Some(copy) = resolution.copy_c() {
                tenet_operations::cuda_transform::CudaSingleMemberRegions::admit(copy.transform())?;
            }
        }
        let member_len = space.space().required_len()?;
        #[cfg(feature = "cuda")]
        let device_plan = match (placement, direct_core) {
            // The exact-sign contract was required above.
            (Placement::Cuda(_), Some((core, swapped))) => {
                resolution.admit_cuda_inactive_regions()?;
                Some((Arc::clone(core), swapped))
            }
            (Placement::Cuda(_), None) => {
                resolution.admit_cuda_dynamic_tree_members()?;
                None
            }
            _ => None,
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
            member_len,
            _payload: PhantomData,
        })
    }

    fn copy_c(&self) -> Option<&tenet_tensors::CopyCRoute<f64>> {
        self.resolution.copy_c()
    }

    /// Signature required of every output member.
    pub fn output_signature(&self) -> &StructureSignature {
        &self.output_signature
    }

    fn check_workspace(&self, workspace: &ContractWorkspace<R, D, S>) -> Result<(), Error> {
        if Arc::ptr_eq(&self.resolution, &workspace.binding) {
            Ok(())
        } else {
            Err(Error::InvalidArgument(
                "contract workspace belongs to another plan".into(),
            ))
        }
    }

    /// Checks both operand stacks and returns their common member count.
    fn check(
        &self,
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
    ) -> Result<usize, Error> {
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

    fn check_destination(
        &self,
        dst: &StackedTensorMap<R, D, S>,
        members: usize,
    ) -> Result<(), Error> {
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
        Ok(())
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
    pub fn workspace(&self) -> Result<ContractWorkspace<R, D>, Error> {
        Ok(ContractWorkspace {
            runtime: self.runtime.clone(),
            binding: Arc::clone(&self.resolution),
            output: OutputSlot::default(),
            members: DynamicTreeMembersWorkspace::default(),
            replay: None,
            copy_c: self.copy_c().map(|_| CopyCWorkspace::default()),
            #[cfg(feature = "cuda")]
            copy_c_temporary: None,
            #[cfg(feature = "cuda")]
            dynamic: Default::default(),
            #[cfg(feature = "cuda")]
            device: DeviceComposeState::default(),
        })
    }

    fn run(
        &self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        dst: &mut Vec<D>,
        members: usize,
        workspace: &mut ContractWorkspace<R, D>,
    ) -> Result<(), Error> {
        if let Some(copy_c) = self.copy_c() {
            return copy_c::run(copy_c, self, lhs, rhs, dst, members, workspace);
        }
        if !self.resolution.is_dynamic_tree() {
            if workspace
                .replay
                .as_ref()
                .is_none_or(|(replay, _)| replay.members() != members)
            {
                workspace.replay = self
                    .resolution
                    .direct_core()
                    .map(|(core, swapped)| {
                        StackedDirectReplay::new_signed(Arc::clone(core), members)
                            .map(|replay| (replay, swapped))
                    })
                    .transpose()?;
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

    /// Overwrites the workspace-owned output and borrows it until the next
    /// call. Any error follows the batched module's failure rule.
    pub fn execute<'a>(
        &self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        workspace: &'a mut ContractWorkspace<R, D>,
    ) -> Result<&'a StackedTensorMap<R, D>, Error> {
        self.check_workspace(workspace)?;
        let buffers = workspace.output.take();
        let checked = self
            .check(lhs, rhs)
            .and_then(|members| Ok((members, self.total_len(self.member_len, members)?)));
        let (members, total) = match checked {
            Ok(checked) => checked,
            Err(error) => return Err(workspace.output.fail(buffers, error)),
        };
        let mut output = buffers
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
        workspace.output.settle(output, result)
    }

    /// Overwrites a checked caller destination. Validation and capability
    /// errors leave it unchanged; after a backend error its payload may be
    /// partially overwritten while its metadata stays valid. Any error
    /// follows the batched module's failure rule.
    pub fn execute_into(
        &self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        dst: &mut StackedTensorMap<R, D>,
        workspace: &mut ContractWorkspace<R, D>,
    ) -> Result<(), Error> {
        self.check_workspace(workspace)?;
        let result = self.check(lhs, rhs).and_then(|members| {
            self.check_destination(dst, members)?;
            self.run(lhs, rhs, &mut dst.storage, members, workspace)
        });
        result.map_err(|error| workspace.output.hide(error))
    }
}

impl<R, D, S> ContractWorkspace<R, D, S> {
    /// Moves the output of the last successful call to the caller; `None`
    /// after a failed call (the batched module's failure rule).
    pub fn take_output(&mut self) -> Option<StackedTensorMap<R, D, S>> {
        self.output.take_output()
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
            + self
                .output
                .buffers()
                .map(|output| {
                    output
                        .storage
                        .capacity()
                        .saturating_mul(std::mem::size_of::<D>())
                })
                .sum::<usize>()
    }
}

#[cfg(feature = "cuda")]
impl<R, D: CudaPayload> ContractWorkspace<R, D, CudaStorage<D>> {
    /// Retained device payload and Host layout scratch, excluding Runtime resources.
    pub fn retained_bytes(&self) -> usize {
        self.retained_scratch_bytes()
            + self.dynamic.retained_bytes()
            + self.copy_c_temporary.as_ref().map_or(0, |(temporary, _)| {
                TensorStorage::len(temporary).saturating_mul(std::mem::size_of::<D>())
            })
            + self
                .output
                .buffers()
                .map(|output| {
                    output
                        .members
                        .saturating_mul(output.member_len)
                        .saturating_mul(std::mem::size_of::<D>())
                })
                .sum::<usize>()
            + self.device.zero_regions.capacity() * std::mem::size_of::<tenet_dense::CudaRegion>()
            + self
                .device
                .copy_regions
                .as_ref()
                .map_or(0, |regions| regions.retained_bytes())
            + self
                .device
                .copy_coefficients
                .as_ref()
                .map_or(0, |coefficients| {
                    coefficients.len() * std::mem::size_of::<D>()
                })
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
            output: OutputSlot::default(),
            members: DynamicTreeMembersWorkspace::default(),
            replay: None,
            copy_c: None,
            copy_c_temporary: None,
            dynamic: Default::default(),
            device: DeviceComposeState::default(),
        };
        let entries = if self.resolution.is_dynamic_tree() {
            self.resolution.admit_cuda_dynamic_tree_members()?
        } else {
            self.direct_plan_entries()?
        };
        workspace.device.reserved_plan_entries = self
            .runtime
            .lease_cuda()?
            .reserve_plan_entries(entries)
            .map_err(tenet_operations::OperationError::Dense)?;
        Ok(workspace)
    }

    fn direct_plan_entries(&self) -> Result<usize, Error> {
        let core = self.core().0.cuda_direct_plan_entries();
        let copy_fills = self.copy_c().map_or(Ok(0), |copy| {
            tenet_operations::cuda_transform::CudaSingleMemberRegions::admit(copy.transform())
        })?;
        Ok(core + copy_fills)
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
        workspace.device.copy_regions = None;
        let core_member_len = if let Some(copy) = self.copy_c() {
            copy.temporary_len()
        } else {
            self.member_len
        };
        self.total_len(core_member_len, members)?;
        let zeros = tenet_operations::cuda_transform::CudaMemberZeroRegions::prepare(
            self.core().0.inactive_destination_regions(),
            core_member_len,
            members,
        )?;
        zeros.reserve_templates::<D>(ctx, members)?;
        let regions = zeros.into_regions();
        workspace.device.zero_regions = regions;
        workspace.device.members = members;
        Ok(())
    }

    fn prepare_copy_regions(
        &self,
        workspace: &mut ContractWorkspace<R, D, CudaStorage<D>>,
        ctx: &mut tenet_dense::CudaDenseContext,
        members: usize,
    ) -> Result<(), Error> {
        let Some(copy) = self.copy_c() else {
            return Ok(());
        };
        if workspace.device.copy_regions.is_some() {
            return Ok(());
        }
        let temporary_len = copy.temporary_len();
        let regions = tenet_operations::cuda_transform::CudaSingleMemberRegions::prepare(
            copy.transform(),
            self.space.space().structure(),
            copy.temporary_structure(),
            self.member_len,
            temporary_len,
            self.total_len(self.member_len, members)?,
            self.total_len(temporary_len, members)?,
            members,
        )?;
        ctx.reserve_zero_template::<D>(regions.max_zero_len().checked_mul(members).ok_or_else(
            || Error::InvalidArgument("CopyC zero template length overflows".into()),
        )?)
        .map_err(tenet_operations::OperationError::Dense)?;
        if !regions.zeros().is_empty() {
            ctx.reserve_ones_template::<D>(1)
                .map_err(tenet_operations::OperationError::Dense)?;
        }
        if regions.has_scaled_moves() && workspace.device.copy_coefficients.is_none() {
            workspace.device.copy_coefficients =
                Some(CudaStorage::<D>::upload_owned(ctx, regions.scaled_coefficients::<D>())?.0);
        }
        workspace.device.copy_regions = Some(regions);
        Ok(())
    }

    fn run_cuda(
        &self,
        workspace: &mut ContractWorkspace<R, D, CudaStorage<D>>,
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
        let temporary_len = self.copy_c().map(|copy| copy.temporary_len());
        let mut temporary = if let Some(temporary_len) = temporary_len {
            let existing = workspace
                .copy_c_temporary
                .take()
                .filter(|(_, members)| *members == lhs.members);
            Some(match existing {
                Some((storage, _)) => (false, storage),
                None => (
                    true,
                    CudaStorage::upload_members(
                        ctx,
                        zeroed_payload(self.total_len(temporary_len, lhs.members)?),
                        temporary_len,
                        lhs.members,
                    )?,
                ),
            })
        } else {
            None
        };
        let device = &workspace.device;
        // The temporary goes back to the workspace on every exit, so a failed
        // call never makes the next one upload it again.
        let result = (|| {
            let (core_storage, core_len, zero_core) = match (&mut temporary, temporary_len) {
                (Some((fresh, storage)), Some(len)) => (storage, len, !*fresh),
                _ => (&mut *dst, self.member_len, zero_inactive),
            };
            let mut core_destination =
                StackedStorageViewMut::new::<D>(core_storage, core_len, lhs.members, core_len)?;
            if zero_core {
                for region in &device.zero_regions {
                    tenet_dense::cuda_region_zero::<D>(
                        ctx,
                        &mut core_destination.storage_mut().0,
                        region,
                    )
                    .map_err(tenet_operations::OperationError::Dense)?;
                }
            }
            plan.execute_direct_on_storage_prezeroed(
                &mut tenet_operations::cuda::CudaStackedStorageGemm::new(ctx),
                &mut core_destination,
                &left,
                &right,
            )?;
            if let Some((_, storage)) = &temporary {
                let regions = device.copy_regions.as_ref().ok_or_else(|| {
                    Error::InvalidArgument("CopyC device regions are unprepared".into())
                })?;
                regions.execute_overwrite(ctx, storage, device.copy_coefficients.as_ref(), dst)?;
            }
            Ok(())
        })();
        if let Some((_, storage)) = temporary {
            workspace.copy_c_temporary = Some((storage, lhs.members));
        }
        result
    }

    /// Overwrites workspace-owned device output; any error follows the
    /// batched module's failure rule. A new member count uploads one zeroed
    /// output buffer (#740); that buffer already has zero inactive regions.
    /// A reused buffer, including one kept after a failed call, has its
    /// inactive regions zeroed in place, and transfers no payload.
    pub fn execute<'a>(
        &self,
        lhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        rhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        workspace: &'a mut ContractWorkspace<R, D, CudaStorage<D>>,
    ) -> Result<&'a StackedTensorMap<R, D, CudaStorage<D>>, Error> {
        self.check_workspace(workspace)?;
        let buffers = workspace.output.take();
        let runtime = self.runtime.clone();
        let dynamic = self.resolution.is_dynamic_tree();
        let prepared = self.check(lhs, rhs).and_then(|members| {
            let total = self.total_len(self.member_len, members)?;
            let mut lease = runtime.lease_cuda()?;
            if !dynamic {
                self.prepare_zero_regions(workspace, &mut lease, members)?;
                self.prepare_copy_regions(workspace, &mut lease, members)?;
            }
            Ok((members, total, lease))
        });
        let (members, total, mut lease) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return Err(workspace.output.fail(buffers, error)),
        };
        let reused = buffers.filter(|output| output.members == members);
        let fresh = reused.is_none();
        let mut output = match reused {
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
        let result = if dynamic {
            self.run_dynamic(workspace, &mut lease, lhs, rhs, &mut output.storage, fresh)
        } else {
            self.run_cuda(workspace, &mut lease, lhs, rhs, &mut output.storage, !fresh)
        };
        drop(lease);
        workspace.output.settle(output, result)
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
        self.check_workspace(workspace)?;
        let result = self.check(lhs, rhs).and_then(|members| {
            self.check_destination(dst, members)?;
            let runtime = self.runtime.clone();
            let mut lease = runtime.lease_cuda()?;
            if self.resolution.is_dynamic_tree() {
                return self.run_dynamic(workspace, &mut lease, lhs, rhs, &mut dst.storage, false);
            }
            self.prepare_zero_regions(workspace, &mut lease, members)?;
            self.prepare_copy_regions(workspace, &mut lease, members)?;
            self.run_cuda(workspace, &mut lease, lhs, rhs, &mut dst.storage, true)
        });
        result.map_err(|error| workspace.output.hide(error))
    }

    fn run_dynamic(
        &self,
        workspace: &mut ContractWorkspace<R, D, CudaStorage<D>>,
        ctx: &mut tenet_dense::CudaDenseContext,
        lhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        rhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        dst: &mut CudaStorage<D>,
        dst_zeroed: bool,
    ) -> Result<(), Error> {
        workspace.dynamic.execute(
            ctx,
            &self.resolution,
            self.space.space().structure(),
            dst,
            &lhs.storage,
            &rhs.storage,
            lhs.members,
            dst_zeroed,
        )?;
        Ok(())
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
mod failure_tests {
    use super::*;
    use crate::sector::{U1FusionRule, U1Irrep};
    use crate::typed::GradedSpace;

    #[test]
    fn a_failure_after_the_output_is_taken_keeps_it_unobservable() {
        // What: on the direct Core and the CopyC route, an error inside the
        // replay, after the output buffer is taken (forced by an operand
        // whose layout and payload are consistently one entry short, which
        // only a corrupted stack carries), leaves no observable output; the
        // buffer is kept and the next call recovers.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let rule = Arc::new(U1FusionRule);
        let v = GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 1),
                (U1Irrep::new(1), 3),
            ],
        )
        .unwrap();
        let w = GradedSpace::try_new(rule, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w], 5).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&v], 6).unwrap();
        let lhs = StackedTensorMap::pack(&[&a, &a]).unwrap();
        let rhs = StackedTensorMap::pack(&[&b, &b]).unwrap();
        let mut corrupt = StackedTensorMap::pack(&[&a, &a]).unwrap();
        corrupt.member_len -= 1;
        corrupt
            .storage
            .truncate(corrupt.member_len * corrupt.members);
        for (codomain, copy_c) in [([0, 1], false), ([1, 0], true)] {
            let spec = super::super::super::ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &codomain,
                domain: &[2],
            };
            let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
            assert_eq!(plan.copy_c().is_some(), copy_c);
            let mut workspace = plan.workspace().unwrap();
            let expected = plan
                .execute(&lhs, &rhs, &mut workspace)
                .unwrap()
                .storage
                .clone();
            let bytes = workspace.retained_bytes();

            // Only the replay compares the operand layout with its own.
            let Err(crate::error::Error::Operation(error)) =
                plan.execute(&corrupt, &rhs, &mut workspace).map(|_| ())
            else {
                panic!("the corrupted stack must fail as an operation error");
            };
            assert!(
                matches!(
                    *error,
                    tenet_tensors::OperationError::InvalidArgument {
                        message: "stacked operand does not match the replay's members or layout"
                    }
                ),
                "{error:?}"
            );
            assert!(
                workspace.output.spare.is_some(),
                "the error is after the take"
            );
            assert!(workspace.take_output().is_none());
            assert_eq!(workspace.retained_bytes(), bytes);
            assert_eq!(
                plan.execute(&lhs, &rhs, &mut workspace).unwrap().storage,
                expected
            );
            assert!(workspace.take_output().is_some());
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_copy_c_failure_keeps_the_temporary_and_the_output() {
        // What: on the CUDA CopyC route, a core error after the temporary
        // and the output are taken (forced as on Host by a consistently
        // short operand) returns both to the workspace, so the next call
        // uploads neither and recovers.
        let runtime = Runtime::builder().cuda(0).build().unwrap();
        let rule = Arc::new(U1FusionRule);
        let v = GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 1),
                (U1Irrep::new(1), 3),
            ],
        )
        .unwrap();
        let w = GradedSpace::try_new(rule, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w], 5).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&v], 6).unwrap();
        let lhs = StackedTensorMap::pack(&[&a, &a])
            .unwrap()
            .to_cuda()
            .unwrap();
        let rhs = StackedTensorMap::pack(&[&b, &b])
            .unwrap()
            .to_cuda()
            .unwrap();
        let mut host_corrupt = StackedTensorMap::pack(&[&a, &a]).unwrap();
        host_corrupt.member_len -= 1;
        host_corrupt
            .storage
            .truncate(host_corrupt.member_len * host_corrupt.members);
        let mut corrupt = host_corrupt.to_cuda().unwrap();
        corrupt.member_len = host_corrupt.member_len;
        let spec = super::super::super::ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[1, 0],
            domain: &[2],
        };
        let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
        assert!(plan.copy_c().is_some());
        let mut workspace = plan.workspace().unwrap();
        let expected = plan
            .execute(&lhs, &rhs, &mut workspace)
            .unwrap()
            .to_host()
            .unwrap()
            .storage;
        let bytes = workspace.retained_bytes();

        let error = plan
            .execute(&corrupt, &rhs, &mut workspace)
            .map(|_| ())
            .unwrap_err();
        eprintln!("cuda CopyC core error: {error:?}");
        assert!(
            workspace.copy_c_temporary.is_some(),
            "the temporary is restored"
        );
        assert!(workspace.output.spare.is_some(), "the output is kept");
        assert!(workspace.take_output().is_none());
        assert_eq!(workspace.retained_bytes(), bytes);
        let output = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        assert_eq!(output.to_host().unwrap().storage, expected);
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
            assert_eq!(plan.copy_c().is_some(), copy_c, "{name}");
            assert!(!plan.resolution.is_dynamic_tree(), "{name}");
            let mut jobs_per_member = 0;
            for members in [1, 2, 17] {
                let (core, swapped) = plan.resolution.direct_core().expect("unit direct core");
                let replay = StackedDirectReplay::new(Arc::clone(core), members).unwrap();
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
        let u1_full = GradedSpace::try_new(
            Arc::new(FermionParityFusionRule.product(U1FusionRule)),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(1)), 1),
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
            check_signed_classes(&runtime, &u1_full, 1, 2, swapped);
            check_signed_classes(&runtime, &su2, 2, 1, swapped);
            check_signed_classes(&runtime, &odd_only, 0, 1, swapped);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn signed_direct_can_leave_inactive_destination_blocks() {
        let runtime = Runtime::builder().build().unwrap();
        let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
        let v = GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
            ],
        )
        .unwrap();
        let w = GradedSpace::try_new(
            rule,
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            ],
        )
        .unwrap();
        let wd = w.try_dual().unwrap();
        let lhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&v], [&wd], |_, _| 1.0).unwrap();
        let rhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&wd], [&v], |_, _| 1.0).unwrap();
        let left = StackedTensorMap::pack(&[&lhs]).unwrap();
        let right = StackedTensorMap::pack(&[&rhs]).unwrap();
        let spec = super::super::super::ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        };
        let plan = ContractPlan::new(&left, &right, &spec).unwrap();
        let core = plan.resolution.direct_core().unwrap().0;
        core.require_identity_signed_direct_replay().unwrap();
        assert!(core.require_identity_direct_replay().is_err());
        assert!(!core.inactive_destination_regions().is_empty());
        assert!(core.distinct_direct_gemm_shapes() > 0);
    }

    #[cfg(feature = "cuda")]
    fn cuda_dynamic_admission<R>(
        runtime: &Runtime,
        codomains: [&[&GradedSpace<R>]; 2],
        domains: [&[&GradedSpace<R>]; 2],
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output: &[usize],
    ) -> (bool, Result<usize, OperationError>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    {
        let [a, b] = [0, 1].map(|i| {
            TensorMap::<_, f64>::from_subblock_fn(
                runtime,
                codomains[i].iter().copied(),
                domains[i].iter().copied(),
                |_, _| 1.0,
            )
            .unwrap()
        });
        let split = codomains[0].len() + domains[0].len() - lhs_axes.len();
        let spec = super::super::super::ContractSpec {
            lhs: lhs_axes,
            rhs: rhs_axes,
            codomain: &output[..split],
            domain: &output[split..],
        };
        let plan = ContractPlan::new(
            &StackedTensorMap::pack(&[&a]).unwrap(),
            &StackedTensorMap::pack(&[&b]).unwrap(),
            &spec,
        )
        .unwrap();
        (
            plan.resolution.is_dynamic_tree(),
            plan.resolution.admit_cuda_dynamic_tree_members(),
        )
    }

    /// Device-free pin of the public CUDA fixtures: U(1) and fermionic
    /// transformed-tree routes have Single-only transforms, SU(2) recoupling
    /// has a Multi transform and stays a typed capability error.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dynamic_tree_admission_is_single_only() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let u1 = |charges: &[(i32, usize)]| {
            GradedSpace::try_new(
                Arc::new(U1FusionRule),
                charges
                    .iter()
                    .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
            )
            .unwrap()
        };
        let v = u1(&[(0, 1), (1, 2), (2, 1)]);
        let w = u1(&[(0, 2), (1, 1)]);
        let x = u1(&[(-1, 1), (0, 2), (1, 2)]);
        for (name, codomains, domains, lhs_axes, rhs_axes, output) in [
            (
                "U(1) transformed lhs, inactive block",
                [&[&v][..], &[&w]],
                [&[&w, &v][..], &[&v]],
                &[1][..],
                &[0][..],
                &[0, 1, 2][..],
            ),
            (
                "U(1) reordered whole-side",
                [&[&v, &v], &[&x, &x]],
                [&[&x, &x], &[&v]],
                &[3, 2],
                &[0, 1],
                &[1, 0, 2],
            ),
        ] {
            let (dynamic, admitted) =
                cuda_dynamic_admission(&runtime, codomains, domains, lhs_axes, rhs_axes, output);
            assert!(dynamic, "{name}");
            assert!(admitted.unwrap() > 0, "{name}");
        }
        let f = GradedSpace::try_new(
            Arc::new(FermionParityFusionRule.product(U1FusionRule)),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(1)), 1),
            ],
        )
        .unwrap();
        let fd = f.try_dual().unwrap();
        // contract_cases::fermionic_twist_roles "A": the copied A carries the twist.
        let (dynamic, admitted) = cuda_dynamic_admission(
            &runtime,
            [&[&f, &f], &[&fd]],
            [&[&f], &[&f, &f]],
            &[1],
            &[0],
            &[0, 1, 2, 3],
        );
        assert!(dynamic);
        admitted.unwrap();
        let s = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
        )
        .unwrap();
        let (dynamic, admitted) = cuda_dynamic_admission(
            &runtime,
            [&[&s, &s], &[&s]],
            [&[&s], &[&s, &s]],
            &[0],
            &[2],
            &[3, 0, 2, 1],
        );
        assert!(dynamic);
        assert!(matches!(
            admitted,
            Err(OperationError::UnsupportedTensorContractScope {
                message: "CUDA member transform requires unconjugated nonzero Single tasks"
            })
        ));
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
        assert!(plan.copy_c().is_none());
        assert!(!plan.resolution.is_dynamic_tree());
        let (core, swapped) = plan.resolution.direct_core().unwrap();
        assert_eq!(core.require_identity_direct_replay().is_ok(), negative == 0);
        core.require_identity_signed_direct_replay().unwrap();
        for members in [1, 2, 17] {
            let replay = StackedDirectReplay::new_signed(Arc::clone(core), members).unwrap();
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
