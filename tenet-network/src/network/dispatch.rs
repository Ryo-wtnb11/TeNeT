use super::*;

mod host_mode_sealed {
    pub trait Sealed {}

    impl Sealed for tenet::sector::MultiplicityFreeAdmissionMode {}
    impl Sealed for tenet::sector::CheckedGenericAdmissionMode {}
}

/// Internal network policy selected by the provider's admission mode and the
/// operand payload storage.
///
/// `S` defaults to Host `Vec<D>`, so `HostNetworkModeDispatch<R, D>` keeps
/// naming the Host policy. A second instantiation per storage carries the
/// device policy: which primitives exist there, whether destinations may be
/// reused, and how an idle workspace is parked.
#[doc(hidden)]
pub trait HostNetworkModeDispatch<R, D, S = Vec<D>>:
    host_mode_sealed::Sealed
    + TypedTensorModeDispatch<R>
    + TypedSpaceModeDispatch<R>
    + TypedAdjointSpace<R>
    + TypedTensorContractDispatch<R, D>
    + TypedTensorRootDispatch<R>
    + TypedTensorTransformDispatch<R, D>
where
    R: TypedSectorAdmission<Mode = Self>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    const REUSE_DESTINATIONS: bool;

    /// The provider's braiding style, which decides whether a network that
    /// contracts is admitted at all.
    fn braiding_style(provider: &R) -> tenet::sector::BraidingStyleKind;

    fn leg_dims<B: TensorStorage<D>>(
        tensor: &TensorMap<R, D, B>,
    ) -> Result<Vec<usize>, HostNetworkError<R>>;

    /// Lowers a `conj`-marked operand to its adjoint on this storage.
    fn adjoint_operand(
        tensor: &TensorMap<R, D, S>,
    ) -> Result<TensorMap<R, D, S>, HostNetworkError<R>>;

    fn contract_step(
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
        destination: &mut Option<TensorMap<R, D, S>>,
        spec: &ContractSpec<'_>,
    ) -> Result<StepOutput<TensorMap<R, D, S>>, HostNetworkError<R>>;

    /// Reorients the operand of a zero-step schedule (a relabel), the one
    /// result no contraction step writes in its output orientation.
    fn permute_final(
        tensor: &TensorMap<R, D, S>,
        codomain: &[usize],
        domain: &[usize],
    ) -> Result<TensorMap<R, D, S>, HostNetworkError<R>>;

    fn activate_parked(
        workspace: &mut NetworkExecutionWorkspace<R, D, S>,
        runtime: &Runtime,
        tensors: &[&TensorMap<R, D, S>],
        steps: &[CompiledStep],
    ) -> Result<(), HostNetworkError<R>>;

    fn park_workspace(workspace: &mut NetworkExecutionWorkspace<R, D, S>);
}

#[doc(hidden)]
pub enum StepOutput<T> {
    Returned(T),
    Overwritten,
}

impl<T> StepOutput<T> {
    pub(super) fn get<'a>(&'a self, destination: &'a Option<T>) -> &'a T {
        match self {
            Self::Returned(value) => value,
            Self::Overwritten => destination
                .as_ref()
                .expect("successful overwrite retains its destination"),
        }
    }

    pub(super) fn take(self, destination: &mut Option<T>) -> T {
        match self {
            Self::Returned(value) => value,
            Self::Overwritten => destination
                .take()
                .expect("successful overwrite retains its destination"),
        }
    }
}

impl<R, D> HostNetworkModeDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    const REUSE_DESTINATIONS: bool = true;

    fn braiding_style(provider: &R) -> tenet::sector::BraidingStyleKind {
        tenet::sector::FusionRule::braiding_style(provider)
    }

    fn leg_dims<B: TensorStorage<D>>(tensor: &TensorMap<R, D, B>) -> Result<Vec<usize>, Error> {
        tensor.leg_dims()
    }

    fn adjoint_operand(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.adjoint()
    }

    fn contract_step(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        destination: &mut Option<TensorMap<R, D>>,
        spec: &ContractSpec<'_>,
    ) -> Result<StepOutput<TensorMap<R, D>>, Error> {
        if let Some(destination) = destination {
            lhs.contract_into(rhs, spec, destination, D::from_real(1.0), D::from_real(0.0))?;
            Ok(StepOutput::Overwritten)
        } else {
            Ok(StepOutput::Returned(lhs.contract(rhs, spec)?))
        }
    }

    fn permute_final(
        tensor: &TensorMap<R, D>,
        codomain: &[usize],
        domain: &[usize],
    ) -> Result<TensorMap<R, D>, Error> {
        tensor.permute(codomain, domain)
    }

    fn activate_parked(
        workspace: &mut NetworkExecutionWorkspace<R, D>,
        runtime: &Runtime,
        tensors: &[&TensorMap<R, D>],
        steps: &[CompiledStep],
    ) -> Result<(), Error> {
        workspace.activate_parked(runtime, tensors, steps)
    }

    fn park_workspace(workspace: &mut NetworkExecutionWorkspace<R, D>) {
        workspace.park_runtime_owners();
    }
}

impl<R, D> HostNetworkModeDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    const REUSE_DESTINATIONS: bool = false;

    fn braiding_style(provider: &R) -> tenet::sector::BraidingStyleKind {
        CheckedGenericFusion::braiding_style(provider)
    }

    fn leg_dims<B: TensorStorage<D>>(
        tensor: &TensorMap<R, D, B>,
    ) -> Result<Vec<usize>, HostNetworkError<R>> {
        tensor
            .codomain()
            .into_iter()
            .chain(tensor.domain())
            .map(|space| space.dim().map(|dimension| dimension.round() as usize))
            .collect()
    }

    fn adjoint_operand(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, HostNetworkError<R>> {
        tensor.adjoint()
    }

    fn contract_step(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        _destination: &mut Option<TensorMap<R, D>>,
        spec: &ContractSpec<'_>,
    ) -> Result<StepOutput<TensorMap<R, D>>, HostNetworkError<R>> {
        Ok(StepOutput::Returned(lhs.contract(rhs, spec)?))
    }

    fn permute_final(
        tensor: &TensorMap<R, D>,
        codomain: &[usize],
        domain: &[usize],
    ) -> Result<TensorMap<R, D>, HostNetworkError<R>> {
        tensor.permute(codomain, domain)
    }

    fn activate_parked(
        _workspace: &mut NetworkExecutionWorkspace<R, D>,
        _runtime: &Runtime,
        _tensors: &[&TensorMap<R, D>],
        _steps: &[CompiledStep],
    ) -> Result<(), HostNetworkError<R>> {
        Ok(())
    }

    fn park_workspace(workspace: &mut NetworkExecutionWorkspace<R, D>) {
        for buffers in &mut workspace.intermediates {
            buffers.output = None;
            buffers.parked = None;
        }
    }
}

/// Device network policy: the Host step sequence on device tensors, with
/// retained device destinations.
///
/// Each arm is the Host arm with the device twin of the same typed operation:
/// general-axes `contract` / `contract_into` with `beta = 0` (the
/// Host-compiled DynamicTree or core route, fermionic twist included), each
/// writing the step's own output orientation. Every step but the last
/// overwrites a destination the workspace kept from the previous call (the
/// device overwrite needs no reset of it); the final schedule slot leaves the
/// workspace and therefore always allocates a fresh returning output. Slots, producers, input
/// snapshots and parking behave exactly as on Host, which is what lets one
/// pool serve both placements. Admission is decided before the first step by
/// `PlannedNetwork::validate_cuda_admission`.
#[cfg(feature = "cuda")]
impl<R, D> HostNetworkModeDispatch<R, D, CudaStorage<D>> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: CudaPayload,
{
    const REUSE_DESTINATIONS: bool = true;

    fn braiding_style(provider: &R) -> tenet::sector::BraidingStyleKind {
        tenet::sector::FusionRule::braiding_style(provider)
    }

    fn leg_dims<B: TensorStorage<D>>(tensor: &TensorMap<R, D, B>) -> Result<Vec<usize>, Error> {
        tensor.leg_dims()
    }

    fn adjoint_operand(
        tensor: &TensorMap<R, D, CudaStorage<D>>,
    ) -> Result<TensorMap<R, D, CudaStorage<D>>, Error> {
        tensor.adjoint()
    }

    fn contract_step(
        lhs: &TensorMap<R, D, CudaStorage<D>>,
        rhs: &TensorMap<R, D, CudaStorage<D>>,
        destination: &mut Option<TensorMap<R, D, CudaStorage<D>>>,
        spec: &ContractSpec<'_>,
    ) -> Result<StepOutput<TensorMap<R, D, CudaStorage<D>>>, Error> {
        #[cfg(test)]
        CUDA_NETWORK_CONTRACT_CALLS.with(|calls| calls.set(calls.get() + 1));
        if let Some(destination) = destination {
            lhs.contract_into(rhs, spec, destination, D::from_real(1.0), D::from_real(0.0))?;
            Ok(StepOutput::Overwritten)
        } else {
            Ok(StepOutput::Returned(lhs.contract(rhs, spec)?))
        }
    }

    fn permute_final(
        tensor: &TensorMap<R, D, CudaStorage<D>>,
        codomain: &[usize],
        domain: &[usize],
    ) -> Result<TensorMap<R, D, CudaStorage<D>>, Error> {
        tensor.permute(codomain, domain)
    }

    fn activate_parked(
        workspace: &mut NetworkExecutionWorkspace<R, D, CudaStorage<D>>,
        runtime: &Runtime,
        tensors: &[&TensorMap<R, D, CudaStorage<D>>],
        steps: &[CompiledStep],
    ) -> Result<(), Error> {
        workspace.activate_parked(runtime, tensors, steps)
    }

    fn park_workspace(workspace: &mut NetworkExecutionWorkspace<R, D, CudaStorage<D>>) {
        workspace.park_runtime_owners();
    }
}
