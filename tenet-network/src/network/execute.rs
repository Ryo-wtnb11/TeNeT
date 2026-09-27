use super::*;

/// A [`Network`] with a resolved contraction order for concrete operand
/// shapes. Inspect the order via [`Self::plan`], run it via
/// [`Self::execute`].
pub struct PlannedNetwork {
    pub(super) owner_token: u64,
    pub(super) plan: ContractionPlan,
    pub(super) conj: Vec<bool>,
    pub(super) input_codomain_ranks: Vec<usize>,
    pub(super) schedule: CompiledSchedule,
}

pub(super) struct CompiledSchedule {
    pub(super) slot_count: usize,
    pub(super) input_ranks: Vec<usize>,
    pub(super) contracted_input_pairs: Vec<InputLegPair>,
    pub(super) steps: Vec<CompiledStep>,
    pub(super) final_slot: usize,
    pub(super) final_permutation: Option<(Vec<usize>, Vec<usize>)>,
}

pub(super) type InputLegPair = ((usize, usize), (usize, usize));

#[doc(hidden)]
pub struct CompiledStep {
    pub(super) lhs_slot: usize,
    pub(super) rhs_slot: usize,
    pub(super) result_slot: usize,
    pub(super) lhs_contract_axes: Vec<usize>,
    pub(super) rhs_contract_axes: Vec<usize>,
    pub(super) result_permutation: Option<(Vec<usize>, Vec<usize>)>,
    pub(super) result_output_axes: Option<Vec<usize>>,
    pub(super) contract_output_axes: Vec<usize>,
    pub(super) authority_input_slot: usize,
}

/// Caller-owned replay state for one planned network at a time.
///
/// The stored payload destinations are private implementation details. Host MF
/// replay can reuse compatible intermediate buffers. Checked Generic replay
/// reuses the plan and workspace containers but admits new intermediate
/// tensors. The final tensor leaves the workspace in both modes.
///
/// `S` is the operand payload storage and defaults to Host `Vec<D>`. A device
/// storage uses the same ownership, parking, quarantine and budget model; what
/// may be reused is decided per `(mode, storage)` by the mode dispatch's
/// `REUSE_DESTINATIONS`.
pub struct NetworkExecutionWorkspace<R, D, S = Vec<D>>
where
    S: NetworkPayloadStorage<D>,
{
    pub(super) slots: Vec<Option<TensorMap<R, D, S>>>,
    pub(super) producers: Vec<Option<(usize, bool)>>,
    pub(super) intermediates: Vec<TypedIntermediateBuffers<R, D, S>>,
    pub(super) owner_token: Option<u64>,
    runtime: Option<RuntimeIdentity>,
    rule_identity: Option<RuleIdentity>,
    input_snapshot: Vec<TypedInputSnapshot>,
}

struct TypedInputSnapshot {
    spaces: Vec<SectorLeg>,
    reuse_class: NetworkReuseClass,
}

pub(super) struct TypedIntermediateBuffers<R, D, S = Vec<D>> {
    pub(super) contracted: Option<TensorMap<R, D, S>>,
    pub(super) oriented: Option<TensorMap<R, D, S>>,
    pub(super) parked_contracted: Option<RuntimeDetachedTensorMap<D, S>>,
    pub(super) parked_oriented: Option<RuntimeDetachedTensorMap<D, S>>,
}

pub(super) struct PayloadMeter {
    limit: usize,
    destination: (usize, usize),
    peak_workspace: usize,
    peak_total: usize,
    base: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub(super) enum PayloadMeterError {
    Limit { limit: usize, required: usize },
    ArithmeticOverflow,
}

impl PayloadMeter {
    pub(super) fn new<R, D: TensorScalar, S: NetworkPayloadStorage<D>>(
        limit: usize,
        destination: &TensorMap<R, D, S>,
    ) -> std::result::Result<Self, PayloadMeterError> {
        let destination = destination
            .network_owned_payload()
            .ok_or(PayloadMeterError::ArithmeticOverflow)?;
        Ok(Self {
            limit,
            destination,
            peak_workspace: 0,
            peak_total: 0,
            base: Vec::new(),
        })
    }

    pub(super) fn stats(&self) -> SymmetricSliceStats {
        SymmetricSliceStats {
            destination_bytes: self.destination.1,
            peak_workspace_bytes: self.peak_workspace,
            peak_total_bytes: self.peak_total,
        }
    }

    pub(super) fn set_base<'a, R: 'a, D: TensorScalar + 'a, S: NetworkPayloadStorage<D> + 'a>(
        &mut self,
        tensors: impl IntoIterator<Item = &'a TensorMap<R, D, S>>,
    ) {
        self.base.clear();
        self.base.extend(
            tensors
                .into_iter()
                .filter_map(TensorMap::network_owned_payload),
        );
    }

    pub(super) fn observe<R, D: TensorScalar, S: NetworkPayloadStorage<D>>(
        &mut self,
        slots: &[Option<TensorMap<R, D, S>>],
        producers: &[Option<(usize, bool)>],
        extra: &[Option<(usize, usize)>],
    ) -> std::result::Result<(), PayloadMeterError> {
        let mut seen = HashSet::from([self.destination.0]);
        let mut workspace = 0usize;
        let mut charge = |payload: Option<(usize, usize)>| {
            let Some((identity, bytes)) = payload else {
                return Ok(());
            };
            if seen.insert(identity) {
                workspace = workspace
                    .checked_add(bytes)
                    .ok_or(PayloadMeterError::ArithmeticOverflow)?;
            }
            Ok(())
        };
        for &(identity, bytes) in &self.base {
            charge(Some((identity, bytes)))?;
        }
        for (slot, producer) in slots.iter().zip(producers) {
            if producer.is_some() {
                charge(slot.as_ref().and_then(TensorMap::network_owned_payload))?;
            }
        }
        for &payload in extra {
            charge(payload)?;
        }
        let total = self
            .destination
            .1
            .checked_add(workspace)
            .ok_or(PayloadMeterError::ArithmeticOverflow)?;
        self.peak_workspace = self.peak_workspace.max(workspace);
        self.peak_total = self.peak_total.max(total);
        if total > self.limit {
            return Err(PayloadMeterError::Limit {
                limit: self.limit,
                required: total,
            });
        }
        Ok(())
    }
}

pub(super) fn intermediate_payloads<R, D: TensorScalar, S: NetworkPayloadStorage<D>>(
    intermediates: &[TypedIntermediateBuffers<R, D, S>],
) -> Vec<Option<(usize, usize)>> {
    #[cfg(test)]
    INTERMEDIATE_PAYLOAD_SNAPSHOT_CALLS.with(|calls| {
        if let Some(count) = calls.get() {
            calls.set(Some(count + 1));
        }
    });
    intermediates
        .iter()
        .flat_map(|buffers| [buffers.contracted.as_ref(), buffers.oriented.as_ref()])
        .flatten()
        .map(TensorMap::network_owned_payload)
        .collect()
}

pub(super) enum MeteredNetworkError<E> {
    Tensor(E),
    Payload(PayloadMeterError),
}

impl<E> From<E> for MeteredNetworkError<E> {
    fn from(error: E) -> Self {
        Self::Tensor(error)
    }
}

pub(super) fn map_payload_error<E>(error: PayloadMeterError) -> SymmetricSliceExecutionError<E> {
    match error {
        PayloadMeterError::Limit { limit, required } => {
            SymmetricSliceExecutionError::WorkspaceLimitExceeded { limit, required }
        }
        PayloadMeterError::ArithmeticOverflow => {
            SymmetricSliceExecutionError::WorkspaceArithmeticOverflow
        }
    }
}

pub(super) fn map_metered_network_error<E>(
    error: MeteredNetworkError<E>,
) -> SymmetricSliceExecutionError<E> {
    match error {
        MeteredNetworkError::Tensor(error) => SymmetricSliceExecutionError::Tensor(error),
        MeteredNetworkError::Payload(error) => map_payload_error(error),
    }
}

impl<R, D, S> Default for TypedIntermediateBuffers<R, D, S> {
    fn default() -> Self {
        Self {
            contracted: None,
            oriented: None,
            parked_contracted: None,
            parked_oriented: None,
        }
    }
}

impl<R, D, S> Default for NetworkExecutionWorkspace<R, D, S>
where
    S: NetworkPayloadStorage<D>,
{
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            producers: Vec::new(),
            intermediates: Vec::new(),
            owner_token: None,
            runtime: None,
            rule_identity: None,
            input_snapshot: Vec::new(),
        }
    }
}

impl<R, D, S> NetworkExecutionWorkspace<R, D, S>
where
    S: NetworkPayloadStorage<D>,
{
    #[cfg(test)]
    pub(crate) fn with_test_slot_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    fn clear_replay_state(&mut self) {
        self.slots.clear();
        self.producers.clear();
        self.intermediates.clear();
        self.owner_token = None;
        self.runtime = None;
        self.rule_identity = None;
        self.input_snapshot.clear();
    }

    pub(crate) fn slot_capacity(&self) -> usize {
        self.slots.capacity()
    }

    pub(crate) fn clear_slots(&mut self) {
        self.slots.clear();
        self.producers.clear();
    }

    /// Bytes retained solely to make this idle workspace reusable.
    ///
    /// This charges dense destination allocation capacities and every
    /// workspace-owned Vec backing. Each parked destination also charges its
    /// complete provider-neutral validated layout conservatively; the Runtime
    /// budget therefore remains a ceiling even if that workspace is the last
    /// owner of a shared layout descendant. Runtime/provider owners are
    /// detached while idle; the provider-neutral rule identity is charged.
    pub(crate) fn retained_idle_bytes(&self) -> usize {
        let mut bytes = self
            .slots
            .capacity()
            .saturating_mul(std::mem::size_of::<Option<TensorMap<R, D, S>>>())
            .saturating_add(
                self.producers
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Option<(usize, bool)>>()),
            )
            .saturating_add(
                self.intermediates
                    .capacity()
                    .saturating_mul(std::mem::size_of::<TypedIntermediateBuffers<R, D, S>>()),
            )
            .saturating_add(
                self.input_snapshot
                    .capacity()
                    .saturating_mul(std::mem::size_of::<TypedInputSnapshot>()),
            )
            .saturating_add(
                self.rule_identity
                    .as_ref()
                    .map_or(0, RuleIdentity::charged_retained_bytes),
            );
        for snapshot in &self.input_snapshot {
            bytes = bytes.saturating_add(
                snapshot
                    .spaces
                    .capacity()
                    .saturating_mul(std::mem::size_of::<SectorLeg>()),
            );
            bytes = snapshot.spaces.iter().fold(bytes, |bytes, leg| {
                bytes.saturating_add(leg.charged_retained_bytes())
            });
        }
        for buffers in &self.intermediates {
            for parked in [
                buffers.parked_contracted.as_ref(),
                buffers.parked_oriented.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                bytes = bytes.saturating_add(parked.retained_dense_capacity_bytes());
            }
        }
        bytes
    }

    pub(crate) fn park_runtime_owners(&mut self)
    where
        R: tenet::core::FusionRule,
    {
        for buffers in &mut self.intermediates {
            debug_assert!(buffers.parked_contracted.is_none());
            debug_assert!(buffers.parked_oriented.is_none());
            buffers.parked_contracted = buffers
                .contracted
                .take()
                .and_then(TensorMap::detach_runtime);
            buffers.parked_oriented = buffers.oriented.take().and_then(TensorMap::detach_runtime);
        }
    }

    pub(super) fn activate_parked(
        &mut self,
        runtime: &Runtime,
        tensors: &[&TensorMap<R, D, S>],
        steps: &[CompiledStep],
    ) -> Result<(), Error>
    where
        R: tenet::core::FusionRule,
    {
        // Validate the complete idle set before consuming any payload. A
        // runtime/layout drift makes the old destinations ineligible, but is
        // not an execution error: discard them and let replay allocate against
        // the current authorities.
        let reusable = self.intermediates.iter().zip(steps).all(|(buffers, step)| {
            let authority = tensors[step.authority_input_slot];
            buffers
                .parked_contracted
                .as_ref()
                .is_none_or(|tensor| tensor.can_attach(runtime, authority).is_ok())
                && buffers
                    .parked_oriented
                    .as_ref()
                    .is_none_or(|tensor| tensor.can_attach(runtime, authority).is_ok())
        });
        if !reusable {
            for buffers in &mut self.intermediates {
                buffers.parked_contracted = None;
                buffers.parked_oriented = None;
            }
            return Ok(());
        }
        for (buffers, step) in self.intermediates.iter_mut().zip(steps) {
            let authority = tensors[step.authority_input_slot];
            if let Some(tensor) = buffers.parked_contracted.take() {
                buffers.contracted = Some(tensor.attach_runtime(runtime, authority)?);
            }
            if let Some(tensor) = buffers.parked_oriented.take() {
                buffers.oriented = Some(tensor.attach_runtime(runtime, authority)?);
            }
        }
        Ok(())
    }
}

impl PlannedNetwork {
    /// The resolved pairwise contraction order with its cost estimates.
    pub fn plan(&self) -> &ContractionPlan {
        &self.plan
    }

    /// Executes the compiled schedule on device tensors: the Host step
    /// sequence, each step through the device twin of the Host typed
    /// operation. Every device rejection — another placement, a compact
    /// operand, non-symmetric braiding — is decided before any allocation or
    /// kernel, and
    /// nothing falls back to Host or transfers.
    #[cfg(feature = "cuda")]
    pub fn execute_cuda<R, D>(
        &self,
        tensors: &[&TensorMap<R, D, CudaStorage<D>>],
    ) -> Result<TensorMap<R, D, CudaStorage<D>>, Error>
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
        D: CudaPayload,
    {
        self.execute_cuda_with_workspace(tensors, &mut NetworkExecutionWorkspace::default())
    }

    /// Device execution with reusable private replay state.
    ///
    /// Device admission is checked here; everything after it — operand count,
    /// Runtime and rule identity, topology drift, contracted-leg spaces,
    /// replay-state revalidation, the step loop and the workspace lifecycle —
    /// is the same storage-generic body Host runs.
    #[cfg(feature = "cuda")]
    pub(crate) fn execute_cuda_with_workspace<R, D>(
        &self,
        tensors: &[&TensorMap<R, D, CudaStorage<D>>],
        workspace: &mut NetworkExecutionWorkspace<R, D, CudaStorage<D>>,
    ) -> Result<TensorMap<R, D, CudaStorage<D>>, Error>
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
        D: CudaPayload,
    {
        self.validate_cuda_admission(tensors)?;
        self.execute(tensors, workspace)
    }

    /// The device network preflight: every rejection class a device step
    /// could raise that the storage-generic body would not already raise
    /// before its first step, decided from the compiled schedule and the
    /// operands' metadata alone — no allocation, no lock, no device work.
    ///
    /// Classes: an operand on another placement ([`Error::PlacementMismatch`])
    /// or a Runtime without a device; then [`device_operand_admission`] — a
    /// compact (diagonal) operand and non-symmetric braiding. The device
    /// contraction's own remaining boundaries are unreachable from a compiled
    /// schedule: every step passes `alpha = 1`, every retained destination is
    /// a canonical device result of the same step, and schedules are produced
    /// only by `compile_schedule`. `tensor!` decides the same classes from
    /// its operand count, before any trace and the plan lookup
    /// ([`cuda_operand_admission`]).
    #[cfg(feature = "cuda")]
    fn validate_cuda_admission<R, D>(
        &self,
        tensors: &[&TensorMap<R, D, CudaStorage<D>>],
    ) -> Result<(), Error>
    where
        R: TypedSectorAdmission + tenet::core::FusionRule,
        D: CudaPayload,
    {
        cuda_operand_admission(tensors, !self.schedule.steps.is_empty())
    }
}

/// The operand half of the device network preflight, shared with the
/// `tensor!` entries, which run it before any trace and the plan lookup: a
/// Runtime with a device, every operand on it, then
/// [`device_operand_admission`].
#[cfg(feature = "cuda")]
pub(super) fn cuda_operand_admission<R, D>(
    tensors: &[&TensorMap<R, D, CudaStorage<D>>],
    contracts: bool,
) -> Result<(), Error>
where
    R: TypedSectorAdmission + tenet::core::FusionRule,
    D: CudaPayload,
{
    let device = tensors
        .first()
        .ok_or_else(|| invalid("network execution requires at least one operand"))?
        .runtime()
        .cuda_device_ordinal()
        .ok_or_else(|| {
            invalid(
                "this runtime was built without a CUDA device; use Runtime::builder().cuda(device)",
            )
        })?;
    for &tensor in tensors {
        if tensor.placement() != Placement::Cuda(device) {
            return Err(Error::PlacementMismatch);
        }
    }
    device_operand_admission(
        contracts,
        tensors[0].provider().braiding_style(),
        tensors
            .iter()
            .map(|tensor| tensor.network_reuse_class(false)),
    )
}

/// Representation and category checks of the device network preflight,
/// written over metadata only so they are testable without a device.
///
/// A compact (diagonal) operand has no device payload form (`to_cuda`
/// densifies), so no device contraction or permute accepts one. A schedule
/// with a contraction step on a non-symmetric (anyonic or unbraided) provider
/// is the typed `contract`'s own boundary, checked by the same function
/// (TensorKit `blas_contract!` requires symmetric braiding);
/// deciding it here, not at the step, keeps earlier steps from allocating.
#[cfg(any(feature = "cuda", test))]
pub(super) fn device_operand_admission(
    contracts: bool,
    braiding: tenet::core::BraidingStyleKind,
    representations: impl IntoIterator<Item = NetworkReuseClass>,
) -> Result<(), Error> {
    if representations
        .into_iter()
        .any(|class| class == NetworkReuseClass::Compact)
    {
        return Err(Error::UnsupportedOnDevice(
            "typed CUDA network execution requires dense device operands".to_string(),
        ));
    }
    if contracts {
        tenet::typed::reject_non_symmetric_contraction(braiding)?;
    }
    Ok(())
}

impl PlannedNetwork {
    /// Executes this plan with reusable private Host replay state.
    ///
    /// This does not accept or preserve a caller-owned output destination;
    /// successful execution returns a new owned tensor.
    pub fn execute<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        workspace: &mut NetworkExecutionWorkspace<R, D, S>,
    ) -> Result<TensorMap<R, D, S>, HostNetworkError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D, S>,
        D: TensorScalar,
        S: NetworkPayloadStorage<D>,
    {
        match self.execute_with_workspace_meter(tensors, workspace, None) {
            Ok(result) => Ok(result),
            Err(MeteredNetworkError::Tensor(error)) => Err(error),
            Err(MeteredNetworkError::Payload(_)) => {
                unreachable!("ordinary execution has no payload meter")
            }
        }
    }

    pub(super) fn execute_with_workspace_meter<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        workspace: &mut NetworkExecutionWorkspace<R, D, S>,
        mut meter: Option<&mut PayloadMeter>,
    ) -> std::result::Result<TensorMap<R, D, S>, MeteredNetworkError<HostNetworkError<R>>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D, S>,
        D: TensorScalar,
        S: NetworkPayloadStorage<D>,
    {
        let prepared: Result<_, HostNetworkError<R>> = (|| {
            if tensors.len() != self.schedule.input_ranks.len() {
                return Err(invalid(format!(
                    "plan has {} operands but {} tensors were given",
                    self.schedule.input_ranks.len(),
                    tensors.len()
                ))
                .into());
            }
            reject_non_symmetric_network(tensors, !self.schedule.steps.is_empty())?;
            let runtime = tensors
                .first()
                .ok_or_else(|| {
                    HostNetworkError::<R>::from(invalid(
                        "network execution requires at least one operand",
                    ))
                })?
                .runtime();
            let runtime_identity = runtime.identity();
            let rule_identity = TypedSectorAdmission::typed_rule_identity(tensors[0].provider());
            for (index, &tensor) in tensors.iter().enumerate() {
                if !runtime_identity.matches(tensor.runtime()) {
                    return Err(invalid(format!("operand {index} uses a different Runtime")).into());
                }
                if rule_identity != TypedSectorAdmission::typed_rule_identity(tensor.provider()) {
                    return Err(Error::RuleMismatch.into());
                }
                if tensor.rank() != self.schedule.input_ranks[index]
                    || tensor.codomain_rank() != self.input_codomain_ranks[index]
                {
                    return Err(invalid(format!(
                        "operand {index} topology drifted: planned rank/split {}/{}, got {}/{}",
                        self.schedule.input_ranks[index],
                        self.input_codomain_ranks[index],
                        tensor.rank(),
                        tensor.codomain_rank()
                    ))
                    .into());
                }
            }
            let snapshot_matches = workspace.owner_token == Some(self.owner_token)
                && workspace
                    .runtime
                    .as_ref()
                    .is_some_and(|cached| cached.matches(runtime))
                && workspace.rule_identity == Some(rule_identity.clone())
                && workspace.input_snapshot.len() == tensors.len()
                && tensors.iter().enumerate().all(|(index, tensor)| {
                    tensor.network_input_metadata_matches(
                        self.conj[index],
                        &workspace.input_snapshot[index].spaces,
                        workspace.input_snapshot[index].reuse_class,
                    )
                });
            let lowered = tensors
                .iter()
                .enumerate()
                .map(|(index, tensor)| {
                    if self.conj[index] {
                        <R::Mode as HostNetworkModeDispatch<R, D, S>>::adjoint_operand(tensor)
                    } else {
                        Ok((*tensor).clone())
                    }
                })
                .collect::<Result<Vec<_>, HostNetworkError<R>>>()?;
            let new_snapshot = if snapshot_matches {
                None
            } else {
                validate_typed_contracted_pairs(&lowered, &self.schedule.contracted_input_pairs)?;
                Some(
                    lowered
                        .iter()
                        .map(|tensor| {
                            let spaces = tensor
                                .codomain()
                                .into_iter()
                                .chain(tensor.domain())
                                .map(|space| space.network_sector_leg().clone())
                                .collect();
                            TypedInputSnapshot {
                                spaces,
                                reuse_class: tensor.network_reuse_class(false),
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            };
            let reuse_enabled = <R::Mode as HostNetworkModeDispatch<R, D, S>>::REUSE_DESTINATIONS
                && new_snapshot
                    .as_ref()
                    .unwrap_or(&workspace.input_snapshot)
                    .iter()
                    .all(|snapshot| snapshot.reuse_class != NetworkReuseClass::Compact);
            Ok((
                runtime_identity,
                rule_identity,
                lowered,
                new_snapshot,
                reuse_enabled,
            ))
        })();
        let (runtime_identity, rule_identity, lowered, new_snapshot, reuse_enabled) = prepared?;
        if new_snapshot.is_none() && reuse_enabled {
            <R::Mode as HostNetworkModeDispatch<R, D, S>>::activate_parked(
                workspace,
                tensors[0].runtime(),
                tensors,
                &self.schedule.steps,
            )?;
        }
        if let Some(snapshot) = new_snapshot {
            workspace.clear_replay_state();
            workspace.owner_token = Some(self.owner_token);
            workspace.runtime = Some(runtime_identity);
            workspace.rule_identity = Some(rule_identity);
            workspace.input_snapshot = snapshot;
        } else if !reuse_enabled {
            workspace.slots.clear();
            workspace.producers.clear();
            workspace.intermediates.clear();
        }
        workspace
            .slots
            .resize_with(self.schedule.slot_count, || None);
        workspace.producers.resize(self.schedule.slot_count, None);
        workspace
            .intermediates
            .resize_with(self.schedule.steps.len(), TypedIntermediateBuffers::default);
        if reuse_enabled {
            for index in 0..workspace.slots.len() {
                if let Some(tensor) = workspace.slots[index].take() {
                    let producer = workspace.producers[index].take();
                    return_typed_intermediate(&mut workspace.intermediates, tensor, producer, true);
                }
            }
        }
        for (step, buffers) in self.schedule.steps.iter().zip(&mut workspace.intermediates) {
            let provider = tensors[step.authority_input_slot].provider();
            if buffers
                .contracted
                .as_ref()
                .is_some_and(|tensor| !std::ptr::eq(tensor.provider(), provider))
            {
                buffers.contracted = None;
            }
            if buffers
                .oriented
                .as_ref()
                .is_some_and(|tensor| !std::ptr::eq(tensor.provider(), provider))
            {
                buffers.oriented = None;
            }
        }
        workspace.slots.fill(None);
        workspace.producers.fill(None);
        for (index, lowered) in lowered.into_iter().enumerate() {
            workspace.slots[index] = Some(lowered);
        }
        if let Some(meter) = meter.as_deref_mut() {
            let payloads = intermediate_payloads(&workspace.intermediates);
            meter
                .observe(&workspace.slots, &workspace.producers, &payloads)
                .map_err(MeteredNetworkError::Payload)?;
        }
        let slots = &mut workspace.slots;
        let producers = &mut workspace.producers;
        let intermediates = &mut workspace.intermediates;
        for (step_index, step) in self.schedule.steps.iter().enumerate() {
            let retained_payloads = meter.as_ref().map(|_| intermediate_payloads(intermediates));
            let lhs = slots[step.lhs_slot].as_ref().ok_or_else(|| {
                HostNetworkError::<R>::from(invalid("lhs operand already consumed"))
            })?;
            let rhs = slots[step.rhs_slot].as_ref().ok_or_else(|| {
                HostNetworkError::<R>::from(invalid("rhs operand already consumed"))
            })?;
            let fused = step.result_output_axes.is_some();
            let TypedIntermediateBuffers {
                contracted: contracted_buffer,
                oriented: oriented_buffer,
                ..
            } = &mut intermediates[step_index];
            let contract_buffer = if fused {
                &mut *oriented_buffer
            } else {
                &mut *contracted_buffer
            };
            // The step keeps TensorOperations' split, every open lhs leg in
            // the codomain; `result_permutation` moves it afterwards.
            let (codomain, domain) = lhs
                .rank()
                .checked_sub(step.lhs_contract_axes.len())
                .and_then(|lhs_open| step.contract_output_axes.split_at_checked(lhs_open))
                .ok_or_else(|| {
                    HostNetworkError::<R>::from(invalid(
                        "contraction step axes exceed its lhs rank",
                    ))
                })?;
            let spec = ContractSpec {
                lhs: &step.lhs_contract_axes,
                rhs: &step.rhs_contract_axes,
                codomain,
                domain,
            };
            let contracted = <R::Mode as HostNetworkModeDispatch<R, D, S>>::contract_step(
                lhs,
                rhs,
                contract_buffer,
                &spec,
            )?;
            if let Some(meter) = meter.as_deref_mut() {
                let payload = contracted.get(contract_buffer).network_owned_payload();
                let mut payloads = retained_payloads
                    .as_ref()
                    .expect("meter requires retained payload snapshot")
                    .clone();
                payloads.push(payload);
                meter
                    .observe(slots, producers, &payloads)
                    .map_err(MeteredNetworkError::Payload)?;
            }
            let result = if fused {
                contracted.take(oriented_buffer)
            } else if let Some((codomain, domain)) = &step.result_permutation {
                let oriented = <R::Mode as HostNetworkModeDispatch<R, D, S>>::permute_step(
                    contracted.get(contracted_buffer),
                    oriented_buffer,
                    codomain,
                    domain,
                )?;
                if let Some(meter) = meter.as_deref_mut() {
                    let mut payloads = retained_payloads
                        .as_ref()
                        .expect("meter requires retained payload snapshot")
                        .clone();
                    payloads.extend([
                        contracted.get(contracted_buffer).network_owned_payload(),
                        oriented.get(oriented_buffer).network_owned_payload(),
                    ]);
                    meter
                        .observe(slots, producers, &payloads)
                        .map_err(MeteredNetworkError::Payload)?;
                }
                contracted.retain(contracted_buffer);
                oriented.take(oriented_buffer)
            } else {
                contracted.take(contracted_buffer)
            };
            let lhs = slots[step.lhs_slot]
                .take()
                .expect("validated lhs remains until step success");
            let lhs_producer = producers[step.lhs_slot].take();
            let rhs = slots[step.rhs_slot]
                .take()
                .expect("validated rhs remains until step success");
            let rhs_producer = producers[step.rhs_slot].take();
            return_typed_intermediate(intermediates, lhs, lhs_producer, reuse_enabled);
            return_typed_intermediate(intermediates, rhs, rhs_producer, reuse_enabled);
            slots[step.result_slot] = Some(result);
            producers[step.result_slot] =
                Some((step_index, fused || step.result_permutation.is_some()));
            if let Some(meter) = meter.as_deref_mut() {
                let payloads = intermediate_payloads(intermediates);
                meter
                    .observe(slots, producers, &payloads)
                    .map_err(MeteredNetworkError::Payload)?;
            }
        }
        let result = slots[self.schedule.final_slot]
            .as_ref()
            .ok_or_else(|| HostNetworkError::<R>::from(invalid("no final tensor produced")))?;
        if let Some((codomain, domain)) = &self.schedule.final_permutation {
            let output = <R::Mode as HostNetworkModeDispatch<R, D, S>>::permute_final(
                result, codomain, domain,
            )?;
            if let Some(meter) = meter {
                let mut payloads = intermediate_payloads(intermediates);
                payloads.push(output.network_owned_payload());
                meter
                    .observe(slots, producers, &payloads)
                    .map_err(MeteredNetworkError::Payload)?;
            }
            let input = slots[self.schedule.final_slot]
                .take()
                .expect("validated final tensor remains until permutation success");
            let producer = producers[self.schedule.final_slot].take();
            return_typed_intermediate(intermediates, input, producer, reuse_enabled);
            Ok(output)
        } else {
            Ok(slots[self.schedule.final_slot]
                .take()
                .expect("validated final tensor remains until return"))
        }
    }
}

fn return_typed_intermediate<R, D, S>(
    intermediates: &mut [TypedIntermediateBuffers<R, D, S>],
    tensor: TensorMap<R, D, S>,
    producer: Option<(usize, bool)>,
    reuse_enabled: bool,
) {
    if !reuse_enabled {
        return;
    }
    if let Some((step, oriented)) = producer {
        let destination = if oriented {
            &mut intermediates[step].oriented
        } else {
            &mut intermediates[step].contracted
        };
        *destination = Some(tensor);
    }
}
