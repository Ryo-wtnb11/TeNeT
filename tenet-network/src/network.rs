//! Typed Host contraction of a labeled tensor network.
//!
//! This is the execution half rewritten for the current user layer: the
//! planner ([`NetworkIR`], [`DenseCostModel`], [`ContractionPlan`]) is pure
//! structure, and each planned pairwise step lowers to
//! typed contraction plus orientation/final permutation calls. The `tensor!`
//! macro enters the same typed schedule directly.

use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU64, Ordering};

use tenet::core::{
    CheckedFusionAlgebra, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, FusionAlgebraError, MultiplicityFreeAdmissionMode,
    MultiplicityFreeRigidSymbols, RuleIdentity, SectorCodec, SectorLeg, TensorStorage,
    TypedSectorAdmission,
};
use tenet::prelude::{Error, Runtime, TensorScalar};
#[cfg(feature = "cuda")]
use tenet::typed::CudaStorage;
use tenet::typed::{
    GradedSpace, NetworkDegeneracyRestriction, NetworkReuseClass, RuntimeDetachedTensorMap,
    TensorMap, TypedSpaceModeDispatch, TypedTensorAdjointDispatch, TypedTensorContractDispatch,
    TypedTensorModeDispatch, TypedTensorRootDispatch, TypedTensorTraceDispatch,
    TypedTensorTransformDispatch,
};
use tenet::RuntimeIdentity;
#[cfg(feature = "cuda")]
use tenet::{core::Placement, operations::OperationError};

use crate::cost::{DenseCostModel, DenseTensorInfo};
use crate::error::{SliceError, SymmetricSliceExecutionError, SymmetricSliceLowerError};
use crate::ir::NetworkIR;
use crate::labels::{TemporaryLabel, TensorId};
use crate::optimizer::{ContractionStep, DenseContractionOptimizer};
use crate::plan::ContractionPlan;
use crate::slice::{
    lower_symmetric_sliced_plan, validate_contraction_plan_for_ir, SlicedPlan, SymmetricSlicePlan,
    SymmetricSliceSpec, SymmetricSlicedPlan,
};

/// Compile-time topology emitted by [`tensor!`].
#[doc(hidden)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StaticTopologySpec {
    pub inputs: &'static [&'static [&'static str]],
    pub conj: &'static [bool],
    pub codomain_splits: &'static [Option<usize>],
    pub output: &'static [&'static str],
    pub output_codomain_rank: Option<usize>,
}

impl StaticTopologySpec {
    pub(crate) fn network(&self) -> Result<Network, Error> {
        Network::new(
            self.inputs
                .iter()
                .map(|labels| {
                    labels
                        .iter()
                        .map(|label| TemporaryLabel::from(*label))
                        .collect()
                })
                .collect(),
            self.conj.to_vec(),
            self.codomain_splits.to_vec(),
            self.output
                .iter()
                .map(|label| TemporaryLabel::from(*label))
                .collect(),
            self.output_codomain_rank,
        )
    }
}

/// A labeled tensor network: per-operand label lists (+ conj markers) and
/// the requested output labels with their codomain/domain split.
///
/// Labels are expression-local identifiers supplied by the [`tensor!`]
/// macro (or directly by a caller); there is no public einsum-string
/// parser. Build with [`Network::new`], then [`Network::plan`] +
/// [`PlannedNetwork::execute`].
///
/// [`tensor!`]: https://docs.rs/tenet-macros
pub struct Network {
    pub(crate) inputs: Vec<Vec<TemporaryLabel>>,
    pub(crate) conj: Vec<bool>,
    pub(crate) codomain_splits: Vec<Option<usize>>,
    pub(crate) output: Vec<TemporaryLabel>,
    /// Number of output labels on the codomain side (`;` position);
    /// `None` = all-codomain output.
    pub(crate) output_codomain_rank: Option<usize>,
}

static NEXT_PLAN_OWNER_TOKEN: AtomicU64 = AtomicU64::new(1);
#[cfg(test)]
static SYMMETRIC_SLICE_COMPLETED_JOBS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(test, feature = "cuda"))]
static CUDA_NETWORK_CONTRACT_CALLS: AtomicUsize = AtomicUsize::new(0);

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(message.to_string())
}

pub(crate) type HostNetworkError<R> =
    <<R as TypedSectorAdmission>::Mode as TypedTensorModeDispatch<R>>::FacadeError;

struct LoweredTypedNetwork<R> {
    ir: NetworkIR,
    infos: Vec<DenseTensorInfo>,
    spaces: Vec<Vec<GradedSpace<R>>>,
    rule_identity: RuleIdentity,
}

#[allow(dead_code)]
pub(crate) struct BoundSymmetricSlicedPlan<R> {
    plan: SymmetricSlicedPlan,
    authorities: Vec<GradedSpace<R>>,
    occurrences: Vec<Vec<BoundSliceOccurrence>>,
    output_effective: Vec<GradedSpace<R>>,
    output_codomain_rank: usize,
}

#[derive(Clone, Copy)]
struct BoundSliceOccurrence {
    slice_index: usize,
    effective_axis: usize,
    partner: bool,
}

#[allow(dead_code)]
impl<R> BoundSymmetricSlicedPlan<R> {
    pub(crate) fn plan(&self) -> &SymmetricSlicedPlan {
        &self.plan
    }
}

mod host_mode_sealed {
    pub trait Sealed {}

    impl Sealed for tenet::core::MultiplicityFreeAdmissionMode {}
    impl Sealed for tenet::core::CheckedGenericAdmissionMode {}
}

/// Internal Host network policy selected by the provider's admission mode.
#[doc(hidden)]
pub trait HostNetworkModeDispatch<R, D>:
    host_mode_sealed::Sealed
    + TypedTensorModeDispatch<R>
    + TypedSpaceModeDispatch<R>
    + TypedTensorAdjointDispatch<R, D>
    + TypedTensorContractDispatch<R, D>
    + TypedTensorRootDispatch<R>
    + TypedTensorTransformDispatch<R, D>
where
    R: TypedSectorAdmission<Mode = Self>,
    D: TensorScalar,
{
    const REUSE_DESTINATIONS: bool;

    fn leg_dims<S: TensorStorage<D>>(
        tensor: &TensorMap<R, D, S>,
    ) -> Result<Vec<usize>, HostNetworkError<R>>;

    fn contract_step(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        destination: &mut Option<TensorMap<R, D>>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<StepOutput<TensorMap<R, D>>, HostNetworkError<R>>;

    fn permute_step(
        tensor: &TensorMap<R, D>,
        destination: &mut Option<TensorMap<R, D>>,
        codomain: &[usize],
        domain: &[usize],
    ) -> Result<StepOutput<TensorMap<R, D>>, HostNetworkError<R>>;

    fn activate_parked(
        workspace: &mut NetworkExecutionWorkspace<R, D>,
        runtime: &Runtime,
        tensors: &[&TensorMap<R, D>],
        steps: &[CompiledStep],
    ) -> Result<(), HostNetworkError<R>>;

    fn park_workspace(workspace: &mut NetworkExecutionWorkspace<R, D>);
}

#[cfg(feature = "cuda")]
fn unsupported_cuda_network() -> Error {
    OperationError::UnsupportedTensorContractScope {
        message: "typed CUDA network execution supports only canonical whole-domain/whole-codomain contractions with identity intermediate and final output order",
    }
    .into()
}

impl Network {
    /// Build and validate a network from written label lists.
    ///
    /// `inputs[i]` are operand `i`'s labels in flat leg order (codomain
    /// then domain of the tensor as passed, i.e. *before* any conj
    /// lowering), `conj[i]` marks adjoint operands, `codomain_splits[i]`
    /// is the written `;` position (validated against the tensor later).
    /// Label structure (each label open-once or contracted-twice, output
    /// labels present and unique) is validated here.
    pub fn new(
        inputs: Vec<Vec<TemporaryLabel>>,
        conj: Vec<bool>,
        codomain_splits: Vec<Option<usize>>,
        output: Vec<TemporaryLabel>,
        output_codomain_rank: Option<usize>,
    ) -> Result<Self, Error> {
        if conj.len() != inputs.len() || codomain_splits.len() != inputs.len() {
            return Err(invalid("operand marker lists must match operand count"));
        }
        if let Some(k) = output_codomain_rank {
            if k > output.len() {
                return Err(invalid(format!(
                    "output codomain rank {k} exceeds output rank {}",
                    output.len()
                )));
            }
        }
        // Validates hyperedge structure (diagonal / hyperedge / batch /
        // reduction rejection) on the WRITTEN labels; conj rotation is a
        // cyclic per-operand relabeling that does not change the structure.
        NetworkIR::from_labels(inputs.clone(), output.clone()).map_err(invalid)?;
        Ok(Self {
            inputs,
            conj,
            codomain_splits,
            output,
            output_codomain_rank,
        })
    }

    /// Plans from storage-independent metadata of homogeneous typed Host
    /// tensors; payload storage is never read or transferred.
    pub fn plan<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        optimizer: &(impl DenseContractionOptimizer + ?Sized),
    ) -> Result<PlannedNetwork, HostNetworkError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let LoweredTypedNetwork { ir, infos, .. } = self.lower_typed(tensors)?;
        let plan = if ir.tensors().len() == 1 {
            ContractionPlan::new(1, self.output.clone(), Vec::new()).map_err(invalid)?
        } else {
            let cost = DenseCostModel::from_network(&ir, &infos).map_err(invalid)?;
            ContractionPlan::from_dense_optimizer(&ir, optimizer, &cost).map_err(invalid)?
        };
        self.finish_typed_plan(tensors, ir, plan)
    }

    #[cfg(feature = "opt-path")]
    pub(crate) fn plan_with_optimizer_fallbacks<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        optimizer: &dyn DenseContractionOptimizer,
        fallbacks: &[&dyn DenseContractionOptimizer],
    ) -> Result<PlannedNetwork, HostNetworkError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let LoweredTypedNetwork { ir, infos, .. } = self.lower_typed(tensors)?;
        let plan = if ir.tensors().len() == 1 {
            ContractionPlan::new(1, self.output.clone(), Vec::new()).map_err(invalid)?
        } else {
            let cost = DenseCostModel::from_network(&ir, &infos).map_err(invalid)?;
            let mut result = ContractionPlan::from_dense_optimizer(&ir, optimizer, &cost);
            for optimizer in fallbacks {
                if result.is_ok() {
                    break;
                }
                result = ContractionPlan::from_dense_optimizer(&ir, *optimizer, &cost);
            }
            result.map_err(invalid)?
        };
        self.finish_typed_plan(tensors, ir, plan)
    }

    /// Wraps an already searched structural order for typed execution.
    pub fn plan_with<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        plan: ContractionPlan,
    ) -> Result<PlannedNetwork, HostNetworkError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let LoweredTypedNetwork { ir, .. } = self.lower_typed(tensors)?;
        self.finish_typed_plan(tensors, ir, plan)
    }

    /// Lowers planner-selected labels into a reconstructable coefficient-free plan.
    ///
    /// The result is unbound: it records self-consistent authority-leg snapshots
    /// but does not prove tensor provenance. The future sliced executor must bind
    /// it again against the actual typed tensors before execution.
    pub fn lower_symmetric_sliced_plan<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        sliced: SlicedPlan,
    ) -> std::result::Result<SymmetricSlicedPlan, SymmetricSliceLowerError<HostNetworkError<R>>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let lowered = self
            .lower_typed(tensors)
            .map_err(SymmetricSliceLowerError::Tensor)?;
        let legs = lowered
            .spaces
            .iter()
            .map(|spaces| {
                spaces
                    .iter()
                    .map(|space| space.network_sector_leg().clone())
                    .collect()
            })
            .collect::<Vec<_>>();
        lower_symmetric_sliced_plan(&lowered.ir, lowered.rule_identity, &legs, sliced)
    }

    #[allow(dead_code)]
    pub(crate) fn bind_symmetric_sliced_plan<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        plan: SymmetricSlicedPlan,
    ) -> std::result::Result<
        BoundSymmetricSlicedPlan<R>,
        SymmetricSliceLowerError<HostNetworkError<R>>,
    >
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let lowered = self
            .lower_typed(tensors)
            .map_err(SymmetricSliceLowerError::Tensor)?;
        validate_contraction_plan_for_ir(&lowered.ir, plan.plan())
            .map_err(SymmetricSliceLowerError::InvalidPlan)?;
        let expected = plan.slices().rule_identity().clone();
        if expected != lowered.rule_identity {
            return Err(SymmetricSliceLowerError::RuleMismatch {
                expected,
                actual: lowered.rule_identity,
            });
        }

        let mut authorities = Vec::with_capacity(plan.slices().indices().len());
        let mut specs = Vec::with_capacity(plan.slices().indices().len());
        for index in plan.slices().indices() {
            let edge = lowered.ir.edge(index.label()).ok_or_else(|| {
                SymmetricSliceLowerError::InvalidSlice(SliceError::UnknownLabel(
                    index.label().clone(),
                ))
            })?;
            let authority = index.authority();
            let expected_authority = edge.occurrences()[0];
            if authority != expected_authority {
                return Err(SymmetricSliceLowerError::InvalidSlice(
                    SliceError::InvalidAuthority {
                        label: index.label().clone(),
                        expected: expected_authority,
                        actual: authority,
                    },
                ));
            }
            let actual = lowered
                .spaces
                .get(authority.tensor().index())
                .and_then(|spaces| spaces.get(authority.axis()))
                .cloned()
                .ok_or_else(|| SymmetricSliceLowerError::MissingAuthority {
                    label: index.label().clone(),
                    authority,
                })?;
            let actual_leg = actual.network_sector_leg().clone();
            if index.authority_leg() != &actual_leg {
                return Err(SymmetricSliceLowerError::AuthorityLegMismatch {
                    label: index.label().clone(),
                    authority,
                    expected: actual_leg,
                    actual: index.authority_leg().clone(),
                });
            }
            authorities.push(actual);
            specs.push(SymmetricSliceSpec::new(
                index.label().clone(),
                authority,
                actual_leg,
                index.pieces().to_vec(),
            ));
        }
        let rebound = SymmetricSlicePlan::try_new(&lowered.ir, lowered.rule_identity, specs)
            .map_err(SymmetricSliceLowerError::InvalidSlice)?;
        debug_assert_eq!(&rebound, plan.slices());
        let mut occurrences = vec![Vec::new(); tensors.len()];
        for (slice_index, index) in rebound.indices().iter().enumerate() {
            let edge = lowered
                .ir
                .edge(index.label())
                .expect("rebound label exists");
            for (occurrence_index, occurrence) in edge.occurrences().iter().enumerate() {
                occurrences[occurrence.tensor().index()].push(BoundSliceOccurrence {
                    slice_index,
                    effective_axis: occurrence.axis(),
                    partner: occurrence_index != 0,
                });
            }
        }
        let output_effective = self
            .output
            .iter()
            .map(|label| {
                let occurrence = lowered
                    .ir
                    .edge(label)
                    .expect("validated output label exists")
                    .occurrences()[0];
                lowered.spaces[occurrence.tensor().index()][occurrence.axis()].clone()
            })
            .collect();
        let output_codomain_rank = self.output_codomain_rank.unwrap_or(self.output.len());
        let plan = SymmetricSlicedPlan::new(plan.plan().clone(), rebound);
        Ok(BoundSymmetricSlicedPlan {
            plan,
            authorities,
            occurrences,
            output_effective,
            output_codomain_rank,
        })
    }

    /// Executes coefficient-free internal symmetric slices just in time and
    /// returns the result with the measured peak network-owned payload bytes.
    ///
    /// Counted payloads are compact sliced inputs, live planned contraction
    /// and permutation destinations, the current partial, the private
    /// accumulator, and path-owned dense buffers. Small metadata and opaque
    /// scratch inside an external backend are excluded because no accounting
    /// hook exists at that boundary. The ceiling is checked against observed
    /// payloads after each tensor kernel returns and before publication; it is
    /// not an allocator reservation or an out-of-memory prevention guarantee.
    /// A successful return proves that the observed peak did not exceed the
    /// ceiling.
    pub fn execute_symmetric_sliced<R, D>(
        &self,
        tensors: &[&TensorMap<R, D>],
        plan: SymmetricSlicedPlan,
        measured_payload_ceiling: usize,
    ) -> std::result::Result<
        (TensorMap<R, D>, usize),
        SymmetricSliceExecutionError<HostNetworkError<R>>,
    >
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
    {
        // This is structural and must precede binding/provider queries.
        if let Some(index) = plan
            .slices()
            .indices()
            .iter()
            .find(|index| index.output_position().is_some())
        {
            return Err(SymmetricSliceExecutionError::OutputSlice {
                label: index.label().clone(),
            });
        }
        // Compact diagonal readback may allocate a hidden full dense cache;
        // reject it before restriction/provider work until that representation
        // has a separately accountable copy leaf.
        if tensors
            .iter()
            .any(|tensor| tensor.network_has_compact_payload())
        {
            return Err(SymmetricSliceExecutionError::Tensor(
                invalid("symmetric sliced execution requires dense Host payloads").into(),
            ));
        }

        let bound = self
            .bind_symmetric_sliced_plan(tensors, plan)
            .map_err(SymmetricSliceExecutionError::Bind)?;
        let planned = self
            .plan_with(tensors, bound.plan.plan().clone())
            .map_err(SymmetricSliceExecutionError::Tensor)?;
        let mut meter = PayloadMeter::new(measured_payload_ceiling);
        let mut workspace = NetworkExecutionWorkspace::default();
        let (codomain, domain) = bound.output_effective.split_at(bound.output_codomain_rank);
        let mut accumulator = tensors[0]
            .network_zeros_from_effective_legs(codomain, domain)
            .map_err(SymmetricSliceExecutionError::Tensor)?;
        if bound.plan.slices().nslices() == 0 {
            meter.set_base([&accumulator]);
            meter
                .observe::<R, D>(&[], &[], &[])
                .map_err(map_payload_error)?;
            return Ok((accumulator, meter.peak));
        }

        for ordinal in 0..bound.plan.slices().nslices() {
            let selected = bound
                .plan
                .slices()
                .combination(ordinal)
                .expect("ordinal is bounded by semantic nslices");
            let mut owned = Vec::with_capacity(tensors.len());
            let mut compact_flags = Vec::with_capacity(tensors.len());
            for (operand, tensor) in tensors.iter().enumerate() {
                let occurrences = &bound.occurrences[operand];
                compact_flags.push(!occurrences.is_empty());
                if occurrences.is_empty() {
                    owned.push((*tensor).clone());
                    continue;
                }
                let restrictions = occurrences
                    .iter()
                    .map(|occurrence| {
                        let piece = selected[occurrence.slice_index];
                        let range = piece.range();
                        NetworkDegeneracyRestriction {
                            effective_axis: occurrence.effective_axis,
                            authority_sector: piece.sector(),
                            range: range.start()..range.end(),
                            partner: occurrence.partner,
                        }
                    })
                    .collect::<Vec<_>>();
                owned.push(
                    tensor
                        .network_restrict_degeneracies(self.conj[operand], &restrictions)
                        .map_err(SymmetricSliceExecutionError::Tensor)?,
                );
            }
            let compact_inputs = owned
                .iter()
                .zip(&compact_flags)
                .filter_map(|(tensor, &compact)| compact.then_some(tensor));
            meter.set_base(compact_inputs.chain(std::iter::once(&accumulator)));
            let payloads = intermediate_payloads(&workspace.intermediates);
            meter
                .observe(&workspace.slots, &workspace.producers, &payloads)
                .map_err(map_payload_error)?;

            let refs = owned.iter().collect::<Vec<_>>();
            let partial = planned
                .execute_with_workspace_meter(&refs, &mut workspace, Some(&mut meter))
                .map_err(map_metered_network_error)?;
            #[cfg(test)]
            SYMMETRIC_SLICE_COMPLETED_JOBS.fetch_add(1, Ordering::SeqCst);

            let compact_inputs = owned
                .iter()
                .zip(&compact_flags)
                .filter_map(|(tensor, &compact)| compact.then_some(tensor));
            meter.set_base(
                compact_inputs
                    .chain(std::iter::once(&accumulator))
                    .chain(std::iter::once(&partial)),
            );
            let payloads = intermediate_payloads(&workspace.intermediates);
            meter
                .observe(&workspace.slots, &workspace.producers, &payloads)
                .map_err(map_payload_error)?;

            accumulator
                .network_add_subset_assign(&partial)
                .map_err(|error| {
                    SymmetricSliceExecutionError::Tensor(HostNetworkError::<R>::from(error))
                })?;
        }
        Ok((accumulator, meter.peak))
    }

    fn finish_typed_plan<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        ir: NetworkIR,
        plan: ContractionPlan,
    ) -> Result<PlannedNetwork, HostNetworkError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let input_codomain_ranks = tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect();
        let lowered_codomain_ranks = tensors
            .iter()
            .enumerate()
            .map(|(i, tensor)| {
                if self.conj[i] {
                    tensor.domain_rank()
                } else {
                    tensor.codomain_rank()
                }
            })
            .collect::<Vec<_>>();
        self.finish_plan(input_codomain_ranks, lowered_codomain_ranks, ir, plan)
            .map_err(Into::into)
    }

    fn finish_plan(
        &self,
        input_codomain_ranks: Vec<usize>,
        lowered_codomain_ranks: Vec<usize>,
        ir: NetworkIR,
        plan: ContractionPlan,
    ) -> Result<PlannedNetwork, Error> {
        let schedule = compile_schedule(
            &ir,
            &plan,
            self.output_codomain_rank,
            &lowered_codomain_ranks,
        )?;
        #[cfg(feature = "cuda")]
        let cuda_direct = cuda_schedule_is_direct(
            &schedule,
            &schedule
                .input_ranks
                .iter()
                .copied()
                .zip(lowered_codomain_ranks.iter().copied())
                .collect::<Vec<_>>(),
        );
        Ok(PlannedNetwork {
            owner_token: NEXT_PLAN_OWNER_TOKEN.fetch_add(1, Ordering::Relaxed),
            plan,
            conj: self.conj.clone(),
            input_codomain_ranks,
            schedule,
            #[cfg(feature = "cuda")]
            cuda_direct,
        })
    }

    fn lower_typed<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
    ) -> Result<LoweredTypedNetwork<R>, HostNetworkError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        if tensors.len() != self.inputs.len() {
            return Err(invalid(format!(
                "network has {} operands but {} tensors were given",
                self.inputs.len(),
                tensors.len()
            ))
            .into());
        }
        let rule_identity = if let Some(first) = tensors.first() {
            let runtime = first.runtime().identity();
            let identity = TypedSectorAdmission::typed_rule_identity(first.provider());
            for (index, tensor) in tensors.iter().enumerate().skip(1) {
                if !runtime.matches(tensor.runtime()) {
                    return Err(invalid(format!("operand {index} uses a different Runtime")).into());
                }
                if identity != TypedSectorAdmission::typed_rule_identity(tensor.provider()) {
                    return Err(Error::RuleMismatch.into());
                }
            }
            identity
        } else {
            unreachable!("a validated Network has at least one operand")
        };

        let mut lowered_labels = Vec::with_capacity(tensors.len());
        let mut infos = Vec::with_capacity(tensors.len());
        let mut lowered_spaces = Vec::with_capacity(tensors.len());
        for (i, (&tensor, labels)) in tensors.iter().zip(&self.inputs).enumerate() {
            if labels.len() != tensor.rank() {
                return Err(invalid(format!(
                    "operand {i} has {} labels but tensor rank {}",
                    labels.len(),
                    tensor.rank()
                ))
                .into());
            }
            if let Some(split) = self.codomain_splits[i] {
                if split != tensor.codomain_rank() {
                    return Err(invalid(format!(
                        "operand {i} puts {split} label(s) before `;` but the tensor's codomain rank is {}",
                        tensor.codomain_rank()
                    ))
                    .into());
                }
            }
            let dims = if self.conj[i] {
                let split = tensor.codomain_rank();
                lowered_labels.push(rotate(labels, split));
                rotate(
                    &<R::Mode as HostNetworkModeDispatch<R, D>>::leg_dims(tensor)?,
                    split,
                )
            } else {
                lowered_labels.push(labels.clone());
                <R::Mode as HostNetworkModeDispatch<R, D>>::leg_dims(tensor)?
            };
            infos.push(DenseTensorInfo::new(dims));
            lowered_spaces.push(typed_effective_spaces(tensor, self.conj[i])?);
        }
        validate_typed_contracted_leg_spaces::<R, D>(&lowered_labels, &lowered_spaces)?;
        let ir = NetworkIR::from_labels(lowered_labels, self.output.clone())
            .map_err(|error| HostNetworkError::<R>::from(invalid(error)))?;
        Ok(LoweredTypedNetwork {
            ir,
            infos,
            spaces: lowered_spaces,
            rule_identity,
        })
    }
}

fn validate_typed_contracted_leg_spaces<R, D>(
    labels: &[Vec<TemporaryLabel>],
    spaces: &[Vec<GradedSpace<R>>],
) -> Result<(), HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
{
    let mut seen: HashMap<&TemporaryLabel, (usize, usize)> = HashMap::new();
    for (operand, operand_labels) in labels.iter().enumerate() {
        for (axis, label) in operand_labels.iter().enumerate() {
            let Some(&(previous_operand, previous_axis)) = seen.get(label) else {
                seen.insert(label, (operand, axis));
                continue;
            };
            let lhs = &spaces[previous_operand][previous_axis];
            let rhs = &spaces[operand][axis];
            if rhs != &lhs.try_dual()? {
                return Err(invalid(format!(
                    "space mismatch for contracted label `{label}` between operand {previous_operand} leg {previous_axis} and operand {operand} leg {axis}"
                ))
                .into());
            }
        }
    }
    Ok(())
}

fn validate_typed_contracted_pairs<R, D>(
    tensors: &[TensorMap<R, D>],
    pairs: &[InputLegPair],
) -> Result<(), HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
{
    let spaces = tensors
        .iter()
        .map(typed_flat_spaces)
        .collect::<Result<Vec<_>, _>>()?;
    for &((lhs_slot, lhs_axis), (rhs_slot, rhs_axis)) in pairs {
        if spaces[rhs_slot][rhs_axis] != spaces[lhs_slot][lhs_axis].try_dual()? {
            return Err(invalid(format!(
                "contracted input spaces mismatch between operand {lhs_slot} leg {lhs_axis} and operand {rhs_slot} leg {rhs_axis}"
            ))
            .into());
        }
    }
    Ok(())
}

fn typed_flat_spaces<R, D, S>(
    tensor: &TensorMap<R, D, S>,
) -> Result<Vec<GradedSpace<R>>, HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    let mut spaces = tensor.codomain();
    spaces.extend(
        tensor
            .domain()
            .iter()
            .map(GradedSpace::try_dual)
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(spaces)
}

fn typed_effective_spaces<R, D, S>(
    tensor: &TensorMap<R, D, S>,
    adjoint: bool,
) -> Result<Vec<GradedSpace<R>>, HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    if !adjoint {
        return typed_flat_spaces(tensor);
    }
    let mut spaces = tensor.domain();
    spaces.extend(
        tensor
            .codomain()
            .iter()
            .map(GradedSpace::try_dual)
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(spaces)
}

#[cfg(feature = "cuda")]
fn validate_typed_input_pairs<R, D, S>(
    tensors: &[&TensorMap<R, D, S>],
    adjoints: &[bool],
    pairs: &[InputLegPair],
) -> Result<(), Error>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    let spaces = tensors
        .iter()
        .zip(adjoints)
        .map(|(&tensor, &adjoint)| typed_effective_spaces(tensor, adjoint))
        .collect::<Result<Vec<_>, Error>>()?;
    for &((lhs_slot, lhs_axis), (rhs_slot, rhs_axis)) in pairs {
        if spaces[rhs_slot][rhs_axis] != spaces[lhs_slot][lhs_axis].try_dual()? {
            return Err(invalid(format!(
                "contracted input spaces mismatch between operand {lhs_slot} leg {lhs_axis} and operand {rhs_slot} leg {rhs_axis}"
            )));
        }
    }
    Ok(())
}

fn rotate<T: Clone>(items: &[T], split: usize) -> Vec<T> {
    items[split..]
        .iter()
        .chain(items[..split].iter())
        .cloned()
        .collect()
}

/// A [`Network`] with a resolved contraction order for concrete operand
/// shapes. Inspect the order via [`Self::plan`], run it via
/// [`Self::execute`].
pub struct PlannedNetwork {
    owner_token: u64,
    plan: ContractionPlan,
    conj: Vec<bool>,
    input_codomain_ranks: Vec<usize>,
    schedule: CompiledSchedule,
    #[cfg(feature = "cuda")]
    cuda_direct: bool,
}

struct CompiledSchedule {
    slot_count: usize,
    input_ranks: Vec<usize>,
    contracted_input_pairs: Vec<InputLegPair>,
    steps: Vec<CompiledStep>,
    final_slot: usize,
    final_permutation: Option<(Vec<usize>, Vec<usize>)>,
}

type InputLegPair = ((usize, usize), (usize, usize));

#[doc(hidden)]
pub struct CompiledStep {
    lhs_slot: usize,
    rhs_slot: usize,
    result_slot: usize,
    lhs_contract_axes: Vec<usize>,
    rhs_contract_axes: Vec<usize>,
    result_permutation: Option<(Vec<usize>, Vec<usize>)>,
    result_output_axes: Option<Vec<usize>>,
    contract_output_axes: Vec<usize>,
    authority_input_slot: usize,
}

/// Caller-owned Host replay state for one planned network at a time.
///
/// The stored payload destinations are private implementation details. Host MF
/// replay can reuse compatible intermediate buffers. Checked Generic replay
/// reuses the plan and workspace containers but admits new intermediate
/// tensors. The final tensor leaves the workspace in both modes.
pub struct NetworkExecutionWorkspace<R, D> {
    slots: Vec<Option<TensorMap<R, D>>>,
    producers: Vec<Option<(usize, bool)>>,
    intermediates: Vec<TypedIntermediateBuffers<R, D>>,
    owner_token: Option<u64>,
    runtime: Option<RuntimeIdentity>,
    rule_identity: Option<RuleIdentity>,
    input_snapshot: Vec<TypedInputSnapshot>,
}

struct TypedInputSnapshot {
    spaces: Vec<SectorLeg>,
    reuse_class: NetworkReuseClass,
}

struct TypedIntermediateBuffers<R, D> {
    contracted: Option<TensorMap<R, D>>,
    oriented: Option<TensorMap<R, D>>,
    parked_contracted: Option<RuntimeDetachedTensorMap<D>>,
    parked_oriented: Option<RuntimeDetachedTensorMap<D>>,
}

struct PayloadMeter {
    limit: usize,
    peak: usize,
    base: Vec<(usize, usize)>,
}

#[derive(Debug)]
enum PayloadMeterError {
    Limit { limit: usize, required: usize },
    ArithmeticOverflow,
}

impl PayloadMeter {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            peak: 0,
            base: Vec::new(),
        }
    }

    fn set_base<'a, R: 'a, D: TensorScalar + 'a>(
        &mut self,
        tensors: impl IntoIterator<Item = &'a TensorMap<R, D>>,
    ) {
        self.base.clear();
        self.base.extend(
            tensors
                .into_iter()
                .filter_map(TensorMap::network_owned_payload),
        );
    }

    fn observe<R, D: TensorScalar>(
        &mut self,
        slots: &[Option<TensorMap<R, D>>],
        producers: &[Option<(usize, bool)>],
        extra: &[Option<(usize, usize)>],
    ) -> std::result::Result<(), PayloadMeterError> {
        let mut seen = HashSet::new();
        let mut total = 0usize;
        let mut charge = |payload: Option<(usize, usize)>| {
            let Some((identity, bytes)) = payload else {
                return Ok(());
            };
            if seen.insert(identity) {
                total = total
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
        self.peak = self.peak.max(total);
        if total > self.limit {
            return Err(PayloadMeterError::Limit {
                limit: self.limit,
                required: total,
            });
        }
        Ok(())
    }
}

fn intermediate_payloads<R, D: TensorScalar>(
    intermediates: &[TypedIntermediateBuffers<R, D>],
) -> Vec<Option<(usize, usize)>> {
    intermediates
        .iter()
        .flat_map(|buffers| [buffers.contracted.as_ref(), buffers.oriented.as_ref()])
        .flatten()
        .map(TensorMap::network_owned_payload)
        .collect()
}

enum MeteredNetworkError<E> {
    Tensor(E),
    Payload(PayloadMeterError),
}

impl<E> From<E> for MeteredNetworkError<E> {
    fn from(error: E) -> Self {
        Self::Tensor(error)
    }
}

fn map_payload_error<E>(error: PayloadMeterError) -> SymmetricSliceExecutionError<E> {
    match error {
        PayloadMeterError::Limit { limit, required } => {
            SymmetricSliceExecutionError::WorkspaceLimitExceeded { limit, required }
        }
        PayloadMeterError::ArithmeticOverflow => {
            SymmetricSliceExecutionError::WorkspaceArithmeticOverflow
        }
    }
}

fn map_metered_network_error<E>(error: MeteredNetworkError<E>) -> SymmetricSliceExecutionError<E> {
    match error {
        MeteredNetworkError::Tensor(error) => SymmetricSliceExecutionError::Tensor(error),
        MeteredNetworkError::Payload(error) => map_payload_error(error),
    }
}

#[doc(hidden)]
pub enum StepOutput<T> {
    Returned(T),
    Overwritten,
}

impl<T> StepOutput<T> {
    fn get<'a>(&'a self, destination: &'a Option<T>) -> &'a T {
        match self {
            Self::Returned(value) => value,
            Self::Overwritten => destination
                .as_ref()
                .expect("successful overwrite retains its destination"),
        }
    }

    fn take(self, destination: &mut Option<T>) -> T {
        match self {
            Self::Returned(value) => value,
            Self::Overwritten => destination
                .take()
                .expect("successful overwrite retains its destination"),
        }
    }

    fn retain(self, destination: &mut Option<T>) {
        if let Self::Returned(value) = self {
            *destination = Some(value);
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

    fn leg_dims<S: TensorStorage<D>>(tensor: &TensorMap<R, D, S>) -> Result<Vec<usize>, Error> {
        tensor.leg_dims()
    }

    fn contract_step(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        destination: &mut Option<TensorMap<R, D>>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<StepOutput<TensorMap<R, D>>, Error> {
        if let Some(destination) = destination {
            lhs.contract_overwrite_into(
                rhs,
                destination,
                lhs_axes,
                rhs_axes,
                output_axes,
                D::from_real(1.0),
            )?;
            Ok(StepOutput::Overwritten)
        } else {
            Ok(StepOutput::Returned(lhs.contract(
                rhs,
                lhs_axes,
                rhs_axes,
                output_axes,
            )?))
        }
    }

    fn permute_step(
        tensor: &TensorMap<R, D>,
        destination: &mut Option<TensorMap<R, D>>,
        codomain: &[usize],
        domain: &[usize],
    ) -> Result<StepOutput<TensorMap<R, D>>, Error> {
        if let Some(destination) = destination {
            tensor.permute_overwrite_into(destination, codomain, domain, D::from_real(1.0))?;
            Ok(StepOutput::Overwritten)
        } else {
            Ok(StepOutput::Returned(tensor.permute(codomain, domain)?))
        }
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

    fn leg_dims<S: TensorStorage<D>>(
        tensor: &TensorMap<R, D, S>,
    ) -> Result<Vec<usize>, HostNetworkError<R>> {
        tensor
            .codomain()
            .into_iter()
            .chain(tensor.domain())
            .map(|space| space.dim().map(|dimension| dimension.round() as usize))
            .collect()
    }

    fn contract_step(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        _destination: &mut Option<TensorMap<R, D>>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<StepOutput<TensorMap<R, D>>, HostNetworkError<R>> {
        Ok(StepOutput::Returned(lhs.contract(
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
        )?))
    }

    fn permute_step(
        tensor: &TensorMap<R, D>,
        _destination: &mut Option<TensorMap<R, D>>,
        codomain: &[usize],
        domain: &[usize],
    ) -> Result<StepOutput<TensorMap<R, D>>, HostNetworkError<R>> {
        Ok(StepOutput::Returned(tensor.permute(codomain, domain)?))
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
            buffers.contracted = None;
            buffers.oriented = None;
            buffers.parked_contracted = None;
            buffers.parked_oriented = None;
        }
    }
}

impl<R, D> Default for TypedIntermediateBuffers<R, D> {
    fn default() -> Self {
        Self {
            contracted: None,
            oriented: None,
            parked_contracted: None,
            parked_oriented: None,
        }
    }
}

impl<R, D> Default for NetworkExecutionWorkspace<R, D> {
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

impl<R, D> NetworkExecutionWorkspace<R, D> {
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
            .saturating_mul(std::mem::size_of::<Option<TensorMap<R, D>>>())
            .saturating_add(
                self.producers
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Option<(usize, bool)>>()),
            )
            .saturating_add(
                self.intermediates
                    .capacity()
                    .saturating_mul(std::mem::size_of::<TypedIntermediateBuffers<R, D>>()),
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

    fn activate_parked(
        &mut self,
        runtime: &Runtime,
        tensors: &[&TensorMap<R, D>],
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

    /// Pure, allocation-free validation used before a CUDA macro call may
    /// observe or publish plan-cache state.
    #[cfg(feature = "cuda")]
    pub(crate) fn validate_cuda_plan_structure(&self) -> Result<(), Error> {
        if self.cuda_direct {
            Ok(())
        } else {
            Err(unsupported_cuda_network())
        }
    }

    /// Executes a schedule expressible entirely by the canonical returning CUDA kernel.
    /// The complete schedule is preflighted before any output allocation or kernel;
    /// unsupported layouts fail without a Host fallback or transfer.
    #[cfg(feature = "cuda")]
    pub fn execute_cuda<R>(
        &self,
        tensors: &[&TensorMap<R, f64, CudaStorage>],
    ) -> Result<TensorMap<R, f64, CudaStorage>, Error>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    {
        if tensors.len() != self.schedule.input_ranks.len() {
            return Err(invalid(format!(
                "plan has {} operands but {} tensors were given",
                self.schedule.input_ranks.len(),
                tensors.len()
            )));
        }
        let runtime = tensors
            .first()
            .ok_or_else(|| invalid("network execution requires at least one operand"))?
            .runtime();
        let runtime_identity = runtime.identity();
        let rule_identity = TypedSectorAdmission::typed_rule_identity(tensors[0].provider());
        let device = runtime.cuda_device_ordinal().ok_or_else(|| {
            invalid(
                "this runtime was built without a CUDA device; use Runtime::builder().cuda(device)",
            )
        })?;
        for (index, &tensor) in tensors.iter().enumerate() {
            if !runtime_identity.matches(tensor.runtime()) {
                return Err(invalid(format!("operand {index} uses a different Runtime")));
            }
            if rule_identity != TypedSectorAdmission::typed_rule_identity(tensor.provider()) {
                return Err(Error::RuleMismatch);
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
                )));
            }
            if tensor.placement() != Placement::Cuda(device) {
                return Err(Error::PlacementMismatch);
            }
        }
        validate_typed_input_pairs(tensors, &self.conj, &self.schedule.contracted_input_pairs)?;

        let input_shapes = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                (
                    tensor.rank(),
                    if self.conj[index] {
                        tensor.domain_rank()
                    } else {
                        tensor.codomain_rank()
                    },
                )
            })
            .collect::<Vec<_>>();
        self.preflight_cuda_schedule(&input_shapes)?;

        let mut slots: Vec<Option<TensorMap<R, f64, CudaStorage>>> =
            (0..self.schedule.slot_count).map(|_| None).collect();
        for (index, &tensor) in tensors.iter().enumerate() {
            slots[index] = Some(if self.conj[index] {
                tensor.adjoint()?
            } else {
                tensor.clone()
            });
        }
        for step in &self.schedule.steps {
            let lhs = slots[step.lhs_slot]
                .take()
                .ok_or_else(|| invalid("lhs operand already consumed"))?;
            let rhs = slots[step.rhs_slot]
                .take()
                .ok_or_else(|| invalid("rhs operand already consumed"))?;
            #[cfg(all(test, feature = "cuda"))]
            CUDA_NETWORK_CONTRACT_CALLS.fetch_add(1, Ordering::Relaxed);
            let result = lhs.contract(
                &rhs,
                &step.lhs_contract_axes,
                &step.rhs_contract_axes,
                &step.contract_output_axes,
            )?;
            slots[step.result_slot] = Some(result);
        }
        slots[self.schedule.final_slot]
            .take()
            .ok_or_else(|| invalid("network execution produced no final tensor"))
    }

    #[cfg(feature = "cuda")]
    fn preflight_cuda_schedule(&self, input_shapes: &[(usize, usize)]) -> Result<(), Error> {
        if cuda_schedule_is_direct(&self.schedule, input_shapes) {
            Ok(())
        } else {
            Err(unsupported_cuda_network())
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_schedule_is_direct(schedule: &CompiledSchedule, input_shapes: &[(usize, usize)]) -> bool {
    let mut shapes = vec![None; schedule.slot_count];
    for (index, &shape) in input_shapes.iter().enumerate() {
        shapes[index] = Some(shape);
    }
    for step in &schedule.steps {
        let Some((lhs_rank, lhs_codomain_rank)) = shapes[step.lhs_slot].take() else {
            return false;
        };
        let Some((rhs_rank, rhs_codomain_rank)) = shapes[step.rhs_slot].take() else {
            return false;
        };
        let result_rank = lhs_codomain_rank + rhs_rank - rhs_codomain_rank;
        if !step
            .lhs_contract_axes
            .iter()
            .copied()
            .eq(lhs_codomain_rank..lhs_rank)
            || !step
                .rhs_contract_axes
                .iter()
                .copied()
                .eq(0..rhs_codomain_rank)
            || !step.contract_output_axes.iter().copied().eq(0..result_rank)
            || step.result_permutation.is_some()
        {
            return false;
        }
        shapes[step.result_slot] = Some((result_rank, lhs_codomain_rank));
    }
    schedule.final_permutation.is_none()
}

impl PlannedNetwork {
    /// Executes this plan with a fresh typed workspace.
    pub fn execute<R, D>(
        &self,
        tensors: &[&TensorMap<R, D>],
    ) -> Result<TensorMap<R, D>, HostNetworkError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
    {
        self.execute_with_workspace(tensors, &mut NetworkExecutionWorkspace::default())
    }

    /// Executes this plan with reusable private Host replay state.
    ///
    /// This does not accept or preserve a caller-owned output destination;
    /// successful execution returns a new owned tensor.
    pub fn execute_with_workspace<R, D>(
        &self,
        tensors: &[&TensorMap<R, D>],
        workspace: &mut NetworkExecutionWorkspace<R, D>,
    ) -> Result<TensorMap<R, D>, HostNetworkError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
    {
        match self.execute_with_workspace_meter(tensors, workspace, None) {
            Ok(result) => Ok(result),
            Err(MeteredNetworkError::Tensor(error)) => Err(error),
            Err(MeteredNetworkError::Payload(_)) => {
                unreachable!("ordinary execution has no payload meter")
            }
        }
    }

    fn execute_with_workspace_meter<R, D>(
        &self,
        tensors: &[&TensorMap<R, D>],
        workspace: &mut NetworkExecutionWorkspace<R, D>,
        mut meter: Option<&mut PayloadMeter>,
    ) -> std::result::Result<TensorMap<R, D>, MeteredNetworkError<HostNetworkError<R>>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
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
                        tensor.adjoint()
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
            let reuse_enabled = <R::Mode as HostNetworkModeDispatch<R, D>>::REUSE_DESTINATIONS
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
            <R::Mode as HostNetworkModeDispatch<R, D>>::activate_parked(
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
            let retained_payloads = intermediate_payloads(intermediates);
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
            let contracted = <R::Mode as HostNetworkModeDispatch<R, D>>::contract_step(
                lhs,
                rhs,
                contract_buffer,
                &step.lhs_contract_axes,
                &step.rhs_contract_axes,
                &step.contract_output_axes,
            )?;
            if let Some(meter) = meter.as_deref_mut() {
                let payload = contracted.get(contract_buffer).network_owned_payload();
                let mut payloads = retained_payloads.clone();
                payloads.push(payload);
                meter
                    .observe(slots, producers, &payloads)
                    .map_err(MeteredNetworkError::Payload)?;
            }
            let result = if fused {
                contracted.take(oriented_buffer)
            } else if let Some((codomain, domain)) = &step.result_permutation {
                let oriented = <R::Mode as HostNetworkModeDispatch<R, D>>::permute_step(
                    contracted.get(contracted_buffer),
                    oriented_buffer,
                    codomain,
                    domain,
                )?;
                if let Some(meter) = meter.as_deref_mut() {
                    let mut payloads = retained_payloads.clone();
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
            let output = result.permute(codomain, domain)?;
            if let Some(meter) = meter.as_deref_mut() {
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

fn return_typed_intermediate<R, D>(
    intermediates: &mut [TypedIntermediateBuffers<R, D>],
    tensor: TensorMap<R, D>,
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

fn contracted_input_pairs(ir: &NetworkIR) -> Vec<InputLegPair> {
    let mut first = HashMap::new();
    let mut pairs = Vec::new();
    for (slot, node) in ir.tensors().iter().enumerate() {
        for (axis, label) in node.labels().iter().enumerate() {
            if let Some(previous) = first.remove(label) {
                pairs.push((previous, (slot, axis)));
            } else {
                first.insert(label.clone(), (slot, axis));
            }
        }
    }
    pairs
}

fn compile_schedule(
    ir: &NetworkIR,
    plan: &ContractionPlan,
    output_codomain_rank: Option<usize>,
    input_codomain_ranks: &[usize],
) -> Result<CompiledSchedule, Error> {
    let contracted_input_pairs = contracted_input_pairs(ir);
    let labels_by_id = planned_label_orders(ir, plan)?;
    let consumers = build_consumers(plan.steps());
    let slot_count = ir.tensors().len() + plan.steps().len();
    let mut current_labels: Vec<Option<Vec<TemporaryLabel>>> = vec![None; slot_count];
    let mut current_codomain_ranks: Vec<Option<usize>> = vec![None; slot_count];
    let mut authority_input_slots: Vec<Option<usize>> = vec![None; slot_count];
    let mut slots_by_id = HashMap::with_capacity(slot_count);
    for (slot, node) in ir.tensors().iter().enumerate() {
        slots_by_id.insert(node.id(), slot);
        current_labels[slot] = Some(node.labels().to_vec());
        current_codomain_ranks[slot] = Some(input_codomain_ranks[slot]);
        authority_input_slots[slot] = Some(slot);
    }

    let mut compiled_steps = Vec::with_capacity(plan.steps().len());
    for (step_index, step) in plan.steps().iter().enumerate() {
        let lhs_slot = *slots_by_id
            .get(&step.lhs())
            .ok_or_else(|| invalid("lhs slot missing while compiling schedule"))?;
        let rhs_slot = *slots_by_id
            .get(&step.rhs())
            .ok_or_else(|| invalid("rhs slot missing while compiling schedule"))?;
        let result_slot = ir.tensors().len() + step_index;
        let lhs_labels = current_labels[lhs_slot]
            .take()
            .ok_or_else(|| invalid("lhs labels already consumed while compiling schedule"))?;
        let rhs_labels = current_labels[rhs_slot]
            .take()
            .ok_or_else(|| invalid("rhs labels already consumed while compiling schedule"))?;
        let _lhs_codomain_rank = current_codomain_ranks[lhs_slot]
            .take()
            .ok_or_else(|| invalid("lhs orientation already consumed while compiling schedule"))?;
        current_codomain_ranks[rhs_slot]
            .take()
            .ok_or_else(|| invalid("rhs orientation already consumed while compiling schedule"))?;
        let authority_input_slot = authority_input_slots[lhs_slot]
            .take()
            .ok_or_else(|| invalid("lhs authority already consumed while compiling schedule"))?;
        authority_input_slots[rhs_slot]
            .take()
            .ok_or_else(|| invalid("rhs authority already consumed while compiling schedule"))?;

        let mut lhs_contract_axes = Vec::new();
        let mut rhs_contract_axes = Vec::new();
        for (lhs_axis, label) in lhs_labels.iter().enumerate() {
            if let Some(rhs_axis) = rhs_labels.iter().position(|other| other == label) {
                lhs_contract_axes.push(lhs_axis);
                rhs_contract_axes.push(rhs_axis);
            }
        }
        let mut result_labels: Vec<TemporaryLabel> = lhs_labels
            .iter()
            .enumerate()
            .filter(|(axis, _)| !lhs_contract_axes.contains(axis))
            .map(|(_, label)| label.clone())
            .collect();
        result_labels.extend(
            rhs_labels
                .iter()
                .enumerate()
                .filter(|(axis, _)| !rhs_contract_axes.contains(axis))
                .map(|(_, label)| label.clone()),
        );

        let lhs_open_count = lhs_labels.len() - lhs_contract_axes.len();
        let result_permutation = compiled_intermediate_permutation(
            &result_labels,
            lhs_open_count,
            step.result(),
            plan.steps(),
            &consumers,
            &labels_by_id,
        )?;
        if let Some((codomain, domain)) = &result_permutation {
            result_labels = codomain
                .iter()
                .chain(domain)
                .map(|&axis| result_labels[axis].clone())
                .collect();
        }
        let result_codomain_rank = result_permutation.as_ref().map_or(
            lhs_labels.len() - lhs_contract_axes.len(),
            |(codomain, _)| codomain.len(),
        );
        let result_rank = result_labels.len();
        current_labels[result_slot] = Some(result_labels);
        current_codomain_ranks[result_slot] = Some(result_codomain_rank);
        authority_input_slots[result_slot] = Some(authority_input_slot);
        slots_by_id.insert(step.result(), result_slot);
        let result_output_axes = result_permutation
            .as_ref()
            .filter(|(codomain, _)| codomain.len() == lhs_open_count)
            .map(|(codomain, domain)| codomain.iter().chain(domain).copied().collect::<Vec<_>>());
        let contract_output_axes = result_output_axes
            .clone()
            .unwrap_or_else(|| (0..result_rank).collect());
        compiled_steps.push(CompiledStep {
            lhs_slot,
            rhs_slot,
            result_slot,
            lhs_contract_axes,
            rhs_contract_axes,
            // pAB reorders axes inside the existing split. Moving the split is
            // a repartition and remains an explicit orientation operation.
            result_output_axes,
            contract_output_axes,
            result_permutation,
            authority_input_slot,
        });
    }

    let final_id = plan
        .steps()
        .last()
        .map(|step| step.result())
        .unwrap_or_else(|| TensorId::new(0));
    let final_slot = *slots_by_id
        .get(&final_id)
        .ok_or_else(|| invalid("final slot missing while compiling schedule"))?;
    let final_labels = current_labels[final_slot]
        .as_ref()
        .ok_or_else(|| invalid("final labels missing while compiling schedule"))?;
    let final_codomain_rank = current_codomain_ranks[final_slot]
        .ok_or_else(|| invalid("final orientation missing while compiling schedule"))?;
    let output = ir.output_labels();
    let split = output_codomain_rank.unwrap_or(output.len());
    let codomain = label_positions(&output[..split], final_labels)?;
    let domain = label_positions(&output[split..], final_labels)?;
    let final_permutation = (!(final_codomain_rank == split
        && codomain
            .iter()
            .chain(&domain)
            .copied()
            .eq(0..final_labels.len())))
    .then_some((codomain, domain));

    Ok(CompiledSchedule {
        slot_count,
        input_ranks: ir
            .tensors()
            .iter()
            .map(|node| node.labels().len())
            .collect(),
        contracted_input_pairs,
        steps: compiled_steps,
        final_slot,
        final_permutation,
    })
}

/// Positions of each `wanted` label within `have` (the current leg labels).
fn label_positions(
    wanted: &[TemporaryLabel],
    have: &[TemporaryLabel],
) -> Result<Vec<usize>, Error> {
    wanted
        .iter()
        .map(|l| {
            have.iter()
                .position(|x| x == l)
                .ok_or_else(|| invalid(format!("label `{l}` not among available legs")))
        })
        .collect()
}

fn compiled_intermediate_permutation(
    labels: &[TemporaryLabel],
    current_codomain_rank: usize,
    result_id: TensorId,
    steps: &[ContractionStep],
    consumers: &HashMap<TensorId, (usize, bool)>,
    labels_by_id: &HashMap<TensorId, Vec<TemporaryLabel>>,
) -> Result<Option<(Vec<usize>, Vec<usize>)>, Error> {
    let Some(&(future_index, result_is_lhs)) = consumers.get(&result_id) else {
        return Ok(None);
    };
    let future_step = &steps[future_index];
    let sibling_id = if result_is_lhs {
        future_step.rhs()
    } else {
        future_step.lhs()
    };
    let sibling_labels = labels_by_id
        .get(&sibling_id)
        .ok_or_else(|| invalid("future sibling labels missing"))?;
    let mut open_axes = Vec::new();
    let mut contracted_axes = Vec::new();
    for (axis, label) in labels.iter().enumerate() {
        if sibling_labels.contains(label) {
            contracted_axes.push(axis);
        } else {
            open_axes.push(axis);
        }
    }
    let permutation = if result_is_lhs {
        (open_axes, contracted_axes)
    } else {
        (contracted_axes, open_axes)
    };
    if permutation.0.len() == current_codomain_rank
        && permutation
            .0
            .iter()
            .chain(&permutation.1)
            .copied()
            .eq(0..labels.len())
    {
        Ok(None)
    } else {
        Ok(Some(permutation))
    }
}

/// Leg-label order of every input and planned intermediate, mirroring the
/// executor's own tracking (open lhs legs then open rhs legs per step).
fn planned_label_orders(
    ir: &NetworkIR,
    plan: &ContractionPlan,
) -> Result<HashMap<TensorId, Vec<TemporaryLabel>>, Error> {
    let mut labels_by_id: HashMap<TensorId, Vec<TemporaryLabel>> = HashMap::new();
    let mut active: HashMap<TensorId, Vec<TemporaryLabel>> = HashMap::new();
    for node in ir.tensors() {
        let labels = node.labels().to_vec();
        labels_by_id.insert(node.id(), labels.clone());
        active.insert(node.id(), labels);
    }
    for step in plan.steps() {
        let ll = active
            .remove(&step.lhs())
            .ok_or_else(|| invalid("lhs operand already consumed while planning labels"))?;
        let rl = active
            .remove(&step.rhs())
            .ok_or_else(|| invalid("rhs operand already consumed while planning labels"))?;
        let mut labels: Vec<TemporaryLabel> =
            ll.iter().filter(|l| !rl.contains(l)).cloned().collect();
        labels.extend(rl.iter().filter(|l| !ll.contains(l)).cloned());
        labels_by_id.insert(step.result(), labels.clone());
        active.insert(step.result(), labels);
    }
    Ok(labels_by_id)
}

/// One forward pass mapping each tensor id to the single later step that
/// consumes it and whether it is that step's lhs. In a pairwise contraction
/// tree every operand/intermediate is consumed exactly once, so this replaces
/// the per-step `steps[i+1..]` scan (`orient_intermediate_for_next_use`) — the
/// whole orientation pass drops from O(steps²) to O(steps). Resolved once here,
/// analogous to TensorKit's `@tensor` sequence being fixed at macro-expansion.
fn build_consumers(steps: &[ContractionStep]) -> HashMap<TensorId, (usize, bool)> {
    let mut consumers = HashMap::with_capacity(steps.len() * 2);
    for (index, step) in steps.iter().enumerate() {
        consumers.insert(step.lhs(), (index, true));
        consumers.insert(step.rhs(), (index, false));
    }
    consumers
}

mod static_operand_sealed {
    pub trait Sealed {}
}

/// Closed dispatch surface used by the `tensor!` expansion.
///
/// Every operand in one invocation must have the same provider, scalar, and
/// supported storage type. These mismatches are rejected by Rust before a
/// network can be planned or executed:
///
/// ```compile_fail
/// use tenet::core::{SU2FusionRule, U1FusionRule};
/// use tenet::typed::TensorMap;
/// use tenet_network::tensor;
///
/// fn mixed_provider(
///     a: &TensorMap<U1FusionRule, f64>,
///     b: &TensorMap<SU2FusionRule, f64>,
/// ) {
///     let _ = tensor!([i; k] = a[i; j] * b[j; k]);
/// }
/// ```
///
/// ```compile_fail
/// use tenet::core::U1FusionRule;
/// use tenet::prelude::Complex64;
/// use tenet::typed::TensorMap;
/// use tenet_network::tensor;
///
/// fn mixed_scalar(
///     a: &TensorMap<U1FusionRule, f64>,
///     b: &TensorMap<U1FusionRule, Complex64>,
/// ) {
///     let _ = tensor!([i; k] = a[i; j] * b[j; k]);
/// }
/// ```
///
/// ```compile_fail
/// use tenet::core::{Placement, TensorStorage, U1FusionRule};
/// use tenet::typed::TensorMap;
/// use tenet_network::tensor;
///
/// struct DeviceStorage;
/// impl TensorStorage<f64> for DeviceStorage {
///     fn len(&self) -> usize { 0 }
///     fn placement(&self) -> Placement { Placement::Cuda(0) }
/// }
///
/// fn mixed_storage(
///     a: &TensorMap<U1FusionRule, f64>,
///     b: &TensorMap<U1FusionRule, f64, DeviceStorage>,
/// ) {
///     let _ = tensor!([i; k] = a[i; j] * b[j; k]);
/// }
/// ```
///
/// Checked Generic providers use the same Host dispatch surface and preserve
/// their typed provider errors. The macro uses [`StaticTraceNetworkOperand`]
/// only when an operand contains a trace, so ordinary networks do not require
/// pivotal provider data.
#[doc(hidden)]
pub trait StaticNetworkOperand: static_operand_sealed::Sealed + Sized {
    type Error;

    fn contract_static(
        tensors: &[&Self],
        spec: &'static StaticTopologySpec,
    ) -> Result<Self, Self::Error>;
}

/// Static network dispatch with the ordinary typed trace operation available.
#[doc(hidden)]
pub trait StaticTraceNetworkOperand: StaticNetworkOperand {
    fn contract_static_trace(
        tensors: &[&Self],
        spec: &'static StaticTopologySpec,
    ) -> Result<Self, Self::Error>;
}

/// Preserves each macro operand's concrete typed tensor without conversion.
#[doc(hidden)]
pub fn normalize_tensor_operand<R, D, S, O>(tensor: &O) -> &TensorMap<R, D, S>
where
    O: AsRef<TensorMap<R, D, S>> + ?Sized,
{
    tensor.as_ref()
}

/// Typed Host/CUDA dispatch used by `tensor!`.
#[doc(hidden)]
pub fn contract_static_network<T: StaticNetworkOperand>(
    tensors: &[&T],
    spec: &'static StaticTopologySpec,
) -> Result<T, T::Error> {
    T::contract_static(tensors, spec)
}

/// Typed Host/CUDA dispatch used by `tensor!` when an operand contains a trace.
#[doc(hidden)]
pub fn contract_static_trace_network<T: StaticTraceNetworkOperand>(
    tensors: &[&T],
    spec: &'static StaticTopologySpec,
) -> Result<T, T::Error> {
    T::contract_static_trace(tensors, spec)
}

fn validate_static_shape<T>(tensors: &[&T], spec: &StaticTopologySpec) -> Result<(), Error> {
    if tensors.len() != spec.inputs.len() {
        return Err(invalid(format!(
            "network has {} operands but {} tensors were given",
            spec.inputs.len(),
            tensors.len()
        )));
    }
    if spec.conj.len() != spec.inputs.len() || spec.codomain_splits.len() != spec.inputs.len() {
        return Err(invalid(
            "static topology marker lists must match operand count",
        ));
    }
    Ok(())
}

impl<R, D> static_operand_sealed::Sealed for TensorMap<R, D>
where
    R: TypedSectorAdmission + Send + Sync,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar + Send + Sync + 'static,
{
}

impl<R, D> StaticNetworkOperand for TensorMap<R, D>
where
    R: TypedSectorAdmission + Send + Sync,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar + Send + Sync + 'static,
{
    type Error = HostNetworkError<R>;

    fn contract_static(
        tensors: &[&Self],
        spec: &'static StaticTopologySpec,
    ) -> Result<Self, Self::Error> {
        validate_static_shape(tensors, spec)?;
        let codomain_ranks = tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect::<Vec<_>>();
        let optimizer = tensors
            .first()
            .map(|tensor| tensor.runtime().plan_cache_config().optimizer)
            .unwrap_or_default();
        crate::plancache::get_or_plan_static(
            spec,
            tensors,
            &codomain_ranks,
            &optimizer,
            |_| Ok(()),
            || spec.network(),
        )?
        .execute_host(tensors)
    }
}

impl<R, D> StaticTraceNetworkOperand for TensorMap<R, D>
where
    R: TypedSectorAdmission + Send + Sync,
    R::Mode: HostNetworkModeDispatch<R, D> + TypedTensorTraceDispatch<R, D>,
    D: TensorScalar + Send + Sync + 'static,
{
    fn contract_static_trace(
        tensors: &[&Self],
        spec: &'static StaticTopologySpec,
    ) -> Result<Self, Self::Error> {
        validate_static_shape(tensors, spec)?;
        let codomain_ranks = tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect::<Vec<_>>();
        let optimizer = tensors
            .first()
            .map(|tensor| tensor.runtime().plan_cache_config().optimizer)
            .unwrap_or_default();
        let mut inputs = Vec::with_capacity(tensors.len());
        let mut conj = Vec::with_capacity(tensors.len());
        let mut splits = Vec::with_capacity(tensors.len());
        let mut lowered = Vec::with_capacity(tensors.len());
        for (index, tensor) in tensors.iter().enumerate() {
            let written = spec.inputs[index]
                .iter()
                .map(|label| TemporaryLabel::from(*label))
                .collect::<Vec<_>>();
            if !has_intra_operand_pair(&written) {
                inputs.push(written);
                conj.push(spec.conj[index]);
                splits.push(spec.codomain_splits[index]);
                lowered.push(None);
                continue;
            }
            if written.len() != tensor.rank() {
                return Err(invalid(format!(
                    "operand {index} has {} labels but tensor rank {}",
                    written.len(),
                    tensor.rank()
                ))
                .into());
            }
            if let Some(split) = spec.codomain_splits[index] {
                if split != tensor.codomain_rank() {
                    return Err(invalid(format!(
                        "operand {index} puts {split} label(s) before `;` but the tensor's codomain rank is {}",
                        tensor.codomain_rank()
                    ))
                    .into());
                }
            }
            let (value, labels) = if spec.conj[index] {
                (tensor.adjoint()?, rotate(&written, tensor.codomain_rank()))
            } else {
                ((*tensor).clone(), written)
            };
            let (pairs, reduced) = split_trace_pairs(index, &labels)?;
            lowered.push(Some(value.trace_pairs(&pairs)?));
            inputs.push(reduced);
            conj.push(false);
            splits.push(None);
        }
        let reduced = tensors
            .iter()
            .zip(&lowered)
            .map(|(tensor, traced)| traced.as_ref().unwrap_or(tensor))
            .collect::<Vec<_>>();
        crate::plancache::get_or_plan_static(
            spec,
            &reduced,
            &codomain_ranks,
            &optimizer,
            |_| Ok(()),
            || {
                Network::new(
                    inputs,
                    conj,
                    splits,
                    spec.output
                        .iter()
                        .map(|label| TemporaryLabel::from(*label))
                        .collect(),
                    spec.output_codomain_rank,
                )
            },
        )?
        .execute_host(&reduced)
    }
}

#[cfg(feature = "cuda")]
impl<R> static_operand_sealed::Sealed for TensorMap<R, f64, CudaStorage> where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
{
}

#[cfg(feature = "cuda")]
impl<R> StaticNetworkOperand for TensorMap<R, f64, CudaStorage>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    type Error = Error;

    fn contract_static(
        tensors: &[&Self],
        spec: &'static StaticTopologySpec,
    ) -> Result<Self, Error> {
        validate_static_shape(tensors, spec)?;
        let codomain_ranks = tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect::<Vec<_>>();
        let optimizer = tensors
            .first()
            .map(|tensor| tensor.runtime().plan_cache_config().optimizer)
            .unwrap_or_default();
        crate::plancache::get_or_plan_static(
            spec,
            tensors,
            &codomain_ranks,
            &optimizer,
            PlannedNetwork::validate_cuda_plan_structure,
            || spec.network(),
        )?
        .execute_cuda(tensors)
    }
}

#[cfg(feature = "cuda")]
impl<R> StaticTraceNetworkOperand for TensorMap<R, f64, CudaStorage>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn contract_static_trace(
        _tensors: &[&Self],
        _spec: &'static StaticTopologySpec,
    ) -> Result<Self, Self::Error> {
        Err(Error::UnsupportedOnDevice(
            "tensor! intra-operand trace is not supported on CUDA".to_string(),
        ))
    }
}

fn has_intra_operand_pair(labels: &[TemporaryLabel]) -> bool {
    labels
        .iter()
        .enumerate()
        .any(|(i, l)| labels[..i].contains(l))
}

/// Splits an operand's (conj-lowered) labels into intra-operand trace pairs
/// (first occurrence, second occurrence) and the surviving open labels in
/// written order. A label written three or more times on one operand is
/// rejected (the macro already rejects it at compile time; this guards the
/// direct API).
fn split_trace_pairs(
    operand: usize,
    labels: &[TemporaryLabel],
) -> Result<(Vec<(usize, usize)>, Vec<TemporaryLabel>), Error> {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut traced = vec![false; labels.len()];
    for (second, label) in labels.iter().enumerate() {
        let occurrences: Vec<usize> = labels[..second]
            .iter()
            .enumerate()
            .filter(|(_, l)| *l == label)
            .map(|(i, _)| i)
            .collect();
        match occurrences.len() {
            0 => {}
            1 => {
                pairs.push((occurrences[0], second));
                traced[occurrences[0]] = true;
                traced[second] = true;
            }
            _ => {
                return Err(invalid(format!(
                    "label `{label}` appears more than twice on operand {operand}"
                )))
            }
        }
    }
    let reduced = labels
        .iter()
        .enumerate()
        .filter(|&(i, _)| !traced[i])
        .map(|(_, l)| l.clone())
        .collect();
    Ok((pairs, reduced))
}

#[cfg(test)]
mod typed_replay_tests {
    use std::sync::Arc;

    #[cfg(feature = "cuda")]
    use tenet::core::{product_sector, ProductFusionRuleExt};
    use tenet::core::{
        FermionParityFusionRule, FusionRule, SectorLeg, U1FusionRule, U1Irrep, Z2FusionRule,
        Z2Irrep,
    };
    use tenet::prelude::Complex64;
    use tenet::typed::{GradedSpace, SectorSpectrum, TensorMap};

    use super::*;
    use crate::GreedyDenseOptimizer;

    fn label(name: &str) -> TemporaryLabel {
        TemporaryLabel::from(name)
    }

    #[test]
    fn symmetric_slice_binding_checks_adjoint_orientation_leg_and_rule() {
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let codomain =
            GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(2), 1)]).unwrap();
        let domain = GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(-1), 2)]).unwrap();
        let tensor = TensorMap::<U1FusionRule, f64>::rand_with_seed(
            &runtime,
            [&codomain],
            [&domain],
            10_280,
        )
        .unwrap();
        let written = vec![label("p"), label("x")];
        let effective = vec![label("x"), label("p")];
        let network = Network::new(
            vec![written],
            vec![true],
            vec![Some(1)],
            effective.clone(),
            Some(1),
        )
        .unwrap();
        let ir = NetworkIR::from_labels(vec![effective.clone()], effective.clone()).unwrap();
        let order = ContractionPlan::new(1, effective.clone(), Vec::new()).unwrap();
        let cost = DenseCostModel::from_network(&ir, &[DenseTensorInfo::new(vec![2, 1])]).unwrap();
        let dense = SlicedPlan::new(
            order,
            crate::slice_plan_for(
                &ir,
                &ContractionPlan::new(1, effective, Vec::new()).unwrap(),
                &cost,
                &[label("x")],
            ),
        );
        let valid = network
            .lower_symmetric_sliced_plan(&[&tensor], dense)
            .unwrap();
        let bound = network
            .bind_symmetric_sliced_plan(&[&tensor], valid.clone())
            .unwrap();
        assert_eq!(bound.plan(), &valid);

        let index = &valid.slices().indices()[0];
        assert_eq!(index.authority_leg().sectors(), &[U1Irrep::new(-1).into()]);
        assert!(!index.authority_leg().is_dual());
        let forged_leg = SectorLeg::new(index.authority_leg().iter(), true);
        let forged = SymmetricSlicePlan::try_new(
            &ir,
            U1FusionRule.rule_identity(),
            vec![SymmetricSliceSpec::new(
                label("x"),
                index.authority(),
                forged_leg,
                index.pieces().to_vec(),
            )],
        )
        .unwrap();
        assert!(matches!(
            network.bind_symmetric_sliced_plan(
                &[&tensor],
                SymmetricSlicedPlan::new(valid.plan().clone(), forged)
            ),
            Err(SymmetricSliceLowerError::AuthorityLegMismatch { .. })
        ));

        let wrong_rule = SymmetricSlicePlan::try_new(
            &ir,
            Z2FusionRule.rule_identity(),
            vec![SymmetricSliceSpec::new(
                label("x"),
                index.authority(),
                index.authority_leg().clone(),
                index.pieces().to_vec(),
            )],
        )
        .unwrap();
        assert!(matches!(
            network.bind_symmetric_sliced_plan(
                &[&tensor],
                SymmetricSlicedPlan::new(valid.plan().clone(), wrong_rule)
            ),
            Err(SymmetricSliceLowerError::RuleMismatch { .. })
        ));
    }

    #[test]
    fn internal_symmetric_slices_match_unsliced_multisector_and_meter_cold_warm() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let bond =
            GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(0), 1), (U1Irrep::new(1), 3)])
                .unwrap();
        let make = |shift| {
            TensorMap::from_block_fn(&runtime, [&bond], [&bond], move |trees, indices| {
                shift
                    + trees.codomain_uncoupled()[0].charge() as f64
                    + indices[0] as f64
                    + 2.0 * indices[1] as f64
            })
            .unwrap()
        };
        let a = make(1.0);
        let b = make(2.0);
        let c = make(3.0);
        let inputs = vec![
            vec![label("a"), label("x")],
            vec![label("x"), label("y")],
            vec![label("y"), label("c")],
        ];
        let output = vec![label("a"), label("c")];
        let network = Network::new(
            inputs.clone(),
            vec![false; 3],
            vec![Some(1); 3],
            output.clone(),
            Some(1),
        )
        .unwrap();
        let tensors = [&a, &b, &c];
        let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
        let expected = planned.execute(&tensors).unwrap();
        let ir = NetworkIR::from_labels(inputs, output).unwrap();
        let cost = DenseCostModel::from_network(
            &ir,
            &[
                DenseTensorInfo::new(vec![4, 4]),
                DenseTensorInfo::new(vec![4, 4]),
                DenseTensorInfo::new(vec![4, 4]),
            ],
        )
        .unwrap();
        let dense = SlicedPlan::new(
            planned.plan().clone(),
            crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("x"), label("y")]),
        );
        let sliced = network
            .lower_symmetric_sliced_plan(&tensors, dense)
            .unwrap();
        assert_eq!(sliced.slices().nslices(), 16);

        let (cold, cold_peak) = network
            .execute_symmetric_sliced(&tensors, sliced.clone(), usize::MAX)
            .unwrap();
        let (warm, warm_peak) = network
            .execute_symmetric_sliced(&tensors, sliced.clone(), usize::MAX)
            .unwrap();
        for actual in [&cold, &warm] {
            assert_eq!(actual.block_count(), expected.block_count());
            for index in 0..actual.block_count() {
                assert_eq!(actual.block(index).unwrap(), expected.block(index).unwrap());
            }
            assert_eq!(actual.data(), expected.data());
        }
        assert_eq!(cold_peak, warm_peak);
        assert!(cold_peak > 0);

        let mut saw_late_limit = false;
        for ceiling in 0..cold_peak {
            SYMMETRIC_SLICE_COMPLETED_JOBS.store(0, Ordering::SeqCst);
            if matches!(
                network.execute_symmetric_sliced(&tensors, sliced.clone(), ceiling),
                Err(SymmetricSliceExecutionError::WorkspaceLimitExceeded { .. })
            ) && SYMMETRIC_SLICE_COMPLETED_JOBS.load(Ordering::SeqCst) > 0
            {
                saw_late_limit = true;
                break;
            }
        }
        assert!(
            saw_late_limit,
            "fixture must reject after a completed partial"
        );
    }

    #[test]
    fn internal_symmetric_slice_legal_empty_returns_unsliced_zero_layout() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let open =
            GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
        let empty = GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(0), 0)]).unwrap();
        let a = TensorMap::<_, f64>::zeros(&runtime, [&open], [&empty]).unwrap();
        let b = TensorMap::<_, f64>::zeros(&runtime, [&empty], [&open]).unwrap();
        let inputs = vec![vec![label("a"), label("x")], vec![label("x"), label("c")]];
        let output = vec![label("a"), label("c")];
        let network = Network::new(
            inputs.clone(),
            vec![false; 2],
            vec![Some(1); 2],
            output.clone(),
            Some(1),
        )
        .unwrap();
        let tensors = [&a, &b];
        let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
        let expected = planned.execute(&tensors).unwrap();
        let ir = NetworkIR::from_labels(inputs, output).unwrap();
        let authority = ir.edge(&label("x")).unwrap().occurrences()[0];
        let leg = typed_effective_spaces(&a, false).unwrap()[authority.axis()]
            .network_sector_leg()
            .clone();
        let slices = SymmetricSlicePlan::try_new(
            &ir,
            U1FusionRule.rule_identity(),
            vec![SymmetricSliceSpec::new(
                label("x"),
                authority,
                leg,
                Vec::new(),
            )],
        )
        .unwrap();
        assert_eq!(slices.nslices(), 0);
        let (actual, peak) = network
            .execute_symmetric_sliced(
                &tensors,
                SymmetricSlicedPlan::new(planned.plan().clone(), slices),
                usize::MAX,
            )
            .unwrap();
        assert_eq!(actual.block_count(), expected.block_count());
        assert_eq!(actual.data(), expected.data());
        assert_eq!(peak, actual.network_owned_payload().unwrap().1);
    }

    #[test]
    fn internal_slice_maps_nonselfdual_partner_from_authority_leg() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let plus = GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(1), 2)]).unwrap();
        let a = TensorMap::from_block_fn(&runtime, [&plus], [&plus], |_, ij| {
            (1 + ij[0] + 2 * ij[1]) as f64
        })
        .unwrap();
        let b = TensorMap::from_block_fn(&runtime, [&plus], [&plus], |_, ij| {
            (2 + 2 * ij[0] + ij[1]) as f64
        })
        .unwrap();
        let inputs = vec![vec![label("a"), label("x")], vec![label("x"), label("c")]];
        let output = vec![label("a"), label("c")];
        let network = Network::new(
            inputs.clone(),
            vec![false; 2],
            vec![Some(1); 2],
            output.clone(),
            Some(1),
        )
        .unwrap();
        let tensors = [&a, &b];
        let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
        let expected = planned.execute(&tensors).unwrap();
        let ir = NetworkIR::from_labels(inputs, output).unwrap();
        let cost = DenseCostModel::from_network(
            &ir,
            &[
                DenseTensorInfo::new(vec![2, 2]),
                DenseTensorInfo::new(vec![2, 2]),
            ],
        )
        .unwrap();
        let dense = SlicedPlan::new(
            planned.plan().clone(),
            crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("x")]),
        );
        let sliced = network
            .lower_symmetric_sliced_plan(&tensors, dense)
            .unwrap();
        let authority = &sliced.slices().indices()[0];
        assert_eq!(
            authority.authority_leg().sectors(),
            &[U1Irrep::new(-1).into()]
        );
        let (actual, _) = network
            .execute_symmetric_sliced(&tensors, sliced, usize::MAX)
            .unwrap();
        assert_eq!(actual.data(), expected.data());
    }

    #[test]
    fn output_slice_rejection_precedes_compact_input_preflight() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(0), 2)]).unwrap();
        let compact = TensorMap::<_, f64>::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, 1.0],
            }],
        )
        .unwrap();
        assert!(compact.network_has_compact_payload());
        let labels = vec![label("a"), label("b")];
        let network = Network::new(
            vec![labels.clone()],
            vec![false],
            vec![Some(1)],
            labels.clone(),
            Some(1),
        )
        .unwrap();
        let ir = NetworkIR::from_labels(vec![labels.clone()], labels.clone()).unwrap();
        let plan = ContractionPlan::new(1, labels, Vec::new()).unwrap();
        let authority = ir.edge(&label("a")).unwrap().occurrences()[0];
        let effective = typed_effective_spaces(&compact, false).unwrap();
        let authority_leg = effective[authority.axis()].network_sector_leg().clone();
        let sector = authority_leg.sectors()[0];
        let slices = SymmetricSlicePlan::try_new(
            &ir,
            U1FusionRule.rule_identity(),
            vec![SymmetricSliceSpec::new(
                label("a"),
                authority,
                authority_leg,
                vec![crate::SectorSlice::new(
                    sector,
                    crate::DegeneracyRange::new(0, 2).unwrap(),
                )],
            )],
        )
        .unwrap();
        assert!(matches!(
            network.execute_symmetric_sliced(
                &[&compact],
                SymmetricSlicedPlan::new(plan, slices),
                usize::MAX,
            ),
            Err(SymmetricSliceExecutionError::OutputSlice { .. })
        ));
        let empty =
            SymmetricSlicePlan::try_new(&ir, U1FusionRule.rule_identity(), Vec::new()).unwrap();
        assert!(matches!(
            network.execute_symmetric_sliced(
                &[&compact],
                SymmetricSlicedPlan::new(
                    ContractionPlan::new(1, vec![label("a"), label("b")], Vec::new()).unwrap(),
                    empty,
                ),
                usize::MAX,
            ),
            Err(SymmetricSliceExecutionError::Tensor(_))
        ));
    }

    fn assert_fermionic_internal_slice<D>()
    where
        D: TensorScalar + PartialEq + std::fmt::Debug,
    {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(FermionParityFusionRule);
        let odd = GradedSpace::try_new_with_arc(provider, [(Z2Irrep::ODD, 2)]).unwrap();
        let a = TensorMap::from_block_fn(&runtime, [&odd], [&odd], |_, ij| {
            D::from_real((1 + ij[0] + 2 * ij[1]) as f64)
        })
        .unwrap();
        let b = TensorMap::from_block_fn(&runtime, [&odd], [&odd], |_, ij| {
            D::from_real((2 + 2 * ij[0] + ij[1]) as f64)
        })
        .unwrap();
        let inputs = vec![vec![label("a"), label("x")], vec![label("x"), label("c")]];
        let output = vec![label("a"), label("c")];
        let network = Network::new(
            inputs.clone(),
            vec![false; 2],
            vec![Some(1); 2],
            output.clone(),
            Some(1),
        )
        .unwrap();
        let tensors = [&a, &b];
        let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
        let expected = planned.execute(&tensors).unwrap();
        let ir = NetworkIR::from_labels(inputs, output).unwrap();
        let cost = DenseCostModel::from_network(
            &ir,
            &[
                DenseTensorInfo::new(vec![2, 2]),
                DenseTensorInfo::new(vec![2, 2]),
            ],
        )
        .unwrap();
        let dense = SlicedPlan::new(
            planned.plan().clone(),
            crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("x")]),
        );
        let sliced = network
            .lower_symmetric_sliced_plan(&tensors, dense)
            .unwrap();
        let (actual, _) = network
            .execute_symmetric_sliced(&tensors, sliced, usize::MAX)
            .unwrap();
        assert_eq!(actual.data(), expected.data());
    }

    #[test]
    fn fermionic_internal_slices_match_unsliced_real_and_complex() {
        assert_fermionic_internal_slice::<f64>();
        assert_fermionic_internal_slice::<Complex64>();
    }

    #[test]
    fn fermionic_complex_conjugated_operand_internal_slice_matches_unsliced() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(FermionParityFusionRule);
        let odd = GradedSpace::try_new_with_arc(provider, [(Z2Irrep::ODD, 2)]).unwrap();
        let a = TensorMap::from_block_fn(&runtime, [&odd], [&odd], |_, ij| {
            Complex64::new(
                (1 + ij[0] + 2 * ij[1]) as f64,
                (2 + 3 * ij[0] + ij[1]) as f64,
            )
        })
        .unwrap();
        let b = TensorMap::from_block_fn(&runtime, [&odd], [&odd], |_, ij| {
            Complex64::new(
                (2 + 2 * ij[0] + ij[1]) as f64,
                -((1 + ij[0] + 4 * ij[1]) as f64),
            )
        })
        .unwrap();

        // The first operand is written as [x; a]. Network lowering rotates it
        // to the effective adjoint order [a; x], so x is the sliced contracted
        // leg. Nonzero imaginary parts make a missed conjugation observable.
        let written_inputs = vec![vec![label("x"), label("a")], vec![label("x"), label("c")]];
        let effective_inputs = vec![vec![label("a"), label("x")], vec![label("x"), label("c")]];
        let output = vec![label("a"), label("c")];
        let network = Network::new(
            written_inputs,
            vec![true, false],
            vec![Some(1); 2],
            output.clone(),
            Some(1),
        )
        .unwrap();
        let tensors = [&a, &b];
        let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
        let expected = planned.execute(&tensors).unwrap();
        let ir = NetworkIR::from_labels(effective_inputs, output).unwrap();
        let cost = DenseCostModel::from_network(
            &ir,
            &[
                DenseTensorInfo::new(vec![2, 2]),
                DenseTensorInfo::new(vec![2, 2]),
            ],
        )
        .unwrap();
        let dense = SlicedPlan::new(
            planned.plan().clone(),
            crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("x")]),
        );
        let sliced = network
            .lower_symmetric_sliced_plan(&tensors, dense)
            .unwrap();
        let (actual, _) = network
            .execute_symmetric_sliced(&tensors, sliced, usize::MAX)
            .unwrap();
        assert_eq!(actual.data(), expected.data());
        assert!(actual.data().iter().any(|value| value.im != 0.0));
    }

    fn crossed_plan() -> PlannedNetwork {
        let inputs = vec![
            vec![label("a"), label("b")],
            vec![label("b"), label("c")],
            vec![label("a"), label("d")],
        ];
        let output = vec![label("c"), label("d")];
        let ir = NetworkIR::from_labels(inputs.clone(), output.clone()).unwrap();
        let plan = ContractionPlan::new(
            3,
            output,
            vec![
                ContractionStep::new(
                    TensorId::new(0),
                    TensorId::new(1),
                    TensorId::new(3),
                    0,
                    vec![label("a"), label("c")],
                ),
                ContractionStep::new(
                    TensorId::new(3),
                    TensorId::new(2),
                    TensorId::new(4),
                    0,
                    vec![label("c"), label("d")],
                ),
            ],
        )
        .unwrap();
        let schedule = compile_schedule(&ir, &plan, Some(1), &[1, 1, 1]).unwrap();
        assert_eq!(
            schedule.steps[0].result_output_axes.as_deref(),
            Some(&[1, 0][..])
        );
        PlannedNetwork {
            owner_token: NEXT_PLAN_OWNER_TOKEN.fetch_add(1, Ordering::Relaxed),
            plan,
            conj: vec![false; 3],
            input_codomain_ranks: vec![1; 3],
            schedule,
            #[cfg(feature = "cuda")]
            cuda_direct: false,
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_schedule_preflight_accepts_only_the_complete_direct_subset() {
        let runtime = Runtime::builder().build().unwrap();
        let space =
            GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
        let tensors = (0..3)
            .map(|seed| {
                TensorMap::<U1FusionRule, f64>::rand_with_seed(
                    &runtime,
                    [&space],
                    [&space],
                    748_200 + seed,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let labels = |names: &[&str]| names.iter().copied().map(label).collect::<Vec<_>>();
        let pair = Network::new(
            vec![labels(&["a", "b"]), labels(&["b", "c"])],
            vec![false; 2],
            vec![Some(1); 2],
            labels(&["a", "c"]),
            Some(1),
        )
        .unwrap()
        .plan(&[&tensors[0], &tensors[1]], &GreedyDenseOptimizer)
        .unwrap();
        assert!(pair.preflight_cuda_schedule(&[(2, 1), (2, 1)]).is_ok());

        let chain_network = Network::new(
            vec![
                labels(&["a", "b"]),
                labels(&["b", "c"]),
                labels(&["c", "d"]),
            ],
            vec![false; 3],
            vec![Some(1); 3],
            labels(&["a", "d"]),
            Some(1),
        )
        .unwrap();
        let chain_order = ContractionPlan::new(
            3,
            labels(&["a", "d"]),
            vec![
                ContractionStep::new(
                    TensorId::new(0),
                    TensorId::new(1),
                    TensorId::new(3),
                    0,
                    labels(&["a", "c"]),
                ),
                ContractionStep::new(
                    TensorId::new(3),
                    TensorId::new(2),
                    TensorId::new(4),
                    0,
                    labels(&["a", "d"]),
                ),
            ],
        )
        .unwrap();
        let chain = chain_network
            .plan_with(&[&tensors[0], &tensors[1], &tensors[2]], chain_order)
            .unwrap();
        assert!(
            chain
                .preflight_cuda_schedule(&[(2, 1), (2, 1), (2, 1)])
                .is_ok(),
            "steps={:?}",
            chain
                .schedule
                .steps
                .iter()
                .map(|step| (
                    &step.lhs_contract_axes,
                    &step.rhs_contract_axes,
                    &step.contract_output_axes,
                    &step.result_permutation
                ))
                .collect::<Vec<_>>()
        );

        let mut late_invalid = chain;
        late_invalid.schedule.steps[1].result_permutation = Some((vec![1], vec![0]));
        assert!(late_invalid
            .preflight_cuda_schedule(&[(2, 1), (2, 1), (2, 1)])
            .is_err());
        assert!(crossed_plan()
            .preflight_cuda_schedule(&[(2, 1), (2, 1), (2, 1)])
            .is_err());

        let final_permutation = Network::new(
            vec![labels(&["a", "b"]), labels(&["b", "c"])],
            vec![false; 2],
            vec![Some(1); 2],
            labels(&["c", "a"]),
            Some(1),
        )
        .unwrap()
        .plan(&[&tensors[0], &tensors[1]], &GreedyDenseOptimizer)
        .unwrap();
        assert!(final_permutation
            .preflight_cuda_schedule(&[(2, 1), (2, 1)])
            .is_err());

        let single = Network::new(
            vec![labels(&["a", "b"])],
            vec![false],
            vec![Some(1)],
            labels(&["a", "b"]),
            Some(1),
        )
        .unwrap()
        .plan(&[&tensors[0]], &GreedyDenseOptimizer)
        .unwrap();
        assert!(single.preflight_cuda_schedule(&[(2, 1)]).is_ok());

        let ket = TensorMap::<U1FusionRule, f64>::rand_with_seed(
            &runtime,
            std::iter::empty::<&GradedSpace<U1FusionRule>>(),
            [&space],
            748_210,
        )
        .unwrap();
        let bra = TensorMap::rand_with_seed(
            &runtime,
            [&space],
            std::iter::empty::<&GradedSpace<U1FusionRule>>(),
            748_211,
        )
        .unwrap();
        let scalar = Network::new(
            vec![labels(&["k"]), labels(&["k"])],
            vec![false; 2],
            vec![Some(0), Some(1)],
            vec![],
            Some(0),
        )
        .unwrap()
        .plan(&[&ket, &bra], &GreedyDenseOptimizer)
        .unwrap();
        assert!(scalar.preflight_cuda_schedule(&[(1, 0), (1, 1)]).is_ok());
    }

    #[cfg(feature = "cuda")]
    fn assert_asymmetric_cuda_plan_parity<R>(
        runtime: &Runtime,
        x0: &GradedSpace<R>,
        x1: &GradedSpace<R>,
        y: &GradedSpace<R>,
        z0: &GradedSpace<R>,
        z1: &GradedSpace<R>,
        seed: u64,
    ) where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
    {
        let a = TensorMap::<R, f64>::rand_with_seed(runtime, [x0, x1], [y], seed).unwrap();
        let b = TensorMap::rand_with_seed(runtime, [x1], [z0, z1], seed + 1).unwrap();
        let a_cuda = a.to_cuda().unwrap();
        let b_cuda = b.to_cuda().unwrap();
        let network = Network::new(
            vec![
                vec![label("i0"), label("i1"), label("j")],
                vec![label("i1"), label("k0"), label("k1")],
            ],
            vec![true, false],
            vec![Some(2), Some(1)],
            vec![label("j"), label("i0"), label("k0"), label("k1")],
            Some(2),
        )
        .unwrap();
        let host_refs = [&a, &b];
        let cuda_refs = [&a_cuda, &b_cuda];
        let host = network.plan(&host_refs, &GreedyDenseOptimizer).unwrap();
        let cuda = network.plan(&cuda_refs, &GreedyDenseOptimizer).unwrap();

        assert_eq!(host.plan.steps(), cuda.plan.steps());
        assert_eq!(host.schedule.input_ranks, cuda.schedule.input_ranks);
        assert_eq!(
            host.schedule.contracted_input_pairs,
            cuda.schedule.contracted_input_pairs
        );
        for (host_step, cuda_step) in host.schedule.steps.iter().zip(&cuda.schedule.steps) {
            assert_eq!(host_step.lhs_contract_axes, cuda_step.lhs_contract_axes);
            assert_eq!(host_step.rhs_contract_axes, cuda_step.rhs_contract_axes);
            assert_eq!(
                host_step.contract_output_axes,
                cuda_step.contract_output_axes
            );
            assert_eq!(host_step.result_permutation, cuda_step.result_permutation);
        }
        assert_eq!(
            host.schedule.final_permutation,
            cuda.schedule.final_permutation
        );
        for ((host_tensor, cuda_tensor), conj) in [(&a, &a_cuda), (&b, &b_cuda)]
            .into_iter()
            .zip([true, false])
        {
            assert_eq!(
                typed_effective_spaces(host_tensor, conj).unwrap(),
                typed_effective_spaces(cuda_tensor, conj).unwrap()
            );
            assert_eq!(
                host_tensor.leg_dims().unwrap(),
                cuda_tensor.leg_dims().unwrap()
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn greedy_three_tensor_chain_has_a_direct_cuda_schedule() {
        let runtime = Runtime::builder().build().unwrap();
        let space =
            GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
        let tensors = (0..3)
            .map(|seed| {
                TensorMap::<U1FusionRule, f64>::rand_with_seed(
                    &runtime,
                    [&space],
                    [&space],
                    750_300 + seed,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let network = Network::new(
            vec![
                vec![label("a"), label("b")],
                vec![label("b"), label("c")],
                vec![label("c"), label("d")],
            ],
            vec![false; 3],
            vec![Some(1); 3],
            vec![label("a"), label("d")],
            Some(1),
        )
        .unwrap();
        let refs = [&tensors[0], &tensors[1], &tensors[2]];
        let planned = network.plan(&refs, &GreedyDenseOptimizer).unwrap();
        assert!(
            planned.validate_cuda_plan_structure().is_ok(),
            "greedy steps: {:?}",
            planned.plan().steps()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn reversed_final_output_is_a_valid_but_nondirect_cuda_schedule() {
        let runtime = Runtime::builder().build().unwrap();
        let space =
            GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
        let a =
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&space], [&space], 750_310)
                .unwrap();
        let b =
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&space], [&space], 750_311)
                .unwrap();
        let network = Network::new(
            vec![vec![label("i"), label("j")], vec![label("j"), label("k")]],
            vec![false; 2],
            vec![Some(1); 2],
            vec![label("k"), label("i")],
            Some(1),
        )
        .unwrap();
        let planned = network.plan(&[&a, &b], &GreedyDenseOptimizer).unwrap();
        assert!(planned.validate_cuda_plan_structure().is_err());
        assert!(planned.schedule.final_permutation.is_some());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_rejections_happen_before_the_first_network_contract() {
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        let u1_rule = Arc::new(U1FusionRule);
        let u1_x0 =
            GradedSpace::try_new_with_arc(Arc::clone(&u1_rule), [(U1Irrep::new(2), 2)]).unwrap();
        let u1_x1 = GradedSpace::try_new_with_arc(Arc::clone(&u1_rule), [(U1Irrep::new(-1), 1)])
            .unwrap()
            .try_dual()
            .unwrap();
        let u1_y =
            GradedSpace::try_new_with_arc(Arc::clone(&u1_rule), [(U1Irrep::new(1), 3)]).unwrap();
        let u1_z0 =
            GradedSpace::try_new_with_arc(Arc::clone(&u1_rule), [(U1Irrep::new(-2), 2)]).unwrap();
        let u1_z1 = GradedSpace::try_new_with_arc(u1_rule, [(U1Irrep::new(0), 1)]).unwrap();
        assert_asymmetric_cuda_plan_parity(
            &runtime, &u1_x0, &u1_x1, &u1_y, &u1_z0, &u1_z1, 748_210,
        );

        let product_rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
        let product_x0 = GradedSpace::try_new_with_arc(
            Arc::clone(&product_rule),
            [(product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1)],
        )
        .unwrap();
        let product_x1 = GradedSpace::try_new_with_arc(
            Arc::clone(&product_rule),
            [(product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 2)],
        )
        .unwrap()
        .try_dual()
        .unwrap();
        let product_y = GradedSpace::try_new_with_arc(
            Arc::clone(&product_rule),
            [(product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2)],
        )
        .unwrap();
        let product_z0 = GradedSpace::try_new_with_arc(
            Arc::clone(&product_rule),
            [(product_sector(Z2Irrep::EVEN, U1Irrep::new(-2)), 1)],
        )
        .unwrap();
        let product_z1 = GradedSpace::try_new_with_arc(
            product_rule,
            [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 1)],
        )
        .unwrap();
        assert_asymmetric_cuda_plan_parity(
            &runtime,
            &product_x0,
            &product_x1,
            &product_y,
            &product_z0,
            &product_z1,
            748_212,
        );

        let provider = Arc::new(U1FusionRule);
        let good =
            GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
        let bad =
            GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(1), 2)]).unwrap();
        let host_tensors = (0..3)
            .map(|seed| {
                TensorMap::<U1FusionRule, f64>::rand_with_seed(
                    &runtime,
                    [&good],
                    [&good],
                    748_220 + seed,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let tensors = host_tensors
            .iter()
            .map(|tensor| tensor.to_cuda().unwrap())
            .collect::<Vec<_>>();
        let labels = |names: &[&str]| names.iter().copied().map(label).collect::<Vec<_>>();
        let network = Network::new(
            vec![
                labels(&["a", "b"]),
                labels(&["b", "c"]),
                labels(&["c", "d"]),
            ],
            vec![false; 3],
            vec![Some(1); 3],
            labels(&["a", "d"]),
            Some(1),
        )
        .unwrap();
        let refs = [&tensors[0], &tensors[1], &tensors[2]];
        let canonical_order = || {
            ContractionPlan::new(
                3,
                labels(&["a", "d"]),
                vec![
                    ContractionStep::new(
                        TensorId::new(0),
                        TensorId::new(1),
                        TensorId::new(3),
                        0,
                        labels(&["a", "c"]),
                    ),
                    ContractionStep::new(
                        TensorId::new(3),
                        TensorId::new(2),
                        TensorId::new(4),
                        0,
                        labels(&["a", "d"]),
                    ),
                ],
            )
            .unwrap()
        };
        let host_refs = [&host_tensors[0], &host_tensors[1], &host_tensors[2]];
        let host_plan = network.plan_with(&host_refs, canonical_order()).unwrap();
        let cuda_plan = network.plan_with(&refs, canonical_order()).unwrap();
        assert_eq!(host_plan.plan.steps(), cuda_plan.plan.steps());
        assert_eq!(
            host_plan.schedule.input_ranks,
            cuda_plan.schedule.input_ranks
        );
        assert_eq!(
            host_plan.schedule.contracted_input_pairs,
            cuda_plan.schedule.contracted_input_pairs
        );
        for (host, cuda) in host_plan
            .schedule
            .steps
            .iter()
            .zip(&cuda_plan.schedule.steps)
        {
            assert_eq!(host.lhs_contract_axes, cuda.lhs_contract_axes);
            assert_eq!(host.rhs_contract_axes, cuda.rhs_contract_axes);
            assert_eq!(host.contract_output_axes, cuda.contract_output_axes);
            assert_eq!(host.result_permutation, cuda.result_permutation);
        }
        assert_eq!(
            host_plan.schedule.final_permutation,
            cuda_plan.schedule.final_permutation
        );
        assert_eq!(
            typed_effective_spaces(&host_tensors[0], true).unwrap(),
            typed_effective_spaces(&tensors[0], true).unwrap()
        );
        assert_eq!(
            host_tensors[0].leg_dims().unwrap(),
            tensors[0].leg_dims().unwrap()
        );
        let mut split_changing = network.plan_with(&refs, canonical_order()).unwrap();
        split_changing.schedule.steps[1].result_permutation = Some((vec![0, 1], vec![]));
        CUDA_NETWORK_CONTRACT_CALLS.store(0, Ordering::Relaxed);
        assert!(split_changing.execute_cuda(&refs).is_err());
        assert_eq!(CUDA_NETWORK_CONTRACT_CALLS.load(Ordering::Relaxed), 0);

        let mut nonidentity_pab = network.plan_with(&refs, canonical_order()).unwrap();
        nonidentity_pab.schedule.steps[1].contract_output_axes = vec![1, 0];
        CUDA_NETWORK_CONTRACT_CALLS.store(0, Ordering::Relaxed);
        assert!(nonidentity_pab.execute_cuda(&refs).is_err());
        assert_eq!(CUDA_NETWORK_CONTRACT_CALLS.load(Ordering::Relaxed), 0);

        let mut final_permutation = network.plan_with(&refs, canonical_order()).unwrap();
        final_permutation.schedule.final_permutation = Some((vec![1], vec![0]));
        CUDA_NETWORK_CONTRACT_CALLS.store(0, Ordering::Relaxed);
        assert!(final_permutation.execute_cuda(&refs).is_err());
        assert_eq!(CUDA_NETWORK_CONTRACT_CALLS.load(Ordering::Relaxed), 0);

        let bad_rhs =
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&bad], [&good], 748_230)
                .unwrap()
                .to_cuda()
                .unwrap();
        let pair = Network::new(
            vec![labels(&["a", "b"]), labels(&["b", "c"])],
            vec![false; 2],
            vec![Some(1); 2],
            labels(&["a", "c"]),
            Some(1),
        )
        .unwrap();
        let planned = pair
            .plan(&[&tensors[0], &tensors[1]], &GreedyDenseOptimizer)
            .unwrap();
        CUDA_NETWORK_CONTRACT_CALLS.store(0, Ordering::Relaxed);
        assert!(planned.execute_cuda(&[&tensors[0], &bad_rhs]).is_err());
        assert_eq!(CUDA_NETWORK_CONTRACT_CALLS.load(Ordering::Relaxed), 0);

        let other_runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        let other_space =
            GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
        let foreign = TensorMap::<U1FusionRule, f64>::rand_with_seed(
            &other_runtime,
            [&other_space],
            [&other_space],
            748_231,
        )
        .unwrap()
        .to_cuda()
        .unwrap();
        CUDA_NETWORK_CONTRACT_CALLS.store(0, Ordering::Relaxed);
        assert!(planned.execute_cuda(&[&tensors[0], &foreign]).is_err());
        assert_eq!(CUDA_NETWORK_CONTRACT_CALLS.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn typed_crossed_schedule_reuses_the_actual_first_step_destination() {
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let other_provider = Arc::new(U1FusionRule);
        let space = |provider: &Arc<U1FusionRule>, degeneracy| {
            GradedSpace::try_new_with_arc(Arc::clone(provider), [(U1Irrep::new(0), degeneracy)])
                .unwrap()
        };
        let left = space(&provider, 9);
        let bond = space(&provider, 8);
        let right = space(&provider, 10);
        let tail = space(&provider, 11);
        let left_dual = left.try_dual().unwrap();
        let a =
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&left], [&bond], 1).unwrap();
        let b = TensorMap::rand_with_seed(&runtime, [&bond], [&right], 2).unwrap();
        let c = TensorMap::rand_with_seed(&runtime, [&left_dual], [&tail], 3).unwrap();
        let planned = crossed_plan();
        let mut workspace = NetworkExecutionWorkspace::default();
        let refs = [&a, &b, &c];
        let oracle = a
            .contract(&b, &[1], &[0], &[1, 0])
            .unwrap()
            .contract(&c, &[1], &[0], &[0, 1])
            .unwrap();
        drop(
            planned
                .execute_with_workspace(&refs, &mut workspace)
                .unwrap(),
        );
        let before = workspace.intermediates[0]
            .oriented
            .as_ref()
            .map(|tensor| (tensor.data().as_ptr(), tensor.data().len()))
            .unwrap();
        let output = planned
            .execute_with_workspace(&refs, &mut workspace)
            .unwrap();
        let after = workspace.intermediates[0]
            .oriented
            .as_ref()
            .map(|tensor| (tensor.data().as_ptr(), tensor.data().len()))
            .unwrap();
        assert_eq!(before, after);
        assert_ne!(after.0, output.data().as_ptr());
        assert_eq!(output.data(), oracle.data());

        let other_bond = space(&other_provider, 8);
        let other_right = space(&other_provider, 10);
        let rhs_drift =
            TensorMap::rand_with_seed(&runtime, [&other_bond], [&other_right], 4).unwrap();
        drop(
            planned
                .execute_with_workspace(&[&a, &rhs_drift, &c], &mut workspace)
                .unwrap(),
        );
        let rhs_only = workspace.intermediates[0].oriented.as_ref().unwrap();
        assert_eq!((rhs_only.data().as_ptr(), rhs_only.data().len()), before);
        assert!(std::ptr::eq(rhs_only.provider(), provider.as_ref()));

        let retained = rhs_only.clone();
        assert!(planned
            .execute_with_workspace(&refs[..2], &mut workspace)
            .is_err());
        assert_eq!(
            workspace.intermediates[0]
                .oriented
                .as_ref()
                .unwrap()
                .data()
                .as_ptr(),
            retained.data().as_ptr()
        );
        let bad_bond = space(&provider, 7);
        let bad_rhs = TensorMap::rand_with_seed(&runtime, [&bad_bond], [&right], 5).unwrap();
        assert!(planned
            .execute_with_workspace(&[&a, &bad_rhs, &c], &mut workspace)
            .is_err());
        assert_eq!(
            workspace.intermediates[0]
                .oriented
                .as_ref()
                .unwrap()
                .data()
                .as_ptr(),
            retained.data().as_ptr()
        );

        let other_left = space(&other_provider, 9);
        let lhs_drift =
            TensorMap::rand_with_seed(&runtime, [&other_left], [&other_bond], 6).unwrap();
        drop(
            planned
                .execute_with_workspace(&[&lhs_drift, &b, &c], &mut workspace)
                .unwrap(),
        );
        let replaced = workspace.intermediates[0].oriented.as_ref().unwrap();
        assert_ne!(replaced.data().as_ptr(), retained.data().as_ptr());
        assert!(std::ptr::eq(replaced.provider(), other_provider.as_ref()));

        let replaced = replaced.clone();
        let wide_left = space(&other_provider, 12);
        let other_tail = space(&other_provider, 11);
        let wide_c =
            TensorMap::rand_with_seed(&runtime, [&wide_left.try_dual().unwrap()], [&other_tail], 7)
                .unwrap();
        let wide_a = TensorMap::rand_with_seed(&runtime, [&wide_left], [&other_bond], 8).unwrap();
        drop(
            planned
                .execute_with_workspace(&[&wide_a, &rhs_drift, &wide_c], &mut workspace)
                .unwrap(),
        );
        let widened = workspace.intermediates[0].oriented.as_ref().unwrap();
        assert_ne!(widened.data().len(), replaced.data().len());

        let widened = widened.clone();
        let other_runtime = Runtime::builder().build().unwrap();
        let foreign_a =
            TensorMap::rand_with_seed(&other_runtime, [&other_left], [&other_bond], 9).unwrap();
        let foreign_b =
            TensorMap::rand_with_seed(&other_runtime, [&other_bond], [&other_right], 10).unwrap();
        let foreign_c = TensorMap::rand_with_seed(
            &other_runtime,
            [&other_left.try_dual().unwrap()],
            [&other_tail],
            11,
        )
        .unwrap();
        drop(
            planned
                .execute_with_workspace(&[&foreign_a, &foreign_b, &foreign_c], &mut workspace)
                .unwrap(),
        );
        assert_ne!(
            workspace.intermediates[0]
                .oriented
                .as_ref()
                .unwrap()
                .data()
                .as_ptr(),
            widened.data().as_ptr()
        );

        let previous = workspace.intermediates[0]
            .oriented
            .as_ref()
            .unwrap()
            .clone();
        let other_plan = crossed_plan();
        drop(
            other_plan
                .execute_with_workspace(&[&foreign_a, &foreign_b, &foreign_c], &mut workspace)
                .unwrap(),
        );
        assert_eq!(workspace.owner_token, Some(other_plan.owner_token));
        assert_ne!(
            workspace.intermediates[0]
                .oriented
                .as_ref()
                .unwrap()
                .data()
                .as_ptr(),
            previous.data().as_ptr()
        );
    }

    #[test]
    fn typed_replay_restores_buffers_after_injected_failures() {
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let space = |degeneracy| {
            GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(0), degeneracy)])
                .unwrap()
        };
        let left = space(5);
        let bond = space(4);
        let right = space(6);
        let tail = space(7);
        let a =
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&left], [&bond], 11).unwrap();
        let b = TensorMap::rand_with_seed(&runtime, [&bond], [&right], 12).unwrap();
        let c =
            TensorMap::rand_with_seed(&runtime, [&left.try_dual().unwrap()], [&tail], 13).unwrap();
        let refs = [&a, &b, &c];
        let mut planned = crossed_plan();
        let mut workspace = NetworkExecutionWorkspace::default();
        drop(
            planned
                .execute_with_workspace(&refs, &mut workspace)
                .unwrap(),
        );

        let rhs_axes = std::mem::replace(
            &mut planned.schedule.steps[1].rhs_contract_axes,
            vec![usize::MAX],
        );
        assert!(planned
            .execute_with_workspace(&refs, &mut workspace)
            .is_err());
        assert!(workspace.slots[planned.schedule.steps[0].result_slot].is_some());
        planned.schedule.steps[1].rhs_contract_axes = rhs_axes;
        drop(
            planned
                .execute_with_workspace(&refs, &mut workspace)
                .unwrap(),
        );

        planned.schedule.final_permutation = Some((vec![usize::MAX], vec![0]));
        assert!(planned
            .execute_with_workspace(&refs, &mut workspace)
            .is_err());
        assert!(workspace.slots[planned.schedule.final_slot].is_some());
        planned.schedule.final_permutation = None;
        drop(
            planned
                .execute_with_workspace(&refs, &mut workspace)
                .unwrap(),
        );
    }

    #[test]
    fn typed_natural_split_change_replays_contract_then_permute() {
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let space = GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(0), 3)]).unwrap();
        let a = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&space], [&space], 31)
            .unwrap();
        let b = TensorMap::rand_with_seed(&runtime, [&space], [&space, &space], 32).unwrap();
        let c = TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 33).unwrap();
        let network = Network::new(
            vec![
                vec![label("a"), label("c")],
                vec![label("c"), label("b"), label("d")],
                vec![label("b"), label("d"), label("e")],
            ],
            vec![false; 3],
            vec![Some(1), Some(1), Some(2)],
            vec![label("e"), label("a")],
            Some(1),
        )
        .unwrap();
        let refs = [&a, &b, &c];
        let planned = network
            .plan(
                &refs,
                &crate::LabelOrderDenseOptimizer::new(vec![label("c"), label("b"), label("d")]),
            )
            .unwrap();
        assert!(planned.schedule.steps[0].result_output_axes.is_none());
        assert!(planned.schedule.steps[0].result_permutation.is_some());
        let expected = planned.execute(&refs).unwrap();
        let mut workspace = NetworkExecutionWorkspace::default();
        for _ in 0..2 {
            assert_eq!(
                planned
                    .execute_with_workspace(&refs, &mut workspace)
                    .unwrap()
                    .data(),
                expected.data()
            );
        }
        assert!(workspace.intermediates[0].contracted.is_some());
        assert!(workspace.intermediates[0].oriented.is_some());
    }

    #[test]
    fn typed_replay_restores_both_orientation_buffers_after_failure() {
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let space = |degeneracy| {
            GradedSpace::try_new_with_arc(Arc::clone(&provider), [(U1Irrep::new(0), degeneracy)])
                .unwrap()
        };
        let left = space(5);
        let bond = space(4);
        let right = space(6);
        let tail = space(7);
        let a =
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&left], [&bond], 21).unwrap();
        let b = TensorMap::rand_with_seed(&runtime, [&bond], [&right], 22).unwrap();
        let c =
            TensorMap::rand_with_seed(&runtime, [&left.try_dual().unwrap()], [&tail], 23).unwrap();
        let refs = [&a, &b, &c];
        let mut planned = crossed_plan();
        planned.schedule.steps[0].result_output_axes = None;
        planned.schedule.steps[0].contract_output_axes = vec![0, 1];
        planned.schedule.steps[0].result_permutation = Some((vec![1], vec![0]));
        let mut workspace = NetworkExecutionWorkspace::default();
        drop(
            planned
                .execute_with_workspace(&refs, &mut workspace)
                .unwrap(),
        );

        planned.schedule.steps[0].result_permutation = Some((vec![usize::MAX], vec![0]));
        assert!(planned
            .execute_with_workspace(&refs, &mut workspace)
            .is_err());
        assert!(workspace.intermediates[0].contracted.is_some());
        assert!(workspace.intermediates[0].oriented.is_some());
        planned.schedule.steps[0].result_permutation = Some((vec![1], vec![0]));
        drop(
            planned
                .execute_with_workspace(&refs, &mut workspace)
                .unwrap(),
        );
    }
}
