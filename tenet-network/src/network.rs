//! Typed Host contraction of a labeled tensor network.
//!
//! This is the execution half rewritten for the current user layer: the
//! planner ([`NetworkIR`], [`DenseCostModel`], [`ContractionPlan`]) is pure
//! structure, and each planned pairwise step lowers to
//! typed contraction plus orientation/final permutation calls. The erased
//! executor remains private solely for the `tensor!` compatibility path.

use std::collections::HashMap;
#[cfg(all(test, feature = "cuda"))]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU64, Ordering};

use tenet::core::{
    CheckedFusionAlgebra, FusionAlgebraError, MultiplicityFreeAdmissionMode,
    MultiplicityFreeRigidSymbols, RuleIdentity, SectorCodec, SectorLeg, TensorStorage,
    TypedSectorAdmission,
};
use tenet::prelude::{
    ContractOverwriteCache, Dtype, Error, OverwriteOutcome, PermuteOverwriteCache, Runtime, Scalar,
    Tensor, TensorExecutionContext, TensorScalar,
};
#[cfg(feature = "cuda")]
use tenet::typed::CudaStorage;
use tenet::typed::{GradedSpace, NetworkReuseClass, TensorMap};
#[cfg(feature = "cuda")]
use tenet::{core::Placement, operations::OperationError};
use tenet::{RuntimeDetachedTensor, RuntimeIdentity};

use crate::cost::{DenseCostModel, DenseTensorInfo};
use crate::ir::NetworkIR;
use crate::labels::{TemporaryLabel, TensorId};
use crate::optimizer::{ContractionStep, DenseContractionOptimizer};
use crate::plan::ContractionPlan;
use crate::plancache::Optimizer;

/// One operand of a labeled network: a tensor reference, an adjoint
/// (`conj`) marker, its leg labels as written (flat order: codomain legs
/// then domain legs of the *original* tensor), and an optional stated
/// codomain rank (the position of `;` in the written label list, checked
/// against the tensor at plan time).
pub(crate) struct NetOperand<'a> {
    pub tensor: &'a Tensor,
    pub conj: bool,
    pub labels: &'a [&'a str],
    pub codomain_split: Option<usize>,
}

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
    cache_id: u64,
    pub(crate) inputs: Vec<Vec<TemporaryLabel>>,
    pub(crate) conj: Vec<bool>,
    pub(crate) codomain_splits: Vec<Option<usize>>,
    pub(crate) output: Vec<TemporaryLabel>,
    /// Number of output labels on the codomain side (`;` position);
    /// `None` = all-codomain output.
    pub(crate) output_codomain_rank: Option<usize>,
}

static NEXT_NETWORK_CACHE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_PLAN_OWNER_TOKEN: AtomicU64 = AtomicU64::new(1);
#[cfg(all(test, feature = "cuda"))]
static CUDA_NETWORK_CONTRACT_CALLS: AtomicUsize = AtomicUsize::new(0);

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(message.to_string())
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
            cache_id: NEXT_NETWORK_CACHE_ID.fetch_add(1, Ordering::Relaxed),
            inputs,
            conj,
            codomain_splits,
            output,
            output_codomain_rank,
        })
    }

    pub(crate) fn cache_id(&self) -> u64 {
        self.cache_id
    }

    /// Plans from storage-independent metadata of homogeneous typed
    /// multiplicity-free tensors; payload storage is never read or transferred.
    pub fn plan<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        optimizer: &(impl DenseContractionOptimizer + ?Sized),
    ) -> Result<PlannedNetwork, Error>
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let (ir, infos) = self.lower_typed(tensors)?;
        let plan = if ir.tensors().len() == 1 {
            ContractionPlan::new(1, self.output.clone(), Vec::new()).map_err(invalid)?
        } else {
            let cost = DenseCostModel::from_network(&ir, &infos).map_err(invalid)?;
            ContractionPlan::from_dense_optimizer(&ir, optimizer, &cost).map_err(invalid)?
        };
        self.finish_typed_plan(tensors, ir, plan)
    }

    /// Wraps an already searched structural order for typed execution.
    pub fn plan_with<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        plan: ContractionPlan,
    ) -> Result<PlannedNetwork, Error>
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let (ir, _) = self.lower_typed(tensors)?;
        self.finish_typed_plan(tensors, ir, plan)
    }

    fn finish_typed_plan<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        ir: NetworkIR,
        plan: ContractionPlan,
    ) -> Result<PlannedNetwork, Error>
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
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
        Ok(PlannedNetwork {
            owner_token: NEXT_PLAN_OWNER_TOKEN.fetch_add(1, Ordering::Relaxed),
            plan,
            conj: self.conj.clone(),
            input_codomain_ranks,
            schedule,
        })
    }

    fn lower_typed<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
    ) -> Result<(NetworkIR, Vec<DenseTensorInfo>), Error>
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        if tensors.len() != self.inputs.len() {
            return Err(invalid(format!(
                "network has {} operands but {} tensors were given",
                self.inputs.len(),
                tensors.len()
            )));
        }
        if let Some(first) = tensors.first() {
            let runtime = first.runtime().identity();
            let identity = TypedSectorAdmission::typed_rule_identity(first.provider());
            for (index, tensor) in tensors.iter().enumerate().skip(1) {
                if !runtime.matches(tensor.runtime()) {
                    return Err(invalid(format!("operand {index} uses a different Runtime")));
                }
                if identity != TypedSectorAdmission::typed_rule_identity(tensor.provider()) {
                    return Err(Error::RuleMismatch);
                }
            }
        }

        let mut lowered_labels = Vec::with_capacity(tensors.len());
        let mut infos = Vec::with_capacity(tensors.len());
        let mut lowered_spaces = Vec::with_capacity(tensors.len());
        for (i, (&tensor, labels)) in tensors.iter().zip(&self.inputs).enumerate() {
            if labels.len() != tensor.rank() {
                return Err(invalid(format!(
                    "operand {i} has {} labels but tensor rank {}",
                    labels.len(),
                    tensor.rank()
                )));
            }
            if let Some(split) = self.codomain_splits[i] {
                if split != tensor.codomain_rank() {
                    return Err(invalid(format!(
                        "operand {i} puts {split} label(s) before `;` but the tensor's codomain rank is {}",
                        tensor.codomain_rank()
                    )));
                }
            }
            let dims = if self.conj[i] {
                let split = tensor.codomain_rank();
                lowered_labels.push(rotate(labels, split));
                rotate(&tensor.leg_dims()?, split)
            } else {
                lowered_labels.push(labels.clone());
                tensor.leg_dims()?
            };
            infos.push(DenseTensorInfo::new(dims));
            lowered_spaces.push(typed_effective_spaces(tensor, self.conj[i])?);
        }
        validate_typed_contracted_leg_spaces(&lowered_labels, &lowered_spaces)?;
        let ir = NetworkIR::from_labels(lowered_labels, self.output.clone()).map_err(invalid)?;
        Ok((ir, infos))
    }

    /// Plan the contraction order for concrete operand tensors using the
    /// given optimizer. The plan is data-independent (labels + leg
    /// dimensions only) and can be executed repeatedly over same-shaped
    /// operands.
    pub(crate) fn plan_erased(
        &self,
        tensors: &[&Tensor],
        optimizer: &(impl DenseContractionOptimizer + ?Sized),
    ) -> Result<PlannedNetwork, Error> {
        let (ir, infos) = self.lower_erased(tensors)?;
        let plan = if ir.tensors().len() == 1 {
            // Single operand: nothing to order; the executor just permutes.
            ContractionPlan::new(1, self.output.clone(), Vec::new()).map_err(invalid)?
        } else {
            let cost = DenseCostModel::from_network(&ir, &infos).map_err(invalid)?;
            ContractionPlan::from_dense_optimizer(&ir, optimizer, &cost).map_err(invalid)?
        };
        let input_codomain_ranks: Vec<usize> = tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect();
        let lowered_codomain_ranks: Vec<usize> = tensors
            .iter()
            .enumerate()
            .map(|(i, tensor)| {
                if self.conj[i] {
                    tensor.rank() - tensor.codomain_rank()
                } else {
                    tensor.codomain_rank()
                }
            })
            .collect();
        self.finish_plan(input_codomain_ranks, lowered_codomain_ranks, ir, plan)
    }

    /// Wrap an already-searched [`ContractionPlan`] (same topology) into a
    /// [`PlannedNetwork`] without re-running the order search. The plan is a
    /// pure pairwise order over operand ids and labels, valid for any leg
    /// dimensions of this topology, so a persisted plan (see the plan cache's
    /// disk save/restore) skips the cold optimal-order search on reuse.
    pub(crate) fn plan_with_erased(
        &self,
        tensors: &[&Tensor],
        plan: ContractionPlan,
    ) -> Result<PlannedNetwork, Error> {
        let (ir, _infos) = self.lower_erased(tensors)?;
        let input_codomain_ranks: Vec<usize> = tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect();
        let lowered_codomain_ranks: Vec<usize> = tensors
            .iter()
            .enumerate()
            .map(|(i, tensor)| {
                if self.conj[i] {
                    tensor.rank() - tensor.codomain_rank()
                } else {
                    tensor.codomain_rank()
                }
            })
            .collect();
        self.finish_plan(input_codomain_ranks, lowered_codomain_ranks, ir, plan)
    }

    /// Validate operand ranks and `;` splits and lower conj markers into the
    /// [`NetworkIR`] and per-operand cost infos shared by [`plan`](Self::plan)
    /// and [`plan_with`](Self::plan_with).
    fn lower_erased(
        &self,
        tensors: &[&Tensor],
    ) -> Result<(NetworkIR, Vec<DenseTensorInfo>), Error> {
        if tensors.len() != self.inputs.len() {
            return Err(invalid(format!(
                "network has {} operands but {} tensors were given",
                self.inputs.len(),
                tensors.len()
            )));
        }

        // Validate ranks and written `;` splits, then lower conj: the
        // adjoint swaps codomain and domain (domain legs lead), so the
        // labels and leg dims rotate by the original codomain rank.
        let mut lowered_labels = Vec::with_capacity(tensors.len());
        let mut infos = Vec::with_capacity(tensors.len());
        let mut lowered_spaces = Vec::with_capacity(tensors.len());
        for (i, (&tensor, labels)) in tensors.iter().zip(&self.inputs).enumerate() {
            if labels.len() != tensor.rank() {
                return Err(invalid(format!(
                    "operand {i} has {} labels but tensor rank {}",
                    labels.len(),
                    tensor.rank()
                )));
            }
            if let Some(split) = self.codomain_splits[i] {
                if split != tensor.codomain_rank() {
                    return Err(invalid(format!(
                        "operand {i} puts {split} label(s) before `;` but the tensor's \
                         codomain rank is {}",
                        tensor.codomain_rank()
                    )));
                }
            }
            let dims = tensor.leg_dims()?;
            let spaces = (0..tensor.rank())
                .map(|axis| tensor.space(axis))
                .collect::<Result<Vec<_>, _>>()?;
            if self.conj[i] {
                let c = tensor.codomain_rank();
                lowered_labels.push(rotate(labels, c));
                infos.push(DenseTensorInfo::new(rotate(&dims, c)));
                // Adjoint legs: `space(t', i) = dual(space(t, sigma(i)))`
                // with sigma the codomain/domain rotation.
                lowered_spaces.push(rotate(&spaces, c).iter().map(|s| s.dual()).collect());
            } else {
                lowered_labels.push(labels.clone());
                infos.push(DenseTensorInfo::new(dims));
                lowered_spaces.push(spaces);
            }
        }
        validate_contracted_leg_spaces(&lowered_labels, &lowered_spaces)?;

        let ir = NetworkIR::from_labels(lowered_labels, self.output.clone()).map_err(invalid)?;
        Ok((ir, infos))
    }

    /// One-shot contraction with the operands' runtime's default
    /// [`Optimizer`] (greedy unless changed on `Runtime::builder()` or via
    /// [`crate::configure_plan_cache`]), going through that runtime's
    /// topology-keyed plan cache. This is what the `tensor!` macro path
    /// runs.
    pub(crate) fn contract_erased(&self, tensors: &[&Tensor]) -> Result<Tensor, Error> {
        let optimizer = tensors
            .first()
            .map(|tensor| tensor.runtime().plan_cache_config().optimizer)
            .unwrap_or_default();
        self.contract_with_erased(tensors, &optimizer)
    }

    /// Private erased one-shot execution with an explicit [`Optimizer`] choice
    /// (still cached; the optimizer is part of the cache key). For a raw
    /// [`DenseContractionOptimizer`] implementation, use [`Self::plan`],
    /// which always plans fresh.
    pub(crate) fn contract_with_erased(
        &self,
        tensors: &[&Tensor],
        optimizer: &Optimizer,
    ) -> Result<Tensor, Error> {
        crate::plancache::get_or_plan(self, tensors, optimizer)?.execute(tensors)
    }
}

/// Structural leg compatibility of every contracted label pair, checked at
/// plan time against the operands' graded leg spaces (sectors, per-sector
/// degeneracies and duality). A contracted pair must be mutually dual
/// spaces — the same rule the expert layer's `validate_composed_leg`
/// enforces after the pre-contraction permutes (verbatim spaces, one side
/// dual). TensorKit `SpaceMismatch` analog with both legs spelled out.
fn validate_contracted_leg_spaces(
    labels: &[Vec<TemporaryLabel>],
    spaces: &[Vec<tenet::prelude::Space>],
) -> Result<(), Error> {
    let mut seen: HashMap<&TemporaryLabel, (usize, usize)> = HashMap::new();
    for (operand, operand_labels) in labels.iter().enumerate() {
        for (axis, label) in operand_labels.iter().enumerate() {
            let Some(&(prev_operand, prev_axis)) = seen.get(label) else {
                seen.insert(label, (operand, axis));
                continue;
            };
            let lhs = &spaces[prev_operand][prev_axis];
            let rhs = &spaces[operand][axis];
            if *rhs != lhs.dual() {
                return Err(invalid(format!(
                    "space mismatch for contracted label `{label}`: operand {prev_operand} \
                     leg {prev_axis} is {lhs:?}, operand {operand} leg {axis} is {rhs:?}; \
                     contracted legs must be mutually dual (same sectors and degeneracies, \
                     one side dual)"
                )));
            }
        }
    }
    Ok(())
}

fn validate_typed_contracted_leg_spaces<R>(
    labels: &[Vec<TemporaryLabel>],
    spaces: &[Vec<GradedSpace<R>>],
) -> Result<(), Error>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
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
                )));
            }
        }
    }
    Ok(())
}

fn validate_typed_contracted_pairs<R, D>(
    tensors: &[TensorMap<R, D>],
    pairs: &[InputLegPair],
) -> Result<(), Error>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    let spaces = tensors
        .iter()
        .map(typed_flat_spaces)
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

fn typed_flat_spaces<R, D, S>(tensor: &TensorMap<R, D, S>) -> Result<Vec<GradedSpace<R>>, Error>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
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
) -> Result<Vec<GradedSpace<R>>, Error>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
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

struct CompiledStep {
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

/// Caller-owned typed Host replay storage for one planned network at a time.
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
}

enum StepOutput<T> {
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

impl<R, D> Default for TypedIntermediateBuffers<R, D> {
    fn default() -> Self {
        Self {
            contracted: None,
            oriented: None,
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
    fn clear_replay_state(&mut self) {
        self.slots.clear();
        self.producers.clear();
        self.intermediates.clear();
        self.owner_token = None;
        self.runtime = None;
        self.rule_identity = None;
        self.input_snapshot.clear();
    }
}

/// Caller-owned tensor slots for repeated execution of a [`PlannedNetwork`].
#[derive(Default)]
pub(crate) struct ErasedNetworkExecutionWorkspace {
    slots: Vec<Option<Tensor>>,
    slot_producers: Vec<Option<(usize, bool)>>,
    intermediates: Vec<IntermediateBuffers>,
    tensor_context: Option<TensorExecutionContext>,
    tensor_runtime: Option<RuntimeIdentity>,
    stats: NetworkExecutionStats,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NetworkExecutionStats {
    pub owned_intermediates: u64,
    pub reused_intermediates: u64,
    pub owned_contractions: u64,
    pub reused_contractions: u64,
    pub owned_orientations: u64,
    pub reused_orientations: u64,
    pub escaped_outputs: u64,
    pub contract_layout_preparations: u64,
    pub orientation_layout_preparations: u64,
    pub contract_structural_comparisons: u64,
    pub orientation_structural_comparisons: u64,
}

#[derive(Default)]
struct IntermediateBuffers {
    contracted: Option<Tensor>,
    oriented: Option<Tensor>,
    parked_contracted: Option<RuntimeDetachedTensor>,
    parked_oriented: Option<RuntimeDetachedTensor>,
    contract_cache: ContractOverwriteCache,
    orientation_cache: PermuteOverwriteCache,
}

impl ErasedNetworkExecutionWorkspace {
    pub(crate) fn slot_capacity(&self) -> usize {
        self.slots.capacity()
    }

    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.slot_producers.clear();
    }

    pub(crate) fn park_runtime_owners(&mut self) {
        for buffers in &mut self.intermediates {
            debug_assert!(buffers.parked_contracted.is_none());
            debug_assert!(buffers.parked_oriented.is_none());
            buffers.parked_contracted = buffers.contracted.take().map(Tensor::detach_runtime);
            buffers.parked_oriented = buffers.oriented.take().map(Tensor::detach_runtime);
        }
        if let Some(context) = &mut self.tensor_context {
            context.release_runtime_binding();
        }
    }

    fn activate_parked(&mut self, runtime: &Runtime) -> Result<(), Error> {
        // Why not attach as we iterate: a later identity mismatch would leave
        // half the workspace rebound. Validate the complete idle set first.
        let same_runtime = self.intermediates.iter().all(|buffers| {
            buffers
                .parked_contracted
                .as_ref()
                .is_none_or(|tensor| tensor.matches_runtime(runtime))
                && buffers
                    .parked_oriented
                    .as_ref()
                    .is_none_or(|tensor| tensor.matches_runtime(runtime))
        });
        if !same_runtime {
            for buffers in &mut self.intermediates {
                buffers.parked_contracted = None;
                buffers.parked_oriented = None;
            }
            return Ok(());
        }
        for buffers in &mut self.intermediates {
            if let Some(tensor) = buffers.parked_contracted.take() {
                buffers.contracted = Some(tensor.attach_runtime(runtime)?);
            }
            if let Some(tensor) = buffers.parked_oriented.take() {
                buffers.oriented = Some(tensor.attach_runtime(runtime)?);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> NetworkExecutionStats {
        self.stats
    }

    #[cfg(test)]
    pub(crate) fn reserve_slots(&mut self, count: usize) {
        self.slots.reserve(count);
    }

    #[cfg(test)]
    pub(crate) fn slot_len(&self) -> usize {
        self.slots.len()
    }

    #[cfg(test)]
    pub(crate) fn retain_tensor(&mut self, tensor: Tensor) {
        self.slots.push(Some(tensor));
    }
}

impl PlannedNetwork {
    /// The resolved pairwise contraction order with its cost estimates.
    pub fn plan(&self) -> &ContractionPlan {
        &self.plan
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
        let mut shapes = vec![None; self.schedule.slot_count];
        for (index, &shape) in input_shapes.iter().enumerate() {
            shapes[index] = Some(shape);
        }
        for step in &self.schedule.steps {
            let (lhs_rank, lhs_codomain_rank) = shapes[step.lhs_slot]
                .take()
                .ok_or_else(|| invalid("lhs shape already consumed"))?;
            let (rhs_rank, rhs_codomain_rank) = shapes[step.rhs_slot]
                .take()
                .ok_or_else(|| invalid("rhs shape already consumed"))?;
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
                return Err(unsupported_cuda_network());
            }
            shapes[step.result_slot] = Some((result_rank, lhs_codomain_rank));
        }
        if self.schedule.final_permutation.is_some() {
            return Err(unsupported_cuda_network());
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn execute_erased(&self, tensors: &[&Tensor]) -> Result<Tensor, Error> {
        self.execute_erased_with_workspace(tensors, &mut ErasedNetworkExecutionWorkspace::default())
    }

    /// Run the compiled schedule while reusing its tensor-slot table and
    /// eligible host intermediate buffers. A returned [`Error`] preserves
    /// checked-out reusable buffers. Backend panics are treated as fatal and
    /// may discard workspace contents; the runtime already applies the same
    /// policy by poisoning its execution-state mutex after an unwind.
    pub(crate) fn execute_erased_with_workspace(
        &self,
        tensors: &[&Tensor],
        workspace: &mut ErasedNetworkExecutionWorkspace,
    ) -> Result<Tensor, Error> {
        if tensors.len() != self.conj.len() {
            return Err(invalid(format!(
                "plan has {} operands but {} tensors were given",
                self.conj.len(),
                tensors.len()
            )));
        }

        let runtime = tensors[0].runtime();
        workspace.activate_parked(runtime)?;

        workspace
            .slots
            .resize_with(self.schedule.slot_count, || None);
        workspace
            .slot_producers
            .resize(self.schedule.slot_count, None);
        workspace
            .intermediates
            .resize_with(self.schedule.steps.len(), IntermediateBuffers::default);
        if workspace
            .tensor_runtime
            .as_ref()
            .is_none_or(|cached| !cached.matches(runtime))
        {
            workspace.tensor_context = Some(TensorExecutionContext::for_runtime(runtime)?);
            workspace.tensor_runtime = Some(runtime.identity());
            workspace.intermediates.clear();
            workspace
                .intermediates
                .resize_with(self.schedule.steps.len(), IntermediateBuffers::default);
        } else if let Some(context) = &mut workspace.tensor_context {
            context.bind_runtime(runtime)?;
        }
        for slot in &mut workspace.slots {
            *slot = None;
        }
        workspace.slot_producers.fill(None);
        for (i, &tensor) in tensors.iter().enumerate() {
            if tensor.rank() != self.schedule.input_ranks[i]
                || tensor.codomain_rank() != self.input_codomain_ranks[i]
            {
                return Err(invalid(format!(
                    "operand {i} topology drifted: planned rank/split {}/{}, got {}/{}",
                    self.schedule.input_ranks[i],
                    self.input_codomain_ranks[i],
                    tensor.rank(),
                    tensor.codomain_rank()
                )));
            }
            let lowered = if self.conj[i] {
                tensor.adjoint()?
            } else {
                tensor.clone()
            };
            workspace.slots[i] = Some(lowered);
        }

        for (step_index, step) in self.schedule.steps.iter().enumerate() {
            let lhs = workspace.slots[step.lhs_slot]
                .take()
                .ok_or_else(|| invalid("lhs operand already consumed"))?;
            let lhs_producer = workspace.slot_producers[step.lhs_slot].take();
            let rhs = workspace.slots[step.rhs_slot]
                .take()
                .ok_or_else(|| invalid("rhs operand already consumed"))?;
            let rhs_producer = workspace.slot_producers[step.rhs_slot].take();

            // Replay a same-split pAB directly into its retained oriented slot.
            // Compile-time filtering leaves boundary-moving orientations on the
            // established two-stage path; incompatible storage also falls through.
            if let Some(output_axes) = &step.result_output_axes {
                if let Some(mut destination) = workspace.intermediates[step_index].oriented.take() {
                    let preparations = workspace.intermediates[step_index]
                        .contract_cache
                        .preparations();
                    let structural_comparisons = workspace.intermediates[step_index]
                        .contract_cache
                        .structural_comparisons();
                    let overwrite = workspace
                        .tensor_context
                        .as_mut()
                        .expect("execution context initialized")
                        .try_contract_ordered_overwrite_into(
                            &mut workspace.intermediates[step_index].contract_cache,
                            &mut destination,
                            &lhs,
                            &rhs,
                            &step.lhs_contract_axes,
                            &step.rhs_contract_axes,
                            output_axes,
                            identity_scalar(lhs.dtype()),
                        );
                    workspace.stats.contract_layout_preparations += workspace.intermediates
                        [step_index]
                        .contract_cache
                        .preparations()
                        - preparations;
                    workspace.stats.contract_structural_comparisons += workspace.intermediates
                        [step_index]
                        .contract_cache
                        .structural_comparisons()
                        - structural_comparisons;
                    match overwrite {
                        Ok(OverwriteOutcome::Written) => {
                            drop(workspace.intermediates[step_index].contracted.take());
                            workspace.stats.reused_intermediates += 1;
                            workspace.stats.reused_contractions += 1;
                            return_intermediate(workspace, lhs, lhs_producer);
                            return_intermediate(workspace, rhs, rhs_producer);
                            workspace.slots[step.result_slot] = Some(destination);
                            workspace.slot_producers[step.result_slot] = Some((step_index, true));
                            continue;
                        }
                        Ok(OverwriteOutcome::Incompatible) => {
                            workspace.intermediates[step_index].oriented = Some(destination);
                        }
                        Err(error) => {
                            workspace.intermediates[step_index].oriented = Some(destination);
                            return_intermediate(workspace, lhs, lhs_producer);
                            return_intermediate(workspace, rhs, rhs_producer);
                            return Err(error);
                        }
                    }
                }
            }

            let contraction = if let Some(mut destination) =
                workspace.intermediates[step_index].contracted.take()
            {
                let preparations = workspace.intermediates[step_index]
                    .contract_cache
                    .preparations();
                let structural_comparisons = workspace.intermediates[step_index]
                    .contract_cache
                    .structural_comparisons();
                let overwrite = workspace
                    .tensor_context
                    .as_mut()
                    .expect("execution context initialized")
                    .try_contract_overwrite_into(
                        &mut workspace.intermediates[step_index].contract_cache,
                        &mut destination,
                        &lhs,
                        &rhs,
                        &step.lhs_contract_axes,
                        &step.rhs_contract_axes,
                        identity_scalar(lhs.dtype()),
                    );
                workspace.stats.contract_layout_preparations += workspace.intermediates[step_index]
                    .contract_cache
                    .preparations()
                    - preparations;
                workspace.stats.contract_structural_comparisons += workspace.intermediates
                    [step_index]
                    .contract_cache
                    .structural_comparisons()
                    - structural_comparisons;
                match overwrite {
                    Ok(OverwriteOutcome::Written) => {
                        workspace.stats.reused_intermediates += 1;
                        workspace.stats.reused_contractions += 1;
                        Ok(destination)
                    }
                    Ok(OverwriteOutcome::Incompatible) => {
                        workspace.stats.owned_intermediates += 1;
                        workspace.stats.owned_contractions += 1;
                        match lhs.contract(&rhs, &step.lhs_contract_axes, &step.rhs_contract_axes) {
                            Ok(result) => Ok(result),
                            Err(error) => {
                                workspace.intermediates[step_index].contracted = Some(destination);
                                Err(error)
                            }
                        }
                    }
                    Err(error) => {
                        workspace.intermediates[step_index].contracted = Some(destination);
                        Err(error)
                    }
                }
            } else {
                workspace.stats.owned_intermediates += 1;
                workspace.stats.owned_contractions += 1;
                lhs.contract(&rhs, &step.lhs_contract_axes, &step.rhs_contract_axes)
            };
            let mut result = match contraction {
                Ok(result) => result,
                Err(error) => {
                    return_intermediate(workspace, lhs, lhs_producer);
                    return_intermediate(workspace, rhs, rhs_producer);
                    return Err(error);
                }
            };
            let mut result_producer = (step_index, false);
            if let Some((codomain, domain)) = &step.result_permutation {
                let permutation = if let Some(mut destination) =
                    workspace.intermediates[step_index].oriented.take()
                {
                    let preparations = workspace.intermediates[step_index]
                        .orientation_cache
                        .preparations();
                    let structural_comparisons = workspace.intermediates[step_index]
                        .orientation_cache
                        .structural_comparisons();
                    let overwrite = workspace
                        .tensor_context
                        .as_mut()
                        .expect("execution context initialized")
                        .try_permute_overwrite_into(
                            &mut workspace.intermediates[step_index].orientation_cache,
                            &mut destination,
                            &result,
                            codomain,
                            domain,
                            identity_scalar(result.dtype()),
                        );
                    workspace.stats.orientation_layout_preparations += workspace.intermediates
                        [step_index]
                        .orientation_cache
                        .preparations()
                        - preparations;
                    workspace.stats.orientation_structural_comparisons += workspace.intermediates
                        [step_index]
                        .orientation_cache
                        .structural_comparisons()
                        - structural_comparisons;
                    match overwrite {
                        Ok(OverwriteOutcome::Written) => {
                            workspace.stats.reused_intermediates += 1;
                            workspace.stats.reused_orientations += 1;
                            Ok(destination)
                        }
                        Ok(OverwriteOutcome::Incompatible) => {
                            workspace.stats.owned_intermediates += 1;
                            workspace.stats.owned_orientations += 1;
                            match result.permute(codomain, domain) {
                                Ok(oriented) => Ok(oriented),
                                Err(error) => {
                                    workspace.intermediates[step_index].oriented =
                                        Some(destination);
                                    Err(error)
                                }
                            }
                        }
                        Err(error) => {
                            workspace.intermediates[step_index].oriented = Some(destination);
                            Err(error)
                        }
                    }
                } else {
                    workspace.stats.owned_intermediates += 1;
                    workspace.stats.owned_orientations += 1;
                    result.permute(codomain, domain)
                };
                let oriented = match permutation {
                    Ok(oriented) => oriented,
                    Err(error) => {
                        workspace.intermediates[step_index].contracted = Some(result);
                        return_intermediate(workspace, lhs, lhs_producer);
                        return_intermediate(workspace, rhs, rhs_producer);
                        return Err(error);
                    }
                };
                workspace.intermediates[step_index].contracted = Some(result);
                result = oriented;
                result_producer = (step_index, true);
            }
            return_intermediate(workspace, lhs, lhs_producer);
            return_intermediate(workspace, rhs, rhs_producer);
            workspace.slots[step.result_slot] = Some(result);
            workspace.slot_producers[step.result_slot] = Some(result_producer);
        }

        let mut result = workspace.slots[self.schedule.final_slot]
            .take()
            .ok_or_else(|| invalid("no final tensor produced"))?;
        let result_producer = workspace.slot_producers[self.schedule.final_slot].take();
        if let Some((codomain, domain)) = &self.schedule.final_permutation {
            let output = match result.permute(codomain, domain) {
                Ok(output) => output,
                Err(error) => {
                    return_intermediate(workspace, result, result_producer);
                    return Err(error);
                }
            };
            return_intermediate(workspace, result, result_producer);
            result = output;
        }
        workspace.stats.escaped_outputs += 1;
        Ok(result)
    }
}

impl PlannedNetwork {
    /// Executes this plan with a fresh typed workspace.
    pub fn execute<R, D>(&self, tensors: &[&TensorMap<R, D>]) -> Result<TensorMap<R, D>, Error>
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
        D: TensorScalar,
    {
        self.execute_with_workspace(tensors, &mut NetworkExecutionWorkspace::default())
    }

    /// Executes this plan while reusing exact-compatible typed Host destinations.
    pub fn execute_with_workspace<R, D>(
        &self,
        tensors: &[&TensorMap<R, D>],
        workspace: &mut NetworkExecutionWorkspace<R, D>,
    ) -> Result<TensorMap<R, D>, Error>
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
        D: TensorScalar,
    {
        let prepared = (|| {
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
                .collect::<Result<Vec<_>, Error>>()?;
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
            let reuse_enabled = new_snapshot
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
        let (runtime_identity, rule_identity, lowered, new_snapshot, reuse_enabled) = match prepared
        {
            Ok(prepared) => prepared,
            Err(error) => return Err(error),
        };
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
        let slots = &mut workspace.slots;
        let producers = &mut workspace.producers;
        let intermediates = &mut workspace.intermediates;
        for (step_index, step) in self.schedule.steps.iter().enumerate() {
            let lhs = slots[step.lhs_slot]
                .as_ref()
                .ok_or_else(|| invalid("lhs operand already consumed"))?;
            let rhs = slots[step.rhs_slot]
                .as_ref()
                .ok_or_else(|| invalid("rhs operand already consumed"))?;
            let fused = step.result_output_axes.is_some();
            let TypedIntermediateBuffers {
                contracted: contracted_buffer,
                oriented: oriented_buffer,
            } = &mut intermediates[step_index];
            let contract_buffer = if fused {
                &mut *oriented_buffer
            } else {
                &mut *contracted_buffer
            };
            let contracted = if reuse_enabled && contract_buffer.is_some() {
                lhs.contract_overwrite_into(
                    rhs,
                    contract_buffer.as_mut().expect("destination checked"),
                    &step.lhs_contract_axes,
                    &step.rhs_contract_axes,
                    &step.contract_output_axes,
                    D::from_real(1.0),
                )?;
                StepOutput::Overwritten
            } else {
                StepOutput::Returned(lhs.contract(
                    rhs,
                    &step.lhs_contract_axes,
                    &step.rhs_contract_axes,
                    &step.contract_output_axes,
                )?)
            };
            let result = if fused {
                contracted.take(oriented_buffer)
            } else if let Some((codomain, domain)) = &step.result_permutation {
                let oriented = if reuse_enabled && oriented_buffer.is_some() {
                    contracted.get(contracted_buffer).permute_overwrite_into(
                        oriented_buffer.as_mut().expect("destination checked"),
                        codomain,
                        domain,
                        D::from_real(1.0),
                    )?;
                    StepOutput::Overwritten
                } else {
                    StepOutput::Returned(
                        contracted
                            .get(contracted_buffer)
                            .permute(codomain, domain)?,
                    )
                };
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
        }
        let result = slots[self.schedule.final_slot]
            .as_ref()
            .ok_or_else(|| invalid("no final tensor produced"))?;
        if let Some((codomain, domain)) = &self.schedule.final_permutation {
            let output = result.permute(codomain, domain)?;
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

fn identity_scalar(dtype: Dtype) -> Scalar {
    match dtype {
        Dtype::F64 => Scalar::F64(1.0),
        Dtype::C64 => Scalar::C64(tenet::prelude::Complex64::new(1.0, 0.0)),
    }
}

fn return_intermediate(
    workspace: &mut ErasedNetworkExecutionWorkspace,
    tensor: Tensor,
    producer: Option<(usize, bool)>,
) {
    if let Some((step, oriented)) = producer {
        let destination = if oriented {
            &mut workspace.intermediates[step].oriented
        } else {
            &mut workspace.intermediates[step].contracted
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

/// One-shot entry point used by the `tensor!` macro expansion: lower
/// intra-operand trace pairs, build a [`Network`] from the (reduced)
/// written labels, plan with the configured optimizer (through the plan
/// cache), and execute over the given operands.
fn contract_network(
    operands: &[NetOperand<'_>],
    output: &[&str],
    output_codomain_rank: Option<usize>,
) -> Result<Tensor, Error> {
    // Pre-pass, mirroring TensorOperations' @tensor lowering: a label
    // written twice on ONE operand is a partial trace of that operand. The
    // operand is traced first (user-layer categorical trace, i.e. the
    // expert tensortrace with quantum-dimension/twist coefficients) and
    // re-enters the pairwise network with its trace labels removed, so the
    // cost model plans over the shrunk dimensions.
    let mut inputs: Vec<Vec<TemporaryLabel>> = Vec::with_capacity(operands.len());
    let mut conj = Vec::with_capacity(operands.len());
    let mut splits = Vec::with_capacity(operands.len());
    let mut lowered: Vec<Option<Tensor>> = Vec::with_capacity(operands.len());
    for (index, op) in operands.iter().enumerate() {
        let written: Vec<TemporaryLabel> =
            op.labels.iter().map(|&l| TemporaryLabel::from(l)).collect();
        if !has_intra_operand_pair(&written) {
            inputs.push(written);
            conj.push(op.conj);
            splits.push(op.codomain_split);
            lowered.push(None);
            continue;
        }
        if written.len() != op.tensor.rank() {
            return Err(invalid(format!(
                "operand {index} has {} labels but tensor rank {}",
                written.len(),
                op.tensor.rank()
            )));
        }
        if let Some(split) = op.codomain_split {
            if split != op.tensor.codomain_rank() {
                return Err(invalid(format!(
                    "operand {index} puts {split} label(s) before `;` but the tensor's \
                     codomain rank is {}",
                    op.tensor.codomain_rank()
                )));
            }
        }
        // conj lowers first (adjoint; domain legs lead), exactly as the
        // executor does, so the trace pairs address the adjointed legs:
        // @tensor conj(a)[i, i] is the trace of a's adjoint.
        let (tensor, labels) = if op.conj {
            (
                op.tensor.adjoint()?,
                rotate(&written, op.tensor.codomain_rank()),
            )
        } else {
            (op.tensor.clone(), written)
        };
        let (pairs, reduced) = split_trace_pairs(index, &labels)?;
        lowered.push(Some(tensor.trace_pairs(&pairs)?));
        inputs.push(reduced);
        conj.push(false);
        splits.push(None);
    }

    let network = Network::new(
        inputs,
        conj,
        splits,
        output.iter().map(|&l| TemporaryLabel::from(l)).collect(),
        output_codomain_rank,
    )?;
    let tensors: Vec<&Tensor> = operands
        .iter()
        .zip(&lowered)
        .map(|(op, traced)| traced.as_ref().unwrap_or(op.tensor))
        .collect();
    network.contract_erased(&tensors)
}

/// Borrowed topology lookup used by [`tensor!`] for networks without
/// intra-operand traces.
#[doc(hidden)]
pub fn contract_static_network(
    tensors: &[&Tensor],
    spec: &'static StaticTopologySpec,
) -> Result<Tensor, Error> {
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
    if spec
        .inputs
        .iter()
        .any(|labels| has_intra_operand_pair_names(labels))
    {
        let operands: Vec<NetOperand<'_>> = tensors
            .iter()
            .enumerate()
            .map(|(index, &tensor)| NetOperand {
                tensor,
                conj: spec.conj[index],
                labels: spec.inputs[index],
                codomain_split: spec.codomain_splits[index],
            })
            .collect();
        return contract_network(&operands, spec.output, spec.output_codomain_rank);
    }
    let optimizer = tensors
        .first()
        .map(|tensor| tensor.runtime().plan_cache_config().optimizer)
        .unwrap_or_default();
    crate::plancache::execute_static(spec, tensors, &optimizer)
}

fn has_intra_operand_pair_names(labels: &[&str]) -> bool {
    labels
        .iter()
        .enumerate()
        .any(|(i, label)| labels[..i].contains(label))
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
    use tenet::core::{product_sector, FermionParityFusionRule, ProductFusionRuleExt, Z2Irrep};
    use tenet::core::{U1FusionRule, U1Irrep};
    use tenet::typed::{GradedSpace, TensorMap};

    use super::*;
    #[cfg(feature = "cuda")]
    use crate::GreedyDenseOptimizer;

    fn label(name: &str) -> TemporaryLabel {
        TemporaryLabel::from(name)
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
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_schedule_preflight_accepts_only_the_complete_direct_subset() {
        let runtime = Runtime::builder().build().unwrap();
        let space =
            GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)], false).unwrap();
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
    #[ignore = "requires a real CUDA device"]
    fn cuda_rejections_happen_before_the_first_network_contract() {
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        let u1_rule = Arc::new(U1FusionRule);
        let u1_x0 =
            GradedSpace::try_new(Arc::clone(&u1_rule), [(U1Irrep::new(2), 2)], false).unwrap();
        let u1_x1 = GradedSpace::try_new(Arc::clone(&u1_rule), [(U1Irrep::new(-1), 1)], false)
            .unwrap()
            .try_dual()
            .unwrap();
        let u1_y =
            GradedSpace::try_new(Arc::clone(&u1_rule), [(U1Irrep::new(1), 3)], false).unwrap();
        let u1_z0 =
            GradedSpace::try_new(Arc::clone(&u1_rule), [(U1Irrep::new(-2), 2)], false).unwrap();
        let u1_z1 = GradedSpace::try_new(u1_rule, [(U1Irrep::new(0), 1)], false).unwrap();
        assert_asymmetric_cuda_plan_parity(
            &runtime, &u1_x0, &u1_x1, &u1_y, &u1_z0, &u1_z1, 748_210,
        );

        let product_rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
        let product_x0 = GradedSpace::try_new(
            Arc::clone(&product_rule),
            [(product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1)],
            false,
        )
        .unwrap();
        let product_x1 = GradedSpace::try_new(
            Arc::clone(&product_rule),
            [(product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 2)],
            false,
        )
        .unwrap()
        .try_dual()
        .unwrap();
        let product_y = GradedSpace::try_new(
            Arc::clone(&product_rule),
            [(product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2)],
            false,
        )
        .unwrap();
        let product_z0 = GradedSpace::try_new(
            Arc::clone(&product_rule),
            [(product_sector(Z2Irrep::EVEN, U1Irrep::new(-2)), 1)],
            false,
        )
        .unwrap();
        let product_z1 = GradedSpace::try_new(
            product_rule,
            [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 1)],
            false,
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
            GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)], false).unwrap();
        let bad =
            GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(1), 2)], false).unwrap();
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
            GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)], false).unwrap();
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
            GradedSpace::try_new(Arc::clone(provider), [(U1Irrep::new(0), degeneracy)], false)
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
            GradedSpace::try_new(
                Arc::clone(&provider),
                [(U1Irrep::new(0), degeneracy)],
                false,
            )
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
        let space = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)], false).unwrap();
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
            GradedSpace::try_new(
                Arc::clone(&provider),
                [(U1Irrep::new(0), degeneracy)],
                false,
            )
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
