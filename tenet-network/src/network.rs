//! Contraction of a labeled tensor network over the user-layer
//! [`tenet::prelude::Tensor`].
//!
//! This is the execution half rewritten for the current user layer: the
//! planner ([`NetworkIR`], [`DenseCostModel`], [`ContractionPlan`]) is pure
//! structure, and each planned pairwise step lowers to
//! [`Tensor::contract`] plus orientation/final [`Tensor::permute`] calls,
//! mirroring the legacy `tenet-contract` executor over the old core.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use tenet::prelude::{
    ContractOverwriteCache, Dtype, Error, OverwriteOutcome, PermuteOverwriteCache, Runtime, Scalar,
    Tensor, TensorExecutionContext,
};

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
pub struct NetOperand<'a> {
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
/// [`PlannedNetwork::execute`], or one-shot [`Network::contract`].
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

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(message.to_string())
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

    /// Convenience constructor from `&str` labels (what the `tensor!`
    /// macro emits).
    pub fn from_names(
        operands: &[NetOperand<'_>],
        output: &[&str],
        output_codomain_rank: Option<usize>,
    ) -> Result<Self, Error> {
        Self::new(
            operands
                .iter()
                .map(|op| op.labels.iter().map(|&l| TemporaryLabel::from(l)).collect())
                .collect(),
            operands.iter().map(|op| op.conj).collect(),
            operands.iter().map(|op| op.codomain_split).collect(),
            output.iter().map(|&l| TemporaryLabel::from(l)).collect(),
            output_codomain_rank,
        )
    }

    /// Plan the contraction order for concrete operand tensors using the
    /// given optimizer. The plan is data-independent (labels + leg
    /// dimensions only) and can be executed repeatedly over same-shaped
    /// operands.
    pub fn plan(
        &self,
        tensors: &[&Tensor],
        optimizer: &(impl DenseContractionOptimizer + ?Sized),
    ) -> Result<PlannedNetwork, Error> {
        let (ir, infos) = self.lower(tensors)?;
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
        let schedule = compile_schedule(
            &ir,
            &plan,
            self.output_codomain_rank,
            &lowered_codomain_ranks,
        )?;
        Ok(PlannedNetwork {
            plan,
            conj: self.conj.clone(),
            input_codomain_ranks,
            schedule,
        })
    }

    /// Wrap an already-searched [`ContractionPlan`] (same topology) into a
    /// [`PlannedNetwork`] without re-running the order search. The plan is a
    /// pure pairwise order over operand ids and labels, valid for any leg
    /// dimensions of this topology, so a persisted plan (see the plan cache's
    /// disk save/restore) skips the cold optimal-order search on reuse.
    pub fn plan_with(
        &self,
        tensors: &[&Tensor],
        plan: ContractionPlan,
    ) -> Result<PlannedNetwork, Error> {
        let (ir, _infos) = self.lower(tensors)?;
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
        let schedule = compile_schedule(
            &ir,
            &plan,
            self.output_codomain_rank,
            &lowered_codomain_ranks,
        )?;
        Ok(PlannedNetwork {
            plan,
            conj: self.conj.clone(),
            input_codomain_ranks,
            schedule,
        })
    }

    /// Validate operand ranks and `;` splits and lower conj markers into the
    /// [`NetworkIR`] and per-operand cost infos shared by [`plan`](Self::plan)
    /// and [`plan_with`](Self::plan_with).
    fn lower(&self, tensors: &[&Tensor]) -> Result<(NetworkIR, Vec<DenseTensorInfo>), Error> {
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
    pub fn contract(&self, tensors: &[&Tensor]) -> Result<Tensor, Error> {
        let optimizer = tensors
            .first()
            .map(|tensor| tensor.runtime().plan_cache_config().optimizer)
            .unwrap_or_default();
        self.contract_with(tensors, &optimizer)
    }

    /// [`Self::contract`] with an explicit per-call [`Optimizer`] choice
    /// (still cached; the optimizer is part of the cache key). For a raw
    /// [`DenseContractionOptimizer`] implementation, use [`Self::plan`],
    /// which always plans fresh.
    pub fn contract_with(
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
    plan: ContractionPlan,
    conj: Vec<bool>,
    input_codomain_ranks: Vec<usize>,
    schedule: CompiledSchedule,
}

struct CompiledSchedule {
    slot_count: usize,
    input_ranks: Vec<usize>,
    steps: Vec<CompiledStep>,
    final_slot: usize,
    final_permutation: Option<(Vec<usize>, Vec<usize>)>,
}

struct CompiledStep {
    lhs_slot: usize,
    rhs_slot: usize,
    result_slot: usize,
    lhs_contract_axes: Vec<usize>,
    rhs_contract_axes: Vec<usize>,
    result_permutation: Option<(Vec<usize>, Vec<usize>)>,
}

/// Caller-owned tensor slots for repeated execution of a [`PlannedNetwork`].
#[derive(Default)]
pub struct NetworkExecutionWorkspace {
    slots: Vec<Option<Tensor>>,
    slot_producers: Vec<Option<(usize, bool)>>,
    intermediates: Vec<IntermediateBuffers>,
    tensor_context: Option<TensorExecutionContext>,
    tensor_runtime: Option<Runtime>,
    stats: NetworkExecutionStats,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NetworkExecutionStats {
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
    contract_cache: ContractOverwriteCache,
    orientation_cache: PermuteOverwriteCache,
}

impl NetworkExecutionWorkspace {
    pub(crate) fn slot_capacity(&self) -> usize {
        self.slots.capacity()
    }

    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.slot_producers.clear();
    }

    #[doc(hidden)]
    pub fn clear_intermediate_buffers(&mut self) {
        self.intermediates.clear();
    }

    #[doc(hidden)]
    pub fn stats(&self) -> NetworkExecutionStats {
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

    /// Run the plan over `tensors` (same operand order and shapes as
    /// given to [`Network::plan`]). Conj-marked operands are adjointed
    /// here; each pairwise step is one [`Tensor::contract`], intermediates
    /// are oriented for their next use, and the final tensor is permuted
    /// to the requested output label order and codomain/domain split.
    pub fn execute(&self, tensors: &[&Tensor]) -> Result<Tensor, Error> {
        self.execute_with_workspace(tensors, &mut NetworkExecutionWorkspace::default())
    }

    /// Run the compiled schedule while reusing its tensor-slot table and
    /// eligible host intermediate buffers. A returned [`Error`] preserves
    /// checked-out reusable buffers. Backend panics are treated as fatal and
    /// may discard workspace contents; the runtime already applies the same
    /// policy by poisoning its execution-state mutex after an unwind.
    pub fn execute_with_workspace(
        &self,
        tensors: &[&Tensor],
        workspace: &mut NetworkExecutionWorkspace,
    ) -> Result<Tensor, Error> {
        if tensors.len() != self.conj.len() {
            return Err(invalid(format!(
                "plan has {} operands but {} tensors were given",
                self.conj.len(),
                tensors.len()
            )));
        }

        workspace
            .slots
            .resize_with(self.schedule.slot_count, || None);
        workspace
            .slot_producers
            .resize(self.schedule.slot_count, None);
        workspace
            .intermediates
            .resize_with(self.schedule.steps.len(), IntermediateBuffers::default);
        let runtime = tensors[0].runtime();
        if workspace
            .tensor_runtime
            .as_ref()
            .is_none_or(|cached| !cached.shares_state_with(runtime))
        {
            workspace.tensor_context = Some(TensorExecutionContext::for_runtime(runtime)?);
            workspace.tensor_runtime = Some(runtime.clone());
            workspace.intermediates.clear();
            workspace
                .intermediates
                .resize_with(self.schedule.steps.len(), IntermediateBuffers::default);
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

fn identity_scalar(dtype: Dtype) -> Scalar {
    match dtype {
        Dtype::F64 => Scalar::F64(1.0),
        Dtype::C64 => Scalar::C64(tenet::prelude::Complex64::new(1.0, 0.0)),
    }
}

fn return_intermediate(
    workspace: &mut NetworkExecutionWorkspace,
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

fn compile_schedule(
    ir: &NetworkIR,
    plan: &ContractionPlan,
    output_codomain_rank: Option<usize>,
    input_codomain_ranks: &[usize],
) -> Result<CompiledSchedule, Error> {
    let labels_by_id = planned_label_orders(ir, plan)?;
    let consumers = build_consumers(plan.steps());
    let slot_count = ir.tensors().len() + plan.steps().len();
    let mut current_labels: Vec<Option<Vec<TemporaryLabel>>> = vec![None; slot_count];
    let mut current_codomain_ranks: Vec<Option<usize>> = vec![None; slot_count];
    let mut slots_by_id = HashMap::with_capacity(slot_count);
    for (slot, node) in ir.tensors().iter().enumerate() {
        slots_by_id.insert(node.id(), slot);
        current_labels[slot] = Some(node.labels().to_vec());
        current_codomain_ranks[slot] = Some(input_codomain_ranks[slot]);
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

        let result_permutation = compiled_intermediate_permutation(
            &result_labels,
            lhs_labels.len() - lhs_contract_axes.len(),
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
        current_labels[result_slot] = Some(result_labels);
        current_codomain_ranks[result_slot] = Some(result_codomain_rank);
        slots_by_id.insert(step.result(), result_slot);
        compiled_steps.push(CompiledStep {
            lhs_slot,
            rhs_slot,
            result_slot,
            lhs_contract_axes,
            rhs_contract_axes,
            result_permutation,
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
pub fn contract_network(
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
    network.contract(&tensors)
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
