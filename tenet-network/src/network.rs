//! Contraction of a labeled tensor network over the user-layer
//! [`tenet::prelude::Tensor`].
//!
//! This is the execution half rewritten for the current user layer: the
//! planner ([`NetworkIR`], [`DenseCostModel`], [`ContractionPlan`]) is pure
//! structure, and each planned pairwise step lowers to
//! [`Tensor::contract`] plus orientation/final [`Tensor::permute`] calls,
//! mirroring the legacy `tenet-contract` executor over the old core.

use std::collections::HashMap;

use tenet::prelude::{Error, Tensor};

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
    pub(crate) inputs: Vec<Vec<TemporaryLabel>>,
    pub(crate) conj: Vec<bool>,
    pub(crate) codomain_splits: Vec<Option<usize>>,
    pub(crate) output: Vec<TemporaryLabel>,
    /// Number of output labels on the codomain side (`;` position);
    /// `None` = all-codomain output.
    pub(crate) output_codomain_rank: Option<usize>,
}

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
            inputs,
            conj,
            codomain_splits,
            output,
            output_codomain_rank,
        })
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
        Ok(PlannedNetwork {
            ir,
            plan,
            conj: self.conj.clone(),
            output_codomain_rank: self.output_codomain_rank,
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
        Ok(PlannedNetwork {
            ir,
            plan,
            conj: self.conj.clone(),
            output_codomain_rank: self.output_codomain_rank,
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
    ir: NetworkIR,
    plan: ContractionPlan,
    conj: Vec<bool>,
    output_codomain_rank: Option<usize>,
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
        if tensors.len() != self.conj.len() {
            return Err(invalid(format!(
                "plan has {} operands but {} tensors were given",
                self.conj.len(),
                tensors.len()
            )));
        }

        // Active tensors keyed by planner TensorId, each with its current
        // leg labels in the tensor's flat leg order.
        let mut active: HashMap<TensorId, (Tensor, Vec<TemporaryLabel>)> = HashMap::new();
        for (i, (node, &tensor)) in self.ir.tensors().iter().zip(tensors).enumerate() {
            let lowered = if self.conj[i] {
                tensor.adjoint()?
            } else {
                tensor.clone()
            };
            if node.labels().len() != lowered.rank() {
                return Err(invalid(format!(
                    "operand {i} has {} labels but tensor rank {}",
                    node.labels().len(),
                    lowered.rank()
                )));
            }
            active.insert(TensorId::new(i), (lowered, node.labels().to_vec()));
        }

        let labels_by_id = planned_label_orders(&self.ir, &self.plan)?;
        let steps = self.plan.steps();
        let consumers = build_consumers(steps);
        for step in steps.iter() {
            let (lt, ll) = active
                .remove(&step.lhs())
                .ok_or_else(|| invalid("lhs operand already consumed"))?;
            let (rt, rl) = active
                .remove(&step.rhs())
                .ok_or_else(|| invalid("rhs operand already consumed"))?;

            // Contracted = labels shared by both operands and absent from
            // the step result (batch labels were rejected at build time).
            let mut a_contracted = Vec::new();
            let mut b_contracted = Vec::new();
            for (ai, la) in ll.iter().enumerate() {
                if let Some(bi) = rl.iter().position(|x| x == la) {
                    if step.result_labels().contains(la) {
                        return Err(invalid(format!(
                            "batch label `{la}` (shared + in result) unsupported"
                        )));
                    }
                    a_contracted.push(ai);
                    b_contracted.push(bi);
                }
            }

            let c = lt.contract(&rt, &a_contracted, &b_contracted)?;
            // `contract` returns (open lhs legs ascending <- open rhs legs
            // ascending); track labels in that flat order.
            let mut c_labels: Vec<TemporaryLabel> = ll
                .iter()
                .enumerate()
                .filter(|(i, _)| !a_contracted.contains(i))
                .map(|(_, l)| l.clone())
                .collect();
            c_labels.extend(
                rl.iter()
                    .enumerate()
                    .filter(|(i, _)| !b_contracted.contains(i))
                    .map(|(_, l)| l.clone()),
            );
            let (c, c_labels) = orient_intermediate_for_next_use(
                c,
                c_labels,
                step.result(),
                steps,
                &consumers,
                &labels_by_id,
            )?;
            active.insert(step.result(), (c, c_labels));
        }

        let final_id = steps
            .last()
            .map(|s| s.result())
            .unwrap_or_else(|| TensorId::new(0));
        let (result, labels) = active
            .remove(&final_id)
            .ok_or_else(|| invalid("no final tensor produced"))?;

        let output = self.ir.output_labels();
        let split = self.output_codomain_rank.unwrap_or(output.len());
        let cod_axes = label_positions(&output[..split], &labels)?;
        let dom_axes = label_positions(&output[split..], &labels)?;
        let identity = result.codomain_rank() == cod_axes.len()
            && cod_axes
                .iter()
                .chain(dom_axes.iter())
                .copied()
                .eq(0..labels.len());
        if identity {
            return Ok(result);
        }
        result.permute(&cod_axes, &dom_axes)
    }
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

/// Materialize an intermediate as (codomain <- domain) matching its next
/// pairwise use: a left child as (open_to_parent <- contracted_with_sibling),
/// a right child mirrored. Keeps later `contract` calls from re-bending
/// surviving legs path-dependently (same convention as the legacy executor
/// and TensorOperations.jl contextual temporaries).
fn orient_intermediate_for_next_use(
    tensor: Tensor,
    labels: Vec<TemporaryLabel>,
    result_id: TensorId,
    steps: &[ContractionStep],
    consumers: &HashMap<TensorId, (usize, bool)>,
    labels_by_id: &HashMap<TensorId, Vec<TemporaryLabel>>,
) -> Result<(Tensor, Vec<TemporaryLabel>), Error> {
    let Some(&(future_index, result_is_lhs)) = consumers.get(&result_id) else {
        return Ok((tensor, labels));
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
    let (cod_axes, dom_axes) = if result_is_lhs {
        (open_axes, contracted_axes)
    } else {
        (contracted_axes, open_axes)
    };
    let cod_rank = tensor.codomain_rank();
    let rank = tensor.rank();
    if cod_axes.iter().copied().eq(0..cod_rank) && dom_axes.iter().copied().eq(cod_rank..rank) {
        return Ok((tensor, labels));
    }
    let mut oriented_labels = Vec::with_capacity(labels.len());
    oriented_labels.extend(cod_axes.iter().map(|&axis| labels[axis].clone()));
    oriented_labels.extend(dom_axes.iter().map(|&axis| labels[axis].clone()));
    let oriented = tensor.permute(&cod_axes, &dom_axes)?;
    Ok((oriented, oriented_labels))
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
