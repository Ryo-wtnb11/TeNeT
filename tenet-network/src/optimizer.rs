use std::collections::{BTreeSet, HashMap};

use crate::cost::{BlockSparseCostModel, BlockSparseTensorInfo, DenseCostModel};
use crate::error::{ContractError, Result};
use crate::ir::NetworkIR;
use crate::labels::{TemporaryLabel, TensorId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionStep {
    lhs: TensorId,
    rhs: TensorId,
    result: TensorId,
    cost: usize,
    result_labels: Vec<TemporaryLabel>,
}

impl ContractionStep {
    pub fn new(
        lhs: TensorId,
        rhs: TensorId,
        result: TensorId,
        cost: usize,
        result_labels: Vec<TemporaryLabel>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            result,
            cost,
            result_labels,
        }
    }

    pub fn lhs(&self) -> TensorId {
        self.lhs
    }

    pub fn rhs(&self) -> TensorId {
        self.rhs
    }

    pub fn result(&self) -> TensorId {
        self.result
    }

    pub fn cost(&self) -> usize {
        self.cost
    }

    pub fn result_labels(&self) -> &[TemporaryLabel] {
        &self.result_labels
    }
}

#[derive(Debug, Clone)]
struct ActiveTensor {
    id: TensorId,
    labels: Vec<TemporaryLabel>,
}

pub fn greedy_order(ir: &NetworkIR, cost_model: &DenseCostModel) -> Result<Vec<ContractionStep>> {
    if ir.tensors().len() < 2 {
        return Err(ContractError::NotEnoughTensors);
    }

    let mut active = ir
        .tensors()
        .iter()
        .map(|tensor| ActiveTensor {
            id: tensor.id(),
            labels: tensor.labels().to_vec(),
        })
        .collect::<Vec<_>>();
    let mut steps = Vec::new();

    while active.len() > 1 {
        let mut best = None::<(usize, usize, usize)>;
        for lhs_index in 0..active.len() {
            for rhs_index in (lhs_index + 1)..active.len() {
                let cost =
                    cost_model.pair_cost(&active[lhs_index].labels, &active[rhs_index].labels);
                match best {
                    Some((_, _, best_cost)) if best_cost <= cost => {}
                    _ => best = Some((lhs_index, rhs_index, cost)),
                }
            }
        }

        let (lhs_index, rhs_index, cost) = best.ok_or(ContractError::NotEnoughTensors)?;
        let rhs = active.remove(rhs_index);
        let lhs = active.remove(lhs_index);
        let remaining_labels = active
            .iter()
            .map(|tensor| tensor.labels.clone())
            .collect::<Vec<_>>();
        let result_labels = cost_model.contraction_result_labels_with_remaining(
            &lhs.labels,
            &rhs.labels,
            &remaining_labels,
            ir.output_labels(),
        );
        let result_id = TensorId::new(ir.tensors().len() + steps.len());
        steps.push(ContractionStep::new(
            lhs.id,
            rhs.id,
            result_id,
            cost,
            result_labels.clone(),
        ));
        active.push(ActiveTensor {
            id: result_id,
            labels: result_labels,
        });
    }

    charge_dense_orientation_costs(ir, cost_model, &mut steps)?;
    Ok(steps)
}

/// Optimizer interface for dense-shape contraction plans.
///
/// Implement this trait to plug in a custom order search while reusing TeNeT's
/// parser, plan validation, and executors.
///
/// ```text
/// use tenet_contract::prelude::*;
///
/// let ir = parse_einsum("ab,bc->ac")?;
/// let infos = vec![
///     DenseTensorInfo::new(vec![2, 3]),
///     DenseTensorInfo::new(vec![3, 4]),
/// ];
/// let cost = DenseCostModel::from_network(&ir, &infos)?;
/// let plan = ContractionPlan::from_dense_optimizer(&ir, &GreedyDenseOptimizer, &cost)?;
/// assert_eq!(plan.active_pair_path()?, vec![ActivePair::new(0, 1)]);
/// ```
pub trait DenseContractionOptimizer {
    fn optimize(&self, ir: &NetworkIR, cost_model: &DenseCostModel)
        -> Result<Vec<ContractionStep>>;
}

/// Optimizer interface for Abelian block-sparse contraction plans.
///
/// The optimizer receives a block-aware cost model, so it can select a different
/// order than dense-shape FLOP estimates would suggest.
pub trait BlockSparseContractionOptimizer<S: Ord + Clone> {
    fn optimize(
        &self,
        ir: &NetworkIR,
        cost_model: &BlockSparseCostModel<S>,
    ) -> Result<Vec<ContractionStep>>;
}

/// Default dense greedy contraction optimizer.
///
/// This is cheap to run and deterministic. For hard networks, import an
/// external active-pair path (e.g. from the optional cotengra backend) and
/// build a [`crate::ContractionPlan`] from that path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GreedyDenseOptimizer;

impl DenseContractionOptimizer for GreedyDenseOptimizer {
    fn optimize(
        &self,
        ir: &NetworkIR,
        cost_model: &DenseCostModel,
    ) -> Result<Vec<ContractionStep>> {
        greedy_order(ir, cost_model)
    }
}

/// Default Abelian block-sparse greedy contraction optimizer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GreedyBlockSparseOptimizer;

impl<S: Ord + Clone> BlockSparseContractionOptimizer<S> for GreedyBlockSparseOptimizer {
    fn optimize(
        &self,
        ir: &NetworkIR,
        cost_model: &BlockSparseCostModel<S>,
    ) -> Result<Vec<ContractionStep>> {
        greedy_order_block_sparse(ir, cost_model)
    }
}

/// Optimizer that contracts labels in a caller-provided order.
///
/// This implements explicit contraction-label ordering over [`NetworkIR`].
///
/// The order should contain only internal labels that appear exactly twice.
/// See crate-level design references for the index-notation and path-ordering
/// APIs that motivated this boundary.
///
/// ```text
/// use tenet_contract::prelude::*;
///
/// let ir = parse_einsum("ab,bc,cd->ad")?;
/// let infos = vec![
///     DenseTensorInfo::new(vec![2, 3]),
///     DenseTensorInfo::new(vec![3, 4]),
///     DenseTensorInfo::new(vec![4, 5]),
/// ];
/// let cost = DenseCostModel::from_network(&ir, &infos)?;
/// let optimizer = LabelOrderDenseOptimizer::new(vec![
///     TemporaryLabel::new("b"),
///     TemporaryLabel::new("c"),
/// ]);
/// let plan = ContractionPlan::from_dense_optimizer(&ir, &optimizer, &cost)?;
/// assert_eq!(plan.active_pair_path()?.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelOrderDenseOptimizer {
    order: Vec<TemporaryLabel>,
}

impl LabelOrderDenseOptimizer {
    /// Create an optimizer from an explicit label order.
    pub fn new(order: Vec<TemporaryLabel>) -> Self {
        Self { order }
    }

    /// Return the explicit label order.
    pub fn order(&self) -> &[TemporaryLabel] {
        &self.order
    }
}

impl DenseContractionOptimizer for LabelOrderDenseOptimizer {
    fn optimize(
        &self,
        ir: &NetworkIR,
        cost_model: &DenseCostModel,
    ) -> Result<Vec<ContractionStep>> {
        dense_order_from_labels(ir, cost_model, &self.order)
    }
}

impl<S: Ord + Clone> BlockSparseContractionOptimizer<S> for LabelOrderDenseOptimizer {
    fn optimize(
        &self,
        ir: &NetworkIR,
        cost_model: &BlockSparseCostModel<S>,
    ) -> Result<Vec<ContractionStep>> {
        block_sparse_order_from_labels(ir, cost_model, &self.order)
    }
}

/// Cost comparison between a dense plan and TeNeT's dense greedy baseline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePlanCostReport {
    plan_cost: usize,
    greedy_cost: usize,
}

impl DensePlanCostReport {
    /// Construct a report from precomputed cost totals.
    pub fn new(plan_cost: usize, greedy_cost: usize) -> Self {
        Self {
            plan_cost,
            greedy_cost,
        }
    }

    /// Total estimated cost of the inspected plan.
    pub fn plan_cost(&self) -> usize {
        self.plan_cost
    }

    /// Total estimated cost of the greedy baseline.
    pub fn greedy_cost(&self) -> usize {
        self.greedy_cost
    }

    /// Return true when the inspected plan is more expensive than greedy.
    pub fn is_suboptimal(&self) -> bool {
        self.plan_cost > self.greedy_cost
    }
}

pub fn dense_plan_cost_report(
    steps: &[ContractionStep],
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
) -> Result<DensePlanCostReport> {
    let plan_cost = steps.iter().map(ContractionStep::cost).sum();
    let greedy_cost = greedy_order(ir, cost_model)?
        .iter()
        .map(ContractionStep::cost)
        .sum();
    Ok(DensePlanCostReport::new(plan_cost, greedy_cost))
}

pub fn dense_order_from_labels(
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
    order: &[TemporaryLabel],
) -> Result<Vec<ContractionStep>> {
    if ir.tensors().len() < 2 {
        return Err(ContractError::NotEnoughTensors);
    }

    validate_explicit_order_labels(ir, order)?;

    let mut active = ir
        .tensors()
        .iter()
        .map(|tensor| ActiveTensor {
            id: tensor.id(),
            labels: tensor.labels().to_vec(),
        })
        .collect::<Vec<_>>();
    let mut steps = Vec::new();

    let mut remaining_order = order.to_vec();
    while let Some(label) = remaining_order.first().cloned() {
        let Some(lhs_index) = active
            .iter()
            .position(|tensor| tensor.labels.contains(&label))
        else {
            remaining_order.remove(0);
            continue;
        };
        let Some(rhs_index) = active
            .iter()
            .enumerate()
            .skip(lhs_index + 1)
            .find_map(|(position, tensor)| tensor.labels.contains(&label).then_some(position))
        else {
            remaining_order.remove(0);
            continue;
        };

        let shared_order_labels = active[lhs_index]
            .labels
            .iter()
            .filter(|candidate| {
                active[rhs_index].labels.contains(candidate) && remaining_order.contains(candidate)
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        remaining_order.retain(|candidate| !shared_order_labels.contains(candidate));
        push_dense_active_contraction_step(
            ir,
            cost_model,
            &mut active,
            &mut steps,
            lhs_index,
            rhs_index,
        )?;
    }

    while active.len() > 1 {
        let rhs_index = active.len() - 1;
        let lhs_index = active.len() - 2;
        push_dense_active_contraction_step(
            ir,
            cost_model,
            &mut active,
            &mut steps,
            lhs_index,
            rhs_index,
        )?;
    }

    charge_dense_orientation_costs(ir, cost_model, &mut steps)?;
    Ok(steps)
}

pub(crate) fn charge_dense_orientation_costs(
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
    steps: &mut [ContractionStep],
) -> Result<()> {
    let planned_labels = planned_dense_label_orders(ir, steps)?;
    // Resolve each result's single later consumer once (O(steps)); the topology
    // (lhs/rhs/result) is fixed even though this loop mutates step costs.
    let consumers = dense_consumers(steps);
    let mut active = ir
        .tensors()
        .iter()
        .map(|tensor| (tensor.id(), tensor.labels().to_vec()))
        .collect::<HashMap<_, _>>();

    for step_index in 0..steps.len() {
        let lhs_id = steps[step_index].lhs;
        let rhs_id = steps[step_index].rhs;
        let result_id = steps[step_index].result;
        let lhs = active.remove(&lhs_id).ok_or_else(|| {
            ContractError::InvalidContractionPlan(format!(
                "step {step_index} lhs {} is not active",
                lhs_id.index()
            ))
        })?;
        let rhs = active.remove(&rhs_id).ok_or_else(|| {
            ContractError::InvalidContractionPlan(format!(
                "step {step_index} rhs {} is not active",
                rhs_id.index()
            ))
        })?;

        let raw_codomain_rank = lhs.iter().filter(|label| !rhs.contains(label)).count();
        let mut labels = pair_result_labels(&lhs, &rhs);

        if let Some(&(future_index, result_is_lhs)) = consumers.get(&result_id) {
            let sibling_id = if result_is_lhs {
                steps[future_index].rhs
            } else {
                steps[future_index].lhs
            };
            let sibling_labels = planned_labels.get(&sibling_id).ok_or_else(|| {
                ContractError::InvalidContractionPlan(format!(
                    "step {step_index} future sibling {} has no planned labels",
                    sibling_id.index()
                ))
            })?;
            let (needs_orientation, oriented_labels) = dense_orientation_for_next_use(
                &labels,
                raw_codomain_rank,
                result_is_lhs,
                sibling_labels,
            );
            if needs_orientation {
                steps[step_index].cost = steps[step_index]
                    .cost
                    .saturating_add(cost_model.tensor_size(&labels));
                labels = oriented_labels;
            }
        }

        active.insert(result_id, labels);
    }

    if let Some(final_labels) = active.values().next() {
        if !steps.is_empty() && final_labels.as_slice() != ir.output_labels() {
            let final_permute_cost = cost_model.tensor_size(final_labels);
            let last_step = steps.len() - 1;
            steps[last_step].cost = steps[last_step].cost.saturating_add(final_permute_cost);
        }
    }

    Ok(())
}

fn planned_dense_label_orders(
    ir: &NetworkIR,
    steps: &[ContractionStep],
) -> Result<HashMap<TensorId, Vec<TemporaryLabel>>> {
    let mut labels_by_id = ir
        .tensors()
        .iter()
        .map(|tensor| (tensor.id(), tensor.labels().to_vec()))
        .collect::<HashMap<_, _>>();
    let mut active = labels_by_id.clone();

    for (step_index, step) in steps.iter().enumerate() {
        active.remove(&step.lhs).ok_or_else(|| {
            ContractError::InvalidContractionPlan(format!(
                "step {step_index} lhs {} is not active while planning labels",
                step.lhs.index()
            ))
        })?;
        active.remove(&step.rhs).ok_or_else(|| {
            ContractError::InvalidContractionPlan(format!(
                "step {step_index} rhs {} is not active while planning labels",
                step.rhs.index()
            ))
        })?;
        let labels = step.result_labels.clone();
        labels_by_id.insert(step.result, labels.clone());
        active.insert(step.result, labels);
    }

    Ok(labels_by_id)
}

fn pair_result_labels(lhs: &[TemporaryLabel], rhs: &[TemporaryLabel]) -> Vec<TemporaryLabel> {
    let mut labels = lhs
        .iter()
        .filter(|label| !rhs.contains(label))
        .cloned()
        .collect::<Vec<_>>();
    labels.extend(rhs.iter().filter(|label| !lhs.contains(label)).cloned());
    labels
}

fn dense_orientation_for_next_use(
    labels: &[TemporaryLabel],
    raw_codomain_rank: usize,
    result_is_lhs: bool,
    sibling_labels: &[TemporaryLabel],
) -> (bool, Vec<TemporaryLabel>) {
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
    let rank = labels.len();
    let identity = cod_axes.iter().copied().eq(0..raw_codomain_rank)
        && dom_axes.iter().copied().eq(raw_codomain_rank..rank);
    if identity {
        return (false, labels.to_vec());
    }

    let mut oriented_labels = Vec::with_capacity(labels.len());
    oriented_labels.extend(cod_axes.iter().map(|&axis| labels[axis].clone()));
    oriented_labels.extend(dom_axes.iter().map(|&axis| labels[axis].clone()));
    (true, oriented_labels)
}

/// Map each tensor id to its single later consuming step and whether it is that
/// step's lhs — one forward pass, so `charge_dense_orientation_costs` drops from
/// O(steps²) to O(steps). Mirrors `network::build_consumers`.
fn dense_consumers(steps: &[ContractionStep]) -> HashMap<TensorId, (usize, bool)> {
    let mut consumers = HashMap::with_capacity(steps.len() * 2);
    for (index, step) in steps.iter().enumerate() {
        consumers.insert(step.lhs, (index, true));
        consumers.insert(step.rhs, (index, false));
    }
    consumers
}

fn push_dense_active_contraction_step(
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
    active: &mut Vec<ActiveTensor>,
    steps: &mut Vec<ContractionStep>,
    lhs_index: usize,
    rhs_index: usize,
) -> Result<()> {
    if lhs_index >= rhs_index || rhs_index >= active.len() {
        return Err(ContractError::InvalidContractionPlan(format!(
            "invalid active pair ({lhs_index}, {rhs_index}) for {} active tensors",
            active.len()
        )));
    }

    let rhs = active.remove(rhs_index);
    let lhs = active.remove(lhs_index);
    let remaining_labels = active
        .iter()
        .map(|tensor| tensor.labels.clone())
        .collect::<Vec<_>>();
    let result_labels = cost_model.contraction_result_labels_with_remaining(
        &lhs.labels,
        &rhs.labels,
        &remaining_labels,
        ir.output_labels(),
    );
    let cost = cost_model.pair_cost(&lhs.labels, &rhs.labels);
    let result_id = TensorId::new(ir.tensors().len() + steps.len());
    steps.push(ContractionStep::new(
        lhs.id,
        rhs.id,
        result_id,
        cost,
        result_labels.clone(),
    ));
    active.push(ActiveTensor {
        id: result_id,
        labels: result_labels,
    });
    Ok(())
}

fn validate_explicit_order_labels(ir: &NetworkIR, order: &[TemporaryLabel]) -> Result<()> {
    let mut order_set = BTreeSet::new();
    for label in order {
        if !order_set.insert(label.clone()) {
            return Err(ContractError::InvalidContractionPlan(format!(
                "explicit contraction order contains duplicate label `{label}`"
            )));
        }
    }

    for edge in ir.edges() {
        let label = edge.label();
        let occurrence_count = edge.occurrences().len();
        if order_set.contains(label) {
            if edge.is_output() {
                return Err(ContractError::InvalidContractionPlan(format!(
                    "explicit contraction order label `{label}` is an output label"
                )));
            }
            if occurrence_count != 2 {
                return Err(ContractError::InvalidContractionPlan(format!(
                    "explicit contraction order label `{label}` must occur exactly twice, got {occurrence_count}"
                )));
            }
        } else if edge.is_output() {
            if occurrence_count != 1 {
                return Err(ContractError::InvalidContractionPlan(format!(
                    "output label `{label}` must occur exactly once for explicit contraction order, got {occurrence_count}"
                )));
            }
        } else {
            return Err(ContractError::InvalidContractionPlan(format!(
                "contracted label `{label}` is absent from explicit contraction order"
            )));
        }
    }

    for label in order {
        if !ir.edges().iter().any(|edge| edge.label() == label) {
            return Err(ContractError::InvalidContractionPlan(format!(
                "explicit contraction order label `{label}` does not occur in inputs"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ActiveBlockSparseTensor<S: Ord + Clone> {
    id: TensorId,
    labels: Vec<TemporaryLabel>,
    info: BlockSparseTensorInfo<S>,
}

pub fn greedy_order_block_sparse<S: Ord + Clone>(
    ir: &NetworkIR,
    cost_model: &BlockSparseCostModel<S>,
) -> Result<Vec<ContractionStep>> {
    if ir.tensors().len() < 2 {
        return Err(ContractError::NotEnoughTensors);
    }

    let mut active = ir
        .tensors()
        .iter()
        .map(|tensor| {
            Ok(ActiveBlockSparseTensor {
                id: tensor.id(),
                labels: tensor.labels().to_vec(),
                info: cost_model.tensor_info(tensor.id())?.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut steps = Vec::new();

    while active.len() > 1 {
        let mut best = None::<(usize, usize, usize)>;
        for lhs_index in 0..active.len() {
            for rhs_index in (lhs_index + 1)..active.len() {
                let cost =
                    cost_model.pair_cost(&active[lhs_index].info, &active[rhs_index].info)?;
                match best {
                    Some((_, _, best_cost)) if best_cost <= cost => {}
                    _ => best = Some((lhs_index, rhs_index, cost)),
                }
            }
        }

        let (lhs_index, rhs_index, cost) = best.ok_or(ContractError::NotEnoughTensors)?;
        let rhs = active.remove(rhs_index);
        let lhs = active.remove(lhs_index);
        let remaining_labels = active
            .iter()
            .map(|tensor| tensor.labels.clone())
            .collect::<Vec<_>>();
        let result_info = cost_model.contraction_result_info(
            &lhs.info,
            &rhs.info,
            &remaining_labels,
            ir.output_labels(),
        )?;
        let result_id = TensorId::new(ir.tensors().len() + steps.len());
        let result_labels = contraction_result_labels(
            &lhs.labels,
            &rhs.labels,
            &remaining_labels,
            ir.output_labels(),
        );
        steps.push(ContractionStep::new(
            lhs.id,
            rhs.id,
            result_id,
            cost,
            result_labels.clone(),
        ));
        active.push(ActiveBlockSparseTensor {
            id: result_id,
            labels: result_labels,
            info: result_info,
        });
    }

    Ok(steps)
}

pub fn block_sparse_order_from_labels<S: Ord + Clone>(
    ir: &NetworkIR,
    cost_model: &BlockSparseCostModel<S>,
    order: &[TemporaryLabel],
) -> Result<Vec<ContractionStep>> {
    if ir.tensors().len() < 2 {
        return Err(ContractError::NotEnoughTensors);
    }

    validate_explicit_order_labels(ir, order)?;

    let mut active = ir
        .tensors()
        .iter()
        .map(|tensor| {
            Ok(ActiveBlockSparseTensor {
                id: tensor.id(),
                labels: tensor.labels().to_vec(),
                info: cost_model.tensor_info(tensor.id())?.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut steps = Vec::new();

    let mut remaining_order = order.to_vec();
    while let Some(label) = remaining_order.first().cloned() {
        let Some(lhs_index) = active
            .iter()
            .position(|tensor| tensor.labels.contains(&label))
        else {
            remaining_order.remove(0);
            continue;
        };
        let Some(rhs_index) = active
            .iter()
            .enumerate()
            .skip(lhs_index + 1)
            .find_map(|(position, tensor)| tensor.labels.contains(&label).then_some(position))
        else {
            remaining_order.remove(0);
            continue;
        };

        let shared_order_labels = active[lhs_index]
            .labels
            .iter()
            .filter(|candidate| {
                active[rhs_index].labels.contains(candidate) && remaining_order.contains(candidate)
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        remaining_order.retain(|candidate| !shared_order_labels.contains(candidate));
        push_block_sparse_active_contraction_step(
            ir,
            cost_model,
            &mut active,
            &mut steps,
            lhs_index,
            rhs_index,
        )?;
    }

    while active.len() > 1 {
        let rhs_index = active.len() - 1;
        let lhs_index = active.len() - 2;
        push_block_sparse_active_contraction_step(
            ir,
            cost_model,
            &mut active,
            &mut steps,
            lhs_index,
            rhs_index,
        )?;
    }

    Ok(steps)
}

fn push_block_sparse_active_contraction_step<S: Ord + Clone>(
    ir: &NetworkIR,
    cost_model: &BlockSparseCostModel<S>,
    active: &mut Vec<ActiveBlockSparseTensor<S>>,
    steps: &mut Vec<ContractionStep>,
    lhs_index: usize,
    rhs_index: usize,
) -> Result<()> {
    if lhs_index >= rhs_index || rhs_index >= active.len() {
        return Err(ContractError::InvalidContractionPlan(format!(
            "invalid active pair ({lhs_index}, {rhs_index}) for {} active tensors",
            active.len()
        )));
    }

    let rhs = active.remove(rhs_index);
    let lhs = active.remove(lhs_index);
    let remaining_labels = active
        .iter()
        .map(|tensor| tensor.labels.clone())
        .collect::<Vec<_>>();
    let result_info = cost_model.contraction_result_info(
        &lhs.info,
        &rhs.info,
        &remaining_labels,
        ir.output_labels(),
    )?;
    let cost = cost_model.pair_cost(&lhs.info, &rhs.info)?;
    let result_labels = contraction_result_labels(
        &lhs.labels,
        &rhs.labels,
        &remaining_labels,
        ir.output_labels(),
    );
    let result_id = TensorId::new(ir.tensors().len() + steps.len());
    steps.push(ContractionStep::new(
        lhs.id,
        rhs.id,
        result_id,
        cost,
        result_labels.clone(),
    ));
    active.push(ActiveBlockSparseTensor {
        id: result_id,
        labels: result_labels,
        info: result_info,
    });
    Ok(())
}

fn contraction_result_labels(
    lhs: &[TemporaryLabel],
    rhs: &[TemporaryLabel],
    remaining_labels: &[Vec<TemporaryLabel>],
    output_labels: &[TemporaryLabel],
) -> Vec<TemporaryLabel> {
    let contracted = lhs
        .iter()
        .filter(|label| {
            rhs.contains(label)
                && !output_labels.contains(label)
                && !remaining_labels.iter().any(|labels| labels.contains(label))
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut labels = lhs
        .iter()
        .filter(|label| !contracted.contains(label))
        .cloned()
        .collect::<Vec<_>>();
    for label in rhs {
        if !contracted.contains(label) && !labels.contains(label) {
            labels.push(label.clone());
        }
    }
    if remaining_labels.is_empty() {
        return output_labels.to_vec();
    }
    labels
}
