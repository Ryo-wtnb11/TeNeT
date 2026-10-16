//! Single authority for how leg labels flow through a pairwise contraction plan.
//!
//! `NetworkIR` construction (`ir::validate_einsum_support`) guarantees every
//! label is either an output label on exactly one operand or a contracted label
//! on exactly two operands. A pairwise step therefore contracts exactly the
//! labels shared by its operands, and the planner, cost model, plan validation
//! and schedule compiler must all agree on the resulting leg order.

use std::collections::HashMap;

use crate::error::{ContractError, Result};
use crate::ir::NetworkIR;
use crate::labels::{TemporaryLabel, TensorId};
use crate::optimizer::ContractionStep;

/// Open legs of a pairwise contraction: lhs-only legs, then rhs-only legs, each
/// in operand order.
pub(crate) fn pair_result_labels(
    lhs: &[TemporaryLabel],
    rhs: &[TemporaryLabel],
) -> Vec<TemporaryLabel> {
    let mut labels = lhs
        .iter()
        .filter(|label| !rhs.contains(label))
        .cloned()
        .collect::<Vec<_>>();
    labels.extend(rhs.iter().filter(|label| !lhs.contains(label)).cloned());
    labels
}

/// Labels a planner declares for a step: the last step of a plan emits the
/// requested output order, every earlier step the pairwise order.
pub(crate) fn declared_step_labels(
    lhs: &[TemporaryLabel],
    rhs: &[TemporaryLabel],
    is_last_step: bool,
    output_labels: &[TemporaryLabel],
) -> Vec<TemporaryLabel> {
    if is_last_step {
        output_labels.to_vec()
    } else {
        pair_result_labels(lhs, rhs)
    }
}

/// Map each tensor id to its single later consuming step and whether it is that
/// step's lhs. In a pairwise tree every operand and intermediate is consumed
/// once, so one forward pass replaces a per-step scan of later steps.
pub(crate) fn consumers(steps: &[ContractionStep]) -> HashMap<TensorId, (usize, bool)> {
    let mut consumers = HashMap::with_capacity(steps.len() * 2);
    for (index, step) in steps.iter().enumerate() {
        consumers.insert(step.lhs(), (index, true));
        consumers.insert(step.rhs(), (index, false));
    }
    consumers
}

/// Leg-label order of every input and planned intermediate, as the executor
/// tracks it before any next-use reorientation.
pub(crate) fn planned_label_orders(
    ir: &NetworkIR,
    steps: &[ContractionStep],
) -> Result<HashMap<TensorId, Vec<TemporaryLabel>>> {
    let mut labels_by_id = ir
        .tensors()
        .iter()
        .map(|tensor| (tensor.id(), tensor.labels().to_vec()))
        .collect::<HashMap<_, _>>();

    for (step_index, step) in steps.iter().enumerate() {
        let labels = {
            let operand = |id: TensorId, side: &str| {
                labels_by_id.get(&id).ok_or_else(|| {
                    ContractError::InvalidContractionPlan(format!(
                        "step {step_index} {side} {} has no planned labels",
                        id.index()
                    ))
                })
            };
            pair_result_labels(operand(step.lhs(), "lhs")?, operand(step.rhs(), "rhs")?)
        };
        labels_by_id.insert(step.result(), labels);
    }

    Ok(labels_by_id)
}
