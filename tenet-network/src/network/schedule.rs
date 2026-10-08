use super::*;
use crate::stepflow::{consumers, pair_result_labels, planned_label_orders};

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

pub(super) fn compile_schedule(
    ir: &NetworkIR,
    plan: &ContractionPlan,
    output_codomain_rank: Option<usize>,
    input_codomain_ranks: &[usize],
) -> Result<CompiledSchedule, Error> {
    let contracted_input_pairs = contracted_input_pairs(ir);
    let labels_by_id = planned_label_orders(ir, plan.steps()).map_err(invalid)?;
    let consumers = consumers(plan.steps());
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
        let mut result_labels = pair_result_labels(&lhs_labels, &rhs_labels);

        // The step's output orientation is its `ContractSpec` split (TensorOperations
        // `pAB`): the next-use orientation for an intermediate, the requested
        // output orientation for the schedule result. The engine writes it in
        // its own output transform, so no step issues a separate permute.
        let (codomain, domain) = match consumers.get(&step.result()) {
            Some(&(future_index, result_is_lhs)) => {
                let future_step = &plan.steps()[future_index];
                let sibling_id = if result_is_lhs {
                    future_step.rhs()
                } else {
                    future_step.lhs()
                };
                let sibling_labels = labels_by_id
                    .get(&sibling_id)
                    .ok_or_else(|| invalid("future sibling labels missing"))?;
                next_use_axes(&result_labels, result_is_lhs, sibling_labels)
            }
            None => {
                let output = ir.output_labels();
                let split = output_codomain_rank.unwrap_or(output.len());
                (
                    label_positions(&output[..split], &result_labels)?,
                    label_positions(&output[split..], &result_labels)?,
                )
            }
        };
        result_labels = codomain
            .iter()
            .chain(&domain)
            .map(|&axis| result_labels[axis].clone())
            .collect();
        current_labels[result_slot] = Some(result_labels);
        current_codomain_ranks[result_slot] = Some(codomain.len());
        authority_input_slots[result_slot] = Some(authority_input_slot);
        slots_by_id.insert(step.result(), result_slot);
        compiled_steps.push(CompiledStep {
            lhs_slot,
            rhs_slot,
            result_slot,
            lhs_contract_axes,
            rhs_contract_axes,
            codomain,
            domain,
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

#[cfg(test)]
mod flow_tests {
    use super::*;
    use crate::cost::{DenseCostModel, DenseTensorInfo};
    use crate::optimizer::{
        charge_dense_orientation_costs, ContractionStep, DenseContractionOptimizer,
        GreedyDenseOptimizer,
    };
    use crate::parse::parse_einsum;

    /// The cost the real planner path (`charge_dense_orientation_costs`, run
    /// inside the optimizer) charges per step must equal the orientation work
    /// the compiled schedule actually performs: one result-sized charge per
    /// intermediate the schedule orients away from the pairwise order, plus the
    /// final permute when the last result is not already in output order.
    #[test]
    fn planner_charges_match_schedule_orientation() {
        let ir = parse_einsum("abx,xcy,ydz,zea->bcde").unwrap();
        let infos = [(2, 3, 4), (4, 5, 6), (6, 7, 8), (8, 9, 2)]
            .map(|(p, q, r)| DenseTensorInfo::new(vec![p, q, r]))
            .to_vec();
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let steps = GreedyDenseOptimizer.optimize(&ir, &cost).unwrap();

        // Uncharged baseline: re-running the charge pass on zero-cost steps
        // yields exactly the orientation charges.
        let mut charged: Vec<_> = steps
            .iter()
            .map(|s| {
                ContractionStep::new(s.lhs(), s.rhs(), s.result(), 0, s.result_labels().to_vec())
            })
            .collect();
        charge_dense_orientation_costs(&ir, &cost, &mut charged).unwrap();

        let plan = ContractionPlan::from_steps(&ir, steps).unwrap();
        let schedule = compile_schedule(&ir, &plan, None, &[1; 4]).unwrap();

        let mut labels: HashMap<usize, Vec<TemporaryLabel>> = ir
            .tensors()
            .iter()
            .enumerate()
            .map(|(slot, node)| (slot, node.labels().to_vec()))
            .collect();
        let mut oriented_any = false;
        let last = schedule.steps.len() - 1;
        for (index, compiled) in schedule.steps.iter().enumerate() {
            let lhs = labels.remove(&compiled.lhs_slot).unwrap();
            let rhs = labels.remove(&compiled.rhs_slot).unwrap();
            let raw = pair_result_labels(&lhs, &rhs);
            let oriented: Vec<_> = compiled
                .codomain
                .iter()
                .chain(&compiled.domain)
                .map(|&axis| raw[axis].clone())
                .collect();
            let mut expected = 0;
            if index < last {
                if oriented != raw {
                    oriented_any = true;
                    expected += cost.tensor_size(&raw);
                }
            } else if raw != ir.output_labels() {
                expected += cost.tensor_size(&raw);
            }
            assert_eq!(charged[index].cost(), expected, "step {index}");
            labels.insert(compiled.result_slot, oriented);
        }
        assert!(schedule.steps.len() >= 3);
        assert!(
            oriented_any,
            "fixture must exercise a non-trivial orientation"
        );
    }
}
