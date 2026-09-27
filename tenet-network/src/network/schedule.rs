use super::*;

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

#[expect(
    clippy::type_complexity,
    reason = "the planner returns the two-sided axis partition as an optional tuple"
)]
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
    let permutation = next_use_axes(labels, result_is_lhs, sibling_labels);
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
