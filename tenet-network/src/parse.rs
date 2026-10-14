//! Einsum-string helper for planner unit tests only. TeNeT has no public
//! einsum parser (#2029); production networks are built from label lists.
use std::collections::BTreeMap;

use crate::error::Result;
use crate::ir::NetworkIR;
use crate::labels::TemporaryLabel;

/// Parses `ab,bc->ac` (one character per label) or whitespace-separated
/// tokens (`x y, y z -> x z`); without `->` the output is every label that
/// occurs once. Only the network structure is validated, by
/// [`NetworkIR::from_labels`].
pub(crate) fn parse_einsum(equation: &str) -> Result<NetworkIR> {
    let (inputs, output) = match equation.split_once("->") {
        Some((inputs, output)) => (parse_inputs(inputs), parse_labels(output)),
        None => {
            let inputs = parse_inputs(equation);
            let output = infer_output_labels(&inputs);
            (inputs, output)
        }
    };
    NetworkIR::from_labels(inputs, output)
}

fn parse_inputs(text: &str) -> Vec<Vec<TemporaryLabel>> {
    text.split(',').map(parse_labels).collect()
}

fn parse_labels(text: &str) -> Vec<TemporaryLabel> {
    let text = text.trim();
    if text.contains(char::is_whitespace) {
        text.split_whitespace().map(TemporaryLabel::from).collect()
    } else {
        text.chars().map(TemporaryLabel::from).collect()
    }
}

fn infer_output_labels(inputs: &[Vec<TemporaryLabel>]) -> Vec<TemporaryLabel> {
    let mut counts = BTreeMap::<TemporaryLabel, usize>::new();
    for labels in inputs {
        for label in labels {
            *counts.entry(label.clone()).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .filter_map(|(label, count)| (count == 1).then_some(label))
        .collect()
}
