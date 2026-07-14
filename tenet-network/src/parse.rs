#![allow(dead_code)] // parsing helpers kept for planner unit tests only
use std::collections::BTreeMap;

use crate::error::{ContractError, Result};
use crate::ir::NetworkIR;
use crate::labels::TemporaryLabel;

/// Parse an einsum equation into TeNeT's [`NetworkIR`].
///
/// Supported syntax is the explicit/implicit label grammar used by NumPy
/// `einsum`, without ellipsis. Labels are temporary expression-local
/// identifiers; they are not persistent tensor indices.
///
/// Design reference:
/// - NumPy `einsum`:
///   <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>
pub fn parse_einsum(equation: &str) -> Result<NetworkIR> {
    let equation = equation.trim();
    if equation.is_empty() {
        return Err(ContractError::EmptyEquation);
    }
    if equation.contains("...") {
        return Err(ContractError::UnsupportedEllipsis);
    }

    let parts = equation.split("->").collect::<Vec<_>>();
    if parts.len() > 2 {
        return Err(ContractError::InvalidArrow);
    }

    let inputs = parse_inputs(parts[0])?;
    let output = if parts.len() == 2 {
        parse_label_list(parts[1].trim())?
    } else {
        infer_output_labels(&inputs)
    };

    NetworkIR::from_labels(inputs, output)
}

fn parse_inputs(input_text: &str) -> Result<Vec<Vec<TemporaryLabel>>> {
    let inputs = input_text
        .split(',')
        .map(str::trim)
        .map(parse_label_list)
        .collect::<Result<Vec<_>>>()?;
    if inputs.is_empty() || inputs.iter().any(Vec::is_empty) {
        return Err(ContractError::EmptyInput);
    }
    Ok(inputs)
}

pub(crate) struct ParsedLabelList {
    labels: Vec<TemporaryLabel>,
    codomain_rank: Option<usize>,
}

impl ParsedLabelList {
    pub(crate) fn labels(self) -> Vec<TemporaryLabel> {
        self.labels
    }

    pub(crate) fn codomain_rank(&self) -> Option<usize> {
        self.codomain_rank
    }
}

pub(crate) fn parse_label_list(text: &str) -> Result<Vec<TemporaryLabel>> {
    parse_label_list_with_split(text).map(ParsedLabelList::labels)
}

pub(crate) fn parse_label_list_with_split(text: &str) -> Result<ParsedLabelList> {
    // `;` (the codomain | domain separator, `@tensor`-style) is cosmetic for matching:
    // operands are matched positionally on the flattened label list, so the caller need
    // not track how a tensor's legs split into codomain/domain. The *output* `;` split is
    // applied after contraction (see `output_codomain_rank` in the executor). Strip it
    // here so both input and output sides flatten to a unique label order.
    let token_mode = text.chars().any(char::is_whitespace);
    if token_mode {
        let mut labels = Vec::new();
        let mut codomain_rank = None;
        for (part_index, part) in text.split(';').enumerate() {
            if part_index == 1 {
                codomain_rank = Some(labels.len());
            }
            for label in part.split_whitespace().filter(|label| !label.is_empty()) {
                labels.push(parse_label_token(label)?);
            }
        }
        Ok(ParsedLabelList {
            labels,
            codomain_rank,
        })
    } else {
        let mut labels = Vec::new();
        let mut codomain_rank = None;
        for label in text.chars() {
            if label == ';' {
                codomain_rank = Some(labels.len());
            } else {
                labels.push(parse_label_char(label)?);
            }
        }
        Ok(ParsedLabelList {
            labels,
            codomain_rank,
        })
    }
}

fn parse_label_char(label: char) -> Result<TemporaryLabel> {
    if label.is_ascii_alphabetic() {
        Ok(TemporaryLabel::from(label))
    } else {
        Err(ContractError::InvalidLabel(label.to_string()))
    }
}

fn parse_label_token(label: &str) -> Result<TemporaryLabel> {
    if label.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        Ok(TemporaryLabel::from(label))
    } else {
        Err(ContractError::InvalidLabel(label.to_string()))
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
