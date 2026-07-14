use std::collections::{BTreeMap, BTreeSet};

use crate::error::{ContractError, Result};
use crate::labels::{TemporaryLabel, TensorAxis, TensorId};

/// One input tensor in a parsed einsum network.
///
/// A node stores only the tensor's expression-local id and its temporary axis
/// labels. The labels follow the usual `einsum` convention; they are not
/// persistent tensor indices.
///
/// Design references:
/// - NumPy `einsum` subscript/output grammar:
///   <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>
/// - opt_einsum path format:
///   <https://optimized-einsum.readthedocs.io/en/stable/path_finding.html#format-of-the-path>
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorNode {
    id: TensorId,
    labels: Vec<TemporaryLabel>,
}

impl TensorNode {
    pub fn new(id: TensorId, labels: Vec<TemporaryLabel>) -> Self {
        Self { id, labels }
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn rank(&self) -> usize {
        self.labels.len()
    }

    pub fn labels(&self) -> &[TemporaryLabel] {
        &self.labels
    }
}

/// One temporary label in a parsed einsum network.
///
/// A label may connect one or more tensor axes. Labels listed in the explicit
/// output are retained; the rest are contracted. This is the hyperedge view used
/// by contraction-order tools such as cotengra, but the type itself is
/// TeNeT-owned.
///
/// Design references:
/// - cotengra hyperedge/einsum network support:
///   <https://github.com/jcmgray/cotengra>
/// - NumPy `einsum` explicit output labels:
///   <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HyperEdge {
    label: TemporaryLabel,
    occurrences: Vec<TensorAxis>,
    output_position: Option<usize>,
}

impl HyperEdge {
    pub fn new(
        label: TemporaryLabel,
        occurrences: Vec<TensorAxis>,
        output_position: Option<usize>,
    ) -> Self {
        Self {
            label,
            occurrences,
            output_position,
        }
    }

    pub fn label(&self) -> &TemporaryLabel {
        &self.label
    }

    pub fn occurrences(&self) -> &[TensorAxis] {
        &self.occurrences
    }

    pub fn output_position(&self) -> Option<usize> {
        self.output_position
    }

    pub fn is_output(&self) -> bool {
        self.output_position.is_some()
    }
}

/// TeNeT's internal representation of a parsed einsum expression.
///
/// `NetworkIR` is not imported from any upstream package. It is a small
/// TeNeT-owned IR that stores tensor nodes, label hyperedges, and the requested
/// output label order. It exists so parsing, cost modeling, contraction-order
/// search, plan serialization, and backend execution can stay separate.
///
/// The design follows standard tensor-network/einsum conventions:
/// - an einsum expression is a list of input label lists plus optional output
///   labels;
/// - a contraction path is a sequence of pairwise contractions over the active
///   tensor list;
/// - external optimizers may provide the path while TeNeT keeps validation and
///   execution local.
///
/// Design references:
/// - NumPy `einsum` notation and explicit `->` output:
///   <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>
/// - opt_einsum pairwise path format:
///   <https://optimized-einsum.readthedocs.io/en/stable/path_finding.html#format-of-the-path>
/// - cotengra contraction trees and hyperedge tensor networks:
///   <https://github.com/jcmgray/cotengra>
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkIR {
    tensors: Vec<TensorNode>,
    edges: Vec<HyperEdge>,
    output_labels: Vec<TemporaryLabel>,
}

impl NetworkIR {
    pub fn from_labels(
        input_labels: Vec<Vec<TemporaryLabel>>,
        output_labels: Vec<TemporaryLabel>,
    ) -> Result<Self> {
        if input_labels.is_empty() {
            return Err(ContractError::EmptyInput);
        }

        validate_unique_output_labels(&output_labels)?;

        let tensors = input_labels
            .into_iter()
            .enumerate()
            .map(|(index, labels)| TensorNode::new(TensorId::new(index), labels))
            .collect::<Vec<_>>();

        let mut occurrences = BTreeMap::<TemporaryLabel, Vec<TensorAxis>>::new();
        for tensor in &tensors {
            for (axis, label) in tensor.labels().iter().enumerate() {
                occurrences
                    .entry(label.clone())
                    .or_default()
                    .push(TensorAxis::new(tensor.id(), axis));
            }
        }

        for label in &output_labels {
            if !occurrences.contains_key(label) {
                return Err(ContractError::UnknownOutputLabel(label.to_string()));
            }
        }

        let output_set: BTreeSet<&TemporaryLabel> = output_labels.iter().collect();
        validate_einsum_support(&occurrences, &output_set)?;

        let output_positions = output_labels
            .iter()
            .enumerate()
            .map(|(position, label)| (label.clone(), position))
            .collect::<BTreeMap<_, _>>();

        let edges = occurrences
            .into_iter()
            .map(|(label, axes)| {
                let output_position = output_positions.get(&label).copied();
                HyperEdge::new(label, axes, output_position)
            })
            .collect();

        Ok(Self {
            tensors,
            edges,
            output_labels,
        })
    }

    pub fn tensors(&self) -> &[TensorNode] {
        &self.tensors
    }

    pub fn edges(&self) -> &[HyperEdge] {
        &self.edges
    }

    pub fn output_labels(&self) -> &[TemporaryLabel] {
        &self.output_labels
    }

    pub fn tensor(&self, id: TensorId) -> Result<&TensorNode> {
        self.tensors
            .get(id.index())
            .ok_or(ContractError::InvalidTensorId {
                tensor: id.index(),
                tensor_count: self.tensors.len(),
            })
    }

    pub fn edge(&self, label: &TemporaryLabel) -> Option<&HyperEdge> {
        self.edges.iter().find(|edge| edge.label() == label)
    }

    pub fn output_rank(&self) -> usize {
        self.output_labels.len()
    }

    pub fn labels_for_tensor(&self, id: TensorId) -> Result<&[TemporaryLabel]> {
        Ok(self.tensor(id)?.labels())
    }
}

fn validate_unique_output_labels(labels: &[TemporaryLabel]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for label in labels {
        if !seen.insert(label.clone()) {
            return Err(ContractError::DuplicateOutputLabel(label.to_string()));
        }
    }
    Ok(())
}

/// Reject einsum patterns the pairwise contraction executor does NOT support,
/// at parse/plan time, with a specific error (instead of a late panic or a
/// silently-wrong result deep in execution).
///
/// The executor lowers a network to a sequence of pairwise `contract`s plus a
/// final `permute`. That model supports exactly: each label kept in the output
/// carried by ONE input (relabel/transpose), and each contracted label shared by
/// EXACTLY TWO operands (one pairwise contraction). Everything else is rejected:
///   * a label repeated within one operand (diagonal / single-tensor trace),
///   * a label on >2 operands (a hyperedge),
///   * an output label shared by >1 input (batch / hadamard),
///   * a contracted label on a single operand (single-operand reduction/sum).
///
/// Standard pairwise einsums (`ij,jk->ik`, `ab,ba->`, `ij,ij->`, the `;`
/// codomain/domain forms, conjugated operands) all pass unchanged.
fn validate_einsum_support(
    occurrences: &BTreeMap<TemporaryLabel, Vec<TensorAxis>>,
    output_set: &BTreeSet<&TemporaryLabel>,
) -> Result<()> {
    for (label, axes) in occurrences {
        // Distinct operands this label appears on, and whether it repeats within
        // any single operand (a diagonal).
        let mut operands = BTreeSet::new();
        let mut seen_axes = BTreeSet::new();
        for axis in axes {
            let tensor = axis.tensor();
            operands.insert(tensor);
            if !seen_axes.insert(tensor) {
                // Second axis on the same operand ⇒ diagonal / trace-on-one-tensor.
                return Err(ContractError::UnsupportedDiagonal {
                    label: label.to_string(),
                    tensor: tensor.index(),
                });
            }
        }
        let operand_count = operands.len();
        let is_output = output_set.contains(label);

        if operand_count > 2 {
            return Err(ContractError::UnsupportedHyperedge {
                label: label.to_string(),
                operand_count,
            });
        }
        if is_output {
            // Output labels must be carried by exactly one input; >1 is a batch
            // index. (Hyperedge already caught >2 above; this catches == 2.)
            if operand_count > 1 {
                return Err(ContractError::UnsupportedBatchLabel {
                    label: label.to_string(),
                    operand_count,
                });
            }
        } else {
            // A contracted label must be shared by exactly two operands; a single
            // operand would need a reduction the executor does not perform.
            if operand_count < 2 {
                return Err(ContractError::UnsupportedReduction {
                    label: label.to_string(),
                });
            }
        }
    }
    Ok(())
}
