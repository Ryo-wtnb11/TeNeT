use std::collections::BTreeMap;

use crate::error::{ContractError, Result};
use crate::labels::{TemporaryLabel, TensorId};
use crate::optimizer::ContractionStep;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractionTree {
    Leaf {
        tensor: TensorId,
    },
    Pair {
        lhs: Box<ContractionTree>,
        rhs: Box<ContractionTree>,
        result: TensorId,
        cost: usize,
        result_labels: Vec<TemporaryLabel>,
    },
}

impl ContractionTree {
    pub fn leaf(tensor: TensorId) -> Self {
        Self::Leaf { tensor }
    }

    pub fn from_steps(tensor_count: usize, steps: &[ContractionStep]) -> Result<Self> {
        if tensor_count == 0 {
            return Err(ContractError::NotEnoughTensors);
        }

        let mut active = (0..tensor_count)
            .map(|index| {
                let id = TensorId::new(index);
                (id, ContractionTree::leaf(id))
            })
            .collect::<BTreeMap<_, _>>();

        for step in steps {
            let lhs = active.remove(&step.lhs()).ok_or_else(|| {
                ContractError::InvalidContractionPlan(format!(
                    "missing lhs tensor {}",
                    step.lhs().index()
                ))
            })?;
            let rhs = active.remove(&step.rhs()).ok_or_else(|| {
                ContractError::InvalidContractionPlan(format!(
                    "missing rhs tensor {}",
                    step.rhs().index()
                ))
            })?;
            if active.contains_key(&step.result()) {
                return Err(ContractError::InvalidContractionPlan(format!(
                    "result tensor {} already exists",
                    step.result().index()
                )));
            }
            active.insert(
                step.result(),
                ContractionTree::Pair {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    result: step.result(),
                    cost: step.cost(),
                    result_labels: step.result_labels().to_vec(),
                },
            );
        }

        if active.len() != 1 {
            return Err(ContractError::InvalidContractionPlan(format!(
                "plan leaves {} active tensors",
                active.len()
            )));
        }
        Ok(active.into_values().next().unwrap())
    }

    pub fn result(&self) -> TensorId {
        match self {
            ContractionTree::Leaf { tensor } => *tensor,
            ContractionTree::Pair { result, .. } => *result,
        }
    }

    pub fn cost(&self) -> usize {
        match self {
            ContractionTree::Leaf { .. } => 0,
            ContractionTree::Pair { cost, .. } => *cost,
        }
    }

    pub fn total_cost(&self) -> usize {
        match self {
            ContractionTree::Leaf { .. } => 0,
            ContractionTree::Pair { lhs, rhs, cost, .. } => {
                lhs.total_cost() + rhs.total_cost() + cost
            }
        }
    }

    pub fn result_labels(&self) -> &[TemporaryLabel] {
        match self {
            ContractionTree::Leaf { .. } => &[],
            ContractionTree::Pair { result_labels, .. } => result_labels,
        }
    }

    /// True for an input leaf (no contraction beneath it).
    pub fn is_leaf(&self) -> bool {
        matches!(self, ContractionTree::Leaf { .. })
    }

    /// The two child subtrees of a contraction node, or `None` for a leaf.
    pub fn children(&self) -> Option<(&ContractionTree, &ContractionTree)> {
        match self {
            ContractionTree::Leaf { .. } => None,
            ContractionTree::Pair { lhs, rhs, .. } => Some((lhs, rhs)),
        }
    }

    /// Input tensor ids (leaves) under this subtree, left-to-right.
    ///
    /// These are original-input leaves used by subtree-reconfiguration to
    /// re-optimize the contraction of a local region.
    pub fn leaves(&self) -> Vec<TensorId> {
        let mut out = Vec::new();
        self.collect_leaves(&mut out);
        out
    }

    fn collect_leaves(&self, out: &mut Vec<TensorId>) {
        match self {
            ContractionTree::Leaf { tensor } => out.push(*tensor),
            ContractionTree::Pair { lhs, rhs, .. } => {
                lhs.collect_leaves(out);
                rhs.collect_leaves(out);
            }
        }
    }

    /// Number of input leaves under this subtree.
    pub fn leaf_count(&self) -> usize {
        match self {
            ContractionTree::Leaf { .. } => 1,
            ContractionTree::Pair { lhs, rhs, .. } => lhs.leaf_count() + rhs.leaf_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(lhs: usize, rhs: usize, result: usize, labels: &[&str]) -> ContractionStep {
        ContractionStep::new(
            TensorId::new(lhs),
            TensorId::new(rhs),
            TensorId::new(result),
            0,
            labels.iter().map(|l| TemporaryLabel::new(*l)).collect(),
        )
    }

    #[test]
    fn children_and_leaves_traverse_the_tree() {
        // Chain of 3 inputs: (0,1)->3, (3,2)->4.
        let steps = vec![step(0, 1, 3, &["a", "c"]), step(3, 2, 4, &["a", "d"])];
        let tree = ContractionTree::from_steps(3, &steps).unwrap();

        assert!(!tree.is_leaf());
        assert_eq!(tree.leaf_count(), 3);
        let mut leaves = tree.leaves();
        leaves.sort_by_key(|t| t.index());
        assert_eq!(
            leaves,
            vec![TensorId::new(0), TensorId::new(1), TensorId::new(2)]
        );

        let (lhs, rhs) = tree.children().unwrap();
        // One child is the (0,1) subtree (2 leaves), the other a leaf (tensor 2).
        let counts = [lhs.leaf_count(), rhs.leaf_count()];
        assert!(counts.contains(&1) && counts.contains(&2));
        // The leaf child has no children.
        let leaf_child = if lhs.is_leaf() { lhs } else { rhs };
        assert!(leaf_child.children().is_none());
    }
}
