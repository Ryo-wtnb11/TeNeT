use std::collections::BTreeMap;

use crate::error::{ContractError, Result};
use crate::ir::{HyperEdge, NetworkIR};
use crate::labels::{TemporaryLabel, TensorId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseTensorInfo {
    shape: Vec<usize>,
}

impl DenseTensorInfo {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseCostModel {
    label_dims: BTreeMap<TemporaryLabel, usize>,
}

impl DenseCostModel {
    pub fn from_network(ir: &NetworkIR, tensors: &[DenseTensorInfo]) -> Result<Self> {
        if tensors.len() != ir.tensors().len() {
            return Err(ContractError::TensorCountMismatch {
                expected: ir.tensors().len(),
                actual: tensors.len(),
            });
        }

        let mut label_dims = BTreeMap::<TemporaryLabel, usize>::new();
        for tensor in ir.tensors() {
            let tensor_info = &tensors[tensor.id().index()];
            if tensor_info.shape().len() != tensor.rank() {
                return Err(ContractError::RankMismatch {
                    tensor: tensor.id().index(),
                    expected: tensor.rank(),
                    actual: tensor_info.shape().len(),
                });
            }
            for (axis, label) in tensor.labels().iter().enumerate() {
                let dim = tensor_info.shape()[axis];
                match label_dims.get(label) {
                    Some(&expected) if expected != dim => {
                        return Err(ContractError::DimensionMismatch {
                            label: label.to_string(),
                            expected,
                            actual: dim,
                        });
                    }
                    Some(_) => {}
                    None => {
                        label_dims.insert(label.clone(), dim);
                    }
                }
            }
        }

        Ok(Self { label_dims })
    }

    pub fn dim(&self, label: &TemporaryLabel) -> Option<usize> {
        self.label_dims.get(label).copied()
    }

    /// A copy of this cost model with the given labels' dimensions forced to 1.
    ///
    /// Used by dynamic slicing to reflect already-sliced indices: a sliced index
    /// contributes a factor of 1 to intermediate sizes and contraction costs.
    pub fn with_unit_dims(&self, labels: &[TemporaryLabel]) -> DenseCostModel {
        let mut label_dims = self.label_dims.clone();
        for label in labels {
            if let Some(dim) = label_dims.get_mut(label) {
                *dim = 1;
            }
        }
        DenseCostModel { label_dims }
    }

    pub fn edge_dim(&self, edge: &HyperEdge) -> usize {
        self.dim(edge.label()).unwrap_or(1)
    }

    /// Number of scalar elements in a tensor with these legs (the product of the
    /// leg dimensions). Used as the dense FLOP/size proxy by `pair_cost`.
    ///
    /// The product is accumulated with [`usize::saturating_mul`] so a large
    /// network whose true size exceeds `usize::MAX` *saturates* to `usize::MAX`
    /// rather than silently wrapping to a small value — a wrapped tiny cost would
    /// otherwise make the greedy optimizer treat a huge contraction as cheap.
    pub fn tensor_size(&self, labels: &[TemporaryLabel]) -> usize {
        labels.iter().fold(1usize, |acc, label| {
            acc.saturating_mul(self.dim(label).unwrap_or(1))
        })
    }

    pub fn pair_cost(&self, lhs: &[TemporaryLabel], rhs: &[TemporaryLabel]) -> usize {
        let mut labels = lhs.to_vec();
        for label in rhs {
            if !labels.contains(label) {
                labels.push(label.clone());
            }
        }
        self.tensor_size(&labels)
    }

    pub fn contraction_result_labels(
        &self,
        lhs: &[TemporaryLabel],
        rhs: &[TemporaryLabel],
        output_labels: &[TemporaryLabel],
    ) -> Vec<TemporaryLabel> {
        self.contraction_result_labels_with_remaining(lhs, rhs, &[], output_labels)
    }

    pub fn contraction_result_labels_with_remaining(
        &self,
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

    pub fn tensor_labels<'a>(
        &self,
        ir: &'a NetworkIR,
        tensor_id: TensorId,
    ) -> Result<&'a [TemporaryLabel]> {
        ir.labels_for_tensor(tensor_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockLabelInfo<S: Ord + Clone> {
    sector: S,
    dim: usize,
}

impl<S: Ord + Clone> BlockLabelInfo<S> {
    pub fn new(sector: S, dim: usize) -> Self {
        Self { sector, dim }
    }

    pub fn sector(&self) -> &S {
        &self.sector
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockInfo<S: Ord + Clone> {
    labels: BTreeMap<TemporaryLabel, BlockLabelInfo<S>>,
}

impl<S: Ord + Clone> BlockInfo<S> {
    pub fn new(axes: Vec<(TemporaryLabel, S, usize)>) -> Result<Self> {
        let mut labels = BTreeMap::new();
        for (label, sector, dim) in axes {
            if labels
                .insert(label.clone(), BlockLabelInfo::new(sector, dim))
                .is_some()
            {
                return Err(ContractError::InvalidBlockStructure(format!(
                    "duplicate label `{label}` in one block"
                )));
            }
        }
        Ok(Self { labels })
    }

    pub fn labels(&self) -> &BTreeMap<TemporaryLabel, BlockLabelInfo<S>> {
        &self.labels
    }

    pub fn label_info(&self, label: &TemporaryLabel) -> Option<&BlockLabelInfo<S>> {
        self.labels.get(label)
    }

    pub fn label_names(&self) -> impl Iterator<Item = &TemporaryLabel> {
        self.labels.keys()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSparseTensorInfo<S: Ord + Clone> {
    blocks: Vec<BlockInfo<S>>,
}

impl<S: Ord + Clone> BlockSparseTensorInfo<S> {
    pub fn new(blocks: Vec<BlockInfo<S>>) -> Self {
        Self { blocks }
    }

    pub fn blocks(&self) -> &[BlockInfo<S>] {
        &self.blocks
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSparseCostModel<S: Ord + Clone> {
    tensors: Vec<BlockSparseTensorInfo<S>>,
}

impl<S: Ord + Clone> BlockSparseCostModel<S> {
    pub fn from_network(ir: &NetworkIR, tensors: &[BlockSparseTensorInfo<S>]) -> Result<Self> {
        if tensors.len() != ir.tensors().len() {
            return Err(ContractError::TensorCountMismatch {
                expected: ir.tensors().len(),
                actual: tensors.len(),
            });
        }

        for tensor in ir.tensors() {
            validate_block_sparse_tensor(
                tensor.id().index(),
                tensor.labels(),
                &tensors[tensor.id().index()],
            )?;
        }

        Ok(Self {
            tensors: tensors.to_vec(),
        })
    }

    pub fn tensor_info(&self, tensor_id: TensorId) -> Result<&BlockSparseTensorInfo<S>> {
        self.tensors
            .get(tensor_id.index())
            .ok_or(ContractError::InvalidTensorId {
                tensor: tensor_id.index(),
                tensor_count: self.tensors.len(),
            })
    }

    pub fn pair_cost(
        &self,
        lhs: &BlockSparseTensorInfo<S>,
        rhs: &BlockSparseTensorInfo<S>,
    ) -> Result<usize> {
        let mut cost = 0usize;
        for lhs_block in lhs.blocks() {
            for rhs_block in rhs.blocks() {
                if !blocks_are_compatible(lhs_block, rhs_block)? {
                    continue;
                }
                // Saturate so a huge block-sparse network's cost estimate clamps
                // at `usize::MAX` instead of wrapping to a misleadingly small one.
                cost = cost.saturating_add(block_pair_work(lhs_block, rhs_block)?);
            }
        }
        Ok(cost)
    }

    pub fn contraction_result_info(
        &self,
        lhs: &BlockSparseTensorInfo<S>,
        rhs: &BlockSparseTensorInfo<S>,
        remaining_labels: &[Vec<TemporaryLabel>],
        output_labels: &[TemporaryLabel],
    ) -> Result<BlockSparseTensorInfo<S>> {
        let mut blocks = std::collections::BTreeSet::new();
        for lhs_block in lhs.blocks() {
            for rhs_block in rhs.blocks() {
                if !blocks_are_compatible(lhs_block, rhs_block)? {
                    continue;
                }
                blocks.insert(block_pair_result(
                    lhs_block,
                    rhs_block,
                    remaining_labels,
                    output_labels,
                )?);
            }
        }
        Ok(BlockSparseTensorInfo::new(blocks.into_iter().collect()))
    }
}

fn validate_block_sparse_tensor<S: Ord + Clone>(
    tensor: usize,
    expected_labels: &[TemporaryLabel],
    info: &BlockSparseTensorInfo<S>,
) -> Result<()> {
    for (block_index, block) in info.blocks().iter().enumerate() {
        if block.labels().len() != expected_labels.len() {
            return Err(ContractError::InvalidBlockStructure(format!(
                "tensor {tensor} block {block_index} has rank {}, expected {}",
                block.labels().len(),
                expected_labels.len()
            )));
        }
        for label in expected_labels {
            if !block.labels().contains_key(label) {
                return Err(ContractError::InvalidBlockStructure(format!(
                    "tensor {tensor} block {block_index} is missing label `{label}`"
                )));
            }
        }
        for label in block.label_names() {
            if !expected_labels.contains(label) {
                return Err(ContractError::InvalidBlockStructure(format!(
                    "tensor {tensor} block {block_index} has unexpected label `{label}`"
                )));
            }
        }
    }
    Ok(())
}

fn blocks_are_compatible<S: Ord + Clone>(lhs: &BlockInfo<S>, rhs: &BlockInfo<S>) -> Result<bool> {
    for (label, lhs_info) in lhs.labels() {
        let Some(rhs_info) = rhs.label_info(label) else {
            continue;
        };
        if lhs_info.sector() != rhs_info.sector() {
            return Ok(false);
        }
        if lhs_info.dim() != rhs_info.dim() {
            return Err(ContractError::DimensionMismatch {
                label: label.to_string(),
                expected: lhs_info.dim(),
                actual: rhs_info.dim(),
            });
        }
    }
    Ok(true)
}

fn block_pair_work<S: Ord + Clone>(lhs: &BlockInfo<S>, rhs: &BlockInfo<S>) -> Result<usize> {
    let mut dims = BTreeMap::<TemporaryLabel, usize>::new();
    for (label, info) in lhs.labels() {
        dims.insert(label.clone(), info.dim());
    }
    for (label, info) in rhs.labels() {
        match dims.get(label) {
            Some(&expected) if expected != info.dim() => {
                return Err(ContractError::DimensionMismatch {
                    label: label.to_string(),
                    expected,
                    actual: info.dim(),
                });
            }
            Some(_) => {}
            None => {
                dims.insert(label.clone(), info.dim());
            }
        }
    }
    // Saturate the block work product (mirrors `DenseCostModel::tensor_size`).
    Ok(dims
        .values()
        .fold(1usize, |acc, &dim| acc.saturating_mul(dim)))
}

fn block_pair_result<S: Ord + Clone>(
    lhs: &BlockInfo<S>,
    rhs: &BlockInfo<S>,
    remaining_labels: &[Vec<TemporaryLabel>],
    output_labels: &[TemporaryLabel],
) -> Result<BlockInfo<S>> {
    let mut labels = BTreeMap::<TemporaryLabel, BlockLabelInfo<S>>::new();
    for (label, info) in lhs.labels() {
        if should_keep_result_label(label, lhs, rhs, remaining_labels, output_labels) {
            labels.insert(label.clone(), info.clone());
        }
    }
    for (label, info) in rhs.labels() {
        if !should_keep_result_label(label, lhs, rhs, remaining_labels, output_labels) {
            continue;
        }
        match labels.get(label) {
            Some(expected) if expected != info => {
                return Err(ContractError::DimensionMismatch {
                    label: label.to_string(),
                    expected: expected.dim(),
                    actual: info.dim(),
                });
            }
            Some(_) => {}
            None => {
                labels.insert(label.clone(), info.clone());
            }
        }
    }
    Ok(BlockInfo { labels })
}

fn should_keep_result_label<S: Ord + Clone>(
    label: &TemporaryLabel,
    lhs: &BlockInfo<S>,
    rhs: &BlockInfo<S>,
    remaining_labels: &[Vec<TemporaryLabel>],
    output_labels: &[TemporaryLabel],
) -> bool {
    if output_labels.contains(label) {
        return true;
    }
    let shared_by_pair = lhs.labels().contains_key(label) && rhs.labels().contains_key(label);
    if !shared_by_pair {
        return true;
    }
    remaining_labels.iter().any(|labels| labels.contains(label))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_einsum;

    fn label(s: &str) -> TemporaryLabel {
        TemporaryLabel::new(s)
    }

    /// `tensor_size` (and thus `pair_cost`) must SATURATE on overflow instead of
    /// wrapping. With five legs of dimension 2^14 the true element count is
    /// 2^70, far beyond `usize::MAX` (2^64-1 on this target): the old `.product()`
    /// would wrap to a tiny value and mislead the optimizer; the saturating
    /// fold must clamp at `usize::MAX`.
    #[test]
    fn dense_tensor_size_saturates_on_overflow() {
        let big = 1usize << 14; // 16384
                                // Five distinct legs, each dim 2^14 -> product 2^70 > usize::MAX.
        let ir = parse_einsum("abcde->abcde").unwrap();
        let infos = vec![DenseTensorInfo::new(vec![big, big, big, big, big])];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();

        let legs = [label("a"), label("b"), label("c"), label("d"), label("e")];
        assert_eq!(
            cost.tensor_size(&legs),
            usize::MAX,
            "overflowing tensor size must saturate, not wrap"
        );

        // A non-overflowing product is still exact (no behavior change).
        assert_eq!(cost.tensor_size(&[label("a"), label("b")]), big * big);
    }

    /// `pair_cost` builds on `tensor_size`, so it must saturate too rather than
    /// wrap to a small (and misleading) number.
    #[test]
    fn dense_pair_cost_saturates_on_overflow() {
        let big = 1usize << 14;
        let ir = parse_einsum("abc,cde->abde").unwrap();
        let infos = vec![
            DenseTensorInfo::new(vec![big, big, big]),
            DenseTensorInfo::new(vec![big, big, big]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        // Union of {a,b,c} and {c,d,e} = {a,b,c,d,e}, five legs of 2^14 -> 2^70.
        let lhs = [label("a"), label("b"), label("c")];
        let rhs = [label("c"), label("d"), label("e")];
        assert_eq!(cost.pair_cost(&lhs, &rhs), usize::MAX);
    }
}
