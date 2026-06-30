use tenet_core::{BlockKey, BlockStructure, CoreError, FusionTreeBlockGroup};

use crate::OperationError;

pub(super) fn fusion_tree_group_block_keys(
    structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    tensor: &'static str,
) -> Result<Vec<BlockKey>, OperationError> {
    let mut keys = Vec::with_capacity(group.block_indices().len());
    for &index in group.block_indices() {
        let block = structure.block(index).map_err(|err| match err {
            CoreError::BlockIndexOutOfBounds { index, count } => {
                OperationError::BlockIndexOutOfBounds {
                    tensor,
                    index,
                    count,
                }
            }
            other => OperationError::Core(other),
        })?;
        match block.key().fusion_tree_group_key() {
            Some(actual) if &actual == group.group_key() => keys.push(block.key().clone()),
            _ => return Err(OperationError::FusionTreeGroupMismatch { tensor, index }),
        }
    }
    Ok(keys)
}

pub(super) fn block_indices_for_keys(
    structure: &BlockStructure,
    keys: &[BlockKey],
) -> Result<Vec<usize>, OperationError> {
    keys.iter()
        .map(|key| {
            structure
                .find_block_index_by_key(key)
                .ok_or_else(|| OperationError::MissingBlockKey { key: key.clone() })
        })
        .collect()
}
