use rustc_hash::FxHashSet;
use tenet_core::{
    BlockKey, BlockStructure, CoreError, FusionTreeBlockGroup, FusionTreeGroupKey,
    FusionTreePairKey,
};

use crate::OperationError;

pub fn fusion_tree_pair_matches_group(key: &FusionTreePairKey, group: &FusionTreeGroupKey) -> bool {
    key.codomain_uncoupled() == group.codomain_uncoupled()
        && key.domain_uncoupled() == group.domain_uncoupled()
        && key.codomain_is_dual() == group.codomain_is_dual()
        && key.domain_is_dual() == group.domain_is_dual()
}

pub fn fusion_tree_pairs_share_group(lhs: &FusionTreePairKey, rhs: &FusionTreePairKey) -> bool {
    lhs.codomain_uncoupled() == rhs.codomain_uncoupled()
        && lhs.domain_uncoupled() == rhs.domain_uncoupled()
        && lhs.codomain_is_dual() == rhs.codomain_is_dual()
        && lhs.domain_is_dual() == rhs.domain_is_dual()
}

pub fn duplicate_fusion_tree_pair_index(keys: &[FusionTreePairKey]) -> Option<usize> {
    duplicate_fusion_tree_pair_indices(keys, &[]).0
}

pub fn duplicate_fusion_tree_pair_indices(
    first: &[FusionTreePairKey],
    second: &[FusionTreePairKey],
) -> (Option<usize>, Option<usize>) {
    let capacity = first.len().max(second.len());
    let mut seen = FxHashSet::with_capacity_and_hasher(capacity, Default::default());
    let first_duplicate = first.iter().position(|key| !seen.insert(key));
    seen.clear();
    let second_duplicate = second.iter().position(|key| !seen.insert(key));
    (first_duplicate, second_duplicate)
}

pub fn fusion_tree_group_block_keys(
    structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    tensor: &'static str,
) -> Result<Vec<FusionTreePairKey>, OperationError> {
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
        match block.key().as_fusion_tree_pair() {
            Some(key) if fusion_tree_pair_matches_group(key, group.group_key()) => {
                keys.push(key.clone());
            }
            _ => return Err(OperationError::FusionTreeGroupMismatch { tensor, index }),
        }
    }
    Ok(keys)
}

pub fn block_indices_for_keys(
    structure: &BlockStructure,
    keys: &[BlockKey],
) -> Result<Vec<usize>, OperationError> {
    keys.iter()
        .map(|key| {
            structure
                .find_block_index_by_key(key)
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: Box::new(key.clone()),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenet_core::BlockSpec;

    #[test]
    fn block_key_lookup_preserves_request_order_and_missing_identity() {
        // What: opaque block-key resolution returns indices in request order
        // and reports the exact first key absent from the structure.
        let first = BlockKey::opaque([2, 3]);
        let second = BlockKey::opaque([5, 7]);
        let missing = BlockKey::opaque([11, 13]);
        let structure = BlockStructure::from_blocks_with_rank(
            1,
            vec![
                BlockSpec::column_major_with_key(first.clone(), vec![1], 0).unwrap(),
                BlockSpec::column_major_with_key(second.clone(), vec![1], 1).unwrap(),
            ],
        )
        .unwrap();

        assert_eq!(
            block_indices_for_keys(&structure, &[second, first]).unwrap(),
            vec![1, 0]
        );
        assert_eq!(
            block_indices_for_keys(&structure, std::slice::from_ref(&missing)),
            Err(OperationError::MissingBlockKey {
                key: Box::new(missing),
            })
        );
    }
}
