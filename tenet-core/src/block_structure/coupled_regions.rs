fn new_coupled_region_cache(rank: usize) -> CoupledRegionCache {
    (0..=rank)
        .map(|_| OnceLock::new())
        .collect::<Vec<_>>()
        .into()
}

fn compile_coupled_sector_regions(
    structure: &BlockStructure,
    nout: usize,
) -> Result<Option<Vec<CoupledSectorRegion>>, CoreError> {
        let mut regions = Vec::new();
        let mut seen_coupled = FxHashMap::<SectorId, ()>::default();
        // Why hoist the per-sector scratch: the eager routes compile these
        // regions for every fresh destination wrapper, and per-sector maps
        // and vectors made the allocation count grow with the sector count.
        // Each region still owns one exact-length tree list.
        let mut row_trees = Vec::<CoupledTreeExtent>::new();
        let mut col_trees = Vec::<CoupledTreeExtent>::new();
        let mut row_indexes = FxHashMap::<&FusionTreeKey, usize>::default();
        let mut col_indexes = FxHashMap::<&FusionTreeKey, usize>::default();
        let mut tree_pairs = Vec::<(usize, usize)>::new();
        let mut seen_pairs = Vec::<bool>::new();
        let mut block_index = 0usize;
        let mut next_offset = 0usize;
        while block_index < structure.block_count() {
            let first = structure.block(block_index)?;
            let BlockKey::FusionTree(first_key) = first.key() else {
                return Ok(None);
            };
            let coupled = first_key.codomain_tree().coupled();
            if first_key.domain_tree().coupled() != coupled
                || seen_coupled.insert(coupled, ()).is_some()
            {
                return Ok(None);
            }

            row_trees.clear();
            col_trees.clear();
            row_indexes.clear();
            col_indexes.clear();
            tree_pairs.clear();
            let mut rows = 0usize;
            let mut cols = 0usize;
            let mut end = block_index;
            while end < structure.block_count() {
                let block = structure.block(end)?;
                let BlockKey::FusionTree(key) = block.key() else {
                    return Ok(None);
                };
                if key.codomain_tree().coupled() != coupled {
                    break;
                }
                if key.domain_tree().coupled() != coupled {
                    return Ok(None);
                }
                let row_shape: DimVec = block.shape()[..nout].iter().copied().collect();
                let col_shape: DimVec = block.shape()[nout..].iter().copied().collect();
                let Some(row_index) = insert_coupled_tree_extent(
                    &mut row_trees,
                    &mut row_indexes,
                    key.codomain_tree(),
                    row_shape,
                    &mut rows,
                )? else {
                    return Ok(None);
                };
                let Some(col_index) = insert_coupled_tree_extent(
                    &mut col_trees,
                    &mut col_indexes,
                    key.domain_tree(),
                    col_shape,
                    &mut cols,
                )? else {
                    return Ok(None);
                };
                tree_pairs.push((row_index, col_index));
                end += 1;
            }

            let expected_blocks = row_trees
                .len()
                .checked_mul(col_trees.len())
                .ok_or_else(|| CoreError::ElementCountOverflow)?;
            if end - block_index != expected_blocks {
                return Ok(None);
            }
            seen_pairs.clear();
            seen_pairs.resize(expected_blocks, false);
            for (index, (row_index, col_index)) in
                (block_index..end).zip(tree_pairs.iter().copied())
            {
                let block = structure.block(index)?;
                let pair_index = col_index * row_trees.len() + row_index;
                if std::mem::replace(&mut seen_pairs[pair_index], true) {
                    return Ok(None);
                }
                let row_offset = row_trees[row_index].offset();
                let col_offset = col_trees[col_index].offset();
                let expected_offset = next_offset
                    .checked_add(row_offset)
                    .and_then(|offset| {
                        rows
                            .checked_mul(col_offset)
                            .and_then(|column| offset.checked_add(column))
                    })
                    .ok_or_else(|| CoreError::ElementCountOverflow)?;
                if block.offset() != expected_offset
                    || !coupled_sector_strides(block.shape(), block.strides(), nout, rows)?
                {
                    return Ok(None);
                }
            }
            let elements = rows
                .checked_mul(cols)
                .ok_or_else(|| CoreError::ElementCountOverflow)?;
            let end_offset = next_offset
                .checked_add(elements)
                .ok_or_else(|| CoreError::ElementCountOverflow)?;
            if end_offset > structure.content.required_len {
                return Ok(None);
            }
            let aligned_diagonal = row_trees == col_trees
                && tree_pairs
                    .iter()
                    .filter_map(|&(row, col)| (row == col).then_some(row))
                    .eq(0..row_trees.len());
            regions.push(CoupledSectorRegion {
                coupled,
                rows,
                cols,
                range: next_offset..end_offset,
                row_tree_count: row_trees.len(),
                trees: row_trees.drain(..).chain(col_trees.drain(..)).collect(),
                aligned_diagonal,
            });
            next_offset = end_offset;
            block_index = end;
        }
        if next_offset != structure.content.required_len {
            return Ok(None);
        }
        Ok(Some(regions))
}

fn insert_coupled_tree_extent<'a>(
    trees: &mut Vec<CoupledTreeExtent>,
    indexes: &mut FxHashMap<&'a FusionTreeKey, usize>,
    tree: &'a FusionTreeKey,
    shape: DimVec,
    total: &mut usize,
) -> Result<Option<usize>, CoreError> {
    if let Some(&index) = indexes.get(tree) {
        return Ok((trees[index].shape() == shape.as_slice()).then_some(index));
    }
    let index = trees.len();
    let offset = *total;
    *total = offset
        .checked_add(checked_element_count(&shape)?)
        .ok_or_else(|| CoreError::ElementCountOverflow)?;
    indexes.insert(tree, index);
    trees.push(CoupledTreeExtent {
        tree: tree.clone(),
        offset,
        shape,
    });
    Ok(Some(index))
}

fn checked_element_count(shape: &[usize]) -> Result<usize, CoreError> {
    shape.iter().try_fold(1usize, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| CoreError::ElementCountOverflow)
    })
}

fn coupled_sector_strides(
    shape: &[usize],
    strides: &[usize],
    nout: usize,
    rows: usize,
) -> Result<bool, CoreError> {
    if shape.len() != strides.len() || nout > shape.len() {
        return Ok(false);
    }
    let mut expected = 1usize;
    for axis in 0..nout {
        if strides[axis] != expected {
            return Ok(false);
        }
        expected = expected
            .checked_mul(shape[axis])
            .ok_or_else(|| CoreError::ElementCountOverflow)?;
    }
    expected = rows;
    for axis in nout..shape.len() {
        if strides[axis] != expected {
            return Ok(false);
        }
        expected = expected
            .checked_mul(shape[axis])
            .ok_or_else(|| CoreError::ElementCountOverflow)?;
    }
    Ok(true)
}
