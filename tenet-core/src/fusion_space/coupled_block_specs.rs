fn degeneracy_shape_for_tree_side(
    space: &FusionProductSpace,
    tree: &FusionTreeKey,
) -> Result<DimVec, CoreError> {
    #[cfg(test)]
    observe_coupled_grid_side_derivation();
    if tree.uncoupled().len() != space.len() {
        return Err(CoreError::StructureRankMismatch {
            expected: space.len(),
            actual: tree.uncoupled().len(),
        });
    }
    space
        .legs()
        .iter()
        .zip(tree.uncoupled())
        .map(|(leg, &sector)| {
            leg.degeneracy(sector)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "fusion tree uses a sector absent from its leg",
                })
        })
        .collect()
}

/// Why its structures carry the storage-tiling proof: `visit_coupled_leg_blocks`
/// places block `(row, col)` of a coupled sector at the positional
/// `(row_offsets[row], col_offsets[col])` window of that sector's column-major
/// `matrix_rows × matrix_cols` matrix, with compact row axes and column
/// strides that are multiples of `matrix_rows`, for every `row < row_count`
/// and `col < col_count` once, and the sector matrices follow each other from
/// offset zero. The blocks are therefore pairwise disjoint and each reaches
/// its window once, independent of tree identity.
fn coupled_subblock_parts_from_leg_degeneracies(
    homspace: &FusionTreeHomSpace,
    layout: &FusionTreeHomSpaceLayoutData,
) -> Result<(SectorStructure, DegeneracyStructure), CoreError> {
    let rank = homspace.rank();
    let mut degeneracy_blocks = Vec::with_capacity(layout.keys.len());
    visit_coupled_leg_blocks(homspace, layout, |block| {
        degeneracy_blocks.push(block);
        Ok(())
    })?;

    let sector_structure =
        SectorStructure::from_keys(rank, layout.keys.iter().cloned().map(BlockKey::from))?;
    let degeneracy_structure =
        DegeneracyStructure::from_blocks_with_rank(rank, degeneracy_blocks)?;
    Ok((sector_structure, degeneracy_structure))
}

fn visit_coupled_leg_blocks<F>(
    homspace: &FusionTreeHomSpace,
    layout: &FusionTreeHomSpaceLayoutData,
    mut visit: F,
) -> Result<(), CoreError>
where
    F: FnMut(DegeneracyBlock) -> Result<(), CoreError>,
{
    let rank = homspace.rank();
    let nout = homspace.codomain().len();
    let mut sector_offset = 0usize;
    let mut row_shapes = SmallVec::<[DimVec; 8]>::new();
    let mut row_dims = DimVec::new();
    let mut col_shapes = SmallVec::<[DimVec; 8]>::new();
    let mut col_dims = DimVec::new();
    let mut row_offsets = DimVec::new();
    let mut col_offsets = DimVec::new();

    for sector in &layout.sectors {
        let block_count = sector
            .row_count
            .checked_mul(sector.col_count)
            .ok_or(CoreError::ElementCountOverflow)?;
        let run_end = sector
            .start
            .checked_add(block_count)
            .ok_or(CoreError::ElementCountOverflow)?;
        if run_end > layout.keys.len() {
            return Err(CoreError::BlockCountMismatch {
                expected: run_end,
                actual: layout.keys.len(),
            });
        }

        row_shapes.clear();
        row_dims.clear();
        for row in 0..sector.row_count {
            let key = &layout.keys[sector.start + row];
            let shape = degeneracy_shape_for_tree_side(homspace.codomain(), key.codomain_tree())?;
            let dim = shape.iter().try_fold(1usize, |product, &axis| {
                product
                    .checked_mul(axis)
                    .ok_or(CoreError::ElementCountOverflow)
            })?;
            row_shapes.push(shape);
            row_dims.push(dim);
        }

        col_shapes.clear();
        col_dims.clear();
        for col in 0..sector.col_count {
            let local_offset = col
                .checked_mul(sector.row_count)
                .ok_or(CoreError::ElementCountOverflow)?;
            let key = &layout.keys[sector.start + local_offset];
            let shape = degeneracy_shape_for_tree_side(homspace.domain(), key.domain_tree())?;
            let dim = shape.iter().try_fold(1usize, |product, &axis| {
                product
                    .checked_mul(axis)
                    .ok_or(CoreError::ElementCountOverflow)
            })?;
            col_shapes.push(shape);
            col_dims.push(dim);
        }

        prefix_offsets_into(&row_dims, &mut row_offsets)?;
        prefix_offsets_into(&col_dims, &mut col_offsets)?;
        let matrix_rows = match row_offsets.last().zip(row_dims.last()) {
            Some((&offset, &dim)) => offset
                .checked_add(dim)
                .ok_or(CoreError::ElementCountOverflow)?,
            None => 0,
        };
        let matrix_cols = match col_offsets.last().zip(col_dims.last()) {
            Some((&offset, &dim)) => offset
                .checked_add(dim)
                .ok_or(CoreError::ElementCountOverflow)?,
            None => 0,
        };

        for col in 0..sector.col_count {
            for row in 0..sector.row_count {
                let mut shape = DimVec::with_capacity(rank);
                shape.extend_from_slice(&row_shapes[row]);
                shape.extend_from_slice(&col_shapes[col]);

                let mut strides = DimVec::new();
                let mut stride = 1usize;
                for &dim in &shape[..nout] {
                    strides.push(stride);
                    stride = stride
                        .checked_mul(dim)
                        .ok_or(CoreError::ElementCountOverflow)?;
                }
                let mut stride = matrix_rows;
                for &dim in &shape[nout..] {
                    strides.push(stride);
                    stride = stride
                        .checked_mul(dim)
                        .ok_or(CoreError::ElementCountOverflow)?;
                }
                let offset = sector_offset
                    .checked_add(row_offsets[row])
                    .and_then(|offset| {
                        matrix_rows
                            .checked_mul(col_offsets[col])
                            .and_then(|column| offset.checked_add(column))
                    })
                    .ok_or(CoreError::ElementCountOverflow)?;
                visit(DegeneracyBlock::new(shape, strides, offset)?)?;
            }
        }

        sector_offset = sector_offset
            .checked_add(
                matrix_rows
                    .checked_mul(matrix_cols)
                    .ok_or(CoreError::ElementCountOverflow)?,
            )
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    Ok(())
}

/// Computes coupled-sector matrix block specs for fusion-tree subblocks.
///
/// Keys must arrive stable-sorted by coupled sector. Both owning callers
/// establish that order before this private helper. Within one coupled sector
/// every codomain tree defines a
/// row block and every domain tree a column block of one column-major sector
/// matrix; the subblock for `(codomain tree, domain tree)` is the strided view
/// at that (row block, column block) position. Full coverage of the
/// `rows × columns` grid is required so the sector matrix has no
/// uninitialized holes.
fn coupled_sector_matrix_block_specs<K, S>(
    nout: usize,
    rank: usize,
    keys: &[K],
    shapes: &[S],
) -> Result<Vec<BlockSpec>, CoreError>
where
    K: std::borrow::Borrow<FusionTreePairKey>,
    S: AsRef<[usize]>,
{
    validate_coupled_sector_matrix_dimensions(nout, rank, shapes.iter())?;
    coupled_sector_matrix_block_specs_after_dimension_validation(nout, rank, keys, shapes)
}

fn validate_coupled_sector_matrix_dimensions<'shape, S>(
    nout: usize,
    rank: usize,
    shapes: impl IntoIterator<Item = &'shape S>,
) -> Result<(), CoreError>
where
    S: AsRef<[usize]> + 'shape,
{
    if nout > rank {
        return Err(CoreError::StructureRankMismatch {
            expected: rank,
            actual: nout,
        });
    }
    for shape in shapes {
        let shape = shape.as_ref();
        if shape.len() != rank {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
    }
    Ok(())
}

fn coupled_sector_matrix_block_specs_after_dimension_validation<K, S>(
    nout: usize,
    rank: usize,
    keys: &[K],
    shapes: &[S],
) -> Result<Vec<BlockSpec>, CoreError>
where
    K: std::borrow::Borrow<FusionTreePairKey>,
    S: AsRef<[usize]>,
{
    let mut specs = Vec::with_capacity(keys.len());
    let mut sector_offset = 0usize;
    let mut run_start = 0usize;
    while run_start < keys.len() {
        let coupled = keys[run_start].borrow().codomain_tree().coupled();
        let mut run_end = run_start;
        while run_end < keys.len()
            && keys[run_end].borrow().codomain_tree().coupled() == coupled
        {
            if keys[run_end].borrow().domain_tree().coupled() != coupled {
                return Err(CoreError::MalformedFusionTree {
                    message: "codomain and domain trees must share the coupled sector",
                });
            }
            run_end += 1;
        }

        // Row/column blocks keep first-seen order (offsets are cumulative), with
        // a hash side-index for O(1) tree lookup instead of a linear scan: a run
        // can hold many blocks, so the scan was O(run^1.5).
        let mut row_blocks: Vec<(&FusionTreeKey, usize, usize)> = Vec::new();
        let mut col_blocks: Vec<(&FusionTreeKey, usize, usize)> = Vec::new();
        let mut row_index: FxHashMap<&FusionTreeKey, usize> = FxHashMap::default();
        let mut col_index: FxHashMap<&FusionTreeKey, usize> = FxHashMap::default();
        for index in run_start..run_end {
            let key = keys[index].borrow();
            let shape = shapes[index].as_ref();
            let row_dim = checked_product(&shape[..nout])?;
            let col_dim = checked_product(&shape[nout..])?;
            match row_index.get(key.codomain_tree()).copied() {
                Some(existing_index) if row_blocks[existing_index].2 != row_dim => {
                    return Err(CoreError::DimensionMismatch {
                        expected: row_blocks[existing_index].2,
                        actual: row_dim,
                    });
                }
                Some(_) => {}
                None => {
                    let offset = match row_blocks.last() {
                        Some((_, start, dim)) => start
                            .checked_add(*dim)
                            .ok_or(CoreError::ElementCountOverflow)?,
                        None => 0,
                    };
                    row_index.insert(key.codomain_tree(), row_blocks.len());
                    row_blocks.push((key.codomain_tree(), offset, row_dim));
                }
            }
            match col_index.get(key.domain_tree()).copied() {
                Some(existing_index) if col_blocks[existing_index].2 != col_dim => {
                    return Err(CoreError::DimensionMismatch {
                        expected: col_blocks[existing_index].2,
                        actual: col_dim,
                    });
                }
                Some(_) => {}
                None => {
                    let offset = match col_blocks.last() {
                        Some((_, start, dim)) => start
                            .checked_add(*dim)
                            .ok_or(CoreError::ElementCountOverflow)?,
                        None => 0,
                    };
                    col_index.insert(key.domain_tree(), col_blocks.len());
                    col_blocks.push((key.domain_tree(), offset, col_dim));
                }
            }
        }
        let expected_blocks = row_blocks
            .len()
            .checked_mul(col_blocks.len())
            .ok_or(CoreError::ElementCountOverflow)?;
        if run_end - run_start != expected_blocks {
            return Err(CoreError::BlockCountMismatch {
                expected: expected_blocks,
                actual: run_end - run_start,
            });
        }
        let matrix_rows = match row_blocks.last() {
            Some((_, start, dim)) => start
                .checked_add(*dim)
                .ok_or(CoreError::ElementCountOverflow)?,
            None => 0,
        };
        let matrix_cols = match col_blocks.last() {
            Some((_, start, dim)) => start
                .checked_add(*dim)
                .ok_or(CoreError::ElementCountOverflow)?,
            None => 0,
        };

        for index in run_start..run_end {
            let key = keys[index].borrow();
            let shape = shapes[index].as_ref();
            let row_start = row_blocks[row_index
                .get(key.codomain_tree())
                .copied()
                .expect("row block registered above")]
            .1;
            let col_start = col_blocks[col_index
                .get(key.domain_tree())
                .copied()
                .expect("column block registered above")]
            .1;
            let mut strides = Vec::with_capacity(rank);
            let mut stride = 1usize;
            for &dim in &shape[..nout] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let mut stride = matrix_rows;
            for &dim in &shape[nout..] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let offset = matrix_rows
                .checked_mul(col_start)
                .and_then(|column| sector_offset.checked_add(row_start)?.checked_add(column))
                .ok_or(CoreError::ElementCountOverflow)?;
            specs.push(BlockSpec::with_key(
                BlockKey::FusionTree(key.clone()),
                shape.to_vec(),
                strides,
                offset,
            )?);
        }

        sector_offset = sector_offset
            .checked_add(
                matrix_rows
                    .checked_mul(matrix_cols)
                    .ok_or(CoreError::ElementCountOverflow)?,
            )
            .ok_or(CoreError::ElementCountOverflow)?;
        run_start = run_end;
    }
    Ok(specs)
}

fn coupled_sector_matrix_block_specs_from_layout<S>(
    nout: usize,
    rank: usize,
    layout: &FusionTreeHomSpaceLayout,
    shapes: &[S],
) -> Result<Vec<BlockSpec>, CoreError>
where
    S: AsRef<[usize]>,
{
    if nout > rank {
        return Err(CoreError::StructureRankMismatch {
            expected: rank,
            actual: nout,
        });
    }
    let keys = layout.keys.as_ref();
    if keys.len() != shapes.len() {
        return Err(CoreError::BlockCountMismatch {
            expected: keys.len(),
            actual: shapes.len(),
        });
    }
    for shape in shapes {
        let shape = shape.as_ref();
        if shape.len() != rank {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
    }

    let mut specs = Vec::with_capacity(keys.len());
    let mut sector_offset = 0usize;
    for sector in &layout.sectors {
        let block_count = sector
            .row_count
            .checked_mul(sector.col_count)
            .ok_or(CoreError::ElementCountOverflow)?;
        let run_end = sector
            .start
            .checked_add(block_count)
            .ok_or(CoreError::ElementCountOverflow)?;
        if run_end > keys.len() {
            return Err(CoreError::BlockCountMismatch {
                expected: run_end,
                actual: keys.len(),
            });
        }

        let mut row_dims = vec![None; sector.row_count];
        let mut col_dims = vec![None; sector.col_count];
        for (col, col_dim) in col_dims.iter_mut().enumerate() {
            for (row, row_dim) in row_dims.iter_mut().enumerate() {
                let local_index = col
                    .checked_mul(sector.row_count)
                    .and_then(|offset| offset.checked_add(row))
                    .ok_or(CoreError::ElementCountOverflow)?;
                let index = sector
                    .start
                    .checked_add(local_index)
                    .ok_or(CoreError::ElementCountOverflow)?;
                let shape = shapes[index].as_ref();
                register_layout_dim(row_dim, checked_product(&shape[..nout])?)?;
                register_layout_dim(col_dim, checked_product(&shape[nout..])?)?;
            }
        }

        let row_dims = row_dims
            .into_iter()
            .map(|dim| {
                dim.ok_or(CoreError::MalformedFusionTree {
                    message: "cached fusion tree layout has an empty row",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let col_dims = col_dims
            .into_iter()
            .map(|dim| {
                dim.ok_or(CoreError::MalformedFusionTree {
                    message: "cached fusion tree layout has an empty column",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let row_offsets = prefix_offsets(&row_dims)?;
        let col_offsets = prefix_offsets(&col_dims)?;
        let matrix_rows = match row_offsets.last().zip(row_dims.last()) {
            Some((&offset, &dim)) => offset
                .checked_add(dim)
                .ok_or(CoreError::ElementCountOverflow)?,
            None => 0,
        };
        let matrix_cols = match col_offsets.last().zip(col_dims.last()) {
            Some((&offset, &dim)) => offset
                .checked_add(dim)
                .ok_or(CoreError::ElementCountOverflow)?,
            None => 0,
        };

        for (col, &col_offset) in col_offsets.iter().enumerate() {
            for (row, &row_offset) in row_offsets.iter().enumerate() {
                let local_index = col
                    .checked_mul(sector.row_count)
                    .and_then(|offset| offset.checked_add(row))
                    .ok_or(CoreError::ElementCountOverflow)?;
                let index = sector
                    .start
                    .checked_add(local_index)
                    .ok_or(CoreError::ElementCountOverflow)?;
                let shape = shapes[index].as_ref();
                let mut strides = Vec::with_capacity(rank);
                let mut stride = 1usize;
                for &dim in &shape[..nout] {
                    strides.push(stride);
                    stride = stride
                        .checked_mul(dim)
                        .ok_or(CoreError::ElementCountOverflow)?;
                }
                let mut stride = matrix_rows;
                for &dim in &shape[nout..] {
                    strides.push(stride);
                    stride = stride
                        .checked_mul(dim)
                        .ok_or(CoreError::ElementCountOverflow)?;
                }
                let offset = sector_offset
                    .checked_add(row_offset)
                    .and_then(|offset| {
                        matrix_rows
                            .checked_mul(col_offset)
                            .and_then(|column| offset.checked_add(column))
                    })
                    .ok_or(CoreError::ElementCountOverflow)?;
                specs.push(BlockSpec::with_key(
                    BlockKey::FusionTree(keys[index].clone()),
                    shape.to_vec(),
                    strides,
                    offset,
                )?);
            }
        }

        sector_offset = sector_offset
            .checked_add(
                matrix_rows
                    .checked_mul(matrix_cols)
                    .ok_or(CoreError::ElementCountOverflow)?,
            )
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    Ok(specs)
}

fn register_layout_dim(slot: &mut Option<usize>, dim: usize) -> Result<(), CoreError> {
    match slot {
        Some(existing) if *existing != dim => Err(CoreError::DimensionMismatch {
            expected: *existing,
            actual: dim,
        }),
        Some(_) => Ok(()),
        None => {
            *slot = Some(dim);
            Ok(())
        }
    }
}

fn prefix_offsets(dims: &[usize]) -> Result<DimVec, CoreError> {
    let mut offsets = DimVec::new();
    prefix_offsets_into(dims, &mut offsets)?;
    Ok(offsets)
}

fn prefix_offsets_into(dims: &[usize], offsets: &mut DimVec) -> Result<(), CoreError> {
    offsets.clear();
    offsets.reserve(dims.len());
    let mut offset = 0usize;
    for &dim in dims {
        offsets.push(offset);
        offset = offset
            .checked_add(dim)
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    Ok(())
}
