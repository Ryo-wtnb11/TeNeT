use super::*;

/// One coupled sector's column-major `rows x cols` matrix, placed at `base`
/// in storage. Row block `r` occupies rows `row_offsets[r]..` and column block
/// `c` columns `col_offsets[c]..`; every coupled-sector layout builder places
/// its blocks through this one rule.
struct CoupledSectorMatrix<'a> {
    base: usize,
    rows: usize,
    cols: usize,
    row_offsets: &'a [usize],
    col_offsets: &'a [usize],
}

impl<'a> CoupledSectorMatrix<'a> {
    /// `rows` (`cols`) is the side's extent, as the placement produced it.
    fn new(
        base: usize,
        row_offsets: &'a [usize],
        rows: usize,
        col_offsets: &'a [usize],
        cols: usize,
    ) -> Self {
        Self {
            base,
            rows,
            cols,
            row_offsets,
            col_offsets,
        }
    }

    /// Pushes the strides of block `(row, col)` with degeneracy `shape`
    /// (codomain axes `..nout`) and returns its storage offset.
    fn place(
        &self,
        row: usize,
        col: usize,
        shape: &[usize],
        nout: usize,
        strides: &mut impl Extend<usize>,
    ) -> Result<usize, CoreError> {
        let mut push = |stride| {
            strides.extend(Some(stride));
            Ok(())
        };
        try_for_each_column_major_stride(1, &shape[..nout], &mut push, || {
            CoreError::ElementCountOverflow
        })?;
        try_for_each_column_major_stride(self.rows, &shape[nout..], push, || {
            CoreError::ElementCountOverflow
        })?;
        self.base
            .checked_add(self.row_offsets[row])
            .and_then(|offset| {
                self.rows
                    .checked_mul(self.col_offsets[col])
                    .and_then(|column| offset.checked_add(column))
            })
            .ok_or(CoreError::ElementCountOverflow)
    }

    /// Storage offset of the next coupled sector.
    fn end(&self) -> Result<usize, CoreError> {
        self.rows
            .checked_mul(self.cols)
            .and_then(|len| self.base.checked_add(len))
            .ok_or(CoreError::ElementCountOverflow)
    }
}

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
/// `rows × cols` [`CoupledSectorMatrix`], with compact row axes and column
/// strides that are multiples of `rows`, for every `row < row_count`
/// and `col < col_count` once, and the sector matrices follow each other from
/// offset zero. The blocks are therefore pairwise disjoint and each reaches
/// its window once, independent of tree identity.
pub(crate) fn coupled_subblock_parts_from_leg_degeneracies(
    homspace: &FusionTreeHomSpace,
    layout: &FusionTreeHomSpaceLayoutData,
) -> Result<(Arc<SectorStructure>, DegeneracyStructure), CoreError> {
    let rank = homspace.rank();
    let mut degeneracy_blocks = Vec::with_capacity(layout.keys.len());
    visit_coupled_leg_blocks(homspace, layout, |block| {
        degeneracy_blocks.push(block);
        Ok(())
    })?;

    let sector_structure = layout.sector.clone()?;
    let degeneracy_structure = DegeneracyStructure::from_blocks_with_rank(rank, degeneracy_blocks)?;
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

        let rows = prefix_offsets_into(&row_dims, &mut row_offsets)?;
        let cols = prefix_offsets_into(&col_dims, &mut col_offsets)?;
        let matrix =
            CoupledSectorMatrix::new(sector_offset, &row_offsets, rows, &col_offsets, cols);

        for col in 0..sector.col_count {
            for row in 0..sector.row_count {
                let mut shape = DimVec::with_capacity(rank);
                shape.extend_from_slice(&row_shapes[row]);
                shape.extend_from_slice(&col_shapes[col]);

                let mut strides = DimVec::new();
                let offset = matrix.place(row, col, &shape, nout, &mut strides)?;
                visit(DegeneracyBlock::new(shape, strides, offset)?)?;
            }
        }

        sector_offset = matrix.end()?;
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
pub(crate) fn coupled_sector_matrix_block_specs<K, S>(
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

pub(crate) fn validate_coupled_sector_matrix_dimensions<'shape, S>(
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

pub(crate) fn coupled_sector_matrix_block_specs_after_dimension_validation<K, S>(
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
    let mut row_side = CoupledMatrixSide::<&FusionTreeKey, usize>::default();
    let mut col_side = CoupledMatrixSide::<&FusionTreeKey, usize>::default();
    let (mut row_offsets, mut row_dims) = (DimVec::new(), DimVec::new());
    let (mut col_offsets, mut col_dims) = (DimVec::new(), DimVec::new());
    while run_start < keys.len() {
        let coupled = keys[run_start].borrow().codomain_tree().coupled();
        let mut run_end = run_start;
        while run_end < keys.len() && keys[run_end].borrow().codomain_tree().coupled() == coupled {
            if keys[run_end].borrow().domain_tree().coupled() != coupled {
                return Err(CoreError::MalformedFusionTree {
                    message: "codomain and domain trees must share the coupled sector",
                });
            }
            run_end += 1;
        }

        // Row/column blocks keep first-seen order through core's placement
        // rule; the positional offsets and dimensions feed the sector matrix.
        row_side.clear();
        col_side.clear();
        row_offsets.clear();
        row_dims.clear();
        col_offsets.clear();
        col_dims.clear();
        for index in run_start..run_end {
            let key = keys[index].borrow();
            let shape = shapes[index].as_ref();
            let row_dim = checked_product(&shape[..nout])?;
            let col_dim = checked_product(&shape[nout..])?;
            register_first_seen_block(
                &mut row_side,
                &mut row_offsets,
                &mut row_dims,
                key.codomain_tree(),
                row_dim,
            )?;
            register_first_seen_block(
                &mut col_side,
                &mut col_offsets,
                &mut col_dims,
                key.domain_tree(),
                col_dim,
            )?;
        }
        let expected_blocks = row_dims
            .len()
            .checked_mul(col_dims.len())
            .ok_or(CoreError::ElementCountOverflow)?;
        if run_end - run_start != expected_blocks {
            return Err(CoreError::BlockCountMismatch {
                expected: expected_blocks,
                actual: run_end - run_start,
            });
        }
        let matrix = CoupledSectorMatrix::new(
            sector_offset,
            &row_offsets,
            row_side.extent(),
            &col_offsets,
            col_side.extent(),
        );

        for index in run_start..run_end {
            let key = keys[index].borrow();
            let shape = shapes[index].as_ref();
            let row = *row_side
                .get(key.codomain_tree())
                .expect("row tree placed above");
            let col = *col_side
                .get(key.domain_tree())
                .expect("column tree placed above");
            let mut strides = Vec::with_capacity(rank);
            let offset = matrix.place(row, col, shape, nout, &mut strides)?;
            specs.push(BlockSpec::with_key(
                BlockKey::FusionTree(key.clone()),
                shape.to_vec(),
                strides,
                offset,
            )?);
        }

        sector_offset = matrix.end()?;
        run_start = run_end;
    }
    Ok(specs)
}

/// Registers `tree`'s row or column block on first sight, after the
/// previously seen blocks; a repeated tree must keep its first extent.
/// Places `tree`'s row or column block through core's placement rule; a
/// repeated tree must keep its first dimension.
fn register_first_seen_block<'k>(
    side: &mut CoupledMatrixSide<&'k FusionTreeKey, usize>,
    offsets: &mut DimVec,
    dims: &mut DimVec,
    tree: &'k FusionTreeKey,
    dim: usize,
) -> Result<(), CoreError> {
    let (index, new) = side.place(
        tree,
        || Ok(dim),
        || CoreError::ElementCountOverflow,
        |placed| {
            offsets.push(placed.offset);
            dims.push(placed.dim);
            placed.index
        },
    )?;
    if !new && dims[index] != dim {
        return Err(CoreError::DimensionMismatch {
            expected: dims[index],
            actual: dim,
        });
    }
    Ok(())
}

/// Fills `offsets` with the running sums of `dims` and returns their total.
fn prefix_offsets_into(dims: &[usize], offsets: &mut DimVec) -> Result<usize, CoreError> {
    offsets.clear();
    offsets.reserve(dims.len());
    let mut offset = 0usize;
    for &dim in dims {
        offsets.push(offset);
        offset = offset
            .checked_add(dim)
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    Ok(offset)
}
