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

    /// Pushes the strides of block `(row, col)` with codomain degeneracies
    /// `row_shape` and domain degeneracies `col_shape`, and returns its
    /// storage offset.
    fn place(
        &self,
        row: usize,
        col: usize,
        row_shape: &[usize],
        col_shape: &[usize],
        strides: &mut impl Extend<usize>,
    ) -> Result<usize, CoreError> {
        let mut push = |stride| {
            strides.extend(Some(stride));
            Ok(())
        };
        try_for_each_column_major_stride(1, row_shape, &mut push, || {
            CoreError::ElementCountOverflow
        })?;
        try_for_each_column_major_stride(self.rows, col_shape, push, || {
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

/// Writes the degeneracy of each leg of `space` in `tree`'s sector to
/// `shape` and returns their product.
fn degeneracy_shape_for_tree_side(
    space: &FusionProductSpace,
    tree: &FusionTreeKey,
    shape: &mut [usize],
) -> Result<usize, CoreError> {
    #[cfg(test)]
    observe_coupled_grid_side_derivation();
    if tree.uncoupled().len() != space.len() {
        return Err(CoreError::StructureRankMismatch {
            expected: space.len(),
            actual: tree.uncoupled().len(),
        });
    }
    for ((leg, &sector), axis) in space.legs().iter().zip(tree.uncoupled()).zip(&mut *shape) {
        *axis = leg
            .degeneracy(sector)
            .ok_or(CoreError::MalformedFusionTree {
                message: "fusion tree uses a sector absent from its leg",
            })?;
    }
    checked_product(shape)
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
    let mut degeneracy =
        DegeneracyStructure::with_block_capacity(homspace.rank(), layout.keys.len());
    visit_coupled_leg_blocks(homspace, layout, &mut degeneracy)?;
    let sector_structure = layout.sector.clone()?;
    Ok((sector_structure, degeneracy))
}

/// One side's row (or column) blocks of the current coupled sector: the
/// degeneracy shapes back to back, each block's dimension, and its offset.
/// Each region is sized for the largest sector of the structure.
struct CoupledSide<'a> {
    legs: usize,
    shapes: &'a mut [usize],
    dims: &'a mut [usize],
    offsets: &'a mut [usize],
}

impl<'a> CoupledSide<'a> {
    /// Splits `trees` blocks of `legs` legs off the front of `scratch`.
    fn split(scratch: &'a mut [usize], trees: usize, legs: usize) -> (Self, &'a mut [usize]) {
        let (shapes, rest) = scratch.split_at_mut(trees * legs);
        let (dims, rest) = rest.split_at_mut(trees);
        let (offsets, rest) = rest.split_at_mut(trees);
        let side = Self {
            legs,
            shapes,
            dims,
            offsets,
        };
        (side, rest)
    }

    /// Derives the shape, dimension and offset of every tree `tree(index)`
    /// for `index < count` on `space`, and returns the side's extent.
    fn derive<'k>(
        &mut self,
        space: &FusionProductSpace,
        count: usize,
        tree: impl Fn(usize) -> Result<&'k FusionTreeKey, CoreError>,
    ) -> Result<usize, CoreError> {
        for index in 0..count {
            self.dims[index] =
                degeneracy_shape_for_tree_side(space, tree(index)?, self.shape_mut(index))?;
        }
        let mut extent = 0usize;
        for index in 0..count {
            self.offsets[index] = extent;
            extent = extent
                .checked_add(self.dims[index])
                .ok_or(CoreError::ElementCountOverflow)?;
        }
        Ok(extent)
    }

    fn shape(&self, index: usize) -> &[usize] {
        &self.shapes[index * self.legs..(index + 1) * self.legs]
    }

    fn shape_mut(&mut self, index: usize) -> &mut [usize] {
        &mut self.shapes[index * self.legs..(index + 1) * self.legs]
    }
}

fn visit_coupled_leg_blocks(
    homspace: &FusionTreeHomSpace,
    layout: &FusionTreeHomSpaceLayoutData,
    degeneracy: &mut DegeneracyStructure,
) -> Result<(), CoreError> {
    let nout = homspace.codomain().len();
    let nin = homspace.domain().len();
    // Why one scratch buffer sized for the largest sector rather than
    // per-tree `DimVec`s or inline buffers: every structure then costs this
    // one allocation at any rank and block count (#2012). A well-formed
    // sector has at most one row or column per block; the clamp leaves a
    // malformed one to the block-count error below.
    let (max_rows, max_cols) = layout.sectors.iter().fold((0, 0), |(rows, cols), sector| {
        (
            rows.max(sector.row_count.min(layout.keys.len())),
            cols.max(sector.col_count.min(layout.keys.len())),
        )
    });
    let mut scratch = vec![0usize; max_rows * (nout + 2) + max_cols * (nin + 2)];
    let (mut rows, rest) = CoupledSide::split(&mut scratch, max_rows, nout);
    let (mut cols, _) = CoupledSide::split(rest, max_cols, nin);
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
        if run_end > layout.keys.len() {
            return Err(CoreError::BlockCountMismatch {
                expected: run_end,
                actual: layout.keys.len(),
            });
        }

        let row_extent = rows.derive(homspace.codomain(), sector.row_count, |row| {
            Ok(layout.keys[sector.start + row].codomain_tree())
        })?;
        let col_extent = cols.derive(homspace.domain(), sector.col_count, |col| {
            let local_offset = col
                .checked_mul(sector.row_count)
                .ok_or(CoreError::ElementCountOverflow)?;
            Ok(layout.keys[sector.start + local_offset].domain_tree())
        })?;
        let matrix = CoupledSectorMatrix::new(
            sector_offset,
            rows.offsets,
            row_extent,
            cols.offsets,
            col_extent,
        );

        for col in 0..sector.col_count {
            for row in 0..sector.row_count {
                let (row_shape, col_shape) = (rows.shape(row), cols.shape(col));
                degeneracy.push_block_with(|dims| {
                    dims.extend_from_slice(row_shape);
                    dims.extend_from_slice(col_shape);
                    matrix.place(row, col, row_shape, col_shape, dims)
                })?;
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
            let (row_shape, col_shape) = shape.split_at(nout);
            let offset = matrix.place(row, col, row_shape, col_shape, &mut strides)?;
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
