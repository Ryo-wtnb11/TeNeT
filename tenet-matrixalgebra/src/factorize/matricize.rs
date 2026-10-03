use super::*;

pub(super) struct SectorMatricization<D> {
    pub(super) sector: SectorId,
    pub(super) rows: usize,
    pub(super) cols: usize,
    /// (codomain tree, row offset, codomain degeneracy shape)
    pub(super) row_trees: Vec<(FusionTreeKey, usize, Vec<usize>)>,
    /// (domain tree, column offset, domain degeneracy shape)
    pub(super) col_trees: Vec<(FusionTreeKey, usize, Vec<usize>)>,
    /// Column-major `rows x cols` matrix.
    pub(super) data: Vec<D>,
}

#[derive(Clone, Copy)]
pub(super) struct TreeExtentRef<'a> {
    pub(super) tree: &'a FusionTreeKey,
    pub(super) offset: usize,
    pub(super) shape: &'a [usize],
}

// Keeps publication generic over owned packs and cached regions without
// allocating a second list of borrowed tree descriptors.
pub(super) trait SectorGeometry {
    fn sector(&self) -> SectorId;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn tree_count(&self, side: FactorSide) -> usize;
    fn tree(&self, side: FactorSide, index: usize) -> Option<TreeExtentRef<'_>>;
}

impl<D> SectorGeometry for SectorMatricization<D> {
    fn sector(&self) -> SectorId {
        self.sector
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn tree_count(&self, side: FactorSide) -> usize {
        match side {
            FactorSide::Left => self.row_trees.len(),
            FactorSide::Right => self.col_trees.len(),
        }
    }

    fn tree(&self, side: FactorSide, index: usize) -> Option<TreeExtentRef<'_>> {
        let trees = match side {
            FactorSide::Left => &self.row_trees,
            FactorSide::Right => &self.col_trees,
        };
        trees.get(index).map(|(tree, offset, shape)| TreeExtentRef {
            tree,
            offset: *offset,
            shape,
        })
    }
}

impl SectorGeometry for CoupledSectorRegion {
    fn sector(&self) -> SectorId {
        self.coupled()
    }

    fn rows(&self) -> usize {
        self.rows()
    }

    fn cols(&self) -> usize {
        self.cols()
    }

    fn tree_count(&self, side: FactorSide) -> usize {
        match side {
            FactorSide::Left => self.row_trees().len(),
            FactorSide::Right => self.col_trees().len(),
        }
    }

    fn tree(&self, side: FactorSide, index: usize) -> Option<TreeExtentRef<'_>> {
        let extent = match side {
            FactorSide::Left => self.row_trees().get(index),
            FactorSide::Right => self.col_trees().get(index),
        }?;
        Some(TreeExtentRef {
            tree: extent.tree(),
            offset: extent.offset(),
            shape: extent.shape(),
        })
    }
}

pub(super) struct SectorMatrixRef<'a, D> {
    pub(super) sector: SectorId,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) data: &'a [D],
}

pub(super) enum InputMatricizations<'a, D> {
    Regions {
        data: &'a [D],
        regions: Arc<[CoupledSectorRegion]>,
    },
    Packed(Vec<SectorMatricization<D>>),
}

impl<'a, D: FactorScalar> InputMatricizations<'a, D> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Regions { regions, .. } => regions.len(),
            Self::Packed(matrices) => matrices.len(),
        }
    }

    pub(super) fn get(&self, index: usize) -> Result<SectorMatrixRef<'_, D>, OperationError> {
        match self {
            Self::Regions { data, regions } => {
                let region = &regions[index];
                let range = region.range();
                let matrix =
                    data.get(range.clone())
                        .ok_or(OperationError::ElementCountMismatch {
                            expected: range.end,
                            actual: data.len(),
                        })?;
                Ok(SectorMatrixRef {
                    sector: region_sector(region),
                    rows: region.rows(),
                    cols: region.cols(),
                    data: matrix,
                })
            }
            Self::Packed(matrices) => {
                let matrix = &matrices[index];
                Ok(SectorMatrixRef {
                    sector: matrix.sector,
                    rows: matrix.rows,
                    cols: matrix.cols,
                    data: &matrix.data,
                })
            }
        }
    }

    #[cfg(test)]
    pub(super) fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }

    /// Eigenvalues are basis-invariant only when row `i` and column `i` name
    /// the same tree state, so each sector must stack its row and column trees
    /// identically (outer-multiplicity vertices included).
    pub(super) fn validate_endomorphism_stacking(
        &self,
        message: &'static str,
    ) -> Result<(), OperationError> {
        match self {
            Self::Regions { regions, .. } => {
                validate_endomorphism_tree_stacking(regions.as_ref(), message)
            }
            Self::Packed(matrices) => validate_endomorphism_tree_stacking(matrices, message),
        }
    }

    pub(super) fn validate_hermitian(&self) -> Result<(), OperationError> {
        match self {
            Self::Regions { data, regions } => validate_hermitian_regions(data, regions),
            Self::Packed(matrices) => validate_hermitian_matricizations(matrices),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckedCompactOperation {
    Qr,
    Svd,
    Lq,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CheckedCompactInputObservation {
    pub operation: CheckedCompactOperation,
    pub input_pointer: usize,
    pub matrix_pointer: usize,
    pub adjoint_pointer: Option<usize>,
    pub elements: usize,
}

#[cfg(test)]
pub(crate) fn reset_checked_compact_input_observations() {
    CHECKED_COMPACT_INPUT_OBSERVATIONS.with(|observations| observations.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn checked_compact_input_observations() -> Vec<CheckedCompactInputObservation> {
    CHECKED_COMPACT_INPUT_OBSERVATIONS.with(|observations| observations.borrow().clone())
}

#[cfg(test)]
pub(super) fn record_checked_compact_input<D>(
    operation: CheckedCompactOperation,
    input: &[D],
    matrix: &[D],
    adjoint: Option<&[D]>,
) {
    CHECKED_COMPACT_INPUT_OBSERVATIONS.with(|observations| {
        observations
            .borrow_mut()
            .push(CheckedCompactInputObservation {
                operation,
                input_pointer: input.as_ptr() as usize,
                matrix_pointer: matrix.as_ptr() as usize,
                adjoint_pointer: adjoint.map(|data| data.as_ptr() as usize),
                elements: matrix.len(),
            });
    });
}

#[cfg(test)]
pub(crate) fn reset_values_matricization_fallbacks() {
    VALUES_MATRICIZATION_FALLBACKS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn values_matricization_fallbacks() -> usize {
    VALUES_MATRICIZATION_FALLBACKS.with(Cell::get)
}

#[cfg(test)]
pub(super) fn record_values_matricization_fallback() {
    VALUES_MATRICIZATION_FALLBACKS.with(|count| count.set(count.get() + 1));
}

#[cfg(feature = "diagnostics")]
#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectorMatricizationDiagnostic {
    pub sector: SectorId,
    pub rows: usize,
    pub cols: usize,
    pub elements: usize,
}

#[cfg(feature = "diagnostics")]
#[doc(hidden)]
pub fn sector_matricization_diagnostic<R, D>(
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorMatricizationDiagnostic>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    Ok(
        sector_matricizations(space.structure(), input.data(), space.nout())?
            .into_iter()
            .map(|matrix| SectorMatricizationDiagnostic {
                sector: matrix.sector,
                rows: matrix.rows,
                cols: matrix.cols,
                elements: matrix.data.len(),
            })
            .collect(),
    )
}

pub(super) fn coupled_of(tree: &FusionTreeKey) -> SectorId {
    tree.coupled()
}

pub(super) fn matricization_map<M: SectorGeometry>(
    matricizations: &[M],
) -> FxHashMap<SectorId, &M> {
    matricizations
        .iter()
        .map(|matrix| (matrix.sector(), matrix))
        .collect()
}

pub(super) fn matricization_of<'a, M>(
    matricizations: &FxHashMap<SectorId, &'a M>,
    sector: SectorId,
) -> Result<&'a M, OperationError> {
    matricizations
        .get(&sector)
        .copied()
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: "factor tree references a coupled sector absent from the source tensor",
        })
}

pub(super) fn validate_dense_shape(
    actual: &[usize],
    expected: &[usize],
) -> Result<(), OperationError> {
    if actual != expected {
        return Err(OperationError::ShapeMismatch {
            dst: expected.to_vec(),
            src: actual.to_vec(),
        });
    }
    Ok(())
}

pub(super) fn data_region<'a, D>(
    data: &'a [D],
    range: &std::ops::Range<usize>,
) -> Result<&'a [D], OperationError> {
    data.get(range.clone())
        .ok_or(OperationError::ElementCountMismatch {
            expected: range.end,
            actual: data.len(),
        })
}

pub(super) fn checked_sector_regions(
    structure: &BlockStructure,
    nout: usize,
) -> Result<Option<Arc<[CoupledSectorRegion]>>, OperationError> {
    structure
        .coupled_sector_regions(nout)
        .map_err(OperationError::from_core_preserving_context)
}

pub(super) fn generic_value_matricizations<'a, D>(
    structure: &BlockStructure,
    data: &'a [D],
    nout: usize,
) -> Result<InputMatricizations<'a, D>, OperationError>
where
    D: FactorScalar,
{
    let matricizations = generic_input_matricizations(structure, data, nout)?;
    #[cfg(test)]
    if matricizations.is_packed() {
        record_values_matricization_fallback();
    }
    Ok(matricizations)
}

pub(super) fn generic_input_matricizations<'a, D>(
    structure: &BlockStructure,
    data: &'a [D],
    nout: usize,
) -> Result<InputMatricizations<'a, D>, OperationError>
where
    D: FactorScalar,
{
    // Why no tree-order admission (`FusionTreeKey` Ord, which the facade
    // builder's multi-leg order fails): a region lists its sector and tree
    // extents in first-appearance block order and proves the dense column-major
    // matrix at `range`, which is exactly the matrix, sector order and tree
    // order `sector_matricizations_generic` would pack from the same tiling.
    // Every consumer therefore sees the same matricization either way; output
    // tree order is proven per output by `factor_output_is_canonical` (else a
    // by-tree scatter), and the eigenvalue ops check endomorphism stacking.
    Ok(match input_regions(structure, nout)? {
        Some(regions) => InputMatricizations::Regions { data, regions },
        None => InputMatricizations::Packed(sector_matricizations_generic(structure, data, nout)?),
    })
}

/// Multiplicity-free sibling of [`generic_input_matricizations`]; the same
/// region admission applies because both packers stack sectors and trees in
/// first-appearance order.
pub(super) fn multiplicity_free_input_matricizations<'a, D>(
    structure: &BlockStructure,
    data: &'a [D],
    nout: usize,
) -> Result<InputMatricizations<'a, D>, OperationError>
where
    D: FactorScalar,
{
    Ok(match input_regions(structure, nout)? {
        Some(regions) => InputMatricizations::Regions { data, regions },
        None => InputMatricizations::Packed(sector_matricizations(structure, data, nout)?),
    })
}

/// The single admission authority for lending input regions to the
/// numerical stages instead of packing them.
pub(super) fn input_regions(
    structure: &BlockStructure,
    nout: usize,
) -> Result<Option<Arc<[CoupledSectorRegion]>>, OperationError> {
    #[cfg(test)]
    if FORCE_INPUT_PACK.with(Cell::get) {
        return Ok(None);
    }
    checked_sector_regions(structure, nout)
}

#[cfg(test)]
thread_local! {
    pub(super) static FORCE_INPUT_PACK: Cell<bool> = const { Cell::new(false) };
    pub(super) static INPUT_PACK_BYTES: Cell<usize> = const { Cell::new(0) };
}

/// Runs `f` with region admission disabled, so every input packs.
#[cfg(test)]
pub(crate) fn with_forced_input_pack<T>(f: impl FnOnce() -> T) -> T {
    let previous = FORCE_INPUT_PACK.with(|force| force.replace(true));
    let result = f();
    FORCE_INPUT_PACK.with(|force| force.set(previous));
    result
}

#[cfg(test)]
pub(crate) fn reset_input_pack_bytes() {
    INPUT_PACK_BYTES.with(|bytes| bytes.set(0));
}

/// Bytes allocated by `sector_matricizations{,_generic}` on this thread.
#[cfg(test)]
pub(crate) fn input_pack_bytes() -> usize {
    INPUT_PACK_BYTES.with(Cell::get)
}

#[cfg(test)]
pub(super) fn record_input_pack_bytes<D>(matricizations: &[SectorMatricization<D>]) {
    let bytes = matricizations
        .iter()
        .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
        .sum::<usize>();
    INPUT_PACK_BYTES.with(|total| total.set(total.get() + bytes));
}

pub(super) fn value_matricizations<'a, D>(
    structure: &BlockStructure,
    data: &'a [D],
    nout: usize,
) -> Result<InputMatricizations<'a, D>, OperationError>
where
    D: FactorScalar,
{
    let matricizations = multiplicity_free_input_matricizations(structure, data, nout)?;
    #[cfg(test)]
    if matricizations.is_packed() {
        record_values_matricization_fallback();
    }
    Ok(matricizations)
}

/// Packs every coupled sector of the source data into its dense column-major
/// matricization, independent of the storage layout.
pub(super) fn sector_matricizations<D>(
    structure: &BlockStructure,
    data: &[D],
    nout: usize,
) -> Result<Vec<SectorMatricization<D>>, OperationError>
where
    D: FactorScalar,
{
    let mut matricizations: Vec<SectorMatricization<D>> = Vec::new();
    let mut matrix_indices = FxHashMap::default();
    let mut row_offsets: Vec<FxHashMap<&FusionTreeKey, usize>> = Vec::new();
    let mut col_offsets: Vec<FxHashMap<&FusionTreeKey, usize>> = Vec::new();
    let mut routes = Vec::with_capacity(structure.block_count());

    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "tsvd",
                index,
            });
        };
        let sector = coupled_of(key.codomain_tree());
        let row_dim: usize = block.shape()[..nout].iter().product();
        let col_dim: usize = block.shape()[nout..].iter().product();
        let matrix_index = match matrix_indices.get(&sector) {
            Some(&matrix_index) => matrix_index,
            None => {
                let matrix_index = matricizations.len();
                matricizations.push(SectorMatricization::<D> {
                    sector,
                    rows: 0,
                    cols: 0,
                    row_trees: Vec::new(),
                    col_trees: Vec::new(),
                    data: Vec::new(),
                });
                matrix_indices.insert(sector, matrix_index);
                row_offsets.push(FxHashMap::default());
                col_offsets.push(FxHashMap::default());
                matrix_index
            }
        };
        let matrix = &mut matricizations[matrix_index];
        let row_offset = match row_offsets[matrix_index].get(key.codomain_tree()) {
            Some(&offset) => offset,
            None => {
                let offset = matrix.rows;
                matrix.row_trees.push((
                    key.codomain_tree().clone(),
                    offset,
                    block.shape()[..nout].to_vec(),
                ));
                matrix.rows += row_dim;
                row_offsets[matrix_index].insert(key.codomain_tree(), offset);
                offset
            }
        };
        let col_offset = match col_offsets[matrix_index].get(key.domain_tree()) {
            Some(&offset) => offset,
            None => {
                let offset = matrix.cols;
                matrix.col_trees.push((
                    key.domain_tree().clone(),
                    offset,
                    block.shape()[nout..].to_vec(),
                ));
                matrix.cols += col_dim;
                col_offsets[matrix_index].insert(key.domain_tree(), offset);
                offset
            }
        };
        routes.push((matrix_index, row_offset, col_offset));
    }
    for matrix in &mut matricizations {
        matrix.data = vec![D::zero(); matrix.rows * matrix.cols];
    }

    for (index, (matrix_index, row_offset, col_offset)) in routes.into_iter().enumerate() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let matrix = &mut matricizations[matrix_index];
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        let rows = matrix.rows;
        copy_tensor_block_to_matrix(
            data,
            shape,
            strides,
            offset,
            nout,
            &mut matrix.data,
            rows,
            row_offset,
            col_offset,
        );
    }
    #[cfg(test)]
    record_input_pack_bytes(&matricizations);
    Ok(matricizations)
}

// ============================================================================
// Stage B3c-2: Generic-fusion (SU(N)) siblings.
//
// Parallel `*_generic` siblings of the mult-free factorization entry points.
// The block-level engine — the dense SVD/QR per coupled sector, the gauges,
// the workspace scatters, `diagonal_bond_data`, and every copy helper — is
// symmetry-agnostic and SHARED. A sibling differs from its original in
// exactly three substitutions:
//   1. bound: `MultiplicityFreeRigidSymbols` -> `FusionRule` (or
//      `GenericRigidSymbols` where the truncation weight needs rigid data);
//   2. key enumeration: `fusion_tree_keys` -> `fusion_tree_keys_generic` and
//      `from_degeneracy_shapes` -> `from_degeneracy_shapes_generic`, so the
//      factor spaces carry multiplicity-aware (vertex-labelled) trees — the
//      matricization already stacks ALL trees of a coupled sector into one
//      dense block (TensorKit `block(t, c)`), so outer multiplicity rides the
//      row/col tree lists with no math change;
//   3. truncation dim weight: `dim_scalar(c)` -> `sqrt_dim(c)²`, preserving
//      non-integer quantum dimensions instead of assuming an SU(N)-only rule,
//      matching the mult-free weighted-truncation convention.
// Duplicated rather than bound-relaxed so the mult-free path stays
// byte-for-byte untouched (the B-series byte-invariance rule; the same
// rationale as the B3c-1 `is_core_form_..._generic` sibling).
// ============================================================================

pub(super) fn coupled_of_generic(tree: &FusionTreeKey) -> SectorId {
    tree.coupled()
}

/// Fallible coupled-sector reduced dimensions for checked Generic providers.
/// This is a structural dynamic program: it never expands dense tensor data or
/// publishes a factor-space cache.
#[doc(hidden)]
pub fn coupled_sector_block_dimensions_generic_checked<R>(
    product: &FusionProductSpace,
    rule: &R,
) -> Result<BTreeMap<SectorId, usize>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
{
    let mut dimensions = BTreeMap::from([(rule.vacuum(), 1usize)]);
    for leg in product.legs() {
        let mut next = BTreeMap::<SectorId, usize>::new();
        for (&left, &left_dimension) in &dimensions {
            for (right, right_degeneracy) in leg.iter() {
                let channels = rule
                    .try_fusion_channels(left, right)
                    .map_err(CheckedGenericFactorPlanError::Provider)?;
                for coupled in channels {
                    let multiplicity = rule
                        .try_nsymbol(left, right, coupled)
                        .map_err(CheckedGenericFactorPlanError::Provider)?;
                    let contribution = left_dimension
                        .checked_mul(right_degeneracy)
                        .and_then(|value| value.checked_mul(multiplicity))
                        .ok_or(CheckedGenericFactorPlanError::Operation(
                            OperationError::ElementCountOverflow,
                        ))?;
                    let entry = next.entry(coupled).or_default();
                    *entry = entry.checked_add(contribution).ok_or(
                        CheckedGenericFactorPlanError::Operation(
                            OperationError::ElementCountOverflow,
                        ),
                    )?;
                }
            }
        }
        dimensions = next;
    }
    Ok(dimensions)
}

/// Generic sibling of [`sector_matricizations`]: identical two-pass stacking
/// (vertex-labelled trees are distinct keys, so OM trees get distinct rows /
/// columns of the coupled block, exactly TensorKit's `block(t, c)` layout).
pub(super) fn sector_matricizations_generic<D>(
    structure: &BlockStructure,
    data: &[D],
    nout: usize,
) -> Result<Vec<SectorMatricization<D>>, OperationError>
where
    D: FactorScalar,
{
    #[derive(Clone, Copy, Default)]
    struct TreePlacement {
        pub(super) row_offset: Option<usize>,
        pub(super) col_offset: Option<usize>,
    }

    let mut matricizations: Vec<SectorMatricization<D>> = Vec::new();
    let mut matrix_indices = FxHashMap::default();
    let mut tree_placements = FxHashMap::default();
    let mut routes = Vec::with_capacity(structure.block_count());

    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "tsvd",
                index,
            });
        };
        let sector = coupled_of_generic(key.codomain_tree());
        let row_dim: usize = block.shape()[..nout].iter().product();
        let col_dim: usize = block.shape()[nout..].iter().product();
        let matrix_index = match matrix_indices.get(&sector).copied() {
            Some(matrix_index) => matrix_index,
            None => {
                let matrix_index = matricizations.len();
                matricizations.push(SectorMatricization::<D> {
                    sector,
                    rows: 0,
                    cols: 0,
                    row_trees: Vec::new(),
                    col_trees: Vec::new(),
                    data: Vec::new(),
                });
                matrix_indices.insert(sector, matrix_index);
                matrix_index
            }
        };
        let matrix = &mut matricizations[matrix_index];
        let row_placement = tree_placements
            .entry((matrix_index, key.codomain_tree()))
            .or_insert_with(TreePlacement::default);
        if row_placement.row_offset.is_none() {
            let offset = matrix.rows;
            matrix.row_trees.push((
                key.codomain_tree().clone(),
                offset,
                block.shape()[..nout].to_vec(),
            ));
            matrix.rows += row_dim;
            row_placement.row_offset = Some(offset);
        }
        let row_offset = row_placement.row_offset.expect("row tree registered above");
        let col_placement = tree_placements
            .entry((matrix_index, key.domain_tree()))
            .or_insert_with(TreePlacement::default);
        if col_placement.col_offset.is_none() {
            let offset = matrix.cols;
            matrix.col_trees.push((
                key.domain_tree().clone(),
                offset,
                block.shape()[nout..].to_vec(),
            ));
            matrix.cols += col_dim;
            col_placement.col_offset = Some(offset);
        }
        let col_offset = col_placement
            .col_offset
            .expect("column tree registered above");
        routes.push((matrix_index, row_offset, col_offset));
    }
    drop(matrix_indices);
    drop(tree_placements);
    for matrix in &mut matricizations {
        matrix.data = vec![D::zero(); matrix.rows * matrix.cols];
    }

    for (index, (matrix_index, row_offset, col_offset)) in routes.into_iter().enumerate() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let matrix = &mut matricizations[matrix_index];

        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        let rows = matrix.rows;
        copy_tensor_block_to_matrix(
            data,
            shape,
            strides,
            offset,
            nout,
            &mut matrix.data,
            rows,
            row_offset,
            col_offset,
        );
    }
    #[cfg(test)]
    record_input_pack_bytes(&matricizations);
    Ok(matricizations)
}

/// Row `i` and column `i` of every sector name the same tree state (tree key
/// including outer-multiplicity vertices, offset and shape), so the block is
/// an endomorphism matrix in one basis and its spectrum is basis-invariant.
/// Generic over [`SectorGeometry`] so packed matricizations and borrowed
/// regions share this one predicate.
pub(super) fn endomorphism_tree_stacking_is_identical<M: SectorGeometry>(matrices: &[M]) -> bool {
    matrices.iter().all(|matrix| {
        let count = matrix.tree_count(FactorSide::Left);
        matrix.rows() == matrix.cols()
            && count == matrix.tree_count(FactorSide::Right)
            && (0..count).all(|index| {
                match (
                    matrix.tree(FactorSide::Left, index),
                    matrix.tree(FactorSide::Right, index),
                ) {
                    (Some(row), Some(col)) => {
                        row.tree == col.tree && row.offset == col.offset && row.shape == col.shape
                    }
                    _ => false,
                }
            })
    })
}

#[doc(hidden)]
pub const EIGH_FULL_STACKING: &str =
    "eigh_full requires identical endomorphism row/column fusion-tree stacking";

pub(super) const EXP_STACKING: &str =
    "exp requires identical endomorphism row/column fusion-tree stacking";

/// [`validate_endomorphism_tree_stacking`] over canonical coupled-sector
/// regions, for device paths outside this crate. `message` names the
/// refusing operation.
#[doc(hidden)]
pub fn validate_endomorphism_region_stacking(
    regions: &[CoupledSectorRegion],
    message: &'static str,
) -> Result<(), OperationError> {
    validate_endomorphism_tree_stacking(regions, message)
}

/// `message` names the refusing operation.
pub(super) fn validate_endomorphism_tree_stacking<M: SectorGeometry>(
    matrices: &[M],
    message: &'static str,
) -> Result<(), OperationError> {
    if endomorphism_tree_stacking_is_identical(matrices) {
        Ok(())
    } else {
        Err(OperationError::UnsupportedTensorContractScope { message })
    }
}
