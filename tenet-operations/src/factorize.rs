use tenet_core::{
    BlockKey, CoreError, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace,
    FusionTreeKey, MultiplicityFreeRigidSymbols, SectorId, SectorLeg, TensorMap, TensorMapSpace,
};
use tenet_dense::{DenseExecutor, DenseRead, DenseView};

use crate::OperationError;

/// Singular values of one coupled sector, descending.
#[derive(Clone, Debug, PartialEq)]
pub struct SectorSingularValues {
    pub sector: SectorId,
    pub values: Vec<f64>,
}

/// Full (untruncated) fusion-tensor SVD `t = U * diag(S) * Vt`.
///
/// TensorKit's `tsvd` semantics: the SVD acts blockwise on the coupled-sector
/// matricization (rows = codomain trees x degeneracies, columns = domain trees
/// x degeneracies). `U : codomain <- W` and `Vt : W <- domain`, where `W` is a
/// new single leg carrying every coupled sector with degeneracy
/// `min(rows, cols)`. Truncation policies come later; this returns the full
/// factorization.
#[derive(Clone, Debug)]
pub struct FusionSvd<const NOUT: usize, const NIN: usize> {
    pub u: TensorMap<f64, NOUT, 1>,
    pub singular_values: Vec<SectorSingularValues>,
    pub vt: TensorMap<f64, 1, NIN>,
}

struct SectorMatricization {
    sector: SectorId,
    rows: usize,
    cols: usize,
    /// (codomain tree, row offset, codomain degeneracy shape)
    row_trees: Vec<(FusionTreeKey, usize, Vec<usize>)>,
    /// (domain tree, column offset, domain degeneracy shape)
    col_trees: Vec<(FusionTreeKey, usize, Vec<usize>)>,
    /// Column-major `rows x cols` matrix.
    data: Vec<f64>,
}

struct SectorFactors {
    sector: SectorId,
    /// Full rank of the dense factorization (leading dimension of `vt`).
    rank: usize,
    /// Kept singular values after truncation.
    kept: usize,
    rows: usize,
    u: Vec<f64>,
    vt: Vec<f64>,
}

/// Applies a TensorKit-style global truncation across sectors, updating the
/// kept counts and singular-value lists in place, and returns the truncation
/// error `sqrt(sum_discarded d_c * sigma^2)`.
fn apply_truncation<R>(
    rule: &R,
    factors: &mut [SectorFactors],
    singular_values: &mut [SectorSingularValues],
    truncation: SvdTruncation,
) -> Result<f64, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if matches!(truncation, SvdTruncation::None) {
        return Ok(0.0);
    }

    // (sector position, value); globally sorted descending by value.
    let mut entries: Vec<(usize, f64, f64)> = Vec::new();
    for (position, values) in singular_values.iter().enumerate() {
        let weight = rule.dim_scalar(values.sector);
        for &value in &values.values {
            entries.push((position, value, weight));
        }
    }
    entries.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).expect("finite singular values"));

    let keep_count = match truncation {
        SvdTruncation::None => entries.len(),
        SvdTruncation::Dim(max_dim) => {
            let mut total = 0.0;
            let mut keep = 0;
            for &(_, _, weight) in &entries {
                if total + weight > max_dim as f64 + 1e-12 {
                    break;
                }
                total += weight;
                keep += 1;
            }
            keep
        }
        SvdTruncation::Error(tolerance) => {
            let total: f64 = entries
                .iter()
                .map(|&(_, value, weight)| weight * value * value)
                .sum();
            let budget = tolerance * tolerance * total;
            let mut discarded = 0.0;
            let mut keep = entries.len();
            while keep > 0 {
                let (_, value, weight) = entries[keep - 1];
                if discarded + weight * value * value > budget + 1e-15 {
                    break;
                }
                discarded += weight * value * value;
                keep -= 1;
            }
            keep
        }
        SvdTruncation::Below(threshold) => entries
            .iter()
            .take_while(|&&(_, value, _)| value >= threshold)
            .count(),
    };

    let mut kept_per_sector = vec![0usize; singular_values.len()];
    for &(position, _, _) in &entries[..keep_count] {
        kept_per_sector[position] += 1;
    }
    let discarded_square: f64 = entries[keep_count..]
        .iter()
        .map(|&(_, value, weight)| weight * value * value)
        .sum();

    for (position, values) in singular_values.iter_mut().enumerate() {
        values.values.truncate(kept_per_sector[position]);
        let factor = factors
            .iter_mut()
            .find(|factor| factor.sector == values.sector)
            .expect("factor exists for every singular-value sector");
        factor.kept = kept_per_sector[position];
    }
    Ok(discarded_square.sqrt())
}

/// Truncation policy for [`tsvd_fusion_truncated`], TensorKit-style.
///
/// Selection is global across coupled sectors and weights every singular
/// value by its sector's quantum dimension: total dimension counts
/// `sum_c d_c * kept_c` and the truncation error is
/// `sqrt(sum_discarded d_c * sigma^2)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SvdTruncation {
    /// Keep everything.
    None,
    /// Keep the largest singular values while the quantum-dimension-weighted
    /// total stays at or below this bound (TensorKit `truncdim`).
    Dim(usize),
    /// Discard the smallest singular values while the relative truncation
    /// error stays at or below this tolerance (TensorKit `truncerr`).
    Error(f64),
    /// Discard singular values strictly below this threshold (TensorKit
    /// `truncbelow`).
    Below(f64),
}

/// Blockwise SVD of a fusion tensor over its coupled-sector matricization.
pub fn tsvd_fusion<E, R, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<f64, NOUT, NIN>,
) -> Result<FusionSvd<NOUT, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    tsvd_fusion_truncated(dense, rule, tensor, SvdTruncation::None).map(|(svd, _)| svd)
}

/// Truncated blockwise SVD; also returns the truncation error
/// `sqrt(sum_discarded d_c * sigma^2)`.
pub fn tsvd_fusion_truncated<E, R, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<f64, NOUT, NIN>,
    truncation: SvdTruncation,
) -> Result<(FusionSvd<NOUT, NIN>, f64), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?
        .clone();
    let matricizations = sector_matricizations(rule, tensor, NOUT)?;

    let mut factors = Vec::with_capacity(matricizations.len());
    let mut singular_values = Vec::with_capacity(matricizations.len());

    for matrix in &matricizations {
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(&matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let outputs = dense
            .svd(DenseRead::F64(view))
            .map_err(OperationError::Dense)?;
        if outputs.len() != 3 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "dense SVD must return exactly (U, S, Vt)",
            });
        }
        let rank = matrix.rows.min(matrix.cols);
        validate_dense_shape(outputs[0].shape(), &[matrix.rows, rank])?;
        validate_dense_shape(outputs[1].shape(), &[rank])?;
        validate_dense_shape(outputs[2].shape(), &[rank, matrix.cols])?;

        singular_values.push(SectorSingularValues {
            sector: matrix.sector,
            values: outputs[1]
                .as_f64_slice()
                .map_err(OperationError::Dense)?
                .to_vec(),
        });
        factors.push(SectorFactors {
            sector: matrix.sector,
            rank,
            kept: rank,
            rows: matrix.rows,
            u: outputs[0]
                .as_f64_slice()
                .map_err(OperationError::Dense)?
                .to_vec(),
            vt: outputs[2]
                .as_f64_slice()
                .map_err(OperationError::Dense)?
                .to_vec(),
        });
    }

    let truncation_error = apply_truncation(rule, &mut factors, &mut singular_values, truncation)?;
    factors.retain(|factor| factor.kept > 0);
    singular_values.retain(|entry| !entry.values.is_empty());
    let mut new_leg_dim = 0usize;
    for factor in &factors {
        new_leg_dim += factor.kept;
    }

    let sector_rank = |sector: SectorId| -> usize {
        factors
            .iter()
            .find(|factor| factor.sector == sector)
            .map(|factor| factor.kept)
            .unwrap_or(0)
    };

    let new_leg = SectorLeg::new(factors.iter().map(|factor| factor.sector), false);
    let dims = tensor.space().dims();
    let mut codomain_dims = [0usize; NOUT];
    codomain_dims.copy_from_slice(&dims[..NOUT]);
    let mut domain_dims = [0usize; NIN];
    domain_dims.copy_from_slice(&dims[NOUT..]);

    let u_hom = FusionTreeHomSpace::new(
        fusion_space.homspace().codomain().clone(),
        FusionProductSpace::new([new_leg.clone()]),
    );
    let u_keys = u_hom.fusion_tree_keys(rule);
    let u_shapes = u_keys
        .iter()
        .map(|key| {
            let sector = coupled_of(rule, key.codomain_tree());
            let mut shape = row_shape_of(&matricizations, sector, key.codomain_tree())?;
            shape.push(sector_rank(sector));
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    let u_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<NOUT, 1>::from_dims(codomain_dims, [new_leg_dim])
            .map_err(OperationError::from_core_preserving_context)?,
        u_hom,
        rule,
        u_shapes,
    )
    .map_err(OperationError::from_core_preserving_context)?;

    let vt_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg]),
        fusion_space.homspace().domain().clone(),
    );
    let vt_keys = vt_hom.fusion_tree_keys(rule);
    let vt_shapes = vt_keys
        .iter()
        .map(|key| {
            let sector = coupled_of(rule, key.domain_tree());
            let mut shape = vec![sector_rank(sector)];
            shape.extend(col_shape_of(&matricizations, sector, key.domain_tree())?);
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    let vt_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, NIN>::from_dims([new_leg_dim], domain_dims)
            .map_err(OperationError::from_core_preserving_context)?,
        vt_hom,
        rule,
        vt_shapes,
    )
    .map_err(OperationError::from_core_preserving_context)?;

    let u_len = u_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut u_tensor =
        TensorMap::<f64, NOUT, 1>::from_vec_with_fusion_space(vec![0.0; u_len], u_space)
            .map_err(OperationError::from_core_preserving_context)?;
    let vt_len = vt_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut vt_tensor =
        TensorMap::<f64, 1, NIN>::from_vec_with_fusion_space(vec![0.0; vt_len], vt_space)
            .map_err(OperationError::from_core_preserving_context)?;

    // Scatter U blocks: block element (i.., j) = u[(row_offset + rowmaj(i)) + rows * j].
    let u_structure = std::sync::Arc::clone(u_tensor.structure());
    for index in 0..u_structure.block_count() {
        let block = u_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = coupled_of(rule, key.codomain_tree());
        let matrix = matricization_of(&matricizations, sector)?;
        let factor = factors
            .iter()
            .find(|factor| factor.sector == sector)
            .expect("factor exists for every matricized sector");
        let (row_offset, _) = row_placement(matrix, key.codomain_tree())?;
        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        scatter_matrix_block(
            u_tensor.data_mut(),
            &shape,
            &strides,
            offset,
            shape.len() - 1,
            &factor.u,
            factor.rows,
            row_offset,
        );
    }

    // Scatter Vt blocks: block element (r, j..) = vt[r + rank * (col_offset + colmaj(j))].
    let vt_structure = std::sync::Arc::clone(vt_tensor.structure());
    for index in 0..vt_structure.block_count() {
        let block = vt_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = coupled_of(rule, key.domain_tree());
        let matrix = matricization_of(&matricizations, sector)?;
        let factor = factors
            .iter()
            .find(|factor| factor.sector == sector)
            .expect("factor exists for every matricized sector");
        let (col_offset, _) = col_placement(matrix, key.domain_tree())?;
        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        scatter_matrix_block(
            vt_tensor.data_mut(),
            &shape,
            &strides,
            offset,
            0,
            &factor.vt,
            factor.rank,
            col_offset,
        );
    }

    Ok((
        FusionSvd {
            u: u_tensor,
            singular_values,
            vt: vt_tensor,
        },
        truncation_error,
    ))
}

/// Copies a dense column-major matrix region into one fusion-tree subblock.
///
/// `matrix_axis` names the block axis that walks the matrix's own leading
/// dimension side; the remaining axes enumerate the offset side column-major.
/// For `U` the matrix axis is the trailing (new leg) axis and the codomain
/// axes select rows at `side_offset`; for `Vt` the matrix axis is the leading
/// (new leg) axis and the domain axes select columns at `side_offset`.
#[allow(clippy::too_many_arguments)]
fn scatter_matrix_block(
    data: &mut [f64],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    matrix_axis: usize,
    matrix: &[f64],
    matrix_rows: usize,
    side_offset: usize,
) {
    let rank = shape.len();
    let mut index = vec![0usize; rank];
    let count: usize = shape.iter().product();
    for _ in 0..count {
        let mut position = offset;
        let mut side = 0usize;
        let mut side_stride = 1usize;
        let mut matrix_index = 0usize;
        for axis in 0..rank {
            position += index[axis] * strides[axis];
            if axis == matrix_axis {
                matrix_index = index[axis];
            } else {
                side += index[axis] * side_stride;
                side_stride *= shape[axis];
            }
        }
        let (row, col) = if matrix_axis == rank - 1 {
            (side_offset + side, matrix_index)
        } else {
            (matrix_index, side_offset + side)
        };
        data[position] = matrix[row + matrix_rows * col];
        for axis in 0..rank {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

fn coupled_of<R>(rule: &R, tree: &FusionTreeKey) -> SectorId
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    tree.coupled().unwrap_or_else(|| rule.vacuum())
}

fn matricization_of(
    matricizations: &[SectorMatricization],
    sector: SectorId,
) -> Result<&SectorMatricization, OperationError> {
    matricizations
        .iter()
        .find(|matrix| matrix.sector == sector)
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: "factor tree references a coupled sector absent from the source tensor",
        })
}

fn row_placement<'a>(
    matrix: &'a SectorMatricization,
    tree: &FusionTreeKey,
) -> Result<(usize, &'a [usize]), OperationError> {
    matrix
        .row_trees
        .iter()
        .find(|(candidate, _, _)| candidate == tree)
        .map(|(_, offset, shape)| (*offset, shape.as_slice()))
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: "factor codomain tree absent from the source matricization",
        })
}

fn col_placement<'a>(
    matrix: &'a SectorMatricization,
    tree: &FusionTreeKey,
) -> Result<(usize, &'a [usize]), OperationError> {
    matrix
        .col_trees
        .iter()
        .find(|(candidate, _, _)| candidate == tree)
        .map(|(_, offset, shape)| (*offset, shape.as_slice()))
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: "factor domain tree absent from the source matricization",
        })
}

fn row_shape_of(
    matricizations: &[SectorMatricization],
    sector: SectorId,
    tree: &FusionTreeKey,
) -> Result<Vec<usize>, OperationError> {
    row_placement(matricization_of(matricizations, sector)?, tree).map(|(_, shape)| shape.to_vec())
}

fn col_shape_of(
    matricizations: &[SectorMatricization],
    sector: SectorId,
    tree: &FusionTreeKey,
) -> Result<Vec<usize>, OperationError> {
    col_placement(matricization_of(matricizations, sector)?, tree).map(|(_, shape)| shape.to_vec())
}

fn validate_dense_shape(actual: &[usize], expected: &[usize]) -> Result<(), OperationError> {
    if actual != expected {
        return Err(OperationError::ShapeMismatch {
            dst: expected.to_vec(),
            src: actual.to_vec(),
        });
    }
    Ok(())
}

/// Packs every coupled sector of `tensor` into its dense column-major
/// matricization, independent of the tensor's storage layout.
fn sector_matricizations<R, const NOUT: usize, const NIN: usize>(
    rule: &R,
    tensor: &TensorMap<f64, NOUT, NIN>,
    nout: usize,
) -> Result<Vec<SectorMatricization>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let structure = std::sync::Arc::clone(tensor.structure());
    let mut matricizations: Vec<SectorMatricization> = Vec::new();

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
        let sector = coupled_of(rule, key.codomain_tree());
        let row_dim: usize = block.shape()[..nout].iter().product();
        let col_dim: usize = block.shape()[nout..].iter().product();
        let matrix = match matricizations
            .iter_mut()
            .find(|matrix| matrix.sector == sector)
        {
            Some(matrix) => matrix,
            None => {
                matricizations.push(SectorMatricization {
                    sector,
                    rows: 0,
                    cols: 0,
                    row_trees: Vec::new(),
                    col_trees: Vec::new(),
                    data: Vec::new(),
                });
                matricizations.last_mut().expect("just pushed")
            }
        };
        if !matrix
            .row_trees
            .iter()
            .any(|(tree, _, _)| tree == key.codomain_tree())
        {
            matrix.row_trees.push((
                key.codomain_tree().clone(),
                matrix.rows,
                block.shape()[..nout].to_vec(),
            ));
            matrix.rows += row_dim;
        }
        if !matrix
            .col_trees
            .iter()
            .any(|(tree, _, _)| tree == key.domain_tree())
        {
            matrix.col_trees.push((
                key.domain_tree().clone(),
                matrix.cols,
                block.shape()[nout..].to_vec(),
            ));
            matrix.cols += col_dim;
        }
    }
    for matrix in &mut matricizations {
        matrix.data = vec![0.0; matrix.rows * matrix.cols];
    }

    let data = tensor.data();
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = coupled_of(rule, key.codomain_tree());
        let matrix = matricizations
            .iter_mut()
            .find(|matrix| matrix.sector == sector)
            .expect("matricization registered in first pass");
        let row_offset = matrix
            .row_trees
            .iter()
            .find(|(tree, _, _)| tree == key.codomain_tree())
            .map(|(_, offset, _)| *offset)
            .expect("row tree registered in first pass");
        let col_offset = matrix
            .col_trees
            .iter()
            .find(|(tree, _, _)| tree == key.domain_tree())
            .map(|(_, offset, _)| *offset)
            .expect("column tree registered in first pass");

        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        let rank = shape.len();
        let count: usize = shape.iter().product();
        let mut index = vec![0usize; rank];
        let rows = matrix.rows;
        for _ in 0..count {
            let mut position = offset;
            let mut row = 0usize;
            let mut row_stride = 1usize;
            let mut col = 0usize;
            let mut col_stride = 1usize;
            for axis in 0..rank {
                position += index[axis] * strides[axis];
                if axis < nout {
                    row += index[axis] * row_stride;
                    row_stride *= shape[axis];
                } else {
                    col += index[axis] * col_stride;
                    col_stride *= shape[axis];
                }
            }
            matrix.data[(row_offset + row) + rows * (col_offset + col)] = data[position];
            for axis in 0..rank {
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
    }
    Ok(matricizations)
}
