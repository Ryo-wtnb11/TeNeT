use tenet_core::{
    BlockKey, CoreError, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace,
    FusionTreeKey, MultiplicityFreeRigidSymbols, SectorId, SectorLeg, TensorMap, TensorMapSpace,
};
use tenet_dense::{DenseExecutor, DenseRead, DenseView};

use crate::truncation::{select_truncation, Truncation, WeightedSpectrum};
use crate::OperationError;

/// One coupled sector's factorization spectrum: singular values (descending)
/// or eigenvalues (signed, stored descending by magnitude).
#[derive(Clone, Debug, PartialEq)]
pub struct SectorSpectrum {
    pub sector: SectorId,
    pub values: Vec<f64>,
}

/// Compact fusion-tensor SVD `t = U * S * Vh` (MatrixAlgebraKit `svd_compact`).
///
/// The factorization acts blockwise on the coupled-sector matricization
/// through the placement-capable [`DenseExecutor`] boundary; the truncation
/// decision is a host-side scalar selection over the per-sector spectra
/// (see [`crate::truncation`]), applied as a leading-columns/rows gather.
/// `U : codomain <- W`, `S : W <- W` diagonal, `Vh : W <- domain`; `error` is
/// the quantum-dimension-weighted 2-norm of the discarded values.
#[derive(Clone, Debug)]
pub struct SvdCompact<const NOUT: usize, const NIN: usize> {
    pub u: TensorMap<f64, NOUT, 1>,
    pub s: TensorMap<f64, 1, 1>,
    pub vh: TensorMap<f64, 1, NIN>,
    pub singular_values: Vec<SectorSpectrum>,
    pub error: f64,
}

/// Full (untruncated) fusion-tensor SVD `t = U * S * Vh`.
///
/// This is the pure device-boundary factorization: the dense per-sector SVDs
/// run through the [`DenseExecutor`] and no truncation logic is involved.
/// Per block it is the economy factorization with bond `min(rows, cols)`.
#[derive(Clone, Debug)]
pub struct SvdFull<const NOUT: usize, const NIN: usize> {
    pub u: TensorMap<f64, NOUT, 1>,
    pub s: TensorMap<f64, 1, 1>,
    pub vh: TensorMap<f64, 1, NIN>,
    pub singular_values: Vec<SectorSpectrum>,
}

/// Materializes per-sector spectra as a diagonal tensor `W <- W` in the
/// coupled layout (`S` for the SVD, `D` for eigendecompositions).
fn diagonal_bond_tensor<R>(
    rule: &R,
    singular_values: &[SectorSpectrum],
) -> Result<TensorMap<f64, 1, 1>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let new_leg = SectorLeg::new(singular_values.iter().map(|entry| entry.sector), false);
    let total_dim: usize = singular_values.iter().map(|entry| entry.values.len()).sum();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg.clone()]),
        FusionProductSpace::new([new_leg]),
    );
    let keys = homspace.fusion_tree_keys(rule);
    let shapes = keys
        .iter()
        .map(|key| {
            let sector = coupled_of(rule, key.codomain_tree());
            let count = singular_values
                .iter()
                .find(|entry| entry.sector == sector)
                .map(|entry| entry.values.len())
                .unwrap_or(0);
            vec![count, count]
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([total_dim], [total_dim])
            .map_err(OperationError::from_core_preserving_context)?,
        homspace,
        rule,
        shapes,
    )
    .map_err(OperationError::from_core_preserving_context)?;
    TensorMap::<f64, 1, 1>::from_block_fn_with_fusion_space(space, 0.0, |key, indices| {
        if indices[0] != indices[1] {
            return 0.0;
        }
        let BlockKey::FusionTree(tree) = key else {
            return 0.0;
        };
        let sector = tree
            .codomain_tree()
            .coupled()
            .unwrap_or_else(|| tree.codomain_tree().uncoupled()[0]);
        singular_values
            .iter()
            .find(|entry| entry.sector == sector)
            .map(|entry| entry.values[indices[0]])
            .unwrap_or(0.0)
    })
    .map_err(OperationError::from_core_preserving_context)
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

/// All singular values per coupled sector, descending (MatrixAlgebraKit
/// `svd_vals`). Runs the dense SVD per sector through the executor and keeps
/// only the spectra.
pub fn svd_vals<E, R, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<f64, NOUT, NIN>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    svd_full(dense, rule, tensor).map(|svd| svd.singular_values)
}

/// Compact fusion-tensor SVD with an in-line truncation policy.
///
/// Layering: the untruncated factorization runs on the device boundary
/// ([`svd_full`]); the truncation decision is host-side scalar work over the
/// spectra and its application slices the leading bond states per sector
/// ([`truncate_svd`]).
pub fn svd_compact<E, R, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<f64, NOUT, NIN>,
    truncation: &Truncation,
) -> Result<SvdCompact<NOUT, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let full = svd_full(dense, rule, tensor)?;
    truncate_svd(rule, full, truncation)
}

/// Full (untruncated) fusion-tensor SVD through the device boundary.
pub fn svd_full<E, R, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<f64, NOUT, NIN>,
) -> Result<SvdFull<NOUT, NIN>, OperationError>
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

        singular_values.push(SectorSpectrum {
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

    let pairs = factors
        .into_iter()
        .map(|factor| FactorPair {
            sector: factor.sector,
            kept: factor.kept,
            left: factor.u,
            left_rows: factor.rows,
            right: factor.vt,
            right_leading: factor.rank,
        })
        .collect::<Vec<_>>();
    let (u_tensor, vt_tensor) = build_left_right_pair(
        rule,
        &fusion_space,
        tensor.space().dims(),
        &matricizations,
        &pairs,
    )?;

    let s_tensor = diagonal_bond_tensor(rule, &singular_values)?;
    Ok(SvdFull {
        u: u_tensor,
        s: s_tensor,
        vh: vt_tensor,
        singular_values,
    })
}

/// Host-side truncation decision shared by every bond factorization: the
/// selection magnitude is `|value|` and each `spectra` entry is stored
/// descending by magnitude (the `*_full` output contract), so the kept set is
/// always a per-sector prefix.
fn decide_bond_truncation<R>(
    rule: &R,
    spectra: &[SectorSpectrum],
    truncation: &Truncation,
) -> crate::truncation::TruncationDecision
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let magnitudes: Vec<Vec<f64>> = spectra
        .iter()
        .map(|entry| entry.values.iter().map(|value| value.abs()).collect())
        .collect();
    let weighted: Vec<WeightedSpectrum<'_>> = spectra
        .iter()
        .zip(&magnitudes)
        .map(|(entry, values)| WeightedSpectrum {
            weight: rule.dim_scalar(entry.sector),
            values,
        })
        .collect();
    select_truncation(&weighted, truncation)
}

/// Applies a truncation policy to a full factorization (the host half of
/// [`svd_compact`]).
///
/// The decision is host-side scalar work over the spectra; the application
/// keeps the leading bond states per coupled sector, which in the coupled
/// layout is a per-sector leading-columns/rows copy (device kernel later).
pub(crate) fn truncate_svd<R, const NOUT: usize, const NIN: usize>(
    rule: &R,
    full: SvdFull<NOUT, NIN>,
    truncation: &Truncation,
) -> Result<SvdCompact<NOUT, NIN>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let decision = decide_bond_truncation(rule, &full.singular_values, truncation);
    if full
        .singular_values
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok(SvdCompact {
            u: full.u,
            s: full.s,
            vh: full.vh,
            singular_values: full.singular_values,
            error: 0.0,
        });
    }

    let mut singular_values = full.singular_values;
    for (entry, &count) in singular_values.iter_mut().zip(&decision.kept) {
        entry.values.truncate(count);
    }
    singular_values.retain(|entry| !entry.values.is_empty());

    let kept_of = |sector: SectorId| -> usize {
        singular_values
            .iter()
            .find(|entry| entry.sector == sector)
            .map(|entry| entry.values.len())
            .unwrap_or(0)
    };

    let u_tensor = sliced_bond_tensor(rule, &full.u, NOUT, &kept_of)?;
    let vh_tensor = sliced_bond_tensor(rule, &full.vh, 0, &kept_of)?;
    let s_tensor = diagonal_bond_tensor(rule, &singular_values)?;
    Ok(SvdCompact {
        u: u_tensor,
        s: s_tensor,
        vh: vh_tensor,
        singular_values,
        error: decision.error,
    })
}

/// Rebuilds a factor tensor with the bond leg (`axis`) shrunk to the kept
/// prefix per coupled sector, copying leading bond states blockwise.
fn sliced_bond_tensor<R, const NOUT: usize, const NIN: usize>(
    rule: &R,
    source: &TensorMap<f64, NOUT, NIN>,
    axis: usize,
    kept_of: &dyn Fn(SectorId) -> usize,
) -> Result<TensorMap<f64, NOUT, NIN>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let source_space = source
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    let source_structure = std::sync::Arc::clone(source.structure());

    // The bond leg carries exactly the kept sectors.
    let kept_sectors: Vec<SectorId> = {
        let homspace = source_space.homspace();
        let leg = if axis < NOUT {
            &homspace.codomain().legs()[axis]
        } else {
            &homspace.domain().legs()[axis - NOUT]
        };
        leg.sectors()
            .iter()
            .copied()
            .filter(|&sector| kept_of(sector) > 0)
            .collect()
    };
    let bond_leg = SectorLeg::new(kept_sectors.iter().copied(), false);
    let homspace = source_space.homspace();
    let new_hom = if axis < NOUT {
        let mut codomain_legs = homspace.codomain().legs().to_vec();
        codomain_legs[axis] = bond_leg;
        FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain_legs),
            homspace.domain().clone(),
        )
    } else {
        let mut domain_legs = homspace.domain().legs().to_vec();
        domain_legs[axis - NOUT] = bond_leg;
        FusionTreeHomSpace::new(
            homspace.codomain().clone(),
            FusionProductSpace::new(domain_legs),
        )
    };

    let keys = new_hom.fusion_tree_keys(rule);
    let shapes = keys
        .iter()
        .map(|key| {
            let old_index = source_structure
                .find_block_index_by_key(&BlockKey::FusionTree(key.clone()))
                .ok_or(OperationError::UnsupportedTensorContractScope {
                    message: "truncated factor tree must exist in the full factor",
                })?;
            let old_block = source_structure
                .block(old_index)
                .map_err(OperationError::from_core_preserving_context)?;
            let mut shape = old_block.shape().to_vec();
            let bond_tree = if axis < NOUT {
                key.codomain_tree()
            } else {
                key.domain_tree()
            };
            shape[axis] = kept_of(coupled_of(rule, bond_tree));
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;

    let mut dims = source.space().dims().to_vec();
    dims[axis] = kept_sectors.iter().map(|&sector| kept_of(sector)).sum();
    let mut codomain_dims = [0usize; NOUT];
    codomain_dims.copy_from_slice(&dims[..NOUT]);
    let mut domain_dims = [0usize; NIN];
    domain_dims.copy_from_slice(&dims[NOUT..]);
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<NOUT, NIN>::from_dims(codomain_dims, domain_dims)
            .map_err(OperationError::from_core_preserving_context)?,
        new_hom,
        rule,
        shapes,
    )
    .map_err(OperationError::from_core_preserving_context)?;
    let len = space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut sliced = TensorMap::<f64, NOUT, NIN>::from_vec_with_fusion_space(vec![0.0; len], space)
        .map_err(OperationError::from_core_preserving_context)?;

    let sliced_structure = std::sync::Arc::clone(sliced.structure());
    for index in 0..sliced_structure.block_count() {
        let new_block = sliced_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let key = new_block.key().clone();
        let old_index = source_structure.find_block_index_by_key(&key).ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: "truncated factor tree must exist in the full factor",
            },
        )?;
        let old_block = source_structure
            .block(old_index)
            .map_err(OperationError::from_core_preserving_context)?;
        let shape = new_block.shape().to_vec();
        let new_strides = new_block.strides().to_vec();
        let new_offset = new_block.offset();
        let old_strides = old_block.strides().to_vec();
        let old_offset = old_block.offset();
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        let source_data = source.data();
        let data = sliced.data_mut();
        for _ in 0..count {
            let new_position = new_offset
                + indices
                    .iter()
                    .zip(&new_strides)
                    .map(|(&i, &stride)| i * stride)
                    .sum::<usize>();
            let old_position = old_offset
                + indices
                    .iter()
                    .zip(&old_strides)
                    .map(|(&i, &stride)| i * stride)
                    .sum::<usize>();
            data[new_position] = source_data[old_position];
            for axis_index in 0..shape.len() {
                indices[axis_index] += 1;
                if indices[axis_index] < shape[axis_index] {
                    break;
                }
                indices[axis_index] = 0;
            }
        }
    }
    Ok(sliced)
}

/// One coupled sector's factor pair: `left` is `left_rows x kept` (leading
/// columns of a column-major matrix), `right` is `kept x cols` (leading rows
/// of a column-major matrix with leading dimension `right_leading`).
struct FactorPair {
    sector: SectorId,
    kept: usize,
    left: Vec<f64>,
    left_rows: usize,
    right: Vec<f64>,
    right_leading: usize,
}

/// Builds the `(codomain <- W, W <- domain)` tensor pair shared by SVD and
/// the orthogonal factorizations, in the coupled-sector matrix layout.
fn build_left_right_pair<R, const NOUT: usize, const NIN: usize>(
    rule: &R,
    fusion_space: &FusionTensorMapSpace<NOUT, NIN>,
    dims: &[usize],
    matricizations: &[SectorMatricization],
    pairs: &[FactorPair],
) -> Result<(TensorMap<f64, NOUT, 1>, TensorMap<f64, 1, NIN>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let mut new_leg_dim = 0usize;
    for pair in pairs {
        new_leg_dim += pair.kept;
    }
    let sector_rank = |sector: SectorId| -> usize {
        pairs
            .iter()
            .find(|pair| pair.sector == sector)
            .map(|pair| pair.kept)
            .unwrap_or(0)
    };

    let new_leg = SectorLeg::new(pairs.iter().map(|pair| pair.sector), false);
    let mut codomain_dims = [0usize; NOUT];
    codomain_dims.copy_from_slice(&dims[..NOUT]);
    let mut domain_dims = [0usize; NIN];
    domain_dims.copy_from_slice(&dims[NOUT..]);

    let left_hom = FusionTreeHomSpace::new(
        fusion_space.homspace().codomain().clone(),
        FusionProductSpace::new([new_leg.clone()]),
    );
    let left_keys = left_hom.fusion_tree_keys(rule);
    let left_shapes = left_keys
        .iter()
        .map(|key| {
            let sector = coupled_of(rule, key.codomain_tree());
            let mut shape = row_shape_of(matricizations, sector, key.codomain_tree())?;
            shape.push(sector_rank(sector));
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    let left_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<NOUT, 1>::from_dims(codomain_dims, [new_leg_dim])
            .map_err(OperationError::from_core_preserving_context)?,
        left_hom,
        rule,
        left_shapes,
    )
    .map_err(OperationError::from_core_preserving_context)?;

    let right_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg]),
        fusion_space.homspace().domain().clone(),
    );
    let right_keys = right_hom.fusion_tree_keys(rule);
    let right_shapes = right_keys
        .iter()
        .map(|key| {
            let sector = coupled_of(rule, key.domain_tree());
            let mut shape = vec![sector_rank(sector)];
            shape.extend(col_shape_of(matricizations, sector, key.domain_tree())?);
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    let right_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, NIN>::from_dims([new_leg_dim], domain_dims)
            .map_err(OperationError::from_core_preserving_context)?,
        right_hom,
        rule,
        right_shapes,
    )
    .map_err(OperationError::from_core_preserving_context)?;

    let left_len = left_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut left_tensor =
        TensorMap::<f64, NOUT, 1>::from_vec_with_fusion_space(vec![0.0; left_len], left_space)
            .map_err(OperationError::from_core_preserving_context)?;
    let right_len = right_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut right_tensor =
        TensorMap::<f64, 1, NIN>::from_vec_with_fusion_space(vec![0.0; right_len], right_space)
            .map_err(OperationError::from_core_preserving_context)?;

    // Scatter left blocks: element (i.., j) = left[(row_offset + rowmaj(i)) + left_rows * j].
    let left_structure = std::sync::Arc::clone(left_tensor.structure());
    for index in 0..left_structure.block_count() {
        let block = left_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = coupled_of(rule, key.codomain_tree());
        let matrix = matricization_of(matricizations, sector)?;
        let pair = pairs
            .iter()
            .find(|pair| pair.sector == sector)
            .expect("factor pair exists for every matricized sector");
        let (row_offset, _) = row_placement(matrix, key.codomain_tree())?;
        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        scatter_matrix_block(
            left_tensor.data_mut(),
            &shape,
            &strides,
            offset,
            shape.len() - 1,
            &pair.left,
            pair.left_rows,
            row_offset,
        );
    }

    // Scatter right blocks: element (r, j..) = right[r + right_leading * (col_offset + colmaj(j))].
    let right_structure = std::sync::Arc::clone(right_tensor.structure());
    for index in 0..right_structure.block_count() {
        let block = right_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = coupled_of(rule, key.domain_tree());
        let matrix = matricization_of(matricizations, sector)?;
        let pair = pairs
            .iter()
            .find(|pair| pair.sector == sector)
            .expect("factor pair exists for every matricized sector");
        let (col_offset, _) = col_placement(matrix, key.domain_tree())?;
        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        scatter_matrix_block(
            right_tensor.data_mut(),
            &shape,
            &strides,
            offset,
            0,
            &pair.right,
            pair.right_leading,
            col_offset,
        );
    }

    Ok((left_tensor, right_tensor))
}

/// Full (untruncated) Hermitian eigendecomposition `t = V * D * Vh`.
///
/// Requires an endomorphism (`codomain == domain`) with Hermitian coupled
/// blocks. Bond states are stored descending by `|eigenvalue|` per sector
/// (the shared `*_full` contract that makes truncation a prefix rule);
/// `eigenvalues` keeps the signed values in that order and `D : W <- W` is
/// their diagonal tensor.
#[derive(Clone, Debug)]
pub struct EighFull<const NOUT: usize, const NIN: usize> {
    pub d: TensorMap<f64, 1, 1>,
    pub v: TensorMap<f64, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum>,
}

/// Truncated Hermitian eigendecomposition; `error` is the
/// quantum-dimension-weighted 2-norm of the discarded eigenvalues.
#[derive(Clone, Debug)]
pub struct EighCompact<const NOUT: usize, const NIN: usize> {
    pub d: TensorMap<f64, 1, 1>,
    pub v: TensorMap<f64, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum>,
    pub error: f64,
}

/// Full Hermitian eigendecomposition through the device boundary.
pub fn eigh_full<E, R, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<f64, NOUT, NIN>,
) -> Result<EighFull<NOUT, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?
        .clone();
    if fusion_space.homspace().codomain() != fusion_space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires an endomorphism (codomain == domain)",
        });
    }
    let matricizations = sector_matricizations(rule, tensor, NOUT)?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(&matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let outputs = dense
            .eigh(DenseRead::F64(view))
            .map_err(OperationError::Dense)?;
        if outputs.len() != 2 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "dense eigh must return exactly (values, vectors)",
            });
        }
        let n = matrix.rows;
        validate_dense_shape(outputs[0].shape(), &[n])?;
        validate_dense_shape(outputs[1].shape(), &[n, n])?;
        let values = outputs[0].as_f64_slice().map_err(OperationError::Dense)?;
        let vectors = outputs[1].as_f64_slice().map_err(OperationError::Dense)?;

        // Reorder bond states descending by |eigenvalue| (stable on ties).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            values[b]
                .abs()
                .partial_cmp(&values[a].abs())
                .expect("finite eigenvalues")
                .then(a.cmp(&b))
        });
        let sorted_values: Vec<f64> = order.iter().map(|&index| values[index]).collect();
        let mut sorted_vectors = vec![0.0; n * n];
        for (position, &index) in order.iter().enumerate() {
            sorted_vectors[position * n..(position + 1) * n]
                .copy_from_slice(&vectors[index * n..(index + 1) * n]);
        }

        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: n,
            right: transpose_col_major(&sorted_vectors, n, n),
            left: sorted_vectors,
            left_rows: n,
            right_leading: n,
        });
    }

    let (v_tensor, _vh_tensor) = build_left_right_pair(
        rule,
        &fusion_space,
        tensor.space().dims(),
        &matricizations,
        &pairs,
    )?;
    let d_tensor = diagonal_bond_tensor(rule, &eigenvalues)?;
    Ok(EighFull {
        d: d_tensor,
        v: v_tensor,
        eigenvalues,
    })
}

/// Truncated Hermitian eigendecomposition: [`eigh_full`] on the device
/// boundary plus the shared host-side truncation by `|eigenvalue|`.
pub fn eigh_compact<E, R, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<f64, NOUT, NIN>,
    truncation: &Truncation,
) -> Result<EighCompact<NOUT, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let full = eigh_full(dense, rule, tensor)?;
    let decision = decide_bond_truncation(rule, &full.eigenvalues, truncation);
    if full
        .eigenvalues
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok(EighCompact {
            d: full.d,
            v: full.v,
            eigenvalues: full.eigenvalues,
            error: 0.0,
        });
    }
    let mut eigenvalues = full.eigenvalues;
    for (entry, &count) in eigenvalues.iter_mut().zip(&decision.kept) {
        entry.values.truncate(count);
    }
    eigenvalues.retain(|entry| !entry.values.is_empty());
    let kept_of = |sector: SectorId| -> usize {
        eigenvalues
            .iter()
            .find(|entry| entry.sector == sector)
            .map(|entry| entry.values.len())
            .unwrap_or(0)
    };
    let v_tensor = sliced_bond_tensor(rule, &full.v, NOUT, &kept_of)?;
    let d_tensor = diagonal_bond_tensor(rule, &eigenvalues)?;
    Ok(EighCompact {
        d: d_tensor,
        v: v_tensor,
        eigenvalues,
        error: decision.error,
    })
}

/// Compact QR `t = Q * R` (MatrixAlgebraKit `qr_compact`):
/// `Q : codomain <- W` has orthonormal columns per coupled sector and
/// `R : W <- domain` with per-sector bond `min(rows, cols)`.
pub fn qr_compact<E, R, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<f64, NOUT, NIN>,
) -> Result<(TensorMap<f64, NOUT, 1>, TensorMap<f64, 1, NIN>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?
        .clone();
    let matricizations = sector_matricizations(rule, tensor, NOUT)?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(&matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let outputs = dense
            .qr(DenseRead::F64(view))
            .map_err(OperationError::Dense)?;
        if outputs.len() != 2 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "dense QR must return exactly (Q, R)",
            });
        }
        let rank = matrix.rows.min(matrix.cols);
        validate_dense_shape(outputs[0].shape(), &[matrix.rows, rank])?;
        validate_dense_shape(outputs[1].shape(), &[rank, matrix.cols])?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rank,
            left: outputs[0]
                .as_f64_slice()
                .map_err(OperationError::Dense)?
                .to_vec(),
            left_rows: matrix.rows,
            right: outputs[1]
                .as_f64_slice()
                .map_err(OperationError::Dense)?
                .to_vec(),
            right_leading: rank,
        });
    }

    build_left_right_pair(
        rule,
        &fusion_space,
        tensor.space().dims(),
        &matricizations,
        &pairs,
    )
}

/// Compact LQ `t = L * Q` (MatrixAlgebraKit `lq_compact`, via the QR of the
/// transposed sector matrices): `Q : W <- domain` has orthonormal rows per
/// coupled sector and `L : codomain <- W`.
pub fn lq_compact<E, R, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<f64, NOUT, NIN>,
) -> Result<(TensorMap<f64, NOUT, 1>, TensorMap<f64, 1, NIN>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?
        .clone();
    let matricizations = sector_matricizations(rule, tensor, NOUT)?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        // QR of the transpose: t^T = Q' R'  =>  t = R'^T Q'^T = L Q.
        let transposed = transpose_col_major(&matrix.data, matrix.rows, matrix.cols);
        let shape = [matrix.cols, matrix.rows];
        let strides = [1usize, matrix.cols];
        let view =
            DenseView::new(&transposed, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let outputs = dense
            .qr(DenseRead::F64(view))
            .map_err(OperationError::Dense)?;
        if outputs.len() != 2 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "dense QR must return exactly (Q, R)",
            });
        }
        let rank = matrix.rows.min(matrix.cols);
        validate_dense_shape(outputs[0].shape(), &[matrix.cols, rank])?;
        validate_dense_shape(outputs[1].shape(), &[rank, matrix.rows])?;
        let q_prime = outputs[0].as_f64_slice().map_err(OperationError::Dense)?;
        let r_prime = outputs[1].as_f64_slice().map_err(OperationError::Dense)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rank,
            // L = R'^T : rows x rank.
            left: transpose_col_major(r_prime, rank, matrix.rows),
            left_rows: matrix.rows,
            // Q = Q'^T : rank x cols.
            right: transpose_col_major(q_prime, matrix.cols, rank),
            right_leading: rank,
        });
    }

    build_left_right_pair(
        rule,
        &fusion_space,
        tensor.space().dims(),
        &matricizations,
        &pairs,
    )
}

/// Transposes a column-major `rows x cols` matrix into column-major
/// `cols x rows`.
fn transpose_col_major(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut transposed = vec![0.0; data.len()];
    for col in 0..cols {
        for row in 0..rows {
            transposed[col + cols * row] = data[row + rows * col];
        }
    }
    transposed
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
