use num_complex::Complex64;
use tenet_core::{
    BlockKey, CoreError, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace,
    FusionTreeKey, MultiplicityFreeRigidSymbols, SectorId, SectorLeg, TensorMap, TensorMapSpace,
};
use tenet_dense::{DenseError, DenseExecutor, DenseTensor, DenseView};

use tenet_tensors::DenseRecouplingScalar;

use crate::truncation::{select_truncation, Truncation, WeightedSpectrum};
use tenet_tensors::OperationError;

/// Scalar contract for the factorization layer: dense-executor I/O plus the
/// adjoint and real-embedding used by the factor builders. Implemented for
/// the double-precision real and complex scalars.
pub trait FactorScalar: DenseRecouplingScalar {
    /// Output scalar of the general (non-Hermitian) eigendecomposition.
    type Eig: FactorScalar;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError>;
    /// Real spectrum output (singular values, Hermitian eigenvalues) widened
    /// to `f64` for the host-side truncation policies.
    fn real_spectrum(tensor: &DenseTensor) -> Result<Vec<f64>, DenseError>;
    fn from_real(value: f64) -> Self;
    /// Widens to `Complex64` (general eigenvalue bookkeeping).
    fn widen_complex(self) -> Complex64;
    /// Narrows from `Complex64` (lossy for the single-precision scalars).
    fn from_complex64(value: Complex64) -> Self;
    fn adjoint(self) -> Self;
}

impl FactorScalar for f32 {
    type Eig = num_complex::Complex32;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
        tensor.as_f32_slice()
    }

    fn real_spectrum(tensor: &DenseTensor) -> Result<Vec<f64>, DenseError> {
        Ok(tensor
            .as_f32_slice()?
            .iter()
            .map(|&value| value as f64)
            .collect())
    }

    fn from_real(value: f64) -> Self {
        value as f32
    }

    fn widen_complex(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }

    fn from_complex64(value: Complex64) -> Self {
        value.re as f32
    }

    fn adjoint(self) -> Self {
        self
    }
}

impl FactorScalar for f64 {
    type Eig = Complex64;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
        tensor.as_f64_slice()
    }

    fn real_spectrum(tensor: &DenseTensor) -> Result<Vec<f64>, DenseError> {
        Ok(tensor.as_f64_slice()?.to_vec())
    }

    fn from_real(value: f64) -> Self {
        value
    }

    fn widen_complex(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }

    fn from_complex64(value: Complex64) -> Self {
        value.re
    }

    fn adjoint(self) -> Self {
        self
    }
}

impl FactorScalar for num_complex::Complex32 {
    type Eig = num_complex::Complex32;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
        tensor.as_c32_slice()
    }

    fn real_spectrum(tensor: &DenseTensor) -> Result<Vec<f64>, DenseError> {
        Ok(tensor
            .as_f32_slice()?
            .iter()
            .map(|&value| value as f64)
            .collect())
    }

    fn from_real(value: f64) -> Self {
        num_complex::Complex32::new(value as f32, 0.0)
    }

    fn widen_complex(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }

    fn from_complex64(value: Complex64) -> Self {
        num_complex::Complex32::new(value.re as f32, value.im as f32)
    }

    fn adjoint(self) -> Self {
        self.conj()
    }
}

impl FactorScalar for Complex64 {
    type Eig = Complex64;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
        tensor.as_c64_slice()
    }

    fn real_spectrum(tensor: &DenseTensor) -> Result<Vec<f64>, DenseError> {
        Ok(tensor.as_f64_slice()?.to_vec())
    }

    fn from_real(value: f64) -> Self {
        Complex64::new(value, 0.0)
    }

    fn widen_complex(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }

    fn adjoint(self) -> Self {
        self.conj()
    }
}

/// Magnitude used by the truncation selection over a spectrum.
pub trait SpectrumMagnitude: Copy {
    fn magnitude(self) -> f64;
}

impl SpectrumMagnitude for f64 {
    fn magnitude(self) -> f64 {
        self.abs()
    }
}

impl SpectrumMagnitude for Complex64 {
    fn magnitude(self) -> f64 {
        self.norm()
    }
}

/// One coupled sector's factorization spectrum, stored descending by
/// magnitude: singular values (`f64`), Hermitian eigenvalues (signed `f64`),
/// or general eigenvalues (`Complex64`).
#[derive(Clone, Debug, PartialEq)]
pub struct SectorSpectrum<V = f64> {
    pub sector: SectorId,
    pub values: Vec<V>,
}

/// Truncated fusion-tensor SVD `t ~ U * S * Vh` (MatrixAlgebraKit `svd_trunc`).
///
/// The factorization acts blockwise on the coupled-sector matricization
/// through the placement-capable [`DenseExecutor`] boundary; the truncation
/// decision is a host-side scalar selection over the per-sector spectra
/// (see [`crate::truncation`]), applied as a leading-columns/rows gather.
/// `U : codomain <- W`, `S : W <- W` diagonal, `Vh : W <- domain`; `error` is
/// the quantum-dimension-weighted 2-norm of the discarded values.
#[derive(Clone, Debug)]
pub struct SvdTrunc<D, const NOUT: usize, const NIN: usize> {
    pub u: TensorMap<D, NOUT, 1>,
    pub s: TensorMap<D, 1, 1>,
    pub vh: TensorMap<D, 1, NIN>,
    pub singular_values: Vec<SectorSpectrum>,
    pub error: f64,
}

/// Compact (thin, untruncated) fusion-tensor SVD `t = U * S * Vh`
/// (MatrixAlgebraKit `svd_compact`).
///
/// This is the pure device-boundary factorization: the dense per-sector SVDs
/// run through the [`DenseExecutor`] and no truncation logic is involved.
/// Per block the bond is `min(rows, cols)`; the square-`U` variant is
/// MatrixAlgebraKit `svd_full` (later batch).
#[derive(Clone, Debug)]
pub struct SvdCompact<D, const NOUT: usize, const NIN: usize> {
    pub u: TensorMap<D, NOUT, 1>,
    pub s: TensorMap<D, 1, 1>,
    pub vh: TensorMap<D, 1, NIN>,
    pub singular_values: Vec<SectorSpectrum>,
}

/// Materializes per-sector spectra as a diagonal tensor `W <- W` in the
/// coupled layout (`S` for the SVD, `D` for eigendecompositions).
pub(crate) fn diagonal_bond_tensor<R, D, V>(
    rule: &R,
    singular_values: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<TensorMap<D, 1, 1>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
    V: Copy,
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
    TensorMap::<D, 1, 1>::from_block_fn_with_fusion_space(space, D::zero(), |key, indices| {
        if indices[0] != indices[1] {
            return D::zero();
        }
        let BlockKey::FusionTree(tree) = key else {
            return D::zero();
        };
        let sector = tree
            .codomain_tree()
            .coupled()
            .unwrap_or_else(|| tree.codomain_tree().uncoupled()[0]);
        singular_values
            .iter()
            .find(|entry| entry.sector == sector)
            .map(|entry| to_scalar(entry.values[indices[0]]))
            .unwrap_or_else(D::zero)
    })
    .map_err(OperationError::from_core_preserving_context)
}

struct SectorMatricization<D> {
    sector: SectorId,
    rows: usize,
    cols: usize,
    /// (codomain tree, row offset, codomain degeneracy shape)
    row_trees: Vec<(FusionTreeKey, usize, Vec<usize>)>,
    /// (domain tree, column offset, domain degeneracy shape)
    col_trees: Vec<(FusionTreeKey, usize, Vec<usize>)>,
    /// Column-major `rows x cols` matrix.
    data: Vec<D>,
}

struct SectorFactors<D> {
    sector: SectorId,
    /// Full rank of the dense factorization (leading dimension of `vt`).
    rank: usize,
    /// Kept singular values after truncation.
    kept: usize,
    rows: usize,
    u: Vec<D>,
    vt: Vec<D>,
}

/// All singular values per coupled sector, descending (MatrixAlgebraKit
/// `svd_vals`). Runs the dense SVD per sector through the executor and keeps
/// only the spectra.
pub fn svd_vals<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    svd_compact(dense, rule, tensor).map(|svd| svd.singular_values)
}

/// Truncated fusion-tensor SVD (MatrixAlgebraKit `svd_trunc`).
///
/// Layering: the untruncated compact factorization runs on the device
/// boundary ([`svd_compact`]); the truncation decision is host-side scalar
/// work over the spectra and its application slices the leading bond states
/// per sector.
pub fn svd_trunc<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
    truncation: &Truncation,
) -> Result<SvdTrunc<D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let full = svd_compact(dense, rule, tensor)?;
    truncate_svd(rule, full, truncation)
}

/// Compact (untruncated) fusion-tensor SVD through the device boundary.
pub fn svd_compact<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<SvdCompact<D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
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
            .svd(D::dense_read(view))
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
            values: D::real_spectrum(&outputs[1]).map_err(OperationError::Dense)?,
        });
        factors.push(SectorFactors {
            sector: matrix.sector,
            rank,
            kept: rank,
            rows: matrix.rows,
            u: D::dense_slice(&outputs[0])
                .map_err(OperationError::Dense)?
                .to_vec(),
            vt: D::dense_slice(&outputs[2])
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

    let s_tensor = diagonal_bond_tensor(rule, &singular_values, &D::from_real)?;
    Ok(SvdCompact {
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
fn decide_bond_truncation<R, V>(
    rule: &R,
    spectra: &[SectorSpectrum<V>],
    truncation: &Truncation,
) -> crate::truncation::TruncationDecision
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    V: SpectrumMagnitude,
{
    let magnitudes: Vec<Vec<f64>> = spectra
        .iter()
        .map(|entry| entry.values.iter().map(|value| value.magnitude()).collect())
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
/// [`svd_trunc`]).
///
/// The decision is host-side scalar work over the spectra; the application
/// keeps the leading bond states per coupled sector, which in the coupled
/// layout is a per-sector leading-columns/rows copy (device kernel later).
pub(crate) fn truncate_svd<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    full: SvdCompact<D, NOUT, NIN>,
    truncation: &Truncation,
) -> Result<SvdTrunc<D, NOUT, NIN>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let decision = decide_bond_truncation(rule, &full.singular_values, truncation);
    if full
        .singular_values
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok(SvdTrunc {
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
    let s_tensor = diagonal_bond_tensor(rule, &singular_values, &D::from_real)?;
    Ok(SvdTrunc {
        u: u_tensor,
        s: s_tensor,
        vh: vh_tensor,
        singular_values,
        error: decision.error,
    })
}

/// Rebuilds a factor tensor with the bond leg (`axis`) shrunk to the kept
/// prefix per coupled sector, copying leading bond states blockwise.
fn sliced_bond_tensor<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    source: &TensorMap<D, NOUT, NIN>,
    axis: usize,
    kept_of: &dyn Fn(SectorId) -> usize,
) -> Result<TensorMap<D, NOUT, NIN>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
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
    let mut sliced =
        TensorMap::<D, NOUT, NIN>::from_vec_with_fusion_space(vec![D::zero(); len], space)
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
struct FactorPair<D> {
    sector: SectorId,
    kept: usize,
    left: Vec<D>,
    left_rows: usize,
    right: Vec<D>,
    right_leading: usize,
}

/// Builds the `(codomain <- W, W <- domain)` tensor pair shared by SVD and
/// the orthogonal factorizations, in the coupled-sector matrix layout.
fn build_left_right_pair<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    fusion_space: &FusionTensorMapSpace<NOUT, NIN>,
    dims: &[usize],
    matricizations: &[SectorMatricization<D>],
    pairs: &[FactorPair<D>],
) -> Result<(TensorMap<D, NOUT, 1>, TensorMap<D, 1, NIN>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
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
        TensorMap::<D, NOUT, 1>::from_vec_with_fusion_space(vec![D::zero(); left_len], left_space)
            .map_err(OperationError::from_core_preserving_context)?;
    let right_len = right_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut right_tensor =
        TensorMap::<D, 1, NIN>::from_vec_with_fusion_space(vec![D::zero(); right_len], right_space)
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
pub struct EighFull<D, const NOUT: usize, const NIN: usize> {
    pub d: TensorMap<D, 1, 1>,
    pub v: TensorMap<D, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum>,
}

/// Truncated Hermitian eigendecomposition; `error` is the
/// quantum-dimension-weighted 2-norm of the discarded eigenvalues.
#[derive(Clone, Debug)]
pub struct EighTrunc<D, const NOUT: usize, const NIN: usize> {
    pub d: TensorMap<D, 1, 1>,
    pub v: TensorMap<D, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum>,
    pub error: f64,
}

/// Full Hermitian eigendecomposition through the device boundary.
pub fn eigh_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<EighFull<D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
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
            .eigh(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        if outputs.len() != 2 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "dense eigh must return exactly (values, vectors)",
            });
        }
        let n = matrix.rows;
        validate_dense_shape(outputs[0].shape(), &[n])?;
        validate_dense_shape(outputs[1].shape(), &[n, n])?;
        let values = D::real_spectrum(&outputs[0]).map_err(OperationError::Dense)?;
        let vectors = D::dense_slice(&outputs[1]).map_err(OperationError::Dense)?;

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
        let mut sorted_vectors = vec![D::zero(); n * n];
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
            // Discarded placeholder; only the left factor (V) is kept.
            right: vec![D::zero(); n * n],
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
    let d_tensor = diagonal_bond_tensor(rule, &eigenvalues, &D::from_real)?;
    Ok(EighFull {
        d: d_tensor,
        v: v_tensor,
        eigenvalues,
    })
}

/// Truncated Hermitian eigendecomposition: [`eigh_full`] on the device
/// boundary plus the shared host-side truncation by `|eigenvalue|`.
pub fn eigh_trunc<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
    truncation: &Truncation,
) -> Result<EighTrunc<D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let full = eigh_full(dense, rule, tensor)?;
    let decision = decide_bond_truncation(rule, &full.eigenvalues, truncation);
    if full
        .eigenvalues
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok(EighTrunc {
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
    let d_tensor = diagonal_bond_tensor(rule, &eigenvalues, &D::from_real)?;
    Ok(EighTrunc {
        d: d_tensor,
        v: v_tensor,
        eigenvalues,
        error: decision.error,
    })
}

/// Full fusion-tensor SVD `t = U * S * Vh` (MatrixAlgebraKit `svd_full`):
/// per sector `U` is the square `m x m` unitary, `S` the rectangular
/// `m x n` diagonal, and `Vh` the square `n x n` unitary.
#[derive(Clone, Debug)]
pub struct SvdFull<D, const NOUT: usize, const NIN: usize> {
    pub u: TensorMap<D, NOUT, 1>,
    pub s: TensorMap<D, 1, 1>,
    pub vh: TensorMap<D, 1, NIN>,
    pub singular_values: Vec<SectorSpectrum>,
}

/// Full fusion-tensor SVD through the device boundary.
///
/// The unitaries are completed from the compact factors with an extra
/// economy QR of `[U1 | I]` per sector (any orthonormal completion is exact
/// because the corresponding rows/columns of `S` are zero), so the whole
/// computation stays on the existing dense-executor boundary.
pub fn svd_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<SvdFull<D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?
        .clone();
    let matricizations = sector_matricizations(rule, tensor, NOUT)?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    let mut singular_values = Vec::with_capacity(matricizations.len());
    let mut col_dims: Vec<(SectorId, usize)> = Vec::new();
    for matrix in &matricizations {
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(&matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let outputs = dense
            .svd(D::dense_read(view))
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
        let u_thin = D::dense_slice(&outputs[0]).map_err(OperationError::Dense)?;
        let s_values = D::real_spectrum(&outputs[1]).map_err(OperationError::Dense)?;
        let vt_thin = D::dense_slice(&outputs[2]).map_err(OperationError::Dense)?;

        let u_full = orthonormal_completion(dense, u_thin, matrix.rows, rank)?;
        // V columns are the adjoint rows of Vh; complete V (n x rank) to
        // n x n, then store Vh = V^H.
        let v_thin = adjoint_col_major(vt_thin, rank, matrix.cols);
        let v_full = orthonormal_completion(dense, &v_thin, matrix.cols, rank)?;
        let vh_full = adjoint_col_major(&v_full, matrix.cols, matrix.cols);

        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s_values.clone(),
        });
        col_dims.push((matrix.sector, matrix.cols));
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: matrix.rows,
            left: u_full,
            left_rows: matrix.rows,
            right: vh_full,
            right_leading: matrix.cols,
        });
    }

    let cols_of = |sector: SectorId| {
        col_dims
            .iter()
            .find(|(candidate, _)| *candidate == sector)
            .map(|(_, cols)| *cols)
            .expect("column dimension recorded per sector")
    };
    // The left/right bond legs differ in the full SVD (rows vs columns), so
    // build the two factors with separate bond dimensions.
    let (u_tensor, _) = build_left_right_pair(
        rule,
        &fusion_space,
        tensor.space().dims(),
        &matricizations,
        &pairs
            .iter()
            .map(|pair| FactorPair {
                sector: pair.sector,
                kept: pair.left_rows,
                left: pair.left.clone(),
                left_rows: pair.left_rows,
                // Discarded placeholder sized kept x cols for the scatter.
                right: vec![D::zero(); pair.left_rows * cols_of(pair.sector)],
                right_leading: pair.left_rows,
            })
            .collect::<Vec<_>>(),
    )?;
    let (_, vh_tensor) = build_left_right_pair(
        rule,
        &fusion_space,
        tensor.space().dims(),
        &matricizations,
        &pairs
            .iter()
            .map(|pair| FactorPair {
                sector: pair.sector,
                kept: cols_of(pair.sector),
                // Discarded placeholder sized rows x kept for the scatter.
                left: vec![D::zero(); pair.left_rows * cols_of(pair.sector)],
                left_rows: pair.left_rows,
                right: pair.right.clone(),
                right_leading: cols_of(pair.sector),
            })
            .collect::<Vec<_>>(),
    )?;
    let s_tensor = rectangular_diagonal_bond_tensor(
        rule,
        &singular_values,
        &|sector| {
            pairs
                .iter()
                .find(|pair| pair.sector == sector)
                .map(|pair| pair.left_rows)
                .unwrap_or(0)
        },
        &cols_of,
    )?;
    Ok(SvdFull {
        u: u_tensor,
        s: s_tensor,
        vh: vh_tensor,
        singular_values,
    })
}

/// Completes `k` orthonormal columns (`m x k`, column-major) to a full
/// `m x m` orthonormal basis via an economy QR of `[Q1 | I]`; the first `k`
/// columns are returned unchanged.
fn orthonormal_completion<E, D>(
    dense: &mut E,
    thin: &[D],
    rows: usize,
    rank: usize,
) -> Result<Vec<D>, OperationError>
where
    E: DenseExecutor,
    D: FactorScalar,
{
    if rank == rows {
        return Ok(thin.to_vec());
    }
    let mut augmented = vec![D::zero(); rows * (rank + rows)];
    augmented[..rows * rank].copy_from_slice(thin);
    for row in 0..rows {
        augmented[rows * rank + row * rows + row] = D::one();
    }
    let shape = [rows, rank + rows];
    let strides = [1usize, rows];
    let view = DenseView::new(&augmented, &shape, &strides, 0).map_err(OperationError::Dense)?;
    let outputs = dense
        .qr(D::dense_read(view))
        .map_err(OperationError::Dense)?;
    if outputs.len() != 2 {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "dense QR must return exactly (Q, R)",
        });
    }
    validate_dense_shape(outputs[0].shape(), &[rows, rows])?;
    let q = D::dense_slice(&outputs[0]).map_err(OperationError::Dense)?;
    let mut full = vec![D::zero(); rows * rows];
    full[..rows * rank].copy_from_slice(thin);
    full[rows * rank..].copy_from_slice(&q[rows * rank..rows * rows]);
    Ok(full)
}

/// Rectangular diagonal `W_row <- W_col` bond tensor (the `S` of the full
/// SVD): per sector shape `[rows, cols]` with the spectrum on the diagonal.
fn rectangular_diagonal_bond_tensor<R, D>(
    rule: &R,
    spectra: &[SectorSpectrum],
    rows_of: &dyn Fn(SectorId) -> usize,
    cols_of: &dyn Fn(SectorId) -> usize,
) -> Result<TensorMap<D, 1, 1>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let row_leg = SectorLeg::new(spectra.iter().map(|entry| entry.sector), false);
    let col_leg = SectorLeg::new(spectra.iter().map(|entry| entry.sector), false);
    let total_rows: usize = spectra.iter().map(|entry| rows_of(entry.sector)).sum();
    let total_cols: usize = spectra.iter().map(|entry| cols_of(entry.sector)).sum();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([row_leg]),
        FusionProductSpace::new([col_leg]),
    );
    let keys = homspace.fusion_tree_keys(rule);
    let shapes = keys
        .iter()
        .map(|key| {
            let sector = coupled_of(rule, key.codomain_tree());
            vec![rows_of(sector), cols_of(sector)]
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([total_rows], [total_cols])
            .map_err(OperationError::from_core_preserving_context)?,
        homspace,
        rule,
        shapes,
    )
    .map_err(OperationError::from_core_preserving_context)?;
    TensorMap::<D, 1, 1>::from_block_fn_with_fusion_space(space, D::zero(), |key, indices| {
        if indices[0] != indices[1] {
            return D::zero();
        }
        let BlockKey::FusionTree(tree) = key else {
            return D::zero();
        };
        let sector = tree
            .codomain_tree()
            .coupled()
            .unwrap_or_else(|| tree.codomain_tree().uncoupled()[0]);
        spectra
            .iter()
            .find(|entry| entry.sector == sector)
            .and_then(|entry| entry.values.get(indices[0]).copied())
            .map(D::from_real)
            .unwrap_or_else(D::zero)
    })
    .map_err(OperationError::from_core_preserving_context)
}

/// Full QR `t = Q * R` (MatrixAlgebraKit `qr_full`): per sector `Q` is the
/// square `m x m` unitary and `R` the upper-trapezoidal `m x n`, obtained
/// from one economy QR of the augmented `[A | I]` on the dense boundary.
pub fn qr_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<(TensorMap<D, NOUT, 1>, TensorMap<D, 1, NIN>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?
        .clone();
    let matricizations = sector_matricizations(rule, tensor, NOUT)?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let mut augmented = vec![D::zero(); rows * (cols + rows)];
        augmented[..rows * cols].copy_from_slice(&matrix.data);
        for row in 0..rows {
            augmented[rows * cols + row * rows + row] = D::one();
        }
        let shape = [rows, cols + rows];
        let strides = [1usize, rows];
        let view =
            DenseView::new(&augmented, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let outputs = dense
            .qr(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        if outputs.len() != 2 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "dense QR must return exactly (Q, R)",
            });
        }
        validate_dense_shape(outputs[0].shape(), &[rows, rows])?;
        validate_dense_shape(outputs[1].shape(), &[rows, cols + rows])?;
        let q = D::dense_slice(&outputs[0]).map_err(OperationError::Dense)?;
        let r_augmented = D::dense_slice(&outputs[1]).map_err(OperationError::Dense)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rows,
            left: q.to_vec(),
            left_rows: rows,
            right: r_augmented[..rows * cols].to_vec(),
            right_leading: rows,
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

/// Full LQ `t = L * Q` (MatrixAlgebraKit `lq_full`): per sector `L` is the
/// lower-trapezoidal `m x n` and `Q` the square `n x n` unitary, via the full
/// QR of the transposed sector matrices.
pub fn lq_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<(TensorMap<D, NOUT, 1>, TensorMap<D, 1, NIN>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?
        .clone();
    let matricizations = sector_matricizations(rule, tensor, NOUT)?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let transposed = adjoint_col_major(&matrix.data, rows, cols);
        let mut augmented = vec![D::zero(); cols * (rows + cols)];
        augmented[..cols * rows].copy_from_slice(&transposed);
        for row in 0..cols {
            augmented[cols * rows + row * cols + row] = D::one();
        }
        let shape = [cols, rows + cols];
        let strides = [1usize, cols];
        let view =
            DenseView::new(&augmented, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let outputs = dense
            .qr(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        if outputs.len() != 2 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "dense QR must return exactly (Q, R)",
            });
        }
        validate_dense_shape(outputs[0].shape(), &[cols, cols])?;
        validate_dense_shape(outputs[1].shape(), &[cols, rows + cols])?;
        let q_prime = D::dense_slice(&outputs[0]).map_err(OperationError::Dense)?;
        let r_prime = D::dense_slice(&outputs[1]).map_err(OperationError::Dense)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: cols,
            // L = R'^H : rows x cols (lower trapezoidal).
            left: adjoint_col_major(&r_prime[..cols * rows], cols, rows),
            left_rows: rows,
            // Q = Q'^H : cols x cols.
            right: adjoint_col_major(q_prime, cols, cols),
            right_leading: cols,
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

/// Full general eigendecomposition `t = V * D * V^-1` (MatrixAlgebraKit
/// `eig_full`): always complex, requires an endomorphism. Bond states are
/// stored descending by `|eigenvalue|` per sector.
#[derive(Clone, Debug)]
pub struct EigFull<D: FactorScalar, const NOUT: usize, const NIN: usize> {
    pub d: TensorMap<D::Eig, 1, 1>,
    pub v: TensorMap<D::Eig, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum<Complex64>>,
}

/// Truncated general eigendecomposition; `error` is the
/// quantum-dimension-weighted 2-norm of the discarded `|eigenvalues|`.
#[derive(Clone, Debug)]
pub struct EigTrunc<D: FactorScalar, const NOUT: usize, const NIN: usize> {
    pub d: TensorMap<D::Eig, 1, 1>,
    pub v: TensorMap<D::Eig, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum<Complex64>>,
    pub error: f64,
}

/// Full general eigendecomposition through the device boundary.
pub fn eig_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<EigFull<D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?
        .clone();
    if fusion_space.homspace().codomain() != fusion_space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eig requires an endomorphism (codomain == domain)",
        });
    }
    let matricizations = sector_matricizations(rule, tensor, NOUT)?;

    let mut pairs: Vec<FactorPair<D::Eig>> = Vec::with_capacity(matricizations.len());
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(&matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let outputs = dense
            .eig(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        if outputs.len() != 2 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "dense eig must return exactly (values, vectors)",
            });
        }
        let n = matrix.rows;
        validate_dense_shape(outputs[0].shape(), &[n])?;
        validate_dense_shape(outputs[1].shape(), &[n, n])?;
        let values =
            <D::Eig as FactorScalar>::dense_slice(&outputs[0]).map_err(OperationError::Dense)?;
        let vectors =
            <D::Eig as FactorScalar>::dense_slice(&outputs[1]).map_err(OperationError::Dense)?;

        let complex_values: Vec<Complex64> =
            values.iter().map(|&value| value.widen_complex()).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            complex_values[b]
                .norm()
                .partial_cmp(&complex_values[a].norm())
                .expect("finite eigenvalues")
                .then(a.cmp(&b))
        });
        let sorted_values: Vec<Complex64> =
            order.iter().map(|&index| complex_values[index]).collect();
        let mut sorted_vectors = vec![<D::Eig as num_traits::Zero>::zero(); n * n];
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
            right: vec![<D::Eig as num_traits::Zero>::zero(); n * n],
            left: sorted_vectors,
            left_rows: n,
            right_leading: n,
        });
    }

    // Rebuild the matricization skeleton at the complex scalar so the pair
    // builder can place blocks (only shapes and offsets are read).
    let complex_matricizations: Vec<SectorMatricization<D::Eig>> = matricizations
        .iter()
        .map(|matrix| SectorMatricization {
            sector: matrix.sector,
            rows: matrix.rows,
            cols: matrix.cols,
            row_trees: matrix.row_trees.clone(),
            col_trees: matrix.col_trees.clone(),
            data: Vec::new(),
        })
        .collect();
    let (v_tensor, _) = build_left_right_pair(
        rule,
        &fusion_space,
        tensor.space().dims(),
        &complex_matricizations,
        &pairs,
    )?;
    let d_tensor = diagonal_bond_tensor(
        rule,
        &eigenvalues,
        &<D::Eig as FactorScalar>::from_complex64,
    )?;
    Ok(EigFull {
        d: d_tensor,
        v: v_tensor,
        eigenvalues,
    })
}

/// Truncated general eigendecomposition: [`eig_full`] plus the shared
/// host-side truncation by `|eigenvalue|`.
pub fn eig_trunc<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
    truncation: &Truncation,
) -> Result<EigTrunc<D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let full = eig_full(dense, rule, tensor)?;
    let decision = decide_bond_truncation(rule, &full.eigenvalues, truncation);
    if full
        .eigenvalues
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok(EigTrunc {
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
    let d_tensor = diagonal_bond_tensor(
        rule,
        &eigenvalues,
        &<D::Eig as FactorScalar>::from_complex64,
    )?;
    Ok(EigTrunc {
        d: d_tensor,
        v: v_tensor,
        eigenvalues,
        error: decision.error,
    })
}

/// All Hermitian eigenvalues per coupled sector, descending by magnitude
/// (MatrixAlgebraKit `eigh_vals`).
pub fn eigh_vals<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    eigh_full(dense, rule, tensor).map(|eigh| eigh.eigenvalues)
}

/// All general eigenvalues per coupled sector, descending by magnitude
/// (MatrixAlgebraKit `eig_vals`).
pub fn eig_vals<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<Vec<SectorSpectrum<Complex64>>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    eig_full(dense, rule, tensor).map(|eig| eig.eigenvalues)
}

/// Left null space `N : codomain <- W` (MatrixAlgebraKit `left_null`): the
/// orthonormal complement of the range, i.e. the full-QR `Q` columns past the
/// compact rank. Sectors with no null directions drop out of `W`.
pub fn left_null<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<TensorMap<D, NOUT, 1>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?
        .clone();
    let matricizations = sector_matricizations(rule, tensor, NOUT)?;

    let mut pairs = Vec::new();
    for matrix in &matricizations {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let rank = rows.min(cols);
        if rank == rows {
            continue;
        }
        let mut augmented = vec![D::zero(); rows * (cols + rows)];
        augmented[..rows * cols].copy_from_slice(&matrix.data);
        for row in 0..rows {
            augmented[rows * cols + row * rows + row] = D::one();
        }
        let shape = [rows, cols + rows];
        let strides = [1usize, rows];
        let view =
            DenseView::new(&augmented, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let outputs = dense
            .qr(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        if outputs.len() != 2 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "dense QR must return exactly (Q, R)",
            });
        }
        validate_dense_shape(outputs[0].shape(), &[rows, rows])?;
        let q = D::dense_slice(&outputs[0]).map_err(OperationError::Dense)?;
        let null_dim = rows - rank;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: null_dim,
            // Null columns are the trailing full-Q columns (contiguous in the
            // column-major layout).
            left: q[rows * rank..rows * rows].to_vec(),
            left_rows: rows,
            // Discarded placeholder for the pair builder.
            right: vec![D::zero(); null_dim * cols],
            right_leading: null_dim,
        });
    }

    let (null_tensor, _) = build_left_right_pair(
        rule,
        &fusion_space,
        tensor.space().dims(),
        &matricizations,
        &pairs,
    )?;
    Ok(null_tensor)
}

/// Right null space `N : W <- domain` (MatrixAlgebraKit `right_null`): the
/// orthonormal rows spanning the kernel, i.e. the full-LQ `Q` rows past the
/// compact rank. Sectors with no null directions drop out of `W`.
pub fn right_null<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<TensorMap<D, 1, NIN>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?
        .clone();
    let matricizations = sector_matricizations(rule, tensor, NOUT)?;

    let mut pairs = Vec::new();
    for matrix in &matricizations {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let rank = rows.min(cols);
        if rank == cols {
            continue;
        }
        let adjoint = adjoint_col_major(&matrix.data, rows, cols);
        let mut augmented = vec![D::zero(); cols * (rows + cols)];
        augmented[..cols * rows].copy_from_slice(&adjoint);
        for row in 0..cols {
            augmented[cols * rows + row * cols + row] = D::one();
        }
        let shape = [cols, rows + cols];
        let strides = [1usize, cols];
        let view =
            DenseView::new(&augmented, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let outputs = dense
            .qr(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        if outputs.len() != 2 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "dense QR must return exactly (Q, R)",
            });
        }
        validate_dense_shape(outputs[0].shape(), &[cols, cols])?;
        let q_prime = D::dense_slice(&outputs[0]).map_err(OperationError::Dense)?;
        let null_dim = cols - rank;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: null_dim,
            // Discarded placeholder for the pair builder.
            left: vec![D::zero(); rows * null_dim],
            left_rows: rows,
            // Null rows are the adjoints of the trailing Q' columns.
            right: adjoint_col_major(&q_prime[cols * rank..cols * cols], cols, null_dim),
            right_leading: null_dim,
        });
    }

    let (_, null_tensor) = build_left_right_pair(
        rule,
        &fusion_space,
        tensor.space().dims(),
        &matricizations,
        &pairs,
    )?;
    Ok(null_tensor)
}

/// Left polar decomposition `t = W * P` (MatrixAlgebraKit `left_polar`):
/// `W` is the isometry `U * Vh` and `P = V * S * Vh` the positive part on
/// the domain.
pub fn left_polar<E, RuleKey, BT, BC, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<(TensorMap<D, NOUT, NIN>, TensorMap<D, NIN, NIN>), OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + std::hash::Hash,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let svd = svd_compact(dense, rule, tensor)?;
    let isometry = crate::compose::compose(context, rule, &svd.u, &svd.vh)?;
    let v = tenet_tensors::adjoint(rule, &svd.vh)?;
    let vs = crate::compose::compose(context, rule, &v, &svd.s)?;
    let positive = crate::compose::compose(context, rule, &vs, &svd.vh)?;
    Ok((isometry, positive))
}

/// Right polar decomposition `t = P * W` (MatrixAlgebraKit `right_polar`):
/// `P = U * S * U^H` is the positive part on the codomain and `W = U * Vh`.
pub fn right_polar<E, RuleKey, BT, BC, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<(TensorMap<D, NOUT, NOUT>, TensorMap<D, NOUT, NIN>), OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + std::hash::Hash,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let svd = svd_compact(dense, rule, tensor)?;
    let uh = tenet_tensors::adjoint(rule, &svd.u)?;
    let us = crate::compose::compose(context, rule, &svd.u, &svd.s)?;
    let positive = crate::compose::compose(context, rule, &us, &uh)?;
    let isometry = crate::compose::compose(context, rule, &svd.u, &svd.vh)?;
    Ok((positive, isometry))
}

/// Compact QR `t = Q * R` (MatrixAlgebraKit `qr_compact`):
/// `Q : codomain <- W` has orthonormal columns per coupled sector and
/// `R : W <- domain` with per-sector bond `min(rows, cols)`.
pub fn qr_compact<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<(TensorMap<D, NOUT, 1>, TensorMap<D, 1, NIN>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
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
            .qr(D::dense_read(view))
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
            left: D::dense_slice(&outputs[0])
                .map_err(OperationError::Dense)?
                .to_vec(),
            left_rows: matrix.rows,
            right: D::dense_slice(&outputs[1])
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
pub fn lq_compact<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<(TensorMap<D, NOUT, 1>, TensorMap<D, 1, NIN>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?
        .clone();
    let matricizations = sector_matricizations(rule, tensor, NOUT)?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        // QR of the adjoint: t^H = Q' R'  =>  t = R'^H Q'^H = L Q.
        let transposed = adjoint_col_major(&matrix.data, matrix.rows, matrix.cols);
        let shape = [matrix.cols, matrix.rows];
        let strides = [1usize, matrix.cols];
        let view =
            DenseView::new(&transposed, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let outputs = dense
            .qr(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        if outputs.len() != 2 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "dense QR must return exactly (Q, R)",
            });
        }
        let rank = matrix.rows.min(matrix.cols);
        validate_dense_shape(outputs[0].shape(), &[matrix.cols, rank])?;
        validate_dense_shape(outputs[1].shape(), &[rank, matrix.rows])?;
        let q_prime = D::dense_slice(&outputs[0]).map_err(OperationError::Dense)?;
        let r_prime = D::dense_slice(&outputs[1]).map_err(OperationError::Dense)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rank,
            // L = R'^H : rows x rank.
            left: adjoint_col_major(r_prime, rank, matrix.rows),
            left_rows: matrix.rows,
            // Q = Q'^H : rank x cols.
            right: adjoint_col_major(q_prime, matrix.cols, rank),
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
/// Adjoint (conjugate transpose) of a column-major `rows x cols` matrix.
fn adjoint_col_major<D: FactorScalar>(data: &[D], rows: usize, cols: usize) -> Vec<D> {
    let mut adjoint = vec![D::zero(); data.len()];
    for col in 0..cols {
        for row in 0..rows {
            adjoint[col + cols * row] = FactorScalar::adjoint(data[row + rows * col]);
        }
    }
    adjoint
}

/// Copies a dense column-major matrix region into one fusion-tree subblock.
///
/// `matrix_axis` names the block axis that walks the matrix's own leading
/// dimension side; the remaining axes enumerate the offset side column-major.
/// For `U` the matrix axis is the trailing (new leg) axis and the codomain
/// axes select rows at `side_offset`; for `Vt` the matrix axis is the leading
/// (new leg) axis and the domain axes select columns at `side_offset`.
#[allow(clippy::too_many_arguments)]
fn scatter_matrix_block<D: Copy>(
    data: &mut [D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    matrix_axis: usize,
    matrix: &[D],
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

fn matricization_of<D>(
    matricizations: &[SectorMatricization<D>],
    sector: SectorId,
) -> Result<&SectorMatricization<D>, OperationError> {
    matricizations
        .iter()
        .find(|matrix| matrix.sector == sector)
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: "factor tree references a coupled sector absent from the source tensor",
        })
}

fn row_placement<'a, D>(
    matrix: &'a SectorMatricization<D>,
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

fn col_placement<'a, D>(
    matrix: &'a SectorMatricization<D>,
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

fn row_shape_of<D>(
    matricizations: &[SectorMatricization<D>],
    sector: SectorId,
    tree: &FusionTreeKey,
) -> Result<Vec<usize>, OperationError> {
    row_placement(matricization_of(matricizations, sector)?, tree).map(|(_, shape)| shape.to_vec())
}

fn col_shape_of<D>(
    matricizations: &[SectorMatricization<D>],
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
fn sector_matricizations<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
    nout: usize,
) -> Result<Vec<SectorMatricization<D>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let structure = std::sync::Arc::clone(tensor.structure());
    let mut matricizations: Vec<SectorMatricization<D>> = Vec::new();

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
                matricizations.push(SectorMatricization::<D> {
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
        matrix.data = vec![D::zero(); matrix.rows * matrix.cols];
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
