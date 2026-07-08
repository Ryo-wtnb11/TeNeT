use std::collections::HashMap;
use std::sync::Arc;

use num_complex::Complex64;
use num_traits::Zero;
use tenet_core::{
    BlockKey, BlockStructure, CoreError, FusionProductSpace, FusionTensorMapSpace,
    FusionTreeHomSpace, FusionTreeKey, MultiplicityFreeRigidSymbols, SectorId, SectorLeg,
    TensorMap, TensorMapSpace,
};
use tenet_dense::{DenseError, DenseExecutor, DenseTensor, DenseView, DenseViewMut};

use tenet_tensors::{DenseBlockScalar, DenseRecouplingScalar, DynamicFusionMapSpace};

use crate::truncation::{select_truncation, Truncation, WeightedSpectrum};
use tenet_tensors::OperationError;

/// Scalar contract for the factorization layer: dense-executor I/O plus the
/// adjoint and real-embedding used by the factor builders. Implemented for
/// the double-precision real and complex scalars.
pub trait FactorScalar: DenseRecouplingScalar {
    /// Output scalar of the general (non-Hermitian) eigendecomposition.
    type Eig: FactorScalar;
    /// Real scalar used by singular-value/eigenvalue outputs.
    type Real: DenseRecouplingScalar + Into<f64>;

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
    type Real = f32;

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
    type Real = f64;

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
    type Real = f32;

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
    type Real = f64;

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
    fn nonnegative_f64_slice(_values: &[Self]) -> Option<&[f64]> {
        None
    }
}

impl SpectrumMagnitude for f64 {
    fn magnitude(self) -> f64 {
        self.abs()
    }

    fn nonnegative_f64_slice(values: &[Self]) -> Option<&[f64]> {
        Some(values)
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

// ---------------------------------------------------------------------------
// Dynamic-rank representation.
// ---------------------------------------------------------------------------

/// Dynamic-rank factor tensor: an expert-layer space handle plus flat data in
/// the coupled-sector matrix layout (the same pair `tenet_tensors::adjoint_dyn`
/// returns).
pub type DynFactor<D> = (DynamicFusionMapSpace, Vec<D>);

/// Rank-erases the fusion space of a typed tensor (shared handles, no copy).
pub(crate) fn dyn_space_of<D, const NOUT: usize, const NIN: usize>(
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<DynamicFusionMapSpace, OperationError> {
    Ok(DynamicFusionMapSpace::from_typed(
        tensor
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    ))
}

/// Rebuilds a typed tensor from a dynamic factor: the subblock structure and
/// hom space are shared as-is (identical layout by construction), only the
/// dense bookkeeping dims are recomputed as per-axis degeneracy totals.
pub(crate) fn typed_from_dyn<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    (space, data): DynFactor<D>,
) -> Result<TensorMap<D, NOUT, NIN>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if space.nout() != NOUT || space.nin() != NIN {
        return Err(OperationError::RankMismatch {
            expected: NOUT + NIN,
            actual: space.rank(),
        });
    }
    let structure = space.structure();
    let rank = NOUT + NIN;
    let mut per_axis: Vec<HashMap<SectorId, usize>> = vec![HashMap::new(); rank];
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sectors = key.external_sectors(rule);
        for (axis, (&sector, &dim)) in sectors.iter().zip(block.shape()).enumerate() {
            per_axis[axis].entry(sector).or_insert(dim);
        }
    }
    let dims: Vec<usize> = per_axis.iter().map(|axis| axis.values().sum()).collect();
    let mut codomain_dims = [0usize; NOUT];
    codomain_dims.copy_from_slice(&dims[..NOUT]);
    let mut domain_dims = [0usize; NIN];
    domain_dims.copy_from_slice(&dims[NOUT..]);
    let typed_space = FusionTensorMapSpace::from_shared_subblock_structure(
        TensorMapSpace::<NOUT, NIN>::from_dims(codomain_dims, domain_dims)
            .map_err(OperationError::from_core_preserving_context)?,
        space.homspace().clone(),
        Arc::clone(space.structure()),
    )
    .map_err(OperationError::from_core_preserving_context)?;
    TensorMap::from_vec_with_fusion_space(data, typed_space)
        .map_err(OperationError::from_core_preserving_context)
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

/// Dynamic-rank [`SvdTrunc`].
#[derive(Clone, Debug)]
pub struct SvdTruncDyn<D> {
    pub u: DynFactor<D>,
    pub s: DynFactor<D>,
    pub vh: DynFactor<D>,
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

/// Dynamic-rank [`SvdCompact`].
#[derive(Clone, Debug)]
pub struct SvdCompactDyn<D> {
    pub u: DynFactor<D>,
    pub s: DynFactor<D>,
    pub vh: DynFactor<D>,
    pub singular_values: Vec<SectorSpectrum>,
}

/// Materializes per-sector spectra as a diagonal factor `W <- W` in the
/// coupled layout (`S` for the SVD, `D` for eigendecompositions).
pub(crate) fn diagonal_bond_tensor_dyn<R, D, V>(
    rule: &R,
    singular_values: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<DynFactor<D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
    V: Copy,
{
    let new_leg = SectorLeg::new(
        singular_values
            .iter()
            .map(|entry| (entry.sector, entry.values.len())),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg.clone()]),
        FusionProductSpace::new([new_leg]),
    );
    let spectrum_by_sector: HashMap<SectorId, &SectorSpectrum<V>> = singular_values
        .iter()
        .map(|entry| (entry.sector, entry))
        .collect();
    let keys = homspace.fusion_tree_keys(rule);
    let shapes = keys
        .iter()
        .map(|key| {
            let sector = coupled_of(rule, key.codomain_tree());
            let count = spectrum_by_sector
                .get(&sector)
                .map(|entry| entry.values.len())
                .unwrap_or(0);
            vec![count, count]
        })
        .collect::<Vec<_>>();
    let space = DynamicFusionMapSpace::from_degeneracy_shapes(rule, homspace, shapes)?;
    let len = space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); len];
    let structure = Arc::clone(space.structure());
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree
            .codomain_tree()
            .coupled()
            .unwrap_or_else(|| tree.codomain_tree().uncoupled()[0]);
        let Some(&entry) = spectrum_by_sector.get(&sector) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        copy_mapped_to_strided_diagonal(
            &mut data,
            offset,
            strides[0] + strides[1],
            &entry.values[..count],
            to_scalar,
        );
    }
    Ok((space, data))
}

/// Multiplies each entry of a left-type factor (`codomain <- bond`, i.e. the
/// bond leg is the trailing axis — the eigenvector / `adjoint(vh)` layout from
/// [`build_left_right_spaces`]) by its bond-index spectrum value, in place.
///
/// This is `factor * D` where `D` is the diagonal `bond <- bond` tensor of
/// `spectrum` (as [`diagonal_bond_tensor_dyn`] would build), but done as an
/// O(size) column scaling instead of materializing the dense `rank x rank`
/// diagonal (99% zeros) and running it through a full block GEMM. See #46:
/// TensorKit never forms the diagonal either — its `DiagonalTensorMap` makes
/// `S * t` an `lmul!`/`rmul!` scaling.
pub(crate) fn scale_bond_axis_by_spectrum<D>(
    factor: &mut DynFactor<D>,
    spectrum: &[SectorSpectrum],
) -> Result<(), OperationError>
where
    D: FactorScalar,
{
    let (space, data) = factor;
    let spectrum_by_sector: HashMap<SectorId, &SectorSpectrum> =
        spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    let structure = Arc::clone(space.structure());
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree
            .codomain_tree()
            .coupled()
            .unwrap_or_else(|| tree.codomain_tree().uncoupled()[0]);
        // Absent sector => this block is a structurally-zero bond; nothing to
        // scale (mirrors the `unwrap_or(0)` shape in `diagonal_bond_tensor_dyn`).
        let Some(&entry) = spectrum_by_sector.get(&sector) else {
            continue;
        };
        let shape = block.shape();
        if shape.is_empty() {
            continue;
        }
        let strides = block.strides();
        let offset = block.offset();
        let last = shape.len() - 1;
        let bond = shape[last];
        let bond_stride = strides[last];
        debug_assert_eq!(
            bond,
            entry.values.len(),
            "bond degeneracy must match the spectrum length"
        );
        let bond = bond.min(entry.values.len());
        // Walk every combination of the leading (non-bond) axes; for each,
        // scale the `bond` trailing entries by the spectrum.
        let lead_shape = &shape[..last];
        let lead_strides = &strides[..last];
        let outer: usize = lead_shape.iter().product();
        let mut coord = vec![0usize; lead_shape.len()];
        for _ in 0..outer {
            let mut base = offset;
            for (c, stride) in coord.iter().zip(lead_strides) {
                base += c * stride;
            }
            for j in 0..bond {
                let scale = D::from_real(entry.values[j]);
                let idx = base + j * bond_stride;
                data[idx] = data[idx] * scale;
            }
            for axis in (0..coord.len()).rev() {
                coord[axis] += 1;
                if coord[axis] < lead_shape[axis] {
                    break;
                }
                coord[axis] = 0;
            }
        }
    }
    Ok(())
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
pub fn sector_matricization_diagnostic<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<Vec<SectorMatricizationDiagnostic>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = dyn_space_of(tensor)?;
    Ok(
        sector_matricizations(rule, space.structure(), tensor.data(), space.nout())?
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
    svd_vals_dyn(dense, rule, &dyn_space_of(tensor)?, tensor.data())
}

/// Dynamic-rank [`svd_vals`].
pub fn svd_vals_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    svd_compact_dyn(dense, rule, space, data).map(|svd| svd.singular_values)
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
    let out = svd_trunc_dyn(
        dense,
        rule,
        &dyn_space_of(tensor)?,
        tensor.data(),
        truncation,
    )?;
    Ok(SvdTrunc {
        u: typed_from_dyn(rule, out.u)?,
        s: typed_from_dyn(rule, out.s)?,
        vh: typed_from_dyn(rule, out.vh)?,
        singular_values: out.singular_values,
        error: out.error,
    })
}

/// Dynamic-rank [`svd_trunc`].
pub fn svd_trunc_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
    truncation: &Truncation,
) -> Result<SvdTruncDyn<D>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let full = svd_compact_dyn(dense, rule, space, data)?;
    truncate_svd_dyn(rule, full, truncation)
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
    let out = svd_compact_dyn(dense, rule, &dyn_space_of(tensor)?, tensor.data())?;
    Ok(SvdCompact {
        u: typed_from_dyn(rule, out.u)?,
        s: typed_from_dyn(rule, out.s)?,
        vh: typed_from_dyn(rule, out.vh)?,
        singular_values: out.singular_values,
    })
}

/// Dynamic-rank [`svd_compact`]: the shared core of every SVD entry point.
pub fn svd_compact_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<SvdCompactDyn<D>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let matricizations = sector_matricizations(rule, space.structure(), data, space.nout())?;

    let ranks = matricizations
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector,
            kept: matrix.rows.min(matrix.cols),
        })
        .collect::<Vec<_>>();
    let (u_space, vt_space) =
        build_left_right_spaces(rule, space.homspace(), &matricizations, &ranks)?;
    let u_len = u_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut u_data = vec![D::zero(); u_len];
    let vt_len = vt_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut vt_data = vec![D::zero(); vt_len];

    let max_rows = matricizations
        .iter()
        .map(|matrix| matrix.rows)
        .max()
        .unwrap_or(0);
    let max_cols = matricizations
        .iter()
        .map(|matrix| matrix.cols)
        .max()
        .unwrap_or(0);
    let max_rank = ranks.iter().map(|rank| rank.kept).max().unwrap_or(0);
    let mut u_workspace = vec![D::zero(); max_rows * max_rank];
    let mut s_workspace = vec![D::Real::zero(); max_rank];
    let mut vt_workspace = vec![D::zero(); max_rank * max_cols];
    let mut singular_values = Vec::with_capacity(matricizations.len());

    for matrix in &matricizations {
        let rank = matrix.rows.min(matrix.cols);
        let input_shape = [matrix.rows, matrix.cols];
        let input_strides = [1usize, matrix.rows];
        let input = DenseView::new(&matrix.data, &input_shape, &input_strides, 0)
            .map_err(OperationError::Dense)?;
        let u_shape = [matrix.rows, rank];
        let u_strides = [1usize, max_rows];
        let s_shape = [rank];
        let s_strides = [1usize];
        let vt_shape = [rank, matrix.cols];
        let vt_strides = [1usize, max_rank];
        let u_view = DenseViewMut::new(&mut u_workspace, &u_shape, &u_strides, 0)
            .map_err(OperationError::Dense)?;
        let s_view = DenseViewMut::new(&mut s_workspace, &s_shape, &s_strides, 0)
            .map_err(OperationError::Dense)?;
        let vt_view = DenseViewMut::new(&mut vt_workspace, &vt_shape, &vt_strides, 0)
            .map_err(OperationError::Dense)?;
        dense
            .svd_into(
                D::dense_read(input),
                D::dense_write(u_view),
                D::Real::dense_write(s_view),
                D::dense_write(vt_view),
            )
            .map_err(OperationError::Dense)?;
        svd_compact_gauge(
            &mut u_workspace,
            matrix.rows,
            max_rows,
            &mut vt_workspace,
            rank,
            matrix.cols,
            max_rank,
        );

        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s_workspace[..rank]
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
        });
        scatter_left_sector_blocks(rule, &u_space, &mut u_data, matrix, &u_workspace, max_rows)?;
        scatter_right_sector_blocks(
            rule,
            &vt_space,
            &mut vt_data,
            matrix,
            &vt_workspace,
            max_rank,
        )?;
    }

    let s_factor = diagonal_bond_tensor_dyn(rule, &singular_values, &D::from_real)?;
    Ok(SvdCompactDyn {
        u: (u_space, u_data),
        s: s_factor,
        vh: (vt_space, vt_data),
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
    values_are_nonnegative: bool,
) -> crate::truncation::TruncationDecision
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    V: SpectrumMagnitude,
{
    enum MagnitudeValues<'a> {
        Borrowed(&'a [f64]),
        Owned(Vec<f64>),
    }

    impl<'a> MagnitudeValues<'a> {
        fn as_slice(&self) -> &[f64] {
            match self {
                MagnitudeValues::Borrowed(values) => values,
                MagnitudeValues::Owned(values) => values,
            }
        }
    }

    let magnitudes: Vec<MagnitudeValues<'_>> = spectra
        .iter()
        .map(|entry| {
            if values_are_nonnegative {
                if let Some(values) = V::nonnegative_f64_slice(&entry.values) {
                    return MagnitudeValues::Borrowed(values);
                }
            }
            MagnitudeValues::Owned(entry.values.iter().map(|value| value.magnitude()).collect())
        })
        .collect();
    let weighted: Vec<WeightedSpectrum<'_>> = spectra
        .iter()
        .zip(&magnitudes)
        .map(|(entry, values)| WeightedSpectrum {
            weight: rule.dim_scalar(entry.sector),
            values: values.as_slice(),
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
#[cfg_attr(not(test), allow(dead_code))] // exercised by the typed test suite
pub(crate) fn truncate_svd<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    full: SvdCompact<D, NOUT, NIN>,
    truncation: &Truncation,
) -> Result<SvdTrunc<D, NOUT, NIN>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let full_dyn = SvdCompactDyn {
        u: (dyn_space_of(&full.u)?, full.u.data().to_vec()),
        s: (dyn_space_of(&full.s)?, full.s.data().to_vec()),
        vh: (dyn_space_of(&full.vh)?, full.vh.data().to_vec()),
        singular_values: full.singular_values,
    };
    let out = truncate_svd_dyn(rule, full_dyn, truncation)?;
    Ok(SvdTrunc {
        u: typed_from_dyn(rule, out.u)?,
        s: typed_from_dyn(rule, out.s)?,
        vh: typed_from_dyn(rule, out.vh)?,
        singular_values: out.singular_values,
        error: out.error,
    })
}

/// Dynamic-rank [`truncate_svd`].
pub(crate) fn truncate_svd_dyn<R, D>(
    rule: &R,
    full: SvdCompactDyn<D>,
    truncation: &Truncation,
) -> Result<SvdTruncDyn<D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let decision = decide_bond_truncation(rule, &full.singular_values, truncation, true);
    if full
        .singular_values
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok(SvdTruncDyn {
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
    let kept_by_sector: HashMap<SectorId, usize> = singular_values
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();

    let kept_of = |sector: SectorId| -> usize { kept_by_sector.get(&sector).copied().unwrap_or(0) };

    let bond_axis = full.u.0.nout();
    let u_factor = sliced_bond_tensor(rule, &full.u.0, &full.u.1, bond_axis, &kept_of)?;
    let vh_factor = sliced_bond_tensor(rule, &full.vh.0, &full.vh.1, 0, &kept_of)?;
    let s_factor = diagonal_bond_tensor_dyn(rule, &singular_values, &D::from_real)?;
    Ok(SvdTruncDyn {
        u: u_factor,
        s: s_factor,
        vh: vh_factor,
        singular_values,
        error: decision.error,
    })
}

/// Rebuilds a factor with the bond leg (`axis`) shrunk to the kept prefix per
/// coupled sector, copying leading bond states blockwise.
fn sliced_bond_tensor<R, D>(
    rule: &R,
    source_space: &DynamicFusionMapSpace,
    source_data: &[D],
    axis: usize,
    kept_of: &dyn Fn(SectorId) -> usize,
) -> Result<DynFactor<D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let nout = source_space.nout();
    let source_structure = Arc::clone(source_space.structure());

    // The bond leg carries exactly the kept sectors.
    let kept_sectors: Vec<SectorId> = {
        let homspace = source_space.homspace();
        let leg = if axis < nout {
            &homspace.codomain().legs()[axis]
        } else {
            &homspace.domain().legs()[axis - nout]
        };
        leg.sectors()
            .iter()
            .copied()
            .filter(|&sector| kept_of(sector) > 0)
            .collect()
    };
    let bond_leg = SectorLeg::new(
        kept_sectors.iter().map(|&sector| (sector, kept_of(sector))),
        false,
    );
    let homspace = source_space.homspace();
    let new_hom = if axis < nout {
        let mut codomain_legs = homspace.codomain().legs().to_vec();
        codomain_legs[axis] = bond_leg;
        FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain_legs),
            homspace.domain().clone(),
        )
    } else {
        let mut domain_legs = homspace.domain().legs().to_vec();
        domain_legs[axis - nout] = bond_leg;
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
            let bond_tree = if axis < nout {
                key.codomain_tree()
            } else {
                key.domain_tree()
            };
            shape[axis] = kept_of(coupled_of(rule, bond_tree));
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;

    let space = DynamicFusionMapSpace::from_degeneracy_shapes(rule, new_hom, shapes)?;
    let len = space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); len];

    let sliced_structure = Arc::clone(space.structure());
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
        copy_matching_block_prefix(
            source_data,
            &old_strides,
            old_offset,
            &mut data,
            &new_strides,
            new_offset,
            &shape,
        );
    }
    Ok((space, data))
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

struct SectorRank {
    sector: SectorId,
    kept: usize,
}

fn build_left_right_spaces<R, D>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
    matricizations: &[SectorMatricization<D>],
    ranks: &[SectorRank],
) -> Result<(DynamicFusionMapSpace, DynamicFusionMapSpace), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rank_by_sector: HashMap<SectorId, usize> =
        ranks.iter().map(|rank| (rank.sector, rank.kept)).collect();
    let matrix_by_sector = matricization_map(matricizations);
    let sector_rank =
        |sector: SectorId| -> usize { rank_by_sector.get(&sector).copied().unwrap_or(0) };

    let new_leg = SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false);

    let left_hom = FusionTreeHomSpace::new(
        homspace.codomain().clone(),
        FusionProductSpace::new([new_leg.clone()]),
    );
    let left_keys = left_hom.fusion_tree_keys(rule);
    let left_shapes = left_keys
        .iter()
        .map(|key| {
            let sector = coupled_of(rule, key.codomain_tree());
            let mut shape = row_shape_of(&matrix_by_sector, sector, key.codomain_tree())?;
            shape.push(sector_rank(sector));
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    let left_space = DynamicFusionMapSpace::from_degeneracy_shapes(rule, left_hom, left_shapes)?;

    let right_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg]),
        homspace.domain().clone(),
    );
    let right_keys = right_hom.fusion_tree_keys(rule);
    let right_shapes = right_keys
        .iter()
        .map(|key| {
            let sector = coupled_of(rule, key.domain_tree());
            let mut shape = vec![sector_rank(sector)];
            shape.extend(col_shape_of(&matrix_by_sector, sector, key.domain_tree())?);
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    let right_space = DynamicFusionMapSpace::from_degeneracy_shapes(rule, right_hom, right_shapes)?;

    Ok((left_space, right_space))
}

/// Builds the `(codomain <- W, W <- domain)` factor pair shared by SVD and
/// the orthogonal factorizations, in the coupled-sector matrix layout.
fn build_left_right_pair<R, D>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
    matricizations: &[SectorMatricization<D>],
    pairs: &[FactorPair<D>],
) -> Result<(DynFactor<D>, DynFactor<D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let ranks = pairs
        .iter()
        .map(|pair| SectorRank {
            sector: pair.sector,
            kept: pair.kept,
        })
        .collect::<Vec<_>>();
    let (left_space, right_space) =
        build_left_right_spaces(rule, homspace, matricizations, &ranks)?;
    let matrix_by_sector = matricization_map(matricizations);
    let pair_by_sector: HashMap<SectorId, &FactorPair<D>> =
        pairs.iter().map(|pair| (pair.sector, pair)).collect();

    let left_len = left_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut left_data = vec![D::zero(); left_len];
    let right_len = right_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut right_data = vec![D::zero(); right_len];

    // Scatter left blocks: element (i.., j) = left[(row_offset + rowmaj(i)) + left_rows * j].
    let left_structure = Arc::clone(left_space.structure());
    for index in 0..left_structure.block_count() {
        let block = left_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = coupled_of(rule, key.codomain_tree());
        let matrix = matricization_of(&matrix_by_sector, sector)?;
        let pair = pair_by_sector
            .get(&sector)
            .copied()
            .expect("factor pair exists for every matricized sector");
        let (row_offset, _) = row_placement(matrix, key.codomain_tree())?;
        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        scatter_matrix_block(
            &mut left_data,
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
    let right_structure = Arc::clone(right_space.structure());
    for index in 0..right_structure.block_count() {
        let block = right_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = coupled_of(rule, key.domain_tree());
        let matrix = matricization_of(&matrix_by_sector, sector)?;
        let pair = pair_by_sector
            .get(&sector)
            .copied()
            .expect("factor pair exists for every matricized sector");
        let (col_offset, _) = col_placement(matrix, key.domain_tree())?;
        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        scatter_matrix_block(
            &mut right_data,
            &shape,
            &strides,
            offset,
            0,
            &pair.right,
            pair.right_leading,
            col_offset,
        );
    }

    Ok(((left_space, left_data), (right_space, right_data)))
}

fn scatter_left_sector_blocks<R, D>(
    rule: &R,
    left_space: &DynamicFusionMapSpace,
    left_data: &mut [D],
    matrix: &SectorMatricization<D>,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let left_structure = Arc::clone(left_space.structure());
    for index in 0..left_structure.block_count() {
        let block = left_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        if coupled_of(rule, key.codomain_tree()) != matrix.sector {
            continue;
        }
        let (row_offset, _) = row_placement(matrix, key.codomain_tree())?;
        scatter_matrix_block(
            left_data,
            block.shape(),
            block.strides(),
            block.offset(),
            block.shape().len() - 1,
            factor,
            factor_rows,
            row_offset,
        );
    }
    Ok(())
}

fn scatter_right_sector_blocks<R, D>(
    rule: &R,
    right_space: &DynamicFusionMapSpace,
    right_data: &mut [D],
    matrix: &SectorMatricization<D>,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let right_structure = Arc::clone(right_space.structure());
    for index in 0..right_structure.block_count() {
        let block = right_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        if coupled_of(rule, key.domain_tree()) != matrix.sector {
            continue;
        }
        let (col_offset, _) = col_placement(matrix, key.domain_tree())?;
        scatter_matrix_block(
            right_data,
            block.shape(),
            block.strides(),
            block.offset(),
            0,
            factor,
            factor_rows,
            col_offset,
        );
    }
    Ok(())
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

/// Dynamic-rank [`EighFull`].
#[derive(Clone, Debug)]
pub struct EighFullDyn<D> {
    pub d: DynFactor<D>,
    pub v: DynFactor<D>,
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

/// Dynamic-rank [`EighTrunc`].
#[derive(Clone, Debug)]
pub struct EighTruncDyn<D> {
    pub d: DynFactor<D>,
    pub v: DynFactor<D>,
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
    let out = eigh_full_dyn(dense, rule, &dyn_space_of(tensor)?, tensor.data())?;
    Ok(EighFull {
        d: typed_from_dyn(rule, out.d)?,
        v: typed_from_dyn(rule, out.v)?,
        eigenvalues: out.eigenvalues,
    })
}

/// Dynamic-rank [`eigh_full`]: the shared core of the Hermitian entries.
pub fn eigh_full_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<EighFullDyn<D>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires an endomorphism (codomain == domain)",
        });
    }
    let matricizations = sector_matricizations(rule, space.structure(), data, space.nout())?;

    let ranks = matricizations
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector,
            kept: matrix.rows,
        })
        .collect::<Vec<_>>();
    let (v_space, _) = build_left_right_spaces(rule, space.homspace(), &matricizations, &ranks)?;
    let v_len = v_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut v_data = vec![D::zero(); v_len];
    let max_n = matricizations
        .iter()
        .map(|matrix| matrix.rows)
        .max()
        .unwrap_or(0);
    let mut values_workspace = vec![D::Real::zero(); max_n];
    let mut vectors_workspace = vec![D::zero(); max_n * max_n];
    let mut sorted_vectors = vec![D::zero(); max_n * max_n];
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(&matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let n = matrix.rows;
        let values_shape = [n];
        let values_strides = [1usize];
        let vectors_shape = [n, n];
        let vectors_strides = [1usize, max_n];
        let values_view =
            DenseViewMut::new(&mut values_workspace, &values_shape, &values_strides, 0)
                .map_err(OperationError::Dense)?;
        let vectors_view =
            DenseViewMut::new(&mut vectors_workspace, &vectors_shape, &vectors_strides, 0)
                .map_err(OperationError::Dense)?;
        dense
            .eigh_into(
                D::dense_read(view),
                D::Real::dense_write(values_view),
                D::dense_write(vectors_view),
            )
            .map_err(OperationError::Dense)?;

        let mut order: Vec<usize> = (0..n).collect();
        // Reorder bond states descending by |eigenvalue| (stable on ties).
        order.sort_by(|&a, &b| {
            let a_value: f64 = values_workspace[a].into();
            let b_value: f64 = values_workspace[b].into();
            b_value
                .abs()
                .partial_cmp(&a_value.abs())
                .expect("finite eigenvalues")
                .then(a.cmp(&b))
        });
        let sorted_values: Vec<f64> = order
            .iter()
            .map(|&index| values_workspace[index].into())
            .collect();
        for (position, &index) in order.iter().enumerate() {
            let dst_start = position * n;
            let src_start = index * max_n;
            sorted_vectors[dst_start..dst_start + n]
                .copy_from_slice(&vectors_workspace[src_start..src_start + n]);
        }
        eigenvector_gauge(&mut sorted_vectors, n, n, n);
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted_values,
        });
        scatter_left_sector_blocks(rule, &v_space, &mut v_data, matrix, &sorted_vectors, n)?;
    }

    let d_factor = diagonal_bond_tensor_dyn(rule, &eigenvalues, &D::from_real)?;
    Ok(EighFullDyn {
        d: d_factor,
        v: (v_space, v_data),
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
    let out = eigh_trunc_dyn(
        dense,
        rule,
        &dyn_space_of(tensor)?,
        tensor.data(),
        truncation,
    )?;
    Ok(EighTrunc {
        d: typed_from_dyn(rule, out.d)?,
        v: typed_from_dyn(rule, out.v)?,
        eigenvalues: out.eigenvalues,
        error: out.error,
    })
}

/// Dynamic-rank [`eigh_trunc`].
pub fn eigh_trunc_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
    truncation: &Truncation,
) -> Result<EighTruncDyn<D>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let full = eigh_full_dyn(dense, rule, space, data)?;
    if matches!(truncation, Truncation::Full) {
        return Ok(EighTruncDyn {
            d: full.d,
            v: full.v,
            eigenvalues: full.eigenvalues,
            error: 0.0,
        });
    }
    let decision = decide_bond_truncation(rule, &full.eigenvalues, truncation, false);
    if full
        .eigenvalues
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok(EighTruncDyn {
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
    let kept_by_sector: HashMap<SectorId, usize> = eigenvalues
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();
    let kept_of = |sector: SectorId| -> usize { kept_by_sector.get(&sector).copied().unwrap_or(0) };
    let bond_axis = full.v.0.nout();
    let v_factor = sliced_bond_tensor(rule, &full.v.0, &full.v.1, bond_axis, &kept_of)?;
    let d_factor = diagonal_bond_tensor_dyn(rule, &eigenvalues, &D::from_real)?;
    Ok(EighTruncDyn {
        d: d_factor,
        v: v_factor,
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

/// Dynamic-rank [`SvdFull`].
#[derive(Clone, Debug)]
pub struct SvdFullDyn<D> {
    pub u: DynFactor<D>,
    pub s: DynFactor<D>,
    pub vh: DynFactor<D>,
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
    let out = svd_full_dyn(dense, rule, &dyn_space_of(tensor)?, tensor.data())?;
    Ok(SvdFull {
        u: typed_from_dyn(rule, out.u)?,
        s: typed_from_dyn(rule, out.s)?,
        vh: typed_from_dyn(rule, out.vh)?,
        singular_values: out.singular_values,
    })
}

/// Dynamic-rank [`svd_full`].
pub fn svd_full_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<SvdFullDyn<D>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let matricizations = sector_matricizations(rule, space.structure(), data, space.nout())?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    let mut singular_values = Vec::with_capacity(matricizations.len());
    let mut col_dims: Vec<(SectorId, usize)> = Vec::new();
    let max_rows = matricizations
        .iter()
        .map(|matrix| matrix.rows)
        .max()
        .unwrap_or(0);
    let max_cols = matricizations
        .iter()
        .map(|matrix| matrix.cols)
        .max()
        .unwrap_or(0);
    let max_rank = matricizations
        .iter()
        .map(|matrix| matrix.rows.min(matrix.cols))
        .max()
        .unwrap_or(0);
    let mut u_workspace = vec![D::zero(); max_rows * max_rank];
    let mut s_workspace = vec![D::Real::zero(); max_rank];
    let mut vt_workspace = vec![D::zero(); max_rank * max_cols];
    for matrix in &matricizations {
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(&matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let rank = matrix.rows.min(matrix.cols);
        let u_shape = [matrix.rows, rank];
        let u_strides = [1usize, max_rows];
        let s_shape = [rank];
        let s_strides = [1usize];
        let vt_shape = [rank, matrix.cols];
        let vt_strides = [1usize, max_rank];
        let u_view = DenseViewMut::new(&mut u_workspace, &u_shape, &u_strides, 0)
            .map_err(OperationError::Dense)?;
        let s_view = DenseViewMut::new(&mut s_workspace, &s_shape, &s_strides, 0)
            .map_err(OperationError::Dense)?;
        let vt_view = DenseViewMut::new(&mut vt_workspace, &vt_shape, &vt_strides, 0)
            .map_err(OperationError::Dense)?;
        dense
            .svd_into(
                D::dense_read(view),
                D::dense_write(u_view),
                D::Real::dense_write(s_view),
                D::dense_write(vt_view),
            )
            .map_err(OperationError::Dense)?;
        let s_values = s_workspace[..rank]
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<_>>();
        let mut u_thin = vec![D::zero(); matrix.rows * rank];
        let mut vt_thin = vec![D::zero(); rank * matrix.cols];
        copy_col_major_strided(
            &u_workspace,
            matrix.rows,
            rank,
            max_rows,
            &mut u_thin,
            matrix.rows,
        );
        copy_col_major_strided(
            &vt_workspace,
            rank,
            matrix.cols,
            max_rank,
            &mut vt_thin,
            rank,
        );

        let mut u_full = orthonormal_completion(dense, &u_thin, matrix.rows, rank)?;
        // V columns are the adjoint rows of Vh; complete V (n x rank) to
        // n x n, then store Vh = V^H.
        let v_thin = adjoint_col_major(&vt_thin, rank, matrix.cols);
        let v_full = orthonormal_completion(dense, &v_thin, matrix.cols, rank)?;
        let mut vh_full = adjoint_col_major(&v_full, matrix.cols, matrix.cols);
        svd_full_gauge(
            &mut u_full,
            matrix.rows,
            matrix.rows,
            &mut vh_full,
            matrix.cols,
            matrix.cols,
        );

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

    let cols_by_sector: HashMap<SectorId, usize> = col_dims.into_iter().collect();
    let cols_of = |sector: SectorId| {
        cols_by_sector
            .get(&sector)
            .copied()
            .expect("column dimension recorded per sector")
    };
    // The left/right bond legs differ in the full SVD (rows vs columns), so
    // build the two factors with separate bond dimensions.
    let (u_factor, _) = build_left_right_pair(
        rule,
        space.homspace(),
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
    let (_, vh_factor) = build_left_right_pair(
        rule,
        space.homspace(),
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
    let rows_by_sector: HashMap<SectorId, usize> = pairs
        .iter()
        .map(|pair| (pair.sector, pair.left_rows))
        .collect();
    let rows_of = |sector: SectorId| rows_by_sector.get(&sector).copied().unwrap_or(0);
    let s_factor = rectangular_diagonal_bond_tensor(rule, &singular_values, &rows_of, &cols_of)?;
    Ok(SvdFullDyn {
        u: u_factor,
        s: s_factor,
        vh: vh_factor,
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
    let mut q = vec![D::zero(); rows * rows];
    let mut r = vec![D::zero(); rows * (rank + rows)];
    qr_into_workspace(
        dense,
        &augmented,
        rows,
        rank + rows,
        rows,
        &mut q,
        rows,
        rows,
        rows,
        &mut r,
        rows,
        rank + rows,
        rows,
    )?;
    let mut full = vec![D::zero(); rows * rows];
    full[..rows * rank].copy_from_slice(thin);
    full[rows * rank..].copy_from_slice(&q[rows * rank..rows * rows]);
    Ok(full)
}

/// Rectangular diagonal `W_row <- W_col` bond factor (the `S` of the full
/// SVD): per sector shape `[rows, cols]` with the spectrum on the diagonal.
fn rectangular_diagonal_bond_tensor<R, D>(
    rule: &R,
    spectra: &[SectorSpectrum],
    rows_of: &dyn Fn(SectorId) -> usize,
    cols_of: &dyn Fn(SectorId) -> usize,
) -> Result<DynFactor<D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let row_leg = SectorLeg::new(
        spectra
            .iter()
            .map(|entry| (entry.sector, rows_of(entry.sector))),
        false,
    );
    let col_leg = SectorLeg::new(
        spectra
            .iter()
            .map(|entry| (entry.sector, cols_of(entry.sector))),
        false,
    );
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
    let space = DynamicFusionMapSpace::from_degeneracy_shapes(rule, homspace, shapes)?;
    let len = space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); len];
    let structure = Arc::clone(space.structure());
    let spectrum_by_sector: HashMap<SectorId, &SectorSpectrum> =
        spectra.iter().map(|entry| (entry.sector, entry)).collect();
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree
            .codomain_tree()
            .coupled()
            .unwrap_or_else(|| tree.codomain_tree().uncoupled()[0]);
        let Some(&entry) = spectrum_by_sector.get(&sector) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        for position in 0..count {
            let Some(&value) = entry.values.get(position) else {
                break;
            };
            data[offset + position * (strides[0] + strides[1])] = D::from_real(value);
        }
    }
    Ok((space, data))
}

/// Positive-diagonal gauge (MatrixAlgebraKit `positive = true`, the default
/// of the Householder QR/LQ algorithms since MAK 0.6.8 / TensorKit 0.17):
/// absorbs the unitary phase `D = diag(phase(R_jj))` into `Q`, i.e.
/// `Q -> Q * D`, `R -> D^H * R`, leaving `Q * R` unchanged with real
/// non-negative `R_jj`. Zero diagonal entries keep phase `1`, exactly like
/// MAK `sign_safe` (no epsilon threshold).
///
/// `q` is column-major `q_rows x nq` (`nq >= min(r_rows, r_cols)`), `r` is
/// column-major `r_rows x r_cols`.
pub(crate) fn positive_diagonal_gauge<D: FactorScalar>(
    q: &mut [D],
    q_rows: usize,
    r: &mut [D],
    r_rows: usize,
    r_cols: usize,
) {
    positive_diagonal_gauge_strided(q, q_rows, q_rows, r, r_rows, r_rows, r_cols);
}

fn positive_diagonal_gauge_strided<D: FactorScalar>(
    q: &mut [D],
    q_rows: usize,
    q_leading: usize,
    r: &mut [D],
    r_rows: usize,
    r_leading: usize,
    r_cols: usize,
) {
    for j in 0..r_rows.min(r_cols) {
        let z = r[j + r_leading * j].widen_complex();
        let norm = z.norm();
        if norm == 0.0 {
            continue; // phase 1: nothing to scale
        }
        let phase = D::from_complex64(z / norm);
        let conj_phase = FactorScalar::adjoint(phase);
        for row in 0..q_rows {
            let index = row + q_leading * j;
            q[index] = q[index] * phase;
        }
        for col in 0..r_cols {
            let index = j + r_leading * col;
            r[index] = conj_phase * r[index];
        }
    }
}

pub(crate) fn svd_compact_gauge<D: FactorScalar>(
    u: &mut [D],
    u_rows: usize,
    u_leading: usize,
    vh: &mut [D],
    vh_rows: usize,
    vh_cols: usize,
    vh_leading: usize,
) {
    for j in 0..vh_rows {
        let (phase, needs_scaling) = phase_of_largest_abs_col(u, u_rows, u_leading, j);
        if needs_scaling {
            scale_col(u, u_rows, u_leading, j, FactorScalar::adjoint(phase));
            scale_row(vh, vh_cols, vh_leading, j, phase);
        }
    }
}

pub(crate) fn svd_full_gauge<D: FactorScalar>(
    u: &mut [D],
    u_rows: usize,
    u_leading: usize,
    vh: &mut [D],
    vh_rows: usize,
    vh_cols: usize,
) {
    let paired = u_leading.min(vh_rows);
    for j in 0..u_leading.max(vh_rows) {
        if j < paired {
            let (phase, needs_scaling) = phase_of_largest_abs_col(u, u_rows, u_leading, j);
            if needs_scaling {
                scale_col(u, u_rows, u_leading, j, FactorScalar::adjoint(phase));
                scale_row(vh, vh_cols, vh_rows, j, phase);
            }
        } else if j < u_leading {
            let (phase, needs_scaling) = phase_of_largest_abs_col(u, u_rows, u_leading, j);
            if needs_scaling {
                scale_col(u, u_rows, u_leading, j, FactorScalar::adjoint(phase));
            }
        } else {
            let (phase, needs_scaling) = phase_of_largest_abs_row(vh, vh_cols, vh_rows, j);
            if needs_scaling {
                scale_row(vh, vh_cols, vh_rows, j, FactorScalar::adjoint(phase));
            }
        }
    }
}

pub(crate) fn eigenvector_gauge<D: FactorScalar>(
    vectors: &mut [D],
    rows: usize,
    leading: usize,
    cols: usize,
) {
    for j in 0..cols {
        let (phase, needs_scaling) = phase_of_largest_abs_col(vectors, rows, leading, j);
        if needs_scaling {
            scale_col(vectors, rows, leading, j, FactorScalar::adjoint(phase));
        }
    }
}

fn phase_of_largest_abs_col<D: FactorScalar>(
    data: &[D],
    rows: usize,
    leading: usize,
    col: usize,
) -> (D, bool) {
    let mut best = Complex64::new(0.0, 0.0);
    let mut best_norm_sqr = 0.0;
    for row in 0..rows {
        let value = data[row + leading * col].widen_complex();
        let norm_sqr = value.norm_sqr();
        if best_norm_sqr < norm_sqr {
            best = value;
            best_norm_sqr = norm_sqr;
        }
    }
    unit_phase(best, best_norm_sqr)
}

fn phase_of_largest_abs_row<D: FactorScalar>(
    data: &[D],
    cols: usize,
    leading: usize,
    row: usize,
) -> (D, bool) {
    let mut best = Complex64::new(0.0, 0.0);
    let mut best_norm_sqr = 0.0;
    for col in 0..cols {
        let value = data[row + leading * col].widen_complex();
        let norm_sqr = value.norm_sqr();
        if best_norm_sqr < norm_sqr {
            best = value;
            best_norm_sqr = norm_sqr;
        }
    }
    unit_phase(best, best_norm_sqr)
}

fn unit_phase<D: FactorScalar>(value: Complex64, norm_sqr: f64) -> (D, bool) {
    if norm_sqr == 0.0 || (value.im == 0.0 && value.re >= 0.0) {
        (D::from_real(1.0), false)
    } else {
        (D::from_complex64(value / norm_sqr.sqrt()), true)
    }
}

fn scale_col<D: FactorScalar>(data: &mut [D], rows: usize, leading: usize, col: usize, phase: D) {
    for row in 0..rows {
        let index = row + leading * col;
        data[index] = data[index] * phase;
    }
}

fn scale_row<D: FactorScalar>(data: &mut [D], cols: usize, leading: usize, row: usize, phase: D) {
    for col in 0..cols {
        let index = row + leading * col;
        data[index] = data[index] * phase;
    }
}

/// Full QR `t = Q * R` (MatrixAlgebraKit `qr_full`): per sector `Q` is the
/// square `m x m` unitary and `R` the upper-trapezoidal `m x n`, obtained
/// from one economy QR of the augmented `[A | I]` on the dense boundary.
/// The positive-diagonal gauge is applied (MAK / TensorKit 0.17 default).
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
    let (q, r) = qr_full_dyn(dense, rule, &dyn_space_of(tensor)?, tensor.data())?;
    Ok((typed_from_dyn(rule, q)?, typed_from_dyn(rule, r)?))
}

/// Dynamic-rank [`qr_full`].
pub fn qr_full_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynFactor<D>, DynFactor<D>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let matricizations = sector_matricizations(rule, space.structure(), data, space.nout())?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    let max_rows = matricizations
        .iter()
        .map(|matrix| matrix.rows)
        .max()
        .unwrap_or(0);
    let max_cols = matricizations
        .iter()
        .map(|matrix| matrix.cols)
        .max()
        .unwrap_or(0);
    let mut q_workspace = vec![D::zero(); max_rows * max_rows];
    let mut r_workspace = vec![D::zero(); max_rows * (max_rows + max_cols)];
    for matrix in &matricizations {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let mut augmented = vec![D::zero(); rows * (cols + rows)];
        augmented[..rows * cols].copy_from_slice(&matrix.data);
        for row in 0..rows {
            augmented[rows * cols + row * rows + row] = D::one();
        }
        qr_into_workspace(
            dense,
            &augmented,
            rows,
            cols + rows,
            rows,
            &mut q_workspace,
            rows,
            rows,
            max_rows,
            &mut r_workspace,
            rows,
            cols + rows,
            max_rows,
        )?;
        let mut q = vec![D::zero(); rows * rows];
        let mut r = vec![D::zero(); rows * cols];
        copy_col_major_strided(&q_workspace, rows, rows, max_rows, &mut q, rows);
        copy_col_major_strided(&r_workspace, rows, cols, max_rows, &mut r, rows);
        positive_diagonal_gauge(&mut q, rows, &mut r, rows, cols);
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rows,
            left: q,
            left_rows: rows,
            right: r,
            right_leading: rows,
        });
    }

    build_left_right_pair(rule, space.homspace(), &matricizations, &pairs)
}

/// Full LQ `t = L * Q` (MatrixAlgebraKit `lq_full`): per sector `L` is the
/// lower-trapezoidal `m x n` and `Q` the square `n x n` unitary, via the full
/// QR of the transposed sector matrices.
/// The positive-diagonal gauge is applied (MAK / TensorKit 0.17 default).
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
    let (l, q) = lq_full_dyn(dense, rule, &dyn_space_of(tensor)?, tensor.data())?;
    Ok((typed_from_dyn(rule, l)?, typed_from_dyn(rule, q)?))
}

/// Dynamic-rank [`lq_full`].
pub fn lq_full_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynFactor<D>, DynFactor<D>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let matricizations = sector_matricizations(rule, space.structure(), data, space.nout())?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    let max_rows = matricizations
        .iter()
        .map(|matrix| matrix.rows)
        .max()
        .unwrap_or(0);
    let max_cols = matricizations
        .iter()
        .map(|matrix| matrix.cols)
        .max()
        .unwrap_or(0);
    let mut q_prime_workspace = vec![D::zero(); max_cols * max_cols];
    let mut r_prime_workspace = vec![D::zero(); max_cols * (max_rows + max_cols)];
    for matrix in &matricizations {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let transposed = adjoint_col_major(&matrix.data, rows, cols);
        let mut augmented = vec![D::zero(); cols * (rows + cols)];
        augmented[..cols * rows].copy_from_slice(&transposed);
        for row in 0..cols {
            augmented[cols * rows + row * cols + row] = D::one();
        }
        qr_into_workspace(
            dense,
            &augmented,
            cols,
            rows + cols,
            cols,
            &mut q_prime_workspace,
            cols,
            cols,
            max_cols,
            &mut r_prime_workspace,
            cols,
            rows + cols,
            max_cols,
        )?;
        let mut q_prime = vec![D::zero(); cols * cols];
        let mut r_prime = vec![D::zero(); cols * rows];
        copy_col_major_strided(&q_prime_workspace, cols, cols, max_cols, &mut q_prime, cols);
        copy_col_major_strided(&r_prime_workspace, cols, rows, max_cols, &mut r_prime, cols);
        // Gauge the QR of t^H; L = R'^H then has a real non-negative diagonal.
        positive_diagonal_gauge(&mut q_prime, cols, &mut r_prime, cols, rows);
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: cols,
            // L = R'^H : rows x cols (lower trapezoidal).
            left: adjoint_col_major(&r_prime, cols, rows),
            left_rows: rows,
            // Q = Q'^H : cols x cols.
            right: adjoint_col_major(&q_prime, cols, cols),
            right_leading: cols,
        });
    }

    build_left_right_pair(rule, space.homspace(), &matricizations, &pairs)
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

/// Dynamic-rank [`EigFull`].
#[derive(Clone, Debug)]
pub struct EigFullDyn<D: FactorScalar> {
    pub d: DynFactor<D::Eig>,
    pub v: DynFactor<D::Eig>,
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

/// Dynamic-rank [`EigTrunc`].
#[derive(Clone, Debug)]
pub struct EigTruncDyn<D: FactorScalar> {
    pub d: DynFactor<D::Eig>,
    pub v: DynFactor<D::Eig>,
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
    let out = eig_full_dyn::<E, R, D>(dense, rule, &dyn_space_of(tensor)?, tensor.data())?;
    Ok(EigFull {
        d: typed_from_dyn(rule, out.d)?,
        v: typed_from_dyn(rule, out.v)?,
        eigenvalues: out.eigenvalues,
    })
}

/// Dynamic-rank [`eig_full`].
pub fn eig_full_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<EigFullDyn<D>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eig requires an endomorphism (codomain == domain)",
        });
    }
    let matricizations = sector_matricizations(rule, space.structure(), data, space.nout())?;

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
        eigenvector_gauge(&mut sorted_vectors, n, n, n);
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
    let (v_factor, _) =
        build_left_right_pair(rule, space.homspace(), &complex_matricizations, &pairs)?;
    let d_factor = diagonal_bond_tensor_dyn(
        rule,
        &eigenvalues,
        &<D::Eig as FactorScalar>::from_complex64,
    )?;
    Ok(EigFullDyn {
        d: d_factor,
        v: v_factor,
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
    let out = eig_trunc_dyn::<E, R, D>(
        dense,
        rule,
        &dyn_space_of(tensor)?,
        tensor.data(),
        truncation,
    )?;
    Ok(EigTrunc {
        d: typed_from_dyn(rule, out.d)?,
        v: typed_from_dyn(rule, out.v)?,
        eigenvalues: out.eigenvalues,
        error: out.error,
    })
}

/// Dynamic-rank [`eig_trunc`].
pub fn eig_trunc_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
    truncation: &Truncation,
) -> Result<EigTruncDyn<D>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let full = eig_full_dyn::<E, R, D>(dense, rule, space, data)?;
    if matches!(truncation, Truncation::Full) {
        return Ok(EigTruncDyn {
            d: full.d,
            v: full.v,
            eigenvalues: full.eigenvalues,
            error: 0.0,
        });
    }
    let decision = decide_bond_truncation(rule, &full.eigenvalues, truncation, false);
    if full
        .eigenvalues
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok(EigTruncDyn {
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
    let kept_by_sector: HashMap<SectorId, usize> = eigenvalues
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();
    let kept_of = |sector: SectorId| -> usize { kept_by_sector.get(&sector).copied().unwrap_or(0) };
    let bond_axis = full.v.0.nout();
    let v_factor = sliced_bond_tensor(rule, &full.v.0, &full.v.1, bond_axis, &kept_of)?;
    let d_factor = diagonal_bond_tensor_dyn(
        rule,
        &eigenvalues,
        &<D::Eig as FactorScalar>::from_complex64,
    )?;
    Ok(EigTruncDyn {
        d: d_factor,
        v: v_factor,
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
    eigh_vals_dyn(dense, rule, &dyn_space_of(tensor)?, tensor.data())
}

/// Dynamic-rank [`eigh_vals`].
pub fn eigh_vals_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    eigh_full_dyn(dense, rule, space, data).map(|eigh| eigh.eigenvalues)
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
    eig_vals_dyn::<E, R, D>(dense, rule, &dyn_space_of(tensor)?, tensor.data())
}

/// Dynamic-rank [`eig_vals`].
pub fn eig_vals_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<Vec<SectorSpectrum<Complex64>>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    eig_full_dyn::<E, R, D>(dense, rule, space, data).map(|eig| eig.eigenvalues)
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
    let out = left_null_dyn(dense, rule, &dyn_space_of(tensor)?, tensor.data())?;
    typed_from_dyn(rule, out)
}

/// Dynamic-rank [`left_null`].
pub fn left_null_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<DynFactor<D>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let matricizations = sector_matricizations(rule, space.structure(), data, space.nout())?;

    let mut pairs = Vec::new();
    let max_rows = matricizations
        .iter()
        .map(|matrix| matrix.rows)
        .max()
        .unwrap_or(0);
    let max_cols = matricizations
        .iter()
        .map(|matrix| matrix.cols)
        .max()
        .unwrap_or(0);
    let mut q_workspace = vec![D::zero(); max_rows * max_rows];
    let mut r_workspace = vec![D::zero(); max_rows * (max_rows + max_cols)];
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
        qr_into_workspace(
            dense,
            &augmented,
            rows,
            cols + rows,
            rows,
            &mut q_workspace,
            rows,
            rows,
            max_rows,
            &mut r_workspace,
            rows,
            cols + rows,
            max_rows,
        )?;
        let null_dim = rows - rank;
        let mut left = vec![D::zero(); rows * null_dim];
        copy_col_major_strided(
            &q_workspace[max_rows * rank..],
            rows,
            null_dim,
            max_rows,
            &mut left,
            rows,
        );
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: null_dim,
            // Null columns are the trailing full-Q columns (contiguous in the
            // column-major layout).
            left,
            left_rows: rows,
            // Discarded placeholder for the pair builder.
            right: vec![D::zero(); null_dim * cols],
            right_leading: null_dim,
        });
    }

    let (null_factor, _) = build_left_right_pair(rule, space.homspace(), &matricizations, &pairs)?;
    Ok(null_factor)
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
    let out = right_null_dyn(dense, rule, &dyn_space_of(tensor)?, tensor.data())?;
    typed_from_dyn(rule, out)
}

/// Dynamic-rank [`right_null`].
pub fn right_null_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<DynFactor<D>, OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let matricizations = sector_matricizations(rule, space.structure(), data, space.nout())?;

    let mut pairs = Vec::new();
    let max_rows = matricizations
        .iter()
        .map(|matrix| matrix.rows)
        .max()
        .unwrap_or(0);
    let max_cols = matricizations
        .iter()
        .map(|matrix| matrix.cols)
        .max()
        .unwrap_or(0);
    let mut q_prime_workspace = vec![D::zero(); max_cols * max_cols];
    let mut r_prime_workspace = vec![D::zero(); max_cols * (max_rows + max_cols)];
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
        qr_into_workspace(
            dense,
            &augmented,
            cols,
            rows + cols,
            cols,
            &mut q_prime_workspace,
            cols,
            cols,
            max_cols,
            &mut r_prime_workspace,
            cols,
            rows + cols,
            max_cols,
        )?;
        let null_dim = cols - rank;
        let mut right = vec![D::zero(); null_dim * cols];
        adjoint_col_major_strided_into(
            &q_prime_workspace[max_cols * rank..],
            cols,
            null_dim,
            max_cols,
            &mut right,
            null_dim,
        );
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: null_dim,
            // Discarded placeholder for the pair builder.
            left: vec![D::zero(); rows * null_dim],
            left_rows: rows,
            // Null rows are the adjoints of the trailing Q' columns.
            right,
            right_leading: null_dim,
        });
    }

    let (_, null_factor) = build_left_right_pair(rule, space.homspace(), &matricizations, &pairs)?;
    Ok(null_factor)
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
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let (w, p) = left_polar_dyn(dense, context, rule, &dyn_space_of(tensor)?, tensor.data())?;
    Ok((typed_from_dyn(rule, w)?, typed_from_dyn(rule, p)?))
}

/// Dynamic-rank [`left_polar`].
pub fn left_polar_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynFactor<D>, DynFactor<D>), OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let svd = svd_compact_dyn(dense, rule, space, data)?;
    let isometry =
        crate::compose::compose_dyn(context, rule, (&svd.u.0, &svd.u.1), (&svd.vh.0, &svd.vh.1))?;
    let v = tenet_tensors::adjoint_dyn(rule, &svd.vh.0, &svd.vh.1)?;
    let vs = crate::compose::compose_dyn(context, rule, (&v.0, &v.1), (&svd.s.0, &svd.s.1))?;
    let positive =
        crate::compose::compose_dyn(context, rule, (&vs.0, &vs.1), (&svd.vh.0, &svd.vh.1))?;
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
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let (p, w) = right_polar_dyn(dense, context, rule, &dyn_space_of(tensor)?, tensor.data())?;
    Ok((typed_from_dyn(rule, p)?, typed_from_dyn(rule, w)?))
}

/// Dynamic-rank [`right_polar`].
pub fn right_polar_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynFactor<D>, DynFactor<D>), OperationError>
where
    E: DenseExecutor,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let svd = svd_compact_dyn(dense, rule, space, data)?;
    let uh = tenet_tensors::adjoint_dyn(rule, &svd.u.0, &svd.u.1)?;
    let us =
        crate::compose::compose_dyn(context, rule, (&svd.u.0, &svd.u.1), (&svd.s.0, &svd.s.1))?;
    let positive = crate::compose::compose_dyn(context, rule, (&us.0, &us.1), (&uh.0, &uh.1))?;
    let isometry =
        crate::compose::compose_dyn(context, rule, (&svd.u.0, &svd.u.1), (&svd.vh.0, &svd.vh.1))?;
    Ok((positive, isometry))
}

/// Compact QR `t = Q * R` (MatrixAlgebraKit `qr_compact`):
/// `Q : codomain <- W` has orthonormal columns per coupled sector and
/// `R : W <- domain` with per-sector bond `min(rows, cols)`. The
/// positive-diagonal gauge is applied (MAK / TensorKit 0.17 default
/// `positive = true`): `R`'s diagonal is real non-negative per sector.
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
    let (q, r) = qr_compact_dyn(dense, rule, &dyn_space_of(tensor)?, tensor.data())?;
    Ok((typed_from_dyn(rule, q)?, typed_from_dyn(rule, r)?))
}

/// Dynamic-rank [`qr_compact`].
pub fn qr_compact_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynFactor<D>, DynFactor<D>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let matricizations = sector_matricizations(rule, space.structure(), data, space.nout())?;

    let ranks = matricizations
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector,
            kept: matrix.rows.min(matrix.cols),
        })
        .collect::<Vec<_>>();
    let (q_space, r_space) =
        build_left_right_spaces(rule, space.homspace(), &matricizations, &ranks)?;
    let q_len = q_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut q_data = vec![D::zero(); q_len];
    let r_len = r_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut r_data = vec![D::zero(); r_len];
    let max_rows = matricizations
        .iter()
        .map(|matrix| matrix.rows)
        .max()
        .unwrap_or(0);
    let max_cols = matricizations
        .iter()
        .map(|matrix| matrix.cols)
        .max()
        .unwrap_or(0);
    let max_rank = ranks.iter().map(|rank| rank.kept).max().unwrap_or(0);
    let mut q_workspace = vec![D::zero(); max_rows * max_rank];
    let mut r_workspace = vec![D::zero(); max_rank * max_cols];
    for matrix in &matricizations {
        let rank = matrix.rows.min(matrix.cols);
        qr_into_workspace(
            dense,
            &matrix.data,
            matrix.rows,
            matrix.cols,
            matrix.rows,
            &mut q_workspace,
            matrix.rows,
            rank,
            max_rows,
            &mut r_workspace,
            rank,
            matrix.cols,
            max_rank,
        )?;
        positive_diagonal_gauge_strided(
            &mut q_workspace,
            matrix.rows,
            max_rows,
            &mut r_workspace,
            rank,
            max_rank,
            matrix.cols,
        );
        scatter_left_sector_blocks(rule, &q_space, &mut q_data, matrix, &q_workspace, max_rows)?;
        scatter_right_sector_blocks(rule, &r_space, &mut r_data, matrix, &r_workspace, max_rank)?;
    }

    Ok(((q_space, q_data), (r_space, r_data)))
}

/// Compact LQ `t = L * Q` (MatrixAlgebraKit `lq_compact`, via the QR of the
/// transposed sector matrices): `Q : W <- domain` has orthonormal rows per
/// coupled sector and `L : codomain <- W`. The positive-diagonal gauge is
/// applied (MAK / TensorKit 0.17 default `positive = true`): `L`'s diagonal
/// is real non-negative per sector.
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
    let (l, q) = lq_compact_dyn(dense, rule, &dyn_space_of(tensor)?, tensor.data())?;
    Ok((typed_from_dyn(rule, l)?, typed_from_dyn(rule, q)?))
}

/// Dynamic-rank [`lq_compact`].
pub fn lq_compact_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynFactor<D>, DynFactor<D>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let matricizations = sector_matricizations(rule, space.structure(), data, space.nout())?;

    let ranks = matricizations
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector,
            kept: matrix.rows.min(matrix.cols),
        })
        .collect::<Vec<_>>();
    let (l_space, q_space) =
        build_left_right_spaces(rule, space.homspace(), &matricizations, &ranks)?;
    let l_len = l_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut l_data = vec![D::zero(); l_len];
    let q_len = q_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut q_data = vec![D::zero(); q_len];
    let max_rows = matricizations
        .iter()
        .map(|matrix| matrix.rows)
        .max()
        .unwrap_or(0);
    let max_cols = matricizations
        .iter()
        .map(|matrix| matrix.cols)
        .max()
        .unwrap_or(0);
    let max_rank = ranks.iter().map(|rank| rank.kept).max().unwrap_or(0);
    let mut transposed_workspace = vec![D::zero(); max_cols * max_rows];
    let mut q_prime_workspace = vec![D::zero(); max_cols * max_rank];
    let mut r_prime_workspace = vec![D::zero(); max_rank * max_rows];
    let mut l_workspace = vec![D::zero(); max_rows * max_rank];
    let mut q_workspace = vec![D::zero(); max_rank * max_cols];
    for matrix in &matricizations {
        // QR of the adjoint: t^H = Q' R'  =>  t = R'^H Q'^H = L Q.
        let rank = matrix.rows.min(matrix.cols);
        adjoint_col_major_strided_into(
            &matrix.data,
            matrix.rows,
            matrix.cols,
            matrix.rows,
            &mut transposed_workspace,
            max_cols,
        );
        qr_into_workspace(
            dense,
            &transposed_workspace,
            matrix.cols,
            matrix.rows,
            max_cols,
            &mut q_prime_workspace,
            matrix.cols,
            rank,
            max_cols,
            &mut r_prime_workspace,
            rank,
            matrix.rows,
            max_rank,
        )?;
        // Gauge the QR of t^H; L = R'^H then has a real non-negative diagonal.
        positive_diagonal_gauge_strided(
            &mut q_prime_workspace,
            matrix.cols,
            max_cols,
            &mut r_prime_workspace,
            rank,
            max_rank,
            matrix.rows,
        );
        // L = R'^H : rows x rank; Q = Q'^H : rank x cols.
        adjoint_col_major_strided_into(
            &r_prime_workspace,
            rank,
            matrix.rows,
            max_rank,
            &mut l_workspace,
            max_rows,
        );
        adjoint_col_major_strided_into(
            &q_prime_workspace,
            matrix.cols,
            rank,
            max_cols,
            &mut q_workspace,
            max_rank,
        );
        scatter_left_sector_blocks(rule, &l_space, &mut l_data, matrix, &l_workspace, max_rows)?;
        scatter_right_sector_blocks(rule, &q_space, &mut q_data, matrix, &q_workspace, max_rank)?;
    }

    Ok(((l_space, l_data), (q_space, q_data)))
}

/// Left isometry factorization `t = V * C` (TensorKit 0.17 / MatrixAlgebraKit
/// `left_orth`): `V : codomain <- W` isometric, `C : W <- domain`.
///
/// TensorKit's default `kind = :qr` maps to [`qr_compact`], which applies the
/// positive-diagonal QR gauge (`positive = true`, the MAK default).
pub fn left_orth<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<(TensorMap<D, NOUT, 1>, TensorMap<D, 1, NIN>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    qr_compact(dense, rule, tensor)
}

/// Dynamic-rank [`left_orth`].
pub fn left_orth_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynFactor<D>, DynFactor<D>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    qr_compact_dyn(dense, rule, space, data)
}

/// Right isometry factorization `t = C * Vh` (TensorKit 0.17 /
/// MatrixAlgebraKit `right_orth`): `C : codomain <- W`, `Vh : W <- domain`
/// with orthonormal rows.
///
/// TensorKit's default `kind = :lq` maps to [`lq_compact`], which applies the
/// positive-diagonal LQ gauge (`positive = true`, the MAK default).
pub fn right_orth<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<(TensorMap<D, NOUT, 1>, TensorMap<D, 1, NIN>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    lq_compact(dense, rule, tensor)
}

/// Dynamic-rank [`right_orth`].
pub fn right_orth_dyn<E, R, D>(
    dense: &mut E,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynFactor<D>, DynFactor<D>), OperationError>
where
    E: DenseExecutor,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    lq_compact_dyn(dense, rule, space, data)
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

#[allow(clippy::too_many_arguments)]
fn qr_into_workspace<E, D>(
    dense: &mut E,
    input: &[D],
    input_rows: usize,
    input_cols: usize,
    input_leading: usize,
    q: &mut [D],
    q_rows: usize,
    q_cols: usize,
    q_leading: usize,
    r: &mut [D],
    r_rows: usize,
    r_cols: usize,
    r_leading: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: FactorScalar,
{
    let input_shape = [input_rows, input_cols];
    let input_strides = [1usize, input_leading];
    let q_shape = [q_rows, q_cols];
    let q_strides = [1usize, q_leading];
    let r_shape = [r_rows, r_cols];
    let r_strides = [1usize, r_leading];
    let input_view =
        DenseView::new(input, &input_shape, &input_strides, 0).map_err(OperationError::Dense)?;
    let q_view = DenseViewMut::new(q, &q_shape, &q_strides, 0).map_err(OperationError::Dense)?;
    let r_view = DenseViewMut::new(r, &r_shape, &r_strides, 0).map_err(OperationError::Dense)?;
    dense
        .qr_into(
            D::dense_read(input_view),
            D::dense_write(q_view),
            D::dense_write(r_view),
        )
        .map_err(OperationError::Dense)
}

fn copy_col_major_strided<D: Copy>(
    source: &[D],
    rows: usize,
    cols: usize,
    source_leading: usize,
    destination: &mut [D],
    destination_leading: usize,
) {
    for col in 0..cols {
        let src_start = source_leading * col;
        let dst_start = destination_leading * col;
        destination[dst_start..dst_start + rows]
            .copy_from_slice(&source[src_start..src_start + rows]);
    }
}

fn adjoint_col_major_strided_into<D: FactorScalar>(
    source: &[D],
    rows: usize,
    cols: usize,
    source_leading: usize,
    destination: &mut [D],
    destination_leading: usize,
) {
    for col in 0..cols {
        for row in 0..rows {
            destination[col + destination_leading * row] =
                FactorScalar::adjoint(source[row + source_leading * col]);
        }
    }
}

fn advance_outer_index(index: &mut [usize], shape: &[usize]) {
    for axis in 1..shape.len() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            break;
        }
        index[axis] = 0;
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_tensor_block_to_matrix<D: Copy>(
    source: &[D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    nout: usize,
    matrix: &mut [D],
    matrix_rows: usize,
    row_offset: usize,
    col_offset: usize,
) {
    if shape.is_empty() {
        matrix[row_offset + matrix_rows * col_offset] = source[offset];
        return;
    }
    let run = shape[0];
    let src_lane_stride = strides[0];
    let dst_lane_stride = if nout > 0 { 1 } else { matrix_rows };
    let outer_count: usize = shape[1..].iter().product();
    let mut index = vec![0usize; shape.len()];
    for _ in 0..outer_count {
        let mut src_start = offset;
        let mut row = 0usize;
        let mut row_stride = if nout > 0 { shape[0] } else { 1 };
        let mut col = 0usize;
        let mut col_stride = if nout == 0 { shape[0] } else { 1 };
        for axis in 1..shape.len() {
            src_start += index[axis] * strides[axis];
            if axis < nout {
                row += index[axis] * row_stride;
                row_stride *= shape[axis];
            } else {
                col += index[axis] * col_stride;
                col_stride *= shape[axis];
            }
        }
        let dst_start = (row_offset + row) + matrix_rows * (col_offset + col);
        if src_lane_stride == 1 && dst_lane_stride == 1 {
            matrix[dst_start..dst_start + run].copy_from_slice(&source[src_start..src_start + run]);
        } else {
            for lane in 0..run {
                matrix[dst_start + lane * dst_lane_stride] =
                    source[src_start + lane * src_lane_stride];
            }
        }
        advance_outer_index(&mut index, shape);
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_matching_block_prefix<D: Copy>(
    source: &[D],
    source_strides: &[usize],
    source_offset: usize,
    destination: &mut [D],
    destination_strides: &[usize],
    destination_offset: usize,
    shape: &[usize],
) {
    if shape.is_empty() {
        destination[destination_offset] = source[source_offset];
        return;
    }
    let run = shape[0];
    let outer_count: usize = shape[1..].iter().product();
    let mut index = vec![0usize; shape.len()];
    for _ in 0..outer_count {
        let mut src_start = source_offset;
        let mut dst_start = destination_offset;
        for axis in 1..shape.len() {
            src_start += index[axis] * source_strides[axis];
            dst_start += index[axis] * destination_strides[axis];
        }
        if source_strides[0] == 1 && destination_strides[0] == 1 {
            destination[dst_start..dst_start + run]
                .copy_from_slice(&source[src_start..src_start + run]);
        } else {
            for lane in 0..run {
                destination[dst_start + lane * destination_strides[0]] =
                    source[src_start + lane * source_strides[0]];
            }
        }
        advance_outer_index(&mut index, shape);
    }
}

fn copy_mapped_to_strided_diagonal<D, V, F>(
    data: &mut [D],
    offset: usize,
    diagonal_stride: usize,
    values: &[V],
    to_scalar: &F,
) where
    V: Copy,
    F: Fn(V) -> D + ?Sized,
{
    for (position, &value) in values.iter().enumerate() {
        data[offset + position * diagonal_stride] = to_scalar(value);
    }
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
    if shape.is_empty() {
        data[offset] = matrix[side_offset];
        return;
    }
    let rank = shape.len();
    let run = shape[0];
    let dst_lane_stride = strides[0];
    let outer_count: usize = shape[1..].iter().product();
    let mut index = vec![0usize; rank];
    for _ in 0..outer_count {
        let mut dst_start = offset;
        let mut side = 0usize;
        let mut side_stride = if matrix_axis == 0 { 1 } else { shape[0] };
        let mut matrix_index = 0usize;
        for axis in 1..rank {
            dst_start += index[axis] * strides[axis];
            if axis == matrix_axis {
                matrix_index = index[axis];
            } else {
                side += index[axis] * side_stride;
                side_stride *= shape[axis];
            }
        }
        let (src_start, src_lane_stride) = if matrix_axis == 0 {
            if matrix_axis == rank - 1 {
                (side_offset + side, matrix_rows)
            } else {
                (matrix_rows * (side_offset + side), 1)
            }
        } else if matrix_axis == rank - 1 {
            ((side_offset + side) + matrix_rows * matrix_index, 1)
        } else {
            (
                matrix_index + matrix_rows * (side_offset + side),
                matrix_rows,
            )
        };
        if src_lane_stride == 1 && dst_lane_stride == 1 {
            data[dst_start..dst_start + run].copy_from_slice(&matrix[src_start..src_start + run]);
        } else {
            for lane in 0..run {
                data[dst_start + lane * dst_lane_stride] =
                    matrix[src_start + lane * src_lane_stride];
            }
        }
        advance_outer_index(&mut index, shape);
    }
}

fn coupled_of<R>(rule: &R, tree: &FusionTreeKey) -> SectorId
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    tree.coupled().unwrap_or_else(|| rule.vacuum())
}

fn matricization_map<D>(
    matricizations: &[SectorMatricization<D>],
) -> HashMap<SectorId, &SectorMatricization<D>> {
    matricizations
        .iter()
        .map(|matrix| (matrix.sector, matrix))
        .collect()
}

fn matricization_of<'a, D>(
    matricizations: &'a HashMap<SectorId, &'a SectorMatricization<D>>,
    sector: SectorId,
) -> Result<&'a SectorMatricization<D>, OperationError> {
    matricizations
        .get(&sector)
        .copied()
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
    matricizations: &HashMap<SectorId, &SectorMatricization<D>>,
    sector: SectorId,
    tree: &FusionTreeKey,
) -> Result<Vec<usize>, OperationError> {
    row_placement(matricization_of(matricizations, sector)?, tree).map(|(_, shape)| shape.to_vec())
}

fn col_shape_of<D>(
    matricizations: &HashMap<SectorId, &SectorMatricization<D>>,
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

/// Packs every coupled sector of the source data into its dense column-major
/// matricization, independent of the storage layout.
fn sector_matricizations<R, D>(
    rule: &R,
    structure: &BlockStructure,
    data: &[D],
    nout: usize,
) -> Result<Vec<SectorMatricization<D>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
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
    Ok(matricizations)
}
