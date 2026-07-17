use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

#[cfg(test)]
use std::cell::Cell;

use num_complex::Complex64;
use num_traits::Zero;
use tenet_core::{
    BlockKey, BlockStructure, CoreError, CoupledSectorRegion, FusionProductSpace, FusionRule,
    FusionTensorMapSpace, FusionTreeHomSpace, FusionTreeKey, GenericRigidSymbols,
    MultiplicityFreeRigidSymbols, SectorId, SectorLeg, TensorMap, TensorMapSpace,
};
use tenet_dense::{DenseError, DenseExecutor, DenseTensor, DenseView, DenseViewMut};

use tenet_tensors::{
    BoundDynamicFusionMapSpace, DenseBlockScalar, DenseRecouplingScalar, DynamicFusionMapSpace,
    ValidatedDynamicFusionLayout,
};

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
    fn epsilon() -> f64;
    fn compute_f64_spectrum<E, F>(
        rank: usize,
        scratch: &mut Vec<Self::Real>,
        compute: F,
    ) -> Result<Vec<f64>, E>
    where
        F: FnOnce(&mut [Self::Real]) -> Result<(), E>,
    {
        scratch.resize(rank, Self::Real::zero());
        compute(&mut scratch[..rank])?;
        Ok(scratch[..rank].iter().copied().map(Into::into).collect())
    }
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

    fn epsilon() -> f64 {
        f32::EPSILON as f64
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

    fn epsilon() -> f64 {
        f64::EPSILON
    }

    fn compute_f64_spectrum<E, F>(
        rank: usize,
        _scratch: &mut Vec<Self::Real>,
        compute: F,
    ) -> Result<Vec<f64>, E>
    where
        F: FnOnce(&mut [Self::Real]) -> Result<(), E>,
    {
        let mut values = vec![0.0; rank];
        compute(&mut values)?;
        Ok(values)
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

    fn epsilon() -> f64 {
        f32::EPSILON as f64
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

    fn epsilon() -> f64 {
        f64::EPSILON
    }

    fn compute_f64_spectrum<E, F>(
        rank: usize,
        _scratch: &mut Vec<Self::Real>,
        compute: F,
    ) -> Result<Vec<f64>, E>
    where
        F: FnOnce(&mut [Self::Real]) -> Result<(), E>,
    {
        let mut values = vec![0.0; rank];
        compute(&mut values)?;
        Ok(values)
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
pub(crate) type DynFactor<D> = (DynamicFusionMapSpace, Vec<D>);

/// Borrowed dynamic factorization input whose provider, complete tree grid,
/// rank, and storage length have been validated before the dense executor can
/// be reached. SVD-derived matrix functions consume the same authority.
pub struct BoundDynamicTensorRef<'a, R, D> {
    space: &'a BoundDynamicFusionMapSpace<R>,
    data: &'a [D],
}

/// Typed tensor plus the sole provider authority accepted by provider-sensitive
/// factorization and matrix-function APIs.
pub struct BoundTensorMapRef<'a, R, D, const NOUT: usize, const NIN: usize> {
    space: &'a BoundDynamicFusionMapSpace<R>,
    tensor: &'a TensorMap<D, NOUT, NIN>,
}

/// Owned typed tensor that retains the provider authority for its fusion
/// space. Provider-sensitive operations consume this type or its borrowed
/// view instead of accepting an independently supplied rule.
///
/// The fields are private so a tensor and an unrelated provider-bound space
/// cannot be paired without validation.
///
/// ```compile_fail
/// use tenet_core::TensorMap;
/// use tenet_matrixalgebra::BoundTensorMap;
/// use tenet_tensors::BoundDynamicFusionMapSpace;
///
/// fn forge<R, D>(
///     space: BoundDynamicFusionMapSpace<R>,
///     tensor: TensorMap<D, 1, 1>,
/// ) -> BoundTensorMap<R, D, 1, 1> {
///     BoundTensorMap { space, tensor }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct BoundTensorMap<R, D, const NOUT: usize, const NIN: usize> {
    space: BoundDynamicFusionMapSpace<R>,
    tensor: TensorMap<D, NOUT, NIN>,
}

impl<R, D, const NOUT: usize, const NIN: usize> BoundTensorMap<R, D, NOUT, NIN>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    pub fn try_new(
        provider: Arc<R>,
        tensor: TensorMap<D, NOUT, NIN>,
    ) -> Result<Self, OperationError> {
        let space =
            BoundDynamicFusionMapSpace::bind_multiplicity_free(dyn_space_of(&tensor)?, provider)?;
        BoundDynamicTensorRef::try_new(&space, tensor.data())?;
        Ok(Self { space, tensor })
    }

    pub fn space(&self) -> &BoundDynamicFusionMapSpace<R> {
        &self.space
    }

    pub fn provider(&self) -> &R {
        self.space.provider()
    }

    pub fn tensor(&self) -> &TensorMap<D, NOUT, NIN> {
        &self.tensor
    }

    pub fn data(&self) -> &[D] {
        self.tensor.data()
    }

    pub fn as_ref(&self) -> BoundTensorMapRef<'_, R, D, NOUT, NIN> {
        BoundTensorMapRef {
            space: &self.space,
            tensor: &self.tensor,
        }
    }

    pub fn into_parts(self) -> (BoundDynamicFusionMapSpace<R>, TensorMap<D, NOUT, NIN>) {
        (self.space, self.tensor)
    }
}

impl<R, D, const NOUT: usize, const NIN: usize> std::ops::Deref
    for BoundTensorMap<R, D, NOUT, NIN>
{
    type Target = TensorMap<D, NOUT, NIN>;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

impl<'a, R, D, const NOUT: usize, const NIN: usize> BoundTensorMapRef<'a, R, D, NOUT, NIN>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    pub fn space(&self) -> &'a BoundDynamicFusionMapSpace<R> {
        self.space
    }

    pub fn tensor(&self) -> &'a TensorMap<D, NOUT, NIN> {
        self.tensor
    }

    pub fn data(&self) -> &'a [D] {
        self.tensor.data()
    }

    pub(crate) fn dynamic(&self) -> BoundDynamicTensorRef<'_, R, D> {
        BoundDynamicTensorRef {
            space: &self.space,
            data: self.tensor.data(),
        }
    }
}

impl<'a, R, D> BoundDynamicTensorRef<'a, R, D>
where
    R: FusionRule,
{
    pub fn try_new(
        space: &'a BoundDynamicFusionMapSpace<R>,
        data: &'a [D],
    ) -> Result<Self, OperationError> {
        let raw = space.space();
        let hom_rank = raw.homspace().codomain().len() + raw.homspace().domain().len();
        if raw.rank() != hom_rank {
            return Err(OperationError::from_core_preserving_context(
                CoreError::StructureRankMismatch {
                    expected: hom_rank,
                    actual: raw.rank(),
                },
            ));
        }
        if raw.structure().rank() != raw.rank() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::StructureRankMismatch {
                    expected: raw.rank(),
                    actual: raw.structure().rank(),
                },
            ));
        }
        let expected = raw
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?;
        if data.len() != expected {
            return Err(OperationError::from_core_preserving_context(
                CoreError::DimensionMismatch {
                    expected,
                    actual: data.len(),
                },
            ));
        }
        Ok(Self { space, data })
    }

    #[inline]
    pub fn space(&self) -> &BoundDynamicFusionMapSpace<R> {
        self.space
    }

    #[inline]
    pub fn data(&self) -> &'a [D] {
        self.data
    }
}

/// Owned dynamic factor that retains the provider used to create its complete
/// fusion space.
pub struct BoundDynFactor<R, D> {
    space: BoundDynamicFusionMapSpace<R>,
    data: Vec<D>,
}

impl<R, D> fmt::Debug for BoundDynFactor<R, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoundDynFactor")
            .field("space", &self.space)
            .field("data_len", &self.data.len())
            .finish()
    }
}

impl<R, D> Clone for BoundDynFactor<R, D>
where
    D: Clone,
{
    fn clone(&self) -> Self {
        Self {
            space: self.space.clone(),
            data: self.data.clone(),
        }
    }
}

impl<R, D> BoundDynFactor<R, D>
where
    R: FusionRule,
{
    pub(crate) fn from_bound(
        space: BoundDynamicFusionMapSpace<R>,
        data: Vec<D>,
        expected_nout: usize,
        expected_nin: usize,
    ) -> Result<Self, OperationError> {
        if space.space().nout() != expected_nout || space.space().nin() != expected_nin {
            return Err(OperationError::RankMismatch {
                expected: expected_nout + expected_nin,
                actual: space.space().rank(),
            });
        }
        let expected = space
            .space()
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?;
        if data.len() != expected {
            return Err(OperationError::from_core_preserving_context(
                CoreError::DimensionMismatch {
                    expected,
                    actual: data.len(),
                },
            ));
        }
        Ok(Self { space, data })
    }

    pub fn space(&self) -> &BoundDynamicFusionMapSpace<R> {
        &self.space
    }

    pub fn data(&self) -> &[D] {
        &self.data
    }

    pub(crate) fn data_mut(&mut self) -> &mut [D] {
        &mut self.data
    }

    pub(crate) fn raw_space_and_data_mut(&mut self) -> (&DynamicFusionMapSpace, &mut [D]) {
        (self.space.space(), &mut self.data)
    }

    pub fn into_parts(self) -> (BoundDynamicFusionMapSpace<R>, Vec<D>) {
        (self.space, self.data)
    }
}

pub(crate) fn adjoint_bound_factor<R, D>(
    factor: &BoundDynFactor<R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (space, data) = tenet_tensors::adjoint_bound_dyn(factor.space(), factor.data())?;
    let nout = space.space().nout();
    let nin = space.space().nin();
    BoundDynFactor::from_bound(space, data, nout, nin)
}

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
    let axis_dim = |leg: &SectorLeg| {
        leg.degeneracies().iter().try_fold(0usize, |total, &dim| {
            total
                .checked_add(dim)
                .ok_or(CoreError::ElementCountOverflow)
        })
    };
    let mut codomain_dims = [0usize; NOUT];
    for (dim, leg) in codomain_dims
        .iter_mut()
        .zip(space.homspace().codomain().legs())
    {
        *dim = axis_dim(leg).map_err(OperationError::from_core_preserving_context)?;
    }
    let mut domain_dims = [0usize; NIN];
    for (dim, leg) in domain_dims.iter_mut().zip(space.homspace().domain().legs()) {
        *dim = axis_dim(leg).map_err(OperationError::from_core_preserving_context)?;
    }
    // Why not recover axis dimensions from populated fusion-tree blocks:
    // SectorLeg is the complete axis-space authority, including sectors with
    // no participating tree. External relabeling is irrelevant to a dimension
    // sum and can fail for a finite encoded dual after dense factorization.
    let typed_space = FusionTensorMapSpace::from_shared_subblock_structure(
        TensorMapSpace::<NOUT, NIN>::from_dims(codomain_dims, domain_dims)
            .map_err(OperationError::from_core_preserving_context)?,
        space.homspace().clone(),
        Arc::clone(space.structure()),
    )
    .map_err(OperationError::from_core_preserving_context)?
    .try_bind_rule(rule)
    .map_err(OperationError::from_core_preserving_context)?;
    TensorMap::from_vec_with_fusion_space(data, typed_space)
        .map_err(OperationError::from_core_preserving_context)
}

pub(crate) fn typed_from_bound_factor<R, D, const NOUT: usize, const NIN: usize>(
    factor: BoundDynFactor<R, D>,
) -> Result<BoundTensorMap<R, D, NOUT, NIN>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let (space, data) = factor.into_parts();
    let provider = Arc::clone(space.provider_arc());
    let tensor = typed_from_dyn(provider.as_ref(), (space.space().clone(), data))?;
    Ok(BoundTensorMap { space, tensor })
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
pub struct SvdTrunc<R, D, const NOUT: usize, const NIN: usize> {
    pub u: BoundTensorMap<R, D, NOUT, 1>,
    pub s: BoundTensorMap<R, D, 1, 1>,
    pub vh: BoundTensorMap<R, D, 1, NIN>,
    pub singular_values: Vec<SectorSpectrum>,
    pub error: f64,
}

/// Dynamic-rank [`SvdTrunc`].
#[derive(Clone, Debug)]
pub struct SvdTruncDyn<R, D> {
    u: BoundDynFactor<R, D>,
    s: BoundDynFactor<R, D>,
    vh: BoundDynFactor<R, D>,
    singular_values: Vec<SectorSpectrum>,
    error: f64,
}

impl<R, D> SvdTruncDyn<R, D> {
    pub fn u(&self) -> &BoundDynFactor<R, D> {
        &self.u
    }

    pub fn s(&self) -> &BoundDynFactor<R, D> {
        &self.s
    }

    pub fn vh(&self) -> &BoundDynFactor<R, D> {
        &self.vh
    }

    pub fn singular_values(&self) -> &[SectorSpectrum] {
        &self.singular_values
    }

    pub fn error(&self) -> f64 {
        self.error
    }

    pub fn into_parts(
        self,
    ) -> (
        BoundDynFactor<R, D>,
        BoundDynFactor<R, D>,
        BoundDynFactor<R, D>,
        Vec<SectorSpectrum>,
        f64,
    ) {
        (self.u, self.s, self.vh, self.singular_values, self.error)
    }
}

/// Compact (thin, untruncated) fusion-tensor SVD `t = U * S * Vh`
/// (MatrixAlgebraKit `svd_compact`).
///
/// This is the pure device-boundary factorization: the dense per-sector SVDs
/// run through the [`DenseExecutor`] and no truncation logic is involved.
/// Per block the bond is `min(rows, cols)`; the square-`U` variant is
/// MatrixAlgebraKit `svd_full` (later batch).
#[derive(Clone, Debug)]
pub struct SvdCompact<R, D, const NOUT: usize, const NIN: usize> {
    pub u: BoundTensorMap<R, D, NOUT, 1>,
    pub s: BoundTensorMap<R, D, 1, 1>,
    pub vh: BoundTensorMap<R, D, 1, NIN>,
    pub singular_values: Vec<SectorSpectrum>,
}

/// Dynamic-rank [`SvdCompact`].
#[derive(Clone, Debug)]
pub struct SvdCompactDyn<R, D> {
    u: BoundDynFactor<R, D>,
    s: BoundDynFactor<R, D>,
    vh: BoundDynFactor<R, D>,
    singular_values: Vec<SectorSpectrum>,
}

impl<R, D> SvdCompactDyn<R, D> {
    pub fn u(&self) -> &BoundDynFactor<R, D> {
        &self.u
    }

    pub fn s(&self) -> &BoundDynFactor<R, D> {
        &self.s
    }

    pub fn vh(&self) -> &BoundDynFactor<R, D> {
        &self.vh
    }

    pub fn singular_values(&self) -> &[SectorSpectrum] {
        &self.singular_values
    }

    pub fn into_parts(
        self,
    ) -> (
        BoundDynFactor<R, D>,
        BoundDynFactor<R, D>,
        BoundDynFactor<R, D>,
        Vec<SectorSpectrum>,
    ) {
        (self.u, self.s, self.vh, self.singular_values)
    }
}

fn diagonal_bond_svd_factor<R, D, V>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
    V: Copy,
{
    #[cfg(test)]
    DIAGONAL_BOND_BUILD_PROBE.with(|probe| {
        let mut current = probe.get();
        current.calls += 1;
        current.values += spectrum
            .iter()
            .map(|entry| entry.values.len())
            .sum::<usize>();
        probe.set(current);
    });
    let space = diagonal_bond_bound_space_like(authority, spectrum)?;
    let data = diagonal_bond_data(space.space(), spectrum, to_scalar)?;
    BoundDynFactor::from_bound(space, data, 1, 1)
}

#[doc(hidden)]
pub fn diagonal_bond_bound_space_like<R, V>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<V>],
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let rule = authority.provider();
    let new_leg = SectorLeg::new(
        spectrum
            .iter()
            .map(|entry| (entry.sector, entry.values.len())),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg.clone()]),
        FusionProductSpace::new([new_leg]),
    );
    let length_by_sector: HashMap<SectorId, usize> = spectrum
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();
    authority.derive_from_fusion_tree_shapes(homspace, |keys| {
        Ok(keys
            .iter()
            .map(|key| {
                let count = length_by_sector
                    .get(&coupled_of(rule, key.codomain_tree()))
                    .copied()
                    .unwrap_or(0);
                vec![count, count]
            })
            .collect::<Vec<_>>())
    })
}

pub fn diagonal_bond_bound_space<R, V>(
    provider: Arc<R>,
    spectrum: &[SectorSpectrum<V>],
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let rule = provider.as_ref();
    let new_leg = SectorLeg::new(
        spectrum
            .iter()
            .map(|entry| (entry.sector, entry.values.len())),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg.clone()]),
        FusionProductSpace::new([new_leg]),
    );
    let length_by_sector: HashMap<SectorId, usize> = spectrum
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();
    let shapes = homspace
        .fusion_tree_keys(rule)
        .iter()
        .map(|key| {
            let count = length_by_sector
                .get(&coupled_of(rule, key.codomain_tree()))
                .copied()
                .unwrap_or(0);
            vec![count, count]
        })
        .collect::<Vec<_>>();
    BoundDynamicFusionMapSpace::from_degeneracy_shapes(provider, homspace, shapes)
}

/// Fills the dense block-diagonal data of `space` from `spectrum`, mapping
/// each value through `to_scalar`. Only the
/// per-block diagonal is written; the rest stays zero. Bit-for-bit identical to
/// the fill inside the former monolithic `diagonal_bond_tensor_dyn`.
pub fn diagonal_bond_data<D, V>(
    space: &DynamicFusionMapSpace,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<Vec<D>, OperationError>
where
    D: FactorScalar,
    V: Copy,
{
    let spectrum_by_sector: HashMap<SectorId, &SectorSpectrum<V>> =
        spectrum.iter().map(|entry| (entry.sector, entry)).collect();
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
    Ok(data)
}

/// Scales one bond axis of `data` (laid out per `space`) by the per-sector
/// `spectrum`, in place — the block-local realization of TensorKit's
/// `DiagonalTensorMap` multiplication. `axis = None` scales each block's
/// trailing axis (`t * D`, `rmul!`, column scaling); `axis = Some(0)` scales
/// the leading axis (`D * t`, `lmul!`, row scaling). Verified twist-free
/// against TK `diagonal.jl`: diagonal multiplication is pure per-block scaling
/// with no braiding or fusion-tree recoupling (`block(D, c)` is a `Diagonal`,
/// so LinearAlgebra dispatches to scaling, not GEMM). A real `spectrum` on a
/// complex `data` promotes each entry the same way (`D::from_real`).
pub fn scale_axis_by_spectrum<D>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    axis: Option<usize>,
    spectrum: &[SectorSpectrum],
) -> Result<(), OperationError>
where
    D: FactorScalar,
{
    scale_axis_by_spectrum_mapped(space, data, axis, spectrum, D::from_real)
}

/// Value-generic sibling of [`scale_axis_by_spectrum`]. Why not convert the
/// spectrum before this call: a complex spectrum cannot pass through the
/// real-only `SectorSpectrum` alias without losing its imaginary component.
pub fn scale_axis_by_spectrum_mapped<D, V>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    axis: Option<usize>,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: impl Fn(V) -> D,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    V: Copy,
{
    let spectrum_by_sector: HashMap<SectorId, &SectorSpectrum<V>> =
        spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    let nout = space.nout();
    let structure = Arc::clone(space.structure());
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let shape = block.shape();
        if shape.is_empty() {
            continue;
        }
        let strides = block.strides();
        let offset = block.offset();
        let bond_axis = axis.unwrap_or(shape.len() - 1);
        // Index the spectrum by the charge ON THE SCALED LEG — its uncoupled
        // charge in this block's fusion tree — NOT the block's coupled charge.
        // For an SVD/eigh factor's sole bond leg the two coincide, but scaling a
        // general tensor leg (diagonal-aware `contract`, #75) is only correct per
        // leg charge.
        let leg_charge = if bond_axis < nout {
            tree.codomain_tree().uncoupled()[bond_axis]
        } else {
            tree.domain_tree().uncoupled()[bond_axis - nout]
        };
        // Absent charge => this leg slice is structurally zero for the spectrum;
        // nothing to scale (mirrors `diagonal_bond_tensor_dyn`'s `unwrap_or(0)`).
        let Some(&entry) = spectrum_by_sector.get(&leg_charge) else {
            continue;
        };
        let bond = shape[bond_axis];
        let bond_stride = strides[bond_axis];
        debug_assert_eq!(
            bond,
            entry.values.len(),
            "bond degeneracy must match the spectrum length"
        );
        let bond = bond.min(entry.values.len());
        // Walk every combination of the non-bond axes; for each, scale the
        // `bond` entries along `bond_axis` by the spectrum.
        let lead_axes: Vec<usize> = (0..shape.len()).filter(|&a| a != bond_axis).collect();
        let outer: usize = lead_axes.iter().map(|&a| shape[a]).product();
        let mut coord = vec![0usize; lead_axes.len()];
        for _ in 0..outer {
            let mut base = offset;
            for (k, &a) in lead_axes.iter().enumerate() {
                base += coord[k] * strides[a];
            }
            for j in 0..bond {
                let scale = to_scalar(entry.values[j]);
                let idx = base + j * bond_stride;
                data[idx] = data[idx] * scale;
            }
            for k in (0..coord.len()).rev() {
                coord[k] += 1;
                if coord[k] < shape[lead_axes[k]] {
                    break;
                }
                coord[k] = 0;
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

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompactSvdCopyProbe {
    pub input_pack_calls: usize,
    pub input_pack_bytes: usize,
    pub output_scatter_calls: usize,
    pub output_scatter_bytes: usize,
}

#[cfg(test)]
thread_local! {
    static COMPACT_SVD_COPY_PROBE: Cell<CompactSvdCopyProbe> = Cell::default();
    static COMPACT_QR_COPY_PROBE: Cell<CompactQrCopyProbe> = Cell::default();
    static EIGH_COPY_PROBE: Cell<EighCopyProbe> = Cell::default();
    static COMPACT_LQ_COPY_PROBE: Cell<CompactLqCopyProbe> = Cell::default();
    static DIAGONAL_BOND_BUILD_PROBE: Cell<DiagonalBondBuildProbe> = Cell::default();
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DiagonalBondBuildProbe {
    pub calls: usize,
    pub values: usize,
}

#[cfg(test)]
pub(crate) fn reset_compact_svd_copy_probe() {
    COMPACT_SVD_COPY_PROBE.with(|probe| probe.set(CompactSvdCopyProbe::default()));
}

#[cfg(test)]
pub(crate) fn compact_svd_copy_probe() -> CompactSvdCopyProbe {
    COMPACT_SVD_COPY_PROBE.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn reset_diagonal_bond_build_probe() {
    DIAGONAL_BOND_BUILD_PROBE.with(|probe| probe.set(DiagonalBondBuildProbe::default()));
}

#[cfg(test)]
pub(crate) fn diagonal_bond_build_probe() -> DiagonalBondBuildProbe {
    DIAGONAL_BOND_BUILD_PROBE.with(Cell::get)
}

#[cfg(test)]
fn record_compact_svd_input_pack<D>(matricizations: &[SectorMatricization<D>]) {
    COMPACT_SVD_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.input_pack_calls += matricizations.len();
        current.input_pack_bytes += matricizations
            .iter()
            .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
            .sum::<usize>();
        probe.set(current);
    });
}

#[cfg(test)]
fn record_compact_svd_output_scatter<D>(elements: usize) {
    COMPACT_SVD_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += 1;
        current.output_scatter_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompactQrCopyProbe {
    pub input_pack_calls: usize,
    pub input_pack_bytes: usize,
    pub output_scatter_calls: usize,
    pub output_scatter_bytes: usize,
}

#[cfg(test)]
pub(crate) fn reset_compact_qr_copy_probe() {
    COMPACT_QR_COPY_PROBE.with(|probe| probe.set(CompactQrCopyProbe::default()));
}

#[cfg(test)]
pub(crate) fn compact_qr_copy_probe() -> CompactQrCopyProbe {
    COMPACT_QR_COPY_PROBE.with(Cell::get)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompactLqCopyProbe {
    pub input_pack_calls: usize,
    pub input_pack_bytes: usize,
    pub output_scatter_calls: usize,
    pub output_scatter_bytes: usize,
    pub scratch_buffer_count: usize,
    pub scratch_capacity_bytes: usize,
    pub adjoint_scratch_fill_calls: usize,
    pub adjoint_scratch_fill_bytes: usize,
    pub final_adjoint_copy_calls: usize,
    pub final_adjoint_copy_bytes: usize,
}

#[cfg(test)]
pub(crate) fn reset_compact_lq_copy_probe() {
    COMPACT_LQ_COPY_PROBE.with(|probe| probe.set(CompactLqCopyProbe::default()));
}

#[cfg(test)]
pub(crate) fn compact_lq_copy_probe() -> CompactLqCopyProbe {
    COMPACT_LQ_COPY_PROBE.with(Cell::get)
}

#[cfg(test)]
fn record_compact_qr_input_pack<D>(matricizations: &[SectorMatricization<D>]) {
    COMPACT_QR_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.input_pack_calls += matricizations.len();
        current.input_pack_bytes += matricizations
            .iter()
            .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
            .sum::<usize>();
        probe.set(current);
    });
}

#[cfg(test)]
fn record_compact_qr_output_scatter<D>(elements: usize) {
    COMPACT_QR_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += 1;
        current.output_scatter_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct EighCopyProbe {
    pub input_pack_calls: usize,
    pub input_pack_bytes: usize,
    pub output_scatter_calls: usize,
    pub output_scatter_bytes: usize,
}

#[cfg(test)]
pub(crate) fn reset_eigh_copy_probe() {
    EIGH_COPY_PROBE.with(|probe| probe.set(EighCopyProbe::default()));
}

#[cfg(test)]
pub(crate) fn eigh_copy_probe() -> EighCopyProbe {
    EIGH_COPY_PROBE.with(Cell::get)
}

#[cfg(test)]
fn record_eigh_input_pack<D>(matricizations: &[SectorMatricization<D>]) {
    EIGH_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.input_pack_calls += matricizations.len();
        current.input_pack_bytes += matricizations
            .iter()
            .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
            .sum::<usize>();
        probe.set(current);
    });
}

#[cfg(test)]
fn record_eigh_output_scatter<D>(elements: usize) {
    EIGH_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += 1;
        current.output_scatter_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
fn record_compact_lq_input_pack<D>(matricizations: &[SectorMatricization<D>]) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.input_pack_calls += matricizations.len();
        current.input_pack_bytes += matricizations
            .iter()
            .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
            .sum::<usize>();
        probe.set(current);
    });
}

#[cfg(test)]
fn record_compact_lq_output_scatter<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += 1;
        current.output_scatter_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
fn record_compact_lq_scratch<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.scratch_buffer_count += 3;
        current.scratch_capacity_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
fn record_compact_lq_adjoint_fill<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.adjoint_scratch_fill_calls += 1;
        current.adjoint_scratch_fill_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
fn record_compact_lq_final_adjoint_copy<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.final_adjoint_copy_calls += 1;
        current.final_adjoint_copy_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
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
    Ok(sector_matricizations(
        input.space().provider(),
        space.structure(),
        input.data(),
        space.nout(),
    )?
    .into_iter()
    .map(|matrix| SectorMatricizationDiagnostic {
        sector: matrix.sector,
        rows: matrix.rows,
        cols: matrix.cols,
        elements: matrix.data.len(),
    })
    .collect())
}

/// All singular values per coupled sector, descending (MatrixAlgebraKit
/// `svd_vals`). Runs the dense SVD per sector through the executor and keeps
/// only the spectra.
pub fn svd_vals<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    svd_vals_dyn(dense, &input.dynamic())
}

/// Dynamic-rank [`svd_vals`].
pub fn svd_vals_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = input.space().provider();
    let space = input.space().space();
    // Values-only: per coupled sector call the no-vector SVD (`svd_vals`,
    // LAPACK `job='N'`) and keep the spectrum. Unlike `svd_compact_dyn` this
    // never builds the U/Vt spaces, allocates the factor buffers, gauge-fixes,
    // or scatters blocks into the fusion-tree layout — all of which the old
    // `svd_compact_dyn(..).map(|svd| svd.singular_values)` computed then threw
    // away. LAPACK computes the singular values identically with or without
    // vectors, so the spectrum is bit-for-bit the full-SVD spectrum.
    let matricizations =
        sector_matricizations(rule, space.structure(), input.data(), space.nout())?;
    let mut singular_values = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let rank = matrix.rows.min(matrix.cols);
        let input_shape = [matrix.rows, matrix.cols];
        let input_strides = [1usize, matrix.rows];
        let input = DenseView::new(&matrix.data, &input_shape, &input_strides, 0)
            .map_err(OperationError::Dense)?;
        let s_tensor = dense
            .svd_vals(D::dense_read(input))
            .map_err(OperationError::Dense)?;
        let mut s = D::real_spectrum(&s_tensor).map_err(OperationError::Dense)?;
        s.truncate(rank);
        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s,
        });
    }
    Ok(singular_values)
}

/// Truncated fusion-tensor SVD (MatrixAlgebraKit `svd_trunc`).
///
/// Layering: the untruncated compact factorization runs on the device
/// boundary ([`svd_compact`]); the truncation decision is host-side scalar
/// work over the spectra and its application slices the leading bond states
/// per sector.
pub fn svd_trunc<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
    truncation: &Truncation,
) -> Result<SvdTrunc<R, D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let out = svd_trunc_dyn(dense, &input.dynamic(), truncation)?;
    Ok(SvdTrunc {
        u: typed_from_bound_factor(out.u)?,
        s: typed_from_bound_factor(out.s)?,
        vh: typed_from_bound_factor(out.vh)?,
        singular_values: out.singular_values,
        error: out.error,
    })
}

/// Dynamic-rank [`svd_trunc`].
pub fn svd_trunc_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    truncation: &Truncation,
) -> Result<SvdTruncDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (u, vh, singular_values) = svd_compact_factors_dyn(dense, input)?;
    truncate_svd_factors_dyn(u, None, vh, singular_values, truncation)
}

/// Compact (untruncated) fusion-tensor SVD through the device boundary.
pub fn svd_compact<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<SvdCompact<R, D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let out = svd_compact_dyn(dense, &input.dynamic())?;
    Ok(SvdCompact {
        u: typed_from_bound_factor(out.u)?,
        s: typed_from_bound_factor(out.s)?,
        vh: typed_from_bound_factor(out.vh)?,
        singular_values: out.singular_values,
    })
}

/// The compact-SVD factors without materializing the diagonal `S`:
/// `(U, Vh, spectrum)`. The shared core of every SVD entry point.
/// [`svd_compact_dyn`] wraps this and adds the dense `S` as a tensor for callers
/// that want it; polar and the matrix-function paths scale by the spectrum
/// directly (TensorKit `DiagonalTensorMap` `rmul!`) and never build `S`.
pub type SvdFactorsDyn<R, D> = (
    BoundDynFactor<R, D>,
    BoundDynFactor<R, D>,
    Vec<SectorSpectrum>,
);

pub fn svd_compact_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = input.space().provider();
    let space = input.space().space();
    if let Some(plan) = compact_factor_plan(input.space())? {
        return svd_compact_direct_regions(dense, input, &plan);
    }
    let matricizations =
        sector_matricizations(rule, space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_svd_input_pack(&matricizations);

    let ranks = matricizations
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector,
            kept: matrix.rows.min(matrix.cols),
        })
        .collect::<Vec<_>>();
    let (u_space, vt_space) =
        build_left_right_bound_spaces(input.space(), space.homspace(), &matricizations, &ranks)?;
    let u_len = u_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut u_data = vec![D::zero(); u_len];
    let vt_len = vt_space
        .space()
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
        scatter_left_sector_blocks(
            rule,
            u_space.space(),
            &mut u_data,
            matrix,
            &u_workspace,
            max_rows,
        )?;
        #[cfg(test)]
        record_compact_svd_output_scatter::<D>(matrix.rows * rank);
        scatter_right_sector_blocks(
            rule,
            vt_space.space(),
            &mut vt_data,
            matrix,
            &vt_workspace,
            max_rank,
        )?;
        #[cfg(test)]
        record_compact_svd_output_scatter::<D>(rank * matrix.cols);
    }

    let u = BoundDynFactor::from_bound(u_space, u_data, space.nout(), 1)?;
    let vh = BoundDynFactor::from_bound(vt_space, vt_data, 1, space.nin())?;
    Ok((u, vh, singular_values))
}

#[derive(Debug)]
struct MatricizationPlan {
    layout: ValidatedDynamicFusionLayout,
    regions: Arc<[CoupledSectorRegion]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompactFactorRoute {
    source_region: usize,
    left_region: Option<usize>,
    right_region: Option<usize>,
    sector: SectorId,
    rank: usize,
}

#[derive(Debug)]
pub(crate) struct CompactFactorPlan {
    source: Arc<MatricizationPlan>,
    left_layout: ValidatedDynamicFusionLayout,
    right_layout: ValidatedDynamicFusionLayout,
    left_regions: Arc<[CoupledSectorRegion]>,
    right_regions: Arc<[CoupledSectorRegion]>,
    routes: Arc<[CompactFactorRoute]>,
}

const COMPACT_FACTOR_PLAN_CACHE_CAP: usize = 1024;

// Why not take the global LRU mutex on every warm factorization: concurrent
// sector factorizations would serialize on metadata lookup. Why not a
// per-thread map: one strong front entry bounds residency per worker.
thread_local! {
    static COMPACT_FACTOR_PLAN_FRONT: RefCell<Option<(
        u64,
        ValidatedDynamicFusionLayout,
        Arc<CompactFactorPlan>,
    )>> = const { RefCell::new(None) };
}

struct CompactFactorPlanCache {
    entries: Mutex<lru::LruCache<ValidatedDynamicFusionLayout, Arc<CompactFactorPlan>>>,
}

impl Default for CompactFactorPlanCache {
    fn default() -> Self {
        Self {
            entries: Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(COMPACT_FACTOR_PLAN_CACHE_CAP).unwrap(),
            )),
        }
    }
}

fn compact_factor_plan_cache() -> (u64, Arc<CompactFactorPlanCache>) {
    tenet_tensors::registered_operation_cache::<CompactFactorPlanCache>()
}

fn compact_factor_plan<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<Arc<CompactFactorPlan>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let source_layout = input.validated_layout();
    let current_epoch = tenet_tensors::operation_cache_reset_epoch();
    if let Some(plan) = COMPACT_FACTOR_PLAN_FRONT.with_borrow_mut(|front| {
        if front
            .as_ref()
            .is_some_and(|(epoch, _, _)| *epoch != current_epoch)
        {
            *front = None;
        }
        front.as_ref().and_then(|(epoch, layout, plan)| {
            (*epoch == current_epoch && layout == &source_layout).then(|| Arc::clone(plan))
        })
    }) {
        return Ok(Some(plan));
    }
    let (cache_epoch, cache) = compact_factor_plan_cache();
    if let Ok(mut entries) = cache.entries.lock() {
        if let Some(plan) = entries.get(&source_layout) {
            let plan = Arc::clone(plan);
            COMPACT_FACTOR_PLAN_FRONT.with_borrow_mut(|front| {
                *front = Some((cache_epoch, source_layout, Arc::clone(&plan)));
            });
            return Ok(Some(plan));
        }
    }

    let Some(built) = build_compact_factor_plan(input, source_layout.clone())? else {
        return Ok(None);
    };
    if let Ok(mut entries) = cache.entries.lock() {
        if let Some(plan) = entries.get(&source_layout) {
            let plan = Arc::clone(plan);
            COMPACT_FACTOR_PLAN_FRONT.with_borrow_mut(|front| {
                *front = Some((cache_epoch, source_layout, Arc::clone(&plan)));
            });
            return Ok(Some(plan));
        }
        entries.put(source_layout.clone(), Arc::clone(&built));
    }
    // Why not make reset a quiescence barrier: a call already holding this
    // immutable plan may finish safely. Its epoch makes the next call discard
    // the front entry and resolve the reset generation instead.
    COMPACT_FACTOR_PLAN_FRONT.with_borrow_mut(|front| {
        *front = Some((cache_epoch, source_layout, Arc::clone(&built)));
    });
    Ok(Some(built))
}

fn build_compact_factor_plan<R>(
    input: &BoundDynamicFusionMapSpace<R>,
    source_layout: ValidatedDynamicFusionLayout,
) -> Result<Option<Arc<CompactFactorPlan>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let space = input.space();
    let Some(regions) = checked_sector_regions(space.structure(), space.nout())? else {
        return Ok(None);
    };
    let source = Arc::new(MatricizationPlan {
        layout: source_layout,
        regions,
    });
    let ranks = source
        .regions
        .iter()
        .map(|region| SectorRank {
            sector: region_sector(input.provider(), region),
            kept: region.rows().min(region.cols()),
        })
        .collect::<Vec<_>>();
    let layouts = region_matricization_skeletons::<R, f64>(input.provider(), &source.regions);
    let (u_space, vh_space) =
        build_left_right_bound_spaces::<R, f64>(input, space.homspace(), &layouts, &ranks)?;
    let left_regions = checked_sector_regions(u_space.space().structure(), u_space.space().nout())?
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: "compact left factor is not a coupled-sector matrix layout",
        })?;
    let right_regions =
        checked_sector_regions(vh_space.space().structure(), vh_space.space().nout())?.ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: "compact right factor is not a coupled-sector matrix layout",
            },
        )?;
    let routes = compile_compact_factor_routes(
        input.provider(),
        &source.regions,
        &left_regions,
        &right_regions,
    )?;
    Ok(Some(Arc::new(CompactFactorPlan {
        source,
        left_layout: u_space.validated_layout(),
        right_layout: vh_space.validated_layout(),
        left_regions,
        right_regions,
        routes,
    })))
}

fn compile_compact_factor_routes<R>(
    rule: &R,
    source_regions: &[CoupledSectorRegion],
    left_regions: &[CoupledSectorRegion],
    right_regions: &[CoupledSectorRegion],
) -> Result<Arc<[CompactFactorRoute]>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let left_by_sector = sector_region_index_map(rule, left_regions)?;
    let right_by_sector = sector_region_index_map(rule, right_regions)?;
    let mut routes = Vec::with_capacity(source_regions.len());
    let mut used_left = vec![false; left_regions.len()];
    let mut used_right = vec![false; right_regions.len()];
    for (source_region, region) in source_regions.iter().enumerate() {
        let sector = region_sector(rule, region);
        let rank = region.rows().min(region.cols());
        let (left_region, right_region) = if rank == 0 {
            (None, None)
        } else {
            let left_region = sector_region_index_of(&left_by_sector, sector, "left")?;
            let right_region = sector_region_index_of(&right_by_sector, sector, "right")?;
            validate_factor_region(&left_regions[left_region], region.rows(), rank, "left")?;
            validate_factor_region(&right_regions[right_region], rank, region.cols(), "right")?;
            used_left[left_region] = true;
            used_right[right_region] = true;
            (Some(left_region), Some(right_region))
        };
        routes.push(CompactFactorRoute {
            source_region,
            left_region,
            right_region,
            sector,
            rank,
        });
    }
    validate_no_unused_factor_regions(left_regions, &used_left, "left")?;
    validate_no_unused_factor_regions(right_regions, &used_right, "right")?;
    Ok(routes.into())
}

#[cfg(test)]
pub(crate) fn compact_factor_plan_for_test<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<Arc<CompactFactorPlan>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    compact_factor_plan(input)
}

#[cfg(test)]
pub(crate) fn compact_factor_plan_regions_for_test(
    plan: &CompactFactorPlan,
) -> (
    Arc<[CoupledSectorRegion]>,
    Arc<[CoupledSectorRegion]>,
    Arc<[CoupledSectorRegion]>,
) {
    (
        Arc::clone(&plan.source.regions),
        Arc::clone(&plan.left_regions),
        Arc::clone(&plan.right_regions),
    )
}

#[cfg(test)]
pub(crate) fn validate_compact_factor_routes_for_test<R>(
    rule: &R,
    source: &[CoupledSectorRegion],
    u: &[CoupledSectorRegion],
    vh: &[CoupledSectorRegion],
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    compile_compact_factor_routes(rule, source, u, vh).map(|_| ())
}

fn svd_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source.layout, input.space().validated_layout());
    let u_space = input.space().rebind_validated(&plan.left_layout)?;
    let vh_space = input.space().rebind_validated(&plan.right_layout)?;
    let mut u_data = vec![D::zero(); plan.left_layout.required_len()?];
    let mut vh_data = vec![D::zero(); plan.right_layout.required_len()?];
    let mut singular_values = Vec::with_capacity(plan.routes.len());
    let mut spectrum_scratch = Vec::<D::Real>::new();

    for route in plan.routes.iter().copied() {
        let region = &plan.source.regions[route.source_region];
        let rank = route.rank;
        if rank == 0 {
            singular_values.push(SectorSpectrum {
                sector: route.sector,
                values: Vec::new(),
            });
            continue;
        }
        let u_region =
            &plan.left_regions[route.left_region.expect("nonzero route has left region")];
        let vh_region =
            &plan.right_regions[route.right_region.expect("nonzero route has right region")];

        let input_shape = [region.rows(), region.cols()];
        let input_strides = [1usize, region.rows()];
        let u_shape = [region.rows(), rank];
        let u_strides = [1usize, region.rows()];
        let s_shape = [rank];
        let s_strides = [1usize];
        let vh_shape = [rank, region.cols()];
        let vh_strides = [1usize, rank];
        let input_view = DenseView::new(
            &input.data()[region.range()],
            &input_shape,
            &input_strides,
            0,
        )
        .map_err(OperationError::Dense)?;
        let u_view = DenseViewMut::new(&mut u_data[u_region.range()], &u_shape, &u_strides, 0)
            .map_err(OperationError::Dense)?;
        let spectrum = D::compute_f64_spectrum(rank, &mut spectrum_scratch, |spectrum| {
            let s_view = DenseViewMut::new(spectrum, &s_shape, &s_strides, 0)
                .map_err(OperationError::Dense)?;
            let vh_view =
                DenseViewMut::new(&mut vh_data[vh_region.range()], &vh_shape, &vh_strides, 0)
                    .map_err(OperationError::Dense)?;
            dense
                .svd_into(
                    D::dense_read(input_view),
                    D::dense_write(u_view),
                    D::Real::dense_write(s_view),
                    D::dense_write(vh_view),
                )
                .map_err(OperationError::Dense)
        })?;
        svd_compact_gauge(
            &mut u_data[u_region.range()],
            region.rows(),
            region.rows(),
            &mut vh_data[vh_region.range()],
            rank,
            region.cols(),
            rank,
        );
        singular_values.push(SectorSpectrum {
            sector: route.sector,
            values: spectrum,
        });
    }

    let u = BoundDynFactor::from_bound(u_space, u_data, space.nout(), 1)?;
    let vh = BoundDynFactor::from_bound(vh_space, vh_data, 1, space.nin())?;
    Ok((u, vh, singular_values))
}

fn region_matricization_skeletons<R, D>(
    rule: &R,
    regions: &[CoupledSectorRegion],
) -> Vec<SectorMatricization<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    regions
        .iter()
        .map(|region| SectorMatricization {
            sector: region_sector(rule, region),
            rows: region.rows(),
            cols: region.cols(),
            row_trees: region
                .row_trees()
                .iter()
                .map(|tree| (tree.tree().clone(), tree.offset(), tree.shape().to_vec()))
                .collect(),
            col_trees: region
                .col_trees()
                .iter()
                .map(|tree| (tree.tree().clone(), tree.offset(), tree.shape().to_vec()))
                .collect(),
            data: Vec::new(),
        })
        .collect()
}

fn region_sector<R>(rule: &R, region: &CoupledSectorRegion) -> SectorId
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    region.coupled().unwrap_or_else(|| rule.vacuum())
}

fn sector_region_index_map<R>(
    rule: &R,
    regions: &[CoupledSectorRegion],
) -> Result<HashMap<SectorId, usize>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let mut by_sector = HashMap::with_capacity(regions.len());
    for (index, region) in regions.iter().enumerate() {
        if by_sector
            .insert(region_sector(rule, region), index)
            .is_some()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "coupled-sector region description contains a duplicate sector",
            });
        }
    }
    Ok(by_sector)
}

fn sector_region_index_of(
    regions: &HashMap<SectorId, usize>,
    sector: SectorId,
    side: &'static str,
) -> Result<usize, OperationError> {
    regions
        .get(&sector)
        .copied()
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: match side {
                "left" => "compact left factor is missing a nonzero-rank sector",
                _ => "compact right factor is missing a nonzero-rank sector",
            },
        })
}

fn validate_factor_region(
    region: &CoupledSectorRegion,
    rows: usize,
    cols: usize,
    side: &'static str,
) -> Result<(), OperationError> {
    if region.rows() != rows || region.cols() != cols {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: match side {
                "left" => "compact left sector region has an unexpected shape",
                _ => "compact right sector region has an unexpected shape",
            },
        });
    }
    Ok(())
}

fn validate_no_unused_factor_regions(
    regions: &[CoupledSectorRegion],
    used: &[bool],
    side: &'static str,
) -> Result<(), OperationError> {
    if regions
        .iter()
        .zip(used)
        .any(|(region, used)| !used && region.rows() != 0 && region.cols() != 0)
    {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: match side {
                "left" => "compact left factor contains an unused nonzero sector",
                _ => "compact right factor contains an unused nonzero sector",
            },
        });
    }
    Ok(())
}

/// Dynamic-rank [`svd_compact`]: the [`svd_compact_factors_dyn`] core plus the
/// diagonal `S` materialized as a `bond <- bond` tensor.
pub fn svd_compact_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdCompactDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (u, vh, singular_values) = svd_compact_factors_dyn(dense, input)?;
    let s = diagonal_bond_svd_factor(input.space(), &singular_values, &D::from_real)?;
    Ok(SvdCompactDyn {
        u,
        s,
        vh,
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

/// Applies a truncation policy to an untruncated compact factorization (the host
/// half of [`svd_trunc`]).
///
/// The decision is host-side scalar work over the spectra; the application
/// keeps the leading bond states per coupled sector, which in the coupled
/// layout is a per-sector leading-columns/rows copy (device kernel later).
#[cfg_attr(not(test), allow(dead_code))] // exercised by the typed test suite
pub(crate) fn truncate_svd<R, D, const NOUT: usize, const NIN: usize>(
    compact: SvdCompact<R, D, NOUT, NIN>,
    truncation: &Truncation,
) -> Result<SvdTrunc<R, D, NOUT, NIN>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let SvdCompact {
        u,
        s,
        vh,
        singular_values,
    } = compact;
    let (u_space, u) = u.into_parts();
    let (s_space, s) = s.into_parts();
    let (vh_space, vh) = vh.into_parts();
    let compact_dyn = SvdCompactDyn {
        u: BoundDynFactor::from_bound(u_space, u.data().to_vec(), NOUT, 1)?,
        s: BoundDynFactor::from_bound(s_space, s.data().to_vec(), 1, 1)?,
        vh: BoundDynFactor::from_bound(vh_space, vh.data().to_vec(), 1, NIN)?,
        singular_values,
    };
    let out = truncate_svd_dyn(compact_dyn, truncation)?;
    Ok(SvdTrunc {
        u: typed_from_bound_factor(out.u)?,
        s: typed_from_bound_factor(out.s)?,
        vh: typed_from_bound_factor(out.vh)?,
        singular_values: out.singular_values,
        error: out.error,
    })
}

/// Dynamic-rank [`truncate_svd`].
pub(crate) fn truncate_svd_dyn<R, D>(
    compact: SvdCompactDyn<R, D>,
    truncation: &Truncation,
) -> Result<SvdTruncDyn<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    truncate_svd_factors_dyn(
        compact.u,
        Some(compact.s),
        compact.vh,
        compact.singular_values,
        truncation,
    )
}

/// Decides and applies SVD truncation to compact factors, then materializes
/// the returned diagonal factor at the selected rank.
///
/// Why not always build `S` here: the public truncating path has no useful
/// untruncated diagonal to reuse. Why not always require a missing `S`: the
/// composed compact-then-truncate path can return its existing factor when the
/// decision keeps every value.
fn truncate_svd_factors_dyn<R, D>(
    u: BoundDynFactor<R, D>,
    untruncated_s: Option<BoundDynFactor<R, D>>,
    vh: BoundDynFactor<R, D>,
    mut singular_values: Vec<SectorSpectrum>,
    truncation: &Truncation,
) -> Result<SvdTruncDyn<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let decision = decide_bond_truncation(u.space().provider(), &singular_values, truncation, true);
    if singular_values
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        let s = match untruncated_s {
            Some(s) => s,
            None => diagonal_bond_svd_factor(u.space(), &singular_values, &D::from_real)?,
        };
        return Ok(SvdTruncDyn {
            u,
            s,
            vh,
            singular_values,
            error: decision.error,
        });
    }

    for (entry, &count) in singular_values.iter_mut().zip(&decision.kept) {
        entry.values.truncate(count);
    }
    singular_values.retain(|entry| !entry.values.is_empty());
    let kept_by_sector: HashMap<SectorId, usize> = singular_values
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();

    let kept_of = |sector: SectorId| -> usize { kept_by_sector.get(&sector).copied().unwrap_or(0) };

    let bond_axis = u.space().space().nout();
    let u_factor = sliced_bond_bound_factor(
        u.space(),
        u.data(),
        bond_axis,
        &kept_of,
        u.space().space().nout(),
        1,
    )?;
    let vh_factor = sliced_bond_bound_factor(
        vh.space(),
        vh.data(),
        0,
        &kept_of,
        1,
        vh.space().space().nin(),
    )?;
    let s_factor = diagonal_bond_svd_factor(u.space(), &singular_values, &D::from_real)?;
    Ok(SvdTruncDyn {
        u: u_factor,
        s: s_factor,
        vh: vh_factor,
        singular_values,
        error: decision.error,
    })
}

fn sliced_bond_bound_factor<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    source_data: &[D],
    axis: usize,
    kept_of: &dyn Fn(SectorId) -> usize,
    expected_nout: usize,
    expected_nin: usize,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = authority.provider();
    let source_space = authority.space();
    let nout = source_space.nout();
    let source_structure = Arc::clone(source_space.structure());
    let homspace = source_space.homspace();
    let leg = if axis < nout {
        &homspace.codomain().legs()[axis]
    } else {
        &homspace.domain().legs()[axis - nout]
    };
    let bond_leg = SectorLeg::new(
        leg.sectors()
            .iter()
            .copied()
            .filter(|&sector| kept_of(sector) > 0)
            .map(|sector| (sector, kept_of(sector))),
        false,
    );
    let new_hom = if axis < nout {
        let mut legs = homspace.codomain().legs().to_vec();
        legs[axis] = bond_leg;
        FusionTreeHomSpace::new(FusionProductSpace::new(legs), homspace.domain().clone())
    } else {
        let mut legs = homspace.domain().legs().to_vec();
        legs[axis - nout] = bond_leg;
        FusionTreeHomSpace::new(homspace.codomain().clone(), FusionProductSpace::new(legs))
    };
    let space = authority.derive_from_fusion_tree_shapes(new_hom, |keys| {
        keys.iter()
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
            .collect::<Result<Vec<_>, OperationError>>()
    })?;
    let mut data = vec![D::zero(); space.space().required_len()?];
    for index in 0..space.space().structure().block_count() {
        let new_block = space.space().structure().block(index)?;
        let old_index = source_structure
            .find_block_index_by_key(new_block.key())
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: "truncated factor tree must exist in the full factor",
            })?;
        let old_block = source_structure.block(old_index)?;
        copy_matching_block_prefix(
            source_data,
            old_block.strides(),
            old_block.offset(),
            &mut data,
            new_block.strides(),
            new_block.offset(),
            new_block.shape(),
        );
    }
    BoundDynFactor::from_bound(space, data, expected_nout, expected_nin)
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

fn build_left_right_bound_spaces<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[SectorMatricization<D>],
    ranks: &[SectorRank],
) -> Result<(BoundDynamicFusionMapSpace<R>, BoundDynamicFusionMapSpace<R>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = authority.provider();
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
    let left = authority.derive_from_fusion_tree_shapes(left_hom, |keys| {
        keys.iter()
            .map(|key| {
                let sector = coupled_of(rule, key.codomain_tree());
                let mut shape = row_shape_of(&matrix_by_sector, sector, key.codomain_tree())?;
                shape.push(sector_rank(sector));
                Ok(shape)
            })
            .collect::<Result<Vec<_>, OperationError>>()
    })?;

    let right_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg]),
        homspace.domain().clone(),
    );
    let right = authority.derive_from_fusion_tree_shapes(right_hom, |keys| {
        keys.iter()
            .map(|key| {
                let sector = coupled_of(rule, key.domain_tree());
                let mut shape = vec![sector_rank(sector)];
                shape.extend(col_shape_of(&matrix_by_sector, sector, key.domain_tree())?);
                Ok(shape)
            })
            .collect::<Result<Vec<_>, OperationError>>()
    })?;
    Ok((left, right))
}

fn build_left_bound_space<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[SectorMatricization<D>],
    ranks: &[SectorRank],
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = authority.provider();
    let rank_by_sector: HashMap<SectorId, usize> =
        ranks.iter().map(|rank| (rank.sector, rank.kept)).collect();
    let matrix_by_sector = matricization_map(matricizations);
    let new_leg = SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false);
    let hom = FusionTreeHomSpace::new(
        homspace.codomain().clone(),
        FusionProductSpace::new([new_leg]),
    );
    authority.derive_from_fusion_tree_shapes(hom, |keys| {
        keys.iter()
            .map(|key| {
                let sector = coupled_of(rule, key.codomain_tree());
                let mut shape = row_shape_of(&matrix_by_sector, sector, key.codomain_tree())?;
                shape.push(rank_by_sector.get(&sector).copied().unwrap_or(0));
                Ok(shape)
            })
            .collect::<Result<Vec<_>, OperationError>>()
    })
}

/// Builds the `(codomain <- W, W <- domain)` factor pair shared by SVD and
/// the orthogonal factorizations, in the coupled-sector matrix layout.
fn build_left_right_bound_pair<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[SectorMatricization<D>],
    pairs: &[FactorPair<D>],
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = authority.provider();
    let ranks = pairs
        .iter()
        .map(|pair| SectorRank {
            sector: pair.sector,
            kept: pair.kept,
        })
        .collect::<Vec<_>>();
    let (left_space, right_space) =
        build_left_right_bound_spaces(authority, homspace, matricizations, &ranks)?;
    let matrix_by_sector = matricization_map(matricizations);
    let pair_by_sector: HashMap<SectorId, &FactorPair<D>> =
        pairs.iter().map(|pair| (pair.sector, pair)).collect();
    let mut left_data = vec![D::zero(); left_space.space().required_len()?];
    for index in 0..left_space.space().structure().block_count() {
        let block = left_space.space().structure().block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = coupled_of(rule, key.codomain_tree());
        let matrix = matricization_of(&matrix_by_sector, sector)?;
        let pair = pair_by_sector[&sector];
        let (row_offset, _) = row_placement(matrix, key.codomain_tree())?;
        scatter_matrix_block(
            &mut left_data,
            block.shape(),
            block.strides(),
            block.offset(),
            block.shape().len() - 1,
            &pair.left,
            pair.left_rows,
            row_offset,
        );
    }
    let mut right_data = vec![D::zero(); right_space.space().required_len()?];
    for index in 0..right_space.space().structure().block_count() {
        let block = right_space.space().structure().block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = coupled_of(rule, key.domain_tree());
        let matrix = matricization_of(&matrix_by_sector, sector)?;
        let pair = pair_by_sector[&sector];
        let (col_offset, _) = col_placement(matrix, key.domain_tree())?;
        scatter_matrix_block(
            &mut right_data,
            block.shape(),
            block.strides(),
            block.offset(),
            0,
            &pair.right,
            pair.right_leading,
            col_offset,
        );
    }
    let left_nout = left_space.space().nout();
    let right_nin = right_space.space().nin();
    Ok((
        BoundDynFactor::from_bound(left_space, left_data, left_nout, 1)?,
        BoundDynFactor::from_bound(right_space, right_data, 1, right_nin)?,
    ))
}

fn build_left_bound_factor<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[SectorMatricization<D>],
    pairs: &[FactorPair<D>],
) -> Result<BoundDynFactor<R, D>, OperationError>
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
    let space = build_left_bound_space(authority, homspace, matricizations, &ranks)?;
    let mut data = vec![D::zero(); space.space().required_len()?];
    let pair_by_sector: HashMap<SectorId, &FactorPair<D>> =
        pairs.iter().map(|pair| (pair.sector, pair)).collect();
    for matrix in matricizations {
        let pair = pair_by_sector[&matrix.sector];
        scatter_left_sector_blocks(
            authority.provider(),
            space.space(),
            &mut data,
            matrix,
            &pair.left,
            pair.left_rows,
        )?;
    }
    let nout = space.space().nout();
    BoundDynFactor::from_bound(space, data, nout, 1)
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
pub struct EighFull<R, D, const NOUT: usize, const NIN: usize> {
    pub d: BoundTensorMap<R, D, 1, 1>,
    pub v: BoundTensorMap<R, D, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum>,
}

/// Dynamic-rank [`EighFull`]. Carries only the eigenvector map and the O(rank)
/// spectrum; the dense diagonal `D` is built on demand by the typed [`eigh_full`]
/// wrapper (which returns a `TensorMap`), so callers that keep `D` diagonal
/// (the user layer, via `Data::Diagonal`) never pay the O(rank²) materialization.
#[derive(Clone, Debug)]
pub struct EighFullDyn<R, D> {
    v: BoundDynFactor<R, D>,
    eigenvalues: Vec<SectorSpectrum>,
}

impl<R, D> EighFullDyn<R, D> {
    pub fn v(&self) -> &BoundDynFactor<R, D> {
        &self.v
    }

    pub fn eigenvalues(&self) -> &[SectorSpectrum] {
        &self.eigenvalues
    }

    pub fn into_parts(self) -> (BoundDynFactor<R, D>, Vec<SectorSpectrum>) {
        (self.v, self.eigenvalues)
    }
}

/// Truncated Hermitian eigendecomposition; `error` is the
/// quantum-dimension-weighted 2-norm of the discarded eigenvalues.
#[derive(Clone, Debug)]
pub struct EighTrunc<R, D, const NOUT: usize, const NIN: usize> {
    pub d: BoundTensorMap<R, D, 1, 1>,
    pub v: BoundTensorMap<R, D, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum>,
    pub error: f64,
}

/// Dynamic-rank [`EighTrunc`]. Spectrum + eigenvectors only; the dense diagonal
/// is materialized by the typed [`eigh_trunc`] wrapper (see [`EighFullDyn`]).
#[derive(Clone, Debug)]
pub struct EighTruncDyn<R, D> {
    v: BoundDynFactor<R, D>,
    eigenvalues: Vec<SectorSpectrum>,
    error: f64,
}

impl<R, D> EighTruncDyn<R, D> {
    pub fn v(&self) -> &BoundDynFactor<R, D> {
        &self.v
    }

    pub fn eigenvalues(&self) -> &[SectorSpectrum] {
        &self.eigenvalues
    }

    pub fn error(&self) -> f64 {
        self.error
    }

    pub fn into_parts(self) -> (BoundDynFactor<R, D>, Vec<SectorSpectrum>, f64) {
        (self.v, self.eigenvalues, self.error)
    }
}

/// Full Hermitian eigendecomposition through the device boundary.
pub fn eigh_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<EighFull<R, D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let dynamic = input.dynamic();
    let out = eigh_full_dyn(dense, &dynamic)?;
    // Materialize the dense diagonal here (the typed API returns a `TensorMap`);
    // the dyn producer no longer builds it (#56 item N).
    let d = diagonal_bond_svd_factor(dynamic.space(), &out.eigenvalues, &D::from_real)?;
    Ok(EighFull {
        d: typed_from_bound_factor(d)?,
        v: typed_from_bound_factor(out.v)?,
        eigenvalues: out.eigenvalues,
    })
}

/// Dynamic-rank [`eigh_full`]: the shared core of the Hermitian entries.
pub fn eigh_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<EighFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = input.space().provider();
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires an endomorphism (codomain == domain)",
        });
    }
    if let Some(plan) = compact_factor_plan(input.space())? {
        return eigh_full_direct_regions(dense, input, &plan);
    }
    let matricizations =
        sector_matricizations(rule, space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_eigh_input_pack(&matricizations);

    let ranks = matricizations
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector,
            kept: matrix.rows,
        })
        .collect::<Vec<_>>();
    let v_space = build_left_bound_space(input.space(), space.homspace(), &matricizations, &ranks)?;
    let v_len = v_space
        .space()
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
        scatter_left_sector_blocks(
            rule,
            v_space.space(),
            &mut v_data,
            matrix,
            &sorted_vectors,
            n,
        )?;
        #[cfg(test)]
        record_eigh_output_scatter::<D>(n * n);
    }

    Ok(EighFullDyn {
        v: BoundDynFactor::from_bound(v_space, v_data, space.nout(), 1)?,
        eigenvalues,
    })
}

fn eigh_full_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
) -> Result<EighFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source.layout, input.space().validated_layout());
    if plan
        .source
        .regions
        .iter()
        .any(|region| region.rows() != region.cols())
    {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires square coupled-sector matrices",
        });
    }

    let v_space = input.space().rebind_validated(&plan.left_layout)?;
    let mut v_data = vec![D::zero(); plan.left_layout.required_len()?];
    let max_n = plan
        .source
        .regions
        .iter()
        .map(CoupledSectorRegion::rows)
        .max()
        .unwrap_or(0);
    let mut values_workspace = vec![D::Real::zero(); max_n];
    let mut order = Vec::with_capacity(max_n);
    let mut visited = vec![false; max_n];
    let mut column_scratch = vec![D::zero(); max_n];
    let mut eigenvalues = Vec::with_capacity(plan.routes.len());

    for route in plan.routes.iter().copied() {
        let source = &plan.source.regions[route.source_region];
        let n = source.rows();
        if n == 0 {
            eigenvalues.push(SectorSpectrum {
                sector: route.sector,
                values: Vec::new(),
            });
            continue;
        }
        let left = &plan.left_regions[route.left_region.expect("nonzero route has left region")];
        let input_shape = [n, n];
        let input_strides = [1usize, n];
        let values_shape = [n];
        let values_strides = [1usize];
        let vectors_shape = [n, n];
        let vectors_strides = [1usize, n];
        let input_view = DenseView::new(
            &input.data()[source.range()],
            &input_shape,
            &input_strides,
            0,
        )
        .map_err(OperationError::Dense)?;
        let values_view = DenseViewMut::new(
            &mut values_workspace[..n],
            &values_shape,
            &values_strides,
            0,
        )
        .map_err(OperationError::Dense)?;
        let vectors_view = DenseViewMut::new(
            &mut v_data[left.range()],
            &vectors_shape,
            &vectors_strides,
            0,
        )
        .map_err(OperationError::Dense)?;
        dense
            .eigh_into(
                D::dense_read(input_view),
                D::Real::dense_write(values_view),
                D::dense_write(vectors_view),
            )
            .map_err(OperationError::Dense)?;

        order.clear();
        order.extend(0..n);
        order.sort_by(|&a, &b| {
            let a_value: f64 = values_workspace[a].into();
            let b_value: f64 = values_workspace[b].into();
            b_value
                .abs()
                .partial_cmp(&a_value.abs())
                .expect("finite eigenvalues")
                .then(a.cmp(&b))
        });
        let sorted_values = order
            .iter()
            .map(|&index| values_workspace[index].into())
            .collect();
        reorder_columns_in_place(
            &mut v_data[left.range()],
            n,
            &order,
            &mut visited,
            &mut column_scratch,
        );
        eigenvector_gauge(&mut v_data[left.range()], n, n, n);
        eigenvalues.push(SectorSpectrum {
            sector: route.sector,
            values: sorted_values,
        });
    }

    Ok(EighFullDyn {
        v: BoundDynFactor::from_bound(v_space, v_data, space.nout(), 1)?,
        eigenvalues,
    })
}

/// Truncated Hermitian eigendecomposition: [`eigh_full`] on the device
/// boundary plus the shared host-side truncation by `|eigenvalue|`.
pub fn eigh_trunc<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
    truncation: &Truncation,
) -> Result<EighTrunc<R, D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let dynamic = input.dynamic();
    let out = eigh_trunc_dyn(dense, &dynamic, truncation)?;
    let d = diagonal_bond_svd_factor(dynamic.space(), &out.eigenvalues, &D::from_real)?;
    Ok(EighTrunc {
        d: typed_from_bound_factor(d)?,
        v: typed_from_bound_factor(out.v)?,
        eigenvalues: out.eigenvalues,
        error: out.error,
    })
}

/// Dynamic-rank [`eigh_trunc`].
pub fn eigh_trunc_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    truncation: &Truncation,
) -> Result<EighTruncDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = input.space().provider();
    let full = eigh_full_dyn(dense, input)?;
    if matches!(truncation, Truncation::Full) {
        return Ok(EighTruncDyn {
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
    let bond_axis = full.v.space().space().nout();
    let v_factor = sliced_bond_bound_factor(
        full.v.space(),
        full.v.data(),
        bond_axis,
        &kept_of,
        bond_axis,
        1,
    )?;
    Ok(EighTruncDyn {
        v: v_factor,
        eigenvalues,
        error: decision.error,
    })
}

/// Full fusion-tensor SVD `t = U * S * Vh` (MatrixAlgebraKit `svd_full`):
/// per sector `U` is the square `m x m` unitary, `S` the rectangular
/// `m x n` diagonal, and `Vh` the square `n x n` unitary.
#[derive(Clone, Debug)]
pub struct SvdFull<R, D, const NOUT: usize, const NIN: usize> {
    pub u: BoundTensorMap<R, D, NOUT, 1>,
    pub s: BoundTensorMap<R, D, 1, 1>,
    pub vh: BoundTensorMap<R, D, 1, NIN>,
    pub singular_values: Vec<SectorSpectrum>,
}

/// Dynamic-rank [`SvdFull`].
#[derive(Clone, Debug)]
pub struct SvdFullDyn<R, D> {
    u: BoundDynFactor<R, D>,
    s: BoundDynFactor<R, D>,
    vh: BoundDynFactor<R, D>,
    singular_values: Vec<SectorSpectrum>,
}

impl<R, D> SvdFullDyn<R, D> {
    pub fn u(&self) -> &BoundDynFactor<R, D> {
        &self.u
    }
    pub fn s(&self) -> &BoundDynFactor<R, D> {
        &self.s
    }
    pub fn vh(&self) -> &BoundDynFactor<R, D> {
        &self.vh
    }
    pub fn singular_values(&self) -> &[SectorSpectrum] {
        &self.singular_values
    }
    pub fn into_parts(
        self,
    ) -> (
        BoundDynFactor<R, D>,
        BoundDynFactor<R, D>,
        BoundDynFactor<R, D>,
        Vec<SectorSpectrum>,
    ) {
        (self.u, self.s, self.vh, self.singular_values)
    }
}

/// Full fusion-tensor SVD through the device boundary.
///
/// The unitaries are completed from the compact factors with an extra
/// economy QR of `[U1 | I]` per sector (any orthonormal completion is exact
/// because the corresponding rows/columns of `S` are zero), so the whole
/// computation stays on the existing dense-executor boundary.
pub fn svd_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<SvdFull<R, D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let out = svd_full_dyn(dense, &input.dynamic())?;
    Ok(SvdFull {
        u: typed_from_bound_factor(out.u)?,
        s: typed_from_bound_factor(out.s)?,
        vh: typed_from_bound_factor(out.vh)?,
        singular_values: out.singular_values,
    })
}

/// Dynamic-rank [`svd_full`].
pub fn svd_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = input.space().provider();
    let space = input.space().space();
    let matricizations =
        sector_matricizations(rule, space.structure(), input.data(), space.nout())?;

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
    let (u_factor, _) = build_left_right_bound_pair(
        input.space(),
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
    let (_, vh_factor) = build_left_right_bound_pair(
        input.space(),
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
    let s_factor =
        rectangular_diagonal_bond_tensor(input.space(), &singular_values, &rows_of, &cols_of)?;
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
    E: DenseExecutor + ?Sized,
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
    authority: &BoundDynamicFusionMapSpace<R>,
    spectra: &[SectorSpectrum],
    rows_of: &dyn Fn(SectorId) -> usize,
    cols_of: &dyn Fn(SectorId) -> usize,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = authority.provider();
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
    let space = authority.derive_from_fusion_tree_shapes(homspace, |keys| {
        Ok(keys
            .iter()
            .map(|key| {
                let sector = coupled_of(rule, key.codomain_tree());
                vec![rows_of(sector), cols_of(sector)]
            })
            .collect::<Vec<_>>())
    })?;
    let len = space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); len];
    let structure = Arc::clone(space.space().structure());
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
    BoundDynFactor::from_bound(space, data, 1, 1)
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

fn reorder_columns_in_place<D: Copy>(
    vectors: &mut [D],
    n: usize,
    order: &[usize],
    visited: &mut [bool],
    scratch: &mut [D],
) {
    visited[..n].fill(false);
    for start in 0..n {
        if visited[start] {
            continue;
        }
        scratch[..n].copy_from_slice(&vectors[start * n..(start + 1) * n]);
        let mut destination = start;
        loop {
            visited[destination] = true;
            let source = order[destination];
            if source == start {
                vectors[destination * n..(destination + 1) * n].copy_from_slice(&scratch[..n]);
                break;
            }
            for row in 0..n {
                vectors[destination * n + row] = vectors[source * n + row];
            }
            destination = source;
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
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<(BoundTensorMap<R, D, NOUT, 1>, BoundTensorMap<R, D, 1, NIN>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (q, r) = qr_full_dyn(dense, &input.dynamic())?;
    Ok((typed_from_bound_factor(q)?, typed_from_bound_factor(r)?))
}

/// Provider-bound dynamic-rank [`qr_full`].
pub fn qr_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = sector_matricizations(
        provider.as_ref(),
        space.structure(),
        input.data(),
        space.nout(),
    )?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let mut augmented = vec![D::zero(); rows * (cols + rows)];
        augmented[..rows * cols].copy_from_slice(&matrix.data);
        for row in 0..rows {
            augmented[rows * cols + row * rows + row] = D::one();
        }
        let mut q = vec![D::zero(); rows * rows];
        let mut work_r = vec![D::zero(); rows * (cols + rows)];
        qr_into_workspace(
            dense,
            &augmented,
            rows,
            cols + rows,
            rows,
            &mut q,
            rows,
            rows,
            rows,
            &mut work_r,
            rows,
            cols + rows,
            rows,
        )?;
        let mut r = work_r[..rows * cols].to_vec();
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
    build_left_right_bound_pair(input.space(), space.homspace(), &matrices, &pairs)
}

/// Full LQ `t = L * Q` (MatrixAlgebraKit `lq_full`): per sector `L` is the
/// lower-trapezoidal `m x n` and `Q` the square `n x n` unitary, via the full
/// QR of the transposed sector matrices.
/// The positive-diagonal gauge is applied (MAK / TensorKit 0.17 default).
pub fn lq_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<(BoundTensorMap<R, D, NOUT, 1>, BoundTensorMap<R, D, 1, NIN>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (l, q) = lq_full_dyn(dense, &input.dynamic())?;
    Ok((typed_from_bound_factor(l)?, typed_from_bound_factor(q)?))
}

/// Provider-bound dynamic-rank [`lq_full`].
pub fn lq_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = sector_matricizations(
        provider.as_ref(),
        space.structure(),
        input.data(),
        space.nout(),
    )?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let transposed = adjoint_col_major(&matrix.data, rows, cols);
        let mut augmented = vec![D::zero(); cols * (rows + cols)];
        augmented[..cols * rows].copy_from_slice(&transposed);
        for row in 0..cols {
            augmented[cols * rows + row * cols + row] = D::one();
        }
        let mut q_prime = vec![D::zero(); cols * cols];
        let mut work_r = vec![D::zero(); cols * (rows + cols)];
        qr_into_workspace(
            dense,
            &augmented,
            cols,
            rows + cols,
            cols,
            &mut q_prime,
            cols,
            cols,
            cols,
            &mut work_r,
            cols,
            rows + cols,
            cols,
        )?;
        let mut r_prime = work_r[..cols * rows].to_vec();
        positive_diagonal_gauge(&mut q_prime, cols, &mut r_prime, cols, rows);
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: cols,
            left: adjoint_col_major(&r_prime, cols, rows),
            left_rows: rows,
            right: adjoint_col_major(&q_prime, cols, cols),
            right_leading: cols,
        });
    }
    build_left_right_bound_pair(input.space(), space.homspace(), &matrices, &pairs)
}

/// Full general eigendecomposition `t = V * D * V^-1` (MatrixAlgebraKit
/// `eig_full`): always complex, requires an endomorphism. Bond states are
/// stored descending by `|eigenvalue|` per sector.
#[derive(Clone, Debug)]
pub struct EigFull<R, D: FactorScalar, const NOUT: usize, const NIN: usize> {
    pub d: BoundTensorMap<R, D::Eig, 1, 1>,
    pub v: BoundTensorMap<R, D::Eig, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum<Complex64>>,
}

/// Dynamic-rank [`EigFull`]. Spectrum + eigenvectors only; the dense diagonal
/// is materialized by the typed [`eig_full`] wrapper (see [`EighFullDyn`], #56 N).
#[derive(Clone, Debug)]
pub struct EigFullDyn<R, D: FactorScalar> {
    v: BoundDynFactor<R, D::Eig>,
    eigenvalues: Vec<SectorSpectrum<Complex64>>,
}

impl<R, D: FactorScalar> EigFullDyn<R, D> {
    pub fn v(&self) -> &BoundDynFactor<R, D::Eig> {
        &self.v
    }

    pub fn eigenvalues(&self) -> &[SectorSpectrum<Complex64>] {
        &self.eigenvalues
    }

    pub fn into_parts(self) -> (BoundDynFactor<R, D::Eig>, Vec<SectorSpectrum<Complex64>>) {
        (self.v, self.eigenvalues)
    }
}

/// Truncated general eigendecomposition; `error` is the
/// quantum-dimension-weighted 2-norm of the discarded `|eigenvalues|`.
#[derive(Clone, Debug)]
pub struct EigTrunc<R, D: FactorScalar, const NOUT: usize, const NIN: usize> {
    pub d: BoundTensorMap<R, D::Eig, 1, 1>,
    pub v: BoundTensorMap<R, D::Eig, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum<Complex64>>,
    pub error: f64,
}

/// Dynamic-rank [`EigTrunc`]. Spectrum + eigenvectors only; the dense diagonal
/// is materialized by the typed [`eig_trunc`] wrapper (see [`EighFullDyn`], #56 N).
#[derive(Clone, Debug)]
pub struct EigTruncDyn<R, D: FactorScalar> {
    v: BoundDynFactor<R, D::Eig>,
    eigenvalues: Vec<SectorSpectrum<Complex64>>,
    error: f64,
}

impl<R, D: FactorScalar> EigTruncDyn<R, D> {
    pub fn v(&self) -> &BoundDynFactor<R, D::Eig> {
        &self.v
    }

    pub fn eigenvalues(&self) -> &[SectorSpectrum<Complex64>] {
        &self.eigenvalues
    }

    pub fn error(&self) -> f64 {
        self.error
    }

    pub fn into_parts(
        self,
    ) -> (
        BoundDynFactor<R, D::Eig>,
        Vec<SectorSpectrum<Complex64>>,
        f64,
    ) {
        (self.v, self.eigenvalues, self.error)
    }
}

/// Full general eigendecomposition through the device boundary.
pub fn eig_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<EigFull<R, D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let dynamic = input.dynamic();
    let out = eig_full_dyn::<E, R, D>(dense, &dynamic)?;
    // Materialize the dense diagonal here (typed API returns a `TensorMap`); the
    // dyn producer no longer builds it (#56 item N).
    let d = diagonal_bond_svd_factor(
        dynamic.space(),
        &out.eigenvalues,
        &<D::Eig as FactorScalar>::from_complex64,
    )?;
    Ok(EigFull {
        d: typed_from_bound_factor(d)?,
        v: typed_from_bound_factor(out.v)?,
        eigenvalues: out.eigenvalues,
    })
}

/// Dynamic-rank [`eig_full`].
pub fn eig_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<EigFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = input.space().provider();
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eig requires an endomorphism (codomain == domain)",
        });
    }
    let matricizations =
        sector_matricizations(rule, space.structure(), input.data(), space.nout())?;

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
    let v_factor = build_left_bound_factor(
        input.space(),
        space.homspace(),
        &complex_matricizations,
        &pairs,
    )?;
    Ok(EigFullDyn {
        v: v_factor,
        eigenvalues,
    })
}

/// Truncated general eigendecomposition: [`eig_full`] plus the shared
/// host-side truncation by `|eigenvalue|`.
pub fn eig_trunc<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
    truncation: &Truncation,
) -> Result<EigTrunc<R, D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let dynamic = input.dynamic();
    let out = eig_trunc_dyn::<E, R, D>(dense, &dynamic, truncation)?;
    let d = diagonal_bond_svd_factor(
        dynamic.space(),
        &out.eigenvalues,
        &<D::Eig as FactorScalar>::from_complex64,
    )?;
    Ok(EigTrunc {
        d: typed_from_bound_factor(d)?,
        v: typed_from_bound_factor(out.v)?,
        eigenvalues: out.eigenvalues,
        error: out.error,
    })
}

/// Dynamic-rank [`eig_trunc`].
pub fn eig_trunc_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    truncation: &Truncation,
) -> Result<EigTruncDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = input.space().provider();
    let full = eig_full_dyn::<E, R, D>(dense, input)?;
    if matches!(truncation, Truncation::Full) {
        return Ok(EigTruncDyn {
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
    let bond_axis = full.v.space().space().nout();
    let v_factor = sliced_bond_bound_factor(
        full.v.space(),
        full.v.data(),
        bond_axis,
        &kept_of,
        bond_axis,
        1,
    )?;
    Ok(EigTruncDyn {
        v: v_factor,
        eigenvalues,
        error: decision.error,
    })
}

/// All Hermitian eigenvalues per coupled sector, descending by magnitude
/// (MatrixAlgebraKit `eigh_vals`).
pub fn eigh_vals<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    eigh_vals_dyn(dense, &input.dynamic())
}

/// Dynamic-rank [`eigh_vals`].
pub fn eigh_vals_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = input.space().provider();
    let space = input.space().space();
    // Values-only: per sector call the no-vector Hermitian eig (`eigh_vals`,
    // LAPACK `job='N'`) and keep the spectrum sorted descending by magnitude.
    // Skips the eigenvector space/buffer, the vector reorder, gauge-fixing, and
    // the block scatter that `eigh_full_dyn` did only to discard here. The sort
    // is stable, so equal-magnitude ties keep LAPACK order — bit-for-bit the
    // ordering `eigh_full_dyn` produces (it breaks ties by original index).
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires an endomorphism (codomain == domain)",
        });
    }
    let matricizations =
        sector_matricizations(rule, space.structure(), input.data(), space.nout())?;
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let n = matrix.rows;
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(&matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let values_tensor = dense
            .eigh_vals(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        let mut sorted = D::real_spectrum(&values_tensor).map_err(OperationError::Dense)?;
        sorted.truncate(n);
        sorted.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).expect("finite eigenvalues"));
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted,
        });
    }
    Ok(eigenvalues)
}

/// All general eigenvalues per coupled sector, descending by magnitude
/// (MatrixAlgebraKit `eig_vals`).
pub fn eig_vals<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<Vec<SectorSpectrum<Complex64>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    eig_vals_dyn::<E, R, D>(dense, &input.dynamic())
}

/// Dynamic-rank [`eig_vals`].
pub fn eig_vals_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum<Complex64>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = input.space().provider();
    let space = input.space().space();
    // Values-only: per sector call the no-vector general eig (`eig_vals`, LAPACK
    // `job='N'`) and keep the complex spectrum sorted descending by magnitude.
    // Skips the eigenvector reorder, gauge-fixing, and the factor-pair block
    // assembly that `eig_full_dyn` did only to discard here. LAPACK's QR
    // iteration yields the same eigenvalues regardless of `jobvr`, and the sort
    // is stable, so this matches `eig_full_dyn`'s ordering bit-for-bit.
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eig requires an endomorphism (codomain == domain)",
        });
    }
    let matricizations =
        sector_matricizations(rule, space.structure(), input.data(), space.nout())?;
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let n = matrix.rows;
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(&matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let values_tensor = dense
            .eig_vals(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        validate_dense_shape(values_tensor.shape(), &[n])?;
        let values =
            <D::Eig as FactorScalar>::dense_slice(&values_tensor).map_err(OperationError::Dense)?;
        let mut sorted: Vec<Complex64> = values[..n].iter().map(|&v| v.widen_complex()).collect();
        sorted.sort_by(|a, b| b.norm().partial_cmp(&a.norm()).expect("finite eigenvalues"));
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted,
        });
    }
    Ok(eigenvalues)
}

/// Left null space `N : codomain <- W` (MatrixAlgebraKit `left_null`).
///
/// Each sector uses its compact SVD and treats `sigma` as nonzero exactly when
/// `sigma > epsilon(dtype) * max(rows, cols) * sigma_max`. The returned columns
/// are the orthonormal complement after that numerical rank; sectors with no
/// null directions drop out of `W`.
pub fn left_null<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<BoundTensorMap<R, D, NOUT, 1>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let out = left_null_dyn(dense, &input.dynamic())?;
    typed_from_bound_factor(out)
}

/// Provider-bound dynamic-rank [`left_null`].
pub fn left_null_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = sector_matricizations(
        provider.as_ref(),
        space.structure(),
        input.data(),
        space.nout(),
    )?;
    let mut pairs = Vec::new();
    for matrix in &matrices {
        let (rows, cols) = (matrix.rows, matrix.cols);
        let (rank, u_compact, _) =
            numerical_rank_and_compact_bases(dense, &matrix.data, rows, cols)?;
        if rank == rows {
            continue;
        }
        // Only the left basis is completed: completing V would run an unused
        // QR for this operation.
        let u = orthonormal_completion(dense, &u_compact, rows, rows.min(cols))?;
        let null_dim = rows - rank;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: null_dim,
            left: u[rows * rank..].to_vec(),
            left_rows: rows,
            right: vec![D::zero(); null_dim * cols],
            right_leading: null_dim,
        });
    }
    let (null, _) =
        build_left_right_bound_pair(input.space(), space.homspace(), &matrices, &pairs)?;
    Ok(null)
}

/// Right null space `N : W <- domain` (MatrixAlgebraKit `right_null`).
///
/// Each sector uses its compact SVD and treats `sigma` as nonzero exactly when
/// `sigma > epsilon(dtype) * max(rows, cols) * sigma_max`. The returned rows
/// span the kernel after that numerical rank; sectors with no null directions
/// drop out of `W`.
pub fn right_null<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<BoundTensorMap<R, D, 1, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let out = right_null_dyn(dense, &input.dynamic())?;
    typed_from_bound_factor(out)
}

/// Provider-bound dynamic-rank [`right_null`].
pub fn right_null_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = sector_matricizations(
        provider.as_ref(),
        space.structure(),
        input.data(),
        space.nout(),
    )?;
    let mut pairs = Vec::new();
    for matrix in &matrices {
        let (rows, cols) = (matrix.rows, matrix.cols);
        let (rank, _, v_compact) =
            numerical_rank_and_compact_bases(dense, &matrix.data, rows, cols)?;
        if rank == cols {
            continue;
        }
        // Only the right basis is completed: completing U would run an unused
        // QR for this operation.
        let v = orthonormal_completion(dense, &v_compact, cols, rows.min(cols))?;
        let null_dim = cols - rank;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: null_dim,
            left: vec![D::zero(); rows * null_dim],
            left_rows: rows,
            right: adjoint_col_major(&v[cols * rank..], cols, null_dim),
            right_leading: null_dim,
        });
    }
    let (_, null) =
        build_left_right_bound_pair(input.space(), space.homspace(), &matrices, &pairs)?;
    Ok(null)
}

/// Computes compact singular-vector bases and the documented numerical rank.
fn numerical_rank_and_compact_bases<E, D>(
    dense: &mut E,
    matrix: &[D],
    rows: usize,
    cols: usize,
) -> Result<(usize, Vec<D>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let compact_rank = rows.min(cols);
    let mut u = vec![D::zero(); rows * compact_rank];
    let mut singular_values = vec![D::Real::zero(); compact_rank];
    let mut vh = vec![D::zero(); compact_rank * cols];
    let input_shape = [rows, cols];
    let input_strides = [1usize, rows];
    let u_shape = [rows, compact_rank];
    let u_strides = [1usize, rows];
    let s_shape = [compact_rank];
    let s_strides = [1usize];
    let vh_shape = [compact_rank, cols];
    let vh_strides = [1usize, compact_rank];
    let input =
        DenseView::new(matrix, &input_shape, &input_strides, 0).map_err(OperationError::Dense)?;
    let u_view =
        DenseViewMut::new(&mut u, &u_shape, &u_strides, 0).map_err(OperationError::Dense)?;
    let s_view = DenseViewMut::new(&mut singular_values, &s_shape, &s_strides, 0)
        .map_err(OperationError::Dense)?;
    let vh_view =
        DenseViewMut::new(&mut vh, &vh_shape, &vh_strides, 0).map_err(OperationError::Dense)?;
    dense
        .svd_into(
            D::dense_read(input),
            D::dense_write(u_view),
            D::Real::dense_write(s_view),
            D::dense_write(vh_view),
        )
        .map_err(OperationError::Dense)?;

    let sigma_max = singular_values
        .first()
        .copied()
        .map(Into::into)
        .unwrap_or(0.0);
    // Why not exact-zero rank: backward-stable SVD represents dependent
    // directions at working precision, not necessarily as bitwise zero.
    let tolerance = D::epsilon() * rows.max(cols) as f64 * sigma_max;
    let rank = singular_values
        .iter()
        .copied()
        .map(Into::into)
        .filter(|&sigma| sigma > tolerance)
        .count();
    let v_compact = adjoint_col_major(&vh, compact_rank, cols);
    Ok((rank, u, v_compact))
}

/// Left polar decomposition `t = W * P` (MatrixAlgebraKit `left_polar`):
/// `W` is the isometry `U * Vh` and `P = V * S * Vh` the positive part on
/// the domain.
pub fn left_polar<E, RuleKey, BT, BC, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<
    (
        BoundTensorMap<R, D, NOUT, NIN>,
        BoundTensorMap<R, D, NIN, NIN>,
    ),
    OperationError,
>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let (w, p) = left_polar_dyn(dense, context, &input.dynamic())?;
    Ok((typed_from_bound_factor(w)?, typed_from_bound_factor(p)?))
}

/// Dynamic-rank [`left_polar`].
pub fn left_polar_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    // Polar needs only U, Vh and the spectrum — not the dense diagonal S — so
    // use the S-free factors core.
    let (u, vh, singular_values) = svd_compact_factors_dyn(dense, input)?;
    let isometry = crate::compose::compose_bound_dyn(context, &u, &vh)?;
    // P = V·S·Vh. Fold S into V as a block-local scaling of V's bond (trailing)
    // axis — TensorKit's `DiagonalTensorMap` `rmul!` — instead of a full block
    // GEMM against the dense diagonal S (99% zeros). `singular_values` carries S
    // in O(rank); see #51 / #55.
    let mut v = adjoint_bound_factor(&vh)?;
    let v_space = v.space().space().clone();
    scale_axis_by_spectrum(&v_space, v.data_mut(), None, &singular_values)?;
    let positive = crate::compose::compose_bound_dyn(context, &v, &vh)?;
    Ok((isometry, positive))
}

/// Right polar decomposition `t = P * W` (MatrixAlgebraKit `right_polar`):
/// `P = U * S * U^H` is the positive part on the codomain and `W = U * Vh`.
pub fn right_polar<E, RuleKey, BT, BC, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<
    (
        BoundTensorMap<R, D, NOUT, NOUT>,
        BoundTensorMap<R, D, NOUT, NIN>,
    ),
    OperationError,
>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let (p, w) = right_polar_dyn(dense, context, &input.dynamic())?;
    Ok((typed_from_bound_factor(p)?, typed_from_bound_factor(w)?))
}

/// Dynamic-rank [`right_polar`].
pub fn right_polar_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    // Polar needs only U, Vh and the spectrum — not the dense diagonal S — so
    // use the S-free factors core.
    let (u, vh, singular_values) = svd_compact_factors_dyn(dense, input)?;
    let uh = adjoint_bound_factor(&u)?;
    let isometry = crate::compose::compose_bound_dyn(context, &u, &vh)?;
    // P = U·S·Uh. Fold S into U's bond (trailing) axis by block-local scaling —
    // TensorKit's `DiagonalTensorMap` `rmul!` — instead of a full block GEMM
    // against the dense diagonal S. U is consumed above for the isometry, so
    // scale the moved-out copy. `singular_values` carries S in O(rank); #51/#55.
    let mut us = u;
    let us_space = us.space().space().clone();
    scale_axis_by_spectrum(&us_space, us.data_mut(), None, &singular_values)?;
    let positive = crate::compose::compose_bound_dyn(context, &us, &uh)?;
    Ok((positive, isometry))
}

/// Compact QR `t = Q * R` (MatrixAlgebraKit `qr_compact`):
/// `Q : codomain <- W` has orthonormal columns per coupled sector and
/// `R : W <- domain` with per-sector bond `min(rows, cols)`. The
/// positive-diagonal gauge is applied (MAK / TensorKit 0.17 default
/// `positive = true`): `R`'s diagonal is real non-negative per sector.
pub fn qr_compact<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<(BoundTensorMap<R, D, NOUT, 1>, BoundTensorMap<R, D, 1, NIN>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (q, r) = qr_compact_dyn(dense, &input.dynamic())?;
    Ok((typed_from_bound_factor(q)?, typed_from_bound_factor(r)?))
}

/// Provider-bound compact QR used by authority-preserving callers.
pub fn qr_compact_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    if let Some(plan) = compact_factor_plan(input.space())? {
        return qr_compact_direct_regions(dense, input, &plan);
    }
    let provider = input.space().provider_arc();
    let rule = provider.as_ref();
    let space = input.space().space();
    let matricizations =
        sector_matricizations(rule, space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_qr_input_pack(&matricizations);
    let mut pairs = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let rank = matrix.rows.min(matrix.cols);
        let mut q = vec![D::zero(); matrix.rows * rank];
        let mut r = vec![D::zero(); rank * matrix.cols];
        qr_into_workspace(
            dense,
            &matrix.data,
            matrix.rows,
            matrix.cols,
            matrix.rows,
            &mut q,
            matrix.rows,
            rank,
            matrix.rows,
            &mut r,
            rank,
            matrix.cols,
            rank,
        )?;
        positive_diagonal_gauge_strided(
            &mut q,
            matrix.rows,
            matrix.rows,
            &mut r,
            rank,
            rank,
            matrix.cols,
        );
        #[cfg(test)]
        {
            record_compact_qr_output_scatter::<D>(q.len());
            record_compact_qr_output_scatter::<D>(r.len());
        }
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rank,
            left: q,
            left_rows: matrix.rows,
            right: r,
            right_leading: rank,
        });
    }
    build_left_right_bound_pair(input.space(), space.homspace(), &matricizations, &pairs)
}

fn qr_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source.layout, input.space().validated_layout());
    let left_space = input.space().rebind_validated(&plan.left_layout)?;
    let right_space = input.space().rebind_validated(&plan.right_layout)?;
    let mut left_data = vec![D::zero(); plan.left_layout.required_len()?];
    let mut right_data = vec![D::zero(); plan.right_layout.required_len()?];

    for route in plan.routes.iter().copied() {
        if route.rank == 0 {
            continue;
        }
        let source = &plan.source.regions[route.source_region];
        let left = &plan.left_regions[route.left_region.expect("nonzero route has left region")];
        let right =
            &plan.right_regions[route.right_region.expect("nonzero route has right region")];
        qr_into_workspace(
            dense,
            &input.data()[source.range()],
            source.rows(),
            source.cols(),
            source.rows(),
            &mut left_data[left.range()],
            source.rows(),
            route.rank,
            source.rows(),
            &mut right_data[right.range()],
            route.rank,
            source.cols(),
            route.rank,
        )?;
        positive_diagonal_gauge_strided(
            &mut left_data[left.range()],
            source.rows(),
            source.rows(),
            &mut right_data[right.range()],
            route.rank,
            route.rank,
            source.cols(),
        );
    }

    let left = BoundDynFactor::from_bound(left_space, left_data, space.nout(), 1)?;
    let right = BoundDynFactor::from_bound(right_space, right_data, 1, space.nin())?;
    Ok((left, right))
}

/// Compact LQ `t = L * Q` (MatrixAlgebraKit `lq_compact`, via the QR of the
/// transposed sector matrices): `Q : W <- domain` has orthonormal rows per
/// coupled sector and `L : codomain <- W`. The positive-diagonal gauge is
/// applied (MAK / TensorKit 0.17 default `positive = true`): `L`'s diagonal
/// is real non-negative per sector.
pub fn lq_compact<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<(BoundTensorMap<R, D, NOUT, 1>, BoundTensorMap<R, D, 1, NIN>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (l, q) = lq_compact_dyn(dense, &input.dynamic())?;
    Ok((typed_from_bound_factor(l)?, typed_from_bound_factor(q)?))
}

/// Provider-bound compact LQ used by authority-preserving callers.
pub fn lq_compact_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    if let Some(plan) = compact_factor_plan(input.space())? {
        return lq_compact_direct_regions(dense, input, &plan);
    }
    let provider = input.space().provider_arc();
    let rule = provider.as_ref();
    let space = input.space().space();
    let matricizations =
        sector_matricizations(rule, space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_lq_input_pack(&matricizations);
    let mut pairs = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let rank = matrix.rows.min(matrix.cols);
        let adjoint = adjoint_col_major(&matrix.data, matrix.rows, matrix.cols);
        let mut q_prime = vec![D::zero(); matrix.cols * rank];
        let mut r_prime = vec![D::zero(); rank * matrix.rows];
        qr_into_workspace(
            dense,
            &adjoint,
            matrix.cols,
            matrix.rows,
            matrix.cols,
            &mut q_prime,
            matrix.cols,
            rank,
            matrix.cols,
            &mut r_prime,
            rank,
            matrix.rows,
            rank,
        )?;
        positive_diagonal_gauge_strided(
            &mut q_prime,
            matrix.cols,
            matrix.cols,
            &mut r_prime,
            rank,
            rank,
            matrix.rows,
        );
        #[cfg(test)]
        {
            record_compact_lq_output_scatter::<D>(r_prime.len());
            record_compact_lq_output_scatter::<D>(q_prime.len());
        }
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rank,
            left: adjoint_col_major(&r_prime, rank, matrix.rows),
            left_rows: matrix.rows,
            right: adjoint_col_major(&q_prime, matrix.cols, rank),
            right_leading: rank,
        });
    }
    build_left_right_bound_pair(input.space(), space.homspace(), &matricizations, &pairs)
}

fn lq_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source.layout, input.space().validated_layout());
    let left_space = input.space().rebind_validated(&plan.left_layout)?;
    let right_space = input.space().rebind_validated(&plan.right_layout)?;
    let mut left_data = vec![D::zero(); plan.left_layout.required_len()?];
    let mut right_data = vec![D::zero(); plan.right_layout.required_len()?];

    let max_adjoint_len = plan
        .routes
        .iter()
        .map(|route| plan.source.regions[route.source_region].range().len())
        .max()
        .unwrap_or(0);
    let max_q_prime_len = plan
        .routes
        .iter()
        .filter_map(|route| {
            route
                .right_region
                .map(|index| plan.right_regions[index].range().len())
        })
        .max()
        .unwrap_or(0);
    let max_r_prime_len = plan
        .routes
        .iter()
        .filter_map(|route| {
            route
                .left_region
                .map(|index| plan.left_regions[index].range().len())
        })
        .max()
        .unwrap_or(0);
    let mut adjoint_scratch = vec![D::zero(); max_adjoint_len];
    let mut q_prime_scratch = vec![D::zero(); max_q_prime_len];
    let mut r_prime_scratch = vec![D::zero(); max_r_prime_len];
    #[cfg(test)]
    record_compact_lq_scratch::<D>(max_adjoint_len + max_q_prime_len + max_r_prime_len);

    for route in plan.routes.iter().copied() {
        if route.rank == 0 {
            continue;
        }
        let source = &plan.source.regions[route.source_region];
        let left = &plan.left_regions[route.left_region.expect("nonzero route has left region")];
        let right =
            &plan.right_regions[route.right_region.expect("nonzero route has right region")];
        let source_data = &input.data()[source.range()];
        let adjoint = &mut adjoint_scratch[..source_data.len()];
        adjoint_col_major_into(source_data, source.rows(), source.cols(), adjoint);
        #[cfg(test)]
        record_compact_lq_adjoint_fill::<D>(source_data.len());

        let q_prime_len = source.cols() * route.rank;
        let r_prime_len = route.rank * source.rows();
        let q_prime = &mut q_prime_scratch[..q_prime_len];
        let r_prime = &mut r_prime_scratch[..r_prime_len];
        qr_into_workspace(
            dense,
            adjoint,
            source.cols(),
            source.rows(),
            source.cols(),
            q_prime,
            source.cols(),
            route.rank,
            source.cols(),
            r_prime,
            route.rank,
            source.rows(),
            route.rank,
        )?;
        positive_diagonal_gauge_strided(
            q_prime,
            source.cols(),
            source.cols(),
            r_prime,
            route.rank,
            route.rank,
            source.rows(),
        );
        adjoint_col_major_into(
            r_prime,
            route.rank,
            source.rows(),
            &mut left_data[left.range()],
        );
        #[cfg(test)]
        record_compact_lq_final_adjoint_copy::<D>(r_prime.len());
        adjoint_col_major_into(
            q_prime,
            source.cols(),
            route.rank,
            &mut right_data[right.range()],
        );
        #[cfg(test)]
        record_compact_lq_final_adjoint_copy::<D>(q_prime.len());
    }

    let left = BoundDynFactor::from_bound(left_space, left_data, space.nout(), 1)?;
    let right = BoundDynFactor::from_bound(right_space, right_data, 1, space.nin())?;
    Ok((left, right))
}

/// Left isometry factorization `t = V * C` (TensorKit 0.17 / MatrixAlgebraKit
/// `left_orth`): `V : codomain <- W` isometric, `C : W <- domain`.
///
/// TensorKit's default `kind = :qr` maps to [`qr_compact`], which applies the
/// positive-diagonal QR gauge (`positive = true`, the MAK default).
pub fn left_orth<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<(BoundTensorMap<R, D, NOUT, 1>, BoundTensorMap<R, D, 1, NIN>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    qr_compact(dense, input)
}

/// Right isometry factorization `t = C * Vh` (TensorKit 0.17 /
/// MatrixAlgebraKit `right_orth`): `C : codomain <- W`, `Vh : W <- domain`
/// with orthonormal rows.
///
/// TensorKit's default `kind = :lq` maps to [`lq_compact`], which applies the
/// positive-diagonal LQ gauge (`positive = true`, the MAK default).
pub fn right_orth<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<(BoundTensorMap<R, D, NOUT, 1>, BoundTensorMap<R, D, 1, NIN>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    lq_compact(dense, input)
}

/// Transposes a column-major `rows x cols` matrix into column-major
/// `cols x rows`.
/// Adjoint (conjugate transpose) of a column-major `rows x cols` matrix.
fn adjoint_col_major<D: FactorScalar>(data: &[D], rows: usize, cols: usize) -> Vec<D> {
    let mut adjoint = vec![D::zero(); data.len()];
    adjoint_col_major_into(data, rows, cols, &mut adjoint);
    adjoint
}

fn adjoint_col_major_into<D: FactorScalar>(
    data: &[D],
    rows: usize,
    cols: usize,
    adjoint: &mut [D],
) {
    debug_assert_eq!(data.len(), rows * cols);
    debug_assert_eq!(adjoint.len(), data.len());
    for col in 0..cols {
        for row in 0..rows {
            adjoint[col + cols * row] = FactorScalar::adjoint(data[row + rows * col]);
        }
    }
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
    E: DenseExecutor + ?Sized,
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

fn checked_sector_regions(
    structure: &BlockStructure,
    nout: usize,
) -> Result<Option<Arc<[CoupledSectorRegion]>>, OperationError> {
    structure
        .coupled_sector_regions(nout)
        .map_err(OperationError::from_core_preserving_context)
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

fn coupled_of_generic<R>(rule: &R, tree: &FusionTreeKey) -> SectorId
where
    R: FusionRule,
{
    tree.coupled().unwrap_or_else(|| rule.vacuum())
}

/// Generic sibling of [`sector_matricizations`]: identical two-pass stacking
/// (vertex-labelled trees are distinct keys, so OM trees get distinct rows /
/// columns of the coupled block, exactly TensorKit's `block(t, c)` layout).
fn sector_matricizations_generic<R, D>(
    rule: &R,
    structure: &BlockStructure,
    data: &[D],
    nout: usize,
) -> Result<Vec<SectorMatricization<D>>, OperationError>
where
    R: FusionRule,
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
        let sector = coupled_of_generic(rule, key.codomain_tree());
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
        let sector = coupled_of_generic(rule, key.codomain_tree());
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

/// Builds provider-bound left and right factor spaces for a generic rule.
fn build_left_right_bound_spaces_generic<R, D>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[SectorMatricization<D>],
    ranks: &[SectorRank],
) -> Result<(BoundDynamicFusionMapSpace<R>, BoundDynamicFusionMapSpace<R>), OperationError>
where
    R: FusionRule,
    D: FactorScalar,
{
    let rule = provider.as_ref();
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
    let left_keys = left_hom
        .fusion_tree_keys_generic(rule)
        .map_err(OperationError::from_core_preserving_context)?;
    let left_shapes = left_keys
        .iter()
        .map(|key| {
            let sector = coupled_of_generic(rule, key.codomain_tree());
            let mut shape = row_shape_of(&matrix_by_sector, sector, key.codomain_tree())?;
            shape.push(sector_rank(sector));
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    let left = BoundDynamicFusionMapSpace::from_degeneracy_shapes_generic(
        Arc::clone(provider),
        left_hom,
        left_shapes,
    )?;
    let right_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg]),
        homspace.domain().clone(),
    );
    let right_keys = right_hom
        .fusion_tree_keys_generic(rule)
        .map_err(OperationError::from_core_preserving_context)?;
    let right_shapes = right_keys
        .iter()
        .map(|key| {
            let sector = coupled_of_generic(rule, key.domain_tree());
            let mut shape = vec![sector_rank(sector)];
            shape.extend(col_shape_of(&matrix_by_sector, sector, key.domain_tree())?);
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    let right = BoundDynamicFusionMapSpace::from_degeneracy_shapes_generic(
        Arc::clone(provider),
        right_hom,
        right_shapes,
    )?;
    Ok((left, right))
}

fn build_left_right_bound_pair_generic<R, D>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[SectorMatricization<D>],
    pairs: &[FactorPair<D>],
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    R: FusionRule,
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
        build_left_right_bound_spaces_generic(provider, homspace, matricizations, &ranks)?;
    let mut left_data = vec![D::zero(); left_space.space().required_len()?];
    let mut right_data = vec![D::zero(); right_space.space().required_len()?];
    for (matrix, pair) in matricizations.iter().zip(pairs) {
        scatter_left_sector_blocks_generic(
            provider.as_ref(),
            left_space.space(),
            &mut left_data,
            matrix,
            &pair.left,
            pair.left_rows,
        )?;
        scatter_right_sector_blocks_generic(
            provider.as_ref(),
            right_space.space(),
            &mut right_data,
            matrix,
            &pair.right,
            pair.right_leading,
        )?;
    }
    let left_nout = left_space.space().nout();
    let right_nin = right_space.space().nin();
    Ok((
        BoundDynFactor::from_bound(left_space, left_data, left_nout, 1)?,
        BoundDynFactor::from_bound(right_space, right_data, 1, right_nin)?,
    ))
}

/// Generic sibling of [`scatter_left_sector_blocks`].
fn scatter_left_sector_blocks_generic<R, D>(
    rule: &R,
    left_space: &DynamicFusionMapSpace,
    left_data: &mut [D],
    matrix: &SectorMatricization<D>,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    R: FusionRule,
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
        if coupled_of_generic(rule, key.codomain_tree()) != matrix.sector {
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

/// Generic sibling of [`scatter_right_sector_blocks`].
fn scatter_right_sector_blocks_generic<R, D>(
    rule: &R,
    right_space: &DynamicFusionMapSpace,
    right_data: &mut [D],
    matrix: &SectorMatricization<D>,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    R: FusionRule,
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
        if coupled_of_generic(rule, key.domain_tree()) != matrix.sector {
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

/// Generic sibling of [`svd_compact_factors_dyn`] (SU(N)): identical dense
/// per-sector SVD + gauge + scatter; only the space builders differ.
pub fn svd_compact_factors_dyn_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let rule = input.space().provider();
    let space = input.space().space();
    let matricizations =
        sector_matricizations_generic(rule, space.structure(), input.data(), space.nout())?;

    let ranks = matricizations
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector,
            kept: matrix.rows.min(matrix.cols),
        })
        .collect::<Vec<_>>();
    let provider = input.space().provider_arc();
    let (u_space, vt_space) =
        build_left_right_bound_spaces_generic(provider, space.homspace(), &matricizations, &ranks)?;
    let u_len = u_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut u_data = vec![D::zero(); u_len];
    let vt_len = vt_space
        .space()
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
        scatter_left_sector_blocks_generic(
            rule,
            u_space.space(),
            &mut u_data,
            matrix,
            &u_workspace,
            max_rows,
        )?;
        scatter_right_sector_blocks_generic(
            rule,
            vt_space.space(),
            &mut vt_data,
            matrix,
            &vt_workspace,
            max_rank,
        )?;
    }

    let u = BoundDynFactor::from_bound(u_space, u_data, space.nout(), 1)?;
    let vh = BoundDynFactor::from_bound(vt_space, vt_data, 1, space.nin())?;
    Ok((u, vh, singular_values))
}

/// Builds a provider-bound diagonal factor for a generic rule.
fn diagonal_bond_svd_factor_generic<R, D, V>(
    provider: Arc<R>,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: FusionRule,
    D: FactorScalar,
    V: Copy,
{
    let space = diagonal_bond_bound_space_generic(provider, spectrum)?;
    let data = diagonal_bond_data(space.space(), spectrum, to_scalar)?;
    BoundDynFactor::from_bound(space, data, 1, 1)
}

pub fn diagonal_bond_bound_space_generic<R, V>(
    provider: Arc<R>,
    spectrum: &[SectorSpectrum<V>],
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: FusionRule,
{
    let rule = provider.as_ref();
    let new_leg = SectorLeg::new(
        spectrum
            .iter()
            .map(|entry| (entry.sector, entry.values.len())),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg.clone()]),
        FusionProductSpace::new([new_leg]),
    );
    let length_by_sector: HashMap<SectorId, usize> = spectrum
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();
    let keys = homspace
        .fusion_tree_keys_generic(rule)
        .map_err(OperationError::from_core_preserving_context)?;
    let shapes = keys
        .iter()
        .map(|key| {
            let count = length_by_sector
                .get(&coupled_of_generic(rule, key.codomain_tree()))
                .copied()
                .unwrap_or(0);
            vec![count, count]
        })
        .collect::<Vec<_>>();
    BoundDynamicFusionMapSpace::from_degeneracy_shapes_generic(provider, homspace, shapes)
}

/// Generic sibling of [`svd_compact_dyn`].
pub(crate) fn svd_compact_dyn_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdCompactDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let (u, vh, singular_values) = svd_compact_factors_dyn_generic(dense, input)?;
    let s = diagonal_bond_svd_factor_generic(
        Arc::clone(input.space().provider_arc()),
        &singular_values,
        &D::from_real,
    )?;
    Ok(SvdCompactDyn {
        u,
        s,
        vh,
        singular_values,
    })
}

/// Generic sibling of [`svd_vals_dyn`].
pub fn svd_vals_dyn_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let rule = input.space().provider();
    let space = input.space().space();
    let matricizations =
        sector_matricizations_generic(rule, space.structure(), input.data(), space.nout())?;
    let mut singular_values = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let rank = matrix.rows.min(matrix.cols);
        let input_shape = [matrix.rows, matrix.cols];
        let input_strides = [1usize, matrix.rows];
        let input = DenseView::new(&matrix.data, &input_shape, &input_strides, 0)
            .map_err(OperationError::Dense)?;
        let s_tensor = dense
            .svd_vals(D::dense_read(input))
            .map_err(OperationError::Dense)?;
        let mut s = D::real_spectrum(&s_tensor).map_err(OperationError::Dense)?;
        s.truncate(rank);
        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s,
        });
    }
    Ok(singular_values)
}

/// Generic sibling of [`decide_bond_truncation`]: the weight is
/// `sqrt_dim(c)²`. Why not round it: generic rigid categories may have
/// non-integer quantum dimensions, so rounding changes the truncation policy.
fn generic_truncation_weight(sqrt_dim: f64) -> f64 {
    sqrt_dim * sqrt_dim
}

fn decide_bond_truncation_generic<R, V>(
    rule: &R,
    spectra: &[SectorSpectrum<V>],
    truncation: &Truncation,
    values_are_nonnegative: bool,
) -> crate::truncation::TruncationDecision
where
    R: GenericRigidSymbols<Scalar = f64>,
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
            weight: generic_truncation_weight(rule.sqrt_dim_scalar(entry.sector)),
            values: values.as_slice(),
        })
        .collect();
    select_truncation(&weighted, truncation)
}

#[cfg(test)]
mod generic_truncation_weight_tests {
    use super::generic_truncation_weight;

    #[test]
    fn preserves_non_integer_quantum_dimension() {
        // What: an anyonic sqrt(qdim) must remain an irrational qdim weight.
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let weight = generic_truncation_weight(golden_ratio.sqrt());
        assert!((weight - golden_ratio).abs() < 1.0e-14);
        assert_ne!(weight, weight.round());
    }
}

/// Generic sibling of [`sliced_bond_tensor`].
fn sliced_bond_tensor_generic<R, D>(
    provider: Arc<R>,
    source_space: &DynamicFusionMapSpace,
    source_data: &[D],
    axis: usize,
    kept_of: &dyn Fn(SectorId) -> usize,
    expected_nout: usize,
    expected_nin: usize,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: FusionRule,
    D: FactorScalar,
{
    let rule = provider.as_ref();
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

    let keys = new_hom
        .fusion_tree_keys_generic(rule)
        .map_err(OperationError::from_core_preserving_context)?;
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
            shape[axis] = kept_of(coupled_of_generic(rule, bond_tree));
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;

    let space =
        BoundDynamicFusionMapSpace::from_degeneracy_shapes_generic(provider, new_hom, shapes)?;
    let len = space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); len];

    let sliced_structure = Arc::clone(space.space().structure());
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
    BoundDynFactor::from_bound(space, data, expected_nout, expected_nin)
}

/// Generic sibling of [`truncate_svd_dyn`].
pub(crate) fn truncate_svd_dyn_generic<R, D>(
    compact: SvdCompactDyn<R, D>,
    truncation: &Truncation,
) -> Result<SvdTruncDyn<R, D>, OperationError>
where
    R: GenericRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = compact.u.space().provider();
    let decision = decide_bond_truncation_generic(rule, &compact.singular_values, truncation, true);
    if compact
        .singular_values
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok(SvdTruncDyn {
            u: compact.u,
            s: compact.s,
            vh: compact.vh,
            singular_values: compact.singular_values,
            error: 0.0,
        });
    }

    let mut singular_values = compact.singular_values;
    for (entry, &count) in singular_values.iter_mut().zip(&decision.kept) {
        entry.values.truncate(count);
    }
    singular_values.retain(|entry| !entry.values.is_empty());
    let kept_by_sector: HashMap<SectorId, usize> = singular_values
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();

    let kept_of = |sector: SectorId| -> usize { kept_by_sector.get(&sector).copied().unwrap_or(0) };

    let bond_axis = compact.u.space().space().nout();
    let provider = Arc::clone(compact.u.space().provider_arc());
    let u_factor = sliced_bond_tensor_generic(
        Arc::clone(&provider),
        compact.u.space().space(),
        compact.u.data(),
        bond_axis,
        &kept_of,
        compact.u.space().space().nout(),
        1,
    )?;
    let vh_factor = sliced_bond_tensor_generic(
        Arc::clone(&provider),
        compact.vh.space().space(),
        compact.vh.data(),
        0,
        &kept_of,
        1,
        compact.vh.space().space().nin(),
    )?;
    let s_factor = diagonal_bond_svd_factor_generic(provider, &singular_values, &D::from_real)?;
    Ok(SvdTruncDyn {
        u: u_factor,
        s: s_factor,
        vh: vh_factor,
        singular_values,
        error: decision.error,
    })
}

/// Generic sibling of [`svd_trunc_dyn`].
pub fn svd_trunc_dyn_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    truncation: &Truncation,
) -> Result<SvdTruncDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: GenericRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let compact = svd_compact_dyn_generic(dense, input)?;
    truncate_svd_dyn_generic(compact, truncation)
}

/// Provider-bound compact QR for a generic rule.
pub fn qr_compact_dyn_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = sector_matricizations_generic(
        provider.as_ref(),
        space.structure(),
        input.data(),
        space.nout(),
    )?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let rank = matrix.rows.min(matrix.cols);
        let mut q = vec![D::zero(); matrix.rows * rank];
        let mut r = vec![D::zero(); rank * matrix.cols];
        qr_into_workspace(
            dense,
            &matrix.data,
            matrix.rows,
            matrix.cols,
            matrix.rows,
            &mut q,
            matrix.rows,
            rank,
            matrix.rows,
            &mut r,
            rank,
            matrix.cols,
            rank,
        )?;
        positive_diagonal_gauge_strided(
            &mut q,
            matrix.rows,
            matrix.rows,
            &mut r,
            rank,
            rank,
            matrix.cols,
        );
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rank,
            left: q,
            left_rows: matrix.rows,
            right: r,
            right_leading: rank,
        });
    }
    build_left_right_bound_pair_generic(provider, space.homspace(), &matrices, &pairs)
}

/// Provider-bound compact LQ for a generic rule.
pub fn lq_compact_dyn_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = sector_matricizations_generic(
        provider.as_ref(),
        space.structure(),
        input.data(),
        space.nout(),
    )?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let rank = matrix.rows.min(matrix.cols);
        let adjoint = adjoint_col_major(&matrix.data, matrix.rows, matrix.cols);
        let mut q_prime = vec![D::zero(); matrix.cols * rank];
        let mut r_prime = vec![D::zero(); rank * matrix.rows];
        qr_into_workspace(
            dense,
            &adjoint,
            matrix.cols,
            matrix.rows,
            matrix.cols,
            &mut q_prime,
            matrix.cols,
            rank,
            matrix.cols,
            &mut r_prime,
            rank,
            matrix.rows,
            rank,
        )?;
        positive_diagonal_gauge_strided(
            &mut q_prime,
            matrix.cols,
            matrix.cols,
            &mut r_prime,
            rank,
            rank,
            matrix.rows,
        );
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rank,
            left: adjoint_col_major(&r_prime, rank, matrix.rows),
            left_rows: matrix.rows,
            right: adjoint_col_major(&q_prime, matrix.cols, rank),
            right_leading: rank,
        });
    }
    build_left_right_bound_pair_generic(provider, space.homspace(), &matrices, &pairs)
}
