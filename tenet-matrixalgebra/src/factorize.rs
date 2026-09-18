use std::collections::BTreeMap;

use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;
use std::sync::Arc;

#[cfg(test)]
use std::cell::{Cell, RefCell};

use num_complex::Complex64;
use num_traits::{Float, Zero};
use tenet_core::{
    BlockKey, BlockStructure, CheckedGenericFusion, CheckedGenericRigidSymbols,
    CheckedGenericStructureError, CoreError, CoupledSectorRegion, CoupledTreeExtent,
    FusionProductSpace, FusionRule, FusionTensorMapSpace, FusionTreeHomSpace, FusionTreeKey,
    FusionTreePairKey, GenericRigidSymbols, InfallibleGeneric, MultiplicityFreeRigidSymbols,
    SectorId, SectorLeg, SectorStructure, TensorMap, TensorMapSpace,
};
use tenet_dense::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseOwned, DenseTensor, DenseView,
    DenseViewMut,
};

pub use tenet_tensors::BoundDynamicTensorRef;
use tenet_tensors::{
    BoundDynamicFusionMapSpace, DenseBlockScalar, DenseRecouplingScalar, DynamicFusionMapSpace,
    PreparedCheckedGenericDynamicSpace, ValidatedDynamicFusionLayout,
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
    /// Consuming implementations may transfer backend-owned host storage.
    /// Custom scalar implementations retain the former borrowed-copy behavior.
    fn dense_into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError> {
        Self::dense_slice(&tensor).map(ToOwned::to_owned)
    }
    /// Converts a sector payload into the dense executor's owned full-SVD
    /// input. Custom scalar implementations retain the payload for the
    /// compatibility route.
    fn dense_into_owned(input: Vec<Self>) -> Result<DenseOwned, Vec<Self>> {
        Err(input)
    }
    /// Real spectrum output (singular values, Hermitian eigenvalues) widened
    /// to `f64` for the host-side truncation policies.
    fn real_spectrum(tensor: &DenseTensor) -> Result<Vec<f64>, DenseError>;
    fn from_real(value: f64) -> Self;
    /// Widens to `Complex64` (general eigenvalue bookkeeping).
    fn widen_complex(self) -> Complex64;
    /// Narrows from `Complex64` (lossy for the single-precision scalars).
    fn from_complex64(value: Complex64) -> Self;
    fn adjoint(self) -> Self;
    /// LAPACK's `xLAMCH('P')` — `eps * base`, the spacing of one — in the
    /// *real component* type of `Self`.
    fn epsilon() -> f64;
    /// LAPACK's `xLAMCH('S')` — the safe minimum, the smallest positive normal
    /// number — in the *real component* type of `Self`.
    ///
    /// Paired with [`FactorScalar::epsilon`] this reproduces the machine bounds
    /// the reference `gebal` derives (`sgebal.f:330` for the single-precision
    /// blocks, `dgebal.f:330` for the double), which have to come from the
    /// component type: the double bounds let the radix loop of a single
    /// precision block reach a factor around `2^970`, and `from_real` of that
    /// is `inf` in `f32`.
    fn safe_minimum() -> f64;
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

#[cfg(test)]
mod numerical_null_tests {
    use super::*;
    use tenet_dense::{DefaultDenseExecutor, DenseRead, DenseWrite};

    #[derive(Default)]
    struct OwnedSvdSpy {
        inner: DefaultDenseExecutor,
        u_pointer: Option<usize>,
        svd_into_calls: usize,
    }

    impl DenseExecutor for OwnedSvdSpy {
        fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            let outputs = self.inner.svd(input)?;
            self.u_pointer = Some(
                outputs[0]
                    .as_f64_slice()
                    .expect("f64 null fixture must return f64 U")
                    .as_ptr() as usize,
            );
            Ok(outputs)
        }

        fn svd_into(
            &mut self,
            _: DenseRead<'_>,
            _: DenseWrite<'_>,
            _: DenseWrite<'_>,
            _: DenseWrite<'_>,
        ) -> Result<(), DenseError> {
            self.svd_into_calls += 1;
            Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "svd_into",
                message: "numerical null must consume owned SVD factors".to_string(),
            })
        }

        fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.qr(input)
        }

        fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.eigh(input)
        }

        fn dot_general_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            self.inner.dot_general_into(output, lhs, rhs, config)
        }
    }

    #[test]
    fn numerical_null_left_basis_keeps_owned_svd_u() {
        let mut dense = OwnedSvdSpy::default();
        let (_, u, _) = numerical_rank_and_compact_bases(
            &mut dense,
            &[1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0],
            3,
            2,
        )
        .unwrap();

        assert_eq!(dense.svd_into_calls, 0);
        assert_eq!(u.as_ptr() as usize, dense.u_pointer.unwrap());
    }
}

trait HermitianReal: Float {
    fn from_f64(value: f64) -> Self;
    fn relative_tolerance() -> Self;
}

impl HermitianReal for f32 {
    fn from_f64(value: f64) -> Self {
        value as Self
    }

    fn relative_tolerance() -> Self {
        64.0 * Self::EPSILON
    }
}

impl HermitianReal for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }

    fn relative_tolerance() -> Self {
        64.0 * Self::EPSILON
    }
}

impl FactorScalar for f32 {
    type Eig = num_complex::Complex32;
    type Real = f32;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
        tensor.as_f32_slice()
    }

    fn dense_into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError> {
        tensor.into_f32_vec()
    }

    fn dense_into_owned(input: Vec<Self>) -> Result<DenseOwned, Vec<Self>> {
        Ok(DenseOwned::F32(input))
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

    fn safe_minimum() -> f64 {
        f32::MIN_POSITIVE as f64
    }
}

impl FactorScalar for f64 {
    type Eig = Complex64;
    type Real = f64;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
        tensor.as_f64_slice()
    }

    fn dense_into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError> {
        tensor.into_f64_vec()
    }

    fn dense_into_owned(input: Vec<Self>) -> Result<DenseOwned, Vec<Self>> {
        Ok(DenseOwned::F64(input))
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

    fn safe_minimum() -> f64 {
        f64::MIN_POSITIVE
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

    fn dense_into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError> {
        tensor.into_c32_vec()
    }

    fn dense_into_owned(input: Vec<Self>) -> Result<DenseOwned, Vec<Self>> {
        Ok(DenseOwned::C32(input))
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

    fn safe_minimum() -> f64 {
        f32::MIN_POSITIVE as f64
    }
}

impl FactorScalar for Complex64 {
    type Eig = Complex64;
    type Real = f64;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
        tensor.as_c64_slice()
    }

    fn dense_into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError> {
        tensor.into_c64_vec()
    }

    fn dense_into_owned(input: Vec<Self>) -> Result<DenseOwned, Vec<Self>> {
        Ok(DenseOwned::C64(input))
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

    fn safe_minimum() -> f64 {
        f64::MIN_POSITIVE
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
        // Why not expose an unchecked constructor: `BoundTensorMap::try_new`
        // establishes this invariant once, while external dynamic inputs must
        // continue through `BoundDynamicTensorRef::try_new`.
        BoundDynamicTensorRef::try_new(self.space, self.tensor.data())
            .expect("BoundTensorMap preserves its validated dynamic layout")
    }
}

/// Owned dynamic factor that retains the provider used to create its complete
/// fusion space.
pub struct BoundDynFactor<R, D> {
    space: BoundDynamicFusionMapSpace<R>,
    data: Vec<D>,
}

type DynamicFactorPair<R, D> = (BoundDynFactor<R, D>, BoundDynFactor<R, D>);

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

impl<R, D> BoundDynFactor<R, D> {
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

    #[expect(
        clippy::type_complexity,
        reason = "the public decomposition accessor returns its named components in documented order"
    )]
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

/// Truncated SVD factors without a materialized diagonal `S`:
/// `(U, Vh, spectrum, error)`.
#[doc(hidden)]
pub type SvdTruncFactorsDyn<R, D> = (
    BoundDynFactor<R, D>,
    BoundDynFactor<R, D>,
    Vec<SectorSpectrum>,
    f64,
);

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

    #[expect(
        clippy::type_complexity,
        reason = "the public decomposition accessor returns its named components in documented order"
    )]
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
    record_diagonal_bond_build(spectrum);
    let space = diagonal_bond_bound_space_like(authority, spectrum)?;
    let data = diagonal_bond_data(space.space(), spectrum, to_scalar)?;
    BoundDynFactor::from_bound(space, data, 1, 1)
}

#[cfg(test)]
fn record_diagonal_bond_build<V>(spectrum: &[SectorSpectrum<V>]) {
    DIAGONAL_BOND_BUILD_PROBE.with(|probe| {
        let mut current = probe.get();
        current.calls += 1;
        current.values += spectrum
            .iter()
            .map(|entry| entry.values.len())
            .sum::<usize>();
        probe.set(current);
    });
}

#[doc(hidden)]
pub fn diagonal_bond_bound_space_like<R, V>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<V>],
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
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
    authority.derive_from_final_homspace(homspace)
}

pub fn diagonal_bond_bound_space<R, V>(
    provider: Arc<R>,
    spectrum: &[SectorSpectrum<V>],
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
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
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(provider, homspace)
}

/// Fills the dense block-diagonal data of `space` from `spectrum`, mapping
/// each value through `to_scalar`. Only the
/// per-block diagonal is written; the rest stays zero. Bit-for-bit identical to
/// the fill inside the former monolithic `diagonal_bond_tensor_dyn`.
#[doc(hidden)]
pub fn diagonal_bond_data<D, V>(
    space: &DynamicFusionMapSpace,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<Vec<D>, OperationError>
where
    D: FactorScalar,
    V: Copy,
{
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum<V>> =
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
        let sector = tree.codomain_tree().coupled();
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
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum<V>> =
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

#[derive(Clone, Copy)]
struct TreeExtentRef<'a> {
    tree: &'a FusionTreeKey,
    offset: usize,
    shape: &'a [usize],
}

// Keeps publication generic over owned packs and cached regions without
// allocating a second list of borrowed tree descriptors.
trait SectorGeometry {
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

struct SectorMatrixRef<'a, D> {
    sector: SectorId,
    rows: usize,
    cols: usize,
    data: &'a [D],
}

enum InputMatricizations<'a, D> {
    Regions {
        data: &'a [D],
        regions: Arc<[CoupledSectorRegion]>,
    },
    Packed(Vec<SectorMatricization<D>>),
}

impl<'a, D: FactorScalar> InputMatricizations<'a, D> {
    fn len(&self) -> usize {
        match self {
            Self::Regions { regions, .. } => regions.len(),
            Self::Packed(matrices) => matrices.len(),
        }
    }

    fn get(&self, index: usize) -> Result<SectorMatrixRef<'_, D>, OperationError> {
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
    fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }

    fn validate_hermitian(&self) -> Result<(), OperationError> {
        match self {
            Self::Regions { data, regions } => validate_hermitian_regions(data, regions),
            Self::Packed(matrices) => validate_hermitian_matricizations(matrices),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompactSvdCopyProbe {
    pub input_pack_calls: usize,
    pub input_pack_bytes: usize,
    pub output_scatter_calls: usize,
    pub output_scatter_bytes: usize,
    pub owned_output_publications: usize,
    pub owned_output_owner_reused: usize,
}

#[cfg(test)]
thread_local! {
    static COMPACT_SVD_COPY_PROBE: Cell<CompactSvdCopyProbe> = Cell::default();
    static COMPACT_QR_COPY_PROBE: Cell<CompactQrCopyProbe> = Cell::default();
    static EIGH_COPY_PROBE: Cell<EighCopyProbe> = Cell::default();
    static EIGH_OWNED_VECTOR_POINTERS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static CHECKED_EIGH_PAIR_POINTERS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static CHECKED_COMPACT_SVD_STAGE_POINTERS: RefCell<Vec<(usize, usize)>> =
        const { RefCell::new(Vec::new()) };
    static GENERIC_COMPACT_SVD_FALLBACK_POINTERS: RefCell<Vec<(usize, usize)>> =
        const { RefCell::new(Vec::new()) };
    static MF_COMPACT_SVD_FALLBACK_POINTERS: RefCell<Vec<(usize, usize)>> =
        const { RefCell::new(Vec::new()) };
    static COMPACT_LQ_COPY_PROBE: Cell<CompactLqCopyProbe> = Cell::default();
    static DIAGONAL_BOND_BUILD_PROBE: Cell<DiagonalBondBuildProbe> = Cell::default();
    static VALUES_MATRICIZATION_FALLBACKS: Cell<usize> = const { Cell::new(0) };
    static CHECKED_COMPACT_INPUT_OBSERVATIONS: RefCell<Vec<CheckedCompactInputObservation>> =
        const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn reset_checked_compact_svd_stage_pointers() {
    CHECKED_COMPACT_SVD_STAGE_POINTERS.with(|pointers| pointers.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn checked_compact_svd_stage_pointers() -> Vec<(usize, usize)> {
    CHECKED_COMPACT_SVD_STAGE_POINTERS.with(|pointers| pointers.borrow().clone())
}

#[cfg(test)]
fn record_checked_compact_svd_stage_gauge<D>(u: &[D], vt: &[D]) {
    CHECKED_COMPACT_SVD_STAGE_POINTERS.with(|pointers| {
        pointers
            .borrow_mut()
            .push((u.as_ptr() as usize, vt.as_ptr() as usize));
    });
}

#[cfg(test)]
pub(crate) fn reset_generic_compact_svd_fallback_pointers() {
    GENERIC_COMPACT_SVD_FALLBACK_POINTERS.with(|pointers| pointers.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn generic_compact_svd_fallback_pointers() -> Vec<(usize, usize)> {
    GENERIC_COMPACT_SVD_FALLBACK_POINTERS.with(|pointers| pointers.borrow().clone())
}

#[cfg(test)]
fn record_generic_compact_svd_fallback_gauge<D>(u: &[D], vt: &[D]) {
    GENERIC_COMPACT_SVD_FALLBACK_POINTERS.with(|pointers| {
        pointers
            .borrow_mut()
            .push((u.as_ptr() as usize, vt.as_ptr() as usize));
    });
}

#[cfg(test)]
pub(crate) fn reset_mf_compact_svd_fallback_pointers() {
    MF_COMPACT_SVD_FALLBACK_POINTERS.with(|pointers| pointers.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn mf_compact_svd_fallback_pointers() -> Vec<(usize, usize)> {
    MF_COMPACT_SVD_FALLBACK_POINTERS.with(|pointers| pointers.borrow().clone())
}

#[cfg(test)]
fn record_mf_compact_svd_fallback_gauge<D>(u: &[D], vt: &[D]) {
    MF_COMPACT_SVD_FALLBACK_POINTERS.with(|pointers| {
        pointers
            .borrow_mut()
            .push((u.as_ptr() as usize, vt.as_ptr() as usize));
    });
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
fn record_checked_compact_input<D>(
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
pub(crate) fn reset_values_matricization_fallbacks() {
    VALUES_MATRICIZATION_FALLBACKS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn values_matricization_fallbacks() -> usize {
    VALUES_MATRICIZATION_FALLBACKS.with(Cell::get)
}

#[cfg(test)]
fn record_values_matricization_fallback() {
    VALUES_MATRICIZATION_FALLBACKS.with(|count| count.set(count.get() + 1));
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
    pub owned_output_publications: usize,
    pub owned_output_owner_reused: usize,
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
    record_compact_qr_output_scatter_work::<D>(1, elements);
}

#[cfg(test)]
fn record_compact_qr_output_scatter_work<D>(calls: usize, elements: usize) {
    COMPACT_QR_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += calls;
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
pub(crate) fn reset_eigh_owned_vector_pointers() {
    EIGH_OWNED_VECTOR_POINTERS.with(|pointers| pointers.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn eigh_owned_vector_pointers() -> Vec<usize> {
    EIGH_OWNED_VECTOR_POINTERS.with(|pointers| pointers.borrow().clone())
}

#[cfg(test)]
fn record_eigh_owned_vector_before_scatter<D>(vectors: &[D]) {
    EIGH_OWNED_VECTOR_POINTERS.with(|pointers| {
        pointers.borrow_mut().push(vectors.as_ptr() as usize);
    });
}

#[cfg(test)]
pub(crate) fn reset_checked_eigh_pair_pointers() {
    CHECKED_EIGH_PAIR_POINTERS.with(|pointers| pointers.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn checked_eigh_pair_pointers() -> Vec<usize> {
    CHECKED_EIGH_PAIR_POINTERS.with(|pointers| pointers.borrow().clone())
}

#[cfg(test)]
fn record_checked_eigh_pair_pointers<D>(pairs: &[FactorPair<D>]) {
    CHECKED_EIGH_PAIR_POINTERS.with(|pointers| {
        *pointers.borrow_mut() = pairs
            .iter()
            .map(|pair| pair.left.as_ptr() as usize)
            .collect();
    });
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
    record_compact_lq_output_scatter_work::<D>(1, elements);
}

#[cfg(test)]
fn record_compact_lq_output_scatter_work<D>(calls: usize, elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += calls;
        current.output_scatter_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
fn record_compact_lq_scratch<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.scratch_buffer_count += 1;
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
    let space = input.space().space();
    // Values-only: per coupled sector call the no-vector SVD (`svd_vals`,
    // LAPACK `job='N'`) and keep the spectrum. Unlike `svd_compact_dyn` this
    // never builds the U/Vt spaces, allocates the factor buffers, gauge-fixes,
    // or scatters blocks into the fusion-tree layout — all of which the old
    // `svd_compact_dyn(..).map(|svd| svd.singular_values)` computed then threw
    // away. Valid no-vector and full-factor drivers can differ in the last
    // bits, so comparisons use dtype-appropriate tolerances.
    let matricizations = value_matricizations(space.structure(), input.data(), space.nout())?;
    let mut singular_values = Vec::with_capacity(matricizations.len());
    for index in 0..matricizations.len() {
        let matrix = matricizations.get(index)?;
        let rank = matrix.rows.min(matrix.cols);
        let input_shape = [matrix.rows, matrix.cols];
        let input_strides = [1usize, matrix.rows];
        let input = DenseView::new(matrix.data, &input_shape, &input_strides, 0)
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
    let (u, vh, singular_values, error) = svd_trunc_factors_dyn(dense, input, truncation)?;
    let s = diagonal_bond_svd_factor(u.space(), &singular_values, &D::from_real)?;
    Ok(SvdTruncDyn {
        u,
        s,
        vh,
        singular_values,
        error,
    })
}

/// Dynamic-rank truncated SVD without materializing the diagonal `S` factor.
#[doc(hidden)]
pub fn svd_trunc_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    truncation: &Truncation,
) -> Result<SvdTruncFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (u, vh, singular_values) = svd_compact_factors_dyn(dense, input)?;
    truncate_svd_factors_only_dyn(u, vh, singular_values, truncation)
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
    svd_compact_factors_dyn_with_direction(dense, input, None, CompactSvdGauge::Left)
}

/// Compact SVD factors for the logical adjoint without constructing its input:
/// if `A = U S Vh`, returns `(V, Uh, spectrum)` with the phase gauge applied
/// to the final left factor `V`.
#[doc(hidden)]
pub fn svd_compact_adjoint_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (u, vh, spectrum) =
        svd_compact_factors_dyn_with_direction(dense, input, None, CompactSvdGauge::AdjointLeft)?;
    Ok((
        adjoint_bound_factor(&vh)?,
        adjoint_bound_factor(&u)?,
        spectrum,
    ))
}

/// Truncated SVD factors for the logical adjoint without constructing its input.
#[doc(hidden)]
pub fn svd_trunc_adjoint_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    truncation: &Truncation,
) -> Result<SvdTruncFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (u, vh, spectrum) =
        svd_compact_factors_dyn_with_direction(dense, input, None, CompactSvdGauge::AdjointLeft)?;
    let (u, vh, spectrum, error) = truncate_svd_factors_only_dyn(u, vh, spectrum, truncation)?;
    Ok((
        adjoint_bound_factor(&vh)?,
        adjoint_bound_factor(&u)?,
        spectrum,
        error,
    ))
}

#[derive(Clone, Copy)]
enum PolarDirection {
    Left,
    Right,
}

struct CompactSvdNumericalStage<D> {
    rows: usize,
    cols: usize,
    rank: usize,
    u: Vec<D>,
    singular_values: Vec<f64>,
    vt: Vec<D>,
}

fn compact_svd_numerical_stage<E, D>(
    dense: &mut E,
    matrix: &[D],
    rows: usize,
    cols: usize,
) -> Result<CompactSvdNumericalStage<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let rank = rows.min(cols);
    if rank == 0 {
        return Ok(CompactSvdNumericalStage {
            rows,
            cols,
            rank,
            u: Vec::new(),
            singular_values: Vec::new(),
            vt: Vec::new(),
        });
    }
    let (mut u, singular_values, mut vt) = compact_svd_owned(dense, matrix, rows, cols)?;
    svd_compact_gauge(&mut u, rows, rows, &mut vt, rank, cols, rank);
    #[cfg(test)]
    record_checked_compact_svd_stage_gauge(&u, &vt);
    Ok(CompactSvdNumericalStage {
        rows,
        cols,
        rank,
        u,
        singular_values,
        vt,
    })
}

#[cfg(test)]
pub(crate) fn compact_svd_numerical_stage_lengths_for_test<E, D>(
    dense: &mut E,
    matrix: &[D],
    rows: usize,
    cols: usize,
) -> Result<(usize, usize, usize), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let stage = compact_svd_numerical_stage(dense, matrix, rows, cols)?;
    Ok((stage.u.len(), stage.singular_values.len(), stage.vt.len()))
}

#[derive(Clone, Copy)]
enum CompactSvdGauge {
    Left,
    AdjointLeft,
}

impl PolarDirection {
    fn accepts(self, rows: usize, cols: usize) -> bool {
        match self {
            Self::Left => rows >= cols,
            Self::Right => cols >= rows,
        }
    }

    fn error(self) -> OperationError {
        OperationError::InvalidArgument {
            message: match self {
                Self::Left => "left_polar requires rows >= columns in every coupled-sector matrix",
                Self::Right => {
                    "right_polar requires columns >= rows in every coupled-sector matrix"
                }
            },
        }
    }
}

fn validate_polar_direction(
    acceptance_direction: PolarDirection,
    error_direction: PolarDirection,
    space: &BoundDynamicFusionMapSpace<impl FusionRule>,
) -> Result<(), OperationError> {
    let row_dimensions = space
        .space()
        .homspace()
        .codomain()
        .coupled_sector_block_dimensions(space.provider())?;
    let col_dimensions = space
        .space()
        .homspace()
        .domain()
        .coupled_sector_block_dimensions(space.provider())?;
    for (&sector, &rows) in &row_dimensions {
        let cols = col_dimensions.get(&sector).copied().unwrap_or(0);
        if !acceptance_direction.accepts(rows, cols) {
            return Err(error_direction.error());
        }
    }
    for (&sector, &cols) in &col_dimensions {
        if !row_dimensions.contains_key(&sector) && !acceptance_direction.accepts(0, cols) {
            return Err(error_direction.error());
        }
    }
    Ok(())
}

fn svd_compact_factors_dyn_with_direction<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    polar_direction: Option<(PolarDirection, PolarDirection)>,
    gauge: CompactSvdGauge,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    if let Some((acceptance_direction, error_direction)) = polar_direction {
        // Stored routes omit side-only sectors, whose logical matrices are
        // rows x 0 or 0 x columns and still constrain the isometry direction.
        validate_polar_direction(acceptance_direction, error_direction, input.space())?;
    }
    if let Some(plan) = compact_factor_plan(input.space())? {
        return svd_compact_direct_regions(dense, input, &plan, gauge);
    }
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_svd_input_pack(&matricizations);

    let ranks = matricizations
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector,
            kept: matrix.rows.min(matrix.cols),
        })
        .collect::<Vec<_>>();
    let bond = SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false);
    let u_space = build_bound_factor_space(
        input.space(),
        space.homspace(),
        bond.clone(),
        FactorSide::Left,
    )?;
    let vt_space =
        build_bound_factor_space(input.space(), space.homspace(), bond, FactorSide::Right)?;
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

    let mut singular_values = Vec::with_capacity(matricizations.len());

    let index = PlacementIndex::new(&matricizations, &[FactorSide::Left, FactorSide::Right]);
    let u_groups = SectorBlockGroups::new(u_space.space().structure(), FactorSide::Left)?;
    let vt_groups = SectorBlockGroups::new(vt_space.space().structure(), FactorSide::Right)?;
    for matrix in &matricizations {
        let rank = matrix.rows.min(matrix.cols);
        let (mut u, values, mut vt) =
            compact_svd_owned(dense, &matrix.data, matrix.rows, matrix.cols)?;
        match gauge {
            CompactSvdGauge::Left => svd_compact_gauge(
                &mut u,
                matrix.rows,
                matrix.rows,
                &mut vt,
                rank,
                matrix.cols,
                rank,
            ),
            CompactSvdGauge::AdjointLeft => svd_compact_adjoint_gauge(
                &mut u,
                matrix.rows,
                matrix.rows,
                &mut vt,
                rank,
                matrix.cols,
                rank,
            ),
        }
        #[cfg(test)]
        record_mf_compact_svd_fallback_gauge(&u, &vt);

        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values,
        });
        scatter_left_sector_blocks(
            u_space.space(),
            &mut u_data,
            matrix,
            &index,
            &u_groups,
            &u,
            matrix.rows,
        )?;
        #[cfg(test)]
        record_compact_svd_output_scatter::<D>(matrix.rows * rank);
        scatter_right_sector_blocks(
            vt_space.space(),
            &mut vt_data,
            matrix,
            &index,
            &vt_groups,
            &vt,
            rank,
        )?;
        #[cfg(test)]
        record_compact_svd_output_scatter::<D>(rank * matrix.cols);
    }

    let u = BoundDynFactor::from_bound(u_space, u_data, space.nout(), 1)?;
    let vh = BoundDynFactor::from_bound(vt_space, vt_data, 1, space.nin())?;
    Ok((u, vh, singular_values))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompactFactorRoute {
    source_region: usize,
    left_region: Option<usize>,
    right_region: Option<usize>,
    sector: SectorId,
    rank: usize,
}

/// Per-call routing from the source coupled-sector regions to the two factor
/// regions. Owned by the calling factorization and dropped with it.
///
/// Why not `Arc` the plan or its source half: nothing shares it beyond the
/// one call that builds it, so the wrappers were two heap allocations per
/// call with no owner to serve.
#[derive(Debug)]
pub(crate) struct CompactFactorPlan {
    source_layout: ValidatedDynamicFusionLayout,
    source_regions: Arc<[CoupledSectorRegion]>,
    left_layout: ValidatedDynamicFusionLayout,
    right_layout: ValidatedDynamicFusionLayout,
    left_regions: Arc<[CoupledSectorRegion]>,
    right_regions: Arc<[CoupledSectorRegion]>,
    routes: Vec<CompactFactorRoute>,
}

#[doc(hidden)]
#[derive(Debug)]
pub enum CheckedGenericFactorPlanError<E> {
    Provider(E),
    Operation(OperationError),
}

impl<E> From<CheckedGenericStructureError<E>> for CheckedGenericFactorPlanError<E> {
    fn from(error: CheckedGenericStructureError<E>) -> Self {
        match error {
            CheckedGenericStructureError::Provider(error) => Self::Provider(error),
            CheckedGenericStructureError::Core(error) => {
                Self::Operation(OperationError::from_core_preserving_context(error))
            }
        }
    }
}

impl<E> From<OperationError> for CheckedGenericFactorPlanError<E> {
    fn from(error: OperationError) -> Self {
        Self::Operation(error)
    }
}

pub(crate) struct PreparedGenericCompactFactorPlan {
    source_layout: ValidatedDynamicFusionLayout,
    source_regions: Arc<[CoupledSectorRegion]>,
    left: PreparedCheckedGenericDynamicSpace,
    right: PreparedCheckedGenericDynamicSpace,
    left_regions: Arc<[CoupledSectorRegion]>,
    right_regions: Arc<[CoupledSectorRegion]>,
    routes: Vec<CompactFactorRoute>,
}

fn compact_factor_plan<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    build_compact_factor_plan(input, input.validated_layout())
}

fn compact_factor_plan_generic<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: FusionRule,
{
    let checked = InfallibleGeneric::new(input.provider());
    match prepare_compact_factor_plan_generic_checked(input, &checked) {
        Ok(Some(prepared)) => finish_compact_factor_plan_generic(input, prepared),
        Ok(None) => Ok(None),
        Err(CheckedGenericFactorPlanError::Provider(never)) => match never {},
        Err(CheckedGenericFactorPlanError::Operation(error)) => Err(error),
    }
}

fn prepare_compact_factor_plan_generic_checked<R, P>(
    input: &BoundDynamicFusionMapSpace<R>,
    provider: &P,
) -> Result<Option<PreparedGenericCompactFactorPlan>, CheckedGenericFactorPlanError<P::Error>>
where
    R: FusionRule,
    P: CheckedGenericFusion,
{
    let space = input.space();
    let Some(regions) = checked_sector_regions(space.structure(), space.nout())? else {
        return Ok(None);
    };
    let new_leg = compact_bond_leg(&regions);
    let left_hom = FusionTreeHomSpace::new(
        space.homspace().codomain().clone(),
        FusionProductSpace::new([new_leg.clone()]),
    );
    let right_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg]),
        space.homspace().domain().clone(),
    );
    let left = input.prepare_final_homspace_generic_checked(provider, left_hom)?;
    let right = input.prepare_final_homspace_generic_checked(provider, right_hom)?;
    let left_regions = checked_sector_regions(left.structure(), space.nout())?.ok_or(
        OperationError::UnsupportedTensorContractScope {
            message: "compact left factor is not a coupled-sector matrix layout",
        },
    )?;
    let right_regions = checked_sector_regions(right.structure(), 1)?.ok_or(
        OperationError::UnsupportedTensorContractScope {
            message: "compact right factor is not a coupled-sector matrix layout",
        },
    )?;
    if !source_factor_tree_extents_match(&regions, &left_regions, &right_regions) {
        return Ok(None);
    }
    let routes = compile_compact_factor_routes(&regions, &left_regions, &right_regions)?;
    Ok(Some(PreparedGenericCompactFactorPlan {
        source_layout: input.validated_layout(),
        source_regions: regions,
        left,
        right,
        left_regions,
        right_regions,
        routes,
    }))
}

fn source_factor_tree_extents_match(
    source: &[CoupledSectorRegion],
    left: &[CoupledSectorRegion],
    right: &[CoupledSectorRegion],
) -> bool {
    let Ok(left_by_sector) = sector_region_index_map(left) else {
        return false;
    };
    let Ok(right_by_sector) = sector_region_index_map(right) else {
        return false;
    };
    source.iter().all(|source_region| {
        let sector = source_region.coupled();
        let Some(&left_index) = left_by_sector.get(&sector) else {
            return false;
        };
        let Some(&right_index) = right_by_sector.get(&sector) else {
            return false;
        };
        source_region.row_trees() == left[left_index].row_trees()
            && source_region.col_trees() == right[right_index].col_trees()
    })
}

fn finish_compact_factor_plan_generic<R>(
    input: &BoundDynamicFusionMapSpace<R>,
    prepared: PreparedGenericCompactFactorPlan,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: FusionRule,
{
    #[cfg(test)]
    GENERIC_FACTOR_PLAN_FINISH_CALLS.with(|calls| calls.set(calls.get() + 1));
    let PreparedGenericCompactFactorPlan {
        source_layout,
        source_regions,
        left,
        right,
        left_regions,
        right_regions,
        routes,
    } = prepared;
    let left = input.commit_final_homspace_generic_checked(left)?;
    let right = input.commit_final_homspace_generic_checked(right)?;
    Ok(Some(CompactFactorPlan {
        source_layout,
        source_regions,
        left_layout: left.validated_layout(),
        right_layout: right.validated_layout(),
        left_regions,
        right_regions,
        routes,
    }))
}

#[cfg(test)]
thread_local! {
    static GENERIC_FACTOR_PLAN_FINISH_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_generic_factor_plan_finish_calls() {
    GENERIC_FACTOR_PLAN_FINISH_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
pub(crate) fn generic_factor_plan_finish_calls() -> usize {
    GENERIC_FACTOR_PLAN_FINISH_CALLS.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn prepare_compact_factor_plan_generic_checked_for_test<R, P>(
    input: &BoundDynamicFusionMapSpace<R>,
    provider: &P,
) -> Result<Option<PreparedGenericCompactFactorPlan>, CheckedGenericFactorPlanError<P::Error>>
where
    R: FusionRule,
    P: CheckedGenericFusion,
{
    prepare_compact_factor_plan_generic_checked(input, provider)
}

#[cfg(test)]
pub(crate) fn finish_compact_factor_plan_generic_for_test<R>(
    input: &BoundDynamicFusionMapSpace<R>,
    prepared: PreparedGenericCompactFactorPlan,
) -> Result<bool, OperationError>
where
    R: FusionRule,
{
    finish_compact_factor_plan_generic(input, prepared).map(|plan| plan.is_some())
}

fn build_compact_factor_plan<R>(
    input: &BoundDynamicFusionMapSpace<R>,
    source_layout: ValidatedDynamicFusionLayout,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let space = input.space();
    let Some(regions) = checked_sector_regions(space.structure(), space.nout())? else {
        return Ok(None);
    };
    let bond = compact_bond_leg(&regions);
    let u_space =
        build_bound_factor_space(input, space.homspace(), bond.clone(), FactorSide::Left)?;
    let vh_space = build_bound_factor_space(input, space.homspace(), bond, FactorSide::Right)?;
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
    let routes = compile_compact_factor_routes(&regions, &left_regions, &right_regions)?;
    Ok(Some(CompactFactorPlan {
        source_layout,
        source_regions: regions,
        left_layout: u_space.validated_layout(),
        right_layout: vh_space.validated_layout(),
        left_regions,
        right_regions,
        routes,
    }))
}

/// The bond leg `W` shared by both compact factors: one sector per source
/// region with degeneracy `min(rows, cols)`.
fn compact_bond_leg(regions: &[CoupledSectorRegion]) -> SectorLeg {
    SectorLeg::new(
        regions
            .iter()
            .map(|region| (region_sector(region), region.rows().min(region.cols()))),
        false,
    )
}

fn compile_compact_factor_routes(
    source_regions: &[CoupledSectorRegion],
    left_regions: &[CoupledSectorRegion],
    right_regions: &[CoupledSectorRegion],
) -> Result<Vec<CompactFactorRoute>, OperationError> {
    let left_by_sector = SectorRegionIndex::new(left_regions)?;
    let right_by_sector = SectorRegionIndex::new(right_regions)?;
    let mut routes = Vec::with_capacity(source_regions.len());
    // Why not per-region `used` tables: the sector -> region index is
    // injective and each nonzero route consumes a distinct index per side, so
    // "an unused nonzero region exists" is exactly "used < nonzero regions".
    let mut used_left = 0usize;
    let mut used_right = 0usize;
    for (source_region, region) in source_regions.iter().enumerate() {
        let sector = region_sector(region);
        let rank = region.rows().min(region.cols());
        let (left_region, right_region) = if rank == 0 {
            (None, None)
        } else {
            let left_region = sector_region_index_of(&left_by_sector, sector, "left")?;
            let right_region = sector_region_index_of(&right_by_sector, sector, "right")?;
            validate_factor_region(&left_regions[left_region], region.rows(), rank, "left")?;
            validate_factor_region(&right_regions[right_region], rank, region.cols(), "right")?;
            used_left += 1;
            used_right += 1;
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
    validate_no_unused_factor_regions(left_regions, used_left, "left")?;
    validate_no_unused_factor_regions(right_regions, used_right, "right")?;
    Ok(routes)
}

#[cfg(test)]
pub(crate) fn compact_factor_plan_for_test<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    compact_factor_plan(input)
}

#[cfg(test)]
#[expect(
    clippy::type_complexity,
    reason = "the test seam returns source, left, and right region tables in documented order"
)]
pub(crate) fn compact_factor_plan_regions_for_test(
    plan: &CompactFactorPlan,
) -> (
    Arc<[CoupledSectorRegion]>,
    Arc<[CoupledSectorRegion]>,
    Arc<[CoupledSectorRegion]>,
) {
    (
        Arc::clone(&plan.source_regions),
        Arc::clone(&plan.left_regions),
        Arc::clone(&plan.right_regions),
    )
}

#[cfg(test)]
pub(crate) fn validate_compact_factor_routes_for_test(
    source: &[CoupledSectorRegion],
    u: &[CoupledSectorRegion],
    vh: &[CoupledSectorRegion],
) -> Result<Vec<CompactFactorRoute>, OperationError> {
    compile_compact_factor_routes(source, u, vh)
}

#[cfg(test)]
pub(crate) fn compact_factor_plan_routes_for_test(
    plan: &CompactFactorPlan,
) -> &[CompactFactorRoute] {
    &plan.routes
}

#[cfg(test)]
impl CompactFactorRoute {
    pub(crate) fn factor_regions_for_test(&self) -> (usize, Option<usize>, Option<usize>) {
        (self.source_region, self.left_region, self.right_region)
    }
}

fn svd_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
    gauge: CompactSvdGauge,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source_layout, input.space().validated_layout());
    let u_space = input.space().rebind_validated(&plan.left_layout)?;
    let vh_space = input.space().rebind_validated(&plan.right_layout)?;
    let mut u_regions = vec![None; plan.left_regions.len()];
    let mut vh_regions = vec![None; plan.right_regions.len()];
    let mut singular_values = Vec::with_capacity(plan.routes.len());

    for route in plan.routes.iter().copied() {
        let region = &plan.source_regions[route.source_region];
        let rank = route.rank;
        if rank == 0 {
            singular_values.push(SectorSpectrum {
                sector: route.sector,
                values: Vec::new(),
            });
            continue;
        }
        let left_region = route.left_region.expect("nonzero route has left region");
        let right_region = route.right_region.expect("nonzero route has right region");
        let (mut u, spectrum, mut vh) = compact_svd_owned(
            dense,
            &input.data()[region.range()],
            region.rows(),
            region.cols(),
        )?;
        match gauge {
            CompactSvdGauge::Left => svd_compact_gauge(
                &mut u,
                region.rows(),
                region.rows(),
                &mut vh,
                rank,
                region.cols(),
                rank,
            ),
            CompactSvdGauge::AdjointLeft => svd_compact_adjoint_gauge(
                &mut u,
                region.rows(),
                region.rows(),
                &mut vh,
                rank,
                region.cols(),
                rank,
            ),
        }
        u_regions[left_region] = Some(u);
        vh_regions[right_region] = Some(vh);
        singular_values.push(SectorSpectrum {
            sector: route.sector,
            values: spectrum,
        });
    }

    let u_data = concat_compact_svd_factor_regions(u_regions, plan.left_layout.required_len()?);
    let vh_data = concat_compact_svd_factor_regions(vh_regions, plan.right_layout.required_len()?);
    let u = BoundDynFactor::from_bound(u_space, u_data, space.nout(), 1)?;
    let vh = BoundDynFactor::from_bound(vh_space, vh_data, 1, space.nin())?;
    Ok((u, vh, singular_values))
}

fn region_sector(region: &CoupledSectorRegion) -> SectorId {
    region.coupled()
}

fn sector_region_index_map(
    regions: &[CoupledSectorRegion],
) -> Result<FxHashMap<SectorId, usize>, OperationError> {
    let mut by_sector = FxHashMap::with_capacity_and_hasher(regions.len(), Default::default());
    for (index, region) in regions.iter().enumerate() {
        if by_sector.insert(region_sector(region), index).is_some() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "coupled-sector region description contains a duplicate sector",
            });
        }
    }
    Ok(by_sector)
}

/// Sector -> region index lookup over one factor's region table.
///
/// Canonical factor layouts list their regions strictly sorted by coupled
/// sector (a structural invariant of the derived layout, checked here in
/// O(G)), so a binary search serves without building a map. Expert region
/// tables that are not sorted keep the hash-map path, which also reports the
/// duplicate-sector error.
enum SectorRegionIndex<'a> {
    Sorted(&'a [CoupledSectorRegion]),
    Map(FxHashMap<SectorId, usize>),
}

impl<'a> SectorRegionIndex<'a> {
    fn new(regions: &'a [CoupledSectorRegion]) -> Result<Self, OperationError> {
        if regions
            .windows(2)
            .all(|pair| region_sector(&pair[0]) < region_sector(&pair[1]))
        {
            Ok(Self::Sorted(regions))
        } else {
            sector_region_index_map(regions).map(Self::Map)
        }
    }

    fn get(&self, sector: SectorId) -> Option<usize> {
        match self {
            Self::Sorted(regions) => regions.binary_search_by_key(&sector, region_sector).ok(),
            Self::Map(map) => map.get(&sector).copied(),
        }
    }
}

fn sector_region_index_of(
    regions: &SectorRegionIndex<'_>,
    sector: SectorId,
    side: &'static str,
) -> Result<usize, OperationError> {
    regions
        .get(sector)
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
    used: usize,
    side: &'static str,
) -> Result<(), OperationError> {
    let nonzero = regions
        .iter()
        .filter(|region| region.rows() != 0 && region.cols() != 0)
        .count();
    if used < nonzero {
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
) -> Result<crate::truncation::TruncationDecision, OperationError>
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
            sector: entry.sector,
            weight: rule.dim_scalar(entry.sector),
            values: values.as_slice(),
        })
        .collect();
    select_truncation(&weighted, truncation, &rule.rule_identity()).map_err(OperationError::from)
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
    let SvdCompactDyn {
        u,
        s,
        vh,
        singular_values,
    } = compact;
    let full_rank = singular_values
        .iter()
        .map(|entry| entry.values.len())
        .sum::<usize>();
    let (u, vh, singular_values, error) =
        truncate_svd_factors_only_dyn(u, vh, singular_values, truncation)?;
    let kept_rank = singular_values
        .iter()
        .map(|entry| entry.values.len())
        .sum::<usize>();
    let s = if kept_rank == full_rank {
        s
    } else {
        diagonal_bond_svd_factor(u.space(), &singular_values, &D::from_real)?
    };
    Ok(SvdTruncDyn {
        u,
        s,
        vh,
        singular_values,
        error,
    })
}

/// Decides and applies SVD truncation without constructing the diagonal factor.
fn truncate_svd_factors_only_dyn<R, D>(
    u: BoundDynFactor<R, D>,
    vh: BoundDynFactor<R, D>,
    mut singular_values: Vec<SectorSpectrum>,
    truncation: &Truncation,
) -> Result<SvdTruncFactorsDyn<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let decision =
        decide_bond_truncation(u.space().provider(), &singular_values, truncation, true)?;
    if singular_values
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok((u, vh, singular_values, decision.error));
    }

    for (entry, &count) in singular_values.iter_mut().zip(&decision.kept) {
        entry.values.truncate(count);
    }
    singular_values.retain(|entry| !entry.values.is_empty());
    let kept_by_sector: FxHashMap<SectorId, usize> = singular_values
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
    Ok((u_factor, vh_factor, singular_values, decision.error))
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
    let space = authority.derive_from_final_homspace(new_hom)?;
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

struct GenericFactorPairSpaces<R> {
    left: BoundDynamicFusionMapSpace<R>,
    right: BoundDynamicFusionMapSpace<R>,
    left_keys: Vec<FusionTreePairKey>,
    right_keys: Vec<FusionTreePairKey>,
    ordered: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct GenericPairPublicationProbe {
    pub ordered_key_validation_events: usize,
    pub fallback_row_lookups: usize,
    pub fallback_col_lookups: usize,
    pub output_blocks_visited: usize,
    pub left_owner_reused: usize,
    pub right_owner_reused: usize,
    pub left_appended_elements: usize,
    pub right_appended_elements: usize,
    pub left_scattered_elements: usize,
    pub right_scattered_elements: usize,
    pub left_scatter_calls: usize,
    pub right_scatter_calls: usize,
    pub canonical_publications: usize,
    pub fallback_publications: usize,
}

#[cfg(test)]
thread_local! {
    static GENERIC_PAIR_PUBLICATION_PROBE: Cell<GenericPairPublicationProbe> = Cell::default();
}

#[cfg(test)]
pub(crate) fn reset_generic_pair_publication_probe() {
    GENERIC_PAIR_PUBLICATION_PROBE.set(GenericPairPublicationProbe::default());
}

#[cfg(test)]
pub(crate) fn generic_pair_publication_probe() -> GenericPairPublicationProbe {
    GENERIC_PAIR_PUBLICATION_PROBE.get()
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct OneSidedPublicationProbe {
    pub canonical_publications: usize,
    pub fallback_publications: usize,
    pub owner_reused: usize,
    pub appended_elements: usize,
}

#[cfg(test)]
thread_local! {
    static ONE_SIDED_PUBLICATION_PROBE: Cell<OneSidedPublicationProbe> = Cell::default();
}

#[cfg(test)]
pub(crate) fn reset_one_sided_publication_probe() {
    ONE_SIDED_PUBLICATION_PROBE.set(OneSidedPublicationProbe::default());
}

#[cfg(test)]
pub(crate) fn one_sided_publication_probe() -> OneSidedPublicationProbe {
    ONE_SIDED_PUBLICATION_PROBE.get()
}

#[cfg(test)]
thread_local! {
    static FACTOR_BUFFER_BUILD_COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

#[cfg(test)]
pub(crate) fn reset_factor_buffer_build_counts_for_test() {
    FACTOR_BUFFER_BUILD_COUNTS.set((0, 0));
}

#[cfg(test)]
pub(crate) fn factor_buffer_build_counts_for_test() -> (usize, usize) {
    FACTOR_BUFFER_BUILD_COUNTS.get()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum FactorSide {
    Left,
    Right,
}

#[derive(Clone, Copy)]
enum FactorPlacement {
    Direct,
    Adjoint,
}

fn build_bound_factor_space<R>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    new_leg: SectorLeg,
    side: FactorSide,
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let bond = FusionProductSpace::new([new_leg]);
    let hom = match side {
        FactorSide::Left => FusionTreeHomSpace::new(homspace.codomain().clone(), bond),
        FactorSide::Right => FusionTreeHomSpace::new(bond, homspace.domain().clone()),
    };
    authority.derive_from_final_homspace(hom)
}

#[doc(hidden)]
pub fn build_bound_factor_space_generic_checked<R>(
    authority: Arc<R>,
    homspace: &FusionTreeHomSpace,
    dimensions: impl IntoIterator<Item = (SectorId, usize)>,
    side: bool,
) -> Result<BoundDynamicFusionMapSpace<R>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
{
    let new_leg = SectorLeg::new(dimensions, false);
    let bond = FusionProductSpace::new([new_leg]);
    let hom = if side {
        FusionTreeHomSpace::new(homspace.codomain().clone(), bond)
    } else {
        FusionTreeHomSpace::new(bond, homspace.domain().clone())
    };
    BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(authority, hom)
        .map_err(CheckedGenericFactorPlanError::from)
}

/// Builds the `(codomain <- W, W <- domain)` factor pair shared by SVD and
/// the orthogonal factorizations, in the coupled-sector matrix layout.
fn build_left_right_bound_pair<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[SectorMatricization<D>],
    pairs: &mut [FactorPair<D>],
) -> Result<DynamicFactorPair<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let dimensions = pairs
        .iter()
        .map(|pair| (pair.sector, pair.kept))
        .collect::<BTreeMap<_, _>>();
    Ok((
        build_bound_factor(
            authority,
            homspace,
            matricizations,
            pairs,
            &dimensions,
            FactorSide::Left,
        )?,
        build_bound_factor(
            authority,
            homspace,
            matricizations,
            pairs,
            &dimensions,
            FactorSide::Right,
        )?,
    ))
}

fn build_left_bound_factor<R, D, M>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: &mut [FactorPair<D>],
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
    M: SectorGeometry,
{
    let dimensions = pairs
        .iter()
        .map(|pair| (pair.sector, pair.kept))
        .collect::<BTreeMap<_, _>>();
    build_bound_factor(
        authority,
        homspace,
        matricizations,
        pairs,
        &dimensions,
        FactorSide::Left,
    )
}

fn build_bound_factor<R, D, M>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: &mut [FactorPair<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
    M: SectorGeometry,
{
    build_bound_factor_with_placement(
        authority,
        homspace,
        matricizations,
        pairs,
        dimensions,
        side,
        FactorPlacement::Direct,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_bound_factor_with_placement<R, D, M>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: &mut [FactorPair<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
    placement: FactorPlacement,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
    M: SectorGeometry,
{
    let space = build_bound_factor_space(
        authority,
        homspace,
        SectorLeg::new(
            dimensions
                .iter()
                .map(|(&sector, &dimension)| (sector, dimension)),
            false,
        ),
        side,
    )?;
    let required_len = space.space().required_len()?;
    #[cfg(test)]
    FACTOR_BUFFER_BUILD_COUNTS.set({
        let (left, right) = FACTOR_BUFFER_BUILD_COUNTS.get();
        match side {
            FactorSide::Left => (left + 1, right),
            FactorSide::Right => (left, right + 1),
        }
    });
    let mut routes = matricizations
        .iter()
        .map(|matrix| (matrix.sector(), (matrix, None)))
        .collect::<FxHashMap<_, _>>();
    for pair in pairs.iter() {
        let route =
            routes
                .get_mut(&pair.sector)
                .ok_or(OperationError::UnsupportedTensorContractScope {
                    message: "factor sector absent from the source tensor",
                })?;
        route.1 = Some(pair);
    }
    let source_trees = match (side, placement) {
        (FactorSide::Left, FactorPlacement::Direct)
        | (FactorSide::Right, FactorPlacement::Adjoint) => FactorSide::Left,
        (FactorSide::Right, FactorPlacement::Direct)
        | (FactorSide::Left, FactorPlacement::Adjoint) => FactorSide::Right,
    };
    if one_sided_factor_output_is_canonical(
        space.space().structure(),
        matricizations,
        pairs,
        dimensions,
        required_len,
        side,
        source_trees,
    ) {
        let data = take_one_sided_factors(pairs, required_len, side);
        let (nout, nin) = match side {
            FactorSide::Left => (space.space().nout(), 1),
            FactorSide::Right => (1, space.space().nin()),
        };
        return BoundDynFactor::from_bound(space, data, nout, nin);
    }
    record_one_sided_fallback_publication();
    let mut data = vec![D::zero(); required_len];
    let mut missing_offsets = FxHashMap::<SectorId, usize>::default();
    let placements = PlacementIndex::new(matricizations, &[source_trees]);
    for index in 0..space.space().structure().block_count() {
        let block = space.space().structure().block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let (sector, matrix_axis) = match side {
            FactorSide::Left => (coupled_of(key.codomain_tree()), block.shape().len() - 1),
            FactorSide::Right => (coupled_of(key.domain_tree()), 0),
        };
        if let Some(&(_, Some(pair))) = routes.get(&sector) {
            let tree = match side {
                FactorSide::Left => key.codomain_tree(),
                FactorSide::Right => key.domain_tree(),
            };
            let side_offset = placements.placement(sector, source_trees, tree)?.0;
            let (factor, factor_rows) = match side {
                FactorSide::Left => (pair.left.as_slice(), pair.left_rows),
                FactorSide::Right => (pair.right.as_slice(), pair.right_leading),
            };
            scatter_matrix_block(
                &mut data,
                block.shape(),
                block.strides(),
                block.offset(),
                matrix_axis,
                side,
                factor,
                factor_rows,
                side_offset,
            );
            continue;
        }
        if routes.contains_key(&sector) {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "factor rank absent for a populated source sector",
            });
        }
        let dimension = dimensions[&sector];
        let side_offset = missing_offsets.entry(sector).or_default();
        let extent = block
            .shape()
            .iter()
            .enumerate()
            .filter(|&(axis, _)| axis != matrix_axis)
            .try_fold(1usize, |acc, (_, &value)| acc.checked_mul(value))
            .ok_or(OperationError::ElementCountOverflow)?;
        let side_end = side_offset
            .checked_add(extent)
            .ok_or(OperationError::ElementCountOverflow)?;
        if side_end > dimension {
            return Err(OperationError::ElementCountMismatch {
                expected: dimension,
                actual: side_end,
            });
        }
        scatter_identity_matrix_block(
            &mut data,
            block.shape(),
            block.strides(),
            block.offset(),
            matrix_axis,
            dimension,
            *side_offset,
            extent,
        )?;
        *side_offset = side_end;
    }
    let (nout, nin) = match side {
        FactorSide::Left => (space.space().nout(), 1),
        FactorSide::Right => (1, space.space().nin()),
    };
    BoundDynFactor::from_bound(space, data, nout, nin)
}

fn scatter_left_sector_blocks<D>(
    left_space: &DynamicFusionMapSpace,
    left_data: &mut [D],
    matrix: &SectorMatricization<D>,
    index: &PlacementIndex<'_>,
    groups: &SectorBlockGroups,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    D: FactorScalar,
{
    let left_structure = Arc::clone(left_space.structure());
    for block_index in groups.blocks(matrix.sector) {
        #[cfg(test)]
        record_scatter_visit(FactorSide::Left);
        let block = left_structure
            .block(block_index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        debug_assert_eq!(coupled_of(key.codomain_tree()), matrix.sector);
        let (row_offset, _) =
            index.placement(matrix.sector, FactorSide::Left, key.codomain_tree())?;
        scatter_matrix_block(
            left_data,
            block.shape(),
            block.strides(),
            block.offset(),
            block.shape().len() - 1,
            FactorSide::Left,
            factor,
            factor_rows,
            row_offset,
        );
    }
    Ok(())
}

fn scatter_right_sector_blocks<D>(
    right_space: &DynamicFusionMapSpace,
    right_data: &mut [D],
    matrix: &SectorMatricization<D>,
    index: &PlacementIndex<'_>,
    groups: &SectorBlockGroups,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    D: FactorScalar,
{
    let right_structure = Arc::clone(right_space.structure());
    for block_index in groups.blocks(matrix.sector) {
        #[cfg(test)]
        record_scatter_visit(FactorSide::Right);
        let block = right_structure
            .block(block_index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        debug_assert_eq!(coupled_of(key.domain_tree()), matrix.sector);
        let (col_offset, _) =
            index.placement(matrix.sector, FactorSide::Right, key.domain_tree())?;
        scatter_matrix_block(
            right_data,
            block.shape(),
            block.strides(),
            block.offset(),
            0,
            FactorSide::Right,
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
/// (the user layer, via compact diagonal storage) never pay the O(rank²)
/// materialization.
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
///
/// Before any dense call, every coupled-sector block `A` must satisfy
/// `||(A - A†)/2||_F <= 64 * eps(real(D)) * ||A||_F`, where `real(D)` is
/// the real component type of `D`. This fixed machine-precision multiple is
/// not currently user-configurable in this API.
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

/// Dynamic-rank [`eigh_full`]: the shared core, with the same fixed
/// relative-Frobenius Hermiticity criterion.
pub fn eigh_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<EighFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires an endomorphism (codomain == domain)",
        });
    }
    if let Some(plan) = compact_factor_plan(input.space())? {
        return eigh_full_direct_regions(dense, input, &plan);
    }
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_eigh_input_pack(&matricizations);
    validate_hermitian_matricizations(&matricizations)?;

    let ranks = matricizations
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector,
            kept: matrix.rows,
        })
        .collect::<Vec<_>>();
    let v_space = build_bound_factor_space(
        input.space(),
        space.homspace(),
        SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false),
        FactorSide::Left,
    )?;
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
    let mut order = Vec::with_capacity(max_n);
    let mut visited = vec![false; max_n];
    let mut column_scratch = vec![D::zero(); max_n];
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    let index = PlacementIndex::new(&matricizations, &[FactorSide::Left]);
    let v_groups = SectorBlockGroups::new(v_space.space().structure(), FactorSide::Left)?;
    for matrix in &matricizations {
        let n = matrix.rows;
        let (real_values, mut vectors) = compact_eigh_owned(dense, &matrix.data, n)?;
        validate_real_eigenvalues(&real_values)?;

        order.clear();
        order.extend(0..n);
        // Reorder bond states descending by |eigenvalue| (stable on ties).
        order.sort_by(|&a, &b| {
            real_values[b]
                .abs()
                .total_cmp(&real_values[a].abs())
                .then(a.cmp(&b))
        });
        let sorted_values: Vec<f64> = order.iter().map(|&index| real_values[index]).collect();
        reorder_columns_in_place(&mut vectors, n, &order, &mut visited, &mut column_scratch);
        eigenvector_gauge(&mut vectors, n, n, n);
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted_values,
        });
        #[cfg(test)]
        record_eigh_owned_vector_before_scatter(&vectors);
        scatter_left_sector_blocks(
            v_space.space(),
            &mut v_data,
            matrix,
            &index,
            &v_groups,
            &vectors,
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
    debug_assert_eq!(plan.source_layout, input.space().validated_layout());
    validate_hermitian_regions(input.data(), &plan.source_regions)?;

    let v_space = input.space().rebind_validated(&plan.left_layout)?;
    let v_len = plan.left_layout.required_len()?;
    let max_n = plan
        .source_regions
        .iter()
        .map(CoupledSectorRegion::rows)
        .max()
        .unwrap_or(0);
    let mut order = Vec::with_capacity(max_n);
    let mut visited = vec![false; max_n];
    let mut column_scratch = vec![D::zero(); max_n];
    let mut eigenvalues = Vec::with_capacity(plan.routes.len());
    let mut regions = vec![None; plan.left_regions.len()];
    let mut next_left_region = 0;
    let mut output = None;

    for route in plan.routes.iter().copied() {
        let source = &plan.source_regions[route.source_region];
        let n = source.rows();
        if n == 0 {
            eigenvalues.push(SectorSpectrum {
                sector: route.sector,
                values: Vec::new(),
            });
            continue;
        }
        let (real_values, mut vectors) =
            compact_eigh_owned(dense, &input.data()[source.range()], n)?;
        validate_real_eigenvalues(&real_values)?;

        order.clear();
        order.extend(0..n);
        order.sort_by(|&a, &b| {
            real_values[b]
                .abs()
                .total_cmp(&real_values[a].abs())
                .then(a.cmp(&b))
        });
        let sorted_values = order.iter().map(|&index| real_values[index]).collect();
        reorder_columns_in_place(&mut vectors, n, &order, &mut visited, &mut column_scratch);
        eigenvector_gauge(&mut vectors, n, n, n);
        regions[route.left_region.expect("nonzero route has left region")] = Some(vectors);
        while next_left_region < plan.left_regions.len() {
            if plan.left_regions[next_left_region].range().is_empty() {
                next_left_region += 1;
                continue;
            }
            let Some(region) = regions[next_left_region].take() else {
                break;
            };
            append_owned_factor(&mut output, region, v_len);
            next_left_region += 1;
        }
        eigenvalues.push(SectorSpectrum {
            sector: route.sector,
            values: sorted_values,
        });
    }

    Ok(EighFullDyn {
        v: BoundDynFactor::from_bound(v_space, output.unwrap_or_default(), space.nout(), 1)?,
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
    let decision = decide_bond_truncation(rule, &full.eigenvalues, truncation, false)?;
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
    let kept_by_sector: FxHashMap<SectorId, usize> = eigenvalues
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
    #[expect(
        clippy::type_complexity,
        reason = "the public decomposition accessor returns its named components in documented order"
    )]
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
    svd_full_oriented_dyn(dense, input, FactorPlacement::Direct)
}

/// Full SVD factors for the logical adjoint without constructing its input.
#[doc(hidden)]
pub fn svd_full_adjoint_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    svd_full_oriented_dyn(dense, input, FactorPlacement::Adjoint)
}

#[expect(
    clippy::type_complexity,
    reason = "the private stage returns the fixed dense full-SVD tuple without another wrapper"
)]
fn owned_full_svd_stage<E, D>(
    dense: &mut E,
    data: &mut Vec<D>,
    rows: usize,
    cols: usize,
) -> Result<Option<(Vec<D>, Vec<f64>, Vec<D>)>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    if !dense.supports_svd_full() {
        return Ok(None);
    }
    let input = match D::dense_into_owned(std::mem::take(data)) {
        Ok(input) => input,
        Err(input_data) => {
            *data = input_data;
            return Ok(None);
        }
    };
    let mut outputs = dense
        .svd_full_owned(input, rows, cols)
        .map_err(OperationError::Dense)?;
    if outputs.len() != 3 {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "svd_full_owned",
            message: "dense full SVD must return exactly (U, S, Vh)".to_string(),
        }));
    }
    let u = compact_factor_output_owned(outputs.remove(0), &[rows, rows], "svd_full_owned")?;
    let singular_values =
        compact_real_spectrum_owned::<D>(outputs.remove(0), &[rows.min(cols)], "svd_full_owned")?;
    let vh = compact_factor_output_owned(outputs.remove(0), &[cols, cols], "svd_full_owned")?;
    Ok(Some((u, singular_values, vh)))
}

fn svd_full_oriented_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    placement: FactorPlacement,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    let mut matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    let row_dimensions = space
        .homspace()
        .codomain()
        .coupled_sector_block_dimensions(input.space().provider())?;
    let col_dimensions = space
        .homspace()
        .domain()
        .coupled_sector_block_dimensions(input.space().provider())?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    let mut singular_values = Vec::with_capacity(matricizations.len());
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
    let mut u_workspace = Vec::new();
    let mut s_workspace = Vec::new();
    let mut vt_workspace = Vec::new();
    for matrix in &mut matricizations {
        let rank = matrix.rows.min(matrix.cols);
        let (mut left, left_rows, mut right, right_leading, s_values) =
            match owned_full_svd_stage(dense, &mut matrix.data, matrix.rows, matrix.cols)? {
                Some((u_full, s_values, vh_full)) => match placement {
                    FactorPlacement::Direct => {
                        (u_full, matrix.rows, vh_full, matrix.cols, s_values)
                    }
                    FactorPlacement::Adjoint => (
                        adjoint_col_major(&vh_full, matrix.cols, matrix.cols),
                        matrix.cols,
                        adjoint_col_major(&u_full, matrix.rows, matrix.rows),
                        matrix.rows,
                        s_values,
                    ),
                },
                None => {
                    if u_workspace.is_empty() && max_rows != 0 && max_rank != 0 {
                        u_workspace = vec![D::zero(); max_rows * max_rank];
                        s_workspace = vec![D::Real::zero(); max_rank];
                        vt_workspace = vec![D::zero(); max_rank * max_cols];
                    }
                    let shape = [matrix.rows, matrix.cols];
                    let strides = [1usize, matrix.rows];
                    let view = DenseView::new(&matrix.data, &shape, &strides, 0)
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
                    let u_full = orthonormal_completion(dense, &u_thin, matrix.rows, rank)?;
                    let v_thin = adjoint_col_major(&vt_thin, rank, matrix.cols);
                    let v_full = orthonormal_completion(dense, &v_thin, matrix.cols, rank)?;
                    match placement {
                        FactorPlacement::Direct => (
                            u_full,
                            matrix.rows,
                            adjoint_col_major(&v_full, matrix.cols, matrix.cols),
                            matrix.cols,
                            s_values,
                        ),
                        FactorPlacement::Adjoint => (
                            v_full,
                            matrix.cols,
                            adjoint_col_major(&u_full, matrix.rows, matrix.rows),
                            matrix.rows,
                            s_values,
                        ),
                    }
                }
            };
        svd_full_gauge(
            &mut left,
            left_rows,
            left_rows,
            &mut right,
            right_leading,
            right_leading,
        );

        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: left_rows,
            left,
            left_rows,
            right,
            right_leading,
        });
    }

    let adjoint_space = match placement {
        FactorPlacement::Direct => None,
        FactorPlacement::Adjoint => Some(tenet_tensors::adjoint_bound_space_dyn(input.space())?),
    };
    let authority = adjoint_space.as_ref().unwrap_or(input.space());
    let homspace = authority.space().homspace();
    let (output_row_dimensions, output_col_dimensions) = match placement {
        FactorPlacement::Direct => (&row_dimensions, &col_dimensions),
        FactorPlacement::Adjoint => (&col_dimensions, &row_dimensions),
    };

    // The left/right bond legs differ in the full SVD (rows vs columns), so
    // build the two factors with separate bond dimensions.
    let u_factor = build_bound_factor_with_placement(
        authority,
        homspace,
        &matricizations,
        &mut pairs,
        output_row_dimensions,
        FactorSide::Left,
        placement,
    )?;
    let vh_factor = build_bound_factor_with_placement(
        authority,
        homspace,
        &matricizations,
        &mut pairs,
        output_col_dimensions,
        FactorSide::Right,
        placement,
    )?;
    let s_factor = rectangular_diagonal_bond_tensor(
        authority,
        &singular_values,
        output_row_dimensions,
        output_col_dimensions,
    )?;
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
    row_dimensions: &BTreeMap<SectorId, usize>,
    col_dimensions: &BTreeMap<SectorId, usize>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let row_leg = SectorLeg::new(
        row_dimensions
            .iter()
            .map(|(&sector, &dimension)| (sector, dimension)),
        false,
    );
    let col_leg = SectorLeg::new(
        col_dimensions
            .iter()
            .map(|(&sector, &dimension)| (sector, dimension)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([row_leg]),
        FusionProductSpace::new([col_leg]),
    );
    let space = authority.derive_from_final_homspace(homspace)?;
    let len = space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); len];
    let structure = Arc::clone(space.space().structure());
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum> =
        spectra.iter().map(|entry| (entry.sector, entry)).collect();
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree.codomain_tree().coupled();
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

#[doc(hidden)]
pub fn rectangular_diagonal_bond_tensor_generic_checked<R, D>(
    provider: Arc<R>,
    spectra: &[SectorSpectrum],
    row_dimensions: &BTreeMap<SectorId, usize>,
    col_dimensions: &BTreeMap<SectorId, usize>,
    to_scalar: &dyn Fn(f64) -> D,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let row_leg = SectorLeg::new(
        row_dimensions
            .iter()
            .map(|(&sector, &dimension)| (sector, dimension)),
        false,
    );
    let col_leg = SectorLeg::new(
        col_dimensions
            .iter()
            .map(|(&sector, &dimension)| (sector, dimension)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([row_leg]),
        FusionProductSpace::new([col_leg]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
        .map_err(CheckedGenericFactorPlanError::from)?;
    let len = space.space().required_len().map_err(|error| {
        CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(
            error,
        ))
    })?;
    let mut data = vec![D::zero(); len];
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum> =
        spectra.iter().map(|entry| (entry.sector, entry)).collect();
    let structure = Arc::clone(space.space().structure());
    for index in 0..structure.block_count() {
        let block = structure.block(index).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(
                error,
            ))
        })?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let Some(entry) = spectrum_by_sector.get(&tree.codomain_tree().coupled()) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        for position in 0..count {
            let Some(&value) = entry.values.get(position) else {
                break;
            };
            data[offset + position * (strides[0] + strides[1])] = to_scalar(value);
        }
    }
    BoundDynFactor::from_bound(space, data, 1, 1).map_err(CheckedGenericFactorPlanError::from)
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

pub(crate) fn svd_compact_adjoint_gauge<D: FactorScalar>(
    u: &mut [D],
    u_rows: usize,
    u_leading: usize,
    vh: &mut [D],
    vh_rows: usize,
    vh_cols: usize,
    vh_leading: usize,
) {
    for j in 0..vh_rows {
        let (phase, needs_scaling) = phase_of_largest_abs_row(vh, vh_cols, vh_leading, j);
        if needs_scaling {
            scale_col(u, u_rows, u_leading, j, phase);
            scale_row(vh, vh_cols, vh_leading, j, FactorScalar::adjoint(phase));
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

#[cfg(test)]
pub(crate) fn reorder_columns_in_place_for_test<D: Copy>(
    vectors: &mut [D],
    n: usize,
    order: &[usize],
    visited: &mut [bool],
    column_scratch: &mut [D],
) {
    reorder_columns_in_place(vectors, n, order, visited, column_scratch);
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

fn full_qr_numerical_stage<E, D>(
    dense: &mut E,
    input: &[D],
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut q = vec![D::zero(); rows * rows];
    let mut r = if rows <= cols {
        let mut r = vec![D::zero(); rows * cols];
        qr_into_workspace(
            dense, input, rows, cols, rows, &mut q, rows, rows, rows, &mut r, rows, cols, rows,
        )?;
        r
    } else {
        // The current full-Q completion retains augmentation until #1140 A3
        // supplies a supported efficient dense-backend path.
        let mut augmented = vec![D::zero(); rows * (cols + rows)];
        augmented[..rows * cols].copy_from_slice(input);
        for row in 0..rows {
            augmented[rows * cols + row * rows + row] = D::one();
        }
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
        work_r[..rows * cols].to_vec()
    };
    positive_diagonal_gauge(&mut q, rows, &mut r, rows, cols);
    Ok((q, r))
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
/// from one economy QR, augmenting with identity columns only when `m > n`.
/// The positive-diagonal gauge is applied (MAK / TensorKit 0.17 default).
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
pub fn qr_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    let matrices = sector_matricizations(space.structure(), input.data(), space.nout())?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let (q, r) = full_qr_numerical_stage(dense, &matrix.data, rows, cols)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rows,
            left: q,
            left_rows: rows,
            right: r,
            right_leading: rows,
        });
    }
    let dimensions = space
        .homspace()
        .codomain()
        .coupled_sector_block_dimensions(input.space().provider())?;
    Ok((
        build_bound_factor(
            input.space(),
            space.homspace(),
            &matrices,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )?,
        build_bound_factor(
            input.space(),
            space.homspace(),
            &matrices,
            &mut pairs,
            &dimensions,
            FactorSide::Right,
        )?,
    ))
}

/// Full LQ `t = L * Q` (MatrixAlgebraKit `lq_full`): per sector `L` is the
/// lower-trapezoidal `m x n` and `Q` the square `n x n` unitary, via the full
/// QR of the adjoint sector matrices.
/// The positive-diagonal gauge is applied (MAK / TensorKit 0.17 default).
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
pub fn lq_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    let matrices = sector_matricizations(space.structure(), input.data(), space.nout())?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let transposed = adjoint_col_major(&matrix.data, rows, cols);
        let (q_prime, r_prime) = full_qr_numerical_stage(dense, &transposed, cols, rows)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: cols,
            left: adjoint_col_major(&r_prime, cols, rows),
            left_rows: rows,
            right: adjoint_col_major(&q_prime, cols, cols),
            right_leading: cols,
        });
    }
    let dimensions = space
        .homspace()
        .domain()
        .coupled_sector_block_dimensions(input.space().provider())?;
    Ok((
        build_bound_factor(
            input.space(),
            space.homspace(),
            &matrices,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )?,
        build_bound_factor(
            input.space(),
            space.homspace(),
            &matrices,
            &mut pairs,
            &dimensions,
            FactorSide::Right,
        )?,
    ))
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
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eig requires an endomorphism (codomain == domain)",
        });
    }
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;

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
        validate_complex_eigenvalues(&complex_values)?;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            complex_values[b]
                .norm()
                .total_cmp(&complex_values[a].norm())
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
            left: sorted_vectors,
            left_rows: n,
            right: Vec::new(),
            right_leading: n,
        });
    }

    let v_factor =
        build_left_bound_factor(input.space(), space.homspace(), &matricizations, &mut pairs)?;
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
    let decision = decide_bond_truncation(rule, &full.eigenvalues, truncation, false)?;
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
    let kept_by_sector: FxHashMap<SectorId, usize> = eigenvalues
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
///
/// Uses the same fixed relative-Frobenius Hermiticity criterion as
/// [`eigh_full`]; this API does not expose `atol` or `rtol`.
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
    let matricizations = value_matricizations(space.structure(), input.data(), space.nout())?;
    matricizations.validate_hermitian()?;
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    for index in 0..matricizations.len() {
        let matrix = matricizations.get(index)?;
        let n = matrix.rows;
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let values_tensor = dense
            .eigh_vals(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        let mut sorted = D::real_spectrum(&values_tensor).map_err(OperationError::Dense)?;
        sorted.truncate(n);
        validate_real_eigenvalues(&sorted)?;
        sorted.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
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
    let matricizations = value_matricizations(space.structure(), input.data(), space.nout())?;
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    for index in 0..matricizations.len() {
        let matrix = matricizations.get(index)?;
        let n = matrix.rows;
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let values_tensor = dense
            .eig_vals(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        validate_dense_shape(values_tensor.shape(), &[n])?;
        let values =
            <D::Eig as FactorScalar>::dense_slice(&values_tensor).map_err(OperationError::Dense)?;
        let mut sorted: Vec<Complex64> = values[..n].iter().map(|&v| v.widen_complex()).collect();
        validate_complex_eigenvalues(&sorted)?;
        sorted.sort_by(|a, b| b.norm().total_cmp(&a.norm()));
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
    let space = input.space().space();
    let matrices = sector_matricizations(space.structure(), input.data(), space.nout())?;
    // A codomain-only sector has no tensor block but is entirely left-null.
    let mut null_dimensions = space
        .homspace()
        .codomain()
        .coupled_sector_block_dimensions(input.space().provider())?;
    let mut pairs = Vec::new();
    for matrix in &matrices {
        let (rows, cols) = (matrix.rows, matrix.cols);
        let (rank, u_compact, _) =
            numerical_rank_and_compact_bases(dense, &matrix.data, rows, cols)?;
        if rank == rows {
            null_dimensions.remove(&matrix.sector);
            continue;
        }
        // Only the left basis is completed: completing V would run an unused
        // QR for this operation.
        let u = orthonormal_completion(dense, &u_compact, rows, rows.min(cols))?;
        let null_dim = rows - rank;
        null_dimensions.insert(matrix.sector, null_dim);
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: null_dim,
            left: u[rows * rank..].to_vec(),
            left_rows: rows,
            right: Vec::new(),
            right_leading: null_dim,
        });
    }
    build_bound_factor(
        input.space(),
        space.homspace(),
        &matrices,
        &mut pairs,
        &null_dimensions,
        FactorSide::Left,
    )
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
    let space = input.space().space();
    let matrices = sector_matricizations(space.structure(), input.data(), space.nout())?;
    // A domain-only sector has no tensor block but is entirely right-null.
    let mut null_dimensions = space
        .homspace()
        .domain()
        .coupled_sector_block_dimensions(input.space().provider())?;
    let mut pairs = Vec::new();
    for matrix in &matrices {
        let (rows, cols) = (matrix.rows, matrix.cols);
        let (rank, _, v_compact) =
            numerical_rank_and_compact_bases(dense, &matrix.data, rows, cols)?;
        if rank == cols {
            null_dimensions.remove(&matrix.sector);
            continue;
        }
        // Only the right basis is completed: completing U would run an unused
        // QR for this operation.
        let v = orthonormal_completion(dense, &v_compact, cols, rows.min(cols))?;
        let null_dim = cols - rank;
        null_dimensions.insert(matrix.sector, null_dim);
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: null_dim,
            left: Vec::new(),
            left_rows: rows,
            right: adjoint_col_major(&v[cols * rank..], cols, null_dim),
            right_leading: null_dim,
        });
    }
    build_bound_factor(
        input.space(),
        space.homspace(),
        &matrices,
        &mut pairs,
        &null_dimensions,
        FactorSide::Right,
    )
}

/// Checked-Generic numerical left null space.
///
/// Structural dimensions are validated before dense work. All SVDs and
/// completions are then staged before the exact data-dependent bond is
/// admitted and scattered by the shared checked factor builder.
#[doc(hidden)]
pub fn left_null_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = sector_matricizations_generic(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut null_dimensions = coupled_sector_block_dimensions_generic_checked(
        space.homspace().codomain(),
        provider.as_ref(),
    )?;
    let mut pairs = Vec::new();
    for matrix in &matrices {
        let (rank, u_compact, _) =
            numerical_rank_and_compact_bases(dense, &matrix.data, matrix.rows, matrix.cols)
                .map_err(CheckedGenericFactorPlanError::from)?;
        if rank == matrix.rows {
            null_dimensions.remove(&matrix.sector);
            continue;
        }
        let u =
            orthonormal_completion(dense, &u_compact, matrix.rows, matrix.rows.min(matrix.cols))
                .map_err(CheckedGenericFactorPlanError::from)?;
        let null_dim = matrix.rows - rank;
        null_dimensions.insert(matrix.sector, null_dim);
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: null_dim,
            left: u[matrix.rows * rank..].to_vec(),
            left_rows: matrix.rows,
            right: Vec::new(),
            right_leading: null_dim,
        });
    }
    build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        &matrices,
        &mut pairs,
        &null_dimensions,
        FactorSide::Left,
    )
}

/// Checked-Generic numerical right null space; see
/// [`left_null_dyn_checked_generic`] for the transaction boundary.
#[doc(hidden)]
pub fn right_null_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = sector_matricizations_generic(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut null_dimensions = coupled_sector_block_dimensions_generic_checked(
        space.homspace().domain(),
        provider.as_ref(),
    )?;
    let mut pairs = Vec::new();
    for matrix in &matrices {
        let (rank, _, v_compact) =
            numerical_rank_and_compact_bases(dense, &matrix.data, matrix.rows, matrix.cols)
                .map_err(CheckedGenericFactorPlanError::from)?;
        if rank == matrix.cols {
            null_dimensions.remove(&matrix.sector);
            continue;
        }
        let v =
            orthonormal_completion(dense, &v_compact, matrix.cols, matrix.rows.min(matrix.cols))
                .map_err(CheckedGenericFactorPlanError::from)?;
        let null_dim = matrix.cols - rank;
        null_dimensions.insert(matrix.sector, null_dim);
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: null_dim,
            left: Vec::new(),
            left_rows: matrix.rows,
            right: adjoint_col_major(&v[matrix.cols * rank..], matrix.cols, null_dim),
            right_leading: null_dim,
        });
    }
    build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        &matrices,
        &mut pairs,
        &null_dimensions,
        FactorSide::Right,
    )
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
/// the domain. Every coupled-sector matrix must have at least as many rows as
/// columns; otherwise this returns [`OperationError::InvalidArgument`] before
/// entering the dense SVD.
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
    left_polar_dyn_reported(dense, context, input, PolarDirection::Left)
}

fn left_polar_dyn_reported<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
    error_direction: PolarDirection,
) -> Result<DynamicFactorPair<R, D>, OperationError>
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
    let (u, vh, singular_values) = svd_compact_factors_dyn_with_direction(
        dense,
        input,
        Some((PolarDirection::Left, error_direction)),
        CompactSvdGauge::Left,
    )?;
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
/// Every coupled-sector matrix must have at least as many columns as rows;
/// otherwise this returns [`OperationError::InvalidArgument`] before entering
/// the dense SVD.
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
    right_polar_dyn_reported(dense, context, input, PolarDirection::Right)
}

fn right_polar_dyn_reported<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
    error_direction: PolarDirection,
) -> Result<DynamicFactorPair<R, D>, OperationError>
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
    let (u, vh, singular_values) = svd_compact_factors_dyn_with_direction(
        dense,
        input,
        Some((PolarDirection::Right, error_direction)),
        CompactSvdGauge::Left,
    )?;
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

/// Left polar factors of an adjoint view, executed on its owned parent.
#[doc(hidden)]
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
pub fn left_polar_adjoint_parent_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    parent: &BoundDynamicTensorRef<'_, R, D>,
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
    let (positive, isometry) =
        right_polar_dyn_reported(dense, context, parent, PolarDirection::Left)?;
    Ok((adjoint_bound_factor(&isometry)?, positive))
}

/// Right polar factors of an adjoint view, executed on its owned parent.
#[doc(hidden)]
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
pub fn right_polar_adjoint_parent_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    parent: &BoundDynamicTensorRef<'_, R, D>,
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
    let (isometry, positive) =
        left_polar_dyn_reported(dense, context, parent, PolarDirection::Right)?;
    Ok((positive, adjoint_bound_factor(&isometry)?))
}

/// Compact QR `t = Q * R` (MatrixAlgebraKit `qr_compact`):
/// `Q : codomain <- W` has orthonormal columns per coupled sector and
/// `R : W <- domain` with per-sector bond `min(rows, cols)`. The
/// positive-diagonal gauge is applied (MAK / TensorKit 0.17 default
/// `positive = true`): `R`'s diagonal is real non-negative per sector.
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
    let space = input.space().space();
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_qr_input_pack(&matricizations);
    let mut pairs = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let rank = matrix.rows.min(matrix.cols);
        let (mut q, mut r) = compact_qr_owned(dense, &matrix.data, matrix.rows, matrix.cols)?;
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
    build_left_right_bound_pair(input.space(), space.homspace(), &matricizations, &mut pairs)
}

fn qr_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
) -> Result<DynamicFactorPair<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source_layout, input.space().validated_layout());
    let left_space = input.space().rebind_validated(&plan.left_layout)?;
    let right_space = input.space().rebind_validated(&plan.right_layout)?;
    let mut left_regions = (0..plan.left_regions.len())
        .map(|_| None)
        .collect::<Vec<_>>();
    let mut right_regions = (0..plan.right_regions.len())
        .map(|_| None)
        .collect::<Vec<_>>();

    for route in plan.routes.iter().copied() {
        if route.rank == 0 {
            continue;
        }
        let source = &plan.source_regions[route.source_region];
        let (mut left_data, mut right_data) = compact_qr_owned(
            dense,
            &input.data()[source.range()],
            source.rows(),
            source.cols(),
        )?;
        positive_diagonal_gauge_strided(
            &mut left_data,
            source.rows(),
            source.rows(),
            &mut right_data,
            route.rank,
            route.rank,
            source.cols(),
        );
        left_regions[route.left_region.expect("nonzero route has left region")] = Some(left_data);
        right_regions[route.right_region.expect("nonzero route has right region")] =
            Some(right_data);
    }

    let left_data = concat_compact_factor_regions(left_regions, plan.left_layout.required_len()?);
    let right_data =
        concat_compact_factor_regions(right_regions, plan.right_layout.required_len()?);

    let left = BoundDynFactor::from_bound(left_space, left_data, space.nout(), 1)?;
    let right = BoundDynFactor::from_bound(right_space, right_data, 1, space.nin())?;
    Ok((left, right))
}

/// Compact LQ `t = L * Q` (MatrixAlgebraKit `lq_compact`, via the QR of the
/// transposed sector matrices): `Q : W <- domain` has orthonormal rows per
/// coupled sector and `L : codomain <- W`. The positive-diagonal gauge is
/// applied (MAK / TensorKit 0.17 default `positive = true`): `L`'s diagonal
/// is real non-negative per sector.
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
    let space = input.space().space();
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_lq_input_pack(&matricizations);
    let mut pairs = Vec::with_capacity(matricizations.len());
    for matrix in &matricizations {
        let rank = matrix.rows.min(matrix.cols);
        let adjoint = adjoint_col_major(&matrix.data, matrix.rows, matrix.cols);
        let (mut q_prime, mut r_prime) =
            compact_qr_owned(dense, &adjoint, matrix.cols, matrix.rows)?;
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
    build_left_right_bound_pair(input.space(), space.homspace(), &matricizations, &mut pairs)
}

fn lq_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
) -> Result<DynamicFactorPair<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source_layout, input.space().validated_layout());
    let left_space = input.space().rebind_validated(&plan.left_layout)?;
    let right_space = input.space().rebind_validated(&plan.right_layout)?;
    let mut left_data = vec![D::zero(); plan.left_layout.required_len()?];
    let mut right_data = vec![D::zero(); plan.right_layout.required_len()?];

    let max_adjoint_len = plan
        .routes
        .iter()
        .map(|route| plan.source_regions[route.source_region].range().len())
        .max()
        .unwrap_or(0);
    let mut adjoint_scratch = vec![D::zero(); max_adjoint_len];
    #[cfg(test)]
    record_compact_lq_scratch::<D>(max_adjoint_len);

    for route in plan.routes.iter().copied() {
        if route.rank == 0 {
            continue;
        }
        let source = &plan.source_regions[route.source_region];
        let left = &plan.left_regions[route.left_region.expect("nonzero route has left region")];
        let right =
            &plan.right_regions[route.right_region.expect("nonzero route has right region")];
        let source_data = &input.data()[source.range()];
        let adjoint = &mut adjoint_scratch[..source_data.len()];
        adjoint_col_major_into(source_data, source.rows(), source.cols(), adjoint);
        #[cfg(test)]
        record_compact_lq_adjoint_fill::<D>(source_data.len());

        let (mut q_prime, mut r_prime) =
            compact_qr_owned(dense, adjoint, source.cols(), source.rows())?;
        positive_diagonal_gauge_strided(
            &mut q_prime,
            source.cols(),
            source.cols(),
            &mut r_prime,
            route.rank,
            route.rank,
            source.rows(),
        );
        adjoint_col_major_into(
            &r_prime,
            route.rank,
            source.rows(),
            &mut left_data[left.range()],
        );
        #[cfg(test)]
        record_compact_lq_final_adjoint_copy::<D>(r_prime.len());
        adjoint_col_major_into(
            &q_prime,
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
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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

/// Compact QR owns both dense outputs, so it can transfer the executor's host
/// buffers directly. Full QR keeps its caller-owned workspace contract above.
fn compact_qr_owned<E, D>(
    dense: &mut E,
    input: &[D],
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let rank = rows.min(cols);
    let input_shape = [rows, cols];
    let input_strides = [1usize, rows];
    let input_view =
        DenseView::new(input, &input_shape, &input_strides, 0).map_err(OperationError::Dense)?;
    let mut outputs = dense
        .qr(D::dense_read(input_view))
        .map_err(OperationError::Dense)?;
    if outputs.len() != 2 {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "qr_into",
            message: "dense QR must return exactly (Q, R)".to_string(),
        }));
    }
    let q = compact_factor_output_owned::<D>(outputs.remove(0), &[rows, rank], "qr_into")?;
    let r = compact_factor_output_owned::<D>(outputs.remove(0), &[rank, cols], "qr_into")?;
    Ok((q, r))
}

/// Compact SVD owns U and Vt, while S remains the host-side spectrum used by
/// the existing truncation and diagonal construction paths.
#[expect(
    clippy::type_complexity,
    reason = "the dense SVD ownership boundary returns its documented U, S, Vt tuple"
)]
fn compact_svd_owned<E, D>(
    dense: &mut E,
    input: &[D],
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<f64>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let rank = rows.min(cols);
    let input_shape = [rows, cols];
    let input_strides = [1usize, rows];
    let input_view =
        DenseView::new(input, &input_shape, &input_strides, 0).map_err(OperationError::Dense)?;
    let mut outputs = dense
        .svd(D::dense_read(input_view))
        .map_err(OperationError::Dense)?;
    if outputs.len() != 3 {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "svd_into",
            message: "dense SVD must return exactly (U, S, Vt)".to_string(),
        }));
    }
    let u = compact_factor_output_owned::<D>(outputs.remove(0), &[rows, rank], "svd_into")?;
    let singular_values = compact_real_spectrum_owned::<D>(outputs.remove(0), &[rank], "svd_into")?;
    let vt = compact_factor_output_owned::<D>(outputs.remove(0), &[rank, cols], "svd_into")?;
    Ok((u, singular_values, vt))
}

fn compact_eigh_owned<E, D>(
    dense: &mut E,
    input: &[D],
    order: usize,
) -> Result<(Vec<f64>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let shape = [order, order];
    let strides = [1usize, order];
    let input = DenseView::new(input, &shape, &strides, 0).map_err(OperationError::Dense)?;
    let mut outputs = dense
        .eigh(D::dense_read(input))
        .map_err(OperationError::Dense)?;
    if outputs.len() != 2 {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eigh_into",
            message: "dense EIGH must return exactly (values, vectors)".to_string(),
        }));
    }
    let values = compact_real_spectrum_owned::<D>(outputs.remove(0), &[order], "eigh_into")?;
    let vectors =
        compact_factor_output_owned::<D>(outputs.remove(0), &[order, order], "eigh_into")?;
    Ok((values, vectors))
}

fn compact_factor_output_owned<D: FactorScalar>(
    tensor: DenseTensor,
    expected_shape: &[usize],
    op: &'static str,
) -> Result<Vec<D>, OperationError> {
    let source = D::dense_slice(&tensor).map_err(OperationError::Dense)?;
    let shape = tensor.shape();
    if shape != expected_shape {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output shape mismatch: source {shape:?}, destination {expected_shape:?}",
            ),
        }));
    }
    let expected_len = shape
        .iter()
        .try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or(DenseError::ElementCountOverflow)
        })
        .map_err(OperationError::Dense)?;
    if source.len() != expected_len {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output storage length mismatch: source {}, expected {}",
                source.len(),
                expected_len
            ),
        }));
    }
    D::dense_into_vec(tensor).map_err(OperationError::Dense)
}

fn compact_real_spectrum_owned<D: FactorScalar>(
    tensor: DenseTensor,
    expected_shape: &[usize],
    op: &'static str,
) -> Result<Vec<f64>, OperationError> {
    let spectrum = D::real_spectrum(&tensor).map_err(OperationError::Dense)?;
    let shape = tensor.shape();
    if shape != expected_shape {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output shape mismatch: source {shape:?}, destination {expected_shape:?}",
            ),
        }));
    }
    let expected_len = shape
        .iter()
        .try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or(DenseError::ElementCountOverflow)
        })
        .map_err(OperationError::Dense)?;
    if spectrum.len() != expected_len {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output storage length mismatch: source {}, expected {}",
                spectrum.len(),
                expected_len
            ),
        }));
    }
    Ok(spectrum)
}

fn concat_compact_factor_regions<D>(regions: Vec<Option<Vec<D>>>, required_len: usize) -> Vec<D> {
    #[cfg(test)]
    let first = regions
        .iter()
        .find_map(|region| region.as_ref().map(Vec::as_ptr));
    let output = concat_owned_factor_regions(regions, required_len);
    #[cfg(test)]
    COMPACT_QR_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.owned_output_publications += 1;
        current.owned_output_owner_reused +=
            usize::from(first.is_some_and(|pointer| std::ptr::eq(pointer, output.as_ptr())));
        probe.set(current);
    });
    output
}

fn concat_owned_factor_regions<D>(regions: Vec<Option<Vec<D>>>, required_len: usize) -> Vec<D> {
    let mut output = None;
    for region in regions.into_iter().flatten() {
        append_owned_factor(&mut output, region, required_len);
    }
    output.unwrap_or_default()
}

fn concat_compact_svd_factor_regions<D>(
    regions: Vec<Option<Vec<D>>>,
    required_len: usize,
) -> Vec<D> {
    #[cfg(test)]
    let first = regions
        .iter()
        .find_map(|region| region.as_ref().map(Vec::as_ptr));
    let output = concat_owned_factor_regions(regions, required_len);
    #[cfg(test)]
    COMPACT_SVD_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.owned_output_publications += 1;
        current.owned_output_owner_reused +=
            usize::from(first.is_some_and(|pointer| std::ptr::eq(pointer, output.as_ptr())));
        probe.set(current);
    });
    output
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
/// `factor_side` names the factor layout (`Left`: `a x b`, element `(o, j)` at
/// `F[o + a*j]`; `Right`: `b x cols`, element `(j, o)` at `F[j + b*o]`).
#[allow(clippy::too_many_arguments)]
fn scatter_matrix_block<D: Copy>(
    data: &mut [D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    matrix_axis: usize,
    factor_side: FactorSide,
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
        // A rank-1 block (zero-leg tree) has `matrix_axis == 0 == rank - 1` on
        // both sides, so `matrix_axis` alone cannot encode the source layout;
        // only a Left factor walks the bond with stride `matrix_rows`.
        let (src_start, src_lane_stride) = if matrix_axis == 0 {
            if matrix_axis == rank - 1 && factor_side == FactorSide::Left {
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

#[allow(clippy::too_many_arguments)]
fn scatter_identity_matrix_block<D: FactorScalar>(
    data: &mut [D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    matrix_axis: usize,
    matrix_dimension: usize,
    side_offset: usize,
    side_extent: usize,
) -> Result<(), OperationError> {
    if shape.get(matrix_axis).copied() != Some(matrix_dimension) {
        return Err(OperationError::ElementCountMismatch {
            expected: matrix_dimension,
            actual: shape.get(matrix_axis).copied().unwrap_or(0),
        });
    }
    for local_side in 0..side_extent {
        let matrix_index = side_offset
            .checked_add(local_side)
            .ok_or(OperationError::ElementCountOverflow)?;
        let mut destination = offset + matrix_index * strides[matrix_axis];
        let mut remaining = local_side;
        for axis in 0..shape.len() {
            if axis == matrix_axis {
                continue;
            }
            let coordinate = remaining % shape[axis];
            remaining /= shape[axis];
            destination += coordinate * strides[axis];
        }
        data[destination] = D::one();
    }
    Ok(())
}

fn coupled_of(tree: &FusionTreeKey) -> SectorId {
    tree.coupled()
}

fn matricization_map<M: SectorGeometry>(matricizations: &[M]) -> FxHashMap<SectorId, &M> {
    matricizations
        .iter()
        .map(|matrix| (matrix.sector(), matrix))
        .collect()
}

fn matricization_of<'a, M>(
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

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PlacementIndexProbe {
    /// Index tables allocated (one per owner call).
    pub index_builds: usize,
    /// `(matricization, side)` pairs inserted into a table.
    pub indexed_sides: usize,
    pub indexed_trees: usize,
    pub lookups: usize,
}

#[cfg(test)]
thread_local! {
    static PLACEMENT_INDEX_PROBE: Cell<PlacementIndexProbe> = Cell::default();
}

#[cfg(test)]
pub(crate) fn reset_placement_index_probe() {
    PLACEMENT_INDEX_PROBE.set(PlacementIndexProbe::default());
}

#[cfg(test)]
pub(crate) fn placement_index_probe() -> PlacementIndexProbe {
    PLACEMENT_INDEX_PROBE.get()
}

/// Call-local first-match index of source trees by `(sector, side, key)`,
/// built once per publication call over every matricization.
///
/// Why not scan `tree(side, i)` per output block: the references reach a
/// block in O(1) (TensorKit's hashed sector dictionary, QSpace's cumulative
/// offsets), while the scan cost O(T_c·K) per block. Why one table for the
/// whole call rather than one per matricization and side: at G=16 with one
/// or two trees per sector the per-table allocation dominated the lookup
/// saving (+72 allocation calls per paired publication). For a handful of
/// trees the scan's early exit can still beat SipHash plus this single
/// allocation; that constant is disclosed, not dispatched on. Why not key
/// by matricization instead of sector: `sector_matricizations*` produce one
/// matricization per coupled sector and region admission rejects duplicate
/// sectors, so the sector already identifies the matricization.
struct PlacementIndex<'a> {
    by_tree: FxHashMap<(SectorId, FactorSide, &'a FusionTreeKey), (usize, &'a [usize])>,
}

impl<'a> PlacementIndex<'a> {
    fn new<M: SectorGeometry>(matricizations: &'a [M], sides: &[FactorSide]) -> Self {
        debug_assert_eq!(
            matricizations
                .iter()
                .map(SectorGeometry::sector)
                .collect::<FxHashSet<_>>()
                .len(),
            matricizations.len(),
            "placement index requires one matricization per coupled sector"
        );
        let capacity = matricizations
            .iter()
            .map(|matrix| {
                sides
                    .iter()
                    .map(|&side| matrix.tree_count(side))
                    .sum::<usize>()
            })
            .sum();
        let mut by_tree = FxHashMap::with_capacity_and_hasher(capacity, Default::default());
        for matrix in matricizations {
            let sector = matrix.sector();
            for &side in sides {
                let count = matrix.tree_count(side);
                for index in 0..count {
                    if let Some(extent) = matrix.tree(side, index) {
                        by_tree
                            .entry((sector, side, extent.tree))
                            .or_insert((extent.offset, extent.shape));
                    }
                }
                #[cfg(test)]
                PLACEMENT_INDEX_PROBE.with(|probe| {
                    let mut value = probe.get();
                    value.indexed_sides += 1;
                    value.indexed_trees += count;
                    probe.set(value);
                });
            }
        }
        #[cfg(test)]
        PLACEMENT_INDEX_PROBE.with(|probe| {
            let mut value = probe.get();
            value.index_builds += 1;
            probe.set(value);
        });
        Self { by_tree }
    }

    fn placement(
        &self,
        sector: SectorId,
        side: FactorSide,
        tree: &FusionTreeKey,
    ) -> Result<(usize, &'a [usize]), OperationError> {
        #[cfg(test)]
        PLACEMENT_INDEX_PROBE.with(|probe| {
            let mut value = probe.get();
            value.lookups += 1;
            probe.set(value);
        });
        self.by_tree.get(&(sector, side, tree)).copied().ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: match side {
                    FactorSide::Left => "factor codomain tree absent from the source matricization",
                    FactorSide::Right => "factor domain tree absent from the source matricization",
                },
            },
        )
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ScatterVisitProbe {
    /// Output blocks scanned by the grouping pass, per side (`B`).
    pub left_grouped: usize,
    pub right_grouped: usize,
    /// Groupings built, per side (one per publication call).
    pub left_groups_built: usize,
    pub right_groups_built: usize,
    /// Output blocks iterated by the paired scatter helpers, per side (`F`).
    pub left_visits: usize,
    pub right_visits: usize,
}

#[cfg(test)]
thread_local! {
    static SCATTER_VISIT_PROBE: Cell<ScatterVisitProbe> = Cell::default();
}

#[cfg(test)]
pub(crate) fn reset_scatter_visit_probe() {
    SCATTER_VISIT_PROBE.set(ScatterVisitProbe::default());
}

#[cfg(test)]
pub(crate) fn scatter_visit_probe() -> ScatterVisitProbe {
    SCATTER_VISIT_PROBE.get()
}

#[cfg(test)]
fn record_scatter_visit(side: FactorSide) {
    SCATTER_VISIT_PROBE.with(|probe| {
        let mut value = probe.get();
        match side {
            FactorSide::Left => value.left_visits += 1,
            FactorSide::Right => value.right_visits += 1,
        }
        probe.set(value);
    });
}

/// Output block indices of one factor side grouped by the coupled sector of
/// that side's tree, in structure order within a sector. Built once per
/// publication call next to [`PlacementIndex`] and dropped at return, so a
/// paired publication over `G_s` matricizations visits `B + F` output blocks
/// per side (one grouping pass plus the scattered blocks) instead of
/// `G_s * B`; the cost is one `16 * B`-byte allocation per side per call,
/// paid also for `G_s = 1`.
///
/// Why not one structure-major pass routing each output block to its
/// matricization: the compact SVD and EIGH consumers keep each sector's dense
/// factor in a workspace reused across sectors, so scattering has to happen
/// per matricization, and a structure-major pass would also change which
/// missing-tree error fires first when two sectors are defective. Why not a
/// tenet-core accessor: nothing exposes the coupled grouping
/// (`sorted_indices` is uncoupled-major and the coupled ranges known at
/// layout time are dropped at `BlockStructure`).
struct SectorBlockGroups {
    /// `(coupled sector, block index)` sorted by sector; the stable sort keeps
    /// ascending structure order within a sector.
    entries: Vec<(SectorId, usize)>,
}

impl SectorBlockGroups {
    fn new(structure: &BlockStructure, side: FactorSide) -> Result<Self, OperationError> {
        let mut entries = Vec::with_capacity(structure.block_count());
        for index in 0..structure.block_count() {
            let block = structure
                .block(index)
                .map_err(OperationError::from_core_preserving_context)?;
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            let tree = match side {
                FactorSide::Left => key.codomain_tree(),
                FactorSide::Right => key.domain_tree(),
            };
            entries.push((coupled_of(tree), index));
        }
        entries.sort_by_key(|&(sector, _)| sector);
        #[cfg(test)]
        SCATTER_VISIT_PROBE.with(|probe| {
            let mut value = probe.get();
            match side {
                FactorSide::Left => {
                    value.left_grouped += structure.block_count();
                    value.left_groups_built += 1;
                }
                FactorSide::Right => {
                    value.right_grouped += structure.block_count();
                    value.right_groups_built += 1;
                }
            }
            probe.set(value);
        });
        Ok(Self { entries })
    }

    fn blocks(&self, sector: SectorId) -> impl Iterator<Item = usize> + '_ {
        let start = self.entries.partition_point(|&(s, _)| s < sector);
        self.entries[start..]
            .iter()
            .take_while(move |&&(s, _)| s == sector)
            .map(|&(_, index)| index)
    }
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

fn scaled_antihermitian_difference<D: FactorScalar, R: HermitianReal>(
    data: &[D],
    n: usize,
    row: usize,
    col: usize,
    scale: R,
) -> (R, R) {
    let upper = data[row + n * col].widen_complex();
    let lower = data[col + n * row].widen_complex();
    let residual_re = R::from_f64(upper.re) / scale - R::from_f64(lower.re) / scale;
    let residual_im = R::from_f64(upper.im) / scale + R::from_f64(lower.im) / scale;
    (residual_re, residual_im)
}

#[derive(Clone, Copy)]
struct ScaledFrobenius<R> {
    scale: R,
    sum_squares: R,
}

impl<R: HermitianReal> ScaledFrobenius<R> {
    fn zero() -> Self {
        Self {
            scale: R::zero(),
            sum_squares: R::one(),
        }
    }

    fn add(&mut self, magnitude: R) -> bool {
        if !magnitude.is_finite() {
            return false;
        }
        if magnitude == R::zero() {
            return true;
        }
        if self.scale < magnitude {
            let ratio = self.scale / magnitude;
            self.sum_squares = R::one() + self.sum_squares * ratio * ratio;
            self.scale = magnitude;
        } else {
            let ratio = magnitude / self.scale;
            self.sum_squares = self.sum_squares + ratio * ratio;
        }
        true
    }

    fn add_complex(&mut self, re: R, im: R) -> bool {
        self.add(re.abs()) && self.add(im.abs())
    }

    fn scaled_norm(self, scale: R) -> R {
        if self.scale == R::zero() {
            R::zero()
        } else {
            (self.scale / scale) * self.sum_squares.sqrt()
        }
    }
}

fn hermitian_residual_norm<D: FactorScalar, R: HermitianReal>(
    data: &[D],
    n: usize,
    input_scale: R,
) -> Option<ScaledFrobenius<R>> {
    const BLOCK_SIZE: usize = 32;
    let mut residual = ScaledFrobenius::zero();
    for block_col in (0..n).step_by(BLOCK_SIZE) {
        let block_width = BLOCK_SIZE.min(n - block_col);
        for local_col in 0..block_width {
            let col = block_col + local_col;
            for local_row in 0..=local_col {
                let row = block_col + local_row;
                let (re, im) =
                    scaled_antihermitian_difference::<D, R>(data, n, row, col, input_scale);
                if !residual.add_complex(re, im) || (row != col && !residual.add_complex(re, im)) {
                    return None;
                }
            }
        }

        for block_row in (0..block_col).step_by(BLOCK_SIZE) {
            for local_col in 0..block_width {
                let col = block_col + local_col;
                for local_row in 0..BLOCK_SIZE {
                    let row = block_row + local_row;
                    let (re, im) =
                        scaled_antihermitian_difference::<D, R>(data, n, row, col, input_scale);
                    if !residual.add_complex(re, im) || !residual.add_complex(re, im) {
                        return None;
                    }
                }
            }
        }
    }
    Some(residual)
}

/// Tests `||(A - A†)/2||_F <= 64 * eps(R) * ||A||_F`.
///
/// The scaled sums keep that relative decision stable when either norm would
/// overflow or underflow if formed directly.
fn normwise_hermitian<D: FactorScalar, R: HermitianReal>(data: &[D], n: usize) -> bool {
    let mut input = ScaledFrobenius::zero();
    for &value in data {
        let value = value.widen_complex();
        let re = R::from_f64(value.re);
        let im = R::from_f64(value.im);
        if !re.is_finite() || !im.is_finite() {
            return false;
        }
        if !input.add_complex(re, im) {
            return false;
        }
    }

    if input.scale == R::zero() {
        return true;
    }
    let Some(residual) = hermitian_residual_norm::<D, R>(data, n, input.scale) else {
        return false;
    };
    residual.scaled_norm(R::one())
        <= (R::one() + R::one()) * R::relative_tolerance() * input.sum_squares.sqrt()
}

fn validate_hermitian_matrix_shape<D>(
    data: &[D],
    rows: usize,
    cols: usize,
) -> Result<(), OperationError> {
    let expected = rows
        .checked_mul(cols)
        .ok_or(OperationError::ElementCountOverflow)?;
    if data.len() != expected {
        return Err(OperationError::ElementCountMismatch {
            expected,
            actual: data.len(),
        });
    }
    if rows != cols {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires square coupled-sector matrices",
        });
    }
    Ok(())
}

/// Scale-invariant normwise hermiticity predicate at the working precision of `D`.
///
/// Extracted from [`validate_hermitian_matrix_contents`] so the `exp` dispatch
/// (issue #577) can ask the question without provoking — and then having to
/// interpret — an EIGH failure. Same data, same tolerance, no second policy.
fn hermitian_matrix_contents<D: FactorScalar>(data: &[D], n: usize) -> bool {
    if D::epsilon() == f32::EPSILON as f64 {
        normwise_hermitian::<D, f32>(data, n)
    } else if D::epsilon() == f64::EPSILON {
        normwise_hermitian::<D, f64>(data, n)
    } else {
        false
    }
}

fn validate_hermitian_matrix_contents<D: FactorScalar>(
    data: &[D],
    n: usize,
) -> Result<(), OperationError> {
    if !hermitian_matrix_contents(data, n) {
        return Err(OperationError::InvalidArgument {
            message: "eigh requires Hermitian coupled-sector blocks",
        });
    }
    Ok(())
}

fn validate_hermitian_matricizations<D: FactorScalar>(
    matricizations: &[SectorMatricization<D>],
) -> Result<(), OperationError> {
    for matrix in matricizations {
        validate_hermitian_matrix_shape(&matrix.data, matrix.rows, matrix.cols)?;
    }
    for matrix in matricizations {
        validate_hermitian_matrix_contents(&matrix.data, matrix.rows)?;
    }
    Ok(())
}

#[doc(hidden)]
pub fn validate_hermitian_regions<D: FactorScalar>(
    data: &[D],
    regions: &[CoupledSectorRegion],
) -> Result<(), OperationError> {
    for region in regions {
        let range = region.range();
        let matrix = data
            .get(range.clone())
            .ok_or(OperationError::ElementCountMismatch {
                expected: range.end,
                actual: data.len(),
            })?;
        validate_hermitian_matrix_shape(matrix, region.rows(), region.cols())?;
    }
    for region in regions {
        let range = region.range();
        let matrix = data
            .get(range.clone())
            .ok_or(OperationError::ElementCountMismatch {
                expected: range.end,
                actual: data.len(),
            })?;
        validate_hermitian_matrix_contents(matrix, region.rows())?;
    }
    Ok(())
}

/// Is this an endomorphism whose coupled-sector blocks are all Hermitian?
///
/// The `exp` dispatch (issue #577) needs the Hermitian question answered
/// *separately* from the eigendecomposition: the spectral route stays for
/// Hermitian input, and everything else goes to blockwise Padé. Inferring it
/// from a failed EIGH would conflate hermiticity with a backend failure, so
/// this asks directly, over the same direct-region / packed matricization
/// split and the same relative Frobenius tolerance [`eigh_full_dyn`] uses.
///
/// A non-endomorphism, a malformed layout or a non-square block is still an
/// error — only non-hermiticity is `Ok(false)`. Nonfinite entries make
/// the predicate `false`, so they arrive at the Padé route, which rejects them
/// in its own words.
///
/// Cost is `O(Σ_c n_c²)`, one pass over the blocks, against the `O(Σ_c n_c³)`
/// factorization that follows.
pub(crate) fn is_hermitian_endomorphism_dyn<R, D>(
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires an endomorphism (codomain == domain)",
        });
    }
    // Why `checked_sector_regions` and not `compact_factor_plan` as
    // `eigh_full_dyn` does: the plan is `Some` exactly when the regions are
    // (`build_compact_factor_plan` returns early otherwise), and building it
    // also builds the factor bond spaces — work a yes/no question must not pay
    // for on the retained Hermitian route.
    if let Some(regions) = checked_sector_regions(space.structure(), space.nout())? {
        for region in regions.iter() {
            let range = region.range();
            let matrix = data_region(input.data(), &range)?;
            validate_hermitian_matrix_shape(matrix, region.rows(), region.cols())?;
        }
        for region in regions.iter() {
            let range = region.range();
            let matrix = data_region(input.data(), &range)?;
            if !hermitian_matrix_contents(matrix, region.rows()) {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    for matrix in &matricizations {
        validate_hermitian_matrix_shape(&matrix.data, matrix.rows, matrix.cols)?;
    }
    Ok(matricizations
        .iter()
        .all(|matrix| hermitian_matrix_contents(&matrix.data, matrix.rows)))
}

fn data_region<'a, D>(
    data: &'a [D],
    range: &std::ops::Range<usize>,
) -> Result<&'a [D], OperationError> {
    data.get(range.clone())
        .ok_or(OperationError::ElementCountMismatch {
            expected: range.end,
            actual: data.len(),
        })
}

fn checked_sector_regions(
    structure: &BlockStructure,
    nout: usize,
) -> Result<Option<Arc<[CoupledSectorRegion]>>, OperationError> {
    structure
        .coupled_sector_regions(nout)
        .map_err(OperationError::from_core_preserving_context)
}

fn canonical_generic_sector_regions(
    structure: &BlockStructure,
    nout: usize,
) -> Result<Option<Arc<[CoupledSectorRegion]>>, OperationError> {
    let Some(regions) = checked_sector_regions(structure, nout)? else {
        return Ok(None);
    };
    let sectors_are_ordered = regions
        .windows(2)
        .all(|pair| pair[0].coupled() < pair[1].coupled());
    let trees_are_ordered = regions.iter().all(|region| {
        region
            .row_trees()
            .windows(2)
            .all(|pair| pair[0].tree() < pair[1].tree())
            && region
                .col_trees()
                .windows(2)
                .all(|pair| pair[0].tree() < pair[1].tree())
    });
    Ok((sectors_are_ordered && trees_are_ordered).then_some(regions))
}

fn generic_value_matricizations<'a, D>(
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

fn generic_input_matricizations<'a, D>(
    structure: &BlockStructure,
    data: &'a [D],
    nout: usize,
) -> Result<InputMatricizations<'a, D>, OperationError>
where
    D: FactorScalar,
{
    let regions = canonical_generic_sector_regions(structure, nout)?;
    Ok(match regions {
        Some(regions) => InputMatricizations::Regions { data, regions },
        None => InputMatricizations::Packed(sector_matricizations_generic(structure, data, nout)?),
    })
}

fn value_matricizations<'a, D>(
    structure: &BlockStructure,
    data: &'a [D],
    nout: usize,
) -> Result<InputMatricizations<'a, D>, OperationError>
where
    D: FactorScalar,
{
    if let Some(regions) = checked_sector_regions(structure, nout)? {
        Ok(InputMatricizations::Regions { data, regions })
    } else {
        #[cfg(test)]
        record_values_matricization_fallback();
        Ok(InputMatricizations::Packed(sector_matricizations(
            structure, data, nout,
        )?))
    }
}

/// Packs every coupled sector of the source data into its dense column-major
/// matricization, independent of the storage layout.
fn sector_matricizations<D>(
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
    Ok(matricizations)
}

#[derive(Clone, Copy)]
struct InverseSectorRoute {
    source: usize,
    output: usize,
}

#[derive(Clone, Copy)]
struct SolveLeftSectorRoute {
    divisor: usize,
    rhs: usize,
    output: usize,
}

struct InverseMatrixRoute {
    source: usize,
    output: usize,
    rows: Vec<InverseBasisExtent>,
    cols: Vec<InverseBasisExtent>,
}

#[derive(Clone, Copy)]
struct InverseBasisExtent {
    source_offset: usize,
    output_offset: usize,
    extent: usize,
}

pub(crate) fn inverse_by_sector_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    if !input.space().codomain_isomorphic_to_domain()? {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inv requires isomorphic codomain and domain",
        });
    }

    let source_space = input.space().space();
    let inverse_homspace = FusionTreeHomSpace::new(
        source_space.homspace().domain().clone(),
        source_space.homspace().codomain().clone(),
    );
    let output_space = input.space().derive_from_final_homspace(inverse_homspace)?;
    inverse_by_sector_dyn_into(dense, input, output_space)
}

/// Coefficient-free inverse execution into an already admitted swapped output.
///
/// The caller owns categorical admission; this body only routes the existing
/// coupled-sector layout and performs the dense solves.
pub(crate) fn inverse_by_sector_dyn_into<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let source_space = input.space().space();
    let mut output_data = vec![D::zero(); output_space.space().required_len()?];

    let source_regions = checked_sector_regions(source_space.structure(), source_space.nout())?;
    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "inverse derived output requires canonical coupled-sector storage",
    })?;
    match source_regions {
        Some(source) => {
            let routes = compile_inverse_region_routes(
                &source,
                &output_regions,
                input.data().len(),
                output_data.len(),
            )?;
            let max_order = routes
                .iter()
                .map(|route| source[route.source].rows())
                .max()
                .unwrap_or(0);
            let identity = identity_workspace::<D>(max_order)?;
            for route in routes {
                let source_region = &source[route.source];
                if source_region.rows() == 0 {
                    continue;
                }
                let source_matrix = &input.data()[source_region.range()];
                let output_matrix = &mut output_data[output_regions[route.output].range()];
                solve_inverse_sector(
                    dense,
                    source_matrix,
                    output_matrix,
                    source_region.rows(),
                    source_region.rows(),
                    &identity,
                    max_order,
                )?;
            }
        }
        None => {
            let source_matrices =
                sector_matricizations(source_space.structure(), input.data(), source_space.nout())?;
            let routes = compile_inverse_matrix_routes(
                &source_matrices,
                &output_regions,
                output_data.len(),
            )?;
            let max_order = routes
                .iter()
                .map(|route| source_matrices[route.source].rows)
                .max()
                .unwrap_or(0);
            let identity = identity_workspace::<D>(max_order)?;
            let mut solution = vec![D::zero(); identity.len()];
            for route in routes {
                let source = &source_matrices[route.source];
                if source.rows == 0 {
                    continue;
                }
                solve_inverse_sector(
                    dense,
                    &source.data,
                    &mut solution,
                    source.rows,
                    max_order,
                    &identity,
                    max_order,
                )?;
                let output = &output_regions[route.output];
                reorder_inverse_solution(
                    &solution,
                    max_order,
                    &mut output_data[output.range()],
                    output.rows(),
                    &route.rows,
                    &route.cols,
                );
            }
        }
    }

    BoundDynFactor::from_bound(
        output_space,
        output_data,
        source_space.nin(),
        source_space.nout(),
    )
}

/// Coefficient-free pseudo-inverse into an already admitted swapped space.
///
/// All layout checks happen before staging or dense work.  The local staging
/// keeps an SVD or GEMM failure from publishing a partial output.
pub(crate) fn pinv_by_sector_dyn_into<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
    rcond: f64,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let source_space = input.space().space();
    let Some(source_identity) = source_space.admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    let Some(output_identity) = output_space.space().admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    if source_identity != output_identity {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch {
                expected: source_identity.clone(),
                actual: output_identity.clone(),
            },
        ));
    }
    if !Arc::ptr_eq(input.space().provider_arc(), output_space.provider_arc()) {
        return Err(OperationError::StructureMismatch {
            tensor: "pinv output provider authority",
        });
    }
    let expected_homspace = FusionTreeHomSpace::new(
        source_space.homspace().domain().clone(),
        source_space.homspace().codomain().clone(),
    );
    let expected_nout = expected_homspace.codomain().len();
    let expected_nin = expected_homspace.domain().len();
    if output_space.space().nout() != expected_nout
        || output_space.space().nin() != expected_nin
        || output_space.space().rank() != expected_nout + expected_nin
        || output_space.space().homspace() != &expected_homspace
    {
        return Err(OperationError::StructureMismatch {
            tensor: "pinv output space",
        });
    }
    let source_regions =
        canonical_generic_sector_regions(source_space.structure(), source_space.nout())?.ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: "pinv requires canonical coupled-sector input storage",
            },
        )?;
    let output_regions = canonical_generic_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "pinv requires canonical coupled-sector output storage",
    })?;
    let routes = compile_pinv_region_routes(
        &source_regions,
        &output_regions,
        input.data().len(),
        output_space.space().required_len()?,
    )?;

    struct Stage<D> {
        route: InverseSectorRoute,
        rows: usize,
        cols: usize,
        rank: usize,
        u: Vec<D>,
        singular_values: Vec<f64>,
        vt: Vec<D>,
    }

    let mut output_data = vec![D::zero(); output_space.space().required_len()?];
    let mut staged = Vec::with_capacity(routes.len());
    for route in routes {
        let source = &source_regions[route.source];
        let rows = source.rows();
        let cols = source.cols();
        let rank = rows.min(cols);
        let (u, singular_values, vt) = if rank == 0 {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            compact_svd_owned(dense, &input.data()[source.range()], rows, cols)?
        };
        staged.push(Stage {
            route,
            rows,
            cols,
            rank,
            u,
            singular_values,
            vt,
        });
    }
    let cutoff = rcond
        * staged
            .iter()
            .flat_map(|stage| stage.singular_values.iter().copied())
            .fold(0.0_f64, f64::max);
    for stage in staged {
        if stage.rank == 0 {
            continue;
        }
        let mut vt = stage.vt;
        for (row, &sigma) in stage.singular_values.iter().enumerate() {
            let reciprocal = D::from_real(if sigma > cutoff { 1.0 / sigma } else { 0.0 });
            for column in 0..stage.cols {
                vt[row + stage.rank * column] = vt[row + stage.rank * column] * reciprocal;
            }
        }
        let output = &mut output_data[output_regions[stage.route.output].range()];
        let output_shape = [stage.cols, stage.rows];
        let output_strides = [1, stage.cols];
        let v_shape = [stage.cols, stage.rank];
        let v_strides = [stage.rank, 1];
        let uh_shape = [stage.rank, stage.rows];
        let uh_strides = [stage.rows, 1];
        let output_view = DenseViewMut::new(output, &output_shape, &output_strides, 0)
            .map_err(OperationError::Dense)?;
        let v_view = DenseView::new(&vt, &v_shape, &v_strides, 0).map_err(OperationError::Dense)?;
        let uh_view =
            DenseView::new(&stage.u, &uh_shape, &uh_strides, 0).map_err(OperationError::Dense)?;
        dense
            .dot_general_into(
                D::dense_write(output_view),
                D::dense_read(v_view),
                D::dense_read(uh_view),
                &DenseDotConfig::matmul().with_conjugation(true, true),
            )
            .map_err(OperationError::Dense)?;
    }
    BoundDynFactor::from_bound(
        output_space,
        output_data,
        source_space.nin(),
        source_space.nout(),
    )
}

fn compile_pinv_region_routes(
    source: &[CoupledSectorRegion],
    output: &[CoupledSectorRegion],
    source_len: usize,
    output_len: usize,
) -> Result<Vec<InverseSectorRoute>, OperationError> {
    let output_by_sector = sector_region_index_map(output)?;
    let mut used = vec![false; output.len()];
    let mut routes = Vec::with_capacity(source.len());
    for (source_index, source_region) in source.iter().enumerate() {
        let output_index = output_by_sector
            .get(&source_region.coupled())
            .copied()
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: "pinv output is missing a source coupled sector",
            })?;
        let output_region = &output[output_index];
        if output_region.rows() != source_region.cols()
            || output_region.cols() != source_region.rows()
            || source_region.col_trees() != output_region.row_trees()
            || source_region.row_trees() != output_region.col_trees()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "pinv output does not transpose the source coupled-sector layout",
            });
        }
        validate_region_range(source_region, source_len)?;
        validate_region_range(output_region, output_len)?;
        used[output_index] = true;
        routes.push(InverseSectorRoute {
            source: source_index,
            output: output_index,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "pinv output contains a coupled sector absent from the source",
        });
    }
    Ok(routes)
}

#[derive(Clone, Copy)]
struct PolarRegionRoute {
    source: usize,
    w: usize,
    p: usize,
}

fn compile_polar_region_routes(
    source: &[CoupledSectorRegion],
    w: &[CoupledSectorRegion],
    p: &[CoupledSectorRegion],
    source_len: usize,
    w_len: usize,
    p_len: usize,
    direction: PolarDirection,
) -> Result<Vec<PolarRegionRoute>, OperationError> {
    let w_by_sector = SectorRegionIndex::new(w)?;
    let p_by_sector = SectorRegionIndex::new(p)?;
    let mut used_w = vec![false; w.len()];
    let mut used_p = vec![false; p.len()];
    let mut routes = Vec::with_capacity(source.len());
    for (source_index, source_region) in source.iter().enumerate() {
        let sector = source_region.coupled();
        let w_index = sector_region_index_of(&w_by_sector, sector, "polar W")?;
        let p_index = sector_region_index_of(&p_by_sector, sector, "polar P")?;
        let w_region = &w[w_index];
        let p_region = &p[p_index];
        if w_region.rows() != source_region.rows()
            || w_region.cols() != source_region.cols()
            || w_region.row_trees() != source_region.row_trees()
            || w_region.col_trees() != source_region.col_trees()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "polar W route does not preserve the source full-tree layout",
            });
        }
        let (dimension, trees) = match direction {
            PolarDirection::Left => (source_region.cols(), source_region.col_trees()),
            PolarDirection::Right => (source_region.rows(), source_region.row_trees()),
        };
        if p_region.rows() != dimension
            || p_region.cols() != dimension
            || p_region.row_trees() != trees
            || p_region.col_trees() != trees
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "polar P route does not preserve the source full-tree layout",
            });
        }
        validate_region_range(source_region, source_len)?;
        validate_region_range(w_region, w_len)?;
        validate_region_range(p_region, p_len)?;
        used_w[w_index] = true;
        used_p[p_index] = true;
        routes.push(PolarRegionRoute {
            source: source_index,
            w: w_index,
            p: p_index,
        });
    }
    if used_w.iter().any(|used| !used) || used_p.iter().any(|used| !used) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "polar output contains a full-tree route absent from the source",
        });
    }
    Ok(routes)
}

fn validate_checked_polar_direction<R>(
    input: &BoundDynamicTensorRef<'_, R, impl DenseBlockScalar>,
    direction: PolarDirection,
    error_direction: PolarDirection,
) -> Result<(), CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
{
    let space = input.space().space();
    let rows = coupled_sector_block_dimensions_generic_checked(
        space.homspace().codomain(),
        input.space().provider(),
    )?;
    let cols = coupled_sector_block_dimensions_generic_checked(
        space.homspace().domain(),
        input.space().provider(),
    )?;
    for (&sector, &row_count) in &rows {
        if !direction.accepts(row_count, cols.get(&sector).copied().unwrap_or(0)) {
            return Err(CheckedGenericFactorPlanError::Operation(
                error_direction.error(),
            ));
        }
    }
    for (&sector, &col_count) in &cols {
        if !rows.contains_key(&sector) && !direction.accepts(0, col_count) {
            return Err(CheckedGenericFactorPlanError::Operation(
                error_direction.error(),
            ));
        }
    }
    Ok(())
}

fn project_hermitian_col_major<D: FactorScalar>(matrix: &mut [D], n: usize) {
    let half = D::from_real(0.5);
    for col in 0..n {
        for row in 0..=col {
            let value =
                (matrix[row + n * col] + FactorScalar::adjoint(matrix[col + n * row])) * half;
            matrix[row + n * col] = value;
            matrix[col + n * row] = FactorScalar::adjoint(value);
        }
    }
}

fn checked_generic_polar_products<E, D>(
    dense: &mut E,
    stage: &CompactSvdNumericalStage<D>,
    direction: PolarDirection,
) -> Result<(Vec<D>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let CompactSvdNumericalStage {
        rows,
        cols,
        rank,
        u,
        singular_values,
        vt,
    } = stage;
    let mut w = vec![D::zero(); rows * cols];
    let w_shape = [*rows, *cols];
    let w_strides = [1, *rows];
    let u_shape = [*rows, *rank];
    let u_strides = [1, *rows];
    let vt_shape = [*rank, *cols];
    let vt_strides = [1, *rank];
    let w_view =
        DenseViewMut::new(&mut w, &w_shape, &w_strides, 0).map_err(OperationError::Dense)?;
    let u_view = DenseView::new(u, &u_shape, &u_strides, 0).map_err(OperationError::Dense)?;
    let vt_view = DenseView::new(vt, &vt_shape, &vt_strides, 0).map_err(OperationError::Dense)?;
    dense
        .dot_general_into(
            D::dense_write(w_view),
            D::dense_read(u_view),
            D::dense_read(vt_view),
            &DenseDotConfig::matmul(),
        )
        .map_err(OperationError::Dense)?;

    let p_order = match direction {
        PolarDirection::Left => *cols,
        PolarDirection::Right => *rows,
    };
    let mut p = vec![D::zero(); p_order * p_order];
    let mut x = match direction {
        PolarDirection::Left => vt.clone(),
        PolarDirection::Right => u.clone(),
    };
    match direction {
        PolarDirection::Left => {
            for (row, sigma) in singular_values.iter().copied().enumerate() {
                let scale = D::from_real(sigma.sqrt());
                for col in 0..*cols {
                    x[row + rank * col] = x[row + rank * col] * scale;
                }
            }
        }
        PolarDirection::Right => {
            for (col, sigma) in singular_values.iter().copied().enumerate() {
                let scale = D::from_real(sigma.sqrt());
                for row in 0..*rows {
                    x[row + rows * col] = x[row + rows * col] * scale;
                }
            }
        }
    }
    let p_shape = [p_order, p_order];
    let p_strides = [1, p_order];
    let p_view =
        DenseViewMut::new(&mut p, &p_shape, &p_strides, 0).map_err(OperationError::Dense)?;
    match direction {
        PolarDirection::Left => {
            let xh_shape = [*cols, *rank];
            let xh_strides = [*rank, 1];
            let x_shape = [*rank, *cols];
            let x_strides = [1, *rank];
            let xh =
                DenseView::new(&x, &xh_shape, &xh_strides, 0).map_err(OperationError::Dense)?;
            let x = DenseView::new(&x, &x_shape, &x_strides, 0).map_err(OperationError::Dense)?;
            dense
                .dot_general_into(
                    D::dense_write(p_view),
                    D::dense_read(xh),
                    D::dense_read(x),
                    &DenseDotConfig::matmul().with_conjugation(true, false),
                )
                .map_err(OperationError::Dense)?;
        }
        PolarDirection::Right => {
            let x_shape = [*rows, *rank];
            let x_strides = [1, *rows];
            let xh_shape = [*rank, *rows];
            let xh_strides = [*rows, 1];
            let x_view =
                DenseView::new(&x, &x_shape, &x_strides, 0).map_err(OperationError::Dense)?;
            let xh =
                DenseView::new(&x, &xh_shape, &xh_strides, 0).map_err(OperationError::Dense)?;
            dense
                .dot_general_into(
                    D::dense_write(p_view),
                    D::dense_read(x_view),
                    D::dense_read(xh),
                    &DenseDotConfig::matmul().with_conjugation(false, true),
                )
                .map_err(OperationError::Dense)?;
        }
    }
    project_hermitian_col_major(&mut p, p_order);
    Ok((w, p))
}

fn polar_dyn_checked_generic_reported<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    direction: PolarDirection,
    error_direction: PolarDirection,
) -> Result<DynamicFactorPair<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    validate_checked_polar_direction(input, direction, error_direction)?;
    let source_space = input.space().space();
    let w_space = input.space().clone();
    let p_homspace = match direction {
        PolarDirection::Left => FusionTreeHomSpace::new(
            source_space.homspace().domain().clone(),
            source_space.homspace().domain().clone(),
        ),
        PolarDirection::Right => FusionTreeHomSpace::new(
            source_space.homspace().codomain().clone(),
            source_space.homspace().codomain().clone(),
        ),
    };
    let p_nout = p_homspace.codomain().len();
    let p_space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(input.space().provider_arc()),
        p_homspace,
    )?;
    let source_regions =
        canonical_generic_sector_regions(source_space.structure(), source_space.nout())?.ok_or(
            CheckedGenericFactorPlanError::Operation(
                OperationError::UnsupportedTensorContractScope {
                    message: "polar requires canonical coupled-sector input storage",
                },
            ),
        )?;
    let w_regions =
        canonical_generic_sector_regions(w_space.space().structure(), w_space.space().nout())?
            .ok_or(CheckedGenericFactorPlanError::Operation(
                OperationError::UnsupportedTensorContractScope {
                    message: "polar requires canonical coupled-sector W storage",
                },
            ))?;
    let p_regions = canonical_generic_sector_regions(p_space.space().structure(), p_nout)?.ok_or(
        CheckedGenericFactorPlanError::Operation(OperationError::UnsupportedTensorContractScope {
            message: "polar requires canonical coupled-sector P storage",
        }),
    )?;
    let w_len = w_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let p_len = p_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let routes = compile_polar_region_routes(
        &source_regions,
        &w_regions,
        &p_regions,
        input.data().len(),
        w_len,
        p_len,
        direction,
    )?;

    let mut stages = Vec::with_capacity(routes.len());
    for route in &routes {
        let region = &source_regions[route.source];
        stages.push(
            compact_svd_numerical_stage(
                dense,
                &input.data()[region.range()],
                region.rows(),
                region.cols(),
            )
            .map_err(CheckedGenericFactorPlanError::from)?,
        );
    }
    let mut w_data = vec![D::zero(); w_len];
    let mut p_data = vec![D::zero(); p_len];
    for (route, stage) in routes.iter().zip(&stages) {
        let (w, p) = checked_generic_polar_products(dense, stage, direction)
            .map_err(CheckedGenericFactorPlanError::from)?;
        w_data[w_regions[route.w].range()].copy_from_slice(&w);
        p_data[p_regions[route.p].range()].copy_from_slice(&p);
    }

    let w = BoundDynFactor::from_bound(w_space, w_data, source_space.nout(), source_space.nin())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let p = BoundDynFactor::from_bound(p_space, p_data, p_nout, p_nout)
        .map_err(CheckedGenericFactorPlanError::from)?;
    Ok((w, p))
}

#[doc(hidden)]
#[expect(
    clippy::type_complexity,
    reason = "the public checked factorization API exposes its ordered factor tuple directly"
)]
pub fn left_polar_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    polar_dyn_checked_generic_reported(dense, input, PolarDirection::Left, PolarDirection::Left)
}

#[doc(hidden)]
#[expect(
    clippy::type_complexity,
    reason = "the public checked factorization API exposes its ordered factor tuple directly"
)]
pub fn right_polar_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let (w, p) = polar_dyn_checked_generic_reported(
        dense,
        input,
        PolarDirection::Right,
        PolarDirection::Right,
    )?;
    Ok((p, w))
}

#[doc(hidden)]
#[expect(
    clippy::type_complexity,
    reason = "the public checked factorization API exposes its ordered factor tuple directly"
)]
pub fn left_polar_adjoint_parent_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let (w, p) = polar_dyn_checked_generic_reported(
        dense,
        parent,
        PolarDirection::Right,
        PolarDirection::Left,
    )?;
    Ok((p, w))
}

#[doc(hidden)]
#[expect(
    clippy::type_complexity,
    reason = "the public checked factorization API exposes its ordered factor tuple directly"
)]
pub fn right_polar_adjoint_parent_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    polar_dyn_checked_generic_reported(dense, parent, PolarDirection::Left, PolarDirection::Right)
}

pub(crate) fn solve_left_by_sector_dyn<E, R, D>(
    dense: &mut E,
    divisor: &BoundDynamicTensorRef<'_, R, D>,
    rhs: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let divisor_space = divisor.space().space();
    let rhs_space = rhs.space().space();
    let expected = divisor.space().provider().rule_identity();
    let actual = rhs.space().provider().rule_identity();
    if expected != actual {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch { expected, actual },
        ));
    }
    if divisor_space.homspace().codomain() != rhs_space.homspace().codomain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "solve requires equal divisor and right-hand-side codomains",
        });
    }
    if !divisor.space().codomain_isomorphic_to_domain()? {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "solve requires an isomorphic divisor codomain and domain",
        });
    }

    let output_homspace = FusionTreeHomSpace::new(
        divisor_space.homspace().domain().clone(),
        rhs_space.homspace().domain().clone(),
    );
    let output_space = divisor
        .space()
        .derive_from_final_homspace(output_homspace)?;
    solve_left_by_sector_dyn_into(dense, divisor, rhs, output_space)
}

pub(crate) fn solve_left_by_sector_dyn_into<E, R, D>(
    dense: &mut E,
    divisor: &BoundDynamicTensorRef<'_, R, D>,
    rhs: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let divisor_space = divisor.space().space();
    let rhs_space = rhs.space().space();
    let Some(divisor_identity) = divisor_space.admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    let Some(rhs_identity) = rhs_space.admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    let Some(output_identity) = output_space.space().admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    if divisor_identity != rhs_identity {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch {
                expected: divisor_identity.clone(),
                actual: rhs_identity.clone(),
            },
        ));
    }
    if divisor_identity != output_identity {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch {
                expected: divisor_identity.clone(),
                actual: output_identity.clone(),
            },
        ));
    }
    if !Arc::ptr_eq(divisor.space().provider_arc(), output_space.provider_arc()) {
        return Err(OperationError::StructureMismatch {
            tensor: "solve output provider authority",
        });
    }
    if divisor_space.homspace().codomain() != rhs_space.homspace().codomain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "solve requires equal divisor and right-hand-side codomains",
        });
    }
    let expected_homspace = FusionTreeHomSpace::new(
        divisor_space.homspace().domain().clone(),
        rhs_space.homspace().domain().clone(),
    );
    let expected_nout = expected_homspace.codomain().len();
    let expected_nin = expected_homspace.domain().len();
    if output_space.space().nout() != expected_nout
        || output_space.space().nin() != expected_nin
        || output_space.space().rank() != expected_nout + expected_nin
        || output_space.space().homspace() != &expected_homspace
    {
        return Err(OperationError::StructureMismatch {
            tensor: "solve output space",
        });
    }

    let divisor_regions = checked_sector_regions(divisor_space.structure(), divisor_space.nout())?
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: "solve requires canonical coupled-sector divisor storage",
        })?;
    for region in divisor_regions.iter() {
        validate_region_range(region, divisor.data().len())?;
        if region.rows() != region.cols() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "solve coupled-sector divisor matrices must be square",
            });
        }
    }
    let rhs_regions = checked_sector_regions(rhs_space.structure(), rhs_space.nout())?.ok_or(
        OperationError::UnsupportedTensorContractScope {
            message: "solve requires canonical coupled-sector right-hand-side storage",
        },
    )?;
    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "solve derived output requires canonical coupled-sector storage",
    })?;
    let output_len = output_space.space().required_len()?;
    let routes = compile_solve_left_region_routes(
        &divisor_regions,
        &rhs_regions,
        &output_regions,
        divisor.data().len(),
        rhs.data().len(),
        output_len,
    )?;

    let mut output_data = vec![D::zero(); output_len];

    for route in routes {
        let divisor_region = &divisor_regions[route.divisor];
        if divisor_region.rows() == 0 || output_regions[route.output].cols() == 0 {
            continue;
        }
        solve_left_sector(
            dense,
            &divisor.data()[divisor_region.range()],
            &rhs.data()[rhs_regions[route.rhs].range()],
            &mut output_data[output_regions[route.output].range()],
            divisor_region.rows(),
            rhs_regions[route.rhs].cols(),
        )?;
    }

    BoundDynFactor::from_bound(
        output_space,
        output_data,
        divisor_space.nin(),
        rhs_space.nin(),
    )
}

fn compile_solve_left_region_routes(
    divisor: &[CoupledSectorRegion],
    rhs: &[CoupledSectorRegion],
    output: &[CoupledSectorRegion],
    divisor_len: usize,
    rhs_len: usize,
    output_len: usize,
) -> Result<Vec<SolveLeftSectorRoute>, OperationError> {
    let divisor_by_sector = sector_region_index_map(divisor)?;
    let rhs_by_sector = sector_region_index_map(rhs)?;
    let mut routes = Vec::with_capacity(output.len());
    for (output_index, output_region) in output.iter().enumerate() {
        let sector = output_region.coupled();
        let divisor_index = divisor_by_sector.get(&sector).copied().ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: "solve divisor is missing an output coupled sector",
            },
        )?;
        let rhs_index = rhs_by_sector.get(&sector).copied().ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: "solve right-hand side is missing an output coupled sector",
            },
        )?;
        let divisor_region = &divisor[divisor_index];
        let rhs_region = &rhs[rhs_index];
        if divisor_region.rows() != divisor_region.cols()
            || rhs_region.rows() != divisor_region.rows()
            || output_region.rows() != divisor_region.cols()
            || output_region.cols() != rhs_region.cols()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "solve coupled-sector matrix dimensions are incompatible",
            });
        }
        if divisor_region.row_trees() != rhs_region.row_trees()
            || divisor_region.col_trees() != output_region.row_trees()
            || rhs_region.col_trees() != output_region.col_trees()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "solve coupled-sector tree bases are incompatible",
            });
        }
        validate_region_range(divisor_region, divisor_len)?;
        validate_region_range(rhs_region, rhs_len)?;
        validate_region_range(output_region, output_len)?;
        routes.push(SolveLeftSectorRoute {
            divisor: divisor_index,
            rhs: rhs_index,
            output: output_index,
        });
    }
    if rhs_by_sector.len() != routes.len() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "solve right-hand side contains a sector absent from the output",
        });
    }
    Ok(routes)
}

fn solve_left_sector<E, D>(
    dense: &mut E,
    divisor: &[D],
    rhs: &[D],
    output: &mut [D],
    order: usize,
    columns: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let divisor_shape = [order, order];
    let rhs_shape = [order, columns];
    let divisor_strides = [1, order];
    let rhs_strides = [1, order];
    let divisor = DenseView::new(divisor, &divisor_shape, &divisor_strides, 0)
        .map_err(OperationError::Dense)?;
    let rhs = DenseView::new(rhs, &rhs_shape, &rhs_strides, 0).map_err(OperationError::Dense)?;
    let output =
        DenseViewMut::new(output, &rhs_shape, &rhs_strides, 0).map_err(OperationError::Dense)?;
    dense
        .solve_into(
            D::dense_read(divisor),
            D::dense_read(rhs),
            D::dense_write(output),
        )
        .map_err(OperationError::Dense)
}

/// Walks the coupled-sector matricization of an endomorphism and replaces each
/// square block by `apply`'s image of it, writing into a freshly derived
/// canonical layout.
///
/// This is the block-data half of a matrix function that is *not* a spectral
/// function — the caller supplies only the dense `n x n -> n x n` kernel and
/// never sees the layout. `init` is handed the largest block order once, before
/// the loop, so a kernel can size its scratch to `O(max_c n_c²)` and allocate
/// nothing per sector.
///
/// That bound is the kernel's own. It is also the whole of the scratch on the
/// canonical direct-region layout, where the blocks are read in place; the
/// packed fallback below matricizes *every* sector up front and so costs
/// `O(Σ_c n_c²)` on top of it, whatever the kernel does.
///
/// `apply(state, source, n, out, out_leading)` reads the column-major `n x n`
/// block at `source` (leading dimension `n`) and writes its image into `out`
/// with leading dimension `out_leading`.
///
/// Publication is atomic: the result tensor is built only after every sector
/// has succeeded, so a failure in the last block leaves no half-written tensor
/// behind and never mutates the input.
///
/// Why the `inverse_*` route compilers are reused rather than copied: they map
/// source blocks onto the derived output layout under a codomain/domain swap,
/// and on an endomorphism that swap is the identity — the two tree lists are
/// the same list. Duplicating 80 lines to spell the identity differently would
/// only give the two copies a chance to drift.
///
/// # Panics
///
/// Debug-asserts `codomain == domain`. Refusing a non-endomorphism is the
/// caller's job, so that the message names the function the user called
/// (`exp`) rather than whichever helper noticed first.
pub(crate) fn map_square_sectors_dyn<R, D, S, I, F>(
    input: &BoundDynamicTensorRef<'_, R, D>,
    init: I,
    apply: F,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
    I: FnOnce(usize) -> Result<S, OperationError>,
    F: FnMut(&mut S, &[D], usize, &mut [D], usize) -> Result<(), OperationError>,
{
    let source_space = input.space().space();
    debug_assert_eq!(
        source_space.homspace().codomain(),
        source_space.homspace().domain(),
        "callers refuse a non-endomorphism in their own words first"
    );
    let output_space = input
        .space()
        .derive_from_final_homspace(source_space.homspace().clone())?;
    map_square_sectors_dyn_into(input, output_space, init, apply)
}

pub(crate) fn map_square_sectors_dyn_into<R, D, S, I, F>(
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
    init: I,
    mut apply: F,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    D: FactorScalar,
    I: FnOnce(usize) -> Result<S, OperationError>,
    F: FnMut(&mut S, &[D], usize, &mut [D], usize) -> Result<(), OperationError>,
{
    let source_space = input.space().space();
    if source_space.homspace() != output_space.space().homspace() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "matrix function output must preserve the input homspace",
        });
    }

    let source_regions = checked_sector_regions(source_space.structure(), source_space.nout())?;
    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "matrix function output requires canonical coupled-sector storage",
    })?;
    let output_len = output_space.space().required_len()?;
    let output_data = match source_regions {
        Some(source) => {
            let routes = compile_inverse_region_routes(
                &source,
                &output_regions,
                input.data().len(),
                output_len,
            )?;
            let max_order = routes
                .iter()
                .map(|route| source[route.source].rows())
                .max()
                .unwrap_or(0);
            let mut output_data = vec![D::zero(); output_len];
            let mut state = init(max_order)?;
            for route in routes {
                let region = &source[route.source];
                let order = region.rows();
                if order == 0 {
                    continue;
                }
                let output = &output_regions[route.output];
                apply(
                    &mut state,
                    &input.data()[region.range()],
                    order,
                    &mut output_data[output.range()],
                    output.rows(),
                )?;
            }
            output_data
        }
        None => {
            let source_matrices =
                sector_matricizations(source_space.structure(), input.data(), source_space.nout())?;
            let routes =
                compile_inverse_matrix_routes(&source_matrices, &output_regions, output_len)?;
            let max_order = routes
                .iter()
                .map(|route| source_matrices[route.source].rows)
                .max()
                .unwrap_or(0);
            let mut output_data = vec![D::zero(); output_len];
            let mut state = init(max_order)?;
            let mut image = vec![
                D::zero();
                max_order
                    .checked_mul(max_order)
                    .ok_or(OperationError::ElementCountOverflow)?
            ];
            for route in routes {
                let source = &source_matrices[route.source];
                if source.rows == 0 {
                    continue;
                }
                apply(&mut state, &source.data, source.rows, &mut image, max_order)?;
                let output = &output_regions[route.output];
                reorder_inverse_solution(
                    &image,
                    max_order,
                    &mut output_data[output.range()],
                    output.rows(),
                    &route.rows,
                    &route.cols,
                );
            }
            output_data
        }
    };

    BoundDynFactor::from_bound(
        output_space,
        output_data,
        source_space.nout(),
        source_space.nin(),
    )
}

fn compile_inverse_region_routes(
    source: &[CoupledSectorRegion],
    output: &[CoupledSectorRegion],
    source_len: usize,
    output_len: usize,
) -> Result<Vec<InverseSectorRoute>, OperationError> {
    let output_by_sector = sector_region_index_map(output)?;
    let mut used = vec![false; output.len()];
    let mut routes = Vec::with_capacity(source.len());
    for (source_index, source_region) in source.iter().enumerate() {
        let output_index = output_by_sector
            .get(&source_region.coupled())
            .copied()
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: "inverse output is missing a source coupled sector",
            })?;
        validate_inverse_region(source_region, &output[output_index])?;
        validate_region_range(source_region, source_len)?;
        validate_region_range(&output[output_index], output_len)?;
        used[output_index] = true;
        routes.push(InverseSectorRoute {
            source: source_index,
            output: output_index,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse output contains a coupled sector absent from the source",
        });
    }
    Ok(routes)
}

#[cfg(test)]
pub(crate) fn validate_inverse_region_routes_for_test(
    source: &[CoupledSectorRegion],
    output: &[CoupledSectorRegion],
) -> Result<(), OperationError> {
    let source_len = source
        .iter()
        .map(|region| region.range().end)
        .max()
        .unwrap_or(0);
    let output_len = output
        .iter()
        .map(|region| region.range().end)
        .max()
        .unwrap_or(0);
    compile_inverse_region_routes(source, output, source_len, output_len).map(|_| ())
}

fn validate_inverse_region(
    source: &CoupledSectorRegion,
    output: &CoupledSectorRegion,
) -> Result<(), OperationError> {
    if source.rows() != source.cols()
        || output.rows() != source.cols()
        || output.cols() != source.rows()
    {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse coupled-sector matrix is not square",
        });
    }
    if source.col_trees() != output.row_trees() || source.row_trees() != output.col_trees() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse output tree basis does not transpose the source basis",
        });
    }
    Ok(())
}

fn validate_region_range(
    region: &CoupledSectorRegion,
    data_len: usize,
) -> Result<(), OperationError> {
    let range = region.range();
    if range.end > data_len {
        return Err(OperationError::ElementCountMismatch {
            expected: range.end,
            actual: data_len,
        });
    }
    Ok(())
}

fn compile_inverse_matrix_routes<D>(
    source: &[SectorMatricization<D>],
    output: &[CoupledSectorRegion],
    output_len: usize,
) -> Result<Vec<InverseMatrixRoute>, OperationError> {
    let output_by_sector = sector_region_index_map(output)?;
    let mut used = vec![false; output.len()];
    let mut routes = Vec::with_capacity(source.len());
    for (source_index, source_matrix) in source.iter().enumerate() {
        let output_index = output_by_sector.get(&source_matrix.sector).copied().ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: "inverse output is missing a source coupled sector",
            },
        )?;
        let output_region = &output[output_index];
        if source_matrix.rows != source_matrix.cols
            || output_region.rows() != source_matrix.cols
            || output_region.cols() != source_matrix.rows
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "inverse coupled-sector matrix is not square",
            });
        }
        validate_region_range(output_region, output_len)?;
        let rows =
            compile_inverse_basis_extents(&source_matrix.col_trees, output_region.row_trees())?;
        let cols =
            compile_inverse_basis_extents(&source_matrix.row_trees, output_region.col_trees())?;
        used[output_index] = true;
        routes.push(InverseMatrixRoute {
            source: source_index,
            output: output_index,
            rows,
            cols,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse output contains a coupled sector absent from the source",
        });
    }
    Ok(routes)
}

fn compile_inverse_basis_extents(
    source: &[(FusionTreeKey, usize, Vec<usize>)],
    output: &[CoupledTreeExtent],
) -> Result<Vec<InverseBasisExtent>, OperationError> {
    let output_by_tree = output
        .iter()
        .enumerate()
        .map(|(index, extent)| (extent.tree(), index))
        .collect::<FxHashMap<_, _>>();
    if output_by_tree.len() != output.len() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse output contains a duplicate tree basis",
        });
    }
    let mut used = vec![false; output.len()];
    let mut extents = Vec::with_capacity(source.len());
    for (tree, source_offset, source_shape) in source {
        let output_index = output_by_tree.get(tree).copied().ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: "inverse output is missing a source tree basis",
            },
        )?;
        let output_extent = &output[output_index];
        if source_shape != output_extent.shape() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "inverse output tree basis has an unexpected shape",
            });
        }
        let extent = output_extent
            .extent()
            .map_err(OperationError::from_core_preserving_context)?;
        used[output_index] = true;
        extents.push(InverseBasisExtent {
            source_offset: *source_offset,
            output_offset: output_extent.offset(),
            extent,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse output contains a tree basis absent from the source",
        });
    }
    Ok(extents)
}

fn identity_workspace<D: FactorScalar>(order: usize) -> Result<Vec<D>, OperationError> {
    let elements = order
        .checked_mul(order)
        .ok_or(OperationError::ElementCountOverflow)?;
    let mut identity = vec![D::zero(); elements];
    for index in 0..order {
        identity[index + order * index] = D::from_real(1.0);
    }
    Ok(identity)
}

fn solve_inverse_sector<E, D>(
    dense: &mut E,
    source: &[D],
    output: &mut [D],
    order: usize,
    output_leading: usize,
    identity: &[D],
    identity_order: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let shape = [order, order];
    let matrix_strides = [1, order];
    let output_strides = [1, output_leading];
    let identity_strides = [1, identity_order];
    let source =
        DenseView::new(source, &shape, &matrix_strides, 0).map_err(OperationError::Dense)?;
    let identity =
        DenseView::new(identity, &shape, &identity_strides, 0).map_err(OperationError::Dense)?;
    let output =
        DenseViewMut::new(output, &shape, &output_strides, 0).map_err(OperationError::Dense)?;
    dense
        .solve_into(
            D::dense_read(source),
            D::dense_read(identity),
            D::dense_write(output),
        )
        .map_err(OperationError::Dense)
}

fn reorder_inverse_solution<D: Copy>(
    source: &[D],
    source_rows: usize,
    output: &mut [D],
    output_rows: usize,
    row_extents: &[InverseBasisExtent],
    col_extents: &[InverseBasisExtent],
) {
    for rows in row_extents {
        for cols in col_extents {
            for col in 0..cols.extent {
                let source_start = rows.source_offset + source_rows * (cols.source_offset + col);
                let output_start = rows.output_offset + output_rows * (cols.output_offset + col);
                output[output_start..output_start + rows.extent]
                    .copy_from_slice(&source[source_start..source_start + rows.extent]);
            }
        }
    }
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

fn coupled_of_generic(tree: &FusionTreeKey) -> SectorId {
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
fn sector_matricizations_generic<D>(
    structure: &BlockStructure,
    data: &[D],
    nout: usize,
) -> Result<Vec<SectorMatricization<D>>, OperationError>
where
    D: FactorScalar,
{
    #[derive(Clone, Copy, Default)]
    struct TreePlacement {
        row_offset: Option<usize>,
        col_offset: Option<usize>,
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
    Ok(matricizations)
}

fn validate_endomorphism_tree_stacking<D>(
    matricizations: &[SectorMatricization<D>],
) -> Result<(), OperationError> {
    for matrix in matricizations {
        if matrix.rows != matrix.cols
            || matrix.row_trees.len() != matrix.col_trees.len()
            || !matrix
                .row_trees
                .iter()
                .zip(&matrix.col_trees)
                .all(|(row, column)| row == column)
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "eigh requires identical endomorphism row/column fusion-tree stacking",
            });
        }
    }
    Ok(())
}

#[cfg(test)]
fn record_generic_pair_ordered_key_validation() {
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.ordered_key_validation_events += 1;
        probe.set(value);
    });
}

#[cfg(not(test))]
fn record_generic_pair_ordered_key_validation() {}

struct FactorTreeCursor<'a, M> {
    matricizations: &'a [M],
    ranks: &'a [SectorRank],
    matrix: usize,
    tree: usize,
    valid: bool,
}

impl<'a, M: SectorGeometry> FactorTreeCursor<'a, M> {
    fn new(matricizations: &'a [M], ranks: &'a [SectorRank]) -> Self {
        let sectors_are_canonical = matricizations
            .windows(2)
            .all(|pair| pair[0].sector() < pair[1].sector());
        Self {
            matricizations,
            ranks,
            matrix: 0,
            tree: 0,
            valid: matricizations.len() == ranks.len() && sectors_are_canonical,
        }
    }

    fn next(&mut self, side: FactorSide) -> Option<(SectorId, &'a FusionTreeKey)> {
        while let (Some(matrix), Some(rank)) = (
            self.matricizations.get(self.matrix),
            self.ranks.get(self.matrix),
        ) {
            self.valid &= matrix.sector() == rank.sector;
            if rank.kept != 0 {
                if let Some(tree) = matrix.tree(side, self.tree) {
                    self.tree += 1;
                    return Some((matrix.sector(), tree.tree));
                }
            }
            self.matrix += 1;
            self.tree = 0;
        }
        None
    }

    fn matches(&mut self, side: FactorSide, key: &FusionTreeKey) -> bool {
        record_generic_pair_ordered_key_validation();
        if !self.valid {
            return false;
        }
        self.next(side)
            .is_some_and(|(sector, tree)| coupled_of_generic(key) == sector && key == tree)
    }

    fn is_exhausted(&mut self, side: FactorSide) -> bool {
        self.next(side).is_none() && self.valid
    }
}

#[cfg(test)]
fn record_generic_pair_fallback_lookup(side: FactorSide) {
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        match side {
            FactorSide::Left => value.fallback_row_lookups += 1,
            FactorSide::Right => value.fallback_col_lookups += 1,
        }
        probe.set(value);
    });
}

#[cfg(not(test))]
fn record_generic_pair_fallback_lookup(_side: FactorSide) {}

/// Staged keys of a prepared checked layout in enumeration order. The
/// prepared structure lists its blocks exactly as the enumeration produced
/// its keys and `commit` interns them without reordering, so this sequence
/// is the key order of the committed space; the checked builders validate
/// and commit from one enumeration instead of enumerating keys separately.
fn staged_fusion_tree_keys(sector: &SectorStructure) -> impl Iterator<Item = &FusionTreePairKey> {
    sector
        .blocks()
        .iter()
        .filter_map(|block| match block.key() {
            BlockKey::FusionTree(key) => Some(key),
            _ => None,
        })
}

fn validate_generic_factor_keys<'a, 'k, M: SectorGeometry>(
    keys: impl IntoIterator<Item = &'k FusionTreePairKey>,
    side: FactorSide,
    mut cursor: Option<&mut FactorTreeCursor<'_, M>>,
    matricizations: &'a [M],
    matrix_by_sector: &mut Option<FxHashMap<SectorId, &'a M>>,
) -> Result<bool, OperationError> {
    let mut ordered = true;
    let mut index: Option<PlacementIndex<'a>> = None;
    for key in keys {
        // Why not build the final layout first: missing source placements must
        // fail before output construction. The cursor only skips the old
        // lookup for the unique, sector-ordered sequence produced by the
        // canonical matricization path.
        let tree = match side {
            FactorSide::Left => key.codomain_tree(),
            FactorSide::Right => key.domain_tree(),
        };
        let aligned = cursor
            .as_deref_mut()
            .map(|cursor| cursor.matches(side, tree));
        if aligned != Some(true) {
            record_generic_pair_fallback_lookup(side);
            let matrix_by_sector =
                matrix_by_sector.get_or_insert_with(|| matricization_map(matricizations));
            let sector = coupled_of_generic(tree);
            matricization_of(matrix_by_sector, sector)?;
            index
                .get_or_insert_with(|| PlacementIndex::new(matricizations, &[side]))
                .placement(sector, side, tree)?;
        }
        if let Some(aligned) = aligned {
            ordered &= aligned;
        }
    }
    if let Some(cursor) = cursor {
        ordered &= cursor.is_exhausted(side);
    }
    Ok(ordered)
}

fn checked_extent(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |extent, &dim| extent.checked_mul(dim))
}

/// `keys` carries the separately enumerated key list of the unchecked paired
/// route for the admitted-key equality against `structure`; the checked route
/// commits the very structure it validated and passes `None`.
fn factor_output_is_canonical<D, M: SectorGeometry>(
    structure: &BlockStructure,
    keys: Option<&[FusionTreePairKey]>,
    matricizations: &[M],
    pairs: &[FactorPair<D>],
    required_len: usize,
    side: FactorSide,
) -> bool {
    if matricizations.len() != pairs.len()
        || keys.is_some_and(|keys| keys.len() != structure.block_count())
    {
        return false;
    }
    let mut block_index = 0usize;
    let mut output_offset = 0usize;
    for (matrix, pair) in matricizations.iter().zip(pairs) {
        if pair.sector != matrix.sector() {
            return false;
        }
        let (factor_len, factor_leading) = match side {
            FactorSide::Left => (pair.left.len(), pair.left_rows),
            FactorSide::Right => (pair.right.len(), pair.right_leading),
        };
        let Some(next_offset) = sector_factor_output_is_canonical(
            structure,
            &mut block_index,
            output_offset,
            matrix,
            side,
            factor_len,
            factor_leading,
            pair.kept,
            side,
            keys,
        ) else {
            return false;
        };
        output_offset = next_offset;
    }
    block_index == structure.block_count() && output_offset == required_len
}

/// Proves that one sector's selected factor (`factor_len` elements, column
/// major with leading dimension `factor_leading`, bond extent `bond`) already
/// occupies the blocks of `structure` starting at `block_index`/`output_offset`
/// in the exact layout `from_bound` expects, and returns the next output
/// offset. Left output reads the source as `F[o + q + a*j]`, right output as
/// `F[j + b*(o + q)]`; `a` is the source-side extent on `source_trees`.
/// `expected_keys` keeps the paired callers' additional admitted-key equality
/// and their probe events; one-sided callers pass `None`.
#[allow(clippy::too_many_arguments)]
fn sector_factor_output_is_canonical<M: SectorGeometry>(
    structure: &BlockStructure,
    block_index: &mut usize,
    output_offset: usize,
    matrix: &M,
    source_trees: FactorSide,
    factor_len: usize,
    factor_leading: usize,
    bond: usize,
    side: FactorSide,
    expected_keys: Option<&[FusionTreePairKey]>,
) -> Option<usize> {
    let source_extent = match source_trees {
        FactorSide::Left => matrix.rows(),
        FactorSide::Right => matrix.cols(),
    };
    let required_leading = match side {
        FactorSide::Left => source_extent,
        FactorSide::Right => bond,
    };
    if factor_leading != required_leading || factor_len != source_extent.checked_mul(bond)? {
        return None;
    }
    let mut tree_prefix = 0usize;
    for tree_index in 0..matrix.tree_count(source_trees) {
        let tree = matrix.tree(source_trees, tree_index)?;
        let extent = checked_extent(tree.shape)?;
        if tree.offset != tree_prefix {
            return None;
        }
        tree_prefix = tree_prefix.checked_add(extent)?;
        if bond == 0 {
            continue;
        }
        let block = structure.block(*block_index).ok()?;
        let BlockKey::FusionTree(actual_key) = block.key() else {
            return None;
        };
        if let Some(keys) = expected_keys {
            record_generic_pair_output_block_visit();
            record_generic_pair_ordered_key_validation();
            if actual_key != keys.get(*block_index)? {
                return None;
            }
        }
        let actual_tree = match side {
            FactorSide::Left => actual_key.codomain_tree(),
            FactorSide::Right => actual_key.domain_tree(),
        };
        if actual_tree != tree.tree || coupled_of(actual_tree) != matrix.sector() {
            return None;
        }
        let block_offset = match side {
            FactorSide::Left => output_offset.checked_add(tree.offset)?,
            FactorSide::Right => output_offset.checked_add(bond.checked_mul(tree.offset)?)?,
        };
        if block.offset() != block_offset
            || block.shape().len() != tree.shape.len() + 1
            || block.strides().len() != block.shape().len()
        {
            return None;
        }
        let shape_matches = match side {
            FactorSide::Left => {
                &block.shape()[..tree.shape.len()] == tree.shape
                    && block.shape()[tree.shape.len()] == bond
            }
            FactorSide::Right => block.shape()[0] == bond && &block.shape()[1..] == tree.shape,
        };
        if !shape_matches {
            return None;
        }
        let mut stride = match side {
            FactorSide::Left => 1,
            FactorSide::Right => bond,
        };
        let stride_offset = usize::from(matches!(side, FactorSide::Right));
        if matches!(side, FactorSide::Right) && block.strides()[0] != 1 {
            return None;
        }
        for (axis, &dim) in tree.shape.iter().enumerate() {
            if block.strides()[axis + stride_offset] != stride {
                return None;
            }
            stride = stride.checked_mul(dim)?;
        }
        if matches!(side, FactorSide::Left) && block.strides()[tree.shape.len()] != source_extent {
            return None;
        }
        *block_index += 1;
    }
    if tree_prefix != source_extent {
        return None;
    }
    output_offset.checked_add(factor_len)
}

/// One-sided sibling of [`factor_output_is_canonical`]: the source
/// matricizations, factor pairs and admitted bond dimensions must be one
/// aligned ascending sector stream (which excludes duplicate, omitted, and
/// identity-only sectors), and the whole admitted output must be exhausted by
/// the selected factors. `source_trees` selects the matricization side whose
/// trees index the selected factor (columns for adjoint placement).
fn one_sided_factor_output_is_canonical<D, M: SectorGeometry>(
    structure: &BlockStructure,
    matricizations: &[M],
    pairs: &[FactorPair<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    required_len: usize,
    side: FactorSide,
    source_trees: FactorSide,
) -> bool {
    if matricizations.len() != pairs.len() || matricizations.len() != dimensions.len() {
        return false;
    }
    let mut block_index = 0usize;
    let mut output_offset = 0usize;
    for ((matrix, pair), (&sector, &bond)) in matricizations.iter().zip(pairs).zip(dimensions) {
        if matrix.sector() != sector || pair.sector != sector {
            return false;
        }
        let (factor_len, factor_leading) = match side {
            FactorSide::Left => (pair.left.len(), pair.left_rows),
            FactorSide::Right => (pair.right.len(), pair.right_leading),
        };
        let Some(next_offset) = sector_factor_output_is_canonical(
            structure,
            &mut block_index,
            output_offset,
            matrix,
            source_trees,
            factor_len,
            factor_leading,
            bond,
            side,
            None,
        ) else {
            return false;
        };
        output_offset = next_offset;
    }
    block_index == structure.block_count() && output_offset == required_len
}

/// Moves the selected side of every pair into one output buffer after
/// [`one_sided_factor_output_is_canonical`] proved the layout; the opposite
/// side stays in place for its own publication.
fn take_one_sided_factors<D>(
    pairs: &mut [FactorPair<D>],
    required_len: usize,
    side: FactorSide,
) -> Vec<D> {
    #[cfg(test)]
    let first = pairs
        .iter()
        .map(|pair| match side {
            FactorSide::Left => &pair.left,
            FactorSide::Right => &pair.right,
        })
        .find(|factor| !factor.is_empty())
        .map(Vec::as_ptr);
    let mut output: Option<Vec<D>> = None;
    let _appended = pairs
        .iter_mut()
        .map(|pair| {
            let factor = match side {
                FactorSide::Left => std::mem::take(&mut pair.left),
                FactorSide::Right => std::mem::take(&mut pair.right),
            };
            append_owned_factor(&mut output, factor, required_len)
        })
        .sum::<usize>();
    let data = output.unwrap_or_default();
    #[cfg(test)]
    ONE_SIDED_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.canonical_publications += 1;
        value.owner_reused +=
            usize::from(first.is_some_and(|pointer| std::ptr::eq(pointer, data.as_ptr())));
        value.appended_elements += _appended;
        probe.set(value);
    });
    data
}

#[cfg(test)]
fn record_one_sided_fallback_publication() {
    ONE_SIDED_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.fallback_publications += 1;
        probe.set(value);
    });
}

#[cfg(not(test))]
fn record_one_sided_fallback_publication() {}

#[cfg(test)]
fn record_generic_pair_output_block_visit() {
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.output_blocks_visited += 1;
        probe.set(value);
    });
}

#[cfg(not(test))]
fn record_generic_pair_output_block_visit() {}

#[cfg(test)]
fn record_generic_pair_appended(left: usize, right: usize) {
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.left_appended_elements += left;
        value.right_appended_elements += right;
        probe.set(value);
    });
}

#[cfg(not(test))]
fn record_generic_pair_appended(_left: usize, _right: usize) {}

/// Appends `factor` to `output`, letting the first nonempty factor keep its
/// allocation (grown once to `required_len`). Returns the element count that
/// was appended to an existing owner, zero otherwise.
fn append_owned_factor<D>(
    output: &mut Option<Vec<D>>,
    factor: Vec<D>,
    required_len: usize,
) -> usize {
    if factor.is_empty() {
        return 0;
    }
    if let Some(data) = output {
        let appended = factor.len();
        data.extend(factor);
        appended
    } else {
        let mut factor = factor;
        factor.reserve_exact(required_len - factor.len());
        *output = Some(factor);
        0
    }
}

fn publish_generic_factor_pairs<D>(
    pairs: Vec<FactorPair<D>>,
    left_len: usize,
    right_len: usize,
) -> (Vec<D>, Vec<D>) {
    #[cfg(test)]
    let first_left = pairs
        .iter()
        .find(|pair| !pair.left.is_empty())
        .map(|pair| pair.left.as_ptr());
    #[cfg(test)]
    let first_right = pairs
        .iter()
        .find(|pair| !pair.right.is_empty())
        .map(|pair| pair.right.as_ptr());
    let mut left_data: Option<Vec<D>> = None;
    let mut right_data: Option<Vec<D>> = None;
    for pair in pairs {
        let left_appended = append_owned_factor(&mut left_data, pair.left, left_len);
        let right_appended = append_owned_factor(&mut right_data, pair.right, right_len);
        record_generic_pair_appended(left_appended, right_appended);
    }
    let left_data = left_data.unwrap_or_default();
    let right_data = right_data.unwrap_or_default();
    #[cfg(test)]
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.left_owner_reused += usize::from(
            first_left.is_some_and(|pointer| std::ptr::eq(pointer, left_data.as_ptr())),
        );
        value.right_owner_reused += usize::from(
            first_right.is_some_and(|pointer| std::ptr::eq(pointer, right_data.as_ptr())),
        );
        probe.set(value);
    });
    (left_data, right_data)
}

/// Builds provider-bound left and right factor spaces for a generic rule.
fn build_left_right_bound_spaces_generic<R, M>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    ranks: &[SectorRank],
) -> Result<(BoundDynamicFusionMapSpace<R>, BoundDynamicFusionMapSpace<R>), OperationError>
where
    R: FusionRule,
    M: SectorGeometry,
{
    let new_leg = SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false);
    let mut matrix_by_sector = None;
    let (left, _, _) = build_left_bound_space_generic(
        provider,
        homspace,
        matricizations,
        new_leg.clone(),
        None,
        &mut matrix_by_sector,
    )?;
    let (right, _, _) = build_right_bound_space_generic(
        provider,
        homspace,
        matricizations,
        new_leg,
        None,
        &mut matrix_by_sector,
    )?;
    Ok((left, right))
}

fn build_left_right_bound_spaces_and_keys_generic<R, M>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    ranks: &[SectorRank],
) -> Result<GenericFactorPairSpaces<R>, OperationError>
where
    R: FusionRule,
    M: SectorGeometry,
{
    let new_leg = SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false);
    let mut matrix_by_sector = None;
    let mut left_cursor = FactorTreeCursor::new(matricizations, ranks);
    let (left, left_keys, left_ordered) = build_left_bound_space_generic(
        provider,
        homspace,
        matricizations,
        new_leg.clone(),
        Some(&mut left_cursor),
        &mut matrix_by_sector,
    )?;
    let mut right_cursor = FactorTreeCursor::new(matricizations, ranks);
    let (right, right_keys, right_ordered) = build_right_bound_space_generic(
        provider,
        homspace,
        matricizations,
        new_leg,
        Some(&mut right_cursor),
        &mut matrix_by_sector,
    )?;
    Ok(GenericFactorPairSpaces {
        left,
        right,
        left_keys,
        right_keys,
        ordered: left_ordered && right_ordered,
    })
}

fn build_left_bound_space_generic<'a, R, M>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &'a [M],
    new_leg: SectorLeg,
    cursor: Option<&mut FactorTreeCursor<'_, M>>,
    matrix_by_sector: &mut Option<FxHashMap<SectorId, &'a M>>,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<FusionTreePairKey>, bool), OperationError>
where
    R: FusionRule,
    M: SectorGeometry,
{
    let left_hom = FusionTreeHomSpace::new(
        homspace.codomain().clone(),
        FusionProductSpace::new([new_leg]),
    );
    let left_keys = left_hom
        .fusion_tree_keys_generic(provider.as_ref())
        .map_err(OperationError::from_core_preserving_context)?;
    let ordered = validate_generic_factor_keys(
        &left_keys,
        FactorSide::Left,
        cursor,
        matricizations,
        matrix_by_sector,
    )?;
    let left =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::clone(provider), left_hom)?;
    Ok((left, left_keys, ordered))
}

fn build_right_bound_space_generic<'a, R, M>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &'a [M],
    new_leg: SectorLeg,
    cursor: Option<&mut FactorTreeCursor<'_, M>>,
    matrix_by_sector: &mut Option<FxHashMap<SectorId, &'a M>>,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<FusionTreePairKey>, bool), OperationError>
where
    R: FusionRule,
    M: SectorGeometry,
{
    let right_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg]),
        homspace.domain().clone(),
    );
    let right_keys = right_hom
        .fusion_tree_keys_generic(provider.as_ref())
        .map_err(OperationError::from_core_preserving_context)?;
    let ordered = validate_generic_factor_keys(
        &right_keys,
        FactorSide::Right,
        cursor,
        matricizations,
        matrix_by_sector,
    )?;
    let right =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::clone(provider), right_hom)?;
    Ok((right, right_keys, ordered))
}

fn build_left_right_bound_pair_generic<R, D, M>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: Vec<FactorPair<D>>,
) -> Result<DynamicFactorPair<R, D>, OperationError>
where
    R: FusionRule,
    D: FactorScalar,
    M: SectorGeometry,
{
    let ranks = pairs
        .iter()
        .map(|pair| SectorRank {
            sector: pair.sector,
            kept: pair.kept,
        })
        .collect::<Vec<_>>();
    let spaces =
        build_left_right_bound_spaces_and_keys_generic(provider, homspace, matricizations, &ranks)?;
    let left_len = spaces.left.space().required_len()?;
    let right_len = spaces.right.space().required_len()?;
    let canonical = spaces.ordered
        && factor_output_is_canonical(
            spaces.left.space().structure(),
            Some(&spaces.left_keys),
            matricizations,
            &pairs,
            left_len,
            FactorSide::Left,
        )
        && factor_output_is_canonical(
            spaces.right.space().structure(),
            Some(&spaces.right_keys),
            matricizations,
            &pairs,
            right_len,
            FactorSide::Right,
        );
    let (left_data, right_data) = if canonical {
        #[cfg(test)]
        GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
            let mut value = probe.get();
            value.canonical_publications += 1;
            probe.set(value);
        });
        publish_generic_factor_pairs(pairs, left_len, right_len)
    } else {
        #[cfg(test)]
        GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
            let mut value = probe.get();
            value.fallback_publications += 1;
            probe.set(value);
        });
        let mut left_data = vec![D::zero(); left_len];
        let mut right_data = vec![D::zero(); right_len];
        let index = PlacementIndex::new(matricizations, &[FactorSide::Left, FactorSide::Right]);
        let left_groups =
            SectorBlockGroups::new(spaces.left.space().structure(), FactorSide::Left)?;
        let right_groups =
            SectorBlockGroups::new(spaces.right.space().structure(), FactorSide::Right)?;
        for (matrix, pair) in matricizations.iter().zip(&pairs) {
            scatter_left_sector_blocks_generic(
                spaces.left.space(),
                &mut left_data,
                matrix,
                &index,
                &left_groups,
                &pair.left,
                pair.left_rows,
            )?;
            scatter_right_sector_blocks_generic(
                spaces.right.space(),
                &mut right_data,
                matrix,
                &index,
                &right_groups,
                &pair.right,
                pair.right_leading,
            )?;
        }
        (left_data, right_data)
    };
    let left_nout = spaces.left.space().nout();
    let right_nin = spaces.right.space().nin();
    Ok((
        BoundDynFactor::from_bound(spaces.left, left_data, left_nout, 1)?,
        BoundDynFactor::from_bound(spaces.right, right_data, 1, right_nin)?,
    ))
}

fn build_left_right_bound_pair_generic_checked<R, D, M>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: Vec<FactorPair<D>>,
) -> Result<DynamicFactorPair<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
    M: SectorGeometry,
{
    let ranks = pairs
        .iter()
        .map(|pair| SectorRank {
            sector: pair.sector,
            kept: pair.kept,
        })
        .collect::<Vec<_>>();
    let mut matrix_by_sector = None;
    let new_leg = SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false);
    // One provider-backed enumeration per side serves both the ordered key
    // validation and the committed space; the left side completes before the
    // right side starts so validation precedence is unchanged.
    let left_hom = FusionTreeHomSpace::new(
        homspace.codomain().clone(),
        FusionProductSpace::new([new_leg.clone()]),
    );
    let left_prepared = left_hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(provider.as_ref())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut left_cursor = FactorTreeCursor::new(matricizations, &ranks);
    let left_ordered = validate_generic_factor_keys(
        staged_fusion_tree_keys(left_prepared.sector_structure()),
        FactorSide::Left,
        Some(&mut left_cursor),
        matricizations,
        &mut matrix_by_sector,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let left = BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(provider),
        left_hom,
        left_prepared,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let right_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg]),
        homspace.domain().clone(),
    );
    let right_prepared = right_hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(provider.as_ref())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut right_cursor = FactorTreeCursor::new(matricizations, &ranks);
    let right_ordered = validate_generic_factor_keys(
        staged_fusion_tree_keys(right_prepared.sector_structure()),
        FactorSide::Right,
        Some(&mut right_cursor),
        matricizations,
        &mut matrix_by_sector,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let right = BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(provider),
        right_hom,
        right_prepared,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let left_len = left.space().required_len().map_err(|e| {
        CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(e))
    })?;
    let right_len = right.space().required_len().map_err(|e| {
        CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(e))
    })?;
    let canonical = left_ordered
        && right_ordered
        && factor_output_is_canonical(
            left.space().structure(),
            None,
            matricizations,
            &pairs,
            left_len,
            FactorSide::Left,
        )
        && factor_output_is_canonical(
            right.space().structure(),
            None,
            matricizations,
            &pairs,
            right_len,
            FactorSide::Right,
        );
    let (left_data, right_data) = if canonical {
        #[cfg(test)]
        GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
            let mut value = probe.get();
            value.canonical_publications += 1;
            probe.set(value);
        });
        publish_generic_factor_pairs(pairs, left_len, right_len)
    } else {
        #[cfg(test)]
        GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
            let mut value = probe.get();
            value.fallback_publications += 1;
            probe.set(value);
        });
        let mut left_data = vec![D::zero(); left_len];
        let mut right_data = vec![D::zero(); right_len];
        let index = PlacementIndex::new(matricizations, &[FactorSide::Left, FactorSide::Right]);
        let left_groups = SectorBlockGroups::new(left.space().structure(), FactorSide::Left)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let right_groups = SectorBlockGroups::new(right.space().structure(), FactorSide::Right)
            .map_err(CheckedGenericFactorPlanError::from)?;
        for (matrix, pair) in matricizations.iter().zip(&pairs) {
            scatter_left_sector_blocks_generic(
                left.space(),
                &mut left_data,
                matrix,
                &index,
                &left_groups,
                &pair.left,
                pair.left_rows,
            )
            .map_err(CheckedGenericFactorPlanError::from)?;
            scatter_right_sector_blocks_generic(
                right.space(),
                &mut right_data,
                matrix,
                &index,
                &right_groups,
                &pair.right,
                pair.right_leading,
            )
            .map_err(CheckedGenericFactorPlanError::from)?;
        }
        (left_data, right_data)
    };
    let left_nout = left.space().nout();
    let right_nin = right.space().nin();
    Ok((
        BoundDynFactor::from_bound(left, left_data, left_nout, 1)
            .map_err(CheckedGenericFactorPlanError::from)?,
        BoundDynFactor::from_bound(right, right_data, 1, right_nin)
            .map_err(CheckedGenericFactorPlanError::from)?,
    ))
}

fn build_checked_pair_from_input<R, D>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &InputMatricizations<'_, D>,
    pairs: Vec<FactorPair<D>>,
) -> Result<DynamicFactorPair<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    match matricizations {
        InputMatricizations::Regions { regions, .. } => {
            build_left_right_bound_pair_generic_checked(provider, homspace, regions.as_ref(), pairs)
        }
        InputMatricizations::Packed(matrices) => {
            build_left_right_bound_pair_generic_checked(provider, homspace, matrices, pairs)
        }
    }
}

/// Checked generic factor materialization with identity completion for sectors
/// absent from the source matricization (the full-factor contract).
fn build_bound_factor_generic_checked<R, D, M>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: &mut [FactorPair<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
    M: SectorGeometry,
{
    let bond = SectorLeg::new(
        dimensions.iter().map(|(&sector, &dim)| (sector, dim)),
        false,
    );
    let output_hom = match side {
        FactorSide::Left => {
            FusionTreeHomSpace::new(homspace.codomain().clone(), FusionProductSpace::new([bond]))
        }
        FactorSide::Right => {
            FusionTreeHomSpace::new(FusionProductSpace::new([bond]), homspace.domain().clone())
        }
    };
    let matrices = matricizations
        .iter()
        .map(|matrix| (matrix.sector(), matrix))
        .collect::<FxHashMap<_, _>>();
    // One provider-backed enumeration serves both the populated-key
    // prevalidation and the committed space; the prepared structure lists its
    // blocks in the enumeration's key order, so the first-match placement
    // semantics are those of the key sequence.
    let prepared = output_hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(provider.as_ref())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let placements = PlacementIndex::new(matricizations, &[side]);
    for block in prepared.sector_structure().blocks() {
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = match side {
            FactorSide::Left => coupled_of_generic(key.codomain_tree()),
            FactorSide::Right => coupled_of_generic(key.domain_tree()),
        };
        if matrices.contains_key(&sector) {
            let tree = match side {
                FactorSide::Left => key.codomain_tree(),
                FactorSide::Right => key.domain_tree(),
            };
            placements.placement(sector, side, tree)?;
        }
    }
    let space = BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(provider),
        output_hom,
        prepared,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let len = space.space().required_len().map_err(|e| {
        CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(e))
    })?;
    let (nout, nin) = match side {
        FactorSide::Left => (space.space().nout(), 1),
        FactorSide::Right => (1, space.space().nin()),
    };
    if one_sided_factor_output_is_canonical(
        space.space().structure(),
        matricizations,
        pairs,
        dimensions,
        len,
        side,
        side,
    ) {
        let data = take_one_sided_factors(pairs, len, side);
        return BoundDynFactor::from_bound(space, data, nout, nin)
            .map_err(CheckedGenericFactorPlanError::from);
    }
    record_one_sided_fallback_publication();
    let mut data = vec![D::zero(); len];
    let pairs = pairs
        .iter()
        .map(|pair| (pair.sector, pair))
        .collect::<FxHashMap<_, _>>();
    let mut missing_offsets = FxHashMap::<SectorId, usize>::default();
    let structure = Arc::clone(space.space().structure());
    for index in 0..structure.block_count() {
        let block = structure.block(index).map_err(|e| {
            CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(
                e,
            ))
        })?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let (sector, axis) = match side {
            FactorSide::Left => (
                coupled_of_generic(key.codomain_tree()),
                block.shape().len() - 1,
            ),
            FactorSide::Right => (coupled_of_generic(key.domain_tree()), 0),
        };
        if matrices.contains_key(&sector) {
            let pair = pairs
                .get(&sector)
                .ok_or(CheckedGenericFactorPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope {
                        message: "factor rank absent for a populated source sector",
                    },
                ))?;
            let tree = match side {
                FactorSide::Left => key.codomain_tree(),
                FactorSide::Right => key.domain_tree(),
            };
            let offset = placements.placement(sector, side, tree)?.0;
            let (factor, factor_rows) = match side {
                FactorSide::Left => (&pair.left, pair.left_rows),
                FactorSide::Right => (&pair.right, pair.right_leading),
            };
            scatter_matrix_block(
                &mut data,
                block.shape(),
                block.strides(),
                block.offset(),
                axis,
                side,
                factor,
                factor_rows,
                offset,
            );
            continue;
        }
        let dimension =
            *dimensions
                .get(&sector)
                .ok_or(CheckedGenericFactorPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope {
                        message: "factor sector absent from the source tensor",
                    },
                ))?;
        let side_offset = missing_offsets.entry(sector).or_default();
        let extent = block
            .shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .try_fold(1usize, |acc, (_, &value)| acc.checked_mul(value))
            .ok_or(CheckedGenericFactorPlanError::Operation(
                OperationError::ElementCountOverflow,
            ))?;
        let end =
            side_offset
                .checked_add(extent)
                .ok_or(CheckedGenericFactorPlanError::Operation(
                    OperationError::ElementCountOverflow,
                ))?;
        if end > dimension {
            return Err(CheckedGenericFactorPlanError::Operation(
                OperationError::ElementCountMismatch {
                    expected: dimension,
                    actual: end,
                },
            ));
        }
        scatter_identity_matrix_block(
            &mut data,
            block.shape(),
            block.strides(),
            block.offset(),
            axis,
            dimension,
            *side_offset,
            extent,
        )?;
        *side_offset = end;
    }
    BoundDynFactor::from_bound(space, data, nout, nin).map_err(CheckedGenericFactorPlanError::from)
}

#[doc(hidden)]
pub fn diagonal_bond_bound_space_generic_checked<R, V>(
    provider: Arc<R>,
    spectrum: &[SectorSpectrum<V>],
) -> Result<BoundDynamicFusionMapSpace<R>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
{
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
    BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
        .map_err(CheckedGenericFactorPlanError::from)
}

#[doc(hidden)]
pub fn diagonal_bond_svd_factor_generic_checked<R, D, V>(
    provider: Arc<R>,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
    V: Copy,
{
    let space = diagonal_bond_bound_space_generic_checked(provider, spectrum)?;
    let data = diagonal_bond_data(space.space(), spectrum, to_scalar)
        .map_err(CheckedGenericFactorPlanError::from)?;
    BoundDynFactor::from_bound(space, data, 1, 1).map_err(CheckedGenericFactorPlanError::from)
}

/// Generic sibling of [`scatter_left_sector_blocks`].
fn scatter_left_sector_blocks_generic<D, M>(
    left_space: &DynamicFusionMapSpace,
    left_data: &mut [D],
    matrix: &M,
    index: &PlacementIndex<'_>,
    groups: &SectorBlockGroups,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    M: SectorGeometry,
{
    let left_structure = Arc::clone(left_space.structure());
    for block_index in groups.blocks(matrix.sector()) {
        #[cfg(test)]
        record_scatter_visit(FactorSide::Left);
        let block = left_structure
            .block(block_index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        debug_assert_eq!(coupled_of_generic(key.codomain_tree()), matrix.sector());
        let (row_offset, _) =
            index.placement(matrix.sector(), FactorSide::Left, key.codomain_tree())?;
        #[cfg(test)]
        GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
            let mut value = probe.get();
            value.left_scatter_calls += 1;
            value.left_scattered_elements +=
                checked_extent(block.shape()).expect("admitted block extent is finite");
            probe.set(value);
        });
        scatter_matrix_block(
            left_data,
            block.shape(),
            block.strides(),
            block.offset(),
            block.shape().len() - 1,
            FactorSide::Left,
            factor,
            factor_rows,
            row_offset,
        );
    }
    Ok(())
}

/// Generic sibling of [`scatter_right_sector_blocks`].
fn scatter_right_sector_blocks_generic<D, M>(
    right_space: &DynamicFusionMapSpace,
    right_data: &mut [D],
    matrix: &M,
    index: &PlacementIndex<'_>,
    groups: &SectorBlockGroups,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    M: SectorGeometry,
{
    let right_structure = Arc::clone(right_space.structure());
    for block_index in groups.blocks(matrix.sector()) {
        #[cfg(test)]
        record_scatter_visit(FactorSide::Right);
        let block = right_structure
            .block(block_index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        debug_assert_eq!(coupled_of_generic(key.domain_tree()), matrix.sector());
        let (col_offset, _) =
            index.placement(matrix.sector(), FactorSide::Right, key.domain_tree())?;
        #[cfg(test)]
        GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
            let mut value = probe.get();
            value.right_scatter_calls += 1;
            value.right_scattered_elements +=
                checked_extent(block.shape()).expect("admitted block extent is finite");
            probe.set(value);
        });
        scatter_matrix_block(
            right_data,
            block.shape(),
            block.strides(),
            block.offset(),
            0,
            FactorSide::Right,
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
    let space = input.space().space();
    if let Some(plan) = compact_factor_plan_generic(input.space())? {
        return svd_compact_direct_regions(dense, input, &plan, CompactSvdGauge::Left);
    }
    let matricizations =
        sector_matricizations_generic(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_svd_input_pack(&matricizations);

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

    let mut singular_values = Vec::with_capacity(matricizations.len());

    let index = PlacementIndex::new(&matricizations, &[FactorSide::Left, FactorSide::Right]);
    let u_groups = SectorBlockGroups::new(u_space.space().structure(), FactorSide::Left)?;
    let vt_groups = SectorBlockGroups::new(vt_space.space().structure(), FactorSide::Right)?;
    for matrix in &matricizations {
        let rank = matrix.rows.min(matrix.cols);
        let (mut u, values, mut vt) =
            compact_svd_owned(dense, &matrix.data, matrix.rows, matrix.cols)?;
        svd_compact_gauge(
            &mut u,
            matrix.rows,
            matrix.rows,
            &mut vt,
            rank,
            matrix.cols,
            rank,
        );
        #[cfg(test)]
        record_generic_compact_svd_fallback_gauge(&u, &vt);

        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values,
        });
        scatter_left_sector_blocks_generic(
            u_space.space(),
            &mut u_data,
            matrix,
            &index,
            &u_groups,
            &u,
            matrix.rows,
        )?;
        scatter_right_sector_blocks_generic(
            vt_space.space(),
            &mut vt_data,
            matrix,
            &index,
            &vt_groups,
            &vt,
            rank,
        )?;
        #[cfg(test)]
        {
            record_compact_svd_output_scatter::<D>(matrix.rows * rank);
            record_compact_svd_output_scatter::<D>(rank * matrix.cols);
        }
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
    #[cfg(test)]
    record_diagonal_bond_build(spectrum);
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
    BoundDynamicFusionMapSpace::from_final_homspace_generic(provider, homspace)
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
    let space = input.space().space();
    let matricizations =
        generic_value_matricizations(space.structure(), input.data(), space.nout())?;
    let mut singular_values = Vec::with_capacity(matricizations.len());
    for index in 0..matricizations.len() {
        let matrix = matricizations.get(index)?;
        let rank = matrix.rows.min(matrix.cols);
        let input_shape = [matrix.rows, matrix.cols];
        let input_strides = [1usize, matrix.rows];
        let input = DenseView::new(matrix.data, &input_shape, &input_strides, 0)
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

fn invalid_eigenvalues() -> OperationError {
    OperationError::InvalidArgument {
        message: "eigenvalues must be finite",
    }
}

fn validate_real_eigenvalues(values: &[f64]) -> Result<(), OperationError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_eigenvalues())
    }
}

#[cfg(test)]
pub(crate) fn validate_real_eigenvalues_for_test(values: &[f64]) -> Result<(), OperationError> {
    validate_real_eigenvalues(values)
}

fn validate_complex_eigenvalues(values: &[Complex64]) -> Result<(), OperationError> {
    if values
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite() && value.norm().is_finite())
    {
        Ok(())
    } else {
        Err(invalid_eigenvalues())
    }
}

fn decide_bond_truncation_generic<R, V>(
    rule: &R,
    spectra: &[SectorSpectrum<V>],
    truncation: &Truncation,
    values_are_nonnegative: bool,
) -> Result<crate::truncation::TruncationDecision, OperationError>
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
            sector: entry.sector,
            weight: generic_truncation_weight(rule.sqrt_dim_scalar(entry.sector)),
            values: values.as_slice(),
        })
        .collect();
    select_truncation(&weighted, truncation, &rule.rule_identity()).map_err(OperationError::from)
}

fn decide_bond_truncation_generic_checked<R, V>(
    rule: &R,
    spectra: &[SectorSpectrum<V>],
    truncation: &Truncation,
) -> Result<crate::truncation::TruncationDecision, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericRigidSymbols<Scalar = f64>,
    V: SpectrumMagnitude,
{
    let magnitudes: Vec<Vec<f64>> = spectra
        .iter()
        .map(|entry| entry.values.iter().map(|value| value.magnitude()).collect())
        .collect();
    let mut weighted = Vec::with_capacity(spectra.len());
    for (entry, values) in spectra.iter().zip(&magnitudes) {
        let sqrt_dim = rule
            .try_sqrt_dim_scalar(entry.sector)
            .map_err(CheckedGenericFactorPlanError::Provider)?;
        weighted.push(WeightedSpectrum {
            sector: entry.sector,
            weight: sqrt_dim * sqrt_dim,
            values,
        });
    }
    select_truncation(&weighted, truncation, &rule.rule_identity())
        .map_err(|error| CheckedGenericFactorPlanError::Operation(error.into()))
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
    for key in keys.iter() {
        // Why not build the truncated layout first: a missing full-factor tree
        // remains a source error, while prepared Generic enumeration belongs
        // to #257 rather than a second layout abstraction here.
        let old_index = source_structure
            .find_block_index_by_key(&BlockKey::FusionTree(key.clone()))
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: "truncated factor tree must exist in the full factor",
            })?;
        source_structure
            .block(old_index)
            .map_err(OperationError::from_core_preserving_context)?;
    }

    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic(provider, new_hom)?;
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

fn sliced_bond_tensor_generic_checked<R, D>(
    provider: Arc<R>,
    source_space: &DynamicFusionMapSpace,
    source_data: &[D],
    axis: usize,
    kept_of: &dyn Fn(SectorId) -> usize,
    expected_nout: usize,
    expected_nin: usize,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
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
    let prepared = new_hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(provider.as_ref())
        .map_err(CheckedGenericFactorPlanError::from)?;
    for key in staged_fusion_tree_keys(prepared.sector_structure()) {
        let old = source_structure
            .find_block_index_by_key(&BlockKey::FusionTree(key.clone()))
            .ok_or(CheckedGenericFactorPlanError::Operation(
                OperationError::UnsupportedTensorContractScope {
                    message: "truncated factor tree must exist in the full factor",
                },
            ))?;
        source_structure.block(old).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(
                error,
            ))
        })?;
    }
    let space = BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        provider, new_hom, prepared,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let len = space.space().required_len().map_err(|error| {
        CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(
            error,
        ))
    })?;
    let mut data = vec![D::zero(); len];
    let sliced_structure = Arc::clone(space.space().structure());
    for index in 0..sliced_structure.block_count() {
        let new_block = sliced_structure.block(index).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(
                error,
            ))
        })?;
        let old_index = source_structure
            .find_block_index_by_key(new_block.key())
            .ok_or(CheckedGenericFactorPlanError::Operation(
                OperationError::UnsupportedTensorContractScope {
                    message: "truncated factor tree must exist in the full factor",
                },
            ))?;
        let old_block = source_structure.block(old_index).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(
                error,
            ))
        })?;
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
        .map_err(CheckedGenericFactorPlanError::from)
}

fn truncate_svd_factors_only_dyn_generic<R, D>(
    u: BoundDynFactor<R, D>,
    vh: BoundDynFactor<R, D>,
    mut singular_values: Vec<SectorSpectrum>,
    truncation: &Truncation,
) -> Result<SvdTruncFactorsDyn<R, D>, OperationError>
where
    R: GenericRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let rule = u.space().provider();
    let decision = decide_bond_truncation_generic(rule, &singular_values, truncation, true)?;
    if singular_values
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok((u, vh, singular_values, decision.error));
    }

    for (entry, &count) in singular_values.iter_mut().zip(&decision.kept) {
        entry.values.truncate(count);
    }
    singular_values.retain(|entry| !entry.values.is_empty());
    let kept_by_sector: FxHashMap<SectorId, usize> = singular_values
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();

    let kept_of = |sector: SectorId| -> usize { kept_by_sector.get(&sector).copied().unwrap_or(0) };

    let bond_axis = u.space().space().nout();
    let provider = Arc::clone(u.space().provider_arc());
    let u_factor = sliced_bond_tensor_generic(
        Arc::clone(&provider),
        u.space().space(),
        u.data(),
        bond_axis,
        &kept_of,
        u.space().space().nout(),
        1,
    )?;
    let vh_factor = sliced_bond_tensor_generic(
        provider,
        vh.space().space(),
        vh.data(),
        0,
        &kept_of,
        1,
        vh.space().space().nin(),
    )?;
    Ok((u_factor, vh_factor, singular_values, decision.error))
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
    let (u, vh, singular_values, error) = svd_trunc_factors_dyn_generic(dense, input, truncation)?;
    let s = diagonal_bond_svd_factor_generic(
        Arc::clone(input.space().provider_arc()),
        &singular_values,
        &D::from_real,
    )?;
    Ok(SvdTruncDyn {
        u,
        s,
        vh,
        singular_values,
        error,
    })
}

/// Generic dynamic-rank truncated SVD without a materialized diagonal `S`.
#[doc(hidden)]
pub fn svd_trunc_factors_dyn_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    truncation: &Truncation,
) -> Result<SvdTruncFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: GenericRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (u, vh, singular_values) = svd_compact_factors_dyn_generic(dense, input)?;
    truncate_svd_factors_only_dyn_generic(u, vh, singular_values, truncation)
}

#[doc(hidden)]
pub fn svd_trunc_factors_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    truncation: &Truncation,
) -> Result<SvdTruncFactorsDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (u, _s, vh, mut singular_values) =
        svd_compact_with_spectrum_dyn_checked_generic(dense, input)?;
    let rule = u.space().provider();
    let decision = decide_bond_truncation_generic_checked(rule, &singular_values, truncation)?;
    if singular_values
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return Ok((u, vh, singular_values, decision.error));
    }
    for (entry, &count) in singular_values.iter_mut().zip(&decision.kept) {
        entry.values.truncate(count);
    }
    singular_values.retain(|entry| !entry.values.is_empty());
    let kept: FxHashMap<SectorId, usize> = singular_values
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();
    let kept_of = |sector: SectorId| kept.get(&sector).copied().unwrap_or(0);
    let bond_axis = u.space().space().nout();
    let provider = Arc::clone(u.space().provider_arc());
    let u_factor = sliced_bond_tensor_generic_checked(
        Arc::clone(&provider),
        u.space().space(),
        u.data(),
        bond_axis,
        &kept_of,
        u.space().space().nout(),
        1,
    )?;
    let vh_factor = sliced_bond_tensor_generic_checked(
        provider,
        vh.space().space(),
        vh.data(),
        0,
        &kept_of,
        1,
        vh.space().space().nin(),
    )?;
    Ok((u_factor, vh_factor, singular_values, decision.error))
}

/// Provider-bound compact QR for a generic rule.
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
    if let Some(plan) = compact_factor_plan_generic(input.space())? {
        return qr_compact_direct_regions(dense, input, &plan);
    }
    let matrices = sector_matricizations_generic(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_qr_input_pack(&matrices);
    let mut pairs = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let rank = matrix.rows.min(matrix.cols);
        let (mut q, mut r) = compact_qr_owned(dense, &matrix.data, matrix.rows, matrix.cols)?;
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
    #[cfg(test)]
    let scatter_before = generic_pair_publication_probe();
    let result = build_left_right_bound_pair_generic(provider, space.homspace(), &matrices, pairs);
    #[cfg(test)]
    {
        let scatter_after = generic_pair_publication_probe();
        let elements = scatter_after.left_scattered_elements
            - scatter_before.left_scattered_elements
            + scatter_after.right_scattered_elements
            - scatter_before.right_scattered_elements;
        let calls = scatter_after.left_scatter_calls - scatter_before.left_scatter_calls
            + scatter_after.right_scatter_calls
            - scatter_before.right_scatter_calls;
        if calls != 0 {
            record_compact_qr_output_scatter_work::<D>(calls, elements);
        }
    }
    result
}

/// Checked-Generic compact QR. Provider-bound output spaces are admitted
/// through the checked staging boundary; dense QR itself performs no provider
/// queries and therefore needs no Tenferro-specific capability.
#[doc(hidden)]
#[expect(
    clippy::type_complexity,
    reason = "the public checked factorization API exposes its ordered factor tuple directly"
)]
pub fn qr_compact_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    #[cfg(test)]
    if let InputMatricizations::Packed(matrices) = &matrices {
        record_compact_qr_input_pack(matrices);
    }
    let mut pairs = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        #[cfg(test)]
        record_checked_compact_input(CheckedCompactOperation::Qr, input.data(), matrix.data, None);
        let rank = matrix.rows.min(matrix.cols);
        let (mut q, mut r) = compact_qr_owned(dense, matrix.data, matrix.rows, matrix.cols)
            .map_err(CheckedGenericFactorPlanError::from)?;
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
    build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)
}

/// Checked-Generic compact SVD. Dense SVD is unchanged; all provider-bound
/// output spaces are admitted through the checked staging boundary.
#[doc(hidden)]
#[expect(
    clippy::type_complexity,
    reason = "the public checked SVD API exposes its ordered U, S, and Vh tuple directly"
)]
pub fn svd_compact_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<
    (
        BoundDynFactor<R, D>,
        BoundDynFactor<R, D>,
        BoundDynFactor<R, D>,
    ),
    CheckedGenericFactorPlanError<R::Error>,
>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let (u, s, vh, _) = svd_compact_with_spectrum_dyn_checked_generic(dense, input)?;
    Ok((u, s, vh))
}

type CheckedCompactSvdWithSpectrum<R, D> = (
    BoundDynFactor<R, D>,
    BoundDynFactor<R, D>,
    BoundDynFactor<R, D>,
    Vec<SectorSpectrum>,
);

fn svd_compact_with_spectrum_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<CheckedCompactSvdWithSpectrum<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    #[cfg(test)]
    if let InputMatricizations::Packed(matrices) = &matrices {
        record_compact_svd_input_pack(matrices);
    }
    let mut pairs = Vec::with_capacity(matrices.len());
    let mut singular_values = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        #[cfg(test)]
        record_checked_compact_input(
            CheckedCompactOperation::Svd,
            input.data(),
            matrix.data,
            None,
        );
        let stage = compact_svd_numerical_stage(dense, matrix.data, matrix.rows, matrix.cols)
            .map_err(CheckedGenericFactorPlanError::from)?;
        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: stage.singular_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: stage.rank,
            left: stage.u,
            left_rows: stage.rows,
            right: stage.vt,
            right_leading: stage.rank,
        });
    }
    let (u, vh) = build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)?;
    let s = diagonal_bond_svd_factor_generic_checked(
        Arc::clone(provider),
        &singular_values,
        &D::from_real,
    )?;
    Ok((u, s, vh, singular_values))
}

/// Checked-Generic compact LQ, implemented through the existing host
/// adjoint-plus-QR boundary; no borrowed conjugated-dot capability is needed.
#[doc(hidden)]
#[expect(
    clippy::type_complexity,
    reason = "the public checked factorization API exposes its ordered factor tuple directly"
)]
pub fn lq_compact_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    #[cfg(test)]
    if let InputMatricizations::Packed(matrices) = &matrices {
        record_compact_lq_input_pack(matrices);
    }
    let mut pairs = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let rank = matrix.rows.min(matrix.cols);
        #[cfg(test)]
        record_compact_lq_adjoint_fill::<D>(matrix.data.len());
        let adjoint = adjoint_col_major(matrix.data, matrix.rows, matrix.cols);
        #[cfg(test)]
        record_checked_compact_input(
            CheckedCompactOperation::Lq,
            input.data(),
            matrix.data,
            Some(&adjoint),
        );
        let (mut q_prime, mut r_prime) =
            compact_qr_owned(dense, &adjoint, matrix.cols, matrix.rows)
                .map_err(CheckedGenericFactorPlanError::from)?;
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
    build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)
}

/// Checked-Generic full QR, augmenting only sectors that require completion.
#[doc(hidden)]
#[expect(
    clippy::type_complexity,
    reason = "the public checked factorization API exposes its ordered factor tuple directly"
)]
pub fn qr_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = sector_matricizations_generic(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let (q, r) = full_qr_numerical_stage(dense, &matrix.data, rows, cols)
            .map_err(CheckedGenericFactorPlanError::from)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rows,
            left: q,
            left_rows: rows,
            right: r,
            right_leading: rows,
        });
    }
    build_left_right_bound_pair_generic_checked(provider, space.homspace(), &matrices, pairs)
}

/// Checked-Generic full SVD. Dense work is performed before any output-space
/// publication; checked factor builders then admit square outer factors and
/// the rectangular diagonal, including unmatched structural sectors.
#[doc(hidden)]
pub fn svd_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let mut matrices = sector_matricizations_generic(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let row_dimensions = coupled_sector_block_dimensions_generic_checked(
        space.homspace().codomain(),
        provider.as_ref(),
    )?;
    let col_dimensions = coupled_sector_block_dimensions_generic_checked(
        space.homspace().domain(),
        provider.as_ref(),
    )?;
    let max_rows = matrices.iter().map(|m| m.rows).max().unwrap_or(0);
    let max_cols = matrices.iter().map(|m| m.cols).max().unwrap_or(0);
    let max_rank = matrices
        .iter()
        .map(|m| m.rows.min(m.cols))
        .max()
        .unwrap_or(0);
    let mut u_workspace = Vec::new();
    let mut s_workspace = Vec::new();
    let mut vt_workspace = Vec::new();
    let mut pairs = Vec::with_capacity(matrices.len());
    let mut singular_values = Vec::with_capacity(matrices.len());
    for matrix in &mut matrices {
        let rank = matrix.rows.min(matrix.cols);
        let (mut left, s_values, mut right) =
            match owned_full_svd_stage(dense, &mut matrix.data, matrix.rows, matrix.cols)
                .map_err(CheckedGenericFactorPlanError::from)?
            {
                Some(outputs) => outputs,
                None => {
                    if u_workspace.is_empty() && max_rows != 0 && max_rank != 0 {
                        u_workspace = vec![D::zero(); max_rows * max_rank];
                        s_workspace = vec![D::Real::zero(); max_rank];
                        vt_workspace = vec![D::zero(); max_rank * max_cols];
                    }
                    let shape = [matrix.rows, matrix.cols];
                    let strides = [1usize, matrix.rows];
                    let u_shape = [matrix.rows, rank];
                    let u_strides = [1usize, max_rows];
                    let s_shape = [rank];
                    let s_strides = [1usize];
                    let vt_shape = [rank, matrix.cols];
                    let vt_strides = [1usize, max_rank];
                    let input_view =
                        DenseView::new(&matrix.data, &shape, &strides, 0).map_err(|e| {
                            CheckedGenericFactorPlanError::Operation(OperationError::Dense(e))
                        })?;
                    let u_view = DenseViewMut::new(&mut u_workspace, &u_shape, &u_strides, 0)
                        .map_err(|e| {
                            CheckedGenericFactorPlanError::Operation(OperationError::Dense(e))
                        })?;
                    let s_view = DenseViewMut::new(&mut s_workspace, &s_shape, &s_strides, 0)
                        .map_err(|e| {
                            CheckedGenericFactorPlanError::Operation(OperationError::Dense(e))
                        })?;
                    let vt_view = DenseViewMut::new(&mut vt_workspace, &vt_shape, &vt_strides, 0)
                        .map_err(|e| {
                        CheckedGenericFactorPlanError::Operation(OperationError::Dense(e))
                    })?;
                    dense
                        .svd_into(
                            D::dense_read(input_view),
                            D::dense_write(u_view),
                            D::Real::dense_write(s_view),
                            D::dense_write(vt_view),
                        )
                        .map_err(|e| {
                            CheckedGenericFactorPlanError::Operation(OperationError::Dense(e))
                        })?;
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
                    let left = orthonormal_completion(dense, &u_thin, matrix.rows, rank)
                        .map_err(CheckedGenericFactorPlanError::from)?;
                    let v_thin = adjoint_col_major(&vt_thin, rank, matrix.cols);
                    let v_full = orthonormal_completion(dense, &v_thin, matrix.cols, rank)
                        .map_err(CheckedGenericFactorPlanError::from)?;
                    (
                        left,
                        s_workspace[..rank]
                            .iter()
                            .copied()
                            .map(Into::into)
                            .collect(),
                        adjoint_col_major(&v_full, matrix.cols, matrix.cols),
                    )
                }
            };
        svd_full_gauge(
            &mut left,
            matrix.rows,
            matrix.rows,
            &mut right,
            matrix.cols,
            matrix.cols,
        );
        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: matrix.rows,
            left,
            left_rows: matrix.rows,
            right,
            right_leading: matrix.cols,
        });
    }
    let u = build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        &matrices,
        &mut pairs,
        &row_dimensions,
        FactorSide::Left,
    )?;
    let vh = build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        &matrices,
        &mut pairs,
        &col_dimensions,
        FactorSide::Right,
    )?;
    let s = rectangular_diagonal_bond_tensor_generic_checked(
        Arc::clone(provider),
        &singular_values,
        &row_dimensions,
        &col_dimensions,
        &D::from_real,
    )?;
    Ok(SvdFullDyn {
        u,
        s,
        vh,
        singular_values,
    })
}

/// Checked-Generic full LQ via the full QR of each sector's adjoint matrix.
#[doc(hidden)]
#[expect(
    clippy::type_complexity,
    reason = "the public checked factorization API exposes its ordered factor tuple directly"
)]
pub fn lq_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynFactor<R, D>, BoundDynFactor<R, D>), CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = sector_matricizations_generic(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let rows = matrix.rows;
        let cols = matrix.cols;
        let transposed = adjoint_col_major(&matrix.data, rows, cols);
        let (q_prime, r_prime) = full_qr_numerical_stage(dense, &transposed, cols, rows)
            .map_err(CheckedGenericFactorPlanError::from)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: cols,
            left: adjoint_col_major(&r_prime, cols, rows),
            left_rows: rows,
            right: adjoint_col_major(&q_prime, cols, cols),
            right_leading: cols,
        });
    }
    build_left_right_bound_pair_generic_checked(provider, space.homspace(), &matrices, pairs)
}

/// Checked-Generic singular values only. No factor-space publication occurs.
#[doc(hidden)]
pub fn svd_vals_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let space = input.space().space();
    let matricizations =
        generic_value_matricizations(space.structure(), input.data(), space.nout())
            .map_err(CheckedGenericFactorPlanError::from)?;
    let mut singular_values = Vec::with_capacity(matricizations.len());
    for index in 0..matricizations.len() {
        let matrix = matricizations
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let input_shape = [matrix.rows, matrix.cols];
        let input_strides = [1usize, matrix.rows];
        let input_view =
            DenseView::new(matrix.data, &input_shape, &input_strides, 0).map_err(|error| {
                CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
            })?;
        let values = dense.svd_vals(D::dense_read(input_view)).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        let mut values = D::real_spectrum(&values).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        values.truncate(matrix.rows.min(matrix.cols));
        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values,
        });
    }
    Ok(singular_values)
}

/// Checked-Generic full Hermitian eigendecomposition. The exact source
/// provider remains the authority for the eigenvector factor.
#[doc(hidden)]
pub fn eigh_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<EighFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope {
                message: "eigh requires an endomorphism (codomain == domain)",
            },
        ));
    }
    let matrices = sector_matricizations_generic(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    // Why not trust equal product spaces alone: outer-multiplicity vertices
    // are part of a tree key, so Hermitian coordinates require identical full
    // tree stacking, not merely equal coupled-sector dimensions.
    validate_endomorphism_tree_stacking(&matrices).map_err(CheckedGenericFactorPlanError::from)?;
    validate_hermitian_matricizations(&matrices).map_err(CheckedGenericFactorPlanError::from)?;

    let max_n = matrices.iter().map(|matrix| matrix.rows).max().unwrap_or(0);
    let mut order = Vec::with_capacity(max_n);
    let mut visited = vec![false; max_n];
    let mut column_scratch = vec![D::zero(); max_n];
    let mut eigenvalues = Vec::with_capacity(matrices.len());
    let mut pairs = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let n = matrix.rows;
        let (real_values, mut vectors) = compact_eigh_owned(dense, &matrix.data, n)
            .map_err(CheckedGenericFactorPlanError::from)?;
        validate_real_eigenvalues(&real_values).map_err(CheckedGenericFactorPlanError::from)?;
        order.clear();
        order.extend(0..n);
        order.sort_by(|&a, &b| {
            real_values[b]
                .abs()
                .total_cmp(&real_values[a].abs())
                .then(a.cmp(&b))
        });
        let sorted_values = order.iter().map(|&index| real_values[index]).collect();
        reorder_columns_in_place(&mut vectors, n, &order, &mut visited, &mut column_scratch);
        eigenvector_gauge(&mut vectors, n, n, n);
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: n,
            left: vectors,
            left_rows: n,
            right: Vec::new(),
            right_leading: 0,
        });
    }
    let dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.rows))
        .collect::<BTreeMap<_, _>>();
    #[cfg(test)]
    record_checked_eigh_pair_pointers(&pairs);
    let v = build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        &matrices,
        &mut pairs,
        &dimensions,
        FactorSide::Left,
    )?;
    Ok(EighFullDyn { v, eigenvalues })
}

/// Checked-Generic truncated Hermitian eigendecomposition.
#[doc(hidden)]
pub fn eigh_trunc_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    truncation: &Truncation,
) -> Result<EighTruncDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let full = eigh_full_dyn_checked_generic(dense, input)?;
    if matches!(truncation, Truncation::Full) {
        return Ok(EighTruncDyn {
            v: full.v,
            eigenvalues: full.eigenvalues,
            error: 0.0,
        });
    }
    let decision = decide_bond_truncation_generic_checked(
        full.v.space().provider(),
        &full.eigenvalues,
        truncation,
    )?;
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
    let kept = eigenvalues
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect::<FxHashMap<_, _>>();
    let kept_of = |sector| kept.get(&sector).copied().unwrap_or(0);
    let bond_axis = full.v.space().space().nout();
    let provider = Arc::clone(full.v.space().provider_arc());
    let v = sliced_bond_tensor_generic_checked(
        provider,
        full.v.space().space(),
        full.v.data(),
        bond_axis,
        &kept_of,
        bond_axis,
        1,
    )?;
    Ok(EighTruncDyn {
        v,
        eigenvalues,
        error: decision.error,
    })
}

fn eig_not_numerically_diagonalizable() -> OperationError {
    OperationError::InvalidArgument {
        message: "eig requires a numerically diagonalizable coupled-sector matrix",
    }
}

pub(crate) fn validate_eigenvector_singular_values(
    singular_values: &[f64],
    n: usize,
    epsilon: f64,
) -> Result<(), OperationError> {
    if singular_values.len() != n || singular_values.iter().any(|value| !value.is_finite()) {
        return Err(eig_not_numerically_diagonalizable());
    }
    let sigma_max = singular_values.first().copied().unwrap_or(0.0);
    let tolerance = n as f64 * epsilon * sigma_max;
    if singular_values.iter().all(|&sigma| sigma > tolerance) {
        Ok(())
    } else {
        Err(eig_not_numerically_diagonalizable())
    }
}

/// Checked-Generic full general eigendecomposition. All input components and
/// dense results are validated before the exact source provider is asked to
/// admit either output factor.
#[doc(hidden)]
pub fn eig_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<EigFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope {
                message: "eig requires an endomorphism (codomain == domain)",
            },
        ));
    }
    let matrices = sector_matricizations_generic(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    validate_endomorphism_tree_stacking(&matrices).map_err(CheckedGenericFactorPlanError::from)?;
    if matrices
        .iter()
        .flat_map(|matrix| &matrix.data)
        .any(|&value| {
            let value = value.widen_complex();
            !value.re.is_finite() || !value.im.is_finite()
        })
    {
        return Err(CheckedGenericFactorPlanError::Operation(
            OperationError::InvalidArgument {
                message: "eig input components must be finite",
            },
        ));
    }

    let mut pairs: Vec<FactorPair<D::Eig>> = Vec::with_capacity(matrices.len());
    let mut eigenvalues = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let n = matrix.rows;
        let shape = [n, n];
        let strides = [1usize, n];
        let view = DenseView::new(&matrix.data, &shape, &strides, 0).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        let outputs = dense.eig(D::dense_read(view)).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        if outputs.len() != 2 {
            return Err(CheckedGenericFactorPlanError::Operation(
                OperationError::UnsupportedTensorContractScope {
                    message: "dense eig must return exactly (values, vectors)",
                },
            ));
        }
        validate_dense_shape(outputs[0].shape(), &[n])
            .map_err(CheckedGenericFactorPlanError::from)?;
        validate_dense_shape(outputs[1].shape(), &[n, n])
            .map_err(CheckedGenericFactorPlanError::from)?;
        let values = <D::Eig as FactorScalar>::dense_slice(&outputs[0]).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        let vectors = <D::Eig as FactorScalar>::dense_slice(&outputs[1]).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        let complex_values = values
            .iter()
            .map(|&value| value.widen_complex())
            .collect::<Vec<_>>();
        validate_complex_eigenvalues(&complex_values)
            .map_err(CheckedGenericFactorPlanError::from)?;
        if vectors.iter().any(|&value| {
            let value = value.widen_complex();
            !value.re.is_finite() || !value.im.is_finite()
        }) {
            return Err(CheckedGenericFactorPlanError::Operation(
                eig_not_numerically_diagonalizable(),
            ));
        }

        let vector_view = DenseView::new(vectors, &shape, &strides, 0).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        let singular_values = dense
            .svd_vals(<D::Eig as DenseBlockScalar>::dense_read(vector_view))
            .map_err(|error| {
                CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
            })?;
        validate_dense_shape(singular_values.shape(), &[n])
            .map_err(CheckedGenericFactorPlanError::from)?;
        let singular_values =
            <D::Eig as FactorScalar>::real_spectrum(&singular_values).map_err(|error| {
                CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
            })?;
        validate_eigenvector_singular_values(
            &singular_values,
            n,
            <D::Eig as FactorScalar>::epsilon(),
        )
        .map_err(CheckedGenericFactorPlanError::from)?;

        let mut order = (0..n).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            complex_values[b]
                .norm()
                .total_cmp(&complex_values[a].norm())
                .then(a.cmp(&b))
        });
        let sorted_values = order.iter().map(|&index| complex_values[index]).collect();
        let mut sorted_vectors = vec![D::Eig::zero(); n * n];
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
            left: sorted_vectors,
            left_rows: n,
            right: Vec::new(),
            right_leading: 0,
        });
    }

    let dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.rows))
        .collect::<BTreeMap<_, _>>();
    let v = build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        &matrices,
        &mut pairs,
        &dimensions,
        FactorSide::Left,
    )?;
    Ok(EigFullDyn { v, eigenvalues })
}

/// Checked-Generic general eigendecomposition with shared global truncation.
#[doc(hidden)]
pub fn eig_trunc_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    truncation: &Truncation,
) -> Result<EigTruncDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let full = eig_full_dyn_checked_generic(dense, input)?;
    if matches!(truncation, Truncation::Full) {
        return Ok(EigTruncDyn {
            v: full.v,
            eigenvalues: full.eigenvalues,
            error: 0.0,
        });
    }
    let decision = decide_bond_truncation_generic_checked(
        full.v.space().provider(),
        &full.eigenvalues,
        truncation,
    )?;
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
    let kept = eigenvalues
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect::<FxHashMap<_, _>>();
    let kept_of = |sector| kept.get(&sector).copied().unwrap_or(0);
    let bond_axis = full.v.space().space().nout();
    let provider = Arc::clone(full.v.space().provider_arc());
    let v = sliced_bond_tensor_generic_checked(
        provider,
        full.v.space().space(),
        full.v.data(),
        bond_axis,
        &kept_of,
        bond_axis,
        1,
    )?;
    Ok(EigTruncDyn {
        v,
        eigenvalues,
        error: decision.error,
    })
}

/// Checked-Generic Hermitian eigenvalues only. No eigenvector or factor-space
/// publication occurs.
#[doc(hidden)]
pub fn eigh_vals_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope {
                message: "eigh requires an endomorphism (codomain == domain)",
            },
        ));
    }
    let matricizations =
        generic_value_matricizations(space.structure(), input.data(), space.nout())
            .map_err(CheckedGenericFactorPlanError::from)?;
    matricizations
        .validate_hermitian()
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    for index in 0..matricizations.len() {
        let matrix = matricizations
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let n = matrix.rows;
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view = DenseView::new(matrix.data, &shape, &strides, 0).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        let values_tensor = dense.eigh_vals(D::dense_read(view)).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        let mut values = D::real_spectrum(&values_tensor).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        values.truncate(n);
        validate_real_eigenvalues(&values).map_err(CheckedGenericFactorPlanError::from)?;
        values.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values,
        });
    }
    Ok(eigenvalues)
}

/// Checked-Generic general eigenvalues only. No eigenvector or factor-space
/// publication occurs.
#[doc(hidden)]
pub fn eig_vals_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum<Complex64>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope {
                message: "eig requires an endomorphism (codomain == domain)",
            },
        ));
    }
    let matricizations =
        generic_value_matricizations(space.structure(), input.data(), space.nout())
            .map_err(CheckedGenericFactorPlanError::from)?;
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    for index in 0..matricizations.len() {
        let matrix = matricizations
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let n = matrix.rows;
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view = DenseView::new(matrix.data, &shape, &strides, 0).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        let values_tensor = dense.eig_vals(D::dense_read(view)).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        validate_dense_shape(values_tensor.shape(), &[n])
            .map_err(CheckedGenericFactorPlanError::from)?;
        let values = <D::Eig as FactorScalar>::dense_slice(&values_tensor).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        let mut values: Vec<Complex64> = values[..n]
            .iter()
            .map(|&value| value.widen_complex())
            .collect();
        validate_complex_eigenvalues(&values).map_err(CheckedGenericFactorPlanError::from)?;
        values.sort_by(|a, b| b.norm().total_cmp(&a.norm()));
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values,
        });
    }
    Ok(eigenvalues)
}

/// Provider-bound compact LQ for a generic rule.
#[expect(
    clippy::type_complexity,
    reason = "the public factorization API exposes its ordered factor tuple directly"
)]
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
    if let Some(plan) = compact_factor_plan_generic(input.space())? {
        return lq_compact_direct_regions(dense, input, &plan);
    }
    let matrices = sector_matricizations_generic(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_lq_input_pack(&matrices);
    let mut pairs = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let rank = matrix.rows.min(matrix.cols);
        let adjoint = adjoint_col_major(&matrix.data, matrix.rows, matrix.cols);
        let (mut q_prime, mut r_prime) =
            compact_qr_owned(dense, &adjoint, matrix.cols, matrix.rows)?;
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
    #[cfg(test)]
    let scatter_before = generic_pair_publication_probe();
    let result = build_left_right_bound_pair_generic(provider, space.homspace(), &matrices, pairs);
    #[cfg(test)]
    {
        let scatter_after = generic_pair_publication_probe();
        let elements = scatter_after.left_scattered_elements
            - scatter_before.left_scattered_elements
            + scatter_after.right_scattered_elements
            - scatter_before.right_scattered_elements;
        let calls = scatter_after.left_scatter_calls - scatter_before.left_scatter_calls
            + scatter_after.right_scatter_calls
            - scatter_before.right_scatter_calls;
        if calls != 0 {
            record_compact_lq_output_scatter_work::<D>(calls, elements);
        }
    }
    result
}

#[cfg(test)]
mod sector_matricization_tests {
    use super::*;
    use tenet_core::{BlockSpec, FusionTreePairKey, Z2FusionRule};

    struct PayloadFreeGeometry {
        sector: SectorId,
        rows: usize,
        cols: usize,
        row_tree: FusionTreeKey,
        row_shape: Vec<usize>,
        col_tree: FusionTreeKey,
        col_shape: Vec<usize>,
    }

    impl SectorGeometry for PayloadFreeGeometry {
        fn sector(&self) -> SectorId {
            self.sector
        }

        fn rows(&self) -> usize {
            self.rows
        }

        fn cols(&self) -> usize {
            self.cols
        }

        fn tree_count(&self, _side: FactorSide) -> usize {
            1
        }

        fn tree(&self, side: FactorSide, index: usize) -> Option<TreeExtentRef<'_>> {
            if index != 0 {
                return None;
            }
            let (tree, shape) = match side {
                FactorSide::Left => (&self.row_tree, self.row_shape.as_slice()),
                FactorSide::Right => (&self.col_tree, self.col_shape.as_slice()),
            };
            Some(TreeExtentRef {
                tree,
                offset: 0,
                shape,
            })
        }
    }

    #[derive(Clone, Copy)]
    struct TestGenericRule;

    impl FusionRule for TestGenericRule {
        fn rule_identity(&self) -> tenet_core::RuleIdentity {
            tenet_core::RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> tenet_core::FusionStyleKind {
            tenet_core::FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> tenet_core::BraidingStyleKind {
            tenet_core::BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            sector
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> tenet_core::SectorVec {
            match (left.id(), right.id()) {
                (0, sector) | (sector, 0) => [SectorId::new(sector)].into_iter().collect(),
                (1, 1) => [SectorId::new(0), SectorId::new(1)].into_iter().collect(),
                _ => tenet_core::SectorVec::new(),
            }
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }

    fn z2_pair(codomain: [usize; 2], domain: [usize; 2], coupled: usize) -> FusionTreePairKey {
        let pair = FusionTreePairKey::try_pair_from_sector_ids(
            codomain,
            domain,
            coupled,
            [false; 2],
            [false; 2],
            std::iter::empty::<usize>(),
            std::iter::empty::<usize>(),
            [1],
            [1],
        )
        .unwrap();
        pair.validate_for_rule(&Z2FusionRule).unwrap();
        pair
    }

    #[test]
    fn left_factor_publication_borrows_real_geometry_for_complex_output() {
        let x = SectorId::new(1);
        let vacuum = SectorId::new(0);
        let source_key = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1, 1],
            [1],
            1,
            [false; 3],
            [false],
            [0],
            std::iter::empty::<usize>(),
            [1, 1],
            std::iter::empty::<usize>(),
        )
        .unwrap();
        source_key.validate_for_rule(&Z2FusionRule).unwrap();
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(x, 2)], false),
                SectorLeg::new([(x, 1)], false),
                SectorLeg::new([(x, 3)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
        );
        let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(Z2FusionRule),
            homspace.clone(),
        )
        .unwrap();
        let geometry = [PayloadFreeGeometry {
            sector: x,
            rows: 6,
            cols: 1,
            row_tree: source_key.codomain_tree().clone(),
            row_shape: vec![2, 1, 3],
            col_tree: source_key.domain_tree().clone(),
            col_shape: vec![1],
        }];
        let row_tree_before = geometry[0].row_tree.clone();
        let col_tree_before = geometry[0].col_tree.clone();
        let expected = (0..12)
            .map(|index| Complex64::new(index as f64 + 0.5, 10.0 - index as f64))
            .collect::<Vec<_>>();
        let mut pairs = [FactorPair {
            sector: x,
            kept: 2,
            left: expected.clone(),
            left_rows: 6,
            right: Vec::new(),
            right_leading: 0,
        }];

        let selected_ptr = pairs[0].left.as_ptr();
        reset_one_sided_publication_probe();
        let factor = build_left_bound_factor(&authority, &homspace, &geometry, &mut pairs).unwrap();

        let probe = one_sided_publication_probe();
        assert_eq!(
            (
                probe.canonical_publications,
                probe.fallback_publications,
                probe.owner_reused,
                probe.appended_elements,
            ),
            (1, 0, 1, 0)
        );
        assert!(std::ptr::eq(selected_ptr, factor.data().as_ptr()));
        assert!(pairs[0].left.is_empty());
        assert_eq!(geometry[0].row_tree, row_tree_before);
        assert_eq!(geometry[0].col_tree, col_tree_before);
        assert_eq!(geometry[0].row_shape, [2, 1, 3]);
        assert_eq!(geometry[0].col_shape, [1]);
        let block = factor.space().space().structure().block(0).unwrap();
        assert_eq!(block.shape(), [2, 1, 3, 2]);
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("left factor must retain the complete fusion-tree key")
        };
        assert_eq!(key.codomain_tree(), &row_tree_before);
        assert_eq!(key.domain_tree().uncoupled(), &[x]);
        assert_eq!(key.domain_tree().coupled(), x);
        assert_eq!(key.domain_tree().vertices(), &[]);
        assert_eq!(key.domain_tree().innerlines(), &[]);
        assert_eq!(factor.data().len(), expected.len());
        for bond in 0..2 {
            for third in 0..3 {
                for first in 0..2 {
                    let output = block.offset()
                        + first * block.strides()[0]
                        + third * block.strides()[2]
                        + bond * block.strides()[3];
                    let matrix_row = first + 2 * third;
                    assert_eq!(factor.data()[output], expected[matrix_row + 6 * bond]);
                }
            }
        }
        assert_eq!(vacuum, row_tree_before.innerlines()[0]);
    }

    fn generic_pair(coupled: usize, row_vertex: usize, col_vertex: usize) -> FusionTreePairKey {
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [1, 1],
            coupled,
            [false; 2],
            [false; 2],
            std::iter::empty::<usize>(),
            std::iter::empty::<usize>(),
            [row_vertex],
            [col_vertex],
        )
        .unwrap()
    }

    fn full_identity_pair(
        row_inner: usize,
        row_dual: bool,
        col_inner: usize,
        col_dual: bool,
    ) -> FusionTreePairKey {
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 2, 3],
            [4, 5, 6],
            9,
            [false, row_dual, false],
            [true, false, col_dual],
            [row_inner],
            [col_inner],
            [1, 2],
            [2, 1],
        )
        .unwrap()
    }

    #[test]
    fn packed_and_region_geometry_preserve_dual_and_innerline_tree_identity() {
        let mut row_trees = vec![
            full_identity_pair(7, false, 11, false)
                .codomain_tree()
                .clone(),
            full_identity_pair(8, true, 11, false)
                .codomain_tree()
                .clone(),
        ];
        let mut col_trees = vec![
            full_identity_pair(7, false, 10, false)
                .domain_tree()
                .clone(),
            full_identity_pair(7, false, 11, true).domain_tree().clone(),
        ];
        row_trees.sort();
        col_trees.sort();
        assert_ne!(row_trees[0].innerlines(), row_trees[1].innerlines());
        assert_ne!(row_trees[0].is_dual(), row_trees[1].is_dual());
        assert_ne!(col_trees[0].innerlines(), col_trees[1].innerlines());
        assert_ne!(col_trees[0].is_dual(), col_trees[1].is_dual());

        let row_shapes = [vec![1, 2, 1], vec![2, 1, 1]];
        let col_shapes = [vec![1, 3, 1], vec![1, 1, 1]];
        let row_extents = row_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .collect::<Vec<_>>();
        let col_extents = col_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .collect::<Vec<_>>();
        let rows = row_extents.iter().sum::<usize>();
        let cols = col_extents.iter().sum::<usize>();
        let mut blocks = Vec::new();
        let mut col_offset = 0;
        for (col, (col_tree, col_shape)) in col_trees.iter().zip(&col_shapes).enumerate() {
            let mut row_offset = 0;
            for (row, (row_tree, row_shape)) in row_trees.iter().zip(&row_shapes).enumerate() {
                let mut shape = row_shape.clone();
                shape.extend_from_slice(col_shape);
                let mut strides = Vec::with_capacity(shape.len());
                let mut stride = 1;
                for &dimension in row_shape {
                    strides.push(stride);
                    stride *= dimension;
                }
                stride = rows;
                for &dimension in col_shape {
                    strides.push(stride);
                    stride *= dimension;
                }
                blocks.push(
                    BlockSpec::with_key(
                        FusionTreePairKey::pair(row_tree.clone(), col_tree.clone()).into(),
                        shape,
                        strides,
                        row_offset + rows * col_offset,
                    )
                    .unwrap(),
                );
                row_offset += row_extents[row];
            }
            col_offset += col_extents[col];
        }
        let structure = BlockStructure::from_blocks_with_rank(6, blocks).unwrap();
        let data = (0..rows * cols)
            .map(|value| value as f64)
            .collect::<Vec<_>>();
        let regions = structure.coupled_sector_regions(3).unwrap().unwrap();
        let packed = sector_matricizations_generic(&structure, &data, 3).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(packed.len(), 1);
        assert!(matches!(
            generic_input_matricizations(&structure, &data, 3).unwrap(),
            InputMatricizations::Regions { .. }
        ));
        let region = &regions[0];
        let matrix = &packed[0];
        assert_eq!(matrix.data, data);
        assert_eq!(region.sector(), SectorId::new(9));
        assert_eq!((region.rows(), region.cols()), (4, 4));
        assert_eq!(region.sector(), matrix.sector());
        assert_eq!(
            (region.rows(), region.cols()),
            (matrix.rows(), matrix.cols())
        );
        for side in [FactorSide::Left, FactorSide::Right] {
            let (expected_trees, expected_offsets, expected_shapes) = match side {
                FactorSide::Left => (&row_trees, [0, 2], &row_shapes),
                FactorSide::Right => (&col_trees, [0, 3], &col_shapes),
            };
            assert_eq!(region.tree_count(side), matrix.tree_count(side));
            for index in 0..region.tree_count(side) {
                let borrowed = region.tree(side, index).unwrap();
                let owned = matrix.tree(side, index).unwrap();
                assert_eq!(borrowed.tree, &expected_trees[index]);
                assert_eq!(borrowed.offset, expected_offsets[index]);
                assert_eq!(borrowed.shape, expected_shapes[index]);
                assert_eq!(owned.tree, &expected_trees[index]);
                assert_eq!(owned.offset, expected_offsets[index]);
                assert_eq!(owned.shape, expected_shapes[index]);
                assert_eq!(borrowed.tree, owned.tree);
                assert_eq!(borrowed.offset, owned.offset);
                assert_eq!(borrowed.shape, owned.shape);
            }
        }
    }

    #[test]
    fn sector_matricizations_preserve_encounter_order_and_padded_block_values() {
        // What: noncanonical storage packs repeated row/column trees into
        // first-encounter sector geometry without changing block copy order.
        let structure = BlockStructure::from_blocks_with_rank(
            4,
            vec![
                BlockSpec::with_key(
                    z2_pair([0, 1], [1, 0], 1).into(),
                    vec![1, 2, 3, 1],
                    vec![2, 9, 1, 20],
                    3,
                )
                .unwrap(),
                BlockSpec::with_key(
                    z2_pair([0, 0], [1, 1], 0).into(),
                    vec![2, 1, 1, 2],
                    vec![2, 20, 1, 7],
                    30,
                )
                .unwrap(),
                BlockSpec::with_key(
                    z2_pair([0, 0], [0, 0], 0).into(),
                    vec![2, 1, 2, 1],
                    vec![3, 20, 1, 50],
                    50,
                )
                .unwrap(),
                BlockSpec::with_key(
                    z2_pair([1, 1], [1, 1], 0).into(),
                    vec![1, 2, 1, 2],
                    vec![4, 1, 20, 7],
                    70,
                )
                .unwrap(),
                BlockSpec::with_key(
                    z2_pair([1, 1], [0, 0], 0).into(),
                    vec![1, 2, 2, 1],
                    vec![4, 1, 5, 20],
                    90,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut data = vec![0.0; structure.required_len().unwrap()];
        for (position, value) in [
            (3, 101.0),
            (12, 102.0),
            (4, 103.0),
            (13, 104.0),
            (5, 105.0),
            (14, 106.0),
            (30, 1.0),
            (32, 2.0),
            (37, 5.0),
            (39, 6.0),
            (50, 3.0),
            (53, 4.0),
            (51, 7.0),
            (54, 8.0),
            (70, 9.0),
            (71, 10.0),
            (77, 13.0),
            (78, 14.0),
            (90, 11.0),
            (91, 12.0),
            (95, 15.0),
            (96, 16.0),
        ] {
            data[position] = value;
        }

        let matrices = sector_matricizations::<f64>(&structure, &data, 2).unwrap();

        assert_eq!(
            matrices
                .iter()
                .map(|matrix| matrix.sector)
                .collect::<Vec<_>>(),
            [SectorId::new(1), SectorId::new(0)]
        );
        assert_eq!(matrices[0].rows, 2);
        assert_eq!(matrices[0].cols, 3);
        assert_eq!(matrices[0].data, [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]);
        assert_eq!(
            matrices[0].row_trees,
            vec![(
                z2_pair([0, 1], [1, 0], 1).codomain_tree().clone(),
                0,
                vec![1, 2],
            )]
        );
        assert_eq!(
            matrices[0].col_trees,
            vec![(
                z2_pair([0, 1], [1, 0], 1).domain_tree().clone(),
                0,
                vec![3, 1],
            )]
        );

        assert_eq!(matrices[1].rows, 4);
        assert_eq!(matrices[1].cols, 4);
        assert_eq!(
            matrices[1].data,
            [
                1.0, 2.0, 9.0, 10.0, 5.0, 6.0, 13.0, 14.0, 3.0, 4.0, 11.0, 12.0, 7.0, 8.0, 15.0,
                16.0,
            ]
        );
        assert_eq!(
            matrices[1].row_trees,
            vec![
                (
                    z2_pair([0, 0], [1, 1], 0).codomain_tree().clone(),
                    0,
                    vec![2, 1],
                ),
                (
                    z2_pair([1, 1], [1, 1], 0).codomain_tree().clone(),
                    2,
                    vec![1, 2],
                ),
            ]
        );
        assert_eq!(
            matrices[1].col_trees,
            vec![
                (
                    z2_pair([0, 0], [1, 1], 0).domain_tree().clone(),
                    0,
                    vec![1, 2],
                ),
                (
                    z2_pair([0, 0], [0, 0], 0).domain_tree().clone(),
                    2,
                    vec![2, 1],
                ),
            ]
        );
    }

    #[test]
    fn generic_sector_matricizations_preserve_full_tree_identity_and_exact_layout() {
        let structure = BlockStructure::from_blocks_with_rank(
            4,
            vec![
                BlockSpec::with_key(
                    generic_pair(1, 2, 1).into(),
                    vec![1, 2, 2, 1],
                    vec![1, 17, 3, 40],
                    5,
                )
                .unwrap(),
                BlockSpec::with_key(
                    generic_pair(0, 1, 1).into(),
                    vec![1, 1, 1, 1],
                    vec![1, 7, 5, 3],
                    80,
                )
                .unwrap(),
                BlockSpec::with_key(
                    generic_pair(1, 1, 2).into(),
                    vec![1, 1, 1, 3],
                    vec![1, 8, 2, 4],
                    60,
                )
                .unwrap(),
                BlockSpec::with_key(
                    generic_pair(1, 1, 1).into(),
                    vec![1, 1, 2, 1],
                    vec![1, 9, 5, 30],
                    40,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut real = vec![0.0; structure.required_len().unwrap()];
        for (index, value) in [
            (5, 11.0),
            (22, 12.0),
            (8, 13.0),
            (25, 14.0),
            (80, 90.0),
            (60, 31.0),
            (64, 32.0),
            (68, 33.0),
            (40, 21.0),
            (45, 22.0),
        ] {
            real[index] = value;
        }
        let expected = [
            11.0, 12.0, 21.0, 13.0, 14.0, 22.0, 0.0, 0.0, 31.0, 0.0, 0.0, 32.0, 0.0, 0.0, 33.0,
        ];

        let matrices = sector_matricizations_generic(&structure, &real, 2).unwrap();
        assert_eq!(
            matrices
                .iter()
                .map(|matrix| matrix.sector)
                .collect::<Vec<_>>(),
            [SectorId::new(1), SectorId::new(0)]
        );
        assert_eq!((matrices[0].rows, matrices[0].cols), (3, 5));
        assert_eq!(matrices[0].data, expected);
        assert_eq!(matrices[1].data, [90.0]);
        assert_eq!(
            matrices[0]
                .row_trees
                .iter()
                .map(|(tree, offset, shape)| (tree.vertices()[0].get(), *offset, shape.clone()))
                .collect::<Vec<_>>(),
            [(2, 0, vec![1, 2]), (1, 2, vec![1, 1])]
        );
        assert_eq!(
            matrices[0]
                .col_trees
                .iter()
                .map(|(tree, offset, shape)| (tree.vertices()[0].get(), *offset, shape.clone()))
                .collect::<Vec<_>>(),
            [(1, 0, vec![2, 1]), (2, 2, vec![1, 3])]
        );

        let complex = real
            .iter()
            .map(|&value| Complex64::new(value, -value / 10.0))
            .collect::<Vec<_>>();
        let complex_matrices = sector_matricizations_generic(&structure, &complex, 2).unwrap();
        assert_eq!(
            complex_matrices[0].data,
            expected.map(|value| Complex64::new(value, -value / 10.0))
        );
        assert_eq!(complex_matrices[1].data, [Complex64::new(90.0, -9.0)]);

        let empty = BlockStructure::from_blocks_with_rank(4, Vec::new()).unwrap();
        assert!(sector_matricizations_generic::<f64>(&empty, &[], 2)
            .unwrap()
            .is_empty());

        let scalar_key =
            FusionTreePairKey::try_pair_from_sector_ids([], [], 0, [], [], [], [], [], []).unwrap();
        let scalar = BlockStructure::from_blocks_with_rank(
            0,
            vec![BlockSpec::with_key(scalar_key.into(), vec![], vec![], 1).unwrap()],
        )
        .unwrap();
        assert_eq!(
            sector_matricizations_generic(&scalar, &[0.0, 7.0], 0).unwrap()[0].data,
            [7.0]
        );

        let zero_extent = BlockStructure::from_blocks_with_rank(
            4,
            vec![BlockSpec::with_key(
                generic_pair(1, 1, 1).into(),
                vec![0, 1, 1, 1],
                vec![1, 1, 1, 1],
                0,
            )
            .unwrap()],
        )
        .unwrap();
        let zero_matrix = sector_matricizations_generic::<f64>(&zero_extent, &[], 2).unwrap();
        assert_eq!((zero_matrix[0].rows, zero_matrix[0].cols), (0, 1));
        assert!(zero_matrix[0].data.is_empty());

        let non_fusion = BlockStructure::from_blocks_with_rank(
            1,
            vec![BlockSpec::with_key(BlockKey::opaque([7]), vec![1], vec![1], 0).unwrap()],
        )
        .unwrap();
        assert!(matches!(
            sector_matricizations_generic::<f64>(&non_fusion, &[1.0], 1),
            Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "tsvd",
                index: 0
            })
        ));
    }

    #[test]
    fn generic_sector_matricizations_keep_tree_offsets_matrix_local() {
        // Raw block structures can pair trees whose coupled sectors differ. A
        // domain tree shared by two codomain-selected matrices therefore has
        // an independent column offset in each matrix.
        let sector_one = generic_pair(1, 1, 1);
        let sector_zero = generic_pair(0, 1, 1);
        let row_one = sector_one.codomain_tree().clone();
        let row_zero = sector_zero.codomain_tree().clone();
        let shared_domain = sector_one.domain_tree().clone();
        let other_domain = sector_zero.domain_tree().clone();
        let structure = BlockStructure::from_blocks_with_rank(
            4,
            vec![
                BlockSpec::with_key(
                    FusionTreePairKey::pair(row_one.clone(), other_domain.clone()).into(),
                    vec![1, 1, 1, 1],
                    vec![1, 1, 1, 1],
                    1,
                )
                .unwrap(),
                BlockSpec::with_key(
                    FusionTreePairKey::pair(row_zero.clone(), shared_domain.clone()).into(),
                    vec![1, 1, 1, 2],
                    vec![1, 1, 1, 1],
                    4,
                )
                .unwrap(),
                BlockSpec::with_key(
                    FusionTreePairKey::pair(row_one.clone(), shared_domain.clone()).into(),
                    vec![1, 1, 1, 2],
                    vec![1, 1, 1, 1],
                    8,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut data = vec![0.0; structure.required_len().unwrap()];
        data[1] = 11.0;
        data[4..6].copy_from_slice(&[21.0, 22.0]);
        data[8..10].copy_from_slice(&[31.0, 32.0]);

        let matrices = sector_matricizations_generic(&structure, &data, 2).unwrap();

        assert_eq!(matrices.len(), 2);
        assert_eq!(matrices[0].sector, SectorId::new(1));
        assert_eq!((matrices[0].rows, matrices[0].cols), (1, 3));
        assert_eq!(matrices[0].data, [11.0, 31.0, 32.0]);
        assert_eq!(
            matrices[0]
                .col_trees
                .iter()
                .map(|(tree, offset, _)| (tree, *offset))
                .collect::<Vec<_>>(),
            [(&other_domain, 0), (&shared_domain, 1)]
        );
        assert_eq!(matrices[1].sector, SectorId::new(0));
        assert_eq!((matrices[1].rows, matrices[1].cols), (1, 2));
        assert_eq!(matrices[1].data, [21.0, 22.0]);
        assert_eq!(matrices[1].col_trees[0].0, shared_domain);
        assert_eq!(matrices[1].col_trees[0].1, 0);
    }

    fn z2_single_sector_matrix(
        rows: usize,
        cols: usize,
    ) -> (FusionTreeHomSpace, SectorMatricization<f64>) {
        let even = SectorId::new(0);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(even, rows)], false)]),
            FusionProductSpace::new([SectorLeg::new([(even, cols)], false)]),
        );
        let key = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap()[0].clone();
        (
            homspace,
            SectorMatricization {
                sector: even,
                rows,
                cols,
                row_trees: vec![(key.codomain_tree().clone(), 0, vec![rows])],
                col_trees: vec![(key.domain_tree().clone(), 0, vec![cols])],
                data: vec![0.0; rows * cols],
            },
        )
    }

    #[test]
    fn mf_one_sided_placement_uses_direct_and_adjoint_source_sides() {
        let (homspace, mut matrix) = z2_single_sector_matrix(2, 3);
        matrix.rows = 3;
        matrix.row_trees[0].1 = 1;
        matrix.cols = 5;
        matrix.col_trees[0].1 = 2;
        matrix.data = Vec::new();
        let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(Z2FusionRule),
            homspace.clone(),
        )
        .unwrap();
        let adjoint = tenet_tensors::adjoint_bound_space_dyn(&authority).unwrap();
        let values = |len: usize, base: f64| {
            (0..len)
                .map(|index| Complex64::new(base + index as f64, -base - index as f64 / 10.0))
                .collect::<Vec<_>>()
        };

        let cases = [
            (
                &authority,
                authority.space().homspace(),
                FactorSide::Left,
                FactorPlacement::Direct,
                2,
                FactorPair {
                    sector: SectorId::new(0),
                    kept: 99,
                    left: values(6, 10.0),
                    left_rows: 3,
                    right: Vec::new(),
                    right_leading: 0,
                },
                vec![1, 2, 4, 5],
            ),
            (
                &authority,
                authority.space().homspace(),
                FactorSide::Right,
                FactorPlacement::Direct,
                3,
                FactorPair {
                    sector: SectorId::new(0),
                    kept: 99,
                    left: Vec::new(),
                    left_rows: 0,
                    right: values(15, 20.0),
                    right_leading: 3,
                },
                (6..15).collect(),
            ),
            (
                &adjoint,
                adjoint.space().homspace(),
                FactorSide::Left,
                FactorPlacement::Adjoint,
                4,
                FactorPair {
                    sector: SectorId::new(0),
                    kept: 99,
                    left: values(20, 30.0),
                    left_rows: 5,
                    right: Vec::new(),
                    right_leading: 0,
                },
                vec![2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19],
            ),
            (
                &adjoint,
                adjoint.space().homspace(),
                FactorSide::Right,
                FactorPlacement::Adjoint,
                2,
                FactorPair {
                    sector: SectorId::new(0),
                    kept: 99,
                    left: Vec::new(),
                    left_rows: 0,
                    right: values(6, 40.0),
                    right_leading: 2,
                },
                (2..6).collect(),
            ),
        ];

        for (authority, output_hom, side, placement, bond, mut pair, selected) in cases {
            let source = match side {
                FactorSide::Left => &pair.left,
                FactorSide::Right => &pair.right,
            };
            let expected = selected
                .into_iter()
                .map(|index| source[index])
                .collect::<Vec<_>>();
            let factor = build_bound_factor_with_placement(
                authority,
                output_hom,
                std::slice::from_ref(&matrix),
                std::slice::from_mut(&mut pair),
                &BTreeMap::from([(SectorId::new(0), bond)]),
                side,
                placement,
            )
            .unwrap();
            assert_eq!(factor.data(), expected);
        }
    }

    #[test]
    fn mf_one_sided_pair_errors_precede_tree_traversal() {
        let (homspace, mut matrix) = z2_single_sector_matrix(1, 1);
        let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(Z2FusionRule),
            homspace.clone(),
        )
        .unwrap();
        matrix.row_trees.clear();
        matrix.col_trees.clear();
        let dimensions = BTreeMap::from([(SectorId::new(0), 1)]);
        let pair = |sector| FactorPair {
            sector,
            kept: 1,
            left: vec![1.0],
            left_rows: 1,
            right: vec![2.0],
            right_leading: 1,
        };

        reset_placement_index_probe();
        let error = build_bound_factor_with_placement(
            &authority,
            &homspace,
            std::slice::from_ref(&matrix),
            &mut [pair(SectorId::new(0)), pair(SectorId::new(1))],
            &dimensions,
            FactorSide::Left,
            FactorPlacement::Direct,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OperationError::UnsupportedTensorContractScope {
                message: "factor sector absent from the source tensor"
            }
        ));
        assert_eq!(placement_index_probe(), PlacementIndexProbe::default());

        let error = build_bound_factor_with_placement::<_, f64, _>(
            &authority,
            &homspace,
            std::slice::from_ref(&matrix),
            &mut [],
            &dimensions,
            FactorSide::Right,
            FactorPlacement::Direct,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OperationError::UnsupportedTensorContractScope {
                message: "factor rank absent for a populated source sector"
            }
        ));
    }

    #[test]
    fn generic_pair_publication_falls_back_for_padded_staged_geometry() {
        let (homspace, matrix) = z2_single_sector_matrix(2, 1);
        let provider = Arc::new(TestGenericRule);
        reset_generic_pair_publication_probe();
        reset_scatter_visit_probe();

        let (left, right) = build_left_right_bound_pair_generic(
            &provider,
            &homspace,
            &[matrix],
            vec![FactorPair {
                sector: SectorId::new(0),
                kept: 1,
                left: vec![2.0, 3.0],
                left_rows: 2,
                right: vec![5.0, 99.0],
                right_leading: 2,
            }],
        )
        .unwrap();

        assert_eq!(left.data(), [2.0, 3.0]);
        assert_eq!(right.data(), [5.0]);
        // G_s = 1 still pays the grouping pass (B = 1 per side) on top of the
        // F = 1 scattered block; disclosed, not dispatched on.
        assert_eq!(
            scatter_visit_probe(),
            ScatterVisitProbe {
                left_grouped: 1,
                right_grouped: 1,
                left_groups_built: 1,
                right_groups_built: 1,
                left_visits: 1,
                right_visits: 1,
            }
        );
        let probe = generic_pair_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 1)
        );
        assert_eq!(
            (
                probe.left_scattered_elements,
                probe.right_scattered_elements
            ),
            (2, 1)
        );
    }

    fn vertex_tree_factor_fixture(
        reverse: bool,
    ) -> (
        FusionTreeHomSpace,
        SectorMatricization<Complex64>,
        FactorPair<Complex64>,
    ) {
        let x = SectorId::new(1);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(x, 2)], false),
                SectorLeg::new([(x, 1)], false),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(x, 1)], false),
                SectorLeg::new([(x, 3)], false),
            ]),
        );
        let keys = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap();
        let mut row_trees = Vec::new();
        let mut col_trees = Vec::new();
        for key in keys
            .iter()
            .filter(|key| coupled_of_generic(key.codomain_tree()) == x)
        {
            if !row_trees.contains(key.codomain_tree()) {
                row_trees.push(key.codomain_tree().clone());
            }
            if !col_trees.contains(key.domain_tree()) {
                col_trees.push(key.domain_tree().clone());
            }
        }
        assert_eq!(row_trees.len(), 2);
        assert_eq!(col_trees.len(), 2);
        assert_eq!(
            row_trees
                .iter()
                .map(|tree| tree.vertices()[0].get())
                .collect::<Vec<_>>(),
            [1, 2]
        );
        assert_eq!(
            col_trees
                .iter()
                .map(|tree| tree.vertices()[0].get())
                .collect::<Vec<_>>(),
            [1, 2]
        );
        if reverse {
            row_trees.reverse();
            col_trees.reverse();
        }
        let row_trees = row_trees
            .into_iter()
            .enumerate()
            .map(|(index, tree)| (tree, 2 * index, vec![2, 1]))
            .collect();
        let col_trees = col_trees
            .into_iter()
            .enumerate()
            .map(|(index, tree)| (tree, 3 * index, vec![1, 3]))
            .collect();
        let left = (0..8)
            .map(|index| Complex64::new(10.0 + index as f64, -1.0 - index as f64 / 4.0))
            .collect::<Vec<_>>();
        let right = (0..12)
            .map(|index| Complex64::new(30.0 + index as f64, 2.0 + index as f64 / 3.0))
            .collect::<Vec<_>>();
        (
            homspace,
            SectorMatricization {
                sector: x,
                rows: 4,
                cols: 6,
                row_trees,
                col_trees,
                data: vec![Complex64::new(0.0, 0.0); 24],
            },
            FactorPair {
                sector: x,
                kept: 2,
                left,
                left_rows: 4,
                right,
                right_leading: 2,
            },
        )
    }

    #[test]
    fn one_sided_row_placement_uses_the_first_matching_duplicate() {
        let a = generic_pair(1, 1, 1).codomain_tree().clone();
        let b = generic_pair(1, 2, 1).codomain_tree().clone();
        let c = generic_pair(0, 1, 1).codomain_tree().clone();
        let matrix = SectorMatricization {
            sector: SectorId::new(1),
            rows: 7,
            cols: 0,
            row_trees: vec![
                (a.clone(), 0, vec![2]),
                (b.clone(), 2, vec![2]),
                (a.clone(), 4, vec![2]),
                (c.clone(), 6, vec![1]),
            ],
            col_trees: Vec::new(),
            data: Vec::<f64>::new(),
        };
        let index = PlacementIndex::new(std::slice::from_ref(&matrix), &[FactorSide::Left]);
        let offsets = [&b, &c, &a].map(|tree| {
            index
                .placement(SectorId::new(1), FactorSide::Left, tree)
                .unwrap()
                .0
        });

        assert_eq!(offsets, [2, 6, 0]);
    }

    #[test]
    fn placement_index_distinguishes_dual_innerline_and_vertex_trees() {
        let base = full_identity_pair(7, false, 11, false);
        let dual = full_identity_pair(7, true, 11, true);
        let inner = full_identity_pair(8, false, 12, false);
        // Same sectors, duals and inner lines as `base`; only the vertices differ.
        let vertex = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 2, 3],
            [4, 5, 6],
            9,
            [false, false, false],
            [true, false, false],
            [7],
            [11],
            [2, 2],
            [1, 1],
        )
        .unwrap();
        let (row_vertex, col_vertex) =
            (vertex.codomain_tree().clone(), vertex.domain_tree().clone());
        let matrix = SectorMatricization {
            sector: SectorId::new(9),
            rows: 4,
            cols: 4,
            row_trees: vec![
                (base.codomain_tree().clone(), 0, vec![1]),
                (dual.codomain_tree().clone(), 1, vec![1]),
                (inner.codomain_tree().clone(), 2, vec![1]),
                (row_vertex.clone(), 3, vec![1]),
            ],
            col_trees: vec![
                (base.domain_tree().clone(), 0, vec![1]),
                (dual.domain_tree().clone(), 1, vec![1]),
                (inner.domain_tree().clone(), 2, vec![1]),
                (col_vertex.clone(), 3, vec![1]),
            ],
            data: Vec::<f64>::new(),
        };
        let index = PlacementIndex::new(
            std::slice::from_ref(&matrix),
            &[FactorSide::Left, FactorSide::Right],
        );
        let sector = SectorId::new(9);
        assert_eq!(
            [
                base.codomain_tree(),
                dual.codomain_tree(),
                inner.codomain_tree(),
                &row_vertex
            ]
            .map(|tree| index.placement(sector, FactorSide::Left, tree).unwrap().0),
            [0, 1, 2, 3]
        );
        assert_eq!(
            [
                base.domain_tree(),
                dual.domain_tree(),
                inner.domain_tree(),
                &col_vertex
            ]
            .map(|tree| index.placement(sector, FactorSide::Right, tree).unwrap().0),
            [0, 1, 2, 3]
        );
        // A domain tree is never a codomain tree of this matrix and vice versa.
        assert!(matches!(
            index.placement(sector, FactorSide::Left, base.domain_tree()),
            Err(OperationError::UnsupportedTensorContractScope {
                message: "factor codomain tree absent from the source matricization"
            })
        ));
        assert!(matches!(
            index.placement(sector, FactorSide::Right, base.codomain_tree()),
            Err(OperationError::UnsupportedTensorContractScope {
                message: "factor domain tree absent from the source matricization"
            })
        ));
    }

    #[test]
    fn mf_one_sided_duplicate_source_trees_publish_the_first_entry() {
        fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
            let (homspace, mut matrix) = z2_single_sector_matrix(2, 3);
            let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
                Arc::new(Z2FusionRule),
                homspace.clone(),
            )
            .unwrap();
            let matrix = {
                let row = matrix.row_trees[0].0.clone();
                let col = matrix.col_trees[0].0.clone();
                matrix.rows = 4;
                matrix.row_trees = vec![(row.clone(), 0, vec![2]), (row, 2, vec![2])];
                matrix.cols = 6;
                matrix.col_trees = vec![(col.clone(), 0, vec![3]), (col, 3, vec![3])];
                SectorMatricization {
                    sector: matrix.sector,
                    rows: matrix.rows,
                    cols: matrix.cols,
                    row_trees: matrix.row_trees,
                    col_trees: matrix.col_trees,
                    data: Vec::<D>::new(),
                }
            };
            let source = (0..12).map(&values).collect::<Vec<_>>();
            let cases = [
                (FactorSide::Left, 2, 4, vec![0, 1, 4, 5], vec![2, 3, 6, 7]),
                (FactorSide::Right, 2, 2, (0..6).collect(), (6..12).collect()),
            ];
            for (side, bond, leading, first, second) in cases {
                let mut pair = match side {
                    FactorSide::Left => FactorPair {
                        sector: SectorId::new(0),
                        kept: 99,
                        left: source.clone(),
                        left_rows: leading,
                        right: Vec::new(),
                        right_leading: 0,
                    },
                    FactorSide::Right => FactorPair {
                        sector: SectorId::new(0),
                        kept: 99,
                        left: Vec::new(),
                        left_rows: 0,
                        right: source.clone(),
                        right_leading: leading,
                    },
                };
                reset_one_sided_publication_probe();
                reset_placement_index_probe();
                let factor = build_bound_factor_with_placement(
                    &authority,
                    &homspace,
                    std::slice::from_ref(&matrix),
                    std::slice::from_mut(&mut pair),
                    &BTreeMap::from([(SectorId::new(0), bond)]),
                    side,
                    FactorPlacement::Direct,
                )
                .unwrap();
                let pick =
                    |indices: &[usize]| indices.iter().map(|&i| source[i]).collect::<Vec<_>>();
                assert_eq!(factor.data(), pick(&first));
                assert_ne!(pick(&first), pick(&second));
                assert_eq!(one_sided_publication_probe().fallback_publications, 1);
                assert_eq!(
                    placement_index_probe(),
                    PlacementIndexProbe {
                        index_builds: 1,
                        indexed_sides: 1,
                        indexed_trees: 2,
                        lookups: 1,
                    }
                );
            }
        }
        run(|k| k as f64 + 0.5);
        run(|k| Complex64::new(k as f64 + 0.5, -(k as f64)));
    }

    /// Z_8 with every leg carrying all eight charges: every coupled sector has
    /// eight row trees and eight column trees, so F_c = T_c = 8 per sector.
    fn z8_many_tree_legs() -> [Vec<(SectorId, usize)>; 4] {
        let all = (0..8).map(|s| (SectorId::new(s), 1)).collect::<Vec<_>>();
        [all.clone(), all.clone(), all.clone(), all]
    }

    fn assert_identity_segment<D: FactorScalar + fmt::Debug>(segment: &[D], ones: usize) {
        assert_eq!(segment.iter().filter(|&&v| v == D::one()).count(), ones);
        assert!(segment.iter().all(|&v| v == D::zero() || v == D::one()));
    }

    #[test]
    fn mf_one_sided_many_tree_fallback_indexes_each_populated_sector_once() {
        fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
            let fresh = || {
                two_leg_geometry::<_, D>(
                    Arc::new(tenet_core::ZNFusionRule::new(8).unwrap()),
                    z8_many_tree_legs(),
                )
            };
            let (authority, matrices) = fresh();
            assert_eq!(matrices.len(), 8);
            assert!(matrices
                .iter()
                .all(|m| m.row_trees.len() == 8 && m.col_trees.len() == 8));
            for (side, placement) in [
                (FactorSide::Left, FactorPlacement::Direct),
                (FactorSide::Right, FactorPlacement::Direct),
            ] {
                let source_trees = source_trees_for(side, placement);
                let dimensions = matrices
                    .iter()
                    .map(|m| (m.sector, source_extent(m, source_trees)))
                    .collect::<BTreeMap<_, _>>();
                let build = |matrices: &[SectorMatricization<D>], pairs: &mut [FactorPair<D>]| {
                    reset_one_sided_publication_probe();
                    reset_placement_index_probe();
                    let factor = build_bound_factor_with_placement(
                        &authority,
                        authority.space().homspace(),
                        matrices,
                        pairs,
                        &dimensions,
                        side,
                        placement,
                    )
                    .unwrap();
                    (
                        factor.data().to_vec(),
                        one_sided_publication_probe(),
                        placement_index_probe(),
                    )
                };
                let staged = |matrices: &[SectorMatricization<D>]| {
                    staged_one_sided_pairs(matrices, &dimensions, side, source_trees, &values)
                };

                let mut pairs = staged(&matrices);
                let (reference, probe, index) = build(&matrices, &mut pairs);
                assert_eq!(probe.canonical_publications, 1);
                assert_eq!(index, PlacementIndexProbe::default());

                // All eight sectors populated, reversed: F = T = 64 over
                // G_s = 8 in one table (formerly one per sector), so 128
                // hashes replace the former 8 * 8 * 8 = 512 key comparisons.
                let (_, mut reversed) = fresh();
                reversed.reverse();
                let mut pairs = staged(&reversed);
                let (data, probe, index) = build(&reversed, &mut pairs);
                assert_eq!(probe.fallback_publications, 1);
                assert_eq!(data, reference);
                assert_eq!(
                    index,
                    PlacementIndexProbe {
                        index_builds: 1,
                        indexed_sides: 8,
                        indexed_trees: 64,
                        lookups: 64,
                    }
                );
                assert!(index.indexed_trees + index.lookups < 8 * 8 * 8);

                // Identity-only sectors 0 (before), 3 (between) and 7 (after)
                // have no matricization and index nothing.
                let (_, kept) = fresh();
                let kept = kept
                    .into_iter()
                    .filter(|m| ![0, 3, 7].contains(&m.sector.id()))
                    .collect::<Vec<_>>();
                let mut pairs = staged(&kept);
                let (data, probe, index) = build(&kept, &mut pairs);
                assert_eq!(probe.fallback_publications, 1);
                assert_eq!(
                    index,
                    PlacementIndexProbe {
                        index_builds: 1,
                        indexed_sides: 5,
                        indexed_trees: 40,
                        lookups: 40,
                    }
                );
                let mut start = 0usize;
                for matrix in &matrices {
                    let len = source_extent(matrix, source_trees) * dimensions[&matrix.sector];
                    let segment = &data[start..start + len];
                    if [0, 3, 7].contains(&matrix.sector.id()) {
                        assert_identity_segment(segment, dimensions[&matrix.sector]);
                    } else {
                        assert_eq!(segment, &reference[start..start + len]);
                    }
                    start += len;
                }
                assert_eq!(start, data.len());
            }
        }
        run(|k| k as f64 + 0.25);
        run(|k| Complex64::new(k as f64 + 0.25, 0.5 - k as f64));
    }

    /// Three `{0, 1}` legs per side under the multiplicity-two test rule:
    /// coupled sector 1 has 14 trees per side and sector 0 has 6, counting
    /// inner lines and vertices, so F_c = T_c per sector (unit degeneracies).
    #[test]
    fn checked_one_sided_many_tree_publication_reuses_one_index_per_sector() {
        fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
            let rule = TestGenericRule;
            let provider = Arc::new(InfallibleGeneric::new(&rule));
            let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
            let homspace = FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(), leg(), leg()]),
                FusionProductSpace::new([leg(), leg(), leg()]),
            );
            let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
                Arc::clone(&provider),
                homspace.clone(),
            )
            .unwrap();
            let zeros = vec![D::zero(); space.space().required_len().unwrap()];
            let fresh = || sector_matricizations(space.space().structure(), &zeros, 3).unwrap();
            let matrices = fresh();
            assert_eq!(
                matrices
                    .iter()
                    .map(|m| (m.sector.id(), m.row_trees.len(), m.col_trees.len()))
                    .collect::<Vec<_>>(),
                [(0, 6, 6), (1, 14, 14)]
            );
            for side in [FactorSide::Left, FactorSide::Right] {
                let dimensions = matrices
                    .iter()
                    .map(|m| (m.sector, source_extent(m, side)))
                    .collect::<BTreeMap<_, _>>();
                let build = |matrices: &[SectorMatricization<D>], pairs: &mut [FactorPair<D>]| {
                    reset_one_sided_publication_probe();
                    reset_placement_index_probe();
                    let factor = build_bound_factor_generic_checked(
                        &provider,
                        &homspace,
                        matrices,
                        pairs,
                        &dimensions,
                        side,
                    )
                    .unwrap();
                    (
                        factor.data().to_vec(),
                        one_sided_publication_probe(),
                        placement_index_probe(),
                    )
                };
                let staged = |matrices: &[SectorMatricization<D>]| {
                    staged_one_sided_pairs(matrices, &dimensions, side, side, &values)
                };

                // Canonical transfer: prevalidation alone, one lookup per key.
                let mut pairs = staged(&matrices);
                let (reference, probe, index) = build(&matrices, &mut pairs);
                assert_eq!(probe.canonical_publications, 1);
                assert_eq!(
                    index,
                    PlacementIndexProbe {
                        index_builds: 1,
                        indexed_sides: 2,
                        indexed_trees: 20,
                        lookups: 20,
                    }
                );

                // Fallback: the single prevalidation table is reused (formerly
                // one table per sector, G_s = 2), so lookups double without a
                // rebuild; 60 hashes replace the former
                // 2 * (6 * 6 + 14 * 14) = 464 key comparisons.
                let mut reversed = fresh();
                reversed.reverse();
                let mut pairs = staged(&reversed);
                let (data, probe, index) = build(&reversed, &mut pairs);
                assert_eq!(probe.fallback_publications, 1);
                assert_eq!(data, reference);
                assert_eq!(
                    index,
                    PlacementIndexProbe {
                        index_builds: 1,
                        indexed_sides: 2,
                        indexed_trees: 20,
                        lookups: 40,
                    }
                );
                assert!(index.indexed_trees + index.lookups < 2 * (6 * 6 + 14 * 14));

                // Identity-only sector after (keep 0) and before (keep 1) the
                // populated sector: only the populated sector is indexed.
                let lens = matrices
                    .iter()
                    .map(|m| source_extent(m, side) * dimensions[&m.sector])
                    .collect::<Vec<_>>();
                for keep in [0usize, 1] {
                    let kept = fresh()
                        .into_iter()
                        .filter(|m| m.sector.id() == keep)
                        .collect::<Vec<_>>();
                    let mut pairs = staged(&kept);
                    let (data, probe, index) = build(&kept, &mut pairs);
                    assert_eq!(probe.fallback_publications, 1);
                    assert_eq!(
                        index,
                        PlacementIndexProbe {
                            index_builds: 1,
                            indexed_sides: 1,
                            indexed_trees: kept[0].row_trees.len(),
                            lookups: 2 * kept[0].row_trees.len(),
                        }
                    );
                    let (first, second) = data.split_at(lens[0]);
                    let (reference_first, reference_second) = reference.split_at(lens[0]);
                    assert_eq!(second.len(), lens[1]);
                    if keep == 0 {
                        assert_eq!(first, reference_first);
                        assert_identity_segment(second, dimensions[&SectorId::new(1)]);
                    } else {
                        assert_identity_segment(first, dimensions[&SectorId::new(0)]);
                        assert_eq!(second, reference_second);
                    }
                }
            }
        }
        run(|k| k as f64 + 0.25);
        run(|k| Complex64::new(k as f64 + 0.25, 0.5 - k as f64));
    }

    #[test]
    fn generic_pair_fallback_validation_indexes_each_sector_once() {
        let (sector_count, trees_per_sector) = (4usize, 8usize);
        let mut matrices = Vec::new();
        let mut left_keys = Vec::new();
        let mut right_keys = Vec::new();
        for sector in 0..sector_count {
            let bond = generic_pair(sector, 1, 1).codomain_tree().clone();
            let mut row_trees = Vec::new();
            let mut col_trees = Vec::new();
            for tree_index in 0..trees_per_sector {
                let source = generic_pair(sector, tree_index + 1, tree_index + 1);
                left_keys.push(FusionTreePairKey::pair(
                    source.codomain_tree().clone(),
                    bond.clone(),
                ));
                right_keys.push(FusionTreePairKey::pair(
                    bond.clone(),
                    source.domain_tree().clone(),
                ));
                row_trees.push((source.codomain_tree().clone(), tree_index, vec![1, 1]));
                col_trees.push((source.domain_tree().clone(), tree_index, vec![1, 1]));
            }
            matrices.push(SectorMatricization {
                sector: SectorId::new(sector),
                rows: trees_per_sector,
                cols: trees_per_sector,
                row_trees,
                col_trees,
                data: Vec::<f64>::new(),
            });
        }
        // Interleave the key order across sectors so each sector's index is
        // reused across non-adjacent keys.
        left_keys.reverse();
        right_keys.reverse();
        let total = sector_count * trees_per_sector;
        for (side, keys) in [
            (FactorSide::Left, &left_keys),
            (FactorSide::Right, &right_keys),
        ] {
            reset_generic_pair_publication_probe();
            reset_placement_index_probe();
            let mut matrix_by_sector = None;
            assert!(validate_generic_factor_keys(
                keys,
                side,
                None,
                &matrices,
                &mut matrix_by_sector
            )
            .unwrap());
            let probe = generic_pair_publication_probe();
            let lookups = match side {
                FactorSide::Left => probe.fallback_row_lookups,
                FactorSide::Right => probe.fallback_col_lookups,
            };
            assert_eq!(lookups, total);
            let index = placement_index_probe();
            assert_eq!(
                index,
                PlacementIndexProbe {
                    index_builds: 1,
                    indexed_sides: sector_count,
                    indexed_trees: total,
                    lookups: total,
                }
            );
            // F + T = 64 hashes versus the former F * T_c = 32 * 8 = 256
            // comparisons.
            assert!(index.indexed_trees + index.lookups < total * trees_per_sector);
        }
    }

    #[test]
    fn checked_one_sided_placement_preserves_both_sides_and_tree_orders() {
        let rule = TestGenericRule;
        let provider = Arc::new(InfallibleGeneric::new(&rule));
        let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);

        for reverse in [false, true] {
            let (homspace, matrix, pair) = vertex_tree_factor_fixture(reverse);
            for side in [FactorSide::Left, FactorSide::Right] {
                let (_, _, mut staged) = vertex_tree_factor_fixture(reverse);
                let factor = build_bound_factor_generic_checked(
                    &provider,
                    &homspace,
                    std::slice::from_ref(&matrix),
                    std::slice::from_mut(&mut staged),
                    &dimensions,
                    side,
                )
                .unwrap();
                let expected = match (side, reverse) {
                    (FactorSide::Left, false) => pair.left.clone(),
                    (FactorSide::Right, false) => pair.right.clone(),
                    (FactorSide::Left, true) => vec![
                        pair.left[2],
                        pair.left[3],
                        pair.left[0],
                        pair.left[1],
                        pair.left[6],
                        pair.left[7],
                        pair.left[4],
                        pair.left[5],
                    ],
                    (FactorSide::Right, true) => vec![
                        pair.right[6],
                        pair.right[7],
                        pair.right[8],
                        pair.right[9],
                        pair.right[10],
                        pair.right[11],
                        pair.right[0],
                        pair.right[1],
                        pair.right[2],
                        pair.right[3],
                        pair.right[4],
                        pair.right[5],
                    ],
                };
                assert_eq!(factor.data(), expected);
                assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));
                let selected_vertices = (0..factor.space().space().structure().block_count())
                    .map(|index| factor.space().space().structure().block(index).unwrap())
                    .map(|block| match block.key() {
                        BlockKey::FusionTree(key) => match side {
                            FactorSide::Left => key.codomain_tree().vertices()[0].get(),
                            FactorSide::Right => key.domain_tree().vertices()[0].get(),
                        },
                        _ => unreachable!("checked factor has fusion-tree keys"),
                    })
                    .collect::<Vec<_>>();
                assert_eq!(selected_vertices, [1, 2]);
            }
        }
    }

    #[test]
    fn checked_one_sided_reports_missing_pairs_and_full_trees() {
        let rule = TestGenericRule;
        let provider = Arc::new(InfallibleGeneric::new(&rule));
        let (homspace, matrix, _) = vertex_tree_factor_fixture(false);
        let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
        for side in [FactorSide::Left, FactorSide::Right] {
            reset_placement_index_probe();
            let error = build_bound_factor_generic_checked::<_, Complex64, _>(
                &provider,
                &homspace,
                std::slice::from_ref(&matrix),
                &mut [],
                &dimensions,
                side,
            )
            .unwrap_err();
            assert!(matches!(
                error,
                CheckedGenericFactorPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope {
                        message: "factor rank absent for a populated source sector"
                    }
                )
            ));
            // Prevalidation indexed the sole sector and looked up both keys;
            // the fallback rejected the missing pair before its own lookup.
            assert_eq!(
                placement_index_probe(),
                PlacementIndexProbe {
                    index_builds: 1,
                    indexed_sides: 1,
                    indexed_trees: 2,
                    lookups: 2,
                }
            );
        }

        for side in [FactorSide::Left, FactorSide::Right] {
            let (_, mut wrong, mut pair) = vertex_tree_factor_fixture(false);
            match side {
                FactorSide::Left => wrong.row_trees[0].0 = wrong.row_trees[1].0.clone(),
                FactorSide::Right => wrong.col_trees[0].0 = wrong.col_trees[1].0.clone(),
            }
            let error = build_bound_factor_generic_checked(
                &provider,
                &homspace,
                std::slice::from_ref(&wrong),
                std::slice::from_mut(&mut pair),
                &dimensions,
                side,
            )
            .unwrap_err();
            let expected = match side {
                FactorSide::Left => "factor codomain tree absent from the source matricization",
                FactorSide::Right => "factor domain tree absent from the source matricization",
            };
            assert!(matches!(
                error,
                CheckedGenericFactorPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope { message }
                ) if message == expected
            ));
        }
    }

    #[test]
    fn generic_pair_publication_preserves_vertex_tree_payload_placement() {
        let provider = Arc::new(TestGenericRule);
        let (homspace, matrix, pair) = vertex_tree_factor_fixture(false);
        let expected_left = pair.left.clone();
        let expected_right = pair.right.clone();
        reset_generic_pair_publication_probe();
        let (left, right) =
            build_left_right_bound_pair_generic(&provider, &homspace, &[matrix], vec![pair])
                .unwrap();
        assert_eq!(left.data(), expected_left);
        assert_eq!(right.data(), expected_right);
        let probe = generic_pair_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );
        assert_eq!(
            (probe.fallback_row_lookups, probe.fallback_col_lookups),
            (0, 0)
        );
        assert_eq!(
            (probe.left_scatter_calls, probe.right_scatter_calls),
            (0, 0)
        );
        assert_eq!(probe.output_blocks_visited, 4);
        assert_eq!(
            probe.ordered_key_validation_events,
            2 * probe.output_blocks_visited
        );

        let (homspace, matrix, pair) = vertex_tree_factor_fixture(true);
        let left_source = pair.left.clone();
        let right_source = pair.right.clone();
        reset_generic_pair_publication_probe();
        reset_placement_index_probe();
        let (left, right) =
            build_left_right_bound_pair_generic(&provider, &homspace, &[matrix], vec![pair])
                .unwrap();
        assert_eq!(
            left.data(),
            [
                left_source[2],
                left_source[3],
                left_source[0],
                left_source[1],
                left_source[6],
                left_source[7],
                left_source[4],
                left_source[5],
            ]
        );
        assert_eq!(
            right.data(),
            [
                right_source[6],
                right_source[7],
                right_source[8],
                right_source[9],
                right_source[10],
                right_source[11],
                right_source[0],
                right_source[1],
                right_source[2],
                right_source[3],
                right_source[4],
                right_source[5],
            ]
        );
        let probe = generic_pair_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 1)
        );
        assert_eq!(
            (probe.fallback_row_lookups, probe.fallback_col_lookups),
            (2, 2)
        );
        assert_eq!(
            (probe.left_scatter_calls, probe.right_scatter_calls),
            (2, 2)
        );
        // Row and column key validation each build a one-sided table; the
        // scatter loop shares one two-sided table (formerly four tables).
        assert_eq!(
            placement_index_probe(),
            PlacementIndexProbe {
                index_builds: 3,
                indexed_sides: 4,
                indexed_trees: 8,
                lookups: 8,
            }
        );
    }

    #[test]
    fn generic_pair_canonical_validation_scales_linearly_in_sectors_and_trees() {
        for (sector_count, trees_per_sector) in [(1, 1), (1, 4), (4, 1), (4, 3)] {
            let mut matrices = Vec::with_capacity(sector_count);
            let mut pairs = Vec::with_capacity(sector_count);
            let mut left_keys = Vec::with_capacity(sector_count * trees_per_sector);
            let mut right_keys = Vec::with_capacity(sector_count * trees_per_sector);
            let mut left_blocks = Vec::with_capacity(sector_count * trees_per_sector);
            let mut right_blocks = Vec::with_capacity(sector_count * trees_per_sector);
            let mut output_offset = 0usize;
            for sector in 0..sector_count {
                let bond = FusionTreePairKey::try_pair_from_sector_ids(
                    [sector],
                    [sector],
                    sector,
                    [false],
                    [false],
                    [],
                    [],
                    [],
                    [],
                )
                .unwrap()
                .codomain_tree()
                .clone();
                let mut row_trees = Vec::with_capacity(trees_per_sector);
                let mut col_trees = Vec::with_capacity(trees_per_sector);
                for tree_index in 0..trees_per_sector {
                    let source = generic_pair(sector, tree_index + 1, tree_index + 1);
                    let row_tree = source.codomain_tree().clone();
                    let col_tree = source.domain_tree().clone();
                    let left_key = FusionTreePairKey::pair(row_tree.clone(), bond.clone());
                    let right_key = FusionTreePairKey::pair(bond.clone(), col_tree.clone());
                    left_blocks.push(
                        BlockSpec::with_key(
                            left_key.clone().into(),
                            vec![1, 1, 1],
                            vec![1, 1, trees_per_sector],
                            output_offset + tree_index,
                        )
                        .unwrap(),
                    );
                    right_blocks.push(
                        BlockSpec::with_key(
                            right_key.clone().into(),
                            vec![1, 1, 1],
                            vec![1, 1, 1],
                            output_offset + tree_index,
                        )
                        .unwrap(),
                    );
                    left_keys.push(left_key);
                    right_keys.push(right_key);
                    row_trees.push((row_tree, tree_index, vec![1, 1]));
                    col_trees.push((col_tree, tree_index, vec![1, 1]));
                }
                let sector = SectorId::new(sector);
                matrices.push(SectorMatricization {
                    sector,
                    rows: trees_per_sector,
                    cols: trees_per_sector,
                    row_trees,
                    col_trees,
                    data: vec![0.0; trees_per_sector * trees_per_sector],
                });
                pairs.push(FactorPair {
                    sector,
                    kept: 1,
                    left: (0..trees_per_sector)
                        .map(|index| 100.0 * sector.id() as f64 + index as f64)
                        .collect(),
                    left_rows: trees_per_sector,
                    right: (0..trees_per_sector)
                        .map(|index| -100.0 * sector.id() as f64 - index as f64)
                        .collect(),
                    right_leading: 1,
                });
                output_offset += trees_per_sector;
            }
            let ranks = pairs
                .iter()
                .map(|pair| SectorRank {
                    sector: pair.sector,
                    kept: pair.kept,
                })
                .collect::<Vec<_>>();
            let left_structure = BlockStructure::from_blocks_with_rank(3, left_blocks).unwrap();
            let right_structure = BlockStructure::from_blocks_with_rank(3, right_blocks).unwrap();
            let expected_left = pairs
                .iter()
                .flat_map(|pair| pair.left.iter().copied())
                .collect::<Vec<_>>();
            let expected_right = pairs
                .iter()
                .flat_map(|pair| pair.right.iter().copied())
                .collect::<Vec<_>>();

            reset_generic_pair_publication_probe();
            let mut matrix_by_sector = None;
            let mut left_cursor = FactorTreeCursor::new(&matrices, &ranks);
            assert!(validate_generic_factor_keys(
                &left_keys,
                FactorSide::Left,
                Some(&mut left_cursor),
                &matrices,
                &mut matrix_by_sector,
            )
            .unwrap());
            let mut right_cursor = FactorTreeCursor::new(&matrices, &ranks);
            assert!(validate_generic_factor_keys(
                &right_keys,
                FactorSide::Right,
                Some(&mut right_cursor),
                &matrices,
                &mut matrix_by_sector,
            )
            .unwrap());
            assert!(factor_output_is_canonical(
                &left_structure,
                Some(&left_keys),
                &matrices,
                &pairs,
                expected_left.len(),
                FactorSide::Left,
            ));
            assert!(factor_output_is_canonical(
                &right_structure,
                Some(&right_keys),
                &matrices,
                &pairs,
                expected_right.len(),
                FactorSide::Right,
            ));
            let output_blocks = left_keys.len() + right_keys.len();
            let probe = generic_pair_publication_probe();
            assert_eq!(probe.output_blocks_visited, output_blocks);
            assert_eq!(probe.ordered_key_validation_events, 2 * output_blocks);
            assert_eq!(
                (probe.fallback_row_lookups, probe.fallback_col_lookups),
                (0, 0)
            );
            let (left, right) =
                publish_generic_factor_pairs(pairs, expected_left.len(), expected_right.len());
            assert_eq!(left, expected_left);
            assert_eq!(right, expected_right);
        }
    }

    #[test]
    fn generic_pair_validation_preserves_missing_sector_and_tree_errors() {
        let (homspace, matrix) = z2_single_sector_matrix(1, 1);
        let provider = Arc::new(TestGenericRule);
        let pair = || FactorPair {
            sector: SectorId::new(0),
            kept: 1,
            left: vec![1.0],
            left_rows: 1,
            right: vec![1.0],
            right_leading: 1,
        };

        let missing_sector = build_left_right_bound_pair_generic(
            &provider,
            &homspace,
            &[] as &[SectorMatricization<f64>],
            vec![pair()],
        )
        .unwrap_err();
        assert!(matches!(
            missing_sector,
            OperationError::UnsupportedTensorContractScope {
                message: "factor tree references a coupled sector absent from the source tensor"
            }
        ));

        let wrong = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0],
            0,
            [false],
            [false],
            [],
            [],
            [],
            [],
        )
        .unwrap()
        .codomain_tree()
        .clone();
        let mut wrong_row = matrix;
        wrong_row.row_trees[0].0 = wrong.clone();
        wrong_row.col_trees[0].0 = wrong.clone();
        let missing_row =
            build_left_right_bound_pair_generic(&provider, &homspace, &[wrong_row], vec![pair()])
                .unwrap_err();
        assert!(matches!(
            missing_row,
            OperationError::UnsupportedTensorContractScope {
                message: "factor codomain tree absent from the source matricization"
            }
        ));

        let (_, mut wrong_col) = z2_single_sector_matrix(1, 1);
        wrong_col.col_trees[0].0 = wrong;
        let missing_col =
            build_left_right_bound_pair_generic(&provider, &homspace, &[wrong_col], vec![pair()])
                .unwrap_err();
        assert!(matches!(
            missing_col,
            OperationError::UnsupportedTensorContractScope {
                message: "factor domain tree absent from the source matricization"
            }
        ));
    }

    #[test]
    fn generic_pair_publication_handles_zero_kept_sector() {
        let (homspace, matrix) = z2_single_sector_matrix(2, 1);
        reset_generic_pair_publication_probe();
        let (left, right) = build_left_right_bound_pair_generic(
            &Arc::new(TestGenericRule),
            &homspace,
            &[matrix],
            vec![FactorPair {
                sector: SectorId::new(0),
                kept: 0,
                left: Vec::<f64>::new(),
                left_rows: 2,
                right: Vec::new(),
                right_leading: 0,
            }],
        )
        .unwrap();
        assert!(left.data().is_empty());
        assert!(right.data().is_empty());
        assert_eq!(
            generic_pair_publication_probe(),
            GenericPairPublicationProbe {
                canonical_publications: 1,
                ..GenericPairPublicationProbe::default()
            }
        );

        let empty = FusionProductSpace::new(std::iter::empty::<SectorLeg>());
        let empty_hom = FusionTreeHomSpace::new(empty.clone(), empty);
        let (left, right) =
            build_left_right_bound_pair_generic::<_, f64, SectorMatricization<f64>>(
                &Arc::new(TestGenericRule),
                &empty_hom,
                &[],
                Vec::new(),
            )
            .unwrap();
        assert!(left.data().is_empty());
        assert!(right.data().is_empty());
    }

    #[test]
    fn generic_pair_publication_handles_scalar_matrix() {
        let empty = FusionProductSpace::new(std::iter::empty::<SectorLeg>());
        let homspace = FusionTreeHomSpace::new(empty.clone(), empty);
        let key = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap()[0].clone();
        reset_generic_pair_publication_probe();
        let (left, right) = build_left_right_bound_pair_generic(
            &Arc::new(TestGenericRule),
            &homspace,
            &[SectorMatricization {
                sector: SectorId::new(0),
                rows: 1,
                cols: 1,
                row_trees: vec![(key.codomain_tree().clone(), 0, Vec::new())],
                col_trees: vec![(key.domain_tree().clone(), 0, Vec::new())],
                data: vec![7.0],
            }],
            vec![FactorPair {
                sector: SectorId::new(0),
                kept: 1,
                left: vec![2.0],
                left_rows: 1,
                right: vec![3.5],
                right_leading: 1,
            }],
        )
        .unwrap();
        assert_eq!(left.data(), [2.0]);
        assert_eq!(right.data(), [3.5]);
        let probe = generic_pair_publication_probe();
        assert_eq!(probe.canonical_publications, 1);
        assert_eq!((probe.left_owner_reused, probe.right_owner_reused), (1, 1));
    }

    /// Two-leg codomain/domain hom space matricized exactly as production
    /// does, so tree orders and offsets are the real ones.
    fn two_leg_geometry<R, D>(
        rule: Arc<R>,
        legs: [Vec<(SectorId, usize)>; 4],
    ) -> (BoundDynamicFusionMapSpace<R>, Vec<SectorMatricization<D>>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
        D: FactorScalar,
    {
        let [a, b, c, d] = legs;
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(a, false), SectorLeg::new(b, false)]),
            FusionProductSpace::new([SectorLeg::new(c, false), SectorLeg::new(d, false)]),
        );
        let authority =
            BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(rule, homspace)
                .unwrap();
        let data = vec![D::zero(); authority.space().required_len().unwrap()];
        let matrices = sector_matricizations(authority.space().structure(), &data, 2).unwrap();
        (authority, matrices)
    }

    fn z2_two_sector_geometry<D: FactorScalar>() -> (
        BoundDynamicFusionMapSpace<Z2FusionRule>,
        Vec<SectorMatricization<D>>,
    ) {
        let even = SectorId::new(0);
        let odd = SectorId::new(1);
        two_leg_geometry(
            Arc::new(Z2FusionRule),
            [
                vec![(even, 2), (odd, 1)],
                vec![(even, 1), (odd, 3)],
                vec![(even, 1), (odd, 2)],
                vec![(even, 2), (odd, 1)],
            ],
        )
    }

    fn source_trees_for(side: FactorSide, placement: FactorPlacement) -> FactorSide {
        match (side, placement) {
            (FactorSide::Left, FactorPlacement::Direct)
            | (FactorSide::Right, FactorPlacement::Adjoint) => FactorSide::Left,
            (FactorSide::Right, FactorPlacement::Direct)
            | (FactorSide::Left, FactorPlacement::Adjoint) => FactorSide::Right,
        }
    }

    fn source_extent<D>(matrix: &SectorMatricization<D>, source_trees: FactorSide) -> usize {
        match source_trees {
            FactorSide::Left => matrix.rows,
            FactorSide::Right => matrix.cols,
        }
    }

    /// Stages `a x b` (left) or `b x a` (right) selected factors per sector
    /// with `kept` deliberately unrelated to the admitted bond; the opposite
    /// side carries a sentinel payload that must survive publication.
    fn staged_one_sided_pairs<D: FactorScalar>(
        matrices: &[SectorMatricization<D>],
        dimensions: &BTreeMap<SectorId, usize>,
        side: FactorSide,
        source_trees: FactorSide,
        values: &dyn Fn(usize) -> D,
    ) -> Vec<FactorPair<D>> {
        matrices
            .iter()
            .map(|matrix| {
                let a = source_extent(matrix, source_trees);
                let b = dimensions[&matrix.sector];
                let tag = matrix.sector.id() + 1;
                let selected = (0..a * b)
                    .map(|k| values(1000 * tag + k))
                    .collect::<Vec<_>>();
                let opposite = (0..3).map(|k| values(7 * tag + k)).collect();
                match side {
                    FactorSide::Left => FactorPair {
                        sector: matrix.sector,
                        kept: 99,
                        left: selected,
                        left_rows: a,
                        right: opposite,
                        right_leading: 5,
                    },
                    FactorSide::Right => FactorPair {
                        sector: matrix.sector,
                        kept: 99,
                        left: opposite,
                        left_rows: 5,
                        right: selected,
                        right_leading: b,
                    },
                }
            })
            .collect()
    }

    fn selected_of<D>(pair: &FactorPair<D>, side: FactorSide) -> &Vec<D> {
        match side {
            FactorSide::Left => &pair.left,
            FactorSide::Right => &pair.right,
        }
    }

    fn opposite_of<D>(pair: &FactorPair<D>, side: FactorSide) -> &Vec<D> {
        match side {
            FactorSide::Left => &pair.right,
            FactorSide::Right => &pair.left,
        }
    }

    /// Checks every output element against the literal coordinates
    /// `F[o + q + a*j]` (left) / `F[j + b*(o + q)]` (right), independent of
    /// the production layout proof.
    fn assert_literal_one_sided_layout<D>(
        structure: &BlockStructure,
        data: &[D],
        matrices: &[SectorMatricization<D>],
        selected: &[Vec<D>],
        dimensions: &BTreeMap<SectorId, usize>,
        side: FactorSide,
        source_trees: FactorSide,
    ) where
        D: FactorScalar + PartialEq + fmt::Debug,
    {
        let mut verified = 0usize;
        for index in 0..structure.block_count() {
            let block = structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                panic!("factor blocks carry fusion-tree keys")
            };
            let tree = match side {
                FactorSide::Left => key.codomain_tree(),
                FactorSide::Right => key.domain_tree(),
            };
            let (matrix_index, matrix) = matrices
                .iter()
                .enumerate()
                .find(|(_, matrix)| matrix.sector == tree.coupled())
                .unwrap();
            let trees = match source_trees {
                FactorSide::Left => &matrix.row_trees,
                FactorSide::Right => &matrix.col_trees,
            };
            let (_, o, tree_shape) = trees
                .iter()
                .find(|(candidate, _, _)| candidate == tree)
                .unwrap();
            let a = source_extent(matrix, source_trees);
            let b = dimensions[&matrix.sector];
            let factor = &selected[matrix_index];
            let shape = block.shape();
            let strides = block.strides();
            let total = shape.iter().product::<usize>();
            assert_eq!(total, tree_shape.iter().product::<usize>() * b);
            for flat in 0..total {
                let mut remaining = flat;
                let mut destination = block.offset();
                let mut coordinates = vec![0usize; shape.len()];
                for axis in 0..shape.len() {
                    coordinates[axis] = remaining % shape[axis];
                    remaining /= shape[axis];
                    destination += coordinates[axis] * strides[axis];
                }
                let tree_axes = match side {
                    FactorSide::Left => 0..shape.len() - 1,
                    FactorSide::Right => 1..shape.len(),
                };
                let mut q = 0usize;
                let mut span = 1usize;
                for axis in tree_axes {
                    q += coordinates[axis] * span;
                    span *= shape[axis];
                }
                let j = match side {
                    FactorSide::Left => coordinates[shape.len() - 1],
                    FactorSide::Right => coordinates[0],
                };
                let expected = match side {
                    FactorSide::Left => factor[o + q + a * j],
                    FactorSide::Right => factor[j + b * (o + q)],
                };
                assert_eq!(data[destination], expected);
                verified += 1;
            }
        }
        assert_eq!(verified, data.len());
    }

    fn mf_one_sided_canonical_transfer_case<D>(values: &dyn Fn(usize) -> D)
    where
        D: FactorScalar + PartialEq + fmt::Debug,
    {
        let (authority, matrices) = z2_two_sector_geometry::<D>();
        let adjoint = tenet_tensors::adjoint_bound_space_dyn(&authority).unwrap();
        let row_dimensions = matrices
            .iter()
            .map(|matrix| (matrix.sector, matrix.rows))
            .collect::<BTreeMap<_, _>>();
        let col_dimensions = matrices
            .iter()
            .map(|matrix| (matrix.sector, matrix.cols))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(matrices.len(), 2);
        assert!(matrices.iter().all(|matrix| matrix.rows != matrix.cols
            && matrix.row_trees.len() == 2
            && matrix.col_trees.len() == 2));
        assert_ne!(matrices[0].rows, matrices[1].rows);

        let cases = [
            (&authority, FactorSide::Left, FactorPlacement::Direct),
            (&authority, FactorSide::Right, FactorPlacement::Direct),
            (&adjoint, FactorSide::Left, FactorPlacement::Adjoint),
            (&adjoint, FactorSide::Right, FactorPlacement::Adjoint),
        ];
        for (space, side, placement) in cases {
            let source_trees = source_trees_for(side, placement);
            let dimensions = match source_trees {
                FactorSide::Left => &row_dimensions,
                FactorSide::Right => &col_dimensions,
            };
            let mut pairs =
                staged_one_sided_pairs(&matrices, dimensions, side, source_trees, values);
            let selected = pairs
                .iter()
                .map(|pair| selected_of(pair, side).clone())
                .collect::<Vec<_>>();
            let opposite = pairs
                .iter()
                .map(|pair| opposite_of(pair, side).clone())
                .collect::<Vec<_>>();

            reset_one_sided_publication_probe();
            let factor = build_bound_factor_with_placement(
                space,
                space.space().homspace(),
                &matrices,
                &mut pairs,
                dimensions,
                side,
                placement,
            )
            .unwrap();

            let probe = one_sided_publication_probe();
            assert_eq!(
                (probe.canonical_publications, probe.fallback_publications),
                (1, 0)
            );
            assert_eq!(probe.appended_elements, selected[1].len());
            assert_literal_one_sided_layout(
                factor.space().space().structure(),
                factor.data(),
                &matrices,
                &selected,
                dimensions,
                side,
                source_trees,
            );
            for (index, pair) in pairs.iter().enumerate() {
                assert!(selected_of(pair, side).is_empty());
                assert_eq!(opposite_of(pair, side), &opposite[index]);
                assert_eq!((pair.kept, pair.left_rows, pair.right_leading).0, 99);
            }
        }
    }

    #[test]
    fn mf_one_sided_canonical_transfer_real_full_svd_bonds() {
        mf_one_sided_canonical_transfer_case(&|k| k as f64 * 0.5 - 7.0);
    }

    #[test]
    fn mf_one_sided_canonical_transfer_complex_full_svd_bonds() {
        mf_one_sided_canonical_transfer_case(&|k| Complex64::new(k as f64, -(k as f64) / 3.0));
    }

    fn assert_buffers_intact<D>(pairs: &[FactorPair<D>], side: FactorSide) {
        assert!(pairs.iter().all(|pair| !selected_of(pair, side).is_empty()));
    }

    #[test]
    fn mf_one_sided_fallbacks_match_canonical_output_and_keep_buffers() {
        let (authority, matrices) = z2_two_sector_geometry::<f64>();
        let homspace = authority.space().homspace();
        let side = FactorSide::Left;
        let placement = FactorPlacement::Direct;
        let row_dimensions = matrices
            .iter()
            .map(|matrix| (matrix.sector, matrix.rows))
            .collect::<BTreeMap<_, _>>();
        let staged = |scale: f64| {
            staged_one_sided_pairs(&matrices, &row_dimensions, side, side, &|k| {
                scale * (k as f64 + 1.0)
            })
        };
        let build = |matrices: &[SectorMatricization<f64>],
                     pairs: &mut [FactorPair<f64>],
                     dimensions: &BTreeMap<SectorId, usize>| {
            reset_one_sided_publication_probe();
            let result = build_bound_factor_with_placement(
                &authority, homspace, matrices, pairs, dimensions, side, placement,
            );
            (result, one_sided_publication_probe())
        };
        let canonical = |scale: f64| {
            let mut pairs = staged(scale);
            let (factor, probe) = build(&matrices, &mut pairs, &row_dimensions);
            assert_eq!(
                (probe.canonical_publications, probe.fallback_publications),
                (1, 0)
            );
            factor.unwrap().data().to_vec()
        };
        let reference = canonical(1.0);
        let sector_len = matrices[0].rows * matrices[0].rows;

        // Reordered matricizations: same admitted output, scatter path.
        let (_, mut reversed) = z2_two_sector_geometry::<f64>();
        reversed.reverse();
        let mut pairs = staged(1.0);
        pairs.reverse();
        let (factor, probe) = build(&reversed, &mut pairs, &row_dimensions);
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 1)
        );
        assert_eq!(factor.unwrap().data(), reference);
        assert_buffers_intact(&pairs, side);

        // Reordered pairs alone.
        let mut pairs = staged(1.0);
        pairs.reverse();
        let (factor, probe) = build(&matrices, &mut pairs, &row_dimensions);
        assert_eq!(probe.fallback_publications, 1);
        assert_eq!(factor.unwrap().data(), reference);
        assert_buffers_intact(&pairs, side);

        // Padded source geometry: the even factor carries one extra leading
        // row that no tree covers, so tree offsets no longer start at zero.
        let (_, mut padded) = z2_two_sector_geometry::<f64>();
        padded[0].rows += 1;
        for tree in &mut padded[0].row_trees {
            tree.1 += 1;
        }
        let mut pairs =
            staged_one_sided_pairs(&padded, &row_dimensions, side, side, &|k| k as f64 + 1.0);
        let selected = pairs
            .iter()
            .map(|pair| pair.left.clone())
            .collect::<Vec<_>>();
        let (factor, probe) = build(&padded, &mut pairs, &row_dimensions);
        assert_eq!(probe.fallback_publications, 1);
        let factor = factor.unwrap();
        assert_eq!(factor.data().len(), reference.len());
        assert_literal_one_sided_layout(
            factor.space().space().structure(),
            factor.data(),
            &padded,
            &selected,
            &row_dimensions,
            side,
            side,
        );
        assert_buffers_intact(&pairs, side);

        // Identity-only sector after (drop odd) and before (drop even) the
        // populated sector; the populated region equals the canonical one.
        let mut pairs = staged(1.0);
        let (factor, probe) = build(&matrices[..1], &mut pairs[..1], &row_dimensions);
        assert_eq!(probe.fallback_publications, 1);
        let factor = factor.unwrap();
        assert_eq!(&factor.data()[..sector_len], &reference[..sector_len]);
        let identity = &factor.data()[sector_len..];
        assert_eq!(
            identity.iter().filter(|&&value| value == 1.0).count(),
            matrices[1].rows
        );
        assert!(identity.iter().all(|&value| value == 0.0 || value == 1.0));
        assert_buffers_intact(&pairs, side);
        let mut pairs = staged(1.0);
        let (factor, probe) = build(&matrices[1..], &mut pairs[1..], &row_dimensions);
        assert_eq!(probe.fallback_publications, 1);
        let factor = factor.unwrap();
        assert_eq!(&factor.data()[sector_len..], &reference[sector_len..]);
        assert_eq!(
            factor.data()[..sector_len]
                .iter()
                .filter(|&&value| value == 1.0)
                .count(),
            matrices[0].rows
        );
        assert_buffers_intact(&pairs, side);

        // Missing pair for a populated sector: unchanged error.
        let mut pairs = staged(1.0);
        let (result, probe) = build(&matrices, &mut pairs[..1], &row_dimensions);
        assert!(matches!(
            result.unwrap_err(),
            OperationError::UnsupportedTensorContractScope {
                message: "factor rank absent for a populated source sector"
            }
        ));
        assert_eq!(probe.fallback_publications, 1);
        assert_buffers_intact(&pairs, side);

        // Extraneous pair: unchanged error before any publication decision.
        let mut pairs = staged(1.0);
        pairs.push(FactorPair {
            sector: SectorId::new(5),
            kept: 1,
            left: vec![1.0],
            left_rows: 1,
            right: Vec::new(),
            right_leading: 0,
        });
        let (result, probe) = build(&matrices, &mut pairs, &row_dimensions);
        assert!(matches!(
            result.unwrap_err(),
            OperationError::UnsupportedTensorContractScope {
                message: "factor sector absent from the source tensor"
            }
        ));
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 0)
        );
        assert_buffers_intact(&pairs, side);

        // Duplicate pair records for the even sector with different payloads:
        // the last record wins, on the scatter path.
        let alternate = canonical(2.0);
        let mut pairs = staged(1.0);
        let mut duplicate = staged(2.0);
        pairs.push(duplicate.swap_remove(0));
        let (factor, probe) = build(&matrices, &mut pairs, &row_dimensions);
        assert_eq!(probe.fallback_publications, 1);
        let factor = factor.unwrap();
        assert_eq!(&factor.data()[..sector_len], &alternate[..sector_len]);
        assert_eq!(&factor.data()[sector_len..], &reference[sector_len..]);
        assert_ne!(&alternate[..sector_len], &reference[..sector_len]);
        assert_buffers_intact(&pairs, side);
    }

    #[test]
    fn mf_one_sided_identity_sector_between_populated_sectors_falls_back() {
        let rule = Arc::new(tenet_core::ZNFusionRule::new(3).unwrap());
        let sectors = [SectorId::new(0), SectorId::new(1), SectorId::new(2)];
        let [s0, s1, s2] = sectors;
        let (authority, matrices) = two_leg_geometry::<_, f64>(
            rule,
            [
                vec![(s0, 1), (s1, 2), (s2, 1)],
                vec![(s0, 2), (s1, 1), (s2, 1)],
                vec![(s0, 1), (s1, 1), (s2, 2)],
                vec![(s0, 2), (s1, 1), (s2, 1)],
            ],
        );
        assert_eq!(
            matrices
                .iter()
                .map(|matrix| matrix.sector)
                .collect::<Vec<_>>(),
            sectors
        );
        let side = FactorSide::Right;
        let col_dimensions = matrices
            .iter()
            .map(|matrix| (matrix.sector, matrix.cols))
            .collect::<BTreeMap<_, _>>();
        let build = |matrices: &[SectorMatricization<f64>], pairs: &mut [FactorPair<f64>]| {
            reset_one_sided_publication_probe();
            let factor = build_bound_factor_with_placement(
                &authority,
                authority.space().homspace(),
                matrices,
                pairs,
                &col_dimensions,
                side,
                FactorPlacement::Direct,
            )
            .unwrap();
            (factor.data().to_vec(), one_sided_publication_probe())
        };
        let mut pairs =
            staged_one_sided_pairs(&matrices, &col_dimensions, side, side, &|k| k as f64 + 0.25);
        let (reference, probe) = build(&matrices, &mut pairs);
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );

        let (_, mut without_middle) = two_leg_geometry::<_, f64>(
            Arc::new(tenet_core::ZNFusionRule::new(3).unwrap()),
            [
                vec![(s0, 1), (s1, 2), (s2, 1)],
                vec![(s0, 2), (s1, 1), (s2, 1)],
                vec![(s0, 1), (s1, 1), (s2, 2)],
                vec![(s0, 2), (s1, 1), (s2, 1)],
            ],
        );
        without_middle.remove(1);
        let mut pairs =
            staged_one_sided_pairs(&without_middle, &col_dimensions, side, side, &|k| {
                k as f64 + 0.25
            });
        let (data, probe) = build(&without_middle, &mut pairs);
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 1)
        );
        assert_buffers_intact(&pairs, side);
        let first = matrices[0].cols * matrices[0].cols;
        let middle = matrices[1].cols * matrices[1].cols;
        assert_eq!(&data[..first], &reference[..first]);
        assert_eq!(&data[first + middle..], &reference[first + middle..]);
        assert_eq!(
            data[first..first + middle]
                .iter()
                .filter(|&&value| value == 1.0)
                .count(),
            matrices[1].cols
        );
    }

    #[test]
    fn mf_one_sided_zero_bond_sector_between_populated_sectors() {
        // Pinned outcome (a): the admitted structure omits zero-extent blocks,
        // so a zero-bond sector contributes no blocks and an empty factor; the
        // populated neighbours still transfer canonically.
        let legs = |s0, s1, s2| {
            [
                vec![(s0, 1), (s1, 2), (s2, 1)],
                vec![(s0, 2), (s1, 1), (s2, 1)],
                vec![(s0, 1), (s1, 1), (s2, 2)],
                vec![(s0, 2), (s1, 1), (s2, 1)],
            ]
        };
        let [s0, s1, s2] = [SectorId::new(0), SectorId::new(1), SectorId::new(2)];
        let (authority, matrices) = two_leg_geometry::<_, f64>(
            Arc::new(tenet_core::ZNFusionRule::new(3).unwrap()),
            legs(s0, s1, s2),
        );
        assert_eq!(
            matrices
                .iter()
                .map(|matrix| matrix.sector)
                .collect::<Vec<_>>(),
            [s0, s1, s2]
        );
        let side = FactorSide::Right;
        let dimensions = BTreeMap::from([(s0, matrices[0].cols), (s1, 0), (s2, matrices[2].cols)]);
        let mut pairs =
            staged_one_sided_pairs(&matrices, &dimensions, side, side, &|k| k as f64 - 0.5);
        pairs[1].kept = 0;
        assert!(pairs[1].right.is_empty());
        let selected = pairs
            .iter()
            .map(|pair| pair.right.clone())
            .collect::<Vec<_>>();
        let opposite = pairs
            .iter()
            .map(|pair| pair.left.clone())
            .collect::<Vec<_>>();

        reset_one_sided_publication_probe();
        let factor = build_bound_factor_with_placement(
            &authority,
            authority.space().homspace(),
            &matrices,
            &mut pairs,
            &dimensions,
            side,
            FactorPlacement::Direct,
        )
        .unwrap();

        let probe = one_sided_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );
        assert_eq!(probe.appended_elements, selected[2].len());
        assert_eq!(factor.data().len(), selected[0].len() + selected[2].len());
        assert_literal_one_sided_layout(
            factor.space().space().structure(),
            factor.data(),
            &matrices,
            &selected,
            &dimensions,
            side,
            side,
        );
        for (index, pair) in pairs.iter().enumerate() {
            assert!(pair.right.is_empty());
            assert_eq!(pair.left, opposite[index]);
        }
    }

    /// One even sector whose selected tree has zero legs: the factor block on
    /// that side is rank-1 (`shape == [b]`). `padding` extra source rows or
    /// columns precede the tree so the canonical proof declines and the
    /// scatter fallback publishes the block.
    fn zero_leg_side_matrix<D: FactorScalar>(
        zero_leg_side: FactorSide,
        extent: usize,
        padding: usize,
    ) -> (FusionTreeHomSpace, SectorMatricization<D>) {
        let even = SectorId::new(0);
        let leg = FusionProductSpace::new([SectorLeg::new([(even, extent)], false)]);
        let homspace = match zero_leg_side {
            FactorSide::Left => FusionTreeHomSpace::new(FusionProductSpace::new([]), leg),
            FactorSide::Right => FusionTreeHomSpace::new(leg, FusionProductSpace::new([])),
        };
        let key = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap()[0].clone();
        let (rows, cols, row_trees, col_trees) = match zero_leg_side {
            FactorSide::Left => (
                1 + padding,
                extent,
                vec![(key.codomain_tree().clone(), padding, vec![])],
                vec![(key.domain_tree().clone(), 0, vec![extent])],
            ),
            FactorSide::Right => (
                extent,
                1 + padding,
                vec![(key.codomain_tree().clone(), 0, vec![extent])],
                vec![(key.domain_tree().clone(), padding, vec![])],
            ),
        };
        (
            homspace,
            SectorMatricization {
                sector: even,
                rows,
                cols,
                row_trees,
                col_trees,
                data: vec![D::zero(); rows * cols],
            },
        )
    }

    /// Rank-1 blocks on both sides through the MF and the checked one-sided
    /// owners (#1197). Right: `b x cols` column-major, the block at column
    /// `o` is `F[j + b*o]`; Left: `a x b`, the block at row `o` is
    /// `F[o + a*j]`. Before the fix the Right block was read with the Left
    /// stride (`F[o + b*j]`), which for `b = 3, cols = 2, o = 1` reaches
    /// `F[7]` past the six-element factor and panicked on the slice bound.
    fn one_sided_rank1_fallback_case<D>(values: &dyn Fn(usize) -> D)
    where
        D: FactorScalar + PartialEq + fmt::Debug,
    {
        let even = SectorId::new(0);
        let b = 3usize;
        let padding = 1usize;
        let factor = (0..2 * b).map(values).collect::<Vec<_>>();
        let pair = |side| match side {
            FactorSide::Right => FactorPair {
                sector: even,
                kept: 99,
                left: vec![values(500)],
                left_rows: 1,
                right: factor.clone(),
                right_leading: b,
            },
            FactorSide::Left => FactorPair {
                sector: even,
                kept: 99,
                left: factor.clone(),
                left_rows: 1 + padding,
                right: vec![values(500)],
                right_leading: 1,
            },
        };
        let cases = [
            (
                FactorSide::Right,
                (0..b).map(|j| factor[j + b * padding]).collect::<Vec<_>>(),
            ),
            (
                FactorSide::Left,
                (0..b)
                    .map(|j| factor[padding + (1 + padding) * j])
                    .collect::<Vec<_>>(),
            ),
        ];
        for (side, expected) in cases {
            let (homspace, matrix) = zero_leg_side_matrix::<D>(side, 2, padding);
            let dimensions = BTreeMap::from([(even, b)]);
            let selected = factor.clone();

            let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
                Arc::new(Z2FusionRule),
                homspace.clone(),
            )
            .unwrap();
            let mut pairs = [pair(side)];
            reset_one_sided_publication_probe();
            let mf = build_bound_factor_with_placement(
                &authority,
                &homspace,
                std::slice::from_ref(&matrix),
                &mut pairs,
                &dimensions,
                side,
                FactorPlacement::Direct,
            )
            .unwrap();
            let probe = one_sided_publication_probe();
            assert_eq!(
                (probe.canonical_publications, probe.fallback_publications),
                (0, 1)
            );
            let block = mf.space().space().structure().block(0).unwrap();
            assert_eq!(block.shape(), [b]);
            assert_eq!(mf.data(), expected);
            assert_eq!(selected_of(&pairs[0], side), &selected);

            let provider = Arc::new(InfallibleGeneric::new(&TestGenericRule));
            let mut pairs = [pair(side)];
            reset_one_sided_publication_probe();
            let checked = build_bound_factor_generic_checked(
                &provider,
                &homspace,
                std::slice::from_ref(&matrix),
                &mut pairs,
                &dimensions,
                side,
            )
            .unwrap();
            let probe = one_sided_publication_probe();
            assert_eq!(
                (probe.canonical_publications, probe.fallback_publications),
                (0, 1)
            );
            assert_eq!(checked.data(), expected);
            assert_eq!(selected_of(&pairs[0], side), &selected);
        }
    }

    #[test]
    fn one_sided_rank1_fallback_scatters_right_and_left_layouts_real() {
        one_sided_rank1_fallback_case(&|k| k as f64 + 0.5);
    }

    #[test]
    fn one_sided_rank1_fallback_scatters_right_and_left_layouts_complex() {
        one_sided_rank1_fallback_case(&|k| Complex64::new(k as f64, -(k as f64) - 0.25));
    }

    #[test]
    fn generic_pair_fallback_scatters_rank1_right_block_with_right_layout() {
        let b = 3usize;
        let padding = 1usize;
        let (homspace, matrix) = zero_leg_side_matrix::<f64>(FactorSide::Right, 2, padding);
        let provider = Arc::new(TestGenericRule);
        let left = (0..2 * b).map(|k| 10.0 + k as f64).collect::<Vec<_>>();
        let right = (0..b * (1 + padding))
            .map(|k| 20.0 + k as f64)
            .collect::<Vec<_>>();
        reset_generic_pair_publication_probe();

        let (left_factor, right_factor) = build_left_right_bound_pair_generic(
            &provider,
            &homspace,
            std::slice::from_ref(&matrix),
            vec![FactorPair {
                sector: SectorId::new(0),
                kept: b,
                left: left.clone(),
                left_rows: 2,
                right: right.clone(),
                right_leading: b,
            }],
        )
        .unwrap();

        let probe = generic_pair_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 1)
        );
        assert_eq!(left_factor.data(), left);
        let block = right_factor.space().space().structure().block(0).unwrap();
        assert_eq!(block.shape(), [b]);
        assert_eq!(
            right_factor.data(),
            (0..b).map(|j| right[j + b * padding]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn checked_one_sided_canonical_transfer_across_sectors_and_extra_pairs() {
        let rule = TestGenericRule;
        let provider = Arc::new(InfallibleGeneric::new(&rule));
        let even = SectorId::new(0);
        let odd = SectorId::new(1);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(even, 2), (odd, 1)], false),
                SectorLeg::new([(even, 1), (odd, 3)], false),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(even, 1), (odd, 2)], false),
                SectorLeg::new([(even, 2), (odd, 1)], false),
            ]),
        );
        let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            homspace.clone(),
        )
        .unwrap();
        let zeros = vec![Complex64::new(0.0, 0.0); space.space().required_len().unwrap()];
        let matrices = sector_matricizations(space.space().structure(), &zeros, 2).unwrap();
        assert_eq!(matrices.len(), 2);
        let values = |k: usize| Complex64::new(0.5 * k as f64, 2.0 - k as f64);

        for side in [FactorSide::Left, FactorSide::Right] {
            let dimensions = matrices
                .iter()
                .map(|matrix| (matrix.sector, source_extent(matrix, side)))
                .collect::<BTreeMap<_, _>>();
            let mut pairs = staged_one_sided_pairs(&matrices, &dimensions, side, side, &values);
            let selected = pairs
                .iter()
                .map(|pair| selected_of(pair, side).clone())
                .collect::<Vec<_>>();
            let opposite = pairs
                .iter()
                .map(|pair| opposite_of(pair, side).clone())
                .collect::<Vec<_>>();
            reset_one_sided_publication_probe();
            let factor = build_bound_factor_generic_checked(
                &provider,
                &homspace,
                &matrices,
                &mut pairs,
                &dimensions,
                side,
            )
            .unwrap();
            let probe = one_sided_publication_probe();
            assert_eq!(
                (probe.canonical_publications, probe.fallback_publications),
                (1, 0)
            );
            assert_eq!(probe.appended_elements, selected[1].len());
            assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));
            assert_literal_one_sided_layout(
                factor.space().space().structure(),
                factor.data(),
                &matrices,
                &selected,
                &dimensions,
                side,
                side,
            );
            for (index, pair) in pairs.iter().enumerate() {
                assert!(selected_of(pair, side).is_empty());
                assert_eq!(opposite_of(pair, side), &opposite[index]);
            }

            // An extra unused pair is ignored on the scatter path, never an
            // error, and the output is identical.
            let mut pairs = staged_one_sided_pairs(&matrices, &dimensions, side, side, &values);
            pairs.push(FactorPair {
                sector: SectorId::new(7),
                kept: 1,
                left: vec![Complex64::new(1.0, 1.0)],
                left_rows: 1,
                right: vec![Complex64::new(1.0, 1.0)],
                right_leading: 1,
            });
            reset_one_sided_publication_probe();
            let fallback = build_bound_factor_generic_checked(
                &provider,
                &homspace,
                &matrices,
                &mut pairs,
                &dimensions,
                side,
            )
            .unwrap();
            let probe = one_sided_publication_probe();
            assert_eq!(
                (probe.canonical_publications, probe.fallback_publications),
                (0, 1)
            );
            assert_eq!(fallback.data(), factor.data());
            assert_buffers_intact(&pairs, side);
        }
    }

    #[test]
    fn checked_one_sided_vertex_trees_transfer_sole_owner_or_fall_back() {
        let rule = TestGenericRule;
        let provider = Arc::new(InfallibleGeneric::new(&rule));
        let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
        for reverse in [false, true] {
            for side in [FactorSide::Left, FactorSide::Right] {
                let (homspace, matrix, mut pair) = vertex_tree_factor_fixture(reverse);
                let (_, _, reference) = vertex_tree_factor_fixture(reverse);
                let selected_ptr = selected_of(&pair, side).as_ptr();
                reset_one_sided_publication_probe();
                let factor = build_bound_factor_generic_checked(
                    &provider,
                    &homspace,
                    std::slice::from_ref(&matrix),
                    std::slice::from_mut(&mut pair),
                    &dimensions,
                    side,
                )
                .unwrap();
                let probe = one_sided_publication_probe();
                assert_eq!(opposite_of(&pair, side), opposite_of(&reference, side));
                if reverse {
                    assert_eq!(
                        (probe.canonical_publications, probe.fallback_publications),
                        (0, 1)
                    );
                    assert_eq!(selected_of(&pair, side), selected_of(&reference, side));
                } else {
                    assert_eq!(
                        (
                            probe.canonical_publications,
                            probe.fallback_publications,
                            probe.owner_reused,
                            probe.appended_elements,
                        ),
                        (1, 0, 1, 0)
                    );
                    assert!(std::ptr::eq(selected_ptr, factor.data().as_ptr()));
                    assert!(selected_of(&pair, side).is_empty());
                    assert_eq!(factor.data(), selected_of(&reference, side));
                }
            }
        }
    }

    /// Records every checked provider query in call order so that two
    /// publication routes can be compared query by query.
    struct RecordingGeneric {
        rule: TestGenericRule,
        log: RefCell<Vec<String>>,
    }

    impl RecordingGeneric {
        fn new() -> Self {
            Self {
                rule: TestGenericRule,
                log: RefCell::new(Vec::new()),
            }
        }

        fn record<T>(&self, entry: String, value: T) -> Result<T, std::convert::Infallible> {
            self.log.borrow_mut().push(entry);
            Ok(value)
        }

        fn log(&self) -> Vec<String> {
            self.log.borrow().clone()
        }
    }

    impl CheckedGenericFusion for RecordingGeneric {
        type Error = std::convert::Infallible;

        fn rule_identity(&self) -> tenet_core::RuleIdentity {
            self.rule.rule_identity()
        }

        fn fusion_style(&self) -> tenet_core::FusionStyleKind {
            self.rule.fusion_style()
        }

        fn braiding_style(&self) -> tenet_core::BraidingStyleKind {
            self.rule.braiding_style()
        }

        fn vacuum(&self) -> SectorId {
            self.rule.vacuum()
        }

        fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
            self.record(format!("dual {sector:?}"), self.rule.dual(sector))
        }

        fn try_fusion_channels(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<tenet_core::SectorVec, Self::Error> {
            self.record(
                format!("channels {left:?} {right:?}"),
                self.rule.fusion_channels(left, right),
            )
        }

        fn try_fusion_channels_in_table(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<tenet_core::SectorVec, Self::Error> {
            self.record(
                format!("channels_in_table {left:?} {right:?}"),
                self.rule.fusion_channels(left, right),
            )
        }

        fn try_nsymbol(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Result<usize, Self::Error> {
            self.record(
                format!("nsymbol {left:?} {right:?} {coupled:?}"),
                self.rule.nsymbol(left, right, coupled),
            )
        }
    }

    /// The output HomSpace exactly as `build_bound_factor_generic_checked`
    /// derives it, built independently of the returned factor.
    fn one_sided_output_hom(
        homspace: &FusionTreeHomSpace,
        dimensions: &BTreeMap<SectorId, usize>,
        side: FactorSide,
    ) -> FusionTreeHomSpace {
        let bond = SectorLeg::new(
            dimensions.iter().map(|(&sector, &dim)| (sector, dim)),
            false,
        );
        match side {
            FactorSide::Left => FusionTreeHomSpace::new(
                homspace.codomain().clone(),
                FusionProductSpace::new([bond]),
            ),
            FactorSide::Right => {
                FusionTreeHomSpace::new(FusionProductSpace::new([bond]), homspace.domain().clone())
            }
        }
    }

    fn block_rows(structure: &BlockStructure) -> Vec<(BlockKey, Vec<usize>, Vec<usize>, usize)> {
        (0..structure.block_count())
            .map(|index| structure.block(index).unwrap())
            .map(|block| {
                (
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
            })
            .collect()
    }

    /// Two-sector checked geometry with `keep` selecting which sectors carry a
    /// matricization; the others become identity-only output sectors.
    type TwoSectorCheckedFixture = (
        FusionTreeHomSpace,
        Vec<SectorMatricization<Complex64>>,
        BTreeMap<SectorId, usize>,
        BTreeMap<SectorId, usize>,
    );

    fn two_sector_checked_fixture(keep: &dyn Fn(SectorId) -> bool) -> TwoSectorCheckedFixture {
        let rule = TestGenericRule;
        let provider = Arc::new(InfallibleGeneric::new(&rule));
        let even = SectorId::new(0);
        let odd = SectorId::new(1);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(even, 2), (odd, 1)], false),
                SectorLeg::new([(even, 1), (odd, 3)], false),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(even, 1), (odd, 2)], false),
                SectorLeg::new([(even, 2), (odd, 1)], false),
            ]),
        );
        let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            provider,
            homspace.clone(),
        )
        .unwrap();
        let zeros = vec![Complex64::new(0.0, 0.0); space.space().required_len().unwrap()];
        let matrices = sector_matricizations(space.space().structure(), &zeros, 2).unwrap();
        assert_eq!(matrices.len(), 2);
        let row_dimensions = matrices
            .iter()
            .map(|matrix| (matrix.sector, matrix.rows))
            .collect();
        let col_dimensions = matrices
            .iter()
            .map(|matrix| (matrix.sector, matrix.cols))
            .collect();
        let matrices = matrices
            .into_iter()
            .filter(|matrix| keep(matrix.sector))
            .collect();
        (homspace, matrices, row_dimensions, col_dimensions)
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
    fn checked_one_sided_enumerates_the_output_layout_once() {
        // What: one publication issues exactly the provider queries of a
        // single layout enumeration; the former route (keys, then a bound
        // space) issued that identical sequence twice, and the committed
        // block order equals the enumerated key order.
        let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
        for reverse in [false, true] {
            for side in [FactorSide::Left, FactorSide::Right] {
                let (homspace, matrix, mut pair) = vertex_tree_factor_fixture(reverse);
                let recorder = Arc::new(RecordingGeneric::new());
                let factor = build_bound_factor_generic_checked(
                    &recorder,
                    &homspace,
                    std::slice::from_ref(&matrix),
                    std::slice::from_mut(&mut pair),
                    &dimensions,
                    side,
                )
                .unwrap();
                let once = recorder.log();
                // Vertex fixture: the one-leg bond side folds once, the
                // two-leg side queries channels and the multiplicity of its
                // single vertex.
                assert_eq!(once.len(), 3, "{once:?}");

                let former = Arc::new(RecordingGeneric::new());
                let output_hom = one_sided_output_hom(&homspace, &dimensions, side);
                let keys = output_hom
                    .fusion_tree_keys_generic_checked(former.as_ref())
                    .unwrap();
                let expected = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
                    Arc::clone(&former),
                    output_hom,
                )
                .unwrap();
                let twice = former.log();
                assert_eq!(twice.len(), 2 * once.len());
                assert_eq!(&twice[..once.len()], once.as_slice());
                assert_eq!(&twice[once.len()..], once.as_slice());

                let block_keys = block_rows(factor.space().space().structure())
                    .into_iter()
                    .map(|(key, ..)| key)
                    .collect::<Vec<_>>();
                let key_order = keys.into_iter().map(BlockKey::from).collect::<Vec<_>>();
                assert_eq!(block_keys, key_order);
                assert_eq!(
                    block_rows(factor.space().space().structure()),
                    block_rows(expected.space().structure())
                );
                assert!(Arc::ptr_eq(factor.space().provider_arc(), &recorder));
            }
        }
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
    fn checked_one_sided_placement_error_precedes_bound_space() {
        // What: a populated output key whose tree the matricization lacks
        // fails after the single enumeration and before any bound space or
        // publication exists; the staged factor buffers stay untouched.
        let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
        for side in [FactorSide::Left, FactorSide::Right] {
            let (homspace, mut matrix, mut pair) = vertex_tree_factor_fixture(false);
            let (_, _, reference) = vertex_tree_factor_fixture(false);
            match side {
                FactorSide::Left => matrix.row_trees.truncate(1),
                FactorSide::Right => matrix.col_trees.truncate(1),
            }
            let recorder = Arc::new(RecordingGeneric::new());
            reset_one_sided_publication_probe();
            reset_placement_index_probe();
            let error = build_bound_factor_generic_checked(
                &recorder,
                &homspace,
                std::slice::from_ref(&matrix),
                std::slice::from_mut(&mut pair),
                &dimensions,
                side,
            )
            .unwrap_err();
            let expected_message = match side {
                FactorSide::Left => "factor codomain tree absent from the source matricization",
                FactorSide::Right => "factor domain tree absent from the source matricization",
            };
            assert!(matches!(
                error,
                CheckedGenericFactorPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope { message }
                ) if message == expected_message
            ));
            assert_eq!(recorder.log().len(), 3);
            let probe = one_sided_publication_probe();
            assert_eq!(
                (probe.canonical_publications, probe.fallback_publications),
                (0, 0)
            );
            let index = placement_index_probe();
            assert_eq!(
                (index.index_builds, index.indexed_sides, index.indexed_trees),
                (1, 1, 1)
            );
            assert_eq!(index.lookups, 2);
            assert_eq!(pair.left, reference.left);
            assert_eq!(pair.right, reference.right);
        }
    }

    #[test]
    fn checked_one_sided_structure_matches_two_enumeration_construction() {
        // What: with identity-only sectors before or after the populated one,
        // the committed structure, required length, data length and provider
        // binding equal those of a space built by the former separate
        // enumeration, for both sides and both scalar types.
        let rule = TestGenericRule;
        let provider = Arc::new(InfallibleGeneric::new(&rule));
        let even = SectorId::new(0);
        let odd = SectorId::new(1);
        let keeps: [&dyn Fn(SectorId) -> bool; 3] =
            [&|_| true, &|sector| sector == odd, &|sector| sector == even];
        for keep in keeps {
            let (homspace, matrices, row_dimensions, col_dimensions) =
                two_sector_checked_fixture(keep);
            for side in [FactorSide::Left, FactorSide::Right] {
                let dimensions = match side {
                    FactorSide::Left => &row_dimensions,
                    FactorSide::Right => &col_dimensions,
                };
                let expected = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
                    Arc::clone(&provider),
                    one_sided_output_hom(&homspace, dimensions, side),
                )
                .unwrap();
                let expected_len = expected.space().required_len().unwrap();

                let complex = |k: usize| Complex64::new(0.5 * k as f64, 2.0 - k as f64);
                let mut pairs = staged_one_sided_pairs(&matrices, dimensions, side, side, &complex);
                let factor = build_bound_factor_generic_checked(
                    &provider, &homspace, &matrices, &mut pairs, dimensions, side,
                )
                .unwrap();
                assert_eq!(
                    block_rows(factor.space().space().structure()),
                    block_rows(expected.space().structure())
                );
                assert_eq!(factor.space().space().required_len().unwrap(), expected_len);
                assert_eq!(factor.data().len(), expected_len);
                assert_eq!(
                    factor.space().space().homspace(),
                    expected.space().homspace()
                );
                assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));

                let real_matrices = matrices
                    .iter()
                    .map(|matrix| SectorMatricization {
                        sector: matrix.sector,
                        rows: matrix.rows,
                        cols: matrix.cols,
                        row_trees: matrix.row_trees.clone(),
                        col_trees: matrix.col_trees.clone(),
                        data: Vec::<f64>::new(),
                    })
                    .collect::<Vec<_>>();
                let real = |k: usize| 0.25 * k as f64;
                let mut pairs =
                    staged_one_sided_pairs(&real_matrices, dimensions, side, side, &real);
                let factor = build_bound_factor_generic_checked(
                    &provider,
                    &homspace,
                    &real_matrices,
                    &mut pairs,
                    dimensions,
                    side,
                )
                .unwrap();
                assert_eq!(
                    block_rows(factor.space().space().structure()),
                    block_rows(expected.space().structure())
                );
                assert_eq!(factor.data().len(), expected_len);
            }
        }
    }

    /// Z_4 fusion declared under Generic style so the generic paired builders
    /// accept it; multiplicity-free, so trees and blocks match the abelian
    /// geometry.
    struct Z4GenericRule;

    static Z4_GENERIC: Z4GenericRule = Z4GenericRule;

    impl FusionRule for Z4GenericRule {
        fn rule_identity(&self) -> tenet_core::RuleIdentity {
            tenet_core::RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> tenet_core::FusionStyleKind {
            tenet_core::FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> tenet_core::BraidingStyleKind {
            tenet_core::BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            SectorId::new((4 - sector.id() % 4) % 4)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> tenet_core::SectorVec {
            [SectorId::new((left.id() + right.id()) % 4)]
                .into_iter()
                .collect()
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            usize::from((left.id() + right.id()) % 4 == coupled.id())
        }
    }

    /// Two-leg codomain/domain hom space under [`Z4GenericRule`], matricized
    /// exactly as the generic production path does.
    fn z4_generic_geometry<D: FactorScalar>(
        legs: [Vec<(SectorId, usize)>; 4],
    ) -> (FusionTreeHomSpace, Vec<SectorMatricization<D>>) {
        let [a, b, c, d] = legs;
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(a, false), SectorLeg::new(b, false)]),
            FusionProductSpace::new([SectorLeg::new(c, false), SectorLeg::new(d, false)]),
        );
        let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::new(InfallibleGeneric::new(&Z4_GENERIC)),
            homspace.clone(),
        )
        .unwrap();
        let zeros = vec![D::zero(); space.space().required_len().unwrap()];
        let matrices = sector_matricizations_generic(space.space().structure(), &zeros, 2).unwrap();
        (homspace, matrices)
    }

    /// Every leg carries all four charges: G_s = 4 coupled sectors with four
    /// row and four column trees each, so each factor side has B = F = 16
    /// output blocks.
    fn z4_all_charge_legs() -> [Vec<(SectorId, usize)>; 4] {
        let all = (0..4).map(|s| (SectorId::new(s), 1)).collect::<Vec<_>>();
        [all.clone(), all.clone(), all.clone(), all]
    }

    /// Stages `rows x kept` left and `kept x cols` right factors per sector
    /// with `kept = min(rows, cols)` and sector-tagged values.
    fn staged_pairs<D: FactorScalar>(
        matrices: &[SectorMatricization<D>],
        values: &dyn Fn(usize) -> D,
    ) -> Vec<FactorPair<D>> {
        matrices
            .iter()
            .map(|matrix| {
                let kept = matrix.rows.min(matrix.cols);
                let tag = matrix.sector.id() + 1;
                FactorPair {
                    sector: matrix.sector,
                    kept,
                    left: (0..matrix.rows * kept)
                        .map(|k| values(1000 * tag + k))
                        .collect(),
                    left_rows: matrix.rows,
                    right: (0..kept * matrix.cols)
                        .map(|k| values(5000 * tag + k))
                        .collect(),
                    right_leading: kept,
                }
            })
            .collect()
    }

    #[test]
    fn generic_pair_fallback_scatters_each_output_block_once() {
        // What: a paired fallback publication over G_s = 4 matricizations
        // groups each factor side once and iterates only the F blocks of the
        // matricization being scattered, publishing the same data as the
        // canonical path, for both paired builders and both scalar types.
        fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
            let fresh = || z4_generic_geometry::<D>(z4_all_charge_legs());
            let (homspace, matrices) = fresh();
            assert_eq!(matrices.len(), 4);
            assert!(matrices
                .iter()
                .all(|m| m.row_trees.len() == 4 && m.col_trees.len() == 4));
            let plain_provider = Arc::new(Z4GenericRule);
            let checked_provider = Arc::new(InfallibleGeneric::new(&Z4_GENERIC));

            let build = |matrices: &[SectorMatricization<D>]| {
                reset_generic_pair_publication_probe();
                reset_scatter_visit_probe();
                let (left, right) = build_left_right_bound_pair_generic(
                    &plain_provider,
                    &homspace,
                    matrices,
                    staged_pairs(matrices, &values),
                )
                .unwrap();
                let plain = (
                    left.data().to_vec(),
                    right.data().to_vec(),
                    generic_pair_publication_probe(),
                    scatter_visit_probe(),
                );
                reset_generic_pair_publication_probe();
                reset_scatter_visit_probe();
                let (left, right) = build_left_right_bound_pair_generic_checked(
                    &checked_provider,
                    &homspace,
                    matrices,
                    staged_pairs(matrices, &values),
                )
                .unwrap();
                assert_eq!(
                    (
                        left.space().space().structure().block_count(),
                        right.space().space().structure().block_count()
                    ),
                    (16, 16)
                );
                let checked = (
                    left.data().to_vec(),
                    right.data().to_vec(),
                    generic_pair_publication_probe(),
                    scatter_visit_probe(),
                );
                (plain, checked)
            };

            let (reference, checked_reference) = build(&matrices);
            for (_, _, probe, visits) in [&reference, &checked_reference] {
                assert_eq!(probe.canonical_publications, 1);
                assert_eq!(*visits, ScatterVisitProbe::default());
            }

            let (_, mut reversed) = fresh();
            reversed.reverse();
            let (plain, checked) = build(&reversed);
            for ((left, right, probe, visits), (left_ref, right_ref, ..)) in
                [(&plain, &reference), (&checked, &checked_reference)]
            {
                assert_eq!(probe.fallback_publications, 1);
                assert_eq!((left, right), (left_ref, right_ref));
                assert_eq!(
                    (probe.left_scatter_calls, probe.right_scatter_calls),
                    (16, 16)
                );
                // Formerly G_s * B = 4 * 16 = 64 block visits per side; now
                // one grouping pass over B = 16 plus the F = 16 scattered
                // blocks.
                assert_eq!(
                    *visits,
                    ScatterVisitProbe {
                        left_grouped: 16,
                        right_grouped: 16,
                        left_groups_built: 1,
                        right_groups_built: 1,
                        left_visits: 16,
                        right_visits: 16,
                    }
                );
                assert!(visits.left_grouped + visits.left_visits < 4 * 16);
                assert!(visits.right_grouped + visits.right_visits < 4 * 16);
            }
        }
        run(|k| k as f64 + 0.125);
        run(|k| Complex64::new(k as f64 + 0.125, 0.75 - k as f64));
    }

    #[test]
    fn generic_pair_validation_reports_left_defect_before_right_across_sectors() {
        // What: validation precedence, not scatter order. With a defective
        // domain tree in the first sector and a defective codomain tree in a
        // later sector, the codomain error fires for both paired builders
        // because `validate_generic_factor_keys` checks every left key before
        // any right key. Scatter-time placement errors are unreachable in the
        // paired builders: validation has already checked every key.
        let two = || {
            let leg = || vec![(SectorId::new(0), 1), (SectorId::new(1), 1)];
            z4_generic_geometry::<f64>([leg(), leg(), leg(), leg()])
        };
        let (homspace, matrices) = two();
        assert_eq!(
            matrices.iter().map(|m| m.sector.id()).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        let plain_provider = Arc::new(Z4GenericRule);
        let checked_provider = Arc::new(InfallibleGeneric::new(&Z4_GENERIC));
        let foreign_row = matrices[0].row_trees[0].0.clone();
        let foreign_col = matrices[1].col_trees[0].0.clone();
        let corrupt = |bad_first_col: bool, bad_later_row: bool| {
            let (_, mut matrices) = two();
            if bad_first_col {
                matrices[0].col_trees[0].0 = foreign_col.clone();
            }
            if bad_later_row {
                matrices[1].row_trees[0].0 = foreign_row.clone();
            }
            matrices
        };
        let codomain = "factor codomain tree absent from the source matricization";
        let domain = "factor domain tree absent from the source matricization";
        for (bad_first_col, bad_later_row, expected) in [
            (true, true, codomain),
            (true, false, domain),
            (false, true, codomain),
        ] {
            let matrices = corrupt(bad_first_col, bad_later_row);
            let error = build_left_right_bound_pair_generic(
                &plain_provider,
                &homspace,
                &matrices,
                staged_pairs(&matrices, &|k| k as f64),
            )
            .unwrap_err();
            assert!(matches!(
                error,
                OperationError::UnsupportedTensorContractScope { message } if message == expected
            ));
            let error = build_left_right_bound_pair_generic_checked(
                &checked_provider,
                &homspace,
                &matrices,
                staged_pairs(&matrices, &|k| k as f64),
            )
            .unwrap_err();
            assert!(matches!(
                error,
                CheckedGenericFactorPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope { message }
                ) if message == expected
            ));
        }
    }

    /// The paired output HomSpaces exactly as
    /// `build_left_right_bound_pair_generic_checked` derives them, built
    /// independently of the returned factors.
    fn paired_output_homs<D>(
        homspace: &FusionTreeHomSpace,
        pairs: &[FactorPair<D>],
    ) -> (FusionTreeHomSpace, FusionTreeHomSpace) {
        let bond = SectorLeg::new(pairs.iter().map(|pair| (pair.sector, pair.kept)), false);
        (
            FusionTreeHomSpace::new(
                homspace.codomain().clone(),
                FusionProductSpace::new([bond.clone()]),
            ),
            FusionTreeHomSpace::new(FusionProductSpace::new([bond]), homspace.domain().clone()),
        )
    }

    /// One checked factor space built the former way: keys enumerated, then
    /// the bound space enumerated again.
    fn two_enumeration_space<R: CheckedGenericFusion>(
        provider: &Arc<R>,
        hom: FusionTreeHomSpace,
    ) -> (Vec<FusionTreePairKey>, BoundDynamicFusionMapSpace<R>) {
        let keys = hom
            .fusion_tree_keys_generic_checked(provider.as_ref())
            .unwrap();
        let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(provider),
            hom,
        )
        .unwrap();
        (keys, space)
    }

    fn assert_factor_matches_space<R: CheckedGenericFusion, D: FactorScalar + fmt::Debug>(
        factor: &BoundDynFactor<R, D>,
        keys: &[FusionTreePairKey],
        expected: &BoundDynamicFusionMapSpace<R>,
        provider: &Arc<R>,
    ) {
        let rows = block_rows(factor.space().space().structure());
        assert_eq!(rows, block_rows(expected.space().structure()));
        assert_eq!(
            rows.into_iter().map(|(key, ..)| key).collect::<Vec<_>>(),
            keys.iter().cloned().map(BlockKey::from).collect::<Vec<_>>()
        );
        let expected_len = expected.space().required_len().unwrap();
        assert_eq!(factor.space().space().required_len().unwrap(), expected_len);
        assert_eq!(factor.data().len(), expected_len);
        assert_eq!(
            factor.space().space().homspace(),
            expected.space().homspace()
        );
        assert!(Arc::ptr_eq(factor.space().provider_arc(), provider));
    }

    /// Checked paired publication equals the two-enumeration construction in
    /// structure, key order, length, homspace and provider binding, and the
    /// unchecked paired builder in data.
    fn assert_checked_pair_matches_two_enumeration_construction<
        P: FusionRule,
        R: CheckedGenericFusion,
        D: FactorScalar + fmt::Debug,
        M: SectorGeometry,
    >(
        plain: &Arc<P>,
        checked: &Arc<R>,
        homspace: &FusionTreeHomSpace,
        matrices: &[M],
        pairs: &dyn Fn() -> Vec<FactorPair<D>>,
        expected_publications: (usize, usize),
    ) {
        let (left_hom, right_hom) = paired_output_homs(homspace, &pairs());
        let (left_keys, left_expected) = two_enumeration_space(checked, left_hom);
        let (right_keys, right_expected) = two_enumeration_space(checked, right_hom);
        let (plain_left, plain_right) =
            build_left_right_bound_pair_generic(plain, homspace, matrices, pairs()).unwrap();
        reset_generic_pair_publication_probe();
        let (left, right) =
            build_left_right_bound_pair_generic_checked(checked, homspace, matrices, pairs())
                .unwrap();
        let probe = generic_pair_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            expected_publications
        );
        assert_eq!(probe.output_blocks_visited, 0);
        assert_eq!(
            probe.ordered_key_validation_events,
            left_keys.len() + right_keys.len()
        );
        assert_factor_matches_space(&left, &left_keys, &left_expected, checked);
        assert_factor_matches_space(&right, &right_keys, &right_expected, checked);
        assert_eq!(left.data(), plain_left.data());
        assert_eq!(right.data(), plain_right.data());
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
    fn checked_paired_enumerates_each_factor_layout_once() {
        // What: one paired publication issues exactly the provider queries of
        // one left enumeration followed by one right enumeration; the former
        // route (keys, then a bound space, per side) issued each of those
        // sequences twice. Canonical (`reverse = false`) and fallback
        // (`reverse = true`) geometries publish the same structure and data
        // as the two-enumeration construction.
        for reverse in [false, true] {
            let (homspace, matrix, pair) = vertex_tree_factor_fixture(reverse);
            let pairs = || vec![vertex_tree_factor_fixture(reverse).2];
            let recorder = Arc::new(RecordingGeneric::new());
            build_left_right_bound_pair_generic_checked(
                &recorder,
                &homspace,
                std::slice::from_ref(&matrix),
                vec![pair],
            )
            .unwrap();
            let once = recorder.log();

            let (left_hom, right_hom) = paired_output_homs(&homspace, &pairs());
            let side_log = |hom: &FusionTreeHomSpace| {
                let probe = RecordingGeneric::new();
                hom.prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(
                    &probe,
                )
                .unwrap();
                probe.log()
            };
            let (once_left, once_right) = (side_log(&left_hom), side_log(&right_hom));
            // Vertex fixture: each side folds its one-leg bond once and
            // queries channels plus the multiplicity of its single vertex.
            assert_eq!((once_left.len(), once_right.len()), (3, 3));
            assert_eq!(once, [once_left.clone(), once_right.clone()].concat());

            let former = Arc::new(RecordingGeneric::new());
            two_enumeration_space(&former, left_hom);
            two_enumeration_space(&former, right_hom);
            assert_eq!(
                former.log(),
                [once_left.clone(), once_left, once_right.clone(), once_right].concat()
            );

            assert_checked_pair_matches_two_enumeration_construction(
                &Arc::new(TestGenericRule),
                &recorder,
                &homspace,
                std::slice::from_ref(&matrix),
                &pairs,
                if reverse { (0, 1) } else { (1, 0) },
            );
        }
    }

    #[test]
    fn checked_paired_structure_matches_two_enumeration_construction() {
        // What: over G_s = 4 Z_4 matricizations with F = 16 blocks per side,
        // canonical and reversed (fallback) geometries, `f64` and
        // `Complex64`, the checked pair equals the two-enumeration
        // construction and the unchecked builder's data.
        fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
            let fresh = || z4_generic_geometry::<D>(z4_all_charge_legs());
            let (homspace, matrices) = fresh();
            let (_, mut reversed) = fresh();
            reversed.reverse();
            let plain = Arc::new(Z4GenericRule);
            let checked = Arc::new(InfallibleGeneric::new(&Z4_GENERIC));
            for (matrices, publications) in [(&matrices, (1, 0)), (&reversed, (0, 1))] {
                assert_checked_pair_matches_two_enumeration_construction(
                    &plain,
                    &checked,
                    &homspace,
                    matrices,
                    &|| staged_pairs(matrices, &values),
                    publications,
                );
            }
        }
        run(|k| k as f64 + 0.125);
        run(|k| Complex64::new(k as f64 + 0.125, 0.75 - k as f64));
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
    fn checked_paired_placement_error_precedes_bound_space() {
        // What: a factor key whose tree the matricization lacks fails after
        // that side's single enumeration and before any publication; the
        // left side is validated completely before the right side is
        // enumerated. Formerly the left error surfaced after 3 queries and
        // the right error after 9 (left keys, left space, right keys); now
        // after 3 and 6.
        for side in [FactorSide::Left, FactorSide::Right] {
            let (homspace, mut matrix, pair) = vertex_tree_factor_fixture(false);
            match side {
                FactorSide::Left => matrix.row_trees.truncate(1),
                FactorSide::Right => matrix.col_trees.truncate(1),
            }
            let recorder = Arc::new(RecordingGeneric::new());
            reset_generic_pair_publication_probe();
            let error = build_left_right_bound_pair_generic_checked(
                &recorder,
                &homspace,
                std::slice::from_ref(&matrix),
                vec![pair],
            )
            .unwrap_err();
            let (expected_message, expected_calls) = match side {
                FactorSide::Left => (
                    "factor codomain tree absent from the source matricization",
                    3,
                ),
                FactorSide::Right => ("factor domain tree absent from the source matricization", 6),
            };
            assert!(matches!(
                error,
                CheckedGenericFactorPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope { message }
                ) if message == expected_message
            ));
            assert_eq!(recorder.log().len(), expected_calls);
            let probe = generic_pair_publication_probe();
            assert_eq!(
                (probe.canonical_publications, probe.fallback_publications),
                (0, 0)
            );
        }
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
    fn checked_sliced_enumerates_the_bond_layout_once() {
        // What: one sliced-bond build issues exactly the provider queries of
        // one enumeration of the sliced HomSpace (formerly twice: keys, then
        // the bound space), on a codomain and a domain axis, and equals the
        // two-enumeration construction in structure and the unchecked sliced
        // builder in data.
        let x = SectorId::new(1);
        let (homspace, ..) = vertex_tree_factor_fixture(false);
        let source = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::new(InfallibleGeneric::new(&TestGenericRule)),
            homspace.clone(),
        )
        .unwrap();
        let data = (0..source.space().required_len().unwrap())
            .map(|k| Complex64::new(k as f64, -0.5 * k as f64))
            .collect::<Vec<_>>();
        let nout = source.space().nout();
        for (axis, kept) in [(0usize, 1usize), (3, 2)] {
            let kept_of = move |sector: SectorId| usize::from(sector == x) * kept;
            let recorder = Arc::new(RecordingGeneric::new());
            let factor = sliced_bond_tensor_generic_checked(
                Arc::clone(&recorder),
                source.space(),
                &data,
                axis,
                &kept_of,
                nout,
                2,
            )
            .unwrap();
            let once = recorder.log();

            let mut legs = if axis < nout {
                homspace.codomain().legs().to_vec()
            } else {
                homspace.domain().legs().to_vec()
            };
            legs[axis % nout] = SectorLeg::new([(x, kept)], false);
            let sliced_hom = if axis < nout {
                FusionTreeHomSpace::new(FusionProductSpace::new(legs), homspace.domain().clone())
            } else {
                FusionTreeHomSpace::new(homspace.codomain().clone(), FusionProductSpace::new(legs))
            };
            let former = Arc::new(RecordingGeneric::new());
            let (keys, expected) = two_enumeration_space(&former, sliced_hom);
            let twice = former.log();
            // Both sides of the sliced HomSpace keep two legs: channels plus
            // the vertex multiplicity per side, and one fold per side.
            assert_eq!(once.len(), 6, "{once:?}");
            assert_eq!(twice, [once.clone(), once].concat());
            assert_factor_matches_space(&factor, &keys, &expected, &recorder);

            let plain = sliced_bond_tensor_generic(
                Arc::new(TestGenericRule),
                source.space(),
                &data,
                axis,
                &kept_of,
                nout,
                2,
            )
            .unwrap();
            assert_eq!(factor.data(), plain.data());
            assert!(factor.data().len() < data.len());
        }
    }
}
