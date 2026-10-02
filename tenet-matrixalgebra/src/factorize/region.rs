use super::*;

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

// The single-precision magnitudes widen *before* the absolute value or the
// hypotenuse, so a `Complex32` whose components straddle the `f32` range still
// reports a finite magnitude and the truncation policies compare the same
// `f64` quantities they compare for a double-precision payload.
impl SpectrumMagnitude for f32 {
    fn magnitude(self) -> f64 {
        f64::from(self).abs()
    }
}

impl SpectrumMagnitude for num_complex::Complex32 {
    fn magnitude(self) -> f64 {
        Complex64::new(f64::from(self.re), f64::from(self.im)).norm()
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
    pub(super) space: &'a BoundDynamicFusionMapSpace<R>,
    pub(super) tensor: &'a TensorMap<D, NOUT, NIN>,
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
    pub(super) space: BoundDynamicFusionMapSpace<R>,
    pub(super) tensor: TensorMap<D, NOUT, NIN>,
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
    pub(super) space: BoundDynamicFusionMapSpace<R>,
    pub(super) data: Vec<D>,
}

pub(super) type DynamicFactorPair<R, D> = (BoundDynFactor<R, D>, BoundDynFactor<R, D>);

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

#[derive(Clone, Copy)]
pub(super) enum PolarDirection {
    Left,
    Right,
}

impl PolarDirection {
    pub(super) fn accepts(self, rows: usize, cols: usize) -> bool {
        match self {
            Self::Left => rows >= cols,
            Self::Right => cols >= rows,
        }
    }

    pub(super) fn error(self) -> OperationError {
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

pub(super) fn validate_polar_direction(
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompactFactorRoute {
    pub(super) source_region: usize,
    pub(super) left_region: Option<usize>,
    pub(super) right_region: Option<usize>,
    pub(super) sector: SectorId,
    pub(super) rank: usize,
}

/// Per-call routing from the source coupled-sector regions to the two factor
/// regions. Owned by the calling factorization and dropped with it.
///
/// Why not `Arc` the plan or its source half: nothing shares it beyond the
/// one call that builds it, so the wrappers were two heap allocations per
/// call with no owner to serve.
#[derive(Debug)]
pub(crate) struct CompactFactorPlan {
    pub(super) source_layout: ValidatedDynamicFusionLayout,
    pub(super) source_regions: Arc<[CoupledSectorRegion]>,
    pub(super) left_layout: ValidatedDynamicFusionLayout,
    pub(super) right_layout: ValidatedDynamicFusionLayout,
    pub(super) left_regions: Arc<[CoupledSectorRegion]>,
    pub(super) right_regions: Arc<[CoupledSectorRegion]>,
    pub(super) routes: Vec<CompactFactorRoute>,
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
    pub(super) source_layout: ValidatedDynamicFusionLayout,
    pub(super) source_regions: Arc<[CoupledSectorRegion]>,
    pub(super) left: PreparedCheckedGenericDynamicSpace,
    pub(super) right: PreparedCheckedGenericDynamicSpace,
    pub(super) left_regions: Arc<[CoupledSectorRegion]>,
    pub(super) right_regions: Arc<[CoupledSectorRegion]>,
    pub(super) routes: Vec<CompactFactorRoute>,
}

pub(super) fn compact_factor_plan<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    build_compact_factor_plan(input, input.validated_layout())
}

pub(super) fn compact_factor_plan_generic<R>(
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

pub(super) fn prepare_compact_factor_plan_generic_checked<R, P>(
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
    let left_regions =
        checked_sector_regions(left.structure(), space.nout())?.ok_or_else(|| {
            OperationError::UnsupportedTensorContractScope {
                message: "compact left factor is not a coupled-sector matrix layout",
            }
        })?;
    let right_regions = checked_sector_regions(right.structure(), 1)?.ok_or_else(|| {
        OperationError::UnsupportedTensorContractScope {
            message: "compact right factor is not a coupled-sector matrix layout",
        }
    })?;
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

pub(super) fn source_factor_tree_extents_match(
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

pub(super) fn finish_compact_factor_plan_generic<R>(
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
    pub(super) static GENERIC_FACTOR_PLAN_FINISH_CALLS: Cell<usize> = const { Cell::new(0) };
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

pub(super) fn build_compact_factor_plan<R>(
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
        .ok_or_else(|| OperationError::UnsupportedTensorContractScope {
            message: "compact left factor is not a coupled-sector matrix layout",
        })?;
    let right_regions =
        checked_sector_regions(vh_space.space().structure(), vh_space.space().nout())?.ok_or_else(
            || OperationError::UnsupportedTensorContractScope {
                message: "compact right factor is not a coupled-sector matrix layout",
            },
        )?;
    let routes = compile_compact_factor_routes(&regions, &left_regions, &right_regions)?;
    // The direct path publishes each dense factor region positionally, so it
    // is sound only when the fresh factor lists the source's trees in the
    // source's order. Why not scatter here: a reordered tiling takes the
    // packed path, whose `PlacementIndex` scatter already maps by tree
    // identity, so one authority owns non-canonical publication.
    if !compact_factor_routes_preserve_tree_order(&routes, &regions, &left_regions, &right_regions)
    {
        return Ok(None);
    }
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

pub(super) fn compact_factor_routes_preserve_tree_order(
    routes: &[CompactFactorRoute],
    source: &[CoupledSectorRegion],
    left: &[CoupledSectorRegion],
    right: &[CoupledSectorRegion],
) -> bool {
    routes.iter().all(|route| {
        let region = &source[route.source_region];
        route
            .left_region
            .is_none_or(|index| left[index].row_trees() == region.row_trees())
            && route
                .right_region
                .is_none_or(|index| right[index].col_trees() == region.col_trees())
    })
}

/// The bond leg `W` shared by both compact factors: one sector per source
/// region with degeneracy `min(rows, cols)`.
pub(super) fn compact_bond_leg(regions: &[CoupledSectorRegion]) -> SectorLeg {
    SectorLeg::new(
        regions
            .iter()
            .map(|region| (region_sector(region), region.rows().min(region.cols()))),
        false,
    )
}

pub(super) fn compile_compact_factor_routes(
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

pub(super) fn region_sector(region: &CoupledSectorRegion) -> SectorId {
    region.coupled()
}

pub(super) fn sector_region_index_map(
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

/// Proves that a compact diagonal spectrum exactly covers aligned square
/// source regions and returns the sole sector lookup used by its caller.
pub(super) fn aligned_diagonal_spectrum_by_sector<'a, D>(
    regions: &[CoupledSectorRegion],
    spectrum: &'a [SectorSpectrum<D>],
) -> Option<FxHashMap<SectorId, &'a SectorSpectrum<D>>> {
    if spectrum.len() != regions.len() {
        return None;
    }
    let mut by_sector = FxHashMap::with_capacity_and_hasher(spectrum.len(), Default::default());
    for entry in spectrum {
        if by_sector.insert(entry.sector, entry).is_some() {
            return None;
        }
    }
    for region in regions {
        let entry = by_sector.get(&region.coupled())?;
        if !region.has_aligned_diagonal()
            || region.rows() != region.cols()
            || entry.values.len() != region.rows()
        {
            return None;
        }
    }
    Some(by_sector)
}

/// Sector -> region index lookup over one factor's region table.
///
/// Canonical factor layouts list their regions strictly sorted by coupled
/// sector (a structural invariant of the derived layout, checked here in
/// O(G)), so a binary search serves without building a map. Expert region
/// tables that are not sorted keep the hash-map path, which also reports the
/// duplicate-sector error.
pub(super) enum SectorRegionIndex<'a> {
    Sorted(&'a [CoupledSectorRegion]),
    Map(FxHashMap<SectorId, usize>),
}

impl<'a> SectorRegionIndex<'a> {
    pub(super) fn new(regions: &'a [CoupledSectorRegion]) -> Result<Self, OperationError> {
        if regions
            .windows(2)
            .all(|pair| region_sector(&pair[0]) < region_sector(&pair[1]))
        {
            Ok(Self::Sorted(regions))
        } else {
            sector_region_index_map(regions).map(Self::Map)
        }
    }

    pub(super) fn get(&self, sector: SectorId) -> Option<usize> {
        match self {
            Self::Sorted(regions) => regions.binary_search_by_key(&sector, region_sector).ok(),
            Self::Map(map) => map.get(&sector).copied(),
        }
    }
}

pub(super) fn sector_region_index_of(
    regions: &SectorRegionIndex<'_>,
    sector: SectorId,
    side: &'static str,
) -> Result<usize, OperationError> {
    regions
        .get(sector)
        .ok_or_else(|| OperationError::UnsupportedTensorContractScope {
            message: match side {
                "left" => "compact left factor is missing a nonzero-rank sector",
                _ => "compact right factor is missing a nonzero-rank sector",
            },
        })
}

pub(super) fn validate_factor_region(
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

pub(super) fn validate_no_unused_factor_regions(
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

/// One coupled sector's factor pair: `left` is `left_rows x kept` (leading
/// columns of a column-major matrix), `right` is `kept x cols` (leading rows
/// of a column-major matrix with leading dimension `right_leading`).
pub(super) struct FactorPair<D> {
    pub(super) sector: SectorId,
    pub(super) kept: usize,
    pub(super) left: Vec<D>,
    pub(super) left_rows: usize,
    pub(super) right: Vec<D>,
    pub(super) right_leading: usize,
}

pub(super) struct SectorRank {
    pub(super) sector: SectorId,
    pub(super) kept: usize,
}

pub(super) struct GenericFactorPairSpaces<R> {
    pub(super) left: BoundDynamicFusionMapSpace<R>,
    pub(super) right: BoundDynamicFusionMapSpace<R>,
    pub(super) left_keys: Vec<FusionTreePairKey>,
    pub(super) right_keys: Vec<FusionTreePairKey>,
    pub(super) ordered: bool,
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
    pub(super) static GENERIC_PAIR_PUBLICATION_PROBE: Cell<GenericPairPublicationProbe> = Cell::default();
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
    /// Elements written by canonical publication beyond the moved first
    /// owner: appended factors plus in-place identity blocks.
    pub appended_elements: usize,
    /// Heap bytes held by publication plans (identity segment lists).
    pub plan_bytes: usize,
}

#[cfg(test)]
thread_local! {
    pub(super) static ONE_SIDED_PUBLICATION_PROBE: Cell<OneSidedPublicationProbe> = Cell::default();
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
    pub(super) static FACTOR_BUFFER_BUILD_COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
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
pub(super) enum FactorSide {
    Left,
    Right,
}

#[derive(Clone, Copy)]
pub(super) enum FactorPlacement {
    Direct,
    Adjoint,
}

pub(super) fn build_bound_factor_space<R>(
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
pub(super) fn build_left_right_bound_pair<R, D>(
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

pub(super) fn build_left_bound_factor<R, D, M>(
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

pub(super) fn build_bound_factor<R, D, M>(
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
pub(super) fn build_bound_factor_with_placement<R, D, M>(
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
    if let Some(identities) = one_sided_factor_output_plan(
        space.space().structure(),
        matricizations,
        pairs,
        dimensions,
        required_len,
        side,
        source_trees,
    ) {
        let data = take_one_sided_factors(pairs, &identities, required_len, side);
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

pub(super) fn scatter_left_sector_blocks<D, M>(
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
        debug_assert_eq!(coupled_of(key.codomain_tree()), matrix.sector());
        let (row_offset, _) =
            index.placement(matrix.sector(), FactorSide::Left, key.codomain_tree())?;
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

pub(super) fn scatter_right_sector_blocks<D, M>(
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
        debug_assert_eq!(coupled_of(key.domain_tree()), matrix.sector());
        let (col_offset, _) =
            index.placement(matrix.sector(), FactorSide::Right, key.domain_tree())?;
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

pub(super) fn reorder_columns_in_place<D: Copy>(
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

/// The test probes follow the scope body: it may run on a backend worker
/// thread, and the probes are thread-local so that parallel tests stay apart.
/// Moving them in and out makes a probe observe the call, whichever thread
/// runs it.
#[cfg(test)]
macro_rules! test_probes {
    ($($field:ident: $key:ident: $ty:ty),* $(,)?) => {
        struct TestProbes {
            $($field: $ty,)*
        }

        impl TestProbes {
            pub(super) fn take() -> Self {
                Self { $($field: $key.take(),)* }
            }

            pub(super) fn put(self) {
                $($key.set(self.$field);)*
            }
        }
    };
}

#[cfg(test)]
test_probes! {
    compact_svd: COMPACT_SVD_COPY_PROBE: CompactSvdCopyProbe,
    compact_qr: COMPACT_QR_COPY_PROBE: CompactQrCopyProbe,
    eigh: EIGH_COPY_PROBE: EighCopyProbe,
    eigh_vectors: EIGH_OWNED_VECTOR_POINTERS: Vec<usize>,
    checked_eigh_pairs: CHECKED_EIGH_PAIR_POINTERS: Vec<usize>,
    checked_svd_stage: CHECKED_COMPACT_SVD_STAGE_POINTERS: Vec<(usize, usize)>,
    generic_svd_fallback: GENERIC_COMPACT_SVD_FALLBACK_POINTERS: Vec<(usize, usize)>,
    mf_svd_fallback: MF_COMPACT_SVD_FALLBACK_POINTERS: Vec<(usize, usize)>,
    compact_lq: COMPACT_LQ_COPY_PROBE: CompactLqCopyProbe,
    diagonal_bond: DIAGONAL_BOND_BUILD_PROBE: DiagonalBondBuildProbe,
    values_fallbacks: VALUES_MATRICIZATION_FALLBACKS: usize,
    checked_inputs: CHECKED_COMPACT_INPUT_OBSERVATIONS: Vec<CheckedCompactInputObservation>,
    plan_finish: GENERIC_FACTOR_PLAN_FINISH_CALLS: usize,
    pair_publication: GENERIC_PAIR_PUBLICATION_PROBE: GenericPairPublicationProbe,
    one_sided: ONE_SIDED_PUBLICATION_PROBE: OneSidedPublicationProbe,
    buffer_builds: FACTOR_BUFFER_BUILD_COUNTS: (usize, usize),
    placement_index: PLACEMENT_INDEX_PROBE: PlacementIndexProbe,
    scatter_visits: SCATTER_VISIT_PROBE: ScatterVisitProbe,
}

/// Runs one streaming per-block factorization loop inside a single executor
/// linear-algebra scope: the backend admits the call once, while the loop
/// still holds only one block's input and output at a time. Batching through
/// `factorize_batch` would instead hold every block's factors at once.
pub(super) fn in_linalg_scope<E, T>(
    dense: &mut E,
    body: impl FnOnce(&mut dyn DenseExecutor) -> Result<T, OperationError> + Send,
) -> Result<T, OperationError>
where
    E: DenseExecutor + ?Sized,
    T: Send,
{
    let mut body = Some(body);
    let mut outcome = None;
    #[cfg(test)]
    let mut probes = Some(TestProbes::take());
    let scoped = dense.with_linalg_scope(&mut |dense| {
        tenet_tensors::host_pool::observe_dense_site();
        if let Some(body) = body.take() {
            #[cfg(test)]
            if let Some(probes) = probes.take() {
                probes.put();
            }
            outcome = Some(body(dense));
            #[cfg(test)]
            {
                probes = Some(TestProbes::take());
            }
        }
        Ok(())
    });
    #[cfg(test)]
    if let Some(probes) = probes.take() {
        probes.put();
    }
    scoped.map_err(OperationError::Dense)?;
    outcome.unwrap_or_else(|| {
        Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "with_linalg_scope",
            message: "dense executor returned without running the scope body".to_string(),
        }))
    })
}

pub(super) fn factorize_col_major_batch<E, D>(
    dense: &mut E,
    op: DenseFactorization,
    blocks: &[(&[D], usize, usize)],
) -> Result<Vec<Vec<DenseTensor>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let layouts = blocks
        .iter()
        .map(|&(_, rows, cols)| ([rows, cols], [1usize, rows]))
        .collect::<Vec<_>>();
    let inputs = blocks
        .iter()
        .zip(&layouts)
        .map(|(&(data, _, _), (shape, strides))| {
            DenseView::new(data, shape, strides, 0).map(D::dense_read)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(OperationError::Dense)?;
    let outputs = dense
        .factorize_batch(op, &inputs)
        .map_err(OperationError::Dense)?;
    // An overriding executor is outside this crate: a short batch would
    // otherwise drop a sector silently or panic in a consumer. Each entry's
    // factor count is checked by `compact_{qr,svd}_outputs`, with the same
    // error the per-matrix path reports.
    if outputs.len() != blocks.len() {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "factorize_batch",
            message: format!(
                "dense factorize_batch returned {} entries for {} inputs",
                outputs.len(),
                blocks.len()
            ),
        }));
    }
    Ok(outputs)
}

pub(super) fn compact_factor_output_owned<D: FactorScalar>(
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
        .try_fold(1usize, |acc, &dim| match acc.checked_mul(dim) {
            Some(count) => Ok(count),
            None => Err(DenseError::ElementCountOverflow),
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

pub(super) fn compact_real_spectrum_owned<D: FactorScalar>(
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
        .try_fold(1usize, |acc, &dim| match acc.checked_mul(dim) {
            Some(count) => Ok(count),
            None => Err(DenseError::ElementCountOverflow),
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

pub(super) fn concat_compact_factor_regions<D>(
    regions: Vec<Option<Vec<D>>>,
    required_len: usize,
) -> Vec<D> {
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

pub(super) fn concat_owned_factor_regions<D>(
    regions: Vec<Option<Vec<D>>>,
    required_len: usize,
) -> Vec<D> {
    let mut output = None;
    for region in regions.into_iter().flatten() {
        append_owned_factor(&mut output, region, required_len);
    }
    output.unwrap_or_default()
}

pub(super) fn copy_col_major_strided<D: Copy>(
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

pub(super) fn advance_outer_index(index: &mut [usize], shape: &[usize]) {
    for axis in 1..shape.len() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            break;
        }
        index[axis] = 0;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn copy_tensor_block_to_matrix<D: Copy>(
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

pub(super) fn copy_mapped_to_strided_diagonal<D, V, F>(
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
pub(super) fn scatter_matrix_block<D: Copy>(
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
pub(super) fn scatter_identity_matrix_block<D: FactorScalar>(
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
    pub(super) static PLACEMENT_INDEX_PROBE: Cell<PlacementIndexProbe> = Cell::default();
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
pub(super) struct PlacementIndex<'a> {
    pub(super) by_tree: FxHashMap<(SectorId, FactorSide, &'a FusionTreeKey), (usize, &'a [usize])>,
}

impl<'a> PlacementIndex<'a> {
    pub(super) fn new<M: SectorGeometry>(matricizations: &'a [M], sides: &[FactorSide]) -> Self {
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

    pub(super) fn placement(
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
    pub(super) static SCATTER_VISIT_PROBE: Cell<ScatterVisitProbe> = Cell::default();
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
pub(super) fn record_scatter_visit(side: FactorSide) {
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
pub(super) struct SectorBlockGroups {
    /// `(coupled sector, block index)` sorted by sector; the stable sort keeps
    /// ascending structure order within a sector.
    pub(super) entries: Vec<(SectorId, usize)>,
}

impl SectorBlockGroups {
    pub(super) fn new(
        structure: &BlockStructure,
        side: FactorSide,
    ) -> Result<Self, OperationError> {
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

    pub(super) fn blocks(&self, sector: SectorId) -> impl Iterator<Item = usize> + '_ {
        let start = self.entries.partition_point(|&(s, _)| s < sector);
        self.entries[start..]
            .iter()
            .take_while(move |&&(s, _)| s == sector)
            .map(|&(_, index)| index)
    }
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

#[cfg(test)]
pub(super) fn record_generic_pair_ordered_key_validation() {
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.ordered_key_validation_events += 1;
        probe.set(value);
    });
}

#[cfg(not(test))]
pub(super) fn record_generic_pair_ordered_key_validation() {}

pub(super) struct FactorTreeCursor<'a, M> {
    pub(super) matricizations: &'a [M],
    pub(super) ranks: &'a [SectorRank],
    pub(super) matrix: usize,
    pub(super) tree: usize,
    pub(super) valid: bool,
}

impl<'a, M: SectorGeometry> FactorTreeCursor<'a, M> {
    pub(super) fn new(matricizations: &'a [M], ranks: &'a [SectorRank]) -> Self {
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

    pub(super) fn next(&mut self, side: FactorSide) -> Option<(SectorId, &'a FusionTreeKey)> {
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

    pub(super) fn matches(&mut self, side: FactorSide, key: &FusionTreeKey) -> bool {
        record_generic_pair_ordered_key_validation();
        if !self.valid {
            return false;
        }
        self.next(side)
            .is_some_and(|(sector, tree)| coupled_of_generic(key) == sector && key == tree)
    }

    pub(super) fn is_exhausted(&mut self, side: FactorSide) -> bool {
        self.next(side).is_none() && self.valid
    }
}

#[cfg(test)]
pub(super) fn record_generic_pair_fallback_lookup(side: FactorSide) {
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
pub(super) fn record_generic_pair_fallback_lookup(_side: FactorSide) {}

/// Staged keys of a prepared checked layout in enumeration order. The
/// prepared structure lists its blocks exactly as the enumeration produced
/// its keys and `commit` interns them without reordering, so this sequence
/// is the key order of the committed space; the checked builders validate
/// and commit from one enumeration instead of enumerating keys separately.
pub(super) fn staged_fusion_tree_keys(
    sector: &SectorStructure,
) -> impl Iterator<Item = &FusionTreePairKey> {
    sector
        .blocks()
        .iter()
        .filter_map(|block| match block.key() {
            BlockKey::FusionTree(key) => Some(key),
            _ => None,
        })
}

pub(super) fn validate_generic_factor_keys<'a, 'k, M: SectorGeometry>(
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

pub(super) fn checked_extent(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |extent, &dim| extent.checked_mul(dim))
}

/// `keys` carries the separately enumerated key list of the unchecked paired
/// route for the admitted-key equality against `structure`; the checked route
/// commits the very structure it validated and passes `None`.
pub(super) fn factor_output_is_canonical<D, M: SectorGeometry>(
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
pub(super) fn sector_factor_output_is_canonical<M: SectorGeometry>(
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
        factor_block_is_canonical(
            &block,
            tree.shape,
            tree.offset,
            output_offset,
            source_extent,
            bond,
            side,
        )?;
        *block_index += 1;
    }
    if tree_prefix != source_extent {
        return None;
    }
    output_offset.checked_add(factor_len)
}

/// Proves that `block` holds the side-tree span `[tree_offset, tree_offset +
/// |tree_shape|)` of a column-major sector factor starting at `output_offset`
/// (left `a x bond`, right `bond x a`, `a = source_extent`).
pub(super) fn factor_block_is_canonical(
    block: &BlockRef<'_>,
    tree_shape: &[usize],
    tree_offset: usize,
    output_offset: usize,
    source_extent: usize,
    bond: usize,
    side: FactorSide,
) -> Option<()> {
    let block_offset = match side {
        FactorSide::Left => output_offset.checked_add(tree_offset)?,
        FactorSide::Right => output_offset.checked_add(bond.checked_mul(tree_offset)?)?,
    };
    if block.offset() != block_offset
        || block.shape().len() != tree_shape.len() + 1
        || block.strides().len() != block.shape().len()
    {
        return None;
    }
    let shape_matches = match side {
        FactorSide::Left => {
            &block.shape()[..tree_shape.len()] == tree_shape
                && block.shape()[tree_shape.len()] == bond
        }
        FactorSide::Right => block.shape()[0] == bond && &block.shape()[1..] == tree_shape,
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
    for (axis, &dim) in tree_shape.iter().enumerate() {
        if block.strides()[axis + stride_offset] != stride {
            return None;
        }
        stride = stride.checked_mul(dim)?;
    }
    if matches!(side, FactorSide::Left) && block.strides()[tree_shape.len()] != source_extent {
        return None;
    }
    Some(())
}

/// Identity-completed sector (bond states but no source matricization): its
/// output blocks, in output order, must tile a column-major `bond x bond`
/// identity at `output_offset`, with the output tree order as the side basis
/// exactly as the scatter fallback numbers it. A sector with no output blocks
/// on this side (e.g. the right factor of a codomain-only full-QR sector)
/// publishes nothing. Returns the next offset and whether an identity is due.
pub(super) fn identity_sector_output_is_canonical(
    structure: &BlockStructure,
    block_index: &mut usize,
    output_offset: usize,
    sector: SectorId,
    bond: usize,
    side: FactorSide,
) -> Option<(usize, bool)> {
    let mut tree_prefix = 0usize;
    while let Ok(block) = structure.block(*block_index) {
        let BlockKey::FusionTree(key) = block.key() else {
            return None;
        };
        let tree = match side {
            FactorSide::Left => key.codomain_tree(),
            FactorSide::Right => key.domain_tree(),
        };
        if coupled_of(tree) != sector {
            break;
        }
        let tree_shape = match side {
            FactorSide::Left => block.shape().split_last()?.1,
            FactorSide::Right => block.shape().split_first()?.1,
        };
        factor_block_is_canonical(
            &block,
            tree_shape,
            tree_prefix,
            output_offset,
            bond,
            bond,
            side,
        )?;
        tree_prefix = tree_prefix.checked_add(checked_extent(tree_shape)?)?;
        *block_index += 1;
    }
    if tree_prefix == 0 {
        Some((output_offset, false))
    } else if tree_prefix == bond {
        Some((output_offset.checked_add(bond.checked_mul(bond)?)?, true))
    } else {
        None
    }
}

/// A column-major `bond x bond` identity for a sector without a source
/// matricization (MatrixAlgebraKit `one!` on a zero-extent input block),
/// published after the first `pairs_before` pairs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct IdentitySegment {
    pub(super) pairs_before: usize,
    pub(super) bond: usize,
}

/// One-sided sibling of [`factor_output_is_canonical`]: walks the ascending
/// union of source sectors and admitted bond sectors. A populated sector must
/// carry its pair and publish it by [`sector_factor_output_is_canonical`]; a
/// populated sector without a pair must have no bond (a null sector omitted
/// as full rank); a bond-only sector must pass
/// [`identity_sector_output_is_canonical`]. The whole admitted output must be
/// exhausted and every pair consumed, so unordered, duplicate or extraneous
/// records return `None` and keep the scatter fallback. `source_trees`
/// selects the matricization side whose trees index the selected factor
/// (columns for adjoint placement).
///
/// Every pair is consumed in order, so the plan records only the identity
/// segments; the list stays unallocated when there are none, keeping the
/// all-populated case free of plan allocation.
pub(super) fn one_sided_factor_output_plan<D, M: SectorGeometry>(
    structure: &BlockStructure,
    matricizations: &[M],
    pairs: &[FactorPair<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    required_len: usize,
    side: FactorSide,
    source_trees: FactorSide,
) -> Option<Vec<IdentitySegment>> {
    let mut identities = Vec::new();
    let mut matrices = matricizations.iter().peekable();
    let mut pair_records = pairs.iter().enumerate().peekable();
    let mut bonds = dimensions.iter().peekable();
    let mut previous_matrix = None;
    let mut block_index = 0usize;
    let mut output_offset = 0usize;
    loop {
        let matrix_sector = matrices.peek().map(|matrix| matrix.sector());
        let sector = match (matrix_sector, bonds.peek().map(|(&sector, _)| sector)) {
            (None, None) => break,
            (Some(matrix), Some(bond)) => matrix.min(bond),
            (Some(sector), None) | (None, Some(sector)) => sector,
        };
        let bond = bonds
            .next_if(|&(&candidate, _)| candidate == sector)
            .map(|(_, &bond)| bond);
        if matrix_sector != Some(sector) {
            let (next_offset, identity) = identity_sector_output_is_canonical(
                structure,
                &mut block_index,
                output_offset,
                sector,
                bond?,
                side,
            )?;
            if identity {
                identities.push(IdentitySegment {
                    pairs_before: pair_records.peek().map_or(pairs.len(), |&(index, _)| index),
                    bond: bond?,
                });
            }
            output_offset = next_offset;
            continue;
        }
        let matrix = matrices.next()?;
        if previous_matrix.is_some_and(|previous| previous >= sector) {
            return None;
        }
        previous_matrix = Some(sector);
        let Some((_, pair)) = pair_records.next_if(|(_, pair)| pair.sector == sector) else {
            if bond.unwrap_or(0) != 0 {
                return None;
            }
            continue;
        };
        let (factor_len, factor_leading) = match side {
            FactorSide::Left => (pair.left.len(), pair.left_rows),
            FactorSide::Right => (pair.right.len(), pair.right_leading),
        };
        output_offset = sector_factor_output_is_canonical(
            structure,
            &mut block_index,
            output_offset,
            matrix,
            source_trees,
            factor_len,
            factor_leading,
            bond?,
            side,
            None,
        )?;
    }
    if pair_records.next().is_some()
        || block_index != structure.block_count()
        || output_offset != required_len
    {
        return None;
    }
    #[cfg(test)]
    ONE_SIDED_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.plan_bytes += identities.capacity() * std::mem::size_of::<IdentitySegment>();
        probe.set(value);
    });
    Some(identities)
}

/// Publishes the plan proved by [`one_sided_factor_output_plan`]: the
/// selected side of each pair is moved or appended and each identity is
/// written once in place, so no output element is written twice. The
/// opposite side stays in place for its own publication.
pub(super) fn take_one_sided_factors<D: FactorScalar>(
    pairs: &mut [FactorPair<D>],
    identities: &[IdentitySegment],
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
    let mut _written = 0usize;
    let mut identities = identities.iter().peekable();
    for index in 0..=pairs.len() {
        while let Some(identity) = identities.next_if(|identity| identity.pairs_before == index) {
            let bond = identity.bond;
            let data = output.get_or_insert_with(|| Vec::with_capacity(required_len));
            let start = data.len();
            data.resize(start + bond * bond, D::zero());
            for diagonal in 0..bond {
                data[start + diagonal * (bond + 1)] = D::one();
            }
            _written += bond * bond;
        }
        let Some(pair) = pairs.get_mut(index) else {
            break;
        };
        let factor = match side {
            FactorSide::Left => std::mem::take(&mut pair.left),
            FactorSide::Right => std::mem::take(&mut pair.right),
        };
        _written += append_owned_factor(&mut output, factor, required_len);
    }
    let data = output.unwrap_or_default();
    #[cfg(test)]
    ONE_SIDED_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.canonical_publications += 1;
        value.owner_reused +=
            usize::from(first.is_some_and(|pointer| std::ptr::eq(pointer, data.as_ptr())));
        value.appended_elements += _written;
        probe.set(value);
    });
    data
}

#[cfg(test)]
pub(super) fn record_one_sided_fallback_publication() {
    ONE_SIDED_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.fallback_publications += 1;
        probe.set(value);
    });
}

#[cfg(not(test))]
pub(super) fn record_one_sided_fallback_publication() {}

#[cfg(test)]
pub(super) fn record_generic_pair_output_block_visit() {
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.output_blocks_visited += 1;
        probe.set(value);
    });
}

#[cfg(not(test))]
pub(super) fn record_generic_pair_output_block_visit() {}

#[cfg(test)]
pub(super) fn record_generic_pair_appended(left: usize, right: usize) {
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.left_appended_elements += left;
        value.right_appended_elements += right;
        probe.set(value);
    });
}

#[cfg(not(test))]
pub(super) fn record_generic_pair_appended(_left: usize, _right: usize) {}

/// Appends `factor` to `output`, letting the first nonempty factor keep its
/// allocation (grown once to `required_len`). Returns the element count that
/// was appended to an existing owner, zero otherwise.
pub(super) fn append_owned_factor<D>(
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

pub(super) fn publish_generic_factor_pairs<D>(
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
pub(super) fn build_left_right_bound_spaces_generic<R, M>(
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

pub(super) fn build_left_right_bound_spaces_and_keys_generic<R, M>(
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

pub(super) fn build_left_bound_space_generic<'a, R, M>(
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

pub(super) fn build_right_bound_space_generic<'a, R, M>(
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

pub(super) fn build_left_right_bound_pair_generic<R, D, M>(
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

pub(super) fn build_left_right_bound_pair_generic_checked<R, D, M>(
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

pub(super) fn build_checked_pair_from_input<R, D>(
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
pub(super) fn build_bound_factor_generic_checked<R, D, M>(
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
    if let Some(identities) = one_sided_factor_output_plan(
        space.space().structure(),
        matricizations,
        pairs,
        dimensions,
        len,
        side,
        side,
    ) {
        let data = take_one_sided_factors(pairs, &identities, len, side);
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

/// Generic sibling of [`scatter_left_sector_blocks`].
pub(super) fn scatter_left_sector_blocks_generic<D, M>(
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
pub(super) fn scatter_right_sector_blocks_generic<D, M>(
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

/// Why not the paired builder: its bond holds only sectors that carry a
/// factor pair, so a side-only sector would be dropped from the full
/// factor's `fuse(codomain)`/`fuse(domain)` bond instead of publishing its
/// identity block, as TensorKit's `initialize_output(qr_full!/lq_full!)` does.
pub(super) fn checked_full_factor_pair<R, D, M>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matrices: &[M],
    mut pairs: Vec<FactorPair<D>>,
    dimensions: &BTreeMap<SectorId, usize>,
) -> Result<DynamicFactorPair<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
    M: SectorGeometry,
{
    let left = build_bound_factor_generic_checked(
        provider,
        homspace,
        matrices,
        &mut pairs,
        dimensions,
        FactorSide::Left,
    )?;
    let right = build_bound_factor_generic_checked(
        provider,
        homspace,
        matrices,
        &mut pairs,
        dimensions,
        FactorSide::Right,
    )?;
    Ok((left, right))
}
