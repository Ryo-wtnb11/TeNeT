#[allow(unused_imports)]
use super::*;

/// Tensor-side layout admission selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorModeDispatch<R>: typed_admission_private::Sealed
where
    R: TypedSectorAdmission,
{
    /// Error returned by the ordinary typed facade.
    type FacadeError: std::error::Error + From<Error>;

    /// Preserves a provider-side admission error.
    fn map_provider_error(error: R::Error) -> Self::FacadeError;

    /// Lowers a facade error to [`Error`] for the provider-neutral
    /// [`TensorRef`] adjoint constructor.
    #[doc(hidden)]
    fn facade_error_into_error(error: Self::FacadeError) -> Error;

    /// TensorKit `blockdim(P, c)`: the reduced dimension of coupled sector
    /// `coupled` in the product space `product`, zero when it does not occur.
    #[doc(hidden)]
    fn coupled_block_dimension(
        provider: &R,
        product: &FusionProductSpace,
        coupled: SectorId,
    ) -> Result<usize, Self::FacadeError>;

    /// TensorKit `fuse` of two sector contents, `N`-symbol weighted and in
    /// sector-id order.
    #[doc(hidden)]
    fn fuse_sector_content(
        provider: &R,
        left: &[(SectorId, usize)],
        right: &[(SectorId, usize)],
    ) -> Result<Vec<(SectorId, usize)>, Self::FacadeError>;
}

/// Tensor-side root construction selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorRootDispatch<R>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
{
    /// Builds one fully admitted root while retaining `provider`.
    fn build_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError>;
}

/// Tensor-side construction gate for payload/provider coefficient pairs.
#[doc(hidden)]
pub trait TypedTensorConstructionDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Builds one fully admitted root while retaining `provider`.
    fn build_construction_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError>;
}

/// Tensor-side tree-transform execution selected by a provider-owned mode.
///
/// ```
/// use tenet::sector::{CheckedGenericAdmissionMode, CheckedGenericRigidSymbols, TypedSectorAdmission};
/// use tenet::typed::TensorMap;
///
/// fn checked_transpose<R>(tensor: &TensorMap<R, f64>)
/// where
///     R: TypedSectorAdmission<
///             Error = <R as tenet::sector::CheckedGenericFusion>::Error,
///             Mode = CheckedGenericAdmissionMode,
///         > + CheckedGenericRigidSymbols<Scalar = f64>,
/// {
///     let _ = tensor.transpose(&[1], &[0]);
/// }
/// ```
#[doc(hidden)]
pub trait TypedTensorTransformDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Executes one admitted permutation, braid, or planar transpose.
    fn tree_transform(
        tensor: &TensorMap<R, D>,
        operation: TreeTransformOperation,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

pub(super) trait MultiplicityFreeTransformExecution<R, C>: TensorScalar {
    fn execute(
        tensor: &TensorMap<R, Self>,
        operation: TreeTransformOperation,
    ) -> Result<TensorMap<R, Self>, Error>
    where
        R: TypedSectorAdmission;
}

pub(super) trait MultiplicityFreeContractExecution<R: TypedSectorAdmission, C>:
    TensorScalar
{
    fn contract(
        lhs: &TensorMap<R, Self>,
        rhs: &TensorMap<R, Self>,
        spec: &ContractSpec<'_>,
    ) -> Result<TensorMap<R, Self>, Error>;

    fn compose(
        lhs: &TensorMap<R, Self>,
        rhs: &TensorMap<R, Self>,
    ) -> Result<TensorMap<R, Self>, Error>;
}

/// Ribbon-twist execution selected by the admitted provider mode.
#[doc(hidden)]
pub trait TypedTensorTwistDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    fn twist(
        tensor: &TensorMap<R, D>,
        legs: &[usize],
        inverse: bool,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

/// Z-isomorphism execution selected by the admitted provider mode.
#[doc(hidden)]
pub trait TypedTensorFlipDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    fn flip(
        tensor: &TensorMap<R, D>,
        legs: &[usize],
        inverse: bool,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTwistDispatch<R, D>,
    D: TensorScalar,
{
    /// TensorKit `twist(t, inds; inv)` (and its in-place `twist!`): multiplies
    /// each fusion-tree block by the product over `legs` (flat leg indices,
    /// codomain first) of that leg's ribbon-twist eigenvalue, or by its
    /// inverse for [`Direction::Inverse`].
    ///
    /// A bosonic provider, a NoBraiding provider restricted to unit legs, or
    /// selected sectors whose staged twists are all one returns a body-sharing
    /// clone, matching TensorKit's `copy = false` behavior. Otherwise the
    /// operation publishes one fresh scaled payload on the exact admitted
    /// space and provider allocation. A lazy adjoint redirects through its
    /// parent with the inverse operation. Multiplicity-free compact spectra
    /// retain their existing representation-preserving path; a checked-Generic
    /// compact factor is densified into an operation-local buffer first.
    ///
    /// # Errors
    ///
    /// An out-of-range leg is rejected before the empty-list short circuit.
    /// Non-unit NoBraiding legs are invalid. Checked-Generic pivotal failures
    /// retain their typed provider error, and no result is published until all
    /// selected twist values have been staged successfully.
    pub fn twist(&self, legs: &[usize], direction: Direction) -> Result<Self, TypedFacadeError<R>> {
        <R::Mode as TypedTensorTwistDispatch<R, D>>::twist(self, legs, direction.is_inverse())
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorFlipDispatch<R, D>,
    D: TensorScalar,
{
    /// TensorKit `flip(t, I; inv)`:
    /// return a tensor isomorphic to `self` where the duality flag of each
    /// leg in `legs` (flat indices, codomain first; a leg listed twice is
    /// flipped twice, sequentially) is toggled,
    /// `space(t', i) = flip(space(t, i))`. The stored sectors and the block
    /// layout are unchanged; each fusion-tree block picks up the
    /// Z-isomorphism phase of TensorKit's fusion-tree `flip`
    /// per flipped leg with uncoupled sector `a` and pre-flip duality `d`
    /// (χ = Frobenius–Schur phase, θ = ribbon twist; both real for every
    /// rule in scope): codomain leg → `d ? χ·θ : 1`; domain leg →
    /// `d ? χ : θ`.
    ///
    /// Like TensorKit's, this `flip` is *not* an involution: flipping the
    /// same leg twice returns to the original spaces but can scale odd
    /// blocks (e.g. by θ = −1 on fermionic legs); only `flip⁴ = id` in
    /// general. [`Direction::Inverse`] applies the inverse isomorphism.
    ///
    /// One scaled copy of the dense payload into a fresh body, O(len); a
    /// compact spectrum factor materializes first (the flipped space is no
    /// longer a bond space, so the result cannot stay compact). The same
    /// facade narrowings as [`Self::twist`] apply: a lazy dense adjoint
    /// redirects through the parent with the inverse categorical map without
    /// materializing; there is no device arm, and checked Generic follows its
    /// provider-mode dispatch.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a leg is out of range, before the
    /// empty-list short-circuit; empty `legs` returns an identical clone.
    /// Otherwise [`Error::Operation`] /
    /// [`Error::Core`] from the layout derivation of the toggled hom space.
    /// Checked-Generic admission and pivotal failures retain their typed
    /// [`GenericTensorError`] variants.
    pub fn flip(&self, legs: &[usize], direction: Direction) -> Result<Self, TypedFacadeError<R>> {
        <R::Mode as TypedTensorFlipDispatch<R, D>>::flip(self, legs, direction.is_inverse())
    }
}

/// Tensor-product execution selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorProductDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Executes the F-only product while preserving the left provider.
    fn tensor_product(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

/// Contraction execution selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorContractDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    fn contract(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        spec: &ContractSpec<'_>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;

    fn compose(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

/// Partial categorical trace execution selected by provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorTraceDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    fn trace_pairs(
        tensor: &TensorMap<R, D>,
        pairs: &[(usize, usize)],
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

mod typed_admission_private {
    use super::{CheckedGenericAdmissionMode, MultiplicityFreeAdmissionMode};

    pub trait Sealed {}

    impl Sealed for MultiplicityFreeAdmissionMode {}
    impl Sealed for CheckedGenericAdmissionMode {}
}

#[doc(hidden)]
pub trait TypedSpaceModeDispatch<R>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
{
    fn vacuum(provider: &R) -> SectorId;
    fn fusion_channels(
        provider: &R,
        left: SectorId,
        right: SectorId,
    ) -> Result<Vec<SectorId>, <Self as TypedTensorModeDispatch<R>>::FacadeError>;
    fn nsymbol(
        provider: &R,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, <Self as TypedTensorModeDispatch<R>>::FacadeError>;
    fn dim(
        provider: &R,
        sector: SectorId,
    ) -> Result<f64, <Self as TypedTensorModeDispatch<R>>::FacadeError>;
}

/// The bond-truncation decision, selected by a provider-owned mode.
///
/// Both arms call `tenet_matrixalgebra::seam::decide_bond_truncation*`, the one
/// adapter over `select_truncation` that supplies the quantum-dimension weight
/// and TensorKit's sector order.
#[doc(hidden)]
pub trait TypedTruncationDispatch<R>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
{
    fn decide_bond_truncation<V>(
        provider: &R,
        spectra: &[tenet_matrixalgebra::SectorSpectrum<V>],
        truncation: &Truncation,
    ) -> Result<tenet_matrixalgebra::TruncationDecision, Self::FacadeError>
    where
        V: SpectrumMagnitude;
}

#[doc(hidden)]
pub trait TypedTensorReductionDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    fn inner(tensor: &TensorMap<R, D>, other: &TensorMap<R, D>) -> Result<D, Self::FacadeError>;
    fn norm(tensor: &TensorMap<R, D>, p: f64) -> Result<f64, Self::FacadeError>;
    fn tr(tensor: &TensorMap<R, D>) -> Result<D, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorAddScaleDispatch<R, D>:
    TypedTensorModeDispatch<R> + TypedTensorAdjointDispatch<R, D>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    fn axpby(
        tensor: &TensorMap<R, D>,
        alpha: D,
        other: &TensorMap<R, D>,
        beta: D,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
    fn scale(tensor: &TensorMap<R, D>, factor: D) -> TensorMap<R, D>;
}

#[doc(hidden)]
pub trait TypedTensorAdjointDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    fn adjoint(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorInvDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn inv(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorSolveDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn solve(
        tensor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorPinvDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn pinv(tensor: &TensorMap<R, D>, rcond: f64) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorNullDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn left_null(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Self::FacadeError>;
    fn right_null(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorPolarDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn left_polar(
        tensor: &TensorMap<R, D>,
    ) -> Result<LeftPolar<TensorMap<R, D>>, Self::FacadeError>;
    fn right_polar(
        tensor: &TensorMap<R, D>,
    ) -> Result<RightPolar<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorExpDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn exp(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorQrDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn qr_compact(tensor: &TensorMap<R, D>) -> Result<Qr<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorSvdDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn svd_compact(tensor: &TensorMap<R, D>) -> Result<Svd<TensorMap<R, D>>, Self::FacadeError>;
    fn svd_full(tensor: &TensorMap<R, D>) -> Result<Svd<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorLqDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn lq_compact(tensor: &TensorMap<R, D>) -> Result<Lq<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorFullQrDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn qr_full(tensor: &TensorMap<R, D>) -> Result<Qr<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorFullLqDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn lq_full(tensor: &TensorMap<R, D>) -> Result<Lq<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorSvdValsDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn svd_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorEighValsDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn eigh_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorEighDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn eigh_full(tensor: &TensorMap<R, D>) -> Result<Eigh<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorEigValsDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn eig_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<
        Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, num_complex::Complex64>>,
        Self::FacadeError,
    >;
}

#[doc(hidden)]
pub trait TypedTensorEigDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn eig_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Eig<TensorMap<R, <D as FactorScalar>::Eig>>, Self::FacadeError>;
}
