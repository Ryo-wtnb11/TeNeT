use super::*;

/// Tensor-side layout admission selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorModeDispatch<R>: typed_admission_private::Sealed
where
    R: TypedSectorAdmission,
{
    /// Error returned by the ordinary typed facade.
    type FacadeError: std::error::Error + From<Error> + From<tenet_tensors::OperationError>;

    /// Preserves a provider-side admission error.
    fn map_provider_error(error: R::Error) -> Self::FacadeError;

    /// The provider's fusion style, which selects TensorKit's
    /// `UniqueFusion` whole-buffer reductions.
    #[doc(hidden)]
    fn fusion_style(provider: &R) -> tenet_core::FusionStyleKind;

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

    /// Derives the final HomSpace before the existing root admission, in one epoch.
    fn build_root_with(
        provider: Arc<R>,
        build_homspace: impl FnOnce() -> Result<FusionTreeHomSpace, Self::FacadeError>,
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

/// The one coefficient step `adjoint` needs from a mode: the adjoint space
/// (`tenet_tensors::CoefficientAlgebra::adjoint_space`) with its error in the
/// facade's type. Implemented for every mode that has both, so `adjoint` and
/// the operations built on it are one body for every mode.
#[doc(hidden)]
pub trait TypedAdjointSpace<R>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
{
    fn adjoint_space(
        space: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError>;
}

impl<R, M> TypedAdjointSpace<R> for M
where
    R: TypedSectorAdmission,
    M: TypedTensorModeDispatch<R> + tenet_tensors::CoefficientAlgebra<R>,
    M::FacadeError: From<<M as tenet_tensors::CoefficientAlgebra<R>>::Error>,
{
    fn adjoint_space(
        space: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        <M as tenet_tensors::CoefficientAlgebra<R>>::adjoint_space(space).map_err(Into::into)
    }
}
