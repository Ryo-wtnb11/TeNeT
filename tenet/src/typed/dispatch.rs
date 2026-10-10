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

    /// The provider's braiding style, which admits contraction and selects
    /// the twist/flip early returns.
    #[doc(hidden)]
    fn braiding_style(provider: &R) -> tenet_core::BraidingStyleKind;

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
    /// `operation` (a rank-(1,1) leg swap or braid) applied to the compact
    /// `spectrum` of `tensor` in one staging: the committed destination and
    /// either the spectrum carried through the resolved single-term square
    /// bijection, or (when that proof or `require_representable` fails) the
    /// dense replay of the densified spectrum with the same structure.
    fn transform_bond_spectrum(
        tensor: &TensorMap<R, D>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
        operation: &TreeTransformOperation,
        require_representable: bool,
    ) -> Result<BondTransform<R, D>, Self::FacadeError>;

    /// The lazy adjoint of the transformed parent for a lazy-adjoint input,
    /// else `None`. The method is where a mode discharges
    /// [`TypedAdjointSpace`]; the default `None` is for a mode that cannot
    /// hold a lazy adjoint.
    fn try_lazy_adjoint_transform(
        _tensor: &TensorMap<R, D>,
        _operation: &TreeTransformOperation,
    ) -> Result<Option<TensorMap<R, D>>, Self::FacadeError> {
        Ok(None)
    }

    /// Transforms the owned dense payload of `tensor` (a compact diagonal
    /// densified) into a fresh destination payload. A lazy adjoint never
    /// reaches it: `try_lazy_adjoint_transform` takes it first.
    fn transform(
        tensor: &TensorMap<R, D>,
        operation: TreeTransformOperation,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError>;
}

/// The committed destination of a compact rank-(1,1) transform and its
/// payload.
///
/// Internal and unstable despite being public for the dispatch traits.
#[doc(hidden)]
pub struct BondTransform<R, D> {
    pub destination: BoundDynamicFusionMapSpace<R>,
    pub output: BondOutput<D>,
}

/// A compact rank-(1,1) transform's payload: the carried spectrum, or the
/// dense replay of the densified spectrum when the bijection proof declined.
///
/// Internal and unstable despite being public for the dispatch traits.
#[doc(hidden)]
pub enum BondOutput<D> {
    Spectrum(Vec<tenet_matrixalgebra::SectorSpectrum<D>>),
    Dense(Vec<D>),
}

pub(super) trait MultiplicityFreeTransformExecution<R, C>: TensorScalar
where
    R: TypedSectorAdmission,
{
    fn transform_bond_spectrum(
        tensor: &TensorMap<R, Self>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<Self>],
        operation: &TreeTransformOperation,
        require_representable: bool,
    ) -> Result<BondTransform<R, Self>, Error>;

    fn try_lazy_adjoint(
        _tensor: &TensorMap<R, Self>,
        _operation: &TreeTransformOperation,
    ) -> Result<Option<TensorMap<R, Self>>, Error> {
        Ok(None)
    }

    fn transform(
        tensor: &TensorMap<R, Self>,
        operation: TreeTransformOperation,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<Self>), Error>;
}

pub(super) trait MultiplicityFreeContractExecution<R: TypedSectorAdmission, C>:
    TensorScalar
{
    fn contract(
        lhs: &TensorMap<R, Self>,
        rhs: &TensorMap<R, Self>,
        spec: &ContractSpec<'_>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<Self>), Error>;

    fn compose(
        lhs: &TensorMap<R, Self>,
        rhs: &TensorMap<R, Self>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<Self>), Error>;
}

/// Ribbon-twist execution selected by the admitted provider mode.
#[doc(hidden)]
pub trait TypedTensorTwistDispatch<R, D>: TypedSpaceModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// θ of every sector that `legs` carry in `structure`'s fusion-tree
    /// blocks, or `None` when every one of them is one. Every fallible
    /// provider query happens here, so scaling by the returned values never
    /// fails on the provider.
    fn twist_values<'a>(
        provider: &'a R,
        structure: &tenet_core::BlockStructure,
        codomain_rank: usize,
        legs: &[usize],
    ) -> Result<Option<impl Fn(SectorId) -> f64 + 'a>, Self::FacadeError>;
}

/// Z-isomorphism execution selected by the admitted provider mode.
#[doc(hidden)]
pub trait TypedTensorFlipDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// The admitted space of `homspace`, bound to `space`'s provider.
    fn root(
        space: &BoundDynamicFusionMapSpace<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError>;

    /// (χ, θ) of every sector the flip `occurrences` carry in `structure`'s
    /// fusion-tree blocks, with every fallible provider query made here.
    fn pivotal_values<'a>(
        provider: &'a R,
        structure: &tenet_core::BlockStructure,
        codomain_rank: usize,
        occurrences: &[(usize, bool)],
    ) -> Result<impl Fn(SectorId) -> (f64, f64) + 'a, Self::FacadeError>;
}

/// Tensor-product execution selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorProductDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Executes the F-only product of two owned dense payloads.
    fn product(
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError>;
}

/// Contraction execution selected by a provider-owned mode.
///
/// Why a transform supertrait: the shared bodies call the mode-free compact
/// arms, whose reordered `t · D` / `D · t` result is laid out by one
/// [`TensorMap::permute`]-style tree transform.
#[doc(hidden)]
pub trait TypedTensorContractDispatch<R, D>: TypedTensorTransformDispatch<R, D>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Contracts two operands, a lazy adjoint read through its parent's
    /// payload, into a fresh destination payload.
    fn contract(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        spec: &ContractSpec<'_>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError>;

    /// Composes two operands, a lazy adjoint read through its parent's
    /// payload, into a fresh destination payload.
    fn compose(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError>;
}

/// Partial categorical trace execution selected by provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorTraceDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Stages the destination of an admitted trace source and runs the one
    /// owned trace into it; `payload` is read only after the compile, and
    /// not at all when `read` answers a scalar destination from the compiled
    /// terms (a compact spectrum).
    fn trace<P: AsRef<[D]>>(
        space: &BoundDynamicFusionMapSpace<R>,
        read: impl FnOnce(&tenet_tensors::TensorTraceFusionStructure<f64>) -> Option<D>,
        payload: impl FnOnce() -> P,
        axes: tenet_tensors::TensorTraceAxisSpec<'_>,
        dst_nout: usize,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError>;
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
