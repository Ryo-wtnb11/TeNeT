use super::*;

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
pub trait TypedTensorEighDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn eigh_full(tensor: &TensorMap<R, D>) -> Result<Eigh<TensorMap<R, D>>, Self::FacadeError>;
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

/// The facade half of a fusion mode's factorization contract. Every
/// factorization has one body over it; the mode chooses only what differs by
/// design or is still owned by a unification leaf (#1862, table in its body):
/// the lazy-adjoint rule (D1, #1755), the error type and provider-label decode
/// (D4, permanent). The numerical stages and factor spaces come from the
/// matrix-algebra [`FactorMode`](tenet_matrixalgebra::seam::FactorMode).
#[doc(hidden)]
pub trait FusionMode<R>:
    TypedTensorModeDispatch<R> + tenet_matrixalgebra::seam::FactorMode<R>
where
    R: TypedSectorAdmission,
{
    /// How `op` reads a lazy adjoint input.
    fn adjoint_rule(op: FactorOp) -> AdjointRule;

    /// A matrix-algebra error as this mode's facade error.
    fn map_factor_error(
        error: <Self as tenet_matrixalgebra::seam::FactorMode<R>>::Error,
    ) -> Self::FacadeError;

    /// The public label of a coupled sector the provider's own algebra
    /// produced.
    fn decode_label(provider: &R, sector: SectorId) -> Result<R::Sector, Self::FacadeError>;
}

/// A factorization, as named in its errors.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactorOp {
    SvdVals,
    EighVals,
    EigVals,
    QrCompact,
    QrFull,
    LqCompact,
    LqFull,
    SvdCompact,
    SvdFull,
}

impl FactorOp {
    /// The checked-mode lazy-adjoint refusal (D1, #1755).
    pub(super) fn lazy_adjoint_refusal(self) -> &'static str {
        match self {
            Self::SvdVals => "checked Generic svd_vals does not accept lazy adjoints",
            Self::EighVals => "checked Generic eigh_vals does not accept lazy adjoints",
            Self::EigVals => "checked Generic eig_vals does not accept lazy adjoints",
            Self::QrCompact => "checked Generic qr_compact does not accept lazy adjoints",
            Self::QrFull => "checked Generic qr_full does not accept lazy adjoints",
            Self::LqCompact => "checked Generic lq_compact does not accept lazy adjoints",
            Self::LqFull => "checked Generic lq_full does not accept lazy adjoints",
            Self::SvdCompact => "checked Generic svd_compact does not accept lazy adjoints",
            Self::SvdFull => "checked Generic svd_full does not accept lazy adjoints",
        }
    }
}

/// What a factorization reads when its input is a lazy adjoint.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdjointRule {
    /// Refused with `InvalidArgument`.
    Reject,
    /// The dense parent, for a result that is invariant under adjoint.
    Parent,
    /// An operation-local materialized adjoint.
    Materialize,
    /// The factorization's adjoint partner on the parent, whose factors are
    /// adjointed back (LQ of `t^H` from QR of `t`).
    Redirect,
    /// The dense parent through the factorization's adjoint-aware stage,
    /// which returns the factors of the adjoint without forming it.
    AdjointSeam,
}

impl<R> FusionMode<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn adjoint_rule(op: FactorOp) -> AdjointRule {
        match op {
            // Singular values and coupled sectors are invariant under adjoint.
            FactorOp::SvdVals => AdjointRule::Parent,
            FactorOp::EighVals | FactorOp::EigVals | FactorOp::QrCompact | FactorOp::QrFull => {
                AdjointRule::Materialize
            }
            FactorOp::LqCompact | FactorOp::LqFull => AdjointRule::Redirect,
            FactorOp::SvdCompact | FactorOp::SvdFull => AdjointRule::AdjointSeam,
        }
    }

    fn map_factor_error(error: tenet_tensors::OperationError) -> Error {
        error.into()
    }

    fn decode_label(
        provider: &R,
        sector: SectorId,
    ) -> Result<<R as TypedSectorAdmission>::Sector, Error> {
        // The blanket `TypedSectorAdmission` impl (the only one a
        // `CheckedFusionAlgebra + SectorCodec` rule can have) decodes through
        // `SectorCodec::decode_sector`.
        Ok(provider.try_decode_label(sector)?)
    }
}

impl<R> FusionMode<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
{
    fn adjoint_rule(_: FactorOp) -> AdjointRule {
        // D1: checked Generic refuses lazy adjoints until #1755.
        AdjointRule::Reject
    }

    fn map_factor_error(
        error: tenet_matrixalgebra::seam::CheckedGenericFactorPlanError<
            <R as CheckedGenericFusion>::Error,
        >,
    ) -> Self::FacadeError {
        error.into()
    }

    fn decode_label(provider: &R, sector: SectorId) -> Result<R::Sector, Self::FacadeError> {
        provider
            .try_decode_label(sector)
            .map_err(|error| GenericTensorError::Plan(CheckedGenericPlanError::Provider(error)))
    }
}
