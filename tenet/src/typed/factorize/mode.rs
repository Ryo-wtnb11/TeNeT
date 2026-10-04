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

    // Polar (D1, #1755; D5, #1752): the multiplicity-free dense stages
    // recouple through the runtime's context lane, which only the facade can
    // lease, and each mode reads a lazy adjoint's parent through its own
    // seam (multiplicity-free adjoints `w` inside it, checked at the facade).
    // These stay per-mode arms until those leaves give polar one kernel.

    /// Left polar of an owned dense `tensor`.
    fn left_polar_dense<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<LeftPolar<TensorMap<R, D>>, Self::FacadeError>;

    /// Right polar of an owned dense `tensor`.
    fn right_polar_dense<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<RightPolar<TensorMap<R, D>>, Self::FacadeError>;

    /// Left polar of the lazy adjoint `tensor`, read through its parent.
    fn left_polar_adjoint<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<LeftPolar<TensorMap<R, D>>, Self::FacadeError>;

    /// Right polar of the lazy adjoint `tensor`, read through its parent.
    fn right_polar_adjoint<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<RightPolar<TensorMap<R, D>>, Self::FacadeError>;
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
    LeftNull,
    RightNull,
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
            Self::LeftNull => "checked Generic left_null does not accept lazy adjoints",
            Self::RightNull => "checked Generic right_null does not accept lazy adjoints",
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
            FactorOp::LeftNull | FactorOp::RightNull => AdjointRule::Redirect,
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

    fn left_polar_dense<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<LeftPolar<TensorMap<R, D>>, Error> {
        // Dense lease before the context lease — the polar seam recouples
        // internally, so unlike QR/LQ/null it takes the context lane; the
        // lease order matches every existing site that takes both lanes.
        let mut dense = tensor.runtime.lease_dense();
        let mut lease = tensor.runtime.lease_context()?;
        let (bound_space, bound_payload) = tensor.bound_payload()?;
        let LeftPolar { w, p } = tenet_matrixalgebra::seam::left_polar_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>()?,
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(LeftPolar {
            w: wrap_factor_on(&tensor.runtime, w),
            p: wrap_factor_on(&tensor.runtime, p),
        })
    }

    fn right_polar_dense<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<RightPolar<TensorMap<R, D>>, Error> {
        // See `left_polar_dense` for the lease order rationale.
        let mut dense = tensor.runtime.lease_dense();
        let mut lease = tensor.runtime.lease_context()?;
        let (bound_space, bound_payload) = tensor.bound_payload()?;
        let RightPolar { p, wh } = tenet_matrixalgebra::seam::right_polar_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>()?,
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(RightPolar {
            p: wrap_factor_on(&tensor.runtime, p),
            wh: wrap_factor_on(&tensor.runtime, wh),
        })
    }

    fn left_polar_adjoint<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<LeftPolar<TensorMap<R, D>>, Error> {
        let TypedTensorRepr::Adjoint(view) = &tensor.repr else {
            return Err(internal_layout_error(
                "adjoint polar input must be a lazy adjoint",
            ));
        };
        let mut dense = tensor.runtime.lease_dense();
        let mut lease = tensor.runtime.lease_context()?;
        let LeftPolar { w, p } = tenet_matrixalgebra::seam::left_polar_adjoint_parent_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>()?,
            &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
        )?;
        Ok(LeftPolar {
            w: wrap_factor_on(&tensor.runtime, w),
            p: wrap_factor_on(&tensor.runtime, p),
        })
    }

    fn right_polar_adjoint<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<RightPolar<TensorMap<R, D>>, Error> {
        let TypedTensorRepr::Adjoint(view) = &tensor.repr else {
            return Err(internal_layout_error(
                "adjoint polar input must be a lazy adjoint",
            ));
        };
        let mut dense = tensor.runtime.lease_dense();
        let mut lease = tensor.runtime.lease_context()?;
        let RightPolar { p, wh } = tenet_matrixalgebra::seam::right_polar_adjoint_parent_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>()?,
            &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
        )?;
        Ok(RightPolar {
            p: wrap_factor_on(&tensor.runtime, p),
            wh: wrap_factor_on(&tensor.runtime, wh),
        })
    }
}

impl<R> FusionMode<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
{
    fn adjoint_rule(op: FactorOp) -> AdjointRule {
        match op {
            // The null space of `t^H` is the adjoint of the opposite null
            // space of `t`.
            FactorOp::LeftNull | FactorOp::RightNull => AdjointRule::Redirect,
            // D1: checked Generic refuses the other lazy adjoints until #1755.
            _ => AdjointRule::Reject,
        }
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

    fn left_polar_dense<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<LeftPolar<TensorMap<R, D>>, Self::FacadeError> {
        let body = tensor.owned_body().ok_or_else(|| {
            internal_layout_error("dense polar input must be owned after adjoint dispatch")
        })?;
        let LeftPolar { w, p } = tenet_matrixalgebra::seam::left_polar_checked_generic(
            RuntimeDense(&tensor.runtime),
            owned_factor_source(body)?,
        )?;
        Ok(LeftPolar {
            w: tensor.factor_output(w),
            p: tensor.factor_output(p),
        })
    }

    fn right_polar_dense<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<RightPolar<TensorMap<R, D>>, Self::FacadeError> {
        let body = tensor.owned_body().ok_or_else(|| {
            internal_layout_error("dense polar input must be owned after adjoint dispatch")
        })?;
        let RightPolar { p, wh } = tenet_matrixalgebra::seam::right_polar_checked_generic(
            RuntimeDense(&tensor.runtime),
            owned_factor_source(body)?,
        )?;
        Ok(RightPolar {
            p: tensor.factor_output(p),
            wh: tensor.factor_output(wh),
        })
    }

    fn left_polar_adjoint<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<LeftPolar<TensorMap<R, D>>, Self::FacadeError> {
        let TypedTensorRepr::Adjoint(view) = &tensor.repr else {
            return Err(internal_layout_error("adjoint polar input must be a lazy adjoint").into());
        };
        let mut dense = tensor.runtime.lease_dense();
        let input = BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
            .map_err(Error::from)?;
        let RightPolar { p, wh: w } =
            tenet_matrixalgebra::seam::left_polar_adjoint_parent_dyn_checked_generic(
                dense.dense(),
                &input,
            )?;
        let w = wrap_factor_on(&tensor.runtime, w)
            .adjoint()?
            .materialized_tensor_uncached()
            .map_err(GenericTensorError::from)?;
        Ok(LeftPolar {
            w,
            p: wrap_factor_on(&tensor.runtime, p),
        })
    }

    fn right_polar_adjoint<D: FactorizationScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<RightPolar<TensorMap<R, D>>, Self::FacadeError> {
        let TypedTensorRepr::Adjoint(view) = &tensor.repr else {
            return Err(internal_layout_error("adjoint polar input must be a lazy adjoint").into());
        };
        let mut dense = tensor.runtime.lease_dense();
        let input = BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
            .map_err(Error::from)?;
        let LeftPolar { w, p } =
            tenet_matrixalgebra::seam::right_polar_adjoint_parent_dyn_checked_generic(
                dense.dense(),
                &input,
            )?;
        let w = wrap_factor_on(&tensor.runtime, w)
            .adjoint()?
            .materialized_tensor_uncached()
            .map_err(GenericTensorError::from)?;
        Ok(RightPolar {
            p: wrap_factor_on(&tensor.runtime, p),
            wh: w,
        })
    }
}
