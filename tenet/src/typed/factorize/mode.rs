use super::exp_solve::{checked_compact_divisor_solve, checked_generic_solve_into};
use super::*;
use tenet_matrixalgebra::seam::FactorSpaceAuthority;

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

    // Inverse, pseudo-inverse, solve and exponential share their bodies and
    // compact value maps. What stays per mode, until its leaf unifies it:
    // - D7 (#1994): which compact diagonals take the compact arm; #1800 and
    //   #1986: the compact pseudo-inverse of a non-finite spectrum;
    // - D8 (#1995): where the categorical preflight runs, and its order;
    // - D5 (#1996, #1752): the dense and context leases and their timing;
    // - D6 (#1752, #1799): the dense pseudo-inverse and exponential
    //   algorithms.

    /// Whether a compact diagonal on `space` takes the compact arm of inv,
    /// pinv, solve and exp (D7, #1994); otherwise it is densified.
    fn compact_spectrum_admitted<D>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    ) -> bool;

    /// The compact pseudo-inverse of an admitted spectrum, or `None` to take
    /// the dense route (D7, #1800, #1986).
    #[allow(clippy::type_complexity)]
    fn pinv_spectrum<D: AdvancedLinalgScalar>(
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
        rcond: f64,
    ) -> Result<Option<Vec<tenet_matrixalgebra::SectorSpectrum<D>>>, Self::FacadeError>;

    /// Inverse of an owned `tensor` on the dense route (D5, D8).
    fn inv_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;

    /// Pseudo-inverse on the dense route; the multiplicity-free mode also
    /// reads a lazy adjoint here, through its parent (D5, D6).
    fn pinv_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
        rcond: f64,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;

    /// Exponential on the dense route, including the endomorphism check and
    /// the lazy-adjoint materialization (D5, D6, D8).
    fn exp_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;

    /// What a solve preflight admits for the arms after it.
    type SolveAdmission;

    /// The solve preflight after the runtime check (D8, #1995).
    fn solve_preflight<D: AdvancedLinalgScalar>(
        divisor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<Self::SolveAdmission, Self::FacadeError>;

    /// `divisor \ rhs` for an admitted, nonsingular compact divisor, or
    /// `None` to take the dense route (D7, #1994).
    fn solve_compact<D: AdvancedLinalgScalar>(
        divisor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
        admission: &Self::SolveAdmission,
    ) -> Result<Option<TensorMap<R, D>>, Self::FacadeError>;

    /// `divisor \ rhs` on the dense route (D5).
    fn solve_dense<D: AdvancedLinalgScalar>(
        divisor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        admission: Self::SolveAdmission,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
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
    EighFull,
    EigFull,
    Inv,
    Pinv,
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
            Self::EighFull => "checked Generic eigh_full does not accept lazy adjoints",
            Self::EigFull => "checked Generic eig_full does not accept lazy adjoints",
            Self::Inv => "checked Generic inv does not accept lazy adjoints",
            Self::Pinv => "checked Generic pinv does not accept lazy adjoints",
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
            FactorOp::EighFull | FactorOp::EigFull => AdjointRule::Materialize,
            // (A^H)^-1 = (A^-1)^H.
            FactorOp::Inv => AdjointRule::Redirect,
            FactorOp::Pinv => AdjointRule::AdjointSeam,
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

    fn compact_spectrum_admitted<D>(
        _space: &BoundDynamicFusionMapSpace<R>,
        _spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    ) -> bool {
        true
    }

    fn pinv_spectrum<D: AdvancedLinalgScalar>(
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
        rcond: f64,
    ) -> Result<Option<Vec<tenet_matrixalgebra::SectorSpectrum<D>>>, Error> {
        // A non-finite entry is rejected rather than folded: `f64::max`
        // would drop a NaN and `NaN > cutoff` would then zero it, a
        // silent finite answer (the dense arm's `pinv_cutoff` contract).
        let sigma_max = spectrum
            .iter()
            .flat_map(|entry| entry.values.iter())
            .try_fold(0.0f64, |largest, &value| {
                let magnitude = value.abs_value();
                magnitude.is_finite().then(|| largest.max(magnitude))
            })
            .ok_or_else(|| {
                Error::InvalidArgument("pinv singular values must be finite".to_string())
            })?;
        let cutoff = rcond * sigma_max;
        // Strict `>`, matching the dense fold: a
        // value exactly on the cutoff is cut. Changing it to `>=` is what
        // `pinv_cuts_a_singular_value_sitting_exactly_on_the_cutoff` kills.
        Ok(Some(map_spectrum(spectrum, |value| {
            Ok(if value.abs_value() > cutoff {
                value.recip_value()
            } else {
                D::from_real(0.0)
            })
        })?))
    }

    fn inv_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Error> {
        // The isomorphism check and the output space are inside the seam
        // (D8, #1995), under the dense lease (D5, #1996).
        let mut dense = tensor.runtime.lease_dense();
        let (bound_space, bound_payload) = tensor.bound_payload()?;
        let out = tenet_matrixalgebra::seam::inv_direct_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(wrap_factor_on(&tensor.runtime, out))
    }

    fn pinv_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
        rcond: f64,
    ) -> Result<TensorMap<R, D>, Error> {
        // SVD and a recoupling contraction through the context lane (D5, D6,
        // #1752); a lazy adjoint is read through its parent's SVD.
        let mut dense = tensor.runtime.lease_dense();
        let mut lease = tensor.runtime.lease_context()?;
        let out = match &tensor.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::seam::pinv_adjoint_parent_dyn(
                dense.dense(),
                lease.context().multiplicity_free_lane::<D>()?,
                &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
                rcond,
            )
            .map_err(pinv_seam_error)?,
            TypedTensorRepr::Owned(_) => {
                let (bound_space, bound_payload) = tensor.bound_payload()?;
                tenet_matrixalgebra::seam::pinv_dyn(
                    dense.dense(),
                    lease.context().multiplicity_free_lane::<D>()?,
                    &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
                    rcond,
                )
                .map_err(pinv_seam_error)?
            }
        };
        Ok(wrap_factor_on(&tensor.runtime, out))
    }

    fn exp_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Error> {
        // The seam checks the endomorphism (D8) and picks the Hermitian
        // spectral route or Padé (D6, #1799), recoupling through the context
        // lane (D5, #1752); both leases precede the adjoint materialization.
        let mut dense = tensor.runtime.lease_dense();
        let mut lease = tensor.runtime.lease_context()?;
        let local = matches!(&tensor.repr, TypedTensorRepr::Adjoint(_))
            .then(|| tensor.materialized_tensor_uncached())
            .transpose()?;
        let body = local
            .as_ref()
            .and_then(TensorMap::owned_body)
            .unwrap_or_else(|| tensor.owned_body().expect("owned representation"));
        let out = tenet_matrixalgebra::seam::exp_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>()?,
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())?,
        )?;
        Ok(wrap_factor_on(&tensor.runtime, out))
    }

    type SolveAdmission = ();

    fn solve_preflight<D: AdvancedLinalgScalar>(
        divisor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<(), Error> {
        if !divisor.same_rule(rhs) {
            return Err(Error::RuleMismatch);
        }
        if divisor.logical_space().space().homspace().codomain()
            != rhs.logical_space().space().homspace().codomain()
        {
            return Err(Error::InvalidArgument(
                "solve requires equal divisor and right-hand-side codomains".to_string(),
            ));
        }
        if !divisor.logical_space().codomain_isomorphic_to_domain()? {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "solve requires an isomorphic divisor codomain and domain",
                },
            ));
        }
        // Only a compact divisor reads a lazy `rhs` in place (through
        // `compose`); the dense route materializes both operands.
        if divisor.spectrum().is_none() {
            rhs.refuse_borrowed_view("solve")?;
        }
        Ok(())
    }

    fn solve_compact<D: AdvancedLinalgScalar>(
        divisor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
        _admission: &(),
    ) -> Result<Option<TensorMap<R, D>>, Error> {
        // D^-1 composed with `rhs`, rebound to the divisor's provider
        // allocation.
        let solved = divisor
            .with_spectrum(inv_spectrum(spectrum)?)
            .compose(rhs)?;
        let TypedTensorRepr::Owned(body) = solved.repr else {
            return Err(internal_layout_error(
                "compact solve must produce an owned result",
            ));
        };
        let space = divisor
            .logical_space()
            .rebind_validated(&body.space.validated_layout())?;
        Ok(Some(TensorMap {
            runtime: divisor.runtime.clone(),
            repr: owned_repr(TypedTensorBody::with_shared_payload(
                space,
                Arc::clone(&body.data),
            )),
        }))
    }

    fn solve_dense<D: AdvancedLinalgScalar>(
        divisor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        _admission: (),
    ) -> Result<TensorMap<R, D>, Error> {
        let lhs_local = matches!(&divisor.repr, TypedTensorRepr::Adjoint(_))
            .then(|| divisor.materialized_tensor_uncached())
            .transpose()?;
        let rhs_local = matches!(&rhs.repr, TypedTensorRepr::Adjoint(_))
            .then(|| rhs.materialized_tensor_uncached())
            .transpose()?;
        let lhs = lhs_local.as_ref().unwrap_or(divisor);
        let rhs = rhs_local.as_ref().unwrap_or(rhs);
        let (lhs_space, lhs_payload) = lhs.bound_payload()?;
        let (rhs_space, rhs_payload) = rhs.bound_payload()?;
        let mut dense = divisor.runtime.lease_dense();
        let out = tenet_matrixalgebra::seam::solve_left_direct_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(lhs_space, &lhs_payload)?,
            &BoundDynamicTensorRef::try_new(rhs_space, &rhs_payload)?,
        )?;
        Ok(wrap_factor_on(&divisor.runtime, out))
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
            FactorOp::EighFull | FactorOp::EigFull => AdjointRule::Materialize,
            // (A^H)^-1 = (A^-1)^H, and likewise for the pseudo-inverse.
            FactorOp::Inv | FactorOp::Pinv => AdjointRule::Redirect,
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

    fn compact_spectrum_admitted<D>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    ) -> bool {
        // A proven bond is its own swapped space (TensorKit
        // `inv`/`pinv(::DiagonalTensorMap)` keep `d.domain`, dual included),
        // so the compact arm needs neither the isomorphism preflight nor a
        // separately admitted output root, and it is an endomorphism.
        checked_compact_spectrum_layout(space, space, spectrum)
    }

    fn pinv_spectrum<D: AdvancedLinalgScalar>(
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
        rcond: f64,
    ) -> Result<Option<Vec<tenet_matrixalgebra::SectorSpectrum<D>>>, Self::FacadeError> {
        let sigma_max =
            spectrum
                .iter()
                .flat_map(|entry| &entry.values)
                .try_fold(0.0_f64, |largest, &value| {
                    let magnitude = value.abs_value();
                    let rounded = D::from_real(magnitude).abs_value();
                    (magnitude.is_finite() && rounded.is_finite()).then(|| largest.max(rounded))
                });
        let Some(sigma_max) = sigma_max else {
            return Ok(None);
        };
        let cutoff = rcond * sigma_max;
        // The dense SVD can flush a retained subnormal to zero.
        if !spectrum
            .iter()
            .flat_map(|entry| &entry.values)
            .all(|&value| {
                D::from_real(value.abs_value()).abs_value() <= cutoff
                    || (value.abs_value() >= D::safe_minimum()
                        && value.recip_value().abs_value().is_finite())
            })
        {
            return Ok(None);
        }
        Ok(Some(map_spectrum(spectrum, |value| {
            Ok(if D::from_real(value.abs_value()).abs_value() > cutoff {
                value.recip_value()
            } else {
                D::from_real(0.0)
            })
        })?))
    }

    fn inv_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        // The isomorphism check and the output space are built here, before
        // the dense lease (D8, #1995; D5, #1996).
        let source = tensor.logical_space();
        let authority = <Self as tenet_matrixalgebra::seam::FactorMode<R>>::authority(source);
        if !authority.isomorphic(source)? {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "inv requires isomorphic codomain and domain",
                },
            )
            .into());
        }
        let output = authority.output_space(FusionTreeHomSpace::new(
            source.space().homspace().domain().clone(),
            source.space().homspace().codomain().clone(),
        ))?;
        let body = tensor
            .owned_body()
            .expect("checked Generic inverse input is owned after lazy dispatch");
        let mut dense = tensor.runtime.lease_dense();
        let factor = tenet_matrixalgebra::seam::inv_direct_into_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())
                .map_err(Error::from)?,
            output,
        )
        .map_err(Error::from)?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }

    fn pinv_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
        rcond: f64,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        // A direct per-sector pseudo-inverse, with no context lane (D5, D6,
        // #1752).
        let body = tensor
            .owned_body()
            .expect("checked Generic pinv input is owned after lazy dispatch");
        let source = tensor.logical_space();
        let output = <Self as tenet_matrixalgebra::seam::FactorMode<R>>::authority(source)
            .output_space(FusionTreeHomSpace::new(
                source.space().homspace().domain().clone(),
                source.space().homspace().codomain().clone(),
            ))?;
        let mut dense = tensor.runtime.lease_dense();
        let factor = tenet_matrixalgebra::seam::pinv_direct_into_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())
                .map_err(Error::from)?,
            output,
            rcond,
        )
        .map_err(pinv_seam_error)?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }

    fn exp_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        // The endomorphism check runs here, before the materialization (D8),
        // and every input takes Padé [13/13] (D6, #1799).
        if tensor.logical_space().space().homspace().codomain()
            != tensor.logical_space().space().homspace().domain()
        {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "exp requires an endomorphism (codomain == domain)",
                },
            )
            .into());
        }
        let local = matches!(&tensor.repr, TypedTensorRepr::Adjoint(_))
            .then(|| tensor.materialized_tensor_uncached())
            .transpose()
            .map_err(GenericTensorError::from)?;
        let body = local
            .as_ref()
            .and_then(TensorMap::owned_body)
            .unwrap_or_else(|| tensor.owned_body().expect("owned representation"));
        let mut dense = tensor.runtime.lease_dense();
        let factor = tenet_matrixalgebra::seam::exp_pade13_direct_into_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())
                .map_err(Error::from)?,
        )
        .map_err(Error::from)?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }

    type SolveAdmission = (
        tenet_operations::host_pool::HostPoolGuard,
        BoundDynamicFusionMapSpace<R>,
    );

    fn solve_preflight<D: AdvancedLinalgScalar>(
        divisor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<Self::SolveAdmission, Self::FacadeError> {
        // The dense route materializes every lazy `rhs`, and the compact one
        // reads it only after this refusal.
        rhs.refuse_borrowed_view("solve")?;
        let host_pool = divisor.runtime.enter_host_pool();
        if divisor.logical_space().space().admission().rule_identity()
            != rhs.logical_space().space().admission().rule_identity()
        {
            return Err(Error::RuleMismatch.into());
        }
        if divisor.logical_space().space().homspace().codomain()
            != rhs.logical_space().space().homspace().codomain()
        {
            return Err(Error::InvalidArgument(
                "solve requires equal divisor and right-hand-side codomains".to_string(),
            )
            .into());
        }
        let lhs_space = divisor.logical_space();
        let authority = <Self as tenet_matrixalgebra::seam::FactorMode<R>>::authority(lhs_space);
        if !authority.isomorphic(lhs_space)? {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "solve requires an isomorphic divisor codomain and domain",
                },
            )
            .into());
        }
        let output = authority.output_space(FusionTreeHomSpace::new(
            lhs_space.space().homspace().domain().clone(),
            rhs.logical_space().space().homspace().domain().clone(),
        ))?;
        Ok((host_pool, output))
    }

    fn solve_compact<D: AdvancedLinalgScalar>(
        divisor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
        (_, output): &Self::SolveAdmission,
    ) -> Result<Option<TensorMap<R, D>>, Self::FacadeError> {
        Ok(checked_compact_divisor_solve(
            divisor, rhs, spectrum, output,
        )?)
    }

    fn solve_dense<D: AdvancedLinalgScalar>(
        divisor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        (_host_pool, output): Self::SolveAdmission,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        let factor =
            checked_generic_solve_into(divisor, rhs, divisor.logical_space().clone(), output)?;
        Ok(wrap_factor_on(&divisor.runtime, factor))
    }
}
