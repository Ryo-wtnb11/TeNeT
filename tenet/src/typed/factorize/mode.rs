use super::exp_solve::{checked_compact_divisor_solve, checked_generic_solve_into};
use super::*;
use tenet_matrixalgebra::seam::FactorSpaceAuthority;

/// The facade half of a fusion mode's factorization contract. Every
/// factorization has one body over it; the mode chooses only what differs by
/// design or is still owned by a unification leaf (#1862, table in its body):
/// the error type and provider-label decode (D4, permanent). The lazy-adjoint
/// rule is the mode-independent [`FactorOp::adjoint_rule`] (#1755). The
/// numerical stages and factor spaces come from the matrix-algebra
/// [`FactorMode`](tenet_matrixalgebra::seam::FactorMode).
#[doc(hidden)]
pub trait FusionMode<R>: TypedAdjointSpace<R> + tenet_matrixalgebra::seam::FactorMode<R>
where
    R: TypedSectorAdmission,
{
    /// A matrix-algebra error as this mode's facade error.
    fn map_factor_error(
        error: <Self as tenet_matrixalgebra::seam::FactorMode<R>>::Error,
    ) -> Self::FacadeError;

    /// The public label of a coupled sector the provider's own algebra
    /// produced.
    fn decode_label(provider: &R, sector: SectorId) -> Result<R::Sector, Self::FacadeError>;

    // Inverse, pseudo-inverse, solve and exponential share their bodies, and
    // their compact arms share one admission and value maps in both modes
    // (#1994, #1800). What stays per mode, until its leaf unifies it:
    // - D8 (#1995): where the categorical preflight runs, and its order;
    // - D5 (#1996): the dense lease and its timing.
    // The dense pseudo-inverse and exponential algorithms are the shared
    // per-sector kernels in both modes (#1752, #1799).

    /// Inverse of an owned `tensor` on the dense route (D5, D8).
    fn inv_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;

    /// Pseudo-inverse on the dense route; a lazy adjoint is read through
    /// its parent's SVD (D5).
    fn pinv_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
        rcond: f64,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;

    /// Exponential on the dense route, including the endomorphism check and
    /// the lazy-adjoint materialization (D5, D8).
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
    /// `None` to take the dense route for an `rhs` the mode does not scale
    /// in place.
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

/// A factorization that reads a possibly lazy-adjoint input.
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
    LeftPolar,
    RightPolar,
    EighFull,
    EigFull,
    Inv,
    Pinv,
    Exp,
}

impl FactorOp {
    /// How `op` reads a lazy adjoint, in every fusion mode (#1755): the
    /// identity relating the operation of `A^H` to one of `A` is fixed by the
    /// mathematics, so only the dense stages that execute it are per mode.
    pub(super) const fn adjoint_rule(self) -> AdjointRule {
        match self {
            // Singular values and coupled sectors are invariant under adjoint.
            Self::SvdVals => AdjointRule::Parent,
            // `A^H = V S U^H`: the mode's adjoint stage factors the parent and
            // gauges the final left factor `V`, as materialize-then-SVD
            // would; and `(A^H)^+ = U S^+ Vh` from the same parent SVD.
            Self::SvdCompact | Self::SvdFull | Self::Pinv => AdjointRule::AdjointSeam,
            // The null space of `A^H` is the adjoint of the opposite null
            // space of `A`, `left_polar(A^H)` is the adjoint-swapped
            // `right_polar(A)` (and vice versa), and `(A^H)^-1 = (A^-1)^H`.
            Self::LeftNull | Self::RightNull | Self::LeftPolar | Self::RightPolar | Self::Inv => {
                AdjointRule::Redirect
            }
            // Why not redirect QR through LQ of the parent: LQ is itself the
            // QR of the adjoint, so it would form this copy anyway and add
            // two factor adjoints. Why not redirect LQ to QR of the parent:
            // detaching `R^H` and `Q^H` copies at least `min(m, n) (m + n)
            // >= m n` elements per sector, never fewer than this input copy.
            Self::QrCompact | Self::QrFull | Self::LqCompact | Self::LqFull => {
                AdjointRule::Materialize
            }
            // Why not read the parent: an admitted near-Hermitian input
            // differs from its adjoint and the solver reads one triangle; the
            // values of `B^H` are `conj` of those of `B` only as a multiset,
            // so the published order would change; and the right
            // eigenvectors of `B^H` are the left ones of `B`.
            // `exp` likewise dispatches a near-Hermitian input to the
            // spectral route. The materialization runs inside `exp_dense`,
            // after that mode's preflight and leases (D5, D8).
            Self::EighVals | Self::EighFull | Self::EigVals | Self::EigFull | Self::Exp => {
                AdjointRule::Materialize
            }
        }
    }
}

/// What a factorization reads when its input is a lazy adjoint.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdjointRule {
    /// The dense parent, for a result that is invariant under adjoint.
    Parent,
    /// An operation-local materialized adjoint.
    Materialize,
    /// The partner operation on the parent, whose results are adjointed
    /// back and detached (the left null space of `t^H` from the right null
    /// space of `t`).
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
        // The shared per-sector pseudo-inverse into the derived swapped
        // layout; a lazy adjoint is read through its parent's SVD.
        let mut dense = tensor.runtime.lease_dense();
        let out = match &tensor.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::seam::pinv_adjoint_parent_dyn(
                dense.dense(),
                &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
                rcond,
            )
            .map_err(pinv_seam_error)?,
            TypedTensorRepr::Owned(_) => {
                let (bound_space, bound_payload) = tensor.bound_payload()?;
                tenet_matrixalgebra::seam::pinv_dyn(
                    dense.dense(),
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
        // The seam checks the endomorphism (D8) and runs the shared
        // per-sector exponential (#1799); the lease precedes the adjoint
        // materialization (D5).
        let mut dense = tensor.runtime.lease_dense();
        let local = matches!(&tensor.repr, TypedTensorRepr::Adjoint(_))
            .then(|| tensor.materialized_tensor_uncached())
            .transpose()?;
        let body = local
            .as_ref()
            .and_then(TensorMap::owned_body)
            .unwrap_or_else(|| tensor.owned_body().expect("owned representation"));
        let out = tenet_matrixalgebra::seam::exp_dyn(
            dense.dense(),
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
        // The shared per-sector pseudo-inverse into the admitted swapped
        // space; a lazy adjoint is read through its parent's SVD.
        let source = tensor.logical_space();
        let output = <Self as tenet_matrixalgebra::seam::FactorMode<R>>::authority(source)
            .output_space(FusionTreeHomSpace::new(
                source.space().homspace().domain().clone(),
                source.space().homspace().codomain().clone(),
            ))?;
        let mut dense = tensor.runtime.lease_dense();
        let factor = match &tensor.repr {
            TypedTensorRepr::Adjoint(view) => {
                tenet_matrixalgebra::seam::pinv_adjoint_parent_direct_into_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
                        .map_err(Error::from)?,
                    output,
                    rcond,
                )
            }
            TypedTensorRepr::Owned(body) => tenet_matrixalgebra::seam::pinv_direct_into_dyn(
                dense.dense(),
                &BoundDynamicTensorRef::try_new(
                    &body.space,
                    body.materialized_dense_data().as_ref(),
                )
                .map_err(Error::from)?,
                output,
                rcond,
            ),
        }
        .map_err(pinv_seam_error)?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }

    fn exp_dense<D: AdvancedLinalgScalar>(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        // The endomorphism check runs here, before the materialization (D8);
        // the shared per-sector exponential writes the input's own layout.
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
        let factor = tenet_matrixalgebra::seam::exp_direct_into_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())
                .map_err(Error::from)?,
            body.space.clone(),
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
