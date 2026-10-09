use super::*;
use tenet_matrixalgebra::seam::FactorSpaceAuthority;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R>,
    D: AdvancedLinalgScalar,
{
    /// The one body of the inverse. A lazy adjoint is redirected,
    /// `(A^H)^-1 = (A^-1)^H`; a compact diagonal inverts its entries
    /// (TensorKit `inv(::DiagonalTensorMap)`).
    pub(super) fn factor_inv(&self) -> Result<Self, TypedFacadeError<R>> {
        self.require_isomorphic("inv requires isomorphic codomain and domain")?;
        self.factor_inv_admitted()
    }

    /// The representation stage of [`Self::factor_inv`]. The redirect
    /// re-enters here rather than at the preflight: `A^H` and `A` are
    /// isomorphic together, so the parent needs no second check.
    fn factor_inv_admitted(&self) -> Result<Self, TypedFacadeError<R>> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            const { assert!(redirects(FactorOp::Inv)) };
            return self.adjoint_of_parent(Self::factor_inv_admitted);
        }
        if let Some(spectrum) = self.spectrum() {
            self.admit_compact(spectrum)?;
            return Ok(self.with_spectrum(inv_spectrum(spectrum)?));
        }
        R::Mode::inv_dense(self, self.swapped_output_space()?)
    }

    /// The one body of the pseudo-inverse: a lazy adjoint `A^H` is read
    /// through its parent's SVD, `(A^H)^+ = U S^+ Vh`; a compact diagonal
    /// inverts its retained entries.
    pub(super) fn factor_pinv(&self, rcond: f64) -> Result<Self, TypedFacadeError<R>> {
        // Ahead of every arm, so all of them answer alike: the dense seams
        // repeat this check for their own expert callers, but the compact arm
        // never reaches a seam. TensorKit's `pinv` has no space check.
        if !rcond.is_finite() || rcond < 0.0 {
            return Err(Error::InvalidArgument(
                "pinv rcond must be finite and non-negative".to_string(),
            )
            .into());
        }
        // A lazy adjoint is never compact; the dense stage reads its parent.
        const {
            assert!(matches!(
                FactorOp::Pinv.adjoint_rule(),
                AdjointRule::AdjointSeam
            ))
        };
        if let Some(spectrum) = self.spectrum() {
            self.admit_compact(spectrum)?;
            let inverted =
                tenet_matrixalgebra::seam::pinv_diagonal_spectrum(spectrum, rcond, |value: D| {
                    value.recip_value()
                })
                .map_err(pinv_seam_error)?;
            return Ok(self.with_spectrum(inverted));
        }
        R::Mode::pinv_dense(self, self.swapped_output_space()?, rcond)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R>,
    D: TensorScalar,
{
    /// TensorKit `cod ≅ dom` (`inv`, `\`), refused with `message`. Equal
    /// sides are isomorphic without a provider query.
    pub(super) fn require_isomorphic(
        &self,
        message: &'static str,
    ) -> Result<(), TypedFacadeError<R>> {
        let space = self.logical_space();
        let homspace = space.space().homspace();
        if homspace.codomain() == homspace.domain()
            || <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::authority(space)
                .isomorphic(space)
                .map_err(R::Mode::map_factor_error)?
        {
            return Ok(());
        }
        Err(Error::from(tenet_tensors::OperationError::SpaceMismatch { message }).into())
    }

    /// The mode's output space of `homspace` on this tensor's provider: the
    /// space TensorKit's `similar(t, T, homspace)` allocates.
    pub(super) fn factor_output_space(
        &self,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, TypedFacadeError<R>> {
        <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::authority(self.logical_space())
            .output_space(homspace)
            .map_err(R::Mode::map_root_error)
    }

    /// `domain <- codomain`: the output of the inverse and pseudo-inverse.
    fn swapped_output_space(&self) -> Result<BoundDynamicFusionMapSpace<R>, TypedFacadeError<R>> {
        let homspace = self.logical_space().space().homspace();
        self.factor_output_space(FusionTreeHomSpace::new(
            homspace.domain().clone(),
            homspace.codomain().clone(),
        ))
    }

    /// The one admission of a compact diagonal's elementwise matrix
    /// functions (inv, pinv, exp, solve) in every fusion mode (#1994): the
    /// matrix-algebra bond admission without a value check. An expert bond
    /// layout is admitted like the canonical one, and the result keeps it.
    pub(super) fn admit_compact(
        &self,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    ) -> Result<(), TypedFacadeError<R>> {
        tenet_matrixalgebra::seam::admit_compact_diagonal::<R::Mode, _, _>(
            self.logical_space(),
            spectrum,
        )
        .map_err(R::Mode::map_factor_error)
    }
}
