use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R>,
    D: AdvancedLinalgScalar,
{
    /// The one body of the inverse. A lazy adjoint is redirected,
    /// `(A^H)^-1 = (A^-1)^H`; an admitted compact diagonal inverts its
    /// entries (TensorKit `inv(::DiagonalTensorMap)`).
    pub(super) fn factor_inv(&self) -> Result<Self, TypedFacadeError<R>> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            const { assert!(redirects(FactorOp::Inv)) };
            return self.adjoint_of_parent(Self::factor_inv);
        }
        if let Some(spectrum) = self.spectrum() {
            if R::Mode::compact_spectrum_admitted(self.logical_space(), spectrum) {
                return Ok(self.with_spectrum(inv_spectrum(spectrum)?));
            }
        }
        R::Mode::inv_dense(self)
    }

    /// The one body of the pseudo-inverse: a lazy adjoint `A^H` is read
    /// through its parent's SVD, `(A^H)^+ = U S^+ Vh`; an admitted compact
    /// diagonal inverts its retained entries.
    pub(super) fn factor_pinv(&self, rcond: f64) -> Result<Self, TypedFacadeError<R>> {
        // Ahead of every arm, so all of them answer alike: the dense seams
        // repeat this check for their own callers, but the compact arm never
        // reaches a seam.
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
            if R::Mode::compact_spectrum_admitted(self.logical_space(), spectrum) {
                if let Some(inverted) = R::Mode::pinv_spectrum(spectrum, rcond)? {
                    return Ok(self.with_spectrum(inverted));
                }
            }
        }
        R::Mode::pinv_dense(self, rcond)
    }
}
