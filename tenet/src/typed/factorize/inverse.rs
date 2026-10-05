use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R>,
    D: AdvancedLinalgScalar,
{
    /// The one body of the inverse. Both modes redirect a lazy adjoint,
    /// `(A^H)^-1 = (A^-1)^H`, and detach the result; an admitted compact
    /// diagonal inverts its entries (TensorKit `inv(::DiagonalTensorMap)`).
    pub(super) fn factor_inv(&self) -> Result<Self, TypedFacadeError<R>> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            match R::Mode::adjoint_rule(FactorOp::Inv) {
                AdjointRule::Redirect => {
                    return Ok(self
                        .adjoint()?
                        .factor_inv()?
                        .adjoint()?
                        .materialized_tensor_uncached()?);
                }
                // The mode's dense stage reads the parent.
                AdjointRule::AdjointSeam => {}
                AdjointRule::Reject | AdjointRule::Parent | AdjointRule::Materialize => {
                    return Err(Error::InvalidArgument(
                        FactorOp::Inv.lazy_adjoint_refusal().to_string(),
                    )
                    .into());
                }
            }
        }
        if let Some(spectrum) = self.spectrum() {
            if R::Mode::compact_spectrum_admitted(self.logical_space(), spectrum) {
                return Ok(self.with_spectrum(inv_spectrum(spectrum)?));
            }
        }
        R::Mode::inv_dense(self)
    }

    /// The one body of the pseudo-inverse; see [`Self::factor_inv`].
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
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            match R::Mode::adjoint_rule(FactorOp::Pinv) {
                AdjointRule::Redirect => {
                    return Ok(self
                        .adjoint()?
                        .factor_pinv(rcond)?
                        .adjoint()?
                        .materialized_tensor_uncached()?);
                }
                // The mode's dense stage reads the parent.
                AdjointRule::AdjointSeam => {}
                AdjointRule::Reject | AdjointRule::Parent | AdjointRule::Materialize => {
                    return Err(Error::InvalidArgument(
                        FactorOp::Pinv.lazy_adjoint_refusal().to_string(),
                    )
                    .into());
                }
            }
        }
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
