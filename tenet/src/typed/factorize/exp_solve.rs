use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R>,
    D: AdvancedLinalgScalar,
{
    /// The one body of the exponential: an admitted compact diagonal
    /// exponentiates its entries, everything else takes the mode's dense
    /// route.
    pub(super) fn factor_exp(&self) -> Result<Self, TypedFacadeError<R>> {
        // A lazy adjoint is never compact; `exp_dense` materializes it.
        const {
            assert!(matches!(
                FactorOp::Exp.adjoint_rule(),
                AdjointRule::Materialize
            ))
        };
        if let Some(spectrum) = self.spectrum() {
            // Why the spectrum is exponentiated unconditionally while the
            // dense arm asks about hermiticity: the dense question picks an
            // algorithm (spectral or Padé), not a domain, and a diagonal is
            // already in its eigenbasis so neither answer would change what
            // happens here. TensorKit splits the same way (#576, #578).
            if R::Mode::compact_spectrum_admitted(self.logical_space(), spectrum) {
                return Ok(self.with_spectrum(exp_spectrum(spectrum)?));
            }
        }
        R::Mode::exp_dense(self)
    }

    /// The one body of `self \ rhs`, solved sector by sector without forming
    /// an inverse.
    ///
    /// The two codomains must be exactly equal and `self` must have isomorphic
    /// codomain and domain. The result is `domain(self) <- domain(rhs)`. An
    /// admitted, nonsingular compact divisor scales `rhs` by its reciprocal
    /// spectrum; every other divisor is solved densely into the final output.
    pub(super) fn factor_solve(&self, rhs: &Self) -> Result<Self, TypedFacadeError<R>> {
        if !self.runtime.same_runtime(&rhs.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        let admission = R::Mode::solve_preflight(self, rhs)?;
        if let Some(spectrum) = self.spectrum() {
            if R::Mode::compact_spectrum_admitted(self.logical_space(), spectrum) {
                reject_singular_compact_divisor(spectrum)?;
                if let Some(solved) = R::Mode::solve_compact(self, rhs, spectrum, &admission)? {
                    return Ok(solved);
                }
            }
        }
        R::Mode::solve_dense(self, rhs, admission)
    }
}

/// TensorKit `D \ t` on a `DiagonalTensorMap` divisor: `D \ D'` divides the
/// spectra and stays compact, and `D \ t` scales each block's leading
/// (bond) axis by the reciprocal spectrum, `O(Σ_c k_c m_c)` with no LU and no
/// `Σ_c k_c²` divisor buffer. The divisor is admitted and nonsingular;
/// `None` leaves every other `rhs` layout to the dense route, which owns its
/// validation order.
pub(super) fn checked_compact_divisor_solve<R, D>(
    tensor: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    output: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<TensorMap<R, D>>, Error>
where
    D: TensorScalar,
{
    let inverse = inv_spectrum(spectrum)?;
    let local = matches!(&rhs.repr, TypedTensorRepr::Adjoint(_))
        .then(|| rhs.materialized_tensor_uncached())
        .transpose()?;
    let rhs_body = local
        .as_ref()
        .and_then(TensorMap::owned_body)
        .unwrap_or_else(|| rhs.owned_body().expect("solve rhs is owned after refusal"));
    // The destination `domain(D) <- domain(rhs)` is `rhs`'s own space,
    // because `D` is a bond (`domain == codomain == codomain(rhs)`); the
    // equality check proves it rather than trusting the provider `Arc`s.
    if output.space() != rhs_body.space.space() {
        return Ok(None);
    }
    match rhs_body.data.as_ref() {
        TypedData::Diagonal(values) => {
            if values.len() != inverse.len()
                || values.iter().zip(&inverse).any(|(value, inverse)| {
                    value.sector != inverse.sector || value.values.len() != inverse.values.len()
                })
            {
                return Ok(None);
            }
            let quotient = values
                .iter()
                .zip(&inverse)
                .map(|(value, inverse)| tenet_matrixalgebra::SectorSpectrum {
                    sector: value.sector,
                    values: value
                        .values
                        .iter()
                        .zip(&inverse.values)
                        .map(|(&value, &inverse)| inverse * value)
                        .collect(),
                })
                .collect();
            Ok(Some(tensor.with_spectrum_on(output.clone(), quotient)))
        }
        TypedData::Dense(values) => {
            let mut data = values.clone();
            tenet_matrixalgebra::seam::scale_axis_by_spectrum_mapped(
                output.space(),
                &mut data,
                Some(0),
                &inverse,
                |value| value,
            )?;
            Ok(Some(TensorMap {
                runtime: tensor.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(output.clone(), data)),
            }))
        }
    }
}

pub(super) fn checked_generic_solve_into<R, D>(
    tensor: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    divisor_authority: BoundDynamicFusionMapSpace<R>,
    output: BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynFactor<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: TensorScalar,
{
    let lhs = tensor
        .materialized_tensor_uncached()
        .map_err(GenericTensorError::from)?;
    let rhs = rhs
        .materialized_tensor_uncached()
        .map_err(GenericTensorError::from)?;
    let lhs_body = lhs.owned_body().expect("uncached solve lhs is owned");
    let rhs_body = rhs.owned_body().expect("uncached solve rhs is owned");
    let mut dense = tensor.runtime.lease_dense();
    tenet_matrixalgebra::seam::solve_left_direct_into_dyn(
        dense.dense(),
        &BoundDynamicTensorRef::try_new(
            &divisor_authority,
            lhs_body.materialized_dense_data().as_ref(),
        )
        .map_err(Error::from)?,
        &BoundDynamicTensorRef::try_new(
            &rhs_body.space,
            rhs_body.materialized_dense_data().as_ref(),
        )
        .map_err(Error::from)?,
        output,
    )
    .map_err(Error::from)
    .map_err(GenericTensorError::from)
}
