use super::*;
use tenet_matrixalgebra::seam::FactorSpaceAuthority;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R>,
    D: AdvancedLinalgScalar,
{
    /// The one body of the exponential: a compact diagonal exponentiates its
    /// entries, everything else takes the mode's dense route.
    pub(super) fn factor_exp(&self) -> Result<Self, TypedFacadeError<R>> {
        // TensorKit `exp!`: `domain == codomain` before anything else.
        let homspace = self.logical_space().space().homspace();
        if homspace.codomain() != homspace.domain() {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "exp requires an endomorphism (codomain == domain)",
                },
            )
            .into());
        }
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
            self.admit_compact(spectrum)?;
            return Ok(self.with_spectrum(exp_spectrum(spectrum)?));
        }
        let space = self.logical_space();
        let output = <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::authority(space)
            .same_homspace_output(space)
            .map_err(R::Mode::map_root_error)?;
        R::Mode::exp_dense(self, output)
    }

    /// The one body of `self \ rhs`, solved sector by sector without forming
    /// an inverse.
    ///
    /// One preflight in every mode (#1995): runtime, rule, codomain
    /// equality, the divisor's isomorphism, then the borrowed-view
    /// admission, before any representation work. The result is
    /// `domain(self) <- domain(rhs)`. A nonsingular compact divisor scales
    /// `rhs` by its reciprocal spectrum; every other divisor is solved
    /// densely into the final output.
    pub(super) fn factor_solve(&self, rhs: &Self) -> Result<Self, TypedFacadeError<R>> {
        self.require_solve_operands(rhs)?;
        if self.logical_space().space().homspace().codomain()
            != rhs.logical_space().space().homspace().codomain()
        {
            return Err(Error::InvalidArgument(
                "solve requires equal divisor and right-hand-side codomains".to_string(),
            )
            .into());
        }
        self.require_isomorphic("solve requires an isomorphic divisor codomain and domain")?;
        let Some(spectrum) = self.spectrum() else {
            // Only a compact divisor reads a lazy `rhs` in place; the dense
            // route materializes both operands.
            rhs.refuse_borrowed_view("solve")?;
            let output = self.factor_output_space(self.solve_homspace(rhs))?;
            return R::Mode::solve_dense(self, rhs, output);
        };
        self.admit_compact(spectrum)?;
        reject_singular_compact_divisor(spectrum)?;
        self.compact_divisor_solve(rhs, spectrum)
    }

    /// The operand checks of a solve, ahead of its operand shapes: runtime
    /// first (a trust boundary, not an algebra error), then the rule.
    pub(super) fn require_solve_operands(&self, rhs: &Self) -> Result<(), TypedFacadeError<R>> {
        if !self.runtime.same_runtime(&rhs.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        if self.logical_space().space().admission().rule_identity()
            != rhs.logical_space().space().admission().rule_identity()
        {
            return Err(Error::RuleMismatch.into());
        }
        Ok(())
    }

    /// `domain(self) <- domain(rhs)`, the space of `self \ rhs`.
    fn solve_homspace(&self, rhs: &Self) -> FusionTreeHomSpace {
        FusionTreeHomSpace::new(
            self.logical_space().space().homspace().domain().clone(),
            rhs.logical_space().space().homspace().domain().clone(),
        )
    }

    /// TensorKit `D \ t` on an admitted, nonsingular `DiagonalTensorMap`
    /// divisor: `D \ D'` divides the spectra and stays compact on `D`'s own
    /// space (`d1.domain`); any other `t` — dense, a mismatched compact
    /// payload or a (borrowed) lazy adjoint read in place — lands in the
    /// mode's output space with its leading (bond) axis scaled by the
    /// reciprocal spectrum, `O(Σ_c k_c m_c)` with no LU and no `Σ_c k_c²`
    /// divisor buffer.
    fn compact_divisor_solve(
        &self,
        rhs: &Self,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    ) -> Result<Self, TypedFacadeError<R>> {
        let inverse = inv_spectrum(spectrum)?;
        if let Some(values) = rhs.spectrum() {
            if values.len() == inverse.len()
                && values.iter().zip(&inverse).all(|(value, inverse)| {
                    value.sector == inverse.sector && value.values.len() == inverse.values.len()
                })
            {
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
                return Ok(self.with_spectrum(quotient));
            }
        }
        let output = self.factor_output_space(self.solve_homspace(rhs))?;
        let _host_pool = self.runtime.enter_host_pool();
        // The scaled payload is the result's own buffer: a dense `rhs` laid
        // out as the output is copied, anything else is read into it.
        let mut data = match &rhs.repr {
            TypedTensorRepr::Owned(body)
                if matches!(body.data.as_ref(), TypedData::Dense(_))
                    && body.space.space() == output.space() =>
            {
                body.materialized_dense_data().into_owned()
            }
            _ => {
                let (operand, source) = rhs.fusion_operand_and_data();
                tenet_tensors::oriented_fusion_add_owned(
                    output.space().structure(),
                    operand,
                    &source,
                    operand,
                    &source,
                    D::from_real(1.0),
                    D::from_real(0.0),
                )
                .map_err(Error::from)?
            }
        };
        tenet_matrixalgebra::seam::scale_axis_by_spectrum_mapped(
            output.space(),
            &mut data,
            Some(0),
            &inverse,
            |value| value,
        )
        .map_err(Error::from)?;
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(output, data)),
        })
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
