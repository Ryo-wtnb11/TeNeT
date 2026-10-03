use super::*;

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// Multiplicity-free implementation of the public mode-dispatched exponential.
    pub(super) fn exp_multiplicity_free(&self) -> Result<Self, Error>
    where
        D: AdvancedLinalgScalar,
    {
        if let Some(spectrum) = self.spectrum() {
            // Why the spectrum is exponentiated unconditionally while the dense
            // arm asks about hermiticity: the dense question picks an algorithm
            // (spectral or Padé), not a domain, and a diagonal is already in its
            // eigenbasis so neither answer would change what happens here.
            // TensorKit splits the same way (#576, #578).
            return Ok(self.with_spectrum(exp_spectrum(spectrum)?));
        }
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let local = matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            .then(|| self.materialized_tensor_uncached())
            .transpose()?;
        let body = local
            .as_ref()
            .and_then(Self::owned_body)
            .unwrap_or_else(|| self.owned_body().expect("owned representation"));
        let out = tenet_matrixalgebra::seam::exp_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>()?,
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())?,
        )?;
        Ok(self.wrap_bound_factor(out))
    }

    /// Solves `self * x = rhs` sector by sector without forming an inverse.
    ///
    /// The two codomains must be exactly equal and `self` must have isomorphic
    /// codomain and domain. The result is `domain(self) <- domain(rhs)` and
    /// keeps `self`'s exact provider allocation. Dense blocks are written
    /// directly into the final output; compact diagonal divisors reuse the
    /// elementwise reciprocal and bond-scaling path.
    pub(super) fn solve_multiplicity_free(&self, rhs: &Self) -> Result<Self, Error>
    where
        D: AdvancedLinalgScalar,
    {
        if !self.runtime.same_runtime(&rhs.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if !self.same_rule(rhs) {
            return Err(Error::RuleMismatch);
        }
        if self.logical_space().space().homspace().codomain()
            != rhs.logical_space().space().homspace().codomain()
        {
            return Err(Error::InvalidArgument(
                "solve requires equal divisor and right-hand-side codomains".to_string(),
            ));
        }
        if !self.logical_space().codomain_isomorphic_to_domain()? {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "solve requires an isomorphic divisor codomain and domain",
                },
            ));
        }
        // Only a compact divisor reads a lazy `rhs` in place (through
        // `compose`); the dense route materializes both operands.
        if self.spectrum().is_none() {
            rhs.refuse_borrowed_view("solve")?;
        }

        if let Some(spectrum) = self.spectrum() {
            reject_singular_compact_divisor(spectrum)?;
            let solved = self.inv_multiplicity_free()?.compose(rhs)?;
            let TypedTensorRepr::Owned(body) = solved.repr else {
                return Err(internal_layout_error(
                    "compact solve must produce an owned result",
                ));
            };
            let space = self
                .logical_space()
                .rebind_validated(&body.space.validated_layout())?;
            return Ok(Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::with_shared_payload(
                    space,
                    Arc::clone(&body.data),
                )),
            });
        }

        let lhs_local = matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            .then(|| self.materialized_tensor_uncached())
            .transpose()?;
        let rhs_local = matches!(&rhs.repr, TypedTensorRepr::Adjoint(_))
            .then(|| rhs.materialized_tensor_uncached())
            .transpose()?;
        let lhs = lhs_local.as_ref().unwrap_or(self);
        let rhs = rhs_local.as_ref().unwrap_or(rhs);
        let (lhs_space, lhs_payload) = lhs.bound_payload()?;
        let (rhs_space, rhs_payload) = rhs.bound_payload()?;
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::seam::solve_left_direct_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(lhs_space, &lhs_payload)?,
            &BoundDynamicTensorRef::try_new(rhs_space, &rhs_payload)?,
        )?;
        Ok(self.wrap_bound_factor(out))
    }
}

impl<R, D> TypedTensorSolveDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
{
    fn solve(tensor: &TensorMap<R, D>, rhs: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.solve_multiplicity_free(rhs)
    }
}

impl<R, D> TypedTensorExpDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
{
    fn exp(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.exp_multiplicity_free()
    }
}

impl<R, D> TypedTensorSolveDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    fn solve(
        tensor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if !tensor.runtime.same_runtime(&rhs.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        // `checked_generic_solve_into` materializes every lazy `rhs`.
        rhs.refuse_borrowed_view("solve")?;
        let _host_pool = tensor.runtime.enter_host_pool();
        if tensor.logical_space().space().admission().rule_identity()
            != rhs.logical_space().space().admission().rule_identity()
        {
            return Err(Error::RuleMismatch.into());
        }
        if tensor.logical_space().space().homspace().codomain()
            != rhs.logical_space().space().homspace().codomain()
        {
            return Err(Error::InvalidArgument(
                "solve requires equal divisor and right-hand-side codomains".to_string(),
            )
            .into());
        }

        let lhs_space = tensor.logical_space();
        let codomain = tenet_matrixalgebra::seam::coupled_sector_block_dimensions_generic_checked(
            lhs_space.space().homspace().codomain(),
            lhs_space.provider(),
        )?;
        let domain = tenet_matrixalgebra::seam::coupled_sector_block_dimensions_generic_checked(
            lhs_space.space().homspace().domain(),
            lhs_space.provider(),
        )?;
        if codomain != domain {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "solve requires an isomorphic divisor codomain and domain",
                },
            )
            .into());
        }

        let output = <R::Mode as TypedTensorRootDispatch<R>>::build_root(
            Arc::clone(lhs_space.provider_arc()),
            FusionTreeHomSpace::new(
                lhs_space.space().homspace().domain().clone(),
                rhs.logical_space().space().homspace().domain().clone(),
            ),
        )?;
        if let Some(solved) = checked_compact_divisor_solve(tensor, rhs, &output)? {
            return Ok(solved);
        }
        let factor =
            checked_generic_solve_into(tensor, rhs, tensor.logical_space().clone(), output)?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }
}

/// TensorKit `D \ t` on a `DiagonalTensorMap` divisor: `D \ D'` divides the
/// spectra and stays compact, and `D \ t` scales each block's leading
/// (bond) axis by the reciprocal spectrum, `O(Σ_c k_c m_c)` with no LU and no
/// `Σ_c k_c²` divisor buffer. `None` leaves every other layout to the dense
/// route, which owns its validation order.
fn checked_compact_divisor_solve<R, D>(
    tensor: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    output: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<TensorMap<R, D>>, Error>
where
    D: TensorScalar,
{
    let Some(spectrum) = tensor.spectrum() else {
        return Ok(None);
    };
    let divisor = &tensor.owned_body().expect("compact divisor is owned").space;
    if !checked_compact_spectrum_layout(divisor, divisor, spectrum) {
        return Ok(None);
    }
    reject_singular_compact_divisor(spectrum)?;
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

impl<R, D> TypedTensorExpDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    fn exp(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
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
        if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            // An endomorphism's swapped space is its own, so the inverse
            // layout proof also certifies `body.space` as the destination.
            if checked_compact_spectrum_layout(&body.space, &body.space, spectrum) {
                return Ok(tensor.with_spectrum(exp_spectrum(spectrum)?));
            }
        }
        let mut dense = tensor.runtime.lease_dense();
        let factor = tenet_matrixalgebra::seam::exp_pade13_direct_into_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())
                .map_err(Error::from)?,
        )
        .map_err(Error::from)?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }
}
