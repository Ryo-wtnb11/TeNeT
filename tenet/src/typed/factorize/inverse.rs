use super::*;

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// Multiplicity-free implementation of the public mode-dispatched inverse.
    pub(super) fn inv_multiplicity_free(&self) -> Result<Self, Error>
    where
        D: AdvancedLinalgScalar,
    {
        if let Some(spectrum) = self.spectrum() {
            return Ok(self.with_spectrum(inv_spectrum(spectrum)?));
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            // (A†)^-1 = (A^-1)†. Avoid materializing the receiver by
            // solving the owned parent, then detach the final adjoint so the
            // result retains neither the parent inverse nor its payload.
            return self
                .adjoint()?
                .inv_multiplicity_free()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::seam::inv_direct_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(self.wrap_bound_factor(out))
    }

    /// Multiplicity-free implementation of [`Self::pinv`].
    pub(super) fn pinv_multiplicity_free(&self, rcond: f64) -> Result<Self, Error>
    where
        D: AdvancedLinalgScalar,
    {
        // Ahead of the storage split, so both arms answer alike: the seam
        // repeats this check for its own callers, but the compact arm never
        // reaches the seam.
        if !rcond.is_finite() || rcond < 0.0 {
            return Err(Error::InvalidArgument(
                "pinv rcond must be finite and non-negative".to_string(),
            ));
        }
        if let Some(spectrum) = self.spectrum() {
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
            return Ok(self.with_spectrum(map_spectrum(spectrum, |value| {
                Ok(if value.abs_value() > cutoff {
                    value.recip_value()
                } else {
                    D::from_real(0.0)
                })
            })?));
        }
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let out = match &self.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::seam::pinv_adjoint_parent_dyn(
                dense.dense(),
                lease.context().multiplicity_free_lane::<D>()?,
                &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
                rcond,
            )
            .map_err(pinv_seam_error)?,
            TypedTensorRepr::Owned(_) => {
                let (bound_space, bound_payload) = self.bound_payload()?;
                tenet_matrixalgebra::seam::pinv_dyn(
                    dense.dense(),
                    lease.context().multiplicity_free_lane::<D>()?,
                    &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
                    rcond,
                )
                .map_err(pinv_seam_error)?
            }
        };
        Ok(self.wrap_bound_factor(out))
    }
}

impl<R, D> TypedTensorInvDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
{
    fn inv(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.inv_multiplicity_free()
    }
}

impl<R, D> TypedTensorPinvDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
{
    fn pinv(tensor: &TensorMap<R, D>, rcond: f64) -> Result<TensorMap<R, D>, Error> {
        tensor.pinv_multiplicity_free(rcond)
    }
}

impl<R, D> TypedTensorInvDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    fn inv(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if matches!(&tensor.repr, TypedTensorRepr::Adjoint(_)) {
            return Self::inv(&tensor.adjoint()?)?
                .adjoint()?
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::from);
        }
        if let Some(spectrum) = tensor.spectrum() {
            // A proven bond is its own swapped space, so, as in
            // multiplicity-free mode, the compact arm needs neither the
            // isomorphism preflight nor a separately admitted output root.
            let space = tensor.logical_space();
            if checked_compact_spectrum_layout(space, space, spectrum) {
                return Ok(tensor.with_spectrum(inv_spectrum(spectrum)?));
            }
        }
        let source = tensor.logical_space();
        if !tenet_matrixalgebra::seam::factor_isomorphic_checked_generic(source)? {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "inv requires isomorphic codomain and domain",
                },
            )
            .into());
        }
        let homspace = FusionTreeHomSpace::new(
            source.space().homspace().domain().clone(),
            source.space().homspace().codomain().clone(),
        );
        let output = tenet_matrixalgebra::seam::factor_output_space_checked_generic(
            source.provider_arc(),
            homspace,
        )?;
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
}

impl<R, D> TypedTensorPinvDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    fn pinv(
        tensor: &TensorMap<R, D>,
        rcond: f64,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if !rcond.is_finite() || rcond < 0.0 {
            return Err(Error::InvalidArgument(
                "pinv rcond must be finite and non-negative".to_string(),
            )
            .into());
        }
        if matches!(&tensor.repr, TypedTensorRepr::Adjoint(_)) {
            return Self::pinv(&tensor.adjoint()?, rcond)?
                .adjoint()?
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::from);
        }
        let source = tensor.logical_space();
        let output = tenet_matrixalgebra::seam::factor_output_space_checked_generic(
            source.provider_arc(),
            FusionTreeHomSpace::new(
                source.space().homspace().domain().clone(),
                source.space().homspace().codomain().clone(),
            ),
        )?;
        let body = tensor
            .owned_body()
            .expect("checked Generic pinv input is owned after lazy dispatch");
        if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if checked_compact_spectrum_layout(&body.space, &output, spectrum) {
                let sigma_max = spectrum.iter().flat_map(|entry| &entry.values).try_fold(
                    0.0_f64,
                    |largest, &value| {
                        let magnitude = value.abs_value();
                        let rounded = D::from_real(magnitude).abs_value();
                        (magnitude.is_finite() && rounded.is_finite()).then(|| largest.max(rounded))
                    },
                );
                if let Some(sigma_max) = sigma_max {
                    let cutoff = rcond * sigma_max;
                    // The dense SVD can flush a retained subnormal to zero.
                    if spectrum
                        .iter()
                        .flat_map(|entry| &entry.values)
                        .all(|&value| {
                            D::from_real(value.abs_value()).abs_value() <= cutoff
                                || (value.abs_value() >= D::safe_minimum()
                                    && value.recip_value().abs_value().is_finite())
                        })
                    {
                        let mapped = map_spectrum(spectrum, |value| {
                            Ok(if D::from_real(value.abs_value()).abs_value() > cutoff {
                                value.recip_value()
                            } else {
                                D::from_real(0.0)
                            })
                        })?;
                        return Ok(tensor.with_spectrum_on(output, mapped));
                    }
                }
            }
        }
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
}
