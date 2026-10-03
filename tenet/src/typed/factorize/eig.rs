use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    /// Checked-Generic general eigenvalues for owned host tensors.
    pub(super) fn eig_vals_checked_generic(
        &self,
    ) -> CheckedGenericSpectrumResult<R, num_complex::Complex64> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic eig_vals does not accept lazy adjoints".to_string(),
            )));
        };
        let direct = if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if is_diagonal_bond_space(body.space.space()) {
                tenet_matrixalgebra::seam::eig_vals_diagonal_dyn(&body.space, spectrum).map_err(
                    |error| GenericTensorError::Plan(CheckedGenericPlanError::Operation(error)),
                )?
            } else {
                None
            }
        } else {
            None
        };
        let raw = if let Some(raw) = direct {
            raw
        } else {
            let mut dense = self.runtime.lease_dense();
            let payload = body.materialized_dense_data();
            let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
                .map_err(|error| GenericTensorError::Facade(error.into()))?;
            tenet_matrixalgebra::seam::eig_vals_dyn_checked_generic(dense.dense(), &input)?
        };
        let provider = self.logical_space().provider();
        let mut decoded = raw
            .into_iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: provider.try_decode_label(entry.sector)?,
                    values: entry.values,
                })
            })
            .collect::<Result<Vec<_>, <R as TypedSectorAdmission>::Error>>()
            .map_err(|error| GenericTensorError::Plan(CheckedGenericPlanError::Provider(error)))?;
        decoded.sort_by(|left, right| left.sector.cmp(&right.sector));
        Ok(decoded)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    /// Checked-Generic Hermitian eigenvalues for owned host tensors.
    pub(super) fn eigh_vals_checked_generic(&self) -> CheckedGenericSpectrumResult<R, f64> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic eigh_vals does not accept lazy adjoints".to_string(),
            )));
        };
        let direct = if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if is_diagonal_bond_space(body.space.space()) {
                tenet_matrixalgebra::seam::eigh_vals_diagonal_dyn(&body.space, spectrum).map_err(
                    |error| GenericTensorError::Plan(CheckedGenericPlanError::Operation(error)),
                )?
            } else {
                None
            }
        } else {
            None
        };
        let raw = if let Some(raw) = direct {
            raw
        } else {
            let mut dense = self.runtime.lease_dense();
            let payload = body.materialized_dense_data();
            let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
                .map_err(|error| GenericTensorError::Facade(error.into()))?;
            tenet_matrixalgebra::seam::eigh_vals_dyn_checked_generic(dense.dense(), &input)?
        };
        let provider = self.logical_space().provider();
        let mut decoded = raw
            .into_iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: provider.try_decode_label(entry.sector)?,
                    values: entry.values,
                })
            })
            .collect::<Result<Vec<_>, <R as TypedSectorAdmission>::Error>>()
            .map_err(|error| GenericTensorError::Plan(CheckedGenericPlanError::Provider(error)))?;
        decoded.sort_by(|left, right| left.sector.cmp(&right.sector));
        Ok(decoded)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    pub(super) fn eigh_full_checked_generic(
        &self,
    ) -> Result<Eigh<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::Facade)?
                .eigh_full_checked_generic();
        }
        let body = self.owned_body().expect("owned checked Generic EIGH input");
        let direct = match body.data.as_ref() {
            TypedData::Diagonal(spectrum) if is_diagonal_bond_space(body.space.space()) => {
                tenet_matrixalgebra::seam::eigh_full_diagonal_dyn_checked_generic(
                    &body.space,
                    spectrum,
                )?
            }
            _ => None,
        };
        let out = if let Some(out) = direct {
            out
        } else {
            let mut dense = self.runtime.lease_dense();
            let payload = body.materialized_dense_data();
            let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
                .map_err(|error| GenericTensorError::Facade(error.into()))?;
            tenet_matrixalgebra::seam::eigh_full_dyn_checked_generic(dense.dense(), &input)?
        };
        let (v, mut eigenvalues) = out.into_parts();
        let d = diagonal_factor_on_checked(
            &self.runtime,
            Arc::clone(body.space.provider_arc()),
            &mut eigenvalues,
            D::from_real,
        )?;
        Ok(Eigh {
            d,
            v: wrap_factor_on(&self.runtime, v),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    #[expect(
        clippy::type_complexity,
        reason = "the checked eigensolver returns its diagonal and eigenvector factors together"
    )]
    pub(super) fn eig_full_checked_generic(
        &self,
    ) -> Result<
        Eig<TensorMap<R, <D as FactorScalar>::Eig>>,
        GenericTensorError<<R as CheckedGenericFusion>::Error>,
    > {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::Facade)?
                .eig_full_checked_generic();
        }
        let body = self.owned_body().expect("owned checked Generic EIG input");
        let direct = match body.data.as_ref() {
            TypedData::Diagonal(spectrum) if is_diagonal_bond_space(body.space.space()) => {
                tenet_matrixalgebra::seam::eig_full_diagonal_dyn_checked_generic(
                    &body.space,
                    spectrum,
                )?
            }
            _ => None,
        };
        let out = if let Some(out) = direct {
            out
        } else {
            let mut dense = self.runtime.lease_dense();
            let payload = body.materialized_dense_data();
            let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
                .map_err(|error| GenericTensorError::Facade(error.into()))?;
            tenet_matrixalgebra::seam::eig_full_dyn_checked_generic(dense.dense(), &input)?
        };
        let (v, mut eigenvalues) = out.into_parts();
        let d = diagonal_factor_on_checked(
            &self.runtime,
            Arc::clone(body.space.provider_arc()),
            &mut eigenvalues,
            <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
        )?;
        Ok(Eig {
            d,
            v: wrap_factor_on(&self.runtime, v),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// TensorKit 0.17 / MatrixAlgebraKit `eigh_full`: the Hermitian
    /// eigendecomposition `t = v * d * v^H` of an endomorphism, returned as an
    /// [`Eigh`].
    ///
    /// `d : bond <- bond` carries the eigenvalues in compact diagonal storage
    /// (TensorKit's `DiagonalTensorMap`), so `v.compose(&d)` takes the
    /// bond-scaling path; `v : codomain <- bond` is the eigenbasis. The
    /// eigenvalues are real for both payload dtypes — TensorKit's Hermitian `D`
    /// is real too — but `d` keeps the payload dtype `D` so it composes with
    /// `v` directly.
    /// An admitted owned compact diagonal builds the sorted spectrum and dense
    /// permutation eigenbasis without materializing a dense input.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] when the tensor is not an endomorphism or its
    /// coupled blocks are not Hermitian, and otherwise
    /// [`Error::Core`] / [`Error::FusionAlgebra`] from the seam — which owns
    /// those rules, so they are not re-checked here.
    pub(super) fn eigh_full_multiplicity_free(&self) -> Result<Eigh<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(out) =
                    tenet_matrixalgebra::seam::eigh_full_diagonal_dyn(&body.space, spectrum)?
                {
                    let (v, mut eigenvalues) = out.into_parts();
                    return Ok(Eigh {
                        d: self.diagonal_factor(&mut eigenvalues, D::from_real)?,
                        v: self.wrap_bound_factor(v),
                    });
                }
            }
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eigh_full_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::seam::eigh_full_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        let (v, mut eigenvalues) = out.into_parts();
        Ok(Eigh {
            d: self.diagonal_factor(&mut eigenvalues, D::from_real)?,
            v: self.wrap_bound_factor(v),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eigh_vals`: the Hermitian eigenvalues
    /// per coupled sector, and nothing else.
    ///
    /// No factor and no bond space is built, so this is the cheap way to ask
    /// about a spectrum — the [`Self::svd_vals`] of the eigendecompositions.
    /// An owned compact diagonal with finite, exactly real entries is read
    /// directly without materializing input blocks or invoking a dense solver.
    /// Other inputs retain the usual Hermiticity admission and dense path.
    ///
    /// # Errors
    ///
    /// [`Self::eigh_full`]'s, plus [`Error::FusionAlgebra`] when the provider
    /// cannot decode a coupled sector its own algebra produced.
    pub(super) fn eigh_vals_multiplicity_free(
        &self,
    ) -> Result<Vec<SectorSpectrum<R::Sector>>, Error>
    where
        D: FactorizationScalar,
    {
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(raw) =
                    tenet_matrixalgebra::seam::eigh_vals_diagonal_dyn(&body.space, spectrum)?
                {
                    return self.decode_spectrum(raw);
                }
            }
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eigh_vals_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let raw = tenet_matrixalgebra::seam::eigh_vals_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        self.decode_spectrum(raw)
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eig_full`: the general
    /// (non-Hermitian) eigendecomposition `t = v * d * v^-1` of an
    /// endomorphism, returned as an [`Eig`].
    ///
    /// Both factors are complex whatever `D` is: a real matrix's eigenpairs are
    /// complex in general, and TensorKit's `eigen` likewise returns
    /// `ComplexF64` `D` and `V` for a real argument. `d` carries the spectrum
    /// in compact diagonal storage.
    /// An admitted owned compact diagonal builds the sorted spectrum and dense
    /// permutation eigenbasis without materializing a dense input.
    ///
    /// # The `D::Eig` bound
    ///
    /// The `where` clause is vacuous for the payload types this facade
    /// admits — `f64` and `Complex64` have `Eig = Complex64`, `f32` and
    /// `Complex32` have `Eig = Complex32`, all [`TensorScalar`]s. It is written
    /// out because
    /// [`tenet_matrixalgebra::FactorScalar::Eig`] is the wider seam's associated
    /// type and is not constrained to this facade's scalars, so without it the
    /// factors could not be `TensorMap`s at all. Per-method rather than on the
    /// impl block, so nothing outside the `eig_*` row pays for it.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] when the tensor is not an endomorphism, and
    /// otherwise [`Error::Core`] / [`Error::FusionAlgebra`] from the seam.
    #[allow(clippy::type_complexity)]
    pub(super) fn eig_full_multiplicity_free(
        &self,
    ) -> Result<Eig<TensorMap<R, <D as FactorScalar>::Eig>>, Error>
    where
        D: AdvancedLinalgScalar,
        <D as FactorScalar>::Eig: TensorScalar,
    {
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(out) =
                    tenet_matrixalgebra::seam::eig_full_diagonal_dyn(&body.space, spectrum)?
                {
                    let (v, mut eigenvalues) = out.into_parts();
                    return Ok(Eig {
                        d: diagonal_factor_on(
                            &self.runtime,
                            self.logical_space(),
                            &mut eigenvalues,
                            <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
                        )?,
                        v: wrap_factor_on(&self.runtime, v),
                    });
                }
            }
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eig_full_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::seam::eig_full_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        let (v, mut eigenvalues) = out.into_parts();
        Ok(Eig {
            d: diagonal_factor_on(
                &self.runtime,
                self.logical_space(),
                &mut eigenvalues,
                <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
            )?,
            v: wrap_factor_on(&self.runtime, v),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eig_vals`: the general eigenvalues
    /// per coupled sector, and nothing else. `Complex64` for every payload dtype.
    /// An admitted owned compact diagonal is read without dense materialization.
    ///
    /// # Errors
    ///
    /// [`Self::eig_full`]'s, plus [`Error::FusionAlgebra`] when the provider
    /// cannot decode a coupled sector its own algebra produced.
    pub(super) fn eig_vals_multiplicity_free(
        &self,
    ) -> Result<Vec<SectorSpectrum<R::Sector, num_complex::Complex64>>, Error>
    where
        D: AdvancedLinalgScalar,
        // Carried across the whole row even though this member builds no
        // factor: the three are one API surface, and a caller who can spell two
        // of them but not the third would be reading an accident.
        <D as FactorScalar>::Eig: TensorScalar,
    {
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(raw) =
                    tenet_matrixalgebra::seam::eig_vals_diagonal_dyn(&body.space, spectrum)?
                {
                    return self.decode_spectrum(raw);
                }
            }
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eig_vals_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let raw = tenet_matrixalgebra::seam::eig_vals_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        self.decode_spectrum(raw)
    }
}

impl<R, D> TypedTensorEighValsDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
            Sector = <R as SectorCodec>::Sector,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn eigh_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, Error> {
        tensor.eigh_vals_multiplicity_free()
    }
}

impl<R, D> TypedTensorEighDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn eigh_full(tensor: &TensorMap<R, D>) -> Result<Eigh<TensorMap<R, D>>, Error> {
        tensor.eigh_full_multiplicity_free()
    }
}

impl<R, D> TypedTensorEigValsDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
            Sector = <R as SectorCodec>::Sector,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    fn eig_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<
        Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, num_complex::Complex64>>,
        Error,
    > {
        tensor.eig_vals_multiplicity_free()
    }
}

impl<R, D> TypedTensorEigDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    fn eig_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Eig<TensorMap<R, <D as FactorScalar>::Eig>>, Error> {
        tensor.eig_full_multiplicity_free()
    }
}

impl<R, D> TypedTensorEighValsDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn eigh_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<
        Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>,
        GenericTensorError<<R as CheckedGenericFusion>::Error>,
    > {
        tensor.eigh_vals_checked_generic()
    }
}

impl<R, D> TypedTensorEighDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn eigh_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Eigh<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.eigh_full_checked_generic()
    }
}

impl<R, D> TypedTensorEigValsDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    fn eig_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<
        Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, num_complex::Complex64>>,
        GenericTensorError<<R as CheckedGenericFusion>::Error>,
    > {
        tensor.eig_vals_checked_generic()
    }
}

impl<R, D> TypedTensorEigDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    fn eig_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<
        Eig<TensorMap<R, <D as FactorScalar>::Eig>>,
        GenericTensorError<<R as CheckedGenericFusion>::Error>,
    > {
        tensor.eig_full_checked_generic()
    }
}
