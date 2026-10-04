use super::*;

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
        let (out, route) = tenet_matrixalgebra::seam::eigh_full_checked_generic(
            RuntimeDense(&self.runtime),
            owned_factor_source(body).map_err(GenericTensorError::Facade)?,
        )?;
        let on_diagonal = route == tenet_matrixalgebra::seam::FactorRoute::Diagonal;
        let (v, mut eigenvalues) = out.into_parts();
        let d = if on_diagonal {
            diagonal_factor_on_source_checked(
                &self.runtime,
                &body.space,
                &mut eigenvalues,
                D::from_real,
            )?
        } else {
            diagonal_factor_on_checked(
                &self.runtime,
                Arc::clone(body.space.provider_arc()),
                &mut eigenvalues,
                D::from_real,
            )?
        };
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
        let (out, route) = tenet_matrixalgebra::seam::eig_full_checked_generic(
            RuntimeDense(&self.runtime),
            owned_factor_source(body).map_err(GenericTensorError::Facade)?,
        )?;
        let on_diagonal = route == tenet_matrixalgebra::seam::FactorRoute::Diagonal;
        let (v, mut eigenvalues) = out.into_parts();
        let d = if on_diagonal {
            diagonal_factor_on_source_checked(
                &self.runtime,
                &body.space,
                &mut eigenvalues,
                <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
            )?
        } else {
            diagonal_factor_on_checked(
                &self.runtime,
                Arc::clone(body.space.provider_arc()),
                &mut eigenvalues,
                <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
            )?
        };
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
    /// An owned compact diagonal builds the sorted spectrum and dense
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
                let out = tenet_matrixalgebra::seam::eigh_full_diagonal_dyn(&body.space, spectrum)?;
                let (v, mut eigenvalues) = out.into_parts();
                return Ok(Eigh {
                    d: self.diagonal_factor(&mut eigenvalues, D::from_real)?,
                    v: self.wrap_bound_factor(v),
                });
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

    /// TensorKit 0.17 / MatrixAlgebraKit `eig_full`: the general
    /// (non-Hermitian) eigendecomposition `t = v * d * v^-1` of an
    /// endomorphism, returned as an [`Eig`].
    ///
    /// Both factors are complex whatever `D` is: a real matrix's eigenpairs are
    /// complex in general, and TensorKit's `eigen` likewise returns
    /// `ComplexF64` `D` and `V` for a real argument. `d` carries the spectrum
    /// in compact diagonal storage.
    /// An owned compact diagonal builds the sorted spectrum and dense
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
                let out = tenet_matrixalgebra::seam::eig_full_diagonal_dyn(&body.space, spectrum)?;
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
