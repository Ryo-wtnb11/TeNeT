use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    /// Checked-Generic full QR for owned host tensors.
    pub(super) fn qr_full_checked_generic(
        &self,
    ) -> Result<Qr<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic qr_full does not accept lazy adjoints".to_string(),
            )));
        };
        let Qr { q, r } = tenet_matrixalgebra::seam::qr_full_checked_generic(
            RuntimeDense(&self.runtime),
            owned_factor_source(body).map_err(GenericTensorError::Facade)?,
        )?;
        Ok(Qr {
            q: self.factor_output(q),
            r: self.factor_output(r),
        })
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
    /// Checked-Generic compact LQ for owned host tensors.
    pub(super) fn lq_compact_checked_generic(
        &self,
    ) -> Result<Lq<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic lq_compact does not accept lazy adjoints".to_string(),
            )));
        };
        let Lq { l, q } = tenet_matrixalgebra::seam::lq_compact_checked_generic(
            RuntimeDense(&self.runtime),
            owned_factor_source(body).map_err(GenericTensorError::Facade)?,
        )?;
        Ok(Lq {
            l: self.factor_output(l),
            q: self.factor_output(q),
        })
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
    /// Checked-Generic compact QR for owned host tensors.
    ///
    /// The first Generic decomposition leaf deliberately rejects lazy-adjoint
    /// inputs; no operation-local whole-payload fallback is introduced here.
    pub(super) fn qr_compact_checked_generic(
        &self,
    ) -> Result<Qr<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic qr_compact does not accept lazy adjoints".to_string(),
            )));
        };
        let Qr { q, r } = tenet_matrixalgebra::seam::qr_compact_checked_generic(
            RuntimeDense(&self.runtime),
            owned_factor_source(body).map_err(GenericTensorError::Facade)?,
        )?;
        Ok(Qr {
            q: self.factor_output(q),
            r: self.factor_output(r),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// The direct QR of an owned compact diagonal, or `None` for any other
    /// storage.
    fn try_qr_diagonal(&self) -> Result<Option<Qr<Self>>, Error>
    where
        D: FactorizationScalar,
    {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Ok(None);
        };
        let TypedData::Diagonal(spectrum) = body.data.as_ref() else {
            return Ok(None);
        };
        let Qr { q, r } = tenet_matrixalgebra::seam::qr_diagonal_dyn(&body.space, spectrum)?;
        let wrap = |values| Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::diagonal(body.space.clone(), values)),
        };
        Ok(Some(Qr {
            q: wrap(q),
            r: wrap(r),
        }))
    }

    /// The direct LQ of an owned compact diagonal, or `None` for any other
    /// storage.
    fn try_lq_diagonal(&self) -> Result<Option<Lq<Self>>, Error>
    where
        D: FactorizationScalar,
    {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Ok(None);
        };
        let TypedData::Diagonal(spectrum) = body.data.as_ref() else {
            return Ok(None);
        };
        let Lq { l, q } = tenet_matrixalgebra::seam::lq_diagonal_dyn(&body.space, spectrum)?;
        let wrap = |values| Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::diagonal(body.space.clone(), values)),
        };
        Ok(Some(Lq {
            l: wrap(l),
            q: wrap(q),
        }))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `qr_compact`: `t = q * r` with `q`
    /// carrying orthonormal columns per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// `O(Σ_c n_c³)` — sectorwise cubic; the seam runs one dense QR per
    /// coupled-sector matrix. A lazy adjoint first allocates its whole logical
    /// dense payload as an operation-local owned tensor, released with the
    /// operation, and the returned factors are owned. An admitted owned compact
    /// diagonal uses O(Σ_c k_c) spectrum work/storage and no dense QR. Both
    /// factors preserve the input bond, including dual orientation.
    pub(super) fn qr_compact_multiplicity_free(&self) -> Result<Qr<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let Some(factors) = self.try_qr_diagonal()? {
            return Ok(factors);
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .qr_compact_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Qr { q, r } = tenet_matrixalgebra::seam::qr_compact_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(Qr {
            q: self.wrap_bound_factor(q),
            r: self.wrap_bound_factor(r),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `qr_full`: `t = q * r` with a square
    /// `q` per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// For sector shape `m_c x n_c`, dense work is `O(m_c² n_c)` when
    /// `m_c <= n_c`, and `O(m_c²(n_c + m_c))` when completion is required.
    /// Source packing and owned factor publication are additional costs. A
    /// lazy adjoint also allocates its whole logical payload for the call. A
    /// compact diagonal uses the same O(Σ_c k_c) route as compact QR.
    pub(super) fn qr_full_multiplicity_free(&self) -> Result<Qr<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let Some(factors) = self.try_qr_diagonal()? {
            return Ok(factors);
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .qr_full_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Qr { q, r } = tenet_matrixalgebra::seam::qr_full_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(Qr {
            q: self.wrap_bound_factor(q),
            r: self.wrap_bound_factor(r),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `lq_compact`: `t = l * q` with `q`
    /// carrying orthonormal rows per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// Sectorwise cubic. A lazy adjoint runs compact QR on its owned parent,
    /// reverses and adjoints the factors, then materializes both outputs into
    /// detached owned tensors, retaining neither parent factor buffer. A
    /// compact diagonal uses the QR spectrum route with exchanged factors,
    /// in O(Σ_c k_c) work/storage.
    pub(super) fn lq_compact_multiplicity_free(&self) -> Result<Lq<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let Some(factors) = self.try_lq_diagonal()? {
            return Ok(factors);
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let Qr { q, r } = self.adjoint()?.qr_compact_multiplicity_free()?;
            return Ok(Lq {
                l: r.adjoint()?.materialized_tensor_uncached()?,
                q: q.adjoint()?.materialized_tensor_uncached()?,
            });
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Lq { l, q } = tenet_matrixalgebra::seam::lq_compact_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(Lq {
            l: self.wrap_bound_factor(l),
            q: self.wrap_bound_factor(q),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `lq_full`: `t = l * q` with a square
    /// `q` per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// For sector shape `m_c x n_c`, dense work is `O(n_c² m_c)` when
    /// `n_c <= m_c`, and `O(n_c²(m_c + n_c))` when completion is required.
    /// Source packing, the sectorwise adjoint, and owned factor publication are
    /// additional costs. A lazy adjoint uses the parent full-QR route and two
    /// detached owned output payloads. An admitted owned compact diagonal uses
    /// the same O(Σ_c k_c) spectrum route as compact LQ.
    pub(super) fn lq_full_multiplicity_free(&self) -> Result<Lq<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let Some(factors) = self.try_lq_diagonal()? {
            return Ok(factors);
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let Qr { q, r } = self.adjoint()?.qr_full_multiplicity_free()?;
            return Ok(Lq {
                l: r.adjoint()?.materialized_tensor_uncached()?,
                q: q.adjoint()?.materialized_tensor_uncached()?,
            });
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Lq { l, q } = tenet_matrixalgebra::seam::lq_full_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(Lq {
            l: self.wrap_bound_factor(l),
            q: self.wrap_bound_factor(q),
        })
    }
}

impl<R, D> TypedTensorQrDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn qr_compact(tensor: &TensorMap<R, D>) -> Result<Qr<TensorMap<R, D>>, Error> {
        tensor.qr_compact_multiplicity_free()
    }
}

impl<R, D> TypedTensorLqDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn lq_compact(tensor: &TensorMap<R, D>) -> Result<Lq<TensorMap<R, D>>, Error> {
        tensor.lq_compact_multiplicity_free()
    }
}

impl<R, D> TypedTensorFullQrDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn qr_full(tensor: &TensorMap<R, D>) -> Result<Qr<TensorMap<R, D>>, Error> {
        tensor.qr_full_multiplicity_free()
    }
}

impl<R, D> TypedTensorFullLqDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn lq_full(tensor: &TensorMap<R, D>) -> Result<Lq<TensorMap<R, D>>, Error> {
        tensor.lq_full_multiplicity_free()
    }
}

impl<R, D> TypedTensorQrDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn qr_compact(
        tensor: &TensorMap<R, D>,
    ) -> Result<Qr<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.qr_compact_checked_generic()
    }
}

impl<R, D> TypedTensorLqDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn lq_compact(
        tensor: &TensorMap<R, D>,
    ) -> Result<Lq<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.lq_compact_checked_generic()
    }
}

impl<R, D> TypedTensorFullQrDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn qr_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Qr<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.qr_full_checked_generic()
    }
}

impl<R, D> TypedTensorFullLqDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn lq_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Lq<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &tensor.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic lq_full does not accept lazy adjoints".to_string(),
            )));
        };
        let Lq { l, q } = tenet_matrixalgebra::seam::lq_full_checked_generic(
            RuntimeDense(&tensor.runtime),
            owned_factor_source(body).map_err(GenericTensorError::Facade)?,
        )?;
        Ok(Lq {
            l: tensor.factor_output(l),
            q: tensor.factor_output(q),
        })
    }
}
