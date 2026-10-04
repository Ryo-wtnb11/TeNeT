use super::*;

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// TensorKit 0.17 / MatrixAlgebraKit `left_null`: `n : codomain <- W` with
    /// `n^H * t = 0`.
    ///
    /// # Null bond
    ///
    /// `W` is a fresh non-dual single-leg bond space carrying, per coupled
    /// sector `c`, the `rows_c − rank_c` null directions; `rank_c` is the
    /// numerical rank under the same cutoff as the sector's compact SVD, counting
    /// `σ > ε(dtype) · max(rows_c, cols_c) · σ_max,c` as nonzero. A sector
    /// with no null directions is absent from `W`, so `W` is empty for a
    /// numerically full-rank tensor. Note this is *not*
    /// TensorKit/MatrixAlgebraKit's default `left_null`, which without a
    /// truncation argument is QR-based and counts only the structural nullity
    /// `rows_c − min(rows_c, cols_c)` (MatrixAlgebraKit
    /// `interface/orthnull.jl`, the `alg::Nothing` mode); the seam's behavior
    /// corresponds to their SVD mode with a tolerance.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// Sectorwise cubic for dense input: one compact SVD per coupled sector
    /// plus an orthonormal completion where needed. An owned Host compact
    /// diagonal reads its spectrum directly and writes rectangular coordinate
    /// factors in O(Σ k_c + Σ k_c q_c) work and storage, where q_c is nullity:
    /// the dense route's rank cutoff `ε max(rows, cols) σ_max,c` is applied to
    /// the magnitudes directly, and a nonfinite entry is rejected before any
    /// work.
    /// A lazy adjoint runs the owned parent's
    /// [`Self::right_null`] and returns its detached adjoint, without
    /// materializing the receiver.
    pub(super) fn left_null_multiplicity_free(&self) -> Result<Self, Error>
    where
        D: FactorizationScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .adjoint()?
                .right_null_multiplicity_free()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                let out = tenet_matrixalgebra::seam::left_null_diagonal_dyn(&body.space, spectrum)?;
                return Ok(self.wrap_bound_factor(out));
            }
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::seam::left_null_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(self.wrap_bound_factor(out))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `right_null`: `n : W <- domain` with
    /// `t * n^H = 0`.
    ///
    /// # Null bond
    ///
    /// As [`Self::left_null`], mirrored: `W` is a fresh non-dual single-leg
    /// bond space with `cols_c − rank_c` directions per coupled sector under
    /// the same SVD numerical-rank cutoff, sectors with none absent — and the
    /// same divergence from TensorKit/MatrixAlgebraKit's QR-based default
    /// applies.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// As [`Self::left_null`], including its direct compact-diagonal route.
    /// A lazy adjoint mirrors the parent
    /// redirect described there.
    pub(super) fn right_null_multiplicity_free(&self) -> Result<Self, Error>
    where
        D: FactorizationScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .adjoint()?
                .left_null_multiplicity_free()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                let out =
                    tenet_matrixalgebra::seam::right_null_diagonal_dyn(&body.space, spectrum)?;
                return Ok(self.wrap_bound_factor(out));
            }
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::seam::right_null_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(self.wrap_bound_factor(out))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `left_polar`: the polar decomposition
    /// `t = w ∘ p`, returned as a [`LeftPolar`] — `w` isometric (`w† ∘ w = id` on the
    /// domain) and `p` Hermitian positive semidefinite.
    ///
    /// Factor spaces per TensorKit 0.17: `w` lives on the input's
    /// own space `codomain <- domain`, `p` on `domain <- domain`. TensorKit
    /// also exposes algorithm kinds for the polars; TeNeT deliberately does
    /// not. A lazy typed adjoint executes
    /// the opposite polar on its exact owned parent, keeps the already-owned
    /// positive factor, and returns an owned adjoint of the isometry without
    /// materializing the receiver.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered — in
    /// particular [`Error::Operation`] when some coupled-sector matrix has
    /// fewer rows than columns (the left polar needs every sector at least as
    /// tall as it is wide).
    ///
    /// # Complexity
    ///
    /// Dense input costs `O(Σ_c n_c³)` sectorwise. An owned compact
    /// diagonal uses `O(Σ_c n_c)` compact factor values and no dense SVD.
    pub(super) fn left_polar_multiplicity_free(&self) -> Result<LeftPolar<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                let LeftPolar { w, p } =
                    tenet_matrixalgebra::seam::left_polar_diagonal_spectra_dyn(
                        &body.space,
                        spectrum,
                    )?;
                return Ok(LeftPolar {
                    w: self.with_spectrum(w),
                    p: self.with_spectrum(p),
                });
            }
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let mut dense = self.runtime.lease_dense();
            let mut lease = self.runtime.lease_context()?;
            let LeftPolar { w, p } = tenet_matrixalgebra::seam::left_polar_adjoint_parent_dyn(
                dense.dense(),
                lease.context().multiplicity_free_lane::<D>()?,
                &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
            )?;
            return Ok(LeftPolar {
                w: self.wrap_bound_factor(w),
                p: self.wrap_bound_factor(p),
            });
        }
        // Dense lease before the context lease — the polar seam recouples
        // internally, so unlike QR/LQ/null it takes the context lane; the
        // lease order matches every existing site that takes both lanes.
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let (bound_space, bound_payload) = self.bound_payload()?;
        let LeftPolar { w, p } = tenet_matrixalgebra::seam::left_polar_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>()?,
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(LeftPolar {
            w: self.wrap_bound_factor(w),
            p: self.wrap_bound_factor(p),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `right_polar`: the polar
    /// decomposition `t = p ∘ wh`, returned as a [`RightPolar`] — `p`
    /// Hermitian positive semidefinite and `wh` a coisometry
    /// (`wh ∘ wh† = id` on the codomain).
    ///
    /// Factor spaces per TensorKit 0.17: `p` on
    /// `codomain <- codomain`, `wh` on the input's own space
    /// `codomain <- domain`. Everything [`Self::left_polar`] says about
    /// algorithm kinds, adjoint views and the compact-diagonal route holds
    /// here unchanged.
    ///
    /// # Errors
    ///
    /// As [`Self::left_polar`], mirrored: [`Error::Operation`] when some
    /// coupled-sector matrix has fewer columns than rows.
    ///
    /// # Complexity
    ///
    /// As [`Self::left_polar`], with compact factor values for an owned
    /// compact diagonal.
    pub(super) fn right_polar_multiplicity_free(&self) -> Result<RightPolar<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                let RightPolar { p, wh } =
                    tenet_matrixalgebra::seam::right_polar_diagonal_spectra_dyn(
                        &body.space,
                        spectrum,
                    )?;
                return Ok(RightPolar {
                    p: self.with_spectrum(p),
                    wh: self.with_spectrum(wh),
                });
            }
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let mut dense = self.runtime.lease_dense();
            let mut lease = self.runtime.lease_context()?;
            let RightPolar { p, wh: w } =
                tenet_matrixalgebra::seam::right_polar_adjoint_parent_dyn(
                    dense.dense(),
                    lease.context().multiplicity_free_lane::<D>()?,
                    &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
                )?;
            return Ok(RightPolar {
                p: self.wrap_bound_factor(p),
                wh: self.wrap_bound_factor(w),
            });
        }
        // See `left_polar` for the lease order rationale.
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let (bound_space, bound_payload) = self.bound_payload()?;
        let RightPolar { p, wh: w } = tenet_matrixalgebra::seam::right_polar_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>()?,
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(RightPolar {
            p: self.wrap_bound_factor(p),
            wh: self.wrap_bound_factor(w),
        })
    }
}

impl<R, D> TypedTensorNullDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn left_null(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.left_null_multiplicity_free()
    }

    fn right_null(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.right_null_multiplicity_free()
    }
}

impl<R, D> TypedTensorPolarDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn left_polar(tensor: &TensorMap<R, D>) -> Result<LeftPolar<TensorMap<R, D>>, Error> {
        tensor.left_polar_multiplicity_free()
    }

    fn right_polar(tensor: &TensorMap<R, D>) -> Result<RightPolar<TensorMap<R, D>>, Error> {
        tensor.right_polar_multiplicity_free()
    }
}

impl<R, D> TypedTensorNullDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn left_null(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if matches!(&tensor.repr, TypedTensorRepr::Adjoint(_)) {
            return Self::right_null(&tensor.adjoint()?)?
                .adjoint()?
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::from);
        }
        let body = tensor
            .owned_body()
            .expect("checked Generic left-null input is owned after lazy dispatch");
        let factor = tenet_matrixalgebra::seam::left_null_checked_generic(
            RuntimeDense(&tensor.runtime),
            owned_factor_source(body)?,
        )?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }

    fn right_null(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if matches!(&tensor.repr, TypedTensorRepr::Adjoint(_)) {
            return Self::left_null(&tensor.adjoint()?)?
                .adjoint()?
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::from);
        }
        let body = tensor
            .owned_body()
            .expect("checked Generic right-null input is owned after lazy dispatch");
        let factor = tenet_matrixalgebra::seam::right_null_checked_generic(
            RuntimeDense(&tensor.runtime),
            owned_factor_source(body)?,
        )?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }
}

impl<R, D> TypedTensorPolarDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn left_polar(
        tensor: &TensorMap<R, D>,
    ) -> Result<LeftPolar<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>>
    {
        match &tensor.repr {
            TypedTensorRepr::Owned(body) => {
                let LeftPolar { w, p } = tenet_matrixalgebra::seam::left_polar_checked_generic(
                    RuntimeDense(&tensor.runtime),
                    owned_factor_source(body)?,
                )?;
                Ok(LeftPolar {
                    w: tensor.factor_output(w),
                    p: tensor.factor_output(p),
                })
            }
            TypedTensorRepr::Adjoint(view) => {
                let mut dense = tensor.runtime.lease_dense();
                let input = BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
                    .map_err(Error::from)?;
                let RightPolar { p, wh: w } =
                    tenet_matrixalgebra::seam::left_polar_adjoint_parent_dyn_checked_generic(
                        dense.dense(),
                        &input,
                    )?;
                let w = wrap_factor_on(&tensor.runtime, w)
                    .adjoint()?
                    .materialized_tensor_uncached()
                    .map_err(GenericTensorError::from)?;
                Ok(LeftPolar {
                    w,
                    p: wrap_factor_on(&tensor.runtime, p),
                })
            }
        }
    }

    fn right_polar(
        tensor: &TensorMap<R, D>,
    ) -> Result<RightPolar<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>>
    {
        match &tensor.repr {
            TypedTensorRepr::Owned(body) => {
                let RightPolar { p, wh } = tenet_matrixalgebra::seam::right_polar_checked_generic(
                    RuntimeDense(&tensor.runtime),
                    owned_factor_source(body)?,
                )?;
                Ok(RightPolar {
                    p: tensor.factor_output(p),
                    wh: tensor.factor_output(wh),
                })
            }
            TypedTensorRepr::Adjoint(view) => {
                let mut dense = tensor.runtime.lease_dense();
                let input = BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
                    .map_err(Error::from)?;
                let LeftPolar { w, p } =
                    tenet_matrixalgebra::seam::right_polar_adjoint_parent_dyn_checked_generic(
                        dense.dense(),
                        &input,
                    )?;
                let w = wrap_factor_on(&tensor.runtime, w)
                    .adjoint()?
                    .materialized_tensor_uncached()
                    .map_err(GenericTensorError::from)?;
                Ok(RightPolar {
                    p: wrap_factor_on(&tensor.runtime, p),
                    wh: w,
                })
            }
        }
    }
}
