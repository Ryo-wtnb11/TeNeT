use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    pub(super) fn svd_full_checked_generic(
        &self,
    ) -> Result<Svd<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic svd_full does not accept lazy adjoints".to_string(),
            )));
        };
        let (factors, route) = tenet_matrixalgebra::seam::svd_full_checked_generic(
            RuntimeDense(&self.runtime),
            owned_factor_source(body).map_err(GenericTensorError::Facade)?,
        )?;
        let on_diagonal = route == tenet_matrixalgebra::seam::FactorRoute::Diagonal;
        let (u, vh, mut spectrum, row_dimensions, col_dimensions) = factors.into_parts();
        if full_svd_compact_bond(&u, &vh, &spectrum) {
            let space = if on_diagonal {
                tenet_matrixalgebra::seam::diagonal_bond_bound_space_on_source_checked_generic(
                    &body.space,
                    &spectrum,
                )?
            } else {
                tenet_matrixalgebra::seam::diagonal_bond_bound_space_generic_checked(
                    Arc::clone(body.space.provider_arc()),
                    &spectrum,
                )?
            };
            if full_svd_compact_layout(&space, &spectrum) {
                return Ok(Svd {
                    u: wrap_factor_on(&self.runtime, u),
                    s: diagonal_factor_on_bound(&self.runtime, space, &mut spectrum, D::from_real),
                    vh: wrap_factor_on(&self.runtime, vh),
                });
            }
        }
        let s = tenet_matrixalgebra::seam::rectangular_diagonal_bond_tensor_generic_checked(
            Arc::clone(body.space.provider_arc()),
            &spectrum,
            &row_dimensions,
            &col_dimensions,
            &D::from_real,
        )?;
        Ok(Svd {
            u: wrap_factor_on(&self.runtime, u),
            s: wrap_factor_on(&self.runtime, s),
            vh: wrap_factor_on(&self.runtime, vh),
        })
    }

    /// Checked-Generic compact SVD for owned host tensors.
    pub(super) fn svd_compact_checked_generic(
        &self,
    ) -> Result<Svd<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic svd_compact does not accept lazy adjoints".to_string(),
            )));
        };
        let ((u, vh, mut singular_values), route) =
            tenet_matrixalgebra::seam::svd_compact_checked_generic(
                RuntimeDense(&self.runtime),
                owned_factor_source(body).map_err(GenericTensorError::Facade)?,
            )?;
        let s = if route == tenet_matrixalgebra::seam::FactorRoute::Diagonal {
            diagonal_factor_on_source_checked(
                &self.runtime,
                &body.space,
                &mut singular_values,
                D::from_real,
            )?
        } else {
            diagonal_factor_on_checked(
                &self.runtime,
                Arc::clone(body.space.provider_arc()),
                &mut singular_values,
                D::from_real,
            )?
        };
        Ok(Svd {
            u: wrap_factor_on(&self.runtime, u),
            s,
            vh: wrap_factor_on(&self.runtime, vh),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// TensorKit 0.17 / MatrixAlgebraKit `svd_compact`: `t = u * s * vh` with
    /// the bond `min(rows, cols)` per coupled sector.
    ///
    /// Returns an [`Svd`] with `u : codomain <- bond`, `s : bond <- bond`
    /// and `vh : bond <- domain`.
    ///
    /// # Storage
    ///
    /// `s` is held in compact diagonal storage — `Σ_c k_c` values, not the
    /// `Σ_c k_c²` block-diagonal buffer — matching the `DiagonalTensorMap`
    /// TensorKit's own `svd_compact` returns. A downstream `u.compose(&s)` or
    /// `s.compose(&vh)` takes the O(d·n) bond-scaling path rather than a dense
    /// GEMM. [`Self::materialize`] builds the dense buffer on request; a
    /// caller who only needs the values should reach for
    /// [`Self::svd_vals`], which builds no factor at all.
    /// An owned compact-diagonal input is sorted directly by sector: no dense
    /// input or dense SVD is needed. The dense `u` and `vh` permutation
    /// factors still require `Σ_c k_c²` storage and writes; sorting costs
    /// `O(Σ_c k_c log k_c)`. A nonfinite entry returns
    /// [`Error::InvalidArgument`] before any work.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    /// straight from the matrix-algebra seam. As everywhere in this facade
    /// there are no pre-checks here: the seam owns the rules, and a second copy
    /// would be free to drift.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, Svd, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 9)?;
    ///
    /// let Svd { u, s, vh } = t.svd_compact(&[0], &[1])?;
    /// // `u` is an isometry: `u† ∘ u` is the identity on its domain.
    /// let gram = u.adjoint()?.compose(&u)?;
    /// let identity = TensorMap::isomorphism(&runtime, &u.domain(), &u.domain())?;
    /// assert!(gram.axpby(1.0, &identity, -1.0)?.norm(2.0)? <= 1e-12 * gram.norm(2.0)?.max(1.0));
    /// let rebuilt = u.compose(&s)?.compose(&vh)?;
    /// let max_err = rebuilt
    ///     .dense_data()?
    ///     .iter()
    ///     .zip(t.dense_data()?)
    ///     .map(|(a, b)| (a - b).abs())
    ///     .fold(0.0f64, f64::max);
    /// assert!(max_err < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub(super) fn svd_compact_multiplicity_free(&self) -> Result<Svd<Self>, Error>
    where
        D: FactorizationScalar,
    {
        let compact = match &self.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Diagonal(spectrum) => {
                    Some(tenet_matrixalgebra::seam::svd_compact_diagonal_factors_dyn(
                        &body.space,
                        spectrum,
                    )?)
                }
                TypedData::Dense(_) => None,
            },
            TypedTensorRepr::Adjoint(_) => None,
        };
        let (u, vh, mut spectrum) = if let Some(factors) = compact {
            factors
        } else {
            // The ordinary route keeps its dense-only lease and its compact-S
            // factor seam.
            let mut dense = self.runtime.lease_dense();
            match &self.repr {
                TypedTensorRepr::Adjoint(view) => {
                    tenet_matrixalgebra::seam::svd_compact_adjoint_factors_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
                    )?
                }
                TypedTensorRepr::Owned(_) => {
                    let (bound_space, bound_payload) = self.bound_payload()?;
                    tenet_matrixalgebra::seam::svd_compact_factors_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
                    )?
                }
            }
        };
        Ok(Svd {
            u: self.wrap_bound_factor(u),
            s: self.diagonal_factor(&mut spectrum, D::from_real)?,
            vh: self.wrap_bound_factor(vh),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `svd_full`: `t = u * s * vh` with
    /// square unitaries and a rectangular `s` per coupled sector.
    ///
    /// Returns an [`Svd`] with `u : codomain <- W`, `s : W <- W'` and
    /// `vh : W' <- domain`.
    ///
    /// `s` is compact when the constructed row and column bond legs coincide
    /// exactly, its spectrum covers every bond sector, and the rank-(1,1)
    /// compact layout is admitted. Otherwise `s` is dense and may be
    /// rectangular. The direct owned compact-diagonal input route also avoids
    /// dense input materialization and a solver call. Use
    /// [`Self::materialize`] when a dense singular-value buffer is required.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    pub(super) fn svd_full_multiplicity_free(&self) -> Result<Svd<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                // A diagonal input has square sectors, so compact and full
                // factor spaces coincide, including their nondual bond W.
                let (u, vh, mut spectrum) =
                    tenet_matrixalgebra::seam::svd_compact_diagonal_factors_dyn(
                        &body.space,
                        spectrum,
                    )?;
                return Ok(Svd {
                    u: self.wrap_bound_factor(u),
                    s: self.diagonal_factor(&mut spectrum, D::from_real)?,
                    vh: self.wrap_bound_factor(vh),
                });
            }
        }
        let mut dense = self.runtime.lease_dense();
        let factors = match &self.repr {
            TypedTensorRepr::Adjoint(view) => {
                tenet_matrixalgebra::seam::svd_full_adjoint_factors_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
                )?
            }
            TypedTensorRepr::Owned(_) => {
                let (bound_space, bound_payload) = self.bound_payload()?;
                tenet_matrixalgebra::seam::svd_full_factors_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
                )?
            }
        };
        let (u, vh, mut spectrum, row_dimensions, col_dimensions) = factors.into_parts();
        if full_svd_compact_bond(&u, &vh, &spectrum) {
            let space = tenet_matrixalgebra::seam::diagonal_bond_bound_space_like(
                self.logical_space(),
                &spectrum,
            )?;
            if full_svd_compact_layout(&space, &spectrum) {
                return Ok(Svd {
                    u: self.wrap_bound_factor(u),
                    s: diagonal_factor_on_bound(&self.runtime, space, &mut spectrum, D::from_real),
                    vh: self.wrap_bound_factor(vh),
                });
            }
        }
        let s = tenet_matrixalgebra::seam::rectangular_diagonal_bond_tensor(
            self.logical_space(),
            &spectrum,
            &row_dimensions,
            &col_dimensions,
        )?;
        Ok(Svd {
            u: self.wrap_bound_factor(u),
            s: self.wrap_bound_factor(s),
            vh: self.wrap_bound_factor(vh),
        })
    }
}

impl<R, D> TypedTensorSvdDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn svd_compact(tensor: &TensorMap<R, D>) -> Result<Svd<TensorMap<R, D>>, Error> {
        tensor.svd_compact_multiplicity_free()
    }

    fn svd_full(tensor: &TensorMap<R, D>) -> Result<Svd<TensorMap<R, D>>, Error> {
        tensor.svd_full_multiplicity_free()
    }
}

impl<R, D> TypedTensorSvdDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn svd_compact(
        tensor: &TensorMap<R, D>,
    ) -> Result<Svd<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.svd_compact_checked_generic()
    }

    fn svd_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Svd<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.svd_full_checked_generic()
    }
}
