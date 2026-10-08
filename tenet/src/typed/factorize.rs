use super::*;

mod api;
mod exp_solve;
mod inverse;
mod mode;

pub use mode::FusionMode;
use mode::{AdjointRule, FactorOp};

/// The runtime as the seam's executor lease: the checked entries lease a
/// dense executor only for a dense stage, so a compact diagonal never takes
/// the runtime lock or mints an executor.
struct RuntimeDense<'a>(&'a Runtime);

impl tenet_matrixalgebra::seam::ExecutorLease for RuntimeDense<'_> {
    type Executor = dyn tenet_dense::DenseExecutor + Send;

    // Why not `DenseLease::dense`: its signature ties the trait object's
    // lifetime to the borrow, while both arms hold `'static` executors.
    fn run<T>(self, stage: impl FnOnce(&mut Self::Executor) -> T) -> T {
        let mut lease = self.0.lease_dense();
        let executor: &mut Self::Executor = match &mut lease {
            crate::runtime::DenseLease::Pooled { executor, .. } => &mut **executor
                .as_mut()
                .expect("dense lease always owns an executor"),
            crate::runtime::DenseLease::Locked { state, .. } => &mut *state.dense,
        };
        stage(executor)
    }
}

impl<R, D> TensorMap<R, D> {
    /// A seam factor in the storage its route produced.
    fn factor_output(&self, output: tenet_matrixalgebra::seam::FactorOutput<R, D>) -> Self {
        match output {
            tenet_matrixalgebra::seam::FactorOutput::Dense(factor) => {
                wrap_factor_on(&self.runtime, factor)
            }
            tenet_matrixalgebra::seam::FactorOutput::Diagonal { space, values } => {
                self.with_spectrum_on(space, values)
            }
        }
    }
}

/// Whether `op` redirects a lazy adjoint: a body whose only lazy-adjoint arm
/// is the redirect asserts it at compile time, so the rule table and the
/// body cannot disagree.
const fn redirects(op: FactorOp) -> bool {
    matches!(op.adjoint_rule(), AdjointRule::Redirect)
}

/// The seam source of an owned checked body: its compact diagonal, or its
/// dense payload bound to its space.
fn owned_factor_source<'a, R, D>(
    body: &'a TypedTensorBody<R, D>,
) -> Result<tenet_matrixalgebra::seam::FactorSource<'a, R, D>, Error> {
    Ok(match body.data.as_ref() {
        TypedData::Diagonal(spectrum) => tenet_matrixalgebra::seam::FactorSource::Diagonal {
            space: &body.space,
            spectrum,
        },
        TypedData::Dense(data) => tenet_matrixalgebra::seam::FactorSource::Dense(
            BoundDynamicTensorRef::try_new(&body.space, data)?,
        ),
    })
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R>,
    D: TensorScalar,
{
    /// The storage `op` reads: the receiver's own, or for a lazy adjoint
    /// whatever [`FactorOp::adjoint_rule`] selects. `local` holds an
    /// operation-local materialization.
    fn factor_input<'a>(
        &'a self,
        op: FactorOp,
        local: &'a mut Option<Self>,
    ) -> Result<tenet_matrixalgebra::seam::FactorSource<'a, R, D>, TypedFacadeError<R>> {
        let view = match &self.repr {
            TypedTensorRepr::Owned(body) => return Ok(owned_factor_source(body)?),
            TypedTensorRepr::Adjoint(view) => view,
        };
        match op.adjoint_rule() {
            AdjointRule::Parent => Ok(tenet_matrixalgebra::seam::FactorSource::Dense(
                BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
                    .map_err(Error::from)?,
            )),
            AdjointRule::Materialize => {
                let local = local.insert(self.materialized_tensor_uncached()?);
                Ok(owned_factor_source(
                    local
                        .owned_body()
                        .expect("materialize returns an owned body"),
                )?)
            }
            AdjointRule::Redirect | AdjointRule::AdjointSeam => Err(internal_layout_error(
                "a redirected factorization must not read its own lazy input",
            )
            .into()),
        }
    }

    /// `f(A)^H` for the lazy adjoint `self = A^H`, detached: the redirect
    /// of an operation with `f(A^H) = f(A)^H`. The parent is owned, so `f`
    /// never redirects again.
    fn adjoint_of_parent(
        &self,
        f: impl FnOnce(&Self) -> Result<Self, TypedFacadeError<R>>,
    ) -> Result<Self, TypedFacadeError<R>> {
        Ok(f(&self.adjoint()?)?
            .adjoint()?
            .materialized_tensor_uncached()?)
    }

    /// The one body of the values-only factorizations: `stage` (the
    /// matrix-algebra entry for `R::Mode`) computes the raw spectra, and the
    /// coupled sectors are decoded into public labels and sorted by label.
    fn factor_values<V>(
        &self,
        op: FactorOp,
        stage: impl FnOnce(
            RuntimeDense<'_>,
            tenet_matrixalgebra::seam::FactorSource<'_, R, D>,
        ) -> Result<
            Vec<tenet_matrixalgebra::SectorSpectrum<V>>,
            <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::Error,
        >,
    ) -> Result<Vec<SectorSpectrum<R::Sector, V>>, TypedFacadeError<R>> {
        let mut local = None;
        let source = self.factor_input(op, &mut local)?;
        let raw = stage(RuntimeDense(&self.runtime), source).map_err(R::Mode::map_factor_error)?;
        let provider = self.logical_space().provider();
        let mut decoded = raw
            .into_iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: R::Mode::decode_label(provider, entry.sector)?,
                    values: entry.values,
                })
            })
            .collect::<Result<Vec<_>, TypedFacadeError<R>>>()?;
        // Public label order, not the engine's opaque sector-id order.
        decoded.sort_by(|left, right| left.sector.cmp(&right.sector));
        Ok(decoded)
    }

    /// The one body of QR: `stage` factors the input storage, and each factor
    /// keeps the storage its route produced.
    fn factor_qr(
        &self,
        op: FactorOp,
        stage: impl FnOnce(
            RuntimeDense<'_>,
            tenet_matrixalgebra::seam::FactorSource<'_, R, D>,
        ) -> Result<
            Qr<tenet_matrixalgebra::seam::FactorOutput<R, D>>,
            <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::Error,
        >,
    ) -> Result<Qr<Self>, TypedFacadeError<R>> {
        let mut local = None;
        let source = self.factor_input(op, &mut local)?;
        let Qr { q, r } =
            stage(RuntimeDense(&self.runtime), source).map_err(R::Mode::map_factor_error)?;
        Ok(Qr {
            q: self.factor_output(q),
            r: self.factor_output(r),
        })
    }

    /// The one body of LQ. Under [`AdjointRule::Redirect`] a lazy adjoint is
    /// factored as `qr` of its parent with both factors adjointed back and
    /// detached.
    fn factor_lq(
        &self,
        op: FactorOp,
        qr: impl FnOnce(&Self) -> Result<Qr<Self>, TypedFacadeError<R>>,
        stage: impl FnOnce(
            RuntimeDense<'_>,
            tenet_matrixalgebra::seam::FactorSource<'_, R, D>,
        ) -> Result<
            Lq<tenet_matrixalgebra::seam::FactorOutput<R, D>>,
            <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::Error,
        >,
    ) -> Result<Lq<Self>, TypedFacadeError<R>>
    where
        R::Mode: TypedAdjointSpace<R>,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            && op.adjoint_rule() == AdjointRule::Redirect
        {
            let Qr { q, r } = qr(&self.adjoint()?)?;
            return Ok(Lq {
                l: r.adjoint()?.materialized_tensor_uncached()?,
                q: q.adjoint()?.materialized_tensor_uncached()?,
            });
        }
        let mut local = None;
        let source = self.factor_input(op, &mut local)?;
        let Lq { l, q } =
            stage(RuntimeDense(&self.runtime), source).map_err(R::Mode::map_factor_error)?;
        Ok(Lq {
            l: self.factor_output(l),
            q: self.factor_output(q),
        })
    }

    /// The one body of the null spaces. Under [`AdjointRule::Redirect`] a
    /// lazy adjoint's null space is the adjoint of `partner` (the opposite
    /// null space) of its parent, detached.
    fn factor_null(
        &self,
        op: FactorOp,
        partner: impl FnOnce(&Self) -> Result<Self, TypedFacadeError<R>>,
        stage: impl FnOnce(
            RuntimeDense<'_>,
            tenet_matrixalgebra::seam::FactorSource<'_, R, D>,
        ) -> Result<
            tenet_matrixalgebra::BoundDynFactor<R, D>,
            <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::Error,
        >,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        R::Mode: TypedAdjointSpace<R>,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            && op.adjoint_rule() == AdjointRule::Redirect
        {
            return self.adjoint_of_parent(partner);
        }
        let mut local = None;
        let source = self.factor_input(op, &mut local)?;
        let factor =
            stage(RuntimeDense(&self.runtime), source).map_err(R::Mode::map_factor_error)?;
        Ok(wrap_factor_on(&self.runtime, factor))
    }

    /// The one body of the left polar decomposition. Under
    /// [`AdjointRule::Redirect`] a lazy adjoint `A^H` is the adjoint-swapped
    /// right polar of its parent `A = p wh`: `A^H = wh^H p`, with `wh^H`
    /// detached. The parent's factors and a dense input's come from the
    /// mode's own stages (D5, #1752); a compact diagonal factors directly on
    /// its bond.
    fn factor_left_polar(&self) -> Result<LeftPolar<Self>, TypedFacadeError<R>>
    where
        D: FactorizationScalar,
    {
        match &self.repr {
            TypedTensorRepr::Adjoint(_) => {
                const { assert!(redirects(FactorOp::LeftPolar)) };
                let RightPolar { p, wh } = R::Mode::left_polar_adjoint(self)?;
                Ok(LeftPolar {
                    w: wh.adjoint()?.materialized_tensor_uncached()?,
                    p,
                })
            }
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Diagonal(spectrum) => {
                    let LeftPolar { w, p } = tenet_matrixalgebra::seam::left_polar_of_diagonal::<
                        R::Mode,
                        R,
                        D,
                    >(&body.space, spectrum)
                    .map_err(R::Mode::map_factor_error)?;
                    Ok(LeftPolar {
                        w: self.factor_output(w),
                        p: self.factor_output(p),
                    })
                }
                TypedData::Dense(_) => R::Mode::left_polar_dense(self),
            },
        }
    }

    /// The one body of the right polar decomposition; see
    /// [`Self::factor_left_polar`].
    fn factor_right_polar(&self) -> Result<RightPolar<Self>, TypedFacadeError<R>>
    where
        D: FactorizationScalar,
    {
        match &self.repr {
            TypedTensorRepr::Adjoint(_) => {
                const { assert!(redirects(FactorOp::RightPolar)) };
                let LeftPolar { w, p } = R::Mode::right_polar_adjoint(self)?;
                Ok(RightPolar {
                    p,
                    wh: w.adjoint()?.materialized_tensor_uncached()?,
                })
            }
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Diagonal(spectrum) => {
                    let RightPolar { p, wh } =
                        tenet_matrixalgebra::seam::right_polar_of_diagonal::<R::Mode, R, D>(
                            &body.space,
                            spectrum,
                        )
                        .map_err(R::Mode::map_factor_error)?;
                    Ok(RightPolar {
                        p: self.factor_output(p),
                        wh: self.factor_output(wh),
                    })
                }
                TypedData::Dense(_) => R::Mode::right_polar_dense(self),
            },
        }
    }

    /// The one body of the Hermitian eigendecomposition: `d` is a compact
    /// diagonal on the bond the mode chooses for the route (D3, #1994).
    fn factor_eigh_full(
        &self,
        hermitian_tol: HermitianTol,
    ) -> Result<Eigh<Self>, TypedFacadeError<R>>
    where
        D: FactorizationScalar,
    {
        let mut local = None;
        let source = self.factor_input(FactorOp::EighFull, &mut local)?;
        let (out, route) = tenet_matrixalgebra::seam::eigh_full_from_source::<R::Mode, _, _, _, _>(
            RuntimeDense(&self.runtime),
            source,
            hermitian_tol,
        )
        .map_err(R::Mode::map_factor_error)?;
        let (v, mut eigenvalues) = out.into_parts();
        let space = <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::spectrum_bond(
            self.logical_space(),
            route,
            &eigenvalues,
        )
        .map_err(R::Mode::map_factor_error)?;
        Ok(Eigh {
            d: diagonal_factor_on_bound(&self.runtime, space, &mut eigenvalues, D::from_real),
            v: wrap_factor_on(&self.runtime, v),
        })
    }

    /// The one body of the general eigendecomposition; complex factors at the
    /// payload's precision, `d` on the mode's bond as for
    /// [`Self::factor_eigh_full`].
    #[allow(clippy::type_complexity)]
    fn factor_eig_full(
        &self,
    ) -> Result<Eig<TensorMap<R, <D as FactorScalar>::Eig>>, TypedFacadeError<R>>
    where
        D: AdvancedLinalgScalar,
        <D as FactorScalar>::Eig: TensorScalar,
    {
        let mut local = None;
        let source = self.factor_input(FactorOp::EigFull, &mut local)?;
        let (out, route) = tenet_matrixalgebra::seam::eig_full_from_source::<R::Mode, _, _, _, _>(
            RuntimeDense(&self.runtime),
            source,
        )
        .map_err(R::Mode::map_factor_error)?;
        let (v, mut eigenvalues) = out.into_parts();
        let space = <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::spectrum_bond(
            self.logical_space(),
            route,
            &eigenvalues,
        )
        .map_err(R::Mode::map_factor_error)?;
        Ok(Eig {
            d: diagonal_factor_on_bound(
                &self.runtime,
                space,
                &mut eigenvalues,
                <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
            ),
            v: wrap_factor_on(&self.runtime, v),
        })
    }

    /// The SVD factors `op` computes, with their route. Under
    /// [`AdjointRule::AdjointSeam`] a lazy adjoint's dense parent is read in
    /// place by `adjoint_stage`.
    fn svd_factors<T>(
        &self,
        op: FactorOp,
        stage: impl FnOnce(
            RuntimeDense<'_>,
            tenet_matrixalgebra::seam::FactorSource<'_, R, D>,
        ) -> Result<
            (T, tenet_matrixalgebra::seam::FactorRoute),
            <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::Error,
        >,
        adjoint_stage: impl FnOnce(
            RuntimeDense<'_>,
            BoundDynamicTensorRef<'_, R, D>,
        ) -> Result<
            T,
            <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::Error,
        >,
    ) -> Result<(T, tenet_matrixalgebra::seam::FactorRoute), TypedFacadeError<R>> {
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            if op.adjoint_rule() == AdjointRule::AdjointSeam {
                let parent = BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
                    .map_err(Error::from)?;
                let factors = adjoint_stage(RuntimeDense(&self.runtime), parent)
                    .map_err(R::Mode::map_factor_error)?;
                // A lazy adjoint's parent is dense by construction, so the
                // spectrum bond follows the dense route (D3).
                return Ok((factors, tenet_matrixalgebra::seam::FactorRoute::Dense));
            }
        }
        let mut local = None;
        let source = self.factor_input(op, &mut local)?;
        stage(RuntimeDense(&self.runtime), source).map_err(R::Mode::map_factor_error)
    }

    /// The one body of compact SVD: `s` is a compact diagonal on the bond the
    /// mode chooses for the route (D3, #1994).
    fn factor_svd_compact(&self) -> Result<Svd<Self>, TypedFacadeError<R>>
    where
        D: FactorizationScalar,
    {
        let ((u, vh, mut spectrum), route) = self.svd_factors(
            FactorOp::SvdCompact,
            |lease, source| {
                tenet_matrixalgebra::seam::svd_compact_from_source::<R::Mode, _, _, _, _>(
                    lease, source,
                )
            },
            |lease, parent| {
                tenet_matrixalgebra::seam::svd_compact_adjoint_from_parent::<R::Mode, _, _, _, _>(
                    lease, parent,
                )
            },
        )?;
        let space = <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::spectrum_bond(
            self.logical_space(),
            route,
            &spectrum,
        )
        .map_err(R::Mode::map_factor_error)?;
        Ok(Svd {
            u: wrap_factor_on(&self.runtime, u),
            s: diagonal_factor_on_bound(&self.runtime, space, &mut spectrum, D::from_real),
            vh: wrap_factor_on(&self.runtime, vh),
        })
    }

    /// The one body of full SVD: `s` is compact only on the exact constructed
    /// bond with an admitted layout, and dense (possibly rectangular)
    /// otherwise.
    fn factor_svd_full(&self) -> Result<Svd<Self>, TypedFacadeError<R>>
    where
        D: FactorizationScalar,
    {
        if <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::SVD_FULL_DIAGONAL_IS_COMPACT
            && self.spectrum().is_some()
        {
            return self.factor_svd_compact();
        }
        let (factors, route) = self.svd_factors(
            FactorOp::SvdFull,
            |lease, source| {
                tenet_matrixalgebra::seam::svd_full_from_source::<R::Mode, _, _, _, _>(
                    lease, source,
                )
            },
            |lease, parent| {
                tenet_matrixalgebra::seam::svd_full_adjoint_from_parent::<R::Mode, _, _, _, _>(
                    lease, parent,
                )
            },
        )?;
        let (u, vh, mut spectrum, row_dimensions, col_dimensions) = factors.into_parts();
        if full_svd_compact_bond(&u, &vh, &spectrum) {
            let space = <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::spectrum_bond(
                self.logical_space(),
                route,
                &spectrum,
            )
            .map_err(R::Mode::map_factor_error)?;
            if full_svd_compact_layout(&space, &spectrum) {
                return Ok(Svd {
                    u: wrap_factor_on(&self.runtime, u),
                    s: diagonal_factor_on_bound(&self.runtime, space, &mut spectrum, D::from_real),
                    vh: wrap_factor_on(&self.runtime, vh),
                });
            }
        }
        let s = <R::Mode as tenet_matrixalgebra::seam::FactorMode<R>>::rectangular_spectrum_factor(
            self.logical_space(),
            &spectrum,
            &row_dimensions,
            &col_dimensions,
        )
        .map_err(R::Mode::map_factor_error)?;
        Ok(Svd {
            u: wrap_factor_on(&self.runtime, u),
            s: wrap_factor_on(&self.runtime, s),
            vh: wrap_factor_on(&self.runtime, vh),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// The bound space and dense payload of this owned tensor map; a compact
    /// diagonal is densified operation-locally.
    #[allow(clippy::type_complexity)]
    fn bound_payload(
        &self,
    ) -> Result<(&BoundDynamicFusionMapSpace<R>, std::borrow::Cow<'_, [D]>), Error> {
        let body = self.owned_body().ok_or_else(|| {
            internal_layout_error("factorization input must be owned after adjoint dispatch")
        })?;
        Ok((&body.space, body.materialized_dense_data()))
    }
}
