use super::*;

mod api;
mod eig;
mod exp_solve;
mod inverse;
mod mode;
mod polar_null;

use mode::{AdjointRule, FactorOp};
pub use mode::{
    FusionMode, TypedTensorEigDispatch, TypedTensorEighDispatch, TypedTensorExpDispatch,
    TypedTensorInvDispatch, TypedTensorNullDispatch, TypedTensorPinvDispatch,
    TypedTensorPolarDispatch, TypedTensorSolveDispatch,
};

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
    /// whatever the mode's adjoint rule (D1, #1755) selects. `local` holds an
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
        match R::Mode::adjoint_rule(op) {
            AdjointRule::Reject => {
                Err(Error::InvalidArgument(op.lazy_adjoint_refusal().to_string()).into())
            }
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
        R::Mode: TypedTensorAdjointDispatch<R, D>,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            && R::Mode::adjoint_rule(op) == AdjointRule::Redirect
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
            if R::Mode::adjoint_rule(op) == AdjointRule::AdjointSeam {
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
    /// Wraps one factor the matrix-algebra seam produced into a typed tensor
    /// map. `BoundDynFactor::into_parts` hands back exactly the pair
    /// [`TypedTensorBody`] stores, so there is nothing to validate here — the
    /// seam already certified the space against its own data.
    fn wrap_bound_factor(&self, factor: BoundDynFactor<R, D>) -> Self {
        wrap_factor_on(&self.runtime, factor)
    }

    /// Wraps a seam spectrum as a factor in compact diagonal storage: the bond
    /// space is derived from the spectrum itself, but the payload stays the
    /// `Σ_c k_c` values rather than the `Σ_c k_c²` block-diagonal buffer they
    /// would fill (TensorKit's `DiagonalTensorMap`).
    ///
    /// The spectrum is stored raw — engine [`crate::sector::SectorId`]s, values in
    /// the payload dtype `D`. Decoding belongs to the caller-facing spectrum
    /// fields, not to storage; a stored payload never leaves this module.
    ///
    /// Sorted by sector id first because the bond leg is built from this order.
    fn diagonal_factor<V: Copy>(
        &self,
        spectrum: &mut [tenet_matrixalgebra::SectorSpectrum<V>],
        to_scalar: impl Fn(V) -> D,
    ) -> Result<Self, Error> {
        diagonal_factor_on(&self.runtime, self.logical_space(), spectrum, to_scalar)
    }

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
