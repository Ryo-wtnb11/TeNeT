use super::*;

mod api;
mod eig;
mod exp_solve;
mod inverse;
mod mode;
mod polar_null;
mod qr_lq;
mod svd;

use mode::{AdjointRule, FactorOp};
pub use mode::{
    FusionMode, TypedTensorEigDispatch, TypedTensorEighDispatch, TypedTensorExpDispatch,
    TypedTensorFullLqDispatch, TypedTensorFullQrDispatch, TypedTensorInvDispatch,
    TypedTensorLqDispatch, TypedTensorNullDispatch, TypedTensorPinvDispatch,
    TypedTensorPolarDispatch, TypedTensorQrDispatch, TypedTensorSolveDispatch,
    TypedTensorSvdDispatch,
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
    /// The one body of the values-only factorizations: the mode's adjoint
    /// rule picks the input storage, `stage` (the matrix-algebra entry for
    /// `R::Mode`) computes the raw spectra, and the coupled sectors are
    /// decoded into public labels and sorted by label.
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
        let local;
        let source = match &self.repr {
            TypedTensorRepr::Owned(body) => owned_factor_source(body)?,
            TypedTensorRepr::Adjoint(view) => match R::Mode::adjoint_rule(op) {
                AdjointRule::Reject => {
                    return Err(
                        Error::InvalidArgument(op.lazy_adjoint_refusal().to_string()).into(),
                    )
                }
                AdjointRule::Parent => tenet_matrixalgebra::seam::FactorSource::Dense(
                    BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
                        .map_err(Error::from)?,
                ),
                AdjointRule::Materialize => {
                    local = self.materialized_tensor_uncached()?;
                    owned_factor_source(
                        local
                            .owned_body()
                            .expect("materialize returns an owned body"),
                    )?
                }
            },
        };
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
