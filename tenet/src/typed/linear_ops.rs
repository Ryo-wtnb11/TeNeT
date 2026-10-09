use super::*;

// The `re`/`im` gates are law checks (`re(t) + i·im(t)` rebuilds `t`).
// TensorKit's real-input
// branches (`real(t) = t`, `imag(t) = zerovector(t)` for a real scalartype)
// are statically unrepresentable here: these methods exist on the `Complex64`
// impl only, and `to_c64().re()` covers the round trip.
impl<R> TensorMap<R, num_complex::Complex64> {
    /// The element-wise real part, as an f64 tensor map on the same spaces
    /// (TensorKit `Base.real`: blockwise element-wise, result scalartype
    /// real).
    ///
    /// A compact spectrum maps spectrum-to-spectrum and stays compact. A cold
    /// lazy adjoint uses an operation-local payload for the `O(stored_len)`
    /// output; the logical space is shared.
    pub fn re(&self) -> TensorMap<R, f64> {
        self.map_parts(|value| value.re)
    }

    /// The element-wise imaginary part, as an f64 tensor map on the same
    /// spaces (TensorKit `Base.imag`).
    ///
    /// A compact spectrum maps spectrum-to-spectrum and stays compact. A cold
    /// lazy adjoint uses an operation-local payload for the `O(stored_len)`
    /// output; the logical space is shared.
    pub fn im(&self) -> TensorMap<R, f64> {
        self.map_parts(|value| value.im)
    }

    /// The shared owned-input route of [`Self::re`] / [`Self::im`].
    fn map_parts(&self, part: impl Fn(num_complex::Complex64) -> f64) -> TensorMap<R, f64> {
        let materialized = self
            .materialized_tensor_uncached()
            .expect("a pre-admitted typed adjoint must materialize");
        let source = materialized
            .owned_body()
            .expect("uncached materialization is owned");
        let body = match source.data.as_ref() {
            TypedData::Dense(data) => TypedTensorBody::dense(
                source.space.clone(),
                data.iter().map(|&value| part(value)).collect(),
            ),
            TypedData::Diagonal(spectrum) => {
                TypedTensorBody::diagonal(source.space.clone(), map_spectrum_dtype(spectrum, part))
            }
        };
        TensorMap {
            runtime: self.runtime.clone(),
            repr: owned_repr(body),
        }
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedAdjointSpace<R>,
    D: TensorScalar,
{
    /// `alpha * self + beta * other`, the one body of [`Self::axpby`] for
    /// every mode.
    fn axpby_host(&self, alpha: D, other: &Self, beta: D) -> Result<Self, Error> {
        // Runtime first, exactly as `contract` does: crossing runtimes is a
        // trust-boundary violation rather than an algebra error.
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let _host_pool = self.runtime.enter_host_pool();
        // `DynamicFusionMapSpace: PartialEq` covers the hom space, the
        // codomain/domain split and the block structure, which is exactly what
        // makes the zipped element-wise combination below meaningful.
        if self.logical_space().space() != other.logical_space().space() {
            return Err(Error::from(tenet_tensors::OperationError::SpaceMismatch {
                message: "tensors live on different spaces or block layouts",
            }));
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            || matches!(&other.repr, TypedTensorRepr::Adjoint(_))
        {
            if let (Some(spectrum), TypedTensorRepr::Adjoint(_)) = (self.spectrum(), &other.repr) {
                let (operand, dense) = other.fusion_operand_and_data();
                let mut data = tenet_tensors::oriented_fusion_add_owned(
                    self.logical_space().space().structure(),
                    operand,
                    &dense,
                    operand,
                    &dense,
                    beta,
                    D::from_real(0.0),
                )?;
                add_spectrum_into(self.logical_space().space(), &mut data, spectrum, alpha)?;
                return Ok(self.with_data(data));
            }
            if let (TypedTensorRepr::Adjoint(_), Some(spectrum)) = (&self.repr, other.spectrum()) {
                let (operand, dense) = self.fusion_operand_and_data();
                let mut data = tenet_tensors::oriented_fusion_add_owned(
                    self.logical_space().space().structure(),
                    operand,
                    &dense,
                    operand,
                    &dense,
                    alpha,
                    D::from_real(0.0),
                )?;
                add_spectrum_into(self.logical_space().space(), &mut data, spectrum, beta)?;
                return Ok(self.with_data(data));
            }
            let (lhs, lhs_data) = self.fusion_operand_and_data();
            let (rhs, rhs_data) = other.fusion_operand_and_data();
            let data = tenet_tensors::oriented_fusion_add_owned(
                self.logical_space().space().structure(),
                lhs,
                &lhs_data,
                rhs,
                &rhs_data,
                alpha,
                beta,
            )?;
            return Ok(self.with_data(data));
        }
        match (self.spectrum(), other.spectrum()) {
            // Two spectra on one bond space: the sum is diagonal too.
            (Some(lhs), Some(rhs)) => {
                if lhs.len() != rhs.len() {
                    return Err(spectra_disagree());
                }
                let sum = lhs
                    .iter()
                    .zip(rhs)
                    .map(|(left, right)| {
                        if left.sector != right.sector || left.values.len() != right.values.len() {
                            return Err(spectra_disagree());
                        }
                        Ok(tenet_matrixalgebra::SectorSpectrum {
                            sector: left.sector,
                            values: left
                                .values
                                .iter()
                                .zip(&right.values)
                                .map(|(&x, &y)| scale_value(x, alpha) + scale_value(y, beta))
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                return Ok(self.with_spectrum(sum));
            }
            // Mixed: the result is dense, but the *diagonal operand* is never
            // materialized to get there — the one O(n²) buffer this needs is
            // the owned result, which scatters the spectrum onto its own
            // diagonal. Materializing first would allocate a second.
            (Some(diagonal), None) => {
                return Ok(self.with_data(scatter_spectrum(
                    self.logical_space().space(),
                    other
                        .owned_body()
                        .expect("owned add input")
                        .materialized_dense_data()
                        .as_ref(),
                    beta,
                    diagonal,
                    alpha,
                )?))
            }
            (None, Some(diagonal)) => {
                return Ok(self.with_data(scatter_spectrum(
                    self.logical_space().space(),
                    self.owned_body()
                        .expect("owned add input")
                        .materialized_dense_data()
                        .as_ref(),
                    alpha,
                    diagonal,
                    beta,
                )?))
            }
            (None, None) => {}
        }
        Ok(self.with_data(
            self.owned_body()
                .expect("owned add input")
                .materialized_dense_data()
                .as_ref()
                .iter()
                .zip(
                    other
                        .owned_body()
                        .expect("owned add input")
                        .materialized_dense_data()
                        .as_ref(),
                )
                .map(|(&x, &y)| scale_value(x, alpha) + scale_value(y, beta))
                .collect(),
        ))
    }

    /// Returns the host-side linear combination
    /// `x.axpby(alpha, &y, beta) == alpha * x + beta * y` on the operands'
    /// common tensor space, where `x` is the receiver.
    ///
    /// The name is BLAS `axpby`: each coefficient sits next to the operand it
    /// multiplies. TeNeT has no `add` with coefficients because
    /// VectorInterface's `add(y, x, α, β)` computes `β·y + α·x`, binding the
    /// coefficients the other way round; a ported `add` call would compile and
    /// silently swap them.
    ///
    /// Both operands must share a runtime and exactly the same tensor space and
    /// block layout. Two compact diagonal inputs stay compact. A mixed
    /// compact/dense pair allocates only the dense result, and lazy inputs are
    /// read in their logical orientation without filling their caches.
    ///
    /// Returns [`Error::RuntimeMismatch`] or `SpaceMismatch` (unequal spaces or block layouts) before
    /// producing a result. See [`Self::norm`] for a runnable example.
    ///
    /// ```compile_fail
    /// use tenet::sector::U1FusionRule;
    /// use tenet::typed::TensorMap;
    ///
    /// fn removed(x: &TensorMap<U1FusionRule, f64>) {
    ///     let _ = x.add(x, 1.0, 2.0);
    /// }
    /// ```
    pub fn axpby<'a>(
        &self,
        alpha: D,
        y: impl Into<TensorRef<'a, R, D>>,
        beta: D,
    ) -> Result<Self, TypedFacadeError<R>> {
        let y = y.into().operand()?;
        let y = &*y;
        self.axpby_host(alpha, y, beta).map_err(Into::into)
    }

    /// Returns `factor * self` in host storage.
    ///
    /// The operation is infallible because `factor` already has the payload
    /// type `D`. Compact diagonal storage stays compact. A dense lazy adjoint
    /// remains lazy by scaling its parent with the conjugated factor, so no
    /// receiver materialization is cached.
    pub fn scale(&self, factor: D) -> Self {
        if let Some(spectrum) = self.spectrum() {
            return self.with_spectrum(
                spectrum
                    .iter()
                    .map(|entry| tenet_matrixalgebra::SectorSpectrum {
                        sector: entry.sector,
                        values: entry
                            .values
                            .iter()
                            .map(|&value| scale_value(value, factor))
                            .collect(),
                    })
                    .collect(),
            );
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent
                .scale(FactorScalar::adjoint(factor))
                .adjoint()
                .expect("scaling a pre-admitted adjoint must preserve its layout");
        }
        self.with_data(
            self.owned_body()
                .expect("owned scale input")
                .materialized_dense_data()
                .as_ref()
                .iter()
                .map(|&value| scale_value(value, factor))
                .collect(),
        )
    }

    /// `self = factor * self` in place, VectorInterface `scale!(t, α)`, on
    /// the receiver's own storage: a dense payload element by element, a
    /// compact diagonal on its stored values (it stays compact, TensorKit's
    /// `scale!` over a `DiagonalTensorMap`'s blocks), and a lazy adjoint on
    /// its parent with the conjugated factor (it stays lazy, as
    /// [`Self::scale`] does). Nothing is allocated.
    ///
    /// # Errors
    ///
    /// [`Error::DestinationShared`] when the receiver shares its storage
    /// with another handle (a shallow `Clone`, or the parent of a lazy
    /// adjoint): writing it would change that handle too, and replacing it
    /// would silently allocate, so the call does neither and leaves the
    /// receiver unchanged. The scaling itself cannot fail.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let mut t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 1)?;
    /// let expected = t.scale(2.0);
    /// t.scale_assign(2.0)?;
    /// assert_eq!(t.dense_data()?, expected.dense_data()?);
    ///
    /// let shared = t.clone();
    /// assert_eq!(t.scale_assign(2.0), Err(Error::DestinationShared));
    /// drop(shared);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn scale_assign(&mut self, factor: D) -> Result<(), TypedFacadeError<R>> {
        let (body, factor) = match &mut self.repr {
            TypedTensorRepr::Owned(body) => (Arc::get_mut(body), factor),
            TypedTensorRepr::Adjoint(view) => (
                Arc::get_mut(view).and_then(|view| Arc::get_mut(&mut view.parent)),
                FactorScalar::adjoint(factor),
            ),
        };
        let data = body
            .and_then(|body| Arc::get_mut(&mut body.data))
            .ok_or(Error::DestinationShared)?;
        let scale = |value: &mut D| *value = scale_value(*value, factor);
        match data {
            TypedData::Dense(data) => data.iter_mut().for_each(scale),
            TypedData::Diagonal(spectrum) => spectrum
                .iter_mut()
                .flat_map(|entry| entry.values.iter_mut())
                .for_each(scale),
        }
        Ok(())
    }

    /// `destination = alpha * self + beta * destination`: BLAS `axpby`,
    /// TensorKit `add!(ty, tx, α, β)`, in place on the destination's dense
    /// host payload.
    ///
    /// Each element is `scale(destination, beta) + scale(self, alpha)` with
    /// VectorInterface's strong zero, in one pass: `beta == 0` does not let a
    /// NaN in `destination` through, and `alpha == 0` does not read `self`.
    /// A lazy-adjoint `self` is read in its logical orientation without
    /// filling its cache; a compact diagonal `self` is added onto the
    /// diagonal after `destination` is scaled.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`]; `SpaceMismatch` when the two tensors live
    /// on different spaces or block layouts; [`Error::InvalidArgument`] when `destination`
    /// is not owned dense host storage, or when it aliases `self`;
    /// [`Error::DestinationShared`] when `destination` shares its storage with
    /// a clone.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let x: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 1)?;
    /// let mut y = x.zeros_like();
    /// x.axpby_into(&mut y, 2.0, 0.0)?;
    /// assert_eq!(y.dense_data()?, x.scale(2.0).dense_data()?);
    ///
    /// // A clone shares the payload, so it is not a valid destination.
    /// let shared = y.clone();
    /// assert_eq!(x.axpby_into(&mut y, 1.0, 1.0), Err(Error::DestinationShared));
    /// drop(shared);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn axpby_into(
        &self,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TypedFacadeError<R>> {
        host_axpby_into(self, destination, alpha, beta).map_err(TypedFacadeError::<R>::from)
    }
}

/// The destination's space against the operation result's, as TensorKit
/// compares `space(tdst)`; the same error as [`unique_dense_destination`]'s
/// space check, for routes that compare the space before their other checks.
pub(super) fn require_destination_space(
    destination: &DynamicFusionMapSpace,
    expected: &DynamicFusionMapSpace,
) -> Result<(), Error> {
    if destination != expected {
        return Err(Error::from(tenet_tensors::OperationError::SpaceMismatch {
            message: "destination fusion space or block layout does not match the operation result",
        }));
    }
    Ok(())
}

/// The destination checks a `*_into` shares on every placement, in one
/// order: owned dense storage, no alias of the input payload, the result's
/// space and block layout, the exact length, then unique ownership.
/// `storage` names the placement in the messages ("host", "CUDA").
pub(super) fn unique_dense_destination<'d, R, D, S>(
    destination: &'d TensorMap<R, D, S>,
    input: &Arc<TypedData<D, S>>,
    expected: &DynamicFusionMapSpace,
    storage: &str,
) -> Result<&'d S, Error>
where
    S: TensorStorage<D>,
{
    let (body, data) = match &destination.repr {
        TypedTensorRepr::Owned(body) => match body.data.as_ref() {
            TypedData::Dense(data) => (body, data),
            TypedData::Diagonal(_) => {
                return Err(Error::InvalidArgument(format!(
                    "destination must use ordinary dense {storage} storage"
                )))
            }
        },
        TypedTensorRepr::Adjoint(_) => {
            return Err(Error::InvalidArgument(format!(
                "destination must use ordinary dense {storage} storage"
            )))
        }
    };
    if Arc::ptr_eq(&body.data, input) {
        return Err(Error::InvalidArgument(
            "destination storage must not alias an input".to_string(),
        ));
    }
    require_destination_space(body.space.space(), expected)?;
    let required = expected.required_len()?;
    let actual = data.len();
    if actual != required {
        return Err(Error::InvalidArgument(format!(
            "destination storage length {actual} does not match required length {required}"
        )));
    }
    if Arc::strong_count(body) != 1 || Arc::strong_count(&body.data) != 1 {
        return Err(Error::DestinationShared);
    }
    Ok(data)
}

/// The host `axpby_into` body, shared with the empty-pair `trace_pairs_into`.
pub(super) fn host_axpby_into<R, D>(
    x: &TensorMap<R, D>,
    destination: &mut TensorMap<R, D>,
    alpha: D,
    beta: D,
) -> Result<(), Error>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    if !x.runtime.same_runtime(&destination.runtime) {
        return Err(Error::RuntimeMismatch);
    }
    if x.logical_space().space() != destination.logical_space().space() {
        return Err(Error::from(tenet_tensors::OperationError::SpaceMismatch {
            message: "tensors live on different spaces or block layouts",
        }));
    }
    unique_dense_destination(
        destination,
        &x.storage_body().data,
        x.logical_space().space(),
        "host",
    )?;
    let _host_pool = x.runtime.enter_host_pool();
    let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
        return Err(internal_layout_error("ordinary destination checked above"));
    };
    let destination_data = Arc::get_mut(destination_body)
        .and_then(|body| Arc::get_mut(&mut body.data))
        .ok_or_else(|| internal_layout_error("unique destination checked above"))?;
    let TypedData::Dense(destination_data) = destination_data else {
        return Err(internal_layout_error("dense destination checked above"));
    };
    match &x.repr {
        TypedTensorRepr::Owned(body) => match body.data.as_ref() {
            TypedData::Dense(source) => {
                if source.len() != destination_data.len() {
                    return Err(Error::InvalidArgument(
                        "tensors have different dense payload lengths".to_string(),
                    ));
                }
                for (value, &source) in destination_data.iter_mut().zip(source) {
                    *value = scale_value(*value, beta) + scale_value(source, alpha);
                }
            }
            TypedData::Diagonal(spectrum) => {
                for value in destination_data.iter_mut() {
                    *value = scale_value(*value, beta);
                }
                add_spectrum_into(x.logical_space().space(), destination_data, spectrum, alpha)?;
            }
        },
        TypedTensorRepr::Adjoint(_) => {
            let (operand, source) = x.fusion_operand_and_data();
            tenet_tensors::oriented_fusion_axpby_into(
                x.logical_space().space().structure(),
                destination_data,
                operand,
                &source,
                alpha,
                beta,
            )?;
        }
    }
    Ok(())
}
