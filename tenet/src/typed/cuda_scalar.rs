use super::*;

#[cfg(feature = "cuda")]
/// Device transfer, arithmetic, reductions, and contraction over a
/// [`CudaStorage`] payload.
///
/// The payload dtype is the only degree of freedom: structural coefficients
/// stay real (`R::Scalar = f64`), and operand conjugation is carried as a GEMM
/// flag, never as a materialized conjugated buffer. All four payload dtypes are
/// admitted (#1336); see [`CudaPayload`].
///
/// Checked Generic providers deliberately have no device execution methods in
/// this leaf:
///
/// ```compile_fail
/// use tenet::sector::{CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols, TypedSectorAdmission};
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn no_checked_generic_cuda_operations<R>(
///     lhs: &TensorMap<R, f64, CudaStorage>,
///     rhs: &TensorMap<R, f64, CudaStorage>,
/// ) where
///     R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>
///         + CheckedGenericFusion
///         + CheckedGenericRigidSymbols<Scalar = f64>,
/// {
///     let _ = lhs.norm(2.0);
///     let _ = lhs.inner(rhs);
///     let _ = lhs.scale(2.0);
///     let _ = lhs.axpby(2.0, rhs, -3.0);
///     let _ = lhs.zeros_like();
/// }
/// ```
impl<R, D> TensorMap<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    fn cuda_axpby_owned(
        &self,
        required_len: usize,
        lhs: (&CudaStorage<D>, D),
        rhs: Option<(&CudaStorage<D>, D)>,
    ) -> Result<CudaStorage<D>, Error> {
        let (lhs, alpha) = lhs;
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        let expected = Placement::Cuda(cuda.device());
        Self::validate_cuda_owned_metadata(expected, lhs.placement(), required_len, lhs.len())?;
        if let Some((rhs, _)) = rhs {
            Self::validate_cuda_owned_metadata(expected, rhs.placement(), required_len, rhs.len())?;
        }

        // ponytail: #740 keeps these proven Host uploads until native device
        // allocation publishes cross-stream writes correctly and wins a bench.
        let coefficient_values = match rhs {
            Some((_, beta)) => vec![alpha, beta],
            None => vec![alpha],
        };
        // Coefficients stay data operands so the descriptor scales stay 1/0.
        // A zero coefficient skips its operand, VectorInterface's
        // `scale(x, 0) = zero(x) * 0` as on the Host (#1442): the output is
        // born zero. Arithmetic does not promise signed-zero bit parity
        // across storage backends.
        let coefficients = CudaStorage::upload_owned(cuda, coefficient_values)?;
        #[cfg(test)]
        observe_cuda_arithmetic(0, 1, 0);
        let mut output = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        #[cfg(test)]
        observe_cuda_arithmetic(1, 0, 0);
        if required_len != 0 && !alpha.is_zero() {
            cuda_gemm_region_into::<D>(
                cuda,
                &mut output.0,
                0,
                required_len,
                &lhs.0,
                0,
                required_len,
                &coefficients.0,
                0,
                1,
                required_len,
                1,
                1,
                D::from_real(1.0),
                D::from_real(0.0),
            )
            .map_err(|err| Error::from(tenet_tensors::OperationError::Dense(err)))?;
            #[cfg(test)]
            observe_cuda_arithmetic(0, 0, 1);
        }
        if let Some((rhs, _)) = rhs.filter(|(_, beta)| required_len != 0 && !beta.is_zero()) {
            cuda_gemm_region_into::<D>(
                cuda,
                &mut output.0,
                0,
                required_len,
                &rhs.0,
                0,
                required_len,
                &coefficients.0,
                1,
                1,
                required_len,
                1,
                1,
                D::from_real(1.0),
                D::from_real(1.0),
            )
            .map_err(|err| Error::from(tenet_tensors::OperationError::Dense(err)))?;
            #[cfg(test)]
            observe_cuda_arithmetic(0, 0, 1);
        }
        Ok(output)
    }

    fn cuda_zeros_owned(
        &self,
        required_len: usize,
        source: &CudaStorage<D>,
    ) -> Result<CudaStorage<D>, Error> {
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        Self::validate_cuda_owned_metadata(
            Placement::Cuda(cuda.device()),
            source.placement(),
            required_len,
            source.len(),
        )?;
        let output = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        #[cfg(test)]
        observe_cuda_arithmetic(1, 0, 0);
        Ok(output)
    }

    /// Fresh device result `factor * self` for owned storage; a lazy adjoint
    /// redirects algebraically through its canonical parent. A zero factor
    /// gives exact zeros whatever `self` holds, as on the Host (TensorKit's
    /// `scale(x, 0)`); signed-zero bits are backend-local.
    pub fn scale(&self, factor: D) -> Result<Self, Error> {
        let required_len = self.logical_space().space().required_len()?;
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            // `alpha * A^H == (conj(alpha) * A)^H`, matching the adjoint arm
            // of the Host `scale_multiplicity_free`.
            return parent.scale(FactorScalar::adjoint(factor))?.adjoint();
        }
        let source = self.direct_cuda_storage("scale")?;
        let output = self.cuda_axpby_owned(required_len, (source, factor), None)?;
        Ok(self.with_owned_cuda_storage(output))
    }

    /// Fresh device result `x.axpby(alpha, &y, beta) == alpha * x + beta * y`,
    /// named and ordered like the Host [`TensorMap::axpby`]. A zero coefficient
    /// drops its operand, NaN and Inf included, as on the Host (TensorKit's
    /// `add`); signed-zero bits are backend-local.
    ///
    /// ```compile_fail
    /// use tenet::sector::U1FusionRule;
    /// use tenet::typed::{CudaStorage, TensorMap};
    ///
    /// fn removed(x: &TensorMap<U1FusionRule, f64, CudaStorage<f64>>) {
    ///     let _ = x.add(x, 1.0, 2.0);
    /// }
    /// ```
    pub fn axpby<'a>(
        &self,
        alpha: D,
        y: impl Into<TensorRef<'a, R, D, CudaStorage<D>>>,
        beta: D,
    ) -> Result<Self, Error> {
        let y = y.into().operand()?;
        let y = &*y;
        let required_len = self.logical_space().space().required_len()?;
        if !self.runtime.same_runtime(&y.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if self.logical_space().space() != y.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        match (&self.repr, &y.repr) {
            (TypedTensorRepr::Adjoint(lhs), TypedTensorRepr::Adjoint(rhs)) => {
                let lhs = Self {
                    runtime: self.runtime.clone(),
                    repr: TypedTensorRepr::Owned(Arc::clone(&lhs.parent)),
                };
                let rhs = Self {
                    runtime: y.runtime.clone(),
                    repr: TypedTensorRepr::Owned(Arc::clone(&rhs.parent)),
                };
                // `(alpha A^H + beta B^H) == (conj(alpha) A + conj(beta) B)^H`,
                // so the folded coefficients are conjugated before the parents
                // are combined.
                return lhs
                    .axpby(
                        FactorScalar::adjoint(alpha),
                        &rhs,
                        FactorScalar::adjoint(beta),
                    )?
                    .adjoint();
            }
            (TypedTensorRepr::Adjoint(_), TypedTensorRepr::Owned(_))
            | (TypedTensorRepr::Owned(_), TypedTensorRepr::Adjoint(_)) => {
                return Err(Error::UnsupportedOnDevice(
                    "axpby does not support mixed owned/lazy CUDA operands".to_string(),
                ));
            }
            (TypedTensorRepr::Owned(_), TypedTensorRepr::Owned(_)) => {}
        }
        let lhs = self.direct_cuda_storage("axpby")?;
        let rhs = y.direct_cuda_storage("axpby")?;
        let output = self.cuda_axpby_owned(required_len, (lhs, alpha), Some((rhs, beta)))?;
        Ok(self.with_owned_cuda_storage(output))
    }

    /// `destination = alpha * self + beta * destination` on the device: the
    /// Host [`TensorMap::axpby_into`], BLAS `axpby`.
    ///
    /// One `dot_general` over the whole payload with `alpha` as its descriptor
    /// scale and `beta` in its epilogue: `beta = 0` never reads the
    /// destination, and a zero `alpha` does not read `self` (the destination
    /// becomes `beta * destination` through a zero-source move). Nothing is
    /// uploaded or allocated on a warm context.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`]; [`Error::InvalidArgument`] for different
    /// spaces or block layouts, a destination that is not owned dense CUDA
    /// storage, or one that aliases `self`; [`Error::UnsupportedOnDevice`] for
    /// a lazy-adjoint or compact `self`, as for the returning
    /// [`Self::axpby`] with mixed operands; [`Error::DestinationShared`];
    /// [`Error::PlacementMismatch`].
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn axpby_into(&self, destination: &mut Self, alpha: D, beta: D) -> Result<(), Error> {
        if !self.runtime.same_runtime(&destination.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if self.logical_space().space() != destination.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        let source = self.direct_cuda_storage("axpby_into")?;
        let destination_storage = unique_cuda_destination(
            destination,
            &self.storage_body().data,
            self.logical_space().space(),
        )?;
        let device = Placement::Cuda(self.runtime.cuda_device_ordinal_checked()?);
        if source.placement() != device || destination_storage.placement() != device {
            return Err(Error::PlacementMismatch);
        }
        let len = TensorStorage::len(source);
        let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
            return Err(internal_layout_error(
                "ordinary CUDA destination checked above",
            ));
        };
        let destination_data = Arc::get_mut(destination_body)
            .and_then(|body| Arc::get_mut(&mut body.data))
            .ok_or_else(|| internal_layout_error("unique CUDA destination checked above"))?;
        let TypedData::Dense(destination_data) = destination_data else {
            return Err(internal_layout_error(
                "dense CUDA destination checked above",
            ));
        };
        if len == 0 {
            return Ok(());
        }
        let dense_err = |err| Error::from(tenet_tensors::OperationError::Dense(err));
        let region = tenet_dense::CudaRegion::packed(&[len], 0).map_err(dense_err)?;
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        if alpha.is_zero() {
            return tenet_dense::cuda_region_scale::<D>(
                cuda,
                &mut destination_data.0,
                &region,
                beta,
            )
            .map_err(dense_err);
        }
        let beta = if beta.is_zero() {
            tenet_dense::CudaRegionBeta::Overwrite
        } else if beta == D::from_real(1.0) {
            tenet_dense::CudaRegionBeta::Accumulate
        } else {
            tenet_dense::CudaRegionBeta::Scale(beta)
        };
        tenet_dense::cuda_region_axpby::<D>(
            cuda,
            &source.0,
            &region,
            false,
            alpha,
            tenet_dense::CudaRegionCoefficient::One,
            beta,
            &mut destination_data.0,
            &region,
        )
        .map_err(dense_err)
    }

    /// Exact positive-zero device tensor, independent of source values.
    pub fn zeros_like(&self) -> Result<Self, Error> {
        let required_len = self.logical_space().space().required_len()?;
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent.zeros_like()?.adjoint();
        }
        let source = self.direct_cuda_storage("zeros_like")?;
        let output = self.cuda_zeros_owned(required_len, source)?;
        Ok(self.with_owned_cuda_storage(output))
    }

    /// Quantum-dimension-weighted Frobenius reduction over owned CUDA storage,
    /// **conjugate-linear in `lhs`** like the Host `coupled_region_inner`,
    /// returned in the wide `f64` lane before any narrowing to `D`.
    ///
    /// The conjugation is the GEMM operand flag `MatrixOp::Adjoint`: the left
    /// row-vector view of a coupled region has extent 1 along its first axis,
    /// so the adjoint strides `[len, 1]` address exactly the same elements as
    /// the identity strides `[1, 1]`, and the orientation only sets
    /// `lhs_conj`. For `D = f64` that flag is a no-op, so the reduction has
    /// the same operands and produces the same values as the unconjugated
    /// form it replaces.
    ///
    /// The device returns one scalar per coupled sector. Category weights stay
    /// with the tensor and are applied only after releasing the Runtime lock.
    ///
    /// # Accumulation contract
    ///
    /// Like the Host `coupled_region_inner`, which widens every entry, the sum
    /// accumulates in `f64` within *and* across coupled sectors at every
    /// payload dtype, so an `f32`/`Complex32` result is finite wherever the
    /// Host's is (#1344 for `norm`, #1383 for `inner`).
    ///
    /// * **Within one coupled sector** the sum runs on the device inside the
    ///   backend GEMM, in the dtype of its operands. Tenferro 0.7.1 offers no
    ///   widening dot or reduction: `dot_general` dispatches on the one dtype
    ///   shared by both operands and the destination (tenferro-gpu 0.7.1
    ///   `src/cubecl/gemm.rs` `dot_general_read_into_accum`), and
    ///   `DotGeneralAccumulation` carries no compute type; the BLAS-1
    ///   `vdot_read` is `cublasSdot_v2`/`cublasCdotc_v2` (`src/cubecl/blas1.rs`)
    ///   and the reduce kernels are generic over one float type
    ///   (`src/kernels/reduce/launch.rs` `launch_sum_float`). A
    ///   single-precision payload is therefore widened on the device first
    ///   ([`CudaStorage::widened`], Tenferro's `cast`; exact, since every
    ///   `f32` is an `f64`) and reduced in `f64`. That costs one device
    ///   allocation of twice the payload bytes and one elementwise pass per
    ///   *distinct* operand buffer — one for `norm` or `inner(a, a)`, two for
    ///   `inner(a, b)` — and no host transfer. `f64`/`Complex64` payloads take
    ///   the reduction directly and pay nothing extra.
    /// * **Across coupled sectors** the quantum-dimension weighting and the
    ///   final sum run on the host in `WideScalar::Wide`, the accumulator the
    ///   Host reductions use.
    ///
    /// Why not TensorKit's accumulation: TensorKit 0.17.1 `dot`
    /// (`src/tensors/linalg.jl`) sums `dim(c) * dot(b1, b2)` in the payload
    /// type, a `Float32` BLAS `sdot` per block, so it saturates exactly where
    /// the pre-#1383 device did. TeNeT's Host semantics are strictly wider,
    /// and the device must agree with its own Host.
    fn weighted_inner_cuda(
        &self,
        lhs: &CudaStorage<D>,
        rhs: &CudaStorage<D>,
    ) -> Result<num_complex::Complex64, Error> {
        match D::DTYPE {
            tenet_dense::DenseDType::F32 => self.weighted_inner_widened_cuda::<f64>(lhs, rhs),
            tenet_dense::DenseDType::C32 => {
                self.weighted_inner_widened_cuda::<num_complex::Complex64>(lhs, rhs)
            }
            _ => Ok(self.weighted_inner_cuda_wide(lhs, rhs)?.widen_complex()),
        }
    }

    /// [`Self::weighted_inner_cuda`] for a single-precision payload: both
    /// operands widened to the double-precision lane `E` on the device, one
    /// cast when they are the same buffer.
    ///
    /// Why widen rather than rescale by `norm(Inf)` (#1344): the reference
    /// only needs *some* overflow-safe sum — LinearAlgebra's `generic_norm2`
    /// accumulates a `Float32` sum in `Float64` and rescales by `normInf`
    /// only when `length * maxabs^2` leaves the payload range — and the Host
    /// already meets it by accumulating every square in `f64`, where an `f32`
    /// payload can neither overflow nor underflow. Widening reproduces that
    /// Host arithmetic exactly in one extra device pass; a scaled norm would
    /// need a max-abs pass, a host round trip for the scale and a scaled copy
    /// before the same reduction, and would still round each square in `f32`.
    ///
    /// The selected reference dispatches are narrower. Julia routes a strided
    /// `BlasFloat` array of `length >= NRM2_CUTOFF` to `BLAS.nrm2`
    /// (`LinearAlgebra/src/dense.jl:107`), a scaled sum in the payload
    /// precision. TensorKit 0.17.1 `norm` (`src/tensors/linalg.jl:277`) takes
    /// `norm(t.data)` for `UniqueFusion` and otherwise `_norm` (`:261`), which
    /// adds `dim(c) * norm(b)^2` in `float(real(scalartype))` — `Float32` for
    /// a `Float32` tensor — so it returns `Inf` once one weighted block square
    /// exceeds `f32::MAX` (the "big" SU(2) fixture of
    /// `typed_cuda_single_precision`). The Host and this device norm
    /// accumulate every square and the cross-sector combine in `f64`, which is
    /// strictly wider than that dispatch. For `inner` a rescale is not even
    /// available: a cancelling sum has no scale that keeps every product in
    /// range and the result accurate.
    fn weighted_inner_widened_cuda<E: CudaPayload>(
        &self,
        lhs: &CudaStorage<D>,
        rhs: &CudaStorage<D>,
    ) -> Result<num_complex::Complex64, Error> {
        let mut lease = self.runtime.lease_cuda()?;
        let wide_lhs = lhs.widened::<E>(&mut lease)?;
        let wide_rhs = if std::ptr::eq(lhs, rhs) {
            None
        } else {
            Some(rhs.widened::<E>(&mut lease)?)
        };
        drop(lease);
        let wide_rhs = wide_rhs.as_ref().unwrap_or(&wide_lhs);
        Ok(self
            .weighted_inner_cuda_wide(&wide_lhs, wide_rhs)?
            .widen_complex())
    }

    /// [`Self::weighted_inner_cuda`] over operands already in the lane `E`: the
    /// cross-sector total in `E`'s wide accumulator.
    fn weighted_inner_cuda_wide<E: CudaPayload>(
        &self,
        lhs: &CudaStorage<E>,
        rhs: &CudaStorage<E>,
    ) -> Result<<E as tenet_tensors::WideScalar>::Wide, Error> {
        let space = self.logical_space().space();
        let regions = sector_regions(space.structure(), space.nout())?;
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        validate_cuda_reduction_placement(
            Placement::Cuda(cuda.device()),
            lhs.placement(),
            rhs.placement(),
        )?;
        // ponytail: #740 keeps the proven host-zero upload until a native
        // allocation has correct cross-stream publication and measured value.
        let mut partials =
            CudaStorage::upload_owned(cuda, vec![E::from_real(0.0); regions.len().max(1)])?;
        {
            let mut gemm = CudaStorageGemm::new(cuda);
            for (index, region) in regions.iter().enumerate() {
                let len = region.rows() * region.cols();
                if len == 0 {
                    continue;
                }
                gemm.matmul_range_with_ops_scaled_into(
                    &mut partials,
                    index,
                    lhs,
                    region.range().start,
                    rhs,
                    region.range().start,
                    1,
                    len,
                    1,
                    tenet_dense::MatrixOp::Adjoint,
                    tenet_dense::MatrixOp::Identity,
                    E::from_real(1.0),
                )?;
            }
        }
        let values = download_cuda_reduction_partials(&partials, cuda)?;
        drop(lease);

        Ok(regions
            .iter()
            .zip(values)
            .map(|(region, value)| {
                value.widen()
                    * <<E as tenet_tensors::WideScalar>::Wide as FactorScalar>::from_real(
                        self.logical_space().provider().dim_scalar(region.coupled()),
                    )
            })
            .fold(
                <<E as tenet_tensors::WideScalar>::Wide as FactorScalar>::from_real(0.0),
                |total, term| total + term,
            ))
    }

    /// Quantum-dimension-weighted Frobenius norm of a device tensor: the
    /// `p == 2` arm of the Host [`TensorMap::norm`].
    ///
    /// Only `p == 2` has a device reduction. Any other valid `p` is
    /// [`Error::UnsupportedOnDevice`] rather than a hidden Host transfer; an
    /// invalid `p` is [`Error::InvalidArgument`], as on the Host.
    ///
    /// A lazy adjoint delegates to its canonical parent because this norm is
    /// adjoint invariant; no logical-adjoint payload is materialized.
    ///
    /// # Accumulation and range
    ///
    /// The norm accumulates in `f64` at every payload dtype, within and
    /// across coupled sectors, exactly like the Host `norm`: an
    /// `f32`/`Complex32` payload is widened on the device by one cast
    /// (one device allocation of twice the payload bytes and one elementwise
    /// pass) before the per-sector reduction, because Tenferro 0.7.1 offers no
    /// widening reduction. Its result is therefore finite wherever every entry
    /// is finite, and nonzero wherever some entry is, matching the Host within
    /// `f64` rounding (#1344). `f64`/`Complex64` payloads take the unwidened
    /// reduction and pay nothing extra. [`Self::inner`] shares this reduction.
    ///
    /// Unlike the Host `norm`, an `f64`/`Complex64` device sum is not
    /// rescaled: it returns `inf` once the squared norm exceeds `f64::MAX`
    /// (entries near `1e154` and above) and loses accuracy once it falls
    /// below `f64::MIN_POSITIVE` (norms near `1e-154` and below), where the
    /// Host returns the representable norm.
    pub fn norm(&self, p: f64) -> Result<f64, Error> {
        validate_norm_p(p)?;
        if p != 2.0 {
            return Err(Error::UnsupportedOnDevice(format!(
                "device norm supports only p = 2, got {p}"
            )));
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            return Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            }
            .norm(p);
        }
        let storage = self.direct_cuda_storage("norm")?;
        // `<t, t>` is real up to rounding; the norm is its real part's root,
        // matching the Host `norm_multiplicity_free`.
        Ok(self.weighted_inner_cuda(storage, storage)?.re.sqrt())
    }

    /// TensorKit `dot(x, y)`: the quantum-dimension-weighted Frobenius inner
    /// product with **`self` conjugated**, matching the Host
    /// `inner_multiplicity_free`. Lazy adjoints remain an explicit
    /// unsupported device scope.
    ///
    /// Accumulates like the Host `inner`: in `f64` within and across coupled
    /// sectors at every payload dtype, then narrows the total to `D` once
    /// (`D::from_complex64`, as the Host does). An `f32`/`Complex32` result is
    /// therefore finite exactly where the Host's is, including a cancelling
    /// sum whose individual products exceed `f32::MAX`. A single-precision
    /// call widens each distinct operand on the device first; the cost is
    /// documented on `weighted_inner_cuda` (#1383).
    #[doc(alias = "dot")]
    pub fn inner<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D, CudaStorage<D>>>,
    ) -> Result<D, Error> {
        let other = other.into().operand()?;
        let other = &*other;
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if self.logical_space().space() != other.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        let lhs = self.direct_cuda_storage("inner")?;
        let rhs = other.direct_cuda_storage("inner")?;
        self.weighted_inner_cuda(lhs, rhs).map(D::from_complex64)
    }
}
