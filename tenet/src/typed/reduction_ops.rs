use super::*;

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// The single element of a rank-0 (scalar) tensor, e.g. the result of
    /// contracting every leg — TensorKit `scalar` (an empty payload reads
    /// as zero there too).
    ///
    /// Returns `D` directly: the value is the sum of the coupled payload. A
    /// lazy adjoint is materialized operation-locally; a rank-0 payload holds
    /// at most one value per coupled sector.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] on a tensor with legs.
    pub fn scalar(&self) -> Result<D, Error> {
        if self.rank() != 0 {
            return Err(Error::InvalidArgument(format!(
                "scalar() requires a rank-0 tensor, got rank {}",
                self.rank()
            )));
        }
        // A rank-0 payload holds at most one element; summing gives the empty
        // payload its zero for free.
        let materialized = self.materialized_tensor_uncached()?;
        Ok(materialized
            .owned_body()
            .expect("uncached materialization is owned")
            .materialized_dense_data()
            .as_ref()
            .iter()
            .fold(D::from_real(0.0), |acc, &value| acc + value))
    }
}

/// The quantum-dimension-weighted inner product of two compact spectra,
/// `Σ_c dim(c) * Σ_i conj(a_i) b_i` — [`coupled_region_inner`]'s reduction
/// with the zeros left out, since a bond space's dense form is zero off the
/// per-sector diagonal.
fn compact_inner<D, E>(
    lhs: &[tenet_matrixalgebra::SectorSpectrum<D>],
    rhs: &[tenet_matrixalgebra::SectorSpectrum<D>],
    mut weight_of: impl FnMut(SectorId) -> Result<f64, E>,
) -> Result<num_complex::Complex64, E>
where
    D: TensorScalar,
    E: From<Error>,
{
    if lhs.len() != rhs.len() {
        return Err(spectra_disagree().into());
    }
    let mut total = num_complex::Complex64::new(0.0, 0.0);
    for (left, right) in lhs.iter().zip(rhs) {
        if left.sector != right.sector || left.values.len() != right.values.len() {
            return Err(spectra_disagree().into());
        }
        // `D::Wide` for the same reason as `coupled_region_inner`: the
        // identity for the double pair, double precision for the single
        // one, so a compact `norm`/`inner` answers like its dense twin.
        let mut partial = D::Wide::from_real(0.0);
        for (&a, &b) in left.values.iter().zip(&right.values) {
            partial = partial + FactorScalar::adjoint(a.widen()) * b.widen();
        }
        total += partial.widen_complex() * weight_of(left.sector)?;
    }
    Ok(total)
}

/// The largest stored magnitude of a compact spectrum, as [`max_abs`].
fn spectrum_max_abs<D: TensorScalar>(spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>]) -> f64 {
    max_abs(
        spectrum
            .iter()
            .flat_map(|entry| entry.values.iter().copied()),
    )
}

/// `Σ_c dim(c) · Σ_k term(s_{c,k})` over a compact spectrum: the stored
/// diagonal of each coupled block, whose off-diagonal zeros add nothing to
/// any power sum with `p > 0`.
fn spectrum_weighted_sum<D, E>(
    spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    mut weight_of: impl FnMut(SectorId) -> Result<f64, E>,
    term: impl Fn(D) -> f64,
) -> Result<f64, E>
where
    D: TensorScalar,
{
    spectrum.iter().try_fold(0.0, |total, entry| {
        Ok(total
            + weight_of(entry.sector)? * entry.values.iter().map(|&value| term(value)).sum::<f64>())
    })
}

/// The dimension-weighted dense inner product `Σ_c dim(c) * <a_c, b_c>`, with
/// TensorKit's `UniqueFusion` specialization: every `dim(c) == 1` and the
/// coupled buffer is a padding-free concatenation of the per-sector blocks, so
/// the weighted per-block sum collapses to one whole-buffer conjugated dot
/// (`vectorinterface.jl:124`, `linalg.jl:277`).
fn dense_weighted_inner<D, E>(
    unique: bool,
    structure: &BlockStructure,
    nout: usize,
    a: &[D],
    b: &[D],
    weight_of: impl FnMut(SectorId) -> Result<f64, E>,
) -> Result<num_complex::Complex64, E>
where
    D: TensorScalar,
    E: From<Error>,
{
    if unique {
        let mut total = D::Wide::from_real(0.0);
        for (&ai, &bi) in a.iter().zip(b) {
            total = total + FactorScalar::adjoint(ai.widen()) * bi.widen();
        }
        return Ok(total.widen_complex());
    }
    coupled_region_inner(structure, nout, a, b, weight_of)
}

/// `dim(c)` through the mode, evaluated in place with a one-entry run cache:
/// a canonical structure lists each coupled sector as one contiguous run of
/// blocks, so this asks once per coupled sector with no allocation and no
/// hashing.
fn run_cached_dim<R>(provider: &R) -> impl FnMut(SectorId) -> Result<f64, TypedFacadeError<R>> + '_
where
    R: TypedSectorAdmission,
    R::Mode: TypedSpaceModeDispatch<R>,
{
    let mut last: Option<(SectorId, f64)> = None;
    move |sector| match last {
        Some((cached, weight)) if cached == sector => Ok(weight),
        _ => {
            let weight = <R::Mode as TypedSpaceModeDispatch<R>>::dim(provider, sector)?;
            last = Some((sector, weight));
            Ok(weight)
        }
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedSpaceModeDispatch<R>,
    D: TensorScalar,
{
    /// Mixed compact/dense inner over the stored diagonal only. A compact
    /// operand defines structural zeros off diagonal, so those dense entries
    /// are not part of this reduction (including non-finite values).
    fn compact_dense_inner(
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
        dense: &Self,
        compact_is_lhs: bool,
    ) -> Result<num_complex::Complex64, TypedFacadeError<R>> {
        let logical_structure = dense.logical_space().space().structure();
        let (data_structure, data, dense_is_adjoint) = match &dense.repr {
            TypedTensorRepr::Owned(body) => {
                let TypedData::Dense(data) = body.data.as_ref() else {
                    unreachable!("mixed compact/dense dispatch requires a dense operand")
                };
                (body.space.space().structure(), data.as_slice(), false)
            }
            TypedTensorRepr::Adjoint(view) => (
                view.parent.space.space().structure(),
                view.parent_data(),
                true,
            ),
        };
        if logical_structure.block_count() != spectrum.len()
            || data_structure.required_len().map_err(Error::from)? != data.len()
        {
            return Err(spectra_disagree().into());
        }
        // A compact operand proves a rank-(1,1) bond endomorphism. Its equal
        // codomain and domain make the adjoint parent's admitted structure
        // content-identical to the logical structure, including block order.
        // Compare that contract once in O(G), then traverse both in lockstep;
        // per-block indexed lookup would make this O(G log G).
        if dense_is_adjoint && logical_structure != data_structure {
            return Err(internal_layout_error(
                "compact/dense inner adjoint parent has a different bond layout",
            )
            .into());
        }

        let provider = dense.logical_space().provider();
        let mut total = num_complex::Complex64::new(0.0, 0.0);
        for (index, entry) in spectrum.iter().enumerate() {
            let logical_block = logical_structure.block(index).map_err(Error::from)?;
            let Some(pair) = logical_block.key().as_fusion_tree_pair() else {
                return Err(spectra_disagree().into());
            };
            if pair.codomain_tree().coupled() != entry.sector
                || logical_block.shape().len() != 2
                || logical_block.shape()[0] != logical_block.shape()[1]
                || logical_block.shape()[0] != entry.values.len()
            {
                return Err(spectra_disagree().into());
            }
            let data_block = data_structure.block(index).map_err(Error::from)?;
            if data_block.shape().len() != 2
                || data_block.shape()[0] != data_block.shape()[1]
                || data_block.shape()[0] != entry.values.len()
            {
                return Err(spectra_disagree().into());
            }
            let stride = data_block.strides()[0] + data_block.strides()[1];
            let mut partial = D::Wide::from_real(0.0);
            for (i, &compact_value) in entry.values.iter().enumerate() {
                let dense_value = *data.get(data_block.offset() + i * stride).ok_or_else(|| {
                    internal_layout_error("compact/dense inner diagonal exceeds scalar storage")
                })?;
                let compact_value = compact_value.widen();
                let dense_value = dense_value.widen();
                partial = partial
                    + match (compact_is_lhs, dense_is_adjoint) {
                        (true, false) => FactorScalar::adjoint(compact_value) * dense_value,
                        (true, true) => {
                            FactorScalar::adjoint(compact_value)
                                * FactorScalar::adjoint(dense_value)
                        }
                        (false, false) => FactorScalar::adjoint(dense_value) * compact_value,
                        (false, true) => dense_value * compact_value,
                    };
            }
            total += partial.widen_complex()
                * <R::Mode as TypedSpaceModeDispatch<R>>::dim(provider, entry.sector)?;
        }
        Ok(total)
    }

    /// The parent of a lazy adjoint, as an owned tensor.
    fn adjoint_parent(view: &Arc<TypedAdjointView<R, D>>, runtime: &Runtime) -> Self {
        Self {
            runtime: runtime.clone(),
            repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
        }
    }

    /// The `p == 2` arm of [`Self::norm`]: TensorKit's Frobenius norm
    /// weighted by the coupled sectors' quantum dimensions,
    /// `norm(t)^2 = Σ_c dim(c) * |block_c|^2`.
    fn frobenius_norm(&self) -> Result<f64, TypedFacadeError<R>> {
        let provider = self.logical_space().provider();
        if let Some(spectrum) = self.spectrum() {
            return rescaled_power_norm(
                compact_inner(spectrum, spectrum, run_cached_dim(provider))?.re,
                2.0,
                || spectrum_max_abs(spectrum),
                |max| {
                    spectrum_weighted_sum(spectrum, run_cached_dim(provider), |value| {
                        scaled_power(value, max, 2.0)
                    })
                },
            );
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            return Self::adjoint_parent(view, &self.runtime).frobenius_norm();
        }
        let payload = self
            .owned_body()
            .expect("owned norm input")
            .materialized_dense_data();
        let data: &[D] = &payload;
        let structure = self.logical_space().space().structure();
        let nout = self.logical_space().space().nout();
        let unique = <R::Mode as TypedTensorModeDispatch<R>>::fusion_style(provider)
            == tenet_core::FusionStyleKind::Unique;
        // Why not `self.inner(self)`: it narrows the wide sum to `D`, which
        // for `f32`/`Complex32` rounds `|t|²` to single precision and can
        // leave it subnormal, above the rescaling threshold.
        rescaled_power_norm(
            dense_weighted_inner(
                unique,
                structure,
                nout,
                data,
                data,
                run_cached_dim(provider),
            )?
            .re,
            2.0,
            || max_abs(data.iter().copied()),
            |max| {
                coupled_region_weighted_sum(
                    structure,
                    nout,
                    data,
                    run_cached_dim(provider),
                    |value| scaled_power(value, max, 2.0),
                )
            },
        )
    }

    /// The `p == Inf` arm of [`Self::norm`]: the largest stored magnitude,
    /// not dimension weighted, NaN-propagating, `+0.0` without entries.
    fn max_norm(&self) -> f64 {
        if let Some(spectrum) = self.spectrum() {
            return spectrum_max_abs(spectrum);
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            return max_abs(view.parent_data().iter().copied());
        }
        max_abs(
            self.owned_body()
                .expect("owned norm input")
                .materialized_dense_data()
                .as_ref()
                .iter()
                .copied(),
        )
    }

    /// The finite `p != 2` arm of [`Self::norm`].
    fn power_norm(&self, p: f64) -> Result<f64, TypedFacadeError<R>> {
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            return Self::adjoint_parent(view, &self.runtime).power_norm(p);
        }
        let provider = self.logical_space().provider();
        let power = |value: D| value.widen_complex().norm().powf(p);
        if let Some(spectrum) = self.spectrum() {
            return rescaled_power_norm(
                spectrum_weighted_sum(spectrum, run_cached_dim(provider), power)?,
                p,
                || spectrum_max_abs(spectrum),
                |max| {
                    spectrum_weighted_sum(spectrum, run_cached_dim(provider), |value| {
                        scaled_power(value, max, p)
                    })
                },
            );
        }
        let structure = self.logical_space().space().structure();
        let nout = self.logical_space().space().nout();
        let payload = self
            .owned_body()
            .expect("owned norm input")
            .materialized_dense_data();
        let data: &[D] = &payload;
        rescaled_power_norm(
            coupled_region_weighted_sum(structure, nout, data, run_cached_dim(provider), power)?,
            p,
            || max_abs(data.iter().copied()),
            |max| {
                coupled_region_weighted_sum(
                    structure,
                    nout,
                    data,
                    run_cached_dim(provider),
                    |value| scaled_power(value, max, p),
                )
            },
        )
    }

    /// TensorKit `norm(t, p)`: the entrywise `p`-norm of the reduced blocks,
    ///
    /// ```text
    /// p == 2       -> sqrt(sum_c dim(c) * sum_ij |self_c[i,j]|^2)
    /// p == Inf     -> max_c max_ij |self_c[i,j]|          (not dim-weighted)
    /// finite p > 0 -> (sum_c dim(c) * sum_ij |self_c[i,j]|^p)^(1/p)
    /// ```
    ///
    /// `p == 2.0` is the quantum-dimension-weighted Frobenius norm and is the
    /// only exponent the CUDA storage supports; `p` is an `f64` exactly as in
    /// TensorKit, so each norm has one spelling. The entrywise norm is never
    /// an operator norm, matrices included.
    ///
    /// For `p == 2`, abelian providers have `dim(c) = 1`, giving the ordinary Frobenius
    /// norm. Compact diagonal input is reduced directly in `O(sum_c k_c)`;
    /// dense input is one pass over the payload. Lazy adjoints read their
    /// parent orientation without materializing. Both admission modes share
    /// this one body; they differ only in how `dim(c)` is asked for.
    ///
    /// The Host norm does not overflow or underflow while the norm itself is
    /// representable: when the unscaled sum of squares leaves the `f64` range,
    /// two more passes rescale every entry by the largest magnitude, as Julia's
    /// `LinearAlgebra.generic_norm2` does. An entry of NaN magnitude `|x|` gives
    /// NaN; otherwise an infinite magnitude gives `inf`.
    ///
    /// `p == Inf` follows Julia's NaN-propagating `max`: a payload holding
    /// any NaN, including a complex entry whose real or imaginary part alone
    /// is NaN, returns NaN; an infinite entry returns `+inf`; a tensor with no
    /// stored entries returns `+0.0`. Every exponent is one pass over the
    /// payload, `O(sum_c k_c)` on compact diagonal storage.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when `p` is NaN, zero, negative, or
    ///   `-inf`; TensorKit throws `ArgumentError` over the same domain.
    /// - If a checked provider cannot supply a quantum dimension, its original
    ///   error is available as the source. An invalid coupled-sector layout
    ///   returns [`Error::Core`].
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let id: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?;
    /// let twice = id.axpby(1.0, &id, 1.0)?;
    /// assert!((twice.norm(2.0)? - 2.0_f64.sqrt() * 2.0).abs() < 1e-12);
    /// assert_eq!(twice.norm(f64::INFINITY)?, 2.0);
    /// assert_eq!(id.inner(&id)?, 2.0);
    /// assert_eq!(id.tr()?, 2.0);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn norm(&self, p: f64) -> Result<f64, TypedFacadeError<R>> {
        // Checked before any dispatch so an invalid `p` is rejected the same
        // way on compact and dense storage.
        validate_norm_p(p)?;
        if p == 2.0 {
            return self.frobenius_norm();
        }
        if p.is_infinite() {
            return Ok(self.max_norm());
        }
        self.power_norm(p)
    }

    /// Returns the quantum-dimension-weighted Frobenius inner product
    /// `sum_c dim(c) * sum_ij conj(self_c[i,j]) * other_c[i,j]`.
    ///
    /// The product is conjugate-linear in `self`, and `self.inner(&self)` is
    /// `self.norm(2.0)^2` up to floating-point error. Both tensors must share the
    /// same runtime, hom space, and block layout. A compact
    /// diagonal operand is reduced directly from its stored spectrum in
    /// `O(sum_c k_c)` payload reads, including against a dense lazy adjoint;
    /// it is never densified. Its off-diagonal entries are structural zeros,
    /// so matching dense off-diagonal values are not read even when they are
    /// `NaN` or infinite. This deliberately differs from TensorKit 0.17's
    /// current generic mixed-block reduction, which visits those stored dense
    /// positions and therefore propagates their non-finite values. See
    /// [`Self::norm`] for the weighting, lazy behavior, and example.
    #[doc(alias = "dot")]
    pub fn inner<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D>>,
    ) -> Result<D, TypedFacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        let _host_pool = self.runtime.enter_host_pool();
        if self.logical_space().space() != other.logical_space().space() {
            return Err(Error::from(tenet_tensors::OperationError::SpaceMismatch {
                message: "tensors live on different spaces or block layouts",
            })
            .into());
        }
        let provider = self.logical_space().provider();
        // Compact operands reduce without materializing their structural zeros.
        if let (Some(lhs), Some(rhs)) = (self.spectrum(), other.spectrum()) {
            return Ok(D::from_complex64(compact_inner(
                lhs,
                rhs,
                run_cached_dim(provider),
            )?));
        }
        if let Some(lhs) = self.spectrum() {
            return Ok(D::from_complex64(Self::compact_dense_inner(
                lhs, other, true,
            )?));
        }
        if let Some(rhs) = other.spectrum() {
            return Ok(D::from_complex64(Self::compact_dense_inner(
                rhs, self, false,
            )?));
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            || matches!(&other.repr, TypedTensorRepr::Adjoint(_))
        {
            let (lhs_operand, lhs_data) = self.fusion_operand_and_data();
            let (rhs_operand, rhs_data) = other.fusion_operand_and_data();
            return match (&self.repr, &other.repr) {
                (TypedTensorRepr::Adjoint(lhs), TypedTensorRepr::Adjoint(rhs)) => {
                    tenet_tensors::oriented_fusion_inner_with(
                        lhs.parent.space.space().structure(),
                        tenet_tensors::FusionOperand::direct(rhs.parent.space.space()),
                        rhs.parent_data(),
                        tenet_tensors::FusionOperand::direct(lhs.parent.space.space()),
                        lhs.parent_data(),
                        run_cached_dim(provider),
                    )
                }
                _ => tenet_tensors::oriented_fusion_inner_with(
                    self.logical_space().space().structure(),
                    lhs_operand,
                    &lhs_data,
                    rhs_operand,
                    &rhs_data,
                    run_cached_dim(provider),
                ),
            };
        }
        let unique = <R::Mode as TypedTensorModeDispatch<R>>::fusion_style(provider)
            == tenet_core::FusionStyleKind::Unique;
        // `D::from_complex64` is `.re` for the real scalar and the identity for
        // the complex one, so one static conversion covers both scalar types.
        Ok(D::from_complex64(dense_weighted_inner(
            unique,
            self.logical_space().space().structure(),
            self.logical_space().space().nout(),
            self.owned_body()
                .expect("owned inner input")
                .materialized_dense_data()
                .as_ref(),
            other
                .owned_body()
                .expect("owned inner input")
                .materialized_dense_data()
                .as_ref(),
            run_cached_dim(provider),
        )?))
    }

    /// Returns the quantum-dimension-weighted block trace
    /// `sum_c dim(c) * Tr(self_c)` of an endomorphism.
    ///
    /// The codomain and domain legs must be exactly equal. This is TensorKit's
    /// `tr`; it is not a positivity claim, and the result may be negative or
    /// complex. It also is not the fermionic supertrace: no twist factor is
    /// applied. Use [`Self::trace_pairs`] for the categorical contraction
    /// trace, which includes the provider's pivotal/twist data.
    ///
    /// Compact diagonal input is summed directly in `O(sum_c k_c)`. A
    /// lazy adjoint returns the conjugate of its parent's trace without
    /// materializing. See [`Self::norm`] for a runnable example.
    pub fn tr(&self) -> Result<D, TypedFacadeError<R>> {
        let hom = self.logical_space().space().homspace();
        // The weighted trace below indexes codomain axis `i` together with
        // domain axis `nout + i` and would be meaningless without this check.
        if hom.codomain().legs() != hom.domain().legs() {
            return Err(Error::from(tenet_tensors::OperationError::SpaceMismatch {
                message: "tr() requires an endomorphism (domain == codomain)",
            })
            .into());
        }
        let provider = self.logical_space().provider();
        if let Some(spectrum) = self.spectrum() {
            // `Σ_c dim(c) * Σ_i d_i`: TensorKit's block trace on a
            // `DiagonalTensorMap`, read straight off the stored values.
            let mut weight_of = run_cached_dim(provider);
            let mut total = num_complex::Complex64::new(0.0, 0.0);
            for entry in spectrum {
                let mut partial = D::Wide::from_real(0.0);
                for &value in &entry.values {
                    partial = partial + value.widen();
                }
                total += partial.widen_complex() * weight_of(entry.sector)?;
            }
            return Ok(D::from_complex64(total));
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            return Ok(FactorScalar::adjoint(
                Self::adjoint_parent(view, &self.runtime).tr()?,
            ));
        }
        Ok(D::from_complex64(weighted_trace(
            self.logical_space().space().structure(),
            self.logical_space().space().nout(),
            self.owned_body()
                .expect("owned trace input")
                .materialized_dense_data()
                .as_ref(),
            run_cached_dim(provider),
        )?))
    }
}
