use super::*;

/// Finite, dtype-representable phase and magnitude, with unit phase at zero.
/// Scaling before normalization keeps complex subnormals on the unit circle.
pub(super) fn diagonal_phase_magnitude<D: FactorScalar>(value: D) -> Option<(D, D)> {
    let value = value.widen_complex();
    if !value.re.is_finite() || !value.im.is_finite() {
        return None;
    }
    let scale = value.re.abs().max(value.im.abs());
    let (phase, magnitude) = if scale == 0.0 {
        (Complex64::new(1.0, 0.0), 0.0)
    } else {
        let normalized = value / scale;
        let norm = normalized.norm();
        (normalized / norm, scale * norm)
    };
    if !magnitude.is_finite() {
        return None;
    }
    let phase = D::from_complex64(phase);
    let magnitude = D::from_real(magnitude);
    let phase_check = phase.widen_complex();
    let magnitude_check = magnitude.widen_complex();
    if !phase_check.re.is_finite()
        || !phase_check.im.is_finite()
        || !magnitude_check.re.is_finite()
        || !magnitude_check.im.is_finite()
    {
        return None;
    }
    Some((phase, magnitude))
}

/// QR spectra on the input bond `V <- V`, including its dual orientation.
/// Full and compact QR coincide; LQ exchanges the phase/magnitude factors.
/// Returns `None` when existing dense execution must decide the result.
#[doc(hidden)]
pub fn qr_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Option<Qr<Vec<SectorSpectrum<D>>>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let source = authority.space();
    if source.nout() != 1
        || source.nin() != 1
        || source.homspace().codomain() != source.homspace().domain()
    {
        return None;
    }
    let regions = checked_sector_regions(source.structure(), source.nout()).ok()??;
    validate_endomorphism_region_stacking(&regions, "diagonal QR requires aligned trees").ok()?;
    let by_sector = aligned_diagonal_spectrum_by_sector(&regions, spectrum)?;
    let mut q = Vec::with_capacity(regions.len());
    let mut r = Vec::with_capacity(regions.len());
    for region in regions.iter() {
        let entry = by_sector.get(&region.coupled())?;
        let mut phases = Vec::with_capacity(entry.values.len());
        let mut magnitudes = Vec::with_capacity(entry.values.len());
        for &value in &entry.values {
            let (phase, magnitude) = diagonal_phase_magnitude(value)?;
            phases.push(phase);
            magnitudes.push(magnitude);
        }
        q.push(SectorSpectrum {
            sector: entry.sector,
            values: phases,
        });
        r.push(SectorSpectrum {
            sector: entry.sector,
            values: magnitudes,
        });
    }
    Some(Qr { q, r })
}

#[allow(clippy::type_complexity)]
fn checked_diagonal_qr_lq<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    full_product: Option<&FusionProductSpace>,
) -> Result<
    Option<(
        BoundDynamicFusionMapSpace<R>,
        BoundDynamicFusionMapSpace<R>,
        Vec<SectorSpectrum<D>>,
        Vec<SectorSpectrum<D>>,
    )>,
    CheckedGenericFactorPlanError<R::Error>,
>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let source = authority.space();
    if source.nout() != 1
        || source.nin() != 1
        || source.homspace().codomain() != source.homspace().domain()
        || source.homspace().codomain().legs()[0].is_dual()
    {
        return Ok(None);
    }
    let Ok(Some(source_regions)) = checked_sector_regions(source.structure(), 1) else {
        return Ok(None);
    };
    if compact_bond_leg(&source_regions) != source.homspace().codomain().legs()[0] {
        return Ok(None);
    }
    let Some(by_sector) = aligned_diagonal_spectrum_by_sector(&source_regions, spectrum) else {
        return Ok(None);
    };
    let mut phases = Vec::with_capacity(source_regions.len());
    let mut magnitudes = Vec::with_capacity(source_regions.len());
    for region in source_regions.iter() {
        let Some(entry) = by_sector.get(&region.coupled()) else {
            return Ok(None);
        };
        if region.row_trees().len() != 1 || region.col_trees().len() != 1 {
            return Ok(None);
        }
        let mut sector_phases = Vec::with_capacity(entry.values.len());
        let mut sector_magnitudes = Vec::with_capacity(entry.values.len());
        for &value in &entry.values {
            let Some((phase, magnitude)) = diagonal_phase_magnitude(value) else {
                return Ok(None);
            };
            sector_phases.push(phase);
            sector_magnitudes.push(magnitude);
        }
        phases.push(SectorSpectrum {
            sector: entry.sector,
            values: sector_phases,
        });
        magnitudes.push(SectorSpectrum {
            sector: entry.sector,
            values: sector_magnitudes,
        });
    }

    if let Some(product) = full_product {
        let dimensions =
            coupled_sector_block_dimensions_generic_checked(product, authority.provider())?;
        let source_dimensions = spectrum
            .iter()
            .map(|entry| (entry.sector, entry.values.len()))
            .collect::<BTreeMap<_, _>>();
        if dimensions != source_dimensions {
            return Ok(None);
        }
    }

    let bond = source.homspace().codomain().legs()[0].clone();
    let left_hom = FusionTreeHomSpace::new(
        source.homspace().codomain().clone(),
        FusionProductSpace::new([bond.clone()]),
    );
    let left_prepared = left_hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(
            authority.provider(),
        )
        .map_err(CheckedGenericFactorPlanError::from)?;
    let left_regions = checked_sector_regions(left_prepared.structure(), 1)?.ok_or(
        CheckedGenericFactorPlanError::Operation(OperationError::UnsupportedTensorContractScope {
            message: "compact diagonal QR/LQ left output is not a coupled-sector matrix layout",
        }),
    )?;
    let right_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([bond]),
        source.homspace().domain().clone(),
    );
    let right_prepared = right_hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(
            authority.provider(),
        )
        .map_err(CheckedGenericFactorPlanError::from)?;
    let right_regions = checked_sector_regions(right_prepared.structure(), 1)?.ok_or(
        CheckedGenericFactorPlanError::Operation(OperationError::UnsupportedTensorContractScope {
            message: "compact diagonal QR/LQ right output is not a coupled-sector matrix layout",
        }),
    )?;
    if !source_factor_tree_extents_match(&source_regions, &left_regions, &right_regions) {
        return Ok(None);
    }
    // For a conforming deterministic provider the admitted rank-1 endomorphism
    // fixes these three region sets to the same HomSpace. Keep the complete
    // route check before publication as a defensive structural boundary.
    if compile_compact_factor_routes(&source_regions, &left_regions, &right_regions).is_err() {
        return Ok(None);
    }
    let left = BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(authority.provider_arc()),
        left_hom,
        left_prepared,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let right = BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(authority.provider_arc()),
        right_hom,
        right_prepared,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    Ok(Some((left, right, phases, magnitudes)))
}

#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn qr_diagonal_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    full: bool,
) -> Result<
    Option<(
        BoundDynamicFusionMapSpace<R>,
        BoundDynamicFusionMapSpace<R>,
        Vec<SectorSpectrum<D>>,
        Vec<SectorSpectrum<D>>,
    )>,
    CheckedGenericFactorPlanError<R::Error>,
>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let full_product = full.then(|| authority.space().homspace().codomain());
    checked_diagonal_qr_lq(authority, spectrum, full_product)
}

#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn lq_diagonal_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    full: bool,
) -> Result<
    Option<(
        BoundDynamicFusionMapSpace<R>,
        BoundDynamicFusionMapSpace<R>,
        Vec<SectorSpectrum<D>>,
        Vec<SectorSpectrum<D>>,
    )>,
    CheckedGenericFactorPlanError<R::Error>,
>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let full_product = full.then(|| authority.space().homspace().domain());
    checked_diagonal_qr_lq(authority, spectrum, full_product)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompactQrCopyProbe {
    pub input_pack_calls: usize,
    pub input_pack_bytes: usize,
    pub output_scatter_calls: usize,
    pub output_scatter_bytes: usize,
    pub owned_output_publications: usize,
    pub owned_output_owner_reused: usize,
}

#[cfg(test)]
pub(crate) fn reset_compact_qr_copy_probe() {
    COMPACT_QR_COPY_PROBE.with(|probe| probe.set(CompactQrCopyProbe::default()));
}

#[cfg(test)]
pub(crate) fn compact_qr_copy_probe() -> CompactQrCopyProbe {
    COMPACT_QR_COPY_PROBE.with(Cell::get)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompactLqCopyProbe {
    pub input_pack_calls: usize,
    pub input_pack_bytes: usize,
    pub output_scatter_calls: usize,
    pub output_scatter_bytes: usize,
    pub scratch_buffer_count: usize,
    pub scratch_capacity_bytes: usize,
    pub adjoint_scratch_fill_calls: usize,
    pub adjoint_scratch_fill_bytes: usize,
    pub final_adjoint_copy_calls: usize,
    pub final_adjoint_copy_bytes: usize,
    pub output_prefill_bytes: usize,
}

#[cfg(test)]
pub(crate) fn reset_compact_lq_copy_probe() {
    COMPACT_LQ_COPY_PROBE.with(|probe| probe.set(CompactLqCopyProbe::default()));
}

#[cfg(test)]
pub(crate) fn compact_lq_copy_probe() -> CompactLqCopyProbe {
    COMPACT_LQ_COPY_PROBE.with(Cell::get)
}

#[cfg(test)]
pub(super) fn record_compact_qr_input_pack<D>(matricizations: &[SectorMatricization<D>]) {
    COMPACT_QR_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.input_pack_calls += matricizations.len();
        current.input_pack_bytes += matricizations
            .iter()
            .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
            .sum::<usize>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_qr_output_scatter<D>(elements: usize) {
    record_compact_qr_output_scatter_work::<D>(1, elements);
}

#[cfg(test)]
pub(super) fn record_compact_qr_output_scatter_work<D>(calls: usize, elements: usize) {
    COMPACT_QR_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += calls;
        current.output_scatter_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_input_pack<D>(matricizations: &[SectorMatricization<D>]) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.input_pack_calls += matricizations.len();
        current.input_pack_bytes += matricizations
            .iter()
            .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
            .sum::<usize>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_output_scatter<D>(elements: usize) {
    record_compact_lq_output_scatter_work::<D>(1, elements);
}

#[cfg(test)]
pub(super) fn record_compact_lq_output_scatter_work<D>(calls: usize, elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += calls;
        current.output_scatter_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_scratch<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.scratch_buffer_count += 1;
        current.scratch_capacity_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_adjoint_fill<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.adjoint_scratch_fill_calls += 1;
        current.adjoint_scratch_fill_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_output_prefill<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_prefill_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_final_adjoint_copy<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.final_adjoint_copy_calls += 1;
        current.final_adjoint_copy_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

pub(super) fn full_qr_numerical_stage<E, D>(
    dense: &mut E,
    input: &[D],
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut q = vec![D::zero(); rows * rows];
    let mut r = if rows <= cols {
        let mut r = vec![D::zero(); rows * cols];
        qr_into_workspace(
            dense, input, rows, cols, rows, &mut q, rows, rows, rows, &mut r, rows, cols, rows,
        )?;
        r
    } else {
        // The current full-Q completion retains augmentation until #1140 A3
        // supplies a supported efficient dense-backend path.
        let mut augmented = vec![D::zero(); rows * (cols + rows)];
        augmented[..rows * cols].copy_from_slice(input);
        for row in 0..rows {
            augmented[rows * cols + row * rows + row] = D::one();
        }
        let mut work_r = vec![D::zero(); rows * (cols + rows)];
        qr_into_workspace(
            dense,
            &augmented,
            rows,
            cols + rows,
            rows,
            &mut q,
            rows,
            rows,
            rows,
            &mut work_r,
            rows,
            cols + rows,
            rows,
        )?;
        work_r[..rows * cols].to_vec()
    };
    positive_diagonal_gauge(&mut q, rows, &mut r, rows, cols);
    Ok((q, r))
}

pub(super) fn scale_col<D: FactorScalar>(
    data: &mut [D],
    rows: usize,
    leading: usize,
    col: usize,
    phase: D,
) {
    for row in 0..rows {
        let index = row + leading * col;
        data[index] = data[index] * phase;
    }
}

pub(super) fn scale_row<D: FactorScalar>(
    data: &mut [D],
    cols: usize,
    leading: usize,
    row: usize,
    phase: D,
) {
    for col in 0..cols {
        let index = row + leading * col;
        data[index] = data[index] * phase;
    }
}

/// Full QR `t = Q * R` (MatrixAlgebraKit `qr_full`): per sector `Q` is the
/// square `m x m` unitary and `R` the upper-trapezoidal `m x n`, obtained
/// from one economy QR, augmenting with identity columns only when `m > n`.
/// The positive-diagonal gauge is applied (MAK / TensorKit 0.17 default).
#[expect(
    clippy::type_complexity,
    reason = "static-rank factors differ in rank, so the named result spells both factor types"
)]
pub fn qr_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<Qr<BoundTensorMap<R, D, NOUT, 1>, BoundTensorMap<R, D, 1, NIN>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let Qr { q, r } = qr_full_dyn(dense, &input.dynamic())?;
    Ok(Qr {
        q: typed_from_bound_factor(q)?,
        r: typed_from_bound_factor(r)?,
    })
}

/// Provider-bound dynamic-rank [`qr_full`].
pub fn qr_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Qr<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    let matrices =
        multiplicity_free_input_matricizations(space.structure(), input.data(), space.nout())?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        let rows = matrix.rows;
        let cols = matrix.cols;
        let (q, r) = full_qr_numerical_stage(dense, matrix.data, rows, cols)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rows,
            left: q,
            left_rows: rows,
            right: r,
            right_leading: rows,
        });
    }
    let dimensions = space
        .homspace()
        .codomain()
        .coupled_sector_block_dimensions(input.space().provider())?;
    with_input_geometry!(&matrices, |geometry| Ok(Qr {
        q: build_bound_factor(
            input.space(),
            space.homspace(),
            geometry,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )?,
        r: build_bound_factor(
            input.space(),
            space.homspace(),
            geometry,
            &mut pairs,
            &dimensions,
            FactorSide::Right,
        )?,
    }))
}

/// Full LQ `t = L * Q` (MatrixAlgebraKit `lq_full`): per sector `L` is the
/// lower-trapezoidal `m x n` and `Q` the square `n x n` unitary, via the full
/// QR of the adjoint sector matrices.
/// The positive-diagonal gauge is applied (MAK / TensorKit 0.17 default).
#[expect(
    clippy::type_complexity,
    reason = "static-rank factors differ in rank, so the named result spells both factor types"
)]
pub fn lq_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<Lq<BoundTensorMap<R, D, NOUT, 1>, BoundTensorMap<R, D, 1, NIN>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let Lq { l, q } = lq_full_dyn(dense, &input.dynamic())?;
    Ok(Lq {
        l: typed_from_bound_factor(l)?,
        q: typed_from_bound_factor(q)?,
    })
}

/// Provider-bound dynamic-rank [`lq_full`].
pub fn lq_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Lq<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    let matrices =
        multiplicity_free_input_matricizations(space.structure(), input.data(), space.nout())?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        let rows = matrix.rows;
        let cols = matrix.cols;
        let transposed = adjoint_col_major(matrix.data, rows, cols);
        let (q_prime, r_prime) = full_qr_numerical_stage(dense, &transposed, cols, rows)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: cols,
            left: adjoint_col_major(&r_prime, cols, rows),
            left_rows: rows,
            right: adjoint_col_major(&q_prime, cols, cols),
            right_leading: cols,
        });
    }
    let dimensions = space
        .homspace()
        .domain()
        .coupled_sector_block_dimensions(input.space().provider())?;
    with_input_geometry!(&matrices, |geometry| Ok(Lq {
        l: build_bound_factor(
            input.space(),
            space.homspace(),
            geometry,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )?,
        q: build_bound_factor(
            input.space(),
            space.homspace(),
            geometry,
            &mut pairs,
            &dimensions,
            FactorSide::Right,
        )?,
    }))
}

/// Compact QR `t = Q * R` (MatrixAlgebraKit `qr_compact`):
/// `Q : codomain <- W` has orthonormal columns per coupled sector and
/// `R : W <- domain` with per-sector bond `min(rows, cols)`. The
/// positive-diagonal gauge is applied (MAK / TensorKit 0.17 default
/// `positive = true`): `R`'s diagonal is real non-negative per sector.
#[expect(
    clippy::type_complexity,
    reason = "static-rank factors differ in rank, so the named result spells both factor types"
)]
pub fn qr_compact<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<Qr<BoundTensorMap<R, D, NOUT, 1>, BoundTensorMap<R, D, 1, NIN>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let Qr { q, r } = qr_compact_dyn(dense, &input.dynamic())?;
    Ok(Qr {
        q: typed_from_bound_factor(q)?,
        r: typed_from_bound_factor(r)?,
    })
}

/// Provider-bound compact QR used by authority-preserving callers.
pub fn qr_compact_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Qr<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    if let Some(plan) = compact_factor_plan(input.space())? {
        return qr_compact_direct_regions(dense, input, &plan).map(|(q, r)| Qr { q, r });
    }
    let space = input.space().space();
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_qr_input_pack(&matricizations);
    let blocks = matricizations
        .iter()
        .map(|matrix| (matrix.data.as_slice(), matrix.rows, matrix.cols))
        .collect::<Vec<_>>();
    let factors = compact_qr_owned_batch(dense, &blocks)?;
    let mut pairs = Vec::with_capacity(matricizations.len());
    for (matrix, (mut q, mut r)) in matricizations.iter().zip(factors) {
        let rank = matrix.rows.min(matrix.cols);
        positive_diagonal_gauge_strided(
            &mut q,
            matrix.rows,
            matrix.rows,
            &mut r,
            rank,
            rank,
            matrix.cols,
        );
        #[cfg(test)]
        {
            record_compact_qr_output_scatter::<D>(q.len());
            record_compact_qr_output_scatter::<D>(r.len());
        }
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rank,
            left: q,
            left_rows: matrix.rows,
            right: r,
            right_leading: rank,
        });
    }
    build_left_right_bound_pair(input.space(), space.homspace(), &matricizations, &mut pairs)
        .map(|(q, r)| Qr { q, r })
}

pub(super) fn qr_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
) -> Result<DynamicFactorPair<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source_layout, input.space().validated_layout());
    let left_space = input.space().rebind_validated(&plan.left_layout)?;
    let right_space = input.space().rebind_validated(&plan.right_layout)?;
    let mut left_regions = (0..plan.left_regions.len())
        .map(|_| None)
        .collect::<Vec<_>>();
    let mut right_regions = (0..plan.right_regions.len())
        .map(|_| None)
        .collect::<Vec<_>>();

    let routes = plan
        .routes
        .iter()
        .copied()
        .filter(|route| route.rank != 0)
        .collect::<Vec<_>>();
    let blocks = routes
        .iter()
        .map(|route| {
            let source = &plan.source_regions[route.source_region];
            (&input.data()[source.range()], source.rows(), source.cols())
        })
        .collect::<Vec<_>>();
    let factors = compact_qr_owned_batch(dense, &blocks)?;
    for (route, (mut left_data, mut right_data)) in routes.into_iter().zip(factors) {
        let source = &plan.source_regions[route.source_region];
        positive_diagonal_gauge_strided(
            &mut left_data,
            source.rows(),
            source.rows(),
            &mut right_data,
            route.rank,
            route.rank,
            source.cols(),
        );
        left_regions[route.left_region.expect("nonzero route has left region")] = Some(left_data);
        right_regions[route.right_region.expect("nonzero route has right region")] =
            Some(right_data);
    }

    let left_data = concat_compact_factor_regions(left_regions, plan.left_layout.required_len()?);
    let right_data =
        concat_compact_factor_regions(right_regions, plan.right_layout.required_len()?);

    let left = BoundDynFactor::from_bound(left_space, left_data, space.nout(), 1)?;
    let right = BoundDynFactor::from_bound(right_space, right_data, 1, space.nin())?;
    Ok((left, right))
}

/// Compact LQ `t = L * Q` (MatrixAlgebraKit `lq_compact`, via the QR of the
/// transposed sector matrices): `Q : W <- domain` has orthonormal rows per
/// coupled sector and `L : codomain <- W`. The positive-diagonal gauge is
/// applied (MAK / TensorKit 0.17 default `positive = true`): `L`'s diagonal
/// is real non-negative per sector.
#[expect(
    clippy::type_complexity,
    reason = "static-rank factors differ in rank, so the named result spells both factor types"
)]
pub fn lq_compact<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<Lq<BoundTensorMap<R, D, NOUT, 1>, BoundTensorMap<R, D, 1, NIN>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let Lq { l, q } = lq_compact_dyn(dense, &input.dynamic())?;
    Ok(Lq {
        l: typed_from_bound_factor(l)?,
        q: typed_from_bound_factor(q)?,
    })
}

/// Provider-bound compact LQ used by authority-preserving callers.
pub fn lq_compact_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Lq<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    if let Some(plan) = compact_factor_plan(input.space())? {
        return lq_compact_direct_regions(dense, input, &plan).map(|(l, q)| Lq { l, q });
    }
    let space = input.space().space();
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_lq_input_pack(&matricizations);
    let mut pairs = Vec::with_capacity(matricizations.len());
    in_linalg_scope(dense, |dense| {
        for matrix in &matricizations {
            let rank = matrix.rows.min(matrix.cols);
            let adjoint = adjoint_col_major(&matrix.data, matrix.rows, matrix.cols);
            let (mut q_prime, mut r_prime) =
                compact_qr_owned(dense, &adjoint, matrix.cols, matrix.rows)?;
            positive_diagonal_gauge_strided(
                &mut q_prime,
                matrix.cols,
                matrix.cols,
                &mut r_prime,
                rank,
                rank,
                matrix.rows,
            );
            #[cfg(test)]
            {
                record_compact_lq_output_scatter::<D>(r_prime.len());
                record_compact_lq_output_scatter::<D>(q_prime.len());
            }
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: rank,
                left: adjoint_col_major(&r_prime, rank, matrix.rows),
                left_rows: matrix.rows,
                right: adjoint_col_major(&q_prime, matrix.cols, rank),
                right_leading: rank,
            });
        }
        Ok(())
    })?;
    build_left_right_bound_pair(input.space(), space.homspace(), &matricizations, &mut pairs)
        .map(|(l, q)| Lq { l, q })
}

pub(super) fn lq_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
) -> Result<DynamicFactorPair<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source_layout, input.space().validated_layout());
    let left_space = input.space().rebind_validated(&plan.left_layout)?;
    let right_space = input.space().rebind_validated(&plan.right_layout)?;
    let left_len = plan.left_layout.required_len()?;
    let right_len = plan.right_layout.required_len()?;
    // Every output element is written once by the final adjoints. When the
    // nonzero routes visit both factor regions in storage order, append them
    // to empty buffers instead of zero-filling first; otherwise overwrite a
    // zeroed buffer region by region.
    let append = lq_routes_append_in_storage_order(plan, left_len, right_len);
    #[cfg(test)]
    let append = append && !FORCE_LQ_ZEROED_PUBLICATION.with(Cell::get);
    let (mut left_data, mut right_data) = if append {
        (Vec::with_capacity(left_len), Vec::with_capacity(right_len))
    } else {
        (vec![D::zero(); left_len], vec![D::zero(); right_len])
    };

    let max_adjoint_len = plan
        .routes
        .iter()
        .map(|route| plan.source_regions[route.source_region].range().len())
        .max()
        .unwrap_or(0);
    let mut adjoint_scratch = Vec::with_capacity(max_adjoint_len);
    #[cfg(test)]
    record_compact_lq_scratch::<D>(max_adjoint_len);

    let data = input.data();
    in_linalg_scope(dense, |dense| {
        for route in plan.routes.iter().copied() {
            if route.rank == 0 {
                continue;
            }
            let source = &plan.source_regions[route.source_region];
            let left =
                &plan.left_regions[route.left_region.expect("nonzero route has left region")];
            let right =
                &plan.right_regions[route.right_region.expect("nonzero route has right region")];
            let source_data = &data[source.range()];
            adjoint_scratch.clear();
            extend_adjoint_col_major(
                &mut adjoint_scratch,
                source_data,
                source.rows(),
                source.cols(),
            );
            #[cfg(test)]
            record_compact_lq_adjoint_fill::<D>(source_data.len());

            let (mut q_prime, mut r_prime) =
                compact_qr_owned(dense, &adjoint_scratch, source.cols(), source.rows())?;
            positive_diagonal_gauge_strided(
                &mut q_prime,
                source.cols(),
                source.cols(),
                &mut r_prime,
                route.rank,
                route.rank,
                source.rows(),
            );
            if append {
                extend_adjoint_col_major(&mut left_data, &r_prime, route.rank, source.rows());
                extend_adjoint_col_major(&mut right_data, &q_prime, source.cols(), route.rank);
            } else {
                adjoint_col_major_into(
                    &r_prime,
                    route.rank,
                    source.rows(),
                    &mut left_data[left.range()],
                );
                adjoint_col_major_into(
                    &q_prime,
                    source.cols(),
                    route.rank,
                    &mut right_data[right.range()],
                );
            }
            #[cfg(test)]
            record_compact_lq_final_adjoint_copy::<D>(r_prime.len());
            #[cfg(test)]
            record_compact_lq_final_adjoint_copy::<D>(q_prime.len());
        }
        Ok(())
    })?;
    #[cfg(test)]
    record_compact_lq_output_prefill::<D>(if append { 0 } else { left_len + right_len });

    let left = BoundDynFactor::from_bound(left_space, left_data, space.nout(), 1)?;
    let right = BoundDynFactor::from_bound(right_space, right_data, 1, space.nin())?;
    Ok((left, right))
}

#[cfg(test)]
thread_local! {
    pub(super) static FORCE_LQ_ZEROED_PUBLICATION: Cell<bool> = const { Cell::new(false) };
}

/// Routes compact LQ through its zero-and-overwrite publication on this
/// thread even when the append proof holds.
#[cfg(test)]
pub(crate) fn force_lq_zeroed_publication_for_test(force: bool) {
    FORCE_LQ_ZEROED_PUBLICATION.with(|cell| cell.set(force));
}

/// [`lq_routes_append_in_storage_order`] after replacing the plan's routes.
#[cfg(test)]
pub(crate) fn lq_routes_append_with_routes_for_test(
    plan: &mut CompactFactorPlan,
    routes: Vec<CompactFactorRoute>,
) -> Result<bool, OperationError> {
    plan.routes = routes;
    Ok(lq_routes_append_in_storage_order(
        plan,
        plan.left_layout.required_len()?,
        plan.right_layout.required_len()?,
    ))
}

/// Whether the nonzero routes of `plan` reach the left and right factor
/// regions contiguously from offset zero, in route order, and cover
/// `left_len`/`right_len` exactly, so appending each route's factors in turn
/// builds both outputs. Rank-zero routes own empty regions.
pub(super) fn lq_routes_append_in_storage_order(
    plan: &CompactFactorPlan,
    left_len: usize,
    right_len: usize,
) -> bool {
    let (mut left_end, mut right_end) = (0usize, 0usize);
    for route in &plan.routes {
        let (Some(left), Some(right)) = (route.left_region, route.right_region) else {
            continue;
        };
        let (left, right) = (
            plan.left_regions[left].range(),
            plan.right_regions[right].range(),
        );
        if left.start != left_end || right.start != right_end {
            return false;
        }
        (left_end, right_end) = (left.end, right.end);
    }
    left_end == left_len && right_end == right_len
}

/// Appends the adjoint of the column-major `rows x cols` matrix `data` to
/// `output` as a column-major `cols x rows` matrix.
pub(super) fn extend_adjoint_col_major<D: FactorScalar>(
    output: &mut Vec<D>,
    data: &[D],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(data.len(), rows * cols);
    output.reserve(data.len());
    for row in 0..rows {
        // The StepBy/Take chain retained an out-of-line fold per row in Release.
        output.extend((0..cols).map(|col| FactorScalar::adjoint(data[row + col * rows])));
    }
}

#[test]
fn append_adjoint_preserves_rectangular_order_conjugation_and_existing_values() {
    macro_rules! check {
        ($ty:ty) => {{
            let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(<$ty as FactorScalar>::from_real);
            let mut output = vec![<$ty as FactorScalar>::from_real(99.0)];
            extend_adjoint_col_major(&mut output, &input, 2, 3);
            assert_eq!(
                output,
                [99.0, 1.0, 3.0, 5.0, 2.0, 4.0, 6.0].map(<$ty as FactorScalar>::from_real)
            );
            for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
                let before = output.clone();
                extend_adjoint_col_major(&mut output, &[], rows, cols);
                assert_eq!(output, before);
            }
        }};
    }
    check!(f32);
    check!(f64);
    check!(num_complex::Complex32);
    check!(Complex64);
    let input = [
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
        Complex64::new(9.0, 10.0),
        Complex64::new(11.0, 12.0),
    ];
    let mut output = vec![Complex64::new(99.0, 1.0)];
    extend_adjoint_col_major(&mut output, &input, 2, 3);
    assert_eq!(
        output,
        [
            Complex64::new(99.0, 1.0),
            Complex64::new(1.0, -2.0),
            Complex64::new(5.0, -6.0),
            Complex64::new(9.0, -10.0),
            Complex64::new(3.0, -4.0),
            Complex64::new(7.0, -8.0),
            Complex64::new(11.0, -12.0)
        ]
    );
}

/// Transposes a column-major `rows x cols` matrix into column-major
/// `cols x rows`.
/// Adjoint (conjugate transpose) of a column-major `rows x cols` matrix.
pub(super) fn adjoint_col_major<D: FactorScalar>(data: &[D], rows: usize, cols: usize) -> Vec<D> {
    let mut adjoint = vec![D::zero(); data.len()];
    adjoint_col_major_into(data, rows, cols, &mut adjoint);
    adjoint
}

pub(super) fn adjoint_col_major_into<D: FactorScalar>(
    data: &[D],
    rows: usize,
    cols: usize,
    adjoint: &mut [D],
) {
    debug_assert_eq!(data.len(), rows * cols);
    debug_assert_eq!(adjoint.len(), data.len());
    for col in 0..cols {
        for row in 0..rows {
            adjoint[col + cols * row] = FactorScalar::adjoint(data[row + rows * col]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn qr_into_workspace<E, D>(
    dense: &mut E,
    input: &[D],
    input_rows: usize,
    input_cols: usize,
    input_leading: usize,
    q: &mut [D],
    q_rows: usize,
    q_cols: usize,
    q_leading: usize,
    r: &mut [D],
    r_rows: usize,
    r_cols: usize,
    r_leading: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let input_shape = [input_rows, input_cols];
    let input_strides = [1usize, input_leading];
    let q_shape = [q_rows, q_cols];
    let q_strides = [1usize, q_leading];
    let r_shape = [r_rows, r_cols];
    let r_strides = [1usize, r_leading];
    let input_view =
        DenseView::new(input, &input_shape, &input_strides, 0).map_err(OperationError::Dense)?;
    let q_view = DenseViewMut::new(q, &q_shape, &q_strides, 0).map_err(OperationError::Dense)?;
    let r_view = DenseViewMut::new(r, &r_shape, &r_strides, 0).map_err(OperationError::Dense)?;
    dense
        .qr_into(
            D::dense_read(input_view),
            D::dense_write(q_view),
            D::dense_write(r_view),
        )
        .map_err(OperationError::Dense)
}

/// Compact QR owns both dense outputs, so it can transfer the executor's host
/// buffers directly. Full QR keeps its caller-owned workspace contract above.
pub(super) fn compact_qr_owned<E, D>(
    dense: &mut E,
    input: &[D],
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let input_shape = [rows, cols];
    let input_strides = [1usize, rows];
    let input_view =
        DenseView::new(input, &input_shape, &input_strides, 0).map_err(OperationError::Dense)?;
    let outputs = dense
        .qr(D::dense_read(input_view))
        .map_err(OperationError::Dense)?;
    compact_qr_outputs(outputs, rows, cols)
}

/// Compact QR of each column-major `(data, rows, cols)` block, submitted as one
/// executor batch so the backend admits the whole coupled-sector loop once.
#[expect(
    clippy::type_complexity,
    reason = "the dense QR ownership boundary returns one documented Q, R pair per block"
)]
pub(super) fn compact_qr_owned_batch<E, D>(
    dense: &mut E,
    blocks: &[(&[D], usize, usize)],
) -> Result<Vec<(Vec<D>, Vec<D>)>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factorize_col_major_batch(dense, DenseFactorization::Qr, blocks)?
        .into_iter()
        .zip(blocks)
        .map(|(outputs, &(_, rows, cols))| compact_qr_outputs(outputs, rows, cols))
        .collect()
}

pub(super) fn compact_qr_outputs<D: FactorScalar>(
    mut outputs: Vec<DenseTensor>,
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<D>), OperationError> {
    let rank = rows.min(cols);
    if outputs.len() != 2 {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "qr_into",
            message: "dense QR must return exactly (Q, R)".to_string(),
        }));
    }
    let q = compact_factor_output_owned::<D>(outputs.remove(0), &[rows, rank], "qr_into")?;
    let r = compact_factor_output_owned::<D>(outputs.remove(0), &[rank, cols], "qr_into")?;
    Ok((q, r))
}

/// Provider-bound compact QR for a generic rule.
pub fn qr_compact_dyn_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Qr<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    if let Some(plan) = compact_factor_plan_generic(input.space())? {
        return qr_compact_direct_regions(dense, input, &plan).map(|(q, r)| Qr { q, r });
    }
    let matrices = sector_matricizations_generic(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_qr_input_pack(&matrices);
    let blocks = matrices
        .iter()
        .map(|matrix| (matrix.data.as_slice(), matrix.rows, matrix.cols))
        .collect::<Vec<_>>();
    let factors = compact_qr_owned_batch(dense, &blocks)?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for (matrix, (mut q, mut r)) in matrices.iter().zip(factors) {
        let rank = matrix.rows.min(matrix.cols);
        positive_diagonal_gauge_strided(
            &mut q,
            matrix.rows,
            matrix.rows,
            &mut r,
            rank,
            rank,
            matrix.cols,
        );
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rank,
            left: q,
            left_rows: matrix.rows,
            right: r,
            right_leading: rank,
        });
    }
    #[cfg(test)]
    let scatter_before = generic_pair_publication_probe();
    let result = build_left_right_bound_pair_generic(provider, space.homspace(), &matrices, pairs);
    #[cfg(test)]
    {
        let scatter_after = generic_pair_publication_probe();
        let elements = scatter_after.left_scattered_elements
            - scatter_before.left_scattered_elements
            + scatter_after.right_scattered_elements
            - scatter_before.right_scattered_elements;
        let calls = scatter_after.left_scatter_calls - scatter_before.left_scatter_calls
            + scatter_after.right_scatter_calls
            - scatter_before.right_scatter_calls;
        if calls != 0 {
            record_compact_qr_output_scatter_work::<D>(calls, elements);
        }
    }
    result.map(|(q, r)| Qr { q, r })
}

/// Checked-Generic compact QR. Provider-bound output spaces are admitted
/// through the checked staging boundary; dense QR itself performs no provider
/// queries and therefore needs no Tenferro-specific capability.
#[doc(hidden)]
pub fn qr_compact_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Qr<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    #[cfg(test)]
    if let InputMatricizations::Packed(matrices) = &matrices {
        record_compact_qr_input_pack(matrices);
    }
    let matrix_refs = (0..matrices.len())
        .map(|index| matrices.get(index))
        .collect::<Result<Vec<_>, _>>()
        .map_err(CheckedGenericFactorPlanError::from)?;
    #[cfg(test)]
    for matrix in &matrix_refs {
        record_checked_compact_input(CheckedCompactOperation::Qr, input.data(), matrix.data, None);
    }
    let blocks = matrix_refs
        .iter()
        .map(|matrix| (matrix.data, matrix.rows, matrix.cols))
        .collect::<Vec<_>>();
    let factors =
        compact_qr_owned_batch(dense, &blocks).map_err(CheckedGenericFactorPlanError::from)?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for (matrix, (mut q, mut r)) in matrix_refs.iter().zip(factors) {
        let rank = matrix.rows.min(matrix.cols);
        positive_diagonal_gauge_strided(
            &mut q,
            matrix.rows,
            matrix.rows,
            &mut r,
            rank,
            rank,
            matrix.cols,
        );
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rank,
            left: q,
            left_rows: matrix.rows,
            right: r,
            right_leading: rank,
        });
    }
    build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)
        .map(|(q, r)| Qr { q, r })
}

/// Checked-Generic compact LQ, implemented through the existing host
/// adjoint-plus-QR boundary; no borrowed conjugated-dot capability is needed.
#[doc(hidden)]
pub fn lq_compact_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Lq<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    #[cfg(test)]
    if let InputMatricizations::Packed(matrices) = &matrices {
        record_compact_lq_input_pack(matrices);
    }
    let mut pairs = Vec::with_capacity(matrices.len());
    #[cfg(test)]
    let data = input.data();
    in_linalg_scope(dense, |dense| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            let rank = matrix.rows.min(matrix.cols);
            #[cfg(test)]
            record_compact_lq_adjoint_fill::<D>(matrix.data.len());
            let adjoint = adjoint_col_major(matrix.data, matrix.rows, matrix.cols);
            #[cfg(test)]
            record_checked_compact_input(
                CheckedCompactOperation::Lq,
                data,
                matrix.data,
                Some(&adjoint),
            );
            let (mut q_prime, mut r_prime) =
                compact_qr_owned(dense, &adjoint, matrix.cols, matrix.rows)?;
            positive_diagonal_gauge_strided(
                &mut q_prime,
                matrix.cols,
                matrix.cols,
                &mut r_prime,
                rank,
                rank,
                matrix.rows,
            );
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: rank,
                left: adjoint_col_major(&r_prime, rank, matrix.rows),
                left_rows: matrix.rows,
                right: adjoint_col_major(&q_prime, matrix.cols, rank),
                right_leading: rank,
            });
        }
        Ok(())
    })
    .map_err(CheckedGenericFactorPlanError::from)?;
    build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)
        .map(|(l, q)| Lq { l, q })
}

/// Checked-Generic full QR, augmenting only sectors that require completion.
#[doc(hidden)]
pub fn qr_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Qr<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let rows = matrix.rows;
        let cols = matrix.cols;
        let (q, r) = full_qr_numerical_stage(dense, matrix.data, rows, cols)
            .map_err(CheckedGenericFactorPlanError::from)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rows,
            left: q,
            left_rows: rows,
            right: r,
            right_leading: rows,
        });
    }
    let dimensions = coupled_sector_block_dimensions_generic_checked(
        space.homspace().codomain(),
        provider.as_ref(),
    )?;
    with_input_geometry!(&matrices, |geometry| checked_full_factor_pair(
        provider,
        space.homspace(),
        geometry,
        pairs,
        &dimensions
    ))
    .map(|(q, r)| Qr { q, r })
}

/// Checked-Generic full LQ via the full QR of each sector's adjoint matrix.
#[doc(hidden)]
pub fn lq_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Lq<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let rows = matrix.rows;
        let cols = matrix.cols;
        let transposed = adjoint_col_major(matrix.data, rows, cols);
        let (q_prime, r_prime) = full_qr_numerical_stage(dense, &transposed, cols, rows)
            .map_err(CheckedGenericFactorPlanError::from)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: cols,
            left: adjoint_col_major(&r_prime, cols, rows),
            left_rows: rows,
            right: adjoint_col_major(&q_prime, cols, cols),
            right_leading: cols,
        });
    }
    let dimensions = coupled_sector_block_dimensions_generic_checked(
        space.homspace().domain(),
        provider.as_ref(),
    )?;
    with_input_geometry!(&matrices, |geometry| checked_full_factor_pair(
        provider,
        space.homspace(),
        geometry,
        pairs,
        &dimensions
    ))
    .map(|(l, q)| Lq { l, q })
}

/// Provider-bound compact LQ for a generic rule.
pub fn lq_compact_dyn_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Lq<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    if let Some(plan) = compact_factor_plan_generic(input.space())? {
        return lq_compact_direct_regions(dense, input, &plan).map(|(l, q)| Lq { l, q });
    }
    let matrices = sector_matricizations_generic(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_lq_input_pack(&matrices);
    let mut pairs = Vec::with_capacity(matrices.len());
    in_linalg_scope(dense, |dense| {
        for matrix in &matrices {
            let rank = matrix.rows.min(matrix.cols);
            let adjoint = adjoint_col_major(&matrix.data, matrix.rows, matrix.cols);
            let (mut q_prime, mut r_prime) =
                compact_qr_owned(dense, &adjoint, matrix.cols, matrix.rows)?;
            positive_diagonal_gauge_strided(
                &mut q_prime,
                matrix.cols,
                matrix.cols,
                &mut r_prime,
                rank,
                rank,
                matrix.rows,
            );
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: rank,
                left: adjoint_col_major(&r_prime, rank, matrix.rows),
                left_rows: matrix.rows,
                right: adjoint_col_major(&q_prime, matrix.cols, rank),
                right_leading: rank,
            });
        }
        Ok(())
    })?;
    #[cfg(test)]
    let scatter_before = generic_pair_publication_probe();
    let result = build_left_right_bound_pair_generic(provider, space.homspace(), &matrices, pairs);
    #[cfg(test)]
    {
        let scatter_after = generic_pair_publication_probe();
        let elements = scatter_after.left_scattered_elements
            - scatter_before.left_scattered_elements
            + scatter_after.right_scattered_elements
            - scatter_before.right_scattered_elements;
        let calls = scatter_after.left_scatter_calls - scatter_before.left_scatter_calls
            + scatter_after.right_scatter_calls
            - scatter_before.right_scatter_calls;
        if calls != 0 {
            record_compact_lq_output_scatter_work::<D>(calls, elements);
        }
    }
    result.map(|(l, q)| Lq { l, q })
}
