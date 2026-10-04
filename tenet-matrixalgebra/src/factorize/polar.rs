use super::*;

/// Phase and magnitude spectra of an admitted compact diagonal endomorphism:
/// `W` is the per-value phase and `P` the magnitude, both on the input's own
/// coupled sectors, so the caller publishes them on the input space. The
/// right polar factors are the same spectra (`Wh = W`).
fn polar_diagonal_spectra<D>(
    authority: &DynamicFusionMapSpace,
    spectrum: &[SectorSpectrum<D>],
) -> Option<LeftPolar<Vec<SectorSpectrum<D>>>>
where
    D: FactorScalar,
{
    if authority.homspace().codomain() != authority.homspace().domain() {
        return None;
    }
    let Ok(Some(source_regions)) = checked_sector_regions(authority.structure(), authority.nout())
    else {
        return None;
    };
    if validate_endomorphism_region_stacking(
        &source_regions,
        "polar requires identical endomorphism row/column fusion-tree stacking",
    )
    .is_err()
    {
        return None;
    }
    let by_sector = aligned_diagonal_spectrum_by_sector(&source_regions, spectrum)?;
    let mut phase = Vec::with_capacity(source_regions.len());
    let mut magnitude = Vec::with_capacity(source_regions.len());
    for region in source_regions.iter() {
        let entry = by_sector.get(&region.coupled())?;
        let mut phases = Vec::with_capacity(entry.values.len());
        let mut magnitudes = Vec::with_capacity(entry.values.len());
        for &value in &entry.values {
            let (value_phase, value_magnitude) = diagonal_phase_magnitude(value)?;
            phases.push(value_phase);
            magnitudes.push(value_magnitude);
        }
        phase.push(SectorSpectrum {
            sector: region.coupled(),
            values: phases,
        });
        magnitude.push(SectorSpectrum {
            sector: region.coupled(),
            values: magnitudes,
        });
    }
    Some(LeftPolar {
        w: phase,
        p: magnitude,
    })
}

/// Compact left polar spectra `W` (phase) and `P` (magnitude) of an owned
/// compact diagonal endomorphism, or `None` when the input is not admitted.
#[doc(hidden)]
pub fn left_polar_diagonal_spectra_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<LeftPolar<Vec<SectorSpectrum<D>>>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    Ok(polar_diagonal_spectra(authority.space(), spectrum))
}

/// Compact right polar spectra `P` (magnitude) and `Wh` (phase) of an owned
/// compact diagonal endomorphism, or `None` when the input is not admitted.
#[doc(hidden)]
pub fn right_polar_diagonal_spectra_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<RightPolar<Vec<SectorSpectrum<D>>>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    Ok(polar_diagonal_spectra(authority.space(), spectrum)
        .map(|LeftPolar { w, p }| RightPolar { p, wh: w }))
}

#[cfg(test)]
/// Left polar decomposition `t = W * P` (MatrixAlgebraKit `left_polar`):
/// `W` is the isometry `U * Vh` and `P = V * S * Vh` the positive part on
/// the domain. Every coupled-sector matrix must have at least as many rows as
/// columns; otherwise this returns [`OperationError::InvalidArgument`] before
/// entering the dense SVD.
#[expect(
    clippy::type_complexity,
    reason = "static-rank factors differ in rank, so the named result spells both factor types"
)]
pub(crate) fn left_polar<E, RuleKey, BT, BC, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<
    LeftPolar<BoundTensorMap<R, D, NOUT, NIN>, BoundTensorMap<R, D, NIN, NIN>>,
    OperationError,
>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let LeftPolar { w, p } = left_polar_dyn(dense, context, &input.dynamic())?;
    Ok(LeftPolar {
        w: typed_from_bound_factor(w)?,
        p: typed_from_bound_factor(p)?,
    })
}

/// Dynamic-rank [`left_polar`].
pub fn left_polar_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<LeftPolar<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    left_polar_dyn_reported(dense, context, input, PolarDirection::Left)
        .map(|(w, p)| LeftPolar { w, p })
}

pub(super) fn left_polar_dyn_reported<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
    error_direction: PolarDirection,
) -> Result<DynamicFactorPair<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    // Polar needs only U, Vh and the spectrum — not the dense diagonal S — so
    // use the S-free factors core.
    let (u, vh, singular_values) = svd_compact_factors_dyn_with_direction(
        dense,
        input,
        Some((PolarDirection::Left, error_direction)),
        CompactSvdGauge::Left,
    )?;
    let isometry = crate::compose::compose_bound_dyn(context, &u, &vh)?;
    // P = V·S·Vh. Fold S into V as a block-local scaling of V's bond (trailing)
    // axis — TensorKit's `DiagonalTensorMap` `rmul!` — instead of a full block
    // GEMM against the dense diagonal S (99% zeros). `singular_values` carries S
    // in O(rank); see #51 / #55.
    let mut v = adjoint_bound_factor(&vh)?;
    let v_space = v.space().space().clone();
    scale_axis_by_spectrum(&v_space, v.data_mut(), None, &singular_values)?;
    let positive = crate::compose::compose_bound_dyn(context, &v, &vh)?;
    Ok((isometry, positive))
}

#[cfg(test)]
/// Right polar decomposition `t = P * W` (MatrixAlgebraKit `right_polar`):
/// `P = U * S * U^H` is the positive part on the codomain and `W = U * Vh`.
/// Every coupled-sector matrix must have at least as many columns as rows;
/// otherwise this returns [`OperationError::InvalidArgument`] before entering
/// the dense SVD.
#[expect(
    clippy::type_complexity,
    reason = "static-rank factors differ in rank, so the named result spells both factor types"
)]
pub(crate) fn right_polar<E, RuleKey, BT, BC, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<
    RightPolar<BoundTensorMap<R, D, NOUT, NOUT>, BoundTensorMap<R, D, NOUT, NIN>>,
    OperationError,
>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let RightPolar { p, wh } = right_polar_dyn(dense, context, &input.dynamic())?;
    Ok(RightPolar {
        p: typed_from_bound_factor(p)?,
        wh: typed_from_bound_factor(wh)?,
    })
}

/// Dynamic-rank [`right_polar`].
pub fn right_polar_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<RightPolar<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    right_polar_dyn_reported(dense, context, input, PolarDirection::Right)
        .map(|(p, wh)| RightPolar { p, wh })
}

pub(super) fn right_polar_dyn_reported<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    input: &BoundDynamicTensorRef<'_, R, D>,
    error_direction: PolarDirection,
) -> Result<DynamicFactorPair<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    // Polar needs only U, Vh and the spectrum — not the dense diagonal S — so
    // use the S-free factors core.
    let (u, vh, singular_values) = svd_compact_factors_dyn_with_direction(
        dense,
        input,
        Some((PolarDirection::Right, error_direction)),
        CompactSvdGauge::Left,
    )?;
    let uh = adjoint_bound_factor(&u)?;
    let isometry = crate::compose::compose_bound_dyn(context, &u, &vh)?;
    // P = U·S·Uh. Fold S into U's bond (trailing) axis by block-local scaling —
    // TensorKit's `DiagonalTensorMap` `rmul!` — instead of a full block GEMM
    // against the dense diagonal S. U is consumed above for the isometry, so
    // scale the moved-out copy. `singular_values` carries S in O(rank); #51/#55.
    let mut us = u;
    let us_space = us.space().space().clone();
    scale_axis_by_spectrum(&us_space, us.data_mut(), None, &singular_values)?;
    let positive = crate::compose::compose_bound_dyn(context, &us, &uh)?;
    Ok((positive, isometry))
}

/// Left polar factors of an adjoint view, executed on its owned parent.
#[doc(hidden)]
pub fn left_polar_adjoint_parent_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<LeftPolar<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let (positive, isometry) =
        right_polar_dyn_reported(dense, context, parent, PolarDirection::Left)?;
    Ok(LeftPolar {
        w: adjoint_bound_factor(&isometry)?,
        p: positive,
    })
}

/// Right polar factors of an adjoint view, executed on its owned parent.
#[doc(hidden)]
pub fn right_polar_adjoint_parent_dyn<E, RuleKey, BT, BC, R, D>(
    dense: &mut E,
    context: &mut tenet_tensors::TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<RightPolar<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    RuleKey: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet_tensors::TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let (isometry, positive) =
        left_polar_dyn_reported(dense, context, parent, PolarDirection::Right)?;
    Ok(RightPolar {
        p: positive,
        wh: adjoint_bound_factor(&isometry)?,
    })
}

#[derive(Clone, Copy)]
pub(super) struct PolarRegionRoute {
    pub(super) source: usize,
    pub(super) w: usize,
    pub(super) p: usize,
}

pub(super) fn compile_polar_region_routes(
    source: &[CoupledSectorRegion],
    w: &[CoupledSectorRegion],
    p: &[CoupledSectorRegion],
    source_len: usize,
    w_len: usize,
    p_len: usize,
    direction: PolarDirection,
) -> Result<Vec<PolarRegionRoute>, OperationError> {
    let w_by_sector = SectorRegionIndex::new(w)?;
    let p_by_sector = SectorRegionIndex::new(p)?;
    let mut used_w = vec![false; w.len()];
    let mut used_p = vec![false; p.len()];
    let mut routes = Vec::with_capacity(source.len());
    for (source_index, source_region) in source.iter().enumerate() {
        let sector = source_region.coupled();
        let w_index = sector_region_index_of(&w_by_sector, sector, "polar W")?;
        let p_index = sector_region_index_of(&p_by_sector, sector, "polar P")?;
        let w_region = &w[w_index];
        let p_region = &p[p_index];
        if w_region.rows() != source_region.rows()
            || w_region.cols() != source_region.cols()
            || w_region.row_trees() != source_region.row_trees()
            || w_region.col_trees() != source_region.col_trees()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "polar W route does not preserve the source full-tree layout",
            });
        }
        let (dimension, trees) = match direction {
            PolarDirection::Left => (source_region.cols(), source_region.col_trees()),
            PolarDirection::Right => (source_region.rows(), source_region.row_trees()),
        };
        if p_region.rows() != dimension
            || p_region.cols() != dimension
            || p_region.row_trees() != trees
            || p_region.col_trees() != trees
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "polar P route does not preserve the source full-tree layout",
            });
        }
        validate_region_range(source_region, source_len)?;
        validate_region_range(w_region, w_len)?;
        validate_region_range(p_region, p_len)?;
        used_w[w_index] = true;
        used_p[p_index] = true;
        routes.push(PolarRegionRoute {
            source: source_index,
            w: w_index,
            p: p_index,
        });
    }
    if used_w.iter().any(|used| !used) || used_p.iter().any(|used| !used) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "polar output contains a full-tree route absent from the source",
        });
    }
    Ok(routes)
}

struct CheckedPolarPlan<R> {
    source_regions: Arc<[CoupledSectorRegion]>,
    w_regions: Arc<[CoupledSectorRegion]>,
    p_regions: Arc<[CoupledSectorRegion]>,
    routes: Vec<PolarRegionRoute>,
    w_space: BoundDynamicFusionMapSpace<R>,
    p_space: BoundDynamicFusionMapSpace<R>,
}

/// Compact checked-provider polar factors after the ordinary checked prelude.
#[doc(hidden)]
pub struct CheckedCompactPolarFactors<R, D> {
    pub w_space: BoundDynamicFusionMapSpace<R>,
    pub p_space: BoundDynamicFusionMapSpace<R>,
    pub phase: Vec<SectorSpectrum<D>>,
    pub magnitude: Vec<SectorSpectrum<D>>,
}

fn checked_polar_plan<R>(
    authority: &BoundDynamicFusionMapSpace<R>,
    source_len: usize,
    direction: PolarDirection,
    error_direction: PolarDirection,
) -> Result<CheckedPolarPlan<R>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
{
    let source_space = authority.space();
    let checked = CheckedAuthority(authority.provider_arc());
    let rows = checked.coupled_dimensions(source_space.homspace().codomain())?;
    let cols = checked.coupled_dimensions(source_space.homspace().domain())?;
    for (&sector, &row_count) in &rows {
        if !direction.accepts(row_count, cols.get(&sector).copied().unwrap_or(0)) {
            return Err(CheckedGenericFactorPlanError::Operation(
                error_direction.error(),
            ));
        }
    }
    for (&sector, &col_count) in &cols {
        if !rows.contains_key(&sector) && !direction.accepts(0, col_count) {
            return Err(CheckedGenericFactorPlanError::Operation(
                error_direction.error(),
            ));
        }
    }
    let w_space = authority.clone();
    let p_homspace = match direction {
        PolarDirection::Left => FusionTreeHomSpace::new(
            source_space.homspace().domain().clone(),
            source_space.homspace().domain().clone(),
        ),
        PolarDirection::Right => FusionTreeHomSpace::new(
            source_space.homspace().codomain().clone(),
            source_space.homspace().codomain().clone(),
        ),
    };
    let p_nout = p_homspace.codomain().len();
    let p_space = checked.output_space(p_homspace)?;
    let source_regions = checked_sector_regions(source_space.structure(), source_space.nout())?
        .ok_or(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope {
                message: "polar requires coupled-sector input storage",
            },
        ))?;
    let w_regions = checked_sector_regions(w_space.space().structure(), w_space.space().nout())?
        .ok_or(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope {
                message: "polar requires coupled-sector W storage",
            },
        ))?;
    let p_regions = checked_sector_regions(p_space.space().structure(), p_nout)?.ok_or(
        CheckedGenericFactorPlanError::Operation(OperationError::UnsupportedTensorContractScope {
            message: "polar requires coupled-sector P storage",
        }),
    )?;
    let w_len = w_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let p_len = p_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let routes = compile_polar_region_routes(
        &source_regions,
        &w_regions,
        &p_regions,
        source_len,
        w_len,
        p_len,
        direction,
    )?;
    Ok(CheckedPolarPlan {
        source_regions,
        w_regions,
        p_regions,
        routes,
        w_space,
        p_space,
    })
}

fn polar_diagonal_spectra_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    direction: PolarDirection,
) -> Result<Option<CheckedCompactPolarFactors<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    // Refused numerics stay on the ordinary dense path, whose prelude remains
    // the sole provider query sequence for that fallback.
    if spectrum.iter().any(|entry| {
        entry
            .values
            .iter()
            .any(|&value| diagonal_phase_magnitude(value).is_none())
    }) {
        return Ok(None);
    }
    let source = authority.space();
    // TensorKit's polar output (`initialize_output(left_polar!)`) is
    // `W = space(t)` and `P = domain(t) <- domain(t)` (`codomain <- codomain`
    // on the right): for a one-leg bond `V <- V` both are the input space, so
    // they need no provider-built space and no admission query.
    if source.nout() == 1
        && source.nin() == 1
        && source.homspace().codomain() == source.homspace().domain()
    {
        if let Ok(Some(regions)) = checked_sector_regions(source.structure(), 1) {
            let Some(by_sector) = aligned_diagonal_spectrum_by_sector(&regions, spectrum) else {
                return Ok(None);
            };
            let Some((phase, magnitude)) = polar_diagonal_values(
                &by_sector,
                regions.iter().map(|region| (region, region, region)),
            ) else {
                return Ok(None);
            };
            return Ok(Some(CheckedCompactPolarFactors {
                w_space: authority.clone(),
                p_space: authority.clone(),
                phase,
                magnitude,
            }));
        }
    }
    let source_len = source
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let plan = checked_polar_plan(authority, source_len, direction, direction)?;
    let Some(by_sector) = aligned_diagonal_spectrum_by_sector(&plan.source_regions, spectrum)
    else {
        return Ok(None);
    };
    let Some((phase, magnitude)) = polar_diagonal_values(
        &by_sector,
        plan.routes.iter().map(|route| {
            (
                &plan.source_regions[route.source],
                &plan.w_regions[route.w],
                &plan.p_regions[route.p],
            )
        }),
    ) else {
        return Ok(None);
    };
    Ok(Some(CheckedCompactPolarFactors {
        w_space: plan.w_space,
        p_space: plan.p_space,
        phase,
        magnitude,
    }))
}

/// Phase and magnitude spectra of a compact diagonal over aligned
/// `(source, W, P)` region triples, or `None` when dense polar must decide.
#[allow(clippy::type_complexity)]
fn polar_diagonal_values<'r, D: FactorScalar>(
    by_sector: &FxHashMap<SectorId, &SectorSpectrum<D>>,
    triples: impl Iterator<
        Item = (
            &'r CoupledSectorRegion,
            &'r CoupledSectorRegion,
            &'r CoupledSectorRegion,
        ),
    >,
) -> Option<(Vec<SectorSpectrum<D>>, Vec<SectorSpectrum<D>>)> {
    let mut phase = Vec::with_capacity(by_sector.len());
    let mut magnitude = Vec::with_capacity(by_sector.len());
    for (source, w, p) in triples {
        let entry = by_sector.get(&source.coupled())?;
        if !w.has_aligned_diagonal()
            || !p.has_aligned_diagonal()
            || source.row_trees().len() != 1
            || source.col_trees().len() != 1
            || w.row_trees().len() != 1
            || w.col_trees().len() != 1
            || p.row_trees().len() != 1
            || p.col_trees().len() != 1
        {
            return None;
        }
        let mut phases = Vec::with_capacity(entry.values.len());
        let mut magnitudes = Vec::with_capacity(entry.values.len());
        for &value in &entry.values {
            let (value_phase, value_magnitude) = diagonal_phase_magnitude(value)?;
            phases.push(value_phase);
            magnitudes.push(value_magnitude);
        }
        phase.push(SectorSpectrum {
            sector: source.coupled(),
            values: phases,
        });
        magnitude.push(SectorSpectrum {
            sector: source.coupled(),
            values: magnitudes,
        });
    }
    Some((phase, magnitude))
}

#[doc(hidden)]
pub fn left_polar_diagonal_spectra_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<CheckedCompactPolarFactors<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    polar_diagonal_spectra_dyn_checked_generic(authority, spectrum, PolarDirection::Left)
}

#[doc(hidden)]
pub fn right_polar_diagonal_spectra_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<CheckedCompactPolarFactors<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    polar_diagonal_spectra_dyn_checked_generic(authority, spectrum, PolarDirection::Right)
}

pub(super) fn project_hermitian_col_major<D: FactorScalar>(matrix: &mut [D], n: usize) {
    let half = D::from_real(0.5);
    for col in 0..n {
        for row in 0..=col {
            let value =
                (matrix[row + n * col] + FactorScalar::adjoint(matrix[col + n * row])) * half;
            matrix[row + n * col] = value;
            matrix[col + n * row] = FactorScalar::adjoint(value);
        }
    }
}

/// Writes `W = U Vh` into `w` and `P` into `p`, both column-major output
/// regions (MatrixAlgebraKit `left_polar!`/`right_polar!` via SVD): the GEMMs
/// target the destination and the `sqrt(S)` scaling is applied in place to the
/// staged factor, which is not read again.
pub(super) fn checked_generic_polar_products<E, D>(
    dense: &mut E,
    stage: &mut CompactSvdNumericalStage<D>,
    direction: PolarDirection,
    w: &mut [D],
    p: &mut [D],
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let CompactSvdNumericalStage {
        rows,
        cols,
        rank,
        u,
        singular_values,
        vt,
    } = stage;
    let (rows, cols, rank) = (&*rows, &*cols, &*rank);
    let w_shape = [*rows, *cols];
    let w_strides = [1, *rows];
    let u_shape = [*rows, *rank];
    let u_strides = [1, *rows];
    let vt_shape = [*rank, *cols];
    let vt_strides = [1, *rank];
    let w_view = DenseViewMut::new(w, &w_shape, &w_strides, 0).map_err(OperationError::Dense)?;
    let u_view = DenseView::new(u, &u_shape, &u_strides, 0).map_err(OperationError::Dense)?;
    let vt_view = DenseView::new(vt, &vt_shape, &vt_strides, 0).map_err(OperationError::Dense)?;
    dense
        .dot_general_into(
            D::dense_write(w_view),
            D::dense_read(u_view),
            D::dense_read(vt_view),
            &DenseDotConfig::matmul(),
        )
        .map_err(OperationError::Dense)?;

    let p_order = match direction {
        PolarDirection::Left => *cols,
        PolarDirection::Right => *rows,
    };
    let x = match direction {
        PolarDirection::Left => vt,
        PolarDirection::Right => u,
    };
    match direction {
        PolarDirection::Left => {
            for (row, sigma) in singular_values.iter().copied().enumerate() {
                let scale = D::from_real(sigma.sqrt());
                for col in 0..*cols {
                    x[row + rank * col] = x[row + rank * col] * scale;
                }
            }
        }
        PolarDirection::Right => {
            for (col, sigma) in singular_values.iter().copied().enumerate() {
                let scale = D::from_real(sigma.sqrt());
                for row in 0..*rows {
                    x[row + rows * col] = x[row + rows * col] * scale;
                }
            }
        }
    }
    let p_shape = [p_order, p_order];
    let p_strides = [1, p_order];
    let p_view = DenseViewMut::new(p, &p_shape, &p_strides, 0).map_err(OperationError::Dense)?;
    match direction {
        PolarDirection::Left => {
            let xh_shape = [*cols, *rank];
            let xh_strides = [*rank, 1];
            let x_shape = [*rank, *cols];
            let x_strides = [1, *rank];
            let xh = DenseView::new(x, &xh_shape, &xh_strides, 0).map_err(OperationError::Dense)?;
            let x = DenseView::new(x, &x_shape, &x_strides, 0).map_err(OperationError::Dense)?;
            dense
                .dot_general_into(
                    D::dense_write(p_view),
                    D::dense_read(xh),
                    D::dense_read(x),
                    &DenseDotConfig::matmul().with_conjugation(true, false),
                )
                .map_err(OperationError::Dense)?;
        }
        PolarDirection::Right => {
            let x_shape = [*rows, *rank];
            let x_strides = [1, *rows];
            let xh_shape = [*rank, *rows];
            let xh_strides = [*rows, 1];
            let x_view =
                DenseView::new(x, &x_shape, &x_strides, 0).map_err(OperationError::Dense)?;
            let xh = DenseView::new(x, &xh_shape, &xh_strides, 0).map_err(OperationError::Dense)?;
            dense
                .dot_general_into(
                    D::dense_write(p_view),
                    D::dense_read(x_view),
                    D::dense_read(xh),
                    &DenseDotConfig::matmul().with_conjugation(false, true),
                )
                .map_err(OperationError::Dense)?;
        }
    }
    project_hermitian_col_major(p, p_order);
    Ok(())
}

pub(super) fn polar_dyn_checked_generic_reported<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    direction: PolarDirection,
    error_direction: PolarDirection,
) -> Result<DynamicFactorPair<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let source_space = input.space().space();
    let CheckedPolarPlan {
        source_regions,
        w_regions,
        p_regions,
        routes,
        w_space,
        p_space,
    } = checked_polar_plan(
        input.space(),
        input.data().len(),
        direction,
        error_direction,
    )?;
    let p_nout = p_space.space().nout();
    let w_len = w_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let p_len = p_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;

    let data = input.data();
    // One scope spans the staged SVDs and the per-sector polar products.
    let (w_data, p_data) = in_linalg_scope(dense, |dense| {
        let mut stages = Vec::with_capacity(routes.len());
        for route in &routes {
            let region = &source_regions[route.source];
            stages.push(compact_svd_numerical_stage(
                dense,
                &data[region.range()],
                region.rows(),
                region.cols(),
            )?);
        }
        // Why zeroed rather than uninitialized: the GEMM destinations are
        // safe initialized views, and every region is then overwritten once.
        let mut w_data = vec![D::zero(); w_len];
        let mut p_data = vec![D::zero(); p_len];
        for (route, stage) in routes.iter().zip(&mut stages) {
            checked_generic_polar_products(
                dense,
                stage,
                direction,
                &mut w_data[w_regions[route.w].range()],
                &mut p_data[p_regions[route.p].range()],
            )?;
        }
        Ok((w_data, p_data))
    })
    .map_err(CheckedGenericFactorPlanError::from)?;

    let w = BoundDynFactor::from_bound(w_space, w_data, source_space.nout(), source_space.nin())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let p = BoundDynFactor::from_bound(p_space, p_data, p_nout, p_nout)
        .map_err(CheckedGenericFactorPlanError::from)?;
    Ok((w, p))
}

#[doc(hidden)]
pub fn left_polar_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<LeftPolar<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    polar_dyn_checked_generic_reported(dense, input, PolarDirection::Left, PolarDirection::Left)
        .map(|(w, p)| LeftPolar { w, p })
}

#[doc(hidden)]
pub fn right_polar_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<RightPolar<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let (wh, p) = polar_dyn_checked_generic_reported(
        dense,
        input,
        PolarDirection::Right,
        PolarDirection::Right,
    )?;
    Ok(RightPolar { p, wh })
}

/// Left polar of an adjoint view, as the *parent's* right polar factors: the
/// view's left polar is `w = wh^H` and the returned `p`.
#[doc(hidden)]
pub fn left_polar_adjoint_parent_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<RightPolar<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let (wh, p) = polar_dyn_checked_generic_reported(
        dense,
        parent,
        PolarDirection::Right,
        PolarDirection::Left,
    )?;
    Ok(RightPolar { p, wh })
}

/// Right polar of an adjoint view, as the *parent's* left polar factors: the
/// view's right polar is the returned `p` and `wh = w^H`.
#[doc(hidden)]
pub fn right_polar_adjoint_parent_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<LeftPolar<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    polar_dyn_checked_generic_reported(dense, parent, PolarDirection::Left, PolarDirection::Right)
        .map(|(w, p)| LeftPolar { w, p })
}

#[derive(Clone, Copy)]
pub(super) enum PolarDirection {
    Left,
    Right,
}

impl PolarDirection {
    pub(super) fn accepts(self, rows: usize, cols: usize) -> bool {
        match self {
            Self::Left => rows >= cols,
            Self::Right => cols >= rows,
        }
    }

    pub(super) fn error(self) -> OperationError {
        OperationError::InvalidArgument {
            message: match self {
                Self::Left => "left_polar requires rows >= columns in every coupled-sector matrix",
                Self::Right => {
                    "right_polar requires columns >= rows in every coupled-sector matrix"
                }
            },
        }
    }
}

pub(super) fn validate_polar_direction<R>(
    acceptance_direction: PolarDirection,
    error_direction: PolarDirection,
    space: &BoundDynamicFusionMapSpace<R>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let authority = MfAuthority(space);
    let row_dimensions = authority.coupled_dimensions(space.space().homspace().codomain())?;
    let col_dimensions = authority.coupled_dimensions(space.space().homspace().domain())?;
    for (&sector, &rows) in &row_dimensions {
        let cols = col_dimensions.get(&sector).copied().unwrap_or(0);
        if !acceptance_direction.accepts(rows, cols) {
            return Err(error_direction.error());
        }
    }
    for (&sector, &cols) in &col_dimensions {
        if !row_dimensions.contains_key(&sector) && !acceptance_direction.accepts(0, cols) {
            return Err(error_direction.error());
        }
    }
    Ok(())
}
