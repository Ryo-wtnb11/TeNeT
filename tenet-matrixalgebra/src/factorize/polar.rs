use super::*;

struct PolarDiagonalAdmission<R, D> {
    source_regions: Arc<[CoupledSectorRegion]>,
    w_regions: Arc<[CoupledSectorRegion]>,
    p_regions: Arc<[CoupledSectorRegion]>,
    routes: Vec<PolarRegionRoute>,
    diagonal: Vec<Vec<(D, D)>>,
    w_space: BoundDynamicFusionMapSpace<R>,
    p_space: BoundDynamicFusionMapSpace<R>,
}

impl<R, D: FactorScalar> PolarDiagonalAdmission<R, D> {
    fn spectra(self) -> (Vec<SectorSpectrum<D>>, Vec<SectorSpectrum<D>>) {
        let mut phase = Vec::with_capacity(self.routes.len());
        let mut magnitude = Vec::with_capacity(self.routes.len());
        for route in self.routes {
            let sector = self.source_regions[route.source].coupled();
            let (phases, magnitudes) = self.diagonal[route.source].iter().copied().unzip();
            phase.push(SectorSpectrum {
                sector,
                values: phases,
            });
            magnitude.push(SectorSpectrum {
                sector,
                values: magnitudes,
            });
        }
        (phase, magnitude)
    }

    fn dense(self) -> Result<DynamicFactorPair<R, D>, OperationError> {
        let source = self.w_space.space();
        let (nout, nin) = (source.nout(), source.nin());
        let w_len = source
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?;
        let p_len = self
            .p_space
            .space()
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?;
        let mut w_data = vec![D::zero(); w_len];
        let mut p_data = vec![D::zero(); p_len];
        for route in self.routes {
            let n = self.source_regions[route.source].rows();
            let w_start = self.w_regions[route.w].range().start;
            let p_start = self.p_regions[route.p].range().start;
            for (index, &(phase, magnitude)) in self.diagonal[route.source].iter().enumerate() {
                w_data[w_start + index * (n + 1)] = phase;
                p_data[p_start + index * (n + 1)] = magnitude;
            }
        }
        let w = BoundDynFactor::from_bound(self.w_space, w_data, nout, nin)?;
        let p_nout = self.p_space.space().nout();
        let p = BoundDynFactor::from_bound(self.p_space, p_data, p_nout, p_nout)?;
        Ok((w, p))
    }
}

fn polar_diagonal_admission<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    direction: PolarDirection,
) -> Result<Option<PolarDiagonalAdmission<R, D>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let source = authority.space();
    if source.homspace().codomain() != source.homspace().domain() {
        return Ok(None);
    }
    let Ok(Some(source_regions)) = checked_sector_regions(source.structure(), source.nout()) else {
        return Ok(None);
    };
    if validate_endomorphism_region_stacking(
        &source_regions,
        "polar requires identical endomorphism row/column fusion-tree stacking",
    )
    .is_err()
    {
        return Ok(None);
    }
    let by_sector: FxHashMap<_, _> = spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    if by_sector.len() != spectrum.len() || by_sector.len() != source_regions.len() {
        return Ok(None);
    }
    let mut diagonal = Vec::with_capacity(source_regions.len());
    for region in source_regions.iter() {
        let Some(entry) = by_sector.get(&region.coupled()) else {
            return Ok(None);
        };
        if !region.has_aligned_diagonal()
            || region.rows() != region.cols()
            || entry.values.len() != region.rows()
        {
            return Ok(None);
        }
        let mut values = Vec::with_capacity(entry.values.len());
        for &value in &entry.values {
            let Some((phase, magnitude)) = diagonal_phase_magnitude(value) else {
                return Ok(None);
            };
            values.push((phase, magnitude));
        }
        diagonal.push(values);
    }

    let p_homspace = match direction {
        PolarDirection::Left => FusionTreeHomSpace::new(
            source.homspace().domain().clone(),
            source.homspace().domain().clone(),
        ),
        PolarDirection::Right => FusionTreeHomSpace::new(
            source.homspace().codomain().clone(),
            source.homspace().codomain().clone(),
        ),
    };
    let p_nout = p_homspace.codomain().len();
    let Ok(p_space) = authority.derive_from_final_homspace(p_homspace) else {
        return Ok(None);
    };
    let w_space = authority.clone();
    let Ok(Some(w_regions)) = checked_sector_regions(w_space.space().structure(), source.nout())
    else {
        return Ok(None);
    };
    let Ok(Some(p_regions)) = checked_sector_regions(p_space.space().structure(), p_nout) else {
        return Ok(None);
    };
    let source_len = source
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let w_len = w_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let p_len = p_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let Ok(routes) = compile_polar_region_routes(
        &source_regions,
        &w_regions,
        &p_regions,
        source_len,
        w_len,
        p_len,
        direction,
    ) else {
        return Ok(None);
    };
    Ok(Some(PolarDiagonalAdmission {
        source_regions,
        w_regions,
        p_regions,
        routes,
        diagonal,
        w_space,
        p_space,
    }))
}

fn polar_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    direction: PolarDirection,
) -> Result<Option<DynamicFactorPair<R, D>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    polar_diagonal_admission(authority, spectrum, direction)?
        .map(PolarDiagonalAdmission::dense)
        .transpose()
}

/// Compact left polar spectra after the same full-tree route admission as the dense seam.
#[doc(hidden)]
pub fn left_polar_diagonal_spectra_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<LeftPolar<Vec<SectorSpectrum<D>>>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    Ok(
        polar_diagonal_admission(authority, spectrum, PolarDirection::Left)?.map(|admission| {
            let (w, p) = admission.spectra();
            LeftPolar { w, p }
        }),
    )
}

/// Compact right polar spectra after the same full-tree route admission as the dense seam.
#[doc(hidden)]
pub fn right_polar_diagonal_spectra_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<RightPolar<Vec<SectorSpectrum<D>>>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    Ok(
        polar_diagonal_admission(authority, spectrum, PolarDirection::Right)?.map(|admission| {
            let (wh, p) = admission.spectra();
            RightPolar { p, wh }
        }),
    )
}

/// Direct left polar factors of an admitted owned compact diagonal.
#[doc(hidden)]
pub fn left_polar_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<LeftPolar<BoundDynFactor<R, D>>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    polar_diagonal_dyn(authority, spectrum, PolarDirection::Left)
        .map(|result| result.map(|(w, p)| LeftPolar { w, p }))
}

/// Direct right polar factors of an admitted owned compact diagonal.
#[doc(hidden)]
pub fn right_polar_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<RightPolar<BoundDynFactor<R, D>>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    polar_diagonal_dyn(authority, spectrum, PolarDirection::Right)
        .map(|result| result.map(|(wh, p)| RightPolar { p, wh }))
}

/// Left polar decomposition `t = W * P` (MatrixAlgebraKit `left_polar`):
/// `W` is the isometry `U * Vh` and `P = V * S * Vh` the positive part on
/// the domain. Every coupled-sector matrix must have at least as many rows as
/// columns; otherwise this returns [`OperationError::InvalidArgument`] before
/// entering the dense SVD.
#[expect(
    clippy::type_complexity,
    reason = "static-rank factors differ in rank, so the named result spells both factor types"
)]
pub fn left_polar<E, RuleKey, BT, BC, R, D, const NOUT: usize, const NIN: usize>(
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

/// Right polar decomposition `t = P * W` (MatrixAlgebraKit `right_polar`):
/// `P = U * S * U^H` is the positive part on the codomain and `W = U * Vh`.
/// Every coupled-sector matrix must have at least as many columns as rows;
/// otherwise this returns [`OperationError::InvalidArgument`] before entering
/// the dense SVD.
#[expect(
    clippy::type_complexity,
    reason = "static-rank factors differ in rank, so the named result spells both factor types"
)]
pub fn right_polar<E, RuleKey, BT, BC, R, D, const NOUT: usize, const NIN: usize>(
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

pub(super) fn validate_checked_polar_direction<R>(
    input: &BoundDynamicTensorRef<'_, R, impl DenseBlockScalar>,
    direction: PolarDirection,
    error_direction: PolarDirection,
) -> Result<(), CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
{
    let space = input.space().space();
    let rows = coupled_sector_block_dimensions_generic_checked(
        space.homspace().codomain(),
        input.space().provider(),
    )?;
    let cols = coupled_sector_block_dimensions_generic_checked(
        space.homspace().domain(),
        input.space().provider(),
    )?;
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
    Ok(())
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
    validate_checked_polar_direction(input, direction, error_direction)?;
    let source_space = input.space().space();
    let w_space = input.space().clone();
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
    let p_space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(input.space().provider_arc()),
        p_homspace,
    )?;
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
        input.data().len(),
        w_len,
        p_len,
        direction,
    )?;

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
