use super::*;

fn compact_singular_value<D: FactorScalar>(value: D) -> Option<f64> {
    let magnitude = finite_compact_magnitude(value)?;
    let rounded = D::from_real(magnitude).widen_complex().re;
    (rounded.is_finite() && (magnitude == 0.0 || rounded > 0.0)).then_some(rounded)
}

fn compact_null_sector<D: FactorScalar>(values: &[D], side: FactorSide) -> Option<(usize, Vec<D>)> {
    let k = values.len();
    let (sigma_max, nullity) =
        values
            .iter()
            .copied()
            .try_fold((0.0_f64, 0_usize), |(largest, zeros), value| {
                compact_singular_value(value).map(|magnitude| {
                    (
                        largest.max(magnitude),
                        zeros + usize::from(magnitude == 0.0),
                    )
                })
            })?;
    let cutoff = D::epsilon() * k as f64 * sigma_max;
    let margin = cutoff.max(D::epsilon().sqrt() * sigma_max);
    if !margin.is_finite()
        || (sigma_max > 0.0 && (margin == 0.0 || sigma_max < D::safe_minimum()))
        || values.iter().copied().any(|value| {
            compact_singular_value(value)
                .is_none_or(|magnitude| magnitude > 0.0 && magnitude <= margin)
        })
    {
        return None;
    }
    let mut coordinates = vec![D::zero(); k * nullity];
    let mut column = 0;
    for (index, &value) in values.iter().enumerate() {
        if compact_singular_value(value) != Some(0.0) {
            continue;
        }
        match side {
            FactorSide::Left => coordinates[index + column * k] = D::from_real(1.0),
            FactorSide::Right => coordinates[column + index * nullity] = D::from_real(1.0),
        }
        column += 1;
    }
    Some((nullity, coordinates))
}

/// Outcome of the compact-diagonal null kernel; each mode's public entry maps
/// the declines onto its own fallback contract.
enum DiagonalNull<R, D, E> {
    Direct(BoundDynFactor<R, D>),
    /// The spectrum does not cover the aligned regions.
    NotDiagonal,
    /// A sector declined the closed form after the coupled dimensions were
    /// queried; the query result is handed to the solver path.
    Declined(Result<BTreeMap<SectorId, usize>, E>),
}

/// Coordinate kernels of an admitted compact diagonal. A sector with any
/// positive magnitude near the rank cutoff stays on the solver path; the
/// margin is conservative, not a provider-specific singular-value bound.
fn null_diagonal<A, R, D>(
    authority: &A,
    space: &DynamicFusionMapSpace,
    regions: &[CoupledSectorRegion],
    spectrum: &[SectorSpectrum<D>],
    side: FactorSide,
) -> Result<DiagonalNull<R, D, A::Error>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    D: FactorScalar,
{
    let by_sector: FxHashMap<_, _> = spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    if by_sector.len() != regions.len() || spectrum.len() != regions.len() {
        return Ok(DiagonalNull::NotDiagonal);
    }
    let mut null_dimensions = match authority.coupled_dimensions(match side {
        FactorSide::Left => space.homspace().codomain(),
        FactorSide::Right => space.homspace().domain(),
    }) {
        Ok(dimensions) => dimensions,
        Err(error) => return Ok(DiagonalNull::Declined(Err(error))),
    };
    let mut pairs = Vec::new();
    for region in regions {
        let k = region.rows();
        let Some(entry) = by_sector.get(&region.coupled()) else {
            return Ok(DiagonalNull::Declined(Ok(null_dimensions)));
        };
        if !region.has_aligned_diagonal() || k != region.cols() || entry.values.len() != k {
            return Ok(DiagonalNull::Declined(Ok(null_dimensions)));
        }
        let Some((nullity, coordinates)) = compact_null_sector(&entry.values, side) else {
            return Ok(DiagonalNull::Declined(Ok(null_dimensions)));
        };
        if nullity == 0 {
            continue;
        }
        let (left_data, right_data) = match side {
            FactorSide::Left => (coordinates, Vec::new()),
            FactorSide::Right => (Vec::new(), coordinates),
        };
        pairs.push(FactorPair {
            sector: region.coupled(),
            kept: nullity,
            left: left_data,
            left_rows: k,
            right: right_data,
            right_leading: nullity,
        });
    }
    // The dimensions change only once every sector is admitted: a decline
    // hands the solver path the queried dimensions unmodified.
    let mut kept = pairs.iter().peekable();
    for region in regions {
        match kept.next_if(|pair| pair.sector == region.coupled()) {
            Some(pair) => null_dimensions.insert(pair.sector, pair.kept),
            None => null_dimensions.remove(&region.coupled()),
        };
    }
    Ok(DiagonalNull::Direct(publish_one_sided_factor(
        authority,
        space.homspace(),
        regions,
        &mut pairs,
        &null_dimensions,
        side,
        FactorPlacement::Direct,
    )?))
}

fn null_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    side: FactorSide,
) -> Result<Option<BoundDynFactor<R, D>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = authority.space();
    let Some(regions) = checked_sector_regions(space.structure(), space.nout())? else {
        return Ok(None);
    };
    match null_diagonal(
        &MfAuthority(authority),
        space,
        regions.as_ref(),
        spectrum,
        side,
    )? {
        DiagonalNull::Direct(factor) => Ok(Some(factor)),
        DiagonalNull::NotDiagonal | DiagonalNull::Declined(Ok(_)) => Ok(None),
        DiagonalNull::Declined(Err(error)) => Err(error),
    }
}

pub type CheckedNullDimensions<E> =
    Result<BTreeMap<SectorId, usize>, CheckedGenericFactorPlanError<E>>;

#[doc(hidden)]
pub enum CheckedDiagonalNullFactor<R, D, E> {
    Direct(BoundDynFactor<R, D>),
    Fallback(Option<CheckedNullDimensions<E>>),
}

fn null_diagonal_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    side: FactorSide,
) -> Result<CheckedDiagonalNullFactor<R, D, R::Error>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let space = authority.space();
    let Ok(Some(regions)) = checked_sector_regions(space.structure(), space.nout()) else {
        return Ok(CheckedDiagonalNullFactor::Fallback(None));
    };
    let checked = CheckedAuthority(authority.provider_arc());
    Ok(
        match null_diagonal(&checked, space, regions.as_ref(), spectrum, side)? {
            DiagonalNull::Direct(factor) => CheckedDiagonalNullFactor::Direct(factor),
            DiagonalNull::NotDiagonal => CheckedDiagonalNullFactor::Fallback(None),
            DiagonalNull::Declined(dimensions) => {
                CheckedDiagonalNullFactor::Fallback(Some(dimensions))
            }
        },
    )
}

#[doc(hidden)]
pub fn left_null_diagonal_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<CheckedDiagonalNullFactor<R, D, R::Error>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    null_diagonal_dyn_checked_generic(authority, spectrum, FactorSide::Left)
}

#[doc(hidden)]
pub fn right_null_diagonal_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<CheckedDiagonalNullFactor<R, D, R::Error>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    null_diagonal_dyn_checked_generic(authority, spectrum, FactorSide::Right)
}

#[doc(hidden)]
pub fn left_null_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<BoundDynFactor<R, D>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    null_diagonal_dyn(authority, spectrum, FactorSide::Left)
}

#[doc(hidden)]
pub fn right_null_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<BoundDynFactor<R, D>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    null_diagonal_dyn(authority, spectrum, FactorSide::Right)
}

#[cfg(test)]
/// Left null space `N : codomain <- W` (MatrixAlgebraKit `left_null`).
///
/// Each sector uses its compact SVD and treats `sigma` as nonzero exactly when
/// `sigma > epsilon(dtype) * max(rows, cols) * sigma_max`. The returned columns
/// are the orthonormal complement after that numerical rank; sectors with no
/// null directions drop out of `W`.
pub(crate) fn left_null<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<BoundTensorMap<R, D, NOUT, 1>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let out = left_null_dyn(dense, &input.dynamic())?;
    typed_from_bound_factor(out)
}

/// Provider-bound dynamic-rank [`left_null`].
pub fn left_null_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    null_dense(
        dense,
        &MfAuthority(input.space()),
        input,
        FactorSide::Left,
        None,
    )
}

#[cfg(test)]
/// Right null space `N : W <- domain` (MatrixAlgebraKit `right_null`).
///
/// Each sector uses its compact SVD and treats `sigma` as nonzero exactly when
/// `sigma > epsilon(dtype) * max(rows, cols) * sigma_max`. The returned rows
/// span the kernel after that numerical rank; sectors with no null directions
/// drop out of `W`.
pub(crate) fn right_null<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<BoundTensorMap<R, D, 1, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let out = right_null_dyn(dense, &input.dynamic())?;
    typed_from_bound_factor(out)
}

/// Provider-bound dynamic-rank [`right_null`].
pub fn right_null_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    null_dense(
        dense,
        &MfAuthority(input.space()),
        input,
        FactorSide::Right,
        None,
    )
}

#[cfg(test)]
/// Checked-Generic numerical left null space.
///
/// Structural dimensions are validated before dense work. All SVDs and
/// completions are then staged before the exact data-dependent bond is
/// admitted and scattered by the shared checked factor builder.
pub(crate) fn left_null_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    left_null_dyn_checked_generic_with_dimensions(dense, input, None)
}

#[doc(hidden)]
pub fn left_null_dyn_checked_generic_with_dimensions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    dimensions: Option<CheckedNullDimensions<R::Error>>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    null_dense(
        dense,
        &CheckedAuthority(input.space().provider_arc()),
        input,
        FactorSide::Left,
        dimensions,
    )
}

#[cfg(test)]
/// Checked-Generic numerical right null space; see
/// [`left_null_dyn_checked_generic`] for the transaction boundary.
pub(crate) fn right_null_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    right_null_dyn_checked_generic_with_dimensions(dense, input, None)
}

#[doc(hidden)]
pub fn right_null_dyn_checked_generic_with_dimensions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    dimensions: Option<CheckedNullDimensions<R::Error>>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    null_dense(
        dense,
        &CheckedAuthority(input.space().provider_arc()),
        input,
        FactorSide::Right,
        dimensions,
    )
}

/// Numerical null space of every coupled sector of `input`, published through
/// `authority`. `dimensions` is a coupled-dimension query the caller already
/// ran (the checked compact-diagonal decline); `None` queries here.
fn null_dense<A, E, R, D>(
    dense: &mut E,
    authority: &A,
    input: &BoundDynamicTensorRef<'_, R, D>,
    side: FactorSide,
    dimensions: Option<Result<BTreeMap<SectorId, usize>, A::Error>>,
) -> Result<BoundDynFactor<R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())?;
    // A sector only on the null side has no tensor block but is entirely null.
    let mut null_dimensions = match dimensions {
        Some(dimensions) => dimensions?,
        None => authority.coupled_dimensions(match side {
            FactorSide::Left => space.homspace().codomain(),
            FactorSide::Right => space.homspace().domain(),
        })?,
    };
    let mut pairs = Vec::new();
    in_linalg_scope(dense, |dense| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            let (rows, cols) = (matrix.rows, matrix.cols);
            let (rank, compact) =
                numerical_rank_and_compact_basis(dense, matrix.data, rows, cols, side)?;
            let extent = match side {
                FactorSide::Left => rows,
                FactorSide::Right => cols,
            };
            if rank == extent {
                null_dimensions.remove(&matrix.sector);
                continue;
            }
            // Only the null side's basis is completed: completing the other
            // one would run an unused QR for this operation.
            let basis = orthonormal_completion(dense, &compact, extent, rows.min(cols))?;
            let null_dim = extent - rank;
            null_dimensions.insert(matrix.sector, null_dim);
            let (left, right) = match side {
                FactorSide::Left => (basis[rows * rank..].to_vec(), Vec::new()),
                FactorSide::Right => (
                    Vec::new(),
                    adjoint_col_major(&basis[cols * rank..], cols, null_dim),
                ),
            };
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: null_dim,
                left,
                left_rows: rows,
                right,
                right_leading: null_dim,
            });
        }
        Ok(())
    })?;
    with_input_geometry!(&matrices, |geometry| publish_one_sided_factor(
        authority,
        space.homspace(),
        geometry,
        &mut pairs,
        &null_dimensions,
        side,
        FactorPlacement::Direct,
    ))
}

/// Computes the requested compact singular-vector basis and the documented numerical rank.
pub(super) fn numerical_rank_and_compact_basis<E, D>(
    dense: &mut E,
    matrix: &[D],
    rows: usize,
    cols: usize,
    side: FactorSide,
) -> Result<(usize, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let compact_rank = rows.min(cols);
    let (u, singular_values, vh) = compact_svd_owned(dense, matrix, rows, cols)?;

    let sigma_max = singular_values.first().copied().unwrap_or(0.0);
    // Why not exact-zero rank: backward-stable SVD represents dependent
    // directions at working precision, not necessarily as bitwise zero.
    let tolerance = D::epsilon() * rows.max(cols) as f64 * sigma_max;
    let rank = singular_values
        .iter()
        .copied()
        .filter(|&sigma| sigma > tolerance)
        .count();
    drop(singular_values);
    let basis = match side {
        FactorSide::Left => u,
        FactorSide::Right => {
            drop(u);
            adjoint_col_major(&vh, compact_rank, cols)
        }
    };
    Ok((rank, basis))
}
