use super::*;

fn compact_singular_value<D: FactorScalar>(value: D) -> Option<f64> {
    let magnitude = finite_compact_magnitude(value)?;
    let rounded = D::from_real(magnitude).widen_complex().re;
    (rounded.is_finite() && (magnitude == 0.0 || rounded > 0.0)).then_some(rounded)
}

/// Coordinate kernels of an admitted compact diagonal. A sector with any
/// positive magnitude near the rank cutoff stays on the solver path; the
/// margin is conservative, not a provider-specific singular-value bound.
fn null_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    left: bool,
) -> Result<Option<BoundDynFactor<R, D>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = authority.space();
    let Some(regions) = checked_sector_regions(space.structure(), space.nout())? else {
        return Ok(None);
    };
    let by_sector: FxHashMap<_, _> = spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    if by_sector.len() != regions.len() || spectrum.len() != regions.len() {
        return Ok(None);
    }
    let mut null_dimensions = if left {
        space
            .homspace()
            .codomain()
            .coupled_sector_block_dimensions(authority.provider())?
    } else {
        space
            .homspace()
            .domain()
            .coupled_sector_block_dimensions(authority.provider())?
    };
    let mut pairs = Vec::new();
    for region in regions.iter() {
        let k = region.rows();
        let Some(entry) = by_sector.get(&region.coupled()) else {
            return Ok(None);
        };
        if !region.has_aligned_diagonal() || k != region.cols() || entry.values.len() != k {
            return Ok(None);
        }
        let Some((sigma_max, q)) =
            entry
                .values
                .iter()
                .copied()
                .try_fold((0.0_f64, 0_usize), |(largest, zeros), value| {
                    compact_singular_value(value).map(|magnitude| {
                        (
                            largest.max(magnitude),
                            zeros + usize::from(magnitude == 0.0),
                        )
                    })
                })
        else {
            return Ok(None);
        };
        let cutoff = D::epsilon() * k as f64 * sigma_max;
        let margin = cutoff.max(D::epsilon().sqrt() * sigma_max);
        if !margin.is_finite()
            || (sigma_max > 0.0 && (margin == 0.0 || sigma_max < D::safe_minimum()))
        {
            return Ok(None);
        }
        for &value in &entry.values {
            let Some(magnitude) = compact_singular_value(value) else {
                return Ok(None);
            };
            if magnitude > 0.0 && magnitude <= margin {
                return Ok(None);
            }
        }
        if q == 0 {
            null_dimensions.remove(&region.coupled());
            continue;
        }
        null_dimensions.insert(region.coupled(), q);
        let mut coordinates = vec![D::zero(); k * q];
        let mut column = 0;
        for (index, &value) in entry.values.iter().enumerate() {
            if compact_singular_value(value) != Some(0.0) {
                continue;
            }
            if left {
                coordinates[index + column * k] = D::from_real(1.0);
            } else {
                coordinates[column + index * q] = D::from_real(1.0);
            }
            column += 1;
        }
        let (left_data, right_data) = if left {
            (coordinates, Vec::new())
        } else {
            (Vec::new(), coordinates)
        };
        pairs.push(FactorPair {
            sector: region.coupled(),
            kept: q,
            left: left_data,
            left_rows: k,
            right: right_data,
            right_leading: q,
        });
    }
    let side = if left {
        FactorSide::Left
    } else {
        FactorSide::Right
    };
    Ok(Some(build_bound_factor(
        authority,
        space.homspace(),
        regions.as_ref(),
        &mut pairs,
        &null_dimensions,
        side,
    )?))
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
    null_diagonal_dyn(authority, spectrum, true)
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
    null_diagonal_dyn(authority, spectrum, false)
}

/// Left null space `N : codomain <- W` (MatrixAlgebraKit `left_null`).
///
/// Each sector uses its compact SVD and treats `sigma` as nonzero exactly when
/// `sigma > epsilon(dtype) * max(rows, cols) * sigma_max`. The returned columns
/// are the orthonormal complement after that numerical rank; sectors with no
/// null directions drop out of `W`.
pub fn left_null<E, R, D, const NOUT: usize, const NIN: usize>(
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
    let space = input.space().space();
    let matrices =
        multiplicity_free_input_matricizations(space.structure(), input.data(), space.nout())?;
    // A codomain-only sector has no tensor block but is entirely left-null.
    let mut null_dimensions = space
        .homspace()
        .codomain()
        .coupled_sector_block_dimensions(input.space().provider())?;
    let mut pairs = Vec::new();
    in_linalg_scope(dense, |dense| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            let (rows, cols) = (matrix.rows, matrix.cols);
            let (rank, u_compact) =
                numerical_rank_and_compact_basis(dense, matrix.data, rows, cols, FactorSide::Left)?;
            if rank == rows {
                null_dimensions.remove(&matrix.sector);
                continue;
            }
            // Only the left basis is completed: completing V would run an unused
            // QR for this operation.
            let u = orthonormal_completion(dense, &u_compact, rows, rows.min(cols))?;
            let null_dim = rows - rank;
            null_dimensions.insert(matrix.sector, null_dim);
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: null_dim,
                left: u[rows * rank..].to_vec(),
                left_rows: rows,
                right: Vec::new(),
                right_leading: null_dim,
            });
        }
        Ok(())
    })?;
    with_input_geometry!(&matrices, |geometry| build_bound_factor(
        input.space(),
        space.homspace(),
        geometry,
        &mut pairs,
        &null_dimensions,
        FactorSide::Left,
    ))
}

/// Right null space `N : W <- domain` (MatrixAlgebraKit `right_null`).
///
/// Each sector uses its compact SVD and treats `sigma` as nonzero exactly when
/// `sigma > epsilon(dtype) * max(rows, cols) * sigma_max`. The returned rows
/// span the kernel after that numerical rank; sectors with no null directions
/// drop out of `W`.
pub fn right_null<E, R, D, const NOUT: usize, const NIN: usize>(
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
    let space = input.space().space();
    let matrices =
        multiplicity_free_input_matricizations(space.structure(), input.data(), space.nout())?;
    // A domain-only sector has no tensor block but is entirely right-null.
    let mut null_dimensions = space
        .homspace()
        .domain()
        .coupled_sector_block_dimensions(input.space().provider())?;
    let mut pairs = Vec::new();
    in_linalg_scope(dense, |dense| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            let (rows, cols) = (matrix.rows, matrix.cols);
            let (rank, v_compact) = numerical_rank_and_compact_basis(
                dense,
                matrix.data,
                rows,
                cols,
                FactorSide::Right,
            )?;
            if rank == cols {
                null_dimensions.remove(&matrix.sector);
                continue;
            }
            // Only the right basis is completed: completing U would run an unused
            // QR for this operation.
            let v = orthonormal_completion(dense, &v_compact, cols, rows.min(cols))?;
            let null_dim = cols - rank;
            null_dimensions.insert(matrix.sector, null_dim);
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: null_dim,
                left: Vec::new(),
                left_rows: rows,
                right: adjoint_col_major(&v[cols * rank..], cols, null_dim),
                right_leading: null_dim,
            });
        }
        Ok(())
    })?;
    with_input_geometry!(&matrices, |geometry| build_bound_factor(
        input.space(),
        space.homspace(),
        geometry,
        &mut pairs,
        &null_dimensions,
        FactorSide::Right,
    ))
}

/// Checked-Generic numerical left null space.
///
/// Structural dimensions are validated before dense work. All SVDs and
/// completions are then staged before the exact data-dependent bond is
/// admitted and scattered by the shared checked factor builder.
#[doc(hidden)]
pub fn left_null_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut null_dimensions = coupled_sector_block_dimensions_generic_checked(
        space.homspace().codomain(),
        provider.as_ref(),
    )?;
    let mut pairs = Vec::new();
    in_linalg_scope(dense, |dense| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            let (rank, u_compact) = numerical_rank_and_compact_basis(
                dense,
                matrix.data,
                matrix.rows,
                matrix.cols,
                FactorSide::Left,
            )?;
            if rank == matrix.rows {
                null_dimensions.remove(&matrix.sector);
                continue;
            }
            let u = orthonormal_completion(
                dense,
                &u_compact,
                matrix.rows,
                matrix.rows.min(matrix.cols),
            )?;
            let null_dim = matrix.rows - rank;
            null_dimensions.insert(matrix.sector, null_dim);
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: null_dim,
                left: u[matrix.rows * rank..].to_vec(),
                left_rows: matrix.rows,
                right: Vec::new(),
                right_leading: null_dim,
            });
        }
        Ok(())
    })
    .map_err(CheckedGenericFactorPlanError::from)?;
    with_input_geometry!(&matrices, |geometry| build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        geometry,
        &mut pairs,
        &null_dimensions,
        FactorSide::Left,
    ))
}

/// Checked-Generic numerical right null space; see
/// [`left_null_dyn_checked_generic`] for the transaction boundary.
#[doc(hidden)]
pub fn right_null_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut null_dimensions = coupled_sector_block_dimensions_generic_checked(
        space.homspace().domain(),
        provider.as_ref(),
    )?;
    let mut pairs = Vec::new();
    in_linalg_scope(dense, |dense| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            let (rank, v_compact) = numerical_rank_and_compact_basis(
                dense,
                matrix.data,
                matrix.rows,
                matrix.cols,
                FactorSide::Right,
            )?;
            if rank == matrix.cols {
                null_dimensions.remove(&matrix.sector);
                continue;
            }
            let v = orthonormal_completion(
                dense,
                &v_compact,
                matrix.cols,
                matrix.rows.min(matrix.cols),
            )?;
            let null_dim = matrix.cols - rank;
            null_dimensions.insert(matrix.sector, null_dim);
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: null_dim,
                left: Vec::new(),
                left_rows: matrix.rows,
                right: adjoint_col_major(&v[matrix.cols * rank..], matrix.cols, null_dim),
                right_leading: null_dim,
            });
        }
        Ok(())
    })
    .map_err(CheckedGenericFactorPlanError::from)?;
    with_input_geometry!(&matrices, |geometry| build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        geometry,
        &mut pairs,
        &null_dimensions,
        FactorSide::Right,
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
