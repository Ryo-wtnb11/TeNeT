use super::*;

/// Null-space coordinates of one finite diagonal sector, by the dense
/// route's numerical-rank rule applied to the singular values `|a_i|`
/// directly.
///
/// The dense route keeps singular values above
/// `epsilon * max(rows, cols) * sigma_max` and spans the null space with the
/// remaining singular vectors in descending singular-value order. On a
/// diagonal those vectors are unit vectors, so the sector's null basis is
/// `e_i` for each `|a_i|` at or below the tolerance, in descending `|a_i|`
/// (stable).
fn compact_null_sector<D: FactorScalar>(values: &[D], side: FactorSide) -> (usize, Vec<D>) {
    let k = values.len();
    let sigma: Vec<f64> = values
        .iter()
        .map(|&value| {
            D::from_real(value.widen_complex().norm())
                .widen_complex()
                .re
        })
        .collect();
    let sigma_max = sigma.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = D::epsilon() * k as f64 * sigma_max;
    let mut null: Vec<usize> = (0..k).filter(|&index| sigma[index] <= tolerance).collect();
    null.sort_by(|&a, &b| sigma[b].total_cmp(&sigma[a]));
    let nullity = null.len();
    let mut coordinates = vec![D::zero(); k * nullity];
    for (column, &index) in null.iter().enumerate() {
        match side {
            FactorSide::Left => coordinates[index + column * k] = D::from_real(1.0),
            FactorSide::Right => coordinates[column + index * nullity] = D::from_real(1.0),
        }
    }
    (nullity, coordinates)
}

/// Null space of a compact diagonal, published through `authority`.
fn null_diagonal<A, R, D>(
    authority: &A,
    space: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    side: FactorSide,
) -> Result<BoundDynFactor<R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
    D: FactorScalar,
{
    let bond = diagonal_bond(authority, space, spectrum, FactorFamily::Null)?;
    let homspace = bond.space().space().homspace();
    let mut null_dimensions = authority.coupled_dimensions(match side {
        FactorSide::Left => homspace.codomain(),
        FactorSide::Right => homspace.domain(),
    })?;
    let mut pairs = Vec::new();
    for region in bond.iter() {
        let k = region.rows();
        let (nullity, coordinates) = compact_null_sector(&bond.entry(region).values, side);
        if nullity == 0 {
            null_dimensions.remove(&region.coupled());
            continue;
        }
        null_dimensions.insert(region.coupled(), nullity);
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
    publish_one_sided_factor(
        authority,
        homspace,
        &bond.regions,
        &mut pairs,
        &null_dimensions,
        side,
        FactorPlacement::Direct,
    )
}

/// Multiplicity-free left null space of a compact diagonal; see
/// [`compact_null_sector`].
#[doc(hidden)]
pub fn left_null_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    null_diagonal(
        &MfAuthority(authority),
        authority,
        spectrum,
        FactorSide::Left,
    )
}

/// Multiplicity-free right null space of a compact diagonal.
#[doc(hidden)]
pub fn right_null_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    null_diagonal(
        &MfAuthority(authority),
        authority,
        spectrum,
        FactorSide::Right,
    )
}

fn null_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
    side: FactorSide,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    factor_from_source(
        lease,
        source,
        |space, spectrum| {
            null_diagonal(
                &CheckedAuthority(space.provider_arc()),
                space,
                spectrum,
                side,
            )
        },
        |dense, input| {
            null_dense(
                dense,
                &CheckedAuthority(input.space().provider_arc()),
                input,
                side,
            )
        },
    )
    .map(|(factor, _)| factor)
}

/// Checked numerical left null space of `source`. All SVDs and completions
/// of a dense input are staged before the data-dependent bond is admitted.
#[doc(hidden)]
pub fn left_null_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    null_checked_generic(lease, source, FactorSide::Left)
}

/// Checked numerical right null space of `source`; see
/// [`left_null_checked_generic`].
#[doc(hidden)]
pub fn right_null_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    null_checked_generic(lease, source, FactorSide::Right)
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
    null_dense(dense, &MfAuthority(input.space()), input, FactorSide::Left)
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
    null_dense(dense, &MfAuthority(input.space()), input, FactorSide::Right)
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
    null_dense(
        dense,
        &CheckedAuthority(input.space().provider_arc()),
        input,
        FactorSide::Left,
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
    null_dense(
        dense,
        &CheckedAuthority(input.space().provider_arc()),
        input,
        FactorSide::Right,
    )
}

/// Numerical null space of every coupled sector of `input`, published through
/// `authority`.
fn null_dense<A, E, R, D>(
    dense: &mut E,
    authority: &A,
    input: &BoundDynamicTensorRef<'_, R, D>,
    side: FactorSide,
) -> Result<BoundDynFactor<R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())?;
    // A sector only on the null side has no tensor block but is entirely null.
    let mut null_dimensions = authority.coupled_dimensions(match side {
        FactorSide::Left => space.homspace().codomain(),
        FactorSide::Right => space.homspace().domain(),
    })?;
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
