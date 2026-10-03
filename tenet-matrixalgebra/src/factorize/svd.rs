use super::*;

#[cfg(test)]
/// Compact (thin, untruncated) fusion-tensor SVD `t = U * S * Vh`
/// (MatrixAlgebraKit `svd_compact`).
///
/// This is the pure device-boundary factorization: the dense per-sector SVDs
/// run through the [`DenseExecutor`] and no truncation logic is involved.
/// Per block the bond is `min(rows, cols)`; the square-`U` variant is
/// MatrixAlgebraKit `svd_full` (later batch).
#[derive(Clone, Debug)]
pub(crate) struct SvdCompact<R, D, const NOUT: usize, const NIN: usize> {
    pub u: BoundTensorMap<R, D, NOUT, 1>,
    pub s: BoundTensorMap<R, D, 1, 1>,
    pub vh: BoundTensorMap<R, D, 1, NIN>,
    pub singular_values: Vec<SectorSpectrum>,
}

#[cfg(test)]
/// Dynamic-rank [`SvdCompact`].
#[derive(Clone, Debug)]
pub(crate) struct SvdCompactDyn<R, D> {
    pub(super) u: BoundDynFactor<R, D>,
    pub(super) s: BoundDynFactor<R, D>,
    pub(super) vh: BoundDynFactor<R, D>,
    pub(super) singular_values: Vec<SectorSpectrum>,
}

#[cfg(test)]
impl<R, D> SvdCompactDyn<R, D> {
    pub fn u(&self) -> &BoundDynFactor<R, D> {
        &self.u
    }

    pub fn s(&self) -> &BoundDynFactor<R, D> {
        &self.s
    }

    pub fn vh(&self) -> &BoundDynFactor<R, D> {
        &self.vh
    }

    pub fn singular_values(&self) -> &[SectorSpectrum] {
        &self.singular_values
    }
}

#[cfg(test)]
pub(super) fn diagonal_bond_svd_factor<R, D, V>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
    V: Copy,
{
    #[cfg(test)]
    record_diagonal_bond_build(spectrum);
    let space = diagonal_bond_bound_space_like(authority, spectrum)?;
    let data = diagonal_bond_data(space.space(), spectrum, to_scalar)?;
    BoundDynFactor::from_bound(space, data, 1, 1)
}

#[doc(hidden)]
pub fn diagonal_bond_bound_space_like<R, V>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<V>],
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let new_leg = SectorLeg::new(
        spectrum
            .iter()
            .map(|entry| (entry.sector, entry.values.len())),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg.clone()]),
        FusionProductSpace::new([new_leg]),
    );
    authority.derive_from_final_homspace(homspace)
}

/// Fills the dense block-diagonal data of `space` from `spectrum`, mapping
/// each value through `to_scalar`. Only the
/// per-block diagonal is written; the rest stays zero. Bit-for-bit identical to
/// the fill inside the former monolithic `diagonal_bond_tensor_dyn`.
#[doc(hidden)]
pub fn diagonal_bond_data<D, V>(
    space: &DynamicFusionMapSpace,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<Vec<D>, OperationError>
where
    D: FactorScalar,
    V: Copy,
{
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum<V>> =
        spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    let len = space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); len];
    let structure = Arc::clone(space.structure());
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree.codomain_tree().coupled();
        let Some(&entry) = spectrum_by_sector.get(&sector) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        copy_mapped_to_strided_diagonal(
            &mut data,
            offset,
            strides[0] + strides[1],
            &entry.values[..count],
            to_scalar,
        );
    }
    Ok(data)
}

/// Scales one bond axis of `data` (laid out per `space`) by the per-sector
/// `spectrum`, in place — the block-local realization of TensorKit's
/// `DiagonalTensorMap` multiplication. `axis = None` scales each block's
/// trailing axis (`t * D`, `rmul!`, column scaling); `axis = Some(0)` scales
/// the leading axis (`D * t`, `lmul!`, row scaling). Verified twist-free
/// against TK `diagonal.jl`: diagonal multiplication is pure per-block scaling
/// with no braiding or fusion-tree recoupling (`block(D, c)` is a `Diagonal`,
/// so LinearAlgebra dispatches to scaling, not GEMM). A real `spectrum` on a
/// complex `data` promotes each entry the same way (`D::from_real`).
pub fn scale_axis_by_spectrum<D>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    axis: Option<usize>,
    spectrum: &[SectorSpectrum],
) -> Result<(), OperationError>
where
    D: FactorScalar,
{
    scale_axis_by_spectrum_mapped(space, data, axis, spectrum, D::from_real)
}

/// Value-generic sibling of [`scale_axis_by_spectrum`]. Why not convert the
/// spectrum before this call: a complex spectrum cannot pass through the
/// real-only `SectorSpectrum` alias without losing its imaginary component.
pub fn scale_axis_by_spectrum_mapped<D, V>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    axis: Option<usize>,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: impl Fn(V) -> D,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    V: Copy,
{
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum<V>> =
        spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    let nout = space.nout();
    let structure = Arc::clone(space.structure());
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let shape = block.shape();
        if shape.is_empty() {
            continue;
        }
        let strides = block.strides();
        let offset = block.offset();
        let bond_axis = axis.unwrap_or(shape.len() - 1);
        // Index the spectrum by the charge ON THE SCALED LEG — its uncoupled
        // charge in this block's fusion tree — NOT the block's coupled charge.
        // For an SVD/eigh factor's sole bond leg the two coincide, but scaling a
        // general tensor leg (diagonal-aware `contract`, #75) is only correct per
        // leg charge.
        let leg_charge = if bond_axis < nout {
            tree.codomain_tree().uncoupled()[bond_axis]
        } else {
            tree.domain_tree().uncoupled()[bond_axis - nout]
        };
        // Absent charge => this leg slice is structurally zero for the spectrum;
        // nothing to scale (mirrors `diagonal_bond_tensor_dyn`'s `unwrap_or(0)`).
        let Some(&entry) = spectrum_by_sector.get(&leg_charge) else {
            continue;
        };
        let bond = shape[bond_axis];
        let bond_stride = strides[bond_axis];
        debug_assert_eq!(
            bond,
            entry.values.len(),
            "bond degeneracy must match the spectrum length"
        );
        let bond = bond.min(entry.values.len());
        // Walk every combination of the non-bond axes; for each, scale the
        // `bond` entries along `bond_axis` by the spectrum.
        let lead_axes: Vec<usize> = (0..shape.len()).filter(|&a| a != bond_axis).collect();
        let outer: usize = lead_axes.iter().map(|&a| shape[a]).product();
        let mut coord = vec![0usize; lead_axes.len()];
        for _ in 0..outer {
            let mut base = offset;
            for (k, &a) in lead_axes.iter().enumerate() {
                base += coord[k] * strides[a];
            }
            for j in 0..bond {
                let scale = to_scalar(entry.values[j]);
                let idx = base + j * bond_stride;
                data[idx] = data[idx] * scale;
            }
            for k in (0..coord.len()).rev() {
                coord[k] += 1;
                if coord[k] < shape[lead_axes[k]] {
                    break;
                }
                coord[k] = 0;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
/// All singular values per coupled sector, descending (MatrixAlgebraKit
/// `svd_vals`). Runs the dense SVD per sector through the executor and keeps
/// only the spectra.
pub(crate) fn svd_vals<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    svd_vals_dyn(dense, &input.dynamic())
}

/// Dynamic-rank [`svd_vals`].
pub fn svd_vals_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    // Values-only: per coupled sector call the no-vector SVD (`svd_vals`,
    // LAPACK `job='N'`) and keep the spectrum. Unlike `svd_compact_dyn` this
    // never builds the U/Vt spaces, allocates the factor buffers, gauge-fixes,
    // or scatters blocks into the fusion-tree layout — all of which the old
    // `svd_compact_dyn(..).map(|svd| svd.singular_values)` computed then threw
    // away. Valid no-vector and full-factor drivers can differ in the last
    // bits, so comparisons use dtype-appropriate tolerances.
    let matricizations = value_matricizations(space.structure(), input.data(), space.nout())?;
    let mut singular_values = Vec::with_capacity(matricizations.len());
    for index in 0..matricizations.len() {
        let matrix = matricizations.get(index)?;
        let rank = matrix.rows.min(matrix.cols);
        let input_shape = [matrix.rows, matrix.cols];
        let input_strides = [1usize, matrix.rows];
        let input = DenseView::new(matrix.data, &input_shape, &input_strides, 0)
            .map_err(OperationError::Dense)?;
        let s_tensor = dense
            .svd_vals(D::dense_read(input))
            .map_err(OperationError::Dense)?;
        let mut s = D::real_spectrum(&s_tensor).map_err(OperationError::Dense)?;
        s.truncate(rank);
        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s,
        });
    }
    Ok(singular_values)
}

pub(super) fn finite_compact_magnitude<D: FactorScalar>(value: D) -> Option<f64> {
    let magnitude = value.widen_complex().norm();
    (magnitude.is_finite() && D::from_real(magnitude).widen_complex().re.is_finite())
        .then_some(magnitude)
}

/// Values-only path for an admitted compact diagonal tensor. Returns `None`
/// when the ordinary dense solver must retain its input or layout behavior.
#[doc(hidden)]
pub fn svd_vals_compact_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<Vec<SectorSpectrum>>, OperationError>
where
    D: FactorScalar,
{
    let Some(regions) =
        checked_sector_regions(authority.space().structure(), authority.space().nout())?
    else {
        return Ok(None);
    };
    let Some(by_sector) = aligned_diagonal_spectrum_by_sector(&regions, spectrum) else {
        return Ok(None);
    };
    let mut values = Vec::with_capacity(regions.len());
    for region in regions.iter() {
        let entry = by_sector[&region.coupled()];
        let mut sorted = Vec::with_capacity(entry.values.len());
        for &value in &entry.values {
            let Some(magnitude) = finite_compact_magnitude(value) else {
                return Ok(None);
            };
            sorted.push(D::from_real(magnitude).widen_complex().re);
        }
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        values.push(SectorSpectrum {
            sector: region.coupled(),
            values: sorted,
        });
    }
    Ok(Some(values))
}

#[cfg(test)]
/// Compact (untruncated) fusion-tensor SVD through the device boundary.
pub(crate) fn svd_compact<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<SvdCompact<R, D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let out = svd_compact_dyn(dense, &input.dynamic())?;
    Ok(SvdCompact {
        u: typed_from_bound_factor(out.u)?,
        s: typed_from_bound_factor(out.s)?,
        vh: typed_from_bound_factor(out.vh)?,
        singular_values: out.singular_values,
    })
}

/// The compact-SVD factors without materializing the diagonal `S`:
/// `(U, Vh, spectrum)`. The shared core of every SVD entry point.
/// [`svd_compact_dyn`] wraps this and adds the dense `S` as a tensor for callers
/// that want it; polar and the matrix-function paths scale by the spectrum
/// directly (TensorKit `DiagonalTensorMap` `rmul!`) and never build `S`.
pub type SvdFactorsDyn<R, D> = (
    BoundDynFactor<R, D>,
    BoundDynFactor<R, D>,
    Vec<SectorSpectrum>,
);

/// Compact diagonal input: sort each sector's magnitudes and write the
/// permutation/phase factors directly into the existing dense factor layout.
/// Nonfinite or unrepresentable magnitudes, and unsupported region layouts,
/// retain the ordinary dense-SVD path and its error behavior.
#[doc(hidden)]
pub fn svd_compact_diagonal_factors_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Option<SvdFactorsDyn<R, D>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    if spectrum
        .iter()
        .flat_map(|entry| &entry.values)
        .any(|&value| finite_compact_magnitude(value).is_none())
    {
        return Ok(None);
    }
    let Some(plan) = compact_factor_plan(authority)? else {
        return Ok(None);
    };
    let (u_data, vh_data, singular_values) = compact_diagonal_svd_factor_data(
        &plan.routes,
        plan.left_regions.len(),
        plan.right_regions.len(),
        plan.left_layout.required_len()?,
        plan.right_layout.required_len()?,
        spectrum,
    )?;
    let u = BoundDynFactor::from_bound(
        authority.rebind_validated(&plan.left_layout)?,
        u_data,
        authority.space().nout(),
        1,
    )?;
    let vh = BoundDynFactor::from_bound(
        authority.rebind_validated(&plan.right_layout)?,
        vh_data,
        1,
        authority.space().nin(),
    )?;
    Ok(Some((u, vh, singular_values)))
}

#[expect(clippy::type_complexity)]
fn compact_diagonal_svd_factor_data<D: FactorScalar>(
    routes: &[CompactFactorRoute],
    left_region_count: usize,
    right_region_count: usize,
    left_len: usize,
    right_len: usize,
    spectrum: &[SectorSpectrum<D>],
) -> Result<(Vec<D>, Vec<D>, Vec<SectorSpectrum>), OperationError> {
    let by_sector: FxHashMap<_, _> = spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    if by_sector.len() != spectrum.len() || spectrum.len() != routes.len() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "compact diagonal spectrum does not match source sectors",
        });
    }
    let mut u_regions = vec![None; left_region_count];
    let mut vh_regions = vec![None; right_region_count];
    let mut singular_values = Vec::with_capacity(routes.len());
    for route in routes {
        let entry =
            by_sector
                .get(&route.sector)
                .ok_or(OperationError::UnsupportedTensorContractScope {
                    message: "compact diagonal spectrum is missing a source sector",
                })?;
        let k = route.rank;
        if entry.values.len() != k {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "compact diagonal spectrum length does not match source region",
            });
        }
        let (u, vh, values) = compact_diagonal_svd_sector(&entry.values);
        if k != 0 {
            u_regions[route.left_region.expect("nonzero route has left region")] = Some(u);
            vh_regions[route.right_region.expect("nonzero route has right region")] = Some(vh);
        }
        singular_values.push(SectorSpectrum {
            sector: route.sector,
            values,
        });
    }
    let u_data = concat_compact_svd_factor_regions(u_regions, left_len);
    let vh_data = concat_compact_svd_factor_regions(vh_regions, right_len);
    Ok((u_data, vh_data, singular_values))
}

fn compact_diagonal_svd_sector<D: FactorScalar>(values: &[D]) -> (Vec<D>, Vec<D>, Vec<f64>) {
    let k = values.len();
    let mut order = values
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value.widen_complex().norm()))
        .collect::<Vec<_>>();
    order.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut u = vec![D::zero(); k * k];
    let mut vh = vec![D::zero(); k * k];
    for (column, &(source, magnitude)) in order.iter().enumerate() {
        u[source + column * k] = D::from_real(1.0);
        vh[column + source * k] = if magnitude == 0.0 {
            D::from_real(1.0)
        } else {
            let value = values[source].widen_complex();
            let scale = value.re.abs().max(value.im.abs());
            let scaled = value / scale;
            D::from_complex64(scaled / scaled.norm())
        };
    }
    (
        u,
        vh,
        order.into_iter().map(|(_, magnitude)| magnitude).collect(),
    )
}

/// Checked-provider diagonal SVD factors. Unsupported source layouts or
/// numerical inputs leave the ordinary checked dense path in charge of its
/// errors and numerics; admitted outputs use the usual checked factor builder.
#[doc(hidden)]
pub fn svd_compact_diagonal_factors_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<
    Option<CheckedCompactSvdFactorsWithSpectrum<R, D>>,
    CheckedGenericFactorPlanError<R::Error>,
>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let source = authority.space();
    if source.nout() != 1
        || source.nin() != 1
        || source.homspace().codomain().legs() != source.homspace().domain().legs()
    {
        return Ok(None);
    }
    let Ok(Some(source_regions)) = checked_sector_regions(source.structure(), 1) else {
        return Ok(None);
    };
    let Some(by_sector) = aligned_diagonal_spectrum_by_sector(&source_regions, spectrum) else {
        return Ok(None);
    };
    for region in source_regions.iter() {
        let entry = by_sector[&region.coupled()];
        if region.row_trees().len() != 1 || region.col_trees().len() != 1 {
            return Ok(None);
        }
        for &value in &entry.values {
            let Some(magnitude) = finite_compact_magnitude(value) else {
                return Ok(None);
            };
            if magnitude > 0.0 && magnitude < D::safe_minimum() {
                return Ok(None);
            }
        }
    }

    let mut pairs = Vec::with_capacity(source_regions.len());
    let mut singular_values = Vec::with_capacity(source_regions.len());
    for region in source_regions.iter() {
        let entry = by_sector[&region.coupled()];
        let (left, right, values) = compact_diagonal_svd_sector(&entry.values);
        pairs.push(FactorPair {
            sector: region.coupled(),
            kept: region.rows(),
            left,
            left_rows: region.rows(),
            right,
            right_leading: region.rows(),
        });
        singular_values.push(SectorSpectrum {
            sector: region.coupled(),
            values,
        });
    }
    if factor_bond_is_input_bond(source, || {
        SectorLeg::new(pairs.iter().map(|pair| (pair.sector, pair.kept)), false)
    }) {
        let (left, right): (Vec<_>, Vec<_>) = pairs
            .into_iter()
            .map(|pair| (pair.left, pair.right))
            .unzip();
        let u = factor_on_input_space(authority, &source_regions, left)?;
        let vh = factor_on_input_space(authority, &source_regions, right)?;
        return Ok(Some((u, vh, singular_values)));
    }
    let (u, vh) = build_left_right_bound_pair_generic_checked(
        authority.provider_arc(),
        source.homspace(),
        &source_regions,
        pairs,
    )?;
    Ok(Some((u, vh, singular_values)))
}

/// Checked-provider full SVD factors read directly from an admitted compact
/// diagonal. The full builders remain the publication authority.
#[doc(hidden)]
pub fn svd_full_diagonal_factors_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<CheckedDiagonalFullSvdFactors<R, D, R::Error>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let source = authority.space();
    if source.nout() != 1
        || source.nin() != 1
        || source.homspace().codomain().legs() != source.homspace().domain().legs()
    {
        return Ok(CheckedDiagonalFullSvdFactors::NotAdmitted);
    }
    let Ok(Some(source_regions)) = checked_sector_regions(source.structure(), 1) else {
        return Ok(CheckedDiagonalFullSvdFactors::NotAdmitted);
    };
    let Some(by_sector) = aligned_diagonal_spectrum_by_sector(&source_regions, spectrum) else {
        return Ok(CheckedDiagonalFullSvdFactors::NotAdmitted);
    };
    let mut source_dimensions = BTreeMap::new();
    for region in source_regions.iter() {
        let entry = by_sector[&region.coupled()];
        if region.rows() == 0
            || region.row_trees().len() != 1
            || region.col_trees().len() != 1
            || source_dimensions
                .insert(region.coupled(), region.rows())
                .is_some()
        {
            return Ok(CheckedDiagonalFullSvdFactors::NotAdmitted);
        }
        for &value in &entry.values {
            let Some(magnitude) = finite_compact_magnitude(value) else {
                return Ok(CheckedDiagonalFullSvdFactors::NotAdmitted);
            };
            if magnitude > 0.0 && magnitude < D::safe_minimum() {
                return Ok(CheckedDiagonalFullSvdFactors::NotAdmitted);
            }
        }
    }

    if factor_bond_is_input_bond(source, || {
        SectorLeg::new(
            source_dimensions
                .iter()
                .map(|(&sector, &dimension)| (sector, dimension)),
            false,
        )
    }) {
        let (pairs, singular_values) = diagonal_full_svd_pairs(&source_regions, &by_sector);
        let (left, right): (Vec<_>, Vec<_>) = pairs
            .into_iter()
            .map(|pair| (pair.left, pair.right))
            .unzip();
        return Ok(CheckedDiagonalFullSvdFactors::Direct(SvdFullFactorsDyn {
            u: factor_on_input_space(authority, &source_regions, left)?,
            vh: factor_on_input_space(authority, &source_regions, right)?,
            singular_values,
            row_dimensions: source_dimensions.clone(),
            col_dimensions: source_dimensions,
            #[cfg(test)]
            adjoint_space: None,
        }));
    }
    let dimensions = coupled_sector_block_dimensions_generic_checked(
        source.homspace().codomain(),
        authority.provider_arc().as_ref(),
    )
    .and_then(|rows| {
        coupled_sector_block_dimensions_generic_checked(
            source.homspace().domain(),
            authority.provider_arc().as_ref(),
        )
        .map(|cols| (rows, cols))
    });
    let (row_dimensions, col_dimensions) = match dimensions {
        Ok(dimensions) => dimensions,
        Err(error) => {
            return Ok(CheckedDiagonalFullSvdFactors::Fallback(Err(error)));
        }
    };
    if row_dimensions != source_dimensions || col_dimensions != source_dimensions {
        return Ok(CheckedDiagonalFullSvdFactors::Fallback(Ok((
            row_dimensions,
            col_dimensions,
        ))));
    }

    let (mut pairs, singular_values) = diagonal_full_svd_pairs(&source_regions, &by_sector);
    let u = build_bound_factor_generic_checked(
        authority.provider_arc(),
        source.homspace(),
        &source_regions,
        &mut pairs,
        &row_dimensions,
        FactorSide::Left,
    )?;
    let vh = build_bound_factor_generic_checked(
        authority.provider_arc(),
        source.homspace(),
        &source_regions,
        &mut pairs,
        &col_dimensions,
        FactorSide::Right,
    )?;
    Ok(CheckedDiagonalFullSvdFactors::Direct(SvdFullFactorsDyn {
        u,
        vh,
        singular_values,
        row_dimensions,
        col_dimensions,
        #[cfg(test)]
        adjoint_space: None,
    }))
}

/// Gauged full-SVD factor pairs and singular values of an admitted compact
/// diagonal, one square sector at a time.
fn diagonal_full_svd_pairs<D: FactorScalar>(
    source_regions: &[CoupledSectorRegion],
    by_sector: &FxHashMap<SectorId, &SectorSpectrum<D>>,
) -> (Vec<FactorPair<D>>, Vec<SectorSpectrum>) {
    let mut pairs = Vec::with_capacity(source_regions.len());
    let mut singular_values = Vec::with_capacity(source_regions.len());
    for region in source_regions.iter() {
        let entry = by_sector[&region.coupled()];
        let (mut left, right, values) = compact_diagonal_svd_sector(&entry.values);
        let mut right = right;
        svd_full_gauge(
            &mut left,
            region.rows(),
            region.rows(),
            &mut right,
            region.cols(),
            region.cols(),
        );
        pairs.push(FactorPair {
            sector: region.coupled(),
            kept: region.rows(),
            left,
            left_rows: region.rows(),
            right,
            right_leading: region.cols(),
        });
        singular_values.push(SectorSpectrum {
            sector: region.coupled(),
            values,
        });
    }
    (pairs, singular_values)
}

pub fn svd_compact_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    svd_compact_factors_dyn_with_direction(dense, input, None, CompactSvdGauge::Left)
}

/// Compact SVD factors for the logical adjoint without constructing its input:
/// if `A = U S Vh`, returns `(V, Uh, spectrum)` with the phase gauge applied
/// to the final left factor `V`.
#[doc(hidden)]
pub fn svd_compact_adjoint_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    svd_compact_factors_dyn_with_direction(dense, input, None, CompactSvdGauge::AdjointLeft)
}

pub(super) struct CompactSvdNumericalStage<D> {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) rank: usize,
    pub(super) u: Vec<D>,
    pub(super) singular_values: Vec<f64>,
    pub(super) vt: Vec<D>,
}

pub(super) fn compact_svd_numerical_stage<E, D>(
    dense: &mut E,
    matrix: &[D],
    rows: usize,
    cols: usize,
) -> Result<CompactSvdNumericalStage<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let rank = rows.min(cols);
    if rank == 0 {
        return Ok(CompactSvdNumericalStage {
            rows,
            cols,
            rank,
            u: Vec::new(),
            singular_values: Vec::new(),
            vt: Vec::new(),
        });
    }
    let (mut u, singular_values, mut vt) = compact_svd_owned(dense, matrix, rows, cols)?;
    svd_compact_gauge(&mut u, rows, rows, &mut vt, rank, cols, rank);
    #[cfg(test)]
    record_checked_compact_svd_stage_gauge(&u, &vt);
    Ok(CompactSvdNumericalStage {
        rows,
        cols,
        rank,
        u,
        singular_values,
        vt,
    })
}

#[cfg(test)]
pub(crate) fn compact_svd_numerical_stage_lengths_for_test<E, D>(
    dense: &mut E,
    matrix: &[D],
    rows: usize,
    cols: usize,
) -> Result<(usize, usize, usize), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let stage = compact_svd_numerical_stage(dense, matrix, rows, cols)?;
    Ok((stage.u.len(), stage.singular_values.len(), stage.vt.len()))
}

#[derive(Clone, Copy)]
pub(super) enum CompactSvdGauge {
    Left,
    AdjointLeft,
}

pub(super) fn svd_compact_factors_dyn_with_direction<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    polar_direction: Option<(PolarDirection, PolarDirection)>,
    gauge: CompactSvdGauge,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    if let Some((acceptance_direction, error_direction)) = polar_direction {
        // Stored routes omit side-only sectors, whose logical matrices are
        // rows x 0 or 0 x columns and still constrain the isometry direction.
        validate_polar_direction(acceptance_direction, error_direction, input.space())?;
    }
    if let Some(plan) = compact_factor_plan(input.space())? {
        let adjoint_spaces = if matches!(gauge, CompactSvdGauge::AdjointLeft) {
            let adjoint = input.space().adjoint_view()?;
            let bond = compact_bond_leg(&plan.source_regions);
            Some((
                build_bound_factor_space(
                    &adjoint,
                    adjoint.space().homspace(),
                    bond.clone(),
                    FactorSide::Left,
                )?,
                build_bound_factor_space(
                    &adjoint,
                    adjoint.space().homspace(),
                    bond,
                    FactorSide::Right,
                )?,
            ))
        } else {
            None
        };
        return svd_compact_direct_regions(dense, input, &plan, gauge, adjoint_spaces);
    }
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    record_compact_svd_input_pack(&matricizations);

    let ranks = matricizations
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector,
            kept: matrix.rows.min(matrix.cols),
        })
        .collect::<Vec<_>>();
    let bond = SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false);
    let (left_space, right_space) = match gauge {
        CompactSvdGauge::Left => (
            build_bound_factor_space(
                input.space(),
                space.homspace(),
                bond.clone(),
                FactorSide::Left,
            )?,
            build_bound_factor_space(input.space(), space.homspace(), bond, FactorSide::Right)?,
        ),
        CompactSvdGauge::AdjointLeft => {
            let adjoint = input.space().adjoint_view()?;
            (
                build_bound_factor_space(
                    &adjoint,
                    adjoint.space().homspace(),
                    bond.clone(),
                    FactorSide::Left,
                )?,
                build_bound_factor_space(
                    &adjoint,
                    adjoint.space().homspace(),
                    bond,
                    FactorSide::Right,
                )?,
            )
        }
    };
    let u_len = left_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut u_data = vec![D::zero(); u_len];
    let vt_len = right_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut vt_data = vec![D::zero(); vt_len];

    let mut singular_values = Vec::with_capacity(matricizations.len());

    let index = PlacementIndex::new(&matricizations, &[FactorSide::Left, FactorSide::Right]);
    let u_groups = SectorBlockGroups::new(left_space.space().structure(), FactorSide::Left)?;
    let vt_groups = SectorBlockGroups::new(right_space.space().structure(), FactorSide::Right)?;
    let (u_target, vt_target) = (left_space.space(), right_space.space());
    in_linalg_scope(dense, |dense| {
        for matrix in &matricizations {
            let rank = matrix.rows.min(matrix.cols);
            let (mut u, values, mut vt) =
                compact_svd_owned(dense, &matrix.data, matrix.rows, matrix.cols)?;
            match gauge {
                CompactSvdGauge::Left => svd_compact_gauge(
                    &mut u,
                    matrix.rows,
                    matrix.rows,
                    &mut vt,
                    rank,
                    matrix.cols,
                    rank,
                ),
                CompactSvdGauge::AdjointLeft => svd_compact_adjoint_gauge(
                    &mut u,
                    matrix.rows,
                    matrix.rows,
                    &mut vt,
                    rank,
                    matrix.cols,
                    rank,
                ),
            }
            #[cfg(test)]
            record_mf_compact_svd_fallback_gauge(&u, &vt);

            singular_values.push(SectorSpectrum {
                sector: matrix.sector,
                values,
            });
            match gauge {
                CompactSvdGauge::Left => scatter_left_sector_blocks(
                    u_target,
                    &mut u_data,
                    matrix,
                    &index,
                    &u_groups,
                    &u,
                    matrix.rows,
                )?,
                CompactSvdGauge::AdjointLeft => scatter_adjoint_svd_factor(
                    u_target,
                    &mut u_data,
                    matrix,
                    &index,
                    &u_groups,
                    &vt,
                    rank,
                    FactorSide::Left,
                )?,
            }
            #[cfg(test)]
            record_compact_svd_output_scatter::<D>(matrix.rows * rank);
            match gauge {
                CompactSvdGauge::Left => scatter_right_sector_blocks(
                    vt_target,
                    &mut vt_data,
                    matrix,
                    &index,
                    &vt_groups,
                    &vt,
                    rank,
                )?,
                CompactSvdGauge::AdjointLeft => scatter_adjoint_svd_factor(
                    vt_target,
                    &mut vt_data,
                    matrix,
                    &index,
                    &vt_groups,
                    &u,
                    matrix.rows,
                    FactorSide::Right,
                )?,
            }
            #[cfg(test)]
            record_compact_svd_output_scatter::<D>(rank * matrix.cols);
        }
        Ok(())
    })?;

    let (u_nout, u_nin) = (u_target.nout(), u_target.nin());
    let (vh_nout, vh_nin) = (vt_target.nout(), vt_target.nin());
    let u = BoundDynFactor::from_bound(left_space, u_data, u_nout, u_nin)?;
    let vh = BoundDynFactor::from_bound(right_space, vt_data, vh_nout, vh_nin)?;
    Ok((u, vh, singular_values))
}

pub(super) fn svd_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
    gauge: CompactSvdGauge,
    adjoint_spaces: Option<(BoundDynamicFusionMapSpace<R>, BoundDynamicFusionMapSpace<R>)>,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source_layout, input.space().validated_layout());
    let (left_space, right_space) = match adjoint_spaces {
        Some(spaces) => spaces,
        None => (
            input.space().rebind_validated(&plan.left_layout)?,
            input.space().rebind_validated(&plan.right_layout)?,
        ),
    };
    let adjoint_routes = if matches!(gauge, CompactSvdGauge::AdjointLeft) {
        let left =
            checked_sector_regions(left_space.space().structure(), left_space.space().nout())?;
        let right =
            checked_sector_regions(right_space.space().structure(), right_space.space().nout())?;
        match (left, right) {
            (Some(left), Some(right)) => {
                let left_by_sector = SectorRegionIndex::new(&left)?;
                let right_by_sector = SectorRegionIndex::new(&right)?;
                let mut routes = Vec::with_capacity(plan.routes.len());
                let mut compatible = true;
                for route in &plan.routes {
                    let source = &plan.source_regions[route.source_region];
                    let pair = left_by_sector
                        .get(route.sector)
                        .zip(right_by_sector.get(route.sector));
                    if let Some((li, ri)) = pair {
                        compatible &= left[li].row_trees() == source.col_trees()
                            && right[ri].col_trees() == source.row_trees()
                            && left[li].rows() == source.cols()
                            && right[ri].cols() == source.rows();
                        routes.push((li, ri));
                    } else {
                        compatible = false;
                        break;
                    }
                }
                compatible.then_some((routes, left.len(), right.len()))
            }
            _ => None,
        }
    } else {
        None
    };
    let reuse_regions = matches!(gauge, CompactSvdGauge::Left) || adjoint_routes.is_some();
    let mut u_regions = if reuse_regions {
        vec![
            None;
            adjoint_routes
                .as_ref()
                .map_or(plan.left_regions.len(), |r| r.1)
        ]
    } else {
        Vec::new()
    };
    let mut vh_regions = if reuse_regions {
        vec![
            None;
            adjoint_routes
                .as_ref()
                .map_or(plan.right_regions.len(), |r| r.2)
        ]
    } else {
        Vec::new()
    };
    let mut singular_values = Vec::with_capacity(plan.routes.len());
    let mut left_data = if matches!(gauge, CompactSvdGauge::AdjointLeft) && !reuse_regions {
        vec![D::zero(); left_space.space().required_len()?]
    } else {
        Vec::new()
    };
    let mut right_data = if matches!(gauge, CompactSvdGauge::AdjointLeft) && !reuse_regions {
        vec![D::zero(); right_space.space().required_len()?]
    } else {
        Vec::new()
    };
    let adjoint_replay = if matches!(gauge, CompactSvdGauge::AdjointLeft) && !reuse_regions {
        Some((
            PlacementIndex::new(&plan.source_regions, &[FactorSide::Left, FactorSide::Right]),
            SectorBlockGroups::new(left_space.space().structure(), FactorSide::Left)?,
            SectorBlockGroups::new(right_space.space().structure(), FactorSide::Right)?,
        ))
    } else {
        None
    };

    let blocks = plan
        .routes
        .iter()
        .filter(|route| route.rank != 0)
        .map(|route| {
            let region = &plan.source_regions[route.source_region];
            (&input.data()[region.range()], region.rows(), region.cols())
        })
        .collect::<Vec<_>>();
    let mut factors = compact_svd_owned_batch(dense, &blocks)?.into_iter();

    for (route_index, route) in plan.routes.iter().copied().enumerate() {
        let region = &plan.source_regions[route.source_region];
        let rank = route.rank;
        if rank == 0 {
            singular_values.push(SectorSpectrum {
                sector: route.sector,
                values: Vec::new(),
            });
            continue;
        }
        let left_region = route.left_region.expect("nonzero route has left region");
        let right_region = route.right_region.expect("nonzero route has right region");
        let (mut u, spectrum, mut vh) = factors
            .next()
            .expect("one batch output per nonzero-rank route");
        match gauge {
            CompactSvdGauge::Left => svd_compact_gauge(
                &mut u,
                region.rows(),
                region.rows(),
                &mut vh,
                rank,
                region.cols(),
                rank,
            ),
            CompactSvdGauge::AdjointLeft => svd_compact_adjoint_gauge(
                &mut u,
                region.rows(),
                region.rows(),
                &mut vh,
                rank,
                region.cols(),
                rank,
            ),
        }
        match gauge {
            CompactSvdGauge::Left => {
                u_regions[left_region] = Some(u);
                vh_regions[right_region] = Some(vh);
            }
            CompactSvdGauge::AdjointLeft => {
                if let Some((routes, _, _)) = &adjoint_routes {
                    adjoint_col_major_in_place(&mut vh, rank, region.cols());
                    adjoint_col_major_in_place(&mut u, region.rows(), rank);
                    let (li, ri) = routes[route_index];
                    u_regions[li] = Some(vh);
                    vh_regions[ri] = Some(u);
                } else {
                    let (index, left_groups, right_groups) = adjoint_replay
                        .as_ref()
                        .expect("adjoint replay is present for adjoint gauge");
                    scatter_adjoint_svd_factor(
                        left_space.space(),
                        &mut left_data,
                        region,
                        index,
                        left_groups,
                        &vh,
                        rank,
                        FactorSide::Left,
                    )?;
                    scatter_adjoint_svd_factor(
                        right_space.space(),
                        &mut right_data,
                        region,
                        index,
                        right_groups,
                        &u,
                        region.rows(),
                        FactorSide::Right,
                    )?;
                }
            }
        }
        singular_values.push(SectorSpectrum {
            sector: route.sector,
            values: spectrum,
        });
    }

    let (u_data, vh_data) = match gauge {
        CompactSvdGauge::Left => (
            concat_compact_svd_factor_regions(u_regions, plan.left_layout.required_len()?),
            concat_compact_svd_factor_regions(vh_regions, plan.right_layout.required_len()?),
        ),
        CompactSvdGauge::AdjointLeft if reuse_regions => (
            concat_compact_svd_factor_regions(u_regions, left_space.space().required_len()?),
            concat_compact_svd_factor_regions(vh_regions, right_space.space().required_len()?),
        ),
        CompactSvdGauge::AdjointLeft => (left_data, right_data),
    };
    let u = BoundDynFactor::from_bound(
        left_space,
        u_data,
        match gauge {
            CompactSvdGauge::Left => space.nout(),
            CompactSvdGauge::AdjointLeft => space.nin(),
        },
        1,
    )?;
    let vh = BoundDynFactor::from_bound(
        right_space,
        vh_data,
        1,
        match gauge {
            CompactSvdGauge::Left => space.nin(),
            CompactSvdGauge::AdjointLeft => space.nout(),
        },
    )?;
    Ok((u, vh, singular_values))
}

#[allow(clippy::too_many_arguments)]
fn scatter_adjoint_svd_factor<D, M>(
    target: &DynamicFusionMapSpace,
    data: &mut [D],
    matrix: &M,
    index: &PlacementIndex<'_>,
    groups: &SectorBlockGroups,
    factor: &[D],
    source_rows: usize,
    side: FactorSide,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    M: SectorGeometry,
{
    let source_side = match side {
        FactorSide::Left => FactorSide::Right,
        FactorSide::Right => FactorSide::Left,
    };
    for block_index in groups.blocks(matrix.sector()) {
        let block = target.structure().block(block_index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let tree = match side {
            FactorSide::Left => key.codomain_tree(),
            FactorSide::Right => key.domain_tree(),
        };
        let (side_offset, _) = index.placement(matrix.sector(), source_side, tree)?;
        let shape = block.shape();
        let strides = block.strides();
        let bond_axis = match side {
            FactorSide::Left => shape.len() - 1,
            FactorSide::Right => 0,
        };
        let side_extent = shape
            .iter()
            .enumerate()
            .filter(|&(axis, _)| axis != bond_axis)
            .map(|(_, &extent)| extent)
            .product::<usize>();
        for side_index in 0..side_extent {
            let mut remaining = side_index;
            let mut target_offset = block.offset();
            for axis in 0..shape.len() {
                if axis != bond_axis {
                    let coordinate = remaining % shape[axis];
                    remaining /= shape[axis];
                    target_offset += coordinate * strides[axis];
                }
            }
            for bond in 0..shape[bond_axis] {
                let source_index = match side {
                    FactorSide::Left => bond + source_rows * (side_offset + side_index),
                    FactorSide::Right => side_offset + side_index + source_rows * bond,
                };
                data[target_offset + bond * strides[bond_axis]] =
                    FactorScalar::adjoint(factor[source_index]);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
/// Dynamic-rank [`svd_compact`]: the [`svd_compact_factors_dyn`] core plus the
/// diagonal `S` materialized as a `bond <- bond` tensor.
pub(crate) fn svd_compact_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdCompactDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (u, vh, singular_values) = svd_compact_factors_dyn(dense, input)?;
    let s = diagonal_bond_svd_factor(input.space(), &singular_values, &D::from_real)?;
    Ok(SvdCompactDyn {
        u,
        s,
        vh,
        singular_values,
    })
}

/// Host-side truncation decision over the spectra of a bond factorization:
/// the selection magnitude is `|value|` and each `spectra` entry is stored
/// descending by magnitude (the `*_full` output contract), so the kept set is
/// always a per-sector prefix.
///
/// Public and `doc(hidden)` only because the typed facade's
/// `GradedSpace::find_truncated`, in the `tenet` crate, is its caller; it is
/// not an API of its own.
#[doc(hidden)]
pub fn decide_bond_truncation<R, V>(
    rule: &R,
    spectra: &[SectorSpectrum<V>],
    truncation: &Truncation,
) -> Result<crate::truncation::TruncationDecision, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    V: SpectrumMagnitude,
{
    let magnitudes: Vec<Vec<f64>> = spectra
        .iter()
        .map(|entry| entry.values.iter().map(|value| value.magnitude()).collect())
        .collect();
    let mut weighted = Vec::with_capacity(spectra.len());
    for (entry, values) in spectra.iter().zip(&magnitudes) {
        weighted.push(WeightedSpectrum {
            sector: entry.sector,
            weight: <MultiplicityFreeAdmissionMode as RigidCoefficientAlgebra<R>>::dim(
                rule,
                entry.sector,
            )?,
            values,
        });
    }
    select_truncation(&weighted, truncation, &rule.rule_identity(), |sector| {
        rule.sector_order_key(sector)
    })
    .map_err(OperationError::from)
}

#[cfg(test)]
/// Full fusion-tensor SVD `t = U * S * Vh` (MatrixAlgebraKit `svd_full`):
/// per sector `U` is the square `m x m` unitary, `S` the rectangular
/// `m x n` diagonal, and `Vh` the square `n x n` unitary.
#[derive(Clone, Debug)]
pub(crate) struct SvdFull<R, D, const NOUT: usize, const NIN: usize> {
    pub u: BoundTensorMap<R, D, NOUT, 1>,
    pub s: BoundTensorMap<R, D, 1, 1>,
    pub vh: BoundTensorMap<R, D, 1, NIN>,
}

#[cfg(test)]
/// Dynamic-rank [`SvdFull`].
#[derive(Clone, Debug)]
pub(crate) struct SvdFullDyn<R, D> {
    pub(super) u: BoundDynFactor<R, D>,
    pub(super) s: BoundDynFactor<R, D>,
    pub(super) vh: BoundDynFactor<R, D>,
    pub(super) singular_values: Vec<SectorSpectrum>,
}

/// Numerical full-SVD factors before choosing the storage of `S`.
#[doc(hidden)]
pub struct SvdFullFactorsDyn<R, D> {
    u: BoundDynFactor<R, D>,
    vh: BoundDynFactor<R, D>,
    singular_values: Vec<SectorSpectrum>,
    row_dimensions: BTreeMap<SectorId, usize>,
    col_dimensions: BTreeMap<SectorId, usize>,
    #[cfg(test)]
    adjoint_space: Option<BoundDynamicFusionMapSpace<R>>,
}

#[doc(hidden)]
pub type CheckedFullSvdDimensions<E> = Result<
    (BTreeMap<SectorId, usize>, BTreeMap<SectorId, usize>),
    CheckedGenericFactorPlanError<E>,
>;

/// Result of compact-diagonal full-SVD admission. A checked dimension result
/// is returned to the dense fallback so a stateful provider is queried once.
#[doc(hidden)]
#[expect(
    clippy::large_enum_variant,
    reason = "boxing the direct factors would add an allocation to the admitted path"
)]
pub enum CheckedDiagonalFullSvdFactors<R, D, E> {
    NotAdmitted,
    Direct(SvdFullFactorsDyn<R, D>),
    Fallback(CheckedFullSvdDimensions<E>),
}

impl<R, D> SvdFullFactorsDyn<R, D> {
    #[expect(
        clippy::type_complexity,
        reason = "full-SVD factors retain both output bond maps"
    )]
    pub fn into_parts(
        self,
    ) -> (
        BoundDynFactor<R, D>,
        BoundDynFactor<R, D>,
        Vec<SectorSpectrum>,
        BTreeMap<SectorId, usize>,
        BTreeMap<SectorId, usize>,
    ) {
        (
            self.u,
            self.vh,
            self.singular_values,
            self.row_dimensions,
            self.col_dimensions,
        )
    }
}

#[cfg(test)]
impl<R, D> SvdFullDyn<R, D> {
    pub fn u(&self) -> &BoundDynFactor<R, D> {
        &self.u
    }
    pub fn s(&self) -> &BoundDynFactor<R, D> {
        &self.s
    }
    pub fn vh(&self) -> &BoundDynFactor<R, D> {
        &self.vh
    }
    pub fn singular_values(&self) -> &[SectorSpectrum] {
        &self.singular_values
    }
}

#[cfg(test)]
/// Full fusion-tensor SVD through the device boundary.
///
/// A provider that advertises [`DenseExecutor::supports_svd_full`] factorizes
/// the full matrices directly, one call per sector. Otherwise the unitaries
/// are completed from the compact factors with an extra economy QR of
/// `[U1 | I]` per sector (any orthonormal completion is exact because the
/// corresponding rows/columns of `S` are zero), so every provider stays on the
/// existing dense-executor boundary. The two routes agree on the reconstructed
/// input, the spectrum and unitarity, but the basis spanning the null space is
/// not unique and differs between them.
pub(crate) fn svd_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<SvdFull<R, D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let out = svd_full_dyn(dense, &input.dynamic())?;
    Ok(SvdFull {
        u: typed_from_bound_factor(out.u)?,
        s: typed_from_bound_factor(out.s)?,
        vh: typed_from_bound_factor(out.vh)?,
    })
}

#[cfg(test)]
/// Dynamic-rank [`svd_full`].
pub(crate) fn svd_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    svd_full_oriented_dyn(dense, input, FactorPlacement::Direct)
}

/// Full-SVD numerical factors without publishing a dense `S` tensor.
#[doc(hidden)]
pub fn svd_full_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    svd_full_oriented_factors_dyn(dense, input, FactorPlacement::Direct)
}

#[doc(hidden)]
pub fn svd_full_adjoint_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    svd_full_oriented_factors_dyn(dense, input, FactorPlacement::Adjoint)
}

#[cfg(test)]
/// Full SVD factors for the logical adjoint without constructing its input.
pub(crate) fn svd_full_adjoint_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    svd_full_oriented_dyn(dense, input, FactorPlacement::Adjoint)
}

#[expect(
    clippy::type_complexity,
    reason = "the private stage returns the fixed dense full-SVD tuple without another wrapper"
)]
pub(super) fn owned_full_svd_stage<E, D>(
    dense: &mut E,
    data: &mut Vec<D>,
    rows: usize,
    cols: usize,
) -> Result<Option<(Vec<D>, Vec<f64>, Vec<D>)>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    if !dense.supports_svd_full() {
        return Ok(None);
    }
    let input = match D::dense_into_owned(std::mem::take(data)) {
        Ok(input) => input,
        Err(input_data) => {
            *data = input_data;
            return Ok(None);
        }
    };
    let mut outputs = dense
        .svd_full_owned(input, rows, cols)
        .map_err(OperationError::Dense)?;
    if outputs.len() != 3 {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "svd_full_owned",
            message: "dense full SVD must return exactly (U, S, Vh)".to_string(),
        }));
    }
    let u = compact_factor_output_owned(outputs.remove(0), &[rows, rows], "svd_full_owned")?;
    let singular_values =
        compact_real_spectrum_owned::<D>(outputs.remove(0), &[rows.min(cols)], "svd_full_owned")?;
    let vh = compact_factor_output_owned(outputs.remove(0), &[cols, cols], "svd_full_owned")?;
    Ok(Some((u, singular_values, vh)))
}

#[cfg(test)]
pub(super) fn svd_full_oriented_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    placement: FactorPlacement,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let parts = svd_full_oriented_factors_dyn(dense, input, placement)?;
    let authority = parts.adjoint_space.as_ref().unwrap_or(input.space());
    let s = rectangular_diagonal_bond_tensor(
        authority,
        &parts.singular_values,
        &parts.row_dimensions,
        &parts.col_dimensions,
    )?;
    Ok(SvdFullDyn {
        u: parts.u,
        s,
        vh: parts.vh,
        singular_values: parts.singular_values,
    })
}

fn svd_full_oriented_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    placement: FactorPlacement,
) -> Result<SvdFullFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    // Why pack rather than borrow admitted regions: `owned_full_svd_stage`
    // moves each matrix into the dense provider, so a borrowed region would
    // be copied into an owned buffer anyway.
    let mut matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    let row_dimensions = space
        .homspace()
        .codomain()
        .coupled_sector_block_dimensions(input.space().provider())?;
    let col_dimensions = space
        .homspace()
        .domain()
        .coupled_sector_block_dimensions(input.space().provider())?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    let mut singular_values = Vec::with_capacity(matricizations.len());
    let max_rows = matricizations
        .iter()
        .map(|matrix| matrix.rows)
        .max()
        .unwrap_or(0);
    let max_cols = matricizations
        .iter()
        .map(|matrix| matrix.cols)
        .max()
        .unwrap_or(0);
    let max_rank = matricizations
        .iter()
        .map(|matrix| matrix.rows.min(matrix.cols))
        .max()
        .unwrap_or(0);
    let mut u_workspace = Vec::new();
    let mut s_workspace = Vec::new();
    let mut vt_workspace = Vec::new();
    for matrix in &mut matricizations {
        let rank = matrix.rows.min(matrix.cols);
        let (mut left, left_rows, mut right, right_leading, s_values) =
            match owned_full_svd_stage(dense, &mut matrix.data, matrix.rows, matrix.cols)? {
                Some((mut u_full, s_values, mut vh_full)) => match placement {
                    FactorPlacement::Direct => {
                        (u_full, matrix.rows, vh_full, matrix.cols, s_values)
                    }
                    FactorPlacement::Adjoint => {
                        adjoint_square_col_major_in_place(&mut vh_full, matrix.cols);
                        adjoint_square_col_major_in_place(&mut u_full, matrix.rows);
                        (vh_full, matrix.cols, u_full, matrix.rows, s_values)
                    }
                },
                None => {
                    if u_workspace.is_empty() && max_rows != 0 && max_rank != 0 {
                        u_workspace = vec![D::zero(); max_rows * max_rank];
                        s_workspace = vec![D::Real::zero(); max_rank];
                        vt_workspace = vec![D::zero(); max_rank * max_cols];
                    }
                    let shape = [matrix.rows, matrix.cols];
                    let strides = [1usize, matrix.rows];
                    let view = DenseView::new(&matrix.data, &shape, &strides, 0)
                        .map_err(OperationError::Dense)?;
                    let u_shape = [matrix.rows, rank];
                    let u_strides = [1usize, max_rows];
                    let s_shape = [rank];
                    let s_strides = [1usize];
                    let vt_shape = [rank, matrix.cols];
                    let vt_strides = [1usize, max_rank];
                    let u_view = DenseViewMut::new(&mut u_workspace, &u_shape, &u_strides, 0)
                        .map_err(OperationError::Dense)?;
                    let s_view = DenseViewMut::new(&mut s_workspace, &s_shape, &s_strides, 0)
                        .map_err(OperationError::Dense)?;
                    let vt_view = DenseViewMut::new(&mut vt_workspace, &vt_shape, &vt_strides, 0)
                        .map_err(OperationError::Dense)?;
                    dense
                        .svd_into(
                            D::dense_read(view),
                            D::dense_write(u_view),
                            D::Real::dense_write(s_view),
                            D::dense_write(vt_view),
                        )
                        .map_err(OperationError::Dense)?;
                    let s_values = s_workspace[..rank]
                        .iter()
                        .copied()
                        .map(Into::into)
                        .collect::<Vec<_>>();
                    let mut u_thin = vec![D::zero(); matrix.rows * rank];
                    let mut vt_thin = vec![D::zero(); rank * matrix.cols];
                    copy_col_major_strided(
                        &u_workspace,
                        matrix.rows,
                        rank,
                        max_rows,
                        &mut u_thin,
                        matrix.rows,
                    );
                    copy_col_major_strided(
                        &vt_workspace,
                        rank,
                        matrix.cols,
                        max_rank,
                        &mut vt_thin,
                        rank,
                    );
                    let mut u_full = orthonormal_completion(dense, &u_thin, matrix.rows, rank)?;
                    let v_thin = adjoint_col_major(&vt_thin, rank, matrix.cols);
                    let v_full = orthonormal_completion(dense, &v_thin, matrix.cols, rank)?;
                    match placement {
                        FactorPlacement::Direct => (
                            u_full,
                            matrix.rows,
                            adjoint_col_major(&v_full, matrix.cols, matrix.cols),
                            matrix.cols,
                            s_values,
                        ),
                        FactorPlacement::Adjoint => {
                            adjoint_square_col_major_in_place(&mut u_full, matrix.rows);
                            (v_full, matrix.cols, u_full, matrix.rows, s_values)
                        }
                    }
                }
            };
        svd_full_gauge(
            &mut left,
            left_rows,
            left_rows,
            &mut right,
            right_leading,
            right_leading,
        );

        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: left_rows,
            left,
            left_rows,
            right,
            right_leading,
        });
    }

    let adjoint_space = match placement {
        FactorPlacement::Direct => None,
        FactorPlacement::Adjoint => Some(<MultiplicityFreeAdmissionMode as CoefficientAlgebra<
            R,
        >>::adjoint_space(input.space())?),
    };
    let authority = adjoint_space.as_ref().unwrap_or(input.space());
    let homspace = authority.space().homspace();
    let (output_row_dimensions, output_col_dimensions) = match placement {
        FactorPlacement::Direct => (&row_dimensions, &col_dimensions),
        FactorPlacement::Adjoint => (&col_dimensions, &row_dimensions),
    };

    // The left/right bond legs differ in the full SVD (rows vs columns), so
    // build the two factors with separate bond dimensions.
    let u_factor = build_bound_factor_with_placement(
        authority,
        homspace,
        &matricizations,
        &mut pairs,
        output_row_dimensions,
        FactorSide::Left,
        placement,
    )?;
    let vh_factor = build_bound_factor_with_placement(
        authority,
        homspace,
        &matricizations,
        &mut pairs,
        output_col_dimensions,
        FactorSide::Right,
        placement,
    )?;
    let (row_dimensions, col_dimensions) = match placement {
        FactorPlacement::Direct => (row_dimensions, col_dimensions),
        FactorPlacement::Adjoint => (col_dimensions, row_dimensions),
    };
    Ok(SvdFullFactorsDyn {
        u: u_factor,
        vh: vh_factor,
        singular_values,
        row_dimensions,
        col_dimensions,
        #[cfg(test)]
        adjoint_space,
    })
}

fn adjoint_square_col_major_in_place<D: FactorScalar>(data: &mut [D], dimension: usize) {
    for column in 0..dimension {
        let diagonal = column * dimension + column;
        data[diagonal] = FactorScalar::adjoint(data[diagonal]);
        for row in 0..column {
            let a = row + dimension * column;
            let b = column + dimension * row;
            let value = data[a];
            data[a] = FactorScalar::adjoint(data[b]);
            data[b] = FactorScalar::adjoint(value);
        }
    }
}

fn adjoint_col_major_in_place<D: FactorScalar>(data: &mut [D], rows: usize, cols: usize) {
    if rows == cols {
        adjoint_square_col_major_in_place(data, rows);
        return;
    }
    if rows <= 1 || cols <= 1 {
        for value in data {
            *value = FactorScalar::adjoint(*value);
        }
        return;
    }
    let mut visited = vec![false; data.len()];
    for start in 0..data.len() {
        if visited[start] {
            continue;
        }
        let mut current = start;
        let mut value = data[current];
        loop {
            let next = current / rows + cols * (current % rows);
            let displaced = data[next];
            data[next] = FactorScalar::adjoint(value);
            visited[current] = true;
            current = next;
            value = displaced;
            if current == start {
                break;
            }
        }
    }
}

/// Completes `k` orthonormal columns (`m x k`, column-major) to a full
/// `m x m` orthonormal basis via an economy QR of `[Q1 | I]`; the first `k`
/// columns are returned unchanged.
pub(super) fn orthonormal_completion<E, D>(
    dense: &mut E,
    thin: &[D],
    rows: usize,
    rank: usize,
) -> Result<Vec<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    if rank == rows {
        return Ok(thin.to_vec());
    }
    let mut augmented = vec![D::zero(); rows * (rank + rows)];
    augmented[..rows * rank].copy_from_slice(thin);
    for row in 0..rows {
        augmented[rows * rank + row * rows + row] = D::one();
    }
    let mut q = vec![D::zero(); rows * rows];
    let mut r = vec![D::zero(); rows * (rank + rows)];
    qr_into_workspace(
        dense,
        &augmented,
        rows,
        rank + rows,
        rows,
        &mut q,
        rows,
        rows,
        rows,
        &mut r,
        rows,
        rank + rows,
        rows,
    )?;
    let mut full = vec![D::zero(); rows * rows];
    full[..rows * rank].copy_from_slice(thin);
    full[rows * rank..].copy_from_slice(&q[rows * rank..rows * rows]);
    Ok(full)
}

/// Rectangular diagonal `W_row <- W_col` bond factor (the `S` of the full
/// SVD): per sector shape `[rows, cols]` with the spectrum on the diagonal.
#[doc(hidden)]
pub fn rectangular_diagonal_bond_tensor<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectra: &[SectorSpectrum],
    row_dimensions: &BTreeMap<SectorId, usize>,
    col_dimensions: &BTreeMap<SectorId, usize>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let row_leg = SectorLeg::new(
        row_dimensions
            .iter()
            .map(|(&sector, &dimension)| (sector, dimension)),
        false,
    );
    let col_leg = SectorLeg::new(
        col_dimensions
            .iter()
            .map(|(&sector, &dimension)| (sector, dimension)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([row_leg]),
        FusionProductSpace::new([col_leg]),
    );
    let space = authority.derive_from_final_homspace(homspace)?;
    let len = space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); len];
    let structure = Arc::clone(space.space().structure());
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum> =
        spectra.iter().map(|entry| (entry.sector, entry)).collect();
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree.codomain_tree().coupled();
        let Some(&entry) = spectrum_by_sector.get(&sector) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        for position in 0..count {
            let Some(&value) = entry.values.get(position) else {
                break;
            };
            data[offset + position * (strides[0] + strides[1])] = D::from_real(value);
        }
    }
    BoundDynFactor::from_bound(space, data, 1, 1)
}

#[doc(hidden)]
pub fn rectangular_diagonal_bond_tensor_generic_checked<R, D>(
    provider: Arc<R>,
    spectra: &[SectorSpectrum],
    row_dimensions: &BTreeMap<SectorId, usize>,
    col_dimensions: &BTreeMap<SectorId, usize>,
    to_scalar: &dyn Fn(f64) -> D,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let row_leg = SectorLeg::new(
        row_dimensions
            .iter()
            .map(|(&sector, &dimension)| (sector, dimension)),
        false,
    );
    let col_leg = SectorLeg::new(
        col_dimensions
            .iter()
            .map(|(&sector, &dimension)| (sector, dimension)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([row_leg]),
        FusionProductSpace::new([col_leg]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
        .map_err(CheckedGenericFactorPlanError::from)?;
    let len = space.space().required_len().map_err(|error| {
        CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(
            error,
        ))
    })?;
    let mut data = vec![D::zero(); len];
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum> =
        spectra.iter().map(|entry| (entry.sector, entry)).collect();
    let structure = Arc::clone(space.space().structure());
    for index in 0..structure.block_count() {
        let block = structure.block(index).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(
                error,
            ))
        })?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let Some(entry) = spectrum_by_sector.get(&tree.codomain_tree().coupled()) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        for position in 0..count {
            let Some(&value) = entry.values.get(position) else {
                break;
            };
            data[offset + position * (strides[0] + strides[1])] = to_scalar(value);
        }
    }
    BoundDynFactor::from_bound(space, data, 1, 1).map_err(CheckedGenericFactorPlanError::from)
}

/// Positive-diagonal gauge (MatrixAlgebraKit `positive = true`, the default
/// of the Householder QR/LQ algorithms since MAK 0.6.8 / TensorKit 0.17):
/// absorbs the unitary phase `D = diag(phase(R_jj))` into `Q`, i.e.
/// `Q -> Q * D`, `R -> D^H * R`, leaving `Q * R` unchanged with real
/// non-negative `R_jj`. Zero diagonal entries keep phase `1`, exactly like
/// MAK `sign_safe` (no epsilon threshold).
///
/// `q` is column-major `q_rows x nq` (`nq >= min(r_rows, r_cols)`), `r` is
/// column-major `r_rows x r_cols`.
pub(crate) fn positive_diagonal_gauge<D: FactorScalar>(
    q: &mut [D],
    q_rows: usize,
    r: &mut [D],
    r_rows: usize,
    r_cols: usize,
) {
    positive_diagonal_gauge_strided(q, q_rows, q_rows, r, r_rows, r_rows, r_cols);
}

pub(super) fn positive_diagonal_gauge_strided<D: FactorScalar>(
    q: &mut [D],
    q_rows: usize,
    q_leading: usize,
    r: &mut [D],
    r_rows: usize,
    r_leading: usize,
    r_cols: usize,
) {
    for j in 0..r_rows.min(r_cols) {
        let z = r[j + r_leading * j].widen_complex();
        let norm = z.norm();
        if norm == 0.0 {
            continue; // phase 1: nothing to scale
        }
        let phase = D::from_complex64(z / norm);
        let conj_phase = FactorScalar::adjoint(phase);
        for row in 0..q_rows {
            let index = row + q_leading * j;
            q[index] = q[index] * phase;
        }
        for col in 0..r_cols {
            let index = j + r_leading * col;
            r[index] = conj_phase * r[index];
        }
    }
}

pub(crate) fn svd_compact_gauge<D: FactorScalar>(
    u: &mut [D],
    u_rows: usize,
    u_leading: usize,
    vh: &mut [D],
    vh_rows: usize,
    vh_cols: usize,
    vh_leading: usize,
) {
    for j in 0..vh_rows {
        let (phase, needs_scaling) = phase_of_largest_abs_col(u, u_rows, u_leading, j);
        if needs_scaling {
            scale_col(u, u_rows, u_leading, j, FactorScalar::adjoint(phase));
            scale_row(vh, vh_cols, vh_leading, j, phase);
        }
    }
}

pub(crate) fn svd_compact_adjoint_gauge<D: FactorScalar>(
    u: &mut [D],
    u_rows: usize,
    u_leading: usize,
    vh: &mut [D],
    vh_rows: usize,
    vh_cols: usize,
    vh_leading: usize,
) {
    for j in 0..vh_rows {
        let (phase, needs_scaling) = phase_of_largest_abs_row(vh, vh_cols, vh_leading, j);
        if needs_scaling {
            scale_col(u, u_rows, u_leading, j, phase);
            scale_row(vh, vh_cols, vh_leading, j, FactorScalar::adjoint(phase));
        }
    }
}

pub(crate) fn svd_full_gauge<D: FactorScalar>(
    u: &mut [D],
    u_rows: usize,
    u_leading: usize,
    vh: &mut [D],
    vh_rows: usize,
    vh_cols: usize,
) {
    let paired = u_leading.min(vh_rows);
    for j in 0..u_leading.max(vh_rows) {
        if j < paired {
            let (phase, needs_scaling) = phase_of_largest_abs_col(u, u_rows, u_leading, j);
            if needs_scaling {
                scale_col(u, u_rows, u_leading, j, FactorScalar::adjoint(phase));
                scale_row(vh, vh_cols, vh_rows, j, phase);
            }
        } else if j < u_leading {
            let (phase, needs_scaling) = phase_of_largest_abs_col(u, u_rows, u_leading, j);
            if needs_scaling {
                scale_col(u, u_rows, u_leading, j, FactorScalar::adjoint(phase));
            }
        } else {
            let (phase, needs_scaling) = phase_of_largest_abs_row(vh, vh_cols, vh_rows, j);
            if needs_scaling {
                scale_row(vh, vh_cols, vh_rows, j, FactorScalar::adjoint(phase));
            }
        }
    }
}

pub(super) fn phase_of_largest_abs_col<D: FactorScalar>(
    data: &[D],
    rows: usize,
    leading: usize,
    col: usize,
) -> (D, bool) {
    let mut best = Complex64::new(0.0, 0.0);
    let mut best_norm_sqr = 0.0;
    for row in 0..rows {
        let value = data[row + leading * col].widen_complex();
        let norm_sqr = value.norm_sqr();
        if best_norm_sqr < norm_sqr {
            best = value;
            best_norm_sqr = norm_sqr;
        }
    }
    unit_phase(best, best_norm_sqr)
}

pub(super) fn phase_of_largest_abs_row<D: FactorScalar>(
    data: &[D],
    cols: usize,
    leading: usize,
    row: usize,
) -> (D, bool) {
    let mut best = Complex64::new(0.0, 0.0);
    let mut best_norm_sqr = 0.0;
    for col in 0..cols {
        let value = data[row + leading * col].widen_complex();
        let norm_sqr = value.norm_sqr();
        if best_norm_sqr < norm_sqr {
            best = value;
            best_norm_sqr = norm_sqr;
        }
    }
    unit_phase(best, best_norm_sqr)
}

pub(super) fn unit_phase<D: FactorScalar>(value: Complex64, norm_sqr: f64) -> (D, bool) {
    if norm_sqr == 0.0 || (value.im == 0.0 && value.re >= 0.0) {
        (D::from_real(1.0), false)
    } else {
        (D::from_complex64(value / norm_sqr.sqrt()), true)
    }
}

/// Compact SVD owns U and Vt, while S remains the host-side spectrum used by
/// the existing truncation and diagonal construction paths.
#[expect(
    clippy::type_complexity,
    reason = "the dense SVD ownership boundary returns its documented U, S, Vt tuple"
)]
pub(super) fn compact_svd_owned<E, D>(
    dense: &mut E,
    input: &[D],
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<f64>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let input_shape = [rows, cols];
    let input_strides = [1usize, rows];
    let input_view =
        DenseView::new(input, &input_shape, &input_strides, 0).map_err(OperationError::Dense)?;
    let outputs = dense
        .svd(D::dense_read(input_view))
        .map_err(OperationError::Dense)?;
    compact_svd_outputs(outputs, rows, cols)
}

/// Compact SVD of each column-major `(data, rows, cols)` block, submitted as one
/// executor batch so the backend admits the whole coupled-sector loop once.
#[expect(
    clippy::type_complexity,
    reason = "the dense SVD ownership boundary returns its documented U, S, Vt tuple"
)]
pub(super) fn compact_svd_owned_batch<E, D>(
    dense: &mut E,
    blocks: &[(&[D], usize, usize)],
) -> Result<Vec<(Vec<D>, Vec<f64>, Vec<D>)>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factorize_col_major_batch(dense, DenseFactorization::Svd, blocks)?
        .into_iter()
        .zip(blocks)
        .map(|(outputs, &(_, rows, cols))| compact_svd_outputs(outputs, rows, cols))
        .collect()
}

#[expect(
    clippy::type_complexity,
    reason = "the dense SVD ownership boundary returns its documented U, S, Vt tuple"
)]
pub(super) fn compact_svd_outputs<D: FactorScalar>(
    mut outputs: Vec<DenseTensor>,
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<f64>, Vec<D>), OperationError> {
    let rank = rows.min(cols);
    if outputs.len() != 3 {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "svd_into",
            message: "dense SVD must return exactly (U, S, Vt)".to_string(),
        }));
    }
    let u = compact_factor_output_owned::<D>(outputs.remove(0), &[rows, rank], "svd_into")?;
    let singular_values = compact_real_spectrum_owned::<D>(outputs.remove(0), &[rank], "svd_into")?;
    let vt = compact_factor_output_owned::<D>(outputs.remove(0), &[rank, cols], "svd_into")?;
    Ok((u, singular_values, vt))
}

pub(super) fn concat_compact_svd_factor_regions<D>(
    regions: Vec<Option<Vec<D>>>,
    required_len: usize,
) -> Vec<D> {
    #[cfg(test)]
    let first = regions
        .iter()
        .find_map(|region| region.as_ref().map(Vec::as_ptr));
    let output = concat_owned_factor_regions(regions, required_len);
    #[cfg(test)]
    COMPACT_SVD_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.owned_output_publications += 1;
        current.owned_output_owner_reused +=
            usize::from(first.is_some_and(|pointer| std::ptr::eq(pointer, output.as_ptr())));
        probe.set(current);
    });
    output
}

#[doc(hidden)]
pub fn diagonal_bond_bound_space_generic_checked<R, V>(
    provider: Arc<R>,
    spectrum: &[SectorSpectrum<V>],
) -> Result<BoundDynamicFusionMapSpace<R>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
{
    let new_leg = SectorLeg::new(
        spectrum
            .iter()
            .map(|entry| (entry.sector, entry.values.len())),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg.clone()]),
        FusionProductSpace::new([new_leg]),
    );
    BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
        .map_err(CheckedGenericFactorPlanError::from)
}

/// The compact diagonal bond space `W <- W` of `spectrum` for a factor of
/// the compact diagonal `source`: `source` itself when `W` is its one-leg bond
/// (TensorKit's `fuse(V) = V`, no provider query), otherwise the checked root
/// [`diagonal_bond_bound_space_generic_checked`] builds.
#[doc(hidden)]
pub fn diagonal_bond_bound_space_on_source_checked_generic<R, V>(
    source: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<V>],
) -> Result<BoundDynamicFusionMapSpace<R>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
{
    if factor_bond_is_input_bond(source.space(), || {
        SectorLeg::new(
            spectrum
                .iter()
                .map(|entry| (entry.sector, entry.values.len())),
            false,
        )
    }) {
        return Ok(source.clone());
    }
    diagonal_bond_bound_space_generic_checked(Arc::clone(source.provider_arc()), spectrum)
}

#[doc(hidden)]
pub fn diagonal_bond_svd_factor_generic_checked<R, D, V>(
    provider: Arc<R>,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
    V: Copy,
{
    let space = diagonal_bond_bound_space_generic_checked(provider, spectrum)?;
    let data = diagonal_bond_data(space.space(), spectrum, to_scalar)
        .map_err(CheckedGenericFactorPlanError::from)?;
    BoundDynFactor::from_bound(space, data, 1, 1).map_err(CheckedGenericFactorPlanError::from)
}

/// Checked-Generic sibling of [`decide_bond_truncation`], public and hidden for
/// the same reason.
#[doc(hidden)]
pub fn decide_bond_truncation_generic_checked<R, V>(
    rule: &R,
    spectra: &[SectorSpectrum<V>],
    truncation: &Truncation,
) -> Result<crate::truncation::TruncationDecision, CheckedGenericPlanError<R::Error>>
where
    R: CheckedGenericRigidSymbols<Scalar = f64>,
    V: SpectrumMagnitude,
{
    let magnitudes: Vec<Vec<f64>> = spectra
        .iter()
        .map(|entry| entry.values.iter().map(|value| value.magnitude()).collect())
        .collect();
    let mut weighted = Vec::with_capacity(spectra.len());
    for (entry, values) in spectra.iter().zip(&magnitudes) {
        weighted.push(WeightedSpectrum {
            sector: entry.sector,
            weight: <CheckedGenericAdmissionMode as RigidCoefficientAlgebra<R>>::dim(
                rule,
                entry.sector,
            )?,
            values,
        });
    }
    select_truncation(&weighted, truncation, &rule.rule_identity(), |sector| {
        rule.sector_order_key(sector)
    })
    .map_err(|error| CheckedGenericPlanError::Operation(error.into()))
}

/// Checked-Generic compact SVD. Dense SVD is unchanged; all provider-bound
/// output spaces are admitted through the checked staging boundary.
#[doc(hidden)]
pub fn svd_compact_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Svd<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let (u, vh, singular_values) =
        svd_compact_factors_with_spectrum_dyn_checked_generic(dense, input)?;
    let s = diagonal_bond_svd_factor_generic_checked(
        Arc::clone(input.space().provider_arc()),
        &singular_values,
        &D::from_real,
    )?;
    Ok(Svd { u, s, vh })
}

#[doc(hidden)]
pub type CheckedCompactSvdFactorsWithSpectrum<R, D> = (
    BoundDynFactor<R, D>,
    BoundDynFactor<R, D>,
    Vec<SectorSpectrum>,
);

/// Checked compact SVD factors and sector values before diagonal publication.
#[doc(hidden)]
pub fn svd_compact_factors_with_spectrum_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<CheckedCompactSvdFactorsWithSpectrum<R, D>, CheckedGenericFactorPlanError<R::Error>>
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
        record_compact_svd_input_pack(matrices);
    }
    let mut pairs = Vec::with_capacity(matrices.len());
    let mut singular_values = Vec::with_capacity(matrices.len());
    #[cfg(test)]
    let data = input.data();
    in_linalg_scope(dense, |dense| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            #[cfg(test)]
            record_checked_compact_input(CheckedCompactOperation::Svd, data, matrix.data, None);
            let stage = compact_svd_numerical_stage(dense, matrix.data, matrix.rows, matrix.cols)?;
            singular_values.push(SectorSpectrum {
                sector: matrix.sector,
                values: stage.singular_values,
            });
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: stage.rank,
                left: stage.u,
                left_rows: stage.rows,
                right: stage.vt,
                right_leading: stage.rank,
            });
        }
        Ok(())
    })
    .map_err(CheckedGenericFactorPlanError::from)?;
    let (u, vh) = build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)?;
    Ok((u, vh, singular_values))
}

#[cfg(test)]
/// Checked-Generic full SVD. Dense work is performed before any output-space
/// publication; checked factor builders then admit square outer factors and
/// the rectangular diagonal, including unmatched structural sectors.
pub(crate) fn svd_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let parts = svd_full_factors_dyn_checked_generic(dense, input)?;
    let (u, vh, singular_values, row_dimensions, col_dimensions) = parts.into_parts();
    let s = rectangular_diagonal_bond_tensor_generic_checked(
        Arc::clone(provider),
        &singular_values,
        &row_dimensions,
        &col_dimensions,
        &D::from_real,
    )?;
    Ok(SvdFullDyn {
        u,
        s,
        vh,
        singular_values,
    })
}

/// Checked full-SVD numerical factors before choosing `S` storage.
#[doc(hidden)]
pub fn svd_full_factors_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_full_factors_dyn_checked_generic_with_dimensions(dense, input, None)
}

/// Checked full SVD using a dimension result already obtained during compact
/// admission. `None` retains the ordinary query path.
#[doc(hidden)]
pub fn svd_full_factors_dyn_checked_generic_with_dimensions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    dimensions: Option<CheckedFullSvdDimensions<R::Error>>,
) -> Result<SvdFullFactorsDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    // Why pack rather than borrow admitted regions: `owned_full_svd_stage`
    // moves each matrix into the dense provider, so a borrowed region would
    // be copied into an owned buffer anyway.
    let mut matrices = sector_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let (row_dimensions, col_dimensions) = match dimensions {
        Some(dimensions) => dimensions?,
        None => (
            coupled_sector_block_dimensions_generic_checked(
                space.homspace().codomain(),
                provider.as_ref(),
            )?,
            coupled_sector_block_dimensions_generic_checked(
                space.homspace().domain(),
                provider.as_ref(),
            )?,
        ),
    };
    let max_rows = matrices.iter().map(|m| m.rows).max().unwrap_or(0);
    let max_cols = matrices.iter().map(|m| m.cols).max().unwrap_or(0);
    let max_rank = matrices
        .iter()
        .map(|m| m.rows.min(m.cols))
        .max()
        .unwrap_or(0);
    let mut u_workspace = Vec::new();
    let mut s_workspace = Vec::new();
    let mut vt_workspace = Vec::new();
    let mut pairs = Vec::with_capacity(matrices.len());
    let mut singular_values = Vec::with_capacity(matrices.len());
    for matrix in &mut matrices {
        let rank = matrix.rows.min(matrix.cols);
        let (mut left, s_values, mut right) =
            match owned_full_svd_stage(dense, &mut matrix.data, matrix.rows, matrix.cols)
                .map_err(CheckedGenericFactorPlanError::from)?
            {
                Some(outputs) => outputs,
                None => {
                    if u_workspace.is_empty() && max_rows != 0 && max_rank != 0 {
                        u_workspace = vec![D::zero(); max_rows * max_rank];
                        s_workspace = vec![D::Real::zero(); max_rank];
                        vt_workspace = vec![D::zero(); max_rank * max_cols];
                    }
                    let shape = [matrix.rows, matrix.cols];
                    let strides = [1usize, matrix.rows];
                    let u_shape = [matrix.rows, rank];
                    let u_strides = [1usize, max_rows];
                    let s_shape = [rank];
                    let s_strides = [1usize];
                    let vt_shape = [rank, matrix.cols];
                    let vt_strides = [1usize, max_rank];
                    let input_view =
                        DenseView::new(&matrix.data, &shape, &strides, 0).map_err(|e| {
                            CheckedGenericFactorPlanError::Operation(OperationError::Dense(e))
                        })?;
                    let u_view = DenseViewMut::new(&mut u_workspace, &u_shape, &u_strides, 0)
                        .map_err(|e| {
                            CheckedGenericFactorPlanError::Operation(OperationError::Dense(e))
                        })?;
                    let s_view = DenseViewMut::new(&mut s_workspace, &s_shape, &s_strides, 0)
                        .map_err(|e| {
                            CheckedGenericFactorPlanError::Operation(OperationError::Dense(e))
                        })?;
                    let vt_view = DenseViewMut::new(&mut vt_workspace, &vt_shape, &vt_strides, 0)
                        .map_err(|e| {
                        CheckedGenericFactorPlanError::Operation(OperationError::Dense(e))
                    })?;
                    dense
                        .svd_into(
                            D::dense_read(input_view),
                            D::dense_write(u_view),
                            D::Real::dense_write(s_view),
                            D::dense_write(vt_view),
                        )
                        .map_err(|e| {
                            CheckedGenericFactorPlanError::Operation(OperationError::Dense(e))
                        })?;
                    let mut u_thin = vec![D::zero(); matrix.rows * rank];
                    let mut vt_thin = vec![D::zero(); rank * matrix.cols];
                    copy_col_major_strided(
                        &u_workspace,
                        matrix.rows,
                        rank,
                        max_rows,
                        &mut u_thin,
                        matrix.rows,
                    );
                    copy_col_major_strided(
                        &vt_workspace,
                        rank,
                        matrix.cols,
                        max_rank,
                        &mut vt_thin,
                        rank,
                    );
                    let left = orthonormal_completion(dense, &u_thin, matrix.rows, rank)
                        .map_err(CheckedGenericFactorPlanError::from)?;
                    let v_thin = adjoint_col_major(&vt_thin, rank, matrix.cols);
                    let v_full = orthonormal_completion(dense, &v_thin, matrix.cols, rank)
                        .map_err(CheckedGenericFactorPlanError::from)?;
                    (
                        left,
                        s_workspace[..rank]
                            .iter()
                            .copied()
                            .map(Into::into)
                            .collect(),
                        adjoint_col_major(&v_full, matrix.cols, matrix.cols),
                    )
                }
            };
        svd_full_gauge(
            &mut left,
            matrix.rows,
            matrix.rows,
            &mut right,
            matrix.cols,
            matrix.cols,
        );
        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: matrix.rows,
            left,
            left_rows: matrix.rows,
            right,
            right_leading: matrix.cols,
        });
    }
    let u = build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        &matrices,
        &mut pairs,
        &row_dimensions,
        FactorSide::Left,
    )?;
    let vh = build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        &matrices,
        &mut pairs,
        &col_dimensions,
        FactorSide::Right,
    )?;
    Ok(SvdFullFactorsDyn {
        u,
        vh,
        singular_values,
        row_dimensions,
        col_dimensions,
        #[cfg(test)]
        adjoint_space: None,
    })
}

/// Checked-Generic singular values only. No factor-space publication occurs.
#[doc(hidden)]
pub fn svd_vals_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let space = input.space().space();
    let matricizations =
        generic_value_matricizations(space.structure(), input.data(), space.nout())
            .map_err(CheckedGenericFactorPlanError::from)?;
    let mut singular_values = Vec::with_capacity(matricizations.len());
    for index in 0..matricizations.len() {
        let matrix = matricizations
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let input_shape = [matrix.rows, matrix.cols];
        let input_strides = [1usize, matrix.rows];
        let input_view =
            DenseView::new(matrix.data, &input_shape, &input_strides, 0).map_err(|error| {
                CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
            })?;
        let values = dense.svd_vals(D::dense_read(input_view)).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        let mut values = D::real_spectrum(&values).map_err(|error| {
            CheckedGenericFactorPlanError::Operation(OperationError::Dense(error))
        })?;
        values.truncate(matrix.rows.min(matrix.cols));
        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values,
        });
    }
    Ok(singular_values)
}
