use super::*;

pub(super) trait HermitianReal: Float {
    fn from_f64(value: f64) -> Self;
}

impl HermitianReal for f32 {
    fn from_f64(value: f64) -> Self {
        value as Self
    }
}

impl HermitianReal for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
}

/// The tolerance of eigh's Hermiticity admission, MatrixAlgebraKit's
/// `hermitian_tol`: every coupled-sector block `A` must satisfy
/// `‖(A − Aᴴ)/2‖_F ≤ tol · ‖A‖_F` at the payload's working precision.
///
/// MAK uses an absolute `atol`, defaulting to `eps(norm(A, Inf))^(3/4)`
/// (`common/defaults.jl:default_hermitian_tol`). TeNeT keeps a relative
/// threshold so admission is invariant under rescaling `A` (#1983, #1987).
/// The defaults agree for a scalar of unit magnitude, not at arbitrary scales.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HermitianTol {
    relative: Option<f64>,
}

impl HermitianTol {
    /// `eps(real(D))^(3/4)`.
    pub const DEFAULT: Self = Self { relative: None };

    /// An explicit relative tolerance.
    ///
    /// # Errors
    ///
    /// [`OperationError::InvalidArgument`] unless `tol` is finite and
    /// non-negative.
    pub fn relative(tol: f64) -> Result<Self, OperationError> {
        if !tol.is_finite() || tol < 0.0 {
            return Err(OperationError::InvalidArgument {
                message: "hermitian_tol must be finite and non-negative",
            });
        }
        Ok(Self {
            relative: Some(tol),
        })
    }

    /// The relative tolerance at a payload precision with machine epsilon
    /// `epsilon`.
    #[doc(hidden)]
    pub fn resolve(self, epsilon: f64) -> f64 {
        self.relative.unwrap_or_else(|| epsilon.powf(0.75))
    }
}

/// `exp`'s spectral-route predicate in machine epsilons. Why not
/// [`HermitianTol::DEFAULT`]: this picks an algorithm, not an admission.
/// The eigensolver reads a Hermitian triangle; retaining this threshold avoids
/// sending additional nearly-Hermitian inputs from Padé to that spectral route.
pub(crate) const EXP_SPECTRAL_ROUTE_EPSILONS: f64 = 64.0;

#[cfg(test)]
/// Full (untruncated) Hermitian eigendecomposition `t = V * D * Vh`.
///
/// Requires an endomorphism (`codomain == domain`) with Hermitian coupled
/// blocks. Bond states are stored descending by `|eigenvalue|` per sector
/// (the shared `*_full` contract that makes truncation a prefix rule);
/// `eigenvalues` keeps the signed values in that order and `D : W <- W` is
/// their diagonal tensor.
#[derive(Clone, Debug)]
pub(crate) struct EighFull<R, D, const NOUT: usize, const NIN: usize> {
    pub d: BoundTensorMap<R, D, 1, 1>,
    pub v: BoundTensorMap<R, D, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum>,
}

/// Dynamic-rank [`EighFull`]. Carries only the eigenvector map and the O(rank)
/// spectrum; the dense diagonal `D` is built on demand by the typed [`eigh_full`]
/// wrapper (which returns a `TensorMap`), so callers that keep `D` diagonal
/// (the user layer, via compact diagonal storage) never pay the O(rank²)
/// materialization.
#[derive(Clone, Debug)]
pub struct EighFullDyn<R, D> {
    pub(super) v: BoundDynFactor<R, D>,
    pub(super) eigenvalues: Vec<SectorSpectrum>,
}

impl<R, D> EighFullDyn<R, D> {
    pub fn v(&self) -> &BoundDynFactor<R, D> {
        &self.v
    }

    pub fn eigenvalues(&self) -> &[SectorSpectrum] {
        &self.eigenvalues
    }

    pub fn into_parts(self) -> (BoundDynFactor<R, D>, Vec<SectorSpectrum>) {
        (self.v, self.eigenvalues)
    }
}

#[cfg(test)]
/// Full Hermitian eigendecomposition through the device boundary.
///
/// Before any dense call, every coupled-sector block must pass the
/// Hermiticity admission at [`HermitianTol::DEFAULT`].
pub(crate) fn eigh_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<EighFull<R, D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let dynamic = input.dynamic();
    let out = eigh_full_dyn(dense, &dynamic, HermitianTol::DEFAULT)?;
    // Materialize the dense diagonal here (the typed API returns a `TensorMap`);
    // the dyn producer no longer builds it (#56 item N).
    let d = diagonal_bond_svd_factor(dynamic.space(), &out.eigenvalues, &D::from_real)?;
    Ok(EighFull {
        d: typed_from_bound_factor(d)?,
        v: typed_from_bound_factor(out.v)?,
        eigenvalues: out.eigenvalues,
    })
}

/// Dynamic-rank [`eigh_full`]: the shared core, admitting Hermitian blocks
/// at `hermitian_tol`.
pub fn eigh_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    hermitian_tol: HermitianTol,
) -> Result<EighFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires an endomorphism (codomain == domain)",
        });
    }
    if let Some(plan) = compact_factor_plan(input.space())? {
        // The plan proves the stacking.
        require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eigh)?;
        return eigh_full_direct_regions(dense, input, &plan, hermitian_tol);
    }
    let matricizations =
        multiplicity_free_input_matricizations(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    if let InputMatricizations::Packed(matrices) = &matricizations {
        record_eigh_input_pack(matrices);
    }
    matricizations.validate_endomorphism_stacking(EIGH_FULL_STACKING)?;
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eigh)?;
    matricizations.validate_hermitian(hermitian_tol)?;
    with_input_geometry!(&matricizations, |geometry| eigh_full_scattered(
        dense,
        input,
        &matricizations,
        geometry
    ))
}

/// Eigendecomposition of an input whose tree order the positional direct
/// path cannot prove; eigenvectors scatter by tree identity.
pub(super) fn eigh_full_scattered<E, R, D, M>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    matricizations: &InputMatricizations<'_, D>,
    geometry: &[M],
) -> Result<EighFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
    M: SectorGeometry + Sync,
{
    let space = input.space().space();
    let ranks = geometry
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector(),
            kept: matrix.rows(),
        })
        .collect::<Vec<_>>();
    let v_space = build_bound_factor_space(
        input.space(),
        space.homspace(),
        SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false),
        FactorSide::Left,
    )?;
    let v_len = v_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut v_data = vec![D::zero(); v_len];
    let max_n = geometry.iter().map(SectorGeometry::rows).max().unwrap_or(0);
    let mut scratch = EighScratch::for_order(max_n);
    let mut eigenvalues = Vec::with_capacity(geometry.len());
    let index = PlacementIndex::new(geometry, &[FactorSide::Left]);
    let v_groups = SectorBlockGroups::new(v_space.space().structure(), FactorSide::Left)?;
    let v_target = v_space.space();
    in_linalg_scope(dense, |dense| {
        for (position, sector_geometry) in geometry.iter().enumerate() {
            let matrix = matricizations.get(position)?;
            let n = matrix.rows;
            let (sorted_values, vectors) = eigh_sector_stage(dense, matrix.data, n, &mut scratch)?;
            eigenvalues.push(SectorSpectrum {
                sector: matrix.sector,
                values: sorted_values,
            });
            #[cfg(test)]
            record_eigh_owned_vector_before_scatter(&vectors);
            scatter_left_sector_blocks(
                v_target,
                &mut v_data,
                sector_geometry,
                &index,
                &v_groups,
                &vectors,
                n,
            )?;
            #[cfg(test)]
            record_eigh_output_scatter::<D>(n * n);
        }
        Ok(())
    })?;

    Ok(EighFullDyn {
        v: BoundDynFactor::from_bound(v_space, v_data, space.nout(), 1)?,
        eigenvalues,
    })
}

pub(super) fn eigh_full_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
    hermitian_tol: HermitianTol,
) -> Result<EighFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source_layout, input.space().validated_layout());
    validate_endomorphism_tree_stacking(plan.source_regions.as_ref(), EIGH_FULL_STACKING)?;
    validate_hermitian_regions(input.data(), &plan.source_regions, hermitian_tol)?;

    let v_space = input.space().rebind_validated(&plan.left_layout)?;
    let v_len = plan.left_layout.required_len()?;
    let max_n = plan
        .source_regions
        .iter()
        .map(CoupledSectorRegion::rows)
        .max()
        .unwrap_or(0);
    let mut scratch = EighScratch::for_order(max_n);
    let mut eigenvalues = Vec::with_capacity(plan.routes.len());
    let mut regions = vec![None; plan.left_regions.len()];
    let mut next_left_region = 0;
    let mut output = None;

    let data = input.data();
    in_linalg_scope(dense, |dense| {
        for route in plan.routes.iter().copied() {
            let source = &plan.source_regions[route.source_region];
            let n = source.rows();
            if n == 0 {
                eigenvalues.push(SectorSpectrum {
                    sector: route.sector,
                    values: Vec::new(),
                });
                continue;
            }
            let (sorted_values, vectors) =
                eigh_sector_stage(dense, &data[source.range()], n, &mut scratch)?;
            regions[route.left_region.expect("nonzero route has left region")] = Some(vectors);
            while next_left_region < plan.left_regions.len() {
                if plan.left_regions[next_left_region].range().is_empty() {
                    next_left_region += 1;
                    continue;
                }
                let Some(region) = regions[next_left_region].take() else {
                    break;
                };
                append_owned_factor(&mut output, region, v_len);
                next_left_region += 1;
            }
            eigenvalues.push(SectorSpectrum {
                sector: route.sector,
                values: sorted_values,
            });
        }
        Ok(())
    })?;

    Ok(EighFullDyn {
        v: BoundDynFactor::from_bound(v_space, output.unwrap_or_default(), space.nout(), 1)?,
        eigenvalues,
    })
}

/// Forms the full eigenbasis of an owned compact diagonal directly, MAK
/// `eigh_full!(::Diagonal, …, ::DiagonalAlgorithm)` in TeNeT's eigenvalue
/// order.
pub(super) fn eigh_full_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    hermitian_tol: HermitianTol,
) -> Result<EighFullDyn<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let bond =
        hermitian_diagonal_bond(&MfAuthority(authority), authority, spectrum, hermitian_tol)?;
    let authority = bond.space();
    let space = authority.space();
    let plan = compact_factor_plan(authority)?.ok_or(missing_compact_plan())?;
    let by_sector = &bond.by_sector;
    let v_space = authority.rebind_validated(&plan.left_layout)?;
    let mut v_data = vec![D::zero(); plan.left_layout.required_len()?];
    let mut eigenvalues = Vec::with_capacity(plan.routes.len());
    for route in plan.routes.iter().copied() {
        let entry = by_sector[&route.sector];
        let n = entry.values.len();
        // An empty sector has no left region and writes nothing.
        let start = route
            .left_region
            .map_or(0, |index| plan.left_regions[index].range().start);
        eigenvalues.push(SectorSpectrum {
            sector: route.sector,
            values: compact_diagonal_eigh_sector(&entry.values, &mut v_data[start..start + n * n]),
        });
    }
    Ok(EighFullDyn {
        v: BoundDynFactor::from_bound(v_space, v_data, space.nout(), 1)?,
        eigenvalues,
    })
}

/// Sorts one admitted real diagonal sector into the dense EIGH order and
/// writes the matching permutation into the zeroed column-major `n x n`
/// `vectors`. Dense EIGH orders signed values ascending, then stably by
/// descending magnitude, so `-x` precedes `+x`.
fn compact_diagonal_eigh_sector<D: FactorScalar>(values: &[D], vectors: &mut [D]) -> Vec<f64> {
    let n = values.len();
    let real = |index: usize| values[index].widen_complex().re;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| real(a).total_cmp(&real(b)));
    order.sort_by(|&a, &b| real(b).abs().total_cmp(&real(a).abs()));
    for (column, &row) in order.iter().enumerate() {
        vectors[column * n + row] = D::from_real(1.0);
    }
    order.into_iter().map(real).collect()
}

/// Checked-provider full eigenbasis of an owned compact diagonal; output
/// uses the checked factor builder over the bond's region geometry.
pub(super) fn eigh_full_diagonal_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    hermitian_tol: HermitianTol,
) -> Result<EighFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let bond = hermitian_diagonal_bond(
        &CheckedAuthority(authority.provider_arc()),
        authority,
        spectrum,
        hermitian_tol,
    )?;
    let authority = bond.space();
    let space = authority.space();
    let regions = &bond.regions;
    let by_sector = &bond.by_sector;
    let mut eigenvalues = Vec::with_capacity(regions.len());
    let mut pairs = Vec::with_capacity(regions.len());
    let mut dimensions = BTreeMap::new();
    for region in regions.iter() {
        let entry = by_sector[&region.coupled()];
        let n = region.rows();
        let mut vectors = vec![D::zero(); n * n];
        eigenvalues.push(SectorSpectrum {
            sector: region.coupled(),
            values: compact_diagonal_eigh_sector(&entry.values, &mut vectors),
        });
        pairs.push(FactorPair {
            sector: region.coupled(),
            kept: n,
            left: vectors,
            left_rows: n,
            right: Vec::new(),
            right_leading: 0,
        });
        dimensions.insert(region.coupled(), n);
    }
    let v = if factor_bond_is_input_bond(space, || {
        SectorLeg::new(
            dimensions.iter().map(|(&sector, &dim)| (sector, dim)),
            false,
        )
    }) {
        factor_on_input_space(authority, regions, pairs.into_iter().map(|pair| pair.left))?
    } else {
        build_bound_factor_generic_checked(
            authority.provider_arc(),
            space.homspace(),
            regions,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )?
    };
    Ok(EighFullDyn { v, eigenvalues })
}

pub(crate) fn eigenvector_gauge<D: FactorScalar>(
    vectors: &mut [D],
    rows: usize,
    leading: usize,
    cols: usize,
) {
    for j in 0..cols {
        let (phase, needs_scaling) = phase_of_largest_abs_col(vectors, rows, leading, j);
        if needs_scaling {
            scale_col(vectors, rows, leading, j, FactorScalar::adjoint(phase));
        }
    }
}

#[cfg(test)]
/// Full general eigendecomposition `t = V * D * V^-1` (MatrixAlgebraKit
/// `eig_full`): always complex, requires an endomorphism. Bond states are
/// stored descending by `|eigenvalue|` per sector.
#[derive(Clone, Debug)]
pub(crate) struct EigFull<R, D: FactorScalar, const NOUT: usize, const NIN: usize> {
    pub d: BoundTensorMap<R, D::Eig, 1, 1>,
    pub v: BoundTensorMap<R, D::Eig, NOUT, 1>,
    pub eigenvalues: Vec<SectorSpectrum<Complex64>>,
}

/// Dynamic-rank [`EigFull`]. Spectrum + eigenvectors only; the dense diagonal
/// is materialized by the typed [`eig_full`] wrapper (see [`EighFullDyn`], #56 N).
#[derive(Clone, Debug)]
pub struct EigFullDyn<R, D: FactorScalar> {
    pub(super) v: BoundDynFactor<R, D::Eig>,
    pub(super) eigenvalues: Vec<SectorSpectrum<Complex64>>,
}

impl<R, D: FactorScalar> EigFullDyn<R, D> {
    pub fn v(&self) -> &BoundDynFactor<R, D::Eig> {
        &self.v
    }

    pub fn eigenvalues(&self) -> &[SectorSpectrum<Complex64>] {
        &self.eigenvalues
    }

    pub fn into_parts(self) -> (BoundDynFactor<R, D::Eig>, Vec<SectorSpectrum<Complex64>>) {
        (self.v, self.eigenvalues)
    }
}

#[cfg(test)]
/// Full general eigendecomposition through the device boundary.
pub(crate) fn eig_full<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<EigFull<R, D, NOUT, NIN>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let dynamic = input.dynamic();
    let out = eig_full_dyn::<E, R, D>(dense, &dynamic)?;
    // Materialize the dense diagonal here (typed API returns a `TensorMap`); the
    // dyn producer no longer builds it (#56 item N).
    let d = diagonal_bond_svd_factor(
        dynamic.space(),
        &out.eigenvalues,
        &<D::Eig as FactorScalar>::from_complex64,
    )?;
    Ok(EigFull {
        d: typed_from_bound_factor(d)?,
        v: typed_from_bound_factor(out.v)?,
        eigenvalues: out.eigenvalues,
    })
}

/// Dynamic-rank [`eig_full`].
pub fn eig_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<EigFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eig requires an endomorphism (codomain == domain)",
        });
    }
    let matricizations =
        multiplicity_free_input_matricizations(space.structure(), input.data(), space.nout())?;
    matricizations.validate_endomorphism_stacking(
        "eig_full requires identical endomorphism row/column fusion-tree stacking",
    )?;
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eig)?;

    let mut pairs: Vec<FactorPair<D::Eig>> = Vec::with_capacity(matricizations.len());
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    for index in 0..matricizations.len() {
        let matrix = matricizations.get(index)?;
        let n = matrix.rows;
        let (sorted_values, sorted_vectors) =
            eig_sector_stage(dense, matrix.data, n, |_, _| Ok(()))?;
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: n,
            left: sorted_vectors,
            left_rows: n,
            right: Vec::new(),
            right_leading: n,
        });
    }

    let v_factor = with_input_geometry!(&matricizations, |geometry| build_left_bound_factor(
        input.space(),
        space.homspace(),
        geometry,
        &mut pairs,
    ))?;
    Ok(EigFullDyn {
        v: v_factor,
        eigenvalues,
    })
}

/// Forms the full eigenbasis of an admitted owned compact diagonal directly.
pub(super) fn eig_full_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<EigFullDyn<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let bond = diagonal_bond(
        &MfAuthority(authority),
        authority,
        spectrum,
        FactorFamily::Eig,
    )?;
    validate_diagonal_eigenvalues(&bond)?;
    let authority = bond.space();
    let space = authority.space();
    let plan = compact_factor_plan(authority)?.ok_or(missing_compact_plan())?;
    let by_sector = &bond.by_sector;
    let v_space = authority.rebind_validated(&plan.left_layout)?;
    let mut v_data = vec![<D::Eig as num_traits::Zero>::zero(); plan.left_layout.required_len()?];
    let mut eigenvalues = Vec::with_capacity(plan.routes.len());
    for route in plan.routes.iter().copied() {
        let entry = by_sector[&route.sector];
        let n = entry.values.len();
        // An empty sector has no left region and writes nothing.
        let start = route
            .left_region
            .map_or(0, |index| plan.left_regions[index].range().start);
        eigenvalues.push(SectorSpectrum {
            sector: route.sector,
            values: compact_diagonal_eig_sector(&entry.values, &mut v_data[start..start + n * n]),
        });
    }
    Ok(EigFullDyn {
        v: BoundDynFactor::from_bound(v_space, v_data, space.nout(), 1)?,
        eigenvalues,
    })
}

/// Sorts one admitted complex diagonal sector by descending magnitude, with
/// ties in stored order, and writes the matching permutation into the zeroed
/// column-major `n x n` `vectors`.
fn compact_diagonal_eig_sector<D: FactorScalar>(
    values: &[D],
    vectors: &mut [D::Eig],
) -> Vec<Complex64> {
    let n = values.len();
    let mut ordered: Vec<_> = values
        .iter()
        .copied()
        .enumerate()
        .map(|(row, value)| {
            let value = value.widen_complex();
            (row, value, value.norm())
        })
        .collect();
    ordered.sort_by(|a, b| b.2.total_cmp(&a.2));
    for (column, &(row, _, _)) in ordered.iter().enumerate() {
        vectors[column * n + row] =
            <D::Eig as FactorScalar>::from_complex64(Complex64::new(1.0, 0.0));
    }
    ordered.into_iter().map(|(_, value, _)| value).collect()
}

/// Checked-provider full general eigenbasis of an owned compact diagonal;
/// output uses the checked factor builder over the bond's region geometry.
/// The dense route's eigenvector `svd_vals` rank gate is not run: every
/// singular value of a permutation is 1, so the gate cannot fail.
pub(super) fn eig_full_diagonal_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<EigFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let bond = diagonal_bond(
        &CheckedAuthority(authority.provider_arc()),
        authority,
        spectrum,
        FactorFamily::Eig,
    )?;
    validate_diagonal_eigenvalues(&bond)?;
    let authority = bond.space();
    let space = authority.space();
    let regions = &bond.regions;
    let by_sector = &bond.by_sector;
    let mut eigenvalues = Vec::with_capacity(regions.len());
    let mut pairs = Vec::with_capacity(regions.len());
    let mut dimensions = BTreeMap::new();
    for region in regions.iter() {
        let entry = by_sector[&region.coupled()];
        let n = region.rows();
        let mut vectors = vec![D::Eig::zero(); n * n];
        eigenvalues.push(SectorSpectrum {
            sector: region.coupled(),
            values: compact_diagonal_eig_sector(&entry.values, &mut vectors),
        });
        pairs.push(FactorPair {
            sector: region.coupled(),
            kept: n,
            left: vectors,
            left_rows: n,
            right: Vec::new(),
            right_leading: 0,
        });
        dimensions.insert(region.coupled(), n);
    }
    let v = if factor_bond_is_input_bond(space, || {
        SectorLeg::new(
            dimensions.iter().map(|(&sector, &dim)| (sector, dim)),
            false,
        )
    }) {
        factor_on_input_space(authority, regions, pairs.into_iter().map(|pair| pair.left))?
    } else {
        build_bound_factor_generic_checked(
            authority.provider_arc(),
            space.homspace(),
            regions,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )?
    };
    Ok(EigFullDyn { v, eigenvalues })
}

/// Hermitian eigenvalues of a compact diagonal: the real parts, after the
/// dense route's Hermiticity check, in TeNeT's eigenvalue order.
fn eigh_vals_diagonal<A, R, D>(
    authority: &A,
    space: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    hermitian_tol: HermitianTol,
) -> Result<Vec<SectorSpectrum>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
    D: FactorScalar,
{
    let bond = hermitian_diagonal_bond(authority, space, spectrum, hermitian_tol)?;
    let mut result = Vec::with_capacity(bond.len());
    for region in bond.iter() {
        let mut values: Vec<f64> = bond
            .entry(region)
            .values
            .iter()
            .map(|&value| value.widen_complex().re)
            .collect();
        validate_real_eigenvalues(&values)?;
        // Dense EIGH first orders signed values ascending, then stably sorts
        // by magnitude. The first pass preserves its tie order for ±x.
        values.sort_by(f64::total_cmp);
        values.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
        result.push(SectorSpectrum {
            sector: region.coupled(),
            values,
        });
    }
    Ok(result)
}

/// A finite compact diagonal that passes the dense route's
/// relative-Frobenius Hermiticity check at `hermitian_tol`, applied to the
/// diagonal matrix itself; its real parts are its eigenvalues.
fn hermitian_diagonal_bond<'a, A, R, D>(
    authority: &A,
    space: &'a BoundDynamicFusionMapSpace<R>,
    spectrum: &'a [SectorSpectrum<D>],
    hermitian_tol: HermitianTol,
) -> Result<DiagonalBond<'a, R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
    D: FactorScalar,
{
    let bond = diagonal_bond(authority, space, spectrum, FactorFamily::Eigh)?;
    let tol = hermitian_tol.resolve(D::epsilon());
    for region in bond.iter() {
        let values = &bond.entry(region).values;
        let hermitian = if D::epsilon() == f32::EPSILON as f64 {
            normwise_hermitian_diagonal::<D, f32>(values, f32::from_f64(tol))
        } else if D::epsilon() == f64::EPSILON {
            normwise_hermitian_diagonal::<D, f64>(values, tol)
        } else {
            false
        };
        if !hermitian {
            return Err(OperationError::InvalidArgument {
                message: "eigh requires Hermitian coupled-sector blocks",
            }
            .into());
        }
    }
    Ok(bond)
}

/// [`normwise_hermitian`] of the diagonal matrix with diagonal `values`,
/// without forming it. Off-diagonal zeros and the zero real residuals of the
/// diagonal add nothing to either scaled sum, and both sums visit the
/// diagonal in the same order as the dense scan, so the decision is equal.
fn normwise_hermitian_diagonal<D: FactorScalar, R: HermitianReal>(values: &[D], tol: R) -> bool {
    let mut input = ScaledFrobenius::zero();
    for &value in values {
        let value = value.widen_complex();
        let re = R::from_f64(value.re);
        let im = R::from_f64(value.im);
        if !re.is_finite() || !im.is_finite() || !input.add_complex(re, im) {
            return false;
        }
    }
    if input.scale == R::zero() {
        return true;
    }
    let mut residual = ScaledFrobenius::zero();
    for &value in values {
        let value = value.widen_complex();
        let re = R::from_f64(value.re) / input.scale - R::from_f64(value.re) / input.scale;
        let im = R::from_f64(value.im) / input.scale + R::from_f64(value.im) / input.scale;
        if !residual.add_complex(re, im) {
            return false;
        }
    }
    residual.scaled_norm(R::one()) <= (R::one() + R::one()) * tol * input.sum_squares.sqrt()
}

/// General eigenvalues of a compact diagonal: the values themselves in
/// TeNeT's eigenvalue order (descending magnitude, stable), after the dense
/// route's eigenvalue check.
fn eig_vals_diagonal<A, R, D>(
    authority: &A,
    space: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Vec<SectorSpectrum<Complex64>>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
    D: FactorScalar,
{
    let bond = diagonal_bond(authority, space, spectrum, FactorFamily::Eig)?;
    let mut result = Vec::with_capacity(bond.len());
    for region in bond.iter() {
        let mut values: Vec<Complex64> = bond
            .entry(region)
            .values
            .iter()
            .map(|&value| value.widen_complex())
            .collect();
        validate_complex_eigenvalues(&values)?;
        values.sort_by(|a, b| b.norm().total_cmp(&a.norm()));
        result.push(SectorSpectrum {
            sector: region.coupled(),
            values,
        });
    }
    Ok(result)
}

fn missing_compact_plan() -> OperationError {
    OperationError::UnsupportedTensorContractScope {
        message: "canonical bond layout has no compact factor plan",
    }
}

#[cfg(test)]
/// All Hermitian eigenvalues per coupled sector, descending by magnitude
/// (MatrixAlgebraKit `eigh_vals`).
///
/// Admits Hermitian blocks at [`HermitianTol::DEFAULT`].
pub(crate) fn eigh_vals<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    eigh_vals_dyn(dense, &input.dynamic(), HermitianTol::DEFAULT)
}

/// Dynamic-rank [`eigh_vals`]: the dense stage, the same in every fusion mode.
pub fn eigh_vals_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    hermitian_tol: HermitianTol,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let space = input.space().space();
    // Values-only: per sector call the no-vector Hermitian eig (`eigh_vals`,
    // LAPACK `job='N'`) and keep the spectrum sorted descending by magnitude.
    // Skips the eigenvector space/buffer, the vector reorder, gauge-fixing, and
    // the block scatter that `eigh_full_dyn` did only to discard here. The sort
    // is stable, so equal-magnitude ties keep LAPACK order — bit-for-bit the
    // ordering `eigh_full_dyn` produces (it breaks ties by original index).
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires an endomorphism (codomain == domain)",
        });
    }
    let matricizations = value_matricizations(space.structure(), input.data(), space.nout())?;
    matricizations.validate_endomorphism_stacking(
        "eigh_vals requires identical endomorphism row/column fusion-tree stacking",
    )?;
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eigh)?;
    matricizations.validate_hermitian(hermitian_tol)?;
    eigh_vals_spectra(dense, &matricizations)
}

#[cfg(test)]
/// All general eigenvalues per coupled sector, descending by magnitude
/// (MatrixAlgebraKit `eig_vals`).
pub(crate) fn eig_vals<E, R, D, const NOUT: usize, const NIN: usize>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D, NOUT, NIN>,
) -> Result<Vec<SectorSpectrum<Complex64>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    eig_vals_dyn::<E, R, D>(dense, &input.dynamic())
}

/// Dynamic-rank [`eig_vals`]: the dense stage, the same in every fusion mode.
pub fn eig_vals_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum<Complex64>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let space = input.space().space();
    // Values-only: per sector call the no-vector general eig (`eig_vals`, LAPACK
    // `job='N'`) and keep the complex spectrum sorted descending by magnitude.
    // Skips the eigenvector reorder, gauge-fixing, and the factor-pair block
    // assembly that `eig_full_dyn` did only to discard here. LAPACK's QR
    // iteration yields the same eigenvalues regardless of `jobvr`, and the sort
    // is stable, so this matches `eig_full_dyn`'s ordering bit-for-bit.
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eig requires an endomorphism (codomain == domain)",
        });
    }
    let matricizations = value_matricizations(space.structure(), input.data(), space.nout())?;
    matricizations.validate_endomorphism_stacking(
        "eig_vals requires identical endomorphism row/column fusion-tree stacking",
    )?;
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eig)?;
    eig_vals_spectra(dense, &matricizations)
}

/// Reorder scratch for [`eigh_sector_stage`], sized for the largest sector.
struct EighScratch<D> {
    order: Vec<usize>,
    visited: Vec<bool>,
    column: Vec<D>,
}

impl<D: FactorScalar> EighScratch<D> {
    fn for_order(max_n: usize) -> Self {
        Self {
            order: Vec::with_capacity(max_n),
            visited: vec![false; max_n],
            column: vec![D::zero(); max_n],
        }
    }
}

/// Hermitian eigendecomposition of one `n x n` coupled-sector matrix: values
/// descending by magnitude (stable on ties), eigenvectors reordered to match
/// and phase-gauged.
#[inline]
fn eigh_sector_stage<E, D>(
    dense: &mut E,
    matrix: &[D],
    n: usize,
    scratch: &mut EighScratch<D>,
) -> Result<(Vec<f64>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (real_values, mut vectors) = compact_eigh_owned(dense, matrix, n)?;
    validate_real_eigenvalues(&real_values)?;
    let EighScratch {
        order,
        visited,
        column,
    } = scratch;
    order.clear();
    order.extend(0..n);
    // Reorder bond states descending by |eigenvalue| (stable on ties).
    order.sort_by(|&a, &b| {
        real_values[b]
            .abs()
            .total_cmp(&real_values[a].abs())
            .then(a.cmp(&b))
    });
    let sorted_values = order.iter().map(|&index| real_values[index]).collect();
    reorder_columns_in_place(&mut vectors, n, order, visited, column);
    eigenvector_gauge(&mut vectors, n, n, n);
    Ok((sorted_values, vectors))
}

/// General eigendecomposition of one `n x n` endomorphism sector:
/// values descending by magnitude (stable on ties), eigenvectors reordered
/// to match and phase-gauged. `rank_gate` runs on the raw eigenvectors after
/// the eigenvalue check; the checked mode passes its diagonalizability gate
/// there until #1798 decides it, the multiplicity-free mode a no-op.
fn eig_sector_stage<E, D>(
    dense: &mut E,
    matrix: &[D],
    n: usize,
    rank_gate: impl FnOnce(&mut E, &[D::Eig]) -> Result<(), OperationError>,
) -> Result<(Vec<Complex64>, Vec<D::Eig>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let shape = [n, n];
    let strides = [1usize, n];
    let view = DenseView::new(matrix, &shape, &strides, 0).map_err(OperationError::Dense)?;
    let outputs = dense
        .eig(D::dense_read(view))
        .map_err(OperationError::Dense)?;
    if outputs.len() != 2 {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "dense eig must return exactly (values, vectors)",
        });
    }
    validate_dense_shape(outputs[0].shape(), &[n])?;
    validate_dense_shape(outputs[1].shape(), &[n, n])?;
    let values =
        <D::Eig as FactorScalar>::dense_slice(&outputs[0]).map_err(OperationError::Dense)?;
    let vectors =
        <D::Eig as FactorScalar>::dense_slice(&outputs[1]).map_err(OperationError::Dense)?;
    let complex_values: Vec<Complex64> =
        values.iter().map(|&value| value.widen_complex()).collect();
    validate_complex_eigenvalues(&complex_values)?;
    rank_gate(dense, vectors)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        complex_values[b]
            .norm()
            .total_cmp(&complex_values[a].norm())
            .then(a.cmp(&b))
    });
    let sorted_values: Vec<Complex64> = order.iter().map(|&index| complex_values[index]).collect();
    let mut sorted_vectors = vec![<D::Eig as num_traits::Zero>::zero(); n * n];
    for (position, &index) in order.iter().enumerate() {
        sorted_vectors[position * n..(position + 1) * n]
            .copy_from_slice(&vectors[index * n..(index + 1) * n]);
    }
    eigenvector_gauge(&mut sorted_vectors, n, n, n);
    Ok((sorted_values, sorted_vectors))
}

/// Hermitian eigenvalues of every sector through the no-vector solver,
/// descending by magnitude (stable on ties, so LAPACK order breaks them).
fn eigh_vals_spectra<E, D>(
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut eigenvalues = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        let n = matrix.rows;
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let values_tensor = dense
            .eigh_vals(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        let mut sorted = D::real_spectrum(&values_tensor).map_err(OperationError::Dense)?;
        sorted.truncate(n);
        validate_real_eigenvalues(&sorted)?;
        sorted.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted,
        });
    }
    Ok(eigenvalues)
}

/// General eigenvalues of every sector through the no-vector solver,
/// descending by magnitude (stable on ties).
fn eig_vals_spectra<E, D>(
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
) -> Result<Vec<SectorSpectrum<Complex64>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut eigenvalues = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        let n = matrix.rows;
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let values_tensor = dense
            .eig_vals(D::dense_read(view))
            .map_err(OperationError::Dense)?;
        validate_dense_shape(values_tensor.shape(), &[n])?;
        let values =
            <D::Eig as FactorScalar>::dense_slice(&values_tensor).map_err(OperationError::Dense)?;
        let mut sorted: Vec<Complex64> = values[..n].iter().map(|&v| v.widen_complex()).collect();
        validate_complex_eigenvalues(&sorted)?;
        sorted.sort_by(|a, b| b.norm().total_cmp(&a.norm()));
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted,
        });
    }
    Ok(eigenvalues)
}

pub(super) fn compact_eigh_owned<E, D>(
    dense: &mut E,
    input: &[D],
    order: usize,
) -> Result<(Vec<f64>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let shape = [order, order];
    let strides = [1usize, order];
    let input = DenseView::new(input, &shape, &strides, 0).map_err(OperationError::Dense)?;
    let mut outputs = dense
        .eigh(D::dense_read(input))
        .map_err(OperationError::Dense)?;
    if outputs.len() != 2 {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eigh_into",
            message: "dense EIGH must return exactly (values, vectors)".to_string(),
        }));
    }
    let values = compact_real_spectrum_owned::<D>(outputs.remove(0), &[order], "eigh_into")?;
    let vectors =
        compact_factor_output_owned::<D>(outputs.remove(0), &[order, order], "eigh_into")?;
    Ok((values, vectors))
}

pub(super) fn scaled_antihermitian_difference<D: FactorScalar, R: HermitianReal>(
    data: &[D],
    n: usize,
    row: usize,
    col: usize,
    scale: R,
) -> (R, R) {
    let upper = data[row + n * col].widen_complex();
    let lower = data[col + n * row].widen_complex();
    let residual_re = R::from_f64(upper.re) / scale - R::from_f64(lower.re) / scale;
    let residual_im = R::from_f64(upper.im) / scale + R::from_f64(lower.im) / scale;
    (residual_re, residual_im)
}

#[derive(Clone, Copy)]
pub(super) struct ScaledFrobenius<R> {
    pub(super) scale: R,
    pub(super) sum_squares: R,
}

impl<R: HermitianReal> ScaledFrobenius<R> {
    pub(super) fn zero() -> Self {
        Self {
            scale: R::zero(),
            sum_squares: R::one(),
        }
    }

    pub(super) fn add(&mut self, magnitude: R) -> bool {
        if !magnitude.is_finite() {
            return false;
        }
        if magnitude == R::zero() {
            return true;
        }
        if self.scale < magnitude {
            let ratio = self.scale / magnitude;
            self.sum_squares = R::one() + self.sum_squares * ratio * ratio;
            self.scale = magnitude;
        } else {
            let ratio = magnitude / self.scale;
            self.sum_squares = self.sum_squares + ratio * ratio;
        }
        true
    }

    pub(super) fn add_complex(&mut self, re: R, im: R) -> bool {
        self.add(re.abs()) && self.add(im.abs())
    }

    pub(super) fn scaled_norm(self, scale: R) -> R {
        if self.scale == R::zero() {
            R::zero()
        } else {
            (self.scale / scale) * self.sum_squares.sqrt()
        }
    }
}

pub(super) fn hermitian_residual_norm<D: FactorScalar, R: HermitianReal>(
    data: &[D],
    n: usize,
    input_scale: R,
) -> Option<ScaledFrobenius<R>> {
    const BLOCK_SIZE: usize = 32;
    let mut residual = ScaledFrobenius::zero();
    for block_col in (0..n).step_by(BLOCK_SIZE) {
        let block_width = BLOCK_SIZE.min(n - block_col);
        for local_col in 0..block_width {
            let col = block_col + local_col;
            for local_row in 0..=local_col {
                let row = block_col + local_row;
                let (re, im) =
                    scaled_antihermitian_difference::<D, R>(data, n, row, col, input_scale);
                if !residual.add_complex(re, im) || (row != col && !residual.add_complex(re, im)) {
                    return None;
                }
            }
        }

        for block_row in (0..block_col).step_by(BLOCK_SIZE) {
            for local_col in 0..block_width {
                let col = block_col + local_col;
                for local_row in 0..BLOCK_SIZE {
                    let row = block_row + local_row;
                    let (re, im) =
                        scaled_antihermitian_difference::<D, R>(data, n, row, col, input_scale);
                    if !residual.add_complex(re, im) || !residual.add_complex(re, im) {
                        return None;
                    }
                }
            }
        }
    }
    Some(residual)
}

/// Tests `||(A - A†)/2||_F <= tol * ||A||_F`.
///
/// The scaled sums keep that relative decision stable when either norm would
/// overflow or underflow if formed directly.
pub(super) fn normwise_hermitian<D: FactorScalar, R: HermitianReal>(
    data: &[D],
    n: usize,
    tol: R,
) -> bool {
    let mut input = ScaledFrobenius::zero();
    for &value in data {
        let value = value.widen_complex();
        let re = R::from_f64(value.re);
        let im = R::from_f64(value.im);
        if !re.is_finite() || !im.is_finite() {
            return false;
        }
        if !input.add_complex(re, im) {
            return false;
        }
    }

    if input.scale == R::zero() {
        return true;
    }
    let Some(residual) = hermitian_residual_norm::<D, R>(data, n, input.scale) else {
        return false;
    };
    residual.scaled_norm(R::one()) <= (R::one() + R::one()) * tol * input.sum_squares.sqrt()
}

pub(super) fn validate_hermitian_matrix_shape<D>(
    data: &[D],
    rows: usize,
    cols: usize,
) -> Result<(), OperationError> {
    let expected = rows
        .checked_mul(cols)
        .ok_or(OperationError::ElementCountOverflow)?;
    if data.len() != expected {
        return Err(OperationError::ElementCountMismatch {
            expected,
            actual: data.len(),
        });
    }
    if rows != cols {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires square coupled-sector matrices",
        });
    }
    Ok(())
}

/// Scale-invariant normwise hermiticity predicate at the working precision of
/// `D`, with relative tolerance `tol`.
///
/// Extracted from [`validate_hermitian_matrix_contents`] so the `exp` dispatch
/// (issue #577) can ask the question without provoking — and then having to
/// interpret — an EIGH failure. Same measure; `exp` passes its own route
/// threshold ([`EXP_SPECTRAL_ROUTE_EPSILONS`]).
pub(super) fn hermitian_matrix_contents<D: FactorScalar>(data: &[D], n: usize, tol: f64) -> bool {
    if D::epsilon() == f32::EPSILON as f64 {
        normwise_hermitian::<D, f32>(data, n, f32::from_f64(tol))
    } else if D::epsilon() == f64::EPSILON {
        normwise_hermitian::<D, f64>(data, n, tol)
    } else {
        false
    }
}

pub(super) fn validate_hermitian_matrix_contents<D: FactorScalar>(
    data: &[D],
    n: usize,
    hermitian_tol: HermitianTol,
) -> Result<(), OperationError> {
    if !hermitian_matrix_contents(data, n, hermitian_tol.resolve(D::epsilon())) {
        return Err(OperationError::InvalidArgument {
            message: "eigh requires Hermitian coupled-sector blocks",
        });
    }
    Ok(())
}

pub(super) fn validate_hermitian_matricizations<D: FactorScalar>(
    matricizations: &[SectorMatricization<D>],
    hermitian_tol: HermitianTol,
) -> Result<(), OperationError> {
    for matrix in matricizations {
        validate_hermitian_matrix_shape(&matrix.data, matrix.rows, matrix.cols)?;
    }
    for matrix in matricizations {
        validate_hermitian_matrix_contents(&matrix.data, matrix.rows, hermitian_tol)?;
    }
    Ok(())
}

#[doc(hidden)]
pub fn validate_hermitian_regions<D: FactorScalar>(
    data: &[D],
    regions: &[CoupledSectorRegion],
    hermitian_tol: HermitianTol,
) -> Result<(), OperationError> {
    for region in regions {
        let range = region.range();
        let matrix = data
            .get(range.clone())
            .ok_or(OperationError::ElementCountMismatch {
                expected: range.end,
                actual: data.len(),
            })?;
        validate_hermitian_matrix_shape(matrix, region.rows(), region.cols())?;
    }
    for region in regions {
        let range = region.range();
        let matrix = data
            .get(range.clone())
            .ok_or(OperationError::ElementCountMismatch {
                expected: range.end,
                actual: data.len(),
            })?;
        validate_hermitian_matrix_contents(matrix, region.rows(), hermitian_tol)?;
    }
    Ok(())
}

/// Is this an endomorphism whose coupled-sector blocks are all Hermitian?
///
/// The `exp` dispatch (issue #577) needs the Hermitian question answered
/// *separately* from the eigendecomposition: the spectral route stays for
/// Hermitian input, and everything else goes to blockwise Padé. Inferring it
/// from a failed EIGH would conflate hermiticity with a backend failure, so
/// this asks directly, over the same direct-region / packed matricization
/// split and relative Frobenius measure as [`eigh_full_dyn`], with the
/// separate threshold [`EXP_SPECTRAL_ROUTE_EPSILONS`].
///
/// A non-endomorphism, a malformed layout or a non-square block is still an
/// error — only non-hermiticity is `Ok(false)`. Nonfinite entries make
/// the predicate `false`, so they arrive at the Padé route, which rejects them
/// in its own words.
///
/// Cost is `O(Σ_c n_c²)`, one pass over the blocks, against the `O(Σ_c n_c³)`
/// factorization that follows.
pub(crate) fn is_hermitian_endomorphism_dyn<R, D>(
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "eigh requires an endomorphism (codomain == domain)",
        });
    }
    let exp_route_tol = EXP_SPECTRAL_ROUTE_EPSILONS * D::epsilon();
    // Why `checked_sector_regions` and not `compact_factor_plan` as
    // `eigh_full_dyn` does: the plan is `Some` exactly when the regions are
    // (`build_compact_factor_plan` returns early otherwise), and building it
    // also builds the factor bond spaces — work a yes/no question must not pay
    // for on the retained Hermitian route.
    if let Some(regions) = checked_sector_regions(space.structure(), space.nout())? {
        validate_endomorphism_tree_stacking(regions.as_ref(), EXP_STACKING)?;
        for region in regions.iter() {
            let range = region.range();
            let matrix = data_region(input.data(), &range)?;
            validate_hermitian_matrix_shape(matrix, region.rows(), region.cols())?;
        }
        for region in regions.iter() {
            let range = region.range();
            let matrix = data_region(input.data(), &range)?;
            if !hermitian_matrix_contents(matrix, region.rows(), exp_route_tol) {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    validate_endomorphism_tree_stacking(&matricizations, EXP_STACKING)?;
    for matrix in &matricizations {
        validate_hermitian_matrix_shape(&matrix.data, matrix.rows, matrix.cols)?;
    }
    Ok(matricizations
        .iter()
        .all(|matrix| hermitian_matrix_contents(&matrix.data, matrix.rows, exp_route_tol)))
}

pub(super) fn invalid_eigenvalues() -> OperationError {
    OperationError::InvalidArgument {
        message: "eigenvalues must be finite",
    }
}

pub(super) fn validate_real_eigenvalues(values: &[f64]) -> Result<(), OperationError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_eigenvalues())
    }
}

#[cfg(test)]
pub(crate) fn validate_real_eigenvalues_for_test(values: &[f64]) -> Result<(), OperationError> {
    validate_real_eigenvalues(values)
}

pub(super) fn validate_complex_eigenvalues(values: &[Complex64]) -> Result<(), OperationError> {
    if values
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite() && value.norm().is_finite())
    {
        Ok(())
    } else {
        Err(invalid_eigenvalues())
    }
}

/// Checked-Generic full Hermitian eigendecomposition. The exact source
/// provider remains the authority for the eigenvector factor.
#[doc(hidden)]
pub(crate) fn eigh_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    hermitian_tol: HermitianTol,
) -> Result<EighFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope {
                message: "eigh requires an endomorphism (codomain == domain)",
            },
        ));
    }
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    // Why not trust equal product spaces alone: outer-multiplicity vertices
    // are part of a tree key, so Hermitian coordinates require identical full
    // tree stacking, not merely equal coupled-sector dimensions.
    matrices
        .validate_endomorphism_stacking(EIGH_FULL_STACKING)
        .map_err(CheckedGenericFactorPlanError::from)?;
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eigh)?;
    matrices
        .validate_hermitian(hermitian_tol)
        .map_err(CheckedGenericFactorPlanError::from)?;

    let max_n = with_input_geometry!(&matrices, |geometry| geometry
        .iter()
        .map(SectorGeometry::rows)
        .max()
        .unwrap_or(0));
    let mut scratch = EighScratch::for_order(max_n);
    let mut eigenvalues = Vec::with_capacity(matrices.len());
    let mut pairs = Vec::with_capacity(matrices.len());
    in_linalg_scope(dense, |dense| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            let n = matrix.rows;
            let (sorted_values, vectors) = eigh_sector_stage(dense, matrix.data, n, &mut scratch)?;
            eigenvalues.push(SectorSpectrum {
                sector: matrix.sector,
                values: sorted_values,
            });
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: n,
                left: vectors,
                left_rows: n,
                right: Vec::new(),
                right_leading: 0,
            });
        }
        Ok(())
    })
    .map_err(CheckedGenericFactorPlanError::from)?;
    let dimensions = with_input_geometry!(&matrices, |geometry| geometry
        .iter()
        .map(|matrix| (matrix.sector(), matrix.rows()))
        .collect::<BTreeMap<_, _>>());
    #[cfg(test)]
    record_checked_eigh_pair_pointers(&pairs);
    let v = with_input_geometry!(&matrices, |geometry| build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        geometry,
        &mut pairs,
        &dimensions,
        FactorSide::Left,
    ))?;
    Ok(EighFullDyn { v, eigenvalues })
}

/// Checked-mode diagonalizability gate on raw eigenvectors; stays an explicit
/// call until #1798 decides whether it belongs to every mode.
fn eig_rank_gate<E, D>(dense: &mut E, vectors: &[D::Eig], n: usize) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    if vectors.iter().any(|&value| {
        let value = value.widen_complex();
        !value.re.is_finite() || !value.im.is_finite()
    }) {
        return Err(eig_not_numerically_diagonalizable());
    }
    let shape = [n, n];
    let strides = [1usize, n];
    let vector_view =
        DenseView::new(vectors, &shape, &strides, 0).map_err(OperationError::Dense)?;
    let singular_values = dense
        .svd_vals(<D::Eig as DenseBlockScalar>::dense_read(vector_view))
        .map_err(OperationError::Dense)?;
    validate_dense_shape(singular_values.shape(), &[n])?;
    let singular_values =
        <D::Eig as FactorScalar>::real_spectrum(&singular_values).map_err(OperationError::Dense)?;
    validate_eigenvector_singular_values(&singular_values, n, <D::Eig as FactorScalar>::epsilon())
}

pub(super) fn eig_not_numerically_diagonalizable() -> OperationError {
    OperationError::InvalidArgument {
        message: "eig requires a numerically diagonalizable coupled-sector matrix",
    }
}

pub(crate) fn validate_eigenvector_singular_values(
    singular_values: &[f64],
    n: usize,
    epsilon: f64,
) -> Result<(), OperationError> {
    if singular_values.len() != n || singular_values.iter().any(|value| !value.is_finite()) {
        return Err(eig_not_numerically_diagonalizable());
    }
    let sigma_max = singular_values.first().copied().unwrap_or(0.0);
    let tolerance = n as f64 * epsilon * sigma_max;
    if singular_values.iter().all(|&sigma| sigma > tolerance) {
        Ok(())
    } else {
        Err(eig_not_numerically_diagonalizable())
    }
}

/// Checked-Generic full general eigendecomposition. All input components and
/// dense results are validated before the exact source provider is asked to
/// admit either output factor.
#[doc(hidden)]
pub(crate) fn eig_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<EigFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    if space.homspace().codomain() != space.homspace().domain() {
        return Err(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope {
                message: "eig requires an endomorphism (codomain == domain)",
            },
        ));
    }
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    matrices
        .validate_endomorphism_stacking(
            "eig_full requires identical endomorphism row/column fusion-tree stacking",
        )
        .map_err(CheckedGenericFactorPlanError::from)?;
    // Every sector is read before the shared finite-input stage reports, so
    // a matricization error still precedes it.
    let mut finite = Ok(());
    for index in 0..matrices.len() {
        let matrix = matrices
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        if finite.is_ok() {
            finite = require_finite_factor_input(matrix.data.iter().copied(), FactorFamily::Eig);
        }
    }
    finite.map_err(CheckedGenericFactorPlanError::from)?;

    let mut pairs: Vec<FactorPair<D::Eig>> = Vec::with_capacity(matrices.len());
    let mut eigenvalues = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let n = matrix.rows;
        let (sorted_values, sorted_vectors) =
            eig_sector_stage(dense, matrix.data, n, |dense, vectors| {
                eig_rank_gate::<E, D>(dense, vectors, n)
            })
            .map_err(CheckedGenericFactorPlanError::from)?;
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: n,
            left: sorted_vectors,
            left_rows: n,
            right: Vec::new(),
            right_leading: 0,
        });
    }

    let v = with_input_geometry!(&matrices, |geometry| {
        let dimensions = geometry
            .iter()
            .map(|matrix| (matrix.sector(), matrix.rows()))
            .collect::<BTreeMap<_, _>>();
        build_bound_factor_generic_checked(
            provider,
            space.homspace(),
            geometry,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )
    })?;
    Ok(EigFullDyn { v, eigenvalues })
}

/// Hermitian eigenvalues of `source` in fusion mode `M`, descending by
/// magnitude per coupled sector. A compact diagonal is read directly; only
/// dense storage leases an executor.
#[doc(hidden)]
pub fn eigh_vals_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
    hermitian_tol: HermitianTol,
) -> Result<Vec<SectorSpectrum>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(
        lease,
        source,
        None,
        |space, spectrum| eigh_vals_diagonal(&M::authority(space), space, spectrum, hermitian_tol),
        |dense, input| Ok(eigh_vals_dyn(dense, input, hermitian_tol)?),
    )
    .map(|(values, _)| values)
}

/// General eigenvalues of `source` in fusion mode `M`, descending by
/// magnitude per coupled sector. A compact diagonal is read directly; only
/// dense storage leases an executor.
#[doc(hidden)]
pub fn eig_vals_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Vec<SectorSpectrum<Complex64>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(
        lease,
        source,
        None,
        |space, spectrum| eig_vals_diagonal(&M::authority(space), space, spectrum),
        |dense, input| Ok(eig_vals_dyn(dense, input)?),
    )
    .map(|(values, _)| values)
}

/// Hermitian eigendecomposition of `source` in fusion mode `M`, with the route that produced it;
/// only dense storage leases an executor.
#[doc(hidden)]
pub fn eigh_full_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
    hermitian_tol: HermitianTol,
) -> Result<(EighFullDyn<R, D>, FactorRoute), M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(
        lease,
        source,
        None,
        |space, spectrum| M::eigh_full_diagonal(space, spectrum, hermitian_tol),
        |dense, input| M::eigh_full_dense(dense, input, hermitian_tol),
    )
}

/// General eigendecomposition of `source` in fusion mode `M`, with the route that produced it;
/// only dense storage leases an executor.
#[doc(hidden)]
pub fn eig_full_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<(EigFullDyn<R, D>, FactorRoute), M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(lease, source, None, M::eig_full_diagonal, M::eig_full_dense)
}

/// The dense route's eigenvalue check (`validate_complex_eigenvalues`) on a
/// compact diagonal's eigenvalues, which are its values: an overflowing
/// magnitude is refused as the dense route refuses it.
fn validate_diagonal_eigenvalues<R, D: FactorScalar>(
    bond: &DiagonalBond<'_, R, D>,
) -> Result<(), OperationError> {
    for region in bond.iter() {
        for &value in &bond.entry(region).values {
            let value = value.widen_complex();
            if !(value.re.is_finite() && value.im.is_finite() && value.norm().is_finite()) {
                return Err(invalid_eigenvalues());
            }
        }
    }
    Ok(())
}
