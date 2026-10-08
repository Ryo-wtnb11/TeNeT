use super::*;

use tenet_core::PreparedBlockStructure;

mod sealed {
    pub trait Sealed {}
}

/// The only mode-dependent step of a factor publication: the provider queries
/// that build a factor's fusion space and size its bond.
///
/// Transaction contract. `stage` performs every provider query of the factor
/// space; `commit` publishes it. `commit` runs `pre_commit`, the shared
/// provider-free placement validation, before publishing exactly when the
/// staged space is not yet published:
/// - `CheckedAuthority` enumerates in `stage` and publishes in `commit`, so a
///   malformed placement fails before any bound space exists;
/// - `MfAuthority`'s `stage` is the cached `derive_from_final_homspace`,
///   which has already published the layout, so its `commit` skips
///   `pre_commit`; the scatter validates placements lazily and the canonical
///   fast path does no placement hashing.
///
/// Why not one `factor_space(source, hom)` call (target design §2.2):
/// publishing before validation would reorder the checked error precedence,
/// which is an observable contract.
pub trait FactorSpaceAuthority<R>: sealed::Sealed {
    type Error: From<OperationError>;
    type Staged;

    fn stage(&self, homspace: FusionTreeHomSpace) -> Result<Self::Staged, Self::Error>;

    fn commit(
        &self,
        staged: Self::Staged,
        pre_commit: impl FnOnce(&SectorStructure) -> Result<(), OperationError>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error>;

    /// Reduced dimension of every coupled sector of `side`.
    fn coupled_dimensions(
        &self,
        side: &FusionProductSpace,
    ) -> Result<BTreeMap<SectorId, usize>, Self::Error>;

    /// Whether `space`'s codomain and domain have equal coupled-sector
    /// dimensions (the inverse and solve admission).
    fn isomorphic(&self, space: &BoundDynamicFusionMapSpace<R>) -> Result<bool, Self::Error>;

    type RootError;

    /// An output space that needs no shared pre-commit validation, such as the
    /// swapped space of an inverse or a solve.
    ///
    /// Why a separate path from `stage` + `commit`: its errors are the root
    /// constructor's (`RootError`), which the checked facade reports as the
    /// public `GenericTensorError::Structure` variant, not as a factor-plan
    /// error. The checked root constructor also checks the provider's fusion
    /// style before it enumerates, while `commit` checks it after.
    fn output_space(
        &self,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::RootError>;
}

/// A fusion mode's factorization authority, chosen statically by the mode
/// marker. Each factorization has one entry generic over it, so a mode
/// supplies only its factor-space authority and its error type.
///
/// Why not a runtime flag or one entry per mode: the two authorities have
/// different provider bounds and error types, and monomorphizing over the
/// marker keeps the multiplicity-free path free of the checked one's costs.
pub trait FactorMode<R>: sealed::Sealed {
    type Error: From<OperationError>;

    /// The error of [`FactorSpaceAuthority::output_space`].
    type RootError;

    /// The factor-space authority of `space`'s provider.
    fn authority(
        space: &BoundDynamicFusionMapSpace<R>,
    ) -> impl FactorSpaceAuthority<R, Error = Self::Error, RootError = Self::RootError> + '_;

    // The dense QR/LQ stages publish through this mode's own factor-space
    // builders. Why not one builder generic over the authority: #1830
    // measured it and kept publication mode-specific (the multiplicity-free
    // one-sided pair is cheaper; checked has no direct-region plan, #1960).
    fn qr_compact_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Qr<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar;

    fn qr_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Qr<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar;

    fn lq_compact_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Lq<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar;

    fn lq_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Lq<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar;

    // SVD (#1862 D3, owned by #1994): the multiplicity-free and checked
    // diagonal and dense stages publish through their own builders, and the
    // bond of a spectrum factor (`S`, `D`) is fresh in multiplicity-free mode
    // and the input bond on a checked diagonal route.

    /// Whether a full SVD of a compact diagonal runs the compact route:
    /// multiplicity-free compact and full factor spaces coincide on a
    /// diagonal and that mode publishes them without a layout check.
    const SVD_FULL_DIAGONAL_IS_COMPACT: bool;

    fn svd_compact_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
    ) -> Result<SvdFactorsDyn<R, D>, Self::Error>;

    fn svd_compact_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar;

    fn svd_compact_adjoint_dense<E, D>(
        dense: &mut E,
        parent: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar;

    fn svd_full_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
    ) -> Result<SvdFullFactorsDyn<R, D>, Self::Error>;

    fn svd_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFullFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar;

    fn svd_full_adjoint_dense<E, D>(
        dense: &mut E,
        parent: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFullFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar;

    fn spectrum_bond<V>(
        space: &BoundDynamicFusionMapSpace<R>,
        route: FactorRoute,
        spectrum: &[SectorSpectrum<V>],
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error>;

    fn rectangular_spectrum_factor<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum],
        row_dimensions: &BTreeMap<SectorId, usize>,
        col_dimensions: &BTreeMap<SectorId, usize>,
    ) -> Result<BoundDynFactor<R, D>, Self::Error>;

    // Eigendecomposition stages, publishing through each mode's own
    // builders (the checked dense `eig_full` also keeps its rank gate, D4 of
    // the one-path audit, #1798).
    fn eigh_full_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
        hermitian_tol: HermitianTol,
    ) -> Result<EighFullDyn<R, D>, Self::Error>;

    fn eigh_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
        hermitian_tol: HermitianTol,
    ) -> Result<EighFullDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar;

    fn eig_full_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
    ) -> Result<EigFullDyn<R, D>, Self::Error>;

    fn eig_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<EigFullDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar;
}

impl sealed::Sealed for MultiplicityFreeAdmissionMode {}
impl sealed::Sealed for CheckedGenericAdmissionMode {}

impl<R> FactorMode<R> for MultiplicityFreeAdmissionMode
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    type Error = OperationError;
    type RootError = OperationError;

    fn authority(
        space: &BoundDynamicFusionMapSpace<R>,
    ) -> impl FactorSpaceAuthority<R, Error = Self::Error, RootError = Self::RootError> + '_ {
        MfAuthority(space)
    }

    const SVD_FULL_DIAGONAL_IS_COMPACT: bool = true;

    fn svd_compact_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
    ) -> Result<SvdFactorsDyn<R, D>, Self::Error> {
        svd_compact_diagonal_factors_dyn(space, spectrum)
    }

    fn svd_compact_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        svd_compact_factors_dyn(dense, input)
    }

    fn svd_compact_adjoint_dense<E, D>(
        dense: &mut E,
        parent: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        svd_compact_adjoint_factors_dyn(dense, parent)
    }

    fn svd_full_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
    ) -> Result<SvdFullFactorsDyn<R, D>, Self::Error> {
        let _ = (space, spectrum);
        Err(OperationError::UnsupportedTensorContractScope {
            message: "multiplicity-free full SVD of a compact diagonal runs the compact route",
        })
    }

    fn svd_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFullFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        svd_full_factors_dyn(dense, input)
    }

    fn svd_full_adjoint_dense<E, D>(
        dense: &mut E,
        parent: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFullFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        svd_full_adjoint_factors_dyn(dense, parent)
    }

    fn spectrum_bond<V>(
        space: &BoundDynamicFusionMapSpace<R>,
        route: FactorRoute,
        spectrum: &[SectorSpectrum<V>],
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error> {
        let _ = route;
        diagonal_bond_bound_space_like(space, spectrum)
    }

    fn rectangular_spectrum_factor<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum],
        row_dimensions: &BTreeMap<SectorId, usize>,
        col_dimensions: &BTreeMap<SectorId, usize>,
    ) -> Result<BoundDynFactor<R, D>, Self::Error> {
        rectangular_diagonal_bond_tensor(space, spectrum, row_dimensions, col_dimensions)
    }

    fn qr_compact_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Qr<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        qr_compact_dyn(dense, input)
    }

    fn qr_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Qr<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        qr_full_dyn(dense, input)
    }

    fn lq_compact_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Lq<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        lq_compact_dyn(dense, input)
    }

    fn lq_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Lq<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        lq_full_dyn(dense, input)
    }

    fn eigh_full_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
        hermitian_tol: HermitianTol,
    ) -> Result<EighFullDyn<R, D>, Self::Error> {
        eigh_full_diagonal_dyn(space, spectrum, hermitian_tol)
    }

    fn eigh_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
        hermitian_tol: HermitianTol,
    ) -> Result<EighFullDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        eigh_full_dyn(dense, input, hermitian_tol)
    }

    fn eig_full_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
    ) -> Result<EigFullDyn<R, D>, Self::Error> {
        eig_full_diagonal_dyn(space, spectrum)
    }

    fn eig_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<EigFullDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        eig_full_dyn(dense, input)
    }
}

impl<R> FactorMode<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericFusion,
{
    type Error = CheckedGenericFactorPlanError<R::Error>;
    type RootError = CheckedGenericStructureError<R::Error>;

    fn authority(
        space: &BoundDynamicFusionMapSpace<R>,
    ) -> impl FactorSpaceAuthority<R, Error = Self::Error, RootError = Self::RootError> + '_ {
        CheckedAuthority(space.provider_arc())
    }

    const SVD_FULL_DIAGONAL_IS_COMPACT: bool = false;

    fn svd_compact_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
    ) -> Result<SvdFactorsDyn<R, D>, Self::Error> {
        svd_compact_diagonal_factors_dyn_checked_generic(space, spectrum)
    }

    fn svd_compact_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        svd_compact_factors_with_spectrum_dyn_checked_generic(dense, input)
    }

    fn svd_compact_adjoint_dense<E, D>(
        dense: &mut E,
        parent: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        let _ = (dense, parent);
        Err(OperationError::UnsupportedTensorContractScope {
            message: "checked Generic SVD does not read lazy adjoints",
        }
        .into())
    }

    fn svd_full_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
    ) -> Result<SvdFullFactorsDyn<R, D>, Self::Error> {
        svd_full_diagonal_factors_dyn_checked_generic(space, spectrum)
    }

    fn svd_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFullFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        svd_full_factors_dyn_checked_generic(dense, input)
    }

    fn svd_full_adjoint_dense<E, D>(
        dense: &mut E,
        parent: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<SvdFullFactorsDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        let _ = (dense, parent);
        Err(OperationError::UnsupportedTensorContractScope {
            message: "checked Generic SVD does not read lazy adjoints",
        }
        .into())
    }

    fn spectrum_bond<V>(
        space: &BoundDynamicFusionMapSpace<R>,
        route: FactorRoute,
        spectrum: &[SectorSpectrum<V>],
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error> {
        match route {
            FactorRoute::Diagonal => {
                diagonal_bond_bound_space_on_source_checked_generic(space, spectrum)
            }
            FactorRoute::Dense => diagonal_bond_bound_space_generic_checked(
                Arc::clone(space.provider_arc()),
                spectrum,
            ),
        }
    }

    fn rectangular_spectrum_factor<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum],
        row_dimensions: &BTreeMap<SectorId, usize>,
        col_dimensions: &BTreeMap<SectorId, usize>,
    ) -> Result<BoundDynFactor<R, D>, Self::Error> {
        rectangular_diagonal_bond_tensor_generic_checked(
            Arc::clone(space.provider_arc()),
            spectrum,
            row_dimensions,
            col_dimensions,
            &D::from_real,
        )
    }

    fn qr_compact_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Qr<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        qr_compact_dyn_checked_generic(dense, input)
    }

    fn qr_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Qr<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        qr_full_dyn_checked_generic(dense, input)
    }

    fn lq_compact_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Lq<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        lq_compact_dyn_checked_generic(dense, input)
    }

    fn lq_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Lq<BoundDynFactor<R, D>>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        lq_full_dyn_checked_generic(dense, input)
    }

    fn eigh_full_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
        hermitian_tol: HermitianTol,
    ) -> Result<EighFullDyn<R, D>, Self::Error> {
        eigh_full_diagonal_dyn_checked_generic(space, spectrum, hermitian_tol)
    }

    fn eigh_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
        hermitian_tol: HermitianTol,
    ) -> Result<EighFullDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        eigh_full_dyn_checked_generic(dense, input, hermitian_tol)
    }

    fn eig_full_diagonal<D: FactorScalar>(
        space: &BoundDynamicFusionMapSpace<R>,
        spectrum: &[SectorSpectrum<D>],
    ) -> Result<EigFullDyn<R, D>, Self::Error> {
        eig_full_diagonal_dyn_checked_generic(space, spectrum)
    }

    fn eig_full_dense<E, D>(
        dense: &mut E,
        input: &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<EigFullDyn<R, D>, Self::Error>
    where
        E: DenseExecutor + ?Sized,
        D: FactorScalar,
    {
        eig_full_dyn_checked_generic(dense, input)
    }
}

/// Multiplicity-free factor spaces, derived through the source space's cached
/// layout authority.
pub(super) struct MfAuthority<'a, R>(pub(super) &'a BoundDynamicFusionMapSpace<R>);

/// Checked Generic factor spaces, enumerated through the fallible provider.
pub(super) struct CheckedAuthority<'a, R>(pub(super) &'a Arc<R>);

impl<R> sealed::Sealed for MfAuthority<'_, R> {}
impl<R> sealed::Sealed for CheckedAuthority<'_, R> {}

impl<R> FactorSpaceAuthority<R> for MfAuthority<'_, R>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    type Error = OperationError;
    type Staged = BoundDynamicFusionMapSpace<R>;

    fn stage(&self, homspace: FusionTreeHomSpace) -> Result<Self::Staged, Self::Error> {
        #[cfg(test)]
        MF_FACTOR_SPACE_STAGES.set(MF_FACTOR_SPACE_STAGES.get() + 1);
        self.0.derive_from_final_homspace(homspace)
    }

    fn commit(
        &self,
        staged: Self::Staged,
        _pre_commit: impl FnOnce(&SectorStructure) -> Result<(), OperationError>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error> {
        Ok(staged)
    }

    fn coupled_dimensions(
        &self,
        side: &FusionProductSpace,
    ) -> Result<BTreeMap<SectorId, usize>, Self::Error> {
        Ok(side.coupled_sector_block_dimensions(self.0.provider())?)
    }

    fn isomorphic(&self, space: &BoundDynamicFusionMapSpace<R>) -> Result<bool, Self::Error> {
        space.codomain_isomorphic_to_domain()
    }

    type RootError = OperationError;

    fn output_space(
        &self,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::RootError> {
        self.0.derive_from_final_homspace(homspace)
    }
}

impl<R> FactorSpaceAuthority<R> for CheckedAuthority<'_, R>
where
    R: CheckedGenericFusion,
{
    type Error = CheckedGenericFactorPlanError<R::Error>;
    type Staged = (FusionTreeHomSpace, PreparedBlockStructure);

    fn stage(&self, homspace: FusionTreeHomSpace) -> Result<Self::Staged, Self::Error> {
        let prepared = homspace
            .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(
                self.0.as_ref(),
            )
            .map_err(CheckedGenericFactorPlanError::from)?;
        Ok((homspace, prepared))
    }

    fn commit(
        &self,
        (homspace, prepared): Self::Staged,
        pre_commit: impl FnOnce(&SectorStructure) -> Result<(), OperationError>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error> {
        pre_commit(prepared.sector_structure())?;
        BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
            Arc::clone(self.0),
            homspace,
            prepared,
        )
        .map_err(CheckedGenericFactorPlanError::from)
    }

    fn coupled_dimensions(
        &self,
        side: &FusionProductSpace,
    ) -> Result<BTreeMap<SectorId, usize>, Self::Error> {
        coupled_sector_block_dimensions_generic_checked(side, self.0.as_ref())
    }

    fn isomorphic(&self, space: &BoundDynamicFusionMapSpace<R>) -> Result<bool, Self::Error> {
        let homspace = space.space().homspace();
        Ok(self.coupled_dimensions(homspace.codomain())?
            == self.coupled_dimensions(homspace.domain())?)
    }

    type RootError = CheckedGenericStructureError<R::Error>;

    fn output_space(
        &self,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::RootError> {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(self.0),
            homspace,
        )
    }
}

/// `domain <- codomain` of `homspace`: the space of its inverse.
pub(super) fn inverse_homspace(homspace: &FusionTreeHomSpace) -> FusionTreeHomSpace {
    FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone())
}

/// `domain(divisor) <- domain(rhs)`: the space of `divisor \ rhs`.
pub(super) fn solve_homspace(
    divisor: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
) -> FusionTreeHomSpace {
    FusionTreeHomSpace::new(divisor.domain().clone(), rhs.domain().clone())
}

#[cfg(test)]
thread_local! {
    /// Calls of [`MfAuthority::stage`], one per factor space built. It counts
    /// calls only; cache hits are shown by structure identity
    /// (`Arc::ptr_eq`) across repeated calls.
    pub(crate) static MF_FACTOR_SPACE_STAGES: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
mod publication_order_tests {
    use super::*;

    #[test]
    fn checked_factor_pre_commit_error_precedes_late_style_and_publication() {
        const FILTER: &str = "factorize::authority::publication_order_tests::checked_factor_pre_commit_error_precedes_late_style_and_publication";
        if std::env::var_os("TENET_FACTOR_LATE_STYLE").is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FILTER])
                .env("TENET_FACTOR_LATE_STYLE", "1")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
                "{stdout} {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        tenet_core::reset_core_intern_tables();
        let rule = tenet_core::SU2FusionRule;
        let provider = Arc::new(tenet_core::InfallibleGeneric::new(&rule));
        let authority = CheckedAuthority(&provider);
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 2)], [(1, 3)]);
        let stage = authority.stage(hom.clone()).unwrap();
        let sentinel = OperationError::InvalidArgument {
            message: "placement sentinel",
        };
        let error = authority
            .commit(stage, |_| Err(sentinel.clone()))
            .unwrap_err();
        assert!(
            matches!(error, CheckedGenericFactorPlanError::Operation(ref actual) if actual == &sentinel)
        );
        let entries = || {
            tenet_core::structure_cache_info(tenet_core::StructureCacheKind::DegeneracyStructure)
                .entries()
        };
        assert_eq!(entries(), 0);
        let stage = authority.stage(hom).unwrap();
        let checked = std::cell::Cell::new(false);
        let error = authority
            .commit(stage, |_| {
                checked.set(true);
                Ok(())
            })
            .unwrap_err();
        assert!(checked.get());
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(OperationError::Core(
                CoreError::UnsupportedFusionStyle { .. }
            ))
        ));
        assert_eq!(entries(), 0);
    }
}
