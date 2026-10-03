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
/// - [`CheckedAuthority`] enumerates in `stage` and publishes in `commit`, so a
///   malformed placement fails before any bound space exists;
/// - [`MfAuthority`]'s `stage` is the cached `derive_from_final_homspace`,
///   which has already published the layout, so its `commit` skips
///   `pre_commit`; the scatter validates placements lazily and the canonical
///   fast path does no placement hashing.
///
/// Why not one `factor_space(source, hom)` call (target design §2.2):
/// publishing before validation would reorder the checked error precedence,
/// which is an observable contract.
pub(super) trait FactorSpaceAuthority<R>: sealed::Sealed {
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

/// Checked inverse and solve admission: whether `space`'s codomain and domain
/// have equal coupled-sector dimensions under its provider, queried codomain
/// first.
#[doc(hidden)]
pub fn factor_isomorphic_checked_generic<R: CheckedGenericFusion>(
    space: &BoundDynamicFusionMapSpace<R>,
) -> Result<bool, CheckedGenericFactorPlanError<R::Error>> {
    CheckedAuthority(space.provider_arc()).isomorphic(space)
}

/// A checked inverse, pseudo-inverse or solve output space over `provider`,
/// built by the checked factor-space authority.
#[doc(hidden)]
pub fn factor_output_space_checked_generic<R: CheckedGenericFusion>(
    provider: &Arc<R>,
    homspace: FusionTreeHomSpace,
) -> Result<BoundDynamicFusionMapSpace<R>, CheckedGenericStructureError<R::Error>> {
    CheckedAuthority(provider).output_space(homspace)
}

#[cfg(test)]
thread_local! {
    /// Calls of [`MfAuthority::stage`], one per factor space built. It counts
    /// calls only; cache hits are shown by structure identity
    /// (`Arc::ptr_eq`) across repeated calls.
    pub(crate) static MF_FACTOR_SPACE_STAGES: Cell<usize> = const { Cell::new(0) };
}
