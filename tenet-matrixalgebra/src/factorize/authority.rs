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
        MF_FACTOR_SPACE_DERIVES.set(MF_FACTOR_SPACE_DERIVES.get() + 1);
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
}

#[cfg(test)]
thread_local! {
    pub(crate) static MF_FACTOR_SPACE_DERIVES: Cell<usize> = const { Cell::new(0) };
}
