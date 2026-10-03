use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompactFactorRoute {
    pub(super) source_region: usize,
    pub(super) left_region: Option<usize>,
    pub(super) right_region: Option<usize>,
    pub(super) sector: SectorId,
    pub(super) rank: usize,
}

/// Per-call routing from the source coupled-sector regions to the two factor
/// regions. Owned by the calling factorization and dropped with it.
///
/// Why not `Arc` the plan or its source half: nothing shares it beyond the
/// one call that builds it, so the wrappers were two heap allocations per
/// call with no owner to serve.
#[derive(Debug)]
pub(crate) struct CompactFactorPlan {
    pub(super) source_layout: ValidatedDynamicFusionLayout,
    pub(super) source_regions: Arc<[CoupledSectorRegion]>,
    pub(super) left_layout: ValidatedDynamicFusionLayout,
    pub(super) right_layout: ValidatedDynamicFusionLayout,
    pub(super) left_regions: Arc<[CoupledSectorRegion]>,
    pub(super) right_regions: Arc<[CoupledSectorRegion]>,
    pub(super) routes: Vec<CompactFactorRoute>,
}

#[doc(hidden)]
#[derive(Debug)]
pub enum CheckedGenericFactorPlanError<E> {
    Provider(E),
    Operation(OperationError),
}

impl<E> From<CheckedGenericStructureError<E>> for CheckedGenericFactorPlanError<E> {
    fn from(error: CheckedGenericStructureError<E>) -> Self {
        match error {
            CheckedGenericStructureError::Provider(error) => Self::Provider(error),
            CheckedGenericStructureError::Core(error) => {
                Self::Operation(OperationError::from_core_preserving_context(error))
            }
        }
    }
}

impl<E> From<OperationError> for CheckedGenericFactorPlanError<E> {
    fn from(error: OperationError) -> Self {
        Self::Operation(error)
    }
}

pub(crate) struct PreparedGenericCompactFactorPlan {
    pub(super) source_layout: ValidatedDynamicFusionLayout,
    pub(super) source_regions: Arc<[CoupledSectorRegion]>,
    pub(super) left: PreparedCheckedGenericDynamicSpace,
    pub(super) right: PreparedCheckedGenericDynamicSpace,
    pub(super) left_regions: Arc<[CoupledSectorRegion]>,
    pub(super) right_regions: Arc<[CoupledSectorRegion]>,
    pub(super) routes: Vec<CompactFactorRoute>,
}

pub(super) fn compact_factor_plan<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    build_compact_factor_plan(input, input.validated_layout())
}

pub(super) fn compact_factor_plan_generic<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: FusionRule,
{
    let checked = InfallibleGeneric::new(input.provider());
    match prepare_compact_factor_plan_generic_checked(input, &checked) {
        Ok(Some(prepared)) => finish_compact_factor_plan_generic(input, prepared),
        Ok(None) => Ok(None),
        Err(CheckedGenericFactorPlanError::Provider(never)) => match never {},
        Err(CheckedGenericFactorPlanError::Operation(error)) => Err(error),
    }
}

pub(super) fn prepare_compact_factor_plan_generic_checked<R, P>(
    input: &BoundDynamicFusionMapSpace<R>,
    provider: &P,
) -> Result<Option<PreparedGenericCompactFactorPlan>, CheckedGenericFactorPlanError<P::Error>>
where
    R: FusionRule,
    P: CheckedGenericFusion,
{
    let space = input.space();
    let Some(regions) = checked_sector_regions(space.structure(), space.nout())? else {
        return Ok(None);
    };
    let new_leg = compact_bond_leg(&regions);
    let left_hom = FusionTreeHomSpace::new(
        space.homspace().codomain().clone(),
        FusionProductSpace::new([new_leg.clone()]),
    );
    let right_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg]),
        space.homspace().domain().clone(),
    );
    let left = input.prepare_final_homspace_generic_checked(provider, left_hom)?;
    let right = input.prepare_final_homspace_generic_checked(provider, right_hom)?;
    let left_regions =
        checked_sector_regions(left.structure(), space.nout())?.ok_or_else(|| {
            OperationError::UnsupportedTensorContractScope {
                message: "compact left factor is not a coupled-sector matrix layout",
            }
        })?;
    let right_regions = checked_sector_regions(right.structure(), 1)?.ok_or_else(|| {
        OperationError::UnsupportedTensorContractScope {
            message: "compact right factor is not a coupled-sector matrix layout",
        }
    })?;
    if !source_factor_tree_extents_match(&regions, &left_regions, &right_regions) {
        return Ok(None);
    }
    let routes = compile_compact_factor_routes(&regions, &left_regions, &right_regions)?;
    Ok(Some(PreparedGenericCompactFactorPlan {
        source_layout: input.validated_layout(),
        source_regions: regions,
        left,
        right,
        left_regions,
        right_regions,
        routes,
    }))
}

pub(super) fn source_factor_tree_extents_match(
    source: &[CoupledSectorRegion],
    left: &[CoupledSectorRegion],
    right: &[CoupledSectorRegion],
) -> bool {
    let Ok(left_by_sector) = sector_region_index_map(left) else {
        return false;
    };
    let Ok(right_by_sector) = sector_region_index_map(right) else {
        return false;
    };
    source.iter().all(|source_region| {
        let sector = source_region.coupled();
        let Some(&left_index) = left_by_sector.get(&sector) else {
            return false;
        };
        let Some(&right_index) = right_by_sector.get(&sector) else {
            return false;
        };
        source_region.row_trees() == left[left_index].row_trees()
            && source_region.col_trees() == right[right_index].col_trees()
    })
}

pub(super) fn finish_compact_factor_plan_generic<R>(
    input: &BoundDynamicFusionMapSpace<R>,
    prepared: PreparedGenericCompactFactorPlan,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: FusionRule,
{
    #[cfg(test)]
    GENERIC_FACTOR_PLAN_FINISH_CALLS.with(|calls| calls.set(calls.get() + 1));
    let PreparedGenericCompactFactorPlan {
        source_layout,
        source_regions,
        left,
        right,
        left_regions,
        right_regions,
        routes,
    } = prepared;
    let left = input.commit_final_homspace_generic_checked(left)?;
    let right = input.commit_final_homspace_generic_checked(right)?;
    Ok(Some(CompactFactorPlan {
        source_layout,
        source_regions,
        left_layout: left.validated_layout(),
        right_layout: right.validated_layout(),
        left_regions,
        right_regions,
        routes,
    }))
}

#[cfg(test)]
thread_local! {
    pub(super) static GENERIC_FACTOR_PLAN_FINISH_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_generic_factor_plan_finish_calls() {
    GENERIC_FACTOR_PLAN_FINISH_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
pub(crate) fn generic_factor_plan_finish_calls() -> usize {
    GENERIC_FACTOR_PLAN_FINISH_CALLS.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn prepare_compact_factor_plan_generic_checked_for_test<R, P>(
    input: &BoundDynamicFusionMapSpace<R>,
    provider: &P,
) -> Result<Option<PreparedGenericCompactFactorPlan>, CheckedGenericFactorPlanError<P::Error>>
where
    R: FusionRule,
    P: CheckedGenericFusion,
{
    prepare_compact_factor_plan_generic_checked(input, provider)
}

#[cfg(test)]
pub(crate) fn finish_compact_factor_plan_generic_for_test<R>(
    input: &BoundDynamicFusionMapSpace<R>,
    prepared: PreparedGenericCompactFactorPlan,
) -> Result<bool, OperationError>
where
    R: FusionRule,
{
    finish_compact_factor_plan_generic(input, prepared).map(|plan| plan.is_some())
}

pub(super) fn build_compact_factor_plan<R>(
    input: &BoundDynamicFusionMapSpace<R>,
    source_layout: ValidatedDynamicFusionLayout,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let space = input.space();
    let Some(regions) = checked_sector_regions(space.structure(), space.nout())? else {
        return Ok(None);
    };
    let bond = compact_bond_leg(&regions);
    let u_space =
        build_bound_factor_space(input, space.homspace(), bond.clone(), FactorSide::Left)?;
    let vh_space = build_bound_factor_space(input, space.homspace(), bond, FactorSide::Right)?;
    let left_regions = checked_sector_regions(u_space.space().structure(), u_space.space().nout())?
        .ok_or_else(|| OperationError::UnsupportedTensorContractScope {
            message: "compact left factor is not a coupled-sector matrix layout",
        })?;
    let right_regions =
        checked_sector_regions(vh_space.space().structure(), vh_space.space().nout())?.ok_or_else(
            || OperationError::UnsupportedTensorContractScope {
                message: "compact right factor is not a coupled-sector matrix layout",
            },
        )?;
    let routes = compile_compact_factor_routes(&regions, &left_regions, &right_regions)?;
    // The direct path publishes each dense factor region positionally, so it
    // is sound only when the fresh factor lists the source's trees in the
    // source's order. Why not scatter here: a reordered tiling takes the
    // packed path, whose `PlacementIndex` scatter already maps by tree
    // identity, so one authority owns non-canonical publication.
    if !compact_factor_routes_preserve_tree_order(&routes, &regions, &left_regions, &right_regions)
    {
        return Ok(None);
    }
    Ok(Some(CompactFactorPlan {
        source_layout,
        source_regions: regions,
        left_layout: u_space.validated_layout(),
        right_layout: vh_space.validated_layout(),
        left_regions,
        right_regions,
        routes,
    }))
}

pub(super) fn compact_factor_routes_preserve_tree_order(
    routes: &[CompactFactorRoute],
    source: &[CoupledSectorRegion],
    left: &[CoupledSectorRegion],
    right: &[CoupledSectorRegion],
) -> bool {
    routes.iter().all(|route| {
        let region = &source[route.source_region];
        route
            .left_region
            .is_none_or(|index| left[index].row_trees() == region.row_trees())
            && route
                .right_region
                .is_none_or(|index| right[index].col_trees() == region.col_trees())
    })
}

/// The bond leg `W` shared by both compact factors: one sector per source
/// region with degeneracy `min(rows, cols)`.
pub(super) fn compact_bond_leg(regions: &[CoupledSectorRegion]) -> SectorLeg {
    SectorLeg::new(
        regions
            .iter()
            .map(|region| (region_sector(region), region.rows().min(region.cols()))),
        false,
    )
}

pub(super) fn compile_compact_factor_routes(
    source_regions: &[CoupledSectorRegion],
    left_regions: &[CoupledSectorRegion],
    right_regions: &[CoupledSectorRegion],
) -> Result<Vec<CompactFactorRoute>, OperationError> {
    let left_by_sector = SectorRegionIndex::new(left_regions)?;
    let right_by_sector = SectorRegionIndex::new(right_regions)?;
    let mut routes = Vec::with_capacity(source_regions.len());
    // Why not per-region `used` tables: the sector -> region index is
    // injective and each nonzero route consumes a distinct index per side, so
    // "an unused nonzero region exists" is exactly "used < nonzero regions".
    let mut used_left = 0usize;
    let mut used_right = 0usize;
    for (source_region, region) in source_regions.iter().enumerate() {
        let sector = region_sector(region);
        let rank = region.rows().min(region.cols());
        let (left_region, right_region) = if rank == 0 {
            (None, None)
        } else {
            let left_region = sector_region_index_of(&left_by_sector, sector, "left")?;
            let right_region = sector_region_index_of(&right_by_sector, sector, "right")?;
            validate_factor_region(&left_regions[left_region], region.rows(), rank, "left")?;
            validate_factor_region(&right_regions[right_region], rank, region.cols(), "right")?;
            used_left += 1;
            used_right += 1;
            (Some(left_region), Some(right_region))
        };
        routes.push(CompactFactorRoute {
            source_region,
            left_region,
            right_region,
            sector,
            rank,
        });
    }
    validate_no_unused_factor_regions(left_regions, used_left, "left")?;
    validate_no_unused_factor_regions(right_regions, used_right, "right")?;
    Ok(routes)
}

#[cfg(test)]
pub(crate) fn compact_factor_plan_for_test<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    compact_factor_plan(input)
}

#[cfg(test)]
#[expect(
    clippy::type_complexity,
    reason = "the test seam returns source, left, and right region tables in documented order"
)]
pub(crate) fn compact_factor_plan_regions_for_test(
    plan: &CompactFactorPlan,
) -> (
    Arc<[CoupledSectorRegion]>,
    Arc<[CoupledSectorRegion]>,
    Arc<[CoupledSectorRegion]>,
) {
    (
        Arc::clone(&plan.source_regions),
        Arc::clone(&plan.left_regions),
        Arc::clone(&plan.right_regions),
    )
}

#[cfg(test)]
pub(crate) fn validate_compact_factor_routes_for_test(
    source: &[CoupledSectorRegion],
    u: &[CoupledSectorRegion],
    vh: &[CoupledSectorRegion],
) -> Result<Vec<CompactFactorRoute>, OperationError> {
    compile_compact_factor_routes(source, u, vh)
}

#[cfg(test)]
pub(crate) fn compact_factor_plan_routes_for_test(
    plan: &CompactFactorPlan,
) -> &[CompactFactorRoute] {
    &plan.routes
}

#[cfg(test)]
impl CompactFactorRoute {
    pub(crate) fn factor_regions_for_test(&self) -> (usize, Option<usize>, Option<usize>) {
        (self.source_region, self.left_region, self.right_region)
    }
}

pub(super) fn region_sector(region: &CoupledSectorRegion) -> SectorId {
    region.coupled()
}

pub(super) fn sector_region_index_map(
    regions: &[CoupledSectorRegion],
) -> Result<FxHashMap<SectorId, usize>, OperationError> {
    let mut by_sector = FxHashMap::with_capacity_and_hasher(regions.len(), Default::default());
    for (index, region) in regions.iter().enumerate() {
        if by_sector.insert(region_sector(region), index).is_some() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "coupled-sector region description contains a duplicate sector",
            });
        }
    }
    Ok(by_sector)
}

/// Proves that a compact diagonal spectrum exactly covers aligned square
/// source regions and returns the sole sector lookup used by its caller.
/// Region labels are unique because callers pass `checked_sector_regions` output.
pub(super) fn aligned_diagonal_spectrum_by_sector<'a, D>(
    regions: &[CoupledSectorRegion],
    spectrum: &'a [SectorSpectrum<D>],
) -> Option<FxHashMap<SectorId, &'a SectorSpectrum<D>>> {
    if spectrum.len() != regions.len() {
        return None;
    }
    let mut by_sector = FxHashMap::with_capacity_and_hasher(spectrum.len(), Default::default());
    for entry in spectrum {
        if by_sector.insert(entry.sector, entry).is_some() {
            return None;
        }
    }
    for region in regions {
        let entry = by_sector.get(&region.coupled())?;
        if !region.has_aligned_diagonal()
            || region.rows() != region.cols()
            || entry.values.len() != region.rows()
        {
            return None;
        }
    }
    Some(by_sector)
}

/// Sector -> region index lookup over one factor's region table.
///
/// Canonical factor layouts list their regions strictly sorted by coupled
/// sector (a structural invariant of the derived layout, checked here in
/// O(G)), so a binary search serves without building a map. Expert region
/// tables that are not sorted keep the hash-map path, which also reports the
/// duplicate-sector error.
pub(super) enum SectorRegionIndex<'a> {
    Sorted(&'a [CoupledSectorRegion]),
    Map(FxHashMap<SectorId, usize>),
}

impl<'a> SectorRegionIndex<'a> {
    pub(super) fn new(regions: &'a [CoupledSectorRegion]) -> Result<Self, OperationError> {
        if regions
            .windows(2)
            .all(|pair| region_sector(&pair[0]) < region_sector(&pair[1]))
        {
            Ok(Self::Sorted(regions))
        } else {
            sector_region_index_map(regions).map(Self::Map)
        }
    }

    pub(super) fn get(&self, sector: SectorId) -> Option<usize> {
        match self {
            Self::Sorted(regions) => regions.binary_search_by_key(&sector, region_sector).ok(),
            Self::Map(map) => map.get(&sector).copied(),
        }
    }
}

pub(super) fn sector_region_index_of(
    regions: &SectorRegionIndex<'_>,
    sector: SectorId,
    side: &'static str,
) -> Result<usize, OperationError> {
    regions
        .get(sector)
        .ok_or_else(|| OperationError::UnsupportedTensorContractScope {
            message: match side {
                "left" => "compact left factor is missing a nonzero-rank sector",
                _ => "compact right factor is missing a nonzero-rank sector",
            },
        })
}

pub(super) fn validate_factor_region(
    region: &CoupledSectorRegion,
    rows: usize,
    cols: usize,
    side: &'static str,
) -> Result<(), OperationError> {
    if region.rows() != rows || region.cols() != cols {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: match side {
                "left" => "compact left sector region has an unexpected shape",
                _ => "compact right sector region has an unexpected shape",
            },
        });
    }
    Ok(())
}

pub(super) fn validate_no_unused_factor_regions(
    regions: &[CoupledSectorRegion],
    used: usize,
    side: &'static str,
) -> Result<(), OperationError> {
    let nonzero = regions
        .iter()
        .filter(|region| region.rows() != 0 && region.cols() != 0)
        .count();
    if used < nonzero {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: match side {
                "left" => "compact left factor contains an unused nonzero sector",
                _ => "compact right factor contains an unused nonzero sector",
            },
        });
    }
    Ok(())
}
