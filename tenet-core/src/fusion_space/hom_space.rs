#[derive(Debug, Eq, PartialEq, Hash)]
struct FusionTreeHomSpaceContent {
    codomain: FusionProductSpace,
    domain: FusionProductSpace,
}

impl FusionTreeHomSpaceContent {
    fn charged_retained_bytes(&self) -> usize {
        fn product_space_bytes(space: &FusionProductSpace) -> usize {
            std::mem::size_of::<FusionProductSpace>()
                .saturating_add(spilled_smallvec_heap_bytes(&space.legs))
                .saturating_add(
                    space
                        .legs
                        .iter()
                        .fold(0usize, |bytes, leg| {
                            bytes.saturating_add(leg.charged_retained_bytes())
                        }),
                )
        }

        product_space_bytes(&self.codomain)
            .saturating_add(product_space_bytes(&self.domain))
            .saturating_add(2 * std::mem::size_of::<usize>())
    }
}

pub struct FusionTreeHomSpace {
    content: Arc<FusionTreeHomSpaceContent>,
    id: OnceLock<HomSpaceId>,
}

/// Describes the geometric placement of one canonical unit leg.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnitLegInsertion {
    Left { position: usize, dual: bool },
    Right { position: usize, dual: bool },
}

impl UnitLegInsertion {
    fn position(self) -> usize {
        match self {
            Self::Left { position, .. } | Self::Right { position, .. } => position,
        }
    }

    fn dual(self) -> bool {
        match self {
            Self::Left { dual, .. } | Self::Right { dual, .. } => dual,
        }
    }
}

fn insert_product_space_leg(
    space: &FusionProductSpace,
    position: usize,
    leg: SectorLeg,
) -> FusionProductSpace {
    let mut legs = space.legs.clone();
    legs.insert(position, leg);
    FusionProductSpace { legs }
}

fn remove_product_space_leg(space: &FusionProductSpace, position: usize) -> FusionProductSpace {
    let mut legs = space.legs.clone();
    legs.remove(position);
    FusionProductSpace { legs }
}

impl std::fmt::Debug for FusionTreeHomSpace {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FusionTreeHomSpace")
            .field("codomain", &self.content.codomain)
            .field("domain", &self.content.domain)
            .field("id", &self.id)
            .finish()
    }
}

impl Clone for FusionTreeHomSpace {
    fn clone(&self) -> Self {
        Self {
            content: Arc::clone(&self.content),
            id: self.id.clone(),
        }
    }
}

impl PartialEq for FusionTreeHomSpace {
    fn eq(&self, other: &Self) -> bool {
        self.content.codomain == other.content.codomain && self.content.domain == other.content.domain
    }
}

impl Eq for FusionTreeHomSpace {}

impl std::hash::Hash for FusionTreeHomSpace {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.content.codomain.hash(state);
        self.content.domain.hash(state);
    }
}

impl FusionTreeHomSpace {
    /// Conservative retained bytes for this HomSpace and its Arc-backed
    /// provider-neutral leg metadata.
    #[doc(hidden)]
    pub fn charged_retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(self.content.charged_retained_bytes())
    }

    /// Builds a fusion-tree hom space from codomain and domain product spaces.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, Z2Irrep};
    ///
    /// let hom = FusionTreeHomSpace::new(
    ///     FusionProductSpace::new([SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false)]),
    ///     FusionProductSpace::new([SectorLeg::new([(Z2Irrep::EVEN, 1)], false)]),
    /// );
    /// assert_eq!(hom.codomain().len(), 1);
    /// assert_eq!(hom.domain().len(), 1);
    /// ```
    pub fn new(codomain: FusionProductSpace, domain: FusionProductSpace) -> Self {
        Self {
            content: Arc::new(FusionTreeHomSpaceContent { codomain, domain }),
            id: OnceLock::new(),
        }
    }

    /// Builds a hom space when each external leg has exactly one sector,
    /// from `(sector, degeneracy)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{FusionTreeHomSpace, Z2Irrep};
    ///
    /// let hom = FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::ODD, 1)]);
    /// assert_eq!(hom.codomain().len(), 1);
    /// assert_eq!(hom.domain().len(), 1);
    /// ```
    pub fn from_sectors<Codomain, Domain, CodomainSector, DomainSector>(
        codomain: Codomain,
        domain: Domain,
    ) -> Self
    where
        Codomain: IntoIterator<Item = (CodomainSector, usize)>,
        Domain: IntoIterator<Item = (DomainSector, usize)>,
        CodomainSector: Into<SectorId>,
        DomainSector: Into<SectorId>,
    {
        Self::new(
            FusionProductSpace::new(
                codomain
                    .into_iter()
                    .map(|(sector, degeneracy)| SectorLeg::new([(sector, degeneracy)], false)),
            ),
            FusionProductSpace::new(
                domain
                    .into_iter()
                    .map(|(sector, degeneracy)| SectorLeg::new([(sector, degeneracy)], false)),
            ),
        )
    }

    /// Builds a hom space from raw `(sector id, degeneracy)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::FusionTreeHomSpace;
    ///
    /// let hom = FusionTreeHomSpace::from_sector_ids([(0, 1), (1, 1)], [(1, 1)]);
    /// assert_eq!(hom.codomain().len(), 2);
    /// assert_eq!(hom.domain().len(), 1);
    /// ```
    pub fn from_sector_ids<Codomain, Domain>(codomain: Codomain, domain: Domain) -> Self
    where
        Codomain: IntoIterator<Item = (usize, usize)>,
        Domain: IntoIterator<Item = (usize, usize)>,
    {
        Self::from_sectors(
            codomain
                .into_iter()
                .map(|(sector, degeneracy)| (SectorId::new(sector), degeneracy)),
            domain
                .into_iter()
                .map(|(sector, degeneracy)| (SectorId::new(sector), degeneracy)),
        )
    }

    #[inline]
    pub fn codomain(&self) -> &FusionProductSpace {
        &self.content.codomain
    }

    #[inline]
    pub fn domain(&self) -> &FusionProductSpace {
        &self.content.domain
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.content.codomain.len() + self.content.domain.len()
    }

    /// Inserts a canonical unit leg at a zero-based position using TensorKit's
    /// left-before seam convention.
    pub fn insert_left_unit<R>(
        &self,
        rule: &R,
        position: usize,
        dual: bool,
    ) -> Result<Self, CoreError>
    where
        R: CanonicalUnitFusionRule,
    {
        if position > self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let unit = SectorLeg::new([(rule.vacuum(), 1)], dual);
        if position < self.codomain().len() {
            Ok(Self::new(
                insert_product_space_leg(self.codomain(), position, unit),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                insert_product_space_leg(self.domain(), position - self.codomain().len(), unit),
            ))
        }
    }

    /// Inserts a canonical unit leg at a zero-based position using TensorKit's
    /// right-after seam convention.
    pub fn insert_right_unit<R>(
        &self,
        rule: &R,
        position: usize,
        dual: bool,
    ) -> Result<Self, CoreError>
    where
        R: CanonicalUnitFusionRule,
    {
        if position > self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let unit = SectorLeg::new([(rule.vacuum(), 1)], dual);
        if position <= self.codomain().len() {
            Ok(Self::new(
                insert_product_space_leg(self.codomain(), position, unit),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                insert_product_space_leg(self.domain(), position - self.codomain().len(), unit),
            ))
        }
    }

    /// Removes one canonical unit leg in flat external-axis order, matching
    /// TensorKit's unit-leg geometry.
    pub fn remove_unit<R>(&self, rule: &R, axis: usize) -> Result<Self, CoreError>
    where
        R: CanonicalUnitFusionRule,
    {
        if axis >= self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let leg = if axis < self.codomain().len() {
            &self.codomain().legs()[axis]
        } else {
            &self.domain().legs()[axis - self.codomain().len()]
        };
        if leg.sectors() != [rule.vacuum()] || leg.degeneracy(rule.vacuum()) != Some(1) {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        if axis < self.codomain().len() {
            Ok(Self::new(
                remove_product_space_leg(self.codomain(), axis),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                remove_product_space_leg(self.domain(), axis - self.codomain().len()),
            ))
        }
    }

    /// Checked Generic counterpart of [`Self::insert_left_unit`].
    #[doc(hidden)]
    pub fn insert_left_unit_checked<R>(
        &self,
        rule: &R,
        position: usize,
        dual: bool,
    ) -> Result<Self, CoreError>
    where
        R: CheckedCanonicalUnitFusionRule,
    {
        if position > self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let unit = SectorLeg::new([(rule.vacuum(), 1)], dual);
        if position < self.codomain().len() {
            Ok(Self::new(
                insert_product_space_leg(self.codomain(), position, unit),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                insert_product_space_leg(self.domain(), position - self.codomain().len(), unit),
            ))
        }
    }

    /// Checked Generic counterpart of [`Self::insert_right_unit`].
    #[doc(hidden)]
    pub fn insert_right_unit_checked<R>(
        &self,
        rule: &R,
        position: usize,
        dual: bool,
    ) -> Result<Self, CoreError>
    where
        R: CheckedCanonicalUnitFusionRule,
    {
        if position > self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let unit = SectorLeg::new([(rule.vacuum(), 1)], dual);
        if position <= self.codomain().len() {
            Ok(Self::new(
                insert_product_space_leg(self.codomain(), position, unit),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                insert_product_space_leg(self.domain(), position - self.codomain().len(), unit),
            ))
        }
    }

    /// Checked Generic counterpart of [`Self::remove_unit`].
    #[doc(hidden)]
    pub fn remove_unit_checked<R>(&self, rule: &R, axis: usize) -> Result<Self, CoreError>
    where
        R: CheckedCanonicalUnitFusionRule,
    {
        if axis >= self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let vacuum = rule.vacuum();
        let leg = if axis < self.codomain().len() {
            &self.codomain().legs()[axis]
        } else {
            &self.domain().legs()[axis - self.codomain().len()]
        };
        if leg.sectors() != [vacuum] || leg.degeneracy(vacuum) != Some(1) {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        if axis < self.codomain().len() {
            Ok(Self::new(
                remove_product_space_leg(self.codomain(), axis),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                remove_product_space_leg(self.domain(), axis - self.codomain().len()),
            ))
        }
    }

    /// Lazily assigned collision-safe process-local semantic identity.
    /// Hashing is O(1), and equal spaces compare equal across intern eviction.
    #[inline]
    pub fn id(&self) -> HomSpaceId {
        self.id
            .get_or_init(|| intern_hom_space(&self.content.codomain, &self.content.domain))
            .clone()
    }

    /// Returns the already-published semantic identity without initializing it.
    ///
    /// Why not call [`Self::id`]: fallible metadata builders must validate
    /// algebra before publishing process-local identity state.
    #[doc(hidden)]
    #[inline]
    pub fn existing_id(&self) -> Option<HomSpaceId> {
        self.id.get().cloned()
    }

    pub fn select<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        let descriptor = self.select_descriptor(codomain_axes, domain_axes)?;
        Ok(descriptor.materialize(rule))
    }

    /// Checked sibling of [`Self::select`] for finite or encoded fusion
    /// algebras whose dual operation may not be representable.
    pub fn try_select_checked<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        let descriptor = self.select_descriptor(codomain_axes, domain_axes)?;
        descriptor.try_materialize(rule).map_err(Into::into)
    }

    pub fn permute<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        let mut axes = SmallVec::<[usize; 8]>::new();
        axes.extend_from_slice(codomain_axes);
        axes.extend_from_slice(domain_axes);
        validate_permutation_inline(&axes, self.rank())?;
        self.select(rule, codomain_axes, domain_axes)
    }

    /// Checked sibling of [`Self::permute`] for finite or encoded fusion
    /// algebras whose dual operation may not be representable.
    pub fn try_permute_checked<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        let mut axes = SmallVec::<[usize; 8]>::new();
        axes.extend_from_slice(codomain_axes);
        axes.extend_from_slice(domain_axes);
        validate_permutation_inline(&axes, self.rank())?;
        let descriptor = self.select_descriptor(codomain_axes, domain_axes)?;
        descriptor.try_materialize(rule).map_err(Into::into)
    }

    /// Checked Generic sibling of [`Self::permute`] that uses the provider's
    /// fallible dual operation without requiring [`FusionRule`].
    #[doc(hidden)]
    pub fn try_permute_generic_checked<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let mut axes = SmallVec::<[usize; 8]>::new();
        axes.extend_from_slice(codomain_axes);
        axes.extend_from_slice(domain_axes);
        validate_permutation_inline(&axes, self.rank())?;
        let descriptor = self.select_descriptor(codomain_axes, domain_axes)?;
        descriptor.try_materialize_generic(rule)
    }

    fn select_descriptor<'a>(
        &'a self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<HomSpaceDescriptor<'a>, CoreError> {
        validate_axis_selection(codomain_axes, domain_axes, self.rank())?;
        Ok(HomSpaceDescriptor::new(
            codomain_axes
                .iter()
                .map(|&axis| self.external_axis_leg_view(axis)),
            domain_axes
                .iter()
                .map(|&axis| self.external_axis_leg_view(axis).toggled()),
        ))
    }

    pub fn compose<R>(rule: &R, lhs: &Self, rhs: &Self) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        if lhs.domain().len() != rhs.codomain().len() {
            return Err(CoreError::DimensionMismatch {
                expected: lhs.domain().len(),
                actual: rhs.codomain().len(),
            });
        }
        let lhs_codomain_rank = lhs.codomain().len();
        for (index, (lhs_domain, rhs_codomain)) in lhs
            .domain()
            .legs()
            .iter()
            .zip(rhs.codomain().legs())
            .enumerate()
        {
            validate_composed_leg(lhs_domain, rhs_codomain, (lhs_codomain_rank + index, index))?;
        }
        let descriptor = HomSpaceDescriptor::new(
            lhs.codomain()
                .legs()
                .iter()
                .map(OrientedLegView::borrowed),
            rhs.domain().legs().iter().map(OrientedLegView::borrowed),
        );
        Ok(descriptor.materialize(rule))
    }

    /// Structural lowering of a general axis contraction, the homspace-level
    /// analog of TensorOperations' `tensorcontract!`: reorders both operands
    /// to `(open, contracted)` x `(contracted, open)`, composes them (see
    /// [`Self::compose`]), and applies the requested output axis order.
    pub fn tensorcontract_homspace<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        OrientedFusionTreeHomSpace::tensorcontract_homspace(
            rule,
            OrientedFusionTreeHomSpace::new(lhs, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs, FusionTreePairOrientation::Direct),
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )
    }

    /// Checked sibling of [`Self::tensorcontract_homspace`] for finite or
    /// encoded fusion algebras whose dual operation may not be representable.
    #[allow(clippy::too_many_arguments)]
    pub fn try_tensorcontract_homspace_checked<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        OrientedFusionTreeHomSpace::try_tensorcontract_homspace_checked(
            rule,
            OrientedFusionTreeHomSpace::new(lhs, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs, FusionTreePairOrientation::Direct),
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )
    }

    /// Whether [`Self::tensorcontract_homspace`] equals `expected`; see
    /// [`OrientedFusionTreeHomSpace::tensorcontract_homspace_matches`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_homspace_matches<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
        expected: &Self,
    ) -> Result<bool, CoreError>
    where
        R: FusionRule,
    {
        OrientedFusionTreeHomSpace::tensorcontract_homspace_matches(
            rule,
            OrientedFusionTreeHomSpace::new(lhs, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs, FusionTreePairOrientation::Direct),
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
            expected,
        )
    }

    /// Checked sibling of [`Self::tensorcontract_homspace_matches`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_tensorcontract_homspace_matches_checked<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
        expected: &Self,
    ) -> Result<bool, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        OrientedFusionTreeHomSpace::try_tensorcontract_homspace_matches_checked(
            rule,
            OrientedFusionTreeHomSpace::new(lhs, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs, FusionTreePairOrientation::Direct),
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
            expected,
        )
    }

    /// Checked Generic sibling of [`Self::tensorcontract_homspace`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_tensorcontract_homspace_generic_checked<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
    ) -> Result<Self, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        OrientedFusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
            rule,
            OrientedFusionTreeHomSpace::new(lhs, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs, FusionTreePairOrientation::Direct),
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )
    }

    /// The cached fusion-tree block keys, shared in O(1) (`Arc::clone`): the
    /// layout already holds them as `Arc<[_]>`, so there is no need to deep-clone
    /// each key (two `FusionTreeKey`s, four `SectorVec`s each) into a fresh `Vec`
    /// on every call. Returns `Arc<[_]>`, which derefs to `[FusionTreePairKey]`,
    /// so iterate / index / `len` callers are unchanged; by-value consumers can
    /// `.to_vec()`. TensorKit's `fusiontrees(W)` likewise returns the cached
    /// index set by reference. See #53.
    pub fn fusion_tree_keys<R>(&self, rule: &R) -> Arc<[FusionTreePairKey]>
    where
        R: MultiplicityFreeFusionRule,
    {
        Arc::clone(&self.cached_fusion_tree_layout(rule).keys)
    }

    /// Stages multiplicity-free layout metadata without publishing it.
    #[doc(hidden)]
    pub fn prepare_fusion_tree_layout<R>(&self, rule: &R) -> PreparedFusionTreeLayout
    where
        R: MultiplicityFreeFusionRule,
    {
        self.prepare_fusion_tree_layout_with(rule, || {
            Ok::<_, std::convert::Infallible>(self.fusion_tree_layout_data_uncached(rule))
        })
        .unwrap_or_else(|error| match error {})
    }

    /// Stages checked layout metadata for an external multiplicity-free
    /// provider without issuing a layout ID or changing cache accounting.
    ///
    /// Every HomSpace leg sector is validated through [`CheckedFusionAlgebra::try_dual_sector`]
    /// before any cache lookup, because a rigid multiplicity-free provider
    /// requires a representable dual for every leg it admits. Enumeration then
    /// reuses the same [`Self::prepare_fusion_tree_layout_with`] staging and
    /// bounded layout cache as the built-in paths.
    #[doc(hidden)]
    pub fn prepare_fusion_tree_layout_checked<R>(
        &self,
        rule: &R,
    ) -> Result<PreparedFusionTreeLayout, FusionAlgebraError>
    where
        R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
    {
        for leg in self.codomain().legs().iter().chain(self.domain().legs()) {
            for &sector in leg.sectors() {
                rule.try_dual_sector(sector)?;
            }
        }
        self.prepare_fusion_tree_layout_with(rule, || {
            self.try_fusion_tree_layout_data_uncached_checked(rule)
        })
    }

    fn prepare_fusion_tree_layout_with<R, E, F>(
        &self,
        rule: &R,
        build: F,
    ) -> Result<PreparedFusionTreeLayout, E>
    where
        R: MultiplicityFreeFusionRule,
        F: FnOnce() -> Result<FusionTreeHomSpaceLayoutData, E>,
    {
        let key = FusionTreeHomSpaceCacheKey::new(rule, self);
        let cache = fusion_tree_layout_cache();
        let read = cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(layout) = read.lookup(&key) {
            return Ok(PreparedFusionTreeLayout {
                state: PreparedFusionTreeLayoutState::Cached { key, layout },
            });
        }
        drop(read);

        let data = build()?;
        Ok(PreparedFusionTreeLayout {
            state: PreparedFusionTreeLayoutState::Cold {
                key,
                data,
            },
        })
    }

    pub fn try_for_each_fusion_tree_key<R, F, E>(&self, rule: &R, mut f: F) -> Result<(), E>
    where
        R: MultiplicityFreeFusionRule,
        F: FnMut(&FusionTreePairKey) -> Result<(), E>,
    {
        let layout = self.cached_fusion_tree_layout(rule);
        for key in layout.keys.iter() {
            f(key)?;
        }
        Ok(())
    }

    pub fn try_with_fusion_tree_keys<R, F, T, E>(&self, rule: &R, f: F) -> Result<T, E>
    where
        R: MultiplicityFreeFusionRule,
        F: FnOnce(&[FusionTreePairKey]) -> Result<T, E>,
    {
        let layout = self.cached_fusion_tree_layout(rule);
        f(layout.keys.as_ref())
    }

    fn cached_fusion_tree_layout<R>(&self, rule: &R) -> Arc<FusionTreeHomSpaceLayout>
    where
        R: MultiplicityFreeFusionRule,
    {
        self.prepare_fusion_tree_layout(rule).commit_layout()
    }

    pub fn coupled_subblock_structure<R, Shapes>(
        &self,
        rule: &R,
        nout: usize,
        shapes: Shapes,
    ) -> Result<Arc<BlockStructure>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let layout = self.cached_fusion_tree_layout(rule);
        let rank = self.codomain().len() + self.domain().len();
        let shapes = shapes
            .into_iter()
            .map(|shape| shape.into().into_iter().collect::<DimVec>())
            .collect::<Vec<_>>();
        if layout.keys.len() != shapes.len() {
            return Err(CoreError::BlockCountMismatch {
                expected: layout.keys.len(),
                actual: shapes.len(),
            });
        }
        self.validate_degeneracy_shapes(layout.keys.as_ref(), &shapes)?;

        let cache_key = CoupledBlockStructureCacheKey {
            layout: layout.id,
            nout,
            rank,
            shapes: Arc::<[DimVec]>::from(shapes),
        };
        let cache = coupled_block_structure_cache();
        // Read-lock fast path uses `peek` (does not bump recency; `get` needs `&mut`).
        let read = cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(structure) = read.peek(&cache_key).and_then(Weak::upgrade) {
            return Ok(structure);
        }
        drop(read);

        let specs = coupled_sector_matrix_block_specs_from_layout(
            nout,
            rank,
            &layout,
            cache_key.shapes.as_ref(),
        )?;
        let structure = BlockStructure::from_blocks_with_rank(rank, specs)?.into_shared();

        let mut write = cache
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(existing) = write.get(&cache_key).and_then(Weak::upgrade) {
            return Ok(existing);
        }
        write.put(cache_key, Arc::downgrade(&structure));
        Ok(structure)
    }

    /// Builds the canonical coupled-sector layout directly from this hom
    /// space's authoritative per-leg degeneracies.
    ///
    /// Unlike [`Self::coupled_subblock_structure`], callers do not first
    /// materialize one owned shape per tree pair. The miss builder derives each
    /// codomain row and domain column once before forming final subblocks.
    pub fn coupled_subblock_structure_from_leg_degeneracies<R>(
        &self,
        rule: &R,
    ) -> Result<Arc<BlockStructure>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let key = CompleteHomSpaceStructureCacheKey::new(rule, self);
        if let Some(structure) = complete_hom_space_structure_cached(&key) {
            return Ok(structure);
        }

        let layout = self.cached_fusion_tree_layout(rule);
        let (sector, degeneracy) =
            coupled_subblock_parts_from_leg_degeneracies(self, &layout)?;
        let built = BlockStructure::from_parts(sector, degeneracy)?;
        built.record_storage_tiling();
        Ok(admit_complete_hom_space_structure(key, built.into_shared()))
    }

    #[doc(hidden)]
    pub fn coupled_subblock_layout_probe_uncached<R>(
        &self,
        rule: &R,
        source: &BlockStructure,
    ) -> Result<(usize, bool), CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let layout = self.fusion_tree_layout_data_uncached(rule);
        let (sector, degeneracy) =
            coupled_subblock_parts_from_leg_degeneracies(self, &layout)?;
        let required_len = degeneracy.required_len()?;
        Ok((
            required_len,
            source.sector_structure() == &sector
                && source.degeneracy_structure() == &degeneracy,
        ))
    }

    /// Multiplicity-aware sibling of
    /// [`Self::coupled_subblock_structure_from_leg_degeneracies`].
    ///
    /// Generic layouts are intentionally not published in the
    /// multiplicity-free layout cache. Their vertex-resolved keys are grouped
    /// ephemerally and fed through the same single-pass degeneracy builder.
    pub fn coupled_subblock_structure_from_leg_degeneracies_generic<R>(
        &self,
        rule: &R,
    ) -> Result<Arc<BlockStructure>, CoreError>
    where
        R: FusionRule,
    {
        let layout = self.fusion_tree_layout_data_generic(rule)?;
        let (sector, degeneracy) =
            coupled_subblock_parts_from_leg_degeneracies(self, &layout)?;
        let built = BlockStructure::from_parts(sector, degeneracy)?;
        built.record_storage_tiling();
        Ok(built.into_shared())
    }

    /// Checked Generic-fusion structural staging from this HomSpace's leg
    /// degeneracies. No layout or complete-HomSpace cache is consulted or
    /// populated until the fallible provider walk has succeeded.
    pub fn coupled_subblock_structure_from_leg_degeneracies_generic_checked<R>(
        &self,
        rule: &R,
    ) -> Result<Arc<BlockStructure>, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        Ok(self
            .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(rule)?
            .commit()
            .into_shared())
    }

    /// Stage checked Generic block metadata without entering the block
    /// interner. This is the transaction input used by owned checked
    /// transforms before their categorical plan is known to succeed.
    #[doc(hidden)]
    pub fn prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked<R>(
        &self,
        rule: &R,
    ) -> Result<PreparedBlockStructure, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let layout = self.fusion_tree_layout_data_generic_checked(rule)?;
        let (sector, degeneracy) =
            coupled_subblock_parts_from_leg_degeneracies(self, &layout)?;
        PreparedBlockStructure::from_parts(sector, degeneracy)
            .map(PreparedBlockStructure::with_storage_tiling)
            .map_err(Into::into)
    }

    #[cfg(test)]
    fn degeneracy_shape_for_key(
        &self,
        key: &FusionTreePairKey,
    ) -> Result<DimVec, CoreError> {
        let rank = self.rank();
        if key.codomain_uncoupled().len() != self.codomain().len()
            || key.domain_uncoupled().len() != self.domain().len()
        {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: key.codomain_uncoupled().len() + key.domain_uncoupled().len(),
            });
        }
        let mut shape = DimVec::new();
        for (leg, &sector) in self
            .codomain()
            .legs()
            .iter()
            .chain(self.domain().legs())
            .zip(
                key.codomain_uncoupled()
                    .iter()
                    .chain(key.domain_uncoupled()),
            )
        {
            shape.push(
                leg.degeneracy(sector)
                    .ok_or(CoreError::MalformedFusionTree {
                        message: "fusion tree uses a sector absent from its leg",
                    })?,
            );
        }
        Ok(shape)
    }

    #[cfg(test)]
    fn fusion_tree_keys_uncached<R>(&self, rule: &R) -> Vec<FusionTreePairKey>
    where
        R: MultiplicityFreeFusionRule,
    {
        let codomain = fusion_trees_by_coupled_for_space(rule, self.codomain());
        let domain = fusion_trees_by_coupled_for_space(rule, self.domain());
        let mut keys = Vec::new();
        let mut codomain_index = 0usize;
        let mut domain_index = 0usize;
        while codomain_index < codomain.len() && domain_index < domain.len() {
            match codomain[codomain_index]
                .coupled
                .cmp(&domain[domain_index].coupled)
            {
                std::cmp::Ordering::Less => codomain_index += 1,
                std::cmp::Ordering::Greater => domain_index += 1,
                std::cmp::Ordering::Equal => {
                    for domain_tree in &domain[domain_index].trees {
                        for codomain_tree in &codomain[codomain_index].trees {
                            keys.push(FusionTreePairKey::pair(
                                codomain_tree.clone(),
                                domain_tree.clone(),
                            ));
                        }
                    }
                    codomain_index += 1;
                    domain_index += 1;
                }
            }
        }
        keys
    }

    fn fusion_tree_layout_data_uncached<R>(&self, rule: &R) -> FusionTreeHomSpaceLayoutData
    where
        R: MultiplicityFreeFusionRule,
    {
        let codomain = fusion_trees_by_coupled_for_space(rule, self.codomain());
        let domain = fusion_trees_by_coupled_for_space(rule, self.domain());
        fusion_tree_layout_data_from_groups(&codomain, &domain)
    }

    fn try_fusion_tree_layout_data_uncached_checked<R>(
        &self,
        rule: &R,
    ) -> Result<FusionTreeHomSpaceLayoutData, FusionAlgebraError>
    where
        R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
    {
        let codomain = try_fusion_trees_by_coupled_for_space_checked(rule, self.codomain())?;
        let domain = try_fusion_trees_by_coupled_for_space_checked(rule, self.domain())?;
        Ok(fusion_tree_layout_data_from_groups(&codomain, &domain))
    }

    /// Generic-fusion (outer-multiplicity) sibling of [`Self::fusion_tree_keys`]:
    /// enumerates multiplicity-aware block keys (codomain × domain tree pairs)
    /// for a `FusionRule` whose `nsymbol` can exceed 1. Not cached —
    /// the Generic path is not on any hot loop yet (mirrors the non-memoized
    /// add a cache only when a Generic workload measures a benefit.
    ///
    /// Bounded-table semantics (Option A, refute/b3b-verify): the result is
    /// either the exact provider-defined block key set or an `Err` — never a silently
    /// truncated one. Full-space enumeration errs as soon as either side's
    /// coupled fold reports escaped candidates, tainted sectors, or a poisoned
    /// (beyond one-hop) fold, even when the offending sectors could not survive
    /// the codomain∩domain merge — conservative on purpose; use
    /// [`Self::fusion_tree_keys_generic_for_coupled`] for a single provably
    /// clean sector. Unbounded Generic rules never err.
    pub fn fusion_tree_keys_generic<R>(
        &self,
        rule: &R,
    ) -> Result<Vec<FusionTreePairKey>, CoreError>
    where
        R: FusionRule,
    {
        Ok(self.fusion_tree_layout_data_generic(rule)?.keys.to_vec())
    }

    /// Checked Generic-fusion sibling of [`Self::fusion_tree_keys_generic`].
    ///
    /// All channel, dual, multiplicity, bounded-table, and fold queries finish
    /// before this method returns keys. Generic layouts are ephemeral, so a
    /// provider error cannot retain a layout-cache or complete-space entry.
    pub fn fusion_tree_keys_generic_checked<R>(
        &self,
        rule: &R,
    ) -> Result<Vec<FusionTreePairKey>, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        Ok(self.fusion_tree_layout_data_generic_checked(rule)?.keys.to_vec())
    }

    fn fusion_tree_layout_data_generic<R>(
        &self,
        rule: &R,
    ) -> Result<FusionTreeHomSpaceLayoutData, CoreError>
    where
        R: FusionRule,
    {
        let checked = InfallibleGeneric::new(rule);
        match self.fusion_tree_layout_data_generic_checked(&checked) {
            Ok(data) => Ok(data),
            Err(CheckedGenericStructureError::Provider(never)) => match never {},
            Err(CheckedGenericStructureError::Core(error)) => Err(error),
        }
    }

    fn fusion_tree_layout_data_generic_checked<R>(
        &self,
        rule: &R,
    ) -> Result<FusionTreeHomSpaceLayoutData, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let (codomain, codomain_fold) =
            fusion_trees_by_coupled_for_space_generic_checked(rule, self.codomain())?;
        let (domain, domain_fold) =
            fusion_trees_by_coupled_for_space_generic_checked(rule, self.domain())?;
        for (side, fold) in [("codomain", &codomain_fold), ("domain", &domain_fold)] {
            if !fold.is_fully_clean() {
                return Err(CoreError::FusionOutsideTable {
                    message: fusion_fold_error_message(side, fold),
                }
                .into());
            }
        }
        Ok(fusion_tree_layout_data_from_groups(&codomain, &domain))
    }

    /// Block keys of ONE coupled sector, for spaces whose full enumeration is
    /// an `Err` but whose requested sector is provably clean (its complete
    /// complete tree set stays inside the provider catalog). `Err` when the sector is
    /// tainted on either side (its trees would need out-of-table intermediates)
    /// or the fold is poisoned; `Ok(vec![])` when the sector is simply not a
    /// shared coupled candidate.
    pub fn fusion_tree_keys_generic_for_coupled<R>(
        &self,
        rule: &R,
        coupled: SectorId,
    ) -> Result<Vec<FusionTreePairKey>, CoreError>
    where
        R: FusionRule,
    {
        let (codomain, codomain_fold) =
            fusion_trees_by_coupled_for_space_generic(rule, self.codomain());
        let (domain, domain_fold) = fusion_trees_by_coupled_for_space_generic(rule, self.domain());
        generic_keys_for_coupled_from_groups(
            &codomain,
            &codomain_fold,
            &domain,
            &domain_fold,
            coupled,
            |side, fold, coupled| CoreError::FusionOutsideTable {
                message: format!(
                    "coupled sector {coupled:?} on the {side} side requires out-of-catalog intermediates. {}",
                    fusion_fold_error_message(side, fold),
                ),
            },
        )
    }

    /// Checked counterpart of [`Self::fusion_tree_keys_generic_for_coupled`].
    pub fn fusion_tree_keys_generic_for_coupled_checked<R>(
        &self,
        rule: &R,
        coupled: SectorId,
    ) -> Result<Vec<FusionTreePairKey>, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let (codomain, codomain_fold) =
            fusion_trees_by_coupled_for_space_generic_checked(rule, self.codomain())?;
        let (domain, domain_fold) =
            fusion_trees_by_coupled_for_space_generic_checked(rule, self.domain())?;
        generic_keys_for_coupled_from_groups(
            &codomain,
            &codomain_fold,
            &domain,
            &domain_fold,
            coupled,
            |side, fold, coupled| CoreError::FusionOutsideTable {
                message: format!(
                    "generic fusion provider cannot exactly represent coupled sector {coupled:?} on the {side} side ({fold:?}); block dimensions are either exact or an error, never truncated.",
                ),
            },
        )
        .map_err(Into::into)
    }

    pub fn sector_structure<R>(&self, rule: &R) -> Result<SectorStructure, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let rank = self.codomain().len() + self.domain().len();
        // `from_keys` builds owned `BlockKey`s, so cloning out of the shared
        // slice is unavoidable here (a cold structure-build path, not a hot loop).
        SectorStructure::from_keys(rank, self.fusion_tree_keys(rule).iter().cloned())
    }

    pub fn unique_fusion_tree_key_from_external_sectors<R>(
        &self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<FusionTreePairKey, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let mut keys = self.fusion_tree_keys_from_external_sectors(rule, sectors)?;
        if keys.len() != 1 {
            return Err(CoreError::BlockCountMismatch {
                expected: 1,
                actual: keys.len(),
            });
        }
        Ok(keys.remove(0))
    }

    /// Lowers external per-leg sectors to fusion-tree keys. Domain-side
    /// external sectors are dualized into internal tree sectors here, the
    /// same convention TensorKit applies in `subblock(t, sectors)`.
    pub fn fusion_tree_keys_from_external_sectors<R>(
        &self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<Vec<FusionTreePairKey>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let rank = self.codomain().len() + self.domain().len();
        if sectors.len() != rank {
            return Err(CoreError::DimensionMismatch {
                expected: rank,
                actual: sectors.len(),
            });
        }

        let codomain = fusion_trees_by_coupled_for_selected_space(
            rule,
            self.codomain(),
            &sectors[..self.codomain().len()],
        )?;
        let domain_sectors = sectors[self.codomain().len()..]
            .iter()
            .map(|&sector| rule.dual(sector))
            .collect::<Vec<_>>();
        let domain =
            fusion_trees_by_coupled_for_selected_space(rule, self.domain(), &domain_sectors)?;
        let mut keys = Vec::new();
        let mut codomain_index = 0usize;
        let mut domain_index = 0usize;
        while codomain_index < codomain.len() && domain_index < domain.len() {
            match codomain[codomain_index]
                .coupled
                .cmp(&domain[domain_index].coupled)
            {
                std::cmp::Ordering::Less => codomain_index += 1,
                std::cmp::Ordering::Greater => domain_index += 1,
                std::cmp::Ordering::Equal => {
                    for domain_tree in &domain[domain_index].trees {
                        for codomain_tree in &codomain[codomain_index].trees {
                            keys.push(FusionTreePairKey::pair(
                                codomain_tree.clone(),
                                domain_tree.clone(),
                            ));
                        }
                    }
                    codomain_index += 1;
                    domain_index += 1;
                }
            }
        }

        Ok(keys)
    }

    /// Validates per-tree degeneracy shapes against the leg-level
    /// degeneracies (the legs are authoritative): for every tree key,
    /// `shape[axis]` must equal the degeneracy the axis' leg stores for the
    /// tree's uncoupled sector on that axis.
    pub fn validate_degeneracy_shapes<S>(
        &self,
        keys: &[FusionTreePairKey],
        shapes: &[S],
    ) -> Result<(), CoreError>
    where
        S: AsRef<[usize]>,
    {
        let legs = self
            .codomain()
            .legs()
            .iter()
            .chain(self.domain().legs())
            .collect::<Vec<_>>();
        for (key, shape) in keys.iter().zip(shapes) {
            let shape = shape.as_ref();
            if shape.len() != legs.len() {
                return Err(CoreError::StructureRankMismatch {
                    expected: legs.len(),
                    actual: shape.len(),
                });
            }
            let uncoupled = key
                .codomain_uncoupled()
                .iter()
                .chain(key.domain_uncoupled());
            for ((leg, &sector), &dim) in legs.iter().zip(uncoupled).zip(shape) {
                let expected = leg
                    .degeneracy(sector)
                    .ok_or(CoreError::MalformedFusionTree {
                        message: "fusion tree uses a sector absent from its leg",
                    })?;
                if expected != dim {
                    return Err(CoreError::LegDegeneracyMismatch {
                        sector,
                        expected,
                        actual: dim,
                    });
                }
            }
        }
        Ok(())
    }

    /// Validates every present fusion-tree block against this HomSpace.
    ///
    /// Missing valid tree pairs are structural zeros. Block order, strides,
    /// offsets, and overlap are storage properties and are intentionally not
    /// part of this categorical subset proof.
    #[doc(hidden)]
    pub fn validate_subblock_structure_subset<R>(
        &self,
        rule: &R,
        structure: &BlockStructure,
    ) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        StructurallyValidatedFusionTreeSubset::try_new(self, structure)?.validate_for_rule(rule)
    }

    /// Checked finite-algebra sibling of [`Self::validate_subblock_structure_subset`].
    #[doc(hidden)]
    pub fn validate_subblock_structure_subset_checked<R>(
        &self,
        rule: &R,
        structure: &BlockStructure,
    ) -> Result<(), CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        StructurallyValidatedFusionTreeSubset::try_new(self, structure)?
            .validate_for_rule_checked(rule)
    }

    pub fn fusion_tree_groups<R>(&self, rule: &R) -> Result<Vec<FusionTreeBlockGroup>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        self.sector_structure(rule)
            .map(SectorStructure::into_fusion_tree_groups)
    }

    /// External leg view of flat axis `axis` (TensorKit's `space(t, i)`
    /// convention, homspace.jl:60-62): codomain legs verbatim, domain legs
    /// dualized. Degeneracies are carried along, keyed by the external
    /// (placement-invariant) sector labels.
    pub fn external_axis_leg<R>(&self, rule: &R, axis: usize) -> SectorLeg
    where
        R: FusionRule,
    {
        if axis < self.codomain().len() {
            self.codomain().legs()[axis].clone()
        } else {
            dual_sector_leg(rule, &self.domain().legs()[axis - self.codomain().len()])
        }
    }

    fn external_axis_leg_view(&self, axis: usize) -> OrientedLegView<'_> {
        if axis < self.codomain().len() {
            OrientedLegView::borrowed(&self.codomain().legs()[axis])
        } else {
            OrientedLegView::borrowed(&self.domain().legs()[axis - self.codomain().len()]).toggled()
        }
    }

    /// Returns the duality flag in the external-axis convention. Why not read
    /// a domain leg verbatim: external domain axes are duals of their stored
    /// hom-space legs, matching [`Self::external_axis_leg`].
    pub fn external_axis_is_dual(&self, axis: usize) -> Option<bool> {
        if axis < self.codomain().len() {
            Some(self.codomain().legs()[axis].is_dual())
        } else if axis < self.rank() {
            Some(!self.domain().legs()[axis - self.codomain().len()].is_dual())
        } else {
            None
        }
    }
}
