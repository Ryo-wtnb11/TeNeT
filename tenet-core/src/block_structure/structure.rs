use super::*;

#[derive(Default)]
pub(crate) struct BlockStructureRegionState {
    coupled_region_cache: OnceLock<CoupledRegionCache>,
}

pub struct BlockStructure {
    pub(super) content: Arc<BlockStructureContent>,
    regions: Arc<BlockStructureRegionState>,
}

impl Clone for BlockStructure {
    fn clone(&self) -> Self {
        Self {
            content: Arc::clone(&self.content),
            regions: Arc::clone(&self.regions),
        }
    }
}

impl core::fmt::Debug for BlockStructure {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("BlockStructure")
            .field("sector", &self.content.sector)
            .field("degeneracy", &self.content.degeneracy)
            .field("content", &self.content)
            .field("required_len", &self.content.required_len)
            .finish()
    }
}

impl PartialEq for BlockStructure {
    fn eq(&self, other: &Self) -> bool {
        self.content == other.content
    }
}

impl Eq for BlockStructure {}

impl BlockStructure {
    pub(crate) fn from_content(content: Arc<BlockStructureContent>) -> Self {
        Self {
            content,
            regions: Arc::new(BlockStructureRegionState::default()),
        }
    }
}

/// Fully validated block metadata that has not entered the global interner.
///
/// The borrowed preview is suitable for fallible plan compilation. Call
/// [`Self::commit`] only after all later validation has succeeded.
#[doc(hidden)]
pub struct PreparedBlockStructure {
    sector: Arc<SectorStructure>,
    degeneracy: DegeneracyStructure,
    required_len: usize,
    preview: OnceLock<BlockStructure>,
    storage_tiling: bool,
}

impl PreparedBlockStructure {
    fn from_blocks_with_rank(rank: usize, blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        let keys = blocks
            .iter()
            .map(|block| block.key().clone())
            .collect::<Vec<_>>();
        let degeneracy_blocks = blocks
            .into_iter()
            .map(|block| DegeneracyBlock::new(block.shape, block.strides, block.offset))
            .collect::<Result<Vec<_>, _>>()?;
        let sector = SectorStructure::from_keys(rank, keys)?;
        let degeneracy = DegeneracyStructure::from_blocks_with_rank(rank, degeneracy_blocks)?;
        Self::from_parts(sector, degeneracy)
    }

    pub(crate) fn from_parts(
        sector: SectorStructure,
        degeneracy: DegeneracyStructure,
    ) -> Result<Self, CoreError> {
        Self::from_shared_parts(Arc::new(sector), degeneracy)
    }

    pub(crate) fn from_shared_parts(
        sector: Arc<SectorStructure>,
        degeneracy: DegeneracyStructure,
    ) -> Result<Self, CoreError> {
        if sector.rank() != degeneracy.rank() {
            return Err(CoreError::StructureRankMismatch {
                expected: sector.rank(),
                actual: degeneracy.rank(),
            });
        }
        if sector.block_count() != degeneracy.block_count() {
            return Err(CoreError::BlockCountMismatch {
                expected: sector.block_count(),
                actual: degeneracy.block_count(),
            });
        }
        let required_len = degeneracy.required_len()?;
        Ok(Self {
            sector,
            degeneracy,
            required_len,
            preview: OnceLock::new(),
            storage_tiling: false,
        })
    }

    /// Marks this staged geometry as built by the canonical coupled-sector
    /// builder; the preview and the committed structure then record
    /// [`BlockStructure::storage_tiling_proven`].
    pub(crate) fn with_storage_tiling(mut self) -> Self {
        self.storage_tiling = true;
        self
    }

    /// Borrow an uninterned structure for validation and plan compilation.
    #[doc(hidden)]
    pub fn structure(&self) -> &BlockStructure {
        self.preview.get_or_init(|| {
            let preview = BlockStructure::from_content(Arc::new(BlockStructureContent {
                id: BLOCK_STRUCTURE_CONTENT_ID.fetch_add(1, Ordering::Relaxed),
                sector: Arc::clone(&self.sector),
                degeneracy: self.degeneracy.clone(),
                required_len: self.required_len,
                storage_tiling: StorageTilingProof::default(),
            }));
            if self.storage_tiling {
                preview.record_storage_tiling();
            }
            preview
        })
    }

    #[doc(hidden)]
    pub fn required_len(&self) -> usize {
        self.required_len
    }

    /// Borrow the staged block keys alone. Why not `structure()`: key-only
    /// prevalidation must not pay for cloning both structures into the preview.
    #[doc(hidden)]
    pub fn sector_structure(&self) -> &SectorStructure {
        &self.sector
    }

    /// Publish the validated structure through the existing interner.
    #[doc(hidden)]
    pub fn commit(self) -> BlockStructure {
        let structure = BlockStructure::from_content(intern_block_structure_content(
            self.sector,
            self.degeneracy,
            self.required_len,
        ));
        if self.storage_tiling {
            structure.record_storage_tiling();
        }
        structure
    }
}

impl BlockStructure {
    /// Conservative retained bytes for this structure's immutable layout.
    ///
    /// The fixed region-state allocation is included, but its lazily derived
    /// coupled-region operation cache is not: that shared cache can be grown by
    /// non-workspace owners and is not required to retain or reactivate an idle
    /// destination. Excluding it also keeps this measurement allocation-free
    /// and stable for the lifetime of the structure.
    #[doc(hidden)]
    pub fn charged_retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.content.charged_retained_bytes())
            .saturating_add(std::mem::size_of::<BlockStructureRegionState>())
            .saturating_add(2 * std::mem::size_of::<usize>())
    }

    pub fn trivial(shape: &[usize]) -> Result<Self, CoreError> {
        Self::from_parts(
            SectorStructure::dense(shape.len()),
            DegeneracyStructure::packed_column_major(shape.len(), [shape.to_vec()])?,
        )
    }

    pub fn empty(rank: usize) -> Self {
        let sector = Arc::new(SectorStructure::empty(rank));
        let degeneracy = DegeneracyStructure::empty(rank);
        Self::from_content(intern_block_structure_content(sector, degeneracy, 0))
    }

    pub fn from_blocks(blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        let rank = blocks.first().map(|block| block.shape().len()).unwrap_or(0);
        Self::from_blocks_with_rank(rank, blocks)
    }

    pub fn from_blocks_with_rank(rank: usize, blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        PreparedBlockStructure::from_blocks_with_rank(rank, blocks)
            .map(|prepared| prepared.commit())
    }

    pub fn from_parts(
        sector: SectorStructure,
        degeneracy: DegeneracyStructure,
    ) -> Result<Self, CoreError> {
        PreparedBlockStructure::from_parts(sector, degeneracy).map(|prepared| prepared.commit())
    }

    /// [`Self::from_parts`] over a sector structure shared with other
    /// structures of the same sector layout.
    pub(crate) fn from_shared_parts(
        sector: Arc<SectorStructure>,
        degeneracy: DegeneracyStructure,
    ) -> Result<Self, CoreError> {
        PreparedBlockStructure::from_shared_parts(sector, degeneracy)
            .map(|prepared| prepared.commit())
    }

    pub fn into_shared(self) -> Arc<Self> {
        canonicalize_block_structure_arc(Arc::new(self))
    }

    pub fn canonicalize_shared(structure: Arc<Self>) -> Arc<Self> {
        canonicalize_block_structure_arc(structure)
    }

    pub fn packed_column_major<I>(rank: usize, shapes: I) -> Result<Self, CoreError>
    where
        I: IntoIterator,
        I::Item: Into<Vec<usize>>,
    {
        let shapes = shapes
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Vec<usize>>>();
        let sector = SectorStructure::from_keys(rank, (0..shapes.len()).map(BlockKey::ordinal))?;
        let degeneracy = DegeneracyStructure::packed_column_major(rank, shapes)?;
        Self::from_parts(sector, degeneracy)
    }

    /// Coupled-sector matrix layout over fusion-tree block keys.
    ///
    /// Every key is first validated against `rule` in caller order. Blocks are
    /// then stable-sorted by coupled sector, and each coupled sector is laid out
    /// as one contiguous column-major matrix with the fusion-tree subblocks as strided views (see
    /// [`FusionTensorMapSpace::from_degeneracy_shapes_coupled`]). Fails when a
    /// coupled sector does not cover its full codomain-tree x domain-tree
    /// grid, because the sector matrix would contain uninitialized holes.
    ///
    /// Each key inherits [`FusionTreePairKey::validate_for_rule`]'s
    /// provider-domain precondition: arbitrary numeric sector IDs are not
    /// checked, and an infallible finite provider may panic on such IDs.
    pub fn coupled_sector_matrix_with_keys<R>(
        rule: &R,
        nout: usize,
        rank: usize,
        blocks: Vec<(FusionTreePairKey, Vec<usize>)>,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        // Why not retain an otherwise-dead provider argument: this public
        // constructor is a categorical admission boundary, so the provider
        // validates every key in caller order before layout arithmetic begins.
        for (key, _) in &blocks {
            key.validate_for_rule(rule)?;
        }
        coupled_sector_matrix_from_validated_keys(nout, rank, blocks)
    }

    /// Checked finite-algebra sibling of [`Self::coupled_sector_matrix_with_keys`].
    pub fn coupled_sector_matrix_with_keys_checked<R>(
        rule: &R,
        nout: usize,
        rank: usize,
        blocks: Vec<(FusionTreePairKey, Vec<usize>)>,
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        validate_coupled_sector_matrix_dimensions(
            nout,
            rank,
            blocks.iter().map(|(_, shape)| shape),
        )?;
        for (key, _) in &blocks {
            ShapeValidatedFusionTree::try_new(key.codomain_tree())?;
            ShapeValidatedFusionTree::try_new(key.domain_tree())?;
            let actual_nout = key.codomain_tree().uncoupled().len();
            let actual_nin = key.domain_tree().uncoupled().len();
            if actual_nout != nout || actual_nin != rank - nout {
                return Err(CoreError::FusionSpaceSplitMismatch {
                    expected_nout: nout,
                    expected_nin: rank - nout,
                    actual_nout,
                    actual_nin,
                }
                .into());
            }
            validate_fusion_tree_pair_coupled(key.codomain_tree(), key.domain_tree())?;
        }
        let prepared = {
            let mut order = (0..blocks.len()).collect::<Vec<_>>();
            order.sort_by_key(|&index| blocks[index].0.codomain_tree().coupled().id());
            let keys = order
                .iter()
                .map(|&index| &blocks[index].0)
                .collect::<Vec<_>>();
            let shapes = order
                .iter()
                .map(|&index| blocks[index].1.as_slice())
                .collect::<Vec<_>>();
            let specs = coupled_sector_matrix_block_specs_after_dimension_validation(
                nout, rank, &keys, &shapes,
            )?;
            PreparedBlockStructure::from_blocks_with_rank(rank, specs)?
        };
        for (key, _) in &blocks {
            validate_fusion_tree_for_rule_checked_after_shape(rule, key.codomain_tree())?;
            validate_fusion_tree_for_rule_checked_after_shape(rule, key.domain_tree())?;
        }
        Ok(prepared.commit())
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.content.sector.rank()
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.content.sector.block_count()
    }

    #[inline]
    pub fn sector_structure(&self) -> &SectorStructure {
        &self.content.sector
    }

    #[inline]
    pub fn degeneracy_structure(&self) -> &DegeneracyStructure {
        &self.content.degeneracy
    }

    #[inline]
    pub fn content_id(&self) -> usize {
        self.content.id()
    }

    #[inline]
    pub fn content_key(&self) -> Arc<BlockStructureContent> {
        Arc::clone(&self.content)
    }

    pub fn fusion_tree_groups(&self) -> Vec<FusionTreeBlockGroup> {
        self.content.sector.fusion_tree_groups()
    }

    /// Borrow construction-time fusion-tree groups without rebuilding them.
    #[inline]
    pub fn fusion_tree_group_slice(&self) -> &[FusionTreeBlockGroup] {
        self.content.sector.fusion_tree_group_slice()
    }

    pub fn find_block_index_by_key(&self, key: &BlockKey) -> Option<usize> {
        self.content.sector.find_index(key)
    }

    pub fn find_block_index_by_fusion_tree_pair(&self, key: &FusionTreePairKey) -> Option<usize> {
        self.content.sector.find_fusion_tree_pair_index(key)
    }

    #[deprecated(
        since = "0.1.0",
        note = "renamed to find_block_index_by_fusion_tree_pair to match FusionTreePairKey"
    )]
    pub fn find_block_index_by_fusion_tree_key(&self, key: &FusionTreePairKey) -> Option<usize> {
        self.find_block_index_by_fusion_tree_pair(key)
    }

    pub fn pair_block_indices_from(&self, src: &BlockStructure) -> Result<Vec<usize>, CoreError> {
        self.content.sector.pair_indices_from(&src.content.sector)
    }

    pub fn only_block(&self) -> Result<BlockRef<'_>, CoreError> {
        if self.block_count() == 1 {
            self.block(0)
        } else {
            Err(CoreError::BlockCountMismatch {
                expected: 1,
                actual: self.block_count(),
            })
        }
    }

    pub fn block(&self, index: usize) -> Result<BlockRef<'_>, CoreError> {
        Ok(BlockRef {
            key: self.content.sector.key(index)?,
            degeneracy: self.content.degeneracy.block(index)?,
        })
    }

    pub fn block_by_key(&self, key: &BlockKey) -> Result<BlockRef<'_>, CoreError> {
        let index =
            self.find_block_index_by_key(key)
                .ok_or_else(|| CoreError::MissingBlockKey {
                    key: Box::new(key.clone()),
                })?;
        self.block(index)
    }

    pub fn fusion_tree_pair_block(
        &self,
        key: &FusionTreePairKey,
    ) -> Result<BlockRef<'_>, CoreError> {
        let index = self
            .find_block_index_by_fusion_tree_pair(key)
            .ok_or_else(|| CoreError::MissingBlockKey {
                key: Box::new(BlockKey::FusionTree(key.clone())),
            })?;
        self.block(index)
    }

    #[doc(hidden)]
    pub fn find_block_index_by_adjoint_fusion_tree_pair(
        &self,
        logical_key: &FusionTreePairKey,
    ) -> Option<usize> {
        self.content
            .sector
            .find_adjoint_fusion_tree_pair_index(logical_key)
    }

    #[deprecated(
        since = "0.1.0",
        note = "renamed to fusion_tree_pair_block to match FusionTreePairKey"
    )]
    pub fn fusion_tree_block(&self, key: &FusionTreePairKey) -> Result<BlockRef<'_>, CoreError> {
        self.fusion_tree_pair_block(key)
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        Ok(self.content.required_len)
    }

    /// Whether this structure's blocks are proved, at construction, to reach
    /// every offset of `0..required_len` exactly once, so writing each block
    /// once fully initializes an owned payload. O(1); `false` only means no
    /// constructor recorded the proof.
    #[doc(hidden)]
    #[inline]
    pub fn storage_tiling_proven(&self) -> bool {
        self.content.storage_tiling_proven()
    }

    /// See [`BlockStructureContent::record_storage_tiling`].
    pub(crate) fn record_storage_tiling(&self) {
        self.content.record_storage_tiling();
    }

    /// Compiles the canonical coupled-sector matrix layout of this structure.
    ///
    /// `Ok(Some(_))` contains immutable checked regions covering storage exactly once.
    /// `Ok(None)` means the structure is valid but is not the canonical contiguous
    /// coupled-sector layout. `Err` reports a structural lookup or size overflow.
    pub fn coupled_sector_regions(
        &self,
        nout: usize,
    ) -> Result<Option<Arc<[CoupledSectorRegion]>>, CoreError> {
        if nout > self.rank() {
            return Ok(None);
        }
        self.regions
            .coupled_region_cache
            .get_or_init(|| new_coupled_region_cache(self.rank()))[nout]
            .get_or_init(|| {
                compile_coupled_sector_regions(self, nout).map(|regions| regions.map(Arc::from))
            })
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn coupled_region_cache_is_initialized(&self) -> bool {
        self.regions.coupled_region_cache.get().is_some()
    }

    #[cfg(test)]
    pub(crate) fn weak_region_state(&self) -> Weak<BlockStructureRegionState> {
        Arc::downgrade(&self.regions)
    }
}

fn coupled_sector_matrix_from_validated_keys(
    nout: usize,
    rank: usize,
    mut blocks: Vec<(FusionTreePairKey, Vec<usize>)>,
) -> Result<BlockStructure, CoreError> {
    blocks.sort_by_key(|(key, _)| key.codomain_tree().coupled().id());
    let (keys, shapes): (Vec<_>, Vec<_>) = blocks.into_iter().unzip();
    let specs = coupled_sector_matrix_block_specs(nout, rank, &keys, &shapes)?;
    BlockStructure::from_blocks_with_rank(rank, specs)
}
