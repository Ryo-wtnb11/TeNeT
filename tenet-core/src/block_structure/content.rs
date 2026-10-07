use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockRef<'a> {
    pub(super) key: &'a BlockKey,
    pub(super) degeneracy: DegeneracyBlockRef<'a>,
}

impl<'a> BlockRef<'a> {
    #[inline]
    pub fn key(&self) -> &'a BlockKey {
        self.key
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.degeneracy.shape()
    }

    #[inline]
    pub fn strides(&self) -> &'a [usize] {
        self.degeneracy.strides()
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.degeneracy.offset()
    }

    pub fn element_count(&self) -> Result<usize, CoreError> {
        self.degeneracy.element_count()
    }

    pub fn storage_end_exclusive(&self) -> Result<usize, CoreError> {
        self.degeneracy.storage_end_exclusive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// One fusion tree's contiguous row or column extent inside a coupled-sector matrix.
pub struct CoupledTreeExtent {
    pub(super) tree: FusionTreeKey,
    pub(super) offset: usize,
    pub(super) shape: DimVec,
}

impl CoupledTreeExtent {
    /// Fusion tree identifying this row or column extent.
    pub fn tree(&self) -> &FusionTreeKey {
        &self.tree
    }

    /// Row or column offset from the start of the coupled-sector matrix.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Degeneracy shape whose element count is this tree's matrix extent.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Checked product of the degeneracy shape.
    pub fn extent(&self) -> Result<usize, CoreError> {
        checked_product(&self.shape)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Checked contiguous column-major storage region for one coupled sector.
pub struct CoupledSectorRegion {
    pub(super) coupled: SectorId,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) range: core::ops::Range<usize>,
    /// Row trees, then column trees: one allocation per region.
    pub(super) trees: Box<[CoupledTreeExtent]>,
    pub(super) row_tree_count: usize,
    pub(super) aligned_diagonal: bool,
}

impl CoupledSectorRegion {
    /// Coupled-sector label shared by every fusion-tree block in this region.
    pub fn coupled(&self) -> SectorId {
        self.coupled
    }

    /// Number of rows across all codomain-tree extents.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns across all domain-tree extents.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Exact element range of the contiguous sector matrix in flat storage.
    pub fn range(&self) -> core::ops::Range<usize> {
        self.range.clone()
    }

    /// Codomain trees with their row offsets and degeneracy shapes.
    pub fn row_trees(&self) -> &[CoupledTreeExtent] {
        &self.trees[..self.row_tree_count]
    }

    /// Domain trees with their column offsets and degeneracy shapes.
    pub fn col_trees(&self) -> &[CoupledTreeExtent] {
        &self.trees[self.row_tree_count..]
    }

    #[doc(hidden)]
    pub fn has_aligned_diagonal(&self) -> bool {
        self.aligned_diagonal
    }
}

type CoupledRegionResult = Result<Option<Arc<[CoupledSectorRegion]>>, CoreError>;
pub(super) type CoupledRegionCache = Arc<[OnceLock<CoupledRegionResult>]>;

#[derive(Clone, Eq)]
pub struct BlockStructureContent {
    pub(crate) id: usize,
    semantic_hash: u64,
    /// Shared with every structure of the same sector layout: a
    /// degeneracy-only change reuses it (TensorKit `sectorstructure`).
    pub(crate) sector: Arc<SectorStructure>,
    pub(crate) degeneracy: DegeneracyStructure,
    pub(crate) required_len: usize,
    pub(crate) storage_tiling: StorageTilingProof,
}

/// Set once a constructor proved that the blocks' reachable offsets partition
/// `0..required_len`, each exactly once. That is a function of the geometry
/// equality compares, so the bit is sound on every structure sharing this
/// content and is excluded from equality. Why not compute it on demand: the
/// general check (`coupled_sector_regions`) allocates and hashes, and is
/// cached on the short-lived wrapper, not on this immutable content.
#[derive(Debug, Default)]
pub(crate) struct StorageTilingProof(core::sync::atomic::AtomicBool);

// A derived fact about the compared geometry, never part of equality.
impl PartialEq for StorageTilingProof {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for StorageTilingProof {}

impl Clone for StorageTilingProof {
    fn clone(&self) -> Self {
        Self(core::sync::atomic::AtomicBool::new(
            self.0.load(Ordering::Acquire),
        ))
    }
}

impl core::fmt::Debug for BlockStructureContent {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("BlockStructureContent")
            .field("id", &self.id)
            .field("rank", &self.sector.rank())
            .field("degeneracy", &self.degeneracy)
            .finish()
    }
}

// Content equality deliberately treats `id` only as a same-instance shortcut:
// independently built equal content has a distinct process-local id and must
// still compare equal by geometry. Id-keyed caches remain intentionally
// stricter: they key on `id()` explicitly and rely on monotonicity.
impl PartialEq for BlockStructureContent {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            || (self.semantic_hash == other.semantic_hash
                && self.sector == other.sector
                && self.degeneracy == other.degeneracy
                && self.required_len == other.required_len)
    }
}

impl std::hash::Hash for BlockStructureContent {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.semantic_hash.hash(state);
    }
}

impl BlockStructureContent {
    pub(crate) fn new(
        sector: Arc<SectorStructure>,
        degeneracy: DegeneracyStructure,
        required_len: usize,
    ) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        sector.rank().hash(&mut hasher);
        for block in sector.blocks() {
            block.key().hash(&mut hasher);
        }
        degeneracy.rank.hash(&mut hasher);
        degeneracy.dims.hash(&mut hasher);
        degeneracy.offsets.hash(&mut hasher);
        required_len.hash(&mut hasher);
        Self {
            id: BLOCK_STRUCTURE_CONTENT_ID.fetch_add(1, Ordering::Relaxed),
            semantic_hash: hasher.finish(),
            sector,
            degeneracy,
            required_len,
            storage_tiling: StorageTilingProof::default(),
        }
    }

    /// Process-local, monotonically assigned, never-reused content identity.
    /// One canonical complete-HomSpace cache entry and its live consumers
    /// share an id. Independently constructed or rebuilt equal content receives
    /// a fresh id. It is not semantic identity and must never be serialized.
    #[inline]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Whether a constructor proved that the blocks tile `0..required_len`
    /// exactly once (see [`BlockStructure::storage_tiling_proven`]).
    #[inline]
    pub(crate) fn storage_tiling_proven(&self) -> bool {
        self.storage_tiling.0.load(Ordering::Acquire)
    }

    /// Records the tiling proof. Callers must have built this exact geometry
    /// so that the blocks are pairwise disjoint and each reaches its offsets
    /// once; the element counts are checked here to sum to `required_len`, so
    /// the disjoint blocks cover all of it.
    pub(super) fn record_storage_tiling(&self) {
        let covered = self.degeneracy.blocks().try_fold(0usize, |total, block| {
            checked_product(block.shape())
                .ok()
                .and_then(|count| total.checked_add(count))
        });
        if covered == Some(self.required_len) {
            self.storage_tiling.0.store(true, Ordering::Release);
        }
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.sector.rank()
    }

    /// The degeneracy-free part of this content: rank and ordered block keys.
    #[doc(hidden)]
    #[inline]
    pub fn sector_structure(&self) -> &SectorStructure {
        &self.sector
    }

    #[doc(hidden)]
    pub fn charged_retained_bytes(&self) -> usize {
        std::mem::size_of::<BlockStructureContent>()
            .saturating_add(2 * std::mem::size_of::<usize>())
            .saturating_add(std::mem::size_of::<SectorStructure>())
            .saturating_add(self.sector.charged_heap_bytes())
            .saturating_add(self.degeneracy.charged_heap_bytes())
    }
}

impl SectorStructure {
    /// Conservative heap bytes this sector structure retains.
    pub(crate) fn charged_heap_bytes(&self) -> usize {
        fn key_bytes(key: &BlockKey, seen: &mut rustc_hash::FxHashSet<usize>) -> usize {
            match key {
                BlockKey::Dense => 0,
                BlockKey::Opaque(key) => spilled_smallvec_heap_bytes(&key.words),
                BlockKey::FusionTree(pair) => {
                    charge_fusion_tree_key_backings(seen, pair.codomain_tree())
                        .saturating_add(charge_fusion_tree_key_backings(seen, pair.domain_tree()))
                }
            }
        }

        let mut frozen_backings = rustc_hash::FxHashSet::default();
        let sector_blocks = self.blocks.iter().fold(0usize, |bytes, block| {
            bytes.saturating_add(key_bytes(block.key(), &mut frozen_backings))
        });
        let groups = self.fusion_tree_groups.iter().fold(0usize, |bytes, group| {
            bytes
                .saturating_add(spilled_smallvec_heap_bytes(&group.block_indices))
                .saturating_add(charge_fusion_tree_group_key_backings(
                    &mut frozen_backings,
                    &group.group_key,
                ))
        });
        let compact_lookup = self
            .compact_lookup
            .as_ref()
            .map_or(0, |lookup| spilled_smallvec_heap_bytes(&lookup.indices));

        self.blocks
            .capacity()
            .saturating_mul(std::mem::size_of::<SectorBlock>())
            .saturating_add(sector_blocks)
            .saturating_add(
                self.fusion_tree_groups
                    .capacity()
                    .saturating_mul(std::mem::size_of::<FusionTreeBlockGroup>()),
            )
            .saturating_add(groups)
            .saturating_add(spilled_smallvec_heap_bytes(&self.sorted_indices))
            .saturating_add(compact_lookup)
    }
}
