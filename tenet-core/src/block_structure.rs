#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreeBlockKey {
    codomain_tree: FusionTreeKey,
    domain_tree: FusionTreeKey,
}

impl FusionTreeBlockKey {
    pub fn new(
        uncoupled: Vec<SectorId>,
        coupled: Option<SectorId>,
        vertices: Vec<SectorId>,
    ) -> Self {
        let is_dual = vec![false; uncoupled.len()];
        Self::pair(
            FusionTreeKey::new(uncoupled, coupled, is_dual, Vec::new(), vertices),
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                None,
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
        )
    }

    pub fn pair(codomain_tree: FusionTreeKey, domain_tree: FusionTreeKey) -> Self {
        Self {
            codomain_tree,
            domain_tree,
        }
    }

    pub fn pair_from_sector_ids<
        Codomain,
        Domain,
        CodomainDual,
        DomainDual,
        CodomainInner,
        DomainInner,
        CodomainVertices,
        DomainVertices,
    >(
        codomain_uncoupled: Codomain,
        domain_uncoupled: Domain,
        coupled: Option<usize>,
        codomain_is_dual: CodomainDual,
        domain_is_dual: DomainDual,
        codomain_innerlines: CodomainInner,
        domain_innerlines: DomainInner,
        codomain_vertices: CodomainVertices,
        domain_vertices: DomainVertices,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
        CodomainDual: IntoIterator<Item = bool>,
        DomainDual: IntoIterator<Item = bool>,
        CodomainInner: IntoIterator<Item = usize>,
        DomainInner: IntoIterator<Item = usize>,
        CodomainVertices: IntoIterator<Item = usize>,
        DomainVertices: IntoIterator<Item = usize>,
    {
        Self::pair(
            FusionTreeKey::from_sector_ids(
                codomain_uncoupled,
                coupled,
                codomain_is_dual,
                codomain_innerlines,
                codomain_vertices,
            ),
            FusionTreeKey::from_sector_ids(
                domain_uncoupled,
                coupled,
                domain_is_dual,
                domain_innerlines,
                domain_vertices,
            ),
        )
    }

    pub fn from_uncoupled<I>(uncoupled: I) -> Self
    where
        I: IntoIterator<Item = SectorId>,
    {
        Self::pair(
            FusionTreeKey::from_uncoupled(uncoupled),
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                None,
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
        )
    }

    #[inline]
    pub fn codomain_tree(&self) -> &FusionTreeKey {
        &self.codomain_tree
    }

    #[inline]
    pub fn domain_tree(&self) -> &FusionTreeKey {
        &self.domain_tree
    }

    #[inline]
    pub fn uncoupled(&self) -> &[SectorId] {
        self.codomain_tree.uncoupled()
    }

    #[inline]
    pub fn codomain_uncoupled(&self) -> &[SectorId] {
        self.codomain_tree.uncoupled()
    }

    #[inline]
    pub fn domain_uncoupled(&self) -> &[SectorId] {
        self.domain_tree.uncoupled()
    }

    #[inline]
    pub fn coupled(&self) -> Option<SectorId> {
        self.codomain_tree.coupled()
    }

    #[inline]
    pub fn vertices(&self) -> &[SectorId] {
        self.codomain_tree.vertices()
    }

    #[inline]
    pub fn codomain_vertices(&self) -> &[SectorId] {
        self.codomain_tree.vertices()
    }

    #[inline]
    pub fn domain_vertices(&self) -> &[SectorId] {
        self.domain_tree.vertices()
    }

    #[inline]
    pub fn codomain_innerlines(&self) -> &[SectorId] {
        self.codomain_tree.innerlines()
    }

    #[inline]
    pub fn domain_innerlines(&self) -> &[SectorId] {
        self.domain_tree.innerlines()
    }

    #[inline]
    pub fn codomain_is_dual(&self) -> &[bool] {
        self.codomain_tree.is_dual()
    }

    #[inline]
    pub fn domain_is_dual(&self) -> &[bool] {
        self.domain_tree.is_dual()
    }

    pub fn external_sectors<R>(&self, rule: &R) -> Vec<SectorId>
    where
        R: FusionRule,
    {
        let mut sectors = Vec::with_capacity(
            self.codomain_tree.uncoupled().len() + self.domain_tree.uncoupled().len(),
        );
        sectors.extend(self.codomain_tree.uncoupled().iter().copied());
        sectors.extend(
            self.domain_tree
                .uncoupled()
                .iter()
                .copied()
                .map(|sector| rule.dual(sector)),
        );
        sectors
    }

    pub fn external_sector<R>(&self, rule: &R, axis: usize) -> Option<SectorId>
    where
        R: FusionRule,
    {
        let codomain_len = self.codomain_tree.uncoupled().len();
        if axis < codomain_len {
            self.codomain_tree.uncoupled().get(axis).copied()
        } else {
            self.domain_tree
                .uncoupled()
                .get(axis.checked_sub(codomain_len)?)
                .copied()
                .map(|sector| rule.dual(sector))
        }
    }

    pub fn external_is_dual(&self) -> Vec<bool> {
        let mut is_dual = Vec::with_capacity(
            self.codomain_tree.is_dual().len() + self.domain_tree.is_dual().len(),
        );
        is_dual.extend(self.codomain_tree.is_dual().iter().copied());
        is_dual.extend(self.domain_tree.is_dual().iter().copied());
        is_dual
    }

    pub fn group_key(&self) -> FusionTreeGroupKey {
        FusionTreeGroupKey::new(
            self.codomain_tree.uncoupled().iter().copied(),
            self.domain_tree.uncoupled().iter().copied(),
            self.codomain_tree.is_dual().iter().copied(),
            self.domain_tree.is_dual().iter().copied(),
        )
    }

    fn compact_id(&self) -> Option<usize> {
        if self.domain_tree.uncoupled().is_empty()
            && self.domain_tree.coupled().is_none()
            && self.domain_tree.innerlines().is_empty()
            && self.domain_tree.vertices().is_empty()
        {
            self.codomain_tree.compact_id()?.checked_add(1)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum BlockKey {
    Dense,
    FusionTree(FusionTreeBlockKey),
}

impl BlockKey {
    pub fn trivial() -> Self {
        Self::Dense
    }

    pub fn sectors<I>(sectors: I) -> Self
    where
        I: IntoIterator<Item = SectorId>,
    {
        Self::FusionTree(FusionTreeBlockKey::from_uncoupled(sectors))
    }

    pub fn sector_ids<I>(sector_ids: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        Self::sectors(sector_ids.into_iter().map(SectorId::new))
    }

    pub fn ordinal(index: usize) -> Self {
        Self::sector_ids([index])
    }

    fn compact_id(&self) -> Option<usize> {
        match self {
            Self::Dense => Some(0),
            Self::FusionTree(tree) => tree.compact_id(),
        }
    }

    pub fn fusion_tree_group_key(&self) -> Option<FusionTreeGroupKey> {
        match self {
            Self::Dense => None,
            Self::FusionTree(tree) => Some(tree.group_key()),
        }
    }
}

impl From<FusionTreeBlockKey> for BlockKey {
    fn from(value: FusionTreeBlockKey) -> Self {
        Self::FusionTree(value)
    }
}

impl<const N: usize> From<[SectorId; N]> for BlockKey {
    fn from(value: [SectorId; N]) -> Self {
        Self::sectors(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockSpec {
    key: BlockKey,
    shape: DimVec,
    strides: DimVec,
    offset: usize,
}

impl BlockSpec {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Result<Self, CoreError> {
        Self::with_key(BlockKey::trivial(), shape, strides, offset)
    }

    pub fn with_key(
        key: BlockKey,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
    ) -> Result<Self, CoreError> {
        if shape.len() != strides.len() {
            return Err(CoreError::RankMismatch {
                shape: shape.len(),
                strides: strides.len(),
            });
        }
        storage_end_exclusive(&shape, &strides, offset)?;
        Ok(Self {
            key,
            shape: shape.into_iter().collect(),
            strides: strides.into_iter().collect(),
            offset,
        })
    }

    pub fn column_major(shape: Vec<usize>, offset: usize) -> Result<Self, CoreError> {
        Self::column_major_with_key(BlockKey::trivial(), shape, offset)
    }

    pub fn column_major_with_key(
        key: BlockKey,
        shape: Vec<usize>,
        offset: usize,
    ) -> Result<Self, CoreError> {
        let strides = column_major_strides(&shape)?;
        Self::with_key(key, shape, strides, offset)
    }

    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn element_count(&self) -> Result<usize, CoreError> {
        checked_product(&self.shape)
    }

    pub fn storage_end_exclusive(&self) -> Result<usize, CoreError> {
        storage_end_exclusive(&self.shape, &self.strides, self.offset)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectorBlock {
    key: BlockKey,
}

impl SectorBlock {
    pub fn new(key: BlockKey) -> Self {
        Self { key }
    }

    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionTreeBlockGroup {
    group_key: FusionTreeGroupKey,
    block_indices: DimVec,
}

impl FusionTreeBlockGroup {
    pub fn new(group_key: FusionTreeGroupKey, block_indices: Vec<usize>) -> Self {
        Self {
            group_key,
            block_indices: block_indices.into_iter().collect(),
        }
    }

    #[inline]
    pub fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }

    #[inline]
    pub fn block_indices(&self) -> &[usize] {
        &self.block_indices
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectorStructure {
    rank: usize,
    blocks: Vec<SectorBlock>,
    sorted_indices: DimVec,
    compact_lookup: Option<CompactBlockLookup>,
}

impl SectorStructure {
    pub fn dense(rank: usize) -> Self {
        Self::from_keys(rank, [BlockKey::trivial()]).expect("dense sector key is unique")
    }

    pub fn empty(rank: usize) -> Self {
        Self {
            rank,
            blocks: Vec::new(),
            sorted_indices: DimVec::new(),
            compact_lookup: None,
        }
    }

    pub fn from_keys<I, K>(rank: usize, keys: I) -> Result<Self, CoreError>
    where
        I: IntoIterator<Item = K>,
        K: Into<BlockKey>,
    {
        let mut blocks = Vec::new();
        for key in keys {
            let key = key.into();
            blocks.push(SectorBlock::new(key));
        }
        let mut sorted_indices = (0..blocks.len()).collect::<DimVec>();
        sorted_indices
            .sort_unstable_by(|&left, &right| blocks[left].key().cmp(blocks[right].key()));
        for pair in sorted_indices.windows(2) {
            let left = blocks[pair[0]].key();
            let right = blocks[pair[1]].key();
            if left == right {
                return Err(CoreError::DuplicateBlockKey {
                    key: Box::new(left.clone()),
                });
            }
        }
        let compact_lookup = CompactBlockLookup::from_blocks(&blocks);
        Ok(Self {
            rank,
            blocks,
            sorted_indices,
            compact_lookup,
        })
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    #[inline]
    pub fn blocks(&self) -> &[SectorBlock] {
        &self.blocks
    }

    pub fn fusion_tree_groups(&self) -> Vec<FusionTreeBlockGroup> {
        let mut groups = Vec::<FusionTreeBlockGroup>::new();
        let mut group_indices = FxHashMap::<FusionTreeGroupKey, usize>::default();
        for (index, block) in self.blocks.iter().enumerate() {
            let Some(group_key) = block.key().fusion_tree_group_key() else {
                continue;
            };
            if let Some(&group_index) = group_indices.get(&group_key) {
                groups[group_index].block_indices.push(index);
            } else {
                group_indices.insert(group_key.clone(), groups.len());
                groups.push(FusionTreeBlockGroup::new(group_key, vec![index]));
            }
        }
        groups
    }

    pub fn block(&self, index: usize) -> Result<&SectorBlock, CoreError> {
        self.blocks
            .get(index)
            .ok_or(CoreError::BlockIndexOutOfBounds {
                index,
                count: self.blocks.len(),
            })
    }

    pub fn key(&self, index: usize) -> Result<&BlockKey, CoreError> {
        Ok(self.block(index)?.key())
    }

    pub fn find_index(&self, key: &BlockKey) -> Option<usize> {
        if let (Some(lookup), Some(id)) = (&self.compact_lookup, key.compact_id()) {
            if let Some(index) = lookup.get(id) {
                return Some(index);
            }
        }
        self.sorted_indices
            .binary_search_by(|&index| self.blocks[index].key().cmp(key))
            .ok()
            .map(|position| self.sorted_indices[position])
    }

    pub fn find_fusion_tree_index(&self, key: &FusionTreeBlockKey) -> Option<usize> {
        if let (Some(lookup), Some(id)) = (&self.compact_lookup, key.compact_id()) {
            if let Some(index) = lookup.get(id) {
                return Some(index);
            }
        }
        self.sorted_indices
            .binary_search_by(|&index| match self.blocks[index].key() {
                BlockKey::Dense => std::cmp::Ordering::Less,
                BlockKey::FusionTree(tree) => tree.cmp(key),
            })
            .ok()
            .map(|position| self.sorted_indices[position])
    }

    #[inline]
    pub fn has_compact_lookup(&self) -> bool {
        self.compact_lookup.is_some()
    }

    #[inline]
    pub fn sorted_indices(&self) -> &[usize] {
        &self.sorted_indices
    }

    pub fn pair_indices_from(&self, src: &Self) -> Result<Vec<usize>, CoreError> {
        if self.block_count() != src.block_count() {
            return Err(CoreError::BlockCountMismatch {
                expected: self.block_count(),
                actual: src.block_count(),
            });
        }
        if let Some(src_lookup) = &src.compact_lookup {
            if self
                .blocks
                .iter()
                .all(|block| block.key().compact_id().is_some())
            {
                return self
                    .blocks
                    .iter()
                    .map(|block| {
                        let id = block.key().compact_id().expect("checked above");
                        src_lookup
                            .get(id)
                            .ok_or_else(|| CoreError::MissingBlockKey {
                                key: Box::new(block.key().clone()),
                            })
                    })
                    .collect();
            }
        }
        self.pair_indices_from_sorted(src)
    }

    fn pair_indices_from_sorted(&self, src: &Self) -> Result<Vec<usize>, CoreError> {
        let mut src_for_dst = vec![usize::MAX; self.block_count()];
        let mut dst_pos = 0usize;
        let mut src_pos = 0usize;
        while dst_pos < self.sorted_indices.len() && src_pos < src.sorted_indices.len() {
            let dst_index = self.sorted_indices[dst_pos];
            let src_index = src.sorted_indices[src_pos];
            let dst_key = self.blocks[dst_index].key();
            let src_key = src.blocks[src_index].key();
            match dst_key.cmp(src_key) {
                std::cmp::Ordering::Less => {
                    return Err(CoreError::MissingBlockKey {
                        key: Box::new(dst_key.clone()),
                    });
                }
                std::cmp::Ordering::Greater => {
                    return Err(CoreError::MissingBlockKey {
                        key: Box::new(src_key.clone()),
                    });
                }
                std::cmp::Ordering::Equal => {
                    src_for_dst[dst_index] = src_index;
                    dst_pos += 1;
                    src_pos += 1;
                }
            }
        }
        if dst_pos < self.sorted_indices.len() {
            let dst_index = self.sorted_indices[dst_pos];
            return Err(CoreError::MissingBlockKey {
                key: Box::new(self.blocks[dst_index].key().clone()),
            });
        }
        if src_pos < src.sorted_indices.len() {
            let src_index = src.sorted_indices[src_pos];
            return Err(CoreError::MissingBlockKey {
                key: Box::new(src.blocks[src_index].key().clone()),
            });
        }
        Ok(src_for_dst)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompactBlockLookup {
    indices: DimVec,
}

impl CompactBlockLookup {
    const MISSING: usize = usize::MAX;

    fn from_blocks(blocks: &[SectorBlock]) -> Option<Self> {
        let mut max_id = 0usize;
        let mut ids = Vec::with_capacity(blocks.len());
        for block in blocks {
            let id = block.key().compact_id()?;
            max_id = max_id.max(id);
            ids.push(id);
        }
        let len = max_id.checked_add(1)?;
        if len > blocks.len().saturating_mul(4).max(1) {
            return None;
        }
        let mut indices = DimVec::from_elem(Self::MISSING, len);
        for (index, id) in ids.into_iter().enumerate() {
            if indices[id] != Self::MISSING {
                return None;
            }
            indices[id] = index;
        }
        Some(Self { indices })
    }

    fn get(&self, id: usize) -> Option<usize> {
        self.indices
            .get(id)
            .copied()
            .filter(|&index| index != Self::MISSING)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DegeneracyBlock {
    shape: DimVec,
    strides: DimVec,
    offset: usize,
}

impl DegeneracyBlock {
    pub fn new(shape: DimVec, strides: DimVec, offset: usize) -> Result<Self, CoreError> {
        if shape.len() != strides.len() {
            return Err(CoreError::RankMismatch {
                shape: shape.len(),
                strides: strides.len(),
            });
        }
        storage_end_exclusive(&shape, &strides, offset)?;
        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    pub fn column_major(shape: Vec<usize>, offset: usize) -> Result<Self, CoreError> {
        let strides = column_major_strides(&shape)?;
        Self::new(
            shape.into_iter().collect(),
            strides.into_iter().collect(),
            offset,
        )
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn element_count(&self) -> Result<usize, CoreError> {
        checked_product(&self.shape)
    }

    pub fn storage_end_exclusive(&self) -> Result<usize, CoreError> {
        storage_end_exclusive(&self.shape, &self.strides, self.offset)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DegeneracyStructure {
    rank: usize,
    blocks: Vec<DegeneracyBlock>,
}

impl DegeneracyStructure {
    pub fn packed_column_major<I>(rank: usize, shapes: I) -> Result<Self, CoreError>
    where
        I: IntoIterator,
        I::Item: Into<Vec<usize>>,
    {
        let mut offset = 0usize;
        let mut blocks = Vec::new();
        for shape in shapes {
            let shape = shape.into();
            if shape.len() != rank {
                return Err(CoreError::StructureRankMismatch {
                    expected: rank,
                    actual: shape.len(),
                });
            }
            let block = DegeneracyBlock::column_major(shape, offset)?;
            offset = block.storage_end_exclusive()?;
            blocks.push(block);
        }
        Self::from_blocks_with_rank(rank, blocks)
    }

    pub fn from_blocks_with_rank(
        rank: usize,
        blocks: Vec<DegeneracyBlock>,
    ) -> Result<Self, CoreError> {
        for block in &blocks {
            if block.shape().len() != rank {
                return Err(CoreError::StructureRankMismatch {
                    expected: rank,
                    actual: block.shape().len(),
                });
            }
            block.storage_end_exclusive()?;
        }
        Ok(Self { rank, blocks })
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    #[inline]
    pub fn blocks(&self) -> &[DegeneracyBlock] {
        &self.blocks
    }

    pub fn block(&self, index: usize) -> Result<&DegeneracyBlock, CoreError> {
        self.blocks
            .get(index)
            .ok_or(CoreError::BlockIndexOutOfBounds {
                index,
                count: self.blocks.len(),
            })
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.blocks.iter().try_fold(0usize, |required, block| {
            Ok(required.max(block.storage_end_exclusive()?))
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockRef<'a> {
    key: &'a BlockKey,
    degeneracy: &'a DegeneracyBlock,
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
    tree: FusionTreeKey,
    offset: usize,
    shape: DimVec,
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
        checked_element_count(&self.shape)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Checked contiguous column-major storage region for one coupled sector.
pub struct CoupledSectorRegion {
    coupled: Option<SectorId>,
    rows: usize,
    cols: usize,
    range: core::ops::Range<usize>,
    row_trees: Vec<CoupledTreeExtent>,
    col_trees: Vec<CoupledTreeExtent>,
}

impl CoupledSectorRegion {
    /// Coupled-sector label, or `None` for a vacuum-coupled degenerate tree.
    pub fn coupled(&self) -> Option<SectorId> {
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
        &self.row_trees
    }

    /// Domain trees with their column offsets and degeneracy shapes.
    pub fn col_trees(&self) -> &[CoupledTreeExtent] {
        &self.col_trees
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct BlockStructureContentBlock {
    key: BlockKey,
    shape: DimVec,
    strides: DimVec,
    offset: usize,
}

impl BlockStructureContentBlock {
    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }
}

type CoupledRegionResult = Result<Option<Arc<[CoupledSectorRegion]>>, CoreError>;
type CoupledRegionCache = Arc<[OnceLock<CoupledRegionResult>]>;

#[derive(Clone, Eq)]
pub struct BlockStructureContent {
    id: usize,
    rank: usize,
    blocks: Arc<[BlockStructureContentBlock]>,
    coupled_region_cache: OnceLock<CoupledRegionCache>,
}

impl core::fmt::Debug for BlockStructureContent {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("BlockStructureContent")
            .field("id", &self.id)
            .field("rank", &self.rank)
            .field("blocks", &self.blocks)
            .finish()
    }
}

// Content equality deliberately ignores `id`: the id is a process-local
// intern handle (monotonic since the LRU-cap change, never reused across
// eviction or reset), not part of the content. Including it in the derived
// PartialEq made content-equal structures interned in different reset
// epochs compare unequal, which broke replay's content-fallback validation
// (caught by reset_and_concurrent_rebuild_keep_structure_semantics in CI).
// Id-keyed caches are unaffected: they key on `id()` explicitly and rely on
// monotonicity, not on equality of the full content struct.
impl PartialEq for BlockStructureContent {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.blocks == other.blocks
    }
}

impl BlockStructureContent {
    /// Process-local intern id (insertion-order counter into the block-structure
    /// intern table). Content-stable within one process — identical content
    /// interns to one `Arc` and thus one id — so it is a sound O(1) cache key.
    /// It is NOT semantically stable across processes or runs and must never be
    /// persisted; a loaded cache reconstructs identity from the block content.
    #[inline]
    pub fn id(&self) -> usize {
        self.id
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn blocks(&self) -> &[BlockStructureContentBlock] {
        &self.blocks
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct BlockStructureInternKey {
    rank: usize,
    blocks: Arc<[BlockStructureContentBlock]>,
}

type BlockStructureInternTable =
    lru::LruCache<BlockStructureInternKey, Arc<BlockStructureContent>, rustc_hash::FxBuildHasher>;

/// LRU cap for the block-structure content intern table (and, reusing the same
/// bound, the arc dedup and coupled-subblock caches). Mirrors
/// `HOM_SPACE_INTERN_CAP`: a long-lived / multi-tenant process can otherwise
/// grow these tables without bound over a χ sweep. See
/// `BLOCK_STRUCTURE_CONTENT_ID` for why capping this particular table is
/// aliasing-safe despite its ids being consumed as cache keys downstream.
const BLOCK_STRUCTURE_INTERN_CAP: usize = 8192;

fn block_structure_intern_table() -> &'static RwLock<BlockStructureInternTable> {
    static TABLE: OnceLock<RwLock<BlockStructureInternTable>> = OnceLock::new();
    TABLE.get_or_init(|| {
        RwLock::new(lru::LruCache::with_hasher(
            std::num::NonZeroUsize::new(BLOCK_STRUCTURE_INTERN_CAP).unwrap(),
            rustc_hash::FxBuildHasher,
        ))
    })
}

/// Process-global, strictly-monotonic id source for interned block-structure
/// content.
///
/// Why-not (`id = table.len() + 1`): the intern table is now LRU-capped (above),
/// so its size is no longer monotonic — `len() + 1` would re-issue an id to
/// DIFFERENT content after an eviction. `BlockStructureCacheKey` (tenet-tensors)
/// keys the tree-transform and contract structure caches *purely* by this id
/// (both `Hash` and `Eq` read only `content.id()`), so a recycled id would
/// silently alias two distinct structures and hand back the wrong cached kernel
/// — an aliasing-class correctness bug. A monotonic counter never reuses an id:
/// not across LRU eviction, and not across `reset_core_intern_tables` (the
/// counter is deliberately NOT reset there). Consequence — a stale tensors-layer
/// entry keyed by an old id can only ever be re-hit by the *same* content `Arc`
/// that minted that id; content re-interned after eviction/reset receives a
/// fresh, higher id and simply misses (recompute), never aliases. A 64-bit
/// counter cannot realistically overflow.
static BLOCK_STRUCTURE_CONTENT_ID: AtomicUsize = AtomicUsize::new(1);

#[cfg(test)]
std::thread_local! {
    static BLOCK_STRUCTURE_INTERN_CALLS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

#[cfg(test)]
pub(crate) fn reset_block_structure_intern_calls() {
    BLOCK_STRUCTURE_INTERN_CALLS.set(0);
}

#[cfg(test)]
pub(crate) fn block_structure_intern_calls() -> usize {
    BLOCK_STRUCTURE_INTERN_CALLS.get()
}

fn intern_block_structure_content(
    sector: &SectorStructure,
    degeneracy: &DegeneracyStructure,
) -> Arc<BlockStructureContent> {
    #[cfg(test)]
    BLOCK_STRUCTURE_INTERN_CALLS.set(BLOCK_STRUCTURE_INTERN_CALLS.get() + 1);
    let mut blocks = Vec::with_capacity(sector.block_count());
    for index in 0..sector.block_count() {
        let sector_key = sector
            .key(index)
            .expect("validated block structure sector index");
        let block = degeneracy
            .block(index)
            .expect("validated block structure degeneracy index");
        blocks.push(BlockStructureContentBlock {
            key: sector_key.clone(),
            shape: block.shape().iter().copied().collect(),
            strides: block.strides().iter().copied().collect(),
            offset: block.offset(),
        });
    }

    let blocks = Arc::<[BlockStructureContentBlock]>::from(blocks);
    let key = BlockStructureInternKey {
        rank: sector.rank(),
        blocks: Arc::clone(&blocks),
    };
    let table = block_structure_intern_table();
    // Read-lock fast path uses `peek` (does not bump recency; `get` needs `&mut`).
    if let Ok(read) = table.read() {
        if let Some(content) = read.peek(&key) {
            return Arc::clone(content);
        }
    }

    let mut write = table
        .write()
        .expect("block structure intern table poisoned");
    if let Some(content) = write.get(&key) {
        return Arc::clone(content);
    }
    let content = Arc::new(BlockStructureContent {
        id: BLOCK_STRUCTURE_CONTENT_ID.fetch_add(1, Ordering::Relaxed),
        rank: sector.rank(),
        blocks,
        coupled_region_cache: OnceLock::new(),
    });
    write.put(key, Arc::clone(&content));
    content
}

type BlockStructureArcTable = lru::LruCache<usize, Weak<BlockStructure>, rustc_hash::FxBuildHasher>;

fn block_structure_arc_table() -> &'static RwLock<BlockStructureArcTable> {
    static TABLE: OnceLock<RwLock<BlockStructureArcTable>> = OnceLock::new();
    TABLE.get_or_init(|| {
        RwLock::new(lru::LruCache::with_hasher(
            std::num::NonZeroUsize::new(BLOCK_STRUCTURE_INTERN_CAP).unwrap(),
            rustc_hash::FxBuildHasher,
        ))
    })
}

fn canonicalize_block_structure_arc(structure: Arc<BlockStructure>) -> Arc<BlockStructure> {
    let id = structure.content_id();
    let table = block_structure_arc_table();
    // Read-lock fast path uses `peek` (does not bump recency; `get` needs `&mut`).
    if let Ok(read) = table.read() {
        if let Some(existing) = read.peek(&id).and_then(Weak::upgrade) {
            return existing;
        }
    }

    let mut write = table.write().expect("block structure arc table poisoned");
    if let Some(existing) = write.get(&id).and_then(Weak::upgrade) {
        return existing;
    }
    write.put(id, Arc::downgrade(&structure));
    structure
}

/// Clears the bounded tenet-core intern tables — lazy hom-space identities,
/// block-structure content, block-structure `Arc` dedup, fusion-tree layouts,
/// and coupled subblock structures. Chained from tenet-tensors'
/// `reset_global_operation_caches` so a long-lived / multi-tenant process can
/// release the tables between workloads.
///
/// Why-safe (id coherence): block-structure content ids come from
/// `BLOCK_STRUCTURE_CONTENT_ID`, a monotonic counter deliberately NOT reset here.
/// Ids are therefore never reused after a reset, so a tensors-layer cache entry
/// keyed by an old content id can only be re-hit by the same content `Arc` that
/// minted it; content re-interned after this reset gets a fresh id and misses
/// cleanly. Reset is thus safe to call on its own — no "all layers at once" API
pub fn reset_core_intern_tables() {
    reset_hom_space_intern_table();
    if let Ok(mut table) = block_structure_intern_table().write() {
        table.clear();
    }
    if let Ok(mut table) = block_structure_arc_table().write() {
        table.clear();
    }
    reset_fusion_tree_layout_caches();
    crate::su2_exact::reset_publication_cache();
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockStructure {
    sector: SectorStructure,
    degeneracy: DegeneracyStructure,
    content: Arc<BlockStructureContent>,
    // Cached at construction; replay validation checks this against storage
    // lengths on every call and must not re-scan all blocks.
    required_len: usize,
}

#[doc(hidden)]
pub struct ValidatedFusionTreeBlockStructure<'rule, 'structure, R> {
    rule: &'rule R,
    structure: &'structure BlockStructure,
}

#[doc(hidden)]
impl<'rule, 'structure, R> ValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: FusionRule,
{
    pub fn try_new(
        rule: &'rule R,
        structure: &'structure BlockStructure,
    ) -> Result<Self, CoreError> {
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            if let BlockKey::FusionTree(key) = block.key() {
                key.validate_for_rule(rule)?;
            }
        }
        Ok(Self { rule, structure })
    }

    #[inline]
    pub fn rule(&self) -> &'rule R {
        self.rule
    }

    #[inline]
    pub fn structure(&self) -> &'structure BlockStructure {
        self.structure
    }

    pub fn fusion_tree_block_key(
        &self,
        index: usize,
    ) -> Result<Option<&'structure FusionTreeBlockKey>, CoreError> {
        Ok(match self.structure.block(index)?.key() {
            BlockKey::Dense => None,
            BlockKey::FusionTree(key) => Some(key),
        })
    }
}

#[doc(hidden)]
impl<'rule, 'structure, R> ValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    pub fn braid_tree_pair_rows_for_block_indices(
        &self,
        block_indices: &[usize],
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        codomain_levels: &[usize],
        domain_levels: &[usize],
    ) -> Result<
        Vec<(
            FusionTreeBlockKey,
            Vec<(FusionTreeBlockKey, R::Scalar)>,
        )>,
        CoreError,
    > {
        let Some(first_index) = block_indices.first().copied() else {
            return Ok(Vec::new());
        };
        let first = self.required_fusion_tree_block_key(first_index)?;
        let prepared = PreparedTreePairOperation::prepare_braid(
            self.rule,
            first.codomain_tree().uncoupled().len(),
            first.domain_tree().uncoupled().len(),
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        )?;
        self.execute_tree_pair_block_indices(block_indices, prepared, false)
    }

    pub fn permute_tree_pair_rows_for_block_indices(
        &self,
        block_indices: &[usize],
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<
        Vec<(
            FusionTreeBlockKey,
            Vec<(FusionTreeBlockKey, R::Scalar)>,
        )>,
        CoreError,
    > {
        if !self.rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: self.rule.braiding_style(),
            });
        }
        let Some(first_index) = block_indices.first().copied() else {
            return Ok(Vec::new());
        };
        let first = self.required_fusion_tree_block_key(first_index)?;
        let prepared = PreparedTreePairOperation::prepare_permute(
            self.rule,
            first.codomain_tree().uncoupled().len(),
            first.domain_tree().uncoupled().len(),
            codomain_permutation,
            domain_permutation,
        )?;
        self.execute_tree_pair_block_indices(block_indices, prepared, false)
    }

    pub fn transpose_tree_pair_rows_for_block_indices(
        &self,
        block_indices: &[usize],
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<
        Vec<(
            FusionTreeBlockKey,
            Vec<(FusionTreeBlockKey, R::Scalar)>,
        )>,
        CoreError,
    > {
        let Some(first_index) = block_indices.first().copied() else {
            return Ok(Vec::new());
        };
        let first = self.required_fusion_tree_block_key(first_index)?;
        let prepared = PreparedTreePairOperation::prepare_transpose(
            first.codomain_tree().uncoupled().len(),
            first.domain_tree().uncoupled().len(),
            codomain_permutation,
            domain_permutation,
        )?;
        self.execute_tree_pair_block_indices(block_indices, prepared, true)
    }

    fn execute_tree_pair_block_indices(
        &self,
        block_indices: &[usize],
        prepared: PreparedTreePairOperation,
        transpose: bool,
    ) -> Result<
        Vec<(
            FusionTreeBlockKey,
            Vec<(FusionTreeBlockKey, R::Scalar)>,
        )>,
        CoreError,
    > {
        let mut source_keys = Vec::with_capacity(block_indices.len());
        for &index in block_indices {
            source_keys.push(self.required_fusion_tree_block_key(index)?.clone());
        }
        let group = tree_pair_block_group_from_validated_structure(self.rule, &source_keys)?
            .expect("nonempty block indices produce a validated group");
        let rows = if transpose {
            multiplicity_free_transpose_tree_pair_block_validated(group, prepared)?
        } else {
            multiplicity_free_braid_tree_pair_block_validated(group, prepared)?
        };
        Ok(source_keys.into_iter().zip(rows).collect())
    }
}

#[doc(hidden)]
impl<'rule, 'structure, R> ValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    pub fn braid_codomain_rows_for_block_indices(
        &self,
        block_indices: &[usize],
        permutation: &[usize],
        levels: &[usize],
    ) -> Result<Vec<(FusionTreeKey, Vec<(FusionTreeKey, R::Scalar)>)>, CoreError> {
        let Some(first_index) = block_indices.first().copied() else {
            return Ok(Vec::new());
        };
        let first = self.required_fusion_tree_block_key(first_index)?;
        let rank = first.codomain_tree().uncoupled().len();
        if levels.len() != rank {
            return Err(CoreError::DimensionMismatch {
                expected: rank,
                actual: levels.len(),
            });
        }
        let prepared = PreparedTreeBraid::new(permutation, levels, rank)?;
        self.execute_codomain_block_indices(block_indices, prepared)
    }

    pub fn permute_codomain_rows_for_block_indices(
        &self,
        block_indices: &[usize],
        permutation: &[usize],
    ) -> Result<Vec<(FusionTreeKey, Vec<(FusionTreeKey, R::Scalar)>)>, CoreError> {
        if !self.rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: self.rule.braiding_style(),
            });
        }
        let Some(first_index) = block_indices.first().copied() else {
            return Ok(Vec::new());
        };
        let first = self.required_fusion_tree_block_key(first_index)?;
        let rank = first.codomain_tree().uncoupled().len();
        let levels = (0..rank).collect::<SmallVec<[usize; 8]>>();
        let prepared = PreparedTreeBraid::new(permutation, &levels, rank)?;
        self.execute_codomain_block_indices(block_indices, prepared)
    }

    fn execute_codomain_block_indices(
        &self,
        block_indices: &[usize],
        prepared: PreparedTreeBraid,
    ) -> Result<Vec<(FusionTreeKey, Vec<(FusionTreeKey, R::Scalar)>)>, CoreError> {
        let mut source_keys = Vec::with_capacity(block_indices.len());
        for &index in block_indices {
            source_keys.push(
                self.required_fusion_tree_block_key(index)?
                    .codomain_tree()
                    .clone(),
            );
        }
        let group = fusion_tree_block_group_from_validated_structure(self.rule, &source_keys)?
            .expect("nonempty block indices produce a validated group");
        let rows = multiplicity_free_braid_tree_block_validated(group, prepared)?;
        Ok(source_keys.into_iter().zip(rows).collect())
    }
}

impl<'rule, 'structure, R> ValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: FusionRule,
{
    fn required_fusion_tree_block_key(
        &self,
        index: usize,
    ) -> Result<&'structure FusionTreeBlockKey, CoreError> {
        self.fusion_tree_block_key(index)?
            .ok_or(CoreError::MalformedFusionTree {
                message: "validated fusion-tree group contains a dense block",
            })
    }
}

impl BlockStructure {
    pub fn trivial(shape: &[usize]) -> Result<Self, CoreError> {
        Self::from_parts(
            SectorStructure::dense(shape.len()),
            DegeneracyStructure::packed_column_major(shape.len(), [shape.to_vec()])?,
        )
    }

    pub fn empty(rank: usize) -> Self {
        let sector = SectorStructure::empty(rank);
        let degeneracy = DegeneracyStructure {
            rank,
            blocks: Vec::new(),
        };
        let content = intern_block_structure_content(&sector, &degeneracy);
        Self {
            sector,
            degeneracy,
            content,
            required_len: 0,
        }
    }

    pub fn from_blocks(blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        let rank = blocks.first().map(|block| block.shape().len()).unwrap_or(0);
        Self::from_blocks_with_rank(rank, blocks)
    }

    pub fn from_blocks_with_rank(rank: usize, blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        let keys = blocks
            .iter()
            .map(|block| block.key().clone())
            .collect::<Vec<_>>();
        let degeneracy_blocks = blocks
            .into_iter()
            .map(|block| DegeneracyBlock::new(block.shape, block.strides, block.offset))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_parts(
            SectorStructure::from_keys(rank, keys)?,
            DegeneracyStructure::from_blocks_with_rank(rank, degeneracy_blocks)?,
        )
    }

    pub fn from_parts(
        sector: SectorStructure,
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
        let content = intern_block_structure_content(&sector, &degeneracy);
        Ok(Self {
            sector,
            degeneracy,
            content,
            required_len,
        })
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
    /// Blocks are stable-sorted by coupled sector, then each coupled sector is
    /// laid out as one contiguous column-major matrix with the fusion-tree
    /// subblocks as strided views (see
    /// [`FusionTensorMapSpace::from_degeneracy_shapes_coupled`]). Fails when a
    /// coupled sector does not cover its full codomain-tree x domain-tree
    /// grid, because the sector matrix would contain uninitialized holes.
    pub fn coupled_sector_matrix_with_keys<R>(
        rule: &R,
        nout: usize,
        rank: usize,
        blocks: Vec<(FusionTreeBlockKey, Vec<usize>)>,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        let mut blocks = blocks;
        blocks.sort_by_key(|(key, _)| coupled_or_vacuum(rule, key.codomain_tree()).id());
        let (keys, shapes): (Vec<_>, Vec<_>) = blocks.into_iter().unzip();
        let specs = coupled_sector_matrix_block_specs(rule, nout, rank, &keys, &shapes)?;
        Self::from_blocks_with_rank(rank, specs)
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.sector.rank()
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.sector.block_count()
    }

    #[inline]
    pub fn sector_structure(&self) -> &SectorStructure {
        &self.sector
    }

    #[inline]
    pub fn degeneracy_structure(&self) -> &DegeneracyStructure {
        &self.degeneracy
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
        self.sector.fusion_tree_groups()
    }

    pub fn find_block_index_by_key(&self, key: &BlockKey) -> Option<usize> {
        self.sector.find_index(key)
    }

    pub fn find_block_index_by_fusion_tree_key(&self, key: &FusionTreeBlockKey) -> Option<usize> {
        self.sector.find_fusion_tree_index(key)
    }

    pub fn pair_block_indices_from(&self, src: &BlockStructure) -> Result<Vec<usize>, CoreError> {
        self.sector.pair_indices_from(&src.sector)
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
            key: self.sector.key(index)?,
            degeneracy: self.degeneracy.block(index)?,
        })
    }

    pub fn block_by_key(&self, key: &BlockKey) -> Result<BlockRef<'_>, CoreError> {
        let index = self
            .find_block_index_by_key(key)
            .ok_or_else(|| CoreError::MissingBlockKey {
                key: Box::new(key.clone()),
            })?;
        self.block(index)
    }

    pub fn fusion_tree_block(&self, key: &FusionTreeBlockKey) -> Result<BlockRef<'_>, CoreError> {
        let index = self
            .find_block_index_by_fusion_tree_key(key)
            .ok_or_else(|| CoreError::MissingBlockKey {
                key: Box::new(BlockKey::FusionTree(key.clone())),
            })?;
        self.block(index)
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        Ok(self.required_len)
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
        self.content
            .coupled_region_cache
            .get_or_init(|| new_coupled_region_cache(self.rank()))[nout]
            .get_or_init(|| {
                compile_coupled_sector_regions(self, nout)
                    .map(|regions| regions.map(Arc::from))
            })
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn coupled_region_cache_is_initialized(&self) -> bool {
        self.content.coupled_region_cache.get().is_some()
    }
}

fn new_coupled_region_cache(rank: usize) -> CoupledRegionCache {
    (0..=rank)
        .map(|_| OnceLock::new())
        .collect::<Vec<_>>()
        .into()
}

fn compile_coupled_sector_regions(
    structure: &BlockStructure,
    nout: usize,
) -> Result<Option<Vec<CoupledSectorRegion>>, CoreError> {
        let mut regions = Vec::new();
        let mut seen_coupled = FxHashMap::<Option<SectorId>, ()>::default();
        let mut block_index = 0usize;
        let mut next_offset = 0usize;
        while block_index < structure.block_count() {
            let first = structure.block(block_index)?;
            let BlockKey::FusionTree(first_key) = first.key() else {
                return Ok(None);
            };
            let coupled = first_key.codomain_tree().coupled();
            if first_key.domain_tree().coupled() != coupled
                || seen_coupled.insert(coupled, ()).is_some()
            {
                return Ok(None);
            }

            let mut row_trees = Vec::<CoupledTreeExtent>::new();
            let mut col_trees = Vec::<CoupledTreeExtent>::new();
            let mut end = block_index;
            while end < structure.block_count() {
                let block = structure.block(end)?;
                let BlockKey::FusionTree(key) = block.key() else {
                    return Ok(None);
                };
                if key.codomain_tree().coupled() != coupled {
                    break;
                }
                if key.domain_tree().coupled() != coupled {
                    return Ok(None);
                }
                let row_shape: DimVec = block.shape()[..nout].iter().copied().collect();
                let col_shape: DimVec = block.shape()[nout..].iter().copied().collect();
                if !insert_coupled_tree_extent(
                    &mut row_trees,
                    key.codomain_tree(),
                    row_shape,
                )? || !insert_coupled_tree_extent(
                    &mut col_trees,
                    key.domain_tree(),
                    col_shape,
                )? {
                    return Ok(None);
                }
                end += 1;
            }

            let rows = coupled_tree_total(&row_trees)?;
            let cols = coupled_tree_total(&col_trees)?;
            let expected_blocks = row_trees
                .len()
                .checked_mul(col_trees.len())
                .ok_or(CoreError::ElementCountOverflow)?;
            if end - block_index != expected_blocks {
                return Ok(None);
            }
            let mut seen_pairs = FxHashMap::<(FusionTreeKey, FusionTreeKey), ()>::default();
            for index in block_index..end {
                let block = structure.block(index)?;
                let BlockKey::FusionTree(key) = block.key() else {
                    unreachable!("fusion-tree keys checked above")
                };
                if seen_pairs
                    .insert(
                        (key.codomain_tree().clone(), key.domain_tree().clone()),
                        (),
                    )
                    .is_some()
                {
                    return Ok(None);
                }
                let row_offset = row_trees
                    .iter()
                    .find(|extent| extent.tree() == key.codomain_tree())
                    .expect("row tree recorded above")
                    .offset();
                let col_offset = col_trees
                    .iter()
                    .find(|extent| extent.tree() == key.domain_tree())
                    .expect("column tree recorded above")
                    .offset();
                let expected_offset = next_offset
                    .checked_add(row_offset)
                    .and_then(|offset| {
                        rows
                            .checked_mul(col_offset)
                            .and_then(|column| offset.checked_add(column))
                    })
                    .ok_or(CoreError::ElementCountOverflow)?;
                if block.offset() != expected_offset
                    || !coupled_sector_strides(block.shape(), block.strides(), nout, rows)?
                {
                    return Ok(None);
                }
            }
            let elements = rows
                .checked_mul(cols)
                .ok_or(CoreError::ElementCountOverflow)?;
            let end_offset = next_offset
                .checked_add(elements)
                .ok_or(CoreError::ElementCountOverflow)?;
            if end_offset > structure.required_len {
                return Ok(None);
            }
            regions.push(CoupledSectorRegion {
                coupled,
                rows,
                cols,
                range: next_offset..end_offset,
                row_trees,
                col_trees,
            });
            next_offset = end_offset;
            block_index = end;
        }
        if next_offset != structure.required_len {
            return Ok(None);
        }
        Ok(Some(regions))
}

fn insert_coupled_tree_extent(
    trees: &mut Vec<CoupledTreeExtent>,
    tree: &FusionTreeKey,
    shape: DimVec,
) -> Result<bool, CoreError> {
    if let Some(known) = trees.iter().find(|known| known.tree() == tree) {
        return Ok(known.shape() == shape.as_slice());
    }
    let offset = coupled_tree_total(trees)?;
    offset
        .checked_add(checked_element_count(&shape)?)
        .ok_or(CoreError::ElementCountOverflow)?;
    trees.push(CoupledTreeExtent {
        tree: tree.clone(),
        offset,
        shape,
    });
    Ok(true)
}

fn coupled_tree_total(trees: &[CoupledTreeExtent]) -> Result<usize, CoreError> {
    trees.iter().try_fold(0usize, |total, tree| {
        total
            .checked_add(tree.extent()?)
            .ok_or(CoreError::ElementCountOverflow)
    })
}

fn checked_element_count(shape: &[usize]) -> Result<usize, CoreError> {
    shape.iter().try_fold(1usize, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or(CoreError::ElementCountOverflow)
    })
}

fn coupled_sector_strides(
    shape: &[usize],
    strides: &[usize],
    nout: usize,
    rows: usize,
) -> Result<bool, CoreError> {
    if shape.len() != strides.len() || nout > shape.len() {
        return Ok(false);
    }
    let mut expected = 1usize;
    for axis in 0..nout {
        if strides[axis] != expected {
            return Ok(false);
        }
        expected = expected
            .checked_mul(shape[axis])
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    expected = rows;
    for axis in nout..shape.len() {
        if strides[axis] != expected {
            return Ok(false);
        }
        expected = expected
            .checked_mul(shape[axis])
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    Ok(true)
}
