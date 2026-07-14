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
                return Err(CoreError::DuplicateBlockKey { key: left.clone() });
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
                                key: block.key().clone(),
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
                        key: dst_key.clone(),
                    });
                }
                std::cmp::Ordering::Greater => {
                    return Err(CoreError::MissingBlockKey {
                        key: src_key.clone(),
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
                key: self.blocks[dst_index].key().clone(),
            });
        }
        if src_pos < src.sorted_indices.len() {
            let src_index = src.sorted_indices[src_pos];
            return Err(CoreError::MissingBlockKey {
                key: src.blocks[src_index].key().clone(),
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockStructureContent {
    id: usize,
    rank: usize,
    blocks: Arc<[BlockStructureContentBlock]>,
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

fn block_structure_intern_table(
) -> &'static RwLock<FxHashMap<BlockStructureInternKey, Arc<BlockStructureContent>>> {
    static TABLE: OnceLock<RwLock<FxHashMap<BlockStructureInternKey, Arc<BlockStructureContent>>>> =
        OnceLock::new();
    TABLE.get_or_init(|| RwLock::new(FxHashMap::default()))
}

fn intern_block_structure_content(
    sector: &SectorStructure,
    degeneracy: &DegeneracyStructure,
) -> Arc<BlockStructureContent> {
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
    if let Ok(read) = table.read() {
        if let Some(content) = read.get(&key) {
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
        id: write.len() + 1,
        rank: sector.rank(),
        blocks,
    });
    write.insert(key, Arc::clone(&content));
    content
}

fn block_structure_arc_table() -> &'static RwLock<FxHashMap<usize, Weak<BlockStructure>>> {
    static TABLE: OnceLock<RwLock<FxHashMap<usize, Weak<BlockStructure>>>> = OnceLock::new();
    TABLE.get_or_init(|| RwLock::new(FxHashMap::default()))
}

fn canonicalize_block_structure_arc(structure: Arc<BlockStructure>) -> Arc<BlockStructure> {
    let id = structure.content_id();
    let table = block_structure_arc_table();
    if let Ok(read) = table.read() {
        if let Some(existing) = read.get(&id).and_then(Weak::upgrade) {
            return existing;
        }
    }

    let mut write = table.write().expect("block structure arc table poisoned");
    if let Some(existing) = write.get(&id).and_then(Weak::upgrade) {
        return existing;
    }
    write.insert(id, Arc::downgrade(&structure));
    structure
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
            .ok_or_else(|| CoreError::MissingBlockKey { key: key.clone() })?;
        self.block(index)
    }

    pub fn fusion_tree_block(&self, key: &FusionTreeBlockKey) -> Result<BlockRef<'_>, CoreError> {
        let index = self
            .find_block_index_by_fusion_tree_key(key)
            .ok_or_else(|| CoreError::MissingBlockKey {
                key: BlockKey::FusionTree(key.clone()),
            })?;
        self.block(index)
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        Ok(self.required_len)
    }
}
