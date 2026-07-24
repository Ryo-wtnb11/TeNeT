/// Categorical identity of one codomain/domain fusion-tree basis pair.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreePairKey {
    codomain_tree: FusionTreeKey,
    domain_tree: FusionTreeKey,
}

impl FusionTreePairKey {
    /// Combine two raw tree identities without categorical validation.
    ///
    /// This constructor does not check provider membership, individual tree
    /// admissibility, or equality of the coupled sectors. Call
    /// [`Self::validate_for_rule`] before categorical use.
    pub fn pair(codomain_tree: FusionTreeKey, domain_tree: FusionTreeKey) -> Self {
        Self {
            codomain_tree,
            domain_tree,
        }
    }

    /// Construct a raw tree pair from numeric labels.
    ///
    /// Sector arrays and `coupled` contain provider-local sector IDs. The two
    /// vertex arrays instead contain one-based outer-multiplicity labels; a
    /// zero vertex returns [`CoreError::InvalidMultiplicityIndex`].
    ///
    /// # Provider-domain precondition
    ///
    /// Every sector ID must already name a sector in the intended provider. This
    /// constructor cannot check provider membership; call
    /// [`Self::validate_for_rule`] before categorical use. Providers with a
    /// finite table may otherwise panic through their infallible
    /// [`FusionRule`] methods.
    pub fn try_pair_from_sector_ids<
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
        coupled: usize,
        codomain_is_dual: CodomainDual,
        domain_is_dual: DomainDual,
        codomain_innerlines: CodomainInner,
        domain_innerlines: DomainInner,
        codomain_vertices: CodomainVertices,
        domain_vertices: DomainVertices,
    ) -> Result<Self, CoreError>
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
        Ok(Self::pair(
            FusionTreeKey::try_from_sector_ids(
                codomain_uncoupled,
                coupled,
                codomain_is_dual,
                codomain_innerlines,
                codomain_vertices,
            )?,
            FusionTreeKey::try_from_sector_ids(
                domain_uncoupled,
                coupled,
                domain_is_dual,
                domain_innerlines,
                domain_vertices,
            )?,
        ))
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
    pub fn coupled(&self) -> SectorId {
        self.codomain_tree.coupled()
    }

    #[inline]
    pub fn vertices(&self) -> &[MultiplicityIndex] {
        self.codomain_tree.vertices()
    }

    #[inline]
    pub fn codomain_vertices(&self) -> &[MultiplicityIndex] {
        self.codomain_tree.vertices()
    }

    #[inline]
    pub fn domain_vertices(&self) -> &[MultiplicityIndex] {
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

}

/// Deprecated name for [`FusionTreePairKey`].
#[deprecated(
    since = "0.1.0",
    note = "renamed to FusionTreePairKey to distinguish categorical tree pairs from opaque block labels"
)]
pub type FusionTreeBlockKey = FusionTreePairKey;

/// Application-defined block identity with no categorical interpretation.
///
/// The number of words is independent of tensor rank. The inline capacity is
/// a storage detail and never selects an execution path.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct OpaqueBlockKey {
    words: SmallVec<[u64; 2]>,
}

impl OpaqueBlockKey {
    /// Construct an opaque key from owned words.
    pub fn new(words: Vec<u64>) -> Self {
        Self {
            words: words.into_iter().collect(),
        }
    }

    /// Construct an opaque key from any sequence of words.
    pub fn from_words<I>(words: I) -> Self
    where
        I: IntoIterator<Item = u64>,
    {
        Self {
            words: words.into_iter().collect(),
        }
    }

    /// Construct the conventional one-word key for an ordinal block index.
    pub fn ordinal(index: u64) -> Self {
        Self::from_words([index])
    }

    /// Return the uninterpreted words that identify this block.
    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    fn compact_id(&self) -> Option<usize> {
        let [word] = self.words.as_slice() else {
            return None;
        };
        usize::try_from(*word).ok()
    }
}

/// Structural namespace occupied by every key in one [`SectorStructure`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BlockKeyKind {
    Dense,
    Opaque,
    FusionTree,
}

impl fmt::Display for BlockKeyKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense => formatter.write_str("dense"),
            Self::Opaque => formatter.write_str("opaque"),
            Self::FusionTree => formatter.write_str("fusion-tree"),
        }
    }
}

/// Identity of a stored tensor block.
///
/// A [`SectorStructure`] contains either the anonymous dense key, only opaque
/// keys, or only categorical fusion-tree pairs.
// Why-not box the categorical variant: every owned block key would pay a heap
// allocation and indirection on the production lookup path. Its size is
// intentional because the complete categorical identity is stored inline.
#[allow(clippy::large_enum_variant)]
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum BlockKey {
    Dense,
    Opaque(OpaqueBlockKey),
    FusionTree(FusionTreePairKey),
}

impl BlockKey {
    /// Return the anonymous key used by a single dense block.
    pub fn trivial() -> Self {
        Self::Dense
    }

    /// Construct an application-defined opaque key.
    pub fn opaque<I>(words: I) -> Self
    where
        I: IntoIterator<Item = u64>,
    {
        Self::Opaque(OpaqueBlockKey::from_words(words))
    }

    /// Construct an opaque key from the numeric identities of sector values.
    ///
    /// This compatibility helper does not preserve categorical meaning.
    #[deprecated(
        since = "0.1.0",
        note = "use BlockKey::opaque with numeric routing words"
    )]
    pub fn sectors<I>(sectors: I) -> Self
    where
        I: IntoIterator<Item = SectorId>,
    {
        Self::opaque(
            sectors
                .into_iter()
                .map(|sector| supported_usize_to_u64(sector.id())),
        )
    }

    /// Construct an opaque key from numeric values formerly interpreted as
    /// sector identifiers.
    #[deprecated(
        since = "0.1.0",
        note = "use BlockKey::opaque with numeric routing words"
    )]
    pub fn sector_ids<I>(sector_ids: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        Self::opaque(sector_ids.into_iter().map(supported_usize_to_u64))
    }

    /// Construct the conventional one-word opaque key for a block ordinal.
    pub fn ordinal(index: usize) -> Self {
        Self::Opaque(OpaqueBlockKey::ordinal(supported_usize_to_u64(index)))
    }

    /// Return whether this is the anonymous dense key.
    #[inline]
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense)
    }

    /// Borrow the opaque identity, if this key is application-defined.
    #[inline]
    pub fn as_opaque(&self) -> Option<&OpaqueBlockKey> {
        match self {
            Self::Opaque(key) => Some(key),
            _ => None,
        }
    }

    /// Borrow the categorical tree pair, if this key names fusion-tree data.
    #[inline]
    pub fn as_fusion_tree_pair(&self) -> Option<&FusionTreePairKey> {
        match self {
            Self::FusionTree(key) => Some(key),
            _ => None,
        }
    }

    /// Return this key's structural namespace.
    #[inline]
    pub fn kind(&self) -> BlockKeyKind {
        match self {
            Self::Dense => BlockKeyKind::Dense,
            Self::Opaque(_) => BlockKeyKind::Opaque,
            Self::FusionTree(_) => BlockKeyKind::FusionTree,
        }
    }

    fn compact_id(&self) -> Option<usize> {
        match self {
            Self::Dense => Some(0),
            Self::Opaque(key) => key.compact_id(),
            Self::FusionTree(_) => None,
        }
    }

    pub fn fusion_tree_group_key(&self) -> Option<FusionTreeGroupKey> {
        match self {
            Self::Dense => None,
            Self::Opaque(_) => None,
            Self::FusionTree(tree) => Some(tree.group_key()),
        }
    }
}

#[cfg(any(
    target_pointer_width = "16",
    target_pointer_width = "32",
    target_pointer_width = "64"
))]
const fn supported_usize_to_u64(value: usize) -> u64 {
    value as u64
}

#[cfg(not(any(
    target_pointer_width = "16",
    target_pointer_width = "32",
    target_pointer_width = "64"
)))]
compile_error!("OpaqueBlockKey requires a target whose usize fits in u64");

impl From<OpaqueBlockKey> for BlockKey {
    fn from(value: OpaqueBlockKey) -> Self {
        Self::Opaque(value)
    }
}

impl From<FusionTreePairKey> for BlockKey {
    fn from(value: FusionTreePairKey) -> Self {
        Self::FusionTree(value)
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

/// Indices of fusion-tree pairs sharing one external-sector group.
///
/// Unlike TensorKit's `FusionTreeBlock`, this value does not own a vector of
/// tree pairs; its indices refer back to the parent [`BlockStructure`].
/// `Group` is intentional because TeNeT keeps one canonical Rust block-storage
/// owner and uses this value only as a grouped execution view.
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

    fn singleton(group_key: FusionTreeGroupKey, block_index: usize) -> Self {
        let mut block_indices = DimVec::new();
        block_indices.push(block_index);
        Self {
            group_key,
            block_indices,
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
    key_kind: Option<BlockKeyKind>,
    blocks: Vec<SectorBlock>,
    fusion_tree_groups: Vec<FusionTreeBlockGroup>,
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
            key_kind: None,
            blocks: Vec::new(),
            fusion_tree_groups: Vec::new(),
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
        let mut expected_kind = None;
        for key in keys {
            let key = key.into();
            let actual_kind = key.kind();
            if let Some(expected) = expected_kind {
                if expected != actual_kind {
                    return Err(CoreError::MixedBlockKeyKinds {
                        expected,
                        actual: actual_kind,
                    });
                }
            } else {
                expected_kind = Some(actual_kind);
            }
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
        let mut fusion_tree_groups = Vec::<FusionTreeBlockGroup>::new();
        let mut fusion_tree_group_indices =
            FxHashMap::<FusionTreeGroupKey, usize>::default();
        for (index, block) in blocks.iter().enumerate() {
            let Some(group_key) = block.key().fusion_tree_group_key() else {
                continue;
            };
            if let Some(&group_index) = fusion_tree_group_indices.get(&group_key) {
                fusion_tree_groups[group_index].block_indices.push(index);
            } else {
                fusion_tree_group_indices.insert(group_key.clone(), fusion_tree_groups.len());
                fusion_tree_groups.push(FusionTreeBlockGroup::singleton(group_key, index));
            }
        }
        let compact_lookup = CompactBlockLookup::from_blocks(&blocks);
        Ok(Self {
            rank,
            key_kind: expected_kind,
            blocks,
            fusion_tree_groups,
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

    /// Return the homogeneous block-key namespace, or `None` when empty.
    #[inline]
    pub fn key_kind(&self) -> Option<BlockKeyKind> {
        self.key_kind
    }

    #[inline]
    pub fn blocks(&self) -> &[SectorBlock] {
        &self.blocks
    }

    /// Return owned fusion-tree groups in first-appearance storage order.
    ///
    /// Use [`Self::fusion_tree_group_slice`] when the groups do not need to
    /// outlive this structure.
    pub fn fusion_tree_groups(&self) -> Vec<FusionTreeBlockGroup> {
        self.fusion_tree_groups.clone()
    }

    /// Borrow construction-time fusion-tree groups without rebuilding them.
    #[inline]
    pub fn fusion_tree_group_slice(&self) -> &[FusionTreeBlockGroup] {
        &self.fusion_tree_groups
    }

    pub(crate) fn into_fusion_tree_groups(self) -> Vec<FusionTreeBlockGroup> {
        self.fusion_tree_groups
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
        if self.key_kind != Some(key.kind()) {
            return None;
        }
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

    pub fn find_fusion_tree_pair_index(&self, key: &FusionTreePairKey) -> Option<usize> {
        if self.key_kind != Some(BlockKeyKind::FusionTree) {
            return None;
        }
        self.sorted_indices
            .binary_search_by(|&index| match self.blocks[index].key() {
                BlockKey::Dense => std::cmp::Ordering::Less,
                BlockKey::Opaque(_) => std::cmp::Ordering::Less,
                BlockKey::FusionTree(tree) => tree.cmp(key),
            })
            .ok()
            .map(|position| self.sorted_indices[position])
    }

    #[doc(hidden)]
    pub fn find_adjoint_fusion_tree_pair_index(
        &self,
        logical_key: &FusionTreePairKey,
    ) -> Option<usize> {
        if self.key_kind != Some(BlockKeyKind::FusionTree) {
            return None;
        }
        self.sorted_indices
            .binary_search_by(|&index| match self.blocks[index].key() {
                BlockKey::FusionTree(storage_key) => storage_key
                    .codomain_tree()
                    .cmp(logical_key.domain_tree())
                    .then_with(|| {
                        storage_key
                            .domain_tree()
                            .cmp(logical_key.codomain_tree())
                    }),
                _ => std::cmp::Ordering::Less,
            })
            .ok()
            .map(|position| self.sorted_indices[position])
    }

    #[deprecated(
        since = "0.1.0",
        note = "renamed to find_fusion_tree_pair_index to match FusionTreePairKey"
    )]
    pub fn find_fusion_tree_index(&self, key: &FusionTreePairKey) -> Option<usize> {
        self.find_fusion_tree_pair_index(key)
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
        if self.key_kind != src.key_kind {
            let (Some(expected), Some(actual)) = (self.key_kind, src.key_kind) else {
                unreachable!("equal nonzero block counts cannot mix empty and nonempty structures");
            };
            return Err(CoreError::MixedBlockKeyKinds { expected, actual });
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
        if blocks.is_empty() {
            return None;
        }
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
    coupled: SectorId,
    rows: usize,
    cols: usize,
    range: core::ops::Range<usize>,
    row_trees: Vec<CoupledTreeExtent>,
    col_trees: Vec<CoupledTreeExtent>,
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
    sector: SectorStructure,
    degeneracy: DegeneracyStructure,
    blocks: Arc<[BlockStructureContentBlock]>,
    required_len: usize,
}

impl core::fmt::Debug for BlockStructureContent {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("BlockStructureContent")
            .field("id", &self.id)
            .field("rank", &self.sector.rank())
            .field("blocks", &self.blocks)
            .finish()
    }
}

// Content equality deliberately ignores `id`: the id is a process-local
// intern handle (monotonic since the bounded-FIFO change, never reused across
// eviction or reset), not part of the content. Including it in the derived
// PartialEq made content-equal structures interned in different reset
// epochs compare unequal, which broke replay's content-fallback validation
// (caught by reset_and_concurrent_rebuild_keep_structure_semantics in CI).
// Id-keyed caches are unaffected: they key on `id()` explicitly and rely on
// monotonicity, not on equality of the full content struct.
impl PartialEq for BlockStructureContent {
    fn eq(&self, other: &Self) -> bool {
        self.sector == other.sector
            && self.degeneracy == other.degeneracy
            && self.blocks == other.blocks
            && self.required_len == other.required_len
    }
}

impl BlockStructureContent {
    /// Process-local intern id (insertion-order counter into the block-structure
    /// intern table). Identical content shares one `Arc` and id while its
    /// interner key remains resident and at least one strong owner is live.
    /// Rebuilding after owner death, eviction, or reset issues a fresh id.
    /// It is not semantic identity and must never be serialized.
    #[inline]
    pub fn id(&self) -> usize {
        self.id
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.sector.rank()
    }

    #[inline]
    pub fn blocks(&self) -> &[BlockStructureContentBlock] {
        &self.blocks
    }

    pub(crate) fn charged_retained_bytes(&self) -> usize {
        fn key_bytes(key: &BlockKey) -> usize {
            match key {
                BlockKey::Dense => 0,
                BlockKey::Opaque(key) => spilled_smallvec_heap_bytes(&key.words),
                BlockKey::FusionTree(pair) => {
                    charged_fusion_tree_key_heap_bytes(pair.codomain_tree())
                        .saturating_add(charged_fusion_tree_key_heap_bytes(pair.domain_tree()))
                }
            }
        }

        let sector_blocks = self.sector.blocks.iter().fold(0usize, |bytes, block| {
            bytes.saturating_add(key_bytes(block.key()))
        });
        let groups = self
            .sector
            .fusion_tree_groups
            .iter()
            .fold(0usize, |bytes, group| {
                bytes
                    .saturating_add(spilled_smallvec_heap_bytes(&group.block_indices))
                    .saturating_add(spilled_smallvec_heap_bytes(
                        &group.group_key.codomain_uncoupled,
                    ))
                    .saturating_add(spilled_smallvec_heap_bytes(
                        &group.group_key.domain_uncoupled,
                    ))
                    .saturating_add(spilled_smallvec_heap_bytes(
                        &group.group_key.codomain_is_dual,
                    ))
                    .saturating_add(spilled_smallvec_heap_bytes(
                        &group.group_key.domain_is_dual,
                    ))
            });
        let compact_lookup = self.sector.compact_lookup.as_ref().map_or(0, |lookup| {
            spilled_smallvec_heap_bytes(&lookup.indices)
        });
        let degeneracy = self.degeneracy.blocks.iter().fold(0usize, |bytes, block| {
            bytes
                .saturating_add(spilled_smallvec_heap_bytes(&block.shape))
                .saturating_add(spilled_smallvec_heap_bytes(&block.strides))
        });
        let copied_blocks = self.blocks.iter().fold(0usize, |bytes, block| {
            bytes
                .saturating_add(key_bytes(&block.key))
                .saturating_add(spilled_smallvec_heap_bytes(&block.shape))
                .saturating_add(spilled_smallvec_heap_bytes(&block.strides))
        });

        std::mem::size_of::<BlockStructureContent>()
            .saturating_add(self.sector.blocks.capacity().saturating_mul(std::mem::size_of::<SectorBlock>()))
            .saturating_add(sector_blocks)
            .saturating_add(
                self.sector
                    .fusion_tree_groups
                    .capacity()
                    .saturating_mul(std::mem::size_of::<FusionTreeBlockGroup>()),
            )
            .saturating_add(groups)
            .saturating_add(spilled_smallvec_heap_bytes(&self.sector.sorted_indices))
            .saturating_add(compact_lookup)
            .saturating_add(
                self.degeneracy
                    .blocks
                    .capacity()
                    .saturating_mul(std::mem::size_of::<DegeneracyBlock>()),
            )
            .saturating_add(degeneracy)
            .saturating_add(
                self.blocks
                    .len()
                    .saturating_mul(std::mem::size_of::<BlockStructureContentBlock>()),
            )
            .saturating_add(copied_blocks)
    }
}

#[derive(Default)]
struct BlockStructureRegionState {
    coupled_region_cache: OnceLock<CoupledRegionCache>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct BlockStructureInternKey {
    rank: usize,
    blocks: Arc<[BlockStructureContentBlock]>,
}

struct BlockStructureInternEntry {
    content: Weak<BlockStructureContent>,
    charged_key_bytes: usize,
}

struct BlockStructureInternTable {
    entries: lru::LruCache<
        BlockStructureInternKey,
        BlockStructureInternEntry,
        rustc_hash::FxBuildHasher,
    >,
    entry_capacity: usize,
    byte_budget: usize,
    max_entry_bytes: usize,
    charged_key_bytes: usize,
    pressure_evictions: usize,
    oversized_admission_bypasses: usize,
}

/// Entry cap for the block-structure content intern table (and, reusing the
/// same bound, the arc dedup and coupled-subblock caches). Mirrors
/// `HOM_SPACE_INTERN_CAP`: a long-lived / multi-tenant process can otherwise
/// grow these tables without bound over a χ sweep. See
/// `BLOCK_STRUCTURE_CONTENT_ID` for why capping this particular table is
/// aliasing-safe despite its ids being consumed as cache keys downstream.
const BLOCK_STRUCTURE_INTERN_CAP: usize = 8192;
const BLOCK_STRUCTURE_INTERN_BYTE_BUDGET: usize = 64 * 1024 * 1024;
const BLOCK_STRUCTURE_INTERN_MAX_ENTRY_BYTES: usize = 8 * 1024 * 1024;
// Why-not allocator-exact accounting: allocator headers are not portable.
// This fixed allowance conservatively covers hash/FIFO nodes, the weak handle,
// and the surviving Arc control-allocation shell.
const BLOCK_STRUCTURE_INTERN_CONTROL_ALLOWANCE_BYTES: usize =
    std::mem::size_of::<BlockStructureContent>() + 8 * std::mem::size_of::<usize>();

impl BlockStructureInternTable {
    fn new(entry_capacity: usize, byte_budget: usize, max_entry_bytes: usize) -> Self {
        assert!(
            entry_capacity > 0,
            "block-structure intern capacity must be positive"
        );
        Self {
            entries: lru::LruCache::with_hasher(
                std::num::NonZeroUsize::new(entry_capacity).unwrap(),
                rustc_hash::FxBuildHasher,
            ),
            entry_capacity,
            byte_budget,
            max_entry_bytes,
            charged_key_bytes: 0,
            pressure_evictions: 0,
            oversized_admission_bypasses: 0,
        }
    }

    fn lookup(&self, key: &BlockStructureInternKey) -> Option<Arc<BlockStructureContent>> {
        self.entries
            .peek(key)
            .and_then(|entry| entry.content.upgrade())
    }

    fn intern_with<C, F>(
        &mut self,
        key: BlockStructureInternKey,
        charge_key: C,
        make_content: F,
    ) -> Arc<BlockStructureContent>
    where
        C: FnOnce(&BlockStructureInternKey) -> usize,
        F: FnOnce() -> Arc<BlockStructureContent>,
    {
        if let Some(entry) = self.entries.peek_mut(&key) {
            if let Some(content) = entry.content.upgrade() {
                return content;
            }
            let content = make_content();
            entry.content = Arc::downgrade(&content);
            return content;
        }

        let charged_key_bytes = charge_key(&key);
        let content = make_content();
        if charged_key_bytes == usize::MAX
            || charged_key_bytes > self.max_entry_bytes
            || charged_key_bytes > self.byte_budget
        {
            self.oversized_admission_bypasses =
                self.oversized_admission_bypasses.saturating_add(1);
            return content;
        }

        while self.entries.len() >= self.entry_capacity
            || self
                .charged_key_bytes
                .saturating_add(charged_key_bytes)
                > self.byte_budget
        {
            let Some((_, evicted)) = self.entries.pop_lru() else {
                break;
            };
            self.charged_key_bytes = self
                .charged_key_bytes
                .saturating_sub(evicted.charged_key_bytes);
            self.pressure_evictions = self.pressure_evictions.saturating_add(1);
        }

        self.charged_key_bytes = self.charged_key_bytes.saturating_add(charged_key_bytes);
        self.entries.put(
            key,
            BlockStructureInternEntry {
                content: Arc::downgrade(&content),
                charged_key_bytes,
            },
        );
        content
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.charged_key_bytes = 0;
        self.pressure_evictions = 0;
        self.oversized_admission_bypasses = 0;
    }

    fn info(&self) -> BlockStructureInternCacheInfo {
        BlockStructureInternCacheInfo {
            entries: self.entries.len(),
            entry_capacity: self.entry_capacity,
            charged_key_bytes: self.charged_key_bytes,
            byte_budget: self.byte_budget,
            max_admitted_entry_bytes: self.max_entry_bytes,
            pressure_evictions: self.pressure_evictions,
            oversized_admission_bypasses: self.oversized_admission_bypasses,
        }
    }
}

/// Snapshot of the process-global block-structure interner resource bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockStructureInternCacheInfo {
    entries: usize,
    entry_capacity: usize,
    charged_key_bytes: usize,
    byte_budget: usize,
    max_admitted_entry_bytes: usize,
    pressure_evictions: usize,
    oversized_admission_bypasses: usize,
}

impl BlockStructureInternCacheInfo {
    pub fn entries(self) -> usize {
        self.entries
    }

    pub fn entry_capacity(self) -> usize {
        self.entry_capacity
    }

    /// Conservative key charge used for admission and eviction.
    ///
    /// This is an accounting contract, not allocator-observed resident bytes.
    pub fn charged_key_bytes(self) -> usize {
        self.charged_key_bytes
    }

    pub fn byte_budget(self) -> usize {
        self.byte_budget
    }

    pub fn max_admitted_entry_bytes(self) -> usize {
        self.max_admitted_entry_bytes
    }

    pub fn pressure_evictions(self) -> usize {
        self.pressure_evictions
    }

    pub fn oversized_admission_bypasses(self) -> usize {
        self.oversized_admission_bypasses
    }
}

fn block_structure_intern_table() -> &'static RwLock<BlockStructureInternTable> {
    static TABLE: OnceLock<RwLock<BlockStructureInternTable>> = OnceLock::new();
    TABLE.get_or_init(|| {
        RwLock::new(BlockStructureInternTable::new(
            BLOCK_STRUCTURE_INTERN_CAP,
            BLOCK_STRUCTURE_INTERN_BYTE_BUDGET,
            BLOCK_STRUCTURE_INTERN_MAX_ENTRY_BYTES,
        ))
    })
}

/// Returns the bounded resource state of the process-global block interner.
pub fn block_structure_intern_cache_info() -> BlockStructureInternCacheInfo {
    block_structure_intern_table()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .info()
}

fn spilled_smallvec_heap_bytes<A>(values: &SmallVec<A>) -> usize
where
    A: smallvec::Array,
{
    if values.spilled() {
        values
            .capacity()
            .saturating_mul(std::mem::size_of::<A::Item>())
    } else {
        0
    }
}

fn charged_fusion_tree_key_heap_bytes(tree: &FusionTreeKey) -> usize {
    spilled_smallvec_heap_bytes(&tree.uncoupled)
        .saturating_add(spilled_smallvec_heap_bytes(&tree.is_dual))
        .saturating_add(spilled_smallvec_heap_bytes(&tree.innerlines))
        .saturating_add(spilled_smallvec_heap_bytes(&tree.vertices))
}

fn charged_block_structure_intern_key_bytes(key: &BlockStructureInternKey) -> usize {
    key.blocks
        .iter()
        .fold(
            std::mem::size_of::<BlockStructureInternKey>()
                .saturating_add(std::mem::size_of::<BlockStructureInternEntry>())
                .saturating_add(
                    key.blocks
                        .len()
                        .saturating_mul(std::mem::size_of::<BlockStructureContentBlock>()),
                )
                .saturating_add(BLOCK_STRUCTURE_INTERN_CONTROL_ALLOWANCE_BYTES),
            |charged, block| {
                let key_heap = match &block.key {
                    BlockKey::Dense => 0,
                    BlockKey::Opaque(key) => spilled_smallvec_heap_bytes(&key.words),
                    BlockKey::FusionTree(pair) => charged_fusion_tree_key_heap_bytes(
                        pair.codomain_tree(),
                    )
                    .saturating_add(charged_fusion_tree_key_heap_bytes(pair.domain_tree())),
                };
                charged
                    .saturating_add(key_heap)
                    .saturating_add(spilled_smallvec_heap_bytes(&block.shape))
                    .saturating_add(spilled_smallvec_heap_bytes(&block.strides))
            },
        )
}

/// Process-global, strictly-monotonic id source for interned block-structure
/// content.
///
/// Why-not (`id = table.len() + 1`): the intern table is bounded FIFO (above),
/// so its size is no longer monotonic — `len() + 1` would re-issue an id to
/// DIFFERENT content after an eviction. `BlockStructureCacheKey` (tenet-tensors)
/// keys the tree-transform and contract structure caches *purely* by this id
/// (both `Hash` and `Eq` read only `content.id()`), so a recycled id would
/// silently alias two distinct structures and hand back the wrong cached kernel
/// — an aliasing-class correctness bug. A monotonic counter never reuses an id:
/// not across FIFO eviction, and not across `reset_core_intern_tables` (the
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
    sector: SectorStructure,
    degeneracy: DegeneracyStructure,
    required_len: usize,
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
        if let Some(content) = read.lookup(&key) {
            return content;
        }
    }

    let mut write = table
        .write()
        .expect("block structure intern table poisoned");
    write.intern_with(key, charged_block_structure_intern_key_bytes, || {
        Arc::new(BlockStructureContent {
            id: BLOCK_STRUCTURE_CONTENT_ID.fetch_add(1, Ordering::Relaxed),
            sector,
            degeneracy,
            blocks,
            required_len,
        })
    })
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
    // Clear the sole strong complete-layout owner before weak canonicalizers.
    // Live wrappers keep their own content and region state through reset.
    reset_complete_hom_space_structure_cache();
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

pub struct BlockStructure {
    content: Arc<BlockStructureContent>,
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

struct PreparedBlockStructure {
    sector: SectorStructure,
    degeneracy: DegeneracyStructure,
    required_len: usize,
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

    fn from_parts(
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
        Ok(Self {
            sector,
            degeneracy,
            required_len,
        })
    }

    fn commit(self) -> BlockStructure {
        BlockStructure::from_content(intern_block_structure_content(
            self.sector,
            self.degeneracy,
            self.required_len,
        ))
    }
}

/// Proof that one exact [`BlockStructure`] is categorically valid for `rule`.
///
/// This proof is deliberately LOCAL: it covers tree shape, fusion and
/// vertex-label admissibility under the provider-owned fusion style, and
/// structure rank for provider-domain sector IDs in this borrowed structure. It is neither a
/// [`FusionTensorMapSpace`] construction proof nor a firewall for arbitrary
/// numeric sector IDs.
///
/// Why not make this crate-private: `tenet-tensors` consumes the proof across
/// the crate boundary while the type remains hidden from generated public API
/// documentation.
#[doc(hidden)]
pub struct LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R> {
    rule: &'rule R,
    structure: &'structure BlockStructure,
}

#[doc(hidden)]
impl<'rule, 'structure, R> LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: FusionRule,
{
    pub fn try_new(
        rule: &'rule R,
        structure: &'structure BlockStructure,
    ) -> Result<Self, CoreError> {
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(CoreError::ExpectedFusionTreePairKey {
                    actual: block.key().kind(),
                });
            };
            let key_rank = key.codomain_tree().uncoupled().len()
                + key.domain_tree().uncoupled().len();
            if key_rank != structure.rank() {
                return Err(CoreError::StructureRankMismatch {
                    expected: structure.rank(),
                    actual: key_rank,
                });
            }
            key.validate_for_rule(rule)?;
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

    #[doc(hidden)]
    pub fn fusion_tree_pair_key(
        &self,
        index: usize,
    ) -> Result<Option<&'structure FusionTreePairKey>, CoreError> {
        Ok(match self.structure.block(index)?.key() {
            BlockKey::FusionTree(key) => Some(key),
            key => {
                return Err(CoreError::ExpectedFusionTreePairKey { actual: key.kind() });
            }
        })
    }

    #[doc(hidden)]
    #[deprecated(
        since = "0.1.0",
        note = "renamed to fusion_tree_pair_key to match FusionTreePairKey"
    )]
    pub fn fusion_tree_block_key(
        &self,
        index: usize,
    ) -> Result<Option<&'structure FusionTreePairKey>, CoreError> {
        self.fusion_tree_pair_key(index)
    }
}

#[doc(hidden)]
impl<'rule, 'structure, R> LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    pub fn execute_unique_rigid_for_block_index(
        &self,
        index: usize,
        operation: &PreparedTreePairOperation<'_>,
    ) -> Result<(FusionTreePairKey, R::Scalar), CoreError> {
        if self.rule.fusion_style() != FusionStyleKind::Unique {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: self.rule.fusion_style(),
            });
        }
        let key = self.required_fusion_tree_pair_key(index)?;
        operation.execute_unique_rigid_proven(ValidatedFusionTreePair {
            rule: self.rule,
            key,
        })
    }

    pub fn execute_multiplicity_free_for_block_index(
        &self,
        index: usize,
        operation: &PreparedTreePairOperation<'_>,
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError> {
        if !self.rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: self.rule.fusion_style(),
            });
        }
        let key = self.required_fusion_tree_pair_key(index)?;
        operation.execute_multiplicity_free_proven(ValidatedFusionTreePair {
            rule: self.rule,
            key,
        })
    }

    pub fn execute_multiplicity_free_braid_for_block_indices<I>(
        &self,
        indices: I,
        operation: PreparedTreePairOperation<'_>,
    ) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        validate_multiplicity_free_execution_style(self.rule)?;
        operation
            .validate_block_preflight(self.rule, PreparedTreePairFamily::BraidLike)?;
        if operation.is_identity() {
            let indices = indices.into_iter();
            let (lower, upper) = indices.size_hint();
            let mut rows = Vec::with_capacity(upper.unwrap_or(lower));
            for index in indices {
                let source = self.required_fusion_tree_pair_key(index)?;
                operation.validate_source_split(
                    source.codomain_tree().uncoupled().len(),
                    source.domain_tree().uncoupled().len(),
                )?;
                rows.push(vec![(source.clone(), self.rule.scalar_one())]);
            }
            return Ok(rows);
        }
        let batch =
            ValidatedMultiplicityFreePairBatch::from_locally_validated(self, indices)?;
        multiplicity_free_braid_tree_pair_block_proven(batch, &operation)
    }

    pub fn execute_multiplicity_free_braid_ordered_for_block_indices<I>(
        &self,
        indices: I,
        operation: PreparedTreePairOperation<'_>,
    ) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        self.execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
            indices, &operation,
        )
    }

    #[doc(hidden)]
    pub fn execute_multiplicity_free_braid_ordered_for_block_indices_borrowed<I>(
        &self,
        indices: I,
        operation: &PreparedTreePairOperation<'_>,
    ) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        validate_multiplicity_free_execution_style(self.rule)?;
        operation
            .validate_block_preflight(self.rule, PreparedTreePairFamily::BraidLike)?;
        let batch =
            ValidatedMultiplicityFreePairBatch::from_locally_validated(self, indices)?;
        multiplicity_free_braid_tree_pair_block_ordered_proven(batch, operation)
    }

    pub fn execute_multiplicity_free_transpose_for_block_indices<I>(
        &self,
        indices: I,
        operation: PreparedTreePairOperation<'_>,
    ) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        validate_multiplicity_free_execution_style(self.rule)?;
        operation.validate_block_preflight(self.rule, PreparedTreePairFamily::Transpose)?;
        if operation.is_identity() {
            let indices = indices.into_iter();
            let (lower, upper) = indices.size_hint();
            let mut rows = Vec::with_capacity(upper.unwrap_or(lower));
            for index in indices {
                let source = self.required_fusion_tree_pair_key(index)?;
                operation.validate_source_split(
                    source.codomain_tree().uncoupled().len(),
                    source.domain_tree().uncoupled().len(),
                )?;
                rows.push(vec![(source.clone(), self.rule.scalar_one())]);
            }
            return Ok(rows);
        }
        let batch =
            ValidatedMultiplicityFreePairBatch::from_locally_validated(self, indices)?;
        multiplicity_free_transpose_tree_pair_block_proven(batch, &operation)
    }

    pub fn execute_multiplicity_free_transpose_ordered_for_block_indices<I>(
        &self,
        indices: I,
        operation: PreparedTreePairOperation<'_>,
    ) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        self.execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
            indices, &operation,
        )
    }

    #[doc(hidden)]
    pub fn execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed<I>(
        &self,
        indices: I,
        operation: &PreparedTreePairOperation<'_>,
    ) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        validate_multiplicity_free_execution_style(self.rule)?;
        operation.validate_block_preflight(self.rule, PreparedTreePairFamily::Transpose)?;
        let batch =
            ValidatedMultiplicityFreePairBatch::from_locally_validated(self, indices)?;
        multiplicity_free_transpose_tree_pair_block_ordered_proven(batch, operation)
    }
}

#[doc(hidden)]
impl<'rule, 'structure, R> LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    pub fn braid_codomain_rows_for_block_index(
        &self,
        block_index: usize,
        permutation: &[usize],
        levels: &[usize],
    ) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError> {
        if !self.rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: self.rule.fusion_style(),
            });
        }
        let source = self.required_fusion_tree_pair_key(block_index)?;
        let tree = source.codomain_tree();
        let rank = tree.uncoupled().len();
        if levels.len() != rank {
            return Err(CoreError::DimensionMismatch {
                expected: rank,
                actual: levels.len(),
            });
        }
        let prepared = PreparedTreeBraid::new(permutation, levels, rank)?;
        execute_multiplicity_free_tree_braid_proven(
            ValidatedFusionTree {
                rule: self.rule,
                key: tree,
            },
            prepared,
        )
    }

    pub fn permute_codomain_rows_for_block_index(
        &self,
        block_index: usize,
        permutation: &[usize],
    ) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError> {
        if !self.rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: self.rule.braiding_style(),
            });
        }
        if !self.rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: self.rule.fusion_style(),
            });
        }
        let source = self.required_fusion_tree_pair_key(block_index)?;
        let tree = source.codomain_tree();
        let rank = tree.uncoupled().len();
        let levels = (0..rank).collect::<SmallVec<[usize; 8]>>();
        let prepared = PreparedTreeBraid::new(permutation, &levels, rank)?;
        execute_multiplicity_free_tree_braid_proven(
            ValidatedFusionTree {
                rule: self.rule,
                key: tree,
            },
            prepared,
        )
    }

    pub fn braid_codomain_rows_for_block_indices<I>(
        &self,
        indices: I,
        permutation: &[usize],
        levels: &[usize],
    ) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        validate_multiplicity_free_execution_style(self.rule)?;
        let batch =
            ValidatedMultiplicityFreeTreeBatch::from_locally_validated(self, indices)?;
        multiplicity_free_braid_tree_block_proven(batch, permutation, levels)
    }

    pub fn permute_codomain_rows_for_block_indices<I>(
        &self,
        indices: I,
        permutation: &[usize],
    ) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
    where
        I: IntoIterator<Item = usize>,
    {
        if !self.rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: self.rule.braiding_style(),
            });
        }
        let indices = indices.into_iter().collect::<SmallVec<[usize; 8]>>();
        let rank = indices
            .first()
            .map(|&index| {
                self.required_fusion_tree_pair_key(index)
                    .map(|key| key.codomain_tree().uncoupled().len())
            })
            .transpose()?
            .unwrap_or(0);
        let levels = (0..rank).collect::<SmallVec<[usize; 8]>>();
        self.braid_codomain_rows_for_block_indices(indices, permutation, &levels)
    }
}

#[doc(hidden)]
impl<'rule, 'structure, R> LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    pub fn generic_permute_tree_pair_for_block_index(
        &self,
        block_index: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError> {
        let source = self.required_generic_fusion_tree_pair_key(block_index)?;
        generic_permute_tree_pair_proven(
            ValidatedFusionTreePair {
                rule: self.rule,
                key: source,
            },
            codomain_permutation,
            domain_permutation,
        )
    }

    pub fn generic_braid_tree_pair_for_block_index(
        &self,
        block_index: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        codomain_levels: &[usize],
        domain_levels: &[usize],
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError> {
        let source = self.required_generic_fusion_tree_pair_key(block_index)?;
        generic_braid_tree_pair_proven(
            ValidatedFusionTreePair {
                rule: self.rule,
                key: source,
            },
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        )
    }

    pub fn generic_transpose_tree_pair_for_block_index(
        &self,
        block_index: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError> {
        let source = self.required_generic_fusion_tree_pair_key(block_index)?;
        generic_transpose_tree_pair_proven(
            ValidatedFusionTreePair {
                rule: self.rule,
                key: source,
            },
            codomain_permutation,
            domain_permutation,
        )
    }

    fn required_generic_fusion_tree_pair_key(
        &self,
        block_index: usize,
    ) -> Result<&FusionTreePairKey, CoreError> {
        if self.rule.fusion_style() != FusionStyleKind::Generic {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual: self.rule.fusion_style(),
            });
        }
        self.required_fusion_tree_pair_key(block_index)
    }
}

impl<'rule, 'structure, R> LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: FusionRule,
{
    fn required_fusion_tree_pair_key(
        &self,
        index: usize,
    ) -> Result<&'structure FusionTreePairKey, CoreError> {
        self.fusion_tree_pair_key(index)?
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
        Self::from_content(intern_block_structure_content(sector, degeneracy, 0))
    }

    pub fn from_blocks(blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        let rank = blocks.first().map(|block| block.shape().len()).unwrap_or(0);
        Self::from_blocks_with_rank(rank, blocks)
    }

    pub fn from_blocks_with_rank(rank: usize, blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        PreparedBlockStructure::from_blocks_with_rank(rank, blocks).map(|prepared| prepared.commit())
    }

    pub fn from_parts(
        sector: SectorStructure,
        degeneracy: DegeneracyStructure,
    ) -> Result<Self, CoreError> {
        PreparedBlockStructure::from_parts(sector, degeneracy).map(|prepared| prepared.commit())
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
        self.content
            .sector
            .pair_indices_from(&src.content.sector)
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
        let index = self
            .find_block_index_by_key(key)
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
                compile_coupled_sector_regions(self, nout)
                    .map(|regions| regions.map(Arc::from))
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

#[cfg(test)]
thread_local! {
    static EXACT_STORAGE_FALLBACKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_exact_storage_fallback_count() {
    EXACT_STORAGE_FALLBACKS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn exact_storage_fallback_count() -> usize {
    EXACT_STORAGE_FALLBACKS.with(std::cell::Cell::get)
}

/// Checks that every logical block element owns a distinct storage offset.
///
/// General [`BlockStructure`] values may describe aliased read views. Owning
/// symmetric spaces and operation destinations call this explicit admission
/// boundary before publishing writable storage.
#[doc(hidden)]
pub fn validate_block_storage_injective(structure: &BlockStructure) -> Result<(), CoreError> {
    #[derive(Clone, Copy)]
    struct BoundedBlock {
        block: usize,
        start: usize,
        end: usize,
    }

    let mut bounded = Vec::with_capacity(structure.block_count());
    for block in 0..structure.block_count() {
        let layout = structure.block(block)?;
        let Some((start, end)) =
            block_layout_bounds(layout.shape(), layout.strides(), layout.offset())?
        else {
            continue;
        };
        let proven_injective =
            block_layout_is_proven_injective(layout.shape(), layout.strides());
        if !proven_injective {
            #[cfg(test)]
            EXACT_STORAGE_FALLBACKS.with(|count| count.set(count.get() + 1));
            let mut offsets = FxHashMap::<usize, ()>::default();
            if let Some(offset) =
                visit_block_layout_offsets(layout.shape(), layout.strides(), layout.offset(), |at| {
                    offsets.insert(at, ()).is_some()
                })?
            {
                return Err(CoreError::OverlappingBlockStorage {
                    first_block: block,
                    second_block: block,
                    offset,
                });
            }
        }
        bounded.push(BoundedBlock {
            block,
            start,
            end,
        });
    }
    bounded.sort_by_key(|entry| entry.start);

    let mut component_start = 0;
    while component_start < bounded.len() {
        let mut component_end = component_start + 1;
        let mut max_end = bounded[component_start].end;
        while component_end < bounded.len() && bounded[component_end].start <= max_end {
            max_end = max_end.max(bounded[component_end].end);
            component_end += 1;
        }
        let component = &mut bounded[component_start..component_end];
        if component.len() == 1 {
            component_start = component_end;
            continue;
        }

        #[cfg(test)]
        EXACT_STORAGE_FALLBACKS.with(|count| count.set(count.get() + 1));
        component.sort_by_key(|entry| entry.block);
        let mut owners = FxHashMap::<usize, usize>::default();
        for entry in component {
            let layout = structure.block(entry.block)?;
            let mut collision = None;
            visit_block_layout_offsets(
                layout.shape(),
                layout.strides(),
                layout.offset(),
                |offset| {
                    if let Some(&first_block) = owners.get(&offset) {
                        collision = Some((first_block, offset));
                        true
                    } else {
                        owners.insert(offset, entry.block);
                        false
                    }
                },
            )?;
            if let Some((first_block, offset)) = collision {
                return Err(CoreError::OverlappingBlockStorage {
                    first_block,
                    second_block: entry.block,
                    offset,
                });
            }
        }
        component_start = component_end;
    }
    Ok(())
}

fn block_layout_is_proven_injective(shape: &[usize], strides: &[usize]) -> bool {
    let mut axes = shape
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .filter(|&(extent, _)| extent > 1)
        .collect::<SmallVec<[(usize, usize); 8]>>();
    axes.sort_unstable_by_key(|&(_, stride)| stride);
    let mut lower_span = 0usize;
    for (extent, stride) in axes {
        if stride == 0 || stride <= lower_span {
            return false;
        }
        let Some(span) = (extent - 1).checked_mul(stride) else {
            return false;
        };
        let Some(next_span) = lower_span.checked_add(span) else {
            return false;
        };
        lower_span = next_span;
    }
    true
}

fn block_layout_bounds(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Result<Option<(usize, usize)>, CoreError> {
    if shape.contains(&0) {
        return Ok(None);
    }
    let end = storage_end_exclusive(shape, strides, offset)?
        .checked_sub(1)
        .ok_or(CoreError::ElementCountOverflow)?;
    Ok(Some((offset, end)))
}

fn visit_block_layout_offsets<F>(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    mut stop: F,
) -> Result<Option<usize>, CoreError>
where
    F: FnMut(usize) -> bool,
{
    if shape.contains(&0) {
        return Ok(None);
    }
    let mut indices = SmallVec::<[usize; 8]>::from_elem(0, shape.len());
    let mut physical = offset;
    loop {
        if stop(physical) {
            return Ok(Some(physical));
        }

        let mut axis = 0;
        while axis < indices.len() {
            indices[axis] += 1;
            if indices[axis] < shape[axis] {
                physical = physical
                    .checked_add(strides[axis])
                    .ok_or(CoreError::ElementCountOverflow)?;
                break;
            }
            indices[axis] = 0;
            physical = physical
                .checked_sub(
                    shape[axis]
                        .saturating_sub(1)
                        .checked_mul(strides[axis])
                        .ok_or(CoreError::ElementCountOverflow)?,
                )
                .ok_or(CoreError::ElementCountOverflow)?;
            axis += 1;
        }
        if axis == indices.len() {
            return Ok(None);
        }
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
        let mut seen_coupled = FxHashMap::<SectorId, ()>::default();
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
            if end_offset > structure.content.required_len {
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
        if next_offset != structure.content.required_len {
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
