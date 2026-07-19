#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FusionProductSpace {
    legs: SmallVec<[SectorLeg; 8]>,
}

impl FusionProductSpace {
    /// Builds a product of external fusion legs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{FusionProductSpace, SectorLeg, Z2Irrep};
    ///
    /// let space = FusionProductSpace::new([
    ///     SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false),
    ///     SectorLeg::new([(Z2Irrep::EVEN, 1)], true),
    /// ]);
    /// assert_eq!(space.len(), 2);
    /// ```
    pub fn new<Legs>(legs: Legs) -> Self
    where
        Legs: IntoIterator<Item = SectorLeg>,
    {
        Self {
            legs: legs.into_iter().collect(),
        }
    }

    /// Builds a product of single-sector legs from `(sector id, degeneracy)`
    /// pairs.
    pub fn from_sector_ids<Sectors>(sectors: Sectors) -> Self
    where
        Sectors: IntoIterator<Item = (usize, usize)>,
    {
        Self::new(
            sectors
                .into_iter()
                .map(|(sector, degeneracy)| SectorLeg::from_sector_id(sector, degeneracy)),
        )
    }

    #[inline]
    pub fn legs(&self) -> &[SectorLeg] {
        &self.legs
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.legs.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.legs.is_empty()
    }

    fn selected_leg_tuples(&self) -> Vec<Vec<FusionTreeLeg>> {
        let mut tuples = Vec::new();
        let mut current = vec![None; self.legs.len()];
        collect_selected_leg_tuples(&self.legs, self.legs.len(), &mut current, &mut tuples);
        tuples
    }
}

#[derive(Clone, Debug)]
pub struct FusionTreeHomSpace {
    codomain: FusionProductSpace,
    domain: FusionProductSpace,
    id: OnceLock<HomSpaceId>,
}

impl PartialEq for FusionTreeHomSpace {
    fn eq(&self, other: &Self) -> bool {
        self.codomain == other.codomain && self.domain == other.domain
    }
}

impl Eq for FusionTreeHomSpace {}

impl std::hash::Hash for FusionTreeHomSpace {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.codomain.hash(state);
        self.domain.hash(state);
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FusionTreeLegSetSignature {
    sectors: SectorVec,
    is_dual: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FusionTreeHomSpaceCacheKey {
    rule: RuleIdentity,
    codomain: Vec<FusionTreeLegSetSignature>,
    domain: Vec<FusionTreeLegSetSignature>,
}

impl FusionTreeHomSpaceCacheKey {
    fn new<R>(rule: &R, homspace: &FusionTreeHomSpace) -> Self
    where
        R: MultiplicityFreeFusionRule,
    {
        Self {
            rule: rule.rule_identity(),
            codomain: fusion_product_space_signature(homspace.codomain()),
            domain: fusion_product_space_signature(homspace.domain()),
        }
    }
}

fn fusion_product_space_signature(space: &FusionProductSpace) -> Vec<FusionTreeLegSetSignature> {
    space
        .legs()
        .iter()
        .map(|leg| FusionTreeLegSetSignature {
            sectors: leg.sectors().iter().copied().collect(),
            is_dual: leg.is_dual(),
        })
        .collect()
}

fn fusion_product_space_matches_signature(
    space: &FusionProductSpace,
    signature: &[FusionTreeLegSetSignature],
) -> bool {
    space.legs().len() == signature.len()
        && space.legs().iter().zip(signature).all(|(leg, expected)| {
            leg.is_dual() == expected.is_dual && leg.sectors() == expected.sectors.as_slice()
        })
}

#[derive(Clone, Debug)]
struct FusionTreeBlockLayoutEntry {
    row: usize,
    col: usize,
}

#[derive(Clone, Debug)]
struct FusionTreeCoupledSectorLayout {
    start: usize,
    row_count: usize,
    col_count: usize,
    entries: Vec<FusionTreeBlockLayoutEntry>,
}

#[derive(Clone, Debug)]
struct FusionTreeHomSpaceLayoutData {
    keys: Arc<[FusionTreePairKey]>,
    sectors: Vec<FusionTreeCoupledSectorLayout>,
}

#[derive(Clone, Debug)]
struct FusionTreeHomSpaceLayout {
    id: FusionTreeLayoutId,
    data: FusionTreeHomSpaceLayoutData,
}

#[derive(Debug)]
enum PreparedLoweredFusionTreeLayoutState {
    Cached {
        key: FusionTreeHomSpaceCacheKey,
        layout: Arc<FusionTreeHomSpaceLayout>,
    },
    Cold {
        key: FusionTreeHomSpaceCacheKey,
        data: FusionTreeHomSpaceLayoutData,
    },
}

/// Checked fusion-tree metadata staged without publishing process-local state.
///
/// This is an expert transaction boundary for downstream builders that still
/// have fallible shape or storage work. Call [`Self::commit`] only after that
/// work succeeds.
#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedLoweredFusionTreeLayout {
    state: PreparedLoweredFusionTreeLayoutState,
}

impl PreparedLoweredFusionTreeLayout {
    fn cache_key(&self) -> &FusionTreeHomSpaceCacheKey {
        match &self.state {
            PreparedLoweredFusionTreeLayoutState::Cached { key, .. }
            | PreparedLoweredFusionTreeLayoutState::Cold { key, .. } => key,
        }
    }

    fn layout_data(&self) -> &FusionTreeHomSpaceLayoutData {
        match &self.state {
            PreparedLoweredFusionTreeLayoutState::Cached { layout, .. } => layout,
            PreparedLoweredFusionTreeLayoutState::Cold { data, .. } => data,
        }
    }

    pub fn keys(&self) -> &[FusionTreePairKey] {
        match &self.state {
            PreparedLoweredFusionTreeLayoutState::Cached { layout, .. } => layout.keys.as_ref(),
            PreparedLoweredFusionTreeLayoutState::Cold { data, .. } => data.keys.as_ref(),
        }
    }

    pub fn keys_arc(&self) -> Arc<[FusionTreePairKey]> {
        match &self.state {
            PreparedLoweredFusionTreeLayoutState::Cached { layout, .. } => Arc::clone(&layout.keys),
            PreparedLoweredFusionTreeLayoutState::Cold { data, .. } => Arc::clone(&data.keys),
        }
    }

    /// Builds final coupled storage directly from authoritative leg
    /// degeneracies while keeping this prepared layout unpublished.
    pub fn build_from_leg_degeneracies(
        &self,
        homspace: &FusionTreeHomSpace,
    ) -> Result<Arc<BlockStructure>, CoreError> {
        let key = self.cache_key();
        if !fusion_product_space_matches_signature(homspace.codomain(), &key.codomain)
            || !fusion_product_space_matches_signature(homspace.domain(), &key.domain)
        {
            return Err(CoreError::MalformedFusionTree {
                message: "prepared layout does not match HomSpace sector signature",
            });
        }
        // Why not call the cached public builder: downstream validation must
        // finish before this transaction publishes a layout ID or admission,
        // and the prepared data already owns the one checked enumeration.
        // Degeneracies are deliberately absent from the signature: the target
        // HomSpace is their authority, while sectors and duality select keys.
        let (sector, degeneracy) = coupled_subblock_parts_from_layout(
            homspace,
            homspace.codomain().len(),
            self.layout_data(),
            |key| homspace.degeneracy_shape_for_key(key),
        )?;
        BlockStructure::from_parts(sector, degeneracy).map(BlockStructure::into_shared)
    }

    /// Publishes the prepared layout and returns its shared key storage.
    ///
    /// Why no fallible return: checked enumeration and layout-data formation
    /// completed in `prepare`; the remaining cache race check, monotonic ID,
    /// and bounded admission are process-local publication only.
    pub fn commit(self) -> Arc<[FusionTreePairKey]> {
        match self.state {
            PreparedLoweredFusionTreeLayoutState::Cached { key, layout } => {
                let cache = fusion_tree_layout_cache();
                let mut write = cache
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(existing) = write.lookup(&key) {
                    return Arc::clone(&existing.keys);
                }
                let charged_bytes = charged_fusion_tree_layout_bytes(&key, &layout);
                let admitted = write.admit(Arc::new(key), layout, charged_bytes);
                Arc::clone(&admitted.keys)
            }
            PreparedLoweredFusionTreeLayoutState::Cold { key, data } => {
                let cache = fusion_tree_layout_cache();
                let mut write = cache
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(existing) = write.lookup(&key) {
                    return Arc::clone(&existing.keys);
                }
                let computed = Arc::new(FusionTreeHomSpaceLayout {
                    id: next_fusion_tree_layout_id(),
                    data,
                });
                let charged_bytes = charged_fusion_tree_layout_bytes(&key, &computed);
                let admitted = write.admit(Arc::new(key), computed, charged_bytes);
                Arc::clone(&admitted.keys)
            }
        }
    }
}

impl std::ops::Deref for FusionTreeHomSpaceLayout {
    type Target = FusionTreeHomSpaceLayoutData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct FusionTreeLayoutId(usize);

static FUSION_TREE_LAYOUT_ID: AtomicUsize = AtomicUsize::new(1);

#[cfg(test)]
std::thread_local! {
    static FUSION_TREE_LAYOUT_ID_CALLS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static FUSION_TREE_LAYOUT_ADMISSIONS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

#[cfg(test)]
pub(crate) fn reset_fusion_tree_layout_probe_side_effect_calls() {
    FUSION_TREE_LAYOUT_ID_CALLS.set(0);
    FUSION_TREE_LAYOUT_ADMISSIONS.set(0);
}

#[cfg(test)]
pub(crate) fn fusion_tree_layout_probe_side_effect_calls() -> (usize, usize) {
    (
        FUSION_TREE_LAYOUT_ID_CALLS.get(),
        FUSION_TREE_LAYOUT_ADMISSIONS.get(),
    )
}

fn next_fusion_tree_layout_id() -> FusionTreeLayoutId {
    #[cfg(test)]
    FUSION_TREE_LAYOUT_ID_CALLS.set(FUSION_TREE_LAYOUT_ID_CALLS.get() + 1);
    let id = FUSION_TREE_LAYOUT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("fusion-tree layout identity space exhausted");
    FusionTreeLayoutId(id)
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct CoupledBlockStructureCacheKey {
    // Why-not `Arc::as_ptr`: eviction/reset may recycle an address while a
    // coupled structure keyed by that address is still alive. This monotonic
    // process-local id is never recycled, including across cache reset.
    layout: FusionTreeLayoutId,
    nout: usize,
    rank: usize,
    shapes: Arc<[DimVec]>,
}

#[derive(Clone)]
struct FusionTreeLayoutCacheEntry {
    layout: Arc<FusionTreeHomSpaceLayout>,
    charged_bytes: usize,
}

/// Bounded insertion-order cache with a one-entry last-inserted front.
/// Lookups never promote an entry; eviction always removes the oldest admitted
/// entry, so this policy is FIFO rather than LRU.
struct FusionTreeLayoutCache {
    entries: lru::LruCache<
        Arc<FusionTreeHomSpaceCacheKey>,
        FusionTreeLayoutCacheEntry,
        rustc_hash::FxBuildHasher,
    >,
    // Why-not route the repeated hit through LruCache: even `peek` regressed
    // the small-layout gate. This front shares the entry key/value Arcs and is
    // replaced only on insertion; it neither adds pointer authority nor claims
    // to implement read-recency.
    last: Option<(
        Arc<FusionTreeHomSpaceCacheKey>,
        Arc<FusionTreeHomSpaceLayout>,
    )>,
    entry_capacity: usize,
    byte_budget: usize,
    max_entry_bytes: usize,
    charged_payload_bytes: usize,
    misses: usize,
    evictions: usize,
    admission_bypasses: usize,
}

const FUSION_TREE_LAYOUT_CACHE_CAP: usize = 8192;
const FUSION_TREE_LAYOUT_CACHE_BYTE_BUDGET: usize = 64 * 1024 * 1024;
const FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES: usize = 8 * 1024 * 1024;

impl FusionTreeLayoutCache {
    fn new(entry_capacity: usize, byte_budget: usize, max_entry_bytes: usize) -> Self {
        assert!(entry_capacity > 0, "fusion-tree layout cache capacity must be positive");
        Self {
            entries: lru::LruCache::with_hasher(
                std::num::NonZeroUsize::new(entry_capacity).unwrap(),
                rustc_hash::FxBuildHasher,
            ),
            last: None,
            entry_capacity,
            byte_budget,
            max_entry_bytes,
            charged_payload_bytes: 0,
            misses: 0,
            evictions: 0,
            admission_bypasses: 0,
        }
    }

    fn lookup(
        &self,
        key: &FusionTreeHomSpaceCacheKey,
    ) -> Option<Arc<FusionTreeHomSpaceLayout>> {
        if let Some((_, layout)) = self.last.as_ref().filter(|(last, _)| last.as_ref() == key) {
            return Some(Arc::clone(layout));
        }
        self.entries
            .peek(key)
            .map(|entry| Arc::clone(&entry.layout))
    }

    fn admit(
        &mut self,
        key: Arc<FusionTreeHomSpaceCacheKey>,
        layout: Arc<FusionTreeHomSpaceLayout>,
        charged_bytes: usize,
    ) -> Arc<FusionTreeHomSpaceLayout> {
        #[cfg(test)]
        FUSION_TREE_LAYOUT_ADMISSIONS.set(FUSION_TREE_LAYOUT_ADMISSIONS.get() + 1);
        if let Some(existing) = self.entries.peek(key.as_ref()) {
            return Arc::clone(&existing.layout);
        }
        self.misses = self.misses.saturating_add(1);
        if charged_bytes > self.max_entry_bytes || charged_bytes > self.byte_budget {
            self.admission_bypasses = self.admission_bypasses.saturating_add(1);
            return layout;
        }
        while self.entries.len() == self.entry_capacity
            || self
                .charged_payload_bytes
                .saturating_add(charged_bytes)
                > self.byte_budget
        {
            let Some((_, evicted)) = self.entries.pop_lru() else {
                break;
            };
            self.charged_payload_bytes = self
                .charged_payload_bytes
                .saturating_sub(evicted.charged_bytes);
            self.evictions = self.evictions.saturating_add(1);
        }
        self.charged_payload_bytes = self.charged_payload_bytes.saturating_add(charged_bytes);
        self.last = Some((Arc::clone(&key), Arc::clone(&layout)));
        self.entries.put(
            key,
            FusionTreeLayoutCacheEntry {
                layout: Arc::clone(&layout),
                charged_bytes,
            },
        );
        layout
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.last = None;
        self.charged_payload_bytes = 0;
        self.misses = 0;
        self.evictions = 0;
        self.admission_bypasses = 0;
    }

    fn info(&self) -> FusionTreeLayoutCacheInfo {
        FusionTreeLayoutCacheInfo {
            entries: self.entries.len(),
            entry_capacity: self.entry_capacity,
            charged_payload_bytes: self.charged_payload_bytes,
            byte_budget: self.byte_budget,
            max_entry_bytes: self.max_entry_bytes,
            misses: self.misses,
            evictions: self.evictions,
            admission_bypasses: self.admission_bypasses,
        }
    }
}

/// Snapshot of the bounded FIFO fusion-layout cache accounting state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FusionTreeLayoutCacheInfo {
    entries: usize,
    entry_capacity: usize,
    charged_payload_bytes: usize,
    byte_budget: usize,
    max_entry_bytes: usize,
    misses: usize,
    evictions: usize,
    admission_bypasses: usize,
}

impl FusionTreeLayoutCacheInfo {
    pub fn entries(self) -> usize {
        self.entries
    }

    pub fn entry_capacity(self) -> usize {
        self.entry_capacity
    }

    /// Conservative payload charge used for cache admission and eviction.
    ///
    /// This is an accounting contract, not allocator-observed resident bytes.
    pub fn charged_payload_bytes(self) -> usize {
        self.charged_payload_bytes
    }

    pub fn byte_budget(self) -> usize {
        self.byte_budget
    }

    pub fn max_entry_bytes(self) -> usize {
        self.max_entry_bytes
    }

    pub fn misses(self) -> usize {
        self.misses
    }

    pub fn evictions(self) -> usize {
        self.evictions
    }

    pub fn admission_bypasses(self) -> usize {
        self.admission_bypasses
    }
}

fn fusion_tree_layout_cache() -> &'static RwLock<FusionTreeLayoutCache> {
    static CACHE: OnceLock<RwLock<FusionTreeLayoutCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        RwLock::new(FusionTreeLayoutCache::new(
            FUSION_TREE_LAYOUT_CACHE_CAP,
            FUSION_TREE_LAYOUT_CACHE_BYTE_BUDGET,
            FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES,
        ))
    })
}

/// Returns entry and charged-payload bounds for the process-global layout cache.
pub fn fusion_tree_layout_cache_info() -> FusionTreeLayoutCacheInfo {
    let cache = fusion_tree_layout_cache()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache.info()
}

fn charged_fusion_tree_layout_bytes(
    identity: &FusionTreeHomSpaceCacheKey,
    layout: &FusionTreeHomSpaceLayout,
) -> usize {
    let key_bytes = identity
        .codomain
        .iter()
        .chain(identity.domain.iter())
        .fold(
            std::mem::size_of::<FusionTreeHomSpaceCacheKey>()
                .saturating_add(
                    identity
                        .codomain
                        .capacity()
                        .saturating_add(identity.domain.capacity())
                        .saturating_mul(std::mem::size_of::<FusionTreeLegSetSignature>()),
                ),
            |charged, leg| {
                charged.saturating_add(
                    leg.sectors
                        .capacity()
                        .saturating_mul(std::mem::size_of::<SectorId>()),
                )
            },
        )
        .saturating_add(identity.rule.charged_retained_bytes());
    let tree_bytes = layout
        .keys
        .iter()
        .map(|key| {
            [key.codomain_tree(), key.domain_tree()]
                .iter()
                .map(|tree| {
                    std::mem::size_of::<FusionTreeKey>()
                        .saturating_add(
                            tree.uncoupled
                                .capacity()
                                .saturating_mul(std::mem::size_of::<SectorId>()),
                        )
                        .saturating_add(
                            tree.is_dual
                                .capacity()
                                .saturating_mul(std::mem::size_of::<bool>()),
                        )
                        .saturating_add(
                            tree.innerlines
                                .capacity()
                                .saturating_mul(std::mem::size_of::<SectorId>()),
                        )
                        .saturating_add(
                            tree.vertices
                                .capacity()
                                .saturating_mul(std::mem::size_of::<SectorId>()),
                        )
                })
                .fold(0usize, usize::saturating_add)
        })
        .fold(0usize, usize::saturating_add);
    let sector_bytes = layout
        .sectors
        .iter()
        .fold(
            layout
                .sectors
                .capacity()
                .saturating_mul(std::mem::size_of::<FusionTreeCoupledSectorLayout>()),
            |charged, sector| {
                charged.saturating_add(
                    sector
                        .entries
                        .capacity()
                        .saturating_mul(std::mem::size_of::<FusionTreeBlockLayoutEntry>()),
                )
            },
        );
    key_bytes
        .saturating_add(std::mem::size_of::<FusionTreeLayoutCacheEntry>())
        .saturating_add(std::mem::size_of::<FusionTreeHomSpaceLayout>())
        .saturating_add(tree_bytes)
        .saturating_add(sector_bytes)
        .saturating_add(8 * std::mem::size_of::<usize>())
}

type CoupledBlockStructureCache =
    lru::LruCache<CoupledBlockStructureCacheKey, Weak<BlockStructure>, rustc_hash::FxBuildHasher>;

fn coupled_block_structure_cache() -> &'static RwLock<CoupledBlockStructureCache> {
    static CACHE: OnceLock<RwLock<CoupledBlockStructureCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        RwLock::new(lru::LruCache::with_hasher(
            std::num::NonZeroUsize::new(BLOCK_STRUCTURE_INTERN_CAP).unwrap(),
            rustc_hash::FxBuildHasher,
        ))
    })
}

fn reset_fusion_tree_layout_caches() {
    let mut layouts = fusion_tree_layout_cache()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    layouts.clear();
    drop(layouts);
    coupled_block_structure_cache()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

/// Process-global intern id for a fusion hom space. [`FusionTreeHomSpace::id`]
/// deep-hashes the space on first demand (the full generic key: every codomain
/// and domain leg's sectors and dual flag — never a multiplicity-free subset)
/// and stores the collision-safe semantic identity in the hom space's lazy
/// cell. Downstream hashing reads its cached prehash in O(1); equality falls
/// back to the full immutable key only for matching prehashes.
///
/// Why-not (persist to disk): the cached prehash is an implementation detail,
/// so a loaded operation cache reconstructs identity from the semantic space
/// value instead of carrying this process-local acceleration object.
///
/// Why-not (unbounded intern): applications can construct arbitrarily many
/// temporary spaces. The bounded table follows TensorKit's metadata-cache
/// policy. The semantic key remains in each live id, so equal spaces remain
/// equal across eviction; the interner only supplies a pointer-equality fast
/// path while an entry is resident.
#[derive(Clone, Debug)]
pub struct HomSpaceId {
    prehash: u64,
    key: Arc<HomSpaceInternKey>,
}

impl PartialEq for HomSpaceId {
    fn eq(&self, other: &Self) -> bool {
        self.prehash == other.prehash
            && (Arc::ptr_eq(&self.key, &other.key) || self.key == other.key)
    }
}

impl Eq for HomSpaceId {}

impl std::hash::Hash for HomSpaceId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.prehash.hash(state);
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct HomSpaceInternKey {
    codomain: FusionProductSpace,
    domain: FusionProductSpace,
}

#[derive(Clone, Copy)]
struct OrientedLegView<'a> {
    source: &'a SectorLeg,
    dualize: bool,
}

impl<'a> OrientedLegView<'a> {
    fn borrowed(source: &'a SectorLeg) -> Self {
        Self {
            source,
            dualize: false,
        }
    }

    fn toggled(self) -> Self {
        Self {
            source: self.source,
            dualize: !self.dualize,
        }
    }

    fn is_dual(self) -> bool {
        self.source.is_dual() ^ self.dualize
    }

    fn mapped_sector<R>(self, rule: &R, sector: SectorId) -> SectorId
    where
        R: FusionRule,
    {
        if self.dualize {
            rule.dual(sector)
        } else {
            sector
        }
    }

    fn try_mapped_sector<R>(
        self,
        rule: &R,
        sector: SectorId,
    ) -> Result<SectorId, FusionAlgebraError>
    where
        R: CheckedFusionAlgebra,
    {
        if self.dualize {
            rule.try_dual_sector(sector)
        } else {
            Ok(sector)
        }
    }

    fn materialize<R>(self, rule: &R) -> SectorLeg
    where
        R: FusionRule,
    {
        if self.dualize {
            self.source.dual(rule)
        } else {
            self.source.clone()
        }
    }

    fn try_materialize<R>(self, rule: &R) -> Result<SectorLeg, FusionAlgebraError>
    where
        R: CheckedFusionAlgebra,
    {
        if self.dualize {
            self.source.try_dual(rule)
        } else {
            Ok(self.source.clone())
        }
    }
}

struct HomSpaceDescriptor<'a> {
    // Why one vector instead of one per side: rank, not side rank, is the
    // natural inline bound. PEPS/MPS metadata up to rank 8 stays entirely on
    // the stack and `nout` splits the stored-orientation views.
    legs: SmallVec<[OrientedLegView<'a>; 8]>,
    nout: usize,
}

impl<'a> HomSpaceDescriptor<'a> {
    fn new(
        codomain: impl IntoIterator<Item = OrientedLegView<'a>>,
        domain: impl IntoIterator<Item = OrientedLegView<'a>>,
    ) -> Self {
        let mut legs = SmallVec::new();
        legs.extend(codomain);
        let nout = legs.len();
        legs.extend(domain);
        Self { legs, nout }
    }

    fn codomain(&self) -> &[OrientedLegView<'a>] {
        &self.legs[..self.nout]
    }

    fn domain(&self) -> &[OrientedLegView<'a>] {
        &self.legs[self.nout..]
    }

    fn materialize<R>(&self, rule: &R) -> FusionTreeHomSpace
    where
        R: FusionRule,
    {
        FusionTreeHomSpace::new(
            FusionProductSpace::new(
                self.codomain()
                    .iter()
                    .copied()
                    .map(|view| view.materialize(rule)),
            ),
            FusionProductSpace::new(
                self.domain()
                    .iter()
                    .copied()
                    .map(|view| view.materialize(rule)),
            ),
        )
    }

    fn try_materialize<R>(
        &self,
        rule: &R,
    ) -> Result<FusionTreeHomSpace, FusionAlgebraError>
    where
        R: CheckedFusionAlgebra,
    {
        let codomain = self
            .codomain()
            .iter()
            .copied()
            .map(|view| view.try_materialize(rule))
            .collect::<Result<SmallVec<[SectorLeg; 8]>, _>>()?;
        let domain = self
            .domain()
            .iter()
            .copied()
            .map(|view| view.try_materialize(rule))
            .collect::<Result<SmallVec<[SectorLeg; 8]>, _>>()?;
        Ok(FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain),
            FusionProductSpace::new(domain),
        ))
    }
}

struct HomSpaceInternTable {
    entries: lru::LruCache<HomSpaceInternKey, Arc<HomSpaceInternKey>>,
}

const HOM_SPACE_INTERN_CAP: usize = 8192;

#[cfg(test)]
std::thread_local! {
    static HOM_SPACE_INTERN_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_hom_space_intern_calls() {
    HOM_SPACE_INTERN_CALLS.set(0);
}

#[cfg(test)]
pub(crate) fn hom_space_intern_calls() -> usize {
    HOM_SPACE_INTERN_CALLS.get()
}

fn hom_space_intern_table() -> &'static RwLock<HomSpaceInternTable> {
    static TABLE: OnceLock<RwLock<HomSpaceInternTable>> = OnceLock::new();
    TABLE.get_or_init(|| {
        RwLock::new(HomSpaceInternTable {
            entries: lru::LruCache::new(
                std::num::NonZeroUsize::new(HOM_SPACE_INTERN_CAP).unwrap(),
            ),
        })
    })
}

fn intern_hom_space(codomain: &FusionProductSpace, domain: &FusionProductSpace) -> HomSpaceId {
    #[cfg(test)]
    HOM_SPACE_INTERN_CALLS.set(HOM_SPACE_INTERN_CALLS.get() + 1);
    let key = HomSpaceInternKey {
        codomain: codomain.clone(),
        domain: domain.clone(),
    };
    let mut hasher = rustc_hash::FxHasher::default();
    key.hash(&mut hasher);
    let prehash = std::hash::Hasher::finish(&hasher);
    let mut table = hom_space_intern_table()
        .write()
        .expect("hom space intern table poisoned");
    if let Some(canonical) = table.entries.get(&key) {
        return HomSpaceId {
            prehash,
            key: Arc::clone(canonical),
        };
    }
    let canonical = Arc::new(key.clone());
    table.entries.put(key, Arc::clone(&canonical));
    HomSpaceId {
        prehash,
        key: canonical,
    }
}

fn reset_hom_space_intern_table() {
    if let Ok(mut table) = hom_space_intern_table().write() {
        table.entries.clear();
    }
}

fn fusion_tree_layout_from_keys<R>(
    rule: &R,
    id: FusionTreeLayoutId,
    keys: Vec<FusionTreePairKey>,
) -> FusionTreeHomSpaceLayout
where
    R: FusionRule,
{
    FusionTreeHomSpaceLayout {
        id,
        data: fusion_tree_layout_data_from_keys(rule.vacuum(), keys),
    }
}

fn fusion_tree_layout_data_from_keys(
    vacuum: SectorId,
    keys: Vec<FusionTreePairKey>,
) -> FusionTreeHomSpaceLayoutData {
    let keys = Arc::<[FusionTreePairKey]>::from(keys);
    let mut sectors = Vec::new();
    let mut run_start = 0usize;
    while run_start < keys.len() {
        let coupled = keys[run_start].codomain_tree().coupled().unwrap_or(vacuum);
        let mut run_end = run_start;
        let mut row_indices = FxHashMap::<FusionTreeKey, usize>::default();
        let mut col_indices = FxHashMap::<FusionTreeKey, usize>::default();
        let mut entries = Vec::new();
        while run_end < keys.len()
            && keys[run_end].codomain_tree().coupled().unwrap_or(vacuum) == coupled
        {
            let row = match row_indices.get(keys[run_end].codomain_tree()) {
                Some(&index) => index,
                None => {
                    let index = row_indices.len();
                    row_indices.insert(keys[run_end].codomain_tree().clone(), index);
                    index
                }
            };
            let col = match col_indices.get(keys[run_end].domain_tree()) {
                Some(&index) => index,
                None => {
                    let index = col_indices.len();
                    col_indices.insert(keys[run_end].domain_tree().clone(), index);
                    index
                }
            };
            entries.push(FusionTreeBlockLayoutEntry { row, col });
            run_end += 1;
        }
        sectors.push(FusionTreeCoupledSectorLayout {
            start: run_start,
            row_count: row_indices.len(),
            col_count: col_indices.len(),
            entries,
        });
        run_start = run_end;
    }
    FusionTreeHomSpaceLayoutData { keys, sectors }
}

impl FusionTreeHomSpace {
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
            codomain,
            domain,
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
        &self.codomain
    }

    #[inline]
    pub fn domain(&self) -> &FusionProductSpace {
        &self.domain
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.codomain.len() + self.domain.len()
    }

    /// Lazily assigned collision-safe process-local semantic identity.
    /// Hashing is O(1), and equal spaces compare equal across intern eviction.
    #[inline]
    pub fn id(&self) -> HomSpaceId {
        self.id
            .get_or_init(|| intern_hom_space(&self.codomain, &self.domain))
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
        if lhs.domain.len() != rhs.codomain.len() {
            return Err(CoreError::DimensionMismatch {
                expected: lhs.domain.len(),
                actual: rhs.codomain.len(),
            });
        }
        for (lhs_domain, rhs_codomain) in lhs.domain.legs().iter().zip(rhs.codomain.legs()) {
            validate_composed_leg(lhs_domain, rhs_codomain)?;
        }
        let descriptor = HomSpaceDescriptor::new(
            lhs.codomain
                .legs()
                .iter()
                .map(OrientedLegView::borrowed),
            rhs.domain.legs().iter().map(OrientedLegView::borrowed),
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
        let descriptor = tensorcontract_descriptor(
            lhs,
            rhs,
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )?;
        for (&lhs_axis, &rhs_axis) in lhs_contracting_axes.iter().zip(rhs_contracting_axes) {
            validate_oriented_composed_leg(
                rule,
                lhs.external_axis_leg_view(lhs_axis).toggled(),
                rhs.external_axis_leg_view(rhs_axis),
            )?;
        }
        Ok(descriptor.materialize(rule))
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
        let descriptor = tensorcontract_descriptor(
            lhs,
            rhs,
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )?;
        for (&lhs_axis, &rhs_axis) in lhs_contracting_axes.iter().zip(rhs_contracting_axes) {
            validate_oriented_composed_leg_checked(
                rule,
                lhs.external_axis_leg_view(lhs_axis).toggled(),
                rhs.external_axis_leg_view(rhs_axis),
            )?;
        }
        descriptor.try_materialize(rule).map_err(Into::into)
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

    #[doc(hidden)]
    pub fn try_fusion_tree_keys_lowered<R>(
        &self,
        rule: &R,
    ) -> Result<Arc<[FusionTreePairKey]>, LoweredFusionTreeBuildError>
    where
        R: LoweredMultiplicityFreeAlgebra,
    {
        Ok(self.prepare_fusion_tree_layout_lowered(rule)?.commit())
    }

    /// Stages checked keys and coupled-sector metadata without issuing a
    /// layout ID or changing cache admission/accounting.
    #[doc(hidden)]
    pub fn prepare_fusion_tree_layout_lowered<R>(
        &self,
        rule: &R,
    ) -> Result<PreparedLoweredFusionTreeLayout, LoweredFusionTreeBuildError>
    where
        R: LoweredMultiplicityFreeAlgebra,
    {
        let key = FusionTreeHomSpaceCacheKey::new(rule, self);
        let cache = fusion_tree_layout_cache();
        let read = cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(layout) = read.lookup(&key) {
            return Ok(PreparedLoweredFusionTreeLayout {
                state: PreparedLoweredFusionTreeLayoutState::Cached { key, layout },
            });
        }
        drop(read);

        let keys = self.try_fusion_tree_keys_uncached_lowered(rule)?;
        let vacuum = rule
            .try_lowered_vacuum()
            .and_then(|sector| rule.try_encode_lowered(sector))?;
        let data = fusion_tree_layout_data_from_keys(vacuum, keys);
        Ok(PreparedLoweredFusionTreeLayout {
            state: PreparedLoweredFusionTreeLayoutState::Cold {
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
        self.try_cached_fusion_tree_layout_with(rule, || {
            Ok::<_, std::convert::Infallible>(self.fusion_tree_keys_uncached(rule))
        })
        .unwrap_or_else(|error| match error {})
    }

    #[cfg(test)]
    fn try_cached_fusion_tree_layout_lowered<R>(
        &self,
        rule: &R,
    ) -> Result<Arc<FusionTreeHomSpaceLayout>, LoweredFusionTreeBuildError>
    where
        R: LoweredMultiplicityFreeAlgebra,
    {
        self.prepare_fusion_tree_layout_lowered(rule)?.commit();
        let key = FusionTreeHomSpaceCacheKey::new(rule, self);
        let cache = fusion_tree_layout_cache();
        let read = cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(read
            .lookup(&key)
            .expect("committed lowered layout must be cache-resident"))
    }

    fn try_cached_fusion_tree_layout_with<R, E, F>(
        &self,
        rule: &R,
        build: F,
    ) -> Result<Arc<FusionTreeHomSpaceLayout>, E>
    where
        R: MultiplicityFreeFusionRule,
        F: FnOnce() -> Result<Vec<FusionTreePairKey>, E>,
    {
        let key = FusionTreeHomSpaceCacheKey::new(rule, self);
        let cache = fusion_tree_layout_cache();
        let read = cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(layout) = read.lookup(&key) {
            return Ok(layout);
        }
        drop(read);

        let key = Arc::new(key);
        let keys = build()?;
        let computed = Arc::new(fusion_tree_layout_from_keys(
            rule,
            next_fusion_tree_layout_id(),
            keys,
        ));
        let charged_bytes = charged_fusion_tree_layout_bytes(&key, &computed);
        let mut write = cache
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(write.admit(key, computed, charged_bytes))
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
    /// materialize one owned shape per tree. The miss builder evaluates each
    /// tree shape once and moves it directly into the final degeneracy
    /// structure.
    pub fn coupled_subblock_structure_from_leg_degeneracies<R>(
        &self,
        rule: &R,
    ) -> Result<Arc<BlockStructure>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let layout = self.cached_fusion_tree_layout(rule);
        coupled_subblock_structure_from_layout(
            self,
            self.codomain.len(),
            &layout,
            |key| self.degeneracy_shape_for_key(key),
        )
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
        let layout =
            fusion_tree_layout_data_from_keys(rule.vacuum(), self.fusion_tree_keys_uncached(rule));
        let (sector, degeneracy) = coupled_subblock_parts_from_layout(
            self,
            self.codomain.len(),
            &layout,
            |key| self.degeneracy_shape_for_key(key),
        )?;
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
        let keys = self.fusion_tree_keys_generic(rule)?;
        let layout = fusion_tree_layout_from_keys(rule, next_fusion_tree_layout_id(), keys);
        coupled_subblock_structure_from_layout(
            self,
            self.codomain.len(),
            &layout,
            |key| self.degeneracy_shape_for_key(key),
        )
    }

    fn degeneracy_shape_for_key(
        &self,
        key: &FusionTreePairKey,
    ) -> Result<DimVec, CoreError> {
        let rank = self.rank();
        if key.codomain_uncoupled().len() != self.codomain.len()
            || key.domain_uncoupled().len() != self.domain.len()
        {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: key.codomain_uncoupled().len() + key.domain_uncoupled().len(),
            });
        }
        let mut shape = DimVec::new();
        for (leg, &sector) in self
            .codomain
            .legs()
            .iter()
            .chain(self.domain.legs())
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

    fn fusion_tree_keys_uncached<R>(&self, rule: &R) -> Vec<FusionTreePairKey>
    where
        R: MultiplicityFreeFusionRule,
    {
        let codomain = fusion_trees_by_coupled_for_space(rule, &self.codomain);
        let domain = fusion_trees_by_coupled_for_space(rule, &self.domain);
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

    fn try_fusion_tree_keys_uncached_lowered<R>(
        &self,
        rule: &R,
    ) -> Result<Vec<FusionTreePairKey>, LoweredFusionTreeBuildError>
    where
        R: LoweredMultiplicityFreeAlgebra,
    {
        let codomain = try_fusion_trees_by_coupled_for_space_lowered(rule, &self.codomain)?;
        let domain = try_fusion_trees_by_coupled_for_space_lowered(rule, &self.domain)?;
        Ok(merge_generic_tree_groups(&codomain, &domain))
    }

    /// Generic-fusion (outer-multiplicity) sibling of [`Self::fusion_tree_keys`]:
    /// enumerates multiplicity-aware block keys (codomain × domain tree pairs)
    /// for a `FusionRule` whose `nsymbol` can exceed 1 (SU(3), …). Not cached —
    /// the Generic path is not on any hot loop yet (mirrors the non-memoized
    /// Stage B3b cache siblings); add a cache when a Generic workload measures it.
    ///
    /// Bounded-table semantics (Option A, refute/b3b-verify): the result is
    /// either the EXACT full-SU(3) block key set or an `Err` — never a silently
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
        let (codomain, codomain_fold) =
            fusion_trees_by_coupled_for_space_generic(rule, &self.codomain);
        let (domain, domain_fold) = fusion_trees_by_coupled_for_space_generic(rule, &self.domain);
        for (side, fold) in [("codomain", &codomain_fold), ("domain", &domain_fold)] {
            if !fold.is_fully_clean() {
                return Err(CoreError::FusionOutsideTable {
                    message: fusion_fold_error_message(side, fold),
                });
            }
        }
        Ok(merge_generic_tree_groups(&codomain, &domain))
    }

    /// Block keys of ONE coupled sector, for spaces whose full enumeration is
    /// an `Err` but whose requested sector is provably clean (its complete
    /// full-SU(3) tree set stays inside the table). `Err` when the sector is
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
            fusion_trees_by_coupled_for_space_generic(rule, &self.codomain);
        let (domain, domain_fold) = fusion_trees_by_coupled_for_space_generic(rule, &self.domain);
        for (side, fold) in [("codomain", &codomain_fold), ("domain", &domain_fold)] {
            if fold.poisoned || fold.tainted.contains(&coupled) {
                return Err(CoreError::FusionOutsideTable {
                    message: format!(
                        "SU(3) coupled sector {coupled:?} on the {side} side requires                          out-of-table intermediates (dim<=27 cut); extend the table                          (Stage B3c). {}",
                        fusion_fold_error_message(side, fold)
                    ),
                });
            }
        }
        let pick = |groups: &[CoupledFusionTrees]| -> Vec<FusionTreeKey> {
            groups
                .iter()
                .find(|group| group.coupled == coupled)
                .map(|group| group.trees.clone())
                .unwrap_or_default()
        };
        let codomain_trees = pick(&codomain);
        let domain_trees = pick(&domain);
        let mut keys = Vec::with_capacity(codomain_trees.len() * domain_trees.len());
        for domain_tree in &domain_trees {
            for codomain_tree in &codomain_trees {
                keys.push(FusionTreePairKey::pair(
                    codomain_tree.clone(),
                    domain_tree.clone(),
                ));
            }
        }
        Ok(keys)
    }

    pub fn sector_structure<R>(&self, rule: &R) -> Result<SectorStructure, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let rank = self.codomain.len() + self.domain.len();
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
        let rank = self.codomain.len() + self.domain.len();
        if sectors.len() != rank {
            return Err(CoreError::DimensionMismatch {
                expected: rank,
                actual: sectors.len(),
            });
        }

        let codomain = fusion_trees_by_coupled_for_selected_space(
            rule,
            &self.codomain,
            &sectors[..self.codomain.len()],
        )?;
        let domain_sectors = sectors[self.codomain.len()..]
            .iter()
            .map(|&sector| rule.dual(sector))
            .collect::<Vec<_>>();
        let domain =
            fusion_trees_by_coupled_for_selected_space(rule, &self.domain, &domain_sectors)?;
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
            .codomain
            .legs()
            .iter()
            .chain(self.domain.legs())
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

    pub fn fusion_tree_groups<R>(&self, rule: &R) -> Result<Vec<FusionTreeBlockGroup>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        self.sector_structure(rule)
            .map(|structure| structure.fusion_tree_groups())
    }

    /// External leg view of flat axis `axis` (TensorKit's `space(t, i)`
    /// convention, homspace.jl:60-62): codomain legs verbatim, domain legs
    /// dualized. Degeneracies are carried along, keyed by the external
    /// (placement-invariant) sector labels.
    pub fn external_axis_leg<R>(&self, rule: &R, axis: usize) -> SectorLeg
    where
        R: FusionRule,
    {
        if axis < self.codomain.len() {
            self.codomain.legs()[axis].clone()
        } else {
            dual_sector_leg(rule, &self.domain.legs()[axis - self.codomain.len()])
        }
    }

    fn external_axis_leg_view(&self, axis: usize) -> OrientedLegView<'_> {
        if axis < self.codomain.len() {
            OrientedLegView::borrowed(&self.codomain.legs()[axis])
        } else {
            OrientedLegView::borrowed(&self.domain.legs()[axis - self.codomain.len()]).toggled()
        }
    }

    /// Returns the duality flag in the external-axis convention. Why not read
    /// a domain leg verbatim: external domain axes are duals of their stored
    /// hom-space legs, matching [`Self::external_axis_leg`].
    pub fn external_axis_is_dual(&self, axis: usize) -> Option<bool> {
        if axis < self.codomain.len() {
            Some(self.codomain.legs()[axis].is_dual())
        } else if axis < self.rank() {
            Some(!self.domain.legs()[axis - self.codomain.len()].is_dual())
        } else {
            None
        }
    }
}

fn dual_sector_leg<R>(rule: &R, leg: &SectorLeg) -> SectorLeg
where
    R: FusionRule,
{
    leg.dual(rule)
}

#[allow(clippy::too_many_arguments)]
fn tensorcontract_descriptor<'a>(
    lhs: &'a FusionTreeHomSpace,
    rhs: &'a FusionTreeHomSpace,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
    output_axes: &[usize],
    dst_codomain_rank: usize,
) -> Result<HomSpaceDescriptor<'a>, CoreError> {
    if lhs_contracting_axes.len() != rhs_contracting_axes.len() {
        return Err(CoreError::DimensionMismatch {
            expected: lhs_contracting_axes.len(),
            actual: rhs_contracting_axes.len(),
        });
    }

    let lhs_seen = validate_axis_subset_inline(lhs_contracting_axes, lhs.rank())?;
    let rhs_seen = validate_axis_subset_inline(rhs_contracting_axes, rhs.rank())?;
    let lhs_open_axes = (0..lhs.rank())
        .filter(|&axis| !lhs_seen[axis])
        .collect::<SmallVec<[usize; 8]>>();
    let rhs_open_axes = (0..rhs.rank())
        .filter(|&axis| !rhs_seen[axis])
        .collect::<SmallVec<[usize; 8]>>();
    let output_rank = lhs_open_axes.len() + rhs_open_axes.len();
    validate_permutation_inline(output_axes, output_rank)?;
    if dst_codomain_rank > output_rank {
        return Err(CoreError::StructureRankMismatch {
            expected: output_rank,
            actual: dst_codomain_rank,
        });
    }

    let mut open_legs = SmallVec::<[OrientedLegView<'a>; 8]>::new();
    open_legs.extend(
        lhs_open_axes
            .iter()
            .map(|&axis| lhs.external_axis_leg_view(axis)),
    );
    open_legs.extend(
        rhs_open_axes
            .iter()
            .map(|&axis| rhs.external_axis_leg_view(axis)),
    );
    // Why not materialize two permuted operands and their composition: output
    // ordering observes only these final external views, and doing so would
    // repeat orientation arithmetic before the final HomSpace exists.
    let descriptor = HomSpaceDescriptor::new(
        output_axes[..dst_codomain_rank]
            .iter()
            .map(|&axis| open_legs[axis]),
        output_axes[dst_codomain_rank..]
            .iter()
            .map(|&axis| open_legs[axis].toggled()),
    );
    Ok(descriptor)
}

fn validate_axis_subset_inline(
    axes: &[usize],
    rank: usize,
) -> Result<SmallVec<[bool; 8]>, CoreError> {
    let mut seen = SmallVec::<[bool; 8]>::new();
    seen.resize(rank, false);
    for &axis in axes {
        if axis >= rank || seen[axis] {
            return Err(CoreError::InvalidPermutation {
                permutation: axes.to_vec(),
                rank,
            });
        }
        seen[axis] = true;
    }
    Ok(seen)
}

fn validate_permutation_inline(permutation: &[usize], rank: usize) -> Result<(), CoreError> {
    if permutation.len() != rank {
        return Err(CoreError::InvalidPermutation {
            permutation: permutation.to_vec(),
            rank,
        });
    }
    validate_axis_subset_inline(permutation, rank).map(|_| ())
}

fn validate_axis_selection(
    codomain_axes: &[usize],
    domain_axes: &[usize],
    rank: usize,
) -> Result<(), CoreError> {
    for &axis in codomain_axes.iter().chain(domain_axes) {
        if axis >= rank {
            let mut axes = Vec::with_capacity(codomain_axes.len() + domain_axes.len());
            axes.extend_from_slice(codomain_axes);
            axes.extend_from_slice(domain_axes);
            return Err(CoreError::InvalidPermutation {
                permutation: axes,
                rank,
            });
        }
    }
    Ok(())
}

fn validate_composed_leg(
    lhs_domain: &SectorLeg,
    rhs_codomain: &SectorLeg,
) -> Result<(), CoreError> {
    if lhs_domain.is_dual() != rhs_codomain.is_dual() {
        return Err(CoreError::MalformedFusionTree {
            message: "contracted fusion leg duality flags do not match",
        });
    }
    // TensorKit parity: `A * B` requires `domain(A) == codomain(B)` as
    // spaces, so the stored legs must match verbatim (domain legs store the
    // domain space's own sectors; verified against TensorKit v0.16:
    // `rand(V ← V) * rand(V ← V)` works for V = U1Space(0=>1, 1=>1), a
    // sector set that is not dualization-closed, while `(V ← V) * (? ← V')`
    // is a SpaceMismatch).
    if lhs_domain.sectors().len() != rhs_codomain.sectors().len() {
        return Err(CoreError::DimensionMismatch {
            expected: lhs_domain.sectors().len(),
            actual: rhs_codomain.sectors().len(),
        });
    }
    for ((expected, expected_deg), (actual, actual_deg)) in
        lhs_domain.iter().zip(rhs_codomain.iter())
    {
        if expected != actual {
            return Err(CoreError::SectorMismatch { expected, actual });
        }
        if expected_deg != actual_deg {
            return Err(CoreError::LegDegeneracyMismatch {
                sector: expected,
                expected: expected_deg,
                actual: actual_deg,
            });
        }
    }
    Ok(())
}

fn validate_oriented_composed_leg<R>(
    rule: &R,
    lhs_domain: OrientedLegView<'_>,
    rhs_codomain: OrientedLegView<'_>,
) -> Result<(), CoreError>
where
    R: FusionRule,
{
    let valid = lhs_domain.is_dual() == rhs_codomain.is_dual()
        && lhs_domain.source.sectors().len() == rhs_codomain.source.sectors().len()
        && lhs_domain.source.iter().all(|(sector, degeneracy)| {
            let expected = lhs_domain.mapped_sector(rule, sector);
            let rhs_source_sector = if rhs_codomain.dualize {
                rule.dual(expected)
            } else {
                expected
            };
            rhs_codomain.source.degeneracy(rhs_source_sector) == Some(degeneracy)
        });
    if valid {
        return Ok(());
    }
    // Preserve the historical error variant and first mismatching sector only
    // on an invalid request. The valid hot path never sorts or rebuilds a leg.
    validate_composed_leg(
        &lhs_domain.materialize(rule),
        &rhs_codomain.materialize(rule),
    )
}

fn validate_oriented_composed_leg_checked<R>(
    rule: &R,
    lhs_domain: OrientedLegView<'_>,
    rhs_codomain: OrientedLegView<'_>,
) -> Result<(), CheckedFusionSpaceError>
where
    R: CheckedFusionAlgebra,
{
    if lhs_domain.is_dual() != rhs_codomain.is_dual() {
        return Err(CoreError::MalformedFusionTree {
            message: "contracted fusion leg duality flags do not match",
        }
        .into());
    }
    if lhs_domain.source.sectors().len() != rhs_codomain.source.sectors().len() {
        return Err(CoreError::DimensionMismatch {
            expected: lhs_domain.source.sectors().len(),
            actual: rhs_codomain.source.sectors().len(),
        }
        .into());
    }
    let mut valid = true;
    for (sector, degeneracy) in lhs_domain.source.iter() {
        let expected = lhs_domain.try_mapped_sector(rule, sector)?;
        let rhs_source_sector = if rhs_codomain.dualize {
            rule.try_dual_sector(expected)?
        } else {
            expected
        };
        if rhs_codomain.source.degeneracy(rhs_source_sector) != Some(degeneracy) {
            valid = false;
            break;
        }
    }
    if valid {
        return Ok(());
    }
    // Why not fall back to the infallible materializer: malformed-space error
    // detail must not turn a representability failure into an unwind.
    validate_composed_leg(
        &lhs_domain.try_materialize(rule)?,
        &rhs_codomain.try_materialize(rule)?,
    )
    .map_err(Into::into)
}

fn coupled_subblock_structure_from_layout<F>(
    homspace: &FusionTreeHomSpace,
    nout: usize,
    layout: &FusionTreeHomSpaceLayout,
    shape_for_key: F,
) -> Result<Arc<BlockStructure>, CoreError>
where
    F: FnMut(&FusionTreePairKey) -> Result<DimVec, CoreError>,
{
    let (sector, degeneracy) =
        coupled_subblock_parts_from_layout(homspace, nout, layout, shape_for_key)?;
    BlockStructure::from_parts(sector, degeneracy).map(BlockStructure::into_shared)
}

fn coupled_subblock_parts_from_layout<F>(
    homspace: &FusionTreeHomSpace,
    nout: usize,
    layout: &FusionTreeHomSpaceLayoutData,
    mut shape_for_key: F,
) -> Result<(SectorStructure, DegeneracyStructure), CoreError>
where
    F: FnMut(&FusionTreePairKey) -> Result<DimVec, CoreError>,
{
    let rank = homspace.rank();
    if nout > rank {
        return Err(CoreError::StructureRankMismatch {
            expected: rank,
            actual: nout,
        });
    }

    let mut degeneracy_blocks = Vec::with_capacity(layout.keys.len());
    let mut sector_offset = 0usize;
    for sector in &layout.sectors {
        if sector.entries.len() != sector.row_count * sector.col_count {
            return Err(CoreError::BlockCountMismatch {
                expected: sector.row_count * sector.col_count,
                actual: sector.entries.len(),
            });
        }

        let mut shapes = Vec::with_capacity(sector.entries.len());
        let mut row_dims = vec![None; sector.row_count];
        let mut col_dims = vec![None; sector.col_count];
        for (local_index, entry) in sector.entries.iter().enumerate() {
            let shape = shape_for_key(&layout.keys[sector.start + local_index])?;
            if shape.len() != rank {
                return Err(CoreError::StructureRankMismatch {
                    expected: rank,
                    actual: shape.len(),
                });
            }
            register_layout_dim(
                &mut row_dims[entry.row],
                shape[..nout].iter().try_fold(1usize, |product, &dim| {
                    product
                        .checked_mul(dim)
                        .ok_or(CoreError::ElementCountOverflow)
                })?,
            )?;
            register_layout_dim(
                &mut col_dims[entry.col],
                shape[nout..].iter().try_fold(1usize, |product, &dim| {
                    product
                        .checked_mul(dim)
                        .ok_or(CoreError::ElementCountOverflow)
                })?,
            )?;
            shapes.push(shape);
        }

        let row_dims = row_dims
            .into_iter()
            .map(|dim| {
                dim.ok_or(CoreError::MalformedFusionTree {
                    message: "fusion tree layout has an empty row",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let col_dims = col_dims
            .into_iter()
            .map(|dim| {
                dim.ok_or(CoreError::MalformedFusionTree {
                    message: "fusion tree layout has an empty column",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let row_offsets = prefix_offsets(&row_dims)?;
        let col_offsets = prefix_offsets(&col_dims)?;
        let matrix_rows = match row_offsets.last().zip(row_dims.last()) {
            Some((&offset, &dim)) => offset
                .checked_add(dim)
                .ok_or(CoreError::ElementCountOverflow)?,
            None => 0,
        };
        let matrix_cols = match col_offsets.last().zip(col_dims.last()) {
            Some((&offset, &dim)) => offset
                .checked_add(dim)
                .ok_or(CoreError::ElementCountOverflow)?,
            None => 0,
        };

        for (shape, entry) in shapes.into_iter().zip(&sector.entries) {
            let mut strides = DimVec::new();
            let mut stride = 1usize;
            for &dim in &shape[..nout] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let mut stride = matrix_rows;
            for &dim in &shape[nout..] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let offset = sector_offset
                .checked_add(row_offsets[entry.row])
                .and_then(|offset| {
                    matrix_rows
                        .checked_mul(col_offsets[entry.col])
                        .and_then(|column| offset.checked_add(column))
                })
                .ok_or(CoreError::ElementCountOverflow)?;
            degeneracy_blocks.push(DegeneracyBlock::new(shape, strides, offset)?);
        }

        sector_offset = sector_offset
            .checked_add(
                matrix_rows
                    .checked_mul(matrix_cols)
                    .ok_or(CoreError::ElementCountOverflow)?,
            )
            .ok_or(CoreError::ElementCountOverflow)?;
    }

    let sector_structure =
        SectorStructure::from_keys(rank, layout.keys.iter().cloned().map(BlockKey::from))?;
    let degeneracy_structure =
        DegeneracyStructure::from_blocks_with_rank(rank, degeneracy_blocks)?;
    Ok((sector_structure, degeneracy_structure))
}

/// Computes coupled-sector matrix block specs for fusion-tree subblocks.
///
/// Keys must arrive grouped by coupled sector (the `fusion_tree_keys`
/// enumeration order). Within one coupled sector every codomain tree defines a
/// row block and every domain tree a column block of one column-major sector
/// matrix; the subblock for `(codomain tree, domain tree)` is the strided view
/// at that (row block, column block) position. Full coverage of the
/// `rows × columns` grid is required so the sector matrix has no
/// uninitialized holes.
fn coupled_sector_matrix_block_specs<R, S>(
    rule: &R,
    nout: usize,
    rank: usize,
    keys: &[FusionTreePairKey],
    shapes: &[S],
) -> Result<Vec<BlockSpec>, CoreError>
where
    R: FusionRule,
    S: AsRef<[usize]>,
{
    for shape in shapes {
        let shape = shape.as_ref();
        if shape.len() != rank {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
    }

    let mut specs = Vec::with_capacity(keys.len());
    let mut seen_sectors: Vec<SectorId> = Vec::new();
    let mut sector_offset = 0usize;
    let mut run_start = 0usize;
    while run_start < keys.len() {
        let coupled = coupled_or_vacuum(rule, keys[run_start].codomain_tree());
        if seen_sectors.contains(&coupled) {
            return Err(CoreError::MalformedFusionTree {
                message: "coupled sectors must be contiguous in fusion tree key order",
            });
        }
        seen_sectors.push(coupled);
        let mut run_end = run_start;
        while run_end < keys.len()
            && coupled_or_vacuum(rule, keys[run_end].codomain_tree()) == coupled
        {
            if coupled_or_vacuum(rule, keys[run_end].domain_tree()) != coupled {
                return Err(CoreError::MalformedFusionTree {
                    message: "codomain and domain trees must share the coupled sector",
                });
            }
            run_end += 1;
        }

        // Row/column blocks keep first-seen order (offsets are cumulative), with
        // a hash side-index for O(1) tree lookup instead of a linear scan: a run
        // can hold many blocks, so the scan was O(run^1.5).
        let mut row_blocks: Vec<(&FusionTreeKey, usize, usize)> = Vec::new();
        let mut col_blocks: Vec<(&FusionTreeKey, usize, usize)> = Vec::new();
        let mut row_index: FxHashMap<&FusionTreeKey, usize> = FxHashMap::default();
        let mut col_index: FxHashMap<&FusionTreeKey, usize> = FxHashMap::default();
        for index in run_start..run_end {
            let key = &keys[index];
            let shape = shapes[index].as_ref();
            let row_dim = shape[..nout].iter().product::<usize>();
            let col_dim = shape[nout..].iter().product::<usize>();
            match row_index.get(key.codomain_tree()).copied() {
                Some(existing_index) if row_blocks[existing_index].2 != row_dim => {
                    return Err(CoreError::DimensionMismatch {
                        expected: row_blocks[existing_index].2,
                        actual: row_dim,
                    });
                }
                Some(_) => {}
                None => {
                    let offset = row_blocks
                        .last()
                        .map(|(_, start, dim)| start + dim)
                        .unwrap_or(0);
                    row_index.insert(key.codomain_tree(), row_blocks.len());
                    row_blocks.push((key.codomain_tree(), offset, row_dim));
                }
            }
            match col_index.get(key.domain_tree()).copied() {
                Some(existing_index) if col_blocks[existing_index].2 != col_dim => {
                    return Err(CoreError::DimensionMismatch {
                        expected: col_blocks[existing_index].2,
                        actual: col_dim,
                    });
                }
                Some(_) => {}
                None => {
                    let offset = col_blocks
                        .last()
                        .map(|(_, start, dim)| start + dim)
                        .unwrap_or(0);
                    col_index.insert(key.domain_tree(), col_blocks.len());
                    col_blocks.push((key.domain_tree(), offset, col_dim));
                }
            }
        }
        if run_end - run_start != row_blocks.len() * col_blocks.len() {
            return Err(CoreError::BlockCountMismatch {
                expected: row_blocks.len() * col_blocks.len(),
                actual: run_end - run_start,
            });
        }
        let matrix_rows = row_blocks
            .last()
            .map(|(_, start, dim)| start + dim)
            .unwrap_or(0);
        let matrix_cols = col_blocks
            .last()
            .map(|(_, start, dim)| start + dim)
            .unwrap_or(0);

        for index in run_start..run_end {
            let key = &keys[index];
            let shape = shapes[index].as_ref();
            let row_start = row_blocks[row_index
                .get(key.codomain_tree())
                .copied()
                .expect("row block registered above")]
            .1;
            let col_start = col_blocks[col_index
                .get(key.domain_tree())
                .copied()
                .expect("column block registered above")]
            .1;
            let mut strides = Vec::with_capacity(rank);
            let mut stride = 1usize;
            for &dim in &shape[..nout] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let mut stride = matrix_rows;
            for &dim in &shape[nout..] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let offset = sector_offset
                + row_start
                + matrix_rows
                    .checked_mul(col_start)
                    .ok_or(CoreError::ElementCountOverflow)?;
            specs.push(BlockSpec::with_key(
                BlockKey::FusionTree(key.clone()),
                shape.to_vec(),
                strides,
                offset,
            )?);
        }

        sector_offset = sector_offset
            .checked_add(
                matrix_rows
                    .checked_mul(matrix_cols)
                    .ok_or(CoreError::ElementCountOverflow)?,
            )
            .ok_or(CoreError::ElementCountOverflow)?;
        run_start = run_end;
    }
    Ok(specs)
}

fn coupled_sector_matrix_block_specs_from_layout<S>(
    nout: usize,
    rank: usize,
    layout: &FusionTreeHomSpaceLayout,
    shapes: &[S],
) -> Result<Vec<BlockSpec>, CoreError>
where
    S: AsRef<[usize]>,
{
    let keys = layout.keys.as_ref();
    if keys.len() != shapes.len() {
        return Err(CoreError::BlockCountMismatch {
            expected: keys.len(),
            actual: shapes.len(),
        });
    }
    for shape in shapes {
        let shape = shape.as_ref();
        if shape.len() != rank {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
    }

    let mut specs = Vec::with_capacity(keys.len());
    let mut sector_offset = 0usize;
    for sector in &layout.sectors {
        if sector.entries.len() != sector.row_count * sector.col_count {
            return Err(CoreError::BlockCountMismatch {
                expected: sector.row_count * sector.col_count,
                actual: sector.entries.len(),
            });
        }

        let mut row_dims = vec![None; sector.row_count];
        let mut col_dims = vec![None; sector.col_count];
        for (local_index, entry) in sector.entries.iter().enumerate() {
            let shape = shapes[sector.start + local_index].as_ref();
            register_layout_dim(&mut row_dims[entry.row], shape[..nout].iter().product())?;
            register_layout_dim(&mut col_dims[entry.col], shape[nout..].iter().product())?;
        }

        let row_dims = row_dims
            .into_iter()
            .map(|dim| {
                dim.ok_or(CoreError::MalformedFusionTree {
                    message: "cached fusion tree layout has an empty row",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let col_dims = col_dims
            .into_iter()
            .map(|dim| {
                dim.ok_or(CoreError::MalformedFusionTree {
                    message: "cached fusion tree layout has an empty column",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let row_offsets = prefix_offsets(&row_dims)?;
        let col_offsets = prefix_offsets(&col_dims)?;
        let matrix_rows = row_offsets
            .last()
            .zip(row_dims.last())
            .map(|(&offset, &dim)| offset + dim)
            .unwrap_or(0);
        let matrix_cols = col_offsets
            .last()
            .zip(col_dims.last())
            .map(|(&offset, &dim)| offset + dim)
            .unwrap_or(0);

        for (local_index, entry) in sector.entries.iter().enumerate() {
            let index = sector.start + local_index;
            let shape = shapes[index].as_ref();
            let mut strides = Vec::with_capacity(rank);
            let mut stride = 1usize;
            for &dim in &shape[..nout] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let mut stride = matrix_rows;
            for &dim in &shape[nout..] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let offset = sector_offset
                + row_offsets[entry.row]
                + matrix_rows
                    .checked_mul(col_offsets[entry.col])
                    .ok_or(CoreError::ElementCountOverflow)?;
            specs.push(BlockSpec::with_key(
                BlockKey::FusionTree(keys[index].clone()),
                shape.to_vec(),
                strides,
                offset,
            )?);
        }

        sector_offset = sector_offset
            .checked_add(
                matrix_rows
                    .checked_mul(matrix_cols)
                    .ok_or(CoreError::ElementCountOverflow)?,
            )
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    Ok(specs)
}

fn register_layout_dim(slot: &mut Option<usize>, dim: usize) -> Result<(), CoreError> {
    match slot {
        Some(existing) if *existing != dim => Err(CoreError::DimensionMismatch {
            expected: *existing,
            actual: dim,
        }),
        Some(_) => Ok(()),
        None => {
            *slot = Some(dim);
            Ok(())
        }
    }
}

fn prefix_offsets(dims: &[usize]) -> Result<Vec<usize>, CoreError> {
    let mut offsets = Vec::with_capacity(dims.len());
    let mut offset = 0usize;
    for &dim in dims {
        offsets.push(offset);
        offset = offset
            .checked_add(dim)
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    Ok(offsets)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionTensorMapSpace<const NOUT: usize, const NIN: usize> {
    dense_space: TensorMapSpace<NOUT, NIN>,
    homspace: Arc<FusionTreeHomSpace>,
    subblock_structure: Arc<BlockStructure>,
    rule_identity: Option<RuleIdentity>,
}

impl<const NOUT: usize, const NIN: usize> FusionTensorMapSpace<NOUT, NIN> {
    /// Builds a symmetric tensor-map space from an explicit block structure.
    ///
    /// Prefer [`Self::from_degeneracy_shapes`] for ordinary construction; use
    /// this method when the block structure has already been prepared.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     BlockStructure, FusionTensorMapSpace, FusionTreeHomSpace, TensorMapSpace,
    /// };
    ///
    /// let dense = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    /// let hom = FusionTreeHomSpace::from_sector_ids([(0, 2)], std::iter::empty::<(usize, usize)>());
    /// let structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
    ///
    /// let space = FusionTensorMapSpace::new_unbound(dense, hom, structure).unwrap();
    /// assert_eq!(space.required_len().unwrap(), 2);
    /// ```
    pub fn new_unbound(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        subblock_structure: BlockStructure,
    ) -> Result<Self, CoreError> {
        Self::from_shared_subblock_structure(
            dense_space,
            homspace,
            subblock_structure.into_shared(),
        )
    }

    pub fn from_shared_subblock_structure(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        subblock_structure: Arc<BlockStructure>,
    ) -> Result<Self, CoreError> {
        Self::validate_homspace_rank(&homspace)?;
        let rank = NOUT + NIN;
        if subblock_structure.rank() != rank {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: subblock_structure.rank(),
            });
        }
        let subblock_structure = BlockStructure::canonicalize_shared(subblock_structure);
        Ok(Self {
            dense_space,
            homspace: Arc::new(homspace),
            subblock_structure,
            rule_identity: None,
        })
    }

    /// Default constructor: TensorKit-equivalent coupled-sector matrix
    /// layout (see [`Self::from_degeneracy_shapes_coupled`]). This is the
    /// only product layout.
    ///
    /// The shapes are given per fusion-tree **subblock** (one entry per
    /// fusion-tree key, in key order), not per coupled-sector matrix block.
    /// This mirrors TensorKit's block/subblock distinction: a coupled-sector
    /// matrix block is assembled from these tree-resolved degeneracy shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionTensorMapSpace, FusionTreeHomSpace, TensorMapSpace, Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let space = FusionTensorMapSpace::from_degeneracy_shapes(
    ///     TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
    ///     FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::EVEN, 1)]),
    ///     &rule,
    ///     [vec![1, 1]],
    /// )
    /// .unwrap();
    /// assert_eq!(space.required_len().unwrap(), 1);
    /// ```
    pub fn from_degeneracy_shapes<R, Shapes>(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        rule: &R,
        shapes: Shapes,
    ) -> Result<Self, CoreError>
    where
        R: MultiplicityFreeFusionRule,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        Self::from_degeneracy_shapes_coupled(dense_space, homspace, rule, shapes)
    }

    /// TensorKit-style coupled-sector matrix layout.
    ///
    /// Each coupled sector stores one contiguous column-major matrix whose
    /// rows enumerate (codomain fusion tree × codomain degeneracies) and whose
    /// columns enumerate (domain fusion tree × domain degeneracies). Fusion
    /// tree subblocks are strided views into that matrix, so the canonical
    /// (codomain | domain) matricization needs no packing. Block keys and
    /// their order are identical to [`Self::from_degeneracy_shapes`]; only
    /// strides and offsets differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, SectorLeg,
    ///     TensorMapSpace, Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let leg = || SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false);
    /// let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
    ///     TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
    ///     FusionTreeHomSpace::new(
    ///         FusionProductSpace::new([leg()]),
    ///         FusionProductSpace::new([leg()]),
    ///     ),
    ///     &rule,
    ///     [vec![1, 1], vec![1, 1]],
    /// )
    /// .unwrap();
    /// assert_eq!(space.required_len().unwrap(), 2);
    /// ```
    pub fn from_degeneracy_shapes_coupled<R, Shapes>(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        rule: &R,
        shapes: Shapes,
    ) -> Result<Self, CoreError>
    where
        R: MultiplicityFreeFusionRule,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        Self::validate_homspace_rank(&homspace)?;
        let subblock_structure = homspace.coupled_subblock_structure(rule, NOUT, shapes)?;
        Self::from_shared_subblock_structure(dense_space, homspace, subblock_structure)
            .map(|space| space.with_rule_identity(rule.rule_identity()))
    }

    fn validate_homspace_rank(homspace: &FusionTreeHomSpace) -> Result<(), CoreError> {
        if homspace.codomain().len() != NOUT {
            return Err(CoreError::StructureRankMismatch {
                expected: NOUT,
                actual: homspace.codomain().len(),
            });
        }
        if homspace.domain().len() != NIN {
            return Err(CoreError::StructureRankMismatch {
                expected: NIN,
                actual: homspace.domain().len(),
            });
        }
        Ok(())
    }

    #[inline]
    pub fn dense_space(&self) -> &TensorMapSpace<NOUT, NIN> {
        &self.dense_space
    }

    #[inline]
    pub fn homspace(&self) -> &FusionTreeHomSpace {
        &self.homspace
    }

    /// Shared handle to the hom space; lets replay caches compare spaces by
    /// pointer identity before falling back to structural equality.
    #[inline]
    pub fn homspace_arc(&self) -> &Arc<FusionTreeHomSpace> {
        &self.homspace
    }

    #[inline]
    pub fn subblock_structure(&self) -> &Arc<BlockStructure> {
        &self.subblock_structure
    }

    #[inline]
    pub fn rule_identity(&self) -> Option<RuleIdentity> {
        self.rule_identity.clone()
    }

    pub fn validate_rule<R: FusionRule>(&self, rule: &R) -> Result<(), CoreError> {
        match self.rule_identity.as_ref() {
            Some(expected) if expected != &rule.rule_identity() => Err(CoreError::FusionRuleMismatch {
                expected: expected.clone(),
                actual: rule.rule_identity(),
            }),
            Some(_) => Ok(()),
            None => Err(CoreError::MissingFusionRuleIdentity),
        }
    }

    pub fn try_bind_rule<R: FusionRule>(mut self, rule: &R) -> Result<Self, CoreError> {
        let actual = rule.rule_identity();
        if let Some(expected) = self.rule_identity.as_ref() {
            if expected != &actual {
                return Err(CoreError::FusionRuleMismatch {
                    expected: expected.clone(),
                    actual,
                });
            }
        }
        // Why not treat an equal identity tag as LOCAL proof: inheritance can
        // copy the tag without a provider. Binding therefore revalidates the
        // exact categorical structure before retaining or returning the tag.
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, self.subblock_structure())?;
        self.rule_identity = Some(actual);
        Ok(self)
    }

    pub fn try_inherit_rule_identity<const OTHER_NOUT: usize, const OTHER_NIN: usize>(
        mut self,
        source: &FusionTensorMapSpace<OTHER_NOUT, OTHER_NIN>,
    ) -> Result<Self, CoreError> {
        match (self.rule_identity.as_ref(), source.rule_identity.as_ref()) {
            (Some(expected), Some(actual)) if expected != actual => {
                return Err(CoreError::FusionRuleMismatch {
                    expected: expected.clone(),
                    actual: actual.clone(),
                });
            }
            (_, None) => return Err(CoreError::MissingFusionRuleIdentity),
            _ => {}
        }
        match self.subblock_structure.sector_structure().key_kind() {
            None | Some(BlockKeyKind::FusionTree) => {}
            Some(actual) => {
                return Err(CoreError::ExpectedFusionTreePairKey { actual });
            }
        }
        if self.rule_identity.is_none() {
            self.rule_identity = source.rule_identity.clone();
        }
        Ok(self)
    }

    // Why not expose a general tag setter: this path is reserved for structures
    // enumerated directly from the same HomSpace and rule above.
    fn with_rule_identity(mut self, identity: RuleIdentity) -> Self {
        self.rule_identity = Some(identity);
        self
    }

    pub fn find_subblock_index(&self, key: &FusionTreePairKey) -> Option<usize> {
        self.subblock_structure
            .find_block_index_by_fusion_tree_pair(key)
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.subblock_structure.required_len()
    }
}
