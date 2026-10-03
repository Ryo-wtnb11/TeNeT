use super::*;

/// Semantic identity of one multiplicity-free fusion-tree layout: the rule
/// plus the HomSpace sector signature.
///
/// Why the shared content rather than an owned per-leg signature: a warm
/// lookup must construct this key without allocating, so it borrows the
/// caller's content the way [`CompleteHomSpaceStructureCacheKey`] does.
///
/// Why not a derived `Eq`/`Hash` over that content: degeneracies are not part
/// of the layout identity, and deriving would split one layout into one entry
/// per degeneracy assignment. Equality and hashing therefore cover only the
/// per-leg sectors and duality, exactly as the previous owned signature did.
#[derive(Clone, Debug)]
pub(crate) struct FusionTreeHomSpaceCacheKey {
    pub(super) rule: RuleIdentity,
    pub(super) homspace: Arc<FusionTreeHomSpaceContent>,
}

impl PartialEq for FusionTreeHomSpaceCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.rule == other.rule
            && (Arc::ptr_eq(&self.homspace, &other.homspace)
                || (product_space_signature_eq(&self.homspace.codomain, &other.homspace.codomain)
                    && product_space_signature_eq(&self.homspace.domain, &other.homspace.domain)))
    }
}

impl Eq for FusionTreeHomSpaceCacheKey {}

impl std::hash::Hash for FusionTreeHomSpaceCacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rule.hash(state);
        hash_product_space_signature(&self.homspace.codomain, state);
        hash_product_space_signature(&self.homspace.domain, state);
    }
}

pub(super) fn product_space_signature_eq(
    left: &FusionProductSpace,
    right: &FusionProductSpace,
) -> bool {
    left.legs().len() == right.legs().len()
        && left.legs().iter().zip(right.legs()).all(|(left, right)| {
            left.is_dual() == right.is_dual() && left.sectors() == right.sectors()
        })
}

fn hash_product_space_signature<H: std::hash::Hasher>(space: &FusionProductSpace, state: &mut H) {
    space.legs().len().hash(state);
    for leg in space.legs() {
        leg.sectors().hash(state);
        leg.is_dual().hash(state);
    }
}

/// Semantic identity of one complete multiplicity-free block layout.
///
/// This deliberately owns neither a lazy `HomSpaceId` nor a layout id: both
/// are process-local accelerators and are not part of the represented space.
#[derive(Clone)]
pub(crate) struct CompleteHomSpaceStructureCacheKey {
    pub(crate) rule: RuleIdentity,
    pub(crate) homspace: Arc<FusionTreeHomSpaceContent>,
}

impl CompleteHomSpaceStructureCacheKey {
    pub(crate) fn new<R>(rule: &R, homspace: &FusionTreeHomSpace) -> Self
    where
        R: MultiplicityFreeFusionRule,
    {
        Self {
            rule: rule.rule_identity(),
            homspace: Arc::clone(&homspace.content),
        }
    }
}

impl PartialEq for CompleteHomSpaceStructureCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.rule == other.rule && self.homspace == other.homspace
    }
}

impl Eq for CompleteHomSpaceStructureCacheKey {}

impl std::hash::Hash for CompleteHomSpaceStructureCacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rule.hash(state);
        self.homspace.hash(state);
    }
}

impl FusionTreeHomSpaceCacheKey {
    pub(crate) fn new<R>(rule: &R, homspace: &FusionTreeHomSpace) -> Self
    where
        R: MultiplicityFreeFusionRule,
    {
        Self {
            rule: rule.rule_identity(),
            homspace: Arc::clone(&homspace.content),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) struct CoupledBlockStructureCacheKey {
    // Why-not `Arc::as_ptr`: eviction/reset may recycle an address while a
    // coupled structure keyed by that address is still alive. This monotonic
    // process-local id is never recycled, including across cache reset.
    pub(super) layout: FusionTreeLayoutId,
    pub(super) nout: usize,
    pub(super) rank: usize,
    pub(super) shapes: Arc<[DimVec]>,
}

#[derive(Clone)]
struct FusionTreeLayoutCacheEntry {
    layout: Arc<FusionTreeHomSpaceLayout>,
    charged_bytes: usize,
}

/// Bounded insertion-order cache with a one-entry last-inserted front.
/// Lookups never promote an entry; eviction always removes the oldest admitted
/// entry, so this policy is FIFO rather than LRU.
pub(crate) struct FusionTreeLayoutCache {
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

pub(crate) const FUSION_TREE_LAYOUT_CACHE_CAP: usize = 8192;
pub(crate) const FUSION_TREE_LAYOUT_CACHE_BYTE_BUDGET: usize = 64 * 1024 * 1024;
pub(crate) const FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES: usize = 8 * 1024 * 1024;

impl FusionTreeLayoutCache {
    pub(crate) fn new(entry_capacity: usize, byte_budget: usize, max_entry_bytes: usize) -> Self {
        assert!(
            entry_capacity > 0,
            "fusion-tree layout cache capacity must be positive"
        );
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

    pub(crate) fn lookup(
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

    pub(crate) fn admit(
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
            || self.charged_payload_bytes.saturating_add(charged_bytes) > self.byte_budget
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

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.last = None;
        self.charged_payload_bytes = 0;
        self.misses = 0;
        self.evictions = 0;
        self.admission_bypasses = 0;
    }

    pub(crate) fn info(&self) -> FusionTreeLayoutCacheInfo {
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

pub(crate) fn fusion_tree_layout_cache() -> &'static RwLock<FusionTreeLayoutCache> {
    static CACHE: OnceLock<RwLock<FusionTreeLayoutCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        RwLock::new(FusionTreeLayoutCache::new(
            FUSION_TREE_LAYOUT_CACHE_CAP,
            FUSION_TREE_LAYOUT_CACHE_BYTE_BUDGET,
            FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES,
        ))
    })
}

/// Acquires the layout cache for admission. Every commit-path writer goes
/// through here, so tests can assert that a warm commit takes none.
pub(super) fn fusion_tree_layout_cache_write(
) -> std::sync::RwLockWriteGuard<'static, FusionTreeLayoutCache> {
    #[cfg(test)]
    FUSION_TREE_LAYOUT_WRITE_LOCKS.set(FUSION_TREE_LAYOUT_WRITE_LOCKS.get() + 1);
    fusion_tree_layout_cache()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
pub(crate) fn fusion_tree_layout_write_locks() -> usize {
    FUSION_TREE_LAYOUT_WRITE_LOCKS.get()
}

/// Returns entry and charged-payload bounds for the process-global layout cache.
pub fn fusion_tree_layout_cache_info() -> FusionTreeLayoutCacheInfo {
    let cache = fusion_tree_layout_cache()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache.info()
}

pub(crate) fn charged_fusion_tree_layout_bytes(
    identity: &FusionTreeHomSpaceCacheKey,
    layout: &FusionTreeHomSpaceLayout,
) -> usize {
    // The key now retains the caller's HomSpace content instead of an owned
    // signature copy, so the charge covers that shared content.
    let key_bytes = std::mem::size_of::<FusionTreeHomSpaceCacheKey>()
        .saturating_add(identity.homspace.charged_retained_bytes())
        .saturating_add(identity.rule.charged_retained_bytes());
    let mut frozen_backings = rustc_hash::FxHashSet::default();
    let tree_bytes = layout.keys.iter().fold(0usize, |bytes, key| {
        [key.codomain_tree(), key.domain_tree()]
            .iter()
            .fold(bytes, |tree_bytes, tree| {
                tree_bytes
                    .saturating_add(std::mem::size_of::<FusionTreeKey>())
                    .saturating_add(charge_fusion_tree_key_backings(&mut frozen_backings, tree))
            })
    });
    let sector_bytes = layout
        .sectors
        .capacity()
        .saturating_mul(std::mem::size_of::<FusionTreeCoupledSectorLayout>());
    key_bytes
        .saturating_add(std::mem::size_of::<FusionTreeLayoutCacheEntry>())
        .saturating_add(std::mem::size_of::<FusionTreeHomSpaceLayout>())
        .saturating_add(tree_bytes)
        .saturating_add(sector_bytes)
        .saturating_add(8 * std::mem::size_of::<usize>())
}

pub(crate) struct CompleteHomSpaceStructureCacheEntry {
    pub(crate) content: Arc<BlockStructureContent>,
    /// Canonical wrapper accelerator. Weak, not strong: the strong owner
    /// retains frozen content only, never wrapper-local region state.
    pub(crate) wrapper: Weak<BlockStructure>,
    charged_bytes: usize,
}

pub(crate) enum CompleteHomSpaceStructureLookup {
    Wrapper(Arc<BlockStructure>),
    Content(Arc<BlockStructureContent>),
}

/// Bounded FIFO owner for complete immutable multiplicity-free layouts.
///
/// Memory tradeoff: retains up to 4 MiB of charged bytes process-wide. The
/// byte budget is the binding bound; the 1024-entry cap only stops the count
/// growing without bound when entries are very small. An entry above
/// `max_entry_bytes` (about 39 % of the budget) is returned uncached, so one
/// outlier can no longer evict the working set and hold the budget alone.
/// Why not shrink `max_entry_bytes` as well: a smaller limit bypasses
/// high-rank structures that are cached today, e.g. the 361,285-byte rank-7
/// U(1) destination of `warm_contract_compile_allocations`; the larger budget,
/// not a smaller entry limit, is what removes the monopolisation. The census of #1365
/// (`benchmarks/history/complete-structure-census-2026-09-24.md`) measured
/// entries of 2.6-47 KB and warm live sets of up to 53 structures and 301 KB
/// in MPS sweeps and `tensor!` networks; a single eager op touches at most 3.
///
/// Eviction is FIFO by admission. `entries` is an `lru::LruCache` used only as
/// an insertion-ordered map: lookups go through `peek`/`peek_mut`, which never
/// promote, and `put` only inserts absent keys, so `pop_lru` pops the oldest
/// admission. Why not true LRU: promotion needs `&mut self`, so every hit
/// would take the write lock; at these bounds FIFO and LRU give the same
/// (compulsory-only) warm misses on the census traces.
///
/// Why not reuse `coupled_block_structure_cache`: that weak table accepts
/// arbitrary `nout` and caller shapes, so it cannot avoid the final-HomSpace
/// construction work and would retain a live wrapper if made strong.
pub(crate) struct CompleteHomSpaceStructureCache {
    /// Insertion-ordered (FIFO) map; never promoted, see the type docs.
    pub(crate) entries: lru::LruCache<
        Arc<CompleteHomSpaceStructureCacheKey>,
        CompleteHomSpaceStructureCacheEntry,
        rustc_hash::FxBuildHasher,
    >,
    entry_capacity: usize,
    byte_budget: usize,
    max_entry_bytes: usize,
    charged_bytes: usize,
    hits: AtomicUsize,
    misses: AtomicUsize,
    admissions: AtomicUsize,
    evictions: AtomicUsize,
    bypasses: AtomicUsize,
}

pub(crate) const COMPLETE_HOM_SPACE_STRUCTURE_CACHE_CAP: usize = 1024;
pub(crate) const COMPLETE_HOM_SPACE_STRUCTURE_CACHE_BYTE_BUDGET: usize = 4 * 1024 * 1024;
pub(crate) const COMPLETE_HOM_SPACE_STRUCTURE_CACHE_MAX_ENTRY_BYTES: usize = 1_650_641;

impl CompleteHomSpaceStructureCache {
    pub(crate) fn new(entry_capacity: usize, byte_budget: usize, max_entry_bytes: usize) -> Self {
        assert!(
            entry_capacity > 0,
            "complete HomSpace cache capacity must be positive"
        );
        Self {
            // Unbounded here because `admit` enforces `entry_capacity`; a
            // bounded constructor would preallocate the table for the whole
            // cap, uncharged, even while the cache is nearly empty.
            entries: lru::LruCache::unbounded_with_hasher(rustc_hash::FxBuildHasher),
            entry_capacity,
            byte_budget,
            max_entry_bytes,
            charged_bytes: 0,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            admissions: AtomicUsize::new(0),
            evictions: AtomicUsize::new(0),
            bypasses: AtomicUsize::new(0),
        }
    }

    /// Counts only hits; a miss is recorded at admission, after its builder
    /// succeeded, so rejected input never changes the statistics.
    pub(crate) fn peek_counting_hit(
        &self,
        key: &CompleteHomSpaceStructureCacheKey,
    ) -> Option<CompleteHomSpaceStructureLookup> {
        let found = self.peek(key);
        if found.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        found
    }

    fn peek(
        &self,
        key: &CompleteHomSpaceStructureCacheKey,
    ) -> Option<CompleteHomSpaceStructureLookup> {
        self.entries
            .peek(key)
            .map(|entry| match entry.wrapper.upgrade() {
                Some(wrapper) => CompleteHomSpaceStructureLookup::Wrapper(wrapper),
                None => CompleteHomSpaceStructureLookup::Content(Arc::clone(&entry.content)),
            })
    }

    /// Repoints a retained entry at the canonical wrapper after its previous
    /// wrapper died; a no-op when the key was evicted meanwhile.
    fn refresh(
        &mut self,
        key: &CompleteHomSpaceStructureCacheKey,
        structure: &Arc<BlockStructure>,
    ) {
        if let Some(entry) = self.entries.peek_mut(key) {
            // Content and wrapper must come from one interning generation:
            // a racing admit after intern-table eviction could otherwise pair
            // an old content id with a wrapper minted under a new one.
            entry.content = structure.content_key();
            entry.wrapper = Arc::downgrade(structure);
        }
    }

    /// Records the miss of a completed build, then admits it; the only
    /// production admission path.
    pub(crate) fn admit_built(
        &mut self,
        key: Arc<CompleteHomSpaceStructureCacheKey>,
        structure: Arc<BlockStructure>,
        charged_bytes: usize,
    ) -> Arc<BlockStructure> {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.admit(key, structure, charged_bytes)
    }

    fn admit(
        &mut self,
        key: Arc<CompleteHomSpaceStructureCacheKey>,
        structure: Arc<BlockStructure>,
        charged_bytes: usize,
    ) -> Arc<BlockStructure> {
        match self.peek(&key) {
            Some(CompleteHomSpaceStructureLookup::Wrapper(existing)) => return existing,
            Some(CompleteHomSpaceStructureLookup::Content(_)) => {
                self.refresh(&key, &structure);
                return structure;
            }
            None => {}
        }
        if charged_bytes == usize::MAX
            || charged_bytes > self.max_entry_bytes
            || charged_bytes > self.byte_budget
        {
            self.bypasses.fetch_add(1, Ordering::Relaxed);
            return structure;
        }

        while self.entries.len() >= self.entry_capacity
            || self.charged_bytes.saturating_add(charged_bytes) > self.byte_budget
        {
            let Some((_, evicted)) = self.entries.pop_lru() else {
                break;
            };
            self.charged_bytes = self.charged_bytes.saturating_sub(evicted.charged_bytes);
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }

        self.charged_bytes = self.charged_bytes.saturating_add(charged_bytes);
        self.entries.put(
            key,
            CompleteHomSpaceStructureCacheEntry {
                content: structure.content_key(),
                wrapper: Arc::downgrade(&structure),
                charged_bytes,
            },
        );
        self.admissions.fetch_add(1, Ordering::Relaxed);
        structure
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.charged_bytes = 0;
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.admissions.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.bypasses.store(0, Ordering::Relaxed);
    }

    pub(crate) fn info(&self) -> CompleteHomSpaceStructureCacheInfo {
        CompleteHomSpaceStructureCacheInfo {
            entries: self.entries.len(),
            charged_bytes: self.charged_bytes,
            entry_capacity: self.entry_capacity,
            byte_budget: self.byte_budget,
            max_entry_bytes: self.max_entry_bytes,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            admissions: self.admissions.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            bypasses: self.bypasses.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of the complete immutable HomSpace layout cache resource state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompleteHomSpaceStructureCacheInfo {
    entries: usize,
    charged_bytes: usize,
    entry_capacity: usize,
    byte_budget: usize,
    max_entry_bytes: usize,
    hits: usize,
    misses: usize,
    admissions: usize,
    evictions: usize,
    bypasses: usize,
}

impl CompleteHomSpaceStructureCacheInfo {
    pub fn entries(self) -> usize {
        self.entries
    }
    pub fn charged_bytes(self) -> usize {
        self.charged_bytes
    }
    pub fn entry_capacity(self) -> usize {
        self.entry_capacity
    }
    pub fn byte_budget(self) -> usize {
        self.byte_budget
    }
    pub fn max_entry_bytes(self) -> usize {
        self.max_entry_bytes
    }
    pub fn hits(self) -> usize {
        self.hits
    }
    /// Completed builds that reached admission, including bypassed entries
    /// and racing duplicates. Failed builds are not counted, and a hit peek
    /// counts no miss, so `hits + misses` is not the lookup count.
    pub fn misses(self) -> usize {
        self.misses
    }
    pub fn admissions(self) -> usize {
        self.admissions
    }
    pub fn evictions(self) -> usize {
        self.evictions
    }
    pub fn bypasses(self) -> usize {
        self.bypasses
    }
}

pub(crate) fn complete_hom_space_structure_cache() -> &'static RwLock<CompleteHomSpaceStructureCache>
{
    static CACHE: OnceLock<RwLock<CompleteHomSpaceStructureCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        RwLock::new(CompleteHomSpaceStructureCache::new(
            COMPLETE_HOM_SPACE_STRUCTURE_CACHE_CAP,
            COMPLETE_HOM_SPACE_STRUCTURE_CACHE_BYTE_BUDGET,
            COMPLETE_HOM_SPACE_STRUCTURE_CACHE_MAX_ENTRY_BYTES,
        ))
    })
}

/// Returns bounds and activity for the complete immutable HomSpace layout cache.
pub fn complete_hom_space_structure_cache_info() -> CompleteHomSpaceStructureCacheInfo {
    complete_hom_space_structure_cache()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .info()
}

pub(crate) fn reset_complete_hom_space_structure_cache() {
    complete_hom_space_structure_cache()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

/// Hit path: one cache read with an uncounted-miss peek, a stack key (`Arc<K>: Borrow<K>`), and the
/// canonical wrapper returned as-is. A dead wrapper keeps the content hit and
/// rebuilds only the wrapper, repointing the entry under the write lock.
pub(super) fn complete_hom_space_structure_cached(
    key: &CompleteHomSpaceStructureCacheKey,
) -> Option<Arc<BlockStructure>> {
    let cache = complete_hom_space_structure_cache();
    let found = cache
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .peek_counting_hit(key);
    match found? {
        CompleteHomSpaceStructureLookup::Wrapper(structure) => Some(structure),
        CompleteHomSpaceStructureLookup::Content(content) => {
            let structure = BlockStructure::from_content(content).into_shared();
            cache
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .refresh(key, &structure);
            Some(structure)
        }
    }
}

pub(super) fn admit_complete_hom_space_structure(
    key: CompleteHomSpaceStructureCacheKey,
    structure: Arc<BlockStructure>,
) -> Arc<BlockStructure> {
    let charged_bytes = charged_complete_hom_space_structure_bytes(&key, &structure.content_key());
    complete_hom_space_structure_cache()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .admit_built(Arc::new(key), structure, charged_bytes)
}

pub(crate) fn charged_complete_hom_space_structure_bytes(
    key: &CompleteHomSpaceStructureCacheKey,
    content: &BlockStructureContent,
) -> usize {
    std::mem::size_of::<CompleteHomSpaceStructureCacheKey>()
        .saturating_add(std::mem::size_of::<CompleteHomSpaceStructureCacheEntry>())
        .saturating_add(key.rule.charged_retained_bytes())
        .saturating_add(key.homspace.charged_retained_bytes())
        .saturating_add(content.charged_retained_bytes())
        // Hash/FIFO nodes and both retained Arc control allocations.
        .saturating_add(10 * std::mem::size_of::<usize>())
        // The entry's Weak keeps the wrapper's `ArcInner` (two counters plus
        // the dropped `BlockStructure` payload) allocated after the last strong
        // owner dies, until the entry is evicted or refreshed.
        .saturating_add(2 * std::mem::size_of::<usize>())
        .saturating_add(std::mem::size_of::<BlockStructure>())
}

type CoupledBlockStructureCache =
    lru::LruCache<CoupledBlockStructureCacheKey, Weak<BlockStructure>, rustc_hash::FxBuildHasher>;

pub(super) fn coupled_block_structure_cache() -> &'static RwLock<CoupledBlockStructureCache> {
    static CACHE: OnceLock<RwLock<CoupledBlockStructureCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        RwLock::new(lru::LruCache::with_hasher(
            std::num::NonZeroUsize::new(BLOCK_STRUCTURE_INTERN_CAP).unwrap(),
            rustc_hash::FxBuildHasher,
        ))
    })
}

pub(crate) fn reset_fusion_tree_layout_caches() {
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
