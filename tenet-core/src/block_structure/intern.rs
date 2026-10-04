use super::*;

/// Key of one interned content: its rank and a hash of its block keys and
/// degeneracy layout.
///
/// Why a hash rather than a copy of the blocks (#1998): a copied key doubled
/// every content's per-block bytes, and at U(1) `V^6 <- V^6` it outgrew the
/// table's entry limit, so that content was never interned and got a new id
/// on every derivation. A lookup checks the live content itself before it
/// hits, so two contents whose hashes collide are never aliased: the later one
/// is returned uninterned.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct BlockStructureInternKey {
    pub(crate) rank: usize,
    pub(crate) hash: u64,
}

impl BlockStructureInternKey {
    pub(crate) fn of(sector: &SectorStructure, degeneracy: &DegeneracyStructure) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        sector.block_count().hash(&mut hasher);
        for block in sector.blocks() {
            block.key().hash(&mut hasher);
        }
        degeneracy.dims.hash(&mut hasher);
        degeneracy.offsets.hash(&mut hasher);
        Self {
            rank: sector.rank(),
            hash: hasher.finish(),
        }
    }
}

struct BlockStructureInternEntry {
    content: Weak<BlockStructureContent>,
    charged_key_bytes: usize,
}

pub(crate) struct BlockStructureInternTable {
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
/// same bound, the arc dedup table, whose entries have a fixed size). Mirrors
/// `HOM_SPACE_INTERN_CAP`: a long-lived / multi-tenant process can otherwise
/// grow these tables without bound over a χ sweep. See
/// `BLOCK_STRUCTURE_CONTENT_ID` for why capping this particular table is
/// aliasing-safe despite its ids being consumed as cache keys downstream.
pub(crate) const BLOCK_STRUCTURE_INTERN_CAP: usize = 8192;
const BLOCK_STRUCTURE_INTERN_BYTE_BUDGET: usize = 64 * 1024 * 1024;
/// Why equal to the budget: the same admission policy as the other structure
/// caches (#1993). Since keys are hashes (#1998) every entry is far below it.
const BLOCK_STRUCTURE_INTERN_MAX_ENTRY_BYTES: usize = BLOCK_STRUCTURE_INTERN_BYTE_BUDGET;
// Why-not allocator-exact accounting: allocator headers are not portable.
// This fixed allowance conservatively covers hash/FIFO nodes, the weak handle,
// and the surviving Arc control-allocation shell.
const BLOCK_STRUCTURE_INTERN_CONTROL_ALLOWANCE_BYTES: usize =
    std::mem::size_of::<BlockStructureContent>() + 8 * std::mem::size_of::<usize>();

impl BlockStructureInternTable {
    pub(crate) fn new(entry_capacity: usize, byte_budget: usize, max_entry_bytes: usize) -> Self {
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

    /// The live content under `key` that `matches` accepts.
    pub(crate) fn lookup(
        &self,
        key: &BlockStructureInternKey,
        matches: impl Fn(&BlockStructureContent) -> bool,
    ) -> Option<Arc<BlockStructureContent>> {
        self.entries
            .peek(key)
            .and_then(|entry| entry.content.upgrade())
            .filter(|content| matches(content))
    }

    /// Interns `make_content()` under `key`. Callers first [`Self::lookup`]
    /// under the same lock, so a live content still under `key` is a hash
    /// collision with different content: it keeps its entry, and the new
    /// content is returned uninterned.
    pub(crate) fn intern_with<C, F>(
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
            if entry.content.strong_count() > 0 {
                return make_content();
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
            self.oversized_admission_bypasses = self.oversized_admission_bypasses.saturating_add(1);
            return content;
        }

        while self.entries.len() >= self.entry_capacity
            || self.charged_key_bytes.saturating_add(charged_key_bytes) > self.byte_budget
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

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.charged_key_bytes = 0;
        self.pressure_evictions = 0;
        self.oversized_admission_bypasses = 0;
    }

    pub(crate) fn info(&self) -> BlockStructureInternCacheInfo {
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

pub(crate) fn spilled_smallvec_heap_bytes<A>(values: &SmallVec<A>) -> usize
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

pub(crate) fn charged_block_structure_intern_key_bytes(_key: &BlockStructureInternKey) -> usize {
    std::mem::size_of::<BlockStructureInternKey>()
        .saturating_add(std::mem::size_of::<BlockStructureInternEntry>())
        .saturating_add(BLOCK_STRUCTURE_INTERN_CONTROL_ALLOWANCE_BYTES)
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
pub(crate) static BLOCK_STRUCTURE_CONTENT_ID: AtomicUsize = AtomicUsize::new(1);

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

pub(super) fn intern_block_structure_content(
    sector: SectorStructure,
    degeneracy: DegeneracyStructure,
    required_len: usize,
) -> Arc<BlockStructureContent> {
    #[cfg(test)]
    BLOCK_STRUCTURE_INTERN_CALLS.set(BLOCK_STRUCTURE_INTERN_CALLS.get() + 1);
    let key = BlockStructureInternKey::of(&sector, &degeneracy);
    let table = block_structure_intern_table();
    // Read-lock fast path uses `peek` (does not bump recency; `get` needs `&mut`).
    if let Ok(read) = table.read() {
        if let Some(content) = read.lookup(&key, |content| {
            content.sector == sector && content.degeneracy == degeneracy
        }) {
            return content;
        }
    }

    let mut write = table
        .write()
        .expect("block structure intern table poisoned");
    if let Some(content) = write.lookup(&key, |content| {
        content.sector == sector && content.degeneracy == degeneracy
    }) {
        return content;
    }
    write.intern_with(key, charged_block_structure_intern_key_bytes, || {
        Arc::new(BlockStructureContent {
            id: BLOCK_STRUCTURE_CONTENT_ID.fetch_add(1, Ordering::Relaxed),
            sector,
            degeneracy,
            required_len,
            storage_tiling: StorageTilingProof::default(),
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

pub(super) fn canonicalize_block_structure_arc(
    structure: Arc<BlockStructure>,
) -> Arc<BlockStructure> {
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
/// is needed.
///
/// Reset contract: no identity created before the reset cleared the owning
/// table is published after it. A build that straddles a reset still returns a correct result,
/// but the result is not cached. The two intern tables uphold this by
/// construction, because each mints its identity inside the write-locked
/// insert, so whatever lands in a cleared table is new. The complete-HomSpace
/// cache publishes a structure built earlier, so its admission checks
/// `may_publish_since` under its write lock, and its hit-path refresh only
/// repoints an entry that still holds the content it looked up.
///
/// Why not the arc dedup table or the fusion-tree layout cache: an arc-table
/// entry is keyed by its own never-reused content id, so a straddling insert
/// can only be found again by a holder of that same content. A layout is pure
/// data under a semantic key and carries no identity.
pub fn reset_core_intern_tables() {
    // Resets are serialized, so odd parity means exactly "a reset is in
    // progress". Why not let them overlap: a second reset would turn the
    // epoch even mid-way through the first, and a build starting then could
    // intern the first reset's not-yet-cleared content and publish it.
    let _serial = CORE_RESET_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Odd for the whole reset: a build that starts before or during it holds
    // a stale epoch and cannot publish afterwards.
    CORE_RESET_EPOCH.fetch_add(1, Ordering::SeqCst);
    // Clear the sole strong complete-layout owner before weak canonicalizers.
    // Live wrappers keep their own content and region state through reset.
    reset_complete_hom_space_structure_cache();
    #[cfg(test)]
    MID_RESET_HOOK.with(|hook| {
        if let Some(hook) = hook.take() {
            hook();
        }
    });
    reset_hom_space_intern_table();
    if let Ok(mut table) = block_structure_intern_table().write() {
        table.clear();
    }
    if let Ok(mut table) = block_structure_arc_table().write() {
        table.clear();
    }
    reset_fusion_tree_layout_caches();
    CORE_RESET_EPOCH.fetch_add(1, Ordering::SeqCst);
}

/// Reset epoch of the core identity tables; odd while a reset runs.
static CORE_RESET_EPOCH: AtomicUsize = AtomicUsize::new(0);

/// Held across a whole reset, both epoch bumps included.
pub(crate) static CORE_RESET_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The epoch a build records before it looks anything up.
pub(crate) fn core_reset_epoch() -> usize {
    CORE_RESET_EPOCH.load(Ordering::SeqCst)
}

/// Whether an identity built since `epoch` may be published: no reset started
/// since then, and none was running at the time.
pub(crate) fn may_publish_since(epoch: usize) -> bool {
    epoch.is_multiple_of(2) && CORE_RESET_EPOCH.load(Ordering::SeqCst) == epoch
}

#[cfg(test)]
std::thread_local! {
    /// Runs once inside the next reset on this thread, after the complete
    /// cache is cleared and before the intern tables are.
    pub(crate) static MID_RESET_HOOK: std::cell::Cell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::Cell::new(None) };
}
