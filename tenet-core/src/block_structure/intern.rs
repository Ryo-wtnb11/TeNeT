use super::*;

/// Compatibility snapshot for the removed block-content and wrapper interners.
///
/// Canonical complete layouts now belong only to the bounded complete-HomSpace
/// cache. This type remains source-compatible for one release and therefore
/// always reports the documented zero state.
#[deprecated(
    note = "block structures are owned by the complete-HomSpace cache; use structure_cache_info"
)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BlockStructureInternCacheInfo;

#[allow(deprecated)]
impl BlockStructureInternCacheInfo {
    pub fn entries(self) -> usize {
        0
    }

    pub fn entry_capacity(self) -> usize {
        0
    }

    pub fn charged_key_bytes(self) -> usize {
        0
    }

    pub fn byte_budget(self) -> usize {
        0
    }

    pub fn max_admitted_entry_bytes(self) -> usize {
        0
    }

    pub fn pressure_evictions(self) -> usize {
        0
    }

    pub fn oversized_admission_bypasses(self) -> usize {
        0
    }
}

/// Compatibility view of the removed process-global block interner.
#[deprecated(
    note = "block structures are owned by the complete-HomSpace cache; use structure_cache_info"
)]
#[allow(deprecated)]
pub fn block_structure_intern_cache_info() -> BlockStructureInternCacheInfo {
    BlockStructureInternCacheInfo
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

/// Process-global, strictly monotonic content identity.
///
/// Tensors-layer operation keys use this id directly. It is therefore never
/// reset or reused, even when the complete-layout cache is cleared.
pub(crate) static BLOCK_STRUCTURE_CONTENT_ID: AtomicUsize = AtomicUsize::new(1);

#[cfg(test)]
pub(crate) fn reset_block_structure_intern_calls() {}

#[cfg(test)]
pub(crate) fn block_structure_intern_calls() -> usize {
    0
}

pub(super) fn new_block_structure_content(
    sector: Arc<SectorStructure>,
    degeneracy: DegeneracyStructure,
    required_len: usize,
) -> Arc<BlockStructureContent> {
    Arc::new(BlockStructureContent::new(sector, degeneracy, required_len))
}

/// Clears the two bounded core structure caches.
///
/// Live HomSpace identities and block-content ids remain valid. Complete-layout
/// builds whose captured epoch straddles the reset cannot publish stale content;
/// they may still use a current canonical hit. Identity-free prepared sector
/// layouts retain their separate rule: commit may publish under the current epoch.
pub fn reset_core_intern_tables() {
    let _serial = CORE_RESET_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    CORE_RESET_EPOCH.fetch_add(1, Ordering::SeqCst);
    crate::fusion_space::reset_structure_caches();
    #[cfg(test)]
    MID_RESET_HOOK.with(|hook| {
        if let Some(hook) = hook.take() {
            hook();
        }
    });
    CORE_RESET_EPOCH.fetch_add(1, Ordering::SeqCst);
}

/// Reset epoch of the core structure caches; odd while a reset runs.
static CORE_RESET_EPOCH: AtomicUsize = AtomicUsize::new(0);

/// Held across a whole reset, both epoch bumps included.
pub(crate) static CORE_RESET_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The epoch a complete build records before it performs admission work.
pub(crate) fn core_reset_epoch() -> usize {
    CORE_RESET_EPOCH.load(Ordering::SeqCst)
}

/// Whether a structure built since `epoch` may be published.
pub(crate) fn may_publish_since(epoch: usize) -> bool {
    epoch.is_multiple_of(2) && CORE_RESET_EPOCH.load(Ordering::SeqCst) == epoch
}

#[cfg(test)]
std::thread_local! {
    pub(crate) static MID_RESET_HOOK: std::cell::Cell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::Cell::new(None) };
}
