use super::*;

/// Process-global intern id for a fusion hom space. [`FusionTreeHomSpace::id`]
/// deep-hashes the space on first demand (the full generic key: every codomain
/// and domain leg's sectors and dual flag — never a multiplicity-free subset)
/// and stores the collision-safe semantic identity in the hom space's lazy
/// cell. Downstream hashing reads its cached prehash in O(1); equality falls
/// back to the full immutable key only for matching prehashes.
///
/// Why-not (persist to disk): the cached prehash is an implementation detail,
/// so it remains a process-local acceleration object while the semantic space
/// value remains the portable identity.
///
/// Why-not (unbounded intern): applications can construct arbitrarily many
/// temporary spaces. The bounded table follows TensorKit's metadata-cache
/// policy. The semantic key remains in each live id, so equal spaces remain
/// equal across eviction; the interner only supplies a pointer-equality fast
/// path while an entry is resident.
#[derive(Clone, Debug)]
pub struct HomSpaceId {
    prehash: u64,
    pub(crate) key: Arc<HomSpaceInternKey>,
}

/// Non-owning process-local identity for one live [`HomSpaceId`].
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct WeakHomSpaceId(Weak<HomSpaceInternKey>);

impl HomSpaceId {
    /// A non-owning identity suitable for bounded runtime admission records.
    #[doc(hidden)]
    pub fn downgrade(&self) -> WeakHomSpaceId {
        WeakHomSpaceId(Arc::downgrade(&self.key))
    }
}

impl WeakHomSpaceId {
    /// Whether `id` still names the same live interned semantic key.
    #[doc(hidden)]
    pub fn matches(&self, id: &HomSpaceId) -> bool {
        Weak::ptr_eq(&self.0, &Arc::downgrade(&id.key))
    }
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
pub(crate) struct HomSpaceInternKey {
    codomain: FusionProductSpace,
    domain: FusionProductSpace,
}

pub(crate) struct HomSpaceInternTable {
    pub(crate) entries: lru::LruCache<HomSpaceInternKey, Arc<HomSpaceInternKey>>,
}

pub(crate) const HOM_SPACE_INTERN_CAP: usize = 8192;

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

pub(crate) fn hom_space_intern_table() -> &'static RwLock<HomSpaceInternTable> {
    static TABLE: OnceLock<RwLock<HomSpaceInternTable>> = OnceLock::new();
    TABLE.get_or_init(|| {
        RwLock::new(HomSpaceInternTable {
            entries: lru::LruCache::new(std::num::NonZeroUsize::new(HOM_SPACE_INTERN_CAP).unwrap()),
        })
    })
}

pub(super) fn intern_hom_space(
    codomain: &FusionProductSpace,
    domain: &FusionProductSpace,
) -> HomSpaceId {
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

pub(crate) fn reset_hom_space_intern_table() {
    if let Ok(mut table) = hom_space_intern_table().write() {
        table.entries.clear();
    }
}
