use super::*;

/// Collision-safe semantic identity of a fusion HomSpace.
///
/// The prehash keeps repeated hashing O(1); the shared immutable content is
/// the collision fallback. Complete-layout cache entries choose the canonical
/// backing for bound spaces. Standalone equal spaces remain equal even when
/// they were created independently or the complete-layout cache was reset.
#[derive(Clone, Debug)]
pub struct HomSpaceId {
    prehash: u64,
    pub(crate) content: Arc<FusionTreeHomSpaceContent>,
}

/// Non-owning identity for one live canonical HomSpace backing.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct WeakHomSpaceId(Weak<FusionTreeHomSpaceContent>);

impl HomSpaceId {
    pub(crate) fn from_content(content: Arc<FusionTreeHomSpaceContent>) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        content.hash(&mut hasher);
        Self {
            prehash: hasher.finish(),
            content,
        }
    }

    /// A non-owning identity suitable for bounded runtime admission records.
    #[doc(hidden)]
    pub fn downgrade(&self) -> WeakHomSpaceId {
        WeakHomSpaceId(Arc::downgrade(&self.content))
    }
}

impl WeakHomSpaceId {
    /// Whether `id` still names the same live canonical backing.
    #[doc(hidden)]
    pub fn matches(&self, id: &HomSpaceId) -> bool {
        std::ptr::eq(self.0.as_ptr(), Arc::as_ptr(&id.content))
    }
}

impl PartialEq for HomSpaceId {
    fn eq(&self, other: &Self) -> bool {
        self.prehash == other.prehash
            && (Arc::ptr_eq(&self.content, &other.content) || self.content == other.content)
    }
}

impl Eq for HomSpaceId {}

impl std::hash::Hash for HomSpaceId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.prehash.hash(state);
    }
}

#[cfg(test)]
pub(crate) fn reset_hom_space_intern_calls() {}

#[cfg(test)]
pub(crate) fn hom_space_intern_calls() -> usize {
    0
}
