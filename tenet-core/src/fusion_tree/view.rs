use super::*;

/// Borrowed identity of one fusion tree.
///
/// Compares, orders and hashes exactly like [`FusionTreeKey`], so a view can
/// look up an owned key in a map or structure cache without building one.
// Why the views exist before the key changes (#2192): they are the one
// comparison authority that an arena-backed key and transient scratch trees
// can both produce.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct FusionTreeRef<'a> {
    uncoupled: &'a [SectorId],
    coupled: SectorId,
    is_dual: &'a [bool],
    innerlines: &'a [SectorId],
    vertices: &'a [MultiplicityIndex],
    hash: u64,
}

impl<'a> FusionTreeRef<'a> {
    /// View raw tree fields; lengths are not validated, as for raw keys.
    pub fn new(
        uncoupled: &'a [SectorId],
        coupled: SectorId,
        is_dual: &'a [bool],
        innerlines: &'a [SectorId],
        vertices: &'a [MultiplicityIndex],
    ) -> Self {
        let hash = fusion_tree_key_hash(uncoupled, coupled, is_dual, innerlines, vertices);
        Self::with_hash(uncoupled, coupled, is_dual, innerlines, vertices, hash)
    }

    #[inline]
    pub(crate) fn with_hash(
        uncoupled: &'a [SectorId],
        coupled: SectorId,
        is_dual: &'a [bool],
        innerlines: &'a [SectorId],
        vertices: &'a [MultiplicityIndex],
        hash: u64,
    ) -> Self {
        Self {
            uncoupled,
            coupled,
            is_dual,
            innerlines,
            vertices,
            hash,
        }
    }

    #[inline]
    pub fn uncoupled(&self) -> &'a [SectorId] {
        self.uncoupled
    }

    /// The hash [`Hash`] writes, as stored with the key.
    #[inline]
    pub(crate) fn stored_hash(&self) -> u64 {
        self.hash
    }

    #[inline]
    pub fn coupled(&self) -> SectorId {
        self.coupled
    }

    #[inline]
    pub fn is_dual(&self) -> &'a [bool] {
        self.is_dual
    }

    #[inline]
    pub fn innerlines(&self) -> &'a [SectorId] {
        self.innerlines
    }

    #[inline]
    pub fn vertices(&self) -> &'a [MultiplicityIndex] {
        self.vertices
    }

    pub(crate) fn fmt_named(&self, name: &str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(name)
            .field("uncoupled", &self.uncoupled)
            .field("coupled", &self.coupled)
            .field("is_dual", &self.is_dual)
            .field("innerlines", &self.innerlines)
            .field("vertices", &self.vertices)
            .finish()
    }
}

impl Hash for FusionTreeRef<'_> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl PartialEq for FusionTreeRef<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && self.coupled == other.coupled
            && self.uncoupled == other.uncoupled
            && self.is_dual == other.is_dual
            && self.innerlines == other.innerlines
            && self.vertices == other.vertices
    }
}

impl Eq for FusionTreeRef<'_> {}

impl Ord for FusionTreeRef<'_> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.uncoupled
            .cmp(other.uncoupled)
            .then_with(|| self.coupled.cmp(&other.coupled))
            .then_with(|| self.is_dual.cmp(other.is_dual))
            .then_with(|| self.innerlines.cmp(other.innerlines))
            .then_with(|| self.vertices.cmp(other.vertices))
    }
}

impl PartialOrd for FusionTreeRef<'_> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for FusionTreeRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_named("FusionTreeRef", f)
    }
}

impl quick_cache::Equivalent<FusionTreeKey> for FusionTreeRef<'_> {
    #[inline]
    fn equivalent(&self, key: &FusionTreeKey) -> bool {
        *self == key.as_view()
    }
}

/// Borrowed identity of one codomain/domain tree pair; compares, orders and
/// hashes exactly like [`FusionTreePairKey`].
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreePairRef<'a> {
    codomain_tree: FusionTreeRef<'a>,
    domain_tree: FusionTreeRef<'a>,
}

impl<'a> FusionTreePairRef<'a> {
    #[inline]
    pub fn new(codomain_tree: FusionTreeRef<'a>, domain_tree: FusionTreeRef<'a>) -> Self {
        Self {
            codomain_tree,
            domain_tree,
        }
    }

    #[inline]
    pub fn codomain_tree(&self) -> FusionTreeRef<'a> {
        self.codomain_tree
    }

    #[inline]
    pub fn domain_tree(&self) -> FusionTreeRef<'a> {
        self.domain_tree
    }

    /// An owned key of this pair, both trees in one exactly sized arena.
    pub fn to_key(&self) -> FusionTreePairKey {
        let trees = [self.codomain_tree, self.domain_tree];
        let mut builder = TreeArenaBuilder::for_views(trees.into_iter());
        let codomain = builder.push_view(self.codomain_tree);
        let domain = builder.push_view(self.domain_tree);
        let arena = builder.freeze();
        FusionTreePairKey::pair(arena.key(codomain), arena.key(domain))
    }

    /// The pair's group identity, borrowed like [`FusionTreePairKey::group_key`].
    #[inline]
    pub fn group(&self) -> FusionTreeGroupRef<'a> {
        FusionTreeGroupRef::new(
            self.codomain_tree.uncoupled,
            self.domain_tree.uncoupled,
            self.codomain_tree.is_dual,
            self.domain_tree.is_dual,
        )
    }
}

impl quick_cache::Equivalent<FusionTreePairKey> for FusionTreePairRef<'_> {
    #[inline]
    fn equivalent(&self, key: &FusionTreePairKey) -> bool {
        *self == key.as_view()
    }
}

/// Borrowed identity of one fusion-tree group; compares, orders and hashes
/// exactly like [`FusionTreeGroupKey`] (derived, field order interleaved).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreeGroupRef<'a> {
    codomain_uncoupled: &'a [SectorId],
    domain_uncoupled: &'a [SectorId],
    codomain_is_dual: &'a [bool],
    domain_is_dual: &'a [bool],
}

impl<'a> FusionTreeGroupRef<'a> {
    #[inline]
    pub fn new(
        codomain_uncoupled: &'a [SectorId],
        domain_uncoupled: &'a [SectorId],
        codomain_is_dual: &'a [bool],
        domain_is_dual: &'a [bool],
    ) -> Self {
        Self {
            codomain_uncoupled,
            domain_uncoupled,
            codomain_is_dual,
            domain_is_dual,
        }
    }

    #[inline]
    pub fn codomain_uncoupled(&self) -> &'a [SectorId] {
        self.codomain_uncoupled
    }

    #[inline]
    pub fn domain_uncoupled(&self) -> &'a [SectorId] {
        self.domain_uncoupled
    }

    #[inline]
    pub fn codomain_is_dual(&self) -> &'a [bool] {
        self.codomain_is_dual
    }

    #[inline]
    pub fn domain_is_dual(&self) -> &'a [bool] {
        self.domain_is_dual
    }
}

impl quick_cache::Equivalent<FusionTreeGroupKey> for FusionTreeGroupRef<'_> {
    #[inline]
    fn equivalent(&self, key: &FusionTreeGroupKey) -> bool {
        *self == key.as_view()
    }
}

impl FusionTreeKey {
    /// Borrow this key as a view; no allocation and no rehash.
    #[doc(hidden)]
    #[inline]
    pub fn as_view(&self) -> FusionTreeRef<'_> {
        self.arena.view(self.index)
    }
}

impl FusionTreePairKey {
    #[doc(hidden)]
    #[inline]
    pub fn as_view(&self) -> FusionTreePairRef<'_> {
        FusionTreePairRef::new(self.codomain_tree.as_view(), self.domain_tree.as_view())
    }
}

impl FusionTreeGroupKey {
    #[doc(hidden)]
    #[inline]
    pub fn as_view(&self) -> FusionTreeGroupRef<'_> {
        FusionTreeGroupRef::new(
            self.codomain.uncoupled(),
            self.domain.uncoupled(),
            self.codomain.is_dual(),
            self.domain.is_dual(),
        )
    }
}
