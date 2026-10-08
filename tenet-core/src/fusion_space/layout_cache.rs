use super::*;
use crate::cache::{StructureCache, StructureCacheKind};

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
///
/// `generic` separates the vertex-resolved Generic enumeration from the
/// multiplicity-free one of a rule identity.
#[derive(Clone, Debug)]
pub(crate) struct FusionTreeHomSpaceCacheKey {
    pub(super) rule: RuleIdentity,
    pub(super) generic: bool,
    pub(super) homspace: Arc<FusionTreeHomSpaceContent>,
}

impl PartialEq for FusionTreeHomSpaceCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.rule == other.rule
            && self.generic == other.generic
            && (Arc::ptr_eq(&self.homspace, &other.homspace)
                || (product_space_signature_eq(&self.homspace.codomain, &other.homspace.codomain)
                    && product_space_signature_eq(&self.homspace.domain, &other.homspace.domain)))
    }
}

impl Eq for FusionTreeHomSpaceCacheKey {}

impl std::hash::Hash for FusionTreeHomSpaceCacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rule.hash(state);
        self.generic.hash(state);
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CompleteFusionMode {
    MultiplicityFree,
    Generic,
}

/// Semantic identity of one complete block layout.
#[derive(Clone, Debug)]
pub(crate) struct CompleteHomSpaceStructureCacheKey {
    pub(crate) rule: RuleIdentity,
    pub(crate) mode: CompleteFusionMode,
    pub(crate) homspace: Arc<FusionTreeHomSpaceContent>,
}

impl CompleteHomSpaceStructureCacheKey {
    pub(crate) fn new<R>(rule: &R, homspace: &FusionTreeHomSpace) -> Self
    where
        R: MultiplicityFreeFusionRule,
    {
        Self {
            rule: rule.rule_identity(),
            mode: CompleteFusionMode::MultiplicityFree,
            homspace: Arc::clone(&homspace.content),
        }
    }

    pub(crate) fn generic(rule: RuleIdentity, homspace: &FusionTreeHomSpace) -> Self {
        Self {
            rule,
            mode: CompleteFusionMode::Generic,
            homspace: Arc::clone(&homspace.content),
        }
    }
}

impl PartialEq for CompleteHomSpaceStructureCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.rule == other.rule && self.mode == other.mode && self.homspace == other.homspace
    }
}

impl Eq for CompleteHomSpaceStructureCacheKey {}

impl std::hash::Hash for CompleteHomSpaceStructureCacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rule.hash(state);
        self.mode.hash(state);
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
            generic: false,
            homspace: Arc::clone(&homspace.content),
        }
    }

    pub(crate) fn generic(rule: RuleIdentity, homspace: &FusionTreeHomSpace) -> Self {
        Self {
            rule,
            generic: true,
            homspace: Arc::clone(&homspace.content),
        }
    }
}

/// The budgets of the caches these replace (#1993).
pub(crate) const SECTOR_STRUCTURE_CACHE_BYTE_BUDGET: u64 = 64 * 1024 * 1024;
pub(crate) const DEGENERACY_STRUCTURE_CACHE_BYTE_BUDGET: u64 = 64 * 1024 * 1024;
/// One shard: entries are heavy-tailed, and the largest must fit one shard.
/// The U(1) `V^6 <- V^6` sector and degeneracy structures charge about
/// 44 and 46 MB (`warm_contract_compile_allocations`, rank 6), so two shards
/// would need a 128 MiB budget. Why one shard costs no contention: a hit
/// takes only the shard's read lock.
const STRUCTURE_CACHE_SHARDS: usize = 1;

/// Fusion-tree keys and the shared sector structure per HomSpace sector
/// signature (TensorKit `sectorstructure`, `structure.jl:41` @cfaa073):
/// degeneracies are not part of the key, so a degeneracy-only change hits.
pub(crate) fn sector_structure_cache(
) -> &'static StructureCache<FusionTreeHomSpaceCacheKey, FusionTreeHomSpaceLayout> {
    static CACHE: OnceLock<StructureCache<FusionTreeHomSpaceCacheKey, FusionTreeHomSpaceLayout>> =
        OnceLock::new();
    CACHE.get_or_init(|| {
        StructureCache::new(
            StructureCacheKind::SectorStructure,
            SECTOR_STRUCTURE_CACHE_BYTE_BUDGET,
            STRUCTURE_CACHE_SHARDS,
        )
    })
}

pub(crate) fn charged_fusion_tree_layout_bytes(
    identity: &FusionTreeHomSpaceCacheKey,
    layout: &FusionTreeHomSpaceLayoutData,
) -> u64 {
    // The key retains the caller's HomSpace content instead of an owned
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
        .saturating_add(std::mem::size_of::<FusionTreeHomSpaceLayout>())
        .saturating_add(tree_bytes)
        .saturating_add(sector_bytes)
        .saturating_add(std::mem::size_of::<SectorStructure>())
        .saturating_add(
            layout
                .sector
                .as_ref()
                .map_or(0, |sector| sector.charged_heap_bytes()),
        )
        // Map node and the retained Arc control allocations.
        .saturating_add(10 * std::mem::size_of::<usize>()) as u64
}

/// One complete block structure: the content strongly, the canonical wrapper
/// weakly. TensorKit's complete dimensions/strides/offsets boundary is
/// [degeneracystructure](https://github.com/QuantumKitHub/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/spaces/structure.jl#L114-L198).
/// [QSpace's QIDX/DATA/CGR ownership](https://bitbucket.org/qspace4u/qspace-v4-pub/src/d2d3d7da6a59a2e8f2cb7dc8f33e7c345af59371/Source/QSpace.hh#lines-124:260)
/// keeps runtime structural records with the tensor; it has no corresponding
/// global interner. Bounded Arc/Weak canonical reuse and reset epochs are the
/// Rust adaptation of that complete-geometry boundary.
///
/// Why strong: the wrapper's coupled-sector region memo is immutable data
/// derived from this key alone (part of the complete structure, as in
/// TensorKit's `degeneracystructure`). A weak link let it die between calls,
/// so every warm call through a derived space rebuilt the wrapper and
/// recompiled its regions. The memo is charged here when it materializes.
pub(crate) struct DegeneracyStructureEntry {
    key: CompleteHomSpaceStructureCacheKey,
    homspace: HomSpaceId,
    structure: Arc<BlockStructure>,
}

impl DegeneracyStructureEntry {
    fn new(key: &CompleteHomSpaceStructureCacheKey, structure: Arc<BlockStructure>) -> Self {
        Self {
            key: key.clone(),
            homspace: HomSpaceId::from_content(Arc::clone(&key.homspace)),
            structure,
        }
    }

    /// The canonical wrapper.
    pub(crate) fn structure(&self) -> Arc<BlockStructure> {
        Arc::clone(&self.structure)
    }

    /// Charges a region memo of the canonical wrapper that just materialized.
    pub(crate) fn charge_regions(self: &Arc<Self>, bytes: u64) {
        degeneracy_structure_cache().add_charge(&self.key, self, bytes);
    }

    pub(crate) fn homspace_id(&self) -> HomSpaceId {
        self.homspace.clone()
    }
}

/// Complete block structure per HomSpace, degeneracies included (TensorKit
/// `degeneracystructure`, `structure.jl:114` @cfaa073). A miss reuses the
/// sector structure of [`sector_structure_cache`] and builds only the
/// degeneracy part.
pub(crate) fn degeneracy_structure_cache(
) -> &'static StructureCache<CompleteHomSpaceStructureCacheKey, DegeneracyStructureEntry> {
    static CACHE: OnceLock<
        StructureCache<CompleteHomSpaceStructureCacheKey, DegeneracyStructureEntry>,
    > = OnceLock::new();
    CACHE.get_or_init(|| {
        StructureCache::new(
            StructureCacheKind::DegeneracyStructure,
            DEGENERACY_STRUCTURE_CACHE_BYTE_BUDGET,
            STRUCTURE_CACHE_SHARDS,
        )
    })
}

/// The cached complete structure of `key`: one shard read and the entry's
/// wrapper.
pub(crate) fn complete_hom_space_structure_cached(
    key: &CompleteHomSpaceStructureCacheKey,
) -> Option<Arc<DegeneracyStructureEntry>> {
    degeneracy_structure_cache().get(key)
}

#[cfg(test)]
std::thread_local! {
    /// This thread's complete-HomSpace admissions offered. Why not the global
    /// counter: concurrent tests advance it between a test's two readings.
    static COMPLETE_HOM_SPACE_MISS_OBSERVATIONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn complete_hom_space_miss_observations() -> usize {
    COMPLETE_HOM_SPACE_MISS_OBSERVATIONS.get()
}

/// Admits a structure whose build began at reset epoch `epoch`. A build that
/// straddles a reset returns its result unpublished (the reset contract at
/// `reset_core_intern_tables`).
pub(crate) fn admit_complete_hom_space_structure(
    key: CompleteHomSpaceStructureCacheKey,
    structure: Arc<BlockStructure>,
    epoch: usize,
) -> (Arc<DegeneracyStructureEntry>, Arc<BlockStructure>) {
    #[cfg(test)]
    COMPLETE_HOM_SPACE_MISS_OBSERVATIONS.set(COMPLETE_HOM_SPACE_MISS_OBSERVATIONS.get() + 1);
    let entry = Arc::new(DegeneracyStructureEntry::new(&key, Arc::clone(&structure)));
    // Link before measuring: a memo that materializes after the snapshot
    // charges itself through the link instead.
    structure.link_region_owner(&entry);
    let charged_bytes = charged_complete_hom_space_structure_bytes(&key, &structure.content_key())
        .saturating_add(structure.materialized_region_bytes());
    let published = degeneracy_structure_cache().publish(&key, entry, charged_bytes, epoch);
    let canonical = published.structure();
    (published, canonical)
}

pub(crate) fn charged_complete_hom_space_structure_bytes(
    key: &CompleteHomSpaceStructureCacheKey,
    content: &BlockStructureContent,
) -> u64 {
    std::mem::size_of::<CompleteHomSpaceStructureCacheKey>()
        .saturating_add(std::mem::size_of::<DegeneracyStructureEntry>())
        .saturating_add(key.rule.charged_retained_bytes())
        .saturating_add(key.homspace.charged_retained_bytes())
        .saturating_add(content.charged_retained_bytes())
        // Map node and the retained Arc control allocations.
        .saturating_add(10 * std::mem::size_of::<usize>())
        // The owned wrapper and its region state; the region memo itself is
        // charged by the admission snapshot and as it materializes.
        .saturating_add(BlockStructure::wrapper_retained_bytes(content.rank())) as u64
}

pub(crate) fn reset_structure_caches() {
    degeneracy_structure_cache().clear();
    sector_structure_cache().clear();
}
