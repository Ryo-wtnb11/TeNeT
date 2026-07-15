#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FusionProductSpace {
    legs: Vec<SectorLeg>,
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
    id: HomSpaceId,
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
struct FusionTreeHomSpaceLayout {
    keys: Arc<[FusionTreeBlockKey]>,
    sectors: Vec<FusionTreeCoupledSectorLayout>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct CoupledBlockStructureCacheKey {
    // `layout_ptr` is a raw-pointer key into `fusion_tree_layout_cache`. It is
    // sound only because that cache is insert-only (`or_insert_with`, never
    // evicted): a layout Arc, once interned, lives forever, so its address is a
    // stable identity. Adding eviction to `fusion_tree_layout_cache` would make
    // this key unsound (freed address recycled by an unrelated layout → ABA).
    // Re-key on the layout's content identity first if that cache ever bounds.
    layout_ptr: usize,
    nout: usize,
    rank: usize,
    shapes: Arc<[DimVec]>,
}

/// Layout cache for fusion-tree hom spaces.
///
/// MUST remain insert-only — do NOT add an LRU cap here. `coupled_block_structure_cache`
/// keys its entries by `layout_ptr = Arc::as_ptr(layout)` (below), so evicting a
/// layout would let its `Arc` be freed and its address recycled by a later layout,
/// aliasing two distinct layouts under one pointer key. Keeping every layout `Arc`
/// resident forever is what makes that pointer key sound (the insert-only safety
/// condition). The table is bounded in practice by the finite set of hom-space
/// shapes a workload constructs; reclaiming it would first require moving the
/// coupled-cache key off the raw pointer onto a content id.
fn fusion_tree_layout_cache(
) -> &'static RwLock<FxHashMap<FusionTreeHomSpaceCacheKey, Arc<FusionTreeHomSpaceLayout>>> {
    static CACHE: OnceLock<
        RwLock<FxHashMap<FusionTreeHomSpaceCacheKey, Arc<FusionTreeHomSpaceLayout>>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(FxHashMap::default()))
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

/// Clears the coupled subblock-structure cache. Part of `reset_core_intern_tables`.
/// Safe to cap/clear (unlike `fusion_tree_layout_cache`): its entries are `Weak`
/// values keyed by data, not by a live pointer anything else depends on.
fn reset_coupled_block_structure_cache() {
    if let Ok(mut cache) = coupled_block_structure_cache().write() {
        cache.clear();
    }
}

/// Process-global intern id for a fusion hom space. [`FusionTreeHomSpace::id`]
/// deep-hashes the space at construction (the full generic key: every codomain
/// and domain leg's sectors and dual flag — never a multiplicity-free subset)
/// and stores a collision-safe semantic identity. Downstream hashing reads its
/// cached prehash in O(1); equality falls back to the full immutable key only
/// for matching prehashes. Mirrors the block-structure content intern.
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

struct HomSpaceInternTable {
    entries: lru::LruCache<HomSpaceInternKey, Arc<HomSpaceInternKey>>,
}

const HOM_SPACE_INTERN_CAP: usize = 8192;

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

fn fusion_tree_layout_from_keys<R>(
    rule: &R,
    keys: Vec<FusionTreeBlockKey>,
) -> FusionTreeHomSpaceLayout
where
    R: MultiplicityFreeFusionRule,
{
    let keys = Arc::<[FusionTreeBlockKey]>::from(keys);
    let mut sectors = Vec::new();
    let mut run_start = 0usize;
    while run_start < keys.len() {
        let coupled = coupled_or_vacuum(rule, keys[run_start].codomain_tree());
        let mut run_end = run_start;
        let mut row_indices = FxHashMap::<FusionTreeKey, usize>::default();
        let mut col_indices = FxHashMap::<FusionTreeKey, usize>::default();
        let mut entries = Vec::new();
        while run_end < keys.len()
            && coupled_or_vacuum(rule, keys[run_end].codomain_tree()) == coupled
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
    FusionTreeHomSpaceLayout { keys, sectors }
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
        let id = intern_hom_space(&codomain, &domain);
        Self {
            codomain,
            domain,
            id,
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

    /// Collision-safe process-local semantic identity assigned at construction.
    /// Hashing is O(1), and equal spaces compare equal across intern eviction.
    #[inline]
    pub fn id(&self) -> HomSpaceId {
        self.id.clone()
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
        validate_axis_selection(codomain_axes, domain_axes, self.rank())?;

        let codomain = codomain_axes
            .iter()
            .map(|&axis| self.external_axis_leg(rule, axis))
            .collect::<Vec<_>>();
        let domain = domain_axes
            .iter()
            .map(|&axis| dual_sector_leg(rule, &self.external_axis_leg(rule, axis)))
            .collect::<Vec<_>>();
        Ok(Self::new(
            FusionProductSpace::new(codomain),
            FusionProductSpace::new(domain),
        ))
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
        let mut axes = Vec::with_capacity(codomain_axes.len() + domain_axes.len());
        axes.extend_from_slice(codomain_axes);
        axes.extend_from_slice(domain_axes);
        validate_permutation(&axes, self.rank())?;
        self.select(rule, codomain_axes, domain_axes)
    }

    pub fn compose<R>(_rule: &R, lhs: &Self, rhs: &Self) -> Result<Self, CoreError>
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
        Ok(Self::new(lhs.codomain.clone(), rhs.domain.clone()))
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
        if lhs_contracting_axes.len() != rhs_contracting_axes.len() {
            return Err(CoreError::DimensionMismatch {
                expected: lhs_contracting_axes.len(),
                actual: rhs_contracting_axes.len(),
            });
        }

        let lhs_seen = validate_axis_subset(lhs_contracting_axes, lhs.rank())?;
        let rhs_seen = validate_axis_subset(rhs_contracting_axes, rhs.rank())?;
        let lhs_open_axes = (0..lhs.rank())
            .filter(|&axis| !lhs_seen[axis])
            .collect::<Vec<_>>();
        let rhs_open_axes = (0..rhs.rank())
            .filter(|&axis| !rhs_seen[axis])
            .collect::<Vec<_>>();
        let output_rank = lhs_open_axes.len() + rhs_open_axes.len();
        validate_permutation(output_axes, output_rank)?;
        if dst_codomain_rank > output_rank {
            return Err(CoreError::StructureRankMismatch {
                expected: output_rank,
                actual: dst_codomain_rank,
            });
        }

        let lhs = lhs.permute(rule, &lhs_open_axes, lhs_contracting_axes)?;
        let rhs = rhs.permute(rule, rhs_contracting_axes, &rhs_open_axes)?;
        let composed = Self::compose(rule, &lhs, &rhs)?;
        composed.permute(
            rule,
            &output_axes[..dst_codomain_rank],
            &output_axes[dst_codomain_rank..],
        )
    }

    /// The cached fusion-tree block keys, shared in O(1) (`Arc::clone`): the
    /// layout already holds them as `Arc<[_]>`, so there is no need to deep-clone
    /// each key (two `FusionTreeKey`s, four `SectorVec`s each) into a fresh `Vec`
    /// on every call. Returns `Arc<[_]>`, which derefs to `[FusionTreeBlockKey]`,
    /// so iterate / index / `len` callers are unchanged; by-value consumers can
    /// `.to_vec()`. TensorKit's `fusiontrees(W)` likewise returns the cached
    /// index set by reference. See #53.
    pub fn fusion_tree_keys<R>(&self, rule: &R) -> Arc<[FusionTreeBlockKey]>
    where
        R: MultiplicityFreeFusionRule,
    {
        Arc::clone(&self.cached_fusion_tree_layout(rule).keys)
    }

    pub fn try_for_each_fusion_tree_key<R, F, E>(&self, rule: &R, mut f: F) -> Result<(), E>
    where
        R: MultiplicityFreeFusionRule,
        F: FnMut(&FusionTreeBlockKey) -> Result<(), E>,
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
        F: FnOnce(&[FusionTreeBlockKey]) -> Result<T, E>,
    {
        let layout = self.cached_fusion_tree_layout(rule);
        f(layout.keys.as_ref())
    }

    fn cached_fusion_tree_layout<R>(&self, rule: &R) -> Arc<FusionTreeHomSpaceLayout>
    where
        R: MultiplicityFreeFusionRule,
    {
        let key = FusionTreeHomSpaceCacheKey::new(rule, self);
        let cache = fusion_tree_layout_cache();
        if let Ok(read) = cache.read() {
            if let Some(layout) = read.get(&key) {
                return Arc::clone(layout);
            }
        }

        let computed = Arc::new(fusion_tree_layout_from_keys(
            rule,
            self.fusion_tree_keys_uncached(rule),
        ));
        if let Ok(mut write) = cache.write() {
            return Arc::clone(write.entry(key).or_insert_with(|| Arc::clone(&computed)));
        }
        computed
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
            layout_ptr: Arc::as_ptr(&layout) as usize,
            nout,
            rank,
            shapes: Arc::<[DimVec]>::from(shapes),
        };
        let cache = coupled_block_structure_cache();
        // Read-lock fast path uses `peek` (does not bump recency; `get` needs `&mut`).
        if let Ok(read) = cache.read() {
            if let Some(structure) = read.peek(&cache_key).and_then(Weak::upgrade) {
                return Ok(structure);
            }
        }

        let specs = coupled_sector_matrix_block_specs_from_layout(
            nout,
            rank,
            &layout,
            cache_key.shapes.as_ref(),
        )?;
        let structure = BlockStructure::from_blocks_with_rank(rank, specs)?.into_shared();

        let mut write = cache
            .write()
            .expect("coupled block structure cache poisoned");
        if let Some(existing) = write.get(&cache_key).and_then(Weak::upgrade) {
            return Ok(existing);
        }
        write.put(cache_key, Arc::downgrade(&structure));
        Ok(structure)
    }

    fn fusion_tree_keys_uncached<R>(&self, rule: &R) -> Vec<FusionTreeBlockKey>
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
                            keys.push(FusionTreeBlockKey::pair(
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
    ) -> Result<Vec<FusionTreeBlockKey>, CoreError>
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
    ) -> Result<Vec<FusionTreeBlockKey>, CoreError>
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
                keys.push(FusionTreeBlockKey::pair(
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
    ) -> Result<FusionTreeBlockKey, CoreError>
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
    ) -> Result<Vec<FusionTreeBlockKey>, CoreError>
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
                            keys.push(FusionTreeBlockKey::pair(
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
        keys: &[FusionTreeBlockKey],
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

fn validate_axis_subset(axes: &[usize], rank: usize) -> Result<Vec<bool>, CoreError> {
    let mut seen = vec![false; rank];
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
    keys: &[FusionTreeBlockKey],
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
        match self.rule_identity.as_ref() {
            Some(expected) if expected != &actual => Err(CoreError::FusionRuleMismatch {
                expected: expected.clone(),
                actual,
            }),
            Some(_) => Ok(self),
            None => {
                self.rule_identity = Some(actual);
                Ok(self)
            }
        }
    }

    pub fn try_inherit_rule_identity<const OTHER_NOUT: usize, const OTHER_NIN: usize>(
        mut self,
        source: &FusionTensorMapSpace<OTHER_NOUT, OTHER_NIN>,
    ) -> Result<Self, CoreError> {
        match (self.rule_identity.as_ref(), source.rule_identity.as_ref()) {
            (Some(expected), Some(actual)) if expected != actual => {
                Err(CoreError::FusionRuleMismatch {
                    expected: expected.clone(),
                    actual: actual.clone(),
                })
            }
            (None, Some(identity)) => {
                self.rule_identity = Some(identity.clone());
                Ok(self)
            }
            (Some(_), Some(_)) => Ok(self),
            (_, None) => Err(CoreError::MissingFusionRuleIdentity),
        }
    }

    fn with_rule_identity(mut self, identity: RuleIdentity) -> Self {
        self.rule_identity = Some(identity);
        self
    }

    pub fn find_subblock_index(&self, key: &FusionTreeBlockKey) -> Option<usize> {
        self.subblock_structure
            .find_block_index_by_fusion_tree_key(key)
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.subblock_structure.required_len()
    }
}
