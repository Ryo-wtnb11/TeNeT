use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) enum TreeTransformScope {
    AllCodomain,
    TreePair,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) struct TreeTransformStructureOperationKey<RuleKey> {
    pub(super) rule: RuleKey,
    pub(super) scope: TreeTransformScope,
    pub(super) operation: TreeTransformOperation,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) struct RuntimeTreeTransformOperationKey {
    pub(super) rule: RuleIdentity,
    pub(super) operation: TreeTransformOperation,
    pub(super) logical_source: Option<BlockStructureCacheKey>,
    // Why not fold this into `storage_conjugate`: a conjugated owned source
    // reads storage keys directly, while an adjoint-oriented source reads
    // them through the adjoint key and axis projection.
    pub(super) orientation: FusionTreePairOrientation,
}

pub(super) type RuntimeTreeTransformKey =
    TreeTransformStructureCacheKey<RuntimeTreeTransformOperationKey>;
type RuntimeTreeTransformLookup<T> = (Option<Arc<TreeTransformStructure<T>>>, u64);

#[derive(Clone)]
struct RuntimeTreeTransformStoreEntry<T> {
    structure: Arc<TreeTransformStructure<T>>,
    charged_bytes: usize,
    contents: ContentRefs,
    exact_layout: Option<RuntimeExactLayoutAdmission>,
}

/// The distinct interned [`BlockStructureContent`]s one entry retains
/// (destination, source, logical source), `0`-padded: content ids start at 1
/// and are never reused.
type ContentRefs = [usize; 3];

/// One entry's distinct content references and, for those the store does not
/// retain yet, their charges.
///
/// Why charge only the new ones: a content's charge walks its blocks, and the
/// plan and structure entries of one transform share theirs.
#[derive(Default)]
struct ContentCharges<'a> {
    ids: ContentRefs,
    contents: [Option<&'a Arc<BlockStructureContent>>; 3],
    bytes: [Option<usize>; 3],
}

impl<'a> ContentCharges<'a> {
    fn of(contents: impl IntoIterator<Item = &'a Arc<BlockStructureContent>>) -> Self {
        let mut charges = Self::default();
        let mut len = 0;
        for content in contents {
            let id = content.id();
            if !charges.ids[..len].contains(&id) {
                charges.ids[len] = id;
                charges.contents[len] = Some(content);
                len += 1;
            }
        }
        charges
    }

    /// Computes the charges of the slots `retained` does not cover. Callers
    /// run it outside the store lock.
    fn charge_unretained(&mut self, retained: [bool; 3]) {
        for (slot, retained) in retained.into_iter().enumerate() {
            if !retained {
                self.bytes(slot);
            }
        }
    }

    /// Slot `slot`'s charge, computed on first use; `0` for an empty slot.
    fn bytes(&mut self, slot: usize) -> usize {
        let content = self.contents[slot];
        *self.bytes[slot]
            .get_or_insert_with(|| content.map_or(0, |content| content.charged_retained_bytes()))
    }
}

/// Reference counts of the interned contents a store's entries retain.
///
/// Why one count per store rather than a charge per entry (#1998): every
/// structure- and plan-tier entry over the same HomSpaces retains the same
/// interned content, which a per-entry charge counted once per reference.
/// Each distinct content is charged once, to the Runtime ledger's content
/// account, from the first retaining admission until the last retaining
/// entry leaves. The account has the tiers' byte budget, so a store retains
/// at most four budgets: the three tiers' payloads and their contents.
/// Why not charge only a handle when the complete-HomSpace cache owns the
/// content: that cache evicts FIFO, after which an entry would keep the
/// content alive with no budget charged for it.
#[derive(Default)]
struct RuntimeContentLedger {
    counts: rustc_hash::FxHashMap<usize, (usize, usize)>,
    charged_bytes: usize,
}

impl RuntimeContentLedger {
    /// Which of `ids` some entry already retains (empty slots count as
    /// retained: there is nothing to charge).
    fn retains(&self, ids: &ContentRefs) -> [bool; 3] {
        ids.map(|id| id == 0 || self.counts.contains_key(&id))
    }

    /// Drops one reference to each of `ids`, releasing a content's charge with
    /// its last reference.
    fn release(&mut self, ledger: &RuntimeTreeTransformCacheLedger, ids: &ContentRefs) {
        for &id in ids.iter().filter(|&&id| id != 0) {
            let Some((refs, bytes)) = self.counts.get_mut(&id) else {
                continue;
            };
            *refs -= 1;
            if *refs == 0 {
                let bytes = *bytes;
                self.counts.remove(&id);
                self.charged_bytes = self.charged_bytes.saturating_sub(bytes);
                ledger.release(RuntimeCacheAccount::Contents, 1, bytes);
            }
        }
    }

    fn clear(&mut self, ledger: &RuntimeTreeTransformCacheLedger) {
        ledger.release(
            RuntimeCacheAccount::Contents,
            self.counts.len(),
            self.charged_bytes,
        );
        self.counts.clear();
        self.charged_bytes = 0;
    }
}

#[derive(Clone, Debug)]
struct RuntimeExactLayoutAdmission {
    source: RuntimeLayoutIdentity,
    destination: RuntimeLayoutIdentity,
}

#[derive(Clone, Debug)]
struct RuntimeLayoutIdentity {
    homspace: WeakHomSpaceId,
    structure: usize,
    nout: usize,
    nin: usize,
}

impl RuntimeLayoutIdentity {
    fn new(homspace: &HomSpaceId, layout: [usize; 3]) -> Self {
        Self {
            homspace: homspace.downgrade(),
            structure: layout[0],
            nout: layout[1],
            nin: layout[2],
        }
    }

    fn matches(&self, homspace: &HomSpaceId, layout: [usize; 3]) -> bool {
        self.structure == layout[0]
            && self.nout == layout[1]
            && self.nin == layout[2]
            && self.homspace.matches(homspace)
    }
}

/// Degeneracy-free identity of one block structure.
///
/// Equality and hashing read only [`SectorStructure`](tenet_core::SectorStructure):
/// the rank and the ordered block keys, whose fusion-tree pairs carry every
/// leg's sector and dual flag, the coupled sector, inner lines and vertices.
/// Why hold the interned content rather than a `SectorStructure` clone: every
/// completed-structure miss probes the plan tier, and a clone allocates per
/// block. The retained degeneracy part is charged, never compared.
#[derive(Clone, Debug)]
struct SectorKey(Arc<BlockStructureContent>);

impl SectorKey {
    fn of(structure: &BlockStructure) -> Self {
        Self(structure.content_key())
    }
}

impl PartialEq for SectorKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0.sector_structure() == other.0.sector_structure()
    }
}

impl Eq for SectorKey {}

impl Hash for SectorKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let sector = self.0.sector_structure();
        sector.rank().hash(state);
        sector.block_count().hash(state);
        for block in sector.blocks() {
            block.key().hash(state);
        }
    }
}

/// Key of one categorical [`TreeTransformGroupPlan`]: everything the plan and
/// its key-only preflight read, and no degeneracy, offset or intern id.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) struct CategoricalTransformKey {
    rule: RuleIdentity,
    operation: TreeTransformOperation,
    orientation: FusionTreePairOrientation,
    // Why keep it although no plan builder reads it: the key mirrors the
    // completed-structure key, so it can only split entries, never merge two.
    storage_conjugate: bool,
    dst: SectorKey,
    src: SectorKey,
    logical_source: Option<SectorKey>,
}

impl CategoricalTransformKey {
    pub(super) fn new(
        rule: RuleIdentity,
        operation: &TreeTransformOperation,
        orientation: FusionTreePairOrientation,
        storage_conjugate: bool,
        dst: &BlockStructure,
        src: &BlockStructure,
        logical_source: Option<&BlockStructure>,
    ) -> Self {
        Self {
            rule,
            operation: operation.clone(),
            orientation,
            storage_conjugate,
            dst: SectorKey::of(dst),
            src: SectorKey::of(src),
            logical_source: logical_source.map(SectorKey::of),
        }
    }

    /// The key's own bytes. Its sector contents are charged once per store,
    /// by `RuntimeContentLedger`.
    fn charged_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
            .saturating_add(self.rule.charged_retained_bytes())
            .saturating_add(self.operation.charged_retained_bytes())
    }
}

pub(super) fn charged_plan_bytes<T>(plan: &TreeTransformGroupPlan<T>) -> usize {
    const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();
    let mut backings = rustc_hash::FxHashSet::default();
    let specs = charged_spec_bytes(plan.specs(), &mut backings);
    (core::mem::size_of::<TreeTransformGroupPlan<T>>())
        .saturating_add(ARC_CONTROL_BYTES)
        .saturating_add(
            plan.spec_capacity()
                .saturating_mul(core::mem::size_of::<TreeTransformGroupBlockSpec<T>>()),
        )
        .saturating_add(specs)
        // Why charge the shared payload before it exists: the first layout
        // binding builds it after admission, and a charge never grows.
        .saturating_add(plan.charged_coefficient_payload_bytes())
}

/// Heap bytes of `specs` (excluding their inline structs), coefficients
/// included once.
fn charged_spec_bytes<T>(
    specs: &[TreeTransformGroupBlockSpec<T>],
    backings: &mut rustc_hash::FxHashSet<usize>,
) -> usize {
    const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();
    let key_bytes = core::mem::size_of::<FusionTreePairKey>();
    let mut bytes = 0usize;
    // A `Multi` spec holds its keys and coefficients in three shared heap
    // slices (dst, src, coefficients); a `Single` spec holds them inline,
    // where the caller's `size_of` charge already counts them. The group key
    // is derived from the source keys, whose backings it shares.
    for spec in specs {
        if spec.has_shared_slices() {
            let coefficients = spec.recoupling_coefficients_dst_src().len();
            bytes = bytes
                .saturating_add(3 * ARC_CONTROL_BYTES)
                .saturating_add(
                    (spec.dst_keys().len().saturating_add(spec.src_keys().len()))
                        .saturating_mul(key_bytes),
                )
                .saturating_add(coefficients.saturating_mul(core::mem::size_of::<T>()));
        }
        for key in spec.dst_keys().iter().chain(spec.src_keys()) {
            bytes = bytes.saturating_add(key.charge_retained_backings(backings));
        }
        if let Some(axes) = spec.source_axes() {
            if backings.insert(axes.as_ptr() as usize) {
                bytes = bytes
                    .saturating_add(ARC_CONTROL_BYTES)
                    .saturating_add(axes.len().saturating_mul(core::mem::size_of::<usize>()));
            }
        }
    }
    bytes
}

/// Everything one source group's specs read besides the group's own trees.
///
/// Why keep `storage_conjugate` although no group builder reads it: it
/// mirrors [`CategoricalTransformKey`], so it can only split entries.
#[derive(Debug, Eq, PartialEq)]
struct GroupContext {
    rule: RuleIdentity,
    operation: TreeTransformOperation,
    orientation: FusionTreePairOrientation,
    storage_conjugate: bool,
}

/// Key of one source fusion-tree group's transform specs, TensorKit's
/// `FSBBraidKey`/`FSBTransposeKey` per `FusionTreeBlock`: the context plus the
/// group's external sectors and its ordered source tree pairs. A TeNeT group
/// can hold a subset of the full `FusionTreeBlock`, hence the ordered trees.
///
/// The hash is computed once from the borrowed source keys; a lookup compares
/// through [`GroupKeyView`] and never clones the tree list.
struct CategoricalGroupKey {
    context: Arc<GroupContext>,
    group_key: FusionTreeGroupKey,
    src_keys: Box<[FusionTreePairKey]>,
    hash: u64,
}

/// Borrowed lookup form of a [`CategoricalGroupKey`].
struct GroupKeyRef<'a> {
    context: &'a GroupContext,
    group_key: &'a FusionTreeGroupKey,
    src_keys: &'a [&'a FusionTreePairKey],
    hash: u64,
}

trait GroupKeyView {
    fn key_hash(&self) -> u64;
    fn context(&self) -> &GroupContext;
    fn group_key(&self) -> &FusionTreeGroupKey;
    fn src_len(&self) -> usize;
    fn src_key(&self, index: usize) -> &FusionTreePairKey;
}

impl GroupKeyView for CategoricalGroupKey {
    fn key_hash(&self) -> u64 {
        self.hash
    }
    fn context(&self) -> &GroupContext {
        &self.context
    }
    fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }
    fn src_len(&self) -> usize {
        self.src_keys.len()
    }
    fn src_key(&self, index: usize) -> &FusionTreePairKey {
        &self.src_keys[index]
    }
}

impl GroupKeyView for GroupKeyRef<'_> {
    fn key_hash(&self) -> u64 {
        self.hash
    }
    fn context(&self) -> &GroupContext {
        self.context
    }
    fn group_key(&self) -> &FusionTreeGroupKey {
        self.group_key
    }
    fn src_len(&self) -> usize {
        self.src_keys.len()
    }
    fn src_key(&self, index: usize) -> &FusionTreePairKey {
        self.src_keys[index]
    }
}

impl Hash for dyn GroupKeyView + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.key_hash());
    }
}

impl PartialEq for dyn GroupKeyView + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.key_hash() == other.key_hash()
            && (std::ptr::eq(self.context(), other.context()) || self.context() == other.context())
            && self.group_key() == other.group_key()
            && self.src_len() == other.src_len()
            && (0..self.src_len()).all(|index| self.src_key(index) == other.src_key(index))
    }
}

impl Eq for dyn GroupKeyView + '_ {}

impl Hash for CategoricalGroupKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self as &dyn GroupKeyView).hash(state);
    }
}

impl PartialEq for CategoricalGroupKey {
    fn eq(&self, other: &Self) -> bool {
        (self as &dyn GroupKeyView) == (other as &dyn GroupKeyView)
    }
}

impl Eq for CategoricalGroupKey {}

impl<'a> Borrow<dyn GroupKeyView + 'a> for CategoricalGroupKey {
    fn borrow(&self) -> &(dyn GroupKeyView + 'a) {
        self
    }
}

#[derive(Clone)]
struct RuntimeGroupEntry<T> {
    specs: Arc<[TreeTransformGroupBlockSpec<T>]>,
    charged_bytes: usize,
}

impl<T> RuntimeCacheCharge for RuntimeGroupEntry<T> {
    fn charged_bytes(&self) -> usize {
        self.charged_bytes
    }
}

fn charged_group_entry_bytes<T>(
    key: &CategoricalGroupKey,
    entry_specs: &[TreeTransformGroupBlockSpec<T>],
) -> usize {
    const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();
    let mut backings = rustc_hash::FxHashSet::default();
    let mut bytes = core::mem::size_of::<CategoricalGroupKey>()
        .saturating_add(core::mem::size_of::<RuntimeGroupEntry<T>>())
        // ponytail: the shared context is charged in full to every entry.
        .saturating_add(ARC_CONTROL_BYTES)
        .saturating_add(core::mem::size_of::<GroupContext>())
        .saturating_add(key.context.rule.charged_retained_bytes())
        .saturating_add(key.context.operation.charged_retained_bytes())
        .saturating_add(key.group_key.charge_retained_backings(&mut backings))
        .saturating_add(
            key.src_keys
                .len()
                .saturating_mul(core::mem::size_of::<FusionTreePairKey>()),
        )
        .saturating_add(ARC_CONTROL_BYTES)
        .saturating_add(
            entry_specs
                .len()
                .saturating_mul(core::mem::size_of::<TreeTransformGroupBlockSpec<T>>()),
        )
        .saturating_add(RUNTIME_TREE_TRANSFORM_LRU_NODE_ALLOWANCE);
    for src in key.src_keys.iter() {
        bytes = bytes.saturating_add(src.charge_retained_backings(&mut backings));
    }
    let specs = charged_spec_bytes(entry_specs, &mut backings);
    bytes.saturating_add(specs)
}

/// One source group: its external-sector key and its ordered tree pairs.
pub(crate) type SourceGroup<'a> = (&'a FusionTreeGroupKey, &'a [&'a FusionTreePairKey]);

/// A freshly built source group with the key hash from its lookup.
pub(crate) type BuiltGroup<'a, T> = (u64, SourceGroup<'a>, Arc<[TreeTransformGroupBlockSpec<T>]>);

/// One source group's cache state during a plan build.
pub(crate) enum GroupSlot<T> {
    /// Cached specs of an unchanged group.
    Hit(Arc<[TreeTransformGroupBlockSpec<T>]>),
    /// The group must be built; carries its key hash for admission.
    Miss(u64),
}

/// Per-group reuse handle that a categorical-plan miss passes to its builder.
///
/// A plan miss (a sector change) builds only the source groups whose key is
/// absent, as TensorKit's `fsbraid`/`fstranspose` do per `FusionTreeBlock`.
/// Builders consult it only for non-Unique fusion: a Unique group is one tree
/// with one scalar, and TensorKit leaves that case uncached (`NoCache`).
pub(crate) struct GroupSpecReuse<'s, T> {
    store: &'s RuntimeTreeTransformStore<T>,
    context: Arc<GroupContext>,
    context_hash: u64,
    generation: u64,
}

impl<T> GroupSpecReuse<'_, T> {
    fn group_hash<'k>(
        &self,
        group_key: &FusionTreeGroupKey,
        src_keys: impl IntoIterator<Item = &'k FusionTreePairKey>,
    ) -> u64 {
        let mut hasher = rustc_hash::FxHasher::default();
        hasher.write_u64(self.context_hash);
        group_key.hash(&mut hasher);
        for key in src_keys {
            key.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Looks every group up under one lock, in order.
    pub(crate) fn lookup(&self, groups: &[SourceGroup<'_>]) -> Vec<GroupSlot<T>> {
        let hashes: Vec<u64> = groups
            .iter()
            .map(|(group_key, src_keys)| self.group_hash(group_key, src_keys.iter().copied()))
            .collect();
        let mut state = self.store.lock();
        let tier = &mut state.groups;
        groups
            .iter()
            .zip(hashes)
            .map(|(&(group_key, src_keys), hash)| {
                let view = GroupKeyRef {
                    context: &self.context,
                    group_key,
                    src_keys,
                    hash,
                };
                match tier.entries.get(&view as &dyn GroupKeyView) {
                    Some(entry) => {
                        tier.hits = tier.hits.saturating_add(1);
                        GroupSlot::Hit(Arc::clone(&entry.specs))
                    }
                    None => {
                        tier.misses = tier.misses.saturating_add(1);
                        GroupSlot::Miss(hash)
                    }
                }
            })
            .collect()
    }

    /// Admits freshly built groups; the tree list is cloned only here.
    pub(crate) fn admit(&self, built: Vec<BuiltGroup<'_, T>>) {
        let entries: Vec<_> = built
            .into_iter()
            .map(|(hash, (group_key, src_keys), specs)| {
                let key = CategoricalGroupKey {
                    context: Arc::clone(&self.context),
                    group_key: group_key.clone(),
                    src_keys: src_keys.iter().map(|key| (*key).clone()).collect(),
                    hash,
                };
                let charged_bytes = charged_group_entry_bytes(&key, &specs);
                (
                    key,
                    RuntimeGroupEntry {
                        specs,
                        charged_bytes,
                    },
                )
            })
            .collect();
        let mut state = self.store.lock();
        if state.generation != self.generation {
            return;
        }
        let state = &mut *state;
        for (key, entry) in entries {
            if !state.groups.entries.contains(&key) {
                state
                    .groups
                    .insert(&self.store.ledger, &mut state.contents, key, entry);
            }
        }
    }
}

/// A cache value whose admission charge is fixed at insertion.
trait RuntimeCacheCharge {
    fn charged_bytes(&self) -> usize;

    /// Interned contents this value retains; their charge is the store's
    /// [`RuntimeContentLedger`]'s, not this value's.
    fn contents(&self) -> ContentRefs {
        [0; 3]
    }
}

impl<T> RuntimeCacheCharge for RuntimeTreeTransformStoreEntry<T> {
    fn charged_bytes(&self) -> usize {
        self.charged_bytes
    }

    fn contents(&self) -> ContentRefs {
        self.contents
    }
}

#[derive(Clone)]
struct RuntimePlanEntry<T> {
    plan: Arc<TreeTransformGroupPlan<T>>,
    charged_bytes: usize,
    contents: ContentRefs,
}

impl<T> RuntimeCacheCharge for RuntimePlanEntry<T> {
    fn charged_bytes(&self) -> usize {
        self.charged_bytes
    }

    fn contents(&self) -> ContentRefs {
        self.contents
    }
}

/// Which Runtime-wide ledger account a tier charges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeCacheAccount {
    Structures,
    Plans,
    Groups,
    Contents,
}

/// One bounded LRU tier of a typed Runtime store.
struct RuntimeCacheTier<K, V> {
    entries: lru::LruCache<K, V, rustc_hash::FxBuildHasher>,
    account: RuntimeCacheAccount,
    entry_capacity: usize,
    byte_budget: usize,
    max_entry_bytes: usize,
    charged_payload_bytes: usize,
    hits: usize,
    misses: usize,
    evictions: usize,
    admission_bypasses: usize,
}

impl<K: Hash + Eq, V: RuntimeCacheCharge> RuntimeCacheTier<K, V> {
    fn new(
        account: RuntimeCacheAccount,
        entry_capacity: usize,
        byte_budget: usize,
        max_entry_bytes: usize,
    ) -> Self {
        Self {
            // Why unbounded: `insert` enforces `entry_capacity` itself, and a
            // bounded LRU would preallocate the 10⁴-entry group table eagerly.
            entries: lru::LruCache::unbounded_with_hasher(rustc_hash::FxBuildHasher),
            account,
            entry_capacity,
            byte_budget,
            max_entry_bytes,
            charged_payload_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            admission_bypasses: 0,
        }
    }

    fn info(&self) -> RuntimeTreeTransformCacheInfo {
        RuntimeTreeTransformCacheInfo {
            entries: self.entries.len(),
            entry_capacity: self.entry_capacity,
            charged_payload_bytes: self.charged_payload_bytes,
            byte_budget: self.byte_budget,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            admission_bypasses: self.admission_bypasses,
        }
    }

    /// Callers clear the store's [`RuntimeContentLedger`] with it.
    fn clear(&mut self, ledger: &RuntimeTreeTransformCacheLedger) {
        ledger.release(self.account, self.entries.len(), self.charged_payload_bytes);
        self.entries.clear();
        self.charged_payload_bytes = 0;
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.admission_bypasses = 0;
    }

    fn evict_lru(
        &mut self,
        ledger: &RuntimeTreeTransformCacheLedger,
        contents: &mut RuntimeContentLedger,
    ) -> bool {
        let Some((_, evicted)) = self.entries.pop_lru() else {
            return false;
        };
        let charged = evicted.charged_bytes();
        self.charged_payload_bytes = self.charged_payload_bytes.saturating_sub(charged);
        self.evictions = self.evictions.saturating_add(1);
        ledger.release(self.account, 1, charged);
        contents.release(ledger, &evicted.contents());
        true
    }

    /// Counts a bypass for a value no eviction could make room for. Callers
    /// ask before referencing its contents, so an oversized entry never evicts
    /// other entries for contents it then does not retain.
    fn bypasses_oversized(&mut self, charged: usize) -> bool {
        let oversized = charged > self.max_entry_bytes || charged > self.byte_budget;
        if oversized {
            self.admission_bypasses = self.admission_bypasses.saturating_add(1);
        }
        oversized
    }

    /// Admits `value` under the local and Runtime-wide limits, evicting LRU
    /// entries as needed. Returns `false` when the value bypassed retention;
    /// its content references are then the caller's to release.
    fn insert(
        &mut self,
        ledger: &RuntimeTreeTransformCacheLedger,
        contents: &mut RuntimeContentLedger,
        key: K,
        value: V,
    ) -> bool {
        let charged = value.charged_bytes();
        if self.bypasses_oversized(charged) {
            return false;
        }
        while self.entries.len() == self.entry_capacity
            || self.charged_payload_bytes.saturating_add(charged) > self.byte_budget
        {
            if !self.evict_lru(ledger, contents) {
                break;
            }
        }
        while !ledger.try_reserve(self.account, 1, charged) {
            if !self.evict_lru(ledger, contents) {
                self.admission_bypasses = self.admission_bypasses.saturating_add(1);
                return false;
            }
        }
        self.charged_payload_bytes = self.charged_payload_bytes.saturating_add(charged);
        self.entries.put(key, value);
        true
    }
}

struct RuntimeTreeTransformStoreState<T> {
    structures: RuntimeCacheTier<RuntimeTreeTransformKey, RuntimeTreeTransformStoreEntry<T>>,
    plans: RuntimeCacheTier<CategoricalTransformKey, RuntimePlanEntry<T>>,
    groups: RuntimeCacheTier<CategoricalGroupKey, RuntimeGroupEntry<T>>,
    contents: RuntimeContentLedger,
    generation: u64,
}

impl<T> RuntimeTreeTransformStoreState<T> {
    /// References `charges`' contents for an entry about to be admitted,
    /// charging the ones no entry retains yet. To make room it evicts LRU
    /// structures, then plans; already-retained contents are referenced first,
    /// so those evictions cannot release them. Returns `false`, with nothing
    /// referenced, when the new contents do not fit.
    fn reference_contents(
        &mut self,
        ledger: &RuntimeTreeTransformCacheLedger,
        charges: &mut ContentCharges<'_>,
    ) -> bool {
        let mut retained = [false; 3];
        for (slot, id) in charges.ids.iter().enumerate() {
            if let Some((refs, _)) = self.contents.counts.get_mut(id) {
                *refs += 1;
                retained[slot] = true;
            }
        }
        let ids = charges.ids;
        let new = || (0..3).filter(move |&slot| ids[slot] != 0 && !retained[slot]);
        // A content released since `charge_unretained` is charged here.
        let new_bytes = new().fold(0usize, |total, slot| {
            total.saturating_add(charges.bytes(slot))
        });
        let new_count = new().count();
        let mut fits = new_bytes <= ledger.byte_budget;
        while fits && !ledger.try_reserve(RuntimeCacheAccount::Contents, new_count, new_bytes) {
            fits = self.structures.evict_lru(ledger, &mut self.contents)
                || self.plans.evict_lru(ledger, &mut self.contents);
        }
        if !fits {
            let referenced =
                std::array::from_fn(|slot| if retained[slot] { charges.ids[slot] } else { 0 });
            self.contents.release(ledger, &referenced);
            return false;
        }
        for slot in new() {
            self.contents
                .counts
                .insert(ids[slot], (1, charges.bytes(slot)));
        }
        self.contents.charged_bytes = self.contents.charged_bytes.saturating_add(new_bytes);
        true
    }
}

const DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_ENTRIES: usize = 256;
/// Why a separate, larger cap: the other tiers hold one entry per sector
/// structure, this one one entry per fusion-tree group, and one structure has
/// many groups. TensorKit's `fsbraid`/`fstranspose` LRU default
/// (`caches.jl:DEFAULT_GLOBALCACHE_SIZE`) is the same 10⁴; the byte budget is
/// the binding limit.
const DEFAULT_RUNTIME_TREE_TRANSFORM_GROUP_ENTRIES: usize = 10_000;
const RUNTIME_TREE_TRANSFORM_LRU_NODE_ALLOWANCE: usize = 8 * core::mem::size_of::<usize>();

/// One Runtime-owned store for immutable tree-transform data of one
/// coefficient dtype, in three tiers:
///
/// - completed [`TreeTransformStructure`]s, keyed on the exact source and
///   destination layouts;
/// - categorical [`TreeTransformGroupPlan`]s, keyed on rule, operation and the
///   sector structures only. A completed-structure miss whose sector structure
///   was seen before binds the cached plan to the new layout instead of
///   recomputing recoupling coefficients;
/// - per-group recoupling specs (non-Unique fusion), keyed on rule, operation,
///   one source group's external sectors and its ordered tree pairs. A plan
///   miss rebuilds only the groups absent here.
///
/// Each tier is charged to its own Runtime-wide ledger account under the same
/// byte budget, so the store retains at most three times that budget.
///
/// Any entry that fits the budget alone is admitted, evicting LRU entries to
/// make room, as TensorKit's `GlobalLRUCache` retains every
/// `treetransposer`/`treebraider` (entry-count bound only, `caches.jl:19,162`
/// @cfaa073); TeNeT keeps the byte bound on top. The ceiling: an entry larger
/// than the budget bypasses retention and is rebuilt per call. Why no
/// per-entry limit below the budget (#1993): the former 8 MiB limit bypassed
/// every rank-5 U(1) `V^5 ← V^5` transform, so each warm call recompiled it;
/// the transform in use is the one worth keeping.
#[doc(hidden)]
pub struct RuntimeTreeTransformStore<T> {
    state: Mutex<RuntimeTreeTransformStoreState<T>>,
    ledger: Arc<RuntimeTreeTransformCacheLedger>,
}

/// Snapshot of one tier of a Runtime's tree-transform cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeTreeTransformCacheInfo {
    entries: usize,
    entry_capacity: usize,
    charged_payload_bytes: usize,
    byte_budget: usize,
    hits: usize,
    misses: usize,
    evictions: usize,
    admission_bypasses: usize,
}

#[derive(Debug)]
struct RuntimeTreeTransformCacheLedgerState {
    entries: usize,
    charged_payload_bytes: usize,
}

impl RuntimeTreeTransformCacheLedgerState {
    const EMPTY: Self = Self {
        entries: 0,
        charged_payload_bytes: 0,
    };
}

/// Shared cold-path accounting for Runtime-owned typed transform stores.
///
/// Each coefficient dtype keeps its own typed LRUs. This ledger only makes the
/// configured entry and byte limits one Runtime-wide limit per tier; warm
/// lookup never locks it.
#[doc(hidden)]
pub struct RuntimeTreeTransformCacheLedger {
    entry_capacity: usize,
    group_entry_capacity: usize,
    byte_budget: usize,
    state: Mutex<RuntimeTreeTransformCacheLedgerState>,
    plan_state: Mutex<RuntimeTreeTransformCacheLedgerState>,
    group_state: Mutex<RuntimeTreeTransformCacheLedgerState>,
    content_state: Mutex<RuntimeTreeTransformCacheLedgerState>,
}

impl RuntimeTreeTransformCacheInfo {
    pub fn entries(self) -> usize {
        self.entries
    }

    pub fn entry_capacity(self) -> usize {
        self.entry_capacity
    }

    /// Conservative cache-owned payload charge, not resident-memory usage.
    pub fn charged_payload_bytes(self) -> usize {
        self.charged_payload_bytes
    }

    pub fn byte_budget(self) -> usize {
        self.byte_budget
    }

    pub fn hits(self) -> usize {
        self.hits
    }

    /// Lookups that found no entry. For the categorical tier each miss starts
    /// one plan build, whether that build succeeds or returns an error.
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

impl RuntimeTreeTransformCacheLedger {
    #[doc(hidden)]
    pub fn new(byte_budget: usize) -> Self {
        Self::with_group_limits(
            DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_ENTRIES,
            DEFAULT_RUNTIME_TREE_TRANSFORM_GROUP_ENTRIES,
            byte_budget,
        )
    }

    pub(super) fn with_limits(entry_capacity: usize, byte_budget: usize) -> Self {
        Self::with_group_limits(entry_capacity, entry_capacity, byte_budget)
    }

    fn with_group_limits(
        entry_capacity: usize,
        group_entry_capacity: usize,
        byte_budget: usize,
    ) -> Self {
        assert!(
            entry_capacity != 0 && group_entry_capacity != 0,
            "tree-transform cache capacity is nonzero"
        );
        Self {
            entry_capacity,
            group_entry_capacity,
            byte_budget,
            state: Mutex::new(RuntimeTreeTransformCacheLedgerState::EMPTY),
            plan_state: Mutex::new(RuntimeTreeTransformCacheLedgerState::EMPTY),
            group_state: Mutex::new(RuntimeTreeTransformCacheLedgerState::EMPTY),
            content_state: Mutex::new(RuntimeTreeTransformCacheLedgerState::EMPTY),
        }
    }

    fn account(
        &self,
        account: RuntimeCacheAccount,
    ) -> std::sync::MutexGuard<'_, RuntimeTreeTransformCacheLedgerState> {
        match account {
            RuntimeCacheAccount::Structures => &self.state,
            RuntimeCacheAccount::Plans => &self.plan_state,
            RuntimeCacheAccount::Groups => &self.group_state,
            RuntimeCacheAccount::Contents => &self.content_state,
        }
        .lock()
        .expect("runtime tree-transform cache ledger poisoned")
    }

    fn entry_capacity_of(&self, account: RuntimeCacheAccount) -> usize {
        match account {
            RuntimeCacheAccount::Structures | RuntimeCacheAccount::Plans => self.entry_capacity,
            RuntimeCacheAccount::Groups => self.group_entry_capacity,
            // Bounded by the byte budget alone: a content is retained only
            // through a structure or plan entry, whose tiers cap the count.
            RuntimeCacheAccount::Contents => usize::MAX,
        }
    }

    fn try_reserve(
        &self,
        account: RuntimeCacheAccount,
        entries: usize,
        charged_bytes: usize,
    ) -> bool {
        let mut state = self.account(account);
        if state.entries.saturating_add(entries) > self.entry_capacity_of(account)
            || state.charged_payload_bytes.saturating_add(charged_bytes) > self.byte_budget
        {
            return false;
        }
        state.entries += entries;
        state.charged_payload_bytes = state.charged_payload_bytes.saturating_add(charged_bytes);
        true
    }

    fn release(&self, account: RuntimeCacheAccount, entries: usize, charged_bytes: usize) {
        let mut state = self.account(account);
        state.entries = state.entries.saturating_sub(entries);
        state.charged_payload_bytes = state.charged_payload_bytes.saturating_sub(charged_bytes);
    }

    fn pair_info(
        &self,
        account: RuntimeCacheAccount,
        first: RuntimeTreeTransformCacheInfo,
        second: Option<RuntimeTreeTransformCacheInfo>,
    ) -> RuntimeTreeTransformCacheInfo {
        let state = self.account(account);
        let mut combined = RuntimeTreeTransformCacheInfo {
            entries: state.entries,
            entry_capacity: self.entry_capacity_of(account),
            charged_payload_bytes: state.charged_payload_bytes,
            byte_budget: self.byte_budget,
            ..RuntimeTreeTransformCacheInfo::default()
        };
        for store in [Some(first), second].into_iter().flatten() {
            combined.hits = combined.hits.saturating_add(store.hits);
            combined.misses = combined.misses.saturating_add(store.misses);
            combined.evictions = combined.evictions.saturating_add(store.evictions);
            combined.admission_bypasses = combined
                .admission_bypasses
                .saturating_add(store.admission_bypasses);
        }
        combined
    }

    fn same_store<T, U>(
        first: &RuntimeTreeTransformStore<T>,
        second: &RuntimeTreeTransformStore<U>,
    ) -> bool {
        std::ptr::eq(
            first as *const RuntimeTreeTransformStore<T> as *const (),
            second as *const RuntimeTreeTransformStore<U> as *const (),
        )
    }

    /// Aggregates two typed-store metric snapshots while reporting shared
    /// resources once. Stores and the ledger are sampled separately, so the
    /// result is intentionally not an atomic cross-store snapshot.
    #[doc(hidden)]
    pub fn store_pair_info<T, U>(
        &self,
        first: &RuntimeTreeTransformStore<T>,
        second: &RuntimeTreeTransformStore<U>,
    ) -> RuntimeTreeTransformCacheInfo {
        let second_info = (!Self::same_store(first, second)).then(|| second.info());
        self.pair_info(RuntimeCacheAccount::Structures, first.info(), second_info)
    }

    /// Categorical-plan sibling of [`Self::store_pair_info`].
    #[doc(hidden)]
    pub fn plan_pair_info<T, U>(
        &self,
        first: &RuntimeTreeTransformStore<T>,
        second: &RuntimeTreeTransformStore<U>,
    ) -> RuntimeTreeTransformCacheInfo {
        let second_info = (!Self::same_store(first, second)).then(|| second.plan_info());
        self.pair_info(RuntimeCacheAccount::Plans, first.plan_info(), second_info)
    }

    /// Categorical-group sibling of [`Self::store_pair_info`].
    #[doc(hidden)]
    pub fn group_pair_info<T, U>(
        &self,
        first: &RuntimeTreeTransformStore<T>,
        second: &RuntimeTreeTransformStore<U>,
    ) -> RuntimeTreeTransformCacheInfo {
        let second_info = (!Self::same_store(first, second)).then(|| second.group_info());
        self.pair_info(RuntimeCacheAccount::Groups, first.group_info(), second_info)
    }
}

impl<T> RuntimeTreeTransformStore<T> {
    #[doc(hidden)]
    pub const DEFAULT_BYTE_BUDGET: usize = 64 * 1024 * 1024;

    pub fn new(byte_budget: usize) -> Self {
        Self::with_limits(
            DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_ENTRIES,
            byte_budget,
            byte_budget,
        )
    }

    pub(super) fn with_limits(
        entry_capacity: usize,
        byte_budget: usize,
        max_entry_bytes: usize,
    ) -> Self {
        let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(
            entry_capacity,
            byte_budget,
        ));
        Self::with_shared_ledger(ledger, max_entry_bytes)
    }

    /// Builds one typed store charged to a Runtime-wide ledger.
    #[doc(hidden)]
    pub fn with_runtime_ledger(ledger: Arc<RuntimeTreeTransformCacheLedger>) -> Self {
        let byte_budget = ledger.byte_budget;
        Self::with_shared_ledger(ledger, byte_budget)
    }

    fn with_shared_ledger(
        ledger: Arc<RuntimeTreeTransformCacheLedger>,
        max_entry_bytes: usize,
    ) -> Self {
        let (capacity, budget) = (ledger.entry_capacity, ledger.byte_budget);
        let structures = RuntimeCacheTier::new(
            RuntimeCacheAccount::Structures,
            capacity,
            budget,
            max_entry_bytes,
        );
        let plans = RuntimeCacheTier::new(
            RuntimeCacheAccount::Plans,
            capacity,
            budget,
            max_entry_bytes,
        );
        let groups = RuntimeCacheTier::new(
            RuntimeCacheAccount::Groups,
            ledger.group_entry_capacity,
            budget,
            max_entry_bytes,
        );
        Self {
            state: Mutex::new(RuntimeTreeTransformStoreState {
                structures,
                plans,
                groups,
                contents: RuntimeContentLedger::default(),
                generation: 0,
            }),
            ledger,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RuntimeTreeTransformStoreState<T>> {
        self.state
            .lock()
            .expect("runtime tree-transform store poisoned")
    }

    /// Completed-structure tier activity.
    pub fn info(&self) -> RuntimeTreeTransformCacheInfo {
        self.lock().structures.info()
    }

    /// Categorical-plan tier activity; `misses` counts attempted plan builds,
    /// including builds that returned an error.
    pub fn plan_info(&self) -> RuntimeTreeTransformCacheInfo {
        self.lock().plans.info()
    }

    /// Categorical-group tier activity; `misses` counts failed group lookups
    /// inside plan-tier misses, each of which is followed by one group build
    /// unless an earlier group's build fails.
    pub fn group_info(&self) -> RuntimeTreeTransformCacheInfo {
        self.lock().groups.info()
    }

    /// Clears every tier and its counters.
    pub fn clear(&self) {
        let mut state = self.lock();
        state.generation = state.generation.wrapping_add(1);
        state.structures.clear(&self.ledger);
        state.plans.clear(&self.ledger);
        state.groups.clear(&self.ledger);
        state.contents.clear(&self.ledger);
    }

    pub(super) fn charged_entry_bytes(
        key: &RuntimeTreeTransformKey,
        structure: &TreeTransformStructure<T>,
    ) -> usize {
        const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();

        // The retained block-structure contents are charged once per store,
        // by `RuntimeContentLedger`, not here.
        core::mem::size_of::<RuntimeTreeTransformKey>()
            .saturating_add(core::mem::size_of::<RuntimeTreeTransformStoreEntry<T>>())
            .saturating_add(key.plan().rule.charged_retained_bytes())
            .saturating_add(key.plan().operation.charged_retained_bytes())
            .saturating_add(structure.charged_payload_bytes())
            .saturating_add(ARC_CONTROL_BYTES)
            .saturating_add(RUNTIME_TREE_TRANSFORM_LRU_NODE_ALLOWANCE)
    }

    fn charged_plan_entry_bytes(
        key: &CategoricalTransformKey,
        plan: &TreeTransformGroupPlan<T>,
    ) -> usize {
        key.charged_bytes()
            .saturating_add(core::mem::size_of::<RuntimePlanEntry<T>>())
            .saturating_add(charged_plan_bytes(plan))
            .saturating_add(RUNTIME_TREE_TRANSFORM_LRU_NODE_ALLOWANCE)
    }

    fn lookup(
        &self,
        key: &RuntimeTreeTransformKey,
    ) -> (Option<Arc<TreeTransformStructure<T>>>, u64) {
        let mut state = self.lock();
        let generation = state.generation;
        let tier = &mut state.structures;
        if let Some(entry) = tier.entries.get(key) {
            let structure = Arc::clone(&entry.structure);
            tier.hits = tier.hits.saturating_add(1);
            return (Some(structure), generation);
        }
        tier.misses = tier.misses.saturating_add(1);
        (None, generation)
    }

    fn admit(
        &self,
        key: RuntimeTreeTransformKey,
        structure: Arc<TreeTransformStructure<T>>,
        generation: u64,
    ) -> Arc<TreeTransformStructure<T>> {
        let charged_bytes = Self::charged_entry_bytes(&key, &structure);
        let mut contents = ContentCharges::of(
            [
                Some(key.dst()),
                Some(key.src()),
                key.plan().logical_source.as_ref(),
            ]
            .into_iter()
            .flatten()
            .map(BlockStructureCacheKey::content),
        );
        let retained = self.lock().contents.retains(&contents.ids);
        contents.charge_unretained(retained);
        let mut state = self.lock();
        if let Some(entry) = state.structures.entries.get(&key) {
            return Arc::clone(&entry.structure);
        }
        if state.generation != generation {
            return structure;
        }
        if state.structures.bypasses_oversized(charged_bytes) {
            return structure;
        }
        if !state.reference_contents(&self.ledger, &mut contents) {
            state.structures.admission_bypasses =
                state.structures.admission_bypasses.saturating_add(1);
            return structure;
        }
        let ids = contents.ids;
        let state = &mut *state;
        if !state.structures.insert(
            &self.ledger,
            &mut state.contents,
            key,
            RuntimeTreeTransformStoreEntry {
                structure: Arc::clone(&structure),
                charged_bytes,
                contents: ids,
                exact_layout: None,
            },
        ) {
            state.contents.release(&self.ledger, &ids);
        }
        structure
    }

    pub(super) fn get_or_compile<E>(
        &self,
        key: RuntimeTreeTransformKey,
        compile: impl FnOnce() -> Result<Arc<TreeTransformStructure<T>>, E>,
    ) -> Result<Arc<TreeTransformStructure<T>>, E> {
        let (cached, generation) = self.lookup(&key);
        if let Some(cached) = cached {
            return Ok(cached);
        }
        let structure = compile()?;
        Ok(self.admit(key, structure, generation))
    }

    /// Returns the categorical plan for `key`, building and admitting it on a
    /// miss. A hit skips `build` entirely, including its key-only preflight:
    /// the key holds every input that preflight reads.
    pub(super) fn get_or_build_plan<E>(
        &self,
        key: CategoricalTransformKey,
        build: impl FnOnce(&GroupSpecReuse<'_, T>) -> Result<TreeTransformGroupPlan<T>, E>,
    ) -> Result<Arc<TreeTransformGroupPlan<T>>, E> {
        let generation = {
            let mut state = self.lock();
            let generation = state.generation;
            let tier = &mut state.plans;
            if let Some(entry) = tier.entries.get(&key) {
                let plan = Arc::clone(&entry.plan);
                tier.hits = tier.hits.saturating_add(1);
                return Ok(plan);
            }
            tier.misses = tier.misses.saturating_add(1);
            generation
        };
        let context = GroupContext {
            rule: key.rule.clone(),
            operation: key.operation.clone(),
            orientation: key.orientation,
            storage_conjugate: key.storage_conjugate,
        };
        let mut hasher = rustc_hash::FxHasher::default();
        context.rule.hash(&mut hasher);
        context.operation.hash(&mut hasher);
        context.orientation.hash(&mut hasher);
        context.storage_conjugate.hash(&mut hasher);
        let reuse = GroupSpecReuse {
            store: self,
            context_hash: hasher.finish(),
            context: Arc::new(context),
            generation,
        };
        let plan = Arc::new(build(&reuse)?);
        let charged_bytes = Self::charged_plan_entry_bytes(&key, &plan);
        let mut contents = ContentCharges::of(
            [Some(&key.dst), Some(&key.src), key.logical_source.as_ref()]
                .into_iter()
                .flatten()
                .map(|content| &content.0),
        );
        let retained = self.lock().contents.retains(&contents.ids);
        contents.charge_unretained(retained);
        let mut state = self.lock();
        if let Some(entry) = state.plans.entries.get(&key) {
            return Ok(Arc::clone(&entry.plan));
        }
        if state.generation != generation {
            return Ok(plan);
        }
        if state.plans.bypasses_oversized(charged_bytes) {
            return Ok(plan);
        }
        if !state.reference_contents(&self.ledger, &mut contents) {
            state.plans.admission_bypasses = state.plans.admission_bypasses.saturating_add(1);
            return Ok(plan);
        }
        let ids = contents.ids;
        let state = &mut *state;
        if !state.plans.insert(
            &self.ledger,
            &mut state.contents,
            key,
            RuntimePlanEntry {
                plan: Arc::clone(&plan),
                charged_bytes,
                contents: ids,
            },
        ) {
            state.contents.release(&self.ledger, &ids);
        }
        Ok(plan)
    }

    pub(crate) fn lookup_checked_generic(
        &self,
        rule: RuleIdentity,
        operation: &TreeTransformOperation,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        logical_src_structure: Option<&BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<RuntimeTreeTransformLookup<T>, OperationError> {
        let key = TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
            RuntimeTreeTransformOperationKey {
                rule,
                operation: operation.clone(),
                orientation: FusionTreePairOrientation::Direct,
                logical_source: logical_src_structure
                    .map(BlockStructureCacheKey::from_structure)
                    .transpose()?,
            },
            dst_structure,
            src_structure,
            storage_conjugate,
        )?;
        let mut state = self.lock();
        let generation = state.generation;
        let tier = &mut state.structures;
        let matched = if tier.entries.contains(&key) {
            Some(key.clone())
        } else {
            // ponytail: the Runtime LRU is capped at 256 entries. Generic
            // previews deliberately have no intern identity before commit, so
            // a bounded semantic scan avoids a second key/index hierarchy.
            tier.entries.iter().find_map(|(candidate, _)| {
                (candidate.plan().rule == key.plan().rule
                    && candidate.plan().operation == key.plan().operation
                    && candidate.plan().orientation == key.plan().orientation
                    && match (&candidate.plan().logical_source, &key.plan().logical_source) {
                        (Some(candidate), Some(key)) => candidate.same_content(key),
                        (None, None) => true,
                        _ => false,
                    }
                    && candidate.storage_conjugate() == key.storage_conjugate()
                    && candidate.src().same_content(key.src())
                    && candidate.dst().same_content(key.dst()))
                .then(|| candidate.clone())
            })
        };
        if let Some(matched) = matched {
            let structure = Arc::clone(
                &tier
                    .entries
                    .get(&matched)
                    .expect("matched Runtime transform entry")
                    .structure,
            );
            tier.hits = tier.hits.saturating_add(1);
            return Ok((Some(structure), generation));
        }
        tier.misses = tier.misses.saturating_add(1);
        Ok((None, generation))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn admit_checked_generic(
        &self,
        rule: RuleIdentity,
        operation: &TreeTransformOperation,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        logical_src_structure: Option<&BlockStructure>,
        storage_conjugate: bool,
        structure: Arc<TreeTransformStructure<T>>,
        generation: u64,
    ) -> Result<(), OperationError> {
        let key = TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
            RuntimeTreeTransformOperationKey {
                rule,
                operation: operation.clone(),
                orientation: FusionTreePairOrientation::Direct,
                logical_source: logical_src_structure
                    .map(BlockStructureCacheKey::from_structure)
                    .transpose()?,
            },
            dst_structure,
            src_structure,
            storage_conjugate,
        )?;
        self.admit(key, structure, generation);
        Ok(())
    }

    /// Checked-Generic entry to the categorical tier. The key takes the same
    /// structures as [`Self::lookup_checked_generic`]; the plan reads the
    /// logical source, which is `src_structure` unless `logical_src_structure`
    /// is given.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_build_checked_generic_plan<E>(
        &self,
        rule: RuleIdentity,
        operation: &TreeTransformOperation,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        logical_src_structure: Option<&BlockStructure>,
        storage_conjugate: bool,
        build: impl FnOnce(&GroupSpecReuse<'_, T>) -> Result<TreeTransformGroupPlan<T>, E>,
    ) -> Result<Arc<TreeTransformGroupPlan<T>>, E> {
        self.get_or_build_plan(
            CategoricalTransformKey::new(
                rule,
                operation,
                FusionTreePairOrientation::Direct,
                storage_conjugate,
                dst_structure,
                src_structure,
                logical_src_structure,
            ),
            build,
        )
    }

    /// Returns a previously admitted exact-layout operation without rebuilding
    /// its owned runtime-rank axis description.
    #[doc(hidden)]
    pub fn admitted_tree_pair_operation(
        &self,
        rule: &RuleIdentity,
        source_homspace: &HomSpaceId,
        source_layout: [usize; 3],
        destination_homspace: &HomSpaceId,
        destination_layout: [usize; 3],
        mut matches: impl FnMut(&TreeTransformOperation) -> bool,
    ) -> Option<TreeTransformOperation> {
        let state = self.lock();
        // ponytail: the Runtime LRU is capped at 256 entries. A bounded scan
        // avoids a second index and borrowed-key hierarchy; add one only if a
        // profile shows this lookup, rather than replay, on the hot path.
        state.structures.entries.iter().find_map(|(key, entry)| {
            (entry.exact_layout.as_ref().is_some_and(|admission| {
                admission.source.matches(source_homspace, source_layout)
                    && admission
                        .destination
                        .matches(destination_homspace, destination_layout)
            }) && !key.storage_conjugate()
                && &key.plan().rule == rule
                && matches(&key.plan().operation))
            .then(|| key.plan().operation.clone())
        })
    }

    /// Marks one completed tree-pair entry as having passed exact typed layout
    /// admission. Missing or evicted entries intentionally retain no proof.
    #[doc(hidden)]
    pub fn admit_exact_tree_pair_layout(
        &self,
        rule: RuleIdentity,
        operation: &TreeTransformOperation,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        source: (&HomSpaceId, [usize; 3]),
        destination: (&HomSpaceId, [usize; 3]),
    ) -> Result<bool, OperationError> {
        let key = TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
            RuntimeTreeTransformOperationKey {
                rule,
                operation: operation.clone(),
                orientation: FusionTreePairOrientation::Direct,
                logical_source: None,
            },
            dst_structure,
            src_structure,
            false,
        )?;
        let mut state = self.lock();
        let Some(entry) = state.structures.entries.get_mut(&key) else {
            return Ok(false);
        };
        entry.exact_layout = Some(RuntimeExactLayoutAdmission {
            source: RuntimeLayoutIdentity::new(source.0, source.1),
            destination: RuntimeLayoutIdentity::new(destination.0, destination.1),
        });
        Ok(true)
    }
}

#[cfg(test)]
impl<T> RuntimeTreeTransformStore<T> {
    /// The content-account charge `key`'s distinct contents add to an empty
    /// store.
    pub(super) fn charged_content_bytes(key: &RuntimeTreeTransformKey) -> usize {
        let mut charges = ContentCharges::of(
            [
                Some(key.dst()),
                Some(key.src()),
                key.plan().logical_source.as_ref(),
            ]
            .into_iter()
            .flatten()
            .map(BlockStructureCacheKey::content),
        );
        (0..3).map(|slot| charges.bytes(slot)).sum()
    }

    /// Bytes this store currently charges to the content account.
    pub(super) fn retained_content_bytes(&self) -> usize {
        self.lock().contents.charged_bytes
    }
}

impl<T> Default for RuntimeTreeTransformStore<T> {
    fn default() -> Self {
        Self::new(Self::DEFAULT_BYTE_BUDGET)
    }
}

impl<T> Drop for RuntimeTreeTransformStore<T> {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (account, entries, bytes) in [
            (
                RuntimeCacheAccount::Structures,
                state.structures.entries.len(),
                state.structures.charged_payload_bytes,
            ),
            (
                RuntimeCacheAccount::Plans,
                state.plans.entries.len(),
                state.plans.charged_payload_bytes,
            ),
            (
                RuntimeCacheAccount::Groups,
                state.groups.entries.len(),
                state.groups.charged_payload_bytes,
            ),
            (
                RuntimeCacheAccount::Contents,
                state.contents.counts.len(),
                state.contents.charged_bytes,
            ),
        ] {
            self.ledger.release(account, entries, bytes);
        }
    }
}
