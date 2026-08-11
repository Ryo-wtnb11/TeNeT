use core::ops::{Add, Mul};
use std::fmt;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, Weak};

use num_traits::Zero;
use rustc_hash::FxHashMap;
use tenet_core::{
    BlockStructure, CategoricalScalar, FusionTreePairKey, FusionTreePairOrientation,
    GenericRigidSymbols, HomSpaceId, LocallyValidatedFusionTreeBlockStructure,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, RuleIdentity, TensorMap,
    TensorStorage, WeakHomSpaceId,
};

use crate::cache::{OperationCachePolicy, TreeTransformStructureCacheKey};
use crate::{OperationError, TreeTransformStructure, TreeTransformStructureCache};

use super::operation::{TreeTransformOperation, TreeTransformRuleCacheKey};
use super::plan::{
    build_all_codomain_tree_transform_group_plan_validated_with_threads,
    build_generic_tree_pair_transform_group_plan_validated,
    build_oriented_tree_pair_transform_group_plan_with_threads,
    compile_multiplicity_free_tree_pair_structure_after_capability_with_threads,
    compile_multiplicity_free_tree_pair_structure_with_threads,
    validate_all_codomain_namespace_before_cache, validate_generic_tree_pair_preflight,
    validate_multiplicity_free_all_codomain_preflight_after_capability,
    validate_multiplicity_free_tree_transform_capability,
    validate_tree_pair_namespace_before_cache,
};
#[cfg(test)]
use super::plan::{
    build_tree_pair_transform_group_plan_validated_with_threads,
    validate_multiplicity_free_tree_pair_preflight,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum TreeTransformScope {
    AllCodomain,
    TreePair,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TreeTransformStructureOperationKey<RuleKey> {
    rule: RuleKey,
    scope: TreeTransformScope,
    operation: TreeTransformOperation,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct RuntimeTreeTransformOperationKey {
    rule: RuleIdentity,
    operation: TreeTransformOperation,
}

type RuntimeTreeTransformKey = TreeTransformStructureCacheKey<RuntimeTreeTransformOperationKey>;
type RuntimeTreeTransformLookup<T> = (Option<Arc<TreeTransformStructure<T>>>, u64);

#[derive(Clone)]
struct RuntimeTreeTransformStoreEntry<T> {
    structure: Arc<TreeTransformStructure<T>>,
    charged_bytes: usize,
    exact_layout: Option<RuntimeExactLayoutAdmission>,
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

struct RuntimeTreeTransformStoreState<T> {
    entries: lru::LruCache<
        RuntimeTreeTransformKey,
        RuntimeTreeTransformStoreEntry<T>,
        rustc_hash::FxBuildHasher,
    >,
    entry_capacity: usize,
    byte_budget: usize,
    max_entry_bytes: usize,
    charged_payload_bytes: usize,
    generation: u64,
    hits: usize,
    misses: usize,
    evictions: usize,
    admission_bypasses: usize,
}

const DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_ENTRIES: usize = 256;
const DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_MAX_ENTRY_BYTES: usize = 8 * 1024 * 1024;
const RUNTIME_TREE_TRANSFORM_LRU_NODE_ALLOWANCE: usize = 8 * core::mem::size_of::<usize>();

/// One Runtime-owned store for completed immutable tree-pair structures.
#[doc(hidden)]
pub struct RuntimeTreeTransformStore<T> {
    state: Mutex<RuntimeTreeTransformStoreState<T>>,
    ledger: Arc<RuntimeTreeTransformCacheLedger>,
}

/// Snapshot of one Runtime's completed tree-transform cache.
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

/// Shared cold-path accounting for Runtime-owned typed transform stores.
///
/// Each coefficient dtype keeps its own typed LRU. This ledger only makes the
/// configured entry and byte limits one Runtime-wide limit; warm lookup never
/// locks it.
#[doc(hidden)]
pub struct RuntimeTreeTransformCacheLedger {
    entry_capacity: usize,
    byte_budget: usize,
    state: Mutex<RuntimeTreeTransformCacheLedgerState>,
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
        Self::with_limits(DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_ENTRIES, byte_budget)
    }

    fn with_limits(entry_capacity: usize, byte_budget: usize) -> Self {
        assert!(
            entry_capacity != 0,
            "tree-transform cache capacity is nonzero"
        );
        Self {
            entry_capacity,
            byte_budget,
            state: Mutex::new(RuntimeTreeTransformCacheLedgerState {
                entries: 0,
                charged_payload_bytes: 0,
            }),
        }
    }

    fn try_reserve(&self, charged_bytes: usize) -> bool {
        let mut state = self
            .state
            .lock()
            .expect("runtime tree-transform cache ledger poisoned");
        if state.entries == self.entry_capacity
            || state.charged_payload_bytes.saturating_add(charged_bytes) > self.byte_budget
        {
            return false;
        }
        state.entries += 1;
        state.charged_payload_bytes = state.charged_payload_bytes.saturating_add(charged_bytes);
        true
    }

    fn release(&self, entries: usize, charged_bytes: usize) {
        let mut state = self
            .state
            .lock()
            .expect("runtime tree-transform cache ledger poisoned");
        state.entries = state.entries.saturating_sub(entries);
        state.charged_payload_bytes = state.charged_payload_bytes.saturating_sub(charged_bytes);
    }

    /// Takes one coherent snapshot across two typed stores while reporting
    /// shared resources once.
    #[doc(hidden)]
    pub fn store_pair_info<T, U>(
        &self,
        first: &RuntimeTreeTransformStore<T>,
        second: &RuntimeTreeTransformStore<U>,
    ) -> RuntimeTreeTransformCacheInfo {
        // Same order as admission/clear: typed store(s), then ledger. No path
        // takes the ledger first, so a snapshot cannot deadlock a writer.
        let first = first
            .state
            .lock()
            .expect("runtime tree-transform store poisoned");
        let second = second
            .state
            .lock()
            .expect("runtime tree-transform store poisoned");
        let state = self
            .state
            .lock()
            .expect("runtime tree-transform cache ledger poisoned");
        let mut combined = RuntimeTreeTransformCacheInfo {
            entries: state.entries,
            entry_capacity: self.entry_capacity,
            charged_payload_bytes: state.charged_payload_bytes,
            byte_budget: self.byte_budget,
            ..RuntimeTreeTransformCacheInfo::default()
        };
        for store in [first.info(), second.info()] {
            combined.hits = combined.hits.saturating_add(store.hits);
            combined.misses = combined.misses.saturating_add(store.misses);
            combined.evictions = combined.evictions.saturating_add(store.evictions);
            combined.admission_bypasses = combined
                .admission_bypasses
                .saturating_add(store.admission_bypasses);
        }
        combined
    }
}

impl<T> RuntimeTreeTransformStoreState<T> {
    fn new(entry_capacity: usize, byte_budget: usize, max_entry_bytes: usize) -> Self {
        Self {
            entries: lru::LruCache::with_hasher(
                NonZeroUsize::new(entry_capacity)
                    .expect("tree-transform cache capacity is nonzero"),
                rustc_hash::FxBuildHasher,
            ),
            entry_capacity,
            byte_budget,
            max_entry_bytes,
            charged_payload_bytes: 0,
            generation: 0,
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
}

impl<T> RuntimeTreeTransformStore<T> {
    #[doc(hidden)]
    pub const DEFAULT_BYTE_BUDGET: usize = 64 * 1024 * 1024;

    pub fn new(byte_budget: usize) -> Self {
        Self::with_limits(
            DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_ENTRIES,
            byte_budget,
            DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_MAX_ENTRY_BYTES,
        )
    }

    fn with_limits(entry_capacity: usize, byte_budget: usize, max_entry_bytes: usize) -> Self {
        let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(
            entry_capacity,
            byte_budget,
        ));
        Self::with_shared_ledger(ledger, max_entry_bytes)
    }

    /// Builds one typed store charged to a Runtime-wide ledger.
    #[doc(hidden)]
    pub fn with_runtime_ledger(ledger: Arc<RuntimeTreeTransformCacheLedger>) -> Self {
        Self::with_shared_ledger(ledger, DEFAULT_RUNTIME_TREE_TRANSFORM_CACHE_MAX_ENTRY_BYTES)
    }

    fn with_shared_ledger(
        ledger: Arc<RuntimeTreeTransformCacheLedger>,
        max_entry_bytes: usize,
    ) -> Self {
        Self {
            state: Mutex::new(RuntimeTreeTransformStoreState::new(
                ledger.entry_capacity,
                ledger.byte_budget,
                max_entry_bytes,
            )),
            ledger,
        }
    }

    pub fn info(&self) -> RuntimeTreeTransformCacheInfo {
        self.state
            .lock()
            .expect("runtime tree-transform store poisoned")
            .info()
    }

    pub fn clear(&self) {
        let mut state = self
            .state
            .lock()
            .expect("runtime tree-transform store poisoned");
        let entries = state.entries.len();
        let charged_payload_bytes = state.charged_payload_bytes;
        state.generation = state.generation.wrapping_add(1);
        state.entries.clear();
        state.charged_payload_bytes = 0;
        state.hits = 0;
        state.misses = 0;
        state.evictions = 0;
        state.admission_bypasses = 0;
        self.ledger.release(entries, charged_payload_bytes);
    }

    fn charged_entry_bytes(
        key: &RuntimeTreeTransformKey,
        structure: &TreeTransformStructure<T>,
    ) -> usize {
        const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();

        let mut dependent_structure_bytes = key.dst().charged_retained_bytes();
        if key.src().id() != key.dst().id() {
            dependent_structure_bytes =
                dependent_structure_bytes.saturating_add(key.src().charged_retained_bytes());
        }

        core::mem::size_of::<RuntimeTreeTransformKey>()
            .saturating_add(core::mem::size_of::<RuntimeTreeTransformStoreEntry<T>>())
            .saturating_add(key.plan().rule.charged_retained_bytes())
            .saturating_add(key.plan().operation.charged_retained_bytes())
            .saturating_add(structure.charged_payload_bytes())
            .saturating_add(dependent_structure_bytes)
            .saturating_add(ARC_CONTROL_BYTES)
            .saturating_add(RUNTIME_TREE_TRANSFORM_LRU_NODE_ALLOWANCE)
    }

    fn lookup(
        &self,
        key: &RuntimeTreeTransformKey,
    ) -> (Option<Arc<TreeTransformStructure<T>>>, u64) {
        let mut state = self
            .state
            .lock()
            .expect("runtime tree-transform store poisoned");
        if let Some(entry) = state.entries.get(key) {
            let structure = Arc::clone(&entry.structure);
            state.hits = state.hits.saturating_add(1);
            return (Some(structure), state.generation);
        }
        state.misses = state.misses.saturating_add(1);
        (None, state.generation)
    }

    fn admit(
        &self,
        key: RuntimeTreeTransformKey,
        structure: Arc<TreeTransformStructure<T>>,
        generation: u64,
    ) -> Arc<TreeTransformStructure<T>> {
        let charged_bytes = Self::charged_entry_bytes(&key, &structure);
        let mut state = self
            .state
            .lock()
            .expect("runtime tree-transform store poisoned");
        if let Some(entry) = state.entries.get(&key) {
            return Arc::clone(&entry.structure);
        }
        if state.generation != generation {
            return structure;
        }
        if charged_bytes > state.max_entry_bytes || charged_bytes > state.byte_budget {
            state.admission_bypasses = state.admission_bypasses.saturating_add(1);
            return structure;
        }
        while state.entries.len() == state.entry_capacity
            || state.charged_payload_bytes.saturating_add(charged_bytes) > state.byte_budget
        {
            let Some((_, evicted)) = state.entries.pop_lru() else {
                break;
            };
            state.charged_payload_bytes = state
                .charged_payload_bytes
                .saturating_sub(evicted.charged_bytes);
            state.evictions = state.evictions.saturating_add(1);
            self.ledger.release(1, evicted.charged_bytes);
        }
        if !self.ledger.try_reserve(charged_bytes) {
            state.admission_bypasses = state.admission_bypasses.saturating_add(1);
            return structure;
        }
        state.charged_payload_bytes = state.charged_payload_bytes.saturating_add(charged_bytes);
        state.entries.put(
            key,
            RuntimeTreeTransformStoreEntry {
                structure: Arc::clone(&structure),
                charged_bytes,
                exact_layout: None,
            },
        );
        structure
    }

    fn get_or_compile<E>(
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

    pub(crate) fn lookup_checked_generic(
        &self,
        rule: RuleIdentity,
        operation: &TreeTransformOperation,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<RuntimeTreeTransformLookup<T>, OperationError> {
        let key = TreeTransformStructureCacheKey::from_structures(
            RuntimeTreeTransformOperationKey {
                rule,
                operation: operation.clone(),
            },
            dst_structure,
            src_structure,
        )?;
        let mut state = self
            .state
            .lock()
            .expect("runtime tree-transform store poisoned");
        let matched = if state.entries.contains(&key) {
            Some(key.clone())
        } else {
            // ponytail: the Runtime LRU is capped at 256 entries. Generic
            // previews deliberately have no intern identity before commit, so
            // a bounded semantic scan avoids a second key/index hierarchy.
            state.entries.iter().find_map(|(candidate, _)| {
                (candidate.plan() == key.plan()
                    && candidate.storage_conjugate() == key.storage_conjugate()
                    && candidate.src().same_content(key.src())
                    && candidate.dst().same_content(key.dst()))
                .then(|| candidate.clone())
            })
        };
        if let Some(matched) = matched {
            let structure = Arc::clone(
                &state
                    .entries
                    .get(&matched)
                    .expect("matched Runtime transform entry")
                    .structure,
            );
            state.hits = state.hits.saturating_add(1);
            return Ok((Some(structure), state.generation));
        }
        state.misses = state.misses.saturating_add(1);
        Ok((None, state.generation))
    }

    pub(crate) fn admit_checked_generic(
        &self,
        rule: RuleIdentity,
        operation: &TreeTransformOperation,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        structure: Arc<TreeTransformStructure<T>>,
        generation: u64,
    ) -> Result<(), OperationError> {
        let key = TreeTransformStructureCacheKey::from_structures(
            RuntimeTreeTransformOperationKey {
                rule,
                operation: operation.clone(),
            },
            dst_structure,
            src_structure,
        )?;
        self.admit(key, structure, generation);
        Ok(())
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
        let state = self
            .state
            .lock()
            .expect("runtime tree-transform store poisoned");
        // ponytail: the Runtime LRU is capped at 256 entries. A bounded scan
        // avoids a second index and borrowed-key hierarchy; add one only if a
        // profile shows this lookup, rather than replay, on the hot path.
        state.entries.iter().find_map(|(key, entry)| {
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
            },
            dst_structure,
            src_structure,
            false,
        )?;
        let mut state = self
            .state
            .lock()
            .expect("runtime tree-transform store poisoned");
        let Some(entry) = state.entries.get_mut(&key) else {
            return Ok(false);
        };
        entry.exact_layout = Some(RuntimeExactLayoutAdmission {
            source: RuntimeLayoutIdentity::new(source.0, source.1),
            destination: RuntimeLayoutIdentity::new(destination.0, destination.1),
        });
        Ok(true)
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
        self.ledger
            .release(state.entries.len(), state.charged_payload_bytes);
    }
}

pub(crate) const DEFAULT_TREE_TRANSFORM_CACHE_ENTRIES: usize = 256;

/// Context-local retention for completed immutable tree-transform structures.
///
/// Standalone expert contexts may retain ordinary multiplicity-free and
/// all-codomain structures according to [`OperationCachePolicy`]. Runtime-bound
/// ordinary and checked Generic tree-pair operations use their Runtime-owned
/// store instead. Prelowered callback paths compile eagerly and are not retained
/// here.
pub struct TreeTransformCache<T, RuleKey> {
    structures: TreeTransformStructureCache<T, TreeTransformStructureOperationKey<RuleKey>>,
    runtime_store: Option<Weak<RuntimeTreeTransformStore<T>>>,
    policy: OperationCachePolicy,
    stats: TreeTransformCacheStats,
    recoupling_threads: usize,
}

impl<T, RuleKey> Clone for TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    fn clone(&self) -> Self {
        Self {
            structures: self.structures.clone(),
            runtime_store: self.runtime_store.clone(),
            policy: self.policy,
            stats: self.stats,
            recoupling_threads: self.recoupling_threads,
        }
    }
}

impl<T, RuleKey> fmt::Debug for TreeTransformCache<T, RuleKey> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TreeTransformCache")
            .field("policy", &self.policy)
            .field("runtime_bound", &self.runtime_store.is_some())
            .field("stats", &self.stats)
            .field("recoupling_threads", &self.recoupling_threads)
            .finish()
    }
}

pub type TreePairTransformCache<T, RuleKey> = TreeTransformCache<T, RuleKey>;

/// Observable completed-structure cache activity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TreeTransformCacheStats {
    structure_hits: usize,
    structure_misses: usize,
}

impl TreeTransformCacheStats {
    #[inline]
    pub fn structure_hits(self) -> usize {
        self.structure_hits
    }

    #[inline]
    pub fn structure_misses(self) -> usize {
        self.structure_misses
    }
}

/// Defaults to a context-local LRU of completed tree-transform structures.
/// Use [`Self::with_policy`] or
/// [`TreeTransformExecutionContext::set_cache_policy`](crate::TreeTransformExecutionContext::set_cache_policy)
/// to select no retention, unbounded context-local retention, or another cap.
impl<T, RuleKey> Default for TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    fn default() -> Self {
        let policy = OperationCachePolicy::task_local_lru(DEFAULT_TREE_TRANSFORM_CACHE_ENTRIES);
        Self {
            structures: TreeTransformStructureCache::with_policy(policy),
            runtime_store: None,
            policy,
            stats: TreeTransformCacheStats::default(),
            recoupling_threads: 1,
        }
    }
}

impl<T, RuleKey> TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_policy(policy: OperationCachePolicy) -> Self {
        Self {
            structures: TreeTransformStructureCache::with_policy(policy),
            runtime_store: None,
            policy,
            stats: TreeTransformCacheStats::default(),
            recoupling_threads: 1,
        }
    }

    #[inline]
    pub fn recoupling_threads(&self) -> usize {
        self.recoupling_threads
    }

    /// Sets the worker count used by whole-group categorical compilation.
    pub fn set_recoupling_threads(&mut self, threads: usize) {
        self.recoupling_threads = threads.max(1);
    }

    #[inline]
    pub fn policy(&self) -> OperationCachePolicy {
        self.policy
    }

    pub fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.structures.set_policy(policy);
    }

    #[inline]
    pub fn structure_len(&self) -> usize {
        self.structures.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty()
    }

    #[inline]
    /// Returns context-local activity only; Runtime-owned stores report through
    /// the user Runtime API.
    pub fn stats(&self) -> TreeTransformCacheStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = TreeTransformCacheStats::default();
    }

    pub(crate) fn runtime_store(&self) -> Option<Arc<RuntimeTreeTransformStore<T>>> {
        self.runtime_store.as_ref().and_then(Weak::upgrade)
    }

    fn structure_key(
        rule: RuleKey,
        scope: TreeTransformScope,
        operation: TreeTransformOperation,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        storage_conjugate: bool,
    ) -> Result<
        TreeTransformStructureCacheKey<TreeTransformStructureOperationKey<RuleKey>>,
        OperationError,
    > {
        TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
            TreeTransformStructureOperationKey {
                rule,
                scope,
                operation,
            },
            dst_structure,
            src_structure,
            storage_conjugate,
        )
    }

    fn cached_structure(
        &mut self,
        key: &TreeTransformStructureCacheKey<TreeTransformStructureOperationKey<RuleKey>>,
    ) -> Option<Arc<TreeTransformStructure<T>>> {
        let structure = self.structures.get_arc(key)?;
        self.stats.structure_hits += 1;
        self.structures.touch(key);
        Some(structure)
    }

    fn retain_structure(
        &mut self,
        key: TreeTransformStructureCacheKey<TreeTransformStructureOperationKey<RuleKey>>,
        structure: Arc<TreeTransformStructure<T>>,
    ) {
        self.structures.insert_arc(key, structure);
    }

    /// Resolve an exact tree-pair replay structure.
    ///
    /// Fusion-tree block keys in `dst` and `src` follow
    /// [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain
    /// precondition.
    pub fn get_or_compile_tree_pair<
        R,
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        self.get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            rule,
            &operation,
            dst.structure(),
            src.structure(),
            false,
        )
    }

    /// Structure-only variant of [`Self::get_or_compile_tree_pair`], with the
    /// same provider-domain precondition.
    pub fn get_or_compile_tree_pair_structures_with_storage_conjugation<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
    {
        self.get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            rule,
            &operation,
            dst_structure,
            src_structure,
            storage_conjugate,
        )
    }

    /// Borrowed-operation variant of
    /// [`Self::get_or_compile_tree_pair_structures_with_storage_conjugation`].
    pub fn get_or_compile_tree_pair_structures_with_storage_conjugation_ref<R>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
    {
        if let Some(runtime_store) = &self.runtime_store {
            let Some(store) = runtime_store.upgrade() else {
                return compile_multiplicity_free_tree_pair_structure_with_threads(
                    rule,
                    operation,
                    Arc::clone(dst_structure),
                    Arc::clone(src_structure),
                    storage_conjugate,
                    self.recoupling_threads,
                )
                .map(Arc::new);
            };
            validate_multiplicity_free_tree_transform_capability(rule, operation)?;
            validate_tree_pair_namespace_before_cache(operation, src_structure)?;
            let key = TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
                RuntimeTreeTransformOperationKey {
                    rule: rule.rule_identity(),
                    operation: operation.clone(),
                },
                dst_structure,
                src_structure,
                storage_conjugate,
            )?;
            return store.get_or_compile(key, || {
                compile_multiplicity_free_tree_pair_structure_after_capability_with_threads(
                    rule,
                    operation,
                    Arc::clone(dst_structure),
                    Arc::clone(src_structure),
                    storage_conjugate,
                    self.recoupling_threads,
                )
                .map(Arc::new)
            });
        }

        if !self.policy.stores_entries() {
            let structure = compile_multiplicity_free_tree_pair_structure_with_threads(
                rule,
                operation,
                Arc::clone(dst_structure),
                Arc::clone(src_structure),
                storage_conjugate,
                self.recoupling_threads,
            )
            .map(Arc::new)?;
            self.stats.structure_misses += 1;
            return Ok(structure);
        }

        validate_multiplicity_free_tree_transform_capability(rule, operation)?;
        validate_tree_pair_namespace_before_cache(operation, src_structure)?;
        let key = Self::structure_key(
            rule.tree_transform_rule_cache_key(),
            TreeTransformScope::TreePair,
            operation.clone(),
            dst_structure,
            src_structure,
            storage_conjugate,
        )?;
        if let Some(structure) = self.cached_structure(&key) {
            return Ok(structure);
        }

        let structure = Arc::new(
            compile_multiplicity_free_tree_pair_structure_after_capability_with_threads(
                rule,
                operation,
                Arc::clone(dst_structure),
                Arc::clone(src_structure),
                storage_conjugate,
                self.recoupling_threads,
            )?,
        );
        self.stats.structure_misses += 1;
        self.retain_structure(key, Arc::clone(&structure));
        Ok(structure)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_compile_tree_pair_prelowered<R, FBlock, FAxis>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        logical_src_structure: &Arc<BlockStructure>,
        storage_src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
        logical_to_storage_block: FBlock,
        logical_to_storage_axis: FAxis,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        FBlock: Fn(usize) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        let source_proof =
            validate_multiplicity_free_tree_pair_preflight(rule, operation, logical_src_structure)?;
        let logical_source_id = logical_src_structure.content_id();
        let storage_source_id = storage_src_structure.content_id();
        let destination_id = dst_structure.content_id();
        if storage_source_id != logical_source_id {
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, storage_src_structure)
                .map_err(OperationError::from_core_preserving_context)?;
        }
        if destination_id != logical_source_id && destination_id != storage_source_id {
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
                .map_err(OperationError::from_core_preserving_context)?;
        }
        self.stats.structure_misses += 1;
        let plan = build_tree_pair_transform_group_plan_validated_with_threads(
            &source_proof,
            operation.clone(),
            self.recoupling_threads,
        )?;
        Ok(Arc::new(
            plan.compile_shared_structures_with_storage_mapping(
                Arc::clone(dst_structure),
                logical_src_structure,
                Arc::clone(storage_src_structure),
                logical_to_storage_block,
                logical_to_storage_axis,
                storage_conjugate,
            )?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_compile_tree_pair_oriented<R, FAxis>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        logical_keys: &[FusionTreePairKey],
        storage_indices: &[usize],
        storage_src_structure: &Arc<BlockStructure>,
        orientation: FusionTreePairOrientation,
        logical_rank: usize,
        logical_to_storage_axis: FAxis,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        if logical_keys.len() != storage_indices.len() {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented source projection",
            });
        }
        let mut projection =
            FxHashMap::with_capacity_and_hasher(logical_keys.len(), rustc_hash::FxBuildHasher);
        // Why not track storage-index uniqueness here: FusionOperandLayout
        // already proves this projection is a bijection onto parent blocks.
        for (position, (key, &storage_index)) in
            logical_keys.iter().zip(storage_indices).enumerate()
        {
            if storage_index >= storage_src_structure.block_count() {
                return Err(OperationError::BlockIndexOutOfBounds {
                    tensor: "oriented src",
                    index: storage_index,
                    count: storage_src_structure.block_count(),
                });
            }
            if projection.insert(key, storage_index).is_some() {
                return Err(OperationError::DuplicateTreeTransformKey {
                    tensor: "src",
                    index: position,
                });
            }
        }
        self.stats.structure_misses += 1;
        let plan = build_oriented_tree_pair_transform_group_plan_with_threads(
            rule,
            operation.clone(),
            logical_keys,
            storage_src_structure,
            orientation,
            logical_rank,
            &projection,
            self.recoupling_threads,
        )?;
        let source_index = |key: &FusionTreePairKey| {
            projection
                .get(key)
                .copied()
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: Box::new(tenet_core::BlockKey::FusionTree(key.clone())),
                })
        };
        Ok(Arc::new(
            plan.compile_shared_structures_with_source_projection(
                Arc::clone(dst_structure),
                Arc::clone(storage_src_structure),
                logical_rank,
                source_index,
                logical_to_storage_axis,
                orientation == FusionTreePairOrientation::Adjoint,
            )?,
        ))
    }

    /// Generic-fusion sibling of [`Self::get_or_compile_tree_pair`].
    ///
    /// This remains eager because completed-transformer retention for Generic
    /// fusion needs its own measured key and ownership contract. Why not retain
    /// a rule key here: provider-domain validation is the eager boundary, and
    /// this path retains no key or completed structure.
    pub fn get_or_compile_tree_pair_generic<
        R,
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: GenericRigidSymbols<Scalar = T>,
        R::Scalar: CategoricalScalar,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        let source_proof = validate_generic_tree_pair_preflight(rule, &operation, src.structure())?;
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
            .map_err(OperationError::from_core_preserving_context)?;
        self.stats.structure_misses += 1;
        let plan =
            build_generic_tree_pair_transform_group_plan_validated(&source_proof, operation)?;
        Ok(Arc::new(plan.compile(dst, src)?))
    }

    /// Structure-only Generic sibling. It has the same eager ownership and
    /// provider-domain contracts as [`Self::get_or_compile_tree_pair_generic`].
    pub fn get_or_compile_tree_pair_structures_generic<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: GenericRigidSymbols<Scalar = T>,
        R::Scalar: CategoricalScalar,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
    {
        let source_proof = validate_generic_tree_pair_preflight(rule, &operation, src_structure)?;
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
            .map_err(OperationError::from_core_preserving_context)?;
        self.stats.structure_misses += 1;
        let plan =
            build_generic_tree_pair_transform_group_plan_validated(&source_proof, operation)?;
        Ok(Arc::new(
            plan.compile_shared_structures_with_storage_conjugation(
                Arc::clone(dst_structure),
                Arc::clone(src_structure),
                false,
            )?,
        ))
    }

    /// Resolve an exact all-codomain replay structure.
    ///
    /// Fusion-tree block keys in `dst` and `src` follow
    /// [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain
    /// precondition.
    pub fn get_or_compile_all_codomain<
        R,
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeFusionSymbols<Scalar = T>
            + TreeTransformRuleCacheKey<Key = RuleKey>
            + Sync,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        validate_multiplicity_free_tree_transform_capability(rule, &operation)?;
        validate_all_codomain_namespace_before_cache(&operation, src.structure())?;

        let key = if self.policy.stores_entries() {
            let key = Self::structure_key(
                rule.tree_transform_rule_cache_key(),
                TreeTransformScope::AllCodomain,
                operation.clone(),
                dst.structure(),
                src.structure(),
                false,
            )?;
            if let Some(structure) = self.cached_structure(&key) {
                return Ok(structure);
            }
            Some(key)
        } else {
            None
        };

        let source_proof = validate_multiplicity_free_all_codomain_preflight_after_capability(
            rule,
            &operation,
            src.structure(),
        )?;
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
            .map_err(OperationError::from_core_preserving_context)?;
        self.stats.structure_misses += 1;
        let plan = build_all_codomain_tree_transform_group_plan_validated_with_threads(
            &source_proof,
            operation,
            self.recoupling_threads,
        )?;
        let structure = Arc::new(plan.compile_shared_structures_with_storage_conjugation(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            false,
        )?);
        if let Some(key) = key {
            self.retain_structure(key, Arc::clone(&structure));
        }
        Ok(structure)
    }
}

#[cfg(test)]
#[expect(
    clippy::items_after_test_module,
    reason = "co-location keeps the large private-helper tests out of the public surface"
)]
mod runtime_store_tests {
    use std::convert::Infallible;
    use std::sync::{Arc, Barrier};

    use num_complex::Complex64;
    use tenet_core::{BlockKey, BlockSpec, BlockStructure, FusionTreeHomSpace, RuleIdentity};

    use super::{
        RuntimeTreeTransformCacheLedger, RuntimeTreeTransformKey, RuntimeTreeTransformOperationKey,
        RuntimeTreeTransformStore,
    };
    use crate::{
        TreeTransformBlockSpec, TreeTransformOperation, TreeTransformStructure,
        TreeTransformStructureCacheKey,
    };

    struct TestRuleIdentity;

    fn fixture(
        tag: usize,
    ) -> (
        RuntimeTreeTransformKey,
        Arc<TreeTransformStructure<f64>>,
        BlockStructure,
    ) {
        fixture_with_rule(tag, RuleIdentity::of_type::<TestRuleIdentity>())
    }

    fn fixture_with_rule(
        tag: usize,
        rule: RuleIdentity,
    ) -> (
        RuntimeTreeTransformKey,
        Arc<TreeTransformStructure<f64>>,
        BlockStructure,
    ) {
        let block = BlockSpec::with_key(BlockKey::ordinal(tag), vec![1], vec![1], 0).unwrap();
        let structure = BlockStructure::from_blocks_with_rank(1, vec![block]).unwrap();
        let compiled = Arc::new(
            TreeTransformStructure::compile_structures(
                &structure,
                &structure,
                &[TreeTransformBlockSpec::single(0, 0, 1.0)],
            )
            .unwrap(),
        );
        let key = TreeTransformStructureCacheKey::from_structures(
            RuntimeTreeTransformOperationKey {
                rule,
                operation: TreeTransformOperation::permute([tag], []),
            },
            &structure,
            &structure,
        )
        .unwrap();
        (key, compiled, structure)
    }

    fn complex_fixture(
        tag: usize,
    ) -> (
        RuntimeTreeTransformKey,
        Arc<TreeTransformStructure<Complex64>>,
    ) {
        let block = BlockSpec::with_key(BlockKey::ordinal(tag), vec![1], vec![1], 0).unwrap();
        let structure = BlockStructure::from_blocks_with_rank(1, vec![block]).unwrap();
        let compiled = Arc::new(
            TreeTransformStructure::compile_structures(
                &structure,
                &structure,
                &[TreeTransformBlockSpec::single(
                    0,
                    0,
                    Complex64::new(0.0, 1.0),
                )],
            )
            .unwrap(),
        );
        let key = TreeTransformStructureCacheKey::from_structures(
            RuntimeTreeTransformOperationKey {
                rule: RuleIdentity::of_type::<TestRuleIdentity>(),
                operation: TreeTransformOperation::permute([tag], []),
            },
            &structure,
            &structure,
        )
        .unwrap();
        (key, compiled)
    }

    fn pair_fixture(
        dst_tag: usize,
        src_tag: usize,
    ) -> (
        RuntimeTreeTransformKey,
        Arc<TreeTransformStructure<f64>>,
        BlockStructure,
        BlockStructure,
    ) {
        let structure = |tag| {
            let block = BlockSpec::with_key(BlockKey::ordinal(tag), vec![1], vec![1], 0).unwrap();
            BlockStructure::from_blocks_with_rank(1, vec![block]).unwrap()
        };
        let dst = structure(dst_tag);
        let src = structure(src_tag);
        let compiled = Arc::new(
            TreeTransformStructure::compile_structures(
                &dst,
                &src,
                &[TreeTransformBlockSpec::single(0, 0, 1.0)],
            )
            .unwrap(),
        );
        let key = TreeTransformStructureCacheKey::from_structures(
            RuntimeTreeTransformOperationKey {
                rule: RuleIdentity::of_type::<TestRuleIdentity>(),
                operation: TreeTransformOperation::permute([0], []),
            },
            &dst,
            &src,
        )
        .unwrap();
        (key, compiled, dst, src)
    }

    #[test]
    fn shared_ledger_preserves_full_capacity_for_one_dtype() {
        // What: adding a second typed store does not partition the configured
        // resources; an f64-only workload can still occupy the complete cache.
        let (key0, structure0, _) = fixture(40);
        let (key1, structure1, _) = fixture(41);
        let charge0 = RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&key0, &structure0);
        let charge1 = RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&key1, &structure1);
        let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(
            2,
            charge0.saturating_add(charge1),
        ));
        let real = RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(&ledger));
        let complex =
            RuntimeTreeTransformStore::<Complex64>::with_runtime_ledger(Arc::clone(&ledger));

        real.get_or_compile(key0, || Ok::<_, Infallible>(structure0))
            .unwrap();
        real.get_or_compile(key1, || Ok::<_, Infallible>(structure1))
            .unwrap();

        let info = ledger.store_pair_info(&real, &complex);
        assert_eq!(info.entries(), 2);
        assert_eq!(info.entry_capacity(), 2);
        assert_eq!(info.byte_budget(), charge0 + charge1);
        assert_eq!(info.admission_bypasses(), 0);
    }

    #[test]
    fn concurrent_cross_dtype_admission_obeys_one_total_byte_budget() {
        // What: the two typed stores race through one atomic cold-path ledger;
        // exactly one entry fits and the other admission bypasses.
        let (real_key, real_structure, _) = fixture(42);
        let (complex_key, complex_structure) = complex_fixture(42);
        let real_charge =
            RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&real_key, &real_structure);
        let complex_charge = RuntimeTreeTransformStore::<Complex64>::charged_entry_bytes(
            &complex_key,
            &complex_structure,
        );
        let budget = real_charge.max(complex_charge);
        let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(2, budget));
        let real = Arc::new(RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(
            &ledger,
        )));
        let complex = Arc::new(RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(
            &ledger,
        )));
        let start = Arc::new(Barrier::new(3));

        let real_worker = {
            let store = Arc::clone(&real);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                store
                    .get_or_compile(real_key, || Ok::<_, Infallible>(real_structure))
                    .unwrap();
            })
        };
        let complex_worker = {
            let store = Arc::clone(&complex);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                store
                    .get_or_compile(complex_key, || Ok::<_, Infallible>(complex_structure))
                    .unwrap();
            })
        };
        start.wait();
        real_worker.join().unwrap();
        complex_worker.join().unwrap();

        let info = ledger.store_pair_info(real.as_ref(), complex.as_ref());
        assert_eq!(info.entries(), 1);
        assert!(info.charged_payload_bytes() <= budget);
        assert_eq!(info.byte_budget(), budget);
        assert_eq!(info.misses(), 2);
        assert_eq!(info.admission_bypasses(), 1);
    }

    #[test]
    fn coefficient_dtype_and_rule_identity_have_independent_keys() {
        // What: equal RuleIdentity values in distinct Rust store types cannot
        // cross-hit, while distinct identities in one dtype retain two entries.
        struct OtherRuleIdentity;

        let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(4, usize::MAX));
        let real = RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(&ledger));
        let complex = RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(&ledger));
        let (real_key, real_structure, _) = fixture(43);
        let (complex_key, complex_structure) = complex_fixture(43);
        real.get_or_compile(real_key.clone(), || {
            Ok::<_, Infallible>(Arc::clone(&real_structure))
        })
        .unwrap();
        complex
            .get_or_compile(complex_key.clone(), || {
                Ok::<_, Infallible>(Arc::clone(&complex_structure))
            })
            .unwrap();
        assert_eq!(real.info().misses(), 1);
        assert_eq!(complex.info().misses(), 1);

        let (other_key, other_structure, _) =
            fixture_with_rule(43, RuleIdentity::of_type::<OtherRuleIdentity>());
        real.get_or_compile(other_key.clone(), || {
            Ok::<_, Infallible>(Arc::clone(&other_structure))
        })
        .unwrap();
        real.get_or_compile(real_key, || -> Result<_, Infallible> {
            unreachable!("real entry is warm")
        })
        .unwrap();
        real.get_or_compile(other_key, || -> Result<_, Infallible> {
            unreachable!("other rule entry is warm")
        })
        .unwrap();
        complex
            .get_or_compile(complex_key, || -> Result<_, Infallible> {
                unreachable!("complex entry is warm")
            })
            .unwrap();

        let info = ledger.store_pair_info(&real, &complex);
        assert_eq!(info.entries(), 3);
        assert_eq!(info.misses(), 3);
        assert_eq!(info.hits(), 3);
    }

    #[test]
    fn shared_clear_blocks_old_generation_and_releases_all_charges() {
        // What: clear of both typed stores releases the shared ledger and a
        // compilation begun before clear cannot republish afterward.
        let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(2, usize::MAX));
        let real = Arc::new(RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(
            &ledger,
        )));
        let complex = Arc::new(RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(
            &ledger,
        )));
        let (real_key, real_structure, _) = fixture(44);
        real.get_or_compile(real_key, || Ok::<_, Infallible>(real_structure))
            .unwrap();

        let (complex_key, complex_structure) = complex_fixture(44);
        let started = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        let worker = {
            let store = Arc::clone(&complex);
            let started = Arc::clone(&started);
            let resume = Arc::clone(&resume);
            std::thread::spawn(move || {
                store
                    .get_or_compile(complex_key, || {
                        started.wait();
                        resume.wait();
                        Ok::<_, Infallible>(complex_structure)
                    })
                    .unwrap()
            })
        };

        started.wait();
        real.clear();
        complex.clear();
        resume.wait();
        assert_eq!(worker.join().unwrap().block_count(), 1);

        let info = ledger.store_pair_info(real.as_ref(), complex.as_ref());
        assert_eq!(info.entries(), 0);
        assert_eq!(info.charged_payload_bytes(), 0);
        assert_eq!(info.hits(), 0);
        assert_eq!(info.misses(), 0);
    }

    #[test]
    fn runtime_store_charges_and_releases_dependent_structures() {
        // What: one content owner is charged once, distinct source/destination
        // content affects admission, and eviction/clear release both owners.
        let (same_key, same_compiled, _, _) = pair_fixture(10, 10);
        let (distinct_key, distinct_compiled, _, _) = pair_fixture(11, 12);
        let same_charge =
            RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&same_key, &same_compiled);
        let distinct_charge = RuntimeTreeTransformStore::<f64>::charged_entry_bytes(
            &distinct_key,
            &distinct_compiled,
        );
        assert_eq!(
            distinct_charge,
            same_charge.saturating_add(distinct_key.src().charged_retained_bytes())
        );

        let budgeted = RuntimeTreeTransformStore::with_limits(
            2,
            distinct_charge.saturating_sub(1),
            usize::MAX,
        );
        budgeted
            .get_or_compile(same_key, || Ok::<_, Infallible>(same_compiled))
            .unwrap();
        budgeted
            .get_or_compile(distinct_key, || Ok::<_, Infallible>(distinct_compiled))
            .unwrap();
        assert_eq!(budgeted.info().entries(), 1);
        assert_eq!(budgeted.info().admission_bypasses(), 1);

        let store = RuntimeTreeTransformStore::with_limits(1, usize::MAX, usize::MAX);
        let (first_key, first_compiled, first_dst, first_src) = pair_fixture(20, 21);
        let first_dst_content = first_dst.content_key();
        let first_src_content = first_src.content_key();
        let first_dst_weak = Arc::downgrade(&first_dst_content);
        let first_src_weak = Arc::downgrade(&first_src_content);
        drop(first_dst_content);
        drop(first_src_content);
        drop(
            store
                .get_or_compile(first_key, || Ok::<_, Infallible>(first_compiled))
                .unwrap(),
        );
        drop(first_dst);
        drop(first_src);
        assert!(first_dst_weak.upgrade().is_some());
        assert!(first_src_weak.upgrade().is_some());

        let (second_key, second_compiled, second_dst, second_src) = pair_fixture(22, 23);
        let second_dst_content = second_dst.content_key();
        let second_src_content = second_src.content_key();
        let second_dst_weak = Arc::downgrade(&second_dst_content);
        let second_src_weak = Arc::downgrade(&second_src_content);
        drop(second_dst_content);
        drop(second_src_content);
        drop(
            store
                .get_or_compile(second_key, || Ok::<_, Infallible>(second_compiled))
                .unwrap(),
        );
        drop(second_dst);
        drop(second_src);
        assert!(first_dst_weak.upgrade().is_none());
        assert!(first_src_weak.upgrade().is_none());
        assert!(second_dst_weak.upgrade().is_some());
        assert!(second_src_weak.upgrade().is_some());

        store.clear();
        assert!(second_dst_weak.upgrade().is_none());
        assert!(second_src_weak.upgrade().is_none());
    }

    #[test]
    fn runtime_store_enforces_resources_and_clear_keeps_returned_arcs_valid() {
        // What: entry and byte pressure evict, oversized entries bypass, and
        // clear resets accounting without invalidating caller-owned payloads.
        let (key0, structure0, _) = fixture(0);
        let (key1, structure1, _) = fixture(1);
        let charge0 = RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&key0, &structure0);
        let charge1 = RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&key1, &structure1);

        let entry_limited = RuntimeTreeTransformStore::with_limits(1, usize::MAX, usize::MAX);
        let active = entry_limited
            .get_or_compile(key0.clone(), || {
                Ok::<_, Infallible>(Arc::clone(&structure0))
            })
            .unwrap();
        entry_limited
            .get_or_compile(key1.clone(), || {
                Ok::<_, Infallible>(Arc::clone(&structure1))
            })
            .unwrap();
        assert_eq!(entry_limited.info().entries(), 1);
        assert_eq!(entry_limited.info().evictions(), 1);
        assert_eq!(active.block_count(), 1);

        let byte_limited = RuntimeTreeTransformStore::with_limits(
            2,
            charge0.saturating_add(charge1).saturating_sub(1),
            usize::MAX,
        );
        byte_limited
            .get_or_compile(key0.clone(), || {
                Ok::<_, Infallible>(Arc::clone(&structure0))
            })
            .unwrap();
        byte_limited
            .get_or_compile(key1, || Ok::<_, Infallible>(Arc::clone(&structure1)))
            .unwrap();
        assert_eq!(byte_limited.info().entries(), 1);
        assert_eq!(byte_limited.info().evictions(), 1);

        let oversized =
            RuntimeTreeTransformStore::with_limits(2, usize::MAX, charge0.saturating_sub(1));
        oversized
            .get_or_compile(key0.clone(), || {
                Ok::<_, Infallible>(Arc::clone(&structure0))
            })
            .unwrap();
        assert_eq!(oversized.info().entries(), 0);
        assert_eq!(oversized.info().admission_bypasses(), 1);

        let disabled = RuntimeTreeTransformStore::new(0);
        disabled
            .get_or_compile(key0, || Ok::<_, Infallible>(Arc::clone(&structure0)))
            .unwrap();
        assert_eq!(disabled.info().entries(), 0);
        assert_eq!(disabled.info().byte_budget(), 0);
        assert_eq!(disabled.info().admission_bypasses(), 1);

        entry_limited.clear();
        let cleared = entry_limited.info();
        assert_eq!(cleared.entries(), 0);
        assert_eq!(cleared.entry_capacity(), 1);
        assert_eq!(cleared.charged_payload_bytes(), 0);
        assert_eq!(cleared.byte_budget(), usize::MAX);
        assert_eq!(cleared.hits(), 0);
        assert_eq!(cleared.misses(), 0);
        assert_eq!(cleared.evictions(), 0);
        assert_eq!(cleared.admission_bypasses(), 0);
        assert_eq!(active.block_count(), 1);
    }

    #[test]
    fn exact_layout_admission_requires_explicit_publication_and_clear_removes_it() {
        // What: an ordinary completed structure is not an exact typed-layout
        // proof; successful publication enables borrowed-operation reuse, and
        // clearing the one Runtime store removes both together.
        let (key, structure, layout) = fixture(0);
        let store = RuntimeTreeTransformStore::default();
        store
            .get_or_compile(key.clone(), || Ok::<_, Infallible>(structure))
            .unwrap();
        let source_homspace = FusionTreeHomSpace::from_sector_ids([(0, 1)], []);
        let destination_homspace = FusionTreeHomSpace::from_sector_ids([], [(0, 1)]);
        let source_id = source_homspace.id();
        let destination_id = destination_homspace.id();
        let rule = RuleIdentity::of_type::<TestRuleIdentity>();
        let operation = TreeTransformOperation::permute([0], []);
        let source_layout = [layout.content_id(), 1, 0];
        let destination_layout = [layout.content_id(), 0, 1];

        assert!(store
            .admitted_tree_pair_operation(
                &rule,
                &source_id,
                source_layout,
                &destination_id,
                destination_layout,
                |_| true,
            )
            .is_none());
        assert!(store
            .admit_exact_tree_pair_layout(
                rule.clone(),
                &operation,
                &layout,
                &layout,
                (&source_id, source_layout),
                (&destination_id, destination_layout),
            )
            .unwrap());
        assert_eq!(
            store
                .admitted_tree_pair_operation(
                    &rule,
                    &source_id,
                    source_layout,
                    &destination_id,
                    destination_layout,
                    |candidate| candidate == &operation,
                )
                .unwrap(),
            operation
        );
        let foreign_destination = FusionTreeHomSpace::from_sector_ids([], [(1, 1)]).id();
        assert!(store
            .admitted_tree_pair_operation(
                &rule,
                &source_id,
                source_layout,
                &foreign_destination,
                destination_layout,
                |_| true,
            )
            .is_none());

        store.clear();
        assert!(store
            .admitted_tree_pair_operation(
                &rule,
                &source_id,
                source_layout,
                &destination_id,
                destination_layout,
                |_| true,
            )
            .is_none());
    }

    #[test]
    fn clear_prevents_a_racing_old_generation_from_reinserting() {
        // What: a compiler that began before clear may finish for its caller but
        // cannot publish into the cleared Runtime generation.
        let store = Arc::new(RuntimeTreeTransformStore::with_limits(
            2,
            usize::MAX,
            usize::MAX,
        ));
        let (key, structure, _) = fixture(2);
        let next_key = key.clone();
        let next_structure = Arc::clone(&structure);
        let started = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        let worker_store = Arc::clone(&store);
        let worker_started = Arc::clone(&started);
        let worker_resume = Arc::clone(&resume);
        let worker = std::thread::spawn(move || {
            worker_store
                .get_or_compile(key, || {
                    worker_started.wait();
                    worker_resume.wait();
                    Ok::<_, Infallible>(structure)
                })
                .unwrap()
        });

        started.wait();
        store.clear();
        resume.wait();
        let returned = worker.join().unwrap();

        assert_eq!(returned.block_count(), 1);
        let cleared = store.info();
        assert_eq!(cleared.entries(), 0);
        assert_eq!(cleared.misses(), 0);
        store
            .get_or_compile(next_key, || Ok::<_, Infallible>(next_structure))
            .unwrap();
        let admitted = store.info();
        assert_eq!(admitted.entries(), 1);
        assert_eq!(admitted.misses(), 1);
        assert_eq!(admitted.hits(), 0);
    }

    #[test]
    fn checked_generic_lookup_survives_core_interner_reset() {
        // What: Runtime ownership, not the process-global interner lifetime,
        // controls a completed Generic structure's semantic reuse.
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let store = RuntimeTreeTransformStore::default();
        let (key, structure, original) = fixture(31);
        store
            .get_or_compile(key, || Ok::<_, Infallible>(structure))
            .unwrap();
        let original_id = original.content_id();

        tenet_core::reset_core_intern_tables();
        let rebuilt = BlockStructure::from_blocks_with_rank(
            1,
            vec![BlockSpec::with_key(BlockKey::ordinal(31), vec![1], vec![1], 0).unwrap()],
        )
        .unwrap();
        assert_ne!(rebuilt.content_id(), original_id);

        let (cached, _) = store
            .lookup_checked_generic(
                RuleIdentity::of_type::<TestRuleIdentity>(),
                &TreeTransformOperation::permute([31], []),
                &rebuilt,
                &rebuilt,
            )
            .unwrap();
        assert!(cached.is_some());
        assert_eq!(store.info().hits(), 1);
        assert_eq!(store.info().misses(), 1);
    }
}

impl<T, RuleKey> TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    /// Binds this cache lane to one Runtime-owned completed-structure store.
    #[doc(hidden)]
    pub fn bind_runtime_store(&mut self, store: Weak<RuntimeTreeTransformStore<T>>) {
        self.structures.set_policy(OperationCachePolicy::NoCache);
        self.policy = OperationCachePolicy::NoCache;
        self.runtime_store = Some(store);
    }
}
