//! The process-global composed-coefficient cache (cache 4 of #2014): the
//! transform specs of one source fusion-tree group, keyed on sectors and
//! trees only.
//!
//! Reference: TensorKit `cfaa073e` `braiding_manipulations.jl:fsbraid`
//! (`:288-339`) and `duality_manipulations.jl:fstranspose` (`:526-569`)
//! memoize `dst block => U` per `FusionTreeBlock` and `p` (and `levels`),
//! with `CacheStyle = NoCache()` for `UniqueFusion` and a global LRU
//! otherwise; `GenericTreeTransformer` (`treetransformers.jl:53-114`) looks
//! each source block up on a completed-transformer miss. QSpace `d2d3d7da`
//! `X3Map::contractDegQ` (`clebsch_aux.cc:4512-4560`) caches composed CGC
//! overlaps process-globally, degeneracy-free, and skips abelian `CRef`s
//! (`:4521-4526`); TeNeT borrows that technique, not its store.
//!
//! A cache-3 miss whose sectors were seen before (a degeneracy-only change,
//! or another HomSpace sharing a source group) therefore makes no F/R
//! provider call. Unique fusion bypasses this cache by builder dispatch: one
//! group is one tree with one scalar, so a lookup costs what it saves.
//!
//! Rust deviations, each deliberate:
//! - entries are byte-weighted under the per-kind budget of `tenet::cache`,
//!   not counted; a group heavier than one shard is rebuilt per miss;
//! - values are erased (`dyn Any`, the coefficient `TypeId` in the key);
//! - the key holds a group's ordered tree subset, since an expert structure
//!   may hold part of a `FusionTreeBlock`; the builder family (`mode`,
//!   `scope`) is keyed because the cache is shared by all of them;
//! - `storage_conjugate` is not keyed: no group builder reads it, binding
//!   applies it;
//! - no single-flight, a reset epoch, and built groups are staged in a
//!   [`PendingCoefficientGroups`] that the caller publishes only after its
//!   transaction succeeded (checked Generic: after the destination commit).

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use tenet_core::{
    ErasedStructureCacheControl, FusionTreeGroupKey, FusionTreePairKey, FusionTreePairOrientation,
    RuleIdentity, StructureCache, StructureCacheInfo, StructureCacheKind,
};

use super::{TransformerMode, TreeTransformScope, ENTRY_OVERHEAD_BYTES};
use crate::tree_transform::operation::TreeTransformOperation;
use crate::TreeTransformGroupBlockSpec;

#[cfg(test)]
#[path = "coefficients_tests.rs"]
mod tests;

type ErasedEntry = dyn Any + Send + Sync;

const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();
/// One shard, as caches 1-3: the largest group must fit it.
const COEFFICIENT_GROUP_SHARDS: usize = 1;

/// Everything one source group's specs read besides the group's own trees.
#[derive(Debug, Eq, Hash, PartialEq)]
struct GroupContext {
    rule: RuleIdentity,
    mode: TransformerMode,
    coefficient: TypeId,
    scope: TreeTransformScope,
    operation: TreeTransformOperation,
    orientation: FusionTreePairOrientation,
}

/// Key of one source group's composed specs: the context plus the group's
/// external sectors and duals and its ordered source tree pairs (coupled
/// sectors, inner lines, vertices, basis order). No degeneracy, offset,
/// block index or content id: no group builder reads them.
///
/// The hash is computed once; a lookup compares through [`GroupKeyRef`]
/// without cloning the tree list.
#[derive(Clone)]
pub(crate) struct CoefficientGroupKey {
    context: Arc<GroupContext>,
    group_key: FusionTreeGroupKey,
    src_keys: Arc<[FusionTreePairKey]>,
    hash: u64,
}

impl PartialEq for CoefficientGroupKey {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && (Arc::ptr_eq(&self.context, &other.context) || self.context == other.context)
            && self.group_key == other.group_key
            && self.src_keys == other.src_keys
    }
}

impl Eq for CoefficientGroupKey {}

impl Hash for CoefficientGroupKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

/// Borrowed lookup form of a [`CoefficientGroupKey`].
struct GroupKeyRef<'a> {
    context: &'a Arc<GroupContext>,
    group_key: &'a FusionTreeGroupKey,
    src_keys: &'a [&'a FusionTreePairKey],
    hash: u64,
}

impl Hash for GroupKeyRef<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl tenet_core::StructureCacheEquivalent<CoefficientGroupKey> for GroupKeyRef<'_> {
    fn equivalent(&self, key: &CoefficientGroupKey) -> bool {
        self.hash == key.hash
            && (Arc::ptr_eq(self.context, &key.context) || **self.context == *key.context)
            && *self.group_key == key.group_key
            && self.src_keys.len() == key.src_keys.len()
            && self
                .src_keys
                .iter()
                .zip(key.src_keys.iter())
                .all(|(lhs, rhs)| **lhs == *rhs)
    }
}

/// The cached value: one source group's specs, in builder order. A `Multi`
/// spec's recoupling matrix is a shared `Arc`, so a cache-3 core bound from
/// it shares those bytes.
pub(crate) struct CoefficientGroupEntry<T> {
    specs: Box<[TreeTransformGroupBlockSpec<T>]>,
}

impl<T> CoefficientGroupEntry<T> {
    pub(crate) fn specs(&self) -> &[TreeTransformGroupBlockSpec<T>] {
        &self.specs
    }
}

struct CoefficientGroupsControl;

impl ErasedStructureCacheControl for CoefficientGroupsControl {
    fn info(&self) -> StructureCacheInfo {
        coefficient_groups().info()
    }

    fn clear(&self) {
        ErasedStructureCacheControl::clear(coefficient_groups())
    }

    fn set_byte_budget(&self, bytes: u64) {
        ErasedStructureCacheControl::set_byte_budget(coefficient_groups(), bytes)
    }
}

fn coefficient_groups() -> &'static StructureCache<CoefficientGroupKey, ErasedEntry> {
    static CACHE: OnceLock<StructureCache<CoefficientGroupKey, ErasedEntry>> = OnceLock::new();
    CACHE.get_or_init(|| {
        // Refused only if another crate claimed the slot first; then nothing
        // is admitted and every group builds per miss, with equal results.
        let budget = tenet_core::register_structure_cache(
            StructureCacheKind::TreeTransformCoefficients,
            &CoefficientGroupsControl,
        )
        .unwrap_or(0);
        StructureCache::new(
            StructureCacheKind::TreeTransformCoefficients,
            budget,
            COEFFICIENT_GROUP_SHARDS,
        )
    })
}

#[cfg(test)]
std::thread_local! {
    static GROUP_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static GROUP_MISSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static GROUP_PUBLICATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// This thread's composed-coefficient activity since the last take.
/// Thread-local, so concurrent tests do not move it.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CoefficientGroupActivity {
    /// Group lookups answered by a resident entry.
    pub(crate) hits: usize,
    /// Group lookups that found none (each is built unless an earlier
    /// group's build fails).
    pub(crate) misses: usize,
    /// Built groups offered for publication.
    pub(crate) publications: usize,
}

#[cfg(test)]
pub(crate) fn take_coefficient_group_activity() -> CoefficientGroupActivity {
    CoefficientGroupActivity {
        hits: GROUP_HITS.replace(0),
        misses: GROUP_MISSES.replace(0),
        publications: GROUP_PUBLICATIONS.replace(0),
    }
}

/// One source group: its external-sector key and its ordered tree pairs.
pub(crate) type SourceGroup<'a> = (&'a FusionTreeGroupKey, &'a [&'a FusionTreePairKey]);

/// One source group's cache state during a plan build.
pub(crate) enum GroupSlot<T> {
    /// The resident specs of an unchanged group.
    Hit(Arc<CoefficientGroupEntry<T>>),
    /// The group must be built; carries its key hash for staging.
    Miss(u64),
}

/// Groups built during one resolution, not yet visible to other callers.
/// Errors are never staged; the owner publishes the batch once its
/// transaction succeeded, or drops it.
#[derive(Default)]
#[must_use = "built groups are cached only when published"]
pub(crate) struct PendingCoefficientGroups {
    groups: Vec<(CoefficientGroupKey, Arc<ErasedEntry>, u64)>,
}

impl PendingCoefficientGroups {
    /// Publishes every staged group built since reset epoch `epoch`; the
    /// first resident value of a key wins.
    pub(crate) fn publish(self, epoch: usize) {
        for (key, entry, bytes) in self.groups {
            #[cfg(test)]
            GROUP_PUBLICATIONS.set(GROUP_PUBLICATIONS.get() + 1);
            let _ = coefficient_groups().publish(&key, entry, bytes, epoch);
        }
    }
}

/// The checked Generic contraction's composed coefficients, staged across
/// its staged-operand and output transforms and published only by
/// [`Self::flush`] once the contraction's destination commit succeeded.
/// One instance per contraction call: holding it across calls would publish
/// groups of a failed call.
#[must_use = "built groups are cached only when flushed after the commit"]
pub(crate) struct CheckedPendingCoefficients {
    pending: PendingCoefficientGroups,
    epoch: usize,
}

impl CheckedPendingCoefficients {
    /// Captures the reset epoch: create it before the call's first build.
    pub(crate) fn new() -> Self {
        Self {
            pending: PendingCoefficientGroups::default(),
            epoch: tenet_core::core_reset_epoch(),
        }
    }

    pub(crate) fn stage(&mut self, built: PendingCoefficientGroups) {
        self.pending.groups.extend(built.groups);
    }

    /// Call only after the destination commit succeeded.
    pub(crate) fn flush(self) {
        self.pending.publish(self.epoch);
    }
}

/// Per-group reuse handle of one plan build: looks source groups up and
/// stages the built ones. Only non-Unique builders consult it.
pub(crate) struct CoefficientGroupReuse<T> {
    context: Arc<GroupContext>,
    context_hash: u64,
    built: RefCell<PendingCoefficientGroups>,
    coefficient: PhantomData<fn() -> T>,
}

impl<T> CoefficientGroupReuse<T>
where
    T: 'static + Send + Sync,
{
    pub(crate) fn new(
        rule: RuleIdentity,
        mode: TransformerMode,
        scope: TreeTransformScope,
        operation: &TreeTransformOperation,
        orientation: FusionTreePairOrientation,
    ) -> Self {
        let context = GroupContext {
            rule,
            mode,
            coefficient: TypeId::of::<T>(),
            scope,
            operation: operation.clone(),
            orientation,
        };
        let mut hasher = rustc_hash::FxHasher::default();
        context.hash(&mut hasher);
        Self {
            context_hash: hasher.finish(),
            context: Arc::new(context),
            built: RefCell::new(PendingCoefficientGroups::default()),
            coefficient: PhantomData,
        }
    }

    fn group_hash(&self, group_key: &FusionTreeGroupKey, src_keys: &[&FusionTreePairKey]) -> u64 {
        let mut hasher = rustc_hash::FxHasher::default();
        hasher.write_u64(self.context_hash);
        group_key.hash(&mut hasher);
        for key in src_keys {
            key.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Looks every group up, in order.
    pub(crate) fn lookup(&self, groups: &[SourceGroup<'_>]) -> Vec<GroupSlot<T>> {
        let cache = coefficient_groups();
        groups
            .iter()
            .map(|&(group_key, src_keys)| {
                let hash = self.group_hash(group_key, src_keys);
                let view = GroupKeyRef {
                    context: &self.context,
                    group_key,
                    src_keys,
                    hash,
                };
                match cache.get(&view) {
                    Some(entry) => {
                        #[cfg(test)]
                        GROUP_HITS.set(GROUP_HITS.get() + 1);
                        GroupSlot::Hit(entry.downcast::<CoefficientGroupEntry<T>>().unwrap_or_else(
                            |_| unreachable!("the key's TypeId fixes the entry type"),
                        ))
                    }
                    None => {
                        #[cfg(test)]
                        GROUP_MISSES.set(GROUP_MISSES.get() + 1);
                        GroupSlot::Miss(hash)
                    }
                }
            })
            .collect()
    }

    /// Stages one freshly built group; its tree list is cloned only here.
    pub(crate) fn stage(
        &self,
        hash: u64,
        (group_key, src_keys): SourceGroup<'_>,
        specs: Vec<TreeTransformGroupBlockSpec<T>>,
    ) -> Arc<CoefficientGroupEntry<T>> {
        let key = CoefficientGroupKey {
            context: Arc::clone(&self.context),
            group_key: group_key.clone(),
            src_keys: src_keys.iter().map(|key| (*key).clone()).collect(),
            hash,
        };
        let entry = Arc::new(CoefficientGroupEntry {
            specs: specs.into_boxed_slice(),
        });
        let bytes = charged_entry_bytes(&key, &entry.specs) as u64;
        self.built
            .borrow_mut()
            .groups
            .push((key, Arc::clone(&entry) as Arc<ErasedEntry>, bytes));
        entry
    }

    /// The groups staged by this build, for the owner to publish.
    pub(crate) fn into_pending(self) -> PendingCoefficientGroups {
        self.built.into_inner()
    }
}

/// What one entry retains, fixed at staging.
fn charged_entry_bytes<T>(
    key: &CoefficientGroupKey,
    specs: &[TreeTransformGroupBlockSpec<T>],
) -> usize {
    let mut backings = rustc_hash::FxHashSet::default();
    let mut bytes = core::mem::size_of::<CoefficientGroupKey>()
        .saturating_add(core::mem::size_of::<CoefficientGroupEntry<T>>())
        .saturating_add(ENTRY_OVERHEAD_BYTES)
        // ponytail: the shared context is charged in full to every entry.
        .saturating_add(ARC_CONTROL_BYTES)
        .saturating_add(core::mem::size_of::<GroupContext>())
        .saturating_add(key.context.rule.charged_retained_bytes())
        .saturating_add(key.context.operation.charged_retained_bytes())
        .saturating_add(key.group_key.charge_retained_backings(&mut backings))
        .saturating_add(ARC_CONTROL_BYTES)
        .saturating_add(
            key.src_keys
                .len()
                .saturating_mul(core::mem::size_of::<FusionTreePairKey>()),
        )
        .saturating_add(
            specs
                .len()
                .saturating_mul(core::mem::size_of::<TreeTransformGroupBlockSpec<T>>()),
        );
    for src in key.src_keys.iter() {
        bytes = bytes.saturating_add(src.charge_retained_backings(&mut backings));
    }
    bytes.saturating_add(charged_spec_bytes(specs, &mut backings))
}

/// Heap bytes of `specs` (excluding their inline structs), coefficients
/// included once.
fn charged_spec_bytes<T>(
    specs: &[TreeTransformGroupBlockSpec<T>],
    backings: &mut rustc_hash::FxHashSet<usize>,
) -> usize {
    let key_bytes = core::mem::size_of::<FusionTreePairKey>();
    let mut bytes = 0usize;
    // A `Multi` spec holds its keys and coefficients in three shared heap
    // slices (dst, src, coefficients); a `Single` spec holds them inline,
    // where the caller's `size_of` charge already counts them.
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
