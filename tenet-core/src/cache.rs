//! TeNeT's structure caches (#2014): process-global, byte-bounded, one
//! wrapper over `quick_cache` for every one of them.
//!
//! They mirror TensorKit's `@cached` functions (cfaa073,
//! `src/auxiliary/caches.jl`): each memoizes a pure function of its key, so a
//! hit is interchangeable with a rebuild, and only the application clears or
//! sizes them (as racah does for its symbol caches).
//!
//! Why `quick_cache` and not one `RwLock<LruCache>` per cache: a warm hit is
//! one shard read with a borrowed key, misses of the same key are
//! single-flight, and eviction weighs entries by their charged bytes.

use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use quick_cache::sync::{Cache, GuardResult};
use quick_cache::{Equivalent, Lifecycle, OptionsBuilder, Weighter};

use crate::block_structure::may_publish_since;

/// The structure caches.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum StructureCacheKind {
    /// Fusion-tree keys and sector structure per HomSpace sector signature,
    /// degeneracies excluded (TensorKit `sectorstructure`).
    SectorStructure,
    /// Complete block structure per HomSpace, degeneracies included
    /// (TensorKit `degeneracystructure`).
    DegeneracyStructure,
}

impl StructureCacheKind {
    /// Every structure cache, in a fixed order.
    pub const ALL: [Self; 2] = [Self::SectorStructure, Self::DegeneracyStructure];
}

/// A snapshot of one structure cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StructureCacheInfo {
    kind: StructureCacheKind,
    entries: usize,
    charged_bytes: u64,
    byte_budget: u64,
    max_entry_bytes: u64,
    hits: u64,
    misses: u64,
    admissions: u64,
    evictions: u64,
    rejections: u64,
}

impl StructureCacheInfo {
    pub fn kind(self) -> StructureCacheKind {
        self.kind
    }
    pub fn entries(self) -> usize {
        self.entries
    }
    /// Bytes charged for the resident entries: a conservative accounting of
    /// what each entry retains, not allocator-observed memory.
    pub fn charged_bytes(self) -> u64 {
        self.charged_bytes
    }
    pub fn byte_budget(self) -> u64 {
        self.byte_budget
    }
    /// The largest entry the cache admits: one shard's share of the budget.
    pub fn max_entry_bytes(self) -> u64 {
        self.max_entry_bytes
    }
    pub fn hits(self) -> u64 {
        self.hits
    }
    /// Completed builds offered for admission, rejected ones included.
    pub fn misses(self) -> u64 {
        self.misses
    }
    pub fn admissions(self) -> u64 {
        self.admissions
    }
    pub fn evictions(self) -> u64 {
        self.evictions
    }
    /// Builds larger than [`Self::max_entry_bytes`], returned uncached.
    pub fn rejections(self) -> u64 {
        self.rejections
    }
}

struct Charged<V> {
    value: Arc<V>,
    bytes: u64,
}

// Why manual: a derive would bound `V: Clone`.
impl<V> Clone for Charged<V> {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
            bytes: self.bytes,
        }
    }
}

#[derive(Clone)]
struct ChargeWeighter;

impl<K, V> Weighter<K, Charged<V>> for ChargeWeighter {
    fn weight(&self, _: &K, charged: &Charged<V>) -> u64 {
        charged.bytes
    }
}

#[derive(Clone)]
struct CountEvictions(Arc<AtomicU64>);

impl<K, V> Lifecycle<K, V> for CountEvictions {
    /// Evicted items, dropped when the request ends, outside the shard lock
    /// (as quick_cache's default lifecycle does).
    type RequestState = Vec<(K, V)>;

    fn on_evict(&self, evicted: &mut Self::RequestState, key: K, value: V) {
        self.0.fetch_add(1, Ordering::Relaxed);
        evicted.push((key, value));
    }
}

/// One structure cache: values are `Arc<V>`, weighed by the bytes the caller
/// charges for them.
pub(crate) struct StructureCache<K, V> {
    kind: StructureCacheKind,
    entries: Cache<K, Charged<V>, ChargeWeighter, rustc_hash::FxBuildHasher, CountEvictions>,
    /// Read-held while an admission checks the reset epoch and inserts;
    /// write-held by [`Self::clear`]. A build that straddles a reset is
    /// therefore either wiped by the clear or refused by the epoch check.
    publication: RwLock<()>,
    hits: AtomicU64,
    misses: AtomicU64,
    admissions: AtomicU64,
    evictions: Arc<AtomicU64>,
    rejections: AtomicU64,
}

impl<K, V> StructureCache<K, V>
where
    K: Eq + Hash + Clone,
{
    /// `shards` (a power of two) bounds the largest entry to
    /// `byte_budget / shards`. Why few: entries are heavy-tailed, and the
    /// largest must still fit one shard.
    pub(crate) fn new(kind: StructureCacheKind, byte_budget: u64, shards: usize) -> Self {
        let evictions = Arc::new(AtomicU64::new(0));
        let options = OptionsBuilder::new()
            .shards(shards)
            .weight_capacity(byte_budget)
            .estimated_items_capacity(1024)
            .build()
            .expect("structure cache options are valid");
        Self {
            kind,
            entries: Cache::with_options(
                options,
                ChargeWeighter,
                rustc_hash::FxBuildHasher,
                CountEvictions(Arc::clone(&evictions)),
            ),
            publication: RwLock::new(()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            admissions: AtomicU64::new(0),
            evictions,
            rejections: AtomicU64::new(0),
        }
    }

    /// A warm hit: one shard read, no allocation.
    pub(crate) fn get<Q>(&self, key: &Q) -> Option<Arc<V>>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let found = self.entries.get(key)?;
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(found.value)
    }

    /// Whether `key` is resident; counts no hit.
    pub(crate) fn contains<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.entries.contains_key(key)
    }

    /// Publishes a value built since reset epoch `epoch`, or returns the
    /// entry another caller published first. Returned uncached when it is
    /// larger than one shard or its build straddled a reset.
    pub(crate) fn publish(&self, key: &K, value: Arc<V>, bytes: u64, epoch: usize) -> Arc<V> {
        match self.entries.get_value_or_guard(key, None) {
            GuardResult::Value(existing) => existing.value,
            GuardResult::Guard(guard) => self.admit(value, bytes, epoch, |charged| {
                let _ = guard.insert(charged);
            }),
            GuardResult::Timeout => unreachable!("a lookup without a timeout never times out"),
        }
    }

    /// The entry for `key`, built by `build` on a miss. Concurrent misses of
    /// one key build once; a failed build is not cached.
    pub(crate) fn get_or_try_build<E>(
        &self,
        key: &K,
        epoch: usize,
        build: impl FnOnce() -> Result<(Arc<V>, u64), E>,
    ) -> Result<Arc<V>, E> {
        match self.entries.get_value_or_guard(key, None) {
            GuardResult::Value(existing) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Ok(existing.value)
            }
            GuardResult::Guard(guard) => {
                let (value, bytes) = build()?;
                Ok(self.admit(value, bytes, epoch, |charged| {
                    let _ = guard.insert(charged);
                }))
            }
            GuardResult::Timeout => unreachable!("a lookup without a timeout never times out"),
        }
    }

    fn admit(
        &self,
        value: Arc<V>,
        bytes: u64,
        epoch: usize,
        insert: impl FnOnce(Charged<V>),
    ) -> Arc<V> {
        self.misses.fetch_add(1, Ordering::Relaxed);
        // quick_cache drops an item heavier than its shard without a trace;
        // the rejection is counted here instead.
        if bytes > self.entries.shard_capacity() {
            self.rejections.fetch_add(1, Ordering::Relaxed);
            return value;
        }
        let _publication = self
            .publication
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !may_publish_since(epoch) {
            return value;
        }
        insert(Charged {
            value: Arc::clone(&value),
            bytes,
        });
        self.admissions.fetch_add(1, Ordering::Relaxed);
        value
    }

    pub(crate) fn clear(&self) {
        let _publication = self
            .publication
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.entries.clear();
        for counter in [&self.hits, &self.misses, &self.admissions, &self.rejections] {
            counter.store(0, Ordering::Relaxed);
        }
        self.evictions.store(0, Ordering::Relaxed);
    }

    pub(crate) fn set_byte_budget(&self, bytes: u64) {
        self.entries.set_capacity(bytes);
    }

    pub(crate) fn info(&self) -> StructureCacheInfo {
        StructureCacheInfo {
            kind: self.kind,
            entries: self.entries.len(),
            charged_bytes: self.entries.weight(),
            byte_budget: self.entries.capacity(),
            max_entry_bytes: self.entries.shard_capacity(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            admissions: self.admissions.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            rejections: self.rejections.load(Ordering::Relaxed),
        }
    }
}

/// A snapshot of one structure cache.
pub fn structure_cache_info(kind: StructureCacheKind) -> StructureCacheInfo {
    match kind {
        StructureCacheKind::SectorStructure => crate::fusion_space::sector_structure_cache().info(),
        StructureCacheKind::DegeneracyStructure => {
            crate::fusion_space::degeneracy_structure_cache().info()
        }
    }
}

/// Snapshots of every structure cache.
pub fn structure_cache_infos() -> Vec<StructureCacheInfo> {
    StructureCacheKind::ALL
        .into_iter()
        .map(structure_cache_info)
        .collect()
}

/// Sets one cache's byte budget; entries beyond it are evicted. Owned by the
/// application: library code never calls it.
pub fn set_structure_cache_byte_budget(kind: StructureCacheKind, bytes: u64) {
    match kind {
        StructureCacheKind::SectorStructure => {
            crate::fusion_space::sector_structure_cache().set_byte_budget(bytes)
        }
        StructureCacheKind::DegeneracyStructure => {
            crate::fusion_space::degeneracy_structure_cache().set_byte_budget(bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_structure::core_reset_epoch;

    fn cache(byte_budget: u64) -> StructureCache<u32, u32> {
        StructureCache::new(StructureCacheKind::SectorStructure, byte_budget, 2)
    }

    #[test]
    fn a_miss_builds_once_and_a_failed_build_is_not_cached() {
        let cache = cache(1 << 20);
        let epoch = core_reset_epoch();
        let failed = cache.get_or_try_build(&1, epoch, || Err::<(Arc<u32>, u64), _>("no"));
        assert_eq!(failed, Err("no"));
        assert!(cache.get(&1).is_none());
        let built = cache
            .get_or_try_build(&1, epoch, || Ok::<_, ()>((Arc::new(7), 64)))
            .unwrap();
        let again = cache
            .get_or_try_build(&1, epoch, || -> Result<_, ()> { panic!("rebuilt") })
            .unwrap();
        // What: the second call hits the published Arc; the error left no trace.
        assert!(Arc::ptr_eq(&built, &again));
        let info = cache.info();
        assert_eq!((info.hits(), info.misses(), info.admissions()), (1, 1, 1));
    }

    #[test]
    fn publish_keeps_the_first_value_of_a_key() {
        let cache = cache(1 << 20);
        let epoch = core_reset_epoch();
        let first = cache.publish(&3, Arc::new(1), 64, epoch);
        let second = cache.publish(&3, Arc::new(2), 64, epoch);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(*cache.get(&3).unwrap(), 1);
    }

    #[test]
    fn an_entry_over_one_shard_is_rejected_and_counted() {
        let cache = cache(1 << 20);
        let shard = cache.info().max_entry_bytes();
        assert_eq!(shard, (1 << 20) / 2);
        let value = cache.publish(&5, Arc::new(9), shard + 1, core_reset_epoch());
        // What: the caller still gets its value; the cache keeps nothing and
        // says so (quick_cache alone would drop it silently).
        assert_eq!(*value, 9);
        assert!(cache.get(&5).is_none());
        let info = cache.info();
        assert_eq!((info.rejections(), info.admissions()), (1, 0));
    }

    #[test]
    fn a_build_from_a_stale_epoch_is_not_published() {
        let cache = cache(1 << 20);
        let stale = core_reset_epoch().wrapping_add(2);
        let value = cache.publish(&6, Arc::new(4), 64, stale);
        assert_eq!(*value, 4);
        assert!(cache.get(&6).is_none());
    }

    #[test]
    fn the_byte_budget_bounds_the_resident_entries() {
        let cache = cache(64 * 1024);
        let epoch = core_reset_epoch();
        for key in 0..1024 {
            cache.publish(&key, Arc::new(key), 1024, epoch);
        }
        let info = cache.info();
        // What: weight, not entry count, bounds retention, and the excess
        // was evicted.
        assert!(info.charged_bytes() <= info.byte_budget());
        assert!(info.evictions() > 0);
        cache.set_byte_budget(0);
        assert_eq!(cache.info().entries(), 0);
        cache.clear();
        assert_eq!(cache.info().evictions(), 0);
    }
}
