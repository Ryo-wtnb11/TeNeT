//! Host-side planning for device tree-transform replay: the region triples a
//! Single block lowers to, the distinct cuTENSOR plan signatures a structure
//! submits, and the keyed cache that owns one uploaded coefficient vector per
//! structure.
//!
//! Pure metadata and bookkeeping: no device, no backend and no tenferro type
//! appears here, so this module compiles — and its tests run — in a CPU-only
//! build. It is kept out of `cuda_transform` precisely so the rules the device
//! executor depends on are executed by CI, which only *checks* the `cuda`
//! feature (the same split `tenet-dense`'s `cuda_region` makes).

use core::any::TypeId;
use std::collections::HashSet;
use std::sync::Weak;

use crate::task_view::TreeTransformTaskView;
use crate::{OperationError, TreeTransformBlock};

/// A strided, offset region of a flat device buffer, in element units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeviceRegionSpec {
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}

/// One Single block lowered to a device region move: the same `dims` on both
/// sides, the axis permutation carried by the destination strides, and the
/// index of the block's coefficient in the structure's coefficient payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeviceMoveSpec {
    pub(crate) dims: Vec<usize>,
    pub(crate) dst_strides: Vec<usize>,
    pub(crate) src_strides: Vec<usize>,
    pub(crate) dst_offset: usize,
    pub(crate) src_offset: usize,
    pub(crate) coefficient: usize,
}

/// The device lowering of one completed structure's Single blocks and, for the
/// Overwrite destination mode, of its inactive destination layouts.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DeviceTransformPlan {
    pub(crate) moves: Vec<DeviceMoveSpec>,
    pub(crate) zeros: Vec<DeviceRegionSpec>,
    /// Longest inactive layout, so the zero template is sized once.
    pub(crate) max_zero_len: usize,
    /// Distinct cuTENSOR operand signatures this plan submits.
    pub(crate) plan_signatures: usize,
}

fn unsupported(message: &'static str) -> OperationError {
    OperationError::UnsupportedDeviceTreeTransform { message }
}

/// Device regions are unsigned: Tenferro's CUDA dot-general rejects a negative
/// view stride, and a baked tree layout never produces one (block strides
/// originate as `usize`). A negative value is therefore a capability boundary,
/// not an expected input.
fn non_negative(values: &[isize], what: &'static str) -> Result<Vec<usize>, OperationError> {
    values
        .iter()
        .map(|value| usize::try_from(*value).map_err(|_| unsupported(what)))
        .collect()
}

fn offset_to_usize(offset: isize) -> Result<usize, OperationError> {
    usize::try_from(offset).map_err(|_| unsupported("device replay requires a non-negative offset"))
}

fn packed_strides(dims: &[usize]) -> Result<Vec<usize>, OperationError> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut running = 1usize;
    for &dim in dims {
        strides.push(running);
        running = running
            .checked_mul(dim)
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(strides)
}

/// Whether the structure contains a recoupling (pack → `Uᵀ` → scatter) block.
pub(crate) fn contains_multi_blocks<C: Copy>(task: TreeTransformTaskView<'_, C>) -> bool {
    task.blocks()
        .iter()
        .any(|block| matches!(block, TreeTransformBlock::Multi { .. }))
}

/// Lowers every Single block of `task` to a device region move, skipping the
/// zero-extent ones, plus every inactive destination layout to a region the
/// Overwrite destination mode fills with zeros.
///
/// Both modes share one plan: the Accumulate mode simply does not execute the
/// zero fills, exactly as host replay returns before touching inactive layouts
/// when `beta == 1`.
///
/// The move's `(dims, dst_strides, src_strides)` is the structure's own baked
/// fused layout, which is the canonical stride signature the host normalizer
/// produced at compile time: using it keeps the device's plan-key diversity at
/// the structural minimum instead of one plan per raw block rank.
pub(crate) fn compile_device_plan<C: Copy>(
    task: TreeTransformTaskView<'_, C>,
) -> Result<DeviceTransformPlan, OperationError> {
    let layouts = task.layouts();
    let mut moves = Vec::new();
    for block in task.blocks() {
        let TreeTransformBlock::Single {
            dst_layout,
            src_layout,
            coefficient,
        } = *block
        else {
            return Err(unsupported(
                "device tree transform replays Single blocks only",
            ));
        };
        let baked = layouts.fused_baked(dst_layout).ok_or_else(|| {
            unsupported("device tree transform requires a baked fused layout per Single block")
        })?;
        if baked.dims.contains(&0) {
            continue;
        }
        moves.push(DeviceMoveSpec {
            dims: baked.dims.to_vec(),
            dst_strides: non_negative(
                baked.dst_strides,
                "device replay requires non-negative destination strides",
            )?,
            src_strides: non_negative(
                baked.src_strides,
                "device replay requires non-negative source strides",
            )?,
            dst_offset: offset_to_usize(layouts.entry(dst_layout).offset)?,
            src_offset: offset_to_usize(layouts.entry(src_layout).offset)?,
            coefficient,
        });
    }

    let mut zeros = Vec::new();
    let mut max_zero_len = 0usize;
    for &layout_index in task.inactive_destination_layouts() {
        let layout = layouts.entry(layout_index);
        let dims = layouts.shape(layout).to_vec();
        if dims.contains(&0) {
            continue;
        }
        let count = dims
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .ok_or(OperationError::ElementCountOverflow)?;
        max_zero_len = max_zero_len.max(count);
        zeros.push(DeviceRegionSpec {
            dims,
            strides: non_negative(
                layouts.strides(layout),
                "device replay requires non-negative destination strides",
            )?,
            offset: offset_to_usize(layout.offset)?,
        });
    }

    let plan_signatures = distinct_plan_signatures(&moves, &zeros)?;
    Ok(DeviceTransformPlan {
        moves,
        zeros,
        max_zero_len,
        plan_signatures,
    })
}

/// Distinct cuTENSOR contraction signatures this plan submits.
///
/// One plan is built per `(dims, destination strides, source strides)` triple;
/// a zero fill reads a packed template of the fill's own extents, so its source
/// strides are the packed ones.
///
/// Exact per conjugation value: Tenferro's contraction plan key is (dtype,
/// the three operand layouts, their alignments, the operand operators,
/// workspace preference). Alignment is the constant `size_of::<D>()` for every
/// view, so offsets do not multiply keys, and the accumulation scalars are not
/// in the key at all; conjugation is, and it is one value per structure. The
/// per-executor *sum* of these counts therefore over-counts signatures two
/// structures share, which is the safe direction for a cap.
fn distinct_plan_signatures(
    moves: &[DeviceMoveSpec],
    zeros: &[DeviceRegionSpec],
) -> Result<usize, OperationError> {
    let mut seen: HashSet<(&[usize], &[usize], Vec<usize>)> = HashSet::new();
    for entry in moves {
        seen.insert((
            entry.dims.as_slice(),
            entry.dst_strides.as_slice(),
            entry.src_strides.clone(),
        ));
    }
    for zero in zeros {
        seen.insert((
            zero.dims.as_slice(),
            zero.strides.as_slice(),
            packed_strides(&zero.dims)?,
        ));
    }
    Ok(seen.len())
}

/// Entries of a cuTENSOR contraction plan cost about this much retained state
/// (`g2-design.md` §4). Used only to bound how far the executor raises the
/// backend's plan entry cap.
pub(crate) const CUTENSOR_PLAN_BYTES: usize = 14 * 1024;

/// How many plan entries `budget_bytes` pays for: the requirement, truncated
/// to what the budget affords. It can be below Tenferro's own default bound,
/// in which case the caller raises nothing and the backend keeps that default.
pub(crate) fn plan_cache_entries_for(required: usize, budget_bytes: usize) -> usize {
    required.min(budget_bytes / CUTENSOR_PLAN_BYTES)
}

/// Identity of device state prepared for one completed structure.
///
/// The `Weak` is the structure's own identity marker, so an entry whose
/// structure has been evicted from the transform cache is unreachable rather
/// than stale: the key can never match a different structure, and the dead
/// entry is purged on the next miss.
#[derive(Clone, Debug)]
pub(crate) struct StructureKey {
    pub(crate) structure: Weak<()>,
    pub(crate) scalar: TypeId,
    pub(crate) context: u64,
}

impl StructureKey {
    fn matches(&self, other: &Self) -> bool {
        self.scalar == other.scalar
            && self.context == other.context
            && Weak::ptr_eq(&self.structure, &other.structure)
    }
}

struct CacheEntry<V> {
    key: StructureKey,
    bytes: usize,
    value: V,
}

/// Small LRU cache of per-structure device state, bounded by retained bytes.
///
/// Ownership is singular: the cached value is the only copy of that device
/// state, and dropping the entry frees it. Bytes are reported so the owner of
/// the device memory budget can charge them.
pub(crate) struct StructureCache<V> {
    entries: Vec<CacheEntry<V>>,
    budget_bytes: usize,
    bytes: usize,
}

impl<V> StructureCache<V> {
    pub(crate) fn new(budget_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            budget_bytes,
            bytes: 0,
        }
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// The entry for `key`, promoted to most-recently-used.
    pub(crate) fn get(&mut self, key: &StructureKey) -> Option<&V> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.key.matches(key))?;
        let entry = self.entries.remove(index);
        self.entries.push(entry);
        self.entries.last().map(|entry| &entry.value)
    }

    /// Drops every entry whose structure no longer exists.
    pub(crate) fn purge_dead(&mut self) {
        self.entries.retain(|entry| {
            let live = entry.key.structure.strong_count() > 0;
            if !live {
                self.bytes = self.bytes.saturating_sub(entry.bytes);
            }
            live
        });
    }

    /// Inserts `value` as most-recently-used, evicting least-recently-used
    /// entries while the cache is over budget.
    ///
    /// The new entry is never evicted, even when it alone exceeds the budget:
    /// the budget bounds *reuse*, and refusing to keep the structure currently
    /// being replayed would only force an upload per replay.
    pub(crate) fn insert(&mut self, key: StructureKey, value: V, bytes: usize) {
        self.purge_dead();
        self.entries.push(CacheEntry { key, bytes, value });
        self.bytes = self.bytes.saturating_add(bytes);
        while self.bytes > self.budget_bytes && self.entries.len() > 1 {
            let evicted = self.entries.remove(0);
            self.bytes = self.bytes.saturating_sub(evicted.bytes);
        }
    }

    /// Every live value, for callers that aggregate over the whole cache.
    pub(crate) fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|entry| &entry.value)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tenet_core::{BlockKey, BlockSpec, BlockStructure};

    use super::*;
    use crate::{TreeTransformBlockSpec, TreeTransformStructure};

    fn structure(shapes: Vec<Vec<usize>>) -> Arc<BlockStructure> {
        Arc::new(BlockStructure::packed_column_major(shapes[0].len(), shapes).unwrap())
    }

    #[test]
    fn a_single_block_lowers_to_the_structures_baked_fused_layout() {
        // What: the device move carries the compiled fused triple, not the raw
        // block rank, and the permutation lives in the destination strides.
        let dst_space = structure(vec![vec![2, 3]]);
        let src_space = structure(vec![vec![3, 2]]);
        let compiled = TreeTransformStructure::compile_structures(
            &dst_space,
            &src_space,
            &[TreeTransformBlockSpec::single(0, 0, -1.0_f64).with_source_axes([1, 0])],
        )
        .unwrap();
        let plan = compile_device_plan(compiled.task_view().unwrap()).unwrap();

        assert_eq!(plan.moves.len(), 1);
        let entry = &plan.moves[0];
        assert_eq!(entry.dims.iter().product::<usize>(), 6);
        assert_eq!(entry.dims.len(), entry.dst_strides.len());
        assert_eq!(entry.dims.len(), entry.src_strides.len());
        assert_ne!(entry.dst_strides, entry.src_strides);
        assert_eq!(entry.coefficient, 0);
        assert!(plan.zeros.is_empty());
        assert_eq!(plan.plan_signatures, 1);
    }

    #[test]
    fn zero_extent_blocks_are_skipped_and_inactive_layouts_are_planned() {
        // What: an untouched destination layout becomes a zero fill sized for
        // the template, and an empty block submits nothing at all.
        let blocks = vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![1, 2], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![3, 2], vec![1, 3], 4).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(2), vec![0, 2], vec![1, 1], 10).unwrap(),
        ];
        let space = Arc::new(BlockStructure::from_blocks_with_rank(2, blocks).unwrap());
        let compiled = TreeTransformStructure::compile_structures(
            &space,
            &space,
            &[
                TreeTransformBlockSpec::single(0, 0, 1.0_f64),
                TreeTransformBlockSpec::single(2, 2, 1.0_f64),
            ],
        )
        .unwrap();

        let plan = compile_device_plan(compiled.task_view().unwrap()).unwrap();
        assert_eq!(plan.moves.len(), 1, "the zero-extent block is skipped");
        assert_eq!(plan.zeros.len(), 1, "block 1 is never written");
        assert_eq!(plan.zeros[0].dims, vec![3, 2]);
        assert_eq!(plan.zeros[0].offset, 4);
        assert_eq!(plan.max_zero_len, 6);
    }

    #[test]
    fn equal_layouts_share_one_plan_signature_and_distinct_ones_do_not() {
        // What: the signature count is what the plan-cache cap is raised to, so
        // it must collapse repeated layouts and separate genuinely distinct
        // ones — including a zero fill, which reads a packed template.
        let same = DeviceMoveSpec {
            dims: vec![4],
            dst_strides: vec![1],
            src_strides: vec![1],
            dst_offset: 0,
            src_offset: 8,
            coefficient: 1,
        };
        let mut twin = same.clone();
        twin.dst_offset = 16;
        let other = DeviceMoveSpec {
            dims: vec![2, 2],
            dst_strides: vec![2, 1],
            src_strides: vec![1, 2],
            dst_offset: 0,
            src_offset: 0,
            coefficient: 0,
        };
        let zero = DeviceRegionSpec {
            dims: vec![4],
            strides: vec![1],
            offset: 32,
        };

        assert_eq!(
            distinct_plan_signatures(&[same.clone(), twin], &[]).unwrap(),
            1
        );
        assert_eq!(
            distinct_plan_signatures(&[same.clone(), other], &[]).unwrap(),
            2
        );
        // The fill's source is packed [1], identical to `same`'s source.
        assert_eq!(distinct_plan_signatures(&[same], &[zero]).unwrap(), 1);
    }

    #[test]
    fn the_plan_cap_never_exceeds_the_byte_budget() {
        assert_eq!(plan_cache_entries_for(200, 300 * CUTENSOR_PLAN_BYTES), 200);
        assert_eq!(plan_cache_entries_for(200, 10 * CUTENSOR_PLAN_BYTES), 10);
        assert_eq!(plan_cache_entries_for(200, 0), 0);
    }

    #[test]
    fn the_cache_keys_on_structure_identity_dtype_and_context() {
        // What: nothing but the exact triple hits, so a second structure, a
        // second dtype or a second context prepares its own device state.
        let live = Arc::new(());
        let other = Arc::new(());
        let key = |structure: &Arc<()>, scalar, context| StructureKey {
            structure: Arc::downgrade(structure),
            scalar,
            context,
        };
        let mut cache = StructureCache::new(1024);
        cache.insert(key(&live, TypeId::of::<f64>(), 1), "f64@1", 8);

        assert_eq!(
            cache.get(&key(&live, TypeId::of::<f64>(), 1)),
            Some(&"f64@1")
        );
        assert_eq!(cache.get(&key(&live, TypeId::of::<u32>(), 1)), None);
        assert_eq!(cache.get(&key(&live, TypeId::of::<f64>(), 2)), None);
        assert_eq!(cache.get(&key(&other, TypeId::of::<f64>(), 1)), None);
        assert_eq!(cache.retained_bytes(), 8);
    }

    #[test]
    fn a_dropped_structure_is_purged_and_its_bytes_released() {
        let live = Arc::new(());
        let doomed = Arc::new(());
        let mut cache = StructureCache::new(1024);
        cache.insert(
            StructureKey {
                structure: Arc::downgrade(&doomed),
                scalar: TypeId::of::<f64>(),
                context: 1,
            },
            "doomed",
            64,
        );
        cache.insert(
            StructureKey {
                structure: Arc::downgrade(&live),
                scalar: TypeId::of::<f64>(),
                context: 1,
            },
            "live",
            16,
        );
        assert_eq!(cache.retained_bytes(), 80);

        drop(doomed);
        cache.purge_dead();

        assert_eq!(cache.entry_count(), 1);
        assert_eq!(cache.retained_bytes(), 16);
        assert_eq!(cache.values().copied().collect::<Vec<_>>(), vec!["live"]);
    }

    #[test]
    fn the_budget_evicts_least_recently_used_and_always_keeps_the_newest() {
        let structures: Vec<Arc<()>> = (0..3).map(|_| Arc::new(())).collect();
        let key = |index: usize| StructureKey {
            structure: Arc::downgrade(&structures[index]),
            scalar: TypeId::of::<f64>(),
            context: 7,
        };
        let mut cache = StructureCache::new(100);
        cache.insert(key(0), 0, 60);
        cache.insert(key(1), 1, 30);
        // Touching 0 makes 1 the least recently used.
        assert_eq!(cache.get(&key(0)), Some(&0));
        cache.insert(key(2), 2, 30);

        assert_eq!(cache.entry_count(), 2);
        assert_eq!(cache.get(&key(1)), None);
        assert_eq!(cache.get(&key(0)), Some(&0));
        assert_eq!(cache.get(&key(2)), Some(&2));

        // An entry larger than the whole budget still replaces everything and
        // stays, so the structure being replayed is never re-uploaded.
        cache.insert(key(1), 1, 4096);
        assert_eq!(cache.entry_count(), 1);
        assert_eq!(cache.get(&key(1)), Some(&1));
        assert_eq!(cache.retained_bytes(), 4096);
    }
}
