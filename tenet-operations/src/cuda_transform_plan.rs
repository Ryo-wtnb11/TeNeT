//! Host-side planning for device tree-transform replay: the region triples a
//! Single block lowers to, the pack → GEMM → scatter lowering of a Multi
//! (recoupling) block, the distinct cuTENSOR plan signatures a structure
//! submits, and the keyed cache that owns one uploaded coefficient vector per
//! structure.
//!
//! Pure metadata and bookkeeping: no device, no backend and no tenferro type
//! appears here, so this module compiles — and its tests run — in a CPU-only
//! build. It is kept out of `cuda_transform` precisely so the rules the device
//! executor depends on are executed by CI, which only *checks* the `cuda`
//! feature (the same split `tenet-dense`'s `cuda_region` makes).

use core::any::TypeId;
use core::ops::Range;
use std::collections::HashSet;
use std::sync::Weak;

use tenet_dense::DenseGemmBatchJob;

use crate::task_view::TreeTransformTaskView;
use crate::{OperationError, TreeTransformBlock};

/// A strided, offset region of a flat device buffer, in element units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeviceRegionSpec {
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}

/// One block, pack column or scatter column lowered to a device region move:
/// the same `dims` on both sides, the axis permutation carried by the
/// destination strides, and the coefficient the move is scaled by.
///
/// `coefficient` is the index of the block's scalar in the structure's
/// coefficient payload for a Single block, and `None` for a pack or scatter
/// column, which the host copies unscaled and the device therefore submits
/// against the context's shared `1`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeviceMoveSpec {
    pub(crate) dims: Vec<usize>,
    pub(crate) dst_strides: Vec<usize>,
    pub(crate) src_strides: Vec<usize>,
    pub(crate) dst_offset: usize,
    pub(crate) src_offset: usize,
    pub(crate) coefficient: Option<usize>,
}

/// One Multi block lowered to the host's pack → `Uᵀ` GEMM → scatter sequence.
///
/// `packs` read the source storage into the workspace source column of `job`,
/// `job` multiplies that column block by the transposed recoupling matrix into
/// the workspace destination, and `scatters` write the result columns out.
/// Every offset in `job` addresses the structure-wide workspace the compile-time
/// recoupling plan sized, so the device reuses the plan's own arithmetic
/// instead of re-deriving per-block offsets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeviceRecouplingSpec {
    pub(crate) packs: Vec<DeviceMoveSpec>,
    pub(crate) scatters: Vec<DeviceMoveSpec>,
    pub(crate) job: DenseGemmBatchJob,
}

/// The device lowering of one completed structure: its Single blocks, its Multi
/// blocks, and — for the Overwrite destination mode — its inactive destination
/// layouts.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DeviceTransformPlan {
    pub(crate) moves: Vec<DeviceMoveSpec>,
    pub(crate) recouplings: Vec<DeviceRecouplingSpec>,
    pub(crate) zeros: Vec<DeviceRegionSpec>,
    /// Ranges of the structure's coefficient payload holding each Multi block's
    /// recoupling matrix, in the recoupling plan's own entry order — the order
    /// the jobs' `rhs_offset`s address.
    pub(crate) coefficient_ranges: Vec<Range<usize>>,
    /// Workspace elements the packs write and the GEMMs read.
    pub(crate) workspace_source_len: usize,
    /// Workspace elements the GEMMs write and the scatters read.
    pub(crate) workspace_destination_len: usize,
    /// Longest inactive layout, so the zero template is sized once.
    pub(crate) max_zero_len: usize,
    /// Distinct cuTENSOR operand signatures this plan submits.
    pub(crate) plan_signatures: usize,
}

impl DeviceTransformPlan {
    /// Recoupling matrix elements the device holds beside the structure's
    /// coefficient vector, in plan-entry order.
    pub(crate) fn coefficient_len(&self) -> usize {
        self.coefficient_ranges.iter().map(Range::len).sum()
    }
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

/// Lowers every Single block of `task` to a device region move and every Multi
/// block to the host's pack → `Uᵀ` → scatter sequence, skipping the
/// zero-extent ones, plus every inactive destination layout to a region the
/// Overwrite destination mode fills with zeros.
///
/// Both modes share one plan: the Accumulate mode simply does not execute the
/// zero fills, exactly as host replay returns before touching inactive layouts
/// when `beta == 1`.
///
/// Every move's `(dims, dst_strides, src_strides)` is the structure's own baked
/// fused layout, which is the canonical stride signature the host normalizer
/// produced at compile time: using it keeps the device's plan-key diversity at
/// the structural minimum instead of one plan per raw block rank. The host
/// bakes a Multi source entry with the packed column as its destination and a
/// Multi destination entry with the packed column as its source
/// (`bake_fused_layouts`), so pack and scatter read the same arena the host
/// kernels do rather than a device-local re-derivation.
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
            continue;
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
            coefficient: Some(coefficient),
        });
    }

    let recoupling_plan = task.recoupling_plan();
    let mut recouplings = Vec::with_capacity(recoupling_plan.jobs().len());
    let mut coefficient_ranges = Vec::with_capacity(recoupling_plan.jobs().len());
    for (block_index, job) in recoupling_plan.entries() {
        let block = task
            .blocks()
            .get(block_index)
            .ok_or_else(|| unsupported("device tree transform recoupling plan is out of range"))?;
        let TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            src_layout_start,
            src_count,
            coefficient_start,
            element_count,
        } = *block
        else {
            return Err(unsupported(
                "device tree transform recoupling plan names a Single block",
            ));
        };
        // The device reuses the plan's offsets, so a plan that disagrees with
        // the block it names would silently read another block's column.
        if job.rows != element_count || job.contracted != src_count || job.cols != dst_count {
            return Err(unsupported(
                "device tree transform recoupling job disagrees with its block",
            ));
        }
        let coefficient_end = src_count
            .checked_mul(dst_count)
            .and_then(|len| coefficient_start.checked_add(len))
            .ok_or(OperationError::ElementCountOverflow)?;
        if coefficient_end > task.coefficients().len() {
            return Err(OperationError::CoefficientCountMismatch {
                expected: coefficient_end,
                actual: task.coefficients().len(),
            });
        }
        coefficient_ranges.push(coefficient_start..coefficient_end);
        let mut packs = Vec::with_capacity(src_count);
        for src_index in 0..src_count {
            let entry = src_layout_start + src_index;
            let column = element_count
                .checked_mul(src_index)
                .and_then(|start| job.lhs_offset.checked_add(start))
                .ok_or(OperationError::ElementCountOverflow)?;
            if let Some(pack) = column_move(task, entry, column, PackDirection::IntoColumn)? {
                packs.push(pack);
            }
        }
        let mut scatters = Vec::with_capacity(dst_count);
        for dst_index in 0..dst_count {
            let entry = dst_layout_start + dst_index;
            let column = element_count
                .checked_mul(dst_index)
                .and_then(|start| job.dst_offset.checked_add(start))
                .ok_or(OperationError::ElementCountOverflow)?;
            if let Some(scatter) = column_move(task, entry, column, PackDirection::OutOfColumn)? {
                scatters.push(scatter);
            }
        }
        recouplings.push(DeviceRecouplingSpec {
            packs,
            scatters,
            job: *job,
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

    let plan_signatures = distinct_plan_signatures(&moves, &recouplings, &zeros)?;
    Ok(DeviceTransformPlan {
        moves,
        recouplings,
        zeros,
        coefficient_ranges,
        workspace_source_len: recoupling_plan.source_len(),
        workspace_destination_len: recoupling_plan.destination_len(),
        max_zero_len,
        plan_signatures,
    })
}

/// Which side of a pack/scatter move the compact workspace column is on.
#[derive(Clone, Copy, Eq, PartialEq)]
enum PackDirection {
    /// Pack: block storage → workspace column.
    IntoColumn,
    /// Scatter: workspace column → block storage.
    OutOfColumn,
}

/// Lowers one Multi layout entry to the region move between its block storage
/// and its compact workspace column at `column`.
///
/// `None` for a zero-extent entry, which addresses nothing and is never
/// submitted — the host's `shape.contains(&0)` rule.
fn column_move<C: Copy>(
    task: TreeTransformTaskView<'_, C>,
    entry: usize,
    column: usize,
    direction: PackDirection,
) -> Result<Option<DeviceMoveSpec>, OperationError> {
    let layouts = task.layouts();
    let baked = layouts.fused_baked(entry).ok_or_else(|| {
        unsupported("device tree transform requires a baked fused layout per recoupling column")
    })?;
    if baked.dims.contains(&0) {
        return Ok(None);
    }
    let dims = baked.dims.to_vec();
    let dst_strides = non_negative(
        baked.dst_strides,
        "device replay requires non-negative destination strides",
    )?;
    let src_strides = non_negative(
        baked.src_strides,
        "device replay requires non-negative source strides",
    )?;
    let block_offset = offset_to_usize(layouts.entry(entry).offset)?;
    let (dst_offset, src_offset) = match direction {
        PackDirection::IntoColumn => (column, block_offset),
        PackDirection::OutOfColumn => (block_offset, column),
    };
    Ok(Some(DeviceMoveSpec {
        dims,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        // Host packs and scatters copy unscaled (`T::one()`); the block's own
        // coefficients are the recoupling matrix the GEMM applies.
        coefficient: None,
    }))
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
    recouplings: &[DeviceRecouplingSpec],
    zeros: &[DeviceRegionSpec],
) -> Result<usize, OperationError> {
    let mut seen: HashSet<(&[usize], &[usize], Vec<usize>)> = HashSet::new();
    let columns = recouplings
        .iter()
        .flat_map(|entry| entry.packs.iter().chain(&entry.scatters));
    for entry in moves.iter().chain(columns) {
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
    // A recoupling GEMM is a contraction of three dense column-major operands,
    // so its plan key is its `(rows, contracted, cols)` shape; jobs of equal
    // shape share one plan.
    let gemms: HashSet<(usize, usize, usize)> = recouplings
        .iter()
        .map(|entry| (entry.job.rows, entry.job.contracted, entry.job.cols))
        .collect();
    Ok(seen.len().saturating_add(gemms.len()))
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

/// Entries a [`StructureCache`] keeps, whatever the byte budget allows.
///
/// A byte budget alone bounds the device memory but not the entry count, so a
/// replay alternating thousands of tiny structures would grow the cache without
/// limit. The cap is also what keeps the lookup O(1): the scan is over at most
/// this many keys. A hash key would have to be the `Weak`'s address, which a
/// later allocation can reuse, so the identity comparison stays a `ptr_eq`.
pub(crate) const MAX_STRUCTURE_CACHE_ENTRIES: usize = 32;

/// Small LRU cache of per-structure device state, bounded by retained bytes and
/// by [`MAX_STRUCTURE_CACHE_ENTRIES`].
///
/// Ownership is singular: the cached value is the only copy of that device
/// state, and dropping the entry frees it. Bytes are reported so the owner of
/// the device memory budget can charge them.
pub(crate) struct StructureCache<V> {
    entries: Vec<CacheEntry<V>>,
    budget_bytes: usize,
    max_entries: usize,
    bytes: usize,
}

impl<V> StructureCache<V> {
    pub(crate) fn new(budget_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            budget_bytes,
            max_entries: MAX_STRUCTURE_CACHE_ENTRIES,
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

    /// The entry for `key` without changing the recency order, for callers that
    /// only describe the state they already prepared.
    pub(crate) fn peek(&self, key: &StructureKey) -> Option<&V> {
        self.entries
            .iter()
            .find(|entry| entry.key.matches(key))
            .map(|entry| &entry.value)
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
        while (self.bytes > self.budget_bytes || self.entries.len() > self.max_entries)
            && self.entries.len() > 1
        {
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
        assert_eq!(entry.coefficient, Some(0));
        assert!(plan.zeros.is_empty());
        assert_eq!(plan.plan_signatures, 1);
    }

    #[test]
    fn a_multi_block_lowers_to_pack_gemm_and_scatter_over_the_plans_offsets() {
        // What: a recoupling block becomes one pack per source layout, one GEMM
        // whose shape is (element_count, src_count, dst_count), and one scatter
        // per destination layout — with every workspace offset taken from the
        // compile-time recoupling plan, and with no per-column coefficient.
        let dst_space = structure(vec![vec![2, 3], vec![2, 3]]);
        let src_space = structure(vec![vec![3, 2], vec![3, 2], vec![3, 2]]);
        let compiled = TreeTransformStructure::compile_structures(
            &dst_space,
            &src_space,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1, 2],
                vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .with_source_axes([1, 0])],
        )
        .unwrap();
        let plan = compile_device_plan(compiled.task_view().unwrap()).unwrap();

        assert!(plan.moves.is_empty(), "no Single block");
        assert_eq!(plan.recouplings.len(), 1);
        let entry = &plan.recouplings[0];
        assert_eq!(entry.packs.len(), 3);
        assert_eq!(entry.scatters.len(), 2);
        assert_eq!(entry.job.rows, 6, "element count");
        assert_eq!(entry.job.contracted, 3, "source count");
        assert_eq!(entry.job.cols, 2, "destination count");
        for column in entry.packs.iter().chain(&entry.scatters) {
            assert_eq!(column.coefficient, None, "columns copy unscaled");
            assert_eq!(column.dims.iter().product::<usize>(), 6);
        }
        // Pack i writes column i of the job's source; scatter j reads column j
        // of its destination.
        let packed: Vec<usize> = entry.packs.iter().map(|pack| pack.dst_offset).collect();
        assert_eq!(packed, vec![0, 6, 12]);
        let scattered: Vec<usize> = entry
            .scatters
            .iter()
            .map(|scatter| scatter.src_offset)
            .collect();
        assert_eq!(scattered, vec![0, 6]);
        assert_eq!(plan.workspace_source_len, 18);
        assert_eq!(plan.workspace_destination_len, 12);
        assert_eq!(plan.coefficient_ranges, vec![0..6]);
        assert_eq!(plan.coefficient_len(), 6);
    }

    #[test]
    fn several_multi_blocks_keep_the_recoupling_plans_order_and_offsets() {
        // What: the plan sorts jobs by element count, so its order is not the
        // block order; the device must address each block's columns through the
        // plan's own offsets, and the coefficient ranges must follow the same
        // order so a job's `rhs_offset` names its own matrix.
        let space = structure(vec![
            vec![2, 3],
            vec![2, 3],
            vec![2, 2],
            vec![2, 2],
            vec![2, 2],
        ]);
        let compiled = TreeTransformStructure::compile_structures(
            &space,
            &space,
            &[
                // Declared first, but 6 elements wide, so the plan runs it second.
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0_f64, 2.0, 3.0, 4.0]),
                TreeTransformBlockSpec::multi(vec![2, 3], vec![2, 3], vec![5.0_f64, 6.0, 7.0, 8.0]),
                TreeTransformBlockSpec::single(4, 4, -1.0_f64),
            ],
        )
        .unwrap();
        let plan = compile_device_plan(compiled.task_view().unwrap()).unwrap();

        assert_eq!(plan.moves.len(), 1, "the Single block is still lowered");
        let shapes: Vec<usize> = plan
            .recouplings
            .iter()
            .map(|entry| entry.job.rows)
            .collect();
        assert_eq!(shapes, vec![4, 6], "sorted by element count");
        // Disjoint, contiguous workspace columns in that same order.
        assert_eq!(plan.recouplings[0].job.lhs_offset, 0);
        assert_eq!(plan.recouplings[1].job.lhs_offset, 8);
        assert_eq!(plan.workspace_source_len, 20);
        assert_eq!(plan.workspace_destination_len, 20);
        // The 4-element block was declared second, so its matrix is the second
        // range of the payload but the first of the plan.
        assert_eq!(plan.coefficient_ranges, vec![4..8, 0..4]);
        assert_eq!(plan.recouplings[0].job.rhs_offset, 0);
        assert_eq!(plan.recouplings[1].job.rhs_offset, 4);
    }

    #[test]
    fn the_cache_evicts_the_least_recently_used_beyond_its_entry_cap() {
        // What: the byte budget alone would let an unbounded number of tiny
        // structures accumulate, so the entry cap bounds the cache — and with
        // it the lookup scan — independently of size.
        let structures: Vec<Arc<()>> = (0..MAX_STRUCTURE_CACHE_ENTRIES + 1)
            .map(|_| Arc::new(()))
            .collect();
        let key = |index: usize| StructureKey {
            structure: Arc::downgrade(&structures[index]),
            scalar: TypeId::of::<f64>(),
            context: 3,
        };
        let mut cache = StructureCache::new(usize::MAX);
        for index in 0..structures.len() {
            cache.insert(key(index), index, 1);
        }

        assert_eq!(cache.entry_count(), MAX_STRUCTURE_CACHE_ENTRIES);
        assert_eq!(cache.get(&key(0)), None, "the oldest entry was evicted");
        assert_eq!(cache.get(&key(1)), Some(&1));
        assert_eq!(cache.retained_bytes(), MAX_STRUCTURE_CACHE_ENTRIES);
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
            coefficient: Some(1),
        };
        let mut twin = same.clone();
        twin.dst_offset = 16;
        let other = DeviceMoveSpec {
            dims: vec![2, 2],
            dst_strides: vec![2, 1],
            src_strides: vec![1, 2],
            dst_offset: 0,
            src_offset: 0,
            coefficient: Some(0),
        };
        let zero = DeviceRegionSpec {
            dims: vec![4],
            strides: vec![1],
            offset: 32,
        };

        assert_eq!(
            distinct_plan_signatures(&[same.clone(), twin], &[], &[]).unwrap(),
            1
        );
        assert_eq!(
            distinct_plan_signatures(&[same.clone(), other], &[], &[]).unwrap(),
            2
        );
        // The fill's source is packed [1], identical to `same`'s source.
        assert_eq!(distinct_plan_signatures(&[same], &[], &[zero]).unwrap(), 1);
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
