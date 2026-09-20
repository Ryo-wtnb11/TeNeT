//! Device replay of Single-block tree transforms on [`CudaStorage`].
//!
//! This is the second executor of the *same* `TreeTransformTaskView` the host
//! path consumes: everything above the completed `TreeTransformStructure` —
//! the operation descriptor, rule validation, the categorical group plan with
//! its F/R moves, the cache keyed by `RuleIdentity` — is reused unchanged, and
//! nothing categorical happens during replay. What is left below the structure
//! is exactly one scalar coefficient per block, a baked
//! `(dims, dst_strides, src_strides)` triple per block, a storage-conjugation
//! flag, and the destination mode.
//!
//! Reference record for the replayed dataflow (`g2-design.md` §8): TensorKit
//! `add_transform!` / `add_transform_kernel!`
//! (src/tensors/indexmanipulations.jl:547/:579/:648/:660 @cfaa073) and QSpace
//! `Permute`/`permute` (Source/QSpace.hh:2837/:2890 @dd2cc7e). Both apply one
//! coefficient-scaled strided move per block pair; this executor submits one
//! `cuda_region_axpby` per block pair, in the structure's block order.
//!
//! Scope: Single blocks only. A structure carrying a recoupling (Multi) block
//! is [`OperationError::UnsupportedDeviceTreeTransform`], rejected before any
//! device work (leaf G2a-3 adds the pack → `Uᵀ` → scatter path).

use core::any::TypeId;
use std::sync::Arc;

use tenet_core::{BlockStructure, Placement, TensorStorage};
use tenet_dense::{
    cuda_region_axpby, cuda_region_zero, CudaDenseContext, CudaDenseStorage, CudaRegion,
    CudaRegionBeta, CudaScalar,
};

use crate::cuda::CudaStorage;
use crate::cuda_transform_plan::{
    compile_device_plan, contains_multi_blocks, plan_cache_entries_for, DeviceTransformPlan,
    StructureCache, StructureKey, CUTENSOR_PLAN_BYTES,
};
use crate::opaque_admission::{
    validate_stage_a, validate_stage_c, AllocationIdentity, CoefficientReadiness, ContextIdentity,
    ExecutorSnapshot, StorageDomain, StorageRegion, StorageSnapshot, WorkspaceSnapshot,
};
use crate::task_view::TreeTransformTaskView;
use crate::{OperationError, RecouplingCoefficientAction, TreeTransformStructure};

/// How a device replay treats the destination it writes.
///
/// [`Self::Overwrite`] is the destination-independent mode the typed
/// `*_overwrite_into` APIs run in: every inactive destination layout is zeroed
/// and every active one is assigned, so a destination holding NaN comes back
/// clean. [`Self::Axpby`] accumulates into the destination and is supported for
/// `beta == 1` only — see [`CudaTreeTransformExecutor::replay`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaTreeTransformDestination<D> {
    Overwrite,
    Axpby(D),
}

/// One Single block as the region pair the primitive takes, built when the
/// structure is prepared so a replay allocates nothing on the host either.
struct PreparedMove {
    source: CudaRegion,
    destination: CudaRegion,
    coefficient: usize,
}

/// Device state prepared once per completed structure.
struct PreparedStructure {
    moves: Vec<PreparedMove>,
    zeros: Vec<CudaRegion>,
    max_zero_len: usize,
    plan_signatures: usize,
    /// The structure's coefficients converted to the payload dtype and
    /// uploaded as one device vector; block `b`'s coefficient is the 1x1
    /// operand at its own index. `None` only for a structure with no
    /// coefficients at all, which has no Single block either.
    coefficients: Option<CudaDenseStorage>,
}

/// Placement-aware executor replaying Single-block tree transforms on device
/// buffers.
///
/// It owns the per-structure device state that makes a warm replay free of
/// host↔device traffic: one uploaded coefficient vector per
/// (structure, payload dtype, context), bounded by a byte budget and purged
/// when the structure itself is dropped. It is *not* a semantic cache — the
/// values it holds are a dtype conversion of data the structure already owns —
/// and dropping it changes nothing but the cost of the next replay.
///
/// One executor belongs next to one device execution context. Entries of other
/// contexts can never be read back, because the context identity is part of
/// every key.
pub struct CudaTreeTransformExecutor {
    prepared: StructureCache<PreparedStructure>,
    coefficient_budget_bytes: usize,
    plan_cache_budget_bytes: usize,
    required_plan_entries: usize,
}

/// Default device budget for uploaded coefficient vectors: 16 MiB, i.e. two
/// million f64 coefficients held for reuse. A structure larger than the whole
/// budget is still prepared; the budget bounds what is *retained* for other
/// structures, and evicting the structure under replay would only re-upload it.
pub const DEFAULT_COEFFICIENT_BUDGET_BYTES: usize = 16 * 1024 * 1024;

/// Default ceiling on the cuTENSOR plan entries this executor asks for: 8 MiB
/// at roughly 14 KB per plan, about 585 distinct operand signatures. Tenferro's
/// own default bound is 64, which a transform with more distinct block layouts
/// than that would thrash.
pub const DEFAULT_PLAN_CACHE_BUDGET_BYTES: usize = 8 * 1024 * 1024;

impl Default for CudaTreeTransformExecutor {
    fn default() -> Self {
        Self::new(
            DEFAULT_COEFFICIENT_BUDGET_BYTES,
            DEFAULT_PLAN_CACHE_BUDGET_BYTES,
        )
    }
}

impl CudaTreeTransformExecutor {
    pub fn new(coefficient_budget_bytes: usize, plan_cache_budget_bytes: usize) -> Self {
        Self {
            prepared: StructureCache::new(coefficient_budget_bytes),
            coefficient_budget_bytes,
            plan_cache_budget_bytes,
            required_plan_entries: 0,
        }
    }

    /// Device bytes this executor retains for reuse, plus the bytes the
    /// context's shared scalar operands (the `1` and the zero template) pin on
    /// its behalf. This is the number the device workspace budget charges.
    pub fn retained_device_bytes(&self, ctx: &CudaDenseContext) -> usize {
        self.prepared
            .retained_bytes()
            .saturating_add(ctx.scalar_operand_bytes())
    }

    /// Structures with device state currently held.
    pub fn prepared_structures(&self) -> usize {
        self.prepared.entry_count()
    }

    /// Distinct cuTENSOR operand signatures the currently prepared structures
    /// submit together, i.e. the plan-cache size a thrash-free warm replay of
    /// all of them needs.
    pub fn required_plan_entries(&self) -> usize {
        self.required_plan_entries
    }

    /// Drops every prepared structure. The next replay re-prepares whatever it
    /// needs, so this is a memory decision, never a correctness one.
    pub fn clear(&mut self) {
        self.prepared = StructureCache::new(self.coefficient_budget_bytes);
        self.required_plan_entries = 0;
    }

    /// Replays `structure` from `src` into `dst` on `ctx`'s device.
    ///
    /// Semantics are the host executor's, for the same structure: each Single
    /// block writes `coefficient * [conj] src_block` into its destination
    /// layout, `Overwrite` additionally zeroes every inactive destination
    /// layout, and block order is the structure's own. The source conjugation
    /// baked into the structure (`storage_conjugate`) is honoured as a flag on
    /// the read, never as a materialized buffer.
    ///
    /// # Capability boundaries
    ///
    /// All of these are reported before any device work — no upload, no
    /// allocation, no plan-cache change:
    ///
    /// - a structure containing a recoupling (Multi) block (leaf G2a-3);
    /// - `Axpby(beta)` with `beta != 1`. Tenferro 0.5.0 has no in-place
    ///   strided scale, so a general `beta` is inexpressible; `beta == 0` is
    ///   rejected as well rather than silently answered with `Overwrite`,
    ///   whose treatment of a NaN destination differs (host `0 * NaN` is NaN,
    ///   an overwriting device region write is clean);
    /// - a layout the device region primitive cannot express (a negative
    ///   stride or offset, or a destination that is not proven injective).
    ///
    /// # Cost
    ///
    /// One `dot_general` submission per non-empty block and per inactive
    /// destination layout, the same count as the host's strided passes. The
    /// first replay of a structure uploads its coefficient vector and sizes
    /// the context zero template; every later replay of the same structure on
    /// the same context and dtype transfers nothing and allocates no device
    /// buffer.
    #[expect(
        clippy::too_many_arguments,
        reason = "the device replay boundary keeps context, structure, replay structures, buffers and destination mode explicit, exactly as the host raw entry points do"
    )]
    pub fn replay<D, C>(
        &mut self,
        ctx: &mut CudaDenseContext,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst: &mut CudaStorage<D>,
        src: &CudaStorage<D>,
        mode: CudaTreeTransformDestination<D>,
    ) -> Result<(), OperationError>
    where
        D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
        C: Copy,
    {
        let beta = match mode {
            CudaTreeTransformDestination::Overwrite => CudaRegionBeta::Overwrite,
            CudaTreeTransformDestination::Axpby(beta) if beta == D::ONE => {
                CudaRegionBeta::Accumulate
            }
            CudaTreeTransformDestination::Axpby(_) => {
                return Err(OperationError::UnsupportedDeviceTreeTransform {
                    message: "device tree transform accumulates with beta = 1 only",
                })
            }
        };
        let task = structure.task_view()?;
        if contains_multi_blocks(task) {
            return Err(OperationError::UnsupportedDeviceTreeTransform {
                message: "device tree transform replays Single blocks only",
            });
        }

        let context = ContextIdentity(ctx.identity());
        let executor = ExecutorSnapshot {
            placement: Placement::Cuda(ctx.device()),
            context,
            supports_strided: true,
            // Recoupling GEMMs are leaf G2a-3; declaring the capability absent
            // is what makes a Multi structure inadmissible rather than
            // half-executed.
            supports_matrix: false,
            scalar: TypeId::of::<D>(),
            // No host fused-index scratch: the device reads the structure's
            // baked fused layout directly, so it declares zero workers.
            fused_index_workers: 0,
        };
        let dst_snapshot = storage_snapshot::<D>(dst, executor);
        let src_snapshot = storage_snapshot::<D>(src, executor);
        let admission = validate_stage_a::<D, C>(
            task,
            dst_structure,
            src_structure,
            dst_snapshot,
            src_snapshot,
            executor,
        )?;

        self.prepare::<D, C>(ctx, task, context)?;
        let workspace = WorkspaceSnapshot {
            packed_source: empty_workspace_snapshot(executor),
            packed_destination: empty_workspace_snapshot(executor),
            converted_coefficients: empty_workspace_snapshot(executor),
            fused_index_capacity: 0,
            fused_index_placement: executor.placement,
            coefficient_readiness: Some(CoefficientReadiness {
                structure_and_layout: task.admission_identity(),
                scalar: TypeId::of::<D>(),
                context,
            }),
        };
        validate_stage_c::<D, C>(
            task,
            dst_snapshot,
            src_snapshot,
            &workspace,
            executor,
            &admission,
        )?;

        let key = StructureKey {
            structure: task.admission_identity(),
            scalar: TypeId::of::<D>(),
            context: ctx.identity(),
        };
        let prepared = self
            .prepared
            .get(&key)
            .ok_or_else(|| OperationError::InvalidArgument {
                message: "device tree transform state disappeared between preparation and replay",
            })?;
        let conjugate = task.storage_conjugate();

        if matches!(mode, CudaTreeTransformDestination::Overwrite) {
            for zero in &prepared.zeros {
                cuda_region_zero::<D>(ctx, &mut dst.0, zero).map_err(OperationError::Dense)?;
            }
        }
        for entry in &prepared.moves {
            let Some(coefficients) = prepared.coefficients.as_ref() else {
                return Err(OperationError::InvalidArgument {
                    message: "device tree transform block has no uploaded coefficient",
                });
            };
            cuda_region_axpby::<D>(
                ctx,
                &src.0,
                &entry.source,
                conjugate,
                Some((coefficients, entry.coefficient)),
                beta,
                &mut dst.0,
                &entry.destination,
            )
            .map_err(OperationError::Dense)?;
        }
        Ok(())
    }

    /// Uploads this structure's coefficient vector and sizes the shared device
    /// operands, unless the state is already prepared for this dtype and
    /// context. Runs after Stage A, so nothing is uploaded for a task the
    /// admission rejects.
    fn prepare<D, C>(
        &mut self,
        ctx: &mut CudaDenseContext,
        task: TreeTransformTaskView<'_, C>,
        context: ContextIdentity,
    ) -> Result<(), OperationError>
    where
        D: CudaScalar + RecouplingCoefficientAction<C> + 'static,
        C: Copy,
    {
        let key = StructureKey {
            structure: task.admission_identity(),
            scalar: TypeId::of::<D>(),
            context: context.0,
        };
        if self.prepared.get(&key).is_none() {
            let plan = compile_device_plan(task)?;
            let values: Vec<D> = task
                .coefficients()
                .iter()
                .map(|coefficient| D::coefficient_as_data(*coefficient))
                .collect();
            let device_bytes = core::mem::size_of_val(values.as_slice());
            let coefficients = if values.is_empty() {
                None
            } else {
                Some(
                    CudaDenseStorage::upload_owned::<D>(ctx, values)
                        .map_err(OperationError::Dense)?,
                )
            };
            self.prepared.insert(
                key.clone(),
                prepared_structure(plan, coefficients)?,
                device_bytes,
            );
            self.refresh_plan_cache(ctx)?;
        }
        // Idempotent once the template is long enough, so a warm replay
        // uploads nothing; sized from the structure's longest inactive layout
        // rather than grown one fill at a time.
        let max_zero_len = self
            .prepared
            .get(&key)
            .map_or(0, |prepared| prepared.max_zero_len);
        ctx.reserve_zero_template::<D>(max_zero_len)
            .map_err(OperationError::Dense)?;
        Ok(())
    }

    /// Raises the backend's cuTENSOR plan entry bound to what every prepared
    /// structure needs together, under this executor's byte budget.
    ///
    /// Counting is structural — the distinct baked fused signatures of a
    /// structure, computed once on the host when it is prepared — not a timing
    /// heuristic, and the raise is monotonic, so a cap another caller set is
    /// never lowered.
    fn refresh_plan_cache(&mut self, ctx: &CudaDenseContext) -> Result<(), OperationError> {
        let required = self.prepared.values().fold(0usize, |total, prepared| {
            total.saturating_add(prepared.plan_signatures)
        });
        self.required_plan_entries = required;
        let entries = plan_cache_entries_for(required, self.plan_cache_budget_bytes);
        ctx.raise_plan_cache_max_entries(entries)
            .map_err(OperationError::Dense)
    }
}

/// Turns the host plan into the device regions the primitive takes. Every
/// layout rule the regions must satisfy is checked by the primitive itself on
/// each call; what is done once here is only the descriptor construction.
fn prepared_structure(
    plan: DeviceTransformPlan,
    coefficients: Option<CudaDenseStorage>,
) -> Result<PreparedStructure, OperationError> {
    let mut moves = Vec::with_capacity(plan.moves.len());
    for entry in plan.moves {
        moves.push(PreparedMove {
            source: CudaRegion::new(entry.dims.clone(), entry.src_strides, entry.src_offset)
                .map_err(OperationError::Dense)?,
            destination: CudaRegion::new(entry.dims, entry.dst_strides, entry.dst_offset)
                .map_err(OperationError::Dense)?,
            coefficient: entry.coefficient,
        });
    }
    let mut zeros = Vec::with_capacity(plan.zeros.len());
    for zero in plan.zeros {
        zeros.push(
            CudaRegion::new(zero.dims, zero.strides, zero.offset).map_err(OperationError::Dense)?,
        );
    }
    Ok(PreparedStructure {
        moves,
        zeros,
        max_zero_len: plan.max_zero_len,
        plan_signatures: plan.plan_signatures,
        coefficients,
    })
}

/// Bytes one cuTENSOR plan is assumed to retain, re-exported for callers
/// sizing the plan-cache budget.
pub const CUTENSOR_PLAN_ENTRY_BYTES: usize = CUTENSOR_PLAN_BYTES;

/// Admission facts of one device buffer.
///
/// The allocation identity is the address of the storage handle. That is a
/// sound identity for the duration of the call: `CudaDenseStorage` is neither
/// `Clone` nor refcounted and owns its device allocation exclusively, so two
/// live handles at the same address cannot exist, and the snapshot never
/// outlives the borrows it is taken from.
fn storage_snapshot<D: CudaScalar>(
    storage: &CudaStorage<D>,
    executor: ExecutorSnapshot,
) -> StorageSnapshot {
    let len = TensorStorage::<D>::len(storage);
    let region = if len == 0 {
        StorageRegion::Empty
    } else {
        StorageRegion::Bytes {
            domain: StorageDomain(storage.0.device() as u64),
            allocation: AllocationIdentity(std::ptr::from_ref(&storage.0) as u64),
            start: 0,
            len: len.saturating_mul(core::mem::size_of::<D>()),
        }
    };
    StorageSnapshot {
        active_len: len,
        usable_capacity: len,
        placement: executor.placement,
        context: executor.context,
        region,
    }
}

/// The workspace slots a Single-block replay does not use. Their required
/// lengths are zero, so an empty region is the honest description rather than
/// a dummy buffer.
fn empty_workspace_snapshot(executor: ExecutorSnapshot) -> StorageSnapshot {
    StorageSnapshot {
        active_len: 0,
        usable_capacity: 0,
        placement: executor.placement,
        context: executor.context,
        region: StorageRegion::Empty,
    }
}
