//! Device replay of tree transforms on [`CudaStorage`].
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
//! A Multi block is the one place where a block pair is not a single move: the
//! host packs each source layout into a compact column, multiplies the column
//! block by the transposed recoupling matrix, and scatters the result columns
//! back out. This executor submits exactly that sequence, in the compile-time
//! recoupling plan's own job order.
//!
//! Reference record for the replayed dataflow (`g2-design.md` §8): TensorKit
//! `add_transform!` / `add_transform_kernel!`
//! (src/tensors/indexmanipulations.jl:547/:579/:648/:660 @cfaa073), whose
//! recoupling step is `mul!(dst, src, transpose(U))`
//! (indexmanipulations.jl:629/:701 @cfaa073), and QSpace `Permute`/`permute`
//! (Source/QSpace.hh:2837/:2890 @dd2cc7e). Host authority for the lowering is
//! `transform_replay.rs`: `pack_layout_into_column`, `recoupling_gemm_batch`,
//! `scatter_column_into_layout`.

use core::any::TypeId;
use std::sync::Arc;

use tenet_core::{BlockStructure, Placement, TensorStorage};
use tenet_dense::{
    cuda_matmul_region_into, cuda_region_axpby, cuda_region_zero, CudaDenseContext,
    CudaDenseStorage, CudaRegion, CudaRegionBeta, CudaRegionCoefficient, CudaScalar, DenseError,
};

use crate::cuda::CudaStorage;
use crate::cuda_transform_plan::{
    compile_device_plan, plan_cache_entries_for, DeviceMoveSpec, DeviceTransformPlan,
    StructureCache, StructureKey, DEFAULT_STRUCTURE_CACHE_ENTRIES,
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

/// One block, pack column or scatter column as the region pair the primitive
/// takes, built when the structure is prepared so a replay allocates nothing on
/// the host either.
///
/// `coefficient` is the index of the 1x1 operand in the uploaded coefficient
/// vector, or `None` where the host copies unscaled and the context's shared
/// `1` is the operand.
struct PreparedMove {
    source: CudaRegion,
    destination: CudaRegion,
    coefficient: Option<usize>,
}

/// One Multi block's pack → `Uᵀ` GEMM → scatter sequence, with every offset
/// already resolved against the structure-wide device workspace.
struct PreparedRecoupling {
    packs: Vec<PreparedMove>,
    scatters: Vec<PreparedMove>,
    /// Workspace destination offset, workspace source offset, and the offset of
    /// this block's recoupling matrix inside the uploaded coefficient vector —
    /// the block's own `coefficient_start`, because the device holds the whole
    /// converted payload and needs no re-pack.
    dst_offset: usize,
    lhs_offset: usize,
    matrix_offset: usize,
    rows: usize,
    contracted: usize,
    cols: usize,
}

/// Device state prepared once per completed structure.
struct PreparedStructure {
    moves: Vec<PreparedMove>,
    recouplings: Vec<PreparedRecoupling>,
    /// Whether the recouplings address any workspace element at all. A Multi
    /// block of zero element count submits nothing, so no workspace is grown
    /// for it and none is looked up.
    needs_workspace: bool,
    zeros: Vec<CudaRegion>,
    max_zero_len: usize,
    plan_signatures: usize,
    /// The structure's whole coefficient payload converted to the payload dtype
    /// and uploaded as one device vector. Block `b`'s scalar is the 1x1 operand
    /// at its own index and Multi block `b`'s recoupling matrix is the
    /// `S_b x D_b` run at its own `coefficient_start`, both addressed exactly as
    /// the structure indexes them. `None` only for a structure with no
    /// coefficients at all.
    coefficients: Option<CudaDenseStorage>,
    /// How many elements of the payload the recoupling matrices occupy, which
    /// is what Stage C requires of the converted-coefficient slot.
    recoupling_len: usize,
    /// The coefficient-readiness fact, recorded when the upload happened rather
    /// than rebuilt from the lookup key at replay time: Stage C then checks
    /// something this executor observed, not something it restated.
    readiness: CoefficientReadiness,
}

/// The device scratch a Multi replay packs into and scatters out of, held per
/// payload dtype and context and grown monotonically.
///
/// It is execution scratch, not a semantic cache: every column it holds is
/// fully written before it is read, so its contents never carry meaning across
/// replays and dropping it changes nothing but the cost of the next one. It is
/// two buffers rather than one because the GEMM borrows its source shared and
/// its destination exclusively.
struct DeviceWorkspace {
    scalar: TypeId,
    context: u64,
    source: CudaDenseStorage,
    destination: CudaDenseStorage,
    bytes: usize,
}

/// Placement-aware executor replaying tree transforms on device buffers.
///
/// It owns the per-structure device state that makes a warm replay free of
/// host↔device traffic: one uploaded coefficient-and-recoupling-matrix vector
/// per (structure, payload dtype, context), bounded by a byte budget and an
/// entry count and purged when the structure itself is dropped, plus the pack/
/// scatter workspace, grown monotonically per (payload dtype, context). Neither
/// is a semantic cache — the vector is a dtype conversion of data the structure
/// already owns and the workspace is fully rewritten before it is read — so
/// dropping them changes nothing but the cost of the next replay.
///
/// One executor belongs next to one device execution context. Entries of other
/// contexts can never be read back, because the context identity is part of
/// every key. Because a replay needs `&mut self` and `&mut CudaDenseContext`
/// at once, its owner in G2b is the runtime's device state beside the
/// `CudaDenseContext`, inside the same device mutex the GL-3 lease takes; it
/// is deliberately not global and not owned by any structure.
pub struct CudaTreeTransformExecutor {
    prepared: StructureCache<PreparedStructure>,
    /// One transform workspace per (payload dtype, context) actually replayed;
    /// in practice one or two. It is keyed rather than shared because a device
    /// buffer carries its payload dtype, and it is owned here rather than by a
    /// structure so that two structures of the same dtype reuse one allocation.
    workspaces: Vec<DeviceWorkspace>,
    coefficient_budget_bytes: usize,
    structure_entries: usize,
    plan_cache_budget_bytes: usize,
    required_plan_entries: usize,
}

/// Default device budget for uploaded coefficient vectors: 16 MiB, i.e. two
/// million f64 coefficients held for reuse. A structure larger than the whole
/// budget is still prepared; the budget bounds what is *retained* for other
/// structures, and evicting the structure under replay would only re-upload it.
pub const DEFAULT_COEFFICIENT_BUDGET_BYTES: usize = 16 * 1024 * 1024;

/// Default ceiling on the cuTENSOR plan entries this executor asks for: 8 MiB
/// at an estimated 14 KB per plan, about 585 distinct operand signatures.
/// Tenferro's own default bound is 64, which a transform with more distinct
/// block layouts than that would thrash. The per-plan figure is an estimate of
/// a Tenferro internal, so it bounds only how far this executor is willing to
/// raise the bound.
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
    /// An executor whose prepared-structure cache keeps at most 256 structures
    /// — the host transform cache's own entry bound, so a working set that is
    /// warm on the host stays warm here. Use [`Self::with_structure_entries`]
    /// for another bound.
    pub fn new(coefficient_budget_bytes: usize, plan_cache_budget_bytes: usize) -> Self {
        Self::with_structure_entries(
            coefficient_budget_bytes,
            plan_cache_budget_bytes,
            DEFAULT_STRUCTURE_CACHE_ENTRIES,
        )
    }

    /// An executor whose prepared-structure cache keeps at most
    /// `structure_entries` structures.
    ///
    /// The bound belongs to the caller because it is the device half of the
    /// host transform cache's own bound: a workload whose structures are warm
    /// on the host must stay warm here, or the zero-transfer replay contract
    /// quietly stops holding. Sizing it below the host's is a memory decision,
    /// never a correctness one.
    pub fn with_structure_entries(
        coefficient_budget_bytes: usize,
        plan_cache_budget_bytes: usize,
        structure_entries: usize,
    ) -> Self {
        Self {
            prepared: StructureCache::new(coefficient_budget_bytes, structure_entries),
            workspaces: Vec::new(),
            coefficient_budget_bytes,
            structure_entries,
            plan_cache_budget_bytes,
            required_plan_entries: 0,
        }
    }

    /// Device bytes this executor retains for reuse: the uploaded coefficient
    /// and recoupling vectors, the transform workspaces, and the bytes the
    /// context's shared scalar operands (the ones and zero templates) pin on
    /// its behalf. This is the number the device workspace budget charges.
    pub fn retained_device_bytes(&self, ctx: &CudaDenseContext) -> usize {
        self.executor_device_bytes()
            .saturating_add(ctx.scalar_operand_bytes())
    }

    /// Device bytes the executor itself retains: the uploaded coefficient and
    /// recoupling vectors plus the transform workspaces, and nothing the
    /// context owns. This is [`Self::retained_device_bytes`] without the shared
    /// context operands, so a caller reporting both can count each once.
    pub fn executor_device_bytes(&self) -> usize {
        self.prepared
            .retained_bytes()
            .saturating_add(self.workspace_device_bytes())
    }

    /// Device bytes the transform workspaces hold. Separate from
    /// [`Self::retained_device_bytes`] so a test can pin that the workspace
    /// does not grow on a warm replay.
    pub fn workspace_device_bytes(&self) -> usize {
        self.workspaces.iter().fold(0usize, |total, workspace| {
            total.saturating_add(workspace.bytes)
        })
    }

    /// Structures with device state currently held.
    pub fn prepared_structures(&self) -> usize {
        self.prepared.entry_count()
    }

    /// Distinct cuTENSOR operand signatures the currently prepared structures
    /// submit together, i.e. the plan-cache size a thrash-free warm replay of
    /// all of them needs.
    ///
    /// The caller scale does not enter it. A zero scale swaps the buffer the
    /// 1x1 coefficient operand is read from, but not that operand's view
    /// metadata (`[1, 1]` extents and strides) nor its alignment, which is
    /// `size_of::<D>()` for every view, and the accumulation scalars are not in
    /// a contraction plan key at all.
    pub fn required_plan_entries(&self) -> usize {
        self.required_plan_entries
    }

    /// Raises the backend's plan entry bound to what the prepared structures
    /// need plus `signatures` more, under this executor's plan-cache budget:
    /// the same rule [`Self::required_plan_entries`] is raised by, for a
    /// caller that submits through the same context without a prepared
    /// structure (the device trace). Monotonic like every raise.
    pub fn raise_plan_cache_for_additional(
        &self,
        ctx: &CudaDenseContext,
        signatures: usize,
    ) -> Result<(), OperationError> {
        let entries = plan_cache_entries_for(
            self.required_plan_entries.saturating_add(signatures),
            self.plan_cache_budget_bytes,
        );
        ctx.raise_plan_cache_max_entries(entries)
            .map_err(OperationError::Dense)
    }

    /// Drops every prepared structure. The next replay re-prepares whatever it
    /// needs, so this is a memory decision, never a correctness one.
    pub fn clear(&mut self) {
        self.prepared = StructureCache::new(self.coefficient_budget_bytes, self.structure_entries);
        self.workspaces.clear();
        self.required_plan_entries = 0;
    }

    /// Replays `structure` from `src` into `dst` on `ctx`'s device.
    ///
    /// Semantics are the host executor's, for the same structure: each Single
    /// block writes `coefficient * [conj] src_block` into its destination
    /// layout, each Multi block packs its source layouts into compact columns,
    /// multiplies them by the transposed recoupling matrix and scatters the
    /// result columns into its destination layouts, `Overwrite` additionally
    /// zeroes every inactive destination layout, and the order is the
    /// structure's own block order followed by the recoupling plan's job order.
    /// The source conjugation baked into the structure (`storage_conjugate`) is
    /// honoured as a flag on the read, never as a materialized buffer.
    ///
    /// # Caller scale
    ///
    /// `alpha` is the caller's scale on the transformed source, the host's own
    /// `alpha` argument (`dst = alpha * T(src)` under [`Overwrite`], plus
    /// `1 * dst` under [`Axpby(1)`]). It reaches exactly where the host applies
    /// it: every Single-block move and every Multi-block *scatter*, never the
    /// pack columns and never the recoupling GEMM, which the host runs with
    /// `1`. Zero fills of inactive destination layouts ignore it, as they do on
    /// the host, so [`Overwrite`] still cleans a poisoned destination.
    ///
    /// `alpha == 0` (IEEE comparison, so `-0.0` is a zero scale) is *not* a
    /// short circuit: the host multiplies there too, so a NaN or infinite
    /// source must still poison the destination. It is submitted as an exact
    /// zero 1x1 *operand* with descriptor `1`, because a descriptor `alpha` of
    /// zero lets CUDA skip the source read.
    ///
    /// Disclosed differences from the host's arithmetic, all within dtype
    /// tolerance and none of them a change of the written block set:
    ///
    /// - a Single block rounds as `alpha * (c * x)` where the host folds the
    ///   scales first and rounds as `(alpha * c) * x`; the overflow position
    ///   moves with it. Multi blocks agree exactly in order (`alpha * (U x)`);
    /// - at `alpha == 0` the written zeros carry the sign of `0 * x` alone,
    ///   where the host's carries `sign(alpha) * sign(c)` as well;
    /// - the caller scale is not part of any cache or plan key: the 1x1 operand
    ///   keeps the same view metadata and alignment whichever buffer it is read
    ///   from, so an `alpha == 0` replay adds no cuTENSOR plan signature to
    ///   [`Self::required_plan_entries`] and no prepared-structure entry.
    ///
    /// [`Overwrite`]: CudaTreeTransformDestination::Overwrite
    /// [`Axpby(1)`]: CudaTreeTransformDestination::Axpby
    ///
    /// # Capability boundaries
    ///
    /// All of these are reported before any device work — no upload, no
    /// allocation, no plan-cache change:
    ///
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
    /// One `dot_general` submission per non-empty Single block, per pack and
    /// scatter column and per inactive destination layout, plus one GEMM per
    /// recoupling job — independent of `alpha`, which costs nothing — the same counts as the host's strided passes and its
    /// `matmul_batch_axpby_into` jobs. The first replay of a structure uploads
    /// its coefficient vector, sizes the context zero template and grows the
    /// transform workspace to `Σ_b element_count_b × (src_count_b + dst_count_b)`
    /// elements; every later replay of the same structure on the same context
    /// and dtype transfers nothing and allocates no device buffer.
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
        alpha: D,
        mode: CudaTreeTransformDestination<D>,
    ) -> Result<(), OperationError>
    where
        D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
        C: Copy,
    {
        self.replay_with_destination_scales(
            ctx,
            structure,
            dst_structure,
            src_structure,
            dst,
            src,
            alpha,
            mode,
            &[],
        )
    }

    /// [`Self::replay`] with the caller scale of destination block `b`
    /// multiplied by `θ_b`: block `b` receives `alpha * θ_b * T(src)_b` —
    /// under [`Overwrite`], the plain replay followed by scaling block `b` by
    /// `θ_b` (up to the sign of the zeros it writes to inactive layouts); under
    /// [`Axpby(1)`], `dst_b + alpha * θ_b * T(src)_b`. `destination_scales`
    /// lists `(destination block offset, θ_b)` sorted by strictly increasing
    /// offset; a block it does not list has `θ_b = 1`. This is the fermionic
    /// core-right twist of a general
    /// contraction, which the host applies as a separate in-place scale of
    /// the transformed operand (`execute_rhs_contract_twist`, TensorKit's
    /// `twist!` after `tensoradd!` in `blas_contract!`, tensoroperations.jl:429
    /// @cfaa073) and QSpace folds into its per-block GEMM scalar
    /// (QSpace_aux.cc:105-115 @dd2cc7e).
    ///
    /// The θ reaches only the descriptor alpha of the Single move or Multi
    /// scatter that writes block `b`, which becomes `alpha * θ_b`. Packs, the
    /// recoupling GEMM and the inactive-layout zero fills stay unscaled, so the
    /// write pass the host follows with a scale pass does both: one pass
    /// fewer than the host and TensorKit. A Multi block's GEMM is shared by
    /// its destination layouts, and scaling each scatter by its own θ is exact
    /// either way; the contraction compiler additionally asserts θ is uniform
    /// per Multi block, which is what makes this the host's "scale after the
    /// transform".
    ///
    /// The prepared structure, its uploaded coefficient vector and its cache
    /// key are those of the unscaled replay — θ is in no key — and a scaled
    /// replay uploads nothing more: θ is a descriptor scalar, not an operand.
    ///
    /// `alpha == 0` keeps the zero-operand route of [`Self::replay`] and ignores
    /// θ: the written values are `0 * x`, NaN for a NaN source, as the host's
    /// `θ * (0 * x)`; only the sign of an exact zero may differ (never compare
    /// bitwise). A fermionic twist is `±1`, so `alpha * θ_b` is non-zero for
    /// every non-zero `alpha`; a θ that would make it zero is rejected, since a
    /// zero descriptor alpha would let CUDA skip the source read.
    ///
    /// [`Overwrite`]: CudaTreeTransformDestination::Overwrite
    /// [`Axpby(1)`]: CudaTreeTransformDestination::Axpby
    ///
    /// # Errors
    ///
    /// Those of [`Self::replay`], and `InvalidArgument` — before any device
    /// work — for offsets that are not strictly increasing or, with a
    /// non-zero `alpha`, a θ for which `alpha * θ` is zero.
    #[doc(hidden)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the replay boundary plus the per-destination-block scale list"
    )]
    pub fn replay_with_destination_scales<D, C>(
        &mut self,
        ctx: &mut CudaDenseContext,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst: &mut CudaStorage<D>,
        src: &CudaStorage<D>,
        alpha: D,
        mode: CudaTreeTransformDestination<D>,
        destination_scales: &[(usize, C)],
    ) -> Result<(), OperationError>
    where
        D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
        C: Copy,
    {
        if destination_scales
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
        {
            return Err(OperationError::InvalidArgument {
                message: "device tree transform destination scales must have strictly \
                          increasing offsets",
            });
        }
        if alpha != D::ZERO
            && destination_scales
                .iter()
                .any(|&(_, theta)| alpha.scale_by_coefficient(theta) == D::ZERO)
        {
            return Err(OperationError::InvalidArgument {
                message: "device tree transform destination scale must not vanish",
            });
        }
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
        let context = ContextIdentity(ctx.identity());
        let executor = ExecutorSnapshot {
            placement: Placement::Cuda(ctx.device()),
            context,
            supports_strided: true,
            // Recoupling GEMMs are submitted through the region GEMM adapter.
            supports_matrix: true,
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

        let key = StructureKey {
            structure: task.admission_identity(),
            scalar: TypeId::of::<D>(),
            context: ctx.identity(),
        };
        // One key scan per replay: the entry is resolved to an index here and
        // everything below reads it through that index.
        let index = match self.prepared.position(&key) {
            Some(index) => index,
            None => {
                self.prepare::<D, C>(ctx, task, &key)?;
                self.prepared
                    .position(&key)
                    .ok_or(OperationError::InvalidArgument {
                        message: "device tree transform state was not prepared",
                    })?
            }
        };

        let Self {
            prepared,
            workspaces,
            ..
        } = self;
        let prepared = prepared
            .touch(index)
            .ok_or(OperationError::InvalidArgument {
                message: "device tree transform state disappeared between preparation and replay",
            })?;
        let scratch = workspaces.iter().position(|workspace| {
            workspace.scalar == TypeId::of::<D>() && workspace.context == key.context
        });
        let workspace =
            workspace_snapshot::<D>(prepared, scratch.map(|index| &workspaces[index]), executor);
        validate_stage_c::<D, C>(
            task,
            dst_snapshot,
            src_snapshot,
            &workspace,
            executor,
            &admission,
        )?;

        // The host never short-circuits the caller scale: with `alpha == 0` it
        // still computes `0 * src`, so a NaN or infinite source reaches the
        // destination (`kernel_adapter.rs` Overwrite kernel, `alpha * value`).
        // A descriptor alpha of 0 would let CUDA skip the read instead, so a
        // zero scale is submitted as the zero *operand* with descriptor 1.
        // `-0.0 == 0.0` under IEEE comparison, which is exactly the intended
        // test: a caller scale of `-0.0` multiplies by zero too.
        let scale = if alpha == D::ZERO {
            CallerScale::ZeroOperand
        } else {
            CallerScale::Descriptor(alpha)
        };

        // Idempotent once the template is long enough, so a warm replay uploads
        // nothing; sized from the structure's longest inactive layout rather
        // than grown one fill at a time. After Stage C, so no admission verdict
        // is still outstanding when it uploads — the design's "reserve before
        // Stage C" wording would upload ahead of a verdict that can still
        // reject, which the rejection-order contract forbids; reserving here is
        // still before every submission, which is what the zero operand needs.
        // A zero caller scale reads element 0 of the same template as its 1x1
        // operand, so one element is reserved even for a structure with no
        // inactive destination layout.
        let template_len = prepared
            .max_zero_len
            .max(usize::from(matches!(scale, CallerScale::ZeroOperand)));
        ctx.reserve_zero_template::<D>(template_len)
            .map_err(OperationError::Dense)?;

        let conjugate = task.storage_conjugate();
        if matches!(mode, CudaTreeTransformDestination::Overwrite) {
            for zero in &prepared.zeros {
                cuda_region_zero::<D>(ctx, &mut dst.0, zero).map_err(OperationError::Dense)?;
            }
        }
        for entry in &prepared.moves {
            submit_move::<D>(
                ctx,
                &src.0,
                prepared.coefficients.as_ref(),
                entry,
                conjugate,
                destination_scaled(scale, destination_scales, entry),
                beta,
                &mut dst.0,
            )?;
        }
        if !prepared.needs_workspace {
            return Ok(());
        }
        let scratch = scratch.map(|index| &mut workspaces[index]).ok_or(
            OperationError::InvalidArgument {
                message:
                    "device tree transform workspace disappeared between preparation and replay",
            },
        )?;
        let coefficients =
            prepared
                .coefficients
                .as_ref()
                .ok_or(OperationError::InvalidArgument {
                    message: "device tree transform recoupling block has no uploaded matrix",
                })?;
        // Host order exactly: pack every source column of one job, multiply,
        // scatter every destination column, then move to the next job
        // (`transform_replay.rs`, the serial `recoupling_plan.entries()` loop).
        for recoupling in &prepared.recouplings {
            for pack in &recoupling.packs {
                submit_move::<D>(
                    ctx,
                    &src.0,
                    None,
                    pack,
                    conjugate,
                    // The host packs and multiplies unscaled and applies the
                    // caller scale at the scatter alone.
                    CallerScale::Descriptor(D::ONE),
                    CudaRegionBeta::Overwrite,
                    &mut scratch.source,
                )?;
            }
            if recoupling.rows != 0 && recoupling.contracted != 0 && recoupling.cols != 0 {
                // `dst[rows x cols] = lhs[rows x contracted] * rhs[contracted x
                // cols]` with the row-major `U[dst, src]` buffer read as the
                // column-major `(src_count x dst_count)` matrix `Uᵀ`: leading
                // dimension `contracted` makes element `(s, d)` of the operand
                // the host's `U[d][s]`. This is the host's
                // `recoupling_gemm_batch` contract and TensorKit's
                // `mul!(dst, src, transpose(U))`.
                cuda_matmul_region_into::<D>(
                    ctx,
                    &mut scratch.destination,
                    recoupling.dst_offset,
                    &scratch.source,
                    recoupling.lhs_offset,
                    coefficients,
                    recoupling.matrix_offset,
                    recoupling.rows,
                    recoupling.contracted,
                    recoupling.cols,
                )
                .map_err(OperationError::Dense)?;
            }
            for scatter in &recoupling.scatters {
                submit_move::<D>(
                    ctx,
                    &scratch.destination,
                    None,
                    scatter,
                    // The host scatter copies the packed column unconjugated:
                    // the conjugation was already applied on the pack.
                    false,
                    destination_scaled(scale, destination_scales, scatter),
                    beta,
                    &mut dst.0,
                )?;
            }
        }
        Ok(())
    }

    fn prepare<D, C>(
        &mut self,
        ctx: &mut CudaDenseContext,
        task: TreeTransformTaskView<'_, C>,
        key: &StructureKey,
    ) -> Result<(), OperationError>
    where
        D: CudaScalar + RecouplingCoefficientAction<C> + 'static,
        C: Copy,
    {
        // Everything that can reject the structure runs here, before the first
        // byte crosses to the device: every region's expressibility and
        // destination-writability, the recoupling plan's agreement with its
        // blocks, and the workspace size arithmetic.
        let plan = compile_device_plan(task)?;
        task.validate_workspace_requirements::<D>()?;
        let mut prepared = prepared_structure(
            &plan,
            CoefficientReadiness {
                structure_and_layout: key.structure.clone(),
                scalar: TypeId::of::<D>(),
                context: ContextIdentity(key.context),
            },
            task.workspace_requirements().converted_coefficient_len,
        )?;

        self.grow_workspace::<D>(
            ctx,
            key.context,
            plan.workspace_source_len,
            plan.workspace_destination_len,
        )?;

        // The whole payload, converted once, indexed exactly as the structure
        // indexes it: a Single block's scalar at its own coefficient index and a
        // Multi block's matrix as the run at its own `coefficient_start`.
        let values: Vec<D> = task
            .coefficients()
            .iter()
            .map(|coefficient| D::coefficient_as_data(*coefficient))
            .collect();
        let device_bytes = core::mem::size_of_val(values.as_slice());
        if !values.is_empty() {
            prepared.coefficients = Some(
                CudaDenseStorage::upload_owned::<D>(ctx, values).map_err(OperationError::Dense)?,
            );
        }
        self.prepared.insert(key.clone(), prepared, device_bytes);
        self.refresh_plan_cache(ctx)
    }

    /// Makes this (dtype, context)'s transform workspace hold at least
    /// `source_len` and `destination_len` elements.
    ///
    /// Growth is monotonic: a shorter requirement reuses the resident buffers,
    /// so a replay sequence pays at most one allocation per high-water mark and
    /// a warm replay pays none. A workspace is not shrunk, because the next
    /// structure of the same dtype would only have to allocate again.
    fn grow_workspace<D: CudaScalar + 'static>(
        &mut self,
        ctx: &CudaDenseContext,
        context: u64,
        source_len: usize,
        destination_len: usize,
    ) -> Result<(), OperationError> {
        if source_len == 0 && destination_len == 0 {
            return Ok(());
        }
        let scalar = TypeId::of::<D>();
        let resident = self
            .workspaces
            .iter()
            .position(|workspace| workspace.scalar == scalar && workspace.context == context);
        let (have_source, have_destination) = resident.map_or((0, 0), |index| {
            let workspace = &self.workspaces[index];
            (workspace.source.len(), workspace.destination.len())
        });
        if have_source >= source_len && have_destination >= destination_len {
            return Ok(());
        }
        let source_len = source_len.max(have_source);
        let destination_len = destination_len.max(have_destination);
        let bytes = source_len
            .checked_add(destination_len)
            .and_then(|len| len.checked_mul(core::mem::size_of::<D>()))
            .ok_or(OperationError::ElementCountOverflow)?;
        // Growth costs one host↔device transfer of zeros per buffer, the same
        // way every other device allocation in TeNeT is made (#740).
        // `cubecl::Session::alloc_zero_output` exists in Tenferro 0.5.0 but is
        // unusable here: it is broken for complex dtypes (tenferro-rs#1833) and
        // unpublished for kernel-written outputs. The values are irrelevant —
        // every column is fully written before it is read — and a warm replay
        // pays none of it.
        let source = CudaDenseStorage::upload_owned::<D>(ctx, vec![D::ZERO; source_len])
            .map_err(OperationError::Dense)?;
        let destination = CudaDenseStorage::upload_owned::<D>(ctx, vec![D::ZERO; destination_len])
            .map_err(OperationError::Dense)?;
        let grown = DeviceWorkspace {
            scalar,
            context,
            source,
            destination,
            bytes,
        };
        match resident {
            Some(index) => self.workspaces[index] = grown,
            None => self.workspaces.push(grown),
        }
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

/// Turns the host plan into the device regions the primitive takes, rejecting
/// every destination layout the device cannot write before the caller's
/// structure has cost anything on device.
///
/// The layout verdict depends on the region alone, so checking it here is the
/// same answer the submission would give — only early enough to be an
/// all-or-nothing capability boundary instead of a partial overwrite. Pack
/// columns are checked too: they write the workspace, and a source layout the
/// device cannot read is caught by the same region construction.
fn prepared_structure(
    plan: &DeviceTransformPlan,
    readiness: CoefficientReadiness,
    recoupling_len: usize,
) -> Result<PreparedStructure, OperationError> {
    let mut moves = Vec::with_capacity(plan.moves.len());
    for entry in &plan.moves {
        moves.push(prepared_move(entry)?);
    }
    let mut recouplings = Vec::with_capacity(plan.recouplings.len());
    for entry in &plan.recouplings {
        let mut packs = Vec::with_capacity(entry.packs.len());
        for pack in &entry.packs {
            packs.push(prepared_move(pack)?);
        }
        let mut scatters = Vec::with_capacity(entry.scatters.len());
        for scatter in &entry.scatters {
            scatters.push(prepared_move(scatter)?);
        }
        recouplings.push(PreparedRecoupling {
            packs,
            scatters,
            dst_offset: entry.job.dst_offset,
            lhs_offset: entry.job.lhs_offset,
            matrix_offset: entry.matrix_offset,
            rows: entry.job.rows,
            contracted: entry.job.contracted,
            cols: entry.job.cols,
        });
    }
    let mut zeros = Vec::with_capacity(plan.zeros.len());
    for zero in &plan.zeros {
        let region = CudaRegion::new(zero.dims.clone(), zero.strides.clone(), zero.offset)
            .map_err(unsupported_layout)?;
        validate_destination(&region)?;
        zeros.push(region);
    }
    Ok(PreparedStructure {
        moves,
        recouplings,
        needs_workspace: plan.workspace_source_len + plan.workspace_destination_len > 0,
        zeros,
        max_zero_len: plan.max_zero_len,
        plan_signatures: plan.plan_signatures,
        coefficients: None,
        recoupling_len,
        readiness,
    })
}

fn prepared_move(entry: &DeviceMoveSpec) -> Result<PreparedMove, OperationError> {
    let destination = CudaRegion::new(
        entry.dims.clone(),
        entry.dst_strides.clone(),
        entry.dst_offset,
    )
    .map_err(unsupported_layout)?;
    validate_destination(&destination)?;
    Ok(PreparedMove {
        source: CudaRegion::new(
            entry.dims.clone(),
            entry.src_strides.clone(),
            entry.src_offset,
        )
        .map_err(unsupported_layout)?,
        destination,
        coefficient: entry.coefficient,
    })
}

fn validate_destination(region: &CudaRegion) -> Result<(), OperationError> {
    region
        .validate_as_destination(OP)
        .map_err(unsupported_layout)
}

const OP: &str = "cuda_tree_transform";

/// A layout the device region primitive cannot express is a capability
/// boundary of *this* executor, not a dense-backend failure: the host replays
/// it, so the caller needs the typed device-transform error to decide to stay
/// on the host rather than a `Dense` error it cannot classify.
fn unsupported_layout(error: DenseError) -> OperationError {
    match error {
        DenseError::Unsupported { .. } => OperationError::UnsupportedDeviceTreeTransform {
            message: "device tree transform cannot write this destination layout",
        },
        DenseError::RankMismatch { .. } => OperationError::UnsupportedDeviceTreeTransform {
            message: "device tree transform requires equal layout and stride ranks",
        },
        error => OperationError::Dense(error),
    }
}

/// How the caller's scale reaches one submission.
///
/// Two variants rather than one scalar because a zero scale is not expressible
/// as a descriptor alpha without changing what the submission computes.
#[derive(Clone, Copy)]
enum CallerScale<D> {
    /// Descriptor alpha: the caller's scale where it is non-zero, and `1` for
    /// the packs and the recoupling GEMM, which the host leaves unscaled.
    Descriptor(D),
    /// A zero caller scale: descriptor `1` and the context zero template as the
    /// 1x1 operand, so `0 * src` is computed rather than skipped.
    ZeroOperand,
}

/// The caller scale of the submission writing `entry`'s destination block:
/// `alpha * θ_b` where `scales` lists that block. A zero caller scale stays the
/// zero operand, whose product with any θ is the same `0 * x`; a non-zero one
/// times θ is non-zero, which the replay checked before any device work.
fn destination_scaled<D, C>(
    scale: CallerScale<D>,
    scales: &[(usize, C)],
    entry: &PreparedMove,
) -> CallerScale<D>
where
    D: RecouplingCoefficientAction<C>,
    C: Copy,
{
    let CallerScale::Descriptor(alpha) = scale else {
        return scale;
    };
    match scales.binary_search_by_key(&entry.destination.offset(), |&(offset, _)| offset) {
        Ok(index) => CallerScale::Descriptor(alpha.scale_by_coefficient(scales[index].1)),
        Err(_) => scale,
    }
}

/// Submits one prepared move, taking the 1x1 coefficient operand from the
/// structure's uploaded vector or, where the host copies unscaled, from the
/// context's shared `1` — or, for a zero caller scale, from its zero template.
#[allow(clippy::too_many_arguments)]
fn submit_move<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    coefficients: Option<&CudaDenseStorage>,
    entry: &PreparedMove,
    conjugate: bool,
    scale: CallerScale<D>,
    beta: CudaRegionBeta,
    dst: &mut CudaDenseStorage,
) -> Result<(), OperationError> {
    let (alpha, coefficient) = match scale {
        CallerScale::ZeroOperand => (D::ONE, CudaRegionCoefficient::Zero),
        CallerScale::Descriptor(alpha) => {
            let coefficient = match entry.coefficient {
                Some(index) => {
                    let Some(coefficients) = coefficients else {
                        return Err(OperationError::InvalidArgument {
                            message: "device tree transform block has no uploaded coefficient",
                        });
                    };
                    CudaRegionCoefficient::Buffer(coefficients, index)
                }
                None => CudaRegionCoefficient::One,
            };
            (alpha, coefficient)
        }
    };
    cuda_region_axpby::<D>(
        ctx,
        src,
        &entry.source,
        conjugate,
        alpha,
        coefficient,
        beta,
        dst,
        &entry.destination,
    )
    .map_err(OperationError::Dense)
}

/// Admission facts of one device buffer.
///
/// Placement is read from the buffer itself, so a buffer resident on another
/// device is rejected by Stage A rather than by the first primitive call —
/// after the coefficient upload.
///
/// The allocation identity is the address of the storage handle. That is a
/// sound identity for the duration of the call: `CudaDenseStorage` is neither
/// `Clone` nor refcounted and owns its device allocation exclusively, so two
/// live handles at the same address cannot exist, and the snapshot never
/// outlives the borrows it is taken from.
///
/// Context is the executor's: a device buffer carries no context tag, and the
/// device ordinal is what distinguishes the resources a wrong context would
/// reach. Tenferro re-checks the operand's device at submission either way.
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
        placement: TensorStorage::<D>::placement(storage),
        context: executor.context,
        region,
    }
}

/// Describes the device workspace and the uploaded recoupling matrices for
/// Stage C, exactly as they now exist.
///
/// The whole resident workspace buffer is the slot's capacity: Stage C compares
/// it with what this structure requires, so reporting the high-water mark a
/// wider structure grew it to is the honest number. The coefficient slot is the
/// whole uploaded payload, of which the recoupling matrices are the part Stage C
/// counts.
fn workspace_snapshot<D: CudaScalar>(
    prepared: &PreparedStructure,
    scratch: Option<&DeviceWorkspace>,
    executor: ExecutorSnapshot,
) -> WorkspaceSnapshot {
    let slot = |storage: Option<&CudaDenseStorage>| {
        storage.map_or_else(
            || empty_workspace_snapshot(executor),
            |storage| device_slot_snapshot::<D>(storage, executor),
        )
    };
    WorkspaceSnapshot {
        packed_source: slot(scratch.map(|scratch| &scratch.source)),
        packed_destination: slot(scratch.map(|scratch| &scratch.destination)),
        converted_coefficients: if prepared.recoupling_len == 0 {
            empty_workspace_snapshot(executor)
        } else {
            slot(prepared.coefficients.as_ref())
        },
        fused_index_capacity: 0,
        fused_index_placement: executor.placement,
        coefficient_readiness: Some(prepared.readiness.clone()),
    }
}

/// Admission facts of a device buffer this executor owns: a transform workspace
/// buffer, or the uploaded coefficient payload.
fn device_slot_snapshot<D: CudaScalar>(
    storage: &CudaDenseStorage,
    executor: ExecutorSnapshot,
) -> StorageSnapshot {
    let capacity = storage.len();
    StorageSnapshot {
        active_len: capacity,
        usable_capacity: capacity,
        placement: executor.placement,
        context: executor.context,
        region: if capacity == 0 {
            StorageRegion::Empty
        } else {
            StorageRegion::Bytes {
                domain: StorageDomain(storage.device() as u64),
                allocation: AllocationIdentity(std::ptr::from_ref(storage) as u64),
                start: 0,
                len: capacity.saturating_mul(core::mem::size_of::<D>()),
            }
        },
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
