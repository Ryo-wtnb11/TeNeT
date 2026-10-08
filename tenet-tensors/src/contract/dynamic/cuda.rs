//! Device executor of the Host-compiled storage contraction route.
//!
//! Nothing categorical happens here: the route, the orientation, every
//! transform structure, the borrow decisions and the core plan were compiled
//! on the host by
//! [`plan_contract`](crate::TensorContractFusionExecutionContext::plan_contract).
//! This module replays that value with the device executors — the tree
//! transform executor for the source and output transforms, the storage GEMM
//! seam for the core — which is TensorKit's `blas_contract!` dataflow
//! (tensoroperations.jl:383-455 @cfaa073: `tensoradd!` A and B into temporaries,
//! `mul!`, `tensoradd!` into C) and QSpace's `QSpace::contract`
//! (QSpace.cc:4141-4260 @dd2cc7e: `permute_to` both operands, grouped GEMMs,
//! `Permute` of the result), each permute skipped when it is the identity.

use std::any::{Any, TypeId};
use std::sync::Arc;

use tenet_core::BlockStructure;
use tenet_dense::{cuda_region_scale, CudaDenseContext, CudaRegion, CudaScalar};
use tenet_operations::cuda::{CudaStorage, CudaStorageGemm};
use tenet_operations::cuda_transform::CudaMemberZeroRegions;
use tenet_operations::{CudaTreeTransformDestination, CudaTreeTransformExecutor};

use super::DynamicTreeExecutionArtifact;
use crate::contract::resolution::{ContractRoute, CopyCRoute, StorageContractResolution};
use crate::{
    ContractDestinationInit, DenseBlockScalar, OperationError, RecouplingCoefficientAction,
};

/// Device scratch a general contraction materializes its transformed operands
/// and its core result into: two source buffers and one core-destination
/// buffer per (payload dtype, device context), grown monotonically.
///
/// It is execution scratch, not a semantic cache. A source buffer is written
/// in overwrite mode before it is read (every active block assigned, every
/// inactive layout zeroed), so it never needs a reset; the core-destination
/// buffer has exactly the plan's inactive destination blocks zeroed before
/// the core GEMMs write the rest. Dropping it changes nothing but the cost of
/// the next contraction.
///
/// Each buffer is narrowed to exactly the length the replay admits
/// (`set_active_len`) rather than reallocated, so alternating operand sizes
/// pay one allocation — one zero upload, #740 — per high-water mark only.
///
/// It also holds the host region list the replay zeroes a destination's
/// inactive core blocks through, rewritten in place from the core plan on
/// each call that zeroes.
#[derive(Default)]
pub struct CudaContractScratch {
    entries: Vec<ScratchEntry>,
    zero_regions: Vec<CudaRegion>,
}

struct ScratchEntry {
    scalar: TypeId,
    context: u64,
    bytes: usize,
    /// `ScratchBuffers<D>` for the entry's `scalar`: a device buffer carries
    /// its payload dtype, so one entry per dtype.
    buffers: Box<dyn Any + Send>,
}

struct ScratchBuffers<D: CudaScalar> {
    lhs: Option<CudaStorage<D>>,
    rhs: Option<CudaStorage<D>>,
    dst: Option<CudaStorage<D>>,
}

impl CudaContractScratch {
    /// Device bytes the scratch buffers hold (their allocations, not their
    /// active prefixes).
    pub fn device_bytes(&self) -> usize {
        self.entries
            .iter()
            .fold(0usize, |total, entry| total.saturating_add(entry.bytes))
    }

    /// Releases every scratch buffer. A memory decision, never a correctness
    /// one: the next contraction grows what it needs again.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.zero_regions = Vec::new();
    }
}

fn scratch_entry<D: CudaScalar + 'static>(
    entries: &mut Vec<ScratchEntry>,
    context: u64,
) -> Result<(&mut usize, &mut ScratchBuffers<D>), OperationError> {
    {
        let scalar = TypeId::of::<D>();
        let index = match entries
            .iter()
            .position(|entry| entry.scalar == scalar && entry.context == context)
        {
            Some(index) => index,
            None => {
                entries.push(ScratchEntry {
                    scalar,
                    context,
                    bytes: 0,
                    buffers: Box::new(ScratchBuffers::<D> {
                        lhs: None,
                        rhs: None,
                        dst: None,
                    }),
                });
                entries.len() - 1
            }
        };
        let ScratchEntry { bytes, buffers, .. } = &mut entries[index];
        let buffers =
            buffers
                .downcast_mut::<ScratchBuffers<D>>()
                .ok_or(OperationError::InvalidArgument {
                    message: "device contraction scratch entry holds another payload dtype",
                })?;
        Ok((bytes, buffers))
    }
}

#[cfg(test)]
impl CudaContractScratch {
    /// Overwrites every retained core-destination buffer of payload `D` with
    /// `value` over its whole allocation; returns the elements poisoned.
    pub(crate) fn poison_core_destination<D: CudaScalar + 'static>(
        &mut self,
        ctx: &CudaDenseContext,
        value: D,
    ) -> usize {
        let mut poisoned = 0;
        for entry in &mut self.entries {
            let Some(buffers) = entry.buffers.downcast_mut::<ScratchBuffers<D>>() else {
                continue;
            };
            if let Some(dst) = buffers.dst.as_mut() {
                let (capacity, active) = (dst.0.capacity(), dst.0.len());
                let mut fresh = CudaStorage::<D>::upload_owned(ctx, vec![value; capacity]).unwrap();
                fresh.0.set_active_len(active).unwrap();
                *dst = fresh;
                poisoned += capacity;
            }
        }
        poisoned
    }
}

/// Makes `slot` hold at least `len` elements and narrows it to exactly `len`.
fn grow<'a, D: CudaScalar>(
    ctx: &CudaDenseContext,
    slot: &'a mut Option<CudaStorage<D>>,
    bytes: &mut usize,
    len: usize,
) -> Result<&'a mut CudaStorage<D>, OperationError> {
    let capacity = slot.as_ref().map(|buffer| buffer.0.capacity());
    if capacity.is_none_or(|capacity| capacity < len) {
        // Growth costs one host-to-device transfer of zeros, the only way a
        // device buffer is made (#740). The values are irrelevant: every
        // element a replay reads is written first.
        let grown = CudaStorage::<D>::upload_owned(ctx, vec![D::ZERO; len])?;
        let size = core::mem::size_of::<D>();
        *bytes = bytes
            .saturating_sub(capacity.unwrap_or(0).saturating_mul(size))
            .saturating_add(len.saturating_mul(size));
        *slot = Some(grown);
    }
    let buffer = slot.as_mut().ok_or(OperationError::InvalidArgument {
        message: "device contraction scratch was not grown",
    })?;
    buffer
        .0
        .set_active_len(len)
        .map_err(OperationError::Dense)?;
    Ok(buffer)
}

/// Replays a Host-compiled storage contraction route on `ctx`'s device into
/// `dst`: `dst = alpha * contract(lhs, rhs) + beta * dst`.
///
/// `init` is [`ContractDestinationInit::Zeroed`] when the caller provides
/// `dst` zero-filled (the returning path's fresh #740 output), and otherwise
/// `Axpby(beta)` for a retained destination (`contract_into`). `beta` rides
/// the epilogue of whatever writes each element: the core GEMMs when they
/// write `dst` directly (their jobs carry `alpha` and `beta`), or the output
/// transform in `Axpby(beta)` mode. Only the core plan's inactive blocks of a
/// directly written `dst` have no writer; they become `beta * dst` through a
/// zero-source region move (a zero fill for `beta = 0`, nothing for `beta = 1`).
/// No pass reads or clears the whole destination.
///
/// `Core`: one storage GEMM per coupled-sector job over the parent buffers.
/// `DynamicTree`: each non-borrowed source is replayed into its scratch
/// buffer in overwrite mode (a borrowed source — an identity transform of an
/// unconjugated operand already in core layout — is read in place, as on the
/// host); the core GEMMs write straight into `dst` when the output transform
/// is the identity, and otherwise into the core-destination scratch, whose
/// inactive blocks — and only those — are zeroed first, followed by the
/// output transform into `dst` in overwrite mode.
///
/// The fermionic contraction twist is folded into the source transform of
/// the operand the artifact twists (the one already materialized, as in
/// TensorKit's `blas_contract!`): the move writing its block `b` runs with
/// descriptor alpha `θ_b` (`replay_with_destination_scales`), where the host
/// scales the materialized operand in place afterwards — the same values in
/// one pass fewer. A twisted operand is never borrowed, so the transform
/// always runs.
///
/// Every check that can reject the route runs before the first device
/// submission.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn execute_storage_contract_resolution_on_cuda<D, C>(
    ctx: &mut CudaDenseContext,
    transforms: &mut CudaTreeTransformExecutor,
    scratch: &mut CudaContractScratch,
    resolution: &StorageContractResolution<C>,
    dst_structure: &Arc<BlockStructure>,
    dst: &mut CudaStorage<D>,
    lhs: &CudaStorage<D>,
    rhs: &CudaStorage<D>,
    alpha: D,
    init: ContractDestinationInit<D>,
) -> Result<(), OperationError>
where
    D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
    C: DenseBlockScalar,
{
    resolution.admit_cuda_inactive_regions()?;
    match &resolution.route {
        ContractRoute::Core { plan, swapped } => {
            let (lhs, rhs) = if *swapped { (rhs, lhs) } else { (lhs, rhs) };
            // Converted before the first submission, like every other check.
            let regions = if direct_regions_need_beta(init) {
                CudaMemberZeroRegions::fill_single(
                    &mut scratch.zero_regions,
                    plan.inactive_destination_regions(),
                )?
            } else {
                &[][..]
            };
            let beta = active_beta(init);
            execute_core_direct(ctx, plan, dst, lhs, rhs, alpha, beta)?;
            scale_regions(ctx, dst, regions, beta)
        }
        ContractRoute::DynamicTree(artifact) => execute_dynamic_tree_on_cuda(
            ctx,
            transforms,
            scratch,
            artifact,
            dst_structure,
            dst,
            lhs,
            rhs,
            alpha,
            init,
        ),
        ContractRoute::CopyC(copy) => execute_copy_c_on_cuda(
            ctx,
            transforms,
            scratch,
            copy,
            dst_structure,
            dst,
            lhs,
            rhs,
            alpha,
            init,
        ),
    }
}

/// TensorKit `blas_contract!`'s `copyC` (`tensoroperations.jl:436-446` @cfaa073): the
/// unscaled core into the retained core-destination scratch (inactive blocks
/// zeroed), then one output transform into `dst` carrying `alpha` and `beta`
/// — TensorKit's `tensoradd!(C, C′, pAB, false, α, β)`.
#[allow(clippy::too_many_arguments)]
fn execute_copy_c_on_cuda<D, C>(
    ctx: &mut CudaDenseContext,
    transforms: &mut CudaTreeTransformExecutor,
    scratch: &mut CudaContractScratch,
    copy: &CopyCRoute<C>,
    dst_structure: &Arc<BlockStructure>,
    dst: &mut CudaStorage<D>,
    lhs: &CudaStorage<D>,
    rhs: &CudaStorage<D>,
    alpha: D,
    init: ContractDestinationInit<D>,
) -> Result<(), OperationError>
where
    D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
    C: DenseBlockScalar,
{
    let (lhs, rhs) = if copy.swapped { (rhs, lhs) } else { (lhs, rhs) };
    let CudaContractScratch {
        entries,
        zero_regions,
    } = scratch;
    let regions =
        CudaMemberZeroRegions::fill_single(zero_regions, copy.core.inactive_destination_regions())?;
    let (bytes, buffers) = scratch_entry::<D>(entries, ctx.identity())?;
    let temporary = grow(ctx, &mut buffers.dst, bytes, copy.temporary_len)?;
    scale_regions(ctx, temporary, regions, D::ZERO)?;
    copy.core.execute_direct_on_storage_prezeroed(
        &mut CudaStorageGemm::new(ctx),
        temporary,
        lhs,
        rhs,
    )?;
    transforms.replay(
        ctx,
        &copy.transform,
        dst_structure,
        &copy.temporary,
        dst,
        temporary,
        alpha,
        match init {
            ContractDestinationInit::Zeroed => CudaTreeTransformDestination::Overwrite,
            ContractDestinationInit::Axpby(beta) => CudaTreeTransformDestination::Axpby(beta),
        },
    )
}

fn active_beta<D: CudaScalar>(init: ContractDestinationInit<D>) -> D {
    match init {
        ContractDestinationInit::Zeroed => D::ZERO,
        ContractDestinationInit::Axpby(beta) => beta,
    }
}

/// The core GEMMs straight into `dst`; `alpha = 1, beta = 0` keeps the
/// unscaled overwrite GEMMs bit for bit.
fn execute_core_direct<D, C>(
    ctx: &mut CudaDenseContext,
    plan: &tenet_operations::FusionBlockContractPlan<C>,
    dst: &mut CudaStorage<D>,
    lhs: &CudaStorage<D>,
    rhs: &CudaStorage<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
    C: DenseBlockScalar,
{
    let mut gemm = CudaStorageGemm::new(ctx);
    if alpha == D::ONE && beta == D::ZERO {
        plan.execute_direct_on_storage_prezeroed(&mut gemm, dst, lhs, rhs)
    } else {
        plan.execute_direct_on_storage_axpby(&mut gemm, dst, lhs, rhs, alpha, beta)
    }
}

/// Whether a `dst` the core GEMMs write directly has inactive blocks to
/// finish: not when it is born zero, and not for `beta = 1`.
fn direct_regions_need_beta<D: CudaScalar>(init: ContractDestinationInit<D>) -> bool {
    matches!(init, ContractDestinationInit::Axpby(beta) if beta != D::ONE)
}

/// Scales `regions` of `dst` by `beta` (zeroes them for `beta = 0`). A plan's
/// inactive blocks are disjoint from every block its GEMMs write, so the pass
/// may follow them; it runs after, so the plan's own range validation still
/// precedes every write to `dst`.
fn scale_regions<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    regions: &[CudaRegion],
    beta: D,
) -> Result<(), OperationError> {
    for region in regions {
        cuda_region_scale::<D>(ctx, &mut dst.0, region, beta).map_err(OperationError::Dense)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_dynamic_tree_on_cuda<D, C>(
    ctx: &mut CudaDenseContext,
    transforms: &mut CudaTreeTransformExecutor,
    scratch: &mut CudaContractScratch,
    artifact: &DynamicTreeExecutionArtifact<C>,
    dst_structure: &Arc<BlockStructure>,
    dst: &mut CudaStorage<D>,
    lhs: &CudaStorage<D>,
    rhs: &CudaStorage<D>,
    alpha: D,
    init: ContractDestinationInit<D>,
) -> Result<(), OperationError>
where
    D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
    C: DenseBlockScalar,
{
    let lhs_len = artifact.lhs_transform.space.required_len()?;
    let rhs_len = artifact.rhs_transform.space.required_len()?;
    let core_dst_len = artifact
        .core_dst
        .as_ref()
        .map(|core_dst| core_dst.space.required_len())
        .transpose()?;
    // The core destination is the retained scratch with an output transform,
    // whose inactive blocks are zeroed; otherwise `dst` itself, whose inactive
    // blocks become `beta * dst`.
    let CudaContractScratch {
        entries,
        zero_regions: region_scratch,
    } = scratch;
    let core_zero_regions = if artifact.core_dst.is_some() || direct_regions_need_beta(init) {
        CudaMemberZeroRegions::fill_single(
            region_scratch,
            artifact.block_plan().inactive_destination_regions(),
        )?
    } else {
        &[]
    };

    let (bytes, buffers) = scratch_entry::<D>(entries, ctx.identity())?;
    let ScratchBuffers {
        lhs: lhs_slot,
        rhs: rhs_slot,
        dst: dst_slot,
    } = buffers;
    let [lhs_scales, rhs_scales] = artifact.stage_scales();
    for (borrowed, transform, slot, source, len, destination_scales) in [
        (
            artifact.lhs_borrowed,
            &artifact.lhs_transform,
            &mut *lhs_slot,
            lhs,
            lhs_len,
            lhs_scales,
        ),
        (
            artifact.rhs_borrowed,
            &artifact.rhs_transform,
            &mut *rhs_slot,
            rhs,
            rhs_len,
            rhs_scales,
        ),
    ] {
        if borrowed {
            continue;
        }
        let buffer = grow(ctx, slot, bytes, len)?;
        transforms.replay_with_destination_scales(
            ctx,
            &transform.transform_structure,
            transform.space.structure(),
            &transform.replay_structure,
            buffer,
            source,
            D::ONE,
            CudaTreeTransformDestination::Overwrite,
            destination_scales,
        )?;
    }

    let physical_lhs = if artifact.lhs_borrowed {
        lhs
    } else {
        materialized(lhs_slot)?
    };
    let physical_rhs = if artifact.rhs_borrowed {
        rhs
    } else {
        materialized(rhs_slot)?
    };
    let (core_left, core_right) = artifact.core_order(physical_lhs, physical_rhs);

    let (Some(core_dst), Some(core_dst_len)) = (artifact.core_dst.as_ref(), core_dst_len) else {
        let beta = active_beta(init);
        execute_core_direct(
            ctx,
            &artifact.block_plan,
            dst,
            core_left,
            core_right,
            alpha,
            beta,
        )?;
        return scale_regions(ctx, dst, core_zero_regions, beta);
    };
    let core_buffer = grow(ctx, dst_slot, bytes, core_dst_len)?;
    scale_regions(ctx, core_buffer, core_zero_regions, D::ZERO)?;
    artifact.block_plan.execute_direct_on_storage_prezeroed(
        &mut CudaStorageGemm::new(ctx),
        core_buffer,
        core_left,
        core_right,
    )?;
    transforms.replay(
        ctx,
        &core_dst.output_transform_structure,
        dst_structure,
        core_dst.space.structure(),
        dst,
        core_buffer,
        alpha,
        match init {
            ContractDestinationInit::Zeroed => CudaTreeTransformDestination::Overwrite,
            ContractDestinationInit::Axpby(beta) => CudaTreeTransformDestination::Axpby(beta),
        },
    )
}

fn materialized<D: CudaScalar>(
    slot: &Option<CudaStorage<D>>,
) -> Result<&CudaStorage<D>, OperationError> {
    slot.as_ref().ok_or(OperationError::InvalidArgument {
        message: "device contraction source was not materialized",
    })
}
