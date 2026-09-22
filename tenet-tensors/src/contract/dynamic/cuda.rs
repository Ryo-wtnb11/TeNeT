//! Device executor of the Host-compiled storage contraction route.
//!
//! Nothing categorical happens here: the route, the orientation, every
//! transform structure, the borrow decisions and the core plan were compiled
//! on the host by
//! [`compile_storage_contract_resolution`](crate::TensorContractFusionExecutionContext::compile_storage_contract_resolution).
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
use tenet_dense::{cuda_region_zero, CudaDenseContext, CudaRegion, CudaScalar};
use tenet_operations::cuda::{CudaStorage, CudaStorageGemm};
use tenet_operations::{CudaTreeTransformDestination, CudaTreeTransformExecutor};

use super::{DynamicTreeExecutionArtifact, FusionContractOrientation};
use crate::contract::resolution::{StorageContractResolution, StorageContractRoute};
use crate::{DenseBlockScalar, OperationError, RecouplingCoefficientAction};

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
/// `dst`, overwriting it.
///
/// `dst_is_zeroed` says whether the caller provides `dst` zero-filled (the
/// returning path's fresh #740 output). When it does not (a retained
/// destination, `contract_overwrite_into`), the blocks the core GEMMs write
/// directly are completed by zeroing exactly the core plan's inactive blocks;
/// an output transform in overwrite mode writes every element itself, so it
/// needs nothing. The host clears the whole destination instead.
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
/// The fermionic core-right twist (the physical rhs under `LhsRhs`, the
/// physical lhs under `RhsLhs`) is folded into that operand's source
/// transform: the move writing core-right block `b` runs with descriptor
/// alpha `θ_b` (`replay_with_destination_scales`), where the host scales the
/// materialized operand in place afterwards — the same values in one pass
/// fewer. A twisted operand is never borrowed, so the transform always runs.
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
    dst_is_zeroed: bool,
    lhs: &CudaStorage<D>,
    rhs: &CudaStorage<D>,
) -> Result<(), OperationError>
where
    D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
    C: DenseBlockScalar,
{
    match &resolution.route {
        StorageContractRoute::Core(plan) => {
            // Converted before the first submission, like every other check.
            let regions = if dst_is_zeroed {
                &[][..]
            } else {
                fill_inactive_regions(&mut scratch.zero_regions, plan)?
            };
            plan.execute_direct_on_storage_prezeroed(
                &mut CudaStorageGemm::new(ctx),
                dst,
                lhs,
                rhs,
            )?;
            zero_regions(ctx, dst, regions)
        }
        StorageContractRoute::DynamicTree(artifact) => execute_dynamic_tree_on_cuda(
            ctx,
            transforms,
            scratch,
            artifact,
            dst_structure,
            dst,
            dst_is_zeroed,
            lhs,
            rhs,
        ),
    }
}

/// Zeroes `regions` of `dst`. A plan's inactive blocks are disjoint from
/// every block its GEMMs write, so the fill may follow them; it runs after,
/// so the plan's own range validation still precedes every write to `dst`.
fn zero_regions<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    regions: &[CudaRegion],
) -> Result<(), OperationError> {
    for region in regions {
        cuda_region_zero::<D>(ctx, &mut dst.0, region).map_err(OperationError::Dense)?;
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
    dst_is_zeroed: bool,
    lhs: &CudaStorage<D>,
    rhs: &CudaStorage<D>,
) -> Result<(), OperationError>
where
    D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
    C: DenseBlockScalar,
{
    let reverse = artifact.orientation == FusionContractOrientation::RhsLhs;
    let scales = artifact.core_right_destination_scales();
    let core_right_borrowed = if reverse {
        artifact.lhs_borrowed
    } else {
        artifact.rhs_borrowed
    };
    if core_right_borrowed && !scales.is_empty() {
        return Err(OperationError::InvalidArgument {
            message: "device contraction artifact borrows its twisted core-right operand",
        });
    }
    let lhs_len = artifact.lhs_transform.space.required_len()?;
    let rhs_len = artifact.rhs_transform.space.required_len()?;
    let core_dst_len = artifact
        .core_dst
        .as_ref()
        .map(|core_dst| core_dst.space.required_len())
        .transpose()?;
    // The core destination is the retained scratch with an output transform,
    // otherwise `dst` itself; either way only a non-zeroed buffer needs its
    // inactive blocks zeroed.
    let CudaContractScratch {
        entries,
        zero_regions: region_scratch,
    } = scratch;
    let core_zero_regions = if artifact.core_dst.is_some() || !dst_is_zeroed {
        fill_inactive_regions(region_scratch, artifact.block_plan())?
    } else {
        &[]
    };

    let (bytes, buffers) = scratch_entry::<D>(entries, ctx.identity())?;
    let ScratchBuffers {
        lhs: lhs_slot,
        rhs: rhs_slot,
        dst: dst_slot,
    } = buffers;
    let no_scales: &[(usize, C)] = &[];
    let (lhs_scales, rhs_scales) = if reverse {
        (scales, no_scales)
    } else {
        (no_scales, scales)
    };
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
            transform.transform_structure.as_ref(),
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
    let (core_left, core_right) = if reverse {
        (physical_rhs, physical_lhs)
    } else {
        (physical_lhs, physical_rhs)
    };

    let (Some(core_dst), Some(core_dst_len)) = (artifact.core_dst.as_ref(), core_dst_len) else {
        artifact.block_plan.execute_direct_on_storage_prezeroed(
            &mut CudaStorageGemm::new(ctx),
            dst,
            core_left,
            core_right,
        )?;
        return zero_regions(ctx, dst, core_zero_regions);
    };
    let core_buffer = grow(ctx, dst_slot, bytes, core_dst_len)?;
    zero_regions(ctx, core_buffer, core_zero_regions)?;
    artifact.block_plan.execute_direct_on_storage_prezeroed(
        &mut CudaStorageGemm::new(ctx),
        core_buffer,
        core_left,
        core_right,
    )?;
    transforms.replay(
        ctx,
        core_dst.output_transform_structure.as_ref(),
        dst_structure,
        core_dst.space.structure(),
        dst,
        core_buffer,
        D::ONE,
        CudaTreeTransformDestination::Overwrite,
    )
}

fn materialized<D: CudaScalar>(
    slot: &Option<CudaStorage<D>>,
) -> Result<&CudaStorage<D>, OperationError> {
    slot.as_ref().ok_or(OperationError::InvalidArgument {
        message: "device contraction source was not materialized",
    })
}

/// Why a negatively strided inactive block is rejected: a device region's
/// strides and offset are unsigned (see [`CudaRegion`]).
fn inactive_region_unsupported() -> OperationError {
    OperationError::UnsupportedTensorContractScope {
        message: "device contraction cannot zero a negatively strided destination block",
    }
}

/// Rejects, at compile time and without building them, the core plan's
/// inactive destination blocks no device region can express.
pub(crate) fn validate_inactive_regions<C>(
    plan: &tenet_operations::FusionBlockContractPlan<C>,
) -> Result<(), OperationError>
where
    C: Copy + PartialEq + num_traits::One,
{
    for layout in plan.inactive_destination_regions() {
        if layout.block.offset < 0 || layout.block.strides.iter().any(|&stride| stride < 0) {
            return Err(inactive_region_unsupported());
        }
    }
    Ok(())
}

/// The core plan's inactive destination blocks as device regions: the exact
/// set a retained core-destination buffer must zero (the host clears the
/// whole buffer instead, `prepare_zeroed_scratch_slot`), written into
/// `regions` in place.
fn fill_inactive_regions<'a, C>(
    regions: &'a mut Vec<CudaRegion>,
    plan: &tenet_operations::FusionBlockContractPlan<C>,
) -> Result<&'a [CudaRegion], OperationError>
where
    C: Copy + PartialEq + num_traits::One,
{
    let layouts = plan.inactive_destination_regions();
    regions.truncate(layouts.len());
    for (index, layout) in layouts.iter().enumerate() {
        let block = &layout.block;
        if block.offset < 0 || block.strides.iter().any(|&stride| stride < 0) {
            return Err(inactive_region_unsupported());
        }
        // Checked non-negative just above, so the casts are exact.
        let strides = block.strides.iter().map(|&stride| stride as usize);
        let offset = block.offset as usize;
        match regions.get_mut(index) {
            Some(region) => region
                .assign(&block.shape, strides, offset)
                .map_err(OperationError::Dense)?,
            None => regions.push(
                CudaRegion::new(block.shape.clone(), strides.collect(), offset)
                    .map_err(OperationError::Dense)?,
            ),
        }
    }
    Ok(regions)
}
