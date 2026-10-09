//! The eager device contraction scratch: the buffers
//! [`execute_storage_contract_resolution_on_cuda`](crate::execute_storage_contract_resolution_on_cuda)
//! materializes transformed operands and a core destination into. The stage
//! sequence that uses them is `contract/route_cuda.rs`.

use std::any::{Any, TypeId};

use tenet_dense::{CudaDenseContext, CudaRegion, CudaScalar};
use tenet_operations::cuda::CudaStorage;

use crate::OperationError;

/// Device scratch a general contraction materializes its transformed operands
/// and its core result into: two source buffers and one core-destination
/// buffer per (payload dtype, device context), grown monotonically.
///
/// It is execution scratch, not a semantic cache. A source buffer is written
/// in overwrite mode before it is read (every active block assigned, every
/// inactive layout zeroed), so it never needs a reset; the core-destination
/// buffer has its active blocks written by the core GEMMs and exactly the
/// plan's inactive destination blocks zeroed after them, on every call.
/// Dropping it changes nothing but the cost of
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
    pub(in crate::contract) entries: Vec<ScratchEntry>,
    pub(in crate::contract) zero_regions: Vec<CudaRegion>,
}

pub(in crate::contract) struct ScratchEntry {
    scalar: TypeId,
    context: u64,
    bytes: usize,
    /// `ScratchBuffers<D>` for the entry's `scalar`: a device buffer carries
    /// its payload dtype, so one entry per dtype.
    buffers: Box<dyn Any + Send>,
}

pub(in crate::contract) struct ScratchBuffers<D: CudaScalar> {
    pub(in crate::contract) lhs: Option<CudaStorage<D>>,
    pub(in crate::contract) rhs: Option<CudaStorage<D>>,
    pub(in crate::contract) dst: Option<CudaStorage<D>>,
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

pub(in crate::contract) fn scratch_entry<D: CudaScalar + 'static>(
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
pub(in crate::contract) fn grow<'a, D: CudaScalar>(
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
