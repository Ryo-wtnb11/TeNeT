use super::*;

/// The device-local state one CUDA device operation may need: the dense
/// context every device kernel is submitted through, the tree-transform
/// executor that holds the per-structure device state (uploaded coefficient
/// vectors, pack/scatter workspaces), and the general contraction's operand
/// and core-destination scratch.
///
/// The executor lives here rather than beside the host caches because a replay
/// needs `&mut CudaDenseContext` and `&mut CudaTreeTransformExecutor` at the
/// same instant, and because its entries are keyed by the device context's own
/// identity: they are meaningless without it. Putting both behind one mutex is
/// what lets a device transform take exactly one lock.
pub(crate) struct CudaDeviceState {
    pub(super) dense: symtn_dense::CudaDenseContext,
    pub(super) tree_transform: symtn_operations::CudaTreeTransformExecutor,
    /// Grow-only execution scratch, bounded by the largest contraction run
    /// and observable through [`Runtime::cuda_contract_scratch_bytes`]. Like
    /// the executor's workspaces it is not charged to
    /// `symtn-network`'s `PlanCacheConfig::workspace_budget_bytes`, which admits idle network
    /// workspaces and has no eviction protocol for a Runtime singleton.
    pub(super) contract_scratch: symtn_tensors::CudaContractScratch,
}

/// A Runtime's device state and the lock of the device it lives on.
///
/// The per-Runtime mutex is kept although the device lock already excludes
/// every other lease on it: it is what hands out `&mut CudaDeviceState`
/// safely, and it is never contended, so it costs one uncontended atomic
/// pair. Replacing it by an `UnsafeCell` guarded by the device lock would buy
/// nothing measurable for an `unsafe` invariant.
pub(super) struct CudaHome {
    pub(super) device_lock: &'static Mutex<()>,
    pub(super) state: Mutex<CudaDeviceState>,
}

/// The process-wide lock of CUDA device `ordinal`, shared by every Runtime on
/// that device.
///
/// Why a lock across Runtimes (#1384): CubeCL's client is process-wide per
/// device and records a binding's stream cursor only when it is bound, not
/// when a later kernel writes it (tensor4all/cubecl#16). A stale read of an
/// output `O` needs some other submission that syncs a stream past `O`'s bind
/// cursor between that bind and `O`'s last write. For a fresh output — bound
/// and fully written inside one lease, as in every operation that returns a
/// new tensor — this lock excludes every other lease on the device, so no
/// symtn submission can fall in that window.
/// Any later reader's sync to the writer's stream then either happened before
/// the writer's lease (its synced cursor is below `O`'s bind cursor, so it
/// waits on a fresh event) or after it (the event already covers every write
/// into `O`). No host synchronization is added: the lock orders enqueue only.
/// A buffer bound in one lease and written in a later one
/// (`*_into` destinations, `CudaContractScratch`, pooled
/// network intermediates) spans two leases and is ordered instead by the
/// single CubeCL stream `CudaDenseContext::new` pins (#1391).
///
/// One leaked entry per ordinal that built a device context, so the registry
/// is bounded by the device count. The lock guards no data, so a poison left
/// by a panicking holder is ignored.
pub(super) fn cuda_device_lock(ordinal: usize) -> &'static Mutex<()> {
    static LOCKS: Mutex<std::collections::BTreeMap<usize, &'static Mutex<()>>> =
        Mutex::new(std::collections::BTreeMap::new());
    LOCKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(ordinal)
        .or_insert_with(|| Box::leak(Box::new(Mutex::new(()))))
}

pub(super) fn lock_device(lock: &'static Mutex<()>) -> MutexGuard<'static, ()> {
    lock.lock().unwrap_or_else(PoisonError::into_inner)
}

/// RAII lease of this runtime's single CUDA context, held only for the device
/// portion of one device operation. A device context cannot be pooled: device
/// tensors are bound to one backend instance, so the guard is the lease.
///
/// It dereferences to the [`symtn_dense::CudaDenseContext`], so every device
/// operation that needs nothing else reads exactly as it did before the
/// tree-transform executor moved in beside it; a transform takes both halves
/// through [`Self::split`].
///
/// It holds the device's process-wide lock ([`cuda_device_lock`]) first and
/// this Runtime's state second, so every lease on one device, from any
/// Runtime, excludes every other. That protects what is bound and fully
/// written within one lease; a buffer written again in a later lease is not
/// covered (see [`cuda_device_lock`]). A host sync under the lease stalls
/// every Runtime on the device.
///
/// Nothing reached while this lease is held may lease again — on this Runtime
/// or on any other Runtime of the same device: neither mutex is re-entrant.
/// Host-side planning — structure compilation, cache lookup, space
/// derivation — is finished, and the host context lease dropped, before this
/// one is taken, and the code run under it only sees the dense context and
/// executor, never a `Runtime`.
pub(crate) struct CudaLease<'a> {
    // Declared first so it is released before the device lock.
    pub(super) state: MutexGuard<'a, CudaDeviceState>,
    pub(super) _device: MutexGuard<'static, ()>,
}

impl CudaLease<'_> {
    /// Borrows the dense context and the tree-transform executor at once, which
    /// a replay needs because the executor submits its work through the
    /// context.
    pub(crate) fn split(
        &mut self,
    ) -> (
        &mut symtn_dense::CudaDenseContext,
        &mut symtn_operations::CudaTreeTransformExecutor,
    ) {
        let state = &mut *self.state;
        (&mut state.dense, &mut state.tree_transform)
    }

    /// [`Self::split`] plus the contraction scratch, which a general device
    /// contraction needs beside the transform executor it replays through.
    pub(crate) fn split_contract(
        &mut self,
    ) -> (
        &mut symtn_dense::CudaDenseContext,
        &mut symtn_operations::CudaTreeTransformExecutor,
        &mut symtn_tensors::CudaContractScratch,
    ) {
        let state = &mut *self.state;
        (
            &mut state.dense,
            &mut state.tree_transform,
            &mut state.contract_scratch,
        )
    }
}

impl std::ops::Deref for CudaLease<'_> {
    type Target = symtn_dense::CudaDenseContext;

    fn deref(&self) -> &Self::Target {
        &self.state.dense
    }
}

impl std::ops::DerefMut for CudaLease<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state.dense
    }
}

/// Device bytes this Runtime's tree-transform executor and its device context
/// hold for reuse, each counted once.
///
/// Read-only: nothing here is a knob. `executor_bytes` is what the executor
/// retains (uploaded coefficient and recoupling vectors plus the pack/scatter
/// workspaces) and is released by [`Runtime::clear_cuda_caches`].
/// `context_scalar_operand_bytes` is
/// the device context's own ones and zero templates, which many device
/// operations share and which the executor does not own; the two are reported
/// separately so neither is charged twice. Neither is charged to
/// `symtn-network`'s `PlanCacheConfig::workspace_budget_bytes`, which admits idle *network*
/// workspaces and has no eviction protocol for a Runtime singleton.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaTreeTransformStats {
    /// Structures with device state currently prepared.
    pub prepared_structures: usize,
    /// Device bytes the executor retains: coefficient vectors and workspaces.
    pub executor_bytes: usize,
    /// Of `executor_bytes`, what the pack/scatter workspaces hold.
    pub workspace_bytes: usize,
    /// Device bytes the shared context scalar operands hold, owned by the
    /// device context rather than by the executor.
    pub context_scalar_operand_bytes: usize,
    /// Distinct cuTENSOR operand signatures the prepared structures submit.
    pub required_plan_entries: usize,
}

/// The one error a device operation reports when its runtime has no device.
pub(super) fn missing_cuda_device() -> Error {
    Error::InvalidArgument(
        "this runtime was built without a CUDA device; use \
         Runtime::builder().cuda(device)"
            .to_string(),
    )
}
