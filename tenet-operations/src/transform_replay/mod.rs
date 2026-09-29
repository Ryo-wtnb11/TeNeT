use core::mem::MaybeUninit;
use core::ops::{Add, Mul};
use std::sync::{Arc, Weak};

#[cfg(test)]
use allocation_oracle::join as pool_join;
#[cfg(not(test))]
use rayon::join as pool_join;

use num_traits::{One, Zero};
use tenet_core::{
    BlockStructure, BlockView, BlockViewMut, HostReadableStorage, HostWritableStorage, Placement,
    ScratchStorage, SimilarStorage, TensorMap,
};
#[cfg(test)]
use tenet_dense::DefaultDenseExecutor;
use tenet_dense::{strided_batch_runs_into, DenseExecutor, DenseGemmBatchJob};

use crate::host_scratch::HostScratchBuffer;
use crate::kernel_adapter::for_each_fused_span;
use crate::owned_overwrite_buffer::initialize_owned;
use crate::scalar::scale_value;
use crate::storage_scratch::{StorageTreeTransformWorkspace, TreeTransformScratchBuffers};
use crate::strided::offset_to_isize;
use crate::task_view::TreeTransformTaskView;
use crate::tensoradd::{TensorAddDescriptor, TensorAddDescriptorTerm};
use crate::transform_structure::{
    TreeTransformPackReplay, TreeTransformParallelSchedule, TreeTransformScatterGroupReplay,
    TreeTransformScatterReplay, TreeTransformSingleReplay,
};
use crate::{
    tensoradd_raw_strided_kernel, tensoradd_raw_strided_kernel_trusted, BakedFusedLayout,
    ConjugateValue, DenseRecouplingScalar, HostAllocator, HostKernelAdapter, OperationError,
    RecouplingCoefficientAction, ReportsPlacement, TensorAddStructure, TransformScale,
    TreeTransformBlock, TreeTransformLayout, TreeTransformLayoutTable, TreeTransformReplayProfile,
    TreeTransformStructure,
};

mod batched;
mod coefficients;
mod entry;
mod kernels;
mod member;
mod owned;

use batched::*;
use coefficients::*;
pub use entry::*;
pub use kernels::*;
pub use member::*;
pub use owned::*;

#[cfg(test)]
mod allocation_oracle {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    thread_local! {
        static MEASURED: Cell<bool> = const { Cell::new(false) };
        static SESSION_ACTIVE: Cell<bool> = const { Cell::new(false) };
    }

    static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
    static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
    static JOINS: AtomicUsize = AtomicUsize::new(0);
    static SESSION: Mutex<()> = Mutex::new(());

    struct CountingAllocator;

    #[allow(unsafe_code)]
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let pointer = unsafe { System.alloc(layout) };
            record(pointer, layout.size());
            pointer
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let pointer = unsafe { System.alloc_zeroed(layout) };
            record(pointer, layout.size());
            pointer
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let pointer = unsafe { System.realloc(pointer, layout, new_size) };
            record(pointer, new_size);
            pointer
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    fn record(pointer: *mut u8, bytes: usize) {
        if !pointer.is_null() && is_measured() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    struct RestoreMeasurement(bool);

    struct RestoreSession(bool);

    impl Drop for RestoreMeasurement {
        fn drop(&mut self) {
            MEASURED.with(|measured| measured.set(self.0));
        }
    }

    impl Drop for RestoreSession {
        fn drop(&mut self) {
            SESSION_ACTIVE.with(|active| active.set(self.0));
        }
    }

    pub(super) fn is_measured() -> bool {
        MEASURED.with(Cell::get)
    }

    pub(super) fn with_session<R>(action: impl FnOnce() -> R) -> (R, usize) {
        assert!(rayon::current_thread_index().is_none());
        assert!(!is_measured());
        assert!(!SESSION_ACTIVE.with(Cell::get));
        let restore = RestoreSession(SESSION_ACTIVE.with(|active| active.replace(true)));
        let session = SESSION
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ALLOCATIONS.store(0, Ordering::Relaxed);
        ALLOCATED_BYTES.store(0, Ordering::Relaxed);
        JOINS.store(0, Ordering::Relaxed);
        let result = action();
        let allocations = ALLOCATIONS.load(Ordering::Relaxed);
        drop(session);
        drop(restore);
        (result, allocations)
    }

    pub(super) fn allocated_bytes() -> usize {
        ALLOCATED_BYTES.load(Ordering::Relaxed)
    }

    pub(super) fn with_measurement<R>(action: impl FnOnce() -> R) -> R {
        let restore = RestoreMeasurement(MEASURED.with(|measured| measured.replace(true)));
        let result = action();
        drop(restore);
        result
    }

    pub(super) fn join_entries() -> usize {
        JOINS.load(Ordering::Relaxed)
    }

    pub(super) fn join<A, B, RA, RB>(left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let measured = is_measured();
        if measured {
            JOINS.fetch_add(1, Ordering::Relaxed);
        }
        let restore = RestoreMeasurement(MEASURED.with(|state| state.replace(false)));
        let result = rayon::join(
            || {
                let restore = RestoreMeasurement(MEASURED.with(|state| state.replace(measured)));
                let result = left();
                drop(restore);
                result
            },
            || {
                let restore = RestoreMeasurement(MEASURED.with(|state| state.replace(measured)));
                let result = right();
                drop(restore);
                result
            },
        );
        drop(restore);
        result
    }
}

#[derive(Clone, Copy)]
enum DestinationMode<D> {
    Axpby(D),
    // Why not use Axpby(D::zero()): IEEE arithmetic still reads NaN destination
    // values, whereas assignment APIs promise destination-independent output.
    Overwrite,
}

struct PhysicalOverwriteProof<'a, C> {
    structure: &'a TreeTransformStructure<C>,
    dst_structure: &'a Arc<BlockStructure>,
    required_len: usize,
    nout: usize,
}

impl<'a, C: Copy> PhysicalOverwriteProof<'a, C> {
    fn new(
        structure: &'a TreeTransformStructure<C>,
        dst_structure: &'a Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        src_len: usize,
        nout: usize,
    ) -> Result<Option<Self>, OperationError> {
        structure.validate_replay_structures(dst_structure, src_structure)?;
        validate_replay_storage_len(src_structure, src_len)?;
        let required_len = dst_structure.required_len()?;
        if structure.physical_overwrite_len() != Some(required_len) || nout > dst_structure.rank() {
            return Ok(None);
        }
        let Some(regions) = dst_structure.coupled_sector_regions(nout)? else {
            return Ok(None);
        };
        let mut next = 0usize;
        for region in regions.iter() {
            let range = region.range();
            if range.start != next || range.end > required_len {
                return Ok(None);
            }
            next = range.end;
        }
        if next != required_len {
            return Ok(None);
        }
        Ok(Some(Self {
            structure,
            dst_structure,
            required_len,
            nout,
        }))
    }
}

/// Host scratch/replay workspace backed by `Vec<T>`.
///
/// Raw replay methods using this workspace operate on host slices. Device
/// execution should use a separate device workspace instead of hiding device
/// storage behind this type.
#[derive(Clone, Debug)]
pub struct HostTreeTransformWorkspace<T> {
    zero_strides: Vec<isize>,
    packed: TreeTransformScratchBuffers<HostScratchBuffer<T>, HostScratchBuffer<T>>,
    // Recoupling matrices converted into the data scalar type and packed in
    // recoupling_plan().entries() execution order. This order is layout
    // dependent and supplies the GEMM RHS offsets; it is not canonical
    // categorical U storage (TensorKit's basistransform buffer).
    coefficient_scratch: Vec<T>,
    // Identity of the structure whose layout-ordered RHS pack is installed.
    coefficient_structure_identity: Option<Weak<()>>,
    chunk_jobs: Vec<DenseGemmBatchJob>,
    chunk_runs: Vec<usize>,
    chunk_scatter_groups: Vec<usize>,
    fused_indices: Vec<usize>,
    member_shape: Vec<usize>,
    member_dst_strides: Vec<isize>,
    member_src_strides: Vec<isize>,
    member_ranges: Vec<(usize, usize)>,
}

pub type TreeTransformWorkspace<T> = HostTreeTransformWorkspace<T>;

impl<T> Default for HostTreeTransformWorkspace<T> {
    fn default() -> Self {
        Self {
            zero_strides: Vec::new(),
            packed: TreeTransformScratchBuffers::default(),
            coefficient_scratch: Vec::new(),
            coefficient_structure_identity: None,
            chunk_jobs: Vec::new(),
            chunk_runs: Vec::new(),
            chunk_scatter_groups: Vec::new(),
            fused_indices: Vec::new(),
            member_shape: Vec::new(),
            member_dst_strides: Vec::new(),
            member_src_strides: Vec::new(),
            member_ranges: Vec::new(),
        }
    }
}

impl<T> HostTreeTransformWorkspace<T> {
    /// Host capacity retained by member replay and ordinary transform scratch.
    #[doc(hidden)]
    pub fn retained_bytes(&self) -> usize {
        let bytes = |capacity: usize, item: usize| capacity.saturating_mul(item);
        bytes(
            self.packed.source().capacity() + self.packed.destination().capacity(),
            std::mem::size_of::<T>(),
        ) + bytes(
            self.coefficient_scratch.capacity(),
            std::mem::size_of::<T>(),
        ) + bytes(
            self.chunk_jobs.capacity(),
            std::mem::size_of::<DenseGemmBatchJob>(),
        ) + bytes(
            self.chunk_runs.capacity()
                + self.chunk_scatter_groups.capacity()
                + self.fused_indices.capacity()
                + self.member_shape.capacity(),
            std::mem::size_of::<usize>(),
        ) + bytes(
            self.zero_strides.capacity()
                + self.member_dst_strides.capacity()
                + self.member_src_strides.capacity(),
            std::mem::size_of::<isize>(),
        ) + bytes(
            self.member_ranges.capacity(),
            std::mem::size_of::<(usize, usize)>(),
        )
    }

    #[inline]
    pub fn placement(&self) -> Placement {
        Placement::Host
    }

    #[inline]
    pub fn is_host_workspace(&self) -> bool {
        self.placement() == Placement::Host
    }

    pub fn source_len(&self) -> usize {
        self.packed.source().len()
    }

    pub fn destination_len(&self) -> usize {
        self.packed.destination().len()
    }

    #[cfg(test)]
    fn packed_capacities(&self) -> (usize, usize) {
        (
            self.packed.source().capacity(),
            self.packed.destination().capacity(),
        )
    }

    fn prepare_packed_buffers(&mut self, source_len: usize, destination_len: usize, zero: T)
    where
        T: Clone,
    {
        self.packed
            .source_mut()
            .resize_filled(source_len, zero.clone());
        self.packed
            .destination_mut()
            .resize_filled(destination_len, zero);
    }

    fn prepare_fused_indices(
        &mut self,
        threads: usize,
        max_fused_rank: usize,
    ) -> Result<(), OperationError> {
        let len = checked_fused_index_len(threads, max_fused_rank)?;
        if len > self.fused_indices.len() {
            self.fused_indices.resize(len, 0);
        }
        Ok(())
    }
}

// Why not reserve here: public drivers call this before beta mutation, while
// executors retain ownership of actual workspace growth and profiler attribution.
fn checked_fused_index_len(threads: usize, max_fused_rank: usize) -> Result<usize, OperationError> {
    let len = threads
        .max(1)
        .checked_mul(max_fused_rank)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    core::alloc::Layout::array::<usize>(len).map_err(|_| OperationError::ElementCountOverflow)?;
    Ok(len)
}

impl<T> ReportsPlacement for HostTreeTransformWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

#[cfg(test)]
mod allocation_oracle_tests {
    use super::{allocation_oracle, replay_join};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn allocation_oracle_inherits_and_restores_join_scopes() {
        assert!(!allocation_oracle::is_measured());
        let (_, allocations) = allocation_oracle::with_session(|| {
            allocation_oracle::with_measurement(|| {
                assert!(allocation_oracle::is_measured());
                allocation_oracle::join(
                    || assert!(allocation_oracle::is_measured()),
                    || assert!(allocation_oracle::is_measured()),
                );
                assert!(allocation_oracle::is_measured());
            })
        });
        assert_eq!(allocations, 0);
        assert!(!allocation_oracle::is_measured());
    }

    #[test]
    fn allocation_oracle_leaves_inactive_joins_unmeasured() {
        allocation_oracle::join(
            || assert!(!allocation_oracle::is_measured()),
            || assert!(!allocation_oracle::is_measured()),
        );
        assert!(!allocation_oracle::is_measured());
    }

    #[test]
    fn allocation_oracle_restores_nested_worker_scopes() {
        let (_, allocations) = allocation_oracle::with_session(|| {
            allocation_oracle::with_measurement(|| {
                allocation_oracle::join(
                    || {
                        allocation_oracle::join(
                            || assert!(allocation_oracle::is_measured()),
                            || assert!(allocation_oracle::is_measured()),
                        );
                        assert!(allocation_oracle::is_measured());
                    },
                    || assert!(allocation_oracle::is_measured()),
                );
                assert!(allocation_oracle::is_measured());
            })
        });
        assert_eq!(allocations, 0);
        assert!(!allocation_oracle::is_measured());
    }

    #[test]
    fn allocation_oracle_recovers_after_a_panicking_session() {
        let panic = catch_unwind(AssertUnwindSafe(|| {
            allocation_oracle::with_session(|| {
                allocation_oracle::with_measurement(|| panic!("expected oracle panic"))
            });
        }));
        assert!(panic.is_err());
        assert!(!allocation_oracle::is_measured());

        let (_, allocations) = allocation_oracle::with_session(|| {
            allocation_oracle::with_measurement(|| assert!(allocation_oracle::is_measured()))
        });
        assert_eq!(allocations, 0);
        assert!(!allocation_oracle::is_measured());
    }

    #[test]
    fn allocation_oracle_rejects_nested_sessions() {
        let nested = catch_unwind(AssertUnwindSafe(|| {
            allocation_oracle::with_session(|| allocation_oracle::with_session(|| ()))
        }));
        assert!(nested.is_err());

        let (_, allocations) = allocation_oracle::with_session(|| ());
        assert_eq!(allocations, 0);
    }

    #[test]
    fn allocation_oracle_rejects_worker_sessions() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let worker_session = pool
            .install(|| catch_unwind(AssertUnwindSafe(|| allocation_oracle::with_session(|| ()))));
        assert!(worker_session.is_err());

        let (_, allocations) = allocation_oracle::with_session(|| ());
        assert_eq!(allocations, 0);
    }

    #[test]
    fn allocation_oracle_restores_after_a_panicking_replay_join() {
        let panic = catch_unwind(AssertUnwindSafe(|| {
            allocation_oracle::with_session(|| {
                allocation_oracle::with_measurement(|| {
                    replay_join(
                        || panic!("expected replay join panic"),
                        || assert!(allocation_oracle::is_measured()),
                    );
                })
            });
        }));
        assert!(panic.is_err());
        assert!(!allocation_oracle::is_measured());

        let (_, allocations) = allocation_oracle::with_session(|| {
            allocation_oracle::with_measurement(|| assert!(allocation_oracle::is_measured()))
        });
        assert_eq!(allocations, 0);
    }
}
