use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet_core::BlockStructure;
use tenet_dense::{
    DenseDotConfig, DenseError, DenseExecutor, DenseGemmBatchJob, DenseRead, DenseScalar,
    DenseTensor, DenseWrite,
};
use tenet_operations::{
    tree_transform_structure_overwrite_with_structural_recoupling_raw,
    tree_transform_structure_with_structural_recoupling_raw, StridedHostKernelAdapter,
    TreeTransformBlockSpec, TreeTransformStructure, TreeTransformWorkspace,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(ptr, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Default)]
struct NoAllocDenseExecutor;

impl DenseExecutor for NoAllocDenseExecutor {
    fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("tree replay does not call SVD")
    }

    fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("tree replay does not call QR")
    }

    fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("tree replay does not call EIGH")
    }

    fn dot_general_into(
        &mut self,
        _output: DenseWrite<'_>,
        _lhs: DenseRead<'_>,
        _rhs: DenseRead<'_>,
        _config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        unreachable!("tree replay uses the batched matmul entry point")
    }

    fn matmul_batch_axpby_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        _runs: &[usize],
        _alpha: DenseScalar,
        _beta: DenseScalar,
    ) -> Result<(), DenseError> {
        let (mut output, lhs, rhs) = match (output, lhs, rhs) {
            (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                (output, lhs, rhs)
            }
            _ => unreachable!("allocation oracle uses f64"),
        };
        let output_offset = output.offset();
        let lhs_offset = lhs.offset();
        let rhs_offset = rhs.offset();
        let output_data = output.data_mut();
        for job in jobs {
            for col in 0..job.cols {
                for row in 0..job.rows {
                    let mut value = 0.0;
                    for contracted in 0..job.contracted {
                        value += lhs.data()
                            [lhs_offset + job.lhs_offset + row + contracted * job.rows]
                            * rhs.data()
                                [rhs_offset + job.rhs_offset + contracted + col * job.contracted];
                    }
                    output_data[output_offset + job.dst_offset + row + col * job.rows] = value;
                }
            }
        }
        Ok(())
    }
}

#[test]
fn warm_threaded_replay_does_not_allocate_schedule_storage() {
    let block_structure = Arc::new(
        BlockStructure::packed_column_major(1, [vec![4], vec![4], vec![4], vec![4]]).unwrap(),
    );
    let structure = TreeTransformStructure::compile_structures(
        &block_structure,
        &block_structure,
        &[
            TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]),
            TreeTransformBlockSpec::single(2, 2, 1.0),
            TreeTransformBlockSpec::single(3, 3, -1.0),
        ],
    )
    .unwrap();
    let src = (0..16).map(|value| value as f64 + 1.0).collect::<Vec<_>>();
    let mut dst = vec![0.0; 16];
    let mut kernels = StridedHostKernelAdapter::default();
    let mut dense = NoAllocDenseExecutor;
    let mut workspace = TreeTransformWorkspace::default();

    let mut replay = || {
        tree_transform_structure_with_structural_recoupling_raw(
            &mut kernels,
            &mut dense,
            &mut workspace,
            &structure,
            &block_structure,
            &block_structure,
            &mut dst,
            &src,
            1.0,
            0.0,
            4,
        )
        .unwrap();
    };

    replay();
    // What: the operation-neutral threaded schedule performs no caller-thread
    // allocation after warmup. Worker/backend allocations are intentionally
    // outside this oracle; the removed schedule Vecs lived on this thread.
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    replay();
    COUNTING.set(false);

    assert_eq!(ALLOCATIONS.get(), 0);
}

#[test]
fn warm_threaded_overwrite_replay_does_not_allocate_on_the_caller_thread() {
    let block_structure = Arc::new(
        BlockStructure::packed_column_major(1, [vec![4], vec![4], vec![4], vec![4]]).unwrap(),
    );
    let structure = TreeTransformStructure::compile_structures(
        &block_structure,
        &block_structure,
        &[
            TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]),
            TreeTransformBlockSpec::single(2, 2, 1.0),
            TreeTransformBlockSpec::single(3, 3, -1.0),
        ],
    )
    .unwrap();
    let src = (0..16).map(|value| value as f64 + 1.0).collect::<Vec<_>>();
    let mut dst = vec![f64::NAN; 16];
    let mut kernels = StridedHostKernelAdapter::default();
    let mut dense = NoAllocDenseExecutor;
    let mut workspace = TreeTransformWorkspace::default();

    let mut replay = || {
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            &mut kernels,
            &mut dense,
            &mut workspace,
            &structure,
            &block_structure,
            &block_structure,
            &mut dst,
            &src,
            1.0,
            4,
        )
        .unwrap();
    };

    replay();
    // Why not count process-wide allocations: worker/backend allocation is
    // outside this canary; the overwrite replay scratch lives on this thread.
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    replay();
    COUNTING.set(false);

    assert_eq!(ALLOCATIONS.get(), 0);
}

#[test]
fn warm_many_group_chunks_do_not_allocate_on_the_caller_thread() {
    const GROUPS: usize = 32;
    let block_structure =
        Arc::new(BlockStructure::packed_column_major(1, vec![vec![4]; 2 * GROUPS]).unwrap());
    let specs = (0..GROUPS)
        .map(|group| {
            let first = 2 * group;
            TreeTransformBlockSpec::multi(
                vec![first, first + 1],
                vec![first, first + 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )
        })
        .collect::<Vec<_>>();
    let structure =
        TreeTransformStructure::compile_structures(&block_structure, &block_structure, &specs)
            .unwrap();
    let src = (0..8 * GROUPS)
        .map(|value| value as f64 + 1.0)
        .collect::<Vec<_>>();
    let mut dst = vec![0.0; src.len()];
    let mut kernels = StridedHostKernelAdapter::default();
    let mut dense = NoAllocDenseExecutor;
    let mut workspace = TreeTransformWorkspace::default();

    let mut replay = || {
        tree_transform_structure_with_structural_recoupling_raw(
            &mut kernels,
            &mut dense,
            &mut workspace,
            &structure,
            &block_structure,
            &block_structure,
            &mut dst,
            &src,
            1.0,
            0.0,
            3,
        )
        .unwrap();
    };

    replay();
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    replay();
    COUNTING.set(false);

    // What: reusable chunk metadata and numerical scratch allocate only during
    // warmup, even when the plan has more groups than the concurrency bound.
    assert_eq!(
        ALLOCATIONS.get(),
        0,
        "warm many-group replay allocated on the caller thread"
    );
    assert_eq!(dst, src);
}
