use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet_core::{BlockStructure, SectorId};
use tenet_operations::{
    FusionBlockContractGroupPlan, FusionBlockContractPlan, FusionBlockContractWorkspace,
    FusionBlockMatrixGroup, FusionStridedBlockLayout, FusionSubblockMatrixLayout, OperationError,
    Rank2Gemm, StridedHostKernelAdapter,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            ALLOCATIONS.with(|count| count.set(count.get() + 1));
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

struct ScalarGemm;

impl Rank2Gemm<f64> for ScalarGemm {
    fn matmul_rank2(
        &mut self,
        dst: &mut [f64],
        lhs: &[f64],
        rhs: &[f64],
        _rows: usize,
        _contracted: usize,
        _cols: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<(), OperationError> {
        dst[0] = alpha * lhs[0] * rhs[0] + beta * dst[0];
        Ok(())
    }
}

fn scalar_group() -> FusionBlockMatrixGroup {
    FusionBlockMatrixGroup {
        coupled: SectorId::new(0),
        rows: 1,
        cols: 1,
        needs_clear: false,
        direct_offset: None,
        block_indices: vec![0],
        subblocks: vec![FusionSubblockMatrixLayout {
            block: FusionStridedBlockLayout {
                shape: vec![1],
                strides: vec![1],
                offset: 0,
            },
            matrix_offset: 0,
            matrix_strides: vec![1],
            coefficient: 1.0,
        }],
    }
}

#[test]
fn warmed_irregular_group_replay_allocates_nothing() {
    let structure = Arc::new(BlockStructure::trivial(&[1]).unwrap());
    let plan = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![
            FusionBlockContractGroupPlan::new(scalar_group(), scalar_group(), scalar_group())
                .unwrap(),
        ],
    )
    .unwrap();
    let mut kernels = StridedHostKernelAdapter::default();
    let mut gemm = ScalarGemm;
    let mut workspace = FusionBlockContractWorkspace::default();
    let lhs = [2.0];
    let rhs = [3.0];
    let mut dst = [5.0];

    for _ in 0..2 {
        plan.execute_raw(
            &mut kernels,
            &mut gemm,
            &mut workspace,
            &structure,
            &mut dst,
            &structure,
            &lhs,
            &structure,
            &rhs,
            1.0,
            0.0,
        )
        .unwrap();
    }

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    plan.execute_raw(
        &mut kernels,
        &mut gemm,
        &mut workspace,
        &structure,
        &mut dst,
        &structure,
        &lhs,
        &structure,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    COUNTING.set(false);

    assert_eq!(ALLOCATIONS.get(), 0);
    assert_eq!(dst, [6.0]);
}
