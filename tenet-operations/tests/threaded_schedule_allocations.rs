use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use tenet_core::BlockStructure;
use tenet_dense::DefaultDenseExecutor;
use tenet_operations::{
    tree_transform_structure_with_structural_recoupling_raw, StridedHostKernelAdapter,
    TreeTransformBlockSpec, TreeTransformStructure, TreeTransformWorkspace,
};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn warm_threaded_replay_does_not_allocate_schedule_storage() {
    let block_structure = Arc::new(
        BlockStructure::packed_column_major(1, [vec![4], vec![4], vec![4], vec![4]])
            .unwrap(),
    );
    let structure = TreeTransformStructure::compile_structures(
        &block_structure,
        &block_structure,
        &[
            TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            TreeTransformBlockSpec::single(2, 2, 1.0),
            TreeTransformBlockSpec::single(3, 3, -1.0),
        ],
    )
    .unwrap();
    let src = (0..16).map(|value| value as f64 + 1.0).collect::<Vec<_>>();
    let mut dst = vec![0.0; 16];
    let mut kernels = StridedHostKernelAdapter::default();
    let mut dense = DefaultDenseExecutor::new();
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
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Release);
    replay();
    COUNTING.store(false, Ordering::Release);

    assert_eq!(ALLOCATIONS.load(Ordering::Acquire), 0);
}
