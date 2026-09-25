use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet_core::{BlockKey, BlockSpec, BlockStructure, FusionTreePairKey};
use tenet_dense::DefaultDenseExecutor;
use tenet_operations::{
    try_tree_transform_structure_overwrite_owned_raw, TreeTransformBlockSpec,
    TreeTransformStructure, TreeTransformWorkspace,
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

fn canonical_structure() -> Arc<BlockStructure> {
    let key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap(),
    );
    Arc::new(
        BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::with_key(key, vec![8, 8], vec![1, 8], 0).unwrap()],
        )
        .unwrap(),
    )
}

#[test]
fn warm_owned_transform_allocates_only_the_output_payload() {
    // What: after plan/region caches are warm, the uninitialized writer makes
    // one allocation for the returned Vec and no pre-zero scratch allocation.
    let structure = canonical_structure();
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::single(0, 0, 1.0)],
    )
    .unwrap();
    let source = vec![2.0; 64];
    let mut dense = DefaultDenseExecutor::new();
    let mut workspace = TreeTransformWorkspace::default();
    let warm = try_tree_transform_structure_overwrite_owned_raw(
        &mut dense,
        &mut workspace,
        &transform,
        &structure,
        &structure,
        1,
        &source,
        1.0,
        1,
    )
    .unwrap()
    .unwrap();
    drop(warm);

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let output = try_tree_transform_structure_overwrite_owned_raw(
        &mut dense,
        &mut workspace,
        &transform,
        &structure,
        &structure,
        1,
        &source,
        1.0,
        1,
    )
    .unwrap()
    .unwrap();
    COUNTING.set(false);

    assert_eq!(ALLOCATIONS.get(), 1);
    assert_eq!(output, source);
}

#[test]
fn warm_complex_recoupling_with_real_matrix_allocates_only_view_metadata() {
    // What: a complex payload recoupled by a real matrix (#1407) runs its
    // componentwise real GEMM on the reused pack buffers, real coefficient
    // scratch and executor job buffer; the result is `U` applied to each
    // component (dyadic values, so exact in every summation order).
    //
    // The only warm allocations are the layout metadata of Tenferro 0.7.1's
    // `as_real_view{,_mut}` (three small Vecs per reinterpreted operand, two
    // operands per GEMM call), independent of block count and size. The
    // promoted complex GEMM this replaced allocated none; zero needs an
    // allocation-free reinterpretation from Tenferro.
    use num_complex::Complex64;
    use tenet_operations::{
        tree_transform_structure_with_structural_recoupling_raw, StridedHostKernelAdapter,
    };
    let structure =
        Arc::new(BlockStructure::packed_column_major(1, [vec![3], vec![3], vec![2]]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[
            TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![0.5, -0.25, 2.0, 0.75]),
            TreeTransformBlockSpec::single(2, 2, -1.0),
        ],
    )
    .unwrap();
    let source = (0..8)
        .map(|i| Complex64::new(f64::from(i) + 1.0, -f64::from(i) * 0.5))
        .collect::<Vec<_>>();
    let mut destination = vec![Complex64::default(); 8];
    let mut kernels = StridedHostKernelAdapter::default();
    let mut dense = DefaultDenseExecutor::new();
    let mut workspace = TreeTransformWorkspace::default();
    let mut replay = |destination: &mut [Complex64]| {
        tree_transform_structure_with_structural_recoupling_raw(
            &mut kernels,
            &mut dense,
            &mut workspace,
            &transform,
            &structure,
            &structure,
            destination,
            &source,
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            1,
        )
        .unwrap();
    };
    replay(&mut destination);

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    replay(&mut destination);
    COUNTING.set(false);

    assert!(ALLOCATIONS.get() <= 6, "{} allocations", ALLOCATIONS.get());
    // U[dst, src] row-major: dst0 = 0.5 s0 - 0.25 s1, dst1 = 2 s0 + 0.75 s1.
    let expected = (0..3)
        .map(|i| 0.5 * source[i] - 0.25 * source[3 + i])
        .chain((0..3).map(|i| 2.0 * source[i] + 0.75 * source[3 + i]))
        .chain(source[6..].iter().map(|&value| -value))
        .collect::<Vec<_>>();
    assert_eq!(destination, expected);
}
