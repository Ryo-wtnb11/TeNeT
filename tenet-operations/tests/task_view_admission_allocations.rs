use std::sync::Arc;

use tenet_core::BlockStructure;
use tenet_operations::{
    DenseTreeTransformOperations, TreeTransformBackend, TreeTransformReplayProfile,
    TreeTransformStructure, TreeTransformWorkspace,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[test]
fn warmed_public_host_admission_without_tasks_is_allocation_free() {
    // An empty completed transform takes the normal profiled Host admission
    // seam but has no scale, block, coefficient, or dense numerical work.
    let structure =
        Arc::new(BlockStructure::packed_column_major(1, std::iter::empty::<Vec<usize>>()).unwrap());
    let transform =
        TreeTransformStructure::<f64>::compile_structures(&structure, &structure, &[]).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    let mut destination = [];
    let source = [];

    let mut replay = |profile: &mut TreeTransformReplayProfile| {
        backend
            .tree_transform_structure_overwrite_into_raw_profiled(
                &mut workspace,
                &transform,
                &structure,
                &structure,
                &mut destination,
                &source,
                1.0,
                &[],
                profile,
            )
            .unwrap();
    };
    replay(&mut TreeTransformReplayProfile::default());
    counting_alloc::start();
    let mut profile = TreeTransformReplayProfile::default();
    for _ in 0..100 {
        replay(&mut profile);
    }
    let allocs = counting_alloc::stop();

    assert_eq!(allocs.calls, 0);
    assert_eq!(profile.single_blocks, 0);
    assert_eq!(profile.multi_blocks, 0);
    assert_eq!(profile.multi_coefficient_prepare, std::time::Duration::ZERO);
}
