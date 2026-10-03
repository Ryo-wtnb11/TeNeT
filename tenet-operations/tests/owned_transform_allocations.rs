use std::sync::Arc;

use tenet_core::{BlockKey, BlockSpec, BlockStructure, FusionTreePairKey};
use tenet_dense::DefaultDenseExecutor;
use tenet_operations::{
    try_tree_transform_structure_overwrite_owned_raw, TreeTransformBlockSpec,
    TreeTransformStructure, TreeTransformWorkspace,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

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
    counting_alloc::start();
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
    let allocs = counting_alloc::stop();

    assert_eq!(allocs.calls, 1);
    assert_eq!(output, source);
}
