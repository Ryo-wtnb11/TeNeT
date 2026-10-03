use tenet_core::BlockStructure;
use tenet_operations::{TreeTransformBlockSpec, TreeTransformStructure};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[test]
fn overwrite_proof_allocation_is_bounded_by_layout_metadata() {
    let logical_elements = 1_000_000;
    let structure = BlockStructure::packed_column_major(1, [vec![logical_elements]]).unwrap();
    let specs = [TreeTransformBlockSpec::single(0, 0, 1.0_f64)];

    counting_alloc::start();
    let compiled = TreeTransformStructure::compile_structures(&structure, &structure, &specs);
    let allocs = counting_alloc::stop();

    // What: compile-time overwrite proof memory scales with block/layout
    // metadata, not with the number of physical scalar destinations.
    assert_eq!(compiled.unwrap().block_count(), 1);
    assert!(allocs.bytes < 64 * 1024, "bytes={}", allocs.bytes);
}

#[test]
fn permuted_layout_compile_avoids_per_block_metadata_scratch() {
    let fixture = |blocks| {
        let shapes = (0..blocks).map(|_| vec![2, 2, 1, 1]).collect::<Vec<_>>();
        let structure = BlockStructure::packed_column_major(4, shapes).unwrap();
        let specs = (0..blocks)
            .map(|block| {
                TreeTransformBlockSpec::single(block, block, 1.0_f64).with_source_axes([1, 0, 2, 3])
            })
            .collect::<Vec<_>>();
        (structure, specs)
    };
    let (small_structure, small_specs) = fixture(8);
    let (large_structure, large_specs) = fixture(64);
    let _ = TreeTransformStructure::compile_structures(
        &large_structure,
        &large_structure,
        &large_specs,
    )
    .unwrap();
    counting_alloc::start();
    let small = TreeTransformStructure::compile_structures(
        &small_structure,
        &small_structure,
        &small_specs,
    )
    .unwrap();
    let allocs = counting_alloc::stop();
    let small_allocations = allocs.calls as usize;
    counting_alloc::start();
    let large = TreeTransformStructure::compile_structures(
        &large_structure,
        &large_structure,
        &large_specs,
    )
    .unwrap();
    let allocs = counting_alloc::stop();
    let large_allocations = allocs.calls as usize;

    // What: increasing block count eightfold grows final metadata but does not
    // introduce one scratch allocation per block.
    assert_eq!(small.block_count(), 8);
    assert_eq!(large.block_count(), 64);
    assert!(
        large_allocations < small_allocations * 2,
        "small={small_allocations} large={large_allocations}"
    );
}
