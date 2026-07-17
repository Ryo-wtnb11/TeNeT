use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenet::prelude::*;

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

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn cached_permute_overwrite_does_not_allocate_on_the_caller_thread() {
    // What: a warmed multiplicity-free non-Abelian permutation reuses its
    // compiled plan and replay workspace without allocating on the caller.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::fz2_u1_su2([
        ((0, 0, 0), 8),
        ((0, 1, 1), 6),
        ((1, -1, 1), 6),
        ((1, 0, 2), 4),
    ])
    .unwrap();
    let source =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space, &space], [&space], 197).unwrap();
    let mut destination = source.permute(&[1], &[2, 0]).unwrap();
    let mut context = TensorExecutionContext::for_runtime(&runtime).unwrap();
    let mut cache = PermuteOverwriteCache::default();

    assert_eq!(
        context
            .try_permute_overwrite_into(
                &mut cache,
                &mut destination,
                &source,
                &[1],
                &[2, 0],
                Scalar::F64(1.0),
            )
            .unwrap(),
        OverwriteOutcome::Written
    );

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let outcome = context
        .try_permute_overwrite_into(
            &mut cache,
            &mut destination,
            &source,
            &[1],
            &[2, 0],
            Scalar::F64(1.0),
        )
        .unwrap();
    COUNTING.set(false);
    black_box(destination.data());

    assert_eq!(outcome, OverwriteOutcome::Written);
    assert_eq!(ALLOCATIONS.get(), 0);
}

#[test]
fn cached_owned_rank_nine_permute_does_not_clone_operation_storage() {
    // What: the warmed public owned path pays for its returned tensor but does
    // not add a rank-spilled operation clone before consulting the plan cache.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::su2([(0, 1), (1, 1)]);
    let source = Tensor::rand_with_seed(
        &runtime,
        Dtype::F64,
        [&space, &space, &space, &space, &space],
        [&space, &space, &space, &space],
        226,
    )
    .unwrap();
    let codomain = [1, 0, 2, 3, 4];
    let domain = [5, 6, 7, 8];
    drop(source.permute(&codomain, &domain).unwrap());

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let output = source.permute(&codomain, &domain).unwrap();
    COUNTING.set(false);
    black_box(output.data());

    assert_eq!(ALLOCATIONS.get(), 3);
}
