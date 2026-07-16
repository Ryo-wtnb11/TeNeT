use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenet::prelude::*;

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            BYTES.set(BYTES.get() + layout.size() as u64);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            BYTES.set(BYTES.get() + new_size as u64);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measure(f: impl FnOnce()) -> (u64, u64) {
    ALLOCATIONS.set(0);
    BYTES.set(0);
    ENABLED.set(true);
    f();
    ENABLED.set(false);
    (ALLOCATIONS.get(), BYTES.get())
}

fn tensor(
    runtime: &Runtime,
    sectors: impl IntoIterator<Item = (i32, usize)>,
    rank: usize,
) -> Tensor {
    assert_eq!(rank % 2, 0);
    let space = Space::u1(sectors);
    Tensor::rand_with_seed(
        runtime,
        Dtype::C64,
        std::iter::repeat_n(&space, rank / 2),
        std::iter::repeat_n(&space, rank / 2),
        261 + rank as u64,
    )
    .unwrap()
}

#[test]
fn adjoint_creation_cost_is_independent_of_block_count() {
    // What: lazy adjoint creation owns one fixed-size view, never a fusion-tree grid.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut reference = None;
    for (rank, radius) in [(2, 0), (2, 2), (2, 6), (2, 12), (4, 4), (6, 1), (8, 0)] {
        let source = tensor(&runtime, (-radius..=radius).map(|charge| (charge, 2)), rank);
        tenet_tensors::reset_global_operation_caches();
        let cost = measure(|| {
            black_box(source.adjoint().unwrap());
        });
        let reference = *reference.get_or_insert(cost);
        assert_eq!(cost, reference, "rank={rank}, sector radius={radius}");
    }
    assert_eq!(reference.unwrap().0, 1, "adjoint allocates one shared view");
}

#[test]
fn adjoint_involution_does_not_allocate() {
    // What: the second dagger restores the parent body without allocating.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let source = tensor(&runtime, (-4..=4).map(|charge| (charge, 2)), 4);
    let adjoint = source.adjoint().unwrap();

    let cost = measure(|| {
        black_box(adjoint.adjoint().unwrap());
    });

    assert_eq!(cost, (0, 0));
}

#[test]
fn ordinary_tensor_clone_does_not_allocate() {
    // What: the representation split keeps an owned tensor's value-like Arc clone cost.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let source = tensor(&runtime, (-4..=4).map(|charge| (charge, 2)), 4);

    let cost = measure(|| {
        black_box(source.clone());
    });

    assert_eq!(cost, (0, 0));
}
