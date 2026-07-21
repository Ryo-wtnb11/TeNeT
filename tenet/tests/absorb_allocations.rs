use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenet::prelude::*;

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATED: Cell<u64> = const { Cell::new(0) };
    static PAYLOAD_BYTES: Cell<usize> = const { Cell::new(0) };
    static PAYLOAD_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATED.set(ALLOCATED.get() + layout.size() as u64);
            if layout.size() == PAYLOAD_BYTES.get() {
                PAYLOAD_ALLOCATIONS.set(PAYLOAD_ALLOCATIONS.get() + 1);
            }
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured<T>(payload_bytes: usize, operation: impl FnOnce() -> T) -> (u64, usize) {
    ALLOCATED.set(0);
    PAYLOAD_BYTES.set(payload_bytes);
    PAYLOAD_ALLOCATIONS.set(0);
    ENABLED.set(true);
    let output = black_box(operation());
    ENABLED.set(false);
    black_box(output);
    (ALLOCATED.get(), PAYLOAD_ALLOCATIONS.get())
}

#[test]
fn ordinary_absorb_allocates_only_one_destination_sized_payload() {
    // What: block matching and prefix traversal allocate no source, key,
    // minimum-shape, descriptor, or per-block heap storage.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let destination_space = Space::u1((-8..=8).map(|charge| (charge, 5)));
    let source_space = Space::u1((-6..=10).map(|charge| (charge, 3)));
    let destination = Tensor::rand_with_seed(
        &runtime,
        Dtype::F64,
        [&destination_space, &destination_space],
        [&destination_space, &destination_space],
        395_020,
    )
    .unwrap();
    let source = Tensor::rand_with_seed(
        &runtime,
        Dtype::F64,
        [&source_space, &source_space],
        [&source_space, &source_space],
        395_021,
    )
    .unwrap();
    black_box(destination.absorb(&source).unwrap());
    let payload_bytes = std::mem::size_of_val(destination.data());

    let (allocated, payload_allocations) =
        measured(payload_bytes, || destination.absorb(&source).unwrap());

    assert_eq!(payload_allocations, 1);
    assert!(
        allocated <= payload_bytes as u64 + 1024,
        "absorb allocated {allocated} B for a {payload_bytes} B destination payload"
    );
}
