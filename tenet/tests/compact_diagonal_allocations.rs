use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use tenet::prelude::*;

struct CountingAllocator;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if ENABLED.load(Ordering::Relaxed) && !pointer.is_null() {
            ALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if ENABLED.load(Ordering::Relaxed) && !pointer.is_null() {
            ALLOCATED.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured_product_bytes(degeneracy: usize) -> u64 {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::u1([(0, degeneracy)]);
    let source = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 801).unwrap();
    let diagonal = source.svd_compact().unwrap().1;

    black_box(diagonal.compose(&diagonal).unwrap());
    ALLOCATED.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Release);
    let output = black_box(diagonal.compose(&diagonal).unwrap());
    ENABLED.store(false, Ordering::Release);
    black_box(output);
    ALLOCATED.load(Ordering::Relaxed)
}

/// A compact diagonal product stores one value per bond basis state. Comparing
/// two sizes makes the gate insensitive to fixed cache/metadata allocations
/// while rejecting the old dense d-by-d materialization.
#[test]
fn diagonal_product_allocation_bytes_scale_linearly() {
    let small = measured_product_bytes(32);
    let large = measured_product_bytes(256);
    assert!(
        large <= small * 16,
        "allocation growth is not O(d): d=32 used {small} bytes, d=256 used {large} bytes"
    );
    assert!(
        large < (256 * 256 * std::mem::size_of::<f64>()) as u64,
        "compact product allocated at least one dense payload: {large} bytes"
    );
}
