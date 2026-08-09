use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{GradedSpace, Runtime, TensorMap};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, size) };
        if !out.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        out
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn tensor(runtime: &Runtime, n: usize, value: f64) -> TensorMap<U1FusionRule, f64> {
    let rule = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new_with_shared_provider(rule, [(U1Irrep::new(0), n)]).unwrap();
    TensorMap::from_block_fn(runtime, [&space], [&space], move |_, ij| {
        if ij[0] == ij[1] {
            value
        } else {
            0.0
        }
    })
    .unwrap()
}

#[test]
fn scale_assign_mutates_unique_dense_payload() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut value = tensor(&runtime, 3, 2.0);
    value.scale_assign(3.0);
    assert_eq!(value.data(), &[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0]);
}

#[test]
fn add_assign_preserves_clone_and_updates_unique_receiver() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut lhs = tensor(&runtime, 2, 2.0);
    let original = lhs.clone();
    let rhs = tensor(&runtime, 2, 3.0);
    lhs.add_assign(&rhs, 2.0, -1.0).unwrap();
    assert_eq!(lhs.data(), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(original.data(), &[2.0, 0.0, 0.0, 2.0]);
}

#[test]
fn add_assign_rejects_layout_mismatch_without_mutating_destination() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut lhs = tensor(&runtime, 2, 2.0);
    let rhs = tensor(&runtime, 3, 3.0);
    let error = lhs.add_assign(&rhs, 1.0, 1.0).unwrap_err();
    assert!(error.to_string().contains("different spaces"));
    assert_eq!(lhs.data(), &[2.0, 0.0, 0.0, 2.0]);
}

#[test]
fn unique_dense_assign_is_allocation_free() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut lhs = tensor(&runtime, 64, 2.0);
    let rhs = tensor(&runtime, 64, 3.0);
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    lhs.add_assign(&rhs, 2.0, -1.0).unwrap();
    COUNTING.set(false);
    assert_eq!(ALLOCATIONS.get(), 0);
    let mut scaled = tensor(&runtime, 64, 2.0);
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    scaled.scale_assign(3.0);
    COUNTING.set(false);
    assert_eq!(ALLOCATIONS.get(), 0);
}
