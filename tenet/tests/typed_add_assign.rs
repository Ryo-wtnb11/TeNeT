use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{GradedSpace, Runtime, TensorMap};
use tenet::typed::Error;

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
    let space = GradedSpace::try_new(rule, [(U1Irrep::new(0), n)]).unwrap();
    TensorMap::from_subblock_fn(runtime, [&space], [&space], move |_, ij| {
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
    assert_eq!(
        value.dense_data().unwrap(),
        &[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0]
    );
}

#[test]
fn axpby_into_updates_a_unique_destination_and_refuses_a_shared_one() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let x = tensor(&runtime, 2, 3.0);
    let mut destination = tensor(&runtime, 2, 2.0);
    x.axpby_into(&mut destination, -1.0, 2.0).unwrap();
    assert_eq!(destination.dense_data().unwrap(), &[1.0, 0.0, 0.0, 1.0]);

    // A clone shares the payload: writing it would change the clone too, and
    // replacing it would allocate behind the caller's back.
    let clone = destination.clone();
    assert_eq!(
        x.axpby_into(&mut destination, 1.0, 1.0),
        Err(Error::DestinationShared)
    );
    assert_eq!(destination.dense_data().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(clone.dense_data().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn axpby_into_rejects_a_layout_mismatch_without_mutating_the_destination() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut destination = tensor(&runtime, 2, 2.0);
    let x = tensor(&runtime, 3, 3.0);
    let error = x.axpby_into(&mut destination, 1.0, 1.0).unwrap_err();
    assert!(error.to_string().contains("different spaces"));
    assert_eq!(destination.dense_data().unwrap(), &[2.0, 0.0, 0.0, 2.0]);
}

#[test]
fn unique_dense_destinations_are_allocation_free() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut destination = tensor(&runtime, 64, 2.0);
    let x = tensor(&runtime, 64, 3.0);
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    x.axpby_into(&mut destination, -1.0, 2.0).unwrap();
    COUNTING.set(false);
    assert_eq!(ALLOCATIONS.get(), 0);
    let mut scaled = tensor(&runtime, 64, 2.0);
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    scaled.scale_assign(3.0);
    COUNTING.set(false);
    assert_eq!(ALLOCATIONS.get(), 0);
}
