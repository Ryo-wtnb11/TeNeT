//! Allocation gate for the multiplicity-free tensor-product executor.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, TensorMap};

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

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATED.set(ALLOCATED.get() + new_size as u64);
            if new_size == PAYLOAD_BYTES.get() {
                PAYLOAD_ALLOCATIONS.set(PAYLOAD_ALLOCATIONS.get() + 1);
            }
        }
        pointer
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

fn leg(rule: &Arc<U1FusionRule>, degeneracy: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_shared(Arc::clone(rule), [(U1Irrep::new(0), degeneracy)]).unwrap()
}

#[test]
fn typed_otimes_allocates_one_output_payload_and_no_dense_kron_temporary() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let lhs_codomain = leg(&rule, 24);
    let lhs_domain = leg(&rule, 20);
    let rhs_codomain = leg(&rule, 18);
    let rhs_domain = leg(&rule, 16);
    let lhs =
        TensorMap::from_block_fn(&runtime, [&lhs_codomain], [&lhs_domain], |_, _| 2.0).unwrap();
    let rhs =
        TensorMap::from_block_fn(&runtime, [&rhs_codomain], [&rhs_domain], |_, _| 3.0).unwrap();
    let warm = lhs.otimes(&rhs).unwrap();
    let output_bytes = warm.data().len() * std::mem::size_of::<f64>();
    drop(warm);

    let (allocated, output_sized_allocations) =
        measured(output_bytes, || lhs.otimes(&rhs).unwrap());

    assert_eq!(output_sized_allocations, 1);
    assert!(
        allocated <= output_bytes as u64 + 128 * 1024,
        "otimes allocated {allocated} B for a {output_bytes} B output"
    );
}
