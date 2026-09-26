//! Warm Host `PreparedCompose` allocation contract (#1498): after one call at
//! a fixed `B`, `execute` and `execute_into` allocate nothing of TeNeT's on
//! the caller thread. The one remaining allocation is Tenferro 0.7.1's
//! grouped-GEMM validation (`tenferro-tensor src/backend.rs:
//! validate_grouped_gemm`), which every grouped submission pays, eager
//! included. Its own binary, because it installs a counting global allocator.

mod common;
#[path = "../../tests/support/numerics.rs"]
mod numerics;
mod prepared;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[allow(unused_imports)]
use num_complex::{Complex32, Complex64};
use tenet::typed::{PreparedCompose, Runtime, StackedTensorMap};

use prepared::{filled, members, u1_legs};

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

fn allocations(f: impl FnOnce()) -> usize {
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    f();
    COUNTING.set(false);
    ALLOCATIONS.get()
}

#[test]
fn warm_host_calls_allocate_only_the_backend_grouped_validation() {
    // What: at a fixed B, a second `execute` and a second `execute_into`
    // (whose plan has inactive destination blocks to zero-fill) reuse the
    // handle's output, job list and fill strides, and the Runtime's pooled
    // context. The zero fill adds nothing (`execute_into` == `execute`), and
    // both stay within the one per-submission allocation of Tenferro's
    // grouped validator. The cold call's count is the control that shows the
    // counter sees this thread's allocations.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (v, w) = u1_legs();
    let count = 5;
    let lhs =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], count, 1)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], count, 2)).unwrap();
    let mut dst = StackedTensorMap::pack(&filled::<_, f64>(
        &runtime,
        &[&v, &v],
        &[&v],
        count,
        f64::NAN,
    ))
    .unwrap();
    let mut handle = PreparedCompose::new(&lhs, &rhs).unwrap();

    let cold = allocations(|| {
        handle.execute(&lhs, &rhs).unwrap();
    });
    assert!(cold > 0, "the cold call allocates its output and job list");
    handle.execute_into(&lhs, &rhs, &mut dst).unwrap();

    let warm = allocations(|| {
        handle.execute(&lhs, &rhs).unwrap();
    });
    let warm_into = allocations(|| {
        handle.execute_into(&lhs, &rhs, &mut dst).unwrap();
    });
    assert!(warm <= 1, "warm execute allocations: {warm}");
    assert_eq!(
        warm_into, warm,
        "the zero fill of execute_into allocates nothing"
    );
    assert!(cold > warm + 2, "cold {cold} against warm {warm}");
}
