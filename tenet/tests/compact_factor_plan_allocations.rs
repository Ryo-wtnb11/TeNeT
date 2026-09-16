use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, TensorMap};

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

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let value = operation();
    COUNTING.set(false);
    (value, ALLOCATIONS.get())
}

/// Rank-4 U(1) tensor with five coupled sectors (charges -2..=2) and
/// degeneracy 2 per leg sector, so every factor region table is nontrivial.
fn tensor(runtime: &Runtime) -> TensorMap<U1FusionRule, f64> {
    let space = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    TensorMap::rand_with_seed(runtime, [&space, &space], [&space, &space], 1216).unwrap()
}

// Allocation-call upper bounds on the SECOND compact factorization of one
// tensor (first call warms the structure caches; the first call's factors are
// kept alive so both factor structures stay live), single dense thread.
// Everything outside `compact_factor_plan` (dense kernels, output storage,
// spectra) is unchanged between the two figures, so the difference is the
// plan's per-call constant. Measured with the pinned faer provider:
//   svd_compact: 153 before #1216, 144 after (plan ≈15 -> 6 allocations);
//   qr_compact:  120 before,        111 after;
//   eigh_full:   129 before,        120 after.
#[test]
fn second_compact_factorization_builds_the_plan_with_a_bounded_constant() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let a = tensor(&runtime);
    let warm_svd = a.svd_compact().unwrap();
    let ((u, s, vh), svd_calls) = measured(|| a.svd_compact().unwrap());
    black_box((&u, &s, &vh));
    assert!(
        svd_calls <= 144,
        "second svd_compact allocated {svd_calls} times"
    );
    drop(warm_svd);

    let warm_qr = a.qr_compact().unwrap();
    let ((q, r), qr_calls) = measured(|| a.qr_compact().unwrap());
    black_box((&q, &r));
    assert!(
        qr_calls <= 111,
        "second qr_compact allocated {qr_calls} times"
    );
    drop(warm_qr);

    let hermitian = a.adjoint().unwrap().compose(&a).unwrap();
    let warm_eigh = hermitian.eigh_full().unwrap();
    let ((v, d), eigh_calls) = measured(|| hermitian.eigh_full().unwrap());
    black_box((&v, &d));
    assert!(
        eigh_calls <= 120,
        "second eigh_full allocated {eigh_calls} times"
    );
    drop(warm_eigh);
}
