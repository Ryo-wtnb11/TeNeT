//! #1546 performance contract: allocation budget of a borrowed view operand.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::{Arc, Mutex};
use tenet::typed::ContractSpec;

use num_complex::Complex64;
use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, TensorMap};

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
    static LIVE_BYTES: Cell<i64> = const { Cell::new(0) };
    static PEAK_BYTES: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            BYTES.set(BYTES.get() + layout.size() as u64);
            let live = LIVE_BYTES.get() + layout.size() as i64;
            LIVE_BYTES.set(live);
            PEAK_BYTES.set(PEAK_BYTES.get().max(live.max(0) as u64));
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if ENABLED.get() {
            LIVE_BYTES.set(LIVE_BYTES.get() - layout.size() as i64);
        }
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            BYTES.set(BYTES.get() + new_size as u64);
            let live = LIVE_BYTES.get() - layout.size() as i64 + new_size as i64;
            LIVE_BYTES.set(live);
            PEAK_BYTES.set(PEAK_BYTES.get().max(live.max(0) as u64));
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

fn measure(f: impl FnOnce()) -> (u64, u64) {
    ALLOCATIONS.set(0);
    BYTES.set(0);
    LIVE_BYTES.set(0);
    PEAK_BYTES.set(0);
    ENABLED.set(true);
    f();
    ENABLED.set(false);
    (ALLOCATIONS.get(), BYTES.get())
}

fn tensor(runtime: &Runtime, degeneracy: usize, seed: u64) -> TensorMap<U1FusionRule, Complex64> {
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (-2..=2).map(|charge| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    TensorMap::rand_with_seed(runtime, [&space, &space], [&space], seed).unwrap()
}

/// What: passing `t.adjoint_view()` costs exactly what building the owned
/// lazy adjoint `&t.adjoint()?` and passing it costs (one adjoint header and
/// its logical layout), and that cost does not grow with degeneracy: no
/// payload is copied for the view.
#[test]
fn adjoint_view_operand_costs_the_owned_lazy_adjoint_header() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut view_costs = Vec::new();
    for degeneracy in [1, 8] {
        let b = tensor(&runtime, degeneracy, 1);
        let a = tensor(&runtime, degeneracy, 2)
            .adjoint()
            .unwrap()
            .materialize()
            .unwrap();
        black_box(a.inner(b.adjoint_view()).unwrap());
        let view = measure(|| {
            black_box(a.inner(b.adjoint_view()).unwrap());
        });
        let owned = measure(|| {
            black_box(a.inner(&b.adjoint().unwrap()).unwrap());
        });
        assert_eq!(view, owned, "degeneracy {degeneracy}: inner");
        view_costs.push(view);

        black_box(
            a.contract(
                b.adjoint_view(),
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2, 3],
                },
            )
            .unwrap(),
        );
        let contract_view = measure(|| {
            black_box(
                a.contract(
                    b.adjoint_view(),
                    &ContractSpec {
                        lhs: &[2],
                        rhs: &[0],
                        codomain: &[0, 1],
                        domain: &[2, 3],
                    },
                )
                .unwrap(),
            );
        });
        let contract_owned = measure(|| {
            black_box(
                a.contract(
                    &b.adjoint().unwrap(),
                    &ContractSpec {
                        lhs: &[2],
                        rhs: &[0],
                        codomain: &[0, 1],
                        domain: &[2, 3],
                    },
                )
                .unwrap(),
            );
        });
        assert_eq!(
            contract_view, contract_owned,
            "degeneracy {degeneracy}: contract"
        );
    }
    assert_eq!(
        view_costs[0], view_costs[1],
        "view cost grew with the payload"
    );
}

/// What: a plain view `&t` allocates nothing beyond the operation itself.
#[test]
fn plain_view_operand_allocates_nothing_extra() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let a = tensor(&runtime, 4, 3);
    black_box(a.inner(&a).unwrap());
    let cost = measure(|| {
        black_box(a.inner(&a).unwrap());
    });
    assert_eq!(cost, (0, 0));
}
