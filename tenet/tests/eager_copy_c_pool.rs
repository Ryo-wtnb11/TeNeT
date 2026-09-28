//! #1626: an eager `contract(spec)` that takes TensorKit `blas_contract!`'s
//! `copyC` route (a zero-copy contraction into a temporary, then one permute
//! into the result) keeps that temporary in the Runtime's pooled execution
//! scratch, as `contract_into` keeps its own. A warm call therefore
//! allocates exactly one output-sized buffer: the returned tensor, and a warm
//! `contract_into` (#1631), which takes the same route, allocates none.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet::sector::{U1FusionRule, U1Irrep};
use tenet::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

struct CountingAllocator;

thread_local! {
    static WATCHED_BYTES: Cell<Option<usize>> = const { Cell::new(None) };
    static MATCHES: Cell<usize> = const { Cell::new(0) };
}

fn record(bytes: usize) {
    // `try_with`: the allocator also runs during thread-local teardown.
    let _ = WATCHED_BYTES.try_with(|watched| {
        if watched.get() == Some(bytes) {
            MATCHES.set(MATCHES.get() + 1);
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: forwards the caller's layout unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` came from `System` with this layout.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        // SAFETY: forwards the caller's pointer and layout unchanged.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Allocations of exactly `bytes` made while `f` runs; its result is
/// dropped after counting stops.
fn sized_allocs<T>(bytes: usize, f: impl FnOnce() -> T) -> usize {
    MATCHES.set(0);
    WATCHED_BYTES.set(Some(bytes));
    let result = f();
    WATCHED_BYTES.set(None);
    drop(result);
    MATCHES.get()
}

type Map = TensorMap<U1FusionRule, f64>;

/// `a: p ⊗ q ← c`, `b: c ← r`, distinct prime degeneracies so the output
/// payload size is unique among the call's allocations.
fn operands(runtime: &Runtime) -> (Map, Map) {
    let space = |n| GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), n)]).unwrap();
    let (p, q, c, r) = (space(13), space(17), space(19), space(23));
    (
        Map::rand_with_seed(runtime, [&p, &q], [&c], 1).unwrap(),
        Map::rand_with_seed(runtime, [&c], [&r], 2).unwrap(),
    )
}

#[test]
fn warm_split_moving_eager_contract_allocates_only_its_result() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (a, b) = operands(&runtime);
    let output_bytes = 13 * 17 * 23 * size_of::<f64>();
    // `r` joins the codomain and `q` moves to the domain: a split move, which
    // no GEMM writes directly (copyC).
    let spec = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[0, 2],
        domain: &[1],
    };
    // Cold: the result and the pooled temporary.
    assert_eq!(
        sized_allocs(output_bytes, || a.contract(&b, &spec).unwrap()),
        2
    );
    for _ in 0..3 {
        assert_eq!(
            sized_allocs(output_bytes, || a.contract(&b, &spec).unwrap()),
            1
        );
    }

    // What: `contract_into` takes the same route (#1631) with the same pooled
    // temporary, so a warm call allocates nothing output-sized at all.
    let mut destination = a.contract(&b, &spec).unwrap();
    for _ in 0..2 {
        a.contract_into(&b, &spec, &mut destination, 1.0, 0.0)
            .unwrap();
    }
    for _ in 0..3 {
        assert_eq!(
            sized_allocs(output_bytes, || a
                .contract_into(&b, &spec, &mut destination, 1.0, 0.0)
                .unwrap()),
            0
        );
    }

    // What: bit-identical to the literal copyC sequence spelled with public
    // operations — the default-split contraction, then the permute.
    let two_step = a
        .contract(
            &b,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap()
        .permute(&[0, 2], &[1])
        .unwrap();
    let fused = a.contract(&b, &spec).unwrap();
    assert_eq!(fused.codomain(), two_step.codomain());
    assert_eq!(fused.domain(), two_step.domain());
    assert_eq!(fused.dense_data().unwrap(), two_step.dense_data().unwrap());
}
