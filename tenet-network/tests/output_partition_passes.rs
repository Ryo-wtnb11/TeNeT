//! What: a warm contraction whose output codomain takes legs from both
//! operands costs one output-sized allocation, by the eager `contract(spec)`
//! and by `tensor!` alike: the network step passes the output split as its
//! `ContractSpec` (TensorOperations `pAB`), so no separate permute pass and no
//! second output-sized buffer follow the contraction. Spelling the same result
//! as contract-then-permute still costs two.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tenet::prelude::*;
use tenet_network::tensor;

struct CountingAllocator;

static ENABLED: AtomicBool = AtomicBool::new(false);
static PAYLOAD_BYTES: AtomicUsize = AtomicUsize::new(0);
static PAYLOAD_ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) && layout.size() == PAYLOAD_BYTES.load(Ordering::Relaxed)
        {
            PAYLOAD_ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwards the caller's layout unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` was returned by `System.alloc` with this layout.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Output-payload-sized allocations made while `f` runs (result dropped after).
fn output_sized_allocs<T>(f: impl FnOnce() -> T) -> usize {
    PAYLOAD_ALLOCS.store(0, Ordering::SeqCst);
    ENABLED.store(true, Ordering::SeqCst);
    let result = f();
    ENABLED.store(false, Ordering::SeqCst);
    drop(result);
    PAYLOAD_ALLOCS.load(Ordering::SeqCst)
}

type Map = TensorMap<U1FusionRule, f64>;

/// `a: p ⊗ q ← c`, `b: c ← r` on a fresh runtime (cold plan cache and pool).
/// Distinct prime degeneracies keep the output payload size unique.
fn operands() -> (Map, Map) {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = |n| {
        GradedSpace::try_new(std::sync::Arc::new(U1FusionRule), [(U1Irrep::new(0), n)]).unwrap()
    };
    let (p, q, c, r) = (space(13), space(17), space(19), space(23));
    (
        Map::rand_with_seed(&runtime, [&p, &q], [&c], 1).unwrap(),
        Map::rand_with_seed(&runtime, [&c], [&r], 2).unwrap(),
    )
}

#[test]
fn mixed_output_partition_costs_one_output_sized_allocation() {
    PAYLOAD_BYTES.store(13 * 17 * 23 * size_of::<f64>(), Ordering::SeqCst);

    let (a, b) = operands();
    assert_eq!(
        output_sized_allocs(|| a
            .contract(
                &b,
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2]
                }
            )
            .unwrap()),
        1
    );
    let (a, b) = operands();
    assert_eq!(
        output_sized_allocs(|| {
            a.contract(
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
            .unwrap()
        }),
        2
    );

    let (a, b) = operands();
    let moved = || {
        a.contract(
            &b,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 2],
                domain: &[1],
            },
        )
        .unwrap()
    };
    // Cold: the result and the Runtime-pooled copyC temporary (#1626).
    assert_eq!(output_sized_allocs(moved), 2);
    assert_eq!(output_sized_allocs(moved), 1);

    let (a, b) = operands();
    assert_eq!(
        output_sized_allocs(|| tensor!([p, q; r] = a[p, q; c] * b[c; r]).unwrap()),
        1
    );
    let (a, b) = operands();
    let mixed = || tensor!([p, r; q] = a[p, q; c] * b[c; r]).unwrap();
    // The step's spec is that same contract: cold it pools its temporary,
    // warm only the result is output-sized — no permute pass follows.
    assert_eq!(output_sized_allocs(mixed), 2);
    assert_eq!(output_sized_allocs(mixed), 1);
    assert_eq!(output_sized_allocs(mixed), 1);
}
