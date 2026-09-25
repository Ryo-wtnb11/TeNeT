//! What: a contraction whose output codomain takes legs from both operands
//! costs one output-sized intermediate plus a permute pass, by the eager
//! route and by `tensor!` alike. Neither produces that partition in one pass
//! (TensorKit `blas_contract!` takes the same `copyC` path when `C` is not a
//! BLAS destination). The warm `tensor!` pool reuses the intermediate, so only
//! its data movement remains.

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
    let space = |n| GradedSpace::try_new(U1FusionRule, [(U1Irrep::new(0), n)]).unwrap();
    let (p, q, c, r) = (space(13), space(17), space(19), space(23));
    (
        Map::rand_with_seed(&runtime, [&p, &q], [&c], 1).unwrap(),
        Map::rand_with_seed(&runtime, [&c], [&r], 2).unwrap(),
    )
}

#[test]
fn mixed_output_partition_costs_one_extra_output_sized_pass() {
    PAYLOAD_BYTES.store(13 * 17 * 23 * size_of::<f64>(), Ordering::SeqCst);

    let (a, b) = operands();
    assert_eq!(
        output_sized_allocs(|| a.contract(&b, &[2], &[0], &[0, 1, 2]).unwrap()),
        1
    );
    let (a, b) = operands();
    assert_eq!(
        output_sized_allocs(|| {
            a.contract(&b, &[2], &[0], &[0, 1, 2])
                .unwrap()
                .permute(&[0, 2], &[1])
                .unwrap()
        }),
        2
    );

    let (a, b) = operands();
    assert_eq!(
        output_sized_allocs(|| tensor!([p, q; r] = a[p, q; c] * b[c; r]).unwrap()),
        1
    );
    let (a, b) = operands();
    let mixed = || tensor!([p, r; q] = a[p, q; c] * b[c; r]).unwrap();
    assert_eq!(output_sized_allocs(mixed), 2);
    // Warm: the pooled intermediate is reused; the permute pass still runs.
    assert_eq!(output_sized_allocs(mixed), 1);
}
