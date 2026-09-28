//! #1614: an intermediate network step whose next-use orientation moves legs
//! across the codomain/domain split is one contraction into one retained
//! buffer. Before #1614 the step contracted into one retained buffer and then
//! made a separate permute call into a second one. On the copyC route the
//! data moved is the same (GEMM into the pooled temporary, then one
//! transform); the saving is the separate call and its second buffer.
//!
//! This test, with the step loop's structure (no permute path per step), is
//! the evidence for #1614's gate: tenet-network has no transform-seam
//! counter, so the warm transform-lookup count and the retained workspace
//! bytes stand in for it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{Complex32, Complex64};
use tenet::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};
use tenet_network::{plan_cache_stats, tensor};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

struct CountingAllocator;

static WATCHED_BYTES: AtomicUsize = AtomicUsize::new(0);
static WATCHED_ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() == WATCHED_BYTES.load(Ordering::Relaxed) {
            WATCHED_ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwards the caller's layout unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` came from `System` with this layout.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size == WATCHED_BYTES.load(Ordering::Relaxed) {
            WATCHED_ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwards the caller's pointer and layout unchanged.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

type Map = TensorMap<U1FusionRule, f64>;

fn transform_lookups(runtime: &Runtime) -> usize {
    let info = runtime.tree_transform_cache_info().structures;
    info.hits() + info.misses()
}

#[test]
fn split_moving_intermediate_retains_one_buffer_and_one_transform() {
    // a[x; c] · b[c; y, z, u] runs first (x < w), giving (x | y, z, u). Its
    // consumer contracts y and z with t[y, z; w], so the next-use orientation
    // is (x, u | y, z) as lhs or (y, z | x, u) as rhs: two codomain legs where
    // the default split has one, a split move either way. Distinct prime
    // degeneracies keep the intermediate payload size unique in the run.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = |n| GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), n)]).unwrap();
    let (x, c, y, z, u, w) = (
        space(11),
        space(13),
        space(17),
        space(19),
        space(23),
        space(29),
    );
    let a = Map::rand_with_seed(&runtime, [&x], [&c], 1).unwrap();
    let b = Map::rand_with_seed(&runtime, [&c], [&y, &z, &u], 2).unwrap();
    let t = Map::rand_with_seed(&runtime, [&y, &z], [&w], 3).unwrap();
    let intermediate_bytes = 11 * 17 * 19 * 23 * size_of::<f64>();
    WATCHED_BYTES.store(intermediate_bytes, Ordering::SeqCst);

    let run = || tensor!([x, u; w] = a[x; c] * b[c; y, z, u] * t[y, z; w]).unwrap();
    let oracle = a
        .contract(
            &b,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1, 2, 3],
            },
        )
        .unwrap()
        .contract(
            &t,
            &ContractSpec {
                lhs: &[1, 2],
                rhs: &[0, 1],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap();
    for _ in 0..2 {
        let output = run();
        numerics::assert_slices_close(
            "network vs chained contract",
            output.dense_data().unwrap(),
            oracle.dense_data().unwrap(),
            13 * 17 * 19,
        );
    }

    let lookups_before = transform_lookups(&runtime);
    let allocs_before = WATCHED_ALLOCS.load(Ordering::SeqCst);
    drop(run());
    let warm_allocs = WATCHED_ALLOCS.load(Ordering::SeqCst) - allocs_before;
    let warm_lookups = transform_lookups(&runtime) - lookups_before;
    let retained = plan_cache_stats(&runtime).retained_workspace_bytes;

    // What: the intermediate's buffer is reused, not reallocated.
    assert_eq!(warm_allocs, 0, "warm intermediate-sized allocations");
    // What: one transform for the one split-moving intermediate — the
    // step's own output permute (copyC). An operand-rebuilding route would
    // look up three.
    assert!(warm_lookups <= 1, "warm transform lookups: {warm_lookups}");
    // What: the idle workspace keeps one intermediate buffer plus metadata.
    // A separate orientation permute retains a second intermediate-sized
    // buffer (1,319,096 B on the pre-#1614 lowering against 660,988 B).
    assert!(
        retained >= intermediate_bytes && retained < intermediate_bytes + intermediate_bytes / 4,
        "retained workspace bytes {retained} vs one intermediate buffer {intermediate_bytes}"
    );
}
