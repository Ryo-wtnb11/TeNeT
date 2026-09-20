//! Allocation contracts of the single-precision payloads (#1315).
//!
//! Two contracts:
//!
//! * admitting a payload dtype to `TensorScalar` adds an execution lane per
//!   `Ctxs`, and those lanes are built on first use — a program that never
//!   names a dtype pays nothing for it at `Runtime::build()`;
//! * a single-precision tensor performs the *same number* of allocations as
//!   its double-precision twin and allocates exactly half the payload bytes.
//!   Narrowing the payload must not change the algorithm, only the bytes it
//!   moves.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{Complex32, Complex64, GradedSpace, Runtime, TensorMap};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            BYTES.set(BYTES.get() + layout.size());
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
            BYTES.set(BYTES.get() + new_size);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCATIONS.set(0);
    BYTES.set(0);
    COUNTING.set(true);
    let value = operation();
    COUNTING.set(false);
    (value, ALLOCATIONS.get(), BYTES.get())
}

/// Measured at `origin/main` f032d121, before `f32`/`Complex32` were admitted
/// to `TensorScalar`: 199 allocation calls, 68512 bytes. After admission the
/// call count is unchanged and the bytes grow by 144 — the two `Option<Box<_>>`
/// lane slots and the retained lane configuration inside the same allocation.
/// Building all four lanes eagerly instead would add four contexts, each with
/// two dense executors and a contract workspace, to every runtime.
const F64_ONLY_CONSTRUCTION_CALLS: usize = 199;

#[test]
fn runtime_construction_does_not_build_unused_dtype_lanes() {
    // Warm anything the first runtime initialises lazily and process-wide
    // (thread pools, env parsing) so the measured build is steady state.
    black_box(Runtime::builder().dense_threads(1).build().unwrap());

    let (runtime, calls, bytes) = measured(|| Runtime::builder().dense_threads(1).build().unwrap());
    eprintln!("Runtime::build: {calls} allocation calls, {bytes} bytes");
    assert!(
        calls <= F64_ONLY_CONSTRUCTION_CALLS,
        "admitting a payload dtype must not charge runtime construction: \
         {calls} allocation calls against {F64_ONLY_CONSTRUCTION_CALLS} before admission"
    );

    // The lane is built on first use, so the first single-precision tensor on
    // this runtime pays for it and the second does not.
    let space =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let build = |seed| {
        TensorMap::<U1FusionRule, Complex32>::rand_with_seed(&runtime, [&space], [&space], seed)
            .unwrap()
    };
    let (cold, cold_calls, _) = measured(|| build(1));
    let (warm, warm_calls, _) = measured(|| build(2));
    black_box((cold, warm));
    eprintln!("first Complex32 tensor: {cold_calls} calls, second: {warm_calls} calls");
    assert!(
        cold_calls > warm_calls,
        "the Complex32 lane should be constructed on first use, not at Runtime::build"
    );
}

fn u1_space() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 4),
            (U1Irrep::new(0), 5),
            (U1Irrep::new(1), 4),
        ],
    )
    .unwrap()
}

/// Builds a tensor, permutes it and reduces it, at one payload dtype.
///
/// A macro because the payload dtype is the only thing that varies and the
/// two instantiations must run the identical sequence of public calls.
macro_rules! measure_pipeline {
    ($runtime:expr, $dtype:ty, $space:expr) => {{
        let space = $space;
        // Warm the lane, the layout admission and the transform plan, so the
        // measured pass is steady state at both precisions.
        let warm: TensorMap<U1FusionRule, $dtype> =
            TensorMap::rand_with_seed($runtime, [space, space], [space], 5_501).unwrap();
        black_box(warm.permute(&[1], &[2, 0]).unwrap());
        black_box(warm.norm().unwrap());

        measured(|| {
            let tensor: TensorMap<U1FusionRule, $dtype> =
                TensorMap::rand_with_seed($runtime, [space, space], [space], 5_502).unwrap();
            let permuted = tensor.permute(&[1], &[2, 0]).unwrap();
            let sum = tensor
                .add(&tensor, <$dtype>::from(1.0), <$dtype>::from(1.0))
                .unwrap();
            black_box((permuted, sum, tensor.norm().unwrap(), tensor.data().len()))
        })
    }};
}

#[test]
fn single_precision_allocates_as_often_as_double_and_half_the_payload_bytes() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = u1_space();

    let (_, f64_calls, f64_bytes) = measure_pipeline!(&runtime, f64, &space);
    let (single, f32_calls, f32_bytes) = measure_pipeline!(&runtime, f32, &space);
    let (_, c64_calls, c64_bytes) = measure_pipeline!(&runtime, Complex64, &space);
    let (_, c32_calls, c32_bytes) = measure_pipeline!(&runtime, Complex32, &space);
    let payload = single.3;

    eprintln!(
        "payload {payload}: f64 {f64_calls} calls/{f64_bytes} B, f32 {f32_calls}/{f32_bytes}, \
         c64 {c64_calls}/{c64_bytes}, c32 {c32_calls}/{c32_bytes}"
    );

    assert_eq!(
        f32_calls, f64_calls,
        "f32 must run the same algorithm as f64, not a different number of allocations"
    );
    assert_eq!(c32_calls, c64_calls, "Complex32 against Complex64");

    // Three payload-sized buffers are produced above (the tensor, the permuted
    // result and the sum), so narrowing the payload saves exactly half of each.
    assert_eq!(
        f64_bytes - f32_bytes,
        3 * payload * (size_of::<f64>() - size_of::<f32>()),
        "f32 must allocate exactly half the payload bytes of f64"
    );
    assert_eq!(
        c64_bytes - c32_bytes,
        3 * payload * (size_of::<Complex64>() - size_of::<Complex32>()),
        "Complex32 must allocate exactly half the payload bytes of Complex64"
    );
}
