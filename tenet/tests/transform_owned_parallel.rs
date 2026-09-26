//! Owned transforms on a multi-threaded recoupling backend (#1217): the
//! uninitialised owned writer runs under the parallel schedule, so the result
//! is bit-identical to the serial owned result and the caller thread makes the
//! same allocations as the serial path, with no allocator-zeroed payload (the
//! `vec![zero; P]` fallback that the thread-count gate used to force).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::{Arc, Mutex};

use num_complex::Complex64;
use tenet::core::{
    product_sector, ProductFusionRuleExt, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, TensorMap};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static ZEROED_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
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

    // Why override: `vec![0.0; P]` lowers to `alloc_zeroed`, which the default
    // implementation would report as a plain `alloc`.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            BYTES.set(BYTES.get() + layout.size());
            ZEROED_ALLOCATIONS.set(ZEROED_ALLOCATIONS.get() + 1);
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
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

/// (allocations, bytes, zeroed allocations) on the caller thread.
fn measure<T>(f: impl FnOnce() -> T) -> (T, usize, usize, usize) {
    ALLOCATIONS.set(0);
    BYTES.set(0);
    ZEROED_ALLOCATIONS.set(0);
    COUNTING.set(true);
    let output = f();
    COUNTING.set(false);
    (
        output,
        ALLOCATIONS.get(),
        BYTES.get(),
        ZEROED_ALLOCATIONS.get(),
    )
}

const THREADS: usize = 3;

/// The parallel runtime replays on this pool so the effective worker count is
/// `THREADS` regardless of the host's core count or rayon's global pool.
fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build()
        .unwrap()
}

fn runtimes() -> (Runtime, Runtime) {
    (
        Runtime::builder().recoupling_threads(1).build().unwrap(),
        Runtime::builder()
            .recoupling_threads(THREADS)
            .build()
            .unwrap(),
    )
}

type U1Su2 = tenet::core::ProductFusionRule<U1FusionRule, SU2FusionRule>;

/// U(1) x SU(2): every non-trivial permutation recouples (Multi scatter
/// groups) and the degeneracies put the payload past the backend's parallel
/// size gate.
fn u1_su2_space(provider: &Arc<U1Su2>) -> GradedSpace<U1Su2> {
    GradedSpace::try_new_with_arc(
        Arc::clone(provider),
        [
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(0)),
                18,
            ),
            (
                product_sector(U1Irrep::new(1), SU2Irrep::from_twice_spin(1)),
                14,
            ),
            (
                product_sector(U1Irrep::new(-1), SU2Irrep::from_twice_spin(1)),
                14,
            ),
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(2)),
                12,
            ),
        ],
    )
    .unwrap()
}

fn u1_space(provider: &Arc<U1FusionRule>) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::clone(provider),
        [
            (U1Irrep::new(-1), 36),
            (U1Irrep::new(0), 48),
            (U1Irrep::new(1), 36),
        ],
    )
    .unwrap()
}

macro_rules! transforms {
    ($source:expr) => {
        vec![
            ("permute", $source.permute(&[1], &[2, 0]).unwrap()),
            ("braid", $source.braid(&[1, 0], &[2], &[0, 2, 1]).unwrap()),
            ("repartition", $source.repartition(1).unwrap()),
            ("transpose", $source.transpose().unwrap()),
            (
                "transpose_axes",
                $source.transpose_axes(&[1, 2], &[0]).unwrap(),
            ),
        ]
    };
}

macro_rules! assert_bit_identical {
    ($expected:expr, $actual:expr) => {
        for ((name, expected), (_, actual)) in $expected.iter().zip(&$actual) {
            assert_eq!(expected.data().len(), actual.data().len(), "{name}");
            assert_eq!(expected.subblock_count(), actual.subblock_count(), "{name}");
            assert!(expected.data() == actual.data(), "{name}: payloads differ");
        }
    };
}

#[test]
fn u1_su2_owned_transforms_are_bit_identical_across_thread_counts() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (serial, parallel) = runtimes();
    let provider = Arc::new(U1FusionRule.product(SU2FusionRule));
    let space = u1_su2_space(&provider);
    let real_serial: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&serial, [&space, &space], [&space], 1217).unwrap();
    let real_parallel: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&parallel, [&space, &space], [&space], 1217).unwrap();
    assert!(real_serial.data() == real_parallel.data());
    assert!(
        real_serial.data().len() > 1 << 15,
        "fixture must exceed the backend's parallel size gate: {}",
        real_serial.data().len()
    );
    let complex_serial: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&serial, [&space, &space], [&space], 1218).unwrap();
    let complex_parallel: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&parallel, [&space, &space], [&space], 1218).unwrap();
    assert!(complex_serial.data() == complex_parallel.data());

    let expected_real = transforms!(real_serial);
    let expected_complex = transforms!(complex_serial);
    let (actual_real, actual_complex) =
        pool().install(|| (transforms!(real_parallel), transforms!(complex_parallel)));
    assert_bit_identical!(expected_real, actual_real);
    assert_bit_identical!(expected_complex, actual_complex);
}

#[test]
fn u1_owned_transforms_are_bit_identical_across_thread_counts() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (serial, parallel) = runtimes();
    let provider = Arc::new(U1FusionRule);
    let space = u1_space(&provider);
    let real_serial: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&serial, [&space, &space], [&space], 1219).unwrap();
    let real_parallel: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&parallel, [&space, &space], [&space], 1219).unwrap();
    assert!(real_serial.data() == real_parallel.data());
    assert!(
        real_serial.data().len() > 1 << 15,
        "fixture must exceed the backend's parallel size gate: {}",
        real_serial.data().len()
    );
    let complex_serial: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&serial, [&space, &space], [&space], 1220).unwrap();
    let complex_parallel: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&parallel, [&space, &space], [&space], 1220).unwrap();

    let expected_real = transforms!(real_serial);
    let expected_complex = transforms!(complex_serial);
    let (actual_real, actual_complex) =
        pool().install(|| (transforms!(real_parallel), transforms!(complex_parallel)));
    assert_bit_identical!(expected_real, actual_real);
    assert_bit_identical!(expected_complex, actual_complex);
}

#[test]
fn parallel_owned_permute_allocates_like_the_serial_owned_path() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // What: with `recoupling_threads > 1` a warmed owned permute makes the
    // same caller-thread allocations as the serial owned path and none of them
    // is allocator-zeroed. Before #1217 the multi-threaded backend returned
    // `None` from the owned writer and fell back to `vec![0.0; P]` (one
    // `alloc_zeroed` of `P * 8` bytes) plus a destination-reading replay.
    let (serial, parallel) = runtimes();
    let provider = Arc::new(U1FusionRule.product(SU2FusionRule));
    let space = u1_su2_space(&provider);
    let source_serial: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&serial, [&space, &space], [&space], 1221).unwrap();
    let source_parallel: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&parallel, [&space, &space], [&space], 1221).unwrap();
    let payload_bytes = size_of_val(source_serial.data());
    assert!(
        source_serial.data().len() > 1 << 15,
        "fixture must exceed the backend's parallel size gate: {}",
        source_serial.data().len()
    );

    let warm_serial = source_serial.permute(&[1], &[2, 0]).unwrap();
    let (result_serial, serial_allocations, serial_bytes, serial_zeroed) =
        measure(|| source_serial.permute(&[1], &[2, 0]).unwrap());
    black_box(result_serial.data());

    let pool = pool();
    let warm_parallel = pool.install(|| source_parallel.permute(&[1], &[2, 0]).unwrap());
    assert!(warm_serial.data() == warm_parallel.data());
    let (result_parallel, parallel_allocations, parallel_bytes, parallel_zeroed) =
        pool.install(|| measure(|| source_parallel.permute(&[1], &[2, 0]).unwrap()));
    black_box(result_parallel.data());

    assert!(result_serial.data() == result_parallel.data());
    assert!(
        serial_bytes >= payload_bytes && parallel_bytes >= payload_bytes,
        "owned outputs allocate at least the payload: {serial_bytes} / {parallel_bytes}"
    );
    assert_eq!(
        (parallel_allocations, parallel_bytes, parallel_zeroed),
        (serial_allocations, serial_bytes, 0),
        "serial owned path: {serial_allocations} allocations, {serial_bytes} bytes, {serial_zeroed} zeroed"
    );
}
