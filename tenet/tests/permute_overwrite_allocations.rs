use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::{Arc, Mutex};

use tenet::core::{
    product_sector, ProductFusionRuleExt, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, TensorMap};

#[cfg(feature = "racah-generated")]
use tenet::typed::SUNFusionRule;

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
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

fn measure(f: impl FnOnce()) -> (usize, usize) {
    ALLOCATIONS.set(0);
    BYTES.set(0);
    COUNTING.set(true);
    f();
    COUNTING.set(false);
    (ALLOCATIONS.get(), BYTES.get())
}

#[cfg(feature = "racah-generated")]
fn measure_value<T>(f: impl FnOnce() -> T) -> (T, usize, usize, u128) {
    ALLOCATIONS.set(0);
    BYTES.set(0);
    COUNTING.set(true);
    let started = std::time::Instant::now();
    let output = f();
    let elapsed = started.elapsed().as_nanos();
    COUNTING.set(false);
    (output, ALLOCATIONS.get(), BYTES.get(), elapsed)
}

#[test]
fn cached_permute_overwrite_does_not_allocate_on_the_caller_thread() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: a warmed multiplicity-free non-Abelian permutation reuses its
    // compiled plan and replay workspace without allocating on the caller.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule.product(SU2FusionRule));
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(0)),
                8,
            ),
            (
                product_sector(U1Irrep::new(1), SU2Irrep::from_twice_spin(1)),
                6,
            ),
            (
                product_sector(U1Irrep::new(-1), SU2Irrep::from_twice_spin(1)),
                6,
            ),
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(2)),
                4,
            ),
        ],
        false,
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 197).unwrap();
    let mut destination = source.permute(&[1], &[2, 0]).unwrap();

    source
        .permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0)
        .unwrap();
    let destination_data = destination.data().as_ptr();

    ALLOCATIONS.set(0);
    BYTES.set(0);
    COUNTING.set(true);
    source
        .permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0)
        .unwrap();
    COUNTING.set(false);
    black_box(destination.data());

    assert_eq!((ALLOCATIONS.get(), BYTES.get()), (0, 0));
    assert_eq!(destination.data().as_ptr(), destination_data);
    assert!(std::ptr::eq(destination.provider(), provider.as_ref()));
}

#[test]
fn cached_u1_permute_overwrite_does_not_allocate_on_the_caller_thread() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: a warmed UniqueFusion permutation reuses its completed transformer
    // and replay workspace without allocating on the caller.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (U1Irrep::new(-1), 4),
            (U1Irrep::new(0), 8),
            (U1Irrep::new(1), 4),
        ],
        false,
    )
    .unwrap();
    let source: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 418).unwrap();
    let mut destination = source.permute(&[1], &[2, 0]).unwrap();

    source
        .permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0)
        .unwrap();
    let destination_data = destination.data().as_ptr();

    ALLOCATIONS.set(0);
    BYTES.set(0);
    COUNTING.set(true);
    source
        .permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0)
        .unwrap();
    COUNTING.set(false);
    black_box(destination.data());

    assert_eq!((ALLOCATIONS.get(), BYTES.get()), (0, 0));
    assert_eq!(destination.data().as_ptr(), destination_data);
    assert!(std::ptr::eq(destination.provider(), provider.as_ref()));
}

#[test]
fn cached_planar_overwrites_do_not_allocate_on_the_caller_thread() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: the shared typed destination seam also reuses admitted full,
    // explicit, and repartition transpose operations without caller allocation.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
        false,
    )
    .unwrap();
    let source: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 779).unwrap();

    let mut full = source.transpose().unwrap();
    source.transpose_overwrite_into(&mut full, 1.0).unwrap();
    assert_eq!(
        measure(|| source.transpose_overwrite_into(&mut full, 1.0).unwrap()),
        (0, 0)
    );

    let mut explicit = source.transpose_axes(&[1, 2], &[0]).unwrap();
    source
        .transpose_axes_overwrite_into(&mut explicit, &[1, 2], &[0], 1.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .transpose_axes_overwrite_into(&mut explicit, &[1, 2], &[0], 1.0)
                .unwrap()
        }),
        (0, 0)
    );

    let mut repartitioned = source.repartition(1).unwrap();
    source
        .repartition_overwrite_into(&mut repartitioned, 1.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .repartition_overwrite_into(&mut repartitioned, 1.0)
                .unwrap()
        }),
        (0, 0)
    );
    assert!(std::ptr::eq(full.provider(), provider.as_ref()));
    assert!(std::ptr::eq(explicit.provider(), provider.as_ref()));
    assert!(std::ptr::eq(repartitioned.provider(), provider.as_ref()));
}

#[cfg(feature = "racah-generated")]
fn assert_same_checked_tensor(
    actual: &TensorMap<SUNFusionRule, f64>,
    expected: &TensorMap<SUNFusionRule, f64>,
) {
    assert_eq!(actual.block_count(), expected.block_count());
    assert_eq!(actual.data().len(), expected.data().len());
    for index in 0..actual.block_count() {
        assert_eq!(
            actual.block_fusion_trees(index).unwrap(),
            expected.block_fusion_trees(index).unwrap()
        );
    }
    for (actual, expected) in actual.data().iter().zip(expected.data()) {
        assert!((actual - expected).abs() <= 1e-12);
    }
}

/// Measurement only: each child process starts with cold provider and Runtime
/// state. Allocation counts cover requested bytes on the caller thread, which
/// is the same scope as the established overwrite allocation gates above.
#[cfg(feature = "racah-generated")]
#[test]
#[ignore = "manual checked-Generic public transform measurement"]
fn checked_generic_public_transform_measurement() {
    const CASE_ENV: &str = "TENET_CHECKED_GENERIC_MEASUREMENT_CASE";
    const TEST_NAME: &str = "checked_generic_public_transform_measurement";

    let Some(case) = std::env::var_os(CASE_ENV) else {
        for case in [
            "su3_permute",
            "su3_braid",
            "su3_repartition",
            "su4_permute",
            "su4_braid",
            "su4_repartition",
            "su2_permute_control",
        ] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    TEST_NAME,
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CASE_ENV, case)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "case={case} status={}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                output.status
            );
            print!("{stdout}");
        }
        return;
    };
    let case = case.to_str().unwrap();
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    println!("case={case} allocation_scope=caller_thread_requested");

    if case == "su2_permute_control" {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(SU2FusionRule);
        let half = SU2Irrep::from_twice_spin(1);
        let half_leg = GradedSpace::try_new(Arc::clone(&provider), [(half, 1)], false).unwrap();
        let coupled_leg = GradedSpace::try_new(
            Arc::clone(&provider),
            [
                (SU2Irrep::from_twice_spin(0), 1),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
            false,
        )
        .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&runtime, [&half_leg, &half_leg], [&coupled_leg], 783)
                .unwrap();
        assert_eq!(source.block_count(), 2);
        let runtime_before = runtime.tree_transform_cache_info();
        let (first, first_allocations, first_bytes, first_ns) =
            measure_value(|| source.permute(&[1, 0], &[2]).unwrap());
        let runtime_after_first = runtime.tree_transform_cache_info();
        let mut repeat_ns = Vec::with_capacity(7);
        let mut repeat_allocations = Vec::with_capacity(7);
        let mut repeat_bytes = Vec::with_capacity(7);
        for _ in 0..7 {
            let (repeated, allocations, bytes, ns) =
                measure_value(|| source.permute(&[1, 0], &[2]).unwrap());
            assert_eq!(repeated.data(), first.data());
            repeat_ns.push(ns);
            repeat_allocations.push(allocations);
            repeat_bytes.push(bytes);
        }
        let runtime_after_repeat = runtime.tree_transform_cache_info();
        let mut sorted_ns = repeat_ns.clone();
        sorted_ns.sort_unstable();
        println!(
            "case={case} phase=public_first ns={first_ns} allocations={first_allocations} requested_bytes={first_bytes} runtime_before={runtime_before:?} runtime_after={runtime_after_first:?}"
        );
        println!(
            "case={case} phase=public_repeat samples_ns={repeat_ns:?} median_ns={} allocations={repeat_allocations:?} requested_bytes={repeat_bytes:?} runtime_after={runtime_after_repeat:?}",
            sorted_ns[sorted_ns.len() / 2]
        );
        assert!(runtime_after_first.entries() > runtime_before.entries());
        assert!(runtime_after_repeat.hits() > runtime_after_first.hits());
        return;
    }

    let (n, adjoint, operation) = match case {
        "su3_permute" => (3, vec![1, 1], "permute"),
        "su3_braid" => (3, vec![1, 1], "braid"),
        "su3_repartition" => (3, vec![1, 1], "repartition"),
        "su4_permute" => (4, vec![1, 0, 1], "permute"),
        "su4_braid" => (4, vec![1, 0, 1], "braid"),
        "su4_repartition" => (4, vec![1, 0, 1], "repartition"),
        _ => panic!("unknown measurement case: {case}"),
    };
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let apply = |source: &TensorMap<SUNFusionRule, f64>| match operation {
        "permute" => source.permute(&[1, 0], &[2]).unwrap(),
        "braid" => source.braid(&[1, 0], &[2], &[0, 1, 2]).unwrap(),
        "repartition" => source.repartition(1).unwrap(),
        _ => unreachable!(),
    };

    let runtime_before = runtime.tree_transform_cache_info();
    let (first, first_allocations, first_bytes, first_ns) = measure_value(|| apply(&source));
    let runtime_after_first = runtime.tree_transform_cache_info();
    let mut repeat_ns = Vec::with_capacity(7);
    let mut repeat_allocations = Vec::with_capacity(7);
    let mut repeat_bytes = Vec::with_capacity(7);
    for _ in 0..7 {
        let (repeated, allocations, bytes, ns) = measure_value(|| apply(&source));
        assert_same_checked_tensor(&repeated, &first);
        assert!(std::ptr::eq(repeated.provider(), provider.as_ref()));
        repeat_ns.push(ns);
        repeat_allocations.push(allocations);
        repeat_bytes.push(bytes);
    }
    let runtime_after_repeat = runtime.tree_transform_cache_info();
    let mut sorted_ns = repeat_ns.clone();
    sorted_ns.sort_unstable();
    println!(
        "case={case} phase=public_first_after_source_construction coefficient_caches=cold ns={first_ns} allocations={first_allocations} requested_bytes={first_bytes} runtime_before={runtime_before:?} runtime_after={runtime_after_first:?}"
    );
    println!(
        "case={case} phase=public_repeat_provider_warm samples_ns={repeat_ns:?} median_ns={} allocations={repeat_allocations:?} requested_bytes={repeat_bytes:?} runtime_after={runtime_after_repeat:?}",
        sorted_ns[sorted_ns.len() / 2]
    );
    assert_eq!(runtime_after_first, runtime_before);
    assert_eq!(runtime_after_repeat, runtime_before);
    assert!(std::ptr::eq(first.provider(), provider.as_ref()));
}
