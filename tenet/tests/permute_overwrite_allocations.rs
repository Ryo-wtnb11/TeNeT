use std::hint::black_box;
use std::sync::Arc;

use tenet::sector::{
    product_sector, ProductFusionRuleExt, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
};
use tenet::typed::Runtime;
use tenet::typed::{GradedSpace, TensorMap};

#[cfg(feature = "racah-generated")]
use tenet::sector::SUNFusionRule;

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

fn measure(f: impl FnOnce()) -> (usize, usize) {
    let ((), allocs) = counting_alloc::measure(f);
    (allocs.calls as usize, allocs.bytes as usize)
}

#[cfg(feature = "racah-generated")]
fn measure_value<T>(f: impl FnOnce() -> T) -> (T, usize, usize, u128) {
    let ((output, elapsed), allocs) = counting_alloc::measure(|| {
        let started = std::time::Instant::now();
        let output = f();
        (output, started.elapsed().as_nanos())
    });
    (
        output,
        allocs.calls as usize,
        allocs.bytes as usize,
        elapsed,
    )
}

#[test]
fn cached_permute_overwrite_does_not_allocate_on_the_caller_thread() {
    let _measurement = counting_alloc::serial();
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
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 197).unwrap();
    let mut destination = source.permute(&[1], &[2, 0]).unwrap();

    source
        .permute_into(&[1], &[2, 0], &mut destination, 1.0, 0.0)
        .unwrap();
    let destination_data = destination.dense_data().unwrap().as_ptr();

    let cost = measure(|| {
        source
            .permute_into(&[1], &[2, 0], &mut destination, 1.0, 0.0)
            .unwrap();
    });
    black_box(destination.dense_data().unwrap());

    assert_eq!(cost, (0, 0));
    assert_eq!(destination.dense_data().unwrap().as_ptr(), destination_data);
    assert!(std::ptr::eq(destination.provider(), provider.as_ref()));
}

#[test]
fn cached_u1_permute_overwrite_does_not_allocate_on_the_caller_thread() {
    let _measurement = counting_alloc::serial();
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
    )
    .unwrap();
    let source: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 418).unwrap();
    let mut destination = source.permute(&[1], &[2, 0]).unwrap();

    source
        .permute_into(&[1], &[2, 0], &mut destination, 1.0, 0.0)
        .unwrap();
    let destination_data = destination.dense_data().unwrap().as_ptr();

    let cost = measure(|| {
        source
            .permute_into(&[1], &[2, 0], &mut destination, 1.0, 0.0)
            .unwrap();
    });
    black_box(destination.dense_data().unwrap());

    assert_eq!(cost, (0, 0));
    assert_eq!(destination.dense_data().unwrap().as_ptr(), destination_data);
    assert!(std::ptr::eq(destination.provider(), provider.as_ref()));
}

#[test]
fn cached_planar_overwrites_do_not_allocate_on_the_caller_thread() {
    let _measurement = counting_alloc::serial();
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
    )
    .unwrap();
    let source: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 779).unwrap();

    let mut full = source.transpose(&[2], &[1, 0]).unwrap();
    source
        .transpose_into(&[2], &[1, 0], &mut full, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .transpose_into(&[2], &[1, 0], &mut full, 1.0, 0.0)
                .unwrap()
        }),
        (0, 0)
    );

    let mut explicit = source.transpose(&[1, 2], &[0]).unwrap();
    source
        .transpose_into(&[1, 2], &[0], &mut explicit, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .transpose_into(&[1, 2], &[0], &mut explicit, 1.0, 0.0)
                .unwrap()
        }),
        (0, 0)
    );

    let mut repartitioned = source.repartition(1).unwrap();
    source
        .repartition_into(&mut repartitioned, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .repartition_into(&mut repartitioned, 1.0, 0.0)
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
    assert_eq!(actual.subblock_count(), expected.subblock_count());
    assert_eq!(
        actual.dense_data().unwrap().len(),
        expected.dense_data().unwrap().len()
    );
    for index in 0..actual.subblock_count() {
        assert_eq!(
            actual.subblock_fusion_trees(index).unwrap(),
            expected.subblock_fusion_trees(index).unwrap()
        );
    }
    for (actual, expected) in actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
    {
        assert!((actual - expected).abs() <= 1e-12);
    }
}

/// Measurement only: each child process starts with cold provider and Runtime
/// state. Allocation counts cover requested bytes on the caller thread, which
/// is the same scope as the established overwrite allocation gates above.
#[cfg(feature = "racah-generated")]
#[test]
#[ignore = "benchmark: run via benchmarks.yml"]
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
    let _measurement = counting_alloc::serial();
    println!("case={case} allocation_scope=caller_thread_requested");

    if case == "su2_permute_control" {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(SU2FusionRule);
        let half = SU2Irrep::from_twice_spin(1);
        let half_leg = GradedSpace::try_new(Arc::clone(&provider), [(half, 1)]).unwrap();
        let coupled_leg = GradedSpace::try_new(
            Arc::clone(&provider),
            [
                (SU2Irrep::from_twice_spin(0), 1),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
        )
        .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&runtime, [&half_leg, &half_leg], [&coupled_leg], 783)
                .unwrap();
        assert_eq!(source.subblock_count(), 2);
        let runtime_before = runtime.tree_transform_cache_info().structures;
        let (first, first_allocations, first_bytes, first_ns) =
            measure_value(|| source.permute(&[1, 0], &[2]).unwrap());
        let runtime_after_first = runtime.tree_transform_cache_info().structures;
        let mut repeat_ns = Vec::with_capacity(7);
        let mut repeat_allocations = Vec::with_capacity(7);
        let mut repeat_bytes = Vec::with_capacity(7);
        for _ in 0..7 {
            let (repeated, allocations, bytes, ns) =
                measure_value(|| source.permute(&[1, 0], &[2]).unwrap());
            assert_eq!(repeated.dense_data().unwrap(), first.dense_data().unwrap());
            repeat_ns.push(ns);
            repeat_allocations.push(allocations);
            repeat_bytes.push(bytes);
        }
        let runtime_after_repeat = runtime.tree_transform_cache_info().structures;
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
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let apply = |source: &TensorMap<SUNFusionRule, f64>| match operation {
        "permute" => source.permute(&[1, 0], &[2]).unwrap(),
        "braid" => source.braid(&[1, 0], &[2], &[0, 1, 2]).unwrap(),
        "repartition" => source.repartition(1).unwrap(),
        _ => unreachable!(),
    };

    let runtime_before = runtime.tree_transform_cache_info().structures;
    let (first, first_allocations, first_bytes, first_ns) = measure_value(|| apply(&source));
    let runtime_after_first = runtime.tree_transform_cache_info().structures;
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
    let runtime_after_repeat = runtime.tree_transform_cache_info().structures;
    let mut sorted_ns = repeat_ns.clone();
    sorted_ns.sort_unstable();
    println!(
        "case={case} phase=public_first_after_source_construction coefficient_caches=cold ns={first_ns} allocations={first_allocations} requested_bytes={first_bytes} runtime_before={runtime_before:?} runtime_after={runtime_after_first:?}"
    );
    println!(
        "case={case} phase=public_repeat_provider_warm samples_ns={repeat_ns:?} median_ns={} allocations={repeat_allocations:?} requested_bytes={repeat_bytes:?} runtime_after={runtime_after_repeat:?}",
        sorted_ns[sorted_ns.len() / 2]
    );
    assert!(runtime_after_first.entries() > runtime_before.entries());
    assert!(runtime_after_repeat.hits() > runtime_after_first.hits());
    assert!(std::ptr::eq(first.provider(), provider.as_ref()));
}
