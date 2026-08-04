//! Same-process cold/warm measurements for public basic tensor operations.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
    time::{Duration, Instant},
};

use tenet::prelude::*;

use tenet::core::{
    complete_hom_space_structure_cache_info, fusion_tree_layout_cache_info,
    CompleteHomSpaceStructureCacheInfo, FusionTreeLayoutCacheInfo,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_CALLS: Cell<usize> = const { Cell::new(0) };
    static REQUESTED_BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATION_CALLS.set(ALLOCATION_CALLS.get() + 1);
            REQUESTED_BYTES.set(REQUESTED_BYTES.get() + layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATION_CALLS.set(ALLOCATION_CALLS.get() + 1);
            REQUESTED_BYTES.set(REQUESTED_BYTES.get() + new_size);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy)]
struct Allocations {
    calls: usize,
    requested_bytes: usize,
}

fn measure_allocations<T>(
    operation: impl FnOnce() -> Result<T, Error>,
) -> Result<(T, Allocations), Error> {
    ALLOCATION_CALLS.set(0);
    REQUESTED_BYTES.set(0);
    COUNTING.set(true);
    let result = operation();
    COUNTING.set(false);
    result.map(|value| {
        (
            value,
            Allocations {
                calls: ALLOCATION_CALLS.get(),
                requested_bytes: REQUESTED_BYTES.get(),
            },
        )
    })
}

#[derive(Clone, Copy)]
struct Counters {
    runtime: RuntimeTreeTransformCacheInfo,
    fusion_layout: FusionTreeLayoutCacheInfo,
    complete_hom: CompleteHomSpaceStructureCacheInfo,
}

fn counters(runtime: &Runtime) -> Counters {
    Counters {
        runtime: runtime.tree_transform_cache_info(),
        fusion_layout: fusion_tree_layout_cache_info(),
        complete_hom: complete_hom_space_structure_cache_info(),
    }
}

fn delta(after: usize, before: usize) -> isize {
    after as isize - before as isize
}

fn print_sample(
    symmetry: &str,
    operation: &str,
    form: &str,
    phase: &str,
    iterations: u64,
    elapsed: Duration,
    allocations: Allocations,
    before: Counters,
    after: Counters,
) {
    let tree_before = before.runtime;
    let tree_after = after.runtime;
    let layout_before = before.fusion_layout;
    let layout_after = after.fusion_layout;
    let hom_before = before.complete_hom;
    let hom_after = after.complete_hom;
    println!(
        "{symmetry},{operation},{form},{phase},{iterations},{us:.3},\
         {tree_hits},{tree_misses},{tree_evictions},{tree_bypasses},{tree_entries_delta},\
         {tree_bytes_before},{tree_bytes_after},{tree_bytes_delta},\
         {layout_misses},{layout_evictions},{layout_bypasses},{layout_entries_delta},\
         {layout_bytes_before},{layout_bytes_after},{layout_bytes_delta},\
         {hom_hits},{hom_misses},{hom_admissions},{hom_evictions},{hom_bypasses},\
         {hom_entries_delta},{hom_bytes_before},{hom_bytes_after},{hom_bytes_delta},\
         NA,{allocation_calls},{requested_bytes},NA,NA,NA,NA,NA",
        us = elapsed.as_secs_f64() * 1e6 / iterations as f64,
        tree_hits = tree_after.hits() - tree_before.hits(),
        tree_misses = tree_after.misses() - tree_before.misses(),
        tree_evictions = tree_after.evictions() - tree_before.evictions(),
        tree_bypasses = tree_after.admission_bypasses() - tree_before.admission_bypasses(),
        tree_entries_delta = delta(tree_after.entries(), tree_before.entries()),
        tree_bytes_before = tree_before.charged_payload_bytes(),
        tree_bytes_after = tree_after.charged_payload_bytes(),
        tree_bytes_delta = delta(
            tree_after.charged_payload_bytes(),
            tree_before.charged_payload_bytes(),
        ),
        layout_misses = layout_after.misses() - layout_before.misses(),
        layout_evictions = layout_after.evictions() - layout_before.evictions(),
        layout_bypasses = layout_after.admission_bypasses() - layout_before.admission_bypasses(),
        layout_entries_delta = delta(layout_after.entries(), layout_before.entries()),
        layout_bytes_before = layout_before.charged_payload_bytes(),
        layout_bytes_after = layout_after.charged_payload_bytes(),
        layout_bytes_delta = delta(
            layout_after.charged_payload_bytes(),
            layout_before.charged_payload_bytes(),
        ),
        hom_hits = hom_after.hits() - hom_before.hits(),
        hom_misses = hom_after.misses() - hom_before.misses(),
        hom_admissions = hom_after.admissions() - hom_before.admissions(),
        hom_evictions = hom_after.evictions() - hom_before.evictions(),
        hom_bypasses = hom_after.bypasses() - hom_before.bypasses(),
        hom_entries_delta = delta(hom_after.entries(), hom_before.entries()),
        hom_bytes_before = hom_before.charged_bytes(),
        hom_bytes_after = hom_after.charged_bytes(),
        hom_bytes_delta = delta(hom_after.charged_bytes(), hom_before.charged_bytes()),
        allocation_calls = allocations.calls,
        requested_bytes = allocations.requested_bytes,
    );
}

fn bench<T>(
    runtime: &Runtime,
    symmetry: &str,
    operation: &str,
    form: &str,
    first_phase: &str,
    repeated_phase: &str,
    min_time: Duration,
    mut operation_fn: impl FnMut() -> Result<T, Error>,
) -> Result<T, Error> {
    let cold_before = counters(runtime);
    let cold_start = Instant::now();
    let (cold_output, cold_allocations) = measure_allocations(&mut operation_fn)?;
    black_box(&cold_output);
    let cold_elapsed = cold_start.elapsed();
    let cold_after = counters(runtime);
    print_sample(
        symmetry,
        operation,
        form,
        first_phase,
        1,
        cold_elapsed,
        cold_allocations,
        cold_before,
        cold_after,
    );

    black_box(operation_fn()?);
    black_box(operation_fn()?);
    let warm_before = counters(runtime);
    let warm_start = Instant::now();
    let mut iterations = 0;
    let (_, warm_allocations) = measure_allocations(|| {
        while iterations < 2 || warm_start.elapsed() < min_time {
            black_box(operation_fn()?);
            iterations += 1;
        }
        Ok(())
    })?;
    let warm_elapsed = warm_start.elapsed();
    let warm_after = counters(runtime);
    print_sample(
        symmetry,
        operation,
        form,
        repeated_phase,
        iterations,
        warm_elapsed,
        warm_allocations,
        warm_before,
        warm_after,
    );
    if let Ok(milliseconds) = std::env::var("OP_MATRIX_PROFILE_PAUSE_MS") {
        std::thread::sleep(Duration::from_millis(
            milliseconds
                .parse()
                .expect("OP_MATRIX_PROFILE_PAUSE_MS must be an integer"),
        ));
    }
    Ok(cold_output)
}

fn benchmark_runtime() -> Result<Runtime, Error> {
    let backend = match std::env::var("OP_MATRIX_GEMM_BACKEND").as_deref() {
        Ok("blas") => LinalgBackend::Blas,
        Ok("faer") | Err(_) => LinalgBackend::Faer,
        Ok(other) => {
            return Err(Error::InvalidArgument(format!(
                "OP_MATRIX_GEMM_BACKEND must be `faer` or `blas`, got `{other}`"
            )))
        }
    };
    Runtime::builder()
        .dense_threads(1)
        .gemm_backend(backend)
        .build()
}

fn operation_enabled(operation: &str) -> bool {
    std::env::var("OP_MATRIX_OPERATION").map_or(true, |selected| selected == operation)
}

fn form_enabled(form: &str) -> bool {
    std::env::var("OP_MATRIX_FORM").map_or(true, |selected| selected == form)
}

macro_rules! run_provider {
    ($symmetry:literal, $rule:ty, $space:expr, $min_time:expr) => {{
        let space = $space;
        if operation_enabled("permute") {
            for form in ["owned", "destination"] {
                if !form_enabled(form) {
                    continue;
                }
                let runtime = benchmark_runtime()?;
                let source = TensorMap::<$rule, f64>::rand_with_seed(
                    &runtime,
                    [&space, &space],
                    [&space],
                    724,
                )?;
                if form == "owned" {
                    let cold = bench(
                        &runtime,
                        $symmetry,
                        "permute",
                        form,
                        "cold",
                        "warm",
                        $min_time,
                        || source.permute(&[1], &[2, 0]),
                    )?;
                    assert!(cold.norm()?.is_finite());
                } else {
                    let expected = source.permute(&[1], &[2, 0])?;
                    let mut destination = expected.zeros_like();
                    bench(
                        &runtime,
                        $symmetry,
                        "permute",
                        form,
                        "first_after_setup",
                        "warm_after_setup",
                        $min_time,
                        || source.permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0),
                    )?;
                    assert_eq!(destination.data(), expected.data());
                }
            }
        }

        if operation_enabled("compose") && form_enabled("owned") {
            let runtime = benchmark_runtime()?;
            let lhs = TensorMap::<$rule, f64>::rand_with_seed(
                &runtime,
                [&space, &space],
                [&space, &space],
                725,
            )?;
            let rhs = TensorMap::<$rule, f64>::rand_with_seed(
                &runtime,
                [&space, &space],
                [&space, &space],
                726,
            )?;
            let composed = bench(
                &runtime,
                $symmetry,
                "compose",
                "owned",
                "cold",
                "warm",
                $min_time,
                || lhs.compose(&rhs),
            )?;
            let contracted = lhs.contract(&rhs, &[2, 3], &[0, 1], &[0, 1, 2, 3])?;
            assert_eq!(composed.data(), contracted.data());
        }

        for (operation, lhs_axes, rhs_axes, output_axes) in [
            (
                "contract_identity",
                &[2, 3][..],
                &[0, 1][..],
                &[0, 1, 2, 3][..],
            ),
            (
                "contract_input_swap",
                &[3, 2][..],
                &[0, 1][..],
                &[0, 1, 2, 3][..],
            ),
            (
                "contract_input_output_swap",
                &[3, 2][..],
                &[0, 1][..],
                &[1, 0, 2, 3][..],
            ),
        ] {
            if !operation_enabled(operation) {
                continue;
            }
            for form in ["owned", "destination"] {
                if !form_enabled(form) {
                    continue;
                }
                let runtime = benchmark_runtime()?;
                let lhs = TensorMap::<$rule, f64>::rand_with_seed(
                    &runtime,
                    [&space, &space],
                    [&space, &space],
                    725,
                )?;
                let rhs = TensorMap::<$rule, f64>::rand_with_seed(
                    &runtime,
                    [&space, &space],
                    [&space, &space],
                    726,
                )?;
                if form == "owned" {
                    let cold = bench(
                        &runtime,
                        $symmetry,
                        operation,
                        form,
                        "cold",
                        "warm",
                        $min_time,
                        || lhs.contract(&rhs, lhs_axes, rhs_axes, output_axes),
                    )?;
                    assert!(cold.norm()?.is_finite());
                } else {
                    let expected = lhs.contract(&rhs, lhs_axes, rhs_axes, output_axes)?;
                    let mut destination = expected.zeros_like();
                    bench(
                        &runtime,
                        $symmetry,
                        operation,
                        form,
                        "first_after_setup",
                        "warm_after_setup",
                        $min_time,
                        || {
                            lhs.contract_overwrite_into(
                                &rhs,
                                &mut destination,
                                lhs_axes,
                                rhs_axes,
                                output_axes,
                                1.0,
                            )
                        },
                    )?;
                    assert_eq!(destination.data(), expected.data());
                }
            }
        }
    }};
}

fn main() -> Result<(), Error> {
    if let Ok(operation) = std::env::var("OP_MATRIX_OPERATION") {
        if !matches!(
            operation.as_str(),
            "permute"
                | "compose"
                | "contract_identity"
                | "contract_input_swap"
                | "contract_input_output_swap"
        ) {
            return Err(Error::InvalidArgument(format!(
                "unknown OP_MATRIX_OPERATION `{operation}`"
            )));
        }
    }
    if let Ok(form) = std::env::var("OP_MATRIX_FORM") {
        if !matches!(form.as_str(), "owned" | "destination") {
            return Err(Error::InvalidArgument(format!(
                "unknown OP_MATRIX_FORM `{form}`"
            )));
        }
    }
    let min_ms = std::env::var("OP_MATRIX_MIN_MS")
        .ok()
        .map(|value| value.parse().expect("OP_MATRIX_MIN_MS must be an integer"))
        .unwrap_or(20);
    let degeneracy = std::env::var("OP_MATRIX_DEGENERACY")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("OP_MATRIX_DEGENERACY must be an integer")
        })
        .unwrap_or(8);
    println!(
        "# tenet_authority={}",
        std::env::var("TENET_AUTHORITY").unwrap_or_else(|_| "unknown".into())
    );
    println!(
        "# tenferro_authority={}",
        std::env::var("TENFERRO_AUTHORITY").unwrap_or_else(|_| "unknown".into())
    );
    println!(
        "# features=cpu-faer:{} blas-provider:{} cuda:{} gemm_backend={}",
        cfg!(feature = "cpu-faer"),
        cfg!(any(
            feature = "cpu-blas",
            feature = "blas-accelerate",
            feature = "blas-openblas",
            feature = "blas-mkl"
        )),
        cfg!(feature = "cuda"),
        std::env::var("OP_MATRIX_GEMM_BACKEND").unwrap_or_else(|_| "faer".into())
    );
    println!("# degeneracy={degeneracy}");
    println!("# threads=RAYON_NUM_THREADS:{} OPENBLAS_NUM_THREADS:{} OMP_NUM_THREADS:{} MKL_NUM_THREADS:{}", env_or_unset("RAYON_NUM_THREADS"), env_or_unset("OPENBLAS_NUM_THREADS"), env_or_unset("OMP_NUM_THREADS"), env_or_unset("MKL_NUM_THREADS"));
    println!("# cold_scope=fresh Runtime tree-transform store; process-global interned structures may already be warm");
    println!("# allocation_scope=caller-thread Rust allocation calls and requested bytes during the measured phase; excludes worker threads, native BLAS allocation, frees, and peak/live bytes");
    println!("# unavailable_counters=exact_layout_admission,operation_local_scratch_bytes,provider_queries,transform_passes,gemm_calls,host_device_transfers");
    println!("symmetry,operation,form,phase,iterations,us_per_iter,tree_hits,tree_misses,tree_evictions,tree_bypasses,tree_entries_delta,tree_charged_payload_bytes_before,tree_charged_payload_bytes_after,tree_charged_payload_bytes_delta,fusion_layout_misses,fusion_layout_evictions,fusion_layout_bypasses,fusion_layout_entries_delta,fusion_layout_charged_payload_bytes_before,fusion_layout_charged_payload_bytes_after,fusion_layout_charged_payload_bytes_delta,complete_hom_hits,complete_hom_misses,complete_hom_admissions,complete_hom_evictions,complete_hom_bypasses,complete_hom_entries_delta,complete_hom_charged_bytes_before,complete_hom_charged_bytes_after,complete_hom_charged_bytes_delta,exact_layout_admission,caller_allocation_calls,caller_requested_allocation_bytes,operation_local_scratch_bytes,provider_queries,transform_passes,gemm_calls,host_device_transfers");

    let min_time = Duration::from_millis(min_ms);
    run_provider!(
        "U1",
        U1FusionRule,
        GradedSpace::try_new_owned(
            U1FusionRule,
            [
                (U1Irrep::new(-1), degeneracy),
                (U1Irrep::new(0), degeneracy),
                (U1Irrep::new(1), degeneracy),
            ],
            false,
        )?,
        min_time
    );
    run_provider!(
        "SU2",
        SU2FusionRule,
        GradedSpace::try_new_owned(
            SU2FusionRule,
            [
                (SU2Irrep::from_twice_spin(0), degeneracy),
                (SU2Irrep::from_twice_spin(1), degeneracy),
                (SU2Irrep::from_twice_spin(2), degeneracy),
            ],
            false,
        )?,
        min_time
    );
    Ok(())
}

fn env_or_unset(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unset".into())
}
