//! Same-process cold/warm measurements for public basic tensor operations.

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use tenet::prelude::*;

#[derive(Clone, Copy)]
struct RuntimeCounters {
    entries: usize,
    bytes: usize,
    hits: usize,
    misses: usize,
    evictions: usize,
    bypasses: usize,
}

fn runtime_counters(runtime: &Runtime) -> RuntimeCounters {
    let info = runtime.tree_transform_cache_info();
    RuntimeCounters {
        entries: info.entries(),
        bytes: info.charged_payload_bytes(),
        hits: info.hits(),
        misses: info.misses(),
        evictions: info.evictions(),
        bypasses: info.admission_bypasses(),
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
    before: RuntimeCounters,
    after: RuntimeCounters,
) {
    println!(
        "{symmetry},{operation},{form},{phase},{iterations},{:.3},{},{},{},{},{},{},{},{},NA,NA,NA,NA,NA,NA,NA,NA",
        elapsed.as_secs_f64() * 1e6 / iterations as f64,
        after.hits - before.hits,
        after.misses - before.misses,
        after.evictions - before.evictions,
        after.bypasses - before.bypasses,
        delta(after.entries, before.entries),
        before.bytes,
        after.bytes,
        delta(after.bytes, before.bytes),
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
    let cold_before = runtime_counters(runtime);
    let cold_start = Instant::now();
    let cold_output = operation_fn()?;
    black_box(&cold_output);
    let cold_elapsed = cold_start.elapsed();
    let cold_after = runtime_counters(runtime);
    print_sample(
        symmetry,
        operation,
        form,
        first_phase,
        1,
        cold_elapsed,
        cold_before,
        cold_after,
    );

    black_box(operation_fn()?);
    black_box(operation_fn()?);
    let warm_before = runtime_counters(runtime);
    let warm_start = Instant::now();
    let mut iterations = 0;
    while iterations < 2 || warm_start.elapsed() < min_time {
        black_box(operation_fn()?);
        iterations += 1;
    }
    let warm_elapsed = warm_start.elapsed();
    let warm_after = runtime_counters(runtime);
    print_sample(
        symmetry,
        operation,
        form,
        repeated_phase,
        iterations,
        warm_elapsed,
        warm_before,
        warm_after,
    );
    Ok(cold_output)
}

fn run_permute(symmetry: &str, space: Space, min_time: Duration) -> Result<(), Error> {
    for form in ["owned", "destination"] {
        let runtime = Runtime::builder().build()?;
        let source = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space, &space], [&space], 724)?;
        if form == "owned" {
            let cold = bench(
                &runtime,
                symmetry,
                "permute",
                form,
                "cold",
                "warm",
                min_time,
                || source.permute(&[1], &[2, 0]),
            )?;
            assert!(cold.norm()?.is_finite());
        } else {
            let expected = source.permute(&[1], &[2, 0])?;
            let mut destination = expected.zeros_like()?;
            let mut context = TensorExecutionContext::for_runtime(&runtime)?;
            bench(
                &runtime,
                symmetry,
                "permute",
                form,
                "first_after_setup",
                "warm_after_setup",
                min_time,
                || {
                    context.permute_overwrite_into(
                        &mut destination,
                        &source,
                        &[1],
                        &[2, 0],
                        Scalar::F64(1.0),
                    )
                },
            )?;
            assert_eq!(destination.data(), expected.data());
        }
    }
    Ok(())
}

fn run_contract(symmetry: &str, space: Space, min_time: Duration) -> Result<(), Error> {
    for form in ["owned", "destination"] {
        let runtime = Runtime::builder().build()?;
        let lhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [&space, &space],
            [&space, &space],
            725,
        )?;
        let rhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [&space, &space],
            [&space, &space],
            726,
        )?;
        if form == "owned" {
            let cold = bench(
                &runtime,
                symmetry,
                "contract",
                form,
                "cold",
                "warm",
                min_time,
                || lhs.contract(&rhs, &[3, 2], &[0, 1]),
            )?;
            assert!(cold.norm()?.is_finite());
        } else {
            let expected = lhs.contract(&rhs, &[3, 2], &[0, 1])?;
            let mut destination = expected.zeros_like()?;
            let mut context = TensorExecutionContext::for_runtime(&runtime)?;
            bench(
                &runtime,
                symmetry,
                "contract",
                form,
                "first_after_setup",
                "warm_after_setup",
                min_time,
                || {
                    context.contract_overwrite_into(
                        &mut destination,
                        &lhs,
                        &rhs,
                        &[3, 2],
                        &[0, 1],
                        Scalar::F64(1.0),
                    )
                },
            )?;
            assert_eq!(destination.data(), expected.data());
        }
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let min_ms = std::env::var("OP_MATRIX_MIN_MS")
        .ok()
        .map(|value| value.parse().expect("OP_MATRIX_MIN_MS must be an integer"))
        .unwrap_or(20);
    println!(
        "# tenet_authority={}",
        std::env::var("TENET_AUTHORITY").unwrap_or_else(|_| "unknown".into())
    );
    println!(
        "# tenferro_authority={}",
        std::env::var("TENFERRO_AUTHORITY").unwrap_or_else(|_| "unknown".into())
    );
    println!(
        "# features=cpu-faer:{} cpu-blas:{} cuda:{} backend=faer(default RuntimeBuilder)",
        cfg!(feature = "cpu-faer"),
        cfg!(feature = "cpu-blas"),
        cfg!(feature = "cuda")
    );
    println!("# threads=RAYON_NUM_THREADS:{} OPENBLAS_NUM_THREADS:{} OMP_NUM_THREADS:{} MKL_NUM_THREADS:{}", env_or_unset("RAYON_NUM_THREADS"), env_or_unset("OPENBLAS_NUM_THREADS"), env_or_unset("OMP_NUM_THREADS"), env_or_unset("MKL_NUM_THREADS"));
    println!("# cold_scope=fresh Runtime tree-transform store; process-global interned structures may already be warm");
    println!("# unavailable_counters=output_allocation_bytes,operation_local_scratch_bytes,provider_queries,transform_passes,gemm_calls,host_device_transfers");
    println!("symmetry,operation,form,phase,iterations,us_per_iter,tree_hits,tree_misses,tree_evictions,tree_bypasses,tree_entries_delta,tree_charged_payload_bytes_before,tree_charged_payload_bytes_after,tree_charged_payload_bytes_delta,destination_preparations,destination_structural_comparisons,output_allocation_bytes,operation_local_scratch_bytes,provider_queries,transform_passes,gemm_calls,host_device_transfers");

    let min_time = Duration::from_millis(min_ms);
    for (name, space) in [
        ("U1", Space::u1([(-1, 2), (0, 2), (1, 2)])),
        ("SU2", Space::su2([(0, 2), (1, 2), (2, 2)])?),
    ] {
        run_permute(name, space.clone(), min_time)?;
        run_contract(name, space, min_time)?;
    }
    Ok(())
}

fn env_or_unset(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unset".into())
}
