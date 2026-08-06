use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::typed::{GradedSpace, Runtime, TensorMap};
use tenet_network::{configure_plan_cache, plan_cache_stats, tensor, PlanCacheConfig};

const ITERATIONS: usize = 20;

fn fixture(
    runtime: &Runtime,
) -> (
    GradedSpace<U1FusionRule>,
    TensorMap<U1FusionRule, f64>,
    TensorMap<U1FusionRule, f64>,
) {
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 8),
            (U1Irrep::new(0), 16),
            (U1Irrep::new(1), 8),
        ],
        false,
    )
    .expect("space");
    let a =
        TensorMap::rand_with_seed(runtime, [&space, &space], [&space, &space], 12401).expect("lhs");
    let b =
        TensorMap::rand_with_seed(runtime, [&space, &space], [&space, &space], 12402).expect("rhs");
    (space, a, b)
}

fn assert_matches_oracle(
    actual: &TensorMap<U1FusionRule, f64>,
    oracle: &TensorMap<U1FusionRule, f64>,
) {
    assert_eq!(actual.data(), oracle.data());
    assert_eq!(actual.codomain(), oracle.codomain());
    assert_eq!(actual.domain(), oracle.domain());
}

fn print_row(cache: &str, phase: &str, iterations: usize, elapsed_us: f64, runtime: &Runtime) {
    let stats = plan_cache_stats(runtime);
    println!(
        "{cache},{phase},{iterations},{:.3},{},{},{},{},{},{},{},{},{},{},{}",
        elapsed_us / iterations as f64,
        stats.entries,
        stats.hits,
        stats.misses,
        stats.workspaces_created,
        stats.workspace_reuses,
        stats.idle_workspaces,
        stats.retained_workspace_bytes,
        stats.peak_retained_workspace_bytes,
        stats.workspace_byte_admissions,
        stats.workspace_byte_rejections,
        stats.workspace_byte_evictions,
    );
}

fn run_case(cache: &str, enabled: bool, oracle: &TensorMap<U1FusionRule, f64>) {
    let runtime = Runtime::builder().build().expect("runtime");
    if !enabled {
        configure_plan_cache(
            &runtime,
            PlanCacheConfig {
                enabled: false,
                ..Default::default()
            },
        );
    }
    let (_, a, b) = fixture(&runtime);

    let cold_start = Instant::now();
    let cold = black_box(tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).expect("cold"));
    let cold_us = cold_start.elapsed().as_secs_f64() * 1e6;
    assert_matches_oracle(&cold, oracle);
    print_row(cache, "cold", 1, cold_us, &runtime);
    let cold_stats = plan_cache_stats(&runtime);
    if enabled {
        assert_eq!(
            (cold_stats.misses, cold_stats.hits, cold_stats.entries),
            (1, 0, 1)
        );
        assert!(cold_stats.retained_workspace_bytes > 0);
    } else {
        assert_eq!(cold_stats, Default::default());
    }

    for _ in 0..2 {
        black_box(tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).expect("warmup"));
    }
    let warm_start = Instant::now();
    let mut warm_output = None;
    for _ in 0..ITERATIONS {
        warm_output = Some(black_box(
            tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).expect("warm"),
        ));
    }
    let warm_us = warm_start.elapsed().as_secs_f64() * 1e6;
    assert_matches_oracle(warm_output.as_ref().expect("warm output"), oracle);
    print_row(cache, "warm", ITERATIONS, warm_us, &runtime);

    let stats = plan_cache_stats(&runtime);
    if enabled {
        assert!(stats.retained_workspace_bytes > 0);
        assert!(stats.workspace_reuses > 0);
    } else {
        assert_eq!(stats, Default::default());
    }
}

fn main() {
    // Keep the oracle isolated from every measured runtime and its plan cache.
    let oracle_runtime = Runtime::builder().build().expect("oracle runtime");
    let (_, oracle_a, oracle_b) = fixture(&oracle_runtime);
    let oracle = oracle_a
        .contract(&oracle_b, &[2, 3], &[0, 1], &[0, 1, 2, 3])
        .expect("oracle");

    println!(
        "cache,phase,iterations,us_per_iter,entries,hits,misses,workspaces_created,workspace_reuses,idle_workspaces,retained_workspace_bytes,peak_retained_workspace_bytes,workspace_byte_admissions,workspace_byte_rejections,workspace_byte_evictions"
    );
    run_case("enabled", true, &oracle);
    run_case("disabled", false, &oracle);
}
