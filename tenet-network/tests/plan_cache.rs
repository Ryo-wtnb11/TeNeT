use std::sync::{Arc, Barrier};

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::typed::{GradedSpace, Runtime, TensorMap};
use tenet_network::{
    clear_plan_cache, configure_plan_cache, load_plan_cache, plan_cache_stats, save_plan_cache,
    tensor, PlanCacheConfig, ReplanPolicy,
};

fn space(provider: Arc<U1FusionRule>, dim: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_shared(provider, [(U1Irrep::new(0), dim)]).unwrap()
}

fn pair(
    runtime: &Runtime,
    space: &GradedSpace<U1FusionRule>,
    seed: u64,
) -> (TensorMap<U1FusionRule, f64>, TensorMap<U1FusionRule, f64>) {
    (
        TensorMap::rand_with_seed(runtime, [space], [space], seed).unwrap(),
        TensorMap::rand_with_seed(runtime, [space], [space], seed + 1).unwrap(),
    )
}

#[test]
fn typed_static_cache_preserves_hit_clear_and_workspace_stats() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = space(provider, 3);
    let (a, b) = pair(&runtime, &space, 10);

    let first = tensor!([i; k] = a[i; j] * b[j; k]).unwrap();
    let cold = plan_cache_stats(&runtime);
    assert_eq!((cold.misses, cold.hits, cold.entries), (1, 0, 1));
    assert_eq!(cold.topology_materializations, 1);
    assert_eq!(cold.workspaces_created, 1);
    assert_eq!(cold.idle_workspaces, 1);
    assert!(cold.retained_workspace_bytes > 0);
    assert_eq!(
        cold.peak_retained_workspace_bytes,
        cold.retained_workspace_bytes
    );
    assert_eq!(cold.workspace_byte_admissions, 1);
    assert_eq!(cold.workspace_byte_rejections, 0);
    #[allow(deprecated)]
    {
        assert_eq!(cold.dynamic_aliases, 0);
    }

    let second = tensor!([i; k] = a[i; j] * b[j; k]).unwrap();
    let warm = plan_cache_stats(&runtime);
    assert_eq!(first.data(), second.data());
    assert_eq!((warm.misses, warm.hits, warm.entries), (1, 1, 1));
    assert_eq!(warm.topology_materializations, 1);
    assert_eq!(warm.workspace_reuses, 1);

    clear_plan_cache(&runtime);
    assert_eq!(plan_cache_stats(&runtime), Default::default());
}

#[test]
fn parked_host_workspace_releases_provider_and_rebinds_exact_current_arc() {
    let runtime = Runtime::builder().build().unwrap();
    let first_provider = Arc::new(U1FusionRule);
    let first_weak = Arc::downgrade(&first_provider);
    let first_space = space(Arc::clone(&first_provider), 2);
    let (a, b) = pair(&runtime, &first_space, 20);
    drop(tensor!([i; k] = a[i; j] * b[j; k]).unwrap());
    drop((a, b, first_space, first_provider));
    assert!(first_weak.upgrade().is_none());

    let current_provider = Arc::new(U1FusionRule);
    let current_space = space(Arc::clone(&current_provider), 2);
    let (c, d) = pair(&runtime, &current_space, 30);
    let output = tensor!([i; k] = c[i; j] * d[j; k]).unwrap();
    assert!(std::ptr::eq(output.provider(), current_provider.as_ref()));
    assert_eq!(plan_cache_stats(&runtime).workspace_reuses, 1);
}

#[test]
fn trace_lowering_and_cache_release_provider_authority() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let weak = Arc::downgrade(&provider);
    let traced_space = space(Arc::clone(&provider), 2);
    let tensor = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        &runtime,
        [&traced_space],
        [&traced_space],
        35,
    )
    .unwrap();
    drop(tensor!([] = tensor[i; i]).unwrap());
    drop((tensor, traced_space, provider));
    assert!(weak.upgrade().is_none());
}

#[test]
fn dimension_drift_replans_without_put_before_residency_recheck() {
    let runtime = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            replan: ReplanPolicy::DriftFactor(1.5),
            ..Default::default()
        },
    );
    let provider = Arc::new(U1FusionRule);
    let small = space(Arc::clone(&provider), 2);
    let large = space(provider, 5);
    let (a, b) = pair(&runtime, &small, 40);
    drop(tensor!([i; k] = a[i; j] * b[j; k]).unwrap());
    let (c, d) = pair(&runtime, &large, 50);
    drop(tensor!([i; k] = c[i; j] * d[j; k]).unwrap());
    let stats = plan_cache_stats(&runtime);
    assert_eq!((stats.misses, stats.replans, stats.entries), (1, 1, 1));
}

#[test]
fn disabled_cache_keeps_entries_and_counters_empty() {
    let runtime = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            enabled: false,
            ..Default::default()
        },
    );
    let space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&runtime, &space, 60);
    drop(tensor!([i; k] = a[i; j] * b[j; k]).unwrap());
    drop(tensor!([i; k] = a[i; j] * b[j; k]).unwrap());
    assert_eq!(plan_cache_stats(&runtime), Default::default());
}

#[test]
fn lru_hit_touches_entry_before_capacity_eviction() {
    let runtime = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            capacity: 2,
            ..Default::default()
        },
    );
    let space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&runtime, &space, 70);
    drop(tensor!([i; k] = a[i; j] * b[j; k]).unwrap());
    drop(tensor!([k; i] = a[i; j] * b[j; k]).unwrap());
    drop(tensor!([i; k] = a[i; j] * b[j; k]).unwrap());
    drop(tensor!([k; i] = b[j; k] * a[i; j]).unwrap());
    let before = plan_cache_stats(&runtime);
    drop(tensor!([i; k] = a[i; j] * b[j; k]).unwrap());
    assert_eq!(plan_cache_stats(&runtime).hits, before.hits + 1);
    drop(tensor!([k; i] = a[i; j] * b[j; k]).unwrap());
    let after = plan_cache_stats(&runtime);
    assert_eq!(after.misses, before.misses + 1);
    assert_eq!(after.entries, 2);
}

#[test]
fn persisted_typed_order_roundtrips_into_a_fresh_runtime() {
    let first = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&first, ""), 0);
    let first_space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&first, &first_space, 80);
    let expected = tensor!([i; k] = a[i; j] * b[j; k]).unwrap();
    let saved = save_plan_cache(&first);
    assert!(saved.contains("TOPO "));

    let second = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&second, &saved), 1);
    let second_space = space(Arc::new(U1FusionRule), 2);
    let (c, d) = pair(&second, &second_space, 80);
    let actual = tensor!([i; k] = c[i; j] * d[j; k]).unwrap();
    assert_eq!(actual.data(), expected.data());
}

#[test]
fn concurrent_macro_calls_share_one_plan_and_bound_idle_pool() {
    let runtime = Runtime::builder().build().unwrap();
    let space = space(Arc::new(U1FusionRule), 8);
    let (a, b) = pair(&runtime, &space, 90);
    let expected = a.contract(&b, &[1], &[0], &[0, 1]).unwrap();
    let barrier = Arc::new(Barrier::new(8));
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..8 {
            let barrier = Arc::clone(&barrier);
            let a = &a;
            let b = &b;
            handles.push(scope.spawn(move || {
                barrier.wait();
                tensor!([i; k] = a[i; j] * b[j; k]).unwrap()
            }));
        }
        for handle in handles {
            assert_eq!(handle.join().unwrap().data(), expected.data());
        }
    });
    let stats = plan_cache_stats(&runtime);
    assert_eq!(stats.entries, 1);
    assert!(stats.idle_workspaces <= 2);
    assert!(stats.workspaces_created >= stats.idle_workspaces as u64);
}

#[test]
fn failed_execution_quarantines_lease_and_next_valid_call_rebuilds() {
    let runtime = Runtime::builder().build().unwrap();
    let other = Runtime::builder().build().unwrap();
    let local_space = space(Arc::new(U1FusionRule), 2);
    let other_space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&runtime, &local_space, 100);
    let (_, foreign) = pair(&other, &other_space, 110);
    drop(tensor!([i; k] = a[i; j] * b[j; k]).unwrap());
    assert!(tensor!([i; k] = a[i; j] * foreign[j; k]).is_err());
    drop(tensor!([i; k] = a[i; j] * b[j; k]).unwrap());
    let stats = plan_cache_stats(&runtime);
    assert_eq!(stats.workspaces_created, 2);
    assert_eq!(stats.workspace_reuses, 1);
    assert_eq!(stats.idle_workspaces, 1);
    assert_eq!(stats.workspace_byte_rejections, 0);
}

#[test]
fn workspace_budget_admits_or_rejects_the_complete_idle_workspace() {
    let runtime = Runtime::builder().build().unwrap();
    let space = space(Arc::new(U1FusionRule), 8);
    let (a, b) = pair(&runtime, &space, 130);
    let (c, _) = pair(&runtime, &space, 132);

    drop(tensor!([i; l] = a[i; j] * b[j; k] * c[k; l]).unwrap());
    let charge = plan_cache_stats(&runtime).retained_workspace_bytes;
    assert!(charge > 8 * 8 * std::mem::size_of::<f64>());

    clear_plan_cache(&runtime);
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            workspace_budget_bytes: charge - 1,
            ..Default::default()
        },
    );
    drop(tensor!([i; l] = a[i; j] * b[j; k] * c[k; l]).unwrap());
    let rejected = plan_cache_stats(&runtime);
    assert_eq!(rejected.retained_workspace_bytes, 0);
    assert_eq!(rejected.peak_retained_workspace_bytes, 0);
    assert_eq!(rejected.idle_workspaces, 0);
    assert_eq!(rejected.workspace_byte_admissions, 0);
    assert_eq!(rejected.workspace_byte_rejections, 1);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            workspace_budget_bytes: charge,
            ..Default::default()
        },
    );
    drop(tensor!([i; l] = a[i; j] * b[j; k] * c[k; l]).unwrap());
    let admitted = plan_cache_stats(&runtime);
    assert_eq!(admitted.retained_workspace_bytes, charge);
    assert_eq!(admitted.peak_retained_workspace_bytes, charge);
    assert_eq!(admitted.idle_workspaces, 1);
    assert_eq!(admitted.workspace_byte_admissions, 1);
    assert_eq!(admitted.workspace_byte_rejections, 1);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            workspace_budget_bytes: 0,
            ..Default::default()
        },
    );
    let zeroed = plan_cache_stats(&runtime);
    assert_eq!(zeroed.retained_workspace_bytes, 0);
    assert_eq!(zeroed.idle_workspaces, 0);
    assert_eq!(zeroed.workspace_byte_evictions, 1);
    drop(tensor!([i; l] = a[i; j] * b[j; k] * c[k; l]).unwrap());
    assert_eq!(plan_cache_stats(&runtime).workspace_byte_rejections, 2);
}

#[test]
fn configuration_flushes_and_capacity_shrink_release_synchronously() {
    let runtime = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            capacity: 2,
            ..Default::default()
        },
    );
    let space = space(Arc::new(U1FusionRule), 8);
    let (a, b) = pair(&runtime, &space, 140);
    let (c, _) = pair(&runtime, &space, 142);
    drop(tensor!([i; l] = a[i; j] * b[j; k] * c[k; l]).unwrap());
    drop(tensor!([l; i] = a[i; j] * b[j; k] * c[k; l]).unwrap());
    let initial = plan_cache_stats(&runtime);
    assert_eq!((initial.entries, initial.idle_workspaces), (2, 2));
    let two_workspace_budget = initial.retained_workspace_bytes;

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            capacity: 2,
            workspace_budget_bytes: two_workspace_budget,
            ..Default::default()
        },
    );
    let flushed = plan_cache_stats(&runtime);
    assert_eq!(flushed.entries, 2);
    assert_eq!(flushed.idle_workspaces, 0);
    assert_eq!(flushed.retained_workspace_bytes, 0);
    assert_eq!(flushed.workspace_byte_evictions, 2);

    drop(tensor!([i; l] = a[i; j] * b[j; k] * c[k; l]).unwrap());
    drop(tensor!([l; i] = a[i; j] * b[j; k] * c[k; l]).unwrap());
    let repopulated = plan_cache_stats(&runtime);
    assert_eq!(repopulated.idle_workspaces, 2);
    assert_eq!(repopulated.retained_workspace_bytes, two_workspace_budget);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            capacity: 1,
            workspace_budget_bytes: two_workspace_budget,
            ..Default::default()
        },
    );
    let shrunk = plan_cache_stats(&runtime);
    assert_eq!(shrunk.entries, 1);
    assert_eq!(shrunk.idle_workspaces, 1);
    assert!(shrunk.retained_workspace_bytes < repopulated.retained_workspace_bytes);
    assert_eq!(shrunk.workspace_byte_evictions, 3);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            enabled: false,
            capacity: 1,
            workspace_budget_bytes: two_workspace_budget,
            ..Default::default()
        },
    );
    let disabled = plan_cache_stats(&runtime);
    assert_eq!((disabled.entries, disabled.idle_workspaces), (0, 0));
    assert_eq!(disabled.retained_workspace_bytes, 0);
    assert_eq!(disabled.workspace_byte_evictions, 4);
}

#[test]
fn dropping_warm_runtime_breaks_the_cache_workspace_cycle() {
    let runtime = Runtime::builder().build().unwrap();
    let identity = runtime.identity();
    let space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&runtime, &space, 120);
    let output = tensor!([i; k] = a[i; j] * b[j; k]).unwrap();
    drop((output, a, b, space, runtime));
    assert!(!identity.is_alive());
}
