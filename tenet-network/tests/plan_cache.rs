//! The topology-keyed plan cache behind `tensor!` / `Network::contract`.
//! The cache is per-Runtime and every #[test] builds its own runtime, so
//! counters start from zero in each test.

use tenet::prelude::*;
use tenet_network::{
    clear_plan_cache, configure_plan_cache, plan_cache_config, plan_cache_stats, tensor, Optimizer,
    PlanCacheConfig, ReplanPolicy,
};

fn assert_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len());
    for (a, b) in lhs.iter().zip(rhs) {
        assert!(
            (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
            "{a} vs {b}"
        );
    }
}

fn chain(rt: &Runtime, dim: usize, seed: u64) -> (Tensor, Tensor) {
    let v = Space::u1([(-1, dim), (0, dim), (1, dim)]);
    let a = Tensor::rand_with_seed(rt, Dtype::F64, [&v, &v], [&v, &v], seed).unwrap();
    let b = Tensor::rand_with_seed(rt, Dtype::F64, [&v, &v], [&v, &v], seed + 1).unwrap();
    (a, b)
}

#[test]
fn second_identical_call_hits() {
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 301);

    let first = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.hits, stats.misses, stats.entries), (0, 1, 1));

    let second = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.hits, stats.misses, stats.entries), (1, 1, 1));
    assert_close(first.data(), second.data(), 0.0);

    clear_plan_cache(&rt);
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.hits, stats.misses, stats.entries), (0, 0, 0));
}

/// The standard macro path materializes topology and grows a slot table once,
/// then keeps both counters unchanged across warm executions.
#[test]
fn warm_macro_path_avoids_repeated_structural_materialization() {
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 305);

    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let cold = plan_cache_stats(&rt);
    assert_eq!(cold.topology_materializations, 1);
    assert_eq!(cold.workspaces_created, 1);
    assert_eq!(cold.workspace_slot_grows, 1);
    assert_eq!(cold.workspace_reuses, 0);
    assert_eq!(cold.dynamic_aliases, 0);

    for _ in 0..8 {
        let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    }
    let warm = plan_cache_stats(&rt);
    assert_eq!(warm.topology_materializations, 1);
    assert_eq!(warm.workspaces_created, 1);
    assert_eq!(warm.workspace_slot_grows, 1);
    assert_eq!(warm.workspace_reuses, 8);
    assert_eq!(warm.dynamic_aliases, 0);
}

/// Reusing one `Network` instance also bypasses owned topology and dimension
/// snapshots before the full cache lookup.
#[test]
fn warm_network_contract_with_uses_identity_alias() {
    use tenet_network::{NetOperand, Network};

    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 306);
    let operands = [
        NetOperand {
            tensor: &a,
            conj: false,
            labels: &["i", "j", "k", "l"],
            codomain_split: Some(2),
        },
        NetOperand {
            tensor: &b,
            conj: false,
            labels: &["k", "l", "m", "n"],
            codomain_split: Some(2),
        },
    ];
    let network = Network::from_names(&operands, &["i", "j", "m", "n"], Some(2)).unwrap();
    let tensors = [&a, &b];

    let _ = network.contract_with(&tensors, &Optimizer::Greedy).unwrap();
    let _ = network.contract_with(&tensors, &Optimizer::Greedy).unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!(stats.topology_materializations, 1);
    assert_eq!(stats.workspaces_created, 1);
    assert_eq!(stats.workspace_slot_grows, 1);
    assert_eq!(stats.workspace_reuses, 1);
    assert_eq!(stats.dynamic_aliases, 1);
}

/// A failed cached execution returns its leased workspace before the next
/// call, so error paths do not force another slot-table growth.
#[test]
fn failed_execution_returns_workspace_to_cache() {
    let rt = Runtime::builder().build().unwrap();
    let other_rt = Runtime::builder().build().unwrap();
    let small = Space::u1([(0, 1)]);
    let large = Space::u1([(0, 16)]);
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&small], [&small], 307).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&small], [&small], 308).unwrap();
    let foreign = Tensor::rand_with_seed(&other_rt, Dtype::F64, [&small], [&small], 309).unwrap();
    let c = Tensor::rand_with_seed(&rt, Dtype::F64, [&small], [&large], 310).unwrap();
    let d = Tensor::rand_with_seed(&rt, Dtype::F64, [&large], [&large], 311).unwrap();

    let _ = tensor!([i; m] = a[i; j] * b[j; k] * c[k; l] * d[l; m]).unwrap();
    let retained_before = c.storage_strong_count();
    let _error = tensor!([i; m] = a[i; j] * foreign[j; k] * c[k; l] * d[l; m]).unwrap_err();
    assert_eq!(c.storage_strong_count(), retained_before);
    assert_eq!(d.storage_strong_count(), 1);
    let _ = tensor!([i; m] = a[i; j] * b[j; k] * c[k; l] * d[l; m]).unwrap();

    let stats = plan_cache_stats(&rt);
    assert_eq!(stats.workspaces_created, 1);
    assert_eq!(stats.workspace_slot_grows, 1);
    assert_eq!(stats.workspace_reuses, 2);
}

/// Same topology, mildly drifted dims (well under the drift factor): the
/// cached order is reused — this is the truncation-sweep case the
/// topology key exists for — and the result is still correct.
#[test]
fn same_topology_different_dims_hits_and_stays_correct() {
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 4, 311);
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    assert_eq!(plan_cache_stats(&rt).misses, 1);

    let (c, d) = chain(&rt, 5, 312); // ratio 5/4 = 1.25 < 2.0 default
    let cached = tensor!([i, j; m, n] = c[i, j; k, l] * d[k, l; m, n]).unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!(
        (stats.hits, stats.misses, stats.replans, stats.entries),
        (1, 1, 0, 1)
    );

    // Correctness against an uncached fresh plan.
    let expected = c.contract(&d, &[2, 3], &[0, 1]).unwrap();
    assert_close(cached.data(), expected.data(), 1e-12);
}

/// Dims drifting beyond the factor re-plan (counted separately) and refresh
/// the snapshot, still one entry per topology. `DriftFactor` is opt-in (the
/// default is `BakeOnce`, which freezes instead — see
/// `bake_once_default_freezes_after_real_dims`).
#[test]
fn drift_beyond_factor_replans() {
    let rt = Runtime::builder()
        .plan_cache(PlanCacheConfig {
            replan: ReplanPolicy::DriftFactor(2.0),
            ..PlanCacheConfig::default()
        })
        .build()
        .unwrap();
    let (a, b) = chain(&rt, 2, 321);
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();

    let (c, d) = chain(&rt, 8, 322); // ratio 4 > 2.0 default
    let result = tensor!([i, j; m, n] = c[i, j; k, l] * d[k, l; m, n]).unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!(
        (stats.hits, stats.misses, stats.replans, stats.entries),
        (0, 1, 1, 1)
    );
    let expected = c.contract(&d, &[2, 3], &[0, 1]).unwrap();
    assert_close(result.data(), expected.data(), 1e-12);

    // The snapshot was refreshed: repeating the large shape now hits.
    let _ = tensor!([i, j; m, n] = c[i, j; k, l] * d[k, l; m, n]).unwrap();
    assert_eq!(plan_cache_stats(&rt).hits, 1);
}

/// The default policy `BakeOnce`: a plan seeded at degenerate dims (some leg
/// trivial) is replaced once real dims arrive, then frozen — a later large
/// drift at real dims reuses the path instead of re-searching (the
/// cotengra/@tensoropt "find once, reuse regardless of rank" design).
#[test]
fn bake_once_default_freezes_after_real_dims() {
    let rt = Runtime::builder().build().unwrap(); // default = BakeOnce
                                                  // Seed at degenerate dims (every leg dim 1).
    let t = Space::u1([(0, 1)]);
    let a0 = Tensor::rand_with_seed(&rt, Dtype::F64, [&t, &t], [&t, &t], 401).unwrap();
    let b0 = Tensor::rand_with_seed(&rt, Dtype::F64, [&t, &t], [&t, &t], 402).unwrap();
    let _ = tensor!([i, j; m, n] = a0[i, j; k, l] * b0[k, l; m, n]).unwrap();
    assert_eq!(plan_cache_stats(&rt).misses, 1);

    // Real dims arrive: the degenerate seed is replaced once (bake-once replan).
    let (c, d) = chain(&rt, 4, 403);
    let _ = tensor!([i, j; m, n] = c[i, j; k, l] * d[k, l; m, n]).unwrap();
    let s = plan_cache_stats(&rt);
    assert_eq!((s.replans, s.misses, s.entries), (1, 1, 1));

    // Frozen: a 4x drift at real dims now HITS (no re-search), unlike DriftFactor.
    let (e, f) = chain(&rt, 16, 404);
    let result = tensor!([i, j; m, n] = e[i, j; k, l] * f[k, l; m, n]).unwrap();
    let s = plan_cache_stats(&rt);
    assert_eq!((s.hits, s.replans), (1, 1));
    let expected = e.contract(&f, &[2, 3], &[0, 1]).unwrap();
    assert_close(result.data(), expected.data(), 1e-12);
}

#[test]
fn always_reuse_policy_never_replans() {
    let rt = Runtime::builder()
        .plan_cache(PlanCacheConfig {
            replan: ReplanPolicy::AlwaysReuse,
            ..PlanCacheConfig::default()
        })
        .build()
        .unwrap();
    let (a, b) = chain(&rt, 2, 331);
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let (c, d) = chain(&rt, 16, 332); // ratio 8, reused anyway
    let result = tensor!([i, j; m, n] = c[i, j; k, l] * d[k, l; m, n]).unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.hits, stats.replans), (1, 0));
    let expected = c.contract(&d, &[2, 3], &[0, 1]).unwrap();
    assert_close(result.data(), expected.data(), 1e-12);
}

/// Persist searched orders and restore them in a fresh runtime: the reloaded
/// order is reused (at drifted dims) and computes the same result, and a blob
/// with a mismatched version header is rejected rather than trusted.
#[test]
fn persisted_orders_round_trip_and_reject_stale() {
    use tenet_network::{load_plan_cache, save_plan_cache};

    // Search once at real dims, then serialize.
    let rt1 = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&rt1, ""), 0);
    let (a, b) = chain(&rt1, 4, 501);
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let blob = save_plan_cache(&rt1);
    assert!(blob.starts_with("TENET_PLANCACHE 1"));
    assert!(blob.contains("TOPO "));

    // A fresh runtime loads the order and reuses it at different dims.
    let rt2 = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&rt2, &blob), 1);
    let (c, d) = chain(&rt2, 7, 502);
    let result = tensor!([i, j; m, n] = c[i, j; k, l] * d[k, l; m, n]).unwrap();
    let expected = c.contract(&d, &[2, 3], &[0, 1]).unwrap();
    assert_close(result.data(), expected.data(), 1e-12);

    // A stale/foreign version header is ignored (returns 0): a mismatched file
    // must not replay a now-suboptimal order.
    let rt3 = Runtime::builder().build().unwrap();
    let stale = blob.replacen("TENET_PLANCACHE 1", "TENET_PLANCACHE 0", 1);
    assert_eq!(load_plan_cache(&rt3, &stale), 0);
}

#[test]
fn different_topologies_get_separate_entries() {
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 341);
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    // Different output order = different topology.
    let _ = tensor!([j, i; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    // conj marker changes the topology too.
    let _ = tensor!([] = conj(a)[i, j; k, l] * a[i, j; k, l]).unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.hits, stats.misses, stats.entries), (0, 3, 3));
}

/// At capacity, inserting a new topology evicts the least-recently-used
/// entry, not the whole cache.
#[test]
fn eviction_drops_least_recently_used_topology() {
    let rt = Runtime::builder()
        .plan_cache(PlanCacheConfig {
            capacity: 2,
            ..PlanCacheConfig::default()
        })
        .build()
        .unwrap();
    let (a, b) = chain(&rt, 2, 371);

    // Three distinct topologies (different output orders).
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T1
    let _ = tensor!([j, i; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T2
    let _ = tensor!([i, j; n, m] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T3 evicts T1
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.misses, stats.entries), (3, 2));

    // T2 and T3 survived (hits), T1 was evicted (miss again).
    let _ = tensor!([j, i; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let _ = tensor!([i, j; n, m] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    assert_eq!(plan_cache_stats(&rt).hits, 2);
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.hits, stats.misses, stats.entries), (2, 4, 2));
}

/// A hit refreshes recency: the touched entry survives the next eviction.
#[test]
fn touched_entry_survives_eviction() {
    let rt = Runtime::builder()
        .plan_cache(PlanCacheConfig {
            capacity: 2,
            ..PlanCacheConfig::default()
        })
        .build()
        .unwrap();
    let (a, b) = chain(&rt, 2, 381);

    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T1
    let _ = tensor!([j, i; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T2
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // touch T1
    let _ = tensor!([i, j; n, m] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T3 evicts T2
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.hits, stats.misses, stats.entries), (1, 3, 2));

    // T1 was touched, so it survived; T2 is gone.
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    assert_eq!(plan_cache_stats(&rt).hits, 2);
    let _ = tensor!([j, i; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.hits, stats.misses, stats.entries), (2, 4, 2));
}

#[test]
fn disabled_cache_plans_fresh_every_call() {
    let rt = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &rt,
        PlanCacheConfig {
            enabled: false,
            ..PlanCacheConfig::default()
        },
    );
    let (a, b) = chain(&rt, 2, 351);
    let first = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let second = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.hits, stats.misses, stats.entries), (0, 0, 0));
    assert_close(first.data(), second.data(), 0.0);
    // Default config really is enabled + greedy.
    assert!(plan_cache_config(&rt).enabled == false);
}

/// Per-call optimizer override through the Network API keys separately.
#[cfg(feature = "opt-path")]
#[test]
fn optimizer_override_keys_separately() {
    use tenet_network::{NetOperand, Network};
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 361);
    let operands = [
        NetOperand {
            tensor: &a,
            conj: false,
            labels: &["i", "j", "k", "l"],
            codomain_split: Some(2),
        },
        NetOperand {
            tensor: &b,
            conj: false,
            labels: &["k", "l", "m", "n"],
            codomain_split: Some(2),
        },
    ];
    let network = Network::from_names(&operands, &["i", "j", "m", "n"], Some(2)).unwrap();
    let tensors = [&a, &b];

    let greedy = network.contract_with(&tensors, &Optimizer::Greedy).unwrap();
    let optimal = network
        .contract_with(&tensors, &Optimizer::Optimal)
        .unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.misses, stats.entries), (2, 2)); // separate keys
    let _ = network
        .contract_with(&tensors, &Optimizer::Optimal)
        .unwrap();
    assert_eq!(plan_cache_stats(&rt).hits, 1);
    assert_close(greedy.data(), optimal.data(), 1e-12);
}

#[cfg(not(feature = "opt-path"))]
#[test]
fn contract_with_explicit_greedy_shares_the_default_key() {
    use tenet_network::{NetOperand, Network};
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 361);
    let operands = [
        NetOperand {
            tensor: &a,
            conj: false,
            labels: &["i", "j", "k", "l"],
            codomain_split: Some(2),
        },
        NetOperand {
            tensor: &b,
            conj: false,
            labels: &["k", "l", "m", "n"],
            codomain_split: Some(2),
        },
    ];
    let network = Network::from_names(&operands, &["i", "j", "m", "n"], Some(2)).unwrap();
    let tensors = [&a, &b];

    let via_default = network.contract(&tensors).unwrap();
    let via_explicit = network.contract_with(&tensors, &Optimizer::Greedy).unwrap();
    let stats = plan_cache_stats(&rt);
    assert_eq!((stats.hits, stats.misses, stats.entries), (1, 1, 1));
    assert_close(via_default.data(), via_explicit.data(), 0.0);
}
