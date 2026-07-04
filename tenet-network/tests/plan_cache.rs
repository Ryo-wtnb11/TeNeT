//! The topology-keyed plan cache behind `tensor!` / `Network::contract`.
//! The cache is thread-local and every #[test] runs on its own thread, so
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
    let a = Tensor::rand_with_seed(rt, [&v, &v], [&v, &v], seed).unwrap();
    let b = Tensor::rand_with_seed(rt, [&v, &v], [&v, &v], seed + 1).unwrap();
    (a, b)
}

#[test]
fn second_identical_call_hits() {
    clear_plan_cache();
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 301);

    let first = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let stats = plan_cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (0, 1, 1));

    let second = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let stats = plan_cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (1, 1, 1));
    assert_close(first.data(), second.data(), 0.0);
}

/// Same topology, mildly drifted dims (well under the drift factor): the
/// cached order is reused — this is the truncation-sweep case the
/// topology key exists for — and the result is still correct.
#[test]
fn same_topology_different_dims_hits_and_stays_correct() {
    clear_plan_cache();
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 4, 311);
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    assert_eq!(plan_cache_stats().misses, 1);

    let (c, d) = chain(&rt, 5, 312); // ratio 5/4 = 1.25 < 2.0 default
    let cached = tensor!([i, j; m, n] = c[i, j; k, l] * d[k, l; m, n]).unwrap();
    let stats = plan_cache_stats();
    assert_eq!(
        (stats.hits, stats.misses, stats.replans, stats.entries),
        (1, 1, 0, 1)
    );

    // Correctness against an uncached fresh plan.
    let expected = c.contract(&d, &[2, 3], &[0, 1]).unwrap();
    assert_close(cached.data(), expected.data(), 1e-12);
}

/// Dims drifting beyond the factor re-plan (counted separately) and refresh
/// the snapshot, still one entry per topology.
#[test]
fn drift_beyond_factor_replans() {
    clear_plan_cache();
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 321);
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();

    let (c, d) = chain(&rt, 8, 322); // ratio 4 > 2.0 default
    let result = tensor!([i, j; m, n] = c[i, j; k, l] * d[k, l; m, n]).unwrap();
    let stats = plan_cache_stats();
    assert_eq!(
        (stats.hits, stats.misses, stats.replans, stats.entries),
        (0, 1, 1, 1)
    );
    let expected = c.contract(&d, &[2, 3], &[0, 1]).unwrap();
    assert_close(result.data(), expected.data(), 1e-12);

    // The snapshot was refreshed: repeating the large shape now hits.
    let _ = tensor!([i, j; m, n] = c[i, j; k, l] * d[k, l; m, n]).unwrap();
    assert_eq!(plan_cache_stats().hits, 1);
}

#[test]
fn always_reuse_policy_never_replans() {
    clear_plan_cache();
    configure_plan_cache(PlanCacheConfig {
        replan: ReplanPolicy::AlwaysReuse,
        ..PlanCacheConfig::default()
    });
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 331);
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let (c, d) = chain(&rt, 16, 332); // ratio 8, reused anyway
    let result = tensor!([i, j; m, n] = c[i, j; k, l] * d[k, l; m, n]).unwrap();
    let stats = plan_cache_stats();
    assert_eq!((stats.hits, stats.replans), (1, 0));
    let expected = c.contract(&d, &[2, 3], &[0, 1]).unwrap();
    assert_close(result.data(), expected.data(), 1e-12);
}

#[test]
fn different_topologies_get_separate_entries() {
    clear_plan_cache();
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 341);
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    // Different output order = different topology.
    let _ = tensor!([j, i; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    // conj marker changes the topology too.
    let _ = tensor!([] = conj(a)[i, j; k, l] * a[i, j; k, l]).unwrap();
    let stats = plan_cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (0, 3, 3));
}

/// At capacity, inserting a new topology evicts the least-recently-used
/// entry, not the whole cache.
#[test]
fn eviction_drops_least_recently_used_topology() {
    clear_plan_cache();
    configure_plan_cache(PlanCacheConfig {
        capacity: 2,
        ..PlanCacheConfig::default()
    });
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 371);

    // Three distinct topologies (different output orders).
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T1
    let _ = tensor!([j, i; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T2
    let _ = tensor!([i, j; n, m] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T3 evicts T1
    let stats = plan_cache_stats();
    assert_eq!((stats.misses, stats.entries), (3, 2));

    // T2 and T3 survived (hits), T1 was evicted (miss again).
    let _ = tensor!([j, i; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let _ = tensor!([i, j; n, m] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    assert_eq!(plan_cache_stats().hits, 2);
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let stats = plan_cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (2, 4, 2));
}

/// A hit refreshes recency: the touched entry survives the next eviction.
#[test]
fn touched_entry_survives_eviction() {
    clear_plan_cache();
    configure_plan_cache(PlanCacheConfig {
        capacity: 2,
        ..PlanCacheConfig::default()
    });
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 381);

    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T1
    let _ = tensor!([j, i; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T2
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // touch T1
    let _ = tensor!([i, j; n, m] = a[i, j; k, l] * b[k, l; m, n]).unwrap(); // T3 evicts T2
    let stats = plan_cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (1, 3, 2));

    // T1 was touched, so it survived; T2 is gone.
    let _ = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    assert_eq!(plan_cache_stats().hits, 2);
    let _ = tensor!([j, i; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let stats = plan_cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (2, 4, 2));
}

#[test]
fn disabled_cache_plans_fresh_every_call() {
    clear_plan_cache();
    configure_plan_cache(PlanCacheConfig {
        enabled: false,
        ..PlanCacheConfig::default()
    });
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = chain(&rt, 2, 351);
    let first = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let second = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).unwrap();
    let stats = plan_cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (0, 0, 0));
    assert_close(first.data(), second.data(), 0.0);
    // Default config really is enabled + greedy.
    assert!(plan_cache_config().enabled == false);
}

/// Per-call optimizer override through the Network API keys separately.
#[cfg(feature = "opt-path")]
#[test]
fn optimizer_override_keys_separately() {
    use tenet_network::{NetOperand, Network};
    clear_plan_cache();
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
    let stats = plan_cache_stats();
    assert_eq!((stats.misses, stats.entries), (2, 2)); // separate keys
    let _ = network
        .contract_with(&tensors, &Optimizer::Optimal)
        .unwrap();
    assert_eq!(plan_cache_stats().hits, 1);
    assert_close(greedy.data(), optimal.data(), 1e-12);
}

#[cfg(not(feature = "opt-path"))]
#[test]
fn contract_with_explicit_greedy_shares_the_default_key() {
    use tenet_network::{NetOperand, Network};
    clear_plan_cache();
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
    let stats = plan_cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (1, 1, 1));
    assert_close(via_default.data(), via_explicit.data(), 0.0);
}
