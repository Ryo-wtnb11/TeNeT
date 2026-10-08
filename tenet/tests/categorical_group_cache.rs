//! The per-group categorical tree-transform tier (#1570).
//!
//! A sector change misses the whole-structure plan tier. For non-Unique
//! fusion the rebuild must then recompute only the source fusion-tree groups
//! whose external sectors are new, as TensorKit's `fsbraid`/`fstranspose`
//! caches do per `FusionTreeBlock`, and the result must equal a cold
//! Runtime's. Unique fusion never uses the tier.

//! The group tier is per Runtime until #2014-4 and observable only through
//! the deprecated `Runtime::tree_transform_cache_info`.
#![allow(deprecated)]

use std::collections::HashSet;
use std::sync::Arc;
use tenet::sector::{
    FibonacciFusionRule, FibonacciSector, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
};
use tenet::typed::{Complex32, Complex64, GradedSpace, Runtime, TensorMap};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

/// Completed transformers are process-global: a test that counts one
/// Runtime's coefficient-tier activity clears them first and must not race
/// another test publishing the same keys.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn bits(data: &[f64]) -> Vec<u64> {
    data.iter().map(|value| value.to_bits()).collect()
}

/// Source fusion-tree groups of `[leg, leg*] <- [leg, leg*]`, read from the
/// stored subblocks' external sectors. Why this is an independent count of
/// the changed groups: a group is identified by its external sectors and
/// dual flags, and its trees depend on those sectors alone, so the groups a
/// sector change adds are exactly the external-sector tuples it adds.
macro_rules! source_groups {
    ($runtime:expr, $leg:expr) => {{
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        let mut groups = HashSet::new();
        TensorMap::<_, f64>::from_subblock_fn($runtime, [leg, &dual], [leg, &dual], |trees, _| {
            groups.insert(format!(
                "{:?}",
                (trees.codomain_uncoupled(), trees.domain_uncoupled())
            ));
            0.0
        })
        .unwrap();
        groups
    }};
}

macro_rules! tensor {
    ($runtime:expr, $leg:expr) => {{
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        TensorMap::<_, f64>::rand_with_seed($runtime, [leg, &dual], [leg, &dual], 17).unwrap()
    }};
}

/// `a` warms both categorical tiers and `c` adds one sector. On the warm
/// Runtime, the plan for `c` must build exactly the groups that `c` adds and
/// reuse every other one. `$exact` asks for bit equality with a cold Runtime.
macro_rules! check_rule {
    ($label:literal, $a:expr, $c:expr, exact: $exact:expr) => {{
        let (a, c) = ($a, $c);
        let probe = Runtime::builder().dense_threads(1).build().unwrap();
        let before = source_groups!(&probe, &a);
        let after = source_groups!(&probe, &c);
        assert!(
            before.is_subset(&after),
            "{}: fixture must only add groups",
            $label
        );
        let changed = after.difference(&before).count();
        assert!(changed > 0 && changed < after.len(), "{}: fixture", $label);

        let operations: [(&str, &dyn Fn(&TensorMap<_, f64>) -> TensorMap<_, f64>); 4] = [
            ("permute", &|t| t.permute(&[2, 0], &[3, 1]).unwrap()),
            ("braid", &|t| {
                t.braid(&[1, 3], &[0, 2], &[0, 1, 2, 3]).unwrap()
            }),
            ("transpose", &|t| t.transpose(&[1, 3], &[0, 2]).unwrap()),
            ("adjoint permute", &|t| {
                t.adjoint().unwrap().permute(&[2, 0], &[3, 1]).unwrap()
            }),
        ];
        for (name, operation) in operations {
            let what = format!("{} {name}", $label);
            tenet::cache::clear();
            let warm = Runtime::builder().dense_threads(1).build().unwrap();
            let _ = operation(&tensor!(&warm, &a));
            let groups_before = warm.tree_transform_cache_info().groups;
            assert_eq!(
                groups_before.misses(),
                before.len(),
                "{what}: the cold plan builds every source group once"
            );
            let plans_before = warm.tree_transform_cache_info().plans;

            let sector_change = operation(&tensor!(&warm, &c));
            let groups = warm.tree_transform_cache_info().groups;
            assert!(
                warm.tree_transform_cache_info().plans.misses() > plans_before.misses(),
                "{what}: a sector change must miss the whole-structure plan"
            );
            assert_eq!(
                groups.misses() - groups_before.misses(),
                changed,
                "{what}: rebuilt groups must equal changed groups"
            );
            assert_eq!(
                groups.hits() - groups_before.hits(),
                after.len() - changed,
                "{what}: every unchanged group must hit"
            );
            assert_eq!(groups.evictions(), 0, "{what}: fixture exceeds the tier");

            tenet::cache::clear();
            let cold = Runtime::builder().dense_threads(1).build().unwrap();
            let expected = operation(&tensor!(&cold, &c));
            assert_eq!(sector_change.codomain(), expected.codomain(), "{what}");
            assert_eq!(sector_change.domain(), expected.domain(), "{what}");
            if $exact {
                assert_eq!(
                    bits(sector_change.materialize().unwrap().dense_data().unwrap()),
                    bits(expected.materialize().unwrap().dense_data().unwrap()),
                    "{what}: partially reused plan differs from a cold Runtime"
                );
            } else {
                // Terms per entry: the recoupled trees of one fusion block,
                // bounded well below 64 for these rank-4 fixtures.
                numerics::assert_slices_close(
                    &what,
                    sector_change.materialize().unwrap().dense_data().unwrap(),
                    expected.materialize().unwrap().dense_data().unwrap(),
                    64,
                );
            }
        }
    }};
}

type Su2Operation = dyn Fn(&TensorMap<SU2FusionRule, f64>) -> TensorMap<SU2FusionRule, f64>;

fn su2_leg(sectors: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        sectors
            .iter()
            .map(|&(j, d)| (SU2Irrep::from_twice_spin(j), d)),
    )
    .unwrap()
}

#[test]
fn su2_sector_change_rebuilds_only_changed_groups() {
    let _serial = serial();
    check_rule!(
        "SU(2)",
        su2_leg(&[(0, 2), (1, 2), (2, 1)]),
        su2_leg(&[(0, 2), (1, 2), (2, 1), (3, 1)]),
        exact: true
    );
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_su3_sector_change_rebuilds_only_changed_groups() {
    let _serial = serial();
    use tenet::sector::SUNFusionRule;

    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = |sectors: &[([i64; 2], usize)]| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            sectors.iter().map(|&(labels, d)| (labels.to_vec(), d)),
        )
        .unwrap()
    };
    check_rule!(
        "SU(3)",
        leg(&[([1, 0], 2), ([1, 1], 1)]),
        leg(&[([0, 0], 1), ([1, 0], 2), ([1, 1], 1)]),
        // Why not bit equality: the checked-Generic SU(3) coefficients are
        // summed through racah's randomly seeded hash maps, so two cold
        // Runtimes already differ in the last bits.
        exact: false
    );
}

#[test]
fn unique_fusion_never_uses_the_group_tier() {
    let _serial = serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = |sectors: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::new(U1FusionRule),
            sectors.iter().map(|&(q, d)| (U1Irrep::new(q), d)),
        )
        .unwrap()
    };
    for space in [
        leg(&[(-1, 2), (0, 1), (1, 2)]),
        leg(&[(-1, 2), (0, 1), (1, 2), (2, 1)]),
    ] {
        let t = tensor!(&runtime, &space);
        let _ = t.permute(&[2, 0], &[3, 1]).unwrap();
        let _ = t.adjoint().unwrap().permute(&[2, 0], &[3, 1]).unwrap();
    }
    assert!(runtime.tree_transform_cache_info().plans.misses() > 0);
    let info = runtime.tree_transform_cache_info().groups;
    assert_eq!(
        (info.hits(), info.misses(), info.entries()),
        (0, 0, 0),
        "Unique fusion must not use the group tier"
    );
}

#[test]
fn clear_resets_the_group_tier() {
    let _serial = serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = su2_leg(&[(0, 1), (1, 1)]);
    let _ = tensor!(&runtime, &leg).permute(&[2, 0], &[3, 1]).unwrap();
    let before = runtime.tree_transform_cache_info().groups;
    assert!(before.entries() > 0 && before.charged_payload_bytes() > 0);
    assert!(before.entries() <= before.entry_capacity());
    // The group tier counts groups, not structures, so its cap is larger.
    assert!(before.entry_capacity() > runtime.tree_transform_cache_info().plans.entry_capacity());
    assert!(before.charged_payload_bytes() <= before.byte_budget());

    runtime.clear_tree_transform_cache();
    let after = runtime.tree_transform_cache_info().groups;
    assert_eq!(after.entries(), 0);
    assert_eq!(after.charged_payload_bytes(), 0);
    assert_eq!((after.hits(), after.misses()), (0, 0));
}

/// Many operations on one Runtime keep far more groups than the 256
/// structures the other tiers hold. Every unchanged group must still hit:
/// the group tier's entry cap is per group, and the byte budget binds.
#[test]
fn shared_runtime_keeps_every_unchanged_group_across_operations() {
    let _serial = serial();
    let (a, c) = (
        su2_leg(&[(0, 2), (1, 2), (2, 1)]),
        su2_leg(&[(0, 2), (1, 2), (2, 1), (3, 1)]),
    );
    let probe = Runtime::builder().dense_threads(1).build().unwrap();
    let before = source_groups!(&probe, &a).len();
    let after = source_groups!(&probe, &c).len();
    let operations: [&Su2Operation; 5] = [
        &|t| t.permute(&[2, 0], &[3, 1]).unwrap(),
        &|t| t.permute(&[1, 3], &[0, 2]).unwrap(),
        &|t| t.permute(&[0, 2], &[1, 3]).unwrap(),
        &|t| t.transpose(&[1, 3], &[0, 2]).unwrap(),
        // Why not `[2, 0], [3, 1]`: on this self-adjoint space that adjoint
        // permute reads storage like the direct `[0, 2], [1, 3]` above and is
        // served by the completed-structure tier without a plan lookup.
        &|t| t.adjoint().unwrap().permute(&[1, 0], &[3, 2]).unwrap(),
    ];
    tenet::cache::clear();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for operation in operations {
        let _ = operation(&tensor!(&runtime, &a));
    }
    let warm = runtime.tree_transform_cache_info().groups;
    for operation in operations {
        let _ = operation(&tensor!(&runtime, &c));
    }
    let info = runtime.tree_transform_cache_info().groups;
    let n = operations.len();
    assert_eq!(
        info.hits() - warm.hits(),
        n * before,
        "every unchanged group must hit"
    );
    assert_eq!(info.misses() - warm.misses(), n * (after - before));
    assert_eq!(info.evictions(), 0);
    assert!(
        info.entries() > 256,
        "fixture must exceed the per-structure tiers' 256 entries"
    );
}

/// Operations that share a source structure but differ in the permutation,
/// the braid levels, or the orientation together with storage conjugation
/// must not share groups: each result on one shared Runtime equals a cold
/// Runtime's bit for bit. Orientation is never varied alone: every reachable
/// adjoint-oriented caller also conjugates storage, so this covers the two
/// key fields jointly, not orientation in isolation.
#[test]
fn shared_runtime_distinguishes_operations_on_the_same_groups() {
    let _serial = serial();
    let leg = su2_leg(&[(0, 2), (1, 2), (2, 1)]);
    let operations: [(&str, &Su2Operation); 5] = [
        ("permute p1", &|t| t.permute(&[2, 0], &[3, 1]).unwrap()),
        ("permute p2", &|t| t.permute(&[3, 1], &[2, 0]).unwrap()),
        ("transpose", &|t| t.transpose(&[1, 3], &[0, 2]).unwrap()),
        ("direct permute", &|t| t.permute(&[1, 0], &[3, 2]).unwrap()),
        ("adjoint permute", &|t| {
            t.adjoint().unwrap().permute(&[1, 0], &[3, 2]).unwrap()
        }),
    ];
    let shared = Runtime::builder().dense_threads(1).build().unwrap();
    for (name, operation) in operations {
        let warm = operation(&tensor!(&shared, &leg));
        let cold_runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let cold = operation(&tensor!(&cold_runtime, &leg));
        assert_eq!(warm.codomain(), cold.codomain(), "{name}");
        assert_eq!(
            bits(warm.materialize().unwrap().dense_data().unwrap()),
            bits(cold.materialize().unwrap().dense_data().unwrap()),
            "{name}: shared-Runtime result differs from a cold Runtime"
        );
    }

    // Levels only matter for non-symmetric braiding.
    let fib = GradedSpace::try_new(
        Arc::new(FibonacciFusionRule),
        [(FibonacciSector::Vacuum, 2), (FibonacciSector::Tau, 2)],
    )
    .unwrap();
    let fib_tensor = |runtime: &Runtime| {
        let dual = fib.try_dual().unwrap();
        TensorMap::<_, Complex64>::rand_with_seed(runtime, [&fib, &dual], [&fib, &dual], 5).unwrap()
    };
    let complex_bits = |t: &TensorMap<FibonacciFusionRule, Complex64>| {
        t.materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
            .collect::<Vec<_>>()
    };
    let shared = Runtime::builder().dense_threads(1).build().unwrap();
    let under = fib_tensor(&shared)
        .braid(&[1, 3], &[0, 2], &[0, 1, 2, 3])
        .unwrap();
    let over = fib_tensor(&shared)
        .braid(&[1, 3], &[0, 2], &[3, 2, 1, 0])
        .unwrap();
    assert_ne!(
        complex_bits(&under),
        complex_bits(&over),
        "fixture: levels matter"
    );
    let cold_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let cold_over = fib_tensor(&cold_runtime)
        .braid(&[1, 3], &[0, 2], &[3, 2, 1, 0])
        .unwrap();
    assert_eq!(
        complex_bits(&over),
        complex_bits(&cold_over),
        "Fibonacci braid levels: shared-Runtime result differs from a cold Runtime"
    );
}
