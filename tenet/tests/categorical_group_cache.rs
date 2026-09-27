//! The per-group categorical tree-transform tier (#1570).
//!
//! A sector change misses the whole-structure plan tier. For non-Unique
//! fusion the rebuild must then recompute only the source fusion-tree groups
//! whose external sectors are new, as TensorKit's `fsbraid`/`fstranspose`
//! caches do per `FusionTreeBlock`, and the result must equal a cold
//! Runtime's. Unique fusion never uses the tier.

use std::collections::HashSet;
use std::sync::Arc;
use tenet::prelude::*;

#[path = "../../tests/support/numerics.rs"]
mod numerics;

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
            let warm = Runtime::builder().dense_threads(1).build().unwrap();
            let _ = operation(&tensor!(&warm, &a));
            let groups_before = warm.tree_transform_group_cache_info();
            assert_eq!(
                groups_before.misses(),
                before.len(),
                "{what}: the cold plan builds every source group once"
            );
            let plans_before = warm.tree_transform_plan_cache_info();

            let sector_change = operation(&tensor!(&warm, &c));
            let groups = warm.tree_transform_group_cache_info();
            assert!(
                warm.tree_transform_plan_cache_info().misses() > plans_before.misses(),
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
    use tenet::typed::SUNFusionRule;

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
    assert!(runtime.tree_transform_plan_cache_info().misses() > 0);
    let info = runtime.tree_transform_group_cache_info();
    assert_eq!(
        (info.hits(), info.misses(), info.entries()),
        (0, 0, 0),
        "Unique fusion must not use the group tier"
    );
}

#[test]
fn clear_resets_the_group_tier() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = su2_leg(&[(0, 1), (1, 1)]);
    let _ = tensor!(&runtime, &leg).permute(&[2, 0], &[3, 1]).unwrap();
    let before = runtime.tree_transform_group_cache_info();
    assert!(before.entries() > 0 && before.charged_payload_bytes() > 0);
    assert!(before.entries() <= before.entry_capacity());
    assert!(before.charged_payload_bytes() <= before.byte_budget());

    runtime.clear_tree_transform_cache();
    let after = runtime.tree_transform_group_cache_info();
    assert_eq!(after.entries(), 0);
    assert_eq!(after.charged_payload_bytes(), 0);
    assert_eq!((after.hits(), after.misses()), (0, 0));
}
