//! The categorical tree-transform plan tier (#1554).
//!
//! A truncation changes block dimensions but usually not which sectors are
//! present. Such a degeneracy-only change misses the completed-structure
//! cache (its key holds the exact layout) but must not rebuild the
//! categorical plan (recoupling coefficients and tree-pair maps), and the
//! result must be bit-identical to a cold Runtime's. A sector change must
//! still rebuild.

use num_complex::{Complex32, Complex64};
use tenet::prelude::*;

#[path = "../../tests/support/numerics.rs"]
mod numerics;

fn bits(data: &[f64]) -> Vec<u64> {
    data.iter().map(|value| value.to_bits()).collect()
}

/// Rank (2,2) tensor on `[leg, leg*] <- [leg, leg*]`, so every operation also
/// moves a dual leg.
macro_rules! tensor {
    ($runtime:expr, $leg:expr) => {{
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        TensorMap::<_, f64>::rand_with_seed($runtime, [leg, &dual], [leg, &dual], 17).unwrap()
    }};
}

/// Runs permute, braid and transpose on three spaces over one rule: `a` warms
/// the plan tier, `b` has the same sectors with other degeneracies, and `c`
/// adds a sector. `$exact` asks for bit equality with a cold Runtime; it is
/// false only where two cold Runtimes already disagree in the last bits.
macro_rules! check_rule {
    ($label:literal, $a:expr, $b:expr, $c:expr, exact: $exact:expr) => {{
        let (a, b, c) = ($a, $b, $c);
        let operations: [(&str, &dyn Fn(&TensorMap<_, f64>) -> TensorMap<_, f64>); 4] = [
            ("permute", &|t| t.permute(&[2, 0], &[3, 1]).unwrap()),
            ("braid", &|t| {
                t.braid(&[1, 3], &[0, 2], &[0, 1, 2, 3]).unwrap()
            }),
            ("transpose", &|t| {
                t.transpose_axes(&[1, 3], &[0, 2]).unwrap()
            }),
            // A lazy adjoint source reads its parent's storage through the
            // adjoint orientation.
            ("adjoint permute", &|t| {
                t.adjoint().unwrap().permute(&[2, 0], &[3, 1]).unwrap()
            }),
        ];
        for (name, operation) in operations {
            let what = format!("{} {name}", $label);
            let warm = Runtime::builder().dense_threads(1).build().unwrap();
            let _ = operation(&tensor!(&warm, &a));
            let plans_before = warm.tree_transform_plan_cache_info();
            let structures_before = warm.tree_transform_cache_info();

            let degeneracy_only = operation(&tensor!(&warm, &b));
            let plans = warm.tree_transform_plan_cache_info();
            let structures = warm.tree_transform_cache_info();
            assert!(
                structures.misses() > structures_before.misses(),
                "{what}: a new layout must miss the completed-structure tier"
            );
            assert_eq!(
                plans.misses(),
                plans_before.misses(),
                "{what}: a degeneracy-only change rebuilt a categorical plan"
            );
            assert!(plans.hits() > plans_before.hits(), "{what}: no plan hit");

            let cold = Runtime::builder().dense_threads(1).build().unwrap();
            let expected = operation(&tensor!(&cold, &b));
            assert!(
                cold.tree_transform_plan_cache_info().misses() > 0,
                "{what}: the cold path must build through the plan tier"
            );
            assert_eq!(degeneracy_only.codomain(), expected.codomain(), "{what}");
            assert_eq!(degeneracy_only.domain(), expected.domain(), "{what}");
            if $exact {
                assert_eq!(
                    bits(degeneracy_only.data()),
                    bits(expected.data()),
                    "{what}: warm plan result differs from a cold Runtime"
                );
            } else {
                // Terms per entry: the recoupled trees of one fusion block,
                // bounded well below 64 for these rank-4 fixtures.
                numerics::assert_slices_close(&what, degeneracy_only.data(), expected.data(), 64);
            }

            let _ = operation(&tensor!(&warm, &c));
            assert!(
                warm.tree_transform_plan_cache_info().misses() > plans.misses(),
                "{what}: a sector change must rebuild the plan"
            );
        }
    }};
}

#[test]
fn u1_degeneracy_change_reuses_the_categorical_plan() {
    let leg = |sectors: &[(i32, usize)]| {
        GradedSpace::try_new(
            U1FusionRule,
            sectors.iter().map(|&(q, d)| (U1Irrep::new(q), d)),
        )
        .unwrap()
    };
    check_rule!(
        "U(1)",
        leg(&[(-1, 2), (0, 1), (1, 2)]),
        leg(&[(-1, 3), (0, 2), (1, 1)]),
        leg(&[(-1, 2), (0, 1), (1, 2), (2, 1)]),
        exact: true
    );
}

#[test]
fn su2_degeneracy_change_reuses_the_categorical_plan() {
    let leg = |sectors: &[(usize, usize)]| {
        GradedSpace::try_new(
            SU2FusionRule,
            sectors
                .iter()
                .map(|&(j, d)| (SU2Irrep::from_twice_spin(j), d)),
        )
        .unwrap()
    };
    check_rule!(
        "SU(2)",
        leg(&[(0, 2), (1, 2), (2, 1)]),
        leg(&[(0, 1), (1, 3), (2, 2)]),
        leg(&[(0, 2), (1, 2), (2, 1), (3, 1)]),
        exact: true
    );
}

#[test]
fn fermion_u1_degeneracy_change_reuses_the_categorical_plan() {
    let leg = |sectors: &[(i32, usize)]| {
        let rule = ProductFusionRule::<FermionParityFusionRule, U1FusionRule>::new(
            FermionParityFusionRule,
            U1FusionRule,
        );
        GradedSpace::try_new(
            rule,
            sectors.iter().map(|&(q, d)| {
                let parity = if q.rem_euclid(2) == 0 {
                    Z2Irrep::EVEN
                } else {
                    Z2Irrep::ODD
                };
                (ProductSector::new(parity, U1Irrep::new(q)), d)
            }),
        )
        .unwrap()
    };
    check_rule!(
        "fZ2xU(1)",
        leg(&[(-1, 2), (0, 1), (1, 2)]),
        leg(&[(-1, 1), (0, 3), (1, 2)]),
        leg(&[(-1, 2), (0, 1), (1, 2), (2, 1)]),
        exact: true
    );
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_su3_degeneracy_change_reuses_the_categorical_plan() {
    use std::sync::Arc;

    use tenet::typed::SUNFusionRule;

    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = |sectors: &[([i64; 2], usize)]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&provider),
            sectors.iter().map(|&(labels, d)| (labels.to_vec(), d)),
        )
        .unwrap()
    };
    check_rule!(
        "SU(3)",
        leg(&[([1, 0], 2), ([1, 1], 1)]),
        leg(&[([1, 0], 1), ([1, 1], 2)]),
        leg(&[([0, 0], 1), ([1, 0], 2), ([1, 1], 1)]),
        // Why not bit equality: the checked-Generic SU(3) coefficients are
        // summed through racah's randomly seeded hash maps, so two cold
        // Runtimes on `origin/main` already differ in the last bits.
        exact: false
    );
}

#[test]
fn clear_resets_the_plan_tier() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        SU2FusionRule,
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let _ = tensor!(&runtime, &leg).permute(&[2, 0], &[3, 1]).unwrap();
    let before = runtime.tree_transform_plan_cache_info();
    assert!(before.entries() > 0 && before.charged_payload_bytes() > 0);
    assert!(before.entries() <= before.entry_capacity());
    assert!(before.charged_payload_bytes() <= before.byte_budget());

    runtime.clear_tree_transform_cache();
    let after = runtime.tree_transform_plan_cache_info();
    assert_eq!(after.entries(), 0);
    assert_eq!(after.charged_payload_bytes(), 0);
    assert_eq!(after.misses(), 0);
    assert_eq!(after.hits(), 0);
}
