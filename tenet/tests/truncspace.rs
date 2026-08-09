//! `truncspace` — fixed per-sector truncation on the typed facade (issue #597,
//! item 4).
//!
//! TensorKit `TruncationSpace` (`src/factorizations/truncation.jl:261-269`)
//! reads the target rank of coupled sector `c` as `dim(strategy.space, c)` and
//! applies a plain `truncrank` inside that block, so the decision is a fixed
//! per-sector prefix count and nothing about the spectrum's magnitudes enters.
//!
//! What that has to buy a caller is *reproducibility*: unlike a rank or
//! tolerance budget, the same profile must produce the same bond space on
//! every sweep regardless of how the singular values happen to fall. That is
//! what the repeated-sweep tests below assert, by comparing the resulting bond
//! space to the target space itself rather than counting values.

use std::sync::Arc;

use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::prelude::{Error, Runtime, Truncation};
use tenet::typed::{GradedSpace, TensorMap};

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

/// A deterministic but structureless fill, so the singular values carry no
/// pattern a magnitude-driven policy could accidentally reproduce.
fn fill(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

fn typed_leg(entries: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        entries
            .iter()
            .map(|&(twice_spin, deg)| (SU2Irrep::from_twice_spin(twice_spin), deg)),
    )
    .unwrap()
}

fn typed_source(seed: u64) -> TensorMap<SU2FusionRule, f64> {
    let leg = typed_leg(&[(0, 4), (1, 4), (2, 4)]);
    let mut state = seed;
    TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap()
}

fn typed_u1_leg(entries: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        entries
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

#[test]
fn typed_truncspace_produces_exactly_the_target_bond_space_on_every_sweep() {
    let target = typed_leg(&[(0, 3), (2, 1)]);
    let truncation = Truncation::space(target.truncspace());
    let expected = target.sectors().unwrap();

    for seed in 0..4u64 {
        let result = typed_source(0x5eed_3000 + seed)
            .svd_trunc(&truncation)
            .unwrap();
        let bond = result.u.domain()[0].clone();
        assert_eq!(
            bond.sectors().unwrap(),
            expected,
            "sweep {seed}: bond sectors are not the requested profile"
        );
        assert_eq!(
            bond.degeneracies(),
            target.degeneracies(),
            "sweep {seed}: bond degeneracies are not the requested prefix counts"
        );
    }
}

#[test]
fn typed_truncspace_composes_with_a_magnitude_policy() {
    // What: `and` keeps the per-sector minimum, so intersecting a profile with
    // a rank budget stays inside prefix-land — the profile is an upper bound
    // the tighter policy may cut further.
    let target = typed_leg(&[(0, 4), (1, 4), (2, 4)]);
    let combined = Truncation::space(target.truncspace()).and(Truncation::rank(2));
    let result = typed_source(0x5eed_4000).svd_trunc(&combined).unwrap();
    let bond = result.u.domain()[0].clone();

    let kept: usize = bond.degeneracies().iter().sum();
    assert!(
        kept > 0 && kept <= 2,
        "the rank-2 budget did not tighten the profile: kept {kept}"
    );
}

#[test]
fn typed_truncspace_clamps_a_request_longer_than_the_spectrum() {
    let source = typed_source(0x5eed_1000);
    let greedy = typed_leg(&[(0, 99), (1, 99), (2, 99)]);
    let clamped = source
        .svd_trunc(&Truncation::space(greedy.truncspace()))
        .unwrap();
    let full = source.svd_compact().unwrap();
    let clamped_bond = &clamped.u.domain()[0];
    let full_bond = &full.0.domain()[0];

    assert_eq!(
        clamped_bond.sectors().unwrap(),
        full_bond.sectors().unwrap()
    );
    assert_eq!(clamped_bond.degeneracies(), full_bond.degeneracies());
    assert_eq!(clamped.error, 0.0);
}

#[test]
fn typed_truncspace_from_another_rule_is_a_typed_error() {
    // What: the typed facade reaches `select_truncation` through its own call
    // site, so the guard needs its own gate here. A U(1) leg's `SectorId`s
    // read as SU(2)'s would name unrelated spins and truncate to nothing.
    let foreign = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let result = typed_source(0x5eed_5000).svd_trunc(&Truncation::space(foreign.truncspace()));
    assert!(
        matches!(result, Err(Error::Operation(_))),
        "a foreign-rule profile must be a typed error, got {result:?}"
    );
}

/// A truncation target names sector *content*, not orientation. On a
/// non-self-dual rule, dualization rewrites labels and flips the flag, so the
/// two effects have to be separated.
#[test]
fn typed_truncspace_follows_stored_sectors_not_the_dual_flag() {
    let leg = typed_u1_leg(&[(-1, 3), (0, 3), (1, 3)]);
    let source =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime(), [&leg], [&leg], 0x5eed_6000)
            .unwrap();
    let bond = |space: &GradedSpace<U1FusionRule>| {
        source
            .svd_trunc(&Truncation::space(space.truncspace()))
            .unwrap()
            .u
            .domain()[0]
            .clone()
    };

    // Content half: an asymmetric target and its dual carry mirrored charges,
    // so they must select mirrored bond spaces — the profile reads the dual's
    // rewritten ids, exactly as TensorKit's `dim(V', c)` does.
    let skewed = typed_u1_leg(&[(-1, 3), (0, 2), (1, 1)]);
    let skewed_dual = skewed.try_dual().unwrap();
    let skewed_bond = bond(&skewed);
    let skewed_dual_bond = bond(&skewed_dual);
    assert_eq!(skewed_bond.sectors().unwrap(), skewed.sectors().unwrap());
    assert_eq!(skewed_bond.degeneracies(), skewed.degeneracies());
    assert_eq!(
        skewed_dual_bond.sectors().unwrap(),
        skewed_dual.sectors().unwrap()
    );
    assert_eq!(skewed_dual_bond.degeneracies(), skewed_dual.degeneracies());
    assert_ne!(
        skewed_bond.degeneracies(),
        skewed_dual_bond.degeneracies(),
        "a non-self-dual target and its dual must not select the same bond"
    );

    // Flag half: a charge-symmetric target has the *same* stored `(id, deg)`
    // pairs as its dual and differs only in the flag, so the two profiles must
    // agree. This is the assertion that dies if the flag ever leaks in.
    let symmetric = typed_u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let symmetric_dual = symmetric.try_dual().unwrap();
    assert!(symmetric_dual.is_dual() && !symmetric.is_dual());
    assert_eq!(
        symmetric.sectors().unwrap(),
        symmetric_dual.sectors().unwrap()
    );
    assert_eq!(symmetric.degeneracies(), symmetric_dual.degeneracies());
    let symmetric_bond = bond(&symmetric);
    let symmetric_dual_bond = bond(&symmetric_dual);
    assert_eq!(
        symmetric_bond.sectors().unwrap(),
        symmetric_dual_bond.sectors().unwrap()
    );
    assert_eq!(
        symmetric_bond.degeneracies(),
        symmetric_dual_bond.degeneracies()
    );
}
