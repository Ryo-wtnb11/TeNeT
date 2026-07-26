//! `truncspace` — fixed per-sector truncation on both facades (issue #597,
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
use tenet::prelude::{Dtype, Error, Runtime, SectorLabel, Space, Tensor, Truncation};
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

// ---------------------------------------------------------------------------
// Erased facade
// ---------------------------------------------------------------------------

/// `V` carries spins 0, 1/2, 1 with room to truncate in each.
fn erased_leg() -> Space {
    Space::su2([(0, 4), (1, 4), (2, 4)]).unwrap()
}

fn erased_source(seed: u64) -> Tensor {
    let leg = erased_leg();
    let mut state = seed;
    Tensor::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap()
}

#[test]
fn erased_truncspace_produces_exactly_the_target_bond_space_on_every_sweep() {
    // What: the same profile, applied to four different tensors on the same
    // legs, yields the same bond space every time — and that space *is* the
    // target. A magnitude-driven policy cannot do this: with random payloads
    // its per-sector split moves from sweep to sweep.
    let target = Space::su2([(0, 3), (2, 1)]).unwrap();
    let truncation = Truncation::space(target.truncspace());

    for seed in 0..4u64 {
        let result = erased_source(0x5eed_0000 + seed)
            .svd_trunc(&truncation)
            .unwrap();
        let bond = result.u.domain_spaces()[0].clone();
        assert_eq!(
            bond.sectors(),
            target.sectors(),
            "sweep {seed}: bond space is not the requested profile"
        );
        // Spin 1/2 is absent from the target, so it is dropped entirely —
        // TensorKit's `dim(V, c)` is zero for a sector the space omits.
        assert!(!bond.has_sector(SectorLabel::SU2 { twice_spin: 1 }));
    }
}

#[test]
fn erased_truncspace_clamps_a_request_longer_than_the_spectrum() {
    // What: a target degeneracy above what the factorization can offer is a
    // request the prefix cannot honour, not an error — the bond simply keeps
    // everything that exists in that sector.
    let source = erased_source(0x5eed_1000);
    let greedy = Space::su2([(0, 99), (1, 99), (2, 99)]).unwrap();
    let clamped = source
        .svd_trunc(&Truncation::space(greedy.truncspace()))
        .unwrap();
    let full = source.svd_compact().unwrap();

    assert_eq!(
        clamped.u.domain_spaces()[0].sectors(),
        full.0.domain_spaces()[0].sectors(),
        "an over-long profile did not clamp to the untruncated bond"
    );
    assert_eq!(
        clamped.error, 0.0,
        "nothing was discarded, so the error is 0"
    );
}

#[test]
fn erased_truncspace_from_another_rule_is_a_typed_error() {
    // What: `SectorId`s are rule-scoped opaque keys, so a U(1) profile read as
    // SU(2) would name unrelated sectors and silently truncate to nothing.
    // It has to fail instead, before any factor data is published.
    let foreign = Space::u1([(-1, 1), (0, 2), (1, 1)]);
    let result = erased_source(0x5eed_2000).svd_trunc(&Truncation::space(foreign.truncspace()));
    assert!(
        matches!(result, Err(Error::Operation(_))),
        "a foreign-rule profile must be a typed error, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Typed facade
// ---------------------------------------------------------------------------

fn typed_leg(entries: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        entries
            .iter()
            .map(|&(twice_spin, deg)| (SU2Irrep::from_twice_spin(twice_spin), deg)),
        false,
    )
    .unwrap()
}

fn typed_source(seed: u64) -> TensorMap<SU2FusionRule, f64> {
    let leg = typed_leg(&[(0, 4), (1, 4), (2, 4)]);
    let mut state = seed;
    TensorMap::from_block_fn(&runtime(), [&leg], [&leg], move |_, _| fill(&mut state)).unwrap()
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
fn typed_truncspace_from_another_rule_is_a_typed_error() {
    // What: the typed facade reaches `select_truncation` through its own call
    // site, so the guard needs its own gate here. A U(1) leg's `SectorId`s
    // read as SU(2)'s would name unrelated spins and truncate to nothing.
    let foreign = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        false,
    )
    .unwrap();
    let result = typed_source(0x5eed_5000).svd_trunc(&Truncation::space(foreign.truncspace()));
    assert!(
        matches!(result, Err(Error::Operation(_))),
        "a foreign-rule profile must be a typed error, got {result:?}"
    );
}

/// A truncation target names sector *content*, not orientation — the claim
/// both `truncspace` rustdocs make. On a non-self-dual rule that has teeth:
/// `Space::dual` rewrites the stored sector ids to their duals *and* flips the
/// flag, so the two halves have to be separated.
#[test]
fn erased_truncspace_follows_stored_sectors_not_the_dual_flag() {
    let rt = runtime();
    let leg = Space::u1([(-1, 3), (0, 3), (1, 3)]);
    let source = Tensor::rand_with_seed(&rt, Dtype::F64, [&leg], [&leg], 0x5eed_6000).unwrap();
    let bond = |space: &Space| {
        source
            .svd_trunc(&Truncation::space(space.truncspace()))
            .unwrap()
            .u
            .domain_spaces()[0]
            .sectors()
    };

    // Content half: an asymmetric target and its dual carry mirrored charges,
    // so they must select mirrored bond spaces — the profile reads the dual's
    // rewritten ids, exactly as TensorKit's `dim(V', c)` does.
    let skewed = Space::u1([(-1, 3), (0, 2), (1, 1)]);
    assert_eq!(bond(&skewed), skewed.sectors());
    assert_eq!(bond(&skewed.dual()), skewed.dual().sectors());
    assert_ne!(
        bond(&skewed),
        bond(&skewed.dual()),
        "a non-self-dual target and its dual must not select the same bond"
    );

    // Flag half: a charge-symmetric target has the *same* stored `(id, deg)`
    // pairs as its dual and differs only in the flag, so the two profiles must
    // agree. This is the assertion that dies if the flag ever leaks in.
    let symmetric = Space::u1([(-1, 2), (0, 3), (1, 2)]);
    assert!(symmetric.dual().is_dual() && !symmetric.is_dual());
    assert_eq!(symmetric.sectors(), symmetric.dual().sectors());
    assert_eq!(bond(&symmetric), bond(&symmetric.dual()));
}
