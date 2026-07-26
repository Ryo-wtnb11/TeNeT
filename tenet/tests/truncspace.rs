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

use tenet::core::{SU2FusionRule, SU2Irrep};
use tenet::prelude::{Error, Runtime, SectorLabel, Space, Tensor, Truncation};
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
