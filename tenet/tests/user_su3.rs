//! Stage B3b Gate 3: SU(3) works at the top-level `tenet::Tensor` layer.
//!
//! Construction + permute/braid round-trips over the real `Su3FusionRule` table,
//! routed through the `RuleKind::Su3` generic path. The recoupling math the
//! transform performs is the same tree-level layer already pinned against
//! TensorKit in `tenet-core` (Stage B3b Gate 1); here we prove the top-level
//! wiring (space build → generic tree transform → back) is correct and
//! invertible.

use tenet::prelude::*;

// A leg carrying the fundamental 3 = (1,0) and its conjugate 3̄ = (0,1). Their
// pairwise fusions (3⊗3̄ ∋ 1,8; 3⊗3 ∋ 3̄,6; …) all stay inside the dim≤27 table.
fn v() -> Space {
    Space::su3([((1, 0), 2), ((0, 1), 1)]).unwrap()
}

#[test]
fn su3_space_construction_and_dim() {
    let v = v();
    // quantum-dim-weighted total: 3 with deg 2, 3̄ with deg 1.
    assert_eq!(v.dim(), 2 * 3 + 1 * 3);
    // dual is an involution and preserves the total dim.
    assert_eq!(v.dual().dim(), v.dim());
    assert_eq!(v.dual().dual(), v);
    // Out-of-table irrep is rejected (label API — sectors()/degeneracy() — is
    // deferred to B3c to avoid a breaking public SectorLabel variant).
    assert!(Space::su3([((5, 5), 1)]).is_err()); // dim(5,5) > 27
}

#[test]
fn su3_permute_round_trip_is_identity() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    // rank 2 -> 0: codomain [v, v], domain []. Coupled = vacuum forces the
    // (3,3̄) and (3̄,3) blocks — a genuine non-empty SU(3) tensor.
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [], 11).unwrap();
    assert!(
        !a.data().is_empty(),
        "SU(3) tensor must have allowed blocks"
    );

    let swapped = a.permute(&[1, 0], &[]).unwrap();
    // The permute genuinely recouples (data is not the identity rearrangement).
    assert_ne!(swapped.data(), a.data());
    // Round-trip: swap back returns the original data (bit-for-bit up to fp).
    let back = swapped.permute(&[1, 0], &[]).unwrap();
    assert_eq!(back.data().len(), a.data().len());
    for (x, y) in back.data().iter().zip(a.data().iter()) {
        assert!((x - y).abs() < 1e-12, "permute round-trip: {x} vs {y}");
    }
}

#[test]
fn su3_braid_round_trip_is_identity() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [], 7).unwrap();
    // Braid legs 0,1 (levels decide the crossing), then braid back with the
    // opposite level order — the group inverse, returning the source.
    let braided = a.braid(&[1, 0], &[], &[0, 1]).unwrap();
    let back = braided.braid(&[1, 0], &[], &[1, 0]).unwrap();
    assert_eq!(back.data().len(), a.data().len());
    for (x, y) in back.data().iter().zip(a.data().iter()) {
        assert!((x - y).abs() < 1e-12, "braid round-trip: {x} vs {y}");
    }
}

// FLIPPED refute test (Option A fix): a codomain of three adjoint (8) legs has
// out-of-table coupled candidates (35, 35̄, 64), so full-space construction from
// the PUBLIC Tensor API now returns a clean Err naming them — it neither panics
// (the refuted behavior) nor silently truncates the block structure. Per-sector
// exactness of the same space is pinned against TK in tenet-core
// (b3b_fix_enum_rank3_888_per_sector_matches_tensorkit, 24 trees).
#[test]
fn su3_rank3_adjoint_codomain_errs_cleanly() {
    let rt = Runtime::builder().build().unwrap();
    let a8 = Space::su3([((1, 1), 1)]).unwrap(); // the adjoint 8
    let err = Tensor::rand_with_seed(&rt, Dtype::F64, [&a8, &a8, &a8], [], 1).unwrap_err();
    let message = format!("{err:?}");
    assert!(
        message.contains("cannot represent this space exactly"),
        "want the bounded-table Err, got: {message}"
    );
}

// FLIPPED refute test: the rank-2 escaping leg pair (27⊗8 ∋ 35, 35̄, 64) also
// returns Err through the fallible constructor instead of panicking.
#[test]
fn su3_rank2_escaping_legpair_errs_cleanly() {
    let rt = Runtime::builder().build().unwrap();
    let s27 = Space::su3([((2, 2), 1)]).unwrap(); // 27
    let s8 = Space::su3([((1, 1), 1)]).unwrap(); // 8
    let err = Tensor::rand_with_seed(&rt, Dtype::F64, [&s27, &s8], [], 1).unwrap_err();
    let message = format!("{err:?}");
    assert!(
        message.contains("cannot represent this space exactly"),
        "want the bounded-table Err, got: {message}"
    );
}

// Positive shield check: the flagship physics shape [8,8] <- [8,8] is fully
// in-table (8⊗8 closes), so construction AND a same-sides permute round-trip
// work — and because construction admits only clean spaces, the transform layer
// can never reach an escaping pair (the panic-freedom argument in su3.rs).
#[test]
fn su3_adjoint_rank4_constructs_and_permutes() {
    let rt = Runtime::builder().build().unwrap();
    let a8 = Space::su3([((1, 1), 1)]).unwrap();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&a8, &a8], [&a8, &a8], 5).unwrap();
    assert!(!a.data().is_empty());
    let swapped = a.permute(&[1, 0], &[2, 3]).unwrap();
    let back = swapped.permute(&[1, 0], &[2, 3]).unwrap();
    assert_eq!(back.data().len(), a.data().len());
    for (x, y) in back.data().iter().zip(a.data().iter()) {
        assert!(
            (x - y).abs() < 1e-12,
            "rank-4 permute round-trip: {x} vs {y}"
        );
    }
}

#[test]
fn su3_rand_seed_is_reproducible() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [], 3).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [], 3).unwrap();
    assert_eq!(a.data(), b.data());
}
