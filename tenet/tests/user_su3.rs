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
    // Out-of-table irrep is rejected.
    assert!(Space::su3([((5, 5), 1)]).is_err()); // dim(5,5) > 27
    assert_eq!(Space::su3([((2, 0), 1)]).unwrap().dim(), 6);
    assert_eq!(Space::su3([((1, 1), 1)]).unwrap().dim(), 8);
    assert_eq!(Space::su3([((2, 2), 1)]).unwrap().dim(), 27);
}

#[test]
fn su3_tensor_leg_dimensions_feed_network_planners() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let tensor = Tensor::zeros(&rt, Dtype::F64, [&v, &v], [&v]).unwrap();

    assert_eq!(tensor.leg_dims().unwrap(), vec![v.dim(); 3]);
    for axis in 0..3 {
        assert_eq!(tensor.leg_dim(axis).unwrap(), v.dim());
    }
}

#[test]
fn su3_sector_readback_is_nonbreaking() {
    // Stage B3c-2: SU(3) read-back rides dedicated `(p, q)` accessors, NOT an
    // `SectorLabel::Su3` variant — the public enum (and every downstream
    // exhaustive match on it, e.g. finite-torus) stays untouched.
    let v = v();
    // Sorted by internal sector id; `((p, q), deg)` round-trips the constructor.
    let mut got = v.su3_sectors().unwrap();
    got.sort();
    assert_eq!(got, vec![((0, 1), 1), ((1, 0), 2)]);
    // Per-label degeneracy lookups.
    assert_eq!(v.su3_degeneracy(1, 0).unwrap(), Some(2));
    assert_eq!(v.su3_degeneracy(0, 1).unwrap(), Some(1));
    assert_eq!(v.su3_degeneracy(1, 1).unwrap(), None); // 8 absent from this leg
    assert!(v.su3_degeneracy(5, 5).is_err()); // out of table
                                              // dual() reports the dualized external sectors (3 <-> 3̄).
    let mut dual = v.dual().su3_sectors().unwrap();
    dual.sort();
    assert_eq!(dual, vec![((0, 1), 2), ((1, 0), 1)]);
    // Rule guard: SU(3) accessor on a non-SU(3) space errors, and vice versa.
    assert!(Space::su2([(0, 1)]).su3_sectors().is_err());
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

// Stage B3c-2 adjoint axioms on SU(3), which is NON-self-dual (3 <-> 3̄). The
// adjoint materializes through the generic block-relabel sibling (never the
// mult-free conjugate/Structure fold whose non-self-dual coupled-sector
// mislabel was the historical bug), so `.scale(1.0)` is used to force the lazy
// adjoint to materialize into owned coupled data before comparing.
#[test]
fn su3_adjoint_axioms_real() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    // Rank-2 endomorphism V <- V so the adjoint lands on the same coupled space.
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 21).unwrap();
    assert!(!a.data().is_empty());

    // norm(A†) == norm(A): materializes the adjoint (norm reads coupled data).
    let ah = a.adjoint().unwrap();
    assert!(
        (ah.norm().unwrap() - a.norm().unwrap()).abs() < 1e-12,
        "norm(A†) must equal norm(A)"
    );

    // A†† == A at the data level. Force materialization at each dagger with
    // scale(1.0) so this exercises the block-relabel path, not the lazy
    // involution short-circuit.
    let ah_mat = a.adjoint().unwrap().scale(1.0).unwrap();
    assert_ne!(ah_mat.data(), a.data(), "adjoint must move data");
    let ahh = ah_mat.adjoint().unwrap().scale(1.0).unwrap();
    assert_eq!(ahh.data().len(), a.data().len());
    for (x, y) in ahh.data().iter().zip(a.data().iter()) {
        assert!((x - y).abs() < 1e-12, "A†† round-trip: {x} vs {y}");
    }
}

#[test]
fn su3_adjoint_axioms_complex() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    // Complex tensor: adjoint conjugates, so A†† round-trips only if the
    // generic materialization applies (and un-applies) the conjugate.
    let a = Tensor::rand_with_seed(&rt, Dtype::C64, [&v], [&v], 22).unwrap();
    assert!(!a.data_c64().is_empty());

    let ah = a.adjoint().unwrap();
    assert!(
        (ah.norm().unwrap() - a.norm().unwrap()).abs() < 1e-12,
        "norm(A†) must equal norm(A) for c64"
    );

    let ah_mat = a.adjoint().unwrap().scale(1.0).unwrap();
    let ahh = ah_mat.adjoint().unwrap().scale(1.0).unwrap();
    assert_eq!(ahh.data_c64().len(), a.data_c64().len());
    for (x, y) in ahh.data_c64().iter().zip(a.data_c64().iter()) {
        assert!((x - y).norm() < 1e-12, "c64 A†† round-trip: {x} vs {y}");
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
