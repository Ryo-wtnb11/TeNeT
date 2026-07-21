//! ADVERSARIAL (refute/b3c2-verify): edge-case termination + correctness for
//! the SU(3) source-transform contraction canonicalization (Stage B3c-2).
//!
//! The shipped gate suite (`su3_b3c2_gates.rs`) covers "generic" non-core
//! arrangements but NOT the three degenerate arrangements the bounded-recursion
//! termination argument specifically hinges on:
//!   * all legs contracted  -> rank-0 scalar result,
//!   * no legs contracted    -> outer product,
//!   * single-leg operands.
//! Each must (a) terminate (bounded recursion, no hang) and (b) match an
//! independent reference. Correctness is cross-checked two ways: against an
//! explicit permute-then-core composition (independent axis bookkeeping, same
//! as the gates) AND, for the scalar case, against a permute-FREE anchor
//! (core-contract + generic `tr`) that exercises entirely different machinery.

use tenet::prelude::*;

fn v() -> Space {
    Space::su3([((1, 0), 2), ((0, 1), 1)]).unwrap()
}

/// The adjoint 8 with degeneracy 2 (carries the N(8,8,8)=2 OM vertex).
fn eight2() -> Space {
    Space::su3([((1, 1), 2)]).unwrap()
}

fn assert_close(a: &Tensor, b: &Tensor, tol: f64, what: &str) {
    assert_eq!(a.data().len(), b.data().len(), "{what}: length");
    assert!(!a.data().is_empty(), "{what}: must be non-trivial");
    for (x, y) in a.data().iter().zip(b.data().iter()) {
        assert!((x - y).abs() < tol, "{what}: {x} vs {y}");
    }
}

/// All legs contracted -> rank-0 scalar. Termination + two references.
#[test]
fn su3_canon_scalar_all_contracted() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 61).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 62).unwrap();

    // a[0](codomain) <-> b[1](domain); a[1](domain) <-> b[0](codomain).
    // Non-core on both operands -> canonicalization fires; result is rank 0.
    let got = a.contract(&b, &[0, 1], &[1, 0]).unwrap();
    assert_eq!(got.rank(), 0, "all-contracted must be a scalar");
    let got_s = got.scalar().unwrap().try_f64().unwrap();

    // Reference 1: explicit permute-then-core (independent bookkeeping).
    let a_core = a.permute(&[], &[0, 1]).unwrap();
    let b_core = b.permute(&[1, 0], &[]).unwrap();
    let want = a_core
        .contract(&b_core, &[0, 1], &[0, 1])
        .unwrap()
        .scalar()
        .unwrap()
        .try_f64()
        .unwrap();
    assert!(
        (got_s - want).abs() < 1e-12,
        "scalar vs permute+core: {got_s} vs {want}"
    );

    // Reference 2 (permute-FREE anchor): the same closed diagram equals
    // tr(A o B). C = A.contract(B, [1], [0]) is ALREADY core form (lhs whole
    // domain, rhs whole codomain, in order) so canonicalization does NOT fire;
    // C.tr() then closes A_codomain with B_domain via the generic trace. The
    // two closed edges are identical to the scalar contraction's, so the
    // answers must agree — through disjoint machinery.
    let c = a.contract(&b, &[1], &[0]).unwrap();
    let anchor = match c.tr().unwrap() {
        Scalar::F64(x) => x,
        other => panic!("expected f64, got {other:?}"),
    };
    assert!(
        (got_s - anchor).abs() < 1e-12,
        "scalar vs core-contract+tr anchor: {got_s} vs {anchor}"
    );
}

/// No legs contracted, but a non-empty domain forces the canonicalization
/// branch (a pure ket would already be canonical) -> outer product, rank 4.
#[test]
fn su3_canon_outer_product() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 63).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 64).unwrap();

    // canonical_lhs = domain axes = [1]; lhs_axes = [] != [1] -> fires.
    let got = a.contract(&b, &[], &[]).unwrap();
    assert_eq!(got.rank(), 4, "outer product rank");

    let a_core = a.permute(&[0, 1], &[]).unwrap();
    let b_core = b.permute(&[], &[0, 1]).unwrap();
    let want = a_core.contract(&b_core, &[], &[]).unwrap();
    assert_close(&got, &want, 1e-12, "outer product vs permute+core");
}

/// A single CONTRACTED leg that needs a bend: contract lhs's codomain axis 0
/// (not its domain) so the arrangement is non-core and canonicalization fires,
/// while both operands stay non-empty (rank-2 endomorphisms).
#[test]
fn su3_canon_single_leg() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 65).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 66).unwrap();

    // canonical_lhs = domain axes = [1]; lhs_axes = [0] != [1] -> fires.
    let got = a.contract(&b, &[0], &[1]).unwrap();
    assert_eq!(got.rank(), 2, "single-leg result rank");

    // a_core : [a1] <- [a0] (contracted leg a0 in domain); b_core : [b1] <- [b0]
    // (contracted leg b1 pulled to codomain).
    let a_core = a.permute(&[1], &[0]).unwrap();
    let b_core = b.permute(&[1], &[0]).unwrap();
    let want = a_core.contract(&b_core, &[1], &[0]).unwrap();
    assert_close(&got, &want, 1e-12, "single-leg vs permute+core");
}

/// Termination stress: the canonicalization recursion must bottom out in one
/// level even when combined with a lazy-adjoint operand AND a degenerate
/// (all-contracted) arrangement. If recursion were unbounded this would hang;
/// the assertion simply reaching this line proves termination.
#[test]
fn su3_canon_adjoint_plus_scalar_terminates() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 67).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 68).unwrap();

    // a† : [v]<-[v]; fully contract against b (non-core, lazy adjoint operand).
    let got = a.adjoint().unwrap().contract(&b, &[0, 1], &[1, 0]).unwrap();
    assert_eq!(got.rank(), 0);
    let got_s = got.scalar().unwrap().try_f64().unwrap();

    // Reference: eager-materialized adjoint + explicit permute + core.
    let a_dag = a.adjoint().unwrap().scale(1.0).unwrap();
    let a_core = a_dag.permute(&[], &[0, 1]).unwrap();
    let b_core = b.permute(&[1, 0], &[]).unwrap();
    let want = a_core
        .contract(&b_core, &[0, 1], &[0, 1])
        .unwrap()
        .scalar()
        .unwrap()
        .try_f64()
        .unwrap();
    assert!(
        (got_s - want).abs() < 1e-12,
        "adjoint+scalar vs reference: {got_s} vs {want}"
    );
}

/// Generalizes the shipped closed-form-2x2 SVD spot checks: a RANDOM-filled OM
/// tensor (8-deg-2 legs, so every internal 8 fusion carries the N(8,8,8)=2
/// vertex and the coupled blocks are genuine dense matrices, not 1x1/2x2 by
/// hand). Correctness is pinned definitionally — `U·S·V† == t` and `U†U == I`
/// on the bond make (U,S,V) an SVD regardless of the impl's matricization —
/// and `svd_vals` is cross-checked against `svd_compact`'s spectrum, so the two
/// separate generic code paths must agree value-for-value.
#[test]
fn su3_svd_random_om_reconstructs_and_isometric_and_vals_agree() {
    let rt = Runtime::builder().build().unwrap();
    let e = eight2();
    // t : [8,8] <- [8] ; codomain fusion 8⊗8 ∋ 8 (N=2) exercises OM.
    let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&e, &e], [&e], 71).unwrap();

    let (u, s, vh) = t.svd_compact().unwrap();
    // Reconstruction via `compose` (the bond-duality-aware idiom used by the
    // mult-free semantic suite); norm-based so it is layout-robust.
    let recon = u.compose(&s).unwrap().compose(&vh).unwrap();
    let diff = recon.add(&t, 1.0, -1.0).unwrap().norm().unwrap();
    assert!(
        diff <= 1e-10 * (1.0 + t.norm().unwrap()),
        "random OM svd reconstruction error {diff}"
    );

    // Bond isometry U†U == id on the bond space (all coupled sectors incl. OM).
    let mid = u.domain_spaces();
    let mid_refs: Vec<&Space> = mid.iter().collect();
    let id = Tensor::id(&rt, Dtype::F64, mid_refs).unwrap();
    let utu = u.adjoint().unwrap().compose(&u).unwrap();
    let iso_err = utu.add(&id, 1.0, -1.0).unwrap().norm().unwrap();
    assert!(iso_err <= 1e-10, "random OM U†U != id ({iso_err})");

    // svd_vals must equal the factor-only svd_trunc(Full) spectrum across
    // separate Generic entry points.
    let _ = &s;
    let vals = t.svd_vals().unwrap();
    let full = t.svd_trunc(&Truncation::Full).unwrap();
    for sc in &full.singular_values {
        let m = vals
            .iter()
            .find(|e| e.sector == sc.sector)
            .expect("matching sector in svd_vals");
        assert_eq!(m.values.len(), sc.values.len(), "sector spectrum length");
        for (x, y) in m.values.iter().zip(&sc.values) {
            assert!((x - y).abs() < 1e-10, "svd_vals vs svd_trunc: {x} vs {y}");
        }
    }
}
