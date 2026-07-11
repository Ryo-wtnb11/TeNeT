//! Stage B3c-2 gates: SU(3) source-transform (recoupling) contractions and
//! svd/qr/trace wiring at the top-level `tenet::Tensor` layer.
//!
//! Gate 1 (route equivalence): a non-core contraction must equal the explicit
//! composition "generic permute (TK-pinned, B3a) then core contract (B3c-1)"
//! at 1e-12 — proving the source-transform wiring adds no mathematics. The
//! reference side performs its OWN axis bookkeeping, so a mapping bug in the
//! built-in route cannot cancel.
//!
//! Gate 3 (svd/qr/trace) lives in `su3_factorize.rs`.

use tenet::prelude::*;

/// Fundamental 3 (deg 2) + conjugate 3̄ (deg 1) — non-self-dual content.
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

/// OM-sector route equivalence: A : [8] <- [8,8] contracted over its FIRST
/// domain leg only (axis 1) with B : [8,8] <- [8] over its SECOND codomain leg
/// (axis 1). Neither operand is in core form, and every internal line runs
/// through 8⊗8 ∋ 8 with N(8,8,8) = 2 — the OM vertex participates in both the
/// permutes and the coupled GEMM.
#[test]
fn su3_route_equivalence_om_sector() {
    let rt = Runtime::builder().build().unwrap();
    let e = eight2();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&e], [&e, &e], 31).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&e, &e], [&e], 32).unwrap();

    // Built-in source-transform route.
    let got = a.contract(&b, &[1], &[1]).unwrap();

    // Independent reference: permute both operands to core/compose form with
    // the TK-pinned generic permute, then the core contract.
    // lhs: open axes [0, 2] -> codomain, contracted [1] -> domain.
    let a_core = a.permute(&[0, 2], &[1]).unwrap();
    // rhs: contracted [1] -> codomain, open [0, 2] -> domain.
    let b_core = b.permute(&[1], &[0, 2]).unwrap();
    let want = a_core.contract(&b_core, &[2], &[0]).unwrap();

    assert_close(&got, &want, 1e-12, "OM route equivalence");
    // Sanity: the result is [8, 8] <- [8, 8] (open lhs axes; open rhs axes).
    assert_eq!(got.rank(), 4);
    assert_eq!(got.codomain_rank(), 2);
}

/// Both operands need a genuine repartition (codomain <-> domain bends), not
/// just a same-side swap: contract A's CODOMAIN leg with B's DOMAIN leg.
#[test]
fn su3_route_equivalence_bend_both_operands() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 33).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 34).unwrap();

    // A[c;a] * B[b;c]: contract A axis 0 (codomain) with B axis 1 (domain).
    let got = a.contract(&b, &[0], &[1]).unwrap();

    // Reference: bend each operand to core form, then core contract.
    let a_core = a.permute(&[1], &[0]).unwrap(); // codomain [a], domain [c]
    let b_core = b.permute(&[1], &[0]).unwrap(); // codomain [c], domain [b]
    let want = a_core.contract(&b_core, &[1], &[0]).unwrap();

    assert_close(&got, &want, 1e-12, "bend-both route equivalence");
}

/// Non-trivial leg reordering on both operands at higher rank: rank-3 x rank-3
/// over two axes given in NON-ascending order on the lhs, crossing sides on
/// the rhs.
#[test]
fn su3_route_equivalence_reordered_axes() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 35).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v, &v], 36).unwrap();

    // Contract a[1] with b[2] and a[2] with b[0]: the lhs pairs a codomain leg
    // AND its domain leg, the rhs pairs its codomain leg and a domain leg —
    // both operands need bends and a reorder.
    let got = a.contract(&b, &[1, 2], &[2, 0]).unwrap();

    // Reference with independent bookkeeping.
    let a_core = a.permute(&[0], &[1, 2]).unwrap();
    let b_core = b.permute(&[2, 0], &[1]).unwrap();
    let want = a_core.contract(&b_core, &[1, 2], &[0, 1]).unwrap();

    assert_close(&got, &want, 1e-12, "reordered-axes route equivalence");
    assert_eq!(got.rank(), 2);
}

/// c64 goes through the same route (recoupling scalars stay f64).
#[test]
fn su3_route_equivalence_c64() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let a = Tensor::rand_with_seed(&rt, Dtype::C64, [&v], [&v], 37).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::C64, [&v], [&v], 38).unwrap();

    let got = a.contract(&b, &[0], &[1]).unwrap();
    let want = a
        .permute(&[1], &[0])
        .unwrap()
        .contract(&b.permute(&[1], &[0]).unwrap(), &[1], &[0])
        .unwrap();

    assert_eq!(got.data_c64().len(), want.data_c64().len());
    assert!(!got.data_c64().is_empty());
    for (x, y) in got.data_c64().iter().zip(want.data_c64().iter()) {
        assert!((x - y).norm() < 1e-12, "c64 route equivalence: {x} vs {y}");
    }
}

/// A lazy-adjoint operand rides the same source-transform route after its
/// mislabel-proof materialization: A† contracted over a non-core axis pair
/// equals the explicitly materialized + permuted reference.
#[test]
fn su3_route_equivalence_with_adjoint_operand() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 39).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 40).unwrap();

    // a† : [v] <- [v, v]; contract its codomain leg (axis 0) with b's domain
    // leg (axis 2) — codomain-against-domain, so the dualities pair up.
    let got = a.adjoint().unwrap().contract(&b, &[0], &[2]).unwrap();

    // Reference: eager-materialized adjoint, explicit permutes, core contract.
    let a_dag = a.adjoint().unwrap().scale(1.0).unwrap();
    let want = a_dag
        .permute(&[1, 2], &[0])
        .unwrap()
        .contract(&b.permute(&[2], &[0, 1]).unwrap(), &[2], &[0])
        .unwrap();

    assert_close(&got, &want, 1e-12, "adjoint-operand route equivalence");
}
