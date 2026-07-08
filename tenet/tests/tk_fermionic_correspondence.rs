//! Numeric correspondence against TensorKit for fermionic (FZ2) contractions,
//! including a diagonal `S` from an SVD used in a closed loop with dual legs
//! (which fires the supertrace twist). The reference numbers come from running
//! TensorKit 0.16 on identical degeneracy-1 FZ2 maps (see
//! `scratchpad/tk_fz2_reference.jl`); every value here is a physical, basis-
//! independent scalar, so a mismatch is a genuine convention divergence — not a
//! layout/gauge artifact. Guards that TeNeT's fermionic contract, and the
//! diagonal-`contract` fast path on top of it, agree with TensorKit.

use tenet::prelude::*;
use tenet_network::tensor;

/// FZ2 map `V <- V`, degeneracy 1, with `even`/`odd` block values — the exact
/// tensor `build(f)` produces in the Julia reference.
fn fz2_map(rt: &Runtime, even: f64, odd: f64) -> Tensor {
    let v = Space::fz2([(0, 1), (1, 1)]);
    Tensor::from_block_fn(rt, [&v], [&v], move |key, _| {
        let BlockKey::FusionTree(key) = key else {
            return 0.0;
        };
        if key.codomain_uncoupled()[0].id() == 0 {
            even
        } else {
            odd
        }
    })
    .unwrap()
}

fn scalar(t: Tensor) -> f64 {
    t.scalar().unwrap().try_f64().unwrap()
}

#[test]
fn fz2_contractions_match_tensorkit() {
    let rt = Runtime::builder().build().unwrap();
    let a = fz2_map(&rt, 1.0, 4.0);
    let b = fz2_map(&rt, 2.0, 1.5);
    let c = fz2_map(&rt, 0.5, 2.5);
    let t = fz2_map(&rt, 3.0, 2.0);

    // S from an SVD is a Data::Diagonal factor; singular values = |T| per sector.
    let (_, s, _) = t.svd_compact().unwrap();
    let sv = s.svd_vals().unwrap();
    for entry in &sv {
        let expect = if entry.sector.id() == 0 { 3.0 } else { 2.0 };
        assert!(
            (entry.values[0] - expect).abs() < 1e-12,
            "singular value for sector {}: {} vs {expect}",
            entry.sector.id(),
            entry.values[0]
        );
    }

    // (1) dense supertrace loop tr(A B) = 2 - 6 = -4.
    let s1 = scalar(tensor!([] = a[i; j] * b[j; i]).unwrap());
    assert!((s1 - (-4.0)).abs() < 1e-12, "s1 = {s1}");

    // (2) dense three-map loop tr(A B C) = 1 - 15 = -14.
    let s2 = scalar(tensor!([] = a[i; j] * b[j; k] * c[k; i]).unwrap());
    assert!((s2 - (-14.0)).abs() < 1e-12, "s2 = {s2}");

    // (3) diagonal S in a closed fermionic loop A[a;b] S[b;c] B[c;a] = 6 - 12 = -6.
    let s3 = scalar(tensor!([] = a[i; j] * s[j; k] * b[k; i]).unwrap());
    assert!((s3 - (-6.0)).abs() < 1e-12, "s3 = {s3}");

    // (4) S on the leading side: S[a;b] A[b;c] B[c;a] = -6.
    let s4 = scalar(tensor!([] = s[i; j] * a[j; k] * b[k; i]).unwrap());
    assert!((s4 - (-6.0)).abs() < 1e-12, "s4 = {s4}");

    // Force the single-axis diagonal `contract` fast path explicitly (rather than
    // whatever order the macro picks): A[i;j] · S[j;k] goes through scale+permute
    // + the fermionic twist fold, and closing with B must still give TK's -6.
    let as_ = a.contract(&s, &[1], &[0]).unwrap();
    let s3_fast = scalar(tensor!([] = as_[i; k] * b[k; i]).unwrap());
    assert!((s3_fast - (-6.0)).abs() < 1e-12, "s3_fast = {s3_fast}");
    // And the leading (D * A) order on `s`.
    let sa = s.contract(&a, &[1], &[0]).unwrap();
    let s4_fast = scalar(tensor!([] = sa[i; k] * b[k; i]).unwrap());
    assert!((s4_fast - (-6.0)).abs() < 1e-12, "s4_fast = {s4_fast}");
}
