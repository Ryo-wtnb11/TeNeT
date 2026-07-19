//! Stage B3c-2 Gate 3: SU(3) svd / qr / trace at the `tenet::Tensor` layer.
//!
//! The per-coupled-sector spectra are pinned against an INDEPENDENT dense SVD
//! written inline (closed-form 2x2 singular values) — never against the
//! implementation under test. Reconstruction, per-sector isometry, the
//! dim-weighted truncation convention, and the dim-weighted trace are all
//! checked on both a plain (3 ⊕ 3̄) case and an OM case (N(8,8,8) = 2).

use tenet::prelude::*;
use tenet_core::Su3FusionRule;

fn v() -> Space {
    Space::su3([((1, 0), 2), ((0, 1), 1)]).unwrap()
}

fn eight() -> Space {
    Space::su3([((1, 1), 1)]).unwrap()
}

fn vertex(tree: &tenet_core::FusionTreeKey) -> usize {
    tree.vertices().first().map(|s| s.get()).unwrap_or(0)
}

/// Singular values of a 2x2 real matrix, descending — the independent
/// reference: sigma_pm = sqrt((t ± sqrt(t² − 4 det²)) / 2) with
/// t = ||M||_F² and det = det(M).
fn svd2x2(m: [[f64; 2]; 2]) -> [f64; 2] {
    let t = m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[1][0] * m[1][0] + m[1][1] * m[1][1];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let gap = (t * t - 4.0 * det * det).max(0.0).sqrt();
    [((t + gap) / 2.0).sqrt(), ((t - gap) / 2.0).sqrt()]
}

/// t : [v] <- [v] with hand-set coupled blocks: 3-sector 2x2 `m3` (block
/// index [i, j] = codomain deg, domain deg), 3̄-sector 1x1 `m3b`.
fn t_v(rt: &Runtime, m3: [[f64; 2]; 2], m3b: f64) -> Tensor {
    let c3 = Su3FusionRule::new().sector_of(1, 0).unwrap();
    let v = v();
    Tensor::from_block_fn(rt, [&v], [&v], move |key, idx| match key {
        BlockKey::FusionTree(k) => {
            if k.codomain_tree().coupled() == c3 {
                m3[idx[0]][idx[1]]
            } else {
                m3b
            }
        }
        _ => 0.0,
    })
    .unwrap()
}

/// t8 : [8,8] <- [8,8] (deg 1): the coupled-8 matricization is the 2x2 OM
/// vertex matrix `q[mu-1][nu-1]` (vertex ids are {1, 2}); every other coupled
/// sector (1, 10, 10̄, 27) is the 1x1 block `g`.
fn t_8(rt: &Runtime, q: [[f64; 2]; 2], g: f64) -> Tensor {
    let c8 = Su3FusionRule::new().sector_of(1, 1).unwrap();
    let e = eight();
    Tensor::from_block_fn(rt, [&e, &e], [&e, &e], move |key, _| match key {
        BlockKey::FusionTree(k) => {
            if k.codomain_tree().coupled() == c8 {
                q[vertex(k.codomain_tree()) - 1][vertex(k.domain_tree()) - 1]
            } else {
                g
            }
        }
        _ => 0.0,
    })
    .unwrap()
}

fn assert_data_close(a: &Tensor, b: &Tensor, tol: f64, what: &str) {
    assert_eq!(a.data().len(), b.data().len(), "{what}: length");
    assert!(!a.data().is_empty(), "{what}: non-trivial");
    for (x, y) in a.data().iter().zip(b.data().iter()) {
        assert!((x - y).abs() < tol, "{what}: {x} vs {y}");
    }
}

/// Per-sector isometry through public API only: `gram` must be the identity
/// on the bond — Hermitian (data-level `g† == g`) with ALL singular values
/// exactly 1; a Hermitian PSD matrix with unit spectrum is the identity.
fn assert_is_identity(gram: &Tensor, expected_kept: &[usize], what: &str) {
    let gh = gram.adjoint().unwrap().scale(1.0).unwrap();
    assert_data_close(&gh, gram, 1e-12, &format!("{what}: hermiticity"));
    let mut counts: Vec<usize> = Vec::new();
    for entry in gram.svd_vals().unwrap() {
        counts.push(entry.values.len());
        for value in &entry.values {
            assert!((value - 1.0).abs() < 1e-12, "{what}: spectrum {value} != 1");
        }
    }
    counts.sort_unstable();
    let mut expected = expected_kept.to_vec();
    expected.sort_unstable();
    assert_eq!(counts, expected, "{what}: kept counts per sector");
}

#[test]
fn su3_svd_vals_match_independent_dense_svd() {
    let rt = Runtime::builder().build().unwrap();
    let m3 = [[3.0, 1.0], [2.0, 4.0]];
    let m3b = -5.0;
    let t = t_v(&rt, m3, m3b);

    let want3 = svd2x2(m3);
    let c3 = Su3FusionRule::new().sector_of(1, 0).unwrap();
    let spectra = t.svd_vals().unwrap();
    assert_eq!(spectra.len(), 2, "two coupled sectors");
    for entry in &spectra {
        if entry.sector == c3 {
            assert_eq!(entry.values.len(), 2);
            assert!((entry.values[0] - want3[0]).abs() < 1e-12);
            assert!((entry.values[1] - want3[1]).abs() < 1e-12);
        } else {
            assert_eq!(entry.values.len(), 1);
            assert!((entry.values[0] - m3b.abs()).abs() < 1e-12);
        }
    }
}

#[test]
fn su3_svd_vals_om_sector_match_independent_dense_svd() {
    let rt = Runtime::builder().build().unwrap();
    // Non-normal OM vertex matrix so the SVD is not just |diagonal|.
    let q = [[3.0, 1.0], [0.0, 5.0]];
    let t = t_8(&rt, q, 2.0);

    let c8 = Su3FusionRule::new().sector_of(1, 1).unwrap();
    let want = svd2x2(q);
    let spectra = t.svd_vals().unwrap();
    let om = spectra
        .iter()
        .find(|entry| entry.sector == c8)
        .expect("coupled-8 spectrum");
    // The coupled-8 block stacks BOTH OM vertex trees: a 2x2 matrix, not two
    // 1x1 blocks. Singular values are row/col-order invariant, so the
    // comparison is layout-free.
    assert_eq!(om.values.len(), 2, "OM sector must stack both vertices");
    assert!((om.values[0] - want[0]).abs() < 1e-12);
    assert!((om.values[1] - want[1]).abs() < 1e-12);
    // Every non-8 coupled sector is 1x1 with value |g| = 2.
    for entry in spectra.iter().filter(|entry| entry.sector != c8) {
        assert_eq!(entry.values.len(), 1);
        assert!((entry.values[0] - 2.0).abs() < 1e-12);
    }
}

#[test]
fn su3_svd_compact_reconstructs_and_is_isometric() {
    let rt = Runtime::builder().build().unwrap();
    let t = t_v(&rt, [[3.0, 1.0], [2.0, 4.0]], -5.0);
    let (u, s, vh) = t.svd_compact().unwrap();

    // U · S · V† == t (exact layout: same hom space as t).
    let recon = u
        .contract(&s, &[1], &[0])
        .unwrap()
        .contract(&vh, &[1], &[0])
        .unwrap();
    assert_data_close(&recon, &t, 1e-12, "svd_compact reconstruction");

    // Per-sector isometry: U†U == I_W and Vh·Vh† == I_W (kept = [2, 1]).
    let gram_u = u.adjoint().unwrap().contract(&u, &[1], &[0]).unwrap();
    assert_is_identity(&gram_u, &[2, 1], "U†U");
    let gram_v = vh.contract(&vh.adjoint().unwrap(), &[1], &[0]).unwrap();
    assert_is_identity(&gram_v, &[2, 1], "VhVh†");
}

#[test]
fn su3_svd_compact_om_reconstructs() {
    let rt = Runtime::builder().build().unwrap();
    let t = t_8(&rt, [[3.0, 1.0], [0.0, 5.0]], 2.0);
    let (u, s, vh) = t.svd_compact().unwrap();
    // u : [8,8] <- [W]; the bond is axis 2.
    let recon = u
        .contract(&s, &[2], &[0])
        .unwrap()
        .contract(&vh, &[2], &[0])
        .unwrap();
    assert_data_close(&recon, &t, 1e-12, "OM svd_compact reconstruction");
    // Bond isometry across ALL coupled sectors incl. the stacked OM 8.
    let gram_u = u.adjoint().unwrap().contract(&u, &[1, 2], &[0, 1]).unwrap();
    assert_is_identity(&gram_u, &[1, 2, 1, 1, 1], "OM U†U");
}

/// Truncation follows the mult-free convention exactly: selection magnitude
/// is the bare value, the budget/error weight is the quantum dimension
/// (dim 3 = dim 3̄ = 3), kept sets are per-sector prefixes, `error` is the
/// dim-weighted 2-norm of everything discarded.
#[test]
fn su3_svd_trunc_dim_weighted() {
    let rt = Runtime::builder().build().unwrap();
    // Diagonal blocks so the singular values are exactly {10, 0.5} on the
    // 3-sector and {7} on 3̄.
    let t = t_v(&rt, [[10.0, 0.0], [0.0, 0.5]], 7.0);
    let c3 = Su3FusionRule::new().sector_of(1, 0).unwrap();

    // Rank(3): the weighted budget fits exactly ONE multiplet (dim 3): the
    // largest value 10 (3-sector). 7 (would push to 6) and 0.5 are discarded.
    let out = t.svd_trunc(&Truncation::Rank(3)).unwrap();
    assert_eq!(out.singular_values.len(), 1, "only the 3-sector survives");
    assert_eq!(out.singular_values[0].sector, c3);
    assert_eq!(out.singular_values[0].values, vec![10.0]);
    let want_error = (3.0f64 * (7.0 * 7.0 + 0.5 * 0.5)).sqrt();
    assert!(
        (out.error - want_error).abs() < 1e-12,
        "dim-weighted discard error: {} vs {want_error}",
        out.error
    );
    // The truncated triple reconstructs the best dim-weighted approximation:
    // 3-sector diag(10, 0), 3̄ zeroed.
    let recon = out
        .u
        .contract(&out.s, &[1], &[0])
        .unwrap()
        .contract(&out.vh, &[1], &[0])
        .unwrap();
    let want = t_v(&rt, [[10.0, 0.0], [0.0, 0.0]], 0.0);
    assert_data_close(&recon, &want, 1e-12, "Rank(3) truncated reconstruction");

    // Rank(6): budget fits two multiplets -> keeps 10 AND 7, discards 0.5.
    let out = t.svd_trunc(&Truncation::Rank(6)).unwrap();
    let mut kept: Vec<f64> = out
        .singular_values
        .iter()
        .flat_map(|entry| {
            assert_eq!(entry.values.len(), 1, "one value kept per sector");
            entry.values.clone()
        })
        .collect();
    kept.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(kept, vec![7.0, 10.0]);
    let want_error = (3.0f64 * 0.5 * 0.5).sqrt();
    assert!((out.error - want_error).abs() < 1e-12);

    // Full keeps everything with zero error.
    let out = t.svd_trunc(&Truncation::Full).unwrap();
    assert!((out.error - 0.0).abs() < 1e-15);
    let recon = out
        .u
        .contract(&out.s, &[1], &[0])
        .unwrap()
        .contract(&out.vh, &[1], &[0])
        .unwrap();
    assert_data_close(&recon, &t, 1e-12, "Full truncation reconstruction");
}

#[test]
fn su3_qr_reconstructs_and_left_orth_is_qr() {
    let rt = Runtime::builder().build().unwrap();
    let t = t_v(&rt, [[3.0, 1.0], [2.0, 4.0]], -5.0);

    let (q, r) = t.qr_compact().unwrap();
    let recon = q.contract(&r, &[1], &[0]).unwrap();
    assert_data_close(&recon, &t, 1e-12, "qr reconstruction");
    let gram = q.adjoint().unwrap().contract(&q, &[1], &[0]).unwrap();
    assert_is_identity(&gram, &[2, 1], "Q†Q");

    // TK 0.17 naming: left_orth defaults to the compact QR.
    let (v_orth, c) = t.left_orth().unwrap();
    assert_data_close(&v_orth, &q, 1e-15, "left_orth == qr_compact (V)");
    assert_data_close(&c, &r, 1e-15, "left_orth == qr_compact (C)");

    // right_orth (compact LQ): t == L · Q with Q an isometry on the right.
    let (l, qr_) = t.right_orth().unwrap();
    let recon = l.contract(&qr_, &[1], &[0]).unwrap();
    assert_data_close(&recon, &t, 1e-12, "lq reconstruction");
    let gram = qr_.contract(&qr_.adjoint().unwrap(), &[1], &[0]).unwrap();
    assert_is_identity(&gram, &[2, 1], "QQ† (lq)");
}

#[test]
fn su3_qr_om_reconstructs() {
    let rt = Runtime::builder().build().unwrap();
    let t = t_8(&rt, [[3.0, 1.0], [0.0, 5.0]], 2.0);
    let (q, r) = t.qr_compact().unwrap();
    let recon = q.contract(&r, &[2], &[0]).unwrap();
    assert_data_close(&recon, &t, 1e-12, "OM qr reconstruction");
}

/// `tr` == the hand-computed quantum-dimension-weighted sum of the diagonal
/// blocks, on both the plain and the OM case.
#[test]
fn su3_trace_matches_hand_computed_dim_weighted_sum() {
    let rt = Runtime::builder().build().unwrap();

    // Plain: tr = dim(3)·tr(m3) + dim(3̄)·m3b = 3·(3+4) + 3·(−5) = 6.
    let t = t_v(&rt, [[3.0, 1.0], [2.0, 4.0]], -5.0);
    let got = match t.tr().unwrap() {
        Scalar::F64(x) => x,
        other => panic!("expected f64 trace, got {other:?}"),
    };
    assert!((got - 6.0).abs() < 1e-12, "tr = {got} vs hand 6");

    // OM: 8⊗8 = 1 ⊕ 8 (N=2) ⊕ 10 ⊕ 10̄ ⊕ 27. Only tree-diagonal blocks
    // contribute — for the doubled 8 that is one 1x1 block PER VERTEX, so
    // tr = (1 + 10 + 10 + 27)·g + dim(8)·(q11 + q22) = 48·1 + 8·(3+5) = 112.
    let t8 = t_8(&rt, [[3.0, 9.0], [4.0, 5.0]], 1.0);
    let got = match t8.tr().unwrap() {
        Scalar::F64(x) => x,
        other => panic!("expected f64 trace, got {other:?}"),
    };
    assert!((got - 112.0).abs() < 1e-12, "OM tr = {got} vs hand 112");
}

/// c64 svd rides the same generic engine.
#[test]
fn su3_svd_compact_c64_reconstructs() {
    let rt = Runtime::builder().build().unwrap();
    let v = v();
    let t = Tensor::rand_with_seed(&rt, Dtype::C64, [&v], [&v], 55).unwrap();
    let (u, s, vh) = t.svd_compact().unwrap();
    let recon = u
        .contract(&s, &[1], &[0])
        .unwrap()
        .contract(&vh, &[1], &[0])
        .unwrap();
    assert_eq!(recon.data_c64().len(), t.data_c64().len());
    for (x, y) in recon.data_c64().iter().zip(t.data_c64().iter()) {
        assert!((x - y).norm() < 1e-12, "c64 svd reconstruction: {x} vs {y}");
    }
}
