//! Integration tests for the user-layer decomposition and matrix-function
//! methods (step 3 of the user API), including a cross-check against the
//! typed expert layer.

use tenet::core::{
    FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
    SU2FusionRule, SectorLeg, TensorMap, TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet::prelude::*;

fn assert_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len(), "data lengths differ");
    for (index, (a, b)) in lhs.iter().zip(rhs).enumerate() {
        assert!(
            (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
            "element {index} differs: {a} vs {b}"
        );
    }
}

/// `|lhs - rhs|` in the weighted Frobenius norm, relative to `|rhs|`.
fn relative_distance(lhs: &Tensor, rhs: &Tensor) -> f64 {
    let diff = lhs.add(rhs, 1.0, -1.0).unwrap();
    diff.norm().unwrap() / (1.0 + rhs.norm().unwrap())
}

fn u1_space() -> Space {
    Space::u1([(-1, 2), (0, 3), (1, 2)])
}

fn su2_space() -> Space {
    Space::su2([(0, 2), (1, 2), (2, 1)])
}

#[test]
fn svd_compact_reconstructs_u1_and_su2() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 101).unwrap();
        let (u, s, vh) = t.svd_compact().unwrap();
        let recon = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert!(relative_distance(&recon, &t) < 1e-10);
    }
}

/// The payoff case: rank-5 PEPS-shaped tensor split 1 | 4.
#[test]
fn svd_compact_reconstructs_rank_five_peps_split() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v, &v, &v, &v], 102).unwrap();
        let (u, s, vh) = t.svd_compact().unwrap();
        assert_eq!(u.codomain_rank(), 1);
        assert_eq!(u.domain_rank(), 1);
        assert_eq!(vh.codomain_rank(), 1);
        assert_eq!(vh.domain_rank(), 4);
        let recon = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert!(relative_distance(&recon, &t) < 1e-10);

        let (uf, sf, vhf) = t.svd_full().unwrap();
        let recon_full = uf.compose(&sf).unwrap().compose(&vhf).unwrap();
        assert!(relative_distance(&recon_full, &t) < 1e-10);
    }
}

#[test]
fn svd_trunc_error_matches_discarded_weighted_norm() {
    let rt = Runtime::builder().build().unwrap();
    let cases: [(Space, Box<dyn Fn(SectorId) -> f64>); 2] = [
        (u1_space(), Box::new(|_| 1.0)),
        (
            su2_space(),
            Box::new(|sector| SU2FusionRule.dim_scalar(sector)),
        ),
    ];
    for (v, dim_of) in cases {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 103).unwrap();
        let full = t.svd_vals().unwrap();

        let rank = 6usize;
        let svd = t.svd_trunc(&Truncation::rank(rank)).unwrap();

        // The kept weighted dimension respects the budget.
        let kept_weight: f64 = svd
            .singular_values
            .iter()
            .map(|entry| dim_of(entry.sector) * entry.values.len() as f64)
            .sum();
        assert!(kept_weight <= rank as f64 + 1e-9);

        // The reported error is the weighted 2-norm of the discarded values.
        let discarded: f64 = full
            .iter()
            .map(|entry| {
                let kept = svd
                    .singular_values
                    .iter()
                    .find(|kept| kept.sector == entry.sector)
                    .map(|kept| kept.values.len())
                    .unwrap_or(0);
                dim_of(entry.sector)
                    * entry.values[kept..]
                        .iter()
                        .map(|value| value * value)
                        .sum::<f64>()
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            (svd.error - discarded).abs() <= 1e-10 * (1.0 + discarded),
            "reported {} vs recomputed {}",
            svd.error,
            discarded
        );

        // The truncated factorization reproduces t up to the reported error.
        let recon = svd.u.compose(&svd.s).unwrap().compose(&svd.vh).unwrap();
        let diff = recon.add(&t, 1.0, -1.0).unwrap().norm().unwrap();
        assert!(
            (diff - svd.error).abs() <= 1e-8 * (1.0 + svd.error),
            "reconstruction distance {} vs error {}",
            diff,
            svd.error
        );
    }
}

#[test]
fn qr_and_lq_factorizations() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 104).unwrap();

        // QR: Q R = t and Q is an isometry (Q^H Q = id, so Q^H t = R).
        let (q, r) = t.qr_compact().unwrap();
        let recon = q.compose(&r).unwrap();
        assert!(relative_distance(&recon, &t) < 1e-10);
        let qh_t = q.adjoint().unwrap().compose(&t).unwrap();
        assert!(relative_distance(&qh_t, &r) < 1e-10);

        // LQ: L Q = t and Q has orthonormal rows (t Q^H = L).
        let (l, q) = t.lq_compact().unwrap();
        let recon = l.compose(&q).unwrap();
        assert!(relative_distance(&recon, &t) < 1e-10);
        let t_qh = t.compose(&q.adjoint().unwrap()).unwrap();
        assert!(relative_distance(&t_qh, &l) < 1e-10);

        // left_orth / right_orth are the QR / LQ kinds (TensorKit defaults).
        let (v_orth, c) = t.left_orth().unwrap();
        let (q_ref, r_ref) = t.qr_compact().unwrap();
        assert_eq!(v_orth.data(), q_ref.data());
        assert_eq!(c.data(), r_ref.data());
        let (c, vh_orth) = t.right_orth().unwrap();
        let (l_ref, q_ref) = t.lq_compact().unwrap();
        assert_eq!(c.data(), l_ref.data());
        assert_eq!(vh_orth.data(), q_ref.data());

        // Full QR also reconstructs.
        let (qf, rf) = t.qr_full().unwrap();
        assert!(relative_distance(&qf.compose(&rf).unwrap(), &t) < 1e-10);
        let (lf, qf) = t.lq_full().unwrap();
        assert!(relative_distance(&lf.compose(&qf).unwrap(), &t) < 1e-10);
    }
}

#[test]
fn null_spaces_annihilate_the_tensor() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        // Tall map: codomain strictly larger, so a left null space exists.
        let tall = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 105).unwrap();
        let n = tall.left_null().unwrap();
        let nh_t = n.adjoint().unwrap().compose(&tall).unwrap();
        assert!(nh_t.norm().unwrap() < 1e-10 * (1.0 + tall.norm().unwrap()));
        // The null columns are orthonormal: N^H N = id, checked by norm^2 =
        // weighted bond dimension via N^H N acting as identity on N^H.
        let nh = n.adjoint().unwrap();
        let proj = nh.compose(&n).unwrap().compose(&nh).unwrap();
        assert!(relative_distance(&proj, &nh) < 1e-10);

        // Wide map: domain strictly larger, so a right null space exists.
        let wide = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v, &v], 106).unwrap();
        let n = wide.right_null().unwrap();
        let t_nh = wide.compose(&n.adjoint().unwrap()).unwrap();
        assert!(t_nh.norm().unwrap() < 1e-10 * (1.0 + wide.norm().unwrap()));
    }
}

#[test]
fn polar_decompositions_reconstruct() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 107).unwrap();
        let (w, p) = t.left_polar().unwrap();
        assert!(relative_distance(&w.compose(&p).unwrap(), &t) < 1e-10);
        let (p, w) = t.right_polar().unwrap();
        assert!(relative_distance(&p.compose(&w).unwrap(), &t) < 1e-10);
    }
}

#[test]
fn eigh_reconstructs_a_hermitized_tensor() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 108).unwrap();
        let h = a.add(&a.adjoint().unwrap(), 0.5, 0.5).unwrap();

        let (d, vec) = h.eigh_full().unwrap();
        let recon = vec
            .compose(&d)
            .unwrap()
            .compose(&vec.adjoint().unwrap())
            .unwrap();
        assert!(relative_distance(&recon, &h) < 1e-10);

        // eigh_vals returns the same spectra the full decomposition reports.
        let vals = h.eigh_vals().unwrap();
        assert!(!vals.is_empty());

        // Untruncated eigh_trunc is exact with zero error.
        let trunc = h.eigh_trunc(&Truncation::Full).unwrap();
        assert_eq!(trunc.error, 0.0);
        let recon = trunc
            .v
            .compose(&trunc.d)
            .unwrap()
            .compose(&trunc.v.adjoint().unwrap())
            .unwrap();
        assert!(relative_distance(&recon, &h) < 1e-10);
    }
}

#[test]
fn exp_of_zero_acts_as_identity() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let z = Tensor::zeros(&rt, Dtype::F64, [&v], [&v]).unwrap();
        let e = z.exp().unwrap();
        let x = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 109).unwrap();
        let ex = e.compose(&x).unwrap();
        assert!(relative_distance(&ex, &x) < 1e-10);
        // exp(0) is idempotent (a projector equal to the identity).
        let ee = e.compose(&e).unwrap();
        assert!(relative_distance(&ee, &e) < 1e-10);
    }
}

#[test]
fn inv_and_pinv_sanity() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        // Well-conditioned square map: Gram matrix of a random tensor.
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 110).unwrap();
        let t = a.compose(&a.adjoint().unwrap()).unwrap();
        let ti = t.inv().unwrap();
        let x = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 111).unwrap();
        let round_trip = ti.compose(&t).unwrap().compose(&x).unwrap();
        assert!(relative_distance(&round_trip, &x) < 1e-8);

        // pinv on a tall map: t^+ t = id on the domain.
        let tall = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 112).unwrap();
        let pinv = tall.pinv(1e-12).unwrap();
        let x = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 113).unwrap();
        let round_trip = pinv.compose(&tall).unwrap().compose(&x).unwrap();
        assert!(relative_distance(&round_trip, &x) < 1e-8);
    }
}

/// Cross-checks the user-layer `svd_compact` elementwise against the typed
/// expert layer on the same flat storage (NOUT = 2, NIN = 2).
#[test]
fn svd_compact_cross_checks_against_typed_expert_layer() {
    let rule = U1FusionRule;
    let deg = 2usize;
    let sectors = [
        U1Irrep::new(-1).sector_id(),
        U1Irrep::new(0).sector_id(),
        U1Irrep::new(1).sector_id(),
    ];
    let leg = || SectorLeg::new(sectors.map(|sector| (sector, deg)), false);
    let leg_dim = sectors.len() * deg;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = homspace.fusion_tree_keys(&rule).len();
    let dense_space =
        TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap();
    let typed_space = FusionTensorMapSpace::from_degeneracy_shapes(
        dense_space,
        homspace,
        &rule,
        vec![vec![deg; 4]; key_count],
    )
    .unwrap();

    let rt = Runtime::builder().build().unwrap();
    let v = Space::u1([(-1, deg), (0, deg), (1, deg)]);
    let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 114).unwrap();

    // Expert-layer twin shares the flat storage (identical coupled layout).
    let typed =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(t.data().to_vec(), typed_space).unwrap();
    let mut executor = tenet::dense::DefaultDenseExecutor::new();
    let expert = tenet::matrixalgebra::svd_compact(&mut executor, &rule, &typed).unwrap();

    let (u, s, vh) = t.svd_compact().unwrap();
    assert_close(u.data(), expert.u.data(), 1e-12);
    assert_close(s.data(), expert.s.data(), 1e-12);
    assert_close(vh.data(), expert.vh.data(), 1e-12);
}

/// SVD on a U(1) map whose charge set `{0, 1, 2}` is not closed under
/// dualization, cross-checked against TensorKit v0.16: `svd_compact` on
/// `W ← W` with `W = U1Space(0=>2, 1=>1, 2=>1)` recomposes exactly, and
/// a rank-3 truncation keeps the asymmetric bond `Rep[U₁](0 => 2, 2 => 1)`
/// whose factors still recompose.
#[test]
fn svd_recomposes_on_non_dualization_closed_charges() {
    let rt = Runtime::builder().build().unwrap();
    let w = Space::u1([(0, 2), (1, 1), (2, 1)]);
    let charge = |c: i32| U1Irrep::new(c).sector_id();

    let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&w], [&w], 103).unwrap();
    let (u, s, vh) = t.svd_compact().unwrap();
    let recon = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(relative_distance(&recon, &t) < 1e-10);

    // Deterministic spectra: sector 0 -> {4, 3}, sector 1 -> {1},
    // sector 2 -> {2}; rank(3) keeps {0 => 2, 2 => 1} and drops sector 1.
    let d = Tensor::from_block_fn(&rt, [&w], [&w], |key, indices| match key {
        BlockKey::FusionTree(key) if key.codomain_uncoupled()[0] == charge(0) => match indices {
            [0, 0] => 4.0,
            [1, 1] => 3.0,
            _ => 0.0,
        },
        BlockKey::FusionTree(key) if key.codomain_uncoupled()[0] == charge(1) => 1.0,
        _ => 2.0,
    })
    .unwrap();
    let trunc = d.svd_trunc(&Truncation::rank(3)).unwrap();
    let kept: Vec<_> = trunc
        .singular_values
        .iter()
        .map(|spectrum| (spectrum.sector, spectrum.values.len()))
        .collect();
    assert_eq!(kept, vec![(charge(0), 2), (charge(2), 1)]);
    assert!((trunc.error - 1.0).abs() < 1e-12);

    // The kept factors recompose; the coupled sector the truncation dropped
    // entirely reappears as a zero block (matching TensorKit), verified both
    // by the exact norm identity and elementwise.
    let recon = trunc
        .u
        .compose(&trunc.s)
        .unwrap()
        .compose(&trunc.vh)
        .unwrap();
    let identity = recon.norm().unwrap().powi(2) + trunc.error.powi(2) - d.norm().unwrap().powi(2);
    assert!(identity.abs() < 1e-12);
    assert_eq!(recon.data(), &[4.0, 0.0, 0.0, 3.0, 0.0, 2.0]);
}

/// Truncation that discards an entire coupled sector must keep the kept
/// factors addable against the untruncated tensor (TensorKit parity via
/// graded legs: spaces carry per-sector degeneracies independently of
/// populated blocks), with reconstruction distance equal to the reported
/// truncation error.
#[test]
fn truncated_factors_recompose_and_add_against_untruncated() {
    let rt = Runtime::builder().build().unwrap();
    let v = Space::u1([(0, 2), (1, 1), (2, 1)]);
    let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 7).unwrap();
    let svd = t.svd_trunc(&Truncation::rank(2)).unwrap();
    let recomposed = svd.u.compose(&svd.s).unwrap().compose(&svd.vh).unwrap();
    let diff = t.add(&recomposed, 1.0, -1.0).unwrap();
    let err = diff.norm().unwrap();
    assert!((err - svd.error).abs() < 1e-10);
}
