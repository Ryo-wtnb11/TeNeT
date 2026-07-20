//! Integration tests for the user-layer decomposition and matrix-function
//! methods (step 3 of the user API), including a cross-check against the
//! typed expert layer.

use std::sync::Arc;

use num_complex::Complex64;
use tenet::core::{
    FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
    SU2FusionRule, SectorLeg, TensorMap, TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet::operations::OperationError;
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

fn assert_isomorphic_inverse(rt: &Runtime, dtype: Dtype, leg: Space) {
    let fused = leg.fuse(&leg).unwrap();
    let map = Tensor::isomorphism(rt, dtype, [&fused], [&leg, &leg]).unwrap();
    let inverse = map.inv().unwrap();

    assert_eq!(inverse.codomain_spaces(), map.domain_spaces());
    assert_eq!(inverse.domain_spaces(), map.codomain_spaces());
    assert!(
        relative_distance(
            &inverse.compose(&map).unwrap(),
            &Tensor::id(rt, dtype, [&leg, &leg]).unwrap()
        ) < 1e-10
    );
    assert!(
        relative_distance(
            &map.compose(&inverse).unwrap(),
            &Tensor::id(rt, dtype, [&fused]).unwrap()
        ) < 1e-10
    );
}

fn assert_left_polar_contract(rt: &Runtime, tensor: &Tensor, domain: &Space) {
    let (isometry, positive) = tensor.left_polar().unwrap();
    assert!(relative_distance(&isometry.compose(&positive).unwrap(), tensor) < 1e-10);
    let identity = Tensor::id(rt, tensor.dtype(), [domain]).unwrap();
    let gram = isometry.adjoint().unwrap().compose(&isometry).unwrap();
    assert!(relative_distance(&gram, &identity) < 1e-10);
}

fn assert_right_polar_contract(rt: &Runtime, tensor: &Tensor, codomain: &Space) {
    let (positive, isometry) = tensor.right_polar().unwrap();
    assert!(relative_distance(&positive.compose(&isometry).unwrap(), tensor) < 1e-10);
    let identity = Tensor::id(rt, tensor.dtype(), [codomain]).unwrap();
    let gram = isometry.compose(&isometry.adjoint().unwrap()).unwrap();
    assert!(relative_distance(&gram, &identity) < 1e-10);
}

fn assert_polar_direction_error(result: Result<(Tensor, Tensor), Error>, operation: &'static str) {
    assert!(matches!(
        result,
        Err(Error::Operation(error))
            if matches!(
                error.as_ref(),
                OperationError::InvalidArgument { message }
                    if message.contains(operation)
                        && message.contains("coupled-sector")
            )
    ));
}

fn assert_hermitian_input_error<T>(result: Result<T, Error>) {
    assert!(matches!(
        result,
        Err(Error::Operation(error))
            if matches!(
                error.as_ref(),
                OperationError::InvalidArgument { message }
                    if *message == "eigh requires Hermitian coupled-sector blocks"
            )
    ));
}

fn assert_polar_direction_error_without_mutation(tensor: &Tensor, operation: &'static str) {
    let before_f64 = tensor.try_data().ok().map(<[f64]>::to_vec);
    let before_c64 = tensor.try_data_c64().ok().map(<[Complex64]>::to_vec);
    let result = match operation {
        "left_polar" => tensor.left_polar(),
        "right_polar" => tensor.right_polar(),
        _ => unreachable!("test only covers the two public polar directions"),
    };
    assert_polar_direction_error(result, operation);
    assert_eq!(tensor.try_data().ok(), before_f64.as_deref());
    assert_eq!(tensor.try_data_c64().ok(), before_c64.as_deref());
}

fn rectangular_polar_spaces() -> Vec<(Space, Space)> {
    vec![
        (Space::u1([(0, 2)]), Space::u1([(0, 3)])),
        (Space::su2([(1, 2)]).unwrap(), Space::su2([(1, 3)]).unwrap()),
        (Space::fz2([(1, 2)]).unwrap(), Space::fz2([(1, 3)]).unwrap()),
        (
            Space::fz2_u1_su2([((1, 1, 1), 2)]).unwrap(),
            Space::fz2_u1_su2([((1, 1, 1), 3)]).unwrap(),
        ),
    ]
}

fn u1_space() -> Space {
    Space::u1([(-1, 2), (0, 3), (1, 2)])
}

fn su2_space() -> Space {
    Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap()
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

/// Order-parity diagonal `contract` (#75): contracting a diagonal-storage `S`
/// against a tensor leg scales that leg and repartitions instead of densifying
/// to O(d²) + GEMM. Cross-checked against a hand-built dense diagonal driven
/// through the ordinary contraction — the only reference independent of the
/// scaling code. Covers sole-leg (edge) AND multi-leg geometries (a bond that is
/// not the operand's only leg on its side, incl. an interior leg — exactly what
/// a naive scale-in-place gets wrong).
#[test]
fn diagonal_contract_matches_dense_diagonal() {
    let rt = Runtime::builder().build().unwrap();
    // Includes a fermionic (FZ2) space: `contract` twists dual contracted legs,
    // so the fast path must fold that supertrace θ to match the dense diagonal.
    for v in [
        u1_space(),
        su2_space(),
        Space::fz2([(0, 2), (1, 2)]).unwrap(),
    ] {
        // A genuine diagonal S on the bond `v`, from an SVD.
        let src = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 105).unwrap();
        let (_, s, _) = src.svd_compact().unwrap();
        let bond = s.codomain_spaces()[0].clone();
        let spectrum = s.svd_vals().unwrap();

        // Dense `bond <- bond` diagonal with the same values (never Data::Diagonal).
        let s_dense = Tensor::from_block_fn(&rt, [&bond], [&bond], |key, idx| {
            let BlockKey::FusionTree(key) = key else {
                return 0.0;
            };
            if idx[0] != idx[1] {
                return 0.0;
            }
            let charge = key.codomain_uncoupled()[0];
            spectrum
                .iter()
                .find(|sp| sp.sector == charge)
                .map(|sp| sp.values[idx[0]])
                .unwrap_or(0.0)
        })
        .unwrap();
        let check = |fast: &Tensor, dense: &Tensor| assert_close(fast.data(), dense.data(), 1e-12);

        // lmul!: D * A on A's sole leading (codomain) leg. A = bond <- phys.
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&bond], [&v], 106).unwrap();
        check(
            &s.contract(&a, &[1], &[0]).unwrap(),
            &s_dense.contract(&a, &[1], &[0]).unwrap(),
        );

        // rmul!: B * D on B's sole trailing (domain) leg. B = phys <- bond.
        let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&bond], 107).unwrap();
        check(
            &b.contract(&s, &[1], &[0]).unwrap(),
            &b.contract(&s_dense, &[1], &[0]).unwrap(),
        );

        // Multi-leg D * A: bond is A's LEADING codomain leg but A has other legs.
        // A = [bond, phys; phys2]; the scaled leg must repartition to codomain 0.
        let g = Tensor::rand_with_seed(&rt, Dtype::F64, [&bond, &v], [&v], 108).unwrap();
        check(
            &s.contract(&g, &[1], &[0]).unwrap(),
            &s_dense.contract(&g, &[1], &[0]).unwrap(),
        );

        // Multi-leg D * A into an INTERIOR leg (axis 1, a codomain leg that is
        // neither first nor last). A = [phys, bond; phys2]. This is the case a
        // scale-in-place gets wrong (wrong codomain/domain split).
        let g_mid = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &bond], [&v], 109).unwrap();
        check(
            &s.contract(&g_mid, &[1], &[1]).unwrap(),
            &s_dense.contract(&g_mid, &[1], &[1]).unwrap(),
        );

        // Multi-leg A * D on a trailing domain leg. A = [phys; phys2, bond].
        let g_r = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v, &bond], 110).unwrap();
        check(
            &g_r.contract(&s, &[2], &[0]).unwrap(),
            &g_r.contract(&s_dense, &[2], &[0]).unwrap(),
        );

        // DUAL contracted rhs leg (fires the fermionic twist fold on FZ2): contract
        // against `s`'s DOMAIN leg (dual), not its codomain. A * D with rhs=s: A's
        // non-dual bond meets s's dual bond. `a` = [bond; phys].
        check(
            &a.contract(&s, &[0], &[1]).unwrap(),
            &a.contract(&s_dense, &[0], &[1]).unwrap(),
        );
        // D * A with rhs=B on B's dual domain bond. `b` = [phys; bond].
        check(
            &s.contract(&b, &[0], &[1]).unwrap(),
            &s_dense.contract(&b, &[0], &[1]).unwrap(),
        );
    }
}

/// PR1 of issue #55: `svd_compact` now stores `S` as an O(rank) per-sector
/// spectrum (`Data::Diagonal`) instead of a dense O(rank²) block-diagonal
/// buffer. The materialized dense buffer must still be exactly diagonal — its
/// nonzero count equals the number of singular values, so nothing bled off the
/// block diagonal — and reconstruction must hold via that lazy materialization.
#[test]
fn svd_compact_s_is_diagonal_storage() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 103).unwrap();
        let (u, s, vh) = t.svd_compact().unwrap();

        let num_values: usize = s.svd_vals().unwrap().iter().map(|sp| sp.values.len()).sum();
        let nonzero = s.data().iter().filter(|x| x.abs() > 1e-30).count();
        assert_eq!(
            nonzero, num_values,
            "materialized S is not diagonal: {nonzero} nonzeros for {num_values} singular values"
        );

        let recon = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert!(relative_distance(&recon, &t) < 1e-10);
    }
}

/// Issue #55: `inv`/`pinv`/`sqrt` on a diagonal-storage tensor stay elementwise
/// (O(rank)) and stay diagonal, instead of densifying to a per-block matrix
/// inverse/SVD/sqrt. Cross-checked against a hand-built dense diagonal driven
/// through the ordinary dense paths — the reference independent of the diagonal
/// code — and the diagonal result must itself still be diagonal storage.
#[test]
fn diagonal_matrix_functions_match_dense() {
    let rt = Runtime::builder().build().unwrap();
    for v in [
        u1_space(),
        su2_space(),
        Space::fz2([(0, 2), (1, 2)]).unwrap(),
    ] {
        let src = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 105).unwrap();
        let (_, s, _) = src.svd_compact().unwrap();
        let bond = s.codomain_spaces()[0].clone();
        let spectrum = s.svd_vals().unwrap();

        // Dense `bond <- bond` diagonal with the same values (never Data::Diagonal).
        let s_dense = Tensor::from_block_fn(&rt, [&bond], [&bond], |key, idx| {
            let BlockKey::FusionTree(key) = key else {
                return 0.0;
            };
            if idx[0] != idx[1] {
                return 0.0;
            }
            let charge = key.codomain_uncoupled()[0];
            spectrum
                .iter()
                .find(|sp| sp.sector == charge)
                .map(|sp| sp.values[idx[0]])
                .unwrap_or(0.0)
        })
        .unwrap();

        let num_values: usize = spectrum.iter().map(|sp| sp.values.len()).sum();
        let still_diagonal = |t: &Tensor| {
            assert_eq!(
                t.data().iter().filter(|x| x.abs() > 1e-30).count(),
                num_values,
                "result densified off the diagonal"
            );
        };

        // Singular values are strictly positive here, so inv == pinv == the true
        // reciprocal and sqrt is real — the diagonal path must equal the dense path.
        for (fast, dense) in [
            (s.inv().unwrap(), s_dense.inv().unwrap()),
            (s.pinv(1e-12).unwrap(), s_dense.pinv(1e-12).unwrap()),
            (s.sqrt().unwrap(), s_dense.sqrt().unwrap()),
        ] {
            assert_close(fast.data(), dense.data(), 1e-12);
            still_diagonal(&fast);
        }
    }
}

/// PR2 of issue #55: composing with the diagonal `S` folds to a block-local
/// bond scaling (TensorKit `DiagonalTensorMap` `rmul!`/`lmul!`) instead of a
/// GEMM. Both association orders must reconstruct `t` and agree, proving the
/// trailing-axis (`u * S`, `rmul!`) and leading-axis (`S * vh`, `lmul!`)
/// scalings are both correct — the leading path is untested by plain recompose.
#[test]
fn svd_compact_diagonal_scales_from_either_side() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 104).unwrap();
        let (u, s, vh) = t.svd_compact().unwrap();
        // rmul! path: (u * s) * vh.   lmul! path: u * (s * vh).
        let right = u.compose(&s).unwrap().compose(&vh).unwrap();
        let left = u.compose(&s.compose(&vh).unwrap()).unwrap();
        assert!(relative_distance(&right, &t) < 1e-10);
        assert!(relative_distance(&left, &t) < 1e-10);
        assert!(relative_distance(&left, &right) < 1e-12);
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
fn polar_rectangular_direction_contracts_hold_for_f64_and_c64() {
    // What: U1, SU2, fZ2, and their non-Abelian product enforce each public
    // polar direction and prove its corresponding one-sided identity.
    let rt = Runtime::builder().build().unwrap();
    for dtype in [Dtype::F64, Dtype::C64] {
        for (small, large) in rectangular_polar_spaces() {
            let tall = Tensor::rand_with_seed(&rt, dtype, [&large], [&small], 701).unwrap();
            assert_left_polar_contract(&rt, &tall, &small);
            assert_polar_direction_error_without_mutation(&tall, "right_polar");

            let wide = Tensor::rand_with_seed(&rt, dtype, [&small], [&large], 702).unwrap();
            assert_right_polar_contract(&rt, &wide, &small);
            assert_polar_direction_error_without_mutation(&wide, "left_polar");

            let square = Tensor::rand_with_seed(&rt, dtype, [&small], [&small], 703).unwrap();
            assert_left_polar_contract(&rt, &square, &small);
            assert_right_polar_contract(&rt, &square, &small);
        }
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
fn public_hermitian_spectral_entries_reject_nonhermitian_f64_and_c64() {
    // What: host full, values-only, truncated EIGH and spectral exp share one checked contract.
    let rt = Runtime::builder().build().unwrap();
    let v = Space::u1([(0, 2)]);
    let real = Tensor::from_block_fn(&rt, [&v], [&v], |_, indices| match indices {
        [0, 0] => 1.0,
        [0, 1] => 2.0,
        [1, 0] => 0.0,
        [1, 1] => 3.0,
        _ => unreachable!(),
    })
    .unwrap();
    let complex = Tensor::from_block_fn(&rt, [&v], [&v], |_, indices| match indices {
        [0, 0] => Complex64::new(1.0, 0.0),
        [0, 1] => Complex64::new(1.0, 2.0),
        [1, 0] => Complex64::new(3.0, 4.0),
        [1, 1] => Complex64::new(2.0, 0.0),
        _ => unreachable!(),
    })
    .unwrap();

    for tensor in [&real, &complex] {
        assert_hermitian_input_error(tensor.eigh_full());
        assert_hermitian_input_error(tensor.eigh_vals());
        assert_hermitian_input_error(tensor.eigh_trunc(&Truncation::Full));
        assert_hermitian_input_error(tensor.exp());
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

/// exp on a *non-trivial* spectrum: `exp(t) · exp(t) == exp(2t)` for a
/// Hermitian `t`. Guards the spectral-function diagonal fold (issue #46):
/// `exp_of_zero_acts_as_identity` only exercises the all-ones spectrum, where
/// every scaling factor is 1 and a mis-mapped column scaling would still pass.
#[test]
fn exp_composes_as_spectral_function() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 210).unwrap();
        // Hermitian with a modest spectrum so exp stays well-conditioned.
        let gram = a.adjoint().unwrap().compose(&a).unwrap();
        let t = gram.add(&gram, 0.5, 0.0).unwrap();
        let two_t = t.add(&t, 1.0, 1.0).unwrap();
        let lhs = t.exp().unwrap().compose(&t.exp().unwrap()).unwrap();
        let rhs = two_t.exp().unwrap();
        let err = relative_distance(&lhs, &rhs);
        assert!(err < 1e-9, "exp(t)^2 != exp(2t): relative error {err}");
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

#[test]
fn inv_accepts_tensorkit_isomorphic_exact_unequal_spaces() {
    // What: Abelian and fermionic non-Abelian fused/product isomorphisms are
    // invertible and return the exact swapped external spaces.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    assert_isomorphic_inverse(&rt, Dtype::F64, Space::u1([(0, 1), (1, 1)]));
    assert_isomorphic_inverse(
        &rt,
        Dtype::C64,
        Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 1, 1), 1)]).unwrap(),
    );
}

#[test]
fn pinv_rejects_invalid_rcond_for_dense_and_diagonal_inputs() {
    // What: public validation runs before both dense SVD and the diagonal shortcut.
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let dense = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 119).unwrap();
    let diagonal = dense.svd_compact().unwrap().1;
    for rcond in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(dense.pinv(rcond).is_err());
        assert!(diagonal.pinv(rcond).is_err());
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
    let input = tenet::matrixalgebra::BoundTensorMap::try_new(Arc::new(rule), typed).unwrap();
    let expert = tenet::matrixalgebra::svd_compact(&mut executor, &input.as_ref()).unwrap();

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
