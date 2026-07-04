//! Integration tests for the user-layer Space / Runtime / Tensor API,
//! including elementwise cross-checks against the expert layer.

use tenet::core::{
    FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, SectorLeg, TensorMap,
    TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet::operations::{
    OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
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

fn u1_space() -> Space {
    Space::u1([(-1, 2), (0, 3), (1, 2)])
}

fn su2_space() -> Space {
    Space::su2([(0, 2), (1, 2), (2, 1)])
}

#[test]
fn rand_and_zeros_construction_u1_and_su2() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let zero = Tensor::zeros(&rt, [&v, &v], [&v, &v]).unwrap();
        assert_eq!(zero.norm().unwrap(), 0.0);
        assert_eq!(zero.codomain_rank(), 2);
        assert_eq!(zero.domain_rank(), 2);

        let a = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 7).unwrap();
        let b = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 7).unwrap();
        assert_eq!(a.data(), b.data(), "same seed must reproduce the data");
        assert_eq!(a.data().len(), zero.data().len());
        assert!(a.norm().unwrap() > 0.0);

        // The runtime's own stream advances between calls.
        let c = Tensor::rand(&rt, [&v, &v], [&v, &v]).unwrap();
        let d = Tensor::rand(&rt, [&v, &v], [&v, &v]).unwrap();
        assert_ne!(c.data(), d.data());
    }
}

#[test]
fn space_dual_roundtrip_and_dim() {
    let v = u1_space();
    assert_eq!(v.dim(), 7);
    assert_eq!(v.dual().dual(), v);
    // SU2 dims are quantum-dimension weighted: 2*1 + 2*2 + 1*3.
    assert_eq!(su2_space().dim(), 9);
}

#[test]
fn compose_equals_contract_on_matching_axes() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let a = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 1).unwrap();
    let b = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 2).unwrap();

    let composed = a.compose(&b).unwrap();
    let contracted = a.contract(&b, &[2, 3], &[0, 1]).unwrap();
    assert_eq!(composed.data(), contracted.data());
    assert_eq!(composed.codomain_rank(), 2);
    assert_eq!(composed.domain_rank(), 2);
    assert!(composed.norm().unwrap() > 0.0);
}

#[test]
fn contract_ordered_matches_permuted_default_order() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let a = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 3).unwrap();
    let b = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 4).unwrap();

    let default_order = a.contract(&b, &[3, 2], &[0, 1]).unwrap();
    let reordered = a
        .contract_ordered(&b, &[3, 2], &[0, 1], &[1, 0, 2, 3])
        .unwrap();
    let expected = default_order.permute(&[1, 0], &[2, 3]).unwrap();
    assert_close(reordered.data(), expected.data(), 1e-12);

    // Identity output order is exactly the default order.
    let identity = a
        .contract_ordered(&b, &[3, 2], &[0, 1], &[0, 1, 2, 3])
        .unwrap();
    assert_eq!(identity.data(), default_order.data());
}

/// Builds the expert-layer analog of `Space::u1([(-1, 2), (0, 2), (1, 2)])`
/// legs with uniform degeneracy 2 and cross-checks the user-layer
/// contraction elementwise against `tensorcontract_fusion_into`.
#[test]
fn contract_cross_checks_against_expert_layer() {
    let rule = U1FusionRule;
    let deg = 2usize;
    let sectors = [
        U1Irrep::new(-1).sector_id(),
        U1Irrep::new(0).sector_id(),
        U1Irrep::new(1).sector_id(),
    ];
    let leg = || SectorLeg::new(sectors, false);
    let leg_dim = sectors.len() * deg;
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space = |hom: FusionTreeHomSpace| {
        let key_count = hom.fusion_tree_keys(&rule).len();
        let dense =
            TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap();
        FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            vec![vec![deg; 4]; key_count],
        )
        .unwrap()
    };

    // User-layer tensors.
    let rt = Runtime::builder().build().unwrap();
    let v = Space::u1([(-1, deg), (0, deg), (1, deg)]);
    let a = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 11).unwrap();
    let b = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 12).unwrap();

    // Expert-layer twins share the flat storage (identical coupled layout).
    let lhs =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(a.data().to_vec(), space(homspace()))
            .unwrap();
    let rhs =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(b.data().to_vec(), space(homspace()))
            .unwrap();

    for (lhs_axes, rhs_axes, output_axes) in [
        ([2, 3], [0, 1], [0, 1, 2, 3]),
        ([3, 2], [0, 1], [0, 1, 2, 3]),
        ([3, 2], [0, 1], [1, 0, 2, 3]),
    ] {
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            &rule,
            lhs.fusion_space().unwrap().homspace(),
            rhs.fusion_space().unwrap().homspace(),
            &lhs_axes,
            &rhs_axes,
            &output_axes,
            2,
        )
        .unwrap();
        let dst_space = space(dst_hom);
        let dst_len = dst_space.required_len().unwrap();
        let mut dst =
            TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; dst_len], dst_space)
                .unwrap();
        let mut context = TensorContractFusionExecutionContext::<f64, _>::default();
        context
            .tensorcontract_fusion_into(
                &rule,
                &mut dst,
                &lhs,
                &rhs,
                TensorContractSpec::new(
                    &lhs_axes,
                    &rhs_axes,
                    OutputAxisOrder::from_axes(&output_axes),
                ),
                1.0,
                0.0,
            )
            .unwrap();

        let user = a
            .contract_ordered(&b, &lhs_axes, &rhs_axes, &output_axes)
            .unwrap();
        assert_close(user.data(), dst.data(), 1e-12);
    }
}

#[test]
fn permute_roundtrip_restores_the_tensor() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let c = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 21).unwrap();
        // [0, 2 | 1, 3] is an involution on axis positions.
        let p = c.permute(&[0, 2], &[1, 3]).unwrap();
        let back = p.permute(&[0, 2], &[1, 3]).unwrap();
        assert_close(back.data(), c.data(), 1e-12);
    }
}

#[test]
fn permute_preserves_the_weighted_norm() {
    let rt = Runtime::builder().build().unwrap();
    // SU2 exercises the quantum-dimension weighting: raw unweighted data
    // norms are *not* preserved when legs bend between codomain and domain.
    for v in [u1_space(), su2_space()] {
        let c = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 22).unwrap();
        let norm = c.norm().unwrap();
        for (cod, dom) in [
            (vec![0, 2], vec![1, 3]),
            (vec![0, 3], vec![1, 2]),
            (vec![0, 1, 2], vec![3]),
        ] {
            match c.permute(&cod, &dom) {
                Ok(p) => {
                    let permuted_norm = p.norm().unwrap();
                    assert!(
                        (permuted_norm - norm).abs() <= 1e-10 * (1.0 + norm),
                        "norm changed under permute {cod:?} | {dom:?}: {norm} -> {permuted_norm}"
                    );
                }
                // (3, 1) splits exceed the current 2-legs-per-side ceiling.
                Err(Error::UnsupportedRank { .. }) => {}
                Err(err) => panic!("unexpected permute error: {err}"),
            }
        }
    }
}

#[test]
fn transpose_and_adjoint_involutions() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let c = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 31).unwrap();

        let h = c.adjoint().unwrap();
        assert_eq!(h.codomain_rank(), 2);
        let hh = h.adjoint().unwrap();
        assert_close(hh.data(), c.data(), 1e-12);

        let t = c.transpose().unwrap();
        let tt = t.transpose().unwrap();
        assert_close(tt.data(), c.data(), 1e-12);
    }
}

#[test]
fn braid_with_trivial_levels_matches_permute_for_bosonic_rules() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let c = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 41).unwrap();
    let p = c.permute(&[1, 0], &[2, 3]).unwrap();
    let b = c.braid(&[1, 0], &[2, 3], &[0, 1, 2, 3]).unwrap();
    assert_close(b.data(), p.data(), 1e-12);
}

#[test]
fn vector_interface_identities() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let c = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 51).unwrap();
        let d = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 52).unwrap();

        let norm = c.norm().unwrap();
        let inner_cc = c.inner(&c).unwrap();
        assert!((inner_cc - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm));

        let scaled = c.scale(0.5).unwrap();
        assert!((scaled.norm().unwrap() - 0.5 * norm).abs() <= 1e-10 * (1.0 + norm));

        // w = c - d; |w|^2 = <c,c> - 2<c,d> + <d,d>.
        let w = c.add(&d, 1.0, -1.0).unwrap();
        let expected = inner_cc - 2.0 * c.inner(&d).unwrap() + d.inner(&d).unwrap();
        let actual = w.inner(&w).unwrap();
        assert!((actual - expected).abs() <= 1e-10 * (1.0 + expected.abs()));
    }
}

#[test]
fn fz2_and_product_rule_smoke() {
    let rt = Runtime::builder().build().unwrap();

    let f = Space::fz2([(0, 2), (1, 2)]);
    let a = Tensor::rand_with_seed(&rt, [&f, &f], [&f, &f], 61).unwrap();
    let b = Tensor::rand_with_seed(&rt, [&f, &f], [&f, &f], 62).unwrap();
    assert!(a.compose(&b).unwrap().norm().unwrap() > 0.0);

    let p = Space::product([((-1, 1), 2), ((0, 0), 2), ((1, 1), 2)]).unwrap();
    let a = Tensor::rand_with_seed(&rt, [&p, &p], [&p, &p], 63).unwrap();
    let b = Tensor::rand_with_seed(&rt, [&p, &p], [&p, &p], 64).unwrap();
    let c = a.compose(&b).unwrap();
    assert!(c.norm().unwrap() > 0.0);
    let back = c
        .permute(&[0, 2], &[1, 3])
        .unwrap()
        .permute(&[0, 2], &[1, 3])
        .unwrap();
    assert_close(back.data(), c.data(), 1e-12);
}

#[test]
fn mixing_rules_or_runtimes_is_rejected() {
    let rt = Runtime::builder().build().unwrap();
    let u = u1_space();
    let z = Space::z2([(0, 1), (1, 1)]);

    // Mixed rules inside one construction.
    assert!(matches!(
        Tensor::rand(&rt, [&u], [&z]),
        Err(Error::RuleMismatch)
    ));

    // Mixed rules across an operation.
    let a = Tensor::rand_with_seed(&rt, [&u], [&u], 71).unwrap();
    let b = Tensor::rand_with_seed(&rt, [&z], [&z], 72).unwrap();
    assert!(matches!(a.compose(&b), Err(Error::RuleMismatch)));
    assert!(matches!(a.add(&b, 1.0, 1.0), Err(Error::RuleMismatch)));
    assert!(matches!(a.inner(&b), Err(Error::RuleMismatch)));

    // Same rule, different runtimes.
    let rt2 = Runtime::builder().build().unwrap();
    let c = Tensor::rand_with_seed(&rt2, [&u], [&u], 73).unwrap();
    assert!(matches!(a.compose(&c), Err(Error::RuntimeMismatch)));

    // Same rule and runtime, different spaces.
    let w = Space::u1([(0, 1), (1, 1)]);
    let d = Tensor::rand_with_seed(&rt, [&w], [&w], 74).unwrap();
    assert!(matches!(
        a.add(&d, 1.0, 1.0),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn rank_ceiling_is_reported() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    assert!(matches!(
        Tensor::rand(&rt, [&v, &v, &v], [&v]),
        Err(Error::UnsupportedRank { nout: 3, nin: 1 })
    ));
}
