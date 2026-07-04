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
    let leg = || SectorLeg::new(sectors.map(|sector| (sector, deg)), false);
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
            let p = c.permute(&cod, &dom).unwrap();
            let permuted_norm = p.norm().unwrap();
            assert!(
                (permuted_norm - norm).abs() <= 1e-10 * (1.0 + norm),
                "norm changed under permute {cod:?} | {dom:?}: {norm} -> {permuted_norm}"
            );
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
        let inner_cc = c.inner(&c).unwrap().re;
        assert!((inner_cc - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm));

        let scaled = c.scale(0.5).unwrap();
        assert!((scaled.norm().unwrap() - 0.5 * norm).abs() <= 1e-10 * (1.0 + norm));

        // w = c - d; |w|^2 = <c,c> - 2<c,d> + <d,d>.
        let w = c.add(&d, 1.0, -1.0).unwrap();
        let expected = inner_cc - 2.0 * c.inner(&d).unwrap().re + d.inner(&d).unwrap().re;
        let actual = w.inner(&w).unwrap().re;
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

/// Rank-5 PEPS-shaped tensors (1 codomain leg, 4 domain legs): construct,
/// contract over shared legs, permute, adjoint, norm — no rank ceiling.
#[test]
fn rank_five_peps_shape_u1_and_su2() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let a = Tensor::rand_with_seed(&rt, [&v], [&v, &v, &v, &v], 81).unwrap();
        assert_eq!(a.codomain_rank(), 1);
        assert_eq!(a.domain_rank(), 4);
        assert_eq!(a.rank(), 5);
        let norm = a.norm().unwrap();
        assert!(norm > 0.0);

        // Contract two rank-5 tensors over two shared legs (a's domain legs
        // against b's dual domain legs): rank-6 result.
        let w = v.dual();
        let b = Tensor::rand_with_seed(&rt, [&v], [&w, &w, &v, &v], 82).unwrap();
        let c = a.contract(&b, &[3, 4], &[1, 2]).unwrap();
        assert_eq!(c.codomain_rank(), 3);
        assert_eq!(c.domain_rank(), 3);
        assert!(c.norm().unwrap() > 0.0);

        // Permute across a (3, 2) split and back; weighted norm invariant.
        let p = a.permute(&[0, 2, 4], &[1, 3]).unwrap();
        assert_eq!(p.codomain_rank(), 3);
        let p_norm = p.norm().unwrap();
        assert!((p_norm - norm).abs() <= 1e-10 * (1.0 + norm));
        let back = p.permute(&[0], &[3, 1, 4, 2]).unwrap();
        assert_close(back.data(), a.data(), 1e-12);

        // Adjoint is an involution and swaps the split.
        let h = a.adjoint().unwrap();
        assert_eq!(h.codomain_rank(), 4);
        assert_eq!(h.domain_rank(), 1);
        let hh = h.adjoint().unwrap();
        assert_close(hh.data(), a.data(), 1e-12);
    }
}

/// Cross-checks a rank-5 x rank-5 contraction elementwise against a typed
/// expert-layer call with `NOUT = 1, NIN = 4`.
#[test]
fn rank_five_contract_cross_checks_against_expert_layer() {
    let rule = U1FusionRule;
    let deg = 2usize;
    let sectors = [
        U1Irrep::new(-1).sector_id(),
        U1Irrep::new(0).sector_id(),
        U1Irrep::new(1).sector_id(),
    ];
    let leg = || SectorLeg::new(sectors.map(|sector| (sector, deg)), false);
    // Dual leg of `Space::u1([(-1, deg), (0, deg), (1, deg)])`: the charge
    // set is symmetric, so only the dual flag flips.
    let dual_leg = || SectorLeg::new(sectors.map(|sector| (sector, deg)), true);
    let leg_dim = sectors.len() * deg;
    let lhs_homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg(), leg(), leg(), leg()]),
        )
    };
    let rhs_homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([dual_leg(), dual_leg(), leg(), leg()]),
        )
    };
    let space_1x4 = |hom: FusionTreeHomSpace| {
        let key_count = hom.fusion_tree_keys(&rule).len();
        let dense =
            TensorMapSpace::<1, 4>::from_dims([leg_dim], [leg_dim, leg_dim, leg_dim, leg_dim])
                .unwrap();
        FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            vec![vec![deg; 5]; key_count],
        )
        .unwrap()
    };

    let rt = Runtime::builder().build().unwrap();
    let v = Space::u1([(-1, deg), (0, deg), (1, deg)]);
    let w = v.dual();
    let a = Tensor::rand_with_seed(&rt, [&v], [&v, &v, &v, &v], 91).unwrap();
    let b = Tensor::rand_with_seed(&rt, [&v], [&w, &w, &v, &v], 92).unwrap();

    // Expert-layer twins share the flat storage (identical coupled layout).
    let lhs = TensorMap::<f64, 1, 4>::from_vec_with_fusion_space(
        a.data().to_vec(),
        space_1x4(lhs_homspace()),
    )
    .unwrap();
    let rhs = TensorMap::<f64, 1, 4>::from_vec_with_fusion_space(
        b.data().to_vec(),
        space_1x4(rhs_homspace()),
    )
    .unwrap();

    let lhs_axes = [3usize, 4];
    let rhs_axes = [1usize, 2];
    let output_axes: Vec<usize> = (0..6).collect();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs.fusion_space().unwrap().homspace(),
        rhs.fusion_space().unwrap().homspace(),
        &lhs_axes,
        &rhs_axes,
        &output_axes,
        3,
    )
    .unwrap();
    let key_count = dst_hom.fusion_tree_keys(&rule).len();
    let dense =
        TensorMapSpace::<3, 3>::from_dims([leg_dim, leg_dim, leg_dim], [leg_dim, leg_dim, leg_dim])
            .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        dense,
        dst_hom,
        &rule,
        vec![vec![deg; 6]; key_count],
    )
    .unwrap();
    let dst_len = dst_space.required_len().unwrap();
    let mut dst =
        TensorMap::<f64, 3, 3>::from_vec_with_fusion_space(vec![0.0; dst_len], dst_space).unwrap();
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

    let user = a.contract(&b, &lhs_axes, &rhs_axes).unwrap();
    assert_close(user.data(), dst.data(), 1e-12);
}

/// Composition on a U(1) charge set that is NOT closed under dualization
/// (`{0, 1}`, a hardcore boson). TensorKit v0.16 ground truth:
/// `A * B` for `A, B : V ← V` with `V = U1Space(0=>1, 1=>1)` is plain
/// block-by-block composition — with `block(A,0)=2, block(A,1)=3` and
/// `block(B,0)=5, block(B,1)=7` the product has blocks `10` and `21`.
#[test]
fn compose_works_on_non_dualization_closed_charge_sets() {
    let rt = Runtime::builder().build().unwrap();
    let v = Space::u1([(0, 1), (1, 1)]);
    let charge = |c: i32| U1Irrep::new(c).sector_id();

    let a = Tensor::from_block_fn(&rt, [&v], [&v], |key, _| match key {
        BlockKey::FusionTree(key) if key.codomain_uncoupled()[0] == charge(0) => 2.0,
        _ => 3.0,
    })
    .unwrap();
    let b = Tensor::from_block_fn(&rt, [&v], [&v], |key, _| match key {
        BlockKey::FusionTree(key) if key.codomain_uncoupled()[0] == charge(0) => 5.0,
        _ => 7.0,
    })
    .unwrap();
    assert_eq!(a.compose(&b).unwrap().data(), &[10.0, 21.0]);

    // The original repro: a random endomorphism composes with itself.
    let r = Tensor::rand(&rt, [&v], [&v]).unwrap();
    assert!(r.compose(&r).is_ok());
}

/// Pairing follows Space identity (TensorKit: `domain(A) == codomain(B)`),
/// independent of dualization closure of the sector content.
#[test]
fn leg_pairing_rules_on_asymmetric_charges() {
    let rt = Runtime::builder().build().unwrap();
    let v = Space::u1([(0, 1), (1, 1)]);
    let a = Tensor::rand(&rt, [&v], [&v]).unwrap();

    // Domain V does NOT pair with codomain V' (TensorKit SpaceMismatch).
    let bad = Tensor::rand(&rt, [&v.dual()], [&v]).unwrap();
    assert!(a.compose(&bad).is_err());

    // Domain-vs-domain legs contract when exactly one side is the dual.
    let b = Tensor::rand(&rt, [&v], [&v.dual()]).unwrap();
    assert!(a.contract(&b, &[1], &[1]).is_ok());

    // ...and are rejected when both sides carry the same Space.
    let c = Tensor::rand(&rt, [&v], [&v]).unwrap();
    assert!(a.contract(&c, &[1], &[1]).is_err());
}

// ---------------------------------------------------------------------------
// fZ2 ⊠ U(1) ⊠ SU(2) triple product (left-associated, TensorKit `Vect[
// FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂]]`). Hardcoded reference values from
// TensorKit (Julia, `benchmarks` env), see comments at each assertion.
// ---------------------------------------------------------------------------

use tenet::core::{
    FermionParityFusionRule, FusionRule, MultiplicityFreeRigidSymbols, ProductFusionRule,
    ProductSectorCodec, SU2FusionRule, SU2Irrep, TensorKitProductCodec,
};

type TripleRule =
    ProductFusionRule<ProductFusionRule<FermionParityFusionRule, U1FusionRule>, SU2FusionRule>;

fn triple_rule() -> TripleRule {
    ProductFusionRule::new(
        ProductFusionRule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    )
}

/// Packed sector id of `(parity ⊠ charge ⊠ twice_spin)`, left-associated.
fn triple_sector(parity: u8, charge: i32, twice_spin: usize) -> SectorId {
    let inner = TensorKitProductCodec::encode(
        SectorId::new(usize::from(parity & 1)),
        U1Irrep::new(charge).sector_id(),
    );
    TensorKitProductCodec::encode(inner, SU2Irrep::from_twice_spin(twice_spin).sector_id())
}

/// The Julia reference space:
/// `S = Vect[FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂]]((0,0,0)=>1, (1,1,1//2)=>1, (0,2,0)=>1)`
fn triple_space() -> Space {
    Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 1, 1), 1), ((0, 2, 0), 1)]).unwrap()
}

#[test]
fn fz2_u1_su2_codec_roundtrip_and_dual() {
    // Encode/decode round-trip of the pairwise TensorKit codec for triples.
    for &(parity, charge, twice_spin) in &[
        (0u8, 0i32, 0usize),
        (1, 1, 1),
        (0, 2, 0),
        (1, -1, 1),
        (0, -2, 4),
        (1, 5, 3),
    ] {
        let sector = triple_sector(parity, charge, twice_spin);
        let (inner, su2) = TensorKitProductCodec::decode(sector).unwrap();
        let (fz2, u1) = TensorKitProductCodec::decode(inner).unwrap();
        assert_eq!(fz2, SectorId::new(usize::from(parity)));
        assert_eq!(u1, U1Irrep::new(charge).sector_id());
        assert_eq!(su2, SU2Irrep::from_twice_spin(twice_spin).sector_id());
    }

    // Dual: parity self-dual, charge negates, spin self-dual. Julia:
    //   sectors(S') = {(0,0,0), (1,-1,1/2), (0,-2,0)}
    let rule = triple_rule();
    assert_eq!(rule.dual(triple_sector(1, 1, 1)), triple_sector(1, -1, 1));
    assert_eq!(rule.dual(triple_sector(0, 2, 0)), triple_sector(0, -2, 0));
    assert_eq!(rule.dual(triple_sector(0, 0, 0)), triple_sector(0, 0, 0));
    assert_eq!(rule.vacuum(), triple_sector(0, 0, 0));
}

#[test]
fn fz2_u1_su2_space_and_identity_invariants_vs_tensorkit() {
    let rt = Runtime::builder().build().unwrap();
    let s = triple_space();

    // Julia: dim(S) = 4, dim(S') = 4, dim(S ⊗ S) = 16.
    assert_eq!(s.dim(), 4);
    assert_eq!(s.dual().dim(), 4);
    assert_eq!(s.dual().dual(), s);

    // Identity on S ⊗ S, built block-by-block: a fusion-tree pair block of
    // `id` is the degeneracy identity when the codomain and domain trees
    // coincide (two uncoupled legs, multiplicity-free: the uncoupled pair
    // plus the shared coupled sector determine the tree).
    let blocks = std::cell::Cell::new(0usize);
    let id = Tensor::from_block_fn(&rt, [&s, &s], [&s, &s], |key, indices| {
        let BlockKey::FusionTree(key) = key else {
            return 0.0;
        };
        blocks.set(blocks.get() + 1);
        let diag = key.codomain_uncoupled() == key.domain_uncoupled()
            && indices[0] == indices[2]
            && indices[1] == indices[3];
        if diag {
            1.0
        } else {
            0.0
        }
    })
    .unwrap();

    // Julia: length(collect(fusiontrees(id(S⊗S)))) = 20 tree pairs; every
    // degeneracy is 1, so the fill runs exactly once per pair block.
    assert_eq!(blocks.get(), 20);

    // Julia: norm(id(S⊗S)) = 4.0 (= sqrt(Σ_c qdim_c · blockdim_c), i.e. the
    // quantum-dimension-weighted Frobenius norm).
    let norm = id.norm().unwrap();
    assert!((norm - 4.0).abs() <= 1e-12, "norm(id) = {norm}");

    // inner(id, id) = ‖id‖² = tr(id† id) = 16.0; Julia: tr(id(S⊗S)) = 16.0.
    let inner = id.inner(&id).unwrap().re;
    assert!((inner - 16.0).abs() <= 1e-12, "inner(id, id) = {inner}");

    // Julia blocksectors of id(S⊗S): 6 coupled sectors with block dims
    //   (0,0,0)=>1 (1,1,1/2)=>2 (0,2,0)=>3 (0,2,1)=>1 (1,3,1/2)=>2 (0,4,0)=>1
    let spectra = id.svd_vals().unwrap();
    assert_eq!(spectra.len(), 6);
    let mut dims: Vec<(SectorId, usize)> = spectra
        .iter()
        .map(|entry| {
            for value in &entry.values {
                assert!((value - 1.0).abs() <= 1e-12);
            }
            (entry.sector, entry.values.len())
        })
        .collect();
    dims.sort();
    let mut expected = vec![
        (triple_sector(0, 0, 0), 1),
        (triple_sector(1, 1, 1), 2),
        (triple_sector(0, 2, 0), 3),
        (triple_sector(0, 2, 2), 1),
        (triple_sector(1, 3, 1), 2),
        (triple_sector(0, 4, 0), 1),
    ];
    expected.sort();
    assert_eq!(dims, expected);
}

#[test]
fn fz2_u1_su2_braid_fermion_sign_vs_tensorkit() {
    let rt = Runtime::builder().build().unwrap();
    let rule = triple_rule();

    // Julia:
    //   odd = Vect[I3]((1,1,1//2) => 1); W = Vect[I3]((0,2,0) => 1, (0,2,1) => 1)
    //   t = ones(Float64, odd ⊗ odd ← W)
    //   tb = braid(t, ((2,1),(3,)), (2,1,3))
    //   block (0,2,0): 1.0 -> 1.0   (fermion −1 × SU2 spin-0 R −1)
    //   block (0,2,1): 1.0 -> −1.0  (fermion −1 × SU2 spin-1 R +1)
    //   braid(tb, ((2,1),(3,)), (1,2,3)) == t (roundtrip)
    let odd = Space::fz2_u1_su2([((1, 1, 1), 1)]).unwrap();
    let w = Space::fz2_u1_su2([((0, 2, 0), 1), ((0, 2, 2), 1)]).unwrap();
    let t = Tensor::from_block_fn(&rt, [&odd, &odd], [&w], |_, _| 1.0).unwrap();
    let tb = t.braid(&[1, 0], &[2], &[2, 1, 3]).unwrap();
    let tbb = tb.braid(&[1, 0], &[2], &[1, 2, 3]).unwrap();
    assert_close(tbb.data(), t.data(), 1e-12);

    // Project the braided tensor onto each coupled sector: inner() weights
    // by the quantum dimension, so the reference tensors see qdim(0,2,0)=1
    // and qdim(0,2,1)=3.
    for (sector, qdim, sign) in [
        (triple_sector(0, 2, 0), 1.0, 1.0),
        (triple_sector(0, 2, 2), 3.0, -1.0),
    ] {
        assert_eq!(rule.dim_scalar(sector), qdim);
        let proj = Tensor::from_block_fn(&rt, [&odd, &odd], [&w], |key, _| match key {
            BlockKey::FusionTree(key) if key.coupled() == Some(sector) => 1.0,
            _ => 0.0,
        })
        .unwrap();
        let before = proj.inner(&t).unwrap().re;
        let after = proj.inner(&tb).unwrap().re;
        assert!((before - qdim).abs() <= 1e-12, "before = {before}");
        assert!(
            (after - sign * qdim).abs() <= 1e-12,
            "sector {sector:?}: after = {after}, expected {}",
            sign * qdim
        );
    }
}

#[test]
fn fz2_u1_su2_contraction_svd_and_rank5_smoke() {
    let rt = Runtime::builder().build().unwrap();
    let v = triple_space();

    // Rank-4 contraction + permute roundtrip.
    let a = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 91).unwrap();
    let b = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 92).unwrap();
    let c = a.compose(&b).unwrap();
    assert!(c.norm().unwrap() > 0.0);
    let back = c
        .permute(&[0, 2], &[1, 3])
        .unwrap()
        .permute(&[0, 2], &[1, 3])
        .unwrap();
    assert_close(back.data(), c.data(), 1e-12);

    // Braid with levels and undo with swapped levels (rand tensor).
    let braided = a.braid(&[1, 0], &[2, 3], &[2, 1, 3, 4]).unwrap();
    let unbraided = braided.braid(&[1, 0], &[2, 3], &[1, 2, 3, 4]).unwrap();
    assert_close(unbraided.data(), a.data(), 1e-12);

    // svd_compact reconstruction + svd_trunc self-consistency: the error
    // reported for a truncation equals the weighted norm distance between
    // the full tensor and the truncated reconstruction.
    let (u, s, vh) = a.svd_compact().unwrap();
    let recon = u.compose(&s).unwrap().compose(&vh).unwrap();
    let diff = recon.add(&a, 1.0, -1.0).unwrap();
    assert!(diff.norm().unwrap() <= 1e-10 * (1.0 + a.norm().unwrap()));

    let trunc = a.svd_trunc(&Truncation::rank(6)).unwrap();
    let recon_t = trunc
        .u
        .compose(&trunc.s)
        .unwrap()
        .compose(&trunc.vh)
        .unwrap();
    let err = recon_t.add(&a, 1.0, -1.0).unwrap().norm().unwrap();
    assert!(
        (err - trunc.error).abs() <= 1e-10 * (1.0 + trunc.error),
        "reconstruction error {err} vs reported {}",
        trunc.error
    );

    // Rank-5 (1|4) PEPS-shaped smoke: construct, contract two shared legs,
    // permute roundtrip, adjoint involution.
    let w = v.dual();
    let p = Tensor::rand_with_seed(&rt, [&v], [&v, &v, &v, &v], 93).unwrap();
    let q = Tensor::rand_with_seed(&rt, [&v], [&w, &w, &v, &v], 94).unwrap();
    let r = p.contract(&q, &[3, 4], &[1, 2]).unwrap();
    assert_eq!(r.rank(), 6);
    assert!(r.norm().unwrap() > 0.0);
    let rp = p.permute(&[0, 2, 4], &[1, 3]).unwrap();
    let p_back = rp.permute(&[0], &[3, 1, 4, 2]).unwrap();
    assert_close(p_back.data(), p.data(), 1e-12);
    let h = p.adjoint().unwrap().adjoint().unwrap();
    assert_close(h.data(), p.data(), 1e-12);
}
