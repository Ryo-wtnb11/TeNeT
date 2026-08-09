//! Public, by-value conformance checks for the built-in multiplicity-free providers.
//!
//! This deliberately covers ordinary structural operations only.  Factorizations
//! and matrix functions have their own #1002 gates.

use std::sync::Arc;

use tenet::prelude::{
    product_sector, CU1FusionRule, CU1Irrep, Complex64, FermionParityFusionRule, GradedSpace,
    ProductFusionRuleExt, Runtime, SU2FusionRule, SU2Irrep, TensorMap, U1FusionRule, U1Irrep,
    Z2FusionRule, Z2Irrep, ZNFusionRule,
};

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

#[test]
fn zn3_index_flip_and_units_keep_the_original_provider() {
    let runtime = runtime();
    let provider = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge = |value| provider.irrep(value);
    let leg = GradedSpace::try_new_shared(
        Arc::clone(&provider),
        [(charge(0), 1), (charge(1), 1), (charge(2), 1)],
    )
    .unwrap();
    let source: TensorMap<ZNFusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |sectors, _| {
            10.0 + sectors.coupled().charge() as f64
        })
        .unwrap();

    // Independent label/value oracle: the three diagonal charge blocks are
    // ordered by their Z3 charge, not merely restored by an inverse operation.
    assert_eq!(source.data(), &[10.0, 11.0, 12.0]);
    assert_eq!(
        (0..source.block_count())
            .map(|i| *source.block_fusion_trees(i).unwrap().coupled())
            .collect::<Vec<_>>(),
        vec![charge(0), charge(1), charge(2)]
    );

    let flipped = source.flip(&[0]).unwrap();
    assert!(std::ptr::eq(flipped.provider(), provider.as_ref()));
    assert!(std::ptr::eq(
        flipped.codomain()[0].provider(),
        provider.as_ref()
    ));
    assert!(std::ptr::eq(
        flipped.domain()[0].provider(),
        provider.as_ref()
    ));
    assert!(flipped.codomain()[0].is_dual());
    assert_eq!(flipped.data(), source.data());
    assert_eq!(flipped.flip_inverse(&[0]).unwrap().data(), source.data());

    let inserted = source.insert_left_unit(1, true).unwrap();
    assert!(std::ptr::eq(inserted.provider(), provider.as_ref()));
    assert!(std::ptr::eq(
        inserted.domain()[0].provider(),
        provider.as_ref()
    ));
    assert_eq!(inserted.domain()[0].sectors().unwrap(), vec![charge(0)]);
    let restored = inserted.remove_unit(1).unwrap();
    assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
    assert_eq!(restored.data(), source.data());
}

#[test]
fn z2_cat_and_absorb_have_hand_computed_slabs() {
    let runtime = runtime();
    let provider = Arc::new(Z2FusionRule);
    let codomain =
        GradedSpace::try_new_shared(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)]).unwrap();
    let lhs_domain =
        GradedSpace::try_new_shared(Arc::clone(&provider), [(Z2Irrep::EVEN, 1)]).unwrap();
    let rhs_domain =
        GradedSpace::try_new_shared(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)]).unwrap();
    let lhs: TensorMap<Z2FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&codomain], [&lhs_domain], |_, i| {
            (i[0] + 1) as f64
        })
        .unwrap();
    let rhs: TensorMap<Z2FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&codomain], [&rhs_domain], |_, i| {
            (3 + i[0] + 2 * i[1]) as f64
        })
        .unwrap();
    let joined = lhs.catdomain(&rhs).unwrap();
    assert!(std::ptr::eq(joined.provider(), provider.as_ref()));
    assert!(std::ptr::eq(
        joined.domain()[0].provider(),
        provider.as_ref()
    ));
    assert_eq!(joined.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(joined.domain()[0].degeneracies(), &[3]);

    let destination: TensorMap<Z2FusionRule, f64> = TensorMap::from_block_fn(
        &runtime,
        [&GradedSpace::try_new_shared(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)]).unwrap()],
        [&GradedSpace::try_new_shared(Arc::clone(&provider), [(Z2Irrep::EVEN, 3)]).unwrap()],
        |_, i| (10 * (i[0] + 1) + i[1] + 1) as f64,
    )
    .unwrap();
    let source: TensorMap<Z2FusionRule, f64> = TensorMap::from_block_fn(
        &runtime,
        [&GradedSpace::try_new_shared(Arc::clone(&provider), [(Z2Irrep::EVEN, 3)]).unwrap()],
        [&GradedSpace::try_new_shared(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)]).unwrap()],
        |_, i| -((10 * (i[0] + 1) + i[1] + 1) as f64),
    )
    .unwrap();
    let absorbed = destination.absorb(&source).unwrap();
    assert!(std::ptr::eq(absorbed.provider(), provider.as_ref()));
    assert_eq!(absorbed.data(), &[-11.0, -21.0, -12.0, -22.0, 13.0, 23.0]);
}

#[test]
fn fermionic_product_contract_otimes_and_reductions_keep_provider_and_signs() {
    let runtime = runtime();
    let provider = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let odd = product_sector(Z2Irrep::ODD, U1Irrep::new(1));
    let leg = GradedSpace::try_new_shared(Arc::clone(&provider), [(odd, 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 3.0).unwrap();

    let contracted = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    assert!(std::ptr::eq(contracted.provider(), provider.as_ref()));
    assert!(std::ptr::eq(
        contracted.codomain()[0].provider(),
        provider.as_ref()
    ));
    assert_eq!(contracted.data(), &[6.0]);
    let product = lhs.otimes(&rhs).unwrap();
    assert!(std::ptr::eq(product.provider(), provider.as_ref()));
    assert_eq!(product.data(), &[6.0]);
    assert_eq!(lhs.inner(&rhs).unwrap(), 6.0);
    assert_eq!(lhs.norm().unwrap(), 2.0);
    assert_eq!(lhs.tr().unwrap(), 2.0);
}

#[test]
fn nested_fermionic_su2_product_and_complex_adjoint_are_publicly_conformant() {
    let runtime = runtime();
    let provider = Arc::new(
        FermionParityFusionRule
            .product(U1FusionRule)
            .product(SU2FusionRule),
    );
    let spin_half = product_sector(
        product_sector(Z2Irrep::ODD, U1Irrep::new(1)),
        SU2Irrep::from_twice_spin(1),
    );
    let leg = GradedSpace::try_new_shared(Arc::clone(&provider), [(spin_half, 1)]).unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| Complex64::new(2.0, 3.0))
            .unwrap();
    let adjoint = source.adjoint().unwrap();
    assert!(std::ptr::eq(adjoint.provider(), provider.as_ref()));
    assert!(std::ptr::eq(
        adjoint.codomain()[0].provider(),
        provider.as_ref()
    ));
    assert!(std::ptr::eq(
        adjoint.domain()[0].provider(),
        provider.as_ref()
    ));
    assert_eq!(source.twist(&[0, 1]).unwrap().data(), source.data());
    assert_eq!(adjoint.data(), &[Complex64::new(2.0, -3.0)]);
    assert_eq!(
        adjoint.codomain()[0].sectors().unwrap(),
        vec![spin_half],
        "the nonzero SU(2) label survives the public product route"
    );
}

#[test]
fn zn3_extended_structural_paths_execute_on_the_original_arc() {
    let runtime = runtime();
    let provider = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge = |value| provider.irrep(value);
    let leg = GradedSpace::try_new_shared(
        Arc::clone(&provider),
        [(charge(0), 1), (charge(1), 2), (charge(2), 1)],
    )
    .unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, index| {
            Complex64::new(
                100.0 * trees.coupled().charge() as f64 + 10.0 * index[0] as f64 + index[1] as f64,
                index[2] as f64,
            )
        })
        .unwrap();
    let permuted = source.permute(&[1, 0], &[2]).unwrap();
    let braided = source.braid(&[1, 0], &[2], &[1, 0, 2]).unwrap();
    assert_eq!(braided.data(), permuted.data(), "Z3 is bosonic");
    assert_ne!(
        permuted.data(),
        source.data(),
        "asymmetric raw payload moved"
    );
    let restored = permuted.permute(&[1, 0], &[2]).unwrap();
    assert_eq!(restored.data(), source.data());
    for index in 0..source.block_count() {
        assert_eq!(
            restored.block_fusion_trees(index).unwrap(),
            source.block_fusion_trees(index).unwrap()
        );
    }
    let twisted = source.twist(&[0, 1, 2]).unwrap();
    assert_eq!(twisted.data(), source.data(), "ZN(3) has trivial twist");
    assert_eq!(twisted.data().as_ptr(), source.data().as_ptr());
    for output in [
        source.adjoint().unwrap(),
        source.transpose().unwrap(),
        source.repartition(1).unwrap(),
        braided,
        source.twist(&[0, 1, 2]).unwrap(),
    ] {
        assert!(std::ptr::eq(output.provider(), provider.as_ref()));
    }
    let adjoint = source.adjoint().unwrap();
    assert!(std::ptr::eq(
        adjoint.codomain()[0].provider(),
        provider.as_ref()
    ));
    assert!(std::ptr::eq(
        adjoint.domain()[0].provider(),
        provider.as_ref()
    ));
}

#[test]
fn cu1_charged_structural_paths_keep_the_original_arc() {
    let runtime = runtime();
    let provider = Arc::new(CU1FusionRule);
    let charged = CU1Irrep::from_twice_charge(1);
    let leg = GradedSpace::try_new_shared(Arc::clone(&provider), [(charged, 1)]).unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| Complex64::new(2.0, 3.0))
            .unwrap();
    for output in [
        source.adjoint().unwrap(),
        source.twist(&[0, 1]).unwrap(),
        source.flip(&[0]).unwrap(),
        source.insert_right_unit(0, false).unwrap(),
    ] {
        assert!(std::ptr::eq(output.provider(), provider.as_ref()));
        assert!(std::ptr::eq(
            output.codomain()[0].provider(),
            provider.as_ref()
        ));
    }
    assert_eq!(
        source.adjoint().unwrap().data(),
        &[Complex64::new(2.0, -3.0)]
    );
    assert_eq!(source.twist(&[0, 1]).unwrap().data(), source.data());
    assert_eq!(source.flip(&[0]).unwrap().data(), source.data());
    let pseudo =
        GradedSpace::try_new_shared(Arc::clone(&provider), [(CU1Irrep::PSEUDOSCALAR, 1)]).unwrap();
    let braid_source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&pseudo], |_, _| 1.0).unwrap();
    let permuted = braid_source.permute(&[1, 0], &[2]).unwrap();
    let braided = braid_source.braid(&[1, 0], &[2], &[0, 1, 2]).unwrap();
    // `permute` is the symmetric-braiding permutation (not a raw ndarray
    // transpose), so it and `braid` coincide for CU1.  Both differ from the
    // raw source by the charged exchange coefficient R(q,q;pseudo) = -1.
    assert_eq!(permuted.data(), &[-1.0]);
    assert_eq!(braided.data(), &[-1.0]);
    assert!(std::ptr::eq(braided.provider(), provider.as_ref()));
    let inserted = source.insert_right_unit(0, false).unwrap();
    assert!(inserted
        .codomain()
        .into_iter()
        .chain(inserted.domain())
        .any(|leg| leg.sectors().unwrap() == vec![CU1Irrep::VACUUM]));
}

#[test]
fn su2_and_exact_products_keep_their_provider_through_flip_and_units() {
    let runtime = runtime();
    macro_rules! check {
        ($provider:expr, $label:expr) => {{
            let provider = Arc::new($provider);
            let leg = GradedSpace::try_new_shared(Arc::clone(&provider), [($label, 1)]).unwrap();
            let source: TensorMap<_, f64> =
                TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
            let flipped = source.flip(&[1]).unwrap();
            let inserted = source.insert_left_unit(1, true).unwrap();
            let restored = inserted.remove_unit(1).unwrap();
            for output in [&flipped, &inserted, &restored] {
                assert!(std::ptr::eq(output.provider(), provider.as_ref()));
                assert!(std::ptr::eq(
                    output.codomain()[0].provider(),
                    provider.as_ref()
                ));
            }
            assert_eq!(restored.data(), source.data());
        }};
    }
    check!(SU2FusionRule, SU2Irrep::from_twice_spin(1));
    check!(
        FermionParityFusionRule.product(U1FusionRule),
        product_sector(Z2Irrep::ODD, U1Irrep::new(1))
    );
    check!(
        FermionParityFusionRule
            .product(U1FusionRule)
            .product(SU2FusionRule),
        product_sector(
            product_sector(Z2Irrep::ODD, U1Irrep::new(0)),
            SU2Irrep::from_twice_spin(1)
        )
    );
}

#[test]
fn dual_nonabelian_flip_pins_the_pivotal_phase() {
    let runtime = runtime();
    macro_rules! check {
        ($provider:expr, $label:expr, $axis:expr, $codomain_dual:expr, $domain_dual:expr) => {{
            let provider = Arc::new($provider);
            let plain = GradedSpace::try_new_shared(Arc::clone(&provider), [($label, 1)]).unwrap();
            let dual = plain.try_dual().unwrap();
            let source: TensorMap<_, f64> =
                TensorMap::from_block_fn(&runtime, [&dual], [&plain], |_, _| 1.0).unwrap();
            let flipped = source.flip(&[$axis]).unwrap();
            assert!(std::ptr::eq(flipped.provider(), provider.as_ref()));
            assert!(std::ptr::eq(
                flipped.codomain()[0].provider(),
                provider.as_ref()
            ));
            assert!(std::ptr::eq(
                flipped.domain()[0].provider(),
                provider.as_ref()
            ));
            assert_eq!(flipped.codomain()[0].is_dual(), $codomain_dual);
            assert_eq!(flipped.domain()[0].is_dual(), $domain_dual);
            assert_eq!(flipped.data(), &[-1.0]);
        }};
    }
    // Dual SU2 codomain: the forward factor is χθ=-1.
    check!(SU2FusionRule, SU2Irrep::from_twice_spin(1), 0, false, false);
    check!(
        FermionParityFusionRule
            .product(U1FusionRule)
            .product(SU2FusionRule),
        product_sector(
            product_sector(Z2Irrep::ODD, U1Irrep::new(0)),
            SU2Irrep::from_twice_spin(1)
        ),
        // Plain nested domain: its forward factor is θ=-1.
        1,
        true,
        true
    );
}

#[test]
fn covered_builtin_multiplicity_free_providers_have_cat_and_absorb_execution() {
    let runtime = runtime();
    macro_rules! check {
        ($provider:expr, $label:expr) => {{
            let provider = Arc::new($provider);
            let codomain =
                GradedSpace::try_new_shared(Arc::clone(&provider), [($label, 1)]).unwrap();
            let left = GradedSpace::try_new_shared(Arc::clone(&provider), [($label, 1)]).unwrap();
            let right = GradedSpace::try_new_shared(Arc::clone(&provider), [($label, 1)]).unwrap();
            let a: TensorMap<_, f64> =
                TensorMap::from_block_fn(&runtime, [&codomain], [&left], |_, _| 1.0).unwrap();
            let b: TensorMap<_, f64> =
                TensorMap::from_block_fn(&runtime, [&codomain], [&right], |_, _| 2.0).unwrap();
            let domain = a.catdomain(&b).unwrap();
            let codomain_join = a
                .adjoint()
                .unwrap()
                .catcodomain(&b.adjoint().unwrap())
                .unwrap();
            let absorbed = a.absorb(&b).unwrap();
            for output in [&domain, &codomain_join, &absorbed] {
                assert!(std::ptr::eq(output.provider(), provider.as_ref()));
            }
            // Single-sector column and row slabs, plus overwrite, are raw
            // payload oracles independent of a round trip.
            assert_eq!(domain.data(), &[1.0, 2.0]);
            assert_eq!(codomain_join.data(), &[1.0, 2.0]);
            assert_eq!(absorbed.data(), &[2.0]);
        }};
    }
    check!(Z2FusionRule, Z2Irrep::EVEN);
    check!(
        ZNFusionRule::new(3).unwrap(),
        ZNFusionRule::new(3).unwrap().irrep(1)
    );
    check!(CU1FusionRule, CU1Irrep::from_twice_charge(1));
    check!(FermionParityFusionRule, Z2Irrep::ODD);
    check!(
        FermionParityFusionRule.product(U1FusionRule),
        product_sector(Z2Irrep::ODD, U1Irrep::new(1))
    );
    check!(SU2FusionRule, SU2Irrep::from_twice_spin(1));
    check!(
        FermionParityFusionRule
            .product(U1FusionRule)
            .product(SU2FusionRule),
        product_sector(
            product_sector(Z2Irrep::ODD, U1Irrep::new(1)),
            SU2Irrep::from_twice_spin(1)
        )
    );
}

#[test]
fn zn3_and_cu1_arithmetic_contraction_and_reductions_have_scalar_oracles() {
    let runtime = runtime();
    macro_rules! check {
        ($provider:expr, $label:expr, $qdim:expr) => {{
            let provider = Arc::new($provider);
            let leg = GradedSpace::try_new_shared(Arc::clone(&provider), [($label, 1)]).unwrap();
            let a: TensorMap<_, f64> =
                TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
            let b: TensorMap<_, f64> =
                TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 3.0).unwrap();
            let ordered = a.contract_ordered(&b, &[1], &[0], &[1, 0]).unwrap();
            let composed = a.compose(&b).unwrap();
            let tensor_product = a.otimes(&b).unwrap();
            let sum = a.add(&b, 1.0, -1.0).unwrap();
            let scaled = a.scale(4.0);
            for output in [&ordered, &composed, &tensor_product, &sum, &scaled] {
                assert!(std::ptr::eq(output.provider(), provider.as_ref()));
            }
            for values in [ordered.data(), composed.data(), tensor_product.data()] {
                assert!((values[0] - 6.0).abs() < 1e-12);
            }
            assert_eq!(sum.data(), &[-1.0]);
            assert_eq!(scaled.data(), &[8.0]);
            assert!((a.norm().unwrap() - 2.0 * ($qdim as f64).sqrt()).abs() < 1e-12);
            assert!((a.inner(&b).unwrap() - 6.0 * $qdim).abs() < 1e-12);
            assert!((a.tr().unwrap() - 2.0 * $qdim).abs() < 1e-12);
            assert!(
                (a.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap() - 2.0 * $qdim).abs() < 1e-12
            );
        }};
    }
    check!(
        ZNFusionRule::new(3).unwrap(),
        ZNFusionRule::new(3).unwrap().irrep(1),
        1.0
    );
    check!(CU1FusionRule, CU1Irrep::from_twice_charge(1), 2.0);

    let provider = Arc::new(FermionParityFusionRule);
    let leg = GradedSpace::try_new_shared(Arc::clone(&provider), [(Z2Irrep::ODD, 1)]).unwrap();
    let a: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let b: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 3.0).unwrap();
    assert_eq!(a.inner(&b).unwrap(), 6.0);
}
