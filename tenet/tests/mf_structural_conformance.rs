//! Public, by-value conformance checks for the built-in multiplicity-free providers.
//!
//! This deliberately covers ordinary structural operations only.  Factorizations
//! and matrix functions have their own #1002 gates.

use std::sync::Arc;

use tenet::prelude::{
    product_sector, Complex64, FermionParityFusionRule, GradedSpace, ProductFusionRuleExt, Runtime,
    SU2FusionRule, SU2Irrep, TensorMap, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep, ZNFusionRule,
};

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

#[test]
fn zn3_index_flip_and_units_keep_the_original_provider() {
    let runtime = runtime();
    let provider = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge = |value| provider.irrep(value);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(charge(0), 1), (charge(1), 1), (charge(2), 1)],
        false,
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
        GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)], false).unwrap();
    let lhs_domain =
        GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 1)], false).unwrap();
    let rhs_domain =
        GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)], false).unwrap();
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
        [&GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)], false).unwrap()],
        [&GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 3)], false).unwrap()],
        |_, i| (10 * (i[0] + 1) + i[1] + 1) as f64,
    )
    .unwrap();
    let source: TensorMap<Z2FusionRule, f64> = TensorMap::from_block_fn(
        &runtime,
        [&GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 3)], false).unwrap()],
        [&GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)], false).unwrap()],
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
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(odd, 1)], false).unwrap();
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
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(spin_half, 1)], false).unwrap();
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
    assert_eq!(adjoint.data(), &[Complex64::new(2.0, -3.0)]);
    assert_eq!(
        adjoint.codomain()[0].sectors().unwrap(),
        vec![spin_half],
        "the nonzero SU(2) label survives the public product route"
    );
}
