//! Integration tests for the user-layer Space / Runtime / Tensor API,
//! including elementwise cross-checks against the expert layer.

use tenet::core::{
    FusionAlgebraError, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, SectorId,
    SectorLeg, TensorMap, TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet::operations::{
    OperationError, OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
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
    Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap()
}

#[test]
fn rand_and_zeros_construction_u1_and_su2() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let zero = Tensor::zeros(&rt, Dtype::F64, [&v, &v], [&v, &v]).unwrap();
        assert_eq!(zero.norm().unwrap(), 0.0);
        assert_eq!(zero.codomain_rank(), 2);
        assert_eq!(zero.domain_rank(), 2);

        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 7).unwrap();
        let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 7).unwrap();
        assert_eq!(a.data(), b.data(), "same seed must reproduce the data");
        assert_eq!(a.data().len(), zero.data().len());
        assert!(a.norm().unwrap() > 0.0);

        // The runtime's own stream advances between calls.
        let c = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v]).unwrap();
        let d = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v]).unwrap();
        assert_ne!(c.data(), d.data());
    }
}

#[test]
fn tensor_construction_preserves_lowered_u1_algebra_errors() {
    // What: public tensor construction returns exact forward-fusion and
    // reverse-recursion dual failures from the lowered U1 algebra.
    let rt = Runtime::builder().build().unwrap();
    let maximum = Space::u1([(i32::MAX, 1)]);
    let one = Space::u1([(1, 1)]);
    assert_eq!(
        Tensor::zeros(&rt, Dtype::F64, [&maximum, &one], [])
            .err()
            .expect("MAX + 1 must fail"),
        Error::FusionAlgebra(Box::new(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }))
    );

    let zero = Space::u1([(0, 1)]);
    let minimum = Space::u1([(i32::MIN, 1)]);
    assert_eq!(
        Tensor::zeros(&rt, Dtype::F64, [&zero, &zero, &minimum], [])
            .err()
            .expect("MIN reverse dual must fail"),
        Error::FusionAlgebra(Box::new(FusionAlgebraError::U1DualOverflow {
            charge: i32::MIN
        }))
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn tensor_construction_preserves_lowered_product_child_errors() {
    // What: both packed multiplicity-free product contexts expose the exact
    // U1 child failure reached during lowered tensor construction.
    let rt = Runtime::builder().build().unwrap();
    let pair_maximum = Space::product([((i32::MAX, 0), 1)]).unwrap();
    let pair_one = Space::product([((1, 1), 1)]).unwrap();
    assert_eq!(
        Tensor::zeros(&rt, Dtype::F64, [&pair_maximum, &pair_one], [])
            .err()
            .expect("product U1 child overflow must fail"),
        Error::FusionAlgebra(Box::new(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }))
    );

    let triple_maximum = Space::fz2_u1_su2([((0, i32::MAX, 0), 1)]).unwrap();
    let triple_one = Space::fz2_u1_su2([((1, 1, 1), 1)]).unwrap();
    assert_eq!(
        Tensor::zeros(&rt, Dtype::F64, [&triple_maximum, &triple_one], [])
            .err()
            .expect("triple-product U1 child overflow must fail"),
        Error::FusionAlgebra(Box::new(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }))
    );
}

#[test]
fn tensor_construction_preserves_lowered_su2_closure_and_valid_boundary() {
    // What: an unrepresentable SU2 output is exact, while representable
    // near-boundary U1 construction retains the ordinary zero-tensor result.
    let rt = Runtime::builder().build().unwrap();
    let spin_64 = Space::su2([(128, 1)]).unwrap();
    let spin_63_5 = Space::su2([(127, 1)]).unwrap();
    assert_eq!(
        Tensor::zeros(&rt, Dtype::F64, [&spin_64, &spin_63_5], [])
            .err()
            .expect("SU2 output beyond 254 must fail"),
        Error::FusionAlgebra(Box::new(FusionAlgebraError::FusionNotRepresentable {
            left: SectorId::new(128),
            right: SectorId::new(127),
        }))
    );

    let maximum = Space::u1([(i32::MAX, 2)]);
    let minimum = Space::u1([(i32::MIN, 3)]);
    let boundary = Tensor::zeros(&rt, Dtype::F64, [&maximum, &minimum], []).unwrap();
    let ordinary_left = Space::u1([(1, 2)]);
    let ordinary_right = Space::u1([(-2, 3)]);
    let ordinary = Tensor::zeros(&rt, Dtype::F64, [&ordinary_left, &ordinary_right], []).unwrap();
    assert_eq!(boundary.data(), ordinary.data());
    assert_eq!(boundary.leg_dims(), ordinary.leg_dims());
}

#[test]
fn operation_fusion_algebra_error_flattens_at_the_user_boundary() {
    // What: only the operation-layer algebra variant becomes the existing
    // user algebra variant; unrelated operation errors remain boxed.
    let cause = FusionAlgebraError::U1DualOverflow { charge: i32::MIN };
    assert_eq!(
        Error::from(OperationError::FusionAlgebra(Box::new(cause.clone()))),
        Error::FusionAlgebra(Box::new(cause))
    );
    let unrelated = OperationError::InvalidArgument {
        message: "unrelated operation error",
    };
    assert_eq!(
        Error::from(unrelated.clone()),
        Error::Operation(Box::new(unrelated))
    );
}

#[test]
fn norm_inf_is_entrywise_max_abs() {
    let rt = Runtime::builder().build().unwrap();
    let v = Space::u1([(0, 2)]);
    let real = Tensor::from_block_fn(&rt, [&v], [&v], |_, indices| match indices {
        [0, 0] => 1.0,
        [0, 1] => 2.0,
        [1, 0] => 3.0,
        [1, 1] => 4.0,
        _ => unreachable!(),
    })
    .unwrap();
    assert_eq!(real.norm_inf().unwrap(), 4.0);

    let complex = Tensor::from_block_fn(&rt, [&v], [&v], |_, indices| match indices {
        [0, 0] => Complex64::new(3.0, 4.0),
        [0, 1] => Complex64::new(5.0, 12.0),
        [1, 0] => Complex64::new(8.0, 15.0),
        [1, 1] => Complex64::new(0.0, 6.0),
        _ => unreachable!(),
    })
    .unwrap();
    assert_eq!(complex.norm_inf().unwrap(), 17.0);
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
fn space_try_dual_reports_u1_boundary_and_preserves_compatibility() {
    // What: the checked public dual reports finite-U1 nonclosure without
    // mutating its input, while every representable dual matches dual().
    let minimum = Space::u1([(i32::MIN, 3)]);
    let unchanged = minimum.clone();
    let expected = FusionAlgebraError::U1DualOverflow { charge: i32::MIN };
    let error = minimum.try_dual().unwrap_err();
    assert_eq!(error, Error::FusionAlgebra(Box::new(expected.clone())));
    assert_eq!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<FusionAlgebraError>()),
        Some(&expected)
    );
    assert_eq!(minimum, unchanged);

    let near_minimum = Space::u1([(i32::MIN + 1, 2)]);
    let checked = near_minimum.try_dual().unwrap();
    assert!(checked.is_dual());
    assert_eq!(checked.sectors(), vec![(SectorLabel::U1(i32::MAX), 2)]);
    assert_eq!(checked, near_minimum.dual());

    let ordinary = u1_space();
    let ordinary_checked = ordinary.try_dual().unwrap();
    assert!(ordinary_checked.is_dual());
    assert_eq!(
        ordinary_checked.sectors(),
        vec![
            (SectorLabel::U1(0), 3),
            (SectorLabel::U1(-1), 2),
            (SectorLabel::U1(1), 2),
        ]
    );
    let ordinary_compatibility = ordinary.dual();
    assert!(ordinary_compatibility.is_dual());
    assert_eq!(
        ordinary_compatibility.sectors(),
        vec![
            (SectorLabel::U1(0), 3),
            (SectorLabel::U1(-1), 2),
            (SectorLabel::U1(1), 2),
        ]
    );
}

#[test]
#[should_panic(expected = "use Space::try_dual")]
fn space_dual_compatibility_panics_with_checked_alternative() {
    // What: the value-returning compatibility method names its checked
    // alternative when a finite algebra cannot represent the dual.
    let _ = Space::u1([(i32::MIN, 1)]).dual();
}

#[cfg(target_pointer_width = "64")]
#[test]
fn space_try_dual_preserves_product_child_errors() {
    // What: checked product duals expose the U1 component's exact closure
    // failure and leave both packed product inputs unchanged.
    let pair = Space::product([((i32::MIN, 1), 2)]).unwrap();
    let pair_before = pair.clone();
    assert_eq!(
        pair.try_dual(),
        Err(Error::FusionAlgebra(Box::new(
            FusionAlgebraError::U1DualOverflow { charge: i32::MIN }
        )))
    );
    assert_eq!(pair, pair_before);

    let triple = Space::fz2_u1_su2([((1, i32::MIN, 1), 2)]).unwrap();
    let triple_before = triple.clone();
    assert_eq!(
        triple.try_dual(),
        Err(Error::FusionAlgebra(Box::new(
            FusionAlgebraError::U1DualOverflow { charge: i32::MIN }
        )))
    );
    assert_eq!(triple, triple_before);
}

#[test]
fn space_try_dual_keeps_su3_table_duality_total() {
    // What: SU3 checked dual conjugates each Dynkin label, retains exact
    // degeneracies and table order, toggles variance, and is involutive.
    let space = Space::su3([((1, 1), 4), ((0, 1), 1), ((1, 0), 2)]).unwrap();
    assert!(!space.is_dual());
    assert_eq!(
        space.su3_sectors().unwrap(),
        vec![((0, 1), 1), ((1, 0), 2), ((1, 1), 4)]
    );

    let dual = space.try_dual().unwrap();
    assert!(dual.is_dual());
    assert_eq!(
        dual.su3_sectors().unwrap(),
        vec![((0, 1), 2), ((1, 0), 1), ((1, 1), 4)]
    );
    assert_eq!(dual.try_dual().unwrap(), space);
}

#[test]
fn compose_equals_contract_on_matching_axes() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 1).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 2).unwrap();

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
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 3).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 4).unwrap();

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
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 11).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 12).unwrap();

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
fn ordinary_nonabelian_contracts_match_legacy_lowering_bitwise() {
    macro_rules! check_rule {
        ($rule:expr, $space:expr, $sectors:expr, $seed:expr) => {{
            let rule = $rule;
            let sectors: Vec<(SectorId, usize)> = $sectors;
            let leg_dim = sectors.iter().map(|(_, degeneracy)| degeneracy).sum();
            let leg = || SectorLeg::new(sectors.iter().copied(), false);
            let homspace = || {
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([leg(), leg()]),
                    FusionProductSpace::new([leg(), leg()]),
                )
            };
            let fusion_space = |hom: FusionTreeHomSpace| {
                let shapes: Vec<Vec<usize>> = hom
                    .fusion_tree_keys(&rule)
                    .iter()
                    .map(|key| {
                        key.codomain_uncoupled()
                            .iter()
                            .chain(key.domain_uncoupled())
                            .map(|sector| {
                                sectors
                                    .iter()
                                    .find_map(|(candidate, degeneracy)| {
                                        (candidate == sector).then_some(*degeneracy)
                                    })
                                    .unwrap()
                            })
                            .collect()
                    })
                    .collect();
                FusionTensorMapSpace::from_degeneracy_shapes(
                    TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim])
                        .unwrap(),
                    hom,
                    &rule,
                    shapes,
                )
                .unwrap()
            };

            let runtime = Runtime::builder().build().unwrap();
            let space = $space;
            let lhs = Tensor::rand_with_seed(
                &runtime,
                Dtype::F64,
                [&space, &space],
                [&space, &space],
                $seed,
            )
            .unwrap();
            let rhs = Tensor::rand_with_seed(
                &runtime,
                Dtype::F64,
                [&space, &space],
                [&space, &space],
                $seed + 1,
            )
            .unwrap();
            let lower_lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
                lhs.data().to_vec(),
                fusion_space(homspace()),
            )
            .unwrap();
            let lower_rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
                rhs.data().to_vec(),
                fusion_space(homspace()),
            )
            .unwrap();

            let lower_contract = |lhs_axes: &[usize], rhs_axes: &[usize], output_axes: &[usize]| {
                let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
                    &rule,
                    lower_lhs.fusion_space().unwrap().homspace(),
                    lower_rhs.fusion_space().unwrap().homspace(),
                    lhs_axes,
                    rhs_axes,
                    output_axes,
                    2,
                )
                .unwrap();
                let dst_space = fusion_space(dst_hom);
                let mut dst = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
                    vec![0.0; dst_space.required_len().unwrap()],
                    dst_space,
                )
                .unwrap();
                TensorContractFusionExecutionContext::<f64, _>::default()
                    .tensorcontract_fusion_into(
                        &rule,
                        &mut dst,
                        &lower_lhs,
                        &lower_rhs,
                        TensorContractSpec::new(
                            lhs_axes,
                            rhs_axes,
                            OutputAxisOrder::from_axes(output_axes),
                        ),
                        1.0,
                        0.0,
                    )
                    .unwrap();
                dst.data().to_vec()
            };

            let default = lower_contract(&[2, 3], &[0, 1], &[0, 1, 2, 3]);
            let ordered = lower_contract(&[3, 2], &[0, 1], &[1, 0, 2, 3]);
            // What: ordinary facade dispatch preserves the exact legacy
            // accumulation order for contract, ordered contract, and compose.
            assert_eq!(
                lhs.contract(&rhs, &[2, 3], &[0, 1]).unwrap().data(),
                default
            );
            assert_eq!(lhs.compose(&rhs).unwrap().data(), default);
            assert_eq!(
                lhs.contract_ordered(&rhs, &[3, 2], &[0, 1], &[1, 0, 2, 3])
                    .unwrap()
                    .data(),
                ordered
            );
        }};
    }

    check_rule!(
        FermionParityFusionRule,
        Space::fz2([(0, 2), (1, 3)]).unwrap(),
        vec![(SectorId::new(0), 2), (SectorId::new(1), 3)],
        261_801
    );
    check_rule!(
        SU2FusionRule,
        Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap(),
        vec![
            (SU2Irrep::from_twice_spin(0).sector_id(), 2),
            (SU2Irrep::from_twice_spin(1).sector_id(), 2),
            (SU2Irrep::from_twice_spin(2).sector_id(), 1),
        ],
        261_811
    );
    check_rule!(
        triple_rule(),
        Space::fz2_u1_su2([
            ((0, 0, 0), 2),
            ((1, 1, 1), 2),
            ((0, -1, 2), 1),
            ((1, 0, 1), 1),
        ])
        .unwrap(),
        vec![
            (triple_sector(0, 0, 0), 2),
            (triple_sector(1, 1, 1), 2),
            (triple_sector(0, -1, 2), 1),
            (triple_sector(1, 0, 1), 1),
        ],
        261_821
    );
}

#[test]
fn permute_roundtrip_restores_the_tensor() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let c = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 21).unwrap();
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
        let c = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 22).unwrap();
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
        let c = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 31).unwrap();

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
    let c = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 41).unwrap();
    let p = c.permute(&[1, 0], &[2, 3]).unwrap();
    let b = c.braid(&[1, 0], &[2, 3], &[0, 1, 2, 3]).unwrap();
    assert_close(b.data(), p.data(), 1e-12);
}

#[test]
fn vector_interface_identities() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), su2_space()] {
        let c = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 51).unwrap();
        let d = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 52).unwrap();

        let norm = c.norm().unwrap();
        let inner_cc = c.inner(&c).unwrap().re();
        assert!((inner_cc - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm));

        let scaled = c.scale(0.5).unwrap();
        assert!((scaled.norm().unwrap() - 0.5 * norm).abs() <= 1e-10 * (1.0 + norm));

        // dot is inner (same conjugation, same weighting).
        assert_eq!(c.dot(&d).unwrap().re(), c.inner(&d).unwrap().re());

        // normalize returns a unit-norm tensor.
        let unit = c.normalize().unwrap();
        assert!((unit.norm().unwrap() - 1.0).abs() <= 1e-10);

        // w = c - d; |w|^2 = <c,c> - 2<c,d> + <d,d>.
        let w = c.add(&d, 1.0, -1.0).unwrap();
        let expected = inner_cc - 2.0 * c.inner(&d).unwrap().re() + d.inner(&d).unwrap().re();
        let actual = w.inner(&w).unwrap().re();
        assert!((actual - expected).abs() <= 1e-10 * (1.0 + expected.abs()));
    }
}

#[test]
fn fz2_and_product_rule_smoke() {
    let rt = Runtime::builder().build().unwrap();

    let f = Space::fz2([(0, 2), (1, 2)]).unwrap();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&f, &f], [&f, &f], 61).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&f, &f], [&f, &f], 62).unwrap();
    assert!(a.compose(&b).unwrap().norm().unwrap() > 0.0);

    let p = Space::product([((-1, 1), 2), ((0, 0), 2), ((1, 1), 2)]).unwrap();
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&p, &p], [&p, &p], 63).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&p, &p], [&p, &p], 64).unwrap();
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
        Tensor::rand(&rt, Dtype::F64, [&u], [&z]),
        Err(Error::RuleMismatch)
    ));

    // Mixed rules across an operation.
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&u], [&u], 71).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&z], [&z], 72).unwrap();
    assert!(matches!(a.compose(&b), Err(Error::RuleMismatch)));
    assert!(matches!(a.add(&b, 1.0, 1.0), Err(Error::RuleMismatch)));
    assert!(matches!(a.inner(&b), Err(Error::RuleMismatch)));

    // Same rule, different runtimes.
    let rt2 = Runtime::builder().build().unwrap();
    let c = Tensor::rand_with_seed(&rt2, Dtype::F64, [&u], [&u], 73).unwrap();
    assert!(matches!(a.compose(&c), Err(Error::RuntimeMismatch)));

    // Same rule and runtime, different spaces.
    let w = Space::u1([(0, 1), (1, 1)]);
    let d = Tensor::rand_with_seed(&rt, Dtype::F64, [&w], [&w], 74).unwrap();
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
        let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v, &v, &v, &v], 81).unwrap();
        assert_eq!(a.codomain_rank(), 1);
        assert_eq!(a.domain_rank(), 4);
        assert_eq!(a.rank(), 5);
        let norm = a.norm().unwrap();
        assert!(norm > 0.0);

        // Contract two rank-5 tensors over two shared legs (a's domain legs
        // against b's dual domain legs): rank-6 result.
        let w = v.dual();
        let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&w, &w, &v, &v], 82).unwrap();
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
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v, &v, &v, &v], 91).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&w, &w, &v, &v], 92).unwrap();

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
    let r = Tensor::rand(&rt, Dtype::F64, [&v], [&v]).unwrap();
    assert!(r.compose(&r).is_ok());
}

/// Pairing follows Space identity (TensorKit: `domain(A) == codomain(B)`),
/// independent of dualization closure of the sector content.
#[test]
fn leg_pairing_rules_on_asymmetric_charges() {
    let rt = Runtime::builder().build().unwrap();
    let v = Space::u1([(0, 1), (1, 1)]);
    let a = Tensor::rand(&rt, Dtype::F64, [&v], [&v]).unwrap();

    // Domain V does NOT pair with codomain V' (TensorKit SpaceMismatch).
    let bad = Tensor::rand(&rt, Dtype::F64, [&v.dual()], [&v]).unwrap();
    assert!(a.compose(&bad).is_err());

    // Domain-vs-domain legs contract when exactly one side is the dual.
    let b = Tensor::rand(&rt, Dtype::F64, [&v], [&v.dual()]).unwrap();
    assert!(a.contract(&b, &[1], &[1]).is_ok());

    // ...and are rejected when both sides carry the same Space.
    let c = Tensor::rand(&rt, Dtype::F64, [&v], [&v]).unwrap();
    assert!(a.contract(&c, &[1], &[1]).is_err());
}

// ---------------------------------------------------------------------------
// fZ2 ⊠ U(1) ⊠ SU(2) triple product (left-associated, TensorKit `Vect[
// FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂]]`). Hardcoded reference values from
// TensorKit (Julia, `benchmarks` env), see comments at each assertion.
// ---------------------------------------------------------------------------

use tenet::core::{
    FermionParityFusionRule, FusionRule, Fz2SectorLayout, MultiplicityFreeRigidSymbols,
    PackedProductCodec, ProductFusionRule, ProductSectorCodec, ProductSectorLayout, SU2FusionRule,
    SU2Irrep, Su2SectorLayout, U1SectorLayout,
};

type FpU1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type FpU1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, FpU1Codec>;
type TripleCodec = PackedProductCodec<FpU1Layout, Su2SectorLayout>;
type TripleRule = ProductFusionRule<FpU1Rule, SU2FusionRule, TripleCodec>;

fn triple_rule() -> TripleRule {
    ProductFusionRule::new(
        ProductFusionRule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    )
}

/// Packed sector id of `(parity ⊠ charge ⊠ twice_spin)`, left-associated.
fn triple_sector(parity: u8, charge: i32, twice_spin: usize) -> SectorId {
    let inner = FpU1Codec::encode(
        SectorId::new(usize::from(parity & 1)),
        U1Irrep::new(charge).sector_id(),
    );
    TripleCodec::encode(inner, SU2Irrep::from_twice_spin(twice_spin).sector_id())
}

/// The Julia reference space:
/// `S = Vect[FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂]]((0,0,0)=>1, (1,1,1//2)=>1, (0,2,0)=>1)`
fn triple_space() -> Space {
    Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 1, 1), 1), ((0, 2, 0), 1)]).unwrap()
}

#[cfg(target_pointer_width = "64")]
#[test]
fn fz2_u1_su2_codec_roundtrip_and_dual() {
    // Encode/decode round-trip of the built-in packed codec for triples.
    for &(parity, charge, twice_spin) in &[
        (0u8, 0i32, 0usize),
        (1, 1, 1),
        (0, 2, 0),
        (1, -1, 1),
        (0, -2, 4),
        (1, 5, 3),
        (0, i32::MIN, 254),
        (1, i32::MAX, 254),
    ] {
        let sector = triple_sector(parity, charge, twice_spin);
        let (inner, su2) = TripleCodec::decode(sector).unwrap();
        let (fz2, u1) = FpU1Codec::decode(inner).unwrap();
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

#[cfg(target_pointer_width = "64")]
#[test]
fn fz2_u1_su2_space_represents_the_full_packed_u1_label_domain() {
    // What: construction and label readback preserve both i32 endpoints;
    // algebraic overflow behavior is tracked separately in issue #274.
    let space = Space::fz2_u1_su2([((0, i32::MIN, 0), 2), ((1, i32::MAX, 254), 3)]).unwrap();
    assert_eq!(
        space.sectors(),
        vec![
            (
                SectorLabel::FZ2U1SU2 {
                    parity: 0,
                    charge: i32::MIN,
                    twice_spin: 0,
                },
                2,
            ),
            (
                SectorLabel::FZ2U1SU2 {
                    parity: 1,
                    charge: i32::MAX,
                    twice_spin: 254,
                },
                3,
            ),
        ]
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn u1_fz2_space_represents_the_full_packed_u1_label_domain() {
    // What: pair construction and label readback preserve both i32 endpoints;
    // algebraic overflow behavior is tracked separately in issue #274.
    let space = Space::product([((i32::MIN, 0), 2), ((i32::MAX, 1), 3)]).unwrap();
    assert_eq!(
        space.sectors(),
        vec![
            (
                SectorLabel::U1FZ2 {
                    charge: i32::MIN,
                    parity: 0,
                },
                2,
            ),
            (
                SectorLabel::U1FZ2 {
                    charge: i32::MAX,
                    parity: 1,
                },
                3,
            ),
        ]
    );
}

#[test]
fn fz2_u1_su2_space_rejects_invalid_su2_child() {
    // What: the user boundary preserves the exact invalid SU2 child label.
    let error = Space::fz2_u1_su2([((1, 0, 255), 1)]).unwrap_err();
    assert_eq!(
        error,
        Error::FusionAlgebra(Box::new(FusionAlgebraError::InvalidSector {
            sector: SectorId::new(255),
        }))
    );
}

#[test]
fn builtin_space_constructors_reject_invalid_numeric_labels() {
    // What: every ordinary fZ2 input and the ordinary SU2 input expose the
    // existing typed invalid-sector authority without normalization or unwind.
    let invalid_parity = Error::FusionAlgebra(Box::new(FusionAlgebraError::InvalidSector {
        sector: SectorId::new(2),
    }));
    assert_eq!(Space::fz2([(2, 1)]).unwrap_err(), invalid_parity);
    assert_eq!(Space::product([((0, 2), 1)]).unwrap_err(), invalid_parity);
    assert_eq!(
        Space::fz2_u1_su2([((2, 0, 0), 1)]).unwrap_err(),
        invalid_parity
    );

    assert_eq!(
        Space::su2([(255, 1)]).unwrap_err(),
        Error::FusionAlgebra(Box::new(FusionAlgebraError::InvalidSector {
            sector: SectorId::new(255),
        }))
    );
}

#[test]
fn builtin_space_constructor_signatures_are_checked() {
    // What: public callers must handle fZ2 and SU2 construction failure.
    let _: Result<Space, Error> = Space::fz2([(0, 1)]);
    let _: Result<Space, Error> = Space::su2([(0, 1)]);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn builtin_product_spaces_report_the_target_width_boundary() {
    // What: 32-bit targets reject the fixed 33/41-bit built-in layouts with a
    // typed user error instead of truncating otherwise valid labels.
    let pair = Space::product([((0, 0), 1)]).unwrap_err();
    assert!(pair.to_string().contains("needs 33 bits"));

    let triple = Space::fz2_u1_su2([((0, 0, 0), 1)]).unwrap_err();
    assert!(triple.to_string().contains("needs 41 bits"));
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
    let inner = id.inner(&id).unwrap().re();
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
            BlockKey::FusionTree(key) if key.coupled() == sector => 1.0,
            _ => 0.0,
        })
        .unwrap();
        let before = proj.inner(&t).unwrap().re();
        let after = proj.inner(&tb).unwrap().re();
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
    let a = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 91).unwrap();
    let b = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 92).unwrap();
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
    let p = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v, &v, &v, &v], 93).unwrap();
    let q = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&w, &w, &v, &v], 94).unwrap();
    let r = p.contract(&q, &[3, 4], &[1, 2]).unwrap();
    assert_eq!(r.rank(), 6);
    assert!(r.norm().unwrap() > 0.0);
    let rp = p.permute(&[0, 2, 4], &[1, 3]).unwrap();
    let p_back = rp.permute(&[0], &[3, 1, 4, 2]).unwrap();
    assert_close(p_back.data(), p.data(), 1e-12);
    let h = p.adjoint().unwrap().adjoint().unwrap();
    assert_close(h.data(), p.data(), 1e-12);
}

// ---------------------------------------------------------------------------
// Space sector introspection and fusion (TensorKit `sectors` / `dim(V,c)` /
// `fuse` analogs). Expected values cross-checked against TensorKit
// (spaces/gradedspace.jl fuse, and a live Julia run for the dual-input and
// triple-product cases).
// ---------------------------------------------------------------------------

fn sector_set(space: &Space) -> std::collections::HashSet<(SectorLabel, usize)> {
    space.sectors().into_iter().collect()
}

fn set_of<const N: usize>(
    pairs: [(SectorLabel, usize); N],
) -> std::collections::HashSet<(SectorLabel, usize)> {
    pairs.into_iter().collect()
}

#[test]
fn space_sectors_round_trip_all_constructors() {
    let u1 = Space::u1([(-1, 2), (0, 3), (1, 2)]);
    assert_eq!(
        sector_set(&u1),
        set_of([
            (SectorLabel::U1(-1), 2),
            (SectorLabel::U1(0), 3),
            (SectorLabel::U1(1), 2),
        ])
    );

    let z2 = Space::z2([(0, 2), (1, 3)]);
    assert_eq!(
        sector_set(&z2),
        set_of([(SectorLabel::Z2(0), 2), (SectorLabel::Z2(1), 3)])
    );

    let fz2 = Space::fz2([(0, 1), (1, 4)]).unwrap();
    assert_eq!(
        sector_set(&fz2),
        set_of([(SectorLabel::FZ2(0), 1), (SectorLabel::FZ2(1), 4)])
    );

    let su2 = Space::su2([(0, 2), (1, 3), (2, 1)]).unwrap();
    assert_eq!(
        sector_set(&su2),
        set_of([
            (SectorLabel::SU2 { twice_spin: 0 }, 2),
            (SectorLabel::SU2 { twice_spin: 1 }, 3),
            (SectorLabel::SU2 { twice_spin: 2 }, 1),
        ])
    );

    let product = Space::product([((0, 0), 2), ((-1, 1), 3)]).unwrap();
    assert_eq!(
        sector_set(&product),
        set_of([
            (
                SectorLabel::U1FZ2 {
                    charge: 0,
                    parity: 0
                },
                2
            ),
            (
                SectorLabel::U1FZ2 {
                    charge: -1,
                    parity: 1
                },
                3
            ),
        ])
    );

    let triple = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, -1, 1), 2)]).unwrap();
    assert_eq!(
        sector_set(&triple),
        set_of([
            (
                SectorLabel::FZ2U1SU2 {
                    parity: 0,
                    charge: 0,
                    twice_spin: 0
                },
                1
            ),
            (
                SectorLabel::FZ2U1SU2 {
                    parity: 1,
                    charge: -1,
                    twice_spin: 1
                },
                2
            ),
        ])
    );
}

#[test]
fn z2_numeric_labels_remain_cyclic_modulo_two() {
    // What: Z2 keeps TensorKit ZNIrrep{2} modulo semantics and valid IDs.
    let space = Space::z2([(2, 3), (3, 4)]);
    assert_eq!(
        space.sectors(),
        vec![(SectorLabel::Z2(0), 3), (SectorLabel::Z2(1), 4)]
    );
    assert_eq!(space.degeneracy(SectorLabel::Z2(4)), Some(3));
    assert_eq!(space.degeneracy(SectorLabel::Z2(5)), Some(4));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn invalid_builtin_sector_labels_are_absent() {
    // What: invalid fZ2 and SU2 labels never normalize or unwind during lookup.
    let fz2 = Space::fz2([(0, 2), (1, 3)]).unwrap();
    assert_eq!(fz2.degeneracy(SectorLabel::FZ2(2)), None);
    assert!(!fz2.has_sector(SectorLabel::FZ2(3)));

    let su2 = Space::su2([(0, 2), (254, 3)]).unwrap();
    assert_eq!(su2.degeneracy(SectorLabel::SU2 { twice_spin: 255 }), None);
    assert!(!su2.has_sector(SectorLabel::SU2 {
        twice_spin: usize::MAX,
    }));

    let pair = Space::product([((0, 0), 2)]).unwrap();
    assert_eq!(
        pair.degeneracy(SectorLabel::U1FZ2 {
            charge: 0,
            parity: 2,
        }),
        None
    );

    let triple = Space::fz2_u1_su2([((0, 0, 0), 2)]).unwrap();
    assert_eq!(
        triple.degeneracy(SectorLabel::FZ2U1SU2 {
            parity: 2,
            charge: 0,
            twice_spin: 0,
        }),
        None
    );
    assert_eq!(
        triple.degeneracy(SectorLabel::FZ2U1SU2 {
            parity: 0,
            charge: 0,
            twice_spin: 255,
        }),
        None
    );
}

#[test]
fn try_sectors_matches_sectors_on_multiplicity_free_rules() {
    // try_sectors is Ok with byte-identical content to sectors() on every
    // mult-free rule (the SU(3) Err path lives in su3_panic_firewall.rs).
    for space in [
        Space::u1([(-1, 2), (0, 3), (1, 2)]),
        Space::z2([(0, 2), (1, 3)]),
        Space::su2([(0, 2), (1, 3)]).unwrap(),
    ] {
        assert_eq!(space.try_sectors(), Ok(space.sectors()));
    }
}

#[test]
fn space_degeneracy_and_is_dual() {
    let v = Space::u1([(-1, 2), (0, 3)]);
    assert_eq!(v.degeneracy(SectorLabel::U1(-1)), Some(2));
    assert_eq!(v.degeneracy(SectorLabel::U1(0)), Some(3));
    assert_eq!(v.degeneracy(SectorLabel::U1(5)), None);
    // Rule-mismatched label is None, not a panic.
    assert_eq!(v.degeneracy(SectorLabel::Z2(0)), None);

    assert!(!v.is_dual());
    let w = v.dual();
    assert!(w.is_dual());
    // dual() stores external sectors: the dual space reports negated charges.
    assert_eq!(w.degeneracy(SectorLabel::U1(1)), Some(2));
    assert_eq!(w.degeneracy(SectorLabel::U1(-1)), None);

    assert!(v.same_rule(&w));
    assert!(!v.same_rule(&Space::z2([(0, 1)])));
}

#[test]
fn space_fuse_u1_is_charge_convolution() {
    // TensorKit: fuse(U1Space(0=>1,1=>1), same) == Rep[U1](0=>1, 1=>2, 2=>1).
    let v = Space::u1([(0, 1), (1, 1)]);
    let fused = v.fuse(&v).unwrap();
    assert_eq!(
        sector_set(&fused),
        set_of([
            (SectorLabel::U1(0), 1),
            (SectorLabel::U1(1), 2),
            (SectorLabel::U1(2), 1),
        ])
    );
    assert!(!fused.is_dual());
    assert_eq!(fused.dim(), v.dim() * v.dim());

    // TensorKit: fuse(dual(V), V) == Rep[U1](-1=>1, 0=>2, 1=>1) — the dual
    // input enters through its external (negated) charges, result non-dual.
    let mixed = v.dual().fuse(&v).unwrap();
    assert_eq!(
        sector_set(&mixed),
        set_of([
            (SectorLabel::U1(-1), 1),
            (SectorLabel::U1(0), 2),
            (SectorLabel::U1(1), 1),
        ])
    );
    assert!(!mixed.is_dual());

    // Mixing rules is an error.
    assert_eq!(
        v.fuse(&Space::z2([(0, 1)])).unwrap_err(),
        Error::RuleMismatch
    );
}

#[test]
fn space_fuse_reports_u1_boundary_and_preserves_valid_extremes() {
    // What: positive and negative finite-U1 nonclosure are exact errors,
    // while the asymmetric representable endpoint sum remains charge -1.
    for (left_charge, right_charge) in [(i32::MAX, 1), (i32::MIN, -1)] {
        let left = Space::u1([(left_charge, 2)]);
        let right = Space::u1([(right_charge, 3)]);
        let left_before = left.clone();
        let right_before = right.clone();
        assert_eq!(
            left.fuse(&right),
            Err(Error::FusionAlgebra(Box::new(
                FusionAlgebraError::U1FusionOverflow {
                    left: left_charge,
                    right: right_charge,
                }
            )))
        );
        assert_eq!(left, left_before);
        assert_eq!(right, right_before);
    }

    let maximum = Space::u1([(i32::MAX, 2)]);
    let minimum = Space::u1([(i32::MIN, 3)]);
    let fused = maximum.fuse(&minimum).unwrap();
    assert_eq!(fused.sectors(), vec![(SectorLabel::U1(-1), 6)]);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn space_fuse_preserves_product_child_errors() {
    // What: both public packed product families preserve the exact U1 child
    // failure instead of converting it to a codec or generic argument error.
    let pair_left = Space::product([((i32::MAX, 0), 1)]).unwrap();
    let pair_right = Space::product([((1, 1), 1)]).unwrap();
    assert_eq!(
        pair_left.fuse(&pair_right),
        Err(Error::FusionAlgebra(Box::new(
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            }
        )))
    );

    let triple_left = Space::fz2_u1_su2([((0, i32::MAX, 0), 1)]).unwrap();
    let triple_right = Space::fz2_u1_su2([((1, 1, 1), 1)]).unwrap();
    assert_eq!(
        triple_left.fuse(&triple_right),
        Err(Error::FusionAlgebra(Box::new(
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            }
        )))
    );
}

#[test]
fn space_fuse_reports_su2_output_outside_supported_domain() {
    // What: valid SU2 input labels whose output exceeds the supported table
    // return closure failure rather than being misclassified as invalid input.
    let left = Space::su2([(128, 1)]).unwrap();
    let right = Space::su2([(127, 1)]).unwrap();
    assert_eq!(
        left.fuse(&right),
        Err(Error::FusionAlgebra(Box::new(
            FusionAlgebraError::FusionNotRepresentable {
                left: SectorId::new(128),
                right: SectorId::new(127),
            }
        )))
    );
}

#[test]
fn space_fuse_su2_half_times_half() {
    // TensorKit: fuse(SU2Space(1/2=>1), same) == Rep[SU2](0=>1, 1=>1).
    let half = Space::su2([(1, 1)]).unwrap();
    let fused = half.fuse(&half).unwrap();
    assert_eq!(
        sector_set(&fused),
        set_of([
            (SectorLabel::SU2 { twice_spin: 0 }, 1),
            (SectorLabel::SU2 { twice_spin: 2 }, 1),
        ])
    );
    // Quantum-dimension-weighted multiplicativity: dim 2 * 2 = 1 + 3.
    assert_eq!(fused.dim(), half.dim() * half.dim());

    // A degenerate multi-spin case: (j=0 x2, j=1/2 x1) squared.
    let a = Space::su2([(0, 2), (1, 1)]).unwrap();
    let fused = a.fuse(&a).unwrap();
    // 0x0 (x4), 1/2x1/2 -> 0: total 5; 0x1/2 + 1/2x0: 4; 1/2x1/2 -> 1: 1.
    assert_eq!(
        sector_set(&fused),
        set_of([
            (SectorLabel::SU2 { twice_spin: 0 }, 5),
            (SectorLabel::SU2 { twice_spin: 1 }, 4),
            (SectorLabel::SU2 { twice_spin: 2 }, 1),
        ])
    );
    assert_eq!(fused.dim(), a.dim() * a.dim());
}

#[test]
fn space_fuse_triple_product_dual_pair() {
    // The finite-torus fuser shape: fuse(dual(l), l). Cross-checked against
    // TensorKit: L = Vect[FermionParity x Irrep[U1] x Irrep[SU2]](
    //   (0,0,0)=>1, (1,1,1/2)=>1);
    // fuse(dual(L), L) == ((0,0,0)=>2, (0,0,1)=>1, (1,1,1/2)=>1,
    //   (1,-1,1/2)=>1), dim 3 -> 9.
    let l = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 1, 1), 1)]).unwrap();
    assert_eq!(l.dim(), 3);
    let fused = l.dual().fuse(&l).unwrap();
    assert_eq!(
        sector_set(&fused),
        set_of([
            (
                SectorLabel::FZ2U1SU2 {
                    parity: 0,
                    charge: 0,
                    twice_spin: 0
                },
                2
            ),
            (
                SectorLabel::FZ2U1SU2 {
                    parity: 0,
                    charge: 0,
                    twice_spin: 2
                },
                1
            ),
            (
                SectorLabel::FZ2U1SU2 {
                    parity: 1,
                    charge: 1,
                    twice_spin: 1
                },
                1
            ),
            (
                SectorLabel::FZ2U1SU2 {
                    parity: 1,
                    charge: -1,
                    twice_spin: 1
                },
                1
            ),
        ])
    );
    assert_eq!(fused.dim(), l.dim() * l.dim());
    assert!(!fused.is_dual());
}

#[test]
fn space_fuse_all_matches_pairwise_fold() {
    let v = Space::u1([(0, 1), (1, 1)]);
    let w = Space::u1([(-1, 1), (0, 2)]);
    let folded = v.fuse(&w).unwrap().fuse(&v).unwrap();
    let nary = Space::fuse_all(&[&v, &w, &v]).unwrap();
    assert_eq!(sector_set(&nary), sector_set(&folded));
    assert_eq!(nary.dim(), v.dim() * w.dim() * v.dim());

    // Unary fuse of a dual space flips it to the isomorphic non-dual space
    // (TensorKit `fuse(V) = isdual(V) ? flip(V) : V`).
    let flipped = Space::fuse_all(&[&v.dual()]).unwrap();
    assert!(!flipped.is_dual());
    assert_eq!(
        sector_set(&flipped),
        set_of([(SectorLabel::U1(0), 1), (SectorLabel::U1(-1), 1)])
    );

    assert!(Space::fuse_all(&[]).is_err());
}

#[test]
fn space_fuse_all_propagates_checked_errors_in_fold_order() {
    // What: variadic fusion returns the first deterministic algebra failure
    // and retains the existing empty and rule-mismatch error contracts.
    let maximum = Space::u1([(i32::MAX, 1)]);
    let one = Space::u1([(1, 1)]);
    let zero = Space::u1([(0, 1)]);
    assert_eq!(
        Space::fuse_all(&[&zero, &maximum, &one]),
        Err(Error::FusionAlgebra(Box::new(
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            }
        )))
    );
    assert_eq!(
        Space::fuse_all(&[&zero, &Space::z2([(0, 1)])]),
        Err(Error::RuleMismatch)
    );
    assert!(matches!(
        Space::fuse_all(&[]),
        Err(Error::InvalidArgument(_))
    ));
}

// ---------------------------------------------------------------------------
// Structural constructors (id / isomorphism / unitary / isometry) and twist,
// cross-checked against TensorKit 0.17.0 (Julia). The generating script and
// its output live in this comment block; every hardcoded number below comes
// from that run.
//
// ```julia
// using TensorKit
// const FermionParity = TensorKit.FermionParity
// label(c::U1Irrep) = c.charge
// label(c::FermionParity) = Int(c.isodd)
// label(c::SU2Irrep) = Int(2 * c.j)
// entry(sectors, idx) =
//     sum(100.0^(k - 1) * label(s) for (k, s) in enumerate(reverse(collect(sectors)))) +
//     sum(0.1^k * i for (k, i) in enumerate(Tuple(idx)))
// function filled!(t)
//     for (f1, f2) in fusiontrees(t)
//         b = t[f1, f2]
//         for idx in CartesianIndices(b)
//             b[idx] = entry((f1.uncoupled..., f2.uncoupled...), idx)
//         end
//     end
//     return t
// end
// function run_case(name, l, m; twists = Int[])
//     P = dual(l) ⊗ l
//     g = fuse(P)
//     F = isomorphism(g, P)
//     t = filled!(zeros(Float64, P ← m))
//     w = filled!(zeros(Float64, g ← m))
//     Ft = isempty(twists) ? F : twist(F, twists)
//     println("$name: dot(w, F*t) = ", dot(w, Ft * t))
// end
// run_case("u1", U1Space(0 => 1, 1 => 2), U1Space(-1 => 1, 0 => 1, 1 => 1))
// l_f = Vect[FermionParity](0 => 1, 1 => 2); m_f = Vect[FermionParity](0 => 1, 1 => 1)
// run_case("fz2", l_f, m_f)
// run_case("fz2 tw[2]", l_f, m_f; twists = [2])
// run_case("fz2 tw[3]", l_f, m_f; twists = [3])
// run_case("fz2 tw[1]", l_f, m_f; twists = [1])
// run_case("su2", SU2Space(0 => 1, 1 // 2 => 2), SU2Space(0 => 1, 1 // 2 => 1, 1 => 1))
// println("norm(id(P_u1)) = ", norm(id(dual(U1Space(0 => 1, 1 => 2)) ⊗ U1Space(0 => 1, 1 => 2))))
// W = isometry(SU2Space(0 => 2, 1 // 2 => 3, 1 => 1), SU2Space(0 => 1, 1 // 2 => 1))
// println("norm(W' * W - id(SU2Space(0 => 1, 1 // 2 => 1))) = ", norm(W' * W - id(SU2Space(0 => 1, 1 // 2 => 1))))
// println("norm(W) = ", norm(W))
// ```
//
// Output (TensorKit v0.17.0, Julia 1.11.6):
//   u1: dot(w, F*t) = 2.0231712673900002e6
//   fz2      : dot(w, F*t) = 2.0584773977899998e6
//   fz2 tw[2]: dot(w, F*t) = -2.01748090133e6
//   fz2 tw[3]: dot(w, F*t) = 1.98839242367e6
//   fz2 tw[1]: dot(w, F*t) = -2.02938887129e6
//   su2: dot(w, F*t) = 2.8621579710249998e7
//   norm(id(P_u1)) = 3.0
//   norm(W' * W - id(...)) = 0.0
//   norm(W) = 1.7320508075688772
// ---------------------------------------------------------------------------

/// The Julia `entry` function: sector labels (codomain legs first, then
/// domain legs, least-significant last) in powers of 100 plus 1-based
/// block-local indices in powers of 0.1.
fn oracle_entry(labels: &[f64], indices: &[usize]) -> f64 {
    let mut value = 0.0;
    for (k, &label) in labels.iter().rev().enumerate() {
        value += 100f64.powi(k as i32) * label;
    }
    for (k, &index) in indices.iter().enumerate() {
        value += 0.1f64.powi(k as i32 + 1) * (index as f64 + 1.0);
    }
    value
}

/// `dot(w, isomorphism(fuse(dual(l) ⊗ l) ← dual(l) ⊗ l).twist(twists) * t)`
/// with the deterministic block entries of the Julia script; `label` decodes
/// a SectorId into the Julia sector label.
fn fuser_oracle_scalar(
    rt: &Runtime,
    l: &Space,
    m: &Space,
    twists: &[usize],
    label: impl Fn(SectorId) -> f64,
) -> f64 {
    let fill = |key: &BlockKey, indices: &[usize]| -> f64 {
        let BlockKey::FusionTree(key) = key else {
            panic!("expected fusion-tree block keys");
        };
        let labels: Vec<f64> = key
            .codomain_uncoupled()
            .iter()
            .chain(key.domain_uncoupled())
            .map(|&sector| label(sector))
            .collect();
        oracle_entry(&labels, indices)
    };
    let fused = l.dual().fuse(l).unwrap();
    let fuser = Tensor::isomorphism(rt, Dtype::F64, [&fused], [&l.dual(), l])
        .unwrap()
        .twist(twists)
        .unwrap();
    let t = Tensor::from_block_fn(rt, [&l.dual(), l], [m], fill).unwrap();
    let w = Tensor::from_block_fn(rt, [&fused], [m], fill).unwrap();
    let value = w.inner(&fuser.compose(&t).unwrap()).unwrap();
    assert_eq!(value.im(), 0.0);
    value.re()
}

fn assert_rel(value: f64, expected: f64) {
    assert!(
        (value - expected).abs() <= 1e-9 * expected.abs(),
        "{value} != {expected}"
    );
}

#[test]
fn fuser_contraction_matches_tensorkit_u1() {
    use tenet::core::U1Irrep;
    let rt = Runtime::builder().build().unwrap();
    let l = Space::u1([(0, 1), (1, 2)]);
    let m = Space::u1([(-1, 1), (0, 1), (1, 1)]);
    let label = |sector: SectorId| f64::from(U1Irrep::from_sector_id(sector).unwrap().charge());
    let value = fuser_oracle_scalar(&rt, &l, &m, &[], label);
    assert_rel(value, 2.0231712673900002e6);
}

#[test]
fn fuser_contraction_and_twist_match_tensorkit_fz2() {
    let rt = Runtime::builder().build().unwrap();
    let l = Space::fz2([(0, 1), (1, 2)]).unwrap();
    let m = Space::fz2([(0, 1), (1, 1)]).unwrap();
    let label = |sector: SectorId| sector.id() as f64;
    // Untwisted fuser, then the twist on each of the three legs (tenet flat
    // leg i is Julia index i+1).
    assert_rel(
        fuser_oracle_scalar(&rt, &l, &m, &[], label),
        2.0584773977899998e6,
    );
    assert_rel(
        fuser_oracle_scalar(&rt, &l, &m, &[1], label),
        -2.01748090133e6,
    );
    assert_rel(
        fuser_oracle_scalar(&rt, &l, &m, &[2], label),
        1.98839242367e6,
    );
    assert_rel(
        fuser_oracle_scalar(&rt, &l, &m, &[0], label),
        -2.02938887129e6,
    );

    // compose is TensorKit `*` (mul!, no twist) while contract is
    // TensorKit `tensorcontract!`, which twists the dual contracted legs
    // (`tensoroperations.jl` blas_contract!): Julia verifies
    // `@tensor F[f; a b] * t[a b; k] == twist(F, 2) * t` exactly, so the
    // contract route must land on the tw[2] oracle instead.
    let fill = |key: &BlockKey, indices: &[usize]| -> f64 {
        let BlockKey::FusionTree(key) = key else {
            panic!("expected fusion-tree block keys");
        };
        let labels: Vec<f64> = key
            .codomain_uncoupled()
            .iter()
            .chain(key.domain_uncoupled())
            .map(|&sector| label(sector))
            .collect();
        oracle_entry(&labels, indices)
    };
    let fused = l.dual().fuse(&l).unwrap();
    let f = Tensor::isomorphism(&rt, Dtype::F64, [&fused], [&l.dual(), &l]).unwrap();
    let t = Tensor::from_block_fn(&rt, [&l.dual(), &l], [&m], fill).unwrap();
    let w = Tensor::from_block_fn(&rt, [&fused], [&m], fill).unwrap();
    let contracted = f.contract(&t, &[1, 2], &[0, 1]).unwrap();
    assert_rel(w.inner(&contracted).unwrap().re(), -2.01748090133e6);
}

#[test]
fn fuser_contraction_matches_tensorkit_su2() {
    use tenet::core::SU2Irrep;
    let rt = Runtime::builder().build().unwrap();
    let l = Space::su2([(0, 1), (1, 2)]).unwrap();
    let m = Space::su2([(0, 1), (1, 1), (2, 1)]).unwrap();
    let label = |sector: SectorId| SU2Irrep::from_sector_id(sector).twice_spin() as f64;
    let value = fuser_oracle_scalar(&rt, &l, &m, &[], label);
    assert_rel(value, 2.8621579710249998e7);
}

#[test]
fn id_is_the_identity_and_has_tensorkit_norm() {
    let rt = Runtime::builder().build().unwrap();
    let l = Space::u1([(0, 1), (1, 2)]);
    let id = Tensor::id(&rt, Dtype::F64, [&l.dual(), &l]).unwrap();
    // Julia: norm(id(dual(l) ⊗ l)) = 3.0.
    assert!((id.norm().unwrap() - 3.0).abs() < 1e-12);
    for seed in [3, 4] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&l.dual(), &l], [&l], seed).unwrap();
        assert_eq!(id.compose(&t).unwrap().data(), t.data());
    }
    // Identity is self-adjoint and idempotent.
    assert_eq!(id.adjoint().unwrap().data(), id.data());
    assert_eq!(id.compose(&id).unwrap().data(), id.data());
}

#[test]
fn fuser_roundtrips_to_identity_on_both_sides() {
    let rt = Runtime::builder().build().unwrap();
    for l in [
        Space::u1([(0, 1), (1, 2)]),
        Space::su2([(0, 1), (1, 2)]).unwrap(),
        Space::fz2([(0, 1), (1, 2)]).unwrap(),
    ] {
        let fused = l.dual().fuse(&l).unwrap();
        let f = Tensor::isomorphism(&rt, Dtype::F64, [&fused], [&l.dual(), &l]).unwrap();
        let product_id = Tensor::id(&rt, Dtype::F64, [&l.dual(), &l]).unwrap();
        let fused_id = Tensor::id(&rt, Dtype::F64, [&fused]).unwrap();
        assert_close(
            f.adjoint().unwrap().compose(&f).unwrap().data(),
            product_id.data(),
            1e-12,
        );
        assert_close(
            f.compose(&f.adjoint().unwrap()).unwrap().data(),
            fused_id.data(),
            1e-12,
        );
    }
}

#[test]
fn unitary_matches_isomorphism_and_rejects_non_isomorphic_spaces() {
    let rt = Runtime::builder().build().unwrap();
    let l = Space::u1([(0, 1), (1, 2)]);
    let fused = l.dual().fuse(&l).unwrap();
    let iso = Tensor::isomorphism(&rt, Dtype::F64, [&fused], [&l.dual(), &l]).unwrap();
    let uni = Tensor::unitary(&rt, Dtype::F64, [&fused], [&l.dual(), &l]).unwrap();
    assert_eq!(iso.data(), uni.data());

    let other = Space::u1([(0, 2), (1, 2)]);
    assert!(Tensor::isomorphism(&rt, Dtype::F64, [&other], [&l]).is_err());
    assert!(Tensor::unitary(&rt, Dtype::F64, [&other], [&l]).is_err());
}

#[test]
fn isometry_embeds_isometrically_and_rejects_too_small_codomains() {
    let rt = Runtime::builder().build().unwrap();
    let small = Space::su2([(0, 1), (1, 1)]).unwrap();
    let big = Space::su2([(0, 2), (1, 3), (2, 1)]).unwrap();
    let w = Tensor::isometry(&rt, Dtype::F64, [&big], [&small]).unwrap();
    // Julia: norm(W' * W - id(small)) = 0.0, norm(W) = sqrt(3).
    let id = Tensor::id(&rt, Dtype::F64, [&small]).unwrap();
    assert_eq!(w.adjoint().unwrap().compose(&w).unwrap().data(), id.data());
    assert!((w.norm().unwrap() - 3f64.sqrt()).abs() < 1e-12);
    assert!(Tensor::isometry(&rt, Dtype::F64, [&small], [&big]).is_err());
}

#[test]
fn twist_is_trivial_on_bosonic_legs_and_involutive_on_fermionic_ones() {
    let rt = Runtime::builder().build().unwrap();
    // Bosonic rules: θ = +1 everywhere, twist is the identity.
    for l in [
        Space::u1([(0, 1), (1, 2)]),
        Space::su2([(0, 1), (1, 2)]).unwrap(),
    ] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&l, &l], [&l], 5).unwrap();
        assert_eq!(t.twist(&[0, 1, 2]).unwrap().data(), t.data());
    }
    // Fermionic rule: θ(odd) = −1, twist² = id and odd blocks flip sign.
    let l = Space::fz2([(0, 1), (1, 2)]).unwrap();
    let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&l, &l], [&l], 6).unwrap();
    let twisted = t.twist(&[2]).unwrap();
    assert_ne!(twisted.data(), t.data());
    assert_eq!(twisted.twist(&[2]).unwrap().data(), t.data());
    assert!(t.twist(&[3]).is_err());
    assert_eq!(t.twist(&[]).unwrap().data(), t.data());
}

/// Two-value fz2 `[v] <- [v]` fixture: even block 2.0, odd block 3.0, with
/// each leg optionally dual, matching the TensorKit oracle tensors below.
fn fz2_two_block(rt: &Runtime, dual: bool) -> Tensor {
    let v = Space::fz2([(0, 1), (1, 1)]).unwrap();
    let v = if dual { v.dual() } else { v };
    Tensor::from_block_fn(rt, [&v], [&v], |key, _| match key {
        BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => 2.0,
        _ => 3.0,
    })
    .unwrap()
}

/// TensorKit 0.17.0 oracle for `flip` (Julia 1.11.6):
///
/// ```julia
/// using TensorKit
/// V = Vect[FermionParity](0 => 1, 1 => 1)
/// function mk(space)
///     t = zeros(Float64, space)
///     block(t, FermionParity(0)) .= 2.0
///     block(t, FermionParity(1)) .= 3.0
///     return t
/// end
/// show_blocks(name, t) = println(name, ": even=", block(t, FermionParity(0))[1, 1],
///     " odd=", block(t, FermionParity(1))[1, 1], "  space=", space(t))
/// t = mk(V ← V)
/// show_blocks("A  flip(t,1)", flip(t, 1))
/// show_blocks("A  flip(t,2)", flip(t, 2))
/// show_blocks("A  flip(t,(1,2))", flip(t, (1, 2)))
/// show_blocks("A  flip^2(t,2)", flip(flip(t, 2), 2))
/// show_blocks("A  flip^4(t,2)", flip(flip(flip(flip(t, 2), 2), 2), 2))
/// tb = mk(V' ← V')
/// show_blocks("B  flip(t,1)", flip(tb, 1))
/// show_blocks("B  flip(t,2)", flip(tb, 2))
/// W = Z2Space(0 => 1, 1 => 1)
/// tc = zeros(Float64, W ← W); block(tc, Z2Irrep(0)) .= 2.0; block(tc, Z2Irrep(1)) .= 3.0
/// println("C  flip(t,(1,2)): odd=", block(flip(tc, (1, 2)), Z2Irrep(1))[1, 1])
/// U = SU2Space(1 // 2 => 1)
/// td = zeros(Float64, U' ← U); block(td, SU2Irrep(1 // 2)) .= 5.0
/// println("D  flip(t,1): ", block(flip(td, 1), SU2Irrep(1 // 2))[1, 1])
/// println("D  flip(t,2): ", block(flip(td, 2), SU2Irrep(1 // 2))[1, 1])
/// ```
///
/// Output:
///
/// ```text
/// A  flip(t,1): even=2.0 odd=3.0   space=V' ← V
/// A  flip(t,2): even=2.0 odd=-3.0  space=V ← V'
/// A  flip(t,(1,2)): even=2.0 odd=-3.0  space=V' ← V'
/// A  flip^2(t,2): even=2.0 odd=-3.0  space=V ← V
/// A  flip^4(t,2): even=2.0 odd=3.0   space=V ← V
/// B  flip(t,1): even=2.0 odd=-3.0  space=V ← V'
/// B  flip(t,2): even=2.0 odd=3.0   space=V' ← V
/// C  flip(t,(1,2)): odd=3.0  (space flags toggle, values unchanged)
/// D  flip(t,1): -5.0
/// D  flip(t,2): 5.0
/// ```
#[test]
fn flip_matches_tensorkit_fermionic_oracle() {
    let rt = Runtime::builder().build().unwrap();

    // Case A: V ← V (both legs non-dual as written).
    let t = fz2_two_block(&rt, false);
    assert_eq!(t.data(), &[2.0, 3.0]);
    // Codomain leg, isdual = false: factor 1, only the space flag toggles.
    let f0 = t.flip(&[0]).unwrap();
    assert_eq!(f0.data(), &[2.0, 3.0]);
    assert!(f0.space(0).unwrap().is_dual());
    assert!(!t.space(0).unwrap().is_dual());
    // Domain leg, isdual(dom) = false: factor θ = −1 on the odd block.
    let f1 = t.flip(&[1]).unwrap();
    assert_eq!(f1.data(), &[2.0, -3.0]);
    // space(t, 1) is the outward dual view: dual before, non-dual after.
    assert!(t.space(1).unwrap().is_dual());
    assert!(!f1.space(1).unwrap().is_dual());
    // Both legs.
    assert_eq!(t.flip(&[0, 1]).unwrap().data(), &[2.0, -3.0]);
    // flip is not an involution: flip² returns to the original spaces but
    // scales the odd block by θ·χ̄ = −1; only flip⁴ = id.
    let f2 = f1.flip(&[1]).unwrap();
    assert_eq!(f2.data(), &[2.0, -3.0]);
    assert_eq!(f2.space(1).unwrap(), t.space(1).unwrap());
    let f4 = f2.flip(&[1]).unwrap().flip(&[1]).unwrap();
    assert_eq!(f4.data(), t.data());
    // A repeated leg in one call means "flip twice", sequentially.
    assert_eq!(t.flip(&[1, 1]).unwrap().data(), f2.data());

    // Case B: V' ← V' (both legs dual as written).
    let tb = fz2_two_block(&rt, true);
    // Codomain leg, isdual = true: factor χ·θ = −1 on the odd block.
    assert_eq!(tb.flip(&[0]).unwrap().data(), &[2.0, -3.0]);
    // Domain leg, isdual(dom) = true: factor χ = +1.
    assert_eq!(tb.flip(&[1]).unwrap().data(), &[2.0, 3.0]);

    // Out of range / empty.
    assert!(t.flip(&[2]).is_err());
    assert_eq!(t.flip(&[]).unwrap().data(), t.data());
}

/// Cases C and D of the oracle above: bosonic Z2 flip is purely structural,
/// while SU(2) j = 1/2 legs pick up the Frobenius-Schur phase χ = −1 on a
/// dual codomain leg (θ = +1: no sign from the domain side).
#[test]
fn flip_bosonic_is_structural_and_su2_carries_frobenius_schur_phase() {
    let rt = Runtime::builder().build().unwrap();

    let w = Space::z2([(0, 1), (1, 1)]);
    let tc = Tensor::rand_with_seed(&rt, Dtype::F64, [&w], [&w], 11).unwrap();
    let flipped = tc.flip(&[0, 1]).unwrap();
    assert_eq!(flipped.data(), tc.data());
    assert!(flipped.space(0).unwrap().is_dual());
    assert!(!flipped.space(1).unwrap().is_dual());

    let u = Space::su2([(1, 1)]).unwrap(); // j = 1/2
    let ud = u.dual();
    let td = Tensor::from_block_fn(&rt, [&ud], [&u], |_, _| 5.0).unwrap();
    assert_eq!(td.flip(&[0]).unwrap().data(), &[-5.0]);
    assert_eq!(td.flip(&[1]).unwrap().data(), &[5.0]);
}

/// `flip` moves both dtypes and composes with `twist` the way the legacy
/// `fliptwist_s` bond-orientation fix does: `twist(flip(s, [0, 1]), [0])`
/// on a fermionic diagonal `s` negates nothing twice (the two −1 factors
/// cancel on the odd block).
#[test]
fn flip_c64_and_fliptwist_composition() {
    let rt = Runtime::builder().build().unwrap();
    let t = fz2_two_block(&rt, false).to_c64();
    let flipped = t.flip(&[1]).unwrap();
    assert_eq!(flipped.data_c64()[1].re, -3.0);

    // fliptwist on s: V ← V. flip([0,1]) scales odd by θ = −1 (domain leg);
    // twist([0]) scales odd by θ = −1 again: values return to the original
    // while both legs are re-oriented.
    let s = fz2_two_block(&rt, false);
    let fixed = s.flip(&[0, 1]).unwrap().twist(&[0]).unwrap();
    assert_eq!(fixed.data(), s.data());
    assert!(fixed.space(0).unwrap().is_dual());
}

/// `sqrt` is the TensorKit `sqrt(::DiagonalTensorMap)` idiom: elementwise
/// on the diagonal of a `[v] <- [v]` bond tensor, `√S · √S == S`, and a
/// typed error on anything that is not a diagonal bond tensor. TensorKit
/// 0.17.0 oracle (Julia 1.11.6):
///
/// ```julia
/// using TensorKit
/// V = Vect[FermionParity](0 => 1, 1 => 1)
/// tt = randn(Float64, V ⊗ V ← V)
/// U2, S, Vh = svd_compact(tt)
/// println("S isa ", typeof(S))          # DiagonalTensorMap{Float64, …}
/// sq = sqrt(S)
/// println(sq.data .^ 2 ≈ S.data, " ", sq * sq ≈ S)   # true true
/// ```
#[test]
fn sqrt_splits_singular_values_and_rejects_non_diagonal_tensors() {
    let rt = Runtime::builder().build().unwrap();
    for v in [u1_space(), Space::fz2([(0, 2), (1, 2)]).unwrap()] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 13).unwrap();
        let s = t.svd_trunc(&Truncation::Full).unwrap().s;
        let sqrt_s = s.sqrt().unwrap();
        // √S · √S == S elementwise (both are diagonal on the same bond).
        assert_close(sqrt_s.compose(&sqrt_s).unwrap().data(), s.data(), 1e-13);
        // c64 branch agrees on nonnegative input.
        let sqrt_c = s.to_c64().sqrt().unwrap();
        for (a, b) in sqrt_c.data_c64().iter().zip(sqrt_s.data()) {
            assert!((a.re - b).abs() < 1e-15 && a.im == 0.0);
        }
        // Not a diagonal bond form: the original rank-(2,1) tensor.
        assert!(t.svd_trunc(&Truncation::Full).unwrap().u.sqrt().is_err());
        assert!(t.sqrt().is_err());
    }

    // Equal legs but dense block: off-diagonal entries are rejected.
    let v = Space::u1([(0, 2)]);
    let dense = Tensor::rand_with_seed(&rt, Dtype::F64, [&v], [&v], 17).unwrap();
    assert!(dense.sqrt().is_err());

    // Negative diagonal entries: error for f64, principal root for c64
    // (Julia: sqrt(-1.0) throws DomainError, sqrt(-1.0 + 0.0im) == im).
    let neg = Tensor::from_block_fn(&rt, [&v], [&v], |_, indices| {
        if indices[0] == indices[1] {
            -4.0
        } else {
            0.0
        }
    })
    .unwrap();
    assert!(neg.sqrt().is_err());
    let root = neg.to_c64().sqrt().unwrap();
    let diag = root.data_c64()[0];
    assert!(diag.re.abs() < 1e-15 && (diag.im - 2.0).abs() < 1e-15);
}

/// TeNeT issue #8: Space constructors enforce the TensorKit GradedSpace
/// sector-map invariant — zero-degeneracy sectors are dropped and duplicate
/// sector labels are rejected at construction, so introspection, dim(), and
/// the lowered SectorLeg can never disagree.
#[test]
fn space_drops_zero_degeneracy_sectors() {
    let v = Space::u1([(0, 0), (1, 1)]);
    assert_eq!(v.sectors(), vec![(SectorLabel::U1(1), 1)]);
    assert_eq!(v.degeneracy(SectorLabel::U1(0)), None);
    let w = Space::u1([(0, 1), (1, 1)]);
    let fused = w.fuse(&v).unwrap();
    assert_eq!(
        fused.sectors(),
        vec![(SectorLabel::U1(1), 1), (SectorLabel::U1(2), 1)]
    );
}

#[test]
fn space_zero_degeneracy_matches_omission_across_supported_constructors() {
    // What: every multiplicity-free user constructor inherits the same
    // zero-is-absent contract from the core leg representation.
    for (explicit_zero, omitted) in [
        (Space::u1([(-1, 0), (0, 2)]), Space::u1([(0, 2)])),
        (
            Space::su2([(2, 0), (0, 2)]).unwrap(),
            Space::su2([(0, 2)]).unwrap(),
        ),
        (
            Space::fz2([(1, 0), (0, 2)]).unwrap(),
            Space::fz2([(0, 2)]).unwrap(),
        ),
    ] {
        assert_eq!(explicit_zero, omitted);
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn product_space_zero_degeneracy_matches_omission() {
    // What: packed pair and triple product constructors share the core
    // zero-is-absent contract without product-specific normalization.
    assert_eq!(
        Space::product([((1, 1), 0), ((0, 0), 2)]).unwrap(),
        Space::product([((0, 0), 2)]).unwrap()
    );
    assert_eq!(
        Space::fz2_u1_su2([((1, 1, 1), 0), ((0, 0, 0), 2)]).unwrap(),
        Space::fz2_u1_su2([((0, 0, 0), 2)]).unwrap()
    );
}

#[test]
fn space_rejects_zero_and_positive_duplicate_sectors_in_both_orders() {
    // What: duplicate rejection is independent of whether a zero-degeneracy
    // declaration appears before or after the positive declaration.
    for sectors in [[(0, 0), (0, 2)], [(0, 2), (0, 0)]] {
        assert!(std::panic::catch_unwind(|| Space::u1(sectors)).is_err());
    }
}

#[test]
#[should_panic(expected = "appears multiple times")]
fn space_rejects_duplicate_sectors_same_degeneracy() {
    let _ = Space::u1([(0, 2), (0, 2)]);
}

#[test]
#[should_panic(expected = "appears multiple times")]
fn space_rejects_duplicate_sectors_conflicting_degeneracy() {
    let _ = Space::u1([(0, 2), (0, 3)]);
}

/// #56 item K: the abelian (UniqueFusion) `inner`/`norm` fast path. For an
/// abelian symmetry every `dim(c) == 1`, so the quantum-dimension-weighted
/// block sum equals the plain unweighted whole-buffer conjugated dot — exactly
/// what TensorKit's `inner(t.data, t.data)` / `norm(t.data)` UniqueFusion
/// specialization computes (vectorinterface.jl:124, linalg.jl:277). A
/// non-abelian symmetry (SU(2), sectors with `dim(c) > 1`) must NOT equal the
/// unweighted whole-buffer sum, pinning that it still applies the `dim(c)`
/// weights (i.e. did not fall into the abelian path).
#[test]
fn abelian_inner_is_unweighted_whole_buffer_dot() {
    let rt = Runtime::builder().build().unwrap();

    // Abelian (U(1), Z2, fermionic Z2): inner(self, self) == Σ x² over the
    // whole coupled buffer, and norm² agrees.
    for v in [
        Space::u1([(-1, 2), (0, 3), (1, 2)]),
        Space::z2([(0, 2), (1, 3)]),
        Space::fz2([(0, 2), (1, 2)]).unwrap(),
    ] {
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 11).unwrap();
        let whole: f64 = t.data().iter().map(|x| x * x).sum();
        let inner = t.inner(&t).unwrap().re();
        assert!(
            (inner - whole).abs() <= 1e-9 * (1.0 + whole.abs()),
            "abelian inner {inner} != whole-buffer dot {whole}"
        );
        assert!((t.norm().unwrap().powi(2) - whole).abs() <= 1e-8 * (1.0 + whole.abs()));
    }

    // Non-abelian SU(2) with dim>1 sectors: dim-weighted, so inner must differ
    // from the unweighted whole-buffer sum.
    let v = Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap();
    let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 11).unwrap();
    let whole: f64 = t.data().iter().map(|x| x * x).sum();
    let inner = t.inner(&t).unwrap().re();
    assert!(
        (inner - whole).abs() > 1e-6 * (1.0 + whole.abs()),
        "SU(2) inner {inner} unexpectedly equals unweighted whole-buffer sum {whole} \
         (dim(c) weights dropped?)"
    );

    // Complex abelian: inner is the conjugated dot Σ |z|², exactly real.
    let v = Space::u1([(0, 2), (1, 2)]);
    let t = Tensor::rand_with_seed(&rt, Dtype::C64, [&v, &v], [&v], 12).unwrap();
    let whole: f64 = t.data_c64().iter().map(|z| z.norm_sqr()).sum();
    let xx = t.inner(&t).unwrap();
    assert!(xx.im().abs() <= 1e-10 * (1.0 + xx.re().abs()));
    assert!((xx.re() - whole).abs() <= 1e-9 * (1.0 + whole.abs()));
}

/// #56 CTRG-lever follow-up: an identity `permute` (new codomain/domain equal
/// the current ones, in natural order) is a no-op and must return the tensor
/// unchanged without allocating/copying — TensorKit's `has_shared_permute`
/// short-circuit (indexmanipulations.jl:91). A genuine repartition still moves
/// data.
#[test]
fn identity_permute_is_a_noop() {
    let rt = Runtime::builder().build().unwrap();
    for v in [
        Space::u1([(-1, 2), (0, 2), (1, 2)]),
        Space::su2([(0, 2), (1, 2)]).unwrap(),
        Space::fz2([(0, 2), (1, 2)]).unwrap(),
    ] {
        // rank-3 tensor, codomain [v, v] (nout = 2), domain [v] (nin = 1).
        let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 21).unwrap();

        // Identity: codomain = [0, 1], domain = [2] → unchanged, bit-for-bit.
        let same = t.permute(&[0, 1], &[2]).unwrap();
        assert_eq!(same.codomain_spaces(), t.codomain_spaces());
        assert_eq!(same.domain_spaces(), t.domain_spaces());
        assert_eq!(same.data(), t.data(), "identity permute changed the data");

        // Genuine repartition (bend the domain leg up into the codomain) must
        // actually transform: the layout/length changes.
        let moved = t.permute(&[0, 1, 2], &[]).unwrap();
        assert_ne!(
            moved.codomain_spaces().len(),
            t.codomain_spaces().len(),
            "repartition should change the codomain rank"
        );
    }
}
