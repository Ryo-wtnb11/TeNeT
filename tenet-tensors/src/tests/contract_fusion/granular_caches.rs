use super::*;

#[test]
fn tensorcontract_fusion_granular_caches_handle_block_structure_variants() {
    // What: per-layout cache counters describe one uninterrupted cache generation.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2), (1, 2), (1, 2)], [(1, 2)]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2)], [(1, 2), (1, 2), (1, 2)]);
    let axes = TensorContractSpec::with_default_output_order(&[0, 1, 2], &[1, 2, 3]);
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        &lhs_hom,
        &rhs_hom,
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        &[0, 1],
        1,
    )
    .unwrap();
    let lhs_keys = lhs_hom.fusion_tree_keys(&rule);
    let make_lhs_space = |case_index: usize| {
        let dense_space = TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap();
        let structure = match case_index {
            0 => packed_fixture_structure(
                4,
                lhs_keys.iter().cloned().map(|key| (key, vec![2, 2, 2, 2])),
            )
            .unwrap(),
            1 => {
                let mut blocks = lhs_keys
                    .iter()
                    .cloned()
                    .map(|key| (key, vec![2, 2, 2, 2]))
                    .collect::<Vec<_>>();
                blocks.reverse();
                packed_fixture_structure(4, blocks).unwrap()
            }
            2 => BlockStructure::from_blocks_with_rank(
                4,
                lhs_keys
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, key)| {
                        BlockSpec::with_key(
                            BlockKey::from(key),
                            vec![2, 2, 2, 2],
                            vec![1, 3, 6, 12],
                            23 * index,
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap(),
            _ => unreachable!("test only has three lhs block-structure cases"),
        };
        FusionTensorMapSpace::new_unbound(dense_space, lhs_hom.clone(), structure)
            .unwrap()
            .try_bind_rule(&rule)
            .unwrap()
    };
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let rhs_data = (0..32)
        .map(|index| -2.0 + 0.125 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = vec![0.5, -1.0, 2.0, -4.0];
    let alpha = 0.75;
    let beta = -0.25;
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    for case_index in 0..3 {
        let lhs_space = make_lhs_space(case_index);
        let lhs_data = (0..lhs_space.required_len().unwrap())
            .map(|index| 1.0 + 0.0625 * index as f64)
            .collect::<Vec<_>>();
        let lhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
        let rhs =
            TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(rhs_data.clone(), rhs_space.clone())
                .unwrap();
        let mut expected = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space.clone(),
        )
        .unwrap();
        fusion_contract_into(&rule, &mut expected, &lhs, &rhs, axes, alpha, beta).unwrap();

        let mut actual = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space.clone(),
        )
        .unwrap();
        if case_index == 2 {
            let mut profile = TensorContractFusionProfile::default();
            context
                .tensorcontract_fusion_into_profiled(
                    &rule,
                    &mut actual,
                    &lhs,
                    &rhs,
                    axes,
                    alpha,
                    beta,
                    &mut profile,
                )
                .unwrap();
            // What: profiling replays the ordinary artifact for padded expert storage.
            assert_eq!(profile.route, TensorContractFusionRoute::DynamicTreeCore);
        } else {
            context
                .tensorcontract_fusion_into(&rule, &mut actual, &lhs, &rhs, axes, alpha, beta)
                .unwrap();
        }
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (actual - expected).abs() < 1.0e-10,
                "actual {actual} expected {expected}"
            );
        }
    }
    let activity = crate::tree_transform::take_completed_transformer_activity();
    assert!(activity.builds + activity.hits >= 3);
}

#[test]
fn tensorcontract_fusion_granular_caches_handle_output_axes() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2), (1, 2), (1, 2)], [(1, 2)]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2)], [(1, 2), (1, 2), (1, 2)]);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        lhs_hom.clone(),
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom.clone(),
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let lhs_data = (0..32)
        .map(|index| 1.0 + 0.125 * index as f64)
        .collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| -3.0 + 0.25 * index as f64)
        .collect::<Vec<_>>();
    let lhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let initial_dst = vec![2.0, -1.0, 4.0, -3.0];
    let alpha = -1.5;
    let beta = 0.25;
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    for output_axes in [[0usize, 1usize], [1usize, 0usize]] {
        let axes = TensorContractSpec::new(
            &[0, 1, 2],
            &[1, 2, 3],
            OutputAxisOrder::from_axes(&output_axes),
        );
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            &rule,
            &lhs_hom,
            &rhs_hom,
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &output_axes,
            1,
        )
        .unwrap();
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
            dst_hom,
            &rule,
            [vec![2, 2]],
        )
        .unwrap();
        let mut expected = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space.clone(),
        )
        .unwrap();
        let mut actual =
            TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst.clone(), dst_space)
                .unwrap();

        fusion_contract_into(&rule, &mut expected, &lhs, &rhs, axes, alpha, beta).unwrap();
        context
            .tensorcontract_fusion_into(&rule, &mut actual, &lhs, &rhs, axes, alpha, beta)
            .unwrap();
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (actual - expected).abs() < 1.0e-10,
                "actual {actual} expected {expected}"
            );
        }
    }
    let activity = crate::tree_transform::take_completed_transformer_activity();
    assert!(activity.builds + activity.hits > 0);
}

#[test]
fn tensorcontract_fusion_granular_caches_distinguish_source_conjugation() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = SU2FusionRule;
    let mut context = TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();
    let alpha = Complex64::new(-1.5, 0.25);
    let beta = Complex64::new(0.25, -0.125);
    let initial_dst = vec![
        Complex64::new(2.0, -1.0),
        Complex64::new(-1.0, 0.5),
        Complex64::new(4.0, 2.0),
        Complex64::new(-3.0, -0.25),
    ];

    for (lhs_hom, rhs_hom, lhs_conjugate, rhs_conjugate) in [
        (
            su2_three_to_one_homspace(false, false),
            su2_one_to_three_homspace(false, false),
            false,
            false,
        ),
        (
            su2_three_to_one_homspace(false, false),
            su2_one_to_three_homspace(true, true),
            true,
            false,
        ),
    ] {
        let axes = TensorContractSpec::with_default_output_order_and_conjugation(
            &[0, 1, 2],
            &[1, 2, 3],
            lhs_conjugate,
            rhs_conjugate,
        );
        let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
            lhs_hom.clone(),
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
            rhs_hom.clone(),
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let lhs_adjoint_space =
            crate::lowering::adjoint_fusion_space_view(&rule, &lhs_space).unwrap();
        let rhs_adjoint_space =
            crate::lowering::adjoint_fusion_space_view(&rule, &rhs_space).unwrap();
        let effective_lhs_hom = if lhs_conjugate {
            lhs_adjoint_space.homspace()
        } else {
            &lhs_hom
        };
        let effective_rhs_hom = if rhs_conjugate {
            rhs_adjoint_space.homspace()
        } else {
            &rhs_hom
        };
        let lowered_lhs_axes = maybe_adjoint_axes::<3, 1>(&[0, 1, 2], lhs_conjugate);
        let lowered_rhs_axes = maybe_adjoint_axes::<1, 3>(&[1, 2, 3], rhs_conjugate);
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            &rule,
            effective_lhs_hom,
            effective_rhs_hom,
            lowered_lhs_axes.as_slice(),
            lowered_rhs_axes.as_slice(),
            &[0, 1],
            1,
        )
        .unwrap();
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
            dst_hom,
            &rule,
            [vec![2, 2]],
        )
        .unwrap();
        let lhs_data = (0..32)
            .map(|index| Complex64::new(1.0 + 0.125 * index as f64, -0.5 + 0.0625 * index as f64))
            .collect::<Vec<_>>();
        let rhs_data = (0..32)
            .map(|index| Complex64::new(-3.0 + 0.25 * index as f64, 0.75 - 0.03125 * index as f64))
            .collect::<Vec<_>>();
        let lhs =
            TensorMap::<Complex64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
        let rhs =
            TensorMap::<Complex64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
        let mut expected = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space.clone(),
        )
        .unwrap();
        let mut actual = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space,
        )
        .unwrap();

        fusion_contract_into(&rule, &mut expected, &lhs, &rhs, axes, alpha, beta).unwrap();
        context
            .tensorcontract_fusion_into(&rule, &mut actual, &lhs, &rhs, axes, alpha, beta)
            .unwrap();
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (actual - expected).norm() < 1.0e-10,
                "actual {actual} expected {expected}"
            );
        }
    }
}
