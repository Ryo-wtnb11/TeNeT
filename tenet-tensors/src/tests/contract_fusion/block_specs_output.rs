use super::*;

#[test]
fn tensorcontract_fusion_block_specs_enumerates_su2_innerline_blocks_from_homspace() {
    let rule = SU2FusionRule;
    let half = SectorId::new(1);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]),
        &rule,
        [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([(1, 1)], [(1, 1)]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]),
        &rule,
        [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
    )
    .unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        &dst_space,
        &lhs_space,
        &rhs_space,
        TensorContractSpec::with_default_output_order(&[3], &[0]),
    )
    .unwrap();

    assert_eq!(
        specs,
        vec![
            TensorContractBlockSpec::new(0, 0, 0),
            TensorContractBlockSpec::new(1, 1, 0),
        ]
    );
    assert_eq!(
        dst_space
            .homspace()
            .fusion_tree_keys_from_external_sectors(&rule, &[half, half, half, half])
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn tensorcontract_fusion_block_specs_rejects_missing_destination_subblock() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let keys = dst_hom.fusion_tree_keys(&rule);
    let dst_structure = packed_fixture_structure(2, [(keys[0].clone(), vec![1, 1])]).unwrap();
    let dst_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        dst_hom,
        dst_structure,
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();

    let err = tensorcontract_fusion_block_specs(
        &rule,
        &dst_space,
        &lhs_space,
        &rhs_space,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: Box::new(keys[1].clone().into())
        }
    );
}

#[test]
fn tensorcontract_fusion_block_specs_rejects_source_tree_transform_terms() {
    let rule = Z2FusionRule;
    let leg = |is_dual| SectorLeg::new([(SectorId::new(0), 1)], is_dual);
    let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false)]),
            FusionProductSpace::new([leg(false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let transformed_dst_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(true)]),
            FusionProductSpace::new([leg(true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();

    let err = tensorcontract_fusion_block_specs(
        &rule,
        &transformed_dst_space,
        &fusion_space,
        &fusion_space,
        TensorContractSpec::with_default_output_order(&[0], &[1]),
    )
    .unwrap_err();

    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        &transformed_dst_space,
        &fusion_space,
        &fusion_space,
        TensorContractSpec::new(&[1], &[0], OutputAxisOrder::from_axes(&[1, 0])),
    )
    .unwrap();

    assert_eq!(
        specs,
        vec![TensorContractBlockSpec::with_coefficient(0, 0, 0, 1.0)]
    );
}

#[test]
fn tensorcontract_fusion_into_absorbs_source_tree_transform_terms() {
    let rule = Z2FusionRule;
    let leg = |is_dual| SectorLeg::new([(SectorId::new(0), 1)], is_dual);
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false)]),
            FusionProductSpace::new([leg(false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(true)]),
            FusionProductSpace::new([leg(true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let lhs =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0], src_space.clone()).unwrap();
    let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![5.0], src_space).unwrap();
    let mut dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![7.0], dst_space).unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[0], &[1]),
        3.0,
        11.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[107.0]);
}

#[test]
fn tensorcontract_fusion_output_recoupling_uses_su2_coefficients() {
    let rule = SU2FusionRule;
    let src_key = all_codomain_fusion_tree_test_key_for_rule(
        &rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let dst_key0 = src_key.clone();
    let dst_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let scalar_key = BlockKey::from(FusionTreePairKey::pair(
        empty_fusion_tree(),
        empty_fusion_tree(),
    ));
    let lhs_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1), (1, 1)], []),
        packed_fixture_structure(4, [(src_key, vec![1, 1, 1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let rhs_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        packed_fixture_structure(0, [(scalar_key, vec![])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let dst_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1), (1, 1)], []),
        packed_fixture_structure(
            4,
            [(dst_key0, vec![1, 1, 1, 1]), (dst_key1, vec![1, 1, 1, 1])],
        )
        .unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let lhs = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![10.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![5.0], rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space).unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[0, 2, 1, 3])),
    )
    .unwrap();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].dst_block(), 0);
    assert_eq!(specs[1].dst_block(), 1);
    assert!((specs[0].coefficient() - 0.5).abs() < 1.0e-12);
    assert!((specs[1].coefficient() - 0.866_025_403_784_438_6).abs() < 1.0e-12);

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[0, 2, 1, 3])),
        2.0,
        3.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 53.0).abs() < 1.0e-12);
    assert!((dst.data()[1] - 92.602_540_378_443_86).abs() < 1.0e-12);
}

#[test]
fn tensorcontract_fusion_explicit_output_transform_materializes_core_dst() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1), (1, 1)], []);
    let lhs_keys = lhs_hom.fusion_tree_keys(&rule);
    assert_eq!(lhs_keys.len(), 2);
    let src_tree = lhs_keys
        .iter()
        .find(|key| key.codomain_tree().innerlines() == [SectorId::new(0), SectorId::new(1)])
        .expect("SU2 fixture should contain the reference source tree")
        .clone();
    let recoupled_tree = lhs_keys
        .iter()
        .find(|key| **key != src_tree)
        .expect("SU2 fixture should contain the recoupled output tree")
        .clone();
    let src_key = BlockKey::from(src_tree.clone());
    let dst_key0 = BlockKey::from(src_tree);
    let dst_key1 = BlockKey::from(recoupled_tree);
    let lhs_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        lhs_hom.clone(),
        packed_fixture_structure(4, [(src_key, vec![1, 1, 1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        &rule,
        [vec![]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        lhs_hom,
        packed_fixture_structure(
            4,
            [(dst_key0, vec![1, 1, 1, 1]), (dst_key1, vec![1, 1, 1, 1])],
        )
        .unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let lhs_core_space = lhs_space.clone();
    let core_dst_space = lhs_space.clone();
    let rhs_core_space = rhs_space.clone();
    let context_dst_space = dst_space.clone();
    let context_core_dst_space = core_dst_space.clone();
    let context_lhs_core_space = lhs_core_space.clone();
    let context_rhs_core_space = rhs_core_space.clone();
    let lhs = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![10.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![5.0], rhs_space).unwrap();
    let mut expected_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let mut explicit_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let mut core_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![999.0], core_dst_space.clone())
            .unwrap();
    let mut expected_core_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![-77.0], core_dst_space).unwrap();
    let mut lhs_core =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![123.0], lhs_core_space.clone())
            .unwrap();
    let mut rhs_core =
        TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![456.0], rhs_core_space.clone())
            .unwrap();
    let mut expected_lhs_core =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![0.0], lhs_core_space).unwrap();
    let mut expected_rhs_core =
        TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![0.0], rhs_core_space).unwrap();
    let axes = TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[0, 2, 1, 3]));
    let plan = prepare_tensorcontract_fusion_plan(
        &rule,
        explicit_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(plan.core_dst_open_lhs_rank(), 4);
    assert_eq!(plan.core_dst_open_rhs_rank(), 0);
    assert_eq!(plan.core_axes().lhs_contracting_axes(), &[] as &[usize]);
    assert_eq!(plan.core_axes().rhs_contracting_axes(), &[] as &[usize]);
    assert_eq!(plan.core_axes().output_axes(), &[0, 1, 2, 3]);
    assert_eq!(
        plan.output_transform(),
        &TreeTransformOperation::permute([0, 2, 1, 3], Vec::<usize>::new())
    );

    let alpha = 2.0;
    let beta = 3.0;
    let err = tensorcontract_fusion_prepared_into(
        &rule,
        &plan,
        &mut expected_dst,
        &mut expected_lhs_core,
        &mut expected_rhs_core,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message: EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CORE_DST,
        }
    );

    tree_transform_into(
        &rule,
        plan.lhs_transform().clone(),
        &mut expected_lhs_core,
        &lhs,
        1.0,
        0.0,
    )
    .unwrap();
    tree_transform_into(
        &rule,
        plan.rhs_transform().clone(),
        &mut expected_rhs_core,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut expected_core_dst,
        &expected_lhs_core,
        &expected_rhs_core,
        plan.core_axes().as_spec(),
        alpha,
        0.0,
    )
    .unwrap();
    tree_transform_into(
        &rule,
        plan.output_transform().clone(),
        &mut expected_dst,
        &expected_core_dst,
        1.0,
        beta,
    )
    .unwrap();

    tensorcontract_fusion_prepared_into_core_dst(
        &rule,
        &plan,
        &mut explicit_dst,
        &mut core_dst,
        &mut lhs_core,
        &mut rhs_core,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(core_dst.data(), expected_core_dst.data());
    assert_eq!(core_dst.data(), &[100.0]);
    for (&actual, &expected) in explicit_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }
    assert!((explicit_dst.data()[0] - 53.0).abs() < 1.0e-12);
    assert!((explicit_dst.data()[1] - 92.602_540_378_443_86).abs() < 1.0e-12);

    let mut automatic_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    tensorcontract_fusion_into(&rule, &mut automatic_dst, &lhs, &rhs, axes, alpha, beta).unwrap();
    for (&actual, &expected) in automatic_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }

    let mut context_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], context_dst_space)
            .unwrap();
    let mut context_core_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![999.0], context_core_dst_space)
            .unwrap();
    let mut context_lhs_core =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![123.0], context_lhs_core_space)
            .unwrap();
    let mut context_rhs_core =
        TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![456.0], context_rhs_core_space)
            .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    context
        .tensorcontract_fusion_prepared_into_core_dst(
            &rule,
            &plan,
            &mut context_dst,
            &mut context_core_dst,
            &mut context_lhs_core,
            &mut context_rhs_core,
            &lhs,
            &rhs,
            alpha,
            beta,
        )
        .unwrap();

    assert_eq!(context_core_dst.data(), expected_core_dst.data());
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }

    let mut automatic_context_dst = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![1.0, 2.0],
        expected_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut automatic_context =
        TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    crate::contract::reset_profiled_artifact_compile_phases();
    let mut profile = TensorContractFusionProfile::default();
    automatic_context
        .tensorcontract_fusion_into_profiled(
            &rule,
            &mut automatic_context_dst,
            &lhs,
            &rhs,
            axes,
            alpha,
            beta,
            &mut profile,
        )
        .unwrap();
    // What: a cold profiled dynamic artifact attributes each owning compile phase.
    assert_eq!(profile.route, TensorContractFusionRoute::DynamicTreeCore);
    assert_ne!(profile.resolution_preflight, std::time::Duration::ZERO);
    assert_ne!(profile.dynamic_tree_plan_build, std::time::Duration::ZERO);
    assert_ne!(profile.source_space_lookup, std::time::Duration::ZERO);
    assert_ne!(profile.core_dst_space_lookup, std::time::Duration::ZERO);
    assert_ne!(profile.core_block_plan_build, std::time::Duration::ZERO);
    assert_eq!(profile.prepared_plan, std::time::Duration::ZERO);
    assert_eq!(profile.dense_contract, std::time::Duration::ZERO);
    assert_eq!(
        crate::contract::profiled_artifact_compile_phases(),
        (true, true, true)
    );
    for (&actual, &expected) in automatic_context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }

    let mut beta_only_dst = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![7.0, 11.0],
        automatic_context_dst
            .fusion_space()
            .unwrap()
            .as_ref()
            .clone(),
    )
    .unwrap();
    automatic_context
        .tensorcontract_fusion_into(&rule, &mut beta_only_dst, &lhs, &rhs, axes, 0.0, 3.0)
        .unwrap();
    assert_eq!(beta_only_dst.data(), &[21.0, 33.0]);
}

#[test]
fn tensorcontract_fusion_su2_keeps_contracted_tree_basis_with_degeneracy() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2)], [(1, 2), (1, 2), (1, 2)]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2), (1, 2), (1, 2)], [(1, 2)]);
    let lhs_keys = lhs_hom.fusion_tree_keys(&rule);
    let rhs_keys = rhs_hom.fusion_tree_keys(&rule);
    assert_eq!(lhs_keys.len(), 2);
    assert_eq!(rhs_keys.len(), 2);
    assert_ne!(
        lhs_keys[0].domain_tree().innerlines()[0],
        lhs_keys[1].domain_tree().innerlines()[0]
    );
    let packed = |hom: &FusionTreeHomSpace| {
        packed_fixture_structure(
            4,
            hom.fusion_tree_keys(&rule)
                .iter()
                .cloned()
                .map(|key| (key, vec![2usize; 4])),
        )
        .unwrap()
    };
    let lhs_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_hom.clone(),
        packed(&lhs_hom),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let rhs_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_hom.clone(),
        packed(&rhs_hom),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let dst_hom = FusionTreeHomSpace::from_sector_ids([(1, 2)], [(1, 2)]);
    let dst_keys = dst_hom.fusion_tree_keys(&rule);
    assert_eq!(dst_keys.len(), 1);
    let dst_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        packed_fixture_structure(2, [(dst_keys[0].clone(), vec![2, 2])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let lhs_data = (0..32).map(|index| 0.25 + index as f64).collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| 10.0 - 0.5 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = vec![1.0, -2.0, 3.0, -4.0];
    let lhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst.clone(), dst_space).unwrap();
    let axes = TensorContractSpec::with_default_output_order(&[1, 2, 3], &[0, 1, 2]);
    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();

    assert_eq!(specs.len(), 2);
    for spec in &specs {
        let lhs_key = match lhs.structure().block(spec.lhs_block()).unwrap().key() {
            BlockKey::FusionTree(key) => key,
            _ => panic!("expected lhs fusion-tree block"),
        };
        let rhs_key = match rhs.structure().block(spec.rhs_block()).unwrap().key() {
            BlockKey::FusionTree(key) => key,
            _ => panic!("expected rhs fusion-tree block"),
        };
        assert_eq!(
            lhs_key.domain_tree().innerlines()[0],
            rhs_key.codomain_tree().innerlines()[0],
            "contracted SU2 tree basis must not cross-contract"
        );
    }

    let alpha = 1.25;
    let beta = -0.5;
    let mut expected = initial_dst
        .into_iter()
        .map(|value| beta * value)
        .collect::<Vec<_>>();
    for spec in &specs {
        let lhs_offset = lhs.structure().block(spec.lhs_block()).unwrap().offset();
        let rhs_offset = rhs.structure().block(spec.rhs_block()).unwrap().offset();
        for lhs_open in 0..2 {
            for rhs_open in 0..2 {
                let mut sum = 0.0;
                for a in 0..2 {
                    for b in 0..2 {
                        for c in 0..2 {
                            let lhs_index = lhs_offset + lhs_open + 2 * a + 4 * b + 8 * c;
                            let rhs_index = rhs_offset + a + 2 * b + 4 * c + 8 * rhs_open;
                            sum += lhs.data()[lhs_index] * rhs.data()[rhs_index];
                        }
                    }
                }
                expected[lhs_open + 2 * rhs_open] += alpha * spec.coefficient() * sum;
            }
        }
    }

    tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();

    for (&actual, expected) in dst.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
}

#[test]
fn contracted_fusion_tree_basis_requires_exact_stored_tree() {
    let rule = U1FusionRule;
    let plus_two = U1Irrep::new(2).sector_id();
    let minus_two = U1Irrep::new(-2).sector_id();
    let lhs_domain = FusionTreeKey::try_new_for_rule(
        &rule,
        [plus_two],
        plus_two,
        [false],
        Vec::<SectorId>::new(),
        Vec::<MultiplicityIndex>::new(),
    )
    .unwrap();
    let dual_label = FusionTreeKey::try_new_for_rule(
        &rule,
        [minus_two],
        minus_two,
        [false],
        Vec::<SectorId>::new(),
        Vec::<MultiplicityIndex>::new(),
    )
    .unwrap();
    assert!(!contracted_fusion_tree_basis_matches(
        &lhs_domain,
        &dual_label
    ));

    let exact = FusionTreeKey::try_new_for_rule(
        &rule,
        [plus_two],
        plus_two,
        [false],
        Vec::<SectorId>::new(),
        Vec::<MultiplicityIndex>::new(),
    )
    .unwrap();
    assert!(contracted_fusion_tree_basis_matches(&lhs_domain, &exact));

    let dual_flag_rhs_codomain = FusionTreeKey::try_new_for_rule(
        &rule,
        [plus_two],
        plus_two,
        [true],
        Vec::<SectorId>::new(),
        Vec::<MultiplicityIndex>::new(),
    )
    .unwrap();
    assert!(!contracted_fusion_tree_basis_matches(
        &lhs_domain,
        &dual_flag_rhs_codomain
    ));
}

#[test]
fn tensorcontract_fusion_non_core_form_su2_absorbs_explicit_transform_sequence() {
    // What: exact cache counters describe one uninterrupted cache generation.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2), (1, 2), (1, 2)], [(1, 2)]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2)], [(1, 2), (1, 2), (1, 2)]);
    let axes = TensorContractSpec::with_default_output_order(&[0, 1, 2], &[1, 2, 3]);
    let output_axes = [0, 1];
    let lhs_core_hom = lhs_hom
        .permute(&rule, &[3], &[0, 1, 2])
        .expect("valid lhs core tree-pair transform");
    let rhs_core_hom = rhs_hom
        .permute(&rule, &[1, 2, 3], &[0])
        .expect("valid rhs core tree-pair transform");
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

    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        lhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let lhs_core_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_core_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_core_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_core_hom,
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
    let lhs_data = (0..32)
        .map(|index| 1.0 + 0.125 * index as f64)
        .collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| -3.0 + 0.25 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = vec![2.0, -1.0, 4.0, -3.0];
    let initial_dst_for_explicit = initial_dst.clone();
    let initial_dst_for_context = initial_dst.clone();
    let initial_dst_for_context_replay = initial_dst.clone();
    let lhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut direct_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst.clone(), dst_space.clone())
            .unwrap();
    let mut expected_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst, dst_space.clone()).unwrap();
    let mut lhs_core = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(
        vec![0.0; lhs_core_space.required_len().unwrap()],
        lhs_core_space.clone(),
    )
    .unwrap();
    let mut rhs_core = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(
        vec![0.0; rhs_core_space.required_len().unwrap()],
        rhs_core_space.clone(),
    )
    .unwrap();
    let plan = prepare_tensorcontract_fusion_plan(
        &rule,
        direct_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(
        plan.lhs_transform(),
        &TreeTransformOperation::permute([3], [0, 1, 2])
    );
    assert_eq!(
        plan.rhs_transform(),
        &TreeTransformOperation::permute([1, 2, 3], [0])
    );
    assert_eq!(plan.core_dst_open_lhs_rank(), 1);
    assert_eq!(plan.core_dst_open_rhs_rank(), 1);
    assert_eq!(plan.core_axes().lhs_contracting_axes(), &[1, 2, 3]);
    assert_eq!(plan.core_axes().rhs_contracting_axes(), &[0, 1, 2]);
    assert_eq!(plan.core_axes().output_axes(), &[0, 1]);
    assert_eq!(
        plan.output_transform(),
        &TreeTransformOperation::permute([0], [1])
    );

    let err = tensorcontract_fusion_block_specs(
        &rule,
        direct_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap_err();
    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );

    tree_transform_into(
        &rule,
        TreeTransformOperation::permute([3], [0, 1, 2]),
        &mut lhs_core,
        &lhs,
        1.0,
        0.0,
    )
    .unwrap();
    tree_transform_into(
        &rule,
        TreeTransformOperation::permute([1, 2, 3], [0]),
        &mut rhs_core,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    let core_specs = tensorcontract_fusion_block_specs(
        &rule,
        expected_dst.fusion_space().unwrap(),
        lhs_core.fusion_space().unwrap(),
        rhs_core.fusion_space().unwrap(),
        TensorContractSpec::with_default_output_order(&[1, 2, 3], &[0, 1, 2]),
    )
    .unwrap();
    assert_eq!(core_specs.len(), 2);

    let alpha = -1.5;
    let beta = 0.25;
    tensorcontract_fusion_into(&rule, &mut direct_dst, &lhs, &rhs, axes, alpha, beta).unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut expected_dst,
        &lhs_core,
        &rhs_core,
        TensorContractSpec::with_default_output_order(&[1, 2, 3], &[0, 1, 2]),
        alpha,
        beta,
    )
    .unwrap();

    for (&actual, &expected) in direct_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }

    let mut explicit_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_explicit,
        dst_space.clone(),
    )
    .unwrap();
    tensorcontract_fusion_prepared_into(
        &rule,
        &plan,
        &mut explicit_dst,
        &mut lhs_core,
        &mut rhs_core,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();
    for (&actual, &expected) in explicit_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }

    let mut context_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst_for_context, dst_space)
            .unwrap();
    let mut context_lhs_core = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(
        vec![0.0; lhs_core_space.required_len().unwrap()],
        lhs_core_space,
    )
    .unwrap();
    let mut context_rhs_core = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(
        vec![0.0; rhs_core_space.required_len().unwrap()],
        rhs_core_space,
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    context
        .tensorcontract_fusion_prepared_into(
            &rule,
            &plan,
            &mut context_dst,
            &mut context_lhs_core,
            &mut context_rhs_core,
            &lhs,
            &rhs,
            alpha,
            beta,
        )
        .unwrap();
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    assert_eq!(context.tree_context().cache().structure_len(), 2);

    let pinned = context
        .prepare_tensorcontract_fusion(&rule, &context_dst, &lhs, &rhs, axes)
        .unwrap();
    context.set_cache_policy(OperationCachePolicy::NoCache);
    for _ in 0..2 {
        context_dst
            .data_mut()
            .copy_from_slice(&initial_dst_for_context_replay);
        context
            .execute_prepared_tensorcontract_fusion(
                &pinned,
                &rule,
                &mut context_dst,
                &lhs,
                &rhs,
                alpha,
                beta,
            )
            .unwrap();
        for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        // What: a prepared dynamic contraction owns the complete artifact;
        // replay does not consult or populate execution-context caches.
        assert_eq!(context.dynamic_fusion_space_cache_len(), 0);
    }
    context.set_cache_policy(OperationCachePolicy::TaskLocal);

    context_dst
        .data_mut()
        .copy_from_slice(&initial_dst_for_context_replay);
    context
        .tensorcontract_fusion_prepared_into(
            &rule,
            &plan,
            &mut context_dst,
            &mut context_lhs_core,
            &mut context_rhs_core,
            &lhs,
            &rhs,
            alpha,
            beta,
        )
        .unwrap();
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    assert_eq!(context.tree_context().cache().structure_len(), 2);

    let mut automatic_context_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context_replay.clone(),
        context_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut automatic_context =
        TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    automatic_context
        .tensorcontract_fusion_into(
            &rule,
            &mut automatic_context_dst,
            &lhs,
            &rhs,
            axes,
            alpha,
            beta,
        )
        .unwrap();
    for (&actual, &expected) in automatic_context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    assert!(automatic_context.tree_context().cache().structure_len() > 0);
    assert!(automatic_context.dynamic_fusion_space_cache_len() > 0);
    assert!(automatic_context.dynamic_fusion_space_cache_misses() > 0);
    assert_eq!(automatic_context.dynamic_fusion_space_cache_hits(), 0);
    assert!(
        automatic_context
            .tree_context()
            .cache()
            .stats()
            .structure_misses()
            > 0
    );
    let cached_contract_bits = automatic_context_dst.data().to_vec();

    let mut no_cache_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context_replay.clone(),
        context_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut no_cache_context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    no_cache_context.set_cache_policy(OperationCachePolicy::NoCache);
    let mut previous_dynamic_misses = 0;
    for _ in 0..2 {
        no_cache_context
            .tensorcontract_fusion_into(&rule, &mut no_cache_dst, &lhs, &rhs, axes, alpha, beta)
            .unwrap();
        for (&actual, &expected) in no_cache_dst.data().iter().zip(expected_dst.data()) {
            assert!(
                (actual - expected).abs() < 1.0e-10,
                "actual {actual} expected {expected}"
            );
        }
        assert_eq!(no_cache_context.tree_context().cache().structure_len(), 0);
        assert_eq!(no_cache_context.dynamic_fusion_space_cache_len(), 0);
        assert_eq!(no_cache_context.dynamic_fusion_space_cache_hits(), 0);
        assert_eq!(no_cache_context.dynamic_fusion_space_cache_fast_hits(), 0);
        let dynamic_misses = no_cache_context.dynamic_fusion_space_cache_misses();
        assert!(dynamic_misses > previous_dynamic_misses);
        previous_dynamic_misses = dynamic_misses;
        // What: disabling all execution caches changes reuse only, not the
        // destination reduced-block values or floating-point operation order.
        assert_f64_bits_eq(
            "cached vs NoCache SU2 non-core contraction",
            no_cache_dst.data(),
            &cached_contract_bits,
        );
        no_cache_dst
            .data_mut()
            .copy_from_slice(&initial_dst_for_context_replay);
    }

    let mut warm_policy_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context_replay.clone(),
        context_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut warm_policy_context =
        TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    warm_policy_context
        .tensorcontract_fusion_into(&rule, &mut warm_policy_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
    assert!(warm_policy_context.dynamic_fusion_space_cache_len() > 1);
    warm_policy_context.set_cache_policy(OperationCachePolicy::task_local_lru(1));
    assert!(warm_policy_context.tree_context().cache().structure_len() <= 1);
    assert!(warm_policy_context.dynamic_fusion_space_cache_len() <= 1);

    let mut lru_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context_replay.clone(),
        context_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut lru_context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    lru_context.set_cache_policy(OperationCachePolicy::task_local_lru(1));
    lru_context
        .tensorcontract_fusion_into(&rule, &mut lru_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
    for (&actual, &expected) in lru_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    assert!(lru_context.tree_context().cache().structure_len() <= 1);
    assert!(lru_context.dynamic_fusion_space_cache_len() <= 1);

    let mut split_backend_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context_replay.clone(),
        context_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut tree_backend = HostTensorOperations;
    let mut tree_workspace = TreeTransformWorkspace::default();
    let mut contract_backend = DenseTreeTransformOperations::default_executor();
    let mut contract_workspace = TensorContractWorkspace::default();
    tensorcontract_fusion_into_with_backends(
        &mut tree_backend,
        &mut tree_workspace,
        &mut contract_backend,
        &mut contract_workspace,
        &rule,
        &mut split_backend_dst,
        &lhs,
        &rhs,
        axes,
        alpha,
        beta,
    )
    .unwrap();
    for (&actual, &expected) in split_backend_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }

    let tree_stats_after_first = automatic_context.tree_context().cache().stats();
    automatic_context_dst
        .data_mut()
        .copy_from_slice(&initial_dst_for_context_replay);
    automatic_context
        .tensorcontract_fusion_into(
            &rule,
            &mut automatic_context_dst,
            &lhs,
            &rhs,
            axes,
            alpha,
            beta,
        )
        .unwrap();
    for (&actual, &expected) in automatic_context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    assert_eq!(
        automatic_context.tree_context().cache().stats(),
        tree_stats_after_first
    );
    assert!(automatic_context.dynamic_fusion_space_cache_hits() > 0);
    assert!(automatic_context.dynamic_fusion_space_cache_fast_hits() > 0);

    automatic_context_dst
        .data_mut()
        .copy_from_slice(&initial_dst_for_context_replay);
    let artifact_hits_before_profile = automatic_context.dynamic_fusion_space_cache_hits();
    let artifact_fast_hits_before_profile =
        automatic_context.dynamic_fusion_space_cache_fast_hits();
    let mut profile = TensorContractFusionProfile::default();
    automatic_context
        .tensorcontract_fusion_into_profiled(
            &rule,
            &mut automatic_context_dst,
            &lhs,
            &rhs,
            axes,
            alpha,
            beta,
            &mut profile,
        )
        .unwrap();
    // What: profiling compiles the eager execution artifact while reusing its
    // two source transforms and core destination.
    assert_eq!(
        automatic_context.dynamic_fusion_space_cache_hits(),
        artifact_hits_before_profile + 3
    );
    assert_eq!(
        automatic_context.dynamic_fusion_space_cache_fast_hits(),
        artifact_fast_hits_before_profile + 3
    );
    for (&actual, &expected) in automatic_context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    assert_eq!(profile.route, TensorContractFusionRoute::DynamicTreeCore);
    assert_eq!(
        automatic_context.last_resolution_orientation(),
        Some(crate::contract::FusionContractOrientation::RhsLhs)
    );
    assert_eq!(profile.lhs_transform_calls, 0);
    assert_eq!(profile.rhs_transform_calls, 0);
    assert_eq!(profile.output_transform_calls, 1);
    assert!(profile.core_contract_groups > 0);
    assert_eq!(profile.tree_replay.cache_lookup.as_nanos(), 0);
    assert_eq!(profile.tree_replay.strided_view_setup.as_nanos(), 0);
    assert_eq!(profile.tree_replay.multi_dense_view_setup.as_nanos(), 0);
    assert_eq!(profile.tree_replay.multi_dense_matmul_call.as_nanos(), 0);
    assert_eq!(
        profile.tree_replay.multi_matmul_total,
        profile.tree_replay.multi_dense_view_setup
            + profile.tree_replay.multi_dense_matmul_call
            + profile.tree_replay.multi_scalar_recoupling
    );
    assert_eq!(profile.tree_replay.multi_blocks, 0);
    assert_eq!(profile.tree_replay.packed_columns, 0);
    assert_eq!(profile.tree_replay.scattered_columns, 0);
}
