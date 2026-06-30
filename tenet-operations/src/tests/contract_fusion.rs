use super::*;

#[test]
fn tensorcontract_fusion_structure_enumerates_z2_compose_blocks_and_replays() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let lhs =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0, 3.0], lhs_space).unwrap();
    let rhs =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![5.0, 7.0], rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], dst_space).unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap();
    assert_eq!(
        specs,
        vec![
            TensorContractBlockSpec::new(0, 0, 0),
            TensorContractBlockSpec::new(1, 1, 1),
        ]
    );

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[50.0, 102.0]);
}

#[test]
fn tensorcontract_fusion_block_specs_enumerates_su2_innerline_blocks_from_homspace() {
    let rule = SU2FusionRule;
    let half = SectorId::new(1);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]),
        &rule,
        [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1], [1]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]),
        &rule,
        [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
    )
    .unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        &dst_space,
        &lhs_space,
        &rhs_space,
        TensorContractAxisSpec::canonical(&[3], &[0]),
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
    let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
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
    let dst_structure =
        BlockStructure::packed_column_major_with_keys(2, [(keys[0].clone(), vec![1, 1])]).unwrap();
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        dst_hom,
        dst_structure,
    )
    .unwrap();

    let err = tensorcontract_fusion_block_specs(
        &rule,
        &dst_space,
        &lhs_space,
        &rhs_space,
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: keys[1].clone().into()
        }
    );
}

#[test]
fn tensorcontract_fusion_block_specs_rejects_source_tree_transform_terms() {
    let rule = Z2FusionRule;
    let leg = |is_dual| SectorLeg::new([SectorId::new(0)], is_dual);
    let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false)]),
            FusionProductSpace::new([leg(false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let transformed_dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
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
        TensorContractAxisSpec::canonical(&[0], &[1]),
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
        TensorContractAxisSpec::new(&[1], &[0], AxisPermutation::from_axes(&[1, 0])),
    )
    .unwrap();

    assert_eq!(
        specs,
        vec![TensorContractBlockSpec::with_coefficient(0, 0, 0, 1.0)]
    );
}

#[test]
fn tensorcontract_fusion_into_rejects_source_tree_transform_terms() {
    let rule = Z2FusionRule;
    let leg = |is_dual| SectorLeg::new([SectorId::new(0)], is_dual);
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false)]),
            FusionProductSpace::new([leg(false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
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

    let err = tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[0], &[1]),
        3.0,
        11.0,
    )
    .unwrap_err();

    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );
}

#[test]
fn tensorcontract_fusion_output_recoupling_uses_su2_coefficients() {
    let rule = SU2FusionRule;
    let src_key = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let dst_key0 = src_key.clone();
    let dst_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let scalar_key = BlockKey::from(FusionTreeBlockKey::pair(
        empty_fusion_tree(),
        empty_fusion_tree(),
    ));
    let lhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1, 1], []),
        BlockStructure::packed_column_major_with_keys(4, [(src_key, vec![1, 1, 1, 1])]).unwrap(),
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        BlockStructure::packed_column_major_with_keys(0, [(scalar_key, vec![])]).unwrap(),
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1, 1], []),
        BlockStructure::packed_column_major_with_keys(
            4,
            [(dst_key0, vec![1, 1, 1, 1]), (dst_key1, vec![1, 1, 1, 1])],
        )
        .unwrap(),
    )
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
        TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3])),
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
        TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3])),
        2.0,
        3.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 53.0).abs() < 1.0e-12);
    assert!((dst.data()[1] - 92.602_540_378_443_86).abs() < 1.0e-12);
}

#[test]
fn tensorcontract_fusion_explicit_output_transform_materializes_canonical_dst() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1, 1], []);
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
    let lhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        lhs_hom.clone(),
        BlockStructure::packed_column_major_with_keys(4, [(src_key, vec![1, 1, 1, 1])]).unwrap(),
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        &rule,
        [vec![]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        lhs_hom,
        BlockStructure::packed_column_major_with_keys(
            4,
            [(dst_key0, vec![1, 1, 1, 1]), (dst_key1, vec![1, 1, 1, 1])],
        )
        .unwrap(),
    )
    .unwrap();
    let lhs_canonical_space = lhs_space.clone();
    let canonical_dst_space = lhs_space.clone();
    let rhs_canonical_space = rhs_space.clone();
    let lhs = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![10.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![5.0], rhs_space).unwrap();
    let mut expected_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let mut explicit_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let mut canonical_dst = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![999.0],
        canonical_dst_space.clone(),
    )
    .unwrap();
    let mut expected_canonical_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![-77.0], canonical_dst_space)
            .unwrap();
    let mut lhs_canonical = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![123.0],
        lhs_canonical_space.clone(),
    )
    .unwrap();
    let mut rhs_canonical = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(
        vec![456.0],
        rhs_canonical_space.clone(),
    )
    .unwrap();
    let mut expected_lhs_canonical =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![0.0], lhs_canonical_space).unwrap();
    let mut expected_rhs_canonical =
        TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![0.0], rhs_canonical_space).unwrap();
    let axes = TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3]));
    let plan = tensorcontract_fusion_explicit_plan(
        &rule,
        explicit_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(plan.canonical_dst_nout(), 4);
    assert_eq!(plan.canonical_dst_nin(), 0);
    assert_eq!(plan.canonical_axes().lhs_contracting_axes(), &[]);
    assert_eq!(plan.canonical_axes().rhs_contracting_axes(), &[]);
    assert_eq!(plan.canonical_axes().output_axes(), &[0, 1, 2, 3]);
    assert_eq!(
        plan.output_transform(),
        &TreeTransformOperationKey::permute([0, 2, 1, 3], Vec::<usize>::new())
    );

    let alpha = 2.0;
    let beta = 3.0;
    let err = tensorcontract_fusion_explicit_plan_into(
        &rule,
        &plan,
        &mut expected_dst,
        &mut expected_lhs_canonical,
        &mut expected_rhs_canonical,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message: EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
        }
    );

    tree_pair_transform_into(
        &rule,
        plan.lhs_transform().clone(),
        &mut expected_lhs_canonical,
        &lhs,
        1.0,
        0.0,
    )
    .unwrap();
    tree_pair_transform_into(
        &rule,
        plan.rhs_transform().clone(),
        &mut expected_rhs_canonical,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut expected_canonical_dst,
        &expected_lhs_canonical,
        &expected_rhs_canonical,
        plan.canonical_axes().as_spec(),
        alpha,
        0.0,
    )
    .unwrap();
    tree_pair_transform_into(
        &rule,
        plan.output_transform().clone(),
        &mut expected_dst,
        &expected_canonical_dst,
        1.0,
        beta,
    )
    .unwrap();

    tensorcontract_fusion_explicit_plan_into_canonical_dst(
        &rule,
        &plan,
        &mut explicit_dst,
        &mut canonical_dst,
        &mut lhs_canonical,
        &mut rhs_canonical,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(canonical_dst.data(), expected_canonical_dst.data());
    assert_eq!(canonical_dst.data(), &[100.0]);
    for (&actual, &expected) in explicit_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }
    assert!((explicit_dst.data()[0] - 53.0).abs() < 1.0e-12);
    assert!((explicit_dst.data()[1] - 92.602_540_378_443_86).abs() < 1.0e-12);
}

#[test]
fn tensorcontract_fusion_su2_keeps_contracted_tree_basis_with_degeneracy() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1], [1, 1, 1]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);
    let lhs_keys = lhs_hom.fusion_tree_keys(&rule);
    let rhs_keys = rhs_hom.fusion_tree_keys(&rule);
    assert_eq!(lhs_keys.len(), 2);
    assert_eq!(rhs_keys.len(), 2);
    assert_ne!(
        lhs_keys[0].domain_tree().innerlines()[0],
        lhs_keys[1].domain_tree().innerlines()[0]
    );
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let dst_hom = FusionTreeHomSpace::from_sector_ids([1], [1]);
    let dst_keys = dst_hom.fusion_tree_keys(&rule);
    assert_eq!(dst_keys.len(), 1);
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        BlockStructure::packed_column_major_with_keys(2, [(dst_keys[0].clone(), vec![2, 2])])
            .unwrap(),
    )
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
    let axes = TensorContractAxisSpec::canonical(&[1, 2, 3], &[0, 1, 2]);
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
            BlockKey::Dense => panic!("expected lhs fusion-tree block"),
        };
        let rhs_key = match rhs.structure().block(spec.rhs_block()).unwrap().key() {
            BlockKey::FusionTree(key) => key,
            BlockKey::Dense => panic!("expected rhs fusion-tree block"),
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
fn contracted_fusion_tree_basis_matches_dual_u1_labels_and_flags() {
    let rule = U1FusionRule;
    let plus_two = U1Irrep::new(2).sector_id();
    let minus_two = U1Irrep::new(-2).sector_id();
    let lhs_domain = FusionTreeKey::new(
        [plus_two],
        Some(plus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    let rhs_codomain = FusionTreeKey::new(
        [minus_two],
        Some(minus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    assert!(contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &rhs_codomain
    ));

    let raw_rhs_codomain = FusionTreeKey::new(
        [plus_two],
        Some(plus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    assert!(!contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &raw_rhs_codomain
    ));

    let dual_flag_rhs_codomain = FusionTreeKey::new(
        [minus_two],
        Some(minus_two),
        [true],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    assert!(!contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &dual_flag_rhs_codomain
    ));
}

#[test]
fn tensorcontract_fusion_noncanonical_su2_requires_explicit_transform_sequence() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([1], [1, 1, 1]);
    let axes = TensorContractAxisSpec::canonical(&[0, 1, 2], &[1, 2, 3]);
    let output_axes = [0, 1];
    let lhs_canonical_hom = lhs_hom
        .permute(&rule, &[3], &[0, 1, 2])
        .expect("valid lhs canonical tree-pair transform");
    let rhs_canonical_hom = rhs_hom
        .permute(&rule, &[1, 2, 3], &[0])
        .expect("valid rhs canonical tree-pair transform");
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

    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        lhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let lhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_canonical_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_canonical_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
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
    let lhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut direct_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst.clone(), dst_space.clone())
            .unwrap();
    let mut expected_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst, dst_space.clone()).unwrap();
    let mut lhs_canonical = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(
        vec![0.0; lhs_canonical_space.required_len().unwrap()],
        lhs_canonical_space.clone(),
    )
    .unwrap();
    let mut rhs_canonical = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(
        vec![0.0; rhs_canonical_space.required_len().unwrap()],
        rhs_canonical_space.clone(),
    )
    .unwrap();
    let plan = tensorcontract_fusion_explicit_plan(
        &rule,
        direct_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(
        plan.lhs_transform(),
        &TreeTransformOperationKey::permute([3], [0, 1, 2])
    );
    assert_eq!(
        plan.rhs_transform(),
        &TreeTransformOperationKey::permute([1, 2, 3], [0])
    );
    assert_eq!(plan.canonical_dst_nout(), 1);
    assert_eq!(plan.canonical_dst_nin(), 1);
    assert_eq!(plan.canonical_axes().lhs_contracting_axes(), &[1, 2, 3]);
    assert_eq!(plan.canonical_axes().rhs_contracting_axes(), &[0, 1, 2]);
    assert_eq!(plan.canonical_axes().output_axes(), &[0, 1]);
    assert_eq!(
        plan.output_transform(),
        &TreeTransformOperationKey::permute([0], [1])
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

    tree_pair_transform_into(
        &rule,
        TreeTransformOperationKey::permute([3], [0, 1, 2]),
        &mut lhs_canonical,
        &lhs,
        1.0,
        0.0,
    )
    .unwrap();
    tree_pair_transform_into(
        &rule,
        TreeTransformOperationKey::permute([1, 2, 3], [0]),
        &mut rhs_canonical,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    let canonical_specs = tensorcontract_fusion_block_specs(
        &rule,
        expected_dst.fusion_space().unwrap(),
        lhs_canonical.fusion_space().unwrap(),
        rhs_canonical.fusion_space().unwrap(),
        TensorContractAxisSpec::canonical(&[1, 2, 3], &[0, 1, 2]),
    )
    .unwrap();
    assert_eq!(canonical_specs.len(), 2);

    let alpha = -1.5;
    let beta = 0.25;
    let err = tensorcontract_fusion_into(&rule, &mut direct_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap_err();
    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );
    tensorcontract_fusion_into(
        &rule,
        &mut expected_dst,
        &lhs_canonical,
        &rhs_canonical,
        TensorContractAxisSpec::canonical(&[1, 2, 3], &[0, 1, 2]),
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(direct_dst.data(), &[2.0, -1.0, 4.0, -3.0]);
    assert_ne!(expected_dst.data(), direct_dst.data());

    let mut explicit_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst_for_explicit, dst_space)
            .unwrap();
    tensorcontract_fusion_explicit_plan_into(
        &rule,
        &plan,
        &mut explicit_dst,
        &mut lhs_canonical,
        &mut rhs_canonical,
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
}

#[test]
fn tensorcontract_fusion_product_noncanonical_requires_explicit_transform() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let scalar_key = BlockKey::from(FusionTreeBlockKey::pair(
        empty_fusion_tree(),
        empty_fusion_tree(),
    ));
    let rhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        BlockStructure::packed_column_major_with_keys(0, [(scalar_key, vec![])]).unwrap(),
    )
    .unwrap();
    let lhs = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
        src_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 0.5)],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)],
        dst_space,
    )
    .unwrap();
    let axes = TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[1, 0, 2]));
    let err = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
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

    let err = tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
    )
    .unwrap_err();

    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );
}
