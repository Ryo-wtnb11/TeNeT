use super::*;

#[test]
fn forced_axis_order_candidates_have_identical_u1_result() {
    // What: both paired caller orders and both operand orientations produce
    // the same U1 contraction result.
    let rule = U1FusionRule;
    let sectors = [
        U1Irrep::new(-1).sector_id(),
        U1Irrep::new(0).sector_id(),
        U1Irrep::new(1).sector_id(),
    ];
    let leg = || SectorLeg::new(sectors.map(|sector| (sector, 1)), false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let dense = TensorMapSpace::<2, 2>::from_dims([3, 3], [3, 3]).unwrap();
    let blocks = hom.fusion_tree_keys(&rule).len();
    let space =
        FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, &rule, vec![vec![1; 4]; blocks])
            .unwrap();
    let len = space.subblock_structure().required_len().unwrap();
    let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..len).map(|i| i as f64 + 1.0).collect(),
        space.clone(),
    )
    .unwrap();
    let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..len).map(|i| 2.0 * i as f64 - 0.5).collect(),
        space.clone(),
    )
    .unwrap();
    let candidates = crate::contract::contracted_axis_order_candidates(&[3, 2], &[0, 1]);
    assert!(candidates.len() >= 2);
    let dst_dyn = DynamicFusionMapSpace::from_typed(&space);
    let lhs_dyn = DynamicFusionMapSpace::from_typed(&space);
    let rhs_dyn = DynamicFusionMapSpace::from_typed(&space);
    let mut outputs = Vec::new();
    for candidate in candidates.iter().take(2) {
        for orientation in [
            crate::contract::FusionContractOrientation::LhsRhs,
            crate::contract::FusionContractOrientation::RhsLhs,
        ] {
            let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
                &rule,
                &dst_dyn,
                &lhs_dyn,
                &rhs_dyn,
                TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
                candidate,
                orientation,
            )
            .unwrap();
            let mut dst =
                TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; len], space.clone())
                    .unwrap();
            crate::contract::execute_dynamic_tree_execution_artifact_for_test(
                &rule, &plan, &mut dst, &lhs, &rhs, 1.0, 0.0,
            )
            .unwrap();
            outputs.push(dst);
        }
    }
    assert_eq!(outputs.len(), 4);
    for output in &outputs[1..] {
        assert_eq!(outputs[0].fusion_space(), output.fusion_space());
        for (a, b) in outputs[0].data().iter().zip(output.data()) {
            assert!((a - b).abs() < 1.0e-10, "candidate mismatch: {a} vs {b}");
        }
    }
    let selected = prepare_tensorcontract_fusion_plan(
        &rule,
        &space,
        &space,
        &space,
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
    )
    .unwrap();
    assert_eq!(
        selected.orientation(),
        crate::contract::FusionContractOrientation::LhsRhs
    );
    assert_eq!(
        selected.lhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.lhs_transform().domain_permutation(), &[2, 3]);
    assert_eq!(
        selected.rhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.rhs_transform().codomain_permutation(), &[1, 0]);
    let mut rhs_permuted =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; len], space.clone()).unwrap();
    tree_transform_into(
        &rule,
        TreeTransformOperation::permute([1, 0], [2, 3]),
        &mut rhs_permuted,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    let mut oracle =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; len], space.clone()).unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut oracle,
        &lhs,
        &rhs_permuted,
        TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]),
        1.0,
        0.0,
    )
    .unwrap();
    let mut normal =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; len], space).unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut normal,
        &lhs,
        &rhs,
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        1.0,
        0.0,
    )
    .unwrap();
    // What: compiling the selected forward orientation preserves its eager bytes.
    assert_eq!(normal.data(), outputs[0].data());
    for ((actual, expected), fixed) in normal
        .data()
        .iter()
        .zip(oracle.data())
        .zip(outputs[0].data())
    {
        assert!((*actual - *expected).abs() < 1.0e-10);
        assert!((*actual - *fixed).abs() < 1.0e-10);
    }
}

#[test]
fn forced_orientations_match_asymmetric_u1_reduced_block_oracle() {
    let rule = U1FusionRule;
    let neutral = U1Irrep::new(0).sector_id();
    let leg = |dimension| SectorLeg::new([(neutral, dimension)], false);
    let lhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(2), leg(3)]),
        FusionProductSpace::new([leg(4), leg(5)]),
    );
    let rhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(5), leg(4)]),
        FusionProductSpace::new([leg(6), leg(7)]),
    );
    let output_axes = [1, 0, 3, 2];
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        &lhs_hom,
        &rhs_hom,
        &[3, 2],
        &[0, 1],
        &output_axes,
        2,
    )
    .unwrap();
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([2, 3], [4, 5]).unwrap(),
        lhs_hom,
        &rule,
        [vec![2, 3, 4, 5]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([5, 4], [6, 7]).unwrap(),
        rhs_hom,
        &rule,
        [vec![5, 4, 6, 7]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([3, 2], [7, 6]).unwrap(),
        dst_hom,
        &rule,
        [vec![3, 2, 7, 6]],
    )
    .unwrap();
    let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_space.required_len().unwrap())
            .map(|index| index as f64 * 0.125 - 2.0)
            .collect(),
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..rhs_space.required_len().unwrap())
            .map(|index| 1.5 - index as f64 * 0.0625)
            .collect(),
        rhs_space,
    )
    .unwrap();
    let initial = (0..dst_space.required_len().unwrap())
        .map(|index| index as f64 * 0.25 + 0.5)
        .collect::<Vec<_>>();
    let alpha = 1.25;
    let beta = -0.75;
    let lhs_block = lhs.structure().block(0).unwrap();
    let rhs_block = rhs.structure().block(0).unwrap();
    let dst_block = dst_space.subblock_structure().block(0).unwrap();
    let mut expected = initial.iter().map(|value| beta * value).collect::<Vec<_>>();
    for l0 in 0..2 {
        for l1 in 0..3 {
            for r2 in 0..6 {
                for r3 in 0..7 {
                    let mut sum = 0.0;
                    for c0 in 0..4 {
                        for c1 in 0..5 {
                            let lhs_offset = lhs_block.offset()
                                + [l0, l1, c0, c1]
                                    .iter()
                                    .zip(lhs_block.strides())
                                    .map(|(&index, &stride)| index * stride)
                                    .sum::<usize>();
                            let rhs_offset = rhs_block.offset()
                                + [c1, c0, r2, r3]
                                    .iter()
                                    .zip(rhs_block.strides())
                                    .map(|(&index, &stride)| index * stride)
                                    .sum::<usize>();
                            sum += lhs.data()[lhs_offset] * rhs.data()[rhs_offset];
                        }
                    }
                    let dst_offset = dst_block.offset()
                        + [l1, l0, r3, r2]
                            .iter()
                            .zip(dst_block.strides())
                            .map(|(&index, &stride)| index * stride)
                            .sum::<usize>();
                    expected[dst_offset] += alpha * sum;
                }
            }
        }
    }

    let axes = TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&output_axes));
    let candidates = crate::contract::contracted_axis_order_candidates(&[3, 2], &[0, 1]);
    let dst_dyn = DynamicFusionMapSpace::from_typed(&dst_space);
    let lhs_dyn = DynamicFusionMapSpace::from_typed(lhs.fusion_space().unwrap());
    let rhs_dyn = DynamicFusionMapSpace::from_typed(rhs.fusion_space().unwrap());
    for candidate in candidates.iter().take(2) {
        for orientation in [
            crate::contract::FusionContractOrientation::LhsRhs,
            crate::contract::FusionContractOrientation::RhsLhs,
        ] {
            let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
                &rule,
                &dst_dyn,
                &lhs_dyn,
                &rhs_dyn,
                axes,
                candidate,
                orientation,
            )
            .unwrap();
            let mut dst = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
                initial.clone(),
                dst_space.clone(),
            )
            .unwrap();
            let mut profiled = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
                initial.clone(),
                dst_space.clone(),
            )
            .unwrap();
            crate::contract::execute_dynamic_tree_execution_artifact_profile_pair_for_test(
                &rule,
                &plan,
                &mut dst,
                &mut profiled,
                &lhs,
                &rhs,
                alpha,
                beta,
            )
            .unwrap();
            assert_eq!(profiled.data(), dst.data());
            for (&actual, &expected) in dst.data().iter().zip(&expected) {
                assert!((actual - expected).abs() < 1.0e-10);
            }
        }
    }
}

#[test]
fn paired_axis_selector_scores_once_and_publishes_only_winner_replay() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = SU2FusionRule;
    let leg = || SectorLeg::new([(SU2Irrep::from_twice_spin(0).sector_id(), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([1, 1], [1, 1]).unwrap(),
        hom,
        &rule,
        [vec![1; 4]],
    )
    .unwrap();
    let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![2.0], space.clone()).unwrap();
    let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![3.0], space.clone()).unwrap();
    let mut dst = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0], space).unwrap();
    let axes = TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]);
    crate::contract::reset_source_layout_homspace_id_comparisons();
    let _pure_plan = prepare_tensorcontract_fusion_plan(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    // What: cold candidate scoring never enters the runtime HomSpace identity path.
    assert_eq!(crate::contract::source_layout_homspace_id_comparisons(), 0);
    reset_global_operation_caches();
    crate::contract::reset_candidate_score_calls();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    context
        .tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, 1.0, 0.0)
        .unwrap();
    // What: eager resolution scores four candidates and retains only reusable
    // transform/core-destination components.
    assert_eq!(crate::contract::candidate_score_calls(), 4);
    assert_eq!(context.tree_context().cache().structure_len(), 2);
    assert_eq!(context.dynamic_fusion_space_cache_len(), 2);
    let cache_len = context.dynamic_fusion_space_cache_len();

    context
        .tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, 1.0, 0.0)
        .unwrap();
    // What: an ordinary repeat resolves eagerly while component caches do not
    // accumulate another complete execution artifact.
    assert_eq!(crate::contract::candidate_score_calls(), 8);
    assert_eq!(context.dynamic_fusion_space_cache_len(), cache_len);
}

#[test]
fn reverse_winner_is_independent_of_first_cached_consumer() {
    struct RejectingStorageGemm;

    impl tenet_operations::fusion_replay::StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>>
        for RejectingStorageGemm
    {
        fn matmul_range_into(
            &mut self,
            _dst: &mut Vec<f64>,
            _dst_offset: usize,
            _lhs: &Vec<f64>,
            _lhs_offset: usize,
            _rhs: &Vec<f64>,
            _rhs_offset: usize,
            _rows: usize,
            _contracted: usize,
            _cols: usize,
        ) -> Result<(), OperationError> {
            panic!("reverse storage-direct resolution must reject before GEMM")
        }
    }

    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = U1FusionRule;
    let space = |dimensions: [usize; 4]| {
        let homspace = FusionTreeHomSpace::from_sector_ids(
            [(0, dimensions[0]), (0, dimensions[1])],
            [(0, dimensions[2]), (0, dimensions[3])],
        );
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<2, 2>::from_dims(
                [dimensions[0], dimensions[1]],
                [dimensions[2], dimensions[3]],
            )
            .unwrap(),
            homspace,
            &rule,
            [dimensions.to_vec()],
        )
        .unwrap()
    };
    let lhs_space = space([2, 3, 5, 7]);
    let rhs_space = space([11, 13, 3, 2]);
    let dst_space = space([11, 13, 5, 7]);
    let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_space.required_len().unwrap())
            .map(|index| index as f64 * 0.125 - 3.0)
            .collect(),
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..rhs_space.required_len().unwrap())
            .map(|index| 2.0 - index as f64 * 0.03125)
            .collect(),
        rhs_space,
    )
    .unwrap();
    let initial = (0..dst_space.required_len().unwrap())
        .map(|index| index as f64 * 0.0625 - 1.0)
        .collect::<Vec<_>>();
    let alpha = 1.25;
    let beta = -0.5;
    let axes =
        || TensorContractSpec::new(&[1, 0], &[2, 3], OutputAxisOrder::from_axes(&[2, 3, 0, 1]));

    let lhs_block = lhs.structure().block(0).unwrap();
    let rhs_block = rhs.structure().block(0).unwrap();
    let dst_block = dst_space.subblock_structure().block(0).unwrap();
    let mut oracle = initial.iter().map(|value| beta * value).collect::<Vec<_>>();
    for r0 in 0..11 {
        for r1 in 0..13 {
            for l2 in 0..5 {
                for l3 in 0..7 {
                    let mut sum = 0.0;
                    for c0 in 0..2 {
                        for c1 in 0..3 {
                            let lhs_offset = lhs_block.offset()
                                + [c0, c1, l2, l3]
                                    .iter()
                                    .zip(lhs_block.strides())
                                    .map(|(&index, &stride)| index * stride)
                                    .sum::<usize>();
                            let rhs_offset = rhs_block.offset()
                                + [r0, r1, c1, c0]
                                    .iter()
                                    .zip(rhs_block.strides())
                                    .map(|(&index, &stride)| index * stride)
                                    .sum::<usize>();
                            sum += lhs.data()[lhs_offset] * rhs.data()[rhs_offset];
                        }
                    }
                    let dst_offset = dst_block.offset()
                        + [r0, r1, l2, l3]
                            .iter()
                            .zip(dst_block.strides())
                            .map(|(&index, &stride)| index * stride)
                            .sum::<usize>();
                    oracle[dst_offset] += alpha * sum;
                }
            }
        }
    }
    let assert_oracle = |actual: &[f64]| {
        for (index, (&actual, &expected)) in actual.iter().zip(&oracle).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-9,
                "element {index}: {actual} != {expected}"
            );
        }
    };
    let lhs_dynamic = DynamicFusionMapSpace::from_typed(lhs.fusion_space().unwrap());
    let rhs_dynamic = DynamicFusionMapSpace::from_typed(rhs.fusion_space().unwrap());
    let dst_dynamic = DynamicFusionMapSpace::from_typed(&dst_space);
    let reverse_plan =
        crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
            &rule,
            &dst_dynamic,
            &lhs_dynamic,
            &rhs_dynamic,
            axes(),
            &crate::contract::contracted_axis_order_candidates(&[1, 0], &[2, 3])[1],
            crate::contract::FusionContractOrientation::RhsLhs,
        )
        .unwrap();
    let reverse_unsupported = OperationError::UnsupportedTensorContractScope {
        message: "caller-owned fusion contraction scratch supports only LhsRhs orientation",
    };
    let mut explicit_dst =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(initial.clone(), dst_space.clone())
            .unwrap();
    let mut explicit_core_dst = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![9.0; initial.len()],
        dst_space.clone(),
    )
    .unwrap();
    let mut explicit_lhs_core = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![8.0; lhs.data().len()],
        lhs.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut explicit_rhs_core = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![7.0; rhs.data().len()],
        rhs.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let explicit_dst_before = explicit_dst.data().to_vec();
    let explicit_core_dst_before = explicit_core_dst.data().to_vec();
    let explicit_lhs_before = explicit_lhs_core.data().to_vec();
    let explicit_rhs_before = explicit_rhs_core.data().to_vec();
    assert_eq!(
        tensorcontract_fusion_prepared_into(
            &rule,
            &reverse_plan,
            &mut explicit_dst,
            &mut explicit_lhs_core,
            &mut explicit_rhs_core,
            &lhs,
            &rhs,
            alpha,
            beta,
        ),
        Err(reverse_unsupported.clone())
    );
    assert_eq!(
        tensorcontract_fusion_prepared_into_core_dst(
            &rule,
            &reverse_plan,
            &mut explicit_dst,
            &mut explicit_core_dst,
            &mut explicit_lhs_core,
            &mut explicit_rhs_core,
            &lhs,
            &rhs,
            alpha,
            beta,
        ),
        Err(reverse_unsupported.clone())
    );
    let mut explicit_context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    assert_eq!(
        explicit_context.tensorcontract_fusion_prepared_into(
            &rule,
            &reverse_plan,
            &mut explicit_dst,
            &mut explicit_lhs_core,
            &mut explicit_rhs_core,
            &lhs,
            &rhs,
            alpha,
            beta,
        ),
        Err(reverse_unsupported.clone())
    );
    assert_eq!(
        explicit_context.tensorcontract_fusion_prepared_into_core_dst(
            &rule,
            &reverse_plan,
            &mut explicit_dst,
            &mut explicit_core_dst,
            &mut explicit_lhs_core,
            &mut explicit_rhs_core,
            &lhs,
            &rhs,
            alpha,
            beta,
        ),
        Err(reverse_unsupported)
    );
    assert_eq!(explicit_dst.data(), explicit_dst_before);
    assert_eq!(explicit_core_dst.data(), explicit_core_dst_before);
    assert_eq!(explicit_lhs_core.data(), explicit_lhs_before);
    assert_eq!(explicit_rhs_core.data(), explicit_rhs_before);

    let fresh_dst = || {
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(initial.clone(), dst_space.clone())
            .unwrap()
    };

    crate::contract::reset_candidate_score_calls();
    let mut one_shot_dst = fresh_dst();
    tensorcontract_fusion_into(&rule, &mut one_shot_dst, &lhs, &rhs, axes(), alpha, beta).unwrap();
    assert_eq!(crate::contract::candidate_score_calls(), 4);
    assert_oracle(one_shot_dst.data());

    crate::contract::reset_candidate_score_calls();
    let mut with_dst = fresh_dst();
    let mut backend = DenseTreeTransformOperations::default_executor();
    let mut workspace = TensorContractWorkspace::default();
    tensorcontract_fusion_into_with(
        &mut backend,
        &mut workspace,
        &rule,
        &mut with_dst,
        &lhs,
        &rhs,
        axes(),
        alpha,
        beta,
    )
    .unwrap();
    assert_eq!(crate::contract::candidate_score_calls(), 4);
    assert_oracle(with_dst.data());

    crate::contract::reset_candidate_score_calls();
    let mut with_backends_dst = fresh_dst();
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
        &mut with_backends_dst,
        &lhs,
        &rhs,
        axes(),
        alpha,
        beta,
    )
    .unwrap();
    assert_eq!(crate::contract::candidate_score_calls(), 4);
    assert_oracle(with_backends_dst.data());

    let provider = Arc::new(rule);
    let lhs_bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        DynamicFusionMapSpace::from_typed(lhs.fusion_space().unwrap()),
        Arc::clone(&provider),
    )
    .unwrap();
    let rhs_bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        DynamicFusionMapSpace::from_typed(rhs.fusion_space().unwrap()),
        Arc::clone(&provider),
    )
    .unwrap();
    let dst_bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        DynamicFusionMapSpace::from_typed(&dst_space),
        Arc::clone(&provider),
    )
    .unwrap();
    let unsupported = OperationError::UnsupportedTensorContractScope {
        message: "storage-direct contraction supports only the canonical fully-direct route; \
                  this contraction needs tree transforms or conjugate structures, which have \
                  no device kernels yet",
    };

    for storage_first in [true, false] {
        reset_global_operation_caches();
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        let mut storage_dst = vec![0.0; initial.len()];
        let mut owned_dst =
            TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(initial.clone(), dst_space.clone())
                .unwrap();
        let storage_call =
            |context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
             storage_dst: &mut Vec<f64>| {
                context.tensorcontract_fusion_dyn_direct_on_storage(
                    &mut RejectingStorageGemm,
                    &dst_bound,
                    storage_dst,
                    &lhs_bound,
                    &lhs.data().to_vec(),
                    &rhs_bound,
                    &rhs.data().to_vec(),
                    axes(),
                )
            };
        if storage_first {
            assert_eq!(
                storage_call(&mut context, &mut storage_dst),
                Err(unsupported.clone())
            );
            context
                .tensorcontract_fusion_into(
                    provider.as_ref(),
                    &mut owned_dst,
                    &lhs,
                    &rhs,
                    axes(),
                    alpha,
                    beta,
                )
                .unwrap();
        } else {
            context
                .tensorcontract_fusion_into(
                    provider.as_ref(),
                    &mut owned_dst,
                    &lhs,
                    &rhs,
                    axes(),
                    alpha,
                    beta,
                )
                .unwrap();
            assert_eq!(
                storage_call(&mut context, &mut storage_dst),
                Err(unsupported.clone())
            );
        }
        assert_eq!(storage_dst, vec![0.0; initial.len()]);
        assert_oracle(owned_dst.data());
        assert_eq!(
            context.last_resolution_orientation(),
            Some(crate::contract::FusionContractOrientation::RhsLhs)
        );
    }

    for typed_first in [true, false] {
        reset_global_operation_caches();
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        let mut typed_dst =
            TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(initial.clone(), dst_space.clone())
                .unwrap();
        let mut dynamic_dst = initial.clone();
        let typed_call = |context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
                          dst: &mut TensorMap<f64, 2, 2>| {
            context.tensorcontract_fusion_into(
                provider.as_ref(),
                dst,
                &lhs,
                &rhs,
                axes(),
                alpha,
                beta,
            )
        };
        let dynamic_call =
            |context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
             dst: &mut [f64]| {
                context.tensorcontract_fusion_dyn_into(
                    &dst_bound,
                    dst,
                    &lhs_bound,
                    lhs.data(),
                    &rhs_bound,
                    rhs.data(),
                    axes(),
                    alpha,
                    beta,
                )
            };
        if typed_first {
            typed_call(&mut context, &mut typed_dst).unwrap();
            dynamic_call(&mut context, &mut dynamic_dst).unwrap();
        } else {
            dynamic_call(&mut context, &mut dynamic_dst).unwrap();
            typed_call(&mut context, &mut typed_dst).unwrap();
        }
        assert_oracle(typed_dst.data());
        assert_oracle(&dynamic_dst);
        assert_eq!(
            context.last_resolution_orientation(),
            Some(crate::contract::FusionContractOrientation::RhsLhs)
        );
    }

    for profiled_first in [true, false] {
        reset_global_operation_caches();
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        let mut ordinary_dst =
            TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(initial.clone(), dst_space.clone())
                .unwrap();
        let mut profiled_dst =
            TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(initial.clone(), dst_space.clone())
                .unwrap();
        let ordinary_call =
            |context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
             dst: &mut TensorMap<f64, 2, 2>| {
                context.tensorcontract_fusion_into(
                    provider.as_ref(),
                    dst,
                    &lhs,
                    &rhs,
                    axes(),
                    alpha,
                    beta,
                )
            };
        let profiled_call =
            |context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
             dst: &mut TensorMap<f64, 2, 2>| {
                let mut profile = TensorContractFusionProfile::default();
                context.tensorcontract_fusion_into_profiled(
                    provider.as_ref(),
                    dst,
                    &lhs,
                    &rhs,
                    axes(),
                    alpha,
                    beta,
                    &mut profile,
                )?;
                assert_eq!(profile.route, TensorContractFusionRoute::DynamicTreeCore);
                Ok::<_, OperationError>(())
            };
        if profiled_first {
            profiled_call(&mut context, &mut profiled_dst).unwrap();
            ordinary_call(&mut context, &mut ordinary_dst).unwrap();
        } else {
            ordinary_call(&mut context, &mut ordinary_dst).unwrap();
            profiled_call(&mut context, &mut profiled_dst).unwrap();
        }
        assert_oracle(ordinary_dst.data());
        assert_oracle(profiled_dst.data());
        assert_eq!(
            context.last_resolution_orientation(),
            Some(crate::contract::FusionContractOrientation::RhsLhs)
        );
    }
}

#[test]
fn paired_axis_selector_rejects_invalid_axes_before_scoring_or_mutation() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = U1FusionRule;
    let leg = || SectorLeg::new([(U1Irrep::new(0).sector_id(), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([1, 1], [1, 1]).unwrap(),
        hom,
        &rule,
        [vec![1; 4]],
    )
    .unwrap();
    let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![2.0], space.clone()).unwrap();
    let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![3.0], space.clone()).unwrap();
    let mut free_dst =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![5.0], space.clone()).unwrap();
    let axes = TensorContractSpec::with_default_output_order(&[3, 3], &[0, 1]);
    let mut free_backend = DenseTreeTransformOperations::default_executor();
    let mut free_workspace = TensorContractWorkspace::default();
    let error = tensorcontract_fusion_into_with(
        &mut free_backend,
        &mut free_workspace,
        &rule,
        &mut free_dst,
        &lhs,
        &rhs,
        axes,
        1.0,
        0.0,
    )
    .unwrap_err();
    // What: duplicate axes retain validation precedence and leave all buffers untouched.
    assert_eq!(
        error,
        OperationError::InvalidAxisSet {
            tensor: "lhs",
            axes: vec![3, 3],
            rank: 4,
        }
    );
    assert_eq!(lhs.data(), &[2.0]);
    assert_eq!(rhs.data(), &[3.0]);
    assert_eq!(free_dst.data(), &[5.0]);

    let mut split_dst =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![5.0], space.clone()).unwrap();
    let mut tree_backend = HostTensorOperations;
    let mut tree_workspace = TreeTransformWorkspace::default();
    let mut contract_backend = DenseTreeTransformOperations::default_executor();
    let mut contract_workspace = TensorContractWorkspace::default();
    let split_error = tensorcontract_fusion_into_with_backends(
        &mut tree_backend,
        &mut tree_workspace,
        &mut contract_backend,
        &mut contract_workspace,
        &rule,
        &mut split_dst,
        &lhs,
        &rhs,
        axes,
        1.0,
        0.0,
    )
    .unwrap_err();
    assert_eq!(split_error, error);
    assert_eq!(split_dst.data(), &[5.0]);

    let mut context_dst =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![5.0], space).unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let context_error = context
        .tensorcontract_fusion_into(&rule, &mut context_dst, &lhs, &rhs, axes, 1.0, 0.0)
        .unwrap_err();
    assert_eq!(context_error, error);
    assert_eq!(context_dst.data(), &[5.0]);
}

#[test]
fn crossed_axis_selection_preserves_real_fermion_parity_complex_result() {
    // What: both crossed caller pair orders produce one odd FermionParity result.
    let rule = FermionParityFusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let leg = || SectorLeg::new(sectors.map(|sector| (sector, 1)), false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let dense = TensorMapSpace::<2, 2>::from_dims([2, 2], [2, 2]).unwrap();
    let blocks = hom.fusion_tree_keys(&rule).len();
    let space =
        FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, &rule, vec![vec![1; 4]; blocks])
            .unwrap();
    let len = space.subblock_structure().required_len().unwrap();
    let lhs = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| Complex64::new(i as f64 + 1.0, 0.25 * i as f64))
            .collect(),
        space.clone(),
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| Complex64::new(0.5 * i as f64 - 0.75, 1.0 - 0.1 * i as f64))
            .collect(),
        space.clone(),
    )
    .unwrap();
    let candidates = crate::contract::contracted_axis_order_candidates(&[3, 2], &[0, 1]);
    assert!(candidates.len() >= 2);
    let dst_dyn = DynamicFusionMapSpace::from_typed(&space);
    let lhs_dyn = DynamicFusionMapSpace::from_typed(&space);
    let rhs_dyn = DynamicFusionMapSpace::from_typed(&space);
    let mut outputs = Vec::new();
    for candidate in candidates.iter().take(2) {
        for orientation in [
            crate::contract::FusionContractOrientation::LhsRhs,
            crate::contract::FusionContractOrientation::RhsLhs,
        ] {
            let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
                &rule,
                &dst_dyn,
                &lhs_dyn,
                &rhs_dyn,
                TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
                candidate,
                orientation,
            )
            .unwrap();
            let mut dst = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
                vec![Complex64::zero(); len],
                space.clone(),
            )
            .unwrap();
            let mut tree_backend = DenseTreeTransformOperations::default_executor();
            let mut tree_workspace = TreeTransformWorkspace::default();
            let mut contract_backend = DenseTreeTransformOperations::default_executor();
            let mut contract_workspace = TensorContractWorkspace::default();
            crate::contract::tensorcontract_fusion_dynamic_plan_into_with(
                &mut tree_backend,
                &mut tree_workspace,
                &mut contract_backend,
                &mut contract_workspace,
                &rule,
                &plan,
                &mut dst,
                &lhs,
                &rhs,
                Complex64::one(),
                Complex64::zero(),
            )
            .unwrap();
            outputs.push(dst);
        }
    }
    assert_eq!(outputs.len(), 4);
    for output in &outputs[1..] {
        assert_eq!(outputs[0].fusion_space(), output.fusion_space());
        for (a, b) in outputs[0].data().iter().zip(output.data()) {
            assert!((a - b).norm() < 1.0e-10, "candidate mismatch: {a} vs {b}");
        }
    }
    let selected = prepare_tensorcontract_fusion_plan(
        &rule,
        &space,
        &space,
        &space,
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
    )
    .unwrap();
    assert_eq!(
        selected.lhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.lhs_transform().domain_permutation(), &[2, 3]);
    assert_eq!(
        selected.rhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.rhs_transform().codomain_permutation(), &[1, 0]);
    let mut rhs_permuted = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero(); len],
        space.clone(),
    )
    .unwrap();
    tree_transform_into(
        &rule,
        TreeTransformOperation::permute([1, 0], [2, 3]),
        &mut rhs_permuted,
        &rhs,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    let mut oracle = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero(); len],
        space.clone(),
    )
    .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut oracle,
        &lhs,
        &rhs_permuted,
        TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    let mut normal = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero(); len],
        space,
    )
    .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut normal,
        &lhs,
        &rhs,
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    for ((actual, expected), fixed) in normal
        .data()
        .iter()
        .zip(oracle.data())
        .zip(outputs[0].data())
    {
        assert!((*actual - *expected).norm() < 1.0e-10);
        assert!((*actual - *fixed).norm() < 1.0e-10);
    }
}

#[test]
fn crossed_axis_selection_preserves_asymmetric_fz2_u1_su2_result() {
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let sector = rule.encode_sector(
        left_rule.encode_sector(SectorId::new(1), U1Irrep::new(0).sector_id()),
        SU2Irrep::from_twice_spin(1).sector_id(),
    );
    let build_space = |dimensions: [usize; 4]| {
        let leg = |dimension| SectorLeg::new([(sector, dimension)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(dimensions[0]), leg(dimensions[1])]),
            FusionProductSpace::new([leg(dimensions[2]), leg(dimensions[3])]),
        );
        let blocks = hom.fusion_tree_keys(&rule).len();
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<2, 2>::from_dims(
                [dimensions[0], dimensions[1]],
                [dimensions[2], dimensions[3]],
            )
            .unwrap(),
            hom,
            &rule,
            vec![dimensions.to_vec(); blocks],
        )
        .unwrap()
    };
    let lhs_space = build_space([1, 1, 2, 3]);
    let rhs_space = build_space([3, 2, 4, 5]);
    let dst_space = build_space([1, 1, 4, 5]);
    let lhs_len = lhs_space.required_len().unwrap();
    let rhs_len = rhs_space.required_len().unwrap();
    let dst_len = dst_space.required_len().unwrap();
    let lhs = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_len)
            .map(|index| Complex64::new(index as f64 + 0.5, -0.25))
            .collect(),
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        (0..rhs_len)
            .map(|index| Complex64::new(1.0 - index as f64 * 0.125, 0.75))
            .collect(),
        rhs_space,
    )
    .unwrap();
    let candidates = crate::contract::contracted_axis_order_candidates(&[3, 2], &[0, 1]);
    let dst_dyn = DynamicFusionMapSpace::from_typed(&dst_space);
    let lhs_dyn = DynamicFusionMapSpace::from_typed(lhs.fusion_space().unwrap());
    let rhs_dyn = DynamicFusionMapSpace::from_typed(rhs.fusion_space().unwrap());
    let mut outputs = Vec::new();
    for candidate in &candidates {
        for orientation in [
            crate::contract::FusionContractOrientation::LhsRhs,
            crate::contract::FusionContractOrientation::RhsLhs,
        ] {
            let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
                &rule,
                &dst_dyn,
                &lhs_dyn,
                &rhs_dyn,
                TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
                candidate,
                orientation,
            )
            .unwrap();
            let mut dst = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
                vec![Complex64::zero(); dst_len],
                dst_space.clone(),
            )
            .unwrap();
            let mut tree_backend = DenseTreeTransformOperations::default_executor();
            let mut tree_workspace = TreeTransformWorkspace::default();
            let mut contract_backend = DenseTreeTransformOperations::default_executor();
            let mut contract_workspace = TensorContractWorkspace::default();
            crate::contract::tensorcontract_fusion_dynamic_plan_into_with(
                &mut tree_backend,
                &mut tree_workspace,
                &mut contract_backend,
                &mut contract_workspace,
                &rule,
                &plan,
                &mut dst,
                &lhs,
                &rhs,
                Complex64::one(),
                Complex64::zero(),
            )
            .unwrap();
            let mut artifact = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
                vec![Complex64::zero(); dst_len],
                dst_space.clone(),
            )
            .unwrap();
            let mut profiled = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
                vec![Complex64::zero(); dst_len],
                dst_space.clone(),
            )
            .unwrap();
            crate::contract::execute_dynamic_tree_execution_artifact_profile_pair_for_test(
                &rule,
                &plan,
                &mut artifact,
                &mut profiled,
                &lhs,
                &rhs,
                Complex64::one(),
                Complex64::zero(),
            )
            .unwrap();
            // What: compiled orientation preserves the eager destination
            // structure and every reduced-block value for the odd, charged,
            // half-spin product sector.
            assert_eq!(artifact.structure(), dst.structure());
            // Path-to-path agreement of two executors of one plan with SU2
            // recoupling, under the tolerance rule (terms bounded by
            // len(lhs) * len(rhs), recoupled trees included); the profiled replay of the same artifact stays
            // exact.
            numerics::assert_slices_close(
                "compiled artifact vs direct plan replay",
                artifact.data(),
                dst.data(),
                lhs.data().len() * rhs.data().len(),
            );
            assert_eq!(profiled.data(), artifact.data());
            outputs.push(dst);
        }
    }
    // What: crossed pair order and operand orientation are immaterial for an
    // odd, charged, half-spin product sector.
    assert_eq!(outputs.len(), 4);
    for output in &outputs[1..] {
        for (lhs, rhs) in outputs[0].data().iter().zip(output.data()) {
            assert!((*lhs - *rhs).norm() < 1.0e-10);
        }
    }
    let selected = prepare_tensorcontract_fusion_plan(
        &rule,
        &dst_space,
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
    )
    .unwrap();
    assert_eq!(
        selected.lhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.lhs_transform().domain_permutation(), &[3, 2]);
    assert_eq!(
        selected.rhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.rhs_transform().codomain_permutation(), &[0, 1]);
    let mut lhs_permuted = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero(); lhs_len],
        build_space([1, 1, 3, 2]),
    )
    .unwrap();
    tree_transform_into(
        &rule,
        TreeTransformOperation::permute([0, 1], [3, 2]),
        &mut lhs_permuted,
        &lhs,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    let mut oracle = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero(); dst_len],
        dst_space.clone(),
    )
    .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut oracle,
        &lhs_permuted,
        &rhs,
        TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    let mut normal = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero(); dst_len],
        dst_space,
    )
    .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut normal,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    for ((actual, expected), fixed) in normal
        .data()
        .iter()
        .zip(oracle.data())
        .zip(outputs[0].data())
    {
        assert!((*actual - *expected).norm() < 1.0e-10);
        assert!((*actual - *fixed).norm() < 1.0e-10);
    }
}
