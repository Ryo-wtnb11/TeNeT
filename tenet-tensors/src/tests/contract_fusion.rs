use super::*;
use std::sync::Arc;

#[test]
fn tensor_contract_fusion_execution_context_reports_host_placement() {
    let context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    assert_eq!(context.tree_backend_placement(), Placement::Host);
    assert_eq!(context.tree_workspace_placement(), Placement::Host);
    assert_eq!(context.contract_backend_placement(), Placement::Host);
    assert_eq!(context.contract_workspace_placement(), Placement::Host);
    assert_eq!(context.fusion_block_workspace_placement(), Placement::Host);
    assert_eq!(
        context.fusion_scratch_workspace_placement(),
        Placement::Host
    );
    assert!(context.is_host_context());
}

#[test]
fn tensorcontract_fusion_structure_enumerates_z2_compose_blocks_and_replays() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
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
        TensorContractSpec::with_default_output_order(&[1], &[0]),
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
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[50.0, 102.0]);

    let mut context_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        vec![10.0, 20.0],
        dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut context_dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            2.0,
            3.0,
        )
        .unwrap();
    assert_eq!(context_dst.data(), &[50.0, 102.0]);
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert_eq!(context.contraction_resolution_cache_hits(), 0);
    assert!(context.contraction_resolution_cache_misses() >= 1);

    context_dst.data_mut().copy_from_slice(&[10.0, 20.0]);
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut context_dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            2.0,
            3.0,
        )
        .unwrap();
    assert_eq!(context_dst.data(), &[50.0, 102.0]);
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert!(context.contraction_resolution_cache_hits() >= 1);
    assert!(context.contraction_resolution_cache_misses() >= 1);

    context_dst.data_mut().copy_from_slice(&[10.0, 20.0]);
    let mut profile = TensorContractFusionProfile::default();
    context
        .tensorcontract_fusion_into_profiled(
            &rule,
            &mut context_dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            2.0,
            3.0,
            &mut profile,
        )
        .unwrap();
    assert_eq!(context_dst.data(), &[50.0, 102.0]);
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert_eq!(profile.route, TensorContractFusionRoute::CoreFusionBlocks);
    assert_eq!(profile.lhs_transform_calls, 0);
    assert_eq!(profile.rhs_transform_calls, 0);
    assert!(profile.core_contract_groups > 0);
    // alpha = 2, beta = 3 and every group still runs the direct GEMM: the
    // accumulate factors ride on the GEMM itself (TensorKit mul! semantics),
    // never on a scatter pass.
    assert_eq!(
        profile.core_direct_gemm_groups,
        profile.core_contract_groups
    );
    assert_eq!(profile.core_scatter, std::time::Duration::ZERO);
    assert_eq!(profile.tree_replay.single_blocks, 0);
    assert_eq!(profile.tree_replay.multi_blocks, 0);
}

#[test]
fn tensorcontract_fusion_default_host_api_accepts_custom_host_storage() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let fusion_space = || {
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap()
    };
    let lhs = test_host_read_fusion_tensor_map(vec![2.0_f64, 3.0], fusion_space());
    let rhs = test_host_read_fusion_tensor_map(vec![5.0_f64, 7.0], fusion_space());
    let mut dst = test_host_fusion_tensor_map(vec![10.0_f64, 20.0], fusion_space());

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[50.0, 102.0]);
}

#[test]
fn tensorcontract_fusion_context_accepts_custom_host_storage() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let fusion_space = || {
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap()
    };
    let lhs = test_host_read_fusion_tensor_map(vec![2.0_f64, 3.0], fusion_space());
    let rhs = test_host_read_fusion_tensor_map(vec![5.0_f64, 7.0], fusion_space());
    let mut dst = test_host_fusion_tensor_map(vec![10.0_f64, 20.0], fusion_space());
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    context
        .tensorcontract_fusion_into(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            2.0,
            3.0,
        )
        .unwrap();

    assert_eq!(dst.data(), &[50.0, 102.0]);
    assert!(context.contraction_resolution_cache_len() >= 1);
}

#[test]
fn tensorcontract_fusion_su2_swap_matches_explicit_permute_then_compose() {
    // What: SU2 C[a b; g h] = A[a b; c d] * B[d c; g h] equals an explicit
    // RHS permutation followed by core composition.
    let rule = SU2FusionRule;
    let sectors = [
        SU2Irrep::from_twice_spin(0).sector_id(),
        SU2Irrep::from_twice_spin(1).sector_id(),
    ];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.map(|sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let dense =
        || TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap();
    let space = |hom: &FusionTreeHomSpace| {
        let count = hom.fusion_tree_keys(&rule).len();
        FusionTensorMapSpace::from_degeneracy_shapes(
            dense(),
            hom.clone(),
            &rule,
            vec![vec![degeneracy; 4]; count],
        )
        .unwrap()
    };
    let tensor_space = space(&homspace);
    let fill = |seed: f64| move |index: usize| 0.25 * seed + ((index * 7 + 3) % 11) as f64 - 5.0;
    let len = tensor_space.subblock_structure().required_len().unwrap();
    let lhs_data = (0..len).map(fill(1.0)).collect::<Vec<_>>();
    let rhs_data = (0..len).map(fill(2.0)).collect::<Vec<_>>();
    let lhs =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(lhs_data, tensor_space.clone()).unwrap();
    let rhs =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(rhs_data, tensor_space.clone()).unwrap();

    // Route under test: swap axes through the fusion contraction facade.
    let mut dst_swap =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; len], tensor_space.clone())
            .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut dst_swap,
        &lhs,
        &rhs,
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        1.0,
        0.0,
    )
    .unwrap();

    // Reference: explicitly permute rhs codomain legs, then core compose.
    let permuted_space = space(&homspace);
    let mut rhs_permuted =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; len], permuted_space).unwrap();
    tree_transform_into(
        &rule,
        TreeTransformOperation::permute([1, 0], [2, 3]),
        &mut rhs_permuted,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    let mut dst_compose =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; len], tensor_space.clone())
            .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut dst_compose,
        &lhs,
        &rhs_permuted,
        TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]),
        1.0,
        0.0,
    )
    .unwrap();

    for (index, (&actual, &expected)) in dst_swap.data().iter().zip(dst_compose.data()).enumerate()
    {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "swap vs permute+compose mismatch at {index}: {actual} vs {expected}"
        );
    }
}

#[test]
fn forced_axis_order_candidates_have_identical_u1_result() {
    // What: both paired caller orders produce the same U1 contraction result.
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
        let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order(
            &rule,
            &dst_dyn,
            &lhs_dyn,
            &rhs_dyn,
            TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
            candidate,
        )
        .unwrap();
        let mut dst =
            TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; len], space.clone())
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
            1.0,
            0.0,
        )
        .unwrap();
        outputs.push(dst);
    }
    assert_eq!(outputs[0].fusion_space(), outputs[1].fusion_space());
    for (a, b) in outputs[0].data().iter().zip(outputs[1].data()) {
        assert!((a - b).abs() < 1.0e-10, "candidate mismatch: {a} vs {b}");
    }
    let selected = prepare_tensorcontract_fusion_plan(
        &rule,
        &space,
        &space,
        &space,
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
    )
    .unwrap();
    assert!(matches!(
        selected.lhs_transform(),
        TreeTransformOperation::Permute {
            domain_permutation,
            ..
        } if domain_permutation.as_slice() == [2, 3]
    ));
    assert!(matches!(
        selected.rhs_transform(),
        TreeTransformOperation::Permute {
            codomain_permutation,
            ..
        } if codomain_permutation.as_slice() == [1, 0]
    ));
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
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    context
        .tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, 1.0, 0.0)
        .unwrap();
    // What: cold resolution scores both candidates but publishes one two-transform artifact.
    assert_eq!(crate::contract::candidate_score_calls(), 2);
    assert_eq!(context.tree_context().cache().plan_len(), 2);
    assert_eq!(context.tree_context().cache().structure_len(), 2);
    assert_eq!(context.dynamic_fusion_space_cache_len(), 3);
    let cache_len = context.dynamic_fusion_space_cache_len();

    context
        .tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, 1.0, 0.0)
        .unwrap();
    // What: a warm resolution reuses the winner without rescoring or publishing loser state.
    assert_eq!(crate::contract::candidate_score_calls(), 2);
    assert_eq!(context.dynamic_fusion_space_cache_len(), cache_len);
    assert!(context.contraction_resolution_cache_hits() >= 1);
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
    let mut dst = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![5.0], space).unwrap();
    let error = tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[3, 3], &[0, 1]),
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
    assert_eq!(dst.data(), &[5.0]);
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
        let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order(
            &rule,
            &dst_dyn,
            &lhs_dyn,
            &rhs_dyn,
            TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
            candidate,
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
    assert_eq!(outputs[0].fusion_space(), outputs[1].fusion_space());
    for (a, b) in outputs[0].data().iter().zip(outputs[1].data()) {
        assert!((a - b).norm() < 1.0e-10, "candidate mismatch: {a} vs {b}");
    }
    let selected = prepare_tensorcontract_fusion_plan(
        &rule,
        &space,
        &space,
        &space,
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
    )
    .unwrap();
    assert!(matches!(
        selected.lhs_transform(),
        TreeTransformOperation::Permute {
            domain_permutation,
            ..
        } if domain_permutation.as_slice() == [2, 3]
    ));
    assert!(matches!(
        selected.rhs_transform(),
        TreeTransformOperation::Permute {
            codomain_permutation,
            ..
        } if codomain_permutation.as_slice() == [1, 0]
    ));
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
        let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order(
            &rule,
            &dst_dyn,
            &lhs_dyn,
            &rhs_dyn,
            TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
            candidate,
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
        outputs.push(dst);
    }
    // What: crossed pair order is immaterial for an odd, charged, half-spin product sector.
    assert_eq!(outputs.len(), 2);
    for (lhs, rhs) in outputs[0].data().iter().zip(outputs[1].data()) {
        assert!((*lhs - *rhs).norm() < 1.0e-10);
    }
    let selected = prepare_tensorcontract_fusion_plan(
        &rule,
        &dst_space,
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
    )
    .unwrap();
    assert!(matches!(
        selected.lhs_transform(),
        TreeTransformOperation::Permute {
            domain_permutation,
            ..
        } if domain_permutation.as_slice() == [3, 2]
    ));
    assert!(matches!(
        selected.rhs_transform(),
        TreeTransformOperation::Permute {
            codomain_permutation,
            ..
        } if codomain_permutation.as_slice() == [0, 1]
    ));
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

#[test]
fn prepared_tensorcontract_fusion_matches_facade_and_rejects_foreign_tensors() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let fusion_space = || {
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap()
    };
    // What: replaying the tensors used to prepare the handle remains valid.
    let space = fusion_space();
    let lhs = test_host_read_fusion_tensor_map(vec![2.0_f64, 3.0], space.clone());
    let rhs = test_host_read_fusion_tensor_map(vec![5.0_f64, 7.0], space.clone());
    let mut dst_facade = test_host_fusion_tensor_map(vec![10.0_f64, 20.0], space.clone());
    let mut dst_prepared = test_host_fusion_tensor_map(vec![10.0_f64, 20.0], space.clone());
    let axes = TensorContractSpec::with_default_output_order(&[1], &[0]);
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    let prepared = context
        .prepare_tensorcontract_fusion(&rule, &dst_prepared, &lhs, &rhs, axes)
        .unwrap();
    for _ in 0..2 {
        context
            .tensorcontract_fusion_into(&rule, &mut dst_facade, &lhs, &rhs, axes, 2.0, 3.0)
            .unwrap();
        context
            .execute_prepared_tensorcontract_fusion(
                &prepared,
                &rule,
                &mut dst_prepared,
                &lhs,
                &rhs,
                2.0,
                3.0,
            )
            .unwrap();
    }
    assert_eq!(dst_prepared.data(), dst_facade.data());

    let mut foreign_dst = test_host_fusion_tensor_map(vec![0.0_f64, 0.0], fusion_space());
    let err = context
        .execute_prepared_tensorcontract_fusion(
            &prepared,
            &rule,
            &mut foreign_dst,
            &lhs,
            &rhs,
            1.0,
            0.0,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        OperationError::StructureMismatch {
            tensor: "prepared contraction"
        }
    ));

    let foreign_lhs = test_host_read_fusion_tensor_map(vec![2.0_f64, 3.0], fusion_space());
    let err = context
        .execute_prepared_tensorcontract_fusion(
            &prepared,
            &rule,
            &mut dst_prepared,
            &foreign_lhs,
            &rhs,
            1.0,
            0.0,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        OperationError::StructureMismatch {
            tensor: "prepared contraction"
        }
    ));

    let foreign_rhs = test_host_read_fusion_tensor_map(vec![5.0_f64, 7.0], fusion_space());
    let err = context
        .execute_prepared_tensorcontract_fusion(
            &prepared,
            &rule,
            &mut dst_prepared,
            &lhs,
            &foreign_rhs,
            1.0,
            0.0,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        OperationError::StructureMismatch {
            tensor: "prepared contraction"
        }
    ));
}

#[test]
fn prepared_tensorcontract_fusion_pins_exact_space_allocations() {
    // What: a prepared handle pins all original fusion-space allocations and
    // rejects semantically equal replacements after the source tensors drop.
    let rule = Z2FusionRule;
    let fusion_space = || {
        let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap()
    };
    let axes = TensorContractSpec::with_default_output_order(&[1], &[0]);
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    let (prepared, dst_weak, lhs_weak, rhs_weak) = {
        let lhs = test_host_read_fusion_tensor_map(vec![2.0_f64, 3.0], fusion_space());
        let rhs = test_host_read_fusion_tensor_map(vec![5.0_f64, 7.0], fusion_space());
        let dst = test_host_fusion_tensor_map(vec![0.0_f64, 0.0], fusion_space());
        let dst_weak = Arc::downgrade(dst.fusion_space().unwrap());
        let lhs_weak = Arc::downgrade(lhs.fusion_space().unwrap());
        let rhs_weak = Arc::downgrade(rhs.fusion_space().unwrap());
        let prepared = context
            .prepare_tensorcontract_fusion(&rule, &dst, &lhs, &rhs, axes)
            .unwrap();
        (prepared, dst_weak, lhs_weak, rhs_weak)
    };

    assert!(dst_weak.upgrade().is_some());
    assert!(lhs_weak.upgrade().is_some());
    assert!(rhs_weak.upgrade().is_some());

    let lhs = test_host_read_fusion_tensor_map(vec![2.0_f64, 3.0], fusion_space());
    let rhs = test_host_read_fusion_tensor_map(vec![5.0_f64, 7.0], fusion_space());
    let mut dst = test_host_fusion_tensor_map(vec![0.0_f64, 0.0], fusion_space());
    let err = context
        .execute_prepared_tensorcontract_fusion(&prepared, &rule, &mut dst, &lhs, &rhs, 1.0, 0.0)
        .unwrap_err();
    assert!(matches!(
        err,
        OperationError::StructureMismatch {
            tensor: "prepared contraction"
        }
    ));
}

#[test]
fn tensorcontract_fusion_block_replay_scales_inactive_dst_blocks_once() {
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule);
    let key_for_sector = |sector| {
        keys.iter()
            .find(|key| key.codomain_tree().coupled() == Some(sector))
            .cloned()
            .expect("Z2 one-leg homspace contains requested sector")
    };
    let even_key = key_for_sector(even);
    let odd_key = key_for_sector(odd);

    let lhs_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        homspace.clone(),
        packed_fixture_structure(2, [(even_key.clone(), vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let rhs_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        homspace.clone(),
        packed_fixture_structure(2, [(even_key.clone(), vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let dst_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        homspace,
        packed_fixture_structure(2, [(even_key, vec![1, 1]), (odd_key, vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();

    let lhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![5.0], rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], dst_space).unwrap();
    let axes = TensorContractSpec::with_default_output_order(&[1], &[0]);
    let alpha = 2.0;
    let beta = 3.0;

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(specs, vec![TensorContractBlockSpec::new(0, 0, 0)]);

    tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();

    assert_eq!(dst.data(), &[50.0, 60.0]);
}

#[test]
fn tensorcontract_fusion_block_replay_scatter_beta_supports_dense_dtypes() {
    assert_fusion_block_scatter_beta_dtype(2.0_f32, 5.0, 10.0, 20.0, 2.0, 3.0);
    assert_fusion_block_scatter_beta_dtype(2.0_f64, 5.0, 10.0, 20.0, 2.0, 3.0);
    assert_fusion_block_scatter_beta_dtype(
        Complex32::new(2.0, 1.0),
        Complex32::new(5.0, -2.0),
        Complex32::new(10.0, 1.0),
        Complex32::new(20.0, -3.0),
        Complex32::new(2.0, -1.0),
        Complex32::new(-1.0, 0.5),
    );
    assert_fusion_block_scatter_beta_dtype(
        Complex64::new(2.0, 1.0),
        Complex64::new(5.0, -2.0),
        Complex64::new(10.0, 1.0),
        Complex64::new(20.0, -3.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(-1.0, 0.5),
    );
}

fn assert_fusion_block_scatter_beta_dtype<T>(
    lhs_value: T,
    rhs_value: T,
    initial_even: T,
    initial_odd: T,
    alpha: T,
    beta: T,
) where
    T: DenseBlockScalar + DenseRecouplingScalar + RecouplingCoefficientAction<f64> + Debug,
{
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule);
    let key_for_sector = |sector| {
        keys.iter()
            .find(|key| key.codomain_tree().coupled() == Some(sector))
            .cloned()
            .expect("Z2 one-leg homspace contains requested sector")
    };
    let even_key = key_for_sector(even);
    let odd_key = key_for_sector(odd);

    let lhs_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        homspace.clone(),
        packed_fixture_structure(2, [(even_key.clone(), vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let rhs_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        homspace.clone(),
        packed_fixture_structure(2, [(even_key.clone(), vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let dst_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        homspace,
        packed_fixture_structure(2, [(even_key, vec![1, 1]), (odd_key, vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();

    let lhs = TensorMap::<T, 1, 1>::from_vec_with_fusion_space(vec![lhs_value], lhs_space).unwrap();
    let rhs = TensorMap::<T, 1, 1>::from_vec_with_fusion_space(vec![rhs_value], rhs_space).unwrap();
    let mut dst = TensorMap::<T, 1, 1>::from_vec_with_fusion_space(
        vec![initial_even, initial_odd],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            beta * initial_even + alpha * lhs_value * rhs_value,
            beta * initial_odd
        ]
    );
}

#[test]
fn tensorcontract_fusion_lowers_lhs_categorical_adjoint_lazily() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let space = || {
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap()
    };
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0)],
        space(),
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 2.0), Complex64::new(7.0, -2.0)],
        space(),
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(10.0, 0.0), Complex64::new(20.0, 0.0)],
        space(),
    )
    .unwrap();
    let axes =
        TensorContractSpec::with_default_output_order_and_conjugation(&[0], &[0], true, false);

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[Complex64::new(12.0, -1.0), Complex64::new(23.0, 1.0)]
    );

    dst.data_mut()
        .copy_from_slice(&[Complex64::new(10.0, 0.0), Complex64::new(20.0, 0.0)]);
    let mut context = TensorContractFusionExecutionContext::<Complex64, _>::default();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            axes,
            Complex64::one(),
            Complex64::zero(),
        )
        .unwrap();
    assert_eq!(
        dst.data(),
        &[Complex64::new(12.0, -1.0), Complex64::new(23.0, 1.0)]
    );
}

#[test]
fn tensorcontract_fusion_lowers_rhs_categorical_adjoint_lazily() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let space = || {
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap()
    };
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0)],
        space(),
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 2.0), Complex64::new(7.0, -2.0)],
        space(),
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(), Complex64::zero()],
        space(),
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[1], false, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[Complex64::new(12.0, 1.0), Complex64::new(23.0, -1.0)]
    );
}

#[test]
fn tensorcontract_fusion_lowers_both_categorical_adjoint_inputs_lazily() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let space = || {
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap()
    };
    let lhs_space = space();
    let rhs_space = space();
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        dst_hom,
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0)],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 2.0), Complex64::new(7.0, -2.0)],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(), Complex64::zero()],
        dst_space,
    )
    .unwrap();
    let axes =
        TensorContractSpec::with_default_output_order_and_conjugation(&[0], &[1], true, true);

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[Complex64::new(8.0, -9.0), Complex64::new(19.0, 13.0)]
    );
}

fn z2_matrix_homspace() -> FusionTreeHomSpace {
    let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
}

fn z2_matrix_space_with_homspace(
    homspace: FusionTreeHomSpace,
    block_shape: Vec<usize>,
) -> FusionTensorMapSpace<1, 1> {
    FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        homspace,
        &Z2FusionRule,
        [block_shape.clone(), block_shape],
    )
    .unwrap()
}

fn fermion_parity_matrix_homspace() -> FusionTreeHomSpace {
    let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
}

fn fermion_parity_matrix_space_with_homspace(
    homspace: FusionTreeHomSpace,
    block_shape: Vec<usize>,
) -> FusionTensorMapSpace<1, 1> {
    FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        homspace,
        &FermionParityFusionRule,
        [block_shape.clone(), block_shape],
    )
    .unwrap()
}

#[test]
fn tensorcontract_fusion_lhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = Z2FusionRule;
    let lhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let rhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = z2_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((2,), (1,)), pB=((1,), (2,)).
        TensorContractSpec::with_default_output_order_and_conjugation(&[0], &[0], true, false),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(16.0, -33.0),
            Complex64::new(44.0, -8.0),
            Complex64::new(35.0, -15.0),
            Complex64::new(41.0, 24.0),
            Complex64::new(6.0, -13.0),
            Complex64::new(-3.0, -4.0),
            Complex64::new(18.0, 10.0),
            Complex64::new(-10.0, 4.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_fermion_lhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = FermionParityFusionRule;
    let lhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let rhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = fermion_parity_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((2,), (1,)), pB=((1,), (2,)).
        TensorContractSpec::with_default_output_order_and_conjugation(&[0], &[0], true, false),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(16.0, -33.0),
            Complex64::new(44.0, -8.0),
            Complex64::new(35.0, -15.0),
            Complex64::new(41.0, 24.0),
            Complex64::new(6.0, -13.0),
            Complex64::new(-3.0, -4.0),
            Complex64::new(18.0, 10.0),
            Complex64::new(-10.0, 4.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_rhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = Z2FusionRule;
    let lhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let rhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = z2_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[1], false, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(14.0, -1.0),
            Complex64::new(34.0, 6.0),
            Complex64::new(17.0, -1.0),
            Complex64::new(43.0, 10.0),
            Complex64::new(4.0, 3.0),
            Complex64::new(14.0, -6.0),
            Complex64::new(-10.0, 14.0),
            Complex64::new(-8.0, 6.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_fermion_rhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = FermionParityFusionRule;
    let lhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let rhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = fermion_parity_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((1,), (2,)), pB=((2,), (1,)).
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[1], false, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(14.0, -1.0),
            Complex64::new(34.0, 6.0),
            Complex64::new(17.0, -1.0),
            Complex64::new(43.0, 10.0),
            Complex64::new(4.0, 3.0),
            Complex64::new(14.0, -6.0),
            Complex64::new(-10.0, 14.0),
            Complex64::new(-8.0, 6.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_both_adjoint_uses_degeneracy_matrix_contract() {
    let rule = Z2FusionRule;
    let lhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let rhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = z2_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order_and_conjugation(&[0], &[1], true, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(23.0, -18.0),
            Complex64::new(33.0, 11.0),
            Complex64::new(31.0, -25.0),
            Complex64::new(44.0, 15.0),
            Complex64::new(-6.0, -15.0),
            Complex64::new(1.0, -4.0),
            Complex64::new(13.0, 7.0),
            Complex64::new(-5.0, -11.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_fermion_both_adjoint_uses_degeneracy_matrix_contract() {
    let rule = FermionParityFusionRule;
    let lhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let rhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = fermion_parity_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((2,), (1,)), pB=((2,), (1,)).
        TensorContractSpec::with_default_output_order_and_conjugation(&[0], &[1], true, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(23.0, -18.0),
            Complex64::new(33.0, 11.0),
            Complex64::new(31.0, -25.0),
            Complex64::new(44.0, 15.0),
            Complex64::new(-6.0, -15.0),
            Complex64::new(1.0, -4.0),
            Complex64::new(13.0, 7.0),
            Complex64::new(-5.0, -11.0),
        ]
    );
}

#[test]
fn tensorproduct_fusion_lowers_lhs_adjoint_through_source_transform() {
    let rule = Z2FusionRule;
    let sector = SectorId::new(0);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 0>::from_dims([1], []).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], false)]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        ),
        &rule,
        [vec![1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 0>::from_dims([1], []).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], false)]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        ),
        &rule,
        [vec![1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], true)]),
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let lhs: TensorMap<Complex64, 1, 0> =
        TensorMap::from_vec_with_fusion_space(vec![Complex64::new(2.0, 1.0)], lhs_space).unwrap();
    let rhs: TensorMap<Complex64, 1, 0> =
        TensorMap::from_vec_with_fusion_space(vec![Complex64::new(3.0, -1.0)], rhs_space).unwrap();
    let mut dst: TensorMap<Complex64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![Complex64::new(0.0, 0.0)], dst_space).unwrap();

    tensorproduct_fusion_into_with_conjugation(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        OutputAxisOrder::identity(),
        true,
        false,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(dst.data(), &[Complex64::new(5.0, -5.0)]);

    dst.data_mut().copy_from_slice(&[Complex64::new(0.0, 0.0)]);
    let mut context = TensorContractFusionExecutionContext::<Complex64, _>::default();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            TensorContractSpec::new_with_conjugation(
                &[],
                &[],
                OutputAxisOrder::identity(),
                true,
                false,
            ),
            Complex64::one(),
            Complex64::zero(),
        )
        .unwrap();
    assert_eq!(dst.data(), &[Complex64::new(5.0, -5.0)]);
}

#[test]
fn tensorcontract_fusion_fermion_rhs_dual_codomain_twists_like_tensorkit() {
    let rule = FermionParityFusionRule;
    let odd = SectorId::new(1);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], true)]),
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let lhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![3.0], rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![10.0], dst_space).unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap();
    assert_eq!(
        specs,
        vec![TensorContractBlockSpec::with_coefficient(0, 0, 0, -1.0)]
    );

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[-6.0]);
}

#[test]
fn tensorcontract_fusion_fermion_twist_deg2_matches_tensorkit_reference() {
    // TensorKit @tensor reference (Julia crosscheck 2026-07-04):
    // V = Vect[FermionParity](0 => 1, 1 => 2); A :: V <- V'; B :: V' <- V
    // A blocks: even [0.5], odd [1.5, 2.5, 3.5, 4.5] (col-major)
    // B blocks: even [-1.25], odd [-0.75, -0.25, 0.25, 0.75]
    // C = @tensor A[a; c] * B[c; b]:
    //   even [-0.625], odd [2.0, 3.0, -3.0, -4.0]  (= -1 * A_odd * B_odd)
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let space = |codomain_dual: bool, domain_dual: bool| {
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([3], [3]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 2)], codomain_dual)]),
                FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 2)], domain_dual)]),
            ),
            &rule,
            [vec![1, 1], vec![2, 2]],
        )
        .unwrap()
    };
    let lhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        vec![0.5, 1.5, 2.5, 3.5, 4.5],
        space(false, true),
    )
    .unwrap();
    let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        vec![-1.25, -0.75, -0.25, 0.25, 0.75],
        space(true, false),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![0.0; 5], space(false, false))
            .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        1.0,
        0.0,
    )
    .unwrap();

    let expected = [-0.625, 2.0, 3.0, -3.0, -4.0];
    for (index, (&actual, &want)) in dst.data().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1.0e-12,
            "element {index}: got {actual}, TensorKit reference {want}"
        );
    }

    dst.data_mut().fill(0.0);
    let mut context = TensorContractFusionExecutionContext::<f64, _>::default();
    let mut profile = TensorContractFusionProfile::default();
    context
        .tensorcontract_fusion_into_profiled(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1.0,
            0.0,
            &mut profile,
        )
        .unwrap();

    // What: an already-core LHS is borrowed while the twist-bearing RHS is materialized.
    assert_eq!(profile.route, TensorContractFusionRoute::DynamicTreeCore);
    assert_eq!(profile.lhs_transform_calls, 0);
    assert_eq!(profile.rhs_transform_calls, 1);
    for (index, (&actual, &want)) in dst.data().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1.0e-12,
            "profiled element {index}: got {actual}, TensorKit reference {want}"
        );
    }
}

#[test]
fn tensorcontract_fusion_block_specs_enumerates_su2_innerline_blocks_from_homspace() {
    let rule = SU2FusionRule;
    let half = SectorId::new(1);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]),
        &rule,
        [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([(1, 1)], [(1, 1)]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
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
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let dst_key0 = src_key.clone();
    let dst_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &rule,
        [1, 1, 1, 1],
        Some(0),
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
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
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
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
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
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert_eq!(context.contraction_resolution_cache_hits(), 0);
    assert!(context.contraction_resolution_cache_misses() >= 1);

    let mut automatic_context_dst = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![1.0, 2.0],
        expected_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut automatic_context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
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
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }
    assert!(automatic_context.contraction_resolution_cache_len() >= 1);
    assert_eq!(automatic_context.contraction_resolution_cache_hits(), 0);
    assert!(automatic_context.contraction_resolution_cache_misses() >= 1);

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
    assert!(automatic_context.contraction_resolution_cache_len() >= 1);
    assert!(automatic_context.contraction_resolution_cache_hits() >= 1);
    assert!(automatic_context.contraction_resolution_cache_fast_hits() >= 1);
    assert!(automatic_context.contraction_resolution_cache_misses() >= 1);
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
fn contracted_fusion_tree_basis_matches_dual_u1_labels_and_flags() {
    let rule = U1FusionRule;
    let plus_two = U1Irrep::new(2).sector_id();
    let minus_two = U1Irrep::new(-2).sector_id();
    let lhs_domain = FusionTreeKey::try_new_for_rule(
        &rule,
        [plus_two],
        Some(plus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    )
    .unwrap();
    let rhs_codomain = FusionTreeKey::try_new_for_rule(
        &rule,
        [minus_two],
        Some(minus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    )
    .unwrap();
    assert!(contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &rhs_codomain
    ));

    let raw_rhs_codomain = FusionTreeKey::try_new_for_rule(
        &rule,
        [plus_two],
        Some(plus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    )
    .unwrap();
    assert!(!contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &raw_rhs_codomain
    ));

    let dual_flag_rhs_codomain = FusionTreeKey::try_new_for_rule(
        &rule,
        [minus_two],
        Some(minus_two),
        [true],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    )
    .unwrap();
    assert!(!contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &dual_flag_rhs_codomain
    ));
}

#[test]
fn tensorcontract_fusion_non_core_form_su2_absorbs_explicit_transform_sequence() {
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
    let lhs_core_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_core_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_core_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_core_hom,
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
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
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
    assert_eq!(context.tree_context().cache().plan_len(), 2);
    assert_eq!(context.tree_context().cache().structure_len(), 2);
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert_eq!(context.contraction_resolution_cache_hits(), 0);
    assert!(context.contraction_resolution_cache_misses() >= 1);

    let pinned = context
        .prepare_tensorcontract_fusion(&rule, &context_dst, &lhs, &rhs, axes)
        .unwrap();
    context.set_cache_policy(OperationCachePolicy::NoCache);
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
    assert_eq!(context.dynamic_fusion_space_cache_len(), 0);
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
    assert_eq!(context.tree_context().cache().plan_len(), 2);
    assert_eq!(context.tree_context().cache().structure_len(), 2);
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert!(context.contraction_resolution_cache_hits() >= 1);
    assert!(context.contraction_resolution_cache_fast_hits() >= 1);
    assert!(context.contraction_resolution_cache_misses() >= 1);

    let mut automatic_context_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context_replay.clone(),
        context_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut automatic_context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
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
    assert!(automatic_context.tree_context().cache().plan_len() > 0);
    assert!(automatic_context.tree_context().cache().structure_len() > 0);
    assert!(automatic_context.dynamic_fusion_space_cache_len() > 0);
    assert!(automatic_context.contraction_resolution_cache_len() >= 1);
    assert!(automatic_context.contraction_resolution_cache_len() > 0);
    assert!(automatic_context.dynamic_fusion_space_cache_misses() > 0);
    assert_eq!(automatic_context.dynamic_fusion_space_cache_hits(), 0);
    assert!(
        automatic_context
            .tree_context()
            .cache()
            .stats()
            .plan_misses()
            > 0
    );
    assert!(
        automatic_context
            .tree_context()
            .cache()
            .stats()
            .structure_misses()
            > 0
    );
    assert!(automatic_context.contraction_resolution_cache_len() >= 1);
    assert_eq!(automatic_context.contraction_resolution_cache_hits(), 0);
    assert!(automatic_context.contraction_resolution_cache_misses() >= 1);

    let mut no_cache_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context_replay.clone(),
        context_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut no_cache_context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
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
        assert_eq!(no_cache_context.tree_context().cache().plan_len(), 0);
        assert_eq!(no_cache_context.tree_context().cache().structure_len(), 0);
        assert_eq!(no_cache_context.dynamic_fusion_space_cache_len(), 0);
        assert_eq!(no_cache_context.dynamic_fusion_space_cache_hits(), 0);
        assert_eq!(no_cache_context.dynamic_fusion_space_cache_fast_hits(), 0);
        let dynamic_misses = no_cache_context.dynamic_fusion_space_cache_misses();
        assert!(dynamic_misses > previous_dynamic_misses);
        previous_dynamic_misses = dynamic_misses;
        assert_eq!(no_cache_context.contraction_resolution_cache_len(), 0);
        assert_eq!(no_cache_context.contraction_resolution_cache_len(), 0);
        assert_eq!(no_cache_context.contraction_resolution_cache_len(), 0);
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
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    warm_policy_context
        .tensorcontract_fusion_into(&rule, &mut warm_policy_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
    assert!(warm_policy_context.dynamic_fusion_space_cache_len() > 1);
    warm_policy_context.set_cache_policy(OperationCachePolicy::task_local_lru(1));
    assert!(warm_policy_context.tree_context().cache().plan_len() <= 1);
    assert!(warm_policy_context.tree_context().cache().structure_len() <= 1);
    assert!(warm_policy_context.dynamic_fusion_space_cache_len() <= 1);
    assert!(warm_policy_context.contraction_resolution_cache_len() <= 1);
    assert!(warm_policy_context.contraction_resolution_cache_len() <= 1);
    assert!(warm_policy_context.contraction_resolution_cache_len() <= 1);

    let mut lru_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context_replay.clone(),
        context_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut lru_context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
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
    assert!(lru_context.tree_context().cache().plan_len() <= 1);
    assert!(lru_context.tree_context().cache().structure_len() <= 1);
    assert!(lru_context.dynamic_fusion_space_cache_len() <= 1);
    assert!(lru_context.contraction_resolution_cache_len() <= 1);
    assert!(lru_context.contraction_resolution_cache_len() <= 1);
    assert!(lru_context.contraction_resolution_cache_len() <= 1);

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
    assert!(automatic_context.contraction_resolution_cache_len() >= 1);
    assert!(automatic_context.contraction_resolution_cache_hits() >= 1);
    assert!(automatic_context.contraction_resolution_cache_fast_hits() >= 1);
    assert!(automatic_context.contraction_resolution_cache_misses() >= 1);

    automatic_context_dst
        .data_mut()
        .copy_from_slice(&initial_dst_for_context_replay);
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
    for (&actual, &expected) in automatic_context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    assert_eq!(profile.route, TensorContractFusionRoute::DynamicTreeCore);
    assert_eq!(profile.lhs_transform_calls, 1);
    assert_eq!(profile.rhs_transform_calls, 1);
    assert_eq!(profile.output_transform_calls, 0);
    assert!(profile.core_contract_groups > 0);
    assert_eq!(profile.tree_replay.cache_lookup.as_nanos(), 0);
    assert_eq!(profile.tree_replay.strided_view_setup.as_nanos(), 0);
    assert_eq!(profile.tree_replay.multi_dense_view_setup.as_nanos(), 0);
    assert_eq!(profile.tree_replay.multi_dense_matmul_call.as_nanos(), 0);
    assert_eq!(
        profile.tree_replay.multi_matmul_total,
        profile.tree_replay.multi_scalar_recoupling
    );
    assert!(profile.tree_replay.multi_blocks > 0);
    assert!(profile.tree_replay.packed_columns > 0);
    assert_eq!(
        profile.tree_replay.packed_columns,
        profile.tree_replay.scattered_columns
    );
}

#[test]
fn tensorcontract_fusion_granular_caches_handle_block_structure_variants() {
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
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom,
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
    let rhs_data = (0..32)
        .map(|index| -2.0 + 0.125 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = vec![0.5, -1.0, 2.0, -4.0];
    let alpha = 0.75;
    let beta = -0.25;
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

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
        tensorcontract_fusion_into(&rule, &mut expected, &lhs, &rhs, axes, alpha, beta).unwrap();

        let mut actual = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space.clone(),
        )
        .unwrap();
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
    assert!(context.tree_context().cache().stats().structure_misses() >= 3);
    assert!(context.dynamic_fusion_space_cache_hits() > 0);
    assert!(context.dynamic_fusion_space_cache_fast_hits() > 0);
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert!(context.contraction_resolution_cache_hits() >= 1);
    assert!(context.contraction_resolution_cache_misses() >= 1);
}

#[test]
fn tensorcontract_fusion_granular_caches_handle_output_axes() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2), (1, 2), (1, 2)], [(1, 2)]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2)], [(1, 2), (1, 2), (1, 2)]);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        lhs_hom.clone(),
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
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
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

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
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
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

        tensorcontract_fusion_into(&rule, &mut expected, &lhs, &rhs, axes, alpha, beta).unwrap();
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
    assert!(context.tree_context().cache().stats().structure_misses() > 0);
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert!(context.contraction_resolution_cache_misses() >= 1);
}

#[test]
fn tensorcontract_fusion_granular_caches_distinguish_source_conjugation() {
    let rule = SU2FusionRule;
    let mut context = TensorContractFusionExecutionContext::<
        Complex64,
        TreeTransformBuiltinRuleCacheKey,
    >::default();
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
        let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
            lhs_hom.clone(),
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
            rhs_hom.clone(),
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
        let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
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
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
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

        tensorcontract_fusion_into(&rule, &mut expected, &lhs, &rhs, axes, alpha, beta).unwrap();
        context
            .tensorcontract_fusion_into(&rule, &mut actual, &lhs, &rhs, axes, alpha, beta)
            .unwrap();
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (actual - expected).norm() < 1.0e-10,
                "actual {actual} expected {expected}"
            );
        }
        assert!(context.contraction_resolution_cache_len() >= 1);
        assert!(context.contraction_resolution_cache_len() >= 1);
        assert!(context.contraction_resolution_cache_misses() >= 1);
        assert_eq!(context.contraction_resolution_cache_hits(), 0);
    }
}

#[test]
fn tensorcontract_fusion_non_core_form_su2_lhs_adjoint_prepared_plan_matches_reference_sequence() {
    assert_non_core_form_su2_adjoint_prepared_plan_matches_reference_sequence(
        su2_three_to_one_homspace(false, false),
        su2_one_to_three_homspace(true, true),
        true,
        false,
    );
}

#[test]
fn tensorcontract_fusion_non_core_form_su2_rhs_adjoint_prepared_plan_matches_reference_sequence() {
    // What: the reference-sequence assertion counts prepared-plan cache
    // activity, which shared-global-map promotions from concurrent sibling
    // tests can shift (the #174 reset-race family; reproduces under default
    // parallel --lib runs on wide machines). Serialize with the shared lock.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_non_core_form_su2_adjoint_prepared_plan_matches_reference_sequence(
        su2_three_to_one_homspace(false, true),
        su2_one_to_three_homspace(false, true),
        false,
        true,
    );
}

#[test]
fn tensorcontract_fusion_non_core_form_su2_both_adjoint_prepared_plan_matches_reference_sequence() {
    assert_non_core_form_su2_adjoint_prepared_plan_matches_reference_sequence(
        su2_three_to_one_homspace(false, false),
        su2_one_to_three_homspace(false, false),
        true,
        true,
    );
}

fn assert_non_core_form_su2_adjoint_prepared_plan_matches_reference_sequence(
    lhs_hom: FusionTreeHomSpace,
    rhs_hom: FusionTreeHomSpace,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
) {
    let rule = SU2FusionRule;
    let source_lhs_contracting_axes = [0, 1, 2];
    let source_rhs_contracting_axes = [1, 2, 3];
    let axes = TensorContractSpec::with_default_output_order_and_conjugation(
        &source_lhs_contracting_axes,
        &source_rhs_contracting_axes,
        lhs_conjugate,
        rhs_conjugate,
    );

    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        lhs_hom.clone(),
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom.clone(),
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let lowered_lhs_axes = maybe_adjoint_axes::<3, 1>(&source_lhs_contracting_axes, lhs_conjugate);
    let lowered_rhs_axes = maybe_adjoint_axes::<1, 3>(&source_rhs_contracting_axes, rhs_conjugate);
    let lowered_lhs_open_axes = complement_axes(4, &lowered_lhs_axes);
    let lowered_rhs_open_axes = complement_axes(4, &lowered_rhs_axes);
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
    let lhs_core_hom = effective_lhs_hom
        .permute(
            &rule,
            lowered_lhs_open_axes.as_slice(),
            lowered_lhs_axes.as_slice(),
        )
        .unwrap();
    let rhs_core_hom = effective_rhs_hom
        .permute(
            &rule,
            lowered_rhs_axes.as_slice(),
            lowered_rhs_open_axes.as_slice(),
        )
        .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let lhs_core_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_core_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_core_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_core_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let lhs_data = (0..32)
        .map(|index| Complex64::new(1.0 + 0.125 * index as f64, -0.5 + 0.0625 * index as f64))
        .collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| Complex64::new(-3.0 + 0.25 * index as f64, 0.75 - 0.03125 * index as f64))
        .collect::<Vec<_>>();
    let initial_dst = vec![
        Complex64::new(2.0, -1.0),
        Complex64::new(-1.0, 0.5),
        Complex64::new(4.0, 2.0),
        Complex64::new(-3.0, -0.25),
    ];
    let initial_dst_for_context = initial_dst.clone();
    let alpha = Complex64::new(-1.5, 0.25);
    let beta = Complex64::new(0.25, -0.125);
    let lhs =
        TensorMap::<Complex64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs =
        TensorMap::<Complex64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut expected_dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        initial_dst.clone(),
        dst_space.clone(),
    )
    .unwrap();
    let mut lhs_core = TensorMap::<Complex64, 1, 3>::from_vec_with_fusion_space(
        vec![Complex64::zero(); lhs_core_space.required_len().unwrap()],
        lhs_core_space.clone(),
    )
    .unwrap();
    let mut rhs_core = TensorMap::<Complex64, 3, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); rhs_core_space.required_len().unwrap()],
        rhs_core_space.clone(),
    )
    .unwrap();

    tensoradd_fusion_into(
        &rule,
        &mut lhs_core,
        &lhs,
        TreeTransformOperation::permute([3], [0, 1, 2]),
        lhs_conjugate,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    tensoradd_fusion_into(
        &rule,
        &mut rhs_core,
        &rhs,
        TreeTransformOperation::permute([1, 2, 3], [0]),
        rhs_conjugate,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
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

    let mut explicit_dst =
        TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(initial_dst, dst_space).unwrap();
    let plan = prepare_tensorcontract_fusion_plan(
        &rule,
        explicit_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(plan.lhs_source_conjugate(), lhs_conjugate);
    assert_eq!(plan.rhs_source_conjugate(), rhs_conjugate);
    assert_eq!(plan.core_axes().lhs_contracting_axes(), &[1, 2, 3]);
    assert_eq!(plan.core_axes().rhs_contracting_axes(), &[0, 1, 2]);

    let mut explicit_lhs_core = TensorMap::<Complex64, 1, 3>::from_vec_with_fusion_space(
        vec![Complex64::zero(); lhs_core_space.required_len().unwrap()],
        lhs_core_space,
    )
    .unwrap();
    let mut explicit_rhs_core = TensorMap::<Complex64, 3, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); rhs_core_space.required_len().unwrap()],
        rhs_core_space,
    )
    .unwrap();
    tensorcontract_fusion_prepared_into(
        &rule,
        &plan,
        &mut explicit_dst,
        &mut explicit_lhs_core,
        &mut explicit_rhs_core,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();

    for (&actual, &expected) in explicit_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).norm() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }

    let mut context_dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context.clone(),
        expected_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<
        Complex64,
        TreeTransformBuiltinRuleCacheKey,
    >::default();
    context
        .tensorcontract_fusion_into(&rule, &mut context_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).norm() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    let expects_dynamic_replay = !(lhs_conjugate && rhs_conjugate);
    if expects_dynamic_replay {
        assert!(context.tree_context().cache().stats().structure_misses() > 0);
        assert!(context.contraction_resolution_cache_len() >= 1);
        assert_eq!(context.contraction_resolution_cache_hits(), 0);
        assert!(context.contraction_resolution_cache_misses() >= 1);
    } else {
        assert_eq!(context.tree_context().cache().structure_len(), 0);
        assert!(context.contraction_resolution_cache_len() >= 1);
    }

    let tree_stats_after_first = context.tree_context().cache().stats();
    context_dst
        .data_mut()
        .copy_from_slice(&initial_dst_for_context);
    context
        .tensorcontract_fusion_into(&rule, &mut context_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).norm() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    if expects_dynamic_replay {
        assert_eq!(
            context.tree_context().cache().stats(),
            tree_stats_after_first
        );
        assert!(context.dynamic_fusion_space_cache_hits() > 0);
        assert!(context.dynamic_fusion_space_cache_fast_hits() > 0);
        assert!(context.contraction_resolution_cache_len() >= 1);
        assert!(context.contraction_resolution_cache_hits() >= 1);
        assert!(context.contraction_resolution_cache_fast_hits() >= 1);
        assert!(context.contraction_resolution_cache_misses() >= 1);
    } else {
        assert_eq!(context.tree_context().cache().structure_len(), 0);
        assert!(context.contraction_resolution_cache_len() >= 1);
    }
}

fn maybe_adjoint_axes<const NOUT: usize, const NIN: usize>(
    axes: &[usize],
    source_conjugate: bool,
) -> Vec<usize> {
    if source_conjugate {
        axes.iter()
            .map(|&axis| crate::lowering::adjoint_tensor_axis(NOUT, NIN, axis).unwrap())
            .collect()
    } else {
        axes.to_vec()
    }
}

fn complement_axes(rank: usize, axes: &[usize]) -> Vec<usize> {
    (0..rank).filter(|axis| !axes.contains(axis)).collect()
}

fn su2_three_to_one_homspace(codomain_dual: bool, domain_dual: bool) -> FusionTreeHomSpace {
    let half = SectorId::new(1);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(half, 2)], codomain_dual),
            SectorLeg::new([(half, 2)], codomain_dual),
            SectorLeg::new([(half, 2)], codomain_dual),
        ]),
        FusionProductSpace::new([SectorLeg::new([(half, 2)], domain_dual)]),
    )
}

fn su2_one_to_three_homspace(codomain_dual: bool, domain_dual: bool) -> FusionTreeHomSpace {
    let half = SectorId::new(1);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(half, 2)], codomain_dual)]),
        FusionProductSpace::new([
            SectorLeg::new([(half, 2)], domain_dual),
            SectorLeg::new([(half, 2)], domain_dual),
            SectorLeg::new([(half, 2)], domain_dual),
        ]),
    )
}

#[test]
fn tensorcontract_fusion_product_non_core_form_absorbs_explicit_transform() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([], []);
    let scalar_key = BlockKey::from(rhs_hom.fusion_tree_keys(&rule)[0].clone());
    let rhs_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        rhs_hom,
        packed_fixture_structure(0, [(scalar_key, vec![])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let lhs_core_hom = src_space
        .homspace()
        .permute(&rule, &[0, 1, 2], &[])
        .unwrap();
    let lhs_core_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 0>::from_dims([1, 1, 1], []).unwrap(),
        lhs_core_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let core_dst_space = lhs_core_space.clone();
    let rhs_core_space = rhs_space.clone();
    let lhs_data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)];
    let rhs_data = vec![Complex64::new(2.0, 0.5)];
    let initial_dst = vec![Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)];
    let lhs = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(lhs_data.clone(), src_space)
        .unwrap();
    let rhs =
        TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut dst = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        initial_dst.clone(),
        dst_space.clone(),
    )
    .unwrap();
    let mut expected_dst =
        TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(initial_dst, dst_space).unwrap();
    let axes = TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[1, 0, 2]));
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

    let plan = prepare_tensorcontract_fusion_plan(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    let mut lhs_core = TensorMap::<Complex64, 3, 0>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); lhs_core_space.required_len().unwrap()],
        lhs_core_space,
    )
    .unwrap();
    let mut rhs_core = TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); rhs_core_space.required_len().unwrap()],
        rhs_core_space,
    )
    .unwrap();
    let mut core_dst = TensorMap::<Complex64, 3, 0>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); core_dst_space.required_len().unwrap()],
        core_dst_space,
    )
    .unwrap();
    let alpha = Complex64::new(2.0, 0.0);
    let beta = Complex64::new(3.0, 0.0);
    tree_transform_into(
        &rule,
        plan.lhs_transform().clone(),
        &mut lhs_core,
        &lhs,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    tree_transform_into(
        &rule,
        plan.rhs_transform().clone(),
        &mut rhs_core,
        &rhs,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut core_dst,
        &lhs_core,
        &rhs_core,
        plan.core_axes().as_spec(),
        alpha,
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    tree_transform_into(
        &rule,
        plan.output_transform().clone(),
        &mut expected_dst,
        &core_dst,
        Complex64::new(1.0, 0.0),
        beta,
    )
    .unwrap();

    tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();

    for (&actual, &expected) in dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }

    let mut context_dst = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)],
        dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<Complex64, _>::default();
    context
        .tensorcontract_fusion_into(&rule, &mut context_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert_eq!(context.contraction_resolution_cache_hits(), 0);
    assert!(context.contraction_resolution_cache_misses() >= 1);

    context_dst
        .data_mut()
        .copy_from_slice(&[Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)]);
    context
        .tensorcontract_fusion_into(&rule, &mut context_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
    assert!(context.contraction_resolution_cache_len() >= 1);
    assert!(context.contraction_resolution_cache_hits() >= 1);
    assert!(context.contraction_resolution_cache_misses() >= 1);
}

#[test]
fn tensorcontract_fusion_product_fz2_u1_su2_contracts_component_channels_with_su2_recoupling() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let left_sector =
        |parity, charge| left_rule.encode_sector(parity, U1Irrep::new(charge).sector_id());
    let sector = |parity, charge, twice_spin| {
        rule.encode_sector(
            left_sector(parity, charge),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };
    let a = sector(odd, 1, 1);
    let b = sector(odd, -1, 1);
    let c0 = sector(even, 0, 0);
    let c1 = sector(even, 0, 2);

    let lhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(a, 1)], true),
            SectorLeg::new([(b, 1)], true),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
    );
    let rhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(c0, 1), (c1, 1)], false),
            SectorLeg::new([(a, 1)], false),
            SectorLeg::new([(b, 1)], false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(a, 1)], true),
            SectorLeg::new([(a, 1)], false),
            SectorLeg::new([(b, 1)], true),
            SectorLeg::new([(b, 1)], false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
        lhs_hom.clone(),
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 0>::from_dims([1, 1, 1], []).unwrap(),
        rhs_hom.clone(),
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        dst_hom.clone(),
        &rule,
        [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
    )
    .unwrap();
    let lhs_data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)];
    let rhs_data = vec![Complex64::new(-2.0, 0.5), Complex64::new(4.0, 3.0)];
    let initial_dst = vec![Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)];
    let lhs = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(lhs_data.clone(), lhs_space)
        .unwrap();
    let rhs = TensorMap::<Complex64, 3, 0>::from_vec_with_fusion_space(rhs_data.clone(), rhs_space)
        .unwrap();
    let mut dst =
        TensorMap::<Complex64, 4, 0>::from_vec_with_fusion_space(initial_dst.clone(), dst_space)
            .unwrap();
    let alpha = Complex64::new(2.0, -0.25);
    let beta = Complex64::new(-1.0, 0.5);
    let axes = TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 2, 1, 3]));

    tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();

    let expected = [
        Complex64::new(-29.12579386826373, -0.7876587736527441),
        Complex64::new(21.57892465101803, 3.5376587736527494),
    ];
    for (&actual, &expected) in dst.data().iter().zip(&expected) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }

    let expected_structure = std::sync::Arc::clone(dst.structure());
    let rebuild_and_contract = || {
        let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
            lhs_hom.clone(),
            &rule,
            [vec![1, 1, 1], vec![1, 1, 1]],
        )
        .unwrap();
        let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<3, 0>::from_dims([1, 1, 1], []).unwrap(),
            rhs_hom.clone(),
            &rule,
            [vec![1, 1, 1], vec![1, 1, 1]],
        )
        .unwrap();
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
            dst_hom.clone(),
            &rule,
            [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
        )
        .unwrap();
        let rebuilt_lhs =
            TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(lhs_data.clone(), lhs_space)
                .unwrap();
        let rebuilt_rhs =
            TensorMap::<Complex64, 3, 0>::from_vec_with_fusion_space(rhs_data.clone(), rhs_space)
                .unwrap();
        let mut rebuilt_dst = TensorMap::<Complex64, 4, 0>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space,
        )
        .unwrap();
        tensorcontract_fusion_into(
            &rule,
            &mut rebuilt_dst,
            &rebuilt_lhs,
            &rebuilt_rhs,
            axes,
            alpha,
            beta,
        )
        .unwrap();
        rebuilt_dst
    };

    force_fusion_layout_eviction();
    let after_eviction = rebuild_and_contract();
    assert_eq!(after_eviction.structure(), &expected_structure);
    for (&actual, &expected) in after_eviction.data().iter().zip(&expected) {
        assert!((actual - expected).norm() < 1.0e-12);
    }
    reset_global_operation_caches();
    let after_reset = rebuild_and_contract();
    assert_eq!(after_reset.structure(), &expected_structure);
    for (&actual, &expected) in after_reset.data().iter().zip(&expected) {
        assert!((actual - expected).norm() < 1.0e-12);
    }
}

#[test]
fn tensorcontract_fusion_product_no_twist_identity_rhs_is_borrowed() {
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let odd_charge = left_rule.encode_sector(SectorId::new(1), U1Irrep::new(0).sector_id());
    let odd = rule.encode_sector(odd_charge, SU2Irrep::from_twice_spin(0).sector_id());
    let matrix_hom = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]),
        )
    };
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        matrix_hom(),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        matrix_hom(),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(odd, 1)], false),
                SectorLeg::new([(odd, 1)], true),
            ]),
            FusionProductSpace::new([]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let lhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![3.0], rhs_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::from_vec_with_fusion_space(vec![5.0], dst_space).unwrap();
    let lhs_before = lhs.data().to_vec();
    let rhs_before = rhs.data().to_vec();
    let mut context = TensorContractFusionExecutionContext::<f64, _>::default();
    let mut profile = TensorContractFusionProfile::default();

    context
        .tensorcontract_fusion_into_profiled(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            2.0,
            3.0,
            &mut profile,
        )
        .unwrap();

    // What: only the output repartitions; the canonical nondual product RHS
    // is read directly while alpha and beta remain on their original writes.
    assert_eq!(profile.route, TensorContractFusionRoute::DynamicTreeCore);
    assert_eq!(profile.rhs_transform_calls, 0);
    assert_eq!(profile.output_transform_calls, 1);
    assert_eq!(dst.data(), &[27.0]);
    assert_eq!(lhs.data(), lhs_before);
    assert_eq!(rhs.data(), rhs_before);
}

fn copy_blocks_between_layouts(dst: &mut TensorMap<f64, 2, 2>, src: &TensorMap<f64, 2, 2>) {
    let dst_structure = std::sync::Arc::clone(dst.structure());
    let src_structure = std::sync::Arc::clone(src.structure());
    assert_eq!(dst_structure.block_count(), src_structure.block_count());
    for index in 0..src_structure.block_count() {
        let src_block = src_structure.block(index).unwrap();
        let dst_block = dst_structure.block(index).unwrap();
        assert_eq!(src_block.key(), dst_block.key());
        assert_eq!(src_block.shape(), dst_block.shape());
        let shape = src_block.shape().to_vec();
        let count = shape.iter().product::<usize>();
        let mut multi_index = vec![0usize; shape.len()];
        for _ in 0..count {
            let src_position = src_block.offset()
                + multi_index
                    .iter()
                    .zip(src_block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let dst_position = dst_block.offset()
                + multi_index
                    .iter()
                    .zip(dst_block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            dst.data_mut()[dst_position] = src.data()[src_position];
            for axis in 0..shape.len() {
                multi_index[axis] += 1;
                if multi_index[axis] < shape[axis] {
                    break;
                }
                multi_index[axis] = 0;
            }
        }
    }
}

fn assert_blocks_match(lhs: &TensorMap<f64, 2, 2>, rhs: &TensorMap<f64, 2, 2>) {
    let lhs_structure = std::sync::Arc::clone(lhs.structure());
    let rhs_structure = std::sync::Arc::clone(rhs.structure());
    assert_eq!(lhs_structure.block_count(), rhs_structure.block_count());
    for index in 0..lhs_structure.block_count() {
        let lhs_block = lhs_structure.block(index).unwrap();
        let rhs_block = rhs_structure.block(index).unwrap();
        assert_eq!(lhs_block.key(), rhs_block.key());
        assert_eq!(lhs_block.shape(), rhs_block.shape());
        let shape = lhs_block.shape().to_vec();
        let count = shape.iter().product::<usize>();
        let mut multi_index = vec![0usize; shape.len()];
        for _ in 0..count {
            let lhs_position = lhs_block.offset()
                + multi_index
                    .iter()
                    .zip(lhs_block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let rhs_position = rhs_block.offset()
                + multi_index
                    .iter()
                    .zip(rhs_block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let lhs_value = lhs.data()[lhs_position];
            let rhs_value = rhs.data()[rhs_position];
            assert!(
                (lhs_value - rhs_value).abs() < 1e-12,
                "block {index} element {multi_index:?}: {lhs_value} != {rhs_value}"
            );
            for axis in 0..shape.len() {
                multi_index[axis] += 1;
                if multi_index[axis] < shape[axis] {
                    break;
                }
                multi_index[axis] = 0;
            }
        }
    }
}

#[test]
fn coupled_layout_contraction_matches_packed_layout() {
    run_coupled_vs_packed_contractions(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
}

#[test]
fn coupled_layout_contraction_matches_packed_layout_su2() {
    run_coupled_vs_packed_contractions(
        &SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
}

fn run_coupled_vs_packed_contractions<R>(rule: &R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = TreeTransformBuiltinRuleCacheKey>,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let leg_dim = 2 * degeneracy;
    let dense =
        || TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap();
    let shapes =
        |hom: &FusionTreeHomSpace| vec![vec![degeneracy; 4]; hom.fusion_tree_keys(rule).len()];
    let packed_space = |hom: FusionTreeHomSpace| {
        let shape_list = shapes(&hom);
        FusionTensorMapSpace::from_degeneracy_shapes(dense(), hom, rule, shape_list).unwrap()
    };
    let coupled_space = |hom: FusionTreeHomSpace| {
        let shape_list = shapes(&hom);
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(dense(), hom, rule, shape_list)
            .unwrap()
    };

    let lhs_packed_space = packed_space(homspace());
    let lhs_len = lhs_packed_space.required_len().unwrap();
    let lhs_packed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_len).map(|i| (i % 11) as f64 * 0.5 - 2.0).collect(),
        lhs_packed_space,
    )
    .unwrap();
    let rhs_packed_space = packed_space(homspace());
    let rhs_len = rhs_packed_space.required_len().unwrap();
    let rhs_packed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..rhs_len).map(|i| (i % 7) as f64 * 0.25 - 1.0).collect(),
        rhs_packed_space,
    )
    .unwrap();

    let mut lhs_coupled = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; lhs_len],
        coupled_space(homspace()),
    )
    .unwrap();
    let mut rhs_coupled = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; rhs_len],
        coupled_space(homspace()),
    )
    .unwrap();
    copy_blocks_between_layouts(&mut lhs_coupled, &lhs_packed);
    copy_blocks_between_layouts(&mut rhs_coupled, &rhs_packed);

    let workloads: [(&[usize; 2], &[usize; 2], &[usize; 4]); 3] = [
        (&[2, 3], &[0, 1], &[0, 1, 2, 3]),
        (&[3, 2], &[0, 1], &[0, 1, 2, 3]),
        (&[3, 2], &[0, 1], &[1, 0, 2, 3]),
    ];
    for (lhs_axes, rhs_axes, output_axes) in workloads {
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs_packed.fusion_space().unwrap().homspace(),
            rhs_packed.fusion_space().unwrap().homspace(),
            lhs_axes,
            rhs_axes,
            output_axes,
            2,
        )
        .unwrap();
        let dst_packed_space = packed_space(dst_hom.clone());
        let dst_len = dst_packed_space.required_len().unwrap();
        let mut dst_packed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            vec![0.0; dst_len],
            dst_packed_space,
        )
        .unwrap();
        let mut dst_coupled = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            vec![0.0; dst_len],
            coupled_space(dst_hom),
        )
        .unwrap();

        let axes =
            || TensorContractSpec::new(lhs_axes, rhs_axes, OutputAxisOrder::from_axes(output_axes));
        let mut packed_context =
            TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default(
            );
        packed_context
            .tensorcontract_fusion_into(
                rule,
                &mut dst_packed,
                &lhs_packed,
                &rhs_packed,
                axes(),
                1.0,
                0.0,
            )
            .unwrap();
        let mut coupled_context =
            TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default(
            );
        coupled_context
            .tensorcontract_fusion_into(
                rule,
                &mut dst_coupled,
                &lhs_coupled,
                &rhs_coupled,
                axes(),
                1.0,
                0.0,
            )
            .unwrap();

        assert_blocks_match(&dst_packed, &dst_coupled);
    }
}

#[test]
fn coupled_layout_compose_uses_direct_gemm_groups() {
    let rule = Z2FusionRule;
    let degeneracy = 2usize;
    let leg = || {
        SectorLeg::new(
            [
                (SectorId::new(0), degeneracy),
                (SectorId::new(1), degeneracy),
            ],
            false,
        )
    };
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let leg_dim = 2 * degeneracy;
    let space = |hom: FusionTreeHomSpace| {
        let key_count = hom.fusion_tree_keys(&rule).len();
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
            hom,
            &rule,
            vec![vec![degeneracy; 4]; key_count],
        )
        .unwrap()
    };
    let lhs_space = space(homspace());
    let lhs_len = lhs_space.required_len().unwrap();
    let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_len).map(|i| i as f64 * 0.5).collect(),
        lhs_space,
    )
    .unwrap();
    let rhs_space = space(homspace());
    let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_len).map(|i| 1.0 - i as f64 * 0.25).collect(),
        rhs_space,
    )
    .unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs.fusion_space().unwrap().homspace(),
        rhs.fusion_space().unwrap().homspace(),
        &[2, 3],
        &[0, 1],
        &[0, 1, 2, 3],
        2,
    )
    .unwrap();
    let dst_space = space(dst_hom);
    let dst_len = dst_space.required_len().unwrap();
    let mut dst =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; dst_len], dst_space).unwrap();

    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    let axes =
        || TensorContractSpec::new(&[2, 3], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3]));
    let mut profile = TensorContractFusionProfile::default();
    context
        .tensorcontract_fusion_into_profiled(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            axes(),
            1.0,
            0.0,
            &mut profile,
        )
        .unwrap();

    assert!(profile.core_contract_groups > 0);
    assert_eq!(
        profile.core_direct_gemm_groups, profile.core_contract_groups,
        "coupled layout compose must GEMM directly into destination blocks"
    );
    // Pack/scatter no longer exist on the core route: replay is
    // direct-GEMM only, so the pack counters stay at their zero defaults.
    assert_eq!(profile.core_direct_pack_skips, 0);
    assert_eq!(profile.core_pack_lhs, std::time::Duration::ZERO);
    assert_eq!(profile.core_scatter, std::time::Duration::ZERO);
}

/// Parallel replay (threads=4, size gate dropped) must match the serial
/// default elementwise: swap+out exercises source and output tree
/// transforms — Singles for the abelian rules (U1, fZ2) and multi-tree
/// pack/GEMM/scatter blocks for SU2.
#[test]
fn tensorcontract_fusion_parallel_transform_replay_matches_serial() {
    fn run_case<R>(rule: &R, sectors: &[SectorId], degeneracy: usize)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
        R::Key: Clone + Eq + std::hash::Hash,
    {
        let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
        let leg_dim = sectors.len() * degeneracy;
        let homspace = || {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(), leg()]),
                FusionProductSpace::new([leg(), leg()]),
            )
        };
        let space = |hom: FusionTreeHomSpace| {
            let key_count = hom.fusion_tree_keys(rule).len();
            let dense =
                TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap();
            let shapes = vec![vec![degeneracy; 4]; key_count];
            FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, rule, shapes).unwrap()
        };

        let lhs_space = space(homspace());
        let rhs_space = space(homspace());
        let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            (0..lhs_space.required_len().unwrap())
                .map(|index| (index % 17) as f64 * 0.25 - 2.0)
                .collect(),
            lhs_space,
        )
        .unwrap();
        let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            (0..rhs_space.required_len().unwrap())
                .map(|index| (index % 13) as f64 * 0.5 - 3.0)
                .collect(),
            rhs_space,
        )
        .unwrap();

        // swap+out: C[b a; g h] = A[a b; c d] * B[d c; g h]
        let lhs_axes = [3usize, 2];
        let rhs_axes = [0usize, 1];
        let output_axes = [1usize, 0, 2, 3];
        let dst_space = || {
            let hom = FusionTreeHomSpace::tensorcontract_homspace(
                rule,
                lhs.fusion_space().unwrap().homspace(),
                rhs.fusion_space().unwrap().homspace(),
                &lhs_axes,
                &rhs_axes,
                &output_axes,
                2,
            )
            .unwrap();
            space(hom)
        };
        let axes = TensorContractSpec::new(
            &lhs_axes,
            &rhs_axes,
            OutputAxisOrder::from_axes(&output_axes),
        );

        let dst_len = dst_space().required_len().unwrap();
        let mut serial_dst =
            TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; dst_len], dst_space())
                .unwrap();
        let mut serial_context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
        serial_context
            .tensorcontract_fusion_into(rule, &mut serial_dst, &lhs, &rhs, axes.clone(), 1.0, 0.0)
            .unwrap();

        let mut parallel_dst =
            TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; dst_len], dst_space())
                .unwrap();
        let mut parallel_context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
        let backend = parallel_context.tree_context_mut().backend_mut();
        backend.set_recoupling_threads(4);
        backend.set_transform_parallel_min_len(0);
        // Two runs: cold (structure compile + replay) and warm (replay only).
        for _ in 0..2 {
            parallel_context
                .tensorcontract_fusion_into(
                    rule,
                    &mut parallel_dst,
                    &lhs,
                    &rhs,
                    axes.clone(),
                    1.0,
                    0.0,
                )
                .unwrap();
        }

        assert_eq!(parallel_dst.data(), serial_dst.data());
    }

    run_case(
        &U1FusionRule,
        &[
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(1).sector_id(),
        ],
        3,
    );
    run_case(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
        3,
    );
    run_case(
        &SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
            SU2Irrep::from_twice_spin(2).sector_id(),
        ],
        3,
    );
    let left_rule = FpU1Rule::default();
    let product_rule = FpU1Su2Rule::default();
    let left_sector =
        |parity, charge| left_rule.encode_sector(parity, U1Irrep::new(charge).sector_id());
    let product_sector = |parity, charge, twice_spin| {
        product_rule.encode_sector(
            left_sector(parity, charge),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };
    run_case(
        &product_rule,
        &[
            product_sector(SectorId::new(0), 0, 0),
            product_sector(SectorId::new(1), 1, 1),
            product_sector(SectorId::new(1), -1, 1),
        ],
        2,
    );
}

/// Parallel replay must preserve the fermionic twist sign: the deg-2 dual-leg
/// contraction from `tensorcontract_fusion_fermion_twist_deg2_matches_tensorkit_reference`
/// run through a threads=4 context against the TensorKit crosscheck values.
#[test]
fn tensorcontract_fusion_parallel_replay_keeps_fermion_twist_reference() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let space = |codomain_dual: bool, domain_dual: bool| {
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([3], [3]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 2)], codomain_dual)]),
                FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 2)], domain_dual)]),
            ),
            &rule,
            [vec![1, 1], vec![2, 2]],
        )
        .unwrap()
    };
    let lhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        vec![0.5, 1.5, 2.5, 3.5, 4.5],
        space(false, true),
    )
    .unwrap();
    let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        vec![-1.25, -0.75, -0.25, 0.25, 0.75],
        space(true, false),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![0.0; 5], space(false, false))
            .unwrap();

    let mut backend = DenseTreeTransformOperations::default_executor();
    backend.set_recoupling_threads(4);
    backend.set_transform_parallel_min_len(0);
    let mut workspace = TensorContractWorkspace::default();
    tensorcontract_fusion_into_with(
        &mut backend,
        &mut workspace,
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        1.0,
        0.0,
    )
    .unwrap();

    let expected = [-0.625, 2.0, 3.0, -3.0, -4.0];
    for (index, (&actual, &want)) in dst.data().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1.0e-12,
            "element {index}: got {actual}, TensorKit reference {want}"
        );
    }
}

// A lhs-conjugate (categorical-adjoint) contraction over a NON-self-dual (U(1))
// symmetry must equal the eager conjugate-transpose composed plainly. The other
// conjugate recipe tests are all self-dual (Z2 / fermion parity), where a charge
// equals its dual, so a sector-dualization mislabel is invisible; here charge +1
// and -1 are distinct duals. The Structure route mislabels the output coupled
// sector for this case, so it declines (`all_sectors_self_dual`) to the
// DynamicTree route for non-core shapes. This rank-(1,1) compose is core form,
// so the parent-storage matrix is consumed through the op-bearing Core batch.
#[test]
fn tensorcontract_fusion_u1_lhs_adjoint_matches_eager_conjugate_transpose() {
    use num_complex::Complex64;
    let rule = U1FusionRule;
    let c = |q: i32| U1Irrep::new(q).sector_id();
    // Non-self-dual charges: +1 and -1 are distinct duals of each other.
    let leg = || SectorLeg::new([(c(-1), 2), (c(1), 2)], false);
    let hom = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        )
    };
    let fusion = || {
        let h = hom();
        let count = h.fusion_tree_keys(&rule).len();
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([4], [4]).unwrap(),
            h,
            &rule,
            vec![vec![2, 2]; count],
        )
        .unwrap()
    };
    let lhs_fusion = fusion();
    let rhs_fusion = fusion();
    let lhs_space = crate::DynamicFusionMapSpace::from_typed(&lhs_fusion);
    let rhs_space = crate::DynamicFusionMapSpace::from_typed(&rhs_fusion);
    let len = lhs_space.required_len().unwrap();
    let mk = |seed: f64| {
        (0..len)
            .map(|i| Complex64::new(seed + i as f64, 1.0 + (i % 3) as f64 - seed))
            .collect::<Vec<_>>()
    };
    let lhs_data = mk(1.0);
    let rhs_data = mk(2.0);

    // Oracle: eager conjugate-transpose, then a plain (no-conjugate) contraction.
    let (adj_space, adj_data) = crate::adjoint::adjoint_dyn(&rule, &lhs_space, &lhs_data).unwrap();
    // compose(a†, b): contract a†'s domain axis (1) with b's codomain axis (0).
    let dst_space =
        crate::DynamicFusionMapSpace::contracted(&rule, &adj_space, &rhs_space, &[1], &[0])
            .unwrap();
    let provider = Arc::new(rule);
    let dst_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dst_space.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let adj_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(adj_space, Arc::clone(&provider))
            .unwrap();
    let lhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(lhs_space, Arc::clone(&provider))
            .unwrap();
    let rhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(rhs_space, Arc::clone(&provider))
            .unwrap();
    let mut oracle = vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
    let mut ctx = crate::TensorContractFusionExecutionContext::<Complex64, _>::default();
    ctx.tensorcontract_fusion_dyn_into(
        &dst_bound,
        &mut oracle,
        &adj_bound,
        &adj_data,
        &rhs_bound,
        &rhs_data,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    let lhs_operand =
        crate::FusionOperand::prelowered_adjoint(adj_bound.space(), lhs_bound.space()).unwrap();
    let rhs_operand = crate::FusionOperand::direct(rhs_bound.space());
    let axes = || {
        TensorContractSpec::new_with_conjugation(
            &[1],
            &[0],
            crate::OutputAxisOrder::identity(),
            true,
            false,
        )
    };
    for policy in [
        OperationCachePolicy::default(),
        OperationCachePolicy::NoCache,
    ] {
        let mut ctx = crate::TensorContractFusionExecutionContext::<Complex64, _>::default();
        ctx.set_cache_policy(policy);
        crate::lowering::reset_adjoint_view_build_count();
        for _ in 0..2 {
            let mut fold = vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
            ctx.tensorcontract_fusion_dyn_prelowered_into(
                &dst_bound,
                &mut fold,
                lhs_operand,
                &lhs_data,
                rhs_operand,
                &rhs_data,
                axes(),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            )
            .unwrap();
            assert!(
                ctx.last_resolution_is_core(),
                "core-form prelowered adjoint must resolve to Core"
            );
            let max = oracle
                .iter()
                .zip(&fold)
                .map(|(o, f)| (o - f).norm())
                .fold(0.0f64, f64::max);
            assert!(max < 1e-10, "fold vs eager oracle: max diff {max}");
        }
        if policy == OperationCachePolicy::default() {
            assert!(ctx.contraction_resolution_cache_fast_hits() >= 1);
            let prelowered_entries = ctx.contraction_resolution_cache_len();
            let mut ordinary = vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
            ctx.tensorcontract_fusion_dyn_into(
                &dst_bound,
                &mut ordinary,
                &lhs_bound,
                &lhs_data,
                &rhs_bound,
                &rhs_data,
                TensorContractSpec::new_with_conjugation(
                    &[0],
                    &[0],
                    crate::OutputAxisOrder::identity(),
                    true,
                    false,
                ),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            )
            .unwrap();
            assert_eq!(ordinary, oracle);
            assert!(
                !ctx.last_resolution_is_core(),
                "ordinary conjugation must not alias the parent-storage Core resolution"
            );
            assert!(ctx.contraction_resolution_cache_len() > prelowered_entries);
            crate::lowering::reset_adjoint_view_build_count();

            let mut folded_again =
                vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
            ctx.tensorcontract_fusion_dyn_prelowered_into(
                &dst_bound,
                &mut folded_again,
                lhs_operand,
                &lhs_data,
                rhs_operand,
                &rhs_data,
                axes(),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            )
            .unwrap();
            assert_eq!(folded_again, oracle);
            assert!(ctx.last_resolution_is_core());
        }
        // What: neither cold/reset execution nor NoCache may reconstruct the
        // full categorical adjoint view hidden behind the prelowered operand.
        assert_eq!(crate::lowering::adjoint_view_build_count(), 0);
    }
}

#[test]
fn tensorcontract_fusion_prelowered_fermion_twist_declines_core_and_matches_eager() {
    use num_complex::Complex64;

    let rule = FermionParityFusionRule;
    let odd = SectorId::new(1);
    let make_space = |codomain_dual: bool, domain_dual: bool| {
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], codomain_dual)]),
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], domain_dual)]),
        );
        crate::DynamicFusionMapSpace::from_degeneracy_shapes(&rule, hom, vec![vec![1, 1]]).unwrap()
    };
    let lhs_space = make_space(true, false);
    // The externally dual RHS codomain requires the fermionic supertrace twist.
    let rhs_space = make_space(true, false);
    let lhs_data = vec![Complex64::new(2.0, 3.0)];
    let rhs_data = vec![Complex64::new(5.0, -1.0)];
    let (adj_space, adj_data) = crate::adjoint::adjoint_dyn(&rule, &lhs_space, &lhs_data).unwrap();
    let dst_space =
        crate::DynamicFusionMapSpace::contracted(&rule, &adj_space, &rhs_space, &[1], &[0])
            .unwrap();
    let provider = Arc::new(rule);
    let dst_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dst_space.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let adj_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(adj_space, Arc::clone(&provider))
            .unwrap();
    let lhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(lhs_space, Arc::clone(&provider))
            .unwrap();
    let rhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(rhs_space, Arc::clone(&provider))
            .unwrap();
    let mut context = crate::TensorContractFusionExecutionContext::<Complex64, _>::default();
    let mut eager = vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
    context
        .tensorcontract_fusion_dyn_into(
            &dst_bound,
            &mut eager,
            &adj_bound,
            &adj_data,
            &rhs_bound,
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();

    let mut lazy = vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
    context
        .tensorcontract_fusion_dyn_prelowered_into(
            &dst_bound,
            &mut lazy,
            crate::FusionOperand::prelowered_adjoint(adj_bound.space(), lhs_bound.space()).unwrap(),
            &lhs_data,
            crate::FusionOperand::direct(rhs_bound.space()),
            &rhs_data,
            TensorContractSpec::new_with_conjugation(
                &[1],
                &[0],
                crate::OutputAxisOrder::identity(),
                true,
                false,
            ),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();

    assert_eq!(lazy, eager);
    assert!(
        !context.last_resolution_is_core(),
        "a nontrivial fermionic RHS twist must stay on a twist-aware fallback route"
    );
}

#[test]
fn nested_product_lowered_dynamic_execution_matches_independent_encoded_oracles() {
    // What: direct and lazy-adjoint contractions for a nested non-Abelian
    // product keep the encoded layout/data semantics with and without replay
    // caching, including source transforms and a nonidentity output transform.
    const ISOLATED_ENV: &str = "TENET_LOWERED_DYNAMIC_ORACLE_CHILD";
    if std::env::var_os(ISOLATED_ENV).is_none() {
        // What: cache resets used to make the two oracles independent cannot
        // change process-global cache generations observed by sibling tests.
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::contract_fusion::nested_product_lowered_dynamic_execution_matches_independent_encoded_oracles",
                "--nocapture",
            ])
            .env(ISOLATED_ENV, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    use tenet_core::{
        FermionParityFusionRule, Fz2SectorLayout, PackedProductCodec, ProductFusionRule,
        ProductSectorCodec, ProductSectorLayout, SU2FusionRule, SU2Irrep, Su2SectorLayout,
        U1FusionRule, U1Irrep, U1SectorLayout, Z2Irrep,
    };

    type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
    type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
    type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;
    type TripleRuleKey = <TripleRule as TreeTransformRuleCacheKey>::Key;

    let rule = TripleRule::new(
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    );
    let sector = |parity: u8, charge: i32, twice_spin: usize| {
        TripleCodec::encode(
            Fz2U1Codec::encode(
                Z2Irrep::new(parity).sector_id(),
                U1Irrep::new(charge).sector_id(),
            ),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };
    let vacuum = sector(0, 0, 0);
    let charged = sector(1, 0, 1);
    let leg = |dual| SectorLeg::new([(vacuum, 1), (charged, 1)], dual);
    let provider = Arc::new(rule);
    let bind_encoded = |homspace: FusionTreeHomSpace| {
        let count = homspace.fusion_tree_keys(provider.as_ref()).len();
        BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&provider),
            homspace,
            vec![vec![1; 3]; count],
        )
        .unwrap()
    };
    let bind_lowered = |homspace: FusionTreeHomSpace| {
        let count = homspace
            .try_fusion_tree_keys_lowered(provider.as_ref())
            .unwrap()
            .len();
        BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
            Arc::clone(&provider),
            homspace,
            vec![vec![1; 3]; count],
        )
        .unwrap()
    };
    let lhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false)]),
    );
    let rhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(true)]),
        FusionProductSpace::new([leg(true), leg(false)]),
    );
    let encoded_lhs = bind_encoded(lhs_hom.clone());
    let encoded_rhs = bind_encoded(rhs_hom.clone());
    let lowered_lhs = bind_lowered(lhs_hom);
    let lowered_rhs = bind_lowered(rhs_hom);
    assert_eq!(encoded_lhs.space(), lowered_lhs.space());
    assert_eq!(encoded_rhs.space(), lowered_rhs.space());
    let lhs_data = (0..encoded_lhs.space().required_len().unwrap())
        .map(|index| index as f64 + 1.0)
        .collect::<Vec<_>>();
    let rhs_data = (0..encoded_rhs.space().required_len().unwrap())
        .map(|index| 0.5 * index as f64 - 2.0)
        .collect::<Vec<_>>();
    let direct_axes =
        TensorContractSpec::new(&[0], &[2], OutputAxisOrder::from_axes(&[2, 0, 3, 1]));

    reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    let encoded_dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &encoded_lhs,
        &encoded_rhs,
        direct_axes.lhs_contracting_axes(),
        direct_axes.rhs_contracting_axes(),
        direct_axes.output_permutation(),
    )
    .unwrap();
    reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    let lowered_dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered_lowered(
        &lowered_lhs,
        &lowered_rhs,
        direct_axes.lhs_contracting_axes(),
        direct_axes.rhs_contracting_axes(),
        direct_axes.output_permutation(),
    )
    .unwrap();
    assert_eq!(encoded_dst.space(), lowered_dst.space());

    for policy in [
        OperationCachePolicy::NoCache,
        OperationCachePolicy::TaskLocal,
    ] {
        reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let mut encoded = vec![0.0; encoded_dst.space().required_len().unwrap()];
        let mut encoded_context =
            TensorContractFusionExecutionContext::<f64, TripleRuleKey>::default();
        encoded_context.set_cache_policy(policy);
        encoded_context
            .tensorcontract_fusion_dyn_into(
                &encoded_dst,
                &mut encoded,
                &encoded_lhs,
                &lhs_data,
                &encoded_rhs,
                &rhs_data,
                direct_axes,
                1.0,
                0.0,
            )
            .unwrap();

        reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let mut lowered = vec![0.0; lowered_dst.space().required_len().unwrap()];
        let mut lowered_context =
            TensorContractFusionExecutionContext::<f64, TripleRuleKey>::default();
        lowered_context.set_cache_policy(policy);
        lowered_context
            .tensorcontract_fusion_dyn_into_lowered(
                &lowered_dst,
                &mut lowered,
                &lowered_lhs,
                &lhs_data,
                &lowered_rhs,
                &rhs_data,
                direct_axes,
                1.0,
                0.0,
            )
            .unwrap();
        assert_eq!(lowered, encoded);
        let cold_misses = lowered_context.dynamic_fusion_space_cache_misses();
        let cold_hits = lowered_context.dynamic_fusion_space_cache_hits();
        assert!(cold_misses >= 3);
        let mut warm = vec![0.0; lowered.len()];
        lowered_context
            .tensorcontract_fusion_dyn_into_lowered(
                &lowered_dst,
                &mut warm,
                &lowered_lhs,
                &lhs_data,
                &lowered_rhs,
                &rhs_data,
                direct_axes,
                1.0,
                0.0,
            )
            .unwrap();
        assert_eq!(warm, lowered);
        if policy == OperationCachePolicy::NoCache {
            assert_eq!(lowered_context.dynamic_fusion_space_cache_len(), 0);
            assert_eq!(lowered_context.dynamic_fusion_space_cache_hits(), 0);
            assert!(lowered_context.dynamic_fusion_space_cache_misses() > cold_misses);
        } else {
            assert!(lowered_context.dynamic_fusion_space_cache_len() >= 3);
            assert_eq!(
                lowered_context.dynamic_fusion_space_cache_misses(),
                cold_misses
            );
            assert!(lowered_context.dynamic_fusion_space_cache_hits() > cold_hits);
        }
    }

    reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    let (eager_lhs, eager_lhs_data) = crate::adjoint_bound_dyn(&encoded_lhs, &lhs_data).unwrap();
    reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    let lazy_lhs = crate::adjoint_bound_space_dyn_lowered(&lowered_lhs).unwrap();
    assert_eq!(eager_lhs.space(), lazy_lhs.space());
    let lazy_axes = TensorContractSpec::new_with_conjugation(
        &[1],
        &[1],
        OutputAxisOrder::from_axes(&[2, 0, 3, 1]),
        true,
        false,
    );
    reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    let encoded_lazy_dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &eager_lhs,
        &encoded_rhs,
        lazy_axes.lhs_contracting_axes(),
        lazy_axes.rhs_contracting_axes(),
        lazy_axes.output_permutation(),
    )
    .unwrap();
    reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    let lazy_dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered_lowered(
        &lazy_lhs,
        &lowered_rhs,
        lazy_axes.lhs_contracting_axes(),
        lazy_axes.rhs_contracting_axes(),
        lazy_axes.output_permutation(),
    )
    .unwrap();
    assert_eq!(encoded_lazy_dst.space(), lazy_dst.space());

    reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    let mut eager = vec![0.0; lazy_dst.space().required_len().unwrap()];
    let mut eager_context = TensorContractFusionExecutionContext::<f64, TripleRuleKey>::default();
    eager_context
        .tensorcontract_fusion_dyn_into(
            &lazy_dst,
            &mut eager,
            &eager_lhs,
            &eager_lhs_data,
            &encoded_rhs,
            &rhs_data,
            TensorContractSpec::new(
                lazy_axes.lhs_contracting_axes(),
                lazy_axes.rhs_contracting_axes(),
                lazy_axes.output_permutation(),
            ),
            1.0,
            0.0,
        )
        .unwrap();

    for policy in [
        OperationCachePolicy::NoCache,
        OperationCachePolicy::TaskLocal,
    ] {
        reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let mut encoded_lazy = vec![0.0; encoded_lazy_dst.space().required_len().unwrap()];
        let mut encoded_lazy_context =
            TensorContractFusionExecutionContext::<f64, TripleRuleKey>::default();
        encoded_lazy_context.set_cache_policy(policy);
        encoded_lazy_context
            .tensorcontract_fusion_dyn_prelowered_into(
                &encoded_lazy_dst,
                &mut encoded_lazy,
                FusionOperand::prelowered_adjoint(eager_lhs.space(), encoded_lhs.space()).unwrap(),
                &lhs_data,
                FusionOperand::direct(encoded_rhs.space()),
                &rhs_data,
                lazy_axes,
                1.0,
                0.0,
            )
            .unwrap();
        assert_eq!(encoded_lazy, eager);

        reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let mut lazy = vec![0.0; lazy_dst.space().required_len().unwrap()];
        let mut lazy_context =
            TensorContractFusionExecutionContext::<f64, TripleRuleKey>::default();
        lazy_context.set_cache_policy(policy);
        let execute_lazy =
            |context: &mut TensorContractFusionExecutionContext<f64, TripleRuleKey>,
             output: &mut [f64]| {
                context.tensorcontract_fusion_dyn_prelowered_into_lowered(
                    &lazy_dst,
                    output,
                    FusionOperand::prelowered_adjoint(lazy_lhs.space(), lowered_lhs.space())
                        .unwrap(),
                    &lhs_data,
                    FusionOperand::direct(lowered_rhs.space()),
                    &rhs_data,
                    lazy_axes,
                    1.0,
                    0.0,
                )
            };
        execute_lazy(&mut lazy_context, &mut lazy).unwrap();
        assert_eq!(lazy, eager);
        let cold_misses = lazy_context.dynamic_fusion_space_cache_misses();
        let cold_hits = lazy_context.dynamic_fusion_space_cache_hits();
        assert!(cold_misses >= 3);
        let mut warm = vec![0.0; lazy.len()];
        execute_lazy(&mut lazy_context, &mut warm).unwrap();
        assert_eq!(warm, lazy);
        if policy == OperationCachePolicy::NoCache {
            assert_eq!(lazy_context.dynamic_fusion_space_cache_len(), 0);
            assert_eq!(lazy_context.dynamic_fusion_space_cache_hits(), 0);
            assert!(lazy_context.dynamic_fusion_space_cache_misses() > cold_misses);
        } else {
            assert!(lazy_context.dynamic_fusion_space_cache_len() >= 3);
            assert_eq!(
                lazy_context.dynamic_fusion_space_cache_misses(),
                cold_misses
            );
            assert!(lazy_context.dynamic_fusion_space_cache_hits() > cold_hits);
        }
    }
}
