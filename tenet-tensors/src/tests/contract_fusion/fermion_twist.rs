use super::*;

#[test]
fn tensorcontract_fusion_fermion_rhs_dual_codomain_twists_like_tensorkit() {
    let rule = FermionParityFusionRule;
    let odd = SectorId::new(1);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], true)]),
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
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

    fusion_contract_into(
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
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(
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
    let initial = [0.25, -0.5, 1.0, -1.5, 2.0];
    let alpha = 1.5;
    let beta = -0.25;
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial.to_vec(), space(false, false))
            .unwrap();
    let facts: Vec<crate::contract::FusionContractCandidateFacts> =
        crate::contract::prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
            &rule,
            &DynamicFusionMapSpace::from_typed(dst.fusion_space().unwrap()),
            &DynamicFusionMapSpace::from_typed(lhs.fusion_space().unwrap()),
            &DynamicFusionMapSpace::from_typed(rhs.fusion_space().unwrap()),
            TensorContractSpec::with_default_output_order(&[1], &[0]),
        )
        .unwrap();

    // What: the canonical RHS layout remains exactly borrowable before the
    // independently recorded fermionic twist forces its materialization.
    assert_eq!(facts.len(), 2);
    assert_eq!(
        facts[0].orientation(),
        crate::contract::FusionContractOrientation::LhsRhs
    );
    assert!(facts[0].lhs_exact_identity_borrowable());
    assert!(facts[0].rhs_exact_identity_borrowable());
    assert!(facts[0].rhs_requires_twist());
    assert_eq!(facts[0].lhs_materialized_elements(), 0);
    assert_eq!(facts[0].rhs_materialized_elements(), 5);
    assert_eq!(facts[0].output_materialized_elements(), 0);
    assert_eq!(facts[0].total_materialized_elements(), 5);
    assert_eq!(
        facts[1].orientation(),
        crate::contract::FusionContractOrientation::RhsLhs
    );
    assert!(!facts[1].rhs_requires_twist());

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        alpha,
        beta,
    )
    .unwrap();

    // What: an explicit dense core calculation with one odd-sector RHS twist
    // remains an independent numerical oracle, including destination axpby.
    let mut contracted = [0.0; 5];
    contracted[0] = lhs.data()[0] * rhs.data()[0];
    for col in 0..2 {
        for row in 0..2 {
            let mut sum = 0.0;
            for inner in 0..2 {
                let lhs_value = lhs.data()[1 + row + 2 * inner];
                let twisted_rhs_value = -rhs.data()[1 + inner + 2 * col];
                sum += lhs_value * twisted_rhs_value;
            }
            contracted[1 + row + 2 * col] = sum;
        }
    }
    assert_eq!(contracted, [-0.625, 2.0, 3.0, -3.0, -4.0]);
    let expected =
        std::array::from_fn::<_, 5, _>(|index| alpha * contracted[index] + beta * initial[index]);
    for (index, (&actual, &want)) in dst.data().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1.0e-12,
            "element {index}: got {actual}, TensorKit reference {want}"
        );
    }

    let dst_dynamic = DynamicFusionMapSpace::from_typed(dst.fusion_space().unwrap());
    let lhs_dynamic = DynamicFusionMapSpace::from_typed(lhs.fusion_space().unwrap());
    let rhs_dynamic = DynamicFusionMapSpace::from_typed(rhs.fusion_space().unwrap());
    let candidate = crate::contract::contracted_axis_order_candidates(&[1], &[0]).remove(0);
    let reverse_plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
        &rule,
        &dst_dynamic,
        &lhs_dynamic,
        &rhs_dynamic,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        &candidate,
        crate::contract::FusionContractOrientation::RhsLhs,
    )
    .unwrap();
    let mut reverse_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial.to_vec(),
        dst.fusion_space().unwrap().as_ref().clone(),
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
        &reverse_plan,
        &mut reverse_dst,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();
    for (index, (&actual, &want)) in reverse_dst.data().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1.0e-12,
            "reverse element {index}: got {actual}, TensorKit reference {want}"
        );
    }
    let mut artifact_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial.to_vec(),
        dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut profiled_artifact_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial.to_vec(),
        dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    crate::contract::execute_dynamic_tree_execution_artifact_profile_pair_for_test(
        &rule,
        &reverse_plan,
        &mut artifact_dst,
        &mut profiled_artifact_dst,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();
    assert_eq!(profiled_artifact_dst.data(), artifact_dst.data());
    for (index, (&actual, &want)) in artifact_dst.data().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1.0e-12,
            "artifact element {index}: got {actual}, TensorKit reference {want}"
        );
    }

    dst.data_mut().copy_from_slice(&initial);
    let mut context = TensorContractFusionExecutionContext::<f64, _>::default();
    let mut profile = TensorContractFusionProfile::default();
    context
        .tensorcontract_fusion_into_profiled(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            alpha,
            beta,
            &mut profile,
        )
        .unwrap();

    // What: the eager route folds the uniform twist into the core's per-job
    // alpha, so neither source is materialized.
    assert_eq!(profile.route, TensorContractFusionRoute::CoreFusionBlocks);
    assert_eq!(profile.lhs_transform_calls, 0);
    assert_eq!(profile.rhs_transform_calls, 0);
    for (index, (&actual, &want)) in dst.data().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1.0e-12,
            "profiled element {index}: got {actual}, TensorKit reference {want}"
        );
    }
}

#[test]
fn fermion_twist_folds_into_core_alpha_by_hand_calculation() {
    // Both operands are already in core form, so TensorKit's `blas_contract!`
    // (tensoroperations.jl:398-409 @cfaa073) copies and twists the smaller
    // one, A (5 elements), not B (10): C[a; b w] = sum_k A[a; k] θ_k
    // B[k; b w] with θ_odd = -1 on B's dual codomain leg V*. W is an even-only
    // leg, so no other sign enters and the result is a hand calculation.
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let v = |dual: bool| SectorLeg::new([(even, 1), (odd, 2)], dual);
    let w = || SectorLeg::new([(even, 2)], false);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([3], [3]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([v(false)]),
            FusionProductSpace::new([v(true)]),
        ),
        &rule,
        [vec![1, 1], vec![2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 2>::from_dims([3], [3, 2]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([v(true)]),
            FusionProductSpace::new([v(false), w()]),
        ),
        &rule,
        [vec![1, 1, 2], vec![2, 2, 2]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 2>::from_dims([3], [3, 2]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([v(false)]),
            FusionProductSpace::new([v(false), w()]),
        ),
        &rule,
        [vec![1, 1, 2], vec![2, 2, 2]],
    )
    .unwrap();
    // Dyadic values: every product and partial sum is exact.
    let lhs_data = vec![0.5, 1.5, -2.5, 3.5, 4.25];
    let rhs_data = vec![-1.25, 0.75, -0.75, -0.25, 0.25, 0.75, 1.5, -2.0, 0.5, 1.0];
    let lhs =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(lhs_data.clone(), lhs_space).unwrap();
    let rhs =
        TensorMap::<f64, 1, 2>::from_vec_with_fusion_space(rhs_data.clone(), rhs_space).unwrap();
    let initial = [0.25, -0.5, 1.0, -1.5, 2.0, 0.5, -0.25, 1.25, -1.0, 0.75];
    let (alpha, beta) = (1.5, -0.25);

    let mut contracted = [0.0; 10];
    for col in 0..2 {
        contracted[col] = lhs_data[0] * rhs_data[col];
    }
    for col in 0..4 {
        for row in 0..2 {
            contracted[2 + row + 2 * col] = (0..2)
                .map(|k| lhs_data[1 + row + 2 * k] * -rhs_data[2 + k + 2 * col])
                .sum();
        }
    }
    let expected: [f64; 10] =
        std::array::from_fn(|index| alpha * contracted[index] + beta * initial[index]);

    let axes = TensorContractSpec::with_default_output_order(&[1], &[0]);
    let facts: Vec<crate::contract::FusionContractCandidateFacts> =
        crate::contract::prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
            &rule,
            &DynamicFusionMapSpace::from_typed(&dst_space),
            &DynamicFusionMapSpace::from_typed(lhs.fusion_space().unwrap()),
            &DynamicFusionMapSpace::from_typed(rhs.fusion_space().unwrap()),
            axes,
        )
        .unwrap();
    // What: the selected LhsRhs candidate materializes only A for the twist
    // (before #1351: B, 10 elements).
    assert_eq!(
        facts[0].orientation(),
        crate::contract::FusionContractOrientation::LhsRhs
    );
    assert!(facts[0].lhs_exact_identity_borrowable());
    assert!(facts[0].rhs_exact_identity_borrowable());
    assert!(facts[0].lhs_requires_twist());
    assert!(!facts[0].rhs_requires_twist());
    assert_eq!(facts[0].lhs_materialized_elements(), 5);
    assert_eq!(facts[0].rhs_materialized_elements(), 0);
    assert_eq!(facts[0].total_materialized_elements(), 5);

    let new_dst = || {
        TensorMap::<f64, 1, 2>::from_vec_with_fusion_space(initial.to_vec(), dst_space.clone())
            .unwrap()
    };
    // The eager entry.
    let mut plain = new_dst();
    fusion_contract_into(&rule, &mut plain, &lhs, &rhs, axes, alpha, beta).unwrap();
    // The same entry profiled: it reports the route the eager call takes.
    let mut artifact = new_dst();
    let mut profile = TensorContractFusionProfile::default();
    TensorContractFusionExecutionContext::<f64, _>::default()
        .tensorcontract_fusion_into_profiled(
            &rule,
            &mut artifact,
            &lhs,
            &rhs,
            axes,
            alpha,
            beta,
            &mut profile,
        )
        .unwrap();
    // What: the twist is uniform within each coupled sector, so the eager
    // route is the canonical core with the twist as per-job alpha: no source
    // is transformed.
    assert_eq!(profile.route, TensorContractFusionRoute::CoreFusionBlocks);
    assert_eq!(profile.lhs_transform_calls, 0);
    assert_eq!(profile.rhs_transform_calls, 0);
    // What: exact (dyadic) agreement with the hand calculation on both paths.
    assert_eq!(plain.data(), &expected);
    assert_eq!(artifact.data(), &expected);
}
