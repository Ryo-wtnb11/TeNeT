use super::*;

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
    // What: all three callers compare exact warm cache state across two
    // executions and therefore cannot overlap a process-global reset.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rule, &lhs_space).unwrap();
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rule, &rhs_space).unwrap();
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
    let mut context = TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();
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
    } else {
        assert_eq!(context.tree_context().cache().structure_len(), 0);
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
    } else {
        assert_eq!(context.tree_context().cache().structure_len(), 0);
    }
}

pub(super) fn maybe_adjoint_axes<const NOUT: usize, const NIN: usize>(
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

pub(super) fn su2_three_to_one_homspace(
    codomain_dual: bool,
    domain_dual: bool,
) -> FusionTreeHomSpace {
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

pub(super) fn su2_one_to_three_homspace(
    codomain_dual: bool,
    domain_dual: bool,
) -> FusionTreeHomSpace {
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
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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

    context_dst
        .data_mut()
        .copy_from_slice(&[Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)]);
    context
        .tensorcontract_fusion_into(&rule, &mut context_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
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
