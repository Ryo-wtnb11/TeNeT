use super::*;

fn copy_blocks_between_layouts<D: Copy>(dst: &mut TensorMap<D, 2, 2>, src: &TensorMap<D, 2, 2>) {
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

#[test]
fn coupled_layout_contraction_matches_packed_layout_asymmetric_u1() {
    run_coupled_vs_packed_contractions(
        &U1FusionRule,
        &[
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(2).sector_id(),
        ],
    );
}

#[test]
fn coupled_layout_contraction_matches_packed_layout_product() {
    let left = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let sector = |parity, charge, twice_spin| {
        rule.encode_component_ids(
            left.encode_component_ids(parity, U1Irrep::new(charge).sector_id()),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };
    run_coupled_vs_packed_contractions(
        &rule,
        &[
            sector(SectorId::new(0), 0, 0),
            sector(SectorId::new(1), 1, 1),
            sector(SectorId::new(1), -1, 1),
        ],
    );
}

#[test]
fn coupled_layout_complex_contraction_matches_packed_fallback() {
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.map(|sector| (sector, degeneracy)), false);
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let dense = || TensorMapSpace::<2, 2>::from_dims([4, 4], [4, 4]).unwrap();
    let packed_space = |hom: FusionTreeHomSpace| {
        let keys = hom.fusion_tree_keys(&rule);
        FusionTensorMapSpace::new_unbound(
            dense(),
            hom,
            crate::tests::packed_fixture_structure(
                4,
                keys.iter().cloned().map(|key| (key, vec![degeneracy; 4])),
            )
            .unwrap(),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap()
    };
    let coupled_space = |hom: FusionTreeHomSpace| {
        let shapes = vec![vec![degeneracy; 4]; hom.fusion_tree_keys(&rule).len()];
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(dense(), hom, &rule, shapes).unwrap()
    };
    let lhs_packed_space = packed_space(homspace());
    assert!(lhs_packed_space
        .subblock_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let lhs_len = lhs_packed_space.required_len().unwrap();
    let lhs_packed = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_len)
            .map(|index| Complex64::new(index as f64 * 0.25 - 1.0, 0.125 * index as f64))
            .collect(),
        lhs_packed_space,
    )
    .unwrap();
    let rhs_packed_space = packed_space(homspace());
    assert!(rhs_packed_space
        .subblock_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let rhs_len = rhs_packed_space.required_len().unwrap();
    let rhs_packed = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        (0..rhs_len)
            .map(|index| Complex64::new(0.75 - index as f64 * 0.1, 0.5))
            .collect(),
        rhs_packed_space,
    )
    .unwrap();
    let lhs_coupled_space = coupled_space(homspace());
    assert!(lhs_coupled_space
        .subblock_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    let rhs_coupled_space = coupled_space(homspace());
    assert!(rhs_coupled_space
        .subblock_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    let mut lhs_coupled = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero(); lhs_len],
        lhs_coupled_space,
    )
    .unwrap();
    let mut rhs_coupled = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero(); rhs_len],
        rhs_coupled_space,
    )
    .unwrap();
    copy_blocks_between_layouts(&mut lhs_coupled, &lhs_packed);
    copy_blocks_between_layouts(&mut rhs_coupled, &rhs_packed);

    let axes = TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]);
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_packed.fusion_space().unwrap().homspace(),
        rhs_packed.fusion_space().unwrap().homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        &[0, 1, 2, 3],
        2,
    )
    .unwrap();
    let dst_packed_space = packed_space(dst_hom.clone());
    let dst_coupled_space = coupled_space(dst_hom);
    assert!(dst_packed_space
        .subblock_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    assert!(dst_coupled_space
        .subblock_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    let dst_len = dst_packed_space.required_len().unwrap();
    let initial = (0..dst_len)
        .map(|index| Complex64::new(1.0 + index as f64 * 0.05, -0.25))
        .collect::<Vec<_>>();
    let mut dst_packed =
        TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(initial, dst_packed_space.clone())
            .unwrap();
    let mut dst_coupled = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero(); dst_len],
        dst_coupled_space,
    )
    .unwrap();
    copy_blocks_between_layouts(&mut dst_coupled, &dst_packed);
    let alpha = Complex64::new(0.75, -0.25);
    let beta = Complex64::new(-0.5, 0.125);
    fusion_contract_into(
        &rule,
        &mut dst_packed,
        &lhs_packed,
        &rhs_packed,
        axes,
        alpha,
        beta,
    )
    .unwrap();
    fusion_contract_into(
        &rule,
        &mut dst_coupled,
        &lhs_coupled,
        &rhs_coupled,
        axes,
        alpha,
        beta,
    )
    .unwrap();
    let mut coupled_in_packed_layout = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero(); dst_len],
        dst_packed_space,
    )
    .unwrap();
    copy_blocks_between_layouts(&mut coupled_in_packed_layout, &dst_coupled);

    // What: multi-sector C64 direct replay preserves the packed fallback's
    // blockwise result, including complex alpha and beta.
    for (&direct, &fallback) in coupled_in_packed_layout
        .data()
        .iter()
        .zip(dst_packed.data())
    {
        assert!((direct - fallback).norm() < 1.0e-12);
    }
}

#[test]
fn coupled_layout_complex_scalar_contraction_matches_closed_form() {
    let rule = Z2FusionRule;
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        &rule,
        [Vec::<usize>::new()],
    )
    .unwrap();
    let lhs_value = Complex64::new(2.0, 1.0);
    let rhs_value = Complex64::new(3.0, -2.0);
    let initial = Complex64::new(5.0, 0.5);
    let alpha = Complex64::new(0.75, -0.25);
    let beta = Complex64::new(-0.5, 0.125);
    let lhs =
        TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(vec![lhs_value], space.clone())
            .unwrap();
    let rhs =
        TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(vec![rhs_value], space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(vec![initial], space).unwrap();
    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[], &[]),
        alpha,
        beta,
    )
    .unwrap();

    // What: scalar direct replay matches an implementation-independent
    // closed form rather than another compiled contraction route.
    assert_eq!(
        dst.data(),
        &[alpha * lhs_value * rhs_value + beta * initial]
    );
}

fn run_coupled_vs_packed_contractions<R>(rule: &R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
    R::Key: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let leg_dim = sectors.len() * degeneracy;
    let dense =
        || TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap();
    let shapes =
        |hom: &FusionTreeHomSpace| vec![vec![degeneracy; 4]; hom.fusion_tree_keys(rule).len()];
    let packed_space = |hom: FusionTreeHomSpace| {
        let keys = hom.fusion_tree_keys(rule);
        FusionTensorMapSpace::new_unbound(
            dense(),
            hom,
            crate::tests::packed_fixture_structure(
                4,
                keys.iter().cloned().map(|key| (key, vec![degeneracy; 4])),
            )
            .unwrap(),
        )
        .unwrap()
        .try_bind_rule(rule)
        .unwrap()
    };
    let coupled_space = |hom: FusionTreeHomSpace| {
        let shape_list = shapes(&hom);
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(dense(), hom, rule, shape_list)
            .unwrap()
    };

    let lhs_packed_space = packed_space(homspace());
    assert!(lhs_packed_space
        .subblock_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let lhs_len = lhs_packed_space.required_len().unwrap();
    let lhs_packed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_len).map(|i| (i % 11) as f64 * 0.5 - 2.0).collect(),
        lhs_packed_space,
    )
    .unwrap();
    let rhs_packed_space = packed_space(homspace());
    assert!(rhs_packed_space
        .subblock_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let rhs_len = rhs_packed_space.required_len().unwrap();
    let rhs_packed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..rhs_len).map(|i| (i % 7) as f64 * 0.25 - 1.0).collect(),
        rhs_packed_space,
    )
    .unwrap();

    let lhs_coupled_space = coupled_space(homspace());
    assert!(lhs_coupled_space
        .subblock_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    let rhs_coupled_space = coupled_space(homspace());
    assert!(rhs_coupled_space
        .subblock_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    let mut lhs_coupled =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; lhs_len], lhs_coupled_space)
            .unwrap();
    let mut rhs_coupled =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; rhs_len], rhs_coupled_space)
            .unwrap();
    copy_blocks_between_layouts(&mut lhs_coupled, &lhs_packed);
    copy_blocks_between_layouts(&mut rhs_coupled, &rhs_packed);

    let workloads: [(&[usize; 2], &[usize; 2], &[usize; 4]); 3] = [
        (&[2, 3], &[0, 1], &[0, 1, 2, 3]),
        (&[3, 2], &[0, 1], &[0, 1, 2, 3]),
        (&[3, 2], &[0, 1], &[1, 0, 2, 3]),
    ];
    for (workload_index, (lhs_axes, rhs_axes, output_axes)) in workloads.into_iter().enumerate() {
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
        assert!(dst_packed_space
            .subblock_structure()
            .coupled_sector_regions(2)
            .unwrap()
            .is_none());
        let dst_coupled_space = coupled_space(dst_hom);
        assert!(dst_coupled_space
            .subblock_structure()
            .coupled_sector_regions(2)
            .unwrap()
            .is_some());
        let dst_len = dst_packed_space.required_len().unwrap();
        let mut dst_packed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            vec![0.0; dst_len],
            dst_packed_space,
        )
        .unwrap();
        let mut dst_coupled = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            vec![0.0; dst_len],
            dst_coupled_space,
        )
        .unwrap();

        let axes =
            || TensorContractSpec::new(lhs_axes, rhs_axes, OutputAxisOrder::from_axes(output_axes));
        let mut packed_context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
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
        if workload_index == 0 {
            let ordinary = dst_packed.data().to_vec();
            let prepared = packed_context
                .prepare_tensorcontract_fusion(rule, &dst_packed, &lhs_packed, &rhs_packed, axes())
                .unwrap();
            dst_packed.data_mut().fill(0.0);
            packed_context
                .execute_prepared_tensorcontract_fusion(
                    &prepared,
                    rule,
                    &mut dst_packed,
                    &lhs_packed,
                    &rhs_packed,
                    1.0,
                    0.0,
                )
                .unwrap();
            assert_eq!(dst_packed.data(), ordinary);

            dst_packed.data_mut().fill(0.0);
            let mut profile = TensorContractFusionProfile::default();
            packed_context
                .tensorcontract_fusion_into_profiled(
                    rule,
                    &mut dst_packed,
                    &lhs_packed,
                    &rhs_packed,
                    axes(),
                    1.0,
                    0.0,
                    &mut profile,
                )
                .unwrap();
            assert_eq!(dst_packed.data(), ordinary);
            assert_eq!(profile.route, TensorContractFusionRoute::CoreFusionBlocks);
            assert!(profile.core_contract_groups > 0);

            let dst_dynamic = DynamicFusionMapSpace::from_typed(dst_packed.fusion_space().unwrap());
            let lhs_dynamic = DynamicFusionMapSpace::from_typed(lhs_packed.fusion_space().unwrap());
            let rhs_dynamic = DynamicFusionMapSpace::from_typed(rhs_packed.fusion_space().unwrap());
            let mut dynamic = vec![0.0; ordinary.len()];
            packed_context
                .tensorcontract_fusion_dyn_into_raw(
                    rule,
                    &dst_dynamic,
                    &mut dynamic,
                    &lhs_dynamic,
                    lhs_packed.data(),
                    &rhs_dynamic,
                    rhs_packed.data(),
                    axes(),
                    1.0,
                    0.0,
                )
                .unwrap();
            assert_eq!(dynamic, ordinary);
        }
        let mut coupled_context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
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

    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
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
    // Pack/scatter no longer exist on the fully-direct core route, so the
    // pack counters stay at their zero defaults.
    assert_eq!(profile.core_direct_pack_skips, 0);
    assert_eq!(profile.core_pack_lhs, std::time::Duration::ZERO);
    assert_eq!(profile.core_scatter, std::time::Duration::ZERO);
}
