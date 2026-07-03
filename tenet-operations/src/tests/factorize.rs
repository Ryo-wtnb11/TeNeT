use super::*;

fn assert_svd_blocks_match<const NOUT: usize, const NIN: usize>(
    lhs: &TensorMap<f64, NOUT, NIN>,
    rhs: &TensorMap<f64, NOUT, NIN>,
) {
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
                (lhs_value - rhs_value).abs() < 1e-10,
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

fn scale_vt_rows_by_singular_values<R, const NIN: usize>(
    rule: &R,
    vt: &mut TensorMap<f64, 1, NIN>,
    singular_values: &[SectorSingularValues],
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let structure = std::sync::Arc::clone(vt.structure());
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = key
            .codomain_tree()
            .coupled()
            .unwrap_or_else(|| rule.vacuum());
        let values = &singular_values
            .iter()
            .find(|entry| entry.sector == sector)
            .expect("singular values for every Vt sector")
            .values;
        let shape = block.shape().to_vec();
        let count = shape.iter().product::<usize>();
        let mut multi_index = vec![0usize; shape.len()];
        for _ in 0..count {
            let position = block.offset()
                + multi_index
                    .iter()
                    .zip(block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            vt.data_mut()[position] *= values[multi_index[0]];
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

fn run_tsvd_reconstruction_case<R>(rule: &R, sectors: &[SectorId], coupled_layout: bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = TreeTransformBuiltinRuleCacheKey>,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().copied(), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = homspace.fusion_tree_keys(rule).len();
    let dense = TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap();
    let shapes = vec![vec![degeneracy; 4]; key_count];
    let space = if coupled_layout {
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(dense, homspace, rule, shapes).unwrap()
    } else {
        FusionTensorMapSpace::from_degeneracy_shapes(dense, homspace, rule, shapes).unwrap()
    };
    let len = space.required_len().unwrap();
    let tensor = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|index| ((index * 7 + 3) % 23) as f64 * 0.5 - 5.0)
            .collect(),
        space,
    )
    .unwrap();

    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_compact(&mut dense_executor, rule, &tensor, &Truncation::Full).unwrap();

    for entry in &svd.singular_values {
        for pair in entry.values.windows(2) {
            assert!(
                pair[0] >= pair[1] - 1e-12,
                "singular values must be descending in sector {:?}",
                entry.sector
            );
        }
        assert!(entry.values.iter().all(|&value| value >= -1e-12));
    }

    let mut scaled_vt = svd.vh.clone();
    scale_vt_rows_by_singular_values(rule, &mut scaled_vt, &svd.singular_values);

    let mut reconstructed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; len],
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    context
        .tensorcontract_fusion_into(
            rule,
            &mut reconstructed,
            &svd.u,
            &scaled_vt,
            TensorContractAxisSpec::new(&[2], &[0], AxisPermutation::from_axes(&[0, 1, 2, 3])),
            1.0,
            0.0,
        )
        .unwrap();

    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn tsvd_fusion_reconstructs_z2_tensor_packed_layout() {
    run_tsvd_reconstruction_case(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)], false);
}

#[test]
fn tsvd_fusion_reconstructs_z2_tensor_coupled_layout() {
    run_tsvd_reconstruction_case(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)], true);
}

#[test]
fn tsvd_fusion_reconstructs_su2_tensor() {
    run_tsvd_reconstruction_case(
        &SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
        true,
    );
}

#[test]
fn tsvd_fusion_reconstructs_u1_tensor() {
    run_tsvd_reconstruction_case(
        &U1FusionRule,
        &[
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(1).sector_id(),
        ],
        false,
    );
}

fn weighted_norm_squared_of_difference<R>(
    rule: &R,
    lhs: &TensorMap<f64, 2, 2>,
    rhs: &TensorMap<f64, 2, 2>,
) -> f64
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let lhs_structure = std::sync::Arc::clone(lhs.structure());
    let rhs_structure = std::sync::Arc::clone(rhs.structure());
    assert_eq!(lhs_structure.block_count(), rhs_structure.block_count());
    let mut total = 0.0;
    for index in 0..lhs_structure.block_count() {
        let lhs_block = lhs_structure.block(index).unwrap();
        let rhs_block = rhs_structure.block(index).unwrap();
        assert_eq!(lhs_block.key(), rhs_block.key());
        let BlockKey::FusionTree(key) = lhs_block.key() else {
            continue;
        };
        let weight = rule.dim_scalar(
            key.codomain_tree()
                .coupled()
                .unwrap_or_else(|| rule.vacuum()),
        );
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
            let difference = lhs.data()[lhs_position] - rhs.data()[rhs_position];
            total += weight * difference * difference;
            for axis in 0..shape.len() {
                multi_index[axis] += 1;
                if multi_index[axis] < shape[axis] {
                    break;
                }
                multi_index[axis] = 0;
            }
        }
    }
    total
}

fn tsvd_test_tensor<R>(rule: &R, sectors: &[SectorId]) -> TensorMap<f64, 2, 2>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().copied(), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = homspace.fusion_tree_keys(rule).len();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
        homspace,
        rule,
        vec![vec![degeneracy; 4]; key_count],
    )
    .unwrap();
    let len = space.required_len().unwrap();
    TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|index| ((index * 11 + 5) % 29) as f64 * 0.25 - 3.0)
            .collect(),
        space,
    )
    .unwrap()
}

fn reconstruct_from_svd<R>(
    rule: &R,
    template: &TensorMap<f64, 2, 2>,
    svd: &SvdCompact<2, 2>,
) -> TensorMap<f64, 2, 2>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = TreeTransformBuiltinRuleCacheKey>,
{
    let mut scaled_vt = svd.vh.clone();
    scale_vt_rows_by_singular_values(rule, &mut scaled_vt, &svd.singular_values);
    let mut reconstructed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; template.data().len()],
        template.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    context
        .tensorcontract_fusion_into(
            rule,
            &mut reconstructed,
            &svd.u,
            &scaled_vt,
            TensorContractAxisSpec::new(&[2], &[0], AxisPermutation::from_axes(&[0, 1, 2, 3])),
            1.0,
            0.0,
        )
        .unwrap();
    reconstructed
}

#[test]
fn tsvd_truncdim_bounds_weighted_dimension_and_reports_error_su2() {
    let rule = SU2FusionRule;
    let sectors = [
        SU2Irrep::from_twice_spin(0).sector_id(),
        SU2Irrep::from_twice_spin(1).sector_id(),
    ];
    let tensor = tsvd_test_tensor(&rule, &sectors);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let max_dim = 10usize;
    let svd = svd_compact(
        &mut dense_executor,
        &rule,
        &tensor,
        &Truncation::rank(max_dim),
    )
    .unwrap();
    let error = svd.error;

    let weighted_dim: f64 = svd
        .singular_values
        .iter()
        .map(|entry| rule.dim_scalar(entry.sector) * entry.values.len() as f64)
        .sum();
    assert!(
        weighted_dim <= max_dim as f64 + 1e-9,
        "weighted dimension {weighted_dim} exceeds bound {max_dim}"
    );
    assert!(error > 0.0, "this cut must discard weight");

    let reconstructed = reconstruct_from_svd(&rule, &tensor, &svd);
    let distance = weighted_norm_squared_of_difference(&rule, &tensor, &reconstructed).sqrt();
    assert!(
        (distance - error).abs() < 1e-8,
        "reconstruction distance {distance} != reported truncation error {error}"
    );
}

#[test]
fn tsvd_truncbelow_drops_exactly_the_small_values() {
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let tensor = tsvd_test_tensor(&rule, &sectors);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let full = svd_compact(&mut dense_executor, &rule, &tensor, &Truncation::Full).unwrap();
    let threshold = {
        let mut all: Vec<f64> = full
            .singular_values
            .iter()
            .flat_map(|entry| entry.values.iter().copied())
            .collect();
        all.sort_by(|a, b| b.partial_cmp(a).unwrap());
        (all[all.len() / 2] + all[all.len() / 2 - 1]) / 2.0
    };

    let svd = svd_compact(
        &mut dense_executor,
        &rule,
        &tensor,
        &Truncation::absolute_cutoff(threshold),
    )
    .unwrap();
    let error = svd.error;

    for entry in &svd.singular_values {
        assert!(entry.values.iter().all(|&value| value >= threshold));
    }
    let kept: usize = svd
        .singular_values
        .iter()
        .map(|entry| entry.values.len())
        .sum();
    let full_count: usize = full
        .singular_values
        .iter()
        .map(|entry| entry.values.len())
        .sum();
    assert!(kept < full_count);
    assert!(error > 0.0);

    let reconstructed = reconstruct_from_svd(&rule, &tensor, &svd);
    let distance = weighted_norm_squared_of_difference(&rule, &tensor, &reconstructed).sqrt();
    assert!((distance - error).abs() < 1e-8);
}

#[test]
fn tsvd_truncerr_respects_relative_tolerance() {
    let rule = U1FusionRule;
    let sectors = [
        U1Irrep::new(-1).sector_id(),
        U1Irrep::new(0).sector_id(),
        U1Irrep::new(1).sector_id(),
    ];
    let tensor = tsvd_test_tensor(&rule, &sectors);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let tolerance = 0.2;
    let svd = svd_compact(
        &mut dense_executor,
        &rule,
        &tensor,
        &Truncation::relative_error(tolerance),
    )
    .unwrap();
    let error = svd.error;

    let norm = weighted_norm_squared_of_difference(
        &rule,
        &tensor,
        &TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            vec![0.0; tensor.data().len()],
            tensor.fusion_space().unwrap().as_ref().clone(),
        )
        .unwrap(),
    )
    .sqrt();
    assert!(
        error <= tolerance * norm + 1e-9,
        "truncation error {error} exceeds tolerance {tolerance} * norm {norm}"
    );
    assert!(error > 0.0, "tolerance 0.2 must discard something here");

    let reconstructed = reconstruct_from_svd(&rule, &tensor, &svd);
    let distance = weighted_norm_squared_of_difference(&rule, &tensor, &reconstructed).sqrt();
    assert!((distance - error).abs() < 1e-8);
}

#[test]
fn leftorth_fusion_reconstructs_z2_and_su2_tensors() {
    for (rule_case, sectors) in [
        (0usize, vec![SectorId::new(0), SectorId::new(1)]),
        (
            1usize,
            vec![
                SU2Irrep::from_twice_spin(0).sector_id(),
                SU2Irrep::from_twice_spin(1).sector_id(),
            ],
        ),
    ] {
        if rule_case == 0 {
            let rule = Z2FusionRule;
            let tensor = tsvd_test_tensor(&rule, &sectors);
            let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
            let (q, r) = qr_compact(&mut dense_executor, &rule, &tensor).unwrap();
            let reconstructed = contract_pair(&rule, &tensor, &q, &r);
            assert_svd_blocks_match(&tensor, &reconstructed);
        } else {
            let rule = SU2FusionRule;
            let tensor = tsvd_test_tensor(&rule, &sectors);
            let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
            let (q, r) = qr_compact(&mut dense_executor, &rule, &tensor).unwrap();
            let reconstructed = contract_pair(&rule, &tensor, &q, &r);
            assert_svd_blocks_match(&tensor, &reconstructed);
        }
    }
}

#[test]
fn rightorth_fusion_reconstructs_z2_and_su2_tensors() {
    {
        let rule = Z2FusionRule;
        let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
        let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
        let (l, q) = lq_compact(&mut dense_executor, &rule, &tensor).unwrap();
        let reconstructed = contract_pair(&rule, &tensor, &l, &q);
        assert_svd_blocks_match(&tensor, &reconstructed);
    }
    {
        let rule = SU2FusionRule;
        let tensor = tsvd_test_tensor(
            &rule,
            &[
                SU2Irrep::from_twice_spin(0).sector_id(),
                SU2Irrep::from_twice_spin(1).sector_id(),
            ],
        );
        let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
        let (l, q) = lq_compact(&mut dense_executor, &rule, &tensor).unwrap();
        let reconstructed = contract_pair(&rule, &tensor, &l, &q);
        assert_svd_blocks_match(&tensor, &reconstructed);
    }
}

fn contract_pair<R>(
    rule: &R,
    template: &TensorMap<f64, 2, 2>,
    left: &TensorMap<f64, 2, 1>,
    right: &TensorMap<f64, 1, 2>,
) -> TensorMap<f64, 2, 2>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = TreeTransformBuiltinRuleCacheKey>,
{
    let mut reconstructed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; template.data().len()],
        template.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    context
        .tensorcontract_fusion_into(
            rule,
            &mut reconstructed,
            left,
            right,
            TensorContractAxisSpec::new(&[2], &[0], AxisPermutation::from_axes(&[0, 1, 2, 3])),
            1.0,
            0.0,
        )
        .unwrap();
    reconstructed
}

#[test]
fn tsvd_singular_tensor_composes_u_s_vt() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_compact(&mut dense_executor, &rule, &tensor, &Truncation::Full).unwrap();
    let s_tensor = svd.s.clone();

    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    let mut u_s = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![0.0; svd.u.data().len()],
        svd.u.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut u_s,
            &svd.u,
            &s_tensor,
            TensorContractAxisSpec::new(&[2], &[0], AxisPermutation::from_axes(&[0, 1, 2])),
            1.0,
            0.0,
        )
        .unwrap();

    let mut reconstructed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; tensor.data().len()],
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut reconstructed,
            &u_s,
            &svd.vh,
            TensorContractAxisSpec::new(&[2], &[0], AxisPermutation::from_axes(&[0, 1, 2, 3])),
            1.0,
            0.0,
        )
        .unwrap();

    assert_svd_blocks_match(&tensor, &reconstructed);
}
