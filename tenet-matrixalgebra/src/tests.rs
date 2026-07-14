use tenet_core::{
    BlockKey, FusionProductSpace, FusionRule, FusionTensorMapSpace, FusionTreeHomSpace,
    FusionTreeKey, MultiplicityFreeRigidSymbols, SU2FusionRule, SU2Irrep, SectorId, SectorLeg,
    TensorMap, TensorMapSpace, U1FusionRule, U1Irrep, Z2FusionRule,
};
use tenet_tensors::{
    OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
    TreeTransformBuiltinRuleCacheKey, TreeTransformRuleCacheKey,
};

use crate::factorize::truncate_svd;
use crate::*;

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
    singular_values: &[SectorSpectrum],
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
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
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
    let svd = svd_trunc(&mut dense_executor, rule, &tensor, &Truncation::Full).unwrap();

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
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
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
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
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
    svd: &SvdTrunc<f64, 2, 2>,
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
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
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
    let svd = svd_trunc(
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

    let full = svd_trunc(&mut dense_executor, &rule, &tensor, &Truncation::Full).unwrap();
    let threshold = {
        let mut all: Vec<f64> = full
            .singular_values
            .iter()
            .flat_map(|entry| entry.values.iter().copied())
            .collect();
        all.sort_by(|a, b| b.partial_cmp(a).unwrap());
        (all[all.len() / 2] + all[all.len() / 2 - 1]) / 2.0
    };

    let svd = svd_trunc(
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
    let svd = svd_trunc(
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
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
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
    let svd = svd_trunc(&mut dense_executor, &rule, &tensor, &Truncation::Full).unwrap();
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
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
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
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
            1.0,
            0.0,
        )
        .unwrap();

    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn svd_trunc_is_svd_compact_plus_host_truncation() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let truncation = Truncation::rank(9).and(Truncation::absolute_cutoff(1e-12));

    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let composed = {
        let full = svd_compact(&mut dense_executor, &rule, &tensor).unwrap();
        truncate_svd(&rule, full, &truncation).unwrap()
    };
    let direct = svd_trunc(&mut dense_executor, &rule, &tensor, &truncation).unwrap();

    assert_eq!(composed.singular_values, direct.singular_values);
    assert!((composed.error - direct.error).abs() < 1e-15);
    assert_eq!(composed.u.data(), direct.u.data());
    assert_eq!(composed.s.data(), direct.s.data());
    assert_eq!(composed.vh.data(), direct.vh.data());
}

fn hermitian_test_tensor<R>(rule: &R, sectors: &[SectorId]) -> TensorMap<f64, 2, 2>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
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
    // Symmetric under swapping the (codomain tree, row indices) and
    // (domain tree, column indices) labels, so every coupled sector matrix is
    // symmetric (real Hermitian).
    let side_label = |tree: &FusionTreeKey, indices: &[usize]| -> u64 {
        let mut label = 17u64;
        for &sector in tree.uncoupled() {
            label = label.wrapping_mul(31).wrapping_add(sector.id() as u64 + 1);
        }
        for &index in indices {
            label = label.wrapping_mul(37).wrapping_add(index as u64 + 1);
        }
        label
    };
    TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(space, 0.0, |key, indices| {
        let BlockKey::FusionTree(tree) = key else {
            return 0.0;
        };
        let row = side_label(tree.codomain_tree(), &indices[..2]);
        let col = side_label(tree.domain_tree(), &indices[2..]);
        let (low, high) = if row <= col { (row, col) } else { (col, row) };
        let hash = low
            .wrapping_mul(6364136223846793005)
            .wrapping_add(high.wrapping_mul(1442695040888963407));
        ((hash >> 33) % 19) as f64 * 0.5 - 4.0
    })
    .unwrap()
}

fn assert_eigen_equation<R>(
    rule: &R,
    tensor: &TensorMap<f64, 2, 2>,
    v: &TensorMap<f64, 2, 1>,
    d: &TensorMap<f64, 1, 1>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = TreeTransformBuiltinRuleCacheKey>,
{
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    // t . V
    let mut tv = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![0.0; v.data().len()],
        v.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            rule,
            &mut tv,
            tensor,
            v,
            TensorContractSpec::new(&[2, 3], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2])),
            1.0,
            0.0,
        )
        .unwrap();
    // V . D
    let mut vd = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![0.0; v.data().len()],
        v.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            rule,
            &mut vd,
            v,
            d,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
            1.0,
            0.0,
        )
        .unwrap();

    for (index, (lhs, rhs)) in tv.data().iter().zip(vd.data()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "eigen equation violated at raw position {index}: {lhs} != {rhs}"
        );
    }
}

#[test]
fn eigh_full_satisfies_the_eigen_equation() {
    let rule = SU2FusionRule;
    let tensor = hermitian_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let eigh = eigh_full(&mut dense_executor, &rule, &tensor).unwrap();

    for entry in &eigh.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!(
                pair[0].abs() >= pair[1].abs() - 1e-12,
                "eigenvalues must be stored descending by magnitude"
            );
        }
    }
    assert_eigen_equation(&rule, &tensor, &eigh.v, &eigh.d);
}

#[test]
fn eigh_trunc_truncates_by_magnitude_and_keeps_eigen_equation() {
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let full = eigh_full(&mut dense_executor, &rule, &tensor).unwrap();
    let full_count: usize = full
        .eigenvalues
        .iter()
        .map(|entry| entry.values.len())
        .sum();
    let max_dim = full_count / 2;
    let eigh = eigh_trunc(
        &mut dense_executor,
        &rule,
        &tensor,
        &Truncation::rank(max_dim),
    )
    .unwrap();

    let kept: usize = eigh
        .eigenvalues
        .iter()
        .map(|entry| entry.values.len())
        .sum();
    assert!(kept <= max_dim);
    assert!(eigh.error > 0.0);
    // Truncated eigenvectors still satisfy t . V = V . D exactly.
    assert_eigen_equation(&rule, &tensor, &eigh.v, &eigh.d);
}

fn dense_sector_matrices<const A: usize, const B: usize>(
    tensor_nout: usize,
    t: &TensorMap<f64, A, B>,
) -> Vec<(SectorId, usize, usize, Vec<f64>)> {
    // Matricize per coupled sector (rows = codomain trees x degeneracy,
    // cols = domain trees x degeneracy) for dense checks in tests.
    struct SectorAccumulator {
        sector: SectorId,
        rows: usize,
        cols: usize,
        row_trees: Vec<(FusionTreeKey, usize)>,
        col_trees: Vec<(FusionTreeKey, usize)>,
        entries: Vec<(usize, usize, f64)>,
    }
    let structure = std::sync::Arc::clone(t.structure());
    let mut sectors: Vec<SectorAccumulator> = Vec::new();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = key
            .codomain_tree()
            .coupled()
            .or_else(|| key.domain_tree().coupled())
            .unwrap_or_else(|| key.domain_tree().uncoupled()[0]);
        let entry = match sectors.iter_mut().find(|entry| entry.sector == sector) {
            Some(entry) => entry,
            None => {
                sectors.push(SectorAccumulator {
                    sector,
                    rows: 0,
                    cols: 0,
                    row_trees: Vec::new(),
                    col_trees: Vec::new(),
                    entries: Vec::new(),
                });
                sectors.last_mut().unwrap()
            }
        };
        let shape = block.shape().to_vec();
        let row_dim: usize = shape[..tensor_nout].iter().product();
        let col_dim: usize = shape[tensor_nout..].iter().product();
        let row_offset = match entry
            .row_trees
            .iter()
            .find(|(tree, _)| tree == key.codomain_tree())
        {
            Some((_, offset)) => *offset,
            None => {
                let offset = entry.rows;
                entry.row_trees.push((key.codomain_tree().clone(), offset));
                entry.rows += row_dim;
                offset
            }
        };
        let col_offset = match entry
            .col_trees
            .iter()
            .find(|(tree, _)| tree == key.domain_tree())
        {
            Some((_, offset)) => *offset,
            None => {
                let offset = entry.cols;
                entry.col_trees.push((key.domain_tree().clone(), offset));
                entry.cols += col_dim;
                offset
            }
        };
        let strides = block.strides().to_vec();
        let offset = block.offset();
        let mut indices = vec![0usize; shape.len()];
        for _ in 0..shape.iter().product::<usize>() {
            let position = offset
                + indices
                    .iter()
                    .zip(&strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let mut row = 0;
            let mut stride = 1;
            for axis in 0..tensor_nout {
                row += indices[axis] * stride;
                stride *= shape[axis];
            }
            let mut col = 0;
            let mut col_stride = 1;
            for axis in tensor_nout..shape.len() {
                col += indices[axis] * col_stride;
                col_stride *= shape[axis];
            }
            entry
                .entries
                .push((row_offset + row, col_offset + col, t.data()[position]));
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    sectors
        .into_iter()
        .map(|entry| {
            let mut matrix = vec![0.0; entry.rows * entry.cols];
            for (row, col, value) in entry.entries {
                matrix[row + entry.rows * col] = value;
            }
            (entry.sector, entry.rows, entry.cols, matrix)
        })
        .collect()
}

fn assert_orthonormal_columns(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    for (sector, rows, cols, matrix) in matrices {
        for left in 0..*cols {
            for right in 0..*cols {
                let mut dot = 0.0;
                for row in 0..*rows {
                    dot += matrix[row + rows * left] * matrix[row + rows * right];
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-9,
                    "sector {sector:?}: column dot ({left},{right}) = {dot}"
                );
            }
        }
    }
}

#[test]
fn qr_full_gives_square_unitary_and_reconstructs() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let (q, r) = qr_full(&mut dense_executor, &rule, &tensor).unwrap();

    let matrices = dense_sector_matrices(2, &q);
    for (_, rows, cols, _) in &matrices {
        assert_eq!(rows, cols, "full Q must be square per sector");
    }
    assert_orthonormal_columns(&matrices);

    let reconstructed = contract_pair(&rule, &tensor, &q, &r);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn lq_full_reconstructs() {
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let (l, q) = lq_full(&mut dense_executor, &rule, &tensor).unwrap();
    let reconstructed = contract_pair(&rule, &tensor, &l, &q);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn svd_full_gives_square_unitaries_and_reconstructs() {
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let full = svd_full(&mut dense_executor, &rule, &tensor).unwrap();

    let matrices = dense_sector_matrices(2, &full.u);
    for (_, rows, cols, _) in &matrices {
        assert_eq!(rows, cols, "full U must be square per sector");
    }
    assert_orthonormal_columns(&matrices);

    // U . S has U's codomain and S's (column) bond as domain; build its space
    // from the contraction homspace and per-tree shapes.
    let us_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        full.u.fusion_space().unwrap().homspace(),
        full.s.fusion_space().unwrap().homspace(),
        &[2],
        &[0],
        &[0, 1, 2],
        2,
    )
    .unwrap();
    let u_structure = std::sync::Arc::clone(full.u.structure());
    let s_structure = std::sync::Arc::clone(full.s.structure());
    let shapes = us_hom
        .fusion_tree_keys(&rule)
        .iter()
        .map(|key| {
            let sector = key
                .domain_tree()
                .coupled()
                .unwrap_or_else(|| key.domain_tree().uncoupled()[0]);
            let mut shape = None;
            for index in 0..u_structure.block_count() {
                let block = u_structure.block(index).unwrap();
                let BlockKey::FusionTree(u_key) = block.key() else {
                    continue;
                };
                if u_key.codomain_tree() == key.codomain_tree() {
                    shape = Some(block.shape()[..2].to_vec());
                    break;
                }
            }
            let mut shape = shape.expect("U tree present");
            let mut s_cols = 0;
            for index in 0..s_structure.block_count() {
                let block = s_structure.block(index).unwrap();
                let BlockKey::FusionTree(s_key) = block.key() else {
                    continue;
                };
                let s_sector = s_key
                    .domain_tree()
                    .coupled()
                    .unwrap_or_else(|| s_key.domain_tree().uncoupled()[0]);
                if s_sector == sector {
                    s_cols = block.shape()[1];
                    break;
                }
            }
            shape.push(s_cols);
            shape
        })
        .collect::<Vec<_>>();
    let dims = full.u.space().dims();
    let us_space = FusionTensorMapSpace::<2, 1>::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 1>::from_dims([dims[0], dims[1]], [full.s.space().dims()[1]]).unwrap(),
        us_hom,
        &rule,
        shapes,
    )
    .unwrap();
    let mut us = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![0.0; us_space.required_len().unwrap()],
        us_space,
    )
    .unwrap();
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut us,
            &full.u,
            &full.s,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
            1.0,
            0.0,
        )
        .unwrap();
    let reconstructed = contract_pair(&rule, &tensor, &us, &full.vh);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn svd_trunc_c64_reconstruction_distance_matches_error() {
    use num_complex::Complex64;
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = homspace.fusion_tree_keys(&rule).len();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
        homspace,
        &rule,
        vec![vec![degeneracy; 4]; key_count],
    )
    .unwrap();
    let len = space.required_len().unwrap();
    let tensor = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| {
                Complex64::new(
                    ((i * 7 + 3) % 23) as f64 * 0.5 - 5.0,
                    ((i * 5 + 1) % 17) as f64 * 0.25 - 2.0,
                )
            })
            .collect(),
        space,
    )
    .unwrap();

    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_trunc(&mut dense_executor, &rule, &tensor, &Truncation::rank(8)).unwrap();
    assert!(svd.error > 0.0);
    for entry in &svd.singular_values {
        for pair in entry.values.windows(2) {
            assert!(pair[0] >= pair[1] - 1e-12);
        }
    }

    // Scale Vh rows by the (real) singular values.
    let mut scaled_vh = svd.vh.clone();
    {
        let structure = std::sync::Arc::clone(scaled_vh.structure());
        for index in 0..structure.block_count() {
            let block = structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            let sector = key
                .codomain_tree()
                .coupled()
                .unwrap_or_else(|| rule.vacuum());
            let values = &svd
                .singular_values
                .iter()
                .find(|entry| entry.sector == sector)
                .unwrap()
                .values;
            let shape = block.shape().to_vec();
            let strides = block.strides().to_vec();
            let offset = block.offset();
            let count = shape.iter().product::<usize>();
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..count {
                let position = offset
                    + indices
                        .iter()
                        .zip(&strides)
                        .map(|(&i, &s)| i * s)
                        .sum::<usize>();
                scaled_vh.data_mut()[position] *= values[indices[0]];
                for axis in 0..shape.len() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
        }
    }

    let mut reconstructed = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); len],
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<
        Complex64,
        TreeTransformBuiltinRuleCacheKey,
    >::default();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut reconstructed,
            &svd.u,
            &scaled_vh,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();

    // Weighted 2-norm of the difference equals the reported error (Z2 has
    // quantum dimension 1 everywhere).
    let distance = tensor
        .data()
        .iter()
        .zip(reconstructed.data())
        .map(|(lhs, rhs)| (lhs - rhs).norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(
        (distance - svd.error).abs() < 1e-8,
        "distance {distance} != error {}",
        svd.error
    );
}

#[test]
fn eig_full_satisfies_the_eigen_equation_for_real_input() {
    use num_complex::Complex64;
    let rule = Z2FusionRule;
    // Non-symmetric endomorphism.
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let eig = eig_full(&mut dense_executor, &rule, &tensor).unwrap();

    for entry in &eig.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!(pair[0].norm() >= pair[1].norm() - 1e-12);
        }
    }

    // Promote t to complex (same space => same layout => elementwise cast).
    let tensor_c = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        tensor
            .data()
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect(),
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();

    let mut context = TensorContractFusionExecutionContext::<
        Complex64,
        TreeTransformBuiltinRuleCacheKey,
    >::default();
    let mut tv = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); eig.v.data().len()],
        eig.v.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut tv,
            &tensor_c,
            &eig.v,
            TensorContractSpec::new(&[2, 3], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2])),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();
    let mut vd = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); eig.v.data().len()],
        eig.v.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut vd,
            &eig.v,
            &eig.d,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();
    for (index, (lhs, rhs)) in tv.data().iter().zip(vd.data()).enumerate() {
        assert!(
            (lhs - rhs).norm() < 1e-8,
            "eigen equation violated at raw position {index}: {lhs} != {rhs}"
        );
    }
}

#[test]
fn null_spaces_are_orthonormal_and_annihilate_the_tensor() {
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;

    // Tall map (2 codomain legs, 1 domain leg): nontrivial left null space.
    let tall_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );
    let key_count = tall_hom.fusion_tree_keys(&rule).len();
    let tall_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 1>::from_dims([leg_dim, leg_dim], [leg_dim]).unwrap(),
        tall_hom,
        &rule,
        vec![vec![degeneracy; 3]; key_count],
    )
    .unwrap();
    let len = tall_space.required_len().unwrap();
    let tall = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        (0..len).map(|i| ((i * 3 + 1) % 13) as f64 - 6.0).collect(),
        tall_space,
    )
    .unwrap();
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let null = left_null(&mut dense_executor, &rule, &tall).unwrap();

    let null_matrices = dense_sector_matrices(2, &null);
    assert!(!null_matrices.is_empty());
    assert_orthonormal_columns(&null_matrices);
    let tensor_matrices = dense_sector_matrices(2, &tall);
    for (sector, n_rows, n_cols, n) in &null_matrices {
        let (_, a_rows, a_cols, a) = tensor_matrices
            .iter()
            .find(|(candidate, ..)| candidate == sector)
            .expect("tensor sector present");
        assert_eq!(n_rows, a_rows);
        assert_eq!(*n_cols, a_rows - (*a_rows).min(*a_cols));
        // N^T A = 0.
        for null_col in 0..*n_cols {
            for a_col in 0..*a_cols {
                let mut dot = 0.0;
                for row in 0..*a_rows {
                    dot += n[row + n_rows * null_col] * a[row + a_rows * a_col];
                }
                assert!(dot.abs() < 1e-9, "left null failed: {dot}");
            }
        }
    }

    // Wide map (1 codomain leg, 2 domain legs): nontrivial right null space.
    let wide_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = wide_hom.fusion_tree_keys(&rule).len();
    let wide_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 2>::from_dims([leg_dim], [leg_dim, leg_dim]).unwrap(),
        wide_hom,
        &rule,
        vec![vec![degeneracy; 3]; key_count],
    )
    .unwrap();
    let len = wide_space.required_len().unwrap();
    let wide = TensorMap::<f64, 1, 2>::from_vec_with_fusion_space(
        (0..len).map(|i| ((i * 5 + 2) % 11) as f64 - 5.0).collect(),
        wide_space,
    )
    .unwrap();
    let null = right_null(&mut dense_executor, &rule, &wide).unwrap();

    let null_matrices = dense_sector_matrices(1, &null);
    assert!(!null_matrices.is_empty());
    let tensor_matrices = dense_sector_matrices(1, &wide);
    for (sector, n_rows, n_cols, n) in &null_matrices {
        let (_, a_rows, a_cols, a) = tensor_matrices
            .iter()
            .find(|(candidate, ..)| candidate == sector)
            .expect("tensor sector present");
        assert_eq!(n_cols, a_cols);
        assert_eq!(*n_rows, a_cols - (*a_cols).min(*a_rows));
        // Rows of N are orthonormal: N N^T = I.
        for left in 0..*n_rows {
            for right in 0..*n_rows {
                let mut dot = 0.0;
                for col in 0..*n_cols {
                    dot += n[left + n_rows * col] * n[right + n_rows * col];
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-9);
            }
        }
        // A N^T = 0 (rows of N span the kernel).
        for a_row in 0..*a_rows {
            for null_row in 0..*n_rows {
                let mut dot = 0.0;
                for col in 0..*a_cols {
                    dot += a[a_row + a_rows * col] * n[null_row + n_rows * col];
                }
                assert!(dot.abs() < 1e-9, "right null failed: {dot}");
            }
        }
    }
}

#[test]
fn spectrum_only_entry_points_return_descending_magnitudes() {
    let rule = Z2FusionRule;
    let hermitian = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let general = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let svd = svd_vals(&mut dense_executor, &rule, &general).unwrap();
    assert!(!svd.is_empty());
    for entry in &svd {
        for pair in entry.values.windows(2) {
            assert!(pair[0] >= pair[1] - 1e-12);
        }
    }
    let eigh = eigh_vals(&mut dense_executor, &rule, &hermitian).unwrap();
    assert!(!eigh.is_empty());
    for entry in &eigh {
        for pair in entry.values.windows(2) {
            assert!(pair[0].abs() >= pair[1].abs() - 1e-12);
        }
    }
    let eig = eig_vals(&mut dense_executor, &rule, &general).unwrap();
    assert!(!eig.is_empty());
    for entry in &eig {
        for pair in entry.values.windows(2) {
            assert!(pair[0].norm() >= pair[1].norm() - 1e-12);
        }
    }
}

#[test]
fn values_only_entry_points_match_the_full_decomposition_spectra() {
    // The `_vals` paths call LAPACK `job='N'` (no vectors) and must reproduce
    // the full decomposition's spectrum. This is a numerical-agreement check,
    // not bit-for-bit: LAPACK backends may route the vectors-vs-no-vectors
    // cases through different routines (e.g. `gesdd` divide-and-conquer for the
    // full SVD vs `gesvd` QR for values-only), which differ in the last ULPs.
    let rule = Z2FusionRule;
    let hermitian = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let general = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let tol = 1e-10;
    let assert_real_close = |vals: &[SectorSpectrum], full: &[SectorSpectrum]| {
        assert_eq!(vals.len(), full.len());
        for (a, b) in vals.iter().zip(full) {
            assert_eq!(a.sector, b.sector);
            assert_eq!(a.values.len(), b.values.len());
            for (x, y) in a.values.iter().zip(&b.values) {
                assert!((x - y).abs() <= tol, "{x} vs {y}");
            }
        }
    };

    let svd_vals_spectra = svd_vals(&mut dense_executor, &rule, &general).unwrap();
    let svd_full_spectra = svd_compact(&mut dense_executor, &rule, &general)
        .unwrap()
        .singular_values;
    assert_real_close(&svd_vals_spectra, &svd_full_spectra);

    let eigh_vals_spectra = eigh_vals(&mut dense_executor, &rule, &hermitian).unwrap();
    let eigh_full_spectra = eigh_full(&mut dense_executor, &rule, &hermitian)
        .unwrap()
        .eigenvalues;
    assert_real_close(&eigh_vals_spectra, &eigh_full_spectra);

    let eig_vals_spectra = eig_vals(&mut dense_executor, &rule, &general).unwrap();
    let eig_full_spectra = eig_full(&mut dense_executor, &rule, &general)
        .unwrap()
        .eigenvalues;
    assert_eq!(eig_vals_spectra.len(), eig_full_spectra.len());
    for (a, b) in eig_vals_spectra.iter().zip(&eig_full_spectra) {
        assert_eq!(a.sector, b.sector);
        assert_eq!(a.values.len(), b.values.len());
        for (x, y) in a.values.iter().zip(&b.values) {
            assert!((x - y).norm() <= tol, "{x} vs {y}");
        }
    }
}

fn assert_identity_matrices(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    assert!(!matrices.is_empty());
    for (sector, rows, cols, matrix) in matrices {
        assert_eq!(rows, cols, "identity block must be square in {sector:?}");
        for col in 0..*cols {
            for row in 0..*rows {
                let expected = if row == col { 1.0 } else { 0.0 };
                let value = matrix[row + rows * col];
                assert!(
                    (value - expected).abs() < 1e-9,
                    "sector {sector:?} ({row},{col}): {value}"
                );
            }
        }
    }
}

fn default_context() -> TensorContractFusionExecutionContext<f64, TreeTransformBuiltinRuleCacheKey>
{
    TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default()
}

#[test]
fn adjoint_composition_gives_the_identity_on_the_bond() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let (q, _) = qr_compact(&mut dense_executor, &rule, &tensor).unwrap();
    let qh = tenet_tensors::adjoint(&rule, &q).unwrap();
    let mut context = default_context();
    let identity = crate::compose::compose(&mut context, &rule, &qh, &q).unwrap();
    assert_identity_matrices(&dense_sector_matrices(1, &identity));
}

#[test]
fn exp_of_a_hermitian_tensor_inverts_under_negation() {
    let rule = Z2FusionRule;
    let raw = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    // Keep the spectrum modest so exp(t) exp(-t) stays well conditioned.
    let tensor = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        raw.data().iter().map(|value| 0.1 * value).collect(),
        raw.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let negated = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        tensor.data().iter().map(|value| -value).collect(),
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let forward = exp(&mut dense_executor, &mut context, &rule, &tensor).unwrap();
    let backward = exp(&mut dense_executor, &mut context, &rule, &negated).unwrap();
    let identity = crate::compose::compose(&mut context, &rule, &forward, &backward).unwrap();
    assert_identity_matrices(&dense_sector_matrices(2, &identity));
}

#[test]
fn pinv_satisfies_the_moore_penrose_identity() {
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );
    let key_count = hom.fusion_tree_keys(&rule).len();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 1>::from_dims([leg_dim, leg_dim], [leg_dim]).unwrap(),
        hom,
        &rule,
        vec![vec![degeneracy; 3]; key_count],
    )
    .unwrap();
    let len = space.required_len().unwrap();
    let tensor = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        (0..len).map(|i| ((i * 3 + 2) % 11) as f64 - 5.0).collect(),
        space,
    )
    .unwrap();
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let plus = pinv(&mut dense_executor, &mut context, &rule, &tensor, 1e-12).unwrap();
    let tp = crate::compose::compose(&mut context, &rule, &tensor, &plus).unwrap();
    let tpt = crate::compose::compose(&mut context, &rule, &tp, &tensor).unwrap();
    for (index, (lhs, rhs)) in tpt.data().iter().zip(tensor.data()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-8,
            "Moore-Penrose violated at raw position {index}: {lhs} != {rhs}"
        );
    }
}

#[test]
fn inv_composes_to_the_identity() {
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();
    let inverse = inv(&mut dense_executor, &mut context, &rule, &tensor).unwrap();
    let identity = crate::compose::compose(&mut context, &rule, &tensor, &inverse).unwrap();
    assert_identity_matrices(&dense_sector_matrices(2, &identity));
}

#[test]
fn polar_decompositions_reconstruct_with_isometric_factors() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let (isometry, positive) =
        left_polar(&mut dense_executor, &mut context, &rule, &tensor).unwrap();
    let reconstructed = crate::compose::compose(&mut context, &rule, &isometry, &positive).unwrap();
    assert_svd_blocks_match(&tensor, &reconstructed);
    let wh = tenet_tensors::adjoint(&rule, &isometry).unwrap();
    let unit = crate::compose::compose(&mut context, &rule, &wh, &isometry).unwrap();
    assert_identity_matrices(&dense_sector_matrices(2, &unit));

    let (positive, isometry) =
        right_polar(&mut dense_executor, &mut context, &rule, &tensor).unwrap();
    let reconstructed = crate::compose::compose(&mut context, &rule, &positive, &isometry).unwrap();
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn single_precision_svd_and_eig_work_end_to_end() {
    use num_complex::Complex32;
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space = || {
        let hom = homspace();
        let key_count = hom.fusion_tree_keys(&rule).len();
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
            hom,
            &rule,
            vec![vec![degeneracy; 4]; key_count],
        )
        .unwrap()
    };
    let f32_space = space();
    let len = f32_space.required_len().unwrap();
    let tensor_f32 = TensorMap::<f32, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.5 - 5.0)
            .collect(),
        f32_space,
    )
    .unwrap();

    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_trunc(
        &mut dense_executor,
        &rule,
        &tensor_f32,
        &Truncation::rank(8),
    )
    .unwrap();
    assert!(svd.error > 0.0);

    // Reconstruct through an f32 contraction and compare against the
    // truncation error at single precision.
    let mut scaled_vh = svd.vh.clone();
    {
        let structure = std::sync::Arc::clone(scaled_vh.structure());
        for index in 0..structure.block_count() {
            let block = structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            let sector = key
                .codomain_tree()
                .coupled()
                .unwrap_or_else(|| rule.vacuum());
            let values = &svd
                .singular_values
                .iter()
                .find(|entry| entry.sector == sector)
                .unwrap()
                .values;
            let shape = block.shape().to_vec();
            let strides = block.strides().to_vec();
            let offset = block.offset();
            let count = shape.iter().product::<usize>();
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..count {
                let position = offset
                    + indices
                        .iter()
                        .zip(&strides)
                        .map(|(&i, &s)| i * s)
                        .sum::<usize>();
                scaled_vh.data_mut()[position] *= values[indices[0]] as f32;
                for axis in 0..shape.len() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
        }
    }
    let mut context =
        TensorContractFusionExecutionContext::<f32, TreeTransformBuiltinRuleCacheKey>::default();
    let reconstructed = crate::compose::compose(&mut context, &rule, &svd.u, &scaled_vh).unwrap();
    let distance = tensor_f32
        .data()
        .iter()
        .zip(reconstructed.data())
        .map(|(lhs, rhs)| ((lhs - rhs) as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        (distance - svd.error).abs() < 1e-3,
        "f32 distance {distance} != error {}",
        svd.error
    );

    // Complex32 general eigendecomposition returns Complex32 factors.
    let c32_space = space();
    let len = c32_space.required_len().unwrap();
    let tensor_c32 = TensorMap::<Complex32, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| {
                Complex32::new(
                    ((i * 3 + 1) % 13) as f32 - 6.0,
                    ((i * 5 + 2) % 11) as f32 - 5.0,
                )
            })
            .collect(),
        c32_space,
    )
    .unwrap();
    let eig = eig_full(&mut dense_executor, &rule, &tensor_c32).unwrap();
    assert!(!eig.eigenvalues.is_empty());
    for entry in &eig.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!(pair[0].norm() >= pair[1].norm() - 1e-6);
        }
    }
    let _: &TensorMap<Complex32, 2, 1> = &eig.v;
}

#[test]
fn positive_diagonal_gauge_matches_tensorkit_qr_reference() {
    // TensorKit 0.17.0 / MatrixAlgebraKit 0.6.8 crosscheck:
    //   A = [-1 2; 3 4; 5 -6]; Q, R = MatrixAlgebraKit.qr_compact(A)
    // (default `positive = true` since MAK 0.6.8). Column-major reference:
    let q_ref = [
        -0.16903085094570325,
        0.50709255283711,
        0.8451542547285166,
        0.21398024625545642,
        0.8559209850218259,
        -0.4707565417620042,
    ];
    let r_ref = [
        5.916079783099615,
        0.0,
        -3.380617018914066,
        6.676183683170241,
    ];
    // Start from the equally valid un-gauged QR with both diagonal signs
    // flipped (Q -> -Q, R -> -R); the gauge must restore the reference.
    let mut q: Vec<f64> = q_ref.iter().map(|v| -v).collect();
    let mut r: Vec<f64> = r_ref.iter().map(|v| -v).collect();
    crate::factorize::positive_diagonal_gauge(&mut q, 3, &mut r, 2, 2);
    for (value, reference) in q.iter().zip(&q_ref) {
        assert!(
            (value - reference).abs() < 1e-14,
            "Q {value} != {reference}"
        );
    }
    for (value, reference) in r.iter().zip(&r_ref) {
        assert!(
            (value - reference).abs() < 1e-14,
            "R {value} != {reference}"
        );
    }
}

#[test]
fn positive_diagonal_gauge_complex_phase_and_zero_diagonal() {
    use num_complex::Complex64;
    let c = Complex64::new;
    // q: 3 x 3, r: 3 x 3 upper triangular with complex diagonal phases and a
    // zero diagonal entry (row 1), column-major.
    let q: Vec<Complex64> = (0..9)
        .map(|i| c((i as f64 * 0.7 - 2.0).sin(), (i as f64 * 1.3 + 0.5).cos()))
        .collect();
    let r = vec![
        c(-3.0, 4.0),
        c(0.0, 0.0),
        c(0.0, 0.0),
        c(1.0, -2.0),
        c(0.0, 0.0),
        c(0.0, 0.0),
        c(0.5, 0.25),
        c(2.0, 1.0),
        c(0.0, -7.0),
    ];
    let product = |q: &[Complex64], r: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 9];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..3 {
                    out[row + 3 * col] += q[row + 3 * k] * r[k + 3 * col];
                }
            }
        }
        out
    };
    let before = product(&q, &r);
    let mut q_gauged = q.clone();
    let mut r_gauged = r.clone();
    crate::factorize::positive_diagonal_gauge(&mut q_gauged, 3, &mut r_gauged, 3, 3);
    // Diagonal of R is real non-negative; the zero entry keeps phase 1.
    for j in 0..3 {
        let diagonal = r_gauged[j + 3 * j];
        assert!(
            diagonal.im.abs() < 1e-14,
            "R[{j},{j}] = {diagonal} not real"
        );
        assert!(diagonal.re >= 0.0, "R[{j},{j}] = {diagonal} negative");
    }
    assert_eq!(r_gauged[1 + 3 * 1], c(0.0, 0.0));
    assert_eq!(q_gauged[3], q[3], "zero diagonal must not rescale Q column");
    // Q * R is unchanged.
    let after = product(&q_gauged, &r_gauged);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn svd_compact_gauge_matches_matrixalgebrakit_phase_rule() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut u = vec![
        c(3.0, 4.0),
        c(1.0, -1.0),
        c(-2.0, 0.5),
        c(0.25, -0.5),
        c(-4.0, 0.0),
        c(1.0, 2.0),
    ];
    let mut vh = vec![
        c(0.5, -1.0),
        c(-0.25, 0.75),
        c(1.0, 0.0),
        c(0.0, -2.0),
        c(-1.5, 0.25),
        c(0.75, -0.5),
    ];
    let sigma = [2.0, 0.75];
    let product = |u: &[Complex64], vh: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 9];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..2 {
                    out[row + 3 * col] += u[row + 3 * k] * sigma[k] * vh[k + 2 * col];
                }
            }
        }
        out
    };
    let before = product(&u, &vh);
    crate::factorize::svd_compact_gauge(&mut u, 3, 3, &mut vh, 2, 3, 2);
    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = u[row + 3 * col];
        assert!(pivot.im.abs() < 1e-14, "pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "pivot {pivot} negative");
    }
    let after = product(&u, &vh);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn eigenvector_gauge_matches_matrixalgebrakit_phase_rule() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut vectors = vec![
        c(3.0, 4.0),
        c(1.0, -1.0),
        c(-2.0, 0.5),
        c(0.25, -0.5),
        c(-4.0, 0.0),
        c(1.0, 2.0),
    ];

    crate::factorize::eigenvector_gauge(&mut vectors, 3, 3, 2);

    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = vectors[row + 3 * col];
        assert!(pivot.im.abs() < 1e-14, "pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "pivot {pivot} negative");
    }
}

#[test]
fn svd_full_gauge_fixes_extra_vh_rows_without_changing_product() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut u = vec![c(0.0, -2.0), c(0.25, 0.5), c(1.0, -1.0), c(-3.0, 0.0)];
    let mut vh = vec![
        c(1.0, 0.5),
        c(-0.25, 0.75),
        c(1.0, -1.0),
        c(0.5, -0.5),
        c(2.0, 0.0),
        c(-0.5, 0.25),
        c(-1.0, 0.75),
        c(0.0, -1.5),
        c(0.25, 0.0),
    ];
    let sigma = [1.5, 0.7];
    let product = |u: &[Complex64], vh: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 6];
        for col in 0..3 {
            for row in 0..2 {
                for k in 0..2 {
                    out[row + 2 * col] += u[row + 2 * k] * sigma[k] * vh[k + 3 * col];
                }
            }
        }
        out
    };
    let before = product(&u, &vh);
    crate::factorize::svd_full_gauge(&mut u, 2, 2, &mut vh, 3, 3);
    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = u[row + 2 * col];
        assert!(pivot.im.abs() < 1e-14, "U pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "U pivot {pivot} negative");
    }
    let extra_pivot = vh[2]; // row 2, col 0 (row + 3 * col)
    assert!(
        extra_pivot.im.abs() < 1e-14,
        "Vh pivot {extra_pivot} not real"
    );
    assert!(extra_pivot.re >= 0.0, "Vh pivot {extra_pivot} negative");
    let after = product(&u, &vh);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn qr_compact_positive_gauge_idempotent_on_isometry() {
    for rule_case in [0usize, 1usize] {
        if rule_case == 0 {
            let rule = Z2FusionRule;
            let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
            let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
            let (q, _) = qr_compact(&mut dense_executor, &rule, &tensor).unwrap();
            let (q2, r2) = qr_compact(&mut dense_executor, &rule, &q).unwrap();
            assert_svd_blocks_match(&q, &q2);
            assert_identity_sector_matrices(&dense_sector_matrices(1, &r2));
        } else {
            let rule = SU2FusionRule;
            let tensor = tsvd_test_tensor(
                &rule,
                &[
                    SU2Irrep::from_twice_spin(0).sector_id(),
                    SU2Irrep::from_twice_spin(1).sector_id(),
                ],
            );
            let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
            let (q, _) = qr_compact(&mut dense_executor, &rule, &tensor).unwrap();
            let (q2, r2) = qr_compact(&mut dense_executor, &rule, &q).unwrap();
            assert_svd_blocks_match(&q, &q2);
            assert_identity_sector_matrices(&dense_sector_matrices(1, &r2));
        }
    }
}

#[test]
fn lq_compact_positive_gauge_idempotent_on_isometry() {
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let (_, q) = lq_compact(&mut dense_executor, &rule, &tensor).unwrap();
    let (l2, q2) = lq_compact(&mut dense_executor, &rule, &q).unwrap();
    assert_svd_blocks_match(&q, &q2);
    assert_identity_sector_matrices(&dense_sector_matrices(1, &l2));
}

fn assert_identity_sector_matrices(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    for (sector, rows, cols, matrix) in matrices {
        assert_eq!(rows, cols, "sector {sector:?}: expected square factor");
        for col in 0..*cols {
            for row in 0..*rows {
                let expected = if row == col { 1.0 } else { 0.0 };
                let value = matrix[row + rows * col];
                assert!(
                    (value - expected).abs() < 1e-9,
                    "sector {sector:?}: entry ({row},{col}) = {value}"
                );
            }
        }
    }
}
