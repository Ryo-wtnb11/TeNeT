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
    let svd = tsvd_fusion(&mut dense_executor, rule, &tensor).unwrap();

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

    let mut scaled_vt = svd.vt.clone();
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
