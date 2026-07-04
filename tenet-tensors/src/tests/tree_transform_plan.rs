use super::*;

#[test]
fn tree_transform_compile_keyed_pairs_tree_blocks_by_key_not_index_for_all_numeric_dtypes() {
    assert_tree_multi_keyed_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        vec![7020.0, 9240.0, 3510.0, 4620.0],
    );
    assert_tree_multi_keyed_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        vec![7020.0, 9240.0, 3510.0, 4620.0],
    );
    assert_tree_multi_keyed_dtype(
        vec![1_i32, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        vec![7020, 9240, 3510, 4620],
    );
    assert_tree_multi_keyed_dtype(
        vec![1_i64, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        vec![7020, 9240, 3510, 4620],
    );
    assert_tree_multi_keyed_dtype(
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
        ],
        vec![
            Complex32::new(10.0, 0.0),
            Complex32::new(100.0, 0.0),
            Complex32::new(1000.0, 0.0),
            Complex32::new(20.0, 0.0),
            Complex32::new(200.0, 0.0),
            Complex32::new(2000.0, 0.0),
        ],
        vec![
            Complex32::new(7020.0, 0.0),
            Complex32::new(9240.0, 0.0),
            Complex32::new(3510.0, 0.0),
            Complex32::new(4620.0, 0.0),
        ],
    );
    assert_tree_multi_keyed_dtype(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(100.0, 0.0),
            Complex64::new(1000.0, 0.0),
            Complex64::new(20.0, 0.0),
            Complex64::new(200.0, 0.0),
            Complex64::new(2000.0, 0.0),
        ],
        vec![
            Complex64::new(7020.0, 0.0),
            Complex64::new(9240.0, 0.0),
            Complex64::new(3510.0, 0.0),
            Complex64::new(4620.0, 0.0),
        ],
    );
}

#[test]
fn tree_transform_rejects_invalid_block_specs_at_compile_time() {
    let space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![1.0; 8],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 8], space, structure).unwrap();

    let err = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
        )],
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::CoefficientCountMismatch {
            expected: 4,
            actual: 2,
        }
    );

    let err = TreeTransformStructure::compile(
        &dst,
        &src,
        &[
            TreeTransformBlockSpec::single(0, 0, 1.0),
            TreeTransformBlockSpec::single(0, 1, 1.0),
        ],
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::DuplicateTransformDestination { dst_block: 0 }
    );
}

#[test]
fn tree_transform_compile_keyed_rejects_missing_tree_block_key() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let key1 = BlockKey::sector_ids([1]);
    let key2 = BlockKey::sector_ids([2]);
    let src_structure = packed_fixture_structure(2, [(key1.clone(), vec![2, 2])]).unwrap();
    let dst_structure = packed_fixture_structure(2, [(key1.clone(), vec![2, 2])]).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 4], src_space, src_structure)
            .unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure)
            .unwrap();

    let err = TreeTransformStructure::compile_keyed(
        &dst,
        &src,
        &[TreeTransformKeyBlockSpec::single(key2.clone(), key1, 1.0)],
    )
    .unwrap_err();

    assert_eq!(err, OperationError::MissingBlockKey { key: key2 });
}

#[test]
fn tree_transform_group_block_spec_preserves_group_identity_and_ordered_keys() {
    let group_key = FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true]);
    let dst_key1 = BlockKey::sector_ids([101, 201]);
    let dst_key2 = BlockKey::sector_ids([102, 202]);
    let src_key = BlockKey::sector_ids([301, 401]);
    let spec = TreeTransformGroupBlockSpec::multi(
        group_key.clone(),
        [dst_key1.clone(), dst_key2.clone()],
        [src_key.clone()],
        vec![2.0_f64, 3.0],
    );

    assert_eq!(spec.group_key(), &group_key);
    assert_eq!(
        spec.group_key()
            .codomain_uncoupled()
            .iter()
            .map(|sector| sector.id())
            .collect::<Vec<_>>(),
        vec![10, 20]
    );
    assert_eq!(
        spec.group_key()
            .domain_uncoupled()
            .iter()
            .map(|sector| sector.id())
            .collect::<Vec<_>>(),
        vec![30]
    );
    assert_eq!(spec.group_key().codomain_is_dual(), &[false, true]);
    assert_eq!(spec.group_key().domain_is_dual(), &[true]);
    assert_eq!(spec.dst_keys(), &[dst_key1, dst_key2]);
    assert_eq!(spec.src_keys(), &[src_key]);
    assert_eq!(spec.coefficients_src_by_dst(), &[2.0, 3.0]);
}

#[test]
fn unique_tree_transform_plan_builder_creates_single_specs_in_source_order() {
    let src_key1 = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_key2 = fusion_tree_test_key([0, 1], [1], 1, [false, false], [false]);
    let dst_key1 = fusion_tree_test_key([0, 1], [1], 1, [false, false], [false]);
    let dst_key2 = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_tree1 = expect_tree_key(&src_key1);
    let src_tree2 = expect_tree_key(&src_key2);
    let dst_tree1 = expect_tree_key(&dst_key1);
    let dst_tree2 = expect_tree_key(&dst_key2);
    let src_structure = packed_fixture_structure(
        2,
        [
            (src_key1.clone(), vec![1, 1]),
            (src_key2.clone(), vec![1, 1]),
        ],
    )
    .unwrap();

    let plan = build_unique_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::transpose([1, 0], [0]),
        &src_structure,
        |src| {
            if src == &src_tree1 {
                Ok((dst_tree1.clone(), 2.0_f64))
            } else if src == &src_tree2 {
                Ok((dst_tree2.clone(), 3.0_f64))
            } else {
                panic!("unexpected source key {src:?}")
            }
        },
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 2);
    assert_eq!(plan.specs()[0].group_key(), &src_tree1.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[src_key1]);
    assert_eq!(plan.specs()[0].dst_keys(), &[dst_key1]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[2.0]);
    assert_eq!(plan.specs()[1].group_key(), &src_tree2.group_key());
    assert_eq!(plan.specs()[1].src_keys(), &[src_key2]);
    assert_eq!(plan.specs()[1].dst_keys(), &[dst_key2]);
    assert_eq!(plan.specs()[1].coefficients_src_by_dst(), &[3.0]);
}

#[test]
fn single_output_unique_tree_transform_helper_rejects_simple_fusion() {
    let src_key = fusion_tree_test_key([1, 1, 1], [1], 1, [false, false, false], [false]);
    let src_structure = packed_fixture_structure(4, [(src_key, vec![1, 1, 1, 1])]).unwrap();
    let operation = TreeTransformOperation::transpose([2, 1, 0], [0]);

    let err = build_unique_tree_transform_group_plan(
        &SimpleSu2Rule,
        operation.clone(),
        &src_structure,
        |_| -> Result<(FusionTreeBlockKey, f64), OperationError> {
            unreachable!("non-Unique fusion must be rejected before transforming keys")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedFusionStyle {
            operation,
            style: FusionStyleKind::Simple,
        }
    );
}

#[test]
fn tree_transform_plan_builder_accepts_simple_multi_destination_callback() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let src_tree0 = expect_tree_key(&src_key0);
    let src_tree1 = expect_tree_key(&src_key1);
    let src_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

    let plan = build_tree_transform_group_plan(&SimpleSu2Rule, operation, &src_structure, |src| {
        if src == &src_tree0 {
            Ok(vec![
                (src_tree0.clone(), 0.5_f64),
                (src_tree1.clone(), 0.866_025_403_784_438_6),
            ])
        } else if src == &src_tree1 {
            Ok(vec![
                (src_tree0.clone(), 0.866_025_403_784_438_6),
                (src_tree1.clone(), -0.5),
            ])
        } else {
            panic!("unexpected source key {src:?}")
        }
    })
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    let spec = &plan.specs()[0];
    assert_eq!(spec.src_keys(), &[src_key0.clone(), src_key1.clone()]);
    assert_eq!(spec.dst_keys(), &[src_key0, src_key1]);
    assert_eq!(
        spec.coefficients_src_by_dst(),
        &[0.5, 0.866_025_403_784_438_6, 0.866_025_403_784_438_6, -0.5]
    );
}

#[test]
fn multiplicity_free_su2_plan_builder_creates_generic_recoupling_block() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let src_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

    let plan =
        build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation, &src_structure)
            .unwrap();

    assert_eq!(plan.specs().len(), 1);
    let spec = &plan.specs()[0];
    assert_eq!(spec.src_keys(), &[src_key0.clone(), src_key1.clone()]);
    assert_eq!(spec.dst_keys(), &[src_key0, src_key1]);
    let expected = [0.5, 0.866_025_403_784_438_6, 0.866_025_403_784_438_6, -0.5];
    assert_eq!(spec.coefficients_src_by_dst().len(), expected.len());
    for (&actual, expected) in spec.coefficients_src_by_dst().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "coefficient {actual} != {expected}"
        );
    }

    let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        src_structure.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![0.0, 0.0],
        dst_space,
        src_structure.clone(),
    )
    .unwrap();
    let structure = plan
        .compile_structures(&src_structure, &src_structure)
        .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert!(structure.has_pack_gemm_scatter_blocks());
    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_pair_transform_public_helper_executes_su2_recoupling_block() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], dst_space, structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

    let compiled = tree_transform_structure(&SU2FusionRule, operation, &dst, &src).unwrap();
    assert!(compiled.has_pack_gemm_scatter_blocks());
    braid_into(
        &SU2FusionRule,
        [0, 2, 1, 3],
        [],
        [0, 1, 2, 3],
        [],
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_transform_recoupling_replays_complex_data_with_real_structural_coefficients() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<Complex64, 4, 0>::from_vec_with_structure(
        vec![Complex64::new(10.0, 1.0), Complex64::new(20.0, -2.0)],
        src_space,
        structure.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 4, 0>::from_vec_with_structure(
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        dst_space,
        structure.clone(),
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let compiled = tree_transform_structure(&SU2FusionRule, operation, &dst, &src).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    let first = dst.data().to_vec();
    assert_eq!(workspace.source_len(), 2);
    assert_eq!(workspace.destination_len(), 2);

    dst.data_mut().fill(Complex64::new(0.0, 0.0));
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    assert_eq!(dst.data(), first.as_slice());
    assert_eq!(workspace.source_len(), 2);
    assert_eq!(workspace.destination_len(), 2);
}

#[test]
fn tree_transform_structure_sorts_replay_blocks_by_tensorkit_weight() {
    let dst_structure =
        BlockStructure::packed_column_major(1, [vec![1], vec![3], vec![3]]).unwrap();
    let src_structure =
        BlockStructure::packed_column_major(1, [vec![1], vec![3], vec![3]]).unwrap();
    let structure = TreeTransformStructure::compile_structures(
        &dst_structure,
        &src_structure,
        &[
            TreeTransformBlockSpec::single(0, 0, 1.0),
            TreeTransformBlockSpec::multi(vec![1, 2], vec![1, 2], vec![1.0, 0.0, 0.0, 1.0]),
        ],
    )
    .unwrap();

    assert_eq!(structure.replay_weights(), vec![12, 1]);
}

#[test]
fn tree_transform_structure_replays_su2_recoupling_without_recompiling() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let mut src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], dst_space, block_structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let structure = tree_transform_structure(&SU2FusionRule, operation, &dst, &src).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    let expected = |initial: [f64; 2], source: [f64; 2], alpha: f64, beta: f64| {
        let c = 0.866_025_403_784_438_6;
        [
            beta * initial[0] + alpha * (0.5 * source[0] + c * source[1]),
            beta * initial[1] + alpha * (c * source[0] - 0.5 * source[1]),
        ]
    };

    assert!(structure.has_pack_gemm_scatter_blocks());
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    let expected_first = expected([0.0, 0.0], [10.0, 20.0], 1.0, 0.0);
    assert!((dst.data()[0] - expected_first[0]).abs() < 1.0e-12);
    assert!((dst.data()[1] - expected_first[1]).abs() < 1.0e-12);
    assert_eq!(workspace.source_len(), 2);
    assert_eq!(workspace.destination_len(), 2);

    src.data_mut().copy_from_slice(&[3.0, -4.0]);
    dst.data_mut().copy_from_slice(&[1.0, 2.0]);
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        2.0,
        -1.0,
    )
    .unwrap();
    let expected_second = expected([1.0, 2.0], [3.0, -4.0], 2.0, -1.0);
    assert!((dst.data()[0] - expected_second[0]).abs() < 1.0e-12);
    assert!((dst.data()[1] - expected_second[1]).abs() < 1.0e-12);
}

#[test]
fn tree_row_memo_survives_structure_change() {
    // TensorKit fstranspose/fsbraid cache parity: a truncation step changes the tree
    // subset of a structure, so the sector-keyed plan cache misses — but
    // recoupling rows for trees shared with earlier structures must be
    // reused from the tree-granular memo instead of recomputing F/R-symbol
    // contractions.
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    let make = |keys: &[BlockKey]| {
        let blocks: Vec<_> = keys
            .iter()
            .map(|key| (key.clone(), vec![1usize, 1, 1, 1]))
            .collect();
        let block_structure = packed_fixture_structure(4, blocks).unwrap();
        let elements = keys.len();
        let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
        let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
        let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
            vec![1.0; elements],
            src_space,
            block_structure.clone(),
        )
        .unwrap();
        let dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
            vec![0.0; elements],
            dst_space,
            block_structure,
        )
        .unwrap();
        (dst, src)
    };

    let (dst1, src1) = make(&[src_key0.clone(), src_key1.clone()]);
    cache
        .get_or_compile_tree_pair(&SU2FusionRule, operation.clone(), &dst1, &src1)
        .unwrap();
    let misses_after_first = cache.stats().tree_row_misses();
    assert!(misses_after_first > 0);
    assert_eq!(cache.stats().tree_row_hits(), 0);

    // Structure change (a new coupled sector appears, e.g. after a bond
    // grows in a sweep): the sector-keyed plan cache misses, but rows for
    // the previously seen trees come from the memo — only the new sector's
    // trees compute fresh F/R-symbol contractions.
    let src_key2 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(2),
        [false, false, false, false],
        [2, 3],
        [1, 1, 1],
    );
    let (dst2, src2) = make(&[src_key0.clone(), src_key1.clone(), src_key2.clone()]);
    cache
        .get_or_compile_tree_pair(&SU2FusionRule, operation.clone(), &dst2, &src2)
        .unwrap();
    assert_eq!(cache.stats().plan_misses(), 2);
    assert!(cache.stats().tree_row_hits() >= misses_after_first);
    assert!(cache.stats().tree_row_misses() > misses_after_first);
}

#[test]
fn tree_transform_cache_reuses_su2_recoupling_descriptor() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], dst_space, block_structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    {
        let structure = cache
            .get_or_compile_tree_pair(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_tree_pair(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    let structure = cache
        .get_or_compile_tree_pair(&SU2FusionRule, operation, &dst, &src)
        .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_transform_cache_reuses_all_codomain_plan_across_degeneracy_shapes() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let small_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let large_structure = packed_fixture_structure(
        4,
        [(src_key0, vec![2, 1, 1, 1]), (src_key1, vec![2, 1, 1, 1])],
    )
    .unwrap();
    let small_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let large_space = TensorMapSpace::<4, 0>::from_dims([2, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        small_space.clone(),
        small_structure.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![0.0, 0.0],
        small_space,
        small_structure,
    )
    .unwrap();
    let src_large = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        large_space.clone(),
        large_structure.clone(),
    )
    .unwrap();
    let dst_large = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![0.0, 0.0, 0.0, 0.0],
        large_space,
        large_structure,
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    {
        let structure = cache
            .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst_large, &src_large)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 2);

    let structure = cache
        .get_or_compile_all_codomain(&SU2FusionRule, operation, &dst, &src)
        .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_transform_execution_context_reuses_all_codomain_cache() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let mut src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut context =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    assert_eq!(context.cache().stats(), TreeTransformCacheStats::default());

    all_codomain_tree_transform_into_with_context(
        &mut context,
        &SU2FusionRule,
        operation.clone(),
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
    assert_eq!(context.cache().stats().plan_hits(), 0);
    assert_eq!(context.cache().stats().plan_misses(), 1);
    assert_eq!(context.cache().stats().structure_hits(), 0);
    assert_eq!(context.cache().stats().structure_misses(), 1);
    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);

    src.data_mut().copy_from_slice(&[3.0, -4.0]);
    dst.data_mut().copy_from_slice(&[1.0, 2.0]);
    context
        .all_codomain_tree_transform_into(&SU2FusionRule, operation, &mut dst, &src, 2.0, -1.0)
        .unwrap();

    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
    assert_eq!(context.cache().stats().plan_hits(), 1);
    assert_eq!(context.cache().stats().plan_misses(), 1);
    assert_eq!(context.cache().stats().structure_hits(), 1);
    assert_eq!(context.cache().stats().structure_misses(), 1);
    let c = 0.866_025_403_784_438_6;
    assert!((dst.data()[0] - (-1.0 + 2.0 * (0.5 * 3.0 + c * -4.0))).abs() < 1.0e-12);
    assert!((dst.data()[1] - (-2.0 + 2.0 * (c * 3.0 - 0.5 * -4.0))).abs() < 1.0e-12);
    context.cache_mut().reset_stats();
    assert_eq!(context.cache().stats(), TreeTransformCacheStats::default());
}

#[test]
fn tree_transform_execution_context_no_cache_rebuilds_without_retaining_entries() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut context =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    context
        .cache_mut()
        .set_policy(OperationCachePolicy::NoCache);

    for expected_misses in 1..=2 {
        context
            .all_codomain_tree_transform_into(
                &SU2FusionRule,
                operation.clone(),
                &mut dst,
                &src,
                1.0,
                0.0,
            )
            .unwrap();
        assert_eq!(context.cache().plan_len(), 0);
        assert_eq!(context.cache().structure_len(), 0);
        assert_eq!(context.cache().stats().plan_hits(), 0);
        assert_eq!(context.cache().stats().structure_hits(), 0);
        assert_eq!(context.cache().stats().plan_misses(), expected_misses);
        assert_eq!(context.cache().stats().structure_misses(), expected_misses);
    }
}

#[test]
fn tree_transform_execution_context_task_local_lru_evicts_old_transformer() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut context =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    context
        .cache_mut()
        .set_policy(OperationCachePolicy::task_local_lru(1));

    context
        .tree_transform_into(&SU2FusionRule, operation.clone(), &mut dst, &src, 1.0, 0.0)
        .unwrap();
    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);

    context
        .all_codomain_tree_transform_into(
            &SU2FusionRule,
            operation.clone(),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);

    context
        .tree_transform_into(&SU2FusionRule, operation, &mut dst, &src, 1.0, 0.0)
        .unwrap();
    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
    assert_eq!(context.cache().stats().plan_hits(), 0);
    assert_eq!(context.cache().stats().structure_hits(), 0);
    assert_eq!(context.cache().stats().plan_misses(), 3);
    assert_eq!(context.cache().stats().structure_misses(), 3);
}

#[test]
fn tree_transform_execution_context_separates_tree_pair_and_all_codomain_scopes() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut context =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    context
        .tree_transform_into(&SU2FusionRule, operation.clone(), &mut dst, &src, 1.0, 0.0)
        .unwrap();
    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);

    dst.data_mut().copy_from_slice(&[0.0, 0.0]);
    context
        .all_codomain_tree_transform_into(&SU2FusionRule, operation, &mut dst, &src, 1.0, 0.0)
        .unwrap();

    assert_eq!(context.cache().plan_len(), 2);
    assert_eq!(context.cache().structure_len(), 2);
    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_pair_plan_builder_handles_su2_one_by_one_domain_crossing() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [false],
        [],
        [],
        [],
        [],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [true],
        [true],
        [],
        [],
        [],
        [],
    ));
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();
    let dst_structure =
        packed_fixture_structure(2, [(expected_dst_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_tree_pair_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::permute([1], [0]),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    let spec = &plan.specs()[0];
    assert_eq!(spec.src_keys(), &[src_key]);
    assert_eq!(spec.dst_keys(), &[expected_dst_key]);
    assert_eq!(spec.coefficients_src_by_dst().len(), 1);
    assert!((spec.coefficients_src_by_dst()[0] - 1.0).abs() < 1.0e-12);
    plan.compile_structures(&dst_structure, &src_structure)
        .unwrap();
}

#[test]
fn tree_pair_transform_public_helper_executes_su2_domain_crossing() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [false],
        [],
        [],
        [],
        [],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [true],
        [true],
        [],
        [],
        [],
        [],
    ));
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let dst_structure =
        packed_fixture_structure(2, [(expected_dst_key.clone(), vec![1, 1])]).unwrap();
    let src_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let dst_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let src = TensorMap::<f64, 1, 1>::from_vec_with_structure(vec![7.0], src_space, src_structure)
        .unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_structure(vec![2.0], dst_space, dst_structure)
            .unwrap();
    permute_into(&SU2FusionRule, [1], [0], &mut dst, &src, 3.0, 5.0).unwrap();

    assert_eq!(dst.structure().block(0).unwrap().key(), &expected_dst_key);
    assert!((dst.data()[0] - 31.0).abs() < 1.0e-12);
}

#[test]
fn tree_pair_transform_public_helper_executes_su2_with_complex_data() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [false],
        [],
        [],
        [],
        [],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [true],
        [true],
        [],
        [],
        [],
        [],
    ));
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let dst_structure =
        packed_fixture_structure(2, [(expected_dst_key.clone(), vec![1, 1])]).unwrap();
    let src_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let dst_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let src = TensorMap::<Complex64, 1, 1>::from_vec_with_structure(
        vec![Complex64::new(7.0, 1.0)],
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_structure(
        vec![Complex64::new(2.0, -3.0)],
        dst_space,
        dst_structure,
    )
    .unwrap();
    let operation = TreeTransformOperation::permute([1], [0]);

    tree_transform_into(
        &SU2FusionRule,
        operation,
        &mut dst,
        &src,
        Complex64::new(3.0, 0.0),
        Complex64::new(5.0, 0.0),
    )
    .unwrap();

    assert_eq!(dst.structure().block(0).unwrap().key(), &expected_dst_key);
    assert!((dst.data()[0] - Complex64::new(31.0, -12.0)).norm() < 1.0e-12);
}

#[test]
fn tree_pair_operation_key_uses_tensorkit_global_source_axes() {
    let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();

    let local_domain_identity = build_tree_pair_transform_group_plan(
        &Z2FusionRule,
        TreeTransformOperation::permute([1, 0], [0]),
        &src_structure,
    )
    .unwrap_err();
    assert_eq!(
        local_domain_identity,
        OperationError::Core(CoreError::InvalidPermutation {
            permutation: vec![1, 0, 0],
            rank: 3,
        })
    );

    build_tree_pair_transform_group_plan(
        &Z2FusionRule,
        TreeTransformOperation::permute([1, 0], [2]),
        &src_structure,
    )
    .unwrap();
}

#[test]
fn tree_pair_transform_public_helper_executes_split_changing_permute() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [0, 1],
        Some(1),
        [false],
        [false, true],
        [],
        [],
        [],
        [1],
    ));
    let src_tree = expect_tree_key(&src_key);
    let operation = TreeTransformOperation::permute([0, 2], [1]);
    let (dst_tree, coefficient) =
        unique_permute_tree_pair(&Z2FusionRule, &src_tree, &[0, 2], &[1]).unwrap();
    let dst_key = BlockKey::from(dst_tree);
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let dst_structure = packed_fixture_structure(3, [(dst_key.clone(), vec![1, 1, 1])]).unwrap();
    let src_space = TensorMapSpace::<1, 2>::from_dims([1], [1, 1]).unwrap();
    let dst_space = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
    let src = TensorMap::<f64, 1, 2>::from_vec_with_structure(vec![7.0], src_space, src_structure)
        .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_structure(vec![2.0], dst_space, dst_structure)
            .unwrap();

    tree_transform_into(&Z2FusionRule, operation, &mut dst, &src, 3.0, 5.0).unwrap();

    assert_eq!(dst.structure().block(0).unwrap().key(), &dst_key);
    assert_eq!(dst.data(), &[3.0 * coefficient * 7.0 + 5.0 * 2.0]);
}

#[test]
fn tree_pair_transform_public_helper_compiles_against_actual_destination_structure() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [0, 1],
        Some(1),
        [false],
        [false, true],
        [],
        [],
        [],
        [1],
    ));
    let src_tree = expect_tree_key(&src_key);
    let operation = TreeTransformOperation::permute([0, 2], [1]);
    let (dst_tree, _) = unique_permute_tree_pair(&Z2FusionRule, &src_tree, &[0, 2], &[1]).unwrap();
    let expected_missing = BlockKey::from(dst_tree);
    let src_structure = packed_fixture_structure(3, [(src_key.clone(), vec![1, 1, 1])]).unwrap();
    let wrong_dst_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let src_space = TensorMapSpace::<1, 2>::from_dims([1], [1, 1]).unwrap();
    let dst_space = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
    let src = TensorMap::<f64, 1, 2>::from_vec_with_structure(vec![7.0], src_space, src_structure)
        .unwrap();
    let dst =
        TensorMap::<f64, 2, 1>::from_vec_with_structure(vec![0.0], dst_space, wrong_dst_structure)
            .unwrap();

    let err = tree_transform_structure(&Z2FusionRule, operation, &dst, &src).unwrap_err();

    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: expected_missing,
        }
    );
}

#[test]
fn multiplicity_free_product_tree_pair_plan_builder_handles_fz2_u1_su2_blocks() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let src_structure = src_space.subblock_structure();
    let dst_structure = dst_space.subblock_structure();

    let plan = build_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::permute([1, 0], [2]),
        src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 2);
    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    plan.compile_structures(dst_structure, src_structure)
        .unwrap();
}

#[test]
fn tree_pair_transform_public_helper_executes_product_fz2_u1_su2_blocks() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let initial_dst = dst.data().to_vec();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    let mut expected = initial_dst
        .iter()
        .map(|value| 3.0 * value)
        .collect::<Vec<_>>();
    for spec in plan.specs() {
        let src_key = &spec.src_keys()[0];
        let dst_key = &spec.dst_keys()[0];
        let src_offset = src.structure().block_by_key(src_key).unwrap().offset();
        let dst_offset = dst.structure().block_by_key(dst_key).unwrap().offset();
        expected[dst_offset] += 2.0 * spec.coefficients_src_by_dst()[0] * src.data()[src_offset];
    }

    tree_transform_into(&rule, operation, &mut dst, &src, 2.0, 3.0).unwrap();

    assert_eq!(dst.structure(), dst_space.subblock_structure());
    for (actual, expected) in dst.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_pair_transform_public_helper_executes_product_with_complex_data() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(10.0, 1.0), Complex64::new(20.0, -2.0)],
        src_space.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(1.0, 3.0), Complex64::new(2.0, -4.0)],
        dst_space.clone(),
    )
    .unwrap();
    let initial_dst = dst.data().to_vec();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    let alpha = Complex64::new(2.0, 0.0);
    let beta = Complex64::new(3.0, 0.0);
    let mut expected = initial_dst
        .iter()
        .map(|value| *value * beta)
        .collect::<Vec<_>>();
    for spec in plan.specs() {
        let src_key = &spec.src_keys()[0];
        let dst_key = &spec.dst_keys()[0];
        let src_offset = src.structure().block_by_key(src_key).unwrap().offset();
        let dst_offset = dst.structure().block_by_key(dst_key).unwrap().offset();
        expected[dst_offset] = expected[dst_offset]
            + src.data()[src_offset].scale_by_coefficient(spec.coefficients_src_by_dst()[0])
                * alpha;
    }

    tree_transform_into(&rule, operation, &mut dst, &src, alpha, beta).unwrap();

    assert_eq!(dst.structure(), dst_space.subblock_structure());
    assert_eq!(dst.data(), expected.as_slice());
}

#[test]
fn tree_transform_structure_replays_product_without_recompiling() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let mut src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    let structure = tree_transform_structure(&rule, operation, &dst, &src).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    assert_eq!(structure.block_count(), 2);
    assert!(!structure.has_pack_gemm_scatter_blocks());
    let expected_first = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        2.0,
        3.0,
    );
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();
    for (actual, expected) in dst.data().iter().zip(expected_first) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
    assert_eq!(workspace.source_len(), 0);
    assert_eq!(workspace.destination_len(), 0);

    src.data_mut().copy_from_slice(&[4.0, 5.0]);
    dst.data_mut().copy_from_slice(&[6.0, 7.0]);
    let expected_second = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        -1.0,
        0.5,
    );
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        -1.0,
        0.5,
    )
    .unwrap();
    for (actual, expected) in dst.data().iter().zip(expected_second) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_pair_transform_context_accepts_custom_host_storage() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src = test_host_read_fusion_tensor_map(vec![10.0_f64, 20.0], src_space);
    let mut dst = test_host_fusion_tensor_map(vec![1.0_f64, 2.0], dst_space);
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    let expected = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        2.0,
        3.0,
    );
    let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();

    tree_transform_into_with_context(&mut context, &rule, operation, &mut dst, &src, 2.0, 3.0)
        .unwrap();

    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
    for (actual, expected) in dst.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_transform_cache_reuses_product_plan_across_degeneracy_shapes() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let src_large_structure =
        column_major_structure_like(src_space.subblock_structure(), vec![2, 1, 1]);
    let dst_large_structure =
        column_major_structure_like(dst_space.subblock_structure(), vec![1, 2, 1]);
    let large_space = TensorMapSpace::<2, 1>::from_dims([2, 1], [1]).unwrap();
    let src_large = TensorMap::<f64, 2, 1>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        large_space.clone(),
        src_large_structure,
    )
    .unwrap();
    let dst_large = TensorMap::<f64, 2, 1>::from_vec_with_structure(
        vec![0.0, 0.0, 0.0, 0.0],
        large_space,
        dst_large_structure,
    )
    .unwrap();
    let mut cache = TreeTransformCache::<f64, RuleKey>::new();

    {
        let structure = cache
            .get_or_compile_tree_pair(&rule, operation.clone(), &dst, &src)
            .unwrap();
        assert_eq!(structure.block_count(), 2);
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_tree_pair(&rule, operation.clone(), &dst, &src)
            .unwrap();
        assert_eq!(structure.block_count(), 2);
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_tree_pair(&rule, operation, &dst_large, &src_large)
            .unwrap();
        assert_eq!(structure.block_count(), 2);
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 2);

    let structure = cache
        .get_or_compile_tree_pair(
            &rule,
            TreeTransformOperation::permute([1, 0], [2]),
            &dst,
            &src,
        )
        .unwrap();
    let plan = build_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::permute([1, 0], [2]),
        src.structure(),
    )
    .unwrap();
    let expected = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        2.0,
        3.0,
    );
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();
    for (actual, expected) in dst.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_pair_transform_context_storage_workspace_replays_and_caches_product_transform() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let allocations = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let src =
        TensorMap::<f64, 2, 1, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
            TrackingStorage::new(vec![10.0, 20.0], "source_ctx", allocations.clone()),
            src_space,
        )
        .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
            TrackingStorage::new(vec![1.0, 2.0], "destination_ctx", allocations),
            dst_space,
        )
        .unwrap();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    let expected = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        2.0,
        3.0,
    );
    let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();
    let mut storage_workspace = crate::storage_scratch::StorageTreeTransformWorkspace::<
        TrackingScratch<f64>,
        TrackingScratch<f64>,
    >::default();

    context
        .tree_transform_into_storage_workspace(
            &mut storage_workspace,
            &rule,
            operation,
            &mut dst,
            &src,
            2.0,
            3.0,
        )
        .unwrap();

    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
    for (actual, expected) in dst.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_transform_execution_context_reuses_product_tree_pair_cache() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let mut src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();
    let expected_first = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        2.0,
        3.0,
    );

    context
        .tree_transform_into(&rule, operation.clone(), &mut dst, &src, 2.0, 3.0)
        .unwrap();

    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
    for (actual, expected) in dst.data().iter().zip(expected_first) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }

    src.data_mut().copy_from_slice(&[4.0, 5.0]);
    dst.data_mut().copy_from_slice(&[6.0, 7.0]);
    let expected_second = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        -1.0,
        0.5,
    );
    tree_transform_into_with_context(&mut context, &rule, operation, &mut dst, &src, -1.0, 0.5)
        .unwrap();

    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
    for (actual, expected) in dst.data().iter().zip(expected_second) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_transform_execution_context_misses_on_different_tree_pair_operation() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();

    context
        .tree_transform_into(
            &rule,
            TreeTransformOperation::permute([1, 0], [2]),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);

    dst.data_mut().copy_from_slice(&[1.0, 2.0]);
    context
        .tree_transform_into(
            &rule,
            TreeTransformOperation::braid([1, 0], [2], [1, 0], [2]),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();

    assert_eq!(context.cache().plan_len(), 2);
    assert_eq!(context.cache().structure_len(), 2);
}

#[test]
fn unique_tree_transform_plan_builder_rejects_generic_fusion() {
    let src_key = fusion_tree_test_key([1, 1], [1], 1, [false, false], [false]);
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let operation = TreeTransformOperation::braid([1, 0], [0], [1, 0], [0]);

    let err = build_unique_tree_transform_group_plan(
        &GenericMultiplicityRule,
        operation.clone(),
        &src_structure,
        |_| -> Result<(FusionTreeBlockKey, f64), OperationError> {
            unreachable!("GenericFusion must be rejected before transforming keys")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedFusionStyle {
            operation,
            style: FusionStyleKind::Generic,
        }
    );
}

#[test]
fn tree_transform_operation_key_distinguishes_permute_from_explicit_braid() {
    assert!(TreeTransformOperation::permute([1, 0], [0]).requires_symmetric_braiding());
    assert!(!TreeTransformOperation::transpose([1, 0], [0]).requires_symmetric_braiding());
    assert!(!TreeTransformOperation::braid([1, 0], [0], [1, 0], [0]).requires_symmetric_braiding());
}

#[test]
fn unique_tree_transform_plan_builder_rejects_permute_without_symmetric_braiding() {
    let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let operation = TreeTransformOperation::permute([1, 0], [0]);

    let err = build_unique_tree_transform_group_plan(
        &UniqueAnyonicRule,
        operation.clone(),
        &src_structure,
        |_| -> Result<(FusionTreeBlockKey, f64), OperationError> {
            unreachable!("permutation must reject non-symmetric braiding before key transform")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedBraidingStyle {
            operation,
            style: BraidingStyleKind::Anyonic,
        }
    );
}

#[test]
fn unique_tree_transform_plan_builder_defers_explicit_no_braiding_to_crossing_logic() {
    let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_tree = expect_tree_key(&src_key);
    let src_structure = packed_fixture_structure(3, [(src_key.clone(), vec![1, 1, 1])]).unwrap();

    let plan = build_unique_tree_transform_group_plan(
        &UniquePlanarRule,
        TreeTransformOperation::braid([1, 0], [0], [1, 0], [0]),
        &src_structure,
        |src| Ok((src.clone(), 1.0_f64)),
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].group_key(), &src_tree.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[src_key.clone()]);
    assert_eq!(plan.specs()[0].dst_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn unique_all_codomain_braid_plan_builder_lowers_codomain_single_tree() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], Some(0), [false, true], [], [1]);
    let expected_dst_key =
        all_codomain_fusion_tree_test_key([1, 1], Some(0), [true, false], [], [1]);
    let src_tree = expect_tree_key(&src_key);
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueAnyonicRule,
        TreeTransformOperation::braid([1, 0], Vec::<usize>::new(), [0, 1], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].group_key(), &src_tree.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[-2.0]);
}

#[test]
fn unique_all_codomain_permute_plan_builder_lowers_symmetric_permutation() {
    let src_key = all_codomain_fusion_tree_test_key([1, 0], Some(1), [false, true], [], [1]);
    let expected_dst_key =
        all_codomain_fusion_tree_test_key([0, 1], Some(1), [true, false], [], [1]);
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn unique_all_codomain_plan_builder_rejects_domain_operation_scope() {
    let src_key = all_codomain_fusion_tree_test_key([1, 0], Some(1), [false, false], [], [1]);
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let operation = TreeTransformOperation::braid([1, 0], [0], [0, 1], [0]);

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        operation.clone(),
        &src_structure,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTreeTransformScope {
            operation,
            message: "all-codomain UniqueFusion lowering requires an empty domain operation",
        }
    );
}

#[test]
fn unique_all_codomain_plan_builder_accepts_explicit_vacuum_empty_domain() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair(
        FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, false], [], [1]),
        empty_fusion_tree_with_coupled(Some(0)),
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair(
        FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, false], [], [1]),
        empty_fusion_tree_with_coupled(Some(0)),
    ));
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn unique_all_codomain_plan_builder_rejects_explicit_nonvacuum_empty_domain() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair(
        FusionTreeKey::from_sector_ids([1, 0], Some(1), [false, false], [], [1]),
        empty_fusion_tree_with_coupled(Some(1)),
    ));
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::ExpectedAllCodomainFusionTree { index: 0 }
    );
}

#[test]
fn unique_all_codomain_plan_builder_rejects_nonempty_domain_tree() {
    let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::ExpectedAllCodomainFusionTree { index: 0 }
    );
}

#[test]
fn unique_all_codomain_permute_plan_builder_rejects_nonsymmetric_braiding() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], Some(0), [false, false], [], [1]);
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let operation = TreeTransformOperation::permute([1, 0], Vec::<usize>::new());

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueAnyonicRule,
        operation.clone(),
        &src_structure,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedBraidingStyle {
            operation,
            style: BraidingStyleKind::Anyonic,
        }
    );
}

#[test]
fn unique_tree_pair_plan_builder_lowers_domain_only_permutation() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [0, 1],
        Some(1),
        [false],
        [false, true],
        [],
        [],
        [],
        [1],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1, 0],
        Some(1),
        [false],
        [true, false],
        [],
        [],
        [],
        [1],
    ));
    let src_tree = expect_tree_key(&src_key);
    let src_structure = packed_fixture_structure(3, [(src_key.clone(), vec![1, 1, 1])]).unwrap();

    let plan = build_unique_tree_pair_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([0], [2, 1]),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].group_key(), &src_tree.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_codomain_domain_crossing_braid() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [true],
        [],
        [],
        [],
        [],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [true],
        [],
        [],
        [],
        [],
    ));
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_tree_pair_transform_group_plan(
        &UniqueAnyonicRule,
        TreeTransformOperation::braid([1], [0], [0], [1]),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[-2.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_cyclic_transpose() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [true],
        [],
        [],
        [],
        [],
    ));
    let expected_dst_key = src_key.clone();
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();
    let operation = TreeTransformOperation::transpose([1], [0]);

    let plan =
        build_unique_tree_pair_transform_group_plan(&UniqueZ2Rule, operation, &src_structure)
            .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_rank_four_cyclic_transpose() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1, 0],
        [1, 0],
        Some(1),
        [false, false],
        [false, false],
        [],
        [],
        [1],
        [1],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1, 1],
        [0, 0],
        Some(0),
        [true, false],
        [false, true],
        [],
        [],
        [1],
        [1],
    ));
    let src_structure = packed_fixture_structure(4, [(src_key.clone(), vec![1, 1, 1, 1])]).unwrap();
    let operation = TreeTransformOperation::transpose([2, 0], [3, 1]);

    let plan =
        build_unique_tree_pair_transform_group_plan(&UniqueZ2Rule, operation, &src_structure)
            .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn tree_transform_compile_grouped_lowers_to_replay_ready_structure() {
    let group_key = FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true]);
    let key10 = BlockKey::sector_ids([10]);
    let key20 = BlockKey::sector_ids([20]);
    let key100 = BlockKey::sector_ids([100]);
    let key200 = BlockKey::sector_ids([200]);
    let key300 = BlockKey::sector_ids([300]);
    let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
    let src_structure = packed_fixture_structure(
        2,
        [
            (key100.clone(), vec![2, 1]),
            (key300.clone(), vec![2, 1]),
            (key200.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_structure = packed_fixture_structure(
        2,
        [(key20.clone(), vec![2, 1]), (key10.clone(), vec![2, 1])],
    )
    .unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure)
            .unwrap();
    let structure = TreeTransformStructure::compile_grouped(
        &dst,
        &src,
        &[TreeTransformGroupBlockSpec::multi(
            group_key,
            [key10, key20],
            [key100, key200, key300],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        )],
    )
    .unwrap();
    let mut backend = HostTensorOperations;
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(structure.block_count(), 1);
    assert_eq!(dst.data(), &[7020.0, 9240.0, 3510.0, 4620.0]);
    assert_eq!(workspace.source_len(), 6);
    assert_eq!(workspace.destination_len(), 4);
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScratchAllocation {
    label: &'static str,
    len: usize,
}

#[derive(Clone, Debug)]
struct TrackingStorage<T> {
    data: Vec<T>,
    label: &'static str,
    allocations: std::rc::Rc<std::cell::RefCell<Vec<ScratchAllocation>>>,
}

#[derive(Clone, Debug)]
struct TrackingScratch<T> {
    data: Vec<T>,
}

impl<T> TrackingStorage<T> {
    fn new(
        data: Vec<T>,
        label: &'static str,
        allocations: std::rc::Rc<std::cell::RefCell<Vec<ScratchAllocation>>>,
    ) -> Self {
        Self {
            data,
            label,
            allocations,
        }
    }
}

impl<T> TensorStorage<T> for TrackingStorage<T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl<T> HostReadableStorage<T> for TrackingStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T> HostWritableStorage<T> for TrackingStorage<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Clone> SimilarStorage<T> for TrackingStorage<T> {
    type Similar = TrackingScratch<T>;

    fn similar_filled(&self, len: usize, value: T) -> Self::Similar
    where
        T: Clone,
    {
        self.allocations.borrow_mut().push(ScratchAllocation {
            label: self.label,
            len,
        });
        TrackingScratch {
            data: vec![value; len],
        }
    }
}

impl<T> TensorStorage<T> for TrackingScratch<T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl<T> HostReadableStorage<T> for TrackingScratch<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T> HostWritableStorage<T> for TrackingScratch<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Clone> tenet_core::ScratchStorage<T> for TrackingScratch<T> {
    fn reset_filled(&mut self, len: usize, value: T)
    where
        T: Clone,
    {
        self.data.clear();
        self.data.resize(len, value);
    }
}

#[test]
fn tree_transform_storage_scratch_allocates_from_source_and_destination_storage() {
    let group_key = FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true]);
    let key10 = BlockKey::sector_ids([10]);
    let key20 = BlockKey::sector_ids([20]);
    let key100 = BlockKey::sector_ids([100]);
    let key200 = BlockKey::sector_ids([200]);
    let key300 = BlockKey::sector_ids([300]);
    let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
    let src_structure = packed_fixture_structure(
        2,
        [
            (key100.clone(), vec![2, 1]),
            (key300.clone(), vec![2, 1]),
            (key200.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_structure = packed_fixture_structure(
        2,
        [(key20.clone(), vec![2, 1]), (key10.clone(), vec![2, 1])],
    )
    .unwrap();
    let allocations = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let src = TensorMap::<f64, 2, 0, Trivial, TrackingStorage<f64>>::from_storage_with_structure(
        TrackingStorage::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "source",
            allocations.clone(),
        ),
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0, Trivial, TrackingStorage<f64>>::from_storage_with_structure(
            TrackingStorage::new(vec![0.0; 4], "destination", allocations.clone()),
            dst_space,
            dst_structure,
        )
        .unwrap();
    let structure = TreeTransformStructure::compile_grouped(
        &dst,
        &src,
        &[TreeTransformGroupBlockSpec::multi(
            group_key,
            [key10, key20],
            [key100, key200, key300],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        )],
    )
    .unwrap();
    let mut workspace = crate::storage_scratch::StorageTreeTransformWorkspace::<
        TrackingScratch<f64>,
        TrackingScratch<f64>,
    >::default();

    tenet_operations::tree_transform_structure_with_storage_workspace_strided_kernel(
        &mut crate::StridedHostKernelAdapter,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[7020.0, 9240.0, 3510.0, 4620.0]);
    assert_eq!(
        allocations.borrow().as_slice(),
        &[
            ScratchAllocation {
                label: "source",
                len: 6,
            },
            ScratchAllocation {
                label: "destination",
                len: 4,
            },
        ],
    );

    let src_space2 = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    let dst_space2 = TensorMapSpace::<1, 0>::from_dims([3], []).unwrap();
    let src_structure2 = BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap();
    let dst_structure2 =
        BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1]]).unwrap();
    let src2 = TensorMap::<f64, 1, 0, Trivial, TrackingStorage<f64>>::from_storage_with_structure(
        TrackingStorage::new(vec![5.0, 7.0], "source2", allocations.clone()),
        src_space2,
        src_structure2,
    )
    .unwrap();
    let mut dst2 =
        TensorMap::<f64, 1, 0, Trivial, TrackingStorage<f64>>::from_storage_with_structure(
            TrackingStorage::new(vec![0.0; 3], "destination2", allocations.clone()),
            dst_space2,
            dst_structure2,
        )
        .unwrap();
    let structure2 = TreeTransformStructure::compile(
        &dst2,
        &src2,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1, 2],
            vec![0, 1],
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
        )],
    )
    .unwrap();

    tenet_operations::tree_transform_structure_with_storage_workspace_strided_kernel(
        &mut crate::StridedHostKernelAdapter,
        &mut workspace,
        &structure2,
        &mut dst2,
        &src2,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst2.data(), &[75.0, 150.0, 225.0]);
    // The second replay reuses the workspace buffers (same host placement),
    // so no new scratch is allocated from the second tensor pair.
    assert_eq!(
        allocations.borrow().as_slice(),
        &[
            ScratchAllocation {
                label: "source",
                len: 6,
            },
            ScratchAllocation {
                label: "destination",
                len: 4,
            },
        ],
    );
}

#[test]
fn tree_transform_compile_grouped_rejects_missing_tree_block_key() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let group_key = FusionTreeGroupKey::from_sector_ids([1], [1], [false], [true]);
    let present_key = BlockKey::sector_ids([1]);
    let missing_key = BlockKey::sector_ids([2]);
    let src_structure = packed_fixture_structure(2, [(present_key.clone(), vec![2, 2])]).unwrap();
    let dst_structure = packed_fixture_structure(2, [(present_key.clone(), vec![2, 2])]).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 4], src_space, src_structure)
            .unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure)
            .unwrap();

    let err = TreeTransformStructure::compile_grouped(
        &dst,
        &src,
        &[TreeTransformGroupBlockSpec::single(
            group_key,
            missing_key.clone(),
            present_key,
            1.0,
        )],
    )
    .unwrap_err();

    assert_eq!(err, OperationError::MissingBlockKey { key: missing_key });
}

#[test]
fn tree_transform_group_block_spec_from_groups_uses_source_group_and_ordered_keys() {
    let src_key1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src_key2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
    let dst_key1 = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
    let dst_key2 = fusion_tree_test_key([20, 10], [30], 8, [true, false], [true]);
    let src_structure = packed_fixture_structure(
        2,
        [
            (src_key1.clone(), vec![1, 1]),
            (src_key2.clone(), vec![1, 1]),
        ],
    )
    .unwrap();
    let dst_structure = packed_fixture_structure(
        2,
        [
            (dst_key1.clone(), vec![1, 1]),
            (dst_key2.clone(), vec![1, 1]),
        ],
    )
    .unwrap();
    let src_groups = src_structure.fusion_tree_groups();
    let dst_groups = dst_structure.fusion_tree_groups();

    let spec = TreeTransformGroupBlockSpec::from_block_groups(
        &dst_structure,
        &dst_groups[0],
        &src_structure,
        &src_groups[0],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .unwrap();

    assert_eq!(spec.group_key(), src_groups[0].group_key());
    assert_ne!(spec.group_key(), dst_groups[0].group_key());
    assert_eq!(spec.src_keys(), &[src_key1, src_key2]);
    assert_eq!(spec.dst_keys(), &[dst_key1, dst_key2]);
    assert_eq!(spec.coefficients_src_by_dst(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tree_transform_group_plan_compiles_across_degeneracy_shapes_without_layout_leakage() {
    let src_key1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src_key2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
    let dst_key1 = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
    let dst_key2 = fusion_tree_test_key([20, 10], [30], 8, [true, false], [true]);
    let src_small = packed_fixture_structure(
        2,
        [
            (src_key1.clone(), vec![2, 1]),
            (src_key2.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_small = packed_fixture_structure(
        2,
        [
            (dst_key1.clone(), vec![2, 1]),
            (dst_key2.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let src_large =
        packed_fixture_structure(2, [(src_key1, vec![3, 1]), (src_key2, vec![3, 1])]).unwrap();
    let dst_large =
        packed_fixture_structure(2, [(dst_key1, vec![3, 1]), (dst_key2, vec![3, 1])]).unwrap();
    let spec = TreeTransformGroupBlockSpec::from_block_groups(
        &dst_small,
        &dst_small.fusion_tree_groups()[0],
        &src_small,
        &src_small.fusion_tree_groups()[0],
        vec![1.0_f64, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let plan = TreeTransformGroupPlan::new(vec![spec]);
    let key =
        TreeTransformGroupPlanKey::from_plan(TreeTransformOperation::transpose([1, 0], [0]), &plan);
    let large_spec = TreeTransformGroupBlockSpec::from_block_groups(
        &dst_large,
        &dst_large.fusion_tree_groups()[0],
        &src_large,
        &src_large.fusion_tree_groups()[0],
        vec![1.0_f64, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let large_plan = TreeTransformGroupPlan::new(vec![large_spec]);
    let large_key = TreeTransformGroupPlanKey::from_plan(
        TreeTransformOperation::transpose([1, 0], [0]),
        &large_plan,
    );
    let mut cache = TreeTransformGroupPlanCache::new();

    cache.insert(key.clone(), plan.clone());

    let small_structure = plan.compile_structures(&dst_small, &src_small).unwrap();
    let cached = cache.get(&large_key).unwrap();
    let large_structure = cached.compile_structures(&dst_large, &src_large).unwrap();

    assert_eq!(key, large_key);
    assert_eq!(cache.len(), 1);
    assert_eq!(plan.specs().len(), 1);
    assert_eq!(small_structure.block_count(), 1);
    assert_eq!(large_structure.block_count(), 1);
    assert_eq!(small_structure.workspace_lens(), (4, 4));
    assert_eq!(large_structure.workspace_lens(), (6, 6));
}

#[test]
fn tree_transform_group_plan_cache_key_tracks_operation_but_not_coefficients() {
    let group_key = FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true]);
    let dst_key = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
    let src_key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let plan_a = TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::single(
        group_key.clone(),
        dst_key.clone(),
        src_key.clone(),
        2.0_f64,
    )]);
    let plan_b = TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::single(
        group_key, dst_key, src_key, 3.0_f64,
    )]);

    let transpose = TreeTransformGroupPlanKey::from_plan(
        TreeTransformOperation::transpose([1, 0], [0]),
        &plan_a,
    );
    let same_operation_different_coefficients = TreeTransformGroupPlanKey::from_plan(
        TreeTransformOperation::transpose([1, 0], [0]),
        &plan_b,
    );
    let different_permutation = TreeTransformGroupPlanKey::from_plan(
        TreeTransformOperation::transpose([0, 1], [0]),
        &plan_a,
    );
    let braid = TreeTransformGroupPlanKey::from_plan(
        TreeTransformOperation::braid([1, 0], [0], [2], [0]),
        &plan_a,
    );

    assert_eq!(transpose, same_operation_different_coefficients);
    assert_ne!(transpose, different_permutation);
    assert_ne!(transpose, braid);
}

#[test]
fn tree_transform_sector_plan_key_is_rule_scope_and_source_sector_only() {
    let src_key1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src_key2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
    let src_small = packed_fixture_structure(
        2,
        [
            (src_key1.clone(), vec![2, 1]),
            (src_key2.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let src_large =
        packed_fixture_structure(2, [(src_key1, vec![3, 1]), (src_key2, vec![3, 1])]).unwrap();
    let operation = TreeTransformOperation::transpose([1, 0], [0]);

    let z2_small =
        TreeTransformSectorPlanKey::tree_pair(&Z2FusionRule, operation.clone(), &src_small)
            .unwrap();
    let z2_large =
        TreeTransformSectorPlanKey::tree_pair(&Z2FusionRule, operation.clone(), &src_large)
            .unwrap();
    let fermion = TreeTransformSectorPlanKey::tree_pair(
        &FermionParityFusionRule,
        operation.clone(),
        &src_small,
    )
    .unwrap();
    let all_codomain =
        TreeTransformSectorPlanKey::all_codomain(&Z2FusionRule, operation, &src_small).unwrap();

    assert_eq!(z2_small, z2_large);
    assert_ne!(z2_small, fermion);
    assert_ne!(z2_small, all_codomain);
}

#[test]
fn tree_transform_structure_cache_key_tracks_concrete_layout() {
    let key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone(), vec![2, 3], vec![1, 2], 0).unwrap()],
    )
    .unwrap();
    let shape_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone(), vec![3, 2], vec![1, 3], 0).unwrap()],
    )
    .unwrap();
    let stride_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone(), vec![2, 3], vec![2, 1], 0).unwrap()],
    )
    .unwrap();
    let offset_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone(), vec![2, 3], vec![1, 2], 1).unwrap()],
    )
    .unwrap();
    let plan_key = TreeTransformSectorPlanKey::tree_pair(
        &Z2FusionRule,
        TreeTransformOperation::transpose([1, 0], [0]),
        &src,
    )
    .unwrap();
    let base =
        TreeTransformStructureCacheKey::from_structures(plan_key.clone(), &src, &src).unwrap();
    let conjugating = TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
        plan_key.clone(),
        &src,
        &src,
        true,
    )
    .unwrap();

    assert_ne!(base, conjugating);
    assert_ne!(
        base,
        TreeTransformStructureCacheKey::from_structures(plan_key.clone(), &shape_changed, &src)
            .unwrap()
    );
    assert_ne!(
        base,
        TreeTransformStructureCacheKey::from_structures(plan_key.clone(), &stride_changed, &src)
            .unwrap()
    );
    assert_ne!(
        base,
        TreeTransformStructureCacheKey::from_structures(plan_key, &offset_changed, &src).unwrap()
    );
}

#[test]
fn tree_transform_group_block_spec_rejects_group_structure_mismatch() {
    let src_key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let dense_structure = BlockStructure::trivial(&[1, 1]).unwrap();
    let src_groups = src_structure.fusion_tree_groups();

    let err = TreeTransformGroupBlockSpec::<f64>::from_block_groups(
        &dense_structure,
        &src_groups[0],
        &src_structure,
        &src_groups[0],
        vec![1.0],
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::FusionTreeGroupMismatch {
            tensor: "dst",
            index: 0,
        }
    );
}

#[test]
fn tree_transform_rejects_incompatible_single_tree_shapes() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 4], src_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let err =
        TreeTransformStructure::compile(&dst, &src, &[TreeTransformBlockSpec::single(0, 0, 1.0)])
            .unwrap_err();

    assert_eq!(
        err,
        OperationError::ShapeMismatch {
            dst: vec![4, 1],
            src: vec![2, 2],
        }
    );
}

#[test]
fn tree_transform_rejects_mismatched_multi_tree_element_count() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src_structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![3, 1], vec![3, 1]]).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 8], src_space, src_structure)
            .unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 6], dst_space, dst_structure)
            .unwrap();

    let err = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 0.0, 0.0, 1.0],
        )],
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::ElementCountMismatch {
            expected: 3,
            actual: 4,
        }
    );
}

#[derive(Debug, Default)]
struct RecordingKernelAdapter {
    add_calls: usize,
    axpby_calls: usize,
    copy_scale_calls: usize,
    scale_calls: usize,
    recoupling_calls: usize,
}

impl crate::HostKernelAdapter<f64> for RecordingKernelAdapter {
    fn add_strided(
        &mut self,
        zero_strides: &mut Vec<isize>,
        dst_data: &mut [f64],
        src_data: &[f64],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: f64,
        beta: f64,
    ) -> Result<(), OperationError> {
        self.add_calls += 1;
        crate::StridedHostKernelAdapter.add_strided(
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            beta,
        )
    }

    fn axpby_strided(
        &mut self,
        dst_data: &mut [f64],
        src_data: &[f64],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        alpha: f64,
        beta: f64,
    ) -> Result<(), OperationError> {
        self.axpby_calls += 1;
        crate::StridedHostKernelAdapter.axpby_strided(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
        )
    }

    fn copy_scale_strided(
        &mut self,
        dst_data: &mut [f64],
        src_data: &[f64],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: f64,
    ) -> Result<(), OperationError> {
        self.copy_scale_calls += 1;
        crate::StridedHostKernelAdapter.copy_scale_strided(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
        )
    }

    fn scale_strided(
        &mut self,
        dst_data: &mut [f64],
        shape: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
        beta: f64,
    ) -> Result<(), OperationError> {
        self.scale_calls += 1;
        crate::StridedHostKernelAdapter.scale_strided(
            dst_data,
            shape,
            dst_strides,
            dst_offset,
            beta,
        )
    }

    fn recoupling_src_times_u_transpose<C>(
        &mut self,
        destination: &mut [f64],
        source: &[f64],
        coefficients_src_by_dst: &[C],
        coefficient_start: usize,
        element_count: usize,
        src_count: usize,
        dst_count: usize,
    ) -> Result<(), OperationError>
    where
        C: Copy,
        f64: RecouplingCoefficientAction<C>,
    {
        self.recoupling_calls += 1;
        crate::StridedHostKernelAdapter.recoupling_src_times_u_transpose(
            destination,
            source,
            coefficients_src_by_dst,
            coefficient_start,
            element_count,
            src_count,
            dst_count,
        )
    }
}

#[test]
fn tree_transform_replay_dispatches_through_kernel_adapter() {
    let group_key = FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true]);
    let key10 = BlockKey::sector_ids([10]);
    let key20 = BlockKey::sector_ids([20]);
    let key100 = BlockKey::sector_ids([100]);
    let key200 = BlockKey::sector_ids([200]);
    let key300 = BlockKey::sector_ids([300]);
    let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
    let src_structure = packed_fixture_structure(
        2,
        [
            (key100.clone(), vec![2, 1]),
            (key300.clone(), vec![2, 1]),
            (key200.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_structure = packed_fixture_structure(
        2,
        [(key20.clone(), vec![2, 1]), (key10.clone(), vec![2, 1])],
    )
    .unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure)
            .unwrap();
    let structure = TreeTransformStructure::compile_grouped(
        &dst,
        &src,
        &[TreeTransformGroupBlockSpec::multi(
            group_key,
            [key10, key20],
            [key100, key200, key300],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        )],
    )
    .unwrap();
    let dst_block_structure = std::sync::Arc::clone(dst.structure());
    let src_block_structure = std::sync::Arc::clone(src.structure());
    let mut workspace = crate::TreeTransformWorkspace::<f64>::default();
    let mut adapter = RecordingKernelAdapter::default();

    tenet_operations::tree_transform_structure_with_strided_kernel_raw(
        &mut adapter,
        &mut workspace,
        &structure,
        &dst_block_structure,
        &src_block_structure,
        dst.data_mut(),
        src.data(),
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[7020.0, 9240.0, 3510.0, 4620.0]);
    assert_eq!(adapter.copy_scale_calls, 3);
    assert_eq!(adapter.recoupling_calls, 1);
    assert_eq!(adapter.axpby_calls, 2);
    assert_eq!(adapter.add_calls, 0);
    assert_eq!(adapter.scale_calls, 0);
}
