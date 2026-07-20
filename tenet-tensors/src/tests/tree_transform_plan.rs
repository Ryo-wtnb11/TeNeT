use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[allow(deprecated)]
fn legacy_tree_row_stats(stats: TreeTransformCacheStats) -> (usize, usize) {
    (stats.tree_row_hits(), stats.tree_row_misses())
}

fn grouped_su2_test_pair(first_inner: usize) -> FusionTreePairKey {
    all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [4, 4, 4, 4],
        0,
        [false, false, false, false],
        [first_inner, 4],
        [1, 1, 1],
    )
}

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
    let key1 = BlockKey::opaque([1]);
    let key2 = BlockKey::opaque([2]);
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

    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: Box::new(key2)
        }
    );
}

#[test]
fn tree_transform_group_block_spec_preserves_group_identity_and_ordered_keys() {
    let group_key = FusionTreeGroupKey::from_sector_ids([4, 4, 4, 4], [], [false; 4], []);
    let pair = grouped_su2_test_pair;
    let dst_key1 = pair(0);
    let dst_key2 = pair(2);
    let src_key = pair(4);
    let spec = TreeTransformGroupBlockSpec::try_multi(
        [dst_key1.clone(), dst_key2.clone()],
        [src_key.clone()],
        vec![2.0_f64, 3.0],
    )
    .unwrap();

    assert_eq!(spec.group_key(), &group_key);
    assert_eq!(
        spec.group_key()
            .codomain_uncoupled()
            .iter()
            .map(|sector| sector.id())
            .collect::<Vec<_>>(),
        vec![4, 4, 4, 4]
    );
    assert!(spec.group_key().domain_uncoupled().is_empty());
    assert_eq!(spec.group_key().codomain_is_dual(), &[false; 4]);
    assert!(spec.group_key().domain_is_dual().is_empty());
    fn assert_categorical_keys(_: &[FusionTreePairKey]) {}
    assert_categorical_keys(spec.dst_keys());
    assert_categorical_keys(spec.src_keys());
    assert_eq!(spec.dst_keys(), &[dst_key1, dst_key2]);
    assert_eq!(spec.src_keys(), &[src_key]);
    assert_eq!(spec.recoupling_coefficients_dst_src(), &[2.0, 3.0]);
}

#[test]
fn grouped_spec_accepts_distinct_source_and_destination_cohorts() {
    let src1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
    let dst1 = fusion_tree_test_key([20, 10], [31], 7, [true, false], [false]);
    let dst2 = fusion_tree_test_key([20, 10], [31], 8, [true, false], [false]);

    let spec = TreeTransformGroupBlockSpec::try_multi(
        [dst1.clone(), dst2.clone()],
        [src1.clone(), src2.clone()],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .unwrap();

    // What: the stored identity is derived from the authoritative source
    // cohort, while a transform may target a different coherent cohort.
    assert_eq!(spec.group_key(), &src1.group_key());
    assert_ne!(spec.group_key(), &dst1.group_key());
    assert_eq!(spec.dst_keys(), &[dst1, dst2]);
    assert_eq!(spec.src_keys(), &[src1, src2]);
}

#[test]
fn grouped_spec_rejects_mixed_source_and_destination_cohorts() {
    let src = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let other_src = fusion_tree_test_key([10, 21], [30], 6, [false, true], [true]);
    let dst = fusion_tree_test_key([20, 10], [31], 7, [true, false], [false]);
    let other_dst = fusion_tree_test_key([20, 11], [31], 8, [true, false], [false]);

    let src_err = TreeTransformGroupBlockSpec::try_multi(
        [dst.clone()],
        [src.clone(), other_src],
        vec![1.0_f64, 2.0],
    )
    .unwrap_err();
    let dst_err =
        TreeTransformGroupBlockSpec::try_multi([dst, other_dst], [src], vec![1.0_f64, 2.0])
            .unwrap_err();

    // What: one grouped matrix cannot silently mix either categorical cohort.
    assert_eq!(
        src_err,
        OperationError::FusionTreeGroupMismatch {
            tensor: "src",
            index: 1,
        }
    );
    assert_eq!(
        dst_err,
        OperationError::FusionTreeGroupMismatch {
            tensor: "dst",
            index: 1,
        }
    );
}

#[test]
fn grouped_spec_rejects_duplicate_source_columns_and_destination_rows() {
    let src = grouped_su2_test_pair(0);
    let dst = grouped_su2_test_pair(2);

    let src_err = TreeTransformGroupBlockSpec::try_multi(
        [dst.clone()],
        [src.clone(), src.clone()],
        vec![1.0_f64, 2.0],
    )
    .unwrap_err();
    let dst_err =
        TreeTransformGroupBlockSpec::try_multi([dst.clone(), dst], [src], vec![1.0_f64, 2.0])
            .unwrap_err();

    // What: each U[dst, src] row and column has one categorical basis identity,
    // while the same identity may still occur once on each opposite side.
    assert_eq!(
        src_err,
        OperationError::DuplicateTreeTransformKey {
            tensor: "src",
            index: 1,
        }
    );
    assert_eq!(
        dst_err,
        OperationError::DuplicateTreeTransformKey {
            tensor: "dst",
            index: 1,
        }
    );
}

#[test]
fn grouped_spec_rejects_empty_sets_and_wrong_coefficient_count() {
    let src = grouped_su2_test_pair(0);
    let dst = grouped_su2_test_pair(2);

    let empty_dst = TreeTransformGroupBlockSpec::try_multi(
        Vec::<FusionTreePairKey>::new(),
        [src.clone()],
        Vec::<f64>::new(),
    )
    .unwrap_err();
    let empty_src = TreeTransformGroupBlockSpec::try_multi(
        [dst.clone()],
        Vec::<FusionTreePairKey>::new(),
        Vec::<f64>::new(),
    )
    .unwrap_err();
    let coefficient_err =
        TreeTransformGroupBlockSpec::try_multi([dst], [src], Vec::<f64>::new()).unwrap_err();

    // What: grouped matrices have at least one row and column and exactly one
    // row-major coefficient per destination/source pair.
    assert_eq!(empty_dst, OperationError::EmptyTransformBlock);
    assert_eq!(empty_src, OperationError::EmptyTransformBlock);
    assert_eq!(
        coefficient_err,
        OperationError::CoefficientCountMismatch {
            expected: 1,
            actual: 0,
        }
    );
}

#[test]
fn unique_tree_transform_plan_builder_creates_single_specs_in_source_order() {
    let key = |codomain| {
        FusionTreePairKey::try_pair_from_sector_ids(
            codomain,
            [1],
            1,
            [false, false],
            [false],
            [],
            [],
            [1],
            [],
        )
        .unwrap()
    };
    let src_key1 = key([1, 0]);
    let src_key2 = key([0, 1]);
    let dst_key1 = key([0, 1]);
    let dst_key2 = key([1, 0]);
    let src_tree1 = expect_tree_key(&src_key1);
    let src_tree2 = expect_tree_key(&src_key2);
    let dst_tree1 = expect_tree_key(&dst_key1);
    let dst_tree2 = expect_tree_key(&dst_key2);
    let src_structure = packed_fixture_structure(
        3,
        [
            (src_key1.clone(), vec![1, 1, 1]),
            (src_key2.clone(), vec![1, 1, 1]),
        ],
    )
    .unwrap();

    let plan = build_unique_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], [2]),
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
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[2.0]);
    assert_eq!(plan.specs()[1].group_key(), &src_tree2.group_key());
    assert_eq!(plan.specs()[1].src_keys(), &[src_key2]);
    assert_eq!(plan.specs()[1].dst_keys(), &[dst_key2]);
    assert_eq!(plan.specs()[1].recoupling_coefficients_dst_src(), &[3.0]);
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
        |_| -> Result<(FusionTreePairKey, f64), OperationError> {
            unreachable!("non-Unique fusion must be rejected before transforming keys")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
            style: FusionStyleKind::Simple,
        }
    );
}

#[test]
fn tree_transform_plan_builder_accepts_simple_multi_destination_callback() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
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
        spec.recoupling_coefficients_dst_src(),
        &[0.5, 0.866_025_403_784_438_6, 0.866_025_403_784_438_6, -0.5]
    );
}

#[test]
fn tree_transform_plan_builder_lowers_injective_singleton_rows_in_source_order() {
    // What: a nonidentity monomial group is represented by ordered direct
    // specs even when its coefficients are neither inferred nor unit-valued.
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
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

    let plan = build_tree_transform_group_plan(
        &SimpleSu2Rule,
        TreeTransformOperation::braid([1, 0, 2, 3], [], [0, 1, 2, 3], []),
        &src_structure,
        |src| {
            if src == &src_tree0 {
                Ok(vec![(src_tree1.clone(), -2.0_f64)])
            } else if src == &src_tree1 {
                Ok(vec![(src_tree0.clone(), 3.0_f64)])
            } else {
                panic!("unexpected source key {src:?}")
            }
        },
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 2);
    assert_eq!(plan.specs()[0].src_keys(), std::slice::from_ref(&src_key0));
    assert_eq!(plan.specs()[0].dst_keys(), std::slice::from_ref(&src_key1));
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[-2.0]);
    assert_eq!(plan.specs()[1].src_keys(), std::slice::from_ref(&src_key1));
    assert_eq!(plan.specs()[1].dst_keys(), std::slice::from_ref(&src_key0));
    assert_eq!(plan.specs()[1].recoupling_coefficients_dst_src(), &[3.0]);
    assert!(plan
        .specs()
        .iter()
        .all(|spec| spec.source_axes() == Some([1, 0, 2, 3].as_slice())));

    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![5.0, 7.0],
        space.clone(),
        src_structure.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![11.0, 13.0],
        space,
        src_structure.clone(),
    )
    .unwrap();
    let compiled = plan
        .compile_structures(&src_structure, &src_structure)
        .unwrap();
    assert!(!compiled.has_pack_gemm_scatter_blocks());
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();

    // dst0 receives 3*src1 and dst1 receives -2*src0.
    assert_eq!(dst.data(), &[75.0, 19.0]);
    assert_eq!(
        (workspace.source_len(), workspace.destination_len()),
        (0, 0)
    );
}

#[test]
fn tree_transform_plan_builder_keeps_destination_collisions_in_multi() {
    // What: singleton rows are not direct when two sources contribute to the
    // same destination, because replay must preserve their sum.
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let dst_tree = expect_tree_key(&src_key0);
    let src_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();

    let plan = build_tree_transform_group_plan(
        &SimpleSu2Rule,
        TreeTransformOperation::braid([1, 0, 2, 3], [], [0, 1, 2, 3], []),
        &src_structure,
        |_| Ok(vec![(dst_tree.clone(), 1.0_f64)]),
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key0, src_key1]);
    assert_eq!(plan.specs()[0].dst_keys().len(), 1);
}

#[test]
fn su2_first_pair_braid_lowers_nonidentity_monomial_group_to_singles() {
    // What: the first-vertex SU(2) R move is direct for every fusion channel,
    // while preserving the channel-dependent coefficients returned by core.
    let keys = [[0, 1], [2, 1]].map(|inner| {
        all_codomain_fusion_tree_test_key_for_rule(
            &SU2FusionRule,
            [1, 1, 1, 1],
            0,
            [false, false, false, false],
            inner,
            [1, 1, 1],
        )
    });
    let structure =
        packed_fixture_structure(4, keys.iter().cloned().map(|key| (key, vec![1usize; 4])))
            .unwrap();
    let operation = TreeTransformOperation::braid([1, 0, 2, 3], [], [0, 1, 2, 3], []);

    let plan =
        build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation.clone(), &structure)
            .unwrap();

    assert_eq!(plan.specs().len(), keys.len());
    for ((spec, key), expected_coefficient) in plan.specs().iter().zip(&keys).zip([-1.0, 1.0]) {
        assert_eq!(spec.src_keys(), std::slice::from_ref(key));
        assert_eq!(spec.dst_keys(), std::slice::from_ref(key));
        assert_eq!(
            spec.recoupling_coefficients_dst_src(),
            &[expected_coefficient]
        );
        assert_eq!(spec.source_axes(), Some([1, 0, 2, 3].as_slice()));
    }
    assert!(!plan
        .compile_structures(&structure, &structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());

    use crate::tree_transform::{
        build_all_codomain_tree_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_all_codomain_preflight,
    };
    let build = |threads: usize| {
        let proof = validate_multiplicity_free_all_codomain_preflight(
            &SU2FusionRule,
            &operation,
            &structure,
        )
        .unwrap();
        build_all_codomain_tree_transform_group_plan_validated_with_threads(
            &proof,
            operation.clone(),
            threads,
        )
        .unwrap()
    };
    let serial = build(1);
    let parallel = build(4);
    assert_eq!(parallel, serial);

    // What: direct replay honors alpha/beta and overwrite on strided blocks
    // without touching storage padding or allocating pack/GEMM/scatter jobs.
    let padded_structure = BlockStructure::from_blocks_with_rank(
        4,
        vec![
            BlockSpec::with_key(
                keys[0].clone().into(),
                vec![2, 2, 1, 1],
                vec![1, 2, 4, 4],
                1,
            )
            .unwrap(),
            BlockSpec::with_key(
                keys[1].clone().into(),
                vec![2, 2, 1, 1],
                vec![1, 2, 4, 4],
                7,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([2, 2, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![99.0, 1.0, 2.0, 3.0, 4.0, 98.0, 97.0, 9.0, 10.0, 11.0, 12.0],
        space.clone(),
        padded_structure.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![91.0, 5.0, 6.0, 7.0, 8.0, 92.0, 93.0, 13.0, 14.0, 15.0, 16.0],
        space,
        padded_structure.clone(),
    )
    .unwrap();
    let compiled = serial
        .compile_structures(&padded_structure, &padded_structure)
        .unwrap();
    assert!(!compiled.has_pack_gemm_scatter_blocks());
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();
    assert_eq!(
        dst.data(),
        &[91.0, 13.0, 12.0, 17.0, 16.0, 92.0, 93.0, 57.0, 64.0, 65.0, 72.0]
    );
    assert_eq!(
        (workspace.source_len(), workspace.destination_len()),
        (0, 0)
    );

    dst.data_mut().fill(f64::NAN);
    backend.set_recoupling_threads(4);
    backend.set_transform_parallel_min_len(0);
    tree_transform_overwrite_execute_with(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        -2.0,
    )
    .unwrap();
    assert!(dst.data()[0].is_nan());
    assert_eq!(&dst.data()[1..5], &[2.0, 6.0, 4.0, 8.0]);
    assert!(dst.data()[5].is_nan() && dst.data()[6].is_nan());
    assert_eq!(&dst.data()[7..11], &[-18.0, -22.0, -20.0, -24.0]);
}

#[test]
fn nested_fz2_u1_su2_first_pair_braid_preserves_product_phases_in_singles() {
    // What: direct lowering preserves the product rule's fermionic sign times
    // the SU(2) channel phase instead of synthesizing a diagonal coefficient.
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let left_sector =
        |parity, charge| left_rule.encode_sector(parity, U1Irrep::new(charge).sector_id());
    let sector = |parity, charge, twice_spin| {
        rule.encode_sector(
            left_sector(parity, charge),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };
    let odd_half = sector(SectorId::new(1), 0, 1);
    let even_singlet = sector(SectorId::new(0), 0, 0);
    let even_triplet = sector(SectorId::new(0), 0, 2);
    let odd_half_inner = sector(SectorId::new(1), 0, 1);
    let keys = [even_singlet, even_triplet].map(|first_inner| {
        FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [odd_half; 4],
                even_singlet,
                [false; 4],
                [first_inner, odd_half_inner],
                [MultiplicityIndex::ONE; 3],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&rule, [], even_singlet, [], [], []).unwrap(),
        )
    });
    let structure =
        packed_fixture_structure(4, keys.iter().cloned().map(|key| (key, vec![1usize; 4])))
            .unwrap();

    let plan = build_all_codomain_tree_transform_group_plan(
        &rule,
        TreeTransformOperation::braid([1, 0, 2, 3], [], [0, 1, 2, 3], []),
        &structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 2);
    for ((spec, key), expected_coefficient) in plan.specs().iter().zip(&keys).zip([1.0, -1.0]) {
        assert_eq!(spec.src_keys(), std::slice::from_ref(key));
        assert_eq!(spec.dst_keys(), std::slice::from_ref(key));
        assert_eq!(
            spec.recoupling_coefficients_dst_src(),
            &[expected_coefficient]
        );
    }
    assert!(!plan
        .compile_structures(&structure, &structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());
}

#[test]
fn multiplicity_free_su2_plan_builder_creates_generic_recoupling_block() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
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
    assert_eq!(spec.recoupling_coefficients_dst_src().len(), expected.len());
    for (&actual, expected) in spec.recoupling_coefficients_dst_src().iter().zip(expected) {
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
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
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
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
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
fn tree_transform_recoupling_plan_groups_same_shape_multi_blocks() {
    let dst_structure = BlockStructure::packed_column_major(
        1,
        [vec![2], vec![2], vec![1], vec![1], vec![1], vec![1]],
    )
    .unwrap();
    let src_structure = dst_structure.clone();
    let structure = TreeTransformStructure::compile_structures(
        &dst_structure,
        &src_structure,
        &[
            TreeTransformBlockSpec::multi(vec![2, 3], vec![2, 3], vec![1.0, 0.0, 0.0, 1.0]),
            TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]),
            TreeTransformBlockSpec::multi(vec![4, 5], vec![4, 5], vec![1.0, 0.0, 0.0, 1.0]),
        ],
    )
    .unwrap();

    assert_eq!(structure.replay_weights(), vec![8, 4, 4]);
    let plan = structure.recoupling_plan();
    assert_eq!(plan.block_indices(), &[1, 2, 0]);
    let jobs = plan.jobs();
    assert_eq!(jobs.len(), 3);
    assert_eq!((jobs[0].rows, jobs[0].contracted, jobs[0].cols), (1, 2, 2));
    assert_eq!((jobs[1].rows, jobs[1].contracted, jobs[1].cols), (1, 2, 2));
    assert_eq!((jobs[2].rows, jobs[2].contracted, jobs[2].cols), (2, 2, 2));
    assert_eq!(jobs[1].lhs_offset - jobs[0].lhs_offset, 2);
    assert_eq!(jobs[1].rhs_offset - jobs[0].rhs_offset, 4);
    assert_eq!(jobs[1].dst_offset - jobs[0].dst_offset, 2);
}

#[test]
fn tree_transform_structure_replays_su2_recoupling_without_recompiling() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
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
fn parallel_plan_compile_matches_serial_plan() {
    // What: shared group compilation produces the same transformer,
    // source order, and group order.
    use crate::tree_transform::{
        build_tree_pair_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_tree_pair_preflight,
    };

    let key = |uncoupled: [usize; 4], inner: [usize; 2]| {
        all_codomain_fusion_tree_test_key_for_rule(
            &SU2FusionRule,
            uncoupled,
            0,
            [false, false, false, false],
            inner,
            [1, 1, 1],
        )
    };
    let keys = [
        key([1, 1, 1, 1], [0, 1]),
        key([1, 1, 1, 1], [2, 1]),
        key([1, 3, 3, 1], [2, 1]),
        key([1, 3, 3, 1], [4, 1]),
    ];
    let src_structure = packed_fixture_structure(
        4,
        keys.iter().map(|key| (key.clone(), vec![1usize, 1, 1, 1])),
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let build = |threads: usize| {
        let proof = validate_multiplicity_free_tree_pair_preflight(
            &SU2FusionRule,
            &operation,
            &src_structure,
        )
        .unwrap();
        build_tree_pair_transform_group_plan_validated_with_threads(
            &proof,
            operation.clone(),
            threads,
        )
        .unwrap()
    };

    let serial_plan = build(1);
    let parallel_plan = build(8);
    assert_eq!(parallel_plan, serial_plan);
}

#[test]
fn all_codomain_canonical_empty_domain_row_is_thread_count_invariant() {
    use crate::tree_transform::{
        build_all_codomain_tree_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_all_codomain_preflight,
    };

    let codomain = FusionTreeKey::try_new_for_rule(
        &SU2FusionRule,
        [SectorId::new(1), SectorId::new(1)],
        SectorId::new(0),
        [false, false],
        [],
        [MultiplicityIndex::ONE],
    )
    .unwrap();
    let keys = [FusionTreePairKey::pair(codomain, empty_fusion_tree())];
    let structure =
        packed_fixture_structure(2, keys.iter().cloned().map(|key| (key, vec![1, 1]))).unwrap();
    for operation in [
        TreeTransformOperation::braid([0, 1], [], [7, 3], []),
        TreeTransformOperation::braid([1, 0], [], [0, 1], []),
    ] {
        let build = |threads| {
            let proof = validate_multiplicity_free_all_codomain_preflight(
                &SU2FusionRule,
                &operation,
                &structure,
            )
            .unwrap();
            build_all_codomain_tree_transform_group_plan_validated_with_threads(
                &proof,
                operation.clone(),
                threads,
            )
            .unwrap()
        };

        let mut expected_plan = None;
        for threads in [1, 2, 4] {
            let cold = build(threads);
            // What: the mandatory vacuum removes the prior empty-domain alias,
            // and its canonical transform is worker-count-independent.
            if let Some(expected) = &expected_plan {
                assert_eq!(&cold, expected);
            } else {
                expected_plan = Some(cold.clone());
            }
        }
    }
}

#[test]
fn all_codomain_invalid_operation_is_thread_count_invariant() {
    let keys = [1, 2].map(|sector| {
        FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [SectorId::new(sector), SectorId::new(sector)],
                SectorId::new(0),
                [false, false],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            empty_fusion_tree(),
        )
    });
    let structure =
        packed_fixture_structure(2, keys.into_iter().map(|key| (key, vec![1, 1]))).unwrap();
    let invalid_operation = TreeTransformOperation::braid([0, 0], [], [0, 1], []);
    for threads in [1, 2, 4] {
        let result = build_all_codomain_tree_transform_group_plan(
            &SU2FusionRule,
            invalid_operation.clone(),
            &structure,
        );

        // What: invalid syntax is rejected before worker count can affect
        // compilation or publish any cache entry.
        assert!(result.is_err());
        let _ = threads;
    }
}

#[test]
fn staged_group_scheduler_builds_bounded_balanced_contiguous_batches() {
    use crate::tree_transform::partition_staged_groups_for_test;

    for (group_count, threads, expected_sizes) in [
        (3, 2, vec![2, 1]),
        (5, 2, vec![3, 2]),
        (3, 8, vec![1, 1, 1]),
    ] {
        let batches = partition_staged_groups_for_test((0..group_count).collect(), threads);
        // What: awkward group/thread ratios still create exactly the bounded
        // task count rather than relying on Rayon's recursive split heuristic.
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            expected_sizes
        );
        assert_eq!(
            batches.into_iter().flatten().collect::<Vec<_>>(),
            (0..group_count).collect::<Vec<_>>()
        );
    }
}

#[test]
fn split_only_repartition_plan_matches_serial_and_parallel_builders() {
    // What: a SU(2) 2|2 -> 3|1 split-only braid compiles to the direct
    // repartition row independently of the compile worker count.
    use crate::tree_transform::{
        build_tree_pair_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_tree_pair_preflight,
    };

    let source = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 2],
        [2, 1],
        1,
        [false, true],
        [true, false],
        [],
        [],
        [1],
        [1],
    )
    .unwrap();
    let source_key = BlockKey::from(source.clone());
    let source_structure = packed_fixture_structure(4, [(source_key, vec![1usize; 4])]).unwrap();
    let operation = TreeTransformOperation::braid([0, 1, 3], [2], [0, 1], [2, 3]);
    let expected = multiplicity_free_repartition_tree_pair(&SU2FusionRule, &source, 3).unwrap();
    let build = |threads| {
        let proof = validate_multiplicity_free_tree_pair_preflight(
            &SU2FusionRule,
            &operation,
            &source_structure,
        )
        .unwrap();
        build_tree_pair_transform_group_plan_validated_with_threads(
            &proof,
            operation.clone(),
            threads,
        )
        .unwrap()
    };

    let serial = build(1);
    let parallel = build(8);
    assert_eq!(parallel, serial);

    let spec = &serial.specs()[0];
    assert_eq!(spec.dst_keys(), &[expected[0].0.clone()]);
    assert!((spec.recoupling_coefficients_dst_src()[0] - expected[0].1).abs() < 1.0e-12);
}

#[test]
fn identity_group_plan_lowers_each_su2_tree_to_a_direct_single() {
    // What: an identity operation over a multi-tree SU2 fusion group compiles
    // to independent direct copies for every supported facade and worker count.
    use crate::tree_transform::{
        build_tree_pair_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_tree_pair_preflight,
    };

    let key = |inner: [usize; 2]| {
        all_codomain_fusion_tree_test_key_for_rule(
            &SU2FusionRule,
            [3, 3, 3, 3],
            0,
            [false, false, false, false],
            inner,
            [1, 1, 1],
        )
    };
    let keys = [key([0, 3]), key([2, 3]), key([4, 3]), key([6, 3])];
    let structure =
        packed_fixture_structure(4, keys.iter().map(|key| (key.clone(), vec![1usize; 4]))).unwrap();
    let operation = TreeTransformOperation::braid([0, 1, 2, 3], [], [17, 3, 11, 5], []);
    let build = |threads| {
        let proof =
            validate_multiplicity_free_tree_pair_preflight(&SU2FusionRule, &operation, &structure)
                .unwrap();
        build_tree_pair_transform_group_plan_validated_with_threads(
            &proof,
            operation.clone(),
            threads,
        )
        .unwrap()
    };

    let serial = build(1);
    let parallel = build(8);
    assert_eq!(parallel, serial);
    assert_eq!(serial.specs().len(), keys.len());
    for spec in serial.specs() {
        assert_eq!(spec.src_keys().len(), 1);
        assert_eq!(spec.dst_keys(), spec.src_keys());
        assert_eq!(spec.recoupling_coefficients_dst_src(), &[1.0]);
        assert_eq!(spec.source_axes(), Some([0, 1, 2, 3].as_slice()));
    }
    assert!(!serial
        .compile_structures(&structure, &structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());

    let transpose = build_tree_pair_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::transpose([0, 1, 2, 3], []),
        &structure,
    )
    .unwrap();
    assert!(transpose.specs().iter().all(|spec| {
        spec.src_keys().len() == 1
            && spec.dst_keys() == spec.src_keys()
            && spec.recoupling_coefficients_dst_src() == [1.0]
    }));
    assert!(!transpose
        .compile_structures(&structure, &structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());

    let all_codomain =
        build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation, &structure)
            .unwrap();
    assert_eq!(all_codomain, serial);

    let tensor_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        tensor_space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![5.0, 6.0, 7.0, 8.0],
        tensor_space,
        structure,
    )
    .unwrap();
    let compiled = serial.compile(&dst, &src).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();
    assert_eq!(dst.data(), &[17.0, 22.0, 27.0, 32.0]);

    dst.data_mut().fill(f64::NAN);
    backend.set_recoupling_threads(4);
    backend.set_transform_parallel_min_len(0);
    tree_transform_overwrite_execute_with(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        -2.0,
    )
    .unwrap();
    assert_eq!(dst.data(), &[-2.0, -4.0, -6.0, -8.0]);
}

#[test]
fn same_split_transpose_is_direct_for_real_tree_pairs_but_split_change_is_not() {
    // What: exact 2|1 fZ2 and SU2 transposes preserve the source tree with a
    // unit coefficient, while a cyclic split change retains recoupling.
    let fz2_source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &FermionParityFusionRule,
            [SectorId::new(1), SectorId::new(0)],
            SectorId::new(1),
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &FermionParityFusionRule,
            [SectorId::new(1)],
            SectorId::new(1),
            [true],
            [],
            [],
        )
        .unwrap(),
    );
    let fz2_key = fz2_source;
    let fz2_structure = packed_fixture_structure(3, [(fz2_key.clone(), vec![1, 1, 1])]).unwrap();
    let exact = TreeTransformOperation::transpose([0, 1], [2]);
    let fz2_plan = build_tree_pair_transform_group_plan(
        &FermionParityFusionRule,
        exact.clone(),
        &fz2_structure,
    )
    .unwrap();
    assert_eq!(fz2_plan.specs().len(), 1);
    assert_eq!(
        fz2_plan.specs()[0].src_keys(),
        std::slice::from_ref(&fz2_key)
    );
    assert_eq!(fz2_plan.specs()[0].dst_keys(), &[fz2_key]);
    assert_eq!(
        fz2_plan.specs()[0].recoupling_coefficients_dst_src(),
        &[1.0]
    );
    assert!(!fz2_plan
        .compile_structures(&fz2_structure, &fz2_structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());

    let one = SU2Irrep::from_twice_spin(2).sector_id();
    let su2_source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [one, one],
            one,
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&SU2FusionRule, [one], one, [true], [], []).unwrap(),
    );
    let su2_rows = tenet_core::multiplicity_free_transpose_tree_pair(
        &SU2FusionRule,
        &su2_source,
        &[0, 1],
        &[2],
    )
    .unwrap();
    assert_eq!(su2_rows, vec![(su2_source.clone(), 1.0)]);

    let su2_key = su2_source;
    let su2_structure = packed_fixture_structure(3, [(su2_key.clone(), vec![1, 1, 1])]).unwrap();
    let su2_plan =
        build_tree_pair_transform_group_plan(&SU2FusionRule, exact, &su2_structure).unwrap();
    assert_eq!(
        su2_plan.specs()[0].src_keys(),
        std::slice::from_ref(&su2_key)
    );
    assert_eq!(su2_plan.specs()[0].dst_keys(), &[su2_key]);
    assert_eq!(
        su2_plan.specs()[0].recoupling_coefficients_dst_src(),
        &[1.0]
    );
    assert!(!su2_plan
        .compile_structures(&su2_structure, &su2_structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());

    let control_keys = [0, 2, 4].map(|inner| {
        BlockKey::from(FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [one, one, one],
                one,
                [false, false, false],
                [SectorId::new(inner)],
                [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&SU2FusionRule, [one], one, [true], [], []).unwrap(),
        ))
    });
    let control_src = packed_fixture_structure(
        4,
        control_keys
            .iter()
            .cloned()
            .map(|key| (key, vec![1, 1, 1, 1])),
    )
    .unwrap();
    let split_change = TreeTransformOperation::transpose([3, 0, 1], [2]);
    assert!(!split_change.is_identity_for(3, 1));
    let control =
        build_tree_pair_transform_group_plan(&SU2FusionRule, split_change, &control_src).unwrap();
    assert_eq!(control.specs()[0].src_keys().len(), control_keys.len());
    let dst_structure = packed_fixture_structure(
        4,
        control
            .specs()
            .iter()
            .flat_map(|spec| spec.dst_keys().iter().cloned())
            .map(|key| (key, vec![1, 1, 1, 1])),
    )
    .unwrap();
    assert!(control
        .compile_structures(&dst_structure, &control_src)
        .unwrap()
        .has_pack_gemm_scatter_blocks());
}

#[test]
fn tree_pair_plan_miss_does_not_retain_source_rows() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [2, 2, 2, 2],
        0,
        [false, false, false, false],
        [0, 2],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [2, 2, 2, 2],
        0,
        [false, false, false, false],
        [2, 2],
        [1, 1, 1],
    );
    let src_key2 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [2, 2, 2, 2],
        0,
        [false, false, false, false],
        [4, 2],
        [1, 1, 1],
    );
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [11, 13, 17, 19], []);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    let make = |src_keys: &[FusionTreePairKey], dst_keys: &[FusionTreePairKey]| {
        let src_blocks: Vec<_> = src_keys
            .iter()
            .map(|key| (key.clone(), vec![1usize, 1, 1, 1]))
            .collect();
        let dst_blocks: Vec<_> = dst_keys
            .iter()
            .map(|key| (key.clone(), vec![1usize, 1, 1, 1]))
            .collect();
        let src_structure = packed_fixture_structure(4, src_blocks).unwrap();
        let dst_structure = packed_fixture_structure(4, dst_blocks).unwrap();
        let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
        let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
        let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
            vec![1.0; src_keys.len()],
            src_space,
            src_structure,
        )
        .unwrap();
        let dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
            vec![0.0; dst_keys.len()],
            dst_space,
            dst_structure,
        )
        .unwrap();
        (dst, src)
    };

    let destination_keys = [src_key0.clone(), src_key1.clone(), src_key2.clone()];
    let (dst1, src1) = make(&destination_keys[..2], &destination_keys);
    cache
        .get_or_compile_tree_pair(&SU2FusionRule, operation.clone(), &dst1, &src1)
        .unwrap();
    assert_eq!(legacy_tree_row_stats(cache.stats()), (0, 0));

    // What: a different source tree set recompiles its whole block without
    // reporting cross-plan row hits from hidden retained state.
    let (dst2, src2) = make(&destination_keys, &destination_keys);
    cache
        .get_or_compile_tree_pair(&SU2FusionRule, operation.clone(), &dst2, &src2)
        .unwrap();
    assert_eq!(cache.stats().plan_misses(), 2);
    assert_eq!(legacy_tree_row_stats(cache.stats()), (0, 0));
}

fn malformed_simple_su2_tree_pair_tensors() -> (TensorMap<f64, 2, 0>, TensorMap<f64, 2, 0>) {
    let valid = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1],
        0,
        [false, false],
        [],
        [1],
    );
    let malformed = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            1,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let dst_structure = packed_fixture_structure(2, [(valid, vec![1, 1])]).unwrap();
    let src_structure = packed_fixture_structure(2, [(malformed, vec![1, 1])]).unwrap();
    let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0], space.clone(), dst_structure)
            .unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0], space, src_structure).unwrap();
    (dst, src)
}

fn simple_su2_vertex_structure(vertex: usize) -> Arc<BlockStructure> {
    let vertices = (vertex != 0).then_some(vertex);
    let key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            vertices,
            [],
        )
        .unwrap(),
    );
    Arc::new(packed_fixture_structure(2, [(key, vec![1, 1])]).unwrap())
}

#[derive(Clone)]
struct AdmissionCountingSu2Rule {
    nsymbol_calls: Arc<AtomicUsize>,
}

impl FusionRule for AdmissionCountingSu2Rule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        SU2FusionRule.rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        SU2FusionRule.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        SU2FusionRule.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        SU2FusionRule.vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        SU2FusionRule.dual(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        SU2FusionRule.fusion_channels(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.nsymbol_calls.fetch_add(1, Ordering::Relaxed);
        SU2FusionRule.nsymbol(left, right, coupled)
    }
}

impl MultiplicityFreeFusionRule for AdmissionCountingSu2Rule {}

impl MultiplicityFreeFusionSymbols for AdmissionCountingSu2Rule {
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        SU2FusionRule.scalar_one()
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        SU2FusionRule.scalar_conj(value)
    }

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        SU2FusionRule.f_symbol_scalar(left, middle, right, coupled, left_coupled, right_coupled)
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        SU2FusionRule.r_symbol_scalar(left, right, coupled)
    }
}

impl MultiplicityFreeRigidSymbols for AdmissionCountingSu2Rule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.dim_scalar(sector)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.inv_dim_scalar(sector)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.sqrt_dim_scalar(sector)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.inv_sqrt_dim_scalar(sector)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.twist_scalar(sector)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.frobenius_schur_phase_scalar(sector)
    }
}

impl TreeTransformRuleCacheKey for AdmissionCountingSu2Rule {
    type Key = TreeTransformBuiltinRuleCacheKey;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        SU2FusionRule.tree_transform_rule_cache_key()
    }
}

fn builtin_tree_cache_state(
    cache: &TreeTransformCache<f64, TreeTransformBuiltinRuleCacheKey>,
) -> (TreeTransformCacheStats, usize, usize, usize) {
    (
        cache.stats(),
        cache.plan_len(),
        cache.structure_len(),
        cache.fast_structure_len(),
    )
}

fn assert_invalid_simple_vertex(error: OperationError) {
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of vertices",
        })
    );
}

fn valid_simple_source_and_nonalias_malformed_destination(
) -> (TensorMap<f64, 2, 0>, TensorMap<f64, 2, 0>) {
    let valid = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1],
        0,
        [false, false],
        [],
        [1],
    );
    let malformed = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            [],
            [],
        )
        .unwrap(),
    );
    let src_structure = packed_fixture_structure(2, [(valid, vec![1, 1])]).unwrap();
    let dst_structure = packed_fixture_structure(2, [(malformed, vec![2, 1])]).unwrap();
    assert_ne!(src_structure.content_id(), dst_structure.content_id());
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![3.0],
        TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap(),
        src_structure,
    )
    .unwrap();
    let dst = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![17.0, 19.0],
        TensorMapSpace::<2, 0>::from_dims([2, 1], []).unwrap(),
        dst_structure,
    )
    .unwrap();
    (dst, src)
}

#[test]
fn callback_builder_admits_whole_source_before_first_callback() {
    let valid = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let malformed_later = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            2,
            [false, false],
            [],
            [],
            [],
            [],
            [],
        )
        .unwrap(),
    );
    let structure =
        packed_fixture_structure(2, [(valid, vec![1, 1]), (malformed_later, vec![1, 1])]).unwrap();
    let callbacks = std::cell::Cell::new(0usize);

    let error = build_tree_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::permute([0, 1], []),
        &structure,
        |source| {
            callbacks.set(callbacks.get() + 1);
            Ok(vec![(source.clone(), 1.0)])
        },
    )
    .unwrap_err();

    // What: a malformed later source block prevents source-major callbacks
    // from observing the valid prefix.
    assert_invalid_simple_vertex(error);
    assert_eq!(callbacks.get(), 0);
}

#[test]
fn callback_builder_rejects_malformed_destination_before_assembly() {
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(0),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&SU2FusionRule, [], SectorId::new(0), [], [], []).unwrap(),
    );
    let malformed_destination = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 1],
        [],
        0,
        [false, false],
        [],
        [],
        [],
        [],
        [],
    )
    .unwrap();
    assert_ne!(source, malformed_destination);
    let structure = packed_fixture_structure(2, [(BlockKey::from(source), vec![1, 1])]).unwrap();

    let error = build_tree_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::permute([0, 1], []),
        &structure,
        |_| Ok(vec![(malformed_destination.clone(), 1.0)]),
    )
    .unwrap_err();

    // What: a malformed callback destination is rejected before group assembly
    // even though its external sectors match the admitted source.
    assert_invalid_simple_vertex(error);
}

#[test]
fn warm_structure_aliases_are_rejected_before_local_cache_lookup() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_global_operation_caches();

    let valid_structure = simple_su2_vertex_structure(1);
    let invalid_structure = simple_su2_vertex_structure(0);
    assert_ne!(valid_structure.content_id(), invalid_structure.content_id());
    assert_ne!(
        valid_structure.block(0).unwrap().key(),
        invalid_structure.block(0).unwrap().key()
    );
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &valid_structure,
            &valid_structure,
            false,
        )
        .unwrap();

    let local_before = builtin_tree_cache_state(&cache);
    let error = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &invalid_structure,
            &valid_structure,
            false,
        )
        .unwrap_err();
    assert_invalid_simple_vertex(error);

    // What: a malformed destination is rejected before local cache observation
    // changes, even after a valid sibling structure is warm.
    assert_eq!(builtin_tree_cache_state(&cache), local_before);

    let error = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &valid_structure,
            &invalid_structure,
            false,
        )
        .unwrap_err();
    assert_invalid_simple_vertex(error);
    assert_eq!(builtin_tree_cache_state(&cache), local_before);

    let mut independent = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    let independent_before = builtin_tree_cache_state(&independent);
    let error = independent
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &invalid_structure,
            &valid_structure,
            false,
        )
        .unwrap_err();
    assert_invalid_simple_vertex(error);

    // What: an independent cache cannot use another context's plan or
    // structure to admit a malformed raw structure.
    assert_eq!(builtin_tree_cache_state(&independent), independent_before);
}

#[test]
fn exact_warm_structure_reuses_prior_local_admission_proof() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_global_operation_caches();

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
    };
    let structure = simple_su2_vertex_structure(1);
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    let cold = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &rule, &operation, &structure, &structure, false,
        )
        .unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);

    calls.store(0, Ordering::Relaxed);
    let same_content = Arc::new((*structure).clone());
    assert_eq!(same_content.content_id(), structure.content_id());
    assert!(!Arc::ptr_eq(&same_content, &structure));
    let warm = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &rule,
            &operation,
            &same_content,
            &same_content,
            false,
        )
        .unwrap();

    // What: an exact semantic-key/content replay reuses the LOCAL admission
    // proof carried by the published compiled structure.
    assert!(Arc::ptr_eq(&cold, &warm));
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    cache.set_policy(OperationCachePolicy::NoCache);
    calls.store(0, Ordering::Relaxed);
    cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &rule, &operation, &structure, &structure, false,
        )
        .unwrap();

    // What: clearing retained entries removes proof reuse; the next call
    // performs LOCAL admission again.
    assert!(calls.load(Ordering::Relaxed) > 0);
}

#[test]
fn prelowered_same_content_roles_share_one_local_admission() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_global_operation_caches();

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
    };
    let logical = simple_su2_vertex_structure(1);
    let storage = Arc::new((*logical).clone());
    let destination = Arc::new((*logical).clone());
    assert_eq!(logical.content_id(), storage.content_id());
    assert_eq!(logical.content_id(), destination.content_id());
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    cache
        .get_or_compile_tree_pair_prelowered(
            &rule,
            &TreeTransformOperation::braid([1, 0], [], [0, 1], []),
            &destination,
            &logical,
            &storage,
            false,
            Ok,
            Ok,
        )
        .unwrap();

    // What: logical, storage, and destination roles sharing immutable content
    // perform one LOCAL categorical admission, even through distinct Arcs.
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn warm_prelowered_raw_arc_aliases_do_not_observe_or_mutate_cache_state() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_global_operation_caches();

    let valid = simple_su2_vertex_structure(1);
    let invalid = simple_su2_vertex_structure(0);
    assert_ne!(valid.content_id(), invalid.content_id());
    assert!(!Arc::ptr_eq(&valid, &invalid));
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    cache
        .get_or_compile_tree_pair_prelowered(
            &SU2FusionRule,
            &operation,
            &valid,
            &valid,
            &valid,
            false,
            Ok,
            Ok,
        )
        .unwrap();
    let local_before = builtin_tree_cache_state(&cache);

    for (destination, logical_source, storage_source) in [
        (&valid, &invalid, &valid),
        (&valid, &valid, &invalid),
        (&invalid, &valid, &valid),
    ] {
        let error = cache
            .get_or_compile_tree_pair_prelowered(
                &SU2FusionRule,
                &operation,
                destination,
                logical_source,
                storage_source,
                false,
                Ok,
                Ok,
            )
            .unwrap_err();
        assert_invalid_simple_vertex(error);

        // What: each raw-Arc role is categorically admitted before a warm
        // plan/structure lookup can observe its shared content identity.
        assert_eq!(builtin_tree_cache_state(&cache), local_before);
    }
}

#[test]
fn invalid_simple_source_does_not_mutate_cached_or_no_cache_state() {
    // What: malformed categorical input is rejected before cache statistics or
    // retained plan/structure state change under either execution policy.
    for policy in [
        OperationCachePolicy::default(),
        OperationCachePolicy::NoCache,
    ] {
        let (dst, src) = malformed_simple_su2_tree_pair_tensors();
        let mut cache =
            TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::with_policy(policy);
        let before_stats = cache.stats();

        let error = cache
            .get_or_compile_tree_pair(
                &SU2FusionRule,
                TreeTransformOperation::braid([0, 1], [], [0, 1], []),
                &dst,
                &src,
            )
            .unwrap_err();

        assert_eq!(
            error,
            OperationError::Core(CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            })
        );
        assert_eq!(cache.stats(), before_stats);
        assert_eq!(cache.plan_len(), 0);
        assert_eq!(cache.structure_len(), 0);
        assert_eq!(cache.fast_structure_len(), 0);
    }
}

#[test]
fn malformed_source_precedes_noncategorical_destination_without_cache_mutation() {
    let (_, src) = malformed_simple_su2_tree_pair_tensors();
    let dst_structure = packed_fixture_structure(2, [(BlockKey::opaque([9]), vec![1, 1])]).unwrap();
    let dst = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![0.0],
        TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap(),
        dst_structure,
    )
    .unwrap();
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    let error = cache
        .get_or_compile_tree_pair(
            &SU2FusionRule,
            TreeTransformOperation::permute([0, 1], []),
            &dst,
            &src,
        )
        .unwrap_err();

    // What: cold admission proves the categorical source before diagnosing
    // the destination namespace, without touching cache state.
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree contains an inadmissible fusion vertex",
        })
    );
    assert_eq!(cache.stats(), TreeTransformCacheStats::default());
    assert!(cache.is_empty());
}

#[test]
fn invalid_operation_precedes_malformed_simple_source_without_cache_mutation() {
    use crate::tree_transform::{
        reset_tree_pair_operation_preparations, tree_pair_operation_preparations,
    };

    // What: operation syntax has deterministic precedence over categorical
    // source admission and neither failure is counted as a cache miss.
    let (dst, src) = malformed_simple_su2_tree_pair_tensors();
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    reset_tree_pair_operation_preparations();
    let error = cache
        .get_or_compile_tree_pair(
            &SU2FusionRule,
            TreeTransformOperation::braid([0, 0], [], [0, 1], []),
            &dst,
            &src,
        )
        .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(CoreError::InvalidPermutation {
            permutation: vec![0, 0],
            rank: 2,
        })
    );
    assert_eq!(tree_pair_operation_preparations(), 0);
    assert_eq!(cache.stats(), TreeTransformCacheStats::default());
    assert!(cache.is_empty());
}

#[test]
fn invalid_operation_precedes_non_categorical_namespace_without_cache_mutation() {
    for key in [BlockKey::Dense, BlockKey::opaque([7, 11])] {
        let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
            key.clone(),
            vec![1, 1],
            0,
        )
        .unwrap()])
        .unwrap();
        let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
        let dst = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            vec![0.0],
            space.clone(),
            structure.clone(),
        )
        .unwrap();
        let src =
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0], space, structure).unwrap();
        let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

        let error = cache
            .get_or_compile_tree_pair(
                &SU2FusionRule,
                TreeTransformOperation::permute([0, 0], []),
                &dst,
                &src,
            )
            .unwrap_err();
        assert_eq!(
            error,
            OperationError::Core(CoreError::InvalidPermutation {
                permutation: vec![0, 0],
                rank: 2,
            })
        );
        assert_eq!(cache.stats(), TreeTransformCacheStats::default());
        assert!(cache.is_empty());

        let error = cache
            .get_or_compile_tree_pair(
                &SU2FusionRule,
                TreeTransformOperation::permute([0, 1], []),
                &dst,
                &src,
            )
            .unwrap_err();
        assert_eq!(
            error,
            OperationError::Core(CoreError::ExpectedFusionTreePairKey { actual: key.kind() })
        );
        // What: syntax and namespace failures occur before any cache lookup or
        // categorical coefficient provider can compile a row.
        assert_eq!(cache.stats(), TreeTransformCacheStats::default());
        assert!(cache.is_empty());
    }
}

#[test]
fn invalid_unique_source_does_not_count_eager_compile_misses() {
    // What: a failed Unique eager build reports no successful plan or structure
    // compile in cache statistics.
    let valid = all_codomain_fusion_tree_test_key([1, 1], 0, [false, false], [], [1]);
    let malformed = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            1,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let dst_structure = packed_fixture_structure(2, [(valid, vec![1, 1])]).unwrap();
    let src_structure = packed_fixture_structure(2, [(malformed, vec![1, 1])]).unwrap();
    let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0], space.clone(), dst_structure)
            .unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0], space, src_structure).unwrap();
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    let error = cache
        .get_or_compile_tree_pair(
            &Z2FusionRule,
            TreeTransformOperation::permute([0, 1], []),
            &dst,
            &src,
        )
        .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree contains an inadmissible fusion vertex",
        })
    );
    assert_eq!(cache.stats(), TreeTransformCacheStats::default());
    assert!(cache.is_empty());
}

#[test]
fn direct_tree_pair_rejects_nonalias_malformed_destination_without_writing() {
    let (mut dst, src) = valid_simple_source_and_nonalias_malformed_destination();
    let before = dst.data().to_vec();

    let error = tree_transform_into(
        &SU2FusionRule,
        TreeTransformOperation::permute([0, 1], []),
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap_err();

    // What: noncached tree-pair admission validates the destination before
    // replay can modify caller-owned storage.
    assert_invalid_simple_vertex(error);
    assert_eq!(dst.data(), before);
}

#[test]
fn typed_all_codomain_rejects_nonalias_malformed_destination_without_state_or_output_change() {
    let (mut dst, src) = valid_simple_source_and_nonalias_malformed_destination();
    let before_output = dst.data().to_vec();
    let mut context =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    let before_cache = builtin_tree_cache_state(context.cache());

    let error = context
        .all_codomain_tree_transform_into(
            &SU2FusionRule,
            TreeTransformOperation::permute([0, 1], []),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap_err();

    // What: typed constructors canonicalize equal content identities, so this
    // nonalias fixture proves all-codomain destination admission without a
    // fabricated raw-Arc alias.
    assert_invalid_simple_vertex(error);
    assert_eq!(builtin_tree_cache_state(context.cache()), before_cache);
    assert_eq!(dst.data(), before_output);
}

#[test]
fn all_codomain_pair_mismatch_is_rejected_before_source_scope_or_cache_state() {
    // What: whole-pair categorical admission precedes the all-codomain source
    // restriction, so mismatched coupled sectors retain the core error.
    let nonempty_domain = BlockKey::from(FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(0),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(0)],
            SectorId::new(0),
            [false],
            [],
            [],
        )
        .unwrap(),
    ));
    let mismatched = BlockKey::from(FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(0),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(2)],
            SectorId::new(2),
            [false],
            [],
            [],
        )
        .unwrap(),
    ));
    let src_structure = packed_fixture_structure(
        3,
        [
            (nonempty_domain, vec![1, 1, 1]),
            (mismatched, vec![1, 1, 1]),
        ],
    )
    .unwrap();
    let dst_key =
        all_codomain_fusion_tree_test_key([0, 0, 0], 0, [false, false, false], [0], [1, 1]);
    let dst_structure = packed_fixture_structure(3, [(dst_key, vec![1, 1, 1])]).unwrap();
    let src_space = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
    let dst_space = TensorMapSpace::<3, 0>::from_dims([1, 1, 1], []).unwrap();
    let src =
        TensorMap::<f64, 2, 1>::from_vec_with_structure(vec![1.0, 2.0], src_space, src_structure)
            .unwrap();
    let dst = TensorMap::<f64, 3, 0>::from_vec_with_structure(vec![0.0], dst_space, dst_structure)
        .unwrap();
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    let error = cache
        .get_or_compile_all_codomain(
            &SU2FusionRule,
            TreeTransformOperation::permute([0, 1, 2], []),
            &dst,
            &src,
        )
        .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        })
    );

    let error = cache
        .get_or_compile_all_codomain(
            &SU2FusionRule,
            TreeTransformOperation::permute([0, 0, 2], []),
            &dst,
            &src,
        )
        .unwrap_err();
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        })
    );

    let error = cache
        .get_or_compile_all_codomain(
            &SU2FusionRule,
            TreeTransformOperation::transpose([0, 1, 2], []),
            &dst,
            &src,
        )
        .unwrap_err();
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        })
    );

    // What: whole-pair source admission also precedes all-codomain operation
    // scope without publishing cache state.
    assert_eq!(cache.stats(), TreeTransformCacheStats::default());
    assert!(cache.is_empty());
}

#[test]
fn independent_tree_transform_contexts_compile_their_own_artifacts() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
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
        vec![3.0, -4.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [23, 29, 31, 37], []);

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
    };
    let mut first =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    let run =
        |context: &mut TreeTransformExecutionContext<f64, TreeTransformBuiltinRuleCacheKey>| {
            let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
                vec![1.0, 2.0],
                space.clone(),
                block_structure.clone(),
            )
            .unwrap();
            context
                .tree_transform_into(&rule, operation.clone(), &mut dst, &src, 2.0, -1.0)
                .unwrap();
            dst.data().to_vec()
        };
    let first_data = run(&mut first);
    assert_eq!(first.cache().stats().plan_misses(), 1);
    assert_eq!(legacy_tree_row_stats(first.cache().stats()), (0, 0));
    assert!(calls.load(Ordering::Relaxed) > 0);

    calls.store(0, Ordering::Relaxed);
    let mut second =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    let second_data = run(&mut second);
    assert_eq!(second.cache().stats().plan_misses(), 1);
    assert_eq!(legacy_tree_row_stats(second.cache().stats()), (0, 0));
    assert!(calls.load(Ordering::Relaxed) > 0);

    // What: a fresh execution context recompiles the exact SU(2) transform
    // instead of inheriting another context's plan, structure, or rows.
    assert_eq!(second_data, first_data);
}

#[test]
fn concurrent_tree_transform_contexts_do_not_share_compiled_artifacts() {
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let run = |barrier: Arc<std::sync::Barrier>| {
        std::thread::spawn(move || {
            let calls = Arc::new(AtomicUsize::new(0));
            let rule = AdmissionCountingSu2Rule {
                nsymbol_calls: Arc::clone(&calls),
            };
            let structure = simple_su2_vertex_structure(1);
            let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
            let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::default();
            barrier.wait();
            cache
                .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
                    &rule, &operation, &structure, &structure, false,
                )
                .unwrap();
            (
                calls.load(Ordering::Relaxed),
                builtin_tree_cache_state(&cache),
            )
        })
    };

    let left = run(Arc::clone(&barrier));
    let right = run(barrier);
    let (left_calls, left_state) = left.join().unwrap();
    let (right_calls, right_state) = right.join().unwrap();

    // What: simultaneous fresh contexts each perform categorical admission and
    // retain one private plan/structure instead of observing sibling state.
    assert!(left_calls > 0);
    assert!(right_calls > 0);
    assert_eq!(left_state.0.plan_misses(), 1);
    assert_eq!(right_state.0.plan_misses(), 1);
    assert_eq!(left_state.1, 1);
    assert_eq!(right_state.1, 1);
    assert_eq!(left_state.2, 1);
    assert_eq!(right_state.2, 1);
}

#[test]
fn tree_transform_cache_reuses_su2_recoupling_descriptor() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
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
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
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
fn all_codomain_plan_miss_does_not_retain_source_rows() {
    let key = |inner: [usize; 2]| {
        all_codomain_fusion_tree_test_key_for_rule(
            &SU2FusionRule,
            [4, 4, 4, 4],
            0,
            [false, false, false, false],
            inner,
            [1, 1, 1],
        )
    };
    let keys = [key([0, 4]), key([2, 4]), key([4, 4]), key([6, 4])];
    let dst_keys = [
        key([0, 4]),
        key([2, 4]),
        key([4, 4]),
        key([6, 4]),
        key([8, 4]),
    ];
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let make = |keys: &[FusionTreePairKey], data: Vec<f64>| {
        let structure = packed_fixture_structure(
            4,
            keys.iter().map(|key| (key.clone(), vec![1usize, 1, 1, 1])),
        )
        .unwrap();
        let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
        TensorMap::<f64, 4, 0>::from_vec_with_structure(data, space, structure).unwrap()
    };

    let src_small = make(&keys[..2], vec![10.0, 20.0]);
    let dst_small = make(&dst_keys, vec![0.0; dst_keys.len()]);
    let src_big = make(&keys, vec![1.0, 2.0, 3.0, 4.0]);
    let mut dst_cached = make(&dst_keys, vec![0.0; dst_keys.len()]);
    let mut dst_uncached = make(&dst_keys, vec![0.0; dst_keys.len()]);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    cache
        .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst_small, &src_small)
        .unwrap();
    assert_eq!(legacy_tree_row_stats(cache.stats()), (0, 0));

    let cached_structure = cache
        .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst_cached, &src_big)
        .unwrap();
    assert_eq!(cache.stats().plan_misses(), 2);
    // What: a different all-codomain source set preserves the direct ordered
    // numerical result without reviving retained row-cache statistics.
    assert_eq!(legacy_tree_row_stats(cache.stats()), (0, 0));

    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &cached_structure,
        &mut dst_cached,
        &src_big,
        1.0,
        0.0,
    )
    .unwrap();

    let uncached_plan = build_all_codomain_tree_transform_group_plan(
        &SU2FusionRule,
        operation,
        src_big.structure(),
    )
    .unwrap();
    let uncached_structure = uncached_plan.compile(&dst_uncached, &src_big).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &uncached_structure,
        &mut dst_uncached,
        &src_big,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(
        dst_cached
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        dst_uncached
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn tree_transform_execution_context_reuses_all_codomain_cache() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
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
fn all_codomain_artifacts_do_not_cross_context_boundaries() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let structure = packed_fixture_structure(
        4,
        [(src_key0, vec![1, 1, 1, 1]), (src_key1, vec![1, 1, 1, 1])],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, structure).unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

    let mut first = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    let cold = first
        .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
        .unwrap();
    let warm = first
        .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
        .unwrap();
    assert!(Arc::ptr_eq(&cold, &warm));

    let mut fresh = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    let rebuilt = fresh
        .get_or_compile_all_codomain(&SU2FusionRule, operation, &dst, &src)
        .unwrap();

    // What: all-codomain plan, structure, and row state remain private to the
    // explicit context while a fresh context rebuilds an equal artifact.
    assert!(!Arc::ptr_eq(&cold, &rebuilt));
    assert_eq!(cold.as_ref(), rebuilt.as_ref());
    assert_eq!(first.stats().plan_hits(), 1);
    assert_eq!(first.stats().structure_hits(), 1);
    assert_eq!(fresh.stats().plan_hits(), 0);
    assert_eq!(fresh.stats().plan_misses(), 1);
    assert_eq!(legacy_tree_row_stats(fresh.stats()), (0, 0));
}

#[test]
fn tree_transform_execution_context_no_cache_rebuilds_without_retaining_entries() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
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
        assert_eq!(context.cache().fast_structure_len(), 0);
        assert_eq!(context.cache().stats().plan_hits(), 0);
        assert_eq!(context.cache().stats().structure_hits(), 0);
        assert_eq!(context.cache().stats().plan_misses(), expected_misses);
        assert_eq!(context.cache().stats().structure_misses(), expected_misses);
        assert_eq!(legacy_tree_row_stats(context.cache().stats()), (0, 0));
    }

    let expected = dst.data().to_vec();
    for policy in [
        OperationCachePolicy::TaskLocal,
        OperationCachePolicy::task_local_lru(1),
    ] {
        let mut policy_dst = dst.clone();
        policy_dst.data_mut().fill(0.0);
        let mut policy_context =
            TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
        policy_context.cache_mut().set_policy(policy);
        policy_context
            .all_codomain_tree_transform_into(
                &SU2FusionRule,
                operation.clone(),
                &mut policy_dst,
                &src,
                1.0,
                0.0,
            )
            .unwrap();

        // What: cache ownership and eviction policy do not change the tensor
        // produced by the shared eager compiler.
        assert_eq!(policy_dst.data(), expected);
        assert_eq!(
            legacy_tree_row_stats(policy_context.cache().stats()),
            (0, 0)
        );
    }
}

#[test]
fn tree_transform_execution_context_task_local_lru_evicts_old_transformer() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
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
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
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
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [true], [true], [], [], [], [])
            .unwrap(),
    );
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
    assert_eq!(spec.src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(spec.dst_keys(), &[expect_tree_key(&expected_dst_key)]);
    assert_eq!(spec.recoupling_coefficients_dst_src().len(), 1);
    assert!((spec.recoupling_coefficients_dst_src()[0] - 1.0).abs() < 1.0e-12);
    plan.compile_structures(&dst_structure, &src_structure)
        .unwrap();
}

#[test]
fn tree_pair_transform_public_helper_executes_su2_domain_crossing() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [true], [true], [], [], [], [])
            .unwrap(),
    );
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
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [true], [true], [], [], [], [])
            .unwrap(),
    );
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
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [1],
            1,
            [false, false],
            [false],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
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
fn unique_tree_pair_compile_bypasses_plan_and_structure_caches() {
    // Why-not: UniqueFusion intentionally bypasses reusable plan/row caches;
    // this protects the process from retaining cheap, layout-specific keys.
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1],
            1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        )
        .unwrap(),
    );
    // The cache API requires the built-in rule-key marker and rigid symbols;
    // use the production Z2 rule, whose semantics match the local UniqueZ2
    // oracle while satisfying both bounds without adding test-only adapters.
    let rule = Z2FusionRule;
    assert_eq!(rule.fusion_style(), FusionStyleKind::Unique);
    let src_tree = expect_tree_key(&src_key);
    let operation = TreeTransformOperation::permute([0, 2], [1]);
    let (dst_tree, _) = unique_permute_tree_pair(&rule, &src_tree, &[0, 2], &[1]).unwrap();
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let dst_structure =
        packed_fixture_structure(3, [(BlockKey::from(dst_tree), vec![1, 1, 1])]).unwrap();
    let src_space = TensorMapSpace::<1, 2>::from_dims([1], [1, 1]).unwrap();
    let dst_space = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
    let src = TensorMap::<f64, 1, 2>::from_vec_with_structure(vec![7.0], src_space, src_structure)
        .unwrap();
    let dst = TensorMap::<f64, 2, 1>::from_vec_with_structure(vec![0.0], dst_space, dst_structure)
        .unwrap();
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    let first = cache
        .get_or_compile_tree_pair(&rule, operation.clone(), &dst, &src)
        .unwrap();
    let second = cache
        .get_or_compile_tree_pair(&rule, operation, &dst, &src)
        .unwrap();
    assert_eq!(first.as_ref(), second.as_ref());
    assert_eq!(cache.plan_len(), 0);
    assert_eq!(cache.structure_len(), 0);
    assert_eq!(cache.stats().plan_hits(), 0);
    assert_eq!(cache.stats().structure_hits(), 0);

    let dst_structure = Arc::new(dst.structure().clone());
    let src_structure = Arc::new(src.structure().clone());
    let direct_plan = build_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::permute([0, 2], [1]),
        &src_structure,
    )
    .unwrap();
    let direct = direct_plan
        .compile_shared_structures_with_storage_conjugation(
            Arc::clone(&dst_structure),
            Arc::clone(&src_structure),
            true,
        )
        .unwrap();
    let _ = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation(
            &rule,
            TreeTransformOperation::permute([0, 2], [1]),
            &dst_structure,
            &src_structure,
            true,
        )
        .unwrap();
    // Why-not: bypassing the cache must not change the compiled structural or
    // numerical result relative to the direct compiler.
    let cached_storage = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation(
            &rule,
            TreeTransformOperation::permute([0, 2], [1]),
            &dst_structure,
            &src_structure,
            true,
        )
        .unwrap();
    assert_eq!(cached_storage.as_ref(), &direct);
    assert!(cached_storage.storage_conjugate());
    assert_eq!(cache.plan_len(), 0);
    assert_eq!(cache.structure_len(), 0);
    assert_eq!(cache.fast_structure_len(), 0);
}

#[test]
fn u1_unique_tree_pair_remains_eager_under_stored_policy() {
    let positive = U1Irrep::new(1).sector_id();
    let negative = U1Irrep::new(-1).sector_id();
    let vacuum = U1FusionRule.vacuum();
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &U1FusionRule,
            [positive, negative],
            vacuum,
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&U1FusionRule, [], vacuum, [], [], []).unwrap(),
    );
    let destination = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &U1FusionRule,
            [negative, positive],
            vacuum,
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&U1FusionRule, [], vacuum, [], [], []).unwrap(),
    );
    let src_structure =
        packed_fixture_structure(2, [(BlockKey::from(source), vec![1, 1])]).unwrap();
    let dst_structure =
        packed_fixture_structure(2, [(BlockKey::from(destination), vec![1, 1])]).unwrap();
    let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![7.0], space.clone(), src_structure)
            .unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0], space, dst_structure).unwrap();
    let operation = TreeTransformOperation::permute([1, 0], []);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    let first = cache
        .get_or_compile_tree_pair(&U1FusionRule, operation.clone(), &dst, &src)
        .unwrap();
    let second = cache
        .get_or_compile_tree_pair(&U1FusionRule, operation, &dst, &src)
        .unwrap();

    // What: U(1)'s Unique transform recompiles under the stored policy and
    // never retains plan, structure, or fast-front state.
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(cache.stats().plan_hits(), 0);
    assert_eq!(cache.stats().plan_misses(), 2);
    assert_eq!(cache.stats().structure_hits(), 0);
    assert_eq!(cache.stats().structure_misses(), 2);
    assert_eq!(cache.plan_len(), 0);
    assert_eq!(cache.structure_len(), 0);
    assert_eq!(cache.fast_structure_len(), 0);
}

#[test]
fn tree_pair_transform_public_helper_executes_split_changing_permute() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1],
            1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        )
        .unwrap(),
    );
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
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1],
            1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        )
        .unwrap(),
    );
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
            key: Box::new(expected_missing),
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
fn eager_product_tree_pair_plan_bypasses_legacy_row_assembly() {
    use crate::tree_transform::{reset_tree_pair_lowering_calls, tree_pair_lowering_calls};

    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let group_count = src_space.subblock_structure().fusion_tree_groups().len();
    let legacy = build_tree_transform_group_plan(
        &rule,
        operation.clone(),
        src_space.subblock_structure(),
        |source| {
            tenet_core::multiplicity_free_permute_tree_pair(&rule, source, &[1, 0], &[2])
                .map_err(OperationError::from_core_preserving_context)
        },
    )
    .unwrap();

    reset_tree_pair_lowering_calls();
    let direct =
        build_tree_pair_transform_group_plan(&rule, operation, src_space.subblock_structure())
            .unwrap();

    // What: the eager Simple builder preserves the exact legacy plan while
    // consuming one core ordered map per whole group instead of rebuilding
    // full-key rows per source column.
    assert_eq!(direct, legacy);
    assert_eq!(tree_pair_lowering_calls(), (group_count, 0));
    direct
        .compile_structures(
            dst_space.subblock_structure(),
            src_space.subblock_structure(),
        )
        .unwrap();
}

#[test]
fn eager_simple_plan_prepares_once_per_distinct_source_split() {
    use crate::tree_transform::{
        reset_tree_pair_operation_preparations, tree_pair_operation_preparations,
    };

    let half_id = 1usize;
    let one_id = 2usize;
    let vacuum_id = 0usize;
    let half = SU2Irrep::from_twice_spin(half_id).sector_id();
    let vacuum = SU2FusionRule.vacuum();
    let group_a = [[vacuum_id, half_id], [one_id, half_id]].map(|inner| {
        all_codomain_fusion_tree_test_key_for_rule(
            &SU2FusionRule,
            [half_id; 4],
            vacuum_id,
            [false; 4],
            inner,
            [1, 1, 1],
        )
    });
    let group_b = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [half_id, half_id, one_id, one_id],
        vacuum_id,
        [false; 4],
        [vacuum_id, one_id],
        [1, 1, 1],
    );
    let split_group = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [half; 3],
            half,
            [false; 3],
            [vacuum],
            [MultiplicityIndex::ONE; 2],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&SU2FusionRule, [half], half, [false], [], []).unwrap(),
    );
    let source_order = [group_a[0].clone(), group_b, group_a[1].clone(), split_group];
    let structure = packed_fixture_structure(
        4,
        source_order
            .iter()
            .cloned()
            .map(|key| (key, vec![1usize; 4])),
    )
    .unwrap();
    assert_eq!(
        structure
            .fusion_tree_group_slice()
            .iter()
            .map(|group| group.block_indices())
            .collect::<Vec<_>>(),
        vec![&[0, 2][..], &[1][..], &[3][..]]
    );
    let operation = TreeTransformOperation::permute([1, 0, 2], [3]);
    let oracle =
        build_tree_transform_group_plan(&SU2FusionRule, operation.clone(), &structure, |source| {
            tenet_core::multiplicity_free_permute_tree_pair(
                &SU2FusionRule,
                source,
                &[1, 0, 2],
                &[3],
            )
            .map_err(OperationError::from_core_preserving_context)
        })
        .unwrap();

    reset_tree_pair_operation_preparations();
    let prepared =
        build_tree_pair_transform_group_plan(&SU2FusionRule, operation, &structure).unwrap();

    // What: two source splits across three interleaved groups prepare exactly
    // twice, while preserving the independent per-source oracle's grouping,
    // source/destination order, and coefficients.
    assert_eq!(tree_pair_operation_preparations(), 2);
    assert_eq!(prepared.specs().len(), oracle.specs().len());
    for (actual, expected) in prepared.specs().iter().zip(oracle.specs()) {
        assert_eq!(actual.group_key(), expected.group_key());
        assert_eq!(actual.src_keys(), expected.src_keys());
        assert_eq!(actual.dst_keys(), expected.dst_keys());
        assert_eq!(actual.source_axes(), expected.source_axes());
        assert_eq!(
            actual.recoupling_coefficients_dst_src().len(),
            expected.recoupling_coefficients_dst_src().len()
        );
        for (&actual, &expected) in actual
            .recoupling_coefficients_dst_src()
            .iter()
            .zip(expected.recoupling_coefficients_dst_src())
        {
            assert!((actual - expected).abs() <= 1.0e-12);
        }
    }
}

#[test]
fn rank_nine_same_split_groups_prepare_once_and_lower_once_each() {
    // What: rank beyond the prepared operation's inline capacity does not
    // clone spilled step storage for each same-split fusion group.
    use crate::tree_transform::{
        build_tree_pair_transform_group_plan_validated_with_threads,
        reset_tree_pair_lowering_calls, reset_tree_pair_operation_preparations,
        tree_pair_lowering_calls, tree_pair_operation_preparations,
        validate_multiplicity_free_tree_pair_preflight,
    };

    let vacuum = SU2FusionRule.vacuum();
    let keys = [0usize, 1, 2].map(|twice_spin| {
        let mut uncoupled = [vacuum; 9];
        uncoupled[0] = SectorId::new(twice_spin);
        uncoupled[1] = SectorId::new(twice_spin);
        BlockKey::from(FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                uncoupled,
                vacuum,
                [false; 9],
                [vacuum; 7],
                [MultiplicityIndex::ONE; 8],
            )
            .unwrap(),
            empty_fusion_tree(),
        ))
    });
    let structure =
        packed_fixture_structure(9, keys.into_iter().map(|key| (key, vec![1usize; 9]))).unwrap();
    let operation = TreeTransformOperation::braid([1, 0, 2, 3, 4, 5, 6, 7, 8], [], 0..9, []);
    let proof =
        validate_multiplicity_free_tree_pair_preflight(&SU2FusionRule, &operation, &structure)
            .unwrap();

    reset_tree_pair_operation_preparations();
    reset_tree_pair_lowering_calls();
    let plan =
        build_tree_pair_transform_group_plan_validated_with_threads(&proof, operation, 1).unwrap();

    assert_eq!(structure.fusion_tree_groups().len(), 3);
    assert_eq!(tree_pair_operation_preparations(), 1);
    assert_eq!(tree_pair_lowering_calls(), (3, 0));
    assert_eq!(plan.specs().len(), 3);
}

#[test]
fn eager_su2_dense_tree_pair_plan_matches_legacy_replay() {
    use crate::tree_transform::{reset_tree_pair_lowering_calls, tree_pair_lowering_calls};

    let source0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let source1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let structure = packed_fixture_structure(
        4,
        [(source0, vec![1, 1, 1, 1]), (source1, vec![1, 1, 1, 1])],
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let legacy =
        build_tree_transform_group_plan(&SU2FusionRule, operation.clone(), &structure, |source| {
            tenet_core::multiplicity_free_braid_tree_pair(
                &SU2FusionRule,
                source,
                &[0, 2, 1, 3],
                &[],
                &[0, 1, 2, 3],
                &[],
            )
            .map_err(OperationError::from_core_preserving_context)
        })
        .unwrap();

    reset_tree_pair_lowering_calls();
    let direct =
        build_tree_pair_transform_group_plan(&SU2FusionRule, operation, &structure).unwrap();
    assert_eq!(direct, legacy);
    assert_eq!(tree_pair_lowering_calls(), (1, 0));

    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut direct_dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![0.0, 0.0],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut legacy_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, structure.clone())
            .unwrap();
    let direct_structure = direct.compile(&direct_dst, &src).unwrap();
    let legacy_structure = legacy.compile(&legacy_dst, &src).unwrap();
    let mut direct_backend = DenseTreeTransformOperations::default();
    let mut legacy_backend = DenseTreeTransformOperations::default();
    let mut direct_workspace = TreeTransformWorkspace::default();
    let mut legacy_workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut direct_backend,
        &mut direct_workspace,
        &direct_structure,
        &mut direct_dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    tree_transform_execute_with(
        &mut legacy_backend,
        &mut legacy_workspace,
        &legacy_structure,
        &mut legacy_dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    assert_eq!(direct_dst.data(), legacy_dst.data());
}

#[test]
fn product_tree_pair_plan_is_thread_count_invariant() {
    use crate::tree_transform::{
        build_tree_pair_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_tree_pair_preflight,
    };

    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let build = |threads| {
        let proof = validate_multiplicity_free_tree_pair_preflight(
            &rule,
            &operation,
            src_space.subblock_structure(),
        )
        .unwrap();
        build_tree_pair_transform_group_plan_validated_with_threads(
            &proof,
            operation.clone(),
            threads,
        )
        .unwrap()
    };

    let serial = build(1);
    for threads in [2, 4] {
        let threaded = build(threads);
        // What: product-symmetry fermionic phases and plan order are identical
        // when scheduling the same fusion groups.
        assert_eq!(threaded, serial);
        threaded
            .compile_structures(
                dst_space.subblock_structure(),
                src_space.subblock_structure(),
            )
            .unwrap();
    }
    assert!((single_transform_coefficient_for_coupled(&serial, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&serial, c1) + 1.0).abs() < 1.0e-12);
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
        let src_offset = block_offset_for_tree_pair(src.structure(), src_key);
        let dst_offset = block_offset_for_tree_pair(dst.structure(), dst_key);
        expected[dst_offset] +=
            2.0 * spec.recoupling_coefficients_dst_src()[0] * src.data()[src_offset];
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
fn product_tree_transform_reuse_is_owned_by_one_explicit_context() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space).unwrap();
    let dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space).unwrap();

    let mut first = TreeTransformCache::<f64, RuleKey>::default();
    let cold = first
        .get_or_compile_tree_pair(&rule, operation.clone(), &dst, &src)
        .unwrap();
    let warm = first
        .get_or_compile_tree_pair(&rule, operation.clone(), &dst, &src)
        .unwrap();
    assert!(Arc::ptr_eq(&cold, &warm));
    assert_eq!(first.stats().plan_hits(), 1);
    assert_eq!(first.stats().structure_hits(), 1);

    let mut fresh = TreeTransformCache::<f64, RuleKey>::default();
    let fresh_structure = fresh
        .get_or_compile_tree_pair(&rule, operation.clone(), &dst, &src)
        .unwrap();
    assert!(!Arc::ptr_eq(&cold, &fresh_structure));
    assert_eq!(fresh_structure.as_ref(), cold.as_ref());
    assert_eq!(fresh.stats().plan_hits(), 0);
    assert_eq!(fresh.stats().plan_misses(), 1);
    assert_eq!(legacy_tree_row_stats(fresh.stats()), (0, 0));

    let mut no_cache =
        TreeTransformCache::<f64, RuleKey>::with_policy(OperationCachePolicy::NoCache);
    let first_eager = no_cache
        .get_or_compile_tree_pair(&rule, operation.clone(), &dst, &src)
        .unwrap();
    let second_eager = no_cache
        .get_or_compile_tree_pair(&rule, operation, &dst, &src)
        .unwrap();

    // What: fZ2 x U(1) x SU(2) reuses an exact compiled artifact only inside
    // the explicit owning context; fresh and NoCache contexts rebuild it.
    assert!(!Arc::ptr_eq(&first_eager, &second_eager));
    assert_eq!(first_eager.as_ref(), cold.as_ref());
    assert_eq!(second_eager.as_ref(), cold.as_ref());
    assert_eq!(no_cache.stats().plan_hits(), 0);
    assert_eq!(no_cache.stats().plan_misses(), 2);
    assert_eq!(no_cache.plan_len(), 0);
    assert_eq!(no_cache.structure_len(), 0);
    assert_eq!(no_cache.fast_structure_len(), 0);
}

#[test]
fn product_tree_transform_rebuilds_after_global_cache_reset_with_old_values_live() {
    // What: a reset may drop every semantic layout/transform artifact while old
    // product-symmetry tensors remain live; rebuilding preserves the fZ2 swap sign.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let src_hom = src_space.homspace().clone();
    let dst_hom = dst_space.homspace().clone();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src_data = vec![10.0, 20.0];
    let initial_dst = vec![1.0, 2.0];
    let src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(src_data.clone(), src_space).unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(initial_dst.clone(), dst_space).unwrap();
    tree_transform_into(&rule, operation.clone(), &mut dst, &src, 2.0, 3.0).unwrap();
    let expected = dst.data().to_vec();

    reset_global_operation_caches();
    let rebuilt_src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
        src_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let rebuilt_dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
        dst_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let rebuilt_src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(src_data, rebuilt_src_space).unwrap();
    let mut rebuilt_dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(initial_dst, rebuilt_dst_space).unwrap();
    tree_transform_into(&rule, operation, &mut rebuilt_dst, &rebuilt_src, 2.0, 3.0).unwrap();

    assert_eq!(rebuilt_dst.data(), expected.as_slice());
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
        let src_offset = block_offset_for_tree_pair(src.structure(), src_key);
        let dst_offset = block_offset_for_tree_pair(dst.structure(), dst_key);
        expected[dst_offset] = expected[dst_offset]
            + src.data()[src_offset]
                .scale_by_coefficient(spec.recoupling_coefficients_dst_src()[0])
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
fn tree_transform_overwrite_facade_and_context_ignore_destination_bits() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space).unwrap();
    let mut expected = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![0.0; dst_space.required_len().unwrap()],
        dst_space.clone(),
    )
    .unwrap();
    tree_transform_into(&rule, operation.clone(), &mut expected, &src, 2.0, 0.0).unwrap();

    let mut one_shot = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![f64::NAN; dst_space.required_len().unwrap()],
        dst_space.clone(),
    )
    .unwrap();
    tree_transform_overwrite_into(&rule, operation.clone(), &mut one_shot, &src, 2.0).unwrap();
    assert_eq!(one_shot.data(), expected.data());

    let mut cached = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![f64::NAN; dst_space.required_len().unwrap()],
        dst_space,
    )
    .unwrap();
    let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();
    for _ in 0..2 {
        cached.data_mut().fill(f64::NAN);
        tree_transform_overwrite_into_with_context(
            &mut context,
            &rule,
            operation.clone(),
            &mut cached,
            &src,
            2.0,
        )
        .unwrap();
        assert_eq!(cached.data(), expected.data());
    }
    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);

    // What: the borrowed-operation overwrite entry point ignores destination bits
    // while reusing the same compiled structure on its warm invocation.
    let dst_structure = Arc::clone(cached.structure());
    let src_structure = Arc::clone(src.structure());
    for _ in 0..2 {
        cached.data_mut().fill(f64::NAN);
        context
            .tree_transform_dyn_overwrite_into_ref(
                &rule,
                &operation,
                &dst_structure,
                &src_structure,
                cached.data_mut(),
                src.data(),
                2.0,
            )
            .unwrap();
        assert_eq!(cached.data(), expected.data());
    }
    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);

    // What: the borrowed-operation accumulating entry point matches the typed
    // facade without adding another compiled plan or structure.
    cached.data_mut().fill(0.0);
    context
        .tree_transform_dyn_into_ref(
            &rule,
            &operation,
            &dst_structure,
            &src_structure,
            cached.data_mut(),
            src.data(),
            2.0,
            0.0,
        )
        .unwrap();
    assert_eq!(cached.data(), expected.data());
    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
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
        |_| -> Result<(FusionTreePairKey, f64), OperationError> {
            unreachable!("GenericFusion must be rejected before transforming keys")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
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
        |_| -> Result<(FusionTreePairKey, f64), OperationError> {
            unreachable!("permutation must reject non-symmetric braiding before key transform")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedBraidingStyle {
            operation: Box::new(operation),
            style: BraidingStyleKind::Anyonic,
        }
    );
}

#[test]
fn unique_tree_transform_plan_builder_defers_explicit_no_braiding_to_crossing_logic() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [1],
            1,
            [false, false],
            [false],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let src_tree = expect_tree_key(&src_key);
    let src_structure = packed_fixture_structure(3, [(src_key.clone(), vec![1, 1, 1])]).unwrap();

    let plan = build_unique_tree_transform_group_plan(
        &UniquePlanarRule,
        TreeTransformOperation::braid([1, 0], [2], [1, 0], [0]),
        &src_structure,
        |src| Ok((src.clone(), 1.0_f64)),
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].group_key(), &src_tree.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[test]
fn unique_all_codomain_braid_plan_builder_lowers_codomain_single_tree() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], 0, [false, true], [], [1]);
    let expected_dst_key = all_codomain_fusion_tree_test_key([1, 1], 0, [true, false], [], [1]);
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
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[-2.0]);
}

#[test]
fn unique_all_codomain_permute_plan_builder_lowers_symmetric_permutation() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], 0, [false, true], [], [1]);
    let expected_dst_key = all_codomain_fusion_tree_test_key([1, 1], 0, [true, false], [], [1]);
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[test]
fn unique_all_codomain_context_bypasses_plan_and_structure_caches() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], 0, [false, true], [], [1]);
    let dst_key = all_codomain_fusion_tree_test_key([1, 1], 0, [true, false], [], [1]);
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let dst_structure = packed_fixture_structure(2, [(dst_key, vec![1, 1])]).unwrap();
    let src_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![3.0], src_space, src_structure)
        .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0], dst_space, dst_structure)
            .unwrap();
    let operation = TreeTransformOperation::permute([1, 0], Vec::<usize>::new());
    let mut context =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    context
        .all_codomain_tree_transform_into(
            &Z2FusionRule,
            operation.clone(),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
    assert_eq!(dst.data(), &[3.0]);
    assert_eq!(context.cache().plan_len(), 0);
    assert_eq!(context.cache().structure_len(), 0);

    dst.data_mut().fill(0.0);
    context
        .all_codomain_tree_transform_into(&Z2FusionRule, operation, &mut dst, &src, 1.0, 0.0)
        .unwrap();
    assert_eq!(dst.data(), &[3.0]);
    assert_eq!(context.cache().plan_len(), 0);
    assert_eq!(context.cache().structure_len(), 0);
}

#[test]
fn unique_all_codomain_plan_builder_rejects_domain_operation_scope() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], 0, [false, false], [], [1]);
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
            operation: Box::new(operation),
            message: "all-codomain UniqueFusion lowering requires an empty domain operation",
        }
    );
}

#[test]
fn unique_all_codomain_plan_builder_accepts_explicit_vacuum_empty_domain() {
    let src_key = BlockKey::from(FusionTreePairKey::pair(
        FusionTreeKey::try_from_sector_ids_for_rule(
            &UniqueZ2Rule,
            [1, 1],
            0,
            [false, false],
            [],
            [1],
        )
        .unwrap(),
        empty_fusion_tree_with_coupled(0),
    ));
    let expected_dst_key = BlockKey::from(FusionTreePairKey::pair(
        FusionTreeKey::try_from_sector_ids_for_rule(
            &UniqueZ2Rule,
            [1, 1],
            0,
            [false, false],
            [],
            [1],
        )
        .unwrap(),
        empty_fusion_tree_with_coupled(0),
    ));
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[test]
fn unique_all_codomain_plan_builder_rejects_explicit_nonvacuum_empty_domain() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [],
            1,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap_err();

    // What: the current all-codomain facade diagnoses its operation scope
    // before the core categorical boundary admits the raw source.
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
    let src_key = all_codomain_fusion_tree_test_key([1, 1], 0, [false, false], [], [1]);
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
            operation: Box::new(operation),
            style: BraidingStyleKind::Anyonic,
        }
    );
}

#[test]
fn unique_tree_pair_plan_builder_lowers_domain_only_permutation() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1],
            1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        )
        .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1, 0],
            1,
            [false],
            [true, false],
            [],
            [],
            [],
            [1],
        )
        .unwrap(),
    );
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
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_codomain_domain_crossing_braid() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap(),
    );
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_tree_pair_transform_group_plan(
        &UniqueAnyonicRule,
        TreeTransformOperation::braid([1], [0], [0], [1]),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[-2.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_cyclic_transpose() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap(),
    );
    let expected_dst_key = src_key.clone();
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();
    let operation = TreeTransformOperation::transpose([1], [0]);

    let plan =
        build_unique_tree_pair_transform_group_plan(&UniqueZ2Rule, operation, &src_structure)
            .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_rank_four_cyclic_transpose() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [1, 0],
            1,
            [false, false],
            [false, false],
            [],
            [],
            [1],
            [1],
        )
        .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [0, 0],
            0,
            [true, false],
            [false, true],
            [],
            [],
            [1],
            [1],
        )
        .unwrap(),
    );
    let src_structure = packed_fixture_structure(4, [(src_key.clone(), vec![1, 1, 1, 1])]).unwrap();
    let operation = TreeTransformOperation::transpose([2, 0], [3, 1]);

    let plan =
        build_unique_tree_pair_transform_group_plan(&UniqueZ2Rule, operation, &src_structure)
            .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[derive(Clone, Copy, Debug)]
struct TensorKitZ4Element2Rule;

impl TensorKitZ4Element2Rule {
    fn label(sector: SectorId) -> usize {
        assert!(sector.id() < 4, "Z4 fixture sector must be in 0..4");
        sector.id()
    }

    fn cispi(exponent: f64) -> Complex64 {
        Complex64::from_polar(1.0, std::f64::consts::PI * exponent)
    }

    fn cocycle(left: SectorId, middle: SectorId, right: SectorId) -> Complex64 {
        let left = Self::label(left);
        let middle = Self::label(middle);
        let right = Self::label(right);
        let wrapped_sum = (middle + right) % 4;
        let carry = middle + right - wrapped_sum;
        Self::cispi((4 * left * carry) as f64 / 16.0)
    }
}

impl FusionRule for TensorKitZ4Element2Rule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        tenet_core::RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        SectorId::new((4 - Self::label(sector)) % 4)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        vec![SectorId::new((Self::label(left) + Self::label(right)) % 4)].into()
    }
}

impl MultiplicityFreeFusionRule for TensorKitZ4Element2Rule {}

impl MultiplicityFreeFusionSymbols for TensorKitZ4Element2Rule {
    type Scalar = Complex64;

    fn scalar_one(&self) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value.conj()
    }

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        let expected_left = SectorId::new((Self::label(left) + Self::label(middle)) % 4);
        let expected_right = SectorId::new((Self::label(middle) + Self::label(right)) % 4);
        let expected_coupled = SectorId::new((Self::label(expected_left) + Self::label(right)) % 4);
        if left_coupled == expected_left
            && right_coupled == expected_right
            && coupled == expected_coupled
        {
            Self::cocycle(left, middle, right)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let expected = SectorId::new((Self::label(left) + Self::label(right)) % 4);
        if coupled == expected {
            Self::cispi((Self::label(left) * Self::label(right)) as f64 / 4.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }
}

impl MultiplicityFreeRigidSymbols for TensorKitZ4Element2Rule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        let label = Self::label(sector);
        Self::cispi((label * label) as f64 / 4.0)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        Self::cocycle(sector, self.dual(sector), sector)
    }
}

fn tensor_kit_z4_rank_three_pair() -> FusionTreePairKey {
    let rule = TensorKitZ4Element2Rule;
    FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(1), SectorId::new(2), SectorId::new(3)],
            SectorId::new(2),
            [false, false, false],
            [SectorId::new(3)],
            [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(2)],
            SectorId::new(2),
            [false],
            [],
            [],
        )
        .unwrap(),
    )
}

fn assert_complex_oracle(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1.0e-14,
        "actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn unique_production_domain_fermion_crossing_matches_tensorkit_oracle() {
    let rule = FermionParityFusionRule;
    let odd = SectorId::new(1);
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [false], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [true], [], []).unwrap(),
    );
    let source_structure =
        packed_fixture_structure(2, [(BlockKey::from(source.clone()), vec![1, 1])]).unwrap();

    let plan = build_unique_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::braid([1], [0], [0], [1]),
        &source_structure,
    )
    .unwrap();

    // What: moving one odd domain leg across one odd codomain leg carries the
    // exact TensorKit fermionic sign, rather than a self-consistency result.
    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].dst_keys(), &[source]);
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[-1.0]);
}

#[test]
fn unique_production_complex_artin_and_inverse_match_tensorkit_oracle() {
    let rule = TensorKitZ4Element2Rule;
    let source = tensor_kit_z4_rank_three_pair();
    let source_structure =
        packed_fixture_structure(4, [(BlockKey::from(source.clone()), vec![1; 4])]).unwrap();
    let expected = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(1), SectorId::new(3), SectorId::new(2)],
            SectorId::new(2),
            [false, false, false],
            [SectorId::new(0)],
            [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
        )
        .unwrap(),
        source.domain_tree().clone(),
    );

    for (codomain_levels, expected_coefficient) in [
        ([0, 1, 2], Complex64::new(0.0, -1.0)),
        ([0, 2, 1], Complex64::new(0.0, 1.0)),
    ] {
        let plan = build_unique_tree_pair_transform_group_plan(
            &rule,
            TreeTransformOperation::braid([0, 2, 1], [3], codomain_levels, [3]),
            &source_structure,
        )
        .unwrap();

        // What: Z4Element{2}'s later Artin crossing preserves the recoupled
        // innerline and conjugates the complex phase when the levels reverse.
        assert_eq!(plan.specs().len(), 1);
        assert_eq!(plan.specs()[0].dst_keys(), std::slice::from_ref(&expected));
        assert_complex_oracle(
            plan.specs()[0].recoupling_coefficients_dst_src()[0],
            expected_coefficient,
        );
    }
}

#[test]
fn unique_production_pivotal_transpose_matches_tensorkit_oracle() {
    let rule = TensorKitZ4Element2Rule;
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(1), SectorId::new(2)],
            SectorId::new(3),
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(3)],
            SectorId::new(3),
            [true],
            [],
            [],
        )
        .unwrap(),
    );
    let expected = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(2), SectorId::new(1)],
            SectorId::new(3),
            [true, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(3)],
            SectorId::new(3),
            [true],
            [],
            [],
        )
        .unwrap(),
    );
    let source_structure =
        packed_fixture_structure(3, [(BlockKey::from(source), vec![1; 3])]).unwrap();

    let plan = build_unique_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::transpose([1, 2], [0]),
        &source_structure,
    )
    .unwrap();

    // What: the nontrivial Z4 Frobenius-Schur/A-symbol phase survives the
    // cyclic transpose with TensorKit's exact destination dual flags.
    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected]);
    assert_complex_oracle(
        plan.specs()[0].recoupling_coefficients_dst_src()[0],
        Complex64::new(-1.0, 0.0),
    );
}

#[test]
fn tensorkit_unique_oracle_provenance_is_pinned() {
    const RAW: &str = include_str!("fixtures/issue306_tensorkit_unique_oracle.txt");

    // What: independent expected values retain the exact reference revision
    // and raw outputs needed to regenerate or audit this fixture.
    assert!(RAW.contains("TensorKit_commit=cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91"));
    assert!(RAW.contains("pair3.coeff=-2.2204460492503131e-16-1im"));
    assert!(RAW.contains("pair3.coeff=-2.2204460492503131e-16+1im"));
    assert!(RAW.contains("transpose.coeff=-1+0im"));
    assert!(RAW.contains("fz2.domain_crossing.coeff=-1+0im"));
}

fn unique_rank_three_tree_pair<R>(rule: &R, left: SectorId, right: SectorId) -> FusionTreePairKey
where
    R: FusionRule,
{
    let coupled = rule.fusion_channels(left, right)[0];
    FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            rule,
            [left, right],
            coupled,
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(rule, [coupled], coupled, [true], [], []).unwrap(),
    )
}

fn assert_unique_and_generic_plan_are_identical<R>(
    rule: &R,
    source: FusionTreePairKey,
    operation: TreeTransformOperation,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let rank = source.codomain_tree().uncoupled().len() + source.domain_tree().uncoupled().len();
    let source_structure =
        packed_fixture_structure(rank, [(BlockKey::from(source), vec![1; rank])]).unwrap();
    let specialized =
        build_unique_tree_pair_transform_group_plan(rule, operation.clone(), &source_structure)
            .unwrap();
    let generic = build_multiplicity_free_tree_pair_transform_group_plan(
        rule,
        operation.clone(),
        &source_structure,
    )
    .unwrap();
    let standard =
        build_tree_pair_transform_group_plan(rule, operation, &source_structure).unwrap();

    // What: the production Unique lowering is the exact one-term form of the
    // explicit multiplicity-free algorithm, including key and coefficient
    // ordering, and is the branch selected by the standard builder.
    assert_eq!(specialized, generic);
    assert_eq!(standard, specialized);
    assert_eq!(specialized.specs().len(), 1);
    assert_eq!(specialized.specs()[0].src_keys().len(), 1);
    assert_eq!(specialized.specs()[0].dst_keys().len(), 1);

    let destination_structure = packed_fixture_structure(
        rank,
        [(specialized.specs()[0].dst_keys()[0].clone(), vec![1; rank])],
    )
    .unwrap();
    let specialized_replay = specialized
        .compile_structures(&destination_structure, &source_structure)
        .unwrap();
    let generic_replay = generic
        .compile_structures(&destination_structure, &source_structure)
        .unwrap();

    // What: specialized and explicit generic plans compile to identical raw
    // replay blocks, layouts, coefficients, schedules, and structure guards.
    assert_eq!(specialized_replay, generic_replay);
}

#[test]
fn unique_production_lowering_matches_generic_across_pointed_rules_and_operations() {
    let u1_source = unique_rank_three_tree_pair(
        &U1FusionRule,
        U1Irrep::new(1).sector_id(),
        U1Irrep::new(-2).sector_id(),
    );
    assert_unique_and_generic_plan_are_identical(
        &U1FusionRule,
        u1_source,
        TreeTransformOperation::permute([1, 0], [2]),
    );

    let z2_source = unique_rank_three_tree_pair(&Z2FusionRule, SectorId::new(1), SectorId::new(0));
    assert_unique_and_generic_plan_are_identical(
        &Z2FusionRule,
        z2_source,
        TreeTransformOperation::braid([2, 0], [1], [0, 2], [1]),
    );

    let fz2_source =
        unique_rank_three_tree_pair(&FermionParityFusionRule, SectorId::new(1), SectorId::new(1));
    assert_unique_and_generic_plan_are_identical(
        &FermionParityFusionRule,
        fz2_source,
        TreeTransformOperation::transpose([2, 0], [1]),
    );

    let product = FpU1Rule::default();
    let odd_charge = product.encode_sector(SectorId::new(1), U1Irrep::new(2).sector_id());
    let even_charge = product.encode_sector(SectorId::new(0), U1Irrep::new(-1).sector_id());
    let coupled = product.fusion_channels(odd_charge, even_charge)[0];
    let product_source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &product,
            [odd_charge, even_charge],
            coupled,
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &product,
            [product.vacuum(), coupled],
            coupled,
            [true, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
    );
    assert_unique_and_generic_plan_are_identical(
        &product,
        product_source,
        TreeTransformOperation::braid([0, 1, 3], [2], [0, 1], [2, 3]),
    );

    let anyonic_source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &UniqueAnyonicRule,
            [SectorId::new(1)],
            SectorId::new(1),
            [false],
            [],
            [],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &UniqueAnyonicRule,
            [SectorId::new(1)],
            SectorId::new(1),
            [true],
            [],
            [],
        )
        .unwrap(),
    );
    assert_unique_and_generic_plan_are_identical(
        &UniqueAnyonicRule,
        anyonic_source,
        TreeTransformOperation::braid([1], [0], [0], [1]),
    );

    let asymmetric_source = unique_rank_three_tree_pair(
        &AsymmetricAnyonicPointedRule,
        SectorId::new(1),
        SectorId::new(2),
    );
    assert_unique_and_generic_plan_are_identical(
        &AsymmetricAnyonicPointedRule,
        asymmetric_source.clone(),
        TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]),
    );
    assert_unique_and_generic_plan_are_identical(
        &AsymmetricAnyonicPointedRule,
        asymmetric_source,
        TreeTransformOperation::transpose([2, 0], [1]),
    );
}

#[test]
fn unique_production_lowering_matches_generic_at_rank_zero_and_one() {
    let scalar = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&U1FusionRule, [], U1FusionRule.vacuum(), [], [], [])
            .unwrap(),
        FusionTreeKey::try_new_for_rule(&U1FusionRule, [], U1FusionRule.vacuum(), [], [], [])
            .unwrap(),
    );
    assert_unique_and_generic_plan_are_identical(
        &U1FusionRule,
        scalar,
        TreeTransformOperation::permute([], []),
    );

    let rank_one = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            [Z2FusionRule.vacuum()],
            Z2FusionRule.vacuum(),
            [true],
            [],
            [],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&Z2FusionRule, [], Z2FusionRule.vacuum(), [], [], [])
            .unwrap(),
    );
    assert_unique_and_generic_plan_are_identical(
        &Z2FusionRule,
        rank_one,
        TreeTransformOperation::transpose([0], []),
    );
}

#[test]
fn unique_all_codomain_production_lowering_matches_generic_replay_exactly() {
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            Z2FusionRule.vacuum(),
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&Z2FusionRule, [], Z2FusionRule.vacuum(), [], [], [])
            .unwrap(),
    );
    let source_structure =
        packed_fixture_structure(2, [(BlockKey::from(source), vec![1; 2])]).unwrap();
    let operation = TreeTransformOperation::permute([1, 0], []);
    let specialized = build_unique_all_codomain_tree_transform_group_plan(
        &Z2FusionRule,
        operation.clone(),
        &source_structure,
    )
    .unwrap();
    let generic = build_multiplicity_free_all_codomain_tree_transform_group_plan(
        &Z2FusionRule,
        operation.clone(),
        &source_structure,
    )
    .unwrap();
    let standard =
        build_all_codomain_tree_transform_group_plan(&Z2FusionRule, operation, &source_structure)
            .unwrap();

    // What: all-codomain dispatch lowers the same key, phase, and raw replay
    // descriptor as the explicit multiplicity-free algorithm.
    assert_eq!(specialized, generic);
    assert_eq!(standard, specialized);
    let destination_structure = packed_fixture_structure(
        2,
        [(specialized.specs()[0].dst_keys()[0].clone(), vec![1; 2])],
    )
    .unwrap();
    assert_eq!(
        specialized
            .compile_structures(&destination_structure, &source_structure)
            .unwrap(),
        generic
            .compile_structures(&destination_structure, &source_structure)
            .unwrap()
    );
}

fn assert_unique_and_generic_error_are_identical<R>(
    rule: &R,
    source_structure: &BlockStructure,
    operation: TreeTransformOperation,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let specialized =
        build_unique_tree_pair_transform_group_plan(rule, operation.clone(), source_structure)
            .unwrap_err();
    let generic =
        build_multiplicity_free_tree_pair_transform_group_plan(rule, operation, source_structure)
            .unwrap_err();
    assert_eq!(specialized, generic);
}

#[test]
fn unique_production_lowering_preserves_generic_error_precedence() {
    let source = unique_rank_three_tree_pair(&Z2FusionRule, SectorId::new(1), SectorId::new(0));
    let source_structure =
        packed_fixture_structure(3, [(BlockKey::from(source), vec![1; 3])]).unwrap();

    // What: malformed permutations, braid levels, and noncyclic transposes
    // fail with the same error and precedence as the explicit generic path.
    assert_unique_and_generic_error_are_identical(
        &Z2FusionRule,
        &source_structure,
        TreeTransformOperation::permute([0, 0], [2]),
    );
    assert_unique_and_generic_error_are_identical(
        &Z2FusionRule,
        &source_structure,
        TreeTransformOperation::braid([1, 0], [2], [0], [2]),
    );
    assert_unique_and_generic_error_are_identical(
        &Z2FusionRule,
        &source_structure,
        TreeTransformOperation::transpose([1, 0], [2]),
    );

    let planar_source =
        unique_rank_three_tree_pair(&UniquePlanarRule, SectorId::new(1), SectorId::new(0));
    let planar_structure =
        packed_fixture_structure(3, [(BlockKey::from(planar_source), vec![1; 3])]).unwrap();
    assert_unique_and_generic_error_are_identical(
        &UniquePlanarRule,
        &planar_structure,
        TreeTransformOperation::permute([1, 0], [2]),
    );
}

fn z2_two_leg_pair_with_empty_domain(uncoupled: [SectorId; 2]) -> FusionTreePairKey {
    let rule = Z2FusionRule;
    let coupled = rule.fusion_channels(uncoupled[0], uncoupled[1])[0];
    FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            uncoupled,
            coupled,
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [], rule.vacuum(), [], [], []).unwrap(),
    )
}

#[test]
fn unique_production_valid_groups_are_singleton_and_preserve_source_order() {
    let rule = Z2FusionRule;
    let odd = SectorId::new(1);
    let group_a = z2_two_leg_pair_with_empty_domain([odd, odd]);
    let group_b = z2_two_leg_pair_with_empty_domain([SectorId::new(0), SectorId::new(0)]);
    let split_group = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [false], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [false], [], []).unwrap(),
    );
    let source_order = [group_a, group_b, split_group];
    let source_structure =
        packed_fixture_structure(2, source_order.iter().cloned().map(|key| (key, vec![1, 1])))
            .unwrap();
    // What: canonical UniqueFusion admits exactly one tree for each external
    // group; the former interleaved same-group fixture relied on the removed
    // rank-zero coupled alias.
    assert!(source_structure
        .fusion_tree_groups()
        .iter()
        .all(|group| group.block_indices().len() == 1));

    let plan = build_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::permute([0], [1]),
        &source_structure,
    )
    .unwrap();

    // What: the Unique/Abelian transformer emits one spec per canonical source
    // block in raw source order.
    assert_eq!(
        plan.specs()
            .iter()
            .map(|spec| spec.src_keys()[0].clone())
            .collect::<Vec<_>>(),
        source_order
    );
}

#[test]
fn unique_production_prepares_each_distinct_source_split() {
    let rule = Z2FusionRule;
    let vacuum = rule.vacuum();
    let odd = SectorId::new(1);
    let split_one_one = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [false], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [false], [], []).unwrap(),
    );
    let split_two_zero = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            [odd, odd],
            vacuum,
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [], vacuum, [], [], []).unwrap(),
    );
    let source_order = [split_one_one, split_two_zero];
    let source_structure =
        packed_fixture_structure(2, source_order.iter().cloned().map(|key| (key, vec![1, 1])))
            .unwrap();
    let operation = TreeTransformOperation::permute([0], [1]);

    let specialized =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), &source_structure).unwrap();
    let generic =
        build_multiplicity_free_tree_pair_transform_group_plan(&rule, operation, &source_structure)
            .unwrap();

    // What: a rank-homogeneous structure may still contain distinct
    // codomain/domain splits; each split keeps the generic destination,
    // coefficient, and raw source order under the prepared Unique path.
    assert_eq!(specialized, generic);
    assert_eq!(
        specialized
            .specs()
            .iter()
            .map(|spec| spec.src_keys()[0].clone())
            .collect::<Vec<_>>(),
        source_order
    );
}

#[test]
fn unique_production_preserves_generic_noncategorical_error_precedence() {
    let dense = BlockStructure::trivial(&[1, 1]).unwrap();
    let opaque = packed_fixture_structure(2, [(BlockKey::opaque([7]), vec![1, 1])]).unwrap();
    let malformed = TreeTransformOperation::permute([0, 0], []);

    for structure in [&dense, &opaque] {
        // What: Unique and general multiplicity-free builders both report
        // malformed syntax before either noncategorical namespace.
        assert_eq!(
            build_tree_pair_transform_group_plan(&Z2FusionRule, malformed.clone(), structure,),
            build_multiplicity_free_tree_pair_transform_group_plan(
                &Z2FusionRule,
                malformed.clone(),
                structure,
            ),
        );
    }
}

#[test]
fn explicit_keyed_replay_accepts_dense_and_opaque_namespaces() {
    for key in [BlockKey::Dense, BlockKey::opaque([7, 11])] {
        let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
            key.clone(),
            vec![1],
            0,
        )
        .unwrap()])
        .unwrap();
        let space = TensorMapSpace::<1, 0>::from_dims([1], []).unwrap();
        let src = TensorMap::<f64, 1, 0>::from_vec_with_structure(
            vec![3.0],
            space.clone(),
            structure.clone(),
        )
        .unwrap();
        let mut dst =
            TensorMap::<f64, 1, 0>::from_vec_with_structure(vec![0.0], space, structure).unwrap();
        let replay = TreeTransformStructure::compile_keyed(
            &dst,
            &src,
            &[TreeTransformKeyBlockSpec::single(key.clone(), key, 2.0)],
        )
        .unwrap();
        let mut backend = HostTensorOperations;
        let mut workspace = TreeTransformWorkspace::default();

        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &replay,
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();

        // What: explicit application-key replay remains namespace-neutral; only
        // categorical tree-transform planning requires FusionTreePairKey.
        assert_eq!(dst.data(), &[6.0]);
    }
}

#[test]
fn tree_transform_compile_grouped_lowers_to_replay_ready_structure() {
    let key10 = grouped_su2_test_pair(0);
    let key20 = grouped_su2_test_pair(2);
    let key100 = grouped_su2_test_pair(0);
    let key200 = grouped_su2_test_pair(2);
    let key300 = grouped_su2_test_pair(4);
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
        &[TreeTransformGroupBlockSpec::try_multi(
            [key10, key20],
            [key100, key200, key300],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        )
        .unwrap()],
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

#[test]
fn keyed_and_grouped_compile_resolve_every_key_before_structural_validation() {
    let present = grouped_su2_test_pair(0);
    let missing = grouped_su2_test_pair(2);
    let dst_structure = packed_fixture_structure(2, [(present.clone(), vec![1, 1])]).unwrap();
    let src_structure = packed_fixture_structure(1, [(present.clone(), vec![1])]).unwrap();
    let structurally_invalid =
        TreeTransformGroupBlockSpec::try_multi([present.clone()], [present.clone()], vec![1.0_f64])
            .unwrap();
    let missing_later = TreeTransformGroupBlockSpec::single(missing.clone(), present.clone(), 1.0);

    let err = TreeTransformStructure::compile_grouped_structures(
        &dst_structure,
        &src_structure,
        &[structurally_invalid.clone(), missing_later],
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: Box::new(BlockKey::from(missing))
        }
    );

    let err = TreeTransformStructure::compile_grouped_structures(
        &dst_structure,
        &src_structure,
        &[structurally_invalid],
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::StructureRankMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let coefficient_mismatch =
        TreeTransformKeyBlockSpec::multi([present.clone()], [present.clone()], Vec::<f64>::new());
    let missing_later = TreeTransformKeyBlockSpec::single(BlockKey::opaque([2]), present, 1.0);
    let err = TreeTransformStructure::compile_keyed_structures(
        &dst_structure,
        &src_structure,
        &[coefficient_mismatch, missing_later],
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: Box::new(BlockKey::opaque([2]))
        }
    );
}

#[test]
fn grouped_storage_mapping_preserves_callback_error_order() {
    let present = grouped_su2_test_pair(0);
    let missing = grouped_su2_test_pair(2);
    let structure = Arc::new(packed_fixture_structure(1, [(present.clone(), vec![1])]).unwrap());
    let plan = TreeTransformGroupPlan::new(vec![
        TreeTransformGroupBlockSpec::single(present.clone(), present.clone(), 1.0_f64),
        TreeTransformGroupBlockSpec::single(missing, present, 1.0),
    ]);
    let axis_called = std::cell::Cell::new(false);

    let err = plan
        .compile_shared_structures_with_storage_mapping(
            Arc::clone(&structure),
            &structure,
            Arc::clone(&structure),
            |_| Err(OperationError::ElementCountOverflow),
            |_| {
                axis_called.set(true);
                Ok(0)
            },
            false,
        )
        .unwrap_err();

    assert_eq!(err, OperationError::ElementCountOverflow);
    assert!(!axis_called.get());
}

#[test]
fn grouped_storage_mapping_owns_coefficients_and_matches_direct_complex_replay() {
    let dst0 = grouped_su2_test_pair(0);
    let dst1 = grouped_su2_test_pair(2);
    let dst2 = grouped_su2_test_pair(4);
    let src0 = grouped_su2_test_pair(0);
    let src1 = grouped_su2_test_pair(2);
    let src2 = grouped_su2_test_pair(4);
    let dst_structure = Arc::new(
        packed_fixture_structure(
            2,
            [
                (dst0.clone(), vec![1, 2]),
                (dst1.clone(), vec![1, 2]),
                (dst2.clone(), vec![1, 2]),
            ],
        )
        .unwrap(),
    );
    let logical_src_structure = packed_fixture_structure(
        2,
        [
            (src0.clone(), vec![2, 1]),
            (src1.clone(), vec![2, 1]),
            (src2.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let storage_src_structure = Arc::new(
        packed_fixture_structure(
            2,
            [
                (src1.clone(), vec![1, 2]),
                (src2.clone(), vec![1, 2]),
                (src0.clone(), vec![1, 2]),
            ],
        )
        .unwrap(),
    );
    let coefficients = vec![
        Complex64::new(0.5, -2.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(-3.0, 0.5),
        Complex64::new(4.0, 2.0),
    ];
    let callback_trace = std::cell::RefCell::new(Vec::new());
    let grouped = {
        let plan = TreeTransformGroupPlan::new(vec![
            TreeTransformGroupBlockSpec::single(dst0, src0, coefficients[0])
                .with_source_axes([1, 0]),
            TreeTransformGroupBlockSpec::try_multi(
                [dst1, dst2],
                [src1, src2],
                coefficients[1..].to_vec(),
            )
            .unwrap()
            .with_source_axes([1, 0]),
        ]);
        plan.compile_shared_structures_with_storage_mapping(
            Arc::clone(&dst_structure),
            &logical_src_structure,
            Arc::clone(&storage_src_structure),
            |block| {
                callback_trace.borrow_mut().push(("block", block));
                Ok(match block {
                    0 => 2,
                    1 => 0,
                    2 => 1,
                    _ => unreachable!("logical source block is resolved from the structure"),
                })
            },
            |axis| {
                callback_trace.borrow_mut().push(("axis", axis));
                Ok(1 - axis)
            },
            true,
        )
        .unwrap()
    };
    let direct_specs = [
        TreeTransformBlockSpec::single(0, 2, coefficients[0]).with_source_axes([0, 1]),
        TreeTransformBlockSpec::multi(vec![1, 2], vec![0, 1], coefficients[1..].to_vec())
            .with_source_axes([0, 1]),
    ];
    let direct = TreeTransformStructure::compile_structures_with_storage_conjugation(
        &dst_structure,
        &storage_src_structure,
        &direct_specs,
        true,
    )
    .unwrap();

    assert_eq!(
        callback_trace.into_inner(),
        [
            ("block", 0),
            ("axis", 1),
            ("axis", 0),
            ("block", 1),
            ("block", 2),
            ("axis", 1),
            ("axis", 0),
        ]
    );
    assert_eq!(grouped, direct);
    assert_eq!(
        grouped.recoupling_coefficients_dst_src(),
        coefficients.as_slice()
    );

    let src_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src = TensorMap::<Complex64, 2, 0>::from_vec_with_structure(
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
            Complex64::new(9.0, 10.0),
            Complex64::new(11.0, 12.0),
        ],
        src_space,
        storage_src_structure.as_ref().clone(),
    )
    .unwrap();
    let mut grouped_dst = TensorMap::<Complex64, 2, 0>::from_vec_with_structure(
        vec![Complex64::new(0.0, 0.0); 6],
        dst_space.clone(),
        dst_structure.as_ref().clone(),
    )
    .unwrap();
    let mut direct_dst = TensorMap::<Complex64, 2, 0>::from_vec_with_structure(
        vec![Complex64::new(0.0, 0.0); 6],
        dst_space,
        dst_structure.as_ref().clone(),
    )
    .unwrap();
    let mut grouped_backend = HostTensorOperations;
    let mut grouped_workspace = TreeTransformWorkspace::default();
    let mut direct_backend = HostTensorOperations;
    let mut direct_workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut grouped_backend,
        &mut grouped_workspace,
        &grouped,
        &mut grouped_dst,
        &src,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    tree_transform_execute_with(
        &mut direct_backend,
        &mut direct_workspace,
        &direct,
        &mut direct_dst,
        &src,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(grouped_dst.data(), direct_dst.data());
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
    let key10 = grouped_su2_test_pair(0);
    let key20 = grouped_su2_test_pair(2);
    let key100 = grouped_su2_test_pair(0);
    let key200 = grouped_su2_test_pair(2);
    let key300 = grouped_su2_test_pair(4);
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
        &[TreeTransformGroupBlockSpec::try_multi(
            [key10, key20],
            [key100, key200, key300],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        )
        .unwrap()],
    )
    .unwrap();
    let mut workspace = crate::storage_scratch::StorageTreeTransformWorkspace::<
        TrackingScratch<f64>,
        TrackingScratch<f64>,
    >::default();

    tenet_operations::tree_transform_structure_with_storage_workspace_strided_kernel(
        &mut crate::StridedHostKernelAdapter::default(),
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
        &mut crate::StridedHostKernelAdapter::default(),
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
    let present_key = grouped_su2_test_pair(0);
    let missing_key = grouped_su2_test_pair(2);
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
            missing_key.clone(),
            present_key,
            1.0,
        )],
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: Box::new(BlockKey::from(missing_key))
        }
    );
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
    assert_eq!(
        spec.recoupling_coefficients_dst_src(),
        &[1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn grouped_spec_from_public_groups_rejects_repeated_block_indices() {
    let key = grouped_su2_test_pair(0);
    let structure = packed_fixture_structure(2, [(key, vec![1, 1])]).unwrap();
    let group = structure.fusion_tree_groups().pop().unwrap();
    let repeated = tenet_core::FusionTreeBlockGroup::new(group.group_key().clone(), vec![0, 0]);

    let src_err = TreeTransformGroupBlockSpec::from_block_groups(
        &structure,
        &group,
        &structure,
        &repeated,
        vec![1.0_f64, 2.0],
    )
    .unwrap_err();
    let dst_err = TreeTransformGroupBlockSpec::from_block_groups(
        &structure,
        &repeated,
        &structure,
        &group,
        vec![1.0_f64, 2.0],
    )
    .unwrap_err();

    // What: a caller-built group cannot duplicate a matrix column or row by
    // repeating one otherwise valid BlockStructure index.
    assert_eq!(
        src_err,
        OperationError::DuplicateTreeTransformKey {
            tensor: "src",
            index: 1,
        }
    );
    assert_eq!(
        dst_err,
        OperationError::DuplicateTreeTransformKey {
            tensor: "dst",
            index: 1,
        }
    );
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
    let dst_key = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
    let src_key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let plan_a = TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::single(
        dst_key.clone(),
        src_key.clone(),
        2.0_f64,
    )]);
    let plan_b = TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::single(
        dst_key, src_key, 3.0_f64,
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
        vec![BlockSpec::with_key(key.clone().into(), vec![2, 3], vec![1, 2], 0).unwrap()],
    )
    .unwrap();
    let shape_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone().into(), vec![3, 2], vec![1, 3], 0).unwrap()],
    )
    .unwrap();
    let stride_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone().into(), vec![2, 3], vec![2, 1], 0).unwrap()],
    )
    .unwrap();
    let offset_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone().into(), vec![2, 3], vec![1, 2], 1).unwrap()],
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
        crate::StridedHostKernelAdapter::default().add_strided(
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
        crate::StridedHostKernelAdapter::default().axpby_strided(
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
        crate::StridedHostKernelAdapter::default().copy_scale_strided(
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
        crate::StridedHostKernelAdapter::default().scale_strided(
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
        recoupling_coefficients_dst_src: &[C],
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
        crate::StridedHostKernelAdapter::default().recoupling_src_times_u_transpose(
            destination,
            source,
            recoupling_coefficients_dst_src,
            coefficient_start,
            element_count,
            src_count,
            dst_count,
        )
    }
}

#[test]
fn tree_transform_replay_dispatches_through_kernel_adapter() {
    let key10 = grouped_su2_test_pair(0);
    let key20 = grouped_su2_test_pair(2);
    let key100 = grouped_su2_test_pair(0);
    let key200 = grouped_su2_test_pair(2);
    let key300 = grouped_su2_test_pair(4);
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
        &[TreeTransformGroupBlockSpec::try_multi(
            [key10, key20],
            [key100, key200, key300],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        )
        .unwrap()],
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

// ======================================================================
// Stage B2c: Generic-fusion (outer-multiplicity) plan compile.
// A minimal Generic rule with one self-dual outer-multiplicity sector (1):
// N(1,1,1)=2 makes the rule genuinely Generic, while the tree pairs used
// below couple [1,1] to the vacuum (N(1,1,0)=1), so the rank-2 codomain
// braid is a pure 1×1 R-symbol — no bends, no F-moves. This exercises the
// `build_generic_tree_pair_transform_group_plan` wiring (style guard, group
// iteration, shared assembly, core-row dispatch) without a full SU(3) symbol
// table; the recoupling math itself is proven in tenet-core's B2c tests.
// ======================================================================
use tenet_core::{
    generic_braid_tree_pair, generic_permute_tree_pair, generic_transpose_tree_pair, GenericFArray,
    GenericFusionSymbols, GenericRMatrix, GenericRigidSymbols,
};

#[derive(Clone, Copy)]
struct ToyGenericRule {
    style: FusionStyleKind,
}

impl FusionRule for ToyGenericRule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        tenet_core::RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.style
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        sector // 0 and 1 self-dual
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => [SectorId::new(x)].into_iter().collect(),
            (1, 1) => [SectorId::new(0), SectorId::new(1)].into_iter().collect(),
            _ => [SectorId::new(0)].into_iter().collect(),
        }
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl GenericFusionSymbols for ToyGenericRule {
    type Scalar = f64;
    fn f_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
        _d: SectorId,
        _e: SectorId,
        _f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        // Unreached for the rank-2 vacuum-coupled pair (pure R braid).
        GenericFArray::new(vec![1.0], (1, 1, 1, 1))
    }
    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        if c == SectorId::new(1) {
            GenericRMatrix::new(vec![0.0, 2.0, 3.0, 0.0], 2, 2)
        } else {
            GenericRMatrix::new(vec![1.0], 1, 1)
        }
    }
}

impl GenericRigidSymbols for ToyGenericRule {
    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

fn b2c_toy_src_pair() -> FusionTreePairKey {
    // cod [1,1]->0 (vacuum-coupled, N(1,1,0)=1, single vertex label 1), dom []->0.
    let cod = FusionTreeKey::try_new_for_rule(
        &ToyGenericRule {
            style: FusionStyleKind::Generic,
        },
        [SectorId::new(1), SectorId::new(1)],
        SectorId::new(0),
        [false, false],
        [],
        [MultiplicityIndex::ONE],
    )
    .unwrap();
    let dom = FusionTreeKey::try_new_for_rule(
        &ToyGenericRule {
            style: FusionStyleKind::Generic,
        },
        [],
        SectorId::new(0),
        [],
        [],
        [],
    )
    .unwrap();
    FusionTreePairKey::pair(cod, dom)
}

// The generic plan compile reproduces the core `generic_permute_tree_pair`
// rows exactly (the assembly adds no math), and its runtime style gate rejects
// a rule that reports a multiplicity-free style — the symmetric sibling of the
// mult-free builders' `UnsupportedFusionStyle` guards.
#[test]
fn build_generic_tree_pair_plan_matches_core_rows_and_guards_style() {
    let rule = ToyGenericRule {
        style: FusionStyleKind::Generic,
    };
    let src_pair = b2c_toy_src_pair();
    let src_key = BlockKey::from(src_pair.clone());
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    // The plan's single group spec must reproduce the per-source core rows for
    // each operation kind (exercises every `transformed_generic_tree_pair_rows`
    // arm). cod [1,1]->0 with an empty domain stays a pure 1×1 R braid, so the
    // codomain permute/braid and the cyclic transpose all resolve numerically.
    let assert_plan_matches =
        |operation: TreeTransformOperation, core_rows: Vec<(FusionTreePairKey, f64)>| {
            assert_eq!(core_rows.len(), 1);
            let (core_dst, core_coeff) = &core_rows[0];
            let plan =
                build_generic_tree_pair_transform_group_plan(&rule, operation, &src_structure)
                    .unwrap();
            assert_eq!(plan.specs().len(), 1);
            let spec = &plan.specs()[0];
            assert_eq!(spec.src_keys(), std::slice::from_ref(&src_pair));
            assert_eq!(spec.dst_keys(), std::slice::from_ref(core_dst));
            assert_eq!(spec.recoupling_coefficients_dst_src().len(), 1);
            assert!((spec.recoupling_coefficients_dst_src()[0] - core_coeff).abs() < 1e-12);
        };

    assert_plan_matches(
        TreeTransformOperation::permute([1, 0], []),
        generic_permute_tree_pair(&rule, &src_pair, &[1, 0], &[]).unwrap(),
    );
    assert_plan_matches(
        TreeTransformOperation::braid([1, 0], [], [0, 1], []),
        generic_braid_tree_pair(&rule, &src_pair, &[1, 0], &[], &[0, 1], &[]).unwrap(),
    );
    assert_plan_matches(
        TreeTransformOperation::braid([0, 1], [], [29, 7], []),
        vec![(src_pair.clone(), 1.0)],
    );
    assert_plan_matches(
        TreeTransformOperation::transpose([1, 0], []),
        generic_transpose_tree_pair(&rule, &src_pair, &[1, 0], &[]).unwrap(),
    );

    // Style guard: a multiplicity-free style is rejected before any compile.
    let mf = ToyGenericRule {
        style: FusionStyleKind::Simple,
    };
    let err = build_generic_tree_pair_transform_group_plan(
        &mf,
        TreeTransformOperation::permute([1, 0], []),
        &src_structure,
    )
    .unwrap_err();
    assert!(matches!(err, OperationError::UnsupportedFusionStyle { .. }));
}

#[test]
fn generic_multiplicity_monomial_rows_compile_and_execute_as_direct_singles() {
    // What: a GenericFusion R matrix whose core rows are structurally
    // singleton and destination-injective uses the same direct replay contract.
    let rule = ToyGenericRule {
        style: FusionStyleKind::Generic,
    };
    let pairs = [MultiplicityIndex::ONE, MultiplicityIndex::new(2).unwrap()].map(|vertex| {
        FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [SectorId::new(1), SectorId::new(1)],
                SectorId::new(1),
                [false, false],
                [],
                [vertex],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &rule,
                [SectorId::new(1)],
                SectorId::new(1),
                [false],
                [],
                [],
            )
            .unwrap(),
        )
    });
    let keys = pairs.clone().map(BlockKey::from);
    let structure =
        packed_fixture_structure(3, keys.iter().cloned().map(|key| (key, vec![1usize; 3])))
            .unwrap();
    let operation = TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]);
    let core_rows = pairs
        .iter()
        .map(|pair| generic_braid_tree_pair(&rule, pair, &[1, 0], &[2], &[0, 1], &[2]).unwrap())
        .collect::<Vec<_>>();
    assert!(core_rows.iter().all(|row| row.len() == 1));
    assert_ne!(core_rows[0][0].0, core_rows[1][0].0);

    let plan = build_generic_tree_pair_transform_group_plan(&rule, operation, &structure).unwrap();
    assert_eq!(plan.specs().len(), 2);
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[2.0]);
    assert_eq!(plan.specs()[1].recoupling_coefficients_dst_src(), &[3.0]);
    let compiled = plan.compile_structures(&structure, &structure).unwrap();
    assert!(!compiled.has_pack_gemm_scatter_blocks());

    let space = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
    let src = TensorMap::<f64, 2, 1>::from_vec_with_structure(
        vec![5.0, 7.0],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_structure(vec![11.0, 13.0], space, structure)
            .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[75.0, 59.0]);
    assert_eq!(
        (workspace.source_len(), workspace.destination_len()),
        (0, 0)
    );
}

// ======================================================================
// Stage B3a: Generic-fusion facade (TensorMap-level) siblings.
// ======================================================================

fn b3a_toy_tensormap(value: f64) -> (BlockStructure, TensorMap<f64, 2, 0>) {
    // cod [1,1]->0 with per-leg degeneracy 1 -> a single-element block.
    let src_key = BlockKey::from(b2c_toy_src_pair());
    let structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let tensor =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![value], space, structure.clone())
            .unwrap();
    (structure, tensor)
}

#[test]
fn direct_generic_tree_pair_rejects_malformed_destination_without_writing() {
    let rule = ToyGenericRule {
        style: FusionStyleKind::Generic,
    };
    let src_structure =
        packed_fixture_structure(2, [(BlockKey::from(b2c_toy_src_pair()), vec![1, 1])]).unwrap();
    let malformed = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            [],
            [],
        )
        .unwrap(),
    );
    let dst_structure = packed_fixture_structure(2, [(malformed, vec![2, 1])]).unwrap();
    assert_ne!(src_structure.content_id(), dst_structure.content_id());
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![3.0],
        TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap(),
        src_structure,
    )
    .unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![17.0, 19.0],
        TensorMapSpace::<2, 0>::from_dims([2, 1], []).unwrap(),
        dst_structure,
    )
    .unwrap();
    let before = dst.data().to_vec();

    let error = tree_transform_into_generic(
        &rule,
        TreeTransformOperation::permute([0, 1], []),
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap_err();

    // What: the noncached Generic facade validates caller-owned destination
    // structure before lowering or replay changes its data.
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of vertices",
        })
    );
    assert_eq!(dst.data(), before);
}

// Gate 1 (highest reachable level = TensorMap facade): each generic facade
// sibling reproduces the Stage B2c plan-level path
// (`build_generic_tree_pair_transform_group_plan` -> compile -> execute)
// byte-for-byte. Combined with B2c's `..._matches_core_rows_...` (plan == core
// tree rows), this transitively proves facade == plan == tree-level hand-chain,
// i.e. the facade wiring adds no recoupling math.
#[test]
fn generic_facade_permute_braid_transpose_match_b2c_plan_level() {
    let rule = ToyGenericRule {
        style: FusionStyleKind::Generic,
    };
    let (structure, _) = b3a_toy_tensormap(0.0);

    // Plan-level (B2c) reference for one operation.
    let plan_level = |operation: TreeTransformOperation| -> Vec<f64> {
        let (_, src) = b3a_toy_tensormap(7.0);
        let (_, mut dst) = b3a_toy_tensormap(0.0);
        let plan =
            build_generic_tree_pair_transform_group_plan(&rule, operation, &structure).unwrap();
        let compiled = plan.compile(&dst, &src).unwrap();
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &compiled,
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
        dst.data().to_vec()
    };

    let (_, src) = b3a_toy_tensormap(7.0);
    let (_, mut dst) = b3a_toy_tensormap(0.0);
    permute_into_generic(&rule, [1, 0], [], &mut dst, &src, 1.0, 0.0).unwrap();
    assert_eq!(
        dst.data(),
        plan_level(TreeTransformOperation::permute([1, 0], [])).as_slice()
    );

    let (_, mut dst) = b3a_toy_tensormap(0.0);
    braid_into_generic(&rule, [1, 0], [], [0, 1], [], &mut dst, &src, 1.0, 0.0).unwrap();
    assert_eq!(
        dst.data(),
        plan_level(TreeTransformOperation::braid([1, 0], [], [0, 1], [])).as_slice()
    );

    let (_, mut dst) = b3a_toy_tensormap(0.0);
    transpose_into_generic(&rule, [1, 0], [], &mut dst, &src, 1.0, 0.0).unwrap();
    assert_eq!(
        dst.data(),
        plan_level(TreeTransformOperation::transpose([1, 0], [])).as_slice()
    );
}

// Gate 3 (mult-free cannot enter the generic path): the compile-time guarantee
// is trait disjointness (`GenericRigidSymbols` vs `MultiplicityFreeRigidSymbols`
// are never both implemented). This is its runtime symmetric sibling — the
// facade's Generic entry rejects a rule that reports a multiplicity-free style,
// mirroring the mult-free builders' `UnsupportedFusionStyle` guards.
#[test]
fn generic_facade_structure_rejects_multiplicity_free_style() {
    let mf = ToyGenericRule {
        style: FusionStyleKind::Simple,
    };
    let (_, src) = b3a_toy_tensormap(7.0);
    let (_, dst) = b3a_toy_tensormap(0.0);
    let err = tree_transform_structure_generic(
        &mf,
        TreeTransformOperation::permute([1, 0], []),
        &dst,
        &src,
    )
    .unwrap_err();
    assert!(matches!(err, OperationError::UnsupportedFusionStyle { .. }));
}

// Stage B3b: the non-memoized generic cache sibling
// (`get_or_compile_tree_pair_generic`) drives the REAL SU(3) table provider and
// must reproduce the (proven) non-cached facade path byte-for-byte, keyed by the
// table's provenance hash.
#[test]
fn b3b_su3_cache_generic_sibling_matches_facade() {
    use crate::TreeTransformRuleCacheKey;
    use tenet_core::Su3FusionRule;

    let rule = Su3FusionRule::new();
    // The Su3 cache key embeds the table provenance: two handles to the same
    // table produce equal keys (so plans are shared), and the key carries the
    // provenance hash (so a swapped table cannot collide).
    let key = rule.tree_transform_rule_cache_key();
    assert_eq!(key, Su3FusionRule::new().tree_transform_rule_cache_key());
    assert_ne!(rule.provenance(), 0);

    let eight = rule.sector_of(1, 1).unwrap();
    let vac = tenet_core::SectorId::new(0);
    // codomain [8,8]->vac (single vertex), domain []->vac: one 1-element block.
    let make = |value: f64| {
        let cod = FusionTreeKey::try_new_for_rule(
            &rule,
            [eight, eight],
            vac,
            [false, false],
            [],
            [tenet_core::MultiplicityIndex::ONE],
        )
        .unwrap();
        let dom = FusionTreeKey::try_new_for_rule(&rule, [], vac, [], [], []).unwrap();
        let key = BlockKey::from(FusionTreePairKey::pair(cod, dom));
        let structure = packed_fixture_structure(2, [(key, vec![1, 1])]).unwrap();
        let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![value], space, structure).unwrap()
    };

    let src = make(7.0);
    // Facade (non-cached) reference.
    let mut dst_facade = make(0.0);
    permute_into_generic(&rule, [1, 0], [], &mut dst_facade, &src, 1.0, 0.0).unwrap();

    // Cache sibling: compile via get_or_compile_tree_pair_generic, then execute.
    let mut cache =
        TreeTransformCache::<f64, crate::tree_transform::TreeTransformSu3RuleCacheKey>::new();
    let mut dst_cache = make(0.0);
    let structure = cache
        .get_or_compile_tree_pair_generic(
            &rule,
            TreeTransformOperation::permute([1, 0], []),
            &dst_cache,
            &src,
        )
        .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst_cache,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst_cache.data(), dst_facade.data());

    // What: the generic-fusion overwrite entry point writes directly over dirty
    // destination storage on repeated non-memoized compilation.
    let mut context = TreeTransformExecutionContext::<
        f64,
        crate::tree_transform::TreeTransformSu3RuleCacheKey,
    >::default();
    let mut dst_overwrite = make(f64::NAN);
    let dst_structure = Arc::clone(dst_overwrite.structure());
    let src_structure = Arc::clone(src.structure());
    for _ in 0..2 {
        dst_overwrite.data_mut().fill(f64::NAN);
        context
            .tree_transform_dyn_overwrite_into_generic(
                &rule,
                TreeTransformOperation::permute([1, 0], []),
                &dst_structure,
                &src_structure,
                dst_overwrite.data_mut(),
                src.data(),
                1.0,
            )
            .unwrap();
        assert_eq!(dst_overwrite.data(), dst_facade.data());
    }
    assert_eq!(context.cache().structure_len(), 0);
}

// ===================== Stage B3c-1: SU(4) DATA-ONLY smoke ==================
//
// The identical Generic pipeline — the `R: FusionRule` / `GenericRigidSymbols`
// tree-transform and contract siblings — driven from a *different* group's
// checked-in blob (a small SU(4), dim ≤ 15) via `TabulatedFusionRule::try_from_bytes`,
// with ZERO Rust changes. Proves permute (real SU(4) F-symbol recoupling) and
// contract (core/compose GEMM) are group-agnostic: a new group is data only.
#[cfg(test)]
mod b3c1_su4_smoke {
    use super::*;
    use crate::permute_into_generic;
    use crate::{
        BoundDynamicFusionMapSpace, DynamicFusionMapSpace, HostTreeFusionExecutionContext,
    };
    use std::sync::Arc;
    use tenet_core::{
        FusionProductSpace, FusionTreeHomSpace, FusionTreeKey, SectorLeg, TabulatedFusionRule,
    };

    static SU4_BYTES: &[u8] = include_bytes!("../../../tenet-core/src/testdata/su4_table.bin");

    fn su4() -> TabulatedFusionRule {
        TabulatedFusionRule::try_from_bytes(SU4_BYTES, "su4_table.bin").unwrap()
    }

    // Construction + permute: a `[4,4̄] <- vac` singlet tensor; swapping the two
    // codomain legs genuinely recouples through the SU(4) F-symbol, and swapping
    // back returns the original data (invertibility over the su4 table).
    #[test]
    fn su4_permute_round_trip_is_data_only() {
        let rule = su4();
        let four = rule.sector_of_label(&[1, 0, 0]).unwrap();
        let fourbar = rule.dual(four); // 4 ⊗ 4̄ ∋ 1 (covered), and 4 ≠ 4̄.
        let vac = SectorId::new(0);
        // `[a, b] <- vac` singlet map. The two codomain sectors differ, so a
        // leg swap really reorders the fusion tree (recouples via SU(4) F/R).
        let make = |a: SectorId, b: SectorId, value: f64| {
            let cod = FusionTreeKey::try_new_for_rule(
                &rule,
                [a, b],
                vac,
                [false, false],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap();
            let dom = FusionTreeKey::try_new_for_rule(&rule, [], vac, [], [], []).unwrap();
            let key = BlockKey::from(FusionTreePairKey::pair(cod, dom));
            let structure = packed_fixture_structure(2, [(key, vec![1, 1])]).unwrap();
            let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![value], space, structure).unwrap()
        };
        let src = make(four, fourbar, 3.5);
        let mut swapped = make(fourbar, four, 0.0); // permuted leg order
        permute_into_generic(&rule, [1, 0], [], &mut swapped, &src, 1.0, 0.0).unwrap();
        let mut back = make(four, fourbar, 0.0);
        permute_into_generic(&rule, [1, 0], [], &mut back, &swapped, 1.0, 0.0).unwrap();
        assert_eq!(back.data().len(), src.data().len());
        for (x, y) in back.data().iter().zip(src.data().iter()) {
            assert!((x - y).abs() < 1e-12, "su4 permute round-trip: {x} vs {y}");
        }
    }

    // Contract (core/compose): `A:[4]<-[4]` composed with `B:[4]<-[4]` over the
    // shared coupled-4 leg. Proves the generic block-GEMM contract plan compiles
    // and executes on SU(4) data. Value = a·b in the single 1×1 coupled block.
    #[test]
    fn su4_contract_core_route_is_data_only() {
        let rule = su4();
        let four = rule.sector_of_label(&[1, 0, 0]).unwrap();
        let map4 = |value: f64| {
            let leg = SectorLeg::new([(four, 1usize)], false);
            let hom = FusionTreeHomSpace::new(
                FusionProductSpace::new([leg.clone()]),
                FusionProductSpace::new([leg]),
            );
            let keys = hom.fusion_tree_keys_generic(&rule).unwrap();
            let shapes: Vec<Vec<usize>> = keys.iter().map(|_| vec![1, 1]).collect();
            let space =
                DynamicFusionMapSpace::from_degeneracy_shapes_generic(&rule, hom, shapes).unwrap();
            let data = vec![value; space.required_len().unwrap()];
            (space, data)
        };
        let (a_space, a_data) = map4(2.0);
        let (b_space, b_data) = map4(5.0);
        // A domain leg 0 with B codomain leg 0 (compose): [4]<-[4].
        let dst = DynamicFusionMapSpace::contracted_generic(&rule, &a_space, &b_space, &[1], &[0])
            .unwrap();
        let provider = Arc::new(rule);
        let dst = BoundDynamicFusionMapSpace::bind_generic(dst, Arc::clone(&provider)).unwrap();
        let a_space =
            BoundDynamicFusionMapSpace::bind_generic(a_space, Arc::clone(&provider)).unwrap();
        let b_space =
            BoundDynamicFusionMapSpace::bind_generic(b_space, Arc::clone(&provider)).unwrap();
        let mut dst_data = vec![0.0f64; dst.space().required_len().unwrap()];
        let mut ctx = HostTreeFusionExecutionContext::<f64, u64>::default();
        ctx.tensorcontract_fusion_dyn_into_generic(
            &dst,
            &mut dst_data,
            &a_space,
            &a_data,
            &b_space,
            &b_data,
            tenet_operations::TensorContractSpec::with_default_output_order(&[1], &[0]),
            1.0,
            0.0,
        )
        .unwrap();
        assert_eq!(dst_data.len(), 1, "single coupled-4 block");
        assert!(
            (dst_data[0] - 10.0).abs() < 1e-12,
            "A∘B = 2·5 = 10, got {}",
            dst_data[0]
        );
    }

    #[test]
    fn baked_fused_layouts_match_recompute_for_su2_plans() {
        // What: on real degeneracy-2 SU(2) plans, every baked fused layout is
        // byte-identical to a fresh fuse_pair_layout of its (block, role) stride
        // pair (issue #232) — covering all three roles: pack + scatter from a
        // generic recoupling Multi block, single from a first-pair braid that
        // lowers to Singles.
        let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
            &SU2FusionRule,
            [1, 1, 1, 1],
            0,
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );
        let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
            &SU2FusionRule,
            [1, 1, 1, 1],
            0,
            [false, false, false, false],
            [2, 1],
            [1, 1, 1],
        );

        let recoupling_structure = packed_fixture_structure(
            4,
            [
                (src_key0.clone(), vec![2, 2, 2, 2]),
                (src_key1.clone(), vec![2, 2, 2, 2]),
            ],
        )
        .unwrap();
        let recoupling = build_all_codomain_tree_transform_group_plan(
            &SU2FusionRule,
            TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []),
            &recoupling_structure,
        )
        .unwrap()
        .compile_structures(&recoupling_structure, &recoupling_structure)
        .unwrap();
        assert!(recoupling.has_pack_gemm_scatter_blocks());
        assert!(recoupling.baked_layouts_match_recomputed());

        let single_structure = packed_fixture_structure(
            4,
            [
                (src_key0.clone(), vec![2, 2, 2, 2]),
                (src_key1.clone(), vec![2, 2, 2, 2]),
            ],
        )
        .unwrap();
        let singles = build_all_codomain_tree_transform_group_plan(
            &SU2FusionRule,
            TreeTransformOperation::braid([1, 0, 2, 3], [], [0, 1, 2, 3], []),
            &single_structure,
        )
        .unwrap()
        .compile_structures(&single_structure, &single_structure)
        .unwrap();
        assert!(!singles.has_pack_gemm_scatter_blocks());
        assert!(singles.baked_layouts_match_recomputed());
    }

    #[test]
    fn baked_arena_growth_reported_for_su2_recoupling_plans() {
        // What: report and bound the baked-arena plan-size growth on the SU(2)
        // pack/scatter recoupling plans at degeneracy 8 and 16 (issue #232
        // plan-size table). The compact arena must beat the fixed 200-byte
        // FusedPairLayout array it replaces for every baked entry.
        for degeneracy in [8usize, 16] {
            let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
                &SU2FusionRule,
                [1, 1, 1, 1],
                0,
                [false, false, false, false],
                [0, 1],
                [1, 1, 1],
            );
            let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
                &SU2FusionRule,
                [1, 1, 1, 1],
                0,
                [false, false, false, false],
                [2, 1],
                [1, 1, 1],
            );
            let structure = packed_fixture_structure(
                4,
                [
                    (src_key0.clone(), vec![degeneracy; 4]),
                    (src_key1.clone(), vec![degeneracy; 4]),
                ],
            )
            .unwrap();
            let plan = build_all_codomain_tree_transform_group_plan(
                &SU2FusionRule,
                TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []),
                &structure,
            )
            .unwrap()
            .compile_structures(&structure, &structure)
            .unwrap();
            assert!(plan.has_pack_gemm_scatter_blocks());
            assert!(plan.baked_layouts_match_recomputed());
            let base = plan.layouts().layout_table_bytes();
            let baked = plan.layouts().baked_arena_bytes();
            eprintln!(
                "su2_d{degeneracy}: base={base}B baked={baked}B growth={:.1}%",
                baked as f64 / base as f64 * 100.0
            );
        }
    }
}
