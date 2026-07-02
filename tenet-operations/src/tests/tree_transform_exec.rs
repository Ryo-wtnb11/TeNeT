use super::*;

#[test]
fn host_tree_transform_workspace_is_explicit_host_workspace() {
    let workspace = HostTreeTransformWorkspace::<f64>::default();
    let alias = TreeTransformWorkspace::<f64>::default();

    assert_eq!(workspace.placement(), Placement::Host);
    assert!(workspace.is_host_workspace());
    assert_eq!(workspace.source_len(), 0);
    assert_eq!(workspace.destination_len(), 0);
    assert_eq!(alias.placement(), Placement::Host);
    assert_eq!(alias.source_len(), workspace.source_len());
    assert_eq!(alias.destination_len(), workspace.destination_len());
}

#[test]
fn tree_transform_structure_replays_custom_host_storage_without_vec_fixing() {
    let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let structure = BlockStructure::packed_column_major(2, [vec![2, 2]]).unwrap();
    let src = test_host_read_tensor_map_with_structure(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        space.clone(),
        structure.clone(),
    );
    let mut dst = test_host_tensor_map_with_structure(vec![10.0_f64; 4], space, structure);
    let transform =
        TreeTransformStructure::compile(&dst, &src, &[TreeTransformBlockSpec::single(0, 0, 3.0)])
            .unwrap();
    let mut backend = HostTensorOperations;
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &transform,
        &mut dst,
        &src,
        2.0,
        4.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[46.0, 52.0, 58.0, 64.0]);
}

#[test]
fn tree_transform_single_replay_supports_all_numeric_dtypes() {
    assert_tree_single_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        10.0,
        3.0,
        2.0,
        4.0,
        vec![46.0, 52.0, 58.0, 64.0],
    );
    assert_tree_single_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        10.0,
        3.0,
        2.0,
        4.0,
        vec![46.0, 52.0, 58.0, 64.0],
    );
    assert_tree_single_dtype(vec![1_i32, 2, 3, 4], 10, 3, 2, 4, vec![46, 52, 58, 64]);
    assert_tree_single_dtype(vec![1_i64, 2, 3, 4], 10, 3, 2, 4, vec![46, 52, 58, 64]);
    assert_tree_single_dtype(
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, -0.5),
        ],
        Complex32::new(10.0, 1.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(4.0, 0.0),
        vec![
            Complex32::new(46.0, 10.0),
            Complex32::new(52.0, -2.0),
            Complex32::new(58.0, 7.0),
            Complex32::new(64.0, 1.0),
        ],
    );
    assert_tree_single_dtype(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        Complex64::new(10.0, 1.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
        vec![
            Complex64::new(46.0, 10.0),
            Complex64::new(52.0, -2.0),
            Complex64::new(58.0, 7.0),
            Complex64::new(64.0, 1.0),
        ],
    );
}

#[test]
fn tree_transform_single_replay_supports_complex_data_with_real_coefficients() {
    assert_tree_single_mixed_dtype(
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, -0.5),
        ],
        Complex32::new(10.0, 1.0),
        3.0_f64,
        Complex32::new(2.0, 0.0),
        Complex32::new(4.0, 0.0),
        vec![
            Complex32::new(46.0, 10.0),
            Complex32::new(52.0, -2.0),
            Complex32::new(58.0, 7.0),
            Complex32::new(64.0, 1.0),
        ],
    );
    assert_tree_single_mixed_dtype(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        Complex64::new(10.0, 1.0),
        3.0_f64,
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
        vec![
            Complex64::new(46.0, 10.0),
            Complex64::new(52.0, -2.0),
            Complex64::new(58.0, 7.0),
            Complex64::new(64.0, 1.0),
        ],
    );
}

#[test]
fn tree_transform_multi_pack_gemm_scatter_supports_all_numeric_dtypes() {
    assert_tree_multi_dtype(
        vec![2.0_f32, 3.0, 5.0, 7.0],
        2.0,
        10.0,
        1.0,
        vec![44.0, 54.0, 64.0, 74.0, 90.0, 114.0, 138.0, 162.0],
    );
    assert_tree_multi_dtype(
        vec![2.0_f64, 3.0, 5.0, 7.0],
        2.0,
        10.0,
        1.0,
        vec![44.0, 54.0, 64.0, 74.0, 90.0, 114.0, 138.0, 162.0],
    );
    assert_tree_multi_dtype(
        vec![2_i32, 3, 5, 7],
        2,
        10,
        1,
        vec![44, 54, 64, 74, 90, 114, 138, 162],
    );
    assert_tree_multi_dtype(
        vec![2_i64, 3, 5, 7],
        2,
        10,
        1,
        vec![44, 54, 64, 74, 90, 114, 138, 162],
    );
    assert_tree_multi_dtype(
        vec![
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(7.0, 0.0),
        ],
        Complex32::new(2.0, 0.0),
        Complex32::new(10.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(44.0, 10.0),
            Complex32::new(54.0, 10.0),
            Complex32::new(64.0, 10.0),
            Complex32::new(74.0, 10.0),
            Complex32::new(90.0, 10.0),
            Complex32::new(114.0, 10.0),
            Complex32::new(138.0, 10.0),
            Complex32::new(162.0, 10.0),
        ],
    );
    assert_tree_multi_dtype(
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(7.0, 0.0),
        ],
        Complex64::new(2.0, 0.0),
        Complex64::new(10.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(44.0, 10.0),
            Complex64::new(54.0, 10.0),
            Complex64::new(64.0, 10.0),
            Complex64::new(74.0, 10.0),
            Complex64::new(90.0, 10.0),
            Complex64::new(114.0, 10.0),
            Complex64::new(138.0, 10.0),
            Complex64::new(162.0, 10.0),
        ],
    );
}

#[test]
fn tree_transform_multi_pack_gemm_scatter_supports_complex_data_with_real_coefficients() {
    assert_tree_multi_mixed_dtype(
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
            Complex32::new(7.0, 0.0),
            Complex32::new(8.0, 0.0),
        ],
        vec![2.0_f64, 3.0, 5.0, 7.0],
        Complex32::new(2.0, 0.0),
        Complex32::new(10.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(44.0, 10.0),
            Complex32::new(54.0, 10.0),
            Complex32::new(64.0, 10.0),
            Complex32::new(74.0, 10.0),
            Complex32::new(90.0, 10.0),
            Complex32::new(114.0, 10.0),
            Complex32::new(138.0, 10.0),
            Complex32::new(162.0, 10.0),
        ],
    );
    assert_tree_multi_mixed_dtype(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0),
            Complex64::new(8.0, 0.0),
        ],
        vec![2.0_f64, 3.0, 5.0, 7.0],
        Complex64::new(2.0, 0.0),
        Complex64::new(10.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(44.0, 10.0),
            Complex64::new(54.0, 10.0),
            Complex64::new(64.0, 10.0),
            Complex64::new(74.0, 10.0),
            Complex64::new(90.0, 10.0),
            Complex64::new(114.0, 10.0),
            Complex64::new(138.0, 10.0),
            Complex64::new(162.0, 10.0),
        ],
    );
}

#[test]
fn tree_transform_multi_uses_tensorkit_recoupling_orientation_for_all_numeric_dtypes() {
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1_i32, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        2,
        3,
        1,
        vec![10623, 12843, 21243, 25683],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1_i64, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        2,
        3,
        1,
        vec![10623, 12843, 21243, 25683],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
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
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(10623.0, 3.0),
            Complex32::new(12843.0, 3.0),
            Complex32::new(21243.0, 3.0),
            Complex32::new(25683.0, 3.0),
        ],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
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
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(10623.0, 3.0),
            Complex64::new(12843.0, 3.0),
            Complex64::new(21243.0, 3.0),
            Complex64::new(25683.0, 3.0),
        ],
    );
}

#[test]
fn tree_transform_dense_backend_matches_tensorkit_recoupling_orientation_for_gemm_dtypes() {
    assert_tree_multi_tensorkit_orientation_dense_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dense_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dense_dtype(
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
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(10623.0, 3.0),
            Complex32::new(12843.0, 3.0),
            Complex32::new(21243.0, 3.0),
            Complex32::new(25683.0, 3.0),
        ],
    );
    assert_tree_multi_tensorkit_orientation_dense_dtype(
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
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(10623.0, 3.0),
            Complex64::new(12843.0, 3.0),
            Complex64::new(21243.0, 3.0),
            Complex64::new(25683.0, 3.0),
        ],
    );
}

#[test]
fn tree_transform_dense_backend_keeps_multi_tree_recoupling_in_replay_kernel() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
    let src_structure =
        BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1], vec![2, 1]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1]]).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 4], dst_space, dst_structure)
            .unwrap();
    let structure = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1, 2],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        )],
    )
    .unwrap();
    let mut backend = DenseTreeTransformOperations::new(CountingDenseExecutor::default());
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

    assert_eq!(backend.dense().dot_general_into_calls, 0);
    assert_eq!(dst.data(), &[10623.0, 12843.0, 21243.0, 25683.0]);
}
