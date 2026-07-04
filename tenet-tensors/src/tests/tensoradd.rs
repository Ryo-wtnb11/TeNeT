use super::*;

#[test]
fn tensoradd_structure_replays_custom_host_storage_without_vec_fixing() {
    let space = TensorMapSpace::<1, 0>::from_dims([4], []).unwrap();
    let src = test_host_read_tensor_map(vec![1.0_f64, 2.0, 3.0, 4.0], space.clone());
    let mut dst = test_host_tensor_map(vec![10.0_f64; 4], space);
    let structure = TensorAddStructure::compile(&dst, &src, OutputAxisOrder::identity()).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    structure
        .execute_with(&mut backend, &mut allocator, &mut dst, &src, 2.0, 3.0)
        .unwrap();

    assert_eq!(dst.data(), &[32.0, 34.0, 36.0, 38.0]);
}

#[test]
fn tensoradd_default_host_api_accepts_custom_host_storage() {
    let space = TensorMapSpace::<1, 0>::from_dims([4], []).unwrap();
    let src = test_host_read_tensor_map(vec![1.0_f64, 2.0, 3.0, 4.0], space.clone());
    let mut dst = test_host_tensor_map(vec![10.0_f64; 4], space);

    tensoradd_into(&mut dst, &src, OutputAxisOrder::identity(), 2.0, 3.0).unwrap();

    assert_eq!(dst.data(), &[32.0, 34.0, 36.0, 38.0]);
}

#[test]
fn tensoradd_assign_and_add_support_all_numeric_dtypes() {
    assert_tensoradd_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        10.0,
        2.0,
        vec![2.0, 4.0, 6.0, 8.0],
        vec![12.0, 14.0, 16.0, 18.0],
    );
    assert_tensoradd_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        10.0,
        2.0,
        vec![2.0, 4.0, 6.0, 8.0],
        vec![12.0, 14.0, 16.0, 18.0],
    );
    assert_tensoradd_dtype(
        vec![1_i32, 2, 3, 4],
        10,
        2,
        vec![2, 4, 6, 8],
        vec![12, 14, 16, 18],
    );
    assert_tensoradd_dtype(
        vec![1_i64, 2, 3, 4],
        10,
        2,
        vec![2, 4, 6, 8],
        vec![12, 14, 16, 18],
    );
    assert_tensoradd_dtype(
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, -0.5),
        ],
        Complex32::new(10.0, 0.0),
        Complex32::new(2.0, 0.0),
        vec![
            Complex32::new(2.0, 2.0),
            Complex32::new(4.0, -2.0),
            Complex32::new(6.0, 1.0),
            Complex32::new(8.0, -1.0),
        ],
        vec![
            Complex32::new(12.0, 2.0),
            Complex32::new(14.0, -2.0),
            Complex32::new(16.0, 1.0),
            Complex32::new(18.0, -1.0),
        ],
    );
    assert_tensoradd_dtype(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        Complex64::new(10.0, 0.0),
        Complex64::new(2.0, 0.0),
        vec![
            Complex64::new(2.0, 2.0),
            Complex64::new(4.0, -2.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, -1.0),
        ],
        vec![
            Complex64::new(12.0, 2.0),
            Complex64::new(14.0, -2.0),
            Complex64::new(16.0, 1.0),
            Complex64::new(18.0, -1.0),
        ],
    );
}

#[test]
fn tensoradd_general_beta_supports_all_numeric_dtypes() {
    assert_tensoradd_general_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        10.0,
        2.0,
        3.0,
        vec![32.0, 34.0, 36.0, 38.0],
    );
    assert_tensoradd_general_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        10.0,
        2.0,
        3.0,
        vec![32.0, 34.0, 36.0, 38.0],
    );
    assert_tensoradd_general_dtype(vec![1_i32, 2, 3, 4], 10, 2, 3, vec![32, 34, 36, 38]);
    assert_tensoradd_general_dtype(vec![1_i64, 2, 3, 4], 10, 2, 3, vec![32, 34, 36, 38]);
    assert_tensoradd_general_dtype(
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, -0.5),
        ],
        Complex32::new(10.0, 1.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        vec![
            Complex32::new(32.0, 5.0),
            Complex32::new(34.0, 1.0),
            Complex32::new(36.0, 4.0),
            Complex32::new(38.0, 2.0),
        ],
    );
    assert_tensoradd_general_dtype(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        Complex64::new(10.0, 1.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        vec![
            Complex64::new(32.0, 5.0),
            Complex64::new(34.0, 1.0),
            Complex64::new(36.0, 4.0),
            Complex64::new(38.0, 2.0),
        ],
    );
}

#[test]
fn tensoradd_permuted_general_beta_supports_all_numeric_dtypes() {
    assert_tensoradd_permuted_general_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        10.0,
        2.0,
        3.0,
        vec![32.0, 36.0, 40.0, 34.0, 38.0, 42.0],
    );
    assert_tensoradd_permuted_general_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        10.0,
        2.0,
        3.0,
        vec![32.0, 36.0, 40.0, 34.0, 38.0, 42.0],
    );
    assert_tensoradd_permuted_general_dtype(
        vec![1_i32, 2, 3, 4, 5, 6],
        10,
        2,
        3,
        vec![32, 36, 40, 34, 38, 42],
    );
    assert_tensoradd_permuted_general_dtype(
        vec![1_i64, 2, 3, 4, 5, 6],
        10,
        2,
        3,
        vec![32, 36, 40, 34, 38, 42],
    );

    let values32 = vec![
        Complex32::new(1.0, 1.0),
        Complex32::new(2.0, -2.0),
        Complex32::new(3.0, 0.5),
        Complex32::new(4.0, -1.0),
        Complex32::new(5.0, 2.0),
        Complex32::new(6.0, -3.0),
    ];
    let fill32 = Complex32::new(10.0, 1.0);
    let alpha32 = Complex32::new(2.0, -1.0);
    let beta32 = Complex32::new(-1.0, 0.5);
    let reordered32 = [
        values32[0],
        values32[2],
        values32[4],
        values32[1],
        values32[3],
        values32[5],
    ];
    let expected32 = reordered32
        .iter()
        .copied()
        .map(|value| beta32 * fill32 + alpha32 * value)
        .collect();
    assert_tensoradd_permuted_general_dtype(values32, fill32, alpha32, beta32, expected32);

    let values64 = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -2.0),
        Complex64::new(3.0, 0.5),
        Complex64::new(4.0, -1.0),
        Complex64::new(5.0, 2.0),
        Complex64::new(6.0, -3.0),
    ];
    let fill64 = Complex64::new(10.0, 1.0);
    let alpha64 = Complex64::new(2.0, -1.0);
    let beta64 = Complex64::new(-1.0, 0.5);
    let reordered64 = [
        values64[0],
        values64[2],
        values64[4],
        values64[1],
        values64[3],
        values64[5],
    ];
    let expected64 = reordered64
        .iter()
        .copied()
        .map(|value| beta64 * fill64 + alpha64 * value)
        .collect();
    assert_tensoradd_permuted_general_dtype(values64, fill64, alpha64, beta64, expected64);
}

#[test]
fn tensoradd_permuted_assign_and_add_support_all_numeric_dtypes() {
    assert_tensoradd_permuted_general_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        10.0,
        2.0,
        0.0,
        vec![2.0, 6.0, 10.0, 4.0, 8.0, 12.0],
    );
    assert_tensoradd_permuted_general_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        10.0,
        2.0,
        1.0,
        vec![12.0, 16.0, 20.0, 14.0, 18.0, 22.0],
    );
    assert_tensoradd_permuted_general_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        10.0,
        2.0,
        0.0,
        vec![2.0, 6.0, 10.0, 4.0, 8.0, 12.0],
    );
    assert_tensoradd_permuted_general_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        10.0,
        2.0,
        1.0,
        vec![12.0, 16.0, 20.0, 14.0, 18.0, 22.0],
    );
    assert_tensoradd_permuted_general_dtype(
        vec![1_i32, 2, 3, 4, 5, 6],
        10,
        2,
        0,
        vec![2, 6, 10, 4, 8, 12],
    );
    assert_tensoradd_permuted_general_dtype(
        vec![1_i32, 2, 3, 4, 5, 6],
        10,
        2,
        1,
        vec![12, 16, 20, 14, 18, 22],
    );
    assert_tensoradd_permuted_general_dtype(
        vec![1_i64, 2, 3, 4, 5, 6],
        10,
        2,
        0,
        vec![2, 6, 10, 4, 8, 12],
    );
    assert_tensoradd_permuted_general_dtype(
        vec![1_i64, 2, 3, 4, 5, 6],
        10,
        2,
        1,
        vec![12, 16, 20, 14, 18, 22],
    );

    let values32 = vec![
        Complex32::new(1.0, 1.0),
        Complex32::new(2.0, -2.0),
        Complex32::new(3.0, 0.5),
        Complex32::new(4.0, -1.0),
        Complex32::new(5.0, 2.0),
        Complex32::new(6.0, -3.0),
    ];
    let alpha32 = Complex32::new(2.0, -1.0);
    let fill32 = Complex32::new(10.0, 1.0);
    let reordered32 = [
        values32[0],
        values32[2],
        values32[4],
        values32[1],
        values32[3],
        values32[5],
    ];
    let assign32 = reordered32
        .iter()
        .copied()
        .map(|value| alpha32 * value)
        .collect();
    let add32 = reordered32
        .iter()
        .copied()
        .map(|value| fill32 + alpha32 * value)
        .collect();
    assert_tensoradd_permuted_general_dtype(
        values32.clone(),
        fill32,
        alpha32,
        Complex32::zero(),
        assign32,
    );
    assert_tensoradd_permuted_general_dtype(values32, fill32, alpha32, Complex32::one(), add32);

    let values64 = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -2.0),
        Complex64::new(3.0, 0.5),
        Complex64::new(4.0, -1.0),
        Complex64::new(5.0, 2.0),
        Complex64::new(6.0, -3.0),
    ];
    let alpha64 = Complex64::new(2.0, -1.0);
    let fill64 = Complex64::new(10.0, 1.0);
    let reordered64 = [
        values64[0],
        values64[2],
        values64[4],
        values64[1],
        values64[3],
        values64[5],
    ];
    let assign64 = reordered64
        .iter()
        .copied()
        .map(|value| alpha64 * value)
        .collect();
    let add64 = reordered64
        .iter()
        .copied()
        .map(|value| fill64 + alpha64 * value)
        .collect();
    assert_tensoradd_permuted_general_dtype(
        values64.clone(),
        fill64,
        alpha64,
        Complex64::zero(),
        assign64,
    );
    assert_tensoradd_permuted_general_dtype(values64, fill64, alpha64, Complex64::one(), add64);
}

#[test]
fn tensoradd_permuted_assign_does_not_read_destination() {
    assert_tensoradd_permuted_general_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        f64::NAN,
        2.0,
        0.0,
        vec![2.0, 6.0, 10.0, 4.0, 8.0, 12.0],
    );
}

#[test]
fn tensoradd_with_backend_allocator_applies_axis_permutation() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], src_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(10.0, dst_space).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    tensoradd_into_with(
        &mut backend,
        &mut allocator,
        &mut dst,
        &src,
        OutputAxisOrder::from_axes(&[1, 0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[32.0, 36.0, 40.0, 34.0, 38.0, 42.0]);
}

#[test]
fn tensoradd_with_conjugation_applies_dense_source_conj_and_permutation() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src = TensorMap::<Complex64, 2, 0>::from_vec(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, -4.0),
            Complex64::new(5.0, 5.0),
            Complex64::new(6.0, -6.0),
        ],
        src_space,
    )
    .unwrap();
    let mut dst =
        TensorMap::<Complex64, 2, 0>::filled(Complex64::new(10.0, 1.0), dst_space).unwrap();

    tensoradd_into_with_conjugation(
        &mut dst,
        &src,
        OutputAxisOrder::from_axes(&[1, 0]),
        true,
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(32.0, 1.0),
            Complex64::new(36.0, -3.0),
            Complex64::new(40.0, -7.0),
            Complex64::new(34.0, 7.0),
            Complex64::new(38.0, 11.0),
            Complex64::new(42.0, 15.0),
        ]
    );
}

#[test]
fn tensoradd_structure_precomputes_permutation_pairing_and_descriptor() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 6], src_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let structure =
        TensorAddStructure::compile(&dst, &src, OutputAxisOrder::from_axes(&[1, 0])).unwrap();

    assert_eq!(structure.rank(), 2);
    assert_eq!(structure.axes(), &[1, 0]);
    assert_eq!(structure.terms().len(), 1);
    assert_eq!(structure.terms()[0].key(), &BlockKey::trivial());
    assert_eq!(structure.terms()[0].dst_block(), 0);
    assert_eq!(structure.terms()[0].src_block(), 0);
}

#[test]
fn tensoradd_structure_replays_without_recompiling() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], src_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(10.0, dst_space).unwrap();
    let structure =
        TensorAddStructure::compile(&dst, &src, OutputAxisOrder::from_axes(&[1, 0])).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        2.0,
        0.0,
    )
    .unwrap();
    tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        1.0,
        1.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[3.0, 9.0, 15.0, 6.0, 12.0, 18.0]);
}

#[test]
fn tensoradd_structure_compiles_concrete_shape_and_replays_it() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([5, 4], []).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec((1..=20).map(|x| x as f64).collect(), src_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
    let structure =
        TensorAddStructure::compile(&dst, &src, OutputAxisOrder::from_axes(&[1, 0])).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        2.0,
        0.0,
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            2.0, 10.0, 18.0, 26.0, 34.0, 4.0, 12.0, 20.0, 28.0, 36.0, 6.0, 14.0, 22.0, 30.0, 38.0,
            8.0, 16.0, 24.0, 32.0, 40.0,
        ]
    );
}

#[test]
fn tensoradd_structure_replays_multiple_packed_blocks() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
    let src_structure = BlockStructure::packed_column_major(2, [vec![2, 3], vec![1, 4]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![3, 2], vec![4, 1]]).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        (1..=10).map(|x| x as f64).collect(),
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 10], dst_space, dst_structure)
            .unwrap();
    let structure =
        TensorAddStructure::compile(&dst, &src, OutputAxisOrder::from_axes(&[1, 0])).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        2.0,
        0.0,
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[2.0, 6.0, 10.0, 4.0, 8.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    );
}

#[test]
fn tensoradd_structure_pairs_blocks_by_key_not_index() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
    let src_structure = packed_fixture_structure(
        2,
        [
            (BlockKey::sector_ids([10]), vec![2, 3]),
            (BlockKey::sector_ids([20]), vec![1, 4]),
        ],
    )
    .unwrap();
    let dst_structure = packed_fixture_structure(
        2,
        [
            (BlockKey::sector_ids([20]), vec![4, 1]),
            (BlockKey::sector_ids([10]), vec![3, 2]),
        ],
    )
    .unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        (1..=10).map(|x| x as f64).collect(),
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 10], dst_space, dst_structure)
            .unwrap();
    let structure =
        TensorAddStructure::compile(&dst, &src, OutputAxisOrder::from_axes(&[1, 0])).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    assert_eq!(structure.terms()[0].key(), &BlockKey::sector_ids([20]));
    assert_eq!(structure.terms()[0].dst_block(), 0);
    assert_eq!(structure.terms()[0].src_block(), 1);
    assert_eq!(structure.terms()[1].key(), &BlockKey::sector_ids([10]));
    assert_eq!(structure.terms()[1].dst_block(), 1);
    assert_eq!(structure.terms()[1].src_block(), 0);

    tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        2.0,
        0.0,
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[14.0, 16.0, 18.0, 20.0, 2.0, 6.0, 10.0, 4.0, 8.0, 12.0]
    );
}

#[test]
fn tensoradd_structure_rejects_invalid_permutation_at_compile_time() {
    let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::filled(1.0, space.clone()).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, space).unwrap();

    let err =
        TensorAddStructure::compile(&dst, &src, OutputAxisOrder::from_axes(&[0, 0])).unwrap_err();

    assert_eq!(
        err,
        OperationError::InvalidPermutation {
            axes: vec![0, 0],
            rank: 2,
        }
    );
}

#[test]
fn plain_tensoradd_rejects_fusion_tree_permutation_without_rule() {
    let rule = Z2FusionRule;
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([0, 0], []),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([0, 0], []),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let src: TensorMap<f64, 2, 0> =
        TensorMap::from_vec_with_fusion_space(vec![1.0], src_space).unwrap();
    let mut dst: TensorMap<f64, 2, 0> =
        TensorMap::from_vec_with_fusion_space(vec![0.0], dst_space).unwrap();

    let err = tensoradd_into(
        &mut dst,
        &src,
        OutputAxisOrder::from_axes(&[1, 0]),
        1.0,
        0.0,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message:
                "plain tensoradd does not lower fusion-tree permutations; use tree_pair_transform_*"
        }
    );
    assert_eq!(dst.data(), &[0.0]);
}

#[test]
fn plain_tensoradd_rejects_fusion_tree_conjugation_without_categorical_adjoint() {
    let rule = Z2FusionRule;
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 0>::from_dims([1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([0], []),
        &rule,
        [vec![1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 0>::from_dims([1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([0], []),
        &rule,
        [vec![1]],
    )
    .unwrap();
    let src: TensorMap<Complex64, 1, 0> =
        TensorMap::from_vec_with_fusion_space(vec![Complex64::new(1.0, 2.0)], src_space).unwrap();
    let mut dst: TensorMap<Complex64, 1, 0> =
        TensorMap::from_vec_with_fusion_space(vec![Complex64::new(0.0, 0.0)], dst_space).unwrap();

    let err = tensoradd_into_with_conjugation(
        &mut dst,
        &src,
        OutputAxisOrder::identity(),
        true,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message:
                "plain tensoradd with fusion-tree conjugation requires categorical adjoint lowering"
        }
    );
    assert_eq!(dst.data(), &[Complex64::new(0.0, 0.0)]);
}

fn z2_tensoradd_adjoint_fixture() -> (
    Z2FusionRule,
    TensorMap<Complex64, 2, 1>,
    TensorMap<Complex64, 1, 2>,
    Vec<Complex64>,
) {
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 1>::from_dims([2, 3], [5]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([odd], false), SectorLeg::new([even], false)]),
            FusionProductSpace::new([SectorLeg::new([odd], false)]),
        ),
        &rule,
        [vec![2, 3, 5]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 2>::from_dims([5], [2, 3]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([odd], false)]),
            FusionProductSpace::new([SectorLeg::new([odd], false), SectorLeg::new([even], false)]),
        ),
        &rule,
        [vec![5, 2, 3]],
    )
    .unwrap();
    let src_data = (0..30)
        .map(|i| Complex64::new(i as f64 + 1.0, -(i as f64 + 0.25)))
        .collect::<Vec<_>>();
    let src = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(src_data.clone(), src_space)
        .unwrap();
    let dst = TensorMap::<Complex64, 1, 2>::from_vec_with_fusion_space(
        vec![Complex64::new(100.0, 7.0); 30],
        dst_space,
    )
    .unwrap();
    let mut expected = Vec::new();
    for z in 0..3 {
        for y in 0..2 {
            for x in 0..5 {
                expected.push(src_data[y + 2 * z + 6 * x].conj());
            }
        }
    }
    (rule, src, dst, expected)
}

#[test]
fn tensoradd_fusion_conjugation_lowers_source_adjoint_like_tensorkit() {
    let (rule, src, mut dst, expected) = z2_tensoradd_adjoint_fixture();

    tensoradd_fusion_into(
        &rule,
        &mut dst,
        &src,
        TreeTransformOperation::permute([2], [0, 1]),
        true,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(dst.data(), expected.as_slice());
}

#[test]
fn tensoradd_fusion_conjugation_context_replays_without_recompiling() {
    let (rule, src, mut dst, expected) = z2_tensoradd_adjoint_fixture();
    type RuleKey = <Z2FusionRule as TreeTransformRuleCacheKey>::Key;
    let mut context = TreeTransformExecutionContext::<
        Complex64,
        RuleKey,
        f64,
        DenseTreeTransformOperations,
    >::default();

    tensoradd_fusion_into_with_context(
        &mut context,
        &rule,
        &mut dst,
        &src,
        TreeTransformOperation::permute([2], [0, 1]),
        true,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    assert_eq!(context.cache().stats().plan_hits(), 0);
    assert_eq!(context.cache().stats().plan_misses(), 1);
    assert_eq!(context.cache().stats().structure_hits(), 0);
    assert_eq!(context.cache().stats().structure_misses(), 1);
    assert_eq!(dst.data(), expected.as_slice());

    tensoradd_fusion_into_with_context(
        &mut context,
        &rule,
        &mut dst,
        &src,
        TreeTransformOperation::permute([2], [0, 1]),
        true,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    assert_eq!(context.cache().stats().plan_hits(), 1);
    assert_eq!(context.cache().stats().plan_misses(), 1);
    assert_eq!(context.cache().stats().structure_hits(), 1);
    assert_eq!(context.cache().stats().structure_misses(), 1);
    assert_eq!(dst.data(), expected.as_slice());
}

#[test]
fn tensoradd_source_adjoint_lowering_remaps_operation_axes_once() {
    let lowered = crate::lowering::lower_tensoradd_source_operation::<2, 1>(
        TreeTransformOperation::permute([2], [0, 1]),
        true,
    )
    .unwrap();

    assert!(lowered.storage_conjugate());
    assert_eq!(
        lowered.into_operation(),
        TreeTransformOperation::permute([0], [1, 2])
    );
}

#[test]
fn tensoradd_source_adjoint_lowering_remaps_explicit_braid_levels_and_reverses_direction() {
    let lowered = crate::lowering::lower_tensoradd_source_operation::<2, 1>(
        TreeTransformOperation::braid([2], [0, 1], [0, 2], [1]),
        true,
    )
    .unwrap();

    assert!(lowered.storage_conjugate());
    assert_eq!(
        lowered.into_operation(),
        TreeTransformOperation::braid([0], [1, 2], [1], [2, 0])
    );
}

#[test]
fn tensoradd_source_adjoint_braid_extension_formula_tracks_source_strands() {
    let codomain_permutation = [3, 0];
    let domain_permutation = [2, 1];
    let codomain_levels = [10, 30];
    let domain_levels = [20, 40];

    let lowered_codomain_permutation =
        crate::lowering::adjoint_tensor_axes(2, 2, &codomain_permutation).unwrap();
    let lowered_domain_permutation =
        crate::lowering::adjoint_tensor_axes(2, 2, &domain_permutation).unwrap();
    let (lowered_codomain_levels, lowered_domain_levels) =
        reference_adjoint_reflected_braid_levels(2, 2, &codomain_levels, &domain_levels);
    let lowered = crate::lowering::lower_tensoradd_source_operation::<2, 2>(
        TreeTransformOperation::braid(
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        ),
        true,
    )
    .unwrap();

    assert_eq!(lowered_codomain_permutation, vec![1, 2]);
    assert_eq!(lowered_domain_permutation, vec![0, 3]);
    assert_eq!(lowered_codomain_levels, vec![30, 10]);
    assert_eq!(lowered_domain_levels, vec![40, 20]);
    assert_eq!(
        lowered.into_operation(),
        TreeTransformOperation::braid([1, 2], [0, 3], [30, 10], [40, 20])
    );
    assert_ne!(
        lowered_codomain_levels,
        vec![40, 20],
        "braid levels are source-strand labels, not output tuple labels"
    );
}

#[test]
fn tensoradd_source_adjoint_braid_lowering_rejects_bad_axis_and_level_inputs() {
    let duplicate_axis = crate::lowering::lower_tensoradd_source_operation::<2, 1>(
        TreeTransformOperation::braid([2], [0, 0], [0, 2], [1]),
        true,
    )
    .unwrap_err();
    assert_eq!(
        duplicate_axis,
        OperationError::InvalidPermutation {
            axes: vec![2, 0, 0],
            rank: 3,
        }
    );

    let bad_level_count = crate::lowering::lower_tensoradd_source_operation::<2, 1>(
        TreeTransformOperation::braid([2], [0, 1], [0], [1]),
        true,
    )
    .unwrap_err();
    assert_eq!(
        bad_level_count,
        OperationError::RankMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let duplicate_level = crate::lowering::lower_tensoradd_source_operation::<2, 1>(
        TreeTransformOperation::braid([2], [0, 1], [0, 1], [1]),
        true,
    )
    .unwrap_err();
    assert_eq!(
        duplicate_level,
        OperationError::InvalidAxisSet {
            tensor: "braid level set",
            axes: vec![0, 1, 1],
            rank: 3,
        }
    );
}

#[test]
fn tensoradd_fusion_source_adjoint_explicit_braid_requires_unitary_dagger_rule() {
    let rule = UniqueAnyonicRule;
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1], [1]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1], [1]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let src = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 3.0)],
        src_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0)],
        dst_space,
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([1], [0], [0], [1]);

    let err = tensoradd_fusion_into(
        &rule,
        &mut dst,
        &src,
        operation.clone(),
        true,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTreeTransformScope {
            operation,
            message:
                "source adjoint explicit braid requires a unitary dagger-compatible braiding rule",
        }
    );
}

#[test]
fn tensoradd_fusion_source_adjoint_explicit_braid_matches_manual_inverse_braid_reference() {
    let rule = UnitaryPhaseAnyonicRule;
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 3], []),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 2>::from_dims([], [1, 1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], [3, 1]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let src_data = vec![Complex64::new(2.0, 3.0)];
    let src = TensorMap::<Complex64, 2, 0>::from_vec_with_fusion_space(src_data.clone(), src_space)
        .unwrap();
    let operation =
        TreeTransformOperation::braid(Vec::<usize>::new(), [1, 0], [0, 1], Vec::<usize>::new());
    let mut actual = TensorMap::<Complex64, 0, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero()],
        dst_space.clone(),
    )
    .unwrap();

    tensoradd_fusion_into(
        &rule,
        &mut actual,
        &src,
        operation,
        true,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    let adjoint_src_space =
        crate::lowering::adjoint_fusion_space_view(src.fusion_space().unwrap()).unwrap();
    let adjoint_src = TensorMap::<Complex64, 0, 2>::from_vec_with_fusion_space(
        src_data.iter().map(|value| value.conj()).collect(),
        adjoint_src_space,
    )
    .unwrap();
    let mut expected = TensorMap::<Complex64, 0, 2>::from_vec_with_fusion_space(
        vec![Complex64::zero()],
        dst_space,
    )
    .unwrap();
    tree_transform_into(
        &rule,
        TreeTransformOperation::braid(Vec::<usize>::new(), [1, 0], Vec::<usize>::new(), [1, 0]),
        &mut expected,
        &adjoint_src,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(actual.data(), expected.data());
}

#[test]
fn tensoradd_fusion_source_adjoint_domain_only_braid_matches_manual_inverse_braid_reference() {
    let rule = UnitaryPhaseAnyonicRule;
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 2>::from_dims([], [1, 1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], [1, 3]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([3, 1], []),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let src_data = vec![Complex64::new(-4.0, 1.5)];
    let src = TensorMap::<Complex64, 0, 2>::from_vec_with_fusion_space(src_data.clone(), src_space)
        .unwrap();
    let operation =
        TreeTransformOperation::braid([1, 0], Vec::<usize>::new(), Vec::<usize>::new(), [0, 1]);
    let mut actual = TensorMap::<Complex64, 2, 0>::from_vec_with_fusion_space(
        vec![Complex64::zero()],
        dst_space.clone(),
    )
    .unwrap();

    tensoradd_fusion_into(
        &rule,
        &mut actual,
        &src,
        operation,
        true,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    let adjoint_src_space =
        crate::lowering::adjoint_fusion_space_view(src.fusion_space().unwrap()).unwrap();
    let adjoint_src = TensorMap::<Complex64, 2, 0>::from_vec_with_fusion_space(
        src_data.iter().map(|value| value.conj()).collect(),
        adjoint_src_space,
    )
    .unwrap();
    let mut expected = TensorMap::<Complex64, 2, 0>::from_vec_with_fusion_space(
        vec![Complex64::zero()],
        dst_space,
    )
    .unwrap();
    tree_transform_into(
        &rule,
        TreeTransformOperation::braid([1, 0], Vec::<usize>::new(), [1, 0], Vec::<usize>::new()),
        &mut expected,
        &adjoint_src,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(actual.data(), expected.data());
}

#[test]
fn tensoradd_fusion_source_adjoint_mixed_braid_matches_manual_inverse_braid_reference() {
    let rule = UnitaryPhaseAnyonicRule;
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1], [1]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([SectorId::new(1)], true)]),
            FusionProductSpace::new([SectorLeg::new([SectorId::new(1)], true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let src_data = vec![Complex64::new(0.5, -2.0)];
    let src = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(src_data.clone(), src_space)
        .unwrap();
    let operation = TreeTransformOperation::braid([0], [1], [0], [1]);
    let mut actual = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero()],
        dst_space.clone(),
    )
    .unwrap();

    tensoradd_fusion_into(
        &rule,
        &mut actual,
        &src,
        operation,
        true,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    let adjoint_src_space =
        crate::lowering::adjoint_fusion_space_view(src.fusion_space().unwrap()).unwrap();
    let adjoint_src = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        src_data.iter().map(|value| value.conj()).collect(),
        adjoint_src_space,
    )
    .unwrap();
    let mut expected = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero()],
        dst_space,
    )
    .unwrap();
    tree_transform_into(
        &rule,
        TreeTransformOperation::braid([1], [0], [0], [1]),
        &mut expected,
        &adjoint_src,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(actual.data(), expected.data());
}

fn reference_adjoint_reflected_braid_levels(
    nout: usize,
    nin: usize,
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    assert_eq!(codomain_levels.len(), nout);
    assert_eq!(domain_levels.len(), nin);
    let min_level = codomain_levels
        .iter()
        .chain(domain_levels)
        .copied()
        .min()
        .unwrap();
    let max_level = codomain_levels
        .iter()
        .chain(domain_levels)
        .copied()
        .max()
        .unwrap();
    let mut lowered_codomain_levels = vec![usize::MAX; nin];
    let mut lowered_domain_levels = vec![usize::MAX; nout];

    for (source_axis, &level) in codomain_levels.iter().enumerate() {
        set_reflected_adjoint_level(
            nout,
            nin,
            source_axis,
            min_level + max_level - level,
            &mut lowered_codomain_levels,
            &mut lowered_domain_levels,
        );
    }
    for (source_domain_axis, &level) in domain_levels.iter().enumerate() {
        set_reflected_adjoint_level(
            nout,
            nin,
            nout + source_domain_axis,
            min_level + max_level - level,
            &mut lowered_codomain_levels,
            &mut lowered_domain_levels,
        );
    }

    assert!(!lowered_codomain_levels.contains(&usize::MAX));
    assert!(!lowered_domain_levels.contains(&usize::MAX));
    (lowered_codomain_levels, lowered_domain_levels)
}

fn set_reflected_adjoint_level(
    nout: usize,
    nin: usize,
    source_axis: usize,
    reflected_level: usize,
    lowered_codomain_levels: &mut [usize],
    lowered_domain_levels: &mut [usize],
) {
    let lowered_axis = crate::lowering::adjoint_tensor_axis(nout, nin, source_axis).unwrap();
    if lowered_axis < nin {
        lowered_codomain_levels[lowered_axis] = reflected_level;
    } else {
        lowered_domain_levels[lowered_axis - nin] = reflected_level;
    }
}

#[test]
fn tensoradd_structure_rejects_incompatible_shape_at_compile_time() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::filled(1.0, src_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let err =
        TensorAddStructure::compile(&dst, &src, OutputAxisOrder::from_axes(&[1, 0])).unwrap_err();

    assert_eq!(
        err,
        OperationError::ShapeMismatch {
            dst: vec![4, 5],
            src: vec![5, 4],
        }
    );
}

#[test]
fn tensoradd_structure_rejects_incompatible_replay_structure() {
    let compile_src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let compile_dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let compile_src = TensorMap::<f64, 2, 0>::filled(1.0, compile_src_space).unwrap();
    let compile_dst = TensorMap::<f64, 2, 0>::filled(0.0, compile_dst_space).unwrap();
    let structure = TensorAddStructure::compile(
        &compile_dst,
        &compile_src,
        OutputAxisOrder::from_axes(&[1, 0]),
    )
    .unwrap();

    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([5, 4], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::filled(1.0, src_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    let err = tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap_err();

    assert_eq!(err, OperationError::StructureMismatch { tensor: "dst" });
}
