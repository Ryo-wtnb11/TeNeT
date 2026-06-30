use super::*;

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
        AxisPermutation::from_axes(&[1, 0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[32.0, 36.0, 40.0, 34.0, 38.0, 42.0]);
}

#[test]
fn tensoradd_structure_precomputes_permutation_pairing_and_descriptor() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 6], src_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let structure =
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();

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
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
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
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
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
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
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
    let src_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (BlockKey::sector_ids([10]), vec![2, 3]),
            (BlockKey::sector_ids([20]), vec![1, 4]),
        ],
    )
    .unwrap();
    let dst_structure = BlockStructure::packed_column_major_with_keys(
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
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
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
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[0, 0])).unwrap_err();

    assert_eq!(
        err,
        OperationError::InvalidPermutation {
            axes: vec![0, 0],
            rank: 2,
        }
    );
}

#[test]
fn tensoradd_structure_rejects_incompatible_shape_at_compile_time() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::filled(1.0, src_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let err =
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap_err();

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
        AxisPermutation::from_axes(&[1, 0]),
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
