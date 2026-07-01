use super::*;

#[test]
fn tensorcontract_structure_precomputes_canonical_dense_descriptor() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let lhs = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 6], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 6], rhs_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let structure = TensorContractStructure::compile(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap();

    assert_eq!(structure.dst_rank(), 2);
    assert_eq!(structure.lhs_rank(), 2);
    assert_eq!(structure.rhs_rank(), 2);
    assert_eq!(structure.lhs_contracting_axes(), &[1]);
    assert_eq!(structure.rhs_contracting_axes(), &[0]);
    assert_eq!(structure.output_axes(), &[0, 1]);
    assert_eq!(structure.terms().len(), 1);
    assert_eq!(structure.terms()[0].key(), &BlockKey::trivial());
    assert_eq!(structure.terms()[0].dst_block(), 0);
    assert_eq!(structure.terms()[0].lhs_block(), 0);
    assert_eq!(structure.terms()[0].rhs_block(), 0);
}

#[test]
fn tensorcontract_into_uses_dense_backend_for_matmul_and_alpha_beta() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let lhs =
        TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_space).unwrap();
    let rhs =
        TensorMap::<f64, 2, 0>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], rhs_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(1.0, dst_space).unwrap();

    tensorcontract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[155.0, 203.0, 209.0, 275.0]);
}

#[test]
fn tensorcontract_dense_route_uses_tensorkit_forward_tie_order() {
    let lhs_space = TensorMapSpace::<4, 0>::from_dims([5, 7, 3, 11], []).unwrap();
    let rhs_space = TensorMapSpace::<4, 0>::from_dims([13, 3, 17, 5], []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims([7, 11, 13, 17], []).unwrap();
    let lhs =
        TensorMap::<f64, 4, 0>::from_vec(vec![1.0; lhs_space.dense_dim()], lhs_space).unwrap();
    let rhs =
        TensorMap::<f64, 4, 0>::from_vec(vec![1.0; rhs_space.dense_dim()], rhs_space).unwrap();
    let dst = TensorMap::<f64, 4, 0>::filled(0.0, dst_space).unwrap();

    let structure = TensorContractStructure::compile(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[2, 0], &[1, 3]),
    )
    .unwrap();

    assert_eq!(
        structure.dense_route_kind(),
        TensorContractDenseRouteKind::ForwardSortLhsContractingAxes
    );
    assert_eq!(structure.lhs_contracting_axes(), &[2, 0]);
    assert_eq!(structure.rhs_contracting_axes(), &[1, 3]);
    let (lhs_route_axes, rhs_route_axes) = structure.dense_route_contracting_axes();
    assert_eq!(lhs_route_axes, &[0, 2]);
    assert_eq!(rhs_route_axes, &[3, 1]);
}

#[test]
fn tensorcontract_dense_route_sorted_by_rhs_matches_independent_oracle() {
    let lhs_shape = [3usize, 2, 2, 3];
    let lhs_strides = [10usize, 1, 5, 2];
    let rhs_shape = [2usize, 2, 2, 3];
    let rhs_strides = [5usize, 1, 10, 2];
    let dst_shape = [2usize, 2, 2, 3];
    let lhs_structure = dense_block_structure(&lhs_shape, &lhs_strides);
    let rhs_structure = dense_block_structure(&rhs_shape, &rhs_strides);
    let dst_structure = BlockStructure::trivial(&dst_shape).unwrap();
    let lhs_space = TensorMapSpace::<4, 0>::from_dims(lhs_shape, []).unwrap();
    let rhs_space = TensorMapSpace::<4, 0>::from_dims(rhs_shape, []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims(dst_shape, []).unwrap();
    let lhs_data = (0..strided_storage_len(&lhs_shape, &lhs_strides))
        .map(|index| 1.0 + index as f64)
        .collect::<Vec<_>>();
    let rhs_data = (0..strided_storage_len(&rhs_shape, &rhs_strides))
        .map(|index| -3.0 + 0.5 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = (0..dst_shape.iter().product::<usize>())
        .map(|index| 0.25 * index as f64 - 1.0)
        .collect::<Vec<_>>();
    let lhs =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(lhs_data.clone(), lhs_space, lhs_structure)
            .unwrap();
    let rhs =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(rhs_data.clone(), rhs_space, rhs_structure)
            .unwrap();
    let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        initial_dst.clone(),
        dst_space,
        dst_structure,
    )
    .unwrap();
    let axes =
        TensorContractAxisSpec::new(&[2, 0], &[1, 3], AxisPermutation::from_axes(&[2, 0, 3, 1]));
    let alpha = -1.25;
    let beta = 0.5;

    let structure = TensorContractStructure::compile(&dst, &lhs, &rhs, axes).unwrap();
    assert_eq!(
        structure.dense_route_kind(),
        TensorContractDenseRouteKind::ForwardSortRhsContractingAxes
    );
    let (lhs_route_axes, rhs_route_axes) = structure.dense_route_contracting_axes();
    assert_eq!(lhs_route_axes, &[2, 0]);
    assert_eq!(rhs_route_axes, &[1, 3]);

    tensorcontract_into(&mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();
    let expected = rank4_contract_oracle(
        &lhs_data,
        &lhs_shape,
        &lhs_strides,
        &rhs_data,
        &rhs_shape,
        &rhs_strides,
        &initial_dst,
        &dst_shape,
        &[2, 0, 3, 1],
        alpha,
        beta,
    );
    for (&actual, &expected) in dst.data().iter().zip(&expected) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
}

fn dense_block_structure(shape: &[usize], strides: &[usize]) -> BlockStructure {
    BlockStructure::from_blocks_with_rank(
        shape.len(),
        vec![
            BlockSpec::with_key(BlockKey::trivial(), shape.to_vec(), strides.to_vec(), 0).unwrap(),
        ],
    )
    .unwrap()
}

fn strided_storage_len(shape: &[usize], strides: &[usize]) -> usize {
    1 + shape
        .iter()
        .zip(strides)
        .map(|(&dim, &stride)| dim.saturating_sub(1) * stride)
        .sum::<usize>()
}

#[allow(clippy::too_many_arguments)]
fn rank4_contract_oracle(
    lhs_data: &[f64],
    lhs_shape: &[usize; 4],
    lhs_strides: &[usize; 4],
    rhs_data: &[f64],
    rhs_shape: &[usize; 4],
    rhs_strides: &[usize; 4],
    initial_dst: &[f64],
    dst_shape: &[usize; 4],
    output_axes: &[usize; 4],
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    assert_eq!(lhs_shape[2], rhs_shape[1]);
    assert_eq!(lhs_shape[0], rhs_shape[3]);
    let mut out = vec![0.0; initial_dst.len()];
    for d3 in 0..dst_shape[3] {
        for d2 in 0..dst_shape[2] {
            for d1 in 0..dst_shape[1] {
                for d0 in 0..dst_shape[0] {
                    let dst_coords = [d0, d1, d2, d3];
                    let mut canonical = [0usize; 4];
                    for (dst_axis, &canonical_axis) in output_axes.iter().enumerate() {
                        canonical[canonical_axis] = dst_coords[dst_axis];
                    }
                    let mut sum = 0.0;
                    for c1 in 0..lhs_shape[0] {
                        for c0 in 0..lhs_shape[2] {
                            let lhs_coords = [c1, canonical[0], c0, canonical[1]];
                            let rhs_coords = [canonical[2], c0, canonical[3], c1];
                            sum += lhs_data[strided_offset(&lhs_coords, lhs_strides)]
                                * rhs_data[strided_offset(&rhs_coords, rhs_strides)];
                        }
                    }
                    let dst_index =
                        (((d3 * dst_shape[2] + d2) * dst_shape[1] + d1) * dst_shape[0]) + d0;
                    out[dst_index] = beta * initial_dst[dst_index] + alpha * sum;
                }
            }
        }
    }
    out
}

fn strided_offset(coords: &[usize; 4], strides: &[usize; 4]) -> usize {
    coords
        .iter()
        .zip(strides)
        .map(|(&coord, &stride)| coord * stride)
        .sum()
}

#[derive(Default)]
struct MatmulOnlyDenseExecutor {
    matmul_into_calls: usize,
}

impl DenseExecutor for MatmulOnlyDenseExecutor {
    fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("rank-2 matmul boundary does not call svd")
    }

    fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("rank-2 matmul boundary does not call qr")
    }

    fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("rank-2 matmul boundary does not call eigh")
    }

    fn dot_general_into(
        &mut self,
        _output: DenseWrite<'_>,
        _lhs: DenseRead<'_>,
        _rhs: DenseRead<'_>,
        _config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("rank-2 fusion-block matmul must call matmul_into directly")
    }

    fn matmul_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        self.matmul_into_calls += 1;
        let (mut output, lhs, rhs) = match (output, lhs, rhs) {
            (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                (output, lhs, rhs)
            }
            _ => panic!("test only covers f64 canonical fusion block matmul"),
        };
        assert_eq!(lhs.shape(), &[2, 3]);
        assert_eq!(lhs.strides(), &[1, 2]);
        assert_eq!(rhs.shape(), &[3, 2]);
        assert_eq!(rhs.strides(), &[1, 3]);
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.strides(), &[1, 2]);
        output
            .data_mut()
            .copy_from_slice(&[76.0, 100.0, 103.0, 136.0]);
        Ok(())
    }
}

#[test]
fn tensorcontract_backend_rank2_matmul_uses_matmul_boundary_not_descriptor_replay() {
    let mut backend = DenseTreeTransformOperations::new(MatmulOnlyDenseExecutor::default());
    let mut workspace = TensorContractWorkspace::<f64>::default();
    let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut dst = vec![0.0; 4];

    <DenseTreeTransformOperations<MatmulOnlyDenseExecutor> as TensorContractBackend<
        f64,
        f64,
    >>::matmul_rank2_into_raw(&mut backend, &mut workspace, &mut dst, &lhs, &rhs, 2, 3, 2)
    .unwrap();

    assert_eq!(backend.dense().matmul_into_calls, 1);
    assert_eq!(dst, vec![76.0, 100.0, 103.0, 136.0]);
}

#[test]
fn tensorcontract_execution_context_replays_without_recompiling() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let lhs =
        TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_space).unwrap();
    let rhs =
        TensorMap::<f64, 2, 0>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], rhs_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(1.0, dst_space).unwrap();
    let mut context = TensorContractExecutionContext::<f64>::default();

    tensorcontract_into_with_context(
        &mut context,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[155.0, 203.0, 209.0, 275.0]);
    assert_eq!(context.cache().structure_len(), 1);
    assert_eq!(context.cache().stats().structure_hits(), 0);
    assert_eq!(context.cache().stats().structure_misses(), 1);

    dst.data_mut().fill(1.0);
    tensorcontract_into_with_context(
        &mut context,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[155.0, 203.0, 209.0, 275.0]);
    assert_eq!(context.cache().structure_len(), 1);
    assert_eq!(context.cache().stats().structure_hits(), 1);
    assert_eq!(context.cache().stats().structure_misses(), 1);
}

#[test]
fn tensorcontract_execution_context_keys_conjugation_flags() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let lhs =
        TensorMap::<Complex64, 2, 0>::from_vec(vec![Complex64::new(1.0, 1.0)], lhs_space).unwrap();
    let rhs =
        TensorMap::<Complex64, 2, 0>::from_vec(vec![Complex64::new(2.0, 3.0)], rhs_space).unwrap();
    let mut dst =
        TensorMap::<Complex64, 2, 0>::filled(Complex64::new(0.0, 0.0), dst_space).unwrap();
    let mut context = TensorContractExecutionContext::<Complex64>::default();

    tensorcontract_into_with_context(
        &mut context,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    assert_eq!(dst.data(), &[Complex64::new(-1.0, 5.0)]);

    tensorcontract_into_with_context(
        &mut context,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical_with_conjugation(&[1], &[0], true, false),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(dst.data(), &[Complex64::new(5.0, 1.0)]);
    assert_eq!(context.cache().structure_len(), 2);
    assert_eq!(context.cache().stats().structure_hits(), 0);
    assert_eq!(context.cache().stats().structure_misses(), 2);
}

#[test]
fn tensorcontract_structure_honors_output_permutation_with_workspace_scatter() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([4, 3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let lhs =
        TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_space).unwrap();
    let rhs =
        TensorMap::<f64, 2, 0>::from_vec((7..=18).map(|value| value as f64).collect(), rhs_space)
            .unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
    let structure = TensorContractStructure::compile(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::new(&[1], &[1], AxisPermutation::from_axes(&[1, 0])),
    )
    .unwrap();
    let mut backend =
        DenseTreeTransformOperations::new(ContractLayoutCheckingDenseExecutor::default());
    let mut workspace = TensorContractWorkspace::default();

    tensorcontract_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(backend.dense().dot_general_into_calls, 1);
    assert_eq!(
        dst.data(),
        &[115.0, 124.0, 133.0, 142.0, 148.0, 160.0, 172.0, 184.0]
    );
    assert_eq!(workspace.output_len(), 8);
}

#[test]
fn tensorcontract_with_conjugation_matches_dense_reference_with_output_permutation() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let lhs = TensorMap::<Complex64, 2, 0>::from_vec(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(4.0, -3.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 2, 0>::from_vec(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(8.0, 2.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst =
        TensorMap::<Complex64, 2, 0>::filled(Complex64::new(0.0, 0.0), dst_space).unwrap();

    tensorcontract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::new_with_conjugation(
            &[1],
            &[0],
            AxisPermutation::from_axes(&[1, 0]),
            true,
            false,
        ),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(23.0, -16.0),
            Complex64::new(31.0, -21.0),
            Complex64::new(33.0, 23.0),
            Complex64::new(44.0, 31.0),
        ]
    );

    tensorcontract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::new_with_conjugation(
            &[1],
            &[0],
            AxisPermutation::from_axes(&[1, 0]),
            false,
            true,
        ),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(23.0, 16.0),
            Complex64::new(31.0, 21.0),
            Complex64::new(33.0, -23.0),
            Complex64::new(44.0, -31.0),
        ]
    );
}

#[test]
fn tensorcontract_dense_backend_covers_all_gemm_dtypes() {
    assert_tensorcontract_scalar_dtype(2.0_f32, 3.0_f32, 5.0_f32, 27.0_f32);
    assert_tensorcontract_scalar_dtype(2.0_f64, 3.0_f64, 5.0_f64, 27.0_f64);
    assert_tensorcontract_scalar_dtype(
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(5.0, 0.0),
        Complex32::new(27.0, 0.0),
    );
    assert_tensorcontract_scalar_dtype(
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(5.0, 0.0),
        Complex64::new(27.0, 0.0),
    );
}

#[test]
fn tensorproduct_into_is_checked_no_contraction_wrapper() {
    let lhs_space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    let rhs_space = TensorMapSpace::<1, 0>::from_dims([3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let lhs = TensorMap::<f64, 1, 0>::from_vec(vec![2.0, 3.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 0>::from_vec(vec![5.0, 7.0, 11.0], rhs_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(1.0, dst_space).unwrap();

    tensorproduct_into(&mut dst, &lhs, &rhs, AxisPermutation::identity(), 2.0, 3.0).unwrap();

    assert_eq!(dst.data(), &[23.0, 33.0, 31.0, 45.0, 47.0, 69.0]);
}

#[test]
fn tensorproduct_with_conjugation_is_empty_contract_wrapper() {
    let lhs_space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    let rhs_space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let lhs = TensorMap::<Complex64, 1, 0>::from_vec(
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -3.0)],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 0>::from_vec(
        vec![Complex64::new(4.0, 2.0), Complex64::new(5.0, -1.0)],
        rhs_space,
    )
    .unwrap();
    let mut dst =
        TensorMap::<Complex64, 2, 0>::filled(Complex64::new(0.0, 0.0), dst_space).unwrap();

    tensorproduct_into_with_conjugation(
        &mut dst,
        &lhs,
        &rhs,
        AxisPermutation::identity(),
        true,
        false,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(6.0, -2.0),
            Complex64::new(2.0, 16.0),
            Complex64::new(4.0, -6.0),
            Complex64::new(13.0, 13.0),
        ]
    );
}

#[test]
fn tensorcontract_weighted_terms_support_all_gemm_dtypes() {
    assert_weighted_tensorcontract_scalar_dtype(2.0_f32, 3.0_f32, 5.0_f32, 21.0_f32);
    assert_weighted_tensorcontract_scalar_dtype(2.0_f64, 3.0_f64, 5.0_f64, 21.0_f64);
    assert_weighted_tensorcontract_scalar_dtype(
        Complex32::new(2.0, 1.0),
        Complex32::new(3.0, -1.0),
        Complex32::new(5.0, 2.0),
        Complex32::new(22.0, 7.0),
    );
    assert_weighted_tensorcontract_scalar_dtype(
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(5.0, 2.0),
        Complex64::new(22.0, 7.0),
    );
}

#[test]
fn tensorcontract_structure_rejects_invalid_axes_at_compile_time() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let lhs = TensorMap::<f64, 2, 0>::filled(1.0, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 2, 0>::filled(1.0, rhs_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let duplicate = TensorContractStructure::compile(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1, 1], &[0, 1]),
    )
    .unwrap_err();
    assert_eq!(
        duplicate,
        OperationError::InvalidAxisSet {
            tensor: "lhs",
            axes: vec![1, 1],
            rank: 2,
        }
    );

    let count = TensorContractStructure::compile(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0, 1]),
    )
    .unwrap_err();
    assert_eq!(
        count,
        OperationError::ContractAxisCountMismatch { lhs: 1, rhs: 2 }
    );
}

#[test]
fn tensorcontract_structure_rejects_dimension_and_output_mismatch_at_compile_time() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 4], []).unwrap();
    let lhs = TensorMap::<f64, 2, 0>::filled(1.0, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 2, 0>::filled(1.0, rhs_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let contracted_dim = TensorContractStructure::compile(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[1]),
    )
    .unwrap_err();
    assert_eq!(
        contracted_dim,
        OperationError::ShapeMismatch {
            dst: vec![3],
            src: vec![2],
        }
    );

    let wrong_dst_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let wrong_dst = TensorMap::<f64, 2, 0>::filled(0.0, wrong_dst_space).unwrap();
    let valid_rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let valid_rhs = TensorMap::<f64, 2, 0>::filled(1.0, valid_rhs_space).unwrap();
    let output = TensorContractStructure::compile(
        &wrong_dst,
        &lhs,
        &valid_rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap_err();
    assert_eq!(
        output,
        OperationError::ShapeMismatch {
            dst: vec![4, 2],
            src: vec![2, 2],
        }
    );
}

#[test]
fn tensorcontract_structure_rejects_incompatible_replay_structure_before_dense_execution() {
    let compile_lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let compile_rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let compile_dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let compile_lhs = TensorMap::<f64, 2, 0>::filled(1.0, compile_lhs_space).unwrap();
    let compile_rhs = TensorMap::<f64, 2, 0>::filled(1.0, compile_rhs_space).unwrap();
    let compile_dst = TensorMap::<f64, 2, 0>::filled(0.0, compile_dst_space).unwrap();
    let structure = TensorContractStructure::compile(
        &compile_dst,
        &compile_lhs,
        &compile_rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap();

    let lhs_space = TensorMapSpace::<2, 0>::from_dims([4, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let lhs = TensorMap::<f64, 2, 0>::filled(1.0, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 2, 0>::filled(1.0, rhs_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
    let mut backend = DenseTreeTransformOperations::new(PanicDenseExecutor);
    let mut workspace = TensorContractWorkspace::default();

    let err = tensorcontract_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap_err();

    assert_eq!(err, OperationError::StructureMismatch { tensor: "dst" });
}

#[test]
fn tensorcontract_structure_rejects_multiblock_until_block_sparse_enumeration_exists() {
    let dense = BlockStructure::trivial(&[2, 2]).unwrap();
    let multiblock = BlockStructure::packed_column_major(2, [vec![1, 2], vec![1, 2]]).unwrap();

    let err = TensorContractStructure::compile_structures(
        &dense,
        &multiblock,
        &dense,
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message: "block-sparse contraction enumeration is not implemented yet",
        }
    );
}

#[test]
fn plain_tensorcontract_rejects_one_block_fusion_tensor_instead_of_dense_contract() {
    let rule = FermionParityFusionRule;
    let odd = SectorId::new(1);
    let hom = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([odd], false)]),
            FusionProductSpace::new([SectorLeg::new([odd], false)]),
        )
    };
    let space = || {
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            hom(),
            &rule,
            [vec![1, 1]],
        )
        .unwrap()
    };
    let lhs: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![2.0], space()).unwrap();
    let rhs: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![3.0], space()).unwrap();
    let mut dst: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![10.0], space()).unwrap();

    let err = tensorcontract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        1.0,
        0.0,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message:
                "plain tensorcontract does not lower fusion-tree blocks; use tensorcontract_fusion_*"
        }
    );
    assert_eq!(dst.data(), &[10.0]);
}

#[test]
fn tensorcontract_structure_replays_explicit_block_terms_and_applies_beta_once() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let lhs_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (BlockKey::sector_ids([10]), vec![1, 2]),
            (BlockKey::sector_ids([20]), vec![1, 2]),
        ],
    )
    .unwrap();
    let rhs_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (BlockKey::sector_ids([30]), vec![2, 1]),
            (BlockKey::sector_ids([40]), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [(BlockKey::sector_ids([99]), vec![1, 1])],
    )
    .unwrap();
    let lhs = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        lhs_space,
        lhs_structure,
    )
    .unwrap();
    let rhs = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![5.0, 6.0, 7.0, 8.0],
        rhs_space,
        rhs_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![10.0], dst_space, dst_structure)
            .unwrap();
    let structure = TensorContractStructure::compile_with_block_specs(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        &[
            TensorContractBlockSpec::with_coefficient(0, 0, 0, 0.5),
            TensorContractBlockSpec::with_coefficient(0, 1, 1, 2.0),
        ],
    )
    .unwrap();

    tensorcontract_execute_with(
        &mut DenseTreeTransformOperations::default_executor(),
        &mut TensorContractWorkspace::default(),
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(structure.terms().len(), 2);
    assert_eq!(dst.data(), &[259.0]);
}

#[test]
fn tensorcontract_structure_rejects_invalid_explicit_block_term_at_compile_time() {
    let dense = BlockStructure::trivial(&[1, 1]).unwrap();

    let err = TensorContractStructure::compile_structures_with_block_specs(
        &dense,
        &dense,
        &dense,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        &[TensorContractBlockSpec::new(0, 1, 0)],
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::BlockIndexOutOfBounds {
            tensor: "lhs",
            index: 1,
            count: 1,
        }
    );
}
