use super::*;

fn ensure_recoupling_coefficients<D, C>(
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    structure_identity: &Arc<()>,
) -> Result<bool, OperationError>
where
    D: RecouplingCoefficientAction<C>,
    C: Copy,
{
    let plan = task.recoupling_plan();
    // Why not key by the shared categorical payload alone: different layout
    // bindings can reorder Multi jobs by element count and therefore require
    // different packed RHS orders for the same categorical matrices.
    let same_structure = workspace
        .coefficient_structure_identity
        .as_ref()
        .and_then(Weak::upgrade)
        .is_some_and(|identity| Arc::ptr_eq(&identity, structure_identity));
    if same_structure && workspace.coefficient_scratch.len() == plan.coefficient_len() {
        return Ok(false);
    }

    workspace.coefficient_scratch.clear();
    workspace
        .coefficient_scratch
        .reserve(plan.coefficient_len());
    // Preserve entry order: each job's rhs_offset addresses this exact pack.
    for (block_index, _) in plan.entries() {
        let block = recoupling_multi_block(task, block_index)?;
        let TreeTransformBlock::Multi {
            dst_count,
            src_count,
            coefficient_start,
            ..
        } = *block
        else {
            continue;
        };
        let coefficient_len = src_count
            .checked_mul(dst_count)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        let coefficient_end = coefficient_start
            .checked_add(coefficient_len)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        let coefficients = task
            .coefficients()
            .get(coefficient_start..coefficient_end)
            .ok_or_else(|| OperationError::CoefficientCountMismatch {
                expected: coefficient_end,
                actual: task.coefficients().len(),
            })?;
        workspace.coefficient_scratch.extend(
            coefficients
                .iter()
                .map(|&coefficient| D::coefficient_as_data(coefficient)),
        );
    }
    if workspace.coefficient_scratch.len() != plan.coefficient_len() {
        return Err(OperationError::CoefficientCountMismatch {
            expected: plan.coefficient_len(),
            actual: workspace.coefficient_scratch.len(),
        });
    }
    workspace.coefficient_structure_identity = Some(Arc::downgrade(structure_identity));
    Ok(true)
}

fn recoupling_multi_block<C: Copy>(
    task: TreeTransformTaskView<'_, C>,
    block_index: usize,
) -> Result<&TreeTransformBlock, OperationError> {
    // Lazy error construction: recoupling_multi_block is called per block on the
    // hot replay path (pack/recouple/scatter). Eager .ok_or built the
    // BlockIndexOutOfBounds struct on every success too, which the d=4 bisect
    // (see issue #103) attributed to the compose regression. .ok_or_else only
    // builds it on the never-taken out-of-bounds path.
    let block =
        task.blocks()
            .get(block_index)
            .ok_or_else(|| OperationError::BlockIndexOutOfBounds {
                tensor: "recoupling block",
                index: block_index,
                count: task.blocks().len(),
            })?;
    match block {
        TreeTransformBlock::Multi { .. } => Ok(block),
        TreeTransformBlock::Single { .. } => Err(OperationError::BlockIndexOutOfBounds {
            tensor: "recoupling block",
            index: block_index,
            count: task.blocks().len(),
        }),
    }
}

fn scale_inactive_destinations<A, D, C>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    task: TreeTransformTaskView<'_, C>,
    dst_data: &mut [D],
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + PartialEq + Zero + One,
    C: Copy,
{
    match mode {
        DestinationMode::Axpby(beta) => {
            if beta == D::one() {
                return Ok(());
            }
            // Scaling the complete storage would also mutate padding not owned by any
            // block, so compile only the destination layouts with no active replay.
            for &layout_index in task.inactive_destination_layouts() {
                let layout = task.layouts().entry(layout_index);
                kernels.scale_strided(
                    dst_data,
                    task.layouts().shape(layout),
                    task.layouts().strides(layout),
                    layout.offset,
                    beta,
                )?;
            }
        }
        DestinationMode::Overwrite => {
            let zero = [D::zero()];
            for &layout_index in task.inactive_destination_layouts() {
                let layout = task.layouts().entry(layout_index);
                zero_strides.clear();
                zero_strides.resize(task.layouts().shape(layout).len(), 0);
                kernels.copy_scale_strided(
                    dst_data,
                    &zero,
                    task.layouts().shape(layout),
                    task.layouts().strides(layout),
                    zero_strides,
                    layout.offset,
                    0,
                    false,
                    D::one(),
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod coefficient_cache_tests {
    use super::*;
    use crate::{TreeTransformBlockSpec, TreeTransformGroupBlockSpec, TreeTransformGroupPlan};
    use tenet_core::{BlockKey, BlockSpec, FusionTreePairKey};

    fn multi_recoupling_structure(coefficients: [f64; 4]) -> TreeTransformStructure<f64> {
        let block_structure = BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap();
        TreeTransformStructure::compile_structures(
            &block_structure,
            &block_structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                coefficients.to_vec(),
            )],
        )
        .unwrap()
    }

    fn group_pair(group: usize, vertex: usize) -> FusionTreePairKey {
        FusionTreePairKey::try_pair_from_sector_ids(
            [group, group],
            [],
            group,
            [false, false],
            [],
            [],
            [],
            [vertex],
            [],
        )
        .unwrap()
    }

    fn two_group_layout(keys: &[FusionTreePairKey; 4], elements: [usize; 2]) -> BlockStructure {
        let mut offset = 0;
        BlockStructure::from_blocks_with_rank(
            2,
            keys.iter()
                .enumerate()
                .map(|(index, key)| {
                    let block_elements = elements[index / 2];
                    let block = BlockSpec::column_major_with_key(
                        BlockKey::from(key.clone()),
                        vec![block_elements, 1],
                        offset,
                    )
                    .unwrap();
                    offset += block_elements;
                    block
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn recoupling_coefficients_cache_uses_live_structure_identity() {
        let structure = multi_recoupling_structure([1.0, 2.0, 3.0, 4.0]);
        let mut workspace = TreeTransformWorkspace::<f64>::default();

        assert!(ensure_recoupling_coefficients(
            &mut workspace,
            structure.task_view().unwrap(),
            structure.identity_marker(),
        )
        .unwrap());
        assert_eq!(workspace.coefficient_scratch, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(!ensure_recoupling_coefficients(
            &mut workspace,
            structure.task_view().unwrap(),
            structure.identity_marker(),
        )
        .unwrap());

        let structure_clone = structure.clone();
        assert!(!ensure_recoupling_coefficients(
            &mut workspace,
            structure_clone.task_view().unwrap(),
            structure_clone.identity_marker(),
        )
        .unwrap());

        let equal_but_distinct = multi_recoupling_structure([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(structure, equal_but_distinct);
        workspace.coefficient_scratch.fill(-1.0);
        assert!(ensure_recoupling_coefficients(
            &mut workspace,
            equal_but_distinct.task_view().unwrap(),
            equal_but_distinct.identity_marker(),
        )
        .unwrap());
        assert_eq!(workspace.coefficient_scratch, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn categorical_payload_is_repacked_when_layout_reorders_multi_jobs() {
        // What: structure identity protects a layout-packed execution buffer;
        // categorical payload identity alone would incorrectly reuse [UA, UB].
        let keys = [
            group_pair(0, 1),
            group_pair(0, 2),
            group_pair(1, 1),
            group_pair(1, 2),
        ];
        let plan = TreeTransformGroupPlan::new(vec![
            TreeTransformGroupBlockSpec::try_multi(
                keys[..2].to_vec(),
                keys[..2].to_vec(),
                vec![1.0, 2.0, 3.0, 4.0],
            )
            .unwrap(),
            TreeTransformGroupBlockSpec::try_multi(
                keys[2..].to_vec(),
                keys[2..].to_vec(),
                vec![5.0, 6.0, 7.0, 8.0],
            )
            .unwrap(),
        ]);
        let a_then_b = two_group_layout(&keys, [1, 2]);
        let b_then_a = two_group_layout(&keys, [3, 1]);
        let first = plan.compile_structures(&a_then_b, &a_then_b).unwrap();
        let second = plan.compile_structures(&b_then_a, &b_then_a).unwrap();
        assert!(first.shares_coefficient_payload_with(&second));
        let mut workspace = TreeTransformWorkspace::<f64>::default();

        assert!(ensure_recoupling_coefficients(
            &mut workspace,
            first.task_view().unwrap(),
            first.identity_marker(),
        )
        .unwrap());
        assert_eq!(
            workspace.coefficient_scratch,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert!(ensure_recoupling_coefficients(
            &mut workspace,
            second.task_view().unwrap(),
            second.identity_marker(),
        )
        .unwrap());
        assert_eq!(
            workspace.coefficient_scratch,
            vec![5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]
        );
    }
}

#[cfg(test)]
mod inactive_destination_tests {
    use super::*;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use std::time::Duration;
    use tenet_core::{BlockKey, BlockSpec, TensorMapSpace, Trivial};
    use tenet_dense::{
        DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
        DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseWrite,
    };

    type TestTensor = TensorMap<f64, 1, 0, Trivial, Vec<f64>>;

    fn fixture() -> (TestTensor, TestTensor, TreeTransformStructure<f64>) {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap();
        let src: TestTensor = TensorMap::from_vec_with_structure(
            vec![3.0],
            TensorMapSpace::from_dims([1], []).unwrap(),
            src_structure,
        )
        .unwrap();
        let dst = TensorMap::from_vec_with_structure(
            vec![10.0, 20.0],
            TensorMapSpace::from_dims([2], []).unwrap(),
            dst_structure,
        )
        .unwrap();
        let structure = TreeTransformStructure::compile(
            &dst,
            &src,
            &[TreeTransformBlockSpec::single(0, 0, 2.0)],
        )
        .unwrap();
        (dst, src, structure)
    }

    fn expected(beta: f64) -> [f64; 2] {
        [6.0 + beta * 10.0, beta * 20.0]
    }

    #[test]
    fn finite_task_view_serial_matches_per_block_replay() -> Result<(), OperationError> {
        let block_structure =
            Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, 2]; 4]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &block_structure,
            &block_structure,
            &[
                TreeTransformBlockSpec::single(0, 0, -2.0).with_source_axes([1, 0]),
                TreeTransformBlockSpec::multi(vec![1, 2], vec![1, 2], vec![1.0, 2.0, 3.0, 4.0])
                    .with_source_axes([1, 0]),
            ],
        )
        .unwrap();
        let source = (1..=16).map(f64::from).collect::<Vec<_>>();

        for overwrite in [false, true] {
            let mut expected = (17..=32).map(f64::from).collect::<Vec<_>>();
            let mut actual = expected.clone();
            let mut kernels = StridedHostKernelAdapter::default();
            if overwrite {
                tree_transform_structure_overwrite_with_strided_kernel_raw(
                    &mut kernels,
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &block_structure,
                    &block_structure,
                    &mut expected,
                    &source,
                    0.5,
                )?;
                tree_transform_structure_overwrite_with_structural_recoupling_raw(
                    &mut kernels,
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &block_structure,
                    &block_structure,
                    &mut actual,
                    &source,
                    0.5,
                    1,
                )?;
            } else {
                tree_transform_structure_with_strided_kernel_raw(
                    &mut kernels,
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &block_structure,
                    &block_structure,
                    &mut expected,
                    &source,
                    0.5,
                    -0.25,
                )?;
                tree_transform_structure_with_structural_recoupling_raw(
                    &mut kernels,
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &block_structure,
                    &block_structure,
                    &mut actual,
                    &source,
                    0.5,
                    -0.25,
                    1,
                )?;
            }
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    fn custom_structure(blocks: Vec<BlockSpec>) -> BlockStructure {
        BlockStructure::from_blocks_with_rank(1, blocks).unwrap()
    }

    fn block(sector: usize, shape: usize, stride: usize, offset: usize) -> BlockSpec {
        BlockSpec::with_key(BlockKey::ordinal(sector), vec![shape], vec![stride], offset).unwrap()
    }

    fn identity_multi_fixture() -> (
        Arc<BlockStructure>,
        Arc<BlockStructure>,
        TreeTransformStructure<f64>,
    ) {
        let src = Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
        let dst =
            Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1]]).unwrap());
        let replay = TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();
        (dst, src, replay)
    }

    struct FailFirstBatchExecutor {
        inner: DefaultDenseExecutor,
        fail_next: bool,
    }

    impl FailFirstBatchExecutor {
        fn new() -> Self {
            Self {
                inner: DefaultDenseExecutor::new(),
                fail_next: true,
            }
        }
    }

    impl DenseExecutor for FailFirstBatchExecutor {
        fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.svd(input)
        }

        fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.qr(input)
        }

        fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.eigh(input)
        }

        fn dot_general_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            self.inner.dot_general_into(output, lhs, rhs, config)
        }

        fn matmul_batch_axpby_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            jobs: &[DenseGemmBatchJob],
            runs: &[usize],
            alpha: DenseScalar,
            beta: DenseScalar,
        ) -> Result<(), DenseError> {
            if self.fail_next {
                self.fail_next = false;
                match output {
                    DenseWrite::F64(mut output) => output.data_mut().fill(f64::NAN),
                    _ => unreachable!("retry oracle uses f64"),
                }
                return Err(DenseError::Backend {
                    backend: DenseBackend::Tenferro,
                    op: "matmul_batch_axpby_into",
                    message: "injected first-call failure".to_string(),
                });
            }
            self.inner
                .matmul_batch_axpby_into(output, lhs, rhs, jobs, runs, alpha, beta)
        }
    }

    #[test]
    fn compile_rejects_inactive_destination_aliases() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 0)]);
        assert_eq!(
            TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[],)
                .unwrap_err(),
            OperationError::InvalidArgument {
                message: "tree transform destination layouts overlap"
            }
        );
    }

    #[test]
    fn compile_rejects_active_inactive_destination_aliases() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 0)]);
        assert_eq!(
            TreeTransformStructure::compile_structures(
                &dst_structure,
                &src_structure,
                &[TreeTransformBlockSpec::single(0, 0, 1.0)],
            )
            .unwrap_err(),
            OperationError::InvalidArgument {
                message: "tree transform destination layouts overlap"
            }
        );
    }

    #[test]
    fn compile_rejects_active_destination_aliases() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 0)]);
        assert_eq!(
            TreeTransformStructure::compile_structures(
                &dst_structure,
                &src_structure,
                &[
                    TreeTransformBlockSpec::single(0, 0, 1.0),
                    TreeTransformBlockSpec::single(1, 1, 1.0),
                ],
            )
            .unwrap_err(),
            OperationError::InvalidArgument {
                message: "tree transform destination layouts overlap"
            }
        );
    }

    #[test]
    fn compile_rejects_self_overlapping_destination_layout() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 2, 0, 0)]);
        assert_eq!(
            TreeTransformStructure::compile_structures(
                &dst_structure,
                &src_structure,
                &[TreeTransformBlockSpec::single(0, 0, 1.0)],
            )
            .unwrap_err(),
            OperationError::InvalidArgument {
                message: "tree transform destination layouts overlap"
            }
        );
    }

    #[test]
    fn compile_rejects_nonzero_stride_self_overlap() {
        let src_structure = BlockStructure::packed_column_major(2, [vec![2, 2]]).unwrap();
        let dst_structure = BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::with_key(BlockKey::opaque([0]), vec![2, 2], vec![1, 1], 0).unwrap()],
        )
        .unwrap();
        assert_eq!(
            TreeTransformStructure::compile_structures(
                &dst_structure,
                &src_structure,
                &[TreeTransformBlockSpec::single(0, 0, 1.0)],
            )
            .unwrap_err(),
            OperationError::InvalidArgument {
                message: "tree transform destination layouts overlap",
            }
        );
    }

    #[test]
    fn interleaved_disjoint_destinations_with_overlapping_ranges_are_valid() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 2, 2, 0), block(1, 2, 2, 1)]);
        let structure =
            TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[])
                .unwrap();
        let mut dst = vec![10.0, 20.0, 30.0, 40.0];
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::new(dst_structure),
            &Arc::new(src_structure),
            &mut dst,
            &[3.0],
            1.0,
            0.5,
            4,
        )
        .unwrap();
        assert_eq!(dst, [5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn threaded_active_interleaved_layout_uses_the_serial_fallback() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 2, 2, 0), block(1, 2, 2, 1)]);
        let structure = TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[
                TreeTransformBlockSpec::single(0, 0, 2.0),
                TreeTransformBlockSpec::single(1, 1, -1.0),
            ],
        )
        .unwrap();
        assert!(!structure.parallel_schedule().singles_slice_disjoint);
        let src = [1.0, 2.0, 3.0, 4.0];
        let mut serial = [10.0, 20.0, 30.0, 40.0];
        let mut threaded = serial;
        let dst_structure = Arc::new(dst_structure);
        let src_structure = Arc::new(src_structure);
        for (dst, threads) in [(&mut serial[..], 1), (&mut threaded[..], 4)] {
            tree_transform_structure_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                dst,
                &src,
                1.0,
                0.5,
                threads,
            )
            .unwrap();
        }

        assert_eq!(threaded, serial);
        assert_eq!(threaded, [7.0, 7.0, 19.0, 16.0]);
    }

    #[test]
    fn threaded_replay_handles_rank_zero_blocks() {
        let structure = Arc::new(
            BlockStructure::packed_column_major(0, [Vec::<usize>::new(), Vec::new()]).unwrap(),
        );
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[
                TreeTransformBlockSpec::single(0, 0, 2.0),
                TreeTransformBlockSpec::single(1, 1, -1.0),
            ],
        )
        .unwrap();
        let src = [3.0, 5.0];
        let mut serial = [10.0, 20.0];
        let mut threaded = serial;
        for (dst, threads) in [(&mut serial[..], 1), (&mut threaded[..], 4)] {
            tree_transform_structure_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                dst,
                &src,
                1.0,
                0.5,
                threads,
            )
            .unwrap();
        }
        assert_eq!(threaded, serial);
        assert_eq!(threaded, [11.0, 5.0]);
    }

    #[test]
    fn threaded_replay_ignores_zero_extent_work() {
        let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![0]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        assert!(transform.parallel_schedule().singles.is_empty());
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut [],
            &[],
            1.0,
            f64::NAN,
            4,
        )
        .unwrap();
    }

    #[test]
    fn zero_extent_profile_counts_match_serial_replay() {
        let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![0]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        let mut serial = TreeTransformReplayProfile::default();
        let mut threaded = TreeTransformReplayProfile::default();
        for (profile, threads) in [(&mut serial, 1), (&mut threaded, 4)] {
            tree_transform_structure_with_structural_recoupling_raw_profiled(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                &mut [],
                &[],
                1.0,
                0.0,
                threads,
                profile,
            )
            .unwrap();
        }

        assert_eq!(threaded.single_blocks, serial.single_blocks);
        assert_eq!(threaded.multi_blocks, serial.multi_blocks);
        assert_eq!(threaded.packed_columns, serial.packed_columns);
        assert_eq!(threaded.scattered_columns, serial.scattered_columns);
    }

    #[test]
    fn zero_extent_multi_profile_counts_match_serial_replay() {
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![0], vec![0]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();
        let mut serial = TreeTransformReplayProfile::default();
        let mut threaded = TreeTransformReplayProfile::default();
        for (profile, threads) in [(&mut serial, 1), (&mut threaded, 4)] {
            tree_transform_structure_with_structural_recoupling_raw_profiled(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                &mut [],
                &[],
                1.0,
                0.0,
                threads,
                profile,
            )
            .unwrap();
        }

        assert_eq!(threaded.multi_blocks, serial.multi_blocks);
        assert_eq!(threaded.packed_columns, serial.packed_columns);
        assert_eq!(threaded.scattered_columns, serial.scattered_columns);
        assert_eq!(threaded.packed_columns, 2);
        assert_eq!(threaded.scattered_columns, 2);
    }

    #[test]
    fn threaded_multi_scatter_falls_back_for_interleaved_destinations() {
        let src_structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap());
        let dst_structure = Arc::new(custom_structure(vec![block(0, 2, 2, 0), block(1, 2, 2, 1)]));
        let transform = TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();
        assert_eq!(transform.parallel_schedule().scatter_groups.len(), 1);
        assert!(!transform.parallel_schedule().scatter_groups[0].slice_disjoint);
        let src = [1.0, 2.0, 3.0, 4.0];
        let mut serial = [10.0, 20.0, 30.0, 40.0];
        let mut threaded = serial;
        for (dst, threads) in [(&mut serial[..], 1), (&mut threaded[..], 4)] {
            tree_transform_structure_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &dst_structure,
                &src_structure,
                dst,
                &src,
                1.0,
                0.5,
                threads,
            )
            .unwrap();
        }

        assert_eq!(threaded, serial);
        assert_eq!(threaded, [6.0, 13.0, 17.0, 24.0]);
    }

    #[test]
    fn many_range_connected_interleaved_destinations_are_valid() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure =
            custom_structure((0..64).map(|offset| block(offset, 2, 64, offset)).collect());
        TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[])
            .unwrap();
    }

    #[test]
    fn inactive_destination_scaling_preserves_storage_padding() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 2, 2, 0)]);
        let structure =
            TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[])
                .unwrap();
        let mut dst = vec![10.0, 99.0, 30.0];
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::new(dst_structure),
            &Arc::new(src_structure),
            &mut dst,
            &[3.0],
            1.0,
            0.5,
            4,
        )
        .unwrap();
        assert_eq!(dst, [5.0, 99.0, 15.0]);
    }

    #[test]
    fn nan_beta_reaches_active_and_inactive_destinations() {
        let (mut dst, src, structure) = fixture();
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
            f64::NAN,
            4,
        )
        .unwrap();
        assert!(dst.data().iter().all(|value| value.is_nan()));
    }

    /// What: `beta = 0` scales an inactive destination by VectorInterface's
    /// `scale(x, 0) = zero(x) * 0`, so a NaN there becomes zero rather than
    /// `0 * NaN` (#1438), as TensorKit's `scale!(tdst, β)` does.
    #[test]
    fn generic_beta_zero_wipes_nan_in_inactive_destinations() {
        let (mut dst, src, structure) = fixture();
        dst.data_mut().fill(f64::NAN);

        tree_transform_structure_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(dst.data()[0], 6.0);
        assert_eq!(dst.data()[1].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn overwrite_single_does_not_read_nan_destinations_in_any_driver() {
        for threads in [1, 2] {
            let (mut dst, src, structure) = fixture();
            dst.data_mut().fill(f64::NAN);
            tree_transform_structure_overwrite_with_structural_recoupling(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &mut dst,
                &src,
                1.0,
                threads,
            )
            .unwrap();
            assert_eq!(dst.data(), &[6.0, 0.0]);
        }

        let (mut dst, src, structure) = fixture();
        dst.data_mut().fill(f64::NAN);
        tree_transform_structure_overwrite_with_storage_workspace_strided_kernel(
            &mut StridedHostKernelAdapter::default(),
            &mut StorageTreeTransformWorkspace::<Vec<f64>, Vec<f64>>::default(),
            &structure,
            &mut dst,
            &src,
            1.0,
        )
        .unwrap();
        assert_eq!(dst.data(), &[6.0, 0.0]);

        let (mut dst, src, structure) = fixture();
        dst.data_mut().fill(f64::NAN);
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
        )
        .unwrap();
        assert_eq!(dst.data(), &[6.0, 0.0]);

        let (mut dst, src, structure) = fixture();
        dst.data_mut().fill(f64::NAN);
        tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
            1,
            &mut TreeTransformReplayProfile::default(),
        )
        .unwrap();
        assert_eq!(dst.data(), &[6.0, 0.0]);
    }

    #[test]
    fn overwrite_multi_does_not_read_nan_active_or_inactive_destinations() {
        let (dst_structure, src_structure, structure) = identity_multi_fixture();
        let mut dst = vec![f64::NAN; 3];

        tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst_structure,
            &src_structure,
            &mut dst,
            &[3.0, 4.0],
            2.0,
        )
        .unwrap();

        assert_eq!(dst, [6.0, 8.0, 0.0]);
    }

    #[test]
    fn overwrite_multi_c64_does_not_read_nan_destinations() {
        let (dst_structure, src_structure, structure) = identity_multi_fixture();
        for threads in [1, 4] {
            let nan = num_complex::Complex64::new(f64::NAN, f64::NAN);
            let mut dst = vec![nan; 3];
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut dst,
                &[
                    num_complex::Complex64::new(3.0, 1.0),
                    num_complex::Complex64::new(4.0, -1.0),
                ],
                num_complex::Complex64::new(2.0, 0.0),
                threads,
            )
            .unwrap();
            assert_eq!(
                dst,
                [
                    num_complex::Complex64::new(6.0, 2.0),
                    num_complex::Complex64::new(8.0, -2.0),
                    num_complex::Complex64::new(0.0, 0.0),
                ]
            );
        }
    }

    #[test]
    fn profiled_multi_attributes_recoupling_to_dense_gemm() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let (dst_structure, src_structure, structure) = identity_multi_fixture();
            for threads in [1, 2] {
                let mut dst = vec![f64::NAN; 3];
                let mut profile = TreeTransformReplayProfile::default();
                tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled(
                    &mut StridedHostKernelAdapter::default(),
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &structure,
                    &dst_structure,
                    &src_structure,
                    &mut dst,
                    &[3.0, 4.0],
                    2.0,
                    threads,
                    &mut profile,
                )
                .unwrap();

                assert_eq!(dst, [6.0, 8.0, 0.0]);
                assert_eq!(profile.multi_blocks, 1);
                assert!(profile.multi_dense_matmul_call > Duration::ZERO);
                assert_eq!(profile.multi_scalar_recoupling, Duration::ZERO);
                assert_eq!(
                    profile.multi_matmul_total,
                    profile.multi_dense_view_setup
                        + profile.multi_dense_matmul_call
                        + profile.multi_scalar_recoupling
                );
            }

            let (mut dst, src, structure) = fixture();
            let mut profile = TreeTransformReplayProfile::default();
            tree_transform_structure_with_structural_recoupling_raw_profiled(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &Arc::clone(dst.structure()),
                &Arc::clone(src.structure()),
                dst.data_mut(),
                src.data(),
                1.0,
                0.0,
                2,
                &mut profile,
            )
            .unwrap();
            assert_eq!(profile.multi_blocks, 0);
            assert_eq!(profile.multi_matmul_total, Duration::ZERO);
            assert_eq!(profile.multi_dense_view_setup, Duration::ZERO);
            assert_eq!(profile.multi_dense_matmul_call, Duration::ZERO);
            assert_eq!(profile.multi_scalar_recoupling, Duration::ZERO);
        });
    }

    #[test]
    fn overwrite_threaded_single_multi_and_storage_multi_ignore_destination_bits() {
        let src_structure = Arc::new(
            BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1], vec![1]]).unwrap(),
        );
        let dst_structure = Arc::new(
            BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1], vec![1], vec![1]])
                .unwrap(),
        );
        let structure = TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[
                TreeTransformBlockSpec::single(0, 0, 2.0),
                TreeTransformBlockSpec::single(1, 1, -1.0),
                TreeTransformBlockSpec::multi(vec![2, 3], vec![2, 3], vec![1.0, 0.0, 0.0, 1.0]),
            ],
        )
        .unwrap();
        let src = [3.0, 4.0, 5.0, 6.0];
        let expected = [6.0, -4.0, 5.0, 6.0, 0.0];

        for threads in [1, 4] {
            let mut dst = [f64::NAN; 5];
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut dst,
                &src,
                1.0,
                threads,
            )
            .unwrap();
            assert_eq!(dst, expected);
        }

        let src: TestTensor = TensorMap::from_vec_with_structure(
            src.to_vec(),
            TensorMapSpace::from_dims([4], []).unwrap(),
            Arc::unwrap_or_clone(src_structure),
        )
        .unwrap();
        let mut dst: TestTensor = TensorMap::from_vec_with_structure(
            vec![f64::NAN; 5],
            TensorMapSpace::from_dims([5], []).unwrap(),
            Arc::unwrap_or_clone(dst_structure),
        )
        .unwrap();
        tree_transform_structure_overwrite_with_storage_workspace_strided_kernel(
            &mut StridedHostKernelAdapter::default(),
            &mut StorageTreeTransformWorkspace::<Vec<f64>, Vec<f64>>::default(),
            &structure,
            &mut dst,
            &src,
            1.0,
        )
        .unwrap();
        assert_eq!(dst.data(), &expected);
    }

    #[test]
    fn overwrite_multi_recovers_from_dirty_packed_scratch_after_failure() {
        let (dst_structure, src_structure, replay) = identity_multi_fixture();
        let mut workspace = TreeTransformWorkspace::default();
        let mut dense = FailFirstBatchExecutor::new();
        let mut dst = [f64::NAN; 3];

        assert!(
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut dense,
                &mut workspace,
                &replay,
                &dst_structure,
                &src_structure,
                &mut dst,
                &[3.0, 4.0],
                2.0,
                1,
            )
            .is_err()
        );
        assert!(workspace
            .packed
            .destination()
            .as_slice()
            .iter()
            .all(|value| value.is_nan()));

        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut workspace,
            &replay,
            &dst_structure,
            &src_structure,
            &mut dst,
            &[3.0, 4.0],
            2.0,
            1,
        )
        .unwrap();
        assert_eq!(dst, [6.0, 8.0, 0.0]);
    }

    #[test]
    fn huge_thread_request_is_capped_by_pool_and_runnable_work() {
        // What: normal and profiled replay use the same effective worker count,
        // preserve serial beta semantics, and never size scratch from a caller's
        // unbounded thread request.
        let block_structure =
            Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, 2]; 4]).unwrap());
        let specs = (0..3)
            .map(|block| TreeTransformBlockSpec::single(block, block, 1.0).with_source_axes([1, 0]))
            .collect::<Vec<_>>();
        let transform =
            TreeTransformStructure::compile_structures(&block_structure, &block_structure, &specs)
                .unwrap();
        assert_eq!(transform.layouts().max_fused_rank(), 2);
        let source = (1..=16).map(f64::from).collect::<Vec<_>>();
        let initial = (10..=25).map(f64::from).collect::<Vec<_>>();

        let mut expected = initial.clone();
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &block_structure,
            &block_structure,
            &mut expected,
            &source,
            1.0,
            0.5,
            1,
        )
        .unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let mut destination = initial.clone();
        let mut workspace = TreeTransformWorkspace::default();
        pool.install(|| {
            tree_transform_structure_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut workspace,
                &transform,
                &block_structure,
                &block_structure,
                &mut destination,
                &source,
                1.0,
                0.5,
                usize::MAX,
            )
        })
        .unwrap();
        assert_eq!(workspace.fused_indices.len(), 3 * 2);
        assert_eq!(destination, expected);

        let mut profiled_destination = initial;
        let mut profiled_workspace = TreeTransformWorkspace::default();
        let mut profile = TreeTransformReplayProfile::default();
        pool.install(|| {
            tree_transform_structure_with_structural_recoupling_raw_profiled(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut profiled_workspace,
                &transform,
                &block_structure,
                &block_structure,
                &mut profiled_destination,
                &source,
                1.0,
                0.5,
                usize::MAX,
                &mut profile,
            )
        })
        .unwrap();
        assert_eq!(profiled_workspace.fused_indices.len(), 3 * 2);
        assert_eq!(profiled_destination, expected);
    }

    #[test]
    fn overwrite_validates_before_mutation_and_accepts_rank_boundaries() {
        let (mut dst, src, structure) = fixture();
        dst.data_mut().fill(f64::NAN);
        let before = dst.data().to_vec();
        let result = tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            &[],
            1.0,
        );
        assert!(result.is_err());
        assert_eq!(
            dst.data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        let result = tree_transform_structure_overwrite_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            &[],
            1.0,
            1,
        );
        assert!(result.is_err());
        assert_eq!(
            dst.data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        let scalar_structure =
            Arc::new(BlockStructure::packed_column_major(0, [Vec::<usize>::new()]).unwrap());
        let scalar_replay = TreeTransformStructure::compile_structures(
            &scalar_structure,
            &scalar_structure,
            &[TreeTransformBlockSpec::single(0, 0, 2.0)],
        )
        .unwrap();
        let mut scalar_dst = [f64::NAN];
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &scalar_replay,
            &scalar_structure,
            &scalar_structure,
            &mut scalar_dst,
            &[3.0],
            1.0,
        )
        .unwrap();
        assert_eq!(scalar_dst, [6.0]);

        let empty_structure = Arc::new(BlockStructure::packed_column_major(1, [vec![0]]).unwrap());
        let empty_replay = TreeTransformStructure::compile_structures(
            &empty_structure,
            &empty_structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &empty_replay,
            &empty_structure,
            &empty_structure,
            &mut [],
            &[],
            1.0,
        )
        .unwrap();
    }

    #[test]
    fn profiled_overwrite_zeros_inactive_layout_without_touching_padding() {
        let src_structure = Arc::new(BlockStructure::packed_column_major(1, [vec![1]]).unwrap());
        let dst_structure = Arc::new(custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 2)]));
        let structure = TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[TreeTransformBlockSpec::single(0, 0, 2.0)],
        )
        .unwrap();
        let mut dst = [f64::NAN, 99.0, f64::NAN];
        let mut profile = TreeTransformReplayProfile::default();

        tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst_structure,
            &src_structure,
            &mut dst,
            &[3.0],
            1.0,
            2,
            &mut profile,
        )
        .unwrap();

        assert_eq!(dst, [6.0, 99.0, 0.0]);
        assert_eq!(profile.single_blocks, 1);
    }

    #[derive(Clone, Default)]
    struct SlowScaleAdapter(StridedHostKernelAdapter);

    impl HostKernelAdapter<f64> for SlowScaleAdapter {
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
            self.0.add_strided(
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
            self.0.axpby_strided(
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
            self.0.copy_scale_strided(
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
            std::thread::sleep(Duration::from_millis(40));
            self.0
                .scale_strided(dst_data, shape, dst_strides, dst_offset, beta)
        }

        fn recoupling_src_times_u_transpose<C>(
            &mut self,
            destination: &mut [f64],
            source: &[f64],
            coefficients: &[C],
            coefficient_start: usize,
            element_count: usize,
            src_count: usize,
            dst_count: usize,
        ) -> Result<(), OperationError>
        where
            C: Copy,
            f64: RecouplingCoefficientAction<C>,
        {
            self.0.recoupling_src_times_u_transpose(
                destination,
                source,
                coefficients,
                coefficient_start,
                element_count,
                src_count,
                dst_count,
            )
        }
    }

    #[test]
    fn profiled_replay_attributes_inactive_destination_scaling() {
        let (mut dst, src, structure) = fixture();
        let mut profile = TreeTransformReplayProfile::default();
        tree_transform_structure_with_structural_recoupling_raw_profiled(
            &mut SlowScaleAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
            0.5,
            1,
            &mut profile,
        )
        .unwrap();
        let attributed = profile.validate
            + profile.single_total
            + profile.strided_kernel.saturating_sub(profile.single_total)
            + profile.multi_workspace_prepare
            + profile.multi_pack
            + profile.multi_coefficient_prepare
            + profile.multi_matmul_total
            + profile.multi_scatter;
        assert!(profile.total.saturating_sub(attributed) < Duration::from_millis(20));
    }

    #[test]
    fn structural_serial_replay_scales_inactive_destinations() {
        for beta in [0.0, 0.5, 1.0] {
            let (mut dst, src, structure) = fixture();
            tree_transform_structure_with_structural_recoupling(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &mut dst,
                &src,
                1.0,
                beta,
                1,
            )
            .unwrap();
            assert_eq!(dst.data(), &expected(beta));
        }
    }

    #[test]
    fn structural_threaded_replay_scales_inactive_destinations() {
        for beta in [0.0, 0.5, 1.0] {
            let (mut dst, src, structure) = fixture();
            tree_transform_structure_with_structural_recoupling(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &mut dst,
                &src,
                1.0,
                beta,
                2,
            )
            .unwrap();
            assert_eq!(dst.data(), &expected(beta));
        }
    }

    #[test]
    fn storage_workspace_replay_scales_inactive_destinations() {
        for beta in [0.0, 0.5, 1.0] {
            let (mut dst, src, structure) = fixture();
            tree_transform_structure_with_storage_workspace_strided_kernel(
                &mut StridedHostKernelAdapter::default(),
                &mut StorageTreeTransformWorkspace::<Vec<f64>, Vec<f64>>::default(),
                &structure,
                &mut dst,
                &src,
                1.0,
                beta,
            )
            .unwrap();
            assert_eq!(dst.data(), &expected(beta));
        }
    }

    #[test]
    fn strided_replay_scales_inactive_destinations() {
        for beta in [0.0, 0.5, 1.0] {
            let (mut dst, src, structure) = fixture();
            let dst_structure = Arc::clone(dst.structure());
            let src_structure = Arc::clone(src.structure());
            tree_transform_structure_with_strided_kernel_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                dst.data_mut(),
                src.data(),
                1.0,
                beta,
            )
            .unwrap();
            assert_eq!(dst.data(), &expected(beta));
        }
    }
}
