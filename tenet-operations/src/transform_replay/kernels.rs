use super::*;

pub fn tensoradd_block_with_strided_kernel<T>(
    allocator: &mut HostAllocator,
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    let dst_shape = dst.shape().to_vec();
    let dst_strides = crate::strided::strides_to_isize(dst.strides())?;
    let dst_offset = offset_to_isize(dst.offset())?;
    let (dst_data, _) = dst.into_parts();
    let src_shape = src.shape().to_vec();
    let src_strides = crate::strided::strides_to_isize(src.strides())?;
    let src_offset = offset_to_isize(src.offset())?;
    let src_data = src.data();

    if dst_shape != src_shape {
        return Err(OperationError::ShapeMismatch {
            dst: dst_shape,
            src: src_shape,
        });
    }

    tensoradd_raw_strided_kernel(
        &mut allocator.zero_strides,
        dst_data,
        src_data,
        &dst_shape,
        &dst_strides,
        &src_strides,
        dst_offset,
        src_offset,
        false,
        alpha,
        beta,
    )
}

pub(super) fn tensoradd_prepared_block_with_strided_kernel<T>(
    zero_strides: &mut Vec<isize>,
    descriptor: &TensorAddDescriptor,
    term: &TensorAddDescriptorTerm,
    dst_data: &mut [T],
    src_data: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    tensoradd_raw_strided_kernel_trusted(
        zero_strides,
        dst_data,
        src_data,
        descriptor.shape(term),
        descriptor.dst_strides(term),
        descriptor.src_strides(term),
        term.dst_offset,
        term.src_offset,
        descriptor.source_conjugate(),
        alpha,
        beta,
    )
}

pub(crate) fn validate_replay_storage_len(
    structure: &BlockStructure,
    actual_len: usize,
) -> Result<(), OperationError> {
    let expected = structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    if actual_len != expected {
        return Err(OperationError::ElementCountMismatch {
            expected,
            actual: actual_len,
        });
    }
    Ok(())
}

pub(crate) fn zero_tree_transform_destination<A, D>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    dst_structure: &BlockStructure,
    dst_data: &mut [D],
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + Zero + One,
{
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    let zero = [D::zero()];
    let mut dst_strides = Vec::new();
    for block_index in 0..dst_structure.block_count() {
        let block = dst_structure.block(block_index)?;
        zero_strides.clear();
        zero_strides.resize(block.shape().len(), 0);
        dst_strides.clear();
        for &stride in block.strides() {
            dst_strides
                .push(isize::try_from(stride).map_err(|_| OperationError::ElementCountOverflow)?);
        }
        kernels.copy_scale_strided(
            dst_data,
            &zero,
            block.shape(),
            &dst_strides,
            zero_strides,
            offset_to_isize(block.offset())?,
            0,
            false,
            D::one(),
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_single_with_strided_kernel<A, D, C>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    fused_index: Option<&mut [usize]>,
    layouts: &TreeTransformLayoutTable,
    dst_index: usize,
    src_index: usize,
    coefficient: C,
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + One + PartialEq + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let dst_layout = layouts.entry(dst_index);
    let src_layout = layouts.entry(src_index);
    let shape = layouts.shape(dst_layout);
    let baked = layouts.fused_baked(dst_index);
    kernels.transform_strided_baked(
        zero_strides,
        dst_data,
        src_data,
        shape,
        layouts.strides(dst_layout),
        layouts.strides(src_layout),
        dst_layout.offset,
        src_layout.offset,
        source_conjugate,
        TransformScale::new(alpha, coefficient),
        match mode {
            DestinationMode::Axpby(beta) => Some(beta),
            DestinationMode::Overwrite => None,
        },
        baked,
        fused_index,
    )
}

/// Applies a batch of Multi-block recoupling matrices over shared flat scratch
/// buffers: per job, the column-major
/// (element_count x dst_count) destination block receives `source_block *
/// U^T`, with `recoupling_coefficients_dst_src` (row-major `U[dst, src]`)
/// reinterpreted as the column-major (src_count x dst_count) matrix `U^T`.
/// This is TensorKit's `_add_transform_multi!` `mul!` step submitted as one
/// grouped call; the naive per-element loop in the kernel adapter remains
/// only for adapters without a dense executor. Job offsets are relative to the
/// supplied chunk scratch, matching the trusted-view validation contract.
pub(super) fn recoupling_gemm_batch<E, D>(
    dense: &mut E,
    destination: &mut [D],
    source: &[D],
    coefficients: &[D],
    jobs: &[DenseGemmBatchJob],
    runs: &[usize],
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar,
{
    let dst_shape = [destination.len()];
    let lhs_shape = [source.len()];
    let rhs_shape = [coefficients.len()];
    let flat_strides = [1];
    let lhs = D::dense_read(tenet_dense::DenseView::new_trusted(
        source,
        &lhs_shape,
        &flat_strides,
        0,
    ));
    let rhs = D::dense_read(tenet_dense::DenseView::new_trusted(
        coefficients,
        &rhs_shape,
        &flat_strides,
        0,
    ));
    let output = D::dense_write(tenet_dense::DenseViewMut::new_trusted(
        destination,
        &dst_shape,
        &flat_strides,
        0,
    ));
    dense
        .matmul_batch_axpby_into(
            output,
            lhs,
            rhs,
            jobs,
            runs,
            D::one().dense_scalar(),
            D::zero().dense_scalar(),
        )
        .map_err(OperationError::Dense)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_multi_with_pack_gemm_scatter<A, D, C>(
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    recoupling_coefficients_dst_src: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + Zero + One + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    workspace.prepare_packed_buffers(source_len, destination_len, D::zero());
    let HostTreeTransformWorkspace {
        zero_strides,
        packed,
        fused_indices,
        ..
    } = workspace;
    tree_transform_multi_with_scratch_buffers(
        kernels,
        zero_strides,
        Some(fused_indices.as_mut_slice()),
        packed,
        layouts,
        dst_layout_start,
        dst_count,
        src_layout_start,
        src_count,
        coefficient_start,
        element_count,
        recoupling_coefficients_dst_src,
        source_conjugate,
        dst_data,
        src_data,
        alpha,
        mode,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_multi_with_scratch_buffers<
    A,
    D,
    C,
    SourceScratch,
    DestinationScratch,
>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    mut fused_index: Option<&mut [usize]>,
    scratch: &mut TreeTransformScratchBuffers<SourceScratch, DestinationScratch>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    recoupling_coefficients_dst_src: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + One + RecouplingCoefficientAction<C>,
    C: Copy,
    SourceScratch: HostWritableStorage<D>,
    DestinationScratch: HostWritableStorage<D>,
{
    for src_index in 0..src_count {
        pack_layout_into_column(
            kernels,
            fused_index.as_deref_mut(),
            layouts,
            src_layout_start + src_index,
            src_data,
            scratch.source_mut().as_mut_slice(),
            src_index * element_count,
            source_conjugate,
        )?;
    }

    {
        let (source, destination) = scratch.source_and_destination_mut();
        kernels.recoupling_src_times_u_transpose(
            destination.as_mut_slice(),
            source.as_slice(),
            recoupling_coefficients_dst_src,
            coefficient_start,
            element_count,
            src_count,
            dst_count,
        )?;
    }

    for dst_index in 0..dst_count {
        scatter_column_into_layout(
            kernels,
            zero_strides,
            fused_index.as_deref_mut(),
            layouts,
            dst_layout_start + dst_index,
            scratch.destination().as_slice(),
            dst_index * element_count,
            dst_data,
            alpha,
            mode,
        )?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the packing kernel keeps layout authority, source and destination slices, offset, and conjugation explicit"
)]
pub(super) fn pack_layout_into_column<A, T>(
    kernels: &mut A,
    fused_index: Option<&mut [usize]>,
    layouts: &TreeTransformLayoutTable,
    entry_index: usize,
    src_data: &[T],
    packed: &mut [T],
    packed_offset: usize,
    source_conjugate: bool,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + One,
{
    let layout = layouts.entry(entry_index);
    let shape = layouts.shape(layout);
    let baked = layouts.fused_baked(entry_index);
    let packed_offset = offset_to_isize(packed_offset)?;
    match fused_index {
        Some(index) => kernels.copy_scale_strided_baked_with_index(
            packed,
            src_data,
            shape,
            layouts.packed_strides(layout),
            layouts.strides(layout),
            packed_offset,
            layout.offset,
            source_conjugate,
            T::one(),
            baked,
            index,
        ),
        None => kernels.copy_scale_strided_baked(
            packed,
            src_data,
            shape,
            layouts.packed_strides(layout),
            layouts.strides(layout),
            packed_offset,
            layout.offset,
            source_conjugate,
            T::one(),
            baked,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_column_into_layout<A, T>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    fused_index: Option<&mut [usize]>,
    layouts: &TreeTransformLayoutTable,
    entry_index: usize,
    packed: &[T],
    packed_offset: usize,
    dst_data: &mut [T],
    alpha: T,
    mode: DestinationMode<T>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy,
{
    let layout = layouts.entry(entry_index);
    let shape = layouts.shape(layout);
    let baked = layouts.fused_baked(entry_index);
    match (mode, fused_index) {
        (DestinationMode::Axpby(beta), Some(index)) => {
            zero_strides.clear();
            kernels.axpby_strided_baked_with_index(
                dst_data,
                packed,
                shape,
                layouts.strides(layout),
                layouts.packed_strides(layout),
                layout.offset,
                offset_to_isize(packed_offset)?,
                alpha,
                beta,
                baked,
                index,
            )
        }
        (DestinationMode::Overwrite, Some(index)) => kernels.copy_scale_strided_baked_with_index(
            dst_data,
            packed,
            shape,
            layouts.strides(layout),
            layouts.packed_strides(layout),
            layout.offset,
            offset_to_isize(packed_offset)?,
            false,
            alpha,
            baked,
            index,
        ),
        (DestinationMode::Axpby(beta), None) => {
            zero_strides.clear();
            kernels.axpby_strided_baked(
                dst_data,
                packed,
                shape,
                layouts.strides(layout),
                layouts.packed_strides(layout),
                layout.offset,
                offset_to_isize(packed_offset)?,
                alpha,
                beta,
                baked,
            )
        }
        (DestinationMode::Overwrite, None) => kernels.copy_scale_strided_baked(
            dst_data,
            packed,
            shape,
            layouts.strides(layout),
            layouts.packed_strides(layout),
            layout.offset,
            offset_to_isize(packed_offset)?,
            false,
            alpha,
            baked,
        ),
    }
}

pub(super) fn recoupling_multi_block<C: Copy>(
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

pub(super) fn scale_inactive_destinations<A, D, C>(
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
mod allocation_replay_tests {
    use super::*;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use tenet_dense::{
        DenseDotConfig, DenseError, DenseRead, DenseScalar, DenseTensor, DenseWrite,
    };

    #[derive(Default)]
    struct NoAllocDenseExecutor;

    impl DenseExecutor for NoAllocDenseExecutor {
        fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            unreachable!("tree replay does not call SVD")
        }

        fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            unreachable!("tree replay does not call QR")
        }

        fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            unreachable!("tree replay does not call EIGH")
        }

        fn dot_general_into(
            &mut self,
            _output: DenseWrite<'_>,
            _lhs: DenseRead<'_>,
            _rhs: DenseRead<'_>,
            _config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            unreachable!("tree replay uses the batched matmul entry point")
        }

        fn matmul_batch_axpby_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            jobs: &[DenseGemmBatchJob],
            _runs: &[usize],
            _alpha: DenseScalar,
            _beta: DenseScalar,
        ) -> Result<(), DenseError> {
            let (mut output, lhs, rhs) = match (output, lhs, rhs) {
                (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                    (output, lhs, rhs)
                }
                _ => unreachable!("allocation oracle uses f64"),
            };
            let output_offset = output.offset();
            let lhs_offset = lhs.offset();
            let rhs_offset = rhs.offset();
            let output_data = output.data_mut();
            for job in jobs {
                for col in 0..job.cols {
                    for row in 0..job.rows {
                        let mut value = 0.0;
                        for contracted in 0..job.contracted {
                            value += lhs.data()
                                [lhs_offset + job.lhs_offset + row + contracted * job.rows]
                                * rhs.data()[rhs_offset
                                    + job.rhs_offset
                                    + contracted
                                    + col * job.contracted];
                        }
                        output_data[output_offset + job.dst_offset + row + col * job.rows] = value;
                    }
                }
            }
            Ok(())
        }
    }

    #[test]
    fn warm_threaded_replay_has_no_owned_allocations() {
        let block_structure = Arc::new(
            BlockStructure::packed_column_major(1, [vec![4], vec![4], vec![4], vec![4]]).unwrap(),
        );
        let structure = TreeTransformStructure::compile_structures(
            &block_structure,
            &block_structure,
            &[
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]),
                TreeTransformBlockSpec::single(2, 2, 1.0),
                TreeTransformBlockSpec::single(3, 3, -1.0),
            ],
        )
        .unwrap();
        let src = (1..=16).map(f64::from).collect::<Vec<_>>();
        let expected = (1..=16)
            .map(|value| {
                if value <= 12 {
                    f64::from(value)
                } else {
                    -f64::from(value)
                }
            })
            .collect::<Vec<_>>();
        let mut dst = vec![0.0; 16];
        let mut kernels = StridedHostKernelAdapter::default();
        let mut dense = NoAllocDenseExecutor;
        let mut workspace = TreeTransformWorkspace::default();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();

        let mut replay = || {
            tree_transform_structure_with_structural_recoupling_raw(
                &mut kernels,
                &mut dense,
                &mut workspace,
                &structure,
                &block_structure,
                &block_structure,
                &mut dst,
                &src,
                1.0,
                0.0,
                3,
            )
            .unwrap();
        };

        pool.install(&mut replay);
        let (joins, allocations) = allocation_oracle::with_session(|| {
            pool.install(|| {
                allocation_oracle::with_measurement(|| {
                    replay();
                    allocation_oracle::join_entries()
                })
            })
        });
        assert_eq!(allocations, 0);
        assert!(joins > 0);
        assert_eq!(dst, expected);
    }

    #[test]
    fn warm_threaded_overwrite_replay_has_no_owned_allocations() {
        let block_structure = Arc::new(
            BlockStructure::packed_column_major(1, [vec![4], vec![4], vec![4], vec![4]]).unwrap(),
        );
        let structure = TreeTransformStructure::compile_structures(
            &block_structure,
            &block_structure,
            &[
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]),
                TreeTransformBlockSpec::single(2, 2, 1.0),
                TreeTransformBlockSpec::single(3, 3, -1.0),
            ],
        )
        .unwrap();
        let src = (1..=16).map(f64::from).collect::<Vec<_>>();
        let expected = (1..=16)
            .map(|value| {
                if value <= 12 {
                    f64::from(value)
                } else {
                    -f64::from(value)
                }
            })
            .collect::<Vec<_>>();
        let mut dst = vec![f64::NAN; 16];
        let mut kernels = StridedHostKernelAdapter::default();
        let mut dense = NoAllocDenseExecutor;
        let mut workspace = TreeTransformWorkspace::default();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();

        let mut replay = || {
            dst.fill(f64::NAN);
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut kernels,
                &mut dense,
                &mut workspace,
                &structure,
                &block_structure,
                &block_structure,
                &mut dst,
                &src,
                1.0,
                3,
            )
            .unwrap();
        };

        pool.install(&mut replay);
        let (_, allocations) = allocation_oracle::with_session(|| {
            pool.install(|| allocation_oracle::with_measurement(&mut replay))
        });
        assert_eq!(allocations, 0);
        assert_eq!(dst, expected);
    }

    #[test]
    fn rank_ten_multi_replay_matches_literal_oracle_without_owned_allocations() {
        const RANK: usize = 10;
        const GROUPS: usize = 4;
        const BLOCKS: usize = 2 * GROUPS;
        let shape = [2, 2, 2, 2, 2, 2, 2, 2, 2, 1];
        let axes = [8, 7, 6, 5, 4, 3, 2, 1, 0, 9];
        let block_structure = Arc::new(
            BlockStructure::packed_column_major(RANK, vec![shape.to_vec(); BLOCKS]).unwrap(),
        );
        let specs = (0..GROUPS)
            .map(|group| {
                let first = 2 * group;
                TreeTransformBlockSpec::multi(
                    vec![first, first + 1],
                    vec![first, first + 1],
                    vec![1.0, 0.0, 0.0, 1.0],
                )
                .with_source_axes(axes)
            })
            .collect::<Vec<_>>();
        let structure =
            TreeTransformStructure::compile_structures(&block_structure, &block_structure, &specs)
                .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
        assert_eq!(structure.recoupling_plan().jobs().len(), GROUPS);

        let elements = shape.iter().product::<usize>();
        let src = (0..BLOCKS * elements)
            .map(|value| value as f64 + 0.25)
            .collect::<Vec<_>>();
        let strides = block_structure.block(0).unwrap().strides();
        let mut expected = vec![0.0; src.len()];
        for block in 0..BLOCKS {
            let base = block * elements;
            for dst_linear in 0..elements {
                let src_linear = (0..RANK).fold(0usize, |offset, axis| {
                    let coordinate = (dst_linear / strides[axis]) % shape[axis];
                    offset + coordinate * strides[axes[axis]]
                });
                expected[base + dst_linear] = src[base + src_linear];
            }
        }

        let mut serial = vec![0.0; src.len()];
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut NoAllocDenseExecutor,
            &mut TreeTransformWorkspace::default(),
            &structure,
            &block_structure,
            &block_structure,
            &mut serial,
            &src,
            1.0,
            0.0,
            1,
        )
        .unwrap();
        assert_eq!(serial, expected);

        let mut threaded = vec![0.0; src.len()];
        let mut kernels = StridedHostKernelAdapter::default();
        let mut dense = NoAllocDenseExecutor;
        let mut workspace = TreeTransformWorkspace::default();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        pool.install(|| {
            tree_transform_structure_with_structural_recoupling_raw(
                &mut kernels,
                &mut dense,
                &mut workspace,
                &structure,
                &block_structure,
                &block_structure,
                &mut threaded,
                &src,
                1.0,
                0.0,
                3,
            )
            .unwrap();
        });

        assert_eq!(threaded, expected);
        threaded.fill(0.0);
        let (_, allocations) = allocation_oracle::with_session(|| {
            pool.install(|| {
                allocation_oracle::with_measurement(|| {
                    tree_transform_structure_with_structural_recoupling_raw(
                        &mut kernels,
                        &mut dense,
                        &mut workspace,
                        &structure,
                        &block_structure,
                        &block_structure,
                        &mut threaded,
                        &src,
                        1.0,
                        0.0,
                        3,
                    )
                    .unwrap();
                })
            })
        });
        assert_eq!(allocations, 0);
        assert_eq!(threaded, expected);
    }
}
