use super::*;

fn validate_storage_range(
    storage_len: usize,
    base: usize,
    rows: usize,
    cols: usize,
) -> Result<(), OperationError> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    let end = base
        .checked_add(len)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    if end > storage_len {
        return Err(OperationError::ElementCountMismatch {
            expected: end,
            actual: storage_len,
        });
    }
    Ok(())
}

fn validate_storage_len(
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

fn direct_matrix_len(rows: usize, cols: usize) -> Result<usize, OperationError> {
    rows.checked_mul(cols)
        .ok_or_else(|| OperationError::ElementCountOverflow)
}

fn validate_group_replay_bounds<C>(
    group: &FusionBlockMatrixGroup<C>,
    storage_len: usize,
    matrix_rows: usize,
    matrix_cols: usize,
) -> Result<(), OperationError> {
    let matrix_len = direct_matrix_len(matrix_rows, matrix_cols)?;
    if let Some(base) = group.direct_offset {
        validate_storage_range(storage_len, base, matrix_rows, matrix_cols)?;
    }
    if group.block_indices.len() != group.subblocks.len() {
        return Err(OperationError::StructureMismatch {
            tensor: "fusion block group",
        });
    }
    for layout in &group.subblocks {
        validate_raw_strided_bounds(
            storage_len,
            &layout.block.shape,
            &layout.block.strides,
            layout.block.offset,
        )?;
        validate_raw_strided_bounds(
            matrix_len,
            &layout.block.shape,
            &layout.matrix_strides,
            layout.matrix_offset,
        )?;
    }
    Ok(())
}

fn validate_compiled_plan_layouts<C>(
    dst_structure: &BlockStructure,
    lhs_structure: &BlockStructure,
    rhs_structure: &BlockStructure,
    inactive_dst_scale_blocks: &[FusionScaleBlockLayout],
    groups: &[FusionBlockContractGroupPlan<C>],
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
) -> Result<(), OperationError>
where
    C: Copy + PartialEq + One,
{
    validate_destination_layouts_injective(
        dst_structure,
        "fusion block contraction destination layouts overlap",
    )?;
    let dst_len = dst_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let lhs_len = lhs_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let rhs_len = rhs_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut active_dst_blocks = vec![false; dst_structure.block_count()];
    for group in groups {
        validate_group_shape(group)?;
        validate_group_against_structure("lhs", &group.lhs, lhs_structure, lhs_len, lhs_op)?;
        validate_group_against_structure("rhs", &group.rhs, rhs_structure, rhs_len, rhs_op)?;
        validate_group_against_structure(
            "dst",
            &group.dst,
            dst_structure,
            dst_len,
            MatrixOp::Identity,
        )?;
        for &block_index in &group.dst.block_indices {
            let owned = active_dst_blocks.get_mut(block_index).ok_or_else(|| {
                OperationError::BlockIndexOutOfBounds {
                    tensor: "dst",
                    index: block_index,
                    count: dst_structure.block_count(),
                }
            })?;
            if *owned {
                return Err(OperationError::DuplicateTransformDestination {
                    dst_block: block_index,
                });
            }
            *owned = true;
        }
    }
    let mut inactive = inactive_dst_scale_blocks.iter();
    for (block_index, active) in active_dst_blocks.into_iter().enumerate() {
        if active {
            continue;
        }
        let layout = inactive
            .next()
            .ok_or_else(|| OperationError::StructureMismatch {
                tensor: "inactive dst blocks",
            })?;
        let block = dst_structure.block(block_index)?;
        if block.shape() != layout.block.shape
            || strides_to_isize(block.strides())? != layout.block.strides
            || offset_to_isize(block.offset())? != layout.block.offset
        {
            return Err(OperationError::StructureMismatch {
                tensor: "inactive dst blocks",
            });
        }
    }
    if inactive.next().is_some() {
        return Err(OperationError::StructureMismatch {
            tensor: "inactive dst blocks",
        });
    }
    Ok(())
}

fn validate_direct_plan_layouts<C>(
    dst_structure: &BlockStructure,
    lhs_structure: &BlockStructure,
    rhs_structure: &BlockStructure,
    inactive_dst_scale_blocks: &[FusionScaleBlockLayout],
    jobs: &[(Rank2GemmBatchJob, C)],
) -> Result<(), OperationError> {
    let dst_len = dst_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let lhs_len = lhs_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let rhs_len = rhs_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;

    let mut previous_job_offset = None;
    for (job, _) in jobs {
        validate_storage_range(lhs_len, job.lhs_offset, job.rows, job.contracted)?;
        validate_storage_range(rhs_len, job.rhs_offset, job.contracted, job.cols)?;
        validate_storage_range(dst_len, job.dst_offset, job.rows, job.cols)?;
        if previous_job_offset.is_some_and(|previous| previous >= job.dst_offset) {
            return Err(OperationError::StructureMismatch {
                tensor: "canonical direct job order",
            });
        }
        previous_job_offset = Some(job.dst_offset);
    }
    let mut previous_inactive_offset = None;
    for layout in inactive_dst_scale_blocks {
        let (start, _) = canonical_inactive_range(layout, dst_len)?;
        if previous_inactive_offset.is_some_and(|previous| previous >= start) {
            return Err(OperationError::StructureMismatch {
                tensor: "canonical inactive range order",
            });
        }
        previous_inactive_offset = Some(start);
    }

    let mut job_index = 0usize;
    let mut inactive_index = 0usize;
    let mut covered = 0usize;
    while job_index < jobs.len() || inactive_index < inactive_dst_scale_blocks.len() {
        let job_range = jobs
            .get(job_index)
            .map(|(job, _)| canonical_job_range(job))
            .transpose()?;
        let inactive_range = inactive_dst_scale_blocks
            .get(inactive_index)
            .map(|layout| canonical_inactive_range(layout, dst_len))
            .transpose()?;
        let (start, end) = match (job_range, inactive_range) {
            (Some(job), Some(inactive)) if job.0 <= inactive.0 => {
                job_index += 1;
                job
            }
            (Some(_), Some(inactive)) => {
                inactive_index += 1;
                inactive
            }
            (Some(job), None) => {
                job_index += 1;
                job
            }
            (None, Some(inactive)) => {
                inactive_index += 1;
                inactive
            }
            (None, None) => break,
        };
        if start != covered {
            return Err(OperationError::StructureMismatch {
                tensor: "canonical dst coverage",
            });
        }
        covered = end;
    }
    if covered != dst_len {
        return Err(OperationError::StructureMismatch {
            tensor: "canonical dst coverage",
        });
    }
    Ok(())
}

fn canonical_job_range(job: &Rank2GemmBatchJob) -> Result<(usize, usize), OperationError> {
    let len = direct_matrix_len(job.rows, job.cols)?;
    let end = job
        .dst_offset
        .checked_add(len)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    Ok((job.dst_offset, end))
}

fn canonical_inactive_range(
    layout: &FusionScaleBlockLayout,
    dst_len: usize,
) -> Result<(usize, usize), OperationError> {
    if layout.block.shape.len() != 1
        || layout.block.strides.as_slice() != [1]
        || layout.block.offset < 0
    {
        return Err(OperationError::StructureMismatch {
            tensor: "inactive dst ranges",
        });
    }
    let start = usize::try_from(layout.block.offset)
        .map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })?;
    let end = start
        .checked_add(layout.block.shape[0])
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    if end > dst_len {
        return Err(OperationError::ElementCountMismatch {
            expected: end,
            actual: dst_len,
        });
    }
    Ok((start, end))
}

fn validate_group_shape<C>(group: &FusionBlockContractGroupPlan<C>) -> Result<(), OperationError> {
    if group.lhs.cols != group.rhs.rows {
        return Err(OperationError::ShapeMismatch {
            dst: vec![group.lhs.cols],
            src: vec![group.rhs.rows],
        });
    }
    if group.dst.rows != group.lhs.rows || group.dst.cols != group.rhs.cols {
        return Err(OperationError::ShapeMismatch {
            dst: vec![group.dst.rows, group.dst.cols],
            src: vec![group.lhs.rows, group.rhs.cols],
        });
    }
    Ok(())
}

fn validate_group_against_structure<C>(
    tensor: &'static str,
    group: &FusionBlockMatrixGroup<C>,
    structure: &BlockStructure,
    storage_len: usize,
    op: MatrixOp,
) -> Result<(), OperationError>
where
    C: Copy + PartialEq + One,
{
    let (matrix_rows, matrix_cols) = match op {
        MatrixOp::Identity => (group.rows, group.cols),
        MatrixOp::Transpose | MatrixOp::Adjoint => (group.cols, group.rows),
    };
    validate_group_replay_bounds(group, storage_len, matrix_rows, matrix_cols)?;
    if group.subblocks.is_empty() {
        return Err(OperationError::StructureMismatch { tensor });
    }
    for (&block_index, layout) in group.block_indices.iter().zip(&group.subblocks) {
        let block =
            structure
                .block(block_index)
                .map_err(|_| OperationError::BlockIndexOutOfBounds {
                    tensor,
                    index: block_index,
                    count: structure.block_count(),
                })?;
        if block.shape() != layout.block.shape
            || strides_to_isize(block.strides())? != layout.block.strides
            || offset_to_isize(block.offset())? != layout.block.offset
        {
            return Err(OperationError::StructureMismatch { tensor });
        }
    }
    let covers_matrix = matrix_layouts_cover_exactly(group, matrix_rows, matrix_cols)
        .map_err(|_| OperationError::StructureMismatch { tensor })?;
    if group.needs_clear == covers_matrix {
        return Err(OperationError::StructureMismatch { tensor });
    }
    if let Some(offset) = group.direct_offset {
        if direct_group_matrix_offset_generic(&group.subblocks, covers_matrix) != Some(offset) {
            return Err(OperationError::StructureMismatch { tensor });
        }
    }
    Ok(())
}

fn matrix_layouts_cover_exactly<C>(
    group: &FusionBlockMatrixGroup<C>,
    matrix_rows: usize,
    matrix_cols: usize,
) -> Result<bool, OperationError> {
    let matrix_len = direct_matrix_len(matrix_rows, matrix_cols)?;
    if matrix_len == 0 {
        return Ok(group
            .subblocks
            .iter()
            .all(|layout| layout.block.shape.contains(&0)));
    }
    let mut rectangles = Vec::with_capacity(group.subblocks.len());
    for layout in &group.subblocks {
        let rectangle = canonical_matrix_rectangle(matrix_rows, matrix_cols, layout)?;
        if rectangle[0].0 != rectangle[0].1 && rectangle[1].0 != rectangle[1].1 {
            rectangles.push(rectangle);
        }
    }

    for axis in 0..2 {
        let mut intervals: Vec<_> = rectangles.iter().map(|rectangle| rectangle[axis]).collect();
        intervals.sort_unstable();
        intervals.dedup();
        if intervals.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(OperationError::StructureMismatch {
                tensor: "fusion block matrix",
            });
        }
    }

    rectangles.sort_unstable();
    if rectangles.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(OperationError::StructureMismatch {
            tensor: "fusion block matrix",
        });
    }
    let occupied = rectangles.iter().try_fold(0usize, |total, rectangle| {
        let rows = rectangle[0].1 - rectangle[0].0;
        let cols = rectangle[1].1 - rectangle[1].0;
        total
            .checked_add(
                rows.checked_mul(cols)
                    .ok_or_else(|| OperationError::ElementCountOverflow)?,
            )
            .ok_or_else(|| OperationError::ElementCountOverflow)
    })?;
    Ok(occupied == matrix_len)
}

fn canonical_matrix_rectangle<C>(
    matrix_rows: usize,
    matrix_cols: usize,
    layout: &FusionSubblockMatrixLayout<C>,
) -> Result<[(usize, usize); 2], OperationError> {
    let offset = usize::try_from(layout.matrix_offset)
        .map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })?;
    for split in 0..=layout.block.shape.len() {
        let mut row_dim = 1usize;
        let mut expected_stride = 1usize;
        let mut canonical = true;
        for (&dim, &stride) in layout.block.shape[..split]
            .iter()
            .zip(&layout.matrix_strides[..split])
        {
            if dim > 1 && usize::try_from(stride).ok() != Some(expected_stride) {
                canonical = false;
                break;
            }
            row_dim = row_dim
                .checked_mul(dim)
                .ok_or_else(|| OperationError::ElementCountOverflow)?;
            expected_stride = row_dim;
        }
        if !canonical || row_dim > matrix_rows {
            continue;
        }

        let mut col_dim = 1usize;
        expected_stride = matrix_rows;
        for (&dim, &stride) in layout.block.shape[split..]
            .iter()
            .zip(&layout.matrix_strides[split..])
        {
            if dim > 1 && usize::try_from(stride).ok() != Some(expected_stride) {
                canonical = false;
                break;
            }
            col_dim = col_dim
                .checked_mul(dim)
                .ok_or_else(|| OperationError::ElementCountOverflow)?;
            expected_stride = expected_stride
                .checked_mul(dim)
                .ok_or_else(|| OperationError::ElementCountOverflow)?;
        }
        if !canonical || col_dim > matrix_cols {
            continue;
        }

        let row = offset % matrix_rows;
        let col = offset / matrix_rows;
        let row_end = row
            .checked_add(row_dim)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        let col_end = col
            .checked_add(col_dim)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        if row_end <= matrix_rows && col_end <= matrix_cols {
            return Ok([(row, row_end), (col, col_end)]);
        }
    }
    Err(OperationError::StructureMismatch {
        tensor: "fusion block matrix",
    })
}
