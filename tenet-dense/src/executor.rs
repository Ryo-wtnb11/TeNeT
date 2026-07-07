use num_complex::{Complex32, Complex64};

use crate::{
    DenseBackend, DenseDotConfig, DenseError, DenseRead, DenseScalar, DenseTensor, DenseView,
    DenseViewMut, DenseWrite,
};

/// One GEMM of a batched matmul over shared flat buffers: the column-major
/// `rows x cols` destination block at `dst_offset` receives
/// `alpha * lhs_block * rhs_block + beta * dst_block`. Offsets are element
/// offsets relative to the corresponding view's own offset. Callers guarantee
/// the destination blocks of a batch are pairwise disjoint, so executors may
/// run jobs in any order or concurrently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseGemmBatchJob {
    pub dst_offset: usize,
    pub lhs_offset: usize,
    pub rhs_offset: usize,
    pub rows: usize,
    pub contracted: usize,
    pub cols: usize,
}

pub trait DenseExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>;
    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>;
    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>;

    fn svd_into(
        &mut self,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let outputs = self.svd(input)?;
        if outputs.len() != 3 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "svd_into",
                message: "dense SVD must return exactly (U, S, Vt)".to_string(),
            });
        }
        copy_dense_tensor_into(&outputs[0], u, "svd_into")?;
        copy_dense_tensor_into(&outputs[1], s, "svd_into")?;
        copy_dense_tensor_into(&outputs[2], vt, "svd_into")
    }

    fn qr_into(
        &mut self,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let outputs = self.qr(input)?;
        if outputs.len() != 2 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "qr_into",
                message: "dense QR must return exactly (Q, R)".to_string(),
            });
        }
        copy_dense_tensor_into(&outputs[0], q, "qr_into")?;
        copy_dense_tensor_into(&outputs[1], r, "qr_into")
    }

    fn eigh_into(
        &mut self,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let outputs = self.eigh(input)?;
        if outputs.len() != 2 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "eigh_into",
                message: "dense EIGH must return exactly (values, vectors)".to_string(),
            });
        }
        copy_dense_tensor_into(&outputs[0], values, "eigh_into")?;
        copy_dense_tensor_into(&outputs[1], vectors, "eigh_into")
    }

    /// General (non-Hermitian) eigendecomposition `(values, vectors)`; both
    /// outputs are complex regardless of the input scalar.
    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let _ = input;
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eig",
            message: "executor does not implement the general eigendecomposition".to_string(),
        })
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError>;

    fn matmul_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        self.dot_general_into(output, lhs, rhs, &DenseDotConfig::matmul())
    }

    /// Accumulate-form matmul: `output = alpha * lhs * rhs + beta * output`
    /// (BLAS gemm semantics). The default supports only the overwrite case
    /// `alpha = 1, beta = 0`; accumulate-capable backends override it.
    fn matmul_axpby_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        if alpha.is_one() && beta.is_zero() {
            return self.matmul_into(output, lhs, rhs);
        }
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "matmul_axpby_into",
            message: "executor does not implement the accumulate-form matmul".to_string(),
        })
    }

    /// Batched accumulate-form matmul over shared flat buffers: for each job,
    /// the destination block receives `alpha * lhs_block * rhs_block + beta *
    /// dst_block` (column-major, BLAS gemm semantics; see
    /// [`DenseGemmBatchJob`]). The default executes the jobs serially through
    /// `matmul_axpby_into`; batch-capable backends override it.
    fn matmul_batch_axpby_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        match (output, lhs, rhs) {
            (DenseWrite::F32(out), DenseRead::F32(lhs), DenseRead::F32(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f32>| DenseWrite::F32(view),
                    |view: DenseView<'_, f32>| DenseRead::F32(view),
                )
            }
            (DenseWrite::F64(out), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f64>| DenseWrite::F64(view),
                    |view: DenseView<'_, f64>| DenseRead::F64(view),
                )
            }
            (DenseWrite::C32(out), DenseRead::C32(lhs), DenseRead::C32(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex32>| DenseWrite::C32(view),
                    |view: DenseView<'_, Complex32>| DenseRead::C32(view),
                )
            }
            (DenseWrite::C64(out), DenseRead::C64(lhs), DenseRead::C64(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex64>| DenseWrite::C64(view),
                    |view: DenseView<'_, Complex64>| DenseRead::C64(view),
                )
            }
            _ => Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "matmul_batch_axpby_into",
                message: "batched matmul requires matching f32/f64/c32/c64 operands".to_string(),
            }),
        }
    }
}

fn copy_dense_tensor_into(
    tensor: &DenseTensor,
    output: DenseWrite<'_>,
    op: &'static str,
) -> Result<(), DenseError> {
    match output {
        DenseWrite::F32(output) => {
            copy_contiguous_tensor_into_view(tensor.as_f32_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::F64(output) => {
            copy_contiguous_tensor_into_view(tensor.as_f64_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::C32(output) => {
            copy_contiguous_tensor_into_view(tensor.as_c32_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::C64(output) => {
            copy_contiguous_tensor_into_view(tensor.as_c64_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::I32(_) | DenseWrite::I64(_) | DenseWrite::Bool(_) => Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!("{op} outputs require f32/f64/c32/c64 destination views"),
        }),
    }
}

fn copy_contiguous_tensor_into_view<T: Copy>(
    source: &[T],
    source_shape: &[usize],
    mut output: DenseViewMut<'_, T>,
    op: &'static str,
) -> Result<(), DenseError> {
    if source_shape != output.shape() {
        return Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output shape mismatch: source {:?}, destination {:?}",
                source_shape,
                output.shape()
            ),
        });
    }
    let expected = source_shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or(DenseError::ElementCountOverflow)
    })?;
    if source.len() != expected {
        return Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output storage length mismatch: source {}, expected {}",
                source.len(),
                expected
            ),
        });
    }
    if expected == 0 {
        return Ok(());
    }
    if source_shape.is_empty() {
        let offset = output.offset();
        output.data_mut()[offset] = source[0];
        return Ok(());
    }

    let shape = output.shape().to_vec();
    let strides = output.strides().to_vec();
    let offset = output.offset();
    let run = shape[0];
    let outer_count = shape[1..].iter().product::<usize>();
    let mut index = vec![0usize; shape.len()];
    let data = output.data_mut();
    for outer in 0..outer_count {
        let src_start = outer * run;
        let mut dst_start = offset;
        for axis in 1..shape.len() {
            dst_start += index[axis] * strides[axis];
        }
        if strides[0] == 1 {
            data[dst_start..dst_start + run].copy_from_slice(&source[src_start..src_start + run]);
        } else {
            for lane in 0..run {
                data[dst_start + lane * strides[0]] = source[src_start + lane];
            }
        }
        for axis in 1..shape.len() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(())
}

pub(crate) fn batch_offset(base: usize, offset: usize) -> Result<usize, DenseError> {
    base.checked_add(offset)
        .ok_or(DenseError::OffsetOverflow { value: offset })
}

pub(crate) fn matrix_len(rows: usize, cols: usize) -> Result<usize, DenseError> {
    rows.checked_mul(cols)
        .ok_or(DenseError::ElementCountOverflow)
}

fn same_gemm_shape(lhs: &DenseGemmBatchJob, rhs: &DenseGemmBatchJob) -> bool {
    lhs.rows == rhs.rows && lhs.contracted == rhs.contracted && lhs.cols == rhs.cols
}

pub(crate) fn strided_batch_run_len(jobs: &[DenseGemmBatchJob], start: usize) -> usize {
    let Some(first) = jobs.get(start) else {
        return 0;
    };
    let Some(second) = jobs.get(start + 1) else {
        return 1;
    };
    if !same_gemm_shape(first, second) {
        return 1;
    }
    let Some(dst_stride) = second.dst_offset.checked_sub(first.dst_offset) else {
        return 1;
    };
    if dst_stride == 0 {
        return 1;
    }
    let Some(lhs_stride) = second.lhs_offset.checked_sub(first.lhs_offset) else {
        return 1;
    };
    let Some(rhs_stride) = second.rhs_offset.checked_sub(first.rhs_offset) else {
        return 1;
    };

    let mut len = 2usize;
    while let Some(next) = jobs.get(start + len) {
        let prev = &jobs[start + len - 1];
        if !same_gemm_shape(first, next) {
            break;
        }
        if prev.dst_offset.checked_add(dst_stride) != Some(next.dst_offset)
            || prev.lhs_offset.checked_add(lhs_stride) != Some(next.lhs_offset)
            || prev.rhs_offset.checked_add(rhs_stride) != Some(next.rhs_offset)
        {
            break;
        }
        len += 1;
    }
    len
}

pub(crate) fn has_strided_batch_run(jobs: &[DenseGemmBatchJob]) -> bool {
    let mut start = 0usize;
    while start < jobs.len() {
        let run_len = strided_batch_run_len(jobs, start);
        if run_len > 1 {
            return true;
        }
        start += run_len;
    }
    false
}

/// Serial fallback for [`DenseExecutor::matmul_batch_axpby_into`]: one
/// `matmul_axpby_into` per job over rank-2 sub-views of the shared buffers.
#[allow(clippy::too_many_arguments)]
fn matmul_batch_axpby_serial<E, T, W, R>(
    executor: &mut E,
    mut output: DenseViewMut<'_, T>,
    lhs: DenseView<'_, T>,
    rhs: DenseView<'_, T>,
    jobs: &[DenseGemmBatchJob],
    alpha: DenseScalar,
    beta: DenseScalar,
    wrap_write: W,
    wrap_read: R,
) -> Result<(), DenseError>
where
    E: DenseExecutor + ?Sized,
    W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
    R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
{
    for job in jobs {
        let lhs_shape = [job.rows, job.contracted];
        let lhs_strides = [1, job.rows];
        let rhs_shape = [job.contracted, job.cols];
        let rhs_strides = [1, job.contracted];
        let dst_shape = [job.rows, job.cols];
        let dst_strides = [1, job.rows];
        let lhs_view = DenseView::new(
            lhs.data(),
            &lhs_shape,
            &lhs_strides,
            batch_offset(lhs.offset(), job.lhs_offset)?,
        )?;
        let rhs_view = DenseView::new(
            rhs.data(),
            &rhs_shape,
            &rhs_strides,
            batch_offset(rhs.offset(), job.rhs_offset)?,
        )?;
        let dst_offset = batch_offset(output.offset(), job.dst_offset)?;
        let dst_view = DenseViewMut::new(output.data_mut(), &dst_shape, &dst_strides, dst_offset)?;
        executor.matmul_axpby_into(
            wrap_write(dst_view),
            wrap_read(lhs_view),
            wrap_read(rhs_view),
            alpha,
            beta,
        )?;
    }
    Ok(())
}
