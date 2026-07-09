use num_complex::{Complex32, Complex64};

use crate::executor::{batch_offset, has_strided_batch_run, matrix_len, strided_batch_run_len};
use crate::layout::strides_to_isize;
use crate::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseGemmBatchJob, DenseRead,
    DenseScalar, DenseTensor, DenseView, DenseViewMut, DenseWrite,
};

use tenferro_cpu::CpuBackend;
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, DotGeneralConfig, Tensor, TensorDot, TensorRead,
    TensorView, TensorViewMut, TensorWrite, TypedTensorView, TypedTensorViewMut,
};

#[derive(Debug)]
pub struct DefaultDenseExecutor {
    backend: CpuBackend,
    matmul_config: DotGeneralConfig,
    strided_batch_matmul_config: DotGeneralConfig,
    grouped_cache: <CpuBackend as BackendRuntimeCache>::RuntimeCache,
    grouped_jobs: Vec<GroupedGemmJob>,
    #[cfg(test)]
    logical_gemm_dispatches: usize,
}

impl DefaultDenseExecutor {
    pub fn new() -> Self {
        Self::from_backend(CpuBackend::new())
    }

    pub fn with_threads(threads: usize) -> Result<Self, DenseError> {
        CpuBackend::with_threads(threads)
            .map(Self::from_backend)
            .map_err(|err| tenferro_error("CpuBackend::with_threads", err))
    }

    fn from_backend(backend: CpuBackend) -> Self {
        Self {
            backend,
            matmul_config: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: Vec::new(),
                rhs_batch_dims: Vec::new(),
            },
            strided_batch_matmul_config: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![2],
                rhs_batch_dims: vec![2],
            },
            grouped_cache: <CpuBackend as BackendRuntimeCache>::RuntimeCache::default(),
            grouped_jobs: Vec::new(),
            #[cfg(test)]
            logical_gemm_dispatches: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn reset_logical_gemm_dispatches(&mut self) {
        self.logical_gemm_dispatches = 0;
    }

    #[cfg(test)]
    pub(crate) fn logical_gemm_dispatches(&self) -> usize {
        self.logical_gemm_dispatches
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_grouped(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        #[cfg(test)]
        {
            self.logical_gemm_dispatches += jobs.len();
        }
        let lhs = TensorRead::from_view(tenferro_view(lhs)?);
        let rhs = TensorRead::from_view(tenferro_view(rhs)?);
        let output = TensorWrite::from_view(tenferro_view_mut(output)?);
        self.grouped_jobs.clear();
        self.grouped_jobs.extend(jobs.iter().map(|job| {
            GroupedGemmJob::new(
                job.dst_offset,
                job.lhs_offset,
                job.rhs_offset,
                job.rows,
                job.contracted,
                job.cols,
            )
        }));
        let accumulation = tenferro_tensor::DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: tenferro_scalar(alpha),
            beta: tenferro_scalar(beta),
        };
        let config = GroupedGemmConfig::new(&self.grouped_jobs, accumulation);
        BackendCachedDot::grouped_gemm_cached(
            &mut self.backend,
            &mut self.grouped_cache,
            None,
            lhs,
            rhs,
            &config,
            output,
        )
        .map_err(|err| tenferro_error("grouped_gemm", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_strided_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        alpha: DenseScalar,
        beta: DenseScalar,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<bool, DenseError>
    where
        T: 'static,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x> + Copy,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x> + Copy,
    {
        if jobs.len() < 2 {
            return Ok(false);
        }

        if has_strided_batch_run(jobs) {
            self.matmul_batch_axpby_strided_runs_typed(
                output, lhs, rhs, jobs, 0, alpha, beta, wrap_write, wrap_read,
            )?;
            return Ok(true);
        }

        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_strided_runs_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        cache_slot_base: usize,
        alpha: DenseScalar,
        beta: DenseScalar,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<(), DenseError>
    where
        T: 'static,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x> + Copy,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x> + Copy,
    {
        let mut start = 0usize;
        while start < jobs.len() {
            let run_len = strided_batch_run_len(jobs, start);
            self.matmul_strided_batch_run_typed(
                output,
                lhs,
                rhs,
                &jobs[start..start + run_len],
                cache_slot_base + start,
                alpha,
                beta,
                wrap_write,
                wrap_read,
            )?;
            start += run_len;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_strided_batch_run_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        run: &[DenseGemmBatchJob],
        cache_slot: usize,
        alpha: DenseScalar,
        beta: DenseScalar,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<(), DenseError>
    where
        T: 'static,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
    {
        #[cfg(test)]
        {
            self.logical_gemm_dispatches += 1;
        }
        let first = &run[0];
        let run_len = run.len();
        let (lhs_batch_stride, rhs_batch_stride, dst_batch_stride) = if run_len > 1 {
            let next = &run[1];
            (
                next.lhs_offset.checked_sub(first.lhs_offset).ok_or(
                    DenseError::OffsetOverflow {
                        value: first.lhs_offset,
                    },
                )?,
                next.rhs_offset.checked_sub(first.rhs_offset).ok_or(
                    DenseError::OffsetOverflow {
                        value: first.rhs_offset,
                    },
                )?,
                next.dst_offset.checked_sub(first.dst_offset).ok_or(
                    DenseError::OffsetOverflow {
                        value: first.dst_offset,
                    },
                )?,
            )
        } else {
            (
                matrix_len(first.rows, first.contracted)?,
                matrix_len(first.contracted, first.cols)?,
                matrix_len(first.rows, first.cols)?,
            )
        };
        let lhs_shape = [first.rows, first.contracted, run_len];
        let lhs_strides = [1, first.rows, lhs_batch_stride];
        let rhs_shape = [first.contracted, first.cols, run_len];
        let rhs_strides = [1, first.contracted, rhs_batch_stride];
        let dst_shape = [first.rows, first.cols, run_len];
        let dst_strides = [1, first.rows, dst_batch_stride];
        let lhs_view = DenseView::new(
            lhs.data(),
            &lhs_shape,
            &lhs_strides,
            batch_offset(lhs.offset(), first.lhs_offset)?,
        )?;
        let rhs_view = DenseView::new(
            rhs.data(),
            &rhs_shape,
            &rhs_strides,
            batch_offset(rhs.offset(), first.rhs_offset)?,
        )?;
        let dst_offset = batch_offset(output.offset(), first.dst_offset)?;
        let dst_view = DenseViewMut::new(output.data_mut(), &dst_shape, &dst_strides, dst_offset)?;
        let lhs = TensorRead::from_view(tenferro_view(wrap_read(lhs_view))?);
        let rhs = TensorRead::from_view(tenferro_view(wrap_read(rhs_view))?);
        let output = TensorWrite::from_view(tenferro_view_mut(wrap_write(dst_view))?);
        let accumulation = tenferro_tensor::DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: tenferro_scalar(alpha),
            beta: tenferro_scalar(beta),
        };
        BackendCachedDot::dot_general_read_into_accum_cached(
            &mut self.backend,
            &mut self.grouped_cache,
            Some(cache_slot),
            lhs,
            rhs,
            &self.strided_batch_matmul_config,
            accumulation,
            output,
        )
        .map_err(|err| tenferro_error("strided_batch_gemm", err))
    }
}

impl Default for DefaultDenseExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl DenseExecutor for DefaultDenseExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let input = tenferro_view(input)?;
        self.backend
            .svd_read(input)
            .map(wrap_outputs)
            .map_err(|err| tenferro_error("svd_read", err))
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let input = tenferro_view(input)?;
        self.backend
            .qr_read(input)
            .map(wrap_outputs)
            .map_err(|err| tenferro_error("qr_read", err))
    }

    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let input = tenferro_view(input)?;
        self.backend
            .eig_read(input)
            .map(wrap_outputs)
            .map_err(|err| tenferro_error("eig_read", err))
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let input = tenferro_view(input)?;
        self.backend
            .eigh_read(input)
            .map(wrap_outputs)
            .map_err(|err| tenferro_error("eigh_read", err))
    }

    // Values-only overrides route to tenferro's no-vector LAPACK entries
    // (`job='N'`), skipping the U/Vt (or eigenvector) computation the default
    // fallback would do. The backend entries take an owned `&Tensor`, so the
    // borrowed view is materialized to a contiguous tensor first — one cheap
    // copy of the input matrix versus the discarded O(n^2) vector work.
    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        let owned = tenferro_view(input)?
            .to_tensor()
            .map_err(|err| tenferro_error("svd_vals", err))?;
        self.backend
            .svd_values(&owned)
            .map(DenseTensor::from_tenferro)
            .map_err(|err| tenferro_error("svd_values", err))
    }

    fn eigh_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        let owned = tenferro_view(input)?
            .to_tensor()
            .map_err(|err| tenferro_error("eigh_vals", err))?;
        self.backend
            .eigh_values(&owned)
            .map(DenseTensor::from_tenferro)
            .map_err(|err| tenferro_error("eigh_values", err))
    }

    fn eig_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        let owned = tenferro_view(input)?
            .to_tensor()
            .map_err(|err| tenferro_error("eig_vals", err))?;
        self.backend
            .eig_values(&owned)
            .map(DenseTensor::from_tenferro)
            .map_err(|err| tenferro_error("eig_values", err))
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        let lhs = TensorRead::from_view(tenferro_view(lhs)?);
        let rhs = TensorRead::from_view(tenferro_view(rhs)?);
        let output = TensorWrite::from_view(tenferro_view_mut(output)?);
        let dot_config = tenferro_dot_config(config);
        // Non-conjugating path stays byte-identical to the plain read_into
        // (which itself just wraps an overwrite accumulation). Conjugation is
        // folded into the kernel via the accumulation's conj flags — no
        // conjugated operand copy — instead of falling back to a scalar loop.
        if config.lhs_conj() || config.rhs_conj() {
            let mut accumulation = tenferro_tensor::DotGeneralAccumulation::overwrite(lhs.dtype())
                .map_err(|err| tenferro_error("dot_general_accum", err))?;
            accumulation.lhs_conj = config.lhs_conj();
            accumulation.rhs_conj = config.rhs_conj();
            self.backend
                .dot_general_read_into_accum(lhs, rhs, &dot_config, accumulation, output)
                .map_err(|err| tenferro_error("dot_general_accum", err))
        } else {
            self.backend
                .dot_general_read_into(lhs, rhs, &dot_config, output)
                .map_err(|err| tenferro_error("dot_general_read_into", err))
        }
    }

    fn matmul_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        // GEMM backend selection is owned by tenferro; this seam only
        // lowers views and reuses the cached rank-2 contraction config.
        let lhs = TensorRead::from_view(tenferro_view(lhs)?);
        let rhs = TensorRead::from_view(tenferro_view(rhs)?);
        let output = TensorWrite::from_view(tenferro_view_mut(output)?);
        self.backend
            .dot_general_read_into(lhs, rhs, &self.matmul_config, output)
            .map_err(|err| tenferro_error("dot_general_read_into", err))
    }

    fn matmul_axpby_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        // Overwrite case keeps the cached-config fast path.
        if alpha.is_one() && beta.is_zero() {
            return self.matmul_into(output, lhs, rhs);
        }
        let lhs = TensorRead::from_view(tenferro_view(lhs)?);
        let rhs = TensorRead::from_view(tenferro_view(rhs)?);
        let output = TensorWrite::from_view(tenferro_view_mut(output)?);
        let accumulation = tenferro_tensor::DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: tenferro_scalar(alpha),
            beta: tenferro_scalar(beta),
        };
        self.backend
            .dot_general_read_into_accum(lhs, rhs, &self.matmul_config, accumulation, output)
            .map_err(|err| tenferro_error("dot_general_accum", err))
    }

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
            (DenseWrite::F32(mut out), DenseRead::F32(lhs), DenseRead::F32(rhs)) => {
                if self.matmul_batch_axpby_strided_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f32>| DenseWrite::F32(view),
                    |view: DenseView<'_, f32>| DenseRead::F32(view),
                )? {
                    return Ok(());
                }
                self.matmul_batch_axpby_grouped(
                    DenseWrite::F32(out),
                    DenseRead::F32(lhs),
                    DenseRead::F32(rhs),
                    jobs,
                    alpha,
                    beta,
                )
            }
            (DenseWrite::F64(mut out), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                if self.matmul_batch_axpby_strided_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f64>| DenseWrite::F64(view),
                    |view: DenseView<'_, f64>| DenseRead::F64(view),
                )? {
                    return Ok(());
                }
                self.matmul_batch_axpby_grouped(
                    DenseWrite::F64(out),
                    DenseRead::F64(lhs),
                    DenseRead::F64(rhs),
                    jobs,
                    alpha,
                    beta,
                )
            }
            (DenseWrite::C32(mut out), DenseRead::C32(lhs), DenseRead::C32(rhs)) => {
                if self.matmul_batch_axpby_strided_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex32>| DenseWrite::C32(view),
                    |view: DenseView<'_, Complex32>| DenseRead::C32(view),
                )? {
                    return Ok(());
                }
                self.matmul_batch_axpby_grouped(
                    DenseWrite::C32(out),
                    DenseRead::C32(lhs),
                    DenseRead::C32(rhs),
                    jobs,
                    alpha,
                    beta,
                )
            }
            (DenseWrite::C64(mut out), DenseRead::C64(lhs), DenseRead::C64(rhs)) => {
                if self.matmul_batch_axpby_strided_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex64>| DenseWrite::C64(view),
                    |view: DenseView<'_, Complex64>| DenseRead::C64(view),
                )? {
                    return Ok(());
                }
                self.matmul_batch_axpby_grouped(
                    DenseWrite::C64(out),
                    DenseRead::C64(lhs),
                    DenseRead::C64(rhs),
                    jobs,
                    alpha,
                    beta,
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

fn tenferro_scalar(value: DenseScalar) -> tenferro_tensor::ContractionScalar {
    match value {
        DenseScalar::F32(value) => tenferro_tensor::ContractionScalar::F32(value),
        DenseScalar::F64(value) => tenferro_tensor::ContractionScalar::F64(value),
        DenseScalar::C32(value) => tenferro_tensor::ContractionScalar::C32(value),
        DenseScalar::C64(value) => tenferro_tensor::ContractionScalar::C64(value),
    }
}

fn wrap_outputs(outputs: Vec<Tensor>) -> Vec<DenseTensor> {
    outputs
        .into_iter()
        .map(DenseTensor::from_tenferro)
        .collect()
}

fn tenferro_view(input: DenseRead<'_>) -> Result<TensorView<'_>, DenseError> {
    match input {
        DenseRead::F32(view) => typed_tenferro_view(view).map(TensorView::F32),
        DenseRead::F64(view) => typed_tenferro_view(view).map(TensorView::F64),
        DenseRead::I32(view) => typed_tenferro_view(view).map(TensorView::I32),
        DenseRead::I64(view) => typed_tenferro_view(view).map(TensorView::I64),
        DenseRead::Bool(view) => typed_tenferro_view(view).map(TensorView::Bool),
        DenseRead::C32(view) => typed_tenferro_view(view).map(TensorView::C32),
        DenseRead::C64(view) => typed_tenferro_view(view).map(TensorView::C64),
    }
}

fn tenferro_view_mut(output: DenseWrite<'_>) -> Result<TensorViewMut<'_>, DenseError> {
    match output {
        DenseWrite::F32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::F32),
        DenseWrite::F64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::F64),
        DenseWrite::I32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::I32),
        DenseWrite::I64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::I64),
        DenseWrite::Bool(view) => typed_tenferro_view_mut(view).map(TensorViewMut::Bool),
        DenseWrite::C32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::C32),
        DenseWrite::C64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::C64),
    }
}

fn typed_tenferro_view<'a, T: 'static>(
    view: DenseView<'a, T>,
) -> Result<TypedTensorView<'a, T>, DenseError> {
    let strides = strides_to_isize(view.strides())?;
    let offset = isize::try_from(view.offset()).map_err(|_| DenseError::OffsetOverflow {
        value: view.offset(),
    })?;
    TypedTensorView::from_slice(view.shape(), strides, offset, view.data())
        .map_err(|err| tenferro_error("TypedTensorView::from_slice", err))
}

fn typed_tenferro_view_mut<'a, T: 'static>(
    view: DenseViewMut<'a, T>,
) -> Result<TypedTensorViewMut<'a, T>, DenseError> {
    let DenseViewMut {
        data,
        shape,
        strides,
        offset,
    } = view;
    let strides = strides_to_isize(strides)?;
    let offset =
        isize::try_from(offset).map_err(|_| DenseError::OffsetOverflow { value: offset })?;
    TypedTensorViewMut::from_slice(shape, strides, offset, data)
        .map_err(|err| tenferro_error("TypedTensorViewMut::from_slice", err))
}

fn tenferro_dot_config(config: &DenseDotConfig) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: config.lhs_contracting_dims().to_vec(),
        rhs_contracting_dims: config.rhs_contracting_dims().to_vec(),
        lhs_batch_dims: config.lhs_batch_dims().to_vec(),
        rhs_batch_dims: config.rhs_batch_dims().to_vec(),
    }
}

pub(crate) fn tenferro_error(op: &'static str, err: tenferro_tensor::Error) -> DenseError {
    DenseError::Backend {
        backend: DenseBackend::Tenferro,
        op,
        message: err.to_string(),
    }
}
