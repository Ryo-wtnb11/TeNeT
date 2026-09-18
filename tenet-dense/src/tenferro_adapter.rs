use num_complex::{Complex32, Complex64};

use crate::executor::{batch_offset, strided_batch_run_len};
use crate::layout::strides_to_isize;
use crate::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseGemmBatchJob, DenseOwned,
    DenseRead, DenseScalar, DenseTensor, DenseView, DenseViewMut, DenseWrite, MatrixOp,
};

use std::sync::Arc;

#[cfg(test)]
use std::cell::RefCell;

#[cfg(test)]
thread_local! {
    static OWNED_FULL_SVD_INPUT_POINTERS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn reset_owned_full_svd_input_pointers() {
    OWNED_FULL_SVD_INPUT_POINTERS.with(|pointers| pointers.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn owned_full_svd_input_pointers() -> Vec<usize> {
    OWNED_FULL_SVD_INPUT_POINTERS.with(|pointers| pointers.borrow().clone())
}

#[cfg(not(feature = "provider-inject"))]
use tenferro_cpu::with_cpu_exec_session;
use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuContext};
#[cfg(not(feature = "provider-inject"))]
use tenferro_linalg::{LinalgBackend, TensorLinalgExt, TensorReadLinalgExt};
#[cfg(not(feature = "provider-inject"))]
use tenferro_tensor::backend::{BackendSession, BackendSessionHost};
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, DotGeneralConfig, TensorDot, TensorRead, TensorView,
    TensorViewMut, TensorWrite, TypedTensorView, TypedTensorViewMut,
};

#[derive(Clone, Copy)]
struct StridedBatchRunLayout {
    lhs_shape: [usize; 3],
    lhs_strides: [usize; 3],
    lhs_job_offset: usize,
    rhs_shape: [usize; 3],
    rhs_strides: [usize; 3],
    rhs_job_offset: usize,
    dst_shape: [usize; 3],
    dst_strides: [usize; 3],
    dst_job_offset: usize,
}

fn run_partition_covers(jobs: &[DenseGemmBatchJob], runs: &[usize]) -> bool {
    let mut start = 0usize;
    for &run_len in runs {
        if run_len == 0 {
            return false;
        }
        let Some(end) = start.checked_add(run_len) else {
            return false;
        };
        if end > jobs.len() {
            return false;
        }
        start = end;
    }
    start == jobs.len()
}

/// One CPU execution context — parallelism hint plus the rayon pool behind
/// multi-threaded CPU work — meant to be shared by EVERY executor a runtime
/// mints (its dense factorization executor, executor-pool mints, and all
/// per-rule transform backends). Sharing is the point: each
/// `CpuBackend::new()`/`with_threads(n>1)` otherwise builds its own eager
/// rayon pool, and a runtime minting dozens of executors multiplies that into
/// hundreds of idle threads (the macOS `WouldBlock` thread-cap failure on
/// TeNeT#155's context pool). Opaque so callers never name tenferro types.
///
/// Buffer pools are deliberately NOT shared: each executor keeps its own
/// `CpuBackend`/`BufferPool` (scratch reuse is per-executor state; sharing it
/// would put a lock on the GEMM scratch path).
#[derive(Clone, Debug)]
pub struct SharedCpuContext {
    ctx: Arc<CpuContext>,
}

impl SharedCpuContext {
    /// Environment-driven context (`RAYON_NUM_THREADS`, else the machine's
    /// available parallelism) — matches what `CpuBackend::new` reads per call.
    pub fn from_env() -> Self {
        Self {
            ctx: Arc::new(CpuContext::from_env()),
        }
    }

    /// Fixed thread count; `1` builds no pool at all (fully serial), same as
    /// the per-executor constructors it replaces.
    pub fn with_threads(threads: usize) -> Result<Self, DenseError> {
        CpuContext::with_threads(threads)
            .map(|ctx| Self { ctx: Arc::new(ctx) })
            .map_err(|err| tenferro_error("SharedCpuContext::with_threads", err))
    }

    pub fn num_threads(&self) -> usize {
        self.ctx.num_threads()
    }

    /// Identity check for regression tests: do two handles share one context
    /// (hence one rayon pool)?
    #[doc(hidden)]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.ctx, &other.ctx)
    }
}

#[derive(Debug)]
pub struct DefaultDenseExecutor {
    backend: CpuBackend,
    matmul_config: DotGeneralConfig,
    strided_batch_matmul_config: DotGeneralConfig,
    grouped_cache: <CpuBackend as BackendRuntimeCache>::RuntimeCache,
    grouped_jobs: Vec<GroupedGemmJob>,
    // The runtime-shared context this executor was built on, if any (see
    // `with_shared_context`). Kept here rather than read back off the backend:
    // tenferro's `CpuBackend::linalg_context` accessor is `cpu-faer`-gated, so
    // a hook based on it does not compile on BLAS-provider builds.
    shared_ctx: Option<SharedCpuContext>,
    // Test-only count of low-level batch seam submissions: one per grouped or
    // strided call and one per job in the op-bearing serial fallback.
    #[cfg(test)]
    seam_dispatches: usize,
}

impl DefaultDenseExecutor {
    pub fn new() -> Self {
        Self::from_backend(CpuBackend::new())
    }

    pub fn with_threads(threads: usize) -> Result<Self, DenseError> {
        // `.into()`: since tenferro #1376 the BLAS-provider builds of these
        // constructors return `CpuBackendError`, while faer builds return the
        // crate `Error`; `Into` is identity on the latter, so one spelling
        // compiles under every provider feature.
        CpuBackend::with_threads(threads)
            .map(Self::from_backend)
            .map_err(|err| tenferro_error("CpuBackend::with_threads", err.into()))
    }

    /// Builds an executor on a specific CPU linear-algebra provider
    /// ([`CpuBackendKind::Faer`] or [`CpuBackendKind::Blas`]). Fails if the
    /// requested provider was not compiled in (e.g. `Blas` without a
    /// `cpu-blas`/`blas-*` feature) — the check happens here, not at first use.
    pub fn with_kind(kind: CpuBackendKind) -> Result<Self, DenseError> {
        CpuBackend::with_kind(kind)
            .map(Self::from_backend)
            .map_err(|err| tenferro_error("CpuBackend::with_kind", err.into()))
    }

    /// [`Self::with_kind`] plus an explicit thread count for the provider.
    pub fn with_threads_and_kind(threads: usize, kind: CpuBackendKind) -> Result<Self, DenseError> {
        CpuBackend::with_threads_and_kind(threads, kind)
            .map(Self::from_backend)
            .map_err(|err| tenferro_error("CpuBackend::with_threads_and_kind", err.into()))
    }

    /// Builds an executor on a runtime's shared [`SharedCpuContext`] (own
    /// backend + buffer pool, shared rayon pool) with an optional explicit
    /// provider. This is the constructor every runtime-minted executor must
    /// use — see [`SharedCpuContext`] for why.
    pub fn with_shared_context(
        ctx: &SharedCpuContext,
        kind: Option<CpuBackendKind>,
    ) -> Result<Self, DenseError> {
        match kind {
            // ponytail: tenferro has no pub context+kind constructor, so the
            // one non-default-kind combination (explicit Faer while a BLAS
            // provider is compiled in) keeps a private context/pool exactly as
            // before this seam existed (`shared_ctx` stays `None`). Lift when
            // tenferro exposes `from_context` with a kind.
            Some(kind) if kind != CpuBackendKind::default_compiled() => {
                Self::with_threads_and_kind(ctx.num_threads(), kind)
            }
            // `from_context` fixes the kind at `default_compiled`, so it also
            // covers an explicit request FOR that default.
            _ => {
                let mut executor =
                    Self::from_backend(CpuBackend::from_context(Arc::clone(&ctx.ctx)));
                executor.shared_ctx = Some(ctx.clone());
                Ok(executor)
            }
        }
    }

    /// Regression-test hook: does this executor run on `ctx`'s rayon pool?
    #[doc(hidden)]
    pub fn shares_cpu_context(&self, ctx: &SharedCpuContext) -> bool {
        self.shared_ctx.as_ref().is_some_and(|own| own.ptr_eq(ctx))
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
            shared_ctx: None,
            #[cfg(test)]
            seam_dispatches: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn reset_seam_dispatches(&mut self) {
        self.seam_dispatches = 0;
    }

    #[cfg(test)]
    pub(crate) fn seam_dispatches(&self) -> usize {
        self.seam_dispatches
    }

    /// Routes one typed identity batch by what the two tenferro entries can
    /// represent, not by a run-length cutoff. The strided entry is one rank-3
    /// contraction, so it can carry a batch iff the batch is ONE affine run:
    /// `runs == [jobs.len()]`, at least two jobs (a step needs two offsets), and
    /// a destination step of at least `rows * cols`, which is the O(1) proof
    /// that the rank-3 destination view is self-disjoint (tenferro's dot path
    /// does not check destination overlap; its grouped validator does). The
    /// grouped entry — TensorKit `mul!` per matched sector, QSpace
    /// `contract_matchAB_groupC` -> `contractDATA_group`, a destination-grouped
    /// GEMM list with no cutoff — represents every batch, so everything else
    /// goes there as ONE submission built from `jobs` (never from `runs`, so a
    /// malformed partition cannot skip or over-index a job).
    ///
    /// Why not keep a per-run cutoff: it was an empirical constant in tensor
    /// semantics (user decision 2026-09-14). The disadvantage accepted here: a
    /// long run plus residual jobs now goes grouped, paying tenferro's
    /// O(J log J) grouped validation per call instead of the cached O(1)
    /// strided analysis plus O(S log S) for the residual.
    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_route_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
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
        if jobs.is_empty() {
            return Ok(());
        }
        let single_self_disjoint_run = runs == [jobs.len()]
            && jobs.len() >= 2
            && jobs[1]
                .dst_offset
                .checked_sub(jobs[0].dst_offset)
                .zip(jobs[0].rows.checked_mul(jobs[0].cols))
                .is_some_and(|(dst_step, block)| dst_step >= block);
        if single_self_disjoint_run {
            return self.matmul_strided_batch_run_typed(
                output,
                lhs,
                rhs,
                jobs,
                0,
                MatrixOp::Identity,
                MatrixOp::Identity,
                alpha,
                beta,
                "strided_batch_gemm",
                wrap_write,
                wrap_read,
            );
        }
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
        self.matmul_grouped_bundle_typed(output, lhs, rhs, alpha, beta, wrap_write, wrap_read)
    }

    /// Single grouped-gemm call over the jobs staged in `self.grouped_jobs`.
    /// One seam dispatch regardless of how many runs fed it.
    #[allow(clippy::too_many_arguments)]
    fn matmul_grouped_bundle_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
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
            self.seam_dispatches += 1;
        }
        // shape/strides carry the view's own 'a lifetime, so capture them before
        // the mutable data borrow when rebuilding the full-buffer write view.
        let shape = output.shape();
        let strides = output.strides();
        let offset = output.offset();
        let out_view = DenseViewMut::new(output.data_mut(), shape, strides, offset)?;
        let lhs = TensorRead::from_view(tenferro_view(wrap_read(lhs))?);
        let rhs = TensorRead::from_view(tenferro_view(wrap_read(rhs))?);
        let output = TensorWrite::from_view(tenferro_view_mut(wrap_write(out_view))?);
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
    fn matmul_batch_axpby_ops_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
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
        // A valid covering partition with one entry per job contains only
        // singletons; malformed equal-count partitions need the same fallback.
        if runs.len() == jobs.len() {
            return self.matmul_batch_axpby_ops_serial_typed(
                output, lhs, rhs, jobs, 0, lhs_op, rhs_op, alpha, beta, wrap_write, wrap_read,
            );
        }
        if !run_partition_covers(jobs, runs) {
            return self.matmul_batch_axpby_ops_serial_typed(
                output, lhs, rhs, jobs, 0, lhs_op, rhs_op, alpha, beta, wrap_write, wrap_read,
            );
        }

        let mut start = 0usize;
        for &run_len in runs {
            let end = start + run_len;
            let run = &jobs[start..end];
            let can_batch = run_len >= 2
                && run[0].rows != 0
                && run[0].contracted != 0
                && run[0].cols != 0
                && strided_batch_run_len(run, 0) == run_len;
            let layout = can_batch
                .then(|| self.strided_batch_run_layout(run, lhs_op, rhs_op))
                .and_then(Result::ok)
                .filter(|layout| self.strided_batch_run_layout_admitted(output, lhs, rhs, layout));
            if let Some(layout) = layout {
                self.matmul_strided_batch_run_layout_typed(
                    output,
                    lhs,
                    rhs,
                    layout,
                    start,
                    lhs_op,
                    rhs_op,
                    alpha,
                    beta,
                    "matmul_batch_axpby_with_ops_into",
                    wrap_write,
                    wrap_read,
                )?;
            } else {
                self.matmul_batch_axpby_ops_serial_typed(
                    output, lhs, rhs, run, start, lhs_op, rhs_op, alpha, beta, wrap_write,
                    wrap_read,
                )?;
            }
            start = end;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_ops_serial_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        cache_start: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
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
        let output_base = output.offset();
        for (relative_slot, job) in jobs.iter().enumerate() {
            let lhs_shape = [job.rows, job.contracted];
            let lhs_strides = match lhs_op {
                MatrixOp::Identity => [1, job.rows],
                MatrixOp::Transpose | MatrixOp::Adjoint => [job.contracted, 1],
            };
            let rhs_shape = [job.contracted, job.cols];
            let rhs_strides = match rhs_op {
                MatrixOp::Identity => [1, job.contracted],
                MatrixOp::Transpose | MatrixOp::Adjoint => [job.cols, 1],
            };
            let output_shape = [job.rows, job.cols];
            let output_strides = [1, job.rows];
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
            let output_view = DenseViewMut::new(
                output.data_mut(),
                &output_shape,
                &output_strides,
                batch_offset(output_base, job.dst_offset)?,
            )?;
            let lhs = TensorRead::from_view(tenferro_view(wrap_read(lhs_view))?);
            let rhs = TensorRead::from_view(tenferro_view(wrap_read(rhs_view))?);
            let output = TensorWrite::from_view(tenferro_view_mut(wrap_write(output_view))?);
            let accumulation = tenferro_tensor::DotGeneralAccumulation {
                lhs_conj: lhs_op == MatrixOp::Adjoint,
                rhs_conj: rhs_op == MatrixOp::Adjoint,
                alpha: tenferro_scalar(alpha),
                beta: tenferro_scalar(beta),
            };
            #[cfg(test)]
            {
                self.seam_dispatches += 1;
            }
            BackendCachedDot::dot_general_read_into_accum_cached(
                &mut self.backend,
                &mut self.grouped_cache,
                Some(cache_start + relative_slot),
                lhs,
                rhs,
                &self.matmul_config,
                accumulation,
                output,
            )
            .map_err(|err| tenferro_error("matmul_batch_axpby_with_ops_into", err))?;
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
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
        error_op: &'static str,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<(), DenseError>
    where
        T: 'static,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
    {
        let layout = self.strided_batch_run_layout(run, lhs_op, rhs_op)?;
        self.matmul_strided_batch_run_layout_typed(
            output, lhs, rhs, layout, cache_slot, lhs_op, rhs_op, alpha, beta, error_op,
            wrap_write, wrap_read,
        )
    }

    fn strided_batch_run_layout(
        &self,
        run: &[DenseGemmBatchJob],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
    ) -> Result<StridedBatchRunLayout, DenseError> {
        // Two jobs define the structural batch stride; singletons cannot.
        debug_assert!(run.len() >= 2, "strided run must hold at least two jobs");
        let first = &run[0];
        let next = &run[1];
        let run_len = run.len();
        let lhs_batch_stride =
            next.lhs_offset
                .checked_sub(first.lhs_offset)
                .ok_or(DenseError::OffsetOverflow {
                    value: first.lhs_offset,
                })?;
        let rhs_batch_stride =
            next.rhs_offset
                .checked_sub(first.rhs_offset)
                .ok_or(DenseError::OffsetOverflow {
                    value: first.rhs_offset,
                })?;
        let dst_batch_stride =
            next.dst_offset
                .checked_sub(first.dst_offset)
                .ok_or(DenseError::OffsetOverflow {
                    value: first.dst_offset,
                })?;
        let lhs_shape = [first.rows, first.contracted, run_len];
        let lhs_strides = match lhs_op {
            MatrixOp::Identity => [1, first.rows, lhs_batch_stride],
            MatrixOp::Transpose | MatrixOp::Adjoint => [first.contracted, 1, lhs_batch_stride],
        };
        let rhs_shape = [first.contracted, first.cols, run_len];
        let rhs_strides = match rhs_op {
            MatrixOp::Identity => [1, first.contracted, rhs_batch_stride],
            MatrixOp::Transpose | MatrixOp::Adjoint => [first.cols, 1, rhs_batch_stride],
        };
        let dst_shape = [first.rows, first.cols, run_len];
        let dst_strides = [1, first.rows, dst_batch_stride];
        Ok(StridedBatchRunLayout {
            lhs_shape,
            lhs_strides,
            lhs_job_offset: first.lhs_offset,
            rhs_shape,
            rhs_strides,
            rhs_job_offset: first.rhs_offset,
            dst_shape,
            dst_strides,
            dst_job_offset: first.dst_offset,
        })
    }

    fn strided_batch_run_layout_admitted<T>(
        &self,
        output: &DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        layout: &StridedBatchRunLayout,
    ) -> bool {
        let Some(lhs_offset) = lhs.offset().checked_add(layout.lhs_job_offset) else {
            return false;
        };
        let Some(rhs_offset) = rhs.offset().checked_add(layout.rhs_job_offset) else {
            return false;
        };
        let Some(dst_offset) = output.offset().checked_add(layout.dst_job_offset) else {
            return false;
        };
        let offsets_fit = [lhs_offset, rhs_offset, dst_offset]
            .into_iter()
            .all(|offset| isize::try_from(offset).is_ok());
        let strides_fit = layout
            .lhs_strides
            .into_iter()
            .chain(layout.rhs_strides)
            .chain(layout.dst_strides)
            .all(|stride| isize::try_from(stride).is_ok());
        offsets_fit
            && strides_fit
            && DenseView::new(
                lhs.data(),
                &layout.lhs_shape,
                &layout.lhs_strides,
                lhs_offset,
            )
            .is_ok()
            && DenseView::new(
                rhs.data(),
                &layout.rhs_shape,
                &layout.rhs_strides,
                rhs_offset,
            )
            .is_ok()
            && DenseView::new(
                output.data(),
                &layout.dst_shape,
                &layout.dst_strides,
                dst_offset,
            )
            .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_strided_batch_run_layout_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        layout: StridedBatchRunLayout,
        cache_slot: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
        error_op: &'static str,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<(), DenseError>
    where
        T: 'static,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
    {
        let lhs_offset = batch_offset(lhs.offset(), layout.lhs_job_offset)?;
        let lhs_view = DenseView::new(
            lhs.data(),
            &layout.lhs_shape,
            &layout.lhs_strides,
            lhs_offset,
        )?;
        let rhs_offset = batch_offset(rhs.offset(), layout.rhs_job_offset)?;
        let rhs_view = DenseView::new(
            rhs.data(),
            &layout.rhs_shape,
            &layout.rhs_strides,
            rhs_offset,
        )?;
        let dst_offset = batch_offset(output.offset(), layout.dst_job_offset)?;
        let dst_view = DenseViewMut::new(
            output.data_mut(),
            &layout.dst_shape,
            &layout.dst_strides,
            dst_offset,
        )?;
        let lhs = TensorRead::from_view(tenferro_view(wrap_read(lhs_view))?);
        let rhs = TensorRead::from_view(tenferro_view(wrap_read(rhs_view))?);
        let output = TensorWrite::from_view(tenferro_view_mut(wrap_write(dst_view))?);
        let accumulation = tenferro_tensor::DotGeneralAccumulation {
            lhs_conj: lhs_op == MatrixOp::Adjoint,
            rhs_conj: rhs_op == MatrixOp::Adjoint,
            alpha: tenferro_scalar(alpha),
            beta: tenferro_scalar(beta),
        };
        #[cfg(test)]
        {
            self.seam_dispatches += 1;
        }
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
        .map_err(|err| tenferro_error(error_op, err))
    }
}

impl Default for DefaultDenseExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl DenseExecutor for DefaultDenseExecutor {
    fn supports_svd_full(&self) -> bool {
        #[cfg(all(feature = "cpu-faer", not(feature = "provider-inject")))]
        {
            self.backend.kind() == CpuBackendKind::Faer
        }
        #[cfg(any(not(feature = "cpu-faer"), feature = "provider-inject"))]
        {
            false
        }
    }

    fn svd_full_owned(
        &mut self,
        input: DenseOwned,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = (input, rows, cols);
            return Err(DenseError::Unsupported {
                op: "svd_full_owned",
                message: "executor does not implement owned full-matrices SVD".to_string(),
            });
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            if !self.supports_svd_full() {
                return Err(DenseError::Unsupported {
                    op: "svd_full_owned",
                    message: "executor does not implement owned full-matrices SVD".to_string(),
                });
            }
            let expected = rows
                .checked_mul(cols)
                .ok_or(DenseError::ElementCountOverflow)?;
            let actual = match &input {
                DenseOwned::F32(data) => data.len(),
                DenseOwned::F64(data) => data.len(),
                DenseOwned::C32(data) => data.len(),
                DenseOwned::C64(data) => data.len(),
            };
            if actual != expected {
                return Err(DenseError::Backend {
                    backend: DenseBackend::Tenferro,
                    op: "svd_full_owned",
                    message: format!(
                        "owned full SVD input storage length mismatch: source {actual}, expected {expected}",
                    ),
                });
            }
            if rows == 0 || cols == 0 {
                return Err(DenseError::Unsupported {
                    op: "svd_full_owned",
                    message: "zero-extent full-matrices SVD is unsupported".to_string(),
                });
            }
            let input = match input {
                DenseOwned::F32(data) => {
                    tenferro_tensor::Tensor::from_vec_col_major(vec![rows, cols], data)
                }
                DenseOwned::F64(data) => {
                    tenferro_tensor::Tensor::from_vec_col_major(vec![rows, cols], data)
                }
                DenseOwned::C32(data) => {
                    tenferro_tensor::Tensor::from_vec_col_major(vec![rows, cols], data)
                }
                DenseOwned::C64(data) => {
                    tenferro_tensor::Tensor::from_vec_col_major(vec![rows, cols], data)
                }
            }
            .map_err(|err| tenferro_error("svd_full_owned", err))?;
            #[cfg(test)]
            OWNED_FULL_SVD_INPUT_POINTERS.with(|pointers| {
                let pointer = match &input {
                    tenferro_tensor::Tensor::F32(tensor) => {
                        tensor.as_slice().unwrap().as_ptr() as usize
                    }
                    tenferro_tensor::Tensor::F64(tensor) => {
                        tensor.as_slice().unwrap().as_ptr() as usize
                    }
                    tenferro_tensor::Tensor::C32(tensor) => {
                        tensor.as_slice().unwrap().as_ptr() as usize
                    }
                    tenferro_tensor::Tensor::C64(tensor) => {
                        tensor.as_slice().unwrap().as_ptr() as usize
                    }
                    _ => unreachable!("DenseOwned only contains supported full-SVD dtypes"),
                };
                pointers.borrow_mut().push(pointer);
            });
            let outputs = self
                .backend
                .with_backend_session(|session| {
                    with_cpu_exec_session(session, |exec| exec.svd_full(&input))
                })
                .ok_or_else(|| DenseError::Unsupported {
                    op: "svd_full_owned",
                    message: "CPU backend session unavailable".to_string(),
                })?
                .map_err(|err| tenferro_error("svd_full_owned", err))?;
            if outputs.len() != 3 {
                return Err(DenseError::Backend {
                    backend: DenseBackend::Tenferro,
                    op: "svd_full_owned",
                    message: "dense full SVD must return exactly (U, S, Vh)".to_string(),
                });
            }
            Ok(outputs
                .into_iter()
                .map(DenseTensor::from_tenferro)
                .collect())
        }
    }

    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            return Err(linalg_unavailable("svd"));
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            with_cpu_linalg(&mut self.backend, |exec| {
                TensorRead::from_view(input).svd_read(exec)
            })
            .map(|(u, s, vt)| {
                vec![u, s, vt]
                    .into_iter()
                    .map(DenseTensor::from_tenferro)
                    .collect()
            })
            .map_err(|err| tenferro_error("svd_read", err))
        }
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            return Err(linalg_unavailable("qr"));
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            with_cpu_linalg(&mut self.backend, |exec| {
                TensorRead::from_view(input).qr_read(exec)
            })
            .map(|(q, r)| {
                vec![q, r]
                    .into_iter()
                    .map(DenseTensor::from_tenferro)
                    .collect()
            })
            .map_err(|err| tenferro_error("qr_read", err))
        }
    }

    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            return Err(linalg_unavailable("eig"));
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            with_cpu_linalg(&mut self.backend, |exec| {
                TensorRead::from_view(input).eig_read(exec)
            })
            .map(|(values, vectors)| {
                vec![values, vectors]
                    .into_iter()
                    .map(DenseTensor::from_tenferro)
                    .collect()
            })
            .map_err(|err| tenferro_error("eig_read", err))
        }
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            return Err(linalg_unavailable("eigh"));
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            with_cpu_linalg(&mut self.backend, |exec| {
                TensorRead::from_view(input).eigh_read(exec)
            })
            .map(|(values, vectors)| {
                vec![values, vectors]
                    .into_iter()
                    .map(DenseTensor::from_tenferro)
                    .collect()
            })
            .map_err(|err| tenferro_error("eigh_read", err))
        }
    }

    fn solve_into(
        &mut self,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = (a, b, x);
            return Err(linalg_unavailable("solve_into"));
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            if x.dtype() != b.dtype() {
                return Err(DenseError::DTypeMismatch {
                    op: "solve_into",
                    expected: b.dtype(),
                    actual: x.dtype(),
                });
            }
            if x.shape() != b.shape() {
                return Err(DenseError::ShapeMismatch {
                    op: "solve_into",
                    expected: b.shape().to_vec(),
                    actual: x.shape().to_vec(),
                });
            }
            let a = tenferro_view(a)?;
            let b = tenferro_view(b)?;
            let x = tenferro_view_mut(x)?;
            with_cpu_linalg(&mut self.backend, |exec| {
                TensorRead::from_view(a).solve_read_into(
                    TensorRead::from_view(b),
                    TensorWrite::from_view(x),
                    exec,
                )
            })
            .map_err(tenferro_solve_error)
        }
    }

    // SVD and EIGH values-only calls currently materialize the borrowed input,
    // compute full factors through supported owned extensions, and discard the
    // vectors; general EIG below already uses native owned `eigvals`.
    // TODO(#880): adopt a supported concrete values-only API once its stability
    // and ownership are confirmed. The installed hidden backend hook is
    // callable, but carries no supported downstream guarantee.
    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            return Err(linalg_unavailable("svd_vals"));
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            let owned = self
                .backend
                .with_backend_session(|exec| exec.to_contiguous_read(TensorRead::from_view(input)))
                .map_err(|err| tenferro_error("svd_values", err))?;
            let (_, values, _) = with_cpu_linalg(&mut self.backend, |exec| owned.svd(exec))
                .map_err(|err| tenferro_error("svd_values", err))?;
            Ok(DenseTensor::from_tenferro(values))
        }
    }

    fn eigh_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            return Err(linalg_unavailable("eigh_vals"));
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            let owned = self
                .backend
                .with_backend_session(|exec| exec.to_contiguous_read(TensorRead::from_view(input)))
                .map_err(|err| tenferro_error("eigh_values", err))?;
            let (values, _) = with_cpu_linalg(&mut self.backend, |exec| owned.eigh(exec))
                .map_err(|err| tenferro_error("eigh_values", err))?;
            Ok(DenseTensor::from_tenferro(values))
        }
    }

    fn eig_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            return Err(linalg_unavailable("eig_vals"));
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            let owned = self
                .backend
                .with_backend_session(|exec| exec.to_contiguous_read(TensorRead::from_view(input)))
                .map_err(|err| tenferro_error("eig_values", err))?;
            with_cpu_linalg(&mut self.backend, |exec| owned.eigvals(exec))
                .map(DenseTensor::from_tenferro)
                .map_err(|err| tenferro_error("eig_values", err))
        }
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
        runs: &[usize],
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        match (output, lhs, rhs) {
            (DenseWrite::F32(mut out), DenseRead::F32(lhs), DenseRead::F32(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f32>| DenseWrite::F32(view),
                    |view: DenseView<'_, f32>| DenseRead::F32(view),
                ),
            (DenseWrite::F64(mut out), DenseRead::F64(lhs), DenseRead::F64(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f64>| DenseWrite::F64(view),
                    |view: DenseView<'_, f64>| DenseRead::F64(view),
                ),
            (DenseWrite::C32(mut out), DenseRead::C32(lhs), DenseRead::C32(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex32>| DenseWrite::C32(view),
                    |view: DenseView<'_, Complex32>| DenseRead::C32(view),
                ),
            (DenseWrite::C64(mut out), DenseRead::C64(lhs), DenseRead::C64(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex64>| DenseWrite::C64(view),
                    |view: DenseView<'_, Complex64>| DenseRead::C64(view),
                ),
            _ => Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "matmul_batch_axpby_into",
                message: "batched matmul requires matching f32/f64/c32/c64 operands".to_string(),
            }),
        }
    }

    #[allow(clippy::redundant_closure)]
    fn matmul_batch_axpby_with_ops_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        if lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity {
            return self.matmul_batch_axpby_into(output, lhs, rhs, jobs, runs, alpha, beta);
        }
        match (output, lhs, rhs) {
            (DenseWrite::F32(mut out), DenseRead::F32(lhs), DenseRead::F32(rhs)) => self
                .matmul_batch_axpby_ops_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    lhs_op,
                    rhs_op,
                    alpha,
                    beta,
                    // Constructor functions capture one concrete lifetime; closures
                    // remain generic over each temporary view borrowed below.
                    |view| DenseWrite::F32(view),
                    |view| DenseRead::F32(view),
                ),
            (DenseWrite::F64(mut out), DenseRead::F64(lhs), DenseRead::F64(rhs)) => self
                .matmul_batch_axpby_ops_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    lhs_op,
                    rhs_op,
                    alpha,
                    beta,
                    |view| DenseWrite::F64(view),
                    |view| DenseRead::F64(view),
                ),
            (DenseWrite::C32(mut out), DenseRead::C32(lhs), DenseRead::C32(rhs)) => self
                .matmul_batch_axpby_ops_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    lhs_op,
                    rhs_op,
                    alpha,
                    beta,
                    |view| DenseWrite::C32(view),
                    |view| DenseRead::C32(view),
                ),
            (DenseWrite::C64(mut out), DenseRead::C64(lhs), DenseRead::C64(rhs)) => self
                .matmul_batch_axpby_ops_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    lhs_op,
                    rhs_op,
                    alpha,
                    beta,
                    |view| DenseWrite::C64(view),
                    |view| DenseRead::C64(view),
                ),
            _ => Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "matmul_batch_axpby_with_ops_into",
                message: "op-bearing batch requires matching f32/f64/c32/c64 operands".to_string(),
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
    let offset = isize::try_from(view.offset()).map_err(|_| DenseError::OffsetOverflow {
        value: view.offset(),
    })?;
    if view.shape().len() == 1 && view.strides() == [1] {
        return TypedTensorView::from_slice([view.shape()[0]], [1_isize], offset, view.data())
            .map_err(|err| tenferro_error("TypedTensorView::from_slice", err));
    }
    let strides = strides_to_isize(view.strides())?;
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
    let offset =
        isize::try_from(offset).map_err(|_| DenseError::OffsetOverflow { value: offset })?;
    if shape.len() == 1 && strides == [1] {
        return TypedTensorViewMut::from_slice([shape[0]], [1_isize], offset, data)
            .map_err(|err| tenferro_error("TypedTensorViewMut::from_slice", err));
    }
    let strides = strides_to_isize(strides)?;
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

#[cfg(not(feature = "provider-inject"))]
fn tenferro_solve_error(err: tenferro_tensor::Error) -> DenseError {
    if err.kind() == tenferro_tensor::ErrorKind::NumericalFailure {
        DenseError::NumericalFailure {
            backend: DenseBackend::Tenferro,
            op: "solve_into",
            message: err.to_string(),
        }
    } else {
        tenferro_error("solve_into", err)
    }
}

#[cfg(not(feature = "provider-inject"))]
fn with_cpu_linalg<R: Send>(
    backend: &mut CpuBackend,
    f: impl FnOnce(&mut dyn BackendSession) -> tenferro_tensor::Result<R> + Send,
) -> tenferro_tensor::Result<R> {
    backend.with_backend_session(f)
}

#[cfg(feature = "provider-inject")]
fn linalg_unavailable(op: &'static str) -> DenseError {
    DenseError::Unsupported {
        op,
        message: "provider-inject requires a registered BLAS/LAPACK provider".to_string(),
    }
}
