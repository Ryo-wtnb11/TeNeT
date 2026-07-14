use num_complex::{Complex32, Complex64};

use crate::executor::batch_offset;
use crate::layout::strides_to_isize;
use crate::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseGemmBatchJob, DenseRead,
    DenseScalar, DenseTensor, DenseView, DenseViewMut, DenseWrite,
};

use std::sync::Arc;

use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuContext};
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, DotGeneralConfig, Tensor, TensorDot, TensorRead,
    TensorView, TensorViewMut, TensorWrite, TypedTensorView, TypedTensorViewMut,
};

/// Minimum plan-time run length routed to the strided-batch seam; shorter runs
/// and singletons are bundled into one grouped-gemm call. The strided seam's
/// per-call setup (`analyse_gemm_cached`, stride normalization, layout checks)
/// only amortizes over a long run — and tenferro loops internally either way
/// (no true batched BLAS), so short runs pay that setup for no fusion win.
///
/// Derivation (issue #103 A/B measurements, Apple M4 Max, Accelerate, 1 thread):
/// forcing the small d=4 5-group batch through grouped restored U1 compose from
/// 6.3 to ~4.2 us and swap+out to ~12.6 us, while d=8/d=16 showed no strided
/// advantage for short runs; the only case where strided wins is a long run
/// (the SU2 recoupling run=7). 4 is the empirically-validated old
/// `STRIDED_BATCH_MIN_JOBS` threshold (pre-b8bb92e), re-derived here as a
/// plan-time cost-model constant (peer of the contraction-order cost model),
/// not a runtime kernel knob.
const STRIDED_RUN_MIN: usize = 4;

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
    // Test-only count of low-level seam dispatches (one per grouped-gemm call,
    // one per strided-batch call). A structural proxy that stays flat as a
    // batch fragments into more runs — see the dispatch-count test for #103.
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

    /// Routes one typed batch by its plan-time run partition (see issue #103).
    /// Runs of length >= [`STRIDED_RUN_MIN`] go to the strided-batch seam; every
    /// shorter run and singleton is bundled — preserving original job order —
    /// into ONE grouped-gemm call. Because a batch's destination ranges are
    /// pairwise disjoint (a [`DenseGemmBatchJob`] invariant), job order never
    /// affects results, so this bundling is byte-identical to the per-run
    /// dispatch it replaces while never fragmenting a small batch into one seam
    /// call per run — which was the d=4 regression this fixes.
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
        debug_assert_eq!(
            runs.iter().sum::<usize>(),
            jobs.len(),
            "run partition must cover every job"
        );
        self.grouped_jobs.clear();
        let mut start = 0usize;
        for &run_len in runs {
            let run = &jobs[start..start + run_len];
            if run_len >= STRIDED_RUN_MIN {
                self.matmul_strided_batch_run_typed(
                    output, lhs, rhs, run, start, alpha, beta, wrap_write, wrap_read,
                )?;
            } else {
                // Bundle short-run/singleton jobs, in order, for one grouped call.
                self.grouped_jobs.extend(run.iter().map(|job| {
                    GroupedGemmJob::new(
                        job.dst_offset,
                        job.lhs_offset,
                        job.rhs_offset,
                        job.rows,
                        job.contracted,
                        job.cols,
                    )
                }));
            }
            start += run_len;
        }
        if !self.grouped_jobs.is_empty() {
            self.matmul_grouped_bundle_typed(output, lhs, rhs, alpha, beta, wrap_write, wrap_read)?;
        }
        Ok(())
    }

    /// Single grouped-gemm call over the jobs already staged in
    /// `self.grouped_jobs` (the bundled short runs and singletons). One seam
    /// dispatch regardless of how many runs fed it.
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
            self.seam_dispatches += 1;
        }
        // Only reached from the router with a run of length >= STRIDED_RUN_MIN
        // (>= 2), so the batch strides come from the first two jobs' constant
        // step (guaranteed constant by the plan-time run partition). The old
        // singleton fallback is gone: singletons now route to the grouped seam.
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
