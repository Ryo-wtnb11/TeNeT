use super::*;

/// A snapshot of the backend's cuTENSOR contraction plan cache.
///
/// Mirrors Tenferro's own cache statistics, plus this context's plan-entry
/// reservation ledger; it exists so TeNeT callers never name a tenferro type.
/// Observability only.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct CudaPlanCacheStats {
    pub entries: usize,
    pub retained_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Plan entries currently reserved through
    /// [`CudaDenseContext::reserve_plan_entries`], over every consumer.
    pub reserved_entries: usize,
    /// Entries requested but not granted because the reservation budget was
    /// exhausted, summed over the context's lifetime.
    pub reservation_shortfall: u64,
}
/// The device operands a region call needs but never varies: the ones
/// template, whose first element is the 1x1 coefficient of an unscaled move
/// and whose packed prefix a trace contracts its diagonal against, and the
/// zero template read as the packed source of a region fill, plus the scaled
/// template a trace with a non-unit scale contracts its diagonal against.
///
/// One pair per payload dtype, created on first use of that dtype rather than
/// at [`CudaDenseContext::warm_up`], so the warm-up's documented fixed cost
/// and its counter deltas are unchanged. Each template grows monotonically to
/// the longest region it has served; it is bounded by the largest single
/// region (fill, or traced extent `prod t_k`) a caller has submitted, not by
/// the number of calls.
///
/// The operands are payload-typed, so the slot is keyed by the dtype
/// ([`CudaScalar::OPERAND_SLOT`]), not by realness: an `f32` call must never
/// be handed the `f64` context `1`.
#[derive(Default)]
pub(super) struct ScalarOperands {
    pub(super) ones: Option<CudaDenseStorage>,
    pub(super) zeros: Option<CudaDenseStorage>,
    /// `α` repeated: the trace's per-element scale (TensorOperations'
    /// `Scaler(α)` before the sum). Filled on the device from the ones
    /// template whenever `α` changes, so only growth uploads.
    pub(super) scaled: Option<CudaDenseStorage>,
    /// The value and length of the filled prefix of `scaled`.
    scaled_prefix: Option<(ContractionScalar, usize)>,
    /// Payload bytes per element of this slot's dtype, recorded when a buffer
    /// is created so [`CudaDenseContext::scalar_operand_bytes`] needs no dtype
    /// parameter. Zero while the slot is empty.
    element_bytes: usize,
}
/// One [`ScalarOperands`] slot per admitted payload dtype.
pub(super) const SCALAR_OPERAND_SLOTS: usize = 4;
/// Owns the tenferro CUDA backend for one device ordinal.
pub struct CudaDenseContext {
    pub(super) backend: CudaBackend,
    pub(super) device: usize,
    identity: u64,
    operands: [ScalarOperands; SCALAR_OPERAND_SLOTS],
    plan_ledger: PlanEntryLedger,
}
/// Process-wide context counter. A monotonic ticket rather than the context's
/// address, so a cache keyed by context identity can never mistake a freshly
/// allocated context for a dropped one that happened to reuse its address.
static NEXT_CONTEXT_IDENTITY: AtomicU64 = AtomicU64::new(1);
/// Puts every CubeCL command and every Tenferro vendor call of this process
/// on one CUDA stream per device, or reports why it cannot (#1391).
///
/// Why: CubeCL decides whether a stream must wait for a buffer's origin
/// stream from a cursor it records only when the buffer is bound, not when a
/// later command writes it (tensor4all/cubecl#16). Once a stream has synced
/// past a buffer's bind, it reads a later write to that buffer, or overwrites
/// a buffer another stream still reads, without waiting. With one stream all
/// work runs in enqueue order, so that cursor is never consulted for
/// correctness. Tenferro sizes its vendor-stream slots from the same setting,
/// so cuBLAS, cuSOLVER and cuTENSOR share the stream too, and its cross-slot
/// host sync disappears.
///
/// Enqueue order follows happens-before between threads because CubeCL's
/// server queue is FIFO and a vendor call first drains it: raw sessions
/// (cuSOLVER), memsets and scalar downloads call `flush_cubecl`; cuTENSOR and
/// cuBLAS rely on the blocking `get_resource` in Tenferro's `typed_device_ptr`.
/// Residual (tensor4all/tenferro-rs#1868, independent of the stream count and
/// present within one thread too): `typed_device_ptr` skips `get_resource`
/// for a buffer created on the calling thread whose address it has cached, so
/// a CubeCL kernel still queued unflushed into that buffer can reach the
/// stream after a later vendor call on it. In TeNeT only the empty-contraction
/// scale or fill of an owned destination can queue such a kernel.
///
/// Why not a TeNeT-side sync per overwrite: it covers only destinations TeNeT
/// knows it rewrote, not scratch, workspaces or pooled intermediates read in
/// flight by another thread, and costs a host sync per call.
///
/// The configuration is process-wide and fixed at CubeCL's first read of it,
/// so this sets it only if nothing has read it yet: from the same
/// `cubecl.toml` and environment CubeCL would read, with `max_streams = 1`
/// (a file value is overridden). If something else loaded it first with
/// another count, the device cannot be opened safely and this fails.
fn single_cubecl_stream() -> Result<(), DenseError> {
    use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};

    let mut slot = CubeClRuntimeConfig::storage().lock();
    let max_streams = match slot.as_ref() {
        Some(config) => config.streaming.max_streams,
        None => {
            let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
            config.streaming.max_streams = 1;
            *slot = Some(std::sync::Arc::new(config));
            1
        }
    };
    if max_streams != 1 {
        return Err(DenseError::Unsupported {
            op: "cuda_context",
            message: format!(
                "CubeCL was configured with streaming.max_streams = {max_streams} before \
                 the first CUDA context opened; TeNeT needs 1 so that a buffer written \
                 after its bind is never read early (tensor4all/cubecl#16)"
            ),
        });
    }
    Ok(())
}
impl CudaDenseContext {
    /// Opens `device`. The first call in the process also fixes CubeCL to one
    /// stream per device (see `single_cubecl_stream`), and every call fails with
    /// [`DenseError::Unsupported`] if CubeCL was already configured otherwise.
    pub fn new(device: usize) -> Result<Self, DenseError> {
        single_cubecl_stream()?;
        let ordinal = u32::try_from(device)
            .map_err(|_| cuda_error("cuda_context", "device ordinal exceeds u32"))?;
        let backend = CudaBackend::new(CudaDeviceId::from_ordinal(ordinal))
            .map_err(|err| cuda_error("cuda_context", err))?;
        let base = backend
            .cutensor_plan_cache_max_entries()
            .map_err(|err| cuda_error("cuda_plan_cache", err))?
            .get();
        Ok(Self {
            backend,
            device,
            identity: NEXT_CONTEXT_IDENTITY.fetch_add(1, Ordering::Relaxed),
            operands: std::array::from_fn(|_| ScalarOperands::default()),
            plan_ledger: PlanEntryLedger::new(
                base,
                plan_cache_entries_for(usize::MAX, DEFAULT_PLAN_CACHE_BUDGET_BYTES),
            ),
        })
    }

    /// Process-unique identity of this context, never reused while it lives.
    ///
    /// Device state cached against a context — an uploaded coefficient vector,
    /// a prepared plan — belongs to exactly one context, and this is the key
    /// that says so.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    fn operands<D: CudaScalar>(&self) -> &ScalarOperands {
        &self.operands[D::OPERAND_SLOT]
    }

    fn operands_mut<D: CudaScalar>(&mut self) -> &mut ScalarOperands {
        &mut self.operands[D::OPERAND_SLOT]
    }

    /// Uploads this dtype's ones template unless a resident one already holds
    /// at least `len` elements. Its first element is the `1` coefficient of an
    /// unscaled move; a longer prefix is the packed ones operand a trace
    /// contracts its diagonal against ([`cuda_region_trace_accumulate`]).
    pub(super) fn ensure_ones<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
        let usable = self
            .operands::<D>()
            .ones
            .as_ref()
            .is_some_and(|ones| ones.len() >= len);
        if !usable {
            self.operands_mut::<D>().ones = None;
            let ones = CudaDenseStorage::upload_owned(self, vec![D::ONE; len.max(1)])?;
            let slot = self.operands_mut::<D>();
            slot.element_bytes = std::mem::size_of::<D>();
            slot.ones = Some(ones);
        }
        Ok(())
    }

    /// Makes the first `len` elements of this dtype's scaled template equal
    /// `alpha`. Growth uploads `alpha` repeated (one H2D); a resident template
    /// of another value is refilled on the device as `alpha * ones`, one region
    /// submission and no transfer; the same value and length does nothing.
    pub(super) fn ensure_scaled<D: CudaScalar>(
        &mut self,
        alpha: D,
        len: usize,
    ) -> Result<(), DenseError> {
        let tag = alpha.contraction_scalar();
        let slot = self.operands::<D>();
        if slot
            .scaled_prefix
            .is_some_and(|(value, filled)| value == tag && filled >= len)
        {
            return Ok(());
        }
        if slot.scaled.as_ref().is_none_or(|scaled| scaled.len() < len) {
            self.operands_mut::<D>().scaled = None;
            let scaled = CudaDenseStorage::upload_owned(self, vec![alpha; len])?;
            let slot = self.operands_mut::<D>();
            slot.element_bytes = std::mem::size_of::<D>();
            slot.scaled = Some(scaled);
            slot.scaled_prefix = Some((tag, len));
            return Ok(());
        }
        self.ensure_ones::<D>(len)?;
        let region = CudaRegion::packed(&[len], 0)?;
        let Self {
            backend, operands, ..
        } = self;
        let slot = &mut operands[D::OPERAND_SLOT];
        let (Some(ones), Some(scaled)) = (slot.ones.as_ref(), slot.scaled.as_mut()) else {
            return Err(cuda_error(
                "cuda_scaled_template",
                "scalar operands are missing",
            ));
        };
        submit_region_axpby::<D>(
            backend,
            "cuda_scaled_template",
            ones,
            &region,
            false,
            alpha,
            ones,
            0,
            CudaRegionBeta::Overwrite,
            scaled,
            &region,
        )?;
        slot.scaled_prefix = Some((tag, len));
        Ok(())
    }

    /// Sizes this dtype's scaled template (see [`cuda_region_trace_accumulate`])
    /// for `len` elements up front, together with the ones template of the
    /// same length it is refilled from, so a trace with a non-unit scale
    /// uploads nothing afterwards: a later change of scale is a device
    /// refill. Never shrinks; reported and released with the other scalar
    /// operands.
    pub fn reserve_scaled_template<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
        if len == 0 {
            return Ok(());
        }
        self.ensure_ones::<D>(len)?;
        if self
            .operands::<D>()
            .scaled
            .as_ref()
            .is_some_and(|scaled| scaled.len() >= len)
        {
            return Ok(());
        }
        self.ensure_scaled::<D>(D::ONE, len)
    }

    /// Sizes this dtype's ones template for `len` elements up front, the
    /// counterpart of [`Self::reserve_zero_template`]: a caller that knows the
    /// largest trace extent it will submit reserves once and every later
    /// [`cuda_region_trace_accumulate`] is upload-free. Never shrinks; the
    /// bytes are reported by [`Self::scalar_operand_bytes`] and released by
    /// [`Self::release_scalar_operands`].
    pub fn reserve_ones_template<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
        if len == 0 {
            return Ok(());
        }
        self.ensure_ones::<D>(len)
    }

    /// Uploads this dtype's zero template unless a resident one already holds
    /// at least `len` elements. A prefix of a compact zero buffer is itself a
    /// compact zero buffer, so a longer template serves every shorter fill.
    pub(super) fn ensure_zeros<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
        let usable = self
            .operands::<D>()
            .zeros
            .as_ref()
            .is_some_and(|zeros| zeros.len() >= len);
        if !usable {
            // Drop the short template before allocating the longer one, so the
            // device holds one of them rather than both.
            self.operands_mut::<D>().zeros = None;
            let zeros = CudaDenseStorage::upload_owned(self, vec![D::ZERO; len])?;
            let slot = self.operands_mut::<D>();
            slot.element_bytes = std::mem::size_of::<D>();
            slot.zeros = Some(zeros);
        }
        Ok(())
    }

    /// Sizes this dtype's zero template for `len` elements up front.
    ///
    /// [`cuda_region_zero`] grows the template to the fill it is given, so a
    /// caller that visits `k` regions in ascending size pays `k` uploads for
    /// what is one buffer. A caller that knows its largest region — a device
    /// transform knows the largest inactive destination layout of its
    /// structure on the host, before any replay — reserves once here and pays
    /// none. It never shrinks an already longer template.
    ///
    /// The bytes it pins stay resident until [`Self::release_scalar_operands`]
    /// or the context is dropped, and are reported by
    /// [`Self::scalar_operand_bytes`]. It is the only device zero source;
    /// tenferro-rs#1834's native device fill would remove it.
    pub fn reserve_zero_template<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
        if len == 0 {
            return Ok(());
        }
        self.ensure_zeros::<D>(len)
    }

    /// Device bytes the lazily created scalar operands currently pin, over
    /// every dtype. Observability for the caller that owns the device memory
    /// budget; nothing here reads it back.
    pub fn scalar_operand_bytes(&self) -> usize {
        self.operands
            .iter()
            .map(|operands| {
                let elements = operands.ones.as_ref().map_or(0, CudaDenseStorage::len)
                    + operands.zeros.as_ref().map_or(0, CudaDenseStorage::len)
                    + operands.scaled.as_ref().map_or(0, CudaDenseStorage::len);
                elements * operands.element_bytes
            })
            .sum()
    }

    /// Frees every lazily created scalar operand. The next region call that
    /// needs one re-creates it, so this is a memory decision, never a
    /// correctness one.
    pub fn release_scalar_operands(&mut self) {
        self.operands = std::array::from_fn(|_| ScalarOperands::default());
    }

    /// Hands out the backend and this dtype's scalar operands at once.
    ///
    /// A whole-`self` accessor cannot express this: submitting against a
    /// context-owned operand needs `&mut` on the backend and `&` on the
    /// operand simultaneously, which is only sound because they are disjoint
    /// fields.
    pub(super) fn split_operands<D: CudaScalar>(&mut self) -> (&mut CudaBackend, &ScalarOperands) {
        (&mut self.backend, &self.operands[D::OPERAND_SLOT])
    }

    pub fn device(&self) -> usize {
        self.device
    }

    /// cuTENSOR contraction plan cache observation for this context's backend.
    ///
    /// Every region move and every GEMM submitted here builds or reuses one
    /// cuTENSOR plan per distinct operand signature, so `evictions` growing
    /// during a replay is the observable form of plan-cache thrash.
    pub fn plan_cache_stats(&self) -> Result<CudaPlanCacheStats, DenseError> {
        let stats = self
            .backend
            .cutensor_plan_cache_stats()
            .map_err(|err| cuda_error("cuda_plan_cache", err))?;
        Ok(CudaPlanCacheStats {
            entries: stats.entries,
            retained_bytes: stats.retained_bytes,
            hits: stats.hits,
            misses: stats.misses,
            evictions: stats.evictions,
            reserved_entries: self.plan_ledger.reserved(),
            reservation_shortfall: self.plan_ledger.shortfall(),
        })
    }

    /// The cuTENSOR contraction plan entry bound (Tenferro's default is 64).
    pub fn plan_cache_max_entries(&self) -> Result<usize, DenseError> {
        self.backend
            .cutensor_plan_cache_max_entries()
            .map(NonZeroUsize::get)
            .map_err(|err| cuda_error("cuda_plan_cache", err))
    }

    /// Reserves `delta` more plan entries for one consumer and returns how
    /// many were granted.
    ///
    /// A consumer that knows how many distinct operand signatures it will
    /// submit again — a compiled transform structure knows exactly — reserves
    /// them so a replay larger than the default bound does not evict the plan
    /// it needs on the next block. Reservations of independent consumers add:
    /// the cap becomes at least the bound found at construction plus every
    /// live reservation. The total is bounded by
    /// [`DEFAULT_PLAN_CACHE_BUDGET_BYTES`]; a request beyond it is truncated,
    /// and the truncation is reported by
    /// [`CudaPlanCacheStats::reservation_shortfall`]. The consumer owns what
    /// it was granted and returns it through [`Self::release_plan_entries`].
    pub fn reserve_plan_entries(&mut self, delta: usize) -> Result<usize, DenseError> {
        let granted = self.plan_ledger.grant(delta);
        self.raise_plan_cache_max_entries(self.plan_ledger.cap_with(granted))?;
        self.plan_ledger.commit(delta, granted);
        Ok(granted)
    }

    /// Returns `entries` a consumer reserved. The cap is not lowered: the
    /// plans stay cached, and the next reservation reuses the headroom rather
    /// than raising the cap again. Infallible so a `Drop` can call it.
    pub fn release_plan_entries(&mut self, entries: usize) {
        self.plan_ledger.release(entries);
    }

    /// Plan entries currently reserved over every consumer.
    pub fn reserved_plan_entries(&self) -> usize {
        self.plan_ledger.reserved()
    }

    /// Raises the cuTENSOR contraction plan entry bound to `entries`, never
    /// lowering it. Private: a consumer raising to its own absolute need would
    /// silently absorb another consumer's reservation.
    fn raise_plan_cache_max_entries(&self, entries: usize) -> Result<(), DenseError> {
        let Some(entries) = NonZeroUsize::new(entries) else {
            return Ok(());
        };
        let current = self
            .backend
            .cutensor_plan_cache_max_entries()
            .map_err(|err| cuda_error("cuda_plan_cache", err))?;
        if entries <= current {
            return Ok(());
        }
        self.backend
            .set_cutensor_plan_cache_max_entries(entries)
            .map_err(|err| cuda_error("cuda_plan_cache", err))
    }

    /// Runs the smallest real operation against each backend library this
    /// context can reach, so their one-time initialization is paid here
    /// rather than by the first user operation.
    ///
    /// Why: Tenferro (0.5.0, and 0.6.0 exposes no warm-up entry point either)
    /// loads cuTENSOR (`dlopen` + `cutensorCreate`) and the cuSOLVER/cuBLAS
    /// handles lazily, per `CudaBackend` instance, on the first submission
    /// that needs them.
    /// The measured cost is 185-657 ms
    /// (`benchmarks/history/cuda-baseline-2026-09-20.md`), independent of the
    /// operation, provider, dtype and size that happens to pay it. The cost is
    /// unavoidable; attributing it to context construction is the honest
    /// placement, and it is not a cache: nothing is memoized on the TeNeT side
    /// and no result changes.
    ///
    /// Because both libraries are touched here, a caller that only contracts
    /// still needs cuSOLVER and cuBLAS to be loadable, not just cuTENSOR: a
    /// missing library is the same typed [`DenseError::Backend`] as before,
    /// surfaced by whoever calls this (`RuntimeBuilder::build`) instead of by
    /// the first factorization. Tenferro resolves the three through
    /// `TENFERRO_CUTENSOR_PATH`, `TENFERRO_CUSOLVER_PATH` and
    /// `TENFERRO_CUBLAS_PATH`.
    ///
    /// What is *not* warmed: CubeCL compiles each kernel family with NVRTC on
    /// its first use (per process, per device), so the first elementwise,
    /// reduction or copy of each family still pays a JIT compile here. That is
    /// a CubeCL concern, not a TeNeT one; its PTX disk cache is configured
    /// through CubeCL's own `cubecl.toml` (`[compilation] cache`) and is off by
    /// default. The cuTENSOR plan cache is also per contraction shape, so a
    /// user contraction still builds its own plan. The region primitive's
    /// scalar operands are not warmed either: they are created on first use
    /// of their dtype, which keeps this warm-up's fixed cost, its counter
    /// deltas, and "every device buffer created here is dropped" true as
    /// written.
    ///
    /// Tenferro's runtime-owned
    /// blas1 cuBLAS handle (tenferro-gpu `cubecl/runtime.rs:688`) is also not
    /// warmed, because TeNeT reaches no blas1 entry point; extend the warm-up
    /// if that changes.
    ///
    /// Observation counters ([`cuda_transfer_stats`]) are per thread and are
    /// deliberately *not* reset here, so a caller's deltas are honest about the
    /// work this seam did. Each call adds exactly: `h2d_calls` 3,
    /// `h2d_bytes` 24, `device_allocs` 4 (three uploads plus the eigenvector
    /// factor), `gemm_calls` 1, `solver_calls` 1, `d2h_calls` 1, `d2h_bytes` 8,
    /// `copy_calls` 0.
    ///
    /// Single precision changes none of this. The warm-up submits `f64` work
    /// only, so a caller that never touches `f32`/[`Complex32`] pays exactly
    /// what it paid before: the libraries this loads (cuTENSOR, cuSOLVER,
    /// cuBLAS) are per backend instance, not per dtype, and the probe measured
    /// no notable per-dtype NVRTC stall
    /// (`benchmarks/history/cuda-single-precision-probe-2026-09-20.md`,
    /// finding 6). What a single-precision caller still pays on its own first
    /// use is the CubeCL kernel JIT for its dtype and the context's `f32`/
    /// `Complex32` scalar operands, both of which are lazy per dtype for the
    /// same reason the `f64` ones are.
    ///
    /// Every device buffer created here is dropped before returning.
    pub fn warm_up(&mut self) -> Result<(), DenseError> {
        // cuTENSOR: handle, library load and one plan, through the same seam
        // every contraction uses.
        let lhs = CudaDenseStorage::upload::<f64>(self, &[1.0_f64])?;
        let rhs = CudaDenseStorage::upload::<f64>(self, &[1.0_f64])?;
        let mut dst = CudaDenseStorage::upload::<f64>(self, &[0.0_f64])?;
        cuda_gemm_region_into::<f64>(
            self, &mut dst, 0, 1, &lhs, 0, 1, &rhs, 0, 1, 1, 1, 1, 1.0, 0.0,
        )?;
        // cuSOLVER + cuBLAS: a 1x1 region is trivially Hermitian, so this is a
        // real `eigh` with no host-side precondition to fake.
        let (values, _vectors) = cuda_eigh_region::<f64>(self, &lhs, 0, 1)?;
        cuda_download_spectra::<f64>(self, &[values])?;
        Ok(())
    }
}
