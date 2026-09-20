//! CUDA dense boundary: flat device buffers and offset-addressed matrix
//! GEMM, delegated to tenferro-gpu. This module is the only place in the
//! tenet workspace that touches tenferro GPU types; upper layers see opaque
//! storage handles and `DenseError`.

use std::num::NonZeroUsize;

use num_complex::Complex64;
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_linalg::{QrGauge, QrOptions, TensorReadLinalgExt};
use tenferro_tensor::backend::{BackendSession, BackendSessionHost};
use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorDot,
    TensorElementwise, TensorRead, TensorReduction, TensorScalar as TenferroScalar,
    TensorStructural, TensorView, TensorViewMut, TensorWrite, TypedTensor,
};

use super::{DenseBackend, DenseDType, DenseError, MatrixOp};
use crate::cuda_region::{validate_destination_layout, validate_region, CudaRegion};
use crate::tensor::dense_dtype_from_tenferro;

mod cuda_scalar_sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex64 {}
}

/// Payload dtypes a TeNeT CUDA buffer may own.
///
/// Sealed to `f64` and [`Complex64`]: structural (fusion-tree) coefficients
/// stay real, so only the *payload* varies, and `f32`/`Complex32` remain a
/// compile-time unsupported boundary rather than a runtime error. Conjugation
/// is never a payload property here — it is carried as a GEMM operand flag.
pub trait CudaScalar: TenferroScalar + cuda_scalar_sealed::Sealed {
    /// TeNeT-side dtype tag, used for [`DenseError::DTypeMismatch`].
    const DTYPE: DenseDType;
    /// Additive identity, for `beta = 0` overwriting GEMMs.
    const ZERO: Self;
    /// Multiplicative identity, for unscaled GEMMs.
    const ONE: Self;
    /// Whether this payload has an imaginary part. Used to skip the
    /// value-preserving `conj`/`abs` passes on the real dtype; it is a dtype
    /// invariant, not a size or workload heuristic.
    const IS_COMPLEX: bool;

    /// The backend's dtype-erased GEMM coefficient for this payload.
    fn contraction_scalar(self) -> ContractionScalar;

    /// The typed tensor behind a dtype-erased device buffer, if the dtypes agree.
    fn typed(tensor: &Tensor) -> Option<&TypedTensor<Self>>;

    /// Mutable counterpart of [`CudaScalar::typed`].
    fn typed_mut(tensor: &mut Tensor) -> Option<&mut TypedTensor<Self>>;
}

impl CudaScalar for f64 {
    const DTYPE: DenseDType = DenseDType::F64;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const IS_COMPLEX: bool = false;

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::F64(self)
    }

    fn typed(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::F64(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn typed_mut(tensor: &mut Tensor) -> Option<&mut TypedTensor<Self>> {
        match tensor {
            Tensor::F64(tensor) => Some(tensor),
            _ => None,
        }
    }
}

impl CudaScalar for Complex64 {
    const DTYPE: DenseDType = DenseDType::C64;
    const ZERO: Self = Complex64::new(0.0, 0.0);
    const ONE: Self = Complex64::new(1.0, 0.0);
    const IS_COMPLEX: bool = true;

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::C64(self)
    }

    fn typed(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::C64(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn typed_mut(tensor: &mut Tensor) -> Option<&mut TypedTensor<Self>> {
        match tensor {
            Tensor::C64(tensor) => Some(tensor),
            _ => None,
        }
    }
}

fn dtype_mismatch<D: CudaScalar>(op: &'static str, tensor: &Tensor) -> DenseError {
    DenseError::DTypeMismatch {
        op,
        expected: D::DTYPE,
        actual: dense_dtype_from_tenferro(tensor.dtype()),
    }
}

use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use std::sync::atomic::AtomicUsize;

#[cfg(test)]
static CUDA_FULL_DOWNLOAD_BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static CUDA_METADATA_DOWNLOAD_BYTES: AtomicUsize = AtomicUsize::new(0);

static H2D_CALLS: AtomicU64 = AtomicU64::new(0);
static H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static D2H_CALLS: AtomicU64 = AtomicU64::new(0);
static D2H_BYTES: AtomicU64 = AtomicU64::new(0);
static DEVICE_ALLOCS: AtomicU64 = AtomicU64::new(0);
static GEMM_CALLS: AtomicU64 = AtomicU64::new(0);
static SOLVER_CALLS: AtomicU64 = AtomicU64::new(0);
static COPY_CALLS: AtomicU64 = AtomicU64::new(0);

/// A snapshot of the backend's cuTENSOR contraction plan cache.
///
/// Mirrors Tenferro's own cache statistics; it exists so TeNeT callers never
/// name a tenferro type. Observability only.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaPlanCacheStats {
    pub entries: usize,
    pub retained_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// A snapshot of the process-wide CUDA boundary observation counters.
///
/// Observability only: nothing in this module reads these values back, so no
/// execution decision, dispatch, or capability depends on them. They exist so
/// a benchmark or a device test can attribute host/device traffic and backend
/// submissions to a measured phase without re-deriving them from an external
/// profiler.
///
/// Scope, as counted at *this* seam:
///
/// - `h2d_*` / `d2h_*`: every host buffer this module uploads or downloads,
///   including the one-element scalar uploads and reduction/spectrum
///   downloads. Bytes are the host-side payload bytes.
/// - `device_allocs`: device buffers this module creates or takes ownership
///   of, i.e. uploads plus tenferro-produced tensors wrapped as
///   [`CudaDenseStorage`] (factorization factors). Tenferro's own solver
///   workspaces and the intermediate tensors of `cuda_is_hermitian_region`
///   (`abs`, `div`, `sub`, the reductions) allocate on device but are not
///   visible as buffers here and are therefore not counted.
/// - `gemm_calls`: `dot_general` submissions, from
///   `cuda_gemm_region_strided_into` and from the region primitives
///   ([`cuda_region_axpby`], [`cuda_region_zero`]) alike.
/// - `solver_calls`: cuSOLVER region calls (SVD, QR, EIGH).
/// - `copy_calls`: `cuda_copy_region_into` calls that move data.
///
/// The counters are `Relaxed` and process-wide: a snapshot taken while another
/// thread submits work is a consistent-per-field sample, not a global instant.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaTransferStats {
    pub h2d_calls: u64,
    pub h2d_bytes: u64,
    pub d2h_calls: u64,
    pub d2h_bytes: u64,
    pub device_allocs: u64,
    pub gemm_calls: u64,
    pub solver_calls: u64,
    pub copy_calls: u64,
}

/// Reads the CUDA boundary observation counters. See [`CudaTransferStats`].
pub fn cuda_transfer_stats() -> CudaTransferStats {
    CudaTransferStats {
        h2d_calls: H2D_CALLS.load(Ordering::Relaxed),
        h2d_bytes: H2D_BYTES.load(Ordering::Relaxed),
        d2h_calls: D2H_CALLS.load(Ordering::Relaxed),
        d2h_bytes: D2H_BYTES.load(Ordering::Relaxed),
        device_allocs: DEVICE_ALLOCS.load(Ordering::Relaxed),
        gemm_calls: GEMM_CALLS.load(Ordering::Relaxed),
        solver_calls: SOLVER_CALLS.load(Ordering::Relaxed),
        copy_calls: COPY_CALLS.load(Ordering::Relaxed),
    }
}

/// Zeroes the CUDA boundary observation counters. See [`CudaTransferStats`].
pub fn reset_cuda_transfer_stats() {
    for counter in [
        &H2D_CALLS,
        &H2D_BYTES,
        &D2H_CALLS,
        &D2H_BYTES,
        &DEVICE_ALLOCS,
        &GEMM_CALLS,
        &SOLVER_CALLS,
        &COPY_CALLS,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

fn record_h2d(bytes: usize) {
    H2D_CALLS.fetch_add(1, Ordering::Relaxed);
    H2D_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    DEVICE_ALLOCS.fetch_add(1, Ordering::Relaxed);
}

fn record_d2h(bytes: usize) {
    D2H_CALLS.fetch_add(1, Ordering::Relaxed);
    D2H_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn cuda_error(op: &'static str, err: impl std::fmt::Display) -> DenseError {
    DenseError::Backend {
        backend: DenseBackend::Cuda,
        op,
        message: err.to_string(),
    }
}

fn with_cuda_linalg<R: Send>(
    backend: &mut CudaBackend,
    f: impl FnOnce(&mut dyn BackendSession) -> tenferro_tensor::Result<R> + Send,
) -> tenferro_tensor::Result<R> {
    backend.with_backend_session(f)
}

fn cuda_operand_view(op: MatrixOp, rows: usize, cols: usize) -> ([usize; 2], bool) {
    match op {
        MatrixOp::Identity => ([1, rows], false),
        MatrixOp::Transpose => ([cols, 1], false),
        MatrixOp::Adjoint => ([cols, 1], true),
    }
}

/// Validates that every operand resides on the context's CUDA device, so a
/// storage handle created from one context can't be silently used against
/// another context's runtime — which would otherwise fail late inside
/// tenferro/CUDA with a confusing error, or run against the wrong runtime.
/// `operands` are `(name, device)` pairs; `ctx_device` is `ctx.device`.
fn ensure_cuda_device(
    ctx_device: usize,
    op: &'static str,
    operands: &[(&str, usize)],
) -> Result<(), DenseError> {
    for (name, device) in operands {
        if *device != ctx_device {
            return Err(cuda_error(
                op,
                format!(
                    "operand `{name}` is on CUDA device {device} but the context is on device {ctx_device}"
                ),
            ));
        }
    }
    Ok(())
}

/// The device operands a region call needs but never varies: the `1` used as
/// the 1x1 coefficient of an unscaled move, and the zero template read as the
/// packed source of a region fill.
///
/// One pair per payload dtype, created on first use of that dtype rather than
/// at [`CudaDenseContext::warm_up`], so the warm-up's documented fixed cost
/// and its counter deltas are unchanged. The zero template grows monotonically
/// to the longest region filled so far; it is bounded by the largest single
/// destination region a caller has asked to zero, not by the number of calls.
#[derive(Default)]
struct ScalarOperands {
    ones: Option<CudaDenseStorage>,
    zeros: Option<CudaDenseStorage>,
}

/// Owns the tenferro CUDA backend for one device ordinal.
pub struct CudaDenseContext {
    backend: CudaBackend,
    device: usize,
    identity: u64,
    real_operands: ScalarOperands,
    complex_operands: ScalarOperands,
}

/// Process-wide context counter. A monotonic ticket rather than the context's
/// address, so a cache keyed by context identity can never mistake a freshly
/// allocated context for a dropped one that happened to reuse its address.
static NEXT_CONTEXT_IDENTITY: AtomicU64 = AtomicU64::new(1);

impl CudaDenseContext {
    pub fn new(device: usize) -> Result<Self, DenseError> {
        let ordinal = u32::try_from(device)
            .map_err(|_| cuda_error("cuda_context", "device ordinal exceeds u32"))?;
        let backend = CudaBackend::new(CudaDeviceId::from_ordinal(ordinal))
            .map_err(|err| cuda_error("cuda_context", err))?;
        Ok(Self {
            backend,
            device,
            identity: NEXT_CONTEXT_IDENTITY.fetch_add(1, Ordering::Relaxed),
            real_operands: ScalarOperands::default(),
            complex_operands: ScalarOperands::default(),
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
        if D::IS_COMPLEX {
            &self.complex_operands
        } else {
            &self.real_operands
        }
    }

    fn operands_mut<D: CudaScalar>(&mut self) -> &mut ScalarOperands {
        if D::IS_COMPLEX {
            &mut self.complex_operands
        } else {
            &mut self.real_operands
        }
    }

    /// Uploads this dtype's `1` operand unless it is already resident.
    fn ensure_ones<D: CudaScalar>(&mut self) -> Result<(), DenseError> {
        if self.operands::<D>().ones.is_none() {
            let ones = CudaDenseStorage::upload_owned(self, vec![D::ONE])?;
            self.operands_mut::<D>().ones = Some(ones);
        }
        Ok(())
    }

    /// Uploads this dtype's zero template unless a resident one already holds
    /// at least `len` elements. A prefix of a compact zero buffer is itself a
    /// compact zero buffer, so a longer template serves every shorter fill.
    fn ensure_zeros<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
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
            self.operands_mut::<D>().zeros = Some(zeros);
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
    /// [`Self::scalar_operand_bytes`]. The device transform executor (G2a-2)
    /// is expected to charge them to the device workspace budget, exactly as
    /// the G3c workspace charges its own `CudaZeroTemplate`; those two zero
    /// sources should become one authority once a caller owns both (or vanish
    /// together with tenferro-rs#1834's native device fill). This leaf
    /// deliberately does not consolidate them.
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
        fn bytes(operands: &ScalarOperands, element: usize) -> usize {
            operands.ones.as_ref().map_or(0, CudaDenseStorage::len) * element
                + operands.zeros.as_ref().map_or(0, CudaDenseStorage::len) * element
        }
        bytes(&self.real_operands, std::mem::size_of::<f64>())
            + bytes(&self.complex_operands, std::mem::size_of::<Complex64>())
    }

    /// Frees every lazily created scalar operand. The next region call that
    /// needs one re-creates it, so this is a memory decision, never a
    /// correctness one.
    pub fn release_scalar_operands(&mut self) {
        self.real_operands = ScalarOperands::default();
        self.complex_operands = ScalarOperands::default();
    }

    /// Hands out the backend and this dtype's scalar operands at once.
    ///
    /// A whole-`self` accessor cannot express this: submitting against a
    /// context-owned operand needs `&mut` on the backend and `&` on the
    /// operand simultaneously, which is only sound because they are disjoint
    /// fields.
    fn split_operands<D: CudaScalar>(&mut self) -> (&mut CudaBackend, &ScalarOperands) {
        if D::IS_COMPLEX {
            (&mut self.backend, &self.complex_operands)
        } else {
            (&mut self.backend, &self.real_operands)
        }
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
        })
    }

    /// The cuTENSOR contraction plan entry bound (Tenferro's default is 64).
    pub fn plan_cache_max_entries(&self) -> Result<usize, DenseError> {
        self.backend
            .cutensor_plan_cache_max_entries()
            .map(NonZeroUsize::get)
            .map_err(|err| cuda_error("cuda_plan_cache", err))
    }

    /// Raises the cuTENSOR contraction plan entry bound to `entries`.
    ///
    /// Monotonic by construction: a request below the current bound is
    /// ignored rather than shrinking a cache another caller sized. A caller
    /// that knows how many distinct operand signatures it is about to submit
    /// — a compiled transform structure knows exactly — uses this so a replay
    /// larger than the default bound does not evict the plan it will need
    /// again on the next block.
    pub fn raise_plan_cache_max_entries(&self, entries: usize) -> Result<(), DenseError> {
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
    /// Why: tenferro 0.5.0 loads cuTENSOR (`dlopen` + `cutensorCreate`) and
    /// the cuSOLVER/cuBLAS handles lazily, per `CudaBackend` instance, on the
    /// first submission that needs them, and exposes no warm-up entry point.
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
    /// Observation counters ([`cuda_transfer_stats`]) are process-wide and are
    /// deliberately *not* reset here, so a caller's deltas are honest about the
    /// work this seam did. Each call adds exactly: `h2d_calls` 3,
    /// `h2d_bytes` 24, `device_allocs` 4 (three uploads plus the eigenvector
    /// factor), `gemm_calls` 1, `solver_calls` 1, `d2h_calls` 1, `d2h_bytes` 8,
    /// `copy_calls` 0.
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
        let (_values, _vectors) = cuda_eigh_region::<f64>(self, &lhs, 0, 1)?;
        Ok(())
    }
}

/// Flat [`CudaScalar`] buffer resident on one CUDA device.
///
/// The handle itself is dtype-erased; every typed access names the payload
/// dtype and reports a mismatch as [`DenseError::DTypeMismatch`].
pub struct CudaDenseStorage {
    tensor: Tensor,
    len: usize,
    device: usize,
}

impl CudaDenseStorage {
    /// Uploads borrowed host data as a flat device buffer.
    ///
    /// Tenferro uploads an owned host tensor (`upload_tensor`, tenferro-gpu
    /// `cubecl/memory.rs:31`), so borrowed data costs exactly one host copy
    /// here: the payload is duplicated into the host tensor that is then
    /// staged to the device. That copy is required by the backend contract,
    /// not by TeNeT. A caller that already owns its buffer — a zero buffer, a
    /// selector, a coefficient or diagonal vector — uses
    /// [`Self::upload_owned`] instead and pays none.
    pub fn upload<D: CudaScalar>(ctx: &CudaDenseContext, data: &[D]) -> Result<Self, DenseError> {
        Self::upload_owned(ctx, data.to_vec())
    }

    /// Uploads owned host data as a flat device buffer, moving `data` into the
    /// host tensor Tenferro uploads instead of copying it.
    ///
    /// Counting, device traffic and the resulting buffer are identical to
    /// [`Self::upload`]; only the redundant host copy is gone.
    pub fn upload_owned<D: CudaScalar>(
        ctx: &CudaDenseContext,
        data: Vec<D>,
    ) -> Result<Self, DenseError> {
        let len = data.len();
        let bytes = std::mem::size_of_val(data.as_slice());
        let host = Tensor::from_vec_col_major(vec![len], data)
            .map_err(|err| cuda_error("cuda_upload", err))?;
        let tensor = upload_tensor(ctx.backend.runtime(), &host)
            .map_err(|err| cuda_error("cuda_upload", err))?;
        record_h2d(bytes);
        Ok(Self {
            tensor,
            len,
            device: ctx.device,
        })
    }

    /// Downloads the flat device buffer back to host data.
    ///
    /// The returned vector is the one Tenferro's download produced: the host
    /// tensor is consumed (`TypedTensor::into_host_vec`, tenferro-tensor
    /// `types.rs:7248`) rather than copied out of again.
    pub fn download<D: CudaScalar>(&self, ctx: &CudaDenseContext) -> Result<Vec<D>, DenseError> {
        ensure_cuda_device(ctx.device, "cuda_download", &[("source", self.device)])?;
        let host = download_tensor(ctx.backend.runtime(), &self.tensor)
            .map_err(|err| cuda_error("cuda_download", err))?;
        if host.dtype() != D::dtype() {
            return Err(dtype_mismatch::<D>("cuda_download", &host));
        }
        let typed = D::into_typed(host).map_err(|err| cuda_error("cuda_download", err))?;
        let data = typed
            .into_host_vec()
            .map_err(|err| cuda_error("cuda_download", err))?;
        let bytes = std::mem::size_of_val(data.as_slice());
        #[cfg(test)]
        CUDA_FULL_DOWNLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
        record_d2h(bytes);
        Ok(data)
    }

    /// The payload dtype this device buffer owns.
    pub fn dtype(&self) -> DenseDType {
        dense_dtype_from_tenferro(self.tensor.dtype())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> usize {
        self.device
    }

    /// Wraps a device tensor produced by a tenferro op (e.g. a cuSOLVER
    /// factor) as flat storage.
    fn from_tensor(tensor: Tensor, device: usize) -> Self {
        DEVICE_ALLOCS.fetch_add(1, Ordering::Relaxed);
        let len = tensor.shape().iter().product();
        Self {
            tensor,
            len,
            device,
        }
    }

    /// Column-major matrix view over a buffer region with an explicit
    /// leading dimension (`ld >= rows`, `ld == rows` for a packed region).
    fn region_view<D: CudaScalar>(
        &self,
        rows: usize,
        cols: usize,
        ld: usize,
        offset: usize,
    ) -> Result<TensorView<'_>, DenseError> {
        self.region_view_strided::<D>([rows, cols], [1, ld], offset)
    }

    fn region_view_strided<D: CudaScalar>(
        &self,
        shape: [usize; 2],
        strides: [usize; 2],
        offset: usize,
    ) -> Result<TensorView<'_>, DenseError> {
        let Some(tensor) = D::typed(&self.tensor) else {
            return Err(dtype_mismatch::<D>("cuda_region", &self.tensor));
        };
        let offset = isize::try_from(offset)
            .map_err(|_| cuda_error("cuda_region", "offset does not fit in isize"))?;
        let strides = strides
            .map(isize::try_from)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| cuda_error("cuda_region", "stride does not fit in isize"))?;
        tensor
            .backend_region_view(shape.to_vec(), strides, offset)
            .map(D::tensor_view)
            .map_err(|err| cuda_error("cuda_region", err))
    }

    /// Rank-N counterpart of [`Self::region_view_strided`]. The caller owns
    /// the bounds and dtype proof only in the sense that both are re-checked
    /// here (dtype) and by [`validate_region`] (bounds) before any submission.
    fn region_view_nd<D: CudaScalar>(
        &self,
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Result<TensorView<'_>, DenseError> {
        let Some(tensor) = D::typed(&self.tensor) else {
            return Err(dtype_mismatch::<D>("cuda_region", &self.tensor));
        };
        tensor
            .backend_region_view(dims.to_vec(), strides.to_vec(), offset)
            .map(D::tensor_view)
            .map_err(|err| cuda_error("cuda_region", err))
    }

    fn region_view_nd_mut<D: CudaScalar>(
        &mut self,
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Result<TensorViewMut<'_>, DenseError> {
        let actual = dense_dtype_from_tenferro(self.tensor.dtype());
        let Some(tensor) = D::typed_mut(&mut self.tensor) else {
            return Err(DenseError::DTypeMismatch {
                op: "cuda_region",
                expected: D::DTYPE,
                actual,
            });
        };
        tensor
            .backend_region_view_mut(dims.to_vec(), strides.to_vec(), offset)
            .map(D::tensor_view_mut)
            .map_err(|err| cuda_error("cuda_region", err))
    }

    fn region_view_mut<D: CudaScalar>(
        &mut self,
        rows: usize,
        cols: usize,
        ld: usize,
        offset: usize,
    ) -> Result<TensorViewMut<'_>, DenseError> {
        let actual = dense_dtype_from_tenferro(self.tensor.dtype());
        let Some(tensor) = D::typed_mut(&mut self.tensor) else {
            return Err(DenseError::DTypeMismatch {
                op: "cuda_region",
                expected: D::DTYPE,
                actual,
            });
        };
        let offset = isize::try_from(offset)
            .map_err(|_| cuda_error("cuda_region", "offset does not fit in isize"))?;
        let ld_isize = isize::try_from(ld)
            .map_err(|_| cuda_error("cuda_region", "leading dimension does not fit in isize"))?;
        tensor
            .backend_region_view_mut(vec![rows, cols], vec![1, ld_isize], offset)
            .map(D::tensor_view_mut)
            .map_err(|err| cuda_error("cuda_region", err))
    }
}

/// Column-major matrix GEMM over device buffer regions:
/// `dst[dst_offset..][rows x cols] = lhs_part * rhs_part` (overwrite).
#[allow(clippy::too_many_arguments)]
pub fn cuda_matmul_region_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    lhs: &CudaDenseStorage,
    lhs_offset: usize,
    rhs: &CudaDenseStorage,
    rhs_offset: usize,
    rows: usize,
    contracted: usize,
    cols: usize,
) -> Result<(), DenseError> {
    cuda_gemm_region_into::<D>(
        ctx,
        dst,
        dst_offset,
        rows,
        lhs,
        lhs_offset,
        rows,
        rhs,
        rhs_offset,
        contracted,
        rows,
        contracted,
        cols,
        D::ONE,
        D::ZERO,
    )
}

/// General column-major GEMM over device buffer regions with explicit
/// per-operand offsets and leading dimensions, plus scaling:
/// `dst_region[m x n] = alpha * lhs_region[m x k] * rhs_region[k x n]
///  + beta * dst_region`.
///
/// This is the single device seam the user layer builds everything
/// non-cuSOLVER on: sector inner products (`m = n = 1`), axpby via a `[1,1]`
/// ones operand (`k = n = 1`), and factor assembly through small selector
/// matrices (identity / prefix / sign / permutation).
#[allow(clippy::too_many_arguments)]
pub fn cuda_gemm_region_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    dst_ld: usize,
    lhs: &CudaDenseStorage,
    lhs_offset: usize,
    lhs_ld: usize,
    rhs: &CudaDenseStorage,
    rhs_offset: usize,
    rhs_ld: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha: D,
    beta: D,
) -> Result<(), DenseError> {
    cuda_gemm_region_strided_into::<D>(
        ctx,
        dst,
        dst_offset,
        dst_ld,
        lhs,
        lhs_offset,
        [1, lhs_ld],
        false,
        rhs,
        rhs_offset,
        [1, rhs_ld],
        false,
        m,
        k,
        n,
        alpha,
        beta,
    )
}

/// GEMM over logical matrix views of packed parent regions. `Adjoint` changes
/// the two strides and carries conjugation metadata without creating a
/// transposed or conjugated payload: for a complex payload the conjugation is
/// a backend operand flag, and for `f64` it is a no-op.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn cuda_gemm_region_with_ops_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    lhs: &CudaDenseStorage,
    lhs_offset: usize,
    rhs: &CudaDenseStorage,
    rhs_offset: usize,
    m: usize,
    k: usize,
    n: usize,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
    alpha: D,
    beta: D,
) -> Result<(), DenseError> {
    let (lhs_strides, lhs_conj) = cuda_operand_view(lhs_op, m, k);
    let (rhs_strides, rhs_conj) = cuda_operand_view(rhs_op, k, n);
    cuda_gemm_region_strided_into::<D>(
        ctx,
        dst,
        dst_offset,
        m,
        lhs,
        lhs_offset,
        lhs_strides,
        lhs_conj,
        rhs,
        rhs_offset,
        rhs_strides,
        rhs_conj,
        m,
        k,
        n,
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
fn cuda_gemm_region_strided_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    dst_ld: usize,
    lhs: &CudaDenseStorage,
    lhs_offset: usize,
    lhs_strides: [usize; 2],
    lhs_conj: bool,
    rhs: &CudaDenseStorage,
    rhs_offset: usize,
    rhs_strides: [usize; 2],
    rhs_conj: bool,
    m: usize,
    k: usize,
    n: usize,
    alpha: D,
    beta: D,
) -> Result<(), DenseError> {
    ensure_cuda_device(
        ctx.device,
        "cuda_matmul",
        &[
            ("dst", dst.device),
            ("lhs", lhs.device),
            ("rhs", rhs.device),
        ],
    )?;
    let lhs_view = lhs.region_view_strided::<D>([m, k], lhs_strides, lhs_offset)?;
    let rhs_view = rhs.region_view_strided::<D>([k, n], rhs_strides, rhs_offset)?;
    let dst_view = dst.region_view_mut::<D>(m, n, dst_ld, dst_offset)?;
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj,
        rhs_conj,
        alpha: alpha.contraction_scalar(),
        beta: beta.contraction_scalar(),
    };
    GEMM_CALLS.fetch_add(1, Ordering::Relaxed);
    ctx.backend
        .dot_general_read_into_accum(
            TensorRead::from_view(lhs_view),
            TensorRead::from_view(rhs_view),
            &config,
            accumulation,
            TensorWrite::from_view(dst_view),
        )
        .map_err(|err| cuda_error("cuda_matmul", err))
}

/// How a region call treats the destination it writes.
///
/// Only these two exist: Tenferro 0.5.0 has no in-place strided scale, so a
/// general `beta` is a capability boundary rather than a parameter. The host
/// replay never needs one either — an overwriting transform zeroes its
/// inactive layouts and assigns the active ones, and an accumulating caller
/// uses `beta = 1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaRegionBeta {
    /// `dst_region = coefficient * src_region` (`beta = 0`).
    Overwrite,
    /// `dst_region += coefficient * src_region` (`beta = 1`).
    Accumulate,
}

impl CudaRegionBeta {
    fn scalar<D: CudaScalar>(self) -> ContractionScalar {
        match self {
            Self::Overwrite => D::ZERO.contraction_scalar(),
            Self::Accumulate => D::ONE.contraction_scalar(),
        }
    }
}

/// The payload dtype of an operand, checked before any device work rather
/// than at view construction, so a mismatched call cannot have uploaded a
/// lazily created scalar operand first.
fn ensure_payload_dtype<D: CudaScalar>(
    op: &'static str,
    storage: &CudaDenseStorage,
) -> Result<(), DenseError> {
    let actual = storage.dtype();
    if actual != D::DTYPE {
        return Err(DenseError::DTypeMismatch {
            op,
            expected: D::DTYPE,
            actual,
        });
    }
    Ok(())
}

/// Submits one validated region move. Takes the backend rather than the whole
/// context so a caller can pass a coefficient the context itself owns.
#[allow(clippy::too_many_arguments)]
fn submit_region_axpby<D: CudaScalar>(
    backend: &mut CudaBackend,
    op: &'static str,
    src: &CudaDenseStorage,
    src_region: &CudaRegion,
    conj: bool,
    coeff: &CudaDenseStorage,
    coeff_offset: usize,
    beta: CudaRegionBeta,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
) -> Result<(), DenseError> {
    let rank = src_region.dims().len();
    let (view_dims, src_strides) = src_region.contraction_view_metadata()?;
    let (_, dst_strides) = dst_region.contraction_view_metadata()?;
    let coeff_offset = isize::try_from(coeff_offset).map_err(|_| DenseError::OffsetOverflow {
        value: coeff_offset,
    })?;

    let lhs = src.region_view_nd::<D>(&view_dims, &src_strides, src_region.offset_isize()?)?;
    let rhs = coeff.region_view_nd::<D>(&[1, 1], &[1, 1], coeff_offset)?;
    let out = dst.region_view_nd_mut::<D>(&view_dims, &dst_strides, dst_region.offset_isize()?)?;

    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![rank],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    };
    // The coefficient is the 1x1 *operand*, never the descriptor alpha: a
    // descriptor alpha of 0 lets CUDA skip the source read and erase NaN/Inf,
    // where the host multiplies and propagates it (typed.rs `cuda_axpby_owned`).
    let accumulation = DotGeneralAccumulation {
        lhs_conj: conj,
        rhs_conj: false,
        alpha: D::ONE.contraction_scalar(),
        beta: beta.scalar::<D>(),
    };
    GEMM_CALLS.fetch_add(1, Ordering::Relaxed);
    backend
        .dot_general_read_into_accum(
            TensorRead::from_view(lhs),
            TensorRead::from_view(rhs),
            &config,
            accumulation,
            TensorWrite::from_view(out),
        )
        .map_err(|err| cuda_error(op, err))
}

/// `dst_region = c * [conj] src_region + beta * dst_region`, where `c` is a
/// 1x1 **data** operand read from `coeff`.
///
/// This is the single strided data-movement primitive the device structural
/// operations are built on: one strided, offset, rank-N source region of one
/// buffer moved into a strided, offset, rank-N destination region of another,
/// with the axis permutation carried by the destination strides. It submits
/// one `dot_general` against the 1x1 coefficient and allocates no device
/// buffer of its own.
///
/// Transfer contract: a call with `coeff = Some(..)` moves nothing across the
/// host boundary, ever. A call with `coeff = None` uploads this context's
/// one-element `1` the first time that dtype is used, and nothing afterwards
/// (see [`CudaDenseContext::scalar_operand_bytes`]).
///
/// `coeff` is `Some((buffer, offset))` to read the coefficient from a
/// caller-owned device vector — block `b` of a structure's uploaded
/// coefficients is the 1x1 view at offset `b` — or `None` to use the
/// context's own `1`, for a move that only relayouts. It is never the
/// contraction's descriptor alpha, which stays 1: a descriptor alpha lets
/// CUDA skip the source read when it is 0 and so erases NaN/Inf that the host
/// propagates.
///
/// One numerical deviation from the host is disclosed and pinned by the device
/// tests: the host copies bit-exactly when the coefficient is 1, while this
/// path always multiplies, and an infinite complex payload is observed to come
/// back as `NaN` in both components: `inf * 0` in the complex product already
/// yields a `NaN` component, which the remaining multiply spreads across both.
/// `f64` infinities and every finite payload are unaffected. See
/// `benchmarks/history/cuda-region-axpby-2026-09-20.md`.
///
/// Validation order, all of it before any device work:
///
/// 1. every operand is on the context's device;
/// 2. every operand has payload dtype `D` (`DTypeMismatch`);
/// 3. source and destination `dims` are equal (`ShapeMismatch`);
/// 4. the coefficient offset is inside `coeff` (`OutOfBounds`);
/// 5. the element count does not overflow (`ElementCountOverflow`) — a
///    zero-extent region returns `Ok` here, with no submission;
/// 6. the destination layout is injective (`Unsupported`);
/// 7. both regions stay inside their buffers and fit `isize`
///    (`OutOfBounds` / `OffsetOverflow` / `StrideOverflow`).
///
/// Two further constraints are type boundaries rather than checks: strides
/// cannot be negative ([`CudaRegion`]), and `src`, `coeff` and `dst` cannot be
/// the same buffer, because [`CudaDenseStorage`] is neither `Clone` nor
/// refcounted, so a shared and an exclusive borrow of one buffer cannot
/// coexist. An in-place strided transform is therefore inexpressible here.
///
/// No rank limit is imposed: the probe accepted every mode count up to 72,
/// far above any rank a fusion tree reaches.
///
/// Disclosed cost: Tenferro's stream slots are per thread, so a context-owned
/// operand (`coeff = None`, or any [`cuda_region_zero`]) that is used from a
/// thread other than the one that created it can force a device-wide
/// synchronize per call. Today every such call happens under the device lease,
/// which serializes them; a future multi-threaded device executor should keep
/// operand creation and use on the same stream slot or pass its own
/// coefficient buffer.
#[allow(clippy::too_many_arguments)]
pub fn cuda_region_axpby<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    src_region: &CudaRegion,
    conj: bool,
    coeff: Option<(&CudaDenseStorage, usize)>,
    beta: CudaRegionBeta,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_region_axpby";
    match coeff {
        Some((coeff, _)) => ensure_cuda_device(
            ctx.device,
            OP,
            &[
                ("src", src.device),
                ("coefficient", coeff.device),
                ("dst", dst.device),
            ],
        )?,
        None => ensure_cuda_device(ctx.device, OP, &[("src", src.device), ("dst", dst.device)])?,
    }
    ensure_payload_dtype::<D>(OP, src)?;
    ensure_payload_dtype::<D>(OP, dst)?;
    if let Some((coeff, _)) = coeff {
        ensure_payload_dtype::<D>(OP, coeff)?;
    }
    if src_region.dims() != dst_region.dims() {
        return Err(DenseError::ShapeMismatch {
            op: OP,
            expected: dst_region.dims().to_vec(),
            actual: src_region.dims().to_vec(),
        });
    }
    if let Some((coeff, offset)) = coeff {
        if offset >= coeff.len {
            return Err(DenseError::OutOfBounds);
        }
    }
    if src_region.is_empty() {
        return Ok(());
    }
    src_region.element_count()?;
    validate_destination_layout(OP, dst_region)?;
    validate_region(src_region, src.len)?;
    validate_region(dst_region, dst.len)?;

    match coeff {
        Some((coeff, offset)) => submit_region_axpby::<D>(
            &mut ctx.backend,
            OP,
            src,
            src_region,
            conj,
            coeff,
            offset,
            beta,
            dst,
            dst_region,
        ),
        None => {
            ctx.ensure_ones::<D>()?;
            let (backend, operands) = ctx.split_operands::<D>();
            let Some(ones) = operands.ones.as_ref() else {
                return Err(cuda_error(OP, "context scalar operand is missing"));
            };
            submit_region_axpby::<D>(
                backend, OP, src, src_region, conj, ones, 0, beta, dst, dst_region,
            )
        }
    }
}

/// Writes zeros over `dst_region`, reading the context's zero template as a
/// packed source of the same extents.
///
/// This is the overwrite-mode destination rule expressed with the same
/// primitive: an inactive destination layout is zeroed rather than left
/// alone, so the result is independent of what the caller's buffer held —
/// including a NaN. It is a region move, not a fill kernel, so it inherits
/// exactly the validation of [`cuda_region_axpby`].
///
/// Transfer contract: it uploads the context's `1` and its zero template on
/// first use of a dtype, and re-uploads the template whenever a fill is longer
/// than the resident one. Size the template once with
/// [`CudaDenseContext::reserve_zero_template`] to make every later fill of a
/// replay transfer-free; otherwise a sequence of ascending fills pays one
/// upload each.
pub fn cuda_region_zero<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_region_zero";
    ensure_cuda_device(ctx.device, OP, &[("dst", dst.device)])?;
    ensure_payload_dtype::<D>(OP, dst)?;
    if dst_region.is_empty() {
        return Ok(());
    }
    let count = dst_region.element_count()?;
    validate_destination_layout(OP, dst_region)?;
    validate_region(dst_region, dst.len)?;
    let src_region = CudaRegion::packed(dst_region.dims(), 0)?;

    ctx.ensure_ones::<D>()?;
    ctx.ensure_zeros::<D>(count)?;
    let (backend, operands) = ctx.split_operands::<D>();
    let (Some(ones), Some(zeros)) = (operands.ones.as_ref(), operands.zeros.as_ref()) else {
        return Err(cuda_error(OP, "context scalar operands are missing"));
    };
    submit_region_axpby::<D>(
        backend,
        OP,
        zeros,
        &src_region,
        false,
        ones,
        0,
        CudaRegionBeta::Overwrite,
        dst,
        dst_region,
    )
}

/// Downloads a small real (f64) device tensor as host values. Only used for
/// spectra / diagonals — the sole tensor-shaped data that is allowed to
/// cross the device boundary implicitly (truncation decisions are host
/// scalar logic).
fn download_values(ctx: &CudaDenseContext, tensor: &Tensor) -> Result<Vec<f64>, DenseError> {
    let host = download_tensor(ctx.backend.runtime(), tensor)
        .map_err(|err| cuda_error("cuda_download", err))?;
    match host {
        Tensor::F64(tensor) => tensor
            .into_host_vec()
            .inspect(|values| {
                let bytes = std::mem::size_of_val(values.as_slice());
                #[cfg(test)]
                CUDA_METADATA_DOWNLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
                record_d2h(bytes);
            })
            .map_err(|err| cuda_error("cuda_download", err)),
        other => Err(cuda_error(
            "cuda_download",
            format!("expected f64 values, got {:?}", other.dtype()),
        )),
    }
}

fn download_scalar(
    ctx: &CudaDenseContext,
    tensor: &Tensor,
    op: &'static str,
) -> Result<f64, DenseError> {
    let values = download_values(ctx, tensor)?;
    if values.len() != 1 {
        return Err(cuda_error(
            op,
            format!(
                "device reduction returned {} values; expected 1",
                values.len()
            ),
        ));
    }
    Ok(values[0])
}

fn upload_scalar(ctx: &CudaDenseContext, value: f64) -> Result<Tensor, DenseError> {
    let host = Tensor::from_vec_col_major(vec![], vec![value])
        .map_err(|err| cuda_error("cuda_hermitian", err))?;
    let tensor = upload_tensor(ctx.backend.runtime(), &host)
        .map_err(|err| cuda_error("cuda_hermitian", err))?;
    record_h2d(std::mem::size_of::<f64>());
    Ok(tensor)
}

fn scaled_hermitian_residual_accepts(input_ss: f64, residual_scale: f64, residual_ss: f64) -> bool {
    input_ss.is_finite()
        && input_ss >= 0.0
        && residual_scale.is_finite()
        && residual_scale >= 0.0
        && residual_ss.is_finite()
        && residual_ss >= 0.0
        && 0.5 * residual_scale * residual_ss.sqrt() <= 64.0 * f64::EPSILON * input_ss.sqrt()
}

/// Real magnitudes of a device tensor, so the real-only sum-of-squares
/// reduction sees `|z|^2` for a complex payload. `abs` leaves an `f64` tensor
/// numerically unchanged under that reduction, so the extra pass is skipped
/// there on the dtype invariant.
fn magnitudes_for_sum_squares<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    op: &'static str,
    tensor: &Tensor,
) -> Result<Option<Tensor>, DenseError> {
    if !D::IS_COMPLEX {
        return Ok(None);
    }
    ctx.backend
        .abs(tensor)
        .map(Some)
        .map_err(|err| cuda_error(op, err))
}

/// Tests one packed CUDA matrix region with the host EIGH rule
/// `||(A - A^H)/2||_F <= 64 eps ||A||_F`. The residual uses the *conjugate*
/// transpose, so a complex-symmetric non-Hermitian block is rejected.
///
/// The normal and conjugate-transposed views are materialized and reduced on
/// device. Only scalar norm metadata is downloaded; the receiver region is
/// never copied to the host. Operation-local workspaces are dropped on return.
#[doc(hidden)]
pub fn cuda_is_hermitian_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<bool, DenseError> {
    const OP: &str = "cuda_hermitian";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    if n == 0 {
        return Ok(true);
    }

    let normal_view = src.region_view_strided::<D>([n, n], [1, n], offset)?;
    let normal = ctx
        .backend
        .to_contiguous_read(TensorRead::from_view(normal_view))
        .map_err(|err| cuda_error(OP, err))?;
    let input_abs = ctx
        .backend
        .abs(&normal)
        .map_err(|err| cuda_error(OP, err))?;
    let input_max = ctx
        .backend
        .reduce_max(&input_abs, &[0, 1])
        .map_err(|err| cuda_error(OP, err))?;
    let input_scale = download_scalar(ctx, &input_max, OP)?;
    // Pinned Tenferro's CUDA reduce_max propagates NaN. Keep this check before
    // the zero fast path so an otherwise-zero matrix containing NaN is rejected.
    if !input_scale.is_finite() {
        return Ok(false);
    }
    if input_scale == 0.0 {
        return Ok(true);
    }

    let transpose_view = src.region_view_strided::<D>([n, n], [n, 1], offset)?;
    let transpose = ctx
        .backend
        .to_contiguous_read(TensorRead::from_view(transpose_view))
        .map_err(|err| cuda_error(OP, err))?;
    // Conjugation is a backend op on the transposed copy, never a TeNeT loop.
    let transpose = if D::IS_COMPLEX {
        ctx.backend
            .conj(&transpose)
            .map_err(|err| cuda_error(OP, err))?
    } else {
        transpose
    };
    let scale = upload_scalar(ctx, input_scale)?;
    let normal_scaled = ctx
        .backend
        .div(&normal, &scale)
        .map_err(|err| cuda_error(OP, err))?;
    let transpose_scaled = ctx
        .backend
        .div(&transpose, &scale)
        .map_err(|err| cuda_error(OP, err))?;
    let input_magnitudes = magnitudes_for_sum_squares::<D>(ctx, OP, &normal_scaled)?;
    let input_ss = ctx
        .backend
        .reduce_sum_squares_read(
            TensorRead::from_tensor(input_magnitudes.as_ref().unwrap_or(&normal_scaled)),
            &[0, 1],
        )
        .map_err(|err| cuda_error(OP, err))?;
    let input_ss = download_scalar(ctx, &input_ss, OP)?;

    let residual = ctx
        .backend
        .sub(&normal_scaled, &transpose_scaled)
        .map_err(|err| cuda_error(OP, err))?;
    let residual_abs = ctx
        .backend
        .abs(&residual)
        .map_err(|err| cuda_error(OP, err))?;
    let residual_max = ctx
        .backend
        .reduce_max(&residual_abs, &[0, 1])
        .map_err(|err| cuda_error(OP, err))?;
    let residual_scale = download_scalar(ctx, &residual_max, OP)?;
    if !residual_scale.is_finite() {
        return Ok(false);
    }
    if residual_scale == 0.0 {
        return Ok(input_ss.is_finite() && input_ss >= 0.0);
    }

    let residual_scale_tensor = upload_scalar(ctx, residual_scale)?;
    let residual_normalized = ctx
        .backend
        .div(&residual, &residual_scale_tensor)
        .map_err(|err| cuda_error(OP, err))?;
    let residual_magnitudes = magnitudes_for_sum_squares::<D>(ctx, OP, &residual_normalized)?;
    let residual_ss = ctx
        .backend
        .reduce_sum_squares_read(
            TensorRead::from_tensor(residual_magnitudes.as_ref().unwrap_or(&residual_normalized)),
            &[0, 1],
        )
        .map_err(|err| cuda_error(OP, err))?;
    let residual_ss = download_scalar(ctx, &residual_ss, OP)?;
    Ok(scaled_hermitian_residual_accepts(
        input_ss,
        residual_scale,
        residual_ss,
    ))
}

fn expect_dtype<D: CudaScalar>(
    op: &'static str,
    tensor: Tensor,
    device: usize,
) -> Result<CudaDenseStorage, DenseError> {
    if D::typed(&tensor).is_none() {
        return Err(dtype_mismatch::<D>(op, &tensor));
    }
    Ok(CudaDenseStorage::from_tensor(tensor, device))
}

/// Writes zeros over the leading `len` elements of `dst`, copying them from
/// this context's zero template.
///
/// This is the reset a caller that retains a device destination pays before
/// overwriting it: Tenferro 0.5 cannot fill an existing device buffer
/// (tenferro-rs#1834), so a zero *source* has to exist somewhere, and the
/// context owns the only one — it is the same template [`cuda_region_zero`]
/// reads, grown monotonically to the longest prefix any caller reserves.
///
/// Why `copy_read_into` rather than the region primitive: both move `len`
/// elements, but this is `cutensorPermute` — a pure copy with its own plan
/// cache — while a region fill is a contraction against the `1` operand, which
/// multiplies every element and consumes an entry of the contraction plan LRU
/// that the caller's GEMMs share. A packed prefix needs neither, and the
/// template prefix is exactly the compact offset-0 source `copy_read_into`
/// requires.
///
/// Transfer contract: it uploads (or grows) the template on first use of a
/// length and dtype, and nothing afterwards. Size it once with
/// [`CudaDenseContext::reserve_zero_template`] to make every later reset
/// transfer-free.
pub fn cuda_zero_prefix<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    len: usize,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_zero_prefix";
    ensure_cuda_device(ctx.device, OP, &[("dst", dst.device)])?;
    ensure_payload_dtype::<D>(OP, dst)?;
    if len == 0 {
        return Ok(());
    }
    if dst.len < len {
        return Err(DenseError::OutOfBounds);
    }
    ctx.ensure_zeros::<D>(len)?;
    let (backend, operands) = ctx.split_operands::<D>();
    let Some(zeros) = operands.zeros.as_ref() else {
        return Err(cuda_error(OP, "context zero template is missing"));
    };
    let src_view = zeros.region_view::<D>(len, 1, len, 0)?;
    let dst_view = dst.region_view_mut::<D>(len, 1, len, 0)?;
    COPY_CALLS.fetch_add(1, Ordering::Relaxed);
    backend
        .copy_read_into(
            TensorRead::from_view(src_view),
            TensorWrite::from_view(dst_view),
        )
        .map_err(|err| cuda_error(OP, err))
}

/// Copies the leading compact `rows x cols` block of a device buffer into a
/// packed sub-region of `dst`.
///
/// Tenferro's `copy_read_into` accepts an offset/strided destination view but
/// requires a compact source view at offset 0, so the source is read from its
/// start; the caller owns the proof that the destination region's tree layout
/// is identical to what it reads (see `compile_cuda_qr_plan`). A source longer
/// than the region is accepted because contiguity is a layout predicate: the
/// leading `rows * cols` elements of a compact buffer are themselves compact,
/// which is what lets one maximum-length zero template reset every
/// destination a `tensor!` workspace retains.
pub fn cuda_copy_region_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    dst_ld: usize,
    src: &CudaDenseStorage,
    rows: usize,
    cols: usize,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_copy_region";
    ensure_cuda_device(ctx.device, OP, &[("dst", dst.device), ("src", src.device)])?;
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    COPY_CALLS.fetch_add(1, Ordering::Relaxed);
    if src.len < rows * cols {
        return Err(cuda_error(
            OP,
            format!(
                "source factor holds {} elements; expected at least {rows} x {cols}",
                src.len
            ),
        ));
    }
    let src_view = src.region_view::<D>(rows, cols, rows, 0)?;
    let dst_view = dst.region_view_mut::<D>(rows, cols, dst_ld, dst_offset)?;
    ctx.backend
        .copy_read_into(
            TensorRead::from_view(src_view),
            TensorWrite::from_view(dst_view),
        )
        .map_err(|err| cuda_error(OP, err))
}

/// cuSOLVER SVD of one packed column-major `rows x cols` region:
/// `region = U * diag(s) * Vt` with `k = min(rows, cols)`. `U` (`rows x k`)
/// and `Vt` (`k x cols`) stay device-resident; only the singular values
/// (descending) are downloaded.
pub fn cuda_svd_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Result<(CudaDenseStorage, Vec<f64>, CudaDenseStorage), DenseError> {
    ensure_cuda_device(ctx.device, "cuda_svd", &[("src", src.device)])?;
    let view = src.region_view::<D>(rows, cols, rows, offset)?;
    SOLVER_CALLS.fetch_add(1, Ordering::Relaxed);
    let (u, s, vt) = with_cuda_linalg(&mut ctx.backend, |exec| {
        TensorRead::from_view(view).svd_read(exec)
    })
    .map_err(|err| cuda_error("cuda_svd", err))?;
    let vt = expect_dtype::<D>("cuda_svd", vt, ctx.device)?;
    let s = download_values(ctx, &s)?;
    let u = expect_dtype::<D>("cuda_svd", u, ctx.device)?;
    validate_svd_factor_shapes(u.tensor.shape(), s.len(), vt.tensor.shape(), rows, cols)?;
    Ok((u, s, vt))
}

fn validate_svd_factor_shapes(
    u_shape: &[usize],
    s_len: usize,
    vt_shape: &[usize],
    rows: usize,
    cols: usize,
) -> Result<(), DenseError> {
    let k = rows.min(cols);
    if u_shape != [rows, k] || s_len != k || vt_shape != [k, cols] {
        return Err(cuda_error(
            "cuda_svd",
            format!(
                "device SVD returned U={u_shape:?}, len(S)={s_len}, Vt={vt_shape:?}; expected U=[{rows}, {k}], len(S)={k}, Vt=[{k}, {cols}]"
            ),
        ));
    }
    Ok(())
}

/// cuSOLVER QR of one packed column-major `rows x cols` region:
/// `region = Q * R` with `k = min(rows, cols)`, `Q` (`rows x k`) and `R`
/// (`k x cols`) device-resident, in the positive-diagonal gauge
/// (`R_jj` real and non-negative, phase 1 kept when `R_jj == 0`).
///
/// The gauge is Tenferro's own [`QrGauge::PositiveDiagonal`]: it is applied on
/// device inside the QR primitive, so no diagonal crosses to the host and no
/// TeNeT-side re-gauging selector exists.
pub fn cuda_qr_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Result<(CudaDenseStorage, CudaDenseStorage), DenseError> {
    ensure_cuda_device(ctx.device, "cuda_qr", &[("src", src.device)])?;
    let view = src.region_view::<D>(rows, cols, rows, offset)?;
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    SOLVER_CALLS.fetch_add(1, Ordering::Relaxed);
    let (q, r) = with_cuda_linalg(&mut ctx.backend, |exec| {
        TensorRead::from_view(view).qr_with_options_read(options, exec)
    })
    .map_err(|err| cuda_error("cuda_qr", err))?;
    let r = expect_dtype::<D>("cuda_qr", r, ctx.device)?;
    let q = expect_dtype::<D>("cuda_qr", q, ctx.device)?;
    validate_qr_factor_shapes(q.tensor.shape(), r.tensor.shape(), rows, cols)?;
    Ok((q, r))
}

fn validate_qr_factor_shapes(
    q_shape: &[usize],
    r_shape: &[usize],
    rows: usize,
    cols: usize,
) -> Result<(), DenseError> {
    let k = rows.min(cols);
    if q_shape != [rows, k] || r_shape != [k, cols] {
        return Err(cuda_error(
            "cuda_qr",
            format!(
                "device QR returned shapes Q={q_shape:?}, R={r_shape:?}; expected Q=[{rows}, {k}], R=[{k}, {cols}]"
            ),
        ));
    }
    Ok(())
}

/// cuSOLVER Hermitian eigendecomposition of one packed column-major
/// `n x n` region: eigenvalues are downloaded (host truncation / ordering
/// decisions), eigenvectors stay device-resident (`n x n`, one eigenvector
/// per column, in cuSOLVER's ascending-eigenvalue order).
pub fn cuda_eigh_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<(Vec<f64>, CudaDenseStorage), DenseError> {
    ensure_cuda_device(ctx.device, "cuda_eigh", &[("src", src.device)])?;
    let view = src.region_view::<D>(n, n, n, offset)?;
    SOLVER_CALLS.fetch_add(1, Ordering::Relaxed);
    let (values, vectors) = with_cuda_linalg(&mut ctx.backend, |exec| {
        TensorRead::from_view(view).eigh_read(exec)
    })
    .map_err(|err| cuda_error("cuda_eigh", err))?;
    let vectors = expect_dtype::<D>("cuda_eigh", vectors, ctx.device)?;
    let values = download_values(ctx, &values)?;
    validate_eigh_factor_shapes(values.len(), vectors.tensor.shape(), n)?;
    Ok((values, vectors))
}

fn validate_eigh_factor_shapes(
    values_len: usize,
    vectors_shape: &[usize],
    n: usize,
) -> Result<(), DenseError> {
    if values_len != n || vectors_shape != [n, n] {
        return Err(cuda_error(
            "cuda_eigh",
            format!(
                "device EIGH returned len(values)={values_len}, vectors={vectors_shape:?}; expected len(values)={n}, vectors=[{n}, {n}]"
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The observation counters are process-wide, so the device tests that
    /// assert on their deltas must not overlap with each other.
    static COUNTER_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn rectangular_operand_views_use_parent_native_strides() {
        assert_eq!(cuda_operand_view(MatrixOp::Identity, 2, 3), ([1, 2], false));
        assert_eq!(cuda_operand_view(MatrixOp::Adjoint, 2, 3), ([3, 1], true));
        assert_eq!(cuda_operand_view(MatrixOp::Identity, 3, 4), ([1, 3], false));
        assert_eq!(cuda_operand_view(MatrixOp::Adjoint, 3, 4), ([4, 1], true));
    }

    #[test]
    fn ensure_cuda_device_accepts_matching_operands() {
        assert!(
            ensure_cuda_device(0, "op", &[("a", 0), ("b", 0)]).is_ok(),
            "operands on the context device must be accepted"
        );
    }

    #[test]
    fn ensure_cuda_device_rejects_a_foreign_operand() {
        let err = ensure_cuda_device(0, "cuda_matmul", &[("lhs", 0), ("rhs", 1)])
            .expect_err("an operand on another device must be rejected");
        match err {
            DenseError::Backend {
                backend,
                op,
                message,
            } => {
                assert_eq!(backend, DenseBackend::Cuda);
                assert_eq!(op, "cuda_matmul");
                assert!(
                    message.contains("rhs"),
                    "message names the operand: {message}"
                );
                assert!(
                    message.contains("device 1"),
                    "message names the device: {message}"
                );
            }
            other => panic!("expected a CUDA backend error, got {other:?}"),
        }
    }

    #[test]
    fn svd_factor_shape_contract_covers_rectangular_and_bad_backend_results() {
        assert!(validate_svd_factor_shapes(&[4, 3], 3, &[3, 3], 4, 3).is_ok());
        assert!(validate_svd_factor_shapes(&[3, 3], 3, &[3, 4], 3, 4).is_ok());
        assert!(validate_svd_factor_shapes(&[4, 4], 3, &[3, 3], 4, 3).is_err());
        assert!(validate_svd_factor_shapes(&[4, 3], 2, &[3, 3], 4, 3).is_err());
        assert!(validate_svd_factor_shapes(&[4, 3], 3, &[4, 3], 4, 3).is_err());
    }

    #[test]
    fn qr_factor_shapes_must_match_the_requested_compact_problem() {
        assert!(validate_qr_factor_shapes(&[4, 3], &[3, 3], 4, 3).is_ok());
        assert!(validate_qr_factor_shapes(&[4, 4], &[3, 3], 4, 3).is_err());
        assert!(validate_qr_factor_shapes(&[4, 3], &[4, 3], 4, 3).is_err());
    }

    #[test]
    fn eigh_factor_shapes_must_match_the_requested_square_problem() {
        assert!(validate_eigh_factor_shapes(3, &[3, 3], 3).is_ok());
        assert!(validate_eigh_factor_shapes(2, &[3, 3], 3).is_err());
        assert!(validate_eigh_factor_shapes(3, &[3, 2], 3).is_err());
    }

    #[test]
    fn scaled_hermitian_rule_uses_the_shared_half_residual_threshold() {
        let input_ss: f64 = 2.0;
        let threshold = 64.0 * f64::EPSILON * input_ss.sqrt();
        assert!(scaled_hermitian_residual_accepts(
            input_ss,
            2.0 * threshold * 0.99,
            1.0,
        ));
        assert!(!scaled_hermitian_residual_accepts(
            input_ss,
            2.0 * threshold * 1.01,
            1.0,
        ));
        assert!(!scaled_hermitian_residual_accepts(input_ss, f64::NAN, 1.0,));
        assert!(!scaled_hermitian_residual_accepts(
            input_ss,
            1.0,
            f64::INFINITY,
        ));
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_hermitian_region_is_scaled_and_downloads_only_scalar_metadata() {
        let _serialized = COUNTER_TESTS.lock().unwrap_or_else(|err| err.into_inner());
        let mut ctx = CudaDenseContext::new(0).unwrap();
        let n = 4;
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i + n * i] = 1.0;
        }
        data[n] = 32.0 * f64::EPSILON;
        data[1] = data[n];
        let storage = CudaDenseStorage::upload::<f64>(&ctx, &data).unwrap();

        CUDA_FULL_DOWNLOAD_BYTES.store(0, Ordering::Relaxed);
        CUDA_METADATA_DOWNLOAD_BYTES.store(0, Ordering::Relaxed);
        assert!(cuda_is_hermitian_region::<f64>(&mut ctx, &storage, 0, n).unwrap());
        assert_eq!(CUDA_FULL_DOWNLOAD_BYTES.load(Ordering::Relaxed), 0);
        assert!(CUDA_METADATA_DOWNLOAD_BYTES.load(Ordering::Relaxed) <= 4 * 8);

        let near_threshold = |ctx: &CudaDenseContext, delta: f64| {
            // For [[1, delta], [0, 1]], the shared half-residual rule changes
            // truth value at delta = 128 eps up to negligible O(delta^2).
            CudaDenseStorage::upload::<f64>(ctx, &[1.0, 0.0, delta, 1.0]).unwrap()
        };
        let below = near_threshold(&ctx, 120.0 * f64::EPSILON);
        let above = near_threshold(&ctx, 136.0 * f64::EPSILON);
        assert!(cuda_is_hermitian_region::<f64>(&mut ctx, &below, 0, 2).unwrap());
        assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &above, 0, 2).unwrap());

        let zero = CudaDenseStorage::upload::<f64>(&ctx, &vec![0.0; n * n]).unwrap();
        assert!(cuda_is_hermitian_region::<f64>(&mut ctx, &zero, 0, n).unwrap());

        data[n] = 256.0 * f64::EPSILON;
        let asymmetric = CudaDenseStorage::upload::<f64>(&ctx, &data).unwrap();
        assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &asymmetric, 0, n).unwrap());

        for scale in [f64::from_bits(0x0010_0000_0000_0000), 2.0_f64.powi(500)] {
            let scaled: Vec<_> = data.iter().map(|value| value * scale).collect();
            let scaled = CudaDenseStorage::upload::<f64>(&ctx, &scaled).unwrap();
            assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &scaled, 0, n).unwrap());
        }

        for bad in [f64::NAN, f64::INFINITY] {
            let mut nonfinite = vec![0.0; n * n];
            nonfinite[0] = bad;
            let nonfinite = CudaDenseStorage::upload::<f64>(&ctx, &nonfinite).unwrap();
            assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &nonfinite, 0, n).unwrap());
        }
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_transfer_bytes_scale_with_the_payload_dtype() {
        let _serialized = COUNTER_TESTS.lock().unwrap_or_else(|err| err.into_inner());
        // #1268: a complex payload must move the same *number* of buffers as
        // the real one and exactly `size_of::<Complex64>() / size_of::<f64>()`
        // times the bytes for the same element count.
        let ctx = CudaDenseContext::new(0).unwrap();
        let elements = 16;
        let real: Vec<f64> = (0..elements).map(|index| index as f64).collect();
        let complex: Vec<Complex64> = (0..elements)
            .map(|index| Complex64::new(index as f64, -(index as f64) - 0.5))
            .collect();

        CUDA_FULL_DOWNLOAD_BYTES.store(0, Ordering::Relaxed);
        let real_device = CudaDenseStorage::upload::<f64>(&ctx, &real).unwrap();
        assert_eq!(real_device.dtype(), DenseDType::F64);
        assert_eq!(real_device.download::<f64>(&ctx).unwrap(), real);
        let real_bytes = CUDA_FULL_DOWNLOAD_BYTES.swap(0, Ordering::Relaxed);

        let complex_device = CudaDenseStorage::upload::<Complex64>(&ctx, &complex).unwrap();
        assert_eq!(complex_device.dtype(), DenseDType::C64);
        assert_eq!(complex_device.download::<Complex64>(&ctx).unwrap(), complex);
        let complex_bytes = CUDA_FULL_DOWNLOAD_BYTES.swap(0, Ordering::Relaxed);

        assert_eq!(real_bytes, elements * std::mem::size_of::<f64>());
        assert_eq!(
            complex_bytes,
            real_bytes * std::mem::size_of::<Complex64>() / std::mem::size_of::<f64>()
        );

        // A dtype mismatch is a typed error, never a reinterpretation.
        assert!(matches!(
            complex_device.download::<f64>(&ctx),
            Err(DenseError::DTypeMismatch {
                expected: DenseDType::F64,
                actual: DenseDType::C64,
                ..
            })
        ));
        assert!(matches!(
            real_device.region_view::<Complex64>(4, 4, 4, 0),
            Err(DenseError::DTypeMismatch {
                expected: DenseDType::C64,
                actual: DenseDType::F64,
                ..
            })
        ));
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_transfer_counters_attribute_one_upload_download_and_gemm() {
        let _serialized = COUNTER_TESTS.lock().unwrap_or_else(|err| err.into_inner());
        let mut ctx = CudaDenseContext::new(0).unwrap();
        let n = 2;
        let lhs_host = vec![1.0_f64, 2.0, 3.0, 4.0];
        let rhs_host = vec![5.0_f64, 6.0, 7.0, 8.0];

        reset_cuda_transfer_stats();
        CUDA_FULL_DOWNLOAD_BYTES.store(0, Ordering::Relaxed);
        let lhs = CudaDenseStorage::upload::<f64>(&ctx, &lhs_host).unwrap();
        let rhs = CudaDenseStorage::upload::<f64>(&ctx, &rhs_host).unwrap();
        let mut dst = CudaDenseStorage::upload::<f64>(&ctx, &vec![0.0_f64; n * n]).unwrap();
        let after_uploads = cuda_transfer_stats();
        assert_eq!(after_uploads.h2d_calls, 3);
        assert_eq!(
            after_uploads.h2d_bytes,
            (3 * n * n * std::mem::size_of::<f64>()) as u64
        );
        assert_eq!(after_uploads.device_allocs, 3);
        assert_eq!(after_uploads.d2h_calls, 0);
        assert_eq!(after_uploads.gemm_calls, 0);

        cuda_gemm_region_into::<f64>(
            &mut ctx, &mut dst, 0, n, &lhs, 0, n, &rhs, 0, n, n, n, n, 1.0, 0.0,
        )
        .unwrap();
        let after_gemm = cuda_transfer_stats();
        assert_eq!(after_gemm.gemm_calls, 1);
        assert_eq!(after_gemm.h2d_calls, after_uploads.h2d_calls);
        assert_eq!(after_gemm.d2h_calls, 0);

        let values = dst.download::<f64>(&ctx).unwrap();
        let after_download = cuda_transfer_stats();
        assert_eq!(after_download.d2h_calls, 1);
        assert_eq!(
            after_download.d2h_bytes,
            (n * n * std::mem::size_of::<f64>()) as u64
        );
        // Column-major 2x2 product, so the counters above describe a real GEMM.
        assert_eq!(values, vec![23.0, 34.0, 31.0, 46.0]);

        // The finer-grained test counter stays consistent with the always
        // compiled one for the same download.
        assert_eq!(
            CUDA_FULL_DOWNLOAD_BYTES.swap(0, Ordering::Relaxed) as u64,
            after_download.d2h_bytes
        );

        reset_cuda_transfer_stats();
        assert_eq!(cuda_transfer_stats(), CudaTransferStats::default());
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn warm_up_costs_one_gemm_one_solver_call_and_a_bounded_fixed_traffic() {
        let _serialized = COUNTER_TESTS.lock().unwrap_or_else(|err| err.into_inner());
        // A context built directly here has never submitted work: only
        // `RuntimeBuilder::build` warms one, so this is a cold backend.
        let mut ctx = CudaDenseContext::new(0).unwrap();

        reset_cuda_transfer_stats();
        ctx.warm_up().unwrap();
        let after = cuda_transfer_stats();
        // The documented fixed cost of the warm-up, as the rustdoc states it.
        // `device_allocs` counts the three 1-element uploads plus the
        // eigenvector factor; all four are local to `warm_up` and dropped
        // before it returns, so nothing of this survives construction. There is
        // no live/peak device-buffer observation at this seam to assert on.
        assert_eq!(
            after,
            CudaTransferStats {
                h2d_calls: 3,
                h2d_bytes: 3 * std::mem::size_of::<f64>() as u64,
                d2h_calls: 1,
                d2h_bytes: std::mem::size_of::<f64>() as u64,
                device_allocs: 4,
                gemm_calls: 1,
                solver_calls: 1,
                copy_calls: 0,
            }
        );

        // Warming an already warm context repeats the same bounded work rather
        // than growing with the number of calls.
        reset_cuda_transfer_stats();
        ctx.warm_up().unwrap();
        assert_eq!(cuda_transfer_stats(), after);
        reset_cuda_transfer_stats();
    }

    #[test]
    fn cuda_errors_are_labelled_as_the_cuda_backend() {
        // Regression for #38: CUDA failures must not be reported as Tenferro.
        let err = cuda_error("cuda_svd", "boom");
        let text = err.to_string();
        assert!(text.contains("Cuda"), "formatted error names CUDA: {text}");
        assert!(
            !text.contains("Tenferro"),
            "must not mislabel as Tenferro: {text}"
        );
    }
}
