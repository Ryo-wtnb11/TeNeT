//! CUDA dense boundary: flat device buffers and offset-addressed matrix
//! GEMM, delegated to tenferro-gpu. This module is the only place in the
//! tenet workspace that touches tenferro GPU types; upper layers see opaque
//! storage handles and `DenseError`.

use std::num::NonZeroUsize;

use num_complex::{Complex32, Complex64};
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_linalg::{QrGauge, QrOptions, TensorReadLinalgExt};
use tenferro_tensor::backend::{BackendSession, BackendSessionHost};
use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorDot,
    TensorElementwise, TensorIndexing, TensorRead, TensorReduction, TensorScalar as TenferroScalar,
    TensorStructural, TensorView, TensorViewMut, TensorWrite, TypedTensor,
};

use super::{DenseBackend, DenseDType, DenseError, MatrixOp};
use crate::cuda_hermitian::{
    power_of_two_normalizer, scaled_hermitian_residual_accepts, HERMITIAN_TOLERANCE_EPSILONS,
};
use crate::cuda_region::{validate_destination_layout, validate_region, CudaRegion};
use crate::tensor::dense_dtype_from_tenferro;

mod cuda_scalar_sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

/// Payload dtypes a TeNeT CUDA buffer may own.
///
/// Sealed to the four floating payloads `f32`, `f64`, [`Complex32`] and
/// [`Complex64`]: structural (fusion-tree) coefficients stay real, so only the
/// *payload* varies. Conjugation is never a payload property here — it is
/// carried as a GEMM operand flag.
///
/// Every payload names a **real lane** ([`TenferroScalar::Real`]: `f32` for
/// `f32` and [`Complex32`], `f64` for `f64` and [`Complex64`]). Real device
/// metadata — a spectrum, a reduction, the rank-0 divisor of the Hermitian
/// test — is produced and consumed in that lane rather than in `f64` by
/// assumption: Tenferro answers a single-precision payload with `F32`
/// metadata and rejects an `F64` rank-0 divisor against it
/// (`benchmarks/history/cuda-single-precision-probe-2026-09-20.md`). Metadata
/// TeNeT decides on is widened to `f64` at the download, so the host-side
/// spectrum contract is unchanged.
///
/// Admitting a dtype here does not by itself open a typed tensor API for it;
/// `tenet`'s `CudaPayload` admits all four (#1336).
pub trait CudaScalar:
    TenferroScalar<Real: CudaRealScalar> + PartialEq + cuda_scalar_sealed::Sealed
{
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

    /// Which of the context's lazily created scalar-operand slots this dtype
    /// owns. One slot per admitted dtype: the operands are *payload*-typed, so
    /// a slot shared between `f32` and `f64` would hand a buffer of the wrong
    /// dtype to the next call.
    #[doc(hidden)]
    const OPERAND_SLOT: usize;

    /// The backend's dtype-erased GEMM coefficient for this payload.
    fn contraction_scalar(self) -> ContractionScalar;

    /// The value's bit pattern, real then imaginary part: a total order key
    /// for grouping equal scales (the float order is only partial).
    #[doc(hidden)]
    fn bit_pattern(self) -> [u64; 2] {
        match self.contraction_scalar() {
            ContractionScalar::F32(value) => [u64::from(value.to_bits()), 0],
            ContractionScalar::F64(value) => [value.to_bits(), 0],
            ContractionScalar::C32(value) => {
                [u64::from(value.re.to_bits()), u64::from(value.im.to_bits())]
            }
            ContractionScalar::C64(value) => [value.re.to_bits(), value.im.to_bits()],
        }
    }

    /// The typed tensor behind a dtype-erased device buffer, if the dtypes agree.
    fn typed(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        tensor.as_typed::<Self>()
    }

    /// Mutable counterpart of [`CudaScalar::typed`].
    fn typed_mut(tensor: &mut Tensor) -> Option<&mut TypedTensor<Self>> {
        tensor.as_typed_mut::<Self>()
    }
}

/// The real lane of a [`CudaScalar`] payload.
///
/// Sealed to `f32` and `f64`, which are exactly the
/// [`TenferroScalar::Real`] types of the admitted payloads. It exists so the
/// adapter's real-metadata paths — spectrum and reduction downloads, the
/// rank-0 divisor upload, the Hermitian tolerance — are typed by the payload
/// instead of assuming double precision, while the values TeNeT's host-side
/// decisions consume stay `f64`.
pub trait CudaRealScalar: TenferroScalar + cuda_scalar_sealed::Sealed {
    /// This lane's machine epsilon, in the `f64` the host decides in.
    const EPSILON: f64;

    /// Widens one downloaded metadata value to the host decision type. Exact:
    /// every `f32` is an `f64`.
    fn widen(self) -> f64;

    /// Narrows a host scalar to this lane, for a rank-0 device operand. The
    /// only values narrowed here are powers of two inside this lane's range,
    /// so the value round-trips.
    fn narrow(value: f64) -> Self;

    /// This lane's `MAX_EXP`: its largest normal power of two is
    /// `2^(MAX_EXP - 1)` and its smallest is `2^-(MAX_EXP - 2)`.
    const MAX_EXP: i32;
}

impl CudaRealScalar for f32 {
    const EPSILON: f64 = f32::EPSILON as f64;
    const MAX_EXP: i32 = f32::MAX_EXP;

    fn widen(self) -> f64 {
        f64::from(self)
    }

    fn narrow(value: f64) -> Self {
        value as Self
    }
}

impl CudaRealScalar for f64 {
    const EPSILON: f64 = f64::EPSILON;
    const MAX_EXP: i32 = f64::MAX_EXP;

    fn widen(self) -> f64 {
        self
    }

    fn narrow(value: f64) -> Self {
        value
    }
}

impl CudaScalar for f32 {
    const DTYPE: DenseDType = DenseDType::F32;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const IS_COMPLEX: bool = false;
    const OPERAND_SLOT: usize = 0;

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::F32(self)
    }
}

impl CudaScalar for Complex32 {
    const DTYPE: DenseDType = DenseDType::C32;
    const ZERO: Self = Complex32::new(0.0, 0.0);
    const ONE: Self = Complex32::new(1.0, 0.0);
    const IS_COMPLEX: bool = true;
    const OPERAND_SLOT: usize = 2;

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::C32(self)
    }
}

impl CudaScalar for f64 {
    const DTYPE: DenseDType = DenseDType::F64;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const IS_COMPLEX: bool = false;
    const OPERAND_SLOT: usize = 1;

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::F64(self)
    }
}

impl CudaScalar for Complex64 {
    const DTYPE: DenseDType = DenseDType::C64;
    const ZERO: Self = Complex64::new(0.0, 0.0);
    const ONE: Self = Complex64::new(1.0, 0.0);
    const IS_COMPLEX: bool = true;
    const OPERAND_SLOT: usize = 3;

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::C64(self)
    }
}

fn dtype_mismatch<D: CudaScalar>(op: &'static str, tensor: &Tensor) -> DenseError {
    match dense_dtype_from_tenferro(tensor.dtype()) {
        Ok(actual) => DenseError::DTypeMismatch {
            op,
            expected: D::DTYPE,
            actual,
        },
        Err(err) => err,
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
///   workspaces and the intermediate tensors of `cuda_hermitian_regions`
///   (`abs`, `div`, `sub`, the reductions) allocate on device but are not
///   visible as buffers here and are therefore not counted.
///   Nor are the solvers' device spectra ([`CudaSpectrum`]) or the
///   `concatenate` output that [`cuda_download_spectra`] gathers them into.
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
struct ScalarOperands {
    ones: Option<CudaDenseStorage>,
    zeros: Option<CudaDenseStorage>,
    /// `α` repeated: the trace's per-element scale (TensorOperations'
    /// `Scaler(α)` before the sum). Filled on the device from the ones
    /// template whenever `α` changes, so only growth uploads.
    scaled: Option<CudaDenseStorage>,
    /// The value and length of the filled prefix of `scaled`.
    scaled_prefix: Option<(ContractionScalar, usize)>,
    /// Payload bytes per element of this slot's dtype, recorded when a buffer
    /// is created so [`CudaDenseContext::scalar_operand_bytes`] needs no dtype
    /// parameter. Zero while the slot is empty.
    element_bytes: usize,
}

/// One [`ScalarOperands`] slot per admitted payload dtype.
const SCALAR_OPERAND_SLOTS: usize = 4;

/// Owns the tenferro CUDA backend for one device ordinal.
pub struct CudaDenseContext {
    backend: CudaBackend,
    device: usize,
    identity: u64,
    operands: [ScalarOperands; SCALAR_OPERAND_SLOTS],
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
        Ok(Self {
            backend,
            device,
            identity: NEXT_CONTEXT_IDENTITY.fetch_add(1, Ordering::Relaxed),
            operands: std::array::from_fn(|_| ScalarOperands::default()),
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
    fn ensure_ones<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
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
    fn ensure_scaled<D: CudaScalar>(&mut self, alpha: D, len: usize) -> Result<(), DenseError> {
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
    fn split_operands<D: CudaScalar>(&mut self) -> (&mut CudaBackend, &ScalarOperands) {
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
    /// Observation counters ([`cuda_transfer_stats`]) are process-wide and are
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

/// Flat [`CudaScalar`] buffer resident on one CUDA device.
///
/// The handle itself is dtype-erased; every typed access names the payload
/// dtype and reports a mismatch as [`DenseError::DTypeMismatch`].
pub struct CudaDenseStorage {
    tensor: Tensor,
    // Fixed by the typed constructor, so `dtype()` never re-maps Tenferro's
    // dtype, which since 0.6.0 includes the unmappable `DType::External`.
    dtype: DenseDType,
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
            dtype: D::DTYPE,
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
        let mut data = typed
            .into_host_vec()
            .map_err(|err| cuda_error("cuda_download", err))?;
        let bytes = std::mem::size_of_val(data.as_slice());
        // A narrowed buffer (`set_active_len`) still transfers its whole
        // allocation; only the active prefix is the value.
        data.truncate(self.len);
        #[cfg(test)]
        CUDA_FULL_DOWNLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
        record_d2h(bytes);
        Ok(data)
    }

    /// The payload dtype this device buffer owns.
    pub fn dtype(&self) -> DenseDType {
        self.dtype
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

    /// Elements the device allocation holds; at least [`Self::len`].
    #[doc(hidden)]
    pub fn capacity(&self) -> usize {
        self.tensor.shape().iter().product()
    }

    /// Sets the active prefix [`Self::len`] reports, and every region bound
    /// is checked against, to `len` elements of the allocation.
    ///
    /// Why: a grow-only device scratch reused across operands of different
    /// sizes must present exactly the length each consumer admits without a
    /// reallocation (and without the zero upload a reallocation costs, #740).
    /// Only the bound narrows; the allocation and its contents are unchanged.
    #[doc(hidden)]
    pub fn set_active_len(&mut self, len: usize) -> Result<(), DenseError> {
        if len > self.capacity() {
            return Err(DenseError::OutOfBounds);
        }
        self.len = len;
        Ok(())
    }

    /// Bounds a matrix view by the active length rather than the
    /// allocation: after [`Self::set_active_len`] the two differ, and the
    /// backend view itself only checks the allocation.
    fn check_matrix_bound(
        &self,
        shape: [usize; 2],
        strides: [usize; 2],
        offset: usize,
    ) -> Result<(), DenseError> {
        if shape.contains(&0) {
            return Ok(());
        }
        let last = shape
            .iter()
            .zip(strides)
            .try_fold(offset, |end, (&dim, stride)| {
                (dim - 1).checked_mul(stride)?.checked_add(end)
            })
            .ok_or(DenseError::ElementCountOverflow)?;
        if last >= self.len {
            return Err(DenseError::OutOfBounds);
        }
        Ok(())
    }

    /// Wraps a device tensor produced by a tenferro op (e.g. a cuSOLVER
    /// factor) as flat storage, after proving it carries `D`'s payload — the
    /// dtype recorded here.
    fn from_tensor<D: CudaScalar>(
        op: &'static str,
        tensor: Tensor,
        device: usize,
    ) -> Result<Self, DenseError> {
        if D::typed(&tensor).is_none() {
            return Err(dtype_mismatch::<D>(op, &tensor));
        }
        DEVICE_ALLOCS.fetch_add(1, Ordering::Relaxed);
        let len = tensor.shape().iter().product();
        Ok(Self {
            tensor,
            dtype: D::DTYPE,
            len,
            device,
        })
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
        self.check_matrix_bound(shape, strides, offset)?;
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
        let actual = self.dtype;
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
        self.check_matrix_bound([rows, cols], [1, ld], offset)?;
        let actual = self.dtype;
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
/// Only these two exist: Tenferro has no in-place strided scale
/// (`scale_tensor_write` is compact-only in 0.5.0 and 0.6.0), so a
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

/// Rejects a descriptor scale of zero, which the backend is free to answer by
/// skipping the source read.
///
/// Every other scale is a plain multiplication, so the caller's NaN and
/// infinities survive it; a zero one is the single value whose backend
/// behaviour does not match `alpha * src`, and silently answering it with a
/// cleared destination is a wrong answer rather than a slow one. The zero
/// scale is expressible — as an exact zero *data* operand — so this is a
/// misuse of the argument, not a missing capability.
fn reject_zero_alpha<D: CudaScalar>(op: &'static str, alpha: D) -> Result<(), DenseError> {
    // IEEE comparison, so `-0.0` is rejected too: it skips the read just as
    // `0.0` does.
    if alpha != D::ZERO {
        return Ok(());
    }
    Err(DenseError::Unsupported {
        op,
        message: "a descriptor alpha of zero would let the backend skip the source read and                   erase NaN/Inf; pass alpha = 1 with CudaRegionCoefficient::Zero to multiply                   by an exact zero instead"
            .to_string(),
    })
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
    alpha: D,
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
    // The structural coefficient is the 1x1 *operand*, never the descriptor
    // alpha: a descriptor alpha of 0 lets CUDA skip the source read and erase
    // NaN/Inf, where the host multiplies and propagates it (typed.rs
    // `cuda_axpby_owned`). The caller's own scale rides the descriptor, and a
    // caller whose scale is zero passes it as the zero *operand* instead
    // ([`CudaRegionCoefficient::Zero`]) for the same reason.
    let accumulation = DotGeneralAccumulation {
        lhs_conj: conj,
        rhs_conj: false,
        alpha: alpha.contraction_scalar(),
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

/// `dst_region = alpha * c * [conj] src_region + beta * dst_region`, where `c`
/// is a 1x1 **data** operand chosen by `coeff` and `alpha` is the caller's own
/// scale, carried by the contraction descriptor.
///
/// This is the single strided data-movement primitive the device structural
/// operations are built on: one strided, offset, rank-N source region of one
/// buffer moved into a strided, offset, rank-N destination region of another,
/// with the axis permutation carried by the destination strides. It submits
/// one `dot_general` against the 1x1 coefficient and allocates no device
/// buffer of its own.
///
/// Transfer contract: a call with [`CudaRegionCoefficient::Buffer`] moves
/// nothing across the host boundary, ever. A call with
/// [`CudaRegionCoefficient::One`] uploads this context's one-element `1` the
/// first time that dtype is used, and a call with
/// [`CudaRegionCoefficient::Zero`] its one-element zero template, and nothing
/// afterwards (see [`CudaDenseContext::scalar_operand_bytes`] and
/// [`CudaDenseContext::reserve_zero_template`]).
///
/// The structural coefficient is a data operand and never the descriptor
/// alpha, because a descriptor alpha lets CUDA skip the source read when it is
/// 0 and so erases NaN/Inf that the host propagates. `alpha` is the caller's
/// *own* scale, which the host applies as one more multiplication. A zero
/// `alpha` is therefore **rejected** (`Unsupported`, IEEE comparison so `-0.0`
/// too), before any device work: pass `alpha = D::ONE` with
/// [`CudaRegionCoefficient::Zero`] to multiply by an exact zero and keep the
/// host's NaN/Inf propagation. Passing `alpha = D::ONE` is the unscaled default
/// this primitive had before it carried a caller scale.
///
/// Disclosed differences from a host `alpha * (c * x)` chain: this path rounds
/// as `alpha * (c * x)` where a host that folds the two scales first rounds as
/// `(alpha * c) * x` (one rounding position, values equal to dtype tolerance,
/// and either order overflows where the other need not); and with
/// [`CudaRegionCoefficient::Zero`] the written zeros carry the sign of
/// `0 * src` alone, not the sign of the caller's `-0.0` or of `c`.
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
/// 0. the descriptor scale is not zero (`Unsupported`);
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
/// operand (a context `1` or zero-template coefficient, or any
/// [`cuda_region_zero`]) that is used from a
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
    alpha: D,
    coeff: CudaRegionCoefficient<'_>,
    beta: CudaRegionBeta,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_region_axpby";
    reject_zero_alpha::<D>(OP, alpha)?;
    match coeff {
        CudaRegionCoefficient::Buffer(coeff, _) => ensure_cuda_device(
            ctx.device,
            OP,
            &[
                ("src", src.device),
                ("coefficient", coeff.device),
                ("dst", dst.device),
            ],
        )?,
        CudaRegionCoefficient::One | CudaRegionCoefficient::Zero => {
            ensure_cuda_device(ctx.device, OP, &[("src", src.device), ("dst", dst.device)])?;
        }
    }
    ensure_payload_dtype::<D>(OP, src)?;
    ensure_payload_dtype::<D>(OP, dst)?;
    if let CudaRegionCoefficient::Buffer(coeff, _) = coeff {
        ensure_payload_dtype::<D>(OP, coeff)?;
    }
    if src_region.dims() != dst_region.dims() {
        return Err(DenseError::ShapeMismatch {
            op: OP,
            expected: dst_region.dims().to_vec(),
            actual: src_region.dims().to_vec(),
        });
    }
    if let CudaRegionCoefficient::Buffer(coeff, offset) = coeff {
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
        CudaRegionCoefficient::Buffer(coeff, offset) => submit_region_axpby::<D>(
            &mut ctx.backend,
            OP,
            src,
            src_region,
            conj,
            alpha,
            coeff,
            offset,
            beta,
            dst,
            dst_region,
        ),
        CudaRegionCoefficient::One => {
            ctx.ensure_ones::<D>(1)?;
            let (backend, operands) = ctx.split_operands::<D>();
            let Some(ones) = operands.ones.as_ref() else {
                return Err(cuda_error(OP, "context scalar operand is missing"));
            };
            submit_region_axpby::<D>(
                backend, OP, src, src_region, conj, alpha, ones, 0, beta, dst, dst_region,
            )
        }
        CudaRegionCoefficient::Zero => {
            // Idempotent once the caller has reserved the template, which is
            // what keeps a warm replay upload-free; one element is all this
            // operand reads.
            ctx.ensure_zeros::<D>(1)?;
            let (backend, operands) = ctx.split_operands::<D>();
            let Some(zeros) = operands.zeros.as_ref() else {
                return Err(cuda_error(OP, "context scalar operand is missing"));
            };
            submit_region_axpby::<D>(
                backend, OP, src, src_region, conj, alpha, zeros, 0, beta, dst, dst_region,
            )
        }
    }
}

/// Where [`cuda_region_axpby`] reads its 1x1 coefficient operand.
///
/// The variants exist because the operand is data, not a descriptor scalar: a
/// caller that needs an exact `0` there — a caller scale of zero, which the
/// host still multiplies by — cannot express it as a descriptor alpha without
/// letting CUDA skip the source read.
#[derive(Clone, Copy)]
pub enum CudaRegionCoefficient<'a> {
    /// The context's shared `1`: an unscaled move, the default for a pure
    /// relayout.
    One,
    /// The first element of the context's zero template. Reserve it with
    /// [`CudaDenseContext::reserve_zero_template`] to keep the submission
    /// upload-free.
    Zero,
    /// A caller-owned device vector at an element offset — block `b` of a
    /// structure's uploaded coefficients is the 1x1 view at offset `b`.
    Buffer(&'a CudaDenseStorage, usize),
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

    ctx.ensure_ones::<D>(1)?;
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
        D::ONE,
        ones,
        0,
        CudaRegionBeta::Overwrite,
        dst,
        dst_region,
    )
}

/// `dst_region += alpha * [conj] trace(src_region)`: the partial trace of one
/// strided source region, accumulated into a strided destination region.
///
/// `src_region` carries the destination's `dims` followed by one axis per
/// traced pair, each already *merged*: extent `t_k` and stride `s_lhs + s_rhs`
/// of the pair's two source axes, so its index walks the diagonal. The
/// trailing axes are contracted jointly against the packed first `prod t_k`
/// elements of the context's ones template, one `dot_general` in all
/// (TensorOperations' cuTENSOR trace builds the same diagonal-stride
/// descriptor and hands it to a reduction, which Tenferro did not expose on
/// views when this route was written against 0.5.0). Pairs are never merged
/// with one another, so no pair needs a stride commensurate with another's.
///
/// `alpha` scales every traced element before the sum, as TensorOperations'
/// `_mapreducedim!(Scaler(α), Adder(), …)` does and the host does: the
/// diagonal is contracted against `alpha` repeated (the scaled template) with
/// a unit descriptor scale, `dst += Σ aᵢ α`. Why not the descriptor: it
/// scales the finished sum, which overflows where `Σ α aᵢ` does not. At
/// `alpha = 1` the ones template is read and nothing is filled. A zero
/// `alpha` adds VectorInterface's `scale(x, 0) = 0` whatever the source
/// holds, so it submits nothing.
///
/// Transfer contract: the ones and scaled templates grow to `prod t_k` on
/// first need and nothing is uploaded afterwards; a change of `alpha` refills
/// the scaled template on the device (one more submission). Reserve them with
/// [`CudaDenseContext::reserve_ones_template`] and
/// [`CudaDenseContext::reserve_scaled_template`] to pay one upload each for a
/// whole replay.
///
/// Validation, all before any device work: devices, payload dtypes, the
/// source rank and leading extents against the destination (`ShapeMismatch`),
/// element counts, an injective destination, and both regions inside their
/// buffers. An empty destination or an empty traced extent adds nothing and
/// returns `Ok` with no submission.
#[allow(clippy::too_many_arguments)]
pub fn cuda_region_trace_accumulate<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    src_region: &CudaRegion,
    conj: bool,
    alpha: D,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_region_trace_accumulate";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device), ("dst", dst.device)])?;
    ensure_payload_dtype::<D>(OP, src)?;
    ensure_payload_dtype::<D>(OP, dst)?;
    let rank = dst_region.dims().len();
    if src_region.dims().len() < rank || src_region.dims()[..rank] != *dst_region.dims() {
        return Err(DenseError::ShapeMismatch {
            op: OP,
            expected: dst_region.dims().to_vec(),
            actual: src_region.dims().to_vec(),
        });
    }
    if src_region.is_empty() {
        return Ok(());
    }
    src_region.element_count()?;
    let trace_dims = &src_region.dims()[rank..];
    let trace_len = trace_dims
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or(DenseError::ElementCountOverflow)?;
    validate_destination_layout(OP, dst_region)?;
    validate_region(src_region, src.len)?;
    validate_region(dst_region, dst.len)?;

    if alpha == D::ZERO {
        return Ok(());
    }
    let unit = alpha == D::ONE;
    if unit {
        ctx.ensure_ones::<D>(trace_len)?;
    } else {
        ctx.ensure_scaled::<D>(alpha, trace_len)?;
    }
    let (backend, operands) = ctx.split_operands::<D>();
    let template = if unit {
        operands.ones.as_ref()
    } else {
        operands.scaled.as_ref()
    };
    let Some(template) = template else {
        return Err(cuda_error(OP, "context scalar operand is missing"));
    };

    let src_strides = src_region
        .strides()
        .iter()
        .map(|&stride| {
            isize::try_from(stride).map_err(|_| DenseError::StrideOverflow { value: stride })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut ones_dims = trace_dims.to_vec();
    ones_dims.push(1);
    let mut ones_strides = Vec::with_capacity(ones_dims.len());
    let mut running = 1isize;
    for &dim in &ones_dims {
        ones_strides.push(running);
        running *= dim as isize;
    }
    let (out_dims, out_strides) = dst_region.contraction_view_metadata()?;

    let lhs =
        src.region_view_nd::<D>(src_region.dims(), &src_strides, src_region.offset_isize()?)?;
    let rhs = template.region_view_nd::<D>(&ones_dims, &ones_strides, 0)?;
    let out = dst.region_view_nd_mut::<D>(&out_dims, &out_strides, dst_region.offset_isize()?)?;
    let traced = trace_dims.len();
    let config = DotGeneralConfig {
        lhs_contracting_dims: (rank..rank + traced).collect(),
        rhs_contracting_dims: (0..traced).collect(),
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj: conj,
        rhs_conj: false,
        alpha: D::ONE.contraction_scalar(),
        beta: D::ONE.contraction_scalar(),
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
        .map_err(|err| cuda_error(OP, err))
}

/// Downloads a small real device tensor of the payload's lane `R` as host
/// values, widened to `f64`. Only used for spectra / diagonals / reductions —
/// the sole tensor-shaped data that is allowed to cross the device boundary
/// implicitly (truncation decisions are host scalar logic).
///
/// The lane is the caller's, not `f64` by assumption: a single-precision
/// payload's singular values, eigenvalues and reductions come back as `F32`
/// (probe `cuda-single-precision-probe-2026-09-20.md`), and demanding `F64`
/// here is what used to make every such call an error. Widening happens on
/// the host after the transfer, so the bytes moved are the lane's own and
/// TeNeT's spectrum contract stays `f64`.
fn download_values<R: CudaRealScalar>(
    ctx: &CudaDenseContext,
    tensor: &Tensor,
) -> Result<Vec<f64>, DenseError> {
    let host = download_tensor(ctx.backend.runtime(), tensor)
        .map_err(|err| cuda_error("cuda_download", err))?;
    if host.dtype() != R::dtype() {
        return Err(cuda_error(
            "cuda_download",
            format!("expected {:?} values, got {:?}", R::dtype(), host.dtype()),
        ));
    }
    let typed = R::into_typed(host).map_err(|err| cuda_error("cuda_download", err))?;
    let values = typed
        .into_host_vec()
        .map_err(|err| cuda_error("cuda_download", err))?;
    let bytes = std::mem::size_of_val(values.as_slice());
    #[cfg(test)]
    CUDA_METADATA_DOWNLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
    record_d2h(bytes);
    Ok(values.into_iter().map(R::widen).collect())
}

/// Uploads a rank-0 operand in the payload's **real** lane.
///
/// The lane is not a symmetry: Tenferro rejects an `F64` rank-0 operand
/// against an `F32` or `C32` tensor as a dtype mismatch, and rejects a
/// *payload*-typed complex operand as a shape mismatch, so "complex tensor
/// with a real rank-0 scalar" is the one production shape (probe
/// `cuda-single-precision-probe-2026-09-20.md`, finding 3).
fn upload_scalar<R: CudaRealScalar>(
    ctx: &CudaDenseContext,
    value: f64,
) -> Result<Tensor, DenseError> {
    let host = R::into_tensor(vec![], vec![R::narrow(value)])
        .map_err(|err| cuda_error("cuda_hermitian", err))?;
    let tensor = upload_tensor(ctx.backend.runtime(), &host)
        .map_err(|err| cuda_error("cuda_hermitian", err))?;
    record_h2d(std::mem::size_of::<R>());
    Ok(tensor)
}

/// The relative anti-Hermitian residual this payload's real lane admits.
fn hermitian_tolerance<D: CudaScalar>() -> f64 {
    HERMITIAN_TOLERANCE_EPSILONS * <D::Real as CudaRealScalar>::EPSILON
}

/// The rank-0 lane operand through which [`scale_by_power_of_two`] multiplies
/// a payload by the exact power of two `normalizer`.
fn power_of_two_operand<D: CudaScalar>(
    ctx: &CudaDenseContext,
    normalizer: f64,
) -> Result<Tensor, DenseError> {
    let operand = if D::IS_COMPLEX {
        normalizer
    } else {
        // Exact: the reciprocal of a power of two inside the lane's range.
        normalizer.recip()
    };
    upload_scalar::<D::Real>(ctx, operand)
}

/// Multiplies a payload tensor by the power of two that `operand` encodes.
fn scale_by_power_of_two<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    op: &'static str,
    tensor: &Tensor,
    operand: &Tensor,
) -> Result<Tensor, DenseError> {
    if D::IS_COMPLEX {
        ctx.backend.mul(tensor, operand)
    } else {
        // Why not `mul`: Tenferro 0.7.1's real `mul` does not broadcast a
        // rank-0 operand, while its real `div` does and, unlike the
        // complex-by-real one, never squares the divisor.
        ctx.backend.div(tensor, operand)
    }
    .map_err(|err| cuda_error(op, err))
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

/// Tests one packed CUDA matrix region with the host EIGH rule; the
/// one-region case of [`cuda_hermitian_regions`].
#[doc(hidden)]
pub fn cuda_is_hermitian_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<bool, DenseError> {
    Ok(cuda_hermitian_regions::<D>(ctx, src, &[(offset, n)])?[0])
}

/// Tests packed square CUDA regions `(offset, n)` of `src` with the host EIGH
/// rule `||(A - A^H)/2||_F <= 64 eps(real(D)) ||A||_F`, returning one decision
/// per region. The residual uses the *conjugate* transpose, so a
/// complex-symmetric non-Hermitian block is rejected.
///
/// The epsilon is the payload's own real lane
/// ([`crate::cuda_hermitian::HERMITIAN_TOLERANCE_EPSILONS`]), matching the
/// host twin `normwise_hermitian`.
///
/// A complex entry whose modulus overflows the lane is rejected here (the
/// `hypot`-based `abs` gives infinity) but admitted by the component-scaled
/// host twin. That disagreement is on the safe side and predates the power
/// of two normalizer; it is why the device tests' scale window stops at
/// `2^(MAX_EXP - 4)`.
///
/// Each region's rule needs three dependent scalar stages: its maximum picks
/// the input normalizer, the residual maximum picks the residual normalizer,
/// and the residual sum of squares decides. Every stage runs over all
/// still-undecided regions before its scalars are gathered with one
/// `concatenate` and downloaded together, so a call makes at most three
/// downloads whatever the region count. Why not per region: each download
/// blocks the host (a D2H plus a CubeCL flush), which made admission the
/// largest TeNeT-owned cost of `eigh` (#1483). The price is that the
/// materialized input, then residual, of every undecided region is live
/// across a stage boundary: one extra copy of the regions' payload, the same
/// order as the eigenvector factor `eigh` allocates anyway.
///
/// Only scalar norm metadata is downloaded; no region is copied to the host.
#[doc(hidden)]
pub fn cuda_hermitian_regions<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    regions: &[(usize, usize)],
) -> Result<Vec<bool>, DenseError> {
    const OP: &str = "cuda_hermitian";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    let max_exp = <D::Real as CudaRealScalar>::MAX_EXP;
    let mut accepted = vec![true; regions.len()];

    // Stage 1: max |A|.
    let mut inputs = Vec::new();
    let mut maxima = Vec::new();
    for (index, &(offset, n)) in regions.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let normal = contiguous_square::<D>(ctx, src, n, [1, n], offset)?;
        let input_abs = ctx
            .backend
            .abs(&normal)
            .map_err(|err| cuda_error(OP, err))?;
        maxima.push(
            ctx.backend
                .reduce_max(&input_abs, &[0, 1])
                .map_err(|err| cuda_error(OP, err))?,
        );
        inputs.push((index, normal));
    }
    let input_scales = download_gathered::<D::Real>(ctx, &maxima, OP)?;
    drop(maxima);

    // Stage 2: sum |A/s|^2 and max |(A - A^H)/s|.
    let mut residuals = Vec::new();
    let mut input_sums = Vec::new();
    let mut residual_maxima = Vec::new();
    for ((index, normal), input_scale) in inputs.into_iter().zip(input_scales) {
        // Pinned Tenferro's CUDA reduce_max propagates NaN. Keep this check
        // before the zero fast path so an otherwise-zero matrix containing NaN
        // is rejected.
        if !input_scale.is_finite() {
            accepted[index] = false;
            continue;
        }
        if input_scale == 0.0 {
            continue;
        }
        let (offset, n) = regions[index];
        let transpose = contiguous_square::<D>(ctx, src, n, [n, 1], offset)?;
        // Conjugation is a backend op on the transposed copy, never a TeNeT loop.
        let transpose = if D::IS_COMPLEX {
            ctx.backend
                .conj(&transpose)
                .map_err(|err| cuda_error(OP, err))?
        } else {
            transpose
        };
        let normalizer =
            power_of_two_operand::<D>(ctx, power_of_two_normalizer(input_scale, max_exp))?;
        let normal_scaled = scale_by_power_of_two::<D>(ctx, OP, &normal, &normalizer)?;
        drop(normal);
        let transpose_scaled = scale_by_power_of_two::<D>(ctx, OP, &transpose, &normalizer)?;
        input_sums.push(sum_of_squares::<D>(ctx, OP, &normal_scaled)?);
        let residual = ctx
            .backend
            .sub(&normal_scaled, &transpose_scaled)
            .map_err(|err| cuda_error(OP, err))?;
        let residual_abs = ctx
            .backend
            .abs(&residual)
            .map_err(|err| cuda_error(OP, err))?;
        residual_maxima.push(
            ctx.backend
                .reduce_max(&residual_abs, &[0, 1])
                .map_err(|err| cuda_error(OP, err))?,
        );
        residuals.push((index, residual));
    }
    let count = residuals.len();
    input_sums.append(&mut residual_maxima);
    let stage2 = download_gathered::<D::Real>(ctx, &input_sums, OP)?;
    drop(input_sums);
    let (input_ss, residual_scales) = stage2.split_at(count);

    // Stage 3: sum |R/r|^2.
    let mut undecided = Vec::new();
    let mut residual_sums = Vec::new();
    for (((index, residual), &input_ss), &residual_scale) in
        residuals.into_iter().zip(input_ss).zip(residual_scales)
    {
        if !residual_scale.is_finite() {
            accepted[index] = false;
            continue;
        }
        if residual_scale == 0.0 {
            accepted[index] = input_ss.is_finite() && input_ss >= 0.0;
            continue;
        }
        let residual_normalizer = power_of_two_normalizer(residual_scale, max_exp);
        let residual_normalizer_tensor = power_of_two_operand::<D>(ctx, residual_normalizer)?;
        let residual_normalized =
            scale_by_power_of_two::<D>(ctx, OP, &residual, &residual_normalizer_tensor)?;
        residual_sums.push(sum_of_squares::<D>(ctx, OP, &residual_normalized)?);
        undecided.push((index, input_ss, residual_normalizer));
    }
    let residual_ss = download_gathered::<D::Real>(ctx, &residual_sums, OP)?;
    for ((index, input_ss, residual_normalizer), residual_ss) in
        undecided.into_iter().zip(residual_ss)
    {
        accepted[index] = scaled_hermitian_residual_accepts(
            input_ss,
            // Exact: the reciprocal of a normal power of two.
            residual_normalizer.recip(),
            residual_ss,
            hermitian_tolerance::<D>(),
        );
    }
    Ok(accepted)
}

/// Materializes the `n x n` region at `offset` with element strides
/// `strides` as a compact `[n, n, 1]` device tensor.
///
/// Why the trailing unit axis: a reduction over axes `[0, 1]` then yields a
/// rank-1 `[1]` tensor, which Tenferro's `concatenate` accepts, whereas the
/// rank-0 result of a rank-2 input would need a reshape copy per scalar.
fn contiguous_square<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    n: usize,
    strides: [usize; 2],
    offset: usize,
) -> Result<Tensor, DenseError> {
    const OP: &str = "cuda_hermitian";
    src.check_matrix_bound([n, n], strides, offset)?;
    let to_isize =
        |value: usize| isize::try_from(value).map_err(|_| cuda_error(OP, "index exceeds isize"));
    let view = src.region_view_nd::<D>(
        &[n, n, 1],
        &[to_isize(strides[0])?, to_isize(strides[1])?, 1],
        to_isize(offset)?,
    )?;
    ctx.backend
        .to_contiguous_read(TensorRead::from_view(view))
        .map_err(|err| cuda_error(OP, err))
}

/// `sum |x|^2` over axes `[0, 1]` of a `[n, n, 1]` payload tensor, in its
/// real lane.
fn sum_of_squares<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    op: &'static str,
    tensor: &Tensor,
) -> Result<Tensor, DenseError> {
    let magnitudes = magnitudes_for_sum_squares::<D>(ctx, op, tensor)?;
    ctx.backend
        .reduce_sum_squares_read(
            TensorRead::from_tensor(magnitudes.as_ref().unwrap_or(tensor)),
            &[0, 1],
        )
        .map_err(|err| cuda_error(op, err))
}

/// Downloads rank-1 real reductions of lane `R` with one transfer: they are
/// concatenated on device first. No parts means no transfer.
fn download_gathered<R: CudaRealScalar>(
    ctx: &mut CudaDenseContext,
    parts: &[Tensor],
    op: &'static str,
) -> Result<Vec<f64>, DenseError> {
    if parts.is_empty() {
        return Ok(Vec::new());
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    let gathered = ctx
        .backend
        .concatenate(&refs, 0)
        .map_err(|err| cuda_error(op, err))?;
    let values = download_values::<R>(ctx, &gathered)?;
    if values.len() != parts.len() {
        return Err(cuda_error(
            op,
            format!(
                "device reductions returned {} values; expected {}",
                values.len(),
                parts.len()
            ),
        ));
    }
    Ok(values)
}

/// Copies the leading compact `rows x cols` block of a device buffer into a
/// packed sub-region of `dst`.
///
/// Tenferro 0.5.0's `copy_read_into` accepted an offset/strided destination
/// view but required a compact source view at offset 0 (0.6.0 accepts strided
/// sources, tenferro-rs#1836; adopting that is leaf M4), so the source is read
/// from its start; the caller owns the proof that the destination region's tree layout
/// is identical to what it reads (see `compile_cuda_qr_plan`). A source longer
/// than the region is accepted because contiguity is a layout predicate: the
/// leading `rows * cols` elements of a compact buffer are themselves compact,
/// so one maximum-length buffer can serve every shorter region.
///
/// One `cutensorPermute` through `copy_read_into` at every destination
/// offset: Tenferro 0.6.0 advertises the shifted operand address's true
/// alignment to cuTENSOR (`device_address_alignment`, tenferro-gpu
/// `cubecl/permutation.rs:771`, tensor4all/tenferro-rs#1836), so an offset view
/// never selects a vectorized kernel its pointer cannot satisfy (#1320). It
/// counts one `copy_calls` and transfers nothing.
///
/// Values pass through cuTENSOR's `alpha = 1` scaling: non-NaN real values
/// (signed zeros, infinities and subnormals included) and finite complex
/// values with no negative-zero component arrive bit-exact. NaN payloads may
/// be canonicalized, and a complex value with a negative-zero or non-finite
/// component follows IEEE multiplication by `(1, 0)`.
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

/// Widens a single-precision device buffer to its double-precision lane,
/// `f32 -> f64` or `Complex32 -> Complex64`, with one device cast
/// (tenferro-gpu 0.7.1 `TensorStructural::cast`, `cubecl/mod.rs:6006`).
///
/// Why: Tenferro 0.7.1 has no reduction that accumulates wider than its
/// operands — the GEMM, `vdot_read` and `norm_squared_read` (cuBLAS
/// `dot`/`dotc`) and `reduce_sum_squares` all sum in the payload dtype — so a
/// reduction that must accumulate as wide as the Host does widens its
/// operands once and then runs the ordinary double-precision reduction. Exact: every `f32` is an
/// `f64`. Costs one device allocation of `capacity * size_of::<W>()` bytes
/// (the whole allocation is cast; [`CudaDenseStorage::len`] carries over) and
/// one elementwise pass; no host transfer.
pub fn cuda_widen<W: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
) -> Result<CudaDenseStorage, DenseError> {
    const OP: &str = "cuda_widen";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    if !matches!(
        (src.dtype, W::DTYPE),
        (DenseDType::F32, DenseDType::F64) | (DenseDType::C32, DenseDType::C64)
    ) {
        return Err(cuda_error(
            OP,
            format!("{:?} does not widen to {:?}", src.dtype, W::DTYPE),
        ));
    }
    let tensor = ctx
        .backend
        .cast(&src.tensor, W::dtype())
        .map_err(|err| cuda_error(OP, err))?;
    let mut wide = CudaDenseStorage::from_tensor::<W>(OP, tensor, ctx.device)?;
    wide.len = src.len;
    Ok(wide)
}

/// A factorization's real spectrum (singular values or eigenvalues), still
/// device-resident in the payload's real lane.
///
/// Why not downloaded by the factorization itself: every download blocks the
/// host (a D2H plus a CubeCL flush), so a per-block download made a
/// block-sparse factorization pay one host sync per coupled sector (#1484).
/// Callers collect the spectra of all their blocks and read them with one
/// [`cuda_download_spectra`].
pub struct CudaSpectrum {
    tensor: Tensor,
}

impl CudaSpectrum {
    /// Number of values.
    pub fn len(&self) -> usize {
        self.tensor.shape().iter().product()
    }

    /// Whether the spectrum holds no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Downloads `spectra` (all from factorizations of payload `D`, so in its real
/// lane) with at most one transfer: two or more nonempty spectra are first concatenated on
/// device. Returns each spectrum's values, widened to `f64`, in input order;
/// the values are the solver's own, since concatenation only moves them.
pub fn cuda_download_spectra<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    spectra: &[CudaSpectrum],
) -> Result<Vec<Vec<f64>>, DenseError> {
    use tenferro_tensor::TensorIndexing;
    const OP: &str = "cuda_download_spectra";
    let parts: Vec<&Tensor> = spectra
        .iter()
        .filter(|spectrum| !spectrum.is_empty())
        .map(|spectrum| &spectrum.tensor)
        .collect();
    let values = match parts.as_slice() {
        [] => Vec::new(),
        [single] => download_values::<D::Real>(ctx, single)?,
        _ => {
            let gathered = ctx
                .backend
                .concatenate(&parts, 0)
                .map_err(|err| cuda_error(OP, err))?;
            download_values::<D::Real>(ctx, &gathered)?
        }
    };
    let total: usize = spectra.iter().map(CudaSpectrum::len).sum();
    if values.len() != total {
        return Err(cuda_error(
            OP,
            format!(
                "downloaded {} spectrum values; expected {total}",
                values.len()
            ),
        ));
    }
    let mut rest = values.as_slice();
    Ok(spectra
        .iter()
        .map(|spectrum| {
            let (head, tail) = rest.split_at(spectrum.len());
            rest = tail;
            head.to_vec()
        })
        .collect())
}

/// cuSOLVER SVD of one packed column-major `rows x cols` region:
/// `region = U * diag(s) * Vt` with `k = min(rows, cols)`. `U` (`rows x k`),
/// the singular values `s` (descending) and `Vt` (`k x cols`) all stay
/// device-resident; read `s` with [`cuda_download_spectra`].
pub fn cuda_svd_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Result<(CudaDenseStorage, CudaSpectrum, CudaDenseStorage), DenseError> {
    ensure_cuda_device(ctx.device, "cuda_svd", &[("src", src.device)])?;
    let view = src.region_view::<D>(rows, cols, rows, offset)?;
    SOLVER_CALLS.fetch_add(1, Ordering::Relaxed);
    let (u, s, vt) = with_cuda_linalg(&mut ctx.backend, |exec| {
        TensorRead::from_view(view).svd_read(exec)
    })
    .map_err(|err| cuda_error("cuda_svd", err))?;
    let vt = CudaDenseStorage::from_tensor::<D>("cuda_svd", vt, ctx.device)?;
    let s = CudaSpectrum { tensor: s };
    let u = CudaDenseStorage::from_tensor::<D>("cuda_svd", u, ctx.device)?;
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
///
/// Every [`CudaScalar`] payload is admitted. Up to Tenferro 0.5.0 the complex
/// payloads could not be: the gauge's `triu` fill materialized its complex zero
/// as `E::cast_from(0u32)`, which NVRTC rejected (tenferro-rs#1833, #1271).
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
    let r = CudaDenseStorage::from_tensor::<D>("cuda_qr", r, ctx.device)?;
    let q = CudaDenseStorage::from_tensor::<D>("cuda_qr", q, ctx.device)?;
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
/// `n x n` region: eigenvalues (ascending) and eigenvectors (`n x n`, one
/// eigenvector per column, in the same order) stay device-resident; read the
/// eigenvalues with [`cuda_download_spectra`] for host ordering and
/// truncation decisions.
pub fn cuda_eigh_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<(CudaSpectrum, CudaDenseStorage), DenseError> {
    ensure_cuda_device(ctx.device, "cuda_eigh", &[("src", src.device)])?;
    let view = src.region_view::<D>(n, n, n, offset)?;
    SOLVER_CALLS.fetch_add(1, Ordering::Relaxed);
    let (values, vectors) = with_cuda_linalg(&mut ctx.backend, |exec| {
        TensorRead::from_view(view).eigh_read(exec)
    })
    .map_err(|err| cuda_error("cuda_eigh", err))?;
    let vectors = CudaDenseStorage::from_tensor::<D>("cuda_eigh", vectors, ctx.device)?;
    let values = CudaSpectrum { tensor: values };
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
    fn a_zero_descriptor_scale_is_rejected_before_any_device_work() {
        // What: `alpha = 0` is the one scale whose backend behaviour is not
        // `alpha * src` — the source read may be skipped, erasing NaN/Inf — so
        // it is a typed rejection rather than a silently different answer. The
        // check needs no device, which is what makes it precede every upload.
        for alpha in [0.0_f64, -0.0] {
            let err = reject_zero_alpha::<f64>("cuda_region_axpby", alpha)
                .expect_err("a zero descriptor scale must be rejected");
            assert!(
                matches!(
                    err,
                    DenseError::Unsupported {
                        op: "cuda_region_axpby",
                        ..
                    }
                ),
                "{err}"
            );
            assert!(
                err.to_string().contains("CudaRegionCoefficient::Zero"),
                "{err}"
            );
        }
        let complex_zero = Complex64::new(-0.0, 0.0);
        assert!(reject_zero_alpha::<Complex64>("cuda_region_axpby", complex_zero).is_err());

        // Every other scale passes, including the non-finite ones: they are
        // ordinary multiplications.
        for alpha in [1.0_f64, -2.5, f64::NAN, f64::INFINITY] {
            assert!(reject_zero_alpha::<f64>("cuda_region_axpby", alpha).is_ok());
        }
        assert!(
            reject_zero_alpha::<Complex64>("cuda_region_axpby", Complex64::new(0.0, -1.0)).is_ok()
        );
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

    /// The tolerance the Hermitian rule is given is the payload's own lane:
    /// unchanged for the double-precision payloads, ~5e8 wider for the
    /// single-precision ones. The rule itself is tested in `cuda_hermitian`,
    /// which ordinary CI runs.
    #[test]
    fn the_hermitian_tolerance_follows_the_payload_real_lane() {
        assert_eq!(hermitian_tolerance::<f64>(), 64.0 * f64::EPSILON);
        assert_eq!(hermitian_tolerance::<Complex64>(), 64.0 * f64::EPSILON);
        let single = 64.0 * f64::from(f32::EPSILON);
        assert_eq!(hermitian_tolerance::<f32>(), single);
        assert_eq!(hermitian_tolerance::<Complex32>(), single);
        assert!(hermitian_tolerance::<f32>() > hermitian_tolerance::<f64>());
    }

    /// Each admitted dtype owns its own scalar-operand slot, so a context used
    /// from two dtypes never hands one of them the other's payload-typed `1`.
    #[test]
    fn every_admitted_dtype_owns_a_distinct_operand_slot() {
        let slots = [
            f32::OPERAND_SLOT,
            f64::OPERAND_SLOT,
            Complex32::OPERAND_SLOT,
            Complex64::OPERAND_SLOT,
        ];
        for (index, slot) in slots.iter().enumerate() {
            assert!(*slot < SCALAR_OPERAND_SLOTS);
            assert!(
                !slots[..index].contains(slot),
                "slot {slot} is claimed twice"
            );
        }
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
