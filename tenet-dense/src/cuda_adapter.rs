//! CUDA dense boundary: flat device buffers and offset-addressed matrix
//! GEMM, delegated to tenferro-gpu. This module is the only place in the
//! tenet workspace that touches tenferro GPU types; upper layers see opaque
//! storage handles and `DenseError`.

use num_complex::Complex64;
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_linalg::TensorReadLinalgExt;
use tenferro_tensor::backend::{BackendSession, BackendSessionHost};
use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorDot,
    TensorElementwise, TensorRead, TensorReduction, TensorScalar as TenferroScalar,
    TensorStructural, TensorView, TensorViewCanonicalization, TensorViewMut, TensorWrite,
    TypedTensor,
};

use super::{DenseBackend, DenseDType, DenseError, MatrixOp};
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

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
static CUDA_FULL_DOWNLOAD_BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static CUDA_METADATA_DOWNLOAD_BYTES: AtomicUsize = AtomicUsize::new(0);

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

/// Owns the tenferro CUDA backend for one device ordinal.
pub struct CudaDenseContext {
    backend: CudaBackend,
    device: usize,
}

impl CudaDenseContext {
    pub fn new(device: usize) -> Result<Self, DenseError> {
        let ordinal = u32::try_from(device)
            .map_err(|_| cuda_error("cuda_context", "device ordinal exceeds u32"))?;
        let backend = CudaBackend::new(CudaDeviceId::from_ordinal(ordinal))
            .map_err(|err| cuda_error("cuda_context", err))?;
        Ok(Self { backend, device })
    }

    pub fn device(&self) -> usize {
        self.device
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
    /// Uploads host data as a flat device buffer.
    pub fn upload<D: CudaScalar>(ctx: &CudaDenseContext, data: &[D]) -> Result<Self, DenseError> {
        let host = Tensor::from_vec_col_major(vec![data.len()], data.to_vec())
            .map_err(|err| cuda_error("cuda_upload", err))?;
        let tensor = upload_tensor(ctx.backend.runtime(), &host)
            .map_err(|err| cuda_error("cuda_upload", err))?;
        Ok(Self {
            tensor,
            len: data.len(),
            device: ctx.device,
        })
    }

    /// Downloads the flat device buffer back to host data.
    pub fn download<D: CudaScalar>(&self, ctx: &CudaDenseContext) -> Result<Vec<D>, DenseError> {
        ensure_cuda_device(ctx.device, "cuda_download", &[("source", self.device)])?;
        let host = download_tensor(ctx.backend.runtime(), &self.tensor)
            .map_err(|err| cuda_error("cuda_download", err))?;
        if host.dtype() != D::dtype() {
            return Err(dtype_mismatch::<D>("cuda_download", &host));
        }
        let data = D::as_slice(&host).map_err(|err| cuda_error("cuda_download", err))?;
        #[cfg(test)]
        CUDA_FULL_DOWNLOAD_BYTES.fetch_add(std::mem::size_of_val(data), Ordering::Relaxed);
        Ok(data.to_vec())
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

/// Downloads a small real (f64) device tensor as host values. Only used for
/// spectra / diagonals — the sole tensor-shaped data that is allowed to
/// cross the device boundary implicitly (truncation decisions are host
/// scalar logic).
fn download_values(ctx: &CudaDenseContext, tensor: &Tensor) -> Result<Vec<f64>, DenseError> {
    let host = download_tensor(ctx.backend.runtime(), tensor)
        .map_err(|err| cuda_error("cuda_download", err))?;
    match host {
        Tensor::F64(tensor) => tensor
            .host_data()
            .map(|data| {
                #[cfg(test)]
                CUDA_METADATA_DOWNLOAD_BYTES
                    .fetch_add(data.len() * std::mem::size_of::<f64>(), Ordering::Relaxed);
                data.to_vec()
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
    upload_tensor(ctx.backend.runtime(), &host).map_err(|err| cuda_error("cuda_hermitian", err))
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

/// Tests one packed f64 CUDA matrix region with the host EIGH rule
/// `||(A - A^T)/2||_F <= 64 eps ||A||_F`.
///
/// The normal and transposed views are materialized and reduced on device.
/// Only scalar norm metadata is downloaded; the receiver region is never
/// copied to the host. Operation-local workspaces are dropped on return.
#[doc(hidden)]
pub fn cuda_is_hermitian_region(
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

    let normal_view = src.region_view_strided::<f64>([n, n], [1, n], offset)?;
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

    let transpose_view = src.region_view_strided::<f64>([n, n], [n, 1], offset)?;
    let transpose = ctx
        .backend
        .to_contiguous_read(TensorRead::from_view(transpose_view))
        .map_err(|err| cuda_error(OP, err))?;
    let scale = upload_scalar(ctx, input_scale)?;
    let normal_scaled = ctx
        .backend
        .div(&normal, &scale)
        .map_err(|err| cuda_error(OP, err))?;
    let transpose_scaled = ctx
        .backend
        .div(&transpose, &scale)
        .map_err(|err| cuda_error(OP, err))?;
    let input_ss = ctx
        .backend
        .reduce_sum_squares_read(TensorRead::from_tensor(&normal_scaled), &[0, 1])
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
    let residual_ss = ctx
        .backend
        .reduce_sum_squares_read(TensorRead::from_tensor(&residual_normalized), &[0, 1])
        .map_err(|err| cuda_error(OP, err))?;
    let residual_ss = download_scalar(ctx, &residual_ss, OP)?;
    Ok(scaled_hermitian_residual_accepts(
        input_ss,
        residual_scale,
        residual_ss,
    ))
}

fn expect_f64(
    op: &'static str,
    tensor: Tensor,
    device: usize,
) -> Result<CudaDenseStorage, DenseError> {
    match &tensor {
        Tensor::F64(_) => Ok(CudaDenseStorage::from_tensor(tensor, device)),
        other => Err(cuda_error(
            op,
            format!("expected an f64 device factor, got {:?}", other.dtype()),
        )),
    }
}

/// cuSOLVER SVD of one packed column-major `rows x cols` region:
/// `region = U * diag(s) * Vt` with `k = min(rows, cols)`. `U` (`rows x k`)
/// and `Vt` (`k x cols`) stay device-resident; only the singular values
/// (descending) are downloaded.
pub fn cuda_svd_region(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Result<(CudaDenseStorage, Vec<f64>, CudaDenseStorage), DenseError> {
    ensure_cuda_device(ctx.device, "cuda_svd", &[("src", src.device)])?;
    let view = src.region_view::<f64>(rows, cols, rows, offset)?;
    let (u, s, vt) = with_cuda_linalg(&mut ctx.backend, |exec| {
        TensorRead::from_view(view).svd_read(exec)
    })
    .map_err(|err| cuda_error("cuda_svd", err))?;
    let vt = expect_f64("cuda_svd", vt, ctx.device)?;
    let s = download_values(ctx, &s)?;
    let u = expect_f64("cuda_svd", u, ctx.device)?;
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
/// (`k x cols`) device-resident. Also returns the host copy of `R`'s
/// diagonal so the caller can apply the positive-diagonal gauge (matching
/// the host `qr_compact`) via sign selectors.
pub fn cuda_qr_region(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Result<(CudaDenseStorage, CudaDenseStorage, Vec<f64>), DenseError> {
    ensure_cuda_device(ctx.device, "cuda_qr", &[("src", src.device)])?;
    let view = src.region_view::<f64>(rows, cols, rows, offset)?;
    let (q, r) = with_cuda_linalg(&mut ctx.backend, |exec| {
        TensorRead::from_view(view).qr_read(exec)
    })
    .map_err(|err| cuda_error("cuda_qr", err))?;
    let r = expect_f64("cuda_qr", r, ctx.device)?;
    let q = expect_f64("cuda_qr", q, ctx.device)?;
    let k = rows.min(cols);
    validate_qr_factor_shapes(q.tensor.shape(), r.tensor.shape(), rows, cols)?;
    // R's diagonal as a strided [k] view (stride k + 1), compacted on
    // device, then downloaded: k scalars, not the factor.
    let diag = {
        let Tensor::F64(tensor) = &r.tensor else {
            return Err(cuda_error("cuda_qr", "device R factor is not f64"));
        };
        let diag_view = tensor
            .backend_region_view(vec![k], vec![k as isize + 1], 0)
            .map_err(|err| cuda_error("cuda_qr", err))?;
        let compact = ctx
            .backend
            .to_contiguous(&diag_view)
            .map_err(|err| cuda_error("cuda_qr", err))?;
        download_values(ctx, &Tensor::F64(compact))?
    };
    if diag.len() != k {
        return Err(cuda_error(
            "cuda_qr",
            format!(
                "device QR returned diagonal length {}; expected {k}",
                diag.len()
            ),
        ));
    }
    Ok((q, r, diag))
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
pub fn cuda_eigh_region(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<(Vec<f64>, CudaDenseStorage), DenseError> {
    ensure_cuda_device(ctx.device, "cuda_eigh", &[("src", src.device)])?;
    let view = src.region_view::<f64>(n, n, n, offset)?;
    let (values, vectors) = with_cuda_linalg(&mut ctx.backend, |exec| {
        TensorRead::from_view(view).eigh_read(exec)
    })
    .map_err(|err| cuda_error("cuda_eigh", err))?;
    let vectors = expect_f64("cuda_eigh", vectors, ctx.device)?;
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
        let mut ctx = CudaDenseContext::new(0).unwrap();
        let n = 4;
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i + n * i] = 1.0;
        }
        data[0 + n] = 32.0 * f64::EPSILON;
        data[1] = data[0 + n];
        let storage = CudaDenseStorage::upload::<f64>(&ctx, &data).unwrap();

        CUDA_FULL_DOWNLOAD_BYTES.store(0, Ordering::Relaxed);
        CUDA_METADATA_DOWNLOAD_BYTES.store(0, Ordering::Relaxed);
        assert!(cuda_is_hermitian_region(&mut ctx, &storage, 0, n).unwrap());
        assert_eq!(CUDA_FULL_DOWNLOAD_BYTES.load(Ordering::Relaxed), 0);
        assert!(CUDA_METADATA_DOWNLOAD_BYTES.load(Ordering::Relaxed) <= 4 * 8);

        let near_threshold = |ctx: &CudaDenseContext, delta: f64| {
            // For [[1, delta], [0, 1]], the shared half-residual rule changes
            // truth value at delta = 128 eps up to negligible O(delta^2).
            CudaDenseStorage::upload::<f64>(ctx, &[1.0, 0.0, delta, 1.0]).unwrap()
        };
        let below = near_threshold(&ctx, 120.0 * f64::EPSILON);
        let above = near_threshold(&ctx, 136.0 * f64::EPSILON);
        assert!(cuda_is_hermitian_region(&mut ctx, &below, 0, 2).unwrap());
        assert!(!cuda_is_hermitian_region(&mut ctx, &above, 0, 2).unwrap());

        let zero = CudaDenseStorage::upload::<f64>(&ctx, &vec![0.0; n * n]).unwrap();
        assert!(cuda_is_hermitian_region(&mut ctx, &zero, 0, n).unwrap());

        data[0 + n] = 256.0 * f64::EPSILON;
        let asymmetric = CudaDenseStorage::upload::<f64>(&ctx, &data).unwrap();
        assert!(!cuda_is_hermitian_region(&mut ctx, &asymmetric, 0, n).unwrap());

        for scale in [f64::from_bits(0x0010_0000_0000_0000), 2.0_f64.powi(500)] {
            let scaled: Vec<_> = data.iter().map(|value| value * scale).collect();
            let scaled = CudaDenseStorage::upload::<f64>(&ctx, &scaled).unwrap();
            assert!(!cuda_is_hermitian_region(&mut ctx, &scaled, 0, n).unwrap());
        }

        for bad in [f64::NAN, f64::INFINITY] {
            let mut nonfinite = vec![0.0; n * n];
            nonfinite[0] = bad;
            let nonfinite = CudaDenseStorage::upload::<f64>(&ctx, &nonfinite).unwrap();
            assert!(!cuda_is_hermitian_region(&mut ctx, &nonfinite, 0, n).unwrap());
        }
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_transfer_bytes_scale_with_the_payload_dtype() {
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
