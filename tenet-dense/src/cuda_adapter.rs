//! CUDA dense boundary: flat device buffers and offset-addressed matrix
//! GEMM, delegated to tenferro-gpu. This module is the only place in the
//! tenet workspace that touches tenferro GPU types; upper layers see opaque
//! storage handles and `DenseError`.

use tenferro_gpu::{download_tensor, upload_tensor, CudaBackend};
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorDot, TensorRead,
    TensorView, TensorViewCanonicalization, TensorViewMut, TensorWrite,
};

use super::{DenseBackend, DenseError};

fn cuda_error(op: &'static str, err: impl std::fmt::Display) -> DenseError {
    DenseError::Backend {
        backend: DenseBackend::Tenferro,
        op,
        message: err.to_string(),
    }
}

/// Owns the tenferro CUDA backend for one device ordinal.
pub struct CudaDenseContext {
    backend: CudaBackend,
    device: usize,
}

impl CudaDenseContext {
    pub fn new(device: usize) -> Result<Self, DenseError> {
        let backend = CudaBackend::new(device).map_err(|err| cuda_error("cuda_context", err))?;
        Ok(Self { backend, device })
    }

    pub fn device(&self) -> usize {
        self.device
    }
}

/// Flat f64 buffer resident on one CUDA device.
pub struct CudaDenseStorage {
    tensor: Tensor,
    len: usize,
    device: usize,
}

impl CudaDenseStorage {
    /// Uploads host data as a flat device buffer.
    pub fn upload_f64(ctx: &CudaDenseContext, data: &[f64]) -> Result<Self, DenseError> {
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
    pub fn download_f64(&self, ctx: &CudaDenseContext) -> Result<Vec<f64>, DenseError> {
        let host = download_tensor(ctx.backend.runtime(), &self.tensor)
            .map_err(|err| cuda_error("cuda_download", err))?;
        match host {
            Tensor::F64(tensor) => tensor
                .host_data()
                .map(|data| data.to_vec())
                .map_err(|err| cuda_error("cuda_download", err)),
            other => Err(cuda_error(
                "cuda_download",
                format!("expected f64 device buffer, got {:?}", other.dtype()),
            )),
        }
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
    fn region_view(
        &self,
        rows: usize,
        cols: usize,
        ld: usize,
        offset: usize,
    ) -> Result<TensorView<'_>, DenseError> {
        let Tensor::F64(tensor) = &self.tensor else {
            return Err(cuda_error("cuda_region", "device buffer is not f64"));
        };
        let offset = isize::try_from(offset)
            .map_err(|_| cuda_error("cuda_region", "offset does not fit in isize"))?;
        let ld_isize = isize::try_from(ld)
            .map_err(|_| cuda_error("cuda_region", "leading dimension does not fit in isize"))?;
        tensor
            .backend_region_view(vec![rows, cols], vec![1, ld_isize], offset)
            .map(TensorView::F64)
            .map_err(|err| cuda_error("cuda_region", err))
    }

    fn region_view_mut(
        &mut self,
        rows: usize,
        cols: usize,
        ld: usize,
        offset: usize,
    ) -> Result<TensorViewMut<'_>, DenseError> {
        let Tensor::F64(tensor) = &mut self.tensor else {
            return Err(cuda_error("cuda_region", "device buffer is not f64"));
        };
        let offset = isize::try_from(offset)
            .map_err(|_| cuda_error("cuda_region", "offset does not fit in isize"))?;
        let ld_isize = isize::try_from(ld)
            .map_err(|_| cuda_error("cuda_region", "leading dimension does not fit in isize"))?;
        tensor
            .backend_region_view_mut(vec![rows, cols], vec![1, ld_isize], offset)
            .map(TensorViewMut::F64)
            .map_err(|err| cuda_error("cuda_region", err))
    }
}

/// Column-major matrix GEMM over device buffer regions:
/// `dst[dst_offset..][rows x cols] = lhs_part * rhs_part` (overwrite).
#[allow(clippy::too_many_arguments)]
pub fn cuda_matmul_region_into(
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
    cuda_gemm_region_into(
        ctx, dst, dst_offset, rows, lhs, lhs_offset, rows, rhs, rhs_offset, contracted, rows,
        contracted, cols, 1.0, 0.0,
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
pub fn cuda_gemm_region_into(
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
    alpha: f64,
    beta: f64,
) -> Result<(), DenseError> {
    let lhs_view = lhs.region_view(m, k, lhs_ld, lhs_offset)?;
    let rhs_view = rhs.region_view(k, n, rhs_ld, rhs_offset)?;
    let dst_view = dst.region_view_mut(m, n, dst_ld, dst_offset)?;
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: ContractionScalar::F64(alpha),
        beta: ContractionScalar::F64(beta),
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
            .map(|data| data.to_vec())
            .map_err(|err| cuda_error("cuda_download", err)),
        other => Err(cuda_error(
            "cuda_download",
            format!("expected f64 values, got {:?}", other.dtype()),
        )),
    }
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
    let view = src.region_view(rows, cols, rows, offset)?;
    let mut outputs = ctx
        .backend
        .svd_read(view)
        .map_err(|err| cuda_error("cuda_svd", err))?;
    if outputs.len() != 3 {
        return Err(cuda_error("cuda_svd", "device SVD must return (U, S, Vt)"));
    }
    let vt = expect_f64("cuda_svd", outputs.pop().expect("len checked"), ctx.device)?;
    let s = download_values(ctx, &outputs.pop().expect("len checked"))?;
    let u = expect_f64("cuda_svd", outputs.pop().expect("len checked"), ctx.device)?;
    Ok((u, s, vt))
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
    let view = src.region_view(rows, cols, rows, offset)?;
    let mut outputs = ctx
        .backend
        .qr_read(view)
        .map_err(|err| cuda_error("cuda_qr", err))?;
    if outputs.len() != 2 {
        return Err(cuda_error("cuda_qr", "device QR must return (Q, R)"));
    }
    let r = expect_f64("cuda_qr", outputs.pop().expect("len checked"), ctx.device)?;
    let q = expect_f64("cuda_qr", outputs.pop().expect("len checked"), ctx.device)?;
    let k = rows.min(cols);
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
    Ok((q, r, diag))
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
    let view = src.region_view(n, n, n, offset)?;
    let mut outputs = ctx
        .backend
        .eigh_read(view)
        .map_err(|err| cuda_error("cuda_eigh", err))?;
    if outputs.len() != 2 {
        return Err(cuda_error(
            "cuda_eigh",
            "device eigh must return (values, vectors)",
        ));
    }
    let vectors = expect_f64("cuda_eigh", outputs.pop().expect("len checked"), ctx.device)?;
    let values = download_values(ctx, &outputs.pop().expect("len checked"))?;
    Ok((values, vectors))
}
