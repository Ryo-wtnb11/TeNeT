//! CUDA dense boundary: flat device buffers and offset-addressed matrix
//! GEMM, delegated to tenferro-gpu. This module is the only place in the
//! tenet workspace that touches tenferro GPU types; upper layers see opaque
//! storage handles and `DenseError`.

use tenferro_gpu::{download_tensor, upload_tensor, CudaBackend};
use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorDot, TensorRead,
    TensorView, TensorViewMut, TensorWrite,
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

    fn region_view(
        &self,
        rows: usize,
        cols: usize,
        offset: usize,
    ) -> Result<TensorView<'_>, DenseError> {
        let Tensor::F64(tensor) = &self.tensor else {
            return Err(cuda_error("cuda_matmul", "device buffer is not f64"));
        };
        let offset = isize::try_from(offset)
            .map_err(|_| cuda_error("cuda_matmul", "offset does not fit in isize"))?;
        let rows_isize = isize::try_from(rows)
            .map_err(|_| cuda_error("cuda_matmul", "rows do not fit in isize"))?;
        tensor
            .backend_region_view(vec![rows, cols], vec![1, rows_isize], offset)
            .map(TensorView::F64)
            .map_err(|err| cuda_error("cuda_matmul", err))
    }

    fn region_view_mut(
        &mut self,
        rows: usize,
        cols: usize,
        offset: usize,
    ) -> Result<TensorViewMut<'_>, DenseError> {
        let Tensor::F64(tensor) = &mut self.tensor else {
            return Err(cuda_error("cuda_matmul", "device buffer is not f64"));
        };
        let offset = isize::try_from(offset)
            .map_err(|_| cuda_error("cuda_matmul", "offset does not fit in isize"))?;
        let rows_isize = isize::try_from(rows)
            .map_err(|_| cuda_error("cuda_matmul", "rows do not fit in isize"))?;
        tensor
            .backend_region_view_mut(vec![rows, cols], vec![1, rows_isize], offset)
            .map(TensorViewMut::F64)
            .map_err(|err| cuda_error("cuda_matmul", err))
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
    let lhs_view = lhs.region_view(rows, contracted, lhs_offset)?;
    let rhs_view = rhs.region_view(contracted, cols, rhs_offset)?;
    let dst_view = dst.region_view_mut(rows, cols, dst_offset)?;
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: ContractionScalar::F64(1.0),
        beta: ContractionScalar::F64(0.0),
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
