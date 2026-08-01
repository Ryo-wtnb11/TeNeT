//! CUDA storage and GEMM seams for the symmetry-free replay layer.
//!
//! `CudaStorage` is a flat f64 device buffer implementing [`TensorStorage`]
//! (never host-readable: no silent transfers), and [`CudaStorageGemm`]
//! implements the [`StorageGemm`] device replay seam by delegating each
//! coupled-sector matrix GEMM to the tenet-dense CUDA boundary.

use tenet_core::{Placement, TensorStorage};
use tenet_dense::{
    cuda_gemm_region_into, cuda_matmul_region_into, CudaDenseContext, CudaDenseStorage, MatrixOp,
};

use crate::fusion_replay::StorageGemm;
use crate::OperationError;

/// Flat f64 device buffer usable as replay storage.
pub struct CudaStorage(pub CudaDenseStorage);

impl std::fmt::Debug for CudaStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStorage")
            .field("len", &self.0.len())
            .field("device", &self.0.device())
            .finish()
    }
}

impl CudaStorage {
    pub fn upload(ctx: &CudaDenseContext, data: &[f64]) -> Result<Self, OperationError> {
        CudaDenseStorage::upload_f64(ctx, data)
            .map(Self)
            .map_err(OperationError::Dense)
    }

    pub fn download(&self, ctx: &CudaDenseContext) -> Result<Vec<f64>, OperationError> {
        self.0.download_f64(ctx).map_err(OperationError::Dense)
    }
}

impl TensorStorage<f64> for CudaStorage {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn placement(&self) -> Placement {
        Placement::Cuda(self.0.device())
    }
}

/// [`StorageGemm`] over CUDA storage: one tenferro dot-general per
/// coupled-sector matrix, executed in place on device buffer regions.
pub struct CudaStorageGemm<'a> {
    ctx: &'a mut CudaDenseContext,
}

impl<'a> CudaStorageGemm<'a> {
    pub fn new(ctx: &'a mut CudaDenseContext) -> Self {
        Self { ctx }
    }
}

impl StorageGemm<f64, CudaStorage, CudaStorage, CudaStorage> for CudaStorageGemm<'_> {
    fn supports_matmul_with_ops_scaled(&self, lhs_op: MatrixOp, rhs_op: MatrixOp) -> bool {
        lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity
    }

    fn matmul_range_into(
        &mut self,
        dst: &mut CudaStorage,
        dst_offset: usize,
        lhs: &CudaStorage,
        lhs_offset: usize,
        rhs: &CudaStorage,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError> {
        cuda_matmul_region_into(
            self.ctx, &mut dst.0, dst_offset, &lhs.0, lhs_offset, &rhs.0, rhs_offset, rows,
            contracted, cols,
        )
        .map_err(OperationError::Dense)
    }

    fn matmul_range_with_ops_scaled_into(
        &mut self,
        dst: &mut CudaStorage,
        dst_offset: usize,
        lhs: &CudaStorage,
        lhs_offset: usize,
        rhs: &CudaStorage,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: f64,
    ) -> Result<(), OperationError> {
        if lhs_op != MatrixOp::Identity || rhs_op != MatrixOp::Identity {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "CUDA storage GEMM does not implement transformed operands",
            });
        }
        cuda_gemm_region_into(
            self.ctx, &mut dst.0, dst_offset, rows, &lhs.0, lhs_offset, rows, &rhs.0, rhs_offset,
            contracted, rows, contracted, cols, alpha, 0.0,
        )
        .map_err(OperationError::Dense)
    }
}
