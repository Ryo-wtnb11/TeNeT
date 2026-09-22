//! CUDA storage and GEMM seams for the symmetry-free replay layer.
//!
//! `CudaStorage<D>` is a flat device buffer of [`CudaScalar`] payload `D`
//! implementing [`TensorStorage`] (never host-readable: no silent transfers),
//! and [`CudaStorageGemm`] implements the [`StorageGemm`] device replay seam by
//! delegating each coupled-sector matrix GEMM to the tenet-dense CUDA boundary.
//!
//! The payload dtype is the only thing that varies: structural (fusion-tree)
//! coefficients stay real, and operand conjugation stays a GEMM flag rather
//! than a materialized buffer.

use std::marker::PhantomData;

use tenet_core::{Placement, TensorStorage};
use tenet_dense::{
    cuda_gemm_region_with_ops_into, cuda_matmul_region_into, cuda_widen, CudaDenseContext,
    CudaDenseStorage, CudaScalar, MatrixOp,
};

use crate::fusion_replay::StorageGemm;
use crate::OperationError;

/// Flat device buffer of payload `D` usable as replay storage.
///
/// The default `D = f64` keeps the historical `CudaStorage` spelling valid.
pub struct CudaStorage<D: CudaScalar = f64>(pub CudaDenseStorage, PhantomData<D>);

impl<D: CudaScalar> std::fmt::Debug for CudaStorage<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStorage")
            .field("len", &self.0.len())
            .field("dtype", &self.0.dtype())
            .field("device", &self.0.device())
            .finish()
    }
}

impl<D: CudaScalar> CudaStorage<D> {
    /// Uploads borrowed host data. Costs the one host copy Tenferro's
    /// owned-host-tensor upload requires; see [`CudaDenseStorage::upload`].
    pub fn upload(ctx: &CudaDenseContext, data: &[D]) -> Result<Self, OperationError> {
        Self::upload_owned(ctx, data.to_vec())
    }

    /// Uploads owned host data by moving it into the uploaded host tensor.
    pub fn upload_owned(ctx: &CudaDenseContext, data: Vec<D>) -> Result<Self, OperationError> {
        CudaDenseStorage::upload_owned(ctx, data)
            .map(|storage| Self(storage, PhantomData))
            .map_err(OperationError::Dense)
    }

    /// A fresh buffer of `len` exact zeros, zeroed on the device with no host
    /// transfer; see [`CudaDenseStorage::zeros`].
    pub fn zeros(ctx: &CudaDenseContext, len: usize) -> Result<Self, OperationError> {
        CudaDenseStorage::zeros::<D>(ctx, len)
            .map(|storage| Self(storage, PhantomData))
            .map_err(OperationError::Dense)
    }

    pub fn download(&self, ctx: &CudaDenseContext) -> Result<Vec<D>, OperationError> {
        self.0.download(ctx).map_err(OperationError::Dense)
    }

    /// This buffer widened to the double-precision lane `W` on the device;
    /// see [`cuda_widen`].
    pub fn widened<W: CudaScalar>(
        &self,
        ctx: &mut CudaDenseContext,
    ) -> Result<CudaStorage<W>, OperationError> {
        cuda_widen::<W>(ctx, &self.0)
            .map(|storage| CudaStorage(storage, PhantomData))
            .map_err(OperationError::Dense)
    }
}

impl<D: CudaScalar> TensorStorage<D> for CudaStorage<D> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn placement(&self) -> Placement {
        Placement::Cuda(self.0.device())
    }
}

/// Operand orientations the device GEMM seam accepts. `Transpose` has no
/// conjugation-free device analogue here and is rejected before device work.
fn cuda_operand_is_supported(op: MatrixOp) -> bool {
    matches!(op, MatrixOp::Identity | MatrixOp::Adjoint)
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

impl<D: CudaScalar> StorageGemm<D, CudaStorage<D>, CudaStorage<D>, CudaStorage<D>>
    for CudaStorageGemm<'_>
{
    fn supports_matmul_with_ops_scaled(&self, lhs_op: MatrixOp, rhs_op: MatrixOp) -> bool {
        cuda_operand_is_supported(lhs_op) && cuda_operand_is_supported(rhs_op)
    }

    fn matmul_range_into(
        &mut self,
        dst: &mut CudaStorage<D>,
        dst_offset: usize,
        lhs: &CudaStorage<D>,
        lhs_offset: usize,
        rhs: &CudaStorage<D>,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError> {
        cuda_matmul_region_into::<D>(
            self.ctx, &mut dst.0, dst_offset, &lhs.0, lhs_offset, &rhs.0, rhs_offset, rows,
            contracted, cols,
        )
        .map_err(OperationError::Dense)
    }

    fn matmul_range_with_ops_scaled_into(
        &mut self,
        dst: &mut CudaStorage<D>,
        dst_offset: usize,
        lhs: &CudaStorage<D>,
        lhs_offset: usize,
        rhs: &CudaStorage<D>,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
    ) -> Result<(), OperationError> {
        if !(cuda_operand_is_supported(lhs_op) && cuda_operand_is_supported(rhs_op)) {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "CUDA storage GEMM supports only identity and adjoint operands",
            });
        }
        cuda_gemm_region_with_ops_into::<D>(
            self.ctx,
            &mut dst.0,
            dst_offset,
            &lhs.0,
            lhs_offset,
            &rhs.0,
            rhs_offset,
            rows,
            contracted,
            cols,
            lhs_op,
            rhs_op,
            alpha,
            D::ZERO,
        )
        .map_err(OperationError::Dense)
    }
}
