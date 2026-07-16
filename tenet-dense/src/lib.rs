#![forbid(unsafe_code)]

//! Dense block execution boundary for TeNeT.
//!
//! Symmetric tensor algorithms lower to this crate through TeNeT-owned storage
//! views and executors. The storage placement determines the execution path:
//! host views use host kernels, and future device views should use device
//! kernels without exposing concrete runtimes to TensorMap-level code.

mod dot;
mod dtype;
mod error;
mod executor;
mod layout;
mod scalar;
mod tensor;
mod view;

#[cfg(feature = "cuda")]
mod cuda_adapter;
#[cfg(feature = "tenferro")]
mod tenferro_adapter;
#[cfg(test)]
mod tests;

pub use dot::DenseDotConfig;
pub use dtype::{DenseBackend, DenseDType, DensePlacement};
pub use error::DenseError;
pub use executor::{strided_batch_runs, DenseExecutor, DenseGemmBatchJob, MatrixOp};
pub use scalar::DenseScalar;
pub use tensor::DenseTensor;
pub use view::{DenseRead, DenseView, DenseViewMut, DenseWrite};

#[cfg(feature = "tenferro")]
pub use tenferro_adapter::{DefaultDenseExecutor, SharedCpuContext};

/// CPU linear-algebra provider selector (faer vs system BLAS/LAPACK), re-exported
/// from tenferro so runtimes can pick a backend via
/// [`DefaultDenseExecutor::with_kind`] without depending on tenferro directly.
#[cfg(feature = "tenferro")]
pub use tenferro_cpu::CpuBackendKind;

#[cfg(feature = "cuda")]
pub use cuda_adapter::{
    cuda_eigh_region, cuda_gemm_region_into, cuda_matmul_region_into, cuda_qr_region,
    cuda_svd_region, CudaDenseContext, CudaDenseStorage,
};
