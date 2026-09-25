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
// Pure host arithmetic: compiled (and tested) without the `cuda` feature for
// the same reason `cuda_region` is.
#[cfg(any(feature = "cuda", test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod cuda_hermitian;
// Layout metadata only: compiled (and tested) without the `cuda` feature so
// CI, which merely `cargo check`s that feature, still executes these rules.
#[cfg(any(feature = "cuda", test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod cuda_region;
// Pure host arithmetic: compiled (and tested) without the `cuda` feature.
#[cfg(any(feature = "cuda", test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod plan_ledger;
#[cfg(feature = "tenferro")]
mod tenferro_adapter;
#[cfg(test)]
mod tests;

pub use dot::DenseDotConfig;
pub use dtype::{DenseBackend, DenseDType, DensePlacement};
pub use error::DenseError;
pub use executor::{
    strided_batch_runs, strided_batch_runs_into, DenseExecutor, DenseFactorization,
    DenseGemmBatchJob, DenseLinalgScopeBody, DenseOwned, MatrixOp,
};
pub use scalar::DenseScalar;
pub use tensor::DenseTensor;
pub use view::{DenseRead, DenseView, DenseViewMut, DenseWrite};

#[cfg(feature = "tenferro")]
pub use tenferro_adapter::{
    cpu_session_stats, reset_cpu_session_stats, CpuSessionStats, DefaultDenseExecutor,
    SharedCpuContext,
};
#[cfg(all(
    test,
    any(feature = "cpu-faer", feature = "cpu-blas-core"),
    not(feature = "provider-inject")
))]
pub(crate) use tenferro_adapter::{
    owned_full_svd_input_pointers, reset_owned_full_svd_input_pointers,
};

/// CPU linear-algebra provider selector (faer vs system BLAS/LAPACK), re-exported
/// from tenferro so runtimes can pick a backend via
/// [`DefaultDenseExecutor::with_kind`] without depending on tenferro directly.
#[cfg(feature = "tenferro")]
pub use tenferro_cpu::CpuBackendKind;

#[cfg(feature = "cuda")]
pub use cuda_adapter::{
    cuda_copy_region_into, cuda_eigh_region, cuda_gemm_region_into, cuda_gemm_region_with_ops_into,
    cuda_hermitian_regions, cuda_is_hermitian_region, cuda_matmul_region_into, cuda_qr_region,
    cuda_region_axpby, cuda_region_trace_accumulate, cuda_region_zero, cuda_svd_region,
    cuda_transfer_stats, cuda_widen, reset_cuda_transfer_stats, CudaDenseContext, CudaDenseStorage,
    CudaPlanCacheStats, CudaRealScalar, CudaRegionBeta, CudaRegionCoefficient, CudaScalar,
    CudaTransferStats,
};
#[cfg(feature = "cuda")]
pub use cuda_adapter::{cuda_download_spectra, CudaSpectrum};
#[cfg(feature = "cuda")]
pub use cuda_region::CudaRegion;
#[cfg(feature = "cuda")]
pub use plan_ledger::{
    plan_cache_entries_for, CUTENSOR_PLAN_BYTES, DEFAULT_PLAN_CACHE_BUDGET_BYTES,
};
