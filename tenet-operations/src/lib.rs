#![forbid(unsafe_code)]

//! Symmetry-free execution layer for TeNeT (the TensorOperations.jl role):
//! scalar contracts, strided kernels and the host kernel adapter, scratch and
//! workspace reuse, and the replay-facing error and profile types. Nothing in
//! this crate consumes fusion rules or enumerates fusion trees; the symmetric
//! compile layer (`tenet-tensors`) hands plans down as offsets, strides,
//! `SectorId`s, and scalar coefficients.

mod error;
mod host_scalar_kernels;
pub mod host_scratch;
pub mod kernel_adapter;
mod placement;
mod scalar;
pub mod storage_scratch;
pub mod strided;
mod transform_key;
mod tree_profile;

pub use error::OperationError;
pub use host_scalar_kernels::{
    axpby_raw_strided_kernel, axpby_raw_strided_kernel_trusted, copy_block_with_strided_kernel,
    scale_raw_strided_kernel_trusted, tensoradd_raw_strided_kernel,
    tensoradd_raw_strided_kernel_trusted, tensortrace_raw_strided_kernel,
    tensortrace_raw_strided_kernel_add_with_coefficient,
};
pub use kernel_adapter::{HostKernelAdapter, StridedHostKernelAdapter};
pub use placement::ReportsPlacement;
pub use scalar::{
    ConjugateValue, DenseBlockScalar, DenseRecouplingScalar, RealStructuralCoefficient,
    RecouplingCoefficientAction, TreeTransformScalar,
};
pub use transform_key::TreeTransformOperationKey;
pub use tree_profile::TreeTransformReplayProfile;
