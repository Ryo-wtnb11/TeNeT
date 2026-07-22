#![deny(unsafe_code)]

//! Symmetry-free execution layer for TeNeT (the TensorOperations.jl role):
//! scalar contracts, strided kernels and the host kernel adapter, scratch and
//! workspace reuse, and the replay-facing error and profile types. Nothing in
//! this crate consumes fusion rules or enumerates fusion trees; the symmetric
//! compile layer (`tenet-tensors`) hands down already-enumerated fusion-tree
//! pair identities and scalar coefficients, which are resolved to offsets and
//! strides before kernel execution.

#[cfg(not(any(
    feature = "cpu-faer",
    feature = "cpu-blas",
    feature = "blas-accelerate",
    feature = "blas-openblas",
    feature = "blas-mkl",
    feature = "provider-inject"
)))]
compile_error!(
    "tenet-operations requires a host execution backend; enable cpu-faer, cpu-blas, a blas-* provider, or provider-inject (cuda still requires host replay)"
);

pub mod axis;
#[cfg(feature = "cuda")]
pub mod cuda;
mod error;
pub mod fusion_replay;
mod host_scalar_kernels;
pub mod host_scratch;
pub mod kernel_adapter;
mod placement;
mod profile;
pub mod replay_backend;
mod scalar;
pub mod storage_scratch;
pub mod strided;
pub mod structure_identity;
pub mod tensoradd;
pub mod transform_helpers;
mod transform_key;
pub mod transform_plan;
pub mod transform_replay;
pub mod transform_structure;
mod tree_profile;

pub use axis::*;
pub use error::OperationError;
pub use fusion_replay::{
    direct_group_matrix_offset, fusion_scale_block_layouts_excluding, FusionBlockContractGroupPlan,
    FusionBlockContractPlan, FusionBlockContractWorkspace, FusionBlockMatrixGroup,
    FusionScaleBlockLayout, FusionStridedBlockLayout, FusionSubblockMatrixLayout,
    HostFusionBlockContractWorkspace, Rank2Gemm, StorageGemm,
};
pub use host_scalar_kernels::{
    axpby_raw_strided_kernel, axpby_raw_strided_kernel_trusted, copy_block_with_strided_kernel,
    scale_raw_strided_kernel_trusted, tensoradd_raw_strided_kernel,
    tensoradd_raw_strided_kernel_trusted, tensortrace_raw_strided_kernel,
    tensortrace_raw_strided_kernel_add_with_coefficient,
};
pub use kernel_adapter::{BakedFusedLayout, HostKernelAdapter, StridedHostKernelAdapter};
pub use placement::ReportsPlacement;
pub use profile::{TensorContractFusionProfile, TensorContractFusionRoute};
pub use replay_backend::*;
pub use scalar::{
    ConjugateValue, DenseBlockScalar, DenseRecouplingScalar, RealStructuralCoefficient,
    RecouplingCoefficientAction, TreeTransformScalar,
};
pub use tensoradd::*;
pub use transform_key::TreeTransformOperation;
pub use transform_plan::{
    TreeTransformBlockSpec, TreeTransformGroupBlockSpec, TreeTransformGroupPlan,
    TreeTransformKeyBlockSpec,
};
pub use transform_replay::*;
pub use transform_structure::*;
pub use tree_profile::TreeTransformReplayProfile;
// Why not retain crate-level `forbid`: Rust does not permit a narrower module
// to override it. `deny` keeps every other module unsafe-free while this one
// owns the audited Vec length transition.
mod owned_cat;
#[allow(unsafe_code)]
mod owned_overwrite_buffer;
mod owned_trace;
#[doc(hidden)]
pub use owned_cat::{
    try_cat_owned_c64_raw, try_cat_owned_raw, OwnedCatC64Source, OwnedCatCopy, OwnedCatSide,
};
#[doc(hidden)]
pub use owned_trace::{try_tensortrace_owned_raw, OwnedTraceTerm};
