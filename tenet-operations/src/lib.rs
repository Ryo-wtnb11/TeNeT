#![forbid(unsafe_code)]

//! Symmetry-free execution layer for TeNeT (the TensorOperations.jl role):
//! scalar contracts, strided kernels and the host kernel adapter, scratch and
//! workspace reuse, and the replay-facing error and profile types. Nothing in
//! this crate consumes fusion rules or enumerates fusion trees; the symmetric
//! compile layer (`tenet-tensors`) hands plans down as offsets, strides,
//! `SectorId`s, and scalar coefficients.

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
    direct_group_matrix_offset, fusion_scale_block_layouts_excluding,
    CanonicalFusionBlockContractGroupPlan, CanonicalFusionBlockContractPlan,
    CanonicalFusionBlockContractWorkspace, FusionBlockMatrixGroup, FusionScaleBlockLayout,
    FusionStridedBlockLayout, FusionSubblockMatrixLayout,
    HostCanonicalFusionBlockContractWorkspace, Rank2Gemm, StorageGemm,
};
pub use host_scalar_kernels::{
    axpby_raw_strided_kernel, axpby_raw_strided_kernel_trusted, copy_block_with_strided_kernel,
    scale_raw_strided_kernel_trusted, tensoradd_raw_strided_kernel,
    tensoradd_raw_strided_kernel_trusted, tensortrace_raw_strided_kernel,
    tensortrace_raw_strided_kernel_add_with_coefficient,
};
pub use kernel_adapter::{HostKernelAdapter, StridedHostKernelAdapter};
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
