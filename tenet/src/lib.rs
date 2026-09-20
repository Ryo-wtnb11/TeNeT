#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Public facade for TeNeT.
//!
//! # Execution model
//!
//! A [`prelude::TensorMap`] is a block-sparse symmetric tensor map stored as
//! TensorKit-equivalent reduced blocks indexed by fusion trees (one coupled
//! sector per block, column-major dense storage inside). Every op is dispatched
//! through the symmetry **rule provider** owned by its
//! [`prelude::GradedSpace`]s. The provider, scalar and storage are concrete type
//! parameters, while tensor rank and sector content remain runtime values. A
//! fusion rule defined outside this workspace therefore uses the same ordinary
//! tensor API as the built-in U(1), Z2, fZ2, SU(2), and product providers when
//! they satisfy the required admission and operation capability bounds.
//!
//! A [`prelude::Runtime`] owns the shared execution state: the per-rule
//! contraction/tree-transform contexts, the dense backend (selectable per
//! [`prelude::LinalgBackend`]), and the
//! contraction-plan cache the `tensor!` frontend keys by network topology.
//!
//! **Parallelism.** A `Runtime` is cheap to clone across threads. Standalone
//! operations normally lease independent per-rule contexts and, for
//! factorizations, dense executors instead of holding the Runtime's coarse state
//! mutex for the full operation. The `tensor!` path uses plan-local workspace
//! pools. Pool checkout and return, plan-cache and structural-store access, and
//! dense providers may still synchronize; an injected non-mintable executor
//! serializes factorization through the Runtime state lock. Device operations
//! take only a device-local mutex over the runtime's single CUDA context (and
//! the device backend's own handle and plan locks), never the state mutex, so
//! Host work is not blocked by device work; device operations still serialize
//! against each other there. Consequently no
//! general lock-free, overlap, or outer-thread scaling guarantee is made. See
//! `docs/backend_policy.md` for the ownership and synchronization model.
//!
#![doc = include_str!("tutorial.md")]

mod error;
pub mod plancache;
mod runtime;
mod tensor_core;
pub mod typed;

// Crate-root re-exports so the `default!` macro's `$crate::set_default_runtime`
// path resolves in user code (the `runtime` module itself is private).
#[doc(hidden)]
pub use runtime::RuntimeIdentity;
pub use runtime::{clear_default_runtime, default_runtime, set_default_runtime};
/// User-layer API: [`prelude::Runtime`], [`prelude::GradedSpace`], and
/// [`prelude::TensorMap`], plus the handful of expert-layer types their
/// signatures mention. `use tenet::prelude::*;` is the intended import for
/// everyday tensor code; expert APIs stay available through [`core`], [`dense`],
/// and the curated [`operations`] and [`matrixalgebra`] facades.
/// Checked-provider failures remain structured as
/// [`prelude::GenericTensorError`] variants whose nested errors retain their
/// standard [`std::error::Error::source`] chains.
///
/// ```
/// use std::error::Error;
/// use tenet::prelude::GenericTensorError;
///
/// fn is_plan<E>(error: &GenericTensorError<E>) -> bool {
///     matches!(error, GenericTensorError::Plan(_))
/// }
///
/// fn source<E: Error + 'static>(error: &GenericTensorError<E>) -> Option<&(dyn Error + 'static)> {
///     error.source()
/// }
/// ```
pub mod prelude {
    pub use crate::error::Error;
    #[cfg(feature = "cotengra-python")]
    pub use crate::plancache::CotengraSlicingConfig;
    #[cfg(feature = "cotengra-python")]
    pub use crate::plancache::{CotengraMinimize, CotengraPythonConfig, CotengraPythonMethod};
    pub use crate::plancache::{
        Optimizer, PlanCacheConfig, ReplanPolicy, DEFAULT_WORKSPACE_BUDGET_BYTES,
    };
    pub use crate::runtime::{
        LinalgBackend, Runtime, RuntimeBuilder, RuntimeTreeTransformCacheInfo,
    };
    pub use crate::typed::{
        CheckedGenericEigTrunc, CheckedGenericEighTrunc, DecodeError, DecodeLimits, EigTrunc,
        EighTrunc, EncodeError, GenericTensorError, GenericUnitTensorMapExt, GradedSpace,
        LegSelection, PhysicalDense, PhysicalDenseError, SectorSpectrum, SvdTrunc, TensorMap,
        TensorScalar, TruncatedSelection, TypedPersistenceCodec,
    };
    pub use num_complex::Complex64;
    #[allow(deprecated)]
    pub use tenet_core::FusionTreeBlockKey;
    pub use tenet_core::{
        product_fusion_rule, product_fusion_rule_with_codec, product_sector, CU1FusionRule,
        CU1Irrep, FermionParityFusionRule, FibonacciFusionRule, FibonacciSector, ProductFusionRule,
        ProductFusionRuleExt, ProductSector, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
        Z2FusionRule, Z2Irrep, ZNFusionRule, ZNIrrep,
    };
    pub use tenet_core::{BlockKey, FusionTreePairKey, MultiplicityIndex, SectorId};
    #[cfg(feature = "racah-generated")]
    pub use tenet_core::{SUNFusionRule, SUNFusionRuleError};
    pub use tenet_matrixalgebra::{Truncation, TruncationSpace};
}

/// Formula-first explanation of TeNeT's tensor-map convention, duals,
/// contractions, block layout, and weighted norms.
pub mod mathematics {
    #![doc = include_str!("mathematics.md")]
}

/// Expert layer: the structural data layer (sectors, fusion rules, fusion-tree
/// spaces, block layout, typed [`core::TensorMap`]). Re-export of `tenet-core`.
pub mod core {
    pub use tenet_core::*;
}

/// Expert layer: the dense block execution boundary (GEMM, transpose kernels).
/// Re-export of `tenet-dense`.
pub mod dense {
    pub use tenet_dense::*;
}

/// Low-level tensor operations for callers that manage output buffers or reuse
/// execution contexts.
///
/// Most applications should use [`prelude::TensorMap`] methods. Use this module
/// when a workflow needs explicit destination buffers, axis specifications, or
/// operation contexts. Documentation for each re-exported item explains buffer
/// ownership, validation, and how the operation changes its destination.
pub mod operations {
    pub use tenet_tensors::{
        braid_into, permute_into, tensoradd_into, tensorcontract_fusion_into, tensorcontract_into,
        tensortrace_into, transpose_into, BoundDynamicFusionMapSpace, DynamicFusionMapSpace,
        OperationError, OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
        TensorTraceAxisSpec, TreeTransformExecutionContext, TreeTransformOperation,
    };
}

/// Low-level compact SVD types and entry points.
///
/// Most applications should use the factorization methods on
/// [`prelude::TensorMap`]. Use this module when a workflow works directly with
/// [`core::TensorMap`] and supplies its own dense executor. Documentation for
/// each re-exported item explains who owns each input and returned value.
pub mod matrixalgebra {
    pub use tenet_matrixalgebra::{
        svd_compact, BoundTensorMap, BoundTensorMapRef, SectorSpectrum, SvdCompact,
    };
}
