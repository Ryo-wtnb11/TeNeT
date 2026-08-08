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
//! tensor API as the built-in U(1), Z2, fZ2, SU(2), and product providers.
//!
//! A [`prelude::Runtime`] owns the shared execution state: the per-rule
//! contraction/tree-transform contexts, the dense backend (selectable per
//! [`prelude::LinalgBackend`]), and the
//! contraction-plan cache the `tensor!` frontend keys by network topology.
//!
//! **Parallelism.** Ops on a shared `Runtime` scale with outer threads: each
//! standalone op leases a per-rule context (and, for factorizations, a dense
//! executor) from a pool for its own duration and runs lock-free, and the
//! `tensor!` cached-plan path holds only its own plan-cache mutex. A `Runtime`
//! is therefore cheap to `clone` across threads; the one path that still
//! serializes is a custom executor injected via
//! [`prelude::RuntimeBuilder::with_dense_executor`]. See `docs/backend_policy.md`
//! for the pool design and measured scaling.
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
        CheckedGenericEigTrunc, CheckedGenericEighTrunc, EigTrunc, EighTrunc,
        GenericUnitTensorMapExt, GradedSpace, SvdTrunc, TensorMap, TensorScalar,
    };
    pub use num_complex::Complex64;
    #[allow(deprecated)]
    pub use tenet_core::FusionTreeBlockKey;
    pub use tenet_core::{
        product_fusion_rule, product_fusion_rule_with_codec, product_sector, CU1FusionRule,
        CU1Irrep, FermionParityFusionRule, FibonacciFusionRule, ProductFusionRule,
        ProductFusionRuleExt, ProductSector, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
        Z2FusionRule, Z2Irrep, ZNFusionRule, ZNIrrep,
    };
    pub use tenet_core::{BlockKey, FusionTreePairKey, MultiplicityIndex, SectorId};
    #[cfg(feature = "racah-generated")]
    pub use tenet_core::{SUNFusionRule, SUNFusionRuleError};
    pub use tenet_matrixalgebra::{SectorSpectrum, Truncation, TruncationSpace};
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

/// Curated expert operations and context types used by documented workflows.
pub mod operations {
    pub use tenet_tensors::{
        braid_into, permute_into, tensoradd_into, tensorcontract_fusion_into, tensorcontract_into,
        tensortrace_into, transpose_into, BoundDynamicFusionMapSpace, DynamicFusionMapSpace,
        OperationError, OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
        TensorTraceAxisSpec, TreeTransformExecutionContext, TreeTransformOperation,
    };
}

/// Curated factorization types and entry points.
pub mod matrixalgebra {
    pub use tenet_matrixalgebra::{
        svd_compact, BoundTensorMap, BoundTensorMapRef, SectorSpectrum, SvdCompact,
    };
}
