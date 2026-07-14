#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Public facade for the active TeNeT rebuild.
//!
//! # Execution model
//!
//! A [`prelude::Tensor`] is a block-sparse symmetric tensor map stored as
//! TensorKit-equivalent reduced blocks indexed by fusion trees (one coupled
//! sector per block, column-major dense storage inside). Every op is dispatched
//! through a symmetry **rule provider** inherited from the tensor's
//! [`prelude::Space`]s (U(1), Z2, fZ2, SU(2), SU(3), and their products —
//! the core layer additionally supports anyonic rules such as Fibonacci), so the user
//! layer is rule-erased while the machinery stays fusion-tree-aware.
//!
//! A [`prelude::Runtime`] owns the shared execution state: the per-rule
//! contraction/tree-transform contexts, the dense backend (selectable per
//! [`prelude::LinalgBackend`] / [`prelude::TransposeBackend`]), and the
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
pub(crate) mod space;
pub(crate) mod tensor;

// Crate-root re-exports so the `default!` macro's `$crate::set_default_runtime`
// path resolves in user code (the `runtime` module itself is private).
pub use runtime::{clear_default_runtime, default_runtime, set_default_runtime};

/// User-layer API: [`prelude::Runtime`], [`prelude::Space`], and
/// [`prelude::Tensor`], plus the handful of expert-layer types their
/// signatures mention. `use tenet::prelude::*;` is the intended import for
/// everyday tensor code; the expert layer stays available through the
/// [`core`], [`operations`], [`dense`], and [`matrixalgebra`] modules.
pub mod prelude {
    pub use crate::default;
    pub use crate::error::Error;
    #[cfg(feature = "cotengra-python")]
    pub use crate::plancache::CotengraSlicingConfig;
    #[cfg(feature = "cotengra-python")]
    pub use crate::plancache::{CotengraMinimize, CotengraPythonConfig, CotengraPythonMethod};
    pub use crate::plancache::{Optimizer, PlanCacheConfig, ReplanPolicy};
    pub use crate::runtime::{
        clear_default_runtime, default_runtime, set_default_runtime, LinalgBackend, Runtime,
        RuntimeBuilder, TransposeBackend,
    };
    pub use crate::space::{SectorLabel, Space};
    pub use crate::tensor::{
        id, rand, rand_with_seed, zeros, ContractOverwriteCache, Dtype, EigTrunc, EighTrunc,
        OverwriteOutcome, PermuteOverwriteCache, Scalar, SvdTrunc, Tensor, TensorExecutionContext,
        TensorScalar,
    };
    pub use num_complex::Complex64;
    pub use tenet_core::{BlockKey, FusionTreeBlockKey, SectorId};
    pub use tenet_matrixalgebra::{SectorSpectrum, Truncation};
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

/// Expert layer: contraction / tree-transform / trace execution and the
/// context and cache types the [`prelude::Runtime`] wraps. Re-export of
/// `tenet-tensors`.
pub mod operations {
    pub use tenet_tensors::*;
}

/// Expert layer: factorizations and matrix functions the [`prelude::Tensor`]
/// decomposition methods pass through to. Re-export of `tenet-matrixalgebra`.
pub mod matrixalgebra {
    pub use tenet_matrixalgebra::*;
}
