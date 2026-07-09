#![forbid(unsafe_code)]

//! Public facade for the active TeNeT rebuild.
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
    pub use crate::plancache::{Optimizer, PlanCacheConfig, ReplanPolicy};
    pub use crate::runtime::{
        clear_default_runtime, default_runtime, set_default_runtime, LinalgBackend, Runtime,
        RuntimeBuilder,
    };
    pub use crate::space::{SectorLabel, Space};
    pub use crate::tensor::{
        id, rand, rand_with_seed, zeros, Dtype, EigTrunc, EighTrunc, Scalar, SvdTrunc, Tensor,
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

pub mod core {
    pub use tenet_core::*;
}

pub mod dense {
    pub use tenet_dense::*;
}

pub mod operations {
    pub use tenet_tensors::*;
}

pub mod matrixalgebra {
    pub use tenet_matrixalgebra::*;
}
