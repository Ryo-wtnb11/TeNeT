#![forbid(unsafe_code)]

//! Public facade for the active TeNeT rebuild.
//!
#![doc = include_str!("tutorial.md")]

mod error;
mod runtime;
pub(crate) mod space;
pub(crate) mod tensor;

/// User-layer API: [`prelude::Runtime`], [`prelude::Space`], and
/// [`prelude::Tensor`], plus the handful of expert-layer types their
/// signatures mention. `use tenet::prelude::*;` is the intended import for
/// everyday tensor code; the expert layer stays available through the
/// [`core`], [`operations`], [`dense`], and [`matrixalgebra`] modules.
pub mod prelude {
    pub use crate::error::Error;
    pub use crate::runtime::{Runtime, RuntimeBuilder};
    pub use crate::space::Space;
    pub use crate::tensor::{EighTrunc, SvdTrunc, Tensor};
    pub use tenet_core::{BlockKey, FusionTreeBlockKey, SectorId};
    pub use tenet_matrixalgebra::{SectorSpectrum, Truncation};
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
