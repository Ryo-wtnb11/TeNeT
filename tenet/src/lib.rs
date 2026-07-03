#![forbid(unsafe_code)]

//! Public facade for the active TeNeT rebuild.
//!
#![doc = include_str!("tutorial.md")]

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
