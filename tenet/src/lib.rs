#![forbid(unsafe_code)]

//! Public facade for the active TeNeT rebuild.

pub mod dense {
    pub use tenet_dense::*;
}

pub mod strided {
    pub use tenet_strided::*;
}
