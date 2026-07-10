#![forbid(unsafe_code)]

//! Matrix-free Krylov solvers.
//!
//! This v0 crate intentionally exposes only Conjugate Gradient over a small
//! vector-like abstraction. Scalars are real `f64`; complex vector
//! implementations can use `dot_real = Re(<x, y>)`.

mod cg;
mod operator;
mod vector;

pub use cg::{cg, CgBreakdown, CgOptions, CgResult, CgStats};
pub use operator::LinearOperator;
pub use vector::KrylovVector;
