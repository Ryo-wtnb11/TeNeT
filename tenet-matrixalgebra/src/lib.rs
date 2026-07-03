#![forbid(unsafe_code)]

//! MatrixAlgebraKit-style factorizations for TeNeT fusion tensors.
//!
//! Crate mapping onto the TensorKit ecosystem: `tenet-core` +
//! `tenet-operations` together play TensorKit.jl's role (structures and
//! symmetric execution, including the fusion-tree transforms); strided-rs and
//! tenferro play Strided.jl / TensorOperations.jl's symmetry-agnostic dense
//! role underneath; and this crate is the MatrixAlgebraKit layer applied at
//! the fusion-tensor level — blockwise factorizations over the coupled-sector
//! matricization, spectrum truncation, and the derived matrix functions.
//!
//! Every operation decomposes into three kinds of work: dense factorizations
//! and GEMM on the device boundary ([`tenet_dense::DenseExecutor`]),
//! scalar decisions over per-sector spectra on the host
//! ([`truncation`], spectrum functions), and mechanical block-data movement
//! (bond slicing, adjoints) that stays behind device-capable seams.

mod compose;
mod factorize;
mod matrix_functions;
pub mod truncation;

pub use factorize::{
    eig_full, eig_trunc, eig_vals, eigh_full, eigh_trunc, eigh_vals, left_null, left_polar,
    lq_compact, lq_full, qr_compact, qr_full, right_null, right_polar, svd_compact, svd_full,
    svd_trunc, svd_vals, EigFull, EigTrunc, EighFull, EighTrunc, FactorScalar, SectorSpectrum,
    SpectrumMagnitude, SvdCompact, SvdFull, SvdTrunc,
};
pub use matrix_functions::{exp, inv, pinv};
pub use truncation::{select_truncation, Truncation, TruncationDecision, WeightedSpectrum};

#[cfg(test)]
mod tests;
