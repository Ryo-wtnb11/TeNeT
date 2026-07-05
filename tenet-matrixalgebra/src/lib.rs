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
    eig_full, eig_full_dyn, eig_trunc, eig_trunc_dyn, eig_vals, eig_vals_dyn, eigh_full,
    eigh_full_dyn, eigh_trunc, eigh_trunc_dyn, eigh_vals, eigh_vals_dyn, left_null, left_null_dyn,
    left_orth, left_orth_dyn, left_polar, left_polar_dyn, lq_compact, lq_compact_dyn, lq_full,
    lq_full_dyn, qr_compact, qr_compact_dyn, qr_full, qr_full_dyn, right_null, right_null_dyn,
    right_orth, right_orth_dyn, right_polar, right_polar_dyn, svd_compact, svd_compact_dyn,
    svd_full, svd_full_dyn, svd_trunc, svd_trunc_dyn, svd_vals, svd_vals_dyn, DynFactor, EigFull,
    EigFullDyn, EigTrunc, EigTruncDyn, EighFull, EighFullDyn, EighTrunc, EighTruncDyn,
    FactorScalar, SectorSpectrum, SpectrumMagnitude, SvdCompact, SvdCompactDyn, SvdFull,
    SvdFullDyn, SvdTrunc, SvdTruncDyn,
};
pub use matrix_functions::{exp, exp_dyn, inv, inv_dyn, pinv, pinv_dyn};
pub use truncation::{select_truncation, Truncation, TruncationDecision, WeightedSpectrum};

#[cfg(feature = "diagnostics")]
#[doc(hidden)]
pub use factorize::{sector_matricization_diagnostic, SectorMatricizationDiagnostic};

#[cfg(test)]
mod tests;
