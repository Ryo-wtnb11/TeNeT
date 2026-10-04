#![cfg_attr(not(test), forbid(unsafe_code))]
// Tests only: the counting allocator in `tests::allocation_bytes` is the one
// exemption; `forbid` could not be relaxed for it.
#![cfg_attr(test, deny(unsafe_code))]

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
mod results;
pub mod truncation;

pub use factorize::{BoundDynFactor, FactorScalar, SectorSpectrum, SpectrumMagnitude};
pub use results::{Eig, Eigh, LeftPolar, Lq, Qr, RightPolar, Svd};
pub use truncation::{
    select_truncation, Truncation, TruncationDecision, TruncationError, TruncationSpace,
    WeightedSpectrum,
};

/// The dynamic-rank entry points the `tenet` facade dispatches to. Not a
/// stable API: every item here exists because a `tenet` module, example or
/// this crate's own integration tests and examples call it, and the export
/// firewall test (`tests/exports.rs`) pins the exact list.
#[doc(hidden)]
pub mod seam {
    pub use crate::factorize::{
        coupled_sector_block_dimensions_generic_checked, decide_bond_truncation,
        decide_bond_truncation_generic_checked, diagonal_bond_bound_space_generic_checked,
        diagonal_bond_bound_space_like, diagonal_bond_bound_space_on_source_checked_generic,
        diagonal_bond_data, eig_full_checked_generic, eig_full_diagonal_dyn, eig_full_dyn,
        eig_vals_dyn, eig_vals_from_source, eigh_full_checked_generic, eigh_full_diagonal_dyn,
        eigh_full_dyn, eigh_vals_dyn, eigh_vals_from_source, factor_isomorphic_checked_generic,
        factor_output_space_checked_generic, left_null_checked_generic, left_null_diagonal_dyn,
        left_null_dyn, left_polar_adjoint_parent_dyn,
        left_polar_adjoint_parent_dyn_checked_generic, left_polar_checked_generic,
        left_polar_diagonal_spectra_dyn, left_polar_dyn, lq_compact_checked_generic,
        lq_compact_dyn, lq_diagonal_dyn, lq_full_checked_generic, lq_full_dyn,
        qr_compact_checked_generic, qr_compact_dyn, qr_diagonal_dyn, qr_full_checked_generic,
        qr_full_dyn, rectangular_diagonal_bond_tensor,
        rectangular_diagonal_bond_tensor_generic_checked, right_null_checked_generic,
        right_null_diagonal_dyn, right_null_dyn, right_polar_adjoint_parent_dyn,
        right_polar_adjoint_parent_dyn_checked_generic, right_polar_checked_generic,
        right_polar_diagonal_spectra_dyn, right_polar_dyn, scale_axis_by_spectrum_mapped,
        svd_compact_adjoint_factors_dyn, svd_compact_checked_generic,
        svd_compact_diagonal_factors_dyn, svd_compact_dyn_checked_generic, svd_compact_factors_dyn,
        svd_full_adjoint_factors_dyn, svd_full_checked_generic, svd_full_factors_dyn, svd_vals_dyn,
        svd_vals_from_source, validate_endomorphism_region_stacking, validate_hermitian_regions,
        BoundDynamicTensorRef, CheckedGenericFactorPlanError, EigFullDyn, EighFullDyn,
        ExecutorLease, FactorMode, FactorOutput, FactorRoute, FactorSource, Routed, SvdFactorsDyn,
        SvdFullFactorsDyn, EIGH_FULL_STACKING,
    };
    #[cfg(feature = "diagnostics")]
    pub use crate::factorize::{sector_matricization_diagnostic, SectorMatricizationDiagnostic};
    pub use crate::matrix_functions::{
        exp_dyn, exp_pade13_direct_into_dyn, inv_direct_dyn, inv_direct_into_dyn,
        pinv_adjoint_parent_dyn, pinv_direct_into_dyn, pinv_dyn, solve_left_direct_dyn,
        solve_left_direct_into_dyn,
    };
    pub use crate::truncation::rescaled_power_norm;
}

// The unit tests drive the typed const-generic wrappers, which have no
// consumer outside them (nor does `scale_axis_by_spectrum` outside this
// crate), and reach the seam through `use crate::*`.
#[cfg(test)]
use factorize::{
    eig_full, eig_vals, eigh_full, eigh_vals, left_null, left_null_dyn_checked_generic, left_polar,
    lq_compact, lq_full, qr_compact, qr_full, right_null, right_null_dyn_checked_generic,
    right_polar, scale_axis_by_spectrum, svd_compact, svd_compact_dyn, svd_full,
    svd_full_adjoint_dyn, svd_full_dyn, svd_full_dyn_checked_generic, svd_vals,
};
#[cfg(test)]
use factorize::{
    eig_full_dyn_checked_generic, eigh_full_dyn_checked_generic, left_polar_dyn_checked_generic,
    lq_compact_dyn_checked_generic, lq_full_dyn_checked_generic, qr_compact_dyn_checked_generic,
    qr_full_dyn_checked_generic, right_polar_dyn_checked_generic,
    svd_compact_factors_with_spectrum_dyn_checked_generic,
    svd_full_diagonal_factors_dyn_checked_generic,
};
#[cfg(test)]
use matrix_functions::{exp, inv, inv_dyn, pinv};
#[cfg(test)]
use seam::*;

#[cfg(test)]
mod tests;

/// The workspace tolerance rule for arithmetic test comparisons
/// (`docs/testing_numerics.md`).
#[cfg(test)]
#[path = "../../tests/support"]
mod test_numerics {
    use num_complex::{Complex32, Complex64};
    pub(crate) mod numerics;
}
