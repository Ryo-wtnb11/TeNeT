use std::collections::BTreeMap;

use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;
use std::sync::Arc;

#[cfg(test)]
use std::cell::{Cell, RefCell};

use num_complex::Complex64;
use num_traits::{Float, Zero};
use tenet_core::{
    BlockKey, BlockRef, BlockStructure, CheckedGenericFusion, CheckedGenericRigidSymbols,
    CheckedGenericStructureError, CoreError, CoupledSectorRegion, CoupledTreeExtent,
    FusionProductSpace, FusionRule, FusionTensorMapSpace, FusionTreeHomSpace, FusionTreeKey,
    FusionTreePairKey, InfallibleGeneric, MultiplicityFreeRigidSymbols, SectorId, SectorLeg,
    SectorStructure, TensorMap, TensorMapSpace,
};
use tenet_dense::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseFactorization, DenseOwned,
    DenseTensor, DenseView, DenseViewMut,
};

pub use tenet_tensors::BoundDynamicTensorRef;
use tenet_tensors::{
    BoundDynamicFusionMapSpace, DenseBlockScalar, DenseRecouplingScalar, DynamicFusionMapSpace,
    PreparedCheckedGenericDynamicSpace, ValidatedDynamicFusionLayout,
};

use crate::results::{LeftPolar, Lq, Qr, RightPolar, Svd};
use crate::truncation::{select_truncation, Truncation, WeightedSpectrum};
use tenet_tensors::OperationError;

/// Binds `$geometry` to the admitted input's per-sector geometry
/// (`&[CoupledSectorRegion]` or `&[SectorMatricization<D>]`) for the
/// `SectorGeometry`-generic publication helpers.
macro_rules! with_input_geometry {
    ($input:expr, |$geometry:ident| $body:expr) => {
        match $input {
            InputMatricizations::Regions { regions, .. } => {
                let $geometry: &[CoupledSectorRegion] = regions;
                $body
            }
            InputMatricizations::Packed(matrices) => {
                let $geometry = matrices.as_slice();
                $body
            }
        }
    };
}

mod eig;
mod inverse;
mod null_space;
mod polar;
mod qr_lq;
mod region;
mod svd;

#[cfg(test)]
mod numerical_null_tests;

#[cfg(test)]
mod hermitian_scale_tests;

#[cfg(test)]
mod sector_matricization_tests;

// Blanket re-exports so every existing `crate::factorize::<name>` path
// (used by `compose.rs`, `matrix_functions.rs`, and `lib.rs`'s own
// `pub use factorize::{...}`) keeps resolving unchanged after the move.
// Each child's own `pub`/`pub(crate)` items pass straight through; each
// child's `pub(super)` items (bumped from fully-private during the move)
// become reachable here and, by re-export, throughout the crate.
// `null_space` and `polar` have no crate-internal consumer outside the
// explicit `pub use` blocks below (every item another file needs is
// already named there), so a blanket glob re-export of either would be
// unused.
pub(crate) use eig::*;
pub(crate) use inverse::*;
pub(crate) use qr_lq::*;
pub(crate) use region::*;
pub(crate) use svd::*;

// Explicit overrides: an item re-exported through the blanket globs above
// resolves at `pub(crate)` (the glob's own visibility), even when the item
// is itself `pub` inside its child module -- Rust's re-export visibility is
// the minimum of the two. `lib.rs`'s own `pub use factorize::{...}` needs
// these specific names to be fully `pub` here, so each one is reimported by
// name; an explicit import always shadows the same name brought in by a
// glob.
pub use eig::{
    eig_full, eig_full_diagonal_dyn, eig_full_dyn, eig_full_dyn_checked_generic, eig_vals,
    eig_vals_diagonal_dyn, eig_vals_dyn, eig_vals_dyn_checked_generic, eigh_full,
    eigh_full_diagonal_dyn, eigh_full_diagonal_dyn_checked_generic, eigh_full_dyn,
    eigh_full_dyn_checked_generic, eigh_vals, eigh_vals_diagonal_dyn, eigh_vals_dyn,
    eigh_vals_dyn_checked_generic, validate_hermitian_regions, EigFull, EigFullDyn, EighFull,
    EighFullDyn,
};
pub use null_space::{
    left_null, left_null_diagonal_dyn, left_null_dyn, left_null_dyn_checked_generic, right_null,
    right_null_diagonal_dyn, right_null_dyn, right_null_dyn_checked_generic,
};
pub use polar::{
    left_polar, left_polar_adjoint_parent_dyn, left_polar_adjoint_parent_dyn_checked_generic,
    left_polar_diagonal_dyn, left_polar_diagonal_spectra_dyn, left_polar_dyn,
    left_polar_dyn_checked_generic, right_polar, right_polar_adjoint_parent_dyn,
    right_polar_adjoint_parent_dyn_checked_generic, right_polar_diagonal_dyn,
    right_polar_diagonal_spectra_dyn, right_polar_dyn, right_polar_dyn_checked_generic,
};
use qr_lq::diagonal_phase_magnitude;
pub use qr_lq::{
    lq_compact, lq_compact_dyn, lq_compact_dyn_checked_generic, lq_compact_dyn_generic, lq_full,
    lq_full_dyn, lq_full_dyn_checked_generic, qr_compact, qr_compact_dyn,
    qr_compact_dyn_checked_generic, qr_compact_dyn_generic, qr_diagonal_dyn, qr_full, qr_full_dyn,
    qr_full_dyn_checked_generic,
};
pub use region::{
    build_bound_factor_space_generic_checked, coupled_sector_block_dimensions_generic_checked,
    validate_endomorphism_region_stacking, BoundDynFactor, BoundTensorMap, BoundTensorMapRef,
    CheckedGenericFactorPlanError, FactorScalar, SectorSpectrum, SpectrumMagnitude,
    EIGH_FULL_STACKING,
};
#[cfg(feature = "diagnostics")]
pub use region::{sector_matricization_diagnostic, SectorMatricizationDiagnostic};
pub use svd::{
    decide_bond_truncation, decide_bond_truncation_generic_checked, diagonal_bond_bound_space,
    diagonal_bond_bound_space_generic, diagonal_bond_bound_space_generic_checked,
    diagonal_bond_bound_space_like, diagonal_bond_data, diagonal_bond_svd_factor_generic_checked,
    rectangular_diagonal_bond_tensor, rectangular_diagonal_bond_tensor_generic_checked,
    scale_axis_by_spectrum, scale_axis_by_spectrum_mapped, svd_compact,
    svd_compact_adjoint_factors_dyn, svd_compact_diagonal_factors_dyn,
    svd_compact_diagonal_factors_dyn_checked_generic, svd_compact_dyn,
    svd_compact_dyn_checked_generic, svd_compact_factors_dyn, svd_compact_factors_dyn_generic,
    svd_compact_factors_with_spectrum_dyn_checked_generic, svd_full, svd_full_adjoint_dyn,
    svd_full_adjoint_factors_dyn, svd_full_diagonal_factors_dyn_checked_generic, svd_full_dyn,
    svd_full_dyn_checked_generic, svd_full_factors_dyn, svd_full_factors_dyn_checked_generic,
    svd_full_factors_dyn_checked_generic_with_dimensions, svd_vals, svd_vals_compact_diagonal_dyn,
    svd_vals_dyn, svd_vals_dyn_checked_generic, svd_vals_dyn_generic,
    CheckedDiagonalFullSvdFactors, CheckedFullSvdDimensions, SvdCompact, SvdCompactDyn,
    SvdFactorsDyn, SvdFull, SvdFullDyn, SvdFullFactorsDyn,
};
