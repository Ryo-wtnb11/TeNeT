//! Export firewall for tenet-matrixalgebra (#1803).
//!
//! The documented crate root carries the spectrum and result types; every
//! entry point the `tenet` facade dispatches to sits in the doc-hidden
//! `seam` module. This test pins the exact names both export (all feature
//! combinations at once), so adding or removing one fails here and is
//! reviewed as an API change.
//!
//! Why a source scan of `lib.rs` and not an import list: an import list only
//! proves presence, not absence. `lib.rs` is the only file that re-exports
//! from the private modules.

use std::collections::BTreeSet;

/// Leaf names of every `pub use` and `pub mod` in `source`, split into the
/// crate root and the `seam` module.
fn exports(source: &str) -> (BTreeSet<String>, BTreeSet<String>) {
    let code: String = source
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    let seam_start = code
        .find("pub mod seam {")
        .expect("lib.rs declares `pub mod seam`");
    let seam_len = code[seam_start..]
        .find("\n}")
        .expect("`seam` closes at column 0");
    let seam_body = &code[seam_start + "pub mod seam {".len()..seam_start + seam_len];
    let root_body = format!("{}{}", &code[..seam_start], &code[seam_start + seam_len..]);
    let mut root = leaves(&root_body);
    root.insert("seam".to_string());
    (root, leaves(seam_body))
}

fn leaves(code: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for statement in code.split(';') {
        let statement = statement.trim();
        let statement = statement
            .rsplit_once(']')
            .filter(|_| statement.starts_with("#["))
            .map_or(statement, |(_, rest)| rest.trim());
        if let Some(name) = statement.strip_prefix("pub mod ") {
            out.insert(name.trim().to_string());
        } else if let Some(path) = statement.strip_prefix("pub use ") {
            let list = path
                .find('{')
                .map_or(path.rsplit("::").next().unwrap(), |open| {
                    path[open + 1..].trim_end_matches('}')
                });
            out.extend(
                list.split(',')
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(str::to_string),
            );
        }
    }
    out
}

fn names(list: &str) -> BTreeSet<String> {
    list.split_whitespace().map(str::to_string).collect()
}

#[test]
fn crate_root_and_seam_export_exactly_the_consumed_items() {
    let source = include_str!("../src/lib.rs");
    let (root, seam) = exports(source);
    assert_eq!(
        root,
        names(
            "BoundDynFactor FactorScalar SectorSpectrum SpectrumMagnitude \
             Eig Eigh LeftPolar Lq Qr RightPolar Svd \
             select_truncation Truncation TruncationDecision TruncationError TruncationSpace \
             WeightedSpectrum truncation seam"
        )
    );
    assert_eq!(
        seam,
        names(
            "coupled_sector_block_dimensions_generic_checked decide_bond_truncation \
             decide_bond_truncation_generic_checked diagonal_bond_bound_space_generic \
             diagonal_bond_bound_space_generic_checked diagonal_bond_bound_space_like \
             diagonal_bond_data eig_full_diagonal_dyn eig_full_diagonal_dyn_checked_generic \
             eig_full_dyn eig_full_dyn_checked_generic eig_vals_diagonal_dyn eig_vals_dyn \
             eig_vals_dyn_checked_generic eigh_full_diagonal_dyn \
             eigh_full_diagonal_dyn_checked_generic eigh_full_dyn eigh_full_dyn_checked_generic \
             eigh_vals_diagonal_dyn eigh_vals_dyn eigh_vals_dyn_checked_generic \
             left_null_diagonal_dyn left_null_diagonal_dyn_checked_generic left_null_dyn \
             left_null_dyn_checked_generic_with_dimensions left_polar_adjoint_parent_dyn \
             left_polar_adjoint_parent_dyn_checked_generic left_polar_diagonal_spectra_dyn \
             left_polar_diagonal_spectra_dyn_checked_generic left_polar_dyn \
             left_polar_dyn_checked_generic lq_compact_dyn lq_compact_dyn_checked_generic \
             lq_compact_dyn_generic lq_diagonal_dyn_checked_generic lq_full_dyn \
             lq_full_dyn_checked_generic qr_compact_dyn qr_compact_dyn_checked_generic \
             qr_compact_dyn_generic qr_diagonal_dyn qr_diagonal_dyn_checked_generic qr_full_dyn \
             qr_full_dyn_checked_generic rectangular_diagonal_bond_tensor \
             rectangular_diagonal_bond_tensor_generic_checked right_null_diagonal_dyn \
             right_null_diagonal_dyn_checked_generic right_null_dyn \
             right_null_dyn_checked_generic_with_dimensions right_polar_adjoint_parent_dyn \
             right_polar_adjoint_parent_dyn_checked_generic right_polar_diagonal_spectra_dyn \
             right_polar_diagonal_spectra_dyn_checked_generic right_polar_dyn \
             right_polar_dyn_checked_generic scale_axis_by_spectrum_mapped \
             svd_compact_adjoint_factors_dyn svd_compact_diagonal_factors_dyn \
             svd_compact_diagonal_factors_dyn_checked_generic svd_compact_dyn_checked_generic \
             svd_compact_factors_dyn svd_compact_factors_dyn_generic \
             svd_compact_factors_with_spectrum_dyn_checked_generic svd_full_adjoint_factors_dyn \
             svd_full_diagonal_factors_dyn_checked_generic svd_full_factors_dyn \
             svd_full_factors_dyn_checked_generic \
             svd_full_factors_dyn_checked_generic_with_dimensions svd_vals_compact_diagonal_dyn \
             svd_vals_dyn svd_vals_dyn_checked_generic svd_vals_dyn_generic \
             validate_endomorphism_region_stacking validate_hermitian_regions \
             BoundDynamicTensorRef CheckedCompactPolarFactors CheckedDiagonalFullSvdFactors \
             CheckedDiagonalNullFactor CheckedFullSvdDimensions CheckedGenericFactorPlanError \
             EigFullDyn EighFullDyn EIGH_FULL_STACKING SvdFactorsDyn SvdFullFactorsDyn \
             sector_matricization_diagnostic SectorMatricizationDiagnostic \
             exp_dyn exp_pade13_direct_into_dyn inv_direct_dyn inv_direct_into_dyn \
             pinv_adjoint_parent_dyn pinv_direct_into_dyn pinv_dyn solve_left_direct_dyn \
             solve_left_direct_into_dyn rescaled_power_norm"
        )
    );
}
