//! Export firewall for tenet-matrixalgebra (#1803).
//!
//! The documented crate root carries the spectrum, scalar, factor and result
//! types; every entry point the `tenet` facade dispatches to sits in the
//! doc-hidden `seam` module. This test pins the exact names both export,
//! under every feature combination at once, so adding or removing one fails
//! here and is reviewed as an API change. The typed const-generic layer
//! (`svd_compact`, `BoundTensorMap`, ...) and the dense compact-diagonal
//! polar seam were removed because nothing outside this crate's tests
//! called them.
//!
//! Why a source scan and not an import list: an import list only proves
//! presence, not absence. Modelled on `tenet-network/tests/network_exports.rs`.

use std::collections::BTreeSet;

/// Drops comments (line, doc and `/* */`) and blanks string literals so that
/// neither can hide or fake an item.
fn strip(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(ch) = rest.chars().next() {
        if rest.starts_with("//") {
            rest = &rest[rest.find('\n').unwrap_or(rest.len())..];
        } else if rest.starts_with("/*") {
            rest = &rest[rest.find("*/").map_or(rest.len(), |end| end + 2)..];
            out.push(' ');
        } else if ch == '"' {
            let mut escaped = false;
            let end = rest[1..]
                .find(|c: char| {
                    let close = c == '"' && !escaped;
                    escaped = c == '\\' && !escaped;
                    close
                })
                .map_or(rest.len(), |end| end + 2);
            rest = &rest[end..];
            out.push_str("\"\"");
        } else {
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
    out
}

/// Names exported by `code`: `pub use` leaves and `pub mod <name>;` module
/// names. Any other inline public item (`pub fn`, `pub struct`, an inline
/// `pub mod` body, ...) or `#[macro_export]` is an error, because this
/// firewall would otherwise not see what it exports.
fn exported_names(code: &str) -> Result<BTreeSet<String>, String> {
    if code.contains("macro_export") {
        return Err("lib.rs exports a macro_rules! macro".into());
    }
    let mut names = BTreeSet::new();
    for (at, _) in code.match_indices("pub") {
        let before = code[..at].chars().next_back();
        if before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let after = &code[at + 3..];
        if after.starts_with('(') || after.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            continue; // `pub(crate)` etc., or an identifier such as `public`
        }
        let item = after.trim_start();
        if let Some(module) = item
            .strip_prefix("mod")
            .filter(|t| t.starts_with(char::is_whitespace))
        {
            let end = module.find([';', '{']);
            match end.map(|end| (module[..end].trim(), &module[end..end + 1])) {
                Some((name, ";")) => {
                    names.insert(name.to_owned());
                    continue;
                }
                _ => return Err("lib.rs gained an inline `pub mod` body".into()),
            }
        }
        let Some(tree) = item
            .strip_prefix("use")
            .filter(|t| t.starts_with(char::is_whitespace))
        else {
            let keyword = item.split_whitespace().next().unwrap_or("");
            return Err(format!("lib.rs gained an inline `pub {keyword}` item"));
        };
        let tree = tree[..tree.find(';').ok_or("unterminated pub use")?].trim();
        let leaves = match tree.find('{') {
            Some(open) => {
                let inner = tree[open + 1..]
                    .trim_end()
                    .strip_suffix('}')
                    .ok_or("bad use tree")?;
                if inner.contains('{') {
                    return Err(format!("nested use tree: {tree}"));
                }
                inner.split(',').collect::<Vec<_>>()
            }
            None => vec![tree],
        };
        for leaf in leaves
            .iter()
            .map(|leaf| leaf.trim())
            .filter(|l| !l.is_empty())
        {
            let name = leaf.rsplit(" as ").next().unwrap();
            names.insert(name.rsplit("::").next().unwrap().trim().to_owned());
        }
    }
    Ok(names)
}

/// Root and `seam` exports of `source`: the one inline module allowed is
/// `pub mod seam { .. }`, whose body is scanned on its own and replaced by
/// `pub mod seam;` in the root scan.
fn exports(source: &str) -> Result<(BTreeSet<String>, BTreeSet<String>), String> {
    const OPEN: &str = "pub mod seam {";
    let code = strip(source);
    let start = code.find(OPEN).ok_or("lib.rs declares no `pub mod seam`")?;
    let body_start = start + OPEN.len();
    let mut depth = 1;
    let body_len = code[body_start..]
        .find(|c: char| {
            depth += match c {
                '{' => 1,
                '}' => -1,
                _ => 0,
            };
            depth == 0
        })
        .ok_or("unterminated `pub mod seam`")?;
    let root = format!(
        "{}pub mod seam;{}",
        &code[..start],
        &code[body_start + body_len + 1..]
    );
    Ok((
        exported_names(&root)?,
        exported_names(&code[body_start..body_start + body_len])?,
    ))
}

fn names(list: &str) -> BTreeSet<String> {
    list.split_whitespace().map(str::to_string).collect()
}

#[test]
fn firewall_rejects_every_inline_public_item() {
    for item in [
        "pub mod m {}",
        "pub fn f() {}",
        "pub struct S;",
        "pub enum E {}",
        "pub trait T {}",
        "pub type A = u8;",
        "pub const C: u8 = 0;",
        "pub static S: u8 = 0;",
        "pub union U { a: u8 }",
        "pub unsafe fn f() {}",
        "pub extern crate core;",
        "#[macro_export]\nmacro_rules! m { () => {} }",
    ] {
        let root = format!("pub use a::B;\n{item}\npub mod seam {{ pub use c::D; }}\n");
        assert!(exports(&root).is_err(), "accepted `{item}` at the root");
        let seam = format!("pub use a::B;\npub mod seam {{ pub use c::D;\n{item}\n}}\n");
        assert!(exports(&seam).is_err(), "accepted `{item}` in seam");
    }
}

#[test]
fn firewall_ignores_comments_strings_and_restricted_visibility() {
    let source = r#"
        //! pub fn doc() {}
        /* pub enum Hidden {} */
        #[path = "pub fn x.rs"]
        pub(crate) mod inner;
        pub(super) fn helper() {}
        pub mod listed;
        pub use a::{B, c::D as E}; // pub struct Trailing;
        pub mod seam {
            pub use f::G;
            #[cfg(feature = "x")]
            pub use h::{I, J};
        }
        #[cfg(test)]
        use k::L;
    "#;
    let (root, seam) = exports(source).unwrap();
    assert_eq!(root, names("listed B E seam"));
    assert_eq!(seam, names("G I J"));
}

#[test]
fn crate_root_and_seam_export_exactly_the_consumed_items() {
    let (root, seam) = exports(include_str!("../src/lib.rs")).unwrap();
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
             decide_bond_truncation_generic_checked \
             diagonal_bond_bound_space_generic_checked diagonal_bond_bound_space_like \
             diagonal_bond_data eig_full_diagonal_dyn eig_full_diagonal_dyn_checked_generic \
             eig_full_dyn eig_full_dyn_checked_generic eig_vals_diagonal_dyn eig_vals_dyn \
             eig_vals_dyn_checked_generic eigh_full_diagonal_dyn \
             eigh_full_diagonal_dyn_checked_generic eigh_full_dyn eigh_full_dyn_checked_generic \
             eigh_vals_diagonal_dyn eigh_vals_dyn eigh_vals_dyn_checked_generic \
             factor_isomorphic_checked_generic factor_output_space_checked_generic \
             left_null_diagonal_dyn left_null_diagonal_dyn_checked_generic left_null_dyn \
             left_null_dyn_checked_generic_with_dimensions left_polar_adjoint_parent_dyn \
             left_polar_adjoint_parent_dyn_checked_generic left_polar_diagonal_spectra_dyn \
             left_polar_diagonal_spectra_dyn_checked_generic left_polar_dyn \
             left_polar_dyn_checked_generic lq_compact_dyn lq_compact_dyn_checked_generic \
             lq_diagonal_dyn_checked_generic lq_full_dyn \
             lq_full_dyn_checked_generic qr_compact_dyn qr_compact_dyn_checked_generic \
             qr_diagonal_dyn qr_diagonal_dyn_checked_generic qr_full_dyn \
             qr_full_dyn_checked_generic rectangular_diagonal_bond_tensor \
             rectangular_diagonal_bond_tensor_generic_checked right_null_diagonal_dyn \
             right_null_diagonal_dyn_checked_generic right_null_dyn \
             right_null_dyn_checked_generic_with_dimensions right_polar_adjoint_parent_dyn \
             right_polar_adjoint_parent_dyn_checked_generic right_polar_diagonal_spectra_dyn \
             right_polar_diagonal_spectra_dyn_checked_generic right_polar_dyn \
             right_polar_dyn_checked_generic scale_axis_by_spectrum_mapped \
             svd_compact_adjoint_factors_dyn svd_compact_diagonal_factors_dyn \
             svd_compact_diagonal_factors_dyn_checked_generic svd_compact_dyn_checked_generic \
             svd_compact_factors_dyn \
             svd_compact_factors_with_spectrum_dyn_checked_generic svd_full_adjoint_factors_dyn \
             svd_full_diagonal_factors_dyn_checked_generic svd_full_factors_dyn \
             svd_full_factors_dyn_checked_generic \
             svd_full_factors_dyn_checked_generic_with_dimensions svd_vals_compact_diagonal_dyn \
             svd_vals_dyn svd_vals_dyn_checked_generic \
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
