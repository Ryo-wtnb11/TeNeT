//! Export firewall for `tenet-operations` (#1804).
//!
//! Raw kernels, fusion-tree helpers and unrouted replay entries are crate
//! internals: no workspace crate calls them, and keeping them public froze
//! signatures that the one-pipeline refactors (#1851) must change. This test
//! fails when one of them becomes `pub` again, so re-exposing one is reviewed
//! as an API change.
//!
//! Why a source scan and not `compile_fail` doctests: one doctest proves the
//! absence of one path at a time, and the crate root re-exports whole modules
//! by glob, so the definition's visibility is the single place that decides
//! whether a name escapes.

use std::fs;
use std::path::{Path, PathBuf};

const CRATE_INTERNAL: &[&str] = &[
    // Raw kernels; the `_trusted` drivers are the routed entries.
    "axpby_raw_strided_kernel",
    "copy_block_with_strided_kernel",
    "tensoradd_raw_strided_kernel_trusted",
    "tensortrace_raw_strided_kernel",
    "tensortrace_raw_strided_kernel_add_with_coefficient",
    // Fusion-tree helpers.
    "block_indices_for_keys",
    "duplicate_fusion_tree_pair_index",
    "duplicate_fusion_tree_pair_indices",
    "fusion_tree_group_block_keys",
    "fusion_tree_pair_matches_group",
    "fusion_tree_pairs_share_group",
    // Misc.
    "direct_group_matrix_offset",
    "permutation_axes",
    "validate_recoupling_lens",
    "with_host_pool",
    // Unrouted replay entries.
    "tensoradd_structure_with_strided_kernel",
    "tree_transform_structure_with_strided_kernel",
    "tree_transform_structure_overwrite_with_storage_workspace_strided_kernel",
    "tree_transform_structure_with_structural_recoupling",
    "tree_transform_structure_overwrite_with_structural_recoupling",
    "tree_transform_structure_with_structural_recoupling_raw",
    "tree_transform_structure_with_structural_recoupling_raw_profiled",
    "tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled",
    // Compile constructors without a production caller; the keyed entry is
    // `compile_keyed_structures_with_storage_conjugation` and grouped specs
    // compile through `TreeTransformGroupPlan`.
    "compile_keyed",
    "compile_keyed_structures",
    "compile_grouped",
    "compile_grouped_structures",
    "compile_grouped_structures_with_storage_conjugation",
    "compile_grouped_shared_structures",
    // Replaced by the non-panicking `single_coefficient`.
    "coefficient",
];

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn crate_internal_items_stay_out_of_the_public_api() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);
    let mut escaped = Vec::new();
    for file in &files {
        let source = fs::read_to_string(file).unwrap();
        for name in CRATE_INTERNAL {
            for open in ['(', '<'] {
                if source.contains(&format!("pub fn {name}{open}")) {
                    escaped.push(format!("{}: {name}", file.display()));
                }
            }
        }
    }
    assert!(escaped.is_empty(), "public again: {escaped:#?}");

    let lib = fs::read_to_string(src.join("lib.rs")).unwrap();
    assert!(!lib.contains("pub mod transform_helpers;"));
}
