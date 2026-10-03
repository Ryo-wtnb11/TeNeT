//! Export firewall for `tenet-core` (#1805).
//!
//! Pins the exact names reachable at the crate root, under every feature
//! combination at once, so adding or removing one fails here and is reviewed
//! as an API change. Test-only tree operations (`generic_braid_tree`,
//! `multiplicity_free_*_block`, `unique_permute_tree`, ...) are crate-private;
//! the few that tests outside the crate use are forwarded only through the
//! `testing` feature's `tenet_core::testing`, pinned separately below.
//! `merge_fusion_trees_generic` and the `from_degeneracy_shapes` alias of
//! `FusionTensorMapSpace::from_degeneracy_shapes_coupled` were deleted.
//!
//! Why a source scan and not an import list: an import list only proves
//! presence, not absence. The crate root re-exports its private modules with
//! `pub use module::*`, so the scan follows each glob into the module's file
//! and collects that file's top-level (column 0, rustfmt-formatted) `pub`
//! items; `pub(crate)` and indented items (impl members, nested modules) are
//! not exports.

use std::collections::BTreeSet;
use std::path::Path;

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

/// Names a module file exports at its top level. `pub use child::*` of a
/// child module recurses into the child's file (`dir/child.rs` from `lib.rs`,
/// `dir/<module>/child.rs` otherwise); `pub mod m` exports `m` without
/// recursing (its items live under `m::`).
fn exported_names(dir: &Path, file: &str) -> Result<BTreeSet<String>, String> {
    let path = dir.join(file);
    let source = std::fs::read_to_string(&path).map_err(|e| format!("{path:?}: {e}"))?;
    let code = strip(&source);
    if code.contains("macro_export") {
        return Err(format!("{file} exports a macro_rules! macro"));
    }
    let child_dir = match file {
        "lib.rs" | "mod.rs" => dir.to_path_buf(),
        _ => dir.join(file.trim_end_matches(".rs")),
    };
    let mut names = BTreeSet::new();
    for (at, _) in code.match_indices("pub") {
        if at != 0 && !code[..at].ends_with('\n') {
            continue; // not a top-level item
        }
        let after = &code[at + 3..];
        if after.starts_with('(') || after.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            continue; // `pub(crate)` etc.
        }
        let item = after.trim_start();
        let mut words = item.split(|c: char| !(c.is_alphanumeric() || c == '_'));
        let keyword = words.next().unwrap_or("");
        if keyword != "use" {
            let name = match keyword {
                "fn" | "struct" | "enum" | "trait" | "type" | "static" | "union" | "mod" => {
                    words.find(|w| !w.is_empty())
                }
                "const" | "unsafe" | "async" | "extern" => {
                    words.find(|w| !w.is_empty() && *w != "fn" && *w != "unsafe")
                }
                _ => None,
            }
            .ok_or_else(|| format!("{file}: unrecognised `pub {keyword}` item"))?;
            names.insert(name.to_owned());
            continue;
        }
        let tree = &item[3..];
        let tree = tree[..tree.find(';').ok_or("unterminated pub use")?].trim();
        if let Some(module) = tree.strip_suffix("::*") {
            names.extend(exported_names(&child_dir, &format!("{module}.rs"))?);
            continue;
        }
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
        for leaf in leaves.iter().map(|l| l.trim()).filter(|l| !l.is_empty()) {
            let name = leaf.rsplit(" as ").next().unwrap();
            names.insert(name.rsplit("::").next().unwrap().trim().to_owned());
        }
    }
    Ok(names)
}

fn src() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"))
}

#[test]
fn firewall_follows_globs_and_ignores_restricted_and_nested_items() {
    let dir = std::env::temp_dir().join(format!("tenet-core-exports-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("inner")).unwrap();
    std::fs::write(
        dir.join("lib.rs"),
        "//! pub fn doc() {}\nmod inner;\npub use inner::*;\npub(crate) use hidden::*;\n\
         pub use a::{B, c::D as E};\npub const fn konst() {}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("inner.rs"),
        "pub struct Shown;\npub(crate) fn hidden() {}\nimpl Shown {\n    pub fn method() {}\n}\n\
         mod nested {\n    pub fn deep() {}\n}\n/* pub enum Gone {} */\nmod leaf;\npub use leaf::*;\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("inner/leaf.rs"),
        "pub type Leaf = u8;\npub mod testing;\n",
    )
    .unwrap();
    let names = exported_names(&dir, "lib.rs").unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(
        names,
        ["B", "E", "Leaf", "Shown", "konst", "testing"]
            .map(str::to_owned)
            .into()
    );
}

#[test]
fn tenet_core_exports_exactly_the_pinned_names() {
    let expected = [
        "BlockKey",
        "BlockKeyKind",
        "BlockLayout",
        "BlockRef",
        "BlockSpec",
        "BlockStructure",
        "BlockStructureContent",
        "BlockStructureContentBlock",
        "BlockStructureInternCacheInfo",
        "BlockView",
        "BlockViewMut",
        "BraidingStyleKind",
        "CU1FusionRule",
        "CU1Irrep",
        "CU1_MAX_TWICE_CHARGE",
        "CanonicalUnitFusionRule",
        "CategoricalScalar",
        "CheckedCanonicalUnitFusionRule",
        "CheckedFusionAlgebra",
        "CheckedFusionSpaceError",
        "CheckedGenericAdmissionMode",
        "CheckedGenericFusion",
        "CheckedGenericPivotal",
        "CheckedGenericRigidSymbols",
        "CheckedGenericStructureError",
        "CheckedGenericSymbolError",
        "CompleteHomSpaceStructureCacheInfo",
        "CoreError",
        "CoupledSectorFold",
        "CoupledSectorFoldBuilder",
        "CoupledMatricizationBuilder",
        "CoupledMatrixPlacement",
        "CoupledSectorRegion",
        "CoupledTreeExtent",
        "DegeneracyBlock",
        "DegeneracyStructure",
        "DimVec",
        "DualVec",
        "FermionParityFusionRule",
        "FibonacciFusionRule",
        "FibonacciSector",
        "FusionAlgebraError",
        "FusionProductSpace",
        "FusionRule",
        "FusionSpaceAdmission",
        "FusionStyleKind",
        "FusionTensorMapSpace",
        "FusionTreeBlockGroup",
        "FusionTreeGroupKey",
        "FusionTreeHomSpace",
        "FusionTreeKey",
        "FusionTreeLayoutCacheInfo",
        "FusionTreePairKey",
        "FusionTreePairOrientation",
        "Fz2SectorLayout",
        "GenericFArray",
        "GenericFusionSymbols",
        "GenericRMatrix",
        "GenericRigidSymbols",
        "HomSpaceId",
        "HostReadableStorage",
        "HostStorage",
        "HostWritableStorage",
        "InfallibleGeneric",
        "LocallyValidatedFusionTreeBlockStructure",
        "MultiplicityFreeAdmissionMode",
        "MultiplicityFreeFusionRule",
        "MultiplicityFreeFusionSymbols",
        "MultiplicityFreeRigidSymbols",
        "MultiplicityIndex",
        "MultiplicityVec",
        "OpaqueBlockKey",
        "OrderedBlockLinearMap",
        "OrderedBlockLinearStorage",
        "OrientedFusionTreeHomSpace",
        "PackedProductCodec",
        "PackedSectorLayout",
        "PhysicalBasisError",
        "PhysicalFusionBasis",
        "Placement",
        "PreparedBlockStructure",
        "PreparedFusionTreeLayout",
        "PreparedTreePairOperation",
        "ProductFusionRule",
        "ProductFusionRuleExt",
        "ProductSector",
        "ProductSectorCodec",
        "ProductSectorCodecError",
        "ProductSectorComponent",
        "ProductSectorLayout",
        "ProductSpace",
        "PromoteCoefficientScalar",
        "RuleIdentity",
        "SU2FusionRule",
        "SU2Irrep",
        "SU2_MAX_DOUBLED_SPIN",
        "SUNFusionRule",
        "SUNFusionRuleError",
        "ScratchStorage",
        "SectorBlock",
        "SectorCodec",
        "SectorId",
        "SectorLeg",
        "SectorLegConstructionError",
        "SectorOrderKey",
        "SectorStructure",
        "SectorVec",
        "SimilarStorage",
        "StructurallyValidatedFusionTreeSubset",
        "Su2SectorLayout",
        "SymbolShapeError",
        "Tensor",
        "TensorKitProductCodec",
        "TensorMap",
        "TensorMapSpace",
        "TensorStorage",
        "Trivial",
        "TypedSectorAdmission",
        "U1FusionRule",
        "U1Irrep",
        "U1SectorLayout",
        "UnitLegInsertion",
        "WeakHomSpaceId",
        "Z2FusionRule",
        "Z2Irrep",
        "ZNFusionRule",
        "ZNIrrep",
        "axes",
        "block_structure_intern_cache_info",
        "checked_product",
        "column_major_strides",
        "complete_hom_space_structure_cache_info",
        "fusion_tree_layout_cache_info",
        "generic_braid_tree_pair_block_ordered",
        "generic_braid_tree_pair_checked",
        "generic_permute_tree_pair_block_ordered",
        "generic_permute_tree_pair_checked",
        "generic_transpose_tree_pair",
        "generic_transpose_tree_pair_block_ordered",
        "generic_transpose_tree_pair_checked",
        "merge_fusion_trees_generic_checked",
        "merge_fusion_trees_multiplicity_free",
        "multiplicity_free_braid_tree_pair",
        "multiplicity_free_braid_tree_pair_block_ordered_indexed",
        "multiplicity_free_permute_tree_pair",
        "multiplicity_free_permute_tree_pair_block_indexed",
        "multiplicity_free_repartition_tree_pair",
        "multiplicity_free_transpose_tree_pair_block_ordered_indexed",
        "product_fusion_rule",
        "product_fusion_rule_with_codec",
        "product_sector",
        "reset_core_intern_tables",
        "split_fusion_tree",
        "split_fusion_tree_generic_checked",
        "testing",
        "try_for_each_column_major_stride",
        "validate_block_storage_injective",
        "validate_generic_fusion_tree_pair_checked",
        "validate_unit_layout_correspondence",
        "validate_unit_layout_correspondence_checked",
        "validate_unit_layout_correspondence_generic_checked",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(exported_names(src(), "lib.rs").unwrap(), expected);
}

#[test]
fn testing_feature_forwards_exactly_the_pinned_oracles() {
    let expected = [
        "generic_braid_tree_pair",
        "generic_permute_tree_pair",
        "multiplicity_free_braid_tree_block",
        "multiplicity_free_permute_tree_pair_block",
        "multiplicity_free_transpose_tree_pair",
        "unique_permute_tree",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(exported_names(src(), "testing.rs").unwrap(), expected);
}
