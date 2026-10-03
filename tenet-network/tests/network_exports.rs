//! Export firewall for `tenet-network` (#1806).
//!
//! Pins the exact names `lib.rs` re-exports, under every feature combination
//! at once, so adding or removing one fails here and is reviewed as an API
//! change. The block-sparse planner (`BlockSparseCostModel`,
//! `GreedyBlockSparseOptimizer`, ...) was removed because nothing fed it
//! `TensorMap` sectors; it must not come back without a sector-aware design.
//!
//! Why a source scan and not an import list: an import list only proves
//! presence, not absence.

use std::collections::BTreeSet;

fn exported_names() -> BTreeSet<String> {
    let source = include_str!("../src/lib.rs");
    let code = source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !code.contains("pub mod ") && !code.contains("pub fn ") && !code.contains("pub struct "),
        "lib.rs gained an inline public item; extend this firewall"
    );
    let mut names = BTreeSet::new();
    for item in code.split("pub use ").skip(1) {
        let tree = &item[..item.find(';').expect("unterminated pub use")];
        let leaves = match tree.find('{') {
            Some(open) => {
                let inner = tree[open + 1..].trim_end().strip_suffix('}').unwrap();
                assert!(!inner.contains('{'), "nested use tree: {tree}");
                inner.split(',').map(str::to_owned).collect::<Vec<_>>()
            }
            None => vec![tree.to_owned()],
        };
        for leaf in leaves
            .iter()
            .map(|leaf| leaf.trim())
            .filter(|l| !l.is_empty())
        {
            let name = leaf.rsplit(" as ").next().unwrap();
            names.insert(name.rsplit("::").next().unwrap().to_owned());
        }
    }
    names
}

#[test]
fn tenet_network_exports_exactly_the_pinned_names() {
    let expected = [
        // cost
        "DenseCostModel",
        "DenseTensorInfo",
        // cotengra-python
        "CotengraPythonOptimizer",
        "CotengraMinimize",
        "CotengraPythonConfig",
        "CotengraPythonMethod",
        "CotengraSlicingConfig",
        // error
        "ContractError",
        "Result",
        "SliceError",
        "SliceResult",
        "SymmetricSliceExecutionError",
        "SymmetricSliceLowerError",
        // ir / labels
        "HyperEdge",
        "NetworkIR",
        "TensorNode",
        "LabelOccurrence",
        "TemporaryLabel",
        "TensorAxis",
        "TensorId",
        // network
        "contract_static_network",
        "contract_static_trace_network",
        "normalize_tensor_operand",
        "static_network_operand_preflight",
        "Network",
        "NetworkExecutionWorkspace",
        "PlannedNetwork",
        "StaticNetworkOperand",
        "StaticTopologySpec",
        "StaticTraceNetworkOperand",
        "SymmetricSliceStats",
        // optimizer
        "ContractionStep",
        "DenseContractionOptimizer",
        "DensePlanCostReport",
        "GreedyDenseOptimizer",
        "LabelOrderDenseOptimizer",
        // opt-path
        "BranchLevel",
        "DpObjective",
        "OptEinsumPathOptimizer",
        "PathMemoryLimit",
        "PathStrategy",
        // plan
        "active_pair_path_from_steps",
        "active_pair_path_from_tree",
        "dense_steps_from_active_pair_path",
        "ActivePair",
        "ContractionPlan",
        // plancache
        "clear_plan_cache",
        "configure_plan_cache",
        "load_plan_cache",
        "plan_cache_stats",
        "save_plan_cache",
        "Optimizer",
        "PlanCacheConfig",
        "PlanCacheStats",
        "ReplanPolicy",
        "DEFAULT_PLAN_CACHE_CAPACITY",
        "DEFAULT_REPLAN_DRIFT_FACTOR",
        // slice
        "best_next_internal_index",
        "best_next_slice_index",
        "contraction_width",
        "greedy_slice",
        "slice_plan_for",
        "DegeneracyRange",
        "SectorSlice",
        "SliceKind",
        "SliceLabels",
        "SlicePlan",
        "SlicedPlan",
        "SymmetricIndexSlice",
        "SymmetricSlicePlan",
        "SymmetricSliceSpec",
        "SymmetricSlicedPlan",
        // tree / macro
        "ContractionTree",
        "tensor",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(exported_names(), expected);
}
