use tenet_network::{
    active_pair_path_from_tree, block_sparse_order_from_labels, dense_steps_from_active_pair_path,
    greedy_slice, ActivePair, BlockInfo, BlockSparseContractionOptimizer, BlockSparseCostModel,
    BlockSparseTensorInfo, ContractionPlan, DenseContractionOptimizer, DenseCostModel,
    DenseTensorInfo, GreedyBlockSparseOptimizer, GreedyDenseOptimizer, LabelOrderDenseOptimizer,
    NetworkIR, TemporaryLabel,
};

// Captured with the corresponding old public constructors and both old slice
// functions on origin/main 98c9f4a8 before the API migration.
const DENSE_PATH: &str =
    "tenet-contract-plan-v1\ntensor_count 3\noutput a d\nstep 0 1 3 32 a c\nstep 2 3 4 50 a d\n";
const DENSE_GREEDY: &str =
    "tenet-contract-plan-v1\ntensor_count 3\noutput a d\nstep 0 1 3 24 a c\nstep 3 2 4 40 a d\n";
const SPARSE: &str =
    "tenet-contract-plan-v1\ntensor_count 3\noutput a d\nstep 0 1 3 8 a c\nstep 2 3 4 8 a d\n";
const SLICE_INTERNAL: &str = "tenet-slice-plan-v1\nnslices 8\nsliced_width 4\nunsliced_width 16\nper_slice_flops 8\nslice internal c\n";
const SLICE_OUTPUT: &str = "tenet-slice-plan-v1\nnslices 6\nsliced_width 6\nunsliced_width 36\nper_slice_flops 7\nslice output a\n";

fn chain_ir() -> NetworkIR {
    NetworkIR::from_labels(
        vec![
            vec!["a".into(), "b".into()],
            vec!["b".into(), "c".into()],
            vec!["c".into(), "d".into()],
        ],
        vec!["a".into(), "d".into()],
    )
    .unwrap()
}

fn dense(ir: &NetworkIR, sizes: [[usize; 2]; 3]) -> DenseCostModel {
    DenseCostModel::from_network(
        ir,
        &sizes
            .into_iter()
            .map(|dims| DenseTensorInfo::new(dims.to_vec()))
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn migrated_plan_entries_match_old_text() {
    let ir = chain_ir();
    let cost = dense(&ir, [[2, 3], [3, 4], [4, 5]]);
    let path = [ActivePair::new(0, 1), ActivePair::new(0, 1)];
    let order = [TemporaryLabel::new("b"), TemporaryLabel::new("c")];
    let path_steps = dense_steps_from_active_pair_path(&ir, &path, &cost).unwrap();
    let path_plan = ContractionPlan::from_steps(&ir, path_steps).unwrap();
    assert_eq!(path_plan.to_text(), DENSE_PATH);

    let label_steps = DenseContractionOptimizer::optimize(
        &LabelOrderDenseOptimizer::new(order.to_vec()),
        &ir,
        &cost,
    )
    .unwrap();
    assert_eq!(
        ContractionPlan::from_steps(&ir, label_steps)
            .unwrap()
            .to_text(),
        DENSE_PATH
    );

    let greedy_steps = GreedyDenseOptimizer.optimize(&ir, &cost).unwrap();
    assert_eq!(
        ContractionPlan::from_steps(&ir, greedy_steps)
            .unwrap()
            .to_text(),
        DENSE_GREEDY
    );

    let tree = path_plan.tree().unwrap();
    let tree_path = active_pair_path_from_tree(ir.tensors().len(), &tree).unwrap();
    let tree_steps = dense_steps_from_active_pair_path(&ir, &tree_path, &cost).unwrap();
    assert_eq!(
        ContractionPlan::from_steps(&ir, tree_steps)
            .unwrap()
            .to_text(),
        DENSE_PATH
    );

    let blocks = [["a", "b"], ["b", "c"], ["c", "d"]]
        .into_iter()
        .map(|axes| {
            BlockSparseTensorInfo::new(vec![BlockInfo::new(
                axes.into_iter()
                    .map(|name| (TemporaryLabel::new(name), 0i32, 2))
                    .collect(),
            )
            .unwrap()])
        })
        .collect::<Vec<_>>();
    let sparse = BlockSparseCostModel::from_network(&ir, &blocks).unwrap();
    let sparse_label = block_sparse_order_from_labels(&ir, &sparse, &order).unwrap();
    assert_eq!(
        ContractionPlan::from_steps(&ir, sparse_label)
            .unwrap()
            .to_text(),
        SPARSE
    );
    let sparse_greedy = GreedyBlockSparseOptimizer.optimize(&ir, &sparse).unwrap();
    assert_eq!(
        ContractionPlan::from_steps(&ir, sparse_greedy)
            .unwrap()
            .to_text(),
        SPARSE
    );

    let internal_cost = dense(&ir, [[2, 2], [2, 8], [8, 2]]);
    let internal_plan = ContractionPlan::from_steps(
        &ir,
        dense_steps_from_active_pair_path(&ir, &path, &internal_cost).unwrap(),
    )
    .unwrap();
    assert_eq!(
        greedy_slice(&ir, &internal_plan, &internal_cost, 8, false).to_text(),
        SLICE_INTERNAL
    );

    let output_cost = dense(&ir, [[6, 1], [1, 1], [1, 6]]);
    let output_plan = ContractionPlan::from_steps(
        &ir,
        dense_steps_from_active_pair_path(&ir, &path, &output_cost).unwrap(),
    )
    .unwrap();
    assert_eq!(
        greedy_slice(&ir, &output_plan, &output_cost, 6, true).to_text(),
        SLICE_OUTPUT
    );
}
