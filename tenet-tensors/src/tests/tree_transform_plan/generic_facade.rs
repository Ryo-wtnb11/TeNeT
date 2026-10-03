use super::*;

// ======================================================================
// Stage B3a: Generic-fusion facade (TensorMap-level) siblings.
// ======================================================================

fn b3a_toy_tensormap(value: f64) -> (BlockStructure, TensorMap<f64, 2, 0>) {
    // cod [1,1]->0 with per-leg degeneracy 1 -> a single-element block.
    let src_key = BlockKey::from(b2c_toy_src_pair());
    let structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let tensor =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![value], space, structure.clone())
            .unwrap();
    (structure, tensor)
}

#[test]
fn direct_generic_tree_pair_rejects_malformed_destination_without_writing() {
    let rule = ToyGenericRule {
        style: FusionStyleKind::Generic,
    };
    let src_structure =
        packed_fixture_structure(2, [(BlockKey::from(b2c_toy_src_pair()), vec![1, 1])]).unwrap();
    let malformed = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            [],
            [],
        )
        .unwrap(),
    );
    let dst_structure = packed_fixture_structure(2, [(malformed, vec![2, 1])]).unwrap();
    assert_ne!(src_structure.content_id(), dst_structure.content_id());
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![3.0],
        TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap(),
        src_structure,
    )
    .unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![17.0, 19.0],
        TensorMapSpace::<2, 0>::from_dims([2, 1], []).unwrap(),
        dst_structure,
    )
    .unwrap();
    let before = dst.data().to_vec();

    let error = tree_transform_into_generic(
        &rule,
        TreeTransformOperation::permute([0, 1], []),
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap_err();

    // What: the noncached Generic facade validates caller-owned destination
    // structure before lowering or replay changes its data.
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of vertices",
        })
    );
    assert_eq!(dst.data(), before);
}

// Gate 1 (highest reachable level = TensorMap facade): each generic facade
// sibling reproduces the Stage B2c plan-level path
// (`build_generic_tree_pair_transform_group_plan` -> compile -> execute)
// byte-for-byte. Combined with B2c's `..._matches_core_rows_...` (plan == core
// tree rows), this transitively proves facade == plan == tree-level hand-chain,
// i.e. the facade wiring adds no recoupling math.
#[test]
fn generic_facade_permute_braid_transpose_match_b2c_plan_level() {
    let rule = ToyGenericRule {
        style: FusionStyleKind::Generic,
    };
    let (structure, _) = b3a_toy_tensormap(0.0);

    // Plan-level (B2c) reference for one operation.
    let plan_level = |operation: TreeTransformOperation| -> Vec<f64> {
        let (_, src) = b3a_toy_tensormap(7.0);
        let (_, mut dst) = b3a_toy_tensormap(0.0);
        let plan =
            build_generic_tree_pair_transform_group_plan(&rule, operation, &structure).unwrap();
        let compiled = plan.compile(&dst, &src).unwrap();
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &compiled,
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
        dst.data().to_vec()
    };

    let (_, src) = b3a_toy_tensormap(7.0);
    let (_, mut dst) = b3a_toy_tensormap(0.0);
    permute_into_generic(&rule, [1, 0], [], &mut dst, &src, 1.0, 0.0).unwrap();
    assert_eq!(
        dst.data(),
        plan_level(TreeTransformOperation::permute([1, 0], [])).as_slice()
    );

    let (_, mut dst) = b3a_toy_tensormap(0.0);
    braid_into_generic(&rule, [1, 0], [], [0, 1], [], &mut dst, &src, 1.0, 0.0).unwrap();
    assert_eq!(
        dst.data(),
        plan_level(TreeTransformOperation::braid([1, 0], [], [0, 1], [])).as_slice()
    );

    let (_, mut dst) = b3a_toy_tensormap(0.0);
    transpose_into_generic(&rule, [1, 0], [], &mut dst, &src, 1.0, 0.0).unwrap();
    assert_eq!(
        dst.data(),
        plan_level(TreeTransformOperation::transpose([1, 0], [])).as_slice()
    );
}

// Gate 3 (mult-free cannot enter the generic path): the compile-time guarantee
// is trait disjointness (`GenericRigidSymbols` vs `MultiplicityFreeRigidSymbols`
// are never both implemented). This is its runtime symmetric sibling — the
// facade's Generic entry rejects a rule that reports a multiplicity-free style,
// mirroring the mult-free builders' `UnsupportedFusionStyle` guards.
#[test]
fn generic_facade_structure_rejects_multiplicity_free_style() {
    let mf = ToyGenericRule {
        style: FusionStyleKind::Simple,
    };
    let (_, src) = b3a_toy_tensormap(7.0);
    let (_, dst) = b3a_toy_tensormap(0.0);
    let err = tree_transform_structure_generic(
        &mf,
        TreeTransformOperation::permute([1, 0], []),
        &dst,
        &src,
    )
    .unwrap_err();
    assert!(matches!(err, OperationError::UnsupportedFusionStyle { .. }));
}
