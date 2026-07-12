use tenet_core::{CoreError, FusionTreeHomSpace, U1FusionRule, Z2FusionRule};
use tenet_tensors::{DynamicFusionMapSpace, OperationError, TreeTransformOperation};

#[test]
fn space_built_for_z2_rejects_u1_operation_with_same_integer_sector() {
    let space = DynamicFusionMapSpace::from_degeneracy_shapes(
        &Z2FusionRule,
        FusionTreeHomSpace::from_sector_ids([(0, 1)], []),
        [vec![1]],
    )
    .unwrap();

    let error = space
        .transformed(&U1FusionRule, &TreeTransformOperation::permute([0], []))
        .unwrap_err();

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}
