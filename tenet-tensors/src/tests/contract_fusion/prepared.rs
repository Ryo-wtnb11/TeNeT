use super::*;

#[test]
fn prepared_tensorcontract_fusion_matches_facade_and_rejects_foreign_tensors() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let fusion_space = || {
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap()
    };
    // What: replaying the tensors used to prepare the handle remains valid.
    let space = fusion_space();
    let lhs = test_host_read_fusion_tensor_map(vec![2.0_f64, 3.0], space.clone());
    let rhs = test_host_read_fusion_tensor_map(vec![5.0_f64, 7.0], space.clone());
    let mut dst_facade = test_host_fusion_tensor_map(vec![10.0_f64, 20.0], space.clone());
    let mut dst_prepared = test_host_fusion_tensor_map(vec![10.0_f64, 20.0], space.clone());
    let axes = TensorContractSpec::with_default_output_order(&[1], &[0]);
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    let prepared = context
        .prepare_tensorcontract_fusion(&rule, &dst_prepared, &lhs, &rhs, axes)
        .unwrap();
    for _ in 0..2 {
        context
            .tensorcontract_fusion_into(&rule, &mut dst_facade, &lhs, &rhs, axes, 2.0, 3.0)
            .unwrap();
        context
            .execute_prepared_tensorcontract_fusion(
                &prepared,
                &rule,
                &mut dst_prepared,
                &lhs,
                &rhs,
                2.0,
                3.0,
            )
            .unwrap();
    }
    assert_eq!(dst_prepared.data(), dst_facade.data());

    let mut foreign_dst = test_host_fusion_tensor_map(vec![0.0_f64, 0.0], fusion_space());
    let err = context
        .execute_prepared_tensorcontract_fusion(
            &prepared,
            &rule,
            &mut foreign_dst,
            &lhs,
            &rhs,
            1.0,
            0.0,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        OperationError::StructureMismatch {
            tensor: "prepared contraction"
        }
    ));
    assert_eq!(foreign_dst.data(), &[0.0, 0.0]);

    let foreign_lhs = test_host_read_fusion_tensor_map(vec![2.0_f64, 3.0], fusion_space());
    let dst_before_error = dst_prepared.data().to_vec();
    let err = context
        .execute_prepared_tensorcontract_fusion(
            &prepared,
            &rule,
            &mut dst_prepared,
            &foreign_lhs,
            &rhs,
            1.0,
            0.0,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        OperationError::StructureMismatch {
            tensor: "prepared contraction"
        }
    ));
    assert_eq!(dst_prepared.data(), dst_before_error);

    let foreign_rhs = test_host_read_fusion_tensor_map(vec![5.0_f64, 7.0], fusion_space());
    let dst_before_error = dst_prepared.data().to_vec();
    let err = context
        .execute_prepared_tensorcontract_fusion(
            &prepared,
            &rule,
            &mut dst_prepared,
            &lhs,
            &foreign_rhs,
            1.0,
            0.0,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        OperationError::StructureMismatch {
            tensor: "prepared contraction"
        }
    ));
    assert_eq!(dst_prepared.data(), dst_before_error);
}

#[test]
fn prepared_tensorcontract_fusion_pins_exact_space_allocations() {
    // What: a prepared handle pins all original fusion-space allocations and
    // rejects semantically equal replacements after the source tensors drop.
    let rule = Z2FusionRule;
    let fusion_space = || {
        let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap()
    };
    let axes = TensorContractSpec::with_default_output_order(&[1], &[0]);
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let (prepared, dst_weak, lhs_weak, rhs_weak) = {
        let lhs = test_host_read_fusion_tensor_map(vec![2.0_f64, 3.0], fusion_space());
        let rhs = test_host_read_fusion_tensor_map(vec![5.0_f64, 7.0], fusion_space());
        let dst = test_host_fusion_tensor_map(vec![0.0_f64, 0.0], fusion_space());
        let dst_weak = Arc::downgrade(dst.fusion_space().unwrap());
        let lhs_weak = Arc::downgrade(lhs.fusion_space().unwrap());
        let rhs_weak = Arc::downgrade(rhs.fusion_space().unwrap());
        let prepared = context
            .prepare_tensorcontract_fusion(&rule, &dst, &lhs, &rhs, axes)
            .unwrap();
        (prepared, dst_weak, lhs_weak, rhs_weak)
    };

    assert!(dst_weak.upgrade().is_some());
    assert!(lhs_weak.upgrade().is_some());
    assert!(rhs_weak.upgrade().is_some());

    let lhs = test_host_read_fusion_tensor_map(vec![2.0_f64, 3.0], fusion_space());
    let rhs = test_host_read_fusion_tensor_map(vec![5.0_f64, 7.0], fusion_space());
    let mut dst = test_host_fusion_tensor_map(vec![0.0_f64, 0.0], fusion_space());
    let err = context
        .execute_prepared_tensorcontract_fusion(&prepared, &rule, &mut dst, &lhs, &rhs, 1.0, 0.0)
        .unwrap_err();
    assert!(matches!(
        err,
        OperationError::StructureMismatch {
            tensor: "prepared contraction"
        }
    ));
}
