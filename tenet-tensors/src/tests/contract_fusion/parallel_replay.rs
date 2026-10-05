use super::*;

/// Parallel replay (threads=4, size gate dropped) must match the serial
/// default elementwise: swap+out exercises source and output tree
/// transforms — Singles for the abelian rules (U1, fZ2) and multi-tree
/// pack/GEMM/scatter blocks for SU2.
#[test]
fn tensorcontract_fusion_parallel_transform_replay_matches_serial() {
    fn run_case<R>(rule: &R, sectors: &[SectorId], degeneracy: usize)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
        R::Key: Clone + Eq + std::hash::Hash,
    {
        let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
        let leg_dim = sectors.len() * degeneracy;
        let homspace = || {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(), leg()]),
                FusionProductSpace::new([leg(), leg()]),
            )
        };
        let space = |hom: FusionTreeHomSpace| {
            let key_count = hom.fusion_tree_keys(rule).len();
            let dense =
                TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap();
            let shapes = vec![vec![degeneracy; 4]; key_count];
            FusionTensorMapSpace::from_degeneracy_shapes_coupled(dense, hom, rule, shapes).unwrap()
        };

        let lhs_space = space(homspace());
        let rhs_space = space(homspace());
        let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            (0..lhs_space.required_len().unwrap())
                .map(|index| (index % 17) as f64 * 0.25 - 2.0)
                .collect(),
            lhs_space,
        )
        .unwrap();
        let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            (0..rhs_space.required_len().unwrap())
                .map(|index| (index % 13) as f64 * 0.5 - 3.0)
                .collect(),
            rhs_space,
        )
        .unwrap();

        // swap+out: C[b a; g h] = A[a b; c d] * B[d c; g h]
        let lhs_axes = [3usize, 2];
        let rhs_axes = [0usize, 1];
        let output_axes = [1usize, 0, 2, 3];
        let dst_space = || {
            let hom = FusionTreeHomSpace::tensorcontract_homspace(
                rule,
                lhs.fusion_space().unwrap().homspace(),
                rhs.fusion_space().unwrap().homspace(),
                &lhs_axes,
                &rhs_axes,
                &output_axes,
                2,
            )
            .unwrap();
            space(hom)
        };
        let axes = TensorContractSpec::new(
            &lhs_axes,
            &rhs_axes,
            OutputAxisOrder::from_axes(&output_axes),
        );

        let dst_len = dst_space().required_len().unwrap();
        let mut serial_dst =
            TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; dst_len], dst_space())
                .unwrap();
        let mut serial_context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
        serial_context
            .tensorcontract_fusion_into(rule, &mut serial_dst, &lhs, &rhs, axes, 1.0, 0.0)
            .unwrap();

        let mut parallel_dst =
            TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; dst_len], dst_space())
                .unwrap();
        let mut parallel_context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
        let backend = parallel_context.tree_context_mut().backend_mut();
        backend.set_recoupling_threads(4);
        backend.set_transform_parallel_min_len(0);
        // Two runs: cold (structure compile + replay) and warm (replay only).
        for _ in 0..2 {
            parallel_context
                .tensorcontract_fusion_into(rule, &mut parallel_dst, &lhs, &rhs, axes, 1.0, 0.0)
                .unwrap();
        }

        assert_eq!(parallel_dst.data(), serial_dst.data());
    }

    run_case(
        &U1FusionRule,
        &[
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(1).sector_id(),
        ],
        3,
    );
    run_case(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
        3,
    );
    run_case(
        &SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
            SU2Irrep::from_twice_spin(2).sector_id(),
        ],
        3,
    );
    let left_rule = FpU1Rule::default();
    let product_rule = FpU1Su2Rule::default();
    let left_sector =
        |parity, charge| left_rule.encode_sector(parity, U1Irrep::new(charge).sector_id());
    let product_sector = |parity, charge, twice_spin| {
        product_rule.encode_sector(
            left_sector(parity, charge),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };
    run_case(
        &product_rule,
        &[
            product_sector(SectorId::new(0), 0, 0),
            product_sector(SectorId::new(1), 1, 1),
            product_sector(SectorId::new(1), -1, 1),
        ],
        2,
    );
}

/// Parallel replay must preserve the fermionic twist sign: the deg-2 dual-leg
/// contraction from `tensorcontract_fusion_fermion_twist_deg2_matches_tensorkit_reference`
/// run through a threads=4 context against the TensorKit crosscheck values.
#[test]
fn tensorcontract_fusion_parallel_replay_keeps_fermion_twist_reference() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let space = |codomain_dual: bool, domain_dual: bool| {
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<1, 1>::from_dims([3], [3]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 2)], codomain_dual)]),
                FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 2)], domain_dual)]),
            ),
            &rule,
            [vec![1, 1], vec![2, 2]],
        )
        .unwrap()
    };
    let lhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        vec![0.5, 1.5, 2.5, 3.5, 4.5],
        space(false, true),
    )
    .unwrap();
    let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        vec![-1.25, -0.75, -0.25, 0.25, 0.75],
        space(true, false),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![0.0; 5], space(false, false))
            .unwrap();

    let mut backend = DenseTreeTransformOperations::default_executor();
    backend.set_recoupling_threads(4);
    backend.set_transform_parallel_min_len(0);
    TensorContractFusionExecutionContext::<f64, RuleIdentity>::new(
        DenseTreeTransformOperations::default_executor(),
        backend,
    )
    .tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        1.0,
        0.0,
    )
    .unwrap();

    let expected = [-0.625, 2.0, 3.0, -3.0, -4.0];
    for (index, (&actual, &want)) in dst.data().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1.0e-12,
            "element {index}: got {actual}, TensorKit reference {want}"
        );
    }
}
