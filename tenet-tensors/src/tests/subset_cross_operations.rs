use super::*;
use tenet_core::FusionSpaceAdmission;

const ALPHA: f64 = 2.0;
const BETA: f64 = 3.0;
const SOURCE_VALUE: f64 = 7.0;

fn scalar_matrix_space<R>(rule: &R, sectors: [SectorId; 2]) -> FusionTensorMapSpace<1, 1>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let leg = || SectorLeg::new(sectors.map(|sector| (sector, 1)), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let block_count = homspace.fusion_tree_keys(rule).len();
    FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        homspace,
        rule,
        vec![vec![1, 1]; block_count],
    )
    .unwrap()
}

fn source_bytes(source: &TensorMap<f64, 1, 1>) -> Vec<u8> {
    source
        .data()
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn sentinel_data(space: &FusionTensorMapSpace<1, 1>) -> Vec<f64> {
    (0..space.required_len().unwrap())
        .map(|index| 11.0 + index as f64)
        .collect()
}

fn assert_complete_destination(
    destination: &TensorMap<f64, 1, 1>,
    initial: &[f64],
    retained_key: &FusionTreePairKey,
) {
    let mut omitted = 0;
    for index in 0..destination.structure().block_count() {
        let block = destination.structure().block(index).unwrap();
        assert_eq!(block.shape(), &[1, 1]);
        let offset = block.offset();
        let contribution = if expect_tree_key(block.key()) == *retained_key {
            SOURCE_VALUE
        } else {
            omitted += 1;
            0.0
        };
        assert_eq!(
            destination.data()[offset],
            BETA * initial[offset] + ALPHA * contribution
        );
    }
    assert!(omitted > 0, "fixture must exercise omitted valid keys");
}

fn assert_subset_cross_operation_oracle<R>(rule: &R, sectors: [SectorId; 2])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let complete_space = scalar_matrix_space(rule, sectors);
    assert!(matches!(
        complete_space.admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    let retained_block = complete_space.subblock_structure().block(0).unwrap();
    let retained_key = expect_tree_key(retained_block.key());
    let subset_space = FusionTensorMapSpace::new_unbound(
        complete_space.dense_space().clone(),
        complete_space.homspace().clone(),
        packed_fixture_structure(2, [(retained_key.clone(), vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(rule)
    .unwrap();
    assert!(matches!(
        subset_space.admission(),
        FusionSpaceAdmission::Subset(_)
    ));

    let source =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![SOURCE_VALUE], subset_space)
            .unwrap();
    let bytes_before = source_bytes(&source);
    let admission_before = source.fusion_space().unwrap().admission().clone();

    let initial = sentinel_data(&complete_space);
    let mut add_destination =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial.clone(), complete_space.clone())
            .unwrap();
    tensoradd_fusion_into(
        rule,
        &mut add_destination,
        &source,
        TreeTransformOperation::permute([0], [1]),
        false,
        ALPHA,
        BETA,
    )
    .unwrap();
    assert_complete_destination(&add_destination, &initial, &retained_key);

    let mut trace_destination =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial.clone(), complete_space.clone())
            .unwrap();
    tensortrace_fusion_into(
        rule,
        &mut trace_destination,
        &source,
        TensorTraceAxisSpec::new(&[0, 1], &[], &[]),
        ALPHA,
        BETA,
    )
    .unwrap();
    assert_complete_destination(&trace_destination, &initial, &retained_key);

    let identity = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        vec![1.0; complete_space.required_len().unwrap()],
        complete_space.clone(),
    )
    .unwrap();
    let mut contract_destination =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial.clone(), complete_space)
            .unwrap();
    tensorcontract_fusion_into(
        rule,
        &mut contract_destination,
        &source,
        &identity,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        ALPHA,
        BETA,
    )
    .unwrap();
    assert_complete_destination(&contract_destination, &initial, &retained_key);

    assert_eq!(source_bytes(&source), bytes_before);
    assert_eq!(
        source.fusion_space().unwrap().admission(),
        &admission_before
    );
}

#[test]
fn bound_subset_is_a_structural_zero_source_across_public_operations() {
    // What: the same admitted Subset source contributes retained blocks and
    // treats omitted valid blocks as zero across add, zero-pair trace, and
    // left-hand contraction into Complete destinations.
    assert_subset_cross_operation_oracle(
        &U1FusionRule,
        [U1Irrep::new(-1).sector_id(), U1Irrep::new(2).sector_id()],
    );
    assert_subset_cross_operation_oracle(
        &SU2FusionRule,
        [
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    assert_subset_cross_operation_oracle(
        &FermionParityFusionRule,
        [SectorId::new(0), SectorId::new(1)],
    );

    let left_rule = FpU1Rule::default();
    let product_rule = FpU1Su2Rule::default();
    let product_sector = |parity, charge, twice_spin| {
        product_rule.encode_sector(
            left_rule.encode_sector(parity, U1Irrep::new(charge).sector_id()),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };
    assert_subset_cross_operation_oracle(
        &product_rule,
        [
            product_sector(SectorId::new(0), 0, 0),
            product_sector(SectorId::new(1), 1, 1),
        ],
    );
}
