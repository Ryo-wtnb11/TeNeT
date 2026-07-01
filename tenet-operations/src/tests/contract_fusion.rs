use super::*;

#[test]
fn tensorcontract_fusion_structure_enumerates_z2_compose_blocks_and_replays() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let lhs =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0, 3.0], lhs_space).unwrap();
    let rhs =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![5.0, 7.0], rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], dst_space).unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap();
    assert_eq!(
        specs,
        vec![
            TensorContractBlockSpec::new(0, 0, 0),
            TensorContractBlockSpec::new(1, 1, 1),
        ]
    );

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[50.0, 102.0]);
}

#[test]
fn tensorcontract_fusion_lowers_lhs_categorical_adjoint_lazily() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
    let space = || {
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
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0)],
        space(),
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 2.0), Complex64::new(7.0, -2.0)],
        space(),
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(10.0, 0.0), Complex64::new(20.0, 0.0)],
        space(),
    )
    .unwrap();
    let axes = TensorContractAxisSpec::canonical_with_conjugation(&[0], &[0], true, false);

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[Complex64::new(12.0, -1.0), Complex64::new(23.0, 1.0)]
    );

    dst.data_mut()
        .copy_from_slice(&[Complex64::new(10.0, 0.0), Complex64::new(20.0, 0.0)]);
    let mut context = TensorContractFusionExecutionContext::<Complex64, _>::default();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            axes,
            Complex64::one(),
            Complex64::zero(),
        )
        .unwrap();
    assert_eq!(
        dst.data(),
        &[Complex64::new(12.0, -1.0), Complex64::new(23.0, 1.0)]
    );
}

#[test]
fn tensorcontract_fusion_lowers_rhs_categorical_adjoint_lazily() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
    let space = || {
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
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0)],
        space(),
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 2.0), Complex64::new(7.0, -2.0)],
        space(),
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(), Complex64::zero()],
        space(),
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical_with_conjugation(&[1], &[1], false, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[Complex64::new(12.0, 1.0), Complex64::new(23.0, -1.0)]
    );
}

#[test]
fn tensorcontract_fusion_lowers_both_categorical_adjoint_inputs_lazily() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
    let space = || {
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
    let lhs_space = space();
    let rhs_space = space();
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        dst_hom,
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0)],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 2.0), Complex64::new(7.0, -2.0)],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(), Complex64::zero()],
        dst_space,
    )
    .unwrap();
    let axes = TensorContractAxisSpec::canonical_with_conjugation(&[0], &[1], true, true);

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[Complex64::new(8.0, -9.0), Complex64::new(19.0, 13.0)]
    );
}

fn z2_matrix_homspace() -> FusionTreeHomSpace {
    let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
}

fn z2_matrix_space_with_homspace(
    homspace: FusionTreeHomSpace,
    block_shape: Vec<usize>,
) -> FusionTensorMapSpace<1, 1> {
    FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        homspace,
        &Z2FusionRule,
        [block_shape.clone(), block_shape],
    )
    .unwrap()
}

fn fermion_parity_matrix_homspace() -> FusionTreeHomSpace {
    let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
}

fn fermion_parity_matrix_space_with_homspace(
    homspace: FusionTreeHomSpace,
    block_shape: Vec<usize>,
) -> FusionTensorMapSpace<1, 1> {
    FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        homspace,
        &FermionParityFusionRule,
        [block_shape.clone(), block_shape],
    )
    .unwrap()
}

#[test]
fn tensorcontract_fusion_lhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = Z2FusionRule;
    let lhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let rhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = z2_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((2,), (1,)), pB=((1,), (2,)).
        TensorContractAxisSpec::canonical_with_conjugation(&[0], &[0], true, false),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(16.0, -33.0),
            Complex64::new(44.0, -8.0),
            Complex64::new(35.0, -15.0),
            Complex64::new(41.0, 24.0),
            Complex64::new(6.0, -13.0),
            Complex64::new(-3.0, -4.0),
            Complex64::new(18.0, 10.0),
            Complex64::new(-10.0, 4.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_fermion_lhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = FermionParityFusionRule;
    let lhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let rhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = fermion_parity_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((2,), (1,)), pB=((1,), (2,)).
        TensorContractAxisSpec::canonical_with_conjugation(&[0], &[0], true, false),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(16.0, -33.0),
            Complex64::new(44.0, -8.0),
            Complex64::new(35.0, -15.0),
            Complex64::new(41.0, 24.0),
            Complex64::new(6.0, -13.0),
            Complex64::new(-3.0, -4.0),
            Complex64::new(18.0, 10.0),
            Complex64::new(-10.0, 4.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_rhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = Z2FusionRule;
    let lhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let rhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = z2_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical_with_conjugation(&[1], &[1], false, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(14.0, -1.0),
            Complex64::new(34.0, 6.0),
            Complex64::new(17.0, -1.0),
            Complex64::new(43.0, 10.0),
            Complex64::new(4.0, 3.0),
            Complex64::new(14.0, -6.0),
            Complex64::new(-10.0, 14.0),
            Complex64::new(-8.0, 6.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_fermion_rhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = FermionParityFusionRule;
    let lhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let rhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = fermion_parity_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((1,), (2,)), pB=((2,), (1,)).
        TensorContractAxisSpec::canonical_with_conjugation(&[1], &[1], false, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(14.0, -1.0),
            Complex64::new(34.0, 6.0),
            Complex64::new(17.0, -1.0),
            Complex64::new(43.0, 10.0),
            Complex64::new(4.0, 3.0),
            Complex64::new(14.0, -6.0),
            Complex64::new(-10.0, 14.0),
            Complex64::new(-8.0, 6.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_both_adjoint_uses_degeneracy_matrix_contract() {
    let rule = Z2FusionRule;
    let lhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let rhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace(), vec![2, 2]);
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = z2_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical_with_conjugation(&[0], &[1], true, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(23.0, -18.0),
            Complex64::new(33.0, 11.0),
            Complex64::new(31.0, -25.0),
            Complex64::new(44.0, 15.0),
            Complex64::new(-6.0, -15.0),
            Complex64::new(1.0, -4.0),
            Complex64::new(13.0, 7.0),
            Complex64::new(-5.0, -11.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_fermion_both_adjoint_uses_degeneracy_matrix_contract() {
    let rule = FermionParityFusionRule;
    let lhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let rhs_space =
        fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace(), vec![2, 2]);
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = fermion_parity_matrix_space_with_homspace(dst_hom, vec![2, 2]);
    let lhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); 8],
        dst_space,
    )
    .unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((2,), (1,)), pB=((2,), (1,)).
        TensorContractAxisSpec::canonical_with_conjugation(&[0], &[1], true, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(23.0, -18.0),
            Complex64::new(33.0, 11.0),
            Complex64::new(31.0, -25.0),
            Complex64::new(44.0, 15.0),
            Complex64::new(-6.0, -15.0),
            Complex64::new(1.0, -4.0),
            Complex64::new(13.0, 7.0),
            Complex64::new(-5.0, -11.0),
        ]
    );
}

#[test]
fn tensorproduct_fusion_lowers_lhs_adjoint_through_source_transform() {
    let rule = Z2FusionRule;
    let sector = SectorId::new(0);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 0>::from_dims([1], []).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([sector], false)]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        ),
        &rule,
        [vec![1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 0>::from_dims([1], []).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([sector], false)]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        ),
        &rule,
        [vec![1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([sector], true)]),
            FusionProductSpace::new([SectorLeg::new([sector], true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let lhs: TensorMap<Complex64, 1, 0> =
        TensorMap::from_vec_with_fusion_space(vec![Complex64::new(2.0, 1.0)], lhs_space).unwrap();
    let rhs: TensorMap<Complex64, 1, 0> =
        TensorMap::from_vec_with_fusion_space(vec![Complex64::new(3.0, -1.0)], rhs_space).unwrap();
    let mut dst: TensorMap<Complex64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![Complex64::new(0.0, 0.0)], dst_space).unwrap();

    tensorproduct_fusion_into_with_conjugation(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        AxisPermutation::identity(),
        true,
        false,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(dst.data(), &[Complex64::new(5.0, -5.0)]);

    dst.data_mut().copy_from_slice(&[Complex64::new(0.0, 0.0)]);
    let mut context = TensorContractFusionExecutionContext::<Complex64, _>::default();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::new_with_conjugation(
                &[],
                &[],
                AxisPermutation::identity(),
                true,
                false,
            ),
            Complex64::one(),
            Complex64::zero(),
        )
        .unwrap();
    assert_eq!(dst.data(), &[Complex64::new(5.0, -5.0)]);
}

#[test]
fn tensorcontract_fusion_fermion_rhs_dual_codomain_twists_like_tensorkit() {
    let rule = FermionParityFusionRule;
    let odd = SectorId::new(1);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([odd], false)]),
            FusionProductSpace::new([SectorLeg::new([odd], true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([odd], true)]),
            FusionProductSpace::new([SectorLeg::new([odd], false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([odd], false)]),
            FusionProductSpace::new([SectorLeg::new([odd], false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let lhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![3.0], rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![10.0], dst_space).unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap();
    assert_eq!(
        specs,
        vec![TensorContractBlockSpec::with_coefficient(0, 0, 0, -1.0)]
    );

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[-6.0]);
}

#[test]
fn tensorcontract_fusion_block_specs_enumerates_su2_innerline_blocks_from_homspace() {
    let rule = SU2FusionRule;
    let half = SectorId::new(1);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]),
        &rule,
        [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1], [1]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]),
        &rule,
        [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
    )
    .unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        &dst_space,
        &lhs_space,
        &rhs_space,
        TensorContractAxisSpec::canonical(&[3], &[0]),
    )
    .unwrap();

    assert_eq!(
        specs,
        vec![
            TensorContractBlockSpec::new(0, 0, 0),
            TensorContractBlockSpec::new(1, 1, 0),
        ]
    );
    assert_eq!(
        dst_space
            .homspace()
            .fusion_tree_keys_from_external_sectors(&rule, &[half, half, half, half])
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn tensorcontract_fusion_block_specs_rejects_missing_destination_subblock() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let keys = dst_hom.fusion_tree_keys(&rule);
    let dst_structure =
        BlockStructure::packed_column_major_with_keys(2, [(keys[0].clone(), vec![1, 1])]).unwrap();
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        dst_hom,
        dst_structure,
    )
    .unwrap();

    let err = tensorcontract_fusion_block_specs(
        &rule,
        &dst_space,
        &lhs_space,
        &rhs_space,
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: keys[1].clone().into()
        }
    );
}

#[test]
fn tensorcontract_fusion_block_specs_rejects_source_tree_transform_terms() {
    let rule = Z2FusionRule;
    let leg = |is_dual| SectorLeg::new([SectorId::new(0)], is_dual);
    let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false)]),
            FusionProductSpace::new([leg(false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let transformed_dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(true)]),
            FusionProductSpace::new([leg(true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();

    let err = tensorcontract_fusion_block_specs(
        &rule,
        &transformed_dst_space,
        &fusion_space,
        &fusion_space,
        TensorContractAxisSpec::canonical(&[0], &[1]),
    )
    .unwrap_err();

    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        &transformed_dst_space,
        &fusion_space,
        &fusion_space,
        TensorContractAxisSpec::new(&[1], &[0], AxisPermutation::from_axes(&[1, 0])),
    )
    .unwrap();

    assert_eq!(
        specs,
        vec![TensorContractBlockSpec::with_coefficient(0, 0, 0, 1.0)]
    );
}

#[test]
fn tensorcontract_fusion_into_absorbs_source_tree_transform_terms() {
    let rule = Z2FusionRule;
    let leg = |is_dual| SectorLeg::new([SectorId::new(0)], is_dual);
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false)]),
            FusionProductSpace::new([leg(false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(true)]),
            FusionProductSpace::new([leg(true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let lhs =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0], src_space.clone()).unwrap();
    let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![5.0], src_space).unwrap();
    let mut dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![7.0], dst_space).unwrap();

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[0], &[1]),
        3.0,
        11.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[107.0]);
}

#[test]
fn tensorcontract_fusion_output_recoupling_uses_su2_coefficients() {
    let rule = SU2FusionRule;
    let src_key = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let dst_key0 = src_key.clone();
    let dst_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let scalar_key = BlockKey::from(FusionTreeBlockKey::pair(
        empty_fusion_tree(),
        empty_fusion_tree(),
    ));
    let lhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1, 1], []),
        BlockStructure::packed_column_major_with_keys(4, [(src_key, vec![1, 1, 1, 1])]).unwrap(),
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        BlockStructure::packed_column_major_with_keys(0, [(scalar_key, vec![])]).unwrap(),
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1, 1], []),
        BlockStructure::packed_column_major_with_keys(
            4,
            [(dst_key0, vec![1, 1, 1, 1]), (dst_key1, vec![1, 1, 1, 1])],
        )
        .unwrap(),
    )
    .unwrap();
    let lhs = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![10.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![5.0], rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space).unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3])),
    )
    .unwrap();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].dst_block(), 0);
    assert_eq!(specs[1].dst_block(), 1);
    assert!((specs[0].coefficient() - 0.5).abs() < 1.0e-12);
    assert!((specs[1].coefficient() - 0.866_025_403_784_438_6).abs() < 1.0e-12);

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3])),
        2.0,
        3.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 53.0).abs() < 1.0e-12);
    assert!((dst.data()[1] - 92.602_540_378_443_86).abs() < 1.0e-12);
}

#[test]
fn tensorcontract_fusion_explicit_output_transform_materializes_canonical_dst() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1, 1], []);
    let lhs_keys = lhs_hom.fusion_tree_keys(&rule);
    assert_eq!(lhs_keys.len(), 2);
    let src_tree = lhs_keys
        .iter()
        .find(|key| key.codomain_tree().innerlines() == [SectorId::new(0), SectorId::new(1)])
        .expect("SU2 fixture should contain the reference source tree")
        .clone();
    let recoupled_tree = lhs_keys
        .iter()
        .find(|key| **key != src_tree)
        .expect("SU2 fixture should contain the recoupled output tree")
        .clone();
    let src_key = BlockKey::from(src_tree.clone());
    let dst_key0 = BlockKey::from(src_tree);
    let dst_key1 = BlockKey::from(recoupled_tree);
    let lhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        lhs_hom.clone(),
        BlockStructure::packed_column_major_with_keys(4, [(src_key, vec![1, 1, 1, 1])]).unwrap(),
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        &rule,
        [vec![]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        lhs_hom,
        BlockStructure::packed_column_major_with_keys(
            4,
            [(dst_key0, vec![1, 1, 1, 1]), (dst_key1, vec![1, 1, 1, 1])],
        )
        .unwrap(),
    )
    .unwrap();
    let lhs_canonical_space = lhs_space.clone();
    let canonical_dst_space = lhs_space.clone();
    let rhs_canonical_space = rhs_space.clone();
    let context_dst_space = dst_space.clone();
    let context_canonical_dst_space = canonical_dst_space.clone();
    let context_lhs_canonical_space = lhs_canonical_space.clone();
    let context_rhs_canonical_space = rhs_canonical_space.clone();
    let lhs = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![10.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![5.0], rhs_space).unwrap();
    let mut expected_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let mut explicit_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let mut canonical_dst = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![999.0],
        canonical_dst_space.clone(),
    )
    .unwrap();
    let mut expected_canonical_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![-77.0], canonical_dst_space)
            .unwrap();
    let mut lhs_canonical = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![123.0],
        lhs_canonical_space.clone(),
    )
    .unwrap();
    let mut rhs_canonical = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(
        vec![456.0],
        rhs_canonical_space.clone(),
    )
    .unwrap();
    let mut expected_lhs_canonical =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![0.0], lhs_canonical_space).unwrap();
    let mut expected_rhs_canonical =
        TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![0.0], rhs_canonical_space).unwrap();
    let axes = TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3]));
    let plan = tensorcontract_fusion_explicit_plan(
        &rule,
        explicit_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(plan.canonical_dst_nout(), 4);
    assert_eq!(plan.canonical_dst_nin(), 0);
    assert_eq!(plan.canonical_axes().lhs_contracting_axes(), &[]);
    assert_eq!(plan.canonical_axes().rhs_contracting_axes(), &[]);
    assert_eq!(plan.canonical_axes().output_axes(), &[0, 1, 2, 3]);
    assert_eq!(
        plan.output_transform(),
        &TreeTransformOperationKey::permute([0, 2, 1, 3], Vec::<usize>::new())
    );

    let alpha = 2.0;
    let beta = 3.0;
    let err = tensorcontract_fusion_explicit_plan_into(
        &rule,
        &plan,
        &mut expected_dst,
        &mut expected_lhs_canonical,
        &mut expected_rhs_canonical,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message: EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
        }
    );

    tree_pair_transform_into(
        &rule,
        plan.lhs_transform().clone(),
        &mut expected_lhs_canonical,
        &lhs,
        1.0,
        0.0,
    )
    .unwrap();
    tree_pair_transform_into(
        &rule,
        plan.rhs_transform().clone(),
        &mut expected_rhs_canonical,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut expected_canonical_dst,
        &expected_lhs_canonical,
        &expected_rhs_canonical,
        plan.canonical_axes().as_spec(),
        alpha,
        0.0,
    )
    .unwrap();
    tree_pair_transform_into(
        &rule,
        plan.output_transform().clone(),
        &mut expected_dst,
        &expected_canonical_dst,
        1.0,
        beta,
    )
    .unwrap();

    tensorcontract_fusion_explicit_plan_into_canonical_dst(
        &rule,
        &plan,
        &mut explicit_dst,
        &mut canonical_dst,
        &mut lhs_canonical,
        &mut rhs_canonical,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(canonical_dst.data(), expected_canonical_dst.data());
    assert_eq!(canonical_dst.data(), &[100.0]);
    for (&actual, &expected) in explicit_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }
    assert!((explicit_dst.data()[0] - 53.0).abs() < 1.0e-12);
    assert!((explicit_dst.data()[1] - 92.602_540_378_443_86).abs() < 1.0e-12);

    let mut automatic_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    tensorcontract_fusion_into(&rule, &mut automatic_dst, &lhs, &rhs, axes, alpha, beta).unwrap();
    for (&actual, &expected) in automatic_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }

    let mut context_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], context_dst_space)
            .unwrap();
    let mut context_canonical_dst = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![999.0],
        context_canonical_dst_space,
    )
    .unwrap();
    let mut context_lhs_canonical = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![123.0],
        context_lhs_canonical_space,
    )
    .unwrap();
    let mut context_rhs_canonical = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(
        vec![456.0],
        context_rhs_canonical_space,
    )
    .unwrap();
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    context
        .tensorcontract_fusion_explicit_plan_into_canonical_dst(
            &rule,
            &plan,
            &mut context_dst,
            &mut context_canonical_dst,
            &mut context_lhs_canonical,
            &mut context_rhs_canonical,
            &lhs,
            &rhs,
            alpha,
            beta,
        )
        .unwrap();

    assert_eq!(context_canonical_dst.data(), expected_canonical_dst.data());
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }
    assert_eq!(context.contract_cache().structure_len(), 1);
    assert_eq!(context.contract_cache().stats().structure_hits(), 0);
    assert_eq!(context.contract_cache().stats().structure_misses(), 1);

    let mut automatic_context_dst = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![1.0, 2.0],
        expected_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut automatic_context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    automatic_context
        .tensorcontract_fusion_into(
            &rule,
            &mut automatic_context_dst,
            &lhs,
            &rhs,
            axes,
            alpha,
            beta,
        )
        .unwrap();
    for (&actual, &expected) in automatic_context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }
    assert_eq!(automatic_context.contract_cache().structure_len(), 1);

    let mut beta_only_dst = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![7.0, 11.0],
        automatic_context_dst
            .fusion_space()
            .unwrap()
            .as_ref()
            .clone(),
    )
    .unwrap();
    automatic_context
        .tensorcontract_fusion_into(&rule, &mut beta_only_dst, &lhs, &rhs, axes, 0.0, 3.0)
        .unwrap();
    assert_eq!(beta_only_dst.data(), &[21.0, 33.0]);
    assert_eq!(
        automatic_context.contract_cache().stats().structure_hits(),
        1
    );
    assert_eq!(
        automatic_context
            .contract_cache()
            .stats()
            .structure_misses(),
        1
    );
}

#[test]
fn tensorcontract_fusion_su2_keeps_contracted_tree_basis_with_degeneracy() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1], [1, 1, 1]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);
    let lhs_keys = lhs_hom.fusion_tree_keys(&rule);
    let rhs_keys = rhs_hom.fusion_tree_keys(&rule);
    assert_eq!(lhs_keys.len(), 2);
    assert_eq!(rhs_keys.len(), 2);
    assert_ne!(
        lhs_keys[0].domain_tree().innerlines()[0],
        lhs_keys[1].domain_tree().innerlines()[0]
    );
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let dst_hom = FusionTreeHomSpace::from_sector_ids([1], [1]);
    let dst_keys = dst_hom.fusion_tree_keys(&rule);
    assert_eq!(dst_keys.len(), 1);
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        BlockStructure::packed_column_major_with_keys(2, [(dst_keys[0].clone(), vec![2, 2])])
            .unwrap(),
    )
    .unwrap();
    let lhs_data = (0..32).map(|index| 0.25 + index as f64).collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| 10.0 - 0.5 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = vec![1.0, -2.0, 3.0, -4.0];
    let lhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst.clone(), dst_space).unwrap();
    let axes = TensorContractAxisSpec::canonical(&[1, 2, 3], &[0, 1, 2]);
    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();

    assert_eq!(specs.len(), 2);
    for spec in &specs {
        let lhs_key = match lhs.structure().block(spec.lhs_block()).unwrap().key() {
            BlockKey::FusionTree(key) => key,
            BlockKey::Dense => panic!("expected lhs fusion-tree block"),
        };
        let rhs_key = match rhs.structure().block(spec.rhs_block()).unwrap().key() {
            BlockKey::FusionTree(key) => key,
            BlockKey::Dense => panic!("expected rhs fusion-tree block"),
        };
        assert_eq!(
            lhs_key.domain_tree().innerlines()[0],
            rhs_key.codomain_tree().innerlines()[0],
            "contracted SU2 tree basis must not cross-contract"
        );
    }

    let alpha = 1.25;
    let beta = -0.5;
    let mut expected = initial_dst
        .into_iter()
        .map(|value| beta * value)
        .collect::<Vec<_>>();
    for spec in &specs {
        let lhs_offset = lhs.structure().block(spec.lhs_block()).unwrap().offset();
        let rhs_offset = rhs.structure().block(spec.rhs_block()).unwrap().offset();
        for lhs_open in 0..2 {
            for rhs_open in 0..2 {
                let mut sum = 0.0;
                for a in 0..2 {
                    for b in 0..2 {
                        for c in 0..2 {
                            let lhs_index = lhs_offset + lhs_open + 2 * a + 4 * b + 8 * c;
                            let rhs_index = rhs_offset + a + 2 * b + 4 * c + 8 * rhs_open;
                            sum += lhs.data()[lhs_index] * rhs.data()[rhs_index];
                        }
                    }
                }
                expected[lhs_open + 2 * rhs_open] += alpha * spec.coefficient() * sum;
            }
        }
    }

    tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();

    for (&actual, expected) in dst.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
}

#[test]
fn contracted_fusion_tree_basis_matches_dual_u1_labels_and_flags() {
    let rule = U1FusionRule;
    let plus_two = U1Irrep::new(2).sector_id();
    let minus_two = U1Irrep::new(-2).sector_id();
    let lhs_domain = FusionTreeKey::new(
        [plus_two],
        Some(plus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    let rhs_codomain = FusionTreeKey::new(
        [minus_two],
        Some(minus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    assert!(contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &rhs_codomain
    ));

    let raw_rhs_codomain = FusionTreeKey::new(
        [plus_two],
        Some(plus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    assert!(!contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &raw_rhs_codomain
    ));

    let dual_flag_rhs_codomain = FusionTreeKey::new(
        [minus_two],
        Some(minus_two),
        [true],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    assert!(!contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &dual_flag_rhs_codomain
    ));
}

#[test]
fn tensorcontract_fusion_noncanonical_su2_absorbs_explicit_transform_sequence() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([1], [1, 1, 1]);
    let axes = TensorContractAxisSpec::canonical(&[0, 1, 2], &[1, 2, 3]);
    let output_axes = [0, 1];
    let lhs_canonical_hom = lhs_hom
        .permute(&rule, &[3], &[0, 1, 2])
        .expect("valid lhs canonical tree-pair transform");
    let rhs_canonical_hom = rhs_hom
        .permute(&rule, &[1, 2, 3], &[0])
        .expect("valid rhs canonical tree-pair transform");
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        &lhs_hom,
        &rhs_hom,
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        &output_axes,
        1,
    )
    .unwrap();

    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        lhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let lhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_canonical_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_canonical_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let lhs_data = (0..32)
        .map(|index| 1.0 + 0.125 * index as f64)
        .collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| -3.0 + 0.25 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = vec![2.0, -1.0, 4.0, -3.0];
    let initial_dst_for_explicit = initial_dst.clone();
    let initial_dst_for_context = initial_dst.clone();
    let initial_dst_for_context_replay = initial_dst.clone();
    let lhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut direct_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst.clone(), dst_space.clone())
            .unwrap();
    let mut expected_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst, dst_space.clone()).unwrap();
    let mut lhs_canonical = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(
        vec![0.0; lhs_canonical_space.required_len().unwrap()],
        lhs_canonical_space.clone(),
    )
    .unwrap();
    let mut rhs_canonical = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(
        vec![0.0; rhs_canonical_space.required_len().unwrap()],
        rhs_canonical_space.clone(),
    )
    .unwrap();
    let plan = tensorcontract_fusion_explicit_plan(
        &rule,
        direct_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(
        plan.lhs_transform(),
        &TreeTransformOperationKey::permute([3], [0, 1, 2])
    );
    assert_eq!(
        plan.rhs_transform(),
        &TreeTransformOperationKey::permute([1, 2, 3], [0])
    );
    assert_eq!(plan.canonical_dst_nout(), 1);
    assert_eq!(plan.canonical_dst_nin(), 1);
    assert_eq!(plan.canonical_axes().lhs_contracting_axes(), &[1, 2, 3]);
    assert_eq!(plan.canonical_axes().rhs_contracting_axes(), &[0, 1, 2]);
    assert_eq!(plan.canonical_axes().output_axes(), &[0, 1]);
    assert_eq!(
        plan.output_transform(),
        &TreeTransformOperationKey::permute([0], [1])
    );

    let err = tensorcontract_fusion_block_specs(
        &rule,
        direct_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap_err();
    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );

    tree_pair_transform_into(
        &rule,
        TreeTransformOperationKey::permute([3], [0, 1, 2]),
        &mut lhs_canonical,
        &lhs,
        1.0,
        0.0,
    )
    .unwrap();
    tree_pair_transform_into(
        &rule,
        TreeTransformOperationKey::permute([1, 2, 3], [0]),
        &mut rhs_canonical,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    let canonical_specs = tensorcontract_fusion_block_specs(
        &rule,
        expected_dst.fusion_space().unwrap(),
        lhs_canonical.fusion_space().unwrap(),
        rhs_canonical.fusion_space().unwrap(),
        TensorContractAxisSpec::canonical(&[1, 2, 3], &[0, 1, 2]),
    )
    .unwrap();
    assert_eq!(canonical_specs.len(), 2);

    let alpha = -1.5;
    let beta = 0.25;
    tensorcontract_fusion_into(&rule, &mut direct_dst, &lhs, &rhs, axes, alpha, beta).unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut expected_dst,
        &lhs_canonical,
        &rhs_canonical,
        TensorContractAxisSpec::canonical(&[1, 2, 3], &[0, 1, 2]),
        alpha,
        beta,
    )
    .unwrap();

    for (&actual, &expected) in direct_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }

    let mut explicit_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_explicit,
        dst_space.clone(),
    )
    .unwrap();
    tensorcontract_fusion_explicit_plan_into(
        &rule,
        &plan,
        &mut explicit_dst,
        &mut lhs_canonical,
        &mut rhs_canonical,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();
    for (&actual, &expected) in explicit_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }

    let mut context_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst_for_context, dst_space)
            .unwrap();
    let mut context_lhs_canonical = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(
        vec![0.0; lhs_canonical_space.required_len().unwrap()],
        lhs_canonical_space,
    )
    .unwrap();
    let mut context_rhs_canonical = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(
        vec![0.0; rhs_canonical_space.required_len().unwrap()],
        rhs_canonical_space,
    )
    .unwrap();
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    context
        .tensorcontract_fusion_explicit_plan_into(
            &rule,
            &plan,
            &mut context_dst,
            &mut context_lhs_canonical,
            &mut context_rhs_canonical,
            &lhs,
            &rhs,
            alpha,
            beta,
        )
        .unwrap();
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    assert_eq!(context.tree_context().cache().plan_len(), 2);
    assert_eq!(context.tree_context().cache().structure_len(), 2);
    assert_eq!(context.contract_cache().structure_len(), 1);
    assert_eq!(context.contract_cache().stats().structure_hits(), 0);
    assert_eq!(context.contract_cache().stats().structure_misses(), 1);

    context_dst
        .data_mut()
        .copy_from_slice(&initial_dst_for_context_replay);
    context
        .tensorcontract_fusion_explicit_plan_into(
            &rule,
            &plan,
            &mut context_dst,
            &mut context_lhs_canonical,
            &mut context_rhs_canonical,
            &lhs,
            &rhs,
            alpha,
            beta,
        )
        .unwrap();
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    assert_eq!(context.tree_context().cache().plan_len(), 2);
    assert_eq!(context.tree_context().cache().structure_len(), 2);
    assert_eq!(context.contract_cache().structure_len(), 1);
    assert_eq!(context.contract_cache().stats().structure_hits(), 1);
    assert_eq!(context.contract_cache().stats().structure_misses(), 1);

    let mut automatic_context_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context_replay.clone(),
        context_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut automatic_context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    automatic_context
        .tensorcontract_fusion_into(
            &rule,
            &mut automatic_context_dst,
            &lhs,
            &rhs,
            axes,
            alpha,
            beta,
        )
        .unwrap();
    for (&actual, &expected) in automatic_context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    assert_eq!(automatic_context.tree_context().cache().plan_len(), 0);
    assert_eq!(automatic_context.tree_context().cache().structure_len(), 0);
    assert_eq!(automatic_context.fusion_execution_plan_cache_len(), 1);
    assert_eq!(
        automatic_context.fusion_execution_plan_cache_replay_hits(),
        0
    );
    assert_eq!(automatic_context.fusion_execution_plan_cache_compiles(), 1);
    // The automatic dynamic path uses the TensorKit-style canonical
    // fusion-block pack/GEMM/scatter executor, not the generic dense
    // TensorContractStructure cache.
    assert_eq!(automatic_context.contract_cache().structure_len(), 0);
    assert_eq!(
        automatic_context.contract_cache().stats().structure_hits(),
        0
    );
    assert_eq!(
        automatic_context
            .contract_cache()
            .stats()
            .structure_misses(),
        0
    );
    assert_eq!(automatic_context.fusion_block_contract_cache_len(), 0);
    assert_eq!(automatic_context.fusion_block_contract_cache_hits(), 0);
    assert_eq!(automatic_context.fusion_block_contract_cache_misses(), 0);

    let mut split_backend_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context_replay.clone(),
        context_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut tree_backend = HostTensorOperations;
    let mut tree_workspace = TreeTransformWorkspace::default();
    let mut contract_backend = DenseTreeTransformOperations::default_executor();
    let mut contract_workspace = TensorContractWorkspace::default();
    tensorcontract_fusion_into_with_backends(
        &mut tree_backend,
        &mut tree_workspace,
        &mut contract_backend,
        &mut contract_workspace,
        &rule,
        &mut split_backend_dst,
        &lhs,
        &rhs,
        axes,
        alpha,
        beta,
    )
    .unwrap();
    for (&actual, &expected) in split_backend_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }

    automatic_context_dst
        .data_mut()
        .copy_from_slice(&initial_dst_for_context_replay);
    automatic_context
        .tensorcontract_fusion_into(
            &rule,
            &mut automatic_context_dst,
            &lhs,
            &rhs,
            axes,
            alpha,
            beta,
        )
        .unwrap();
    for (&actual, &expected) in automatic_context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    assert_eq!(automatic_context.tree_context().cache().plan_len(), 0);
    assert_eq!(automatic_context.tree_context().cache().structure_len(), 0);
    assert_eq!(automatic_context.fusion_execution_plan_cache_len(), 1);
    assert_eq!(
        automatic_context.fusion_execution_plan_cache_replay_hits(),
        1
    );
    assert_eq!(automatic_context.fusion_execution_plan_cache_compiles(), 1);
    assert_eq!(automatic_context.contract_cache().structure_len(), 0);
    assert_eq!(
        automatic_context.contract_cache().stats().structure_hits(),
        0
    );
    assert_eq!(
        automatic_context
            .contract_cache()
            .stats()
            .structure_misses(),
        0
    );
    assert_eq!(automatic_context.fusion_block_contract_cache_len(), 0);
    assert_eq!(automatic_context.fusion_block_contract_cache_hits(), 0);
    assert_eq!(automatic_context.fusion_block_contract_cache_misses(), 0);
}

#[test]
fn tensorcontract_fusion_execution_plan_cache_distinguishes_block_structure() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([1], [1, 1, 1]);
    let axes = TensorContractAxisSpec::canonical(&[0, 1, 2], &[1, 2, 3]);
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        &lhs_hom,
        &rhs_hom,
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        &[0, 1],
        1,
    )
    .unwrap();
    let lhs_keys = lhs_hom.fusion_tree_keys(&rule);
    let make_lhs_space = |case_index: usize| {
        let dense_space = TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap();
        let structure = match case_index {
            0 => BlockStructure::packed_column_major_with_keys(
                4,
                lhs_keys.iter().cloned().map(|key| (key, vec![2, 2, 2, 2])),
            )
            .unwrap(),
            1 => {
                let mut blocks = lhs_keys
                    .iter()
                    .cloned()
                    .map(|key| (key, vec![2, 2, 2, 2]))
                    .collect::<Vec<_>>();
                blocks.reverse();
                BlockStructure::packed_column_major_with_keys(4, blocks).unwrap()
            }
            2 => BlockStructure::from_blocks_with_rank(
                4,
                lhs_keys
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, key)| {
                        BlockSpec::with_key(
                            BlockKey::from(key),
                            vec![2, 2, 2, 2],
                            vec![1, 3, 6, 12],
                            23 * index,
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap(),
            _ => unreachable!("test only has three lhs block-structure cases"),
        };
        FusionTensorMapSpace::new(dense_space, lhs_hom.clone(), structure).unwrap()
    };
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let rhs_data = (0..32)
        .map(|index| -2.0 + 0.125 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = vec![0.5, -1.0, 2.0, -4.0];
    let alpha = 0.75;
    let beta = -0.25;
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    for case_index in 0..3 {
        let lhs_space = make_lhs_space(case_index);
        let lhs_data = (0..lhs_space.required_len().unwrap())
            .map(|index| 1.0 + 0.0625 * index as f64)
            .collect::<Vec<_>>();
        let lhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
        let rhs =
            TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(rhs_data.clone(), rhs_space.clone())
                .unwrap();
        let mut expected = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space.clone(),
        )
        .unwrap();
        tensorcontract_fusion_into(&rule, &mut expected, &lhs, &rhs, axes, alpha, beta).unwrap();

        let mut actual = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space.clone(),
        )
        .unwrap();
        context
            .tensorcontract_fusion_into(&rule, &mut actual, &lhs, &rhs, axes, alpha, beta)
            .unwrap();
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (actual - expected).abs() < 1.0e-10,
                "actual {actual} expected {expected}"
            );
        }

        assert_eq!(context.fusion_execution_plan_cache_len(), case_index + 1);
        assert_eq!(
            context.fusion_execution_plan_cache_compiles(),
            case_index + 1
        );
        assert_eq!(context.fusion_execution_plan_cache_replay_hits(), 0);
    }
}

#[test]
fn tensorcontract_fusion_execution_plan_cache_distinguishes_output_axes() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([1], [1, 1, 1]);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        lhs_hom.clone(),
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom.clone(),
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let lhs_data = (0..32)
        .map(|index| 1.0 + 0.125 * index as f64)
        .collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| -3.0 + 0.25 * index as f64)
        .collect::<Vec<_>>();
    let lhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let initial_dst = vec![2.0, -1.0, 4.0, -3.0];
    let alpha = -1.5;
    let beta = 0.25;
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    for (case_index, output_axes) in [[0usize, 1usize], [1usize, 0usize]].into_iter().enumerate() {
        let axes = TensorContractAxisSpec::new(
            &[0, 1, 2],
            &[1, 2, 3],
            AxisPermutation::from_axes(&output_axes),
        );
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            &rule,
            &lhs_hom,
            &rhs_hom,
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &output_axes,
            1,
        )
        .unwrap();
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
            dst_hom,
            &rule,
            [vec![2, 2]],
        )
        .unwrap();
        let mut expected = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space.clone(),
        )
        .unwrap();
        let mut actual =
            TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst.clone(), dst_space)
                .unwrap();

        tensorcontract_fusion_into(&rule, &mut expected, &lhs, &rhs, axes, alpha, beta).unwrap();
        context
            .tensorcontract_fusion_into(&rule, &mut actual, &lhs, &rhs, axes, alpha, beta)
            .unwrap();
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (actual - expected).abs() < 1.0e-10,
                "actual {actual} expected {expected}"
            );
        }
        assert_eq!(context.fusion_execution_plan_cache_len(), case_index + 1);
        assert_eq!(
            context.fusion_execution_plan_cache_compiles(),
            case_index + 1
        );
        assert_eq!(context.fusion_execution_plan_cache_replay_hits(), 0);
    }
}

#[test]
fn tensorcontract_fusion_execution_plan_cache_distinguishes_source_conjugation() {
    let rule = SU2FusionRule;
    let mut context = TensorContractFusionExecutionContext::<
        Complex64,
        TreeTransformBuiltinRuleCacheKey,
    >::default();
    let alpha = Complex64::new(-1.5, 0.25);
    let beta = Complex64::new(0.25, -0.125);
    let initial_dst = vec![
        Complex64::new(2.0, -1.0),
        Complex64::new(-1.0, 0.5),
        Complex64::new(4.0, 2.0),
        Complex64::new(-3.0, -0.25),
    ];

    for (case_index, (lhs_hom, rhs_hom, lhs_conjugate, rhs_conjugate)) in [
        (
            su2_three_to_one_homspace(false, false),
            su2_one_to_three_homspace(false, false),
            false,
            false,
        ),
        (
            su2_three_to_one_homspace(false, false),
            su2_one_to_three_homspace(true, true),
            true,
            false,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let axes = TensorContractAxisSpec::canonical_with_conjugation(
            &[0, 1, 2],
            &[1, 2, 3],
            lhs_conjugate,
            rhs_conjugate,
        );
        let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
            lhs_hom.clone(),
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
            rhs_hom.clone(),
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
        let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
        let effective_lhs_hom = if lhs_conjugate {
            lhs_adjoint_space.homspace()
        } else {
            &lhs_hom
        };
        let effective_rhs_hom = if rhs_conjugate {
            rhs_adjoint_space.homspace()
        } else {
            &rhs_hom
        };
        let lowered_lhs_axes = maybe_adjoint_axes::<3, 1>(&[0, 1, 2], lhs_conjugate);
        let lowered_rhs_axes = maybe_adjoint_axes::<1, 3>(&[1, 2, 3], rhs_conjugate);
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            &rule,
            effective_lhs_hom,
            effective_rhs_hom,
            lowered_lhs_axes.as_slice(),
            lowered_rhs_axes.as_slice(),
            &[0, 1],
            1,
        )
        .unwrap();
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
            dst_hom,
            &rule,
            [vec![2, 2]],
        )
        .unwrap();
        let lhs_data = (0..32)
            .map(|index| Complex64::new(1.0 + 0.125 * index as f64, -0.5 + 0.0625 * index as f64))
            .collect::<Vec<_>>();
        let rhs_data = (0..32)
            .map(|index| Complex64::new(-3.0 + 0.25 * index as f64, 0.75 - 0.03125 * index as f64))
            .collect::<Vec<_>>();
        let lhs =
            TensorMap::<Complex64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
        let rhs =
            TensorMap::<Complex64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
        let mut expected = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space.clone(),
        )
        .unwrap();
        let mut actual = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
            initial_dst.clone(),
            dst_space,
        )
        .unwrap();

        tensorcontract_fusion_into(&rule, &mut expected, &lhs, &rhs, axes, alpha, beta).unwrap();
        context
            .tensorcontract_fusion_into(&rule, &mut actual, &lhs, &rhs, axes, alpha, beta)
            .unwrap();
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (actual - expected).norm() < 1.0e-10,
                "actual {actual} expected {expected}"
            );
        }
        assert_eq!(context.fusion_execution_plan_cache_len(), case_index + 1);
        assert_eq!(
            context.fusion_execution_plan_cache_compiles(),
            case_index + 1
        );
        assert_eq!(context.fusion_execution_plan_cache_replay_hits(), 0);
    }
}

#[test]
fn tensorcontract_fusion_noncanonical_su2_lhs_adjoint_explicit_plan_matches_reference_sequence() {
    assert_noncanonical_su2_adjoint_explicit_plan_matches_reference_sequence(
        su2_three_to_one_homspace(false, false),
        su2_one_to_three_homspace(true, true),
        true,
        false,
    );
}

#[test]
fn tensorcontract_fusion_noncanonical_su2_rhs_adjoint_explicit_plan_matches_reference_sequence() {
    assert_noncanonical_su2_adjoint_explicit_plan_matches_reference_sequence(
        su2_three_to_one_homspace(false, true),
        su2_one_to_three_homspace(false, true),
        false,
        true,
    );
}

#[test]
fn tensorcontract_fusion_noncanonical_su2_both_adjoint_explicit_plan_matches_reference_sequence() {
    assert_noncanonical_su2_adjoint_explicit_plan_matches_reference_sequence(
        su2_three_to_one_homspace(false, false),
        su2_one_to_three_homspace(false, false),
        true,
        true,
    );
}

fn assert_noncanonical_su2_adjoint_explicit_plan_matches_reference_sequence(
    lhs_hom: FusionTreeHomSpace,
    rhs_hom: FusionTreeHomSpace,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
) {
    let rule = SU2FusionRule;
    let source_lhs_contracting_axes = [0, 1, 2];
    let source_rhs_contracting_axes = [1, 2, 3];
    let axes = TensorContractAxisSpec::canonical_with_conjugation(
        &source_lhs_contracting_axes,
        &source_rhs_contracting_axes,
        lhs_conjugate,
        rhs_conjugate,
    );

    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        lhs_hom.clone(),
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom.clone(),
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let lhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&lhs_space).unwrap();
    let rhs_adjoint_space = crate::lowering::adjoint_fusion_space_view(&rhs_space).unwrap();
    let lowered_lhs_axes = maybe_adjoint_axes::<3, 1>(&source_lhs_contracting_axes, lhs_conjugate);
    let lowered_rhs_axes = maybe_adjoint_axes::<1, 3>(&source_rhs_contracting_axes, rhs_conjugate);
    let lowered_lhs_open_axes = complement_axes(4, &lowered_lhs_axes);
    let lowered_rhs_open_axes = complement_axes(4, &lowered_rhs_axes);
    let effective_lhs_hom = if lhs_conjugate {
        lhs_adjoint_space.homspace()
    } else {
        &lhs_hom
    };
    let effective_rhs_hom = if rhs_conjugate {
        rhs_adjoint_space.homspace()
    } else {
        &rhs_hom
    };
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        effective_lhs_hom,
        effective_rhs_hom,
        lowered_lhs_axes.as_slice(),
        lowered_rhs_axes.as_slice(),
        &[0, 1],
        1,
    )
    .unwrap();
    let lhs_canonical_hom = effective_lhs_hom
        .permute(
            &rule,
            lowered_lhs_open_axes.as_slice(),
            lowered_lhs_axes.as_slice(),
        )
        .unwrap();
    let rhs_canonical_hom = effective_rhs_hom
        .permute(
            &rule,
            lowered_rhs_axes.as_slice(),
            lowered_rhs_open_axes.as_slice(),
        )
        .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let lhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_canonical_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_canonical_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let lhs_data = (0..32)
        .map(|index| Complex64::new(1.0 + 0.125 * index as f64, -0.5 + 0.0625 * index as f64))
        .collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| Complex64::new(-3.0 + 0.25 * index as f64, 0.75 - 0.03125 * index as f64))
        .collect::<Vec<_>>();
    let initial_dst = vec![
        Complex64::new(2.0, -1.0),
        Complex64::new(-1.0, 0.5),
        Complex64::new(4.0, 2.0),
        Complex64::new(-3.0, -0.25),
    ];
    let initial_dst_for_context = initial_dst.clone();
    let alpha = Complex64::new(-1.5, 0.25);
    let beta = Complex64::new(0.25, -0.125);
    let lhs =
        TensorMap::<Complex64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs =
        TensorMap::<Complex64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut expected_dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        initial_dst.clone(),
        dst_space.clone(),
    )
    .unwrap();
    let mut lhs_canonical = TensorMap::<Complex64, 1, 3>::from_vec_with_fusion_space(
        vec![Complex64::zero(); lhs_canonical_space.required_len().unwrap()],
        lhs_canonical_space.clone(),
    )
    .unwrap();
    let mut rhs_canonical = TensorMap::<Complex64, 3, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); rhs_canonical_space.required_len().unwrap()],
        rhs_canonical_space.clone(),
    )
    .unwrap();

    tensoradd_fusion_into(
        &rule,
        &mut lhs_canonical,
        &lhs,
        TreeTransformOperationKey::permute([3], [0, 1, 2]),
        lhs_conjugate,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    tensoradd_fusion_into(
        &rule,
        &mut rhs_canonical,
        &rhs,
        TreeTransformOperationKey::permute([1, 2, 3], [0]),
        rhs_conjugate,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut expected_dst,
        &lhs_canonical,
        &rhs_canonical,
        TensorContractAxisSpec::canonical(&[1, 2, 3], &[0, 1, 2]),
        alpha,
        beta,
    )
    .unwrap();

    let mut explicit_dst =
        TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(initial_dst, dst_space).unwrap();
    let plan = tensorcontract_fusion_explicit_plan(
        &rule,
        explicit_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(plan.lhs_source_conjugate(), lhs_conjugate);
    assert_eq!(plan.rhs_source_conjugate(), rhs_conjugate);
    assert_eq!(plan.canonical_axes().lhs_contracting_axes(), &[1, 2, 3]);
    assert_eq!(plan.canonical_axes().rhs_contracting_axes(), &[0, 1, 2]);

    let mut explicit_lhs_canonical = TensorMap::<Complex64, 1, 3>::from_vec_with_fusion_space(
        vec![Complex64::zero(); lhs_canonical_space.required_len().unwrap()],
        lhs_canonical_space,
    )
    .unwrap();
    let mut explicit_rhs_canonical = TensorMap::<Complex64, 3, 1>::from_vec_with_fusion_space(
        vec![Complex64::zero(); rhs_canonical_space.required_len().unwrap()],
        rhs_canonical_space,
    )
    .unwrap();
    tensorcontract_fusion_explicit_plan_into(
        &rule,
        &plan,
        &mut explicit_dst,
        &mut explicit_lhs_canonical,
        &mut explicit_rhs_canonical,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();

    for (&actual, &expected) in explicit_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).norm() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }

    let mut context_dst = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        initial_dst_for_context.clone(),
        expected_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<
        Complex64,
        TreeTransformBuiltinRuleCacheKey,
    >::default();
    context
        .tensorcontract_fusion_into(&rule, &mut context_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).norm() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    let expects_dynamic_replay = !(lhs_conjugate && rhs_conjugate);
    if expects_dynamic_replay {
        assert_eq!(context.fusion_execution_plan_cache_len(), 1);
        assert_eq!(context.fusion_execution_plan_cache_replay_hits(), 0);
        assert_eq!(context.fusion_execution_plan_cache_compiles(), 1);
    } else {
        assert_eq!(context.fusion_execution_plan_cache_len(), 0);
    }

    context_dst
        .data_mut()
        .copy_from_slice(&initial_dst_for_context);
    context
        .tensorcontract_fusion_into(&rule, &mut context_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).norm() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
    if expects_dynamic_replay {
        assert_eq!(context.fusion_execution_plan_cache_len(), 1);
        assert_eq!(context.fusion_execution_plan_cache_replay_hits(), 1);
        assert_eq!(context.fusion_execution_plan_cache_compiles(), 1);
    } else {
        assert_eq!(context.fusion_execution_plan_cache_len(), 0);
    }
}

fn maybe_adjoint_axes<const NOUT: usize, const NIN: usize>(
    axes: &[usize],
    source_conjugate: bool,
) -> Vec<usize> {
    if source_conjugate {
        axes.iter()
            .map(|&axis| crate::lowering::adjoint_tensor_axis(NOUT, NIN, axis).unwrap())
            .collect()
    } else {
        axes.to_vec()
    }
}

fn complement_axes(rank: usize, axes: &[usize]) -> Vec<usize> {
    (0..rank).filter(|axis| !axes.contains(axis)).collect()
}

fn su2_three_to_one_homspace(codomain_dual: bool, domain_dual: bool) -> FusionTreeHomSpace {
    let half = SectorId::new(1);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([half], codomain_dual),
            SectorLeg::new([half], codomain_dual),
            SectorLeg::new([half], codomain_dual),
        ]),
        FusionProductSpace::new([SectorLeg::new([half], domain_dual)]),
    )
}

fn su2_one_to_three_homspace(codomain_dual: bool, domain_dual: bool) -> FusionTreeHomSpace {
    let half = SectorId::new(1);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([half], codomain_dual)]),
        FusionProductSpace::new([
            SectorLeg::new([half], domain_dual),
            SectorLeg::new([half], domain_dual),
            SectorLeg::new([half], domain_dual),
        ]),
    )
}

#[test]
fn tensorcontract_fusion_product_noncanonical_absorbs_explicit_transform() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([], []);
    let scalar_key = BlockKey::from(rhs_hom.fusion_tree_keys(&rule)[0].clone());
    let rhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        rhs_hom,
        BlockStructure::packed_column_major_with_keys(0, [(scalar_key, vec![])]).unwrap(),
    )
    .unwrap();
    let lhs_canonical_hom = src_space
        .homspace()
        .permute(&rule, &[0, 1, 2], &[])
        .unwrap();
    let lhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 0>::from_dims([1, 1, 1], []).unwrap(),
        lhs_canonical_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let canonical_dst_space = lhs_canonical_space.clone();
    let rhs_canonical_space = rhs_space.clone();
    let lhs_data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)];
    let rhs_data = vec![Complex64::new(2.0, 0.5)];
    let initial_dst = vec![Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)];
    let lhs = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(lhs_data.clone(), src_space)
        .unwrap();
    let rhs =
        TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut dst = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        initial_dst.clone(),
        dst_space.clone(),
    )
    .unwrap();
    let mut expected_dst =
        TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(initial_dst, dst_space).unwrap();
    let axes = TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[1, 0, 2]));
    let err = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap_err();
    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );

    let plan = tensorcontract_fusion_explicit_plan(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    let mut lhs_canonical = TensorMap::<Complex64, 3, 0>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); lhs_canonical_space.required_len().unwrap()],
        lhs_canonical_space,
    )
    .unwrap();
    let mut rhs_canonical = TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); rhs_canonical_space.required_len().unwrap()],
        rhs_canonical_space,
    )
    .unwrap();
    let mut canonical_dst = TensorMap::<Complex64, 3, 0>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); canonical_dst_space.required_len().unwrap()],
        canonical_dst_space,
    )
    .unwrap();
    let alpha = Complex64::new(2.0, 0.0);
    let beta = Complex64::new(3.0, 0.0);
    tree_pair_transform_into(
        &rule,
        plan.lhs_transform().clone(),
        &mut lhs_canonical,
        &lhs,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    tree_pair_transform_into(
        &rule,
        plan.rhs_transform().clone(),
        &mut rhs_canonical,
        &rhs,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut canonical_dst,
        &lhs_canonical,
        &rhs_canonical,
        plan.canonical_axes().as_spec(),
        alpha,
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    tree_pair_transform_into(
        &rule,
        plan.output_transform().clone(),
        &mut expected_dst,
        &canonical_dst,
        Complex64::new(1.0, 0.0),
        beta,
    )
    .unwrap();

    tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();

    for (&actual, &expected) in dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }

    let mut context_dst = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)],
        dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<Complex64, _>::default();
    context
        .tensorcontract_fusion_into(&rule, &mut context_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
    for (&actual, &expected) in context_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }
    // This path uses canonical fusion-block pack/GEMM/scatter directly; the
    // generic TensorContractStructure cache is only used by dense/block-spec
    // contraction paths.
    assert_eq!(context.contract_cache().structure_len(), 0);
    assert_eq!(context.fusion_execution_plan_cache_len(), 1);
    assert_eq!(context.fusion_execution_plan_cache_replay_hits(), 0);
    assert_eq!(context.fusion_execution_plan_cache_compiles(), 1);
    assert_eq!(context.fusion_block_contract_cache_len(), 0);
    assert_eq!(context.fusion_block_contract_cache_hits(), 0);
    assert_eq!(context.fusion_block_contract_cache_misses(), 0);

    context_dst
        .data_mut()
        .copy_from_slice(&[Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)]);
    context
        .tensorcontract_fusion_into(&rule, &mut context_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap();
    assert_eq!(context.contract_cache().stats().structure_hits(), 0);
    assert_eq!(context.fusion_execution_plan_cache_len(), 1);
    assert_eq!(context.fusion_execution_plan_cache_replay_hits(), 1);
    assert_eq!(context.fusion_execution_plan_cache_compiles(), 1);
    assert_eq!(context.fusion_block_contract_cache_len(), 0);
    assert_eq!(context.fusion_block_contract_cache_hits(), 0);
    assert_eq!(context.fusion_block_contract_cache_misses(), 0);
}

#[test]
fn tensorcontract_fusion_product_fz2_u1_su2_contracts_component_channels_with_su2_recoupling() {
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let left_sector =
        |parity, charge| left_rule.encode_sector(parity, U1Irrep::new(charge).sector_id());
    let sector = |parity, charge, twice_spin| {
        rule.encode_sector(
            left_sector(parity, charge),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };
    let a = sector(odd, 1, 1);
    let b = sector(odd, -1, 1);
    let c0 = sector(even, 0, 0);
    let c1 = sector(even, 0, 2);

    let lhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([a], false), SectorLeg::new([b], false)]),
        FusionProductSpace::new([SectorLeg::new([c0, c1], false)]),
    );
    let rhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([c0, c1], false),
            SectorLeg::new([a], false),
            SectorLeg::new([b], false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([a], false),
            SectorLeg::new([a], false),
            SectorLeg::new([b], false),
            SectorLeg::new([b], false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
        lhs_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 0>::from_dims([1, 1, 1], []).unwrap(),
        rhs_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        dst_hom,
        &rule,
        [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
    )
    .unwrap();
    let lhs = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 3, 0>::from_vec_with_fusion_space(
        vec![Complex64::new(-2.0, 0.5), Complex64::new(4.0, 3.0)],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 4, 0>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)],
        dst_space,
    )
    .unwrap();
    let alpha = Complex64::new(2.0, -0.25);
    let beta = Complex64::new(-1.0, 0.5);
    let axes = TensorContractAxisSpec::new(&[2], &[0], AxisPermutation::from_axes(&[0, 2, 1, 3]));

    tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();

    let expected = [
        Complex64::new(-29.12579386826373, -0.7876587736527441),
        Complex64::new(21.57892465101803, 3.5376587736527494),
    ];
    for (&actual, &expected) in dst.data().iter().zip(&expected) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }
}
