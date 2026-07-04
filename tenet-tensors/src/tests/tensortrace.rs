use super::*;
use tenet_core::Trivial;

#[test]
fn tensortrace_default_host_api_accepts_custom_host_storage() {
    let src_space = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
    let src = test_host_read_tensor_map(vec![1.0_f64, 2.0, 3.0, 4.0], src_space);
    let dst_space = TensorMapSpace::<0, 0>::from_dims([], []).unwrap();
    let mut dst = test_host_tensor_map(vec![10.0_f64], dst_space);

    tensortrace_into(
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[40.0]);
}

#[test]
fn tensortrace_repeated_dense_axis_matches_tensoroperations_trace_lowering() {
    let src_space = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
    let src: TensorMap<f64, 1, 1, Trivial> =
        TensorMap::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], src_space).unwrap();
    let dst_space = TensorMapSpace::<0, 0>::from_dims([], []).unwrap();
    let mut dst: TensorMap<f64, 0, 0, Trivial> =
        TensorMap::from_vec(vec![10.0_f64], dst_space).unwrap();

    tensortrace_into(
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[40.0]);
}

#[test]
fn tensortrace_with_conjugation_applies_dense_source_conj() {
    let src_space = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
    let src: TensorMap<Complex64, 1, 1, Trivial> = TensorMap::from_vec(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 10.0),
            Complex64::new(3.0, -4.0),
            Complex64::new(4.0, -2.0),
        ],
        src_space,
    )
    .unwrap();
    let dst_space = TensorMapSpace::<0, 0>::from_dims([], []).unwrap();
    let mut dst: TensorMap<Complex64, 0, 0, Trivial> =
        TensorMap::from_vec(vec![Complex64::new(0.0, 0.0)], dst_space).unwrap();

    tensortrace_into(
        &mut dst,
        &src,
        TensorTraceAxisSpec::new_with_conjugation(&[], &[0], &[1], true),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(dst.data(), &[Complex64::new(5.0, 1.0)]);
}

#[test]
fn tensortrace_with_conjugation_keeps_requested_dense_output_axes() {
    let src_space = TensorMapSpace::<2, 2>::from_dims([2, 2], [2, 2]).unwrap();
    let src: TensorMap<Complex64, 2, 2, Trivial> = TensorMap::from_vec(
        (1..=16)
            .map(|value| Complex64::new(value as f64, value as f64))
            .collect(),
        src_space,
    )
    .unwrap();
    let dst_space = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
    let mut dst: TensorMap<Complex64, 1, 1, Trivial> =
        TensorMap::from_vec(vec![Complex64::new(0.0, 0.0); 4], dst_space).unwrap();

    tensortrace_into(
        &mut dst,
        &src,
        TensorTraceAxisSpec::new_with_conjugation(&[3, 0], &[1], &[2], true),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(8.0, -8.0),
            Complex64::new(24.0, -24.0),
            Complex64::new(10.0, -10.0),
            Complex64::new(26.0, -26.0),
        ]
    );
}

#[test]
fn tensortrace_keeps_output_axes_and_sums_diagonal_trace_axes() {
    let src_space = TensorMapSpace::<2, 1>::from_dims([2, 3], [3]).unwrap();
    let src: TensorMap<f64, 2, 1, Trivial> =
        TensorMap::from_vec((0..18).map(|value| value as f64).collect(), src_space).unwrap();
    let dst_space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    let mut dst: TensorMap<f64, 1, 0, Trivial> =
        TensorMap::from_vec(vec![100.0_f64, 200.0], dst_space).unwrap();

    tensortrace_into(
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[0], &[1], &[2]),
        1.5,
        0.5,
    )
    .unwrap();

    assert_eq!(dst.data(), &[86.0, 140.5]);
}

#[test]
fn tensortrace_structure_replays_without_recompiling() {
    let src_space = TensorMapSpace::<2, 1>::from_dims([2, 3], [3]).unwrap();
    let src1: TensorMap<f64, 2, 1, Trivial> = TensorMap::from_vec(
        (0..18).map(|value| value as f64).collect(),
        src_space.clone(),
    )
    .unwrap();
    let src2: TensorMap<f64, 2, 1, Trivial> =
        TensorMap::from_vec((100..118).map(|value| value as f64).collect(), src_space).unwrap();
    let dst_space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    let mut dst1: TensorMap<f64, 1, 0, Trivial> =
        TensorMap::from_vec(vec![0.0_f64, 0.0], dst_space.clone()).unwrap();
    let mut dst2: TensorMap<f64, 1, 0, Trivial> =
        TensorMap::from_vec(vec![0.0_f64, 0.0], dst_space).unwrap();
    let structure =
        tensortrace_structure(&dst1, &src1, TensorTraceAxisSpec::new(&[0], &[1], &[2])).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    tensortrace_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst1,
        &src1,
        1.0,
        0.0,
    )
    .unwrap();
    tensortrace_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst2,
        &src2,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst1.data(), &[24.0, 27.0]);
    assert_eq!(dst2.data(), &[324.0, 327.0]);
}

#[test]
fn tensortrace_rejects_invalid_axis_sets_at_compile_time() {
    let src_space = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
    let src: TensorMap<f64, 1, 1, Trivial> =
        TensorMap::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], src_space).unwrap();
    let dst_space = TensorMapSpace::<0, 0>::from_dims([], []).unwrap();
    let dst: TensorMap<f64, 0, 0, Trivial> = TensorMap::from_vec(vec![0.0_f64], dst_space).unwrap();

    let err =
        tensortrace_structure(&dst, &src, TensorTraceAxisSpec::new(&[], &[0], &[0])).unwrap_err();
    assert!(matches!(err, OperationError::InvalidAxisSet { .. }));

    let err =
        tensortrace_structure(&dst, &src, TensorTraceAxisSpec::new(&[], &[0], &[])).unwrap_err();
    assert_eq!(
        err,
        OperationError::TraceAxisCountMismatch { lhs: 1, rhs: 0 }
    );
}

#[test]
fn categorical_adjoint_view_swaps_homspace_and_block_layout_without_dualizing() {
    let cod_sector = SectorId::new(1);
    let dom_sector = SectorId::new(2);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(cod_sector, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(dom_sector, 3)], false)]),
    );
    let structure = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(
            fusion_tree_test_key([1], [2], 3, [false], [false]),
            vec![2, 3],
            vec![1, 2],
            5,
        )
        .unwrap()],
    )
    .unwrap();
    let space = FusionTensorMapSpace::new(
        TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
        hom,
        structure,
    )
    .unwrap();

    let adjoint = crate::lowering::adjoint_fusion_space_view(&space).unwrap();

    assert_eq!(
        adjoint.homspace().codomain().legs(),
        space.homspace().domain().legs()
    );
    assert_eq!(
        adjoint.homspace().domain().legs(),
        space.homspace().codomain().legs()
    );
    assert_eq!(adjoint.dense_space().codomain().dims(), &[3]);
    assert_eq!(adjoint.dense_space().domain().dims(), &[2]);
    let block = adjoint.subblock_structure().block(0).unwrap();
    assert_eq!(block.shape(), &[3, 2]);
    assert_eq!(block.strides(), &[2, 1]);
    assert_eq!(block.offset(), 5);
    let key = expect_tree_key(block.key());
    assert_eq!(key.codomain_tree().uncoupled(), &[dom_sector]);
    assert_eq!(key.domain_tree().uncoupled(), &[cod_sector]);
}

#[test]
fn categorical_adjoint_view_does_not_dualize_nonselfdual_u1_stored_legs() {
    let q1 = U1Irrep::new(1).sector_id();
    let q2 = U1Irrep::new(2).sector_id();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(q1, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(q2, 1)], false)]),
    );
    let space = FusionTensorMapSpace::new(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        hom,
        BlockStructure::empty(2),
    )
    .unwrap();

    let adjoint = crate::lowering::adjoint_fusion_space_view(&space).unwrap();

    assert_eq!(
        adjoint.homspace().codomain().legs(),
        &[SectorLeg::new([(q2, 1)], false)]
    );
    assert_eq!(
        adjoint.homspace().domain().legs(),
        &[SectorLeg::new([(q1, 1)], false)]
    );
    assert_ne!(
        adjoint.homspace().codomain().legs(),
        &[SectorLeg::new([(U1Irrep::new(-2).sector_id(), 1)], true)]
    );
}

#[test]
fn categorical_adjoint_axis_map_matches_tensorkit_adjointtensorindex() {
    assert_eq!(crate::lowering::adjoint_tensor_axis(2, 1, 0).unwrap(), 1);
    assert_eq!(crate::lowering::adjoint_tensor_axis(2, 1, 1).unwrap(), 2);
    assert_eq!(crate::lowering::adjoint_tensor_axis(2, 1, 2).unwrap(), 0);

    let err = crate::lowering::adjoint_tensor_axis(2, 1, 3).unwrap_err();
    assert!(matches!(err, OperationError::InvalidAxisSet { .. }));
}

#[test]
fn tensortrace_rejects_block_sparse_trace_until_categorical_trace_is_implemented() {
    let src_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let src_structure = packed_fixture_structure(
        2,
        [
            (BlockKey::ordinal(0), vec![1, 1]),
            (BlockKey::ordinal(1), vec![1, 1]),
        ],
    )
    .unwrap();
    let src: TensorMap<f64, 1, 1, Trivial> =
        TensorMap::from_vec_with_structure(vec![2.0, 3.0], src_space, src_structure).unwrap();
    let dst_space = TensorMapSpace::<0, 0>::from_dims([], []).unwrap();
    let dst_structure = packed_fixture_structure(
        0,
        [
            (BlockKey::ordinal(0), vec![]),
            (BlockKey::ordinal(1), vec![]),
        ],
    )
    .unwrap();
    let dst: TensorMap<f64, 0, 0, Trivial> =
        TensorMap::from_vec_with_structure(vec![0.0, 0.0], dst_space, dst_structure).unwrap();

    let err =
        tensortrace_structure(&dst, &src, TensorTraceAxisSpec::new(&[], &[0], &[1])).unwrap_err();
    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message: "block-sparse tensortrace enumeration is not implemented yet"
        }
    );
}

#[test]
fn tensortrace_fusion_fermion_parity_matches_tensorkit_supertrace() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        src_hom,
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let src: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![2.0, 3.0], src_space).unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        dst_hom,
        &rule,
        [vec![]],
    )
    .unwrap();
    let mut dst: TensorMap<f64, 0, 0> =
        TensorMap::from_vec_with_fusion_space(vec![0.0], dst_space).unwrap();

    let structure =
        tensortrace_fusion_structure(&rule, &dst, &src, TensorTraceAxisSpec::new(&[], &[0], &[1]))
            .unwrap();
    assert_eq!(structure.terms().len(), 2);

    tensortrace_fusion_execute_with(
        &mut HostTensorOperations,
        &mut HostAllocator::default(),
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[-1.0]);
}

#[test]
fn tensortrace_fusion_default_host_api_accepts_custom_host_storage() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        src_hom,
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let src = test_host_read_fusion_tensor_map(vec![2.0_f64, 3.0], src_space);
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        dst_hom,
        &rule,
        [vec![]],
    )
    .unwrap();
    let mut dst = test_host_fusion_tensor_map(vec![0.0_f64], dst_space);

    tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[-1.0]);
}

#[test]
fn tensortrace_fusion_fermion_supertrace_uses_degeneracy_diagonals() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 2), (odd, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 2), (odd, 2)], false)]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        src_hom,
        &rule,
        [vec![2, 2], vec![2, 2]],
    )
    .unwrap();
    let src: TensorMap<f64, 1, 1> = TensorMap::from_vec_with_fusion_space(
        vec![
            1.0, 2.0, 3.0, 4.0, // even block, column-major diagonal 1 + 4
            5.0, 6.0, 7.0, 8.0, // odd block, column-major diagonal 5 + 8
        ],
        src_space,
    )
    .unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        dst_hom,
        &rule,
        [vec![]],
    )
    .unwrap();
    let mut dst: TensorMap<f64, 0, 0> =
        TensorMap::from_vec_with_fusion_space(vec![10.0], dst_space).unwrap();

    tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[14.0]);
}

#[test]
fn tensortrace_fusion_with_conjugation_lowers_lazy_adjoint_supertrace() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        src_hom,
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let src: TensorMap<Complex64, 1, 1> = TensorMap::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, 4.0)],
        src_space,
    )
    .unwrap();
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        dst_hom,
        &rule,
        [vec![]],
    )
    .unwrap();
    let mut dst: TensorMap<Complex64, 0, 0> =
        TensorMap::from_vec_with_fusion_space(vec![Complex64::new(10.0, 20.0)], dst_space).unwrap();

    tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new_with_conjugation(&[], &[0], &[1], true),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(dst.data(), &[Complex64::new(-1.0, 3.0)]);
}

#[test]
fn tensortrace_fusion_scales_destination_once_for_multiple_source_terms() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        src_hom,
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let src: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![2.0, 3.0], src_space).unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        dst_hom,
        &rule,
        [vec![]],
    )
    .unwrap();
    let mut dst: TensorMap<f64, 0, 0> =
        TensorMap::from_vec_with_fusion_space(vec![10.0], dst_space).unwrap();

    tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        1.0,
        5.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[49.0]);
}

#[test]
fn tensortrace_fusion_fermion_open_output_matches_tensorkit_oracle() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([1, 1], [1, 1]).unwrap(),
        src_hom,
        &rule,
        (0..8).map(|_| vec![1, 1, 1, 1]),
    )
    .unwrap();
    let src: TensorMap<f64, 2, 2> = TensorMap::from_vec_with_fusion_space(
        (1..=8).map(|value| value as f64).collect(),
        src_space,
    )
    .unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        dst_hom,
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let mut dst: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![10.0, 20.0], dst_space).unwrap();

    tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        // TensorKit 1-based: output ((1,), (3,)), trace ((2,), (4,)).
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[16.0, 62.0]);
}

#[test]
fn tensortrace_fusion_fermion_two_trace_pairs_match_tensorkit_oracle() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([1, 1], [1, 1]).unwrap(),
        src_hom,
        &rule,
        (0..8).map(|_| vec![1, 1, 1, 1]),
    )
    .unwrap();
    let src: TensorMap<f64, 2, 2> = TensorMap::from_vec_with_fusion_space(
        (1..=8).map(|value| value as f64).collect(),
        src_space,
    )
    .unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        dst_hom,
        &rule,
        [vec![]],
    )
    .unwrap();
    let mut dst: TensorMap<f64, 0, 0> =
        TensorMap::from_vec_with_fusion_space(vec![5.0], dst_space).unwrap();

    tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        // TensorKit 1-based: output ((), ()), trace ((1, 2), (3, 4)).
        TensorTraceAxisSpec::new(&[], &[0, 1], &[2, 3]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[-1.0]);
}

#[test]
fn tensortrace_fusion_rejects_nonsymmetric_braiding_like_tensorkit_tensortrace() {
    let rule = UniqueAnyonicRule;
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([(1, 1)], [(1, 1)]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let src: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![2.0], src_space).unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        &rule,
        [vec![]],
    )
    .unwrap();
    let dst: TensorMap<f64, 0, 0> =
        TensorMap::from_vec_with_fusion_space(vec![0.0], dst_space).unwrap();

    let err =
        tensortrace_fusion_structure(&rule, &dst, &src, TensorTraceAxisSpec::new(&[], &[0], &[1]))
            .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message: "fusion tensortrace requires symmetric braiding"
        }
    );
}

#[test]
fn tensortrace_fusion_su2_includes_quantum_dimension_factor() {
    let rule = SU2FusionRule;
    let half = SU2Irrep::from_twice_spin(1).sector_id();
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(half, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(half, 1)], false)]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        src_hom,
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let src: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![7.0], src_space).unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        dst_hom,
        &rule,
        [vec![]],
    )
    .unwrap();
    let mut dst: TensorMap<f64, 0, 0> =
        TensorMap::from_vec_with_fusion_space(vec![0.0], dst_space).unwrap();

    tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        1.0,
        0.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 14.0).abs() < 1.0e-12);
}

#[test]
fn plain_tensortrace_rejects_one_block_fusion_tensor_instead_of_dense_trace() {
    let rule = SU2FusionRule;
    let half = SU2Irrep::from_twice_spin(1).sector_id();
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(half, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(half, 1)], false)]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        src_hom,
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let src: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![7.0], src_space).unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        dst_hom,
        &rule,
        [vec![]],
    )
    .unwrap();
    let mut dst: TensorMap<f64, 0, 0> =
        TensorMap::from_vec_with_fusion_space(vec![0.0], dst_space).unwrap();

    let err = tensortrace_into(
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        1.0,
        0.0,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message:
                "plain tensortrace does not lower fusion-tree blocks; use tensortrace_fusion_*"
        }
    );
    assert_eq!(dst.data(), &[0.0]);
}
