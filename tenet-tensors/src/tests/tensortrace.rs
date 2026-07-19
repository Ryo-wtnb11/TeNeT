use super::*;
use tenet_core::Trivial;

fn lowered_u1_dynamic_space(
    homspace: FusionTreeHomSpace,
    shapes: Vec<Vec<usize>>,
) -> BoundDynamicFusionMapSpace<U1FusionRule> {
    BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        std::sync::Arc::new(U1FusionRule),
        homspace,
        shapes,
    )
    .unwrap()
}

#[derive(Clone, Debug)]
struct ReportedLenHostStorage<T> {
    data: Vec<T>,
    reported_len: usize,
}

impl<T> TensorStorage<T> for ReportedLenHostStorage<T> {
    fn len(&self) -> usize {
        self.reported_len
    }

    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl<T> HostReadableStorage<T> for ReportedLenHostStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T> HostWritableStorage<T> for ReportedLenHostStorage<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

#[test]
fn public_dynamic_fz2_trace_validates_exact_extents_before_mutation() {
    struct Case {
        name: &'static str,
        dst: Vec<f64>,
        src: Vec<f64>,
        error: Option<(usize, usize)>,
        expected_dst: Vec<f64>,
    }

    let provider = std::sync::Arc::new(FermionParityFusionRule);
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let traced_leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let src_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([traced_leg()]),
        FusionProductSpace::new([traced_leg()]),
    );
    let src_shapes = vec![vec![1, 1]; src_homspace.fusion_tree_keys(provider.as_ref()).len()];
    let src_space = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
        std::sync::Arc::clone(&provider),
        src_homspace,
        src_shapes,
    )
    .unwrap();
    let dst_space = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
        provider,
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
        [Vec::<usize>::new()],
    )
    .unwrap();
    let cases = [
        Case {
            name: "oversized destination",
            dst: vec![10.0, 999.0],
            src: vec![2.0, 3.0],
            error: Some((1, 2)),
            expected_dst: vec![10.0, 999.0],
        },
        Case {
            name: "short destination",
            dst: vec![],
            src: vec![2.0, 3.0],
            error: Some((1, 0)),
            expected_dst: vec![],
        },
        Case {
            name: "oversized source",
            dst: vec![10.0],
            src: vec![2.0, 3.0, 999.0],
            error: Some((2, 3)),
            expected_dst: vec![10.0],
        },
        Case {
            name: "short source",
            dst: vec![10.0],
            src: vec![2.0],
            error: Some((2, 1)),
            expected_dst: vec![10.0],
        },
        Case {
            name: "exact destination and source",
            dst: vec![10.0],
            src: vec![2.0, 3.0],
            error: None,
            expected_dst: vec![4.0],
        },
    ];

    for case in cases {
        let mut dst = case.dst;
        let result = tensortrace_fusion_dyn_into(
            &dst_space,
            &mut dst,
            &src_space,
            &case.src,
            TensorTraceAxisSpec::new(&[], &[0], &[1]),
            1.0,
            0.5,
        );
        match case.error {
            Some((expected, actual)) => {
                assert_eq!(
                    result,
                    Err(OperationError::ElementCountMismatch { expected, actual }),
                    "{}",
                    case.name
                );
                assert_eq!(dst, case.expected_dst, "{}", case.name);
            }
            None => {
                result.unwrap();
                assert_eq!(dst, case.expected_dst, "{}", case.name);
            }
        }
    }
}

fn strided_fz2_identity_trace_spaces() -> (FusionTensorMapSpace<1, 1>, FusionTensorMapSpace<1, 1>) {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 2), (odd, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
    );
    let dense_space = TensorMapSpace::<1, 1>::from_dims([4], [2]).unwrap();
    let canonical = FusionTensorMapSpace::from_degeneracy_shapes(
        dense_space.clone(),
        homspace.clone(),
        &rule,
        [vec![2, 1], vec![2, 1]],
    )
    .unwrap();
    assert_eq!(canonical.subblock_structure().block_count(), 2);

    let destination = BlockStructure::from_blocks_with_rank(
        2,
        (0..2)
            .map(|index| {
                let block = canonical.subblock_structure().block(index).unwrap();
                BlockSpec::with_key(
                    block.key().clone(),
                    block.shape().to_vec(),
                    vec![2, 4],
                    index * 4,
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let source_block = canonical.subblock_structure().block(0).unwrap();
    let source = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(
            source_block.key().clone(),
            source_block.shape().to_vec(),
            vec![1, 2],
            0,
        )
        .unwrap()],
    )
    .unwrap();

    (
        FusionTensorMapSpace::new_unbound(dense_space.clone(), homspace.clone(), destination)
            .unwrap()
            .try_bind_rule(&rule)
            .unwrap(),
        FusionTensorMapSpace::new_unbound(dense_space, homspace, source)
            .unwrap()
            .try_bind_rule(&rule)
            .unwrap(),
    )
}

fn relayout_fz2_destination(
    reference: &FusionTensorMapSpace<1, 1>,
    strides: [[usize; 2]; 2],
    offsets: [usize; 2],
) -> FusionTensorMapSpace<1, 1> {
    let structure = BlockStructure::from_blocks_with_rank(
        2,
        (0..2)
            .map(|index| {
                let block = reference.subblock_structure().block(index).unwrap();
                BlockSpec::with_key(
                    block.key().clone(),
                    block.shape().to_vec(),
                    strides[index].to_vec(),
                    offsets[index],
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    FusionTensorMapSpace::new_unbound(
        reference.dense_space().clone(),
        reference.homspace().clone(),
        structure,
    )
    .unwrap()
    .try_bind_rule(&FermionParityFusionRule)
    .unwrap()
}

#[test]
fn typed_fz2_trace_scales_each_logical_destination_without_touching_padding() {
    let rule = FermionParityFusionRule;
    let (dst_space, src_space) = strided_fz2_identity_trace_spaces();
    let src: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![1.0, 2.0], src_space).unwrap();
    let cases = [
        (0.0, vec![1.0, 99.0, 2.0, 88.0, 0.0, 77.0, 0.0]),
        (0.5, vec![6.0, 99.0, 12.0, 88.0, 15.0, 77.0, 20.0]),
        (1.0, vec![11.0, 99.0, 22.0, 88.0, 30.0, 77.0, 40.0]),
    ];

    for (beta, expected) in cases {
        let mut dst: TensorMap<f64, 1, 1> = TensorMap::from_vec_with_fusion_space(
            vec![10.0, 99.0, 20.0, 88.0, 30.0, 77.0, 40.0],
            dst_space.clone(),
        )
        .unwrap();

        tensortrace_fusion_into(
            &rule,
            &mut dst,
            &src,
            TensorTraceAxisSpec::new(&[0, 1], &[], &[]),
            1.0,
            beta,
        )
        .unwrap();

        // What: beta owns every logical destination block exactly once, while
        // storage gaps remain outside tensortrace semantics.
        assert_eq!(dst.data(), expected, "beta={beta}");
    }
}

#[test]
fn typed_fz2_trace_beta_zero_overwrites_logical_nan_without_reading_padding() {
    let rule = FermionParityFusionRule;
    let (dst_space, src_space) = strided_fz2_identity_trace_spaces();
    let src: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![1.0, 2.0], src_space).unwrap();
    let mut dst: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![f64::NAN; 7], dst_space).unwrap();

    tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[0, 1], &[], &[]),
        1.0,
        0.0,
    )
    .unwrap();

    // What: beta=0 is an overwrite for active and no-term logical elements,
    // while physical padding remains outside the operation.
    assert_eq!(dst.data()[0], 1.0);
    assert_eq!(dst.data()[2], 2.0);
    assert_eq!(dst.data()[4], 0.0);
    assert_eq!(dst.data()[6], 0.0);
    assert!(dst.data()[1].is_nan());
    assert!(dst.data()[3].is_nan());
    assert!(dst.data()[5].is_nan());
}

#[test]
fn typed_fz2_zero_term_trace_scales_all_logical_destinations() {
    let rule = FermionParityFusionRule;
    let (dst_space, _) = strided_fz2_identity_trace_spaces();
    let src_space = FusionTensorMapSpace::new_unbound(
        dst_space.dense_space().clone(),
        dst_space.homspace().clone(),
        BlockStructure::empty(2),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let src: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(Vec::new(), src_space).unwrap();
    let mut dst: TensorMap<f64, 1, 1> = TensorMap::from_vec_with_fusion_space(
        vec![10.0, 99.0, 20.0, 88.0, 30.0, 77.0, 40.0],
        dst_space,
    )
    .unwrap();

    tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[0, 1], &[], &[]),
        1.0,
        0.5,
    )
    .unwrap();

    // What: no active source terms still leaves beta responsible for every
    // logical destination, but never for padding between those layouts.
    assert_eq!(dst.data(), &[5.0, 99.0, 10.0, 88.0, 15.0, 77.0, 20.0]);
}

#[test]
fn typed_fz2_multiple_trace_terms_and_inactive_block_scale_beta_once() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let src_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let src_key_count = src_homspace.fusion_tree_keys(&rule).len();
    let canonical_src = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([2, 2], [2, 2]).unwrap(),
        src_homspace.clone(),
        &rule,
        vec![vec![1, 1, 1, 1]; src_key_count],
    )
    .unwrap();
    let dst_homspace = src_homspace.select(&rule, &[0], &[2]).unwrap();
    let dst_key_count = dst_homspace.fusion_tree_keys(&rule).len();
    let canonical_dst = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_homspace.clone(),
        &rule,
        vec![vec![1, 1]; dst_key_count],
    )
    .unwrap();
    let full_structure = TensorTraceFusionStructure::<f64>::compile_fusion_spaces(
        &rule,
        &canonical_dst,
        &canonical_src,
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
    )
    .unwrap();
    let active_block = (0..canonical_dst.subblock_structure().block_count())
        .find(|&dst_block| {
            full_structure
                .terms()
                .iter()
                .filter(|term| term.dst_block() == dst_block)
                .count()
                == 2
        })
        .expect("fixture needs two traced sectors for one external block");
    let selected_src_blocks = full_structure
        .terms()
        .iter()
        .filter(|term| term.dst_block() == active_block)
        .map(TensorTraceFusionStructureTerm::src_block)
        .collect::<Vec<_>>();
    assert_eq!(selected_src_blocks.len(), 2);
    let src_structure = packed_fixture_structure(
        4,
        selected_src_blocks.iter().map(|&src_block| {
            let block = canonical_src.subblock_structure().block(src_block).unwrap();
            (block.key().clone(), block.shape().to_vec())
        }),
    )
    .unwrap();
    let src_space = FusionTensorMapSpace::new_unbound(
        canonical_src.dense_space().clone(),
        src_homspace,
        src_structure,
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let src_data = selected_src_blocks
        .iter()
        .map(|&src_block| {
            let coefficient = full_structure
                .terms()
                .iter()
                .find(|term| term.src_block() == src_block && term.dst_block() == active_block)
                .unwrap()
                .coefficient();
            if *coefficient > 0.0 {
                2.0
            } else {
                3.0
            }
        })
        .collect::<Vec<_>>();
    let src: TensorMap<f64, 2, 2> =
        TensorMap::from_vec_with_fusion_space(src_data, src_space).unwrap();

    let dst_structure = BlockStructure::from_blocks_with_rank(
        2,
        (0..canonical_dst.subblock_structure().block_count())
            .map(|dst_block| {
                let block = canonical_dst.subblock_structure().block(dst_block).unwrap();
                BlockSpec::with_key(
                    block.key().clone(),
                    block.shape().to_vec(),
                    vec![1, 1],
                    dst_block * 2,
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::new_unbound(
        canonical_dst.dense_space().clone(),
        dst_homspace,
        dst_structure,
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let mut dst: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![10.0, 99.0, 30.0], dst_space).unwrap();
    let structure = tensortrace_fusion_structure(
        &rule,
        &dst,
        &src,
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
    )
    .unwrap();
    assert_eq!(structure.terms().len(), 2);
    assert!(structure
        .terms()
        .iter()
        .all(|term| term.dst_block() == active_block));
    let mut coefficients = structure
        .terms()
        .iter()
        .map(|term| *term.coefficient())
        .collect::<Vec<_>>();
    coefficients.sort_by(f64::total_cmp);
    assert_eq!(coefficients, vec![-1.0, 1.0]);
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    tensortrace_fusion_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.5,
    )
    .unwrap();

    let expected = if active_block == 0 {
        vec![4.0, 99.0, 15.0]
    } else {
        vec![5.0, 99.0, 14.0]
    };
    // What: beta is applied once before the even/odd supertrace terms both
    // accumulate into the active scalar, and once to the no-term scalar.
    assert_eq!(dst.data(), expected);
}

#[test]
fn typed_fz2_trace_validates_destination_layout_injectivity() {
    let rule = FermionParityFusionRule;
    let (canonical_dst, one_term_src) = strided_fz2_identity_trace_spaces();
    let aliased_dst = relayout_fz2_destination(&canonical_dst, [[2, 4], [2, 4]], [0, 0]);
    let self_overlapping_dst = relayout_fz2_destination(&canonical_dst, [[0, 4], [2, 4]], [0, 4]);
    let interleaved_dst = relayout_fz2_destination(&canonical_dst, [[2, 4], [2, 4]], [0, 1]);
    let full_src = FusionTensorMapSpace::from_degeneracy_shapes(
        canonical_dst.dense_space().clone(),
        canonical_dst.homspace().clone(),
        &rule,
        [vec![2, 1], vec![2, 1]],
    )
    .unwrap();
    let empty_src = FusionTensorMapSpace::new_unbound(
        canonical_dst.dense_space().clone(),
        canonical_dst.homspace().clone(),
        BlockStructure::empty(2),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let expected = OperationError::InvalidArgument {
        message: "tensor trace destination layouts overlap",
    };

    for (name, src) in [
        ("active-active", &full_src),
        ("active-inactive", &one_term_src),
        ("inactive-inactive", &empty_src),
    ] {
        assert_eq!(
            TensorTraceFusionStructure::<f64>::compile_fusion_spaces(
                &rule,
                &aliased_dst,
                src,
                TensorTraceAxisSpec::new(&[0, 1], &[], &[]),
            )
            .unwrap_err(),
            expected,
            "{name}"
        );
    }
    assert_eq!(
        TensorTraceFusionStructure::<f64>::compile_fusion_spaces(
            &rule,
            &self_overlapping_dst,
            &one_term_src,
            TensorTraceAxisSpec::new(&[0, 1], &[], &[]),
        )
        .unwrap_err(),
        expected,
        "self-overlap"
    );

    // What: intersecting bounds are valid when exact logical footprints remain
    // disjoint, including coupled-style even/odd interleaving.
    TensorTraceFusionStructure::<f64>::compile_fusion_spaces(
        &rule,
        &interleaved_dst,
        &full_src,
        TensorTraceAxisSpec::new(&[0, 1], &[], &[]),
    )
    .unwrap();
}

#[test]
fn plain_trace_rejects_self_overlapping_destination_layout() {
    let dst = BlockStructure::from_blocks_with_rank(
        1,
        vec![BlockSpec::new(vec![2], vec![0], 0).unwrap()],
    )
    .unwrap();
    let src = BlockStructure::trivial(&[2]).unwrap();

    let error = TensorTraceStructure::compile_structures(
        &dst,
        &src,
        TensorTraceAxisSpec::new(&[0], &[], &[]),
    )
    .unwrap_err();

    // What: the plain compiler cannot publish a replay descriptor that writes
    // more than one logical output to the same destination element.
    assert_eq!(
        error,
        OperationError::InvalidArgument {
            message: "tensor trace destination layouts overlap",
        }
    );
}

#[test]
fn plain_strided_trace_updates_logical_output_without_touching_padding() {
    let src_space = TensorMapSpace::<2, 1>::from_dims([2, 2], [2]).unwrap();
    let src: TensorMap<f64, 2, 1, Trivial> =
        TensorMap::from_vec((1..=8).map(f64::from).collect(), src_space).unwrap();
    let dst_space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    let dst_structure = BlockStructure::from_blocks_with_rank(
        1,
        vec![BlockSpec::new(vec![2], vec![2], 0).unwrap()],
    )
    .unwrap();
    let mut dst: TensorMap<f64, 1, 0, Trivial> =
        TensorMap::from_vec_with_structure(vec![10.0, 999.0, 20.0], dst_space, dst_structure)
            .unwrap();

    tensortrace_into(
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[0], &[1], &[2]),
        1.0,
        1.0,
    )
    .unwrap();

    // What: successful plain replay updates only the strided logical view;
    // exact-length storage padding is not part of the tensortrace output.
    assert_eq!(dst.data(), &[18.0, 999.0, 30.0]);
}

#[test]
fn plain_trace_validates_host_slice_extents_before_mutation() {
    struct Case {
        name: &'static str,
        dst: Vec<f64>,
        src: Vec<f64>,
        expected: usize,
        actual: usize,
    }

    let dst_structure = BlockStructure::from_blocks_with_rank(
        1,
        vec![BlockSpec::new(vec![2], vec![2], 0).unwrap()],
    )
    .unwrap();
    let src_structure = BlockStructure::trivial(&[2]).unwrap();
    let cases = [
        Case {
            name: "short destination",
            dst: vec![10.0, 99.0],
            src: vec![1.0, 2.0],
            expected: 3,
            actual: 2,
        },
        Case {
            name: "oversized destination",
            dst: vec![10.0, 99.0, 20.0, 77.0],
            src: vec![1.0, 2.0],
            expected: 3,
            actual: 4,
        },
        Case {
            name: "short source",
            dst: vec![10.0, 99.0, 20.0],
            src: vec![1.0],
            expected: 2,
            actual: 1,
        },
        Case {
            name: "oversized source",
            dst: vec![10.0, 99.0, 20.0],
            src: vec![1.0, 2.0, 3.0],
            expected: 2,
            actual: 3,
        },
    ];

    for case in cases {
        let initial_dst = case.dst.clone();
        let mut dst: TensorMap<f64, 1, 0, Trivial, ReportedLenHostStorage<f64>> =
            TensorMap::from_storage_with_structure(
                ReportedLenHostStorage {
                    data: case.dst,
                    reported_len: 3,
                },
                TensorMapSpace::from_dims([2], []).unwrap(),
                dst_structure.clone(),
            )
            .unwrap();
        let src: TensorMap<f64, 1, 0, Trivial, ReportedLenHostStorage<f64>> =
            TensorMap::from_storage_with_structure(
                ReportedLenHostStorage {
                    data: case.src,
                    reported_len: 2,
                },
                TensorMapSpace::from_dims([2], []).unwrap(),
                src_structure.clone(),
            )
            .unwrap();
        let structure =
            tensortrace_structure(&dst, &src, TensorTraceAxisSpec::new(&[0], &[], &[])).unwrap();
        let mut backend = HostTensorOperations;
        let mut allocator = HostAllocator::default();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tensortrace_execute_with(
                &mut backend,
                &mut allocator,
                &structure,
                &mut dst,
                &src,
                1.0,
                0.5,
            )
        }))
        .expect("plain trace extent mismatch must return an error, not panic");

        // What: destination length has precedence, and either mismatch leaves
        // even the first reachable logical destination unchanged.
        assert_eq!(
            result,
            Err(OperationError::ElementCountMismatch {
                expected: case.expected,
                actual: case.actual,
            }),
            "{}",
            case.name
        );
        assert_eq!(dst.data(), initial_dst, "{}", case.name);
    }
}

#[test]
fn typed_fz2_trace_validates_host_slice_extents_before_mutation() {
    struct Case {
        name: &'static str,
        dst: Vec<f64>,
        src: Vec<f64>,
        expected: usize,
        actual: usize,
    }

    let rule = FermionParityFusionRule;
    let (dst_space, src_space) = strided_fz2_identity_trace_spaces();
    let exact_dst = vec![10.0, 99.0, 20.0, 88.0, 30.0, 77.0, 40.0];
    let cases = [
        Case {
            name: "short destination",
            dst: exact_dst[..6].to_vec(),
            src: vec![1.0, 2.0],
            expected: 7,
            actual: 6,
        },
        Case {
            name: "oversized destination",
            dst: [exact_dst.as_slice(), &[55.0]].concat(),
            src: vec![1.0, 2.0],
            expected: 7,
            actual: 8,
        },
        Case {
            name: "short source",
            dst: exact_dst.clone(),
            src: vec![1.0],
            expected: 2,
            actual: 1,
        },
        Case {
            name: "oversized source",
            dst: exact_dst,
            src: vec![1.0, 2.0, 3.0],
            expected: 2,
            actual: 3,
        },
    ];

    for case in cases {
        let initial_dst = case.dst.clone();
        let mut dst: TensorMap<f64, 1, 1, Trivial, ReportedLenHostStorage<f64>> =
            TensorMap::from_storage_with_fusion_space(
                ReportedLenHostStorage {
                    data: case.dst,
                    reported_len: 7,
                },
                dst_space.clone(),
            )
            .unwrap();
        let src: TensorMap<f64, 1, 1, Trivial, ReportedLenHostStorage<f64>> =
            TensorMap::from_storage_with_fusion_space(
                ReportedLenHostStorage {
                    data: case.src,
                    reported_len: 2,
                },
                src_space.clone(),
            )
            .unwrap();

        let result = tensortrace_fusion_into(
            &rule,
            &mut dst,
            &src,
            TensorTraceAxisSpec::new(&[0, 1], &[], &[]),
            1.0,
            0.5,
        );

        // What: typed replay enforces the same exact destination-first extent
        // contract as dynamic replay before beta can reach a logical layout.
        assert_eq!(
            result,
            Err(OperationError::ElementCountMismatch {
                expected: case.expected,
                actual: case.actual,
            }),
            "{}",
            case.name
        );
        assert_eq!(dst.data(), initial_dst, "{}", case.name);
    }
}

#[test]
fn lowered_dynamic_trace_select_reports_u1_min_without_mutating_destination() {
    // What: checked output selection returns the exact U1 MIN dual failure
    // before trace terms, result-layout publication, or destination scaling.
    let min = U1Irrep::new(i32::MIN).sector_id();
    let zero = U1Irrep::new(0).sector_id();
    let src = lowered_u1_dynamic_space(
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(min, 1)], false)]),
            FusionProductSpace::new([]),
        ),
        Vec::new(),
    );
    let dst = lowered_u1_dynamic_space(
        FusionTreeHomSpace::new(
            FusionProductSpace::new([]),
            FusionProductSpace::new([SectorLeg::new([(zero, 1)], false)]),
        ),
        vec![vec![1]],
    );
    let mut dst_data = vec![37.0_f64];
    crate::contract::reset_scratch_publication_observations();
    crate::tensortrace::reset_trace_transform_invocations();

    let error = tensortrace_fusion_dyn_into_checked(
        &dst,
        &mut dst_data,
        &src,
        &[],
        TensorTraceAxisSpec::new(&[0], &[], &[]),
        2.0,
        3.0,
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(tenet_core::FusionAlgebraError::U1DualOverflow {
            charge: i32::MIN
        },))
    );
    assert_eq!(dst_data, [37.0]);
    assert_eq!(
        crate::contract::scratch_publication_observations(),
        (0, 0, 0, 0)
    );
    assert_eq!(crate::tensortrace::take_trace_transform_invocations(), 0);
}

#[test]
fn lowered_dynamic_trace_outward_leg_reports_u1_min_without_mutating_destination() {
    // What: checked trace-pair orientation returns the exact U1 MIN dual
    // failure before trace terms, result publication, or destination scaling.
    let min = U1Irrep::new(i32::MIN).sector_id();
    let zero = U1Irrep::new(0).sector_id();
    let src = lowered_u1_dynamic_space(
        FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(zero, 1)], false),
                SectorLeg::new([(min, 1)], false),
            ]),
            FusionProductSpace::new([]),
        ),
        Vec::new(),
    );
    let dst = lowered_u1_dynamic_space(
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
        vec![Vec::new()],
    );
    let mut dst_data = vec![41.0_f64];
    crate::contract::reset_scratch_publication_observations();
    crate::tensortrace::reset_trace_transform_invocations();

    let error = tensortrace_fusion_dyn_into_checked(
        &dst,
        &mut dst_data,
        &src,
        &[],
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        2.0,
        3.0,
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(tenet_core::FusionAlgebraError::U1DualOverflow {
            charge: i32::MIN
        },))
    );
    assert_eq!(dst_data, [41.0]);
    assert_eq!(
        crate::contract::scratch_publication_observations(),
        (0, 0, 0, 0)
    );
    assert_eq!(crate::tensortrace::take_trace_transform_invocations(), 0);
}

#[test]
fn lowered_dynamic_trace_invalid_axis_precedes_u1_min_dual_failure() {
    // What: trace axis validation retains its structural precedence before
    // checked selection can attempt to dual a U1 MIN source leg.
    let min = U1Irrep::new(i32::MIN).sector_id();
    let zero = U1Irrep::new(0).sector_id();
    let src = lowered_u1_dynamic_space(
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(min, 1)], false)]),
            FusionProductSpace::new([]),
        ),
        Vec::new(),
    );
    let dst = lowered_u1_dynamic_space(
        FusionTreeHomSpace::new(
            FusionProductSpace::new([]),
            FusionProductSpace::new([SectorLeg::new([(zero, 1)], false)]),
        ),
        vec![vec![1]],
    );
    let mut dst_data = vec![43.0_f64];

    let error = tensortrace_fusion_dyn_into(
        &dst,
        &mut dst_data,
        &src,
        &[],
        TensorTraceAxisSpec::new(&[1], &[], &[]),
        2.0,
        3.0,
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::InvalidAxisSet {
            tensor: "trace output",
            axes: vec![1],
            rank: 1,
        }
    );
    assert_eq!(dst_data, [43.0]);
}

#[test]
fn fz2_lowered_dynamic_trace_matches_encoded_data_oracle() {
    // What: the closed fZ2 trace produces byte-for-byte identical reduced data
    // through encoded and lowered bindings.
    let provider = std::sync::Arc::new(tenet_core::FermionParityFusionRule);
    let odd = tenet_core::Z2Irrep::ODD.sector_id();
    let leg = || FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]);
    let src_homspace = FusionTreeHomSpace::new(leg(), leg());
    let src_count = src_homspace.fusion_tree_keys(provider.as_ref()).len();
    let src_shapes = vec![vec![1, 1]; src_count];
    let encoded_src = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
        std::sync::Arc::clone(&provider),
        src_homspace.clone(),
        src_shapes.clone(),
    )
    .unwrap();
    let lowered_src = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        std::sync::Arc::clone(&provider),
        src_homspace,
        src_shapes,
    )
    .unwrap();
    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let encoded_dst = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
        std::sync::Arc::clone(&provider),
        scalar.clone(),
        [Vec::<usize>::new()],
    )
    .unwrap();
    let lowered_dst = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        provider,
        scalar,
        [Vec::<usize>::new()],
    )
    .unwrap();
    let axes = TensorTraceAxisSpec::new(&[], &[0], &[1]);
    let mut encoded_data = vec![7.0_f64];
    let mut lowered_data = encoded_data.clone();
    tensortrace_fusion_dyn_into(
        &encoded_dst,
        &mut encoded_data,
        &encoded_src,
        &[5.0],
        axes,
        2.0,
        3.0,
    )
    .unwrap();
    tensortrace_fusion_dyn_into(
        &lowered_dst,
        &mut lowered_data,
        &lowered_src,
        &[5.0],
        axes,
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(lowered_data, encoded_data);
}

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
    let dom_sector = SectorId::new(1);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(cod_sector, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(dom_sector, 3)], false)]),
    );
    let structure = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(
            BlockKey::from(FusionTreePairKey::pair(
                FusionTreeKey::try_new_for_rule(
                    &Z2FusionRule,
                    [cod_sector],
                    cod_sector,
                    [false],
                    [],
                    [],
                )
                .unwrap(),
                FusionTreeKey::try_new_for_rule(
                    &Z2FusionRule,
                    [dom_sector],
                    dom_sector,
                    [false],
                    [],
                    [],
                )
                .unwrap(),
            )),
            vec![2, 3],
            vec![1, 2],
            5,
        )
        .unwrap()],
    )
    .unwrap();
    let space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
        hom,
        structure,
    )
    .unwrap()
    .try_bind_rule(&Z2FusionRule)
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
    let space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        hom,
        BlockStructure::empty(2),
    )
    .unwrap()
    .try_bind_rule(&U1FusionRule)
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

    crate::tensortrace::reset_trace_transform_invocations();
    let structure =
        tensortrace_fusion_structure(&rule, &dst, &src, TensorTraceAxisSpec::new(&[], &[0], &[1]))
            .unwrap();
    assert_eq!(structure.terms().len(), 2);
    // What: Unique fusion retains the scalar direct path instead of constructing
    // a fusion-group recoupling matrix.
    assert_eq!(
        crate::tensortrace::take_trace_transform_invocations(),
        src.structure().block_count()
    );

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

fn scalar_trace_term_oracle<R>(
    rule: &R,
    dst_structure: &BlockStructure,
    src_structure: &BlockStructure,
    output_axes: &[usize],
    trace_lhs_axes: &[usize],
    trace_rhs_axes: &[usize],
    dst_codomain_rank: usize,
) -> Vec<(FusionTreePairKey, FusionTreePairKey, usize, usize, f64)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    use tenet_core::{multiplicity_free_permute_tree_pair, split_fusion_tree};

    let mut codomain_permutation = Vec::with_capacity(dst_codomain_rank + trace_lhs_axes.len());
    codomain_permutation.extend_from_slice(&output_axes[..dst_codomain_rank]);
    codomain_permutation.extend_from_slice(trace_lhs_axes);
    let mut domain_permutation =
        Vec::with_capacity(output_axes.len() - dst_codomain_rank + trace_rhs_axes.len());
    domain_permutation.extend_from_slice(&output_axes[dst_codomain_rank..]);
    domain_permutation.extend_from_slice(trace_rhs_axes);

    let mut terms = Vec::new();
    for src_block_index in 0..src_structure.block_count() {
        let src_key = expect_tree_key(src_structure.block(src_block_index).unwrap().key());
        for (permuted_key, permutation_coefficient) in multiplicity_free_permute_tree_pair(
            rule,
            &src_key,
            &codomain_permutation,
            &domain_permutation,
        )
        .unwrap()
        {
            let (dst_codomain_tree, trace_codomain_tree) =
                split_fusion_tree(rule, permuted_key.codomain_tree(), dst_codomain_rank).unwrap();
            let (dst_domain_tree, trace_domain_tree) = split_fusion_tree(
                rule,
                permuted_key.domain_tree(),
                output_axes.len() - dst_codomain_rank,
            )
            .unwrap();
            if trace_codomain_tree != trace_domain_tree {
                continue;
            }

            let coupled = trace_codomain_tree.coupled();
            let first = trace_codomain_tree.uncoupled()[0];
            let mut trace_factor = rule.dim_scalar(coupled) * rule.inv_dim_scalar(first);
            for (&sector, &is_dual) in trace_codomain_tree
                .uncoupled()
                .iter()
                .zip(trace_codomain_tree.is_dual())
                .skip(1)
            {
                if !is_dual {
                    trace_factor *= rule.twist_scalar(sector);
                }
            }
            let dst_key = FusionTreePairKey::pair(dst_codomain_tree, dst_domain_tree);
            let dst_block = dst_structure
                .find_block_index_by_fusion_tree_pair(&dst_key)
                .unwrap();
            terms.push((
                dst_key,
                src_key.clone(),
                dst_block,
                src_block_index,
                permutation_coefficient * trace_factor,
            ));
        }
    }
    terms
}

fn assert_trace_terms_match_scalar_oracle<R, C>(
    rule: &R,
    structure: &TensorTraceFusionStructure<f64>,
    dst: &TensorMap<f64, 1, 1, Trivial, C>,
    src_structure: &BlockStructure,
    output_axes: &[usize],
    trace_lhs_axes: &[usize],
    trace_rhs_axes: &[usize],
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    C: TensorStorage<f64>,
{
    let oracle = scalar_trace_term_oracle(
        rule,
        dst.structure(),
        src_structure,
        output_axes,
        trace_lhs_axes,
        trace_rhs_axes,
        1,
    );
    assert_eq!(structure.terms().len(), oracle.len());
    for (actual, (dst_key, src_key, dst_block, src_block, coefficient)) in
        structure.terms().iter().zip(oracle)
    {
        // What: block lowering preserves scalar-path term identity and global
        // source/destination order; only floating reduction may round differently.
        assert_eq!(actual.dst_key(), &dst_key);
        assert_eq!(actual.src_key(), &src_key);
        assert_eq!(actual.dst_block(), dst_block);
        assert_eq!(actual.src_block(), src_block);
        assert!((actual.coefficient() - coefficient).abs() <= 1.0e-12 * (1.0 + coefficient.abs()));
    }
}

#[test]
fn tensortrace_fusion_recouples_once_per_simple_fusion_group() {
    let rule = SU2FusionRule;
    let spin_one = SU2Irrep::from_twice_spin(2).sector_id();
    let leg = || SectorLeg::new([(spin_one, 1)], false);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let src_key_count = src_hom.fusion_tree_keys(&rule).len();
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([1, 1], [1, 1]).unwrap(),
        src_hom.clone(),
        &rule,
        (0..src_key_count).map(|_| vec![1, 1, 1, 1]),
    )
    .unwrap();
    let src: TensorMap<f64, 2, 2> = TensorMap::from_vec_with_fusion_space(
        vec![0.0; src_space.required_len().unwrap()],
        src_space,
    )
    .unwrap();

    let dst_hom = src_hom.select(&rule, &[1], &[3]).unwrap();
    let dst_key_count = dst_hom.fusion_tree_keys(&rule).len();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        dst_hom,
        &rule,
        (0..dst_key_count).map(|_| vec![1, 1]),
    )
    .unwrap();
    let dst: TensorMap<f64, 1, 1> = TensorMap::from_vec_with_fusion_space(
        vec![0.0; dst_space.required_len().unwrap()],
        dst_space,
    )
    .unwrap();
    let group_count = src.structure().fusion_tree_groups().len();
    assert!(
        src.structure().block_count() > group_count,
        "fixture must contain multiple source trees in one fusion group"
    );

    crate::tensortrace::reset_trace_transform_invocations();
    let structure = tensortrace_fusion_structure(
        &rule,
        &dst,
        &src,
        TensorTraceAxisSpec::new(&[1, 3], &[0], &[2]),
    )
    .unwrap();

    // What: non-Abelian partial trace walks the recoupling transform once per
    // external-sector group, not once per source fusion tree.
    assert_eq!(
        crate::tensortrace::take_trace_transform_invocations(),
        group_count
    );
    assert_trace_terms_match_scalar_oracle(
        &rule,
        &structure,
        &dst,
        src.structure(),
        &[1, 3],
        &[0],
        &[2],
    );
    assert!(structure
        .terms()
        .windows(2)
        .all(|pair| pair[0].src_block() <= pair[1].src_block()));

    let conjugate_axes = TensorTraceAxisSpec::new_with_conjugation(&[1, 3], &[0], &[2], true);
    let adjoint_src =
        crate::lowering::adjoint_fusion_space_view(src.fusion_space().unwrap()).unwrap();
    let lowered_axes =
        crate::lowering::lower_tensortrace_source_adjoint_axes::<2, 2>(conjugate_axes).unwrap();
    let lowered_axes = lowered_axes.as_spec();
    let conjugate_dst_hom = adjoint_src
        .homspace()
        .select(
            &rule,
            &lowered_axes.output_axes()[..1],
            &lowered_axes.output_axes()[1..],
        )
        .unwrap();
    let conjugate_dst_key_count = conjugate_dst_hom.fusion_tree_keys(&rule).len();
    let conjugate_dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        conjugate_dst_hom,
        &rule,
        (0..conjugate_dst_key_count).map(|_| vec![1, 1]),
    )
    .unwrap();
    let conjugate_dst: TensorMap<f64, 1, 1> = TensorMap::from_vec_with_fusion_space(
        vec![0.0; conjugate_dst_space.required_len().unwrap()],
        conjugate_dst_space,
    )
    .unwrap();
    let conjugate_structure =
        tensortrace_fusion_structure(&rule, &conjugate_dst, &src, conjugate_axes).unwrap();
    // What: source-conjugate lowering still batches the logical adjoint tree
    // groups while retaining the scalar adjoint-axis trace semantics.
    assert_trace_terms_match_scalar_oracle(
        &rule,
        &conjugate_structure,
        &conjugate_dst,
        adjoint_src.subblock_structure(),
        lowered_axes.output_axes(),
        lowered_axes.trace_lhs_axes(),
        lowered_axes.trace_rhs_axes(),
    );
}

#[test]
fn tensortrace_fusion_product_block_matches_scalar_trace_terms() {
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let odd = SectorId::new(1);
    let product_sector = |charge| {
        rule.encode_sector(
            left_rule.encode_sector(odd, U1Irrep::new(charge).sector_id()),
            SU2Irrep::from_twice_spin(1).sector_id(),
        )
    };
    let a = product_sector(1);
    let b = product_sector(-1);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(a, 1)], false),
            SectorLeg::new([(b, 1)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(b, 1)], false),
            SectorLeg::new([(a, 1)], false),
        ]),
    );
    let src_key_count = src_hom.fusion_tree_keys(&rule).len();
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([1, 1], [1, 1]).unwrap(),
        src_hom.clone(),
        &rule,
        (0..src_key_count).map(|_| vec![1, 1, 1, 1]),
    )
    .unwrap();
    let src: TensorMap<f64, 2, 2> = TensorMap::from_vec_with_fusion_space(
        vec![0.0; src_space.required_len().unwrap()],
        src_space,
    )
    .unwrap();
    let dst_hom = src_hom.select(&rule, &[1], &[2]).unwrap();
    let dst_key_count = dst_hom.fusion_tree_keys(&rule).len();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        dst_hom,
        &rule,
        (0..dst_key_count).map(|_| vec![1, 1]),
    )
    .unwrap();
    let dst: TensorMap<f64, 1, 1> = TensorMap::from_vec_with_fusion_space(
        vec![0.0; dst_space.required_len().unwrap()],
        dst_space,
    )
    .unwrap();

    let structure = tensortrace_fusion_structure(
        &rule,
        &dst,
        &src,
        TensorTraceAxisSpec::new(&[1, 2], &[0], &[3]),
    )
    .unwrap();

    assert_trace_terms_match_scalar_oracle(
        &rule,
        &structure,
        &dst,
        src.structure(),
        &[1, 2],
        &[0],
        &[3],
    );

    let conjugate_axes = TensorTraceAxisSpec::new_with_conjugation(&[1, 2], &[0], &[3], true);
    let adjoint_src =
        crate::lowering::adjoint_fusion_space_view(src.fusion_space().unwrap()).unwrap();
    let lowered_axes =
        crate::lowering::lower_tensortrace_source_adjoint_axes::<2, 2>(conjugate_axes).unwrap();
    let lowered_axes = lowered_axes.as_spec();
    let conjugate_dst_hom = adjoint_src
        .homspace()
        .select(
            &rule,
            &lowered_axes.output_axes()[..1],
            &lowered_axes.output_axes()[1..],
        )
        .unwrap();
    let conjugate_dst_key_count = conjugate_dst_hom.fusion_tree_keys(&rule).len();
    let conjugate_dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        conjugate_dst_hom,
        &rule,
        (0..conjugate_dst_key_count).map(|_| vec![1, 1]),
    )
    .unwrap();
    let conjugate_dst: TensorMap<f64, 1, 1> = TensorMap::from_vec_with_fusion_space(
        vec![0.0; conjugate_dst_space.required_len().unwrap()],
        conjugate_dst_space,
    )
    .unwrap();
    let conjugate_structure =
        tensortrace_fusion_structure(&rule, &conjugate_dst, &src, conjugate_axes).unwrap();

    // What: the Simple product path preserves source-conjugate adjoint lowering,
    // asymmetric U(1), half-integer SU(2), and the fZ2 odd supertrace sign.
    assert_trace_terms_match_scalar_oracle(
        &rule,
        &conjugate_structure,
        &conjugate_dst,
        adjoint_src.subblock_structure(),
        lowered_axes.output_axes(),
        lowered_axes.trace_lhs_axes(),
        lowered_axes.trace_rhs_axes(),
    );
    assert!(
        conjugate_structure
            .terms()
            .iter()
            .any(|term| *term.coefficient() < 0.0),
        "odd-sector conjugate trace fixture must retain a negative structural coefficient"
    );
}

#[test]
fn tensortrace_fusion_interleaved_groups_lower_in_global_source_order() {
    let rule = SU2FusionRule;
    let spin_zero = SU2Irrep::from_twice_spin(0).sector_id();
    let spin_one = SU2Irrep::from_twice_spin(2).sector_id();
    let leg = || SectorLeg::new([(spin_zero, 1), (spin_one, 1)], false);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = src_hom.fusion_tree_keys(&rule).len();
    let canonical = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([2, 2], [2, 2]).unwrap(),
        src_hom.clone(),
        &rule,
        (0..key_count).map(|_| vec![1, 1, 1, 1]),
    )
    .unwrap();
    let groups = canonical.subblock_structure().fusion_tree_groups();
    let first = groups
        .iter()
        .position(|group| group.block_indices().len() >= 2)
        .expect("fixture needs a multi-tree fusion group");
    let second = (0..groups.len())
        .find(|&index| index != first)
        .expect("fixture needs a second fusion group");
    let leading = [
        groups[first].block_indices()[0],
        groups[second].block_indices()[0],
        groups[first].block_indices()[1],
    ];
    let same_group_later_member = leading
        .iter()
        .position(|&index| index == groups[first].block_indices()[1])
        .unwrap();
    let mut order = leading.to_vec();
    order.extend(
        (0..canonical.subblock_structure().block_count()).filter(|index| !leading.contains(index)),
    );
    let reordered = BlockStructure::from_blocks_with_rank(
        4,
        order
            .into_iter()
            .map(|index| {
                let block = canonical.subblock_structure().block(index).unwrap();
                BlockSpec::with_key(
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let src_space = FusionTensorMapSpace::new_unbound(
        canonical.dense_space().clone(),
        src_hom.clone(),
        reordered,
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let src: TensorMap<f64, 2, 2> = TensorMap::from_vec_with_fusion_space(
        vec![0.0; src_space.required_len().unwrap()],
        src_space,
    )
    .unwrap();

    let dst_hom = src_hom.select(&rule, &[0], &[3]).unwrap();
    let dst_key_count = dst_hom.fusion_tree_keys(&rule).len();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        &rule,
        (0..dst_key_count).map(|_| vec![1, 1]),
    )
    .unwrap();
    let dst: TensorMap<f64, 1, 1> = TensorMap::from_vec_with_fusion_space(
        vec![0.0; dst_space.required_len().unwrap()],
        dst_space,
    )
    .unwrap();

    let structure = tensortrace_fusion_structure(
        &rule,
        &dst,
        &src,
        TensorTraceAxisSpec::new(&[0, 3], &[1], &[2]),
    )
    .unwrap();

    // What: group batching may visit A1,B1,A2 together internally, but published
    // trace terms retain the original global source-block order.
    assert!(structure
        .terms()
        .windows(2)
        .all(|pair| pair[0].src_block() <= pair[1].src_block()));
    assert_trace_terms_match_scalar_oracle(
        &rule,
        &structure,
        &dst,
        src.structure(),
        &[0, 3],
        &[1],
        &[2],
    );

    let missing_key = structure
        .terms()
        .iter()
        .find(|term| term.src_block() == 0)
        .expect("first source block must contribute a trace term")
        .dst_key()
        .clone();
    let incomplete_dst = BlockStructure::from_blocks_with_rank(
        dst.structure().rank(),
        (0..dst.structure().block_count())
            .filter_map(|index| {
                let block = dst.structure().block(index).unwrap();
                (block.key() != &BlockKey::from(missing_key.clone())).then(|| {
                    BlockSpec::with_key(
                        block.key().clone(),
                        block.shape().to_vec(),
                        block.strides().to_vec(),
                        block.offset(),
                    )
                    .unwrap()
                })
            })
            .collect(),
    )
    .unwrap();
    let incomplete_space = FusionTensorMapSpace::new_unbound(
        dst.fusion_space().unwrap().dense_space().clone(),
        dst.fusion_space().unwrap().homspace().clone(),
        incomplete_dst.clone(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    // What: the expert explicit-structure constructor permits incomplete
    // layouts; ordinary from_degeneracy_shapes construction remains complete.
    assert_eq!(
        incomplete_space.subblock_structure().block_count() + 1,
        dst.structure().block_count()
    );

    for _ in 0..2 {
        crate::tensortrace::reset_trace_transform_invocations();
        let error = crate::tensortrace::build_fusion_trace_terms_for_test(
            &rule,
            &incomplete_dst,
            src.structure(),
            TensorTraceAxisSpec::new(&[0, 3], &[1], &[2]),
            1,
        )
        .unwrap_err();
        // What: an earlier source lowering error stops before transforming a
        // later external-sector group, deterministically across compilations.
        assert!(matches!(
            error,
            OperationError::MissingBlockKey { ref key }
                if key.as_ref() == &BlockKey::from(missing_key.clone())
        ));
        assert_eq!(crate::tensortrace::take_trace_transform_sources(), vec![0]);
    }

    let malformed_structure = BlockStructure::from_blocks_with_rank(
        src.structure().rank(),
        (0..src.structure().block_count())
            .map(|index| {
                let block = src.structure().block(index).unwrap();
                let key = if index == same_group_later_member {
                    let BlockKey::FusionTree(key) = block.key() else {
                        panic!("fixture source must use fusion-tree keys");
                    };
                    let codomain = key.codomain_tree();
                    // What: the later member omits its rank-two vertex while
                    // retaining the same external group, so the public trace
                    // boundary proves source-major group admission.
                    let raw = FusionTreePairKey::try_pair_from_sector_ids(
                        codomain.uncoupled().iter().map(|sector| sector.id()),
                        Vec::<usize>::new(),
                        codomain.coupled().id(),
                        codomain.is_dual().iter().copied(),
                        Vec::<bool>::new(),
                        codomain.innerlines().iter().map(|sector| sector.id()),
                        Vec::<usize>::new(),
                        Vec::<usize>::new(),
                        Vec::<usize>::new(),
                    )
                    .unwrap();
                    FusionTreePairKey::pair(raw.codomain_tree().clone(), key.domain_tree().clone())
                        .into()
                } else {
                    block.key().clone()
                };
                BlockSpec::with_key(
                    key,
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let malformed_src_space = FusionTensorMapSpace::new_unbound(
        src.fusion_space().unwrap().dense_space().clone(),
        src.fusion_space().unwrap().homspace().clone(),
        malformed_structure,
    )
    .unwrap()
    .try_inherit_rule_identity(src.fusion_space().unwrap())
    .unwrap();
    let malformed_src: TensorMap<f64, 2, 2> = TensorMap::from_vec_with_fusion_space(
        vec![0.0; malformed_src_space.required_len().unwrap()],
        malformed_src_space,
    )
    .unwrap();
    let incomplete_dst_tensor: TensorMap<f64, 1, 1> = TensorMap::from_vec_with_fusion_space(
        vec![0.0; incomplete_space.required_len().unwrap()],
        incomplete_space,
    )
    .unwrap();

    for _ in 0..2 {
        crate::tensortrace::reset_trace_transform_invocations();
        let error = tensortrace_fusion_structure(
            &rule,
            &incomplete_dst_tensor,
            &malformed_src,
            TensorTraceAxisSpec::new(&[0, 3], &[1], &[2]),
        )
        .unwrap_err();
        // What: a Simple-fusion group transforms atomically, so its later
        // malformed member is diagnosed before lowering its first member.
        assert!(
            matches!(
                error,
                OperationError::Core(tenet_core::CoreError::MalformedFusionTree { .. })
            ),
            "unexpected block-atomic error: {error:?}"
        );
        assert_eq!(crate::tensortrace::take_trace_transform_sources(), vec![0]);
    }
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
