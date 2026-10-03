use super::*;

    #[test]
    fn block_fn_construction_is_layout_independent() {
        let rule = Z2FusionRule;
        let leg = |dual| SectorLeg::new([(z2_even(), 2), (z2_odd(), 2)], dual);
        let homspace = || {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(false), leg(false)]),
                FusionProductSpace::new([leg(false), leg(false)]),
            )
        };
        let shapes = |hom: &FusionTreeHomSpace| {
            hom.fusion_tree_keys(&rule)
                .iter()
                .map(|_| vec![2usize; 4])
                .collect::<Vec<_>>()
        };
        let dense = || TensorMapSpace::<2, 2>::from_dims([4, 4], [4, 4]).unwrap();
        let hom = homspace();
        let packed_structure =
            packed_fixture_structure(
                4,
                hom.fusion_tree_keys(&rule).iter().cloned().zip(shapes(&hom)),
            )
                .unwrap();
        let packed_space =
            FusionTensorMapSpace::<2, 2>::new_unbound(dense(), hom.clone(), packed_structure)
                .unwrap();
        let coupled_space = FusionTensorMapSpace::<2, 2>::from_degeneracy_shapes_coupled(
            dense(),
            hom.clone(),
            &rule,
            shapes(&hom),
        )
        .unwrap();

        let fill = |key: &BlockKey, indices: &[usize]| -> f64 {
            let BlockKey::FusionTree(tree) = key else {
                panic!("fusion tree keys expected");
            };
            let mut value = 0.0;
            for (axis, &sector) in tree
                .codomain_tree()
                .uncoupled()
                .iter()
                .chain(tree.domain_tree().uncoupled())
                .enumerate()
            {
                value += (axis as f64 + 1.0) * (sector.id() as f64 + 0.5);
            }
            for (axis, &index) in indices.iter().enumerate() {
                value += (axis as f64 + 2.0) * index as f64;
            }
            value
        };
        let packed =
            TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(packed_space, 0.0, fill)
                .unwrap();
        let coupled =
            TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(coupled_space, 0.0, fill)
                .unwrap();

        // Raw storage differs between layouts...
        assert_ne!(packed.data(), coupled.data());
        // ...but the logical block content is identical.
        let mut packed_elements = Vec::new();
        packed
            .for_each_block_element(|key, indices, value| {
                packed_elements.push((key.clone(), indices.to_vec(), *value));
            })
            .unwrap();
        let mut cursor = 0;
        coupled
            .for_each_block_element(|key, indices, value| {
                let (expected_key, expected_indices, expected_value) = &packed_elements[cursor];
                assert_eq!(key, expected_key);
                assert_eq!(indices, expected_indices.as_slice());
                assert_eq!(value, expected_value);
                cursor += 1;
            })
            .unwrap();
        assert_eq!(cursor, packed_elements.len());
    }

    #[test]
    fn coupled_layout_embeds_subblocks_into_sector_matrices() {
        let rule = Z2FusionRule;
        let leg = |degeneracy, dual| {
            SectorLeg::new([(z2_even(), degeneracy), (z2_odd(), degeneracy)], dual)
        };
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(2, false), leg(3, false)]),
            FusionProductSpace::new([leg(2, false), leg(3, false)]),
        );
        let keys = homspace.fusion_tree_keys(&rule);
        let shapes = keys
            .iter()
            .map(|_| vec![2usize, 3, 2, 3])
            .collect::<Vec<_>>();
        let packed = FusionTensorMapSpace::<2, 2>::from_degeneracy_shapes(
            TensorMapSpace::<2, 2>::from_dims([10, 10], [10, 10]).unwrap(),
            homspace.clone(),
            &rule,
            shapes.clone(),
        )
        .unwrap();
        let coupled = FusionTensorMapSpace::<2, 2>::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims([10, 10], [10, 10]).unwrap(),
            homspace,
            &rule,
            shapes,
        )
        .unwrap();

        let packed_structure = packed.subblock_structure();
        let coupled_structure = coupled.subblock_structure();
        assert_eq!(
            packed_structure.block_count(),
            coupled_structure.block_count()
        );
        assert_eq!(
            packed_structure.required_len().unwrap(),
            coupled_structure.required_len().unwrap()
        );

        // Two coupled sectors (even/odd), each with two codomain and two
        // domain trees of subblock row dim 6 and column dim 6: sector matrices
        // are 12 x 12.
        let matrix_rows = 12usize;
        let mut covered = vec![false; coupled_structure.required_len().unwrap()];
        for index in 0..coupled_structure.block_count() {
            let packed_block = packed_structure.block(index).unwrap();
            let coupled_block = coupled_structure.block(index).unwrap();
            assert_eq!(packed_block.key(), coupled_block.key());
            assert_eq!(packed_block.shape(), coupled_block.shape());
            // Codomain legs stay column-major inside the row block; domain
            // legs step whole matrix columns.
            assert_eq!(coupled_block.strides()[0], 1);
            assert_eq!(coupled_block.strides()[1], 2);
            assert_eq!(coupled_block.strides()[2], matrix_rows);
            assert_eq!(coupled_block.strides()[3], matrix_rows * 2);
            for i3 in 0..3 {
                for i2 in 0..2 {
                    for i1 in 0..3 {
                        for i0 in 0..2 {
                            let strides = coupled_block.strides();
                            let position = coupled_block.offset()
                                + i0 * strides[0]
                                + i1 * strides[1]
                                + i2 * strides[2]
                                + i3 * strides[3];
                            assert!(
                                !covered[position],
                                "coupled layout must not overlap between subblocks"
                            );
                            covered[position] = true;
                        }
                    }
                }
            }
        }
        assert!(
            covered.iter().all(|&flag| flag),
            "coupled layout must cover the sector matrices without holes"
        );
    }

    #[derive(Clone, Copy, Debug)]
    struct MisreportedGenericMultiplicityFreeRule;

    impl FusionRule for MisreportedGenericMultiplicityFreeRule {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            IdentitySymbolPanicRule.vacuum()
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            IdentitySymbolPanicRule.fusion_channels(left, right)
        }
    }

    impl MultiplicityFreeFusionRule for MisreportedGenericMultiplicityFreeRule {}

    impl MultiplicityFreeFusionSymbols for MisreportedGenericMultiplicityFreeRule {
        type Scalar = f64;

        fn f_symbol_scalar(
            &self,
            _left: SectorId,
            _middle: SectorId,
            _right: SectorId,
            _coupled: SectorId,
            _left_coupled: SectorId,
            _right_coupled: SectorId,
        ) -> Self::Scalar {
            panic!("misreported Generic provider evaluated an F symbol")
        }

        fn r_symbol_scalar(
            &self,
            _left: SectorId,
            _right: SectorId,
            _coupled: SectorId,
        ) -> Self::Scalar {
            panic!("misreported Generic provider evaluated an R symbol")
        }
    }

    impl MultiplicityFreeRigidSymbols for MisreportedGenericMultiplicityFreeRule {
        fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
    }

    #[test]
    fn unique_artin_braid_at_rejects_out_of_range_index() {
        let tree = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, false], [], [1]).unwrap();

        // What: index 1 is out of range for a rank-2 tree; a swap starting at
        // that index has no adjacent partner, so a permutation cannot express
        // it either. Repoint at the live single-step primitive that the
        // surviving multiplicity-free braid decomposes each Artin swap into.
        let err = apply_unique_artin_braid_at_with_inverse(
            &FermionParityFusionRule,
            &mut UnhashedFusionTree::from(tree.clone()),
            1,
            false,
        )
        .unwrap_err();

        assert_eq!(err, CoreError::InvalidBraidIndex { index: 1, rank: 2 });
    }

    #[test]
    fn rule_aware_constructor_rejects_inadmissible_unique_innerline() {
        // What: a rule-aware constructor rejects a stored vertex that is
        // outside the Unique rule's fusion graph.
        assert_eq!(
            FusionTreeKey::try_new_for_rule(
            &FermionParityFusionRule,
            [SectorId::new(1), SectorId::new(1), SectorId::new(1)], SectorId::new(1),
            [false, false, false],
            [SectorId::new(1)],
            [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
            )
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            }
        );
    }

    fn assert_invalid_rank_one_checked<R>(
        rule: &R,
        invalid: SectorId,
        expected: FusionAlgebraError,
    ) where
        R: CheckedFusionAlgebra,
    {
        let tree = FusionTreeKey::new([invalid], invalid, [false], [], []);
        assert_eq!(
            tree.validate_for_rule_checked(rule),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(expected)))
        );
    }

    #[test]
    fn checked_rank_one_validates_ids_by_unit_fusion() {
        // What: rank-one raw imports reject every unrepresentable built-in ID,
        // including the zigzag ID reserved for the excluded U1 charge.
        let invalid = SectorId::new(2);
        assert_invalid_rank_one_checked(
            &Z2FusionRule,
            invalid,
            FusionAlgebraError::InvalidSector { sector: invalid },
        );
        assert_invalid_rank_one_checked(
            &FermionParityFusionRule,
            invalid,
            FusionAlgebraError::InvalidSector { sector: invalid },
        );
        assert_invalid_rank_one_checked(
            &FibonacciFusionRule,
            invalid,
            FusionAlgebraError::InvalidSector { sector: invalid },
        );
        let invalid_su2 = SectorId::new(255);
        assert_invalid_rank_one_checked(
            &SU2FusionRule,
            invalid_su2,
            FusionAlgebraError::InvalidSector {
                sector: invalid_su2,
            },
        );
        #[cfg(target_pointer_width = "64")]
        {
            let invalid_u1 = SectorId::new(u32::MAX as usize + 1);
            assert_invalid_rank_one_checked(
                &U1FusionRule,
                invalid_u1,
                FusionAlgebraError::InvalidSector { sector: invalid_u1 },
            );

            type Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
            type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
            type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
            let rule = Rule::new(FermionParityFusionRule, U1FusionRule);
            let invalid_product = SectorId::new(1usize << Layout::BITS);
            assert_invalid_rank_one_checked(
                &rule,
                invalid_product,
                FusionAlgebraError::ProductCodec(
                    ProductSectorCodecError::InvalidHighBits {
                        sector: invalid_product,
                        total_bits: Layout::BITS,
                    },
                ),
            );
        }

        let invalid_u1 = excluded_u1_id();
        assert_invalid_rank_one_checked(
            &U1FusionRule,
            invalid_u1,
            FusionAlgebraError::InvalidSector { sector: invalid_u1 },
        );
    }

    #[test]
    fn checked_tree_normalizes_structure_before_provider_calls() {
        // What: shape and rank errors touch no checked provider operation, and
        // generated-channel failure precedes stored multiplicity validation.
        let rule = CheckedTreeProbe::default();
        for tree in [
            FusionTreeKey::new(
                [SectorId::new(0); 2],
                SectorId::new(0),
                [false],
                [],
                [MultiplicityIndex::ONE],
            ),
            FusionTreeKey::new(
                [SectorId::new(0)],
                SectorId::new(1),
                [false],
                [],
                [],
            ),
        ] {
            assert!(matches!(
                tree.validate_for_rule_checked(&rule),
                Err(CheckedFusionSpaceError::Core(_))
            ));
        }
        assert_eq!(rule.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(rule.nsymbol_calls.load(Ordering::Relaxed), 0);

        let failure = FusionTreeKey::new(
            [SectorId::new(9), SectorId::new(0)],
            SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::new(2).unwrap()],
        )
        .validate_for_rule_checked(&rule);
        assert_eq!(
            failure,
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::FusionNotRepresentable {
                    left: SectorId::new(9),
                    right: SectorId::new(0),
                }
            )))
        );
        assert_eq!(rule.channel_calls.load(Ordering::Relaxed), 1);
        assert_eq!(rule.nsymbol_calls.load(Ordering::Relaxed), 0);

        rule.channel_calls.store(0, Ordering::Relaxed);
        rule.nsymbol_calls.store(0, Ordering::Relaxed);
        assert_eq!(
            FusionTreeKey::new(
                [SectorId::new(0); 2],
                SectorId::new(0),
                [false; 2],
                [],
                [MultiplicityIndex::new(2).unwrap()],
            )
            .validate_for_rule_checked(&rule),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::MalformedFusionTree {
                    message: "fusion tree vertex label exceeds its fusion multiplicity",
                }
            )))
        );
        assert_eq!(rule.channel_calls.load(Ordering::Relaxed), 1);
        assert_eq!(rule.nsymbol_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn checked_tree_pair_preserves_validation_precedence() {
        // What: pair validation reports codomain, then domain, then coupled
        // mismatch without reordering finite-algebra checks.
        let bad_shape = FusionTreeKey::new(
            [u1(0); 2],
            u1(0),
            [false],
            [],
            [MultiplicityIndex::ONE],
        );
        let overflow = FusionTreeKey::new(
            [u1(i32::MAX), u1(1)],
            u1(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        assert!(matches!(
            FusionTreePairKey::pair(bad_shape, overflow.clone())
                .validate_for_rule_checked(&U1FusionRule),
            Err(CheckedFusionSpaceError::Core(_))
        ));
        assert_eq!(
            FusionTreePairKey::pair(
                FusionTreeKey::new([], u1(0), [], [], []),
                overflow,
            )
            .validate_for_rule_checked(&U1FusionRule),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::U1FusionOverflow {
                    left: i32::MAX,
                    right: 1,
                }
            )))
        );
        assert_eq!(
            FusionTreePairKey::pair(
                FusionTreeKey::new([u1(1)], u1(1), [false], [], []),
                FusionTreeKey::new([u1(2)], u1(2), [false], [], []),
            )
            .validate_for_rule_checked(&U1FusionRule),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::MalformedFusionTree {
                    message: "fusion tree pair requires matching coupled sectors",
                }
            )))
        );
    }

    #[test]
    fn fusion_subset_structural_proof_precedes_algebra_and_preserves_legacy_work() {
        // What: HomSpace metadata is proved without provider calls, while the
        // legacy categorical phase retains the former local-validator work.
        let scalar = FusionTreeKey::new([], SectorId::new(0), [], [], []);
        let malformed = FusionTreeKey::new(
            [SectorId::new(0); 2],
            SectorId::new(0),
            [false],
            [],
            [MultiplicityIndex::ONE],
        );
        let malformed_structure = packed_fixture_structure(
            2,
            [(
                FusionTreePairKey::pair(malformed, scalar.clone()),
                vec![1, 1],
            )],
        )
        .unwrap();
        let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1), (0, 1)], []);
        let checked = CheckedTreeProbe::default();
        assert!(matches!(
            homspace.validate_subblock_structure_subset_checked(
                &checked,
                &malformed_structure,
            ),
            Err(CheckedFusionSpaceError::Core(_))
        ));
        assert_eq!(checked.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(checked.nsymbol_calls.load(Ordering::Relaxed), 0);

        let valid = FusionTreeKey::new(
            [SectorId::new(0); 2],
            SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        let coupling_probe = CheckedTreeProbe::default();
        let mismatched_structure = packed_fixture_structure(
            2,
            [
                (
                    FusionTreePairKey::pair(valid.clone(), scalar.clone()),
                    vec![1, 1],
                ),
                (
                    FusionTreePairKey::pair(
                        valid.clone(),
                        FusionTreeKey::new([], SectorId::new(1), [], [], []),
                    ),
                    vec![1, 1],
                ),
            ],
        )
        .unwrap();
        assert!(matches!(
            homspace.validate_subblock_structure_subset_checked(
                &coupling_probe,
                &mismatched_structure,
            ),
            Err(CheckedFusionSpaceError::Core(_))
        ));
        assert_eq!(coupling_probe.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(coupling_probe.nsymbol_calls.load(Ordering::Relaxed), 0);

        let structure = packed_fixture_structure(
            2,
            [(FusionTreePairKey::pair(valid, scalar), vec![1, 1])],
        )
        .unwrap();
        let direct = CheckedTreeProbe::default();
        LocallyValidatedFusionTreeBlockStructure::try_new(&direct, &structure).unwrap();
        let expected_calls = direct.legacy_nsymbol_calls.load(Ordering::Relaxed);
        let admitted = CheckedTreeProbe::default();
        homspace
            .validate_subblock_structure_subset(&admitted, &structure)
            .unwrap();
        assert!(expected_calls > 0);
        assert_eq!(
            admitted.legacy_nsymbol_calls.load(Ordering::Relaxed),
            expected_calls
        );
    }

    #[test]
    fn checked_layout_and_space_admission_reject_finite_nonclosure_transactionally() {
        // What: caller-order layout validation and same-rule legacy admission
        // surface the exact finite-algebra error without publishing a new stamp.
        let scalar_zero = FusionTreeKey::new([], SectorId::new(0), [], [], []);
        let valid = FusionTreeKey::new(
            [SectorId::new(0); 2],
            SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        let malformed = FusionTreeKey::new(
            [SectorId::new(0); 2],
            SectorId::new(0),
            [false],
            [],
            [MultiplicityIndex::ONE],
        );
        let structural_probe = CheckedTreeProbe::default();
        assert!(matches!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &structural_probe,
                2,
                2,
                vec![
                    (
                        FusionTreePairKey::pair(valid.clone(), scalar_zero.clone()),
                        vec![1, 1],
                    ),
                    (
                        FusionTreePairKey::pair(malformed, scalar_zero.clone()),
                        vec![1, 1],
                    ),
                ],
            ),
            Err(CheckedFusionSpaceError::Core(_))
        ));
        assert_eq!(structural_probe.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(structural_probe.nsymbol_calls.load(Ordering::Relaxed), 0);

        let split_probe = CheckedTreeProbe::default();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &split_probe,
                3,
                2,
                vec![(
                    FusionTreePairKey::pair(valid.clone(), scalar_zero.clone()),
                    vec![1, 1],
                )],
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::StructureRankMismatch {
                    expected: 2,
                    actual: 3,
                },
            )))
        );
        assert_eq!(split_probe.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(split_probe.nsymbol_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys(
                &CheckedTreeProbe::default(),
                3,
                2,
                vec![(FusionTreePairKey::pair(valid, scalar_zero), vec![1, 1])],
            ),
            Err(CoreError::StructureRankMismatch {
                expected: 2,
                actual: 3,
            })
        );

        let overflow = FusionTreeKey::new(
            [u1(i32::MAX), u1(1)],
            u1(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        let scalar = FusionTreeKey::new([], u1(0), [], [], []);
        let pair = FusionTreePairKey::pair(overflow, scalar);
        let expected = CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            },
        ));
        reset_block_structure_intern_calls();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &U1FusionRule,
                2,
                2,
                vec![(pair.clone(), vec![1, 1])],
            ),
            Err(expected.clone())
        );
        assert_eq!(block_structure_intern_calls(), 0);

        let probe = CheckedTreeProbe::default();
        let failing_pair = FusionTreePairKey::pair(
            FusionTreeKey::new(
                [SectorId::new(9), SectorId::new(0)],
                SectorId::new(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            ),
            FusionTreeKey::new([], SectorId::new(0), [], [], []),
        );
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(SectorId::new(9), 1)], false),
                SectorLeg::new([(SectorId::new(0), 1)], false),
            ]),
            FusionProductSpace::new([]),
        );
        let legacy = FusionTensorMapSpace::<2, 0>::new_unbound(
            TensorMapSpace::from_dims([1, 1], []).unwrap(),
            homspace,
            packed_fixture_structure(2, [(failing_pair, vec![1, 1])]).unwrap(),
        )
        .unwrap()
        .try_bind_rule(&probe)
        .unwrap();
        assert!(matches!(legacy.admission(), FusionSpaceAdmission::Subset(_)));
        assert_eq!(
            legacy.try_bind_rule_checked(&probe),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::FusionNotRepresentable {
                    left: SectorId::new(9),
                    right: SectorId::new(0),
                },
            )))
        );
    }

    #[test]
    fn checked_coupled_layout_finishes_structural_preflight_before_algebra() {
        // What: incomplete grids, conflicting row extents, and overflowing
        // dimensions fail before checked algebra and publish no structure.
        let tree = |left| {
            FusionTreeKey::new(
                [SectorId::new(left), SectorId::new(0)],
                SectorId::new(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
        };
        let row_zero = tree(0);
        let row_one = tree(1);
        let col_zero = tree(2);
        let col_one = tree(3);

        let wrong_split = CheckedTreeProbe::default();
        let rank_one = FusionTreeKey::new(
            [SectorId::new(0)],
            SectorId::new(0),
            [false],
            [],
            [],
        );
        reset_block_structure_intern_calls();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &wrong_split,
                2,
                2,
                vec![(
                    FusionTreePairKey::pair(rank_one.clone(), rank_one),
                    vec![1; 2],
                )],
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::FusionSpaceSplitMismatch {
                    expected_nout: 2,
                    expected_nin: 0,
                    actual_nout: 1,
                    actual_nin: 1,
                },
            )))
        );
        assert_eq!(wrong_split.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(wrong_split.nsymbol_calls.load(Ordering::Relaxed), 0);
        assert_eq!(block_structure_intern_calls(), 0);

        let missing_grid = CheckedTreeProbe::default();
        reset_block_structure_intern_calls();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &missing_grid,
                2,
                4,
                vec![
                    (
                        FusionTreePairKey::pair(row_zero.clone(), col_zero.clone()),
                        vec![1; 4],
                    ),
                    (
                        FusionTreePairKey::pair(row_one, col_one.clone()),
                        vec![1; 4],
                    ),
                ],
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::BlockCountMismatch {
                    expected: 4,
                    actual: 2,
                },
            )))
        );
        assert_eq!(missing_grid.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(missing_grid.nsymbol_calls.load(Ordering::Relaxed), 0);
        assert_eq!(block_structure_intern_calls(), 0);

        let conflicting_extent = CheckedTreeProbe::default();
        reset_block_structure_intern_calls();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &conflicting_extent,
                2,
                4,
                vec![
                    (
                        FusionTreePairKey::pair(row_zero.clone(), col_zero.clone()),
                        vec![1; 4],
                    ),
                    (
                        FusionTreePairKey::pair(row_zero.clone(), col_one.clone()),
                        vec![2, 1, 1, 1],
                    ),
                ],
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::DimensionMismatch {
                    expected: 1,
                    actual: 2,
                },
            )))
        );
        assert_eq!(
            conflicting_extent.channel_calls.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            conflicting_extent.nsymbol_calls.load(Ordering::Relaxed),
            0
        );
        assert_eq!(block_structure_intern_calls(), 0);

        let overflowing_extent = CheckedTreeProbe::default();
        reset_block_structure_intern_calls();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &overflowing_extent,
                2,
                4,
                vec![(
                    FusionTreePairKey::pair(row_zero, col_zero),
                    vec![usize::MAX, 2, 1, 1],
                )],
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::ElementCountOverflow,
            )))
        );
        assert_eq!(
            overflowing_extent.channel_calls.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            overflowing_extent.nsymbol_calls.load(Ordering::Relaxed),
            0
        );
        assert_eq!(block_structure_intern_calls(), 0);
    }

    #[test]
    fn raw_tree_pair_constructor_rejects_zero_vertices_in_source_order() {
        let domain_vertices = std::iter::once_with(|| {
            panic!("domain vertices must not be consumed after a codomain error")
        });
        let error = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [1, 1],
            0,
            [false; 2],
            [false; 2],
            [],
            [],
            [0],
            domain_vertices,
        )
        .unwrap_err();
        // What: the public numeric import rejects zero while constructing the
        // codomain and does not continue into the domain after that error.
        assert_eq!(error, CoreError::InvalidMultiplicityIndex { value: 0 });

        assert_eq!(
            FusionTreePairKey::try_pair_from_sector_ids(
                [1, 1],
                [1, 1],
                0,
                [false; 2],
                [false; 2],
                [],
                [],
                [1],
                [0],
            )
            .unwrap_err(),
            CoreError::InvalidMultiplicityIndex { value: 0 },
        );
    }

    #[test]
    fn fusion_tree_block_structure_proof_reports_the_first_physical_invalid_pair() {
        // What: structure admission reports the earliest malformed
        // fusion-tree pair in physical block order.
        let valid = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(1), u1(-1)], u1(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::new([], u1(0), [], [], []),
        );
        let mismatched_pair = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(1)], u1(1),
                [false],
                [],
                [],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(2)], u1(2),
                [false],
                [],
                [],
            )
            .unwrap(),
        );
        let later_bad_shape = FusionTreePairKey::pair(
            FusionTreeKey::new([u1(1), u1(-1)], u1(0), [false], [], [MultiplicityIndex::ONE]),
            FusionTreeKey::new([], u1(0), [], [], []),
        );
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(valid.into(), vec![1, 1], 0).unwrap(),
            BlockSpec::column_major_with_key(mismatched_pair.into(), vec![1, 1], 1).unwrap(),
            BlockSpec::column_major_with_key(later_bad_shape.into(), vec![1, 1], 2).unwrap(),
        ])
        .unwrap();

        let error =
            match LocallyValidatedFusionTreeBlockStructure::try_new(&U1FusionRule, &structure) {
                Ok(_) => panic!("malformed structure unexpectedly admitted"),
                Err(error) => error,
            };
        assert_eq!(
            error,
            CoreError::MalformedFusionTree {
                message: "fusion tree pair requires matching coupled sectors",
            }
        );
    }

    #[test]
    fn fusion_tree_block_structure_proof_rejects_nontrivial_vertex_for_unique_provider() {
        let invalid = FusionTreePairKey::pair(
            FusionTreeKey::new(
                [z2_odd(), z2_odd()],
                z2_even(),
                [false; 2],
                [],
                [MultiplicityIndex::new(2).unwrap()],
            ),
            FusionTreeKey::new([z2_even()], z2_even(), [false], [], []),
        );
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(invalid.into(), vec![1, 1, 1], 0).unwrap(),
        ])
        .unwrap();

        // What: a raw label-two key cannot acquire the local proof required
        // by compact multiplicity-free batch execution.
        let error =
            match LocallyValidatedFusionTreeBlockStructure::try_new(&Z2FusionRule, &structure) {
                Ok(_) => panic!("nontrivial multiplicity label unexpectedly admitted"),
                Err(error) => error,
            };
        assert_eq!(
            error,
            CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            }
        );
    }

    #[test]
    fn coupled_sector_constructor_validates_in_caller_order_before_sorting() {
        let first = FusionTreePairKey::pair(
            FusionTreeKey::new(
                [SectorId::new(1), SectorId::new(2)],
                SectorId::new(3),
                [false; 2],
                [],
                [MultiplicityIndex::new(2).unwrap()],
            ),
            FusionTreeKey::new(
                [SectorId::new(3)],
                SectorId::new(3),
                [false],
                [],
                [],
            ),
        );
        let later_lower_coupled = FusionTreePairKey::pair(
            FusionTreeKey::new(
                [SectorId::new(0), SectorId::new(1)],
                SectorId::new(1),
                [false],
                [],
                [MultiplicityIndex::ONE],
            ),
            FusionTreeKey::new(
                [SectorId::new(1)],
                SectorId::new(1),
                [false],
                [],
                [],
            ),
        );

        // What: the first caller-supplied categorical error wins even though
        // coupled-sector layout order would move the later key before it.
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys(
                &IdentitySymbolPanicRule,
                2,
                3,
                vec![
                    (first, vec![1, 1, 1]),
                    (later_lower_coupled, vec![1, 1, 1]),
                ],
            )
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            }
        );
    }

    #[test]
    fn fusion_tree_block_structure_proof_rejects_non_categorical_namespaces() {
        // What: LOCAL categorical admission rejects both anonymous dense
        // storage and arbitrary application routing keys.
        for key in [BlockKey::Dense, BlockKey::opaque([7, 11])] {
            let structure = BlockStructure::from_blocks(vec![
                BlockSpec::column_major_with_key(key.clone(), vec![1], 0).unwrap(),
            ])
            .unwrap();
            let error =
                match LocallyValidatedFusionTreeBlockStructure::try_new(&Z2FusionRule, &structure) {
                    Ok(_) => panic!("non-categorical structure unexpectedly admitted"),
                    Err(error) => error,
                };
            assert_eq!(
                error,
                CoreError::ExpectedFusionTreePairKey { actual: key.kind() }
            );
        }
    }

    #[test]
    fn fusion_tree_block_structure_proof_rejects_local_rank_mismatch() {
        // What: categorical admission requires every tree pair to describe the
        // same number of external legs as its containing block structure.
        let key = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &Z2FusionRule,
                [z2_even()], z2_even(),
                [false],
                [],
                [],
            )
            .unwrap(),
            FusionTreeKey::new([], z2_even(), [], [], []),
        );
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(key.into(), vec![1, 1], 0).unwrap(),
        ])
        .unwrap();

        let error = match LocallyValidatedFusionTreeBlockStructure::try_new(&Z2FusionRule, &structure) {
            Ok(_) => panic!("rank-mismatched structure unexpectedly admitted"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            CoreError::StructureRankMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn malformed_compact_tree_key_is_panic_free_for_structure_lookup() {
        // What: a raw one-leg key with missing dual metadata remains a fallible
        // categorical input rather than panicking during structure indexing.
        let malformed = FusionTreePairKey::pair(
            FusionTreeKey::new([z2_even()], z2_even(), [], [], []),
            FusionTreeKey::new([], z2_even(), [], [], []),
        );
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(malformed.clone().into(), vec![1], 0).unwrap(),
        ])
        .unwrap();

        assert_eq!(
            structure.find_block_index_by_fusion_tree_pair(&malformed),
            Some(0)
        );
        assert_eq!(
            structure.fusion_tree_pair_block(&malformed).unwrap().key(),
            &BlockKey::from(malformed)
        );
    }

    #[test]
    fn fusion_tree_block_structure_proof_indexes_only_its_bound_structure() {
        // What: a successful proof exposes borrowed keys by physical index from
        // its exact categorical structure, including out-of-bounds errors.
        let valid = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(1), u1(-1)], u1(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::new([], u1(0), [], [], []),
        );
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(valid.clone().into(), vec![1, 1], 0).unwrap(),
        ])
        .unwrap();

        let proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&U1FusionRule, &structure).unwrap();
        assert!(std::ptr::eq(proof.rule(), &U1FusionRule));
        assert!(std::ptr::eq(proof.structure(), &structure));
        let canonical = proof.fusion_tree_pair_key(0);
        #[allow(deprecated)]
        let legacy = proof.fusion_tree_block_key(0);
        assert_eq!(canonical, legacy);
        assert_eq!(canonical.unwrap(), Some(&valid));
        let canonical_missing = proof.fusion_tree_pair_key(1);
        #[allow(deprecated)]
        let legacy_missing = proof.fusion_tree_block_key(1);
        assert_eq!(canonical_missing, legacy_missing);
        assert_eq!(
            canonical_missing.unwrap_err(),
            CoreError::BlockIndexOutOfBounds { index: 1, count: 1 }
        );
    }

    #[test]
    fn validated_per_index_permute_preserves_planar_braiding_rejection_before_index_lookup() {
        let structure = BlockStructure::empty(0);
        let proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&PlanarZ2Rule, &structure).unwrap();

        // What: a proof-consuming per-index call observes the symmetric-
        // braiding boundary before attempting to read its requested block.
        assert_eq!(
            proof.permute_codomain_rows_for_block_index(0, &[]),
            Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: BraidingStyleKind::NoBraiding,
            })
        );
    }

    #[test]
    fn multiplicity_free_block_boundaries_reject_misreported_style_before_empty_identity() {
        let rule = MisreportedGenericMultiplicityFreeRule;
        let structure = BlockStructure::empty(0);
        let proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&rule, &structure).unwrap();
        let identity = PreparedTreePairOperation::prepare_transpose(0, 0, &[], &[]).unwrap();
        let expected = CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: FusionStyleKind::Generic,
        };

        // What: neither proof-bound nor raw block entry points let empty or
        // identity work bypass the runtime fusion-style capability.
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_for_block_indices(
                    std::iter::empty(),
                    identity.clone(),
                )
                .unwrap_err(),
            expected
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_transpose_for_block_indices(
                    std::iter::empty(),
                    identity,
                )
                .unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_braid_tree_block(&rule, &[], &[], &[]).unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_permute_tree_block(&rule, &[], &[]).unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_braid_tree_pair_block(&rule, &[], &[], &[], &[], &[])
                .unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(&rule, &[], &[], &[]).unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_transpose_tree_pair_block(&rule, &[], &[], &[]).unwrap_err(),
            expected
        );
    }

    #[test]
    fn borrowed_block_execution_rejects_prepared_family_and_source_split_mismatch() {
        let vacuum = su2(0);
        let half = su2(1);
        let pair = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [half, half],
                vacuum,
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [],
                vacuum,
                [],
                [],
                [],
            )
            .unwrap(),
        );
        let structure = packed_fixture_structure(2, [(pair, vec![1, 1])]).unwrap();
        let proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&SU2FusionRule, &structure).unwrap();
        let transpose =
            PreparedTreePairOperation::prepare_transpose(2, 0, &[1], &[0]).unwrap();
        let transpose_identity =
            PreparedTreePairOperation::prepare_transpose(2, 0, &[0, 1], &[]).unwrap();
        let braid =
            PreparedTreePairOperation::prepare_permute(&SU2FusionRule, 2, 0, &[1, 0], &[])
                .unwrap();
        let wrong_split =
            PreparedTreePairOperation::prepare_permute(&SU2FusionRule, 1, 1, &[1], &[0])
                .unwrap();

        // What: safe borrowed block executors reject a prepared operation from
        // the other family before its private plan variant reaches execution.
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    [0],
                    &transpose,
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "prepared tree-pair operation is incompatible with braid block execution",
            }
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    [0],
                    &transpose_identity,
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "prepared tree-pair operation is incompatible with braid block execution",
            }
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
                    [0], &braid,
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message:
                    "prepared tree-pair operation is incompatible with transpose block execution",
            }
        );

        // What: operation family is a source-independent preflight. Empty and
        // invalid-index calls cannot bypass it, while a valid empty operation
        // remains the empty linear map.
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    std::iter::empty(),
                    &transpose,
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "prepared tree-pair operation is incompatible with braid block execution",
            }
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
                    std::iter::empty(),
                    &braid,
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message:
                    "prepared tree-pair operation is incompatible with transpose block execution",
            }
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices(
                    [usize::MAX],
                    transpose.clone(),
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "prepared tree-pair operation is incompatible with braid block execution",
            }
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_transpose_ordered_for_block_indices(
                    [usize::MAX],
                    braid.clone(),
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message:
                    "prepared tree-pair operation is incompatible with transpose block execution",
            }
        );
        for empty in [
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    std::iter::empty(),
                    &braid,
                )
                .unwrap(),
            proof
                .execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
                    std::iter::empty(),
                    &transpose,
                )
                .unwrap(),
        ] {
            assert!(empty.destinations().is_empty());
            assert_eq!(empty.source_count(), 0);
        }

        // What: preparation for the right operation family is still bound to
        // its exact source codomain/domain split.
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    [0],
                    &wrong_split,
                )
                .unwrap_err(),
            CoreError::DimensionMismatch {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn block_preflight_rejects_prepared_capability_before_empty_or_invalid_source() {
        let structure = BlockStructure::empty(0);
        let proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&PlanarZ2Rule, &structure).unwrap();
        let symmetric_identity =
            PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 0, 0, &[], &[]).unwrap();
        let expected = CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: BraidingStyleKind::NoBraiding,
        };

        // What: rule capability belongs to operation preflight, so neither an
        // empty iterator nor an invalid source index can bypass it.
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    std::iter::empty(),
                    &symmetric_identity,
                )
                .unwrap_err(),
            expected
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices(
                    [usize::MAX],
                    symmetric_identity,
                )
                .unwrap_err(),
            expected
        );
    }

    #[test]
    fn inadmissible_tree_is_rejected_before_every_identity_path() {
        // What: a shape-correct but inadmissible tree cannot pass through
        // scalar, prepared, or whole-block identity shortcuts.
        fn assert_inadmissible<T: std::fmt::Debug>(result: Result<T, CoreError>) {
            assert_eq!(
                result.unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "fusion tree contains an inadmissible fusion vertex",
                }
            );
        }

        let rule = SplitOnlyCountingRule::default();
        let odd = SectorId::new(1);
        let vacuum = SectorId::new(0);
        let source = FusionTreeKey::new(
            [odd; 3], vacuum,
            [false; 3],
            [vacuum],
            [MultiplicityIndex::ONE; 2],
        );
        let identity = [0, 1, 2];
        let levels = [0, 1, 2];
        assert_inadmissible(multiplicity_free_braid_tree(&rule, &source, &identity, &levels));
        assert_inadmissible(multiplicity_free_permute_tree(
            &rule, &source, &identity,
        ));
        assert_inadmissible(multiplicity_free_braid_tree_block(
            &rule,
            std::slice::from_ref(&source),
            &identity,
            &levels,
        ));
        assert_inadmissible(multiplicity_free_permute_tree_block(
            &rule,
            std::slice::from_ref(&source),
            &identity,
        ));

        let pair = FusionTreePairKey::pair(
            source,
            FusionTreeKey::new([], vacuum, [], [], []),
        );
        let prepared = PreparedTreePairOperation::prepare_braid(
            &rule,
            3,
            0,
            &identity,
            &[],
            &levels,
            &[],
        )
        .unwrap();
        assert_inadmissible(prepared.execute_multiplicity_free(&rule, &pair));
        assert_inadmissible(multiplicity_free_braid_tree_pair(
            &rule, &pair, &identity, &[], &levels, &[],
        ));
        assert_inadmissible(multiplicity_free_permute_tree_pair(
            &rule, &pair, &identity, &[],
        ));
        assert_inadmissible(multiplicity_free_transpose_tree_pair(
            &rule, &pair, &identity, &[],
        ));
        assert_inadmissible(multiplicity_free_braid_tree_pair_block(
            &rule,
            std::slice::from_ref(&pair),
            &identity,
            &[],
            &levels,
            &[],
        ));
        assert_inadmissible(multiplicity_free_permute_tree_pair_block(
            &rule,
            std::slice::from_ref(&pair),
            &identity,
            &[],
        ));
        assert_inadmissible(multiplicity_free_transpose_tree_pair_block(
            &rule,
            std::slice::from_ref(&pair),
            &identity,
            &[],
        ));
        assert_eq!(
            rule.f_calls.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            rule.r_calls.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn malformed_sources_do_not_evaluate_symbols() {
        // What: categorical rejection occurs before an F or R provider can
        // observe a malformed source.
        //
        // Bends are not counted here: the coefficient comes from
        // b_symbol_scalar / sqrt_dim_scalar on MultiplicityFreeRigidSymbols,
        // which have no separate counter, so a bend-specific assertion would
        // only be able to observe zero and would prove nothing.
        let rule = SplitOnlyCountingRule::default();
        let invalid = FusionTreeKey::new(
            [SectorId::new(1); 2], SectorId::new(1),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        assert!(multiplicity_free_braid_tree(&rule, &invalid, &[0, 1], &[0, 1]).is_err());
        let pair = FusionTreePairKey::pair(
            invalid,
            FusionTreeKey::new(
                [SectorId::new(1)], SectorId::new(1),
                [false],
                [],
                [],
            ),
        );
        assert!(multiplicity_free_repartition_tree_pair(&rule, &pair, 1).is_err());
        assert_eq!(
            rule.f_calls.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            rule.r_calls.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn block_validation_is_source_major_and_runs_once_per_source() {
        // What: block proofs report the first source in slice order and do one
        // N-symbol validation pass before identity execution.
        let tau = SectorId::new(1);
        let vacuum = SectorId::new(0);
        let invalid_first = FusionTreeKey::new(
            [tau; 3], vacuum,
            [false; 3],
            [vacuum],
            [MultiplicityIndex::ONE; 2],
        );
        let valid_first = FusionTreeKey::new(
            [tau; 3], tau,
            [false; 3],
            [vacuum],
            [MultiplicityIndex::ONE; 2],
        );
        let different_group = FusionTreeKey::new(
            [tau; 2], vacuum,
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        assert_eq!(
            multiplicity_free_braid_tree_block(
                &FibonacciFusionRule,
                &[invalid_first, different_group.clone()],
                &[0, 1, 2],
                &[0, 1, 2],
            )
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            }
        );
        assert_eq!(
            multiplicity_free_braid_tree_block(
                &FibonacciFusionRule,
                &[valid_first, different_group],
                &[0, 1, 2],
                &[0, 1, 2],
            )
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion-tree keys must share one group",
            }
        );

        let rule = SplitOnlyCountingRule::default();
        let valid = FusionTreeKey::new(
            [SectorId::new(1); 2], SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        let rows = multiplicity_free_braid_tree_block(
            &rule,
            &[valid.clone(), valid.clone()],
            &[0, 1],
            &[0, 1],
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rule.n_calls.load(std::sync::atomic::Ordering::Relaxed),
            2
        );

        rule.n_calls
            .store(0, std::sync::atomic::Ordering::Relaxed);
        let pair = FusionTreePairKey::pair(
            valid,
            FusionTreeKey::new(
                [SectorId::new(0)], SectorId::new(0),
                [false],
                [],
                [],
            ),
        );
        let rows = multiplicity_free_permute_tree_pair_block(
            &rule,
            &[pair.clone(), pair],
            &[0, 1],
            &[2],
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rule.n_calls.load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }

    fn fibonacci_multi_associator_counterexample() -> (FusionTreeKey, FusionTreeKey) {
        let long =
            FusionTreeKey::try_from_sector_ids([1; 4], 1, [false; 4], [1, 0], [1, 1, 1])
                .unwrap();
        let short =
            FusionTreeKey::try_from_sector_ids([1; 3], 1, [false; 3], [0], [1, 1]).unwrap();
        let tau = SectorId::new(1);
        assert!(
            collect_fusion_trees_for_coupled(
                &FibonacciFusionRule,
                &[tau; 4],
                &[false; 4],
                &[tau; 4],
                tau,
            )
            .contains(&long)
        );
        assert!(
            collect_fusion_trees_for_coupled(
                &FibonacciFusionRule,
                &[tau; 3],
                &[false; 3],
                &[tau; 3],
                tau,
            )
            .contains(&short)
        );
        (long, short)
    }

    #[test]
    fn fibonacci_multi_associator_filters_cross_inadmissible_candidates() {
        let rule = FibonacciFAdmissibilityProbe::new();
        let (long, short) = fibonacci_multi_associator_counterexample();
        let first = long.uncoupled()[0];
        let right = long.uncoupled()[2];
        let (middle_left, middle_right) = fusion_tree_vertex_neighbors(&long, 2).unwrap();
        let (short_left, short_right) = fusion_tree_vertex_neighbors(&short, 1).unwrap();

        // What: both stored trees are valid, but their staged cross vertex is
        // absent and therefore does not name an F-symbol coefficient.
        assert_ne!(
            rule.nsymbol(middle_left, right, middle_right),
            0
        );
        assert_ne!(rule.nsymbol(short_left, right, short_right), 0);
        assert_ne!(rule.nsymbol(first, short_left, middle_left), 0);
        assert_eq!(rule.nsymbol(first, short_right, middle_right), 0);
        assert_eq!(
            rule.f_symbol_scalar(
                first,
                short_left,
                right,
                middle_right,
                middle_left,
                short_right,
            ),
            FibonacciFAdmissibilityProbe::SENTINEL
        );
        assert_eq!(rule.take_calls().len(), 1);

        assert_eq!(
            multiplicity_free_multi_associator_scalar(&rule, &long, &short).unwrap(),
            None
        );
        assert!(rule.take_calls().is_empty());
    }

    #[test]
    fn fibonacci_multi_fmove_forward_and_inverse_call_only_admissible_f() {
        let rule = FibonacciFAdmissibilityProbe::new();
        let (long, short) = fibonacci_multi_associator_counterexample();
        let actual_forward = multiplicity_free_multi_fmove_tree(&rule, &long).unwrap();
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        // What: TensorKit's Stage-1 candidate intersection retains these two
        // tails in this order; Stage 2 gives the listed Fibonacci F products.
        assert_eq!(
            actual_forward
                .iter()
                .map(|(tree, _)| (tree.coupled(), tree.innerlines().to_vec()))
                .collect::<Vec<_>>(),
            vec![
                (SectorId::new(0), vec![SectorId::new(1)]),
                (SectorId::new(1), vec![SectorId::new(1)]),
            ]
        );
        assert!(
            (actual_forward[0].1 - Complex64::new(1.0 / phi, 0.0)).norm() < 1.0e-12
        );
        assert!(
            (actual_forward[1].1 - Complex64::new(1.0 / phi.sqrt(), 0.0)).norm()
                < 1.0e-12
        );
        let calls = rule.take_calls();
        assert!(!calls.is_empty());
        assert_fibonacci_f_calls_are_admissible(&calls);

        let actual_inverse = multiplicity_free_multi_fmove_inv_tree(
            &rule,
            SectorId::new(1),
            SectorId::new(1),
            &short,
            false,
        )
        .unwrap();
        // What: TensorKit's right-to-left inverse construction retains these
        // two rank-4 trees in canonical order and conjugates the same real
        // Fibonacci coefficients.
        assert_eq!(
            actual_inverse
                .iter()
                .map(|(tree, _)| (tree.coupled(), tree.innerlines().to_vec()))
                .collect::<Vec<_>>(),
            vec![
                (
                    SectorId::new(1),
                    vec![SectorId::new(0), SectorId::new(1)]
                ),
                (
                    SectorId::new(1),
                    vec![SectorId::new(1), SectorId::new(1)]
                ),
            ]
        );
        assert!(
            (actual_inverse[0].1 - Complex64::new(1.0 / phi, 0.0)).norm() < 1.0e-12
        );
        assert!(
            (actual_inverse[1].1 - Complex64::new(1.0 / phi.sqrt(), 0.0)).norm()
                < 1.0e-12
        );
        let calls = rule.take_calls();
        assert!(!calls.is_empty());
        assert_fibonacci_f_calls_are_admissible(&calls);
    }

    #[test]
    fn grouped_multi_fmove_matches_legacy_order_and_reuses_stage_symbols() {
        let rule = FibonacciFAdmissibilityProbe::with_complex_f_phase();
        let tau = SectorId::new(1);
        let trees = collect_fusion_trees_for_coupled(
            &rule,
            &[tau; 6],
            &[false; 6],
            &[tau; 6],
            tau,
        );
        let mut fixture = None;
        for tree in trees {
            let grouped = multiplicity_free_multi_fmove_tree(&rule, &tree).unwrap();
            let grouped_calls = rule.take_calls();
            let legacy =
                multiplicity_free_multi_fmove_tree_legacy_oracle(&rule, &tree).unwrap();
            let legacy_calls = rule.take_calls();
            if grouped_calls.len() < legacy_calls.len() {
                fixture = Some((tree, grouped, grouped_calls, legacy, legacy_calls));
                break;
            }
        }
        let (tree, grouped, grouped_calls, legacy, legacy_calls) =
            fixture.expect("rank-six Fibonacci must repeat stage-local F arguments");

        // What: grouped forward execution preserves the legacy candidate order
        // and coefficients while evaluating one complete F sextuple per stage.
        assert_eq!(grouped, legacy);
        assert_eq!(grouped_calls.len(), 7);
        assert_eq!(legacy_calls.len(), 18);
        assert_fibonacci_f_calls_are_admissible(&grouped_calls);
        assert!(grouped
            .iter()
            .any(|(_, coefficient)| coefficient.im.abs() > 1.0e-12));

        let tail = grouped
            .first()
            .expect("rank-six Fibonacci forward move has a tail")
            .0
            .clone();
        let grouped_inverse =
            multiplicity_free_multi_fmove_inv_tree(&rule, tau, tree.coupled(), &tail, false)
                .unwrap();
        let grouped_inverse_calls = rule.take_calls();
        let legacy_inverse = multiplicity_free_multi_fmove_inv_tree_legacy_oracle(
            &rule,
            tau,
            tree.coupled(),
            &tail,
            false,
        )
        .unwrap();
        let legacy_inverse_calls = rule.take_calls();

        // What: inverse execution uses the same canonical candidates and applies
        // conjugation after the same grouped associator products.
        assert_eq!(grouped_inverse, legacy_inverse);
        assert!(grouped_inverse_calls.len() <= legacy_inverse_calls.len());
        assert_fibonacci_f_calls_are_admissible(&grouped_inverse_calls);
        assert!(grouped_inverse
            .iter()
            .any(|(_, coefficient)| coefficient.im.abs() > 1.0e-12));
    }

    #[test]
    fn fibonacci_braid_then_inverse_braid_is_identity() {
        // Self-consistency (a): braiding a crossing and then undoing it
        // (reflected levels select the inverse-artin branch) must return the
        // exact original tree with total coefficient 1 — this only holds
        // because R^{ττ}_* is a genuine unit-modulus phase.
        let rule = FibonacciFusionRule;
        for coupled in [0usize, 1usize] {
            let tree =
                FusionTreeKey::try_from_sector_ids([1, 1], coupled, [false, false], [], [1]).unwrap();

            let forward = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
            assert_eq!(forward.len(), 1);
            let backward =
                multiplicity_free_braid_tree(&rule, &forward[0].0, &[1, 0], &[1, 0]).unwrap();
            assert_eq!(backward.len(), 1);

            assert_eq!(backward[0].0.uncoupled(), tree.uncoupled());
            assert_eq!(backward[0].0.coupled(), tree.coupled());
            let total = forward[0].1 * backward[0].1;
            assert!((total - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        }

        // Same check through the rank > 2 loop branch, where the round trip
        // additionally exercises the F-symbol: this only returns to the
        // identity because TensorKitSectors' F^{τττ}_τ block is a genuine
        // (real, orthogonal) unitary matrix.
        let tree =
            FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1]).unwrap();
        let forward = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
        assert_eq!(forward.len(), 2);
        let mut total = Complex64::new(0.0, 0.0);
        for (intermediate, coeff) in &forward {
            let backward =
                multiplicity_free_braid_tree(&rule, intermediate, &[0, 2, 1], &[0, 2, 1]).unwrap();
            for (roundtrip, back_coeff) in &backward {
                if roundtrip.innerlines() == tree.innerlines() {
                    total += *coeff * *back_coeff;
                }
            }
        }
        assert!((total - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
    }

    #[test]
    fn tree_pair_block_apis_reject_mixed_product_sector_components() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = FpU1Rule::default();
        let sector_a = rule.encode_sector(z2_even(), u1(2)).id();
        let sector_b = rule.encode_sector(z2_even(), u1(3)).id();
        let coupled_a = rule.encode_sector(z2_even(), u1(4)).id();
        let coupled_b = rule.encode_sector(z2_even(), u1(5)).id();
        let keys = [
            tree_pair_group_fixture(
                &[sector_a, sector_a],
                &[coupled_a],
                coupled_a,
                &[false; 2],
                &[false],
            ),
            tree_pair_group_fixture(
                &[sector_a, sector_b],
                &[coupled_b],
                coupled_b,
                &[false; 2],
                &[false],
            ),
        ];

        // What: a changed component of an interned product-sector label is a
        // different shared basis group, even when ranks and duality match.
        assert_mixed_tree_pair_block_group_is_rejected(
            &rule,
            &keys,
            CoreError::MalformedFusionTree {
                message: TREE_PAIR_BLOCK_GROUP_ERROR,
            },
        );
    }

    #[test]
    fn tree_pair_block_empty_and_valid_group_identity_policies_are_stable() {
        let rule = IdentitySymbolPanicRule;
        let empty: &[FusionTreePairKey] = &[];
        // What: an empty block remains a valid empty coefficient transform.
        assert_eq!(
            multiplicity_free_braid_tree_pair_block(&rule, empty, &[], &[], &[], &[]).unwrap(),
            Vec::<Vec<(FusionTreePairKey, f64)>>::new()
        );
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(&rule, empty, &[], &[]).unwrap(),
            Vec::<Vec<(FusionTreePairKey, f64)>>::new()
        );
        assert_eq!(
            multiplicity_free_transpose_tree_pair_block(&rule, empty, &[], &[]).unwrap(),
            Vec::<Vec<(FusionTreePairKey, f64)>>::new()
        );

        let half = su2(1).id();
        let keys = vec![
            tree_pair_group_fixture(&[half, half], &[half, half], su2(0).id(), &[false; 2], &[false; 2]),
            tree_pair_group_fixture(&[half, half], &[half, half], su2(2).id(), &[false; 2], &[false; 2]),
        ];
        let expected = keys
            .iter()
            .cloned()
            .map(|key| vec![(key, 1.0)])
            .collect::<Vec<_>>();
        // What: distinct coupled labels remain valid basis states in one
        // external-sector group, preserving source order, destinations, and
        // coefficients on the symbol-free identity path.
        assert_eq!(
            multiplicity_free_braid_tree_pair_block(
                &SU2FusionRule,
                &keys,
                &[0, 1],
                &[2, 3],
                &[0, 1],
                &[2, 3],
            )
            .unwrap(),
            expected
        );
    }

    #[test]
    fn indexed_adjoint_tree_pair_block_reuses_parent_group_order() {
        let half = su2(1).id();
        let keys = [
            tree_pair_group_fixture(
                &[half, half],
                &[half, half],
                su2(0).id(),
                &[false; 2],
                &[false; 2],
            ),
            tree_pair_group_fixture(
                &[half, half],
                &[half, half],
                su2(0).id(),
                &[true, false],
                &[false; 2],
            ),
            tree_pair_group_fixture(
                &[half, half],
                &[half, half],
                su2(2).id(),
                &[false; 2],
                &[false; 2],
            ),
            tree_pair_group_fixture(
                &[half, half],
                &[half, half],
                su2(2).id(),
                &[true, false],
                &[false; 2],
            ),
        ];
        let structure = packed_fixture_structure(
            4,
            keys.iter().cloned().map(|key| (key, vec![1; 4])),
        )
        .unwrap();
        let eager_adjoint_keys = keys
            .iter()
            .map(|key| {
                FusionTreePairKey::pair(
                    key.domain_tree().clone(),
                    key.codomain_tree().clone(),
                )
            })
            .collect::<Vec<_>>();
        let eager_adjoint = packed_fixture_structure(
            4,
            eager_adjoint_keys
                .iter()
                .cloned()
                .map(|key| (key, vec![1; 4])),
        )
        .unwrap();

        // What: side swapping is a bijection on group labels, so an adjoint
        // projection preserves the parent group partition and storage order.
        assert_eq!(
            structure
                .fusion_tree_group_slice()
                .iter()
                .map(FusionTreeBlockGroup::block_indices)
                .collect::<Vec<_>>(),
            eager_adjoint
                .fusion_tree_group_slice()
                .iter()
                .map(FusionTreeBlockGroup::block_indices)
                .collect::<Vec<_>>(),
        );

        for group in structure.fusion_tree_group_slice() {
            let rows = multiplicity_free_permute_tree_pair_block_indexed(
                &SU2FusionRule,
                &structure,
                group.block_indices(),
                FusionTreePairOrientation::Adjoint,
                &[0, 1],
                &[2, 3],
            )
            .unwrap();
            // What: indexed adjoint identity emits logical swapped keys once,
            // in parent source order, without changing compact transform rows.
            assert_eq!(
                rows,
                group
                    .block_indices()
                    .iter()
                    .map(|&index| vec![(eager_adjoint_keys[index].clone(), 1.0)])
                    .collect::<Vec<_>>()
            );

            let prepared = PreparedTreePairOperation::prepare_permute(
                &SU2FusionRule,
                2,
                2,
                &[0, 1],
                &[2, 3],
            )
            .unwrap();
            let ordered = multiplicity_free_braid_tree_pair_block_ordered_indexed(
                &SU2FusionRule,
                &structure,
                group.block_indices(),
                FusionTreePairOrientation::Adjoint,
                &prepared,
            )
            .unwrap();
            // What: the ordered indexed kernel exposes the same logical
            // adjoint columns while retaining parent block-index order.
            assert_eq!(ordered.source_count(), group.block_indices().len());
            assert_eq!(
                ordered.destinations(),
                group
                    .block_indices()
                    .iter()
                    .map(|&index| eager_adjoint_keys[index].clone())
                    .collect::<Vec<_>>()
            );

            let prepared = PreparedTreePairOperation::prepare_transpose(
                2,
                2,
                &[0, 1],
                &[2, 3],
            )
            .unwrap();
            let transposed = multiplicity_free_transpose_tree_pair_block_ordered_indexed(
                &SU2FusionRule,
                &structure,
                group.block_indices(),
                FusionTreePairOrientation::Adjoint,
                &prepared,
            )
            .unwrap();
            // What: transpose uses the same oriented parent projection and
            // preserves the canonical logical source-column count.
            assert_eq!(transposed.source_count(), group.block_indices().len());
            assert_eq!(transposed.destinations(), ordered.destinations());
        }
    }

    #[test]
    fn empty_block_permute_preserves_braiding_style_error_precedence() {
        // What: an empty source block does not bypass the public permutation
        // API's symmetric-braiding capability check.
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(
                &FibonacciFusionRule,
                &[],
                &[],
                &[],
            )
            .unwrap_err(),
            CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: BraidingStyleKind::Anyonic,
            }
        );
    }

    #[test]
    fn split_only_tree_pair_braid_uses_only_the_required_bend() {
        // What: moving the split from 1|2 to 2|1 with unchanged linearized
        // external-leg order evaluates one bend and no braid symbols, for both
        // the single-source and block entry points.
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1], 1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        ).unwrap();

        let rule = SplitOnlyCountingRule::default();
        let single = multiplicity_free_braid_tree_pair(
            &rule,
            &source,
            &[0, 2],
            &[1],
            &[0],
            &[1, 2],
        )
        .unwrap();
        assert_eq!(rule.f_calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(rule.r_calls.load(std::sync::atomic::Ordering::Relaxed), 0);

        rule.f_calls
            .store(0, std::sync::atomic::Ordering::Relaxed);
        let block = multiplicity_free_braid_tree_pair_block(
            &rule,
            std::slice::from_ref(&source),
            &[0, 2],
            &[1],
            &[0],
            &[1, 2],
        )
        .unwrap();
        assert_eq!(rule.f_calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(rule.r_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(block, vec![single]);
    }

    #[test]
    fn split_only_nested_product_braid_matches_legacy_composition() {
        // What: a non-Abelian fZ2 x U(1) x SU(2) tree preserves the product
        // bend sign and duality bookkeeping of the old all-codomain route in
        // both split directions.
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = ProductRule::default();
        let coupled = rule.encode_sector(
            left_rule.encode_sector(z2_even(), u1(0)),
            su2(1),
        );
        let domain_left = rule.encode_sector(
            left_rule.encode_sector(z2_odd(), u1(1)),
            su2(1),
        );
        let domain_right = rule.encode_sector(
            left_rule.encode_sector(z2_odd(), u1(-1)),
            su2(2),
        );
        let source = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [coupled], coupled,
                [false],
                [],
                [],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &rule,
                [domain_left, domain_right], coupled,
                [false, true],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
        );

        let forward = multiplicity_free_braid_tree_pair(
            &rule,
            &source,
            &[0, 2],
            &[1],
            &[0],
            &[1, 2],
        )
        .unwrap();
        let expected = legacy_split_only_tree_pair_route(&rule, &source, 2).unwrap();
        assert_eq!(forward.len(), expected.len());
        assert!(forward[0].1 < 0.0);
        for ((actual_key, actual_coefficient), (expected_key, expected_coefficient)) in
            forward.iter().zip(&expected)
        {
            assert_eq!(actual_key, expected_key);
            assert!((actual_coefficient - expected_coefficient).abs() < 1.0e-12);
        }

        let reverse_source = &forward[0].0;
        let reverse = multiplicity_free_braid_tree_pair(
            &rule,
            reverse_source,
            &[0],
            &[2, 1],
            &[0, 1],
            &[2],
        )
        .unwrap();
        let reverse_expected =
            legacy_split_only_tree_pair_route(&rule, reverse_source, 1).unwrap();
        assert_eq!(reverse.len(), reverse_expected.len());
        assert!(reverse[0].1 < 0.0);
        for ((actual_key, actual_coefficient), (expected_key, expected_coefficient)) in
            reverse.iter().zip(&reverse_expected)
        {
            assert_eq!(actual_key, expected_key);
            assert!((actual_coefficient - expected_coefficient).abs() < 1.0e-12);
        }
    }

    #[test]
    fn identity_braid_rows_are_exact_for_supported_symmetry_families_and_rank_zero() {
        // What: identity rows preserve their exact source key and unit
        // coefficient for fermionic, non-abelian, product, and scalar spaces.
        let pair = |sector: SectorId| {
            FusionTreePairKey::try_pair_from_sector_ids(
                [sector.id()],
                [sector.id()], sector.id(),
                [false],
                [false],
                [],
                [],
                [],
                [],
            ).unwrap()
        };

        let fz2_source = pair(z2_odd());
        assert_eq!(
            multiplicity_free_braid_tree_pair(
                &FermionParityFusionRule,
                &fz2_source,
                &[0],
                &[1],
                &[41],
                &[2],
            )
            .unwrap(),
            vec![(fz2_source, 1.0)]
        );

        let su2_source = pair(su2(1));
        assert_eq!(
            multiplicity_free_braid_tree_pair(
                &SU2FusionRule,
                &su2_source,
                &[0],
                &[1],
                &[13],
                &[5],
            )
            .unwrap(),
            vec![(su2_source, 1.0)]
        );

        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let product_rule = FpU1Su2Rule::default();
        let product_sector = product_rule.encode_sector(
            left_rule.encode_sector(z2_odd(), u1(2)),
            su2(1),
        );
        let product_source = pair(product_sector);
        assert_eq!(
            multiplicity_free_braid_tree_pair(
                &product_rule,
                &product_source,
                &[0],
                &[1],
                &[8],
                &[3],
            )
            .unwrap(),
            vec![(product_source, 1.0)]
        );

        let scalar_source = FusionTreePairKey::try_pair_from_sector_ids(
            Vec::<usize>::new(),
            Vec::<usize>::new(), z2_even().id(),
            Vec::<bool>::new(),
            Vec::<bool>::new(),
            Vec::<usize>::new(),
            Vec::<usize>::new(),
            Vec::<usize>::new(),
            Vec::<usize>::new(),
        ).unwrap();
        assert_eq!(
            multiplicity_free_braid_tree_pair(
                &Z2FusionRule,
                &scalar_source,
                &[],
                &[],
                &[],
                &[],
            )
            .unwrap(),
            vec![(scalar_source, 1.0)]
        );
    }

    #[test]
    fn borrowed_unique_schedule_derivation_uses_quadratic_position_queries() {
        // What: operation-local inverse positions keep lazy Artin schedule
        // derivation quadratic in runtime rank; this deliberately measures
        // schedule lowering only, not the subsequent tree execution.
        let rank = 129;
        let permutation = (0..rank).rev().collect::<Vec<_>>();
        let raw_axis_positions = (0..rank).rev().collect::<Vec<_>>();
        let prepared = PreparedTreePairOperation::prepare_permute_with_raw_axis_positions(
            &Z2FusionRule,
            rank,
            0,
            &permutation,
            &[],
            &raw_axis_positions,
        )
        .unwrap();

        reset_unique_borrowed_position_queries();
        let steps = prepared
            .plan
            .artin_steps()
            .expect("reverse permutation must prepare a braid")
            .count();
        assert_eq!(steps, rank * (rank - 1) / 2);
        assert_eq!(unique_borrowed_position_queries(), rank * (rank - 1));
    }

    #[test]
    fn block_structure_finds_fusion_tree_subblock_by_key() {
        let first = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let second = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [0], 0,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let structure = packed_fixture_structure(
            2,
            [
                (BlockKey::from(second.clone()), vec![1, 4]),
                (BlockKey::from(first.clone()), vec![2, 3]),
            ],
        )
        .unwrap();

        let first_block = structure.fusion_tree_pair_block(&first).unwrap();
        let second_block = structure
            .block_by_key(&BlockKey::from(second.clone()))
            .unwrap();

        assert_eq!(first_block.key(), &BlockKey::from(first));
        assert_eq!(first_block.shape(), &[2, 3]);
        assert_eq!(first_block.offset(), 4);
        assert_eq!(second_block.key(), &BlockKey::from(second));
        assert_eq!(second_block.shape(), &[1, 4]);
        assert_eq!(second_block.offset(), 0);
    }

    #[test]
    fn tensormap_subblock_by_tree_returns_matching_view() {
        let first = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let second = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [0], 0,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let structure = packed_fixture_structure(
            2,
            [
                (BlockKey::from(second.clone()), vec![1, 2]),
                (BlockKey::from(first.clone()), vec![2, 2]),
            ],
        )
        .unwrap();
        let space = TensorMapSpace::<1, 1>::from_dims([3], [3]).unwrap();
        let tensor = TensorMap::<i32, 1, 1>::from_vec_with_structure(
            vec![10, 20, 30, 40, 50, 60],
            space,
            structure,
        )
        .unwrap();

        let first_view = tensor.subblock_by_tree(&first).unwrap();
        let second_view = tensor.block_by_key(&BlockKey::from(second)).unwrap();

        assert_eq!(first_view.shape(), &[2, 2]);
        assert_eq!(first_view.offset(), 2);
        assert_eq!(
            &first_view.data()[first_view.offset()..first_view.offset() + 4],
            &[30, 40, 50, 60]
        );
        assert_eq!(second_view.shape(), &[1, 2]);
        assert_eq!(second_view.offset(), 0);
    }

    #[test]
    fn subblock_by_tree_reports_missing_fusion_tree_key() {
        let existing = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [0], 0,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let missing = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let structure =
            packed_fixture_structure(2, [(BlockKey::from(existing), vec![1, 1])]).unwrap();
        let space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let tensor =
            TensorMap::<f64, 1, 1>::from_vec_with_structure(vec![1.0], space, structure).unwrap();

        let err = tensor.subblock_by_tree(&missing).unwrap_err();

        assert_eq!(
            err,
            CoreError::MissingBlockKey {
                key: Box::new(BlockKey::from(missing)),
            }
        );
    }

    #[test]
    fn public_u1_irrep_roundtrips_compact_ids_and_fuses() {
        let rule = U1FusionRule;
        let charges = [
            U1Irrep::new(-2),
            U1Irrep::new(-1),
            U1Irrep::new(0),
            U1Irrep::new(1),
            U1Irrep::new(2),
        ];
        let ids = charges.map(SectorId::from);

        assert_eq!(
            ids,
            [
                SectorId::new(3),
                SectorId::new(1),
                SectorId::new(0),
                SectorId::new(2),
                SectorId::new(4),
            ]
        );
        for charge in charges {
            assert_eq!(U1Irrep::from_sector_id(charge.sector_id()), Some(charge));
        }
        assert_eq!(rule.vacuum(), U1Irrep::new(0).sector_id());
        assert_eq!(
            rule.dual(U1Irrep::new(3).sector_id()),
            U1Irrep::new(-3).sector_id()
        );
        assert_eq!(
            rule.fusion_channels(U1Irrep::new(-2).sector_id(), U1Irrep::new(5).sector_id())
                .to_vec(),
            vec![U1Irrep::new(3).sector_id()]
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn packed_product_codec_covers_the_builtin_leaf_domains() {
        // What: the packed codec preserves the full 32-bit U1 component range,
        // including the raw ID reserved by the semantic label constructor,
        // together with every currently supported SU2 label.
        type FpU1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type FpU1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
        type TripleCodec = PackedProductCodec<FpU1Layout, Su2SectorLayout>;

        for charge in [
            excluded_u1_id(),
            u1(-1),
            u1(0),
            u1(1),
            u1(i32::MAX),
        ] {
            for twice_spin in [0, 1, 127, 254] {
                let inner = FpU1Codec::encode(z2_odd(), charge);
                let encoded = TripleCodec::encode(inner, su2(twice_spin));
                let (decoded_inner, decoded_spin) = TripleCodec::decode(encoded).unwrap();
                let (decoded_parity, decoded_charge) =
                    FpU1Codec::decode(decoded_inner).unwrap();
                assert_eq!(decoded_parity, z2_odd());
                assert_eq!(decoded_charge, charge);
                assert_eq!(decoded_spin, su2(twice_spin));
            }
        }
    }

    #[test]
    fn product_sector_api_exposes_only_generic_composition() {
        let pair = product_sector(z2_odd(), u1(2));
        let encoded = pair.sector_id_with::<TensorKitProductCodec>();
        assert_eq!(encoded, TensorKitProductCodec::encode(z2_odd(), u1(2)));
        assert_eq!(pair.left(), &z2_odd());
        assert_eq!(pair.right(), &u1(2));

        let left_rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
        let chained_rule = FermionParityFusionRule
            .product(U1FusionRule)
            .product(SU2FusionRule);
        let left_sector = |parity, charge| left_rule.encode_sector(parity, u1(charge));
        let chained_sector = |parity, charge, twice_spin| {
            chained_rule.encode_sector(left_sector(parity, charge), su2(twice_spin))
        };

        let a = chained_sector(z2_odd(), 1, 1);
        let b = chained_sector(z2_odd(), -1, 1);
        let c0 = chained_sector(z2_even(), 0, 0);
        let c2 = chained_sector(z2_even(), 0, 2);

        assert_eq!(chained_rule.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(chained_rule.braiding_style(), BraidingStyleKind::Fermionic);
        assert_eq!(chained_rule.fusion_channels(a, b).to_vec(), vec![c0, c2]);
    }

    #[test]
    fn product_fusion_rule_combines_fermion_parity_and_u1_componentwise() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = FpU1Rule::default();
        let sector = |parity, charge| rule.encode_sector(parity, u1(charge));
        let odd_two = sector(z2_odd(), 2);
        let odd_minus_five = sector(z2_odd(), -5);
        let even_minus_three = sector(z2_even(), -3);

        assert_eq!(rule.fusion_style(), FusionStyleKind::Unique);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Fermionic);
        assert_eq!(rule.vacuum(), sector(z2_even(), 0));
        assert_eq!(rule.dual(odd_two), sector(z2_odd(), -2));
        assert_eq!(
            rule.fusion_channels(odd_two, odd_minus_five).to_vec(),
            vec![even_minus_three]
        );
        assert_eq!(rule.nsymbol(odd_two, odd_minus_five, even_minus_three), 1);
        assert_eq!(
            rule.r_symbol_scalar(odd_two, odd_minus_five, even_minus_three),
            -1.0
        );
        assert_eq!(rule.sqrt_dim_scalar(odd_two), 1.0);
    }

    #[test]
    fn product_fusion_rule_nested_fz2_u1_su2_channels_and_symbols_match_tensorkit() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = FpU1Su2Rule::default();
        let left_sector = |parity, charge| left_rule.encode_sector(parity, u1(charge));
        let sector = |parity, charge, twice_spin| {
            rule.encode_sector(left_sector(parity, charge), su2(twice_spin))
        };

        let a = sector(z2_odd(), 1, 1);
        let b = sector(z2_odd(), -1, 1);
        let c0 = sector(z2_even(), 0, 0);
        let c2 = sector(z2_even(), 0, 2);

        assert_eq!(rule.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Fermionic);
        assert_eq!(rule.dual(a), sector(z2_odd(), -1, 1));
        assert_eq!(rule.fusion_channels(a, b).to_vec(), vec![c0, c2]);
        assert_eq!(rule.r_symbol_scalar(a, b, c0), 1.0);
        assert_eq!(rule.r_symbol_scalar(a, b, c2), -1.0);
        assert!((rule.sqrt_dim_scalar(c2) - 3.0_f64.sqrt()).abs() < 1.0e-12);

        let vacuum_left = left_sector(z2_even(), 0);
        let spin_half = rule.encode_sector(vacuum_left, su2(1));
        let spin_zero = rule.encode_sector(vacuum_left, su2(0));
        assert!(
            (rule.f_symbol_scalar(
                spin_half, spin_half, spin_half, spin_half, spin_zero, spin_zero,
            ) + 0.5)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn product_fusion_tree_homspace_matches_tensorkit_fz2_u1_su2_fixture() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = FpU1Su2Rule::default();
        let left_sector = |parity, charge| left_rule.encode_sector(parity, u1(charge));
        let sector = |parity, charge, twice_spin| {
            rule.encode_sector(left_sector(parity, charge), su2(twice_spin))
        };

        let a = sector(z2_odd(), 1, 1);
        let b = sector(z2_odd(), -1, 1);
        let c0 = sector(z2_even(), 0, 0);
        let c1 = sector(z2_even(), 0, 2);
        assert_eq!(a.id(), 43);
        assert_eq!(b.id(), 19);
        assert_eq!(c0.id(), 0);
        assert_eq!(c1.id(), 3);

        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(a, 1)], false),
                SectorLeg::new([(b, 1)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
        );
        let keys = hom.fusion_tree_keys(&rule);

        assert_eq!(keys.len(), 2);
        for (key, coupled) in keys.iter().zip([c0, c1]) {
            assert_eq!(key.coupled(), coupled);
            assert_eq!(key.codomain_uncoupled(), &[a, b]);
            assert_eq!(key.domain_uncoupled(), &[coupled]);
            assert_eq!(key.codomain_is_dual(), &[false, false]);
            assert_eq!(key.domain_is_dual(), &[false]);
            assert_eq!(key.codomain_innerlines(), &[]);
            assert_eq!(key.domain_innerlines(), &[]);
            assert_eq!(key.codomain_vertices(), &[MultiplicityIndex::ONE]);
            assert_eq!(key.domain_vertices(), &[]);
        }
    }

    #[test]
    // Message updated for #971: the component's FusionRule::fusion_channels
    // panic is now derived from CheckedFusionAlgebra::try_fusion_channels
    // rather than restated separately; the invalid sector it names is
    // unchanged.
    #[should_panic(expected = "invalid fusion sector SectorId(2)")]
    fn product_fusion_rule_panics_on_component_invalid_sector_like_existing_rules() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = FpU1Rule::default();
        let invalid_left_component = rule.encode_sector(SectorId::new(2), u1(0));
        let valid = rule.encode_sector(z2_even(), u1(0));

        let _ = rule.fusion_channels(invalid_left_component, valid);
    }

    #[test]
    fn public_su2_irrep_fusion_channels_match_doubled_spin_order() {
        let rule = SU2FusionRule;

        assert_eq!(
            rule.fusion_channels(
                SU2Irrep::from_twice_spin(1).sector_id(),
                SU2Irrep::from_twice_spin(2).sector_id(),
            )
            .to_vec(),
            vec![
                SU2Irrep::from_twice_spin(1).sector_id(),
                SU2Irrep::from_twice_spin(3).sector_id(),
            ]
        );
    }

    #[test]
    fn public_su2_f_and_r_symbols_match_tensorkit_values() {
        let rule = SU2FusionRule;
        let s = |twice_spin| SU2Irrep::from_twice_spin(twice_spin).sector_id();
        let cases = [
            ((1, 1, 1, 1, 0, 0), -0.5),
            ((1, 1, 1, 1, 0, 2), 0.866_025_403_784_438_6),
            ((1, 1, 1, 1, 2, 0), 0.866_025_403_784_438_6),
            ((1, 1, 1, 1, 2, 2), 0.5),
            ((1, 2, 1, 2, 1, 1), -1.0 / 3.0),
            ((2, 2, 2, 2, 0, 2), -0.577_350_269_189_625_7),
            ((2, 2, 2, 2, 2, 2), 0.5),
            ((1, 1, 2, 2, 1, 1), 0.0),
        ];

        for ((a, b, c, d, e, f), expected) in cases {
            let actual = rule.f_symbol_scalar(s(a), s(b), s(c), s(d), s(e), s(f));
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "F({a},{b},{c},{d},{e},{f}) = {actual}, expected {expected}"
            );
        }
        assert_eq!(rule.r_symbol_scalar(s(1), s(1), s(0)), -1.0);
        assert_eq!(rule.r_symbol_scalar(s(1), s(1), s(2)), 1.0);
        assert_eq!(rule.r_symbol_scalar(s(1), s(2), s(0)), 0.0);
    }

    #[test]
    fn multiplicity_free_su2_repartition_matches_tensorkit_bend_factor() {
        let rule = SU2FusionRule;
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [false],
            [],
            [],
            [],
            [],
        ).unwrap();

        let all_codomain = multiplicity_free_repartition_tree_pair(&rule, &source, 2).unwrap();
        assert_eq!(all_codomain.len(), 1);
        assert_eq!(
            all_codomain[0].0.codomain_uncoupled(),
            &[SectorId::new(1); 2]
        );
        assert_eq!(all_codomain[0].0.codomain_is_dual(), &[false, true]);
        assert_eq!(all_codomain[0].0.codomain_innerlines(), &[]);
        assert_eq!(all_codomain[0].0.codomain_vertices(), &[MultiplicityIndex::ONE]);
        assert_eq!(
            all_codomain[0].0.codomain_tree().coupled(),
            SectorId::new(0)
        );
        assert_eq!(all_codomain[0].0.domain_uncoupled(), &[]);
        assert_eq!(
            all_codomain[0].0.domain_tree().coupled(),
            SectorId::new(0)
        );
        assert!((all_codomain[0].1 - 2.0_f64.sqrt()).abs() < 1.0e-12);

        let all_domain = multiplicity_free_repartition_tree_pair(&rule, &source, 0).unwrap();
        assert_eq!(all_domain.len(), 1);
        assert_eq!(all_domain[0].0.codomain_uncoupled(), &[]);
        assert_eq!(
            all_domain[0].0.codomain_tree().coupled(),
            SectorId::new(0)
        );
        assert_eq!(all_domain[0].0.domain_uncoupled(), &[SectorId::new(1); 2]);
        assert_eq!(all_domain[0].0.domain_is_dual(), &[false, true]);
        assert!((all_domain[0].1 - 2.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn multiplicity_free_su2_permute_tree_pair_matches_tensorkit_swap() {
        let rule = SU2FusionRule;
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [false],
            [],
            [],
            [],
            [],
        ).unwrap();

        let permuted = multiplicity_free_permute_tree_pair(&rule, &source, &[1], &[0]).unwrap();

        assert_eq!(permuted.len(), 1);
        assert_eq!(permuted[0].0.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted[0].0.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted[0].0.codomain_is_dual(), &[true]);
        assert_eq!(permuted[0].0.domain_is_dual(), &[true]);
        assert_eq!(
            permuted[0].0.codomain_tree().coupled(),
            SectorId::new(1)
        );
        assert_eq!(
            permuted[0].0.domain_tree().coupled(),
            SectorId::new(1)
        );
        assert!((permuted[0].1 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn prepared_simple_pair_matches_explicit_generic_composition() {
        // What: a prepared Simple-fusion pair operation equals the explicit
        // repartition -> all-codomain Artin -> repartition composition.
        let rule = SU2FusionRule;
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [false],
            [],
            [],
            [],
            [],
        ).unwrap();
        let prepared =
            PreparedTreePairOperation::prepare_permute(&rule, 1, 1, &[1], &[0]).unwrap();
        let actual = prepared
            .execute_multiplicity_free(&rule, &source)
            .unwrap();

        let all_codomain = multiplicity_free_repartition_tree_pair(&rule, &source, 2).unwrap();
        let braided = compose_tree_pair_terms(&rule, all_codomain, |rule, key| {
            multiplicity_free_braid_tree(
                rule,
                key.codomain_tree(),
                &[1, 0],
                &[0, 1],
            )
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|(tree, coefficient)| {
                        (
                            FusionTreePairKey::pair(
                                tree,
                                key.domain_tree().clone(),
                            ),
                            coefficient,
                        )
                    })
                    .collect::<Vec<_>>()
            })
        })
        .unwrap();
        let expected = multiplicity_free_repartition_terms(&rule, braided, 1).unwrap();

        assert_eq!(
            actual.iter().map(|(key, _)| key).collect::<Vec<_>>(),
            expected.iter().map(|(key, _)| key).collect::<Vec<_>>()
        );
        for ((_, actual), (_, expected)) in actual.iter().zip(expected) {
            assert!((*actual - expected).abs() < 1.0e-12);
        }
    }

    fn u1_nonselfdual_tree_pair_fixture() -> FusionTreePairKey {
        FusionTreePairKey::pair(
            FusionTreeKey::new(
                [u1(1), u1(2)], u1(3),
                [false, false],
                Vec::<SectorId>::new(),
                [MultiplicityIndex::ONE],
            ),
            FusionTreeKey::new(
                [u1(3)], u1(3),
                [false],
                Vec::<SectorId>::new(),
                Vec::<MultiplicityIndex>::new(),
            ),
        )
    }

    #[test]
    fn u1_bendright_dualizes_visible_sector_and_flips_isdual_like_tensorkit() {
        let out = multiplicity_free_bendright_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(1)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), u1(1));
        assert_eq!(out[0].0.codomain_is_dual(), &[false]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(3), u1(-2)]);
        assert_eq!(out[0].0.domain_tree().coupled(), u1(1));
        assert_eq!(out[0].0.domain_is_dual(), &[false, true]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[MultiplicityIndex::ONE]);
    }

    #[test]
    fn u1_foldright_dualizes_first_visible_sector_and_flips_isdual_like_tensorkit() {
        let out = multiplicity_free_foldright_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(2)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), u1(2));
        assert_eq!(out[0].0.codomain_is_dual(), &[false]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(-1), u1(3)]);
        assert_eq!(out[0].0.domain_tree().coupled(), u1(2));
        assert_eq!(out[0].0.domain_is_dual(), &[true, false]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[MultiplicityIndex::ONE]);
    }

    #[test]
    fn u1_repartition_to_all_domain_matches_tensorkit_nonselfdual_fixture() {
        let out = multiplicity_free_repartition_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
            0,
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[]);
        assert_eq!(out[0].0.codomain_tree().coupled(), u1(0));
        assert_eq!(out[0].0.codomain_is_dual(), &[] as &[bool]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(3), u1(-2), u1(-1)]);
        assert_eq!(out[0].0.domain_tree().coupled(), u1(0));
        assert_eq!(out[0].0.domain_is_dual(), &[false, true, true]);
        assert_eq!(out[0].0.domain_innerlines(), &[u1(1)]);
        assert_eq!(
            out[0].0.domain_vertices(),
            &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
        );
    }

    #[test]
    fn u1_repartition_to_all_codomain_matches_tensorkit_nonselfdual_fixture() {
        let out = multiplicity_free_repartition_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
            3,
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(1), u1(2), u1(-3)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), u1(0));
        assert_eq!(out[0].0.codomain_is_dual(), &[false, false, true]);
        assert_eq!(out[0].0.codomain_innerlines(), &[u1(3)]);
        assert_eq!(
            out[0].0.codomain_vertices(),
            &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
        );
        assert_eq!(out[0].0.domain_uncoupled(), &[]);
        assert_eq!(out[0].0.domain_tree().coupled(), u1(0));
        assert_eq!(out[0].0.domain_is_dual(), &[] as &[bool]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[]);
    }

    #[test]
    fn u1_transpose_cyclic_23_1_matches_tensorkit_nonselfdual_fixture() {
        let out = multiplicity_free_transpose_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
            &[1, 2],
            &[0],
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(2), u1(-3)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), u1(-1));
        assert_eq!(out[0].0.codomain_is_dual(), &[false, true]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[MultiplicityIndex::ONE]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(-1)]);
        assert_eq!(out[0].0.domain_tree().coupled(), u1(-1));
        assert_eq!(out[0].0.domain_is_dual(), &[true]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[]);
    }

    #[test]
    fn nested_product_elementary_bend_keeps_the_fermionic_phase() {
        // What: the elementary bend of an odd fZ2 pair retains the negative
        // product-category phase independently of the block/per-source runners.
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = ProductRule::default();
        let coupled =
            rule.encode_sector(left_rule.encode_sector(z2_even(), u1(0)), su2(1));
        let odd_half =
            rule.encode_sector(left_rule.encode_sector(z2_odd(), u1(1)), su2(1));
        let odd_one =
            rule.encode_sector(left_rule.encode_sector(z2_odd(), u1(-1)), su2(2));
        let source = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [coupled], coupled,
                [false],
                [],
                [],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &rule,
                [odd_half, odd_one], coupled,
                [false, true],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
        );

        let bent = multiplicity_free_bendleft_tree_pair(&rule, &source).unwrap();
        assert_eq!(bent.len(), 1);
        assert!(bent[0].1 < 0.0);
        let restored = multiplicity_free_bendright_tree_pair(&rule, &bent[0].0).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].0, source);
        assert!((bent[0].1 * restored[0].1 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn typed_sector_homspace_builds_u1_tree_key() {
        let rule = U1FusionRule;
        let hom = FusionTreeHomSpace::from_sectors([(U1Irrep::new(2), 1)], [(U1Irrep::new(2), 1)]);

        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[U1Irrep::new(2).sector_id(), U1Irrep::new(-2).sector_id()],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[U1Irrep::new(2).sector_id()]);
        assert_eq!(key.domain_uncoupled(), &[U1Irrep::new(2).sector_id()]);
        assert_eq!(key.coupled(), U1Irrep::new(2).sector_id());
    }

    #[test]
    fn fusion_tensor_space_builds_subblockstructure_from_homspace() {
        let rule = Z2FusionRule;
        let dense = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                [(SectorId::new(0), 1), (SectorId::new(1), 3)],
                false,
            )]),
            FusionProductSpace::new([SectorLeg::new(
                [(SectorId::new(0), 2), (SectorId::new(1), 1)],
                false,
            )]),
        );

        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            [vec![1, 2], vec![3, 1]],
        )
        .unwrap();

        assert_eq!(fusion_space.subblock_structure().block_count(), 2);
        assert_eq!(fusion_space.required_len().unwrap(), 5);
        assert_eq!(
            fusion_space.subblock_structure().block(0).unwrap().key(),
            &BlockKey::from(FusionTreePairKey::try_pair_from_sector_ids(
                [0],
                [0], 0,
                [false],
                [false],
                [],
                [],
                [],
                [],
            ).unwrap())
        );
        assert_eq!(
            fusion_space.subblock_structure().block(1).unwrap().shape(),
            &[3, 1]
        );
    }

    #[test]
    fn packed_block_structure_records_rank_offsets_and_required_len() {
        let structure = BlockStructure::packed_column_major(2, [vec![2, 3], vec![1, 4]]).unwrap();

        assert_eq!(structure.rank(), 2);
        assert_eq!(structure.block_count(), 2);
        assert_eq!(structure.sector_structure().block_count(), 2);
        assert_eq!(structure.degeneracy_structure().block_count(), 2);
        let first = structure.block(0).unwrap();
        assert_eq!(first.key(), &BlockKey::ordinal(0));
        assert_eq!(first.shape(), &[2, 3]);
        assert_eq!(first.strides(), &[1, 2]);
        assert_eq!(first.offset(), 0);
        let second = structure.block(1).unwrap();
        assert_eq!(second.key(), &BlockKey::ordinal(1));
        assert_eq!(second.shape(), &[1, 4]);
        assert_eq!(second.strides(), &[1, 1]);
        assert_eq!(second.offset(), 6);
        assert_eq!(structure.required_len().unwrap(), 10);
    }

    #[test]
    fn block_structure_rejects_duplicate_keys() {
        let first =
            BlockSpec::column_major_with_key(BlockKey::opaque([7]), vec![2, 2], 0).unwrap();
        let second =
            BlockSpec::column_major_with_key(BlockKey::opaque([7]), vec![1, 3], 4).unwrap();

        let err = BlockStructure::from_blocks_with_rank(2, vec![first, second]).unwrap_err();

        assert_eq!(
            err,
            CoreError::DuplicateBlockKey {
                key: Box::new(BlockKey::opaque([7]))
            }
        );
    }

    #[test]
    fn block_structure_validates_degeneracy_before_sector_keys() {
        // What: preparing an owned block structure preserves the historical
        // error order when both degeneracy metadata and sector keys are bad.
        let key = BlockKey::opaque([7]);
        let first = BlockSpec::column_major_with_key(key.clone(), vec![2, 2], 0).unwrap();
        let malformed = BlockSpec {
            key,
            shape: smallvec![1, 3],
            strides: smallvec![1],
            offset: 4,
        };

        assert_eq!(
            BlockStructure::from_blocks_with_rank(2, vec![first, malformed]),
            Err(CoreError::RankMismatch {
                shape: 2,
                strides: 1,
            })
        );
    }

    #[test]
    fn fusion_tree_pair_key_records_tensorkit_subblock_pair_fields() {
        let key = FusionTreePairKey::try_pair_from_sector_ids(
            [2, 3],
            [5, 7], 11,
            [false, true],
            [true, false],
            [13],
            [17],
            [19, 23],
            [29, 31],
        ).unwrap();

        assert_eq!(
            key.codomain_uncoupled(),
            &[SectorId::new(2), SectorId::new(3)]
        );
        assert_eq!(
            key.domain_uncoupled(),
            &[SectorId::new(5), SectorId::new(7)]
        );
        assert_eq!(key.coupled(), SectorId::new(11));
        assert_eq!(key.codomain_is_dual(), &[false, true]);
        assert_eq!(key.domain_is_dual(), &[true, false]);
        assert_eq!(key.codomain_innerlines(), &[SectorId::new(13)]);
        assert_eq!(key.domain_innerlines(), &[SectorId::new(17)]);
        assert_eq!(
            key.codomain_vertices(),
            &[
                MultiplicityIndex::new(19).unwrap(),
                MultiplicityIndex::new(23).unwrap(),
            ]
        );
        assert_eq!(
            key.domain_vertices(),
            &[
                MultiplicityIndex::new(29).unwrap(),
                MultiplicityIndex::new(31).unwrap(),
            ]
        );

        let group = key.group_key();
        assert_eq!(group.codomain_uncoupled(), key.codomain_uncoupled());
        assert_eq!(group.domain_uncoupled(), key.domain_uncoupled());
        assert_eq!(group.codomain_is_dual(), key.codomain_is_dual());
        assert_eq!(group.domain_is_dual(), key.domain_is_dual());
    }

    #[test]
    fn sorted_lookup_distinguishes_rank1_tree_duality() {
        let nondual = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [], 0,
            [false],
            [],
            [],
            [],
            [],
            [],
        ).unwrap();
        let dual = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [], 0,
            [true],
            [],
            [],
            [],
            [],
            [],
        ).unwrap();
        let structure = SectorStructure::from_keys(1, [BlockKey::from(nondual)]).unwrap();

        assert!(!structure.has_compact_lookup());
        assert_eq!(structure.find_index(&BlockKey::from(dual.clone())), None);
        assert_eq!(structure.find_fusion_tree_pair_index(&dual), None);
    }

    #[test]
    fn unique_homspace_builds_subblock_key_from_external_sectors() {
        let rule = Z2FusionRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1)], [(1, 1)]);

        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[SectorId::new(1), SectorId::new(1)],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.coupled(), SectorId::new(1));
        assert_eq!(key.codomain_is_dual(), &[false]);
        assert_eq!(key.domain_is_dual(), &[false]);
    }

    #[test]
    fn unique_homspace_dualizes_domain_external_sectors_like_tensorkit() {
        let rule = Z4PointedRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        );

        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[SectorId::new(1), SectorId::new(3)],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.coupled(), SectorId::new(1));
    }

    #[test]
    fn fusion_tree_pair_key_external_sectors_restore_visible_domain_sector() {
        let rule = Z4PointedRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], true)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        );
        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[SectorId::new(1), SectorId::new(3)],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(
            key.external_sectors(&rule),
            vec![SectorId::new(1), SectorId::new(3)]
        );
        assert_eq!(key.external_is_dual(), vec![true, false]);
    }

    #[test]
    fn canonical_coupled_grid_matches_literal_u1_sector_matrices() {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(0), 2), (u1(1), 3)], false),
                SectorLeg::new([(u1(0), 5), (u1(1), 7)], true),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(u1(0), 11), (u1(1), 13)], true),
                SectorLeg::new([(u1(0), 17), (u1(1), 19)], false),
            ]),
        );

        let structure = homspace
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap();
        let expected = [
            (
                [u1(0), u1(0)],
                [u1(0), u1(0)],
                u1(0),
                [2, 5, 11, 17],
                [1, 2, 10, 110],
                0,
            ),
            (
                [u1(1), u1(0)],
                [u1(1), u1(0)],
                u1(1),
                [3, 5, 13, 17],
                [1, 3, 29, 377],
                1870,
            ),
            (
                [u1(0), u1(1)],
                [u1(1), u1(0)],
                u1(1),
                [2, 7, 13, 17],
                [1, 2, 29, 377],
                1885,
            ),
            (
                [u1(1), u1(0)],
                [u1(0), u1(1)],
                u1(1),
                [3, 5, 11, 19],
                [1, 3, 29, 319],
                8279,
            ),
            (
                [u1(0), u1(1)],
                [u1(0), u1(1)],
                u1(1),
                [2, 7, 11, 19],
                [1, 2, 29, 319],
                8294,
            ),
            (
                [u1(1), u1(1)],
                [u1(1), u1(1)],
                u1(2),
                [3, 7, 13, 19],
                [1, 3, 21, 273],
                14340,
            ),
        ];

        assert_eq!(structure.block_count(), expected.len());
        for (index, (codomain, domain, coupled, shape, strides, offset)) in
            expected.iter().enumerate()
        {
            let block = structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                panic!("canonical symmetric block must use a fusion-tree key");
            };
            assert_eq!(key.codomain_uncoupled(), codomain);
            assert_eq!(key.domain_uncoupled(), domain);
            assert_eq!(key.codomain_is_dual(), &[false, true]);
            assert_eq!(key.domain_is_dual(), &[true, false]);
            assert_eq!(key.coupled(), *coupled);
            assert_eq!(block.shape(), shape);
            assert_eq!(block.strides(), strides);
            assert_eq!(block.offset(), *offset);
        }
        assert_eq!(structure.required_len().unwrap(), 19527);

        let regions = structure.coupled_sector_regions(2).unwrap().unwrap();
        assert_eq!(regions.len(), 3);
        assert_eq!(
            regions
                .iter()
                .map(|region| (
                    region.coupled(),
                    region.rows(),
                    region.cols(),
                    region.range(),
                ))
                .collect::<Vec<_>>(),
            vec![
                (u1(0), 10, 187, 0..1870),
                (u1(1), 29, 430, 1870..14340),
                (u1(2), 21, 247, 14340..19527),
            ]
        );
    }

    #[test]
    fn canonical_su2_innerline_grid_matches_literal_sector_matrices() {
        let half = su2(1);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(half, 2)], false),
                SectorLeg::new([(half, 3)], true),
                SectorLeg::new([(half, 5)], false),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(half, 7)], true),
                SectorLeg::new([(half, 11)], false),
                SectorLeg::new([(half, 13)], true),
            ]),
        );

        let layout = homspace.fusion_tree_layout_data_uncached(&SU2FusionRule);
        assert_eq!(layout.keys.len(), 5);
        assert_eq!(layout.sectors.len(), 2);
        assert_eq!(
            (
                layout.sectors[0].start,
                layout.sectors[0].row_count,
                layout.sectors[0].col_count,
            ),
            (0, 2, 2)
        );
        assert_eq!(
            (
                layout.sectors[1].start,
                layout.sectors[1].row_count,
                layout.sectors[1].col_count,
            ),
            (4, 1, 1)
        );
        let expected_innerlines = [
            (su2(0), su2(0), su2(1)),
            (su2(2), su2(0), su2(1)),
            (su2(0), su2(2), su2(1)),
            (su2(2), su2(2), su2(1)),
            (su2(2), su2(2), su2(3)),
        ];
        for (key, &(codomain_inner, domain_inner, coupled)) in
            layout.keys.iter().zip(&expected_innerlines)
        {
            assert_eq!(key.codomain_uncoupled(), &[half, half, half]);
            assert_eq!(key.domain_uncoupled(), &[half, half, half]);
            assert_eq!(key.codomain_is_dual(), &[false, true, false]);
            assert_eq!(key.domain_is_dual(), &[true, false, true]);
            assert_eq!(key.codomain_innerlines(), &[codomain_inner]);
            assert_eq!(key.domain_innerlines(), &[domain_inner]);
            assert_eq!(key.coupled(), coupled);
        }

        let structure = homspace
            .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
            .unwrap();
        let expected_offsets = [0, 30, 60060, 60090, 120120];
        for (index, &offset) in expected_offsets.iter().enumerate() {
            let block = structure.block(index).unwrap();
            assert_eq!(block.shape(), &[2, 3, 5, 7, 11, 13]);
            if index < 4 {
                assert_eq!(block.strides(), &[1, 2, 6, 60, 420, 4620]);
            } else {
                assert_eq!(block.strides(), &[1, 2, 6, 30, 210, 2310]);
            }
            assert_eq!(block.offset(), offset);
        }
        assert_eq!(structure.required_len().unwrap(), 150150);
        let regions = structure.coupled_sector_regions(3).unwrap().unwrap();
        assert_eq!(regions.len(), 2);
        assert_eq!(
            (
                regions[0].coupled(),
                regions[0].rows(),
                regions[0].cols(),
                regions[0].range(),
            ),
            (su2(1), 60, 2002, 0..120120)
        );
        assert_eq!(
            (
                regions[1].coupled(),
                regions[1].rows(),
                regions[1].cols(),
                regions[1].range(),
            ),
            (su2(3), 30, 1001, 120120..150150)
        );
    }

    #[test]
    fn expert_cantor_product_keeps_the_encoded_fallback() {
        // What: custom-codec/expert product construction remains available via
        // the unchanged encoded public entry and preserves its historical IDs.
        type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = Rule::new(FermionParityFusionRule, U1FusionRule);
        let sector = TensorKitProductCodec::encode(z2_odd(), u1(2));
        let hom = singleton_rank_hom(sector, 3);
        assert_eq!(
            hom.fusion_tree_keys(&rule).as_ref(),
            hom.fusion_tree_keys_uncached(&rule)
        );
    }

    #[test]
    fn fusion_layout_lookup_and_reset_are_concurrent_safe_after_poison() {
        // What: a poisoned layout lock and concurrent lookup/reset cannot publish
        // partial layout content or return a structure with the wrong keys/shapes.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        let poisoned = std::panic::catch_unwind(|| {
            let _write = fusion_tree_layout_cache()
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            panic!("poison fusion layout cache for recovery test");
        });
        assert!(poisoned.is_err());

        let workers = (0..4)
            .map(|worker| {
                std::thread::spawn(move || {
                    let rule = U1FusionRule;
                    for iteration in 0..64 {
                        if worker == 0 && iteration % 8 == 0 {
                            reset_core_intern_tables();
                        }
                        let charge = worker * 100 + iteration;
                        let hom = FusionTreeHomSpace::from_sectors(
                            [(U1Irrep::new(charge), 2)],
                            [(U1Irrep::new(charge), 3)],
                        );
                        let keys = hom.fusion_tree_keys(&rule);
                        assert_eq!(keys.len(), 1);
                        assert_eq!(keys[0].coupled(), U1Irrep::new(charge).sector_id());
                        let structure = hom
                            .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
                            .unwrap();
                        assert_eq!(structure.block_count(), 1);
                        assert_eq!(structure.block(0).unwrap().shape(), &[2, 3]);
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    }

    #[test]
    fn braid_tree_pair_block_matches_per_source() {
        use std::collections::BTreeMap;
        let rule = SU2FusionRule;
        let leg = || {
            SectorLeg::new(
                [
                    (SectorId::new(0), 1),
                    (SectorId::new(1), 1),
                    (SectorId::new(2), 1),
                ],
                false,
            )
        };
        // (V⊗V⊗V) ← V spans many uncoupled blocks; test each block.
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg(), leg()]),
            FusionProductSpace::new([leg()]),
        );
        let keys = hom.fusion_tree_keys(&rule);

        // Group source tree-pairs by their uncoupled block (the batching unit).
        let mut blocks: BTreeMap<Vec<usize>, Vec<FusionTreePairKey>> = BTreeMap::new();
        for key in keys.iter() {
            let tag: Vec<usize> = key
                .codomain_tree()
                .uncoupled()
                .iter()
                .chain(key.domain_tree().uncoupled())
                .map(|s| s.id())
                .collect();
            blocks.entry(tag).or_default().push(key.clone());
        }

        // Global leg indices: codomain legs 0,1,2 and domain leg 3. Reverse the
        // codomain, keep the domain leg in place.
        let codomain_permutation = [2usize, 1, 0];
        let domain_permutation = [3usize];
        let mut checked_blocks = 0;
        for src_keys in blocks.values() {
            let batched = multiplicity_free_permute_tree_pair_block(
                &rule,
                src_keys,
                &codomain_permutation,
                &domain_permutation,
            )
            .unwrap();
            assert_eq!(batched.len(), src_keys.len());
            for (src, batched_rows) in src_keys.iter().zip(&batched) {
                let per_source = multiplicity_free_permute_tree_pair(
                    &rule,
                    src,
                    &codomain_permutation,
                    &domain_permutation,
                )
                .unwrap();
                // Compare as key -> coefficient maps within double-precision tol.
                let mut want: BTreeMap<FusionTreePairKey, f64> = BTreeMap::new();
                for (k, c) in &per_source {
                    *want.entry(k.clone()).or_insert(0.0) += c;
                }
                let mut got: BTreeMap<FusionTreePairKey, f64> = BTreeMap::new();
                for (k, c) in batched_rows {
                    *got.entry(k.clone()).or_insert(0.0) += c;
                }
                assert_eq!(
                    want.keys().collect::<Vec<_>>(),
                    got.keys().collect::<Vec<_>>(),
                    "destination trees differ for a source in block"
                );
                for (k, wc) in &want {
                    let gc = got[k];
                    assert!(
                        (wc - gc).abs() <= 1e-12 * (1.0 + wc.abs()),
                        "coefficient mismatch {wc} vs {gc}"
                    );
                }
            }
            checked_blocks += 1;
        }
        assert!(checked_blocks > 0, "expected at least one block");
    }

    fn assert_compact_operator_cohorts<R>(
        rule: &R,
        sources: &[FusionTreePairKey],
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + MultiplicityFreeFusionRule,
    {
        for cohort_len in [1usize, 2, 4, 8, 16] {
            let cohort = &sources[..cohort_len];

            // What: a direct compact bend-left produces the exact public
            // full-key rows in source and first-appearance destination order.
            let group = validate_tree_pair_block_group_for_rule(rule, cohort)
                .unwrap()
                .unwrap();
            let basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
            let (basis, columns) = compact_bendleft_block_first(rule, basis).unwrap();
            let got = scatter_compact_block(basis, columns);
            let want = cohort
                .iter()
                .map(|source| {
                    multiplicity_free_bendleft_tree_pair(rule, source)
                        .unwrap()
                        .into_vec()
                })
                .collect::<Vec<_>>();
            assert_eq!(got, want);

            // What: a direct compact bend-right preserves the same one-row
            // oracle and ordering for every requested block cohort size.
            let group = validate_tree_pair_block_group_for_rule(rule, cohort)
                .unwrap()
                .unwrap();
            let basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
            let (basis, columns) = compact_bendright_block_first(rule, basis).unwrap();
            let got = scatter_compact_block(basis, columns);
            let want = cohort
                .iter()
                .map(|source| {
                    multiplicity_free_bendright_tree_pair(rule, source)
                        .unwrap()
                        .into_vec()
                })
                .collect::<Vec<_>>();
            assert_eq!(got, want);

            // What: compact non-first Artin rows use the public F/R kernel and
            // retain its channel order for SU(2)-branching source cohorts.
            let group = validate_tree_pair_block_group_for_rule(rule, cohort)
                .unwrap()
                .unwrap();
            let basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
            let (basis, columns) =
                compact_codomain_artin_block_first(rule, basis, 3, false).unwrap();
            let got = scatter_compact_block(basis, columns);
            let want = cohort
                .iter()
                .map(|source| {
                    let domain = source.domain_tree().clone();
                    multiplicity_free_artin_braid_at_with_inverse(
                        rule,
                        source.codomain_tree(),
                        3,
                        false,
                    )
                    .unwrap()
                    .into_iter()
                    .map(|(codomain, coefficient)| {
                        (
                            FusionTreePairKey::pair(codomain, domain.clone()),
                            coefficient,
                        )
                    })
                    .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn compact_block_operators_match_su2_cohort_oracles() {
        let rule = SU2FusionRule;
        let sources = compact_operator_cohort_fixture(&rule, su2(2), su2(2));

        assert_compact_operator_cohorts(&rule, &sources);
    }

    #[test]
    fn compact_block_operators_match_fermionic_product_cohort_oracles() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left = FpU1Rule::default();
        let rule = ProductRule::default();
        let external =
            rule.encode_sector(left.encode_sector(z2_odd(), u1(0)), su2(2));
        let coupled =
            rule.encode_sector(left.encode_sector(z2_even(), u1(0)), su2(2));
        let sources = compact_operator_cohort_fixture(&rule, external, coupled);

        assert_compact_operator_cohorts(&rule, &sources);
    }

    fn assert_all_codomain_compact_cohorts<R>(
        rule: &R,
        sources: &[FusionTreeKey],
    ) where
        R: MultiplicityFreeFusionSymbols<Scalar = f64> + MultiplicityFreeFusionRule,
    {
        let cases = [
            ([1usize, 0, 2, 3, 4, 5, 6, 7], [0usize, 1, 2, 3, 4, 5, 6, 7]),
            ([2usize, 0, 1, 3, 4, 5, 6, 7], [2usize, 0, 1, 3, 4, 5, 6, 7]),
        ];
        for cohort_len in [1usize, 2, 4, 8, 16] {
            let cohort = &sources[..cohort_len];
            for (permutation, levels) in cases {
                let got =
                    multiplicity_free_braid_tree_block(rule, cohort, &permutation, &levels)
                        .unwrap();
                let want = cohort
                    .iter()
                    .map(|source| {
                        multiplicity_free_braid_tree(rule, source, &permutation, &levels)
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(got.len(), want.len());
                for (got_row, want_row) in got.iter().zip(&want) {
                    // What: compact all-codomain execution preserves the scalar
                    // kernel's destination order as well as every full key.
                    assert_eq!(
                        got_row.iter().map(|(key, _)| key).collect::<Vec<_>>(),
                        want_row.iter().map(|(key, _)| key).collect::<Vec<_>>()
                    );
                    assert_eq!(got_row.len(), want_row.len());
                    for ((_, got_coefficient), (_, want_coefficient)) in
                        got_row.iter().zip(want_row)
                    {
                        assert!(
                            (got_coefficient - want_coefficient).abs()
                                <= 1.0e-12 * (1.0 + want_coefficient.abs())
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn all_codomain_compact_block_matches_su2_cohorts() {
        let rule = SU2FusionRule;
        let sources = compact_operator_cohort_fixture(&rule, su2(2), su2(2))
            .into_iter()
            .map(|source| source.codomain_tree().clone())
            .collect::<Vec<_>>();

        assert_all_codomain_compact_cohorts(&rule, &sources);
    }

    #[test]
    fn all_codomain_compact_block_matches_fermionic_product_cohorts() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left = FpU1Rule::default();
        let rule = ProductRule::default();
        let external =
            rule.encode_sector(left.encode_sector(z2_odd(), u1(0)), su2(2));
        let coupled =
            rule.encode_sector(left.encode_sector(z2_even(), u1(0)), su2(2));
        let sources = compact_operator_cohort_fixture(&rule, external, coupled)
            .into_iter()
            .map(|source| source.codomain_tree().clone())
            .collect::<Vec<_>>();

        assert_all_codomain_compact_cohorts(&rule, &sources);
    }

    #[test]
    fn transpose_tree_pair_block_matches_full_key_su2_cycles_and_repartition() {
        use std::collections::BTreeMap;
        let rule = SU2FusionRule;
        let leg = || {
            SectorLeg::new(
                [
                    (SectorId::new(0), 1),
                    (SectorId::new(1), 1),
                    (SectorId::new(2), 1),
                ],
                false,
            )
        };
        // (V⊗V⊗V) ← V spans many uncoupled blocks; test each block.
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg(), leg()]),
            FusionProductSpace::new([leg()]),
        );
        let keys = hom.fusion_tree_keys(&rule);

        // Group source tree-pairs by their uncoupled block (the batching unit).
        let mut blocks: BTreeMap<Vec<usize>, Vec<FusionTreePairKey>> = BTreeMap::new();
        for key in keys.iter() {
            let tag: Vec<usize> = key
                .codomain_tree()
                .uncoupled()
                .iter()
                .chain(key.domain_tree().uncoupled())
                .map(|s| s.id())
                .collect();
            blocks.entry(tag).or_default().push(key.clone());
        }

        let mut checked_blocks = 0;
        for (codomain_permutation, domain_permutation, uses_dense_block) in [
            (vec![3usize], vec![2usize, 1, 0], true),
            (vec![1usize, 2, 3], vec![0usize], true),
            (vec![0usize, 1], vec![3usize, 2], false),
            (vec![0usize, 1, 2, 3], vec![], false),
        ] {
            for src_keys in blocks.values() {
                assert_compact_transpose_matches_full_key_oracle(
                    &rule,
                    src_keys,
                    &codomain_permutation,
                    &domain_permutation,
                    uses_dense_block,
                );
                checked_blocks += 1;
            }
        }
        assert!(checked_blocks > 0, "expected at least one block");
    }

    #[test]
    fn transpose_tree_pair_block_matches_full_key_fermionic_product_cycle() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left = FpU1Rule::default();
        let rule = ProductRule::default();
        let coupled = rule.encode_sector(left.encode_sector(z2_even(), u1(0)), su2(1));
        let odd_half = rule.encode_sector(left.encode_sector(z2_odd(), u1(1)), su2(1));
        let odd_one = rule.encode_sector(left.encode_sector(z2_odd(), u1(-1)), su2(2));
        let source = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [coupled],
                coupled,
                [false],
                [],
                [],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &rule,
                [odd_half, odd_one],
                coupled,
                [false, true],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
        );

        // What: nested fZ2 x U1 x SU2 keeps the fermionic pivotal phase,
        // non-self-dual charge, and non-Abelian channel through a cycle.
        assert_compact_transpose_matches_full_key_oracle(
            &rule,
            &[source],
            &[2, 1],
            &[0],
            true,
        );
    }

    #[test]
    fn transpose_tree_pair_block_matches_low_rank_and_nonselfdual_u1_oracles() {
        let empty = FusionTreeKey::new([], u1(0), [], [], []);
        let rank_zero = [FusionTreePairKey::pair(empty.clone(), empty)];
        // What: scalar transpose remains the symbol-free identity operation.
        assert_compact_transpose_matches_full_key_oracle(
            &U1FusionRule,
            &rank_zero,
            &[],
            &[],
            false,
        );

        let rank_one_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], false)]),
            FusionProductSpace::new([]),
        );
        let rank_one = rank_one_hom.fusion_tree_keys(&U1FusionRule);
        assert_eq!(rank_one.len(), 1);
        // What: moving one vacuum leg across the partition uses only the final
        // full-key reconstruction boundary.
        assert_compact_transpose_matches_full_key_oracle(
            &U1FusionRule,
            &rank_one,
            &[],
            &[0],
            false,
        );

        // What: a non-self-dual U1 cycle preserves sector dualization and flags.
        assert_compact_transpose_matches_full_key_oracle(
            &U1FusionRule,
            &[u1_nonselfdual_tree_pair_fixture()],
            &[1, 2],
            &[0],
            true,
        );
    }

    #[test]
    fn tensormap_subblocks_by_sectors_returns_all_su2_simple_innerline_blocks() {
        let rule = SU2FusionRule;
        let half = SectorId::new(1);
        let dense = TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap();
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);
        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
        )
        .unwrap();
        let tensor =
            TensorMap::<i32, 3, 1>::from_vec_with_fusion_space(vec![11, 22], fusion_space).unwrap();

        let blocks = tensor
            .subblocks_by_sectors(&rule, &[half, half, half, half])
            .unwrap();

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].offset(), 0);
        assert_eq!(blocks[0].data()[blocks[0].offset()], 11);
        assert_eq!(blocks[1].offset(), 1);
        assert_eq!(blocks[1].data()[blocks[1].offset()], 22);

        let err = tensor
            .subblock_by_sectors(&rule, &[half, half, half, half])
            .unwrap_err();
        assert_eq!(
            err,
            CoreError::BlockCountMismatch {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn fusion_tree_groups_preserve_structure_order_and_ignore_internal_tree_data() {
        let first = BlockKey::from(FusionTreePairKey::try_pair_from_sector_ids(
            [10, 20],
            [30], 5,
            [false, true],
            [true],
            [101],
            [201],
            [301, 302],
            [401],
        ).unwrap());
        let second = BlockKey::from(FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [2, 3], 4,
            [true],
            [false, true],
            [],
            [202],
            [303],
            [402, 403],
        ).unwrap());
        let same_group_as_first = BlockKey::from(FusionTreePairKey::try_pair_from_sector_ids(
            [10, 20],
            [30], 6,
            [false, true],
            [true],
            [102],
            [203],
            [304, 305],
            [404],
        ).unwrap());

        let keys = vec![first.clone(), second.clone(), same_group_as_first.clone()];
        let sector = SectorStructure::from_keys(2, keys.clone()).unwrap();
        let block_structure =
            packed_fixture_structure(2, keys.into_iter().map(|key| (key, vec![1, 1]))).unwrap();

        let sector_groups = sector.fusion_tree_groups();
        let block_groups = block_structure.fusion_tree_groups();
        assert_eq!(sector_groups, block_groups);
        let mut legacy_groups = Vec::<FusionTreeBlockGroup>::new();
        for (index, key) in [first, second, same_group_as_first].iter().enumerate() {
            let group_key = key.fusion_tree_group_key().unwrap();
            if let Some(group) = legacy_groups
                .iter_mut()
                .find(|group| group.group_key() == &group_key)
            {
                group.block_indices.push(index);
            } else {
                legacy_groups.push(FusionTreeBlockGroup::new(group_key, vec![index]));
            }
        }
        // What: construction-time metadata is exactly the former eager
        // first-appearance grouping, including interleaved storage indices.
        assert_eq!(sector_groups, legacy_groups);
        assert_eq!(sector_groups.len(), 2);
        assert_eq!(sector_groups[0].block_indices(), &[0, 2]);
        assert_eq!(sector_groups[1].block_indices(), &[1]);
        assert_eq!(
            sector_groups[0].group_key(),
            &FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true])
        );
        assert_eq!(
            sector_groups[1].group_key(),
            &FusionTreeGroupKey::from_sector_ids([1], [2, 3], [true], [false, true])
        );
    }

    #[test]
    fn sector_structure_rejects_every_mixed_key_kind_pair() {
        let fusion = BlockKey::from(FusionTreePairKey::try_pair_from_sector_ids(
            [7],
            [8], 9,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap());
        let keys = [
            BlockKey::trivial(),
            BlockKey::opaque([7]),
            fusion,
        ];
        for expected in 0..keys.len() {
            for actual in 0..keys.len() {
                if expected == actual {
                    continue;
                }
                assert_eq!(
                    SectorStructure::from_keys(
                        2,
                        [keys[expected].clone(), keys[actual].clone()]
                    )
                    .unwrap_err(),
                    CoreError::MixedBlockKeyKinds {
                        expected: keys[expected].kind(),
                        actual: keys[actual].kind(),
                    }
                );
            }
        }

        let dense = BlockStructure::trivial(&[2, 3]).unwrap();
        let empty = BlockStructure::empty(2);
        assert!(dense.fusion_tree_groups().is_empty());
        assert!(empty.fusion_tree_groups().is_empty());
    }

    #[test]
    fn opaque_block_key_words_are_rank_independent_application_identity() {
        let key = OpaqueBlockKey::from_words([3, 5, 8, 13, 21]);
        let block_key = BlockKey::from(key.clone());
        let rank_one = SectorStructure::from_keys(1, [block_key.clone()]).unwrap();
        let rank_seven = SectorStructure::from_keys(7, [block_key.clone()]).unwrap();

        // What: opaque word count has no relationship to tensor rank, and the
        // public constructors preserve all application routing words.
        assert_eq!(key.words(), &[3, 5, 8, 13, 21]);
        assert_eq!(OpaqueBlockKey::new(vec![3, 5, 8, 13, 21]), key);
        assert_eq!(rank_one.key(0).unwrap(), &block_key);
        assert_eq!(rank_seven.key(0).unwrap(), &block_key);
        assert_eq!(OpaqueBlockKey::ordinal(34).words(), &[34]);
    }

    #[allow(deprecated)]
    #[test]
    fn deprecated_sector_key_constructors_are_opaque_compatibility_helpers() {
        let from_sectors = BlockKey::sectors([SectorId::new(3), SectorId::new(5)]);
        let from_ids = BlockKey::sector_ids([3, 5]);

        // What: legacy numeric block labels preserve routing identity without
        // being promoted to categorical fusion-tree pairs.
        assert_eq!(from_sectors, BlockKey::opaque([3, 5]));
        assert_eq!(from_ids, BlockKey::opaque([3, 5]));
        assert!(matches!(from_sectors, BlockKey::Opaque(_)));
        assert!(matches!(from_ids, BlockKey::Opaque(_)));
    }

    #[test]
    fn empty_sector_structure_has_one_canonical_namespace_free_form() {
        let constructed =
            SectorStructure::from_keys(3, std::iter::empty::<BlockKey>()).unwrap();
        let canonical = SectorStructure::empty(3);

        // What: both public empty constructors produce the same namespace-free
        // structure and never allocate a meaningless compact lookup.
        assert_eq!(constructed, canonical);
        assert_eq!(constructed.key_kind(), None);
        assert!(!constructed.has_compact_lookup());
    }

    #[test]
    fn block_structure_separates_sector_and_degeneracy_data() {
        let sector = SectorStructure::from_keys(
            2,
            [BlockKey::opaque([0, 1]), BlockKey::opaque([1, 0])],
        )
        .unwrap();
        let degeneracy =
            DegeneracyStructure::packed_column_major(2, [vec![2, 3], vec![3, 2]]).unwrap();
        let structure = BlockStructure::from_parts(sector, degeneracy).unwrap();

        assert_eq!(structure.rank(), 2);
        assert_eq!(
            structure.sector_structure().key(0).unwrap(),
            &BlockKey::opaque([0, 1])
        );
        assert_eq!(
            structure.sector_structure().key(1).unwrap(),
            &BlockKey::opaque([1, 0])
        );
        assert_eq!(
            structure.degeneracy_structure().block(0).unwrap().shape(),
            &[2, 3]
        );
        assert_eq!(
            structure.degeneracy_structure().block(1).unwrap().offset(),
            6
        );
        assert_eq!(structure.required_len().unwrap(), 12);
    }

    #[test]
    fn sector_structure_pairs_compact_keys_without_map_lookup() {
        let dst = SectorStructure::from_keys(
            2,
            [
                BlockKey::opaque([2]),
                BlockKey::opaque([0]),
                BlockKey::opaque([1]),
            ],
        )
        .unwrap();
        let src = SectorStructure::from_keys(
            2,
            [
                BlockKey::opaque([0]),
                BlockKey::opaque([1]),
                BlockKey::opaque([2]),
            ],
        )
        .unwrap();

        assert!(src.has_compact_lookup());
        assert_eq!(dst.find_index(&BlockKey::opaque([0])), Some(1));
        assert_eq!(src.find_index(&BlockKey::opaque([2])), Some(2));
        assert_eq!(dst.pair_indices_from(&src).unwrap(), vec![2, 0, 1]);
    }

    #[test]
    fn lookup_never_aliases_different_key_namespaces() {
        let dense = SectorStructure::dense(0);
        let opaque_zero = SectorStructure::from_keys(0, [BlockKey::ordinal(0)]).unwrap();
        let fusion_pair = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [], 0,
            [false],
            [],
            [],
            [],
            [],
            [],
        ).unwrap();
        let fusion = SectorStructure::from_keys(1, [fusion_pair.clone()]).unwrap();
        let opaque_one = SectorStructure::from_keys(1, [BlockKey::ordinal(1)]).unwrap();

        // What: compact integer routing is only an accelerator inside the
        // Dense/Opaque namespaces and never establishes categorical identity.
        assert_eq!(dense.find_index(&BlockKey::ordinal(0)), None);
        assert_eq!(opaque_zero.find_index(&BlockKey::Dense), None);
        assert_eq!(
            opaque_one.find_index(&BlockKey::from(fusion_pair.clone())),
            None
        );
        assert_eq!(
            opaque_one.find_fusion_tree_pair_index(&fusion_pair),
            None
        );
        assert!(!fusion.has_compact_lookup());
        assert_eq!(
            opaque_one.pair_indices_from(&fusion),
            Err(CoreError::MixedBlockKeyKinds {
                expected: BlockKeyKind::Opaque,
                actual: BlockKeyKind::FusionTree,
            })
        );
    }

    #[allow(deprecated)]
    #[test]
    fn deprecated_fusion_tree_lookup_forwarders_match_canonical_results() {
        let present = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [], 1,
            [false],
            [],
            [],
            [],
            [],
            [],
        ).unwrap();
        let missing = FusionTreePairKey::try_pair_from_sector_ids(
            [2],
            [], 2,
            [false],
            [],
            [],
            [],
            [],
            [],
        ).unwrap();
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(present.clone().into(), vec![1], 0).unwrap(),
        ])
        .unwrap();
        let sectors = structure.sector_structure();

        // What: all three renamed lookup/block APIs retain exact success and
        // missing behavior through their deprecated forwarding layer.
        assert_eq!(
            sectors.find_fusion_tree_index(&present),
            sectors.find_fusion_tree_pair_index(&present)
        );
        assert_eq!(
            sectors.find_fusion_tree_index(&missing),
            sectors.find_fusion_tree_pair_index(&missing)
        );
        assert_eq!(
            structure.find_block_index_by_fusion_tree_key(&present),
            structure.find_block_index_by_fusion_tree_pair(&present)
        );
        assert_eq!(
            structure.find_block_index_by_fusion_tree_key(&missing),
            structure.find_block_index_by_fusion_tree_pair(&missing)
        );
        assert_eq!(
            structure.fusion_tree_block(&present).unwrap().key(),
            structure.fusion_tree_pair_block(&present).unwrap().key()
        );
        assert_eq!(
            structure.fusion_tree_block(&missing).unwrap_err(),
            structure.fusion_tree_pair_block(&missing).unwrap_err()
        );
    }

    #[test]
    fn sector_structure_pairs_general_opaque_keys_by_sorted_merge() {
        let key_a = BlockKey::opaque([0, 1]);
        let key_b = BlockKey::opaque([1, 0]);
        let dst = SectorStructure::from_keys(2, [key_b.clone(), key_a.clone()]).unwrap();
        let src = SectorStructure::from_keys(2, [key_a.clone(), key_b.clone()]).unwrap();

        assert!(!src.has_compact_lookup());
        assert_eq!(dst.find_index(&key_a), Some(1));
        assert_eq!(src.find_index(&key_b), Some(1));
        assert_eq!(dst.pair_indices_from(&src).unwrap(), vec![1, 0]);
    }

    #[test]
    fn toy_om_rule_nsymbol_reports_outer_multiplicity() {
        let rule = ToyOmRule;
        let a = SectorId::new(ToyOmRule::A);
        let c = SectorId::new(ToyOmRule::C);
        let vacuum = SectorId::new(ToyOmRule::VACUUM);

        assert_eq!(rule.fusion_style(), FusionStyleKind::Generic);
        assert_eq!(rule.nsymbol(a, a, c), 2);
        // Everything else in this toy rule stays multiplicity-free (0 or 1).
        assert_eq!(rule.nsymbol(a, vacuum, a), 1);
        assert_eq!(rule.nsymbol(a, a, vacuum), 0);
    }

    #[test]
    fn toy_om_rule_f_symbol_generic_has_nsymbol_shaped_block() {
        let rule = ToyOmRule;
        let a = SectorId::new(ToyOmRule::A);
        let c = SectorId::new(ToyOmRule::C);
        let vacuum = SectorId::new(ToyOmRule::VACUUM);

        // F(a,a,0,c,c,a): shape (N(a,a,c), N(c,0,c), N(a,0,a), N(a,a,c))
        // = (2, 1, 1, 2).
        let f = rule.f_symbol_generic(a, a, vacuum, c, c, a);
        assert_eq!(f.shape(), (2, 1, 1, 2));
        assert_eq!(f.data().len(), 4);

        // The (mu, lambda) 2x2 block (nu = kappa = 0) is an orthogonal
        // rotation: R^T R == I.
        let m = [
            [*f.get(0, 0, 0, 0), *f.get(0, 0, 0, 1)],
            [*f.get(1, 0, 0, 0), *f.get(1, 0, 0, 1)],
        ];
        for col in 0..2 {
            for other in 0..2 {
                let dot: f64 = (0..2).map(|row| m[row][col] * m[row][other]).sum();
                let expected = if col == other { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "F (mu,lambda) block is not orthogonal at columns {col},{other}: {dot}"
                );
            }
        }
    }

    #[test]
    fn toy_om_rule_r_symbol_generic_has_nsymbol_shaped_matrix() {
        let rule = ToyOmRule;
        let a = SectorId::new(ToyOmRule::A);
        let c = SectorId::new(ToyOmRule::C);

        let r = rule.r_symbol_generic(a, a, c);
        assert_eq!(r.shape(), (2, 2));
        assert_eq!(r.data().len(), 4);
    }

    #[test]
    fn frozen_fusion_tree_identity_and_group_projection_are_content_based() {
        // What: independently allocated equal keys compare and hash by content,
        // while group projection reuses the pair's external owners.
        let a = SectorId::new(ToyOmRule::A);
        let c = SectorId::new(ToyOmRule::C);
        let first = FusionTreeKey::new(
            [a, a],
            c,
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        );
        let second = FusionTreeKey::new(
            [a, a],
            c,
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        );
        assert!(!Arc::ptr_eq(&first.uncoupled, &second.uncoupled));
        assert_eq!(first, second);
        assert_eq!(first.cmp(&second), std::cmp::Ordering::Equal);

        let mut cache = std::collections::HashMap::new();
        cache.insert(first.clone(), 7usize);
        assert_eq!(cache.get(&second), Some(&7));

        let pair = FusionTreePairKey::pair(first, second);
        let group = pair.group_key();
        assert!(Arc::ptr_eq(
            &group.codomain_uncoupled,
            &pair.codomain_tree.uncoupled
        ));
        assert!(Arc::ptr_eq(
            &group.domain_uncoupled,
            &pair.domain_tree.uncoupled
        ));
        assert!(Arc::ptr_eq(
            &group.codomain_is_dual,
            &pair.codomain_tree.is_dual
        ));
        assert!(Arc::ptr_eq(
            &group.domain_is_dual,
            &pair.domain_tree.is_dual
        ));
    }

    #[test]
    fn tree_pair_axis_validation_remains_linear_above_inline_bitset_capacity() {
        let rule = Z2FusionRule;
        let rank = 129;
        let codomain = FusionTreeKey::try_new_for_rule(
            &rule,
            vec![SectorId::new(0); rank], SectorId::new(0),
            vec![false; rank],
            vec![SectorId::new(0); rank - 2],
            vec![MultiplicityIndex::ONE; rank - 1],
        )
        .unwrap();
        let domain =
            FusionTreeKey::try_new_for_rule(&rule, [], SectorId::new(0), [], [], [])
                .unwrap();
        let pair = FusionTreePairKey::pair(codomain, domain);
        let identity = (0..rank).collect::<Vec<_>>();

        let rows =
            multiplicity_free_permute_tree_pair(&rule, &pair, &identity, &[]).unwrap();
        assert_eq!(rows, vec![(pair.clone(), 1.0)]);

        let mut duplicate = identity;
        duplicate[rank - 1] = rank - 2;
        assert!(matches!(
            multiplicity_free_permute_tree_pair(&rule, &pair, &duplicate, &[]),
            Err(CoreError::InvalidPermutation { .. })
        ));
    }

    #[test]
    fn checked_generic_tree_split_is_structural_and_provider_typed() {
        let rule = UnitaryToyOmRule;
        let a = SectorId::new(UnitaryToyOmRule::A);
        let c = SectorId::new(UnitaryToyOmRule::C);
        let tree = FusionTreeKey::new([a, a, a], a, [false, false, false], [c], [
            MultiplicityIndex::ONE,
            MultiplicityIndex::ONE,
        ]);

        let (front, tail) = split_fusion_tree_generic_checked(&rule, &tree, 1).unwrap();
        assert_eq!(front.uncoupled(), &[a]);
        assert_eq!(front.coupled(), a);
        assert_eq!(tail.uncoupled(), &[a, a, a]);
        assert_eq!(tail.coupled(), a);
        assert_eq!(tail.is_dual(), &[false, false, false]);
    }

    #[test]
    fn checked_generic_artin_matches_legacy_outer_and_inner_rows() {
        let rule = UnitaryToyOmRule;
        for (tree, index) in [(unitary_rank2_tree(1), 0), (unitary_rank3_tree(1), 1)] {
            let legacy = generic_artin_braid_at_with_inverse(&rule, &tree, index, false).unwrap();
            let checked = generic_artin_braid_at_with_inverse_checked(&rule, &tree, index, false).unwrap();
            assert_eq!(checked, legacy);
        }
    }

    #[test]
    fn checked_generic_merge_rejects_a_malformed_f_before_emitting_terms() {
        let rule = ArtinSpy { bad_f: true, ..ArtinSpy::new() };
        let lhs = unitary_rank3_tree(1);
        let rhs = unitary_rank2_tree(1);
        let result = merge_fusion_trees_generic_checked(
                &rule,
                &lhs,
                &rhs,
                SectorId::new(UnitaryToyOmRule::A),
                MultiplicityIndex::ONE,
            );
        assert!(matches!(result, Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })), "{result:?}");
        assert!(rule.f_calls.get() > 0);
    }

    // Braid at `index` (inv=false) then braid every output at `index`
    // (inv=true), summing coefficients per final tree. TensorKit's inverse
    // identity: this must equal the input tree with coefficient 1.
    fn braid_then_inverse_braid(
        rule: &UnitaryToyOmRule,
        tree: &FusionTreeKey,
        index: usize,
    ) -> std::collections::HashMap<FusionTreeKey, f64> {
        let forward = generic_artin_braid_at_with_inverse(rule, tree, index, false).unwrap();
        let mut totals: std::collections::HashMap<FusionTreeKey, f64> =
            std::collections::HashMap::new();
        for (mid, c_forward) in forward {
            let inverse = generic_artin_braid_at_with_inverse(rule, &mid, index, true).unwrap();
            for (final_tree, c_inverse) in inverse {
                *totals.entry(final_tree).or_insert(0.0) += c_forward * c_inverse;
            }
        }
        totals
    }

    fn assert_is_identity_on(
        totals: &std::collections::HashMap<FusionTreeKey, f64>,
        original: &FusionTreeKey,
    ) {
        let on_original = totals.get(original).copied().unwrap_or(0.0);
        assert!(
            (on_original - 1.0).abs() < 1e-12,
            "braid × inverse must be 1 on the original tree, got {on_original}"
        );
        for (tree, coeff) in totals {
            if tree != original {
                assert!(
                    coeff.abs() < 1e-12,
                    "braid × inverse must be 0 off the original tree, got {coeff}"
                );
            }
        }
    }

    #[test]
    fn unitary_toy_om_rule_r_and_f_blocks_are_unitary() {
        // Precondition of the round-trip tests: the F/R blocks the braid uses
        // are genuinely unitary. Assert it here so the identity tests below
        // rest on a checked assumption, not an asserted-by-fiat one.
        let rule = UnitaryToyOmRule;
        let a = SectorId::new(UnitaryToyOmRule::A);
        let c = SectorId::new(UnitaryToyOmRule::C);
        let vacuum = SectorId::new(UnitaryToyOmRule::VACUUM);

        // R(a,a,c): 2×2, must satisfy Rᵀ R = I.
        let r = rule.r_symbol_generic(a, a, c);
        assert_eq!(r.shape(), (2, 2));
        for col in 0..2 {
            for other in 0..2 {
                let dot: f64 = (0..2).map(|row| r.get(row, col) * r.get(row, other)).sum();
                let expected = if col == other { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "R(a,a,c) is not unitary at columns {col},{other}: {dot}"
                );
            }
        }

        // F(a,a,a,a,c,c): the braid's F block, shape (2,1,2,1) => 2×2 in
        // (mu,kappa). Must be unitary (it is the identity here).
        let f_braid = rule.f_symbol_generic(a, a, a, a, c, c);
        assert_eq!(f_braid.shape(), (2, 1, 2, 1));
        for mu in 0..2 {
            for other in 0..2 {
                let dot: f64 = (0..2)
                    .map(|kappa| f_braid.get(mu, 0, kappa, 0) * f_braid.get(other, 0, kappa, 0))
                    .sum();
                let expected = if mu == other { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "F(a,a,a,a,c,c) is not unitary at rows {mu},{other}: {dot}"
                );
            }
        }

        // F(a,a,0,c,c,a): the non-identity unitary F block, (mu,lambda) 2×2.
        let f_rot = rule.f_symbol_generic(a, a, vacuum, c, c, a);
        assert_eq!(f_rot.shape(), (2, 1, 1, 2));
        for col in 0..2 {
            for other in 0..2 {
                let dot: f64 = (0..2)
                    .map(|row| f_rot.get(row, 0, 0, col) * f_rot.get(row, 0, 0, other))
                    .sum();
                let expected = if col == other { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "F(a,a,0,c,c,a) is not unitary at columns {col},{other}: {dot}"
                );
            }
        }
    }

    #[test]
    fn generic_braid_index0_output_count_matches_nsymbol() {
        // Test (2): the forward braid of [a,a]->c at index 0 produces exactly
        // N(a,a,c) = 2 output trees, labelled with vertices 1 and 2 (both
        // nonzero because Rθ is a full rotation).
        let rule = UnitaryToyOmRule;
        let tree = unitary_rank2_tree(1);
        let outputs = generic_artin_braid_at_with_inverse(&rule, &tree, 0, false).unwrap();
        assert_eq!(outputs.len(), 2, "expected N(a,a,c)=2 output trees");
        let mut labels: Vec<usize> = outputs
            .iter()
            .map(|(t, _)| t.vertices()[0].get())
            .collect();
        labels.sort_unstable();
        assert_eq!(labels, vec![1, 2], "output vertex labels must be {{1,2}}");
    }

    #[test]
    fn generic_braid_index1_output_count_matches_nsymbol() {
        // Test (2), index>0 branch: braid of [a,a,a]->a at index 1 produces the
        // single c'=c channel, σ ∈ {1,2} (N(a,a,c)=2), λ = 1 (N(c,a,a)=1).
        let rule = UnitaryToyOmRule;
        let tree = unitary_rank3_tree(1);
        let outputs = generic_artin_braid_at_with_inverse(&rule, &tree, 1, false).unwrap();
        assert_eq!(outputs.len(), 2, "expected 2 (σ) output trees at index 1");
        let mut labels: Vec<(usize, usize)> = outputs
            .iter()
            .map(|(t, _)| {
                assert_eq!(t.innerlines(), &[SectorId::new(UnitaryToyOmRule::C)]);
                (t.vertices()[0].get(), t.vertices()[1].get())
            })
            .collect();
        labels.sort_unstable();
        assert_eq!(labels, vec![(1, 1), (2, 1)]);
    }

    #[test]
    fn generic_braid_inverse_is_identity_rank2_index0() {
        // Test (1), index==0 branch, over every enumerated vertex assignment.
        let rule = UnitaryToyOmRule;
        for vertex in 1..=2 {
            let tree = unitary_rank2_tree(vertex);
            let totals = braid_then_inverse_braid(&rule, &tree, 0);
            assert_is_identity_on(&totals, &tree);
        }
    }

    #[test]
    fn generic_braid_inverse_is_identity_rank3_index0() {
        // Test (1) + (4): index==0 on a rank>2 tree (uses the innerline as the
        // coupled sector for R), over every OM vertex assignment.
        let rule = UnitaryToyOmRule;
        for vertex1 in 1..=2 {
            let tree = unitary_rank3_tree(vertex1);
            let totals = braid_then_inverse_braid(&rule, &tree, 0);
            assert_is_identity_on(&totals, &tree);
        }
    }

    #[test]
    fn generic_braid_inverse_is_identity_rank3_index1() {
        // Test (1) + (4): the index>1 branch (F-move + R·F̄·R̄ contraction),
        // over every OM vertex assignment.
        let rule = UnitaryToyOmRule;
        for vertex1 in 1..=2 {
            let tree = unitary_rank3_tree(vertex1);
            let totals = braid_then_inverse_braid(&rule, &tree, 1);
            assert_is_identity_on(&totals, &tree);
        }
    }

    #[test]
    fn generic_braid_tree_roundtrip_is_identity() {
        // Exercises `generic_braid_tree`: braid [a,a]->c with the swap
        // permutation under levels [0,1] (=> inv=false), then braid the outputs
        // back under levels [1,0] (=> inv=true). The composite must be identity.
        let rule = UnitaryToyOmRule;
        for vertex in 1..=2 {
            let tree = unitary_rank2_tree(vertex);
            let forward = generic_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
            let mut totals: std::collections::HashMap<FusionTreeKey, f64> =
                std::collections::HashMap::new();
            for (mid, c_forward) in forward {
                let back = generic_braid_tree(&rule, &mid, &[1, 0], &[1, 0]).unwrap();
                for (final_tree, c_back) in back {
                    *totals.entry(final_tree).or_insert(0.0) += c_forward * c_back;
                }
            }
            assert_is_identity_on(&totals, &tree);
        }
    }

    #[test]
    fn generic_braid_tree_identity_permutation_is_noop() {
        // The identity permutation decomposes to zero swaps: the tree returns
        // unchanged with coefficient 1.
        let rule = UnitaryToyOmRule;
        let tree = unitary_rank3_tree(2);
        let out = generic_braid_tree(&rule, &tree, &[0, 1, 2], &[0, 1, 2]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, tree);
        assert!((out[0].1 - 1.0).abs() < 1e-12);
    }

    // ===================================================================
    // ADVERSARIAL REFUTATION (refute/b1-verify2): stress the nontrivial-F
    // path that the B1 round-trip tests deliberately avoid (F forced = I).
    // ===================================================================

    // Same structure as UnitaryToyOmRule, but F(a,a,a,a,c,c) — the block the
    // index>1 braid actually reads — is a NONTRIVIAL 2x2 rotation (in the
    // (mu,kappa) plane), not the identity. A single elementary braid needs no
    // hexagon consistency, so we can (a) compare the impl's coefficients to an
    // INDEPENDENT re-evaluation of TK's formula written here from scratch, and
    // (b) assert the elementary braid matrix is unitary (F,R all unitary).
    #[derive(Clone, Copy, Debug)]
    struct RefuteOmRule;

    impl RefuteOmRule {
        const VACUUM: usize = 0;
        const A: usize = 1;
        const C: usize = 3;
        const R_THETA: f64 = std::f64::consts::PI / 5.0;
        const F_PHI: f64 = std::f64::consts::PI / 7.0; // nontrivial F angle
    }

    impl FusionRule for RefuteOmRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }
        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }
        fn vacuum(&self) -> SectorId {
            SectorId::new(Self::VACUUM)
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (Self::VACUUM, x) | (x, Self::VACUUM) => smallvec![SectorId::new(x)],
                (Self::A, Self::A) => smallvec![SectorId::new(Self::C)],
                (Self::A, Self::C) | (Self::C, Self::A) => smallvec![SectorId::new(Self::A)],
                _ => smallvec![SectorId::new(Self::VACUUM)],
            }
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (Self::A, Self::A, Self::C) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }

    impl GenericFusionSymbols for RefuteOmRule {
        type Scalar = f64;
        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            let shape = (
                self.nsymbol(a, b, e),
                self.nsymbol(e, c, d),
                self.nsymbol(b, c, f),
                self.nsymbol(a, f, d),
            );
            let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
            if ids == (Self::A, Self::A, Self::A, Self::A, Self::C, Self::C) {
                // shape (2,1,2,1); row-major flat idx = mu*2 + kappa. Rotation
                // R_phi in the (mu,kappa) plane: F[mu,0,kappa,0] = Rphi[mu,kappa].
                let (s, co) = Self::F_PHI.sin_cos();
                GenericFArray::new(vec![co, -s, s, co], shape)
            } else {
                // 1x1 blocks = 1.0 on every path these tests touch.
                let n = shape.0 * shape.1 * shape.2 * shape.3;
                GenericFArray::new(vec![1.0; n], shape)
            }
        }
        fn r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            let rows = self.nsymbol(a, b, c);
            let cols = self.nsymbol(b, a, c);
            if (a.id(), b.id(), c.id()) == (Self::A, Self::A, Self::C) {
                let (s, co) = Self::R_THETA.sin_cos();
                GenericRMatrix::new(vec![co, -s, s, co], rows, cols)
            } else {
                GenericRMatrix::new(vec![1.0; rows * cols], rows, cols)
            }
        }
    }

    fn refute_rank3_tree(vertex1: usize) -> FusionTreeKey {
        let a = SectorId::new(RefuteOmRule::A);
        let c = SectorId::new(RefuteOmRule::C);
        FusionTreeKey::new(
            [a, a, a], a,
            [false, false, false],
            [c],
            [MultiplicityIndex::new(vertex1).expect("test multiplicity label is one-based"), MultiplicityIndex::ONE],
        )
    }

    // INDEPENDENT re-evaluation of TK braiding_manipulations.jl:170-194 for the
    // rank-3 tree [a,a,a]->a braided at index 1. Written directly from the TK
    // source WITHOUT calling generic_artin_braid_at_with_inverse. Returns the
    // 2x2 braid matrix M[sigma][mu] over the OM channel (nu=lambda=rho=0 fixed,
    // c'=c, n_sigma=n_kappa=2).
    // Index-form loops kept on purpose: they mirror the TK formula verbatim.
    #[allow(clippy::needless_range_loop)]
    fn independent_braid_matrix_index1(rule: &RefuteOmRule, inverse: bool) -> [[f64; 2]; 2] {
        let a = SectorId::new(RefuteOmRule::A);
        let c = SectorId::new(RefuteOmRule::C);
        // TK naming for i>1 (0-based index==1 here): a=inner_ext[i-1]=leg0=a,
        // b=uncoupled[i]=a, c=inner_ext[i]=c, d=uncoupled[i+1]=a, e=coupled=a.
        let (a_s, b_s, c_s, d_s, e_s) = (a, a, c, a, a);
        let c_prime = c; // only channel in a (x) a
        let nu = 0usize; // vertices[i]=1 -> 0-based 0
        let n_sigma = rule.nsymbol(a_s, d_s, c_prime); // 2
        let n_lambda = rule.nsymbol(c_prime, b_s, e_s); // 1
        let n_rho = rule.nsymbol(d_s, c_s, e_s); // 1
        let n_kappa = rule.nsymbol(d_s, a_s, c_prime); // 2
        assert_eq!((n_sigma, n_lambda, n_rho, n_kappa), (2, 1, 1, 2));
        // Rmat1 = inv ? R(d,c,e)' : R(c,d,e); Rmat2 = inv ? R(d,a,c')' : R(a,d,c')
        let rmat1 = if inverse {
            rule.r_symbol_generic(d_s, c_s, e_s)
        } else {
            rule.r_symbol_generic(c_s, d_s, e_s)
        };
        let rmat2 = if inverse {
            rule.r_symbol_generic(d_s, a_s, c_prime)
        } else {
            rule.r_symbol_generic(a_s, d_s, c_prime)
        };
        let fmat = rule.f_symbol_generic(d_s, a_s, b_s, e_s, c_prime, c_s);
        let mut m = [[0.0f64; 2]; 2];
        for mu in 0..2usize {
            for sigma in 0..n_sigma {
                let lambda = 0usize;
                let mut coeff = 0.0f64;
                for rho in 0..n_rho {
                    for kappa in 0..n_kappa {
                        // Rmat1[nu,rho] (adjoint => conj(base[rho,nu]))
                        let r1 = if inverse {
                            rmat1.get(rho, nu) // 1x1, conj of real = itself
                        } else {
                            rmat1.get(nu, rho)
                        };
                        // conj(Fmat[kappa,lambda,mu,rho]); real => itself
                        let fc = fmat.get(kappa, lambda, mu, rho);
                        // conj(Rmat2[sigma,kappa]); inv => base[kappa,sigma]
                        let r2 = if inverse {
                            rmat2.get(kappa, sigma)
                        } else {
                            rmat2.get(sigma, kappa)
                        };
                        coeff += r1 * fc * r2;
                    }
                }
                m[sigma][mu] = coeff;
            }
        }
        m
    }

    // Extract the impl's 2x2 braid matrix M[sigma][mu] for the same case.
    fn impl_braid_matrix_index1(rule: &RefuteOmRule, inverse: bool) -> [[f64; 2]; 2] {
        let mut m = [[0.0f64; 2]; 2];
        for mu1 in 1..=2usize {
            let tree = refute_rank3_tree(mu1);
            let outs =
                generic_artin_braid_at_with_inverse(rule, &tree, 1, inverse).unwrap();
            for (out, coeff) in outs {
                assert_eq!(out.innerlines(), &[SectorId::new(RefuteOmRule::C)]);
                assert_eq!(out.vertices()[1].get(), 1, "lambda must be 1");
                let sigma = out.vertices()[0].get() - 1;
                m[sigma][mu1 - 1] = coeff;
            }
        }
        m
    }

    #[test]
    fn refute_impl_matches_independent_tk_formula_nontrivial_f() {
        let rule = RefuteOmRule;
        for &inverse in &[false, true] {
            let indep = independent_braid_matrix_index1(&rule, inverse);
            let got = impl_braid_matrix_index1(&rule, inverse);
            for s in 0..2 {
                for m in 0..2 {
                    assert!(
                        (indep[s][m] - got[s][m]).abs() < 1e-12,
                        "inverse={inverse} mismatch at [{s}][{m}]: indep={} impl={}",
                        indep[s][m],
                        got[s][m]
                    );
                }
            }
        }
    }

    #[test]
    fn refute_elementary_braid_is_unitary_nontrivial_f() {
        // With all R,F unitary the elementary braid matrix must be unitary,
        // even though F here is a nontrivial rotation (no hexagon needed).
        let rule = RefuteOmRule;
        let m = impl_braid_matrix_index1(&rule, false);
        // M^T M == I
        for i in 0..2 {
            for j in 0..2 {
                let dot: f64 = (0..2).map(|k| m[k][i] * m[k][j]).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expect).abs() < 1e-12,
                    "braid matrix not unitary at ({i},{j}): {dot}"
                );
            }
        }
    }

    #[test]
    fn refute_roundtrip_matches_analytic_not_identity_when_f_nontrivial() {
        // For a NON-hexagon-consistent F, forward-then-inverse does NOT recover
        // the input for the index>0 path: the composite = M_inv * M_fwd, which
        // analytically is R(theta)^T . R(phi) . R(theta) . R(phi) = R(2*phi)
        // (commuting SO(2)), = I iff phi=0 (F=I). This is exactly why B1's own
        // round-trip test forces F=I; it is NOT a bug. Here we confirm the impl
        // reproduces the analytic composite (independent M_inv . M_fwd) and that
        // it deviates from identity by precisely R(2*phi).
        let rule = RefuteOmRule;
        let m_fwd = independent_braid_matrix_index1(&rule, false);
        let m_inv = independent_braid_matrix_index1(&rule, true);
        // Analytic composite M_inv . M_fwd (apply forward, then inverse).
        let mut comp = [[0.0f64; 2]; 2];
        for s in 0..2 {
            for m in 0..2 {
                comp[s][m] = (0..2).map(|t| m_inv[s][t] * m_fwd[t][m]).sum();
            }
        }
        // Impl round-trip matrix over mu -> final sigma.
        let mut impl_rt = [[0.0f64; 2]; 2];
        for mu1 in 1..=2usize {
            let tree = refute_rank3_tree(mu1);
            let mut col: std::collections::HashMap<usize, f64> =
                std::collections::HashMap::new();
            for (mid, cf) in generic_artin_braid_at_with_inverse(&rule, &tree, 1, false).unwrap()
            {
                for (fin, ci) in
                    generic_artin_braid_at_with_inverse(&rule, &mid, 1, true).unwrap()
                {
                    assert_eq!(fin.innerlines(), &[SectorId::new(RefuteOmRule::C)]);
                    let sigma = fin.vertices()[0].get() - 1;
                    *col.entry(sigma).or_insert(0.0) += cf * ci;
                }
            }
            for (sigma, v) in col {
                impl_rt[sigma][mu1 - 1] = v;
            }
        }
        // (a) impl reproduces the analytic composite.
        for s in 0..2 {
            for m in 0..2 {
                assert!(
                    (impl_rt[s][m] - comp[s][m]).abs() < 1e-12,
                    "impl round-trip != analytic at [{s}][{m}]: {} vs {}",
                    impl_rt[s][m],
                    comp[s][m]
                );
            }
        }
        // (b) composite == R(2*phi), NOT identity.
        let (s2, c2) = (2.0 * RefuteOmRule::F_PHI).sin_cos();
        let r2phi = [[c2, -s2], [s2, c2]];
        for s in 0..2 {
            for m in 0..2 {
                assert!((comp[s][m] - r2phi[s][m]).abs() < 1e-12);
            }
        }
        assert!(
            (comp[0][0] - 1.0).abs() > 1e-3,
            "sanity: nontrivial-F round-trip should deviate from identity"
        );
    }

    // ---- TK numeric oracle: real A4Irrep(3) sector, GenericFusion N=2 ----
    //
    // Values transcribed from TensorKit.jl v0.17 + TensorKitSectors v0.3.9
    // (A4Irrep, FusionStyle == GenericFusion()) — computed by TK's OWN
    // artin_braid on a FusionTreeBlock, independent of this port. We model just
    // the a=b=c=d=e=c'=3 OM sub-block (the braid coefficient formula is local:
    // it reads only F(3,3,3,3,3,3), R(3,3,3), which we transcribe exactly),
    // and reproduce it with generic_artin_braid_at_with_inverse on the tree
    // [3,3,3,3]->0, inner=[3,3], braided at index 1 (= TK i=2). If the F index
    // order [κ,λ,μ,ρ] or any conj/adjoint were transposed, this rich
    // (non-diagonal) 4x4 (μ,ν)->(σ,λ) block would not match.
    #[derive(Clone, Copy, Debug)]
    struct A4SubBlockRule;

    impl A4SubBlockRule {
        const VACUUM: usize = 0;
        const THREE: usize = 3;
        // F(3,3,3,3,3,3), row-major over (κ,λ,μ,ρ), dims (2,2,2,2). Verbatim
        // from TK (oracle4.jl FLAT_F_rowmajor_klmr), -0.0 normalised to 0.0.
        const F333333: [f64; 16] = [
            0.5, 0.0, 0.0, -0.5, 0.0, -0.5, -0.5, 0.0, 0.0, -0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.5,
        ];
        // R(3,3,3) = [[-1,0],[0,1]] (TK).
        const R333: [f64; 4] = [-1.0, 0.0, 0.0, 1.0];
    }

    impl FusionRule for A4SubBlockRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }
        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }
        fn vacuum(&self) -> SectorId {
            SectorId::new(Self::VACUUM)
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            // Truncated to the OM channel; only fusion_channels(3,3) is read by
            // the index-1 braid (for c' enumeration). 3⊗3 ∋ 3 with N=2 in A4.
            if (left.id(), right.id()) == (Self::THREE, Self::THREE) {
                smallvec![SectorId::new(Self::THREE)]
            } else {
                smallvec![SectorId::new(Self::VACUUM)]
            }
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (Self::THREE, Self::THREE, Self::THREE) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }

    impl GenericFusionSymbols for A4SubBlockRule {
        type Scalar = f64;
        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            let all3 = [a, b, c, d, e, f]
                .iter()
                .all(|s| s.id() == Self::THREE);
            assert!(all3, "only F(3,3,3,3,3,3) is modelled");
            GenericFArray::new(Self::F333333.to_vec(), (2, 2, 2, 2))
        }
        fn r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            assert_eq!(
                (a.id(), b.id(), c.id()),
                (Self::THREE, Self::THREE, Self::THREE),
                "only R(3,3,3) is modelled"
            );
            GenericRMatrix::new(Self::R333.to_vec(), 2, 2)
        }
    }

    #[test]
    #[allow(clippy::type_complexity)] // oracle table typed to match the TK dump verbatim
    fn tk_oracle_a4_generic_braid_matches_tensorkit() {
        let rule = A4SubBlockRule;
        let three = SectorId::new(A4SubBlockRule::THREE);
        let vac = SectorId::new(A4SubBlockRule::VACUUM);
        // TK oracle sub-block (identical for inv=false and inv=true here):
        // (mu,nu) -> (sigma,lambda) => coeff.  [oracle4.jl SUBBLOCK]
        let oracle: [((usize, usize), (usize, usize), f64); 8] = [
            ((1, 1), (1, 1), 0.5),
            ((2, 2), (1, 1), 0.5),
            ((2, 1), (2, 1), 0.5),
            ((1, 2), (2, 1), -0.5),
            ((2, 1), (1, 2), -0.5),
            ((1, 2), (1, 2), 0.5),
            ((1, 1), (2, 2), 0.5),
            ((2, 2), (2, 2), 0.5),
        ];
        for &inverse in &[false, true] {
            // Build impl matrix keyed by ((mu,nu),(sigma,lambda)).
            let mut got: std::collections::HashMap<((usize, usize), (usize, usize)), f64> =
                std::collections::HashMap::new();
            for mu in 1..=2usize {
                for nu in 1..=2usize {
                    let tree = FusionTreeKey::new(
                        [three, three, three, three], vac,
                        [false, false, false, false],
                        [three, three],
                        [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based"), MultiplicityIndex::new(nu).expect("test multiplicity label is one-based"), MultiplicityIndex::ONE],
                    );
                    for (out, coeff) in
                        generic_artin_braid_at_with_inverse(&rule, &tree, 1, inverse).unwrap()
                    {
                        assert_eq!(out.innerlines(), &[three, three], "c'=3, e=3 unchanged");
                        assert_eq!(out.vertices()[2].get(), 1);
                        let sigma = out.vertices()[0].get();
                        let lambda = out.vertices()[1].get();
                        got.insert(((mu, nu), (sigma, lambda)), coeff);
                    }
                }
            }
            // Every oracle entry must be reproduced.
            for &(inp, outp, val) in &oracle {
                let g = got.get(&(inp, outp)).copied().unwrap_or(0.0);
                assert!(
                    (g - val).abs() < 1e-10,
                    "inv={inverse} {inp:?}->{outp:?}: impl={g} TK={val}"
                );
            }
            // And the impl must produce NO nonzero outside the oracle set.
            for (&(inp, outp), &g) in &got {
                if g.abs() > 1e-10 {
                    let known = oracle
                        .iter()
                        .any(|&(i, o, v)| i == inp && o == outp && (v - g).abs() < 1e-10);
                    assert!(known, "inv={inverse} spurious nonzero {inp:?}->{outp:?}={g}");
                }
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct MisreportedSimpleA4Rule;

    impl FusionRule for MisreportedSimpleA4Rule {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Simple
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            FusionRule::braiding_style(&A4BendRule)
        }

        fn vacuum(&self) -> SectorId {
            FusionRule::vacuum(&A4BendRule)
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            A4BendRule.dual(sector)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            A4BendRule.fusion_channels(left, right)
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            A4BendRule.nsymbol(left, right, coupled)
        }
    }

    impl GenericFusionSymbols for MisreportedSimpleA4Rule {
        type Scalar = f64;

        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            A4BendRule.f_symbol_generic(a, b, c, d, e, f)
        }

        fn r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            A4BendRule.r_symbol_generic(a, b, c)
        }
    }

    impl GenericRigidSymbols for MisreportedSimpleA4Rule {
        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            A4BendRule.sqrt_dim_scalar(sector)
        }

        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            A4BendRule.inv_sqrt_dim_scalar(sector)
        }

        fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
            A4BendRule.frobenius_schur_phase_scalar(sector)
        }
    }

    // cod [3,3,3]->3, inner=[x] (vertices v1,v2), dom [3]->3.
    fn a4_pair_rank3(inner: usize, v1: usize, v2: usize) -> FusionTreePairKey {
        let t = a4_three();
        let cod = FusionTreeKey::new(
            [t, t, t], t,
            [false, false, false],
            [SectorId::new(inner)],
            [MultiplicityIndex::new(v1).expect("test multiplicity label is one-based"), MultiplicityIndex::new(v2).expect("test multiplicity label is one-based")],
        );
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        FusionTreePairKey::pair(cod, dom)
    }

    #[test]
    fn checked_generic_bend_queries_only_the_rigid_data_it_uses() {
        let pair = a4_dual_pair_rank2(1);
        let rule = CheckedA4Spy::new();
        generic_bendright_tree_pair_checked(&rule, &pair).unwrap();
        assert_eq!(rule.dual_calls.get(), 3);
        // Includes the rank-1 domain tree's N(c, 1, c) admission probe,
        // which the multiplicity-free checked validator also makes.
        assert_eq!(rule.n_calls.get(), 8);
        assert_eq!(rule.f_calls.get(), 1);
        assert_eq!(rule.sqrt_calls.get(), 3);
        assert_eq!(rule.inv_sqrt_calls.get(), 2);
        assert_eq!(rule.fs_calls.get(), 1);
    }

    #[test]
    fn checked_generic_bend_stores_the_b_column_as_the_output_vertex() {
        let rule = CheckedA4Spy {
            non_diagonal_b: true,
            ..CheckedA4Spy::new()
        };
        let out =
            generic_bendright_tree_pair_checked(&rule, &a4_pair_rank2(1)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(
            out.iter()
                .map(|(key, _)| key.domain_tree().vertices()[0].get())
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn checked_generic_bend_preserves_each_provider_failure_source() {
        let pair = a4_dual_pair_rank2(1);

        let rule = CheckedA4Spy {
            fail_n: Some(2),
            ..CheckedA4Spy::new()
        };
        assert_rigid_provider_error(
            generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
            RigidSpyError::N,
        );

        let rule = CheckedA4Spy {
            fail_dual: Some(2),
            ..CheckedA4Spy::new()
        };
        assert_rigid_provider_error(
            generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
            RigidSpyError::Dual,
        );

        let rule = CheckedA4Spy {
            fail_sqrt: Some(1),
            ..CheckedA4Spy::new()
        };
        assert_rigid_provider_error(
            generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
            RigidSpyError::Sqrt,
        );

        let rule = CheckedA4Spy {
            fail_inv_sqrt: Some(1),
            ..CheckedA4Spy::new()
        };
        assert_rigid_provider_error(
            generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
            RigidSpyError::InvSqrt,
        );

        let rule = CheckedA4Spy {
            fail_fs: Some(1),
            ..CheckedA4Spy::new()
        };
        assert_rigid_provider_error(
            generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
            RigidSpyError::Fs,
        );

        let rule = CheckedA4Spy {
            fail_f: Some(1),
            ..CheckedA4Spy::new()
        };
        assert_rigid_provider_error(
            generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
            RigidSpyError::F,
        );
    }

    #[test]
    fn generic_proof_hook_rechecks_reported_style() {
        // What: the proof-consuming Generic hook rejects a provider that
        // implements Generic symbols but reports a multiplicity-free style.
        let empty = BlockStructure::empty(0);
        let wrong_style =
            LocallyValidatedFusionTreeBlockStructure::try_new(&MisreportedSimpleA4Rule, &empty).unwrap();
        assert_eq!(
            wrong_style
                .generic_permute_tree_pair_for_block_index(0, &[], &[])
                .unwrap_err(),
            CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual: FusionStyleKind::Simple,
            }
        );
    }

    fn round_trip_bend(
        rule: &A4BendRule,
        pair: &FusionTreePairKey,
    ) -> std::collections::HashMap<FusionTreePairKey, f64> {
        let mut totals = std::collections::HashMap::new();
        for (mid, c1) in generic_bendright_tree_pair(rule, pair).unwrap() {
            for (out, c2) in generic_bendleft_tree_pair(rule, &mid).unwrap() {
                *totals.entry(out).or_insert(0.0) += c1 * c2;
            }
        }
        totals
    }

    fn assert_identity_map(
        totals: &std::collections::HashMap<FusionTreePairKey, f64>,
        expected_self: &FusionTreePairKey,
        label: &str,
    ) {
        for (key, coeff) in totals {
            let want = if key == expected_self { 1.0 } else { 0.0 };
            assert!(
                (coeff - want).abs() < 1e-12,
                "{label}: coeff {coeff} for is_self={} (want {want})",
                key == expected_self
            );
        }
        assert!(
            (totals.get(expected_self).copied().unwrap_or(0.0) - 1.0).abs() < 1e-12,
            "{label}: self coefficient missing"
        );
    }

    // Gate 1: bendright∘bendleft == identity (the B-matrix is a Hom-space
    // isomorphism), enumerated over all vertex assignments, rank 2 and 3.
    #[test]
    fn b2a_generic_bend_round_trip_identity() {
        let rule = A4BendRule;
        let t = a4_three();
        // Premise the round-trip depends on: N(a,b,c)==N(c,dual(b),a) so the
        // bend is square/invertible on the bent triple (a=b=c=3).
        assert_eq!(rule.nsymbol(t, t, t), rule.nsymbol(t, rule.dual(t), t));
        assert_eq!(rule.nsymbol(t, t, t), 2);

        for mu in 1..=2 {
            let pair = a4_pair_rank2(mu);
            assert_identity_map(&round_trip_bend(&rule, &pair), &pair, &format!("rank2 μ={mu}"));
        }
        // rank 3: inner∈{0,1,2} forces v1=v2=1 (N=1); inner=3 opens both OM
        // vertices v1,v2∈{1,2}. All vertex assignments enumerated.
        for inner in 0..=2 {
            let pair = a4_pair_rank3(inner, 1, 1);
            assert_identity_map(
                &round_trip_bend(&rule, &pair),
                &pair,
                &format!("rank3 inner={inner}"),
            );
        }
        for v1 in 1..=2 {
            for v2 in 1..=2 {
                let pair = a4_pair_rank3(3, v1, v2);
                assert_identity_map(
                    &round_trip_bend(&rule, &pair),
                    &pair,
                    &format!("rank3 inner=3 v=({v1},{v2})"),
                );
            }
        }
    }

    // Gate 2: repartition(N) then repartition back to the original N == identity.
    // via_n=1 exercises one bend each way; via_n=0 exercises two bends each way,
    // covering the rank-1-codomain (left_coupled=vacuum) branch of bendright.
    #[test]
    fn b2a_generic_repartition_round_trip_identity() {
        let rule = A4BendRule;
        for via_n in [1usize, 0usize] {
            for mu in 1..=2 {
                let pair = a4_pair_rank2(mu); // codomain rank 2
                let mut totals = std::collections::HashMap::new();
                for (mid, c1) in generic_repartition_tree_pair(&rule, &pair, via_n).unwrap() {
                    for (out, c2) in generic_repartition_tree_pair(&rule, &mid, 2).unwrap() {
                        *totals.entry(out).or_insert(0.0) += c1 * c2;
                    }
                }
                assert_identity_map(&totals, &pair, &format!("repartition via {via_n} μ={mu}"));
            }
        }
    }

    // Oracle: b_symbol_generic / a_symbol_generic match TK's Bsymbol / Asymbol.
    #[test]
    fn b2a_a4_b_and_a_symbol_match_tensorkit() {
        let rule = A4BendRule;
        let t = a4_three();
        // TK: Bsymbol(3,3,3) == I₂, Asymbol(3,3,3) == I₂ (TKS v0.3.6).
        let b = rule.b_symbol_generic(t, t, t);
        assert_eq!(b.shape(), (2, 2));
        let a = rule.a_symbol_generic(t, t, t);
        assert_eq!(a.shape(), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((b.get(i, j) - want).abs() < 1e-10, "B[{i},{j}]={}", b.get(i, j));
                assert!((a.get(i, j) - want).abs() < 1e-10, "A[{i},{j}]={}", a.get(i, j));
            }
        }
    }

    // Focused unit test for the domain-empty keep-last overwrite: mirrors TK's
    // block assignment `U[row, col] = coeff` (duality_manipulations.jl:110),
    // where every ν collapses onto the same output key (no vertex to store) and
    // the LAST non-zero ν wins. A4's Bsymbol is diagonal so it never puts two
    // non-zeros in one row — this needs a synthetic non-diagonal B. b_symbol is
    // overridden directly (default-method override), so no F is consulted.
    #[derive(Clone, Copy, Debug)]
    struct OverwriteProbeRule;

    impl FusionRule for OverwriteProbeRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }
        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }
        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 1) => smallvec![SectorId::new(0)],
                _ => smallvec![SectorId::new(0)],
            }
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (1, 1, 0) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }

    impl GenericFusionSymbols for OverwriteProbeRule {
        type Scalar = f64;
        fn f_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
            _d: SectorId,
            _e: SectorId,
            _f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            unreachable!("b_symbol_generic is overridden; F is never read")
        }
        fn r_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            GenericRMatrix::new(vec![1.0], 1, 1)
        }
    }

    impl GenericRigidSymbols for OverwriteProbeRule {
        fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
        fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
        fn b_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            // Row 0 = [0.3, 0.7]: two non-zeros, distinct, so keep-last is
            // distinguishable from keep-first (0.3) and from sum (1.0).
            GenericRMatrix::new(vec![0.3, 0.7, 0.0, 0.0], 2, 2)
        }
    }

    #[test]
    fn b2a_generic_bendright_domain_empty_keeps_last_nu() {
        let rule = OverwriteProbeRule;
        let a = SectorId::new(1);
        let c = rule.vacuum();
        // cod [1,1]->vac (vertex 1); dom []->vac (EMPTY domain ⇒ ν has nowhere to go).
        let cod = FusionTreeKey::new([a, a], c, [false, false], [], [MultiplicityIndex::ONE]);
        let dom = FusionTreeKey::new([], c, [], [], []);
        let pair = FusionTreePairKey::pair(cod, dom);
        pair.validate_for_rule(&rule).unwrap();
        let out = generic_bendright_tree_pair(&rule, &pair).unwrap();
        assert_eq!(out.len(), 1, "empty domain collapses ν to one key");
        // coeff0 = √dim(vac)·(1/√dim(1)) = 1; keep-last ⇒ B[0,1] = 0.7.
        assert!((out[0].1 - 0.7).abs() < 1e-12, "keep-last ν: got {}", out[0].1);
    }

    // Oracle: tree-level bendright tables vs TensorKit's own bendright.
    #[test]
    fn b2a_a4_bendright_tree_table_matches_tensorkit() {
        let rule = A4BendRule;
        let sq3 = 3.0_f64.sqrt();

        // --- rank 2: cod [3,3]->3 (μ), dom [3]->3.  TK (probe5.jl):
        //   μ=1 -> dom vertex 1, coeff 1 ;  μ=2 -> dom vertex 2, coeff 1.
        for mu in 1..=2 {
            let out = generic_bendright_tree_pair(&rule, &a4_pair_rank2(mu)).unwrap();
            let nonzero: Vec<_> = out.iter().filter(|(_, c)| c.abs() > 1e-10).collect();
            assert_eq!(nonzero.len(), 1, "rank2 μ={mu} expects one nonzero");
            let (key, coeff) = nonzero[0];
            assert_eq!(key.codomain_tree().uncoupled(), [a4_three()], "rank2 cod");
            assert!(key.codomain_tree().vertices().is_empty(), "rank2 cod rank-1 no vtx");
            assert_eq!(key.domain_tree().vertices()[0].get(), mu, "rank2 ν == μ (B diagonal)");
            assert!((coeff - 1.0).abs() < 1e-10, "rank2 μ={mu} coeff {coeff}");
        }

        // --- rank 3: cod [3,3,3]->3 inner=[x] (v1,v2), dom [3]->3.
        // TK (probe6.jl): coeff = √dim(3)/√dim(inner) = √3 (inner∈{0,1,2}) or 1
        // (inner=3); output cod vertex = v1, dom vertex = ν = v2 (B=I diagonal).
        // Table rows: (inner, v1, v2) -> (cod_vtx, dom_vtx, coeff).
        let table: [(usize, usize, usize, usize, usize, f64); 7] = [
            (0, 1, 1, 1, 1, sq3),
            (1, 1, 1, 1, 1, sq3),
            (2, 1, 1, 1, 1, sq3),
            (3, 1, 1, 1, 1, 1.0),
            (3, 2, 1, 2, 1, 1.0),
            (3, 1, 2, 1, 2, 1.0),
            (3, 2, 2, 2, 2, 1.0),
        ];
        for (inner, v1, v2, cod_vtx, dom_vtx, coeff) in table {
            let out = generic_bendright_tree_pair(&rule, &a4_pair_rank3(inner, v1, v2)).unwrap();
            let nonzero: Vec<_> = out.iter().filter(|(_, c)| c.abs() > 1e-10).collect();
            assert_eq!(nonzero.len(), 1, "rank3 ({inner},{v1},{v2}) one nonzero");
            let (key, got) = nonzero[0];
            let left_coupled = key.codomain_tree().coupled().id();
            assert_eq!(left_coupled, inner, "rank3 left_coupled == innerline");
            assert_eq!(key.codomain_tree().vertices()[0].get(), cod_vtx, "rank3 cod vtx");
            assert_eq!(key.domain_tree().vertices()[0].get(), dom_vtx, "rank3 dom vtx");
            assert!((got - coeff).abs() < 1e-10, "rank3 ({inner},{v1},{v2}) coeff {got} want {coeff}");
        }
    }

    fn tp_expected_a() -> [[f64; 2]; 2] {
        let factor = 2.0 * 2.0 * 0.5;
        let kappa_a = 1.0f64; // FS phase, real
        let mut a = [[0.0; 2]; 2];
        for k in 0..2 {
            for l in 0..2 {
                // conj(κ_a · F[0,0,κ,λ]) · factor; all real here.
                a[k][l] = factor * (kappa_a * TP_FA[k * 2 + l]);
            }
        }
        a
    }

    #[test]
    fn refute_b2a_b_symbol_is_not_transposed() {
        let rule = TransposeProbeRule;
        let s = SectorId::new(1);
        let b = rule.b_symbol_generic(s, s, s);
        assert_eq!(b.shape(), (2, 2));
        let want = tp_expected_b();
        // Sanity: the oracle itself must be non-symmetric, else no discrimination.
        assert!((want[0][1] - want[1][0]).abs() > 0.1, "oracle B must be non-symmetric");
        for (mu, row) in want.iter().enumerate() {
            for (nu, &want_value) in row.iter().enumerate() {
                assert!(
                    (b.get(mu, nu) - want_value).abs() < 1e-12,
                    "B[{mu},{nu}]={} want {} (μ↔ν transpose?)",
                    b.get(mu, nu),
                    want_value
                );
            }
        }
    }

    #[test]
    fn refute_b2a_a_symbol_is_not_transposed() {
        // a_symbol_generic is UNUSED by any other B2a test (fold is B2b), so this
        // is the ONLY thing exercising its κ↔λ index order today.
        let rule = TransposeProbeRule;
        let s = SectorId::new(1);
        let a = rule.a_symbol_generic(s, s, s);
        assert_eq!(a.shape(), (2, 2));
        let want = tp_expected_a();
        assert!((want[0][1] - want[1][0]).abs() > 0.1, "oracle A must be non-symmetric");
        for (k, row) in want.iter().enumerate() {
            for (l, &want_value) in row.iter().enumerate() {
                assert!(
                    (a.get(k, l) - want_value).abs() < 1e-12,
                    "A[{k},{l}]={} want {} (κ↔λ transpose?)",
                    a.get(k, l),
                    want_value
                );
            }
        }
    }

    struct LegacyA4FPolicyProbe {
        n_calls: std::cell::Cell<usize>,
        f_calls: std::cell::Cell<usize>,
    }

    impl FusionRule for LegacyA4FPolicyProbe {
        fn rule_identity(&self) -> RuleIdentity {
            FusionRule::rule_identity(&A4FoldRule)
        }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionRule::fusion_style(&A4FoldRule)
        }
        fn braiding_style(&self) -> BraidingStyleKind {
            FusionRule::braiding_style(&A4FoldRule)
        }
        fn vacuum(&self) -> SectorId {
            FusionRule::vacuum(&A4FoldRule)
        }
        fn dual(&self, sector: SectorId) -> SectorId {
            FusionRule::dual(&A4FoldRule, sector)
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            FusionRule::fusion_channels(&A4FoldRule, left, right)
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            self.n_calls.set(self.n_calls.get() + 1);
            FusionRule::nsymbol(&A4FoldRule, left, right, coupled)
        }
    }

    impl GenericFusionSymbols for LegacyA4FPolicyProbe {
        type Scalar = f64;

        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            self.f_calls.set(self.f_calls.get() + 1);
            GenericFusionSymbols::f_symbol_generic(&A4FoldRule, a, b, c, d, e, f)
        }

        fn r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            GenericFusionSymbols::r_symbol_generic(&A4FoldRule, a, b, c)
        }
    }

    // Look up the coeff vector for the multi_Fmove output tree with the given
    // coupled sector and (single) vertex label.
    fn find_coeff(
        out: &[(FusionTreeKey, Vec<f64>)],
        coupled: usize,
        vtx: usize,
    ) -> Vec<f64> {
        out.iter()
            .find(|(tr, _)| {
                tr.coupled().id() == coupled && tr.vertices()[0].get() == vtx
            })
            .unwrap_or_else(|| panic!("no output tree coupled={coupled} vtx={vtx}"))
            .1
            .clone()
    }

    fn assert_vec(got: &[f64], want: &[f64], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length {} != {}", got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-10, "{label}[{i}]: {g} != {w}");
        }
    }

    // Gate 4a (A4 oracle): multi_Fmove of every rank-3 A4 (3,3,3)->3 tree matches
    // TensorKit.multi_Fmove exactly, INCLUDING the coefficient VECTORS. The
    // inner=3 rows exercise the non-trivial F(3,3,3,3,3,3) block, so the vector
    // machinery (F-slice selection, μ/ν/κ vertex indexing, λ free axis) is fully
    // discriminated here — unlike the B2a bend oracle whose A/B are I₂.
    #[test]
    fn b2b_a4_multi_fmove_matches_tensorkit() {
        let rule = A4FoldRule;
        let s = 1.0 / 3.0_f64.sqrt();
        let m = -1.0 / (2.0 * 3.0_f64.sqrt());
        let o = 1.0 / 3.0;
        // (inner,v1,v2) -> [(coupled, vtx, coeff)]. TK.multi_Fmove gold values.
        type Row = (usize, usize, usize, Vec<(usize, usize, Vec<f64>)>);
        let table: Vec<Row> = vec![
            (0, 1, 1, vec![
                (0, 1, vec![o]), (1, 1, vec![o]), (2, 1, vec![o]),
                (3, 1, vec![s, 0.0]), (3, 2, vec![0.0, s])]),
            (1, 1, 1, vec![
                (0, 1, vec![o]), (1, 1, vec![o]), (2, 1, vec![o]),
                (3, 1, vec![m, -0.5]), (3, 2, vec![0.5, m])]),
            (2, 1, 1, vec![
                (0, 1, vec![o]), (1, 1, vec![o]), (2, 1, vec![o]),
                (3, 1, vec![m, 0.5]), (3, 2, vec![-0.5, m])]),
            (3, 1, 1, vec![
                (0, 1, vec![s]), (1, 1, vec![m]), (2, 1, vec![m]),
                (3, 1, vec![0.5, 0.0]), (3, 2, vec![0.0, -0.5])]),
            (3, 1, 2, vec![
                (0, 1, vec![0.0]), (1, 1, vec![0.5]), (2, 1, vec![-0.5]),
                (3, 1, vec![0.0, -0.5]), (3, 2, vec![-0.5, 0.0])]),
            (3, 2, 1, vec![
                (0, 1, vec![0.0]), (1, 1, vec![-0.5]), (2, 1, vec![0.5]),
                (3, 1, vec![0.0, -0.5]), (3, 2, vec![-0.5, 0.0])]),
            (3, 2, 2, vec![
                (0, 1, vec![s]), (1, 1, vec![m]), (2, 1, vec![m]),
                (3, 1, vec![-0.5, 0.0]), (3, 2, vec![0.0, 0.5])]),
        ];
        for (inner, v1, v2, outputs) in table {
            let out = generic_multi_fmove_tree(&rule, &a4f_rank3(inner, v1, v2)).unwrap();
            assert_eq!(out.len(), 5, "in({inner},{v1},{v2}): 5 tails");
            // What: every tail-coupled candidate reuses the same frozen
            // external runtime-rank identity.
            assert!(out.windows(2).all(|terms| Arc::ptr_eq(
                &terms[0].0.uncoupled,
                &terms[1].0.uncoupled
            )));
            assert!(out.windows(2).all(|terms| Arc::ptr_eq(
                &terms[0].0.is_dual,
                &terms[1].0.is_dual
            )));
            for (coupled, vtx, want) in outputs {
                let got = find_coeff(&out, coupled, vtx);
                assert_vec(&got, &want, &format!("in({inner},{v1},{v2}) out(c={coupled},v={vtx})"));
            }
        }
    }

    #[test]
    fn legacy_generic_associator_keeps_raw_f_shape_policy() {
        let long = a4f_rank3(3, 2, 1);
        let tail = generic_multi_fmove_tree(&A4FoldRule, &long)
            .unwrap()
            .into_iter()
            .find(|(tree, _)| tree.coupled().id() == 3 && tree.vertices()[0].get() == 1)
            .unwrap()
            .0;
        let probe = LegacyA4FPolicyProbe {
            n_calls: std::cell::Cell::new(0),
            f_calls: std::cell::Cell::new(0),
        };
        generic_multi_associator_result(&InfallibleGenericFR(&probe), &long, &tail)
            .unwrap()
            .unwrap();
        assert_eq!(probe.f_calls.get(), 1);
        assert_eq!(
            probe.n_calls.get(),
            1,
            "legacy access must not add four F-shape Nsymbol queries"
        );
    }

    // Build the full foldright coefficient map keyed by output tree pair. The
    // output collapses multiple (codomain', domain') paths per pair (the A-matrix
    // contraction) — the accumulator already summed them.
    fn foldright_map(
        rule: &A4FoldRule,
        pair: &FusionTreePairKey,
    ) -> std::collections::HashMap<FusionTreePairKey, f64> {
        let mut map = std::collections::HashMap::new();
        for (out, coeff) in generic_foldright_tree_pair(rule, pair).unwrap() {
            *map.entry(out).or_insert(0.0) += coeff;
        }
        map
    }

    // Gate 4b (A4 oracle): tree-level foldright U-matrix vs TensorKit.foldright.
    // rank-2 codomain: dst domain first leg dualized, U == I₂ (A4 Asymbol=I₂).
    #[test]
    fn b2b_a4_foldright_rank2_matches_tensorkit() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        for mu in 1..=2 {
            // src: cod [3,3]->3 (vtx μ), dom [3]->3.
            let cod = FusionTreeKey::new([t, t], t, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
            let dom = FusionTreeKey::new([t], t, [false], [], []);
            let map = foldright_map(&rule, &FusionTreePairKey::pair(cod, dom));
            // TK dst: cod [3]->3, dom [3,3]->3 (isdual=(true,false)) vtx μ, U[μ,μ]=1.
            let exp_cod =
                FusionTreeKey::new([t], t, [false], [], []);
            let exp_dom =
                FusionTreeKey::new([t, t], t, [true, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
            let exp = FusionTreePairKey::pair(exp_cod, exp_dom);
            for (key, coeff) in &map {
                let want = if key == &exp { 1.0 } else { 0.0 };
                assert!((coeff - want).abs() < 1e-10, "rank2 μ={mu}: coeff {coeff} want {want}");
            }
            assert!((map.get(&exp).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10, "rank2 μ={mu} self");
        }
    }

    // Gate 4b (A4 oracle): rank-3 foldright — the 7×7 U-matrix that fully
    // exercises F(3,3,3,3,3,3) combined with the √dim coeff factors (the ±√3/2 =
    // ±0.8660 entries). This is the strongest fold discriminator available.
    #[test]
    fn b2b_a4_foldright_rank3_matches_tensorkit() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        // src columns (cod [3,3,3]->3 inner=x vtx=(v1,v2), dom [3]->3).
        let cols: [(usize, usize, usize); 7] = [
            (0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 2), (3, 2, 2),
        ];
        // dst rows: (cod_coupled, cod_vtx, dom_coupled, dom_vtx). dom isdual=(true,false).
        let rows: [(usize, usize, usize, usize); 7] = [
            (0, 1, 0, 1), (1, 1, 1, 1), (2, 1, 2, 1),
            (3, 1, 3, 1), (3, 2, 3, 1), (3, 1, 3, 2), (3, 2, 3, 2),
        ];
        let sq = 1.0 / 3.0_f64.sqrt(); // 0.57735
        let hs = 3.0_f64.sqrt() / 2.0; // 0.86603
        // TK.foldright U (row,col) nonzeros; zeros elsewhere. 1-based -> 0-based.
        let u: [[f64; 7]; 7] = [
            [sq, sq, sq, 1.0, 0.0, 0.0, 1.0],
            [sq, sq, sq, -0.5, -hs, hs, -0.5],
            [sq, sq, sq, -0.5, hs, -hs, -0.5],
            [sq, -0.5 * sq, -0.5 * sq, 0.5, 0.0, 0.0, -0.5],
            [0.0, 0.5, -0.5, 0.0, -0.5, -0.5, 0.0],
            [0.0, -0.5, 0.5, 0.0, -0.5, -0.5, 0.0],
            [sq, -0.5 * sq, -0.5 * sq, -0.5, 0.0, 0.0, 0.5],
        ];
        for (ci, &(inner, v1, v2)) in cols.iter().enumerate() {
            let cod = a4f_rank3(inner, v1, v2);
            let pair = FusionTreePairKey::pair(cod, dom.clone());
            let map = foldright_map(&rule, &pair);
            for (ri, &(cc, cv, dc, dv)) in rows.iter().enumerate() {
                let ex_cod = FusionTreeKey::new(
                    [t, t], SectorId::new(cc), [false, false], [], [MultiplicityIndex::new(cv).expect("test multiplicity label is one-based")],
                );
                let ex_dom = FusionTreeKey::new(
                    [t, t], SectorId::new(dc), [true, false], [], [MultiplicityIndex::new(dv).expect("test multiplicity label is one-based")],
                );
                let key = FusionTreePairKey::pair(ex_cod, ex_dom);
                let got = map.get(&key).copied().unwrap_or(0.0);
                assert!(
                    (got - u[ri][ci]).abs() < 1e-10,
                    "U[row{ri},col{ci}] (in={inner},{v1},{v2}) got {got} want {}",
                    u[ri][ci]
                );
            }
        }
    }

    // Gate 1: fold round-trip identity. foldright then foldleft returns the
    // original pair with coefficient 1 (A-unitarity: A A† = I on the bent
    // triple). Enumerated over all rank-3 A4 vertex assignments.
    #[test]
    fn b2b_a4_fold_round_trip_identity() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        for (inner, v1, v2) in
            [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 2), (3, 2, 2)]
        {
            let pair = FusionTreePairKey::pair(a4f_rank3(inner, v1, v2), dom.clone());
            let mut totals = std::collections::HashMap::new();
            for (mid, c1) in generic_foldright_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_foldleft_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(0.0) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { 1.0 } else { 0.0 };
                assert!(
                    (coeff - want).abs() < 1e-10,
                    "fold rt in({inner},{v1},{v2}): coeff {coeff} want {want}"
                );
            }
            assert!(
                (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                "fold rt in({inner},{v1},{v2}): self missing"
            );
        }
    }

    // Gate 2: cycle round-trip. cycleclockwise then cycleanticlockwise == id.
    #[test]
    fn b2b_a4_cycle_round_trip_identity() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        for (inner, v1, v2) in [(0, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 2), (3, 2, 2)] {
            let pair = FusionTreePairKey::pair(a4f_rank3(inner, v1, v2), dom.clone());
            let mut totals = std::collections::HashMap::new();
            for (mid, c1) in generic_cycle_clockwise_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_cycle_anticlockwise_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(0.0) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { 1.0 } else { 0.0 };
                assert!(
                    (coeff - want).abs() < 1e-10,
                    "cycle rt in({inner},{v1},{v2}): coeff {coeff} want {want}"
                );
            }
            assert!(
                (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                "cycle rt in({inner},{v1},{v2}): self missing"
            );
        }
    }

    // Residual (c): domain-rank ≥ 2. All prior generic bend/fold tests use a
    // rank-1 domain; these exercise multi_Fmove_inv on a rank-2 domain (its
    // candidates are rank-3, so the associator F-chain runs on the domain side)
    // and the rank-2-domain bend surgery. Round-trip identities, all A4 vertex
    // assignments enumerated.
    #[test]
    fn b2b_a4_fold_round_trip_domain_rank2() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        for cod_mu in 1..=2 {
            for (dom_inner, dv) in [(0, 1), (1, 1), (2, 1), (3, 1), (3, 2)] {
                // cod [3,3]->3 (vtx cod_mu); dom [3,3]->3 inner=dom_inner (vtx dv).
                let cod = FusionTreeKey::new(
                    [t, t], t, [false, false], [], [MultiplicityIndex::new(cod_mu).expect("test multiplicity label is one-based")],
                );
                let dom = FusionTreeKey::new(
                    [t, t], t, [false, false], [], [MultiplicityIndex::new(dv).expect("test multiplicity label is one-based")],
                );
                let _ = dom_inner; // rank-2 dom has no innerline; kept for label clarity
                let pair = FusionTreePairKey::pair(cod, dom);
                let mut totals = std::collections::HashMap::new();
                for (mid, c1) in generic_foldright_tree_pair(&rule, &pair).unwrap() {
                    for (out, c2) in generic_foldleft_tree_pair(&rule, &mid).unwrap() {
                        *totals.entry(out).or_insert(0.0) += c1 * c2;
                    }
                }
                for (key, coeff) in &totals {
                    let want = if key == &pair { 1.0 } else { 0.0 };
                    assert!(
                        (coeff - want).abs() < 1e-10,
                        "fold rt dom-rank2 cod_mu={cod_mu} dv={dv}: {coeff} want {want}"
                    );
                }
                assert!(
                    (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                    "fold rt dom-rank2 cod_mu={cod_mu} dv={dv}: self missing"
                );
            }
        }
    }

    #[test]
    fn b2b_a4_bend_round_trip_domain_rank2() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        for cod_mu in 1..=2 {
            for dv in 1..=2 {
                let cod = FusionTreeKey::new(
                    [t, t], t, [false, false], [], [MultiplicityIndex::new(cod_mu).expect("test multiplicity label is one-based")],
                );
                let dom = FusionTreeKey::new(
                    [t, t], t, [false, false], [], [MultiplicityIndex::new(dv).expect("test multiplicity label is one-based")],
                );
                let pair = FusionTreePairKey::pair(cod, dom);
                let mut totals = std::collections::HashMap::new();
                for (mid, c1) in generic_bendright_tree_pair(&rule, &pair).unwrap() {
                    for (out, c2) in generic_bendleft_tree_pair(&rule, &mid).unwrap() {
                        *totals.entry(out).or_insert(0.0) += c1 * c2;
                    }
                }
                for (key, coeff) in &totals {
                    let want = if key == &pair { 1.0 } else { 0.0 };
                    assert!(
                        (coeff - want).abs() < 1e-10,
                        "bend rt dom-rank2 cod_mu={cod_mu} dv={dv}: {coeff} want {want}"
                    );
                }
                assert!(
                    (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                    "bend rt dom-rank2 cod_mu={cod_mu} dv={dv}: self missing"
                );
            }
        }
    }

    // Direct: foldright distributes ROW μ of the COMPLEX A-matrix (=U) to the
    // domain vertices ν. coeff(out ν) = coeff0·A[μ,ν] = U[μ,ν]. A missing conj
    // or a μ↔ν swap would produce conj(U)/Uᵀ — distinct complex numbers.
    #[test]
    fn refute_b2b_complex_foldright_reads_a_row_unconjugated() {
        let rule = ComplexUnitaryRule;
        let s = SectorId::new(1);
        let u = cx_u();
        // Sanity: U genuinely complex and non-Hermitian.
        assert!(u[1].im.abs() > 0.1, "U must be complex");
        assert!((u[1] - u[2].conj()).norm() > 0.1, "U must be non-Hermitian");
        for mu in 1..=2usize {
            let cod = FusionTreeKey::new([s, s], s, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
            let dom = FusionTreeKey::new([s], s, [false], [], []);
            let pair = FusionTreePairKey::pair(cod, dom);
            let out = generic_foldright_tree_pair(&rule, &pair).unwrap();
            let mut got = [cx(0.0, 0.0); 2];
            for (key, coeff) in &out {
                let nu = key.domain_tree().vertices()[0].get();
                got[nu - 1] = *coeff;
            }
            for nu in 0..2 {
                let want = u[(mu - 1) * 2 + nu]; // ROW μ of U
                assert!(
                    (got[nu] - want).norm() < 1e-10,
                    "μ={mu} ν={nu}: {} want ROW-μ {} (conj/transpose?)",
                    got[nu],
                    want
                );
            }
            // Distinguishable from the conjugated reading.
            let want_conj = u[(mu - 1) * 2].conj();
            assert!(
                (got[0] - want_conj).norm() > 1e-9 || u[(mu - 1) * 2].im.abs() < 1e-12,
                "conj reading coincides — test cannot discriminate"
            );
        }
    }

    // Round-trip with a COMPLEX unitary A: foldright∘foldleft == id requires
    // U U† = I, so the conj in the return fold must be exactly right.
    #[test]
    fn b2b_complex_fold_round_trip_identity() {
        let rule = ComplexUnitaryRule;
        let s = SectorId::new(1);
        for mu in 1..=2usize {
            let cod = FusionTreeKey::new([s, s], s, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
            let dom = FusionTreeKey::new([s], s, [false], [], []);
            let pair = FusionTreePairKey::pair(cod, dom);
            let mut totals: std::collections::HashMap<FusionTreePairKey, Complex64> =
                std::collections::HashMap::new();
            for (mid, c1) in generic_foldright_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_foldleft_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(cx(0.0, 0.0)) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { cx(1.0, 0.0) } else { cx(0.0, 0.0) };
                assert!(
                    (coeff - want).norm() < 1e-10,
                    "cx fold rt μ={mu}: {coeff} want {want}"
                );
            }
            assert!(
                (totals.get(&pair).copied().unwrap_or(cx(0.0, 0.0)) - cx(1.0, 0.0)).norm() < 1e-10,
                "cx fold rt μ={mu}: self missing"
            );
        }
    }

    // Bend round-trip with a COMPLEX unitary B: bendright∘bendleft == id.
    #[test]
    fn b2b_complex_bend_round_trip_identity() {
        let rule = ComplexUnitaryRule;
        let s = SectorId::new(1);
        for mu in 1..=2usize {
            let cod = FusionTreeKey::new([s, s], s, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
            let dom = FusionTreeKey::new([s], s, [false], [], []);
            let pair = FusionTreePairKey::pair(cod, dom);
            let mut totals: std::collections::HashMap<FusionTreePairKey, Complex64> =
                std::collections::HashMap::new();
            for (mid, c1) in generic_bendright_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_bendleft_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(cx(0.0, 0.0)) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { cx(1.0, 0.0) } else { cx(0.0, 0.0) };
                assert!(
                    (coeff - want).norm() < 1e-10,
                    "cx bend rt μ={mu}: {coeff} want {want}"
                );
            }
        }
    }

    fn su3_bfwd() -> [[f64; 2]; 2] {
        let e = -1.0 / (2.0 * 2.0_f64.sqrt());
        let g = (7.0_f64 / 8.0).sqrt();
        [[e, -g], [g, e]]
    }

    // Real-categorical bend oracle: bendright distributes coeff₀·ROW μ of the
    // NON-DIAGONAL SU(3) B to the domain vertices ν. A μ↔ν swap would emit
    // COLUMN μ; B is non-symmetric (B[0,1]≠B[1,0]) so the two are distinct.
    #[test]
    fn b2b_su3_bendright_uses_b_row_not_column() {
        let rule = Su3BendRule;
        let s42 = SectorId::new(1);
        let s31 = SectorId::new(2);
        let b = su3_bfwd();
        let coeff0 = 15.0_f64.sqrt() / 27.0_f64.sqrt(); // √dim(31)/√dim(42)
        for mu in 1..=2usize {
            // cod [42,31]->31 (vtx μ), dom [31]->31.
            let cod = FusionTreeKey::new(
                [s42, s31], s31, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
            );
            let dom =
                FusionTreeKey::new([s31], s31, [false], [], []);
            let out = generic_bendright_tree_pair(&rule, &FusionTreePairKey::pair(cod, dom))
                .unwrap();
            let mut got = [0.0f64; 2];
            for (key, coeff) in &out {
                let nu = key.domain_tree().vertices().last().unwrap().get();
                got[nu - 1] = *coeff;
            }
            for nu in 0..2 {
                let want = coeff0 * b[mu - 1][nu]; // ROW μ
                assert!(
                    (got[nu] - want).abs() < 1e-10,
                    "μ={mu} ν={nu}: {} want coeff0·ROW-μ {} (transpose ⇒ column)",
                    got[nu],
                    want
                );
            }
            // Distinguishable from the transposed (column) reading.
            let col = coeff0 * b[if mu == 1 { 1 } else { 0 }][mu - 1];
            assert!((got[0] - col).abs() > 1e-9, "μ={mu}: row/column coincide");
        }
    }

    // Round-trip with a real non-diagonal SU(3) B: bendright∘bendleft == id
    // (B_fwd · B_ret = I₂), exercising the non-trivial off-diagonal mixing.
    #[test]
    fn b2b_su3_bend_round_trip_identity() {
        let rule = Su3BendRule;
        let s42 = SectorId::new(1);
        let s31 = SectorId::new(2);
        for mu in 1..=2usize {
            let cod = FusionTreeKey::new(
                [s42, s31], s31, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
            );
            let dom =
                FusionTreeKey::new([s31], s31, [false], [], []);
            let pair = FusionTreePairKey::pair(cod, dom);
            let mut totals = std::collections::HashMap::new();
            for (mid, c1) in generic_bendright_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_bendleft_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(0.0) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { 1.0 } else { 0.0 };
                assert!(
                    (coeff - want).abs() < 1e-10,
                    "su3 bend rt μ={mu}: {coeff} want {want}"
                );
            }
            assert!(
                (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                "su3 bend rt μ={mu}: self missing"
            );
        }
    }

    fn assert_identity_term_map(
        got: &HashMap<FusionTreePairKey, f64>,
        self_pair: &FusionTreePairKey,
        label: &str,
    ) {
        for (key, coeff) in got {
            let want = if key == self_pair { 1.0 } else { 0.0 };
            assert!((coeff - want).abs() < 1e-10, "{label}: coeff {coeff} != {want}");
        }
        assert!(
            (got.get(self_pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
            "{label}: self coefficient missing"
        );
    }

    // A4 rank-1/rank-1 pair: cod [3]->3, dom [3]->3 (coupled sector 3).
    fn a4_pair_rank1_1() -> FusionTreePairKey {
        let t = SectorId::new(3);
        let cod = FusionTreeKey::new([t], t, [false], [], []);
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        FusionTreePairKey::pair(cod, dom)
    }

    // A4 rank-2/rank-1 pair: cod [3,3]->3 (vtx μ), dom [3]->3 — an
    // outer-multiplicity tree pair with N(3,3,3)=2.
    fn a4_pair_rank2_1(mu: usize) -> FusionTreePairKey {
        let t = SectorId::new(3);
        let cod = FusionTreeKey::new([t, t], t, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        FusionTreePairKey::pair(cod, dom)
    }

    // Gate B2c-1: transpose (planar cyclic permutation) round-trips to the
    // identity. `generic_transpose_tree_pair` chains repartition + the fold/bend
    // A-move (`generic_cycle_*`); applying the swap [1],[0] and then its inverse
    // [1],[0] again must return the original pair with coefficient 1. Run on
    // A4FoldRule, whose A-move is a genuinely non-diagonal outer-multiplicity
    // move, so any coefficient error in the composition breaks the identity.
    #[test]
    fn b2c_generic_transpose_round_trips_to_identity() {
        let rule = A4FoldRule;
        let pair = a4_pair_rank1_1();
        let forward = generic_transpose_tree_pair(&rule, &pair, &[1], &[0]).unwrap();
        let mut totals = HashMap::new();
        for (mid, c1) in forward {
            for (out, c2) in generic_transpose_tree_pair(&rule, &mid, &[1], &[0]).unwrap() {
                *totals.entry(out).or_insert(0.0) += c1 * c2;
            }
        }
        assert_identity_term_map(&totals, &pair, "A4 transpose round-trip");
    }

    // Gate B2c-2: braid with the IDENTITY permutation is the identity map on an
    // outer-multiplicity tree pair. The braid decomposes to zero swaps, so the
    // composer runs repartition-to-all-codomain and back (a verified bend
    // round-trip) around a no-op braid, plus the tree-pair reconstruction
    // closure. Run on A4BendRule (rigid, OM); no braid R-symbol is invoked.
    #[test]
    fn b2c_generic_braid_identity_permutation_is_identity() {
        let rule = A4BendRule;
        for mu in 1..=2 {
            let pair = a4_pair_rank2_1(mu);
            // codomain axes [0,1], domain axis [2]; identity level order.
            let got = map_terms(
                generic_braid_tree_pair(&rule, &pair, &[0, 1], &[2], &[0, 1], &[2]).unwrap(),
            );
            assert_identity_term_map(&got, &pair, &format!("A4 braid-id μ={mu}"));
        }
    }

    // Gate B2c-3: `generic_permute_tree_pair` == `generic_braid_tree_pair` under
    // the identity level order (the definitional relation the mult-free path
    // relies on), and the symmetric-braiding guard is honored. Uses the identity
    // permutation because a non-trivial multi-leg *braid* needs a fully-modeled
    // braiding generic rule (the SU(3) provider, Stage B3); the composition
    // itself is a line-for-line mirror of the fully-tested
    // `multiplicity_free_braid_tree_pair`, and its braid step
    // (`generic_braid_tree`) is independently adversarially verified in B1.
    #[test]
    fn b2c_generic_permute_agrees_with_default_level_braid() {
        let rule = A4BendRule; // Bosonic ⇒ symmetric braiding.
        for mu in 1..=2 {
            let pair = a4_pair_rank2_1(mu);
            let permuted = map_terms(
                generic_permute_tree_pair(&rule, &pair, &[0, 1], &[2]).unwrap(),
            );
            // default levels: codomain [0,1], domain [2].
            let braided = map_terms(
                generic_braid_tree_pair(&rule, &pair, &[0, 1], &[2], &[0, 1], &[2]).unwrap(),
            );
            assert_term_maps_eq(&permuted, &braided, &format!("A4 permute==braid μ={mu}"));
            // And the identity permutation is a genuine no-op.
            assert_identity_term_map(&permuted, &pair, &format!("A4 permute-id μ={mu}"));
        }
    }

    #[test]
    fn concurrent_equal_hom_spaces_share_semantic_identity() {
        // What: asserts ptr_eq across concurrently-built identical hom spaces
        // in the shared intern table; a concurrent flood from
        // `hom_space_id_remains_semantic_after_intern_eviction` could evict an
        // entry mid-build and hand a later thread a fresh (non-aliased) Arc.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ids = std::thread::scope(|scope| {
            (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        FusionTreeHomSpace::new(
                            FusionProductSpace::new([u1_leg(41, 7, false)]),
                            FusionProductSpace::new([u1_leg(41, 9, true)]),
                        )
                        .id()
                    })
                })
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(ids.windows(2).all(|pair| pair[0] == pair[1]));
        assert!(ids
            .windows(2)
            .all(|pair| Arc::ptr_eq(&pair[0].key, &pair[1].key)));
    }

    #[test]
    fn coupled_sector_regions_describe_canonical_matrix_spans() {
        // What: canonical coupled storage compiles to exact sector ranges and tree extents.
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(z2_even(), 2), (z2_odd(), 2)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );
        let keys = homspace.fusion_tree_keys(&rule);
        let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims([4, 4], [4, 4]).unwrap(),
            homspace,
            &rule,
            vec![vec![2; 4]; keys.len()],
        )
        .unwrap();

        let structure = space.subblock_structure();
        let cloned_before_query = structure.as_ref().clone();
        assert!(!structure.coupled_region_cache_is_initialized());
        let cold_charge = structure.charged_retained_bytes();
        assert!(!structure.coupled_region_cache_is_initialized());
        let regions = cloned_before_query
            .coupled_sector_regions(2)
            .unwrap()
            .unwrap();
        assert!(structure.coupled_region_cache_is_initialized());
        let original_regions = structure
            .coupled_sector_regions(2)
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&regions, &original_regions));

        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].range().start, 0);
        assert_eq!(regions[0].range().len(), regions[0].rows() * regions[0].cols());
        assert_eq!(regions[1].range().start, regions[0].range().end);
        assert_eq!(regions[1].range().end, space.required_len().unwrap());
        assert!(regions.iter().all(|region| {
            !region.row_trees().is_empty()
                && !region.col_trees().is_empty()
                && region.has_aligned_diagonal()
                && region
                    .row_trees()
                    .iter()
                    .all(|tree| tree.extent().unwrap() > 0)
        }));

        assert_eq!(structure.charged_retained_bytes(), cold_charge);
        for nout in 0..=structure.rank() {
            let _ = structure.coupled_sector_regions(nout);
        }
        assert_eq!(structure.charged_retained_bytes(), cold_charge);
    }

    #[test]
    fn coupled_sector_regions_preserve_literal_expert_tree_order_and_extents() {
        // What: rank-five expert metadata preserves independent first-seen row and
        // column order while compiling a rectangular, nonuniform coupled matrix.
        let row_a = FusionTreeKey::try_from_sector_ids(
            [9, 1, 4],
            7,
            [false, true, false],
            [5],
            [1, 1],
        )
        .unwrap();
        let row_b = FusionTreeKey::try_from_sector_ids(
            [2, 8, 3],
            7,
            [true, false, false],
            [6],
            [1, 1],
        )
        .unwrap();
        let row_c = FusionTreeKey::try_from_sector_ids(
            [6, 0, 5],
            7,
            [false, false, true],
            [4],
            [1, 1],
        )
        .unwrap();
        let col_y =
            FusionTreeKey::try_from_sector_ids([8, 1], 7, [true, false], [], [1]).unwrap();
        let col_x =
            FusionTreeKey::try_from_sector_ids([1, 3], 7, [false, true], [], [1]).unwrap();

        let block = |row: &FusionTreeKey,
                     col: &FusionTreeKey,
                     shape: [usize; 5],
                     strides: [usize; 5],
                     offset| {
            BlockSpec::with_key(
                BlockKey::FusionTree(FusionTreePairKey::pair(row.clone(), col.clone())),
                shape.to_vec(),
                strides.to_vec(),
                offset,
            )
            .unwrap()
        };
        let structure = BlockStructure::from_blocks(vec![
            block(&row_a, &col_y, [2, 1, 1, 2, 1], [1, 2, 2, 9, 18], 0),
            block(&row_b, &col_y, [1, 3, 1, 2, 1], [1, 1, 3, 9, 18], 2),
            block(&row_c, &col_y, [1, 1, 4, 2, 1], [1, 1, 1, 9, 18], 5),
            block(&row_a, &col_x, [2, 1, 1, 1, 3], [1, 2, 2, 9, 9], 18),
            block(&row_b, &col_x, [1, 3, 1, 1, 3], [1, 1, 3, 9, 9], 20),
            block(&row_c, &col_x, [1, 1, 4, 1, 3], [1, 1, 1, 9, 9], 23),
        ])
        .unwrap();

        let regions = structure.coupled_sector_regions(3).unwrap().unwrap();
        let warm_regions = structure.coupled_sector_regions(3).unwrap().unwrap();
        assert!(Arc::ptr_eq(&regions, &warm_regions));
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].coupled(), SectorId::new(7));
        assert_eq!(regions[0].rows(), 9);
        assert_eq!(regions[0].cols(), 5);
        assert_eq!(regions[0].range(), 0..45);
        assert!(!regions[0].has_aligned_diagonal());
        assert_eq!(
            regions[0]
                .row_trees()
                .iter()
                .map(|extent| (extent.tree(), extent.offset(), extent.shape()))
                .collect::<Vec<_>>(),
            vec![
                (&row_a, 0, [2, 1, 1].as_slice()),
                (&row_b, 2, [1, 3, 1].as_slice()),
                (&row_c, 5, [1, 1, 4].as_slice()),
            ]
        );
        assert_eq!(
            regions[0]
                .col_trees()
                .iter()
                .map(|extent| (extent.tree(), extent.offset(), extent.shape()))
                .collect::<Vec<_>>(),
            vec![
                (&col_y, 0, [2, 1].as_slice()),
                (&col_x, 2, [1, 3].as_slice()),
            ]
        );
    }

    #[test]
    fn coupled_sector_regions_require_source_ordered_tree_diagonals() {
        // What: matching row/column extents alone do not prove that walking
        // them preserves the source diagonal-block encounter order.
        let tree = |label| {
            FusionTreeKey::try_from_sector_ids([label], 7, [false], [], []).unwrap()
        };
        let trees = [tree(0), tree(1), tree(2)];
        let block = |row: usize, col: usize| {
            BlockSpec::with_key(
                BlockKey::FusionTree(FusionTreePairKey::pair(
                    trees[row].clone(),
                    trees[col].clone(),
                )),
                vec![1, 1],
                vec![1, 3],
                row + 3 * col,
            )
            .unwrap()
        };
        let ordered = BlockStructure::from_blocks(
            [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]
                .into_iter()
                .map(|(row, col)| block(row, col))
                .collect(),
        )
        .unwrap();
        assert!(ordered.coupled_sector_regions(1).unwrap().unwrap()[0].has_aligned_diagonal());

        let reordered = BlockStructure::from_blocks(
            [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (2, 2), (1, 1), (1, 2), (2, 1)]
                .into_iter()
                .map(|(row, col)| block(row, col))
                .collect(),
        )
        .unwrap();
        let regions = reordered.coupled_sector_regions(1).unwrap().unwrap();
        assert!(!regions[0].has_aligned_diagonal());
    }

    #[test]
    fn coupled_sector_regions_preserve_empty_scalar_and_zero_extents() {
        let empty = BlockStructure::from_blocks_with_rank(3, vec![]).unwrap();
        assert_eq!(empty.coupled_sector_regions(2).unwrap().unwrap().len(), 0);

        let scalar_tree = FusionTreeKey::try_from_sector_ids([], 0, [], [], []).unwrap();
        let scalar = BlockStructure::from_blocks(vec![
            BlockSpec::with_key(
                BlockKey::FusionTree(FusionTreePairKey::pair(
                    scalar_tree.clone(),
                    scalar_tree,
                )),
                vec![],
                vec![],
                0,
            )
            .unwrap(),
        ])
        .unwrap();
        let scalar_regions = scalar.coupled_sector_regions(0).unwrap().unwrap();
        assert_eq!(
            (
                scalar_regions[0].rows(),
                scalar_regions[0].cols(),
                scalar_regions[0].range(),
                scalar_regions[0].row_trees()[0].shape(),
                scalar_regions[0].col_trees()[0].shape(),
            ),
            (1, 1, 0..1, [].as_slice(), [].as_slice())
        );

        let row = FusionTreeKey::try_from_sector_ids([3, 4], 2, [false; 2], [], [1]).unwrap();
        let col = FusionTreeKey::try_from_sector_ids([5], 2, [true], [], []).unwrap();
        let zero = BlockStructure::from_blocks(vec![
            BlockSpec::with_key(
                BlockKey::FusionTree(FusionTreePairKey::pair(row, col)),
                vec![0, usize::MAX, usize::MAX],
                vec![1, 0, 0],
                0,
            )
            .unwrap(),
        ])
        .unwrap();
        let zero_regions = zero.coupled_sector_regions(2).unwrap().unwrap();
        assert_eq!(
            (
                zero_regions[0].rows(),
                zero_regions[0].cols(),
                zero_regions[0].range(),
                zero_regions[0].row_trees()[0].shape(),
                zero_regions[0].col_trees()[0].shape(),
            ),
            (0, usize::MAX, 0..0, [0, usize::MAX].as_slice(), [usize::MAX].as_slice())
        );
    }

    #[test]
    fn coupled_sector_regions_reject_shape_mismatch_before_irrelevant_overflow() {
        // What: a repeated row tree must keep its exact shape, and that `None`
        // decision precedes computing the new column tree's overflowing extent.
        let row =
            FusionTreeKey::try_from_sector_ids([3, 4], 2, [false; 2], [], [1]).unwrap();
        let first_col =
            FusionTreeKey::try_from_sector_ids([5, 6], 2, [false; 2], [], [1]).unwrap();
        let overflowing_col =
            FusionTreeKey::try_from_sector_ids([7, 8], 2, [false; 2], [], [1]).unwrap();
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::with_key(
                BlockKey::FusionTree(FusionTreePairKey::pair(
                    row.clone(),
                    first_col,
                )),
                vec![2, 3, 1, 1],
                vec![0; 4],
                0,
            )
            .unwrap(),
            BlockSpec::with_key(
                BlockKey::FusionTree(FusionTreePairKey::pair(row, overflowing_col)),
                vec![1, 6, usize::MAX, 2],
                vec![0; 4],
                0,
            )
            .unwrap(),
        ])
        .unwrap();

        assert_eq!(structure.coupled_sector_regions(2), Ok(None));
    }

    #[test]
    fn coupled_sector_regions_reject_noncanonical_and_incomplete_grids() {
        // What: independently packed subblocks and a missing tree pair cannot claim direct spans.
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );
        let keys = homspace.fusion_tree_keys(&rule);
        let mut offset = 0usize;
        let independently_packed = keys
            .iter()
            .map(|key| {
                let block = BlockSpec::column_major_with_key(
                    BlockKey::FusionTree(key.clone()),
                    vec![1; 4],
                    offset,
                )
                .unwrap();
                offset += 1;
                block
            })
            .collect();
        let independently_packed = BlockStructure::from_blocks(independently_packed).unwrap();
        assert_eq!(
            independently_packed.coupled_sector_regions(2).unwrap(),
            None
        );

        let coupled = BlockStructure::coupled_sector_matrix_with_keys(
            &rule,
            2,
            4,
            keys.iter()
                .cloned()
                .map(|key| (key, vec![1; 4]))
                .collect(),
        )
        .unwrap();
        let incomplete = BlockStructure::from_blocks(
            (0..coupled.block_count() - 1)
                .map(|index| {
                    let block = coupled.block(index).unwrap();
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
        assert_eq!(incomplete.coupled_sector_regions(2).unwrap(), None);
    }

    fn unbound_expert_space<const NOUT: usize, const NIN: usize>(
        blocks: Vec<BlockSpec>,
    ) -> Result<FusionTensorMapSpace<NOUT, NIN>, CoreError> {
        // Storage-only fixtures use opaque ordinals deliberately: categorical
        // key and shape admission is covered independently by try_bind_rule.
        let dense = TensorMapSpace::<NOUT, NIN>::from_dims([1; NOUT], [1; NIN]).unwrap();
        let homspace = FusionTreeHomSpace::from_sector_ids(
            (0..NOUT).map(|_| (0, 1)),
            (0..NIN).map(|_| (0, 1)),
        );
        FusionTensorMapSpace::new_unbound(
            dense,
            homspace,
            BlockStructure::from_blocks_with_rank(NOUT + NIN, blocks).unwrap(),
        )
    }

    #[test]
    fn expert_fusion_space_rejects_self_overlapping_storage() {
        // What: an owning symmetric space cannot assign two logical elements of
        // one block to the same physical element.
        for (block, offset) in [
            (
                BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![1, 1], 0)
                    .unwrap(),
                1,
            ),
            (
                BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 1], vec![0, 1], 0)
                    .unwrap(),
                0,
            ),
        ] {
            assert_eq!(
                unbound_expert_space::<1, 1>(vec![block]),
                Err(CoreError::OverlappingBlockStorage {
                    first_block: 0,
                    second_block: 0,
                    offset,
                })
            );
        }
    }

    #[test]
    fn expert_fusion_space_rejects_cross_block_storage_aliases() {
        // What: distinct logical symmetric blocks cannot own the same physical
        // destination element, and diagnostics preserve caller block order.
        let blocks = vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![2], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 1).unwrap(),
        ];
        assert_eq!(
            unbound_expert_space::<1, 0>(blocks),
            Err(CoreError::OverlappingBlockStorage {
                first_block: 0,
                second_block: 1,
                offset: 2,
            })
        );
    }

    #[test]
    fn expert_fusion_space_accepts_exact_non_overlapping_strided_storage() {
        // What: expert admission retains arbitrary legal layouts, including
        // layouts that a conservative sorted-span proof cannot establish.
        let rank_two_cases = [
            vec![BlockSpec::with_key(
                BlockKey::ordinal(0),
                vec![3, 2],
                vec![2, 3],
                0,
            )
            .unwrap()],
            vec![BlockSpec::with_key(
                BlockKey::ordinal(0),
                vec![2, 3],
                vec![3, 1],
                0,
            )
            .unwrap()],
            vec![
                BlockSpec::with_key(BlockKey::ordinal(0), vec![1, 2], vec![0, 1], 0)
                    .unwrap(),
            ],
        ];
        reset_exact_storage_fallback_count();
        for blocks in rank_two_cases {
            unbound_expert_space::<1, 1>(blocks).unwrap();
        }
        assert!(exact_storage_fallback_count() > 0);

        let rank_one_cases = [
            vec![
                BlockSpec::with_key(BlockKey::ordinal(0), vec![4], vec![2], 0).unwrap(),
                BlockSpec::with_key(BlockKey::ordinal(1), vec![4], vec![2], 1).unwrap(),
            ],
            vec![
                BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![1], 4).unwrap(),
                BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 0).unwrap(),
            ],
            vec![
                BlockSpec::with_key(BlockKey::ordinal(0), vec![0], vec![0], 0).unwrap(),
                BlockSpec::with_key(BlockKey::ordinal(1), vec![1], vec![1], 0).unwrap(),
            ],
        ];
        for blocks in rank_one_cases {
            unbound_expert_space::<1, 0>(blocks).unwrap();
        }

        unbound_expert_space::<0, 0>(vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![], vec![], 0).unwrap(),
        ])
        .unwrap();
        unbound_expert_space::<1, 0>(vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![0], vec![1], 0).unwrap(),
        ])
        .unwrap();
    }

    #[test]
    fn general_block_structure_retains_aliasing_read_view_contract() {
        // What: storage ownership admission belongs to FusionTensorMapSpace;
        // general block metadata remains usable for intentionally aliased views.
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![1], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 0).unwrap(),
        ])
        .unwrap();
        assert_eq!(structure.required_len().unwrap(), 2);
    }

    fn assert_expert_storage_admission_for_rule<R>(rule: &R, sector: SectorId)
    where
        R: MultiplicityFreeFusionRule,
    {
        let homspace = FusionTreeHomSpace::from_sectors([(sector, 2)], [(sector, 3)]);
        let key = homspace.fusion_tree_keys(rule)[0].clone();
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::with_key(BlockKey::FusionTree(key), vec![2, 3], vec![1, 4], 2).unwrap(),
        ])
        .unwrap();
        FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
            homspace,
            structure,
        )
        .unwrap()
        .try_bind_rule(rule)
        .unwrap();
    }

    #[test]
    fn expert_storage_admission_is_symmetry_independent() {
        // What: the same exact custom-layout admission and categorical binding
        // succeeds for U(1), SU(2), fZ2, and their nested product.
        assert_expert_storage_admission_for_rule(&U1FusionRule, u1(0));
        assert_expert_storage_admission_for_rule(&SU2FusionRule, su2(0));
        assert_expert_storage_admission_for_rule(&FermionParityFusionRule, z2_even());

        type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type Triple = ProductFusionRule<Fz2U1, SU2FusionRule>;
        let pair = Fz2U1::new(FermionParityFusionRule, U1FusionRule);
        let pair_vacuum = pair.encode_sector(z2_even(), u1(0));
        let triple = Triple::new(pair, SU2FusionRule);
        let vacuum = triple.encode_sector(pair_vacuum, su2(0));
        assert_expert_storage_admission_for_rule(&triple, vacuum);
    }

    #[cfg(feature = "racah-generated")]
    #[test]
    fn checked_generic_coupled_storage_is_admitted_by_witness() {
        // What: checked-Generic SU(3) destinations with outer multiplicity use
        // the same coupled-sector builder, so both the staged preview and the
        // committed structure carry the witness.
        let rule = SUNFusionRule::new(3).unwrap();
        let fundamental = rule.encode_dynkin(&[1, 0]).unwrap();
        let adjoint = rule.encode_dynkin(&[1, 1]).unwrap();
        let trivial = rule.encode_dynkin(&[0, 0]).unwrap();
        let leg = |dual| SectorLeg::new([(fundamental, 3), (adjoint, 2), (trivial, 2)], dual);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false), leg(true)]),
            FusionProductSpace::new([leg(false), leg(true)]),
        );
        let prepared = hom
            .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(&rule)
            .unwrap();
        assert_canonical_storage_admitted_without_enumeration(prepared.structure());
        assert_canonical_storage_admitted_without_enumeration(&prepared.commit());
    }

    #[test]
    fn external_aliased_storage_keeps_exact_rejection() {
        // What: layouts without the construction witness keep the exact check
        // and its error, even when they interleave like coupled subblocks.
        let aliased = BlockStructure::from_blocks(vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![1, 3], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![2, 2], vec![1, 3], 1).unwrap(),
        ])
        .unwrap();
        assert!(!aliased.storage_tiling_proven());
        reset_exact_storage_fallback_count();
        assert_eq!(
            validate_block_storage_injective(&aliased),
            Err(CoreError::OverlappingBlockStorage {
                first_block: 0,
                second_block: 1,
                offset: 1,
            })
        );
        assert!(exact_storage_fallback_count() > 0);
    }

    fn local_block_structure_intern_key(index: usize) -> BlockStructureInternKey {
        BlockStructureInternKey {
            rank: 1,
            blocks: Arc::from([BlockStructureContentBlock {
                key: BlockKey::ordinal(index),
                shape: smallvec![1],
                strides: smallvec![1],
                offset: 0,
            }]),
        }
    }

    fn local_block_structure_intern(
        table: &mut BlockStructureInternTable,
        key: BlockStructureInternKey,
        charged_key_bytes: usize,
    ) -> Arc<BlockStructureContent> {
        let rank = key.rank;
        let blocks = Arc::clone(&key.blocks);
        let sector = SectorStructure::from_keys(
            rank,
            blocks.iter().map(|block| block.key.clone()).collect::<Vec<_>>(),
        )
        .unwrap();
        let degeneracy = DegeneracyStructure::from_blocks_with_rank(
            rank,
            blocks
                .iter()
                .map(|block| {
                    DegeneracyBlock::new(
                        block.shape.clone(),
                        block.strides.clone(),
                        block.offset,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
        )
        .unwrap();
        let required_len = degeneracy.required_len().unwrap();
        table.intern_with(key, |_| charged_key_bytes, || {
            Arc::new(BlockStructureContent {
                id: BLOCK_STRUCTURE_CONTENT_ID.fetch_add(1, Ordering::Relaxed),
                sector,
                degeneracy,
                blocks,
                required_len,
                storage_tiling: Default::default(),
            })
        })
    }

    #[test]
    fn block_structure_intern_charge_counts_only_spilled_smallvec_storage() {
        // What: inline SmallVec storage adds no heap charge, while spilled
        // storage contributes its full heap capacity in item bytes.
        let inline: SmallVec<[u64; 2]> = smallvec::smallvec![1_u64, 2];
        let spilled: SmallVec<[u64; 2]> = smallvec::smallvec![1_u64, 2, 3];
        assert!(!inline.spilled());
        assert!(spilled.spilled());
        assert_eq!(spilled_smallvec_heap_bytes(&inline), 0);
        assert_eq!(
            spilled_smallvec_heap_bytes(&spilled),
            spilled.capacity() * std::mem::size_of::<u64>()
        );
    }

    #[test]
    fn block_structure_intern_entry_pressure_evicts_oldest() {
        // What: entry pressure removes the oldest admitted key, and a read hit
        // does not promote it in the FIFO order.
        let key0 = local_block_structure_intern_key(0);
        let key1 = local_block_structure_intern_key(1);
        let key2 = local_block_structure_intern_key(2);
        let charge = charged_block_structure_intern_key_bytes(&key0);
        assert_eq!(charged_block_structure_intern_key_bytes(&key1), charge);
        assert_eq!(charged_block_structure_intern_key_bytes(&key2), charge);
        let mut table = BlockStructureInternTable::new(
            2,
            charge.saturating_mul(3),
            charge,
        );

        let _content0 = local_block_structure_intern(&mut table, key0.clone(), charge);
        let _content1 = local_block_structure_intern(&mut table, key1.clone(), charge);
        assert!(table.lookup(&key0).is_some());
        let _content2 = local_block_structure_intern(&mut table, key2.clone(), charge);

        assert!(table.lookup(&key0).is_none());
        assert!(table.lookup(&key1).is_some());
        assert!(table.lookup(&key2).is_some());
        let info = table.info();
        assert_eq!(info.entries(), 2);
        assert_eq!(info.pressure_evictions(), 1);
    }

    #[test]
    fn block_structure_intern_byte_pressure_subtracts_exact_charge() {
        // What: byte pressure subtracts the evicted entry's unequal stored
        // charge exactly before admitting the incoming key.
        let key0 = local_block_structure_intern_key(10);
        let key1 = local_block_structure_intern_key(11);
        let key2 = local_block_structure_intern_key(12);
        let base_charge = charged_block_structure_intern_key_bytes(&key0);
        let charges = [base_charge, base_charge + 1, base_charge + 2];
        let budget = charges[1].saturating_add(charges[2]);
        let mut table = BlockStructureInternTable::new(3, budget, charges[2]);

        let _content0 = local_block_structure_intern(&mut table, key0.clone(), charges[0]);
        let _content1 = local_block_structure_intern(&mut table, key1.clone(), charges[1]);
        assert_eq!(
            table.info().charged_key_bytes(),
            charges[0].saturating_add(charges[1])
        );
        assert_eq!(table.info().pressure_evictions(), 0);
        assert!(table.lookup(&key0).is_some());

        let _content2 = local_block_structure_intern(&mut table, key2.clone(), charges[2]);
        assert!(table.lookup(&key0).is_none());
        assert!(table.lookup(&key1).is_some());
        assert!(table.lookup(&key2).is_some());
        assert_eq!(table.info().charged_key_bytes(), budget);
        assert_eq!(table.info().pressure_evictions(), 1);
    }

    #[test]
    fn block_structure_intern_bypasses_oversized_and_saturated_charges() {
        // What: oversized and saturated charges return complete content but
        // never consume an entry or charged-byte budget.
        let oversized_key = local_block_structure_intern_key(20);
        let saturated_key = local_block_structure_intern_key(21);
        let charge = charged_block_structure_intern_key_bytes(&oversized_key);
        let mut oversized_table = BlockStructureInternTable::new(2, charge, charge - 1);

        let oversized =
            local_block_structure_intern(&mut oversized_table, oversized_key.clone(), charge);
        assert_eq!(oversized.rank(), oversized_key.rank);
        assert_eq!(oversized.blocks(), oversized_key.blocks.as_ref());
        assert!(oversized_table.lookup(&oversized_key).is_none());
        assert_eq!(oversized_table.info().oversized_admission_bypasses(), 1);

        let mut saturated_table = BlockStructureInternTable::new(2, usize::MAX, usize::MAX);
        let saturated = local_block_structure_intern(
            &mut saturated_table,
            saturated_key.clone(),
            usize::MAX,
        );
        assert_eq!(saturated.rank(), saturated_key.rank);
        assert_eq!(saturated.blocks(), saturated_key.blocks.as_ref());
        assert!(saturated_table.lookup(&saturated_key).is_none());

        let info = saturated_table.info();
        assert_eq!(info.entries(), 0);
        assert_eq!(info.charged_key_bytes(), 0);
        assert_eq!(info.oversized_admission_bypasses(), 1);
    }

    #[test]
    fn block_structure_intern_dead_replacement_preserves_fifo_accounting() {
        // What: replacing a dead Weak changes only its content epoch; entry
        // count, charge, counters, and oldest-first eviction order stay fixed.
        let key0 = local_block_structure_intern_key(30);
        let key1 = local_block_structure_intern_key(31);
        let key2 = local_block_structure_intern_key(32);
        let charge = charged_block_structure_intern_key_bytes(&key0);
        let mut table = BlockStructureInternTable::new(
            2,
            charge.saturating_mul(2),
            charge,
        );

        let content0 = local_block_structure_intern(&mut table, key0.clone(), charge);
        let id0 = content0.id();
        let _content1 = local_block_structure_intern(&mut table, key1.clone(), charge);
        let before = table.info();
        drop(content0);
        assert!(table.lookup(&key0).is_none());

        let replacement = local_block_structure_intern(&mut table, key0.clone(), charge);
        assert!(replacement.id() > id0);
        assert_eq!(table.info(), before);

        let _content2 = local_block_structure_intern(&mut table, key2.clone(), charge);
        assert!(table.lookup(&key0).is_none());
        assert!(table.lookup(&key1).is_some());
        assert!(table.lookup(&key2).is_some());
        assert_eq!(table.info().pressure_evictions(), 1);
    }

    #[test]
    fn block_structure_intern_clear_resets_resources_and_counters() {
        // What: clear releases every admitted key and resets byte, eviction,
        // and bypass accounting while preserving configured limits.
        let key0 = local_block_structure_intern_key(40);
        let key1 = local_block_structure_intern_key(41);
        let key2 = local_block_structure_intern_key(42);
        let charge = charged_block_structure_intern_key_bytes(&key0);
        let mut table = BlockStructureInternTable::new(1, charge, charge);
        let _content0 = local_block_structure_intern(&mut table, key0, charge);
        let _content1 = local_block_structure_intern(&mut table, key1, charge);
        let _content2 = local_block_structure_intern(&mut table, key2, usize::MAX);
        assert_eq!(table.info().pressure_evictions(), 1);
        assert_eq!(table.info().oversized_admission_bypasses(), 1);

        table.clear();

        let info = table.info();
        assert_eq!(info.entries(), 0);
        assert_eq!(info.entry_capacity(), 1);
        assert_eq!(info.charged_key_bytes(), 0);
        assert_eq!(info.byte_budget(), charge);
        assert_eq!(info.max_admitted_entry_bytes(), charge);
        assert_eq!(info.pressure_evictions(), 0);
        assert_eq!(info.oversized_admission_bypasses(), 0);
    }

    fn coupled_z2_matrix_structure() -> BlockStructure {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(z2_even(), 2), (z2_odd(), 3)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        );
        let blocks = homspace
            .fusion_tree_keys(&rule)
            .iter()
            .cloned()
            .map(|key| (key, vec![2, 3]))
            .collect();
        BlockStructure::coupled_sector_matrix_with_keys(&rule, 1, 2, blocks).unwrap()
    }

    #[test]
    fn frozen_content_outlives_wrapper_local_coupled_regions() {
        // What: equal live wrappers canonicalized through the weak table share
        // coupled-region state, while retained immutable content cannot retain it.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();

        let first = coupled_z2_matrix_structure().into_shared();
        let second = coupled_z2_matrix_structure().into_shared();
        assert!(Arc::ptr_eq(&first, &second));
        let first_content = first.content_key();
        let second_content = second.content_key();
        assert!(Arc::ptr_eq(&first_content, &second_content));

        let first_regions = first.coupled_sector_regions(1).unwrap().unwrap();
        let second_regions = second.coupled_sector_regions(1).unwrap().unwrap();
        assert!(Arc::ptr_eq(&first_regions, &second_regions));

        let expected_sector = first.sector_structure().clone();
        let expected_degeneracy = first.degeneracy_structure().clone();
        let expected_len = first.required_len().unwrap();
        let expected_regions = first_regions.as_ref().to_vec();
        let expected_blocks = (0..first.block_count())
            .map(|index| {
                let block = first.block(index).unwrap();
                (
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
            })
            .collect::<Vec<_>>();
        let weak_regions = first.weak_region_state();

        drop(first_regions);
        drop(second_regions);
        drop(second_content);
        drop(first);
        drop(second);
        assert!(weak_regions.upgrade().is_none());

        let rebuilt = coupled_z2_matrix_structure();
        assert_eq!(rebuilt.sector_structure(), &expected_sector);
        assert_eq!(rebuilt.degeneracy_structure(), &expected_degeneracy);
        assert_eq!(rebuilt.required_len().unwrap(), expected_len);
        assert_eq!(
            (0..rebuilt.block_count())
                .map(|index| {
                    let block = rebuilt.block(index).unwrap();
                    (
                        block.key().clone(),
                        block.shape().to_vec(),
                        block.strides().to_vec(),
                        block.offset(),
                    )
                })
                .collect::<Vec<_>>(),
            expected_blocks
        );
        assert_eq!(
            rebuilt
                .coupled_sector_regions(1)
                .unwrap()
                .unwrap()
                .as_ref(),
            expected_regions
        );
        drop(first_content);
    }

    #[test]
    fn concurrent_equal_live_content_canonicalizes_once() {
        // What: concurrent equal construction shares one content Arc and id
        // while every returned structure remains live.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        let barrier = std::sync::Barrier::new(8);
        let structures = std::thread::scope(|scope| {
            let barrier = &barrier;
            let threads = (0..8)
                .map(|_| {
                    scope.spawn(move || {
                        barrier.wait();
                        BlockStructure::trivial(&[17, 19]).unwrap()
                    })
                })
                .collect::<Vec<_>>();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        let content = structures[0].content_key();
        let id = content.id();
        for structure in &structures[1..] {
            let candidate = structure.content_key();
            assert!(Arc::ptr_eq(&content, &candidate));
            assert_eq!(candidate.id(), id);
        }
    }

    #[test]
    fn reset_core_intern_tables_clears_without_reusing_ids() {
        // What: reset preserves a surviving structure and its published region
        // while equal content rebuilt afterward receives a fresh monotonic id.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let before = coupled_z2_matrix_structure();
        let id_before = before.content_id();
        let regions_before = before.coupled_sector_regions(1).unwrap().unwrap();

        reset_core_intern_tables();

        let surviving_regions = before.coupled_sector_regions(1).unwrap().unwrap();
        assert!(Arc::ptr_eq(&regions_before, &surviving_regions));
        assert_eq!(before.content_id(), id_before);

        let after = coupled_z2_matrix_structure();
        let id_after = after.content_id();
        assert!(
            id_after > id_before,
            "reset must not reuse content ids, got before={id_before} after={id_after}"
        );
    }

    #[test]
    fn checked_u1_rejects_the_unlabelled_id_and_preserves_valid_boundaries() {
        // What: the excluded zigzag ID is typed, while boundary sums that
        // remain representable are identical to the expert infallible path.
        let rule = U1FusionRule;
        assert_eq!(
            rule.try_dual_sector(excluded_u1_id()),
            Err(FusionAlgebraError::InvalidSector {
                sector: excluded_u1_id()
            })
        );
        for (left, right) in [(i32::MAX, 1), (i32::MIN + 1, -1)] {
            assert_eq!(
                rule.try_fusion_channels(u1(left), u1(right)),
                Err(FusionAlgebraError::U1FusionOverflow { left, right })
            );
        }
        for (left, right, expected) in [
            (i32::MAX, 0, i32::MAX),
            (i32::MIN + 1, 0, i32::MIN + 1),
            (i32::MAX, i32::MIN + 1, 0),
        ] {
            let checked = rule.try_fusion_channels(u1(left), u1(right)).unwrap();
            assert_eq!(checked.as_slice(), &[u1(expected)]);
            assert_eq!(checked, rule.fusion_channels(u1(left), u1(right)));
            assert_eq!(
                rule.try_nsymbol(u1(left), u1(right), u1(expected)),
                Ok(rule.nsymbol(u1(left), u1(right), u1(expected)))
            );
        }
        assert_eq!(
            rule.try_dual_sector(u1(i32::MAX)).unwrap(),
            rule.dual(u1(i32::MAX))
        );
    }

    #[test]
    fn checked_su2_distinguishes_invalid_inputs_from_unrepresentable_fusion() {
        // What: valid SU2 inputs whose output exceeds the supported algebra
        // report closure failure, while the exact boundary matches the hot path.
        let rule = SU2FusionRule;
        let boundary = rule
            .try_fusion_channels(su2(127), su2(127))
            .unwrap();
        assert_eq!(boundary, rule.fusion_channels(su2(127), su2(127)));
        assert_eq!(
            rule.try_fusion_channels(su2(128), su2(127)),
            Err(FusionAlgebraError::FusionNotRepresentable {
                left: su2(128),
                right: su2(127),
            })
        );
        assert_eq!(
            rule.try_fusion_channels(SectorId::new(255), su2(0)),
            Err(FusionAlgebraError::InvalidSector {
                sector: SectorId::new(255),
            })
        );
    }

    #[test]
    fn checked_fibonacci_matches_valid_operations_and_rejects_unknown_sectors() {
        // What: Fibonacci's checked companion preserves every valid operation
        // and rejects IDs outside the two-sector algebra with the exact input.
        let rule = FibonacciFusionRule;
        let vacuum = SectorId::new(0);
        let tau = SectorId::new(1);
        assert_eq!(rule.try_dual_sector(tau), Ok(rule.dual(tau)));
        assert_eq!(
            rule.try_fusion_channels(tau, tau),
            Ok(rule.fusion_channels(tau, tau))
        );
        for coupled in [vacuum, tau] {
            assert_eq!(
                rule.try_nsymbol(tau, tau, coupled),
                Ok(rule.nsymbol(tau, tau, coupled))
            );
        }
        let invalid = SectorId::new(2);
        assert_eq!(
            rule.try_dual_sector(invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_fusion_channels(tau, invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_nsymbol(tau, tau, invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
    }

    #[derive(Clone, Copy, Debug)]
    struct CheckedMultiplicityRule<const N: usize>;

    impl<const N: usize> FusionRule for CheckedMultiplicityRule<N> {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, _left: SectorId, _right: SectorId) -> SectorVec {
            core::iter::once(SectorId::new(0)).collect()
        }

        fn nsymbol(&self, _left: SectorId, _right: SectorId, _coupled: SectorId) -> usize {
            N
        }
    }

    impl<const N: usize> CheckedFusionAlgebra for CheckedMultiplicityRule<N> {
        fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
            if sector == SectorId::new(0) {
                Ok(sector)
            } else {
                Err(FusionAlgebraError::InvalidSector { sector })
            }
        }

        fn try_fusion_channels(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, FusionAlgebraError> {
            self.try_dual_sector(left)?;
            self.try_dual_sector(right)?;
            Ok(self.fusion_channels(left, right))
        }

        fn try_nsymbol(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Result<usize, FusionAlgebraError> {
            self.try_dual_sector(left)?;
            self.try_dual_sector(right)?;
            self.try_dual_sector(coupled)?;
            Ok(N)
        }
    }

    #[test]
    fn checked_product_reports_multiplicity_overflow_without_panicking() {
        // What: product multiplicities that exceed usize return the exact
        // structured overflow instead of wrapping or entering the hot path.
        type Rule =
            ProductFusionRule<CheckedMultiplicityRule<{ usize::MAX }>, CheckedMultiplicityRule<2>>;
        let rule = Rule::new(CheckedMultiplicityRule, CheckedMultiplicityRule);
        let sector = rule
            .try_encode_sector(SectorId::new(0), SectorId::new(0))
            .unwrap();
        assert_eq!(
            rule.try_nsymbol(sector, sector, sector),
            Err(FusionAlgebraError::MultiplicityOverflow {
                left: sector,
                right: sector,
                coupled: sector,
            })
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn checked_products_preserve_child_u1_errors_and_distinguish_codec_errors() {
        // What: recursive products retain the exact U1 closure cause, while
        // malformed packed IDs remain a distinct codec failure.
        type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
        type Fz2U1Rule =
            ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
        type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
        type TripleLayout = ProductSectorLayout<Fz2U1Layout, Su2SectorLayout>;
        type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

        let pair = Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule);
        let pair_min = pair
            .try_encode_sector(z2_odd(), excluded_u1_id())
            .unwrap();
        assert_eq!(
            pair.try_dual_sector(pair_min),
            Err(FusionAlgebraError::InvalidSector {
                sector: excluded_u1_id()
            })
        );
        let pair_max = pair.try_encode_sector(z2_even(), u1(i32::MAX)).unwrap();
        let pair_one = pair.try_encode_sector(z2_odd(), u1(1)).unwrap();
        assert_eq!(
            pair.try_fusion_channels(pair_max, pair_one),
            Err(FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            })
        );

        let triple = TripleRule::new(pair, SU2FusionRule);
        let triple_min = triple.try_encode_sector(pair_min, su2(1)).unwrap();
        assert_eq!(
            triple.try_dual_sector(triple_min),
            Err(FusionAlgebraError::InvalidSector {
                sector: excluded_u1_id()
            })
        );
        let invalid = SectorId::new(1usize << TripleLayout::BITS);
        assert!(matches!(
            triple.try_dual_sector(invalid),
            Err(FusionAlgebraError::ProductCodec(
                ProductSectorCodecError::InvalidHighBits { .. }
            ))
        ));
    }

    #[test]
    fn checked_fusion_algebra_is_object_safe_and_matches_closed_builtins() {
        // What: callers can use checked algebra through one provider object,
        // and closed built-ins retain their infallible results exactly.
        let checked: &dyn CheckedFusionAlgebra = &U1FusionRule;
        assert_eq!(checked.try_dual_sector(u1(7)), Ok(u1(-7)));
        for rule in [
            &Z2FusionRule as &dyn CheckedFusionAlgebra,
            &FermionParityFusionRule,
            &SU2FusionRule,
        ] {
            let left = rule.vacuum();
            let right = rule.vacuum();
            assert_eq!(rule.try_dual_sector(left), Ok(rule.dual(left)));
            assert_eq!(
                rule.try_fusion_channels(left, right),
                Ok(rule.fusion_channels(left, right))
            );
            assert_eq!(
                rule.try_nsymbol(left, right, rule.vacuum()),
                Ok(rule.nsymbol(left, right, rule.vacuum()))
            );
        }
    }

    #[test]
    fn u1_trivial_a_b_symbols_accept_lowest_charge_valid_triples() {
        // What: trivial U1 rigidity symbols remain exactly one at the lowest
        // representable charge.
        let rule = U1FusionRule;
        assert_eq!(
            rule.a_symbol_scalar(u1(i32::MIN + 1), u1(0), u1(i32::MIN + 1)),
            1.0
        );
        assert_eq!(
            rule.b_symbol_scalar(u1(0), u1(i32::MIN + 1), u1(i32::MIN + 1)),
            1.0
        );
    }

    fn assert_checked_contract_matches_infallible<R>(rule: &R, sector: SectorId)
    where
        R: CheckedFusionAlgebra,
    {
        let leg = SectorLeg::new([(sector, 2)], false);
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg]),
            FusionProductSpace::new([]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([]),
            FusionProductSpace::new([]),
        );
        let expected =
            FusionTreeHomSpace::tensorcontract_homspace(rule, &lhs, &rhs, &[], &[], &[0], 0)
                .unwrap();
        let actual = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            rule,
            &lhs,
            &rhs,
            &[],
            &[],
            &[0],
            0,
        )
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn checked_tensorcontract_matches_closed_builtin_orientation() {
        // What: checked orientation is semantically identical to the established
        // infallible path for every closed built-in algebra and a product rule.
        assert_checked_contract_matches_infallible(&Z2FusionRule, z2_odd());
        assert_checked_contract_matches_infallible(&FermionParityFusionRule, z2_odd());
        assert_checked_contract_matches_infallible(&U1FusionRule, u1(7));
        assert_checked_contract_matches_infallible(&SU2FusionRule, su2(3));

        #[cfg(target_pointer_width = "64")]
        {
            type Rule = ProductFusionRule<U1FusionRule, Z2FusionRule, TensorKitProductCodec>;
            let rule = Rule::new(U1FusionRule, Z2FusionRule);
            let sector = TensorKitProductCodec::encode(u1(4), z2_odd());
            assert_checked_contract_matches_infallible(&rule, sector);
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct UnitLayoutGenericRule;

    impl FusionRule for UnitLayoutGenericRule {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, 0) => [SectorId::new(0)].into_iter().collect(),
                (0, 1) | (1, 0) => [SectorId::new(1)].into_iter().collect(),
                (1, 1) => [SectorId::new(0)].into_iter().collect(),
                _ => SectorVec::new(),
            }
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            match (left.id(), right.id(), coupled.id()) {
                (1, 1, 0) => 2,
                (0, a, b) | (a, 0, b) if a == b && a < 2 => 1,
                _ => 0,
            }
        }
    }

    impl CanonicalUnitFusionRule for UnitLayoutGenericRule {}

    impl CheckedFusionAlgebra for UnitLayoutGenericRule {
        fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
            (sector.id() < 2)
                .then_some(sector)
                .ok_or(FusionAlgebraError::InvalidSector { sector })
        }

        fn try_fusion_channels(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, FusionAlgebraError> {
            self.try_dual_sector(left)?;
            self.try_dual_sector(right)?;
            Ok(self.fusion_channels(left, right))
        }

        fn try_nsymbol(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Result<usize, FusionAlgebraError> {
            self.try_dual_sector(left)?;
            self.try_dual_sector(right)?;
            self.try_dual_sector(coupled)?;
            Ok(self.nsymbol(left, right, coupled))
        }
    }

    #[test]
    fn unit_layout_correspondence_preserves_generic_tree_and_storage_order() {
        // What: canonical-unit insertion preserves every nonunit tree label and
        // every existing layout coordinate, including Generic multiplicity.
        let rule = UnitLayoutGenericRule;
        let one = SectorId::new(1);
        let vacuum = rule.vacuum();
        let two = MultiplicityIndex::new(2).unwrap();
        let small_homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(vacuum, 2), (one, 2)], false),
                SectorLeg::new([(vacuum, 3), (one, 3)], false),
                SectorLeg::new([(vacuum, 4), (one, 4)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(vacuum, 5), (one, 5)], false)]),
        );
        for (position, shape, strides) in [
            (0, vec![1, 2, 3, 4, 5], vec![99, 1, 2, 6, 24]),
            (1, vec![2, 1, 3, 4, 5], vec![1, 99, 2, 6, 24]),
            (2, vec![2, 3, 1, 4, 5], vec![1, 2, 99, 6, 24]),
            (3, vec![2, 3, 4, 1, 5], vec![1, 2, 6, 99, 24]),
        ] {
            let large_homspace = small_homspace.insert_right_unit(&rule, position, true).unwrap();
            let mut small_specs = Vec::new();
            let mut large_specs = Vec::new();
            for (offset, (sector, vertex)) in [(0, (one, two)), (120, (vacuum, MultiplicityIndex::ONE))] {
                let small_key = FusionTreePairKey::pair(
                    FusionTreeKey::new([sector, sector, sector], sector, [false, false, false], [vacuum], [vertex, MultiplicityIndex::ONE]),
                    FusionTreeKey::new([sector], sector, [false], [], []),
                );
                let (uncoupled, duals, innerlines, vertices) = match position {
                    0 => (
                        vec![vacuum, sector, sector, sector],
                        vec![true, false, false, false],
                        vec![sector, vacuum],
                        vec![MultiplicityIndex::ONE, vertex, MultiplicityIndex::ONE],
                    ),
                    1 => (
                        vec![sector, vacuum, sector, sector],
                        vec![false, true, false, false],
                        vec![sector, vacuum],
                        vec![MultiplicityIndex::ONE, vertex, MultiplicityIndex::ONE],
                    ),
                    2 => (
                        vec![sector, sector, vacuum, sector],
                        vec![false, false, true, false],
                        vec![vacuum, vacuum],
                        vec![vertex, MultiplicityIndex::ONE, MultiplicityIndex::ONE],
                    ),
                    3 => (
                        vec![sector, sector, sector, vacuum],
                        vec![false, false, false, true],
                        vec![vacuum, sector],
                        vec![vertex, MultiplicityIndex::ONE, MultiplicityIndex::ONE],
                    ),
                    _ => unreachable!(),
                };
                let large_key = FusionTreePairKey::pair(
                    FusionTreeKey::new(uncoupled, sector, duals, innerlines, vertices),
                    FusionTreeKey::new([sector], sector, [false], [], []),
                );
                small_specs.push(BlockSpec::with_key(BlockKey::FusionTree(small_key), vec![2, 3, 4, 5], vec![1, 2, 6, 24], offset).unwrap());
                large_specs.push(BlockSpec::with_key(BlockKey::FusionTree(large_key), shape.clone(), strides.clone(), offset).unwrap());
            }
            let small = BlockStructure::from_blocks(small_specs).unwrap();
            let large = BlockStructure::from_blocks(large_specs).unwrap();
            let insertion = UnitLegInsertion::Right { position, dual: true };
            validate_unit_layout_correspondence(&rule, (&small_homspace, &small), (&large_homspace, &large), insertion).unwrap();
            validate_unit_layout_correspondence_checked(&rule, (&small_homspace, &small), (&large_homspace, &large), insertion).unwrap();
            assert_eq!(large_homspace.remove_unit(&rule, position).unwrap(), small_homspace);
            assert_eq!(
                validate_unit_layout_correspondence(
                    &rule,
                    (&small_homspace, &small),
                    (&large_homspace, &large),
                    UnitLegInsertion::Right { position, dual: false },
                ),
                Err(CoreError::UnitLayoutCorrespondence)
            );
            let mut swapped = vec![large.block(1).unwrap(), large.block(0).unwrap()];
            let swapped = BlockStructure::from_blocks(swapped.drain(..).map(|block| BlockSpec::with_key(block.key().clone(), block.shape().to_vec(), block.strides().to_vec(), block.offset()).unwrap()).collect()).unwrap();
            assert_eq!(validate_unit_layout_correspondence(&rule, (&small_homspace, &small), (&large_homspace, &swapped), insertion), Err(CoreError::UnitLayoutCorrespondence));
        }
    }

    #[test]
    fn unit_layout_validator_handles_local_rank_zero_and_one() {
        // What: tree rank is local to the modified side, including the two
        // TensorKit seams where total HomSpace rank does not decide the case.
        let rule = U1FusionRule;
        let vacuum = rule.vacuum();
        let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
        let scalar_structure = BlockStructure::from_blocks(vec![
            BlockSpec::with_key(
                BlockKey::FusionTree(FusionTreePairKey::pair(
                    FusionTreeKey::new([], vacuum, [], [], []),
                    FusionTreeKey::new([], vacuum, [], [], []),
                )),
                vec![], vec![], 0,
            ).unwrap(),
        ]).unwrap();
        for (insertion, larger_homspace, key) in [
            (
                UnitLegInsertion::Left { position: 0, dual: true },
                scalar.insert_left_unit(&rule, 0, true).unwrap(),
                FusionTreePairKey::pair(FusionTreeKey::new([], vacuum, [], [], []), FusionTreeKey::new([vacuum], vacuum, [true], [], [])),
            ),
            (
                UnitLegInsertion::Right { position: 0, dual: true },
                scalar.insert_right_unit(&rule, 0, true).unwrap(),
                FusionTreePairKey::pair(FusionTreeKey::new([vacuum], vacuum, [true], [], []), FusionTreeKey::new([], vacuum, [], [], [])),
            ),
        ] {
            let larger = BlockStructure::from_blocks(vec![BlockSpec::with_key(BlockKey::FusionTree(key), vec![1], vec![7], 0).unwrap()]).unwrap();
            validate_unit_layout_correspondence(&rule, (&scalar, &scalar_structure), (&larger_homspace, &larger), insertion).unwrap();
        }

        let rank_one = FusionTreeHomSpace::new(FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]), FusionProductSpace::new([]));
        let small = BlockStructure::from_blocks(vec![BlockSpec::with_key(BlockKey::FusionTree(FusionTreePairKey::pair(FusionTreeKey::new([vacuum], vacuum, [false], [], []), FusionTreeKey::new([], vacuum, [], [], []))), vec![1], vec![1], 0).unwrap()]).unwrap();
        for (position, insertion) in [(0, UnitLegInsertion::Right { position: 0, dual: true }), (1, UnitLegInsertion::Right { position: 1, dual: true })] {
            let larger_homspace = rank_one.insert_right_unit(&rule, position, true).unwrap();
            let mut duals = vec![false];
            duals.insert(position, true);
            let mut strides = vec![1];
            strides.insert(position, 9);
            let large = BlockStructure::from_blocks(vec![BlockSpec::with_key(BlockKey::FusionTree(FusionTreePairKey::pair(FusionTreeKey::new([vacuum, vacuum], vacuum, duals, [], [MultiplicityIndex::ONE]), FusionTreeKey::new([], vacuum, [], [], []))), vec![1, 1], strides, 0).unwrap()]).unwrap();
            validate_unit_layout_correspondence(&rule, (&rank_one, &small), (&larger_homspace, &large), insertion).unwrap();
        }

        let left_larger = rank_one.insert_left_unit(&rule, 0, true).unwrap();
        let left_large = BlockStructure::from_blocks(vec![BlockSpec::with_key(BlockKey::FusionTree(FusionTreePairKey::pair(FusionTreeKey::new([vacuum, vacuum], vacuum, [true, false], [], [MultiplicityIndex::ONE]), FusionTreeKey::new([], vacuum, [], [], []))), vec![1, 1], vec![9, 1], 0).unwrap()]).unwrap();
        validate_unit_layout_correspondence(&rule, (&rank_one, &small), (&left_larger, &left_large), UnitLegInsertion::Left { position: 0, dual: true }).unwrap();

        let domain_small_homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]),
        );
        let domain_small = BlockStructure::from_blocks(vec![BlockSpec::with_key(BlockKey::FusionTree(FusionTreePairKey::pair(FusionTreeKey::new([vacuum], vacuum, [false], [], []), FusionTreeKey::new([vacuum], vacuum, [false], [], []))), vec![1, 1], vec![1, 1], 0).unwrap()]).unwrap();
        let domain_larger = domain_small_homspace.insert_right_unit(&rule, 2, true).unwrap();
        let domain_large = BlockStructure::from_blocks(vec![BlockSpec::with_key(BlockKey::FusionTree(FusionTreePairKey::pair(FusionTreeKey::new([vacuum], vacuum, [false], [], []), FusionTreeKey::new([vacuum, vacuum], vacuum, [false, true], [], [MultiplicityIndex::ONE]))), vec![1, 1, 1], vec![1, 1, 9], 0).unwrap()]).unwrap();
        validate_unit_layout_correspondence(&rule, (&domain_small_homspace, &domain_small), (&domain_larger, &domain_large), UnitLegInsertion::Right { position: 2, dual: true }).unwrap();
    }

    #[test]
    fn checked_unit_layout_validation_reports_algebra_before_correspondence() {
        // What: checked finite algebra rejects an invalid supplied sector before
        // inspecting the deliberately mismatched unit descriptor.
        let rule = UnitLayoutGenericRule;
        let invalid = SectorId::new(9);
        let vacuum = rule.vacuum();
        let small_homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(invalid, 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(invalid, 1)], false)]),
        );
        let larger_homspace = small_homspace.insert_right_unit(&rule, 0, true).unwrap();
        let small = BlockStructure::from_blocks(vec![BlockSpec::with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(
                FusionTreeKey::new([invalid], invalid, [false], [], []),
                FusionTreeKey::new([invalid], invalid, [false], [], []),
            )),
            vec![1, 1], vec![1, 1], 0,
        ).unwrap()]).unwrap();
        let large = BlockStructure::from_blocks(vec![BlockSpec::with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(
                FusionTreeKey::new([vacuum, invalid], invalid, [true, false], [], [MultiplicityIndex::ONE]),
                FusionTreeKey::new([invalid], invalid, [false], [], []),
            )),
            vec![1, 1, 1], vec![9, 1, 1], 0,
        ).unwrap()]).unwrap();
        assert!(matches!(
            validate_unit_layout_correspondence_checked(
                &rule,
                (&small_homspace, &small),
                (&larger_homspace, &large),
                UnitLegInsertion::Right { position: 0, dual: false },
            ),
            Err(CheckedFusionSpaceError::FusionAlgebra(error))
                if *error == FusionAlgebraError::InvalidSector { sector: invalid }
        ));
    }
