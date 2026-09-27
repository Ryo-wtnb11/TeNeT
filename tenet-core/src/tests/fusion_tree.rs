mod fusion_tree {
    use super::*;

        #[derive(Clone, Copy, Debug)]
        struct UncertifiedCustomSymbolsRule;

        impl FusionRule for UncertifiedCustomSymbolsRule {
            fn rule_identity(&self) -> RuleIdentity {
                RuleIdentity::of_type::<Self>()
            }

            fn fusion_style(&self) -> FusionStyleKind {
                FusionStyleKind::Unique
            }

            fn braiding_style(&self) -> BraidingStyleKind {
                BraidingStyleKind::Bosonic
            }

            fn vacuum(&self) -> SectorId {
                SectorId::new(0)
            }

            fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
                smallvec![SectorId::new(left.id() ^ right.id())]
            }
        }

        impl MultiplicityFreeFusionRule for UncertifiedCustomSymbolsRule {}

        impl MultiplicityFreeFusionSymbols for UncertifiedCustomSymbolsRule {
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
                2.0
            }

            fn r_symbol_scalar(
                &self,
                _left: SectorId,
                _right: SectorId,
                coupled: SectorId,
            ) -> Self::Scalar {
                if coupled.id() == 0 { 3.0 } else { 5.0 }
            }
        }

        #[derive(Clone, Copy, Debug)]
        struct ComplexAsymmetricUniqueRule;

        impl FusionRule for ComplexAsymmetricUniqueRule {
            fn rule_identity(&self) -> RuleIdentity {
                RuleIdentity::of_type::<Self>()
            }

            fn fusion_style(&self) -> FusionStyleKind {
                FusionStyleKind::Unique
            }

            fn braiding_style(&self) -> BraidingStyleKind {
                BraidingStyleKind::Anyonic
            }

            fn vacuum(&self) -> SectorId {
                SectorId::new(0)
            }

            fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
                smallvec![SectorId::new((left.id() + right.id()) % 4)]
            }
        }

        impl MultiplicityFreeFusionRule for ComplexAsymmetricUniqueRule {}

        impl MultiplicityFreeFusionSymbols for ComplexAsymmetricUniqueRule {
            type Scalar = Complex64;

            fn f_symbol_scalar(
                &self,
                _left: SectorId,
                _middle: SectorId,
                _right: SectorId,
                _coupled: SectorId,
                _left_coupled: SectorId,
                _right_coupled: SectorId,
            ) -> Self::Scalar {
                Complex64::new(1.0, 0.0)
            }

            fn r_symbol_scalar(
                &self,
                left: SectorId,
                right: SectorId,
                _coupled: SectorId,
            ) -> Self::Scalar {
                let angle = match (left.id(), right.id()) {
                    (1, 2) => std::f64::consts::FRAC_PI_3,
                    (2, 1) => std::f64::consts::FRAC_PI_6,
                    _ => 0.0,
                };
                Complex64::from_polar(1.0, angle)
            }
        }

        #[test]
        fn fusion_rule_exposes_unique_outputs_and_nsymbol_separately() {
            let z2 = Z2FusionRule;
            let su2 = SU2FusionRule;

            assert_eq!(z2.fusion_style(), FusionStyleKind::Unique);
            assert_eq!(
                z2.fusion_channels(SectorId::new(1), SectorId::new(1))
                    .to_vec(),
                vec![SectorId::new(0)]
            );
            assert_eq!(
                z2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(0)),
                1
            );
            assert_eq!(
                z2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(1)),
                0
            );

            assert_eq!(su2.fusion_style(), FusionStyleKind::Simple);
            assert_eq!(
                su2.fusion_channels(SectorId::new(1), SectorId::new(1))
                    .to_vec(),
                vec![SectorId::new(0), SectorId::new(2)]
            );
            assert_eq!(
                su2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(2)),
                1
            );
        }

        #[test]
        fn multiplicity_free_symbols_are_a_separate_scalar_api() {
            let z2 = Z2FusionRule;

            assert_eq!(<Z2FusionRule as MultiplicityFreeFusionSymbols>::Scalar::one(), 1.0);
            assert_eq!(
                z2.f_symbol_scalar(
                    SectorId::new(1),
                    SectorId::new(1),
                    SectorId::new(1),
                    SectorId::new(1),
                    SectorId::new(0),
                    SectorId::new(0),
                ),
                1.0
            );
            assert_eq!(
                z2.r_symbol_scalar(SectorId::new(1), SectorId::new(1), SectorId::new(0)),
                1.0
            );
        }

        #[test]
        fn unique_artin_braid_first_allows_unit_crossing_without_braiding() {
            let tree = FusionTreeKey::try_from_sector_ids([0, 1], 1, [false, true], [], [1]).unwrap();

            let terms =
                multiplicity_free_braid_tree(&PlanarZ2Rule, &tree, &[1, 0], &[0, 1]).unwrap();
            assert_eq!(terms.len(), 1);
            let (braided, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, 1.0);
            assert_eq!(braided.uncoupled(), &[SectorId::new(1), SectorId::new(0)]);
            assert_eq!(braided.is_dual(), &[true, false]);
            assert_eq!(braided.coupled(), SectorId::new(1));
            assert!(braided.innerlines().is_empty());
            assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE]);
        }

        #[test]
        fn unique_artin_braid_first_rejects_nonunit_crossing_without_braiding() {
            let tree = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, false], [], [1]).unwrap();

            let err =
                multiplicity_free_braid_tree(&PlanarZ2Rule, &tree, &[1, 0], &[0, 1]).unwrap_err();

            assert_eq!(
                err,
                CoreError::UnsupportedSectorBraid {
                    left: SectorId::new(1),
                    right: SectorId::new(1),
                    style: BraidingStyleKind::NoBraiding,
                }
            );
        }

        #[test]
        fn merge_fusion_trees_treats_rank_zero_as_the_tensor_unit() {
            let unit = FusionTreeKey::try_from_sector_ids([], 0, [], [], []).unwrap();
            let tree =
                FusionTreeKey::try_from_sector_ids([1, 0], 1, [true, false], [], [1]).unwrap();

            for (lhs, rhs) in [(&unit, &tree), (&tree, &unit)] {
                let terms =
                    merge_fusion_trees_multiplicity_free(&Z2FusionRule, lhs, rhs, SectorId::new(1))
                        .unwrap();
                assert_eq!(terms, vec![(tree.clone(), 1.0)]);
            }
        }

        #[test]
        fn merge_fusion_trees_pins_nontrivial_su2_f_coefficients() {
            // What: merging a spin-1 tree into three spin-1/2 leaves is a genuine
            // associator expansion, not a pointed-rule relabelling.
            let lhs =
                FusionTreeKey::try_from_sector_ids([1, 1], 2, [false, false], [], [1]).unwrap();
            let rhs =
                FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false; 3], [0], [1, 1]).unwrap();

            let terms =
                merge_fusion_trees_multiplicity_free(&SU2FusionRule, &lhs, &rhs, SectorId::new(1))
                    .unwrap();

            assert_eq!(terms.len(), 2);
            assert_eq!(
                terms
                    .iter()
                    .map(|(tree, _)| tree.innerlines().to_vec())
                    .collect::<Vec<_>>(),
                [
                    vec![SectorId::new(2), SectorId::new(1), SectorId::new(2)],
                    vec![SectorId::new(2), SectorId::new(3), SectorId::new(2)],
                ]
            );
            assert!((terms[0].1 + 1.0 / 3.0_f64.sqrt()).abs() < 1.0e-14);
            assert!((terms[1].1 - (2.0 / 3.0_f64).sqrt()).abs() < 1.0e-14);
        }

        #[test]
        fn unique_artin_braid_first_uses_r_symbol_for_first_crossing() {
            let tree = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, true], [], [1]).unwrap();

            let terms =
                multiplicity_free_braid_tree(&FermionParityFusionRule, &tree, &[1, 0], &[0, 1])
                    .unwrap();
            assert_eq!(terms.len(), 1);
            let (braided, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, -1.0);
            assert_eq!(braided.uncoupled(), &[SectorId::new(1), SectorId::new(1)]);
            assert_eq!(braided.is_dual(), &[true, false]);
            assert_eq!(braided.coupled(), SectorId::new(0));
            assert!(braided.innerlines().is_empty());
            assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE]);
        }

        #[test]
        fn unique_artin_braid_first_uses_first_innerline_for_rank_three() {
            let tree =
                FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1]).unwrap();

            let terms = multiplicity_free_braid_tree(
                &FermionParityFusionRule,
                &tree,
                &[1, 0, 2],
                &[0, 1, 2],
            )
            .unwrap();
            assert_eq!(terms.len(), 1);
            let (braided, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, -1.0);
            assert_eq!(
                braided.uncoupled(),
                &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
            );
            assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
            assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]);
        }

        #[test]
        fn unique_artin_braid_at_updates_innerline_for_later_unit_crossing() {
            let tree =
                FusionTreeKey::try_from_sector_ids([1, 0, 1], 0, [false, false, true], [1], [1, 1]).unwrap();

            let terms =
                multiplicity_free_braid_tree(&PlanarZ2Rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
            assert_eq!(terms.len(), 1);
            let (braided, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, 1.0);
            assert_eq!(
                braided.uncoupled(),
                &[SectorId::new(1), SectorId::new(1), SectorId::new(0)]
            );
            assert_eq!(braided.is_dual(), &[false, true, false]);
            assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
            assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]);
        }

        #[test]
        fn unique_artin_braid_at_uses_f_and_r_symbols_for_later_crossing() {
            let tree =
                FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, true, false], [0], [1, 1]).unwrap();

            let terms = multiplicity_free_braid_tree(
                &FermionParityFusionRule,
                &tree,
                &[0, 2, 1],
                &[0, 1, 2],
            )
            .unwrap();
            assert_eq!(terms.len(), 1);
            let (braided, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, -1.0);
            assert_eq!(
                braided.uncoupled(),
                &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
            );
            assert_eq!(braided.is_dual(), &[false, false, true]);
            assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
            assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]);
        }

        #[test]
        fn permutation_to_adjacent_swaps_matches_tensorkit_order() {
            assert_eq!(
                permutation_to_adjacent_swaps(&[2, 0, 1], 3).unwrap(),
                vec![1, 0]
            );
            assert_eq!(
                permutation_to_adjacent_swaps(&[3, 0, 2, 1], 4).unwrap(),
                vec![2, 1, 0, 2]
            );
        }

        #[test]
        fn unique_braid_tree_replays_tensorkit_swap_order_and_level_updates() {
            let tree =
                FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1]).unwrap();

            let terms = multiplicity_free_braid_tree(
                &FermionParityFusionRule,
                &tree,
                &[2, 0, 1],
                &[0, 1, 2],
            )
            .unwrap();
            assert_eq!(terms.len(), 1);
            let (braided, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, 1.0);
            assert_eq!(
                braided.uncoupled(),
                &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
            );
            assert_eq!(braided.is_dual(), &[false, false, false]);
            assert_eq!(braided.coupled(), SectorId::new(1));
            assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
            assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]);
        }

        #[test]
        fn unique_braid_tree_uses_inverse_artin_branch_from_levels() {
            let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();

            let forward_terms =
                multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[0, 1])
                    .unwrap();
            assert_eq!(forward_terms.len(), 1);
            let (braided_forward, forward) = forward_terms.into_iter().next().unwrap();
            let inverse_terms =
                multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[1, 0])
                    .unwrap();
            assert_eq!(inverse_terms.len(), 1);
            let (braided_inverse, inverse) = inverse_terms.into_iter().next().unwrap();

            assert_eq!(forward, 5.0);
            assert_eq!(inverse, 7.0);
            assert_eq!(braided_forward, braided_inverse);
            assert_eq!(
                braided_forward.uncoupled(),
                &[SectorId::new(2), SectorId::new(1)]
            );
            assert_eq!(braided_forward.coupled(), SectorId::new(3));
        }

        #[test]
        fn unique_braid_tree_reflected_levels_select_inverse_artin_branch() {
            let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();
            let levels = [3, 8];
            let min_level = levels.iter().copied().min().unwrap();
            let max_level = levels.iter().copied().max().unwrap();
            let reflected_levels = levels
                .iter()
                .map(|&level| min_level + max_level - level)
                .collect::<Vec<_>>();

            let forward_terms =
                multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &levels)
                    .unwrap();
            assert_eq!(forward_terms.len(), 1);
            let (forward_tree, forward_coeff) = forward_terms.into_iter().next().unwrap();
            let inverse_terms = multiplicity_free_braid_tree(
                &AsymmetricAnyonicRule,
                &tree,
                &[1, 0],
                &reflected_levels,
            )
            .unwrap();
            assert_eq!(inverse_terms.len(), 1);
            let (inverse_tree, inverse_coeff) = inverse_terms.into_iter().next().unwrap();

            assert_eq!(reflected_levels, vec![8, 3]);
            assert_eq!(forward_tree, inverse_tree);
            assert_eq!(forward_coeff, 5.0);
            assert_eq!(inverse_coeff, 7.0);
        }

        #[test]
        fn symmetric_unique_direct_braid_matches_artin_replay_exactly() {
            // What: every rank-4 fZ2 permutation has the exact tree and sign of
            // TensorKit's adjacent-Artin semantics, including multiplication order.
            let tree = FusionTreeKey::try_from_sector_ids(
                [1, 1, 0, 1], 1,
                [false, true, false, true],
                [0, 0],
                [1, 1, 1],
            ).unwrap();
            let levels = [0, 1, 2, 3];
            let mut permutation = [0usize, 1, 2, 3];
            loop {
                let direct_terms =
                    multiplicity_free_braid_tree(&FermionParityFusionRule, &tree, &permutation, &levels)
                        .unwrap();
                assert_eq!(direct_terms.len(), 1);
                let direct = direct_terms.into_iter().next().unwrap();

                let mut replay_tree = UnhashedFusionTree::from(tree.clone());
                let mut replay_coefficient = 1.0;
                let mut replay_levels = levels;
                for swap in permutation_to_adjacent_swaps(&permutation, 4).unwrap() {
                    let inverse = replay_levels[swap] > replay_levels[swap + 1];
                    let coefficient = apply_unique_artin_braid_at_with_inverse(
                        &FermionParityFusionRule,
                        &mut replay_tree,
                        swap,
                        inverse,
                    )
                    .unwrap();
                    replay_coefficient *= coefficient;
                    replay_levels.swap(swap, swap + 1);
                }
                assert_eq!(direct, (replay_tree.freeze(), replay_coefficient));

                let Some(pivot) =
                    (0..permutation.len() - 1).rfind(|&index| permutation[index] < permutation[index + 1])
                else {
                    break;
                };
                let successor = (pivot + 1..permutation.len())
                    .rfind(|&index| permutation[index] > permutation[pivot])
                    .unwrap();
                permutation.swap(pivot, successor);
                permutation[pivot + 1..].reverse();
            }
        }

        #[test]
        fn rule_aware_tree_validation_covers_rank_shape_style_and_vertices() {
            // What: categorical interpretation requires the canonical empty-tree
            // vacuum and rejects every representable malformed tree field.
            let empty_vacuum = FusionTreeKey::new([], z2_even(), [], [], []);
            empty_vacuum.validate_for_rule(&Z2FusionRule).unwrap();
            assert_eq!(
                FusionTreeKey::new([], z2_odd(), [], [], [])
                    .validate_for_rule(&Z2FusionRule)
                    .unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "rank-0 fusion tree coupled sector must equal the vacuum",
                }
            );

            let rank_one =
                FusionTreeKey::new([z2_odd()], z2_odd(), [true], [], []);
            rank_one.validate_for_rule(&Z2FusionRule).unwrap();
            assert_eq!(
                FusionTreeKey::new([z2_odd()], z2_even(), [true], [], [])
                    .validate_for_rule(&Z2FusionRule)
                    .unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "rank-1 fusion tree coupled sector must equal its uncoupled sector",
                }
            );

            let bad_shapes = [
                (
                    FusionTreeKey::new([z2_odd(), z2_odd()], z2_even(), [false], [], [MultiplicityIndex::ONE]),
                    "fusion tree sectors and duality flags must have matching length",
                ),
                (
                    FusionTreeKey::new(
                        [z2_odd(), z2_odd(), z2_odd()], z2_odd(),
                        [false; 3],
                        [],
                        [MultiplicityIndex::ONE; 2],
                    ),
                    "fusion tree has an invalid number of innerlines",
                ),
                (
                    FusionTreeKey::new(
                        [z2_odd(), z2_odd()], z2_even(),
                        [false; 2],
                        [],
                        [],
                    ),
                    "fusion tree has an invalid number of vertices",
                ),
            ];
            for (tree, message) in bad_shapes {
                assert_eq!(
                    tree.validate_for_rule(&Z2FusionRule).unwrap_err(),
                    CoreError::MalformedFusionTree { message }
                );
            }

            for tree in [
                FusionTreeKey::new(
                    [z2_odd(); 4], z2_even(),
                    [false; 4],
                    [z2_even(), z2_even()],
                    [MultiplicityIndex::ONE; 3],
                ),
                FusionTreeKey::new(
                    [z2_odd(); 4], z2_odd(),
                    [false; 4],
                    [z2_even(), z2_odd()],
                    [MultiplicityIndex::ONE; 3],
                ),
            ] {
                assert_eq!(
                    tree.validate_for_rule(&Z2FusionRule).unwrap_err(),
                    CoreError::MalformedFusionTree {
                        message: "fusion tree contains an inadmissible fusion vertex",
                    }
                );
            }
        }

        #[test]
        fn rule_aware_constructor_enforces_one_based_vertex_bounds() {
            // What: vertex labels are checked against the actual N-symbol for
            // both multiplicity-free and Generic rules.
            assert_eq!(
                FusionTreeKey::try_new_for_rule(
                    &Z2FusionRule,
                    [z2_odd(), z2_odd()], z2_even(),
                    [false; 2],
                    [],
                    [MultiplicityIndex::new(2).expect("test multiplicity label is one-based")],
                )
                .unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "fusion tree vertex label exceeds its fusion multiplicity",
                }
            );
            assert_eq!(
                MultiplicityIndex::try_from(0).unwrap_err(),
                CoreError::InvalidMultiplicityIndex { value: 0 }
            );
            assert_eq!(
                FusionTreeKey::try_new_for_rule(
                    &ToyOmRule,
                    [SectorId::new(ToyOmRule::A), SectorId::new(ToyOmRule::A)],
                    SectorId::new(ToyOmRule::C),
                    [false; 2],
                    [],
                    [MultiplicityIndex::new(3).unwrap()],
                )
                .unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "fusion tree vertex label exceeds its fusion multiplicity",
                }
            );
        }

        #[test]
        fn checked_tree_constructors_report_builtin_closure_without_unwinding() {
            // What: raw checked construction preserves exact finite-algebra
            // failures instead of entering the infallible provider path.
            assert_eq!(
                FusionTreeKey::try_new_for_rule_checked(
                    &U1FusionRule,
                    [u1(i32::MAX), u1(1)],
                    u1(0),
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                ),
                Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                    FusionAlgebraError::U1FusionOverflow {
                        left: i32::MAX,
                        right: 1,
                    }
                )))
            );
            assert_eq!(
                FusionTreeKey::try_from_sector_ids_for_rule_checked(
                    &SU2FusionRule,
                    [128, 127],
                    0,
                    [false; 2],
                    [],
                    [1],
                ),
                Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                    FusionAlgebraError::FusionNotRepresentable {
                        left: su2(128),
                        right: su2(127),
                    }
                )))
            );
        }

        #[test]
        fn checked_tree_distinguishes_absent_channels_from_algebra_failure() {
            // What: a representable but mathematically absent stored channel is a
            // malformed tree, while invalid stored IDs and recursive product
            // closure keep their exact algebra causes.
            assert_eq!(
                FusionTreeKey::new(
                    [u1(1), u1(2)],
                    u1(4),
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                )
                .validate_for_rule_checked(&U1FusionRule),
                Err(CheckedFusionSpaceError::Core(Box::new(
                    CoreError::MalformedFusionTree {
                        message: "fusion tree contains an inadmissible fusion vertex",
                    }
                )))
            );
            let invalid = SectorId::new(2);
            assert_eq!(
                FusionTreeKey::new(
                    [z2_odd(), z2_odd()],
                    invalid,
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                )
                .validate_for_rule_checked(&Z2FusionRule),
                Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                    FusionAlgebraError::InvalidSector { sector: invalid }
                )))
            );

            #[cfg(target_pointer_width = "64")]
            {
                type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
                type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
                let rule = Rule::new(FermionParityFusionRule, U1FusionRule);
                let left = Codec::encode(z2_even(), u1(i32::MAX));
                let right = Codec::encode(z2_odd(), u1(1));
                let coupled = Codec::encode(z2_odd(), u1(0));
                assert_eq!(
                    FusionTreeKey::new(
                        [left, right],
                        coupled,
                        [false; 2],
                        [],
                        [MultiplicityIndex::ONE],
                    )
                    .validate_for_rule_checked(&rule),
                    Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                        FusionAlgebraError::U1FusionOverflow {
                            left: i32::MAX,
                            right: 1,
                        }
                    )))
                );
            }
        }

        fn assert_checked_tree_matches_infallible<R>(rule: &R, tree: FusionTreeKey)
        where
            R: CheckedFusionAlgebra,
        {
            tree.validate_for_rule(rule).unwrap();
            tree.validate_for_rule_checked(rule).unwrap();
        }

        #[test]
        fn checked_tree_matches_closed_builtin_and_nested_product_rules() {
            // What: checked validation accepts the same valid raw trees as the
            // established validator across built-in and recursive product rules.
            assert_checked_tree_matches_infallible(
                &Z2FusionRule,
                FusionTreeKey::new(
                    [z2_odd(); 2],
                    z2_even(),
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                ),
            );
            assert_checked_tree_matches_infallible(
                &FermionParityFusionRule,
                FusionTreeKey::new(
                    [z2_odd(); 2],
                    z2_even(),
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                ),
            );
            assert_checked_tree_matches_infallible(
                &U1FusionRule,
                FusionTreeKey::new(
                    [u1(2), u1(-1)],
                    u1(1),
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                ),
            );
            assert_checked_tree_matches_infallible(
                &SU2FusionRule,
                FusionTreeKey::new(
                    [su2(1); 2],
                    su2(0),
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                ),
            );
            assert_checked_tree_matches_infallible(
                &FibonacciFusionRule,
                FusionTreeKey::new(
                    [SectorId::new(1); 2],
                    SectorId::new(1),
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                ),
            );

            #[cfg(target_pointer_width = "64")]
            {
                type InnerCodec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
                type InnerRule =
                    ProductFusionRule<FermionParityFusionRule, U1FusionRule, InnerCodec>;
                type InnerLayout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
                type Codec = PackedProductCodec<InnerLayout, Su2SectorLayout>;
                type Rule = ProductFusionRule<InnerRule, SU2FusionRule, Codec>;
                let rule = Rule::new(
                    InnerRule::new(FermionParityFusionRule, U1FusionRule),
                    SU2FusionRule,
                );
                let sector = |parity, charge, spin| {
                    Codec::encode(InnerCodec::encode(parity, charge), spin)
                };
                assert_checked_tree_matches_infallible(
                    &rule,
                    FusionTreeKey::new(
                        [sector(z2_odd(), u1(2), su2(1)), sector(z2_odd(), u1(-1), su2(1))],
                        sector(z2_even(), u1(1), su2(0)),
                        [false; 2],
                        [],
                        [MultiplicityIndex::ONE],
                    ),
                );
            }

            assert_checked_tree_matches_infallible(
                &CheckedTreeProbe::default(),
                FusionTreeKey::new(
                    [SectorId::new(0); 2],
                    SectorId::new(0),
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                ),
            );
        }

        #[test]
        fn tree_pair_validation_requires_exact_coupled_sector() {
            // What: pair compatibility compares the same coupled sector on both
            // sides, including the canonical rank-zero vacuum.
            let empty_pair = FusionTreePairKey::pair(
                FusionTreeKey::new([], u1(0), [], [], []),
                FusionTreeKey::new([], u1(0), [], [], []),
            );
            empty_pair.validate_for_rule(&U1FusionRule).unwrap();
            FusionTreePairKey::pair(
                FusionTreeKey::new([], u1(0), [], [], []),
                FusionTreeKey::new([u1(0)], u1(0), [false], [], []),
            )
            .validate_for_rule(&U1FusionRule)
            .unwrap();
            assert_eq!(
                FusionTreePairKey::pair(
                    FusionTreeKey::new([], u1(0), [], [], []),
                    FusionTreeKey::new([u1(1)], u1(1), [false], [], []),
                )
                .validate_for_rule(&U1FusionRule)
                .unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "fusion tree pair requires matching coupled sectors",
                }
            );

            let codomain = FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(1), u1(2)], u1(3),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap();
            let same_c_domain = FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(4), u1(-1)], u1(3),
                [true, false],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap();
            FusionTreePairKey::pair(codomain.clone(), same_c_domain)
                .validate_for_rule(&U1FusionRule)
                .unwrap();
            let dual_c_domain = FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(-4), u1(1)], u1(-3),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap();
            assert_eq!(
                FusionTreePairKey::pair(codomain, dual_c_domain)
                    .validate_for_rule(&U1FusionRule)
                    .unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "fusion tree pair requires matching coupled sectors",
                }
            );
        }

        #[test]
        fn unique_proof_hook_rechecks_style_and_prepared_source_rank() {
            // What: downstream callers cannot use a general categorical proof to
            // bypass the Unique-style or prepared source-split contract.
            let vacuum = su2(0);
            let half = su2(1);
            let simple_pair = FusionTreePairKey::pair(
                FusionTreeKey::try_new_for_rule(
                    &SU2FusionRule,
                    [half, half], vacuum,
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                )
                .unwrap(),
                FusionTreeKey::try_new_for_rule(
                    &SU2FusionRule,
                    [], vacuum,
                    [],
                    [],
                    [],
                )
                .unwrap(),
            );
            let simple_structure =
                packed_fixture_structure(2, [(simple_pair, vec![1, 1])]).unwrap();
            let simple_proof =
                LocallyValidatedFusionTreeBlockStructure::try_new(&SU2FusionRule, &simple_structure).unwrap();
            let rank_two_identity =
                PreparedTreePairOperation::prepare_transpose(2, 0, &[0, 1], &[]).unwrap();
            assert_eq!(
                simple_proof
                    .execute_unique_rigid_for_block_index(0, &rank_two_identity)
                    .unwrap_err(),
                CoreError::UnsupportedFusionStyle {
                    expected: FusionStyleKind::Unique,
                    actual: FusionStyleKind::Simple,
                }
            );

            let unique_pair = FusionTreePairKey::pair(
                FusionTreeKey::try_new_for_rule(
                    &U1FusionRule,
                    [u1(1), u1(-1)], u1(0),
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                )
                .unwrap(),
                FusionTreeKey::try_new_for_rule(&U1FusionRule, [], u1(0), [], [], []).unwrap(),
            );
            let unique_structure =
                packed_fixture_structure(2, [(unique_pair, vec![1, 1])]).unwrap();
            let unique_proof =
                LocallyValidatedFusionTreeBlockStructure::try_new(&U1FusionRule, &unique_structure).unwrap();
            let rank_one_identity =
                PreparedTreePairOperation::prepare_transpose(1, 0, &[0], &[]).unwrap();
            assert_eq!(
                unique_proof
                    .execute_unique_rigid_for_block_index(0, &rank_one_identity)
                    .unwrap_err(),
                CoreError::DimensionMismatch {
                    expected: 1,
                    actual: 2,
                }
            );
        }

        #[test]
        fn rule_validation_preserves_builtin_multiplicity_free_keys() {
            // What: validation is observational for representative abelian,
            // non-abelian, anyonic, fermionic, and product fusion trees.
            fn assert_preserved<R: FusionRule>(rule: &R, tree: FusionTreeKey) {
                let before = tree.clone();
                tree.validate_for_rule(rule).unwrap();
                assert_eq!(tree, before);
            }

            assert_preserved(
                &U1FusionRule,
                FusionTreeKey::try_new_for_rule(
                    &U1FusionRule,
                    [u1(1), u1(-2), u1(3)], u1(2),
                    [false; 3],
                    [u1(-1)],
                    [MultiplicityIndex::ONE; 2],
                )
                .unwrap(),
            );
            let half = su2(1);
            let one = su2(2);
            assert_preserved(
                &SU2FusionRule,
                FusionTreeKey::try_new_for_rule(
                    &SU2FusionRule,
                    [half, half, one], one,
                    [false, true, false],
                    [su2(0)],
                    [MultiplicityIndex::ONE; 2],
                )
                .unwrap(),
            );
            assert_preserved(
                &FibonacciFusionRule,
                FusionTreeKey::try_new_for_rule(
                    &FibonacciFusionRule,
                    [SectorId::new(1); 3], SectorId::new(1),
                    [false; 3],
                    [SectorId::new(0)],
                    [MultiplicityIndex::ONE; 2],
                )
                .unwrap(),
            );
            assert_preserved(
                &FermionParityFusionRule,
                FusionTreeKey::try_new_for_rule(
                    &FermionParityFusionRule,
                    [z2_odd(); 3], z2_odd(),
                    [false, true, false],
                    [z2_even()],
                    [MultiplicityIndex::ONE; 2],
                )
                .unwrap(),
            );

            type ProductRule = ProductFusionRule<U1FusionRule, SU2FusionRule>;
            let product = ProductRule::default();
            let left = product.encode_sector(u1(1), half);
            let right = product.encode_sector(u1(-1), half);
            let coupled = product.encode_sector(u1(0), su2(0));
            assert_preserved(
                &product,
                FusionTreeKey::try_new_for_rule(
                    &product,
                    [left, right], coupled,
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                )
                .unwrap(),
            );
        }

        #[test]
        fn public_operation_errors_precede_categorical_source_errors() {
            // What: deterministic argument and capability failures retain their
            // public precedence over a malformed source.
            let tau = SectorId::new(1);
            let vacuum = SectorId::new(0);
            let invalid = FusionTreeKey::new(
                [tau; 3], vacuum,
                [false; 3],
                [vacuum],
                [MultiplicityIndex::ONE; 2],
            );
            assert_eq!(
                multiplicity_free_braid_tree(
                    &FibonacciFusionRule,
                    &invalid,
                    &[0, 1, 2],
                    &[],
                )
                .unwrap_err(),
                CoreError::DimensionMismatch {
                    expected: 3,
                    actual: 0,
                }
            );
            assert!(matches!(
                multiplicity_free_braid_tree(
                    &FibonacciFusionRule,
                    &invalid,
                    &[0, 0, 2],
                    &[0, 1, 2],
                ),
                Err(CoreError::InvalidPermutation { .. })
            ));
            // Not asserted here: `unique_braid_tree`'s own rejection of a Simple
            // provider. That entry point is gone and
            // `multiplicity_free_braid_tree` accepts Simple providers by design.
            // The Unique-only style guard itself still exists on the surviving
            // route and is asserted elsewhere — `execute_unique_rigid` at
            // `fusion_tree.rs`, exercised with a Simple rule below in this file.
            assert!(matches!(
                multiplicity_free_permute_tree(
                    &FibonacciFusionRule,
                    &invalid,
                    &[0, 1, 2],
                ),
                Err(CoreError::UnsupportedBraidingStyle { .. })
            ));
            assert_eq!(
                multiplicity_free_braid_tree_block(
                    &FibonacciFusionRule,
                    std::slice::from_ref(&invalid),
                    &[0, 1, 2],
                    &[],
                )
                .unwrap_err(),
                CoreError::DimensionMismatch {
                    expected: 3,
                    actual: 0,
                }
            );

            let pair = FusionTreePairKey::pair(
                invalid,
                FusionTreeKey::new([], vacuum, [], [], []),
            );
            let wrong_rank = PreparedTreePairOperation::prepare_braid(
                &FibonacciFusionRule,
                2,
                0,
                &[0, 1],
                &[],
                &[0, 1],
                &[],
            )
            .unwrap();
            assert_eq!(
                wrong_rank
                    .execute_multiplicity_free(&FibonacciFusionRule, &pair)
                    .unwrap_err(),
                CoreError::DimensionMismatch {
                    expected: 2,
                    actual: 3,
                }
            );
        }

        #[test]
        fn unique_direct_braid_eligibility_excludes_rank_zero() {
            // What: the canonical empty tree carries the vacuum, but no rank-zero
            // operation is eligible for a direct braid rebuild.
            let empty = FusionTreeKey::try_new_for_rule(
                &Z2FusionRule,
                [],
                Z2FusionRule.vacuum(),
                [],
                [],
                [],
            )
            .unwrap();

            assert_eq!(empty.coupled(), Z2FusionRule.vacuum());
            assert!(!is_unique_direct_braid_source(&Z2FusionRule, &empty));
        }

        #[test]
        fn uncertified_custom_symbols_stay_on_general_artin_path() {
            // What: the public provider trait does not runtime-check coherence, so
            // arbitrary custom F/R data must remain on the default-false Artin
            // boundary instead of claiming TensorKit's certified F=1 shortcut.
            let rule = UncertifiedCustomSymbolsRule;
            assert!(!rule.has_trivial_associator_gauge());
            let tree =
                FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1]).unwrap();

            let terms = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
            assert_eq!(terms.len(), 1);
            let (destination, coefficient) = terms.into_iter().next().unwrap();

            let mut expected_tree = UnhashedFusionTree::from(tree.clone());
            let expected_coefficient =
                apply_unique_artin_braid_at_with_inverse(&rule, &mut expected_tree, 1, false).unwrap();

            assert_eq!(
                (destination, coefficient),
                (expected_tree.freeze(), expected_coefficient)
            );
            assert_eq!(coefficient, 30.0);
        }

        #[test]
        fn complex_unique_prepared_steps_keep_asymmetric_inverse_orientation() {
            // What: prepared Artin steps retain Complex64 conjugation and reverse
            // R-symbol argument order for an inverse anyonic crossing.
            let rule = ComplexAsymmetricUniqueRule;
            let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, true], [], [1]).unwrap();

            let forward_terms = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
            assert_eq!(forward_terms.len(), 1);
            let forward = forward_terms.into_iter().next().unwrap();
            let inverse_terms = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[1, 0]).unwrap();
            assert_eq!(inverse_terms.len(), 1);
            let inverse = inverse_terms.into_iter().next().unwrap();
            let expected_forward =
                Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_3);
            let expected_inverse =
                Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_6);

            assert_eq!(forward.0, inverse.0);
            assert!((forward.1 - expected_forward).norm() < 1.0e-12);
            assert!((inverse.1 - expected_inverse).norm() < 1.0e-12);
        }

        #[test]
        fn prepared_tree_pair_operation_size_excludes_a_second_step_arena() {
            // What: the expert plan stores one inline Artin lowering, not parallel
            // Artin and inversion arrays for mutually exclusive execution paths.
            assert!(std::mem::size_of::<PreparedTreePairOperation<'static>>() <= 600);
        }

        #[test]
        fn unique_braid_tree_rejects_invalid_permutation_and_level_count() {
            let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();

            let err =
                multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 1], &[0, 1])
                    .unwrap_err();
            assert_eq!(
                err,
                CoreError::InvalidPermutation {
                    permutation: vec![1, 1],
                    rank: 2,
                }
            );

            let err =
                multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[0]).unwrap_err();
            assert_eq!(
                err,
                CoreError::DimensionMismatch {
                    expected: 2,
                    actual: 1,
                }
            );
        }

        #[test]
        fn unique_permute_tree_requires_symmetric_braiding() {
            let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();

            let err = unique_permute_tree(&AsymmetricAnyonicRule, &tree, &[1, 0]).unwrap_err();

            assert_eq!(
                err,
                CoreError::UnsupportedBraidingStyle {
                    expected: "symmetric braiding",
                    actual: BraidingStyleKind::Anyonic,
                }
            );
        }

        // --- Stage 0 spike: complex-`Scalar` fusion rule through the recoupling
        // engine. tenet-core has so far only ever instantiated `Scalar = f64`
        // providers; Fibonacci anyons need `Scalar = Complex64`. This probe rule
        // is *not* a physical anyon model (its F-symbol is a constant 1, so it
        // makes no pentagon claim) — it exists purely to prove that
        // `multiplicity_free_braid_tree` / `FusionTermAccumulator` compile and
        // run correctly when `Scalar: num_complex::Complex64` (Add/Mul/Clone from
        // `num_complex`, plus a genuinely complex `CategoricalScalar::conj`).
        #[derive(Clone, Copy, Debug)]
        struct ComplexScalarProbeRule;

        impl FusionRule for ComplexScalarProbeRule {
            fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
            fn fusion_style(&self) -> FusionStyleKind {
                FusionStyleKind::Simple
            }

            fn braiding_style(&self) -> BraidingStyleKind {
                BraidingStyleKind::Anyonic
            }

            fn vacuum(&self) -> SectorId {
                SectorId::new(0)
            }

            fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
                match (left.id(), right.id()) {
                    (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                    // Fibonacci-shaped multi-channel fusion: x⊗x = {vacuum, x}.
                    // This is what forces `FusionStyleKind::Simple` (not `Unique`)
                    // and exercises the multi-term loop in the braid engine.
                    (1, 1) => smallvec![SectorId::new(0), SectorId::new(1)],
                    _ => SectorVec::new(),
                }
            }
        }

        impl MultiplicityFreeFusionRule for ComplexScalarProbeRule {}

        const PROBE_ANGLE_ALPHA: f64 = std::f64::consts::FRAC_PI_3;

        const PROBE_ANGLE_BETA: f64 = 2.0 * std::f64::consts::FRAC_PI_3;

        impl MultiplicityFreeFusionSymbols for ComplexScalarProbeRule {
            type Scalar = num_complex::Complex64;

            // Trivial associator (1 on every allowed channel, since
            // `fusion_channels` already zeroes out disallowed ones via the
            // engine's `nsymbol` gate): this probe only needs to exercise the
            // complex-scalar plumbing, not satisfy the pentagon identity.
            fn f_symbol_scalar(
                &self,
                _left: SectorId,
                _middle: SectorId,
                _right: SectorId,
                _coupled: SectorId,
                _left_coupled: SectorId,
                _right_coupled: SectorId,
            ) -> Self::Scalar {
                num_complex::Complex64::new(1.0, 0.0)
            }

            // The one place a genuine complex phase enters: R^{xx}_vacuum = e^{iα},
            // R^{xx}_x = e^{iβ}, distinct angles so the two channels are
            // distinguishable in the assertions below.
            fn r_symbol_scalar(
                &self,
                left: SectorId,
                right: SectorId,
                coupled: SectorId,
            ) -> Self::Scalar {
                if self.nsymbol(left, right, coupled) == 0 {
                    return num_complex::Complex64::new(0.0, 0.0);
                }
                if left.id() == 0 || right.id() == 0 {
                    return num_complex::Complex64::new(1.0, 0.0);
                }
                if coupled.id() == 0 {
                    num_complex::Complex64::from_polar(1.0, PROBE_ANGLE_ALPHA)
                } else {
                    num_complex::Complex64::from_polar(1.0, PROBE_ANGLE_BETA)
                }
            }
        }

        #[test]
        fn complex_scalar_r_symbol_and_conjugate_inverse_braid_stage0_spike() {
            let rule = ComplexScalarProbeRule;
            for coupled in [0usize, 1usize] {
                let tree = FusionTreeKey::try_from_sector_ids([1, 1], coupled, [false, false], [], [1]).unwrap();
                let expected = num_complex::Complex64::from_polar(
                    1.0,
                    if coupled == 0 {
                        PROBE_ANGLE_ALPHA
                    } else {
                        PROBE_ANGLE_BETA
                    },
                );

                let forward = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
                assert_eq!(forward.len(), 1);
                assert!((forward[0].1 - expected).norm() < 1.0e-12);

                // Reflected levels select the inverse-artin branch: the
                // coefficient must come back as the complex conjugate, proving
                // `CategoricalScalar::conj` (not just `Clone`/`Mul`) is wired through
                // for a non-real `Scalar`.
                let backward = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[1, 0]).unwrap();
                assert_eq!(backward.len(), 1);
                assert!((backward[0].1 - expected.conj()).norm() < 1.0e-12);
            }
        }

        #[test]
        fn complex_scalar_braid_tree_expands_multichannel_loop_stage0_spike() {
            // Rank-3 tree with an index>0 swap: exercises the `fusion_channels(a,
            // d)` loop branch of `multiplicity_free_artin_braid_at_with_inverse`
            // (f_symbol_scalar * r_symbol_scalar * conj composition) —
            // this is the part of the engine Fibonacci's Simple-fusion braid
            // actually needs (the rank-2 spike above only reaches the
            // single-r-symbol `index == 0` branch).
            let rule = ComplexScalarProbeRule;
            let tree =
                FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1]).unwrap();

            let braided = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();

            // Hand-derived from the engine formula in the index>0 branch with
            // this rule's constant F=1: coefficient(c') = R(c,d,e) * conj(R(a,d,c')).
            // Here a=b=c=d=e=x, so R(c,d,e) = e^{iβ}; c'=vacuum -> R(a,d,c')=e^{iα},
            // c'=x -> e^{iβ}.
            assert_eq!(braided.len(), 2);
            let coeff_for = |innerline: usize| {
                braided
                    .iter()
                    .find(|(t, _)| t.innerlines() == [SectorId::new(innerline)])
                    .unwrap()
                    .1
            };
            let expected_vacuum_channel =
                num_complex::Complex64::from_polar(1.0, PROBE_ANGLE_BETA - PROBE_ANGLE_ALPHA);
            let expected_x_channel = num_complex::Complex64::new(1.0, 0.0);
            assert!((coeff_for(0) - expected_vacuum_channel).norm() < 1.0e-12);
            assert!((coeff_for(1) - expected_x_channel).norm() < 1.0e-12);
        }

        #[test]
        fn fibonacci_multi_fmove_low_ranks_keep_their_existing_contracts() {
            let rule = FibonacciFAdmissibilityProbe::new();
            let vacuum = SectorId::new(0);
            let tau = SectorId::new(1);
            let empty = FusionTreeKey::new([], vacuum, [], [], []);
            let rank_one = FusionTreeKey::new([tau], tau, [false], [], []);
            let rank_two =
                FusionTreeKey::new([tau, tau], vacuum, [false, false], [], [MultiplicityIndex::ONE]);

            // What: forward rank 0/1/2 and inverse rank 0/1 retain their prior
            // results and do not enter an F-symbol provider.
            assert_eq!(
                multiplicity_free_multi_fmove_tree(&rule, &empty),
                multiplicity_free_multi_fmove_tree(&FibonacciFusionRule, &empty)
            );
            assert_eq!(
                multiplicity_free_multi_fmove_tree(&rule, &rank_one),
                multiplicity_free_multi_fmove_tree(&FibonacciFusionRule, &rank_one)
            );
            assert_eq!(
                multiplicity_free_multi_fmove_tree(&rule, &rank_two),
                multiplicity_free_multi_fmove_tree(&FibonacciFusionRule, &rank_two)
            );
            assert_eq!(
                multiplicity_free_multi_fmove_inv_tree(&rule, tau, tau, &empty, false),
                multiplicity_free_multi_fmove_inv_tree(
                    &FibonacciFusionRule,
                    tau,
                    tau,
                    &empty,
                    false,
                )
            );
            assert_eq!(
                multiplicity_free_multi_fmove_inv_tree(&rule, tau, vacuum, &rank_one, false),
                multiplicity_free_multi_fmove_inv_tree(
                    &FibonacciFusionRule,
                    tau,
                    vacuum,
                    &rank_one,
                    false,
                )
            );
            assert!(rule.take_calls().is_empty());

            // What: inverse rank 2 still performs its one associator step, with
            // unchanged data and an admissible provider call.
            assert_eq!(
                multiplicity_free_multi_fmove_inv_tree(&rule, tau, tau, &rank_two, false),
                multiplicity_free_multi_fmove_inv_tree(
                    &FibonacciFusionRule,
                    tau,
                    tau,
                    &rank_two,
                    false,
                )
            );
            let calls = rule.take_calls();
            assert!(!calls.is_empty());
            assert_fibonacci_f_calls_are_admissible(&calls);
        }

        struct UniqueFAdmissibilityProbe {
            calls: std::sync::Mutex<Vec<[SectorId; 6]>>,
        }

        impl UniqueFAdmissibilityProbe {
            fn new() -> Self {
                Self {
                    calls: std::sync::Mutex::new(Vec::new()),
                }
            }

            fn take_calls(&self) -> Vec<[SectorId; 6]> {
                std::mem::take(
                    &mut *self
                        .calls
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                )
            }
        }

        impl FusionRule for UniqueFAdmissibilityProbe {
            fn rule_identity(&self) -> RuleIdentity {
                RuleIdentity::of_type::<Self>()
            }

            fn fusion_style(&self) -> FusionStyleKind {
                Z2FusionRule.fusion_style()
            }

            fn braiding_style(&self) -> BraidingStyleKind {
                Z2FusionRule.braiding_style()
            }

            fn vacuum(&self) -> SectorId {
                Z2FusionRule.vacuum()
            }

            fn supports_unitary_braid_dagger(&self) -> bool {
                Z2FusionRule.supports_unitary_braid_dagger()
            }

            fn dual(&self, sector: SectorId) -> SectorId {
                Z2FusionRule.dual(sector)
            }

            fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
                Z2FusionRule.fusion_channels(left, right)
            }

            fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
                Z2FusionRule.nsymbol(left, right, coupled)
            }
        }

        impl MultiplicityFreeFusionRule for UniqueFAdmissibilityProbe {}

        impl MultiplicityFreeFusionSymbols for UniqueFAdmissibilityProbe {
            type Scalar = f64;

            fn f_symbol_scalar(
                &self,
                left: SectorId,
                middle: SectorId,
                right: SectorId,
                coupled: SectorId,
                left_coupled: SectorId,
                right_coupled: SectorId,
            ) -> Self::Scalar {
                self.calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push([
                        left,
                        middle,
                        right,
                        coupled,
                        left_coupled,
                        right_coupled,
                    ]);
                let admissible = Z2FusionRule.nsymbol(left, middle, left_coupled) != 0
                    && Z2FusionRule.nsymbol(left_coupled, right, coupled) != 0
                    && Z2FusionRule.nsymbol(middle, right, right_coupled) != 0
                    && Z2FusionRule.nsymbol(left, right_coupled, coupled) != 0;
                if admissible {
                    Z2FusionRule.f_symbol_scalar(
                        left,
                        middle,
                        right,
                        coupled,
                        left_coupled,
                        right_coupled,
                    )
                } else {
                    997.0
                }
            }

            fn r_symbol_scalar(
                &self,
                left: SectorId,
                right: SectorId,
                coupled: SectorId,
            ) -> Self::Scalar {
                Z2FusionRule.r_symbol_scalar(left, right, coupled)
            }
        }

        impl MultiplicityFreeRigidSymbols for UniqueFAdmissibilityProbe {
            fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
                Z2FusionRule.dim_scalar(sector)
            }

            fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
                Z2FusionRule.inv_dim_scalar(sector)
            }

            fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
                Z2FusionRule.sqrt_dim_scalar(sector)
            }

            fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
                Z2FusionRule.inv_sqrt_dim_scalar(sector)
            }

            fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
                Z2FusionRule.twist_scalar(sector)
            }

            fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
                Z2FusionRule.frobenius_schur_phase_scalar(sector)
            }
        }

        #[test]
        fn unique_multi_fmove_callers_preserve_admissible_z2_results() {
            let rule = UniqueFAdmissibilityProbe::new();
            let odd = z2_odd();
            let even = z2_even();
            let long = FusionTreeKey::new(
                [odd; 4],
                even,
                [false; 4],
                [even, odd],
                [MultiplicityIndex::ONE; 3],
            );
            let short = FusionTreeKey::new(
                [odd; 3],
                odd,
                [false; 3],
                [even],
                [MultiplicityIndex::ONE; 2],
            );

            // What: the Unique-fusion wrappers that share the associator boundary
            // retain exact output trees, coefficients, and conjugation direction.
            assert_eq!(
                unique_rigid_multi_fmove_tree(&rule, &long),
                unique_rigid_multi_fmove_tree(&Z2FusionRule, &long)
            );
            let calls = rule.take_calls();
            assert!(!calls.is_empty());
            for &[left, middle, right, coupled, left_coupled, right_coupled] in &calls {
                assert_ne!(Z2FusionRule.nsymbol(left, middle, left_coupled), 0);
                assert_ne!(Z2FusionRule.nsymbol(left_coupled, right, coupled), 0);
                assert_ne!(Z2FusionRule.nsymbol(middle, right, right_coupled), 0);
                assert_ne!(Z2FusionRule.nsymbol(left, right_coupled, coupled), 0);
            }

            assert_eq!(
                unique_rigid_multi_fmove_inv_tree(&rule, odd, even, &short, false),
                unique_rigid_multi_fmove_inv_tree(&Z2FusionRule, odd, even, &short, false)
            );
            let calls = rule.take_calls();
            assert!(!calls.is_empty());
            for &[left, middle, right, coupled, left_coupled, right_coupled] in &calls {
                assert_ne!(Z2FusionRule.nsymbol(left, middle, left_coupled), 0);
                assert_ne!(Z2FusionRule.nsymbol(left_coupled, right, coupled), 0);
                assert_ne!(Z2FusionRule.nsymbol(middle, right, right_coupled), 0);
                assert_ne!(Z2FusionRule.nsymbol(left, right_coupled, coupled), 0);
            }
        }

        #[test]
        fn fibonacci_braid_tree_end_to_end_matches_hand_derived_coefficients() {
            // End-to-end: Simple fusion + Anyonic braiding + complex Scalar
            // through `multiplicity_free_braid_tree` on a rank-3 tree, with
            // coefficients hand-derived from the engine's own
            // R(c,d,e) * conj(F(d,a,b,e,c',c) * R(a,d,c')) formula
            // (`multiplicity_free_artin_braid_at_with_inverse`, index > 0
            // branch) substituting TensorKitSectors' F/R values directly.
            let rule = FibonacciFusionRule;
            let tree =
                FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1]).unwrap();

            let braided = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
            assert_eq!(braided.len(), 2);

            let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
            let cispi = |x: f64| Complex64::from_polar(1.0, std::f64::consts::PI * x);
            let coeff_for = |innerline: usize| {
                braided
                    .iter()
                    .find(|(t, _)| t.innerlines() == [SectorId::new(innerline)])
                    .unwrap()
                    .1
            };

            let expected_vacuum_channel = Complex64::new(1.0 / phi.sqrt(), 0.0) * cispi(3.0 / 5.0);
            let expected_tau_channel = Complex64::new(-1.0 / phi, 0.0);
            assert!((coeff_for(0) - expected_vacuum_channel).norm() < 1.0e-12);
            assert!((coeff_for(1) - expected_tau_channel).norm() < 1.0e-12);
        }

        #[test]
        fn fibonacci_elementary_artin_rows_match_tensorkit_coefficients() {
            // What: the private elementary Artin operation preserves TensorKit's
            // destination order and the independently substituted F/R coefficients
            // for both crossing orientations.
            let rule = FibonacciFusionRule;
            let tree =
                FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1]).unwrap();
            let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
            let cispi = |x: f64| Complex64::from_polar(1.0, std::f64::consts::PI * x);

            let forward =
                multiplicity_free_artin_braid_at_with_inverse(&rule, &tree, 1, false).unwrap();
            let inverse =
                multiplicity_free_artin_braid_at_with_inverse(&rule, &tree, 1, true).unwrap();

            assert_eq!(
                forward
                    .iter()
                    .map(|(key, _)| key.innerlines()[0].id())
                    .collect::<Vec<_>>(),
                vec![0, 1]
            );
            assert_eq!(
                inverse
                    .iter()
                    .map(|(key, _)| key.innerlines()[0].id())
                    .collect::<Vec<_>>(),
                vec![0, 1]
            );
            let forward_expected = [
                Complex64::new(1.0 / phi.sqrt(), 0.0) * cispi(3.0 / 5.0),
                Complex64::new(-1.0 / phi, 0.0),
            ];
            let inverse_expected = [
                Complex64::new(1.0 / phi.sqrt(), 0.0) * cispi(-3.0 / 5.0),
                Complex64::new(-1.0 / phi, 0.0),
            ];
            for ((_, actual), expected) in forward.iter().zip(forward_expected) {
                assert!((*actual - expected).norm() < 1.0e-12);
            }
            for ((_, actual), expected) in inverse.iter().zip(inverse_expected) {
                assert!((*actual - expected).norm() < 1.0e-12);
            }
        }

        #[test]
        fn linearize_tree_pair_permutation_matches_tensorkit_zero_based_formula() {
            assert_eq!(
                linearize_tree_pair_permutation(&[0, 1], &[2, 3], 2, 2).unwrap(),
                vec![0, 1, 2, 3]
            );
            assert_eq!(
                linearize_tree_pair_permutation(&[3, 0], &[1, 2], 2, 2).unwrap(),
                vec![2, 0, 3, 1]
            );

            let err = linearize_tree_pair_permutation(&[0, 0], &[1, 2], 2, 2).unwrap_err();
            assert_eq!(
                err,
                CoreError::InvalidPermutation {
                    permutation: vec![0, 0, 1, 2],
                    rank: 4,
                }
            );
        }

        #[test]
        fn identity_braid_tree_pair_skips_symbols_and_repartition() {
            // What: an exact same-split braid is source => one for both Unique and
            // multiplicity-free entry points, without consulting F/R/bend data.
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

            // Not covered here: the deleted execute_unique_pivotal route's own
            // identity-braid shortcut (`unique_braid_tree_pair`). Its no-op
            // symbol-free contract is already covered below by
            // multiplicity_free_braid_tree_pair and its block variant, which are
            // the live entry points for this behavior.
            let multiplicity_free = multiplicity_free_braid_tree_pair(
                &IdentitySymbolPanicRule,
                &source,
                &[0],
                &[1],
                &[19],
                &[3],
            )
            .unwrap();
            assert_eq!(multiplicity_free, vec![(source.clone(), 1.0)]);

            let block = multiplicity_free_braid_tree_pair_block(
                &IdentitySymbolPanicRule,
                std::slice::from_ref(&source),
                &[0],
                &[1],
                &[19],
                &[3],
            )
            .unwrap();
            assert_eq!(block, vec![vec![(source, 1.0)]]);
        }

        #[test]
        fn braid_tree_block_matches_per_source_su2_rows() {
            use std::collections::BTreeMap;

            let rule = SU2FusionRule;
            let sources = [
                FusionTreeKey::try_from_sector_ids(
                    [1, 1, 1], 1,
                    [false, false, false],
                    [0],
                    [1, 1],
                ).unwrap(),
                FusionTreeKey::try_from_sector_ids(
                    [1, 1, 1], 1,
                    [false, false, false],
                    [2],
                    [1, 1],
                ).unwrap(),
            ];

            let block =
                multiplicity_free_braid_tree_block(&rule, &sources, &[0, 2, 1], &[0, 1, 2]).unwrap();
            assert_eq!(block.len(), sources.len());
            for (source, block_rows) in sources.iter().zip(&block) {
                let per_source =
                    multiplicity_free_braid_tree(&rule, source, &[0, 2, 1], &[0, 1, 2]).unwrap();
                let collect = |rows: &[(FusionTreeKey, f64)]| {
                    let mut coefficients = BTreeMap::<FusionTreeKey, f64>::new();
                    for (key, coefficient) in rows {
                        *coefficients.entry(key.clone()).or_default() += coefficient;
                    }
                    coefficients
                };
                let expected = collect(&per_source);
                let actual = collect(block_rows);

                // What: the whole-block walk preserves every destination tree and
                // coefficient produced by independent SU(2) source transforms.
                assert_eq!(
                    expected.keys().collect::<Vec<_>>(),
                    actual.keys().collect::<Vec<_>>()
                );
                for (key, expected_coefficient) in expected {
                    let actual_coefficient = actual[&key];
                    assert!(
                        (expected_coefficient - actual_coefficient).abs()
                            <= 1.0e-12 * (1.0 + expected_coefficient.abs())
                    );
                }
            }
        }

        #[test]
        fn tree_block_admits_every_source_before_group_check_and_symbols() {
            let base =
                FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();
            let valid_mixed = [
                FusionTreeKey::try_from_sector_ids([1, 4], 5, [false, false], [], [1]).unwrap(),
                FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, true], [], [1]).unwrap(),
            ];
            let expected = CoreError::MalformedFusionTree {
                message: "fusion-tree keys must share one group",
            };

            for other in valid_mixed {
                let sources = [base.clone(), other];
                // What: admitted mixed groups fail before the panic-on-symbol
                // fixture can evaluate F or R data.
                assert_eq!(
                    multiplicity_free_braid_tree_block(
                        &IdentitySymbolPanicRule,
                        &sources,
                        &[0, 1],
                        &[0, 1],
                    )
                    .unwrap_err(),
                    expected
                );
                assert_eq!(
                    multiplicity_free_permute_tree_block(
                        &IdentitySymbolPanicRule,
                        &sources,
                        &[1, 0],
                    )
                    .unwrap_err(),
                    expected
                );
            }

            let invalid_vertex = FusionTreeKey::new(
                base.uncoupled().iter().copied(),
                base.coupled(),
                base.is_dual().iter().copied(),
                base.innerlines().iter().copied(),
                [MultiplicityIndex::new(2).unwrap()],
            );
            let invalid_sources = [base, invalid_vertex];
            let expected_vertex = CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            };
            // What: provider-owned multiplicity admission checks every source
            // before a block shortcut can discard explicit vertex identity.
            assert_eq!(
                multiplicity_free_braid_tree_block(
                    &IdentitySymbolPanicRule,
                    &invalid_sources,
                    &[0, 1],
                    &[0, 1],
                )
                .unwrap_err(),
                expected_vertex
            );
        }

        #[test]
        fn tree_block_empty_and_identity_contracts_preserve_source_order() {
            let empty: &[FusionTreeKey] = &[];
            assert_eq!(
                multiplicity_free_braid_tree_block(
                    &IdentitySymbolPanicRule,
                    empty,
                    &[],
                    &[],
                )
                .unwrap(),
                Vec::<Vec<(FusionTreeKey, f64)>>::new()
            );
            assert_eq!(
                multiplicity_free_permute_tree_block(&IdentitySymbolPanicRule, empty, &[]).unwrap(),
                Vec::<Vec<(FusionTreeKey, f64)>>::new()
            );

            let half = su2(1);
            let sources = [
                FusionTreeKey::try_from_sector_ids(
                    [half.id(), half.id()], su2(0).id(),
                    [false; 2],
                    [],
                    [1],
                ).unwrap(),
                FusionTreeKey::try_from_sector_ids(
                    [half.id(), half.id()], su2(2).id(),
                    [false; 2],
                    [],
                    [1],
                ).unwrap(),
            ];
            let expected = sources
                .iter()
                .cloned()
                .map(|source| vec![(source, 1.0)])
                .collect::<Vec<_>>();
            // What: distinct coupled labels remain one external-sector group and
            // the symbol-free identity path returns exact rows in source order.
            assert_eq!(
                multiplicity_free_braid_tree_block(
                    &SU2FusionRule,
                    &sources,
                    &[0, 1],
                    &[13, 5],
                )
                .unwrap(),
                expected
            );
            assert_eq!(
                multiplicity_free_permute_tree_block(
                    &SU2FusionRule,
                    &sources,
                    &[0, 1],
                )
                .unwrap(),
                expected
            );
        }

        #[test]
        fn tree_pair_block_apis_reject_invalid_later_vertex_before_symbols() {
            let codomain =
                FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();
            let domain = FusionTreeKey::try_from_sector_ids([3], 3, [false], [], []).unwrap();
            let base = FusionTreePairKey::pair(codomain.clone(), domain.clone());
            let invalid_codomain = FusionTreeKey::new(
                codomain.uncoupled().iter().copied(),
                codomain.coupled(),
                codomain.is_dual().iter().copied(),
                codomain.innerlines().iter().copied(),
                [MultiplicityIndex::new(2).unwrap()],
            );
            let invalid = FusionTreePairKey::pair(invalid_codomain, domain);

            // What: every source pair is categorically admitted before block
            // identity shortcuts or symbol evaluation.
            assert_mixed_tree_pair_block_group_is_rejected(
                &IdentitySymbolPanicRule,
                &[base, invalid],
                CoreError::MalformedFusionTree {
                    message: "fusion tree vertex label exceeds its fusion multiplicity",
                },
            );
        }

        #[test]
        fn split_only_su2_braid_matches_legacy_composition_in_both_directions() {
            // What: SU(2) 2|2 -> 3|1 and 2|2 -> 1|3 retain the exact tree keys,
            // dual flags, and bend coefficients of the old all-codomain route.
            let source = FusionTreePairKey::try_pair_from_sector_ids(
                [1, 2],
                [2, 1], 1,
                [false, true],
                [true, false],
                [],
                [],
                [1],
                [1],
            ).unwrap();

            for (codomain_axes, domain_axes, target_rank) in
                [(&[0, 1, 3][..], &[2][..], 3), (&[0][..], &[2, 3, 1][..], 1)]
            {
                let actual = multiplicity_free_braid_tree_pair(
                    &SU2FusionRule,
                    &source,
                    codomain_axes,
                    domain_axes,
                    &[0, 1],
                    &[2, 3],
                )
                .unwrap();
                let expected =
                    legacy_split_only_tree_pair_route(&SU2FusionRule, &source, target_rank).unwrap();
                assert_eq!(actual.len(), expected.len());
                for ((actual_key, actual_coefficient), (expected_key, expected_coefficient)) in
                    actual.iter().zip(&expected)
                {
                    assert_eq!(actual_key, expected_key);
                    assert!((actual_coefficient - expected_coefficient).abs() < 1.0e-12);
                }
            }
        }

        #[test]
        fn split_only_tree_pair_braid_handles_empty_split_boundaries() {
            // What: the two extreme repartitions 0|2 -> 2|0 and 2|0 -> 0|2
            // preserve the same keys, dual flags, and coefficients as the direct
            // primitive, including an empty codomain or domain tree.
            let all_domain = FusionTreePairKey::try_pair_from_sector_ids(
                [],
                [1, 1], 0,
                [],
                [false, true],
                [],
                [],
                [],
                [1],
            ).unwrap();
            let to_codomain = multiplicity_free_braid_tree_pair(
                &SU2FusionRule,
                &all_domain,
                &[1, 0],
                &[],
                &[],
                &[0, 1],
            )
            .unwrap();
            let expected_codomain =
                multiplicity_free_repartition_tree_pair(&SU2FusionRule, &all_domain, 2).unwrap();
            assert_eq!(to_codomain, expected_codomain);

            let all_codomain = &to_codomain[0].0;
            let to_domain = multiplicity_free_braid_tree_pair(
                &SU2FusionRule,
                all_codomain,
                &[],
                &[1, 0],
                &[0, 1],
                &[],
            )
            .unwrap();
            let expected_domain =
                multiplicity_free_repartition_tree_pair(&SU2FusionRule, all_codomain, 0).unwrap();
            assert_eq!(to_domain, expected_domain);
        }

        #[test]
        fn nonidentity_tree_pair_braid_does_not_enter_split_only_path() {
            // What: changing the split does not suppress a real external-leg
            // permutation, and malformed axis maps still fail validation.
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
            multiplicity_free_braid_tree_pair(
                &rule,
                &source,
                &[2, 0],
                &[1],
                &[0],
                &[1, 2],
            )
            .unwrap();
            assert!(rule.r_calls.load(std::sync::atomic::Ordering::Relaxed) > 0);

            for (codomain_axes, domain_axes) in
                [(&[0, 0][..], &[1][..]), (&[0, 3][..], &[1][..]), (&[0][..], &[1][..])]
            {
                assert!(multiplicity_free_braid_tree_pair(
                    &rule,
                    &source,
                    codomain_axes,
                    domain_axes,
                    &[0],
                    &[1, 2],
                )
                .is_err());
            }
        }

        #[test]
        fn identity_braid_tree_pair_validates_levels_before_symbol_free_return() {
            // What: malformed levels remain errors even when the axis map is the
            // exact current split.
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

            assert!(multiplicity_free_braid_tree_pair(
                &IdentitySymbolPanicRule,
                &source,
                &[0],
                &[1],
                &[],
                &[3],
            )
            .is_err());
            assert!(multiplicity_free_braid_tree_pair(
                &IdentitySymbolPanicRule,
                &source,
                &[0],
                &[1],
                &[19],
                &[],
            )
            .is_err());
            assert!(multiplicity_free_braid_tree_pair(
                &IdentitySymbolPanicRule,
                &source,
                &[0],
                &[0],
                &[19],
                &[3],
            )
            .is_err());
        }

        #[test]
        fn unique_identity_tree_operations_reject_simple_fusion_rules() {
            // What: identity axes do not let a Simple rule enter an API whose
            // contract requires Unique fusion.
            //
            // Not covered here: the deleted execute_unique_pivotal route's own
            // identity-axis rejection of Simple providers. That contract no
            // longer exists — the surviving multiplicity_free_braid_tree accepts
            // Simple providers by design, so there is nothing left to assert for
            // the braid side of this test.
            let tree = FusionTreeKey::try_from_sector_ids([1], 1, [false], [], []).unwrap();
            let expected = CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: FusionStyleKind::Simple,
            };

            assert_eq!(
                unique_permute_tree(&SU2FusionRule, &tree, &[0]).unwrap_err(),
                expected
            );
        }

        #[test]
        fn same_split_transpose_of_dual_tree_pair_skips_bend_symbols() {
            // What: a real codomain/domain tree pair at its current 2|1 split is
            // source => one without consulting bend/fold data.
            let source = FusionTreePairKey::try_pair_from_sector_ids(
                [1, 0],
                [1], 1,
                [false, true],
                [true],
                [],
                [],
                [1],
                [],
            ).unwrap();

            assert_eq!(
                multiplicity_free_transpose_tree_pair(&IdentitySymbolPanicRule, &source, &[0, 1], &[2])
                    .unwrap(),
                vec![(source, 1.0)]
            );
        }

        #[test]
        fn unique_repartition_tree_pair_moves_domain_to_reversed_dual_codomain() {
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

            let terms = multiplicity_free_repartition_tree_pair(&Z2FusionRule, &source, 3).unwrap();
            assert_eq!(terms.len(), 1);
            let (all_out, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, 1.0);
            assert_eq!(
                all_out.codomain_uncoupled(),
                &[SectorId::new(1), SectorId::new(1), SectorId::new(0)]
            );
            assert_eq!(all_out.codomain_is_dual(), &[false, false, true]);
            assert_eq!(all_out.codomain_innerlines(), &[SectorId::new(0)]);
            assert_eq!(
                all_out.codomain_vertices(),
                &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
            );
            assert!(all_out.domain_uncoupled().is_empty());
            assert_eq!(all_out.domain_tree().coupled(), SectorId::new(0));
        }

        #[test]
        fn unique_braid_tree_pair_matches_single_tree_when_domain_is_empty() {
            let source = FusionTreePairKey::pair(
                FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, true], [], [1]).unwrap(),
                FusionTreeKey::new(
                    Vec::<SectorId>::new(),
                    FermionParityFusionRule.vacuum(),
                    Vec::<bool>::new(),
                    Vec::<SectorId>::new(),
                    Vec::<MultiplicityIndex>::new(),
                ),
            );

            let terms = multiplicity_free_braid_tree_pair(
                &FermionParityFusionRule,
                &source,
                &[1, 0],
                &[],
                &[0, 1],
                &[],
            )
            .unwrap();
            assert_eq!(terms.len(), 1);
            let (braided, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, -1.0);
            assert_eq!(
                braided.codomain_uncoupled(),
                &[SectorId::new(1), SectorId::new(1)]
            );
            assert_eq!(braided.codomain_is_dual(), &[true, false]);
            assert!(braided.domain_uncoupled().is_empty());
            assert_eq!(
                braided.domain_tree().coupled(),
                FermionParityFusionRule.vacuum()
            );
        }

        #[test]
        fn unique_permute_tree_pair_handles_domain_only_swap() {
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

            let terms =
                multiplicity_free_permute_tree_pair(&Z2FusionRule, &source, &[0], &[2, 1]).unwrap();
            assert_eq!(terms.len(), 1);
            let (permuted, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, 1.0);
            assert_eq!(permuted.codomain_uncoupled(), &[SectorId::new(1)]);
            assert_eq!(
                permuted.domain_uncoupled(),
                &[SectorId::new(1), SectorId::new(0)]
            );
            assert_eq!(permuted.domain_is_dual(), &[true, false]);
            assert_eq!(permuted.domain_vertices(), &[MultiplicityIndex::ONE]);
        }

        #[test]
        fn unique_permute_tree_pair_includes_codomain_domain_crossing() {
            let source = FusionTreePairKey::try_pair_from_sector_ids(
                [1],
                [1], 1,
                [false],
                [true],
                [],
                [],
                [],
                [],
            ).unwrap();

            let terms =
                multiplicity_free_permute_tree_pair(&FermionParityFusionRule, &source, &[1], &[0])
                    .unwrap();
            assert_eq!(terms.len(), 1);
            let (permuted, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, -1.0);
            assert_eq!(permuted.codomain_uncoupled(), &[SectorId::new(1)]);
            assert_eq!(permuted.codomain_is_dual(), &[false]);
            assert_eq!(permuted.domain_uncoupled(), &[SectorId::new(1)]);
            assert_eq!(permuted.domain_is_dual(), &[true]);
        }

        #[test]
        fn prepared_fermionic_domain_crossing_is_exactly_negative_one() {
            // What: an actual odd fZ2 domain leg crossing an odd codomain leg
            // retains the exact TensorKit fermionic phase through prepared replay.
            let source = FusionTreePairKey::try_pair_from_sector_ids(
                [1],
                [1], 1,
                [false],
                [true],
                [],
                [],
                [],
                [],
            ).unwrap();
            let prepared = PreparedTreePairOperation::prepare_permute(
                &FermionParityFusionRule,
                1,
                1,
                &[1],
                &[0],
            )
            .unwrap();

            let (destination, coefficient) = prepared
                .execute_unique_rigid(&FermionParityFusionRule, &source)
                .unwrap();

            assert_eq!(coefficient, -1.0);
            assert_eq!(destination.codomain_uncoupled(), &[SectorId::new(1)]);
            assert_eq!(destination.domain_uncoupled(), &[SectorId::new(1)]);
            assert_eq!(destination.codomain_is_dual(), &[false]);
            assert_eq!(destination.domain_is_dual(), &[true]);
        }

        #[test]
        fn prepared_permute_reverses_domain_levels_before_artin_lowering() {
            // What: TensorKit linearizes incoming legs in reverse order, so a
            // domain-only swap is prepared as the inverse Artin generator.
            let prepared =
                PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 2, &[0], &[2, 1])
                    .unwrap();
            assert_eq!(
                prepared
                    .plan
                    .artin_steps()
                    .expect("domain-only swap must prepare a braid")
                    .collect::<Vec<_>>(),
                &[PreparedArtinStep {
                    index: 1,
                    inverse: true,
                }]
            );
        }

        #[test]
        fn unique_multistep_braid_mutates_one_operation_local_tree() {
            // What: a prepared UniqueFusion schedule has the exact frozen-tree
            // fields and coefficient bits of one-step replay without publishing
            // a frozen intermediate key for every Artin crossing.
            let rule = AsymmetricAnyonicRule;
            let source = FusionTreeKey::try_from_sector_ids(
                [1, 1, 2, 0],
                0,
                [false, true, false, true],
                [2, 0],
                [1, 1, 1],
            )
            .unwrap();
            let permutation = [3, 2, 1, 0];
            let levels = [0, 3, 1, 2];
            let owned = PreparedTreeBraid::new(&permutation, &levels, source.uncoupled().len()).unwrap();

            let mut expected_tree = source.clone();
            let mut expected_coefficient = <AsymmetricAnyonicRule as MultiplicityFreeFusionSymbols>::Scalar::one();
            for step in &owned.artin_steps {
                let (next, coefficient) = immutable_unique_artin_braid_at_with_inverse_oracle(
                    &rule,
                    &expected_tree,
                    step.index,
                    step.inverse,
                )
                .unwrap();
                expected_tree = next;
                expected_coefficient *= coefficient;
            }
            let assert_replay = |actual: &(FusionTreeKey, f64)| {
                assert_eq!(actual.0.uncoupled(), expected_tree.uncoupled());
                assert_eq!(actual.0.is_dual(), expected_tree.is_dual());
                assert_eq!(actual.0.coupled(), expected_tree.coupled());
                assert_eq!(actual.0.innerlines(), expected_tree.innerlines());
                assert_eq!(actual.0.vertices(), expected_tree.vertices());
                assert_eq!(actual.1.to_bits(), expected_coefficient.to_bits());
            };

            let actual = execute_unique_tree_braid(&rule, &source, &owned.permutation, &owned.artin_steps)
                .unwrap();
            assert_replay(&actual);

            let generic = execute_multiplicity_free_tree_braid(
                &rule,
                &source,
                &owned.permutation,
                &owned.artin_steps,
            )
            .unwrap();
            assert_eq!(generic.len(), 1);
            assert_replay(&generic[0]);

            let raw_axis_positions = [3, 2, 1, 0];
            let prepared = PreparedTreePairOperation::prepare_braid_with_raw_axis_positions(
                &rule,
                4,
                0,
                &permutation,
                &[],
                &levels,
                &[],
                &raw_axis_positions,
            )
            .unwrap();
            let PreparedTreePairPlan::UniqueBraid(borrowed) = &prepared.plan else {
                panic!("UniqueFusion braid must retain borrowed operation metadata");
            };
            let actual = execute_unique_tree_braid_borrowed(&rule, &source, borrowed).unwrap();
            assert_replay(&actual);

            let domain = FusionTreeKey::try_new_for_rule(
                &rule,
                [],
                rule.vacuum(),
                [],
                [],
                [],
            )
            .unwrap();
            let second = FusionTreeKey::try_from_sector_ids(
                [1, 1, 2, 0],
                0,
                [true, false, true, false],
                [2, 0],
                [1, 1, 1],
            )
            .unwrap();
            let sources = [
                FusionTreePairKey::pair(source.clone(), domain.clone()),
                FusionTreePairKey::pair(second, domain),
            ];
            let source_snapshots = sources.clone();
            for source in &sources {
                let mut expected_tree = source.codomain_tree().clone();
                let mut expected_coefficient = <AsymmetricAnyonicRule as MultiplicityFreeFusionSymbols>::Scalar::one();
                for step in &owned.artin_steps {
                    let (next, coefficient) = immutable_unique_artin_braid_at_with_inverse_oracle(
                        &rule,
                        &expected_tree,
                        step.index,
                        step.inverse,
                    )
                    .unwrap();
                    expected_tree = next;
                    expected_coefficient *= coefficient;
                }
                let actual = prepared.execute_multiplicity_free(&rule, source).unwrap();
                assert_eq!(actual.len(), 1);
                assert_eq!(
                    actual[0].0,
                    FusionTreePairKey::pair(expected_tree, source.domain_tree().clone())
                );
                assert_eq!(actual[0].1.to_bits(), expected_coefficient.to_bits());
            }
            assert_eq!(sources, source_snapshots);
        }

        #[test]
        fn borrowed_unique_metadata_rejects_inconsistent_inverse_positions() {
            // What: the doc-hidden safe lowering entry points reject a same-length
            // inverse table that would otherwise erase a nontrivial swap.
            let expected = CoreError::InconsistentAxisPosition {
                logical_axis: 1,
                expected_position: 0,
                actual_position: 1,
            };
            assert_eq!(
                PreparedTreePairOperation::prepare_permute_with_raw_axis_positions(
                    &Z2FusionRule,
                    2,
                    0,
                    &[1, 0],
                    &[],
                    &[0, 1],
                ),
                Err(expected.clone())
            );
            assert_eq!(
                PreparedTreePairOperation::prepare_braid_with_raw_axis_positions(
                    &Z2FusionRule,
                    2,
                    0,
                    &[1, 0],
                    &[],
                    &[0, 1],
                    &[],
                    &[0, 1],
                ),
                Err(expected)
            );
            assert_eq!(
                PreparedTreePairOperation::prepare_permute_with_raw_axis_positions(
                    &Z2FusionRule,
                    2,
                    0,
                    &[0, 0],
                    &[],
                    &[0, 1],
                ),
                Err(CoreError::InvalidPermutation {
                    permutation: vec![0, 0],
                    rank: 2,
                })
            );
        }

        #[test]
        fn prepared_permute_revalidates_symmetric_capability_for_reused_rule() {
            // What: a prepared permutation cannot become an identity, repartition,
            // or general braid when executed with a non-symmetric provider.
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
            let plans = [
                PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 1, &[0], &[1]).unwrap(),
                PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 1, &[0, 1], &[]).unwrap(),
                PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 1, &[1], &[0]).unwrap(),
            ];
            assert!(matches!(plans[0].plan, PreparedTreePairPlan::Identity));
            assert!(matches!(
                plans[1].plan,
                PreparedTreePairPlan::Repartition
            ));
            assert!(matches!(
                plans[2].plan,
                PreparedTreePairPlan::Braid(_) | PreparedTreePairPlan::UniqueBraid(_)
            ));

            for prepared in plans {
                assert_eq!(
                    prepared.execute_unique_rigid(&AsymmetricAnyonicRule, &source),
                    Err(CoreError::UnsupportedBraidingStyle {
                        expected: "symmetric braiding",
                        actual: BraidingStyleKind::Anyonic,
                    })
                );
            }
        }

        #[test]
        fn prepared_transpose_fixes_both_cycle_directions_once() {
            // What: clockwise and anticlockwise TensorKit cyclic permutations are
            // lowered to one fixed direction/count before any source execution.
            let source = FusionTreePairKey::try_pair_from_sector_ids(
                [1, 0],
                [1, 0], 1,
                [false, false],
                [false, false],
                [],
                [],
                [1],
                [1],
            ).unwrap();
            let cases = [
                (
                    &[1, 3][..],
                    &[0, 2][..],
                    PreparedCycleDirection::Clockwise,
                ),
                (
                    &[2, 0][..],
                    &[3, 1][..],
                    PreparedCycleDirection::Anticlockwise,
                ),
            ];
            for (codomain, domain, expected_direction) in cases {
                let prepared =
                    PreparedTreePairOperation::prepare_transpose(2, 2, codomain, domain).unwrap();
                assert_eq!(
                    prepared.plan,
                    PreparedTreePairPlan::Transpose {
                        direction: expected_direction,
                        count: 1,
                    }
                );
                let actual = prepared.execute_unique_rigid(&Z2FusionRule, &source).unwrap();
                let repartitioned = unique_rigid_repartition_tree_pair_unchecked(
                    &Z2FusionRule,
                    &source,
                    codomain.len(),
                )
                .unwrap();
                let (oracle_tree, cycle_coefficient) = match expected_direction {
                    PreparedCycleDirection::Clockwise => {
                        unique_rigid_cycle_clockwise_tree_pair(&Z2FusionRule, &repartitioned.0).unwrap()
                    }
                    PreparedCycleDirection::Anticlockwise => {
                        unique_rigid_cycle_anticlockwise_tree_pair(&Z2FusionRule, &repartitioned.0)
                            .unwrap()
                    }
                };
                let oracle = (oracle_tree, repartitioned.1 * cycle_coefficient);
                assert_eq!(actual, oracle);
            }
        }

        #[test]
        fn unique_prepared_executor_rejects_simple_for_every_plan_variant() {
            // What: a general multiplicity-free plan never becomes a Unique plan
            // merely because its operation variant is an identity or repartition.
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
            let plans = [
                PreparedTreePairOperation::prepare_braid(
                    &SU2FusionRule,
                    1,
                    1,
                    &[0],
                    &[1],
                    &[0],
                    &[1],
                )
                .unwrap(),
                PreparedTreePairOperation::prepare_braid(
                    &SU2FusionRule,
                    1,
                    1,
                    &[0, 1],
                    &[],
                    &[0],
                    &[1],
                )
                .unwrap(),
                PreparedTreePairOperation::prepare_braid(
                    &SU2FusionRule,
                    1,
                    1,
                    &[1],
                    &[0],
                    &[0],
                    &[1],
                )
                .unwrap(),
                PreparedTreePairOperation::prepare_transpose(1, 1, &[1], &[0]).unwrap(),
            ];
            assert!(matches!(plans[0].plan, PreparedTreePairPlan::Identity));
            assert!(matches!(plans[1].plan, PreparedTreePairPlan::Repartition));
            assert!(matches!(plans[2].plan, PreparedTreePairPlan::Braid(_)));
            assert!(matches!(
                plans[3].plan,
                PreparedTreePairPlan::Transpose { .. }
            ));

            for prepared in plans {
                assert_eq!(
                    prepared.execute_unique_rigid(&SU2FusionRule, &source),
                    Err(CoreError::UnsupportedFusionStyle {
                        expected: FusionStyleKind::Unique,
                        actual: FusionStyleKind::Simple,
                    })
                );
            }
        }

        #[test]
        fn prepared_operation_preserves_validation_error_precedence() {
            // What: level dimensions still fail before identity short-circuiting,
            // while transpose still reports the original unlinearized permutation.
            assert_eq!(
                PreparedTreePairOperation::prepare_braid(
                    &SU2FusionRule,
                    1,
                    1,
                    &[0],
                    &[1],
                    &[],
                    &[1],
                ),
                Err(CoreError::DimensionMismatch {
                    expected: 1,
                    actual: 0,
                })
            );
            assert_eq!(
                PreparedTreePairOperation::prepare_transpose(2, 1, &[0, 2], &[2]),
                Err(CoreError::InvalidPermutation {
                    permutation: vec![0, 2, 2],
                    rank: 3,
                })
            );
        }

        #[test]
        fn validation_only_tree_pair_syntax_handles_large_and_rank_zero_maps() {
            let codomain_rank = 10;
            let domain_rank = 9;
            let codomain = (0..codomain_rank).collect::<Vec<_>>();
            let domain =
                (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
            let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
            let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();

            // What: the validation-only API handles ranks beyond SmallVec's inline
            // permutation capacity without requiring a prepared Artin plan.
            PreparedTreePairOperation::validate_permute_syntax(
                codomain_rank,
                domain_rank,
                &codomain,
                &domain,
            )
            .unwrap();
            PreparedTreePairOperation::validate_braid_syntax(
                codomain_rank,
                domain_rank,
                &codomain,
                &domain,
                &codomain_levels,
                &domain_levels,
            )
            .unwrap();
            PreparedTreePairOperation::validate_transpose_syntax(
                codomain_rank,
                domain_rank,
                &codomain,
                &domain,
            )
            .unwrap();

            // What: the empty tensor map remains a valid identity operation.
            PreparedTreePairOperation::validate_permute_syntax(0, 0, &[], &[]).unwrap();
            PreparedTreePairOperation::validate_braid_syntax(0, 0, &[], &[], &[], &[]).unwrap();
            PreparedTreePairOperation::validate_transpose_syntax(0, 0, &[], &[]).unwrap();
        }

        #[test]
        fn unique_transpose_tree_pair_is_cyclic_and_reversible() {
            let source = FusionTreePairKey::try_pair_from_sector_ids(
                [1],
                [1], 1,
                [false],
                [true],
                [],
                [],
                [],
                [],
            ).unwrap();

            let forward_terms =
                multiplicity_free_transpose_tree_pair(&Z2FusionRule, &source, &[1], &[0]).unwrap();
            assert_eq!(forward_terms.len(), 1);
            let (transposed, coefficient) = forward_terms.into_iter().next().unwrap();
            let roundtrip_terms =
                multiplicity_free_transpose_tree_pair(&Z2FusionRule, &transposed, &[1], &[0])
                    .unwrap();
            assert_eq!(roundtrip_terms.len(), 1);
            let (roundtrip, inverse_coefficient) = roundtrip_terms.into_iter().next().unwrap();

            assert_eq!(coefficient, 1.0);
            assert_eq!(inverse_coefficient, 1.0);
            assert_eq!(roundtrip, source);
        }

        #[test]
        fn unique_transpose_tree_pair_matches_tensorkit_clockwise_cycle() {
            let source = FusionTreePairKey::try_pair_from_sector_ids(
                [1, 0],
                [1, 0], 1,
                [false, false],
                [false, false],
                [],
                [],
                [1],
                [1],
            ).unwrap();
            let expected = FusionTreePairKey::try_pair_from_sector_ids(
                [0, 0],
                [1, 1], 0,
                [false, true],
                [true, false],
                [],
                [],
                [1],
                [1],
            ).unwrap();

            let terms =
                multiplicity_free_transpose_tree_pair(&Z2FusionRule, &source, &[1, 3], &[0, 2])
                    .unwrap();
            assert_eq!(terms.len(), 1);
            let (transposed, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, 1.0);
            assert_eq!(transposed, expected);
        }

        #[test]
        fn unique_transpose_tree_pair_matches_tensorkit_anticlockwise_cycle() {
            let source = FusionTreePairKey::try_pair_from_sector_ids(
                [1, 0],
                [1, 0], 1,
                [false, false],
                [false, false],
                [],
                [],
                [1],
                [1],
            ).unwrap();
            let expected = FusionTreePairKey::try_pair_from_sector_ids(
                [1, 1],
                [0, 0], 0,
                [true, false],
                [false, true],
                [],
                [],
                [1],
                [1],
            ).unwrap();

            let terms =
                multiplicity_free_transpose_tree_pair(&Z2FusionRule, &source, &[2, 0], &[3, 1])
                    .unwrap();
            assert_eq!(terms.len(), 1);
            let (transposed, coefficient) = terms.into_iter().next().unwrap();

            assert_eq!(coefficient, 1.0);
            assert_eq!(transposed, expected);
        }

        #[test]
        fn unique_transpose_tree_pair_rejects_noncyclic_permutation() {
            let source = FusionTreePairKey::try_pair_from_sector_ids(
                [1, 0],
                [1], 1,
                [false, false],
                [false],
                [],
                [],
                [1],
                [],
            ).unwrap();

            let err =
                multiplicity_free_transpose_tree_pair(&Z2FusionRule, &source, &[0, 2], &[1])
                    .unwrap_err();

            assert_eq!(
                err,
                CoreError::InvalidPermutation {
                    permutation: vec![0, 2, 1],
                    rank: 3,
                }
            );
        }

        #[derive(Clone, Debug)]
        struct ProbeTreeKey(usize);

        static PROBE_TREE_KEY_EQ_CALLS: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);

        static PROBE_TREE_KEY_HASH_CALLS: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);

        impl PartialEq for ProbeTreeKey {
            fn eq(&self, other: &Self) -> bool {
                PROBE_TREE_KEY_EQ_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.0 == other.0
            }
        }

        impl Eq for ProbeTreeKey {}

        impl std::hash::Hash for ProbeTreeKey {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                PROBE_TREE_KEY_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                std::hash::Hash::hash(&self.0, state);
            }
        }

        #[test]
        fn fusion_term_accumulator_keeps_singleton_path_and_hashes_multi_terms() {
            PROBE_TREE_KEY_EQ_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
            PROBE_TREE_KEY_HASH_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
            let mut singleton = FusionTermAccumulator::new();
            singleton.push(ProbeTreeKey(7), 3usize);
            let singleton_terms = singleton.into_vec();
            assert_eq!(singleton_terms.len(), 1);
            let (singleton_key, singleton_coefficient) = &singleton_terms[0];
            assert_eq!(singleton_key.0, 7);
            assert_eq!(*singleton_coefficient, 3);
            assert_eq!(
                PROBE_TREE_KEY_HASH_CALLS.load(std::sync::atomic::Ordering::Relaxed),
                0
            );

            const DISTINCT: usize = 512;
            const ROUNDS: usize = 4;
            PROBE_TREE_KEY_EQ_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
            PROBE_TREE_KEY_HASH_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
            let mut accumulator = FusionTermAccumulator::new();
            for _ in 0..ROUNDS {
                for key in 0..DISTINCT {
                    accumulator.push(ProbeTreeKey(key), 1usize);
                }
            }
            let terms = accumulator.into_vec();
            assert_eq!(terms.len(), DISTINCT);
            for (index, (key, coefficient)) in terms.iter().enumerate() {
                assert_eq!(key.0, index);
                assert_eq!(*coefficient, ROUNDS);
            }
            let eq_calls = PROBE_TREE_KEY_EQ_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            assert!(
                eq_calls < DISTINCT * ROUNDS * 8,
                "HashMap-backed accumulation should stay linear; saw {eq_calls} equality checks"
            );
            assert!(
                PROBE_TREE_KEY_HASH_CALLS.load(std::sync::atomic::Ordering::Relaxed) > DISTINCT,
                "multi-term accumulation should use the hash path"
            );
        }

        #[test]
        fn multiplicity_free_su2_braid_expands_innerline_channels() {
            let rule = SU2FusionRule;
            let tree = FusionTreeKey::try_from_sector_ids(
                [1, 1, 1, 1], 0,
                [false, false, false, false],
                [0, 1],
                [1, 1, 1],
            ).unwrap();

            let braided =
                multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1, 3], &[0, 1, 2, 3]).unwrap();

            assert_eq!(braided.len(), 2);
            assert_eq!(braided[0].0.uncoupled(), &[SectorId::new(1); 4]);
            assert_eq!(
                braided[0].0.innerlines(),
                &[SectorId::new(0), SectorId::new(1)]
            );
            assert!((braided[0].1 - 0.5).abs() < 1.0e-12);
            assert_eq!(
                braided[1].0.innerlines(),
                &[SectorId::new(2), SectorId::new(1)]
            );
            assert!((braided[1].1 - 0.866_025_403_784_438_6).abs() < 1.0e-12);
        }

        #[test]
        fn bendright_preserves_local_error_precedence_before_missing_duality() {
            // What: local coupled/innerline errors remain observable before a
            // missing final duality flag when several malformed fields coexist.
            let missing_innerline = FusionTreePairKey::pair(
                FusionTreeKey::try_from_sector_ids(
                    [1, 1, 1], 1,
                    [false, false],
                    [],
                    [1, 1],
                ).unwrap(),
                FusionTreeKey::try_from_sector_ids([], 0, [], [], []).unwrap(),
            );
            assert_eq!(
                multiplicity_free_bendright_tree_pair(&SU2FusionRule, &missing_innerline).unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "bendright requires the last codomain innerline",
                }
            );

            let mismatched_coupled = FusionTreePairKey::pair(
                FusionTreeKey::try_from_sector_ids([1, 1], 0, [false], [], [1]).unwrap(),
                FusionTreeKey::try_from_sector_ids([1], 1, [false], [], []).unwrap(),
            );
            assert_eq!(
                multiplicity_free_bendright_tree_pair(&SU2FusionRule, &mismatched_coupled).unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "fusion tree pair requires matching coupled sectors",
                }
            );
        }

        #[test]
        fn fusion_tree_group_key_records_external_sector_tuples_and_duality() {
            let group = FusionTreeGroupKey::from_sector_ids([2, 3], [5], [false, true], [true]);

            assert_eq!(
                group.codomain_uncoupled(),
                &[SectorId::new(2), SectorId::new(3)]
            );
            assert_eq!(group.domain_uncoupled(), &[SectorId::new(5)]);
            assert_eq!(group.codomain_is_dual(), &[false, true]);
            assert_eq!(group.domain_is_dual(), &[true]);

            let same = FusionTreeGroupKey::new(
                [SectorId::new(2), SectorId::new(3)],
                [SectorId::new(5)],
                [false, true],
                [true],
            );
            assert_eq!(group, same);
        }

        #[test]
        fn selected_leg_tuple_visitor_is_fallible_and_restartable() {
            // What: an early visitor error stops traversal without corrupting the
            // reusable scratch used by a later traversal.
            let space = FusionProductSpace::new([
                SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false),
                SectorLeg::new([(SectorId::new(2), 1), (SectorId::new(3), 1)], true),
            ]);
            let expected = materialized_leg_tuple_oracle(&space)
                .into_iter()
                .map(|tuple| {
                    tuple
                        .into_iter()
                        .map(|leg| (leg.sector(), leg.is_dual()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let mut partial = Vec::new();
            let err = space
                .try_visit_selected_leg_tuples(&mut |tuple| {
                    partial.push(
                        tuple
                            .iter()
                            .map(|leg| (leg.sector(), leg.is_dual()))
                            .collect::<Vec<_>>(),
                    );
                    if partial.len() == 2 {
                        Err("stop")
                    } else {
                        Ok(())
                    }
                })
                .unwrap_err();
            assert_eq!(err, "stop");
            assert_eq!(partial, expected[..2]);

            let mut restarted = Vec::new();
            space
                .try_visit_selected_leg_tuples::<(), _>(&mut |tuple| {
                    restarted.push(
                        tuple
                            .iter()
                            .map(|leg| (leg.sector(), leg.is_dual()))
                            .collect::<Vec<_>>(),
                    );
                    Ok(())
                })
                .unwrap();
            assert_eq!(restarted, expected);
        }

        fn assert_literal_binary_choice_sector_grids<R>(
            rule: &R,
            vacuum: SectorId,
            external: SectorId,
            mut groups: Vec<(SectorId, Vec<[SectorId; 2]>)>,
        ) where
            R: MultiplicityFreeFusionRule,
        {
            let homspace = FusionTreeHomSpace::new(
                FusionProductSpace::new([
                    SectorLeg::new([(vacuum, 2), (external, 3)], false),
                    SectorLeg::new([(vacuum, 5), (external, 7)], true),
                ]),
                FusionProductSpace::new([
                    SectorLeg::new([(vacuum, 11), (external, 13)], true),
                    SectorLeg::new([(vacuum, 17), (external, 19)], false),
                ]),
            );

            groups.sort_by_key(|(coupled, _)| *coupled);
            let layout = homspace.fusion_tree_layout_data_uncached(rule);
            assert_eq!(layout.sectors.len(), groups.len());
            let expected_key_count = groups
                .iter()
                .map(|(_, trees)| trees.len() * trees.len())
                .sum::<usize>();
            assert_eq!(layout.keys.len(), expected_key_count);

            let structure = homspace
                .coupled_subblock_structure_from_leg_degeneracies(rule)
                .unwrap();
            let regions = structure.coupled_sector_regions(2).unwrap().unwrap();
            assert_eq!(regions.len(), groups.len());
            let mut block_index = 0usize;
            let mut sector_offset = 0usize;
            for (sector_index, ((expected_coupled, trees), sector)) in
                groups.iter().zip(&layout.sectors).enumerate()
            {
                assert_eq!(
                    (sector.start, sector.row_count, sector.col_count),
                    (block_index, trees.len(), trees.len())
                );

                let row_shapes = trees
                    .iter()
                    .map(|tree| {
                        [
                            if tree[0] == vacuum { 2 } else { 3 },
                            if tree[1] == vacuum { 5 } else { 7 },
                        ]
                    })
                    .collect::<Vec<_>>();
                let col_shapes = trees
                    .iter()
                    .map(|tree| {
                        [
                            if tree[0] == vacuum { 11 } else { 13 },
                            if tree[1] == vacuum { 17 } else { 19 },
                        ]
                    })
                    .collect::<Vec<_>>();
                let row_dims = row_shapes
                    .iter()
                    .map(|shape| shape[0] * shape[1])
                    .collect::<Vec<_>>();
                let col_dims = col_shapes
                    .iter()
                    .map(|shape| shape[0] * shape[1])
                    .collect::<Vec<_>>();
                let matrix_rows = row_dims.iter().sum::<usize>();
                let matrix_cols = col_dims.iter().sum::<usize>();
                let mut col_offset = 0usize;
                for (col, domain_tree) in trees.iter().enumerate() {
                    let mut row_offset = 0usize;
                    for (row, codomain_tree) in trees.iter().enumerate() {
                        let key = &layout.keys[block_index];
                        assert_eq!(key.codomain_uncoupled(), codomain_tree);
                        assert_eq!(key.domain_uncoupled(), domain_tree);
                        assert_eq!(key.codomain_is_dual(), &[false, true]);
                        assert_eq!(key.domain_is_dual(), &[true, false]);
                        assert_eq!(key.coupled(), *expected_coupled);

                        let block = structure.block(block_index).unwrap();
                        assert_eq!(
                            block.shape(),
                            &[
                                row_shapes[row][0],
                                row_shapes[row][1],
                                col_shapes[col][0],
                                col_shapes[col][1],
                            ]
                        );
                        assert_eq!(
                            block.strides(),
                            &[
                                1,
                                row_shapes[row][0],
                                matrix_rows,
                                matrix_rows * col_shapes[col][0],
                            ]
                        );
                        assert_eq!(
                            block.offset(),
                            sector_offset + row_offset + matrix_rows * col_offset
                        );
                        row_offset += row_dims[row];
                        block_index += 1;
                    }
                    col_offset += col_dims[col];
                }

                let sector_end = sector_offset + matrix_rows * matrix_cols;
                assert_eq!(regions[sector_index].coupled(), *expected_coupled);
                assert_eq!(regions[sector_index].rows(), matrix_rows);
                assert_eq!(regions[sector_index].cols(), matrix_cols);
                assert_eq!(regions[sector_index].range(), sector_offset..sector_end);
                sector_offset = sector_end;
            }
            assert_eq!(block_index, expected_key_count);
            assert_eq!(structure.required_len().unwrap(), sector_offset);
        }

        #[test]
        fn canonical_coupled_grid_has_literal_nonabelian_and_product_metadata() {
            assert_literal_binary_choice_sector_grids(
                &FermionParityFusionRule,
                z2_even(),
                z2_odd(),
                vec![
                    (z2_even(), vec![[z2_even(), z2_even()], [z2_odd(), z2_odd()]]),
                    (z2_odd(), vec![[z2_odd(), z2_even()], [z2_even(), z2_odd()]]),
                ],
            );
            assert_literal_binary_choice_sector_grids(
                &SU2FusionRule,
                su2(0),
                su2(1),
                vec![
                    (su2(0), vec![[su2(0), su2(0)], [su2(1), su2(1)]]),
                    (su2(1), vec![[su2(1), su2(0)], [su2(0), su2(1)]]),
                    (su2(2), vec![[su2(1), su2(1)]]),
                ],
            );

            type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
            type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
            type Fz2U1Rule =
                ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
            type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
            type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

            let triple_rule = TripleRule::new(
                Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
                SU2FusionRule,
            );
            let vacuum =
                TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(0)), su2(0));
            let external =
                TripleCodec::encode(Fz2U1Codec::encode(z2_odd(), u1(0)), su2(1));
            let second_channel =
                TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(0)), su2(2));
            assert_literal_binary_choice_sector_grids(
                &triple_rule,
                vacuum,
                external,
                vec![
                    (vacuum, vec![[vacuum, vacuum], [external, external]]),
                    (
                        external,
                        vec![[external, vacuum], [vacuum, external]],
                    ),
                    (second_channel, vec![[external, external]]),
                ],
            );
        }

        #[test]
        fn generic_tree_cannot_enter_multiplicity_free_projection() {
            // What: a genuine multiplicity-two tree is rejected
            // before the compact path can erase its vertex identity.
            let rule = UnitaryToyOmRule;
            let a = SectorId::new(UnitaryToyOmRule::A);
            let coupled = SectorId::new(UnitaryToyOmRule::C);
            let tree = FusionTreeKey::try_new_for_rule(
                &rule,
                [a, a],
                coupled,
                [false, false],
                [],
                [MultiplicityIndex::new(2).unwrap()],
            )
            .unwrap();

            let error = match project_multiplicity_free_tree(&rule, &tree) {
                Ok(_) => panic!("Generic tree entered multiplicity-free projection"),
                Err(error) => error,
            };
            assert_eq!(
                error,
                CoreError::UnsupportedFusionStyle {
                    expected: FusionStyleKind::Simple,
                    actual: FusionStyleKind::Generic,
                }
            );

            // A proof is an indexed view of the exact slice it checked. Creating
            // another key later cannot extend that authority.
            let checked = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false; 2], [], [1]).unwrap();
            let separate =
                FusionTreeKey::try_from_sector_ids([1, 1], 0, [false; 2], [], [2]).unwrap();
            let projection =
                MultiplicityFreeTreeProjection::checked(&SU2FusionRule, std::slice::from_ref(&checked))
                    .unwrap();
            assert!(projection.tree_at(0).is_some());
            assert!(projection.tree_at(1).is_none());
            assert_eq!(separate.vertices(), &[MultiplicityIndex::new(2).unwrap()]);
        }

        #[test]
        fn compact_block_error_does_not_publish_partial_rows() {
            let rule = SU2FusionRule;
            let sources = compact_operator_cohort_fixture(&rule, su2(2), su2(2));
            let valid = sources[..4].to_vec();
            let permutation = [7usize, 6, 5, 4, 3, 2, 1, 0];
            let domain = [8usize];
            let baseline =
                multiplicity_free_permute_tree_pair_block(&rule, &valid, &permutation, &domain)
                    .unwrap();

            let mut malformed = valid.clone();
            let source = &valid[1];
            let codomain = source.codomain_tree();
            let shortened = FusionTreeKey::new(
                codomain.uncoupled().iter().copied(),
                codomain.coupled(),
                codomain.is_dual().iter().copied(),
                codomain.innerlines()[..codomain.innerlines().len() - 1]
                    .iter()
                    .copied(),
                codomain.vertices().iter().copied(),
            );
            malformed[1] =
                FusionTreePairKey::pair(shortened, source.domain_tree().clone());
            let snapshot = malformed.clone();

            // What: an error after earlier source rows were staged leaves caller
            // keys unchanged and cannot affect a later successful block transform.
            assert!(multiplicity_free_permute_tree_pair_block(
                &rule,
                &malformed,
                &permutation,
                &domain,
            )
            .is_err());
            assert_eq!(malformed, snapshot);
            assert_eq!(
                multiplicity_free_permute_tree_pair_block(
                    &rule,
                    &valid,
                    &permutation,
                    &domain,
                )
                .unwrap(),
                baseline
            );
        }

        #[test]
        fn compact_repartition_preserves_source_major_error_precedence() {
            let codomain = |coupled, innerlines: &[SectorId]| {
                FusionTreeKey::new(
                    [u1(1), u1(1), u1(1), u1(1)], coupled,
                    [false; 4],
                    innerlines.iter().copied(),
                    [MultiplicityIndex::ONE; 3],
                )
            };
            let domain = |coupled| {
                FusionTreeKey::new(
                    [u1(4)], coupled,
                    [false],
                    [],
                    [],
                )
            };
            let sources = [
                FusionTreePairKey::pair(
                    codomain(u1(4), &[u1(2)]),
                    domain(u1(4)),
                ),
                FusionTreePairKey::pair(
                    codomain(u1(4), &[]),
                    domain(u1(99)),
                ),
            ];

            // What: the public categorical boundary rejects source 0's malformed
            // tree before compact bend-local validation can inspect later sources.
            assert_eq!(
                multiplicity_free_permute_tree_pair_block(
                    &U1FusionRule,
                    &sources,
                    &[0, 1],
                    &[4, 3, 2],
                )
                .unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "fusion tree has an invalid number of innerlines",
                }
            );
        }

        #[test]
        fn ordered_block_linear_storage_preserves_absent_and_zero_structure() {
            let rule = SU2FusionRule;
            let half = SectorLeg::new([(su2(1), 1)], false);
            let hom = FusionTreeHomSpace::new(
                FusionProductSpace::new([half.clone(), half.clone(), half]),
                FusionProductSpace::new([SectorLeg::new([(su2(1), 1)], false)]),
            );
            let keys = hom.fusion_tree_keys(&rule);
            assert_eq!(keys.len(), 2);
            let group = validate_tree_pair_block_group_for_rule(&rule, &keys)
                .unwrap()
                .expect("SU2 cohort is nonempty");

            let singleton_basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
            let mut singleton_columns = DenseColumns::with_capacity(2, 2);
            let row0 = singleton_columns.push_empty_row();
            let row1 = singleton_columns.push_empty_row();
            singleton_columns.row_mut(row0)[0] = Some(0.0);
            singleton_columns.row_mut(row1)[1] = Some(2.0);
            let singleton = order_compact_block(singleton_basis, singleton_columns);
            assert_eq!(singleton.destinations(), keys.as_ref());
            assert_eq!(
                singleton.storage(),
                &OrderedBlockLinearStorage::SingletonColumns {
                    destination_rows: vec![0, 1],
                    coefficients: vec![0.0, 2.0],
                }
            );

            let dense_basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
            let mut dense_columns = DenseColumns::with_capacity(2, 2);
            let row0 = dense_columns.push_empty_row();
            let row1 = dense_columns.push_empty_row();
            dense_columns.row_mut(row0)[0] = Some(0.0);
            dense_columns.row_mut(row1)[0] = Some(1.0);
            dense_columns.row_mut(row1)[1] = Some(2.0);
            let dense = order_compact_block(dense_basis, dense_columns);
            assert_eq!(dense.destinations(), keys.as_ref());
            assert_eq!(
                dense.storage(),
                &OrderedBlockLinearStorage::DenseDstSrc(vec![
                    Some(0.0),
                    None,
                    Some(1.0),
                    Some(2.0),
                ])
            );
        }

        #[test]
        fn transpose_tree_pair_block_matches_complex_f_oracle() {
            let rule = FibonacciFAdmissibilityProbe::with_complex_f_phase();
            let tau = || SectorLeg::new([(SectorId::new(1), 1)], false);
            let hom = FusionTreeHomSpace::new(
                FusionProductSpace::new([tau(), tau()]),
                FusionProductSpace::new([tau(), tau()]),
            );
            let sources = hom.fusion_tree_keys(&rule);
            assert!(sources.len() > 1);
            for (codomain_permutation, domain_permutation, expected_direction) in [
                (
                    [1usize, 3],
                    [0usize, 2],
                    PreparedCycleDirection::Clockwise,
                ),
                (
                    [2usize, 0],
                    [3usize, 1],
                    PreparedCycleDirection::Anticlockwise,
                ),
            ] {
                let prepared = PreparedTreePairOperation::prepare_transpose(
                    2,
                    2,
                    &codomain_permutation,
                    &domain_permutation,
                )
                .unwrap();
                assert!(matches!(
                    prepared.plan,
                    PreparedTreePairPlan::Transpose { direction, .. }
                        if direction == expected_direction
                ));
                let rows = multiplicity_free_transpose_tree_pair_block(
                    &rule,
                    &sources,
                    &codomain_permutation,
                    &domain_permutation,
                )
                .unwrap();
                assert!(rows
                    .iter()
                    .flatten()
                    .any(|(_, coefficient)| coefficient.im.abs() > 1.0e-12));

                // What: both compact cycle directions preserve ordered multi-row
                // non-real F products and conjugation against the old full keys.
                assert_compact_transpose_matches_full_key_oracle(
                    &rule,
                    &sources,
                    &codomain_permutation,
                    &domain_permutation,
                    true,
                );
            }
        }

        #[test]
        fn split_fusion_tree_matches_tensorkit_front_tail_convention() {
            let rule = SU2FusionRule;
            let half = SU2Irrep::from_twice_spin(1).sector_id();
            let one = SU2Irrep::from_twice_spin(2).sector_id();
            let tree = FusionTreeKey::new(
                [half, half, one], one,
                [false, false, true],
                [SectorId::new(0)],
                [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
            );

            let (front, tail) = split_fusion_tree(&rule, &tree, 2).unwrap();

            assert_eq!(front.uncoupled(), &[half, half]);
            assert_eq!(front.coupled(), SectorId::new(0));
            assert_eq!(front.is_dual(), &[false, false]);
            assert_eq!(front.innerlines(), &[]);
            assert_eq!(front.vertices(), &[MultiplicityIndex::ONE]);
            assert_eq!(tail.uncoupled(), &[SectorId::new(0), one]);
            assert_eq!(tail.coupled(), one);
            assert_eq!(tail.is_dual(), &[false, true]);
            assert_eq!(tail.innerlines(), &[]);
            assert_eq!(tail.vertices(), &[MultiplicityIndex::ONE]);
        }

        #[test]
        fn generic_split_preserves_multiplicity_style_and_vertex_labels() {
            // What: splitting a valid Generic tree returns two keys that remain
            // valid for the same rule and retain the selected multiplicity basis.
            let rule = ToyOmRule;
            let a = SectorId::new(ToyOmRule::A);
            let c = SectorId::new(ToyOmRule::C);
            let source = FusionTreeKey::try_new_for_rule(
                &rule,
                [a, a, a], a,
                [false; 3],
                [c],
                [MultiplicityIndex::new(2).expect("test multiplicity label is one-based"), MultiplicityIndex::ONE],
            )
            .unwrap();

            let (front, tail) = split_fusion_tree(&rule, &source, 2).unwrap();

            front.validate_for_rule(&rule).unwrap();
            tail.validate_for_rule(&rule).unwrap();
            assert_eq!(front.vertices(), &[MultiplicityIndex::new(2).unwrap()]);
            assert_eq!(tail.vertices(), &[MultiplicityIndex::ONE]);
        }

        #[test]
        fn generic_identity_braid_rejects_an_inadmissible_source() {
            // What: the Generic identity shortcut is behind the same categorical
            // boundary as multiplicity-free tree operations.
            let a = SectorId::new(ToyOmRule::A);
            let c = SectorId::new(ToyOmRule::C);
            let invalid = FusionTreeKey::new(
                [a; 3], c,
                [false; 3],
                [c],
                [MultiplicityIndex::ONE; 2],
            );

            assert_eq!(
                generic_braid_tree(&ToyOmRule, &invalid, &[0, 1, 2], &[0, 1, 2])
                    .unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "fusion tree contains an inadmissible fusion vertex",
                }
            );
        }

        #[derive(Debug, Default)]
        struct CrossIncompatibleGenericRule {
            f_calls: std::sync::atomic::AtomicUsize,
        }

        impl CrossIncompatibleGenericRule {
            const VACUUM: usize = 0;
            const A: usize = 1;
            const B: usize = 2;
            const C: usize = 3;
            const X: usize = 4;
            const E: usize = 5;
            const D: usize = 6;
            const F: usize = 7;
            const G: usize = 8;
            const Q: usize = 9;
        }

        impl FusionRule for CrossIncompatibleGenericRule {
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
                SectorId::new(Self::VACUUM)
            }

            fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
                let channel = match (left.id(), right.id()) {
                    (Self::VACUUM, sector) | (sector, Self::VACUUM) => Some(sector),
                    (Self::A, Self::B) => Some(Self::E),
                    (Self::E, Self::C) => Some(Self::D),
                    (Self::D, Self::X) => Some(Self::Q),
                    (Self::B, Self::C) => Some(Self::F),
                    (Self::F, Self::X) => Some(Self::G),
                    (Self::G, Self::X) => Some(Self::F),
                    (Self::A, Self::G) => Some(Self::Q),
                    (Self::A, Self::Q) => Some(Self::G),
                    (Self::Q, Self::X) => Some(Self::D),
                    (Self::D, Self::C) => Some(Self::E),
                    _ => None,
                };
                channel
                    .map(|sector| smallvec![SectorId::new(sector)])
                    .unwrap_or_default()
            }
        }

        impl GenericFusionSymbols for CrossIncompatibleGenericRule {
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
                self.f_calls
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                GenericFArray::new(vec![997.0], (1, 1, 1, 1))
            }

            fn r_symbol_generic(
                &self,
                a: SectorId,
                b: SectorId,
                c: SectorId,
            ) -> GenericRMatrix<Self::Scalar> {
                let rows = self.nsymbol(a, b, c);
                let cols = self.nsymbol(b, a, c);
                GenericRMatrix::new(vec![1.0; rows * cols], rows, cols)
            }
        }

        #[test]
        fn generic_multi_fmove_filters_cross_incompatible_trees_before_f() {
            // What: forward and inverse multi-F moves discard individually valid
            // long/short trees whose cross-tree F vertex does not exist, even when
            // the provider returns a nonzero sentinel outside its valid domain.
            let rule = CrossIncompatibleGenericRule::default();
            let sector = SectorId::new;
            let long = FusionTreeKey::new(
                [
                    sector(CrossIncompatibleGenericRule::A),
                    sector(CrossIncompatibleGenericRule::B),
                    sector(CrossIncompatibleGenericRule::C),
                    sector(CrossIncompatibleGenericRule::X),
                ], sector(CrossIncompatibleGenericRule::Q),
                [false; 4],
                [
                    sector(CrossIncompatibleGenericRule::E),
                    sector(CrossIncompatibleGenericRule::D),
                ],
                [MultiplicityIndex::ONE; 3],
            );
            let short = FusionTreeKey::new(
                [
                    sector(CrossIncompatibleGenericRule::B),
                    sector(CrossIncompatibleGenericRule::C),
                    sector(CrossIncompatibleGenericRule::X),
                ], sector(CrossIncompatibleGenericRule::G),
                [false; 3],
                [sector(CrossIncompatibleGenericRule::F)],
                [MultiplicityIndex::ONE; 2],
            );
            long.validate_for_rule(&rule).unwrap();
            short.validate_for_rule(&rule).unwrap();

            assert!(generic_multi_fmove_tree(&rule, &long).unwrap().is_empty());
            assert!(generic_multi_fmove_inv_tree(
                &rule,
                sector(CrossIncompatibleGenericRule::A),
                sector(CrossIncompatibleGenericRule::Q),
                &short,
                false,
            )
            .unwrap()
            .is_empty());
            assert_eq!(
                rule.f_calls.load(std::sync::atomic::Ordering::Relaxed),
                0
            );
        }

        fn generic_tree_pair(vertices_first: usize, vertices_second: usize) -> (FusionTreeKey, FusionTreeKey) {
            let a = SectorId::new(ToyOmRule::A);
            let c = SectorId::new(ToyOmRule::C);
            let first = FusionTreeKey::new([a, a], c, [false, false], [], [MultiplicityIndex::new(vertices_first).expect("test multiplicity label is one-based")]);
            let second = FusionTreeKey::new([a, a], c, [false, false], [], [MultiplicityIndex::new(vertices_second).expect("test multiplicity label is one-based")]);
            (first, second)
        }

        #[test]
        fn fusion_tree_key_generic_distinguishes_vertices() {
            // What: multiplicity vertices are always part of categorical identity;
            // fusion style comes from the provider rather than a duplicated flag.
            let (first, second) = generic_tree_pair(1, 2);
            assert_ne!(first, second);
            assert_ne!(first.cmp(&second), std::cmp::Ordering::Equal);

            let mut set = std::collections::HashSet::new();
            set.insert(first);
            set.insert(second);
            assert_eq!(set.len(), 2, "Generic keys differing in vertices must stay distinct");
        }

        #[test]
        fn frozen_generic_enumeration_shares_externals_and_charges_each_owner_once() {
            // What: Generic fanout shares external fields; retained-byte accounting
            // deduplicates one owner's shared backing but charges independently
            // allocated equal-content backing separately.
            let rule = ToyOmRule;
            let a = SectorId::new(ToyOmRule::A);
            let c = SectorId::new(ToyOmRule::C);
            let trees = collect_generic_fusion_trees_for_coupled(
                &rule,
                &[a, a],
                &[false, false],
                &[a, a],
                c,
            );
            assert!(trees.len() > 1);
            assert!(trees
                .windows(2)
                .all(|trees| Arc::ptr_eq(&trees[0].uncoupled, &trees[1].uncoupled)));
            assert!(trees
                .windows(2)
                .all(|trees| Arc::ptr_eq(&trees[0].is_dual, &trees[1].is_dual)));

            let independent = FusionTreeKey::new(
                trees[0].uncoupled().iter().copied(),
                trees[0].coupled(),
                trees[0].is_dual().iter().copied(),
                trees[0].innerlines().iter().copied(),
                trees[0].vertices().iter().copied(),
            );
            let mut shared_seen = rustc_hash::FxHashSet::default();
            let shared_charge = charge_fusion_tree_key_backings(&mut shared_seen, &trees[0])
                .saturating_add(charge_fusion_tree_key_backings(
                    &mut shared_seen,
                    &trees[0],
                ));
            let mut independent_seen = rustc_hash::FxHashSet::default();
            let independent_charge =
                charge_fusion_tree_key_backings(&mut independent_seen, &trees[0])
                    .saturating_add(charge_fusion_tree_key_backings(
                        &mut independent_seen,
                        &independent,
                    ));
            assert_eq!(
                shared_charge,
                charge_fusion_tree_key_backings(
                    &mut rustc_hash::FxHashSet::default(),
                    &trees[0]
                )
            );
            assert!(independent_charge > shared_charge);
        }

        #[test]
        fn checked_generic_artin_calls_only_required_symbols_and_preserves_failures() {
            let outer = ArtinSpy::new();
            generic_artin_braid_at_with_inverse_checked(&outer, &unitary_rank2_tree(1), 0, false).unwrap();
            assert_eq!((outer.f_calls.get(), outer.r_calls.get()), (0, 1));
            assert_eq!(outer.rigid_calls.get(), 0);
            let inner = ArtinSpy::new();
            generic_artin_braid_at_with_inverse_checked(&inner, &unitary_rank3_tree(1), 1, false).unwrap();
            assert!(inner.f_calls.get() > 0 && inner.r_calls.get() > 0);
            let fail_r = ArtinSpy { fail_r: Some(1), ..ArtinSpy::new() };
            assert!(matches!(generic_artin_braid_at_with_inverse_checked(&fail_r, &unitary_rank2_tree(1), 0, false), Err(CheckedGenericSymbolError::Provider(ArtinSpyError::R))));
            let fail_f = ArtinSpy { fail_f: Some(1), ..ArtinSpy::new() };
            assert!(matches!(generic_artin_braid_at_with_inverse_checked(&fail_f, &unitary_rank3_tree(1), 1, false), Err(CheckedGenericSymbolError::Provider(ArtinSpyError::F))));
        }

        #[test]
        fn checked_generic_artin_rejects_categorical_symbol_shapes_before_indexing() {
            let bad_r = ArtinSpy { bad_r: true, ..ArtinSpy::new() };
            assert!(matches!(generic_artin_braid_at_with_inverse_checked(&bad_r, &unitary_rank2_tree(1), 0, false), Err(CheckedGenericSymbolError::Shape { symbol: "R", .. })));
            let bad_f = ArtinSpy { bad_f: true, ..ArtinSpy::new() };
            assert!(matches!(generic_artin_braid_at_with_inverse_checked(&bad_f, &unitary_rank3_tree(1), 1, false), Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })));
        }

        #[test]
        fn checked_generic_bend_and_repartition_match_legacy_rows_and_order() {
            let pair = a4_dual_pair_rank2(2);
            let legacy_bend = generic_bendright_tree_pair(&A4BendRule, &pair).unwrap();
            let checked = A4BendRule;
            let checked_bend =
                generic_bendright_tree_pair_checked(&checked, &pair).unwrap();
            assert_eq!(checked_bend, legacy_bend);

            let legacy_repartition =
                generic_repartition_tree_pair(&A4BendRule, &pair, 0).unwrap();
            let checked = A4BendRule;
            let checked_repartition =
                generic_repartition_tree_pair_checked(&checked, &pair, 0).unwrap();
            assert_eq!(checked_repartition, legacy_repartition);

            let checked = A4BendRule;
            assert_eq!(
                generic_bendleft_tree_pair_checked(&checked, &legacy_bend[0].0).unwrap(),
                generic_bendleft_tree_pair(&A4BendRule, &legacy_bend[0].0).unwrap()
            );
        }

        #[test]
        fn checked_generic_b_and_a_reject_malformed_categorical_f_shapes() {
            let bad_b = CheckedA4Spy {
                bad_b_f: true,
                ..CheckedA4Spy::new()
            };
            assert!(matches!(
                generic_bendright_tree_pair_checked(&bad_b, &a4_pair_rank2(1)),
                Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })
            ));

            let bad_a = CheckedA4Spy {
                bad_a_f: true,
                ..CheckedA4Spy::new()
            };
            let t = a4_three();
            assert!(matches!(
                GenericRigidAccess::try_a_symbol_generic(&bad_a, t, t, t),
                Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })
            ));
        }

        #[test]
        fn refute_b2a_bendright_uses_b_row_not_column() {
            // End-to-end: bend the codomain vertex μ; the ν output distribution must
            // equal ROW μ of B (coeff = coeff0·Bmat[μ,ν], coeff0=1 here). A μ↔ν swap
            // in the Bmat.get(μ,ν) call would emit COLUMN μ instead.
            let rule = TransposeProbeRule;
            let s = SectorId::new(1);
            let b = tp_expected_b();
            for mu in 1..=2usize {
                // cod [1,1]->1 vertex μ ; dom [1]->1.
                let cod = FusionTreeKey::new([s, s], s, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
                let dom = FusionTreeKey::new([s], s, [false], [], []);
                let pair = FusionTreePairKey::pair(cod, dom);
                let out = generic_bendright_tree_pair(&rule, &pair).unwrap();
                // What: ν fanout owns only its changed vertex labels; the three
                // unchanged domain fields share one frozen owner.
                assert_eq!(out.len(), 2);
                for terms in out.windows(2) {
                    let left = terms[0].0.domain_tree();
                    let right = terms[1].0.domain_tree();
                    assert!(Arc::ptr_eq(&left.uncoupled, &right.uncoupled));
                    assert!(Arc::ptr_eq(&left.is_dual, &right.is_dual));
                    assert!(Arc::ptr_eq(&left.innerlines, &right.innerlines));
                    assert!(!Arc::ptr_eq(&left.vertices, &right.vertices));
                }
                // Collect coeff keyed by output domain vertex label (=ν+1).
                let mut got = [0.0f64; 2];
                for (key, coeff) in &out {
                    let nu = key.domain_tree().vertices()[0].get(); // 1-based ν label
                    got[nu - 1] = *coeff;
                }
                let row = &b[mu - 1];
                for nu in 0..2 {
                    assert!(
                        (got[nu] - row[nu]).abs() < 1e-12,
                        "μ={mu}: ν={nu} coeff {} want ROW-μ {} (transpose ⇒ column-μ)",
                        got[nu],
                        row[nu]
                    );
                }
                // Guard: distinguishable from the column (transposed) reading.
                let col = [b[0][mu - 1], b[1][mu - 1]];
                assert!(
                    (got[0] - col[0]).abs() > 1e-9 || (got[1] - col[1]).abs() > 1e-9,
                    "μ={mu}: row and column coincide — test cannot discriminate"
                );
            }
        }

        // ==== REFUTE (adversarial): coeff2 adjoint conj on GENUINELY complex data ====
        //
        // Gap found by the verifier: in `ComplexUnitaryRule` the domain vector
        // `coeff2` is always a real UNIT vector (rank-1 domain → seed case), so the
        // `coeff₂'` adjoint (TK `duality_manipulations.jl:279`) and the
        // `multi_Fmove_inv = conj(associator)` step (TK `basic_manipulations.jl:
        // 439/462`) are NEVER exercised on complex data by any existing test — the
        // A4 oracle is fully real, and the complex fold round-trip cancels a
        // consistent double conj error.
        //
        // This synthetic (deliberately NOT pentagon-consistent — a pure algebraic
        // fixture) rule drives one COMPLEX interior F into `coeff2` while keeping the
        // A-matrix REAL, isolating the two conj sites:
        //   * F(1,1,2,2,0,3) = 1  (real)  ⇒ Asymbol(1,2,3) is real, = 1.
        //   * F(1,3,3,2,2,3) = w  (complex) ⇒ multi_associator seed = w.
        // TK's `multi_Fmove_inv` returns conj(associator) = conj(w); TK's foldright
        // contracts coeff₂' (a SECOND conj) against transpose(A)·coeff₁, so the two
        // conjs cancel and the observable foldright coefficient is the RAW
        // associator w. Test A pins `multi_Fmove_inv = conj(w)` alone (breaks the
        // double-error symmetry); Test B pins the foldright net = w. Together they
        // rule out both a single and a double conj slip. A single missing conj at
        // EITHER site flips the observable to conj(w) ≠ w.
        #[derive(Clone, Copy, Debug)]
        struct Coeff2ConjRule;

        fn c2c_w() -> Complex64 {
            Complex64::new(0.6, 0.8) // |w| = 1, genuinely complex
        }

        impl FusionRule for Coeff2ConjRule {
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
            fn dual(&self, sector: SectorId) -> SectorId {
                sector // all self-dual
            }
            fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
                match (left.id(), right.id()) {
                    (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                    (1, 1) => smallvec![SectorId::new(0)],
                    (1, 2) | (2, 1) => smallvec![SectorId::new(3)],
                    (1, 3) | (3, 1) => smallvec![SectorId::new(2)],
                    (2, 3) | (3, 2) => smallvec![SectorId::new(2)],
                    (3, 3) => smallvec![SectorId::new(3)],
                    _ => smallvec![SectorId::new(0)],
                }
            }
            fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }

        impl GenericFusionSymbols for Coeff2ConjRule {
            type Scalar = Complex64;
            fn f_symbol_generic(
                &self,
                a: SectorId,
                b: SectorId,
                c: SectorId,
                d: SectorId,
                e: SectorId,
                f: SectorId,
            ) -> GenericFArray<Self::Scalar> {
                let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
                match ids {
                    // Asymbol(1,2,3) reads F(dual1,1,2,2,0,3) = F(1,1,2,2,0,3): REAL.
                    (1, 1, 2, 2, 0, 3) => {
                        GenericFArray::new(vec![Complex64::new(1.0, 0.0)], (1, 1, 1, 1))
                    }
                    // multi_associator seed for domain [3,3]->3 folded onto b=2: COMPLEX.
                    (1, 3, 3, 2, 2, 3) => GenericFArray::new(vec![c2c_w()], (1, 1, 1, 1)),
                    _ => {
                        let shape = (
                            self.nsymbol(a, b, e),
                            self.nsymbol(e, c, d),
                            self.nsymbol(b, c, f),
                            self.nsymbol(a, f, d),
                        );
                        if shape == (1, 1, 1, 1) {
                            GenericFArray::new(vec![Complex64::new(1.0, 0.0)], shape)
                        } else {
                            panic!("Coeff2ConjRule: unmodelled non-singleton F{ids:?} shape={shape:?}");
                        }
                    }
                }
            }
            fn r_symbol_generic(
                &self,
                _a: SectorId,
                _b: SectorId,
                _c: SectorId,
            ) -> GenericRMatrix<Self::Scalar> {
                GenericRMatrix::new(vec![Complex64::new(1.0, 0.0)], 1, 1)
            }
        }

        impl GenericRigidSymbols for Coeff2ConjRule {
            fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
                Complex64::new(1.0, 0.0)
            }
            fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
                Complex64::new(1.0, 0.0)
            }
            fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
                Complex64::new(1.0, 0.0)
            }
        }

        // Test A: `multi_Fmove_inv` alone returns conj(associator) on complex F.
        #[test]
        fn refute_b2b_multi_fmove_inv_is_conj_associator_complex() {
            let rule = Coeff2ConjRule;
            let s1 = SectorId::new(1);
            let s3 = SectorId::new(3);
            let w = c2c_w();
            // domain tree [3,3] -> 3 (single vertex); leading dual(a)=1, target b=2.
            let domain = FusionTreeKey::new(
                [s3, s3], s3,
                [false, false],
                [],
                [MultiplicityIndex::ONE],
            );
            let terms =
                generic_multi_fmove_inv_tree(&rule, s1, SectorId::new(2), &domain, true).unwrap();
            assert_eq!(terms.len(), 1, "expected a single recoupled candidate");
            let (_, coeff) = &terms[0];
            assert_eq!(coeff.len(), 1, "coeff2 must be length-1 here");
            // Independent TK reading: inv coeff = conj(seed associator) = conj(w).
            assert!(
                (coeff[0] - w.conj()).norm() < 1e-12,
                "multi_Fmove_inv gave {} want conj(w)={} (A2 conj missing?)",
                coeff[0],
                w.conj()
            );
            // Discriminating: conj(w) must differ from w so the check has teeth.
            assert!((w - w.conj()).norm() > 0.1, "w not complex enough");
        }

        // Test B: foldright net observable = raw associator w (the two conjs cancel).
        // A single dropped conj at EITHER site would surface as conj(w).
        #[test]
        fn refute_b2b_foldright_net_is_raw_associator_complex() {
            let rule = Coeff2ConjRule;
            let s1 = SectorId::new(1);
            let s2 = SectorId::new(2);
            let s3 = SectorId::new(3);
            let w = c2c_w();
            // codomain [1,2] -> 3 (coeff1 = unit, A = Asymbol(1,2,3) real = 1),
            // domain [3,3] -> 3 (drives complex coeff2).
            let codomain =
                FusionTreeKey::new([s1, s2], s3, [false, false], [], [MultiplicityIndex::ONE]);
            let domain =
                FusionTreeKey::new([s3, s3], s3, [false, false], [], [MultiplicityIndex::ONE]);
            let pair = FusionTreePairKey::pair(codomain, domain);
            let out = generic_foldright_tree_pair(&rule, &pair).unwrap();
            assert_eq!(out.len(), 1, "expected a single folded term");
            let coeff = out[0].1;
            assert!(
                (coeff - w).norm() < 1e-12,
                "foldright net = {coeff} want raw associator w={w} (odd # of conj slips ⇒ conj(w))"
            );
            // The wrong (single-conj-dropped) answer is conj(w); prove distinguishable.
            assert!(
                (coeff - w.conj()).norm() > 0.1,
                "test cannot discriminate conj(w) from w"
            );
        }

        #[test]
        fn generic_full_key_block_composition_matches_per_source_replay() {
            let rule = Su3BendRule;
            let s42 = SectorId::new(1);
            let s31 = SectorId::new(2);
            let basis = (1..=2)
                .map(|mu| {
                    FusionTreePairKey::pair(
                        FusionTreeKey::new(
                            [s42, s31],
                            s31,
                            [false, false],
                            [],
                            [MultiplicityIndex::new(mu)
                                .expect("test multiplicity label is one-based")],
                        ),
                        FusionTreeKey::new([s31], s31, [false], [], []),
                    )
                })
                .collect::<Vec<_>>();
            let mut columns = DenseColumns::with_capacity(basis.len(), basis.len());
            for source in 0..basis.len() {
                let row = columns.push_empty_row();
                columns.row_mut(row)[source] = Some(1.0);
            }

            let (dst_basis, dst_columns) =
                compose_generic_block_terms(&rule, &basis, &columns, |rule, key| {
                    generic_bendright_tree_pair(rule, key)
                })
                .unwrap();

            let oracle = basis
                .iter()
                .map(|source| {
                    compose_generic_tree_pair_terms(
                        &rule,
                        vec![(source.clone(), 1.0)],
                        generic_bendright_tree_pair,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let mut expected_order = Vec::<FusionTreePairKey>::new();
            for rows in &oracle {
                for (key, _) in rows {
                    if !expected_order.iter().any(|existing| existing == key) {
                        expected_order.push(key.clone());
                    }
                }
            }
            assert_eq!(dst_basis, expected_order);
            assert_eq!(dst_columns.num_src, basis.len());
            assert_eq!(dst_columns.num_rows, dst_basis.len());
            assert_eq!(oracle.len(), basis.len());
            for (destination_row, destination) in dst_basis.iter().enumerate() {
                let domain_vertices = destination.domain_tree().vertices();
                assert_eq!(
                    domain_vertices.last().map(|index| index.get()),
                    Some(destination_row + 1)
                );
                for (source, oracle_row) in oracle.iter().enumerate() {
                    let got = dst_columns.row(destination_row)[source].unwrap_or(0.0);
                    let want = oracle_row
                        .iter()
                        .find_map(|(key, coeff)| (key == destination).then_some(*coeff))
                        .unwrap_or(0.0);
                    assert!((got - want).abs() < 1e-12);
                }
            }
        }

        #[test]
        fn checked_generic_cyclic_transpose_matches_legacy_rows() {
            let rule = A4FoldRule;
            let t = SectorId::new(3);
            let checked = InfallibleGeneric::new(&rule);
            for (pair, codomain, domain, label) in [
                (
                    FusionTreePairKey::pair(
                        a4f_rank3(3, 2, 1),
                        FusionTreeKey::new([t], t, [false], [], []),
                    ),
                    vec![1, 2, 3],
                    vec![0],
                    "clockwise",
                ),
                (
                    FusionTreePairKey::pair(
                        a4f_rank3(3, 1, 2),
                        FusionTreeKey::new([t], t, [false], [], []),
                    ),
                    vec![3, 0, 1],
                    vec![2],
                    "anticlockwise",
                ),
                (
                    FusionTreePairKey::pair(
                        FusionTreeKey::new(
                            [t, t, t],
                            t,
                            [true, false, true],
                            [t],
                            [
                                MultiplicityIndex::new(2).unwrap(),
                                MultiplicityIndex::ONE,
                            ],
                        ),
                        FusionTreeKey::new([t], t, [true], [], []),
                    ),
                    vec![2, 3, 0],
                    vec![1],
                    "multi-step dual flags",
                ),
            ] {
                let legacy = map_terms(
                    generic_transpose_tree_pair(&rule, &pair, &codomain, &domain).unwrap(),
                );
                let actual = map_terms(
                    generic_transpose_tree_pair_checked(&checked, &pair, &codomain, &domain).unwrap(),
                );
                assert_term_maps_eq(&actual, &legacy, label);
            }
        }

        #[test]
        fn checked_generic_cyclic_transpose_preserves_provider_and_shape_errors() {
            let pair = a4_pair_rank2(1);
            let provider_error = CheckedA4Spy {
                fail_f: Some(1),
                ..CheckedA4Spy::new()
            };
            assert_rigid_provider_error(
                generic_transpose_tree_pair_checked(
                    &provider_error,
                    &pair,
                    &[1, 2],
                    &[0],
                )
                .unwrap_err(),
                RigidSpyError::F,
            );

            let malformed = CheckedA4Spy {
                bad_a_f: true,
                ..CheckedA4Spy::new()
            };
            assert!(matches!(
                generic_transpose_tree_pair_checked(&malformed, &pair, &[1, 2], &[0]),
                Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })
            ));
        }

        #[test]
        fn fusion_tree_key_hash_is_consistent_with_eq() {
            let (distinct, shared, fresh) = key_hash_fixture();
            let base = &distinct[0];
            for copy in [&shared, &fresh] {
                assert_eq!(base, copy);
                assert_eq!(base.cached_hash_for_test(), copy.cached_hash_for_test());
                assert_eq!(fx_hash_of(base), fx_hash_of(copy));
            }
            for key in &distinct {
                let recomputed = fusion_tree_key_hash(
                    key.uncoupled(),
                    key.coupled(),
                    key.is_dual(),
                    key.innerlines(),
                    key.vertices(),
                );
                assert_eq!(key.cached_hash_for_test(), recomputed);
                // `Hash` writes exactly the cached value.
                let mut hasher = rustc_hash::FxHasher::default();
                hasher.write_u64(key.cached_hash_for_test());
                assert_eq!(fx_hash_of(key), hasher.finish());
            }
            // Pairwise set includes the shared-backing and fresh-backing
            // duplicates of the base key so the equal-key branch covers real
            // distinct objects, not only a key against itself.
            let mut pairwise = distinct.clone();
            pairwise.push(shared);
            pairwise.push(fresh);
            let mut equal_pairs_between_distinct_objects = 0;
            for (i, a) in pairwise.iter().enumerate() {
                for (j, b) in pairwise.iter().enumerate() {
                    if a == b {
                        assert_eq!(a.cached_hash_for_test(), b.cached_hash_for_test());
                        if i != j {
                            equal_pairs_between_distinct_objects += 1;
                        }
                    }
                }
            }
            assert_eq!(equal_pairs_between_distinct_objects, 6);
            assert_eq!(pairwise.iter().collect::<rustc_hash::FxHashSet<_>>().len(), distinct.len());
        }

        #[test]
        fn fusion_tree_key_ordering_matches_field_wise_comparison() {
            let (mut keys, shared, fresh) = key_hash_fixture();
            keys.push(shared);
            keys.push(fresh);
            let fields = |k: &FusionTreeKey| {
                (
                    k.uncoupled().to_vec(),
                    k.coupled(),
                    k.is_dual().to_vec(),
                    k.innerlines().to_vec(),
                    k.vertices().to_vec(),
                )
            };
            for a in &keys {
                for b in &keys {
                    assert_eq!(a.cmp(b), fields(a).cmp(&fields(b)));
                    assert_eq!(a.partial_cmp(b), Some(a.cmp(b)));
                    assert_eq!(a.cmp(b) == std::cmp::Ordering::Equal, a == b);
                }
            }
        }

        #[test]
        fn braided_fusion_tree_key_carries_a_fresh_hash() {
            let tree = FusionTreeKey::try_from_sector_ids([1, 1, 0], 0, [false, true, false], [0], [1, 1]).unwrap();
            let steps = [
                PreparedArtinStep { index: 1, inverse: false },
                PreparedArtinStep { index: 0, inverse: true },
            ];
            let (braided, _) = execute_unique_tree_braid_steps(&FermionParityFusionRule, &tree, steps).unwrap();
            assert_ne!(braided, tree);
            let recomputed = fusion_tree_key_hash(
                braided.uncoupled(),
                braided.coupled(),
                braided.is_dual(),
                braided.innerlines(),
                braided.vertices(),
            );
            assert_eq!(braided.cached_hash_for_test(), recomputed);
            // Hand oracle: (1⊗1→0)(0⊗0→0) with the vacuum leg braided to the
            // front becomes (0⊗1→1)(1⊗1→0); the dual flag travels with its leg.
            let expected = FusionTreeKey::try_from_sector_ids([0, 1, 1], 0, [false, false, true], [1], [1, 1]).unwrap();
            assert_eq!(braided, expected);
            assert_eq!(fx_hash_of(&braided), fx_hash_of(&expected));
        }

        #[test]
        fn fusion_tree_key_debug_excludes_cache() {
            let key = FusionTreeKey::try_from_sector_ids([3, 4], 5, [false, true], [], [1]).unwrap();
            assert_eq!(
                format!("{key:?}"),
                "FusionTreeKey { uncoupled: [SectorId(3), SectorId(4)], coupled: SectorId(5), \
                 is_dual: [false, true], innerlines: [], vertices: [MultiplicityIndex(1)] }"
            );
        }

        // Canary (#153) against silent growth of the hottest recoupling-plan key.
        // A smaller representation is allowed; growth requires re-checking the
        // compact Hash/Eq/Ord and allocation contracts.
        #[test]
        fn fusion_tree_key_size_has_not_silently_grown() {
            assert!(std::mem::size_of::<FusionTreeKey>() <= 264);
            assert_eq!(
                std::mem::size_of::<MultiplicityIndex>(),
                std::mem::size_of::<usize>()
            );
        }

        #[test]
        fn checked_recursive_product_preserves_channel_order_and_valid_multiplicity() {
            // What: recursive checked products retain the established
            // right-outer/left-inner channel order and valid nsymbol products.
            type Pair = ProductFusionRule<FibonacciFusionRule, FibonacciFusionRule>;
            type Triple = ProductFusionRule<Pair, FibonacciFusionRule>;

            let pair = Pair::new(FibonacciFusionRule, FibonacciFusionRule);
            let pair_tau = pair.try_encode_sector(SectorId::new(1), SectorId::new(1)).unwrap();
            let triple = Triple::new(pair, FibonacciFusionRule);
            let input = triple
                .try_encode_sector(pair_tau, SectorId::new(1))
                .unwrap();
            let checked = triple.try_fusion_channels(input, input).unwrap();
            let infallible = triple.fusion_channels(input, input);
            assert_eq!(checked, infallible);

            let mut expected = SectorVec::new();
            for right in [SectorId::new(0), SectorId::new(1)] {
                for pair_right in [SectorId::new(0), SectorId::new(1)] {
                    for pair_left in [SectorId::new(0), SectorId::new(1)] {
                        let pair_channel = triple
                            .left_rule()
                            .try_encode_sector(pair_left, pair_right)
                            .unwrap();
                        expected.push(triple.try_encode_sector(pair_channel, right).unwrap());
                    }
                }
            }
            assert_eq!(checked, expected);
            for coupled in checked {
                assert_eq!(
                    triple.try_nsymbol(input, input, coupled),
                    Ok(triple.nsymbol(input, input, coupled))
                );
            }
        }

        #[test]
        fn checked_sector_leg_dual_rejects_a_malformed_id_transactionally() {
            // What: a checked dual rejects an excluded raw ID without changing the
            // source leg.
            let source = SectorLeg::new([(excluded_u1_id(), 2), (u1(1), 3)], false);
            let before = source.clone();
            assert_eq!(
                source.try_dual(&U1FusionRule),
                Err(FusionAlgebraError::InvalidSector {
                    sector: excluded_u1_id()
                })
            );
            assert_eq!(source, before);

            #[cfg(target_pointer_width = "64")]
            {
                type Rule = ProductFusionRule<U1FusionRule, Z2FusionRule, TensorKitProductCodec>;
                let rule = Rule::new(U1FusionRule, Z2FusionRule);
                let min = TensorKitProductCodec::encode(excluded_u1_id(), z2_odd());
                let product = SectorLeg::new([(min, 1)], false);
                assert_eq!(
                    product.try_dual(&rule),
                    Err(FusionAlgebraError::InvalidSector {
                        sector: excluded_u1_id()
                    })
                );
            }
        }

        #[test]
        fn checked_enumeration_matches_infallible_for_simple_and_unique_rules() {
            // What: the checked SectorId enumerator reproduces the exact
            // CoupledFusionTrees set, order, and prepared layout keys of the
            // infallible hot path for a Simple (SU2) and a Unique (U1) provider —
            // the TensorKit tree-grid oracle plus a byte/order-identity regression.
            let _guard = crate::test_support::CACHE_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            let su2_space = FusionTreeHomSpace::from_sectors(
                [(su2(1), 2), (su2(1), 2)],
                [(su2(1), 2), (su2(1), 2)],
            );
            for space in [su2_space.codomain(), su2_space.domain()] {
                assert_eq!(
                    try_fusion_trees_by_coupled_for_space_checked(&SU2FusionRule, space).unwrap(),
                    fusion_trees_by_coupled_for_space(&SU2FusionRule, space),
                );
            }
            // Multi-block: SU2 spin-1/2 x spin-1/2 couples to spin 0 and spin 1.
            assert!(
                try_fusion_trees_by_coupled_for_space_checked(&SU2FusionRule, su2_space.codomain())
                    .unwrap()
                    .len()
                    >= 2
            );

            reset_core_intern_tables();
            let checked = su2_space
                .prepare_fusion_tree_layout_checked(&SU2FusionRule)
                .unwrap();
            let infallible = su2_space.prepare_fusion_tree_layout(&SU2FusionRule);
            assert_eq!(checked.keys(), infallible.keys());

            let u1_space = FusionTreeHomSpace::from_sectors(
                [(U1Irrep::new(1).sector_id(), 1), (U1Irrep::new(-1).sector_id(), 1)],
                [(U1Irrep::new(0).sector_id(), 1)],
            );
            for space in [u1_space.codomain(), u1_space.domain()] {
                assert_eq!(
                    try_fusion_trees_by_coupled_for_space_checked(&U1FusionRule, space).unwrap(),
                    fusion_trees_by_coupled_for_space(&U1FusionRule, space),
                );
            }

            // Dual legs change the stored FusionTreeKey duality flags; the checked
            // walk must still reproduce the infallible key set exactly.
            let dual_space = FusionProductSpace::new([
                SectorLeg::new([(su2(1), 2)], true),
                SectorLeg::new([(su2(1), 2)], false),
            ]);
            assert!(dual_space.legs().iter().any(SectorLeg::is_dual));
            assert_eq!(
                try_fusion_trees_by_coupled_for_space_checked(&SU2FusionRule, &dual_space).unwrap(),
                fusion_trees_by_coupled_for_space(&SU2FusionRule, &dual_space),
            );
        }
}
