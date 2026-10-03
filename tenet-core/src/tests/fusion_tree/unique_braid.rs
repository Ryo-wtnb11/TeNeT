use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct UncertifiedCustomSymbolsRule;

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
        if coupled.id() == 0 {
            3.0
        } else {
            5.0
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ComplexAsymmetricUniqueRule;

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

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
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

    assert_eq!(
        <Z2FusionRule as MultiplicityFreeFusionSymbols>::Scalar::one(),
        1.0
    );
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

    let terms = multiplicity_free_braid_tree(&PlanarZ2Rule, &tree, &[1, 0], &[0, 1]).unwrap();
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

    let err = multiplicity_free_braid_tree(&PlanarZ2Rule, &tree, &[1, 0], &[0, 1]).unwrap_err();

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
    let tree = FusionTreeKey::try_from_sector_ids([1, 0], 1, [true, false], [], [1]).unwrap();

    for (lhs, rhs) in [(&unit, &tree), (&tree, &unit)] {
        let terms = merge_fusion_trees_multiplicity_free(&Z2FusionRule, lhs, rhs, SectorId::new(1))
            .unwrap();
        assert_eq!(terms, vec![(tree.clone(), 1.0)]);
    }
}

#[test]
fn merge_fusion_trees_pins_nontrivial_su2_f_coefficients() {
    // What: merging a spin-1 tree into three spin-1/2 leaves is a genuine
    // associator expansion, not a pointed-rule relabelling.
    let lhs = FusionTreeKey::try_from_sector_ids([1, 1], 2, [false, false], [], [1]).unwrap();
    let rhs = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false; 3], [0], [1, 1]).unwrap();

    let terms =
        merge_fusion_trees_multiplicity_free(&SU2FusionRule, &lhs, &rhs, SectorId::new(1)).unwrap();

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
        multiplicity_free_braid_tree(&FermionParityFusionRule, &tree, &[1, 0], &[0, 1]).unwrap();
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
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1])
        .unwrap();

    let terms =
        multiplicity_free_braid_tree(&FermionParityFusionRule, &tree, &[1, 0, 2], &[0, 1, 2])
            .unwrap();
    assert_eq!(terms.len(), 1);
    let (braided, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, -1.0);
    assert_eq!(
        braided.uncoupled(),
        &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
    );
    assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
    assert_eq!(
        braided.vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
}

#[test]
fn unique_artin_braid_at_updates_innerline_for_later_unit_crossing() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 0, 1], 0, [false, false, true], [1], [1, 1])
        .unwrap();

    let terms = multiplicity_free_braid_tree(&PlanarZ2Rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
    assert_eq!(terms.len(), 1);
    let (braided, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, 1.0);
    assert_eq!(
        braided.uncoupled(),
        &[SectorId::new(1), SectorId::new(1), SectorId::new(0)]
    );
    assert_eq!(braided.is_dual(), &[false, true, false]);
    assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
    assert_eq!(
        braided.vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
}

#[test]
fn unique_artin_braid_at_uses_f_and_r_symbols_for_later_crossing() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, true, false], [0], [1, 1])
        .unwrap();

    let terms =
        multiplicity_free_braid_tree(&FermionParityFusionRule, &tree, &[0, 2, 1], &[0, 1, 2])
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
    assert_eq!(
        braided.vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
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
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1])
        .unwrap();

    let terms =
        multiplicity_free_braid_tree(&FermionParityFusionRule, &tree, &[2, 0, 1], &[0, 1, 2])
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
    assert_eq!(
        braided.vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
}

#[test]
fn unique_braid_tree_uses_inverse_artin_branch_from_levels() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();

    let forward_terms =
        multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[0, 1]).unwrap();
    assert_eq!(forward_terms.len(), 1);
    let (braided_forward, forward) = forward_terms.into_iter().next().unwrap();
    let inverse_terms =
        multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[1, 0]).unwrap();
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
        multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &levels).unwrap();
    assert_eq!(forward_terms.len(), 1);
    let (forward_tree, forward_coeff) = forward_terms.into_iter().next().unwrap();
    let inverse_terms =
        multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &reflected_levels)
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
        [1, 1, 0, 1],
        1,
        [false, true, false, true],
        [0, 0],
        [1, 1, 1],
    )
    .unwrap();
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
