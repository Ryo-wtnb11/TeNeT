use super::*;

#[test]
fn coupled_sector_dimensions_cover_empty_and_multiplicity_free_products() {
    // What: product-space dimensions use the tensor unit for rank zero,
    // annihilate on an empty leg, and reproduce U1/SU2 fusion dimensions.
    let empty = FusionProductSpace::new(std::iter::empty::<SectorLeg>());
    assert_eq!(
        empty
            .coupled_sector_block_dimensions(&U1FusionRule)
            .unwrap(),
        BTreeMap::from([(u1(0), 1)])
    );

    let empty_leg = FusionProductSpace::new([SectorLeg::new(
        std::iter::empty::<(SectorId, usize)>(),
        false,
    )]);
    assert!(empty_leg
        .coupled_sector_block_dimensions(&U1FusionRule)
        .unwrap()
        .is_empty());

    let u1_leg = SectorLeg::new([(u1(0), 1), (u1(1), 1)], false);
    let u1_product = FusionProductSpace::new([u1_leg.clone(), u1_leg]);
    assert_eq!(
        u1_product
            .coupled_sector_block_dimensions(&U1FusionRule)
            .unwrap(),
        BTreeMap::from([(u1(0), 1), (u1(1), 2), (u1(2), 1)])
    );

    let half = SectorLeg::new([(su2(1), 1)], false);
    let su2_product = FusionProductSpace::new([half.clone(), half]);
    assert_eq!(
        su2_product
            .coupled_sector_block_dimensions(&SU2FusionRule)
            .unwrap(),
        BTreeMap::from([(su2(0), 1), (su2(2), 1)])
    );
}

#[test]
fn coupled_sector_dimensions_keep_outward_labels_for_dual_legs() {
    // What: the pivotal dual flag does not dualize an already-outward U1
    // sector label a second time.
    let product = FusionProductSpace::new([SectorLeg::new([(u1(-3), 2)], true)]);
    assert_eq!(
        product
            .coupled_sector_block_dimensions(&U1FusionRule)
            .unwrap(),
        BTreeMap::from([(u1(-3), 2)])
    );
}

#[derive(Clone, Copy, Debug)]
struct BranchingMultiplicityFreeRule;

impl FusionRule for BranchingMultiplicityFreeRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        match sector.id() {
            3 => SectorId::new(1),
            other => SectorId::new(other),
        }
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (1, 1) => smallvec![SectorId::new(0), SectorId::new(2)],
            (1, 2) | (2, 1) => smallvec![SectorId::new(1), SectorId::new(3)],
            (2, 2) => smallvec![SectorId::new(0)],
            _ => SmallVec::new(),
        }
    }
}

impl MultiplicityFreeFusionRule for BranchingMultiplicityFreeRule {}

#[derive(Clone, Copy, Debug)]
struct UnsortedFusionIteratorOrderRule;

impl FusionRule for UnsortedFusionIteratorOrderRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (1, 1) => smallvec![SectorId::new(2), SectorId::new(0)],
            (1, 2) | (2, 1) => smallvec![SectorId::new(1)],
            (2, 2) => smallvec![SectorId::new(0)],
            _ => SmallVec::new(),
        }
    }
}

impl MultiplicityFreeFusionRule for UnsortedFusionIteratorOrderRule {}

fn fusion_tree_pair_order(keys: &[FusionTreePairKey]) -> Vec<(Vec<usize>, Vec<usize>, usize)> {
    keys.iter()
        .map(|key| {
            (
                sector_ids(key.codomain_uncoupled()),
                sector_ids(key.domain_uncoupled()),
                key.coupled().id(),
            )
        })
        .collect()
}

#[test]
fn checked_revalidation_preserves_complete_admission() {
    // What: adding finite-algebra proof to canonical built-in Complete
    // spaces preserves both their layouts and complete-grid admission.
    fn assert_rule<R>(rule: &R, sector: SectorId)
    where
        R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
    {
        let space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::from_sectors([(sector, 1)], [(sector, 1)]),
            rule,
            [vec![1, 1]],
        )
        .unwrap();
        let structure = Arc::clone(space.subblock_structure());
        assert!(matches!(
            space.admission(),
            FusionSpaceAdmission::Complete(_)
        ));
        let checked = space.try_bind_rule_checked(rule).unwrap();
        assert!(matches!(
            checked.admission(),
            FusionSpaceAdmission::Complete(_)
        ));
        assert!(Arc::ptr_eq(&structure, checked.subblock_structure()));
    }

    assert_rule(&Z2FusionRule, z2_even());
    assert_rule(&FermionParityFusionRule, z2_odd());
    assert_rule(&U1FusionRule, u1(7));
    assert_rule(&SU2FusionRule, su2(3));
    assert_rule(&FibonacciFusionRule, SectorId::new(1));

    type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type Triple = ProductFusionRule<Fz2U1, SU2FusionRule>;
    let pair = Fz2U1::new(FermionParityFusionRule, U1FusionRule);
    let pair_sector = pair.encode_sector(z2_odd(), u1(2));
    let triple = Triple::new(pair, SU2FusionRule);
    let triple_sector = triple.encode_sector(pair_sector, su2(1));
    assert_rule(&triple, triple_sector);
}

#[test]
fn tree_pair_block_apis_reject_mixed_fusion_tree_groups_before_symbols() {
    let base = tree_pair_group_fixture(&[1, 2], &[3], 3, &[false, false], &[false]);
    let mixed = [
        tree_pair_group_fixture(&[1, 4], &[5], 5, &[false, false], &[false]),
        tree_pair_group_fixture(&[1, 2], &[3], 3, &[false, true], &[false]),
        tree_pair_group_fixture(&[1, 2], &[3], 3, &[false, false], &[true]),
    ];

    for other in mixed {
        let keys = [base.clone(), other];
        let snapshot = keys.clone();
        assert_mixed_tree_pair_block_group_is_rejected(
            &IdentitySymbolPanicRule,
            &keys,
            CoreError::MalformedFusionTree {
                message: TREE_PAIR_BLOCK_GROUP_ERROR,
            },
        );
        // What: validation errors do not alter caller-owned source keys.
        assert_eq!(keys, snapshot);
    }
}

#[test]
fn simple_prepared_braid_has_runtime_rank_schedule() {
    // What: a nontrivial SimpleFusion braid keeps the complete high-rank
    // Artin schedule through compilation without a rank-specialized plan.
    let rank = 19;
    let permutation = (0..rank).rev().collect::<Vec<_>>();
    let levels = (0..rank).collect::<Vec<_>>();
    let prepared = PreparedTreePairOperation::prepare_braid(
        &SU2FusionRule,
        rank,
        0,
        &permutation,
        &[],
        &levels,
        &[],
    )
    .unwrap();

    assert!(matches!(prepared.plan, PreparedTreePairPlan::Braid(_)));
    let prepared = prepared.into_compiler_owned_simple_braid();
    assert!(matches!(
        prepared.plan,
        PreparedTreePairPlan::SimpleBraid(_)
    ));
    assert_eq!(
        prepared.plan.artin_steps().unwrap().count(),
        rank * (rank - 1) / 2
    );
}

#[test]
fn borrowed_unique_artin_lowering_matches_materialized_replay() {
    // What: operation-local inverse positions preserve the former
    // materialized Artin word and inverse flag for every small-rank
    // permutation, not only a selected reverse ordering.
    for rank in 0..=8 {
        let mut permutation = (0..rank).collect::<Vec<_>>();
        let levels = (0..rank).map(|index| (index * 7) % 11).collect::<Vec<_>>();
        loop {
            let mut raw_axis_positions = vec![usize::MAX; rank];
            for (position, &axis) in permutation.iter().enumerate() {
                raw_axis_positions[axis] = position;
            }
            let expected = PreparedTreeBraid::new(&permutation, &levels, rank)
                .unwrap()
                .artin_steps
                .into_vec();
            let prepared = PreparedTreePairOperation::prepare_braid_with_raw_axis_positions(
                &Z2FusionRule,
                rank,
                0,
                &permutation,
                &[],
                &levels,
                &[],
                &raw_axis_positions,
            )
            .unwrap();
            assert_eq!(
                prepared
                    .plan
                    .artin_steps()
                    .map(|steps| steps.collect::<Vec<_>>())
                    .unwrap_or_default(),
                expected,
                "rank-{rank} permutation {permutation:?} changed"
            );
            if !next_lexicographic_permutation(&mut permutation) {
                break;
            }
        }
    }
}

fn next_lexicographic_permutation(values: &mut [usize]) -> bool {
    let Some(pivot) = (0..values.len().saturating_sub(1))
        .rev()
        .find(|&index| values[index] < values[index + 1])
    else {
        return false;
    };
    let successor = (pivot + 1..values.len())
        .rev()
        .find(|&index| values[pivot] < values[index])
        .expect("increasing pivot has a successor");
    values.swap(pivot, successor);
    values[pivot + 1..].reverse();
    true
}

struct PermissiveOneBitLayout;

impl PackedSectorLayout for PermissiveOneBitLayout {
    const BITS: u32 = 1;

    fn validate(_sector: SectorId) -> Result<(), ProductSectorCodecError> {
        Ok(())
    }
}

struct ZeroBitLayout;

impl PackedSectorLayout for ZeroBitLayout {
    const BITS: u32 = 0;

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError> {
        (sector.id() == 0)
            .then_some(())
            .ok_or(ProductSectorCodecError::InvalidHighBits {
                sector,
                total_bits: 0,
            })
    }
}

struct FullWidthLayout;

impl PackedSectorLayout for FullWidthLayout {
    const BITS: u32 = usize::BITS;

    fn validate(_sector: SectorId) -> Result<(), ProductSectorCodecError> {
        Ok(())
    }
}

#[test]
fn packed_product_codec_reports_invalid_components_and_widths() {
    // What: malformed packed IDs and layouts wider than usize fail with a
    // typed reason instead of panicking, wrapping, or silently truncating.
    type FpU1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    let invalid_parity = FpU1Codec::encode_checked(SectorId::new(2), u1(0));
    assert!(matches!(
        invalid_parity,
        Err(ProductSectorCodecError::ComponentOutOfRange {
            component: ProductSectorComponent::Left,
            ..
        })
    ));

    type MaliciousCodec = PackedProductCodec<PermissiveOneBitLayout, Fz2SectorLayout>;
    assert!(matches!(
        MaliciousCodec::encode_checked(SectorId::new(2), SectorId::new(0)),
        Err(ProductSectorCodecError::ComponentOutOfRange {
            component: ProductSectorComponent::Left,
            ..
        })
    ));

    type TripleLayout =
        ProductSectorLayout<ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>, Su2SectorLayout>;
    assert_eq!(TripleLayout::BITS, 41);
    #[cfg(target_pointer_width = "64")]
    {
        let invalid_high_bits = SectorId::new(1usize << TripleLayout::BITS);
        assert!(matches!(
            TripleLayout::validate(invalid_high_bits),
            Err(ProductSectorCodecError::InvalidHighBits { .. })
        ));
    }

    type TooWide =
        PackedProductCodec<ProductSectorLayout<U1SectorLayout, U1SectorLayout>, U1SectorLayout>;
    assert!(matches!(
        TooWide::encode_checked(SectorId::new(0), SectorId::new(0)),
        Err(ProductSectorCodecError::WidthOverflow { .. })
    ));

    type FullThenZero = PackedProductCodec<FullWidthLayout, ZeroBitLayout>;
    let full = FullThenZero::encode_checked(SectorId::new(usize::MAX), SectorId::new(0)).unwrap();
    assert_eq!(full, SectorId::new(usize::MAX));
    assert_eq!(
        FullThenZero::decode_checked(full).unwrap(),
        (SectorId::new(usize::MAX), SectorId::new(0))
    );

    type ZeroThenParity = PackedProductCodec<ZeroBitLayout, Fz2SectorLayout>;
    assert_eq!(
        ZeroThenParity::encode_checked(SectorId::new(0), SectorId::new(1)).unwrap(),
        SectorId::new(1)
    );
}

#[test]
fn packed_and_tensorkit_codecs_remain_distinct_compatible_options() {
    // What: the expert Cantor codec keeps its historical IDs while the
    // packed codec has a distinct rule identity and round-trips the same
    // semantic components.
    type PackedCodec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type PackedRule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, PackedCodec>;
    type CantorRule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

    let parity = z2_odd();
    let charge = u1(2);
    let packed = PackedCodec::encode(parity, charge);
    let cantor = TensorKitProductCodec::encode(parity, charge);
    assert_ne!(packed, cantor);
    assert_eq!(PackedCodec::decode(packed), Some((parity, charge)));
    assert_eq!(
        TensorKitProductCodec::decode(cantor),
        Some((parity, charge))
    );
    assert_ne!(
        PackedRule::default().rule_identity(),
        CantorRule::default().rule_identity()
    );
}

#[test]
fn product_external_domain_sector_is_dualized_componentwise() {
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    let rule = FpU1Rule::default();
    let a = rule.encode_sector(z2_odd(), u1(2));
    let external_domain = rule.dual(a);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(a, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(a, 1)], false)]),
    );

    let keys = hom
        .fusion_tree_keys_from_external_sectors(&rule, &[a, external_domain])
        .unwrap();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].codomain_uncoupled(), &[a]);
    assert_eq!(keys[0].domain_uncoupled(), &[a]);
    assert_eq!(keys[0].coupled(), a);

    let err = hom
        .fusion_tree_keys_from_external_sectors(&rule, &[a, a])
        .unwrap_err();
    assert_eq!(
        err,
        CoreError::InvalidSector {
            sector: external_domain
        }
    );
}

#[test]
fn fusion_tensor_space_rejects_homspace_rank_mismatch() {
    let rule = Z2FusionRule;
    let dense = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
    let hom = FusionTreeHomSpace::from_sector_ids([(0, 1), (1, 1)], [(0, 1)]);

    let err =
        FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, &rule, [vec![1, 1], vec![1, 1]])
            .unwrap_err();

    assert_eq!(
        err,
        CoreError::StructureRankMismatch {
            expected: 1,
            actual: 2,
        }
    );
}

fn materialized_multiplicity_free_group_oracle<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> Vec<CoupledFusionTrees>
where
    R: MultiplicityFreeFusionRule,
{
    let mut grouped = Vec::<CoupledFusionTrees>::new();
    let mut index: FxHashMap<SectorId, usize> = FxHashMap::default();
    for tuple in materialized_leg_tuple_oracle(space) {
        let uncoupled = tuple.iter().map(|leg| leg.sector()).collect::<Vec<_>>();
        let is_dual = tuple.iter().map(|leg| leg.is_dual()).collect::<Vec<_>>();
        let effective = uncoupled.clone();
        for coupled in reachable_coupled_sectors(rule, &effective) {
            let trees =
                collect_fusion_trees_for_coupled(rule, &uncoupled, &is_dual, &effective, coupled);
            match index.get(&coupled) {
                Some(&i) => grouped[i].trees.extend(trees),
                None => {
                    index.insert(coupled, grouped.len());
                    grouped.push(CoupledFusionTrees { coupled, trees });
                }
            }
        }
    }
    grouped.sort_by_key(|group| group.coupled);
    grouped
}

fn materialized_multiplicity_free_key_oracle<R>(
    rule: &R,
    codomain: &FusionProductSpace,
    domain: &FusionProductSpace,
) -> Vec<FusionTreePairKey>
where
    R: MultiplicityFreeFusionRule,
{
    let codomain = materialized_multiplicity_free_group_oracle(rule, codomain);
    let domain = materialized_multiplicity_free_group_oracle(rule, domain);
    merge_generic_tree_groups(&codomain, &domain)
}

#[test]
fn fusion_tree_keys_match_materialized_cartesian_oracle_across_ranks_and_duals() {
    // What: public key enumeration keeps the old tuple order without
    // reusing the production visitor in the test oracle.
    let rule = Z4PointedRule;
    let leg = |sectors: &[usize], is_dual| {
        SectorLeg::new(
            sectors
                .iter()
                .copied()
                .map(|sector| (SectorId::new(sector), 1usize)),
            is_dual,
        )
    };

    let cases = [
        (
            "empty",
            FusionProductSpace::new([]),
            FusionProductSpace::new([]),
        ),
        (
            "rank1",
            FusionProductSpace::new([leg(&[1, 3], true)]),
            FusionProductSpace::new([leg(&[1, 3], false)]),
        ),
        (
            "rank2",
            FusionProductSpace::new([leg(&[0, 1], false), leg(&[2, 3], true)]),
            FusionProductSpace::new([leg(&[1, 2, 3], true)]),
        ),
        (
            "rank8",
            FusionProductSpace::new(
                (0..8).map(|axis| leg(&[axis % 4, (axis + 1) % 4], axis % 2 == 1)),
            ),
            FusionProductSpace::new([leg(&[0, 2], true)]),
        ),
    ];

    for (name, codomain, domain) in cases {
        let expected = materialized_multiplicity_free_key_oracle(&rule, &codomain, &domain);
        let hom = FusionTreeHomSpace::new(codomain, domain);
        assert_eq!(
            hom.fusion_tree_keys(&rule).as_ref(),
            expected.as_slice(),
            "{name}"
        );
    }
}

#[test]
fn fusion_tree_homspace_generates_canonical_coupled_sector_order() {
    let rule = BranchingMultiplicityFreeRule;
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1)], [(1, 1), (1, 1)]);

    let keys = hom.fusion_tree_keys(&rule);

    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].coupled(), SectorId::new(0));
    assert_eq!(keys[1].coupled(), SectorId::new(2));
    assert_eq!(
        keys[0].codomain_uncoupled(),
        &[SectorId::new(1), SectorId::new(1)]
    );
    assert_eq!(
        keys[0].domain_uncoupled(),
        &[SectorId::new(1), SectorId::new(1)]
    );
    assert!(keys[0].codomain_innerlines().is_empty());
    assert!(keys[0].domain_innerlines().is_empty());
    assert_eq!(keys[0].codomain_vertices(), &[MultiplicityIndex::ONE]);
    assert_eq!(keys[0].domain_vertices(), &[MultiplicityIndex::ONE]);

    let sector = hom.sector_structure(&rule).unwrap();
    let groups = sector.fusion_tree_groups();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].block_indices(), &[0, 1]);
    assert_eq!(
        groups[0].group_key(),
        &FusionTreeGroupKey::from_sector_ids([1, 1], [1, 1], [false, false], [false, false])
    );
}

#[test]
fn uncached_coupled_layout_probe_does_not_publish_identity_or_cache_state() {
    // What: the cold cost probe computes layout size/equality without
    // interning BlockStructure content or publishing a fusion-layout entry.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = Z2FusionRule;
    let leg = || {
        SectorLeg::new(
            [
                (Z2Irrep::EVEN.sector_id(), 2),
                (Z2Irrep::ODD.sector_id(), 3),
            ],
            false,
        )
    };
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let source = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&rule)
        .unwrap();

    crate::reset_block_structure_intern_calls();
    crate::reset_fusion_tree_layout_probe_side_effect_calls();
    crate::reset_hom_space_intern_calls();
    let (required_len, source_matches) = homspace
        .coupled_subblock_layout_probe_uncached(&rule, source.as_ref())
        .unwrap();

    assert_eq!(required_len, source.required_len().unwrap());
    assert!(source_matches);
    assert_eq!(crate::block_structure_intern_calls(), 0);
    assert_eq!(crate::fusion_tree_layout_probe_side_effect_calls(), (0, 0));
    assert_eq!(crate::hom_space_intern_calls(), 0);
    let cached_again = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&rule)
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&source, &cached_again));
}

#[test]
fn fusion_tree_homspace_compose_matches_nonselfdual_domain_convention() {
    // TensorKit: `A * B` needs `domain(A) == codomain(B)` as spaces, so
    // the stored legs pair verbatim even for non-self-dual sectors
    // (Julia check: `rand(U1Space(0=>1,1=>1) ← same) * itself` works).
    let rule = U1FusionRule;
    let physical = u1(1);
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(physical, 1)], false)]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(physical, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
    );

    let composed = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap();

    assert_eq!(composed.codomain().legs()[0].sectors(), &[u1(2)]);
    assert_eq!(composed.domain().legs()[0].sectors(), &[u1(3)]);
}

#[test]
fn fusion_tree_homspace_select_dualizes_axes_like_tensorkit() {
    let rule = U1FusionRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(1), 1)], false),
            SectorLeg::new([(u1(2), 1)], true),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(u1(3), 1)], false),
            SectorLeg::new([(u1(-5), 1)], true),
        ]),
    );

    let selected = hom.select(&rule, &[2, 0], &[1, 3]).unwrap();

    assert_eq!(selected.codomain().legs()[0].sectors(), &[u1(-3)]);
    assert!(selected.codomain().legs()[0].is_dual());
    assert_eq!(selected.codomain().legs()[1].sectors(), &[u1(1)]);
    assert!(!selected.codomain().legs()[1].is_dual());
    assert_eq!(selected.domain().legs()[0].sectors(), &[u1(-2)]);
    assert!(!selected.domain().legs()[0].is_dual());
    assert_eq!(selected.domain().legs()[1].sectors(), &[u1(-5)]);
    assert!(selected.domain().legs()[1].is_dual());
}

#[test]
fn fusion_tree_homspace_permute_requires_full_axis_permutation() {
    let rule = U1FusionRule;
    let hom = FusionTreeHomSpace::from_sectors([(u1(0), 1), (u1(1), 1)], [(u1(2), 1)]);

    let err = hom.permute(&rule, &[0], &[2]).unwrap_err();
    assert_eq!(
        err,
        CoreError::InvalidPermutation {
            permutation: vec![0, 2],
            rank: 3,
        }
    );

    let err = hom.permute(&rule, &[0, 0], &[2]).unwrap_err();
    assert_eq!(
        err,
        CoreError::InvalidPermutation {
            permutation: vec![0, 0, 2],
            rank: 3,
        }
    );
}

#[test]
fn fusion_tree_homspace_tensorcontract_preserves_canonical_compose() {
    let rule = U1FusionRule;
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
    );

    let expected = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap();
    let actual =
        FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[1], &[0], &[0, 1], 1)
            .unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn fusion_tree_homspace_tensorcontract_matches_tensorkit_structural_formula() {
    let rule = U1FusionRule;
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(5), 1)], false)]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(7), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
    );

    let lhs_permuted = lhs.permute(&rule, &[1], &[0]).unwrap();
    let rhs_permuted = rhs.permute(&rule, &[1], &[0]).unwrap();
    let expected = FusionTreeHomSpace::compose(&rule, &lhs_permuted, &rhs_permuted)
        .unwrap()
        .permute(&rule, &[0], &[1])
        .unwrap();
    let actual =
        FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[0], &[1], &[0, 1], 1)
            .unwrap();

    assert_eq!(actual, expected);
    assert_eq!(actual.codomain().legs()[0].sectors(), &[u1(-5)]);
    assert!(actual.codomain().legs()[0].is_dual());
    assert_eq!(actual.domain().legs()[0].sectors(), &[u1(-7)]);
    assert!(actual.domain().legs()[0].is_dual());
}

#[test]
fn fusion_tree_homspace_tensorcontract_accepts_output_permutation_structurally() {
    let rule = U1FusionRule;
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
    );

    let composed =
        FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[1], &[0], &[1, 0], 1)
            .unwrap();
    assert_eq!(composed.codomain().len(), 1);
    assert_eq!(composed.domain().len(), 1);
    assert_eq!(composed.codomain().legs()[0].sectors(), &[u1(-3)]);
    assert!(composed.codomain().legs()[0].is_dual());
    assert_eq!(composed.domain().legs()[0].sectors(), &[u1(-2)]);
    assert!(composed.domain().legs()[0].is_dual());
}

#[test]
fn direct_homspace_derivation_matches_old_sequence_for_supported_rules() {
    let mixed_leg =
        |sectors: &[(SectorId, usize)], dual| SectorLeg::new(sectors.iter().copied(), dual);

    let u1_sectors = [(u1(-2), 1), (u1(1), 2)];
    let u1_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([mixed_leg(&u1_sectors, false), mixed_leg(&u1_sectors, true)]),
        FusionProductSpace::new([mixed_leg(&u1_sectors, true), mixed_leg(&u1_sectors, false)]),
    );
    assert_direct_contract_matches_legacy(
        &U1FusionRule,
        &u1_hom,
        &u1_hom,
        &[3, 2],
        &[0, 1],
        &[2, 0, 3, 1],
        2,
    );

    let parity = [(SectorId::new(0), 2), (SectorId::new(1), 1)];
    let fz2_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([mixed_leg(&parity, false), mixed_leg(&parity, true)]),
        FusionProductSpace::new([mixed_leg(&parity, true), mixed_leg(&parity, false)]),
    );
    assert_direct_contract_matches_legacy(
        &FermionParityFusionRule,
        &fz2_hom,
        &fz2_hom,
        &[3, 2],
        &[0, 1],
        &[1, 3, 0, 2],
        2,
    );

    let su2 = [
        (SU2Irrep::from_twice_spin(0).sector_id(), 2),
        (SU2Irrep::from_twice_spin(1).sector_id(), 1),
    ];
    let su2_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([mixed_leg(&su2, false), mixed_leg(&su2, true)]),
        FusionProductSpace::new([mixed_leg(&su2, true), mixed_leg(&su2, false)]),
    );
    assert_direct_contract_matches_legacy(
        &SU2FusionRule,
        &su2_hom,
        &su2_hom,
        &[3, 2],
        &[0, 1],
        &[3, 1, 2, 0],
        2,
    );

    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
    let product_rule = ProductRule::new(
        FpU1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    );
    let encode = |parity, charge, spin| {
        let inner =
            TensorKitProductCodec::encode(SectorId::new(parity), U1Irrep::new(charge).sector_id());
        TensorKitProductCodec::encode(inner, SU2Irrep::from_twice_spin(spin).sector_id())
    };
    let product = [(encode(0, 0, 0), 2), (encode(1, 1, 1), 1)];
    let product_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([mixed_leg(&product, false), mixed_leg(&product, true)]),
        FusionProductSpace::new([mixed_leg(&product, true), mixed_leg(&product, false)]),
    );
    assert_direct_contract_matches_legacy(
        &product_rule,
        &product_hom,
        &product_hom,
        &[3, 2],
        &[0, 1],
        &[2, 1, 3, 0],
        2,
    );
}

#[derive(Clone)]
struct DualCountingRule<R> {
    inner: R,
    dual_calls: Arc<AtomicUsize>,
}

impl<R> DualCountingRule<R>
where
    R: FusionRule,
{
    fn new(inner: R) -> Self {
        Self {
            inner,
            dual_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn reset_dual_calls(&self) {
        self.dual_calls.store(0, Ordering::Relaxed);
    }

    fn dual_calls(&self) -> usize {
        self.dual_calls.load(Ordering::Relaxed)
    }
}

impl<R> FusionRule for DualCountingRule<R>
where
    R: FusionRule,
{
    fn rule_identity(&self) -> RuleIdentity {
        self.inner.rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.inner.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.inner.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        self.inner.vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        self.dual_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.dual(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.inner.fusion_channels(left, right)
    }
}

impl<R> CheckedFusionAlgebra for DualCountingRule<R>
where
    R: CheckedFusionAlgebra,
{
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        self.dual_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.try_dual_sector(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        self.inner.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        self.inner.try_nsymbol(left, right, coupled)
    }
}

/// U(1) with a deliberately broken dual: charges in `collide` all map
/// to the dual of the first of them (not injective), and `fail` has no
/// representable dual. Everything else is U(1).
#[derive(Clone)]
struct FaultyDualRule {
    collide: Vec<SectorId>,
    fail: Option<SectorId>,
}

impl FaultyDualRule {
    fn mapped(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        if Some(sector) == self.fail {
            return Err(FusionAlgebraError::InvalidSector { sector });
        }
        let sector = if self.collide.contains(&sector) {
            self.collide[0]
        } else {
            sector
        };
        U1FusionRule.try_dual_sector(sector)
    }
}

impl FusionRule for FaultyDualRule {
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
        U1FusionRule.vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        self.mapped(sector).unwrap()
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        U1FusionRule.fusion_channels(left, right)
    }
}

impl CheckedFusionAlgebra for FaultyDualRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        self.mapped(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        U1FusionRule.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        U1FusionRule.try_nsymbol(left, right, coupled)
    }
}

/// `A(v ← w)` contracted with `B(w ← x)` over `w`, output `(x ← v)`:
/// both open legs change side, so the matcher sees two dualized views.
fn crossing_contraction(v: SectorLeg, x: SectorLeg) -> (FusionTreeHomSpace, FusionTreeHomSpace) {
    let w = || SectorLeg::new([(u1(0), 2)], false);
    (
        FusionTreeHomSpace::new(FusionProductSpace::new([v]), FusionProductSpace::new([w()])),
        FusionTreeHomSpace::new(FusionProductSpace::new([w()]), FusionProductSpace::new([x])),
    )
}

/// The matcher's answer and the build path's answer for `expected`, both
/// unchecked and checked, plus whether the matcher materialized.
fn match_against_build<R: CheckedFusionAlgebra>(
    rule: &R,
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    expected: &FusionTreeHomSpace,
) -> (
    Result<bool, CheckedFusionSpaceError>,
    Result<bool, CheckedFusionSpaceError>,
    usize,
) {
    let args = (&[1usize][..], &[0usize][..], &[1usize, 0][..], 1usize);
    let built = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
        rule, lhs, rhs, args.0, args.1, args.2, args.3,
    )
    .map(|built| built == *expected);
    let before = DESCRIPTOR_MATERIALIZATIONS.get();
    let matched = FusionTreeHomSpace::try_tensorcontract_homspace_matches_checked(
        rule, lhs, rhs, args.0, args.1, args.2, args.3, expected,
    );
    let materialized = DESCRIPTOR_MATERIALIZATIONS.get() - before;
    (matched, built, materialized)
}

#[test]
fn homspace_matcher_falls_back_to_the_build_answer_on_every_unproven_leg() {
    let rule = U1FusionRule;
    let leg = |charges: std::ops::Range<i32>, dual| {
        SectorLeg::new(
            charges.map(|charge| (u1(charge), (charge.unsigned_abs() % 3) as usize + 1)),
            dual,
        )
    };
    let build = |lhs: &FusionTreeHomSpace, rhs: &FusionTreeHomSpace| {
        FusionTreeHomSpace::tensorcontract_homspace(&rule, lhs, rhs, &[1], &[0], &[1, 0], 1)
            .unwrap()
    };
    let unchecked =
        |lhs: &FusionTreeHomSpace, rhs: &FusionTreeHomSpace, expected: &FusionTreeHomSpace| {
            FusionTreeHomSpace::tensorcontract_homspace_matches(
                &rule,
                lhs,
                rhs,
                &[1],
                &[0],
                &[1, 0],
                1,
                expected,
            )
            .unwrap()
        };

    // What: 64 sectors on a dualized leg are proven without building;
    // 65 exceed the injectivity mask and fall back, with the same answer.
    for (sectors, materializations) in [(64, 0), (65, 1)] {
        let (lhs, rhs) = crossing_contraction(leg(0..3, false), leg(-32..sectors - 32, false));
        let expected = build(&lhs, &rhs);
        let (matched, built, materialized) = match_against_build(&rule, &lhs, &rhs, &expected);
        assert_eq!((matched, built), (Ok(true), Ok(true)), "{sectors} sectors");
        assert_eq!(materialized, materializations, "{sectors} sectors");
        assert!(unchecked(&lhs, &rhs, &expected));
    }

    // What: a degeneracy mismatch and a sector mismatch on a dualized
    // leg are rejected, after the same fallback build.
    let (lhs, rhs) = crossing_contraction(leg(0..3, false), leg(-2..3, false));
    let expected = build(&lhs, &rhs);
    let replaced = |edit: fn(&mut [(SectorId, usize)])| {
        let x = &expected.codomain().legs()[0];
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                {
                    let mut pairs = x.iter().collect::<Vec<_>>();
                    edit(&mut pairs);
                    pairs
                },
                x.is_dual(),
            )]),
            expected.domain().clone(),
        )
    };
    let degeneracy = replaced(|pairs| pairs[1].1 += 1);
    let sector = replaced(|pairs| pairs[4].0 = u1(7));
    for wrong in [&degeneracy, &sector] {
        let (matched, built, materialized) = match_against_build(&rule, &lhs, &rhs, wrong);
        assert_eq!((matched, built), (Ok(false), Ok(false)));
        assert_eq!(materialized, 1);
        assert!(!unchecked(&lhs, &rhs, wrong));
    }

    // What: a dual that is not injective on one leg is never taken as
    // proof: the checked matcher reports the build's DualNotInjective,
    // and the unchecked one panics exactly where the build panics.
    let collide = FaultyDualRule {
        collide: vec![u1(1), u1(2)],
        fail: None,
    };
    let (lhs, rhs) = crossing_contraction(leg(0..1, false), leg(0..4, false));
    let candidate = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 1), (u1(-1), 2), (u1(-3), 1)], true)]),
        FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], true)]),
    );
    let (matched, built, _) = match_against_build(&collide, &lhs, &rhs, &candidate);
    assert!(matches!(
        built,
        Err(CheckedFusionSpaceError::FusionAlgebra(ref error))
            if matches!(**error, FusionAlgebraError::DualNotInjective { .. })
    ));
    assert_eq!(matched, built);
    let panics =
        |run: &dyn Fn()| std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).is_err();
    assert!(panics(&|| {
        let _ = FusionTreeHomSpace::tensorcontract_homspace(
            &collide,
            &lhs,
            &rhs,
            &[1],
            &[0],
            &[1, 0],
            1,
        );
    }));
    assert!(panics(&|| {
        let _ = FusionTreeHomSpace::tensorcontract_homspace_matches(
            &collide,
            &lhs,
            &rhs,
            &[1],
            &[0],
            &[1, 0],
            1,
            &candidate,
        );
    }));

    // What: a dual error is the build path's first error, whether the
    // matcher meets it before or after an unproven leg.
    for fail in [u1(0), u1(1), u1(3)] {
        let failing = FaultyDualRule {
            collide: Vec::new(),
            fail: Some(fail),
        };
        let (lhs, rhs) = crossing_contraction(leg(0..2, false), leg(0..4, false));
        let proven = build(&lhs, &rhs);
        for expected in [&proven, &degeneracy] {
            let (matched, built, _) = match_against_build(&failing, &lhs, &rhs, expected);
            assert!(built.is_err(), "{fail:?}");
            assert_eq!(matched, built, "{fail:?}");
        }
    }
}

fn assert_oriented_validation_dual_calls_are_linear<R>(
    rule: &DualCountingRule<R>,
    sectors: impl IntoIterator<Item = (SectorId, usize)>,
    sector_count: usize,
) where
    R: FusionRule,
{
    let leg = SectorLeg::new(sectors, false);
    let lhs = OrientedLegView::borrowed(&leg).toggled();
    let rhs = OrientedLegView::borrowed(&leg).toggled();
    rule.reset_dual_calls();
    validate_oriented_composed_leg(rule, lhs, rhs, (0, 0)).unwrap();
    assert_eq!(rule.dual_calls(), 2 * sector_count);
}

#[test]
fn oriented_composed_leg_membership_is_linear_in_dual_operations() {
    const SECTORS: usize = 256;
    let u1_rule = DualCountingRule::new(U1FusionRule);
    assert_oriented_validation_dual_calls_are_linear(
        &u1_rule,
        (0..SECTORS).map(|index| (u1(index as i32 - 97), index % 5 + 1)),
        SECTORS,
    );

    let product_rule = DualCountingRule::new(ProductFusionRule::<
        U1FusionRule,
        FermionParityFusionRule,
        TensorKitProductCodec,
    >::new(U1FusionRule, FermionParityFusionRule));
    assert_oriented_validation_dual_calls_are_linear(
        &product_rule,
        (0..SECTORS).map(|index| {
            (
                TensorKitProductCodec::encode(u1(index as i32 - 113), SectorId::new(index % 2)),
                index % 7 + 1,
            )
        }),
        SECTORS,
    );
}

#[test]
fn oriented_composed_leg_invalid_path_preserves_legacy_error_order() {
    let rule = U1FusionRule;
    let lhs = SectorLeg::new([(u1(-3), 1), (u1(2), 2)], false);
    let rhs = SectorLeg::new([(u1(-3), 4), (u1(2), 2)], false);
    let lhs_view = OrientedLegView::borrowed(&lhs).toggled();
    let rhs_view = OrientedLegView::borrowed(&rhs).toggled();
    let expected = validate_composed_leg(
        &lhs_view.materialize(&rule),
        &rhs_view.materialize(&rule),
        (0, 0),
    );
    assert_eq!(
        validate_oriented_composed_leg(&rule, lhs_view, rhs_view, (0, 0)),
        expected
    );
}

#[test]
fn generic_select_matches_old_sequence() {
    let rule = UnitaryToyOmRule;
    let leg = SectorLeg::new(
        [
            (SectorId::new(UnitaryToyOmRule::VACUUM), 2),
            (SectorId::new(UnitaryToyOmRule::A), 1),
        ],
        false,
    );
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let expected = legacy_select(&rule, &hom, &[3, 0], &[2, 1]);
    let actual = hom.select(&rule, &[3, 0], &[2, 1]).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn explicit_shape_coupled_grid_reports_extent_overflow() {
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(0), usize::MAX)], false),
            SectorLeg::new([(u1(0), 2)], false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );

    let error = homspace
        .coupled_subblock_structure(&U1FusionRule, 2, [[usize::MAX, 2]])
        .unwrap_err();
    assert_eq!(error, CoreError::ElementCountOverflow);
}

fn assert_direct_generic_leg_degeneracy_structure_matches_legacy<R>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
) where
    R: FusionRule,
{
    let keys = homspace.fusion_tree_keys_generic(rule).unwrap();
    let blocks = keys
        .iter()
        .map(|key| {
            (
                key.clone(),
                homspace.degeneracy_shape_for_key(key).unwrap().to_vec(),
            )
        })
        .collect();
    let expected = BlockStructure::coupled_sector_matrix_with_keys(
        rule,
        homspace.codomain().len(),
        homspace.rank(),
        blocks,
    )
    .unwrap()
    .into_shared();
    let actual = homspace
        .coupled_subblock_structure_from_leg_degeneracies_generic(rule)
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.content_id(), expected.content_id());
}

#[test]
fn direct_generic_leg_degeneracy_layout_matches_legacy() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = UnitaryToyOmRule;
    let a = SectorId::new(UnitaryToyOmRule::A);
    let c = SectorId::new(UnitaryToyOmRule::C);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(a, 2)], false),
            SectorLeg::new([(a, 2)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c, 3)], false)]),
    );
    assert_direct_generic_leg_degeneracy_structure_matches_legacy(&rule, &homspace);
}

#[test]
fn canonical_coupled_grid_derives_each_row_and_column_once() {
    let rule = U1FusionRule;
    let leg = SectorLeg::new([(u1(-1), 2), (u1(0), 3), (u1(2), 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    reset_coupled_grid_build_observations();
    let layout = homspace.fusion_tree_layout_data_uncached(&rule);
    let expected_derivations = layout
        .sectors
        .iter()
        .map(|sector| sector.row_count + sector.col_count)
        .sum::<usize>();
    let actual = coupled_subblock_parts_from_leg_degeneracies(&homspace, &layout).unwrap();
    let expected = legacy_leg_degeneracy_structure(&rule, &homspace);
    assert_eq!(actual.0, *expected.sector_structure());
    assert_eq!(actual.1, *expected.degeneracy_structure());
    assert_eq!(coupled_grid_build_observations(), (0, expected_derivations));
}

#[test]
fn fusion_tree_homspace_compose_rejects_unmatched_contracted_sector() {
    // Pairing a domain leg with the *dual* codomain leg is a
    // SpaceMismatch in TensorKit (`(X ← V) * (V' ← Y)` fails).
    let rule = U1FusionRule;
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(rule.dual(u1(1)), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], false)]),
    );

    let err = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap_err();

    assert_eq!(
        err,
        CoreError::SectorMismatch {
            expected: u1(1),
            actual: rule.dual(u1(1)),
        }
    );
}

#[test]
fn unique_homspace_rejects_invalid_external_sector_tuple() {
    let rule = Z4PointedRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
    );

    let err = hom
        .unique_fusion_tree_key_from_external_sectors(&rule, &[SectorId::new(1), SectorId::new(2)])
        .unwrap_err();

    assert_eq!(
        err,
        CoreError::InvalidSector {
            sector: SectorId::new(2),
        }
    );
}

#[test]
fn fusion_tree_homspace_generates_innerline_paths_for_simple_fusion() {
    let rule = BranchingMultiplicityFreeRule;
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

    let keys = hom.fusion_tree_keys(&rule);

    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].coupled(), SectorId::new(1));
    assert_eq!(keys[1].coupled(), SectorId::new(1));
    assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
    assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
    assert_eq!(
        keys[0].codomain_vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
    assert!(keys[0].domain_innerlines().is_empty());
    assert!(keys[0].domain_vertices().is_empty());
    assert_eq!(keys[0].domain_uncoupled(), &[SectorId::new(1)]);

    let groups = hom.fusion_tree_groups(&rule).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].block_indices(), &[0, 1]);
}

#[test]
fn fusion_tree_homspace_matches_tensorkit_z2_fusiontreelist_order() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );

    let keys = hom.fusion_tree_keys(&rule);

    // TensorKit.jl 6Camk:
    // V=Vect[Z2Irrep](0=>1,1=>1); W=(V⊗V)←(V⊗V);
    // [(f1.uncoupled, f2.uncoupled, f1.coupled) for (f1,f2) in fusiontrees(W)]
    assert_eq!(
        fusion_tree_pair_order(&keys),
        vec![
            (vec![0, 0], vec![0, 0], 0),
            (vec![1, 1], vec![0, 0], 0),
            (vec![0, 0], vec![1, 1], 0),
            (vec![1, 1], vec![1, 1], 0),
            (vec![1, 0], vec![1, 0], 1),
            (vec![0, 1], vec![1, 0], 1),
            (vec![1, 0], vec![0, 1], 1),
            (vec![0, 1], vec![0, 1], 1),
        ]
    );

    let groups = hom.fusion_tree_groups(&rule).unwrap();
    assert_eq!(groups.len(), keys.len());
    for (index, group) in groups.iter().enumerate() {
        assert_eq!(group.block_indices(), &[index]);
    }
}

#[test]
fn fusion_tree_key_cache_hits_across_degeneracy_and_keeps_dual_signature() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = SU2FusionRule;
    let mk_leg = |degeneracy| {
        SectorLeg::new(
            [
                (SU2Irrep::from_twice_spin(0).sector_id(), degeneracy),
                (SU2Irrep::from_twice_spin(1).sector_id(), degeneracy + 1),
            ],
            false,
        )
    };
    let hom_small = FusionTreeHomSpace::new(
        FusionProductSpace::new([mk_leg(1), mk_leg(1)]),
        FusionProductSpace::new([mk_leg(1)]),
    );
    let hom_large = FusionTreeHomSpace::new(
        FusionProductSpace::new([mk_leg(4), mk_leg(4)]),
        FusionProductSpace::new([mk_leg(4)]),
    );

    let small_layout = hom_small.cached_fusion_tree_layout(&rule);
    let large_layout = hom_large.cached_fusion_tree_layout(&rule);
    assert!(Arc::ptr_eq(&small_layout, &large_layout));
    assert_eq!(small_layout.keys.as_ref(), large_layout.keys.as_ref());

    let dual_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([mk_leg(1).dual(&rule), mk_leg(1)]),
        FusionProductSpace::new([mk_leg(1)]),
    );
    let dual_layout = dual_hom.cached_fusion_tree_layout(&rule);
    assert!(!Arc::ptr_eq(&small_layout, &dual_layout));
    assert_ne!(small_layout.keys.as_ref(), dual_layout.keys.as_ref());
}

#[test]
fn fusion_layout_identity_hashes_inner_semantics_not_arc_address() {
    // What: independently allocated identity Arcs compare and hash by their
    // complete rule/sector/duality value, while distinct rules and splits do not alias.
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false),
            SectorLeg::new([(z2_even(), 1)], true),
        ]),
        FusionProductSpace::new([SectorLeg::new([(z2_odd(), 1)], false)]),
    );
    let first = Arc::new(FusionTreeHomSpaceCacheKey::new(&Z2FusionRule, &hom));
    let second = Arc::new(FusionTreeHomSpaceCacheKey::new(&Z2FusionRule, &hom));
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(first, second);

    let mut first_hash = rustc_hash::FxHasher::default();
    first.hash(&mut first_hash);
    let mut second_hash = rustc_hash::FxHasher::default();
    second.hash(&mut second_hash);
    assert_eq!(first_hash.finish(), second_hash.finish());

    let fermionic = Arc::new(FusionTreeHomSpaceCacheKey::new(
        &FermionParityFusionRule,
        &hom,
    ));
    assert_ne!(first, fermionic);

    let repartitioned = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false)]),
        FusionProductSpace::new([
            SectorLeg::new([(z2_even(), 1)], true),
            SectorLeg::new([(z2_odd(), 1)], false),
        ]),
    );
    let repartitioned = Arc::new(FusionTreeHomSpaceCacheKey::new(
        &Z2FusionRule,
        &repartitioned,
    ));
    assert_ne!(first, repartitioned);
}

#[test]
fn fusion_layout_global_churn_and_reset_preserve_coupled_structure() {
    // What: cap overflow gives the rebuilt layout a fresh non-recycling id;
    // coupled content remains equal, and reset cannot stale-alias live old values.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let rule = U1FusionRule;
    let hom = FusionTreeHomSpace::from_sectors([(U1Irrep::new(0), 2)], [(U1Irrep::new(0), 3)]);
    let old_layout = hom.cached_fusion_tree_layout(&rule);
    let old_id = old_layout.id;
    let old_structure = hom
        .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
        .unwrap();
    drop(old_layout);

    for charge in 1..=(FUSION_TREE_LAYOUT_CACHE_CAP as i32 + 64) {
        let distinct = FusionTreeHomSpace::from_sectors(
            [(U1Irrep::new(charge), 1)],
            [(U1Irrep::new(charge), 1)],
        );
        let _ = distinct.fusion_tree_keys(&rule);
    }
    let global_info = fusion_tree_layout_cache_info();
    assert!(global_info.entries() <= global_info.entry_capacity());
    assert!(global_info.charged_payload_bytes() <= global_info.byte_budget());

    let rebuilt_layout = hom.cached_fusion_tree_layout(&rule);
    assert!(rebuilt_layout.id > old_id);
    let reused_structure = hom
        .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
        .unwrap();
    assert_eq!(old_structure.as_ref(), reused_structure.as_ref());

    reset_core_intern_tables();
    let after_reset_layout = hom.cached_fusion_tree_layout(&rule);
    let after_reset_structure = hom
        .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
        .unwrap();
    assert!(!Arc::ptr_eq(&rebuilt_layout, &after_reset_layout));
    assert!(after_reset_layout.id > rebuilt_layout.id);
    assert!(!Arc::ptr_eq(&old_structure, &after_reset_structure));
    assert_eq!(old_structure.as_ref(), after_reset_structure.as_ref());
}

fn local_u1_layout(
    charge: i32,
) -> (
    Arc<FusionTreeHomSpaceCacheKey>,
    Arc<FusionTreeHomSpaceLayout>,
) {
    let rule = U1FusionRule;
    let hom =
        FusionTreeHomSpace::from_sectors([(U1Irrep::new(charge), 1)], [(U1Irrep::new(charge), 1)]);
    let key = Arc::new(FusionTreeHomSpaceCacheKey::new(&rule, &hom));
    let layout = Arc::new(fusion_tree_layout_from_data(
        next_fusion_tree_layout_id(),
        hom.fusion_tree_layout_data_uncached(&rule),
    ));
    (key, layout)
}

fn assert_checked_keys_match_encoded_oracle<R>(rule: &R, hom: &FusionTreeHomSpace)
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    let encoded_layout = hom.fusion_tree_layout_data_uncached(rule);
    let checked_layout = hom
        .try_fusion_tree_layout_data_uncached_checked(rule)
        .unwrap();
    assert_eq!(checked_layout.keys, encoded_layout.keys);
    assert_eq!(
        checked_layout.keys.as_ref(),
        hom.fusion_tree_keys_uncached(rule).as_slice()
    );
    assert_eq!(checked_layout.sectors.len(), encoded_layout.sectors.len());
    for (actual, expected) in checked_layout.sectors.iter().zip(&encoded_layout.sectors) {
        assert_eq!(actual.start, expected.start);
        assert_eq!(actual.row_count, expected.row_count);
        assert_eq!(actual.col_count, expected.col_count);
    }
}

#[test]
fn checked_builder_matches_encoded_oracle_for_builtin_ranks_and_products() {
    // What: every persistent key field and key order stays identical for
    // ranks 0 through 6 across all built-in multiplicity-free algebras.
    for rank in 0..=6 {
        assert_checked_keys_match_encoded_oracle(&U1FusionRule, &singleton_rank_hom(u1(1), rank));
        assert_checked_keys_match_encoded_oracle(
            &Z2FusionRule,
            &singleton_rank_hom(z2_odd(), rank),
        );
        assert_checked_keys_match_encoded_oracle(
            &FermionParityFusionRule,
            &singleton_rank_hom(z2_odd(), rank),
        );
        assert_checked_keys_match_encoded_oracle(&SU2FusionRule, &singleton_rank_hom(su2(1), rank));
    }

    type U1Fz2Codec = PackedProductCodec<U1SectorLayout, Fz2SectorLayout>;
    type U1Fz2Rule = ProductFusionRule<U1FusionRule, FermionParityFusionRule, U1Fz2Codec>;
    let pair_rule = U1Fz2Rule::new(U1FusionRule, FermionParityFusionRule);
    let pair_sector = U1Fz2Codec::encode(u1(1), z2_odd());

    type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
    type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
    type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;
    let triple_rule = TripleRule::new(
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    );
    let triple_sector = TripleCodec::encode(Fz2U1Codec::encode(z2_odd(), u1(1)), su2(1));
    let triple_pair_coupled = TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(2)), su2(0));

    for rank in 0..=6 {
        assert_checked_keys_match_encoded_oracle(
            &pair_rule,
            &singleton_rank_hom(pair_sector, rank),
        );
        assert_checked_keys_match_encoded_oracle(
            &triple_rule,
            &singleton_rank_hom(triple_sector, rank),
        );
    }

    let triple_vacuum = TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(0)), su2(0));
    let multi_tuple_side = |invert_dual| {
        FusionProductSpace::new((0..4).map(|axis| {
            SectorLeg::new(
                [(triple_vacuum, 1), (triple_sector, 2)],
                (axis % 2 == 0) ^ invert_dual,
            )
        }))
    };
    let multi_tuple_rank_eight =
        FusionTreeHomSpace::new(multi_tuple_side(false), multi_tuple_side(true));
    assert_checked_keys_match_encoded_oracle(&triple_rule, &multi_tuple_rank_eight);

    for dual_mask in 0usize..(1 << 3) {
        let all_dual_masks = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(triple_sector, 2)], dual_mask & 1 != 0),
                SectorLeg::new([(triple_sector, 3)], dual_mask & 2 != 0),
            ]),
            FusionProductSpace::new([SectorLeg::new(
                [(triple_pair_coupled, 4)],
                dual_mask & 4 != 0,
            )]),
        );
        assert_checked_keys_match_encoded_oracle(&triple_rule, &all_dual_masks);
    }

    let asymmetric = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(triple_sector, 2)], true),
            SectorLeg::new([(triple_sector, 3)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new(
                [(
                    TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(2)), su2(0)),
                    4,
                )],
                true,
            ),
            SectorLeg::new(
                [(
                    TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(0)), su2(0)),
                    5,
                )],
                false,
            ),
        ]),
    );
    assert_checked_keys_match_encoded_oracle(&triple_rule, &asymmetric);
}

#[test]
fn checked_and_encoded_entries_share_the_same_layout_cache() {
    // What: old-first and lowered-first construction converge on the same
    // Arc rather than publishing parallel layouts for one semantic key.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let hom = singleton_rank_hom(su2(1), 4);

    reset_core_intern_tables();
    let encoded_first = hom.cached_fusion_tree_layout(&SU2FusionRule);
    let checked_second = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap()
        .commit_layout();
    assert!(Arc::ptr_eq(&encoded_first, &checked_second));

    reset_core_intern_tables();
    let checked_first = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap()
        .commit_layout();
    let encoded_second = hom.cached_fusion_tree_layout(&SU2FusionRule);
    assert!(Arc::ptr_eq(&checked_first, &encoded_second));
}

#[test]
fn warm_layout_commit_takes_no_write_lock() {
    // What: publishing a layout takes the process-global writer exactly
    // once; committing the same layout again re-finds it under the read
    // lock, so concurrent warm calls do not serialize on that writer.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let hom = singleton_rank_hom(su2(1), 5);

    let before_cold = fusion_tree_layout_write_locks();
    let cold = hom.cached_fusion_tree_layout(&SU2FusionRule);
    assert_eq!(fusion_tree_layout_write_locks(), before_cold + 1);

    let warm = hom.cached_fusion_tree_layout(&SU2FusionRule);
    assert_eq!(fusion_tree_layout_write_locks(), before_cold + 1);
    assert!(Arc::ptr_eq(&cold, &warm));
}

#[test]
fn prepared_layout_publishes_only_at_commit() {
    // What: cold preparation enumerates exactly once but does not consume
    // identity or cache admission until its explicit commit point.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    reset_fusion_tree_layout_probe_side_effect_calls();
    let hom = singleton_rank_hom(su2(1), 5);

    let prepared = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap();
    // Why not inspect global cache totals: unrelated parallel tests may
    // populate the same process cache. These thread-local probes attribute
    // publication exactly to this transaction.
    assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));

    let keys = prepared.commit();
    assert!(!keys.is_empty());
    assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (1, 1));
}

#[test]
fn prepared_final_structure_reuses_one_checked_enumeration() {
    // What: cold checked preparation and the direct leg-degeneracy builder
    // match the established single-pass structure without a second
    // decode/channel enumeration or early cache publication.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let hom = singleton_rank_hom(su2(1), 5);
    reset_core_intern_tables();
    let expected = hom
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();

    reset_core_intern_tables();
    reset_fusion_tree_layout_probe_side_effect_calls();
    let prepared = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap();
    assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));

    let actual = prepared.build_from_leg_degeneracies(&hom).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));

    prepared.commit();
    assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (1, 1));
}

#[test]
fn prepared_complete_structure_hits_without_rebuilding_layout() {
    // Isolated like tenet-tensors #649/#650's checked_bind_failure
    // test: this asserts an absolute process-global cache admission
    // count right after a reset, which CACHE_TEST_LOCK does not
    // protect from the crate's many ordinary, unlocked complete-
    // structure builds landing in the same narrow window.
    if test_support::run_isolated_or_return(
        "TENET_CORE_PREPARED_COMPLETE_STRUCTURE_HITS_ISOLATED",
        "tests::fusion_space::prepared_complete_structure_hits_without_rebuilding_layout",
    ) {
        return;
    }
    // What: a repeated valid finalization validates its target
    // locally, then reuses the retained complete content without another
    // tree enumeration or cache admission.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let hom = singleton_rank_hom(su2(1), 5);

    let first_prepared = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap();
    let first = first_prepared
        .build_complete_from_leg_degeneracies(&hom)
        .unwrap();
    first_prepared.commit();
    let after_first = complete_hom_space_structure_cache_info();
    assert_eq!(after_first.admissions(), 1);

    let second_prepared = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap();
    let second = second_prepared
        .build_complete_from_leg_degeneracies(&hom)
        .unwrap();
    second_prepared.commit();
    let after_second = complete_hom_space_structure_cache_info();

    assert_eq!(first.content_id(), second.content_id());
    assert_eq!(after_second.hits(), after_first.hits() + 1);
    assert_eq!(after_second.admissions(), after_first.admissions());
}

fn finalize_complete<R>(
    rule: &R,
    hom: &FusionTreeHomSpace,
) -> Result<Arc<BlockStructure>, CoreError>
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    let prepared = hom.prepare_fusion_tree_layout_checked(rule).unwrap();
    let structure = prepared.build_complete_from_leg_degeneracies(hom)?;
    prepared.commit();
    Ok(structure)
}

/// Side derivations of one full extent walk over `hom`.
fn one_walk_side_derivations<R>(rule: &R, hom: &FusionTreeHomSpace) -> usize
where
    R: MultiplicityFreeFusionRule,
{
    hom.fusion_tree_layout_data_uncached(rule)
        .sectors
        .iter()
        .map(|sector| sector.row_count + sector.col_count)
        .sum()
}

#[test]
fn complete_structure_hit_skips_extent_walk_until_evicted() {
    // What: the miss walks the per-block extents exactly once (inside the
    // builder); a hit walks none; after FIFO eviction the next call is a
    // miss that walks exactly once again.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let leg = SectorLeg::new([(u1(-1), 2), (u1(0), 3), (u1(2), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let walk = one_walk_side_derivations(&U1FusionRule, &hom);
    assert!(walk > 2);

    reset_coupled_grid_build_observations();
    let first = finalize_complete(&U1FusionRule, &hom).unwrap();
    assert_eq!(coupled_grid_build_observations().1, walk);
    let after_miss = complete_hom_space_structure_cache_info();
    assert_eq!((after_miss.misses(), after_miss.admissions()), (1, 1));

    reset_coupled_grid_build_observations();
    let second = finalize_complete(&U1FusionRule, &hom).unwrap();
    assert_eq!(coupled_grid_build_observations().1, 0);
    assert!(Arc::ptr_eq(&first, &second));
    let after_hit = complete_hom_space_structure_cache_info();
    assert_eq!(after_hit.hits(), after_miss.hits() + 1);
    assert_eq!(after_hit.misses(), after_miss.misses());

    // Admit distinct neighbours until the first eviction, whichever bound
    // binds; FIFO makes `hom`, the oldest admission, its victim.
    let mut degeneracy = 10;
    while complete_hom_space_structure_cache_info().evictions() == after_hit.evictions() {
        let other = FusionTreeHomSpace::from_sectors([(u1(0), degeneracy)], [(u1(0), 1)]);
        finalize_complete(&U1FusionRule, &other).unwrap();
        degeneracy += 1;
        assert!(degeneracy <= after_hit.entry_capacity() + 10);
    }
    let filled = complete_hom_space_structure_cache_info();
    assert_eq!(filled.evictions(), after_hit.evictions() + 1);

    drop((first, second));
    reset_coupled_grid_build_observations();
    finalize_complete(&U1FusionRule, &hom).unwrap();
    assert_eq!(coupled_grid_build_observations().1, walk);
    let rewalked = complete_hom_space_structure_cache_info();
    assert_eq!(rewalked.misses(), filled.misses() + 1);
    assert_eq!(rewalked.admissions(), filled.admissions() + 1);
    assert_eq!(rewalked.hits(), filled.hits());
}

#[test]
fn complete_structure_split_and_fermionic_rule_force_misses() {
    // What: equal legs under another codomain/domain split, and equal
    // sectors under Z2 versus fermion parity, are misses that admit their
    // own entries rather than hits on a same-content neighbour.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let leg = SectorLeg::new([(u1(0), 2), (u1(1), 3)], false);
    let split = |nout: usize| {
        let legs = [leg.clone(), leg.clone(), leg.clone()];
        FusionTreeHomSpace::new(
            FusionProductSpace::new(legs[..nout].iter().cloned()),
            FusionProductSpace::new(legs[nout..].iter().cloned()),
        )
    };
    let two_one = finalize_complete(&U1FusionRule, &split(2)).unwrap();
    let before = complete_hom_space_structure_cache_info();
    let one_two = finalize_complete(&U1FusionRule, &split(1)).unwrap();
    let after = complete_hom_space_structure_cache_info();
    assert_ne!(two_one.content_id(), one_two.content_id());
    assert_eq!(after.misses(), before.misses() + 1);
    assert_eq!(after.admissions(), before.admissions() + 1);
    assert_eq!(after.hits(), before.hits());

    let parity_leg = SectorLeg::new([(z2_even(), 2), (z2_odd(), 3)], false);
    let parity_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([parity_leg.clone(), parity_leg.clone()]),
        FusionProductSpace::new([parity_leg]),
    );
    let bosonic = finalize_complete(&Z2FusionRule, &parity_hom).unwrap();
    let before = complete_hom_space_structure_cache_info();
    reset_coupled_grid_build_observations();
    let fermionic = finalize_complete(&FermionParityFusionRule, &parity_hom).unwrap();
    let after = complete_hom_space_structure_cache_info();
    // Equal content ids are expected: the block-structure interner keys
    // on rank and blocks only, so the rule shows up in the cache key.
    assert_eq!(bosonic.content_id(), fermionic.content_id());
    assert!(coupled_grid_build_observations().1 > 0);
    assert_eq!(after.misses(), before.misses() + 1);
    assert_eq!(after.admissions(), before.admissions() + 1);
    assert_eq!(after.entries(), before.entries() + 1);
    assert_eq!(after.hits(), before.hits());
}

#[test]
fn complete_structure_overflow_is_rejected_beside_cached_neighbour() {
    // What: an extent overflow whose sectors and duals equal a cached
    // valid neighbour is still walked and rejected without touching the
    // statistics, and the neighbour keeps hitting without a walk.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let hom = |first_degeneracy: usize| {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(0), first_degeneracy)], false),
                SectorLeg::new([(u1(0), 2)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(u1(0), 2)], false)]),
        )
    };
    let neighbour = hom(3);
    let cached = finalize_complete(&U1FusionRule, &neighbour).unwrap();
    let overflow = hom(usize::MAX);
    let before = complete_hom_space_structure_cache_info();

    let error = finalize_complete(&U1FusionRule, &overflow).unwrap_err();
    assert_eq!(error, CoreError::ElementCountOverflow);
    assert_eq!(complete_hom_space_structure_cache_info(), before);

    reset_coupled_grid_build_observations();
    let hit = finalize_complete(&U1FusionRule, &neighbour).unwrap();
    assert!(Arc::ptr_eq(&cached, &hit));
    assert_eq!(coupled_grid_build_observations().1, 0);
    assert_eq!(
        complete_hom_space_structure_cache_info().hits(),
        before.hits() + 1
    );
}

#[test]
fn prepared_lowered_final_structure_checks_signature_but_reads_target_degeneracies() {
    // What: a prepared layout rejects another same-rank sector signature
    // without publication, while the same sectors/duality with different
    // degeneracies are accepted as the target structure authority.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let source = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 3)], true)]),
    );
    let prepared = source
        .prepare_fusion_tree_layout_checked(&U1FusionRule)
        .unwrap();
    reset_fusion_tree_layout_probe_side_effect_calls();
    let mismatched = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(2), 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(2), 3)], true)]),
    );

    let error = prepared
        .build_from_leg_degeneracies(&mismatched)
        .unwrap_err();
    assert_eq!(
        error,
        CoreError::MalformedFusionTree {
            message: "prepared layout does not match HomSpace sector signature",
        }
    );
    assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));
    let duality_mismatched = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 2)], true)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 3)], true)]),
    );
    assert_eq!(
        prepared
            .build_from_leg_degeneracies(&duality_mismatched)
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "prepared layout does not match HomSpace sector signature",
        }
    );
    assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));

    let target = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 5)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 7)], true)]),
    );
    let structure = prepared.build_from_leg_degeneracies(&target).unwrap();
    assert_eq!(
        structure
            .degeneracy_structure()
            .blocks()
            .first()
            .unwrap()
            .shape(),
        &[5, 7]
    );
    assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));
}

#[test]
fn cached_commit_readmits_after_core_reset() {
    // What: a cached preparation that survives reset republishes its exact
    // retained keys without consuming a fresh layout identity.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let hom = singleton_rank_hom(su2(1), 5);
    hom.prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap()
        .commit();
    let prepared = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap();
    let retained = prepared.keys_arc();
    reset_core_intern_tables();
    reset_fusion_tree_layout_probe_side_effect_calls();

    let committed = prepared.commit();

    assert!(Arc::ptr_eq(&retained, &committed));
    assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 1));
}

#[test]
fn concurrent_commits_share_one_layout_admission() {
    // What: two cold preparations racing to commit converge on one Arc
    // and one cache miss without the losing transaction issuing an ID.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let hom = singleton_rank_hom(su2(1), 5);
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let handles = (0..2)
        .map(|_| {
            let hom = hom.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                reset_fusion_tree_layout_probe_side_effect_calls();
                let prepared = hom
                    .prepare_fusion_tree_layout_checked(&SU2FusionRule)
                    .unwrap();
                barrier.wait();
                let keys = prepared.commit();
                (keys, fusion_tree_layout_probe_side_effect_calls())
            })
        })
        .collect::<Vec<_>>();
    let mut results = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    let (second, second_calls) = results.pop().unwrap();
    let (first, first_calls) = results.pop().unwrap();

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(first_calls.0 + second_calls.0, 1);
    assert_eq!(first_calls.1 + second_calls.1, 1);
}

#[test]
fn checked_builder_reports_malformed_ids_and_algebra_closure_without_panicking() {
    // What: packed decode stays a codec error, while U(1), SU(2), and
    // recursive product closure failures retain exact causes.
    type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
    let rule = Rule::new(FermionParityFusionRule, U1FusionRule);
    let malformed = singleton_rank_hom(SectorId::new(usize::MAX), 1);
    let error = malformed
        .prepare_fusion_tree_layout_checked(&rule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::ProductCodec(
            Codec::decode_checked(SectorId::new(usize::MAX)).unwrap_err(),
        )
    );

    let invalid_z2 = singleton_rank_hom(SectorId::new(2), 1)
        .prepare_fusion_tree_layout_checked(&Z2FusionRule)
        .unwrap_err();
    assert_eq!(
        invalid_z2,
        FusionAlgebraError::InvalidSector {
            sector: SectorId::new(2),
        }
    );

    let u1_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(i32::MAX), 1)], false),
            SectorLeg::new([(u1(1), 1)], false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let error = u1_overflow
        .try_fusion_tree_layout_data_uncached_checked(&U1FusionRule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }
    );

    let invalid_u1_label = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(0), 1)], false),
            SectorLeg::new([(u1(0), 1)], false),
            SectorLeg::new([(excluded_u1_id(), 1)], false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let error = invalid_u1_label
        .try_fusion_tree_layout_data_uncached_checked(&U1FusionRule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::InvalidSector {
            sector: excluded_u1_id()
        }
    );

    let su2_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(su2(128), 1)], false),
            SectorLeg::new([(su2(127), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let error = su2_overflow
        .try_fusion_tree_layout_data_uncached_checked(&SU2FusionRule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::FusionNotRepresentable {
            left: su2(128),
            right: su2(127),
        }
    );

    let pair_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(Codec::encode(z2_even(), u1(i32::MAX)), 1)], false),
            SectorLeg::new([(Codec::encode(z2_odd(), u1(1)), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let error = pair_overflow
        .try_fusion_tree_layout_data_uncached_checked(&rule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }
    );

    type PairLayout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type TripleCodec = PackedProductCodec<PairLayout, Su2SectorLayout>;
    type TripleRule = ProductFusionRule<Rule, SU2FusionRule, TripleCodec>;
    let triple_rule = TripleRule::new(rule, SU2FusionRule);
    let triple = |parity, charge, spin| TripleCodec::encode(Codec::encode(parity, charge), spin);
    let triple_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(triple(z2_even(), u1(i32::MAX), su2(0)), 1)], false),
            SectorLeg::new([(triple(z2_odd(), u1(1), su2(1)), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let error = triple_overflow
        .try_fusion_tree_layout_data_uncached_checked(&triple_rule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }
    );
}

fn assert_failed_checked_build_is_transactional<R>(
    rule: &R,
    hom: &FusionTreeHomSpace,
    expected: FusionAlgebraError,
) where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    reset_fusion_tree_layout_probe_side_effect_calls();
    reset_hom_space_intern_calls();
    reset_block_structure_intern_calls();
    let error = hom.prepare_fusion_tree_layout_checked(rule).unwrap_err();
    assert_eq!(error, expected);
    assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));
    assert_eq!(hom_space_intern_calls(), 0);
    assert_eq!(block_structure_intern_calls(), 0);
}

#[test]
fn failed_checked_algebra_builds_publish_no_identity_or_intern_state() {
    // What: invalid built-in U1, SU2, and product closure leaves layout
    // identity/admission, HomSpace, and BlockStructure state untouched.
    let u1_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(i32::MAX), 1)], false),
            SectorLeg::new([(u1(1), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    assert_failed_checked_build_is_transactional(
        &U1FusionRule,
        &u1_overflow,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        },
    );

    let su2_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(su2(128), 1)], false),
            SectorLeg::new([(su2(127), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    assert_failed_checked_build_is_transactional(
        &SU2FusionRule,
        &su2_overflow,
        FusionAlgebraError::FusionNotRepresentable {
            left: su2(128),
            right: su2(127),
        },
    );

    type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
    let product_rule = Rule::new(FermionParityFusionRule, U1FusionRule);
    let product_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(Codec::encode(z2_even(), u1(i32::MAX)), 1)], false),
            SectorLeg::new([(Codec::encode(z2_odd(), u1(1)), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    assert_failed_checked_build_is_transactional(
        &product_rule,
        &product_overflow,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        },
    );
}

#[test]
fn empty_lowered_leg_short_circuits_before_other_leg_decode() {
    // What: an empty product has no tuples and returns empty even when a
    // different leg carries an ID that would fail lowered decoding.
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(SectorId::new(usize::MAX), 1)], false),
            SectorLeg::new(Vec::<(SectorId, usize)>::new(), false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let layout = hom
        .try_fusion_tree_layout_data_uncached_checked(&U1FusionRule)
        .unwrap();
    assert!(layout.keys.is_empty());
}

#[test]
fn fusion_layout_local_cache_is_strict_insertion_order_and_resets_exactly() {
    // What: read hits do not promote FIFO order; entry eviction and reset
    // update charged bytes and counters deterministically in isolated state.
    let mut cache = FusionTreeLayoutCache::new(2, 100, 100);
    let (key0, layout0) = local_u1_layout(40_000);
    let (key1, layout1) = local_u1_layout(40_001);
    let (key2, layout2) = local_u1_layout(40_002);
    cache.admit(Arc::clone(&key0), layout0, 30);
    cache.admit(Arc::clone(&key1), layout1, 30);
    assert!(cache.lookup(&key0).is_some());
    cache.admit(Arc::clone(&key2), layout2, 30);

    assert!(cache.lookup(&key0).is_none());
    assert!(cache.lookup(&key1).is_some());
    assert!(cache.lookup(&key2).is_some());
    assert_eq!(cache.info().entries(), 2);
    assert_eq!(cache.info().charged_payload_bytes(), 60);
    assert_eq!(cache.info().evictions(), 1);

    cache.clear();
    assert_eq!(cache.info().entries(), 0);
    assert_eq!(cache.info().charged_payload_bytes(), 0);
    assert_eq!(cache.info().misses(), 0);
    assert_eq!(cache.info().evictions(), 0);
    assert_eq!(cache.info().admission_bypasses(), 0);
}

#[test]
fn fusion_layout_local_cache_enforces_byte_and_max_entry_admission() {
    // What: charged-byte pressure evicts oldest entries, while an oversized
    // entry is returned to its caller but never retained by the cache.
    let mut cache = FusionTreeLayoutCache::new(8, 50, 40);
    let (key0, layout0) = local_u1_layout(50_000);
    let (key1, layout1) = local_u1_layout(50_001);
    let (oversized_key, oversized_layout) = local_u1_layout(50_002);
    cache.admit(Arc::clone(&key0), layout0, 30);
    cache.admit(Arc::clone(&key1), layout1, 30);

    assert!(cache.lookup(&key0).is_none());
    assert!(cache.lookup(&key1).is_some());
    assert_eq!(cache.info().charged_payload_bytes(), 30);
    assert_eq!(cache.info().evictions(), 1);

    let returned = cache.admit(
        Arc::clone(&oversized_key),
        Arc::clone(&oversized_layout),
        41,
    );
    assert!(Arc::ptr_eq(&returned, &oversized_layout));
    assert!(cache.lookup(&oversized_key).is_none());
    assert_eq!(cache.info().entries(), 1);
    assert_eq!(cache.info().charged_payload_bytes(), 30);
    assert_eq!(cache.info().admission_bypasses(), 1);
}

#[test]
fn fusion_layout_local_cache_bypasses_oversized_rule_identity() {
    #[derive(Clone)]
    struct OversizedIdentityRule {
        identity: RuleIdentity,
    }

    impl FusionRule for OversizedIdentityRule {
        fn rule_identity(&self) -> RuleIdentity {
            self.identity.clone()
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

    impl MultiplicityFreeFusionRule for OversizedIdentityRule {}

    // What: canonical rule bytes participate in admission accounting, so
    // an identity alone above the per-entry limit is computed but not retained.
    let canonical_bytes = Arc::<[u8]>::from(vec![
        0;
        FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES
            .saturating_add(1)
    ]);
    let rule = OversizedIdentityRule {
        identity: RuleIdentity::from_canonical_bytes::<OversizedIdentityRule>(0, canonical_bytes),
    };
    assert!(rule.identity.charged_retained_bytes() > FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES);
    let hom =
        FusionTreeHomSpace::from_sectors([(SectorId::new(0), 1)], Vec::<(SectorId, usize)>::new());
    let key = Arc::new(FusionTreeHomSpaceCacheKey::new(&rule, &hom));
    let layout = Arc::new(fusion_tree_layout_from_data(
        next_fusion_tree_layout_id(),
        hom.fusion_tree_layout_data_uncached(&rule),
    ));
    let charged_bytes = charged_fusion_tree_layout_bytes(&key, &layout);
    assert!(charged_bytes > FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES);

    let mut cache = FusionTreeLayoutCache::new(
        8,
        FUSION_TREE_LAYOUT_CACHE_BYTE_BUDGET,
        FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES,
    );
    let returned = cache.admit(Arc::clone(&key), Arc::clone(&layout), charged_bytes);
    assert!(Arc::ptr_eq(&returned, &layout));
    assert!(cache.lookup(&key).is_none());
    let info = cache.info();
    assert_eq!(info.entries(), 0);
    assert_eq!(info.admission_bypasses(), 1);
}

#[test]
fn fusion_layout_local_cache_bypasses_oversized_product_rule_identity() {
    #[derive(Clone)]
    struct OversizedProductIdentityRule {
        identity: RuleIdentity,
    }

    impl FusionRule for OversizedProductIdentityRule {
        fn rule_identity(&self) -> RuleIdentity {
            self.identity.clone()
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

    impl MultiplicityFreeFusionRule for OversizedProductIdentityRule {}

    struct ProductIdentityCodec;

    // What: a product identity retains the canonical bytes of both child
    // identities, so core admission continues to reject an oversized key.
    let canonical_bytes = Arc::<[u8]>::from(vec![
        0;
        FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES
            .saturating_add(1)
    ]);
    let rule = OversizedProductIdentityRule {
        identity: RuleIdentity::compose_with_codec::<ProductIdentityCodec>(
            RuleIdentity::from_canonical_bytes::<OversizedProductIdentityRule>(0, canonical_bytes),
            RuleIdentity::of_type::<Z2FusionRule>(),
        ),
    };
    assert!(rule.identity.charged_retained_bytes() > FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES);
    let hom =
        FusionTreeHomSpace::from_sectors([(SectorId::new(0), 1)], Vec::<(SectorId, usize)>::new());
    let key = Arc::new(FusionTreeHomSpaceCacheKey::new(&rule, &hom));
    let layout = Arc::new(fusion_tree_layout_from_data(
        next_fusion_tree_layout_id(),
        hom.fusion_tree_layout_data_uncached(&rule),
    ));
    let charged_bytes = charged_fusion_tree_layout_bytes(&key, &layout);
    assert!(charged_bytes > FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES);

    let mut cache = FusionTreeLayoutCache::new(
        8,
        FUSION_TREE_LAYOUT_CACHE_BYTE_BUDGET,
        FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES,
    );
    let returned = cache.admit(Arc::clone(&key), Arc::clone(&layout), charged_bytes);
    assert!(Arc::ptr_eq(&returned, &layout));
    assert!(cache.lookup(&key).is_none());
    let info = cache.info();
    assert_eq!(info.entries(), 0);
    assert_eq!(info.admission_bypasses(), 1);
}

#[test]
fn complete_homspace_layout_cache_is_fifo_bounded_and_bypasses_one_over_limit() {
    // What: complete immutable layouts retain at most the configured
    // charge, evict the oldest admission without read promotion, and
    // return oversized eager content without retaining it.
    let hom = FusionTreeHomSpace::from_sectors([(u1(0), 1)], Vec::<(SectorId, usize)>::new());
    let structure = BlockStructure::trivial(&[1]).unwrap().into_shared();
    let key = || {
        Arc::new(CompleteHomSpaceStructureCacheKey {
            rule: RuleIdentity::new_unique::<usize>(),
            homspace: Arc::clone(&hom.content),
        })
    };
    let key0 = key();
    let key1 = key();
    let key2 = key();
    let mut cache = CompleteHomSpaceStructureCache::new(2, 20, 10);

    // Charged bytes include the entry's `Weak<BlockStructure>` word
    // (`size_of::<CompleteHomSpaceStructureCacheEntry>()` grew from 2 to
    // 3 words, 16 -> 24 bytes on 64-bit) and the wrapper `ArcInner` the
    // Weak keeps allocated after the last strong owner dies.
    let charged = charged_complete_hom_space_structure_bytes(&key0, &structure.content_key());
    assert!(
        charged
            >= std::mem::size_of::<CompleteHomSpaceStructureCacheKey>()
                + 3 * std::mem::size_of::<usize>()
                + 12 * std::mem::size_of::<usize>()
                + std::mem::size_of::<BlockStructure>()
    );

    cache.admit_built(Arc::clone(&key0), Arc::clone(&structure), 10);
    cache.admit_built(Arc::clone(&key1), Arc::clone(&structure), 10);
    assert!(cache.peek_counting_hit(&key0).is_some());
    cache.admit_built(Arc::clone(&key2), Arc::clone(&structure), 10);
    assert!(cache.peek_counting_hit(&key0).is_none());
    assert!(cache.peek_counting_hit(&key1).is_some());
    assert!(cache.peek_counting_hit(&key2).is_some());
    assert_eq!(cache.info().entries(), 2);
    assert_eq!(cache.info().charged_bytes(), 20);
    assert_eq!(cache.info().evictions(), 1);

    let oversize = key();
    let returned = cache.admit_built(Arc::clone(&oversize), Arc::clone(&structure), 11);
    assert!(Arc::ptr_eq(&returned, &structure));
    assert!(cache.peek_counting_hit(&oversize).is_none());
    assert_eq!(cache.info().entries(), 2);
    assert_eq!(cache.info().bypasses(), 1);
    assert_eq!(cache.info().hits(), 3);
    // Misses are completed builds reaching admission, bypass included.
    assert_eq!(cache.info().misses(), 4);
    assert_eq!(cache.info().admissions(), 3);

    cache.clear();
    let cleared = cache.info();
    assert_eq!(cleared.entries(), 0);
    assert_eq!(cleared.charged_bytes(), 0);
    assert_eq!(cleared.hits(), 0);
    assert_eq!(cleared.misses(), 0);
    assert_eq!(cleared.admissions(), 0);
    assert_eq!(cleared.evictions(), 0);
    assert_eq!(cleared.bypasses(), 0);
}

#[test]
fn complete_homspace_layout_cache_bounds_bind_by_bytes_and_bypass_outliers() {
    // What: at the production bounds the byte budget, not the entry cap,
    // evicts first for entries of the smallest measured median size
    // (4455 bytes, #1365 census); the budget holds two maximum-size
    // entries; an entry above the limit is returned uncached, one at the
    // limit is retained; the global cache reports these bounds and reset
    // zeroes only its activity.
    let hom = FusionTreeHomSpace::from_sectors([(u1(0), 1)], Vec::<(SectorId, usize)>::new());
    let structure = BlockStructure::trivial(&[1]).unwrap().into_shared();
    let key = || {
        Arc::new(CompleteHomSpaceStructureCacheKey {
            rule: RuleIdentity::new_unique::<usize>(),
            homspace: Arc::clone(&hom.content),
        })
    };
    let mut cache = CompleteHomSpaceStructureCache::new(
        COMPLETE_HOM_SPACE_STRUCTURE_CACHE_CAP,
        COMPLETE_HOM_SPACE_STRUCTURE_CACHE_BYTE_BUDGET,
        COMPLETE_HOM_SPACE_STRUCTURE_CACHE_MAX_ENTRY_BYTES,
    );
    let typical = 4455;
    let fits = COMPLETE_HOM_SPACE_STRUCTURE_CACHE_BYTE_BUDGET / typical;
    assert!(fits < COMPLETE_HOM_SPACE_STRUCTURE_CACHE_CAP);
    for _ in 0..fits {
        cache.admit_built(key(), Arc::clone(&structure), typical);
    }
    assert_eq!(cache.info().evictions(), 0);
    cache.admit_built(key(), Arc::clone(&structure), typical);
    let info = cache.info();
    assert_eq!(info.evictions(), 1);
    assert_eq!(info.entries(), fits);
    assert!(info.charged_bytes() <= info.byte_budget());

    let outlier = key();
    let returned = cache.admit_built(
        Arc::clone(&outlier),
        Arc::clone(&structure),
        COMPLETE_HOM_SPACE_STRUCTURE_CACHE_MAX_ENTRY_BYTES + 1,
    );
    assert!(Arc::ptr_eq(&returned, &structure));
    assert!(cache.peek_counting_hit(&outlier).is_none());
    assert_eq!(cache.info().bypasses(), 1);
    cache.clear();
    let at_limit = [key(), key()];
    for limit_key in &at_limit {
        cache.admit_built(
            Arc::clone(limit_key),
            Arc::clone(&structure),
            COMPLETE_HOM_SPACE_STRUCTURE_CACHE_MAX_ENTRY_BYTES,
        );
    }
    assert!(at_limit
        .iter()
        .all(|limit_key| cache.peek_counting_hit(limit_key).is_some()));
    assert_eq!((cache.info().evictions(), cache.info().bypasses()), (0, 0));

    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let global = complete_hom_space_structure_cache_info();
    assert_eq!(global.entry_capacity(), 1024);
    assert_eq!(global.byte_budget(), 4 * 1024 * 1024);
    assert_eq!(
        global.max_entry_bytes(),
        COMPLETE_HOM_SPACE_STRUCTURE_CACHE_MAX_ENTRY_BYTES
    );
    assert_eq!(
        (
            global.entries(),
            global.charged_bytes(),
            global.hits(),
            global.misses()
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(
        (global.admissions(), global.evictions(), global.bypasses()),
        (0, 0, 0)
    );
}

#[test]
fn complete_homspace_layout_cache_reuses_semantic_content_and_excludes_regions() {
    // What: independently constructed complete multiplicity-free U1,
    // SU2, and product HomSpaces share frozen content by value; cached
    // content never owns a wrapper-local coupled-region state.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();

    fn assert_reused<R>(rule: &R, first_hom: FusionTreeHomSpace, second_hom: FusionTreeHomSpace)
    where
        R: MultiplicityFreeFusionRule,
    {
        let first = first_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        let info = complete_hom_space_structure_cache_info();
        let interned = block_structure_intern_calls();
        // Live wrapper: the hit returns the canonical Arc itself, without
        // re-interning content.
        let second = second_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let after = complete_hom_space_structure_cache_info();
        assert_eq!(after.hits(), info.hits() + 1);
        assert_eq!(after.misses(), info.misses());
        assert_eq!(after.admissions(), info.admissions());
        assert_eq!(block_structure_intern_calls(), interned);

        // Dead wrapper: content stays cached (a hit, not a miss), only the
        // wrapper is rebuilt; region state died with the old wrapper.
        let content = first.content_key();
        let region = first.weak_region_state();
        drop(first);
        drop(second);
        assert!(region.upgrade().is_none());
        let third = second_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        assert!(Arc::ptr_eq(&third.content_key(), &content));
        assert!(third.weak_region_state().upgrade().is_some());
        let after = complete_hom_space_structure_cache_info();
        assert_eq!(after.hits(), info.hits() + 2);
        assert_eq!(after.misses(), info.misses());
        assert_eq!(after.admissions(), info.admissions());
        assert_eq!(block_structure_intern_calls(), interned);
        // The entry's Weak was refreshed: the next hit is the rebuilt Arc.
        let fourth = first_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        assert!(Arc::ptr_eq(&third, &fourth));
        let region = third.weak_region_state();
        drop(third);
        drop(fourth);
        assert!(region.upgrade().is_none());
    }

    let u1_hom = || FusionTreeHomSpace::from_sectors([(u1(1), 2)], [(u1(1), 3)]);
    assert_reused(&U1FusionRule, u1_hom(), u1_hom());
    let su2_hom = || FusionTreeHomSpace::from_sectors([(su2(1), 2)], [(su2(1), 3)]);
    assert_reused(&SU2FusionRule, su2_hom(), su2_hom());

    type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type Product = ProductFusionRule<Fz2U1, SU2FusionRule>;
    let pair = Fz2U1::new(FermionParityFusionRule, U1FusionRule);
    let sector = pair.encode_sector(z2_odd(), u1(1));
    let rule = Product::new(pair, SU2FusionRule);
    let product_hom = || {
        FusionTreeHomSpace::from_sectors(
            [(rule.encode_sector(sector, su2(1)), 2)],
            [(rule.encode_sector(sector, su2(1)), 3)],
        )
    };
    assert_reused(&rule, product_hom(), product_hom());

    let info = complete_hom_space_structure_cache_info();
    assert_eq!(info.entries(), 3);
    assert_eq!(info.admissions(), 3);
    assert!(info.hits() >= 3);
}

#[test]
fn complete_homspace_layout_cache_concurrent_hits_share_one_canonical_arc() {
    // Isolated like tenet-tensors #649/#650's checked_bind_failure
    // test: this asserts absolute process-global admissions()/
    // misses()/hits() counts, which CACHE_TEST_LOCK does not protect
    // from the crate's many ordinary, unlocked complete-structure
    // builds landing in the same narrow window. Found flaking
    // (left: 3/2, right: 1 at line 2898-ish) during #1598/#1606
    // verification; same class as that pair's four named instances.
    if test_support::run_isolated_or_return(
            "TENET_CORE_COMPLETE_HOMSPACE_CONCURRENT_HITS_ISOLATED",
            "tests::fusion_space::complete_homspace_layout_cache_concurrent_hits_share_one_canonical_arc",
        ) {
            return;
        }
    // What: threads looking up one key concurrently, with and without a
    // live wrapper between rounds, all receive the same canonical Arc from
    // a single admission.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let hom = || FusionTreeHomSpace::from_sectors([(su2(1), 2), (su2(3), 1)], [(su2(1), 3)]);
    let held = hom()
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();

    let round = || {
        let barrier = std::sync::Barrier::new(4);
        let results = std::thread::scope(|scope| {
            let handles = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        hom()
                            .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results.iter().all(|r| Arc::ptr_eq(r, &results[0])));
        results
    };

    let live = round();
    assert!(Arc::ptr_eq(&live[0], &held));
    drop(live);
    drop(held);
    let rebuilt = round();
    // The refreshed entry pairs the rebuilt wrapper with its own content.
    let key = CompleteHomSpaceStructureCacheKey::new(&SU2FusionRule, &hom());
    let cache = complete_hom_space_structure_cache().read().unwrap();
    let entry = cache.entries.peek(&key).unwrap();
    assert!(Arc::ptr_eq(&entry.content, &rebuilt[0].content_key()));
    assert!(Arc::ptr_eq(&entry.wrapper.upgrade().unwrap(), &rebuilt[0]));
    drop(cache);
    assert_eq!(complete_hom_space_structure_cache_info().admissions(), 1);
    assert_eq!(complete_hom_space_structure_cache_info().misses(), 1);
    assert_eq!(complete_hom_space_structure_cache_info().hits(), 8);
    drop(rebuilt);
}

#[test]
fn complete_homspace_layout_cache_keys_semantics_and_preserves_direct_layout() {
    // Isolated like tenet-tensors #649/#650's checked_bind_failure
    // test: this asserts an absolute process-global entries() count,
    // which CACHE_TEST_LOCK does not protect from the crate's many
    // ordinary, unlocked complete-structure builds landing in the
    // same narrow window. Found flaking (left: 5, right: 4 at line
    // 2960-ish) during #1598/#1606 verification; same class as that
    // pair's four named instances.
    if test_support::run_isolated_or_return(
            "TENET_CORE_COMPLETE_HOMSPACE_KEYS_SEMANTICS_ISOLATED",
            "tests::fusion_space::complete_homspace_layout_cache_keys_semantics_and_preserves_direct_layout",
        ) {
            return;
        }
    // What: rule identity, degeneracies, and dual flags are distinct
    // complete-layout keys, while cache admission preserves the direct
    // builder's ordered block tuples and required storage length.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();

    let base = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], false)]),
    );
    let cached = base
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    let layout = base.fusion_tree_layout_data_uncached(&U1FusionRule);
    let (sector, degeneracy) =
        coupled_subblock_parts_from_leg_degeneracies(&base, &layout).unwrap();
    let direct = BlockStructure::from_parts(sector, degeneracy).unwrap();
    let signature = |structure: &BlockStructure| {
        (0..structure.block_count())
            .map(|index| {
                let block = structure.block(index).unwrap();
                (
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(signature(&cached), signature(&direct));
    assert_eq!(cached.required_len(), direct.required_len());

    let different_degeneracy = FusionTreeHomSpace::from_sectors([(u1(0), 4)], [(u1(0), 2)]);
    let dual = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], true)]),
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], false)]),
    );
    let rule_changed = FusionTreeHomSpace::from_sectors([(z2_even(), 2)], [(z2_even(), 2)]);
    different_degeneracy
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    dual.coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    rule_changed
        .coupled_subblock_structure_from_leg_degeneracies(&Z2FusionRule)
        .unwrap();
    assert_eq!(complete_hom_space_structure_cache_info().entries(), 4);

    let live = Arc::clone(&cached);
    reset_core_intern_tables();
    assert_eq!(live.required_len(), cached.required_len());
    assert_eq!(complete_hom_space_structure_cache_info().entries(), 0);
}

#[test]
fn fusion_layout_shape_and_fermionic_rule_provenance_do_not_alias() {
    // What: one sector layout may be shared across degeneracies, but concrete
    // shapes and bosonic/fermionic rule provenance select distinct structures/layouts.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false)]),
    );
    let bosonic_layout = hom.cached_fusion_tree_layout(&Z2FusionRule);
    let fermionic_layout = hom.cached_fusion_tree_layout(&FermionParityFusionRule);
    assert_ne!(bosonic_layout.id, fermionic_layout.id);

    let small = hom
        .coupled_subblock_structure(&FermionParityFusionRule, 1, [vec![1, 1], vec![1, 1]])
        .unwrap();
    let large_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 2), (z2_odd(), 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 2), (z2_odd(), 3)], false)]),
    );
    let large = large_hom
        .coupled_subblock_structure(&FermionParityFusionRule, 1, [vec![2, 2], vec![3, 3]])
        .unwrap();
    assert!(!Arc::ptr_eq(&small, &large));
    assert_ne!(small.as_ref(), large.as_ref());

    let transient_hom =
        FusionTreeHomSpace::from_sectors([(U1Irrep::new(17), 4)], [(U1Irrep::new(17), 5)]);
    let transient = transient_hom
        .coupled_subblock_structure(&U1FusionRule, 1, [vec![4, 5]])
        .unwrap();
    let expired = Arc::downgrade(&transient);
    drop(transient);
    assert!(expired.upgrade().is_none());
    let rebuilt = transient_hom
        .coupled_subblock_structure(&U1FusionRule, 1, [vec![4, 5]])
        .unwrap();
    assert_eq!(rebuilt.block(0).unwrap().shape(), &[4, 5]);
}

#[test]
fn fusion_tree_homspace_matches_tensorkit_su2_simple_order() {
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
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );

    let keys = hom.fusion_tree_keys(&rule);

    // TensorKit.jl 6Camk with sector id = twice spin:
    // V=Vect[SU2Irrep](0=>1,1//2=>1,1=>1); W=(V⊗V)←V;
    // [(2f1.uncoupled, 2f2.uncoupled, 2f1.coupled) for (f1,f2) in fusiontrees(W)]
    assert_eq!(
        fusion_tree_pair_order(&keys),
        vec![
            (vec![0, 0], vec![0], 0),
            (vec![1, 1], vec![0], 0),
            (vec![2, 2], vec![0], 0),
            (vec![1, 0], vec![1], 1),
            (vec![0, 1], vec![1], 1),
            (vec![2, 1], vec![1], 1),
            (vec![1, 2], vec![1], 1),
            (vec![2, 0], vec![2], 2),
            (vec![1, 1], vec![2], 2),
            (vec![0, 2], vec![2], 2),
            (vec![2, 2], vec![2], 2),
        ]
    );
    assert!(keys
        .iter()
        .all(|key| key.codomain_vertices() == [MultiplicityIndex::ONE]));
    assert!(keys.iter().all(|key| key.domain_vertices().is_empty()));
}

#[test]
fn ordered_transpose_preserves_custom_source_cohort_order() {
    let rule = FibonacciFAdmissibilityProbe::with_complex_f_phase();
    let tau = || SectorLeg::new([(SectorId::new(1), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([tau(), tau()]),
        FusionProductSpace::new([tau(), tau()]),
    );
    let mut sources = hom.fusion_tree_keys(&rule).as_ref().to_vec();
    assert!(sources.len() > 1);
    sources.reverse();

    // What: a caller-selected source subset/order determines destination
    // first appearance; canonical HomSpace basis order is not substituted.
    assert_compact_transpose_matches_full_key_oracle(&rule, &sources, &[1, 3], &[0, 2], true);
    assert_compact_transpose_matches_full_key_oracle(&rule, &sources[..2], &[1, 3], &[0, 2], true);
}

#[test]
fn transpose_tree_pair_block_matches_fz2_odd_pivotal_oracle() {
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(z2_odd(), 1)], true),
            SectorLeg::new([(z2_odd(), 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 1)], false)]),
    );
    let sources = hom.fusion_tree_keys(&FermionParityFusionRule);
    assert!(!sources.is_empty());

    // What: cycling a dual odd leg retains the Frobenius-Schur/pivotal sign.
    assert_compact_transpose_matches_full_key_oracle(
        &FermionParityFusionRule,
        &sources,
        &[1, 2],
        &[0],
        true,
    );
}

#[test]
fn fusion_tree_homspace_matches_tensorkit_su2_innerline_order() {
    let rule = SU2FusionRule;
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

    let keys = hom.fusion_tree_keys(&rule);

    // TensorKit.jl 6Camk with sector id = twice spin:
    // V=Vect[SU2Irrep](1//2=>1); W=(V⊗V⊗V)←V;
    // codomain innerlines for fusiontrees(W) are [0], then [2].
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
    assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
    assert_eq!(
        fusion_tree_pair_order(&keys),
        vec![(vec![1, 1, 1], vec![1], 1), (vec![1, 1, 1], vec![1], 1),]
    );

    let groups = hom.fusion_tree_groups(&rule).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].block_indices(), &[0, 1]);
}

#[test]
fn fusion_tree_homspace_external_sectors_preserve_su2_simple_innerline_order() {
    let rule = SU2FusionRule;
    let half = SectorId::new(1);
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

    let keys = hom
        .fusion_tree_keys_from_external_sectors(&rule, &[half, half, half, half])
        .unwrap();

    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].codomain_uncoupled(), &[half, half, half]);
    assert_eq!(keys[0].domain_uncoupled(), &[half]);
    assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
    assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
    assert_eq!(
        fusion_tree_pair_order(&keys),
        vec![(vec![1, 1, 1], vec![1], 1), (vec![1, 1, 1], vec![1], 1),]
    );
}

#[test]
fn fusion_tree_homspace_uses_tensorkit_parent_iterator_order_not_ord_sort() {
    let rule = UnsortedFusionIteratorOrderRule;
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

    let keys = hom.fusion_tree_keys(&rule);

    // TensorKit rank >= 3 iterator picks the parent line from
    // `coupled ⊗ dual(last)` order. This toy rule returns 1 ⊗ 1 as [2, 0],
    // deliberately opposite to `SectorId` Ord, so an Ord-based replay would
    // produce [0], [2].
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(2)]);
    assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(0)]);
}

#[test]
fn fusion_tree_homspace_uses_visible_dual_space_sector_label_like_tensorkit() {
    let rule = U1FusionRule;
    let minus_one = U1Irrep::new(-1);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(minus_one, 1)], true)]),
        FusionProductSpace::new([SectorLeg::new([(minus_one, 1)], false)]),
    );

    let keys = hom.fusion_tree_keys(&rule);

    // TensorKit:
    // collect(sectors(Vect[U1Irrep](1=>1)')) == [U1Irrep(-1)]
    // fusiontrees((U1Irrep(-1),), U1Irrep(-1), (true,)) keeps uncoupled = -1.
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].coupled(), minus_one.into());
    assert_eq!(keys[0].codomain_uncoupled(), &[minus_one.into()]);
    assert_eq!(keys[0].codomain_is_dual(), &[true]);
    assert_eq!(keys[0].domain_uncoupled(), &[minus_one.into()]);
    assert_eq!(keys[0].domain_is_dual(), &[false]);
}

#[test]
fn fusion_tree_homspace_does_not_dualize_selected_dual_leg_again() {
    let rule = BranchingMultiplicityFreeRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], true)]),
        FusionProductSpace::from_sector_ids([(1, 1)]),
    );

    let keys = hom.fusion_tree_keys(&rule);

    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].coupled(), SectorId::new(1));
    assert_eq!(keys[0].codomain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(keys[0].codomain_is_dual(), &[true]);
    assert_eq!(keys[0].domain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(keys[0].domain_is_dual(), &[false]);
}

#[test]
fn fusion_tree_homspace_fusionblocks_follow_domain_outer_codomain_inner_order() {
    let rule = BranchingMultiplicityFreeRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(SectorId::new(1), 1), (SectorId::new(2), 1)], false),
            SectorLeg::new([(SectorId::new(1), 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new(
            [(SectorId::new(1), 1), (SectorId::new(2), 1)],
            false,
        )]),
    );

    let groups = hom.fusion_tree_groups(&rule).unwrap();

    assert_eq!(groups.len(), 2);
    assert_eq!(
        groups[0].group_key(),
        &FusionTreeGroupKey::from_sector_ids([2, 1], [1], [false, false], [false])
    );
    assert_eq!(
        groups[1].group_key(),
        &FusionTreeGroupKey::from_sector_ids([1, 1], [2], [false, false], [false])
    );
}

#[test]
fn checked_generic_external_axis_leg_keeps_provider_boundary() {
    let rule = UnitaryToyOmRule;
    let a = SectorId::new(UnitaryToyOmRule::A);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(a, 1)], false)]),
        FusionProductSpace::new([]),
    );
    let oriented = OrientedFusionTreeHomSpace::new(&homspace, FusionTreePairOrientation::Direct);
    let leg = oriented
        .try_external_axis_leg_generic(&rule, 0)
        .unwrap()
        .unwrap();
    assert_eq!(leg.sectors(), &[a]);
    assert_eq!(leg.degeneracies(), &[1]);
    assert!(!leg.is_dual());
    let dual = leg.try_dual_generic(&rule).unwrap();
    assert_eq!(dual.sectors(), &[a]);
    assert!(dual.is_dual());
}

#[test]
fn hom_space_id_is_idempotent() {
    let build = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(1, 2, false)]),
            FusionProductSpace::new([u1_leg(1, 2, false)]),
        )
    };
    assert_eq!(build().id(), build().id());
}

#[test]
fn hom_space_clone_shares_content_but_not_unpublished_id_state() {
    // What: cloning reuses immutable HomSpace data while each handle keeps
    // its own lazy identity publication snapshot.
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, false)]),
        FusionProductSpace::new([u1_leg(-1, 3, true)]),
    );
    let before_id = hom.clone();

    assert!(Arc::ptr_eq(&hom.content, &before_id.content));
    assert!(before_id.existing_id().is_none());

    let id = hom.id();
    assert!(before_id.existing_id().is_none());

    let after_id = hom.clone();
    assert!(Arc::ptr_eq(&hom.content, &after_id.content));
    assert_eq!(after_id.existing_id(), Some(id));
}

#[test]
fn hom_space_id_separates_dual_flip() {
    // Rank-1 duality analog of the #119 regression: flipping one leg's dual
    // bit must not alias, on either the codomain or the domain side.
    let base = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, false)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    );
    let cod_dual = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, true)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    );
    let dom_dual = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, false)]),
        FusionProductSpace::new([u1_leg(1, 2, true)]),
    );
    assert_ne!(base.id(), cod_dual.id());
    assert_ne!(base.id(), dom_dual.id());
}

#[test]
fn hom_space_id_separates_sectors_degeneracy_and_rank() {
    let base = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, false)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    )
    .id();
    let other_charge = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(3, 2, false)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    )
    .id();
    let other_deg = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 5, false)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    )
    .id();
    let higher_rank = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, false), u1_leg(0, 2, false)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    )
    .id();
    assert_ne!(base, other_charge);
    assert_ne!(base, other_deg);
    assert_ne!(base, higher_rank);
}

#[test]
fn hom_space_id_remains_semantic_after_intern_eviction() {
    // What: floods the shared hom-space intern table past its cap, which
    // races `concurrent_equal_hom_spaces_share_semantic_identity` (asserts
    // ptr_eq on entries of that same table) if both run concurrently.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let build = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(17, 2, false)]),
            FusionProductSpace::new([u1_leg(17, 3, true)]),
        )
    };
    let before = build().id();
    for charge in 10_000..10_000 + HOM_SPACE_INTERN_CAP as i32 + 1 {
        let _ = FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(charge, 1, false)]),
            FusionProductSpace::new([u1_leg(charge, 1, false)]),
        )
        .id();
    }
    let after = build().id();
    assert!(!Arc::ptr_eq(&before.key, &after.key));
    assert_eq!(before, after);
    let hash = |id: &HomSpaceId| {
        let mut state = rustc_hash::FxHasher::default();
        id.hash(&mut state);
        std::hash::Hasher::finish(&state)
    };
    assert_eq!(hash(&before), hash(&after));
    assert_eq!(
        hom_space_intern_table().read().unwrap().entries.len(),
        HOM_SPACE_INTERN_CAP
    );
}

#[test]
fn eager_hom_space_derivation_does_not_touch_lazy_id_interner() {
    let rule = U1FusionRule;
    let leg = SectorLeg::new([(u1(-1), 2), (u1(2), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );

    let selected = hom.select(&rule, &[1, 0], &[3, 2]).unwrap();
    let permuted = hom.permute(&rule, &[1, 0], &[3, 2]).unwrap();
    let composed = FusionTreeHomSpace::compose(&rule, &hom, &hom).unwrap();
    let contracted = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        &hom,
        &hom,
        &[2, 3],
        &[0, 1],
        &[0, 1, 2, 3],
        2,
    )
    .unwrap();
    assert!(selected.existing_id().is_none());
    assert!(permuted.existing_id().is_none());
    assert!(composed.existing_id().is_none());
    assert!(contracted.existing_id().is_none());

    let selected_id = selected.id();
    assert_eq!(selected_id, permuted.id());
    assert_eq!(composed.id(), contracted.id());
}

#[test]
fn concurrent_eager_hom_space_derivation_does_not_touch_lazy_id_interner() {
    let rule = U1FusionRule;
    let leg = SectorLeg::new([(u1(-1), 2), (u1(2), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                for _ in 0..100 {
                    let permuted = hom.permute(&rule, &[1, 0], &[3, 2]).unwrap();
                    let contracted = FusionTreeHomSpace::tensorcontract_homspace(
                        &rule,
                        &hom,
                        &hom,
                        &[2, 3],
                        &[0, 1],
                        &[0, 1, 2, 3],
                        2,
                    )
                    .unwrap();
                    assert!(permuted.existing_id().is_none());
                    assert!(contracted.existing_id().is_none());
                }
            });
        }
    });
}

#[test]
fn resetting_lazy_hom_space_interner_preserves_semantic_identity() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_hom_space_intern_table();
    let build = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(23, 2, false)]),
            FusionProductSpace::new([u1_leg(-23, 3, true)]),
        )
    };
    let before = build().id();
    reset_hom_space_intern_table();
    let after = build().id();
    assert_eq!(before, after);
    assert!(!Arc::ptr_eq(&before.key, &after.key));
}

#[test]
fn fusion_tree_key_collision_falls_back_to_field_comparison() {
    // Why the hook exists: a genuine Fx collision on this fixture is not
    // constructible on demand, so the cache is forced equal to prove that
    // `Eq` and map insertion still distinguish keys by their fields.
    let (distinct, _, _) = key_hash_fixture();
    let forced = 0x5eed_u64;
    let left = distinct[0].clone().with_cached_hash_for_test(forced);
    for other in &distinct[1..] {
        let right = other.clone().with_cached_hash_for_test(forced);
        assert_eq!(fx_hash_of(&left), fx_hash_of(&right));
        assert_ne!(left, right);
        let mut map: rustc_hash::FxHashMap<FusionTreeKey, usize> = rustc_hash::FxHashMap::default();
        map.insert(left.clone(), 0);
        map.insert(right.clone(), 1);
        assert_eq!(map.len(), 2);
        assert_eq!(map[&left], 0);
        assert_eq!(map[&right], 1);
    }
}

#[test]
fn fusion_space_adjoint_view_preserves_custom_storage_footprint() {
    // What: adjoint swaps categorical sides and block axes while retaining
    // the already-admitted physical footprint; applying it twice is exact.
    let dense = TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap();
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 2)], [(0, 3)]);
    let structure = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::ordinal(0),
        vec![2, 3],
        vec![1, 4],
        2,
    )
    .unwrap()])
    .unwrap();
    let source = FusionTensorMapSpace::new_unbound(dense, homspace, structure).unwrap();

    reset_exact_storage_fallback_count();
    let adjoint = source.adjoint_view().unwrap();
    assert_eq!(exact_storage_fallback_count(), 0);
    assert_eq!(adjoint.dense_space().codomain().dims(), &[3]);
    assert_eq!(adjoint.dense_space().domain().dims(), &[2]);
    assert_eq!(adjoint.homspace().codomain(), source.homspace().domain());
    assert_eq!(adjoint.homspace().domain(), source.homspace().codomain());
    let block = adjoint.subblock_structure().block(0).unwrap();
    assert_eq!(block.shape(), &[3, 2]);
    assert_eq!(block.strides(), &[4, 1]);
    assert_eq!(block.offset(), 2);
    assert_eq!(adjoint.adjoint_view().unwrap(), source);
}

#[test]
fn canonical_coupled_storage_is_admitted_by_witness_across_symmetries() {
    // What: TeNeT-built coupled-sector destinations are admitted as
    // injective without visiting elements for Abelian, non-Abelian, and
    // fermionic product sectors with dual legs and nontrivial
    // degeneracies, while the exact check agrees on the same geometry.
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(-1), 2), (u1(0), 1), (u1(1), 3)], false),
            SectorLeg::new([(u1(-1), 2), (u1(1), 2)], true),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(u1(0), 2), (u1(1), 1), (u1(-1), 2)], false),
            SectorLeg::new([(u1(1), 3), (u1(-1), 1)], true),
        ]),
    );
    assert_canonical_storage_admitted_without_enumeration(
        &hom.coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap(),
    );

    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(su2(1), 2), (su2(3), 1)], false),
            SectorLeg::new([(su2(0), 1), (su2(2), 2)], true),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(su2(1), 3), (su2(2), 1)], true),
            SectorLeg::new([(su2(0), 2), (su2(1), 2)], false),
        ]),
    );
    assert_canonical_storage_admitted_without_enumeration(
        &hom.coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
            .unwrap(),
    );

    type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    let rule = Fz2U1::new(FermionParityFusionRule, U1FusionRule);
    let even0 = rule.encode_sector(z2_even(), u1(0));
    let odd_p = rule.encode_sector(z2_odd(), u1(1));
    let odd_m = rule.encode_sector(z2_odd(), u1(-1));
    let leg = |dual| SectorLeg::new([(even0, 2), (odd_p, 1), (odd_m, 3)], dual);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(true), leg(false)]),
    );
    assert_canonical_storage_admitted_without_enumeration(
        &hom.coupled_subblock_structure_from_leg_degeneracies(&rule)
            .unwrap(),
    );

    let (s0, s1) = (SectorId::new(0), SectorId::new(1));
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(s0, 2), (s1, 1)], false),
            SectorLeg::new([(s0, 1), (s1, 3)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(s0, 2), (s1, 2)], false),
            SectorLeg::new([(s0, 3), (s1, 1)], false),
        ]),
    );
    assert_canonical_storage_admitted_without_enumeration(
        &hom.coupled_subblock_structure_from_leg_degeneracies_generic(&IsomorphismMultiplicityRule)
            .unwrap(),
    );
}

#[test]
fn canonical_and_factor_bridge_storage_skip_exact_enumeration() {
    // What: hom-space-generated coupled storage and its typed factor-style
    // shared bridge are admitted by structural proof, not element scans.
    let rule = SU2FusionRule;
    let homspace = FusionTreeHomSpace::from_sectors([(su2(1), 2)], [(su2(1), 3)]);
    reset_exact_storage_fallback_count();
    let canonical = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
        homspace.clone(),
        &rule,
        [vec![2, 3]],
    )
    .unwrap();
    assert_eq!(exact_storage_fallback_count(), 0);

    reset_exact_storage_fallback_count();
    FusionTensorMapSpace::from_shared_subblock_structure(
        TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
        homspace,
        Arc::clone(canonical.subblock_structure()),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    assert_eq!(exact_storage_fallback_count(), 0);
}

#[test]
fn checked_select_and_permute_reject_a_malformed_id_when_orientation_needs_it() {
    // What: moving an excluded raw U1 ID across the HomSpace boundary
    // reports it, while same-side selection and identity permutation do not
    // inspect sectors they do not dualize.
    let min_leg = SectorLeg::new([(excluded_u1_id(), 2)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([min_leg]),
        FusionProductSpace::new([]),
    );
    let rule = DualCountingRule::new(U1FusionRule);

    let same_side = hom.try_select_checked(&rule, &[0], &[]).unwrap();
    assert_eq!(same_side, hom);
    assert_eq!(rule.dual_calls(), 0);
    let identity = hom.try_permute_checked(&rule, &[0], &[]).unwrap();
    assert_eq!(identity, hom);
    assert_eq!(rule.dual_calls(), 0);

    assert_eq!(
        hom.try_select_checked(&rule, &[], &[0]),
        Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::InvalidSector {
                sector: excluded_u1_id()
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 1);

    #[cfg(target_pointer_width = "64")]
    {
        type Rule = ProductFusionRule<U1FusionRule, Z2FusionRule, TensorKitProductCodec>;
        let product_rule = Rule::new(U1FusionRule, Z2FusionRule);
        let product_min = TensorKitProductCodec::encode(excluded_u1_id(), z2_odd());
        let product_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(product_min, 1)], false)]),
            FusionProductSpace::new([]),
        );
        assert_eq!(
            product_hom.try_permute_checked(&product_rule, &[], &[0]),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::InvalidSector {
                    sector: excluded_u1_id()
                }
            )))
        );
    }
}

#[test]
fn checked_orientation_validates_axes_before_sector_ids() {
    // What: malformed axis requests retain Core validation precedence even
    // when the first legal orientation mapping would inspect an invalid ID.
    let min_leg = SectorLeg::new([(excluded_u1_id(), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([min_leg]),
        FusionProductSpace::new([]),
    );
    let rule = DualCountingRule::new(U1FusionRule);
    assert_eq!(
        hom.try_select_checked(&rule, &[], &[1]),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::InvalidPermutation {
                permutation: vec![1],
                rank: 1,
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 0);
    assert_eq!(
        hom.try_permute_checked(&rule, &[], &[]),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::InvalidPermutation {
                permutation: vec![],
                rank: 1,
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 0);

    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    assert_eq!(
        FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            &rule,
            &hom,
            &scalar,
            &[],
            &[],
            &[1],
            0,
        ),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::InvalidPermutation {
                permutation: vec![1],
                rank: 1,
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 0);
}

#[test]
fn checked_tensorcontract_separates_pair_validation_and_output_orientation_failures() {
    // What: structural pair mismatches retain Core precedence without
    // inspecting sectors, while an excluded raw ID moved to the output
    // domain reports the exact algebra failure.
    let min_leg = || SectorLeg::new([(excluded_u1_id(), 1)], false);
    let scalar =
        || FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([min_leg()]),
        FusionProductSpace::new([]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([min_leg()]),
        FusionProductSpace::new([]),
    );
    let rule = DualCountingRule::new(U1FusionRule);
    assert_eq!(
        FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            &rule,
            &lhs,
            &rhs,
            &[0],
            &[0],
            &[],
            0,
        ),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::ContractedLegDualityMismatch {
                lhs_axis: 0,
                rhs_axis: 0,
                lhs_is_dual: false,
                rhs_is_dual: false,
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 0);

    let rhs_count_mismatch = FusionTreeHomSpace::new(
        FusionProductSpace::new([]),
        FusionProductSpace::new([SectorLeg::new([(excluded_u1_id(), 1), (u1(0), 1)], false)]),
    );
    assert_eq!(
        FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            &rule,
            &lhs,
            &rhs_count_mismatch,
            &[0],
            &[0],
            &[],
            0,
        ),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::DimensionMismatch {
                expected: 1,
                actual: 2,
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 0);

    assert_eq!(
        FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            &rule,
            &lhs,
            &scalar(),
            &[],
            &[],
            &[0],
            0,
        ),
        Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::InvalidSector {
                sector: excluded_u1_id()
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 1);
}

#[test]
fn checked_homspace_derivation_keeps_semantic_identity_lazy() {
    // What: successful checked metadata derivation does not publish or
    // resolve a process-local HomSpace identity until a caller asks for it.
    let rule = U1FusionRule;
    let leg = SectorLeg::new([(u1(-2), 1), (u1(1), 2)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg]),
        FusionProductSpace::new([]),
    );
    crate::reset_hom_space_intern_calls();
    let selected = hom.try_select_checked(&rule, &[1, 0], &[]).unwrap();
    let permuted = hom.try_permute_checked(&rule, &[1, 0], &[]).unwrap();
    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let contracted = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
        &rule,
        &hom,
        &scalar,
        &[],
        &[],
        &[1, 0],
        2,
    )
    .unwrap();
    assert_eq!(crate::hom_space_intern_calls(), 0);
    assert_eq!(selected, permuted);
    assert_eq!(contracted, permuted);
}

#[test]
fn unit_homspace_geometry_keeps_tensor_kit_seams_and_stored_duality() {
    // What: left/right placement, including the seam, preserves stored
    // domain duality while its external-axis view remains complemented.
    let rule = U1FusionRule;
    let base = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(-3), 1)], false)]),
    );
    for dual in [false, true] {
        for (left, position, codomain_duals, domain_duals) in [
            (true, 0, vec![dual, false], vec![false]),
            (true, 1, vec![false], vec![dual, false]),
            (true, 2, vec![false], vec![false, dual]),
            (false, 0, vec![dual, false], vec![false]),
            (false, 1, vec![false, dual], vec![false]),
            (false, 2, vec![false], vec![false, dual]),
        ] {
            let inserted = if left {
                base.insert_left_unit(&rule, position, dual)
            } else {
                base.insert_right_unit(&rule, position, dual)
            }
            .unwrap();
            assert_eq!(
                inserted
                    .codomain()
                    .legs()
                    .iter()
                    .map(SectorLeg::is_dual)
                    .collect::<Vec<_>>(),
                codomain_duals
            );
            assert_eq!(
                inserted
                    .domain()
                    .legs()
                    .iter()
                    .map(SectorLeg::is_dual)
                    .collect::<Vec<_>>(),
                domain_duals
            );
            let axis = if left || position != 1 { position } else { 1 };
            assert_eq!(
                inserted.external_axis_is_dual(axis),
                Some(if axis < inserted.codomain().len() {
                    dual
                } else {
                    !dual
                })
            );
            assert_eq!(inserted.remove_unit(&rule, axis).unwrap(), base);
        }
    }
    assert!(base.remove_unit(&rule, 0).is_err());

    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    assert_eq!(
        scalar
            .insert_left_unit(&rule, 0, true)
            .unwrap()
            .domain()
            .len(),
        1
    );
    assert_eq!(
        scalar
            .insert_right_unit(&rule, 0, true)
            .unwrap()
            .codomain()
            .len(),
        1
    );
}

#[derive(Clone, Copy)]
enum CheckedFailStage {
    Dual,
    Channels,
    Nsymbol,
}

// Trivial external multiplicity-free algebra (every product fuses to the
// vacuum) that can be told to fail exactly one checked primitive, so the
// checked enumeration transaction can be probed one fallible stage at a time.
#[derive(Clone, Copy)]
struct CheckedFailRule {
    fail: CheckedFailStage,
}

impl FusionRule for CheckedFailRule {
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
    fn fusion_channels(&self, _left: SectorId, _right: SectorId) -> SectorVec {
        core::iter::once(SectorId::new(0)).collect()
    }
}

impl MultiplicityFreeFusionRule for CheckedFailRule {}

impl CheckedFusionAlgebra for CheckedFailRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        if matches!(self.fail, CheckedFailStage::Dual) {
            return Err(FusionAlgebraError::InvalidSector { sector });
        }
        Ok(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        if matches!(self.fail, CheckedFailStage::Channels) {
            return Err(FusionAlgebraError::FusionNotRepresentable { left, right });
        }
        Ok(self.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        if matches!(self.fail, CheckedFailStage::Nsymbol) {
            return Err(FusionAlgebraError::MultiplicityOverflow {
                left,
                right,
                coupled,
            });
        }
        Ok(self.nsymbol(left, right, coupled))
    }
}

#[test]
fn checked_prepare_surfaces_each_stage_failure_without_publishing() {
    // What: an injected failure in dual, channels, or nsymbol returns the
    // exact typed error and leaves the layout cache statistics untouched
    // (no layout id issued, no admission).
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1), (1, 1)], []);

    for (stage, expected) in [
        (
            CheckedFailStage::Dual,
            FusionAlgebraError::InvalidSector {
                sector: SectorId::new(0),
            },
        ),
        (
            CheckedFailStage::Channels,
            FusionAlgebraError::FusionNotRepresentable {
                left: SectorId::new(0),
                right: SectorId::new(1),
            },
        ),
        (
            CheckedFailStage::Nsymbol,
            FusionAlgebraError::MultiplicityOverflow {
                left: SectorId::new(0),
                right: SectorId::new(1),
                coupled: SectorId::new(0),
            },
        ),
    ] {
        reset_core_intern_tables();
        reset_fusion_tree_layout_probe_side_effect_calls();
        let error = homspace
            .prepare_fusion_tree_layout_checked(&CheckedFailRule { fail: stage })
            .unwrap_err();
        assert_eq!(error, expected);
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));
    }
}
