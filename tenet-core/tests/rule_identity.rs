use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tenet_core::{
    product_fusion_rule, BlockKey, BlockSpec, BlockStructure, BraidingStyleKind, CoreError,
    FusionProductSpace, FusionRule, FusionStyleKind, FusionTensorMapSpace, FusionTreeHomSpace,
    FusionTreeKey, FusionTreePairKey, MultiplicityFreeFusionRule, MultiplicityIndex, RuleIdentity,
    SectorId, SectorLeg, SectorVec, TabulatedFusionRule, TensorMapSpace, Z2FusionRule,
};

#[derive(Clone, Debug)]
struct StatefulPointedRule {
    identity: RuleIdentity,
    one_times_one: SectorId,
}

impl StatefulPointedRule {
    fn new(one_times_one: usize) -> Self {
        Self {
            identity: RuleIdentity::new_unique::<Self>(),
            one_times_one: SectorId::new(one_times_one),
        }
    }
}

impl FusionRule for StatefulPointedRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
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

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let output = if left.id() == 1 && right.id() == 1 {
            self.one_times_one
        } else {
            SectorId::new(left.id().max(right.id()))
        };
        [output].into_iter().collect()
    }
}

impl MultiplicityFreeFusionRule for StatefulPointedRule {}

fn two_ones_to_vacuum_homspace() -> FusionTreeHomSpace {
    FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::from_sector_id(1, 1),
            SectorLeg::from_sector_id(1, 1),
        ]),
        FusionProductSpace::new([]),
    )
}

#[test]
fn same_rule_type_with_distinct_semantics_does_not_share_fusion_tree_layout() {
    let fuses_to_vacuum = StatefulPointedRule::new(0);
    let fuses_to_one = StatefulPointedRule::new(1);
    let homspace = two_ones_to_vacuum_homspace();

    assert_eq!(homspace.fusion_tree_keys(&fuses_to_vacuum).len(), 1);
    assert_eq!(homspace.fusion_tree_keys(&fuses_to_one).len(), 0);
}

#[test]
fn cloned_stateful_rule_preserves_identity() {
    let rule = StatefulPointedRule::new(0);

    assert_eq!(rule.rule_identity(), rule.clone().rule_identity());
}

#[test]
fn independently_loaded_identical_tables_share_full_content_identity() {
    const TABLE: &[u8] = include_bytes!("../src/su3_table.bin");
    let first = TabulatedFusionRule::try_from_bytes(TABLE, "first-su3-table.bin").unwrap();
    let second = TabulatedFusionRule::try_from_bytes(TABLE, "second-su3-table.bin").unwrap();

    assert_eq!(first.provenance(), second.provenance());
    assert_eq!(first.rule_identity(), second.rule_identity());
    assert_eq!(first.rule_identity(), first.clone().rule_identity());
}

#[derive(Default)]
struct CountingHasher(usize);

impl Hasher for CountingHasher {
    fn finish(&self) -> u64 {
        self.0 as u64
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0 += bytes.len();
    }
}

#[test]
fn content_identity_hash_cost_does_not_scale_with_table_bytes() {
    let short = RuleIdentity::from_canonical_bytes::<StatefulPointedRule>(7, Arc::from([1u8]));
    let long = RuleIdentity::from_canonical_bytes::<StatefulPointedRule>(
        7,
        Arc::from(vec![2u8; 1_000_000]),
    );
    let mut short_hasher = CountingHasher::default();
    let mut long_hasher = CountingHasher::default();

    short.hash(&mut short_hasher);
    long.hash(&mut long_hasher);

    assert_eq!(short_hasher.0, long_hasher.0);
    assert_ne!(short, long);
}

fn raw_scalar_space() -> FusionTensorMapSpace<0, 0> {
    let key =
        FusionTreePairKey::try_pair_from_sector_ids([], [], 0, [], [], [], [], [], []).unwrap();
    FusionTensorMapSpace::new_unbound(
        TensorMapSpace::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
            key.into(),
            vec![],
            0,
        )
        .unwrap()])
        .unwrap(),
    )
    .unwrap()
}

fn raw_scalar_space_with_key(key: BlockKey) -> FusionTensorMapSpace<0, 0> {
    FusionTensorMapSpace::new_unbound(
        TensorMapSpace::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(key, vec![], 0).unwrap()
        ])
        .unwrap(),
    )
    .unwrap()
}

fn raw_scalar_space_for_rule<R: FusionRule>(rule: &R) -> FusionTensorMapSpace<0, 0> {
    let empty_tree = FusionTreeKey::try_new_for_rule(rule, [], rule.vacuum(), [], [], []).unwrap();
    let key = FusionTreePairKey::pair(empty_tree.clone(), empty_tree);
    raw_scalar_space_with_key(key.into())
}

fn z2_rank_one_pair(sector: SectorId, is_dual: bool) -> FusionTreePairKey {
    let tree = FusionTreeKey::try_new_for_rule(&Z2FusionRule, [sector], sector, [is_dual], [], [])
        .unwrap();
    FusionTreePairKey::pair(tree.clone(), tree)
}

fn raw_z2_matrix_space(
    leg_sector: SectorId,
    degeneracy: usize,
    key: FusionTreePairKey,
    shape: Vec<usize>,
) -> FusionTensorMapSpace<1, 1> {
    let leg = || FusionProductSpace::new([SectorLeg::new([(leg_sector, degeneracy)], false)]);
    FusionTensorMapSpace::new_unbound(
        TensorMapSpace::from_dims([degeneracy], [degeneracy]).unwrap(),
        FusionTreeHomSpace::new(leg(), leg()),
        BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::column_major_with_key(key.into(), shape, 0).unwrap()],
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn binding_a_rule_rejects_non_categorical_block_namespaces() {
    // What: neither the anonymous dense key nor application routing metadata
    // can acquire a categorical rule identity.
    for key in [BlockKey::Dense, BlockKey::opaque([0])] {
        let error = raw_scalar_space_with_key(key)
            .try_bind_rule(&Z2FusionRule)
            .unwrap_err();
        assert!(matches!(error, CoreError::ExpectedFusionTreePairKey { .. }));
    }
}

#[test]
fn rule_binding_requires_every_present_block_to_be_a_homspace_subset() {
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let split_codomain = FusionTreeKey::try_new_for_rule(
        &Z2FusionRule,
        [even, even],
        even,
        [false, false],
        [],
        [MultiplicityIndex::ONE],
    )
    .unwrap();
    let split_domain =
        FusionTreeKey::try_new_for_rule(&Z2FusionRule, [], even, [], [], []).unwrap();
    let cases = [
        (
            "codomain/domain split",
            raw_z2_matrix_space(
                even,
                1,
                FusionTreePairKey::pair(split_codomain, split_domain),
                vec![1, 1],
            ),
            CoreError::FusionSpaceSplitMismatch {
                expected_nout: 1,
                expected_nin: 1,
                actual_nout: 2,
                actual_nin: 0,
            },
        ),
        (
            "leg membership",
            raw_z2_matrix_space(even, 1, z2_rank_one_pair(odd, false), vec![1, 1]),
            CoreError::MalformedFusionTree {
                message: "fusion tree uses a sector absent from its HomSpace leg",
            },
        ),
        (
            "leg duality",
            raw_z2_matrix_space(even, 1, z2_rank_one_pair(even, true), vec![1, 1]),
            CoreError::MalformedFusionTree {
                message: "fusion tree duality disagrees with its HomSpace leg",
            },
        ),
        (
            "leg degeneracy",
            raw_z2_matrix_space(even, 2, z2_rank_one_pair(even, false), vec![1, 2]),
            CoreError::LegDegeneracyMismatch {
                sector: even,
                expected: 2,
                actual: 1,
            },
        ),
    ];

    for (name, space, expected) in cases {
        // What: a locally admissible tree cannot acquire a rule identity when
        // its external sectors, orientations, or logical shape disagree with
        // the declared HomSpace.
        assert_eq!(space.rule_identity(), None, "{name}");
        assert_eq!(space.try_bind_rule(&Z2FusionRule), Err(expected), "{name}");
    }
}

#[test]
fn bound_space_rejects_rebinding_to_same_type_with_different_semantics() {
    let first = StatefulPointedRule::new(0);
    let second = StatefulPointedRule::new(1);
    let space = raw_scalar_space().try_bind_rule(&first).unwrap();

    assert!(matches!(
        space.try_bind_rule(&second),
        Err(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn generic_space_rejects_a_different_tabulated_rule() {
    const SU3: &[u8] = include_bytes!("../src/su3_table.bin");
    const SU4: &[u8] = include_bytes!("../src/testdata/su4_table.bin");
    let su3 = TabulatedFusionRule::try_from_bytes(SU3, "su3-table.bin").unwrap();
    let su4 = TabulatedFusionRule::try_from_bytes(SU4, "su4-table.bin").unwrap();
    let space = raw_scalar_space_for_rule(&su3).try_bind_rule(&su3).unwrap();

    assert!(matches!(
        space.validate_rule(&su4),
        Err(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn product_rule_identity_includes_stateful_child_identity() {
    let first = product_fusion_rule(StatefulPointedRule::new(0), Z2FusionRule);
    let second = product_fusion_rule(StatefulPointedRule::new(1), Z2FusionRule);

    assert_ne!(first.rule_identity(), second.rule_identity());
}

#[test]
fn multiplicity_free_tree_rejects_out_of_range_vertices() {
    // What: rule-aware construction reports the precise lower and upper
    // 1-based multiplicity bounds for a multiplicity-free vertex.
    assert_eq!(
        MultiplicityIndex::try_from(0).unwrap_err(),
        CoreError::InvalidMultiplicityIndex { value: 0 }
    );
    let error = FusionTreeKey::try_new_for_rule(
        &Z2FusionRule,
        [SectorId::new(1), SectorId::new(1)],
        SectorId::new(0),
        [false, false],
        [],
        [MultiplicityIndex::new(2).unwrap()],
    )
    .unwrap_err();
    assert_eq!(
        error,
        CoreError::MalformedFusionTree {
            message: "fusion tree vertex label exceeds its fusion multiplicity",
        }
    );
}
