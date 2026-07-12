use tenet_core::{
    product_fusion_rule, BraidingStyleKind, CoreError, FusionProductSpace, FusionRule,
    FusionStyleKind, FusionTreeHomSpace, FusionTreeKey, MultiplicityFreeFusionRule, RuleIdentity,
    SectorId, SectorLeg, SectorVec, TabulatedFusionRule, Z2FusionRule,
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
fn independently_loaded_identical_tables_do_not_use_fnv_provenance_as_identity() {
    const TABLE: &[u8] = include_bytes!("../src/su3_table.bin");
    let first = TabulatedFusionRule::try_from_bytes(TABLE, "first-su3-table.bin").unwrap();
    let second = TabulatedFusionRule::try_from_bytes(TABLE, "second-su3-table.bin").unwrap();

    assert_eq!(first.provenance(), second.provenance());
    assert_ne!(first.rule_identity(), second.rule_identity());
    assert_eq!(first.rule_identity(), first.clone().rule_identity());
}

#[test]
fn product_rule_identity_includes_stateful_child_identity() {
    let first = product_fusion_rule(StatefulPointedRule::new(0), Z2FusionRule);
    let second = product_fusion_rule(StatefulPointedRule::new(1), Z2FusionRule);

    assert_ne!(first.rule_identity(), second.rule_identity());
}

#[test]
fn multiplicity_free_tree_rejects_nontrivial_vertex() {
    let error = FusionTreeKey::try_new_for_rule(
        &Z2FusionRule,
        [SectorId::new(1), SectorId::new(1)],
        Some(SectorId::new(0)),
        [false, false],
        [],
        [SectorId::new(2)],
    )
    .unwrap_err();

    assert_eq!(
        error,
        CoreError::MalformedFusionTree {
            message: "multiplicity-free fusion tree has a nontrivial vertex",
        }
    );
}
