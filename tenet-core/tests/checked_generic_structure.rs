use std::{cell::Cell, fmt};

use tenet_core::{
    complete_hom_space_structure_cache_info, fusion_tree_layout_cache_info, BraidingStyleKind,
    CheckedGenericFusion, CheckedGenericStructureError, CoupledSectorFold, FusionProductSpace,
    FusionStyleKind, FusionTreeHomSpace, InfallibleGeneric, RuleIdentity, SectorId, SectorLeg,
    SectorVec, Su3FusionRule,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Failure {
    None,
    Dual,
    Channel,
    InTable,
    Fold,
    Multiplicity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ToyError(Failure);
impl fmt::Display for ToyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
impl std::error::Error for ToyError {}

struct Toy {
    fail: Failure,
    calls: Cell<usize>,
}
impl Toy {
    fn new(fail: Failure) -> Self {
        Self {
            fail,
            calls: Cell::new(0),
        }
    }
    fn hit(&self, at: Failure) -> Result<(), ToyError> {
        self.calls.set(self.calls.get() + 1);
        if self.fail == at {
            Err(ToyError(at))
        } else {
            Ok(())
        }
    }
}
impl CheckedGenericFusion for Toy {
    type Error = ToyError;
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
    fn try_dual(&self, s: SectorId) -> Result<SectorId, ToyError> {
        self.hit(Failure::Dual)?;
        Ok(s)
    }
    fn try_fusion_channels(&self, a: SectorId, b: SectorId) -> Result<SectorVec, ToyError> {
        self.hit(Failure::Channel)?;
        Ok(if a.id() == 0 {
            [b].into_iter().collect()
        } else if b.id() == 0 {
            [a].into_iter().collect()
        } else {
            [SectorId::new(1)].into_iter().collect()
        })
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, ToyError> {
        self.hit(Failure::InTable)?;
        self.try_fusion_channels(a, b)
    }
    fn try_coupled_sector_fold(&self, _: &[SectorId]) -> Result<CoupledSectorFold, ToyError> {
        self.hit(Failure::Fold)?;
        Ok(CoupledSectorFold {
            clean: vec![SectorId::new(1)],
            ..Default::default()
        })
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, ToyError> {
        self.hit(Failure::Multiplicity)?;
        Ok(usize::from(a.id() == 1 && b.id() == 1 && c.id() == 1) * 2)
    }
}

fn hom() -> FusionTreeHomSpace {
    let leg = |dual| SectorLeg::new([(SectorId::new(1), 1)], dual);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true), leg(false)]),
        FusionProductSpace::new([leg(false)]),
    )
}

#[test]
fn checked_generic_structure_preserves_order_and_never_publishes_on_provider_failure() {
    for fail in [
        Failure::Dual,
        Failure::Channel,
        Failure::InTable,
        Failure::Fold,
        Failure::Multiplicity,
    ] {
        let before = (
            fusion_tree_layout_cache_info(),
            complete_hom_space_structure_cache_info(),
        );
        let error = hom()
            .fusion_tree_keys_generic_checked(&Toy::new(fail))
            .unwrap_err();
        assert!(
            matches!(error, CheckedGenericStructureError::Provider(ToyError(actual)) if actual == fail)
        );
        assert_eq!(
            before,
            (
                fusion_tree_layout_cache_info(),
                complete_hom_space_structure_cache_info()
            )
        );
    }
    let keys = hom()
        .fusion_tree_keys_generic_checked(&Toy::new(Failure::None))
        .unwrap();
    assert_eq!(keys.len(), 4);
    assert_eq!(keys[0].codomain_tree().vertices()[0].get(), 1);
    assert_eq!(keys[1].codomain_tree().vertices()[0].get(), 2);
    assert_eq!(keys[0].codomain_tree().innerlines(), &[SectorId::new(1)]);
}

#[test]
fn infallible_adapter_keeps_su3_generic_tree_order() {
    let rule = Su3FusionRule::new();
    let eight = rule.sector_of(1, 1).unwrap();
    let space = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(eight, 1)], false),
            SectorLeg::new([(eight, 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    assert_eq!(
        space.fusion_tree_keys_generic(&rule).unwrap(),
        space
            .fusion_tree_keys_generic_checked(&InfallibleGeneric::new(&rule))
            .unwrap()
    );
}
