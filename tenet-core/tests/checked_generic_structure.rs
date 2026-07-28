use std::{cell::Cell, error::Error, fmt};

use tenet_core::{
    block_structure_intern_cache_info, complete_hom_space_structure_cache_info,
    fusion_tree_layout_cache_info, CheckedGenericFusion, CheckedGenericStructureError,
    CoupledSectorFold, FusionProductSpace, FusionTreeHomSpace, InfallibleGeneric, SectorId,
    SectorLeg, SectorVec, Su3FusionRule,
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
    skips: Cell<usize>,
}
impl Toy {
    fn new(fail: Failure) -> Self {
        Self {
            fail,
            calls: Cell::new(0),
            skips: Cell::new(0),
        }
    }
    fn late_nsymbol(skip: usize) -> Self {
        Self {
            fail: Failure::Multiplicity,
            calls: Cell::new(0),
            skips: Cell::new(skip),
        }
    }
    fn hit(&self, at: Failure) -> Result<(), ToyError> {
        self.calls.set(self.calls.get() + 1);
        if self.fail == at && self.skips.get() == 0 {
            Err(ToyError(at))
        } else {
            if self.fail == at {
                self.skips.set(self.skips.get() - 1);
            }
            Ok(())
        }
    }
}

#[test]
fn checked_generic_late_nsymbol_failure_propagates() {
    let error = hom()
        .fusion_tree_keys_generic_checked(&Toy::late_nsymbol(2))
        .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Provider(ToyError(Failure::Multiplicity))
    ));
}
impl CheckedGenericFusion for Toy {
    type Error = ToyError;
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

struct DefaultFoldToy(Toy);
impl CheckedGenericFusion for DefaultFoldToy {
    type Error = ToyError;
    fn vacuum(&self) -> SectorId {
        self.0.vacuum()
    }
    fn try_dual(&self, s: SectorId) -> Result<SectorId, ToyError> {
        self.0.try_dual(s)
    }
    fn try_fusion_channels(&self, a: SectorId, b: SectorId) -> Result<SectorVec, ToyError> {
        self.0.try_fusion_channels(a, b)
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, ToyError> {
        self.0.try_fusion_channels_in_table(a, b)
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, ToyError> {
        self.0.try_nsymbol(a, b, c)
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
fn checked_generic_default_fold_keeps_channel_failure_live() {
    let error = hom()
        .fusion_tree_keys_generic_checked(&DefaultFoldToy(Toy::new(Failure::Channel)))
        .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Provider(ToyError(Failure::Channel))
    ));
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
            block_structure_intern_cache_info(),
        );
        let error = hom()
            .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&Toy::new(fail))
            .unwrap_err();
        assert!(
            matches!(error, CheckedGenericStructureError::Provider(ToyError(actual)) if actual == fail)
        );
        assert_eq!(
            error
                .source()
                .and_then(|source| source.downcast_ref::<ToyError>()),
            Some(&ToyError(fail))
        );
        assert_eq!(
            before,
            (
                fusion_tree_layout_cache_info(),
                complete_hom_space_structure_cache_info(),
                block_structure_intern_cache_info()
            )
        );
    }
    let keys = hom()
        .fusion_tree_keys_generic_checked(&Toy::new(Failure::None))
        .unwrap();
    assert_eq!(keys.len(), 4);
    assert_eq!(keys[0].codomain_tree().vertices()[0].get(), 1);
    assert_eq!(keys[1].codomain_tree().vertices()[0].get(), 2);
    assert_eq!(
        keys.iter()
            .map(|key| (
                key.codomain_tree().vertices()[0].get(),
                key.codomain_tree().vertices()[1].get()
            ))
            .collect::<Vec<_>>(),
        vec![(1, 1), (2, 1), (1, 2), (2, 2)]
    );
    assert_eq!(keys[0].codomain_tree().innerlines(), &[SectorId::new(1)]);
    let structure = hom()
        .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&Toy::new(Failure::None))
        .unwrap();
    assert_eq!(structure.block_count(), keys.len());
    for (index, key) in keys.iter().enumerate() {
        assert_eq!(
            structure.block(index).unwrap().key().as_fusion_tree_pair(),
            Some(key)
        );
    }
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

#[test]
fn checked_adapter_pins_tensor_kit_su3_generic_boundaries() {
    let rule = Su3FusionRule::new();
    let checked = InfallibleGeneric::new(&rule);
    let eight = rule.sector_of(1, 1).unwrap();
    let t27 = rule.sector_of(2, 2).unwrap();
    let leg = |s| SectorLeg::new([(s, 1)], false);
    let rank2 = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(eight), leg(eight)]),
        FusionProductSpace::new([SectorLeg::new(
            [
                (SectorId::new(0), 1),
                (SectorId::new(5), 1),
                (SectorId::new(6), 1),
                (SectorId::new(7), 1),
                (SectorId::new(16), 1),
            ],
            false,
        )]),
    );
    assert_eq!(
        rank2
            .fusion_tree_keys_generic_checked(&checked)
            .unwrap()
            .iter()
            .map(|k| (k.coupled().id(), k.codomain_vertices()[0].get()))
            .collect::<Vec<_>>(),
        vec![(0, 1), (5, 1), (5, 2), (6, 1), (7, 1), (16, 1)]
    );
    let oracle: [(usize, &[(usize, usize, usize)]); 5] = [
        (0, &[(5, 1, 1), (5, 2, 1)]),
        (
            5,
            &[
                (0, 1, 1),
                (5, 1, 1),
                (5, 2, 1),
                (5, 1, 2),
                (5, 2, 2),
                (16, 1, 1),
                (6, 1, 1),
                (7, 1, 1),
            ],
        ),
        (6, &[(5, 1, 1), (5, 2, 1), (16, 1, 1), (6, 1, 1)]),
        (7, &[(5, 1, 1), (5, 2, 1), (16, 1, 1), (7, 1, 1)]),
        (
            16,
            &[
                (5, 1, 1),
                (5, 2, 1),
                (16, 1, 1),
                (16, 1, 2),
                (6, 1, 1),
                (7, 1, 1),
            ],
        ),
    ];
    let mut total = 0;
    for (c, expected) in oracle {
        let c = SectorId::new(c);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(eight), leg(eight), leg(eight)]),
            FusionProductSpace::new([leg(c)]),
        );
        let got = hom
            .fusion_tree_keys_generic_for_coupled_checked(&checked, c)
            .unwrap()
            .iter()
            .map(|k| {
                (
                    k.codomain_tree().innerlines()[0].id(),
                    k.codomain_tree().vertices()[0].get(),
                    k.codomain_tree().vertices()[1].get(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(got, expected);
        total += got.len();
    }
    assert_eq!(total, 24);
    let frontier = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(t27), leg(eight)]),
        FusionProductSpace::new([leg(eight)]),
    );
    assert!(frontier
        .fusion_tree_keys_generic_checked(&checked)
        .unwrap_err()
        .to_string()
        .contains("cannot represent this space exactly"));
}
