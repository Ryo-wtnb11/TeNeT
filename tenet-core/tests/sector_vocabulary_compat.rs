use tenet_core::{BraidingStyleKind, FusionStyleKind, SectorId, SectorVec};

#[test]
fn core_root_reexports_are_tenet_sectors_types() {
    let _: tenet_sectors::SectorId = SectorId::new(1);
    let _: tenet_sectors::FusionStyleKind = FusionStyleKind::Unique;
    let _: tenet_sectors::BraidingStyleKind = BraidingStyleKind::NoBraiding;
    let _: tenet_sectors::SectorVec = SectorVec::new();
}
