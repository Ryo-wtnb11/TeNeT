use symtn_core::{BraidingStyleKind, FusionStyleKind, SectorId, SectorVec};

#[test]
fn core_root_reexports_are_symtn_sectors_types() {
    let _: symtn_sectors::SectorId = SectorId::new(1);
    let _: symtn_sectors::FusionStyleKind = FusionStyleKind::Unique;
    let _: symtn_sectors::BraidingStyleKind = BraidingStyleKind::NoBraiding;
    let _: symtn_sectors::SectorVec = SectorVec::new();
}
