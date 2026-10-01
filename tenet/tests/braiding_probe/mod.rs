//! One-sector real rules declaring a non-symmetric braiding style (#1355,
//! #1372), shared by the `tenet` and `tenet-network` suites.
//!
//! The default probe has every symbol equal to 1, so only an explicit
//! braiding guard can reject. `R_SCALE` injects extreme R values for fallback
//! tests. (No built-in `Scalar = f64` provider is anyonic or unbraided;
//! Fibonacci is complex.)

// Each including suite uses a subset.
#![allow(dead_code)]

use std::sync::Arc;

use tenet::sector::SectorId;
use tenet::sector::{
    BraidingStyleKind, CheckedFusionAlgebra, FusionRule, FusionStyleKind,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    RuleIdentity, SectorCodec, SectorVec,
};
use tenet::typed::FusionAlgebraError;

/// `ANYONIC` selects `Anyonic`, otherwise `NoBraiding`.
pub struct RealBraidingProbe<const ANYONIC: bool, const R_SCALE: u8 = 0>;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProbeSector;

impl<const ANYONIC: bool, const R_SCALE: u8> FusionRule for RealBraidingProbe<ANYONIC, R_SCALE> {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(
            0x1372_0000_0000_0000 | u64::from(ANYONIC) | (u64::from(R_SCALE) << 8),
            Arc::<[u8]>::from([]),
        )
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        if ANYONIC {
            BraidingStyleKind::Anyonic
        } else {
            BraidingStyleKind::NoBraiding
        }
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let coupled = if R_SCALE == 0 {
            0
        } else {
            left.id() ^ right.id()
        };
        core::iter::once(SectorId::new(coupled)).collect()
    }
}

impl<const ANYONIC: bool, const R_SCALE: u8> MultiplicityFreeFusionRule
    for RealBraidingProbe<ANYONIC, R_SCALE>
{
}

impl<const ANYONIC: bool, const R_SCALE: u8> MultiplicityFreeFusionSymbols
    for RealBraidingProbe<ANYONIC, R_SCALE>
{
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> f64 {
        1.0
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        match R_SCALE {
            1 => 1e-100,
            2 => 1e100,
            _ => 1.0,
        }
    }
}

impl<const ANYONIC: bool, const R_SCALE: u8> MultiplicityFreeRigidSymbols
    for RealBraidingProbe<ANYONIC, R_SCALE>
{
    fn dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn twist_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
}

impl<const ANYONIC: bool, const R_SCALE: u8> CheckedFusionAlgebra
    for RealBraidingProbe<ANYONIC, R_SCALE>
{
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(self.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(self.nsymbol(left, right, coupled))
    }
}

impl<const ANYONIC: bool, const R_SCALE: u8> SectorCodec for RealBraidingProbe<ANYONIC, R_SCALE> {
    type Sector = ProbeSector;
    fn encode_sector(&self, _: &ProbeSector) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(usize::from(R_SCALE != 0)))
    }
    fn decode_sector(&self, sector: SectorId) -> Result<ProbeSector, FusionAlgebraError> {
        if sector == SectorId::new(usize::from(R_SCALE != 0)) {
            Ok(ProbeSector)
        } else {
            Err(FusionAlgebraError::InvalidSector { sector })
        }
    }
}
