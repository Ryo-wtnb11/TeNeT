//! One-sector real rules declaring a non-symmetric braiding style (#1355,
//! #1372), shared by the `tenet` and `tenet-network` suites.
//!
//! Every symbol is 1, so any operation that runs anyway produces a value and
//! only an explicit braiding guard can reject. (No built-in `Scalar = f64`
//! provider is anyonic or unbraided; Fibonacci is complex.)

// Each including suite uses a subset.
#![allow(dead_code)]

use std::sync::Arc;

use tenet::core::{
    BraidingStyleKind, CheckedFusionAlgebra, FusionAlgebraError, FusionRule, FusionStyleKind,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    RuleIdentity, SectorCodec, SectorId, SectorVec,
};

/// `ANYONIC` selects `Anyonic`, otherwise `NoBraiding`.
pub struct RealBraidingProbe<const ANYONIC: bool>;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProbeSector;

impl<const ANYONIC: bool> FusionRule for RealBraidingProbe<ANYONIC> {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(
            0x1372_0000_0000_0000 | u64::from(ANYONIC),
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
    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        core::iter::once(SectorId::new(0)).collect()
    }
}

impl<const ANYONIC: bool> MultiplicityFreeFusionRule for RealBraidingProbe<ANYONIC> {}

impl<const ANYONIC: bool> MultiplicityFreeFusionSymbols for RealBraidingProbe<ANYONIC> {
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
        1.0
    }
}

impl<const ANYONIC: bool> MultiplicityFreeRigidSymbols for RealBraidingProbe<ANYONIC> {
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

impl<const ANYONIC: bool> CheckedFusionAlgebra for RealBraidingProbe<ANYONIC> {
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

impl<const ANYONIC: bool> SectorCodec for RealBraidingProbe<ANYONIC> {
    type Sector = ProbeSector;
    fn encode_sector(&self, _: &ProbeSector) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(0))
    }
    fn decode_sector(&self, sector: SectorId) -> Result<ProbeSector, FusionAlgebraError> {
        if sector == SectorId::new(0) {
            Ok(ProbeSector)
        } else {
            Err(FusionAlgebraError::InvalidSector { sector })
        }
    }
}
