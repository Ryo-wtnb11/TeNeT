use std::sync::{Arc, OnceLock};

use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError,
    FusionRule, FusionStyleKind, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, RuleIdentity, SectorCodec, SectorId, SectorVec,
};

/// Largest doubled spin representable by TeNeT's compact SU(2) sector encoding.
pub const SU2_MAX_DOUBLED_SPIN: usize = (u8::MAX - 1) as usize;

/// TeNeT's compact SU(2) sector label (`twice_spin = 2j`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SU2Irrep {
    twice_spin: usize,
}

impl SU2Irrep {
    pub const fn try_from_twice_spin(twice_spin: usize) -> Option<Self> {
        if twice_spin <= SU2_MAX_DOUBLED_SPIN {
            Some(Self { twice_spin })
        } else {
            None
        }
    }

    pub const fn from_twice_spin(twice_spin: usize) -> Self {
        match Self::try_from_twice_spin(twice_spin) {
            Some(irrep) => irrep,
            None => panic!("SU(2) doubled spin exceeds the supported maximum 254"),
        }
    }

    #[inline]
    pub const fn twice_spin(self) -> usize {
        self.twice_spin
    }

    #[inline]
    pub const fn sector_id(self) -> SectorId {
        SectorId::new(self.twice_spin)
    }

    pub const fn try_from_sector_id(sector: SectorId) -> Option<Self> {
        Self::try_from_twice_spin(sector.id())
    }

    pub const fn from_sector_id(sector: SectorId) -> Self {
        match Self::try_from_sector_id(sector) {
            Some(irrep) => irrep,
            None => panic!("SU(2) sector exceeds the supported maximum doubled spin 254"),
        }
    }
}

impl From<SU2Irrep> for SectorId {
    fn from(value: SU2Irrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct SU2FusionRule;

const SU2_RACAH_IDENTITY_SCHEMA: u64 = 0x5355_3252_4143_4148;
const SU2_ENCODING_IDENTITY: &[u8] =
    b"tenet:su2:sector-id=dj:max-doubled-spin=254:channels=ascending-step-2";

fn racah_irrep(irrep: SU2Irrep) -> racah::Su2Irrep {
    racah::Su2Irrep::new(irrep.twice_spin() as u32)
}

fn checked_irrep(sector: SectorId) -> Result<SU2Irrep, FusionAlgebraError> {
    SU2Irrep::try_from_sector_id(sector).ok_or(FusionAlgebraError::InvalidSector { sector })
}

fn checked_irreps(
    left: SectorId,
    right: SectorId,
) -> Result<(SU2Irrep, SU2Irrep), FusionAlgebraError> {
    let left_irrep = checked_irrep(left)?;
    let right_irrep = checked_irrep(right)?;
    Ok((left_irrep, right_irrep))
}

impl FusionRule for SU2FusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        static IDENTITY: OnceLock<RuleIdentity> = OnceLock::new();
        IDENTITY
            .get_or_init(|| {
                let mut bytes = Vec::with_capacity(
                    racah::su2_authority_fingerprint().len() + SU2_ENCODING_IDENTITY.len(),
                );
                bytes.extend_from_slice(racah::su2_authority_fingerprint());
                bytes.extend_from_slice(SU2_ENCODING_IDENTITY);
                RuleIdentity::from_canonical_bytes::<SU2FusionRule>(
                    SU2_RACAH_IDENTITY_SCHEMA,
                    Arc::<[u8]>::from(bytes),
                )
            })
            .clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SU2Irrep::from_twice_spin(0).into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let left_irrep = SU2Irrep::from_sector_id(left);
        let right_irrep = SU2Irrep::from_sector_id(right);
        assert!(
            left_irrep.twice_spin() + right_irrep.twice_spin() <= SU2_MAX_DOUBLED_SPIN,
            "SU(2) fusion closure exceeds the supported maximum doubled spin 254"
        );
        racah_irrep(left_irrep)
            .fusion(racah_irrep(right_irrep))
            .expect("TeNeT SU(2) label range cannot overflow racah labels")
            .map(|irrep| SU2Irrep::from_twice_spin(irrep.dj() as usize).into())
            .collect()
    }
}

impl SU2FusionRule {
    #[doc(hidden)]
    pub fn try_for_each_representable_channel<B>(
        &self,
        left: SU2Irrep,
        right: SU2Irrep,
        mut emit: impl FnMut(SU2Irrep) -> core::ops::ControlFlow<B>,
    ) -> Result<core::ops::ControlFlow<B>, FusionAlgebraError> {
        let left_sector = left.sector_id();
        let right_sector = right.sector_id();
        if left.twice_spin() + right.twice_spin() > SU2_MAX_DOUBLED_SPIN {
            return Err(FusionAlgebraError::FusionNotRepresentable {
                left: left_sector,
                right: right_sector,
            });
        }
        for channel in racah_irrep(left)
            .fusion(racah_irrep(right))
            .expect("TeNeT SU(2) label range cannot overflow racah labels")
        {
            if let core::ops::ControlFlow::Break(error) =
                emit(SU2Irrep::from_twice_spin(channel.dj() as usize))
            {
                return Ok(core::ops::ControlFlow::Break(error));
            }
        }
        Ok(core::ops::ControlFlow::Continue(()))
    }
}

impl CheckedFusionAlgebra for SU2FusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        let irrep = checked_irrep(sector)?;
        Ok(SU2Irrep::from_twice_spin(racah_irrep(irrep).dual().dj() as usize).into())
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let (left_irrep, right_irrep) = checked_irreps(left, right)?;
        let mut channels = SectorVec::new();
        let _ = self.try_for_each_representable_channel(left_irrep, right_irrep, |channel| {
            channels.push(channel.into());
            core::ops::ControlFlow::<()>::Continue(())
        })?;
        Ok(channels)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let coupled = checked_irrep(coupled)?;
        let (left_irrep, right_irrep) = checked_irreps(left, right)?;
        let mut multiplicity = 0;
        let _ = self.try_for_each_representable_channel(left_irrep, right_irrep, |channel| {
            if channel == coupled {
                multiplicity = 1;
            }
            core::ops::ControlFlow::<()>::Continue(())
        })?;
        Ok(multiplicity)
    }
}

impl SectorCodec for SU2FusionRule {
    type Sector = SU2Irrep;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        Ok(value.sector_id())
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        checked_irrep(id)
    }
}

impl MultiplicityFreeFusionRule for SU2FusionRule {}

impl CanonicalUnitFusionRule for SU2FusionRule {}

impl MultiplicityFreeFusionSymbols for SU2FusionRule {
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        racah::su2_f_symbol(
            SU2Irrep::from_sector_id(left).twice_spin() as u32,
            SU2Irrep::from_sector_id(middle).twice_spin() as u32,
            SU2Irrep::from_sector_id(right).twice_spin() as u32,
            SU2Irrep::from_sector_id(coupled).twice_spin() as u32,
            SU2Irrep::from_sector_id(left_coupled).twice_spin() as u32,
            SU2Irrep::from_sector_id(right_coupled).twice_spin() as u32,
        )
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let left = SU2Irrep::from_sector_id(left);
        let right = SU2Irrep::from_sector_id(right);
        let mut admissible = false;
        let _ = self
            .try_for_each_representable_channel(left, right, |channel| {
                if channel.sector_id() == coupled {
                    admissible = true;
                    core::ops::ControlFlow::Break(())
                } else {
                    core::ops::ControlFlow::Continue(())
                }
            })
            .unwrap_or_else(|_| {
                panic!("SU(2) fusion closure exceeds the supported maximum doubled spin 254")
            });
        if !admissible {
            return 0.0;
        }
        racah::su2_r_symbol(
            left.twice_spin() as u32,
            right.twice_spin() as u32,
            coupled.id() as u32,
        )
    }
}

impl MultiplicityFreeRigidSymbols for SU2FusionRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        racah_irrep(SU2Irrep::from_sector_id(sector)).dim() as f64
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        1.0 / self.dim_scalar(sector)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.dim_scalar(sector).sqrt()
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        1.0 / self.sqrt_dim_scalar(sector)
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        racah::su2_frobenius_schur(SU2Irrep::from_sector_id(sector).twice_spin() as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_su2_symbol_and_channel_oracle() {
        let rule = SU2FusionRule;
        let sector = SectorId::new;

        assert_eq!(
            rule.fusion_channels(sector(1), sector(1)).as_slice(),
            &[sector(0), sector(2)]
        );
        assert_eq!(
            rule.try_fusion_channels(sector(127), sector(127))
                .unwrap()
                .last(),
            Some(&sector(254))
        );
        assert_eq!(
            rule.f_symbol_scalar(
                sector(0),
                sector(0),
                sector(0),
                sector(0),
                sector(0),
                sector(0)
            ),
            1.0
        );
        assert_eq!(
            rule.f_symbol_scalar(
                sector(1),
                sector(1),
                sector(1),
                sector(1),
                sector(0),
                sector(0)
            ),
            -0.5
        );
        assert_eq!(rule.r_symbol_scalar(sector(1), sector(1), sector(0)), -1.0);
        assert_eq!(rule.r_symbol_scalar(sector(1), sector(1), sector(255)), 0.0);
        assert_eq!(rule.frobenius_schur_phase_scalar(sector(253)), -1.0);
        assert_eq!(
            rule.f_symbol_scalar(
                sector(1),
                sector(1),
                sector(1),
                sector(1),
                sector(1),
                sector(1)
            ),
            0.0
        );
    }

    #[test]
    fn su2_has_canonical_unit_channels_and_associator() {
        // What: the SU(2) vacuum remains a unique unit at the supported
        // label boundary, with exact unit associator.
        fn accepts_canonical_unit<R: CanonicalUnitFusionRule>(_rule: &R) {}

        let rule = SU2FusionRule;
        let vacuum = rule.vacuum();
        let boundary = SectorId::new(254);
        let sample = SectorId::new(127);
        accepts_canonical_unit(&rule);
        assert_eq!(rule.dual(vacuum), vacuum);
        assert_eq!(
            rule.fusion_channels(vacuum, boundary).as_slice(),
            &[boundary]
        );
        assert_eq!(
            rule.fusion_channels(boundary, vacuum).as_slice(),
            &[boundary]
        );
        assert_eq!(rule.nsymbol(vacuum, boundary, boundary), 1);
        assert_eq!(rule.nsymbol(boundary, vacuum, boundary), 1);
        assert_eq!(rule.nsymbol(vacuum, boundary, vacuum), 0);
        for fused in rule.fusion_channels(sample, sample) {
            assert_eq!(
                rule.f_symbol_scalar(vacuum, sample, sample, fused, sample, fused),
                1.0
            );
            assert_eq!(
                rule.f_symbol_scalar(sample, vacuum, sample, fused, sample, sample),
                1.0
            );
            assert_eq!(
                rule.f_symbol_scalar(sample, sample, vacuum, fused, fused, sample),
                1.0
            );
        }
    }

    #[test]
    fn checked_su2_preserves_validation_order_and_domain() {
        let rule = SU2FusionRule;
        let invalid = SectorId::new(255);
        let later_invalid = SectorId::new(256);

        assert_eq!(
            rule.try_fusion_channels(invalid, SectorId::new(254)),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_fusion_channels(SectorId::new(254), invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_fusion_channels(invalid, later_invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_fusion_channels(SectorId::new(128), SectorId::new(127)),
            Err(FusionAlgebraError::FusionNotRepresentable {
                left: SectorId::new(128),
                right: SectorId::new(127),
            })
        );
        assert_eq!(
            rule.try_nsymbol(SectorId::new(254), invalid, SectorId::new(0)),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_nsymbol(invalid, SectorId::new(0), later_invalid),
            Err(FusionAlgebraError::InvalidSector {
                sector: later_invalid,
            })
        );
    }

    #[test]
    #[should_panic(
        expected = "SU(2) fusion closure exceeds the supported maximum doubled spin 254"
    )]
    fn r_symbol_preserves_left_right_closure_validation() {
        let _ =
            SU2FusionRule.r_symbol_scalar(SectorId::new(254), SectorId::new(1), SectorId::new(253));
    }
}
