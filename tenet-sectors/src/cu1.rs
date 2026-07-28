use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError,
    FusionRule, FusionStyleKind, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, RuleIdentity, SectorCodec, SectorId, SectorVec,
};

/// Largest `twice_charge` accepted by TeNeT's CU(1) sector codec.
pub const CU1_MAX_TWICE_CHARGE: u32 = u32::MAX - 1;

/// TensorKit's `CU1Irrep`: `0+`, `0-`, or a positive half-integral charge.
///
/// The representation is private so `Charged(0)` and the forbidden
/// charge-conjugation labels cannot be constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CU1Irrep(CU1Kind);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum CU1Kind {
    Vacuum,
    Pseudoscalar,
    Charged(u32),
}

impl CU1Irrep {
    pub const VACUUM: Self = Self(CU1Kind::Vacuum);
    pub const PSEUDOSCALAR: Self = Self(CU1Kind::Pseudoscalar);

    pub const fn try_from_twice_charge(twice_charge: u32) -> Option<Self> {
        if twice_charge > 0 && twice_charge <= CU1_MAX_TWICE_CHARGE {
            Some(Self(CU1Kind::Charged(twice_charge)))
        } else {
            None
        }
    }

    pub const fn from_twice_charge(twice_charge: u32) -> Self {
        match Self::try_from_twice_charge(twice_charge) {
            Some(irrep) => irrep,
            None => panic!("CU(1) twice charge must be in 1..=u32::MAX-1"),
        }
    }

    pub const fn twice_charge(self) -> u32 {
        match self.0 {
            CU1Kind::Charged(charge) => charge,
            CU1Kind::Vacuum | CU1Kind::Pseudoscalar => 0,
        }
    }

    /// TensorKit's `s`: 0 for `0+`, 1 for `0-`, and 2 for charged sectors.
    pub const fn conjugation_sector(self) -> u8 {
        match self.0 {
            CU1Kind::Vacuum => 0,
            CU1Kind::Pseudoscalar => 1,
            CU1Kind::Charged(_) => 2,
        }
    }

    pub const fn sector_id(self) -> SectorId {
        match self.0 {
            CU1Kind::Vacuum => SectorId::new(0),
            CU1Kind::Pseudoscalar => SectorId::new(1),
            CU1Kind::Charged(charge) => SectorId::new(charge as usize + 1),
        }
    }

    pub fn try_from_sector_id(sector: SectorId) -> Option<Self> {
        match sector.id() {
            0 => Some(Self::VACUUM),
            1 => Some(Self::PSEUDOSCALAR),
            id => match u32::try_from(id - 1) {
                Ok(charge) => Self::try_from_twice_charge(charge),
                Err(_) => None,
            },
        }
    }

    fn is_charged(self) -> bool {
        matches!(self.0, CU1Kind::Charged(_))
    }
}

impl From<CU1Irrep> for SectorId {
    fn from(value: CU1Irrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct CU1FusionRule;

fn checked_irrep(sector: SectorId) -> Result<CU1Irrep, FusionAlgebraError> {
    CU1Irrep::try_from_sector_id(sector).ok_or(FusionAlgebraError::InvalidSector { sector })
}

fn checked_irreps(
    left: SectorId,
    right: SectorId,
) -> Result<(CU1Irrep, CU1Irrep), FusionAlgebraError> {
    Ok((checked_irrep(left)?, checked_irrep(right)?))
}

fn channels(left: CU1Irrep, right: CU1Irrep) -> Result<SectorVec, FusionAlgebraError> {
    match (left.0, right.0) {
        (CU1Kind::Vacuum, _) => Ok(core::iter::once(right.into()).collect()),
        (_, CU1Kind::Vacuum) => Ok(core::iter::once(left.into()).collect()),
        (CU1Kind::Pseudoscalar, CU1Kind::Pseudoscalar) => {
            Ok(core::iter::once(CU1Irrep::VACUUM.into()).collect())
        }
        (CU1Kind::Pseudoscalar, CU1Kind::Charged(_))
        | (CU1Kind::Charged(_), CU1Kind::Pseudoscalar) => {
            Ok(core::iter::once(left.max(right).into()).collect())
        }
        (CU1Kind::Charged(a), CU1Kind::Charged(b)) if a == b => {
            let doubled = a.checked_mul(2).filter(|&q| q <= CU1_MAX_TWICE_CHARGE);
            let Some(doubled) = doubled else {
                return Err(FusionAlgebraError::FusionNotRepresentable {
                    left: left.into(),
                    right: right.into(),
                });
            };
            Ok([
                CU1Irrep::VACUUM,
                CU1Irrep::PSEUDOSCALAR,
                CU1Irrep::from_twice_charge(doubled),
            ]
            .into_iter()
            .map(Into::into)
            .collect())
        }
        (CU1Kind::Charged(a), CU1Kind::Charged(b)) => {
            let sum = a.checked_add(b).filter(|&q| q <= CU1_MAX_TWICE_CHARGE);
            let Some(sum) = sum else {
                return Err(FusionAlgebraError::FusionNotRepresentable {
                    left: left.into(),
                    right: right.into(),
                });
            };
            Ok([
                CU1Irrep::from_twice_charge(a.abs_diff(b)),
                CU1Irrep::from_twice_charge(sum),
            ]
            .into_iter()
            .map(Into::into)
            .collect())
        }
    }
}

impl FusionRule for CU1FusionRule {
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
        CU1Irrep::VACUUM.into()
    }
    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        channels(
            CU1Irrep::try_from_sector_id(left).expect("CU(1) fusion received an invalid sector"),
            CU1Irrep::try_from_sector_id(right).expect("CU(1) fusion received an invalid sector"),
        )
        .expect("CU(1) fusion output is not representable")
    }
}

impl CheckedFusionAlgebra for CU1FusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(checked_irrep(sector)?.into())
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let (left, right) = checked_irreps(left, right)?;
        channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let coupled = checked_irrep(coupled)?;
        let (left, right) = checked_irreps(left, right)?;
        Ok(usize::from(
            channels(left, right)?
                .iter()
                .any(|&sector| sector == coupled.into()),
        ))
    }
}

impl SectorCodec for CU1FusionRule {
    type Sector = CU1Irrep;
    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        Ok((*value).into())
    }
    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        checked_irrep(id)
    }
}

impl MultiplicityFreeFusionRule for CU1FusionRule {}
impl CanonicalUnitFusionRule for CU1FusionRule {}

impl MultiplicityFreeFusionSymbols for CU1FusionRule {
    type Scalar = f64;
    fn scalar_one(&self) -> f64 {
        1.0
    }
    fn scalar_conj(&self, value: f64) -> f64 {
        value
    }

    // TensorKitSectors 0.3.4 `cu1irrep.jl:123-196`, including its gauge.
    fn f_symbol_scalar(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> f64 {
        if self.nsymbol(a, b, e) == 0
            || self.nsymbol(e, c, d) == 0
            || self.nsymbol(b, c, f) == 0
            || self.nsymbol(a, f, d) == 0
        {
            return 0.0;
        }
        let (a, b, c, d, e, f) = (
            CU1Irrep::try_from_sector_id(a).unwrap(),
            CU1Irrep::try_from_sector_id(b).unwrap(),
            CU1Irrep::try_from_sector_id(c).unwrap(),
            CU1Irrep::try_from_sector_id(d).unwrap(),
            CU1Irrep::try_from_sector_id(e).unwrap(),
            CU1Irrep::try_from_sector_id(f).unwrap(),
        );
        let op = CU1Irrep::VACUUM;
        let om = CU1Irrep::PSEUDOSCALAR;
        if a == op
            || b == op
            || c == op
            || (a == b && b == om)
            || (a == c && c == om)
            || (b == c && c == om)
        {
            return 1.0;
        }
        if a == om {
            return if d.twice_charge() == 0
                || d.twice_charge() as i64 != c.twice_charge() as i64 - b.twice_charge() as i64
            {
                1.0
            } else {
                -1.0
            };
        }
        if b == om {
            return if d.twice_charge() == a.twice_charge().abs_diff(c.twice_charge()) {
                -1.0
            } else {
                1.0
            };
        }
        if c == om {
            return if d.twice_charge() as i64 == a.twice_charge() as i64 - b.twice_charge() as i64 {
                -1.0
            } else {
                1.0
            };
        }
        let s = 2.0_f64.sqrt() / 2.0;
        if a == b && b == c {
            if d == a {
                if e.twice_charge() == 0 {
                    if f.twice_charge() == 0 {
                        return if f == om { -0.5 } else { 0.5 };
                    }
                    return if e == om { -s } else { s };
                }
                return if f.twice_charge() == 0 { s } else { 0.0 };
            }
            return 1.0;
        }
        if a == b {
            return if d == c {
                if f.twice_charge() as u64 == b.twice_charge() as u64 + c.twice_charge() as u64 {
                    if e == om {
                        -s
                    } else {
                        s
                    }
                } else {
                    s
                }
            } else {
                1.0
            };
        }
        if b == c {
            return if d == a {
                if e.twice_charge() as u64 == a.twice_charge() as u64 + b.twice_charge() as u64 {
                    s
                } else if f == om {
                    -s
                } else {
                    s
                }
            } else {
                1.0
            };
        }
        if a == c {
            return if d == b {
                if e.twice_charge() == f.twice_charge() {
                    0.0
                } else {
                    1.0
                }
            } else if d == om {
                -1.0
            } else {
                1.0
            };
        }
        if d == om && b.twice_charge() as u64 == a.twice_charge() as u64 + c.twice_charge() as u64 {
            -1.0
        } else {
            1.0
        }
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> f64 {
        let left = CU1Irrep::try_from_sector_id(left).expect("CU(1) R received an invalid sector");
        if self.nsymbol(left.into(), right, coupled) == 0 {
            return 0.0;
        }
        let coupled =
            CU1Irrep::try_from_sector_id(coupled).expect("CU(1) R received an invalid sector");
        if coupled == CU1Irrep::PSEUDOSCALAR && left.is_charged() {
            -1.0
        } else {
            1.0
        }
    }
}

impl MultiplicityFreeRigidSymbols for CU1FusionRule {
    fn dim_scalar(&self, sector: SectorId) -> f64 {
        if checked_irrep(sector)
            .expect("CU(1) dimension received an invalid sector")
            .is_charged()
        {
            2.0
        } else {
            1.0
        }
    }
    fn inv_dim_scalar(&self, sector: SectorId) -> f64 {
        1.0 / self.dim_scalar(sector)
    }
    fn sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
        self.dim_scalar(sector).sqrt()
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
        1.0 / self.sqrt_dim_scalar(sector)
    }
    fn twist_scalar(&self, _sector: SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> f64 {
        1.0
    }
}
