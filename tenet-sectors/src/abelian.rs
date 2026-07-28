use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError,
    FusionRule, FusionStyleKind, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, RuleIdentity, SectorCodec, SectorId, SectorVec,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Z2Irrep {
    parity: u8,
}

/// Irreducible sector of the finite cyclic group Z_N.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ZNIrrep {
    charge: u32,
}

impl ZNIrrep {
    pub const fn charge(self) -> u32 {
        self.charge
    }

    pub const fn sector_id(self) -> SectorId {
        SectorId::new(self.charge as usize)
    }
}

impl From<ZNIrrep> for SectorId {
    fn from(value: ZNIrrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ZNFusionRule {
    modulus: std::num::NonZeroU32,
}

impl ZNFusionRule {
    pub fn new(modulus: u32) -> Result<Self, FusionAlgebraError> {
        let Some(modulus) = std::num::NonZeroU32::new(modulus) else {
            return Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: RuleIdentity::of_type::<Self>(),
                label: "Z_N modulus must be nonzero".into(),
            });
        };
        Ok(Self { modulus })
    }

    pub const fn modulus(&self) -> u32 {
        self.modulus.get()
    }

    pub fn irrep(&self, charge: i64) -> ZNIrrep {
        let n = self.modulus() as i64;
        ZNIrrep {
            charge: charge.rem_euclid(n) as u32,
        }
    }

    fn checked(&self, sector: SectorId) -> Result<ZNIrrep, FusionAlgebraError> {
        let charge = u32::try_from(sector.id()).ok();
        match charge.filter(|&charge| charge < self.modulus()) {
            Some(charge) => Ok(ZNIrrep { charge }),
            None => Err(FusionAlgebraError::InvalidSector { sector }),
        }
    }

    fn identity(&self) -> RuleIdentity {
        let bytes: std::sync::Arc<[u8]> = self.modulus().to_le_bytes().to_vec().into();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&self.modulus(), &mut hasher);
        RuleIdentity::from_canonical_bytes::<Self>(std::hash::Hasher::finish(&hasher), bytes)
    }
}

impl FusionRule for ZNFusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        self.irrep(0).into()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        let sector = self
            .checked(sector)
            .expect("Z_N dual received an invalid sector");
        self.irrep(-(sector.charge as i64)).into()
    }
    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let left = self
            .checked(left)
            .expect("Z_N fusion received an invalid sector");
        let right = self
            .checked(right)
            .expect("Z_N fusion received an invalid sector");
        core::iter::once(
            self.irrep((left.charge as u64 + right.charge as u64) as i64)
                .into(),
        )
        .collect()
    }
}

impl CheckedFusionAlgebra for ZNFusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        let sector = self.checked(sector)?;
        Ok(self.irrep(-(sector.charge as i64)).into())
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let left = self.checked(left)?;
        let right = self.checked(right)?;
        Ok(core::iter::once(
            self.irrep((left.charge as u64 + right.charge as u64) as i64)
                .into(),
        )
        .collect())
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        self.checked(coupled)?;
        Ok(usize::from(
            coupled == self.try_fusion_channels(left, right)?[0],
        ))
    }
}

impl SectorCodec for ZNFusionRule {
    type Sector = ZNIrrep;
    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        if value.charge < self.modulus() {
            Ok((*value).into())
        } else {
            Err(FusionAlgebraError::InvalidSector {
                sector: value.sector_id(),
            })
        }
    }
    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        self.checked(id)
    }
}

impl MultiplicityFreeFusionRule for ZNFusionRule {}
impl CanonicalUnitFusionRule for ZNFusionRule {}
impl MultiplicityFreeFusionSymbols for ZNFusionRule {
    type Scalar = f64;
    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }
    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }
    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> Self::Scalar {
        1.0
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> Self::Scalar {
        1.0
    }
}
impl MultiplicityFreeRigidSymbols for ZNFusionRule {
    fn dim_scalar(&self, _: SectorId) -> Self::Scalar {
        1.0
    }
    fn inv_dim_scalar(&self, _: SectorId) -> Self::Scalar {
        1.0
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> Self::Scalar {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> Self::Scalar {
        1.0
    }
    fn twist_scalar(&self, _: SectorId) -> Self::Scalar {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> Self::Scalar {
        1.0
    }
}

impl Z2Irrep {
    pub const EVEN: Self = Self { parity: 0 };
    pub const ODD: Self = Self { parity: 1 };

    pub const fn new(parity: u8) -> Self {
        Self { parity: parity & 1 }
    }

    #[inline]
    pub const fn parity(self) -> u8 {
        self.parity
    }

    #[inline]
    pub const fn sector_id(self) -> SectorId {
        SectorId::new(self.parity as usize)
    }

    pub const fn from_sector_id(sector: SectorId) -> Option<Self> {
        match sector.id() {
            0 => Some(Self::EVEN),
            1 => Some(Self::ODD),
            _ => None,
        }
    }
}

impl From<Z2Irrep> for SectorId {
    fn from(value: Z2Irrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Z2FusionRule;

impl FusionRule for Z2FusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        Z2Irrep::EVEN.into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let left = Z2Irrep::from_sector_id(left).expect("Z2 fusion received an invalid sector");
        let right = Z2Irrep::from_sector_id(right).expect("Z2 fusion received an invalid sector");
        core::iter::once(Z2Irrep::new(left.parity() ^ right.parity()).into()).collect()
    }
}

fn checked_z2_irrep(sector: SectorId) -> Result<Z2Irrep, FusionAlgebraError> {
    Z2Irrep::from_sector_id(sector).ok_or(FusionAlgebraError::InvalidSector { sector })
}

impl CheckedFusionAlgebra for Z2FusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        checked_z2_irrep(sector)?;
        Ok(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let left = checked_z2_irrep(left)?;
        let right = checked_z2_irrep(right)?;
        Ok(core::iter::once(Z2Irrep::new(left.parity() ^ right.parity()).into()).collect())
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let left = checked_z2_irrep(left)?;
        let right = checked_z2_irrep(right)?;
        let coupled = checked_z2_irrep(coupled)?;
        Ok(usize::from(
            coupled.parity() == (left.parity() ^ right.parity()),
        ))
    }
}

impl SectorCodec for Z2FusionRule {
    type Sector = Z2Irrep;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        Ok(value.sector_id())
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        checked_z2_irrep(id)
    }
}

impl MultiplicityFreeFusionRule for Z2FusionRule {}

impl CanonicalUnitFusionRule for Z2FusionRule {}

impl MultiplicityFreeFusionSymbols for Z2FusionRule {
    type Scalar = f64;

    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for Z2FusionRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct FermionParityFusionRule;

impl FusionRule for FermionParityFusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Fermionic
    }

    fn vacuum(&self) -> SectorId {
        Z2Irrep::EVEN.into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        Z2FusionRule.fusion_channels(left, right)
    }
}

impl CheckedFusionAlgebra for FermionParityFusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Z2FusionRule.try_dual_sector(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Z2FusionRule.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Z2FusionRule.try_nsymbol(left, right, coupled)
    }
}

impl SectorCodec for FermionParityFusionRule {
    type Sector = Z2Irrep;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        Ok(value.sector_id())
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        checked_z2_irrep(id)
    }
}

impl MultiplicityFreeFusionRule for FermionParityFusionRule {}

impl CanonicalUnitFusionRule for FermionParityFusionRule {}

impl MultiplicityFreeFusionSymbols for FermionParityFusionRule {
    type Scalar = f64;

    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
        if left == Z2Irrep::ODD.into() && right == Z2Irrep::ODD.into() {
            -1.0
        } else {
            1.0
        }
    }
}

impl MultiplicityFreeRigidSymbols for FermionParityFusionRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector == Z2Irrep::ODD.into() {
            -1.0
        } else {
            1.0
        }
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct U1Irrep {
    charge: i32,
}

impl U1Irrep {
    pub const fn new(charge: i32) -> Self {
        Self { charge }
    }

    #[inline]
    pub const fn charge(self) -> i32 {
        self.charge
    }

    pub const fn sector_id(self) -> SectorId {
        SectorId::new(u1_charge_to_zigzag_u32(self.charge) as usize)
    }

    pub fn from_sector_id(sector: SectorId) -> Option<Self> {
        u32::try_from(sector.id())
            .ok()
            .map(u1_charge_from_zigzag_u32)
            .map(Self::new)
    }

    pub fn checked_dual(self) -> Result<Self, FusionAlgebraError> {
        self.charge
            .checked_neg()
            .map(Self::new)
            .ok_or(FusionAlgebraError::U1DualOverflow {
                charge: self.charge,
            })
    }

    pub fn checked_fuse(self, other: Self) -> Result<Self, FusionAlgebraError> {
        self.charge.checked_add(other.charge).map(Self::new).ok_or(
            FusionAlgebraError::U1FusionOverflow {
                left: self.charge,
                right: other.charge,
            },
        )
    }
}

const fn u1_charge_to_zigzag_u32(charge: i32) -> u32 {
    ((charge as u32) << 1) ^ ((charge >> 31) as u32)
}

const fn u1_charge_from_zigzag_u32(encoded: u32) -> i32 {
    ((encoded >> 1) as i32) ^ -((encoded & 1) as i32)
}

fn checked_u1_irrep(sector: SectorId) -> Result<U1Irrep, FusionAlgebraError> {
    U1Irrep::from_sector_id(sector).ok_or(FusionAlgebraError::InvalidSector { sector })
}

impl From<U1Irrep> for SectorId {
    fn from(value: U1Irrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct U1FusionRule;

impl FusionRule for U1FusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        U1Irrep::new(0).into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        let sector = U1Irrep::from_sector_id(sector).expect("U(1) dual received an invalid sector");
        sector
            .checked_dual()
            .expect("U(1) dual charge overflow")
            .into()
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let left = U1Irrep::from_sector_id(left).expect("U(1) fusion received an invalid sector");
        let right = U1Irrep::from_sector_id(right).expect("U(1) fusion received an invalid sector");
        core::iter::once(
            left.checked_fuse(right)
                .expect("U(1) fusion charge overflow")
                .into(),
        )
        .collect()
    }
}

impl CheckedFusionAlgebra for U1FusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(checked_u1_irrep(sector)?.checked_dual()?.into())
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let left = checked_u1_irrep(left)?;
        let right = checked_u1_irrep(right)?;
        Ok(core::iter::once(left.checked_fuse(right)?.into()).collect())
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let left = checked_u1_irrep(left)?;
        let right = checked_u1_irrep(right)?;
        let coupled = checked_u1_irrep(coupled)?;
        Ok(usize::from(coupled == left.checked_fuse(right)?))
    }
}

impl SectorCodec for U1FusionRule {
    type Sector = U1Irrep;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        Ok(value.sector_id())
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        checked_u1_irrep(id)
    }
}

impl MultiplicityFreeFusionRule for U1FusionRule {}

impl CanonicalUnitFusionRule for U1FusionRule {}

impl MultiplicityFreeFusionSymbols for U1FusionRule {
    type Scalar = f64;

    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for U1FusionRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn a_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn b_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        u1_charge_from_zigzag_u32, u1_charge_to_zigzag_u32, FermionParityFusionRule, U1FusionRule,
        U1Irrep, Z2FusionRule, Z2Irrep, ZNFusionRule,
    };
    use crate::{
        CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionRule, MultiplicityFreeFusionSymbols,
        SectorId,
    };

    fn assert_canonical_unit<R>(rule: &R, sector: SectorId)
    where
        R: CanonicalUnitFusionRule + MultiplicityFreeFusionSymbols<Scalar = f64>,
    {
        let vacuum = rule.vacuum();
        let fused = rule.fusion_channels(sector, sector)[0];
        assert_eq!(rule.dual(vacuum), vacuum);
        assert_eq!(rule.fusion_channels(vacuum, sector).as_slice(), &[sector]);
        assert_eq!(rule.fusion_channels(sector, vacuum).as_slice(), &[sector]);
        assert_eq!(rule.nsymbol(vacuum, sector, sector), 1);
        assert_eq!(rule.nsymbol(sector, vacuum, sector), 1);
        assert_eq!(rule.nsymbol(vacuum, sector, vacuum), 0);
        assert_eq!(
            rule.f_symbol_scalar(vacuum, sector, sector, fused, sector, fused),
            1.0
        );
        assert_eq!(
            rule.f_symbol_scalar(sector, vacuum, sector, fused, sector, sector),
            1.0
        );
        assert_eq!(
            rule.f_symbol_scalar(sector, sector, vacuum, fused, fused, sector),
            1.0
        );
    }

    #[test]
    fn abelian_providers_have_canonical_unit_channels_and_associators() {
        // What: certified abelian providers keep the vacuum as a unique,
        // self-dual identity with an exact unit F-symbol.
        assert_canonical_unit(&Z2FusionRule, Z2Irrep::ODD.into());
        assert_canonical_unit(&FermionParityFusionRule, Z2Irrep::ODD.into());
        assert_canonical_unit(&U1FusionRule, U1Irrep::new(-17).into());
        let u1_boundary: SectorId = U1Irrep::new(i32::MAX).into();
        let u1_vacuum = U1FusionRule.vacuum();
        assert_eq!(
            U1FusionRule
                .fusion_channels(u1_vacuum, u1_boundary)
                .as_slice(),
            &[u1_boundary]
        );
        assert_eq!(
            U1FusionRule
                .fusion_channels(u1_boundary, u1_vacuum)
                .as_slice(),
            &[u1_boundary]
        );
    }

    #[test]
    fn u1_zigzag_roundtrips_native_and_simulated_32_bit_extremes() {
        // What: every i32 charge has its historical u32 zigzag ID without
        // target-width arithmetic.
        for (charge, encoded) in [
            (i32::MIN, u32::MAX),
            (-1, 1),
            (0, 0),
            (1, 2),
            (i32::MAX, u32::MAX - 1),
        ] {
            assert_eq!(u1_charge_to_zigzag_u32(charge), encoded);
            assert_eq!(u1_charge_from_zigzag_u32(encoded), charge);
            let sector = U1Irrep::new(charge).sector_id();
            assert_eq!(sector.id(), encoded as usize);
            assert_eq!(U1Irrep::from_sector_id(sector), Some(U1Irrep::new(charge)));
        }
    }

    #[test]
    fn zn_fuses_wraps_and_identity_contains_modulus() {
        let a = ZNFusionRule::new(3).unwrap();
        let b = ZNFusionRule::new(4).unwrap();
        assert_eq!(a.irrep(-1).charge(), 2);
        assert_eq!(
            a.try_fusion_channels(a.irrep(2).into(), a.irrep(2).into())
                .unwrap()[0],
            a.irrep(1).into()
        );
        assert_ne!(a.rule_identity(), b.rule_identity());
        assert!(ZNFusionRule::new(0).is_err());
    }

    #[test]
    fn zn_checked_rejects_out_of_domain_sector() {
        let rule = ZNFusionRule::new(3).unwrap();
        assert!(matches!(
            rule.try_dual_sector(SectorId::new(3)),
            Err(crate::FusionAlgebraError::InvalidSector { .. })
        ));
    }
}
