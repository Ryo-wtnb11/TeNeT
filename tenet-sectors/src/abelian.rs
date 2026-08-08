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

#[derive(Clone, Debug)]
pub struct ZNFusionRule {
    modulus: std::num::NonZeroU32,
    identity: RuleIdentity,
}

impl PartialEq for ZNFusionRule {
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus
    }
}
impl Eq for ZNFusionRule {}
impl std::hash::Hash for ZNFusionRule {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.modulus.hash(state);
    }
}

impl ZNFusionRule {
    pub fn new(modulus: u32) -> Result<Self, FusionAlgebraError> {
        let Some(modulus) = std::num::NonZeroU32::new(modulus) else {
            return Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: RuleIdentity::of_type::<Self>(),
                label: "Z_N modulus must be nonzero".into(),
            });
        };
        let bytes: std::sync::Arc<[u8]> = modulus.get().to_le_bytes().to_vec().into();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&modulus.get(), &mut hasher);
        let identity =
            RuleIdentity::from_canonical_bytes::<Self>(std::hash::Hasher::finish(&hasher), bytes);
        Ok(Self { modulus, identity })
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
}

impl FusionRule for ZNFusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
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
        self.dual_or_panic(sector)
    }
    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.fusion_channels_or_panic(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.nsymbol_or_panic(left, right, coupled)
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

    fn dual(&self, sector: SectorId) -> SectorId {
        self.dual_or_panic(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.fusion_channels_or_panic(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.nsymbol_or_panic(left, right, coupled)
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

    fn dual(&self, sector: SectorId) -> SectorId {
        self.dual_or_panic(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.fusion_channels_or_panic(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.nsymbol_or_panic(left, right, coupled)
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
    /// Returns the U(1) label when its dual is representable by this encoding.
    pub const fn try_new(charge: i32) -> Option<Self> {
        if charge == i32::MIN {
            None
        } else {
            Some(Self { charge })
        }
    }

    /// Creates a U(1) label.
    ///
    /// # Panics
    ///
    /// Panics for `i32::MIN`, whose dual charge cannot be represented by `i32`.
    pub const fn new(charge: i32) -> Self {
        match Self::try_new(charge) {
            Some(irrep) => irrep,
            None => panic!("U(1) charge i32::MIN has no representable dual label"),
        }
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
            .and_then(Self::try_new)
    }

    pub fn checked_fuse(self, other: Self) -> Result<Self, FusionAlgebraError> {
        self.charge
            .checked_add(other.charge)
            .and_then(Self::try_new)
            .ok_or(FusionAlgebraError::U1FusionOverflow {
                left: self.charge,
                right: other.charge,
            })
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
        self.dual_or_panic(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.fusion_channels_or_panic(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.nsymbol_or_panic(left, right, coupled)
    }
}

impl CheckedFusionAlgebra for U1FusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        let irrep = checked_u1_irrep(sector)?;
        Ok(U1Irrep::new(-irrep.charge()).into())
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
        if Self::Sector::try_new(value.charge()).is_some() {
            Ok(value.sector_id())
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: self.rule_identity(),
                label: value.charge().to_string(),
            })
        }
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
        CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError, FusionRule,
        MultiplicityFreeFusionSymbols, SectorCodec, SectorId,
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
    fn u1_zigzag_encoding_is_unchanged_at_32_bit_extremes() {
        // What: rejecting one label does not change the packed zigzag layout.
        for (charge, encoded) in [
            (i32::MIN, u32::MAX),
            (-1, 1),
            (0, 0),
            (1, 2),
            (i32::MAX, u32::MAX - 1),
        ] {
            assert_eq!(u1_charge_to_zigzag_u32(charge), encoded);
            assert_eq!(u1_charge_from_zigzag_u32(encoded), charge);
            if let Some(irrep) = U1Irrep::try_new(charge) {
                let sector = irrep.sector_id();
                assert_eq!(sector.id(), encoded as usize);
                assert_eq!(U1Irrep::from_sector_id(sector), Some(irrep));
            }
        }
    }

    #[test]
    fn u1_label_boundary_rejects_only_i32_min() {
        // What: i32::MIN has no label, while both adjacent representable
        // extremes encode, decode, and dualize normally.
        let invalid = SectorId::new(u32::MAX as usize);
        assert_eq!(U1Irrep::try_new(i32::MIN), None);
        assert_eq!(U1Irrep::from_sector_id(invalid), None);
        assert_eq!(
            U1FusionRule.decode_sector(invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );

        for charge in [i32::MIN + 1, i32::MAX] {
            let irrep = U1Irrep::new(charge);
            assert_eq!(U1FusionRule.encode_sector(&irrep), Ok(irrep.sector_id()));
            assert_eq!(U1Irrep::from_sector_id(irrep.sector_id()), Some(irrep));
            assert_eq!(
                U1FusionRule
                    .try_dual_sector(irrep.sector_id())
                    .and_then(|dual| U1FusionRule.try_dual_sector(dual)),
                Ok(irrep.sector_id())
            );
        }
    }

    #[test]
    #[should_panic(expected = "U(1) charge i32::MIN has no representable dual label")]
    fn u1_new_rejects_i32_min() {
        let _ = U1Irrep::new(i32::MIN);
    }

    #[test]
    fn u1_codec_defensively_rejects_an_invalid_label() {
        // What: the codec keeps its own trust-boundary check even though safe
        // public constructors cannot create this value.
        let invalid = U1Irrep { charge: i32::MIN };
        assert_eq!(
            U1FusionRule.encode_sector(&invalid),
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: U1FusionRule.rule_identity(),
                label: i32::MIN.to_string(),
            })
        );
    }

    #[test]
    fn u1_fusion_rejects_the_unlabelled_boundary_without_panicking() {
        // What: valid inputs whose mathematical sum is i32::MIN retain the
        // checked fusion failure contract.
        let left = U1Irrep::new(i32::MIN + 1);
        let right = U1Irrep::new(-1);
        assert_eq!(
            left.checked_fuse(right),
            Err(FusionAlgebraError::U1FusionOverflow {
                left: i32::MIN + 1,
                right: -1,
            })
        );
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

    #[test]
    fn zn_boundary_labels_fuse_and_symbols_are_canonical() {
        for modulus in [1, 2, 3, 17, u32::MAX] {
            let rule = ZNFusionRule::new(modulus).unwrap();
            for charge in [i64::MIN, -1, 0, 1, i64::MAX] {
                let sector = rule.irrep(charge);
                assert_eq!(
                    rule.try_dual_sector(rule.try_dual_sector(sector.into()).unwrap())
                        .unwrap(),
                    sector.into()
                );
                assert_eq!(rule.decode_sector(sector.into()).unwrap(), sector);
            }
            let a = rule.irrep(u32::MAX as i64);
            let b = rule.irrep(1);
            let fused = rule.try_fusion_channels(a.into(), b.into()).unwrap()[0];
            assert_eq!(fused, rule.irrep((u32::MAX as u64 + 1) as i64).into());
            if modulus == u32::MAX {
                let edge = rule.irrep((u32::MAX - 1) as i64);
                assert_eq!(
                    rule.try_fusion_channels(edge.into(), edge.into()).unwrap()[0],
                    rule.irrep((u32::MAX - 2) as i64).into()
                );
            }
            assert_eq!(
                rule.f_symbol_scalar(a.into(), b.into(), rule.vacuum(), fused, a.into(), b.into()),
                1.0
            );
            assert_eq!(rule.r_symbol_scalar(a.into(), b.into(), fused), 1.0);
        }
        assert_ne!(
            ZNFusionRule::new(3).unwrap().rule_identity(),
            ZNFusionRule::new(4).unwrap().rule_identity()
        );
        assert_eq!(
            ZNFusionRule::new(3).unwrap().rule_identity(),
            ZNFusionRule::new(3).unwrap().rule_identity()
        );
    }
}
