use core::marker::PhantomData;
use std::sync::OnceLock;

use crate::{
    BraidingStyleKind, CheckedFusionAlgebra, FusionAlgebraError, FusionRule, FusionStyleKind,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    ProductSectorCodec, RuleIdentity, SectorId, SectorVec, TensorKitProductCodec,
};

#[derive(Clone, Debug)]
pub struct ProductFusionRule<LeftRule, RightRule, Codec = TensorKitProductCodec> {
    left: LeftRule,
    right: RightRule,
    _codec: PhantomData<Codec>,
    identity: OnceLock<RuleIdentity>,
}

impl<LeftRule, RightRule, Codec> ProductFusionRule<LeftRule, RightRule, Codec> {
    pub const fn new(left: LeftRule, right: RightRule) -> Self {
        Self {
            left,
            right,
            _codec: PhantomData,
            identity: OnceLock::new(),
        }
    }

    #[inline]
    pub const fn left_rule(&self) -> &LeftRule {
        &self.left
    }

    #[inline]
    pub const fn right_rule(&self) -> &RightRule {
        &self.right
    }

    pub fn encode_sector(&self, left: SectorId, right: SectorId) -> SectorId
    where
        Codec: ProductSectorCodec,
    {
        Codec::encode(left, right)
    }

    pub fn try_encode_sector(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorId, FusionAlgebraError>
    where
        Codec: ProductSectorCodec,
    {
        Codec::encode_checked(left, right).map_err(FusionAlgebraError::ProductCodec)
    }

    pub fn decode_sector(&self, sector: SectorId) -> Option<(SectorId, SectorId)>
    where
        Codec: ProductSectorCodec,
    {
        Codec::decode(sector)
    }

    fn decode_sector_or_panic(&self, sector: SectorId) -> (SectorId, SectorId)
    where
        Codec: ProductSectorCodec,
    {
        self.decode_sector(sector)
            .expect("product fusion rule received an invalid product sector")
    }
}

pub const fn product_fusion_rule<LeftRule, RightRule>(
    left: LeftRule,
    right: RightRule,
) -> ProductFusionRule<LeftRule, RightRule> {
    ProductFusionRule::new(left, right)
}

pub const fn product_fusion_rule_with_codec<LeftRule, RightRule, Codec>(
    left: LeftRule,
    right: RightRule,
) -> ProductFusionRule<LeftRule, RightRule, Codec> {
    ProductFusionRule::new(left, right)
}

pub trait ProductFusionRuleExt: FusionRule + Sized {
    fn product<RightRule>(self, right: RightRule) -> ProductFusionRule<Self, RightRule>
    where
        RightRule: FusionRule,
    {
        ProductFusionRule::new(self, right)
    }
}

impl<Rule> ProductFusionRuleExt for Rule where Rule: FusionRule + Sized {}

impl<LeftRule, RightRule, Codec> Default for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: Default,
    RightRule: Default,
{
    fn default() -> Self {
        Self::new(LeftRule::default(), RightRule::default())
    }
}

impl<LeftRule, RightRule, Codec> FusionRule for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: FusionRule,
    RightRule: FusionRule,
    Codec: ProductSectorCodec + 'static,
{
    fn rule_identity(&self) -> RuleIdentity {
        self.identity
            .get_or_init(|| {
                RuleIdentity::compose_with_codec::<Codec>(
                    self.left.rule_identity(),
                    self.right.rule_identity(),
                )
            })
            .clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.left
            .fusion_style()
            .combined_with(self.right.fusion_style())
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.left
            .braiding_style()
            .combined_with(self.right.braiding_style())
    }

    fn vacuum(&self) -> SectorId {
        self.encode_sector(self.left.vacuum(), self.right.vacuum())
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        self.left.supports_unitary_braid_dagger() && self.right.supports_unitary_braid_dagger()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.encode_sector(self.left.dual(left), self.right.dual(right))
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let (left_left, left_right) = self.decode_sector_or_panic(left);
        let (right_left, right_right) = self.decode_sector_or_panic(right);
        let left_channels = self.left.fusion_channels(left_left, right_left);
        let right_channels = self.right.fusion_channels(left_right, right_right);
        // Cartesian product of the two sub-rules' channels, matching TensorKit's
        // `⊗(p1,p2) = SectorSet(product(map(⊗, ...)))`. No dedup: each sub-rule
        // is multiplicity-free (distinct channels) and `encode_sector` is the
        // Cantor pairing (a bijection), so distinct (left,right) pairs always
        // encode to distinct ids — the old `channels.contains()` guard was
        // provably dead and made this O(k²) instead of O(k) in k = |L|·|R|.
        let mut channels = SectorVec::with_capacity(left_channels.len() * right_channels.len());
        for right_channel in right_channels {
            for &left_channel in &left_channels {
                channels.push(self.encode_sector(left_channel, right_channel));
            }
        }
        channels
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        let (left_left, left_right) = self.decode_sector_or_panic(left);
        let (right_left, right_right) = self.decode_sector_or_panic(right);
        let (coupled_left, coupled_right) = self.decode_sector_or_panic(coupled);
        self.left.nsymbol(left_left, right_left, coupled_left)
            * self.right.nsymbol(left_right, right_right, coupled_right)
    }
}

impl<LeftRule, RightRule, Codec> MultiplicityFreeFusionRule
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeFusionRule,
    RightRule: MultiplicityFreeFusionRule,
    Codec: ProductSectorCodec + 'static,
{
}

impl<LeftRule, RightRule, Codec> CheckedFusionAlgebra
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: CheckedFusionAlgebra,
    RightRule: CheckedFusionAlgebra,
    Codec: ProductSectorCodec + 'static,
{
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        let (left, right) =
            Codec::decode_checked(sector).map_err(FusionAlgebraError::ProductCodec)?;
        self.try_encode_sector(
            self.left.try_dual_sector(left)?,
            self.right.try_dual_sector(right)?,
        )
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let (left_left, left_right) =
            Codec::decode_checked(left).map_err(FusionAlgebraError::ProductCodec)?;
        let (right_left, right_right) =
            Codec::decode_checked(right).map_err(FusionAlgebraError::ProductCodec)?;
        let left_channels = self.left.try_fusion_channels(left_left, right_left)?;
        let right_channels = self.right.try_fusion_channels(left_right, right_right)?;
        let mut channels = SectorVec::new();
        for right_channel in right_channels {
            for &left_channel in &left_channels {
                channels.push(self.try_encode_sector(left_channel, right_channel)?);
            }
        }
        Ok(channels)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let (left_left, left_right) =
            Codec::decode_checked(left).map_err(FusionAlgebraError::ProductCodec)?;
        let (right_left, right_right) =
            Codec::decode_checked(right).map_err(FusionAlgebraError::ProductCodec)?;
        let (coupled_left, coupled_right) =
            Codec::decode_checked(coupled).map_err(FusionAlgebraError::ProductCodec)?;
        self.left
            .try_nsymbol(left_left, right_left, coupled_left)?
            .checked_mul(
                self.right
                    .try_nsymbol(left_right, right_right, coupled_right)?,
            )
            .ok_or(FusionAlgebraError::MultiplicityOverflow {
                left,
                right,
                coupled,
            })
    }
}

impl<LeftRule, RightRule, Codec> MultiplicityFreeFusionSymbols
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeFusionSymbols<Scalar = f64>,
    RightRule: MultiplicityFreeFusionSymbols<Scalar = f64>,
    Codec: ProductSectorCodec + 'static,
{
    type Scalar = f64;

    fn has_trivial_associator_gauge(&self) -> bool {
        self.left.has_trivial_associator_gauge() && self.right.has_trivial_associator_gauge()
    }

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
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (middle_l, middle_r) = self.decode_sector_or_panic(middle);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        let (left_coupled_l, left_coupled_r) = self.decode_sector_or_panic(left_coupled);
        let (right_coupled_l, right_coupled_r) = self.decode_sector_or_panic(right_coupled);
        self.left.f_symbol_scalar(
            left_l,
            middle_l,
            right_l,
            coupled_l,
            left_coupled_l,
            right_coupled_l,
        ) * self.right.f_symbol_scalar(
            left_r,
            middle_r,
            right_r,
            coupled_r,
            left_coupled_r,
            right_coupled_r,
        )
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        self.left.r_symbol_scalar(left_l, right_l, coupled_l)
            * self.right.r_symbol_scalar(left_r, right_r, coupled_r)
    }
}

impl<LeftRule, RightRule, Codec> MultiplicityFreeRigidSymbols
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeRigidSymbols<Scalar = f64>,
    RightRule: MultiplicityFreeRigidSymbols<Scalar = f64>,
    // Sync via the trait's supertrait; the codec is a PhantomData marker.
    Codec: ProductSectorCodec + Sync + 'static,
{
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.dim_scalar(left) * self.right.dim_scalar(right)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.inv_dim_scalar(left) * self.right.inv_dim_scalar(right)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.sqrt_dim_scalar(left) * self.right.sqrt_dim_scalar(right)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.inv_sqrt_dim_scalar(left) * self.right.inv_sqrt_dim_scalar(right)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.twist_scalar(left) * self.right.twist_scalar(right)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.frobenius_schur_phase_scalar(left)
            * self.right.frobenius_schur_phase_scalar(right)
    }

    fn a_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        self.left.a_symbol_scalar(left_l, right_l, coupled_l)
            * self.right.a_symbol_scalar(left_r, right_r, coupled_r)
    }

    fn b_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        self.left.b_symbol_scalar(left_l, right_l, coupled_l)
            * self.right.b_symbol_scalar(left_r, right_r, coupled_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FermionParityFusionRule, U1FusionRule, U1Irrep, Z2Irrep};

    #[test]
    fn product_rule_composes_checked_symbols_rigidity_and_ordered_identity() {
        let rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
        let odd_zero = rule.encode_sector(Z2Irrep::ODD.into(), U1Irrep::new(0).into());
        let vacuum = rule.vacuum();

        assert_eq!(
            rule.try_fusion_channels(odd_zero, odd_zero),
            Ok(std::iter::once(vacuum).collect())
        );
        assert_eq!(rule.r_symbol_scalar(odd_zero, odd_zero, vacuum), -1.0);
        assert_eq!(
            rule.f_symbol_scalar(vacuum, vacuum, vacuum, vacuum, vacuum, vacuum),
            1.0
        );
        assert_eq!(rule.dim_scalar(odd_zero), 1.0);
        assert!(rule.identity.get().is_none());
        let first = rule.rule_identity();
        let cached = rule.identity.get().unwrap() as *const RuleIdentity;
        assert_eq!(first, rule.rule_identity());
        assert_eq!(cached, rule.identity.get().unwrap() as *const RuleIdentity);
        assert_ne!(
            rule.rule_identity(),
            product_fusion_rule(U1FusionRule, FermionParityFusionRule).rule_identity()
        );
    }
}
