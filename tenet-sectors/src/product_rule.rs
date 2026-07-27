use core::marker::PhantomData;
use std::sync::OnceLock;

use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError,
    FusionRule, FusionStyleKind, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, ProductSector, ProductSectorCodec, RuleIdentity, SectorCodec,
    SectorId, SectorVec, TensorKitProductCodec,
};

/// The Deligne product `Left ⊠ Right` of two providers: one fusion rule over
/// [`ProductSector<L, R>`](ProductSector) labels whose fusion, dual, F/R and
/// dimension data are the componentwise combination of its factors'.
///
/// This is the generic product mechanism. Any ordered pair of providers is a
/// product, and because a `ProductFusionRule` is itself a provider, three or
/// more factors are the same construction applied again — no new type, enum
/// arm or dispatch branch per physical symmetry. Build one with
/// [`ProductFusionRuleExt::product`], which is where the worked example lives.
///
/// `Codec` decides how a component id pair is packed into one
/// [`SectorId`](crate::SectorId); the default
/// [`TensorKitProductCodec`](crate::TensorKitProductCodec) works for any pair
/// of providers, and a fixed-width
/// [`PackedProductCodec`](crate::PackedProductCodec) is available where the
/// component ranges are known. The codec is an id-packing choice only: two
/// rules that differ solely in `Codec` label the same category, but they are
/// different Rust types and report different
/// [`RuleIdentity`](crate::RuleIdentity)s, so tensors built on them do not mix.
///
/// TensorKit correspondence: `ProductSector` in TensorKitSectors 0.3.4
/// (`src/product.jl:245-294`), documented in TensorKit 0.17.0
/// `docs/src/man/sectors.md:468-506`. TensorKit's convenience aliases
/// `FermionNumber = U1Irrep ⊠ FermionParity` and
/// `FermionSpin = SU2Irrep ⊠ FermionParity` (`src/fermions.jl:71-102`) are
/// *not* a universal factor-order rule: each exists because it enforces an
/// extra invariant (parity tied to the charge, respectively to the spin), not
/// because that order is canonical.
///
/// One deliberate divergence from that correspondence: TensorKit's `⊠`
/// *flattens*, so `(A ⊠ B) ⊠ C` and `A ⊠ (B ⊠ C)` are the same
/// `ProductSector{Tuple{A,B,C}}` and association is unobservable there.
/// TeNeT keeps the nesting in the Rust type — `ProductFusionRule<ProductFusionRule<A, B>, C>`
/// over `ProductSector<ProductSector<A, B>, C>` labels — so the two
/// associations are distinct types with distinct
/// [`RuleIdentity`](crate::RuleIdentity)s, and converting between them is an
/// explicit relabel. Only the numeric ids can agree: a
/// [`PackedProductCodec`](crate::PackedProductCodec) layout is
/// association-independent by construction (see its own documentation), and
/// that numeric equality deliberately does not authorize mixing tensors across
/// the two providers.
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

/// `⊠` for providers: gives every [`FusionRule`] a `.product(other)` method.
///
/// Blanket-implemented, so it applies to a provider defined outside this
/// workspace exactly as it applies to a built-in one. Import the trait to use
/// the method (`use tenet::core::ProductFusionRuleExt;`) — without it in
/// scope, `.product(…)` resolves to [`Iterator::product`] and the error
/// message is about iterators.
pub trait ProductFusionRuleExt: FusionRule + Sized {
    /// The ordered product `self ⊠ right`, with the default
    /// [`TensorKitProductCodec`](crate::TensorKitProductCodec).
    ///
    /// This is the canonical way to obtain a product symmetry. Labels are
    /// built the same way with
    /// [`product_sector`](crate::product_sector), and the two nest in step:
    ///
    /// ```
    /// use tenet_sectors::{
    ///     product_sector, FermionParityFusionRule, FusionRule, ProductFusionRuleExt,
    ///     SU2FusionRule, SU2Irrep, SectorCodec, U1FusionRule, U1Irrep, Z2Irrep,
    /// };
    ///
    /// // Two factors: fZ2 ⊠ U(1). The `SectorCodec::` prefix disambiguates
    /// // against `ProductFusionRule`'s inherent id-level `encode_sector` /
    /// // `decode_sector`, which speak component `SectorId`s rather than
    /// // labels; callers of `tenet::typed` never write either by hand.
    /// let rule = FermionParityFusionRule.product(U1FusionRule);
    /// let odd = product_sector(Z2Irrep::ODD, U1Irrep::new(1));
    /// let id = SectorCodec::encode_sector(&rule, &odd)?;
    /// assert_eq!(SectorCodec::decode_sector(&rule, id)?, odd);
    ///
    /// // Three factors: the same call again, left-associated as
    /// // (fZ2 ⊠ U(1)) ⊠ SU(2). Nothing new is needed at the core.
    /// let rule = FermionParityFusionRule
    ///     .product(U1FusionRule)
    ///     .product(SU2FusionRule);
    /// let label = product_sector(odd, SU2Irrep::from_twice_spin(1));
    /// let id = SectorCodec::encode_sector(&rule, &label)?;
    /// assert_eq!(SectorCodec::decode_sector(&rule, id)?, label);
    ///
    /// // Factor order and association are structure, not an equivalence:
    /// // U(1) ⊠ fZ2 is a different Rust type with a different identity, and
    /// // converting between the two is an explicit component swap.
    /// let swapped = U1FusionRule.product(FermionParityFusionRule);
    /// assert_ne!(
    ///     FermionParityFusionRule.product(U1FusionRule).rule_identity(),
    ///     swapped.rule_identity(),
    /// );
    /// # Ok::<(), tenet_sectors::FusionAlgebraError>(())
    /// ```
    ///
    /// The `tenet::typed` module documentation carries the same product
    /// driving a `GradedSpace` and a `TensorMap`; that layer is where a
    /// product symmetry is actually used. It is not linked from here because
    /// `tenet-sectors` does not depend on `tenet` — the dependency runs the
    /// other way.
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

/// The product label is the component label pair; the id is whatever `Codec`
/// packs those component ids into.
///
/// Note the name clash with the inherent id-level
/// [`ProductFusionRule::encode_sector`] / [`ProductFusionRule::decode_sector`]
/// pair: on a concrete `ProductFusionRule` value the inherent methods win, so
/// call these through the trait (`SectorCodec::decode_sector(&rule, id)`).
/// Generic `R: SectorCodec` code — the facade's only caller — is unaffected.
impl<LeftRule, RightRule, Codec> SectorCodec for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: SectorCodec,
    RightRule: SectorCodec,
    Codec: ProductSectorCodec + 'static,
{
    type Sector = ProductSector<LeftRule::Sector, RightRule::Sector>;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        let left = self.left.encode_sector(value.left())?;
        let right = self.right.encode_sector(value.right())?;
        // Why not map this to `UnrepresentableSectorLabel`: the packed codec
        // already names the offending component and its bit budget, which a
        // label string would only blur.
        Ok(Codec::encode_checked(left, right)?)
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        let (left, right) = Codec::decode_checked(id)?;
        Ok(ProductSector::new(
            self.left.decode_sector(left)?,
            self.right.decode_sector(right)?,
        ))
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

impl<LeftRule, RightRule, Codec> CanonicalUnitFusionRule
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: CanonicalUnitFusionRule,
    RightRule: CanonicalUnitFusionRule,
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
    use crate::{
        CanonicalUnitFusionRule, FermionParityFusionRule, MultiplicityFreeFusionSymbols,
        U1FusionRule, U1Irrep, Z2Irrep,
    };

    #[test]
    fn product_of_canonical_unit_rules_has_canonical_unit() {
        // What: product construction preserves the unit law componentwise.
        fn accepts_canonical_unit<R: CanonicalUnitFusionRule>(_rule: &R) {}

        let rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
        let vacuum = rule.vacuum();
        let sector = rule.encode_sector(Z2Irrep::ODD.into(), U1Irrep::new(-17).into());
        let fused = rule.fusion_channels(sector, sector)[0];
        accepts_canonical_unit(&rule);
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
