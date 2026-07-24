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
        let left_channels = self
            .left
            .try_fusion_channels(left_left, right_left)?;
        let right_channels = self
            .right
            .try_fusion_channels(left_right, right_right)?;
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

impl<LeftRule, RightRule, LeftLayout, RightLayout>
    lowered_multiplicity_free_sealed::Sealed
    for ProductFusionRule<
        LeftRule,
        RightRule,
        PackedProductCodec<LeftLayout, RightLayout>,
    >
where
    LeftRule: LoweredMultiplicityFreeAlgebra,
    RightRule: LoweredMultiplicityFreeAlgebra,
    LeftLayout: PackedSectorLayout + 'static,
    RightLayout: PackedSectorLayout + 'static,
{
}

impl<LeftRule, RightRule, LeftLayout, RightLayout> LoweredMultiplicityFreeAlgebra
    for ProductFusionRule<
        LeftRule,
        RightRule,
        PackedProductCodec<LeftLayout, RightLayout>,
    >
where
    LeftRule: LoweredMultiplicityFreeAlgebra,
    RightRule: LoweredMultiplicityFreeAlgebra,
    LeftLayout: PackedSectorLayout + 'static,
    RightLayout: PackedSectorLayout + 'static,
{
    type Sector = ProductSector<LeftRule::Sector, RightRule::Sector>;

    fn try_decode_lowered(
        &self,
        sector: SectorId,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        let (left, right) =
            PackedProductCodec::<LeftLayout, RightLayout>::decode_checked(sector)
                .map_err(LoweredFusionTreeBuildError::codec)?;
        Ok(ProductSector::new(
            self.left.try_decode_lowered(left)?,
            self.right.try_decode_lowered(right)?,
        ))
    }

    fn try_encode_lowered(
        &self,
        sector: Self::Sector,
    ) -> Result<SectorId, LoweredFusionTreeBuildError> {
        let left = self.left.try_encode_lowered(*sector.left())?;
        let right = self.right.try_encode_lowered(*sector.right())?;
        PackedProductCodec::<LeftLayout, RightLayout>::encode_checked(left, right)
            .map_err(LoweredFusionTreeBuildError::codec)
    }

    fn try_lowered_vacuum(&self) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(ProductSector::new(
            self.left.try_lowered_vacuum()?,
            self.right.try_lowered_vacuum()?,
        ))
    }

    fn try_lowered_dual(
        &self,
        sector: Self::Sector,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(ProductSector::new(
            self.left.try_lowered_dual(*sector.left())?,
            self.right.try_lowered_dual(*sector.right())?,
        ))
    }

    fn try_for_each_lowered_channel<F>(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        emit: &mut F,
    ) -> Result<(), LoweredFusionTreeBuildError>
    where
        F: FnMut(Self::Sector) -> Result<(), LoweredFusionTreeBuildError>,
    {
        self.right.try_for_each_lowered_channel(
            *left.right(),
            *right.right(),
            &mut |right_channel| {
                self.left.try_for_each_lowered_channel(
                    *left.left(),
                    *right.left(),
                    &mut |left_channel| {
                        emit(ProductSector::new(left_channel, right_channel))
                    },
                )
            },
        )
    }

    fn try_lowered_nsymbol(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        coupled: Self::Sector,
    ) -> Result<usize, LoweredFusionTreeBuildError> {
        let left_n = self.left.try_lowered_nsymbol(
            *left.left(),
            *right.left(),
            *coupled.left(),
        )?;
        let right_n = self.right.try_lowered_nsymbol(
            *left.right(),
            *right.right(),
            *coupled.right(),
        )?;
        match left_n.checked_mul(right_n) {
            Some(multiplicity) => Ok(multiplicity),
            None => {
                // Why not encode every successful call: persistent IDs are
                // needed only to diagnose the exceptional overflow branch.
                let left = self.try_encode_lowered(left)?;
                let right = self.try_encode_lowered(right)?;
                let coupled = self.try_encode_lowered(coupled)?;
                Err(LoweredFusionTreeBuildError::fusion_algebra(
                    FusionAlgebraError::MultiplicityOverflow {
                        left,
                        right,
                        coupled,
                    },
                ))
            }
        }
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
        self.left.has_trivial_associator_gauge()
            && self.right.has_trivial_associator_gauge()
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SU2Irrep {
    twice_spin: usize,
}

/// Why not admit larger labels: the exact authority's canonical Regge key
/// reserves one `u8` value when sizing its complete key domain.
pub const SU2_MAX_DOUBLED_SPIN: usize = (u8::MAX - 1) as usize;

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

#[doc(hidden)]
pub const SU2_EXACT_AUTHORITY_VERSION: u8 = 1;

const SU2_EXACT_AUTHORITY_IDENTITY_SCHEMA: u64 = 0x5355_3245_5841_4354;

impl FusionRule for SU2FusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        static IDENTITY: OnceLock<RuleIdentity> = OnceLock::new();
        IDENTITY
            .get_or_init(|| {
                RuleIdentity::from_canonical_bytes::<SU2FusionRule>(
                    SU2_EXACT_AUTHORITY_IDENTITY_SCHEMA,
                    Arc::<[u8]>::from([SU2_EXACT_AUTHORITY_VERSION]),
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
        let left = SU2Irrep::from_sector_id(left).twice_spin();
        let right = SU2Irrep::from_sector_id(right).twice_spin();
        let min = left.abs_diff(right);
        let max = left + right;
        // Why not return only the representable channels: truncating a fusion
        // closure would define a different category while appearing to be SU(2).
        assert!(
            max <= SU2_MAX_DOUBLED_SPIN,
            "SU(2) fusion closure exceeds the supported maximum doubled spin 254"
        );
        (min..=max)
            .step_by(2)
            .map(|twice_spin| SU2Irrep::from_twice_spin(twice_spin).into())
            .collect()
    }
}

#[cfg(test)]
std::thread_local! {
    static CHECKED_SU2_ID_VALIDATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static LOWERED_SU2_ID_ENCODINGS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reset_su2_id_boundary_observations() {
    CHECKED_SU2_ID_VALIDATIONS.with(|count| count.set(0));
    LOWERED_SU2_ID_ENCODINGS.with(|count| count.set(0));
}

#[cfg(test)]
fn su2_id_boundary_observations() -> (usize, usize) {
    (
        CHECKED_SU2_ID_VALIDATIONS.with(std::cell::Cell::get),
        LOWERED_SU2_ID_ENCODINGS.with(std::cell::Cell::get),
    )
}

fn checked_su2_irrep(sector: SectorId) -> Result<SU2Irrep, FusionAlgebraError> {
    #[cfg(test)]
    CHECKED_SU2_ID_VALIDATIONS.with(|count| count.set(count.get() + 1));
    SU2Irrep::try_from_sector_id(sector).ok_or(FusionAlgebraError::InvalidSector { sector })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Su2FusionClosureError;

fn su2_channel_bounds(
    left: SU2Irrep,
    right: SU2Irrep,
) -> Result<(usize, usize), Su2FusionClosureError> {
    let left_spin = left.twice_spin();
    let right_spin = right.twice_spin();
    let max = left_spin + right_spin;
    if max > SU2_MAX_DOUBLED_SPIN {
        return Err(Su2FusionClosureError);
    }
    Ok((left_spin.abs_diff(right_spin), max))
}

fn checked_su2_channels(
    left: SectorId,
    right: SectorId,
) -> Result<(usize, usize), FusionAlgebraError> {
    let left_irrep = checked_su2_irrep(left)?;
    let right_irrep = checked_su2_irrep(right)?;
    su2_channel_bounds(left_irrep, right_irrep).map_err(|_| {
        FusionAlgebraError::FusionNotRepresentable {
            left,
            right,
        }
    })
}

impl CheckedFusionAlgebra for SU2FusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        checked_su2_irrep(sector)?;
        Ok(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let (min, max) = checked_su2_channels(left, right)?;
        Ok((min..=max)
            .step_by(2)
            .map(|twice_spin| SU2Irrep::from_twice_spin(twice_spin).into())
            .collect())
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let coupled = checked_su2_irrep(coupled)?.twice_spin();
        let (min, max) = checked_su2_channels(left, right)?;
        Ok(usize::from(
            coupled >= min && coupled <= max && (coupled - min) % 2 == 0,
        ))
    }
}

impl MultiplicityFreeFusionRule for SU2FusionRule {}

impl lowered_multiplicity_free_sealed::Sealed for SU2FusionRule {}

impl LoweredMultiplicityFreeAlgebra for SU2FusionRule {
    type Sector = SU2Irrep;

    fn try_decode_lowered(
        &self,
        sector: SectorId,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        SU2Irrep::try_from_sector_id(sector)
            .ok_or_else(|| LoweredFusionTreeBuildError::invalid_sector(sector))
    }

    fn try_encode_lowered(
        &self,
        sector: Self::Sector,
    ) -> Result<SectorId, LoweredFusionTreeBuildError> {
        #[cfg(test)]
        LOWERED_SU2_ID_ENCODINGS.with(|count| count.set(count.get() + 1));
        Ok(sector.into())
    }

    fn try_lowered_vacuum(&self) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(SU2Irrep::from_twice_spin(0))
    }

    fn try_lowered_dual(
        &self,
        sector: Self::Sector,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(sector)
    }

    fn try_for_each_lowered_channel<F>(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        emit: &mut F,
    ) -> Result<(), LoweredFusionTreeBuildError>
    where
        F: FnMut(Self::Sector) -> Result<(), LoweredFusionTreeBuildError>,
    {
        let (min, max) = su2_channel_bounds(left, right).map_err(|_| {
            LoweredFusionTreeBuildError::fusion_algebra(
                FusionAlgebraError::FusionNotRepresentable {
                    left: left.sector_id(),
                    right: right.sector_id(),
                },
            )
        })?;
        for twice_spin in (min..=max).step_by(2) {
            emit(SU2Irrep::from_twice_spin(twice_spin))?;
        }
        Ok(())
    }

    fn try_lowered_nsymbol(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        coupled: Self::Sector,
    ) -> Result<usize, LoweredFusionTreeBuildError> {
        let (min, max) = su2_channel_bounds(left, right).map_err(|_| {
            LoweredFusionTreeBuildError::fusion_algebra(
                FusionAlgebraError::FusionNotRepresentable {
                    left: left.sector_id(),
                    right: right.sector_id(),
                },
            )
        })?;
        let coupled = coupled.twice_spin();
        Ok(usize::from(coupled >= min && coupled <= max && (coupled - min) % 2 == 0))
    }
}

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
        let j1 = SU2Irrep::from_sector_id(left).twice_spin();
        let j2 = SU2Irrep::from_sector_id(middle).twice_spin();
        let j3 = SU2Irrep::from_sector_id(right).twice_spin();
        let j4 = SU2Irrep::from_sector_id(coupled).twice_spin();
        let j5 = SU2Irrep::from_sector_id(left_coupled).twice_spin();
        let j6 = SU2Irrep::from_sector_id(right_coupled).twice_spin();
        su2_f_symbol_from_doubled_spins(j1, j2, j3, j4, j5, j6)
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        if self.nsymbol(left, right, coupled) == 0 {
            return 0.0;
        }
        let left = SU2Irrep::from_sector_id(left).twice_spin();
        let right = SU2Irrep::from_sector_id(right).twice_spin();
        let coupled = SU2Irrep::from_sector_id(coupled).twice_spin();
        let exponent = (left + right - coupled) / 2;
        if exponent % 2 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

impl MultiplicityFreeRigidSymbols for SU2FusionRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        (SU2Irrep::from_sector_id(sector).twice_spin() + 1) as f64
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
        if SU2Irrep::from_sector_id(sector).twice_spin() % 2 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

fn su2_f_symbol_from_doubled_spins(
    j1: usize,
    j2: usize,
    j3: usize,
    j4: usize,
    j5: usize,
    j6: usize,
) -> f64 {
    crate::su2_exact::validate_supported_spins([j1, j2, j3, j4, j5, j6]);
    if [j1, j2, j3, j4, j5, j6].iter().all(|&j| j == 0) {
        return 1.0;
    }
    let phase_exponent = (j1 + j2 + j3 + j4) / 2;
    let phase = if phase_exponent % 2 == 0 { 1.0 } else { -1.0 };
    let dimension_factor = ((j5 + 1) as f64 * (j6 + 1) as f64).sqrt();
    phase
        * dimension_factor
        * crate::su2_exact::wigner_6j_doubled([j1, j2, j5, j3, j4, j6])
}

// FibonacciAnyon: the simplest genuinely non-abelian anyon model (Simple
// fusion + Anyonic braiding + complex F/R symbols) — SectorId 0 = vacuum
// `𝟙`, SectorId 1 = `τ`, with `τ⊗τ = 𝟙 ⊕ τ`. All numeric F/R/dim/twist
