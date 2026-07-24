use tenet_sectors::{
    BraidingStyleKind, CheckedFusionAlgebra, FermionParityFusionRule, FusionAlgebraError,
    FusionRule, FusionStyleKind, GenericFArray, GenericFusionSymbols, GenericRMatrix,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    RuleIdentity, SectorId, SectorVec, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
};

#[derive(Clone, Copy)]
struct CheckedMultiplicityFreeRule;

impl FusionRule for CheckedMultiplicityFreeRule {
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
        SectorId::new(0)
    }

    fn fusion_channels(&self, _left: SectorId, _right: SectorId) -> SectorVec {
        let mut channels = SectorVec::new();
        channels.push(SectorId::new(0));
        channels
    }
}

impl CheckedFusionAlgebra for CheckedMultiplicityFreeRule {
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

impl MultiplicityFreeFusionRule for CheckedMultiplicityFreeRule {}

impl MultiplicityFreeFusionSymbols for CheckedMultiplicityFreeRule {
    type Scalar = f64;

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

#[derive(Clone, Copy)]
struct GenericRule;

impl FusionRule for GenericRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, _left: SectorId, _right: SectorId) -> SectorVec {
        let mut channels = SectorVec::new();
        channels.push(SectorId::new(0));
        channels
    }
}

impl GenericFusionSymbols for GenericRule {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
        _d: SectorId,
        _e: SectorId,
        _f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        GenericFArray::new(vec![1.0], (1, 1, 1, 1))
    }

    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        GenericRMatrix::new(vec![1.0], 1, 1)
    }
}

#[test]
fn portable_contracts_are_implementable_without_tenet_core() {
    // What: downstream sector providers can implement the checked
    // multiplicity-free and Generic contracts without importing the engine.
    let multiplicity_free = CheckedMultiplicityFreeRule;
    assert_eq!(
        multiplicity_free
            .try_fusion_channels(SectorId::new(0), SectorId::new(0))
            .unwrap(),
        [SectorId::new(0)].into()
    );
    assert_eq!(
        multiplicity_free.f_symbol_scalar(
            SectorId::new(0),
            SectorId::new(0),
            SectorId::new(0),
            SectorId::new(0),
            SectorId::new(0),
            SectorId::new(0),
        ),
        1.0
    );

    let generic = GenericRule;
    assert_eq!(
        generic
            .f_symbol_generic(
                SectorId::new(0),
                SectorId::new(0),
                SectorId::new(0),
                SectorId::new(0),
                SectorId::new(0),
                SectorId::new(0),
            )
            .data(),
        [1.0]
    );
    assert_eq!(
        generic
            .r_symbol_generic(SectorId::new(0), SectorId::new(0), SectorId::new(0))
            .data(),
        [1.0]
    );
}

#[test]
fn u1_irrep_checked_arithmetic_preserves_normal_and_overflow_results() {
    // What: portable U(1) component arithmetic has the exact checked algebra
    // results used by both public provider calls and core's lowered hot path.
    assert_eq!(U1Irrep::new(7).checked_dual(), Ok(U1Irrep::new(-7)));
    assert_eq!(
        U1Irrep::new(7).checked_fuse(U1Irrep::new(-3)),
        Ok(U1Irrep::new(4))
    );
    assert_eq!(
        U1Irrep::new(i32::MIN).checked_dual(),
        Err(FusionAlgebraError::U1DualOverflow { charge: i32::MIN })
    );
    assert_eq!(
        U1Irrep::new(i32::MAX).checked_fuse(U1Irrep::new(1)),
        Err(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        })
    );
}

#[test]
fn builtin_abelian_providers_keep_their_portable_symbol_contracts() {
    // What: the provider types now owned by tenet-sectors retain their
    // representative fusion, dual, F/R, and rigid-symbol behavior without
    // importing the fusion-tree engine.
    let even = Z2Irrep::EVEN.sector_id();
    let odd = Z2Irrep::ODD.sector_id();
    let z2 = Z2FusionRule;
    assert_eq!(z2.dual(odd), odd);
    assert_eq!(z2.fusion_channels(odd, odd), [even].into());
    assert_eq!(z2.f_symbol_scalar(odd, odd, odd, odd, even, even), 1.0);
    assert_eq!(z2.r_symbol_scalar(odd, odd, even), 1.0);
    assert_eq!(z2.dim_scalar(odd), 1.0);
    assert_eq!(z2.twist_scalar(odd), 1.0);
    assert_eq!(z2.frobenius_schur_phase_scalar(odd), 1.0);

    let fermion = FermionParityFusionRule;
    assert_eq!(fermion.dual(odd), odd);
    assert_eq!(fermion.fusion_channels(odd, odd), [even].into());
    assert_eq!(fermion.f_symbol_scalar(odd, odd, odd, odd, even, even), 1.0);
    assert_eq!(fermion.r_symbol_scalar(odd, odd, even), -1.0);
    assert_eq!(fermion.twist_scalar(odd), -1.0);
    assert_eq!(fermion.frobenius_schur_phase_scalar(odd), 1.0);
    assert_eq!(fermion.dim_scalar(odd), 1.0);
    assert_eq!(fermion.a_symbol_scalar(odd, odd, even), 1.0);
    assert_eq!(fermion.b_symbol_scalar(odd, odd, even), 1.0);

    let minus_three = U1Irrep::new(-3).sector_id();
    let four = U1Irrep::new(4).sector_id();
    let one = U1Irrep::new(1).sector_id();
    let two = U1Irrep::new(2).sector_id();
    let five = U1Irrep::new(5).sector_id();
    let u1 = U1FusionRule;
    assert_eq!(u1.dual(minus_three), U1Irrep::new(3).sector_id());
    assert_eq!(u1.fusion_channels(minus_three, four), [one].into());
    assert_eq!(
        u1.f_symbol_scalar(minus_three, four, one, two, one, five),
        1.0
    );
    assert_eq!(u1.r_symbol_scalar(minus_three, four, one), 1.0);
    assert_eq!(u1.dim_scalar(minus_three), 1.0);
    assert_eq!(u1.twist_scalar(minus_three), 1.0);
    assert_eq!(u1.frobenius_schur_phase_scalar(minus_three), 1.0);
}
