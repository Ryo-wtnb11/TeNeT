use tenet_sectors::{
    BraidingStyleKind, CheckedFusionAlgebra, FusionAlgebraError, FusionRule, FusionStyleKind,
    GenericFArray, GenericFusionSymbols, GenericRMatrix, MultiplicityFreeFusionRule,
    MultiplicityFreeFusionSymbols, RuleIdentity, SectorId, SectorVec,
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
