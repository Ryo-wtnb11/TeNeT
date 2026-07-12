use tenet_core::{
    BraidingStyleKind, CoreError, FusionRule, FusionStyleKind, FusionTensorMapSpace,
    FusionTreeHomSpace, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, RuleIdentity, SectorId, SectorVec, TensorMap, TensorMapSpace,
    U1FusionRule, Z2FusionRule,
};
use tenet_tensors::{adjoint, TensorTraceAxisSpec, TensorTraceFusionStructure};
use tenet_tensors::{DynamicFusionMapSpace, OperationError, TreeTransformOperation};

#[test]
fn space_built_for_z2_rejects_u1_operation_with_same_integer_sector() {
    let space = DynamicFusionMapSpace::from_degeneracy_shapes(
        &Z2FusionRule,
        FusionTreeHomSpace::from_sector_ids([(0, 1)], []),
        [vec![1]],
    )
    .unwrap();

    let error = space
        .transformed(&U1FusionRule, &TreeTransformOperation::permute([0], []))
        .unwrap_err();

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[derive(Clone)]
struct SymbolOnlyRule {
    identity: RuleIdentity,
    symbol: f64,
}

impl SymbolOnlyRule {
    fn new(symbol: f64) -> Self {
        Self {
            identity: RuleIdentity::new_unique::<Self>(),
            symbol,
        }
    }
}

impl FusionRule for SymbolOnlyRule {
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
        SectorId::new(0)
    }
    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        [SectorId::new(0)].into_iter().collect()
    }
}

impl MultiplicityFreeFusionRule for SymbolOnlyRule {}

impl MultiplicityFreeFusionSymbols for SymbolOnlyRule {
    type Scalar = f64;
    fn scalar_one(&self) -> f64 {
        1.0
    }
    fn scalar_conj(&self, value: f64) -> f64 {
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
    ) -> f64 {
        self.symbol
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        self.symbol
    }
}

impl MultiplicityFreeRigidSymbols for SymbolOnlyRule {
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

#[test]
fn space_rejects_same_type_rule_when_only_symbols_differ() {
    let first = SymbolOnlyRule::new(1.0);
    let second = SymbolOnlyRule::new(-1.0);
    let space = DynamicFusionMapSpace::from_degeneracy_shapes(
        &first,
        FusionTreeHomSpace::from_sector_ids([(0, 1)], []),
        [vec![1]],
    )
    .unwrap();

    let error = space
        .transformed(&second, &TreeTransformOperation::permute([0], []))
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn dynamic_trace_rejects_same_type_rule_when_only_symbols_differ() {
    let first = SymbolOnlyRule::new(1.0);
    let second = SymbolOnlyRule::new(-1.0);
    let space = DynamicFusionMapSpace::from_degeneracy_shapes(
        &first,
        FusionTreeHomSpace::from_sector_ids([(0, 1)], []),
        [vec![1]],
    )
    .unwrap();

    let error = TensorTraceFusionStructure::<f64>::compile_fusion_dyn(
        &second,
        &space,
        &space,
        TensorTraceAxisSpec::new(&[0], &[], &[]),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn typed_eager_adjoint_rejects_same_type_rule_when_only_symbols_differ() {
    let first = SymbolOnlyRule::new(1.0);
    let second = SymbolOnlyRule::new(-1.0);
    let space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 0>::from_dims([1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([(0, 1)], []),
        &first,
        [vec![1]],
    )
    .unwrap();
    let tensor = TensorMap::from_vec_with_fusion_space(vec![1.0], space).unwrap();

    let error = adjoint(&second, &tensor).unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}
