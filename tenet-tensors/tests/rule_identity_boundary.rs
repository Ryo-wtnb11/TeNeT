use tenet_core::{
    BraidingStyleKind, CoreError, FusionRule, FusionStyleKind, FusionTensorMapSpace,
    FusionTreeHomSpace, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, RuleIdentity, SectorId, SectorVec, TensorMap, TensorMapSpace,
    U1FusionRule, Z2FusionRule,
};
use tenet_tensors::{
    adjoint, tensorcontract_fusion_block_specs, tensorcontract_fusion_into,
    TensorContractFusionExecutionContext, TensorContractSpec, TensorTraceAxisSpec,
    TensorTraceFusionStructure, TreeTransformBuiltinRuleCacheKey,
};
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

fn bound_scalar_space<R>(rule: &R) -> FusionTensorMapSpace<0, 0>
where
    R: MultiplicityFreeFusionRule,
{
    FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        rule,
        [vec![]],
    )
    .unwrap()
}

fn unbound_scalar_space<R>(rule: &R) -> FusionTensorMapSpace<0, 0>
where
    R: MultiplicityFreeFusionRule,
{
    let bound = bound_scalar_space(rule);
    unbound_scalar_space_like(&bound)
}

fn unbound_scalar_space_like(bound: &FusionTensorMapSpace<0, 0>) -> FusionTensorMapSpace<0, 0> {
    FusionTensorMapSpace::from_shared_subblock_structure(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        bound.homspace().clone(),
        bound.subblock_structure().clone(),
    )
    .unwrap()
}

fn scalar_contract_error<R>(
    rule: &R,
    dst_space: FusionTensorMapSpace<0, 0>,
    lhs_space: FusionTensorMapSpace<0, 0>,
    rhs_space: FusionTensorMapSpace<0, 0>,
    axes: TensorContractSpec<'_>,
) -> OperationError
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let mut dst = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![0.0], dst_space).unwrap();
    let lhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![2.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![3.0], rhs_space).unwrap();
    tensorcontract_fusion_into(rule, &mut dst, &lhs, &rhs, axes, 1.0, 0.0).unwrap_err()
}

fn assert_missing_rule_identity(error: OperationError) {
    assert!(matches!(
        error,
        OperationError::Core(CoreError::MissingFusionRuleIdentity)
    ));
}

fn assert_rule_mismatch(error: OperationError) {
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn fusion_contract_one_shot_rejects_each_unbound_operand_on_direct_route() {
    let rule = Z2FusionRule;
    let bound = || bound_scalar_space(&rule);
    let unbound = || unbound_scalar_space(&rule);
    let axes = TensorContractSpec::with_default_output_order(&[], &[]);

    assert_missing_rule_identity(scalar_contract_error(
        &rule,
        unbound(),
        bound(),
        bound(),
        axes,
    ));
    assert_missing_rule_identity(scalar_contract_error(
        &rule,
        bound(),
        unbound(),
        bound(),
        axes,
    ));
    assert_missing_rule_identity(scalar_contract_error(
        &rule,
        bound(),
        bound(),
        unbound(),
        axes,
    ));
}

#[test]
fn fusion_contract_block_specs_rejects_each_unbound_operand() {
    let rule = Z2FusionRule;
    let bound = || bound_scalar_space(&rule);
    let unbound = || unbound_scalar_space(&rule);
    let axes = TensorContractSpec::with_default_output_order(&[], &[]);

    assert_missing_rule_identity(
        tensorcontract_fusion_block_specs(&rule, &unbound(), &bound(), &bound(), axes).unwrap_err(),
    );
    assert_missing_rule_identity(
        tensorcontract_fusion_block_specs(&rule, &bound(), &unbound(), &bound(), axes).unwrap_err(),
    );
    assert_missing_rule_identity(
        tensorcontract_fusion_block_specs(&rule, &bound(), &bound(), &unbound(), axes).unwrap_err(),
    );
}

#[test]
fn fusion_contract_fallback_rejects_unbound_operand_before_conjugation_lowering() {
    let rule = Z2FusionRule;
    let error = scalar_contract_error(
        &rule,
        bound_scalar_space(&rule),
        unbound_scalar_space(&rule),
        bound_scalar_space(&rule),
        TensorContractSpec::with_default_output_order_and_conjugation(&[], &[], true, false),
    );

    assert_missing_rule_identity(error);
}

#[test]
fn fusion_contract_rejects_z2_spaces_with_u1_despite_overlapping_sector_ids() {
    let z2 = Z2FusionRule;
    let error = scalar_contract_error(
        &U1FusionRule,
        bound_scalar_space(&z2),
        bound_scalar_space(&z2),
        bound_scalar_space(&z2),
        TensorContractSpec::with_default_output_order(&[], &[]),
    );

    assert_rule_mismatch(error);
}

#[test]
fn fusion_contract_rejects_same_type_rule_with_distinct_identity() {
    let first = SymbolOnlyRule::new(1.0);
    let second = SymbolOnlyRule::new(-1.0);
    let error = scalar_contract_error(
        &second,
        bound_scalar_space(&first),
        bound_scalar_space(&first),
        bound_scalar_space(&first),
        TensorContractSpec::with_default_output_order(&[], &[]),
    );

    assert_rule_mismatch(error);
}

#[test]
fn fusion_contract_bound_same_rule_matches_on_direct_and_fallback_routes() {
    let rule = Z2FusionRule;
    for axes in [
        TensorContractSpec::with_default_output_order(&[], &[]),
        TensorContractSpec::with_default_output_order_and_conjugation(&[], &[], true, false),
    ] {
        let mut dst = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(
            vec![7.0],
            bound_scalar_space(&rule),
        )
        .unwrap();
        let lhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(
            vec![2.0],
            bound_scalar_space(&rule),
        )
        .unwrap();
        let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(
            vec![5.0],
            bound_scalar_space(&rule),
        )
        .unwrap();

        tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, 2.0, 3.0).unwrap();

        assert_eq!(dst.data(), &[41.0]);
    }
}

#[test]
fn fusion_contract_context_rejects_unbound_space_before_warm_resolution_cache_hit() {
    let rule = Z2FusionRule;
    let bound = bound_scalar_space(&rule);
    let axes = TensorContractSpec::with_default_output_order(&[], &[]);
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    let mut dst =
        TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![0.0], bound.clone()).unwrap();
    let lhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![2.0], bound.clone()).unwrap();
    let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![3.0], bound.clone()).unwrap();
    context
        .tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, 1.0, 0.0)
        .unwrap();
    let hits = context.contraction_resolution_cache_hits();
    let fast_hits = context.contraction_resolution_cache_fast_hits();
    let misses = context.contraction_resolution_cache_misses();
    let mut unbound_dst = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(
        vec![0.0],
        unbound_scalar_space_like(&bound),
    )
    .unwrap();

    let error = context
        .tensorcontract_fusion_into(&rule, &mut unbound_dst, &lhs, &rhs, axes, 1.0, 0.0)
        .unwrap_err();

    assert_missing_rule_identity(error);
    assert_eq!(context.contraction_resolution_cache_hits(), hits);
    assert_eq!(context.contraction_resolution_cache_fast_hits(), fast_hits);
    assert_eq!(context.contraction_resolution_cache_misses(), misses);
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
