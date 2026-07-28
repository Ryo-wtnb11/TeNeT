//! External-crate executable contract for the curated expert facades.
use std::sync::Arc;

use tenet::core::{
    FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, RuleIdentity, SectorId,
    SectorLeg, TensorMap, TensorMapSpace, U1FusionRule,
};
use tenet::dense::DefaultDenseExecutor;
use tenet::matrixalgebra::{
    svd_compact, BoundTensorMap, BoundTensorMapRef, SectorSpectrum, SvdCompact,
};
use tenet::operations::{
    braid_into, permute_into, tensoradd_into, tensorcontract_fusion_into, tensorcontract_into,
    tensortrace_into, transpose_into, BoundDynamicFusionMapSpace, DynamicFusionMapSpace,
    OperationError, OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
    TensorTraceAxisSpec, TreeTransformExecutionContext, TreeTransformOperation,
};

fn scalar_space() -> FusionTensorMapSpace<0, 0> {
    FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        &U1FusionRule,
        [vec![]],
    )
    .unwrap()
}

fn map_space() -> FusionTensorMapSpace<1, 1> {
    let leg = SectorLeg::new([(SectorId::new(0), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::from_dims([1], [1]).unwrap(),
        hom.clone(),
        &U1FusionRule,
        vec![vec![1, 1]; hom.fusion_tree_keys(&U1FusionRule).len()],
    )
    .unwrap()
}

fn plain_scalar() -> TensorMap<f64, 0, 0> {
    TensorMap::from_vec(vec![2.0], TensorMapSpace::from_dims([], []).unwrap()).unwrap()
}

fn fusion_scalar() -> TensorMap<f64, 0, 0> {
    TensorMap::from_vec_with_fusion_space(vec![2.0], scalar_space()).unwrap()
}

fn dynamic_scalar() -> BoundDynamicFusionMapSpace<U1FusionRule> {
    BoundDynamicFusionMapSpace::from_degeneracy_shapes(
        Arc::new(U1FusionRule),
        FusionTreeHomSpace::from_sector_ids([], []),
        [vec![]],
    )
    .unwrap()
}

#[test]
fn curated_expert_facades_execute_externally() {
    let rule = U1FusionRule;
    let source: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![2.0], map_space()).unwrap();
    let plain_source: TensorMap<f64, 1, 1> =
        TensorMap::from_vec(vec![2.0], TensorMapSpace::from_dims([1], [1]).unwrap()).unwrap();

    let mut sum = plain_scalar();
    tensoradd_into(
        &mut sum,
        &plain_scalar(),
        OutputAxisOrder::from_axes(&[]),
        1.0,
        0.0,
    )
    .unwrap();
    let mut contract = plain_scalar();
    let contract_spec = TensorContractSpec::with_default_output_order(&[], &[]);
    let contract_result: Result<(), OperationError> = tensorcontract_into(
        &mut contract,
        &plain_scalar(),
        &plain_scalar(),
        contract_spec,
        1.0,
        0.0,
    );
    contract_result.unwrap();

    let mut fused = fusion_scalar();
    tensorcontract_fusion_into(
        &rule,
        &mut fused,
        &fusion_scalar(),
        &fusion_scalar(),
        TensorContractSpec::with_default_output_order(&[], &[]),
        1.0,
        0.0,
    )
    .unwrap();
    let mut fusion_context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    fusion_context
        .tensorcontract_fusion_into(
            &rule,
            &mut fused,
            &fusion_scalar(),
            &fusion_scalar(),
            TensorContractSpec::with_default_output_order(&[], &[]),
            1.0,
            0.0,
        )
        .unwrap();

    let mut traced = plain_scalar();
    let trace_result: Result<(), OperationError> = tensortrace_into(
        &mut traced,
        &plain_source,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        1.0,
        0.0,
    );
    trace_result.unwrap();
    let mut dst: TensorMap<f64, 1, 1> =
        TensorMap::from_vec_with_fusion_space(vec![0.0], map_space()).unwrap();
    permute_into(&rule, [0], [1], &mut dst, &source, 1.0, 0.0).unwrap();
    braid_into(&rule, [0], [1], [0], [0], &mut dst, &source, 1.0, 0.0).unwrap();
    transpose_into(&rule, [0], [1], &mut dst, &source, 1.0, 0.0).unwrap();

    let dynamic: BoundDynamicFusionMapSpace<U1FusionRule> = dynamic_scalar();
    let _dynamic_space: &DynamicFusionMapSpace = dynamic.space();
    let mut dynamic_context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let mut dynamic_dst = vec![0.0];
    dynamic_context
        .tensorcontract_fusion_dyn_into(
            &dynamic,
            &mut dynamic_dst,
            &dynamic,
            &[2.0],
            &dynamic,
            &[3.0],
            TensorContractSpec::with_default_output_order(&[], &[]),
            1.0,
            0.0,
        )
        .unwrap();
    let mut tree_context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    tree_context
        .tree_transform_dyn_into(
            &rule,
            TreeTransformOperation::permute([], []),
            dynamic.space().structure(),
            dynamic.space().structure(),
            &mut dynamic_dst,
            &[6.0],
            1.0,
            0.0,
        )
        .unwrap();

    let bound: BoundTensorMap<U1FusionRule, f64, 1, 1> =
        BoundTensorMap::try_new(Arc::new(U1FusionRule), source).unwrap();
    let input: BoundTensorMapRef<'_, U1FusionRule, f64, 1, 1> = bound.as_ref();
    let compact: SvdCompact<U1FusionRule, f64, 1, 1> =
        svd_compact(&mut DefaultDenseExecutor::new(), &input).unwrap();
    let spectrum: &[SectorSpectrum] = &compact.singular_values;
    assert_eq!(
        spectrum
            .iter()
            .map(|sector| sector.values.len())
            .sum::<usize>(),
        1
    );
    assert_eq!(sum.data(), &[2.0]);
    assert_eq!(contract.data(), &[4.0]);
    assert_eq!(fused.data(), &[4.0]);
    assert_eq!(traced.data(), &[2.0]);
    assert_eq!(dynamic_dst, [6.0]);
}
