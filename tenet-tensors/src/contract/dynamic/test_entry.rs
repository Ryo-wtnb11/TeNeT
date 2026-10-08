use super::*;
use tenet_core::{CoreError, HostReadableStorage, HostWritableStorage, TensorMap};

// Non-profiled reference contraction path. Its only production caller
// (`tensorcontract_fusion_dynamic_into_context`) was removed as dead code;
// the profiled twin (`..._profiled`) is the live path via `context.rs`. Kept
// as the allocation/output reference the `run_host_reference` tests compare
// against, so it is test-only now.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_fusion_dynamic_plan_into_context<
    RuleKey,
    BT,
    BC,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    SDst,
    SLhs,
    SRhs,
    DDst,
    DLhs,
    DRhs,
>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    rule: &R,
    plan: &FusionContractPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    let dst_space = DynamicFusionMapSpace::from_typed(
        dst.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let lhs_space = DynamicFusionMapSpace::from_typed(
        lhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let rhs_space = DynamicFusionMapSpace::from_typed(
        rhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let dst_structure = std::sync::Arc::clone(dst.structure());
    let lhs_structure = std::sync::Arc::clone(lhs.structure());
    let rhs_structure = std::sync::Arc::clone(rhs.structure());
    tensorcontract_fusion_dynamic_plan_dyn_into_context(
        tree_context,
        contract_backend,
        contract_workspace,
        fusion_block_workspace,
        scratch,
        rule,
        encoded_layout_primer::<R>,
        plan,
        &dst_space,
        &dst_structure,
        dst.data_mut(),
        &lhs_space,
        &lhs_structure,
        lhs.data(),
        &rhs_space,
        &rhs_structure,
        rhs.data(),
        alpha,
        beta,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_dynamic_tree_execution_artifact_for_test<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
>(
    rule: &R,
    plan: &FusionContractPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let mut tree_context =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let mut contract_backend = DenseTreeTransformOperations::default();
    let mut contract_workspace = super::backend::TensorContractWorkspace::default();
    let mut fusion_block_workspace = FusionBlockContractWorkspace::default();
    let mut scratch = DynamicFusionScratchWorkspace::default();
    tensorcontract_fusion_dynamic_plan_into_context(
        &mut tree_context,
        &mut contract_backend,
        &mut contract_workspace,
        &mut fusion_block_workspace,
        &mut scratch,
        rule,
        plan,
        dst,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_dynamic_tree_execution_artifact_profile_pair_for_test<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
>(
    rule: &R,
    plan: &FusionContractPlan,
    ordinary_dst: &mut TensorMap<D, DST_NOUT, DST_NIN>,
    profiled_dst: &mut TensorMap<D, DST_NOUT, DST_NIN>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN>,
    alpha: D,
    beta: D,
) -> Result<TensorContractFusionProfile, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let mut tree_context =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let mut contract_backend = DenseTreeTransformOperations::default();
    let mut contract_workspace = super::backend::TensorContractWorkspace::default();
    let mut fusion_block_workspace = FusionBlockContractWorkspace::default();
    let dst_space = DynamicFusionMapSpace::from_typed(
        ordinary_dst
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let lhs_space = DynamicFusionMapSpace::from_typed(
        lhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let rhs_space = DynamicFusionMapSpace::from_typed(
        rhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let artifact = compile_dynamic_tree_execution_artifact::<_, _, _, _, _, false>(
        &mut tree_context,
        rule,
        encoded_layout_primer::<R>,
        plan,
        &dst_space,
        &lhs_space,
        lhs.structure(),
        &rhs_space,
        rhs.structure(),
        None,
    )?;
    let mut ordinary_scratch = DynamicFusionScratchWorkspace::default();
    let ordinary_dst_structure = Arc::clone(ordinary_dst.structure());
    execute_dynamic_tree_execution_artifact(
        &mut tree_context,
        &mut contract_backend,
        &mut contract_workspace,
        &mut fusion_block_workspace,
        &mut ordinary_scratch,
        &artifact,
        &ordinary_dst_structure,
        ordinary_dst.data_mut(),
        lhs.data(),
        rhs.data(),
        alpha,
        beta,
    )?;
    let mut profiled_scratch = DynamicFusionScratchWorkspace::default();
    let mut profile = TensorContractFusionProfile::default();
    let profiled_dst_structure = Arc::clone(profiled_dst.structure());
    execute_dynamic_tree_execution_artifact_profiled(
        &mut tree_context,
        &mut contract_backend,
        &mut contract_workspace,
        &mut fusion_block_workspace,
        &mut profiled_scratch,
        &artifact,
        &profiled_dst_structure,
        profiled_dst.data_mut(),
        lhs.data(),
        rhs.data(),
        alpha,
        beta,
        &mut profile,
    )?;
    Ok(profile)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_prelowered_dynamic_tree_execution_artifact_for_test<R, D>(
    rule: &R,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs: super::dynamic_space::FusionOperand<'_>,
    lhs_data: &[D],
    rhs: super::dynamic_space::FusionOperand<'_>,
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = crate::RuleIdentity>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let mut tree_context =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let mut contract_backend = DenseTreeTransformOperations::default();
    let mut contract_workspace = super::backend::TensorContractWorkspace::default();
    let mut fusion_block_workspace = FusionBlockContractWorkspace::default();
    let mut scratch = DynamicFusionScratchWorkspace::default();
    let layout_primer = encoded_layout_primer::<R>;
    let lhs_layout = lhs.prepare(rule, layout_primer)?;
    let rhs_layout = rhs.prepare(rule, layout_primer)?;
    let artifact = compile_prelowered_dynamic_tree_execution_artifact::<_, _, _, _, _, false>(
        &mut tree_context,
        rule,
        layout_primer,
        plan,
        dst_space,
        &lhs_layout,
        &rhs_layout,
        None,
    )?;
    execute_dynamic_tree_execution_artifact(
        &mut tree_context,
        &mut contract_backend,
        &mut contract_workspace,
        &mut fusion_block_workspace,
        &mut scratch,
        &artifact,
        dst_space.structure(),
        dst_data,
        lhs_data,
        rhs_data,
        alpha,
        beta,
    )
}

/// Dynamic-rank core of the TensorKit `@tensor`-shaped route: source
/// tree-pair transforms, core coupled GEMM, optional output transform. All
/// operands are (space, storage structure, raw slice) triples.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn tensorcontract_fusion_dynamic_plan_dyn_into_context<RuleKey, BT, BC, R, D>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    rule: &R,
    layout_primer: LayoutKeyBuilder<R>,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    dst_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    lhs_space: &DynamicFusionMapSpace,
    lhs_structure: &Arc<BlockStructure>,
    lhs_data: &[D],
    rhs_space: &DynamicFusionMapSpace,
    rhs_structure: &Arc<BlockStructure>,
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let artifact = compile_dynamic_tree_execution_artifact::<_, _, _, _, _, false>(
        tree_context,
        rule,
        layout_primer,
        plan,
        dst_space,
        lhs_space,
        lhs_structure,
        rhs_space,
        rhs_structure,
        None,
    )?;
    execute_dynamic_tree_execution_artifact(
        tree_context,
        contract_backend,
        contract_workspace,
        fusion_block_workspace,
        scratch,
        &artifact,
        dst_structure,
        dst_data,
        lhs_data,
        rhs_data,
        alpha,
        beta,
    )
}
