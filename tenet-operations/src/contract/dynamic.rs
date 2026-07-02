use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

use tenet_core::{
    BlockStructure, CoreError, FusionTreeHomSpace, HostReadableStorage, HostWritableStorage,
    MultiplicityFreeRigidSymbols, TensorMap, TensorStorage,
};

use crate::axis::{OwnedTensorContractAxisSpec, TensorContractAxisSpec};
use crate::cache::{touch_lru_key, BlockStructureCacheKey, OperationCachePolicy};
use crate::lowering::adjoint_fusion_space_view;
use crate::tree_context::TreeTransformExecutionContext;
use crate::tree_transform::build_tree_pair_transform_group_plan;
use crate::{
    DenseRecouplingScalar, DenseTreeTransformOperations, OperationError,
    RecouplingCoefficientAction, TreeTransformBackend, TreeTransformOperationKey,
    TreeTransformRuleCacheKey, TreeTransformStructure, TreeTransformWorkspace,
};

use super::backend::TensorContractBackend;
use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::{tensorcontract_fusion_explicit_plan, TensorContractFusionExplicitPlan};
use super::fusion_block::{
    tensorcontract_canonical_fusion_blocks_into_raw, CanonicalFusionBlockContractCache,
    CanonicalFusionBlockContractWorkspace,
};
use super::profile::{TensorContractFusionProfile, TensorContractFusionRoute};
use super::scratch::{DynamicFusionScratch, DynamicFusionScratchWorkspace};

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_fusion_dynamic_transforms_into_with<
    B,
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
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    let plan = tensorcontract_fusion_explicit_plan(
        rule,
        dst.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        lhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        rhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        axes,
    )?;
    let mut tree_backend = DenseTreeTransformOperations::default_executor();
    let mut tree_workspace = TreeTransformWorkspace::default();
    tensorcontract_fusion_dynamic_plan_into_with(
        &mut tree_backend,
        &mut tree_workspace,
        backend,
        workspace,
        rule,
        &plan,
        dst,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_fusion_dynamic_plan_into_with<
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
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    let (lhs_space, lhs_replay_structure) = transformed_source_space_and_structure(
        rule,
        lhs,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    )?;
    let (rhs_space, rhs_replay_structure) = transformed_source_space_and_structure(
        rule,
        rhs,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    )?;
    let mut lhs_canonical = DynamicFusionScratch::<D>::zeroed(Arc::new(lhs_space))?;
    let mut rhs_canonical = DynamicFusionScratch::<D>::zeroed(Arc::new(rhs_space))?;

    tree_pair_transform_typed_to_dynamic(
        tree_backend,
        tree_workspace,
        rule,
        plan.lhs_transform().clone(),
        &mut lhs_canonical,
        lhs,
        &lhs_replay_structure,
        plan.lhs_source_conjugate(),
        D::one(),
        D::zero(),
    )?;
    tree_pair_transform_typed_to_dynamic(
        tree_backend,
        tree_workspace,
        rule,
        plan.rhs_transform().clone(),
        &mut rhs_canonical,
        rhs,
        &rhs_replay_structure,
        plan.rhs_source_conjugate(),
        D::one(),
        D::zero(),
    )?;

    if plan.output_transform_is_identity() {
        let dst_space = DynamicFusionMapSpace::from_typed(
            dst.fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let dst_structure = std::sync::Arc::clone(dst.structure());
        return tensorcontract_dynamic_canonical_into_raw(
            contract_backend,
            contract_workspace,
            rule,
            &dst_space,
            &dst_structure,
            dst.data_mut(),
            &lhs_canonical,
            &rhs_canonical,
            plan.canonical_axes().as_spec(),
            alpha,
            beta,
        );
    }

    let output_dst_space = DynamicFusionMapSpace::from_typed(
        dst.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let canonical_dst_space = DynamicFusionMapSpace::canonical_dst(
        rule,
        lhs_canonical.space(),
        rhs_canonical.space(),
        plan,
        Some(&output_dst_space),
    )?;
    let mut canonical_dst = DynamicFusionScratch::<D>::zeroed(Arc::new(canonical_dst_space))?;
    let canonical_dst_space_for_contract = canonical_dst.space().clone();
    let canonical_dst_structure = std::sync::Arc::clone(canonical_dst.space().structure());
    tensorcontract_dynamic_canonical_into_raw(
        contract_backend,
        contract_workspace,
        rule,
        &canonical_dst_space_for_contract,
        &canonical_dst_structure,
        canonical_dst.data_mut(),
        &lhs_canonical,
        &rhs_canonical,
        plan.canonical_axes().as_spec(),
        alpha,
        D::zero(),
    )?;
    tree_pair_transform_dynamic_to_typed(
        tree_backend,
        tree_workspace,
        rule,
        plan.output_transform().clone(),
        dst,
        &canonical_dst,
        D::one(),
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn tensorcontract_fusion_dynamic_into_context<
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
    dynamic_space_cache: &mut DynamicFusionSpaceCache<RuleKey>,
    fusion_block_cache: &mut CanonicalFusionBlockContractCache<RuleKey>,
    fusion_block_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    rule: &R,
    axes: TensorContractAxisSpec<'_>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    RuleKey: Clone + Eq + std::hash::Hash,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    let plan = tensorcontract_fusion_explicit_plan(
        rule,
        dst.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        lhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        rhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        axes,
    )?;
    tensorcontract_fusion_dynamic_plan_into_context(
        tree_context,
        contract_backend,
        contract_workspace,
        dynamic_space_cache,
        fusion_block_cache,
        fusion_block_workspace,
        scratch,
        rule,
        &plan,
        dst,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn tensorcontract_fusion_dynamic_into_context_profiled<
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
    dynamic_space_cache: &mut DynamicFusionSpaceCache<RuleKey>,
    fusion_block_cache: &mut CanonicalFusionBlockContractCache<RuleKey>,
    fusion_block_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    rule: &R,
    axes: TensorContractAxisSpec<'_>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
    profile: &mut TensorContractFusionProfile,
) -> Result<(), OperationError>
where
    RuleKey: Clone + Eq + std::hash::Hash,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    profile.route = TensorContractFusionRoute::DynamicTreeCanonical;
    let start = std::time::Instant::now();
    let plan = tensorcontract_fusion_explicit_plan(
        rule,
        dst.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        lhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        rhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        axes,
    )?;
    profile.explicit_plan += start.elapsed();

    tensorcontract_fusion_dynamic_plan_into_context_profiled(
        tree_context,
        contract_backend,
        contract_workspace,
        dynamic_space_cache,
        fusion_block_cache,
        fusion_block_workspace,
        scratch,
        rule,
        &plan,
        dst,
        lhs,
        rhs,
        alpha,
        beta,
        profile,
    )
}

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
    dynamic_space_cache: &mut DynamicFusionSpaceCache<RuleKey>,
    fusion_block_cache: &mut CanonicalFusionBlockContractCache<RuleKey>,
    fusion_block_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    RuleKey: Clone + Eq + std::hash::Hash,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    let lhs_transform = dynamic_space_cache.get_or_compile_transformed_source(
        tree_context,
        rule,
        lhs,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    )?;
    let rhs_transform = dynamic_space_cache.get_or_compile_transformed_source(
        tree_context,
        rule,
        rhs,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    )?;
    let lhs_space = lhs_transform.space.clone();
    let rhs_space = rhs_transform.space.clone();

    {
        let lhs_dst_structure = std::sync::Arc::clone(lhs_space.structure());
        let lhs_scratch = scratch.prepare_lhs(lhs_space.clone())?;
        tree_context.tree_pair_transform_structure_into_raw(
            lhs_transform.transform_structure.as_ref(),
            &lhs_dst_structure,
            &lhs_transform.replay_structure,
            lhs_scratch.data_mut(),
            lhs.data(),
            D::one(),
            D::zero(),
        )?;
    }
    {
        let rhs_dst_structure = std::sync::Arc::clone(rhs_space.structure());
        let rhs_scratch = scratch.prepare_rhs(rhs_space.clone())?;
        tree_context.tree_pair_transform_structure_into_raw(
            rhs_transform.transform_structure.as_ref(),
            &rhs_dst_structure,
            &rhs_transform.replay_structure,
            rhs_scratch.data_mut(),
            rhs.data(),
            D::one(),
            D::zero(),
        )?;
    }

    if plan.output_transform_is_identity() {
        let dst_space = DynamicFusionMapSpace::from_typed(
            dst.fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let block_plan = fusion_block_cache.get_or_compile(
            rule,
            &dst_space,
            &lhs_space,
            &rhs_space,
            plan.canonical_axes().as_spec(),
        )?;
        let dst_structure = std::sync::Arc::clone(dst.structure());
        let (lhs_canonical, rhs_canonical) = scratch.lhs_rhs();
        return block_plan.execute_raw(
            contract_backend,
            contract_workspace,
            fusion_block_workspace,
            &dst_structure,
            dst.data_mut(),
            lhs_canonical.space().structure(),
            lhs_canonical.data(),
            rhs_canonical.space().structure(),
            rhs_canonical.data(),
            alpha,
            beta,
        );
    }

    let output_dst_space = DynamicFusionMapSpace::from_typed(
        dst.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let canonical_dst = dynamic_space_cache.get_or_compile_canonical_dst(
        tree_context,
        rule,
        &lhs_space,
        &rhs_space,
        plan,
        &output_dst_space,
    )?;
    let canonical_dst_space = canonical_dst.space.clone();
    let block_plan = fusion_block_cache.get_or_compile(
        rule,
        &canonical_dst_space,
        &lhs_space,
        &rhs_space,
        plan.canonical_axes().as_spec(),
    )?;
    let canonical_dst_structure = std::sync::Arc::clone(canonical_dst_space.structure());
    scratch.prepare_dst(canonical_dst_space.clone())?;
    {
        let (lhs_canonical, rhs_canonical, canonical_dst) = scratch.lhs_rhs_dst_mut();
        block_plan.execute_raw(
            contract_backend,
            contract_workspace,
            fusion_block_workspace,
            &canonical_dst_structure,
            canonical_dst.data_mut(),
            lhs_canonical.space().structure(),
            lhs_canonical.data(),
            rhs_canonical.space().structure(),
            rhs_canonical.data(),
            alpha,
            D::zero(),
        )?;
    }
    let dst_structure = std::sync::Arc::clone(dst.structure());
    tree_context.tree_pair_transform_structure_into_raw(
        canonical_dst.output_transform_structure.as_ref(),
        &dst_structure,
        &canonical_dst_structure,
        dst.data_mut(),
        scratch.dst().data(),
        D::one(),
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_fusion_dynamic_plan_into_context_profiled<
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
    dynamic_space_cache: &mut DynamicFusionSpaceCache<RuleKey>,
    fusion_block_cache: &mut CanonicalFusionBlockContractCache<RuleKey>,
    fusion_block_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
    profile: &mut TensorContractFusionProfile,
) -> Result<(), OperationError>
where
    RuleKey: Clone + Eq + std::hash::Hash,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    let start = std::time::Instant::now();
    let lhs_transform = dynamic_space_cache.get_or_compile_transformed_source(
        tree_context,
        rule,
        lhs,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    )?;
    let rhs_transform = dynamic_space_cache.get_or_compile_transformed_source(
        tree_context,
        rule,
        rhs,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    )?;
    let lhs_space = lhs_transform.space.clone();
    let rhs_space = rhs_transform.space.clone();
    profile.source_space_lookup += start.elapsed();

    {
        let start = std::time::Instant::now();
        let lhs_dst_structure = std::sync::Arc::clone(lhs_space.structure());
        let lhs_scratch = scratch.prepare_lhs(lhs_space.clone())?;
        profile.lhs_scratch_prepare += start.elapsed();

        let start = std::time::Instant::now();
        tree_context.tree_pair_transform_structure_into_raw_profiled(
            lhs_transform.transform_structure.as_ref(),
            &lhs_dst_structure,
            &lhs_transform.replay_structure,
            lhs_scratch.data_mut(),
            lhs.data(),
            D::one(),
            D::zero(),
            &mut profile.tree_replay,
        )?;
        profile.lhs_transform += start.elapsed();
        profile.lhs_transform_calls += 1;
    }
    {
        let start = std::time::Instant::now();
        let rhs_dst_structure = std::sync::Arc::clone(rhs_space.structure());
        let rhs_scratch = scratch.prepare_rhs(rhs_space.clone())?;
        profile.rhs_scratch_prepare += start.elapsed();

        let start = std::time::Instant::now();
        tree_context.tree_pair_transform_structure_into_raw_profiled(
            rhs_transform.transform_structure.as_ref(),
            &rhs_dst_structure,
            &rhs_transform.replay_structure,
            rhs_scratch.data_mut(),
            rhs.data(),
            D::one(),
            D::zero(),
            &mut profile.tree_replay,
        )?;
        profile.rhs_transform += start.elapsed();
        profile.rhs_transform_calls += 1;
    }

    if plan.output_transform_is_identity() {
        let dst_space = DynamicFusionMapSpace::from_typed(
            dst.fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let start = std::time::Instant::now();
        let block_plan = fusion_block_cache.get_or_compile(
            rule,
            &dst_space,
            &lhs_space,
            &rhs_space,
            plan.canonical_axes().as_spec(),
        )?;
        profile.fusion_block_plan_lookup += start.elapsed();

        let dst_structure = std::sync::Arc::clone(dst.structure());
        let (lhs_canonical, rhs_canonical) = scratch.lhs_rhs();
        return block_plan.execute_raw_profiled(
            contract_backend,
            contract_workspace,
            fusion_block_workspace,
            &dst_structure,
            dst.data_mut(),
            lhs_canonical.space().structure(),
            lhs_canonical.data(),
            rhs_canonical.space().structure(),
            rhs_canonical.data(),
            alpha,
            beta,
            profile,
        );
    }

    let output_dst_space = DynamicFusionMapSpace::from_typed(
        dst.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let start = std::time::Instant::now();
    let canonical_dst = dynamic_space_cache.get_or_compile_canonical_dst(
        tree_context,
        rule,
        &lhs_space,
        &rhs_space,
        plan,
        &output_dst_space,
    )?;
    let canonical_dst_space = canonical_dst.space.clone();
    profile.canonical_dst_space_lookup += start.elapsed();

    let start = std::time::Instant::now();
    let block_plan = fusion_block_cache.get_or_compile(
        rule,
        &canonical_dst_space,
        &lhs_space,
        &rhs_space,
        plan.canonical_axes().as_spec(),
    )?;
    profile.fusion_block_plan_lookup += start.elapsed();

    let canonical_dst_structure = std::sync::Arc::clone(canonical_dst_space.structure());
    let start = std::time::Instant::now();
    scratch.prepare_dst(canonical_dst_space.clone())?;
    profile.dst_scratch_prepare += start.elapsed();

    {
        let (lhs_canonical, rhs_canonical, canonical_dst) = scratch.lhs_rhs_dst_mut();
        block_plan.execute_raw_profiled(
            contract_backend,
            contract_workspace,
            fusion_block_workspace,
            &canonical_dst_structure,
            canonical_dst.data_mut(),
            lhs_canonical.space().structure(),
            lhs_canonical.data(),
            rhs_canonical.space().structure(),
            rhs_canonical.data(),
            alpha,
            D::zero(),
            profile,
        )?;
    }

    let dst_structure = std::sync::Arc::clone(dst.structure());
    let start = std::time::Instant::now();
    tree_context.tree_pair_transform_structure_into_raw_profiled(
        canonical_dst.output_transform_structure.as_ref(),
        &dst_structure,
        &canonical_dst_structure,
        dst.data_mut(),
        scratch.dst().data(),
        D::one(),
        beta,
        &mut profile.tree_replay,
    )?;
    profile.output_transform += start.elapsed();
    profile.output_transform_calls += 1;
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DynamicFusionSpaceCacheStats {
    hits: usize,
    fast_hits: usize,
    misses: usize,
}

impl DynamicFusionSpaceCacheStats {
    #[inline]
    pub(crate) fn hits(self) -> usize {
        self.hits
    }

    #[inline]
    pub(crate) fn fast_hits(self) -> usize {
        self.fast_hits
    }

    #[inline]
    pub(crate) fn misses(self) -> usize {
        self.misses
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionSpaceCache<RuleKey> {
    last_transformed_sources: Vec<DynamicFusionTransformedSourceLastEntry<RuleKey>>,
    fast_transformed_sources: HashMap<
        DynamicFusionTransformedSourceFastKey<RuleKey>,
        DynamicFusionTransformedSourceEntry,
    >,
    transformed_sources: HashMap<
        DynamicFusionTransformedSourceSpaceKey<RuleKey>,
        DynamicFusionTransformedSourceEntry,
    >,
    lru_order: VecDeque<DynamicFusionSpaceCacheEntryKey<RuleKey>>,
    last_canonical_dst: Option<DynamicFusionCanonicalDstLastEntry<RuleKey>>,
    fast_canonical_dsts:
        HashMap<DynamicFusionCanonicalDstFastKey<RuleKey>, DynamicFusionCanonicalDstEntry>,
    canonical_dsts:
        HashMap<DynamicFusionCanonicalDstSpaceKey<RuleKey>, DynamicFusionCanonicalDstEntry>,
    policy: OperationCachePolicy,
    stats: DynamicFusionSpaceCacheStats,
}

#[derive(Clone, Debug)]
struct DynamicFusionTransformedSourceEntry {
    space: Arc<DynamicFusionMapSpace>,
    replay_structure: Arc<BlockStructure>,
    transform_structure: Arc<TreeTransformStructure<f64>>,
}

#[derive(Clone, Debug)]
struct DynamicFusionCanonicalDstEntry {
    space: Arc<DynamicFusionMapSpace>,
    output_transform_structure: Arc<TreeTransformStructure<f64>>,
}

impl<RuleKey> Default for DynamicFusionSpaceCache<RuleKey> {
    fn default() -> Self {
        Self {
            last_transformed_sources: Vec::new(),
            fast_transformed_sources: HashMap::new(),
            transformed_sources: HashMap::new(),
            lru_order: VecDeque::new(),
            last_canonical_dst: None,
            fast_canonical_dsts: HashMap::new(),
            canonical_dsts: HashMap::new(),
            policy: OperationCachePolicy::default(),
            stats: DynamicFusionSpaceCacheStats::default(),
        }
    }
}

#[derive(Clone, Debug)]
struct DynamicFusionTransformedSourceLastEntry<RuleKey> {
    key: Option<DynamicFusionTransformedSourceSpaceKey<RuleKey>>,
    rule: RuleKey,
    nout: usize,
    homspace: FusionTreeHomSpace,
    replay_structure: Arc<BlockStructure>,
    operation: TreeTransformOperationKey,
    source_conjugate: bool,
    entry: DynamicFusionTransformedSourceEntry,
}

impl<RuleKey> DynamicFusionTransformedSourceLastEntry<RuleKey>
where
    RuleKey: Eq,
{
    fn matches(
        &self,
        rule: &RuleKey,
        nout: usize,
        homspace: &FusionTreeHomSpace,
        replay_structure: &Arc<BlockStructure>,
        operation: &TreeTransformOperationKey,
        source_conjugate: bool,
    ) -> bool {
        &self.rule == rule
            && self.nout == nout
            && self.homspace == *homspace
            && Arc::ptr_eq(&self.replay_structure, replay_structure)
            && &self.operation == operation
            && self.source_conjugate == source_conjugate
    }
}

#[derive(Clone, Debug)]
struct DynamicFusionCanonicalDstLastEntry<RuleKey> {
    key: Option<DynamicFusionCanonicalDstSpaceKey<RuleKey>>,
    rule: RuleKey,
    lhs: DynamicFusionLastSpaceKey,
    rhs: DynamicFusionLastSpaceKey,
    canonical_axes: OwnedTensorContractAxisSpec,
    canonical_dst_nout: usize,
    canonical_dst_nin: usize,
    output_transform: TreeTransformOperationKey,
    output_dst: DynamicFusionLastSpaceKey,
    entry: DynamicFusionCanonicalDstEntry,
}

impl<RuleKey> DynamicFusionCanonicalDstLastEntry<RuleKey>
where
    RuleKey: Eq,
{
    fn matches(
        &self,
        rule: &RuleKey,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        plan: &TensorContractFusionExplicitPlan,
        output_dst: &DynamicFusionMapSpace,
    ) -> bool {
        &self.rule == rule
            && self.lhs.matches(lhs)
            && self.rhs.matches(rhs)
            && self.canonical_axes == *plan.canonical_axes()
            && self.canonical_dst_nout == plan.canonical_dst_nout()
            && self.canonical_dst_nin == plan.canonical_dst_nin()
            && self.output_transform == *plan.output_transform()
            && self.output_dst.matches(output_dst)
    }
}

#[derive(Clone, Debug)]
struct DynamicFusionLastSpaceKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure: Arc<BlockStructure>,
}

impl DynamicFusionLastSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure: Arc::clone(space.structure()),
        }
    }

    fn matches(&self, space: &DynamicFusionMapSpace) -> bool {
        self.nout == space.nout()
            && self.homspace == *space.homspace()
            && Arc::ptr_eq(&self.structure, space.structure())
    }
}

impl<RuleKey> DynamicFusionSpaceCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.transformed_sources.len() + self.canonical_dsts.len()
    }

    #[inline]
    pub(crate) fn stats(&self) -> DynamicFusionSpaceCacheStats {
        self.stats
    }

    pub(crate) fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.clear_fast_entries();
        if !policy.stores_entries() {
            self.transformed_sources.clear();
            self.lru_order.clear();
            self.canonical_dsts.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            self.rebuild_lru_order();
            self.enforce_lru_limit(max_entries);
        }
    }

    fn clear_fast_entries(&mut self) {
        self.last_transformed_sources.clear();
        self.fast_transformed_sources.clear();
        self.last_canonical_dst = None;
        self.fast_canonical_dsts.clear();
    }

    fn rebuild_lru_order(&mut self) {
        self.lru_order.clear();
        self.lru_order.extend(
            self.transformed_sources
                .keys()
                .cloned()
                .map(DynamicFusionSpaceCacheEntryKey::TransformedSource),
        );
        self.lru_order.extend(
            self.canonical_dsts
                .keys()
                .cloned()
                .map(DynamicFusionSpaceCacheEntryKey::CanonicalDst),
        );
    }

    fn remember_transformed_source(
        &mut self,
        entry: DynamicFusionTransformedSourceLastEntry<RuleKey>,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        const LAST_TRANSFORMED_SOURCE_LIMIT: usize = 4;
        if self.last_transformed_sources.len() == LAST_TRANSFORMED_SOURCE_LIMIT {
            self.last_transformed_sources.remove(0);
        }
        self.last_transformed_sources.push(entry);
    }

    fn touch_transformed_source(&mut self, key: &DynamicFusionTransformedSourceSpaceKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.transformed_sources.contains_key(key) {
            touch_lru_key(
                &mut self.lru_order,
                &DynamicFusionSpaceCacheEntryKey::TransformedSource(key.clone()),
            );
        }
    }

    fn insert_transformed_source(
        &mut self,
        key: DynamicFusionTransformedSourceSpaceKey<RuleKey>,
        fast_key: DynamicFusionTransformedSourceFastKey<RuleKey>,
        entry: DynamicFusionTransformedSourceEntry,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.transformed_sources.insert(key.clone(), entry.clone());
        self.fast_transformed_sources.insert(fast_key, entry);
        if self.policy.max_entries().is_some() {
            self.touch_transformed_source(&key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_lru_limit(max_entries);
        }
    }

    fn touch_canonical_dst(&mut self, key: &DynamicFusionCanonicalDstSpaceKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.canonical_dsts.contains_key(key) {
            touch_lru_key(
                &mut self.lru_order,
                &DynamicFusionSpaceCacheEntryKey::CanonicalDst(key.clone()),
            );
        }
    }

    fn insert_canonical_dst(
        &mut self,
        key: DynamicFusionCanonicalDstSpaceKey<RuleKey>,
        fast_key: DynamicFusionCanonicalDstFastKey<RuleKey>,
        entry: DynamicFusionCanonicalDstEntry,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.canonical_dsts.insert(key.clone(), entry.clone());
        self.fast_canonical_dsts.insert(fast_key, entry);
        if self.policy.max_entries().is_some() {
            self.touch_canonical_dst(&key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_lru_limit(max_entries);
        }
    }

    fn enforce_lru_limit(&mut self, max_entries: usize) {
        let mut evicted_transformed_source = false;
        let mut evicted_canonical_dst = false;
        while self.len() > max_entries {
            let Some(oldest) = self.lru_order.pop_front() else {
                break;
            };
            match oldest {
                DynamicFusionSpaceCacheEntryKey::TransformedSource(key) => {
                    evicted_transformed_source |= self.transformed_sources.remove(&key).is_some();
                }
                DynamicFusionSpaceCacheEntryKey::CanonicalDst(key) => {
                    evicted_canonical_dst |= self.canonical_dsts.remove(&key).is_some();
                }
            }
        }
        if evicted_transformed_source {
            self.last_transformed_sources.clear();
            self.fast_transformed_sources.clear();
        }
        if evicted_canonical_dst {
            self.last_canonical_dst = None;
            self.fast_canonical_dsts.clear();
        }
    }

    fn get_or_compile_transformed_source<
        R,
        D,
        BT,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SSrc,
        DSrc,
    >(
        &mut self,
        tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        rule: &R,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        operation: &TreeTransformOperationKey,
        source_conjugate: bool,
    ) -> Result<DynamicFusionTransformedSourceEntry, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar,
        BT: TreeTransformBackend<D, f64>,
        DSrc: TensorStorage<D>,
    {
        let src_fusion = src
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let rule_key = rule.tree_transform_rule_cache_key();
        let nout = if source_conjugate { SRC_NIN } else { SRC_NOUT };
        if self.policy.stores_entries() && !source_conjugate {
            let refresh_lru = self.policy.max_entries().is_some();
            let homspace = src_fusion.homspace();
            let replay_structure = src.structure();
            let last_hit = self.last_transformed_sources.iter().find_map(|last| {
                if last.matches(
                    &rule_key,
                    nout,
                    homspace,
                    replay_structure,
                    operation,
                    source_conjugate,
                ) {
                    Some((
                        refresh_lru.then(|| last.key.clone()).flatten(),
                        last.entry.clone(),
                    ))
                } else {
                    None
                }
            });
            if let Some((key, entry)) = last_hit {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if let Some(key) = key.as_ref() {
                    self.touch_transformed_source(key);
                }
                return Ok(entry);
            }
        }
        let (homspace, replay_structure) = if source_conjugate {
            let adjoint = adjoint_fusion_space_view(src_fusion)?;
            (
                adjoint.homspace().clone(),
                std::sync::Arc::clone(adjoint.subblock_structure()),
            )
        } else {
            (
                src_fusion.homspace().clone(),
                std::sync::Arc::clone(src.structure()),
            )
        };
        if self.policy.stores_entries() && source_conjugate {
            let refresh_lru = self.policy.max_entries().is_some();
            let last_hit = self.last_transformed_sources.iter().find_map(|last| {
                if last.matches(
                    &rule_key,
                    nout,
                    &homspace,
                    &replay_structure,
                    operation,
                    source_conjugate,
                ) {
                    Some((
                        refresh_lru.then(|| last.key.clone()).flatten(),
                        last.entry.clone(),
                    ))
                } else {
                    None
                }
            });
            if let Some((key, entry)) = last_hit {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if let Some(key) = key.as_ref() {
                    self.touch_transformed_source(key);
                }
                return Ok(entry);
            }
        }
        if !self.policy.stores_entries() {
            self.stats.misses += 1;
            let space = if source_conjugate {
                let adjoint = adjoint_fusion_space_view(src_fusion)?;
                DynamicFusionMapSpace::transformed_from_typed(rule, &adjoint, operation)?
            } else {
                DynamicFusionMapSpace::transformed_from_typed(rule, src_fusion, operation)?
            };
            let dst_structure = Arc::clone(space.structure());
            let transform_structure = tree_context
                .get_or_compile_tree_pair_structure_with_storage_conjugation(
                    rule,
                    operation.clone(),
                    &dst_structure,
                    &replay_structure,
                    source_conjugate,
                )?;
            return Ok(DynamicFusionTransformedSourceEntry {
                space: Arc::new(space),
                replay_structure,
                transform_structure,
            });
        }

        let fast_key = DynamicFusionTransformedSourceFastKey {
            rule: rule_key.clone(),
            nout,
            homspace: homspace.clone(),
            replay_structure_ptr: Arc::as_ptr(&replay_structure) as usize,
            operation: operation.clone(),
            source_conjugate,
        };
        let lru_key = if self.policy.max_entries().is_some() {
            Some(DynamicFusionTransformedSourceSpaceKey {
                rule: rule_key.clone(),
                nout,
                homspace: homspace.clone(),
                structure: BlockStructureCacheKey::from_structure(&replay_structure)?,
                operation: operation.clone(),
                source_conjugate,
            })
        } else {
            None
        };
        if let Some(entry) = self.fast_transformed_sources.get(&fast_key) {
            let entry = entry.clone();
            self.stats.hits += 1;
            self.stats.fast_hits += 1;
            if let Some(key) = lru_key.as_ref() {
                self.touch_transformed_source(key);
            }
            self.remember_transformed_source(DynamicFusionTransformedSourceLastEntry {
                key: lru_key,
                rule: rule_key,
                nout,
                homspace,
                replay_structure,
                operation: operation.clone(),
                source_conjugate,
                entry: entry.clone(),
            });
            return Ok(entry);
        }
        let key = if let Some(key) = lru_key {
            key
        } else {
            DynamicFusionTransformedSourceSpaceKey {
                rule: rule_key.clone(),
                nout,
                homspace: homspace.clone(),
                structure: BlockStructureCacheKey::from_structure(&replay_structure)?,
                operation: operation.clone(),
                source_conjugate,
            }
        };
        if let Some(entry) = self.transformed_sources.get(&key) {
            let entry = entry.clone();
            self.stats.hits += 1;
            self.touch_transformed_source(&key);
            self.fast_transformed_sources
                .insert(fast_key, entry.clone());
            self.remember_transformed_source(DynamicFusionTransformedSourceLastEntry {
                key: Some(key.clone()),
                rule: rule_key,
                nout,
                homspace,
                replay_structure,
                operation: operation.clone(),
                source_conjugate,
                entry: entry.clone(),
            });
            return Ok(entry);
        }

        self.stats.misses += 1;
        let space = if source_conjugate {
            let adjoint = adjoint_fusion_space_view(src_fusion)?;
            DynamicFusionMapSpace::transformed_from_typed(rule, &adjoint, operation)?
        } else {
            DynamicFusionMapSpace::transformed_from_typed(rule, src_fusion, operation)?
        };
        let dst_structure = Arc::clone(space.structure());
        let transform_structure = tree_context
            .get_or_compile_tree_pair_structure_with_storage_conjugation(
                rule,
                operation.clone(),
                &dst_structure,
                &replay_structure,
                source_conjugate,
            )?;
        let entry = DynamicFusionTransformedSourceEntry {
            space: Arc::new(space),
            replay_structure,
            transform_structure,
        };
        let last_key = key.clone();
        self.insert_transformed_source(key, fast_key, entry.clone());
        self.remember_transformed_source(DynamicFusionTransformedSourceLastEntry {
            key: Some(last_key),
            rule: rule_key,
            nout,
            homspace,
            replay_structure: Arc::clone(&entry.replay_structure),
            operation: operation.clone(),
            source_conjugate,
            entry: entry.clone(),
        });
        Ok(entry)
    }

    fn get_or_compile_canonical_dst<R, D, BT>(
        &mut self,
        tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        rule: &R,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        plan: &TensorContractFusionExplicitPlan,
        output_dst: &DynamicFusionMapSpace,
    ) -> Result<DynamicFusionCanonicalDstEntry, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar,
        BT: TreeTransformBackend<D, f64>,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        if !self.policy.stores_entries() {
            self.stats.misses += 1;
            let space =
                DynamicFusionMapSpace::canonical_dst(rule, lhs, rhs, plan, Some(output_dst))?;
            let dst_structure = Arc::clone(output_dst.structure());
            let src_structure = Arc::clone(space.structure());
            let output_transform_structure = tree_context
                .get_or_compile_tree_pair_structure_with_storage_conjugation(
                    rule,
                    plan.output_transform().clone(),
                    &dst_structure,
                    &src_structure,
                    false,
                )?;
            return Ok(DynamicFusionCanonicalDstEntry {
                space: Arc::new(space),
                output_transform_structure,
            });
        }
        if let Some(last) = &self.last_canonical_dst {
            if last.matches(&rule_key, lhs, rhs, plan, output_dst) {
                let key = self
                    .policy
                    .max_entries()
                    .is_some()
                    .then(|| last.key.clone())
                    .flatten();
                let entry = last.entry.clone();
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if let Some(key) = key.as_ref() {
                    self.touch_canonical_dst(key);
                }
                return Ok(entry);
            }
        }
        let fast_key = DynamicFusionCanonicalDstFastKey {
            rule: rule_key.clone(),
            lhs: DynamicFusionFastSpaceKey::from_space(lhs),
            rhs: DynamicFusionFastSpaceKey::from_space(rhs),
            canonical_axes: plan.canonical_axes().clone(),
            canonical_dst_nout: plan.canonical_dst_nout(),
            canonical_dst_nin: plan.canonical_dst_nin(),
            output_transform: plan.output_transform().clone(),
            output_dst: DynamicFusionFastSpaceKey::from_space(output_dst),
        };
        let lru_key = if self.policy.max_entries().is_some() {
            Some(DynamicFusionCanonicalDstSpaceKey {
                rule: rule_key.clone(),
                lhs: DynamicFusionSpaceKey::from_space(lhs)?,
                rhs: DynamicFusionSpaceKey::from_space(rhs)?,
                canonical_axes: plan.canonical_axes().clone(),
                canonical_dst_nout: plan.canonical_dst_nout(),
                canonical_dst_nin: plan.canonical_dst_nin(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionSpaceKey::from_space(output_dst)?,
            })
        } else {
            None
        };
        if let Some(entry) = self.fast_canonical_dsts.get(&fast_key) {
            let entry = entry.clone();
            self.stats.hits += 1;
            self.stats.fast_hits += 1;
            if let Some(key) = lru_key.as_ref() {
                self.touch_canonical_dst(key);
            }
            self.last_canonical_dst = Some(DynamicFusionCanonicalDstLastEntry {
                key: lru_key,
                rule: rule_key,
                lhs: DynamicFusionLastSpaceKey::from_space(lhs),
                rhs: DynamicFusionLastSpaceKey::from_space(rhs),
                canonical_axes: plan.canonical_axes().clone(),
                canonical_dst_nout: plan.canonical_dst_nout(),
                canonical_dst_nin: plan.canonical_dst_nin(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionLastSpaceKey::from_space(output_dst),
                entry: entry.clone(),
            });
            return Ok(entry);
        }
        let key = if let Some(key) = lru_key {
            key
        } else {
            DynamicFusionCanonicalDstSpaceKey {
                rule: rule_key.clone(),
                lhs: DynamicFusionSpaceKey::from_space(lhs)?,
                rhs: DynamicFusionSpaceKey::from_space(rhs)?,
                canonical_axes: plan.canonical_axes().clone(),
                canonical_dst_nout: plan.canonical_dst_nout(),
                canonical_dst_nin: plan.canonical_dst_nin(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionSpaceKey::from_space(output_dst)?,
            }
        };
        if let Some(entry) = self.canonical_dsts.get(&key) {
            let entry = entry.clone();
            self.stats.hits += 1;
            self.touch_canonical_dst(&key);
            self.fast_canonical_dsts.insert(fast_key, entry.clone());
            self.last_canonical_dst = Some(DynamicFusionCanonicalDstLastEntry {
                key: Some(key.clone()),
                rule: rule_key,
                lhs: DynamicFusionLastSpaceKey::from_space(lhs),
                rhs: DynamicFusionLastSpaceKey::from_space(rhs),
                canonical_axes: plan.canonical_axes().clone(),
                canonical_dst_nout: plan.canonical_dst_nout(),
                canonical_dst_nin: plan.canonical_dst_nin(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionLastSpaceKey::from_space(output_dst),
                entry: entry.clone(),
            });
            return Ok(entry);
        }

        self.stats.misses += 1;
        let space = DynamicFusionMapSpace::canonical_dst(rule, lhs, rhs, plan, Some(output_dst))?;
        let dst_structure = Arc::clone(output_dst.structure());
        let src_structure = Arc::clone(space.structure());
        let output_transform_structure = tree_context
            .get_or_compile_tree_pair_structure_with_storage_conjugation(
                rule,
                plan.output_transform().clone(),
                &dst_structure,
                &src_structure,
                false,
            )?;
        let entry = DynamicFusionCanonicalDstEntry {
            space: Arc::new(space),
            output_transform_structure,
        };
        let last_key = key.clone();
        self.insert_canonical_dst(key, fast_key, entry.clone());
        self.last_canonical_dst = Some(DynamicFusionCanonicalDstLastEntry {
            key: Some(last_key),
            rule: rule_key,
            lhs: DynamicFusionLastSpaceKey::from_space(lhs),
            rhs: DynamicFusionLastSpaceKey::from_space(rhs),
            canonical_axes: plan.canonical_axes().clone(),
            canonical_dst_nout: plan.canonical_dst_nout(),
            canonical_dst_nin: plan.canonical_dst_nin(),
            output_transform: plan.output_transform().clone(),
            output_dst: DynamicFusionLastSpaceKey::from_space(output_dst),
            entry: entry.clone(),
        });
        Ok(entry)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DynamicFusionSpaceCacheEntryKey<RuleKey> {
    TransformedSource(DynamicFusionTransformedSourceSpaceKey<RuleKey>),
    CanonicalDst(DynamicFusionCanonicalDstSpaceKey<RuleKey>),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionTransformedSourceFastKey<RuleKey> {
    rule: RuleKey,
    nout: usize,
    homspace: FusionTreeHomSpace,
    replay_structure_ptr: usize,
    operation: TreeTransformOperationKey,
    source_conjugate: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionTransformedSourceSpaceKey<RuleKey> {
    rule: RuleKey,
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure: BlockStructureCacheKey,
    operation: TreeTransformOperationKey,
    source_conjugate: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionCanonicalDstFastKey<RuleKey> {
    rule: RuleKey,
    lhs: DynamicFusionFastSpaceKey,
    rhs: DynamicFusionFastSpaceKey,
    canonical_axes: OwnedTensorContractAxisSpec,
    canonical_dst_nout: usize,
    canonical_dst_nin: usize,
    output_transform: TreeTransformOperationKey,
    output_dst: DynamicFusionFastSpaceKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionCanonicalDstSpaceKey<RuleKey> {
    rule: RuleKey,
    lhs: DynamicFusionSpaceKey,
    rhs: DynamicFusionSpaceKey,
    canonical_axes: OwnedTensorContractAxisSpec,
    canonical_dst_nout: usize,
    canonical_dst_nin: usize,
    output_transform: TreeTransformOperationKey,
    output_dst: DynamicFusionSpaceKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionFastSpaceKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure_ptr: usize,
}

impl DynamicFusionFastSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure_ptr: Arc::as_ptr(space.structure()) as usize,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionSpaceKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure: BlockStructureCacheKey,
}

impl DynamicFusionSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Result<Self, OperationError> {
        Ok(Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure: BlockStructureCacheKey::from_structure(space.structure())?,
        })
    }
}

fn transformed_source_space_and_structure<
    R,
    D,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SSrc,
    DSrc,
>(
    rule: &R,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    operation: &TreeTransformOperationKey,
    source_conjugate: bool,
) -> Result<(DynamicFusionMapSpace, std::sync::Arc<BlockStructure>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    DSrc: TensorStorage<D>,
{
    let src_fusion = src
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    if source_conjugate {
        let adjoint = adjoint_fusion_space_view(src_fusion)?;
        let replay_structure = std::sync::Arc::clone(adjoint.subblock_structure());
        let space = DynamicFusionMapSpace::transformed_from_typed(rule, &adjoint, operation)?;
        Ok((space, replay_structure))
    } else {
        let replay_structure = std::sync::Arc::clone(src.structure());
        let space = DynamicFusionMapSpace::transformed_from_typed(rule, src_fusion, operation)?;
        Ok((space, replay_structure))
    }
}

fn tree_pair_transform_typed_to_dynamic<
    BT,
    R,
    D,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SSrc,
    DSrc,
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut DynamicFusionScratch<D>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    src_replay_structure: &std::sync::Arc<BlockStructure>,
    source_conjugate: bool,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DSrc: HostReadableStorage<D>,
{
    dst.fill_zero();
    let plan = build_tree_pair_transform_group_plan(rule, operation, src_replay_structure)?;
    let structure = plan.compile_structures_with_storage_conjugation(
        dst.space().structure(),
        src_replay_structure,
        source_conjugate,
    )?;
    let dst_structure = std::sync::Arc::clone(dst.space().structure());
    tree_backend.tree_transform_structure_into_raw(
        tree_workspace,
        &structure,
        &dst_structure,
        src_replay_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
    )
}

fn tree_pair_transform_dynamic_to_typed<
    BT,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    SDst,
    DDst,
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &DynamicFusionScratch<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
{
    let plan = build_tree_pair_transform_group_plan(rule, operation, src.space().structure())?;
    let structure = plan.compile_structures(dst.structure(), src.space().structure())?;
    let dst_structure = std::sync::Arc::clone(dst.structure());
    let src_structure = std::sync::Arc::clone(src.space().structure());
    tree_backend.tree_transform_structure_into_raw(
        tree_workspace,
        &structure,
        &dst_structure,
        &src_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
fn tensorcontract_dynamic_canonical_into_raw<B, R, D>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_structure: &std::sync::Arc<BlockStructure>,
    dst_data: &mut [D],
    lhs: &DynamicFusionScratch<D>,
    rhs: &DynamicFusionScratch<D>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let _ = dst_structure;
    tensorcontract_canonical_fusion_blocks_into_raw(
        backend,
        workspace,
        rule,
        dst_space,
        dst_data,
        lhs.space(),
        lhs.data(),
        rhs.space(),
        rhs.data(),
        axes,
        alpha,
        beta,
    )
}
