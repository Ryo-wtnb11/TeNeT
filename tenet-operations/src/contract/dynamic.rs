use tenet_core::{BlockStructure, CoreError, MultiplicityFreeRigidSymbols, TensorMap};

use crate::axis::TensorContractAxisSpec;
use crate::lowering::adjoint_fusion_space_view;
use crate::tree_context::TreeTransformExecutionContext;
use crate::tree_transform::build_tree_pair_transform_group_plan;
use crate::{
    DenseRecouplingScalar, DenseTreeTransformOperations, OperationError,
    RecouplingCoefficientAction, TreeTransformBackend, TreeTransformOperationKey,
    TreeTransformRuleCacheKey, TreeTransformWorkspace,
};

use super::backend::TensorContractBackend;
use super::dynamic_space::DynamicFusionMapSpace;
use super::dynamic_space_cache::DynamicFusionSpaceCache;
use super::fusion::{tensorcontract_fusion_explicit_plan, TensorContractFusionExplicitPlan};
use super::fusion_block::{
    tensorcontract_canonical_fusion_blocks_into_raw, CanonicalFusionBlockContractCache,
    CanonicalFusionBlockContractWorkspace,
};
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
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
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
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
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
    let mut lhs_canonical = DynamicFusionScratch::<D>::zeroed(lhs_space)?;
    let mut rhs_canonical = DynamicFusionScratch::<D>::zeroed(rhs_space)?;

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
    let mut canonical_dst = DynamicFusionScratch::<D>::zeroed(canonical_dst_space)?;
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
>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    fusion_block_cache: &mut CanonicalFusionBlockContractCache<RuleKey>,
    fusion_block_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    space_cache: &mut DynamicFusionSpaceCache<RuleKey>,
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    RuleKey: Clone + Eq + std::hash::Hash,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let (lhs_space, lhs_replay_structure) = cached_transformed_source_space_and_structure(
        space_cache,
        rule,
        lhs,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    )?;
    let (rhs_space, rhs_replay_structure) = cached_transformed_source_space_and_structure(
        space_cache,
        rule,
        rhs,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    )?;

    tree_pair_transform_typed_to_dynamic_with_context(
        tree_context,
        rule,
        plan.lhs_transform().clone(),
        scratch.prepare_lhs(lhs_space)?,
        lhs,
        &lhs_replay_structure,
        plan.lhs_source_conjugate(),
        D::one(),
        D::zero(),
    )?;
    tree_pair_transform_typed_to_dynamic_with_context(
        tree_context,
        rule,
        plan.rhs_transform().clone(),
        scratch.prepare_rhs(rhs_space)?,
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
        let (lhs_canonical, rhs_canonical) = scratch.lhs_rhs();
        return tensorcontract_dynamic_canonical_into_raw_with_cache(
            contract_backend,
            contract_workspace,
            fusion_block_cache,
            fusion_block_workspace,
            rule,
            &dst_space,
            &dst_structure,
            dst.data_mut(),
            lhs_canonical,
            rhs_canonical,
            plan.canonical_axes().as_spec(),
            alpha,
            beta,
        );
    }

    let output_dst_space = DynamicFusionMapSpace::from_typed(
        dst.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let (lhs_canonical, rhs_canonical) = scratch.lhs_rhs();
    let canonical_dst_space = space_cache.canonical_dst(
        rule,
        lhs_canonical.space(),
        rhs_canonical.space(),
        plan,
        Some(&output_dst_space),
    )?;
    scratch.prepare_dst(canonical_dst_space)?;
    {
        let (lhs_canonical, rhs_canonical, canonical_dst) = scratch.lhs_rhs_dst_mut();
        let canonical_dst_space_for_contract = canonical_dst.space().clone();
        let canonical_dst_structure = std::sync::Arc::clone(canonical_dst.space().structure());
        tensorcontract_dynamic_canonical_into_raw_with_cache(
            contract_backend,
            contract_workspace,
            fusion_block_cache,
            fusion_block_workspace,
            rule,
            &canonical_dst_space_for_contract,
            &canonical_dst_structure,
            canonical_dst.data_mut(),
            lhs_canonical,
            rhs_canonical,
            plan.canonical_axes().as_spec(),
            alpha,
            D::zero(),
        )?;
    }
    tree_pair_transform_dynamic_to_typed_with_context(
        tree_context,
        rule,
        plan.output_transform().clone(),
        dst,
        scratch.dst(),
        D::one(),
        beta,
    )
}

fn transformed_source_space_and_structure<R, D, const SRC_NOUT: usize, const SRC_NIN: usize, SSrc>(
    rule: &R,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    operation: &TreeTransformOperationKey,
    source_conjugate: bool,
) -> Result<(DynamicFusionMapSpace, std::sync::Arc<BlockStructure>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
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

fn cached_transformed_source_space_and_structure<
    RuleKey,
    R,
    D,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SSrc,
>(
    space_cache: &mut DynamicFusionSpaceCache<RuleKey>,
    rule: &R,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    operation: &TreeTransformOperationKey,
    source_conjugate: bool,
) -> Result<(DynamicFusionMapSpace, std::sync::Arc<BlockStructure>), OperationError>
where
    RuleKey: Clone + Eq + std::hash::Hash,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
{
    let src_fusion = src
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    if source_conjugate {
        let adjoint = adjoint_fusion_space_view(src_fusion)?;
        let replay_structure = std::sync::Arc::clone(adjoint.subblock_structure());
        let space = space_cache.transformed_from_typed(rule, &adjoint, operation)?;
        Ok((space, replay_structure))
    } else {
        let replay_structure = std::sync::Arc::clone(src.structure());
        let space = space_cache.transformed_from_typed(rule, src_fusion, operation)?;
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
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut DynamicFusionScratch<D>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    src_replay_structure: &std::sync::Arc<BlockStructure>,
    source_conjugate: bool,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
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

fn tree_pair_transform_typed_to_dynamic_with_context<
    RuleKey,
    BT,
    R,
    D,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SSrc,
>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut DynamicFusionScratch<D>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    src_replay_structure: &std::sync::Arc<BlockStructure>,
    source_conjugate: bool,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    RuleKey: Clone + Eq + std::hash::Hash,
    BT: TreeTransformBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    dst.fill_zero();
    let dst_structure = std::sync::Arc::clone(dst.space().structure());
    tree_context.tree_pair_transform_into_raw_with_storage_conjugation(
        rule,
        operation,
        &dst_structure,
        src_replay_structure,
        dst.data_mut(),
        src.data(),
        source_conjugate,
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
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &DynamicFusionScratch<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
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

fn tree_pair_transform_dynamic_to_typed_with_context<
    RuleKey,
    BT,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    SDst,
>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &DynamicFusionScratch<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    RuleKey: Clone + Eq + std::hash::Hash,
    BT: TreeTransformBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let dst_structure = std::sync::Arc::clone(dst.structure());
    let src_structure = std::sync::Arc::clone(src.space().structure());
    tree_context.tree_pair_transform_into_raw(
        rule,
        operation,
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

#[allow(clippy::too_many_arguments)]
fn tensorcontract_dynamic_canonical_into_raw_with_cache<B, R, D>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    cache: &mut CanonicalFusionBlockContractCache<R::Key>,
    fusion_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
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
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let plan = cache.get_or_compile(rule, dst_space, lhs.space(), rhs.space(), axes)?;
    plan.execute_raw(
        backend,
        workspace,
        fusion_workspace,
        dst_structure,
        dst_data,
        lhs.space().structure(),
        lhs.data(),
        rhs.space().structure(),
        rhs.data(),
        alpha,
        beta,
    )
}
