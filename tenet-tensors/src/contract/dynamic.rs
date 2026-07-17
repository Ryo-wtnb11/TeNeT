use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::sync::Arc;

use tenet_core::{
    BlockStructure, CoreError, FusionTreeHomSpace, HostReadableStorage, HostWritableStorage,
    MultiplicityFreeRigidSymbols, ScratchStorage, SimilarStorage, TensorMap, TensorStorage,
};

use crate::cache::{touch_lru_key, BlockStructureCacheKey, OperationCachePolicy};
use crate::lowering::{
    adjoint_fusion_space_view, prelowered_storage_axis, prelowered_storage_block_index,
};
use crate::tree_context::TreeTransformExecutionContext;
use crate::tree_transform::build_tree_pair_transform_group_plan;
use crate::{
    DenseRecouplingScalar, DenseTreeTransformOperations, OperationError,
    RecouplingCoefficientAction, TreeTransformBackend, TreeTransformOperation,
    TreeTransformRuleCacheKey, TreeTransformStructure, TreeTransformWorkspace,
};
use tenet_operations::fusion_replay::FusionBlockContractPlan;
use tenet_operations::{TensorContractSpec, TensorContractSpecOwned};

use super::backend::TensorContractBackend;
use super::dynamic_space::{encoded_layout_primer, DynamicFusionMapSpace, LayoutKeyBuilder};
use super::fusion::{prepare_tensorcontract_fusion_plan, FusionContractPlan};
use super::fusion_block::{
    tensorcontract_core_fusion_blocks_into_raw, FusionBlockContractWorkspace,
};
use super::resolution::rhs_contract_requires_twist;
use super::scratch::{
    DynamicFusionScratch, DynamicFusionScratchWorkspace, StorageDynamicFusionScratch,
    StorageDynamicFusionScratchWorkspace,
};
use crate::storage_scratch::StorageFusionBlockContractWorkspace;
use tenet_operations::TensorContractFusionProfile;

#[derive(Clone, Copy)]
struct CoreSource<'a, D> {
    space: &'a DynamicFusionMapSpace,
    data: &'a [D],
}

impl<'a, D> CoreSource<'a, D> {
    fn borrowed(space: &'a DynamicFusionMapSpace, data: &'a [D]) -> Self {
        Self { space, data }
    }

    fn materialized(space: &'a DynamicFusionMapSpace, data: &'a [D]) -> Self {
        Self { space, data }
    }

    fn from_host_scratch(scratch: &'a DynamicFusionScratch<D>) -> Self {
        Self::materialized(scratch.space(), scratch.data())
    }

    fn space(self) -> &'a DynamicFusionMapSpace {
        self.space
    }

    fn structure(self) -> &'a Arc<BlockStructure> {
        // Why not retain the input structure separately: borrowability proves
        // identical core layout, so the core space remains the single authority.
        self.space().structure()
    }

    fn data(self) -> &'a [D] {
        self.data
    }
}

fn select_core_source<'a, D>(
    borrow: bool,
    borrowed_space: &'a DynamicFusionMapSpace,
    borrowed_data: &'a [D],
    materialize: impl FnOnce() -> CoreSource<'a, D>,
) -> CoreSource<'a, D> {
    if borrow {
        CoreSource::borrowed(borrowed_space, borrowed_data)
    } else {
        materialize()
    }
}

pub(super) fn source_layout_metadata_is_borrowable<HomSpaceMatches>(
    source_space: &DynamicFusionMapSpace,
    core_nout: usize,
    core_rank: usize,
    homspace_matches: HomSpaceMatches,
    operation: &TreeTransformOperation,
    source_conjugate: bool,
) -> bool
where
    HomSpaceMatches: FnOnce() -> bool,
{
    if source_conjugate {
        return false;
    }
    let TreeTransformOperation::Permute {
        codomain_permutation,
        domain_permutation,
    } = operation
    else {
        return false;
    };
    if !codomain_permutation
        .iter()
        .copied()
        .eq(0..source_space.nout())
        || !domain_permutation
            .iter()
            .copied()
            .eq(source_space.nout()..source_space.rank())
    {
        return false;
    }
    if core_nout != source_space.nout() || core_rank != source_space.rank() || !homspace_matches() {
        return false;
    }
    true
}

#[cfg(test)]
std::thread_local! {
    static SOURCE_LAYOUT_HOMSPACE_ID_COMPARISONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_source_layout_homspace_id_comparisons() {
    SOURCE_LAYOUT_HOMSPACE_ID_COMPARISONS.set(0);
}

#[cfg(test)]
pub(crate) fn source_layout_homspace_id_comparisons() -> usize {
    SOURCE_LAYOUT_HOMSPACE_ID_COMPARISONS.get()
}

fn source_layout_homspaces_match_by_id(
    source_space: &DynamicFusionMapSpace,
    core_space: &DynamicFusionMapSpace,
) -> bool {
    #[cfg(test)]
    SOURCE_LAYOUT_HOMSPACE_ID_COMPARISONS.set(SOURCE_LAYOUT_HOMSPACE_ID_COMPARISONS.get() + 1);
    core_space.homspace().id() == source_space.homspace().id()
}

fn source_is_borrowable_core_layout(
    source_space: &DynamicFusionMapSpace,
    source_structure: &Arc<BlockStructure>,
    core_space: &DynamicFusionMapSpace,
    operation: &TreeTransformOperation,
    source_conjugate: bool,
) -> bool {
    if !source_layout_metadata_is_borrowable(
        source_space,
        core_space.nout(),
        core_space.rank(),
        || source_layout_homspaces_match_by_id(source_space, core_space),
        operation,
        source_conjugate,
    ) {
        return false;
    }
    let core_structure = core_space.structure();
    // Why not compare only the source's declared structure: even identity axes
    // can complete a sparse fusion-tree grid with structural-zero core blocks.
    Arc::ptr_eq(core_structure, source_structure)
        || core_structure.content_id() == source_structure.content_id()
        // Why not rely on content ids alone: an intern reset can assign a new
        // monotonic id to equal live content while an operation cache pins both.
        || core_structure.as_ref() == source_structure.as_ref()
}

fn rhs_source_is_borrowable<R>(
    rule: &R,
    source_space: &DynamicFusionMapSpace,
    source_structure: &Arc<BlockStructure>,
    core_space: &DynamicFusionMapSpace,
    operation: &TreeTransformOperation,
    source_conjugate: bool,
    core_axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if !source_is_borrowable_core_layout(
        source_space,
        source_structure,
        core_space,
        operation,
        source_conjugate,
    ) {
        return Ok(false);
    }
    Ok(!rhs_contract_requires_twist(rule, core_space, core_axes)?)
}

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
    axes: TensorContractSpec<'_>,
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
    let plan = prepare_tensorcontract_fusion_plan(
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
    plan: &FusionContractPlan,
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
    let lhs_source_space = DynamicFusionMapSpace::from_typed(
        lhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let lhs_transformed = transformed_source_space_and_structure(
        rule,
        lhs,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    )?;
    let lhs_borrowed = source_is_borrowable_core_layout(
        &lhs_source_space,
        lhs.structure(),
        &lhs_transformed.0,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    );
    let (rhs_space, rhs_replay_structure) = transformed_source_space_and_structure(
        rule,
        rhs,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    )?;
    let rhs_source_space = DynamicFusionMapSpace::from_typed(
        rhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let rhs_borrowed = rhs_source_is_borrowable(
        rule,
        &rhs_source_space,
        rhs.structure(),
        &rhs_space,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
        plan.core_axes().as_spec(),
    )?;
    let mut lhs_core = (!lhs_borrowed)
        .then(|| DynamicFusionScratch::<D>::zeroed(Arc::new(lhs_transformed.0.clone())))
        .transpose()?;
    let mut rhs_core = (!rhs_borrowed)
        .then(|| DynamicFusionScratch::<D>::zeroed(Arc::new(rhs_space.clone())))
        .transpose()?;

    if let Some(lhs_core) = lhs_core.as_mut() {
        tree_pair_transform_typed_to_dynamic(
            tree_backend,
            tree_workspace,
            rule,
            plan.lhs_transform().clone(),
            lhs_core,
            lhs,
            &lhs_transformed.1,
            plan.lhs_source_conjugate(),
            D::one(),
        )?;
    }
    if let Some(rhs_core) = rhs_core.as_mut() {
        tree_pair_transform_typed_to_dynamic(
            tree_backend,
            tree_workspace,
            rule,
            plan.rhs_transform().clone(),
            rhs_core,
            rhs,
            &rhs_replay_structure,
            plan.rhs_source_conjugate(),
            D::one(),
        )?;
        let rhs_scratch_space = rhs_core.space().clone();
        apply_rhs_contract_twist(
            &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                tree_backend.transpose_backend(),
            ),
            rule,
            &rhs_scratch_space,
            rhs_core.data_mut(),
            plan.core_axes().as_spec().rhs_contracting_axes(),
        )?;
    }

    let lhs_core = match lhs_core.as_ref() {
        Some(scratch) => CoreSource::from_host_scratch(scratch),
        None => CoreSource::borrowed(&lhs_transformed.0, lhs.data()),
    };
    let rhs_core_view = select_core_source(rhs_borrowed, &rhs_space, rhs.data(), || {
        CoreSource::from_host_scratch(
            rhs_core
                .as_ref()
                .expect("non-borrowed RHS materialized before core contraction"),
        )
    });

    if plan.output_transform_is_identity() {
        let dst_space = DynamicFusionMapSpace::from_typed(
            dst.fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let dst_structure = std::sync::Arc::clone(dst.structure());
        return tensorcontract_dynamic_core_into_raw(
            contract_backend,
            contract_workspace,
            rule,
            &dst_space,
            &dst_structure,
            dst.data_mut(),
            lhs_core,
            rhs_core_view,
            plan.core_axes().as_spec(),
            alpha,
            beta,
        );
    }

    let core_dst_space =
        DynamicFusionMapSpace::core_dst(rule, lhs_core.space(), rhs_core_view.space(), plan)?;
    let mut core_dst = DynamicFusionScratch::<D>::zeroed(Arc::new(core_dst_space))?;
    let core_dst_space_for_contract = core_dst.space().clone();
    let core_dst_structure = std::sync::Arc::clone(core_dst.space().structure());
    tensorcontract_dynamic_core_into_raw(
        contract_backend,
        contract_workspace,
        rule,
        &core_dst_space_for_contract,
        &core_dst_structure,
        core_dst.data_mut(),
        lhs_core,
        rhs_core_view,
        plan.core_axes().as_spec(),
        alpha,
        D::zero(),
    )?;
    tree_pair_transform_dynamic_to_typed(
        tree_backend,
        tree_workspace,
        rule,
        plan.output_transform().clone(),
        dst,
        &core_dst,
        D::one(),
        beta,
    )
}

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
    dynamic_space_cache: &mut DynamicFusionSpaceCache<RuleKey>,
    fusion_block_cache: &mut super::resolution::ContractionResolutionCache<RuleKey>,
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
        dynamic_space_cache,
        fusion_block_cache,
        fusion_block_workspace,
        scratch,
        rule,
        encoded_layout_primer::<R>,
        plan,
        &dst_space,
        &dst_structure,
        dst.data_mut(),
        &lhs_space,
        None,
        &lhs_structure,
        lhs.data(),
        &rhs_space,
        None,
        &rhs_structure,
        rhs.data(),
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
    dynamic_space_cache: &mut DynamicFusionSpaceCache<RuleKey>,
    fusion_block_cache: &mut super::resolution::ContractionResolutionCache<RuleKey>,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    rule: &R,
    layout_primer: LayoutKeyBuilder<R>,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    dst_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    lhs_space: &DynamicFusionMapSpace,
    lhs_storage_space: Option<&DynamicFusionMapSpace>,
    lhs_structure: &Arc<BlockStructure>,
    lhs_data: &[D],
    rhs_space: &DynamicFusionMapSpace,
    rhs_storage_space: Option<&DynamicFusionMapSpace>,
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
    let artifact = compile_dynamic_tree_execution_artifact(
        tree_context,
        dynamic_space_cache,
        fusion_block_cache,
        rule,
        layout_primer,
        plan,
        dst_space,
        lhs_space,
        lhs_storage_space,
        lhs_structure,
        rhs_space,
        rhs_storage_space,
        rhs_structure,
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

#[derive(Clone, Debug)]
pub(crate) struct DynamicTreeExecutionArtifact {
    lhs_transform: DynamicFusionTransformedSourceEntry,
    rhs_transform: DynamicFusionTransformedSourceEntry,
    lhs_borrowed: bool,
    rhs_borrowed: bool,
    rhs_twist: Arc<[RhsTwistAction]>,
    core_dst: Option<DynamicFusionCoreDstEntry>,
    block_plan: Arc<FusionBlockContractPlan>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_dynamic_tree_execution_artifact<RuleKey, BT, R, D>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    dynamic_space_cache: &mut DynamicFusionSpaceCache<RuleKey>,
    fusion_block_cache: &mut super::resolution::ContractionResolutionCache<RuleKey>,
    rule: &R,
    layout_primer: LayoutKeyBuilder<R>,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    lhs_storage_space: Option<&DynamicFusionMapSpace>,
    lhs_structure: &Arc<BlockStructure>,
    rhs_space: &DynamicFusionMapSpace,
    rhs_storage_space: Option<&DynamicFusionMapSpace>,
    rhs_structure: &Arc<BlockStructure>,
) -> Result<DynamicTreeExecutionArtifact, OperationError>
where
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar,
{
    let lhs_transform = match lhs_storage_space {
        Some(storage_space) => dynamic_space_cache.get_or_compile_transformed_source_prelowered(
            tree_context,
            rule,
            lhs_space,
            storage_space,
            lhs_structure,
            plan.lhs_transform(),
            plan.lhs_source_conjugate(),
            layout_primer,
        )?,
        None => dynamic_space_cache.get_or_compile_transformed_source(
            tree_context,
            rule,
            lhs_space,
            lhs_structure,
            plan.lhs_transform(),
            plan.lhs_source_conjugate(),
            layout_primer,
        )?,
    };
    let lhs_borrowed = source_is_borrowable_core_layout(
        lhs_space,
        lhs_structure,
        &lhs_transform.space,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    );
    let rhs_transform = match rhs_storage_space {
        Some(storage_space) => dynamic_space_cache.get_or_compile_transformed_source_prelowered(
            tree_context,
            rule,
            rhs_space,
            storage_space,
            rhs_structure,
            plan.rhs_transform(),
            plan.rhs_source_conjugate(),
            layout_primer,
        )?,
        None => dynamic_space_cache.get_or_compile_transformed_source(
            tree_context,
            rule,
            rhs_space,
            rhs_structure,
            plan.rhs_transform(),
            plan.rhs_source_conjugate(),
            layout_primer,
        )?,
    };
    let lhs_core_space = lhs_transform.space.clone();
    let rhs_core_space = rhs_transform.space.clone();
    let rhs_borrowed = rhs_source_is_borrowable(
        rule,
        rhs_space,
        rhs_structure,
        &rhs_core_space,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
        plan.core_axes().as_spec(),
    )?;
    let rhs_twist = compile_rhs_contract_twist(
        rule,
        &rhs_core_space,
        plan.core_axes().as_spec().rhs_contracting_axes(),
    )?;

    let core_dst = if plan.output_transform_is_identity() {
        None
    } else {
        Some(dynamic_space_cache.get_or_compile_core_dst(
            tree_context,
            rule,
            &lhs_core_space,
            &rhs_core_space,
            plan,
            dst_space,
            layout_primer,
        )?)
    };
    let block_dst_space = core_dst
        .as_ref()
        .map_or(dst_space, |entry| entry.space.as_ref());
    let block_plan = fusion_block_cache.get_or_compile_core_plan(
        rule,
        block_dst_space,
        &lhs_core_space,
        &rhs_core_space,
        plan.core_axes().as_spec(),
    )?;
    Ok(DynamicTreeExecutionArtifact {
        lhs_transform,
        rhs_transform,
        lhs_borrowed,
        rhs_borrowed,
        rhs_twist,
        core_dst,
        block_plan,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_dynamic_tree_execution_artifact<RuleKey, BT, BC, D>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    artifact: &DynamicTreeExecutionArtifact,
    dst_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    lhs_data: &[D],
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let lhs_transform = &artifact.lhs_transform;
    let rhs_transform = &artifact.rhs_transform;
    let lhs_borrowed = artifact.lhs_borrowed;
    let rhs_borrowed = artifact.rhs_borrowed;
    let lhs_core_space = lhs_transform.space.clone();
    let rhs_core_space = rhs_transform.space.clone();

    if !lhs_borrowed {
        let lhs_dst_structure = std::sync::Arc::clone(lhs_core_space.structure());
        let lhs_scratch = scratch.prepare_lhs(lhs_transform.space.clone())?;
        tree_context.tree_transform_structure_overwrite_into_raw(
            lhs_transform.transform_structure.as_ref(),
            &lhs_dst_structure,
            &lhs_transform.replay_structure,
            lhs_scratch.data_mut(),
            lhs_data,
            D::one(),
        )?;
    }
    if !rhs_borrowed {
        let rhs_dst_structure = std::sync::Arc::clone(rhs_core_space.structure());
        let rhs_scratch = scratch.prepare_rhs(rhs_core_space.clone())?;
        tree_context.tree_transform_structure_overwrite_into_raw(
            rhs_transform.transform_structure.as_ref(),
            &rhs_dst_structure,
            &rhs_transform.replay_structure,
            rhs_scratch.data_mut(),
            rhs_data,
            D::one(),
        )?;
        execute_rhs_contract_twist(
            &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                tree_context.backend().transpose_backend(),
            ),
            rhs_scratch.data_mut(),
            &artifact.rhs_twist,
        )?;
    }

    if artifact.core_dst.is_none() {
        let lhs_core = select_core_source(lhs_borrowed, &lhs_core_space, lhs_data, || {
            CoreSource::from_host_scratch(scratch.lhs())
        });
        let rhs_core = select_core_source(rhs_borrowed, &rhs_core_space, rhs_data, || {
            CoreSource::from_host_scratch(scratch.rhs())
        });
        return artifact.block_plan.execute_raw(
            &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                tree_context.backend().transpose_backend(),
            ),
            &mut super::fusion_block::BackendRank2Gemm {
                backend: contract_backend,
                workspace: contract_workspace,
            },
            fusion_block_workspace,
            dst_structure,
            dst_data,
            lhs_core.structure(),
            lhs_core.data(),
            rhs_core.structure(),
            rhs_core.data(),
            alpha,
            beta,
        );
    }

    let core_dst = artifact
        .core_dst
        .as_ref()
        .expect("non-identity output artifact carries its destination transform");
    let core_dst_space = core_dst.space.clone();
    let core_dst_structure = std::sync::Arc::clone(core_dst_space.structure());
    scratch.prepare_dst(core_dst_space.clone())?;
    {
        let mut execute = |lhs_core: CoreSource<'_, D>,
                           rhs_core: CoreSource<'_, D>,
                           core_dst: &mut DynamicFusionScratch<D>| {
            artifact.block_plan.execute_raw(
                &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                    tree_context.backend().transpose_backend(),
                ),
                &mut super::fusion_block::BackendRank2Gemm {
                    backend: contract_backend,
                    workspace: contract_workspace,
                },
                fusion_block_workspace,
                &core_dst_structure,
                core_dst.data_mut(),
                lhs_core.structure(),
                lhs_core.data(),
                rhs_core.structure(),
                rhs_core.data(),
                alpha,
                D::zero(),
            )
        };
        let (lhs_scratch, rhs_scratch, core_dst) = scratch.optional_sources_dst_mut();
        let lhs_core = select_core_source(lhs_borrowed, &lhs_core_space, lhs_data, || {
            CoreSource::from_host_scratch(
                lhs_scratch.expect("non-borrowed LHS materialized before core contraction"),
            )
        });
        let rhs_core = select_core_source(rhs_borrowed, &rhs_core_space, rhs_data, || {
            CoreSource::from_host_scratch(
                rhs_scratch.expect("non-borrowed RHS materialized before core contraction"),
            )
        });
        execute(lhs_core, rhs_core, core_dst)?;
    }
    tree_context.tree_transform_structure_into_raw(
        core_dst.output_transform_structure.as_ref(),
        dst_structure,
        &core_dst_structure,
        dst_data,
        scratch.dst().data(),
        D::one(),
        beta,
    )
}

/// Storage-aware dynamic core route.
///
/// Scratch allocation origins are explicit: the LHS core scratch comes
/// from LHS storage, the RHS core scratch from RHS storage, and the
/// core destination scratch from destination storage. Structure caches
/// (`DynamicFusionSpaceCache`, fusion-block plans, tree-transform structures)
/// stay placement-neutral. Replay still runs on host slices; this boundary does
/// not imply device execution.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_fusion_dynamic_plan_into_storage_context<
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
    fusion_block_cache: &mut super::resolution::ContractionResolutionCache<RuleKey>,
    fusion_block_workspace: &mut StorageFusionBlockContractWorkspace<
        DLhs::Similar,
        DRhs::Similar,
        DDst::Similar,
    >,
    scratch: &mut StorageDynamicFusionScratchWorkspace<DLhs::Similar, DRhs::Similar, DDst::Similar>,
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
    DDst: HostWritableStorage<D> + SimilarStorage<D>,
    DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    DLhs: HostReadableStorage<D> + SimilarStorage<D>,
    DLhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    DRhs: HostReadableStorage<D> + SimilarStorage<D>,
    DRhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
{
    let lhs_src_space = DynamicFusionMapSpace::from_typed(
        lhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let rhs_src_space = DynamicFusionMapSpace::from_typed(
        rhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let lhs_transform = dynamic_space_cache.get_or_compile_transformed_source(
        tree_context,
        rule,
        &lhs_src_space,
        lhs.structure(),
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
        encoded_layout_primer::<R>,
    )?;
    let lhs_borrowed = source_is_borrowable_core_layout(
        &lhs_src_space,
        lhs.structure(),
        &lhs_transform.space,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    );
    let rhs_transform = dynamic_space_cache.get_or_compile_transformed_source(
        tree_context,
        rule,
        &rhs_src_space,
        rhs.structure(),
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
        encoded_layout_primer::<R>,
    )?;
    let lhs_space = lhs_transform.space.clone();
    let rhs_space = rhs_transform.space.clone();
    let rhs_borrowed = rhs_source_is_borrowable(
        rule,
        &rhs_src_space,
        rhs.structure(),
        &rhs_space,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
        plan.core_axes().as_spec(),
    )?;

    if !lhs_borrowed {
        let lhs_dst_structure = std::sync::Arc::clone(lhs_space.structure());
        let lhs_scratch =
            scratch.prepare_lhs_from_storage(lhs_space.clone(), lhs.storage(), D::zero())?;
        tree_context.tree_transform_structure_overwrite_into_raw(
            lhs_transform.transform_structure.as_ref(),
            &lhs_dst_structure,
            &lhs_transform.replay_structure,
            lhs_scratch.buffer_mut().as_mut_slice(),
            lhs.data(),
            D::one(),
        )?;
    }
    if !rhs_borrowed {
        let rhs_dst_structure = std::sync::Arc::clone(rhs_space.structure());
        let rhs_scratch =
            scratch.prepare_rhs_from_storage(rhs_space.clone(), rhs.storage(), D::zero())?;
        tree_context.tree_transform_structure_overwrite_into_raw(
            rhs_transform.transform_structure.as_ref(),
            &rhs_dst_structure,
            &rhs_transform.replay_structure,
            rhs_scratch.buffer_mut().as_mut_slice(),
            rhs.data(),
            D::one(),
        )?;
        apply_rhs_contract_twist(
            &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                tree_context.backend().transpose_backend(),
            ),
            rule,
            &rhs_space,
            rhs_scratch.buffer_mut().as_mut_slice(),
            plan.core_axes().as_spec().rhs_contracting_axes(),
        )?;
    }

    if plan.output_transform_is_identity() {
        let dst_space = DynamicFusionMapSpace::from_typed(
            dst.fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let block_plan = fusion_block_cache.get_or_compile_core_plan(
            rule,
            &dst_space,
            &lhs_space,
            &rhs_space,
            plan.core_axes().as_spec(),
        )?;
        let lhs_core = select_core_source(lhs_borrowed, &lhs_space, lhs.data(), || {
            let lhs_core = scratch.lhs();
            CoreSource::materialized(lhs_core.space(), lhs_core.buffer().as_slice())
        });
        let rhs_core = select_core_source(rhs_borrowed, &rhs_space, rhs.data(), || {
            let rhs_core = scratch.rhs();
            CoreSource::materialized(rhs_core.space(), rhs_core.buffer().as_slice())
        });
        return block_plan.execute_storage_raw_sources(
            &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                tree_context.backend().transpose_backend(),
            ),
            &mut super::fusion_block::BackendRank2Gemm {
                backend: contract_backend,
                workspace: contract_workspace,
            },
            fusion_block_workspace,
            lhs.storage(),
            rhs.storage(),
            dst,
            lhs_core.structure(),
            lhs_core.data(),
            rhs_core.structure(),
            rhs_core.data(),
            alpha,
            beta,
        );
    }

    let output_dst_space = DynamicFusionMapSpace::from_typed(
        dst.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let core_dst = dynamic_space_cache.get_or_compile_core_dst(
        tree_context,
        rule,
        &lhs_space,
        &rhs_space,
        plan,
        &output_dst_space,
        encoded_layout_primer::<R>,
    )?;
    let core_dst_space = core_dst.space.clone();
    let block_plan = fusion_block_cache.get_or_compile_core_plan(
        rule,
        &core_dst_space,
        &lhs_space,
        &rhs_space,
        plan.core_axes().as_spec(),
    )?;
    let core_dst_structure = std::sync::Arc::clone(core_dst_space.structure());
    scratch.prepare_dst_from_storage(core_dst_space.clone(), dst.storage(), D::zero())?;
    {
        let mut execute =
            |lhs_core: CoreSource<'_, D>,
             rhs_core: CoreSource<'_, D>,
             core_dst: &mut StorageDynamicFusionScratch<DDst::Similar>| {
                block_plan.execute_storage_raw(
                    &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                        tree_context.backend().transpose_backend(),
                    ),
                    &mut super::fusion_block::BackendRank2Gemm {
                        backend: contract_backend,
                        workspace: contract_workspace,
                    },
                    fusion_block_workspace,
                    lhs.storage(),
                    rhs.storage(),
                    dst.storage(),
                    &core_dst_structure,
                    core_dst.buffer_mut().as_mut_slice(),
                    lhs_core.structure(),
                    lhs_core.data(),
                    rhs_core.structure(),
                    rhs_core.data(),
                    alpha,
                    D::zero(),
                )
            };
        let (lhs_scratch, rhs_scratch, core_dst) = scratch.optional_sources_dst_mut();
        let lhs_core = select_core_source(lhs_borrowed, &lhs_space, lhs.data(), || {
            let lhs_scratch =
                lhs_scratch.expect("non-borrowed LHS materialized before core contraction");
            CoreSource::materialized(lhs_scratch.space(), lhs_scratch.buffer().as_slice())
        });
        let rhs_core = select_core_source(rhs_borrowed, &rhs_space, rhs.data(), || {
            let rhs_scratch =
                rhs_scratch.expect("non-borrowed RHS materialized before core contraction");
            CoreSource::materialized(rhs_scratch.space(), rhs_scratch.buffer().as_slice())
        });
        execute(lhs_core, rhs_core, core_dst)?;
    }
    let dst_structure = std::sync::Arc::clone(dst.structure());
    tree_context.tree_transform_structure_into_raw(
        core_dst.output_transform_structure.as_ref(),
        &dst_structure,
        &core_dst_structure,
        dst.data_mut(),
        scratch.dst().buffer().as_slice(),
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
    fusion_block_cache: &mut super::resolution::ContractionResolutionCache<RuleKey>,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    rule: &R,
    plan: &FusionContractPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
    profile: &mut TensorContractFusionProfile,
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
    let start = std::time::Instant::now();
    let lhs_src_space = DynamicFusionMapSpace::from_typed(
        lhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let rhs_src_space = DynamicFusionMapSpace::from_typed(
        rhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let lhs_transform = dynamic_space_cache.get_or_compile_transformed_source(
        tree_context,
        rule,
        &lhs_src_space,
        lhs.structure(),
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
        encoded_layout_primer::<R>,
    )?;
    let lhs_borrowed = source_is_borrowable_core_layout(
        &lhs_src_space,
        lhs.structure(),
        &lhs_transform.space,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    );
    let rhs_transform = dynamic_space_cache.get_or_compile_transformed_source(
        tree_context,
        rule,
        &rhs_src_space,
        rhs.structure(),
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
        encoded_layout_primer::<R>,
    )?;
    let lhs_space = lhs_transform.space.clone();
    let rhs_space = rhs_transform.space.clone();
    let rhs_borrowed = rhs_source_is_borrowable(
        rule,
        &rhs_src_space,
        rhs.structure(),
        &rhs_space,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
        plan.core_axes().as_spec(),
    )?;
    profile.source_space_lookup += start.elapsed();

    if !lhs_borrowed {
        let start = std::time::Instant::now();
        let lhs_dst_structure = std::sync::Arc::clone(lhs_space.structure());
        let lhs_scratch = scratch.prepare_lhs(lhs_space.clone())?;
        profile.lhs_scratch_prepare += start.elapsed();

        let start = std::time::Instant::now();
        tree_context.tree_transform_structure_overwrite_into_raw_profiled(
            lhs_transform.transform_structure.as_ref(),
            &lhs_dst_structure,
            &lhs_transform.replay_structure,
            lhs_scratch.data_mut(),
            lhs.data(),
            D::one(),
            &mut profile.tree_replay,
        )?;
        profile.lhs_transform += start.elapsed();
        profile.lhs_transform_calls += 1;
    }
    if !rhs_borrowed {
        let start = std::time::Instant::now();
        let rhs_dst_structure = std::sync::Arc::clone(rhs_space.structure());
        let rhs_scratch = scratch.prepare_rhs(rhs_space.clone())?;
        profile.rhs_scratch_prepare += start.elapsed();

        let start = std::time::Instant::now();
        tree_context.tree_transform_structure_overwrite_into_raw_profiled(
            rhs_transform.transform_structure.as_ref(),
            &rhs_dst_structure,
            &rhs_transform.replay_structure,
            rhs_scratch.data_mut(),
            rhs.data(),
            D::one(),
            &mut profile.tree_replay,
        )?;
        apply_rhs_contract_twist(
            &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                tree_context.backend().transpose_backend(),
            ),
            rule,
            &rhs_space,
            rhs_scratch.data_mut(),
            plan.core_axes().as_spec().rhs_contracting_axes(),
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
        let block_plan = fusion_block_cache.get_or_compile_core_plan(
            rule,
            &dst_space,
            &lhs_space,
            &rhs_space,
            plan.core_axes().as_spec(),
        )?;
        profile.fusion_block_plan_lookup += start.elapsed();

        let dst_structure = std::sync::Arc::clone(dst.structure());
        let lhs_core = select_core_source(lhs_borrowed, &lhs_space, lhs.data(), || {
            CoreSource::from_host_scratch(scratch.lhs())
        });
        let rhs_core = select_core_source(rhs_borrowed, &rhs_space, rhs.data(), || {
            CoreSource::from_host_scratch(scratch.rhs())
        });
        return block_plan.execute_raw_profiled(
            &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                tree_context.backend().transpose_backend(),
            ),
            &mut super::fusion_block::BackendRank2Gemm {
                backend: contract_backend,
                workspace: contract_workspace,
            },
            fusion_block_workspace,
            &dst_structure,
            dst.data_mut(),
            lhs_core.structure(),
            lhs_core.data(),
            rhs_core.structure(),
            rhs_core.data(),
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
    let core_dst = dynamic_space_cache.get_or_compile_core_dst(
        tree_context,
        rule,
        &lhs_space,
        &rhs_space,
        plan,
        &output_dst_space,
        encoded_layout_primer::<R>,
    )?;
    let core_dst_space = core_dst.space.clone();
    profile.core_dst_space_lookup += start.elapsed();

    let start = std::time::Instant::now();
    let block_plan = fusion_block_cache.get_or_compile_core_plan(
        rule,
        &core_dst_space,
        &lhs_space,
        &rhs_space,
        plan.core_axes().as_spec(),
    )?;
    profile.fusion_block_plan_lookup += start.elapsed();

    let core_dst_structure = std::sync::Arc::clone(core_dst_space.structure());
    let start = std::time::Instant::now();
    scratch.prepare_dst(core_dst_space.clone())?;
    profile.dst_scratch_prepare += start.elapsed();

    {
        let mut execute = |lhs: CoreSource<'_, D>,
                           rhs_core: CoreSource<'_, D>,
                           core_dst: &mut DynamicFusionScratch<D>| {
            block_plan.execute_raw_profiled(
                &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                    tree_context.backend().transpose_backend(),
                ),
                &mut super::fusion_block::BackendRank2Gemm {
                    backend: contract_backend,
                    workspace: contract_workspace,
                },
                fusion_block_workspace,
                &core_dst_structure,
                core_dst.data_mut(),
                lhs.structure(),
                lhs.data(),
                rhs_core.structure(),
                rhs_core.data(),
                alpha,
                D::zero(),
                profile,
            )
        };
        let (lhs_scratch, rhs_scratch, core_dst) = scratch.optional_sources_dst_mut();
        let lhs_core = select_core_source(lhs_borrowed, &lhs_space, lhs.data(), || {
            CoreSource::from_host_scratch(
                lhs_scratch.expect("non-borrowed LHS materialized before core contraction"),
            )
        });
        let rhs_core = select_core_source(rhs_borrowed, &rhs_space, rhs.data(), || {
            CoreSource::from_host_scratch(
                rhs_scratch.expect("non-borrowed RHS materialized before core contraction"),
            )
        });
        execute(lhs_core, rhs_core, core_dst)?;
    }

    let dst_structure = std::sync::Arc::clone(dst.structure());
    let start = std::time::Instant::now();
    tree_context.tree_transform_structure_into_raw_profiled(
        core_dst.output_transform_structure.as_ref(),
        &dst_structure,
        &core_dst_structure,
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
    last_execution_artifact: Option<DynamicTreeExecutionArtifactLastEntry>,
    execution_artifacts:
        FxHashMap<DynamicTreeExecutionArtifactKey, DynamicTreeExecutionArtifactCacheEntry>,
    last_transformed_sources: Vec<DynamicFusionTransformedSourceLastEntry<RuleKey>>,
    fast_transformed_sources: FxHashMap<
        DynamicFusionTransformedSourceFastKey<RuleKey>,
        DynamicFusionTransformedSourceEntry,
    >,
    transformed_sources: FxHashMap<
        DynamicFusionTransformedSourceSpaceKey<RuleKey>,
        DynamicFusionTransformedSourceEntry,
    >,
    lru_order: VecDeque<DynamicFusionSpaceCacheEntryKey<RuleKey>>,
    last_core_dst: Option<DynamicFusionCoreDstLastEntry<RuleKey>>,
    fast_core_dsts: FxHashMap<DynamicFusionCoreDstFastKey<RuleKey>, DynamicFusionCoreDstEntry>,
    core_dsts: FxHashMap<DynamicFusionCoreDstSpaceKey<RuleKey>, DynamicFusionCoreDstEntry>,
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
struct DynamicFusionCoreDstEntry {
    space: Arc<DynamicFusionMapSpace>,
    output_transform_structure: Arc<TreeTransformStructure<f64>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct DynamicTreeExecutionArtifactKey {
    plan: usize,
    dst_structure: usize,
    lhs_structure: usize,
    lhs_storage_structure: Option<usize>,
    rhs_structure: usize,
    rhs_storage_structure: Option<usize>,
}

impl DynamicTreeExecutionArtifactKey {
    fn new(
        plan: &Arc<FusionContractPlan>,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        lhs_storage_space: Option<&DynamicFusionMapSpace>,
        rhs_structure: &Arc<BlockStructure>,
        rhs_storage_space: Option<&DynamicFusionMapSpace>,
    ) -> Self {
        Self {
            plan: Arc::as_ptr(plan) as usize,
            dst_structure: dst_structure.content_id(),
            lhs_structure: lhs_structure.content_id(),
            lhs_storage_structure: lhs_storage_space.map(|space| space.structure().content_id()),
            rhs_structure: rhs_structure.content_id(),
            rhs_storage_structure: rhs_storage_space.map(|space| space.structure().content_id()),
        }
    }
}

#[derive(Clone, Debug)]
struct DynamicTreeExecutionArtifactCacheEntry {
    _plan: Arc<FusionContractPlan>,
    artifact: Arc<DynamicTreeExecutionArtifact>,
}

#[derive(Clone, Debug)]
struct DynamicTreeExecutionArtifactLastEntry {
    key: DynamicTreeExecutionArtifactKey,
    entry: DynamicTreeExecutionArtifactCacheEntry,
}

impl<RuleKey> Default for DynamicFusionSpaceCache<RuleKey> {
    fn default() -> Self {
        Self {
            last_execution_artifact: None,
            execution_artifacts: FxHashMap::default(),
            last_transformed_sources: Vec::new(),
            fast_transformed_sources: FxHashMap::default(),
            transformed_sources: FxHashMap::default(),
            lru_order: VecDeque::new(),
            last_core_dst: None,
            fast_core_dsts: FxHashMap::default(),
            core_dsts: FxHashMap::default(),
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
    operation: TreeTransformOperation,
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
        operation: &TreeTransformOperation,
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
struct DynamicFusionCoreDstLastEntry<RuleKey> {
    key: Option<DynamicFusionCoreDstSpaceKey<RuleKey>>,
    rule: RuleKey,
    lhs: DynamicFusionLastSpaceKey,
    rhs: DynamicFusionLastSpaceKey,
    core_axes: TensorContractSpecOwned,
    core_dst_open_lhs_rank: usize,
    core_dst_open_rhs_rank: usize,
    output_transform: TreeTransformOperation,
    output_dst: DynamicFusionLastSpaceKey,
    entry: DynamicFusionCoreDstEntry,
}

impl<RuleKey> DynamicFusionCoreDstLastEntry<RuleKey>
where
    RuleKey: Eq,
{
    fn matches(
        &self,
        rule: &RuleKey,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        plan: &FusionContractPlan,
        output_dst: &DynamicFusionMapSpace,
    ) -> bool {
        &self.rule == rule
            && self.lhs.matches(lhs)
            && self.rhs.matches(rhs)
            && self.core_axes == *plan.core_axes()
            && self.core_dst_open_lhs_rank == plan.core_dst_open_lhs_rank()
            && self.core_dst_open_rhs_rank == plan.core_dst_open_rhs_rank()
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
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.execution_artifacts.len() + self.transformed_sources.len() + self.core_dsts.len()
    }

    #[inline]
    pub(crate) fn stats(&self) -> DynamicFusionSpaceCacheStats {
        self.stats
    }

    pub(crate) fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.clear_fast_entries();
        if !policy.stores_entries() {
            self.execution_artifacts.clear();
            self.transformed_sources.clear();
            self.lru_order.clear();
            self.core_dsts.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            self.rebuild_lru_order();
            self.enforce_lru_limit(max_entries);
        }
    }

    fn clear_fast_entries(&mut self) {
        self.last_execution_artifact = None;
        self.last_transformed_sources.clear();
        self.fast_transformed_sources.clear();
        self.last_core_dst = None;
        self.fast_core_dsts.clear();
    }

    fn rebuild_lru_order(&mut self) {
        self.lru_order.clear();
        self.lru_order.extend(
            self.execution_artifacts
                .keys()
                .copied()
                .map(DynamicFusionSpaceCacheEntryKey::ExecutionArtifact),
        );
        self.lru_order.extend(
            self.transformed_sources
                .keys()
                .cloned()
                .map(DynamicFusionSpaceCacheEntryKey::TransformedSource),
        );
        self.lru_order.extend(
            self.core_dsts
                .keys()
                .cloned()
                .map(DynamicFusionSpaceCacheEntryKey::CoreDst),
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

    fn touch_core_dst(&mut self, key: &DynamicFusionCoreDstSpaceKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.core_dsts.contains_key(key) {
            touch_lru_key(
                &mut self.lru_order,
                &DynamicFusionSpaceCacheEntryKey::CoreDst(key.clone()),
            );
        }
    }

    fn insert_core_dst(
        &mut self,
        key: DynamicFusionCoreDstSpaceKey<RuleKey>,
        fast_key: DynamicFusionCoreDstFastKey<RuleKey>,
        entry: DynamicFusionCoreDstEntry,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.core_dsts.insert(key.clone(), entry.clone());
        self.fast_core_dsts.insert(fast_key, entry);
        if self.policy.max_entries().is_some() {
            self.touch_core_dst(&key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_lru_limit(max_entries);
        }
    }

    fn enforce_lru_limit(&mut self, max_entries: usize) {
        let mut evicted_execution_artifact = false;
        let mut evicted_transformed_source = false;
        let mut evicted_core_dst = false;
        while self.len() > max_entries {
            let Some(oldest) = self.lru_order.pop_front() else {
                break;
            };
            match oldest {
                DynamicFusionSpaceCacheEntryKey::ExecutionArtifact(key) => {
                    evicted_execution_artifact |= self.execution_artifacts.remove(&key).is_some();
                }
                DynamicFusionSpaceCacheEntryKey::TransformedSource(key) => {
                    evicted_transformed_source |= self.transformed_sources.remove(&key).is_some();
                }
                DynamicFusionSpaceCacheEntryKey::CoreDst(key) => {
                    evicted_core_dst |= self.core_dsts.remove(&key).is_some();
                }
            }
        }
        if evicted_execution_artifact {
            self.last_execution_artifact = None;
        }
        if evicted_transformed_source {
            self.last_transformed_sources.clear();
            self.fast_transformed_sources.clear();
        }
        if evicted_core_dst {
            self.last_core_dst = None;
            self.fast_core_dsts.clear();
        }
    }

    pub(crate) fn get_execution_artifact(
        &mut self,
        plan: &Arc<FusionContractPlan>,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        lhs_storage_space: Option<&DynamicFusionMapSpace>,
        rhs_structure: &Arc<BlockStructure>,
        rhs_storage_space: Option<&DynamicFusionMapSpace>,
    ) -> Option<Arc<DynamicTreeExecutionArtifact>> {
        if !self.policy.stores_entries() {
            return None;
        }
        let key = DynamicTreeExecutionArtifactKey::new(
            plan,
            dst_structure,
            lhs_structure,
            lhs_storage_space,
            rhs_structure,
            rhs_storage_space,
        );
        if let Some(last) = &self.last_execution_artifact {
            if last.key == key {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                return Some(Arc::clone(&last.entry.artifact));
            }
        }
        let entry = self.execution_artifacts.get(&key)?.clone();
        self.stats.hits += 1;
        if self.policy.max_entries().is_some() {
            touch_lru_key(
                &mut self.lru_order,
                &DynamicFusionSpaceCacheEntryKey::ExecutionArtifact(key),
            );
        }
        self.last_execution_artifact = Some(DynamicTreeExecutionArtifactLastEntry {
            key,
            entry: entry.clone(),
        });
        Some(entry.artifact)
    }

    pub(crate) fn insert_execution_artifact(
        &mut self,
        plan: Arc<FusionContractPlan>,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        lhs_storage_space: Option<&DynamicFusionMapSpace>,
        rhs_structure: &Arc<BlockStructure>,
        rhs_storage_space: Option<&DynamicFusionMapSpace>,
        artifact: Arc<DynamicTreeExecutionArtifact>,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        let key = DynamicTreeExecutionArtifactKey::new(
            &plan,
            dst_structure,
            lhs_structure,
            lhs_storage_space,
            rhs_structure,
            rhs_storage_space,
        );
        let entry = DynamicTreeExecutionArtifactCacheEntry {
            _plan: plan,
            artifact,
        };
        self.execution_artifacts.insert(key, entry.clone());
        self.last_execution_artifact = Some(DynamicTreeExecutionArtifactLastEntry { key, entry });
        if self.policy.max_entries().is_some() {
            touch_lru_key(
                &mut self.lru_order,
                &DynamicFusionSpaceCacheEntryKey::ExecutionArtifact(key),
            );
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_lru_limit(max_entries);
        }
    }

    fn get_or_compile_transformed_source<R, D, BT>(
        &mut self,
        tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        rule: &R,
        src_space: &DynamicFusionMapSpace,
        src_storage_structure: &Arc<BlockStructure>,
        operation: &TreeTransformOperation,
        source_conjugate: bool,
        layout_primer: LayoutKeyBuilder<R>,
    ) -> Result<DynamicFusionTransformedSourceEntry, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar,
        BT: TreeTransformBackend<D, f64>,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        let nout = if source_conjugate {
            src_space.nin()
        } else {
            src_space.nout()
        };
        if self.policy.stores_entries() && !source_conjugate {
            let refresh_lru = self.policy.max_entries().is_some();
            let homspace = src_space.homspace();
            let replay_structure = src_storage_structure;
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
            let adjoint = src_space.adjoint_view()?;
            (
                adjoint.homspace().clone(),
                std::sync::Arc::clone(adjoint.structure()),
            )
        } else {
            (
                src_space.homspace().clone(),
                std::sync::Arc::clone(src_storage_structure),
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
                src_space
                    .adjoint_view()?
                    .transformed_with_primer(rule, operation, layout_primer)?
            } else {
                src_space.transformed_with_primer(rule, operation, layout_primer)?
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
            replay_structure_id: replay_structure.content_id(),
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
            src_space
                .adjoint_view()?
                .transformed_with_primer(rule, operation, layout_primer)?
        } else {
            src_space.transformed_with_primer(rule, operation, layout_primer)?
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

    fn get_or_compile_transformed_source_prelowered<R, D, BT>(
        &mut self,
        tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        rule: &R,
        logical_space: &DynamicFusionMapSpace,
        storage_space: &DynamicFusionMapSpace,
        storage_structure: &Arc<BlockStructure>,
        operation: &TreeTransformOperation,
        storage_conjugate: bool,
        layout_primer: LayoutKeyBuilder<R>,
    ) -> Result<DynamicFusionTransformedSourceEntry, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar,
        BT: TreeTransformBackend<D, f64>,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        let nout = logical_space.nout();
        let homspace = logical_space.homspace().clone();
        let replay_structure = Arc::clone(storage_structure);
        let fast_key = DynamicFusionTransformedSourceFastKey {
            rule: rule_key.clone(),
            nout,
            homspace: homspace.clone(),
            replay_structure_id: replay_structure.content_id(),
            operation: operation.clone(),
            source_conjugate: storage_conjugate,
        };
        let key = DynamicFusionTransformedSourceSpaceKey {
            rule: rule_key.clone(),
            nout,
            homspace: homspace.clone(),
            structure: BlockStructureCacheKey::from_structure(&replay_structure)?,
            operation: operation.clone(),
            source_conjugate: storage_conjugate,
        };
        if self.policy.stores_entries() {
            if let Some(entry) = self.fast_transformed_sources.get(&fast_key).cloned() {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                self.touch_transformed_source(&key);
                return Ok(entry);
            }
            if let Some(entry) = self.transformed_sources.get(&key).cloned() {
                self.stats.hits += 1;
                self.touch_transformed_source(&key);
                self.fast_transformed_sources
                    .insert(fast_key, entry.clone());
                return Ok(entry);
            }
        }

        self.stats.misses += 1;
        let space = logical_space.transformed_with_primer(rule, operation, layout_primer)?;
        let dst_structure = Arc::clone(space.structure());
        let transform_structure = tree_context.get_or_compile_tree_pair_structure_prelowered(
            rule,
            operation,
            &dst_structure,
            logical_space.structure(),
            storage_structure,
            storage_conjugate,
            prelowered_storage_block_index(logical_space, storage_space, storage_conjugate),
            prelowered_storage_axis(logical_space, storage_space, storage_conjugate),
        )?;
        let entry = DynamicFusionTransformedSourceEntry {
            space: Arc::new(space),
            replay_structure,
            transform_structure,
        };
        if self.policy.stores_entries() {
            self.insert_transformed_source(key, fast_key, entry.clone());
        }
        Ok(entry)
    }

    fn get_or_compile_core_dst<R, D, BT>(
        &mut self,
        tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        rule: &R,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        plan: &FusionContractPlan,
        output_dst: &DynamicFusionMapSpace,
        layout_primer: LayoutKeyBuilder<R>,
    ) -> Result<DynamicFusionCoreDstEntry, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar,
        BT: TreeTransformBackend<D, f64>,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        if !self.policy.stores_entries() {
            self.stats.misses += 1;
            let space =
                DynamicFusionMapSpace::core_dst_with_primer(rule, lhs, rhs, plan, layout_primer)?;
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
            return Ok(DynamicFusionCoreDstEntry {
                space: Arc::new(space),
                output_transform_structure,
            });
        }
        if let Some(last) = &self.last_core_dst {
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
                    self.touch_core_dst(key);
                }
                return Ok(entry);
            }
        }
        let fast_key = DynamicFusionCoreDstFastKey {
            rule: rule_key.clone(),
            lhs: DynamicFusionFastSpaceKey::from_space(lhs),
            rhs: DynamicFusionFastSpaceKey::from_space(rhs),
            core_axes: plan.core_axes().clone(),
            core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
            core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
            output_transform: plan.output_transform().clone(),
            output_dst: DynamicFusionFastSpaceKey::from_space(output_dst),
        };
        let lru_key = if self.policy.max_entries().is_some() {
            Some(DynamicFusionCoreDstSpaceKey {
                rule: rule_key.clone(),
                lhs: DynamicFusionSpaceKey::from_space(lhs)?,
                rhs: DynamicFusionSpaceKey::from_space(rhs)?,
                core_axes: plan.core_axes().clone(),
                core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
                core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionSpaceKey::from_space(output_dst)?,
            })
        } else {
            None
        };
        if let Some(entry) = self.fast_core_dsts.get(&fast_key) {
            let entry = entry.clone();
            self.stats.hits += 1;
            self.stats.fast_hits += 1;
            if let Some(key) = lru_key.as_ref() {
                self.touch_core_dst(key);
            }
            self.last_core_dst = Some(DynamicFusionCoreDstLastEntry {
                key: lru_key,
                rule: rule_key,
                lhs: DynamicFusionLastSpaceKey::from_space(lhs),
                rhs: DynamicFusionLastSpaceKey::from_space(rhs),
                core_axes: plan.core_axes().clone(),
                core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
                core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionLastSpaceKey::from_space(output_dst),
                entry: entry.clone(),
            });
            return Ok(entry);
        }
        let key = if let Some(key) = lru_key {
            key
        } else {
            DynamicFusionCoreDstSpaceKey {
                rule: rule_key.clone(),
                lhs: DynamicFusionSpaceKey::from_space(lhs)?,
                rhs: DynamicFusionSpaceKey::from_space(rhs)?,
                core_axes: plan.core_axes().clone(),
                core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
                core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionSpaceKey::from_space(output_dst)?,
            }
        };
        if let Some(entry) = self.core_dsts.get(&key) {
            let entry = entry.clone();
            self.stats.hits += 1;
            self.touch_core_dst(&key);
            self.fast_core_dsts.insert(fast_key, entry.clone());
            self.last_core_dst = Some(DynamicFusionCoreDstLastEntry {
                key: Some(key.clone()),
                rule: rule_key,
                lhs: DynamicFusionLastSpaceKey::from_space(lhs),
                rhs: DynamicFusionLastSpaceKey::from_space(rhs),
                core_axes: plan.core_axes().clone(),
                core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
                core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionLastSpaceKey::from_space(output_dst),
                entry: entry.clone(),
            });
            return Ok(entry);
        }

        self.stats.misses += 1;
        let space =
            DynamicFusionMapSpace::core_dst_with_primer(rule, lhs, rhs, plan, layout_primer)?;
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
        let entry = DynamicFusionCoreDstEntry {
            space: Arc::new(space),
            output_transform_structure,
        };
        let last_key = key.clone();
        self.insert_core_dst(key, fast_key, entry.clone());
        self.last_core_dst = Some(DynamicFusionCoreDstLastEntry {
            key: Some(last_key),
            rule: rule_key,
            lhs: DynamicFusionLastSpaceKey::from_space(lhs),
            rhs: DynamicFusionLastSpaceKey::from_space(rhs),
            core_axes: plan.core_axes().clone(),
            core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
            core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
            output_transform: plan.output_transform().clone(),
            output_dst: DynamicFusionLastSpaceKey::from_space(output_dst),
            entry: entry.clone(),
        });
        Ok(entry)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DynamicFusionSpaceCacheEntryKey<RuleKey> {
    ExecutionArtifact(DynamicTreeExecutionArtifactKey),
    TransformedSource(DynamicFusionTransformedSourceSpaceKey<RuleKey>),
    CoreDst(DynamicFusionCoreDstSpaceKey<RuleKey>),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionTransformedSourceFastKey<RuleKey> {
    rule: RuleKey,
    nout: usize,
    homspace: FusionTreeHomSpace,
    replay_structure_id: usize,
    operation: TreeTransformOperation,
    source_conjugate: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionTransformedSourceSpaceKey<RuleKey> {
    rule: RuleKey,
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure: BlockStructureCacheKey,
    operation: TreeTransformOperation,
    source_conjugate: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionCoreDstFastKey<RuleKey> {
    rule: RuleKey,
    lhs: DynamicFusionFastSpaceKey,
    rhs: DynamicFusionFastSpaceKey,
    core_axes: TensorContractSpecOwned,
    core_dst_open_lhs_rank: usize,
    core_dst_open_rhs_rank: usize,
    output_transform: TreeTransformOperation,
    output_dst: DynamicFusionFastSpaceKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionCoreDstSpaceKey<RuleKey> {
    rule: RuleKey,
    lhs: DynamicFusionSpaceKey,
    rhs: DynamicFusionSpaceKey,
    core_axes: TensorContractSpecOwned,
    core_dst_open_lhs_rank: usize,
    core_dst_open_rhs_rank: usize,
    output_transform: TreeTransformOperation,
    output_dst: DynamicFusionSpaceKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionFastSpaceKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure_id: usize,
}

impl DynamicFusionFastSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure_id: space.structure().content_id(),
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
    operation: &TreeTransformOperation,
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

#[derive(Clone, Debug)]
struct RhsTwistAction {
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    factor: f64,
}

fn compile_rhs_contract_twist<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    rhs_contracting_axes: &[usize],
) -> Result<Arc<[RhsTwistAction]>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if rule.braiding_style() != tenet_core::BraidingStyleKind::Fermionic {
        return Ok(Arc::from([]));
    }
    let structure = std::sync::Arc::clone(space.structure());
    let mut actions = Vec::new();
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let tenet_core::BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let factor = super::fusion::rhs_contract_twist_factor(
            rule,
            space.homspace(),
            rhs_contracting_axes,
            key.codomain_tree(),
        )?;
        if factor != 1.0 {
            actions.push(RhsTwistAction {
                shape: block.shape().to_vec(),
                strides: tenet_operations::strided::strides_to_isize(block.strides())?,
                offset: tenet_operations::strided::offset_to_isize(block.offset())?,
                factor,
            });
        }
    }
    Ok(actions.into())
}

fn apply_rhs_contract_twist<A, R, D>(
    kernels: &mut A,
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    rhs_contracting_axes: &[usize],
) -> Result<(), OperationError>
where
    A: crate::HostKernelAdapter<D>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let actions = compile_rhs_contract_twist(rule, space, rhs_contracting_axes)?;
    execute_rhs_contract_twist(kernels, data, &actions)
}

/// Applies the fermionic supertrace actions compiled with the tree artifact.
/// Why not retain the rule here: numerical replay must not re-enter categorical
/// coefficient evaluation or rebuild strided descriptors.
fn execute_rhs_contract_twist<A, D>(
    kernels: &mut A,
    data: &mut [D],
    actions: &[RhsTwistAction],
) -> Result<(), OperationError>
where
    A: crate::HostKernelAdapter<D>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    for action in actions {
        kernels.scale_strided(
            data,
            &action.shape,
            &action.strides,
            action.offset,
            D::coefficient_as_data(action.factor),
        )?;
    }
    Ok(())
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
    operation: TreeTransformOperation,
    dst: &mut DynamicFusionScratch<D>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    src_replay_structure: &std::sync::Arc<BlockStructure>,
    source_conjugate: bool,
    alpha: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DSrc: HostReadableStorage<D>,
{
    let plan = build_tree_pair_transform_group_plan(rule, operation, src_replay_structure)?;
    let structure = plan.compile_structures_with_storage_conjugation(
        dst.space().structure(),
        src_replay_structure,
        source_conjugate,
    )?;
    let dst_structure = std::sync::Arc::clone(dst.space().structure());
    tree_backend.tree_transform_structure_overwrite_into_raw(
        tree_workspace,
        &structure,
        &dst_structure,
        src_replay_structure,
        dst.data_mut(),
        src.data(),
        alpha,
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
    operation: TreeTransformOperation,
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
fn tensorcontract_dynamic_core_into_raw<B, R, D>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_structure: &std::sync::Arc<BlockStructure>,
    dst_data: &mut [D],
    lhs: CoreSource<'_, D>,
    rhs: CoreSource<'_, D>,
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let _ = dst_structure;
    tensorcontract_core_fusion_blocks_into_raw(
        &mut crate::StridedHostKernelAdapter::with_transpose_backend(backend.transpose_backend()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use tenet_core::{
        BlockKey, BlockSpec, FusionProductSpace, FusionTensorMapSpace, Placement, SU2FusionRule,
        SectorId, SectorLeg, TensorMapSpace, Trivial, Z2FusionRule,
    };

    use crate::storage_scratch::StorageFusionBlockContractWorkspace;
    use crate::tree_context::TreeTransformExecutionContext;
    use crate::{DenseTreeTransformOperations, TensorContractWorkspace};
    use tenet_operations::OutputAxisOrder;

    use super::super::dynamic_space::lowered_layout_primer;
    use super::super::fusion_block::FusionBlockContractWorkspace;
    use super::super::scratch::StorageDynamicFusionScratchWorkspace;

    thread_local! {
        static EXECUTION_PRIMER_CALLS: Cell<usize> = const { Cell::new(0) };
    }

    fn counting_su2_primer(
        rule: &SU2FusionRule,
        homspace: &FusionTreeHomSpace,
    ) -> Result<Option<Arc<[tenet_core::FusionTreeBlockKey]>>, OperationError> {
        EXECUTION_PRIMER_CALLS.with(|calls| calls.set(calls.get() + 1));
        lowered_layout_primer(rule, homspace)
    }

    fn reset_execution_primer_calls() {
        EXECUTION_PRIMER_CALLS.with(|calls| calls.set(0));
    }

    fn execution_primer_calls() -> usize {
        EXECUTION_PRIMER_CALLS.with(Cell::get)
    }

    fn one_block_structure() -> Arc<BlockStructure> {
        Arc::new(
            BlockStructure::from_blocks_with_rank(
                1,
                vec![
                    BlockSpec::column_major_with_key(BlockKey::sector_ids([0]), vec![2], 0)
                        .unwrap(),
                ],
            )
            .unwrap(),
        )
    }

    #[test]
    fn execution_layout_primer_runs_only_after_dynamic_space_cache_misses() {
        // What: transformed-source and nonidentity-output core spaces invoke
        // the selected primer on a cold miss, while a task-local replay hit
        // returns the shared entry without invoking it again.
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rule = SU2FusionRule;
        let homspace = FusionTreeHomSpace::from_sector_ids([(7, 3); 4], []);
        let key_count = homspace.fusion_tree_keys(&rule).len();
        let source = DynamicFusionMapSpace::from_degeneracy_shapes(
            &rule,
            homspace,
            vec![vec![3; 4]; key_count],
        )
        .unwrap();
        let scalar_homspace = FusionTreeHomSpace::from_sector_ids([], []);
        let scalar =
            DynamicFusionMapSpace::from_degeneracy_shapes(&rule, scalar_homspace, [vec![]])
                .unwrap();
        let operation = TreeTransformOperation::permute([0, 2, 1, 3], []);
        let axes = TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[0, 2, 1, 3]));
        let provider = Arc::new(rule);
        let source_bound =
            super::super::dynamic_space::BoundDynamicFusionMapSpace::bind_multiplicity_free(
                source.clone(),
                Arc::clone(&provider),
            )
            .unwrap();
        let scalar_bound =
            super::super::dynamic_space::BoundDynamicFusionMapSpace::bind_multiplicity_free(
                scalar.clone(),
                Arc::clone(&provider),
            )
            .unwrap();
        let output_bound = super::super::dynamic_space::BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            &source_bound,
            &scalar_bound,
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            axes.output_permutation(),
        )
        .unwrap();
        let output = output_bound.space().clone();
        let plan = super::super::fusion::prepare_tensorcontract_fusion_plan_dyn_raw(
            &rule, &output, &source, &scalar, axes,
        )
        .unwrap();
        assert!(!plan.output_transform_is_identity());

        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let mut tree_context =
            TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
        let mut cache = DynamicFusionSpaceCache::default();
        reset_execution_primer_calls();
        let cold_transform = cache
            .get_or_compile_transformed_source::<_, f64, _>(
                &mut tree_context,
                &rule,
                &source,
                source.structure(),
                &operation,
                false,
                counting_su2_primer,
            )
            .unwrap();
        assert_eq!(execution_primer_calls(), 1);
        let warm_transform = cache
            .get_or_compile_transformed_source::<_, f64, _>(
                &mut tree_context,
                &rule,
                &source,
                source.structure(),
                &operation,
                false,
                counting_su2_primer,
            )
            .unwrap();
        assert_eq!(execution_primer_calls(), 1);
        assert_eq!(cold_transform.space, warm_transform.space);

        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        reset_execution_primer_calls();
        let cold_core = cache
            .get_or_compile_core_dst::<_, f64, _>(
                &mut tree_context,
                &rule,
                &source,
                &scalar,
                &plan,
                &output,
                counting_su2_primer,
            )
            .unwrap();
        assert_eq!(execution_primer_calls(), 1);
        let warm_core = cache
            .get_or_compile_core_dst::<_, f64, _>(
                &mut tree_context,
                &rule,
                &source,
                &scalar,
                &plan,
                &output,
                counting_su2_primer,
            )
            .unwrap();
        assert_eq!(execution_primer_calls(), 1);
        assert_eq!(cold_core.space, warm_core.space);
        assert!(cache.stats().hits() >= 2);

        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let mut no_cache = DynamicFusionSpaceCache::default();
        no_cache.set_policy(OperationCachePolicy::NoCache);
        reset_execution_primer_calls();
        no_cache
            .get_or_compile_transformed_source::<_, f64, _>(
                &mut tree_context,
                &rule,
                &source,
                source.structure(),
                &TreeTransformOperation::permute([3, 1, 2, 0], []),
                false,
                counting_su2_primer,
            )
            .unwrap();
        assert_eq!(execution_primer_calls(), 1);
        assert_eq!(no_cache.len(), 0);
        assert_eq!(no_cache.stats().hits(), 0);

        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        reset_execution_primer_calls();
        let counted_source = source_bound
            .clone()
            .with_test_layout_primer(counting_su2_primer);
        let mut context = crate::TensorContractFusionExecutionContext::<
            f64,
            crate::TreeTransformBuiltinRuleCacheKey,
        >::default();
        let mut dst_data = vec![0.0; output_bound.space().required_len().unwrap()];
        let lhs_data = vec![0.0; counted_source.space().required_len().unwrap()];
        let rhs_data = vec![0.0; scalar_bound.space().required_len().unwrap()];
        context
            .tensorcontract_fusion_dyn_into(
                &output_bound,
                &mut dst_data,
                &counted_source,
                &lhs_data,
                &scalar_bound,
                &rhs_data,
                axes,
                1.0,
                0.0,
            )
            .unwrap();
        let cold_calls = execution_primer_calls();
        assert!(cold_calls > 0);
        context
            .tensorcontract_fusion_dyn_into(
                &output_bound,
                &mut dst_data,
                &counted_source,
                &lhs_data,
                &scalar_bound,
                &rhs_data,
                axes,
                1.0,
                0.0,
            )
            .unwrap();
        assert_eq!(execution_primer_calls(), cold_calls);

        let mut no_cache_context = crate::TensorContractFusionExecutionContext::<
            f64,
            crate::TreeTransformBuiltinRuleCacheKey,
        >::default();
        no_cache_context.set_cache_policy(OperationCachePolicy::NoCache);
        reset_execution_primer_calls();
        for expected_minimum in 1..=2 {
            no_cache_context
                .tensorcontract_fusion_dyn_into(
                    &output_bound,
                    &mut dst_data,
                    &counted_source,
                    &lhs_data,
                    &scalar_bound,
                    &rhs_data,
                    axes,
                    1.0,
                    0.0,
                )
                .unwrap();
            assert!(execution_primer_calls() >= expected_minimum);
            assert_eq!(no_cache_context.dynamic_fusion_space_cache_len(), 0);
        }

        let other_axes =
            TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[1, 0, 2, 3]));
        let other_output = super::super::dynamic_space::BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            &source_bound,
            &scalar_bound,
            other_axes.lhs_contracting_axes(),
            other_axes.rhs_contracting_axes(),
            other_axes.output_permutation(),
        )
        .unwrap();
        let mut other_dst_data = vec![0.0; other_output.space().required_len().unwrap()];
        let mut lru_context = crate::TensorContractFusionExecutionContext::<
            f64,
            crate::TreeTransformBuiltinRuleCacheKey,
        >::default();
        lru_context.set_cache_policy(OperationCachePolicy::task_local_lru(1));
        reset_execution_primer_calls();
        macro_rules! execute_lru {
            ($output:expr, $data:expr, $axes:expr) => {{
                lru_context
                    .tensorcontract_fusion_dyn_into(
                        $output,
                        $data,
                        &counted_source,
                        &lhs_data,
                        &scalar_bound,
                        &rhs_data,
                        $axes,
                        1.0,
                        0.0,
                    )
                    .unwrap();
                assert!(lru_context.dynamic_fusion_space_cache_len() <= 1);
            }};
        }
        execute_lru!(&output_bound, &mut dst_data, axes);
        execute_lru!(&other_output, &mut other_dst_data, other_axes);
        execute_lru!(&output_bound, &mut dst_data, axes);
        assert!(execution_primer_calls() >= 3);
    }

    #[test]
    fn dynamic_fusion_fast_space_key_uses_structure_content_identity() {
        // What: held across both builds so a concurrent `reset_global_operation_caches`
        // cannot evict `first_structure` and hand `second_structure` a fresh id.
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let first_structure = one_block_structure();
        let second_structure = one_block_structure();
        assert!(!Arc::ptr_eq(&first_structure, &second_structure));
        assert_eq!(first_structure.content_id(), second_structure.content_id());

        let homspace = Arc::new(FusionTreeHomSpace::from_sector_ids([(0, 2)], []));
        let first = DynamicFusionFastSpaceKey {
            nout: 1,
            homspace: homspace.as_ref().clone(),
            structure_id: first_structure.content_id(),
        };
        let second = DynamicFusionFastSpaceKey {
            nout: 1,
            homspace: homspace.as_ref().clone(),
            structure_id: second_structure.content_id(),
        };

        assert_eq!(first, second);

        let operation = TreeTransformOperation::permute([0], []);
        let first_transform = DynamicFusionTransformedSourceFastKey::<&'static str> {
            rule: "test",
            nout: 1,
            homspace: homspace.as_ref().clone(),
            replay_structure_id: first_structure.content_id(),
            operation: operation.clone(),
            source_conjugate: false,
        };
        let second_transform = DynamicFusionTransformedSourceFastKey::<&'static str> {
            rule: "test",
            nout: 1,
            homspace: homspace.as_ref().clone(),
            replay_structure_id: second_structure.content_id(),
            operation,
            source_conjugate: false,
        };
        assert_eq!(first_transform, second_transform);
    }

    #[test]
    fn borrowable_core_layout_accepts_equal_structure_across_intern_reset() {
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rule = Z2FusionRule;
        let build = || {
            FusionTensorMapSpace::from_degeneracy_shapes(
                TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
                FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]),
                &rule,
                [vec![1, 1]],
            )
            .unwrap()
            .try_bind_rule(&rule)
            .unwrap()
        };
        let source = DynamicFusionMapSpace::from_typed(&build());
        let source_structure = Arc::clone(source.structure());

        crate::cache::reset_global_operation_caches();
        let core = DynamicFusionMapSpace::from_typed(&build());

        // What: equal live layouts created on opposite sides of an intern reset
        // remain borrowable even though their process-local content ids differ.
        assert_ne!(source_structure.content_id(), core.structure().content_id());
        assert!(source_is_borrowable_core_layout(
            &source,
            &source_structure,
            &core,
            &TreeTransformOperation::permute([0], [1]),
            false,
        ));
    }

    #[test]
    fn borrowable_core_layout_defers_homspace_identity_until_after_cheap_gates() {
        // What: nonidentity and conjugating sources skip HomSpace identity,
        // while an otherwise borrowable identity layout performs one comparison.
        let rule = Z2FusionRule;
        let typed = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]),
            &rule,
            [vec![1, 1]],
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        let source = DynamicFusionMapSpace::from_typed(&typed);
        let source_structure = Arc::clone(source.structure());

        reset_source_layout_homspace_id_comparisons();
        assert!(!source_is_borrowable_core_layout(
            &source,
            &source_structure,
            &source,
            &TreeTransformOperation::permute([1], [0]),
            false,
        ));
        assert_eq!(source_layout_homspace_id_comparisons(), 0);

        reset_source_layout_homspace_id_comparisons();
        assert!(!source_is_borrowable_core_layout(
            &source,
            &source_structure,
            &source,
            &TreeTransformOperation::permute([0], [1]),
            true,
        ));
        assert_eq!(source_layout_homspace_id_comparisons(), 0);

        reset_source_layout_homspace_id_comparisons();
        assert!(source_is_borrowable_core_layout(
            &source,
            &source_structure,
            &source,
            &TreeTransformOperation::permute([0], [1]),
            false,
        ));
        assert_eq!(source_layout_homspace_id_comparisons(), 1);
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct ScratchAllocation {
        label: &'static str,
        len: usize,
    }

    #[derive(Clone, Debug)]
    struct TrackingStorage<T> {
        data: Vec<T>,
        label: &'static str,
        allocations: Rc<RefCell<Vec<ScratchAllocation>>>,
    }

    #[derive(Clone, Debug)]
    struct TrackingScratch<T> {
        data: Vec<T>,
    }

    impl<T> TrackingStorage<T> {
        fn new(
            data: Vec<T>,
            label: &'static str,
            allocations: Rc<RefCell<Vec<ScratchAllocation>>>,
        ) -> Self {
            Self {
                data,
                label,
                allocations,
            }
        }
    }

    impl<T> TensorStorage<T> for TrackingStorage<T> {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn placement(&self) -> Placement {
            Placement::Host
        }
    }

    impl<T> tenet_core::HostReadableStorage<T> for TrackingStorage<T> {
        fn as_slice(&self) -> &[T] {
            &self.data
        }
    }

    impl<T> tenet_core::HostWritableStorage<T> for TrackingStorage<T> {
        fn as_mut_slice(&mut self) -> &mut [T] {
            &mut self.data
        }
    }

    impl<T: Clone> SimilarStorage<T> for TrackingStorage<T> {
        type Similar = TrackingScratch<T>;

        fn similar_filled(&self, len: usize, value: T) -> Self::Similar
        where
            T: Clone,
        {
            self.allocations.borrow_mut().push(ScratchAllocation {
                label: self.label,
                len,
            });
            TrackingScratch {
                data: vec![value; len],
            }
        }
    }

    impl<T> TensorStorage<T> for TrackingScratch<T> {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn placement(&self) -> Placement {
            Placement::Host
        }
    }

    impl<T> tenet_core::HostReadableStorage<T> for TrackingScratch<T> {
        fn as_slice(&self) -> &[T] {
            &self.data
        }
    }

    impl<T> tenet_core::HostWritableStorage<T> for TrackingScratch<T> {
        fn as_mut_slice(&mut self) -> &mut [T] {
            &mut self.data
        }
    }

    impl<T: Clone> tenet_core::ScratchStorage<T> for TrackingScratch<T> {
        fn reset_filled(&mut self, len: usize, value: T)
        where
            T: Clone,
        {
            self.data.clear();
            self.data.resize(len, value);
        }
    }

    #[test]
    fn dynamic_storage_context_identity_output_borrows_both_sources() {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
        let fusion_space = || {
            FusionTensorMapSpace::from_degeneracy_shapes(
                TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([leg()]),
                    FusionProductSpace::new([leg()]),
                ),
                &rule,
                [vec![1, 1], vec![1, 1]],
            )
            .unwrap()
            .try_bind_rule(&rule)
            .unwrap()
        };
        let allocations = Rc::new(RefCell::new(Vec::new()));
        let lhs =
            TensorMap::<f64, 1, 1, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![2.0, 3.0], "lhs", allocations.clone()),
                fusion_space(),
            )
            .unwrap();
        let rhs =
            TensorMap::<f64, 1, 1, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![5.0, 7.0], "rhs", allocations.clone()),
                fusion_space(),
            )
            .unwrap();
        let mut dst =
            TensorMap::<f64, 1, 1, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![10.0, 20.0], "destination", allocations.clone()),
                fusion_space(),
            )
            .unwrap();
        let plan = prepare_tensorcontract_fusion_plan(
            &rule,
            dst.fusion_space().unwrap(),
            lhs.fusion_space().unwrap(),
            rhs.fusion_space().unwrap(),
            TensorContractSpec::with_default_output_order(&[1], &[0]),
        )
        .unwrap();
        assert!(plan.output_transform_is_identity());

        let mut expected_dst =
            TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], fusion_space())
                .unwrap();
        let expected_lhs =
            TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0, 3.0], fusion_space())
                .unwrap();
        let expected_rhs =
            TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![5.0, 7.0], fusion_space())
                .unwrap();
        run_host_reference(
            &rule,
            &plan,
            &mut expected_dst,
            &expected_lhs,
            &expected_rhs,
            2.0,
            3.0,
        );

        let mut tree_context =
            TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
        let mut contract_backend = DenseTreeTransformOperations::default();
        let mut contract_workspace = TensorContractWorkspace::default();
        let mut dynamic_space_cache = DynamicFusionSpaceCache::default();
        let mut fusion_block_cache =
            super::super::resolution::ContractionResolutionCache::default();
        let mut fusion_block_workspace = StorageFusionBlockContractWorkspace::<
            TrackingScratch<f64>,
            TrackingScratch<f64>,
            TrackingScratch<f64>,
        >::default();
        let mut scratch = StorageDynamicFusionScratchWorkspace::<
            TrackingScratch<f64>,
            TrackingScratch<f64>,
            TrackingScratch<f64>,
        >::default();
        let lhs_before = lhs.data().to_vec();
        let rhs_before = rhs.data().to_vec();

        for _ in 0..2 {
            dst.data_mut().copy_from_slice(&[10.0, 20.0]);
            tensorcontract_fusion_dynamic_plan_into_storage_context(
                &mut tree_context,
                &mut contract_backend,
                &mut contract_workspace,
                &mut dynamic_space_cache,
                &mut fusion_block_cache,
                &mut fusion_block_workspace,
                &mut scratch,
                &rule,
                &plan,
                &mut dst,
                &lhs,
                &rhs,
                2.0,
                3.0,
            )
            .unwrap();

            assert_eq!(dst.data(), expected_dst.data());
            // What: borrowing already-core sources never mutates either input.
            assert_eq!(lhs.data(), lhs_before);
            assert_eq!(rhs.data(), rhs_before);
        }
        let allocations = allocations.borrow();
        // What: identity no-twist sources need no transform allocation, and
        // the core contraction GEMMs directly without pack/scatter allocation.
        assert_eq!(allocations.as_slice(), []);
    }

    #[test]
    fn dynamic_storage_context_output_transform_allocates_core_dst_from_destination_storage() {
        let rule = SU2FusionRule;
        let lhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1), (1, 1)], []);
        let lhs_keys = lhs_hom.fusion_tree_keys(&rule);
        assert_eq!(lhs_keys.len(), 2);
        let src_tree = lhs_keys
            .iter()
            .find(|key| key.codomain_tree().innerlines() == [SectorId::new(0), SectorId::new(1)])
            .expect("SU2 fixture should contain the reference source tree")
            .clone();
        let recoupled_tree = lhs_keys
            .iter()
            .find(|key| **key != src_tree)
            .expect("SU2 fixture should contain the recoupled output tree")
            .clone();
        let lhs_space = || {
            FusionTensorMapSpace::new_unbound(
                TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
                lhs_hom.clone(),
                crate::tests::packed_fixture_structure(
                    4,
                    [(BlockKey::from(src_tree.clone()), vec![1, 1, 1, 1])],
                )
                .unwrap(),
            )
            .unwrap()
            .try_bind_rule(&rule)
            .unwrap()
        };
        let rhs_space = || {
            FusionTensorMapSpace::from_degeneracy_shapes(
                TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
                FusionTreeHomSpace::from_sector_ids([], []),
                &rule,
                [vec![]],
            )
            .unwrap()
        };
        let dst_space = || {
            FusionTensorMapSpace::new_unbound(
                TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
                lhs_hom.clone(),
                crate::tests::packed_fixture_structure(
                    4,
                    [
                        (BlockKey::from(src_tree.clone()), vec![1, 1, 1, 1]),
                        (BlockKey::from(recoupled_tree.clone()), vec![1, 1, 1, 1]),
                    ],
                )
                .unwrap(),
            )
            .unwrap()
            .try_bind_rule(&rule)
            .unwrap()
        };
        let allocations = Rc::new(RefCell::new(Vec::new()));
        let lhs =
            TensorMap::<f64, 4, 0, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![10.0], "lhs", allocations.clone()),
                lhs_space(),
            )
            .unwrap();
        let rhs =
            TensorMap::<f64, 0, 0, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![5.0], "rhs", allocations.clone()),
                rhs_space(),
            )
            .unwrap();
        let mut dst =
            TensorMap::<f64, 4, 0, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![1.0, 2.0], "destination", allocations.clone()),
                dst_space(),
            )
            .unwrap();
        let axes = TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[0, 2, 1, 3]));
        let plan = prepare_tensorcontract_fusion_plan(
            &rule,
            dst.fusion_space().unwrap(),
            lhs.fusion_space().unwrap(),
            rhs.fusion_space().unwrap(),
            axes,
        )
        .unwrap();
        assert!(!plan.output_transform_is_identity());

        let mut expected_dst =
            TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space())
                .unwrap();
        let expected_lhs =
            TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![10.0], lhs_space()).unwrap();
        let expected_rhs =
            TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![5.0], rhs_space()).unwrap();
        run_host_reference(
            &rule,
            &plan,
            &mut expected_dst,
            &expected_lhs,
            &expected_rhs,
            2.0,
            3.0,
        );

        let mut tree_context =
            TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
        let mut contract_backend = DenseTreeTransformOperations::default();
        let mut contract_workspace = TensorContractWorkspace::default();
        let mut dynamic_space_cache = DynamicFusionSpaceCache::default();
        let mut fusion_block_cache =
            super::super::resolution::ContractionResolutionCache::default();
        let mut fusion_block_workspace = StorageFusionBlockContractWorkspace::<
            TrackingScratch<f64>,
            TrackingScratch<f64>,
            TrackingScratch<f64>,
        >::default();
        let mut scratch = StorageDynamicFusionScratchWorkspace::<
            TrackingScratch<f64>,
            TrackingScratch<f64>,
            TrackingScratch<f64>,
        >::default();

        for _ in 0..2 {
            dst.data_mut().copy_from_slice(&[1.0, 2.0]);
            tensorcontract_fusion_dynamic_plan_into_storage_context(
                &mut tree_context,
                &mut contract_backend,
                &mut contract_workspace,
                &mut dynamic_space_cache,
                &mut fusion_block_cache,
                &mut fusion_block_workspace,
                &mut scratch,
                &rule,
                &plan,
                &mut dst,
                &lhs,
                &rhs,
                2.0,
                3.0,
            )
            .unwrap();

            assert_eq!(dst.data(), expected_dst.data());
        }
        let allocations = allocations.borrow();
        // Scratch spaces enumerate the full tree set of their hom spaces
        // (structural zeros materialized), so the transformed lhs and the
        // core destination hold both SU2 trees. The scalar RHS is borrowed.
        assert_eq!(
            allocations[..2],
            [
                ScratchAllocation {
                    label: "lhs",
                    len: 2,
                },
                ScratchAllocation {
                    label: "destination",
                    len: 2,
                },
            ]
        );
        // Core-form transform scratch only: the contraction GEMMs directly on
        // the coupled scratch, with no pack/scatter allocations.
        assert_eq!(allocations[2..], []);
    }

    #[test]
    fn dynamic_storage_context_incomplete_identity_rhs_materializes_core_grid() {
        let rule = SU2FusionRule;
        let rhs_hom = FusionTreeHomSpace::from_sector_ids([], [(1, 1), (1, 1), (1, 1), (1, 1)]);
        let rhs_keys = rhs_hom.fusion_tree_keys(&rule);
        assert_eq!(rhs_keys.len(), 2);
        let rhs_tree = rhs_keys[0].clone();
        let dst_hom = rhs_hom.permute(&rule, &[0, 1, 2, 3], &[]).unwrap();
        let dst_keys = dst_hom.fusion_tree_keys(&rule);
        assert_eq!(dst_keys.len(), 2);
        let lhs_space = || {
            FusionTensorMapSpace::from_degeneracy_shapes(
                TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
                FusionTreeHomSpace::from_sector_ids([], []),
                &rule,
                [vec![]],
            )
            .unwrap()
        };
        let rhs_space = || {
            FusionTensorMapSpace::new_unbound(
                TensorMapSpace::<0, 4>::from_dims([], [1, 1, 1, 1]).unwrap(),
                rhs_hom.clone(),
                crate::tests::packed_fixture_structure(
                    4,
                    [(BlockKey::from(rhs_tree.clone()), vec![1, 1, 1, 1])],
                )
                .unwrap(),
            )
            .unwrap()
            .try_bind_rule(&rule)
            .unwrap()
        };
        let dst_space = || {
            FusionTensorMapSpace::new_unbound(
                TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
                dst_hom.clone(),
                crate::tests::packed_fixture_structure(
                    4,
                    dst_keys
                        .iter()
                        .cloned()
                        .map(|key| (BlockKey::from(key), vec![1, 1, 1, 1])),
                )
                .unwrap(),
            )
            .unwrap()
            .try_bind_rule(&rule)
            .unwrap()
        };
        let allocations = Rc::new(RefCell::new(Vec::new()));
        let lhs =
            TensorMap::<f64, 0, 0, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![2.0], "lhs", allocations.clone()),
                lhs_space(),
            )
            .unwrap();
        let rhs =
            TensorMap::<f64, 0, 4, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![3.0], "rhs", allocations.clone()),
                rhs_space(),
            )
            .unwrap();
        let mut dst =
            TensorMap::<f64, 4, 0, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![5.0, 7.0], "destination", allocations.clone()),
                dst_space(),
            )
            .unwrap();
        let axes = TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[0, 1, 2, 3]));
        let plan = prepare_tensorcontract_fusion_plan(
            &rule,
            dst.fusion_space().unwrap(),
            lhs.fusion_space().unwrap(),
            rhs.fusion_space().unwrap(),
            axes,
        )
        .unwrap();
        let mut expected =
            TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![5.0, 7.0], dst_space())
                .unwrap();
        let expected_lhs =
            TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![2.0], lhs_space()).unwrap();
        let expected_rhs =
            TensorMap::<f64, 0, 4>::from_vec_with_fusion_space(vec![3.0], rhs_space()).unwrap();
        run_host_reference(
            &rule,
            &plan,
            &mut expected,
            &expected_lhs,
            &expected_rhs,
            2.0,
            3.0,
        );
        let mut tree_context =
            TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
        let mut contract_backend = DenseTreeTransformOperations::default();
        let mut contract_workspace = TensorContractWorkspace::default();
        let mut dynamic_space_cache = DynamicFusionSpaceCache::default();
        let mut fusion_block_cache =
            super::super::resolution::ContractionResolutionCache::default();
        let mut fusion_block_workspace = StorageFusionBlockContractWorkspace::<
            TrackingScratch<f64>,
            TrackingScratch<f64>,
            TrackingScratch<f64>,
        >::default();
        let mut scratch = StorageDynamicFusionScratchWorkspace::<
            TrackingScratch<f64>,
            TrackingScratch<f64>,
            TrackingScratch<f64>,
        >::default();

        tensorcontract_fusion_dynamic_plan_into_storage_context(
            &mut tree_context,
            &mut contract_backend,
            &mut contract_workspace,
            &mut dynamic_space_cache,
            &mut fusion_block_cache,
            &mut fusion_block_workspace,
            &mut scratch,
            &rule,
            &plan,
            &mut dst,
            &lhs,
            &rhs,
            2.0,
            3.0,
        )
        .unwrap();

        assert_eq!(dst.data(), expected.data());
        // What: identity axes do not borrow an incomplete RHS layout; the
        // materialized core grid contains both structural-zero SU2 blocks.
        assert_eq!(
            allocations.borrow().as_slice(),
            [
                ScratchAllocation {
                    label: "rhs",
                    len: 2,
                },
                ScratchAllocation {
                    label: "destination",
                    len: 2,
                },
            ]
        );
    }

    fn run_host_reference<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
    >(
        rule: &R,
        plan: &FusionContractPlan,
        dst: &mut TensorMap<f64, DST_NOUT, DST_NIN>,
        lhs: &TensorMap<f64, LHS_NOUT, LHS_NIN>,
        rhs: &TensorMap<f64, RHS_NOUT, RHS_NIN>,
        alpha: f64,
        beta: f64,
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + TreeTransformRuleCacheKey<Key = crate::tree_transform::TreeTransformBuiltinRuleCacheKey>,
    {
        let mut tree_context =
            TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
        let mut contract_backend = DenseTreeTransformOperations::default();
        let mut contract_workspace = TensorContractWorkspace::default();
        let mut dynamic_space_cache = DynamicFusionSpaceCache::default();
        let mut fusion_block_cache =
            super::super::resolution::ContractionResolutionCache::default();
        let mut fusion_block_workspace = FusionBlockContractWorkspace::<f64>::default();
        let mut scratch = DynamicFusionScratchWorkspace::<f64>::default();
        tensorcontract_fusion_dynamic_plan_into_context(
            &mut tree_context,
            &mut contract_backend,
            &mut contract_workspace,
            &mut dynamic_space_cache,
            &mut fusion_block_cache,
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
        .unwrap();
    }
}
