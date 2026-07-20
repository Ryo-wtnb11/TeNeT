use std::any::Any;
use std::hash::Hash;
use std::sync::Arc;

use tenet_core::{
    BlockStructure, CheckedFusionAlgebra, CoreError, FusionRule, FusionTensorMapSpace,
    HostReadableStorage, HostWritableStorage, LoweredMultiplicityFreeAlgebra,
    MultiplicityFreeRigidSymbols, Placement, ScratchStorage, SimilarStorage, TensorMap,
    TensorStorage,
};

use crate::cache::{
    OperationCachePolicy, TensorContractStructureCache, TensorContractStructureCacheKey,
};
use crate::lowering::adjoint_fusion_space_view;
use crate::storage_scratch::StorageTensorContractWorkspace;
use crate::tree_context::TreeTransformExecutionContext;
use crate::tree_transform::TreeTransformRuleCacheKey;
use crate::{
    DenseBlockScalar, DenseRecouplingScalar, DenseTreeTransformOperations, HostTensorOperations,
    OperationError, RecouplingCoefficientAction, ReportsPlacement, TreeTransformBackend,
};
use tenet_operations::{TensorContractSpec, TensorContractSpecOwned};

use super::backend::{
    tensorcontract_structure_with_storage_workspace_dense_executor, TensorContractBackend,
};
use super::dynamic::DynamicFusionSpaceCache;
use super::dynamic_space::{
    encoded_layout_primer, BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand,
    LayoutKeyBuilder,
};
use super::fusion::{
    prepare_tensorcontract_fusion_plan, prepare_tensorcontract_fusion_plan_dyn_prelowered,
    prepare_tensorcontract_fusion_plan_dyn_prelowered_with_primer, tensorcontract_fusion_structure,
    tensorcontract_fusion_structure_dyn_prelowered, FusionContractPlan,
    EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CORE_DST, SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
};
use super::fusion_block::FusionBlockContractWorkspace;
use super::resolution::{ContractionResolutionCache, Resolution};
use super::scratch::DynamicFusionScratchWorkspace;
use super::structure::{TensorContractAxisPlan, TensorContractStructure};
use tenet_operations::{TensorContractFusionProfile, TensorContractFusionRoute};

fn prelowered_plan_builder<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
    _primer: LayoutKeyBuilder<R>,
) -> Result<Arc<FusionContractPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    prepare_tensorcontract_fusion_plan_dyn_prelowered(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        lhs_conjugate,
        rhs_conjugate,
    )
    .map(Arc::new)
}

fn lowered_prelowered_plan_builder<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
    primer: LayoutKeyBuilder<R>,
) -> Result<Arc<FusionContractPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + LoweredMultiplicityFreeAlgebra
        + CheckedFusionAlgebra,
{
    prepare_tensorcontract_fusion_plan_dyn_prelowered_with_primer(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        lhs_conjugate,
        rhs_conjugate,
        primer,
    )
    .map(Arc::new)
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TensorContractPlanKey {
    axes: TensorContractSpecOwned,
}

impl TensorContractPlanKey {
    pub fn from_axes(
        lhs_rank: usize,
        rhs_rank: usize,
        dst_rank: usize,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError> {
        let axis_plan = TensorContractAxisPlan::compile(lhs_rank, rhs_rank, dst_rank, axes)?;
        Ok(Self {
            axes: TensorContractSpecOwned::new_with_conjugation(
                axis_plan.lhs_contracting_axes,
                axis_plan.rhs_contracting_axes,
                axis_plan.output_axes,
                axis_plan.lhs_conjugate,
                axis_plan.rhs_conjugate,
            ),
        })
    }

    #[inline]
    pub fn axes(&self) -> &TensorContractSpecOwned {
        &self.axes
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorContractCacheStats {
    structure_hits: usize,
    structure_misses: usize,
}

impl TensorContractCacheStats {
    #[inline]
    pub fn structure_hits(self) -> usize {
        self.structure_hits
    }

    #[inline]
    pub fn structure_misses(self) -> usize {
        self.structure_misses
    }
}

#[derive(Clone, Debug)]
pub struct TensorContractCache<PlanKey = TensorContractPlanKey> {
    structures: TensorContractStructureCache<f64, PlanKey>,
    ephemeral_structure: Option<(
        TensorContractStructureCacheKey<PlanKey>,
        TensorContractStructure<f64>,
    )>,
    stats: TensorContractCacheStats,
}

impl<PlanKey> Default for TensorContractCache<PlanKey> {
    fn default() -> Self {
        Self {
            structures: TensorContractStructureCache::default(),
            ephemeral_structure: None,
            stats: TensorContractCacheStats::default(),
        }
    }
}

impl<PlanKey> TensorContractCache<PlanKey>
where
    PlanKey: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_policy(policy: OperationCachePolicy) -> Self {
        Self {
            structures: TensorContractStructureCache::with_policy(policy),
            ephemeral_structure: None,
            stats: TensorContractCacheStats::default(),
        }
    }

    #[inline]
    pub fn policy(&self) -> OperationCachePolicy {
        self.structures.policy()
    }

    pub fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.structures.set_policy(policy);
        self.ephemeral_structure = None;
    }

    #[inline]
    pub fn structure_len(&self) -> usize {
        self.structures.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty()
    }

    #[inline]
    pub fn stats(&self) -> TensorContractCacheStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = TensorContractCacheStats::default();
    }
}

impl TensorContractCache<TensorContractPlanKey> {
    pub fn get_or_compile<
        TDst,
        TLhs,
        TRhs,
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
        &mut self,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        axes: TensorContractSpec<'_>,
    ) -> Result<&TensorContractStructure, OperationError>
    where
        DDst: TensorStorage<TDst>,
        DLhs: TensorStorage<TLhs>,
        DRhs: TensorStorage<TRhs>,
    {
        let plan_key = TensorContractPlanKey::from_axes(
            lhs.structure().rank(),
            rhs.structure().rank(),
            dst.structure().rank(),
            axes,
        )?;
        let structure_key = TensorContractStructureCacheKey::from_structures(
            plan_key.clone(),
            dst.structure(),
            lhs.structure(),
            rhs.structure(),
        )?;
        if !self.structures.policy().stores_entries() {
            self.stats.structure_misses += 1;
            let structure =
                TensorContractStructure::compile(dst, lhs, rhs, plan_key.axes().as_spec())?;
            self.ephemeral_structure = Some((structure_key, structure));
            return Ok(&self
                .ephemeral_structure
                .as_ref()
                .expect("ephemeral tensor contract structure inserted before replay")
                .1);
        }
        if self.structures.get(&structure_key).is_some() {
            self.stats.structure_hits += 1;
            self.structures.touch(&structure_key);
        } else {
            self.stats.structure_misses += 1;
            let structure = Arc::new(TensorContractStructure::compile(
                dst,
                lhs,
                rhs,
                plan_key.axes().as_spec(),
            )?);
            self.structures.insert_arc(structure_key.clone(), structure);
        }
        Ok(self
            .structures
            .get(&structure_key)
            .expect("tensor contract structure inserted before replay"))
    }
}

#[derive(Debug)]
pub struct TensorContractExecutionContext<D, B = DenseTreeTransformOperations>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    B: TensorContractBackend<D, f64>,
{
    backend: B,
    workspace: B::Workspace,
    cache: TensorContractCache,
}

impl<D, B> TensorContractExecutionContext<D, B>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    B: TensorContractBackend<D, f64>,
{
    pub fn with_parts(backend: B, workspace: B::Workspace, cache: TensorContractCache) -> Self {
        Self {
            backend,
            workspace,
            cache,
        }
    }

    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    #[inline]
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[inline]
    pub fn workspace(&self) -> &B::Workspace {
        &self.workspace
    }

    #[inline]
    pub fn cache(&self) -> &TensorContractCache {
        &self.cache
    }

    #[inline]
    pub fn cache_mut(&mut self) -> &mut TensorContractCache {
        &mut self.cache
    }

    pub fn set_cache_policy(&mut self, policy: OperationCachePolicy) {
        self.cache.set_policy(policy);
    }

    pub fn into_parts(self) -> (B, B::Workspace, TensorContractCache) {
        (self.backend, self.workspace, self.cache)
    }
}

impl<D, B> TensorContractExecutionContext<D, B>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    B: TensorContractBackend<D, f64> + ReportsPlacement,
    B::Workspace: ReportsPlacement,
{
    #[inline]
    pub fn backend_placement(&self) -> Placement {
        self.backend.placement()
    }

    #[inline]
    pub fn workspace_placement(&self) -> Placement {
        self.workspace.placement()
    }

    #[inline]
    pub fn is_host_context(&self) -> bool {
        self.backend.is_host_placement() && self.workspace.is_host_placement()
    }
}

impl<D, B> TensorContractExecutionContext<D, B>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    B: TensorContractBackend<D, f64>,
    B::Workspace: Default,
{
    pub fn new(backend: B) -> Self {
        Self::with_parts(backend, B::Workspace::default(), TensorContractCache::new())
    }
}

impl<D, B> Default for TensorContractExecutionContext<D, B>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    B: TensorContractBackend<D, f64> + Default,
    B::Workspace: Default,
{
    fn default() -> Self {
        Self::new(B::default())
    }
}

impl<D, B> TensorContractExecutionContext<D, B>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    B: TensorContractBackend<D, f64>,
{
    pub fn tensorcontract_into<
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
        &mut self,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        let Self {
            backend,
            workspace,
            cache,
        } = self;
        let structure = cache.get_or_compile(dst, lhs, rhs, axes)?;
        backend.tensorcontract_structure_into(workspace, structure, dst, lhs, rhs, alpha, beta)
    }
}

impl<D> TensorContractExecutionContext<D, DenseTreeTransformOperations>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tensorcontract_into_storage_workspace<
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
        &mut self,
        storage_workspace: &mut StorageTensorContractWorkspace<DDst::Similar>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        DDst: HostWritableStorage<D> + SimilarStorage<D>,
        DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>,
    {
        let structure = self.cache.get_or_compile(dst, lhs, rhs, axes)?;
        tensorcontract_structure_with_storage_workspace_dense_executor(
            self.backend.dense_mut(),
            storage_workspace,
            structure,
            dst,
            lhs,
            rhs,
            alpha,
            beta,
        )
    }
}

pub fn tensorcontract_into_with_context<
    B,
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
    context: &mut TensorContractExecutionContext<D, B>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    context.tensorcontract_into(dst, lhs, rhs, axes, alpha, beta)
}

pub struct TensorContractFusionExecutionContext<
    D,
    RuleKey,
    BT = DenseTreeTransformOperations,
    BC = DenseTreeTransformOperations,
> where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
{
    tree_context: TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    dynamic_space_cache: DynamicFusionSpaceCache<RuleKey>,
    // One cache entry per (rule, spaces, axes): route and plan resolve
    // together (TensorKit keeps exactly one transformer entry per
    // (spaces, permutation); it never caches a route separately).
    resolution_cache: ContractionResolutionCache<RuleKey>,
    contract_backend: BC,
    contract_workspace: BC::Workspace,
    fusion_block_workspace: FusionBlockContractWorkspace<D>,
    fusion_scratch: DynamicFusionScratchWorkspace<D>,
    #[cfg(test)]
    last_top_level_resolution_was_core: bool,
}

pub type HostTreeFusionExecutionContext<D, RuleKey> = TensorContractFusionExecutionContext<
    D,
    RuleKey,
    HostTensorOperations,
    DenseTreeTransformOperations,
>;

impl<D, RuleKey, BT, BC> TensorContractFusionExecutionContext<D, RuleKey, BT, BC>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
{
    pub fn with_parts(
        tree_context: TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        contract_backend: BC,
        contract_workspace: BC::Workspace,
    ) -> Self {
        Self {
            tree_context,
            dynamic_space_cache: DynamicFusionSpaceCache::default(),
            resolution_cache: ContractionResolutionCache::default(),
            contract_backend,
            contract_workspace,
            fusion_block_workspace: FusionBlockContractWorkspace::default(),
            fusion_scratch: DynamicFusionScratchWorkspace::default(),
            #[cfg(test)]
            last_top_level_resolution_was_core: false,
        }
    }

    #[inline]
    pub fn tree_context(&self) -> &TreeTransformExecutionContext<D, RuleKey, f64, BT> {
        &self.tree_context
    }

    #[inline]
    pub fn tree_context_mut(&mut self) -> &mut TreeTransformExecutionContext<D, RuleKey, f64, BT> {
        &mut self.tree_context
    }

    #[inline]
    pub fn dynamic_fusion_space_cache_len(&self) -> usize {
        self.dynamic_space_cache.len()
    }

    #[inline]
    pub fn dynamic_fusion_space_cache_hits(&self) -> usize {
        self.dynamic_space_cache.stats().hits()
    }

    #[inline]
    pub fn dynamic_fusion_space_cache_fast_hits(&self) -> usize {
        self.dynamic_space_cache.stats().fast_hits()
    }

    #[inline]
    pub fn dynamic_fusion_space_cache_misses(&self) -> usize {
        self.dynamic_space_cache.stats().misses()
    }

    #[inline]
    pub fn contract_backend(&self) -> &BC {
        &self.contract_backend
    }

    #[inline]
    pub fn contract_backend_mut(&mut self) -> &mut BC {
        &mut self.contract_backend
    }

    #[inline]
    pub fn contract_workspace(&self) -> &BC::Workspace {
        &self.contract_workspace
    }

    /// Entries in the unified contraction resolution cache (route + plan
    /// resolve together; one entry per (rule, spaces, axes)).
    #[inline]
    pub fn contraction_resolution_cache_len(&self) -> usize {
        self.resolution_cache.len()
    }

    #[inline]
    pub fn contraction_resolution_cache_hits(&self) -> usize {
        self.resolution_cache.stats().hits
    }

    #[inline]
    pub fn contraction_resolution_cache_fast_hits(&self) -> usize {
        self.resolution_cache.stats().fast_hits
    }

    #[inline]
    pub fn contraction_resolution_cache_misses(&self) -> usize {
        self.resolution_cache.stats().misses
    }

    #[cfg(test)]
    pub(crate) fn last_resolution_is_core(&self) -> bool {
        self.last_top_level_resolution_was_core
    }

    pub fn set_cache_policy(&mut self, policy: OperationCachePolicy) {
        self.tree_context.set_cache_policy(policy);
        self.dynamic_space_cache.set_policy(policy);
        self.resolution_cache.set_policy(policy);
    }

    pub fn into_parts(
        self,
    ) -> (
        TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        BC,
        BC::Workspace,
    ) {
        (
            self.tree_context,
            self.contract_backend,
            self.contract_workspace,
        )
    }
}

impl<D, RuleKey, BT, BC> TensorContractFusionExecutionContext<D, RuleKey, BT, BC>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64> + ReportsPlacement,
    BT::Workspace: ReportsPlacement,
    BC: TensorContractBackend<D, f64> + ReportsPlacement,
    BC::Workspace: ReportsPlacement,
{
    #[inline]
    pub fn tree_backend_placement(&self) -> Placement {
        self.tree_context.backend_placement()
    }

    #[inline]
    pub fn tree_workspace_placement(&self) -> Placement {
        self.tree_context.workspace_placement()
    }

    #[inline]
    pub fn contract_backend_placement(&self) -> Placement {
        self.contract_backend.placement()
    }

    #[inline]
    pub fn contract_workspace_placement(&self) -> Placement {
        self.contract_workspace.placement()
    }

    #[inline]
    pub fn fusion_block_workspace_placement(&self) -> Placement {
        self.fusion_block_workspace.placement()
    }

    #[inline]
    pub fn fusion_scratch_workspace_placement(&self) -> Placement {
        self.fusion_scratch.placement()
    }

    #[inline]
    pub fn is_host_context(&self) -> bool {
        self.tree_context.is_host_context()
            && self.contract_backend.is_host_placement()
            && self.contract_workspace.is_host_placement()
            && self.fusion_block_workspace.is_host_placement()
            && self.fusion_scratch.is_host_placement()
    }
}

impl<D, RuleKey, BT, BC> TensorContractFusionExecutionContext<D, RuleKey, BT, BC>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    BT::Workspace: Default,
    BC::Workspace: Default,
{
    pub fn new(tree_backend: BT, contract_backend: BC) -> Self {
        Self::with_parts(
            TreeTransformExecutionContext::new(tree_backend),
            contract_backend,
            BC::Workspace::default(),
        )
    }
}

impl<D, RuleKey, BT, BC> Default for TensorContractFusionExecutionContext<D, RuleKey, BT, BC>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64> + Default,
    BC: TensorContractBackend<D, f64> + Default,
    BT::Workspace: Default,
    BC::Workspace: Default,
{
    fn default() -> Self {
        Self::new(BT::default(), BC::default())
    }
}

impl<D, RuleKey, BT, BC> TensorContractFusionExecutionContext<D, RuleKey, BT, BC>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
{
    pub fn tensorcontract_fusion_into<
        R,
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
        &mut self,
        rule: &R,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
        DDst: HostWritableStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>,
    {
        let dst_fusion = dst
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let lhs_fusion = lhs
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let rhs_fusion = rhs
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;

        let dst_dynamic = DynamicFusionMapSpace::from_typed(dst_fusion);
        let lhs_dynamic = DynamicFusionMapSpace::from_typed(lhs_fusion);
        let rhs_dynamic = DynamicFusionMapSpace::from_typed(rhs_fusion);
        let resolution = self.resolution_cache.get_or_resolve(
            rule,
            &dst_dynamic,
            &lhs_dynamic,
            &rhs_dynamic,
            axes,
            || match tensorcontract_fusion_structure(rule, dst, lhs, rhs, axes) {
                Ok(structure) => Ok(Some(std::sync::Arc::new(structure))),
                Err(OperationError::UnsupportedTensorContractScope {
                    message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
                }) => Ok(None),
                Err(err) => Err(err),
            },
            || {
                prepare_tensorcontract_fusion_plan(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)
                    .map(std::sync::Arc::new)
            },
        )?;
        self.execute_resolution(&resolution, rule, dst, lhs, rhs, alpha, beta)
    }

    /// Dynamic-rank `tensorcontract!`: same resolution-cache path and route
    /// gates as [`Self::tensorcontract_fusion_into`], operating on
    /// [`DynamicFusionMapSpace`] handles plus raw slices in the
    /// coupled-sector matrix layout. `dst_data` must be sized for
    /// `dst_space.required_len()`.
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_fusion_dyn_into<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        // Why not accept a separate rule: the lhs bound space is the authority
        // used for planning and execution; the raw core only checks identities.
        self.tensorcontract_fusion_dyn_into_raw_with_primer(
            lhs_space.provider(),
            dst_space.space(),
            dst_data,
            lhs_space.space(),
            lhs_data,
            rhs_space.space(),
            rhs_data,
            axes,
            alpha,
            beta,
            lhs_space.layout_primer(),
        )
    }

    /// Built-in multiplicity-free sibling that carries typed sectors through
    /// cold layout enumeration before encoding reusable block keys.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_fusion_dyn_into_lowered<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + LoweredMultiplicityFreeAlgebra
            + CheckedFusionAlgebra
            + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        self.tensorcontract_fusion_dyn_into(
            dst_space, dst_data, lhs_space, lhs_data, rhs_space, rhs_data, axes, alpha, beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(crate) fn tensorcontract_fusion_dyn_into_raw<R>(
        &mut self,
        rule: &R,
        dst_space: &DynamicFusionMapSpace,
        dst_data: &mut [D],
        lhs_space: &DynamicFusionMapSpace,
        lhs_data: &[D],
        rhs_space: &DynamicFusionMapSpace,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        self.tensorcontract_fusion_dyn_into_raw_with_primer(
            rule,
            dst_space,
            dst_data,
            lhs_space,
            lhs_data,
            rhs_space,
            rhs_data,
            axes,
            alpha,
            beta,
            encoded_layout_primer::<R>,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn tensorcontract_fusion_dyn_into_raw_with_primer<R>(
        &mut self,
        rule: &R,
        dst_space: &DynamicFusionMapSpace,
        dst_data: &mut [D],
        lhs_space: &DynamicFusionMapSpace,
        lhs_data: &[D],
        rhs_space: &DynamicFusionMapSpace,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
        layout_primer: LayoutKeyBuilder<R>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        let resolution = self.resolution_cache.get_or_resolve(
            rule,
            dst_space,
            lhs_space,
            rhs_space,
            axes,
            || match super::fusion::tensorcontract_fusion_structure_dyn_raw(
                rule,
                dst_space,
                lhs_space,
                rhs_space,
                Arc::clone(lhs_space.structure()),
                Arc::clone(rhs_space.structure()),
                axes,
            ) {
                Ok(structure) => Ok(Some(Arc::new(structure))),
                Err(OperationError::UnsupportedTensorContractScope {
                    message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
                }) => Ok(None),
                Err(err) => Err(err),
            },
            || {
                super::fusion::prepare_tensorcontract_fusion_plan_dyn_raw(
                    rule, dst_space, lhs_space, rhs_space, axes,
                )
                .map(Arc::new)
            },
        )?;
        #[cfg(test)]
        {
            self.last_top_level_resolution_was_core = matches!(resolution, Resolution::Core(_));
        }
        let dynamic_artifact = self.prepare_dynamic_execution_artifact::<_, false>(
            &resolution,
            rule,
            Some(dst_space),
            dst_space.structure(),
            Some(lhs_space),
            None,
            lhs_space.structure(),
            Some(rhs_space),
            None,
            rhs_space.structure(),
            layout_primer,
            None,
        )?;
        self.execute_resolution_dyn(
            &resolution,
            dynamic_artifact.as_ref(),
            dst_space.structure(),
            dst_data,
            lhs_space.structure(),
            lhs_data,
            rhs_space.structure(),
            rhs_data,
            alpha,
            beta,
        )
    }

    /// Executes TeNeT's validated prelowered operand seam.
    ///
    /// Logical spaces are the categorical authority; storage spaces and slices
    /// are the physical authority. Why not make this a normal entrypoint:
    /// arbitrary callers cannot establish the lazy-adjoint coherence contract.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_fusion_dyn_prelowered_into<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs: FusionOperand<'_>,
        lhs_data: &[D],
        rhs: FusionOperand<'_>,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        self.tensorcontract_fusion_dyn_prelowered_into_core(
            dst_space,
            dst_data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            axes,
            alpha,
            beta,
            dst_space.layout_primer(),
            prelowered_plan_builder::<R>,
        )
    }

    /// Built-in multiplicity-free sibling of the validated prelowered seam.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_fusion_dyn_prelowered_into_lowered<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs: FusionOperand<'_>,
        lhs_data: &[D],
        rhs: FusionOperand<'_>,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + LoweredMultiplicityFreeAlgebra
            + CheckedFusionAlgebra
            + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        self.tensorcontract_fusion_dyn_prelowered_into_core(
            dst_space,
            dst_data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            axes,
            alpha,
            beta,
            dst_space.layout_primer(),
            lowered_prelowered_plan_builder::<R>,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn tensorcontract_fusion_dyn_prelowered_into_core<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs: FusionOperand<'_>,
        lhs_data: &[D],
        rhs: FusionOperand<'_>,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
        layout_primer: LayoutKeyBuilder<R>,
        plan_builder: fn(
            &R,
            &DynamicFusionMapSpace,
            &DynamicFusionMapSpace,
            &DynamicFusionMapSpace,
            TensorContractSpec<'_>,
            bool,
            bool,
            LayoutKeyBuilder<R>,
        ) -> Result<Arc<FusionContractPlan>, OperationError>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        let rule = dst_space.provider();
        for space in [
            lhs.logical_space(),
            lhs.storage_space(),
            rhs.logical_space(),
            rhs.storage_space(),
        ] {
            space.validate_rule(rule)?;
        }
        if axes.lhs_conjugate() != lhs.storage_conjugate()
            || axes.rhs_conjugate() != rhs.storage_conjugate()
        {
            return Err(OperationError::InvalidArgument {
                message: "prelowered operand flags must match the contraction cache key",
            });
        }
        let resolution = self.resolution_cache.get_or_resolve_prelowered(
            rule,
            dst_space.space(),
            lhs.logical_space(),
            lhs.storage_space(),
            rhs.logical_space(),
            rhs.storage_space(),
            axes,
            || match tensorcontract_fusion_structure_dyn_prelowered(
                rule,
                dst_space.space(),
                lhs.logical_space(),
                lhs.storage_space(),
                rhs.logical_space(),
                rhs.storage_space(),
                axes,
            ) {
                Ok(structure) => Ok(Some(Arc::new(structure))),
                Err(OperationError::UnsupportedTensorContractScope {
                    message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
                }) => Ok(None),
                Err(err) => Err(err),
            },
            || {
                plan_builder(
                    rule,
                    dst_space.space(),
                    lhs.logical_space(),
                    rhs.logical_space(),
                    axes,
                    lhs.storage_conjugate(),
                    rhs.storage_conjugate(),
                    layout_primer,
                )
            },
        )?;
        #[cfg(test)]
        {
            self.last_top_level_resolution_was_core = matches!(resolution, Resolution::Core(_));
        }
        let dynamic_artifact = self.prepare_dynamic_execution_artifact::<_, false>(
            &resolution,
            rule,
            Some(dst_space.space()),
            dst_space.space().structure(),
            Some(lhs.logical_space()),
            Some(lhs.storage_space()),
            lhs.storage_space().structure(),
            Some(rhs.logical_space()),
            Some(rhs.storage_space()),
            rhs.storage_space().structure(),
            layout_primer,
            None,
        )?;
        self.execute_resolution_dyn(
            &resolution,
            dynamic_artifact.as_ref(),
            dst_space.space().structure(),
            dst_data,
            lhs.storage_space().structure(),
            lhs_data,
            rhs.storage_space().structure(),
            rhs_data,
            alpha,
            beta,
        )
    }

    /// Categorical map composition on the coupled-sector block matrices.
    ///
    /// Unlike `tensorcontract!`, TensorKit `mul!` does not insert a
    /// fermionic supertrace twist. The logical/storage split still carries
    /// lazy adjoints without materializing either operand.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcompose_fusion_dyn_into_lowered<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs: FusionOperand<'_>,
        lhs_data: &[D],
        rhs: FusionOperand<'_>,
        rhs_data: &[D],
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + LoweredMultiplicityFreeAlgebra
            + CheckedFusionAlgebra
            + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        let rule = dst_space.provider();
        for space in [
            lhs.logical_space(),
            lhs.storage_space(),
            rhs.logical_space(),
            rhs.storage_space(),
        ] {
            space.validate_rule(rule)?;
        }
        let axes = TensorContractSpec::new_with_conjugation(
            lhs_axes,
            rhs_axes,
            tenet_operations::OutputAxisOrder::identity(),
            lhs.storage_conjugate(),
            rhs.storage_conjugate(),
        );
        // Why not give bosonic composition its own cache namespace: without a
        // supertrace twist it is exactly the existing contract operation, so a
        // second plan would duplicate cold layout work and retained state.
        if rule.braiding_style() != tenet_core::BraidingStyleKind::Fermionic {
            if !lhs.storage_conjugate() && !rhs.storage_conjugate() {
                return self.tensorcontract_fusion_dyn_into_raw_with_primer(
                    rule,
                    dst_space.space(),
                    dst_data,
                    lhs.logical_space(),
                    lhs_data,
                    rhs.logical_space(),
                    rhs_data,
                    axes,
                    alpha,
                    beta,
                    dst_space.layout_primer(),
                );
            }
            return self.tensorcontract_fusion_dyn_prelowered_into_lowered(
                dst_space, dst_data, lhs, lhs_data, rhs, rhs_data, axes, alpha, beta,
            );
        }
        let plan = self.resolution_cache.get_or_compile_composition_plan(
            rule,
            dst_space.space(),
            lhs.logical_space(),
            lhs.storage_space(),
            rhs.logical_space(),
            rhs.storage_space(),
            axes,
        )?;
        #[cfg(test)]
        {
            self.last_top_level_resolution_was_core = true;
        }
        self.execute_resolution_dyn(
            &Resolution::Core(plan),
            None,
            dst_space.space().structure(),
            dst_data,
            lhs.storage_space().structure(),
            lhs_data,
            rhs.storage_space().structure(),
            rhs_data,
            alpha,
            beta,
        )
    }

    /// Generic-fusion (Stage B3c-1) sibling of [`Self::tensorcontract_fusion_dyn_into`]:
    /// the SU(N) core/compose (fully-direct GEMM) route. Non-memoized (mirrors
    /// the generic tree-transform path) — the block GEMM is symmetry-agnostic,
    /// so it just needs the group-agnostic block plan. A contraction that would
    /// need source tree-pair transforms (open contracted legs) or conjugated
    /// operands is an explicit B3c-2 error; there is NO change to the dense GEMM
    /// seam. `dst_data` must be sized for `dst_space.required_len()` and
    /// zero-filled for `beta == 0` (blocks without a contributing GEMM stay).
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_fusion_dyn_into_generic<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: FusionRule,
    {
        self.tensorcontract_fusion_dyn_into_generic_raw(
            lhs_space.provider(),
            dst_space.space(),
            dst_data,
            lhs_space.space(),
            lhs_data,
            rhs_space.space(),
            rhs_data,
            axes,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tensorcontract_fusion_dyn_into_generic_raw<R>(
        &mut self,
        rule: &R,
        dst_space: &DynamicFusionMapSpace,
        dst_data: &mut [D],
        lhs_space: &DynamicFusionMapSpace,
        lhs_data: &[D],
        rhs_space: &DynamicFusionMapSpace,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: FusionRule,
    {
        let plan = super::fusion_block::compile_fusion_block_contract_plan_generic(
            rule, dst_space, lhs_space, rhs_space, axes,
        )?;
        let Self {
            contract_backend,
            contract_workspace,
            fusion_block_workspace,
            ..
        } = self;
        let mut kernels = crate::StridedHostKernelAdapter::with_transpose_backend(
            contract_backend.transpose_backend(),
        );
        let mut gemm = super::fusion_block::BackendRank2Gemm {
            backend: contract_backend,
            workspace: contract_workspace,
        };
        plan.execute_raw(
            &mut kernels,
            &mut gemm,
            fusion_block_workspace,
            dst_space.structure(),
            dst_data,
            lhs_space.structure(),
            lhs_data,
            rhs_space.structure(),
            rhs_data,
            alpha,
            beta,
        )
    }

    /// Dynamic-rank contraction replayed directly on opaque storages (the
    /// device path): resolves through the same resolution cache and route /
    /// twist gates as [`Self::tensorcontract_fusion_dyn_into`], but only the
    /// canonical fully-direct coupled-layout route executes — one
    /// [`StorageGemm`](tenet_operations::fusion_replay::StorageGemm) call
    /// per coupled-sector matrix, `alpha = 1`, `beta = 0`. The caller must
    /// pass a zero-filled destination: destination blocks without a
    /// contributing GEMM stay untouched (overwrite-on-zero semantics).
    /// Every other resolution (dynamic tree transforms, conjugate
    /// structures) is an explicit
    /// [`OperationError::UnsupportedTensorContractScope`]; there is no
    /// silent host fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_fusion_dyn_direct_on_storage<R, G, DDst, DLhs, DRhs>(
        &mut self,
        gemm: &mut G,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst: &mut DDst,
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs: &DLhs,
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs: &DRhs,
        axes: TensorContractSpec<'_>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        G: tenet_operations::fusion_replay::StorageGemm<D, DDst, DLhs, DRhs>,
        DDst: TensorStorage<D>,
        DLhs: TensorStorage<D>,
        DRhs: TensorStorage<D>,
    {
        self.tensorcontract_fusion_dyn_direct_on_storage_raw(
            lhs_space.provider(),
            gemm,
            dst_space.space(),
            dst,
            lhs_space.space(),
            lhs,
            rhs_space.space(),
            rhs,
            axes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tensorcontract_fusion_dyn_direct_on_storage_raw<R, G, DDst, DLhs, DRhs>(
        &mut self,
        rule: &R,
        gemm: &mut G,
        dst_space: &DynamicFusionMapSpace,
        dst: &mut DDst,
        lhs_space: &DynamicFusionMapSpace,
        lhs: &DLhs,
        rhs_space: &DynamicFusionMapSpace,
        rhs: &DRhs,
        axes: TensorContractSpec<'_>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        G: tenet_operations::fusion_replay::StorageGemm<D, DDst, DLhs, DRhs>,
        DDst: TensorStorage<D>,
        DLhs: TensorStorage<D>,
        DRhs: TensorStorage<D>,
    {
        let resolution = self.resolution_cache.get_or_resolve(
            rule,
            dst_space,
            lhs_space,
            rhs_space,
            axes,
            || match super::fusion::tensorcontract_fusion_structure_dyn_raw(
                rule,
                dst_space,
                lhs_space,
                rhs_space,
                Arc::clone(lhs_space.structure()),
                Arc::clone(rhs_space.structure()),
                axes,
            ) {
                Ok(structure) => Ok(Some(Arc::new(structure))),
                Err(OperationError::UnsupportedTensorContractScope {
                    message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
                }) => Ok(None),
                Err(err) => Err(err),
            },
            || {
                super::fusion::prepare_tensorcontract_fusion_plan_dyn_raw(
                    rule, dst_space, lhs_space, rhs_space, axes,
                )
                .map(Arc::new)
            },
        )?;
        match resolution {
            Resolution::Core(plan) if plan.is_fully_direct() => {
                plan.execute_direct_on_storage_prezeroed(gemm, dst, lhs, rhs)
            }
            Resolution::Core(_) | Resolution::DynamicTree(_) | Resolution::Structure(_) => {
                Err(OperationError::UnsupportedTensorContractScope {
                    message: "storage-direct contraction supports only the canonical \
                              fully-direct route; this contraction needs tree transforms \
                              or conjugate structures, which have no device kernels yet",
                })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_dynamic_execution_artifact<R, const PROFILED: bool>(
        &mut self,
        resolution: &Resolution,
        rule: &R,
        dst_space: Option<&DynamicFusionMapSpace>,
        dst_structure: &Arc<BlockStructure>,
        lhs_space: Option<&DynamicFusionMapSpace>,
        lhs_storage_space: Option<&DynamicFusionMapSpace>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_space: Option<&DynamicFusionMapSpace>,
        rhs_storage_space: Option<&DynamicFusionMapSpace>,
        rhs_structure: &Arc<BlockStructure>,
        layout_primer: LayoutKeyBuilder<R>,
        mut profile: Option<&mut TensorContractFusionProfile>,
    ) -> Result<Option<Arc<super::dynamic::DynamicTreeExecutionArtifact>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar,
    {
        let Resolution::DynamicTree(plan) = resolution else {
            return Ok(None);
        };
        let missing = || OperationError::Core(CoreError::MissingFusionSpace);
        let dst_space = dst_space.ok_or_else(missing)?;
        let lhs_space = lhs_space.ok_or_else(missing)?;
        let rhs_space = rhs_space.ok_or_else(missing)?;
        let lookup_start = PROFILED.then(std::time::Instant::now);
        let cached = self.dynamic_space_cache.get_execution_artifact(
            plan,
            dst_structure,
            lhs_structure,
            lhs_storage_space,
            rhs_structure,
            rhs_storage_space,
        );
        if let Some(start) = lookup_start {
            profile
                .as_deref_mut()
                .expect("profiled artifact preparation carries a profile")
                .prepared_plan += start.elapsed();
        }
        if let Some(artifact) = cached {
            return Ok(Some(artifact));
        }
        let artifact = Arc::new(super::dynamic::compile_dynamic_tree_execution_artifact::<
            _,
            _,
            _,
            _,
            PROFILED,
        >(
            &mut self.tree_context,
            &mut self.dynamic_space_cache,
            &mut self.resolution_cache,
            rule,
            layout_primer,
            plan.as_ref(),
            dst_space,
            lhs_space,
            lhs_storage_space,
            lhs_structure,
            rhs_space,
            rhs_storage_space,
            rhs_structure,
            profile.as_deref_mut(),
        )?);
        let publish_start = PROFILED.then(std::time::Instant::now);
        self.dynamic_space_cache.insert_execution_artifact(
            Arc::clone(plan),
            dst_structure,
            lhs_structure,
            lhs_storage_space,
            rhs_structure,
            rhs_storage_space,
            Arc::clone(&artifact),
        );
        if let Some(start) = publish_start {
            profile
                .as_deref_mut()
                .expect("profiled artifact preparation carries a profile")
                .prepared_plan += start.elapsed();
        }
        Ok(Some(artifact))
    }

    /// Executes a resolved contraction on raw slices; shared by the
    /// dynamic-rank entry point and the typed facade / prepared-handle path.
    ///
    /// The structures are passed separately from the spaces because the two
    /// callers replay on different structures: the dynamic entry replays on
    /// the space's canonical structure, while the typed wrapper replays on
    /// each tensor's own *storage* structure (which
    /// `from_storage_with_structure` lets differ from the space's). The
    /// spaces themselves are consumed only by the dynamic-tree route, so
    /// they are optional: a typed tensor without a fusion space errors
    /// there and only there (as before the merge).
    #[allow(clippy::too_many_arguments)]
    fn execute_resolution_dyn(
        &mut self,
        resolution: &Resolution,
        dynamic_artifact: Option<&Arc<super::dynamic::DynamicTreeExecutionArtifact>>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        match resolution {
            Resolution::Core(block_plan) => {
                let Self {
                    contract_backend,
                    contract_workspace,
                    fusion_block_workspace,
                    ..
                } = self;
                let mut kernels = crate::StridedHostKernelAdapter::with_transpose_backend(
                    contract_backend.transpose_backend(),
                );
                let mut gemm = super::fusion_block::BackendRank2Gemm {
                    backend: contract_backend,
                    workspace: contract_workspace,
                };
                block_plan.execute_raw(
                    &mut kernels,
                    &mut gemm,
                    fusion_block_workspace,
                    dst_structure,
                    dst_data,
                    lhs_structure,
                    lhs_data,
                    rhs_structure,
                    rhs_data,
                    alpha,
                    beta,
                )
            }
            Resolution::DynamicTree(_) => {
                let Self {
                    tree_context,
                    contract_backend,
                    contract_workspace,
                    fusion_block_workspace,
                    fusion_scratch,
                    ..
                } = self;
                let artifact =
                    dynamic_artifact.ok_or(OperationError::UnsupportedTensorContractScope {
                        message: "dynamic-tree resolution requires a compiled execution artifact",
                    })?;
                super::dynamic::execute_dynamic_tree_execution_artifact(
                    tree_context,
                    contract_backend,
                    contract_workspace,
                    fusion_block_workspace,
                    fusion_scratch,
                    artifact,
                    dst_structure,
                    dst_data,
                    lhs_data,
                    rhs_data,
                    alpha,
                    beta,
                )
            }
            Resolution::Structure(structure) => {
                self.contract_backend.tensorcontract_structure_into_raw(
                    &mut self.contract_workspace,
                    structure,
                    dst_structure,
                    lhs_structure,
                    rhs_structure,
                    dst_data,
                    lhs_data,
                    rhs_data,
                    alpha,
                    beta,
                )
            }
        }
    }

    /// Executes a resolved contraction; shared by the facade and the
    /// prepared-handle path.
    #[allow(clippy::too_many_arguments)]
    fn execute_resolution<
        R,
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
        &mut self,
        resolution: &Resolution,
        rule: &R,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
        DDst: HostWritableStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>,
    {
        let dst_space = dst
            .fusion_space()
            .map(|s| DynamicFusionMapSpace::from_typed(s));
        let lhs_space = lhs
            .fusion_space()
            .map(|s| DynamicFusionMapSpace::from_typed(s));
        let rhs_space = rhs
            .fusion_space()
            .map(|s| DynamicFusionMapSpace::from_typed(s));
        let dst_structure = Arc::clone(dst.structure());
        let lhs_structure = Arc::clone(lhs.structure());
        let rhs_structure = Arc::clone(rhs.structure());
        let dynamic_artifact = self.prepare_dynamic_execution_artifact::<_, false>(
            resolution,
            rule,
            dst_space.as_ref(),
            &dst_structure,
            lhs_space.as_ref(),
            None,
            &lhs_structure,
            rhs_space.as_ref(),
            None,
            &rhs_structure,
            encoded_layout_primer::<R>,
            None,
        )?;
        self.execute_resolution_dyn(
            resolution,
            dynamic_artifact.as_ref(),
            &dst_structure,
            dst.data_mut(),
            &lhs_structure,
            lhs.data(),
            &rhs_structure,
            rhs.data(),
            alpha,
            beta,
        )
    }

    /// Resolves the contraction route and plan once, returning a handle that
    /// [`Self::execute_prepared_tensorcontract_fusion`] replays without any
    /// cache lookups. Valid for tensors that share the prepared tensors'
    /// fusion-space handles.
    pub fn prepare_tensorcontract_fusion<
        R,
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
        &mut self,
        rule: &R,
        dst: &TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        axes: TensorContractSpec<'_>,
    ) -> Result<PreparedTensorContractFusion<RuleKey>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
        DDst: HostWritableStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>,
    {
        let dst_fusion = dst
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let lhs_fusion = lhs
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let rhs_fusion = rhs
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let dst_dynamic = DynamicFusionMapSpace::from_typed(dst_fusion);
        let lhs_dynamic = DynamicFusionMapSpace::from_typed(lhs_fusion);
        let rhs_dynamic = DynamicFusionMapSpace::from_typed(rhs_fusion);
        let resolution = self.resolution_cache.get_or_resolve(
            rule,
            &dst_dynamic,
            &lhs_dynamic,
            &rhs_dynamic,
            axes,
            || match tensorcontract_fusion_structure(rule, dst, lhs, rhs, axes) {
                Ok(structure) => Ok(Some(Arc::new(structure))),
                Err(OperationError::UnsupportedTensorContractScope {
                    message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
                }) => Ok(None),
                Err(err) => Err(err),
            },
            || {
                prepare_tensorcontract_fusion_plan(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)
                    .map(Arc::new)
            },
        )?;
        let dst_structure = Arc::clone(dst.structure());
        let lhs_structure = Arc::clone(lhs.structure());
        let rhs_structure = Arc::clone(rhs.structure());
        let dynamic_artifact = self.prepare_dynamic_execution_artifact::<_, false>(
            &resolution,
            rule,
            Some(&dst_dynamic),
            &dst_structure,
            Some(&lhs_dynamic),
            None,
            &lhs_structure,
            Some(&rhs_dynamic),
            None,
            &rhs_structure,
            encoded_layout_primer::<R>,
            None,
        )?;
        Ok(PreparedTensorContractFusion {
            rule: rule.tree_transform_rule_cache_key(),
            dst_fusion_space: PreparedFusionSpaceWitness::new(dst_fusion, dst.structure()),
            lhs_fusion_space: PreparedFusionSpaceWitness::new(lhs_fusion, lhs.structure()),
            rhs_fusion_space: PreparedFusionSpaceWitness::new(rhs_fusion, rhs.structure()),
            resolution,
            dynamic_artifact,
        })
    }

    /// Replays a prepared contraction. The tensors must share the prepared
    /// tensors' fusion-space handles, so tensors created from the same
    /// `FusionTensorMapSpace` handle (or clones of the prepared ones) are
    /// valid.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_prepared_tensorcontract_fusion<
        R,
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
        &mut self,
        prepared: &PreparedTensorContractFusion<RuleKey>,
        rule: &R,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
        DDst: HostWritableStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>,
    {
        let Some(dst_fusion) = dst.fusion_space() else {
            return Err(OperationError::StructureMismatch {
                tensor: "prepared contraction",
            });
        };
        let Some(lhs_fusion) = lhs.fusion_space() else {
            return Err(OperationError::StructureMismatch {
                tensor: "prepared contraction",
            });
        };
        let Some(rhs_fusion) = rhs.fusion_space() else {
            return Err(OperationError::StructureMismatch {
                tensor: "prepared contraction",
            });
        };
        if prepared.rule != rule.tree_transform_rule_cache_key()
            || !prepared
                .dst_fusion_space
                .matches(dst_fusion, dst.structure())
            || !prepared
                .lhs_fusion_space
                .matches(lhs_fusion, lhs.structure())
            || !prepared
                .rhs_fusion_space
                .matches(rhs_fusion, rhs.structure())
        {
            return Err(OperationError::StructureMismatch {
                tensor: "prepared contraction",
            });
        }
        let dst_structure = Arc::clone(dst.structure());
        let lhs_structure = Arc::clone(lhs.structure());
        let rhs_structure = Arc::clone(rhs.structure());
        self.execute_resolution_dyn(
            &prepared.resolution,
            prepared.dynamic_artifact.as_ref(),
            &dst_structure,
            dst.data_mut(),
            &lhs_structure,
            lhs.data(),
            &rhs_structure,
            rhs.data(),
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_fusion_into_profiled<
        R,
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
        &mut self,
        rule: &R,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
        profile: &mut TensorContractFusionProfile,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
        DDst: HostWritableStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>,
    {
        let total_start = std::time::Instant::now();

        let start = std::time::Instant::now();
        let dst_fusion = dst
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let lhs_fusion = lhs
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let rhs_fusion = rhs
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let dst_dynamic = DynamicFusionMapSpace::from_typed(dst_fusion);
        let lhs_dynamic = DynamicFusionMapSpace::from_typed(lhs_fusion);
        let rhs_dynamic = DynamicFusionMapSpace::from_typed(rhs_fusion);
        profile.typed_space_setup += start.elapsed();

        let start = std::time::Instant::now();
        let resolution = self.resolution_cache.get_or_resolve(
            rule,
            &dst_dynamic,
            &lhs_dynamic,
            &rhs_dynamic,
            axes,
            || match tensorcontract_fusion_structure(rule, dst, lhs, rhs, axes) {
                Ok(structure) => Ok(Some(std::sync::Arc::new(structure))),
                Err(OperationError::UnsupportedTensorContractScope {
                    message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
                }) => Ok(None),
                Err(err) => Err(err),
            },
            || {
                prepare_tensorcontract_fusion_plan(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)
                    .map(std::sync::Arc::new)
            },
        )?;
        profile.fusion_block_plan_lookup += start.elapsed();

        match &resolution {
            Resolution::Core(block_plan) => {
                let Self {
                    contract_backend,
                    contract_workspace,
                    fusion_block_workspace,
                    ..
                } = self;
                let dst_structure = std::sync::Arc::clone(dst.structure());
                let lhs_structure = std::sync::Arc::clone(lhs.structure());
                let rhs_structure = std::sync::Arc::clone(rhs.structure());
                let mut kernels = crate::StridedHostKernelAdapter::with_transpose_backend(
                    contract_backend.transpose_backend(),
                );
                let mut gemm = super::fusion_block::BackendRank2Gemm {
                    backend: contract_backend,
                    workspace: contract_workspace,
                };
                profile.route = TensorContractFusionRoute::CoreFusionBlocks;
                let result = block_plan.execute_raw_profiled(
                    &mut kernels,
                    &mut gemm,
                    fusion_block_workspace,
                    &dst_structure,
                    dst.data_mut(),
                    &lhs_structure,
                    lhs.data(),
                    &rhs_structure,
                    rhs.data(),
                    alpha,
                    beta,
                    profile,
                );
                profile.total += total_start.elapsed();
                result
            }
            Resolution::DynamicTree(_) => {
                profile.route = TensorContractFusionRoute::DynamicTreeCore;
                let dst_structure = Arc::clone(dst.structure());
                let lhs_structure = Arc::clone(lhs.structure());
                let rhs_structure = Arc::clone(rhs.structure());
                let artifact = self
                    .prepare_dynamic_execution_artifact::<_, true>(
                        &resolution,
                        rule,
                        Some(&dst_dynamic),
                        &dst_structure,
                        Some(&lhs_dynamic),
                        None,
                        &lhs_structure,
                        Some(&rhs_dynamic),
                        None,
                        &rhs_structure,
                        encoded_layout_primer::<R>,
                        Some(profile),
                    )?
                    .expect("dynamic-tree resolution compiles an execution artifact");
                let Self {
                    tree_context,
                    contract_backend,
                    contract_workspace,
                    fusion_block_workspace,
                    fusion_scratch,
                    ..
                } = self;
                let result = super::dynamic::execute_dynamic_tree_execution_artifact_profiled(
                    tree_context,
                    contract_backend,
                    contract_workspace,
                    fusion_block_workspace,
                    fusion_scratch,
                    artifact.as_ref(),
                    &dst_structure,
                    dst.data_mut(),
                    lhs.data(),
                    rhs.data(),
                    alpha,
                    beta,
                    profile,
                );
                profile.total += total_start.elapsed();
                result
            }
            Resolution::Structure(structure) => {
                profile.route = TensorContractFusionRoute::DenseConjugateStructure;
                let start = std::time::Instant::now();
                let result = self.contract_backend.tensorcontract_structure_into(
                    &mut self.contract_workspace,
                    structure,
                    dst,
                    lhs,
                    rhs,
                    alpha,
                    beta,
                );
                profile.dense_contract += start.elapsed();
                profile.total += total_start.elapsed();
                result
            }
        }
    }

    pub fn tensorcontract_fusion_prepared_into<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        const LHS_CAN_NOUT: usize,
        const LHS_CAN_NIN: usize,
        const RHS_CAN_NOUT: usize,
        const RHS_CAN_NIN: usize,
        SDst,
        SLhs,
        SRhs,
        SLhsCan,
        SRhsCan,
    >(
        &mut self,
        rule: &R,
        plan: &FusionContractPlan,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs_core: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan>,
        rhs_core: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        if !plan.output_transform_is_identity()
            || DST_NOUT != plan.core_dst_open_lhs_rank()
            || DST_NIN != plan.core_dst_open_rhs_rank()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CORE_DST,
            });
        }
        self.transform_sources_and_contract(
            rule, plan, dst, lhs_core, rhs_core, lhs, rhs, alpha, beta,
        )
    }

    pub fn tensorcontract_fusion_prepared_into_core_dst<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const DST_CAN_NOUT: usize,
        const DST_CAN_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        const LHS_CAN_NOUT: usize,
        const LHS_CAN_NIN: usize,
        const RHS_CAN_NOUT: usize,
        const RHS_CAN_NIN: usize,
        SDst,
        SDstCan,
        SLhs,
        SRhs,
        SLhsCan,
        SRhsCan,
    >(
        &mut self,
        rule: &R,
        plan: &FusionContractPlan,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        core_dst: &mut TensorMap<D, DST_CAN_NOUT, DST_CAN_NIN, SDstCan>,
        lhs_core: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan>,
        rhs_core: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        if DST_CAN_NOUT != plan.core_dst_open_lhs_rank()
            || DST_CAN_NIN != plan.core_dst_open_rhs_rank()
        {
            return Err(OperationError::StructureRankMismatch {
                expected: plan.core_dst_open_lhs_rank() + plan.core_dst_open_rhs_rank(),
                actual: DST_CAN_NOUT + DST_CAN_NIN,
            });
        }
        core_dst.data_mut().fill(D::zero());
        self.transform_sources_and_contract(
            rule,
            plan,
            core_dst,
            lhs_core,
            rhs_core,
            lhs,
            rhs,
            alpha,
            D::zero(),
        )?;
        self.tree_context.tree_transform_into(
            rule,
            plan.output_transform().clone(),
            dst,
            core_dst,
            D::one(),
            beta,
        )
    }

    fn transform_sources_and_contract<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        const LHS_CAN_NOUT: usize,
        const LHS_CAN_NIN: usize,
        const RHS_CAN_NOUT: usize,
        const RHS_CAN_NIN: usize,
        SDst,
        SLhs,
        SRhs,
        SLhsCan,
        SRhsCan,
    >(
        &mut self,
        rule: &R,
        plan: &FusionContractPlan,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs_core: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan>,
        rhs_core: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        if LHS_CAN_NOUT != plan.lhs_open_rank() || LHS_CAN_NIN != plan.lhs_contract_rank() {
            return Err(OperationError::StructureRankMismatch {
                expected: plan.lhs_open_rank() + plan.lhs_contract_rank(),
                actual: LHS_CAN_NOUT + LHS_CAN_NIN,
            });
        }
        if RHS_CAN_NOUT != plan.rhs_contract_rank() || RHS_CAN_NIN != plan.rhs_open_rank() {
            return Err(OperationError::StructureRankMismatch {
                expected: plan.rhs_contract_rank() + plan.rhs_open_rank(),
                actual: RHS_CAN_NOUT + RHS_CAN_NIN,
            });
        }

        self.transform_source_into_core(
            rule,
            plan.lhs_transform().clone(),
            plan.lhs_source_conjugate(),
            lhs_core,
            lhs,
        )?;
        self.transform_source_into_core(
            rule,
            plan.rhs_transform().clone(),
            plan.rhs_source_conjugate(),
            rhs_core,
            rhs,
        )?;

        let dst_space = DynamicFusionMapSpace::from_typed(
            dst.fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let lhs_space = DynamicFusionMapSpace::from_typed(
            lhs_core
                .fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let rhs_space = DynamicFusionMapSpace::from_typed(
            rhs_core
                .fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let block_plan = self.resolution_cache.get_or_compile_core_plan(
            rule,
            &dst_space,
            &lhs_space,
            &rhs_space,
            plan.core_axes().as_spec(),
        )?;
        let dst_structure = std::sync::Arc::clone(dst.structure());
        let lhs_structure = std::sync::Arc::clone(lhs_core.structure());
        let rhs_structure = std::sync::Arc::clone(rhs_core.structure());
        block_plan.execute_raw(
            &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                self.contract_backend.transpose_backend(),
            ),
            &mut super::fusion_block::BackendRank2Gemm {
                backend: &mut self.contract_backend,
                workspace: &mut self.contract_workspace,
            },
            &mut self.fusion_block_workspace,
            &dst_structure,
            dst.data_mut(),
            &lhs_structure,
            lhs_core.data(),
            &rhs_structure,
            rhs_core.data(),
            alpha,
            beta,
        )
    }

    fn transform_source_into_core<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &mut self,
        rule: &R,
        operation: crate::TreeTransformOperation,
        source_conjugate: bool,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        let src_fusion = src
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let src_replay_structure = if source_conjugate {
            Arc::clone(adjoint_fusion_space_view(rule, src_fusion)?.subblock_structure())
        } else {
            Arc::clone(src.structure())
        };
        let dst_structure = Arc::clone(dst.structure());
        let structure = self
            .tree_context
            .get_or_compile_tree_pair_structure_with_storage_conjugation(
                rule,
                operation,
                &dst_structure,
                &src_replay_structure,
                source_conjugate,
            )?;
        self.tree_context
            .tree_transform_structure_overwrite_into_raw(
                structure.as_ref(),
                &dst_structure,
                &src_replay_structure,
                dst.data_mut(),
                src.data(),
                D::one(),
            )
    }
}

#[derive(Clone)]
struct PreparedFusionSpaceWitness {
    allocation: Arc<dyn Any + Send + Sync>,
    structure: Arc<BlockStructure>,
}

impl PreparedFusionSpaceWitness {
    fn new<const NOUT: usize, const NIN: usize>(
        fusion_space: &Arc<FusionTensorMapSpace<NOUT, NIN>>,
        structure: &Arc<BlockStructure>,
    ) -> Self {
        Self {
            allocation: Arc::clone(fusion_space) as Arc<dyn Any + Send + Sync>,
            structure: Arc::clone(structure),
        }
    }

    fn matches<const NOUT: usize, const NIN: usize>(
        &self,
        fusion_space: &Arc<FusionTensorMapSpace<NOUT, NIN>>,
        structure: &Arc<BlockStructure>,
    ) -> bool {
        let same_allocation = self
            .allocation
            .downcast_ref::<FusionTensorMapSpace<NOUT, NIN>>()
            .is_some_and(|prepared| std::ptr::eq(prepared, fusion_space.as_ref()));
        let same_structure = Arc::ptr_eq(&self.structure, structure)
            || self.structure.content_id() == structure.content_id()
            || self.structure.as_ref() == structure.as_ref();
        same_allocation && same_structure
    }
}

impl std::fmt::Debug for PreparedFusionSpaceWitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedFusionSpaceWitness")
            .field("structure_content_id", &self.structure.content_id())
            .finish_non_exhaustive()
    }
}

/// Resolved contraction handle: plan-once/execute-many without per-call
/// cache lookups. Created by
/// [`TensorContractFusionExecutionContext::prepare_tensorcontract_fusion`].
#[derive(Clone, Debug)]
pub struct PreparedTensorContractFusion<RuleKey> {
    rule: RuleKey,
    dst_fusion_space: PreparedFusionSpaceWitness,
    lhs_fusion_space: PreparedFusionSpaceWitness,
    rhs_fusion_space: PreparedFusionSpaceWitness,
    resolution: Resolution,
    dynamic_artifact: Option<Arc<super::dynamic::DynamicTreeExecutionArtifact>>,
}
