use std::any::Any;
use std::hash::Hash;
use std::sync::Arc;

use tenet_core::{
    BlockStructure, CheckedFusionAlgebra, CoreError, FusionTensorMapSpace, HostReadableStorage,
    HostWritableStorage, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, Placement,
    TensorMap, TensorStorage,
};

use super::route_host::Stage;
use crate::cache::{
    OperationCachePolicy, TensorContractStructureCache, TensorContractStructureCacheKey,
};
use crate::mode::{
    ContractRequest, ContractStaging, MultiplicityFreePlanningScalar, PlanningAlgebra,
    StagedContraction, TreeStructureSource,
};
use crate::tree_context::TreeTransformExecutionContext;
use crate::tree_transform::{TreeTransformPlanning, TreeTransformRuleCacheKey};
use crate::{
    DenseBlockScalar, DenseRecouplingScalar, DenseTreeTransformOperations, HostTensorOperations,
    OperationError, RecouplingCoefficientAction, ReportsPlacement, TreeTransformBackend, ZeroBytes,
};
use tenet_operations::{
    ContractDestinationInit, OutputAxisOrder, TensorContractSpec, TensorContractSpecOwned,
    TreeTransformWorkspace,
};

use super::backend::TensorContractBackend;
use super::dynamic_space::{
    encoded_layout_primer, BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand,
    LayoutKeyBuilder,
};
#[cfg(test)]
use super::fusion::FusionContractOrientation;
use super::fusion::{
    prepare_tensorcontract_fusion_plan_dyn_prelowered_canonical, FusionContractPlan,
    EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CORE_DST,
};
use super::fusion_block::FusionBlockContractWorkspace;
use super::resolution::try_compile_oriented_storage_contract_plan;
use super::resolution::{
    compile_core_plan, try_compile_oriented_storage_contract_candidate_plan, ContractKind,
    ContractRoute, CopyCRoute, CoreMiss, CoreRoute, ExecCaps, StorageContractResolution,
};
use super::scratch::DynamicFusionScratchWorkspace;
use super::structure::{TensorContractAxisPlan, TensorContractStructure};
use crate::host_scratch::HostScratchBuffer;
use tenet_operations::{TensorContractFusionProfile, TensorContractFusionRoute};

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
            axes: TensorContractSpecOwned::from_axis_vecs(
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
    ephemeral_structure: Option<TensorContractStructure<f64>>,
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
        if !self.structures.policy().stores_entries() {
            self.stats.structure_misses += 1;
            self.ephemeral_structure = Some(TensorContractStructure::compile(
                dst,
                lhs,
                rhs,
                plan_key.axes().as_spec(),
            )?);
            return Ok(self
                .ephemeral_structure
                .as_ref()
                .expect("ephemeral tensor contract structure inserted before replay"));
        }
        let structure_key = TensorContractStructureCacheKey::from_structures(
            plan_key.clone(),
            dst.structure(),
            lhs.structure(),
            rhs.structure(),
        )?;
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
    C = f64,
> where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    BC: TensorContractBackend<D, C>,
{
    tree_context: TreeTransformExecutionContext<D, RuleKey, C, BT>,
    /// One transform workspace per contraction stage, indexed by `Stage`.
    stage_workspaces: [BT::Workspace; 3],
    contract_backend: BC,
    contract_workspace: BC::Workspace,
    fusion_block_workspace: FusionBlockContractWorkspace<D>,
    fusion_scratch: DynamicFusionScratchWorkspace<D>,
    copy_c_scratch: HostScratchBuffer<D>,
    #[cfg(test)]
    last_top_level_resolution_was_core: bool,
    #[cfg(test)]
    last_top_level_resolution_orientation: Option<FusionContractOrientation>,
}

pub type HostTreeFusionExecutionContext<D, RuleKey> = TensorContractFusionExecutionContext<
    D,
    RuleKey,
    HostTensorOperations,
    DenseTreeTransformOperations,
>;

impl<D, RuleKey, C>
    TensorContractFusionExecutionContext<
        D,
        RuleKey,
        DenseTreeTransformOperations,
        DenseTreeTransformOperations,
        C,
    >
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    /// Replays an admitted route (Core, CopyC or DynamicTree) over uniform
    /// member-major Host stacks through the one Host route executor, with
    /// this lane's tree and contract resources and the caller's workspace:
    /// `dst = contract(lhs, rhs)` per member. `init` is
    /// [`ContractDestinationInit::Zeroed`] only when `dst`'s inactive blocks
    /// are already zero, else `Axpby(0)`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_storage_contract_members_host(
        &mut self,
        resolution: &StorageContractResolution<C>,
        dst: (&Arc<BlockStructure>, &mut [D]),
        lhs: (&Arc<BlockStructure>, &[D]),
        rhs: (&Arc<BlockStructure>, &[D]),
        workspace: &mut super::route_host::HostContractMembersWorkspace<D, C>,
        members: usize,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        C: std::ops::Neg<Output = C>,
    {
        let (tree_backend, _) = self.tree_context.backend_workspace_mut();
        super::route_host::execute_members_host(
            tree_backend,
            &mut self.contract_backend,
            &mut self.contract_workspace,
            &mut self.fusion_block_workspace,
            super::route_host::MemberRoute::Resolution {
                resolution,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst.0,
            workspace,
            dst.1,
            lhs.1,
            rhs.1,
            members,
            init,
        )
    }
}

impl<D, RuleKey, BT, BC, C> TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    BC: TensorContractBackend<D, C>,
{
    pub fn with_parts(
        tree_context: TreeTransformExecutionContext<D, RuleKey, C, BT>,
        contract_backend: BC,
        contract_workspace: BC::Workspace,
    ) -> Self
    where
        BT::Workspace: Default,
    {
        Self {
            tree_context,
            stage_workspaces: Default::default(),
            contract_backend,
            contract_workspace,
            fusion_block_workspace: FusionBlockContractWorkspace::default(),
            fusion_scratch: DynamicFusionScratchWorkspace::default(),
            copy_c_scratch: HostScratchBuffer::default(),
            #[cfg(test)]
            last_top_level_resolution_was_core: false,
            #[cfg(test)]
            last_top_level_resolution_orientation: None,
        }
    }

    #[inline]
    pub fn tree_context(&self) -> &TreeTransformExecutionContext<D, RuleKey, C, BT> {
        &self.tree_context
    }

    #[inline]
    pub fn tree_context_mut(&mut self) -> &mut TreeTransformExecutionContext<D, RuleKey, C, BT> {
        &mut self.tree_context
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

    /// Lends the grow-only Host buffer that holds TensorKit
    /// `blas_contract!`'s `copyC` temporary: the default-output contraction
    /// an eager caller then permutes into its own result. It lives with this
    /// context's other execution scratch, so it is pooled and bounded the same
    /// way (largest temporary seen, per idle context). Return it with
    /// [`Self::restore_copy_c_scratch`]; a buffer lost to an error is simply
    /// reallocated by the next caller.
    #[doc(hidden)]
    pub fn take_copy_c_scratch(&mut self) -> HostScratchBuffer<D> {
        std::mem::take(&mut self.copy_c_scratch)
    }

    #[doc(hidden)]
    pub fn restore_copy_c_scratch(&mut self, scratch: HostScratchBuffer<D>) {
        self.copy_c_scratch = scratch;
    }

    /// Logical length of the retained `copyC` buffer.
    #[doc(hidden)]
    pub fn copy_c_scratch_len(&self) -> usize {
        self.copy_c_scratch.len()
    }

    #[cfg(test)]
    pub(crate) fn eager_coefficient_pack_builds(&self) -> [usize; 3]
    where
        BT: TreeTransformBackend<D, C, Workspace = TreeTransformWorkspace<D>>,
    {
        std::array::from_fn(|stage| self.stage_workspaces[stage].coefficient_pack_builds())
    }

    #[cfg(test)]
    pub(crate) fn stage_workspace_retained_bytes(&self) -> usize
    where
        BT: TreeTransformBackend<D, C, Workspace = TreeTransformWorkspace<D>>,
    {
        self.stage_workspaces
            .iter()
            .map(TreeTransformWorkspace::retained_bytes)
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn last_resolution_is_core(&self) -> bool {
        self.last_top_level_resolution_was_core
    }

    #[cfg(test)]
    pub(crate) fn last_resolution_orientation(&self) -> Option<FusionContractOrientation> {
        self.last_top_level_resolution_orientation
    }

    #[cfg(test)]
    pub(super) fn record_contract_route(&mut self, resolution: &StorageContractResolution<C>) {
        self.last_top_level_resolution_was_core =
            matches!(resolution.route, ContractRoute::Core { .. });
        self.last_top_level_resolution_orientation = match &resolution.route {
            ContractRoute::DynamicTree(artifact) => Some(artifact.orientation()),
            ContractRoute::Core { swapped: true, .. } => Some(FusionContractOrientation::RhsLhs),
            ContractRoute::Core { swapped: false, .. } | ContractRoute::CopyC(_) => None,
        };
    }

    pub fn into_parts(
        self,
    ) -> (
        TreeTransformExecutionContext<D, RuleKey, C, BT>,
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

impl<D, RuleKey, BT, BC, C> TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C, Workspace = TreeTransformWorkspace<D>>,
    BC: TensorContractBackend<D, C>,
{
    /// Allocated Host capacity retained by fusion lhs/rhs/destination scratch,
    /// the `copyC` temporary, and the tree-transform workspaces (the tree
    /// context's and one per contraction stage, coefficient packs included).
    #[doc(hidden)]
    pub fn retained_host_scratch_bytes(&self) -> usize {
        std::iter::once(self.tree_context.workspace())
            .chain(&self.stage_workspaces)
            .map(TreeTransformWorkspace::retained_bytes)
            .fold(
                self.fusion_scratch.retained_bytes().saturating_add(
                    self.copy_c_scratch
                        .capacity()
                        .saturating_mul(std::mem::size_of::<D>()),
                ),
                usize::saturating_add,
            )
    }

    /// Releases this context's retained Host fusion, `copyC` and
    /// tree-transform buffers.
    #[doc(hidden)]
    pub fn trim_host_scratch(&mut self) {
        self.fusion_scratch.clear();
        self.copy_c_scratch = HostScratchBuffer::default();
        *self.tree_context.backend_workspace_mut().1 = TreeTransformWorkspace::default();
        self.stage_workspaces = Default::default();
    }
}

impl<D, RuleKey, BT, BC, C> TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C> + ReportsPlacement,
    BT::Workspace: ReportsPlacement,
    BC: TensorContractBackend<D, C> + ReportsPlacement,
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

impl<D, RuleKey, BT, BC, C> TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    BC: TensorContractBackend<D, C>,
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

impl<D, RuleKey, BT, BC, C> Default for TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C> + Default,
    BC: TensorContractBackend<D, C> + Default,
    BT::Workspace: Default,
    BC::Workspace: Default,
{
    fn default() -> Self {
        Self::new(BT::default(), BC::default())
    }
}

impl<D, RuleKey, BT, BC, C> TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    BC: TensorContractBackend<D, C>,
{
    #[expect(
        clippy::too_many_arguments,
        reason = "the context already owns execution resources, leaving the public contraction operands, spec, and alpha/beta explicit"
    )]
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
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
        DDst: HostWritableStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>,
    {
        let dst_space = typed_dynamic_space(dst)?;
        let lhs_space = typed_dynamic_space(lhs)?;
        let rhs_space = typed_dynamic_space(rhs)?;
        let lowered = lower_legacy_axes(&lhs_space, &rhs_space, axes)?;
        let axes = lowered.as_spec();
        let output_axes = spec_output_axes(axes, dst_space.rank());
        self.tensorcontract_planned_raw_into::<MultiplicityFreeAdmissionMode, R>(
            &mut (),
            PlanTarget {
                rule,
                space: &dst_space,
                authority: encoded_layout_primer::<R>,
            },
            dst.data_mut(),
            (legacy_operand(&lhs_space, axes.lhs_conjugate()), lhs.data()),
            (legacy_operand(&rhs_space, axes.rhs_conjugate()), rhs.data()),
            PlannedContraction::Contract((
                axes.lhs_contracting_axes(),
                axes.rhs_contracting_axes(),
                &output_axes,
            )),
            alpha,
            ContractDestinationInit::Axpby(beta),
        )
        .map(drop)
    }

    /// Dynamic-rank `tensorcontract!`: same eager route selection and gates
    /// as [`Self::tensorcontract_fusion_into`], operating on
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
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.tensorcontract_fusion_dyn_into_with_init(
            dst_space,
            dst_data,
            lhs_space,
            lhs_data,
            rhs_space,
            rhs_data,
            axes,
            alpha,
            ContractDestinationInit::Axpby(beta),
        )
    }

    /// [`Self::tensorcontract_fusion_dyn_into`] with the destination
    /// initialisation made explicit: an owned output born all-zero passes
    /// [`ContractDestinationInit::Zeroed`] so its inactive blocks are never
    /// touched.
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_fusion_dyn_into_with_init<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        if !axes.lhs_conjugate() && !axes.rhs_conjugate() {
            return self.tensorcontract_fusion_dyn_prelowered_into_core(
                dst_space,
                dst_data,
                FusionOperand::direct(lhs_space.space()),
                lhs_data,
                FusionOperand::direct(rhs_space.space()),
                rhs_data,
                axes,
                alpha,
                init,
            );
        }
        let lowered = lower_legacy_axes(lhs_space.space(), rhs_space.space(), axes)?;
        self.tensorcontract_fusion_dyn_prelowered_into_core(
            dst_space,
            dst_data,
            legacy_operand(lhs_space.space(), axes.lhs_conjugate()),
            lhs_data,
            legacy_operand(rhs_space.space(), axes.rhs_conjugate()),
            rhs_data,
            lowered.as_spec(),
            alpha,
            init,
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
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
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        let lowered = lower_legacy_axes(lhs_space, rhs_space, axes)?;
        let axes = lowered.as_spec();
        let output_axes = spec_output_axes(axes, dst_space.rank());
        self.tensorcontract_planned_raw_into::<MultiplicityFreeAdmissionMode, R>(
            &mut (),
            PlanTarget {
                rule,
                space: dst_space,
                authority: encoded_layout_primer::<R>,
            },
            dst_data,
            (legacy_operand(lhs_space, axes.lhs_conjugate()), lhs_data),
            (legacy_operand(rhs_space, axes.rhs_conjugate()), rhs_data),
            PlannedContraction::Contract((
                axes.lhs_contracting_axes(),
                axes.rhs_contracting_axes(),
                &output_axes,
            )),
            alpha,
            ContractDestinationInit::Axpby(beta),
        )
        .map(drop)
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
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.tensorcontract_fusion_dyn_prelowered_into_with_init(
            dst_space,
            dst_data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            axes,
            alpha,
            ContractDestinationInit::Axpby(beta),
        )
    }

    /// [`Self::tensorcontract_fusion_dyn_prelowered_into`] with the
    /// destination initialisation made explicit.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_fusion_dyn_prelowered_into_with_init<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs: FusionOperand<'_>,
        lhs_data: &[D],
        rhs: FusionOperand<'_>,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.tensorcontract_fusion_dyn_prelowered_into_core(
            dst_space, dst_data, lhs, lhs_data, rhs, rhs_data, axes, alpha, init,
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
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        // The planner rejects operand flags that disagree with `axes`.
        if axes.lhs_conjugate() != lhs.storage_conjugate()
            || axes.rhs_conjugate() != rhs.storage_conjugate()
        {
            return Err(OperationError::InvalidArgument {
                message: "prelowered operand flags must match the contraction request",
            });
        }
        let output_axes = spec_output_axes(axes, dst_space.space().rank());
        self.tensorcontract_planned_into(
            dst_space,
            dst_data,
            (lhs, lhs_data),
            (rhs, rhs_data),
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &output_axes,
            alpha,
            init,
        )
    }

    /// Host eager contraction: [`Self::plan_contract`] for the Host eager
    /// executor, then its replay, `dst = alpha * contract(lhs, rhs) + beta *
    /// dst` (`init`). The one route authority Host eager, the device and
    /// `ContractPlan` share; a lazy adjoint is a storage-conjugate operand.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_planned_into<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        (lhs, lhs_data): (FusionOperand<'_>, &[D]),
        (rhs, rhs_data): (FusionOperand<'_>, &[D]),
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.tensorcontract_planned_raw_into::<MultiplicityFreeAdmissionMode, R>(
            &mut (),
            PlanTarget::bound(dst_space),
            dst_data,
            (lhs, lhs_data),
            (rhs, rhs_data),
            PlannedContraction::Contract((lhs_axes, rhs_axes, output_axes)),
            alpha,
            init,
        )
        .map(drop)
    }

    /// The Host plan + execute path of every contraction and composition
    /// entry in either admission mode `M`: [`plan_contract_in`] (under the
    /// context's planning state or the call's transaction `txn`) or
    /// [`plan_compose_in`], then the route replay `dst = alpha · op +
    /// init(dst)` into an existing destination. Returns the replayed route,
    /// which a checked call's commit publishes over.
    #[allow(clippy::too_many_arguments)]
    fn tensorcontract_planned_raw_into<M, R>(
        &mut self,
        txn: &mut M::ContractTxn,
        target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
        dst_data: &mut [D],
        (lhs, lhs_data): (FusionOperand<'_>, &[D]),
        (rhs, rhs_data): (FusionOperand<'_>, &[D]),
        planned: PlannedContraction<'_>,
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<StorageContractResolution<C>, M::Error>
    where
        M: PlanningAlgebra<R, Scalar = C>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        let resolution = match planned {
            PlannedContraction::Contract(axes) => {
                plan_contract_in::<super::resolution::HostEagerExecutor, M, R>(
                    M::contract_cache(self.tree_context.planning(), txn),
                    target,
                    lhs,
                    rhs,
                    axes,
                    None,
                )?
            }
            PlannedContraction::Compose => {
                plan_compose_in::<super::resolution::HostEagerExecutor, M, R>(target, lhs, rhs)?
            }
        };
        #[cfg(test)]
        self.record_contract_route(&resolution);
        self.execute_contract_route_host(
            &resolution,
            target.space.structure(),
            dst_data,
            (lhs.storage_space().structure(), lhs_data),
            (rhs.storage_space().structure(), rhs_data),
            alpha,
            init,
        )?;
        Ok(resolution)
    }

    /// Categorical map composition on the coupled-sector block matrices,
    /// TensorKit `mul!`, into an existing destination: `dst = alpha · (lhs ∘
    /// rhs) + init(dst)`, the composition sibling of
    /// [`Self::tensorcontract_planned_into`]. A's domain against B's
    /// codomain, no fermionic supertrace twist, every braiding style. It
    /// plans through [`plan_compose`], as `ComposePlan` does, for the Host
    /// eager executor, whose irregular core packs a non-canonical tiling.
    /// Lazy adjoints stay storage-conjugate operands; neither is
    /// materialized.
    #[doc(hidden)]
    pub fn tensorcompose_planned_into<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        (lhs, lhs_data): (FusionOperand<'_>, &[D]),
        (rhs, rhs_data): (FusionOperand<'_>, &[D]),
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.tensorcontract_planned_raw_into::<MultiplicityFreeAdmissionMode, R>(
            &mut (),
            PlanTarget::bound(dst_space),
            dst_data,
            (lhs, lhs_data),
            (rhs, rhs_data),
            PlannedContraction::Compose,
            alpha,
            init,
        )
        .map(drop)
    }

    /// The owned multiplicity-free contraction `lhs·rhs` into a fresh
    /// payload, split after its first `codomain_rank` output axes. Each
    /// operand is its logical space, its storage operand (a lazy adjoint is
    /// storage-conjugate) and its storage payload.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `tenet`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn tensorcontract_multiplicity_free_in<R>(
        &mut self,
        (lhs_authority, lhs, lhs_data): (&BoundDynamicFusionMapSpace<R>, FusionOperand<'_>, &[D]),
        (rhs_authority, rhs, rhs_data): (&BoundDynamicFusionMapSpace<R>, FusionOperand<'_>, &[D]),
        (lhs_axes, rhs_axes, output_order): (&[usize], &[usize], OutputAxisOrder<'_>),
        codomain_rank: usize,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>
            + CheckedFusionAlgebra
            + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ZeroBytes,
    {
        self.tensorcontract_in::<MultiplicityFreeAdmissionMode, R>(
            (lhs_authority, lhs, lhs_data),
            (rhs_authority, rhs, rhs_data),
            ContractRequest::Contract {
                axes: TensorContractSpec::new_with_conjugation(
                    lhs_axes,
                    rhs_axes,
                    output_order,
                    lhs.storage_conjugate(),
                    rhs.storage_conjugate(),
                ),
                codomain_rank,
            },
        )
    }

    /// The owned multiplicity-free composition `lhs ∘ rhs` (TensorKit
    /// `mul!`) into a fresh payload; operands as in
    /// [`Self::tensorcontract_multiplicity_free_in`].
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `tenet`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn tensorcompose_multiplicity_free_in<R>(
        &mut self,
        lhs: (&BoundDynamicFusionMapSpace<R>, FusionOperand<'_>, &[D]),
        rhs: (&BoundDynamicFusionMapSpace<R>, FusionOperand<'_>, &[D]),
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>
            + CheckedFusionAlgebra
            + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ZeroBytes,
    {
        self.tensorcontract_in::<MultiplicityFreeAdmissionMode, R>(
            lhs,
            rhs,
            ContractRequest::Compose,
        )
    }

    /// The one owned contraction and composition of both modes: stage the
    /// destination (multiplicity-free derives it; checked Generic admits the
    /// operands and stages it), plan and replay into a fresh zeroed payload
    /// through [`Self::tensorcontract_planned_raw_into`], then commit and
    /// publish (#2063).
    #[allow(clippy::type_complexity)]
    pub(super) fn tensorcontract_in<M, R>(
        &mut self,
        (lhs_authority, lhs, lhs_data): (&BoundDynamicFusionMapSpace<R>, FusionOperand<'_>, &[D]),
        (rhs_authority, rhs, rhs_data): (&BoundDynamicFusionMapSpace<R>, FusionOperand<'_>, &[D]),
        request: ContractRequest<'_>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), M::Error>
    where
        M: ContractStaging<R, Scalar = C>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ZeroBytes,
    {
        let StagedContraction {
            stage,
            authority,
            mut txn,
            len,
        } = M::stage_contraction(
            (lhs_authority, lhs, lhs_data.len()),
            (rhs_authority, rhs, rhs_data.len()),
            request,
        )?;
        let (rule, space) = M::contract_destination(&stage, lhs_authority);
        let output_axes;
        let planned = match request {
            ContractRequest::Contract { axes, .. } => {
                output_axes = spec_output_axes(axes, space.rank());
                PlannedContraction::Contract((
                    axes.lhs_contracting_axes(),
                    axes.rhs_contracting_axes(),
                    &output_axes,
                ))
            }
            ContractRequest::Compose => PlannedContraction::Compose,
        };
        let mut data = crate::zeroed_payload(len);
        let resolution = self.tensorcontract_planned_raw_into::<M, R>(
            &mut txn,
            PlanTarget {
                rule,
                space,
                authority,
            },
            &mut data,
            (lhs, lhs_data),
            (rhs, rhs_data),
            planned,
            D::one(),
            ContractDestinationInit::Zeroed,
        )?;
        let destination = M::commit_contraction(lhs_authority, stage, txn, &resolution)?;
        Ok((destination, data))
    }

    /// The contraction planner: TensorKit `contract!`
    /// (`tensoroperations.jl:314-357` @cfaa073) under `contract_memcost`
    /// (`:358-368`), for an executor with the capabilities `X`. Every
    /// contraction route — Host eager, the device, `ContractPlan`, prepared and
    /// profiled — is chosen here.
    ///
    /// The ladder: the canonical fully-direct core
    /// ([`try_compile_storage_contract_core_route`]), then
    /// [`Self::plan_contract_beyond_core`] — TensorKit's `copyC`, then the
    /// `DynamicTree` artifact. A caller that wants the canonical route
    /// without taking a context lock calls the two halves itself, in that
    /// order. The requested order's zero-copy candidates are walked once.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn plan_contract<X: ExecCaps, R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.plan_contract_raw::<X, R>(
            PlanTarget::bound(dst_space),
            lhs,
            rhs,
            (lhs_axes, rhs_axes, output_axes),
            None,
        )
    }

    /// [`Self::plan_contract`] for a destination given as provider, space and
    /// layout primer, not bound: the static-rank entries plan with their
    /// caller's rule without binding a space per call. `profile` attributes
    /// the planning time (see the [`TensorContractFusionProfile`] fields).
    pub(crate) fn plan_contract_raw<X: ExecCaps, R>(
        &mut self,
        target: PlanTarget<'_, R>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        axes: (&[usize], &[usize], &[usize]),
        profile: Option<&mut TensorContractFusionProfile>,
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        plan_contract_in::<X, MultiplicityFreeAdmissionMode, R>(
            self.tree_context.planning(),
            target,
            lhs,
            rhs,
            axes,
            profile,
        )
    }

    /// [`Self::plan_contract`] after its canonical core declined (`miss`, from
    /// [`try_compile_storage_contract_core_route`] on the same request):
    /// TensorKit's `copyC` (`blas_contract!`, `tensoroperations.jl:436-446`
    /// @cfaa073) when [`zero_copy_contract_order_for_output_permute`] takes it
    /// — a zero-copy core into a temporary in its own default output, then one
    /// permute — else the `DynamicTree` artifact.
    ///
    /// Why the core precedes `copyC`: a contraction that has the canonical
    /// core needs no output permute (`zero_copy_contract_order_for_output_permute`
    /// returns `None` for it), so TensorKit's memcost never charges it `dim(C)`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn plan_contract_beyond_core<X: ExecCaps, R>(
        &mut self,
        miss: CoreMiss,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        plan_contract_beyond_core_in::<X, MultiplicityFreeAdmissionMode, R>(
            self.tree_context.planning(),
            miss,
            PlanTarget::bound(dst_space),
            lhs,
            rhs,
            (lhs_axes, rhs_axes, output_axes),
            None,
        )
    }

    /// Test-only: [`try_compile_storage_contract_core_route`] followed, on a
    /// miss, by [`Self::compile_storage_contract_dynamic_tree`] — the
    /// planner's ladder without `CopyC`, for gates that pin those two routes.
    /// Production planning is [`Self::plan_contract`].
    #[cfg(test)]
    pub(crate) fn compile_storage_contract_resolution<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        axes: TensorContractSpec<'_>,
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        match try_compile_storage_contract_core_route::<super::resolution::DirectCoreExecutor, R>(
            dst_space, lhs, rhs, axes,
        )? {
            CoreRoute::Hit(core) => Ok(core),
            CoreRoute::Miss(_) => {
                self.compile_storage_contract_dynamic_tree(dst_space, lhs, rhs, axes)
            }
        }
    }

    /// The last rung of [`Self::plan_contract`]: the
    /// Host `DynamicTree` artifact for a contraction that
    /// [`try_compile_storage_contract_core_route`] declined, compiled by the
    /// same entry the Host contraction uses for this representation — the
    /// owned compiler for two owned operands, the prelowered compiler when
    /// either is a lazy adjoint — with this context's space cache and tree
    /// context, so its transform structures are the Arcs a device executor's
    /// prepared cache is keyed by.
    ///
    /// Host eager and the device plan the same route (#1858); the one
    /// executor difference: owned core geometry over a non-canonical storage
    /// layout (an expert tiling, #1517) is a core with pack/scatter on the
    /// Host (`IRREGULAR_CORE`) and this artifact on the device.
    ///
    /// A fermionic contraction twist travels in the artifact, both as the
    /// Host's in-place scale actions and as the sorted per-block scale list
    /// the device folds into the twisted operand's source transform. A
    /// canonical contraction whose twist varies within one coupled sector
    /// reaches here (the core declines it); a uniform one never does.
    ///
    /// Validates the request itself (provider against the spaces, operand
    /// flags against `axes`), so it is sound to call without the first half.
    #[doc(hidden)]
    pub fn compile_storage_contract_dynamic_tree<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        axes: TensorContractSpec<'_>,
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        compile_dynamic_tree_in::<MultiplicityFreeAdmissionMode, R>(
            self.tree_context.planning(),
            PlanTarget::bound(dst_space),
            lhs,
            rhs,
            axes,
            false,
            None,
        )
    }

    /// Replays a planned route ([`Self::plan_contract`] with
    /// [`HostEagerExecutor`](super::resolution::HostEagerExecutor)) on Host
    /// slices through the one Host route executor at one member:
    /// `dst = alpha * contract(lhs, rhs) + beta * dst`.
    ///
    /// `CopyC` is TensorKit `blas_contract!`'s `copyC`
    /// (`tensoroperations.jl:436-446` @cfaa073): `mul!(Cnew, A, B)` into the
    /// lane's pooled temporary, then `tensoradd!(C, Cnew, pAB, false, α, β)`.
    /// TensorKit allocates `Cnew` per call (`tensoralloc_add(..., Val(true))`);
    /// the pooled buffer plays that role, so a warm call allocates no
    /// output-sized temporary. Stale pooled values are never read: eager
    /// scratch carries no core slot, so the core writes the temporary with a
    /// strong-zero `beta = 0`, inactive blocks included.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_contract_route_host(
        &mut self,
        resolution: &StorageContractResolution<C>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs: (&Arc<BlockStructure>, &[D]),
        rhs: (&Arc<BlockStructure>, &[D]),
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.execute_route_host(
            resolution,
            (dst_structure, dst_data),
            lhs,
            rhs,
            alpha,
            init,
            None,
        )
    }

    /// [`Self::execute_contract_route_host`] with each stage timed into
    /// `profile`; `CopyC`'s permute counts as the output transform.
    #[allow(clippy::too_many_arguments)]
    fn execute_contract_route_host_profiled(
        &mut self,
        resolution: &StorageContractResolution<C>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs: (&Arc<BlockStructure>, &[D]),
        rhs: (&Arc<BlockStructure>, &[D]),
        alpha: D,
        beta: D,
        profile: &mut TensorContractFusionProfile,
    ) -> Result<(), OperationError>
    where
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        profile.route = match &resolution.route {
            ContractRoute::Core { .. } => TensorContractFusionRoute::CoreFusionBlocks,
            ContractRoute::DynamicTree(_) => TensorContractFusionRoute::DynamicTreeCore,
            ContractRoute::CopyC(_) => TensorContractFusionRoute::CopyC,
        };
        self.execute_route_host(
            resolution,
            (dst_structure, dst_data),
            lhs,
            rhs,
            alpha,
            ContractDestinationInit::Axpby(beta),
            Some(profile),
        )
    }

    /// The eager arm of every route: the one route executor at one member,
    /// over this context's tree workspace and scratch (the pooled `copyC`
    /// temporary as a CopyC core destination).
    #[allow(clippy::too_many_arguments)]
    fn execute_route_host(
        &mut self,
        resolution: &StorageContractResolution<C>,
        dst: (&Arc<BlockStructure>, &mut [D]),
        lhs: (&Arc<BlockStructure>, &[D]),
        rhs: (&Arc<BlockStructure>, &[D]),
        alpha: D,
        init: ContractDestinationInit<D>,
        profile: Option<&mut TensorContractFusionProfile>,
    ) -> Result<(), OperationError>
    where
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        let Self {
            tree_context,
            stage_workspaces,
            contract_backend,
            contract_workspace,
            fusion_block_workspace,
            fusion_scratch,
            copy_c_scratch,
            ..
        } = self;
        let mut scratch = fusion_scratch.route_scratch();
        if resolution.copy_c().is_some() {
            scratch.core_dst = copy_c_scratch;
        }
        super::route_host::execute_route_host(
            &mut super::route_host::EagerTreeStage {
                backend: tree_context.backend_mut(),
                workspaces: stage_workspaces,
            },
            &mut super::fusion_block::BackendRank2Gemm::new(contract_backend, contract_workspace),
            fusion_block_workspace,
            scratch,
            resolution,
            dst,
            lhs,
            rhs,
            1,
            alpha,
            init,
            profile,
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
    ) -> Result<PreparedTensorContractFusion<RuleKey, C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
        DDst: HostWritableStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>,
    {
        let missing = || OperationError::Core(CoreError::MissingFusionSpace);
        let dst_fusion = dst.fusion_space().ok_or_else(missing)?;
        let lhs_fusion = lhs.fusion_space().ok_or_else(missing)?;
        let rhs_fusion = rhs.fusion_space().ok_or_else(missing)?;
        let dst_space = DynamicFusionMapSpace::from_typed(dst_fusion);
        let lhs_space = DynamicFusionMapSpace::from_typed(lhs_fusion);
        let rhs_space = DynamicFusionMapSpace::from_typed(rhs_fusion);
        let lowered = lower_legacy_axes(&lhs_space, &rhs_space, axes)?;
        let axes = lowered.as_spec();
        let output_axes = spec_output_axes(axes, dst_space.rank());
        let resolution = self.plan_contract_raw::<super::resolution::HostEagerExecutor, R>(
            PlanTarget {
                rule,
                space: &dst_space,
                authority: encoded_layout_primer::<R>,
            },
            legacy_operand(&lhs_space, axes.lhs_conjugate()),
            legacy_operand(&rhs_space, axes.rhs_conjugate()),
            (
                axes.lhs_contracting_axes(),
                axes.rhs_contracting_axes(),
                &output_axes,
            ),
            None,
        )?;
        Ok(PreparedTensorContractFusion {
            rule: rule.tree_transform_rule_cache_key(),
            dst_fusion_space: PreparedFusionSpaceWitness::new(dst_fusion, dst.structure()),
            lhs_fusion_space: PreparedFusionSpaceWitness::new(lhs_fusion, lhs.structure()),
            rhs_fusion_space: PreparedFusionSpaceWitness::new(rhs_fusion, rhs.structure()),
            resolution,
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
        prepared: &PreparedTensorContractFusion<RuleKey, C>,
        rule: &R,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
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
        self.execute_contract_route_host(
            &prepared.resolution,
            &dst_structure,
            dst.data_mut(),
            (lhs.structure(), lhs.data()),
            (rhs.structure(), rhs.data()),
            alpha,
            ContractDestinationInit::Axpby(beta),
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
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
        DDst: HostWritableStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>,
    {
        let total_start = std::time::Instant::now();

        let start = std::time::Instant::now();
        let dst_space = typed_dynamic_space(dst)?;
        let lhs_space = typed_dynamic_space(lhs)?;
        let rhs_space = typed_dynamic_space(rhs)?;
        let lowered = lower_legacy_axes(&lhs_space, &rhs_space, axes)?;
        let axes = lowered.as_spec();
        let output_axes = spec_output_axes(axes, dst_space.rank());
        let (lhs_operand, rhs_operand) = (
            legacy_operand(&lhs_space, axes.lhs_conjugate()),
            legacy_operand(&rhs_space, axes.rhs_conjugate()),
        );
        profile.typed_space_setup += start.elapsed();

        // The eager planner and executor, timed: the profile reports the
        // route the unprofiled entry takes.
        let resolution = self.plan_contract_raw::<super::resolution::HostEagerExecutor, R>(
            PlanTarget {
                rule,
                space: &dst_space,
                authority: encoded_layout_primer::<R>,
            },
            lhs_operand,
            rhs_operand,
            (
                axes.lhs_contracting_axes(),
                axes.rhs_contracting_axes(),
                &output_axes,
            ),
            Some(profile),
        )?;
        #[cfg(test)]
        self.record_contract_route(&resolution);
        let result = self.execute_contract_route_host_profiled(
            &resolution,
            dst_space.structure(),
            dst.data_mut(),
            (lhs_space.structure(), lhs.data()),
            (rhs_space.structure(), rhs.data()),
            alpha,
            beta,
            profile,
        );
        profile.total += total_start.elapsed();
        result
    }

    /// Executes a caller-supplied source-transform plan and compiles its core
    /// block plan eagerly. Use [`Self::prepare_tensorcontract_fusion`] and
    /// [`PreparedTensorContractFusion`] when the complete contraction should
    /// be compiled once and replayed without lookups.
    #[expect(
        clippy::too_many_arguments,
        reason = "the context owns execution resources while FusionContractPlan and caller-owned core tensors retain separate authority"
    )]
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
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        plan.require_forward_scratch()?;
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

    /// Executes a caller-supplied source/output-transform plan and compiles its
    /// core block plan eagerly. Use [`Self::prepare_tensorcontract_fusion`] and
    /// [`PreparedTensorContractFusion`] for complete compile-once replay.
    #[expect(
        clippy::too_many_arguments,
        reason = "the context owns execution resources while the prepared plan's source, core, and output tensors remain explicit"
    )]
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
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        plan.require_forward_scratch()?;
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
        let structure = self.tree_context.tree_structure(
            rule,
            plan.output_transform(),
            dst.structure(),
            TreeStructureSource::Stored {
                structure: core_dst.structure(),
                storage_conjugate: false,
            },
        )?;
        self.tree_context
            .backend_mut()
            .tree_transform_structure_into(
                &mut self.stage_workspaces[Stage::Output as usize],
                &structure,
                dst,
                core_dst,
                D::one(),
                beta,
            )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the private prepared replay keeps each source beside its caller-owned transformed scratch tensor"
    )]
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
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
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
            Stage::Lhs,
            plan.lhs_transform().clone(),
            plan.lhs_source_conjugate(),
            lhs_core,
            lhs,
        )?;
        self.transform_source_into_core(
            rule,
            Stage::Rhs,
            plan.rhs_transform().clone(),
            plan.rhs_source_conjugate(),
            rhs_core,
            rhs,
        )?;

        let dst_space = DynamicFusionMapSpace::from_typed(
            dst.fusion_space()
                .ok_or_else(|| OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let lhs_space = DynamicFusionMapSpace::from_typed(
            lhs_core
                .fusion_space()
                .ok_or_else(|| OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let rhs_space = DynamicFusionMapSpace::from_typed(
            rhs_core
                .fusion_space()
                .ok_or_else(|| OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let block_plan = compile_core_plan(
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
            &mut crate::StridedHostKernelAdapter::default(),
            &mut super::fusion_block::BackendRank2Gemm::new(
                &mut self.contract_backend,
                &mut self.contract_workspace,
            ),
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
        stage: Stage,
        operation: crate::TreeTransformOperation,
        source_conjugate: bool,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        let src_fusion = src
            .fusion_space()
            .ok_or_else(|| OperationError::Core(CoreError::MissingFusionSpace))?;
        let dst_structure = Arc::clone(dst.structure());
        // A conjugated source is read through its parent's storage under the
        // oriented key (parent content, orientation, basis order), as the
        // context `tensoradd` does: a per-call adjoint view would key every
        // call on a fresh content id that only evicts live transformers.
        let structure = if source_conjugate {
            let storage_src = DynamicFusionMapSpace::from_typed(src_fusion);
            let oriented_src = FusionOperand::prepare_storage_ordered_adjoint(&storage_src, rule)?;
            self.tree_context.tree_structure(
                rule,
                &operation,
                &dst_structure,
                TreeStructureSource::Oriented(&oriented_src),
            )?
        } else {
            self.tree_context.tree_structure(
                rule,
                &operation,
                &dst_structure,
                TreeStructureSource::Stored {
                    structure: src.structure(),
                    storage_conjugate: false,
                },
            )?
        };
        crate::tree_context::replay_structure_overwrite(
            self.tree_context.backend_mut(),
            &mut self.stage_workspaces[stage as usize],
            &structure,
            &dst_structure,
            src.structure(),
            dst.data_mut(),
            src.data(),
            D::one(),
            &[],
            None,
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
pub struct PreparedTensorContractFusion<RuleKey, C = f64> {
    rule: RuleKey,
    dst_fusion_space: PreparedFusionSpaceWitness,
    lhs_fusion_space: PreparedFusionSpaceWitness,
    rhs_fusion_space: PreparedFusionSpaceWitness,
    resolution: StorageContractResolution<C>,
}

/// The fusion space of a typed tensor, as the planner's dynamic space.
fn typed_dynamic_space<D, const NOUT: usize, const NIN: usize, S, DD>(
    tensor: &TensorMap<D, NOUT, NIN, S, DD>,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    DD: tenet_core::TensorStorage<D>,
{
    tensor
        .fusion_space()
        .map(|space| DynamicFusionMapSpace::from_typed(space))
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))
}

/// A legacy conjugation flag on an owned space requests its categorical
/// adjoint, with `axes` indexing the stored tensor's legs (TensorOperations'
/// `conjA`): the lazy-adjoint operand of the space ([`legacy_operand`]), its
/// contracted axes mapped onto the adjoint's legs here. Without a flag the
/// axes are unchanged.
fn lower_legacy_axes(
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<crate::lowering::LoweredTensorContractSpec, OperationError> {
    crate::lowering::lower_tensorcontract_adjoint_axes(
        lhs.nout(),
        lhs.nin(),
        rhs.nout(),
        rhs.nin(),
        axes,
    )
}

/// The operand an owned space and a legacy conjugation flag request: the flag
/// asks for the categorical adjoint, which is the lazy adjoint of the space.
fn legacy_operand(space: &DynamicFusionMapSpace, conjugate: bool) -> FusionOperand<'_> {
    if conjugate {
        FusionOperand::adjoint(space)
    } else {
        FusionOperand::direct(space)
    }
}

/// What [`TensorContractFusionExecutionContext::tensorcontract_planned_raw_into`]
/// plans: a contraction over `(lhs, rhs, output)` axes, or a composition.
#[derive(Clone, Copy)]
pub(crate) enum PlannedContraction<'s> {
    Contract((&'s [usize], &'s [usize], &'s [usize])),
    Compose,
}

/// The output permutation of `axes` over a rank-`rank` destination.
fn spec_output_axes(axes: TensorContractSpec<'_>, rank: usize) -> smallvec::SmallVec<[usize; 16]> {
    match axes.output_permutation() {
        tenet_operations::OutputAxisOrder::Axes(axes) => axes.iter().copied().collect(),
        tenet_operations::OutputAxisOrder::Identity => (0..rank).collect(),
    }
}

/// The destination a contraction is planned for: its provider, its space and
/// the mode's authority over spaces derived from it (the multiplicity-free
/// layout primer by default). Bound spaces carry all three; the static-rank
/// entries pass their caller's rule.
pub(crate) struct PlanTarget<'a, R, A = LayoutKeyBuilder<R>> {
    pub(crate) rule: &'a R,
    pub(crate) space: &'a DynamicFusionMapSpace,
    pub(crate) authority: A,
}

// Why manual: a derive would bound `R: Copy`.
impl<R, A: Copy> Clone for PlanTarget<'_, R, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R, A: Copy> Copy for PlanTarget<'_, R, A> {}

impl<'a, R> PlanTarget<'a, R> {
    pub(crate) fn bound(space: &'a BoundDynamicFusionMapSpace<R>) -> Self
    where
        R: MultiplicityFreeRigidSymbols,
    {
        Self {
            rule: space.provider(),
            space: space.space(),
            authority: space.layout_primer(),
        }
    }
}

/// Adds the plan build since `start` to the profile and starts the artifact
/// clock.
fn attribute_plan_build(
    start: Option<std::time::Instant>,
    profile: Option<&mut TensorContractFusionProfile>,
) -> Option<std::time::Instant> {
    let (start, profile) = (start?, profile?);
    profile.dynamic_tree_plan_build += start.elapsed();
    Some(std::time::Instant::now())
}

fn attributed_artifact_parts(profile: Option<&TensorContractFusionProfile>) -> std::time::Duration {
    profile.map_or(std::time::Duration::ZERO, |profile| {
        profile.source_space_lookup + profile.core_dst_space_lookup + profile.core_block_plan_build
    })
}

/// The artifact assembly time, less what the artifact compiler attributed to
/// its structure lookups and core plan.
fn attribute_artifact_prepare(
    start: Option<std::time::Instant>,
    before: std::time::Duration,
    profile: Option<&mut TensorContractFusionProfile>,
) {
    if let (Some(start), Some(profile)) = (start, profile) {
        let attributed = attributed_artifact_parts(Some(profile)).saturating_sub(before);
        profile.dynamic_tree_artifact_prepare += start.elapsed().saturating_sub(attributed);
    }
}

/// The request checks both halves of the contraction route run: the provider
/// against the three spaces, and the operand conjugation flags against the
/// request.
fn validate_raw_contract_request<M, R>(
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
) -> Result<(), M::Error>
where
    M: PlanningAlgebra<R>,
{
    M::validate_spaces(
        target.rule,
        target.authority,
        target.space,
        lhs.storage_space(),
        rhs.storage_space(),
    )?;
    if axes.lhs_conjugate() != lhs.storage_conjugate()
        || axes.rhs_conjugate() != rhs.storage_conjugate()
    {
        return Err(OperationError::InvalidArgument {
            message: "prelowered operand flags must match the contraction request",
        }
        .into());
    }
    Ok(())
}

/// First half of the contraction route: validation and the canonical core
/// over the parent buffers — lazy adjoints as GEMM operand flags and a
/// uniform fermionic twist folded into per-job alpha. A [`CoreRoute::Miss`]
/// is the input of the rest of the planner
/// ([`TensorContractFusionExecutionContext::plan_contract_beyond_core`]).
///
/// Free function, not a context method: this half resolves entirely from
/// the operands, so the canonical device contraction takes no Runtime
/// context lock.
#[doc(hidden)]
pub fn try_compile_storage_contract_core_route<X: ExecCaps, R>(
    dst_space: &BoundDynamicFusionMapSpace<R>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
) -> Result<CoreRoute<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    try_compile_core_route::<X, MultiplicityFreeAdmissionMode, R>(
        PlanTarget::bound(dst_space),
        lhs,
        rhs,
        axes,
        ContractKind::Contract,
    )
}

/// The planner for a composition, TensorKit `mul!(C, A, B)`
/// (`linalg.jl:330-370` @cfaa073): A's domain against B's codomain in
/// order, the identity output, no twist and no candidate walk, resolved to
/// `Core { swapped: false }` with unit job coefficients, or the packed core
/// of a non-canonical tiling for an executor with
/// [`ExecCaps::IRREGULAR_CORE`]. Any other composition is
/// `UnsupportedTensorContractScope`: `mul!` has no transform route, so
/// none is planned.
///
/// The braiding check stays with the caller: `mul!` admits every braiding
/// style, since composition has no braid-dependent step.
///
/// Free function, not a context method: the core resolves entirely from
/// the operands, so planning a composition takes no Runtime context lock.
#[doc(hidden)]
pub fn plan_compose<X: ExecCaps, R>(
    dst_space: &BoundDynamicFusionMapSpace<R>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
) -> Result<StorageContractResolution<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    plan_compose_in::<X, MultiplicityFreeAdmissionMode, R>(PlanTarget::bound(dst_space), lhs, rhs)
}

/// [`plan_compose`] in either admission mode `M`.
pub(crate) fn plan_compose_in<X: ExecCaps, M, R>(
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
) -> Result<StorageContractResolution<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let (left, right) = (lhs.oriented_homspace(), rhs.oriented_homspace());
    let lhs_axes: Vec<usize> = (left.nout()..left.rank()).collect();
    let rhs_axes: Vec<usize> = (0..right.nout()).collect();
    let axes = TensorContractSpec::new_with_conjugation(
        &lhs_axes,
        &rhs_axes,
        tenet_operations::OutputAxisOrder::identity(),
        lhs.storage_conjugate(),
        rhs.storage_conjugate(),
    );
    match try_compile_core_route::<X, M, R>(target, lhs, rhs, axes, ContractKind::Compose)? {
        CoreRoute::Hit(resolution) => Ok(resolution),
        CoreRoute::Miss(_) => Err(OperationError::UnsupportedTensorContractScope {
            message:
                "storage-direct composition supports only canonical fully-direct oriented operands",
        }
        .into()),
    }
}

/// [`TensorContractFusionExecutionContext::plan_contract`] in admission mode
/// `M`, deriving spaces and transformers under `cache` (the context's
/// planning state, or a checked call's transaction).
pub(crate) fn plan_contract_in<X: ExecCaps, M, R>(
    cache: &mut M::StructureCache,
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    (lhs_axes, rhs_axes, output_axes): (&[usize], &[usize], &[usize]),
    mut profile: Option<&mut TensorContractFusionProfile>,
) -> Result<StorageContractResolution<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let axes = TensorContractSpec::new_with_conjugation(
        lhs_axes,
        rhs_axes,
        tenet_operations::OutputAxisOrder::from_axes(output_axes),
        lhs.storage_conjugate(),
        rhs.storage_conjugate(),
    );
    let start = profile.is_some().then(std::time::Instant::now);
    let core = try_compile_core_route::<X, M, R>(target, lhs, rhs, axes, ContractKind::Contract)?;
    if let (Some(start), Some(profile)) = (start, profile.as_deref_mut()) {
        // A hit is dominated by its core plan compile, a miss by the walk.
        match core {
            CoreRoute::Hit(_) => profile.core_block_plan_build += start.elapsed(),
            CoreRoute::Miss(_) => profile.resolution_preflight += start.elapsed(),
        }
    }
    match core {
        CoreRoute::Hit(core) => Ok(core),
        CoreRoute::Miss(miss) => plan_contract_beyond_core_in::<X, M, R>(
            cache,
            miss,
            target,
            lhs,
            rhs,
            (lhs_axes, rhs_axes, output_axes),
            profile,
        ),
    }
}

fn plan_contract_beyond_core_in<X: ExecCaps, M, R>(
    cache: &mut M::StructureCache,
    miss: CoreMiss,
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    (lhs_axes, rhs_axes, output_axes): (&[usize], &[usize], &[usize]),
    mut profile: Option<&mut TensorContractFusionProfile>,
) -> Result<StorageContractResolution<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let start = profile.is_some().then(std::time::Instant::now);
    let order = super::resolution::copy_c_order(
        || M::contract_twist_possible(target.rule, target.authority),
        |homspace, axes| {
            super::resolution::contract_axes_require_twist::<M, R>(
                target.rule,
                target.authority,
                homspace,
                axes,
            )
        },
        target.space,
        lhs,
        rhs,
        lhs_axes,
        rhs_axes,
        output_axes,
        Some(miss.requested_zero_copy),
        true,
    );
    if let (Some(start), Some(profile)) = (start, profile.as_deref_mut()) {
        profile.resolution_preflight += start.elapsed();
    }
    if let Some(orientation) = order {
        if let Some(copy) = plan_copy_c_in::<X, M, R>(
            cache,
            target,
            lhs,
            rhs,
            (lhs_axes, rhs_axes, output_axes),
            orientation,
            profile.as_deref_mut(),
        )? {
            return Ok(copy);
        }
    }
    let axes = TensorContractSpec::new_with_conjugation(
        lhs_axes,
        rhs_axes,
        tenet_operations::OutputAxisOrder::from_axes(output_axes),
        lhs.storage_conjugate(),
        rhs.storage_conjugate(),
    );
    compile_dynamic_tree_in::<M, R>(cache, target, lhs, rhs, axes, X::IRREGULAR_CORE, profile)
}

/// The `CopyC` route in `orientation`, or `None` when its temporary has
/// no core the executor runs (the `DynamicTree` artifact then applies).
/// Profiled, the temporary's core counts as core plan build and the rest
/// (temporary derivation, transform lookup) as route preflight.
fn plan_copy_c_in<X: ExecCaps, M, R>(
    cache: &mut M::StructureCache,
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    (lhs_axes, rhs_axes, output_axes): (&[usize], &[usize], &[usize]),
    orientation: super::fusion::FusionContractOrientation,
    profile: Option<&mut TensorContractFusionProfile>,
) -> Result<Option<StorageContractResolution<M::Scalar>>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let start = profile.is_some().then(std::time::Instant::now);
    let swapped = orientation == super::fusion::FusionContractOrientation::RhsLhs;
    let ((first, first_axes), (second, second_axes)) = if swapped {
        ((rhs, rhs_axes), (lhs, lhs_axes))
    } else {
        ((lhs, lhs_axes), (rhs, rhs_axes))
    };
    let identity = tenet_operations::OutputAxisOrder::identity();
    // The temporary is `first·second`'s own default-order result over the
    // oriented HomSpaces, as any contraction result is, derived under the
    // destination's authority (the request was validated against its
    // provider).
    let first_open = first.storage_space().rank() - first_axes.len();
    let default_axes: smallvec::SmallVec<[usize; 16]> =
        (0..first_open + second.storage_space().rank() - second_axes.len()).collect();
    let temporary = M::copy_c_temporary(
        cache,
        target.authority,
        target.rule,
        first,
        second,
        first_axes,
        second_axes,
        &default_axes,
        first_open,
    )?;
    let temporary_target = PlanTarget {
        rule: target.rule,
        space: &temporary,
        authority: target.authority,
    };
    let core_start = profile.is_some().then(std::time::Instant::now);
    let core = try_compile_core_route::<X, M, R>(
        temporary_target,
        first,
        second,
        TensorContractSpec::new_with_conjugation(
            first_axes,
            second_axes,
            identity,
            first.storage_conjugate(),
            second.storage_conjugate(),
        ),
        ContractKind::Contract,
    )?;
    let core_time = core_start.map(|start| start.elapsed());
    let CoreRoute::Hit(core) = core else {
        return Ok(None);
    };
    let Some((core, core_swapped)) = core.direct_core() else {
        return Ok(None);
    };
    let operation = super::resolution::copy_c_output_transform(
        orientation,
        lhs.storage_space().rank(),
        rhs.storage_space().rank(),
        lhs_axes,
        rhs_axes,
        output_axes,
        target.space.nout(),
    );
    let temporary_structure = Arc::clone(temporary.structure());
    let transform = M::tree_structure(
        cache,
        target.rule,
        &operation,
        target.space.structure(),
        TreeStructureSource::Stored {
            structure: &temporary_structure,
            storage_conjugate: false,
        },
    )?;
    if let (Some(start), Some(core_time), Some(profile)) = (start, core_time, profile) {
        profile.core_block_plan_build += core_time;
        profile.resolution_preflight += start.elapsed().saturating_sub(core_time);
    }
    Ok(Some(StorageContractResolution::new(ContractRoute::CopyC(
        CopyCRoute {
            core: Arc::clone(core),
            swapped: core_swapped ^ swapped,
            temporary_len: temporary.required_len().map_err(OperationError::from)?,
            temporary: temporary_structure,
            transform,
        },
    ))))
}

/// The last rung of the planner: the Host `DynamicTree` artifact for a
/// contraction the core declined (see
/// [`TensorContractFusionExecutionContext::compile_storage_contract_dynamic_tree`]).
///
/// `irregular_core`: the executor packs and scatters a core plan that is
/// not fully direct ([`ExecCaps::IRREGULAR_CORE`]), as the artifact's core
/// over a non-canonical destination tiling (#1517) is.
///
/// Profiled, the `FusionContractPlan` build counts as `DynamicTree` plan
/// build and the artifact assembly as artifact prepare (its structure
/// lookups and core plan attributed separately, as before).
fn compile_dynamic_tree_in<M, R>(
    cache: &mut M::StructureCache,
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
    irregular_core: bool,
    mut profile: Option<&mut TensorContractFusionProfile>,
) -> Result<StorageContractResolution<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    // Re-run here, not only in the first half: this entry is callable on
    // its own, and the artifact compilers assume a validated request.
    validate_raw_contract_request::<M, R>(target, lhs, rhs, axes)?;
    let artifact = if !lhs.storage_conjugate() && !rhs.storage_conjugate() {
        let PlanTarget {
            rule,
            space: dst_space,
            authority,
        } = target;
        let plan_start = profile.is_some().then(std::time::Instant::now);
        let (lhs_space, rhs_space) = (lhs.storage_space(), rhs.storage_space());
        let plan = M::dynamic_tree_plan(rule, authority, dst_space, lhs_space, rhs_space, axes)?;
        let artifact_start = attribute_plan_build(plan_start, profile.as_deref_mut());
        let before = attributed_artifact_parts(profile.as_deref());
        let artifact = if profile.is_some() {
            super::dynamic::compile_dynamic_tree_execution_artifact::<M, R, true>(
                cache,
                rule,
                authority,
                &plan,
                dst_space,
                lhs_space,
                lhs_space.structure(),
                rhs_space,
                rhs_space.structure(),
                profile.as_deref_mut(),
            )?
        } else {
            super::dynamic::compile_dynamic_tree_execution_artifact::<M, R, false>(
                cache,
                rule,
                authority,
                &plan,
                dst_space,
                lhs_space,
                lhs_space.structure(),
                rhs_space,
                rhs_space.structure(),
                None,
            )?
        };
        attribute_artifact_prepare(artifact_start, before, profile.as_deref_mut());
        artifact
    } else {
        M::prelowered_dynamic_tree_artifact(cache, target, lhs, rhs, axes, profile)?
    };
    // Transformed sources are canonical coupled layouts, so only a
    // non-canonical destination tiling leaves the core plan not fully
    // direct; a direct-core executor reports it before any device work.
    if !irregular_core && !artifact.block_plan_is_fully_direct() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "dynamic-tree core plan over transformed sources is not fully direct",
        }
        .into());
    }
    Ok(StorageContractResolution::new(ContractRoute::DynamicTree(
        Arc::new(artifact),
    )))
}

/// The multiplicity-free `DynamicTree` artifact of a contraction with a
/// lazy-adjoint operand: both operands prelowered to logical-key layouts
/// under the destination's primer.
pub(crate) fn compile_prelowered_dynamic_tree<R>(
    planning: &mut TreeTransformPlanning,
    target: PlanTarget<'_, R>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
    mut profile: Option<&mut TensorContractFusionProfile>,
) -> Result<super::dynamic::DynamicTreeExecutionArtifact<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Scalar: DenseBlockScalar + MultiplicityFreePlanningScalar,
{
    let PlanTarget {
        rule,
        space: dst_space,
        authority: layout_primer,
    } = target;
    let plan_start = profile.is_some().then(std::time::Instant::now);
    let lhs_layout = lhs.prepare(rule, layout_primer)?;
    let rhs_layout = rhs.prepare(rule, layout_primer)?;
    let plan = prepare_tensorcontract_fusion_plan_dyn_prelowered_canonical(
        rule,
        dst_space,
        &lhs_layout,
        &rhs_layout,
        axes,
        layout_primer,
    )?;
    let artifact_start = attribute_plan_build(plan_start, profile.as_deref_mut());
    let before = attributed_artifact_parts(profile.as_deref());
    let artifact = if profile.is_some() {
        super::dynamic::compile_prelowered_dynamic_tree_execution_artifact::<_, true>(
            planning,
            rule,
            layout_primer,
            &plan,
            dst_space,
            &lhs_layout,
            &rhs_layout,
            profile.as_deref_mut(),
        )?
    } else {
        super::dynamic::compile_prelowered_dynamic_tree_execution_artifact::<_, false>(
            planning,
            rule,
            layout_primer,
            &plan,
            dst_space,
            &lhs_layout,
            &rhs_layout,
            None,
        )?
    };
    attribute_artifact_prepare(artifact_start, before, profile);
    Ok(artifact)
}

fn try_compile_core_route<X: ExecCaps, M, R>(
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
    kind: ContractKind,
) -> Result<CoreRoute<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    validate_raw_contract_request::<M, R>(target, lhs, rhs, axes)?;
    let irregular = X::IRREGULAR_CORE;
    // A twist that is not uniform within one coupled-sector matrix has no
    // per-job alpha; the DynamicTree artifact applies it per block instead.
    let mut requested_zero_copy = false;
    let route = match kind {
        ContractKind::Contract => try_compile_oriented_storage_contract_candidate_plan::<M, R>(
            target,
            lhs,
            rhs,
            axes,
            &mut requested_zero_copy,
            irregular,
        )?,
        ContractKind::Compose => try_compile_oriented_storage_contract_plan::<M, R>(
            target, lhs, rhs, axes, kind, irregular,
        )?
        .filter(|plan| irregular || plan.is_fully_direct())
        .map(|plan| ContractRoute::Core {
            plan,
            swapped: false,
        }),
    };
    Ok(match route {
        Some(route) => CoreRoute::Hit(StorageContractResolution::new(route)),
        None => CoreRoute::Miss(CoreMiss {
            requested_zero_copy,
        }),
    })
}

#[cfg(test)]
/// Canonical storage contraction over parent buffers with lazy operand
/// orientation. A miss is unsupported; this device leaf never prepares
/// logical-key projections or source transforms.
///
/// Free function, not a context method: the storage-direct route resolves and
/// replays entirely from the operands, so a device operation calling it needs
/// no execution context and therefore no Runtime state lock.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensorcontract_fusion_dyn_prelowered_direct_on_storage<R, G, D, DDst, DLhs, DRhs>(
    gemm: &mut G,
    dst_space: &BoundDynamicFusionMapSpace<R>,
    dst: &mut DDst,
    lhs: FusionOperand<'_>,
    lhs_storage: &DLhs,
    rhs: FusionOperand<'_>,
    rhs_storage: &DRhs,
    axes: TensorContractSpec<'_>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    D: DenseBlockScalar + RecouplingCoefficientAction<R::Scalar>,
    G: tenet_operations::fusion_replay::StorageGemm<D, DDst, DLhs, DRhs>,
    DDst: TensorStorage<D>,
    DLhs: TensorStorage<D>,
    DRhs: TensorStorage<D>,
{
    let target = PlanTarget::bound(dst_space);
    validate_raw_contract_request::<MultiplicityFreeAdmissionMode, R>(target, lhs, rhs, axes)?;
    let plan = try_compile_oriented_storage_contract_plan::<MultiplicityFreeAdmissionMode, R>(
        target,
        lhs,
        rhs,
        axes,
        ContractKind::Contract,
        false,
    )?
    .filter(|plan| plan.is_fully_direct())
    .ok_or_else(|| OperationError::UnsupportedTensorContractScope {
        message:
            "storage-direct contraction supports only canonical fully-direct oriented operands",
    })?;
    plan.execute_direct_on_storage_prezeroed(gemm, dst, lhs_storage, rhs_storage)
}
