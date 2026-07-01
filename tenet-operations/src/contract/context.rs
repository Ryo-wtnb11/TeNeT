use std::collections::HashMap;
use std::hash::Hash;

use tenet_core::{
    CoreError, FusionTensorMapSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols, TensorMap,
};

use crate::axis::{OwnedTensorContractAxisSpec, TensorContractAxisSpec};
use crate::backend::DenseTreeTransformOperations;
use crate::cache::{TensorContractStructureCache, TensorContractStructureCacheKey};
use crate::tree_context::TreeTransformExecutionContext;
use crate::tree_transform::TreeTransformRuleCacheKey;
use crate::{
    DenseBlockScalar, DenseRecouplingScalar, OperationError, RecouplingCoefficientAction,
    TreeTransformBackend,
};

use super::backend::TensorContractBackend;
use super::dynamic::tensorcontract_fusion_dynamic_plan_into_context;
use super::dynamic_space::DynamicFusionMapSpace;
use super::dynamic_space_cache::{DynamicFusionSpaceCache, TensorContractFusionSpaceCacheStats};
use super::fusion::{
    tensorcontract_fusion_block_specs, tensorcontract_fusion_explicit_plan,
    tensorcontract_fusion_structure, TensorContractFusionExplicitPlan,
    EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST, SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
};
use super::fusion_block::{
    is_canonical_fusion_block_contract, tensorcontract_canonical_fusion_blocks_into_raw,
};
use super::scratch::DynamicFusionScratchWorkspace;
use super::structure::{TensorContractAxisPlan, TensorContractBlockSpec, TensorContractStructure};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TensorContractPlanKey {
    axes: OwnedTensorContractAxisSpec,
}

impl TensorContractPlanKey {
    pub fn from_axes(
        lhs_rank: usize,
        rhs_rank: usize,
        dst_rank: usize,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Self, OperationError> {
        let axis_plan = TensorContractAxisPlan::compile(lhs_rank, rhs_rank, dst_rank, axes)?;
        Ok(Self {
            axes: OwnedTensorContractAxisSpec::new_with_conjugation(
                axis_plan.lhs_contracting_axes,
                axis_plan.rhs_contracting_axes,
                axis_plan.output_axes,
                axis_plan.lhs_conjugate,
                axis_plan.rhs_conjugate,
            ),
        })
    }

    #[inline]
    pub fn axes(&self) -> &OwnedTensorContractAxisSpec {
        &self.axes
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TensorContractBlockPlanKey {
    axes: OwnedTensorContractAxisSpec,
    block_specs: Vec<TensorContractBlockPlanTerm>,
}

impl TensorContractBlockPlanKey {
    pub fn from_block_specs(
        lhs_rank: usize,
        rhs_rank: usize,
        dst_rank: usize,
        axes: TensorContractAxisSpec<'_>,
        block_specs: &[TensorContractBlockSpec],
    ) -> Result<Self, OperationError> {
        let axis_plan = TensorContractAxisPlan::compile(lhs_rank, rhs_rank, dst_rank, axes)?;
        Ok(Self {
            axes: OwnedTensorContractAxisSpec::new_with_conjugation(
                axis_plan.lhs_contracting_axes,
                axis_plan.rhs_contracting_axes,
                axis_plan.output_axes,
                axis_plan.lhs_conjugate,
                axis_plan.rhs_conjugate,
            ),
            block_specs: block_specs
                .iter()
                .map(TensorContractBlockPlanTerm::from_block_spec)
                .collect(),
        })
    }

    #[inline]
    pub fn axes(&self) -> &OwnedTensorContractAxisSpec {
        &self.axes
    }

    #[inline]
    pub fn block_specs(&self) -> &[TensorContractBlockPlanTerm] {
        &self.block_specs
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TensorContractBlockPlanTerm {
    dst_block: usize,
    lhs_block: usize,
    rhs_block: usize,
    coefficient_bits: u64,
}

impl TensorContractBlockPlanTerm {
    fn from_block_spec(spec: &TensorContractBlockSpec) -> Self {
        Self {
            dst_block: spec.dst_block(),
            lhs_block: spec.lhs_block(),
            rhs_block: spec.rhs_block(),
            coefficient_bits: spec.coefficient().to_bits(),
        }
    }

    #[inline]
    pub fn dst_block(&self) -> usize {
        self.dst_block
    }

    #[inline]
    pub fn lhs_block(&self) -> usize {
        self.lhs_block
    }

    #[inline]
    pub fn rhs_block(&self) -> usize {
        self.rhs_block
    }

    #[inline]
    pub fn coefficient_bits(&self) -> u64 {
        self.coefficient_bits
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TensorContractFusionPlanCacheKey<RuleKey> {
    rule: RuleKey,
    dst_homspace: FusionTreeHomSpace,
    lhs_homspace: FusionTreeHomSpace,
    rhs_homspace: FusionTreeHomSpace,
    axes: OwnedTensorContractAxisSpec,
}

impl<RuleKey> TensorContractFusionPlanCacheKey<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    fn from_spaces<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
    >(
        rule: &R,
        dst: &FusionTensorMapSpace<DST_NOUT, DST_NIN>,
        lhs: &FusionTensorMapSpace<LHS_NOUT, LHS_NIN>,
        rhs: &FusionTensorMapSpace<RHS_NOUT, RHS_NIN>,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let axis_plan = TensorContractAxisPlan::compile(
            lhs.subblock_structure().rank(),
            rhs.subblock_structure().rank(),
            dst.subblock_structure().rank(),
            axes,
        )?;
        Ok(Self {
            rule: rule.tree_transform_rule_cache_key(),
            dst_homspace: dst.homspace().clone(),
            lhs_homspace: lhs.homspace().clone(),
            rhs_homspace: rhs.homspace().clone(),
            axes: OwnedTensorContractAxisSpec::new_with_conjugation(
                axis_plan.lhs_contracting_axes,
                axis_plan.rhs_contracting_axes,
                axis_plan.output_axes,
                axis_plan.lhs_conjugate,
                axis_plan.rhs_conjugate,
            ),
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorContractFusionPlanCacheStats {
    hits: usize,
    misses: usize,
}

impl TensorContractFusionPlanCacheStats {
    #[inline]
    pub fn hits(self) -> usize {
        self.hits
    }

    #[inline]
    pub fn misses(self) -> usize {
        self.misses
    }
}

#[derive(Clone, Debug)]
struct TensorContractFusionPlanCache<RuleKey> {
    plans: HashMap<TensorContractFusionPlanCacheKey<RuleKey>, TensorContractFusionExplicitPlan>,
    stats: TensorContractFusionPlanCacheStats,
}

impl<RuleKey> Default for TensorContractFusionPlanCache<RuleKey> {
    fn default() -> Self {
        Self {
            plans: HashMap::new(),
            stats: TensorContractFusionPlanCacheStats::default(),
        }
    }
}

impl<RuleKey> TensorContractFusionPlanCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    fn len(&self) -> usize {
        self.plans.len()
    }

    fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    fn stats(&self) -> TensorContractFusionPlanCacheStats {
        self.stats
    }

    fn get_or_compile<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
    >(
        &mut self,
        rule: &R,
        dst: &FusionTensorMapSpace<DST_NOUT, DST_NIN>,
        lhs: &FusionTensorMapSpace<LHS_NOUT, LHS_NIN>,
        rhs: &FusionTensorMapSpace<RHS_NOUT, RHS_NIN>,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<&TensorContractFusionExplicitPlan, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let key = TensorContractFusionPlanCacheKey::from_spaces(rule, dst, lhs, rhs, axes)?;
        if self.plans.get(&key).is_some() {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
            let plan =
                tensorcontract_fusion_explicit_plan(rule, dst, lhs, rhs, key.axes.as_spec())?;
            self.plans.insert(key.clone(), plan);
        }
        Ok(self
            .plans
            .get(&key)
            .expect("fusion plan inserted before replay"))
    }

    fn get_cached_by_key(
        &mut self,
        key: &TensorContractFusionPlanCacheKey<RuleKey>,
    ) -> Option<&TensorContractFusionExplicitPlan> {
        let plan = self.plans.get(key)?;
        self.stats.hits += 1;
        Some(plan)
    }

    fn contains_key(&self, key: &TensorContractFusionPlanCacheKey<RuleKey>) -> bool {
        self.plans.contains_key(key)
    }
}

fn fusion_plan_cache_key<
    RuleKey,
    R,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
>(
    rule: &R,
    dst: &FusionTensorMapSpace<DST_NOUT, DST_NIN>,
    lhs: &FusionTensorMapSpace<LHS_NOUT, LHS_NIN>,
    rhs: &FusionTensorMapSpace<RHS_NOUT, RHS_NIN>,
    axes: TensorContractAxisSpec<'_>,
) -> Result<TensorContractFusionPlanCacheKey<RuleKey>, OperationError>
where
    RuleKey: Clone + Eq + Hash,
    R: TreeTransformRuleCacheKey<Key = RuleKey>,
{
    TensorContractFusionPlanCacheKey::from_spaces(rule, dst, lhs, rhs, axes)
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
    stats: TensorContractCacheStats,
}

impl<PlanKey> Default for TensorContractCache<PlanKey> {
    fn default() -> Self {
        Self {
            structures: TensorContractStructureCache::default(),
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
    >(
        &mut self,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs>,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<&TensorContractStructure, OperationError> {
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
        if self.structures.get(&structure_key).is_some() {
            self.stats.structure_hits += 1;
        } else {
            self.stats.structure_misses += 1;
            let structure =
                TensorContractStructure::compile(dst, lhs, rhs, plan_key.axes().as_spec())?;
            self.structures.insert(structure_key.clone(), structure);
        }
        Ok(self
            .structures
            .get(&structure_key)
            .expect("tensor contract structure inserted before replay"))
    }
}

impl TensorContractCache<TensorContractBlockPlanKey> {
    pub fn get_or_compile_with_block_specs<
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
    >(
        &mut self,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs>,
        axes: TensorContractAxisSpec<'_>,
        block_specs: &[TensorContractBlockSpec],
    ) -> Result<&TensorContractStructure, OperationError> {
        let plan_key = TensorContractBlockPlanKey::from_block_specs(
            lhs.structure().rank(),
            rhs.structure().rank(),
            dst.structure().rank(),
            axes,
            block_specs,
        )?;
        let structure_key = TensorContractStructureCacheKey::from_structures(
            plan_key.clone(),
            dst.structure(),
            lhs.structure(),
            rhs.structure(),
        )?;
        if self.structures.get(&structure_key).is_some() {
            self.stats.structure_hits += 1;
        } else {
            self.stats.structure_misses += 1;
            let structure = TensorContractStructure::compile_with_block_specs(
                dst,
                lhs,
                rhs,
                plan_key.axes().as_spec(),
                block_specs,
            )?;
            self.structures.insert(structure_key.clone(), structure);
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
    pub fn workspace_mut(&mut self) -> &mut B::Workspace {
        &mut self.workspace
    }

    #[inline]
    pub fn cache(&self) -> &TensorContractCache {
        &self.cache
    }

    #[inline]
    pub fn cache_mut(&mut self) -> &mut TensorContractCache {
        &mut self.cache
    }

    pub fn into_parts(self) -> (B, B::Workspace, TensorContractCache) {
        (self.backend, self.workspace, self.cache)
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
        axes: TensorContractAxisSpec<'_>,
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
    axes: TensorContractAxisSpec<'_>,
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
    RuleKey: Clone + Eq + Hash,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
{
    tree_context: TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    contract_backend: BC,
    contract_workspace: BC::Workspace,
    contract_cache: TensorContractCache<TensorContractBlockPlanKey>,
    fusion_plan_cache: TensorContractFusionPlanCache<RuleKey>,
    fusion_scratch: DynamicFusionScratchWorkspace<D>,
    fusion_space_cache: DynamicFusionSpaceCache<RuleKey>,
}

impl<D, RuleKey, BT, BC> TensorContractFusionExecutionContext<D, RuleKey, BT, BC>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    RuleKey: Clone + Eq + Hash,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
{
    pub fn with_parts(
        tree_context: TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        contract_backend: BC,
        contract_workspace: BC::Workspace,
        contract_cache: TensorContractCache<TensorContractBlockPlanKey>,
    ) -> Self {
        Self {
            tree_context,
            contract_backend,
            contract_workspace,
            contract_cache,
            fusion_plan_cache: TensorContractFusionPlanCache::default(),
            fusion_scratch: DynamicFusionScratchWorkspace::default(),
            fusion_space_cache: DynamicFusionSpaceCache::default(),
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

    #[inline]
    pub fn contract_workspace_mut(&mut self) -> &mut BC::Workspace {
        &mut self.contract_workspace
    }

    #[inline]
    pub fn contract_cache(&self) -> &TensorContractCache<TensorContractBlockPlanKey> {
        &self.contract_cache
    }

    #[inline]
    pub fn contract_cache_mut(&mut self) -> &mut TensorContractCache<TensorContractBlockPlanKey> {
        &mut self.contract_cache
    }

    #[inline]
    pub fn fusion_plan_cache_len(&self) -> usize {
        self.fusion_plan_cache.len()
    }

    #[inline]
    pub fn fusion_plan_cache_stats(&self) -> TensorContractFusionPlanCacheStats {
        self.fusion_plan_cache.stats()
    }

    #[inline]
    pub fn fusion_space_cache_len(&self) -> usize {
        self.fusion_space_cache.len()
    }

    #[inline]
    pub fn fusion_transformed_space_cache_len(&self) -> usize {
        self.fusion_space_cache.transformed_len()
    }

    #[inline]
    pub fn fusion_canonical_dst_space_cache_len(&self) -> usize {
        self.fusion_space_cache.canonical_dst_len()
    }

    #[inline]
    pub fn fusion_space_cache_stats(&self) -> TensorContractFusionSpaceCacheStats {
        self.fusion_space_cache.stats()
    }

    pub fn into_parts(
        self,
    ) -> (
        TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        BC,
        BC::Workspace,
        TensorContractCache<TensorContractBlockPlanKey>,
    ) {
        (
            self.tree_context,
            self.contract_backend,
            self.contract_workspace,
            self.contract_cache,
        )
    }
}

impl<D, RuleKey, BT, BC> TensorContractFusionExecutionContext<D, RuleKey, BT, BC>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    RuleKey: Clone + Eq + Hash,
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
            TensorContractCache::new(),
        )
    }
}

impl<D, RuleKey, BT, BC> Default for TensorContractFusionExecutionContext<D, RuleKey, BT, BC>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    RuleKey: Clone + Eq + Hash,
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
    RuleKey: Clone + Eq + Hash,
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
    >(
        &mut self,
        rule: &R,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        axes: TensorContractAxisSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
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
        if !axes.lhs_conjugate()
            && !axes.rhs_conjugate()
            && is_canonical_fusion_block_contract(
                rule,
                &dst_dynamic,
                &lhs_dynamic,
                &rhs_dynamic,
                axes,
            )?
        {
            return tensorcontract_canonical_fusion_blocks_into_raw(
                &mut self.contract_backend,
                &mut self.contract_workspace,
                rule,
                &dst_dynamic,
                dst.data_mut(),
                &lhs_dynamic,
                lhs.data(),
                &rhs_dynamic,
                rhs.data(),
                axes,
                alpha,
                beta,
            );
        }

        if !self.fusion_plan_cache.is_empty() {
            let plan_key = fusion_plan_cache_key(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)?;
            if self.fusion_plan_cache.contains_key(&plan_key) {
                let Self {
                    tree_context,
                    contract_backend,
                    contract_workspace,
                    contract_cache,
                    fusion_plan_cache,
                    fusion_scratch,
                    fusion_space_cache,
                } = self;
                let plan = fusion_plan_cache
                    .get_cached_by_key(&plan_key)
                    .expect("fusion plan cache key was present before replay");
                return tensorcontract_fusion_dynamic_plan_into_context(
                    tree_context,
                    contract_backend,
                    contract_workspace,
                    contract_cache,
                    fusion_scratch,
                    fusion_space_cache,
                    rule,
                    plan,
                    dst,
                    lhs,
                    rhs,
                    alpha,
                    beta,
                );
            }
        }

        if axes.lhs_conjugate() || axes.rhs_conjugate() {
            match tensorcontract_fusion_structure(rule, dst, lhs, rhs, axes) {
                Ok(structure) => {
                    return self.contract_backend.tensorcontract_structure_into(
                        &mut self.contract_workspace,
                        &structure,
                        dst,
                        lhs,
                        rhs,
                        alpha,
                        beta,
                    );
                }
                Err(OperationError::UnsupportedTensorContractScope {
                    message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
                }) => {
                    let Self {
                        tree_context,
                        contract_backend,
                        contract_workspace,
                        contract_cache,
                        fusion_plan_cache,
                        fusion_scratch,
                        fusion_space_cache,
                    } = self;
                    let plan = fusion_plan_cache
                        .get_or_compile(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)?;
                    return tensorcontract_fusion_dynamic_plan_into_context(
                        tree_context,
                        contract_backend,
                        contract_workspace,
                        contract_cache,
                        fusion_scratch,
                        fusion_space_cache,
                        rule,
                        plan,
                        dst,
                        lhs,
                        rhs,
                        alpha,
                        beta,
                    );
                }
                Err(err) => return Err(err),
            }
        }

        match tensorcontract_fusion_block_specs(rule, dst_fusion, lhs_fusion, rhs_fusion, axes) {
            Ok(block_specs) => {
                let structure = self.contract_cache.get_or_compile_with_block_specs(
                    dst,
                    lhs,
                    rhs,
                    axes,
                    &block_specs,
                )?;
                self.contract_backend.tensorcontract_structure_into(
                    &mut self.contract_workspace,
                    structure,
                    dst,
                    lhs,
                    rhs,
                    alpha,
                    beta,
                )
            }
            Err(OperationError::UnsupportedTensorContractScope {
                message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
            }) => {
                let Self {
                    tree_context,
                    contract_backend,
                    contract_workspace,
                    contract_cache,
                    fusion_plan_cache,
                    fusion_scratch,
                    fusion_space_cache,
                } = self;
                let plan = fusion_plan_cache
                    .get_or_compile(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)?;
                return tensorcontract_fusion_dynamic_plan_into_context(
                    tree_context,
                    contract_backend,
                    contract_workspace,
                    contract_cache,
                    fusion_scratch,
                    fusion_space_cache,
                    rule,
                    plan,
                    dst,
                    lhs,
                    rhs,
                    alpha,
                    beta,
                );
            }
            Err(err) => Err(err),
        }
    }

    pub fn tensorcontract_fusion_explicit_plan_into<
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
        plan: &TensorContractFusionExplicitPlan,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan>,
        rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        if !plan.output_transform_is_identity()
            || DST_NOUT != plan.canonical_dst_nout()
            || DST_NIN != plan.canonical_dst_nin()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
            });
        }
        self.transform_sources_and_contract(
            rule,
            plan,
            dst,
            lhs_canonical,
            rhs_canonical,
            lhs,
            rhs,
            alpha,
            beta,
        )
    }

    pub fn tensorcontract_fusion_explicit_plan_into_canonical_dst<
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
        plan: &TensorContractFusionExplicitPlan,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        canonical_dst: &mut TensorMap<D, DST_CAN_NOUT, DST_CAN_NIN, SDstCan>,
        lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan>,
        rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        if DST_CAN_NOUT != plan.canonical_dst_nout() || DST_CAN_NIN != plan.canonical_dst_nin() {
            return Err(OperationError::StructureRankMismatch {
                expected: plan.canonical_dst_nout() + plan.canonical_dst_nin(),
                actual: DST_CAN_NOUT + DST_CAN_NIN,
            });
        }
        canonical_dst.data_mut().fill(D::zero());
        self.transform_sources_and_contract(
            rule,
            plan,
            canonical_dst,
            lhs_canonical,
            rhs_canonical,
            lhs,
            rhs,
            alpha,
            D::zero(),
        )?;
        self.tree_context.tree_pair_transform_into(
            rule,
            plan.output_transform().clone(),
            dst,
            canonical_dst,
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
        plan: &TensorContractFusionExplicitPlan,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan>,
        rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        if LHS_CAN_NOUT != plan.lhs_canonical_nout() || LHS_CAN_NIN != plan.lhs_canonical_nin() {
            return Err(OperationError::StructureRankMismatch {
                expected: plan.lhs_canonical_nout() + plan.lhs_canonical_nin(),
                actual: LHS_CAN_NOUT + LHS_CAN_NIN,
            });
        }
        if RHS_CAN_NOUT != plan.rhs_canonical_nout() || RHS_CAN_NIN != plan.rhs_canonical_nin() {
            return Err(OperationError::StructureRankMismatch {
                expected: plan.rhs_canonical_nout() + plan.rhs_canonical_nin(),
                actual: RHS_CAN_NOUT + RHS_CAN_NIN,
            });
        }

        lhs_canonical.data_mut().fill(D::zero());
        rhs_canonical.data_mut().fill(D::zero());
        self.tree_context.tree_pair_transform_into(
            rule,
            plan.lhs_transform().clone(),
            lhs_canonical,
            lhs,
            D::one(),
            D::zero(),
        )?;
        self.tree_context.tree_pair_transform_into(
            rule,
            plan.rhs_transform().clone(),
            rhs_canonical,
            rhs,
            D::one(),
            D::zero(),
        )?;

        let block_specs = tensorcontract_fusion_block_specs(
            rule,
            dst.fusion_space().ok_or(OperationError::Core(
                tenet_core::CoreError::MissingFusionSpace,
            ))?,
            lhs_canonical.fusion_space().ok_or(OperationError::Core(
                tenet_core::CoreError::MissingFusionSpace,
            ))?,
            rhs_canonical.fusion_space().ok_or(OperationError::Core(
                tenet_core::CoreError::MissingFusionSpace,
            ))?,
            plan.canonical_axes().as_spec(),
        )?;
        let structure = self.contract_cache.get_or_compile_with_block_specs(
            dst,
            lhs_canonical,
            rhs_canonical,
            plan.canonical_axes().as_spec(),
            &block_specs,
        )?;
        self.contract_backend.tensorcontract_structure_into(
            &mut self.contract_workspace,
            structure,
            dst,
            lhs_canonical,
            rhs_canonical,
            alpha,
            beta,
        )
    }
}
