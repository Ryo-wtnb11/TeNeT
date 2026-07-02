use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

use tenet_core::{
    BlockStructure, CoreError, FusionTensorMapSpace, FusionTreeHomSpace, HostReadableStorage,
    HostWritableStorage, MultiplicityFreeRigidSymbols, TensorMap,
};

use crate::axis::{AxisPermutation, OwnedTensorContractAxisSpec, TensorContractAxisSpec};
use crate::cache::{
    enforce_lru_limit, rebuild_lru_order_from_keys, touch_lru_key, OperationCachePolicy,
    TensorContractStructureCache, TensorContractStructureCacheKey,
};
use crate::lowering::adjoint_fusion_space_view;
use crate::tree_context::TreeTransformExecutionContext;
use crate::tree_transform::TreeTransformRuleCacheKey;
use crate::{
    DenseBlockScalar, DenseRecouplingScalar, DenseTreeTransformOperations, HostTensorOperations,
    OperationError, RecouplingCoefficientAction, TreeTransformBackend,
};

use super::backend::TensorContractBackend;
use super::dynamic::{
    tensorcontract_fusion_dynamic_plan_into_context,
    tensorcontract_fusion_dynamic_plan_into_context_profiled, DynamicFusionSpaceCache,
};
use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::{
    tensorcontract_fusion_explicit_plan, tensorcontract_fusion_structure,
    TensorContractFusionExplicitPlan, EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
    SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
};
use super::fusion_block::{
    CanonicalFusionBlockContractCache, CanonicalFusionBlockContractWorkspace,
};
use super::profile::{TensorContractFusionProfile, TensorContractFusionRoute};
use super::route_cache::{FusionRouteCache, TensorContractFusionRouteDecision};
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct RawTensorContractAxisSpecKey {
    lhs_contracting_axes: Vec<usize>,
    rhs_contracting_axes: Vec<usize>,
    output_permutation: RawAxisPermutationKey,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
}

impl RawTensorContractAxisSpecKey {
    fn from_axes(axes: TensorContractAxisSpec<'_>) -> Self {
        Self {
            lhs_contracting_axes: axes.lhs_contracting_axes().to_vec(),
            rhs_contracting_axes: axes.rhs_contracting_axes().to_vec(),
            output_permutation: RawAxisPermutationKey::from_axes(axes.output_permutation()),
            lhs_conjugate: axes.lhs_conjugate(),
            rhs_conjugate: axes.rhs_conjugate(),
        }
    }

    fn matches(&self, axes: TensorContractAxisSpec<'_>) -> bool {
        self.lhs_contracting_axes == axes.lhs_contracting_axes()
            && self.rhs_contracting_axes == axes.rhs_contracting_axes()
            && self.output_permutation.matches(axes.output_permutation())
            && self.lhs_conjugate == axes.lhs_conjugate()
            && self.rhs_conjugate == axes.rhs_conjugate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum RawAxisPermutationKey {
    Identity,
    Axes(Vec<usize>),
}

impl RawAxisPermutationKey {
    fn from_axes(axes: AxisPermutation<'_>) -> Self {
        match axes {
            AxisPermutation::Identity => Self::Identity,
            AxisPermutation::Axes(axes) => Self::Axes(axes.to_vec()),
        }
    }

    fn matches(&self, axes: AxisPermutation<'_>) -> bool {
        match (self, axes) {
            (Self::Identity, AxisPermutation::Identity) => true,
            (Self::Axes(stored), AxisPermutation::Axes(axes)) => stored == axes,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
struct FusionExplicitPlanCache<RuleKey> {
    last: Option<FusionExplicitPlanLastEntry<RuleKey>>,
    plans: HashMap<FusionExplicitPlanCacheKey<RuleKey>, Arc<TensorContractFusionExplicitPlan>>,
    lru_order: VecDeque<FusionExplicitPlanCacheKey<RuleKey>>,
    policy: OperationCachePolicy,
}

impl<RuleKey> Default for FusionExplicitPlanCache<RuleKey> {
    fn default() -> Self {
        Self {
            last: None,
            plans: HashMap::new(),
            lru_order: VecDeque::new(),
            policy: OperationCachePolicy::default(),
        }
    }
}

#[derive(Clone, Debug)]
struct FusionExplicitPlanLastEntry<RuleKey> {
    key: FusionExplicitPlanCacheKey<RuleKey>,
    rule: RuleKey,
    dst_nout: usize,
    dst_nin: usize,
    dst_rank: usize,
    dst_homspace: FusionTreeHomSpace,
    dst_structure: Arc<BlockStructure>,
    lhs_nout: usize,
    lhs_nin: usize,
    lhs_rank: usize,
    lhs_homspace: FusionTreeHomSpace,
    lhs_structure: Arc<BlockStructure>,
    rhs_nout: usize,
    rhs_nin: usize,
    rhs_rank: usize,
    rhs_homspace: FusionTreeHomSpace,
    rhs_structure: Arc<BlockStructure>,
    axes: RawTensorContractAxisSpecKey,
    plan: Arc<TensorContractFusionExplicitPlan>,
}

impl<RuleKey> FusionExplicitPlanLastEntry<RuleKey>
where
    RuleKey: Eq,
{
    fn matches<
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
    >(
        &self,
        rule: &RuleKey,
        dst: &FusionTensorMapSpace<DST_NOUT, DST_NIN>,
        lhs: &FusionTensorMapSpace<LHS_NOUT, LHS_NIN>,
        rhs: &FusionTensorMapSpace<RHS_NOUT, RHS_NIN>,
        axes: TensorContractAxisSpec<'_>,
    ) -> bool {
        &self.rule == rule
            && self.dst_nout == DST_NOUT
            && self.dst_nin == DST_NIN
            && self.dst_rank == dst.subblock_structure().rank()
            && self.lhs_nout == LHS_NOUT
            && self.lhs_nin == LHS_NIN
            && self.lhs_rank == lhs.subblock_structure().rank()
            && self.rhs_nout == RHS_NOUT
            && self.rhs_nin == RHS_NIN
            && self.rhs_rank == rhs.subblock_structure().rank()
            && Arc::ptr_eq(&self.dst_structure, dst.subblock_structure())
            && Arc::ptr_eq(&self.lhs_structure, lhs.subblock_structure())
            && Arc::ptr_eq(&self.rhs_structure, rhs.subblock_structure())
            && self.dst_homspace == *dst.homspace()
            && self.lhs_homspace == *lhs.homspace()
            && self.rhs_homspace == *rhs.homspace()
            && self.axes.matches(axes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FusionExplicitPlanCacheKey<RuleKey> {
    rule: RuleKey,
    dst_nout: usize,
    dst_nin: usize,
    dst_rank: usize,
    dst_homspace: FusionTreeHomSpace,
    lhs_nout: usize,
    lhs_nin: usize,
    lhs_rank: usize,
    lhs_homspace: FusionTreeHomSpace,
    rhs_nout: usize,
    rhs_nin: usize,
    rhs_rank: usize,
    rhs_homspace: FusionTreeHomSpace,
    axes: OwnedTensorContractAxisSpec,
}

impl<RuleKey> FusionExplicitPlanCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[inline]
    fn len(&self) -> usize {
        self.plans.len()
    }

    fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.last = None;
        if !policy.stores_entries() {
            self.plans.clear();
            self.lru_order.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            rebuild_lru_order_from_keys(&self.plans, &mut self.lru_order);
            self.enforce_lru_limit(max_entries);
        }
    }

    fn touch_plan(&mut self, key: &FusionExplicitPlanCacheKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.plans.contains_key(key) {
            touch_lru_key(&mut self.lru_order, key);
        }
    }

    fn insert_plan(
        &mut self,
        key: FusionExplicitPlanCacheKey<RuleKey>,
        plan: Arc<TensorContractFusionExplicitPlan>,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.plans.insert(key.clone(), plan);
        if self.policy.max_entries().is_some() {
            self.touch_plan(&key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_lru_limit(max_entries);
        }
    }

    fn enforce_lru_limit(&mut self, max_entries: usize) {
        let before = self.plans.len();
        enforce_lru_limit(&mut self.plans, &mut self.lru_order, max_entries);
        if self.plans.len() != before {
            self.last = None;
        }
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
    ) -> Result<Arc<TensorContractFusionExplicitPlan>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        if self.policy.stores_entries() {
            let refresh_lru = self.policy.max_entries().is_some();
            let last_hit = self.last.as_ref().and_then(|last| {
                if last.matches(&rule_key, dst, lhs, rhs, axes) {
                    Some((
                        refresh_lru.then(|| last.key.clone()),
                        Arc::clone(&last.plan),
                    ))
                } else {
                    None
                }
            });
            if let Some((key, plan)) = last_hit {
                if let Some(key) = key.as_ref() {
                    self.touch_plan(key);
                }
                return Ok(plan);
            }
        }
        let raw_axes = RawTensorContractAxisSpecKey::from_axes(axes);
        let axis_plan = TensorContractAxisPlan::compile(
            lhs.subblock_structure().rank(),
            rhs.subblock_structure().rank(),
            dst.subblock_structure().rank(),
            axes,
        )?;
        let axes_key = OwnedTensorContractAxisSpec::new_with_conjugation(
            axis_plan.lhs_contracting_axes,
            axis_plan.rhs_contracting_axes,
            axis_plan.output_axes,
            axis_plan.lhs_conjugate,
            axis_plan.rhs_conjugate,
        );
        let key = FusionExplicitPlanCacheKey {
            rule: rule_key.clone(),
            dst_nout: DST_NOUT,
            dst_nin: DST_NIN,
            dst_rank: dst.subblock_structure().rank(),
            dst_homspace: dst.homspace().clone(),
            lhs_nout: LHS_NOUT,
            lhs_nin: LHS_NIN,
            lhs_rank: lhs.subblock_structure().rank(),
            lhs_homspace: lhs.homspace().clone(),
            rhs_nout: RHS_NOUT,
            rhs_nin: RHS_NIN,
            rhs_rank: rhs.subblock_structure().rank(),
            rhs_homspace: rhs.homspace().clone(),
            axes: axes_key,
        };
        if !self.policy.stores_entries() {
            return Ok(Arc::new(tensorcontract_fusion_explicit_plan(
                rule, dst, lhs, rhs, axes,
            )?));
        }
        if let Some(plan) = self.plans.get(&key) {
            let plan = Arc::clone(plan);
            self.touch_plan(&key);
            self.last = Some(FusionExplicitPlanLastEntry {
                key: key.clone(),
                rule: rule_key,
                dst_nout: DST_NOUT,
                dst_nin: DST_NIN,
                dst_rank: dst.subblock_structure().rank(),
                dst_homspace: dst.homspace().clone(),
                dst_structure: Arc::clone(dst.subblock_structure()),
                lhs_nout: LHS_NOUT,
                lhs_nin: LHS_NIN,
                lhs_rank: lhs.subblock_structure().rank(),
                lhs_homspace: lhs.homspace().clone(),
                lhs_structure: Arc::clone(lhs.subblock_structure()),
                rhs_nout: RHS_NOUT,
                rhs_nin: RHS_NIN,
                rhs_rank: rhs.subblock_structure().rank(),
                rhs_homspace: rhs.homspace().clone(),
                rhs_structure: Arc::clone(rhs.subblock_structure()),
                axes: raw_axes,
                plan: Arc::clone(&plan),
            });
            return Ok(plan);
        }
        let plan = Arc::new(tensorcontract_fusion_explicit_plan(
            rule, dst, lhs, rhs, axes,
        )?);
        let last_key = key.clone();
        self.insert_plan(key, Arc::clone(&plan));
        self.last = Some(FusionExplicitPlanLastEntry {
            key: last_key,
            rule: rule_key,
            dst_nout: DST_NOUT,
            dst_nin: DST_NIN,
            dst_rank: dst.subblock_structure().rank(),
            dst_homspace: dst.homspace().clone(),
            dst_structure: Arc::clone(dst.subblock_structure()),
            lhs_nout: LHS_NOUT,
            lhs_nin: LHS_NIN,
            lhs_rank: lhs.subblock_structure().rank(),
            lhs_homspace: lhs.homspace().clone(),
            lhs_structure: Arc::clone(lhs.subblock_structure()),
            rhs_nout: RHS_NOUT,
            rhs_nin: RHS_NIN,
            rhs_rank: rhs.subblock_structure().rank(),
            rhs_homspace: rhs.homspace().clone(),
            rhs_structure: Arc::clone(rhs.subblock_structure()),
            axes: raw_axes,
            plan: Arc::clone(&plan),
        });
        Ok(plan)
    }
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
        if !self.structures.policy().stores_entries() {
            self.stats.structure_misses += 1;
            let structure = TensorContractStructure::compile_with_block_specs(
                dst,
                lhs,
                rhs,
                plan_key.axes().as_spec(),
                block_specs,
            )?;
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
    dynamic_space_cache: DynamicFusionSpaceCache<RuleKey>,
    route_cache: FusionRouteCache<RuleKey>,
    explicit_plan_cache: FusionExplicitPlanCache<RuleKey>,
    contract_backend: BC,
    contract_workspace: BC::Workspace,
    contract_cache: TensorContractCache<TensorContractBlockPlanKey>,
    // TensorKit-style canonical block pack/GEMM/scatter plans. Automatic fusion
    // contractions replay through this cache directly instead of storing a
    // monolithic contraction execution plan.
    fusion_block_cache: CanonicalFusionBlockContractCache<RuleKey>,
    fusion_block_workspace: CanonicalFusionBlockContractWorkspace<D>,
    fusion_scratch: DynamicFusionScratchWorkspace<D>,
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
            dynamic_space_cache: DynamicFusionSpaceCache::default(),
            route_cache: FusionRouteCache::default(),
            explicit_plan_cache: FusionExplicitPlanCache::default(),
            contract_backend,
            contract_workspace,
            contract_cache,
            fusion_block_cache: CanonicalFusionBlockContractCache::default(),
            fusion_block_workspace: CanonicalFusionBlockContractWorkspace::default(),
            fusion_scratch: DynamicFusionScratchWorkspace::default(),
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
    pub fn fusion_explicit_plan_cache_len(&self) -> usize {
        self.explicit_plan_cache.len()
    }

    #[inline]
    pub fn fusion_route_cache_len(&self) -> usize {
        self.route_cache.len()
    }

    #[inline]
    pub fn fusion_block_contract_cache_len(&self) -> usize {
        self.fusion_block_cache.len()
    }

    #[inline]
    pub fn fusion_block_contract_cache_hits(&self) -> usize {
        self.fusion_block_cache.stats().hits()
    }

    #[inline]
    pub fn fusion_block_contract_cache_fast_hits(&self) -> usize {
        self.fusion_block_cache.stats().fast_hits()
    }

    #[inline]
    pub fn fusion_block_contract_cache_misses(&self) -> usize {
        self.fusion_block_cache.stats().misses()
    }

    pub fn set_cache_policy(&mut self, policy: OperationCachePolicy) {
        self.tree_context.set_cache_policy(policy);
        self.dynamic_space_cache.set_policy(policy);
        self.route_cache.set_policy(policy);
        self.explicit_plan_cache.set_policy(policy);
        self.contract_cache.set_policy(policy);
        self.fusion_block_cache.set_policy(policy);
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
        DDst,
        DLhs,
        DRhs,
    >(
        &mut self,
        rule: &R,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        axes: TensorContractAxisSpec<'_>,
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
        if !axes.lhs_conjugate() && !axes.rhs_conjugate() {
            let route = self.route_cache.get_or_compile_nonconjugate(
                rule,
                &dst_dynamic,
                &lhs_dynamic,
                &rhs_dynamic,
                axes,
            )?;
            if route == TensorContractFusionRouteDecision::CanonicalFusionBlocks {
                let Self {
                    contract_backend,
                    contract_workspace,
                    route_cache: _,
                    explicit_plan_cache: _,
                    contract_cache: _,
                    fusion_block_cache,
                    fusion_block_workspace,
                    tree_context: _,
                    dynamic_space_cache: _,
                    fusion_scratch: _,
                } = self;
                let block_plan = fusion_block_cache.get_or_compile(
                    rule,
                    &dst_dynamic,
                    &lhs_dynamic,
                    &rhs_dynamic,
                    axes,
                )?;
                let dst_structure = std::sync::Arc::clone(dst.structure());
                let lhs_structure = std::sync::Arc::clone(lhs.structure());
                let rhs_structure = std::sync::Arc::clone(rhs.structure());
                block_plan.execute_raw(
                    contract_backend,
                    contract_workspace,
                    fusion_block_workspace,
                    &dst_structure,
                    dst.data_mut(),
                    &lhs_structure,
                    lhs.data(),
                    &rhs_structure,
                    rhs.data(),
                    alpha,
                    beta,
                )?;
                return Ok(());
            }

            let plan = self
                .explicit_plan_cache
                .get_or_compile(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)?;
            let Self {
                tree_context,
                dynamic_space_cache,
                route_cache: _,
                explicit_plan_cache: _,
                contract_backend,
                contract_workspace,
                contract_cache: _,
                fusion_block_cache,
                fusion_block_workspace,
                fusion_scratch,
            } = self;
            return tensorcontract_fusion_dynamic_plan_into_context(
                tree_context,
                contract_backend,
                contract_workspace,
                dynamic_space_cache,
                fusion_block_cache,
                fusion_block_workspace,
                fusion_scratch,
                rule,
                plan.as_ref(),
                dst,
                lhs,
                rhs,
                alpha,
                beta,
            );
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
                    let plan = self
                        .explicit_plan_cache
                        .get_or_compile(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)?;
                    let Self {
                        tree_context,
                        dynamic_space_cache,
                        route_cache: _,
                        explicit_plan_cache: _,
                        contract_backend,
                        contract_workspace,
                        contract_cache: _,
                        fusion_block_cache,
                        fusion_block_workspace,
                        fusion_scratch,
                    } = self;
                    return tensorcontract_fusion_dynamic_plan_into_context(
                        tree_context,
                        contract_backend,
                        contract_workspace,
                        dynamic_space_cache,
                        fusion_block_cache,
                        fusion_block_workspace,
                        fusion_scratch,
                        rule,
                        plan.as_ref(),
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

        let plan = self
            .explicit_plan_cache
            .get_or_compile(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)?;
        let Self {
            tree_context,
            dynamic_space_cache,
            route_cache: _,
            explicit_plan_cache: _,
            contract_backend,
            contract_workspace,
            contract_cache: _,
            fusion_block_cache,
            fusion_block_workspace,
            fusion_scratch,
        } = self;
        tensorcontract_fusion_dynamic_plan_into_context(
            tree_context,
            contract_backend,
            contract_workspace,
            dynamic_space_cache,
            fusion_block_cache,
            fusion_block_workspace,
            fusion_scratch,
            rule,
            plan.as_ref(),
            dst,
            lhs,
            rhs,
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
        axes: TensorContractAxisSpec<'_>,
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
        let route = if !axes.lhs_conjugate() && !axes.rhs_conjugate() {
            Some(self.route_cache.get_or_compile_nonconjugate(
                rule,
                &dst_dynamic,
                &lhs_dynamic,
                &rhs_dynamic,
                axes,
            )?)
        } else {
            None
        };
        profile.canonical_route_check += start.elapsed();
        if route == Some(TensorContractFusionRouteDecision::CanonicalFusionBlocks) {
            profile.route = TensorContractFusionRoute::CanonicalFusionBlocks;
            let Self {
                contract_backend,
                contract_workspace,
                route_cache: _,
                explicit_plan_cache: _,
                contract_cache: _,
                fusion_block_cache,
                fusion_block_workspace,
                tree_context: _,
                dynamic_space_cache: _,
                fusion_scratch: _,
            } = self;
            let start = std::time::Instant::now();
            let block_plan = fusion_block_cache.get_or_compile(
                rule,
                &dst_dynamic,
                &lhs_dynamic,
                &rhs_dynamic,
                axes,
            )?;
            profile.fusion_block_plan_lookup += start.elapsed();
            let dst_structure = std::sync::Arc::clone(dst.structure());
            let lhs_structure = std::sync::Arc::clone(lhs.structure());
            let rhs_structure = std::sync::Arc::clone(rhs.structure());
            let result = block_plan.execute_raw_profiled(
                contract_backend,
                contract_workspace,
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
            return result;
        }
        if route == Some(TensorContractFusionRouteDecision::DynamicTreeCanonical) {
            let start = std::time::Instant::now();
            let plan = self
                .explicit_plan_cache
                .get_or_compile(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)?;
            profile.explicit_plan += start.elapsed();
            profile.route = TensorContractFusionRoute::DynamicTreeCanonical;
            let Self {
                tree_context,
                dynamic_space_cache,
                route_cache: _,
                explicit_plan_cache: _,
                contract_backend,
                contract_workspace,
                contract_cache: _,
                fusion_block_cache,
                fusion_block_workspace,
                fusion_scratch,
            } = self;
            let result = tensorcontract_fusion_dynamic_plan_into_context_profiled(
                tree_context,
                contract_backend,
                contract_workspace,
                dynamic_space_cache,
                fusion_block_cache,
                fusion_block_workspace,
                fusion_scratch,
                rule,
                plan.as_ref(),
                dst,
                lhs,
                rhs,
                alpha,
                beta,
                profile,
            );
            profile.total += total_start.elapsed();
            return result;
        }

        if axes.lhs_conjugate() || axes.rhs_conjugate() {
            let start = std::time::Instant::now();
            match tensorcontract_fusion_structure(rule, dst, lhs, rhs, axes) {
                Ok(structure) => {
                    profile.route = TensorContractFusionRoute::DenseConjugateStructure;
                    profile.dense_structure_lookup += start.elapsed();
                    let start = std::time::Instant::now();
                    let result = self.contract_backend.tensorcontract_structure_into(
                        &mut self.contract_workspace,
                        &structure,
                        dst,
                        lhs,
                        rhs,
                        alpha,
                        beta,
                    );
                    profile.dense_contract += start.elapsed();
                    profile.total += total_start.elapsed();
                    return result;
                }
                Err(OperationError::UnsupportedTensorContractScope {
                    message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
                }) => {
                    profile.dense_structure_lookup += start.elapsed();
                    let start = std::time::Instant::now();
                    let plan = self
                        .explicit_plan_cache
                        .get_or_compile(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)?;
                    profile.explicit_plan += start.elapsed();
                    profile.route = TensorContractFusionRoute::DynamicTreeCanonical;
                    let Self {
                        tree_context,
                        dynamic_space_cache,
                        route_cache: _,
                        explicit_plan_cache: _,
                        contract_backend,
                        contract_workspace,
                        contract_cache: _,
                        fusion_block_cache,
                        fusion_block_workspace,
                        fusion_scratch,
                    } = self;
                    let result = tensorcontract_fusion_dynamic_plan_into_context_profiled(
                        tree_context,
                        contract_backend,
                        contract_workspace,
                        dynamic_space_cache,
                        fusion_block_cache,
                        fusion_block_workspace,
                        fusion_scratch,
                        rule,
                        plan.as_ref(),
                        dst,
                        lhs,
                        rhs,
                        alpha,
                        beta,
                        profile,
                    );
                    profile.total += total_start.elapsed();
                    return result;
                }
                Err(err) => {
                    profile.dense_structure_lookup += start.elapsed();
                    profile.total += total_start.elapsed();
                    return Err(err);
                }
            }
        }

        let start = std::time::Instant::now();
        let plan = self
            .explicit_plan_cache
            .get_or_compile(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)?;
        profile.explicit_plan += start.elapsed();
        profile.route = TensorContractFusionRoute::DynamicTreeCanonical;
        let Self {
            tree_context,
            dynamic_space_cache,
            route_cache: _,
            explicit_plan_cache: _,
            contract_backend,
            contract_workspace,
            contract_cache: _,
            fusion_block_cache,
            fusion_block_workspace,
            fusion_scratch,
        } = self;
        let result = tensorcontract_fusion_dynamic_plan_into_context_profiled(
            tree_context,
            contract_backend,
            contract_workspace,
            dynamic_space_cache,
            fusion_block_cache,
            fusion_block_workspace,
            fusion_scratch,
            rule,
            plan.as_ref(),
            dst,
            lhs,
            rhs,
            alpha,
            beta,
            profile,
        );
        profile.total += total_start.elapsed();
        result
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
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
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
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
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
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
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

        self.transform_source_into_canonical(
            rule,
            plan.lhs_transform().clone(),
            plan.lhs_source_conjugate(),
            lhs_canonical,
            lhs,
        )?;
        self.transform_source_into_canonical(
            rule,
            plan.rhs_transform().clone(),
            plan.rhs_source_conjugate(),
            rhs_canonical,
            rhs,
        )?;

        let dst_space = DynamicFusionMapSpace::from_typed(
            dst.fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let lhs_space = DynamicFusionMapSpace::from_typed(
            lhs_canonical
                .fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let rhs_space = DynamicFusionMapSpace::from_typed(
            rhs_canonical
                .fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let block_plan = self.fusion_block_cache.get_or_compile(
            rule,
            &dst_space,
            &lhs_space,
            &rhs_space,
            plan.canonical_axes().as_spec(),
        )?;
        let dst_structure = std::sync::Arc::clone(dst.structure());
        let lhs_structure = std::sync::Arc::clone(lhs_canonical.structure());
        let rhs_structure = std::sync::Arc::clone(rhs_canonical.structure());
        block_plan.execute_raw(
            &mut self.contract_backend,
            &mut self.contract_workspace,
            &mut self.fusion_block_workspace,
            &dst_structure,
            dst.data_mut(),
            &lhs_structure,
            lhs_canonical.data(),
            &rhs_structure,
            rhs_canonical.data(),
            alpha,
            beta,
        )
    }

    fn transform_source_into_canonical<
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
        operation: crate::TreeTransformOperationKey,
        source_conjugate: bool,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        dst.data_mut().fill(D::zero());
        let src_fusion = src
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let src_replay_structure = if source_conjugate {
            Arc::clone(adjoint_fusion_space_view(src_fusion)?.subblock_structure())
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
        self.tree_context.tree_pair_transform_structure_into_raw(
            structure.as_ref(),
            &dst_structure,
            &src_replay_structure,
            dst.data_mut(),
            src.data(),
            D::one(),
            D::zero(),
        )
    }
}
