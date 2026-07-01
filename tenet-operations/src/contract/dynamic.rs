use std::collections::HashMap;
use std::hash::Hash;

use tenet_core::{BlockStructure, CoreError, MultiplicityFreeRigidSymbols, TensorMap};

use crate::axis::{AxisPermutation, OwnedTensorContractAxisSpec, TensorContractAxisSpec};
use crate::cache::BlockStructureCacheKey;
use crate::lowering::{adjoint_fusion_space_view, lower_tensorcontract_adjoint_axes};
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
    tensorcontract_canonical_fusion_blocks_into_raw, CanonicalFusionBlockContractPlan,
    CanonicalFusionBlockContractWorkspace,
};
use super::scratch::{DynamicFusionScratch, DynamicFusionScratchWorkspace};
use super::structure::TensorContractAxisPlan;

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
#[allow(clippy::too_many_arguments)]
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
>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    execution_cache: &mut DynamicFusionExecutionPlanCache<RuleKey>,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    fusion_block_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    precomputed_key: Option<DynamicFusionExecutionPlanPreparedCacheKey<RuleKey>>,
    rule: &R,
    axes: TensorContractAxisSpec<'_>,
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
    let execution_plan =
        execution_cache.get_or_compile_with_key(precomputed_key, rule, axes, dst, lhs, rhs)?;
    execution_plan.execute(
        tree_context,
        contract_backend,
        contract_workspace,
        fusion_block_workspace,
        scratch,
        dst,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DynamicFusionExecutionPlanCacheStats {
    hits: usize,
    misses: usize,
}

impl DynamicFusionExecutionPlanCacheStats {
    #[inline]
    pub(crate) fn hits(self) -> usize {
        self.hits
    }

    #[inline]
    pub(crate) fn misses(self) -> usize {
        self.misses
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionExecutionPlanCache<RuleKey> {
    plans: HashMap<DynamicFusionExecutionPlanCacheKey<RuleKey>, DynamicFusionExecutionPlan>,
    last_plan: Option<DynamicFusionExecutionPlanLastKey<RuleKey>>,
    stats: DynamicFusionExecutionPlanCacheStats,
}

impl<RuleKey> Default for DynamicFusionExecutionPlanCache<RuleKey> {
    fn default() -> Self {
        Self {
            plans: HashMap::new(),
            last_plan: None,
            stats: DynamicFusionExecutionPlanCacheStats::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DynamicFusionExecutionPlanRawFastKey<RuleKey> {
    rule: RuleKey,
    dst_nout: usize,
    lhs_nout: usize,
    rhs_nout: usize,
    dst_fusion_ptr: *const (),
    lhs_fusion_ptr: *const (),
    rhs_fusion_ptr: *const (),
    dst_structure_ptr: *const BlockStructure,
    lhs_structure_ptr: *const BlockStructure,
    rhs_structure_ptr: *const BlockStructure,
    lhs_contracting_axes: Vec<usize>,
    rhs_contracting_axes: Vec<usize>,
    output_axes: Option<Vec<usize>>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionExecutionPlanPreparedCacheKey<RuleKey> {
    raw_fast: DynamicFusionExecutionPlanRawFastKey<RuleKey>,
    full: DynamicFusionExecutionPlanCacheKey<RuleKey>,
}

#[derive(Clone, Debug)]
struct DynamicFusionExecutionPlanLastKey<RuleKey> {
    raw_fast: DynamicFusionExecutionPlanRawFastKey<RuleKey>,
    full: DynamicFusionExecutionPlanCacheKey<RuleKey>,
}

pub(crate) enum DynamicFusionExecutionPlanCacheLookup<'a, RuleKey> {
    Hit(&'a DynamicFusionExecutionPlan),
    Miss(DynamicFusionExecutionPlanPreparedCacheKey<RuleKey>),
}

impl<RuleKey> DynamicFusionExecutionPlanCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.plans.len()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    #[inline]
    pub(crate) fn stats(&self) -> DynamicFusionExecutionPlanCacheStats {
        self.stats
    }

    pub(crate) fn get_cached_or_key<
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
        &mut self,
        rule: &R,
        axes: TensorContractAxisSpec<'_>,
        dst: &TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    ) -> Result<DynamicFusionExecutionPlanCacheLookup<'_, RuleKey>, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        if self
            .last_plan
            .as_ref()
            .is_some_and(|last| last.raw_fast.matches_inputs(rule, axes, dst, lhs, rhs))
        {
            let key = self
                .last_plan
                .as_ref()
                .expect("last plan checked above")
                .full
                .clone();
            self.stats.hits += 1;
            return Ok(DynamicFusionExecutionPlanCacheLookup::Hit(
                self.plans
                    .get(&key)
                    .expect("last dynamic fusion plan key must reference a cached plan"),
            ));
        }
        let prepared =
            DynamicFusionExecutionPlanPreparedCacheKey::from_inputs(rule, axes, dst, lhs, rhs)?;
        if self.plans.get(&prepared.full).is_some() {
            self.stats.hits += 1;
            self.last_plan = Some(DynamicFusionExecutionPlanLastKey {
                raw_fast: prepared.raw_fast.clone(),
                full: prepared.full.clone(),
            });
            return Ok(DynamicFusionExecutionPlanCacheLookup::Hit(
                self.plans
                    .get(&prepared.full)
                    .expect("dynamic fusion plan found before replay"),
            ));
        }
        Ok(DynamicFusionExecutionPlanCacheLookup::Miss(prepared))
    }

    pub(crate) fn get_or_compile_with_key<
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
        &mut self,
        key: Option<DynamicFusionExecutionPlanPreparedCacheKey<RuleKey>>,
        rule: &R,
        axes: TensorContractAxisSpec<'_>,
        dst: &TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    ) -> Result<&DynamicFusionExecutionPlan, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let prepared = match key {
            Some(key) => key,
            None => {
                DynamicFusionExecutionPlanPreparedCacheKey::from_inputs(rule, axes, dst, lhs, rhs)?
            }
        };
        if self.plans.get(&prepared.full).is_some() {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
            let dst_fusion = dst
                .fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
            let lhs_fusion = lhs
                .fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
            let rhs_fusion = rhs
                .fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
            let plan = tensorcontract_fusion_explicit_plan(
                rule, dst_fusion, lhs_fusion, rhs_fusion, axes,
            )?;
            let execution_plan = DynamicFusionExecutionPlan::compile(
                rule,
                &plan,
                dst_fusion,
                dst.structure(),
                lhs,
                rhs,
            )?;
            self.plans.insert(prepared.full.clone(), execution_plan);
        }
        self.last_plan = Some(DynamicFusionExecutionPlanLastKey {
            raw_fast: prepared.raw_fast.clone(),
            full: prepared.full.clone(),
        });
        Ok(self
            .plans
            .get(&prepared.full)
            .expect("dynamic fusion execution plan inserted before replay"))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionExecutionPlan {
    lhs: DynamicFusionSourceTransformPlan,
    rhs: DynamicFusionSourceTransformPlan,
    contract: DynamicFusionCanonicalContractPlan,
    output: Option<DynamicFusionOutputTransformPlan>,
}

impl DynamicFusionExecutionPlan {
    fn compile<
        R,
        D,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        SLhs,
        SRhs,
    >(
        rule: &R,
        plan: &TensorContractFusionExplicitPlan,
        dst_fusion: &tenet_core::FusionTensorMapSpace<DST_NOUT, DST_NIN>,
        dst_structure: &std::sync::Arc<BlockStructure>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let lhs = DynamicFusionSourceTransformPlan::compile(
            rule,
            lhs,
            plan.lhs_transform().clone(),
            plan.lhs_source_conjugate(),
        )?;
        let rhs = DynamicFusionSourceTransformPlan::compile(
            rule,
            rhs,
            plan.rhs_transform().clone(),
            plan.rhs_source_conjugate(),
        )?;

        if plan.output_transform_is_identity() {
            let dst_space = DynamicFusionMapSpace::from_typed(dst_fusion);
            let block_plan = CanonicalFusionBlockContractPlan::compile(
                rule,
                &dst_space,
                &lhs.dst_space,
                &rhs.dst_space,
                plan.canonical_axes().as_spec(),
            )?;
            return Ok(Self {
                lhs,
                rhs,
                contract: DynamicFusionCanonicalContractPlan {
                    dst_space,
                    block_plan,
                },
                output: None,
            });
        }

        let output_dst_space = DynamicFusionMapSpace::from_typed(dst_fusion);
        let canonical_dst_space = DynamicFusionMapSpace::canonical_dst(
            rule,
            &lhs.dst_space,
            &rhs.dst_space,
            plan,
            Some(&output_dst_space),
        )?;
        let block_plan = CanonicalFusionBlockContractPlan::compile(
            rule,
            &canonical_dst_space,
            &lhs.dst_space,
            &rhs.dst_space,
            plan.canonical_axes().as_spec(),
        )?;
        let output_structure = compile_tree_pair_structure(
            rule,
            plan.output_transform().clone(),
            dst_structure,
            canonical_dst_space.structure(),
            false,
        )?;
        Ok(Self {
            lhs,
            rhs,
            contract: DynamicFusionCanonicalContractPlan {
                dst_space: canonical_dst_space.clone(),
                block_plan,
            },
            output: Some(DynamicFusionOutputTransformPlan {
                src_space: canonical_dst_space,
                structure: output_structure,
            }),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute<
        BT,
        BC,
        D,
        RuleKey,
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
        &self,
        tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        contract_backend: &mut BC,
        contract_workspace: &mut BC::Workspace,
        fusion_block_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
        scratch: &mut DynamicFusionScratchWorkspace<D>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        RuleKey: Clone + Eq + Hash,
        BT: TreeTransformBackend<D, f64>,
        BC: TensorContractBackend<D, f64>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        {
            let lhs_scratch = scratch.prepare_lhs(self.lhs.dst_space.clone())?;
            self.lhs.execute(tree_context, lhs_scratch, lhs)?;
        }
        {
            let rhs_scratch = scratch.prepare_rhs(self.rhs.dst_space.clone())?;
            self.rhs.execute(tree_context, rhs_scratch, rhs)?;
        }

        if let Some(output) = &self.output {
            scratch.prepare_dst(output.src_space.clone())?;
            {
                let (lhs_canonical, rhs_canonical, canonical_dst) = scratch.lhs_rhs_dst_mut();
                self.contract.block_plan.execute_raw(
                    contract_backend,
                    contract_workspace,
                    fusion_block_workspace,
                    self.contract.dst_space.structure(),
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
            output.execute(
                tree_context,
                &dst_structure,
                dst.data_mut(),
                scratch.dst(),
                D::one(),
                beta,
            )
        } else {
            let dst_structure = std::sync::Arc::clone(dst.structure());
            let (lhs_canonical, rhs_canonical) = scratch.lhs_rhs();
            self.contract.block_plan.execute_raw(
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
            )
        }
    }
}

#[derive(Clone, Debug)]
struct DynamicFusionSourceTransformPlan {
    dst_space: DynamicFusionMapSpace,
    src_replay_structure: std::sync::Arc<BlockStructure>,
    structure: TreeTransformStructure<f64>,
}

impl DynamicFusionSourceTransformPlan {
    fn compile<R, D, const SRC_NOUT: usize, const SRC_NIN: usize, SSrc>(
        rule: &R,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
        operation: TreeTransformOperationKey,
        source_conjugate: bool,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let src_fusion = src
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
        let (dst_space, src_replay_structure) = if source_conjugate {
            let adjoint = adjoint_fusion_space_view(src_fusion)?;
            let replay_structure = std::sync::Arc::clone(adjoint.subblock_structure());
            let space = DynamicFusionMapSpace::transformed_from_typed(rule, &adjoint, &operation)?;
            (space, replay_structure)
        } else {
            let replay_structure = std::sync::Arc::clone(src.structure());
            let space =
                DynamicFusionMapSpace::transformed_from_typed(rule, src_fusion, &operation)?;
            (space, replay_structure)
        };
        let structure = compile_tree_pair_structure(
            rule,
            operation,
            dst_space.structure(),
            &src_replay_structure,
            source_conjugate,
        )?;
        Ok(Self {
            dst_space,
            src_replay_structure,
            structure,
        })
    }

    fn execute<BT, D, RuleKey, const SRC_NOUT: usize, const SRC_NIN: usize, SSrc>(
        &self,
        tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        dst: &mut DynamicFusionScratch<D>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    ) -> Result<(), OperationError>
    where
        RuleKey: Clone + Eq + Hash,
        BT: TreeTransformBackend<D, f64>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        dst.fill_zero();
        tree_context.tree_transform_structure_into_raw(
            &self.structure,
            self.dst_space.structure(),
            &self.src_replay_structure,
            dst.data_mut(),
            src.data(),
            D::one(),
            D::zero(),
        )
    }
}

#[derive(Clone, Debug)]
struct DynamicFusionCanonicalContractPlan {
    dst_space: DynamicFusionMapSpace,
    block_plan: CanonicalFusionBlockContractPlan,
}

#[derive(Clone, Debug)]
struct DynamicFusionOutputTransformPlan {
    src_space: DynamicFusionMapSpace,
    structure: TreeTransformStructure<f64>,
}

impl DynamicFusionOutputTransformPlan {
    fn execute<BT, D, RuleKey>(
        &self,
        tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
        dst_structure: &std::sync::Arc<BlockStructure>,
        dst_data: &mut [D],
        src: &DynamicFusionScratch<D>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        RuleKey: Clone + Eq + Hash,
        BT: TreeTransformBackend<D, f64>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    {
        tree_context.tree_transform_structure_into_raw(
            &self.structure,
            dst_structure,
            self.src_space.structure(),
            dst_data,
            src.data(),
            alpha,
            beta,
        )
    }
}

fn compile_tree_pair_structure<R>(
    rule: &R,
    operation: TreeTransformOperationKey,
    dst_structure: &std::sync::Arc<BlockStructure>,
    src_structure: &std::sync::Arc<BlockStructure>,
    storage_conjugate: bool,
) -> Result<TreeTransformStructure<f64>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let plan = build_tree_pair_transform_group_plan(rule, operation, src_structure)?;
    plan.compile_shared_structures_with_storage_conjugation(
        std::sync::Arc::clone(dst_structure),
        std::sync::Arc::clone(src_structure),
        storage_conjugate,
    )
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct DynamicFusionExecutionPlanCacheKey<RuleKey> {
    rule: RuleKey,
    dst_nout: usize,
    dst_homspace: tenet_core::FusionTreeHomSpace,
    dst_structure: BlockStructureCacheKey,
    lhs_nout: usize,
    lhs_homspace: tenet_core::FusionTreeHomSpace,
    lhs_structure: BlockStructureCacheKey,
    rhs_nout: usize,
    rhs_homspace: tenet_core::FusionTreeHomSpace,
    rhs_structure: BlockStructureCacheKey,
    axes: OwnedTensorContractAxisSpec,
}

impl<RuleKey> DynamicFusionExecutionPlanCacheKey<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[allow(clippy::too_many_arguments)]
    fn from_inputs<
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
        rule: &R,
        axes: TensorContractAxisSpec<'_>,
        dst: &TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    ) -> Result<Self, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
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
        let lowered_axes =
            lower_tensorcontract_adjoint_axes::<LHS_NOUT, LHS_NIN, RHS_NOUT, RHS_NIN>(axes)?;
        let lowered_spec = lowered_axes.as_spec();
        let axis_plan = TensorContractAxisPlan::compile(
            lhs.structure().rank(),
            rhs.structure().rank(),
            dst.structure().rank(),
            lowered_spec,
        )?;
        Ok(Self {
            rule: rule.tree_transform_rule_cache_key(),
            dst_nout: DST_NOUT,
            dst_homspace: dst_fusion.homspace().clone(),
            dst_structure: BlockStructureCacheKey::from_structure(dst.structure())?,
            lhs_nout: LHS_NOUT,
            lhs_homspace: lhs_fusion.homspace().clone(),
            lhs_structure: BlockStructureCacheKey::from_structure(lhs.structure())?,
            rhs_nout: RHS_NOUT,
            rhs_homspace: rhs_fusion.homspace().clone(),
            rhs_structure: BlockStructureCacheKey::from_structure(rhs.structure())?,
            axes: OwnedTensorContractAxisSpec::new_with_conjugation(
                axis_plan.lhs_contracting_axes,
                axis_plan.rhs_contracting_axes,
                axis_plan.output_axes,
                lowered_spec.lhs_conjugate(),
                lowered_spec.rhs_conjugate(),
            ),
        })
    }
}

impl<RuleKey> DynamicFusionExecutionPlanRawFastKey<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[allow(clippy::too_many_arguments)]
    fn from_inputs<
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
        rule: &R,
        axes: TensorContractAxisSpec<'_>,
        dst: &TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    ) -> Result<Self, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
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
        Ok(Self {
            rule: rule.tree_transform_rule_cache_key(),
            dst_nout: DST_NOUT,
            lhs_nout: LHS_NOUT,
            rhs_nout: RHS_NOUT,
            dst_fusion_ptr: std::sync::Arc::as_ptr(dst_fusion) as *const (),
            lhs_fusion_ptr: std::sync::Arc::as_ptr(lhs_fusion) as *const (),
            rhs_fusion_ptr: std::sync::Arc::as_ptr(rhs_fusion) as *const (),
            dst_structure_ptr: std::sync::Arc::as_ptr(dst.structure()),
            lhs_structure_ptr: std::sync::Arc::as_ptr(lhs.structure()),
            rhs_structure_ptr: std::sync::Arc::as_ptr(rhs.structure()),
            lhs_contracting_axes: axes.lhs_contracting_axes().to_vec(),
            rhs_contracting_axes: axes.rhs_contracting_axes().to_vec(),
            output_axes: match axes.output_permutation() {
                AxisPermutation::Identity => None,
                AxisPermutation::Axes(axes) => Some(axes.to_vec()),
            },
            lhs_conjugate: axes.lhs_conjugate(),
            rhs_conjugate: axes.rhs_conjugate(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn matches_inputs<
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
        &self,
        rule: &R,
        axes: TensorContractAxisSpec<'_>,
        dst: &TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    ) -> bool
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let Some(dst_fusion) = dst.fusion_space() else {
            return false;
        };
        let Some(lhs_fusion) = lhs.fusion_space() else {
            return false;
        };
        let Some(rhs_fusion) = rhs.fusion_space() else {
            return false;
        };
        self.rule == rule.tree_transform_rule_cache_key()
            && self.dst_nout == DST_NOUT
            && self.lhs_nout == LHS_NOUT
            && self.rhs_nout == RHS_NOUT
            && self.dst_fusion_ptr == std::sync::Arc::as_ptr(dst_fusion) as *const ()
            && self.lhs_fusion_ptr == std::sync::Arc::as_ptr(lhs_fusion) as *const ()
            && self.rhs_fusion_ptr == std::sync::Arc::as_ptr(rhs_fusion) as *const ()
            && self.dst_structure_ptr == std::sync::Arc::as_ptr(dst.structure())
            && self.lhs_structure_ptr == std::sync::Arc::as_ptr(lhs.structure())
            && self.rhs_structure_ptr == std::sync::Arc::as_ptr(rhs.structure())
            && self.lhs_contracting_axes.as_slice() == axes.lhs_contracting_axes()
            && self.rhs_contracting_axes.as_slice() == axes.rhs_contracting_axes()
            && self.lhs_conjugate == axes.lhs_conjugate()
            && self.rhs_conjugate == axes.rhs_conjugate()
            && match (self.output_axes.as_deref(), axes.output_permutation()) {
                (None, AxisPermutation::Identity) => true,
                (Some(expected), AxisPermutation::Axes(actual)) => expected == actual,
                _ => false,
            }
    }
}

impl<RuleKey> DynamicFusionExecutionPlanPreparedCacheKey<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[allow(clippy::too_many_arguments)]
    fn from_inputs<
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
        rule: &R,
        axes: TensorContractAxisSpec<'_>,
        dst: &TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    ) -> Result<Self, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        Ok(Self {
            raw_fast: DynamicFusionExecutionPlanRawFastKey::from_inputs(rule, axes, dst, lhs, rhs)?,
            full: DynamicFusionExecutionPlanCacheKey::from_inputs(rule, axes, dst, lhs, rhs)?,
        })
    }
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
