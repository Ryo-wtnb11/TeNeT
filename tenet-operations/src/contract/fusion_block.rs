use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{
    BlockKey, BlockStructure, FusionTreeHomSpace, FusionTreeKey, MultiplicityFreeRigidSymbols,
    Placement, SectorId,
};

use crate::axis::{AxisPermutation, OwnedTensorContractAxisSpec, TensorContractAxisSpec};
use crate::cache::{
    rebuild_lru_order_from_keys, touch_lru_key, BlockStructureCacheKey, OperationCachePolicy,
};
use crate::host_scratch::HostScratchBuffer;
use crate::strided::{
    column_major_strides_isize, column_major_strides_usize, element_count, offset_to_isize,
    strides_to_isize,
};
use crate::structure_identity::validate_structure_identity;
use crate::{
    axpby_raw_strided_kernel_trusted, copy_scale_raw_strided_kernel_trusted,
    scale_raw_strided_kernel_trusted, ConjugateValue, DenseBlockScalar, OperationError,
    RecouplingCoefficientAction, ReportsPlacement, TreeTransformRuleCacheKey,
};

use super::backend::TensorContractBackend;
use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::{reject_fusion_contract_conjugation, rhs_contract_twist_factor};
use super::profile::TensorContractFusionProfile;
use super::structure::TensorContractAxisPlan;

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_canonical_fusion_blocks_into_raw<B, R, D>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs_space: &DynamicFusionMapSpace,
    lhs_data: &[D],
    rhs_space: &DynamicFusionMapSpace,
    rhs_data: &[D],
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    let plan =
        CanonicalFusionBlockContractPlan::compile(rule, dst_space, lhs_space, rhs_space, axes)?;
    let mut fusion_workspace = CanonicalFusionBlockContractWorkspace::<D>::default();
    plan.execute_raw(
        backend,
        workspace,
        &mut fusion_workspace,
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

pub(crate) fn is_canonical_fusion_block_contract<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractAxisSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    reject_fusion_contract_conjugation(axes)?;
    let axis_plan = TensorContractAxisPlan::compile(
        lhs_space.rank(),
        rhs_space.rank(),
        dst_space.rank(),
        axes,
    )?;
    if !is_canonical_source(lhs_space, rhs_space, &axis_plan)
        || !is_canonical_output(dst_space, lhs_space, rhs_space, &axis_plan)
    {
        return Ok(false);
    }
    let expected_homspace = FusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs_space.homspace(),
        rhs_space.homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axis_plan.output_axes.as_slice(),
        dst_space.nout(),
    )
    .map_err(OperationError::from_core_preserving_context)?;
    if expected_homspace != *dst_space.homspace() {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    Ok(true)
}

fn validate_canonical_compose<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractAxisSpec<'_>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if is_canonical_fusion_block_contract(rule, dst_space, lhs_space, rhs_space, axes)? {
        Ok(())
    } else {
        Err(OperationError::UnsupportedTensorContractScope {
            message: "canonical fusion-block contraction requires canonical source and output axes",
        })
    }
}

fn is_canonical_source(
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let lhs_domain_axes = (lhs_space.nout()..lhs_space.rank()).collect::<Vec<_>>();
    let rhs_codomain_axes = (0..rhs_space.nout()).collect::<Vec<_>>();
    axis_plan.lhs_contracting_axes == lhs_domain_axes
        && axis_plan.rhs_contracting_axes == rhs_codomain_axes
}

fn is_canonical_output(
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let output_rank = lhs_space.nout() + (rhs_space.rank() - rhs_space.nout());
    let canonical_output_axes = (0..output_rank).collect::<Vec<_>>();
    dst_space.nout() == lhs_space.nout() && axis_plan.output_axes == canonical_output_axes
}

/// Host scratch workspace for canonical fusion-block contraction.
///
/// This workspace packs blocks into host `Vec<T>` buffers before dense replay.
/// Device execution needs a separate device workspace.
#[derive(Clone, Debug)]
pub(crate) struct HostCanonicalFusionBlockContractWorkspace<T> {
    buffers: HostFusionBlockContractBuffers<T>,
}

pub(crate) type CanonicalFusionBlockContractWorkspace<T> =
    HostCanonicalFusionBlockContractWorkspace<T>;

impl<T> Default for HostCanonicalFusionBlockContractWorkspace<T> {
    fn default() -> Self {
        Self {
            buffers: HostFusionBlockContractBuffers::default(),
        }
    }
}

impl<T> ReportsPlacement for HostCanonicalFusionBlockContractWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

#[derive(Clone, Debug)]
struct HostFusionBlockContractBuffers<T> {
    lhs: HostScratchBuffer<T>,
    rhs: HostScratchBuffer<T>,
    dst: HostScratchBuffer<T>,
}

impl<T> Default for HostFusionBlockContractBuffers<T> {
    fn default() -> Self {
        Self {
            lhs: HostScratchBuffer::default(),
            rhs: HostScratchBuffer::default(),
            dst: HostScratchBuffer::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_fusion_block_workspace_is_explicit_host_workspace() {
        let workspace = HostCanonicalFusionBlockContractWorkspace::<f64>::default();
        let alias = CanonicalFusionBlockContractWorkspace::<f64>::default();

        assert_eq!(workspace.placement(), Placement::Host);
        assert!(workspace.is_host_placement());
        assert_eq!(alias.placement(), Placement::Host);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CanonicalFusionBlockContractPlan {
    dst_structure: Arc<BlockStructure>,
    lhs_structure: Arc<BlockStructure>,
    rhs_structure: Arc<BlockStructure>,
    inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
    groups: Vec<CanonicalFusionBlockContractGroupPlan>,
}

impl CanonicalFusionBlockContractPlan {
    pub(crate) fn compile<R>(
        rule: &R,
        dst_space: &DynamicFusionMapSpace,
        lhs_space: &DynamicFusionMapSpace,
        rhs_space: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        reject_fusion_contract_conjugation(axes)?;
        validate_canonical_compose(rule, dst_space, lhs_space, rhs_space, axes)?;
        let axis_plan = TensorContractAxisPlan::compile(
            lhs_space.rank(),
            rhs_space.rank(),
            dst_space.rank(),
            axes,
        )?;

        let lhs_layout = FusionBlockMatrixLayout::compile(rule, lhs_space, None)?;
        let rhs_layout = FusionBlockMatrixLayout::compile(
            rule,
            rhs_space,
            Some(&axis_plan.rhs_contracting_axes),
        )?;
        let dst_layout = FusionBlockMatrixLayout::compile(rule, dst_space, None)?;

        let mut groups = Vec::new();
        let mut active_dst_blocks = HashSet::<usize>::new();
        for lhs_group in lhs_layout.groups {
            let Some(rhs_group) = rhs_layout.group(lhs_group.coupled) else {
                continue;
            };
            let Some(dst_group) = dst_layout.group(lhs_group.coupled) else {
                continue;
            };
            for block_index in &dst_group.block_indices {
                debug_assert!(
                    !active_dst_blocks.contains(block_index),
                    "canonical fusion-block dst subblock must be scattered exactly once"
                );
            }
            active_dst_blocks.extend(dst_group.block_indices.iter().copied());
            groups.push(CanonicalFusionBlockContractGroupPlan::compile(
                lhs_group,
                rhs_group.clone(),
                dst_group.clone(),
            )?);
        }
        Ok(Self {
            dst_structure: Arc::clone(dst_space.structure()),
            lhs_structure: Arc::clone(lhs_space.structure()),
            rhs_structure: Arc::clone(rhs_space.structure()),
            inactive_dst_scale_blocks: fusion_scale_block_layouts_excluding(
                dst_space.structure(),
                &active_dst_blocks,
            )?,
            groups,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_raw<B, D>(
        &self,
        backend: &mut B,
        workspace: &mut B::Workspace,
        fusion_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
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
        B: TensorContractBackend<D, f64>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    {
        self.validate_replay_inputs(
            dst_structure,
            dst_data.len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        scale_all_blocks(&self.inactive_dst_scale_blocks, dst_data, beta)?;

        for group in &self.groups {
            fusion_workspace
                .buffers
                .prepare(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
            fusion_workspace
                .buffers
                .clear_inputs(group.lhs.needs_clear, group.rhs.needs_clear);
            pack_group(
                &group.lhs,
                lhs_data,
                fusion_workspace.buffers.lhs.as_mut_slice(),
            )?;
            pack_group(
                &group.rhs,
                rhs_data,
                fusion_workspace.buffers.rhs.as_mut_slice(),
            )?;
            matmul_group_plan(
                backend,
                workspace,
                group,
                fusion_workspace.buffers.lhs.as_slice(),
                fusion_workspace.buffers.rhs.as_slice(),
                fusion_workspace.buffers.dst.as_mut_slice(),
            )?;
            scatter_group(
                &group.dst,
                dst_data,
                fusion_workspace.buffers.dst.as_slice(),
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_raw_profiled<B, D>(
        &self,
        backend: &mut B,
        workspace: &mut B::Workspace,
        fusion_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
        profile: &mut TensorContractFusionProfile,
    ) -> Result<(), OperationError>
    where
        B: TensorContractBackend<D, f64>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    {
        let total_start = std::time::Instant::now();

        let start = std::time::Instant::now();
        self.validate_replay_inputs(
            dst_structure,
            dst_data.len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        profile.canonical_validate += start.elapsed();

        let start = std::time::Instant::now();
        scale_all_blocks(&self.inactive_dst_scale_blocks, dst_data, beta)?;
        profile.canonical_scale += start.elapsed();

        for group in &self.groups {
            profile.canonical_contract_groups += 1;

            let start = std::time::Instant::now();
            fusion_workspace
                .buffers
                .prepare(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
            fusion_workspace
                .buffers
                .clear_inputs(group.lhs.needs_clear, group.rhs.needs_clear);
            profile.canonical_workspace_prepare += start.elapsed();

            let start = std::time::Instant::now();
            pack_group(
                &group.lhs,
                lhs_data,
                fusion_workspace.buffers.lhs.as_mut_slice(),
            )?;
            profile.canonical_pack_lhs += start.elapsed();

            let start = std::time::Instant::now();
            pack_group(
                &group.rhs,
                rhs_data,
                fusion_workspace.buffers.rhs.as_mut_slice(),
            )?;
            profile.canonical_pack_rhs += start.elapsed();

            let start = std::time::Instant::now();
            matmul_group_plan(
                backend,
                workspace,
                group,
                fusion_workspace.buffers.lhs.as_slice(),
                fusion_workspace.buffers.rhs.as_slice(),
                fusion_workspace.buffers.dst.as_mut_slice(),
            )?;
            profile.canonical_matmul += start.elapsed();

            let start = std::time::Instant::now();
            scatter_group(
                &group.dst,
                dst_data,
                fusion_workspace.buffers.dst.as_slice(),
                alpha,
                beta,
            )?;
            profile.canonical_scatter += start.elapsed();
        }

        profile.canonical_contract_total += total_start.elapsed();
        Ok(())
    }

    fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("lhs", &self.lhs_structure, lhs_structure)?;
        validate_structure_identity("rhs", &self.rhs_structure, rhs_structure)
    }

    fn validate_replay_inputs(
        &self,
        dst_structure: &Arc<BlockStructure>,
        dst_len: usize,
        lhs_structure: &Arc<BlockStructure>,
        lhs_len: usize,
        rhs_structure: &Arc<BlockStructure>,
        rhs_len: usize,
    ) -> Result<(), OperationError> {
        self.validate_replay_structures(dst_structure, lhs_structure, rhs_structure)?;
        validate_storage_len(dst_structure, dst_len)?;
        validate_storage_len(lhs_structure, lhs_len)?;
        validate_storage_len(rhs_structure, rhs_len)
    }
}

fn validate_storage_len(
    structure: &BlockStructure,
    actual_len: usize,
) -> Result<(), OperationError> {
    let expected = structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    if actual_len != expected {
        return Err(OperationError::ElementCountMismatch {
            expected,
            actual: actual_len,
        });
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct CanonicalFusionBlockContractGroupPlan {
    lhs: FusionBlockMatrixGroup,
    rhs: FusionBlockMatrixGroup,
    dst: FusionBlockMatrixGroup,
}

impl CanonicalFusionBlockContractGroupPlan {
    fn compile(
        lhs: FusionBlockMatrixGroup,
        rhs: FusionBlockMatrixGroup,
        dst: FusionBlockMatrixGroup,
    ) -> Result<Self, OperationError> {
        if lhs.cols != rhs.rows {
            return Err(OperationError::ShapeMismatch {
                dst: vec![lhs.cols],
                src: vec![rhs.rows],
            });
        }
        if dst.rows != lhs.rows || dst.cols != rhs.cols {
            return Err(OperationError::ShapeMismatch {
                dst: vec![dst.rows, dst.cols],
                src: vec![lhs.rows, rhs.cols],
            });
        }

        Ok(Self { lhs, rhs, dst })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CanonicalFusionBlockContractCacheStats {
    hits: usize,
    fast_hits: usize,
    misses: usize,
}

impl CanonicalFusionBlockContractCacheStats {
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
pub(crate) struct CanonicalFusionBlockContractCache<RuleKey> {
    last: Option<CanonicalFusionBlockContractLastEntry<RuleKey>>,
    fast_plans: HashMap<
        CanonicalFusionBlockContractFastKey<RuleKey>,
        Arc<CanonicalFusionBlockContractPlan>,
    >,
    plans: HashMap<
        CanonicalFusionBlockContractCacheKey<RuleKey>,
        Arc<CanonicalFusionBlockContractPlan>,
    >,
    plan_lru_order: VecDeque<CanonicalFusionBlockContractCacheKey<RuleKey>>,
    policy: OperationCachePolicy,
    stats: CanonicalFusionBlockContractCacheStats,
}

impl<RuleKey> Default for CanonicalFusionBlockContractCache<RuleKey> {
    fn default() -> Self {
        Self {
            last: None,
            fast_plans: HashMap::new(),
            plans: HashMap::new(),
            plan_lru_order: VecDeque::new(),
            policy: OperationCachePolicy::default(),
            stats: CanonicalFusionBlockContractCacheStats::default(),
        }
    }
}

impl<RuleKey> CanonicalFusionBlockContractCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.plans.len()
    }

    #[inline]
    pub(crate) fn stats(&self) -> CanonicalFusionBlockContractCacheStats {
        self.stats
    }

    pub(crate) fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.last = None;
        self.fast_plans.clear();
        if !policy.stores_entries() {
            self.plans.clear();
            self.plan_lru_order.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            rebuild_lru_order_from_keys(&self.plans, &mut self.plan_lru_order);
            self.enforce_lru_limit(max_entries);
        }
    }

    fn touch_plan(&mut self, key: &CanonicalFusionBlockContractCacheKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.plans.contains_key(key) {
            touch_lru_key(&mut self.plan_lru_order, key);
        }
    }

    fn insert_plan(
        &mut self,
        key: CanonicalFusionBlockContractCacheKey<RuleKey>,
        fast_key: CanonicalFusionBlockContractFastKey<RuleKey>,
        plan: Arc<CanonicalFusionBlockContractPlan>,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.plans.insert(key.clone(), Arc::clone(&plan));
        self.fast_plans.insert(fast_key, plan);
        if self.policy.max_entries().is_some() {
            self.touch_plan(&key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_lru_limit(max_entries);
        }
    }

    fn enforce_lru_limit(&mut self, max_entries: usize) {
        let mut evicted = false;
        while self.plans.len() > max_entries {
            let Some(oldest) = self.plan_lru_order.pop_front() else {
                break;
            };
            evicted |= self.plans.remove(&oldest).is_some();
        }
        if evicted {
            self.fast_plans.clear();
            self.last = None;
        }
    }

    pub(crate) fn get_or_compile<R>(
        &mut self,
        rule: &R,
        dst_space: &DynamicFusionMapSpace,
        lhs_space: &DynamicFusionMapSpace,
        rhs_space: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Arc<CanonicalFusionBlockContractPlan>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        let raw_axes = RawTensorContractAxisSpecKey::from_axes(axes);
        if self.policy.stores_entries() {
            let refresh_lru = self.policy.max_entries().is_some();
            let last_hit = self.last.as_ref().and_then(|last| {
                if last.matches(&rule_key, dst_space, lhs_space, rhs_space, axes) {
                    Some((
                        refresh_lru.then(|| last.key.clone()).flatten(),
                        Arc::clone(&last.plan),
                    ))
                } else {
                    None
                }
            });
            if let Some((key, plan)) = last_hit {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if let Some(key) = key.as_ref() {
                    self.touch_plan(key);
                }
                return Ok(plan);
            }
        }
        let axis_plan = TensorContractAxisPlan::compile(
            lhs_space.rank(),
            rhs_space.rank(),
            dst_space.rank(),
            axes,
        )?;
        let axes_key = OwnedTensorContractAxisSpec::new_with_conjugation(
            axis_plan.lhs_contracting_axes,
            axis_plan.rhs_contracting_axes,
            axis_plan.output_axes,
            axis_plan.lhs_conjugate,
            axis_plan.rhs_conjugate,
        );
        let fast_key = CanonicalFusionBlockContractFastKey {
            rule: rule_key.clone(),
            dst: CanonicalFusionBlockFastSpaceKey::from_space(dst_space),
            lhs: CanonicalFusionBlockFastSpaceKey::from_space(lhs_space),
            rhs: CanonicalFusionBlockFastSpaceKey::from_space(rhs_space),
            axes: axes_key.clone(),
        };
        if self.policy.stores_entries() {
            let lru_key = if self.policy.max_entries().is_some() {
                Some(CanonicalFusionBlockContractCacheKey::from_parts(
                    rule_key.clone(),
                    dst_space,
                    lhs_space,
                    rhs_space,
                    axes_key.clone(),
                )?)
            } else {
                None
            };
            if let Some(plan) = self.fast_plans.get(&fast_key) {
                let plan = Arc::clone(plan);
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if let Some(key) = lru_key.as_ref() {
                    self.touch_plan(key);
                }
                self.last = Some(CanonicalFusionBlockContractLastEntry {
                    key: lru_key,
                    rule: rule_key,
                    dst: CanonicalFusionBlockLastSpaceKey::from_space(dst_space),
                    lhs: CanonicalFusionBlockLastSpaceKey::from_space(lhs_space),
                    rhs: CanonicalFusionBlockLastSpaceKey::from_space(rhs_space),
                    axes: raw_axes,
                    plan: Arc::clone(&plan),
                });
                return Ok(plan);
            }
        }

        let key = CanonicalFusionBlockContractCacheKey::from_parts(
            rule_key.clone(),
            dst_space,
            lhs_space,
            rhs_space,
            axes_key,
        )?;
        if !self.policy.stores_entries() {
            self.stats.misses += 1;
            return Ok(Arc::new(CanonicalFusionBlockContractPlan::compile(
                rule, dst_space, lhs_space, rhs_space, axes,
            )?));
        }
        if let Some(plan) = self.plans.get(&key) {
            self.stats.hits += 1;
            let plan = Arc::clone(plan);
            self.touch_plan(&key);
            self.fast_plans.insert(fast_key, Arc::clone(&plan));
            self.last = Some(CanonicalFusionBlockContractLastEntry {
                key: Some(key.clone()),
                rule: rule_key,
                dst: CanonicalFusionBlockLastSpaceKey::from_space(dst_space),
                lhs: CanonicalFusionBlockLastSpaceKey::from_space(lhs_space),
                rhs: CanonicalFusionBlockLastSpaceKey::from_space(rhs_space),
                axes: raw_axes,
                plan: Arc::clone(&plan),
            });
            return Ok(plan);
        } else {
            self.stats.misses += 1;
            let plan = CanonicalFusionBlockContractPlan::compile(
                rule, dst_space, lhs_space, rhs_space, axes,
            )?;
            let plan = Arc::new(plan);
            let last_key = key.clone();
            self.insert_plan(key, fast_key, Arc::clone(&plan));
            self.last = Some(CanonicalFusionBlockContractLastEntry {
                key: Some(last_key),
                rule: rule_key,
                dst: CanonicalFusionBlockLastSpaceKey::from_space(dst_space),
                lhs: CanonicalFusionBlockLastSpaceKey::from_space(lhs_space),
                rhs: CanonicalFusionBlockLastSpaceKey::from_space(rhs_space),
                axes: raw_axes,
                plan: Arc::clone(&plan),
            });
            return Ok(plan);
        }
    }
}

#[derive(Clone, Debug)]
struct CanonicalFusionBlockContractLastEntry<RuleKey> {
    key: Option<CanonicalFusionBlockContractCacheKey<RuleKey>>,
    rule: RuleKey,
    dst: CanonicalFusionBlockLastSpaceKey,
    lhs: CanonicalFusionBlockLastSpaceKey,
    rhs: CanonicalFusionBlockLastSpaceKey,
    axes: RawTensorContractAxisSpecKey,
    plan: Arc<CanonicalFusionBlockContractPlan>,
}

impl<RuleKey> CanonicalFusionBlockContractLastEntry<RuleKey>
where
    RuleKey: Eq,
{
    fn matches(
        &self,
        rule: &RuleKey,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> bool {
        &self.rule == rule
            && self.dst.matches(dst)
            && self.lhs.matches(lhs)
            && self.rhs.matches(rhs)
            && self.axes.matches(axes)
    }
}

#[derive(Clone, Debug)]
struct CanonicalFusionBlockLastSpaceKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure: Arc<BlockStructure>,
}

impl CanonicalFusionBlockLastSpaceKey {
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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct CanonicalFusionBlockContractFastKey<RuleKey> {
    rule: RuleKey,
    dst: CanonicalFusionBlockFastSpaceKey,
    lhs: CanonicalFusionBlockFastSpaceKey,
    rhs: CanonicalFusionBlockFastSpaceKey,
    axes: OwnedTensorContractAxisSpec,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct CanonicalFusionBlockFastSpaceKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure_ptr: usize,
}

impl CanonicalFusionBlockFastSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure_ptr: Arc::as_ptr(space.structure()) as usize,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct CanonicalFusionBlockContractCacheKey<RuleKey> {
    rule: RuleKey,
    dst_nout: usize,
    dst_homspace: FusionTreeHomSpace,
    dst_structure: BlockStructureCacheKey,
    lhs_nout: usize,
    lhs_homspace: FusionTreeHomSpace,
    lhs_structure: BlockStructureCacheKey,
    rhs_nout: usize,
    rhs_homspace: FusionTreeHomSpace,
    rhs_structure: BlockStructureCacheKey,
    axes: OwnedTensorContractAxisSpec,
}

impl<RuleKey> CanonicalFusionBlockContractCacheKey<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    fn from_parts(
        rule: RuleKey,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: OwnedTensorContractAxisSpec,
    ) -> Result<Self, OperationError> {
        Ok(Self {
            rule,
            dst_nout: dst.nout(),
            dst_homspace: dst.homspace().clone(),
            dst_structure: BlockStructureCacheKey::from_structure(dst.structure())?,
            lhs_nout: lhs.nout(),
            lhs_homspace: lhs.homspace().clone(),
            lhs_structure: BlockStructureCacheKey::from_structure(lhs.structure())?,
            rhs_nout: rhs.nout(),
            rhs_homspace: rhs.homspace().clone(),
            rhs_structure: BlockStructureCacheKey::from_structure(rhs.structure())?,
            axes,
        })
    }
}

impl<T> HostFusionBlockContractBuffers<T>
where
    T: Clone + Zero,
{
    fn prepare(
        &mut self,
        lhs_rows: usize,
        contracted: usize,
        rhs_cols: usize,
    ) -> Result<(), OperationError> {
        let lhs_len = lhs_rows
            .checked_mul(contracted)
            .ok_or(OperationError::ElementCountOverflow)?;
        let rhs_len = contracted
            .checked_mul(rhs_cols)
            .ok_or(OperationError::ElementCountOverflow)?;
        let dst_len = lhs_rows
            .checked_mul(rhs_cols)
            .ok_or(OperationError::ElementCountOverflow)?;
        self.lhs.resize_filled(lhs_len, T::zero());
        self.rhs.resize_filled(rhs_len, T::zero());
        self.dst.resize_filled(dst_len, T::zero());
        Ok(())
    }

    fn clear_inputs(&mut self, clear_lhs: bool, clear_rhs: bool) {
        if clear_lhs {
            self.lhs.fill(T::zero());
        }
        if clear_rhs {
            self.rhs.fill(T::zero());
        }
    }
}

#[derive(Clone, Debug)]
struct FusionBlockMatrixLayout {
    groups: Vec<FusionBlockMatrixGroup>,
}

impl FusionBlockMatrixLayout {
    fn compile<R>(
        rule: &R,
        space: &DynamicFusionMapSpace,
        rhs_contracting_axes: Option<&[usize]>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let mut builders = Vec::<FusionBlockMatrixGroupBuilder>::new();
        let mut group_indices = HashMap::<SectorId, usize>::new();
        for block_index in 0..space.structure().block_count() {
            let block = space.structure().block(block_index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "fusion",
                    index: block_index,
                });
            };
            let coupled = coupled_sector(rule, key.codomain_tree());
            if coupled != coupled_sector(rule, key.domain_tree()) {
                return Err(OperationError::FusionTreeGroupMismatch {
                    tensor: "fusion",
                    index: block_index,
                });
            }
            let group_index = if let Some(&group_index) = group_indices.get(&coupled) {
                group_index
            } else {
                let group_index = builders.len();
                group_indices.insert(coupled, group_index);
                builders.push(FusionBlockMatrixGroupBuilder::new(coupled));
                group_index
            };
            let row_dim = element_count(&block.shape()[..space.nout()])?;
            let col_dim = element_count(&block.shape()[space.nout()..])?;
            builders[group_index].add_tree_pair(
                key.codomain_tree().clone(),
                row_dim,
                key.domain_tree().clone(),
                col_dim,
                block_index,
            )?;
        }
        let mut groups = Vec::with_capacity(builders.len());
        for builder in builders {
            groups.push(builder.finish(rule, space, rhs_contracting_axes)?);
        }
        Ok(Self { groups })
    }

    fn group(&self, coupled: SectorId) -> Option<&FusionBlockMatrixGroup> {
        self.groups.iter().find(|group| group.coupled == coupled)
    }
}

#[derive(Clone, Debug)]
struct FusionBlockMatrixGroupBuilder {
    coupled: SectorId,
    row_offsets: HashMap<FusionTreeKey, TreeMatrixOffset>,
    col_offsets: HashMap<FusionTreeKey, TreeMatrixOffset>,
    tree_pairs: HashSet<(FusionTreeKey, FusionTreeKey)>,
    blocks: Vec<usize>,
    occupied_elements: usize,
    rows: usize,
    cols: usize,
}

impl FusionBlockMatrixGroupBuilder {
    fn new(coupled: SectorId) -> Self {
        Self {
            coupled,
            row_offsets: HashMap::new(),
            col_offsets: HashMap::new(),
            tree_pairs: HashSet::new(),
            blocks: Vec::new(),
            occupied_elements: 0,
            rows: 0,
            cols: 0,
        }
    }

    fn add_tree_pair(
        &mut self,
        row_tree: FusionTreeKey,
        row_dim: usize,
        col_tree: FusionTreeKey,
        col_dim: usize,
        block_index: usize,
    ) -> Result<(), OperationError> {
        if !self.tree_pairs.insert((row_tree.clone(), col_tree.clone())) {
            return Err(OperationError::StructureMismatch { tensor: "fusion" });
        }
        match self.row_offsets.get(&row_tree) {
            Some(offset) if offset.dim != row_dim => {
                return Err(OperationError::ShapeMismatch {
                    dst: vec![offset.dim],
                    src: vec![row_dim],
                });
            }
            Some(_) => {}
            None => {
                let offset = self.rows;
                self.rows = self
                    .rows
                    .checked_add(row_dim)
                    .ok_or(OperationError::ElementCountOverflow)?;
                self.row_offsets.insert(
                    row_tree,
                    TreeMatrixOffset {
                        offset,
                        dim: row_dim,
                    },
                );
            }
        }
        match self.col_offsets.get(&col_tree) {
            Some(offset) if offset.dim != col_dim => {
                return Err(OperationError::ShapeMismatch {
                    dst: vec![offset.dim],
                    src: vec![col_dim],
                });
            }
            Some(_) => {}
            None => {
                let offset = self.cols;
                self.cols = self
                    .cols
                    .checked_add(col_dim)
                    .ok_or(OperationError::ElementCountOverflow)?;
                self.col_offsets.insert(
                    col_tree,
                    TreeMatrixOffset {
                        offset,
                        dim: col_dim,
                    },
                );
            }
        }
        let block_elements = row_dim
            .checked_mul(col_dim)
            .ok_or(OperationError::ElementCountOverflow)?;
        self.occupied_elements = self
            .occupied_elements
            .checked_add(block_elements)
            .ok_or(OperationError::ElementCountOverflow)?;
        self.blocks.push(block_index);
        Ok(())
    }

    fn finish<R>(
        self,
        rule: &R,
        space: &DynamicFusionMapSpace,
        rhs_contracting_axes: Option<&[usize]>,
    ) -> Result<FusionBlockMatrixGroup, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let mut subblocks = Vec::with_capacity(self.blocks.len());
        let block_indices = self.blocks;
        for &block_index in &block_indices {
            let block = space.structure().block(block_index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "fusion",
                    index: block_index,
                });
            };
            let row = self
                .row_offsets
                .get(key.codomain_tree())
                .expect("row tree offset collected before finish");
            let col = self
                .col_offsets
                .get(key.domain_tree())
                .expect("column tree offset collected before finish");
            let mut matrix_strides = Vec::<isize>::with_capacity(block.shape().len());
            matrix_strides.extend(column_major_strides_isize(&block.shape()[..space.nout()])?);
            let domain_strides = column_major_strides_usize(&block.shape()[space.nout()..])?;
            for stride in domain_strides {
                let matrix_stride = stride
                    .checked_mul(self.rows)
                    .ok_or(OperationError::ElementCountOverflow)?;
                matrix_strides.push(isize::try_from(matrix_stride).map_err(|_| {
                    OperationError::StrideOverflow {
                        value: matrix_stride,
                    }
                })?);
            }
            let matrix_offset = col
                .offset
                .checked_mul(self.rows)
                .and_then(|offset| offset.checked_add(row.offset))
                .ok_or(OperationError::ElementCountOverflow)?;
            let matrix_offset = offset_to_isize(matrix_offset)?;
            let coefficient = if let Some(rhs_contracting_axes) = rhs_contracting_axes {
                rhs_contract_twist_factor(
                    rule,
                    space.homspace(),
                    rhs_contracting_axes,
                    key.codomain_tree(),
                )?
            } else {
                rule.scalar_one()
            };
            subblocks.push(FusionSubblockMatrixLayout {
                block: FusionStridedBlockLayout {
                    shape: block.shape().to_vec(),
                    strides: strides_to_isize(block.strides())?,
                    offset: offset_to_isize(block.offset())?,
                },
                matrix_offset,
                matrix_strides,
                coefficient,
            });
        }
        let matrix_elements = self
            .rows
            .checked_mul(self.cols)
            .ok_or(OperationError::ElementCountOverflow)?;
        Ok(FusionBlockMatrixGroup {
            coupled: self.coupled,
            rows: self.rows,
            cols: self.cols,
            needs_clear: self.occupied_elements != matrix_elements,
            block_indices,
            subblocks,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct TreeMatrixOffset {
    offset: usize,
    dim: usize,
}

#[derive(Clone, Debug)]
struct FusionBlockMatrixGroup {
    coupled: SectorId,
    rows: usize,
    cols: usize,
    // False only when the group's subblocks cover the packed matrix exactly.
    // Sparse fusion layouts keep this true so stale workspace cannot leak into GEMM.
    needs_clear: bool,
    block_indices: Vec<usize>,
    subblocks: Vec<FusionSubblockMatrixLayout>,
}

#[derive(Clone, Debug)]
struct FusionSubblockMatrixLayout {
    block: FusionStridedBlockLayout,
    matrix_offset: isize,
    matrix_strides: Vec<isize>,
    coefficient: f64,
}

#[derive(Clone, Debug)]
struct FusionStridedBlockLayout {
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
}

#[derive(Clone, Debug)]
struct FusionScaleBlockLayout {
    block: FusionStridedBlockLayout,
}

fn fusion_scale_block_layouts_excluding(
    structure: &BlockStructure,
    excluded_blocks: &HashSet<usize>,
) -> Result<Vec<FusionScaleBlockLayout>, OperationError> {
    let mut layouts = Vec::with_capacity(structure.block_count());
    for block_index in 0..structure.block_count() {
        if excluded_blocks.contains(&block_index) {
            continue;
        }
        let block = structure.block(block_index)?;
        layouts.push(FusionScaleBlockLayout {
            block: FusionStridedBlockLayout {
                shape: block.shape().to_vec(),
                strides: strides_to_isize(block.strides())?,
                offset: offset_to_isize(block.offset())?,
            },
        });
    }
    Ok(layouts)
}

fn coupled_sector<R>(rule: &R, tree: &FusionTreeKey) -> SectorId
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    tree.coupled().unwrap_or_else(|| rule.vacuum())
}

fn pack_group<T>(
    group: &FusionBlockMatrixGroup,
    data: &[T],
    packed: &mut [T],
) -> Result<(), OperationError>
where
    T: Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + ConjugateValue
        + RecouplingCoefficientAction<f64>
        + strided_kernel::MaybeSendSync,
{
    for layout in &group.subblocks {
        copy_scale_raw_strided_kernel_trusted(
            packed,
            data,
            &layout.block.shape,
            &layout.matrix_strides,
            &layout.block.strides,
            layout.matrix_offset,
            layout.block.offset,
            T::coefficient_as_data(layout.coefficient),
        )?;
    }
    Ok(())
}

fn scatter_group<T>(
    group: &FusionBlockMatrixGroup,
    data: &mut [T],
    packed: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    for layout in &group.subblocks {
        axpby_raw_strided_kernel_trusted(
            data,
            packed,
            &layout.block.shape,
            &layout.block.strides,
            &layout.matrix_strides,
            layout.block.offset,
            layout.matrix_offset,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

fn scale_all_blocks<T>(
    blocks: &[FusionScaleBlockLayout],
    data: &mut [T],
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + std::ops::Mul<T, Output = T> + PartialEq + Zero + One + strided_kernel::MaybeSendSync,
{
    if beta.is_one() {
        return Ok(());
    }
    for layout in blocks {
        scale_raw_strided_kernel_trusted(
            data,
            &layout.block.shape,
            &layout.block.strides,
            layout.block.offset,
            beta,
        )?;
    }
    Ok(())
}

fn matmul_group_plan<B, D>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    group: &CanonicalFusionBlockContractGroupPlan,
    lhs: &[D],
    rhs: &[D],
    dst: &mut [D],
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    backend.matmul_rank2_into_raw(
        workspace,
        dst,
        lhs,
        rhs,
        group.lhs.rows,
        group.lhs.cols,
        group.rhs.cols,
    )
}
