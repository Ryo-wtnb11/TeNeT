//! Unified contraction resolution: one cache maps
//! `(rule, dst/lhs/rhs spaces, axes)` to the resolved execution artifact —
//! route and plan together.
//!
//! TensorKit keeps exactly one cache entry per (spaces, permutation)
//! (`treepermuter` and friends); its route decision is compile-time dispatch
//! plus a cheap uncached cost heuristic and is never memoized. The previous
//! three TeNeT caches (route / core fusion-block plan / explicit plan)
//! all shared this key with different payloads — this module restores the
//! one-entry granularity. Prepared contraction handles wrap the same
//! [`Resolution`] value, so the facade and the plan-once API share one
//! resolution machinery.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use tenet_core::{BlockStructure, FusionTreeHomSpace, MultiplicityFreeRigidSymbols};

use super::structure::TensorContractStructure;
use crate::cache::{
    operation_global_registry, typed_global_map, BlockStructureCacheKey, OperationCachePolicy,
};
use crate::OperationError;
use tenet_operations::axis::{OutputAxisOrder, TensorContractSpec, TensorContractSpecOwned};
use tenet_operations::fusion_replay::FusionBlockContractPlan;

use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::{external_axis_is_dual, FusionContractPlan};
use super::fusion_block::{
    compile_fusion_block_contract_plan_validated, is_core_form_fusion_block_contract,
    validate_fusion_contract_rule,
};
use super::structure::TensorContractAxisPlan;

type GlobalContractionResolutionMap<RuleKey> = RwLock<FxHashMap<FullKey<RuleKey>, Resolution>>;

fn global_contraction_resolutions<RuleKey>() -> Arc<GlobalContractionResolutionMap<RuleKey>>
where
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    typed_global_map(operation_global_registry())
}

/// Resolved execution artifact for one contraction key: the route decision
/// and its compiled plan are one value, never cached separately.
#[derive(Clone, Debug)]
pub(crate) enum Resolution {
    /// Coupled-sector direct GEMM (TensorKit `mul!` shape).
    Core(Arc<FusionBlockContractPlan>),
    /// Source/output tree transforms around a core core
    /// (TensorKit `@tensor` shape).
    DynamicTree(Arc<FusionContractPlan>),
    /// Dense one-shot structure for conjugated operands (TeNeT optimization
    /// over the faithful transform-then-contract path).
    Structure(Arc<TensorContractStructure<f64>>),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ContractionResolutionStats {
    pub(crate) hits: usize,
    pub(crate) fast_hits: usize,
    pub(crate) misses: usize,
}

/// Structural full key: reachable identically from typed and dynamic callers.
/// `core_only` separates the route-resolution namespace from the dynamic
/// route's internal scratch plans: a scratch contraction can carry exactly
/// the spaces/axes of the facade contraction that spawned it (identity
/// operand transforms, e.g. a fermionic twist-only contraction), and its
/// core plan must never alias the facade's dynamic resolution.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FullKey<RuleKey> {
    rule: RuleKey,
    dst: FullSpaceKey,
    lhs: FullSpaceKey,
    rhs: FullSpaceKey,
    axes: TensorContractSpecOwned,
    core_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FullSpaceKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure: BlockStructureCacheKey,
}

impl FullSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Result<Self, OperationError> {
        Ok(Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure: BlockStructureCacheKey::from_structure(space.structure())?,
        })
    }
}

/// Pointer-identity fast key: serves many distinct keys without structural
/// hashing, so loops cycling through several fixed contractions (an iPEPS
/// unit cell, energy environments) stay O(1) even when they alternate.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FastKey<RuleKey> {
    rule: RuleKey,
    dst: FastSpaceKey,
    lhs: FastSpaceKey,
    rhs: FastSpaceKey,
    axes: TensorContractSpecOwned,
    core_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FastSpaceKey {
    nout: usize,
    homspace_ptr: usize,
    structure_ptr: usize,
}

impl FastSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace_ptr: Arc::as_ptr(space.homspace_arc()) as usize,
            structure_ptr: Arc::as_ptr(space.structure()) as usize,
        }
    }
}

#[derive(Clone, Debug)]
struct LastEntry<RuleKey> {
    rule: RuleKey,
    dst: LastSpace,
    lhs: LastSpace,
    rhs: LastSpace,
    axes: RawAxes,
    core_only: bool,
    full_key: Option<FullKey<RuleKey>>,
    resolution: Resolution,
}

#[derive(Clone, Debug)]
struct LastSpace {
    nout: usize,
    homspace: Arc<FusionTreeHomSpace>,
    structure: Arc<BlockStructure>,
}

impl LastSpace {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace: Arc::clone(space.homspace_arc()),
            structure: Arc::clone(space.structure()),
        }
    }

    fn matches(&self, space: &DynamicFusionMapSpace) -> bool {
        self.nout == space.nout()
            && Arc::ptr_eq(&self.structure, space.structure())
            && (Arc::ptr_eq(&self.homspace, space.homspace_arc())
                || *self.homspace == *space.homspace())
    }
}

/// Raw axes copy for last-entry comparison without recompiling the axis plan.
#[derive(Clone, Debug, Eq, PartialEq)]
struct RawAxes {
    lhs_contracting: Vec<usize>,
    rhs_contracting: Vec<usize>,
    output_identity: bool,
    output_axes: Vec<usize>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
}

impl RawAxes {
    fn from_axes(axes: TensorContractSpec<'_>) -> Self {
        let (output_identity, output_axes) = match axes.output_permutation() {
            OutputAxisOrder::Identity => (true, Vec::new()),
            OutputAxisOrder::Axes(list) => (false, list.to_vec()),
        };
        Self {
            lhs_contracting: axes.lhs_contracting_axes().to_vec(),
            rhs_contracting: axes.rhs_contracting_axes().to_vec(),
            output_identity,
            output_axes,
            lhs_conjugate: axes.lhs_conjugate(),
            rhs_conjugate: axes.rhs_conjugate(),
        }
    }

    fn matches(&self, axes: TensorContractSpec<'_>) -> bool {
        let output_matches = match axes.output_permutation() {
            OutputAxisOrder::Identity => self.output_identity,
            OutputAxisOrder::Axes(list) => !self.output_identity && self.output_axes == list,
        };
        output_matches
            && self.lhs_contracting == axes.lhs_contracting_axes()
            && self.rhs_contracting == axes.rhs_contracting_axes()
            && self.lhs_conjugate == axes.lhs_conjugate()
            && self.rhs_conjugate == axes.rhs_conjugate()
    }
}

/// Recently used entries kept for pointer-compared lookups. A dynamic
/// contraction interleaves the facade key with several distinct internal
/// scratch keys per replay, so a single slot would thrash on every call.
const LAST_RING_CAPACITY: usize = 16;

#[derive(Clone, Debug)]
pub(crate) struct ContractionResolutionCache<RuleKey> {
    last: Vec<LastEntry<RuleKey>>,
    fast: FxHashMap<FastKey<RuleKey>, Resolution>,
    resolved: FxHashMap<FullKey<RuleKey>, Resolution>,
    lru_order: VecDeque<FullKey<RuleKey>>,
    policy: OperationCachePolicy,
    stats: ContractionResolutionStats,
}

impl<RuleKey> Default for ContractionResolutionCache<RuleKey> {
    fn default() -> Self {
        Self {
            last: Vec::new(),
            fast: FxHashMap::default(),
            resolved: FxHashMap::default(),
            lru_order: VecDeque::new(),
            policy: OperationCachePolicy::default(),
            stats: ContractionResolutionStats::default(),
        }
    }
}

impl<RuleKey> ContractionResolutionCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.resolved.len()
    }

    #[inline]
    pub(crate) fn stats(&self) -> ContractionResolutionStats {
        self.stats
    }

    pub(crate) fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.last.clear();
        self.fast.clear();
        if !policy.stores_entries() {
            self.resolved.clear();
            self.lru_order.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            crate::cache::rebuild_lru_order_from_keys(&self.resolved, &mut self.lru_order);
            self.enforce_lru_limit(max_entries);
        }
    }

    fn enforce_lru_limit(&mut self, max_entries: usize) {
        let mut evicted = false;
        while self.resolved.len() > max_entries {
            let Some(oldest) = self.lru_order.pop_front() else {
                break;
            };
            evicted |= self.resolved.remove(&oldest).is_some();
        }
        if evicted {
            self.fast.clear();
            self.last.clear();
        }
    }

    fn touch(&mut self, key: &FullKey<RuleKey>) {
        if self.policy.max_entries().is_some() {
            crate::cache::touch_lru_key(&mut self.lru_order, key);
        }
    }

    /// Resolve route and plan in one lookup. On a cold key the core
    /// plan compiled during the route test is stored, never discarded;
    /// non-core keys resolve through `compile_dynamic` (which has the
    /// caller's typed context) or, for conjugated axes, `compile_structure`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_resolve<R>(
        &mut self,
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
        compile_structure: impl FnOnce() -> Result<
            Option<Arc<TensorContractStructure<f64>>>,
            OperationError,
        >,
        compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
    ) -> Result<Resolution, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleKey>,
        RuleKey: 'static + Send + Sync,
    {
        self.get_or_resolve_with(rule, dst, lhs, rhs, axes, false, || {
            resolve(
                rule,
                dst,
                lhs,
                rhs,
                axes,
                compile_structure,
                compile_dynamic,
            )
        })
    }

    /// Core-only entry for the dynamic route's internal scratch
    /// contractions: operands are already materialized in coupled scratch
    /// (twists applied), so the route test and twist gate do not apply.
    pub(crate) fn get_or_compile_core_plan<R>(
        &mut self,
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
    ) -> Result<Arc<FusionBlockContractPlan>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleKey>,
        RuleKey: 'static + Send + Sync,
    {
        let resolution = self.get_or_resolve_with(rule, dst, lhs, rhs, axes, true, || {
            compile_fusion_block_contract_plan_validated(rule, dst, lhs, rhs, axes)
                .map(|plan| Resolution::Core(Arc::new(plan)))
        })?;
        match resolution {
            Resolution::Core(plan) => Ok(plan),
            _ => Err(OperationError::UnsupportedTensorContractScope {
                message: "internal scratch contraction resolved to a non-core plan",
            }),
        }
    }

    /// Shared lookup skeleton: fast paths, then `resolve_cold` on a miss.
    fn get_or_resolve_with<R>(
        &mut self,
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
        core_only: bool,
        resolve_cold: impl FnOnce() -> Result<Resolution, OperationError>,
    ) -> Result<Resolution, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleKey>,
        RuleKey: 'static + Send + Sync,
    {
        validate_fusion_contract_rule(rule, dst, lhs, rhs)?;
        let rule_key = rule.tree_transform_rule_cache_key();
        if self.policy.stores_entries() {
            let position = self.last.iter().position(|last| {
                last.rule == rule_key
                    && last.core_only == core_only
                    && last.dst.matches(dst)
                    && last.lhs.matches(lhs)
                    && last.rhs.matches(rhs)
                    && last.axes.matches(axes)
            });
            if let Some(index) = position {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if index != 0 {
                    let entry = self.last.remove(index);
                    self.last.insert(0, entry);
                }
                let resolution = self.last[0].resolution.clone();
                // Clone the deep structural key only when an LRU limit
                // actually needs the touch; the clone dwarfs the whole
                // hit path otherwise.
                let touch_key = if self.policy.max_entries().is_some() {
                    self.last[0].full_key.clone()
                } else {
                    None
                };
                if let Some(key) = touch_key {
                    self.touch(&key);
                }
                return Ok(resolution);
            }
        }

        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), dst.rank(), axes)?;
        let axes_key = TensorContractSpecOwned::new_with_conjugation(
            axis_plan.lhs_contracting_axes,
            axis_plan.rhs_contracting_axes,
            axis_plan.output_axes,
            axis_plan.lhs_conjugate,
            axis_plan.rhs_conjugate,
        );

        if self.policy.stores_entries() {
            let fast_key = FastKey {
                rule: rule_key.clone(),
                dst: FastSpaceKey::from_space(dst),
                lhs: FastSpaceKey::from_space(lhs),
                rhs: FastSpaceKey::from_space(rhs),
                axes: axes_key.clone(),
                core_only,
            };
            if let Some(resolution) = self.fast.get(&fast_key) {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                let resolution = resolution.clone();
                self.remember_last(&rule_key, dst, lhs, rhs, axes, core_only, None, &resolution);
                return Ok(resolution);
            }

            let full_key = FullKey {
                rule: rule_key.clone(),
                dst: FullSpaceKey::from_space(dst)?,
                lhs: FullSpaceKey::from_space(lhs)?,
                rhs: FullSpaceKey::from_space(rhs)?,
                axes: axes_key.clone(),
                core_only,
            };
            if let Some(resolution) = self.resolved.get(&full_key) {
                self.stats.hits += 1;
                let resolution = resolution.clone();
                self.touch(&full_key);
                self.fast.insert(fast_key, resolution.clone());
                self.remember_last(
                    &rule_key,
                    dst,
                    lhs,
                    rhs,
                    axes,
                    core_only,
                    Some(full_key),
                    &resolution,
                );
                return Ok(resolution);
            }

            self.stats.misses += 1;
            let global = global_contraction_resolutions::<RuleKey>();
            if let Some(resolution) = global
                .read()
                .expect("global contraction resolution cache poisoned")
                .get(&full_key)
                .cloned()
            {
                self.resolved.insert(full_key.clone(), resolution.clone());
                if let Some(max_entries) = self.policy.max_entries() {
                    self.lru_order.push_back(full_key.clone());
                    self.enforce_lru_limit(max_entries);
                }
                self.fast.insert(fast_key, resolution.clone());
                self.remember_last(
                    &rule_key,
                    dst,
                    lhs,
                    rhs,
                    axes,
                    core_only,
                    Some(full_key),
                    &resolution,
                );
                return Ok(resolution);
            }
            let resolution = resolve_cold()?;
            global
                .write()
                .expect("global contraction resolution cache poisoned")
                .entry(full_key.clone())
                .or_insert_with(|| resolution.clone());
            self.resolved.insert(full_key.clone(), resolution.clone());
            if let Some(max_entries) = self.policy.max_entries() {
                self.lru_order.push_back(full_key.clone());
                self.enforce_lru_limit(max_entries);
            }
            self.fast.insert(fast_key, resolution.clone());
            self.remember_last(
                &rule_key,
                dst,
                lhs,
                rhs,
                axes,
                core_only,
                Some(full_key),
                &resolution,
            );
            return Ok(resolution);
        }

        self.stats.misses += 1;
        resolve_cold()
    }

    #[allow(clippy::too_many_arguments)]
    fn remember_last(
        &mut self,
        rule_key: &RuleKey,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
        core_only: bool,
        full_key: Option<FullKey<RuleKey>>,
        resolution: &Resolution,
    ) {
        self.last.insert(
            0,
            LastEntry {
                rule: rule_key.clone(),
                dst: LastSpace::from_space(dst),
                lhs: LastSpace::from_space(lhs),
                rhs: LastSpace::from_space(rhs),
                axes: RawAxes::from_axes(axes),
                core_only,
                full_key,
                resolution: resolution.clone(),
            },
        );
        self.last.truncate(LAST_RING_CAPACITY);
    }
}

/// Uncached resolution: the route test and the plan compile are one step.
fn resolve<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce()
        -> Result<Option<Arc<TensorContractStructure<f64>>>, OperationError>,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
) -> Result<Resolution, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if !axes.lhs_conjugate() && !axes.rhs_conjugate() {
        if is_core_form_fusion_block_contract(rule, dst, lhs, rhs, axes)?
            && !rhs_contract_requires_twist(rule, rhs, axes)?
        {
            let plan = compile_fusion_block_contract_plan_validated(rule, dst, lhs, rhs, axes)?;
            if plan.is_fully_direct() {
                return Ok(Resolution::Core(Arc::new(plan)));
            }
        }
        return Ok(Resolution::DynamicTree(compile_dynamic()?));
    }
    if let Some(structure) = compile_structure()? {
        return Ok(Resolution::Structure(structure));
    }
    Ok(Resolution::DynamicTree(compile_dynamic()?))
}

/// True when the fermionic supertrace twist can be nontrivial: such
/// contractions take the dynamic route, where the twist is applied during
/// rhs materialization; the core direct-GEMM route stays
/// coefficient-free (TensorKit mul! parity).
pub(crate) fn rhs_contract_requires_twist<R>(
    rule: &R,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if rule.braiding_style() != tenet_core::BraidingStyleKind::Fermionic {
        return Ok(false);
    }
    for &axis in axes.rhs_contracting_axes() {
        if external_axis_is_dual(rhs.homspace(), axis)? {
            return Ok(true);
        }
    }
    Ok(false)
}
