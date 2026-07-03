use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

use tenet_core::{BlockStructure, FusionTreeHomSpace, MultiplicityFreeRigidSymbols};

use crate::axis::{AxisPermutation, OwnedTensorContractAxisSpec, TensorContractAxisSpec};
use crate::cache::{
    enforce_lru_limit, rebuild_lru_order_from_keys, touch_lru_key, BlockStructureCacheKey,
    OperationCachePolicy,
};
use crate::{OperationError, TreeTransformRuleCacheKey};

use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion_block::is_canonical_fusion_block_contract;
use super::structure::TensorContractAxisPlan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TensorContractFusionRouteDecision {
    CanonicalFusionBlocks,
    DynamicTreeCanonical,
}

#[derive(Clone, Debug)]
pub(super) struct FusionRouteCache<RuleKey> {
    last: Option<FusionRouteLastEntry<RuleKey>>,
    routes: HashMap<FusionRouteCacheKey<RuleKey>, TensorContractFusionRouteDecision>,
    lru_order: VecDeque<FusionRouteCacheKey<RuleKey>>,
    policy: OperationCachePolicy,
}

impl<RuleKey> Default for FusionRouteCache<RuleKey> {
    fn default() -> Self {
        Self {
            last: None,
            routes: HashMap::new(),
            lru_order: VecDeque::new(),
            policy: OperationCachePolicy::default(),
        }
    }
}

#[derive(Clone, Debug)]
struct FusionRouteLastEntry<RuleKey> {
    key: FusionRouteCacheKey<RuleKey>,
    rule: RuleKey,
    dst_nout: usize,
    dst_homspace: FusionTreeHomSpace,
    dst_structure: Arc<BlockStructure>,
    lhs_nout: usize,
    lhs_homspace: FusionTreeHomSpace,
    lhs_structure: Arc<BlockStructure>,
    rhs_nout: usize,
    rhs_homspace: FusionTreeHomSpace,
    rhs_structure: Arc<BlockStructure>,
    axes: RouteRawTensorContractAxisSpecKey,
    route: TensorContractFusionRouteDecision,
}

impl<RuleKey> FusionRouteLastEntry<RuleKey>
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
            && self.dst_nout == dst.nout()
            && Arc::ptr_eq(&self.dst_structure, dst.structure())
            && self.dst_homspace == *dst.homspace()
            && self.lhs_nout == lhs.nout()
            && Arc::ptr_eq(&self.lhs_structure, lhs.structure())
            && self.lhs_homspace == *lhs.homspace()
            && self.rhs_nout == rhs.nout()
            && Arc::ptr_eq(&self.rhs_structure, rhs.structure())
            && self.rhs_homspace == *rhs.homspace()
            && self.axes.matches(axes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FusionRouteCacheKey<RuleKey> {
    rule: RuleKey,
    dst: FusionRouteSpaceCacheKey,
    lhs: FusionRouteSpaceCacheKey,
    rhs: FusionRouteSpaceCacheKey,
    axes: OwnedTensorContractAxisSpec,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FusionRouteSpaceCacheKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure: BlockStructureCacheKey,
}

impl FusionRouteSpaceCacheKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Result<Self, OperationError> {
        Ok(Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure: BlockStructureCacheKey::from_structure(space.structure())?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RouteRawTensorContractAxisSpecKey {
    lhs_contracting_axes: Vec<usize>,
    rhs_contracting_axes: Vec<usize>,
    output_permutation: RouteRawAxisPermutationKey,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
}

impl RouteRawTensorContractAxisSpecKey {
    fn from_axes(axes: TensorContractAxisSpec<'_>) -> Self {
        Self {
            lhs_contracting_axes: axes.lhs_contracting_axes().to_vec(),
            rhs_contracting_axes: axes.rhs_contracting_axes().to_vec(),
            output_permutation: RouteRawAxisPermutationKey::from_axes(axes.output_permutation()),
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
enum RouteRawAxisPermutationKey {
    Identity,
    Axes(Vec<usize>),
}

impl RouteRawAxisPermutationKey {
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

impl<RuleKey> FusionRouteCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.routes.len()
    }

    pub(super) fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.last = None;
        if !policy.stores_entries() {
            self.routes.clear();
            self.lru_order.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            rebuild_lru_order_from_keys(&self.routes, &mut self.lru_order);
            self.enforce_lru_limit(max_entries);
        }
    }

    pub(super) fn get_or_compile_nonconjugate<R>(
        &mut self,
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<TensorContractFusionRouteDecision, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        if !self.policy.stores_entries() {
            return Self::compile_nonconjugate_route(rule, dst, lhs, rhs, axes);
        }

        let rule_key = rule.tree_transform_rule_cache_key();
        let refresh_lru = self.policy.max_entries().is_some();
        let last_hit = self.last.as_ref().and_then(|last| {
            if last.matches(&rule_key, dst, lhs, rhs, axes) {
                Some((refresh_lru.then(|| last.key.clone()), last.route))
            } else {
                None
            }
        });
        if let Some((key, route)) = last_hit {
            if let Some(key) = key.as_ref() {
                self.touch_route(key);
            }
            return Ok(route);
        }

        let raw_axes = RouteRawTensorContractAxisSpecKey::from_axes(axes);
        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), dst.rank(), axes)?;
        let axes_key = OwnedTensorContractAxisSpec::new_with_conjugation(
            axis_plan.lhs_contracting_axes,
            axis_plan.rhs_contracting_axes,
            axis_plan.output_axes,
            axis_plan.lhs_conjugate,
            axis_plan.rhs_conjugate,
        );
        let key = FusionRouteCacheKey {
            rule: rule_key.clone(),
            dst: FusionRouteSpaceCacheKey::from_space(dst)?,
            lhs: FusionRouteSpaceCacheKey::from_space(lhs)?,
            rhs: FusionRouteSpaceCacheKey::from_space(rhs)?,
            axes: axes_key,
        };
        if let Some(&route) = self.routes.get(&key) {
            self.touch_route(&key);
            self.last = Some(FusionRouteLastEntry {
                key: key.clone(),
                rule: rule_key,
                dst_nout: dst.nout(),
                dst_homspace: dst.homspace().clone(),
                dst_structure: Arc::clone(dst.structure()),
                lhs_nout: lhs.nout(),
                lhs_homspace: lhs.homspace().clone(),
                lhs_structure: Arc::clone(lhs.structure()),
                rhs_nout: rhs.nout(),
                rhs_homspace: rhs.homspace().clone(),
                rhs_structure: Arc::clone(rhs.structure()),
                axes: raw_axes,
                route,
            });
            return Ok(route);
        }

        let route = Self::compile_nonconjugate_route(rule, dst, lhs, rhs, axes)?;
        let last_key = key.clone();
        self.insert_route(key, route);
        self.last = Some(FusionRouteLastEntry {
            key: last_key,
            rule: rule_key,
            dst_nout: dst.nout(),
            dst_homspace: dst.homspace().clone(),
            dst_structure: Arc::clone(dst.structure()),
            lhs_nout: lhs.nout(),
            lhs_homspace: lhs.homspace().clone(),
            lhs_structure: Arc::clone(lhs.structure()),
            rhs_nout: rhs.nout(),
            rhs_homspace: rhs.homspace().clone(),
            rhs_structure: Arc::clone(rhs.structure()),
            axes: raw_axes,
            route,
        });
        Ok(route)
    }

    fn touch_route(&mut self, key: &FusionRouteCacheKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.routes.contains_key(key) {
            touch_lru_key(&mut self.lru_order, key);
        }
    }

    fn insert_route(
        &mut self,
        key: FusionRouteCacheKey<RuleKey>,
        route: TensorContractFusionRouteDecision,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.routes.insert(key.clone(), route);
        if self.policy.max_entries().is_some() {
            self.touch_route(&key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_lru_limit(max_entries);
        }
    }

    fn enforce_lru_limit(&mut self, max_entries: usize) {
        let before = self.routes.len();
        enforce_lru_limit(&mut self.routes, &mut self.lru_order, max_entries);
        if self.routes.len() != before {
            self.last = None;
        }
    }

    fn compile_nonconjugate_route<R>(
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<TensorContractFusionRouteDecision, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        if is_canonical_fusion_block_contract(rule, dst, lhs, rhs, axes)? {
            Ok(TensorContractFusionRouteDecision::CanonicalFusionBlocks)
        } else {
            Ok(TensorContractFusionRouteDecision::DynamicTreeCanonical)
        }
    }
}
