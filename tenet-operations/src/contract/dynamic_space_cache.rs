use std::collections::HashMap;
use std::hash::Hash;

use tenet_core::{FusionTensorMapSpace, MultiplicityFreeRigidSymbols};

use crate::axis::OwnedTensorContractAxisSpec;
use crate::{OperationError, TreeTransformOperationKey, TreeTransformRuleCacheKey};

use super::dynamic_space::{DynamicFusionMapSpace, DynamicFusionMapSpaceCacheKey};
use super::fusion::TensorContractFusionExplicitPlan;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionTransformedSpaceCacheKey<RuleKey> {
    rule: RuleKey,
    source: DynamicFusionMapSpaceCacheKey,
    operation: TreeTransformOperationKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionCanonicalDstSpaceCacheKey<RuleKey> {
    rule: RuleKey,
    lhs: DynamicFusionMapSpaceCacheKey,
    rhs: DynamicFusionMapSpaceCacheKey,
    axes: OwnedTensorContractAxisSpec,
    canonical_dst_nout: usize,
    canonical_dst_nin: usize,
    output_transform: TreeTransformOperationKey,
    output_dst: Option<DynamicFusionMapSpaceCacheKey>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorContractFusionSpaceCacheStats {
    transformed_hits: usize,
    transformed_misses: usize,
    canonical_dst_hits: usize,
    canonical_dst_misses: usize,
}

impl TensorContractFusionSpaceCacheStats {
    #[inline]
    pub fn transformed_hits(self) -> usize {
        self.transformed_hits
    }

    #[inline]
    pub fn transformed_misses(self) -> usize {
        self.transformed_misses
    }

    #[inline]
    pub fn canonical_dst_hits(self) -> usize {
        self.canonical_dst_hits
    }

    #[inline]
    pub fn canonical_dst_misses(self) -> usize {
        self.canonical_dst_misses
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionSpaceCache<RuleKey> {
    transformed: HashMap<DynamicFusionTransformedSpaceCacheKey<RuleKey>, DynamicFusionMapSpace>,
    canonical_dst: HashMap<DynamicFusionCanonicalDstSpaceCacheKey<RuleKey>, DynamicFusionMapSpace>,
    stats: TensorContractFusionSpaceCacheStats,
}

impl<RuleKey> Default for DynamicFusionSpaceCache<RuleKey> {
    fn default() -> Self {
        Self {
            transformed: HashMap::new(),
            canonical_dst: HashMap::new(),
            stats: TensorContractFusionSpaceCacheStats::default(),
        }
    }
}

impl<RuleKey> DynamicFusionSpaceCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.transformed.len() + self.canonical_dst.len()
    }

    #[inline]
    pub(crate) fn transformed_len(&self) -> usize {
        self.transformed.len()
    }

    #[inline]
    pub(crate) fn canonical_dst_len(&self) -> usize {
        self.canonical_dst.len()
    }

    #[inline]
    pub(crate) fn stats(&self) -> TensorContractFusionSpaceCacheStats {
        self.stats
    }

    pub(crate) fn transformed_from_typed<R, const NOUT: usize, const NIN: usize>(
        &mut self,
        rule: &R,
        source: &FusionTensorMapSpace<NOUT, NIN>,
        operation: &TreeTransformOperationKey,
    ) -> Result<DynamicFusionMapSpace, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let key = DynamicFusionTransformedSpaceCacheKey {
            rule: rule.tree_transform_rule_cache_key(),
            source: DynamicFusionMapSpaceCacheKey::from_typed_space(source)?,
            operation: operation.clone(),
        };
        if let Some(space) = self.transformed.get(&key) {
            self.stats.transformed_hits += 1;
            return Ok(space.clone());
        }
        self.stats.transformed_misses += 1;
        let space = DynamicFusionMapSpace::transformed_from_typed(rule, source, operation)?;
        self.transformed.insert(key, space.clone());
        Ok(space)
    }

    pub(crate) fn canonical_dst<R>(
        &mut self,
        rule: &R,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        plan: &TensorContractFusionExplicitPlan,
        output_dst: Option<&DynamicFusionMapSpace>,
    ) -> Result<DynamicFusionMapSpace, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let key = DynamicFusionCanonicalDstSpaceCacheKey {
            rule: rule.tree_transform_rule_cache_key(),
            lhs: lhs.cache_key()?,
            rhs: rhs.cache_key()?,
            axes: plan.canonical_axes().clone(),
            canonical_dst_nout: plan.canonical_dst_nout(),
            canonical_dst_nin: plan.canonical_dst_nin(),
            output_transform: plan.output_transform().clone(),
            output_dst: output_dst
                .map(DynamicFusionMapSpace::cache_key)
                .transpose()?,
        };
        if let Some(space) = self.canonical_dst.get(&key) {
            self.stats.canonical_dst_hits += 1;
            return Ok(space.clone());
        }
        self.stats.canonical_dst_misses += 1;
        let space = DynamicFusionMapSpace::canonical_dst(rule, lhs, rhs, plan, output_dst)?;
        self.canonical_dst.insert(key, space.clone());
        Ok(space)
    }
}
