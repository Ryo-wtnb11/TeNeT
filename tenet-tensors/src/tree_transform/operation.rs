use std::any::TypeId;
use std::hash::Hash;

use tenet_core::{
    FermionParityFusionRule, FusionRule, ProductFusionRule, ProductSectorCodec, SU2FusionRule,
    U1FusionRule, Z2FusionRule,
};

use crate::OperationError;

pub use tenet_operations::TreeTransformOperationKey;

/// Rule-aware validation for [`TreeTransformOperationKey`]; lives in the
/// symmetric layer because it consumes the fusion rule's braiding style.
pub trait ValidateBraidingSupport {
    fn validate_braiding_support<R>(&self, rule: &R) -> Result<(), OperationError>
    where
        R: FusionRule;
}

impl ValidateBraidingSupport for TreeTransformOperationKey {
    fn validate_braiding_support<R>(&self, rule: &R) -> Result<(), OperationError>
    where
        R: FusionRule,
    {
        if self.requires_symmetric_braiding() && !rule.braiding_style().is_symmetric() {
            return Err(OperationError::UnsupportedBraidingStyle {
                operation: self.clone(),
                style: rule.braiding_style(),
            });
        }
        Ok(())
    }
}

/// Semantic cache identity for fusion-tree transformation replay.
///
/// Equal keys must imply identical fusion, duality, braiding, and recoupling
/// coefficients for every sector/tree combination the rule can produce.
/// Cached tree-transform and fusion-contraction replay plans may be reused
/// solely from this key plus the operand structures, so custom rules must
/// include every parameter that can change those coefficients.
pub trait TreeTransformRuleCacheKey {
    type Key: Clone + Eq + Hash;

    fn tree_transform_rule_cache_key(&self) -> Self::Key;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformBuiltinRuleCacheKey {
    Z2,
    FermionParity,
    U1,
    SU2,
}

impl TreeTransformRuleCacheKey for Z2FusionRule {
    type Key = TreeTransformBuiltinRuleCacheKey;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        TreeTransformBuiltinRuleCacheKey::Z2
    }
}

impl TreeTransformRuleCacheKey for FermionParityFusionRule {
    type Key = TreeTransformBuiltinRuleCacheKey;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        TreeTransformBuiltinRuleCacheKey::FermionParity
    }
}

impl TreeTransformRuleCacheKey for U1FusionRule {
    type Key = TreeTransformBuiltinRuleCacheKey;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        TreeTransformBuiltinRuleCacheKey::U1
    }
}

impl TreeTransformRuleCacheKey for SU2FusionRule {
    type Key = TreeTransformBuiltinRuleCacheKey;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        TreeTransformBuiltinRuleCacheKey::SU2
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformProductRuleCacheKey<LeftKey, RightKey> {
    left: LeftKey,
    right: RightKey,
    codec: TypeId,
}

impl<LeftKey, RightKey> TreeTransformProductRuleCacheKey<LeftKey, RightKey> {
    pub fn new<Codec>(left: LeftKey, right: RightKey) -> Self
    where
        Codec: 'static,
    {
        Self {
            left,
            right,
            codec: TypeId::of::<Codec>(),
        }
    }

    #[inline]
    pub fn left(&self) -> &LeftKey {
        &self.left
    }

    #[inline]
    pub fn right(&self) -> &RightKey {
        &self.right
    }
}

impl<LeftRule, RightRule, Codec> TreeTransformRuleCacheKey
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: TreeTransformRuleCacheKey,
    RightRule: TreeTransformRuleCacheKey,
    Codec: ProductSectorCodec + 'static,
{
    type Key = TreeTransformProductRuleCacheKey<LeftRule::Key, RightRule::Key>;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        TreeTransformProductRuleCacheKey::new::<Codec>(
            self.left_rule().tree_transform_rule_cache_key(),
            self.right_rule().tree_transform_rule_cache_key(),
        )
    }
}
