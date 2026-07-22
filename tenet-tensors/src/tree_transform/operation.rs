use std::any::TypeId;
use std::hash::Hash;

use tenet_core::{
    FermionParityFusionRule, FusionRule, ProductFusionRule, ProductSectorCodec, SU2FusionRule,
    Su3FusionRule, U1FusionRule, Z2FusionRule,
};

use crate::OperationError;

pub use tenet_operations::{TreeTransformOperation, TreeTransformOperationKind};

/// Rule-aware validation for [`TreeTransformOperation`]; lives in the
/// symmetric layer because it consumes the fusion rule's braiding style.
pub trait ValidateBraidingSupport {
    fn validate_braiding_support<R>(&self, rule: &R) -> Result<(), OperationError>
    where
        R: FusionRule;
}

impl ValidateBraidingSupport for TreeTransformOperation {
    fn validate_braiding_support<R>(&self, rule: &R) -> Result<(), OperationError>
    where
        R: FusionRule,
    {
        if self.requires_symmetric_braiding() && !rule.braiding_style().is_symmetric() {
            return Err(OperationError::UnsupportedBraidingStyle {
                operation: Box::new(self.clone()),
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
    type Key: 'static + Clone + Eq + Hash + Send + Sync;

    fn tree_transform_rule_cache_key(&self) -> Self::Key;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformBuiltinRuleCacheKey {
    Z2,
    FermionParity,
    U1,
    SU2Exact { authority_version: u8 },
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
        TreeTransformBuiltinRuleCacheKey::SU2Exact {
            authority_version: tenet_core::SU2_EXACT_AUTHORITY_VERSION,
        }
    }
}

/// Cache identity for the Stage B3b SU(3) table provider. Keyed by the table's
/// provenance hash (payload FNV-1a-64), NOT a unit marker: a regenerated /
/// swapped table produces different recoupling coefficients, so its compiled
/// plans must never be reused. A distinct `Key` type also means the SU(3)
/// cache — monomorphized per `RuleKey` — shares no map with the mult-free
/// `TreeTransformBuiltinRuleCacheKey` instance and cannot collide with it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformSu3RuleCacheKey {
    provenance: u64,
}

impl TreeTransformRuleCacheKey for Su3FusionRule {
    type Key = TreeTransformSu3RuleCacheKey;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        TreeTransformSu3RuleCacheKey {
            provenance: self.provenance(),
        }
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
