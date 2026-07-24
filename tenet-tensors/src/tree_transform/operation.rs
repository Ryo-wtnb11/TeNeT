use std::hash::Hash;

use tenet_core::{FusionRule, MultiplicityFreeRigidSymbols, RuleIdentity};

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

/// Marks a multiplicity-free rule as eligible for keyed tree-transform replay.
///
/// Its key is the rule's canonical identity. Why not permit an independent
/// cache key: `FusionRule::rule_identity` already owns the complete semantic
/// identity contract, and a second value could omit a coefficient determinant.
pub trait TreeTransformRuleCacheKey {
    type Key: 'static + Clone + Eq + Hash + Send + Sync;

    fn tree_transform_rule_cache_key(&self) -> Self::Key;
}

impl<R> TreeTransformRuleCacheKey for R
where
    R: MultiplicityFreeRigidSymbols,
{
    type Key = RuleIdentity;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        self.rule_identity()
    }
}
