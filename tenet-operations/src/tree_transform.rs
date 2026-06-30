use std::any::TypeId;
use std::hash::Hash;

use tenet_core::{
    FermionParityFusionRule, FusionRule, ProductFusionRule, ProductSectorCodec, SU2FusionRule,
    U1FusionRule, Z2FusionRule,
};

use crate::OperationError;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformOperationKey {
    Transpose {
        codomain_permutation: Vec<usize>,
        domain_permutation: Vec<usize>,
    },
    Permute {
        codomain_permutation: Vec<usize>,
        domain_permutation: Vec<usize>,
    },
    Braid {
        codomain_permutation: Vec<usize>,
        domain_permutation: Vec<usize>,
        codomain_levels: Vec<usize>,
        domain_levels: Vec<usize>,
    },
}

impl TreeTransformOperationKey {
    /// Build a planar transpose operation.
    ///
    /// The two permutations follow TensorKit's `Index2Tuple` convention:
    /// both `codomain_permutation` and `domain_permutation` contain source
    /// tensor axis numbers in the full `0..numind` range. They are not local
    /// permutations within the old codomain/domain parts. For example, for a
    /// `(NOUT, NIN) = (2, 1)` tensor, keeping the domain leg in the domain uses
    /// `domain_permutation = [2]`, not `[0]`.
    pub fn transpose<Codomain, Domain>(
        codomain_permutation: Codomain,
        domain_permutation: Domain,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
    {
        Self::Transpose {
            codomain_permutation: codomain_permutation.into_iter().collect(),
            domain_permutation: domain_permutation.into_iter().collect(),
        }
    }

    /// Build a symmetric-braiding permutation operation.
    ///
    /// Axis numbering follows TensorKit's `Index2Tuple` convention; see
    /// [`Self::transpose`].
    pub fn permute<Codomain, Domain>(
        codomain_permutation: Codomain,
        domain_permutation: Domain,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
    {
        Self::Permute {
            codomain_permutation: codomain_permutation.into_iter().collect(),
            domain_permutation: domain_permutation.into_iter().collect(),
        }
    }

    /// Build an explicit braid operation with source-axis permutations and levels.
    ///
    /// Axis numbering follows TensorKit's `Index2Tuple` convention; see
    /// [`Self::transpose`]. `codomain_levels` and `domain_levels` are the
    /// levels of the source axes selected by each output tuple.
    pub fn braid<Codomain, Domain, CodomainLevels, DomainLevels>(
        codomain_permutation: Codomain,
        domain_permutation: Domain,
        codomain_levels: CodomainLevels,
        domain_levels: DomainLevels,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
        CodomainLevels: IntoIterator<Item = usize>,
        DomainLevels: IntoIterator<Item = usize>,
    {
        Self::Braid {
            codomain_permutation: codomain_permutation.into_iter().collect(),
            domain_permutation: domain_permutation.into_iter().collect(),
            codomain_levels: codomain_levels.into_iter().collect(),
            domain_levels: domain_levels.into_iter().collect(),
        }
    }

    pub fn requires_symmetric_braiding(&self) -> bool {
        matches!(self, Self::Permute { .. })
    }

    pub fn validate_braiding_support<R>(&self, rule: &R) -> Result<(), OperationError>
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformPlanScope {
    AllCodomain,
    TreePair,
}
