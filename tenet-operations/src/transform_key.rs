//! Tree-transform operation keys: pure permutation/braid descriptions with
//! no symmetry knowledge (the rule-aware cache keys stay in the symmetric
//! execution crate).

use smallvec::SmallVec;

/// Axis permutation / level list, inline up to rank 8 (the common tensor
/// rank). This type is a hot HashMap-key component in the recoupling plan
/// memo — keeping it stack-allocated makes the per-lookup key clone
/// allocation-free (matching TensorKit's stack-allocated `NTuple`), which was
/// ~35% of all cold-path allocations when it was a `Vec`.
pub type AxisVec = SmallVec<[usize; 8]>;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformOperation {
    Transpose {
        codomain_permutation: AxisVec,
        domain_permutation: AxisVec,
    },
    Permute {
        codomain_permutation: AxisVec,
        domain_permutation: AxisVec,
    },
    Braid {
        codomain_permutation: AxisVec,
        domain_permutation: AxisVec,
        codomain_levels: AxisVec,
        domain_levels: AxisVec,
    },
}

impl TreeTransformOperation {
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
    /// [`Self::transpose`]. `codomain_levels` and `domain_levels` are split by
    /// the source tensor's codomain/domain tree axes, independent of the output
    /// tuple positions selected by `codomain_permutation` and
    /// `domain_permutation`. This mirrors TensorKit's `add_braid!`, which
    /// splits the full source `levels` tuple with `codomainind(tsrc)` and
    /// `domainind(tsrc)`.
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
}

#[cfg(test)]
mod tests {
    use super::TreeTransformOperation;

    #[test]
    fn identity_axis_map_requires_the_current_split_and_valid_braid_levels() {
        // What: identity classification accepts exact axes only after the
        // operation itself has a complete source-level description.
        assert!(TreeTransformOperation::permute([0, 1], [2]).is_identity_for(2, 1));
        assert!(TreeTransformOperation::braid([0, 1], [2], [7, 3], [5]).is_identity_for(2, 1));
        assert!(!TreeTransformOperation::braid([0, 1], [2], [7], [5]).is_identity_for(2, 1));
        assert!(!TreeTransformOperation::permute([0], [1, 2]).is_identity_for(2, 1));
        assert!(!TreeTransformOperation::permute([1, 0], [2]).is_identity_for(2, 1));
        assert!(!TreeTransformOperation::transpose([0, 1], [2]).is_identity_for(2, 1));
    }

    #[test]
    fn rank_zero_identity_axis_map_is_well_formed() {
        // What: empty Permute/Braid descriptions are valid rank-zero
        // identities, while an empty Transpose remains outside this slice.
        assert!(TreeTransformOperation::permute([], []).is_identity_for(0, 0));
        assert!(TreeTransformOperation::braid([], [], [], []).is_identity_for(0, 0));
        assert!(!TreeTransformOperation::transpose([], []).is_identity_for(0, 0));
    }
}
