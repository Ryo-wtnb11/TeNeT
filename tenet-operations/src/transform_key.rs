//! Tree-transform operation keys: pure permutation/braid descriptions with
//! no symmetry knowledge (the rule-aware cache keys stay in the symmetric
//! execution crate).

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
