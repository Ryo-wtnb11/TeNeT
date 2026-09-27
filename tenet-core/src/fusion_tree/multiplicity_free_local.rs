/// Braid state that is mutated step by step without a cached hash.
///
/// `FusionTreeKey` caches its hash, so in-place `Arc::make_mut` swaps on a key
/// would leave that cache stale between steps; keeping the working state in a
/// hash-free struct makes the stale state unrepresentable and costs one hash
/// at `freeze` instead of one per Artin step.
struct UnhashedFusionTree {
    uncoupled: Arc<[SectorId]>,
    coupled: SectorId,
    is_dual: Arc<[bool]>,
    innerlines: Arc<[SectorId]>,
    vertices: Arc<[MultiplicityIndex]>,
}

impl From<FusionTreeKey> for UnhashedFusionTree {
    fn from(key: FusionTreeKey) -> Self {
        let FusionTreeKey {
            uncoupled,
            coupled,
            is_dual,
            innerlines,
            vertices,
            hash: _,
        } = key;
        Self {
            uncoupled,
            coupled,
            is_dual,
            innerlines,
            vertices,
        }
    }
}

impl UnhashedFusionTree {
    fn freeze(self) -> FusionTreeKey {
        FusionTreeKey::from_frozen(
            self.uncoupled,
            self.coupled,
            self.is_dual,
            self.innerlines,
            self.vertices,
        )
    }
}

impl MultiplicityFreeTreeLocalData for UnhashedFusionTree {
    #[inline]
    fn coupled(&self) -> SectorId {
        self.coupled
    }

    #[inline]
    fn innerlines(&self) -> &[SectorId] {
        &self.innerlines
    }
}

impl MultiplicityFreeTreeData for UnhashedFusionTree {
    #[inline]
    fn uncoupled(&self) -> &[SectorId] {
        &self.uncoupled
    }
}

fn apply_unique_artin_braid_at_with_inverse<R>(
    rule: &R,
    tree: &mut UnhashedFusionTree,
    index: usize,
    inverse: bool,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: rule.fusion_style(),
        });
    }

    let rank = tree.uncoupled().len();
    if index + 1 >= rank {
        return Err(CoreError::InvalidBraidIndex { index, rank });
    }

    let left = tree.uncoupled()[index];
    let right = tree.uncoupled()[index + 1];

    if left == rule.vacuum() || right == rule.vacuum() {
        if index > 0 {
            let inner_source = if left == rule.vacuum() {
                inner_extended_sector(tree, index + 1)?
            } else {
                inner_extended_sector(tree, index - 1)?
            };
            *Arc::make_mut(&mut tree.innerlines)
                .get_mut(index - 1)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires an innerline",
                })? = inner_source;
            if tree.vertices.len() <= index {
                return Err(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires adjacent vertices",
                });
            }
            Arc::make_mut(&mut tree.vertices).swap(index - 1, index);
        }

        Arc::make_mut(&mut tree.uncoupled).swap(index, index + 1);
        Arc::make_mut(&mut tree.is_dual).swap(index, index + 1);
        return Ok(R::Scalar::one());
    }

    if !rule.braiding_style().has_braiding() {
        return Err(CoreError::UnsupportedSectorBraid {
            left,
            right,
            style: rule.braiding_style(),
        });
    }

    if index == 0 {
        let coupled = if rank > 2 {
            tree.innerlines()
                .first()
                .copied()
                .ok_or(CoreError::MalformedFusionTree {
                    message: "first braid of a rank > 2 tree requires the first innerline",
                })?
        } else {
            tree.coupled()
        };

        let coefficient = if inverse {
            (rule.r_symbol_scalar(right, left, coupled)).conj()
        } else {
            rule.r_symbol_scalar(left, right, coupled)
        };
        Arc::make_mut(&mut tree.uncoupled).swap(index, index + 1);
        Arc::make_mut(&mut tree.is_dual).swap(index, index + 1);
        return Ok(coefficient);
    }

    let a = inner_extended_sector(tree, index - 1)?;
    let b = left;
    let c = inner_extended_sector(tree, index)?;
    let d = right;
    let e = inner_extended_sector(tree, index + 1)?;
    let c_prime = only_fusion_channel(rule, a, d)?;
    *Arc::make_mut(&mut tree.innerlines)
        .get_mut(index - 1)
        .ok_or(CoreError::MalformedFusionTree {
            message: "non-first braid requires an innerline to update",
        })? = c_prime;
    let f_symbol = rule.f_symbol_scalar(d, a, b, e, c_prime, c);
    let coefficient = if inverse {
        let left = rule.r_symbol_scalar(d, c, e);
        let right = rule.r_symbol_scalar(d, a, c_prime);
        (left * f_symbol).conj() * right
    } else {
        let left = rule.r_symbol_scalar(c, d, e);
        let right = rule.r_symbol_scalar(a, d, c_prime);
        left * (f_symbol * right).conj()
    };
    Arc::make_mut(&mut tree.uncoupled).swap(index, index + 1);
    Arc::make_mut(&mut tree.is_dual).swap(index, index + 1);
    Ok(coefficient)
}

trait MultiplicityFreeTreeLocalData {
    fn coupled(&self) -> SectorId;
    fn innerlines(&self) -> &[SectorId];
}

impl MultiplicityFreeTreeLocalData for FusionTreeKey {
    #[inline]
    fn coupled(&self) -> SectorId {
        self.coupled()
    }

    #[inline]
    fn innerlines(&self) -> &[SectorId] {
        self.innerlines()
    }

}

trait MultiplicityFreeTreeData: MultiplicityFreeTreeLocalData {
    fn uncoupled(&self) -> &[SectorId];
}

impl MultiplicityFreeTreeData for FusionTreeKey {
    #[inline]
    fn uncoupled(&self) -> &[SectorId] {
        self.uncoupled()
    }
}

#[derive(Clone, PartialEq, Eq)]
struct MultiplicityFreeTreeFrame {
    uncoupled: Arc<[SectorId]>,
    is_dual: Arc<[bool]>,
    vertices: Arc<[MultiplicityIndex]>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct MultiplicityFreeTreeLocal {
    coupled: SectorId,
    innerlines: SectorVec,
}

type MultiplicityFreeTreeLocalTerms<S> = Vec<(MultiplicityFreeTreeLocal, S)>;
type MultiplicityFreeFoldInverseCache<S> =
    FxHashMap<(SectorId, MultiplicityFreeTreeLocal), MultiplicityFreeTreeLocalTerms<S>>;

mod multiplicity_free_projection {
    use super::*;

    #[derive(Clone, Copy)]
    pub(super) struct Trees<'a> {
        keys: &'a [FusionTreeKey],
    }

    #[derive(Clone, Copy)]
    pub(super) struct Pairs<'a> {
        source: PairSource<'a>,
    }

    #[derive(Clone, Copy)]
    pub(super) struct Tree<'a> {
        key: &'a FusionTreeKey,
    }

    #[derive(Clone, Copy)]
    pub(super) struct Pair<'a> {
        codomain: &'a FusionTreeKey,
        domain: &'a FusionTreeKey,
    }

    #[derive(Clone, Copy)]
    enum PairSource<'a> {
        Slice(&'a [FusionTreePairKey]),
        Structure {
            structure: &'a BlockStructure,
            indices: &'a [usize],
            orientation: FusionTreePairOrientation,
        },
    }

    pub(super) struct TreeBatch<'rule, R> {
        rule: &'rule R,
        keys: SmallVec<[FusionTreeKey; 8]>,
    }

    pub(super) struct PairBatch<'rule, R> {
        rule: &'rule R,
        keys: SmallVec<[FusionTreePairKey; 8]>,
    }

    impl<'rule, R> TreeBatch<'rule, R>
    where
        R: FusionRule,
    {
        pub(super) fn from_locally_validated<'structure, I>(
            proof: &LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>,
            indices: I,
        ) -> Result<Self, CoreError>
        where
            I: IntoIterator<Item = usize>,
        {
            let keys = indices
                .into_iter()
                .map(|index| {
                    proof
                        .required_fusion_tree_pair_key(index)
                        .map(|key| key.codomain_tree().clone())
                })
                .collect::<Result<SmallVec<_>, _>>()?;
            Ok(Self {
                rule: proof.rule,
                keys,
            })
        }

        pub(super) fn parts(&self) -> (&R, &[FusionTreeKey]) {
            (self.rule, &self.keys)
        }
    }

    impl<'rule, R> PairBatch<'rule, R>
    where
        R: FusionRule,
    {
        pub(super) fn from_locally_validated<'structure, I>(
            proof: &LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>,
            indices: I,
        ) -> Result<Self, CoreError>
        where
            I: IntoIterator<Item = usize>,
        {
            let keys = indices
                .into_iter()
                .map(|index| proof.required_fusion_tree_pair_key(index).cloned())
                .collect::<Result<SmallVec<_>, _>>()?;
            Ok(Self {
                rule: proof.rule,
                keys,
            })
        }

        pub(super) fn parts(&self) -> (&R, &[FusionTreePairKey]) {
            (self.rule, &self.keys)
        }
    }

    impl<'a> Trees<'a> {
        pub(super) fn checked<R>(
            rule: &R,
            trees: &'a [FusionTreeKey],
        ) -> Result<Self, CoreError>
        where
            R: FusionRule,
        {
            validate_multiplicity_free_execution_style(rule)?;
            for tree in trees {
                Self::check_vertices(tree)?;
            }
            Ok(Self { keys: trees })
        }

        pub(super) fn from_validated<R>(
            rule: &R,
            trees: &'a [FusionTreeKey],
        ) -> Result<Self, CoreError>
        where
            R: FusionRule,
        {
            // Why not rescan vertices: categorical validation already proved
            // every multiplicity-free N-symbol bound, hence label ONE. The
            // `_proven` callers receive keys from a LocallyValidated block.
            validate_multiplicity_free_execution_style(rule)?;
            Ok(Self { keys: trees })
        }

        pub(super) fn tree_at(&self, index: usize) -> Option<Tree<'a>> {
            self.keys.get(index).map(|key| Tree { key })
        }

        fn check_vertices(tree: &FusionTreeKey) -> Result<(), CoreError> {
            check_vertices(tree)
        }
    }

    impl<'a> Pairs<'a> {
        pub(super) fn checked<R>(
            rule: &R,
            pairs: &'a [FusionTreePairKey],
        ) -> Result<Self, CoreError>
        where
            R: FusionRule,
        {
            validate_multiplicity_free_execution_style(rule)?;
            for pair in pairs {
                check_vertices(pair.codomain_tree())?;
                check_vertices(pair.domain_tree())?;
            }
            Ok(Self {
                source: PairSource::Slice(pairs),
            })
        }

        pub(super) fn from_validated<R>(
            rule: &R,
            pairs: &'a [FusionTreePairKey],
        ) -> Result<Self, CoreError>
        where
            R: FusionRule,
        {
            // Pair validation proves the same vertex invariant for both trees
            // in the single categorical validation pass.
            validate_multiplicity_free_execution_style(rule)?;
            Ok(Self {
                source: PairSource::Slice(pairs),
            })
        }

        pub(super) fn checked_structure<R>(
            rule: &R,
            structure: &'a BlockStructure,
            indices: &'a [usize],
            orientation: FusionTreePairOrientation,
        ) -> Result<Self, CoreError>
        where
            R: FusionRule,
        {
            validate_multiplicity_free_execution_style(rule)?;
            let projection = Self {
                source: PairSource::Structure {
                    structure,
                    indices,
                    orientation,
                },
            };
            for (index, &block_index) in indices.iter().enumerate() {
                let block = structure.block(block_index)?;
                if !matches!(block.key(), BlockKey::FusionTree(_)) {
                    return Err(CoreError::ExpectedFusionTreePairKey {
                        actual: block.key().kind(),
                    });
                }
                let pair = projection
                    .pair_at(index)
                    .expect("validated block index and key kind produce a pair");
                let codomain = validate_fusion_tree_for_rule(rule, pair.codomain().key())?;
                let domain = validate_fusion_tree_for_rule(rule, pair.domain().key())?;
                if codomain.key.coupled() != domain.key.coupled() {
                    return Err(CoreError::MalformedFusionTree {
                        message: "fusion tree pair requires matching coupled sectors",
                    });
                }
                check_vertices(pair.codomain().key())?;
                check_vertices(pair.domain().key())?;
            }
            Ok(projection)
        }

        pub(super) fn len(&self) -> usize {
            match self.source {
                PairSource::Slice(keys) => keys.len(),
                PairSource::Structure { indices, .. } => indices.len(),
            }
        }

        pub(super) fn pair_at(&self, index: usize) -> Option<Pair<'a>> {
            match self.source {
                PairSource::Slice(keys) => keys.get(index).map(|key| Pair {
                    codomain: key.codomain_tree(),
                    domain: key.domain_tree(),
                }),
                PairSource::Structure {
                    structure,
                    indices,
                    orientation,
                } => {
                    let block = structure.block(*indices.get(index)?).ok()?;
                    let BlockKey::FusionTree(key) = block.key() else {
                        return None;
                    };
                    let (codomain, domain) = match orientation {
                        FusionTreePairOrientation::Direct => {
                            (key.codomain_tree(), key.domain_tree())
                        }
                        FusionTreePairOrientation::Adjoint => {
                            (key.domain_tree(), key.codomain_tree())
                        }
                    };
                    Some(Pair { codomain, domain })
                }
            }
        }
    }

    fn check_vertices(tree: &FusionTreeKey) -> Result<(), CoreError> {
        if tree
            .vertices()
            .iter()
            .any(|&vertex| vertex != MultiplicityIndex::ONE)
        {
            return Err(CoreError::MalformedFusionTree {
                message: "multiplicity-free projection requires vertex label one",
            });
        }
        Ok(())
    }

    impl<'a> Tree<'a> {
        pub(super) fn key(self) -> &'a FusionTreeKey {
            self.key
        }
    }

    impl<'a> Pair<'a> {
        pub(super) fn materialize(self) -> FusionTreePairKey {
            FusionTreePairKey::pair(self.codomain.clone(), self.domain.clone())
        }

        pub(super) fn codomain(self) -> Tree<'a> {
            Tree { key: self.codomain }
        }

        pub(super) fn domain(self) -> Tree<'a> {
            Tree { key: self.domain }
        }
    }
}

use multiplicity_free_projection::{
    Pair as ValidatedMultiplicityFreeTreePair, PairBatch as ValidatedMultiplicityFreePairBatch,
    Pairs as MultiplicityFreePairProjection, Tree as ValidatedMultiplicityFreeTree,
    TreeBatch as ValidatedMultiplicityFreeTreeBatch, Trees as MultiplicityFreeTreeProjection,
};

impl MultiplicityFreeTreeLocalData for MultiplicityFreeTreeLocal {
    #[inline]
    fn coupled(&self) -> SectorId {
        self.coupled
    }

    #[inline]
    fn innerlines(&self) -> &[SectorId] {
        &self.innerlines
    }

}

type MultiplicityFreeArtinTerms<S> = SmallVec<[(MultiplicityFreeTreeLocal, S); 2]>;

impl MultiplicityFreeTreeLocal {
    fn from_proven(tree: ValidatedMultiplicityFreeTree<'_>) -> Self {
        let tree = tree.key();
        Self {
            coupled: tree.coupled(),
            innerlines: tree.innerlines().iter().copied().collect(),
        }
    }
}

impl MultiplicityFreeTreeFrame {
    fn from_frozen_externals(
        uncoupled: Arc<[SectorId]>,
        is_dual: Arc<[bool]>,
    ) -> Self {
        let vertices = std::iter::repeat_n(
            MultiplicityIndex::ONE,
            uncoupled.len().saturating_sub(1),
        )
        .collect::<Vec<_>>()
        .into();
        Self {
            uncoupled,
            is_dual,
            vertices,
        }
    }

    fn from_tree(tree: &FusionTreeKey) -> Self {
        Self {
            uncoupled: Arc::clone(&tree.uncoupled),
            is_dual: Arc::clone(&tree.is_dual),
            vertices: Arc::clone(&tree.vertices),
        }
    }

    fn split(
        tree: ValidatedMultiplicityFreeTree<'_>,
    ) -> (Self, MultiplicityFreeTreeLocal) {
        let key = tree.key();
        (
            Self::from_tree(key),
            MultiplicityFreeTreeLocal::from_proven(tree),
        )
    }

    fn matches_tree(&self, tree: &FusionTreeKey) -> bool {
        // Why not split and compare frames: every source shares these slices,
        // while collecting rank > 8 frames would allocate once per source.
        self.uncoupled.as_ref() == tree.uncoupled()
            && self.is_dual.as_ref() == tree.is_dual()
    }

    fn materialize(&self, local: MultiplicityFreeTreeLocal) -> FusionTreeKey {
        FusionTreeKey::from_frozen(
            Arc::clone(&self.uncoupled),
            local.coupled,
            Arc::clone(&self.is_dual),
            Arc::from(local.innerlines.as_slice()),
            Arc::clone(&self.vertices),
        )
    }
}

fn project_multiplicity_free_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<(MultiplicityFreeTreeFrame, MultiplicityFreeTreeLocal), CoreError>
where
    R: FusionRule,
{
    let projection =
        MultiplicityFreeTreeProjection::checked(rule, std::slice::from_ref(tree))?;
    Ok(MultiplicityFreeTreeFrame::split(
        projection
            .tree_at(0)
            .expect("single-tree projection contains index zero"),
    ))
}

#[derive(Clone, PartialEq, Eq)]
struct MultiplicityFreeTreePairFrame {
    codomain: MultiplicityFreeTreeFrame,
    domain: MultiplicityFreeTreeFrame,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct MultiplicityFreeTreePairLocal {
    codomain: MultiplicityFreeTreeLocal,
    domain: MultiplicityFreeTreeLocal,
}

impl MultiplicityFreeTreePairLocal {
    fn from_proven(tree_pair: ValidatedMultiplicityFreeTreePair<'_>) -> Self {
        Self {
            codomain: MultiplicityFreeTreeLocal::from_proven(tree_pair.codomain()),
            domain: MultiplicityFreeTreeLocal::from_proven(tree_pair.domain()),
        }
    }
}

impl MultiplicityFreeTreePairFrame {
    fn split(
        tree_pair: ValidatedMultiplicityFreeTreePair<'_>,
    ) -> (Self, MultiplicityFreeTreePairLocal) {
        let codomain = MultiplicityFreeTreeFrame::from_tree(tree_pair.codomain().key());
        let domain = MultiplicityFreeTreeFrame::from_tree(tree_pair.domain().key());
        (
            Self { codomain, domain },
            MultiplicityFreeTreePairLocal::from_proven(tree_pair),
        )
    }

    fn matches_tree_pair_ref(&self, tree_pair: ValidatedMultiplicityFreeTreePair<'_>) -> bool {
        self.codomain.matches_tree(tree_pair.codomain().key())
            && self.domain.matches_tree(tree_pair.domain().key())
    }

    fn materialize(&self, local: MultiplicityFreeTreePairLocal) -> FusionTreePairKey {
        FusionTreePairKey::pair(
            self.codomain.materialize(local.codomain),
            self.domain.materialize(local.domain),
        )
    }
}

fn project_multiplicity_free_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<
    (
        MultiplicityFreeTreePairFrame,
        MultiplicityFreeTreePairLocal,
    ),
    CoreError,
>
where
    R: FusionRule,
{
    let projection =
        MultiplicityFreePairProjection::checked(rule, std::slice::from_ref(tree_pair))?;
    Ok(MultiplicityFreeTreePairFrame::split(
        projection
            .pair_at(0)
            .expect("single-pair projection contains index zero"),
    ))
}

struct PreparedMultiplicityFreeArtin {
    output_frame: MultiplicityFreeTreeFrame,
    rank: usize,
    first: SectorId,
    index: usize,
    inverse: bool,
    left: SectorId,
    right: SectorId,
}

fn prepare_multiplicity_free_artin<R>(
    rule: &R,
    frame: &MultiplicityFreeTreeFrame,
    index: usize,
    inverse: bool,
) -> Result<PreparedMultiplicityFreeArtin, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
{
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: rule.fusion_style(),
        });
    }

    let rank = frame.uncoupled.len();
    if index + 1 >= rank {
        return Err(CoreError::InvalidBraidIndex { index, rank });
    }

    let left = frame.uncoupled[index];
    let right = frame.uncoupled[index + 1];
    let mut uncoupled = frame.uncoupled.to_vec();
    uncoupled.swap(index, index + 1);
    let mut is_dual = frame.is_dual.to_vec();
    is_dual.swap(index, index + 1);
    let frame = MultiplicityFreeTreeFrame {
        uncoupled: uncoupled.into(),
        is_dual: is_dual.into(),
        vertices: Arc::clone(&frame.vertices),
    };

    if left != rule.vacuum()
        && right != rule.vacuum()
        && !rule.braiding_style().has_braiding()
    {
        return Err(CoreError::UnsupportedSectorBraid {
            left,
            right,
            style: rule.braiding_style(),
        });
    }
    let first = frame.uncoupled[0];
    Ok(PreparedMultiplicityFreeArtin {
        output_frame: frame,
        rank,
        first,
        index,
        inverse,
        left,
        right,
    })
}

impl PreparedMultiplicityFreeArtin {
    // External frame data stays out of this kernel: the block runner must enumerate
    // locals from this prepared frame rather than rebuilding full tree keys.
    fn apply<R, T>(
        &self,
        rule: &R,
        tree: &T,
    ) -> Result<MultiplicityFreeArtinTerms<R::Scalar>, CoreError>
    where
        R: MultiplicityFreeFusionSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
        T: MultiplicityFreeTreeLocalData,
    {
        let index = self.index;
        let left = self.left;
        let right = self.right;
        if left == rule.vacuum() || right == rule.vacuum() {
            let mut innerlines = tree.innerlines().to_vec();
            if index > 0 {
                let inner_source = if left == rule.vacuum() {
                    self.inner_extended(tree, index + 1)?
                } else {
                    self.inner_extended(tree, index - 1)?
                };
                *innerlines
                    .get_mut(index - 1)
                    .ok_or(CoreError::MalformedFusionTree {
                        message: "unit braid past the first adjacent pair requires an innerline",
                    })? = inner_source;
            }
            let mut terms = SmallVec::new();
            terms.push((
                MultiplicityFreeTreeLocal {
                    coupled: tree.coupled(),
                    innerlines: innerlines.into_iter().collect(),
                },
                R::Scalar::one(),
            ));
            return Ok(terms);
        }

        if index == 0 {
            let coupled = if self.rank > 2 {
                tree.innerlines()
                    .first()
                    .copied()
                    .ok_or(CoreError::MalformedFusionTree {
                        message: "first braid of a rank > 2 tree requires the first innerline",
                    })?
            } else {
                tree.coupled()
            };
            let coefficient = if self.inverse {
                (rule.r_symbol_scalar(right, left, coupled)).conj()
            } else {
                rule.r_symbol_scalar(left, right, coupled)
            };
            let mut terms = SmallVec::new();
            terms.push((
                MultiplicityFreeTreeLocal {
                    coupled: tree.coupled(),
                    innerlines: tree.innerlines().iter().copied().collect(),
                },
                coefficient,
            ));
            return Ok(terms);
        }

        let a = self.inner_extended(tree, index - 1)?;
        let b = left;
        let c = self.inner_extended(tree, index)?;
        let d = right;
        let e = self.inner_extended(tree, index + 1)?;
        let mut terms: MultiplicityFreeArtinTerms<R::Scalar> = SmallVec::new();
        for c_prime in rule.fusion_channels(a, d) {
            if rule.nsymbol(c_prime, b, e) == 0 {
                continue;
            }
            let mut innerlines: SectorVec = tree.innerlines().iter().copied().collect();
            *innerlines
                .get_mut(index - 1)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "non-first braid requires an innerline to update",
                })? = c_prime;
            let braided = MultiplicityFreeTreeLocal {
                coupled: tree.coupled(),
                innerlines,
            };
            let f_symbol = rule.f_symbol_scalar(d, a, b, e, c_prime, c);
            let coefficient = if self.inverse {
                let left = rule.r_symbol_scalar(d, c, e);
                let right = rule.r_symbol_scalar(d, a, c_prime);
                (left * f_symbol).conj() * right
            } else {
                let left = rule.r_symbol_scalar(c, d, e);
                let right = rule.r_symbol_scalar(a, d, c_prime);
                left * (f_symbol * right).conj()
            };
            terms.push((braided, coefficient));
        }
        Ok(terms)
    }

    fn inner_extended<T>(&self, tree: &T, index: usize) -> Result<SectorId, CoreError>
    where
        T: MultiplicityFreeTreeLocalData + ?Sized,
    {
        if index == 0 {
            return Ok(self.first);
        }
        if index + 1 == self.rank {
            return Ok(tree.coupled());
        }
        tree.innerlines()
            .get(index - 1)
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "inner-extended tree is missing an innerline",
            })
    }
}

#[expect(
    clippy::type_complexity,
    reason = "the SmallVec inline capacity is part of this local braid allocation contract"
)]
fn multiplicity_free_artin_braid_at_with_inverse<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Result<SmallVec<[(FusionTreeKey, R::Scalar); 2]>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    // Why not keep a second full-key formula for block transforms: the compact
    // block basis must reuse exactly the same F/R ordering and error gates.
    let (frame, local) = project_multiplicity_free_tree(rule, tree)?;
    let prepared = prepare_multiplicity_free_artin(rule, &frame, index, inverse)?;
    let terms = prepared.apply(rule, &local)?;
    Ok(terms
        .into_iter()
        .map(|(local, coefficient)| {
            (
                prepared.output_frame.materialize(local),
                coefficient,
            )
        })
        .collect())
}
