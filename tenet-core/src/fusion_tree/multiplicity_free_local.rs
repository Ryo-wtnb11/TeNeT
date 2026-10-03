use super::*;

/// Braid state that is mutated step by step without a cached hash.
///
/// `FusionTreeKey` caches its hash, so in-place `Arc::make_mut` swaps on a key
/// would leave that cache stale between steps; keeping the working state in a
/// hash-free struct makes the stale state unrepresentable and costs one hash
/// at `freeze` instead of one per Artin step.
pub(crate) struct UnhashedFusionTree {
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
    pub(crate) fn vertex_at(&self, position: usize) -> Option<MultiplicityIndex> {
        self.vertices.get(position).copied()
    }

    pub(crate) fn freeze(self) -> FusionTreeKey {
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

#[cfg(test)]
impl MultiplicityFreeTreeData for UnhashedFusionTree {
    #[inline]
    fn uncoupled(&self) -> &[SectorId] {
        &self.uncoupled
    }
}

pub(crate) fn apply_unique_artin_braid_at_with_inverse<R>(
    rule: &R,
    tree: &mut UnhashedFusionTree,
    index: usize,
    inverse: bool,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let kernel = UniqueK(rule);
    let site = ArtinSite::new(&kernel, &tree.uncoupled, index, inverse)?;
    let mut out = InPlaceArtinEdit {
        innerline_count: tree.innerlines.len(),
        vertex_count: tree.vertices.len(),
        index,
        edit: None,
    };
    artin_surgery(&kernel, &site, tree, &mut out)?;
    let (innerline, vertices, coefficient) = out.edit.ok_or(CoreError::MalformedFusionTree {
        message: "a unique-fusion braid emits exactly one tree",
    })?;
    if let Some((position, sector)) = innerline {
        Arc::make_mut(&mut tree.innerlines)[position] = sector;
    }
    if vertices == ArtinVertices::SwapUnit {
        Arc::make_mut(&mut tree.vertices).swap(index - 1, index);
    }
    Arc::make_mut(&mut tree.uncoupled).swap(index, index + 1);
    Arc::make_mut(&mut tree.is_dual).swap(index, index + 1);
    Ok(coefficient)
}

/// The one output of an in-place unique-fusion swap, validated against the
/// tree's shape while the surgery still reads it and applied afterwards.
struct InPlaceArtinEdit<S> {
    innerline_count: usize,
    vertex_count: usize,
    index: usize,
    edit: Option<(InnerlineEdit, ArtinVertices, S)>,
}

type InnerlineEdit = Option<(usize, SectorId)>;

impl<S> ArtinWriter<S, CoreError> for InPlaceArtinEdit<S> {
    type Slot = (InnerlineEdit, ArtinVertices);

    fn begin(
        &mut self,
        innerline: Option<(usize, SectorId)>,
        vertices: ArtinVertices,
    ) -> Result<Self::Slot, CoreError> {
        if innerline.is_some_and(|(position, _)| position >= self.innerline_count) {
            return Err(CoreError::MalformedFusionTree {
                message: artin_innerline_message(vertices),
            });
        }
        if vertices == ArtinVertices::SwapUnit && self.vertex_count <= self.index {
            return Err(CoreError::MalformedFusionTree {
                message: "unit braid past the first adjacent pair requires adjacent vertices",
            });
        }
        Ok((innerline, vertices))
    }

    fn finish(
        &mut self,
        (innerline, vertices): Self::Slot,
        coefficient: S,
    ) -> Result<(), CoreError> {
        self.edit = Some((innerline, vertices, coefficient));
        Ok(())
    }
}

/// First-pair multiplicity-free Artin coefficient: TensorKit
/// `artin_braid(f::FusionTree, i; inv)` at `i == 1`
/// (`braiding_manipulations.jl:56`), `inv ? conj(R(b, a, c)) : R(a, b, c)`.
pub(super) fn mf_artin_first_coefficient<R>(
    rule: &R,
    left: SectorId,
    right: SectorId,
    coupled: SectorId,
    inverse: bool,
) -> R::Scalar
where
    R: MultiplicityFreeFusionSymbols,
{
    if inverse {
        rule.r_symbol_scalar(right, left, coupled).conj()
    } else {
        rule.r_symbol_scalar(left, right, coupled)
    }
}

/// The one multiplicity-free Artin coefficient for a braid past the first
/// pair, over the inner-extended lines `[a, b, c, d, e, c′]`: TensorKit
/// `artin_braid(f::FusionTree, i; inv)` (`braiding_manipulations.jl:70-73`),
/// `inv ? conj(R(d, c, e)·F(d, a, b, e, c′, c))·R(d, a, c′)
///      : R(c, d, e)·conj(F(d, a, b, e, c′, c)·R(a, d, c′))`.
pub(super) fn mf_artin_coefficient<R>(
    rule: &R,
    [a, b, c, d, e, c_prime]: [SectorId; 6],
    inverse: bool,
) -> R::Scalar
where
    R: MultiplicityFreeFusionSymbols,
{
    let f_symbol = rule.f_symbol_scalar(d, a, b, e, c_prime, c);
    if inverse {
        let left = rule.r_symbol_scalar(d, c, e);
        let right = rule.r_symbol_scalar(d, a, c_prime);
        (left * f_symbol).conj() * right
    } else {
        let left = rule.r_symbol_scalar(c, d, e);
        let right = rule.r_symbol_scalar(a, d, c_prime);
        left * (f_symbol * right).conj()
    }
}

pub(super) trait MultiplicityFreeTreeLocalData {
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

#[cfg(test)]
pub(super) trait MultiplicityFreeTreeData: MultiplicityFreeTreeLocalData {
    fn uncoupled(&self) -> &[SectorId];
}

#[cfg(test)]
impl MultiplicityFreeTreeData for FusionTreeKey {
    #[inline]
    fn uncoupled(&self) -> &[SectorId] {
        self.uncoupled()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct MultiplicityFreeTreeFrame {
    pub(super) uncoupled: Arc<[SectorId]>,
    pub(super) is_dual: Arc<[bool]>,
    vertices: Arc<[MultiplicityIndex]>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct MultiplicityFreeTreeLocal {
    pub(super) coupled: SectorId,
    pub(super) innerlines: SectorVec,
}

pub(super) type MultiplicityFreeTreeLocalTerms<S> = Vec<(MultiplicityFreeTreeLocal, S)>;
pub(super) type MultiplicityFreeFoldInverseCache<S> =
    FxHashMap<(SectorId, MultiplicityFreeTreeLocal), MultiplicityFreeTreeLocalTerms<S>>;

mod multiplicity_free_projection {
    use super::*;

    #[derive(Clone, Copy)]
    pub(crate) struct Trees<'a> {
        keys: &'a [FusionTreeKey],
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Pairs<'a> {
        source: PairSource<'a>,
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Tree<'a> {
        key: &'a FusionTreeKey,
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Pair<'a> {
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

    pub(crate) struct TreeBatch<'rule, R> {
        rule: &'rule R,
        keys: SmallVec<[FusionTreeKey; 8]>,
    }

    pub(crate) struct PairBatch<'rule, R> {
        rule: &'rule R,
        keys: SmallVec<[FusionTreePairKey; 8]>,
    }

    impl<'rule, R> TreeBatch<'rule, R>
    where
        R: FusionRule,
    {
        pub(crate) fn from_locally_validated<'structure, I>(
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

        pub(crate) fn parts(&self) -> (&R, &[FusionTreeKey]) {
            (self.rule, &self.keys)
        }
    }

    impl<'rule, R> PairBatch<'rule, R>
    where
        R: FusionRule,
    {
        pub(crate) fn from_locally_validated<'structure, I>(
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

        pub(crate) fn parts(&self) -> (&R, &[FusionTreePairKey]) {
            (self.rule, &self.keys)
        }
    }

    impl<'a> Trees<'a> {
        pub(crate) fn checked<R>(rule: &R, trees: &'a [FusionTreeKey]) -> Result<Self, CoreError>
        where
            R: FusionRule,
        {
            validate_multiplicity_free_execution_style(rule)?;
            for tree in trees {
                Self::check_vertices(tree)?;
            }
            Ok(Self { keys: trees })
        }

        pub(crate) fn from_validated<R>(
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

        pub(crate) fn tree_at(&self, index: usize) -> Option<Tree<'a>> {
            self.keys.get(index).map(|key| Tree { key })
        }

        fn check_vertices(tree: &FusionTreeKey) -> Result<(), CoreError> {
            check_vertices(tree)
        }
    }

    impl<'a> Pairs<'a> {
        pub(crate) fn checked<R>(
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

        pub(crate) fn from_validated<R>(
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

        pub(crate) fn checked_structure<R>(
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

        pub(crate) fn len(&self) -> usize {
            match self.source {
                PairSource::Slice(keys) => keys.len(),
                PairSource::Structure { indices, .. } => indices.len(),
            }
        }

        pub(crate) fn pair_at(&self, index: usize) -> Option<Pair<'a>> {
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
        pub(crate) fn key(self) -> &'a FusionTreeKey {
            self.key
        }
    }

    impl<'a> Pair<'a> {
        pub(crate) fn materialize(self) -> FusionTreePairKey {
            FusionTreePairKey::pair(self.codomain.clone(), self.domain.clone())
        }

        pub(crate) fn codomain(self) -> Tree<'a> {
            Tree { key: self.codomain }
        }

        pub(crate) fn domain(self) -> Tree<'a> {
            Tree { key: self.domain }
        }
    }
}

pub(crate) use multiplicity_free_projection::{
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
    pub(super) fn from_proven(tree: ValidatedMultiplicityFreeTree<'_>) -> Self {
        let tree = tree.key();
        Self {
            coupled: tree.coupled(),
            innerlines: tree.innerlines().iter().copied().collect(),
        }
    }
}

impl MultiplicityFreeTreeFrame {
    pub(super) fn from_frozen_externals(uncoupled: Arc<[SectorId]>, is_dual: Arc<[bool]>) -> Self {
        let vertices =
            std::iter::repeat_n(MultiplicityIndex::ONE, uncoupled.len().saturating_sub(1))
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

    pub(super) fn split(
        tree: ValidatedMultiplicityFreeTree<'_>,
    ) -> (Self, MultiplicityFreeTreeLocal) {
        let key = tree.key();
        (
            Self::from_tree(key),
            MultiplicityFreeTreeLocal::from_proven(tree),
        )
    }

    pub(super) fn matches_tree(&self, tree: &FusionTreeKey) -> bool {
        // Why not split and compare frames: every source shares these slices,
        // while collecting rank > 8 frames would allocate once per source.
        self.uncoupled.as_ref() == tree.uncoupled() && self.is_dual.as_ref() == tree.is_dual()
    }

    pub(super) fn materialize(&self, local: MultiplicityFreeTreeLocal) -> FusionTreeKey {
        FusionTreeKey::from_frozen(
            Arc::clone(&self.uncoupled),
            local.coupled,
            Arc::clone(&self.is_dual),
            Arc::from(local.innerlines.as_slice()),
            Arc::clone(&self.vertices),
        )
    }
}

pub(crate) fn project_multiplicity_free_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<(MultiplicityFreeTreeFrame, MultiplicityFreeTreeLocal), CoreError>
where
    R: FusionRule,
{
    let projection = MultiplicityFreeTreeProjection::checked(rule, std::slice::from_ref(tree))?;
    Ok(MultiplicityFreeTreeFrame::split(
        projection
            .tree_at(0)
            .expect("single-tree projection contains index zero"),
    ))
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct MultiplicityFreeTreePairFrame {
    pub(super) codomain: MultiplicityFreeTreeFrame,
    pub(super) domain: MultiplicityFreeTreeFrame,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct MultiplicityFreeTreePairLocal {
    pub(super) codomain: MultiplicityFreeTreeLocal,
    pub(super) domain: MultiplicityFreeTreeLocal,
}

impl MultiplicityFreeTreePairLocal {
    pub(super) fn from_proven(tree_pair: ValidatedMultiplicityFreeTreePair<'_>) -> Self {
        Self {
            codomain: MultiplicityFreeTreeLocal::from_proven(tree_pair.codomain()),
            domain: MultiplicityFreeTreeLocal::from_proven(tree_pair.domain()),
        }
    }
}

impl MultiplicityFreeTreePairFrame {
    pub(super) fn split(
        tree_pair: ValidatedMultiplicityFreeTreePair<'_>,
    ) -> (Self, MultiplicityFreeTreePairLocal) {
        let codomain = MultiplicityFreeTreeFrame::from_tree(tree_pair.codomain().key());
        let domain = MultiplicityFreeTreeFrame::from_tree(tree_pair.domain().key());
        (
            Self { codomain, domain },
            MultiplicityFreeTreePairLocal::from_proven(tree_pair),
        )
    }

    pub(super) fn matches_tree_pair_ref(
        &self,
        tree_pair: ValidatedMultiplicityFreeTreePair<'_>,
    ) -> bool {
        self.codomain.matches_tree(tree_pair.codomain().key())
            && self.domain.matches_tree(tree_pair.domain().key())
    }

    pub(super) fn materialize(&self, local: MultiplicityFreeTreePairLocal) -> FusionTreePairKey {
        FusionTreePairKey::pair(
            self.codomain.materialize(local.codomain),
            self.domain.materialize(local.domain),
        )
    }
}

pub(super) fn project_multiplicity_free_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(MultiplicityFreeTreePairFrame, MultiplicityFreeTreePairLocal), CoreError>
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

pub(super) struct PreparedMultiplicityFreeArtin {
    pub(super) output_frame: MultiplicityFreeTreeFrame,
    site: ArtinSite,
}

pub(super) fn prepare_multiplicity_free_artin<R>(
    rule: &R,
    frame: &MultiplicityFreeTreeFrame,
    index: usize,
    inverse: bool,
) -> Result<PreparedMultiplicityFreeArtin, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
{
    let site = ArtinSite::new(&SimpleK(rule), &frame.uncoupled, index, inverse)?;
    let mut uncoupled = frame.uncoupled.to_vec();
    uncoupled.swap(index, index + 1);
    let mut is_dual = frame.is_dual.to_vec();
    is_dual.swap(index, index + 1);
    let output_frame = MultiplicityFreeTreeFrame {
        uncoupled: uncoupled.into(),
        is_dual: is_dual.into(),
        vertices: Arc::clone(&frame.vertices),
    };
    Ok(PreparedMultiplicityFreeArtin { output_frame, site })
}

impl PreparedMultiplicityFreeArtin {
    // External frame data stays out of this kernel: the block runner must enumerate
    // locals from this prepared frame rather than rebuilding full tree keys.
    pub(super) fn apply<R>(
        &self,
        rule: &R,
        tree: &MultiplicityFreeTreeLocal,
    ) -> Result<MultiplicityFreeArtinTerms<R::Scalar>, CoreError>
    where
        R: MultiplicityFreeFusionSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        let mut out = LocalArtinTerms {
            tree,
            terms: SmallVec::new(),
        };
        artin_surgery(&SimpleK(rule), &self.site, tree, &mut out)?;
        Ok(out.terms)
    }
}

/// Multiplicity-free Artin outputs as compact locals.
struct LocalArtinTerms<'t, S> {
    tree: &'t MultiplicityFreeTreeLocal,
    terms: MultiplicityFreeArtinTerms<S>,
}

impl<S> ArtinWriter<S, CoreError> for LocalArtinTerms<'_, S> {
    type Slot = MultiplicityFreeTreeLocal;

    fn begin(
        &mut self,
        innerline: Option<(usize, SectorId)>,
        vertices: ArtinVertices,
    ) -> Result<Self::Slot, CoreError> {
        let mut innerlines: SectorVec = self.tree.innerlines.iter().copied().collect();
        if let Some((position, sector)) = innerline {
            *innerlines
                .get_mut(position)
                .ok_or(CoreError::MalformedFusionTree {
                    message: artin_innerline_message(vertices),
                })? = sector;
        }
        Ok(MultiplicityFreeTreeLocal {
            coupled: self.tree.coupled,
            innerlines,
        })
    }

    fn finish(&mut self, local: Self::Slot, coefficient: S) -> Result<(), CoreError> {
        self.terms.push((local, coefficient));
        Ok(())
    }
}

#[expect(
    clippy::type_complexity,
    reason = "the SmallVec inline capacity is part of this local braid allocation contract"
)]
pub(crate) fn multiplicity_free_artin_braid_at_with_inverse<R>(
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
        .map(|(local, coefficient)| (prepared.output_frame.materialize(local), coefficient))
        .collect())
}
