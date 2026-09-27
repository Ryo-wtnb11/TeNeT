#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreeGroupKey {
    codomain_uncoupled: Arc<[SectorId]>,
    domain_uncoupled: Arc<[SectorId]>,
    codomain_is_dual: Arc<[bool]>,
    domain_is_dual: Arc<[bool]>,
}

impl FusionTreeGroupKey {
    /// Heap bytes of this group's shared backing slices, each counted once
    /// across calls that share `seen`.
    #[doc(hidden)]
    pub fn charge_retained_backings(&self, seen: &mut rustc_hash::FxHashSet<usize>) -> usize {
        charge_fusion_tree_group_key_backings(seen, self)
    }

    pub fn new<Codomain, Domain, CodomainDual, DomainDual>(
        codomain_uncoupled: Codomain,
        domain_uncoupled: Domain,
        codomain_is_dual: CodomainDual,
        domain_is_dual: DomainDual,
    ) -> Self
    where
        Codomain: IntoIterator<Item = SectorId>,
        Domain: IntoIterator<Item = SectorId>,
        CodomainDual: IntoIterator<Item = bool>,
        DomainDual: IntoIterator<Item = bool>,
    {
        Self {
            codomain_uncoupled: codomain_uncoupled.into_iter().collect::<Vec<_>>().into(),
            domain_uncoupled: domain_uncoupled.into_iter().collect::<Vec<_>>().into(),
            codomain_is_dual: codomain_is_dual.into_iter().collect::<Vec<_>>().into(),
            domain_is_dual: domain_is_dual.into_iter().collect::<Vec<_>>().into(),
        }
    }

    fn from_frozen(
        codomain_uncoupled: Arc<[SectorId]>,
        domain_uncoupled: Arc<[SectorId]>,
        codomain_is_dual: Arc<[bool]>,
        domain_is_dual: Arc<[bool]>,
    ) -> Self {
        Self {
            codomain_uncoupled,
            domain_uncoupled,
            codomain_is_dual,
            domain_is_dual,
        }
    }

    pub fn from_sector_ids<Codomain, Domain, CodomainDual, DomainDual>(
        codomain_uncoupled: Codomain,
        domain_uncoupled: Domain,
        codomain_is_dual: CodomainDual,
        domain_is_dual: DomainDual,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
        CodomainDual: IntoIterator<Item = bool>,
        DomainDual: IntoIterator<Item = bool>,
    {
        Self::new(
            codomain_uncoupled.into_iter().map(SectorId::new),
            domain_uncoupled.into_iter().map(SectorId::new),
            codomain_is_dual,
            domain_is_dual,
        )
    }

    #[inline]
    pub fn codomain_uncoupled(&self) -> &[SectorId] {
        &self.codomain_uncoupled
    }

    #[inline]
    pub fn domain_uncoupled(&self) -> &[SectorId] {
        &self.domain_uncoupled
    }

    #[inline]
    pub fn codomain_is_dual(&self) -> &[bool] {
        &self.codomain_is_dual
    }

    #[inline]
    pub fn domain_is_dual(&self) -> &[bool] {
        &self.domain_is_dual
    }
}

// Why the hash is stored: keys are built once per structure and looked up
// many times per publication, so an `O(K)` hash per lookup (TensorKit's
// `hash(::FusionTree)` per `Dict` access) is avoidable constant work. The
// cache is a pure function of the five identity fields, is excluded from
// `Ord` and `Debug`, and is never persisted.
#[derive(Clone)]
pub struct FusionTreeKey {
    uncoupled: Arc<[SectorId]>,
    coupled: SectorId,
    is_dual: Arc<[bool]>,
    innerlines: Arc<[SectorId]>,
    vertices: Arc<[MultiplicityIndex]>,
    hash: u64,
}

#[inline]
fn fusion_tree_key_hash(
    uncoupled: &[SectorId],
    coupled: SectorId,
    is_dual: &[bool],
    innerlines: &[SectorId],
    vertices: &[MultiplicityIndex],
) -> u64 {
    use std::hash::Hasher;
    let mut hasher = rustc_hash::FxHasher::default();
    uncoupled.hash(&mut hasher);
    coupled.hash(&mut hasher);
    is_dual.hash(&mut hasher);
    innerlines.hash(&mut hasher);
    vertices.hash(&mut hasher);
    hasher.finish()
}

impl Hash for FusionTreeKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl PartialEq for FusionTreeKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && self.coupled == other.coupled
            && self.uncoupled == other.uncoupled
            && self.is_dual == other.is_dual
            && self.innerlines == other.innerlines
            && self.vertices == other.vertices
    }
}

impl Eq for FusionTreeKey {}

impl Ord for FusionTreeKey {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.uncoupled
            .cmp(&other.uncoupled)
            .then_with(|| self.coupled.cmp(&other.coupled))
            .then_with(|| self.is_dual.cmp(&other.is_dual))
            .then_with(|| self.innerlines.cmp(&other.innerlines))
            .then_with(|| self.vertices.cmp(&other.vertices))
    }
}

impl PartialOrd for FusionTreeKey {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for FusionTreeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FusionTreeKey")
            .field("uncoupled", &self.uncoupled)
            .field("coupled", &self.coupled)
            .field("is_dual", &self.is_dual)
            .field("innerlines", &self.innerlines)
            .field("vertices", &self.vertices)
            .finish()
    }
}

impl FusionTreeKey {
    #[cfg(test)]
    pub(crate) fn cached_hash_for_test(&self) -> u64 {
        self.hash
    }

    /// Test-only hook that overrides the cached hash so collision handling can
    /// be exercised deterministically; it never changes the identity fields.
    #[cfg(test)]
    pub(crate) fn with_cached_hash_for_test(mut self, hash: u64) -> Self {
        self.hash = hash;
        self
    }

    /// Construct and validate a categorical fusion tree for `rule`.
    ///
    /// # Provider-domain precondition
    ///
    /// Every input [`SectorId`] and every channel generated by an exercised
    /// algebra query must be representable by `rule`. This validates
    /// categorical structure, not finite-provider closure, and has the same
    /// limitation documented on [`Self::validate_for_rule`].
    pub fn try_new_for_rule<R, Uncoupled, Dual, Innerlines, Vertices>(
        rule: &R,
        uncoupled: Uncoupled,
        coupled: SectorId,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
        Uncoupled: IntoIterator<Item = SectorId>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = SectorId>,
        Vertices: IntoIterator<Item = MultiplicityIndex>,
    {
        let tree = Self::new(uncoupled, coupled, is_dual, innerlines, vertices);
        tree.validate_for_rule(rule)?;
        Ok(tree)
    }

    /// Construct and validate a fusion tree through checked finite algebra.
    pub fn try_new_for_rule_checked<R, Uncoupled, Dual, Innerlines, Vertices>(
        rule: &R,
        uncoupled: Uncoupled,
        coupled: SectorId,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
        Uncoupled: IntoIterator<Item = SectorId>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = SectorId>,
        Vertices: IntoIterator<Item = MultiplicityIndex>,
    {
        let tree = Self::new(uncoupled, coupled, is_dual, innerlines, vertices);
        tree.validate_for_rule_checked(rule)?;
        Ok(tree)
    }

    // Why not store a fusion style here: the bound rule is the authority for
    // categorical capabilities; duplicating it in identity can become stale.
    pub(crate) fn new<Uncoupled, Dual, Innerlines, Vertices>(
        uncoupled: Uncoupled,
        coupled: SectorId,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Self
    where
        Uncoupled: IntoIterator<Item = SectorId>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = SectorId>,
        Vertices: IntoIterator<Item = MultiplicityIndex>,
    {
        Self::from_frozen(
            uncoupled.into_iter().collect::<Vec<_>>().into(),
            coupled,
            is_dual.into_iter().collect::<Vec<_>>().into(),
            innerlines.into_iter().collect::<Vec<_>>().into(),
            vertices.into_iter().collect::<Vec<_>>().into(),
        )
    }

    fn from_frozen(
        uncoupled: Arc<[SectorId]>,
        coupled: SectorId,
        is_dual: Arc<[bool]>,
        innerlines: Arc<[SectorId]>,
        vertices: Arc<[MultiplicityIndex]>,
    ) -> Self {
        let hash = fusion_tree_key_hash(&uncoupled, coupled, &is_dual, &innerlines, &vertices);
        Self {
            uncoupled,
            coupled,
            is_dual,
            innerlines,
            vertices,
            hash,
        }
    }

    pub(crate) fn try_from_sector_ids<Uncoupled, Dual, Innerlines, Vertices>(
        uncoupled: Uncoupled,
        coupled: usize,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CoreError>
    where
        Uncoupled: IntoIterator<Item = usize>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = usize>,
        Vertices: IntoIterator<Item = usize>,
    {
        Ok(Self::new(
            uncoupled.into_iter().map(SectorId::new),
            SectorId::new(coupled),
            is_dual,
            innerlines.into_iter().map(SectorId::new),
            vertices
                .into_iter()
                .map(MultiplicityIndex::try_from)
                .collect::<Result<MultiplicityVec, _>>()?,
        ))
    }

    /// Construct and validate a categorical fusion tree from numeric labels.
    ///
    /// `uncoupled`, `coupled`, and `innerlines` contain provider-local sector
    /// IDs. `vertices` instead contains one-based outer-multiplicity labels;
    /// a zero vertex returns [`CoreError::InvalidMultiplicityIndex`].
    ///
    /// # Provider-domain precondition
    ///
    /// Every input sector ID and every channel generated by an exercised
    /// algebra query must be representable by `rule`. This method checks
    /// categorical structure, not finite-provider closure, and has the same
    /// limitation documented on [`Self::validate_for_rule`].
    pub fn try_from_sector_ids_for_rule<R, Uncoupled, Dual, Innerlines, Vertices>(
        rule: &R,
        uncoupled: Uncoupled,
        coupled: usize,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
        Uncoupled: IntoIterator<Item = usize>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = usize>,
        Vertices: IntoIterator<Item = usize>,
    {
        Self::try_new_for_rule(
            rule,
            uncoupled.into_iter().map(SectorId::new),
            SectorId::new(coupled),
            is_dual,
            innerlines.into_iter().map(SectorId::new),
            vertices
                .into_iter()
                .map(MultiplicityIndex::try_from)
                .collect::<Result<MultiplicityVec, _>>()?,
        )
    }

    /// Construct and checked-validate a fusion tree from numeric labels.
    ///
    /// Sector values are provider-local IDs. Vertex values are one-based
    /// outer-multiplicity labels.
    pub fn try_from_sector_ids_for_rule_checked<
        R,
        Uncoupled,
        Dual,
        Innerlines,
        Vertices,
    >(
        rule: &R,
        uncoupled: Uncoupled,
        coupled: usize,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
        Uncoupled: IntoIterator<Item = usize>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = usize>,
        Vertices: IntoIterator<Item = usize>,
    {
        let vertices = vertices
            .into_iter()
            .map(MultiplicityIndex::try_from)
            .collect::<Result<MultiplicityVec, _>>()?;
        Self::try_new_for_rule_checked(
            rule,
            uncoupled.into_iter().map(SectorId::new),
            SectorId::new(coupled),
            is_dual,
            innerlines.into_iter().map(SectorId::new),
            vertices,
        )
    }

    #[inline]
    pub fn uncoupled(&self) -> &[SectorId] {
        &self.uncoupled
    }

    #[inline]
    pub fn coupled(&self) -> SectorId {
        self.coupled
    }

    #[inline]
    pub fn is_dual(&self) -> &[bool] {
        &self.is_dual
    }

    #[inline]
    pub fn innerlines(&self) -> &[SectorId] {
        &self.innerlines
    }

    #[inline]
    pub fn vertices(&self) -> &[MultiplicityIndex] {
        &self.vertices
    }

    /// Validate this raw key as a categorical fusion tree for `rule`.
    ///
    /// Ruleless construction remains available for exact categorical
    /// reconstruction, deserialization, and expert import. Call this boundary
    /// before categorical execution; application routing labels are represented
    /// by [`OpaqueBlockKey`], not raw fusion trees.
    ///
    /// # Provider-domain precondition
    ///
    /// The infallible [`FusionRule`] contract assumes every [`SectorId`]
    /// belongs to the provider domain, and every channel generated by an
    /// exercised algebra query is representable. Arbitrary out-of-domain
    /// labels, provider nonclosure, and coherence are not checked here; a
    /// finite-table provider may panic when queried outside that precondition.
    pub fn validate_for_rule<R>(&self, rule: &R) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        validate_fusion_tree_for_rule(rule, self)?;
        Ok(())
    }

    /// Validate this raw key through checked finite algebra.
    pub fn validate_for_rule_checked<R>(
        &self,
        rule: &R,
    ) -> Result<(), CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        validate_fusion_tree_for_rule_checked(rule, self)?;
        Ok(())
    }
}

const FROZEN_SLICE_CONTROL_BYTES: usize = 2 * std::mem::size_of::<usize>();

fn charge_frozen_slice<T>(seen: &mut rustc_hash::FxHashSet<usize>, slice: &Arc<[T]>) -> usize {
    let pointer = Arc::as_ptr(slice) as *const T as usize;
    if seen.insert(pointer) {
        FROZEN_SLICE_CONTROL_BYTES.saturating_add(
            slice
                .len()
                .saturating_mul(std::mem::size_of::<T>()),
        )
    } else {
        0
    }
}

fn charge_fusion_tree_key_backings(
    seen: &mut rustc_hash::FxHashSet<usize>,
    tree: &FusionTreeKey,
) -> usize {
    charge_frozen_slice(seen, &tree.uncoupled)
        .saturating_add(charge_frozen_slice(seen, &tree.is_dual))
        .saturating_add(charge_frozen_slice(seen, &tree.innerlines))
        .saturating_add(charge_frozen_slice(seen, &tree.vertices))
}

fn charge_fusion_tree_group_key_backings(
    seen: &mut rustc_hash::FxHashSet<usize>,
    group: &FusionTreeGroupKey,
) -> usize {
    charge_frozen_slice(seen, &group.codomain_uncoupled)
        .saturating_add(charge_frozen_slice(seen, &group.domain_uncoupled))
        .saturating_add(charge_frozen_slice(seen, &group.codomain_is_dual))
        .saturating_add(charge_frozen_slice(seen, &group.domain_is_dual))
}

#[derive(Clone, Copy)]
struct ValidatedFusionTree<'a, R> {
    rule: &'a R,
    key: &'a FusionTreeKey,
}

fn validate_fusion_tree_for_rule<'a, R>(
    rule: &'a R,
    tree: &'a FusionTreeKey,
) -> Result<ValidatedFusionTree<'a, R>, CoreError>
where
    R: FusionRule,
{
    validate_fusion_tree_structure(rule, tree)?;
    validate_fusion_tree_vertices(tree, |left, right, coupled| {
        Ok(rule.nsymbol(left, right, coupled))
    })?;

    Ok(ValidatedFusionTree { rule, key: tree })
}

fn validate_fusion_tree_for_rule_checked<'a, R>(
    rule: &'a R,
    tree: &'a FusionTreeKey,
) -> Result<ValidatedFusionTree<'a, R>, CheckedFusionSpaceError>
where
    R: CheckedFusionAlgebra,
{
    ShapeValidatedFusionTree::try_new(tree)?.validate_for_rule_checked(rule)
}

fn validate_fusion_tree_for_rule_checked_after_shape<'tree, R>(
    rule: &'tree R,
    tree: &'tree FusionTreeKey,
) -> Result<ValidatedFusionTree<'tree, R>, CheckedFusionSpaceError>
where
    R: CheckedFusionAlgebra,
{
    ShapeValidatedFusionTree { tree }.validate_for_rule_checked(rule)
}

#[derive(Clone, Copy)]
struct ShapeValidatedFusionTree<'tree> {
    tree: &'tree FusionTreeKey,
}

impl<'tree> ShapeValidatedFusionTree<'tree> {
    fn try_new(tree: &'tree FusionTreeKey) -> Result<Self, CoreError> {
        validate_fusion_tree_key_shape(tree)?;
        Ok(Self { tree })
    }

    fn validate_for_rule_checked<R>(
        self,
        rule: &'tree R,
    ) -> Result<ValidatedFusionTree<'tree, R>, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        validate_fusion_tree_structure_after_shape(rule, self.tree)?;
        if self.tree.uncoupled().len() == 1 {
        let multiplicity =
                rule.try_nsymbol(self.tree.coupled(), rule.vacuum(), self.tree.coupled())?;
            if multiplicity == 0 {
                return Err(CoreError::MalformedFusionTree {
                    message: "rank-1 fusion tree sector is absent from unit fusion",
                }
                .into());
            }
        }
        validate_fusion_tree_vertices(self.tree, |left, right, coupled| {
            rule.try_fusion_channels(left, right)?;
            rule.try_nsymbol(left, right, coupled)
                .map_err(CheckedFusionSpaceError::from)
        })?;
        Ok(ValidatedFusionTree {
            rule,
            key: self.tree,
        })
    }
}

fn validate_fusion_tree_structure<R>(rule: &R, tree: &FusionTreeKey) -> Result<(), CoreError>
where
    R: FusionRule,
{
    validate_fusion_tree_key_shape(tree)?;
    validate_fusion_tree_structure_after_shape(rule, tree)
}

fn validate_fusion_tree_structure_after_shape<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<(), CoreError>
where
    R: FusionRule,
{
    match tree.uncoupled().len() {
        0 if tree.coupled() != rule.vacuum() => Err(CoreError::MalformedFusionTree {
            message: "rank-0 fusion tree coupled sector must equal the vacuum",
        }),
        1 if Some(tree.coupled()) != tree.uncoupled().first().copied() => {
            Err(CoreError::MalformedFusionTree {
                message: "rank-1 fusion tree coupled sector must equal its uncoupled sector",
            })
        }
        _ => Ok(()),
    }
}

fn validate_fusion_tree_vertices<E>(
    tree: &FusionTreeKey,
    mut multiplicity: impl FnMut(SectorId, SectorId, SectorId) -> Result<usize, E>,
) -> Result<(), E>
where
    E: From<CoreError>,
{
    let rank = tree.uncoupled().len();
    for vertex_index in 0..rank.saturating_sub(1) {
        let left = if vertex_index == 0 {
            tree.uncoupled()[0]
        } else {
            tree.innerlines()[vertex_index - 1]
        };
        let right = tree.uncoupled()[vertex_index + 1];
        let coupled = if vertex_index + 2 == rank {
            tree.coupled()
        } else {
            tree.innerlines()[vertex_index]
        };
        let multiplicity = multiplicity(left, right, coupled)?;
        if multiplicity == 0 {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            }
            .into());
        }
        if tree.vertices()[vertex_index].get() > multiplicity {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            }
            .into());
        }
    }
    Ok(())
}

impl FusionTreePairKey {
    /// Heap bytes of this pair's shared backing slices, each counted once
    /// across calls that share `seen`.
    #[doc(hidden)]
    pub fn charge_retained_backings(&self, seen: &mut rustc_hash::FxHashSet<usize>) -> usize {
        charge_fusion_tree_key_backings(seen, &self.codomain_tree).saturating_add(
            charge_fusion_tree_key_backings(seen, &self.domain_tree),
        )
    }

    /// Validate both trees and their shared coupled sector.
    ///
    /// # Provider-domain precondition
    ///
    /// Both trees follow [`FusionTreeKey::validate_for_rule`]'s provider-domain
    /// precondition.
    pub fn validate_for_rule<R>(&self, rule: &R) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        validate_fusion_tree_pair_for_rule(rule, self)?;
        Ok(())
    }

    /// Validate both trees and their shared coupled sector through checked
    /// finite algebra.
    pub fn validate_for_rule_checked<R>(
        &self,
        rule: &R,
    ) -> Result<(), CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        validate_fusion_tree_pair_for_rule_checked(rule, self)?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ValidatedFusionTreePair<'a, R> {
    rule: &'a R,
    key: &'a FusionTreePairKey,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FusionTreePairOrientation {
    Direct,
    Adjoint,
}

fn validate_fusion_tree_pair_for_rule<'a, R>(
    rule: &'a R,
    tree_pair: &'a FusionTreePairKey,
) -> Result<ValidatedFusionTreePair<'a, R>, CoreError>
where
    R: FusionRule,
{
    let codomain = validate_fusion_tree_for_rule(rule, tree_pair.codomain_tree())?;
    let domain = validate_fusion_tree_for_rule(rule, tree_pair.domain_tree())?;
    validate_fusion_tree_pair_coupled(codomain.key, domain.key)?;
    Ok(ValidatedFusionTreePair {
        rule,
        key: tree_pair,
    })
}

fn validate_fusion_tree_pair_for_rule_checked<'a, R>(
    rule: &'a R,
    tree_pair: &'a FusionTreePairKey,
) -> Result<ValidatedFusionTreePair<'a, R>, CheckedFusionSpaceError>
where
    R: CheckedFusionAlgebra,
{
    let codomain = validate_fusion_tree_for_rule_checked(rule, tree_pair.codomain_tree())?;
    let domain = validate_fusion_tree_for_rule_checked(rule, tree_pair.domain_tree())?;
    validate_fusion_tree_pair_coupled(codomain.key, domain.key)?;
    Ok(ValidatedFusionTreePair {
        rule,
        key: tree_pair,
    })
}

fn validate_fusion_tree_pair_coupled(
    codomain: &FusionTreeKey,
    domain: &FusionTreeKey,
) -> Result<(), CoreError> {
    if codomain.coupled() == domain.coupled() {
        Ok(())
    } else {
        Err(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        })
    }
}

fn validate_fusion_tree_key_shape(tree: &FusionTreeKey) -> Result<(), CoreError> {
    let rank = tree.uncoupled().len();
    if tree.is_dual().len() != rank {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree sectors and duality flags must have matching length",
        });
    }
    let expected_innerlines = rank.saturating_sub(2);
    if tree.innerlines().len() != expected_innerlines {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of innerlines",
        });
    }
    let expected_vertices = rank.saturating_sub(1);
    if tree.vertices().len() != expected_vertices {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of vertices",
        });
    }
    Ok(())
}
