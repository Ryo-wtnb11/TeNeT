#[derive(Clone, Debug, Eq, PartialEq)]
struct CoupledFusionTrees {
    coupled: SectorId,
    trees: Vec<FusionTreeKey>,
}

fn collect_selected_leg_tuples(
    legs: &[SectorLeg],
    remaining: usize,
    current: &mut [Option<FusionTreeLeg>],
    tuples: &mut Vec<Vec<FusionTreeLeg>>,
) {
    if remaining == 0 {
        tuples.push(
            current
                .iter()
                .map(|leg| leg.expect("fusion tree leg tuple should be fully assigned"))
                .collect(),
        );
        return;
    }

    let index = remaining - 1;
    for &sector in legs[index].sectors() {
        current[index] = Some(FusionTreeLeg::new(sector, legs[index].is_dual()));
        collect_selected_leg_tuples(legs, remaining - 1, current, tuples);
    }
}

fn fusion_trees_by_coupled_for_space<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> Vec<CoupledFusionTrees>
where
    R: MultiplicityFreeFusionRule,
{
    // Group trees by coupled sector via a `coupled -> index` map so the merge
    // is O(1) per (tuple, coupled) pair. The previous `grouped.iter_mut().find`
    // linear scan was O(P·C) (P = tuple×coupled iterations, C = distinct
    // coupled sectors); the map removes the C factor. The final `sort_by_key`
    // still fixes the canonical order, so the map need not preserve it.
    let mut grouped = Vec::<CoupledFusionTrees>::new();
    let mut index: FxHashMap<SectorId, usize> = FxHashMap::default();
    for tuple in space.selected_leg_tuples() {
        let effective = effective_sectors(rule, &tuple);
        let uncoupled: Vec<SectorId> = tuple.iter().map(|leg| leg.sector()).collect();
        let is_dual: Vec<bool> = tuple.iter().map(|leg| leg.is_dual()).collect();
        for coupled in reachable_coupled_sectors(rule, &effective) {
            let trees =
                collect_fusion_trees_for_coupled(rule, &uncoupled, &is_dual, &effective, coupled);
            match index.get(&coupled) {
                Some(&i) => grouped[i].trees.extend(trees),
                None => {
                    index.insert(coupled, grouped.len());
                    grouped.push(CoupledFusionTrees { coupled, trees });
                }
            }
        }
    }
    grouped.sort_by_key(|group| group.coupled);
    grouped
}

fn fusion_trees_by_coupled_for_selected_space<R>(
    rule: &R,
    space: &FusionProductSpace,
    selected: &[SectorId],
) -> Result<Vec<CoupledFusionTrees>, CoreError>
where
    R: MultiplicityFreeFusionRule,
{
    if selected.len() != space.len() {
        return Err(CoreError::DimensionMismatch {
            expected: space.len(),
            actual: selected.len(),
        });
    }
    for (&sector, leg) in selected.iter().zip(space.legs()) {
        // Sectors are stored sorted (SortedVectorDict invariant); binary-search
        // to stay consistent with `SectorLeg::degeneracy`.
        if leg.sectors().binary_search(&sector).is_err() {
            return Err(CoreError::InvalidSector { sector });
        }
    }

    let legs = selected
        .iter()
        .zip(space.legs())
        .map(|(&sector, leg)| FusionTreeLeg::new(sector, leg.is_dual()))
        .collect::<Vec<_>>();
    let effective = effective_sectors(rule, &legs);
    let uncoupled: Vec<SectorId> = legs.iter().map(|leg| leg.sector()).collect();
    let is_dual: Vec<bool> = legs.iter().map(|leg| leg.is_dual()).collect();
    let mut grouped = Vec::new();
    for coupled in reachable_coupled_sectors(rule, &effective) {
        let trees =
            collect_fusion_trees_for_coupled(rule, &uncoupled, &is_dual, &effective, coupled);
        if !trees.is_empty() {
            grouped.push(CoupledFusionTrees { coupled, trees });
        }
    }
    grouped.sort_by_key(|group| group.coupled);
    Ok(grouped)
}

/// Coupled sectors reachable by fusing all legs — TensorKit's `blocksectors`.
/// Computed once per leg tuple (not per enumeration node): the forward fold
/// `⊗` over the legs with dedup. Used only to drive the per-coupled grouping;
/// the tree enumeration itself does not consult it (see below).
fn reachable_coupled_sectors<R>(rule: &R, effective: &[SectorId]) -> Vec<SectorId>
where
    R: MultiplicityFreeFusionRule,
{
    let mut acc: Vec<SectorId> = match effective.first() {
        None => vec![rule.vacuum()],
        Some(&first) => vec![first],
    };
    for &last in effective.iter().skip(1) {
        acc = acc
            .iter()
            .flat_map(|&front| rule.fusion_channels(front, last))
            .collect();
        acc.sort_unstable();
        acc.dedup();
    }
    acc.sort_unstable();
    acc.dedup();
    acc
}

/// Enumerate the fusion trees of `uncoupled` (with `is_dual`) into `coupled`,
/// ported from TensorKit's `_fusiontree_iterate` (fusiontrees/iterator.jl).
/// It walks the inner lines *backward* from `coupled`: peel the last leg `b`,
/// let the adjacent inner line `a` range over `coupled ⊗ dual(b)`, recurse on
/// the front legs fusing to `a`, and prune dead branches by the recursion
/// yielding nothing — no forward `possible_coupled` reachability set, matching
/// TensorKit. Like TensorKit's *lazy* iterator it never materializes an
/// intermediate tree list per recursion level: a single `visit` walk pushes
/// each completed key straight into `out`, threading one reused inner-line
/// stack. Multiplicity-free, so every vertex is the trivial label.
fn collect_fusion_trees_for_coupled<R>(
    rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
) -> Vec<FusionTreeKey>
where
    R: MultiplicityFreeFusionRule,
{
    // Vertices are the trivial label for every multiplicity-free tree; a tree of
    // `n` legs has `n - 1` vertices (and `n - 2` inner lines), or none for n < 2.
    let vertex_count = uncoupled.len().saturating_sub(1);
    let mut out = Vec::new();
    // `inner_rev` accumulates the inner lines outermost-first as the walk
    // descends; the stored key wants innermost-first, so emit reverses it.
    let mut inner_rev: Vec<SectorId> = Vec::new();
    visit_fusion_trees(rule, effective, coupled, &mut inner_rev, &mut |inner_rev| {
        out.push(FusionTreeKey::new(
            uncoupled.iter().copied(),
            Some(coupled),
            is_dual.iter().copied(),
            inner_rev.iter().rev().copied(),
            std::iter::repeat(SectorId::new(1)).take(vertex_count),
        ));
    });
    out
}

fn visit_fusion_trees<R, F>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
    inner_rev: &mut Vec<SectorId>,
    emit: &mut F,
) where
    R: MultiplicityFreeFusionRule,
    F: FnMut(&[SectorId]),
{
    match effective.len() {
        0 => {
            if coupled == rule.vacuum() {
                emit(inner_rev);
            }
        }
        1 => {
            if effective[0] == coupled {
                emit(inner_rev);
            }
        }
        2 => {
            if rule.nsymbol(effective[0], effective[1], coupled) != 0 {
                emit(inner_rev);
            }
        }
        _ => {
            let last = effective[effective.len() - 1];
            let front_effective = &effective[..effective.len() - 1];
            // Inner line `a` ranges over `coupled ⊗ dual(last)` (TensorKit's
            // `vertexiterN = coupled ⊗ dual(b)`); `Nsymbol(a, last, coupled)` is
            // the last vertex. No forward-reachability filter — dead `a` simply
            // emit nothing from the recursion.
            for front_coupled in rule.fusion_channels(coupled, rule.dual(last)) {
                if rule.nsymbol(front_coupled, last, coupled) == 0 {
                    continue;
                }
                inner_rev.push(front_coupled);
                visit_fusion_trees(rule, front_effective, front_coupled, inner_rev, emit);
                inner_rev.pop();
            }
        }
    }
}

fn effective_sectors<R>(_rule: &R, legs: &[FusionTreeLeg]) -> Vec<SectorId>
where
    R: MultiplicityFreeFusionRule,
{
    legs.iter().map(|leg| leg.sector()).collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreeGroupKey {
    codomain_uncoupled: SectorVec,
    domain_uncoupled: SectorVec,
    codomain_is_dual: DualVec,
    domain_is_dual: DualVec,
}

impl FusionTreeGroupKey {
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
            codomain_uncoupled: codomain_uncoupled.into_iter().collect(),
            domain_uncoupled: domain_uncoupled.into_iter().collect(),
            codomain_is_dual: codomain_is_dual.into_iter().collect(),
            domain_is_dual: domain_is_dual.into_iter().collect(),
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

#[derive(Clone, Debug)]
pub struct FusionTreeKey {
    uncoupled: SectorVec,
    coupled: Option<SectorId>,
    is_dual: DualVec,
    innerlines: SectorVec,
    vertices: SectorVec,
}

// Identity of a `FusionTreeKey` is `(uncoupled, coupled, is_dual, innerlines)`
// — `vertices` is deliberately excluded from Hash/Eq/Ord. For multiplicity-free
// fusion (every rule in this crate) the vertex labels are functionally
// determined by those four fields (always the trivial vertex), so two keys that
// agree on them agree on `vertices` too: excluding it changes no equivalence
// class or ordering, only the per-op cost. FusionTreeKey comparison/hashing is
// the hottest logic in the cold recoupling-plan build; TensorKit likewise keys
// its `SimpleFusion` fusion trees on the sectors alone. All three impls use the
// SAME four fields so the Hash/Eq and Ord/Eq contracts hold.
impl std::hash::Hash for FusionTreeKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.uncoupled.hash(state);
        self.coupled.hash(state);
        self.is_dual.hash(state);
        self.innerlines.hash(state);
    }
}

impl PartialEq for FusionTreeKey {
    fn eq(&self, other: &Self) -> bool {
        self.uncoupled == other.uncoupled
            && self.coupled == other.coupled
            && self.is_dual == other.is_dual
            && self.innerlines == other.innerlines
    }
}

impl Eq for FusionTreeKey {}

impl Ord for FusionTreeKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.uncoupled
            .cmp(&other.uncoupled)
            .then_with(|| self.coupled.cmp(&other.coupled))
            .then_with(|| self.is_dual.cmp(&other.is_dual))
            .then_with(|| self.innerlines.cmp(&other.innerlines))
    }
}

impl PartialOrd for FusionTreeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl FusionTreeKey {
    pub fn new<Uncoupled, Dual, Innerlines, Vertices>(
        uncoupled: Uncoupled,
        coupled: Option<SectorId>,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Self
    where
        Uncoupled: IntoIterator<Item = SectorId>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = SectorId>,
        Vertices: IntoIterator<Item = SectorId>,
    {
        Self {
            uncoupled: uncoupled.into_iter().collect(),
            coupled,
            is_dual: is_dual.into_iter().collect(),
            innerlines: innerlines.into_iter().collect(),
            vertices: vertices.into_iter().collect(),
        }
    }

    pub fn from_sector_ids<Uncoupled, Dual, Innerlines, Vertices>(
        uncoupled: Uncoupled,
        coupled: Option<usize>,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Self
    where
        Uncoupled: IntoIterator<Item = usize>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = usize>,
        Vertices: IntoIterator<Item = usize>,
    {
        Self::new(
            uncoupled.into_iter().map(SectorId::new),
            coupled.map(SectorId::new),
            is_dual,
            innerlines.into_iter().map(SectorId::new),
            vertices.into_iter().map(SectorId::new),
        )
    }

    pub fn from_uncoupled<I>(uncoupled: I) -> Self
    where
        I: IntoIterator<Item = SectorId>,
    {
        let uncoupled = uncoupled.into_iter().collect::<Vec<_>>();
        Self::new(
            uncoupled.clone(),
            None,
            vec![false; uncoupled.len()],
            Vec::new(),
            Vec::new(),
        )
    }

    #[inline]
    pub fn uncoupled(&self) -> &[SectorId] {
        &self.uncoupled
    }

    #[inline]
    pub fn coupled(&self) -> Option<SectorId> {
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
    pub fn vertices(&self) -> &[SectorId] {
        &self.vertices
    }

    fn compact_id(&self) -> Option<usize> {
        if self.uncoupled.len() == 1
            && self.coupled.is_none()
            && self.innerlines.is_empty()
            && self.vertices.is_empty()
        {
            Some(self.uncoupled[0].id())
        } else {
            None
        }
    }
}

/// Split a left-associated fusion tree using TensorKit's `split(f, m)`
/// convention.
///
/// The first output contains the first `front_rank` uncoupled sectors. The
/// second output starts with the intermediate sector between the two pieces and
/// then contains the remaining uncoupled sectors. This is a structural
/// categorical operation: no dense storage is touched and no coefficient is
/// introduced.
pub fn split_fusion_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    front_rank: usize,
) -> Result<(FusionTreeKey, FusionTreeKey), CoreError>
where
    R: FusionRule,
{
    let rank = tree.uncoupled().len();
    if front_rank > rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: front_rank,
        });
    }
    validate_fusion_tree_key_shape(tree)?;

    if front_rank == rank {
        let coupled = coupled_or_vacuum(rule, tree);
        let trace_tree = FusionTreeKey::new(
            [coupled],
            Some(coupled),
            [false],
            Vec::<SectorId>::new(),
            Vec::<SectorId>::new(),
        );
        return Ok((tree.clone(), trace_tree));
    }

    if front_rank == 1 {
        let first = tree.uncoupled()[0];
        let front_tree = FusionTreeKey::new(
            [first],
            Some(first),
            [tree.is_dual()[0]],
            Vec::<SectorId>::new(),
            Vec::<SectorId>::new(),
        );
        let mut tail_is_dual = tree.is_dual().to_vec();
        tail_is_dual[0] = false;
        let tail_tree = FusionTreeKey::new(
            tree.uncoupled().to_vec(),
            tree.coupled(),
            tail_is_dual,
            tree.innerlines().to_vec(),
            tree.vertices().to_vec(),
        );
        return Ok((front_tree, tail_tree));
    }

    if front_rank == 0 {
        if rank == 0 {
            return Err(CoreError::MalformedFusionTree {
                message: "split at zero requires a non-empty source fusion tree",
            });
        }
        let unit = rule.vacuum();
        let front_tree = FusionTreeKey::new(
            Vec::<SectorId>::new(),
            Some(unit),
            Vec::<bool>::new(),
            Vec::<SectorId>::new(),
            Vec::<SectorId>::new(),
        );
        let mut tail_uncoupled = Vec::with_capacity(rank + 1);
        tail_uncoupled.push(unit);
        tail_uncoupled.extend_from_slice(tree.uncoupled());
        let mut tail_is_dual = Vec::with_capacity(rank + 1);
        tail_is_dual.push(false);
        tail_is_dual.extend_from_slice(tree.is_dual());
        let mut tail_innerlines = Vec::with_capacity(rank.saturating_sub(1));
        if rank >= 2 {
            tail_innerlines.push(tree.uncoupled()[0]);
            tail_innerlines.extend_from_slice(tree.innerlines());
        }
        let mut tail_vertices = Vec::with_capacity(rank);
        tail_vertices.push(SectorId::new(1));
        tail_vertices.extend_from_slice(tree.vertices());
        let tail_tree = FusionTreeKey::new(
            tail_uncoupled,
            tree.coupled(),
            tail_is_dual,
            tail_innerlines,
            tail_vertices,
        );
        return Ok((front_tree, tail_tree));
    }

    let intermediate =
        *tree
            .innerlines()
            .get(front_rank - 2)
            .ok_or(CoreError::MalformedFusionTree {
                message: "split requires the intermediate innerline",
            })?;
    let front_tree = FusionTreeKey::new(
        tree.uncoupled()[..front_rank].to_vec(),
        Some(intermediate),
        tree.is_dual()[..front_rank].to_vec(),
        tree.innerlines()[..front_rank.saturating_sub(2)].to_vec(),
        tree.vertices()[..front_rank - 1].to_vec(),
    );

    let mut tail_uncoupled = Vec::with_capacity(rank - front_rank + 1);
    tail_uncoupled.push(intermediate);
    tail_uncoupled.extend_from_slice(&tree.uncoupled()[front_rank..]);
    let mut tail_is_dual = Vec::with_capacity(rank - front_rank + 1);
    tail_is_dual.push(false);
    tail_is_dual.extend_from_slice(&tree.is_dual()[front_rank..]);
    let tail_tree = FusionTreeKey::new(
        tail_uncoupled,
        tree.coupled(),
        tail_is_dual,
        tree.innerlines()[front_rank - 1..].to_vec(),
        tree.vertices()[front_rank - 1..].to_vec(),
    );
    Ok((front_tree, tail_tree))
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

pub fn unique_artin_braid_first<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    unique_artin_braid_at(rule, tree, 0)
}

pub fn unique_artin_braid_at<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    unique_artin_braid_at_with_inverse(rule, tree, index, false)
}

pub fn unique_braid_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    levels: &[usize],
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rank = tree.uncoupled().len();
    if levels.len() != rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: levels.len(),
        });
    }
    let swaps = permutation_to_adjacent_swaps(permutation, rank)?;
    let mut current = tree.clone();
    let mut coefficient = rule.scalar_one();
    let mut current_levels = levels.to_vec();
    for swap in swaps {
        let inverse = current_levels[swap] > current_levels[swap + 1];
        let (next, step_coefficient) =
            unique_artin_braid_at_with_inverse(rule, &current, swap, inverse)?;
        coefficient = coefficient * step_coefficient;
        current_levels.swap(swap, swap + 1);
        current = next;
    }
    Ok((current, coefficient))
}

pub fn unique_permute_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    let levels = (0..tree.uncoupled().len()).collect::<Vec<_>>();
    unique_braid_tree(rule, tree, permutation, &levels)
}

pub fn multiplicity_free_artin_braid_at<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    Ok(
        multiplicity_free_artin_braid_at_with_inverse(rule, tree, index, false)?
            .into_iter()
            .collect(),
    )
}

pub fn multiplicity_free_braid_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    levels: &[usize],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: rule.fusion_style(),
        });
    }
    let rank = tree.uncoupled().len();
    if levels.len() != rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: levels.len(),
        });
    }
    let swaps = permutation_to_adjacent_swaps(permutation, rank)?;
    let mut current = vec![(tree.clone(), rule.scalar_one())];
    let mut current_levels = levels.to_vec();
    for swap in swaps {
        let inverse = current_levels[swap] > current_levels[swap + 1];
        let mut next_terms = FusionTermAccumulator::new();
        for (tree, coefficient) in current {
            for (next_tree, step_coefficient) in
                multiplicity_free_artin_braid_at_with_inverse(rule, &tree, swap, inverse)?
            {
                next_terms.push(next_tree, coefficient.clone() * step_coefficient);
            }
        }
        current_levels.swap(swap, swap + 1);
        current = next_terms.into_vec();
    }
    Ok(current)
}

pub fn multiplicity_free_permute_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    let levels = (0..tree.uncoupled().len()).collect::<Vec<_>>();
    multiplicity_free_braid_tree(rule, tree, permutation, &levels)
}

pub fn multiplicity_free_repartition_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let total_rank =
        tree_pair.codomain_tree().uncoupled().len() + tree_pair.domain_tree().uncoupled().len();
    if target_codomain_rank > total_rank {
        return Err(CoreError::DimensionMismatch {
            expected: total_rank,
            actual: target_codomain_rank,
        });
    }

    let mut current = vec![(tree_pair.clone(), rule.scalar_one())];
    let mut current_codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    while current_codomain_rank < target_codomain_rank {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_bendleft_tree_pair(rule, key)
        })?;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_bendright_tree_pair(rule, key)
        })?;
        current_codomain_rank -= 1;
    }
    Ok(current)
}

pub fn multiplicity_free_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    if codomain_levels.len() != codomain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: codomain_rank,
            actual: codomain_levels.len(),
        });
    }
    if domain_levels.len() != domain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: domain_rank,
            actual: domain_levels.len(),
        });
    }

    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    let mut levels = Vec::with_capacity(codomain_rank + domain_rank);
    levels.extend_from_slice(codomain_levels);
    levels.extend(domain_levels.iter().rev().copied());

    let all_rank = codomain_rank + domain_rank;
    let mut current = multiplicity_free_repartition_tree_pair(rule, tree_pair, all_rank)?;
    current = compose_tree_pair_terms(rule, current, |rule, key| {
        multiplicity_free_braid_tree(rule, key.codomain_tree(), &permutation, &levels).map(
            |terms| {
                terms
                    .into_iter()
                    .map(|(codomain_tree, coefficient)| {
                        (
                            FusionTreeBlockKey::pair(codomain_tree, key.domain_tree().clone()),
                            coefficient,
                        )
                    })
                    .collect::<Vec<_>>()
            },
        )
    })?;
    multiplicity_free_repartition_terms(rule, current, codomain_permutation.len())
}

pub fn multiplicity_free_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    multiplicity_free_braid_tree_pair(
        rule,
        tree_pair,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

pub fn multiplicity_free_transpose_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    if !is_cyclic_permutation(&permutation) {
        return Err(CoreError::InvalidPermutation {
            permutation,
            rank: codomain_rank + domain_rank,
        });
    }

    let mut position = match permutation.iter().position(|&axis| axis == 0) {
        Some(position) => position,
        None => return Ok(vec![(tree_pair.clone(), rule.scalar_one())]),
    };
    let mut current =
        multiplicity_free_repartition_tree_pair(rule, tree_pair, codomain_permutation.len())?;
    let total_rank = codomain_rank + domain_rank;
    if total_rank == 0 || position == 0 {
        return Ok(current);
    }

    let half_rank = total_rank >> 1;
    while position > 0 && position < half_rank {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_cycle_anticlockwise_tree_pair(rule, key)
        })?;
        position -= 1;
    }
    while position >= half_rank && position > 0 {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_cycle_clockwise_tree_pair(rule, key)
        })?;
        position = (position + 1) % total_rank;
    }

    Ok(current)
}

enum FusionTermAccumulator<K, S> {
    Empty,
    Singleton(K, S),
    Map {
        order: Vec<K>,
        coefficients: FxHashMap<K, S>,
    },
}

impl<K, S> FusionTermAccumulator<K, S>
where
    K: Clone + Eq + Hash,
    S: Clone + Add<Output = S>,
{
    fn new() -> Self {
        Self::Empty
    }

    fn push(&mut self, key: K, coefficient: S) {
        match self {
            Self::Empty => {
                *self = Self::Singleton(key, coefficient);
            }
            Self::Singleton(existing_key, existing) if existing_key == &key => {
                *existing = existing.clone() + coefficient;
            }
            Self::Singleton(_, _) => {
                let previous = std::mem::replace(self, Self::Empty);
                let Self::Singleton(existing_key, existing_coefficient) = previous else {
                    unreachable!("matched singleton state");
                };
                let mut order = Vec::with_capacity(2);
                let mut coefficients = FxHashMap::default();
                Self::push_map_term(
                    &mut order,
                    &mut coefficients,
                    existing_key,
                    existing_coefficient,
                );
                Self::push_map_term(&mut order, &mut coefficients, key, coefficient);
                *self = Self::Map {
                    order,
                    coefficients,
                };
            }
            Self::Map {
                order,
                coefficients,
            } => {
                Self::push_map_term(order, coefficients, key, coefficient);
            }
        }
    }

    fn push_map_term(
        order: &mut Vec<K>,
        coefficients: &mut FxHashMap<K, S>,
        key: K,
        coefficient: S,
    ) {
        match coefficients.entry(key) {
            Entry::Occupied(mut entry) => {
                let existing = entry.get_mut();
                *existing = existing.clone() + coefficient;
            }
            Entry::Vacant(entry) => {
                order.push(entry.key().clone());
                entry.insert(coefficient);
            }
        }
    }

    fn into_vec(self) -> Vec<(K, S)> {
        match self {
            Self::Empty => Vec::new(),
            Self::Singleton(key, coefficient) => vec![(key, coefficient)],
            Self::Map {
                order,
                mut coefficients,
            } => {
                let mut terms = Vec::with_capacity(order.len());
                for key in order {
                    let coefficient = coefficients
                        .remove(&key)
                        .expect("accumulator order only contains inserted keys");
                    terms.push((key, coefficient));
                }
                terms
            }
        }
    }
}

fn compose_tree_pair_terms<R, F, I>(
    rule: &R,
    terms: Vec<(FusionTreeBlockKey, R::Scalar)>,
    mut transform: F,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    F: FnMut(&R, &FusionTreeBlockKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreeBlockKey, R::Scalar)>,
{
    let mut output = FusionTermAccumulator::new();
    for (key, coefficient) in terms {
        for (next_key, next_coefficient) in transform(rule, &key)? {
            output.push(next_key, coefficient.clone() * next_coefficient);
        }
    }
    Ok(output.into_vec())
}

/// Batched analog of [`compose_tree_pair_terms`]: apply `transform` to every
/// tree-pair of a whole block at once, threading a coefficient *matrix* (a
/// sparse column per original source) instead of re-running the per-source
/// term list. `columns[i]` maps `src index -> coefficient` for `basis[i]`.
///
/// This is the TensorKit 0.17 `artin_braid`/`fsbraid` batching: the elementary
/// step (bend / Artin braid) is walked ONCE for the block and its coefficients
/// are spread across all source columns, so intermediate allocation is
/// O(steps) rather than O(steps × sources) — the term-list style TeNeT used
/// (equivalent to TensorKit ≤0.16's per-tree `FusionTreeDict`) allocated a
/// fresh accumulator and cloned keys per source per step.
/// Dense coefficient matrix (TK's `Matrix{E}`): rows are destination basis
/// trees, columns the original sources. Stored row-major in ONE flat
/// allocation that grows amortized as rows are added, instead of a
/// `Vec<Vec<_>>` that heap-allocs a fresh column per destination tree (the
/// batched braid over a whole block adds hundreds of thousands of rows across
/// its bend/braid steps, so the per-row allocation dominated the cold
/// recoupling build).
struct DenseColumns<S> {
    data: Vec<Option<S>>,
    num_src: usize,
    num_rows: usize,
}

impl<S: Clone> DenseColumns<S> {
    fn with_capacity(num_src: usize, rows_hint: usize) -> Self {
        Self {
            data: Vec::with_capacity(rows_hint.saturating_mul(num_src)),
            num_src,
            num_rows: 0,
        }
    }

    /// Append a new all-empty row, returning its index.
    fn push_empty_row(&mut self) -> usize {
        let row = self.num_rows;
        self.data.resize_with(self.data.len() + self.num_src, || None);
        self.num_rows += 1;
        row
    }

    #[inline]
    fn row(&self, row: usize) -> &[Option<S>] {
        let start = row * self.num_src;
        &self.data[start..start + self.num_src]
    }

    #[inline]
    fn row_mut(&mut self, row: usize) -> &mut [Option<S>] {
        let start = row * self.num_src;
        &mut self.data[start..start + self.num_src]
    }
}

fn compose_block_terms<R, F, I>(
    rule: &R,
    basis: &[FusionTreeBlockKey],
    columns: &DenseColumns<R::Scalar>,
    mut transform: F,
) -> Result<(Vec<FusionTreeBlockKey>, DenseColumns<R::Scalar>), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    F: FnMut(&R, &FusionTreeBlockKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreeBlockKey, R::Scalar)>,
{
    let num_src = columns.num_src;
    let mut index: FxHashMap<FusionTreeBlockKey, usize> = FxHashMap::default();
    let mut next_basis: Vec<FusionTreeBlockKey> = Vec::new();
    let mut next_columns: DenseColumns<R::Scalar> = DenseColumns::with_capacity(num_src, basis.len());
    for (source_row, source_key) in basis.iter().enumerate() {
        for (dst_key, step_coefficient) in transform(rule, source_key)? {
            let row = match index.get(&dst_key) {
                Some(&row) => row,
                None => {
                    let row = next_columns.push_empty_row();
                    index.insert(dst_key.clone(), row);
                    next_basis.push(dst_key);
                    row
                }
            };
            // dst_column[src] += step_coefficient * source_column[src] for each
            // source that reaches this basis tree. Source and destination live
            // in different matrices, so the borrows don't overlap.
            let source_column = columns.row(source_row);
            let dst_column = next_columns.row_mut(row);
            for (src, source_coefficient) in source_column.iter().enumerate() {
                let Some(source_coefficient) = source_coefficient else {
                    continue;
                };
                let contribution = step_coefficient.clone() * source_coefficient.clone();
                dst_column[src] = Some(match dst_column[src].take() {
                    Some(existing) => existing + contribution,
                    None => contribution,
                });
            }
        }
    }
    Ok((next_basis, next_columns))
}

/// Batched [`multiplicity_free_braid_tree_pair`] over every source tree-pair of
/// a block (all sharing the same uncoupled sectors / duality). Returns, per
/// source (in `src_keys` order), its `(destination tree-pair, coefficient)`
/// rows — identical content to calling the per-source function on each, but the
/// bend/braid step structure is walked once for the block.
///
/// The floating-point *summation order* of coefficients that reach a
/// destination by several paths differs from the per-source accumulator, so
/// results agree with the per-source version to double-precision rounding, not
/// necessarily bit-for-bit.
pub fn multiplicity_free_braid_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreeBlockKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<Vec<(FusionTreeBlockKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if src_keys.is_empty() {
        return Ok(Vec::new());
    }
    let codomain_rank = src_keys[0].codomain_tree().uncoupled().len();
    let domain_rank = src_keys[0].domain_tree().uncoupled().len();
    if codomain_levels.len() != codomain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: codomain_rank,
            actual: codomain_levels.len(),
        });
    }
    if domain_levels.len() != domain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: domain_rank,
            actual: domain_levels.len(),
        });
    }

    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    let mut levels = Vec::with_capacity(codomain_rank + domain_rank);
    levels.extend_from_slice(codomain_levels);
    levels.extend(domain_levels.iter().rev().copied());
    let all_rank = codomain_rank + domain_rank;

    // Identity matrix: source `i` starts as its own basis tree with coeff one.
    let num_src = src_keys.len();
    let mut basis = src_keys.to_vec();
    let mut columns: DenseColumns<R::Scalar> = DenseColumns::with_capacity(num_src, num_src);
    for i in 0..num_src {
        let row = columns.push_empty_row();
        columns.row_mut(row)[i] = Some(rule.scalar_one());
    }

    // Step A: repartition everything into the codomain (bendleft chain).
    let mut current_codomain_rank = codomain_rank;
    while current_codomain_rank < all_rank {
        let (next_basis, next_columns) =
            compose_block_terms(rule, &basis, &columns, |rule, key| {
                multiplicity_free_bendleft_tree_pair(rule, key)
            })?;
        basis = next_basis;
        columns = next_columns;
        current_codomain_rank += 1;
    }

    // Step B: braid the (now all-codomain) tree ONE adjacent swap at a time,
    // each swap batched across the whole block. This replaces the per-source
    // inner braid (`multiplicity_free_braid_tree`, whose `FusionTermAccumulator`
    // and elementary-swap term lists ran once per source tree) with the shared
    // block matrix walk — the TensorKit 0.17 `artin_braid`-on-a-block scheme.
    let swaps = permutation_to_adjacent_swaps(&permutation, all_rank)?;
    let mut current_levels = levels.clone();
    for swap in swaps {
        let inverse = current_levels[swap] > current_levels[swap + 1];
        let (next_basis, next_columns) =
            compose_block_terms(rule, &basis, &columns, |rule, key| {
                let domain = key.domain_tree().clone();
                Ok(multiplicity_free_artin_braid_at_with_inverse(
                    rule,
                    key.codomain_tree(),
                    swap,
                    inverse,
                )?
                .into_iter()
                .map(move |(codomain_tree, coefficient)| {
                    (
                        FusionTreeBlockKey::pair(codomain_tree, domain.clone()),
                        coefficient,
                    )
                }))
            })?;
        basis = next_basis;
        columns = next_columns;
        current_levels.swap(swap, swap + 1);
    }

    // Step C: repartition back to the requested codomain rank.
    let target_codomain_rank = codomain_permutation.len();
    while current_codomain_rank > target_codomain_rank {
        let (next_basis, next_columns) =
            compose_block_terms(rule, &basis, &columns, |rule, key| {
                multiplicity_free_bendright_tree_pair(rule, key)
            })?;
        basis = next_basis;
        columns = next_columns;
        current_codomain_rank -= 1;
    }
    while current_codomain_rank < target_codomain_rank {
        let (next_basis, next_columns) =
            compose_block_terms(rule, &basis, &columns, |rule, key| {
                multiplicity_free_bendleft_tree_pair(rule, key)
            })?;
        basis = next_basis;
        columns = next_columns;
        current_codomain_rank += 1;
    }

    // Scatter the dense matrix back into per-source row lists. Columns are
    // indexed by source, so iterating in source order needs no sort.
    let mut rows_per_source: Vec<Vec<(FusionTreeBlockKey, R::Scalar)>> = vec![Vec::new(); num_src];
    for (dst_row, dst_key) in basis.iter().enumerate() {
        for (src, coefficient) in columns.row(dst_row).iter().enumerate() {
            if let Some(coefficient) = coefficient {
                rows_per_source[src].push((dst_key.clone(), coefficient.clone()));
            }
        }
    }
    Ok(rows_per_source)
}

/// Batched [`multiplicity_free_permute_tree_pair`] over a block: symmetric
/// braiding with the trivial level ordering.
pub fn multiplicity_free_permute_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreeBlockKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<Vec<(FusionTreeBlockKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    if src_keys.is_empty() {
        return Ok(Vec::new());
    }
    let codomain_rank = src_keys[0].codomain_tree().uncoupled().len();
    let domain_rank = src_keys[0].domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    multiplicity_free_braid_tree_pair_block(
        rule,
        src_keys,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

fn multiplicity_free_repartition_terms<R>(
    rule: &R,
    terms: Vec<(FusionTreeBlockKey, R::Scalar)>,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let mut current = terms;
    let Some((first_key, _)) = current.first() else {
        return Ok(current);
    };
    let total_rank =
        first_key.codomain_tree().uncoupled().len() + first_key.domain_tree().uncoupled().len();
    if target_codomain_rank > total_rank {
        return Err(CoreError::DimensionMismatch {
            expected: total_rank,
            actual: target_codomain_rank,
        });
    }
    let mut current_codomain_rank = first_key.codomain_tree().uncoupled().len();
    while current_codomain_rank < target_codomain_rank {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_bendleft_tree_pair(rule, key)
        })?;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_bendright_tree_pair(rule, key)
        })?;
        current_codomain_rank -= 1;
    }
    Ok(current)
}

pub fn unique_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    if codomain_levels.len() != codomain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: codomain_rank,
            actual: codomain_levels.len(),
        });
    }
    if domain_levels.len() != domain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: domain_rank,
            actual: domain_levels.len(),
        });
    }

    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    let mut levels = Vec::with_capacity(codomain_rank + domain_rank);
    levels.extend_from_slice(codomain_levels);
    levels.extend(domain_levels.iter().rev().copied());

    let (all_codomain_pair, repartition_to_all_coeff) =
        unique_repartition_tree_pair(rule, tree_pair, codomain_rank + domain_rank)?;
    let (braided_codomain_tree, braid_coeff) = unique_braid_tree(
        rule,
        all_codomain_pair.codomain_tree(),
        &permutation,
        &levels,
    )?;
    let braided_pair = FusionTreeBlockKey::pair(
        braided_codomain_tree,
        all_codomain_pair.domain_tree().clone(),
    );
    let (dst_pair, repartition_back_coeff) =
        unique_repartition_tree_pair(rule, &braided_pair, codomain_permutation.len())?;

    Ok((
        dst_pair,
        repartition_to_all_coeff * braid_coeff * repartition_back_coeff,
    ))
}

pub fn unique_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    unique_braid_tree_pair(
        rule,
        tree_pair,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

pub fn unique_transpose_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    if !is_cyclic_permutation(&permutation) {
        return Err(CoreError::InvalidPermutation {
            permutation,
            rank: codomain_rank + domain_rank,
        });
    }

    let mut position = match permutation.iter().position(|&axis| axis == 0) {
        Some(position) => position,
        None => return Ok((tree_pair.clone(), rule.scalar_one())),
    };
    let mut current = unique_repartition_tree_pair(rule, tree_pair, codomain_permutation.len())?;
    let total_rank = codomain_rank + domain_rank;
    if total_rank == 0 || position == 0 {
        return Ok(current);
    }

    let half_rank = total_rank >> 1;
    while position > 0 && position < half_rank {
        let (next, coefficient) = unique_cycle_anticlockwise_tree_pair(rule, &current.0)?;
        current = (next, current.1 * coefficient);
        position -= 1;
    }
    while position >= half_rank && position > 0 {
        let (next, coefficient) = unique_cycle_clockwise_tree_pair(rule, &current.0)?;
        current = (next, current.1 * coefficient);
        position = (position + 1) % total_rank;
    }

    Ok(current)
}

pub fn unique_repartition_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    target_codomain_rank: usize,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: rule.fusion_style(),
        });
    }

    let total_rank =
        tree_pair.codomain_tree().uncoupled().len() + tree_pair.domain_tree().uncoupled().len();
    if target_codomain_rank > total_rank {
        return Err(CoreError::DimensionMismatch {
            expected: total_rank,
            actual: target_codomain_rank,
        });
    }

    let mut current = tree_pair.clone();
    let mut current_codomain_rank = current.codomain_tree().uncoupled().len();
    let mut coefficient = rule.scalar_one();
    while current_codomain_rank < target_codomain_rank {
        let (next, step_coefficient) = unique_bendleft_tree_pair(rule, &current)?;
        coefficient = coefficient * step_coefficient;
        current = next;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        let (next, step_coefficient) = unique_bendright_tree_pair(rule, &current)?;
        coefficient = coefficient * step_coefficient;
        current = next;
        current_codomain_rank -= 1;
    }
    Ok((current, coefficient))
}

pub fn linearize_tree_pair_permutation(
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_rank: usize,
    domain_rank: usize,
) -> Result<Vec<usize>, CoreError> {
    let total_rank = codomain_rank + domain_rank;
    let mut original_permutation =
        Vec::with_capacity(codomain_permutation.len() + domain_permutation.len());
    original_permutation.extend_from_slice(codomain_permutation);
    original_permutation.extend_from_slice(domain_permutation);
    validate_permutation(&original_permutation, total_rank)?;

    let mut linearized = Vec::with_capacity(total_rank);
    linearized.extend(
        codomain_permutation
            .iter()
            .map(|&axis| linearize_tree_pair_axis(axis, codomain_rank, domain_rank)),
    );
    linearized.extend(
        domain_permutation
            .iter()
            .rev()
            .map(|&axis| linearize_tree_pair_axis(axis, codomain_rank, domain_rank)),
    );
    validate_permutation(&linearized, total_rank)?;
    Ok(linearized)
}

fn unique_artin_braid_at_with_inverse<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
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
    let mut uncoupled = tree.uncoupled().to_vec();
    uncoupled.swap(index, index + 1);
    let mut is_dual = tree.is_dual().to_vec();
    is_dual.swap(index, index + 1);
    let mut innerlines = tree.innerlines().to_vec();
    let mut vertices = tree.vertices().to_vec();

    if left == rule.vacuum() || right == rule.vacuum() {
        if index > 0 {
            let inner_source = if left == rule.vacuum() {
                inner_extended_sector(tree, index + 1)?
            } else {
                inner_extended_sector(tree, index - 1)?
            };
            *innerlines
                .get_mut(index - 1)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires an innerline",
                })? = inner_source;
            if vertices.len() <= index {
                return Err(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires adjacent vertices",
                });
            }
            vertices.swap(index - 1, index);
        }

        let braided = FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices);
        return Ok((braided, rule.scalar_one()));
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
            tree.coupled().ok_or(CoreError::MalformedFusionTree {
                message: "first braid of a rank 2 tree requires a coupled sector",
            })?
        };

        let braided = FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices);
        let coefficient = if inverse {
            rule.scalar_conj(rule.r_symbol_scalar(right, left, coupled))
        } else {
            rule.r_symbol_scalar(left, right, coupled)
        };
        return Ok((braided, coefficient));
    }

    let a = inner_extended_sector(tree, index - 1)?;
    let b = left;
    let c = inner_extended_sector(tree, index)?;
    let d = right;
    let e = inner_extended_sector(tree, index + 1)?;
    let c_prime = only_fusion_channel(rule, a, d)?;
    *innerlines
        .get_mut(index - 1)
        .ok_or(CoreError::MalformedFusionTree {
            message: "non-first braid requires an innerline to update",
        })? = c_prime;
    let braided = FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices);
    let f_symbol = rule.f_symbol_scalar(d, a, b, e, c_prime, c);
    let coefficient = if inverse {
        let left = rule.r_symbol_scalar(d, c, e);
        let right = rule.r_symbol_scalar(d, a, c_prime);
        rule.scalar_conj(left * f_symbol) * right
    } else {
        let left = rule.r_symbol_scalar(c, d, e);
        let right = rule.r_symbol_scalar(a, d, c_prime);
        left * rule.scalar_conj(f_symbol * right)
    };
    Ok((braided, coefficient))
}

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
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: rule.fusion_style(),
        });
    }

    let rank = tree.uncoupled().len();
    if index + 1 >= rank {
        return Err(CoreError::InvalidBraidIndex { index, rank });
    }

    let left = tree.uncoupled()[index];
    let right = tree.uncoupled()[index + 1];
    // Collect into the inline `SmallVec` types (stack-resident for ≤8 legs)
    // rather than heap `Vec`s: this is on the per-swap braid hot path.
    let mut uncoupled: SectorVec = tree.uncoupled().iter().copied().collect();
    uncoupled.swap(index, index + 1);
    let mut is_dual: DualVec = tree.is_dual().iter().copied().collect();
    is_dual.swap(index, index + 1);

    if left == rule.vacuum() || right == rule.vacuum() {
        let mut innerlines = tree.innerlines().to_vec();
        let mut vertices = tree.vertices().to_vec();
        if index > 0 {
            let inner_source = if left == rule.vacuum() {
                inner_extended_sector(tree, index + 1)?
            } else {
                inner_extended_sector(tree, index - 1)?
            };
            *innerlines
                .get_mut(index - 1)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires an innerline",
                })? = inner_source;
            if vertices.len() <= index {
                return Err(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires adjacent vertices",
                });
            }
            vertices.swap(index - 1, index);
        }
        let mut out = SmallVec::new();
        out.push((
            FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices),
            rule.scalar_one(),
        ));
        return Ok(out);
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
            tree.coupled().ok_or(CoreError::MalformedFusionTree {
                message: "first braid of a rank 2 tree requires a coupled sector",
            })?
        };
        let coefficient = if inverse {
            rule.scalar_conj(rule.r_symbol_scalar(right, left, coupled))
        } else {
            rule.r_symbol_scalar(left, right, coupled)
        };
        let mut out = SmallVec::new();
        out.push((
            FusionTreeKey::new(
                uncoupled,
                tree.coupled(),
                is_dual,
                tree.innerlines().iter().copied(),
                tree.vertices().iter().copied(),
            ),
            coefficient,
        ));
        return Ok(out);
    }

    let a = inner_extended_sector(tree, index - 1)?;
    let b = left;
    let c = inner_extended_sector(tree, index)?;
    let d = right;
    let e = inner_extended_sector(tree, index + 1)?;
    let mut terms: SmallVec<[(FusionTreeKey, R::Scalar); 2]> = SmallVec::new();
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
        let braided = FusionTreeKey::new(
            uncoupled.clone(),
            tree.coupled(),
            is_dual.clone(),
            innerlines,
            tree.vertices().iter().copied(),
        );
        let f_symbol = rule.f_symbol_scalar(d, a, b, e, c_prime, c);
        let coefficient = if inverse {
            let left = rule.r_symbol_scalar(d, c, e);
            let right = rule.r_symbol_scalar(d, a, c_prime);
            rule.scalar_conj(left * f_symbol) * right
        } else {
            let left = rule.r_symbol_scalar(c, d, e);
            let right = rule.r_symbol_scalar(a, d, c_prime);
            left * rule.scalar_conj(f_symbol * right)
        };
        terms.push((braided, coefficient));
    }
    Ok(terms)
}

fn multiplicity_free_bendright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<SmallVec<[(FusionTreeBlockKey, R::Scalar); 1]>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let codomain = tree_pair.codomain_tree();
    let domain = tree_pair.domain_tree();
    let codomain_rank = codomain.uncoupled().len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "bendright requires at least one codomain leg",
        });
    }

    let coupled = coupled_or_vacuum(rule, codomain);
    if !domain.uncoupled().is_empty() {
        let domain_coupled = coupled_or_vacuum(rule, domain);
        if domain_coupled != coupled {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion tree pair requires matching coupled sectors",
            });
        }
    }

    let left_coupled = match codomain_rank {
        1 => rule.vacuum(),
        2 => codomain.uncoupled()[0],
        _ => codomain
            .innerlines()
            .last()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "bendright requires the last codomain innerline",
            })?,
    };
    let bent_sector = codomain.uncoupled()[codomain_rank - 1];
    let bent_is_dual = codomain.is_dual().get(codomain_rank - 1).copied().ok_or(
        CoreError::MalformedFusionTree {
            message: "codomain tree is missing a duality flag",
        },
    )?;

    // Build the new trees straight from slice iterators: `FusionTreeKey::new`
    // collects into inline `SmallVec`, so passing iterators (not `.to_vec()`)
    // keeps small trees stack-resident — the intermediate heap `Vec`s here were
    // a large share of the cold recoupling-compile malloc traffic.
    let cod_inner = codomain.innerlines();
    let new_codomain_innerlines: &[SectorId] = if codomain_rank > 2 {
        &cod_inner[..cod_inner.len() - 1]
    } else {
        &[]
    };
    let cod_vertices = codomain.vertices();
    let new_codomain_vertices: &[SectorId] = if codomain_rank > 1 {
        &cod_vertices[..cod_vertices.len() - 1]
    } else {
        &[]
    };
    let new_codomain = FusionTreeKey::new(
        codomain.uncoupled()[..codomain_rank - 1].iter().copied(),
        Some(left_coupled),
        codomain.is_dual()[..codomain_rank - 1].iter().copied(),
        new_codomain_innerlines.iter().copied(),
        new_codomain_vertices.iter().copied(),
    );

    let domain_rank = domain.uncoupled().len();
    let new_domain = FusionTreeKey::new(
        domain
            .uncoupled()
            .iter()
            .copied()
            .chain(std::iter::once(rule.dual(bent_sector))),
        Some(left_coupled),
        domain
            .is_dual()
            .iter()
            .copied()
            .chain(std::iter::once(!bent_is_dual)),
        domain
            .innerlines()
            .iter()
            .copied()
            .chain((domain_rank > 1).then_some(coupled)),
        domain
            .vertices()
            .iter()
            .copied()
            .chain((domain_rank > 0).then_some(SectorId::new(1))),
    );

    let mut coefficient = rule.sqrt_dim_scalar(coupled)
        * rule.inv_sqrt_dim_scalar(left_coupled)
        * rule.b_symbol_scalar(left_coupled, bent_sector, coupled);
    if bent_is_dual {
        coefficient = coefficient
            * rule.scalar_conj(rule.frobenius_schur_phase_scalar(rule.dual(bent_sector)));
    }
    let mut out = SmallVec::new();
    out.push((
        FusionTreeBlockKey::pair(new_codomain, new_domain),
        coefficient,
    ));
    Ok(out)
}

fn multiplicity_free_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<SmallVec<[(FusionTreeBlockKey, R::Scalar); 1]>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let swapped = FusionTreeBlockKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(multiplicity_free_bendright_tree_pair(rule, &swapped)?
        .into_iter()
        .map(|(bent, coefficient)| {
            (
                FusionTreeBlockKey::pair(bent.domain_tree().clone(), bent.codomain_tree().clone()),
                rule.scalar_conj(coefficient),
            )
        })
        .collect())
}

fn multiplicity_free_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let codomain = tree_pair.codomain_tree();
    let codomain_rank = codomain.uncoupled().len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "foldright requires at least one codomain leg",
        });
    }
    let a = codomain.uncoupled()[0];
    let is_dual_a = codomain
        .is_dual()
        .first()
        .copied()
        .ok_or(CoreError::MalformedFusionTree {
            message: "codomain tree is missing the first duality flag",
        })?;
    let kappa = rule.frobenius_schur_phase_scalar(a);
    let c = coupled_or_vacuum(rule, codomain);

    let mut terms = FusionTermAccumulator::new();
    for (codomain_prime, coeff1) in multiplicity_free_multi_fmove_tree(rule, codomain)? {
        let b = coupled_or_vacuum(rule, &codomain_prime);
        let a_symbol = rule.a_symbol_scalar(a, b, c);
        let coeff0 = rule.sqrt_dim_scalar(c) * rule.inv_sqrt_dim_scalar(b);
        for (domain_prime, coeff2) in multiplicity_free_multi_fmove_inv_tree(
            rule,
            rule.dual(a),
            b,
            tree_pair.domain_tree(),
            !is_dual_a,
        )? {
            let mut coefficient =
                coeff0.clone() * rule.scalar_conj(coeff2) * a_symbol.clone() * coeff1.clone();
            if is_dual_a {
                coefficient = coefficient * kappa.clone();
            }
            terms.push(
                FusionTreeBlockKey::pair(codomain_prime.clone(), domain_prime),
                coefficient,
            );
        }
    }
    Ok(terms.into_vec())
}

fn multiplicity_free_foldleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let swapped = FusionTreeBlockKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(multiplicity_free_foldright_tree_pair(rule, &swapped)?
        .into_iter()
        .map(|(folded, coefficient)| {
            (
                FusionTreeBlockKey::pair(
                    folded.domain_tree().clone(),
                    folded.codomain_tree().clone(),
                ),
                rule.scalar_conj(coefficient),
            )
        })
        .collect())
}

fn multiplicity_free_cycle_clockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let first: Vec<_> = if tree_pair.codomain_tree().uncoupled().is_empty() {
        multiplicity_free_bendleft_tree_pair(rule, tree_pair)?
            .into_iter()
            .collect()
    } else {
        multiplicity_free_foldright_tree_pair(rule, tree_pair)?
    };
    if tree_pair.codomain_tree().uncoupled().is_empty() {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_foldright_tree_pair(rule, key)
        })
    } else {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_bendleft_tree_pair(rule, key)
        })
    }
}

fn multiplicity_free_cycle_anticlockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let first: Vec<_> = if tree_pair.domain_tree().uncoupled().is_empty() {
        multiplicity_free_bendright_tree_pair(rule, tree_pair)?
            .into_iter()
            .collect()
    } else {
        multiplicity_free_foldleft_tree_pair(rule, tree_pair)?
    };
    if tree_pair.domain_tree().uncoupled().is_empty() {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_foldleft_tree_pair(rule, key)
        })
    } else {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_bendright_tree_pair(rule, key)
        })
    }
}

fn multiplicity_free_multi_fmove_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let rank = tree.uncoupled().len();
    if rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "multi_Fmove requires at least one uncoupled sector",
        });
    }
    if rank == 1 {
        return Ok(vec![(
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                Some(rule.vacuum()),
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
            rule.scalar_one(),
        )]);
    }
    if rank == 2 {
        return Ok(vec![(
            FusionTreeKey::new(
                vec![tree.uncoupled()[1]],
                Some(tree.uncoupled()[1]),
                vec![tree.is_dual()[1]],
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
            rule.scalar_one(),
        )]);
    }

    let first = tree.uncoupled()[0];
    let coupled = coupled_or_vacuum(rule, tree);
    let tail_uncoupled = &tree.uncoupled()[1..];
    let tail_is_dual = &tree.is_dual()[1..];
    let mut terms = Vec::new();
    for tail_coupled in rule.fusion_channels(rule.dual(first), coupled) {
        let tail_effective = effective_sectors_for_uncoupled(rule, tail_uncoupled, tail_is_dual)?;
        for tail_tree in collect_fusion_trees_for_coupled(
            rule,
            tail_uncoupled,
            tail_is_dual,
            &tail_effective,
            tail_coupled,
        ) {
            if let Some(coefficient) =
                multiplicity_free_multi_associator_scalar(rule, tree, &tail_tree)?
            {
                terms.push((tail_tree, coefficient));
            }
        }
    }
    Ok(terms)
}

fn multiplicity_free_multi_fmove_inv_tree<R>(
    rule: &R,
    leading_sector: SectorId,
    coupled: SectorId,
    tree: &FusionTreeKey,
    leading_is_dual: bool,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let tree_coupled = coupled_or_vacuum(rule, tree);
    if rule.nsymbol(leading_sector, tree_coupled, coupled) == 0 {
        return Err(CoreError::SectorMismatch {
            expected: coupled,
            actual: tree_coupled,
        });
    }

    let mut uncoupled = Vec::with_capacity(tree.uncoupled().len() + 1);
    uncoupled.push(leading_sector);
    uncoupled.extend_from_slice(tree.uncoupled());
    let mut is_dual = Vec::with_capacity(tree.is_dual().len() + 1);
    is_dual.push(leading_is_dual);
    is_dual.extend_from_slice(tree.is_dual());
    let effective = effective_sectors_for_uncoupled(rule, &uncoupled, &is_dual)?;
    let candidates =
        collect_fusion_trees_for_coupled(rule, &uncoupled, &is_dual, &effective, coupled);

    let mut terms = Vec::new();
    for candidate in candidates {
        for (short_tree, coefficient) in multiplicity_free_multi_fmove_tree(rule, &candidate)? {
            if fusion_tree_keys_match_with_empty_vacuum(rule, &short_tree, tree) {
                terms.push((candidate.clone(), rule.scalar_conj(coefficient)));
            }
        }
    }
    Ok(terms)
}

fn multiplicity_free_multi_associator_scalar<R>(
    rule: &R,
    long: &FusionTreeKey,
    short: &FusionTreeKey,
) -> Result<Option<R::Scalar>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rank = long.uncoupled().len();
    if short.uncoupled().len() + 1 != rank {
        return Ok(None);
    }
    if long.uncoupled()[1..] != *short.uncoupled() || long.is_dual()[1..] != *short.is_dual() {
        return Ok(None);
    }

    let mut coefficient = rule.scalar_one();
    let first = long.uncoupled()[0];
    for tensor_kit_k in 2..rank {
        let right_sector = long.uncoupled()[tensor_kit_k];
        let (middle_left, middle_right) = fusion_tree_vertex_neighbors(long, tensor_kit_k)?;
        let (short_left, short_right) = fusion_tree_vertex_neighbors(short, tensor_kit_k - 1)?;
        coefficient = coefficient
            * rule.f_symbol_scalar(
                first,
                short_left,
                right_sector,
                middle_right,
                middle_left,
                short_right,
            );
    }
    Ok(Some(coefficient))
}

fn fusion_tree_vertex_neighbors(
    tree: &FusionTreeKey,
    leg_index: usize,
) -> Result<(SectorId, SectorId), CoreError> {
    if leg_index == 0 || leg_index >= tree.uncoupled().len() {
        return Err(CoreError::MalformedFusionTree {
            message: "vertex_info requires a non-first uncoupled leg",
        });
    }
    let left = if leg_index == 1 {
        tree.uncoupled()[0]
    } else {
        tree.innerlines()
            .get(leg_index - 2)
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "fusion tree is missing a left innerline",
            })?
    };
    let right = if leg_index + 1 == tree.uncoupled().len() {
        coupled_or_vacuum_for_tree(tree)?
    } else {
        tree.innerlines()
            .get(leg_index - 1)
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "fusion tree is missing a right innerline",
            })?
    };
    Ok((left, right))
}

fn coupled_or_vacuum<R>(rule: &R, tree: &FusionTreeKey) -> SectorId
where
    R: FusionRule,
{
    tree.coupled().unwrap_or_else(|| rule.vacuum())
}

fn coupled_or_vacuum_for_tree(tree: &FusionTreeKey) -> Result<SectorId, CoreError> {
    tree.coupled().ok_or(CoreError::MalformedFusionTree {
        message: "non-empty fusion tree requires a coupled sector",
    })
}

fn effective_sectors_for_uncoupled<R>(
    _rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
) -> Result<Vec<SectorId>, CoreError>
where
    R: FusionRule,
{
    if uncoupled.len() != is_dual.len() {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree sectors and duality flags must have matching length",
        });
    }
    Ok(uncoupled.to_vec())
}

fn fusion_tree_keys_match_with_empty_vacuum<R>(
    rule: &R,
    left: &FusionTreeKey,
    right: &FusionTreeKey,
) -> bool
where
    R: FusionRule,
{
    left.uncoupled() == right.uncoupled()
        && left.is_dual() == right.is_dual()
        && left.innerlines() == right.innerlines()
        && left.vertices() == right.vertices()
        && coupled_or_vacuum(rule, left) == coupled_or_vacuum(rule, right)
}

fn permutation_to_adjacent_swaps(
    permutation: &[usize],
    rank: usize,
) -> Result<Vec<usize>, CoreError> {
    if permutation.len() != rank {
        return Err(CoreError::InvalidPermutation {
            permutation: permutation.to_vec(),
            rank,
        });
    }
    let mut seen = vec![false; rank];
    for &axis in permutation {
        if axis >= rank || seen[axis] {
            return Err(CoreError::InvalidPermutation {
                permutation: permutation.to_vec(),
                rank,
            });
        }
        seen[axis] = true;
    }

    let mut work = permutation.to_vec();
    let mut swaps = Vec::new();
    for target in 0..rank.saturating_sub(1) {
        let source = work[target];
        for swap in (target..source).rev() {
            swaps.push(swap);
        }
        for item in work.iter_mut().take(rank).skip(target + 1) {
            if *item < source {
                *item += 1;
            }
        }
        work[target] = target;
    }
    Ok(swaps)
}

fn inner_extended_sector(tree: &FusionTreeKey, index: usize) -> Result<SectorId, CoreError> {
    let rank = tree.uncoupled().len();
    if index == 0 {
        return tree
            .uncoupled()
            .first()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "inner-extended tree requires at least one uncoupled sector",
            });
    }
    if index + 1 == rank {
        return tree.coupled().ok_or(CoreError::MalformedFusionTree {
            message: "inner-extended tree requires a coupled sector",
        });
    }
    tree.innerlines()
        .get(index - 1)
        .copied()
        .ok_or(CoreError::MalformedFusionTree {
            message: "inner-extended tree is missing an innerline",
        })
}

fn only_fusion_channel<R>(rule: &R, left: SectorId, right: SectorId) -> Result<SectorId, CoreError>
where
    R: FusionRule,
{
    let channels = rule.fusion_channels(left, right);
    match channels.as_slice() {
        [sector] => Ok(*sector),
        _ => Err(CoreError::FusionChannelCount {
            left,
            right,
            count: channels.len(),
        }),
    }
}

fn unique_bendright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
{
    let codomain = tree_pair.codomain_tree();
    let domain = tree_pair.domain_tree();
    let codomain_rank = codomain.uncoupled().len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "bendright requires at least one codomain leg",
        });
    }

    let coupled = codomain.coupled().ok_or(CoreError::MalformedFusionTree {
        message: "bendright requires a coupled sector on the codomain tree",
    })?;
    if !domain.uncoupled().is_empty() {
        match domain.coupled() {
            Some(domain_coupled) if domain_coupled == coupled => {}
            _ => {
                return Err(CoreError::MalformedFusionTree {
                    message: "fusion tree pair requires matching coupled sectors",
                });
            }
        }
    }

    let bent_sector = codomain.uncoupled()[codomain_rank - 1];
    let bent_is_dual = codomain.is_dual().get(codomain_rank - 1).copied().ok_or(
        CoreError::MalformedFusionTree {
            message: "codomain tree is missing a duality flag",
        },
    )?;
    let left_coupled = match codomain_rank {
        1 => rule.vacuum(),
        2 => codomain.uncoupled()[0],
        _ => codomain
            .innerlines()
            .last()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "bendright requires the last codomain innerline",
            })?,
    };

    let new_codomain_innerlines = if codomain_rank > 2 {
        let innerlines = codomain.innerlines();
        innerlines
            .get(..innerlines.len() - 1)
            .ok_or(CoreError::MalformedFusionTree {
                message: "codomain tree has malformed innerlines",
            })?
            .to_vec()
    } else {
        Vec::new()
    };
    let new_codomain_vertices = if codomain_rank > 1 {
        let vertices = codomain.vertices();
        vertices
            .get(..vertices.len() - 1)
            .ok_or(CoreError::MalformedFusionTree {
                message: "codomain tree has malformed vertices",
            })?
            .to_vec()
    } else {
        Vec::new()
    };
    let new_codomain = FusionTreeKey::new(
        codomain.uncoupled()[..codomain_rank - 1].to_vec(),
        Some(left_coupled),
        codomain.is_dual()[..codomain_rank - 1].to_vec(),
        new_codomain_innerlines,
        new_codomain_vertices,
    );

    let domain_rank = domain.uncoupled().len();
    let mut new_domain_uncoupled = domain.uncoupled().to_vec();
    new_domain_uncoupled.push(rule.dual(bent_sector));
    let mut new_domain_is_dual = domain.is_dual().to_vec();
    new_domain_is_dual.push(!bent_is_dual);
    let mut new_domain_innerlines = domain.innerlines().to_vec();
    if domain_rank > 1 {
        new_domain_innerlines.push(coupled);
    }
    let mut new_domain_vertices = domain.vertices().to_vec();
    if domain_rank > 0 {
        new_domain_vertices.push(SectorId::new(1));
    }
    let new_domain = FusionTreeKey::new(
        new_domain_uncoupled,
        Some(left_coupled),
        new_domain_is_dual,
        new_domain_innerlines,
        new_domain_vertices,
    );

    let coefficient = rule.bendright_scalar(left_coupled, bent_sector, coupled, bent_is_dual);
    Ok((
        FusionTreeBlockKey::pair(new_codomain, new_domain),
        coefficient,
    ))
}

fn unique_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
{
    let swapped = FusionTreeBlockKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    let (bent, coefficient) = unique_bendright_tree_pair(rule, &swapped)?;
    Ok((
        FusionTreeBlockKey::pair(bent.domain_tree().clone(), bent.codomain_tree().clone()),
        rule.scalar_conj(coefficient),
    ))
}

fn unique_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
{
    let codomain = tree_pair.codomain_tree();
    let codomain_rank = codomain.uncoupled().len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "foldright requires at least one codomain leg",
        });
    }
    let first = codomain.uncoupled()[0];
    let first_is_dual =
        codomain
            .is_dual()
            .first()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "codomain tree is missing the first duality flag",
            })?;
    let codomain_prime = unique_multi_fmove_tree(rule, codomain)?;
    let recoupled = codomain_prime
        .coupled()
        .ok_or(CoreError::MalformedFusionTree {
            message: "foldright recoupled codomain tree requires a coupled sector",
        })?;
    let domain_prime = unique_multi_fmove_inv_tree(
        rule,
        rule.dual(first),
        recoupled,
        tree_pair.domain_tree(),
        !first_is_dual,
    )?;
    let destination = FusionTreeBlockKey::pair(codomain_prime, domain_prime);
    let coefficient = rule.foldright_scalar(tree_pair, &destination);
    Ok((destination, coefficient))
}

fn unique_foldleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
{
    let swapped = FusionTreeBlockKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    let (folded, coefficient) = unique_foldright_tree_pair(rule, &swapped)?;
    Ok((
        FusionTreeBlockKey::pair(folded.domain_tree().clone(), folded.codomain_tree().clone()),
        rule.scalar_conj(coefficient),
    ))
}

fn unique_cycle_clockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let (tmp, first_coefficient) = if tree_pair.codomain_tree().uncoupled().is_empty() {
        unique_bendleft_tree_pair(rule, tree_pair)?
    } else {
        unique_foldright_tree_pair(rule, tree_pair)?
    };
    let (dst, second_coefficient) = if tree_pair.codomain_tree().uncoupled().is_empty() {
        unique_foldright_tree_pair(rule, &tmp)?
    } else {
        unique_bendleft_tree_pair(rule, &tmp)?
    };
    Ok((dst, first_coefficient * second_coefficient))
}

fn unique_cycle_anticlockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let (tmp, first_coefficient) = if tree_pair.domain_tree().uncoupled().is_empty() {
        unique_bendright_tree_pair(rule, tree_pair)?
    } else {
        unique_foldleft_tree_pair(rule, tree_pair)?
    };
    let (dst, second_coefficient) = if tree_pair.domain_tree().uncoupled().is_empty() {
        unique_foldleft_tree_pair(rule, &tmp)?
    } else {
        unique_bendright_tree_pair(rule, &tmp)?
    };
    Ok((dst, first_coefficient * second_coefficient))
}

fn unique_multi_fmove_tree<R>(rule: &R, tree: &FusionTreeKey) -> Result<FusionTreeKey, CoreError>
where
    R: MultiplicityFreeFusionRule,
{
    let first = tree
        .uncoupled()
        .first()
        .copied()
        .ok_or(CoreError::MalformedFusionTree {
            message: "multi_Fmove requires at least one uncoupled sector",
        })?;
    let coupled = tree.coupled().ok_or(CoreError::MalformedFusionTree {
        message: "multi_Fmove requires a coupled sector",
    })?;
    let recoupled = only_fusion_channel(rule, rule.dual(first), coupled)?;
    unique_standard_fusion_tree(
        rule,
        &tree.uncoupled()[1..],
        recoupled,
        &tree.is_dual()[1..],
    )
}

fn unique_multi_fmove_inv_tree<R>(
    rule: &R,
    leading_sector: SectorId,
    coupled: SectorId,
    tree: &FusionTreeKey,
    leading_is_dual: bool,
) -> Result<FusionTreeKey, CoreError>
where
    R: MultiplicityFreeFusionRule,
{
    let mut uncoupled = Vec::with_capacity(tree.uncoupled().len() + 1);
    uncoupled.push(leading_sector);
    uncoupled.extend_from_slice(tree.uncoupled());
    let mut is_dual = Vec::with_capacity(tree.is_dual().len() + 1);
    is_dual.push(leading_is_dual);
    is_dual.extend_from_slice(tree.is_dual());
    unique_standard_fusion_tree(rule, &uncoupled, coupled, &is_dual)
}

fn unique_standard_fusion_tree<R>(
    rule: &R,
    uncoupled: &[SectorId],
    coupled: SectorId,
    is_dual: &[bool],
) -> Result<FusionTreeKey, CoreError>
where
    R: MultiplicityFreeFusionRule,
{
    if uncoupled.len() != is_dual.len() {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree sectors and duality flags must have matching length",
        });
    }
    let effective = uncoupled.to_vec();
    let trees = collect_fusion_trees_for_coupled(rule, uncoupled, is_dual, &effective, coupled);
    match trees.as_slice() {
        [tree] => Ok(tree.clone()),
        _ => Err(CoreError::FusionChannelCount {
            left: coupled,
            right: coupled,
            count: trees.len(),
        }),
    }
}

fn linearize_tree_pair_axis(axis: usize, codomain_rank: usize, domain_rank: usize) -> usize {
    if axis < codomain_rank {
        axis
    } else {
        domain_rank + 2 * codomain_rank - 1 - axis
    }
}

fn validate_permutation(permutation: &[usize], rank: usize) -> Result<(), CoreError> {
    if permutation.len() != rank {
        return Err(CoreError::InvalidPermutation {
            permutation: permutation.to_vec(),
            rank,
        });
    }
    let mut seen = vec![false; rank];
    for &axis in permutation {
        if axis >= rank || seen[axis] {
            return Err(CoreError::InvalidPermutation {
                permutation: permutation.to_vec(),
                rank,
            });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn is_cyclic_permutation(permutation: &[usize]) -> bool {
    let rank = permutation.len();
    for index in 0..rank {
        if permutation[(index + 1) % rank] != (permutation[index] + 1) % rank {
            return false;
        }
    }
    true
}
