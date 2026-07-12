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

/// Generic-fusion (outer-multiplicity) sibling of
/// [`fusion_trees_by_coupled_for_space`]: emits multiplicity-aware fusion-tree
/// keys (one per vertex-label combination) via
/// [`collect_generic_fusion_trees_for_coupled`]. `R: FusionRule` (not
/// `MultiplicityFreeFusionRule`) so SU(3)/SO(N≥7)/Sp(N) rules can drive the
/// Space layer.
///
/// Escape semantics (Option A, refute/b3b-verify): the coupled candidates of
/// each leg tuple are classified by [`FusionRule::coupled_sector_fold`]. Trees
/// are enumerated for CLEAN sectors only (their tree set is exactly the
/// full-SU(3) set); tainted / escaped / poisoned candidates are reported in the
/// returned aggregate so the caller can refuse construction with an `Err` —
/// block dimensions are either exactly right or an error, never silently
/// truncated. A sector clean in one tuple but tainted in another is tainted
/// overall (its block would mix complete and incomplete tree sets).
fn fusion_trees_by_coupled_for_space_generic<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> (Vec<CoupledFusionTrees>, CoupledSectorFold)
where
    R: FusionRule,
{
    let mut grouped = Vec::<CoupledFusionTrees>::new();
    let mut index: FxHashMap<SectorId, usize> = FxHashMap::default();
    let mut aggregate = CoupledSectorFold::default();
    let mut clean_set: Vec<SectorId> = Vec::new();
    for tuple in space.selected_leg_tuples() {
        // `effective_sectors` is the uncoupled sectors verbatim (it ignores the
        // rule); inlined here to avoid its mult-free bound.
        let uncoupled: Vec<SectorId> = tuple.iter().map(|leg| leg.sector()).collect();
        let effective = uncoupled.clone();
        let is_dual: Vec<bool> = tuple.iter().map(|leg| leg.is_dual()).collect();
        let fold = rule.coupled_sector_fold(&effective);
        for &coupled in &fold.clean {
            let trees = collect_generic_fusion_trees_for_coupled(
                rule, &uncoupled, &is_dual, &effective, coupled,
            );
            match index.get(&coupled) {
                Some(&i) => grouped[i].trees.extend(trees),
                None => {
                    index.insert(coupled, grouped.len());
                    grouped.push(CoupledFusionTrees { coupled, trees });
                }
            }
        }
        clean_set.extend(fold.clean);
        aggregate.tainted.extend(fold.tainted);
        aggregate.out_of_table.extend(fold.out_of_table);
        aggregate.poisoned |= fold.poisoned;
    }
    aggregate.tainted.sort_unstable();
    aggregate.tainted.dedup();
    aggregate.out_of_table.sort();
    aggregate.out_of_table.dedup();
    clean_set.sort_unstable();
    clean_set.dedup();
    // Tainted-anywhere wins over clean-somewhere.
    clean_set.retain(|s| !aggregate.tainted.contains(s));
    aggregate.clean = clean_set;
    if aggregate.poisoned {
        // Same conservative contract as the per-tuple fold.
        let mut demoted = std::mem::take(&mut aggregate.clean);
        aggregate.tainted.append(&mut demoted);
        aggregate.tainted.sort_unstable();
        aggregate.tainted.dedup();
    }
    // Drop tree groups of sectors that lost their clean status across tuples.
    grouped.retain(|group| aggregate.clean.contains(&group.coupled));
    grouped.sort_by_key(|group| group.coupled);
    (grouped, aggregate)
}

/// Shared codomain×domain merge on equal coupled sectors (the generic sibling
/// of the loop in `fusion_tree_keys_uncached`).
fn merge_generic_tree_groups(
    codomain: &[CoupledFusionTrees],
    domain: &[CoupledFusionTrees],
) -> Vec<FusionTreeBlockKey> {
    let mut keys = Vec::new();
    let mut codomain_index = 0usize;
    let mut domain_index = 0usize;
    while codomain_index < codomain.len() && domain_index < domain.len() {
        match codomain[codomain_index]
            .coupled
            .cmp(&domain[domain_index].coupled)
        {
            std::cmp::Ordering::Less => codomain_index += 1,
            std::cmp::Ordering::Greater => domain_index += 1,
            std::cmp::Ordering::Equal => {
                for domain_tree in &domain[domain_index].trees {
                    for codomain_tree in &codomain[codomain_index].trees {
                        keys.push(FusionTreeBlockKey::pair(
                            codomain_tree.clone(),
                            domain_tree.clone(),
                        ));
                    }
                }
                codomain_index += 1;
                domain_index += 1;
            }
        }
    }
    keys
}

/// Human-readable summary of a non-clean coupled fold, for the construction
/// `Err` (names the escaping sectors — never silently dropped).
fn fusion_fold_error_message(side: &str, fold: &CoupledSectorFold) -> String {
    let mut parts = Vec::new();
    if !fold.out_of_table.is_empty() {
        parts.push(format!(
            "out-of-table coupled candidates on the {side} side: {}",
            fold.out_of_table.join(", ")
        ));
    }
    if !fold.tainted.is_empty() {
        parts.push(format!(
            "sectors requiring out-of-table intermediates on the {side} side: {:?}",
            fold.tainted
        ));
    }
    if fold.poisoned {
        parts.push(format!(
            "the {side}-side fold left the one-hop frontier shell (conservative)"
        ));
    }
    format!(
        "SU(3) dim<=27 table cannot represent this space exactly ({}); block \
         dimensions are either exact or an error, never truncated. Use \
         fusion_tree_keys_generic_for_coupled for provably-clean sectors, or \
         extend the table (Stage B3c).",
        parts.join("; ")
    )
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
    // Outer-multiplicity flag (`FusionStyleKind::has_multiplicity`, i.e.
    // `Generic`). Gates whether `vertices` participates in Hash/Eq/Ord below.
    // See the big comment on the Hash impl for why this exists and why it
    // is itself compared in Eq/Ord (not just used to gate `vertices`).
    has_multiplicity: bool,
}

// Identity of a `FusionTreeKey` is `(uncoupled, coupled, is_dual, innerlines)`
// — `vertices` is deliberately excluded from Hash/Eq/Ord *when the tree comes
// from a multiplicity-free rule* (`has_multiplicity == false`, every rule in
// this crate today). For multiplicity-free fusion the vertex labels are
// functionally determined by those four fields (always the trivial vertex),
// so two keys that agree on them agree on `vertices` too: excluding it
// changes no equivalence class or ordering, only the per-op cost.
// FusionTreeKey comparison/hashing is the hottest logic in the cold
// recoupling-plan build; TensorKit likewise keys its `SimpleFusion` fusion
// trees on the sectors alone.
//
// For outer-multiplicity (`FusionStyleKind::Generic`, `has_multiplicity ==
// true`) rules, `vertices` distinguishes trees that share the same four
// fields but took different fusion channels at a vertex with nsymbol > 1
// (e.g. SU(3)), so it must be included.
//
// `has_multiplicity` is included in Eq/Ord (not just used to *gate* the
// `vertices` comparison) because gating alone is order-dependent and breaks
// the Eq/Ord contracts: with `eq(a,b) = <4 fields> && (!a.has_multiplicity
// || a.vertices == b.vertices)`, if `a.has_multiplicity == false` and
// `b.has_multiplicity == true` with differing vertices, `eq(a,b)` would be
// true (vertices check skipped, using `a`'s flag) while `eq(b,a)` would be
// false (vertices check applied, using `b`'s flag) — not symmetric. Same
// issue for `cmp`'s antisymmetry. Comparing `has_multiplicity` itself first
// closes that hole: once it's confirmed equal on both sides, "use self's
// flag to decide whether to compare vertices" is unambiguous. The extra
// bool comparison is negligible next to a `SectorVec` compare and does not
// reopen the zero-cost gate: `Hash` still hashes `vertices` (and nothing
// else new) only when `has_multiplicity` is true, so mult-free hashing is
// byte-identical to before this field existed.
impl std::hash::Hash for FusionTreeKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.uncoupled.hash(state);
        self.coupled.hash(state);
        self.is_dual.hash(state);
        self.innerlines.hash(state);
        if self.has_multiplicity {
            self.vertices.hash(state);
        }
    }
}

impl PartialEq for FusionTreeKey {
    fn eq(&self, other: &Self) -> bool {
        self.uncoupled == other.uncoupled
            && self.coupled == other.coupled
            && self.is_dual == other.is_dual
            && self.innerlines == other.innerlines
            && self.has_multiplicity == other.has_multiplicity
            && (!self.has_multiplicity || self.vertices == other.vertices)
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
            .then_with(|| self.has_multiplicity.cmp(&other.has_multiplicity))
            .then_with(|| {
                if self.has_multiplicity {
                    self.vertices.cmp(&other.vertices)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
    }
}

impl PartialOrd for FusionTreeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl FusionTreeKey {
    pub fn try_new_for_rule<R, Uncoupled, Dual, Innerlines, Vertices>(
        rule: &R,
        uncoupled: Uncoupled,
        coupled: Option<SectorId>,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
        Uncoupled: IntoIterator<Item = SectorId>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = SectorId>,
        Vertices: IntoIterator<Item = SectorId>,
    {
        Self::try_new_for_style(
            rule.fusion_style(), uncoupled, coupled, is_dual, innerlines, vertices,
        )
    }

    pub(crate) fn try_new_for_style<Uncoupled, Dual, Innerlines, Vertices>(
        style: FusionStyleKind,
        uncoupled: Uncoupled,
        coupled: Option<SectorId>,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CoreError>
    where
        Uncoupled: IntoIterator<Item = SectorId>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = SectorId>,
        Vertices: IntoIterator<Item = SectorId>,
    {
        let has_multiplicity = style.has_multiplicity();
        let tree = Self::new(uncoupled, coupled, is_dual, innerlines, vertices)
            .with_has_multiplicity(has_multiplicity);
        if !has_multiplicity && tree.vertices.iter().any(|vertex| vertex.id() != 1) {
            return Err(CoreError::MalformedFusionTree {
                message: "multiplicity-free fusion tree has a nontrivial vertex",
            });
        }
        Ok(tree)
    }

    // `has_multiplicity` is NOT a parameter of `new`/`from_sector_ids`/
    // `from_uncoupled` on purpose. These three constructors have ~57 call
    // sites across tenet-core and tenet-tensors (production tree-transform
    // code, benches, tests) — every one of them today builds a
    // multiplicity-free tree. Threading a new required argument through all
    // of them would be a large, purely mechanical diff for a flag that is
    // `false` at every existing call site. Instead the constructors keep
    // their signatures and default to `false` internally (identical
    // behavior, zero call sites touched), and `with_has_multiplicity` below
    // is a chainable setter for the one place that needs `true`: the Stage A
    // toy OM rule's tests. When Stage B's recouple wrapper starts producing
    // real Generic-fusion trees, it can call `.with_has_multiplicity(true)`
    // at its own construction sites rather than everyone else's.
    pub(crate) fn new<Uncoupled, Dual, Innerlines, Vertices>(
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
            has_multiplicity: false,
        }
    }

    /// Set the outer-multiplicity flag (see the Hash impl comment). Chainable
    /// setter rather than a constructor parameter — see the rationale on
    /// `new` above for why the existing constructors were left alone.
    #[must_use]
    pub(crate) fn with_has_multiplicity(mut self, has_multiplicity: bool) -> Self {
        self.has_multiplicity = has_multiplicity;
        self
    }

    #[inline]
    pub fn has_multiplicity(&self) -> bool {
        self.has_multiplicity
    }

    pub(crate) fn from_sector_ids<Uncoupled, Dual, Innerlines, Vertices>(
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

    pub fn try_from_sector_ids_for_rule<R, Uncoupled, Dual, Innerlines, Vertices>(
        rule: &R,
        uncoupled: Uncoupled,
        coupled: Option<usize>,
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
            self.uncoupled[0]
                .id()
                .checked_mul(2)?
                .checked_add(usize::from(self.is_dual[0]))
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
    // Dedup destination tree-pairs to dense rows. The key is *moved* into the
    // map (no per-destination clone — this dedup is the hottest FusionTreeKey
    // clone/eq/hash site on the cold recoupling path); `next_basis` is rebuilt
    // from the map by row index afterwards. Rows are assigned in first-
    // appearance order, so the rebuilt `next_basis` order — and therefore every
    // coefficient — is bit-for-bit identical to pushing the key eagerly.
    let mut index: FxHashMap<FusionTreeBlockKey, usize> = FxHashMap::default();
    let mut next_columns: DenseColumns<R::Scalar> = DenseColumns::with_capacity(num_src, basis.len());
    for (source_row, source_key) in basis.iter().enumerate() {
        for (dst_key, step_coefficient) in transform(rule, source_key)? {
            let row = match index.get(&dst_key) {
                Some(&row) => row,
                None => {
                    let row = next_columns.push_empty_row();
                    index.insert(dst_key, row);
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
    // Rebuild the basis in row order (= first-appearance order). Rows are dense
    // `0..index.len()`, so place each moved key at its row index.
    let mut slots: Vec<Option<FusionTreeBlockKey>> = (0..index.len()).map(|_| None).collect();
    for (key, row) in index {
        slots[row] = Some(key);
    }
    let next_basis: Vec<FusionTreeBlockKey> = slots
        .into_iter()
        .map(|key| key.expect("dense rows 0..len are all filled"))
        .collect();
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

/// Batched [`multiplicity_free_transpose_tree_pair`] over every source
/// tree-pair of a block (all sharing uncoupled sectors / duality). The planar
/// cyclic-transpose step sequence — repartition to the target codomain rank,
/// then rotate the coupled loop one leg at a time — depends only on the ranks
/// and permutation, so it is identical for every source. Walk it once over the
/// shared `DenseColumns` matrix instead of replaying the repartition and cyclic
/// bends per source (TensorKit 0.17's block `fstranspose`). Returns, per source
/// in `src_keys` order, its `(destination tree-pair, coefficient)` rows.
///
/// As with the braid block port, coefficients that reach a destination by
/// several paths sum in a different order than the per-source accumulator, so
/// results agree to double-precision rounding, not necessarily bit-for-bit.
pub fn multiplicity_free_transpose_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreeBlockKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
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

    let num_src = src_keys.len();

    // A cyclic permutation whose `0` already sits at position 0 (or that has no
    // `0` at all) is the identity transpose: every source maps to itself with
    // no repartition, matching the per-source early return.
    let mut position = match permutation.iter().position(|&axis| axis == 0) {
        Some(position) => position,
        None => {
            return Ok(src_keys
                .iter()
                .map(|key| vec![(key.clone(), rule.scalar_one())])
                .collect());
        }
    };

    // Identity matrix: source `i` starts as its own basis tree with coeff one.
    let mut basis = src_keys.to_vec();
    let mut columns: DenseColumns<R::Scalar> = DenseColumns::with_capacity(num_src, num_src);
    for i in 0..num_src {
        let row = columns.push_empty_row();
        columns.row_mut(row)[i] = Some(rule.scalar_one());
    }

    // Repartition into the requested codomain rank (bendleft / bendright chain),
    // batched across the block.
    let target_codomain_rank = codomain_permutation.len();
    let mut current_codomain_rank = codomain_rank;
    while current_codomain_rank < target_codomain_rank {
        let (next_basis, next_columns) = compose_block_terms(rule, &basis, &columns, |rule, key| {
            multiplicity_free_bendleft_tree_pair(rule, key)
        })?;
        basis = next_basis;
        columns = next_columns;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        let (next_basis, next_columns) = compose_block_terms(rule, &basis, &columns, |rule, key| {
            multiplicity_free_bendright_tree_pair(rule, key)
        })?;
        basis = next_basis;
        columns = next_columns;
        current_codomain_rank -= 1;
    }

    let total_rank = codomain_rank + domain_rank;
    if total_rank != 0 && position != 0 {
        // Rotate the coupled fusion loop one leg per step (anticlockwise while
        // `0` is in the near half, clockwise past the midpoint), each cycle
        // batched — the block port of the per-source cyclic bends.
        let half_rank = total_rank >> 1;
        while position > 0 && position < half_rank {
            let (next_basis, next_columns) =
                compose_block_terms(rule, &basis, &columns, |rule, key| {
                    multiplicity_free_cycle_anticlockwise_tree_pair(rule, key)
                })?;
            basis = next_basis;
            columns = next_columns;
            position -= 1;
        }
        while position >= half_rank && position > 0 {
            let (next_basis, next_columns) =
                compose_block_terms(rule, &basis, &columns, |rule, key| {
                    multiplicity_free_cycle_clockwise_tree_pair(rule, key)
                })?;
            basis = next_basis;
            columns = next_columns;
            position = (position + 1) % total_rank;
        }
    }

    // Scatter the dense matrix back into per-source row lists (columns are
    // indexed by source, so source order needs no sort).
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

/// Elementary Artin braid of neighbouring uncoupled legs `index` and `index+1`
/// for an outer-multiplicity (`FusionStyleKind::Generic`) rule — the verbatim
/// mirror of TensorKit's `GenericFusion` branches of
/// `artin_braid(src::FusionTreeBlock, i; inv)`
/// (`fusiontrees/braiding_manipulations.jl:81-198`).
///
/// Where the multiplicity-free sibling
/// [`multiplicity_free_artin_braid_at_with_inverse`] returns a scalar per
/// output tree, here every vertex carries an outer-multiplicity label (1-based,
/// stored as `SectorId::new(label)` exactly like the trivial `SectorId::new(1)`
/// the mult-free enumerator writes), and one input tree can braid into several
/// output trees that differ *only* in their vertex labels. Each output's scalar
/// coefficient is the `R · F̄ · R̄` inner-index contraction TensorKit writes at
/// `braiding_manipulations.jl:181-182`.
///
/// Outputs are built `.with_has_multiplicity(true)` so the Stage A
/// `FusionTreeKey` identity gate keeps vertex-distinct trees distinct.
///
/// The `inverse` flag is handled exactly as TensorKit does — the R-matrices
/// become adjoints (`Rsymbol(...)'`, `braiding_manipulations.jl:139,172-173`),
/// the F-symbol is *not* adjointed, and the contraction formula is otherwise
/// unchanged — rather than being derived here. Applying the `inverse=true`
/// braid to every output of the `inverse=false` braid recovers the original
/// tree with coefficient 1 (unit F/R), which the tests check.
fn generic_artin_braid_at_with_inverse<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: GenericBraidScalar,
{
    // Entry gate: Generic-fusion only. `has_multiplicity()` is exactly the
    // `FusionStyle(I) isa GenericFusion` predicate TensorKit branches on
    // (braiding_manipulations.jl:137,170). Mult-free rules must use the
    // scalar-coefficient path instead.
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }

    let rank = tree.uncoupled().len();
    if index + 1 >= rank {
        return Err(CoreError::InvalidBraidIndex { index, rank });
    }

    // a, b = uncoupled[i], uncoupled[i+1]; swap them into uncoupled′, isdual′
    // (braiding_manipulations.jl:86-93). `left`/`right` keep the a/b naming for
    // the i == 1 special case; the i > 1 case renames below, matching TK's
    // "other naming convention" comment at :151.
    let left = tree.uncoupled()[index];
    let right = tree.uncoupled()[index + 1];
    let mut uncoupled: SectorVec = tree.uncoupled().iter().copied().collect();
    uncoupled.swap(index, index + 1);
    let mut is_dual: DualVec = tree.is_dual().iter().copied().collect();
    is_dual.swap(index, index + 1);

    // Braiding with the trivial sector: simple and always possible, coefficient
    // 1, no F/R needed (braiding_manipulations.jl:101-120). Identical bookkeeping
    // to the mult-free branch; the vertices just carry OM labels now.
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
        return Ok(vec![(
            FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices)
                .with_has_multiplicity(true),
            R::Scalar::braid_one(),
        )]);
    }

    // NoBraiding rules cannot braid non-trivial sectors
    // (braiding_manipulations.jl:122-123).
    if !rule.braiding_style().has_braiding() {
        return Err(CoreError::UnsupportedSectorBraid {
            left,
            right,
            style: rule.braiding_style(),
        });
    }

    if index == 0 {
        // c = N > 2 ? inner[1] : coupled′  (braiding_manipulations.jl:131)
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
        // GenericFusion i == 1 branch (braiding_manipulations.jl:137-148).
        // μ = vertices[1] (the single input vertex), 1-based label -> 0-based
        // matrix row (:138).
        let mu0 = mu_index(tree, 0)?;
        // Rmat = inv ? Rsymbol(b,a,c)' : Rsymbol(a,b,c)  (:139). We fetch the
        // *un-adjointed* base and take the adjoint at the element read below.
        let rmat = if inverse {
            rule.r_symbol_generic(right, left, coupled)
        } else {
            rule.r_symbol_generic(left, right, coupled)
        };
        // ν ranges over the output vertex = the columns of Rmat (:140). The
        // adjoint flips the shape back, so that count is N(b,a,c) either way.
        let n_nu = rule.nsymbol(right, left, coupled);
        let mut out = Vec::with_capacity(n_nu);
        for nu0 in 0..n_nu {
            // R = Rmat[μ, ν]  (:141). For the adjoint, Rmat[μ,ν] = conj(base[ν,μ]).
            let r = if inverse {
                rmat.get(nu0, mu0).braid_conj()
            } else {
                rmat.get(mu0, nu0).clone()
            };
            if r.braid_is_zero() {
                continue; // iszero(R) && continue  (:142)
            }
            // vertices′ = setindex(vertices, ν, 1)  (:143)
            let mut vertices: SectorVec = tree.vertices().iter().copied().collect();
            *vertices
                .get_mut(0)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "first braid of a Generic tree requires a vertex",
                })? = SectorId::new(nu0 + 1);
            out.push((
                FusionTreeKey::new(
                    uncoupled.clone(),
                    tree.coupled(),
                    is_dual.clone(),
                    tree.innerlines().iter().copied(),
                    vertices,
                )
                .with_has_multiplicity(true),
                r,
            ));
        }
        return Ok(out);
    }

    // case i > 1: other naming convention (braiding_manipulations.jl:151-156).
    // b = uncoupled[i]; d = uncoupled[i+1]; a = inner_ext[i-1]; c = inner_ext[i];
    // e = inner_ext[i+1].
    let a = inner_extended_sector(tree, index - 1)?;
    let b = left;
    let c = inner_extended_sector(tree, index)?;
    let d = right;
    let e = inner_extended_sector(tree, index + 1)?;
    // μ = vertices[i-1]; ν = vertices[i]  (:175-176), 1-based label -> 0-based.
    let mu0 = mu_index(tree, index - 1)?;
    let nu0 = mu_index(tree, index)?;

    let mut out = Vec::new();
    // for c′ in intersect(a ⊗ d, e ⊗ conj(b))  (:171). `c' ∈ a⊗d` filtered by
    // N(c',b,e) > 0 is exactly that intersection (N(c',b,e) > 0 ⟺ c' ∈ e⊗conj(b)),
    // the same rewrite the mult-free branch uses.
    //
    // `fusion_channels_in_table` (not `fusion_channels`): for a bounded table
    // rule (SU(3)) the pair (a, d) can escape even on a legal tree (e.g.
    // a=27, d=8). Skipped frontier c′ are provably dead: transforms only run
    // on structures whose sectors the coupled fold admitted as clean, and a
    // nonzero frontier-c′ term would be a full-SU(3) tree through an
    // out-of-table inner line — contradicting cleanness.
    for c_prime in rule.fusion_channels_in_table(a, d) {
        if rule.nsymbol(c_prime, b, e) == 0 {
            continue;
        }
        // Rmat1 = inv ? Rsymbol(d,c,e)' : Rsymbol(c,d,e)   (:172)
        // Rmat2 = inv ? Rsymbol(d,a,c')' : Rsymbol(a,d,c')  (:173)
        // Fmat = Fsymbol(d,a,b,e,c',c)                      (:174)
        let rmat1 = if inverse {
            rule.r_symbol_generic(d, c, e)
        } else {
            rule.r_symbol_generic(c, d, e)
        };
        let rmat2 = if inverse {
            rule.r_symbol_generic(d, a, c_prime)
        } else {
            rule.r_symbol_generic(a, d, c_prime)
        };
        let fmat = rule.f_symbol_generic(d, a, b, e, c_prime, c);
        // Output vertex ranges σ ∈ 1:N(a,d,c'), λ ∈ 1:N(c',b,e)  (:177-178);
        // inner-sum ranges ρ ∈ 1:N(d,c,e), κ ∈ 1:N(d,a,c')  (:180).
        let n_sigma = rule.nsymbol(a, d, c_prime);
        let n_lambda = rule.nsymbol(c_prime, b, e);
        let n_rho = rule.nsymbol(d, c, e);
        let n_kappa = rule.nsymbol(d, a, c_prime);
        for sigma0 in 0..n_sigma {
            for lambda0 in 0..n_lambda {
                // coeff = zero(oneT)  (:179)
                let mut coeff = R::Scalar::braid_zero();
                for rho0 in 0..n_rho {
                    for kappa0 in 0..n_kappa {
                        // coeff += Rmat1[ν,ρ] * conj(Fmat[κ,λ,μ,ρ]) * conj(Rmat2[σ,κ])
                        // (:181-182). Adjoint element reads (see the trait doc):
                        //   Rmat1[ν,ρ]      : base[ν,ρ]      | conj(base[ρ,ν])   (inv)
                        //   conj(Rmat2[σ,κ]): conj(base[σ,κ])| base[κ,σ]         (inv, double-conj cancels)
                        let r1 = if inverse {
                            rmat1.get(rho0, nu0).braid_conj()
                        } else {
                            rmat1.get(nu0, rho0).clone()
                        };
                        let f_conj = fmat.get(kappa0, lambda0, mu0, rho0).braid_conj();
                        let r2_conj = if inverse {
                            rmat2.get(kappa0, sigma0).clone()
                        } else {
                            rmat2.get(sigma0, kappa0).braid_conj()
                        };
                        coeff = coeff + r1 * f_conj * r2_conj;
                    }
                }
                if coeff.braid_is_zero() {
                    continue; // iszero(coeff) && continue  (:184)
                }
                // vertices′ = setindex(setindex(vertices, σ, i-1), λ, i)  (:185-186)
                // inner′ = setindex(inner, c′, i-1)  (:187)
                let mut innerlines: SectorVec = tree.innerlines().iter().copied().collect();
                *innerlines
                    .get_mut(index - 1)
                    .ok_or(CoreError::MalformedFusionTree {
                        message: "non-first braid requires an innerline to update",
                    })? = c_prime;
                let mut vertices: SectorVec = tree.vertices().iter().copied().collect();
                if vertices.len() <= index {
                    return Err(CoreError::MalformedFusionTree {
                        message: "non-first Generic braid requires adjacent vertices",
                    });
                }
                vertices[index - 1] = SectorId::new(sigma0 + 1);
                vertices[index] = SectorId::new(lambda0 + 1);
                out.push((
                    FusionTreeKey::new(
                        uncoupled.clone(),
                        tree.coupled(),
                        is_dual.clone(),
                        innerlines,
                        vertices,
                    )
                    .with_has_multiplicity(true),
                    coeff,
                ));
            }
        }
    }
    Ok(out)
}

/// Read the 0-based outer-multiplicity matrix index of the vertex at position
/// `vertex_index`. Vertex labels are stored 1-based (`SectorId::new(label)`,
/// the same convention as the trivial `SectorId::new(1)` the mult-free
/// enumerator writes), and TensorKit's `Rmat[μ, ν]` / `Fmat[κ, λ, μ, ρ]` are
/// 1-based Julia indices, so the stored label maps to the 0-based Rust index by
/// subtracting one.
fn mu_index(tree: &FusionTreeKey, vertex_index: usize) -> Result<usize, CoreError> {
    let label = tree
        .vertices()
        .get(vertex_index)
        .copied()
        .ok_or(CoreError::MalformedFusionTree {
            message: "Generic braid requires a vertex label at the braided position",
        })?
        .id();
    label.checked_sub(1).ok_or(CoreError::MalformedFusionTree {
        message: "Generic vertex labels are 1-based; label 0 is invalid",
    })
}

/// Braid the uncoupled legs of a Generic-fusion tree by `permutation` under the
/// given `levels`, the outer-multiplicity mirror of
/// [`multiplicity_free_braid_tree`] and of TensorKit's `braid(f, p, levels)`
/// swap-decomposition loop (`braiding_manipulations.jl:235-248`,
/// non-`SymmetricBraiding` branch). The permutation is decomposed into
/// neighbouring swaps; each swap is an [`generic_artin_braid_at_with_inverse`]
/// with `inverse = levels[s] > levels[s+1]` (:239), and the running level
/// tuple is swapped after each step (:243-244).
///
/// Because one input tree can fan out to several vertex-labelled outputs, the
/// coefficients are threaded through a [`FusionTermAccumulator`] (summing paths
/// that reconverge on the same output tree), exactly as the multiplicity-free
/// braid does.
// `pub` to mirror the mult-free split (`multiplicity_free_braid_tree` is `pub`,
// its per-swap artin helper private). Being a public root also keeps
// `generic_artin_braid_at_with_inverse` / `mu_index` reachable, so they emit no
// dead-code warning before Stage B2's recouple wrapper consumes them.
pub fn generic_braid_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    levels: &[usize],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: GenericBraidScalar,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
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
    let mut current = vec![(tree.clone(), R::Scalar::braid_one())];
    let mut current_levels = levels.to_vec();
    for swap in swaps {
        let inverse = current_levels[swap] > current_levels[swap + 1];
        let mut next_terms = FusionTermAccumulator::new();
        for (tree, coefficient) in current {
            for (next_tree, step_coefficient) in
                generic_artin_braid_at_with_inverse(rule, &tree, swap, inverse)?
            {
                next_terms.push(next_tree, coefficient.clone() * step_coefficient);
            }
        }
        current_levels.swap(swap, swap + 1);
        current = next_terms.into_vec();
    }
    Ok(current)
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

/// Generic-fusion (outer multiplicity) `bendright`: map the final splitting
/// vertex `a ⊗ b ← c` of the codomain to a fusion vertex on the domain,
/// producing a *fanout* of vertex-labelled output tree pairs.
///
/// Verbatim mirror of TensorKit `bendright(src::FusionTreeBlock)`, GenericFusion
/// branch (`duality_manipulations.jl:69-114`, specifically the `else` at
/// `:97-112`), applied to a single input tree pair. The tree-key surgery is
/// identical to [`multiplicity_free_bendright_tree_pair`] (bookkeeping is
/// scalar-independent — TK `_bendright_treepair` :33-54); only the coefficient
/// becomes a `B[μ, ν]` row/column read instead of a bare `B` scalar.
///
/// The `ν`-loop mirrors TK's inner `for ν in axes(Bmat, 2)` (:104). When the
/// original domain is empty (`N₂ == 0`) TK stores no new vertex, so every `ν`
/// collapses onto the same output key and the block's `U[row, col] = coeff`
/// assignment (:110) keeps the *last* non-skipped `ν`; we reproduce that with a
/// keep-last overwrite on key collision. When the domain is non-empty, `ν` is
/// stored on the new domain tree, keys are distinct, and no overwrite occurs.
fn generic_bendright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
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

    // a = N₁==1 ? unit : N₁==2 ? uncoupled[1] : innerlines[end]  (TK :37).
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
    // b = uncoupled[N₁]  (TK :38).
    let bent_sector = codomain.uncoupled()[codomain_rank - 1];
    let bent_is_dual = codomain.is_dual().get(codomain_rank - 1).copied().ok_or(
        CoreError::MalformedFusionTree {
            message: "codomain tree is missing a duality flag",
        },
    )?;

    // New codomain tree: drop the last leg (TK `_bendright_treepair` :41-45);
    // has_multiplicity kept so the surviving vertex labels stay meaningful.
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
    )
    .with_has_multiplicity(true);

    let domain_rank = domain.uncoupled().len();
    // Base domain data shared by every ν; only the appended vertex label varies
    // (TK :100-103, `uncoupled₂/coupled₂/isdual₂/inner₂` hoisted out of the loop).
    let domain_uncoupled: SectorVec = domain
        .uncoupled()
        .iter()
        .copied()
        .chain(std::iter::once(rule.dual(bent_sector)))
        .collect();
    let domain_is_dual: DualVec = domain
        .is_dual()
        .iter()
        .copied()
        .chain(std::iter::once(!bent_is_dual))
        .collect();
    let domain_innerlines: SectorVec = domain
        .innerlines()
        .iter()
        .copied()
        .chain((domain_rank > 1).then_some(coupled))
        .collect();

    // coeff₀ = √dim(c)·(1/√dim(a)); ·conj(κ_{dual(b)}) if the bent leg is dual
    // (TK :89-92, same placement as the mult-free bend :2424-2429).
    let mut coeff0 = rule.sqrt_dim_scalar(coupled) * rule.inv_sqrt_dim_scalar(left_coupled);
    if bent_is_dual {
        coeff0 = coeff0
            * rule
                .frobenius_schur_phase_scalar(rule.dual(bent_sector))
                .braid_conj();
    }

    // Bmat = Bsymbol(a, b, c)  (TK :98); μ = N₁>1 ? vertices[end] : 1  (TK :99).
    let bmat = rule.b_symbol_generic(left_coupled, bent_sector, coupled);
    let mu0 = if codomain_rank > 1 {
        mu_index(codomain, codomain_rank - 2)?
    } else {
        0
    };

    let (_, cols) = bmat.shape();
    let mut out: Vec<(FusionTreeBlockKey, R::Scalar)> = Vec::new();
    for nu0 in 0..cols {
        // coeff = coeff₀ · Bmat[μ, ν]  (TK :105); iszero → skip  (TK :106).
        let coeff = coeff0.clone() * bmat.get(mu0, nu0).clone();
        if coeff.braid_is_zero() {
            continue;
        }
        // vertices₂ = N₂>0 ? (f₂.vertices..., ν) : ()  (TK :107). ν is the
        // 1-based output vertex label (mu_index inverts this on the way back).
        let new_domain = FusionTreeKey::new(
            domain_uncoupled.iter().copied(),
            Some(left_coupled),
            domain_is_dual.iter().copied(),
            domain_innerlines.iter().copied(),
            domain
                .vertices()
                .iter()
                .copied()
                .chain((domain_rank > 0).then_some(SectorId::new(nu0 + 1))),
        )
        .with_has_multiplicity(true);
        let key = FusionTreeBlockKey::pair(new_codomain.clone(), new_domain);
        // TK block writes `U[row, col] = coeff` (:110), so a repeated key (only
        // when the domain was empty) is overwritten, keeping the last ν.
        if let Some(slot) = out.iter_mut().find(|(existing, _)| *existing == key) {
            slot.1 = coeff;
        } else {
            out.push((key, coeff));
        }
    }
    Ok(out)
}

/// Generic-fusion `bendleft`: inverse planar move of [`generic_bendright_tree_pair`],
/// mapping the final domain (fusion) vertex back to a codomain splitting vertex.
///
/// Verbatim mirror of TensorKit `bendleft` (`duality_manipulations.jl:140-144`,
/// the "copy of bendright through (f₂,f₁) => conj(coeff)" note at :146-147):
/// swap codomain/domain, run `bendright`, swap back, and conjugate every
/// coefficient. Structurally identical to the mult-free
/// [`multiplicity_free_bendleft_tree_pair`] :2439-2460.
fn generic_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let swapped = FusionTreeBlockKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(generic_bendright_tree_pair(rule, &swapped)?
        .into_iter()
        .map(|(bent, coefficient)| {
            (
                FusionTreeBlockKey::pair(bent.domain_tree().clone(), bent.codomain_tree().clone()),
                coefficient.braid_conj(),
            )
        })
        .collect())
}

/// Compose a term list with an elementary Generic-fusion transform, summing
/// coefficients over coincident output trees (matrix product over the
/// intermediate basis). Generic sibling of [`compose_tree_pair_terms`] — same
/// [`FusionTermAccumulator`], different rule bound.
fn compose_generic_tree_pair_terms<R, F, I>(
    rule: &R,
    terms: Vec<(FusionTreeBlockKey, R::Scalar)>,
    mut transform: F,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
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

/// Generic-fusion `repartition`: bend legs between codomain and domain until the
/// codomain has `target_codomain_rank` legs. Verbatim mirror of TensorKit
/// `repartition` / `_repartition_body` (`duality_manipulations.jl:460-505`): the
/// generated function unrolls `|N|` `bendleft`/`bendright` steps and composes
/// their coefficient matrices (`U = Utmp * U`), which is exactly this
/// accumulate-and-compose loop. Structural twin of
/// [`multiplicity_free_repartition_tree_pair`] :794-827.
pub fn generic_repartition_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let total_rank =
        tree_pair.codomain_tree().uncoupled().len() + tree_pair.domain_tree().uncoupled().len();
    if target_codomain_rank > total_rank {
        return Err(CoreError::DimensionMismatch {
            expected: total_rank,
            actual: target_codomain_rank,
        });
    }

    let mut current = vec![(tree_pair.clone(), R::Scalar::braid_one())];
    let mut current_codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    // N = numout - target > 0 ⇒ bendright; < 0 ⇒ bendleft (TK :492).
    while current_codomain_rank < target_codomain_rank {
        current = compose_generic_tree_pair_terms(rule, current, |rule, key| {
            generic_bendleft_tree_pair(rule, key)
        })?;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        current = compose_generic_tree_pair_terms(rule, current, |rule, key| {
            generic_bendright_tree_pair(rule, key)
        })?;
        current_codomain_rank -= 1;
    }
    Ok(current)
}

// ======================================================================
// Stage B2b: Generic-fusion coefficient-vector layer (multi_Fmove /
// multi_associator) plus foldright/foldleft/cycle. Outer-multiplicity
// mirror of the multiplicity-free tree functions below, and of TensorKit
// `basic_manipulations.jl` / `duality_manipulations.jl` (the GenericFusion
// else-branches).
//
// COEFFICIENT-VECTOR INDEX CONVENTION (documented once, referenced below).
// A `multi_Fmove` / `multi_associator` on a splitting tree that splits off
// the leftmost sector `a` leaves a family of standard-form trees, each
// carrying a coefficient *vector* rather than a scalar. The vector index is
// the label `λ` of the TOPMOST fusion vertex `a ⊗ b → c`, where
//   a = tree.uncoupled[0]      (the split-off sector, fixed),
//   b = tail_tree.coupled      (the coupled sector of the (N-1)-leg tail),
//   c = tree.coupled           (the overall coupled sector, fixed),
// so the vector has length `Nsymbol(a, b, c)`. This is exactly TensorKit's
// convention (`basic_manipulations.jl:133-135, 186-187, 346-347`): "vectors
// of length Nsymbol(a,b,c), representing the coefficients associated with the
// different vertex labels λ of the topmost vertex". The λ vertex is NOT part
// of the emitted tail tree — it is the *free* index that a downstream
// operation (fold: the A-move; a further recoupling) contracts against. At
// completion the vector is distributed to scalar coefficients (fold's
// A-matrix contraction collapses it to one scalar per output tree pair).
// ======================================================================

/// A generic `multi_Fmove` / `multi_Fmove_inv` result: each recoupled
/// standard-form tree paired with its coefficient VECTOR (indexed by the
/// topmost `a ⊗ b → c` vertex `λ`; see the convention block above). Aliased to
/// keep the tree-function signatures readable and satisfy
/// `clippy::type_complexity`.
type GenericFmoveTerms<S> = Vec<(FusionTreeKey, Vec<S>)>;

/// Enumerate every standard-form fusion tree with the given `uncoupled` legs,
/// `is_dual` flags and `coupled` sector, INCLUDING all outer-multiplicity
/// vertex-label assignments. Generic sibling of
/// [`collect_fusion_trees_for_coupled`] (which hard-codes `SectorId::new(1)`
/// for every vertex and is bounded on `MultiplicityFreeFusionRule`): here each
/// vertex with `Nsymbol > 1` branches over its `1..=N` labels, producing one
/// tree per (innerlines, vertices) combination. This is the enumeration
/// TensorKit's `multi_Fmove` Stage 1 performs inline (`for μ in 1:Nbce′` at
/// `basic_manipulations.jl:265`); factoring it out keeps `generic_multi_fmove_*`
/// structurally identical to the multiplicity-free tree functions.
fn collect_generic_fusion_trees_for_coupled<R>(
    rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
) -> Vec<FusionTreeKey>
where
    R: FusionRule,
{
    let mut out = Vec::new();
    // `inner_rev` / `vtx_rev` accumulate outermost-first as the walk descends
    // (the top vertex/innerline is pushed first); the stored key wants
    // innermost-first, so emit reverses both — same discipline as
    // `visit_fusion_trees`, extended to vertex labels.
    let mut inner_rev: Vec<SectorId> = Vec::new();
    let mut vtx_rev: Vec<usize> = Vec::new();
    visit_generic_fusion_trees(
        rule,
        effective,
        coupled,
        &mut inner_rev,
        &mut vtx_rev,
        &mut |inner_rev, vtx_rev| {
            out.push(
                FusionTreeKey::new(
                    uncoupled.iter().copied(),
                    Some(coupled),
                    is_dual.iter().copied(),
                    inner_rev.iter().rev().copied(),
                    vtx_rev.iter().rev().map(|&label| SectorId::new(label)),
                )
                .with_has_multiplicity(true),
            );
        },
    );
    out
}

/// Recursive walker for [`collect_generic_fusion_trees_for_coupled`]. Mirrors
/// [`visit_fusion_trees`] (peels the LAST leg, recursing inward), but at every
/// vertex it iterates `1..=Nsymbol(...)` and records the 1-based label. Vertex
/// labels are stored 1-based (`SectorId::new(label)`, the same convention
/// [`mu_index`] decodes).
fn visit_generic_fusion_trees<R, F>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
    inner_rev: &mut Vec<SectorId>,
    vtx_rev: &mut Vec<usize>,
    emit: &mut F,
) where
    R: FusionRule,
    F: FnMut(&[SectorId], &[usize]),
{
    match effective.len() {
        0 => {
            if coupled == rule.vacuum() {
                emit(inner_rev, vtx_rev);
            }
        }
        1 => {
            if effective[0] == coupled {
                emit(inner_rev, vtx_rev);
            }
        }
        2 => {
            // Base vertex `e0 ⊗ e1 → coupled`, labels 1..=N(e0,e1,coupled).
            let n = rule.nsymbol(effective[0], effective[1], coupled);
            for label in 1..=n {
                vtx_rev.push(label);
                emit(inner_rev, vtx_rev);
                vtx_rev.pop();
            }
        }
        _ => {
            let last = effective[effective.len() - 1];
            let front_effective = &effective[..effective.len() - 1];
            // Inner line ranges over `coupled ⊗ dual(last)`; the top vertex
            // `front_coupled ⊗ last → coupled` has `N(front_coupled,last,coupled)`
            // labels. (Same `vertexiterN` structure as the mult-free walker.)
            // `fusion_channels_in_table`: only clean sectors are ever walked
            // (tainted/escaped are an Err upstream), and clean sectors have no
            // tree through a frontier inner line — skipping frontier
            // `front_coupled` candidates drops only provably-dead branches.
            for front_coupled in rule.fusion_channels_in_table(coupled, rule.dual(last)) {
                let n_last = rule.nsymbol(front_coupled, last, coupled);
                if n_last == 0 {
                    continue;
                }
                inner_rev.push(front_coupled);
                for label in 1..=n_last {
                    vtx_rev.push(label);
                    visit_generic_fusion_trees(
                        rule,
                        front_effective,
                        front_coupled,
                        inner_rev,
                        vtx_rev,
                        emit,
                    );
                    vtx_rev.pop();
                }
                inner_rev.pop();
            }
        }
    }
}

/// Generic-fusion `multi_associator`: the coefficient VECTOR relating a long
/// (`N`-leg) splitting tree to a short (`N-1`-leg) tail tree, indexed by the
/// topmost `a ⊗ short.coupled → long.coupled` vertex `λ` (see the module
/// convention block above). Verbatim mirror of TensorKit `multi_associator`
/// GenericFusion branch (`basic_manipulations.jl:144-166`); the
/// multiplicity-free sibling [`multiplicity_free_multi_associator_scalar`]
/// returns a bare scalar (this is that scalar chain lifted to a length-`Nλ`
/// vector).
///
/// Returns `None` iff the uncoupled/dual tails do not match (the `zero(...)`
/// early return at TK `:141-142`), so callers filter exactly as the mult-free
/// tree functions do.
fn generic_multi_associator<R>(
    rule: &R,
    long: &FusionTreeKey,
    short: &FusionTreeKey,
) -> Result<Option<Vec<R::Scalar>>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rank = long.uncoupled().len();
    if short.uncoupled().len() + 1 != rank {
        return Ok(None);
    }
    if long.uncoupled()[1..] != *short.uncoupled() || long.is_dual()[1..] != *short.is_dual() {
        return Ok(None);
    }
    let first = long.uncoupled()[0];

    // Base case `rank == 2` (TK's `2:(N-1)` loop is empty): there is no F to
    // apply, and the topmost vertex IS `long`'s single vertex μ. The vector is
    // the unit vector `e_μ` over `a ⊗ b → c`, length `N(a,b,c)`. This is the
    // `μ = f.vertices[1]; coeff = e_μ` special case TK's `multi_Fmove`
    // (`:229-232`) and `multi_Fmove_inv` N==1 (`:373-377`) inline; here it lives
    // in the associator so the inv path (which reuses this associator on a
    // rank-2 candidate over a rank-1 tail) gets the right seed too.
    if rank == 2 {
        let b = long.uncoupled()[1];
        let c = coupled_or_vacuum(rule, long);
        let n = rule.nsymbol(first, b, c);
        let mu0 = mu_index(long, 0)?;
        let mut coeff = vec![R::Scalar::braid_zero(); n];
        if let Some(slot) = coeff.get_mut(mu0) {
            *slot = R::Scalar::braid_one();
        } else {
            return Err(CoreError::MalformedFusionTree {
                message: "multi_associator: vertex label exceeds Nsymbol",
            });
        }
        return Ok(Some(coeff));
    }

    // General chain (TK `:150-165`). `coeff` starts as the length-1 seed and is
    // transformed by one F-slice per interior leg. After each step it is indexed
    // by the current step's `λ` axis (F axis 4, `N(a, e′, d)`), which becomes
    // the next step's `μ` axis (F axis 1, `N(a, b, e)`) — the associator chain.
    let mut coeff = vec![R::Scalar::braid_one()];
    for tensor_kit_k in 2..rank {
        let right_sector = long.uncoupled()[tensor_kit_k]; // c
        // vertex_info(long, k+1) = (e, d); ν = its vertex label.
        let (middle_left, middle_right) = fusion_tree_vertex_neighbors(long, tensor_kit_k)?;
        let nu0 = mu_index(long, tensor_kit_k - 1)?;
        // vertex_info(short, k) = (b, e′); κ = its vertex label.
        let (short_left, short_right) = fusion_tree_vertex_neighbors(short, tensor_kit_k - 1)?;
        let kappa0 = mu_index(short, tensor_kit_k - 2)?;
        // F = Fsymbol(a, b, c, d, e, e′); axis order (μ, ν, κ, λ) =
        // (N(a,b,e), N(e,c,d), N(b,c,e′), N(a,e′,d)). Same argument order the
        // mult-free scalar associator passes to `f_symbol_scalar`.
        let f = rule.f_symbol_generic(
            first,
            short_left,
            right_sector,
            middle_right,
            middle_left,
            short_right,
        );
        let n_lambda = f.shape().3;
        let mut next = vec![R::Scalar::braid_zero(); n_lambda];
        if tensor_kit_k == 2 {
            // `transpose(view(F, μ:μ, ν, κ, :)) * coeff` (TK `:159-160`): the μ
            // axis is fixed to `long.vertices[0]`, seed has length 1.
            let mu0 = mu_index(long, 0)?;
            for (lambda, slot) in next.iter_mut().enumerate() {
                *slot = f.get(mu0, nu0, kappa0, lambda).clone() * coeff[0].clone();
            }
        } else {
            // `transpose(view(F, :, ν, κ, :)) * coeff` (TK `:162`): sum over the
            // μ axis (= incoming vector index) into the λ axis.
            for (lambda, slot) in next.iter_mut().enumerate() {
                let mut acc = R::Scalar::braid_zero();
                for (mu, coeff_mu) in coeff.iter().enumerate() {
                    acc = acc + f.get(mu, nu0, kappa0, lambda).clone() * coeff_mu.clone();
                }
                *slot = acc;
            }
        }
        coeff = next;
    }
    Ok(Some(coeff))
}

/// Generic-fusion `multi_Fmove`: recouple a splitting tree to split off its
/// first uncoupled sector, returning `(tail_tree, coeff_vector)` pairs. Mirror
/// of TensorKit `multi_Fmove` GenericFusion branch (`basic_manipulations.jl:
/// 218-232, 234-327`) and structural twin of
/// [`multiplicity_free_multi_fmove_tree`] — same Stage 1 tail enumeration, but
/// coefficients are the `generic_multi_associator` vectors (see the convention
/// block above for the vector index).
fn generic_multi_fmove_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<GenericFmoveTerms<R::Scalar>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rank = tree.uncoupled().len();
    if rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "multi_Fmove requires at least one uncoupled sector",
        });
    }
    if rank == 1 {
        // TK `:218-220`: empty tail coupled to the unit, coeff `ones(T, 1)`.
        return Ok(vec![(
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                Some(rule.vacuum()),
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            )
            .with_has_multiplicity(true),
            vec![R::Scalar::braid_one()],
        )]);
    }
    if rank == 2 {
        // TK `:221-232`: single tail `(b,) → b`, coeff = unit vector `e_μ` over
        // the (unchanged) topmost vertex `a ⊗ b → c`, μ = tree.vertices[0].
        let a = tree.uncoupled()[0];
        let b = tree.uncoupled()[1];
        let c = coupled_or_vacuum(rule, tree);
        let n = rule.nsymbol(a, b, c);
        let mu0 = mu_index(tree, 0)?;
        let mut coeff = vec![R::Scalar::braid_zero(); n];
        if let Some(slot) = coeff.get_mut(mu0) {
            *slot = R::Scalar::braid_one();
        } else {
            return Err(CoreError::MalformedFusionTree {
                message: "multi_Fmove: vertex label exceeds Nsymbol",
            });
        }
        let tail = FusionTreeKey::new(
            vec![b],
            Some(b),
            vec![tree.is_dual()[1]],
            Vec::<SectorId>::new(),
            Vec::<SectorId>::new(),
        )
        .with_has_multiplicity(true);
        return Ok(vec![(tail, coeff)]);
    }

    let first = tree.uncoupled()[0];
    let coupled = coupled_or_vacuum(rule, tree);
    let tail_uncoupled = &tree.uncoupled()[1..];
    let tail_is_dual = &tree.is_dual()[1..];
    let mut terms = Vec::new();
    // `fusion_channels_in_table`: same clean-sector argument as the braid
    // above — frontier tail_coupled candidates are dead on clean trees.
    for tail_coupled in rule.fusion_channels_in_table(rule.dual(first), coupled) {
        let tail_effective = effective_sectors_for_uncoupled(rule, tail_uncoupled, tail_is_dual)?;
        for tail_tree in collect_generic_fusion_trees_for_coupled(
            rule,
            tail_uncoupled,
            tail_is_dual,
            &tail_effective,
            tail_coupled,
        ) {
            if let Some(coeff) = generic_multi_associator(rule, tree, &tail_tree)? {
                terms.push((tail_tree, coeff));
            }
        }
    }
    Ok(terms)
}

/// Generic-fusion `multi_Fmove_inv`: fuse a leading sector `a` onto an existing
/// tree (coupled `b`) to a coupled sector `c`, recoupling into standard-form
/// trees with per-tree coefficient vectors indexed by the topmost INPUT vertex
/// `a ⊗ b → c` (TK `:343-347`). Structural twin of
/// [`multiplicity_free_multi_fmove_inv_tree`].
///
/// Like the mult-free version, the per-candidate coefficient is the
/// `generic_multi_associator(candidate, tree)` vector, CONJUGATED. This is
/// exact because TensorKit's inverse Stage 2 applies the adjoint of the same
/// F-slices in the same order: with `Tₖ = transpose(view(F,:,ν,κ,:))` the
/// forward associator computes `v = Tₙ⋯T₂·seed`, while the inverse computes
/// `w = conj(Tₙ)⋯conj(T₃)·conj(T₂·seed) = conj(v)` (TK `:437-439, 460-462`,
/// the `conj!`/`'` on each factor). No separate inverse F-chain is needed.
fn generic_multi_fmove_inv_tree<R>(
    rule: &R,
    leading_sector: SectorId,
    coupled: SectorId,
    tree: &FusionTreeKey,
    leading_is_dual: bool,
) -> Result<GenericFmoveTerms<R::Scalar>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: GenericBraidScalar,
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
        collect_generic_fusion_trees_for_coupled(rule, &uncoupled, &is_dual, &effective, coupled);

    let mut terms = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if let Some(coeff) = generic_multi_associator(rule, &candidate, tree)? {
            terms.push((
                candidate,
                coeff.into_iter().map(|value| value.braid_conj()).collect(),
            ));
        }
    }
    Ok(terms)
}

/// Generic-fusion `foldright`: bend the first codomain vertex `a ⊗ b ← c` to a
/// domain vertex `b ← dual(a) ⊗ c`. Verbatim mirror of TensorKit `foldright`
/// GenericFusion branch (`duality_manipulations.jl:238-289`), especially the
/// coefficient-vector × A × coefficient-vector contraction at `:277-284`:
///   `coeff₀ · (coeff₂' · (transpose(A) · coeff₁))`.
/// Structural twin of [`multiplicity_free_foldright_tree_pair`], with the scalar
/// `coeff₁ · A · conj(coeff₂)` promoted to the vector–matrix–vector contraction
/// through the A-move matrix (which connects the two topmost `λ` vertices).
pub fn generic_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
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
    for (codomain_prime, coeff1) in generic_multi_fmove_tree(rule, codomain)? {
        let b = coupled_or_vacuum(rule, &codomain_prime);
        // A = Asymbol(a, b, c): rows = topmost codomain vertex λ₁ ∈ N(a,b,c)
        // (indexes coeff1), cols = topmost domain vertex λ₂ ∈ N(dual(a),c,b)
        // (indexes coeff2). `a_symbol_generic` already bakes in κ_a and the
        // outer conj per TK `Asymbol_from_Fsymbol`.
        let a_matrix = rule.a_symbol_generic(a, b, c);
        let (rows, cols) = a_matrix.shape();
        let coeff0 = rule.sqrt_dim_scalar(c) * rule.inv_sqrt_dim_scalar(b);
        for (domain_prime, coeff2) in generic_multi_fmove_inv_tree(
            rule,
            rule.dual(a),
            b,
            tree_pair.domain_tree(),
            !is_dual_a,
        )? {
            if coeff1.len() != rows || coeff2.len() != cols {
                return Err(CoreError::MalformedFusionTree {
                    message: "foldright: coefficient-vector length disagrees with A-matrix shape",
                });
            }
            // coeff₂' · (transpose(A) · coeff₁)
            //   = Σ_j conj(coeff₂[j]) · Σ_i A[i,j] · coeff₁[i].
            let mut inner = R::Scalar::braid_zero();
            for (j, coeff2_j) in coeff2.iter().enumerate() {
                let mut a_transpose_coeff1 = R::Scalar::braid_zero();
                for (i, coeff1_i) in coeff1.iter().enumerate() {
                    a_transpose_coeff1 =
                        a_transpose_coeff1 + a_matrix.get(i, j).clone() * coeff1_i.clone();
                }
                inner = inner + coeff2_j.braid_conj() * a_transpose_coeff1;
            }
            let mut coefficient = coeff0.clone() * inner;
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

/// Generic-fusion `foldleft` = swap + conjugate of `foldright`, verbatim mirror
/// of TensorKit `foldleft((f₁,f₂))` (`duality_manipulations.jl:315-319`).
/// Structural twin of [`multiplicity_free_foldleft_tree_pair`].
pub fn generic_foldleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let swapped = FusionTreeBlockKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(generic_foldright_tree_pair(rule, &swapped)?
        .into_iter()
        .map(|(folded, coefficient)| {
            (
                FusionTreeBlockKey::pair(
                    folded.domain_tree().clone(),
                    folded.codomain_tree().clone(),
                ),
                coefficient.braid_conj(),
            )
        })
        .collect())
}

/// Generic-fusion `cycleclockwise` = foldright ∘ bendleft (or the reverse order
/// when the codomain is empty), composing coefficient matrices. Verbatim mirror
/// of TensorKit `cycleclockwise` (`duality_manipulations.jl:401-410`) and
/// structural twin of [`multiplicity_free_cycle_clockwise_tree_pair`].
pub fn generic_cycle_clockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if tree_pair.codomain_tree().uncoupled().is_empty() {
        let first = generic_bendleft_tree_pair(rule, tree_pair)?;
        compose_generic_tree_pair_terms(rule, first, |rule, key| {
            generic_foldright_tree_pair(rule, key)
        })
    } else {
        let first = generic_foldright_tree_pair(rule, tree_pair)?;
        compose_generic_tree_pair_terms(rule, first, |rule, key| {
            generic_bendleft_tree_pair(rule, key)
        })
    }
}

/// Generic-fusion `cycleanticlockwise` = foldleft ∘ bendright (or the reverse
/// order when the domain is empty). Verbatim mirror of TensorKit
/// `cycleanticlockwise` (`duality_manipulations.jl:431-440`) and structural
/// twin of [`multiplicity_free_cycle_anticlockwise_tree_pair`].
pub fn generic_cycle_anticlockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if tree_pair.domain_tree().uncoupled().is_empty() {
        let first = generic_bendright_tree_pair(rule, tree_pair)?;
        compose_generic_tree_pair_terms(rule, first, |rule, key| {
            generic_foldleft_tree_pair(rule, key)
        })
    } else {
        let first = generic_foldleft_tree_pair(rule, tree_pair)?;
        compose_generic_tree_pair_terms(rule, first, |rule, key| {
            generic_bendright_tree_pair(rule, key)
        })
    }
}

/// Generic-fusion sibling of [`multiplicity_free_repartition_terms`]: repartition
/// a whole term list to `target_codomain_rank` legs, composing the bend
/// coefficient matrices. Same accumulate-and-compose loop, different rule bound.
fn generic_repartition_terms<R>(
    rule: &R,
    terms: Vec<(FusionTreeBlockKey, R::Scalar)>,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
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
        current = compose_generic_tree_pair_terms(rule, current, |rule, key| {
            generic_bendleft_tree_pair(rule, key)
        })?;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        current = compose_generic_tree_pair_terms(rule, current, |rule, key| {
            generic_bendright_tree_pair(rule, key)
        })?;
        current_codomain_rank -= 1;
    }
    Ok(current)
}

/// Generic-fusion `braid` on a full tree pair: bend everything into the codomain,
/// braid there, bend back — the TensorKit `braid`/`fsbraid` decomposition.
/// Structural twin of [`multiplicity_free_braid_tree_pair`] (:829): the only
/// difference is the primitive family (`generic_repartition_tree_pair` /
/// `generic_braid_tree` / `generic_repartition_terms`) and the `braid_one` seed;
/// no new recoupling formula is introduced.
pub fn generic_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
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
    let mut current = generic_repartition_tree_pair(rule, tree_pair, all_rank)?;
    current = compose_generic_tree_pair_terms(rule, current, |rule, key| {
        generic_braid_tree(rule, key.codomain_tree(), &permutation, &levels).map(|terms| {
            terms
                .into_iter()
                .map(|(codomain_tree, coefficient)| {
                    (
                        FusionTreeBlockKey::pair(codomain_tree, key.domain_tree().clone()),
                        coefficient,
                    )
                })
                .collect::<Vec<_>>()
        })
    })?;
    generic_repartition_terms(rule, current, codomain_permutation.len())
}

/// Generic-fusion `permute` = [`generic_braid_tree_pair`] with the identity
/// level order (symmetric braiding only). Structural twin of
/// [`multiplicity_free_permute_tree_pair`] (:886).
pub fn generic_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
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
    generic_braid_tree_pair(
        rule,
        tree_pair,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

/// Generic-fusion `transpose` (planar cyclic permutation): bend into the target
/// partition, then cycle the coupled tree into place via fold/bend. Structural
/// twin of [`multiplicity_free_transpose_tree_pair`] (:916); braid-free, so it
/// runs on planar (non-symmetric) Generic rules too.
pub fn generic_transpose_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
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
        None => return Ok(vec![(tree_pair.clone(), R::Scalar::braid_one())]),
    };
    let mut current =
        generic_repartition_tree_pair(rule, tree_pair, codomain_permutation.len())?;
    let total_rank = codomain_rank + domain_rank;
    if total_rank == 0 || position == 0 {
        return Ok(current);
    }

    let half_rank = total_rank >> 1;
    while position > 0 && position < half_rank {
        current = compose_generic_tree_pair_terms(rule, current, |rule, key| {
            generic_cycle_anticlockwise_tree_pair(rule, key)
        })?;
        position -= 1;
    }
    while position >= half_rank && position > 0 {
        current = compose_generic_tree_pair_terms(rule, current, |rule, key| {
            generic_cycle_clockwise_tree_pair(rule, key)
        })?;
        position = (position + 1) % total_rank;
    }

    Ok(current)
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

    // Constructive inverse (TensorKit `multi_Fmove_inv`,
    // `basic_manipulations.jl`): every candidate shares `tree`'s uncoupled
    // tail, so the expansion coefficient is directly the conjugated
    // multi-associator relating the (N+1)-leg candidate to the N-leg `tree` —
    // exactly the F-symbol product the forward `multi_Fmove` would assign that
    // tail. Reuse it instead of running the full forward recoupling on every
    // candidate and searching for the matching tail (was invert-by-search,
    // O(candidates²·N); now O(candidates·N)). The two are term-for-term
    // identical: `multi_associator_scalar` returns `Some` iff the uncoupled/dual
    // tails match — which they do for every enumerated candidate — with the same
    // coefficient (zeros included) the forward pass produced.
    let mut terms = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if let Some(coefficient) =
            multiplicity_free_multi_associator_scalar(rule, &candidate, tree)?
        {
            terms.push((candidate, rule.scalar_conj(coefficient)));
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
