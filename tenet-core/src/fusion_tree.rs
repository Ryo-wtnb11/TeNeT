#[derive(Clone, Debug, Eq, PartialEq)]
struct CoupledFusionTrees {
    coupled: SectorId,
    trees: Vec<FusionTreeKey>,
}

fn validate_multiplicity_free_execution_style<R>(rule: &R) -> Result<(), CoreError>
where
    R: FusionRule,
{
    if rule.fusion_style().is_multiplicity_free() {
        Ok(())
    } else {
        Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: rule.fusion_style(),
        })
    }
}

fn try_visit_selected_leg_tuples<E, F>(
    legs: &[SectorLeg],
    remaining: usize,
    current: &mut [FusionTreeLeg],
    emit: &mut F,
) -> Result<(), E>
where
    F: FnMut(&[FusionTreeLeg]) -> Result<(), E>,
{
    if remaining == 0 {
        return emit(current);
    }

    let index = remaining - 1;
    for &sector in legs[index].sectors() {
        current[index] = FusionTreeLeg::new(sector, legs[index].is_dual());
        try_visit_selected_leg_tuples(legs, remaining - 1, current, emit)?;
    }
    Ok(())
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
    let mut uncoupled = Vec::with_capacity(space.len());
    let mut is_dual = Vec::with_capacity(space.len());
    let mut effective = Vec::with_capacity(space.len());
    let result: Result<(), std::convert::Infallible> =
        space.try_visit_selected_leg_tuples(&mut |tuple| {
            uncoupled.clear();
            is_dual.clear();
            effective.clear();
            for leg in tuple {
                uncoupled.push(leg.sector());
                is_dual.push(leg.is_dual());
                effective.push(leg.sector());
            }
            for coupled in reachable_coupled_sectors(rule, &effective) {
                let trees = collect_fusion_trees_for_coupled(
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
            Ok(())
        });
    match result {
        Ok(()) => {}
        Err(never) => match never {},
    }
    grouped.sort_by_key(|group| group.coupled);
    grouped
}

#[derive(Clone, Copy)]
struct LoweredFusionTreeLeg<S> {
    encoded: SectorId,
    lowered: S,
    is_dual: bool,
}

struct LoweredReachableWorkspace<S>
where
    S: Eq + Hash,
{
    acc: Vec<S>,
    next: Vec<S>,
    seen: rustc_hash::FxHashSet<S>,
}

impl<S> Default for LoweredReachableWorkspace<S>
where
    S: Eq + Hash,
{
    fn default() -> Self {
        Self {
            acc: Vec::new(),
            next: Vec::new(),
            seen: rustc_hash::FxHashSet::default(),
        }
    }
}

#[cfg(test)]
thread_local! {
    static LOWERED_LAYOUT_BUILD_OBSERVATIONS: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

#[cfg(test)]
fn reset_lowered_layout_build_observations() {
    LOWERED_LAYOUT_BUILD_OBSERVATIONS.with(|observation| observation.set((0, 0)));
}

#[cfg(test)]
fn lowered_layout_build_observations() -> (usize, usize) {
    LOWERED_LAYOUT_BUILD_OBSERVATIONS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn observe_lowered_layout_decode() {
    LOWERED_LAYOUT_BUILD_OBSERVATIONS.with(|observation| {
        let (decodes, channels) = observation.get();
        observation.set((decodes + 1, channels));
    });
}

#[cfg(test)]
fn observe_lowered_layout_channel() {
    LOWERED_LAYOUT_BUILD_OBSERVATIONS.with(|observation| {
        let (decodes, channels) = observation.get();
        observation.set((decodes, channels + 1));
    });
}

fn try_fusion_trees_by_coupled_for_space_lowered<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> Result<Vec<CoupledFusionTrees>, LoweredFusionTreeBuildError>
where
    R: LoweredMultiplicityFreeAlgebra,
{
    if space.legs().iter().any(|leg| leg.sectors().is_empty()) {
        return Ok(Vec::new());
    }

    let choices = space
        .legs()
        .iter()
        .map(|leg| {
            leg.sectors()
                .iter()
                .map(|&encoded| {
                    #[cfg(test)]
                    observe_lowered_layout_decode();
                    Ok(LoweredFusionTreeLeg {
                        encoded,
                        lowered: rule.try_decode_lowered(encoded)?,
                        is_dual: leg.is_dual(),
                    })
                })
                .collect::<Result<Vec<_>, LoweredFusionTreeBuildError>>()
        })
        .collect::<Result<Vec<_>, LoweredFusionTreeBuildError>>()?;

    struct LoweredGroup<S> {
        coupled: S,
        trees: Vec<FusionTreeKey>,
    }

    let mut grouped = Vec::<LoweredGroup<R::Sector>>::new();
    let mut index = FxHashMap::<R::Sector, usize>::default();
    let mut current = vec![None; choices.len()];
    let mut uncoupled = Vec::with_capacity(choices.len());
    let mut is_dual = Vec::with_capacity(choices.len());
    let mut effective = Vec::with_capacity(choices.len());
    let mut reachable = LoweredReachableWorkspace::default();
    visit_lowered_leg_tuples(
        &choices,
        choices.len(),
        &mut current,
        &mut |tuple| {
            uncoupled.clear();
            is_dual.clear();
            effective.clear();
            for &leg in tuple {
                let leg = leg.expect("lowered tuple must be assigned");
                uncoupled.push(leg.encoded);
                is_dual.push(leg.is_dual);
                effective.push(leg.lowered);
            }
            try_for_each_reachable_coupled_lowered(
                rule,
                &effective,
                &mut reachable,
                &mut |coupled| {
                let group = match index.get(&coupled) {
                    Some(&group) => group,
                    None => {
                        let group = grouped.len();
                        index.insert(coupled, group);
                        grouped.push(LoweredGroup {
                            coupled,
                            trees: Vec::new(),
                        });
                        group
                    }
                };
                try_append_fusion_trees_for_coupled_lowered(
                    rule,
                    &uncoupled,
                    &is_dual,
                    &effective,
                    coupled,
                    &mut grouped[group].trees,
                )
            },
            )?;
            Ok(())
        },
    )?;

    let mut encoded = grouped
        .into_iter()
        .map(|group| {
            Ok(CoupledFusionTrees {
                coupled: rule.try_encode_lowered(group.coupled)?,
                trees: group.trees,
            })
        })
        .collect::<Result<Vec<_>, LoweredFusionTreeBuildError>>()?;
    encoded.sort_by_key(|group| group.coupled);
    Ok(encoded)
}

fn visit_lowered_leg_tuples<S, F>(
    choices: &[Vec<LoweredFusionTreeLeg<S>>],
    remaining: usize,
    current: &mut [Option<LoweredFusionTreeLeg<S>>],
    emit: &mut F,
) -> Result<(), LoweredFusionTreeBuildError>
where
    S: Copy,
    F: FnMut(&[Option<LoweredFusionTreeLeg<S>>]) -> Result<(), LoweredFusionTreeBuildError>,
{
    if remaining == 0 {
        return emit(current);
    }
    let index = remaining - 1;
    for &leg in &choices[index] {
        current[index] = Some(leg);
        visit_lowered_leg_tuples(choices, remaining - 1, current, emit)?;
    }
    Ok(())
}

fn try_for_each_reachable_coupled_lowered<R, F>(
    rule: &R,
    effective: &[R::Sector],
    workspace: &mut LoweredReachableWorkspace<R::Sector>,
    emit: &mut F,
) -> Result<(), LoweredFusionTreeBuildError>
where
    R: LoweredMultiplicityFreeAlgebra,
    F: FnMut(R::Sector) -> Result<(), LoweredFusionTreeBuildError>,
{
    workspace.acc.clear();
    workspace.next.clear();
    workspace.seen.clear();
    workspace.acc.push(match effective.first() {
        None => rule.try_lowered_vacuum()?,
        Some(&first) => first,
    });
    for &last in effective.iter().skip(1) {
        workspace.next.clear();
        workspace.seen.clear();
        for &front in &workspace.acc {
            #[cfg(test)]
            observe_lowered_layout_channel();
            rule.try_for_each_lowered_channel(front, last, &mut |coupled| {
                if workspace.seen.insert(coupled) {
                    workspace.next.push(coupled);
                }
                Ok(())
            })?;
        }
        std::mem::swap(&mut workspace.acc, &mut workspace.next);
    }
    for &coupled in &workspace.acc {
        emit(coupled)?;
    }
    Ok(())
}

fn try_append_fusion_trees_for_coupled_lowered<R>(
    rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[R::Sector],
    coupled: R::Sector,
    out: &mut Vec<FusionTreeKey>,
) -> Result<(), LoweredFusionTreeBuildError>
where
    R: LoweredMultiplicityFreeAlgebra,
{
    let coupled_encoded = rule.try_encode_lowered(coupled)?;
    let vertex_count = uncoupled.len().saturating_sub(1);
    let mut inner_rev = SmallVec::<[R::Sector; 8]>::new();
    try_visit_fusion_trees_lowered(
        rule,
        effective,
        coupled,
        &mut inner_rev,
        &mut |inner_rev| {
            let mut innerlines = SectorVec::new();
            for &sector in inner_rev.iter().rev() {
                innerlines.push(rule.try_encode_lowered(sector)?);
            }
            out.push(FusionTreeKey::new(
                uncoupled.iter().copied(),
                coupled_encoded,
                is_dual.iter().copied(),
                innerlines,
                std::iter::repeat_n(MultiplicityIndex::ONE, vertex_count),
            ));
            Ok(())
        },
    )
}

fn try_visit_fusion_trees_lowered<R, F>(
    rule: &R,
    effective: &[R::Sector],
    coupled: R::Sector,
    inner_rev: &mut SmallVec<[R::Sector; 8]>,
    emit: &mut F,
) -> Result<(), LoweredFusionTreeBuildError>
where
    R: LoweredMultiplicityFreeAlgebra,
    F: FnMut(&[R::Sector]) -> Result<(), LoweredFusionTreeBuildError>,
{
    match effective.len() {
        0 => {
            if coupled == rule.try_lowered_vacuum()? {
                emit(inner_rev)?;
            }
        }
        1 => {
            if effective[0] == coupled {
                emit(inner_rev)?;
            }
        }
        2 => {
            if rule.try_lowered_nsymbol(effective[0], effective[1], coupled)? != 0 {
                emit(inner_rev)?;
            }
        }
        _ => {
            let last = effective[effective.len() - 1];
            let dual_last = rule.try_lowered_dual(last)?;
            let front_effective = &effective[..effective.len() - 1];
            #[cfg(test)]
            observe_lowered_layout_channel();
            rule.try_for_each_lowered_channel(coupled, dual_last, &mut |front_coupled| {
                if rule.try_lowered_nsymbol(front_coupled, last, coupled)? == 0 {
                    return Ok(());
                }
                inner_rev.push(front_coupled);
                let result = try_visit_fusion_trees_lowered(
                    rule,
                    front_effective,
                    front_coupled,
                    inner_rev,
                    emit,
                );
                inner_rev.pop();
                result
            })?;
        }
    }
    Ok(())
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
    let mut uncoupled = Vec::with_capacity(space.len());
    let mut is_dual = Vec::with_capacity(space.len());
    let mut effective = Vec::with_capacity(space.len());
    let result: Result<(), std::convert::Infallible> =
        space.try_visit_selected_leg_tuples(&mut |tuple| {
            // `effective_sectors` is the uncoupled sectors verbatim (it ignores the
            // rule); inlined here to avoid its mult-free bound.
            uncoupled.clear();
            is_dual.clear();
            effective.clear();
            for leg in tuple {
                uncoupled.push(leg.sector());
                is_dual.push(leg.is_dual());
                effective.push(leg.sector());
            }
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
            Ok(())
        });
    match result {
        Ok(()) => {}
        Err(never) => match never {},
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
#[cfg(test)]
fn merge_generic_tree_groups(
    codomain: &[CoupledFusionTrees],
    domain: &[CoupledFusionTrees],
) -> Vec<FusionTreePairKey> {
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
                        keys.push(FusionTreePairKey::pair(
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
            coupled,
            is_dual.iter().copied(),
            inner_rev.iter().rev().copied(),
            std::iter::repeat_n(MultiplicityIndex::ONE, vertex_count),
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
    visit_fusion_trees_where(
        rule,
        effective,
        coupled,
        inner_rev,
        &|_, _| true,
        emit,
    );
}

fn visit_fusion_trees_where<R, P, F>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
    inner_rev: &mut Vec<SectorId>,
    prefix_allowed: &P,
    emit: &mut F,
) where
    R: MultiplicityFreeFusionRule,
    P: Fn(usize, SectorId) -> bool,
    F: FnMut(&[SectorId]),
{
    if !prefix_allowed(effective.len(), coupled) {
        return;
    }
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
                visit_fusion_trees_where(
                    rule,
                    front_effective,
                    front_coupled,
                    inner_rev,
                    prefix_allowed,
                    emit,
                );
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

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreeKey {
    uncoupled: SectorVec,
    coupled: SectorId,
    is_dual: DualVec,
    innerlines: SectorVec,
    vertices: MultiplicityVec,
}

impl FusionTreeKey {
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
        Self {
            uncoupled: uncoupled.into_iter().collect(),
            coupled,
            is_dual: is_dual.into_iter().collect(),
            innerlines: innerlines.into_iter().collect(),
            vertices: vertices.into_iter().collect(),
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
    /// by [`OpaqueBlockKey`](crate::OpaqueBlockKey), not raw fusion trees.
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

/// Split a left-associated fusion tree using TensorKit's `split(f, m)`
/// convention.
///
/// The first output contains the first `front_rank` uncoupled sectors. The
/// second output starts with the intermediate sector between the two pieces and
/// then contains the remaining uncoupled sectors. This is a structural
/// categorical operation: no dense storage is touched and no coefficient is
/// introduced.
///
/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
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
    validate_fusion_tree_for_rule(rule, tree)?;

    if front_rank == rank {
        let coupled = tree.coupled();
        let trace_tree = FusionTreeKey::new(
            [coupled],
            coupled,
            [false],
            Vec::<SectorId>::new(),
            Vec::<MultiplicityIndex>::new(),
        );
        return Ok((tree.clone(), trace_tree));
    }

    if front_rank == 1 {
        let first = tree.uncoupled()[0];
        let front_tree = FusionTreeKey::new(
            [first],
            first,
            [tree.is_dual()[0]],
            Vec::<SectorId>::new(),
            Vec::<MultiplicityIndex>::new(),
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
            unit,
            Vec::<bool>::new(),
            Vec::<SectorId>::new(),
            Vec::<MultiplicityIndex>::new(),
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
        tail_vertices.push(MultiplicityIndex::ONE);
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
        intermediate,
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

/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
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

/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
pub fn unique_artin_braid_at<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
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
    validate_fusion_tree_for_rule(rule, tree)?;
    unique_artin_braid_at_with_inverse(rule, tree, index, false)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedArtinStep {
    index: usize,
    inverse: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedTreeBraid {
    permutation: SmallVec<[usize; 8]>,
    artin_steps: SmallVec<[PreparedArtinStep; 28]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UniqueBorrowedBraidLevels<'operation> {
    Symmetric,
    Explicit {
        codomain: &'operation [usize],
        domain: &'operation [usize],
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct UniqueBorrowedTreePairBraid<'operation> {
    codomain_permutation: &'operation [usize],
    domain_permutation: &'operation [usize],
    source_codomain_rank: usize,
    source_domain_rank: usize,
    raw_axis_positions: Option<&'operation [usize]>,
    levels: UniqueBorrowedBraidLevels<'operation>,
}

#[cfg(test)]
thread_local! {
    static UNIQUE_BORROWED_POSITION_QUERIES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reset_unique_borrowed_position_queries() {
    UNIQUE_BORROWED_POSITION_QUERIES.with(|queries| queries.set(0));
}

#[cfg(test)]
fn unique_borrowed_position_queries() -> usize {
    UNIQUE_BORROWED_POSITION_QUERIES.with(std::cell::Cell::get)
}

impl<'operation> UniqueBorrowedTreePairBraid<'operation> {
    fn permutation_at(&self, position: usize) -> usize {
        linearized_tree_pair_axis_at(
            self.codomain_permutation,
            self.domain_permutation,
            self.source_codomain_rank,
            self.source_domain_rank,
            position,
        )
    }

    fn linear_position_of_axis(&self, axis: usize) -> usize {
        #[cfg(test)]
        if self.raw_axis_positions.is_some() {
            UNIQUE_BORROWED_POSITION_QUERIES.with(|queries| queries.set(queries.get() + 1));
        }
        let logical_axis = if axis < self.source_codomain_rank {
            axis
        } else {
            self.source_codomain_rank + self.source_domain_rank + self.source_codomain_rank
                - 1
                - axis
        };
        // Why not materialize the linear permutation here: the operation
        // compiler already owns this inverse table, while direct core callers
        // intentionally retain their allocation-free compatibility path.
        let raw_position = self.raw_axis_positions.map_or_else(
            || {
                self.codomain_permutation
                    .iter()
                    .chain(self.domain_permutation)
                    .position(|&candidate| candidate == logical_axis)
                    .expect("validated tree-pair axis map contains every logical axis")
            },
            |positions| positions[logical_axis],
        );
        if raw_position < self.codomain_permutation.len() {
            raw_position
        } else {
            self.codomain_permutation.len()
                + self.source_codomain_rank
                + self.source_domain_rank
                - 1
                - raw_position
        }
    }

    fn level_at(&self, axis: usize) -> usize {
        match self.levels {
            UniqueBorrowedBraidLevels::Symmetric => {
                if axis < self.source_codomain_rank {
                    axis
                } else {
                    self.source_codomain_rank + self.source_domain_rank - 1 - axis
                        + self.source_codomain_rank
                }
            }
            UniqueBorrowedBraidLevels::Explicit { codomain, domain } => {
                if axis < self.source_codomain_rank {
                    codomain[axis]
                } else {
                    domain[self.source_codomain_rank + self.source_domain_rank - 1 - axis]
                }
            }
        }
    }

    fn artin_steps(&self) -> UniqueBorrowedArtinSteps<'_> {
        UniqueBorrowedArtinSteps {
            braid: self,
            target: 0,
            source_axis: None,
            candidate_exclusive: 0,
            next_index: 0,
        }
    }
}

struct UniqueBorrowedArtinSteps<'operation> {
    braid: &'operation UniqueBorrowedTreePairBraid<'operation>,
    target: usize,
    source_axis: Option<usize>,
    candidate_exclusive: usize,
    next_index: usize,
}

impl Iterator for UniqueBorrowedArtinSteps<'_> {
    type Item = PreparedArtinStep;

    fn next(&mut self) -> Option<Self::Item> {
        let rank = self.braid.source_codomain_rank + self.braid.source_domain_rank;
        loop {
            if let Some(source_axis) = self.source_axis {
                while self.candidate_exclusive > 0 {
                    self.candidate_exclusive -= 1;
                    let candidate = self.candidate_exclusive;
                    if self.braid.linear_position_of_axis(candidate) > self.target {
                        self.next_index -= 1;
                        return Some(PreparedArtinStep {
                            index: self.next_index,
                            inverse: self.braid.level_at(candidate)
                                > self.braid.level_at(source_axis),
                        });
                    }
                }
                self.source_axis = None;
                self.target += 1;
                continue;
            }
            if self.target >= rank.saturating_sub(1) {
                return None;
            }
            let source_axis = self.braid.permutation_at(self.target);
            let crossings = (0..source_axis)
                .filter(|&candidate| self.braid.linear_position_of_axis(candidate) > self.target)
                .count();
            if crossings == 0 {
                self.target += 1;
            } else {
                self.source_axis = Some(source_axis);
                self.candidate_exclusive = source_axis;
                self.next_index = self.target + crossings;
            }
        }
    }
}

impl PreparedTreeBraid {
    fn new(permutation: &[usize], levels: &[usize], rank: usize) -> Result<Self, CoreError> {
        validate_permutation_inline(permutation, rank)?;
        debug_assert_eq!(levels.len(), rank);

        let mut work = SmallVec::<[usize; 8]>::from_slice(permutation);
        let mut current_levels = SmallVec::<[usize; 8]>::from_slice(levels);
        let mut artin_steps = SmallVec::new();
        for target in 0..rank.saturating_sub(1) {
            let source = work[target];
            for index in (target..source).rev() {
                artin_steps.push(PreparedArtinStep {
                    index,
                    inverse: current_levels[index] > current_levels[index + 1],
                });
                current_levels.swap(index, index + 1);
            }
            for item in work.iter_mut().take(rank).skip(target + 1) {
                if *item < source {
                    *item += 1;
                }
            }
            work[target] = target;
        }

        Ok(Self {
            permutation: SmallVec::from_slice(permutation),
            artin_steps,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedCycleDirection {
    Clockwise,
    Anticlockwise,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedTreePairFamily {
    BraidLike,
    Transpose,
}

const PREPARED_BRAID_BLOCK_FAMILY_ERROR: &str =
    "prepared tree-pair operation is incompatible with braid block execution";
const PREPARED_TRANSPOSE_BLOCK_FAMILY_ERROR: &str =
    "prepared tree-pair operation is incompatible with transpose block execution";

// Why not box the braid variant: rank<=8 preparation is intentionally
// allocation-free, and this expert plan is reused instead of copied per tree.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Eq, PartialEq)]
enum PreparedTreePairPlan<'operation> {
    Identity,
    Repartition,
    Braid(PreparedTreeBraid),
    UniqueBraid(UniqueBorrowedTreePairBraid<'operation>),
    Transpose {
        direction: PreparedCycleDirection,
        count: usize,
    },
}

enum PreparedTreePairArtinSteps<'operation> {
    Owned(std::slice::Iter<'operation, PreparedArtinStep>),
    Unique(UniqueBorrowedArtinSteps<'operation>),
}

impl Iterator for PreparedTreePairArtinSteps<'_> {
    type Item = PreparedArtinStep;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Owned(steps) => steps.next().copied(),
            Self::Unique(steps) => steps.next(),
        }
    }
}

impl PreparedTreePairPlan<'_> {
    fn artin_steps(&self) -> Option<PreparedTreePairArtinSteps<'_>> {
        match self {
            Self::Braid(braid) => Some(PreparedTreePairArtinSteps::Owned(braid.artin_steps.iter())),
            Self::UniqueBraid(braid) => Some(PreparedTreePairArtinSteps::Unique(braid.artin_steps())),
            Self::Identity | Self::Repartition | Self::Transpose { .. } => None,
        }
    }
}

/// Rank-dependent preparation for one fusion-tree-pair operation.
///
/// The tensor-plan compiler may prepare this once and execute it for every
/// tree in one structural block. The representation is intentionally opaque:
/// operation validation and braid/cycle lowering are core semantics.
///
/// Why not expose the individual Artin or cycle steps: doing so would let
/// downstream crates duplicate TensorKit's domain reversal and inverse-level
/// conventions, recreating the semantic split this type removes.
#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedTreePairOperation<'operation> {
    source_codomain_rank: usize,
    source_domain_rank: usize,
    target_codomain_rank: usize,
    requires_symmetric_braiding: bool,
    family: PreparedTreePairFamily,
    plan: PreparedTreePairPlan<'operation>,
}

impl<'operation> PreparedTreePairOperation<'operation> {
    pub(crate) fn is_identity(&self) -> bool {
        matches!(self.plan, PreparedTreePairPlan::Identity)
    }

    /// Validate operation metadata that depends only on total tensor rank.
    ///
    /// The exact codomain/domain split is categorical information and remains
    /// the responsibility of the operation-specific validators below.
    #[doc(hidden)]
    pub fn validate_rank_syntax(
        total_rank: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<(), CoreError> {
        validate_tree_pair_axis_map_inline(
            codomain_permutation,
            domain_permutation,
            total_rank,
            0,
        )
    }

    pub fn prepare_braid<R>(
        rule: &R,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
        codomain_levels: &'operation [usize],
        domain_levels: &'operation [usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        Self::validate_braid_level_lengths(
            source_codomain_rank,
            source_domain_rank,
            codomain_levels,
            domain_levels,
        )?;
        if !rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: rule.fusion_style(),
            });
        }
        Self::prepare_braid_lowered(
            rule.fusion_style() == FusionStyleKind::Unique,
            None,
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        )
    }

    #[doc(hidden)]
    pub fn prepare_braid_with_raw_axis_positions<R>(
        rule: &R,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
        codomain_levels: &'operation [usize],
        domain_levels: &'operation [usize],
        raw_axis_positions: &'operation [usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        Self::validate_braid_level_lengths(
            source_codomain_rank,
            source_domain_rank,
            codomain_levels,
            domain_levels,
        )?;
        if !rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: rule.fusion_style(),
            });
        }
        Self::prepare_braid_lowered(
            rule.fusion_style() == FusionStyleKind::Unique,
            Some(raw_axis_positions),
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        )
    }

    #[doc(hidden)]
    pub fn validate_braid_syntax(
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        codomain_levels: &[usize],
        domain_levels: &[usize],
    ) -> Result<(), CoreError> {
        Self::validate_braid_level_lengths(
            source_codomain_rank,
            source_domain_rank,
            codomain_levels,
            domain_levels,
        )?;
        validate_tree_pair_axis_map_inline(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        )
    }

    fn prepare_braid_lowered(
        unique: bool,
        raw_axis_positions: Option<&'operation [usize]>,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
        codomain_levels: &'operation [usize],
        domain_levels: &'operation [usize],
    ) -> Result<Self, CoreError> {
        let target_codomain_rank = codomain_permutation.len();
        if tree_pair_axis_map_is_identity(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        ) {
            Self::validate_raw_axis_positions(
                raw_axis_positions,
                codomain_permutation,
                domain_permutation,
                source_codomain_rank + source_domain_rank,
            )?;
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: false,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::Identity,
            });
        }

        let total_rank = source_codomain_rank + source_domain_rank;
        if unique {
            validate_tree_pair_axis_map_without_scratch(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
            )?;
        } else {
            validate_tree_pair_axis_map_inline(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
            )?;
        }
        Self::validate_raw_axis_positions(
            raw_axis_positions,
            codomain_permutation,
            domain_permutation,
            total_rank,
        )?;
        if (0..total_rank).all(|position| {
            linearized_tree_pair_axis_at(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
                position,
            ) == position
        }) {
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: false,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::Repartition,
            });
        }

        if unique {
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: false,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::UniqueBraid(UniqueBorrowedTreePairBraid {
                    codomain_permutation,
                    domain_permutation,
                    source_codomain_rank,
                    source_domain_rank,
                    raw_axis_positions,
                    levels: UniqueBorrowedBraidLevels::Explicit {
                        codomain: codomain_levels,
                        domain: domain_levels,
                    },
                }),
            });
        }
        let permutation = materialize_linearized_tree_pair_permutation(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        );
        let mut levels = SmallVec::<[usize; 8]>::with_capacity(total_rank);
        levels.extend_from_slice(codomain_levels);
        levels.extend(domain_levels.iter().rev().copied());
        let braid = PreparedTreeBraid::new(&permutation, &levels, total_rank)?;
        Ok(Self {
            source_codomain_rank,
            source_domain_rank,
            target_codomain_rank,
            requires_symmetric_braiding: false,
            family: PreparedTreePairFamily::BraidLike,
            plan: PreparedTreePairPlan::Braid(braid),
        })
    }

    fn validate_braid_level_lengths(
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_levels: &[usize],
        domain_levels: &[usize],
    ) -> Result<(), CoreError> {
        if codomain_levels.len() != source_codomain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: source_codomain_rank,
                actual: codomain_levels.len(),
            });
        }
        if domain_levels.len() != source_domain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: source_domain_rank,
                actual: domain_levels.len(),
            });
        }
        Ok(())
    }

    fn validate_raw_axis_positions(
        raw_axis_positions: Option<&[usize]>,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        total_rank: usize,
    ) -> Result<(), CoreError> {
        if let Some(raw_axis_positions) = raw_axis_positions {
            if raw_axis_positions.len() != total_rank {
                return Err(CoreError::DimensionMismatch {
                    expected: total_rank,
                    actual: raw_axis_positions.len(),
                });
            }
            for position in 0..total_rank {
                let logical_axis = raw_tree_pair_axis_at(
                    codomain_permutation,
                    domain_permutation,
                    position,
                );
                if raw_axis_positions[logical_axis] != position {
                    return Err(CoreError::InconsistentAxisPosition {
                        logical_axis,
                        expected_position: position,
                        actual_position: raw_axis_positions[logical_axis],
                    });
                }
            }
        }
        Ok(())
    }

    pub fn prepare_permute<R>(
        rule: &R,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        if !rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: rule.braiding_style(),
            });
        }
        if !rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: rule.fusion_style(),
            });
        }
        Self::prepare_permute_lowered(
            rule.fusion_style() == FusionStyleKind::Unique,
            None,
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
        )
    }

    #[doc(hidden)]
    pub fn prepare_permute_with_raw_axis_positions<R>(
        rule: &R,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
        raw_axis_positions: &'operation [usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        if !rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: rule.braiding_style(),
            });
        }
        if !rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: rule.fusion_style(),
            });
        }
        Self::prepare_permute_lowered(
            rule.fusion_style() == FusionStyleKind::Unique,
            Some(raw_axis_positions),
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
        )
    }

    #[doc(hidden)]
    pub fn validate_permute_syntax(
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<(), CoreError> {
        validate_tree_pair_axis_map_inline(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        )
    }

    fn prepare_permute_lowered(
        unique: bool,
        raw_axis_positions: Option<&'operation [usize]>,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
    ) -> Result<Self, CoreError> {
        let target_codomain_rank = codomain_permutation.len();
        if tree_pair_axis_map_is_identity(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        ) {
            Self::validate_raw_axis_positions(
                raw_axis_positions,
                codomain_permutation,
                domain_permutation,
                source_codomain_rank + source_domain_rank,
            )?;
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: true,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::Identity,
            });
        }

        let total_rank = source_codomain_rank + source_domain_rank;
        if unique {
            validate_tree_pair_axis_map_without_scratch(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
            )?;
        } else {
            validate_tree_pair_axis_map_inline(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
            )?;
        }
        Self::validate_raw_axis_positions(
            raw_axis_positions,
            codomain_permutation,
            domain_permutation,
            total_rank,
        )?;
        if (0..total_rank).all(|position| {
            linearized_tree_pair_axis_at(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
                position,
            ) == position
        }) {
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: true,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::Repartition,
            });
        }
        if unique {
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: true,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::UniqueBraid(UniqueBorrowedTreePairBraid {
                    codomain_permutation,
                    domain_permutation,
                    source_codomain_rank,
                    source_domain_rank,
                    raw_axis_positions,
                    levels: UniqueBorrowedBraidLevels::Symmetric,
                }),
            });
        }
        let permutation = materialize_linearized_tree_pair_permutation(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        );
        let mut levels = SmallVec::<[usize; 8]>::with_capacity(total_rank);
        levels.extend(0..source_codomain_rank);
        levels.extend(
            (source_codomain_rank..source_codomain_rank + source_domain_rank).rev(),
        );
        let braid = PreparedTreeBraid::new(&permutation, &levels, total_rank)?;
        Ok(Self {
            source_codomain_rank,
            source_domain_rank,
            target_codomain_rank,
            requires_symmetric_braiding: true,
            family: PreparedTreePairFamily::BraidLike,
            plan: PreparedTreePairPlan::Braid(braid),
        })
    }

    pub fn prepare_transpose(
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<Self, CoreError> {
        let permutation = Self::validated_transpose_permutation(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        )?;
        let total_rank = source_codomain_rank + source_domain_rank;
        let target_codomain_rank = codomain_permutation.len();
        let Some(position) = permutation.iter().position(|&axis| axis == 0) else {
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: false,
                family: PreparedTreePairFamily::Transpose,
                plan: PreparedTreePairPlan::Identity,
            });
        };
        let plan = if position == 0 {
            PreparedTreePairPlan::Repartition
        } else if position < total_rank >> 1 {
            PreparedTreePairPlan::Transpose {
                direction: PreparedCycleDirection::Anticlockwise,
                count: position,
            }
        } else {
            PreparedTreePairPlan::Transpose {
                direction: PreparedCycleDirection::Clockwise,
                count: total_rank - position,
            }
        };
        Ok(Self {
            source_codomain_rank,
            source_domain_rank,
            target_codomain_rank,
            requires_symmetric_braiding: false,
            family: PreparedTreePairFamily::Transpose,
            plan,
        })
    }

    #[doc(hidden)]
    pub fn validate_transpose_syntax(
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<(), CoreError> {
        validate_cyclic_tree_pair_axis_map_inline(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        )
    }

    fn validated_transpose_permutation(
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        source_codomain_rank: usize,
        source_domain_rank: usize,
    ) -> Result<SmallVec<[usize; 8]>, CoreError> {
        validate_cyclic_tree_pair_axis_map_inline(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        )?;
        Ok(materialize_linearized_tree_pair_permutation(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        ))
    }

    fn validate_source(&self, tree_pair: &FusionTreePairKey) -> Result<(), CoreError> {
        let actual_codomain_rank = tree_pair.codomain_tree().uncoupled().len();
        if actual_codomain_rank != self.source_codomain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: self.source_codomain_rank,
                actual: actual_codomain_rank,
            });
        }
        let actual_domain_rank = tree_pair.domain_tree().uncoupled().len();
        if actual_domain_rank != self.source_domain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: self.source_domain_rank,
                actual: actual_domain_rank,
            });
        }
        Ok(())
    }

    fn validate_rule_capabilities<R>(&self, rule: &R) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        if self.requires_symmetric_braiding && !rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: rule.braiding_style(),
            });
        }
        Ok(())
    }

    fn validate_block_preflight<R>(
        &self,
        rule: &R,
        expected_family: PreparedTreePairFamily,
    ) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        if self.family != expected_family {
            return Err(CoreError::MalformedFusionTree {
                message: match expected_family {
                    PreparedTreePairFamily::BraidLike => PREPARED_BRAID_BLOCK_FAMILY_ERROR,
                    PreparedTreePairFamily::Transpose => PREPARED_TRANSPOSE_BLOCK_FAMILY_ERROR,
                },
            });
        }
        self.validate_rule_capabilities(rule)
    }

    fn validate_source_split(
        &self,
        source_codomain_rank: usize,
        source_domain_rank: usize,
    ) -> Result<(), CoreError> {
        if source_codomain_rank != self.source_codomain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: self.source_codomain_rank,
                actual: source_codomain_rank,
            });
        }
        if source_domain_rank != self.source_domain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: self.source_domain_rank,
                actual: source_domain_rank,
            });
        }
        Ok(())
    }
}

impl<'operation> PreparedTreePairOperation<'operation> {
    /// Execute this prepared operation on one multiplicity-free tree pair.
    ///
    /// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
    /// provider-domain precondition.
    pub fn execute_multiplicity_free<R>(
        &self,
        rule: &R,
        tree_pair: &FusionTreePairKey,
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    {
        self.validate_source(tree_pair)?;
        self.validate_rule_capabilities(rule)?;
        if !rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: rule.fusion_style(),
            });
        }
        let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
        self.execute_multiplicity_free_validated(validated)
    }

    pub(crate) fn execute_multiplicity_free_proven<R>(
        &self,
        validated: ValidatedFusionTreePair<'_, R>,
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    {
        self.validate_source(validated.key)?;
        self.validate_rule_capabilities(validated.rule)?;
        if !validated.rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: validated.rule.fusion_style(),
            });
        }
        self.execute_multiplicity_free_validated(validated)
    }

    fn execute_multiplicity_free_validated<R>(
        &self,
        validated: ValidatedFusionTreePair<'_, R>,
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    {
        let rule = validated.rule;
        let tree_pair = validated.key;
        match &self.plan {
            PreparedTreePairPlan::Identity => {
                Ok(vec![(tree_pair.clone(), rule.scalar_one())])
            }
            PreparedTreePairPlan::Repartition => {
                multiplicity_free_repartition_tree_pair_validated(
                    validated,
                    self.target_codomain_rank,
                )
            }
            PreparedTreePairPlan::Braid(braid) => {
                let all_rank = self.source_codomain_rank + self.source_domain_rank;
                let all_codomain =
                    multiplicity_free_repartition_tree_pair_validated(validated, all_rank)?;
                let braided = compose_tree_pair_terms(rule, all_codomain, |rule, key| {
                    execute_multiplicity_free_tree_braid(
                        rule,
                        key.codomain_tree(),
                        braid,
                    )
                    .map(|terms| {
                        terms
                            .into_iter()
                            .map(|(codomain_tree, coefficient)| {
                                (
                                    FusionTreePairKey::pair(
                                        codomain_tree,
                                        key.domain_tree().clone(),
                                    ),
                                    coefficient,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })?;
                multiplicity_free_repartition_terms(
                    rule,
                    braided,
                    self.target_codomain_rank,
                )
            }
            PreparedTreePairPlan::UniqueBraid(braid) => {
                let all_rank = self.source_codomain_rank + self.source_domain_rank;
                let all_codomain =
                    multiplicity_free_repartition_tree_pair_validated(validated, all_rank)?;
                let braided = compose_tree_pair_terms(rule, all_codomain, |rule, key| {
                    execute_multiplicity_free_tree_braid_steps(
                        rule,
                        key.codomain_tree(),
                        braid.artin_steps(),
                    )
                    .map(|terms| {
                        terms
                            .into_iter()
                            .map(|(codomain_tree, coefficient)| {
                                (
                                    FusionTreePairKey::pair(
                                        codomain_tree,
                                        key.domain_tree().clone(),
                                    ),
                                    coefficient,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })?;
                multiplicity_free_repartition_terms(
                    rule,
                    braided,
                    self.target_codomain_rank,
                )
            }
            PreparedTreePairPlan::Transpose { direction, count } => {
                let mut current = multiplicity_free_repartition_tree_pair_validated(
                    validated,
                    self.target_codomain_rank,
                )?;
                for _ in 0..*count {
                    current = match direction {
                        PreparedCycleDirection::Clockwise => {
                            compose_tree_pair_terms(rule, current, |rule, key| {
                                multiplicity_free_cycle_clockwise_tree_pair(rule, key)
                            })?
                        }
                        PreparedCycleDirection::Anticlockwise => {
                            compose_tree_pair_terms(rule, current, |rule, key| {
                                multiplicity_free_cycle_anticlockwise_tree_pair(rule, key)
                            })?
                        }
                    };
                }
                Ok(current)
            }
        }
    }

    /// Execute this prepared operation on one unique-fusion tree pair.
    ///
    /// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
    /// provider-domain precondition.
    pub fn execute_unique_rigid<R>(
        &self,
        rule: &R,
        tree_pair: &FusionTreePairKey,
    ) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        self.validate_source(tree_pair)?;
        self.validate_rule_capabilities(rule)?;
        if rule.fusion_style() != FusionStyleKind::Unique {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: rule.fusion_style(),
            });
        }
        let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
        self.execute_unique_rigid_validated(validated)
    }

    pub(crate) fn execute_unique_rigid_proven<R>(
        &self,
        validated: ValidatedFusionTreePair<'_, R>,
    ) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        self.validate_source(validated.key)?;
        self.validate_rule_capabilities(validated.rule)?;
        if validated.rule.fusion_style() != FusionStyleKind::Unique {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: validated.rule.fusion_style(),
            });
        }
        self.execute_unique_rigid_validated(validated)
    }

    fn execute_unique_rigid_validated<R>(
        &self,
        validated: ValidatedFusionTreePair<'_, R>,
    ) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        let rule = validated.rule;
        match &self.plan {
            PreparedTreePairPlan::Identity => {
                Ok((validated.key.clone(), rule.scalar_one()))
            }
            PreparedTreePairPlan::Repartition => {
                unique_rigid_repartition_tree_pair_validated(
                    validated,
                    self.target_codomain_rank,
                )
            }
            PreparedTreePairPlan::Braid(braid) => {
                let all_rank = self.source_codomain_rank + self.source_domain_rank;
                let (all_codomain, repartition_to_all) =
                    unique_rigid_repartition_tree_pair_validated(validated, all_rank)?;
                let (braided_tree, braid_coefficient) = execute_unique_tree_braid(
                    rule,
                    all_codomain.codomain_tree(),
                    braid,
                )?;
                let braided_pair = FusionTreePairKey::pair(
                    braided_tree,
                    all_codomain.domain_tree().clone(),
                );
                let (destination, repartition_back) =
                    unique_rigid_repartition_tree_pair_unchecked(
                        rule,
                        &braided_pair,
                        self.target_codomain_rank,
                    )?;
                Ok((
                    destination,
                    repartition_to_all * braid_coefficient * repartition_back,
                ))
            }
            PreparedTreePairPlan::UniqueBraid(braid) => {
                let all_rank = self.source_codomain_rank + self.source_domain_rank;
                let (all_codomain, repartition_to_all) =
                    unique_rigid_repartition_tree_pair_validated(validated, all_rank)?;
                let (braided_tree, braid_coefficient) = execute_unique_tree_braid_borrowed(
                    rule,
                    all_codomain.codomain_tree(),
                    braid,
                )?;
                let braided_pair = FusionTreePairKey::pair(
                    braided_tree,
                    all_codomain.domain_tree().clone(),
                );
                let (destination, repartition_back) = unique_rigid_repartition_tree_pair_unchecked(
                    rule,
                    &braided_pair,
                    self.target_codomain_rank,
                )?;
                Ok((
                    destination,
                    repartition_to_all * braid_coefficient * repartition_back,
                ))
            }
            PreparedTreePairPlan::Transpose { direction, count } => {
                let mut current = unique_rigid_repartition_tree_pair_validated(
                    validated,
                    self.target_codomain_rank,
                )?;
                for _ in 0..*count {
                    let (next, coefficient) = match direction {
                        PreparedCycleDirection::Clockwise => {
                            unique_rigid_cycle_clockwise_tree_pair(rule, &current.0)?
                        }
                        PreparedCycleDirection::Anticlockwise => {
                            unique_rigid_cycle_anticlockwise_tree_pair(rule, &current.0)?
                        }
                    };
                    current = (next, current.1 * coefficient);
                }
                Ok(current)
            }
        }
    }

    fn execute_unique_pivotal<R>(
        &self,
        rule: &R,
        tree_pair: &FusionTreePairKey,
    ) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
    where
        R: MultiplicityFreePivotalSymbols,
        R::Scalar: Mul<Output = R::Scalar>,
    {
        self.validate_source(tree_pair)?;
        self.validate_rule_capabilities(rule)?;
        if rule.fusion_style() != FusionStyleKind::Unique {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: rule.fusion_style(),
            });
        }
        let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
        match &self.plan {
            PreparedTreePairPlan::Identity => {
                Ok((tree_pair.clone(), rule.scalar_one()))
            }
            PreparedTreePairPlan::Repartition => {
                unique_repartition_tree_pair_validated(validated, self.target_codomain_rank)
            }
            PreparedTreePairPlan::Braid(braid) => {
                let all_rank = self.source_codomain_rank + self.source_domain_rank;
                let (all_codomain, repartition_to_all) =
                    unique_repartition_tree_pair_validated(validated, all_rank)?;
                let (braided_tree, braid_coefficient) = execute_unique_tree_braid(
                    rule,
                    all_codomain.codomain_tree(),
                    braid,
                )?;
                let braided_pair = FusionTreePairKey::pair(
                    braided_tree,
                    all_codomain.domain_tree().clone(),
                );
                let (destination, repartition_back) = unique_repartition_tree_pair_unchecked(
                    rule, &braided_pair, self.target_codomain_rank,
                )?;
                Ok((
                    destination,
                    repartition_to_all * braid_coefficient * repartition_back,
                ))
            }
            PreparedTreePairPlan::UniqueBraid(braid) => {
                let all_rank = self.source_codomain_rank + self.source_domain_rank;
                let (all_codomain, repartition_to_all) =
                    unique_repartition_tree_pair_validated(validated, all_rank)?;
                let (braided_tree, braid_coefficient) = execute_unique_tree_braid_borrowed(
                    rule,
                    all_codomain.codomain_tree(),
                    braid,
                )?;
                let braided_pair = FusionTreePairKey::pair(
                    braided_tree,
                    all_codomain.domain_tree().clone(),
                );
                let (destination, repartition_back) = unique_repartition_tree_pair_unchecked(
                    rule, &braided_pair, self.target_codomain_rank,
                )?;
                Ok((
                    destination,
                    repartition_to_all * braid_coefficient * repartition_back,
                ))
            }
            PreparedTreePairPlan::Transpose { direction, count } => {
                let mut current =
                    unique_repartition_tree_pair_validated(validated, self.target_codomain_rank)?;
                for _ in 0..*count {
                    let (next, coefficient) = match direction {
                        PreparedCycleDirection::Clockwise => {
                            unique_cycle_clockwise_tree_pair(rule, &current.0)?
                        }
                        PreparedCycleDirection::Anticlockwise => {
                            unique_cycle_anticlockwise_tree_pair(rule, &current.0)?
                        }
                    };
                    current = (next, current.1 * coefficient);
                }
                Ok(current)
            }
        }
    }
}

/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
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
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
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
    let prepared = PreparedTreeBraid::new(permutation, levels, rank)?;
    validate_fusion_tree_for_rule(rule, tree)?;
    if prepared.permutation.iter().copied().eq(0..rank) {
        return Ok((tree.clone(), rule.scalar_one()));
    }
    execute_unique_tree_braid(rule, tree, &prepared)
}

/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
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
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: rule.fusion_style(),
        });
    }
    let rank = tree.uncoupled().len();
    let levels = (0..rank).collect::<SmallVec<[usize; 8]>>();
    let prepared = PreparedTreeBraid::new(permutation, &levels, rank)?;
    validate_fusion_tree_for_rule(rule, tree)?;
    if permutation.iter().copied().eq(0..rank) {
        return Ok((tree.clone(), rule.scalar_one()));
    }
    execute_unique_tree_braid(rule, tree, &prepared)
}

/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
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
    let prepared = PreparedTreeBraid::new(permutation, levels, rank)?;
    let validated = validate_fusion_tree_for_rule(rule, tree)?;
    execute_multiplicity_free_tree_braid_proven(validated, prepared)
}

fn execute_multiplicity_free_tree_braid_proven<R>(
    validated: ValidatedFusionTree<'_, R>,
    prepared: PreparedTreeBraid,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !validated.rule.fusion_style().is_multiplicity_free() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: validated.rule.fusion_style(),
        });
    }
    if prepared.permutation.len() != validated.key.uncoupled().len() {
        return Err(CoreError::DimensionMismatch {
            expected: validated.key.uncoupled().len(),
            actual: prepared.permutation.len(),
        });
    }
    let rule = validated.rule;
    let tree = validated.key;
    let rank = tree.uncoupled().len();
    if prepared.permutation.iter().copied().eq(0..rank) {
        return Ok(vec![(tree.clone(), rule.scalar_one())]);
    }
    execute_multiplicity_free_tree_braid(rule, tree, &prepared)
}

fn execute_multiplicity_free_tree_braid<R>(
    rule: &R,
    tree: &FusionTreeKey,
    prepared: &PreparedTreeBraid,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if rule.fusion_style() == FusionStyleKind::Unique
        && rule.braiding_style().is_symmetric()
        && rule.has_trivial_associator_gauge()
    {
        let (destination, coefficient) = execute_unique_tree_braid(rule, tree, prepared)?;
        return Ok(vec![(destination, coefficient)]);
    }

    execute_multiplicity_free_tree_braid_steps(rule, tree, prepared.artin_steps.iter().copied())
}

fn execute_multiplicity_free_tree_braid_steps<R, I>(
    rule: &R,
    tree: &FusionTreeKey,
    steps: I,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    I: IntoIterator<Item = PreparedArtinStep>,
{
    let mut current = vec![(tree.clone(), rule.scalar_one())];
    for step in steps {
        let mut next_terms = FusionTermAccumulator::new();
        for (tree, coefficient) in current {
            for (next_tree, step_coefficient) in
                multiplicity_free_artin_braid_at_with_inverse(
                    rule,
                    &tree,
                    step.index,
                    step.inverse,
                )?
            {
                next_terms.push(next_tree, coefficient.clone() * step_coefficient);
            }
        }
        current = next_terms.into_vec();
    }
    Ok(current)
}

/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
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
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: rule.fusion_style(),
        });
    }
    let rank = tree.uncoupled().len();
    let levels = (0..rank).collect::<SmallVec<[usize; 8]>>();
    let prepared = PreparedTreeBraid::new(permutation, &levels, rank)?;
    let validated = validate_fusion_tree_for_rule(rule, tree)?;
    execute_multiplicity_free_tree_braid_proven(validated, prepared)
}

/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn multiplicity_free_repartition_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    multiplicity_free_repartition_tree_pair_validated(validated, target_codomain_rank)
}

fn multiplicity_free_repartition_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rule = tree_pair.rule;
    let tree_pair = tree_pair.key;
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

/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn multiplicity_free_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    PreparedTreePairOperation::prepare_braid(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )?
    .execute_multiplicity_free(rule, tree_pair)
}

/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn multiplicity_free_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    PreparedTreePairOperation::prepare_permute(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?
    .execute_multiplicity_free(rule, tree_pair)
}

/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn multiplicity_free_transpose_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    PreparedTreePairOperation::prepare_transpose(
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?
    .execute_multiplicity_free(rule, tree_pair)
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
    terms: Vec<(FusionTreePairKey, R::Scalar)>,
    mut transform: F,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    F: FnMut(&R, &FusionTreePairKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreePairKey, R::Scalar)>,
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

/// Ordered block-linear result produced by one categorical block transform.
///
/// This is an internal crate-boundary vocabulary for lowering reduced blocks.
/// Destination keys are unique and ordered by source-major first appearance;
/// columns retain the caller's source order. Structurally absent entries remain
/// distinct from present zero coefficients, and no destination row is wholly
/// absent.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct OrderedBlockLinearMap<K, S> {
    destinations: Vec<K>,
    source_count: usize,
    storage: OrderedBlockLinearStorage<S>,
}

/// Structural storage for an [`OrderedBlockLinearMap`].
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub enum OrderedBlockLinearStorage<S> {
    /// Exactly one structurally present destination for every source column.
    SingletonColumns {
        destination_rows: Vec<usize>,
        coefficients: Vec<S>,
    },
    /// Row-major destination-by-source coefficients.
    DenseDstSrc(Vec<Option<S>>),
}

#[doc(hidden)]
impl<K, S> OrderedBlockLinearMap<K, S> {
    #[inline]
    pub fn destinations(&self) -> &[K] {
        &self.destinations
    }

    #[inline]
    pub fn source_count(&self) -> usize {
        self.source_count
    }

    #[inline]
    pub fn storage(&self) -> &OrderedBlockLinearStorage<S> {
        &self.storage
    }

    #[inline]
    pub fn into_parts(self) -> (Vec<K>, usize, OrderedBlockLinearStorage<S>) {
        (self.destinations, self.source_count, self.storage)
    }
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

#[cfg(test)]
fn compose_block_terms<R, F, I>(
    rule: &R,
    basis: &[FusionTreePairKey],
    columns: &DenseColumns<R::Scalar>,
    mut transform: F,
) -> Result<(Vec<FusionTreePairKey>, DenseColumns<R::Scalar>), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    F: FnMut(&R, &FusionTreePairKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreePairKey, R::Scalar)>,
{
    let num_src = columns.num_src;
    // Dedup destination tree-pairs to dense rows. The key is *moved* into the
    // map (no per-destination clone — this dedup is the hottest FusionTreeKey
    // clone/eq/hash site on the cold recoupling path); `next_basis` is rebuilt
    // from the map by row index afterwards. Rows are assigned in first-
    // appearance order, so the rebuilt `next_basis` order — and therefore every
    // coefficient — is bit-for-bit identical to pushing the key eagerly.
    let mut index: FxHashMap<FusionTreePairKey, usize> = FxHashMap::default();
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
    let mut slots: Vec<Option<FusionTreePairKey>> = (0..index.len()).map(|_| None).collect();
    for (key, row) in index {
        slots[row] = Some(key);
    }
    let next_basis: Vec<FusionTreePairKey> = slots
        .into_iter()
        .map(|key| key.expect("dense rows 0..len are all filled"))
        .collect();
    Ok((next_basis, next_columns))
}

struct CompactMultiplicityFreeTreeBasis {
    frame: MultiplicityFreeTreeFrame,
    locals: Vec<MultiplicityFreeTreeLocal>,
}

impl CompactMultiplicityFreeTreeBasis {
    fn from_group<R>(
        group: ValidatedFusionTreeBlockGroup<'_, R>,
    ) -> Result<Self, CoreError> {
        let src_keys = group.src_keys;
        let (frame, first_local) = MultiplicityFreeTreeFrame::split(
            group
                .projection
                .tree_at(0)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "compact block basis requires at least one source",
                })?,
        );
        let mut locals = Vec::with_capacity(src_keys.len());
        locals.push(first_local);
        for (index, source) in src_keys.iter().enumerate().skip(1) {
            if !frame.matches_tree(source) {
                return Err(CoreError::MalformedFusionTree {
                    message: "fusion-tree keys must share one group",
                });
            }
            locals.push(MultiplicityFreeTreeLocal::from_proven(
                group
                    .projection
                    .tree_at(index)
                    .expect("validated projection covers every source"),
            ));
        }
        Ok(Self { frame, locals })
    }
}

struct CompactMultiplicityFreeTreePairBasis {
    frame: MultiplicityFreeTreePairFrame,
    locals: Vec<MultiplicityFreeTreePairLocal>,
}

struct CompactMultiplicityFreeTreePairBlock<S> {
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: DenseColumns<S>,
    records_dimensions: bool,
}

type CompactMultiplicityFreeTreePairRows<S> = Vec<Vec<(FusionTreePairKey, S)>>;

impl CompactMultiplicityFreeTreePairBasis {
    fn from_group<R>(
        group: ValidatedTreePairBlockGroup<'_, R>,
    ) -> Result<Self, CoreError> {
        let (frame, first_local) = MultiplicityFreeTreePairFrame::split(
            group
                .projection
                .pair_at(0)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "compact block basis requires at least one source",
                })?,
        );
        let mut locals = Vec::with_capacity(group.source_len);
        locals.push(first_local);
        for index in 1..group.source_len {
            let source = group
                .projection
                .pair_at(index)
                .expect("validated projection covers every source");
            if !frame.matches_tree_pair_ref(source) {
                return Err(CoreError::MalformedFusionTree {
                    message: TREE_PAIR_BLOCK_GROUP_ERROR,
                });
            }
            locals.push(MultiplicityFreeTreePairLocal::from_proven(
                source,
            ));
        }
        Ok(Self { frame, locals })
    }
}

fn compose_compact_block_terms<R, K, F, I>(
    rule: &R,
    basis: &[K],
    columns: &DenseColumns<R::Scalar>,
    mut transform: F,
) -> Result<(Vec<K>, DenseColumns<R::Scalar>), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    K: Eq + Hash,
    F: FnMut(&R, &K) -> Result<I, CoreError>,
    I: IntoIterator<Item = (K, R::Scalar)>,
{
    let num_src = columns.num_src;
    let mut index: FxHashMap<K, usize> = FxHashMap::default();
    let mut next_columns = DenseColumns::with_capacity(num_src, basis.len());
    for (source_row, source_local) in basis.iter().enumerate() {
        for (destination_local, step_coefficient) in transform(rule, source_local)? {
            let row = match index.get(&destination_local) {
                Some(&row) => row,
                None => {
                    let row = next_columns.push_empty_row();
                    index.insert(destination_local, row);
                    row
                }
            };
            let source_column = columns.row(source_row);
            let destination_column = next_columns.row_mut(row);
            for (src, source_coefficient) in source_column.iter().enumerate() {
                let Some(source_coefficient) = source_coefficient else {
                    continue;
                };
                let contribution = step_coefficient.clone() * source_coefficient.clone();
                destination_column[src] = Some(match destination_column[src].take() {
                    Some(existing) => existing + contribution,
                    None => contribution,
                });
            }
        }
    }
    let mut slots: Vec<Option<K>> = (0..index.len()).map(|_| None).collect();
    for (local, row) in index {
        slots[row] = Some(local);
    }
    let locals = slots
        .into_iter()
        .map(|local| local.expect("dense rows 0..len are all filled"))
        .collect();
    Ok((locals, next_columns))
}

fn compose_generic_block_terms<R, F, I>(
    rule: &R,
    basis: &[FusionTreePairKey],
    columns: &DenseColumns<R::Scalar>,
    mut transform: F,
) -> Result<(Vec<FusionTreePairKey>, DenseColumns<R::Scalar>), CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
    F: FnMut(&R, &FusionTreePairKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreePairKey, R::Scalar)>,
{
    let num_src = columns.num_src;
    let mut index: FxHashMap<FusionTreePairKey, usize> = FxHashMap::default();
    let mut next_columns = DenseColumns::with_capacity(num_src, basis.len());
    for (source_row, source_key) in basis.iter().enumerate() {
        for (destination_key, step_coefficient) in transform(rule, source_key)? {
            let row = match index.get(&destination_key) {
                Some(&row) => row,
                None => {
                    let row = next_columns.push_empty_row();
                    index.insert(destination_key, row);
                    row
                }
            };
            let source_column = columns.row(source_row);
            let destination_column = next_columns.row_mut(row);
            for (source, source_coefficient) in source_column.iter().enumerate() {
                let Some(source_coefficient) = source_coefficient else {
                    continue;
                };
                let contribution = step_coefficient.clone() * source_coefficient.clone();
                destination_column[source] = Some(match destination_column[source].take() {
                    Some(existing) => existing + contribution,
                    None => contribution,
                });
            }
        }
    }
    let mut slots: Vec<Option<FusionTreePairKey>> = (0..index.len()).map(|_| None).collect();
    for (key, row) in index {
        slots[row] = Some(key);
    }
    let basis = slots
        .into_iter()
        .map(|key| key.expect("dense rows 0..len are all filled"))
        .collect();
    Ok((basis, next_columns))
}

fn seed_generic_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
) -> Result<(Vec<FusionTreePairKey>, DenseColumns<R::Scalar>), CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if rule.fusion_style() != FusionStyleKind::Generic {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let mut basis = Vec::with_capacity(src_keys.len());
    for source in src_keys {
        validate_fusion_tree_pair_for_rule(rule, source)?;
        basis.push(source.clone());
    }
    let mut columns = DenseColumns::with_capacity(src_keys.len(), src_keys.len());
    for source in 0..src_keys.len() {
        let row = columns.push_empty_row();
        columns.row_mut(row)[source] = Some(R::Scalar::braid_one());
    }
    Ok((basis, columns))
}

fn order_generic_tree_pair_block<S: Clone>(
    basis: Vec<FusionTreePairKey>,
    columns: DenseColumns<S>,
) -> OrderedBlockLinearMap<FusionTreePairKey, S> {
    let source_count = columns.num_src;
    let mut ordered_basis_rows = Vec::with_capacity(basis.len());
    let mut ordered_row_for_basis = vec![usize::MAX; basis.len()];
    let mut singleton_basis_rows = Vec::with_capacity(source_count);
    let mut is_singleton = true;

    for source in 0..source_count {
        let mut only_basis_row = None;
        for (basis_row, ordered_row) in ordered_row_for_basis.iter_mut().enumerate() {
            if columns.row(basis_row)[source].is_none() {
                continue;
            }
            if *ordered_row == usize::MAX {
                *ordered_row = ordered_basis_rows.len();
                ordered_basis_rows.push(basis_row);
            }
            if only_basis_row.replace(basis_row).is_some() {
                is_singleton = false;
            }
        }
        match only_basis_row {
            Some(basis_row) => singleton_basis_rows.push(basis_row),
            None => {
                is_singleton = false;
                singleton_basis_rows.push(usize::MAX);
            }
        }
    }

    let destinations = ordered_basis_rows
        .iter()
        .map(|&basis_row| basis[basis_row].clone())
        .collect::<Vec<_>>();
    let storage = if is_singleton {
        let mut destination_rows = Vec::with_capacity(source_count);
        let mut coefficients = Vec::with_capacity(source_count);
        for (source, basis_row) in singleton_basis_rows.into_iter().enumerate() {
            destination_rows.push(ordered_row_for_basis[basis_row]);
            coefficients.push(
                columns.data[basis_row * source_count + source]
                    .clone()
                    .expect("singleton source has one present coefficient"),
            );
        }
        OrderedBlockLinearStorage::SingletonColumns {
            destination_rows,
            coefficients,
        }
    } else {
        let mut coefficients =
            Vec::with_capacity(ordered_basis_rows.len().saturating_mul(source_count));
        for basis_row in ordered_basis_rows {
            let row_start = basis_row * source_count;
            coefficients.extend(columns.data[row_start..row_start + source_count].iter().cloned());
        }
        OrderedBlockLinearStorage::DenseDstSrc(coefficients)
    };

    OrderedBlockLinearMap {
        destinations,
        source_count: columns.num_src,
        storage,
    }
}

fn generic_repartition_tree_pair_block_terms<R>(
    rule: &R,
    mut basis: Vec<FusionTreePairKey>,
    mut columns: DenseColumns<R::Scalar>,
    target_codomain_rank: usize,
) -> Result<(Vec<FusionTreePairKey>, DenseColumns<R::Scalar>), CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let Some(first) = basis.first() else {
        return Ok((basis, columns));
    };
    let total_rank = first.codomain_tree().uncoupled().len() + first.domain_tree().uncoupled().len();
    if target_codomain_rank > total_rank {
        return Err(CoreError::DimensionMismatch {
            expected: total_rank,
            actual: target_codomain_rank,
        });
    }
    let mut current_codomain_rank = first.codomain_tree().uncoupled().len();
    while current_codomain_rank < target_codomain_rank {
        (basis, columns) = compose_generic_block_terms(rule, &basis, &columns, |rule, key| {
            generic_bendleft_tree_pair(rule, key)
        })?;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        (basis, columns) = compose_generic_block_terms(rule, &basis, &columns, |rule, key| {
            generic_bendright_tree_pair(rule, key)
        })?;
        current_codomain_rank -= 1;
    }
    Ok((basis, columns))
}

#[doc(hidden)]
pub fn generic_braid_tree_pair_block_ordered<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if src_keys.is_empty() {
        return Ok(order_generic_tree_pair_block(
            Vec::new(),
            DenseColumns::with_capacity(0, 0),
        ));
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
    for source in src_keys {
        if source.codomain_tree().uncoupled().len() != codomain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: codomain_rank,
                actual: source.codomain_tree().uncoupled().len(),
            });
        }
        if source.domain_tree().uncoupled().len() != domain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: domain_rank,
                actual: source.domain_tree().uncoupled().len(),
            });
        }
    }
    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    let swaps = permutation_to_adjacent_swaps(&permutation, codomain_rank + domain_rank)?;
    let identity = tree_pair_axis_map_is_identity(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    );
    let mut levels = Vec::with_capacity(codomain_rank + domain_rank);
    levels.extend_from_slice(codomain_levels);
    levels.extend(domain_levels.iter().rev().copied());

    let (mut basis, mut columns) = seed_generic_tree_pair_block(rule, src_keys)?;
    if identity {
        return Ok(order_generic_tree_pair_block(basis, columns));
    }
    let all_rank = permutation.len();
    (basis, columns) = generic_repartition_tree_pair_block_terms(rule, basis, columns, all_rank)?;
    (basis, columns) = compose_generic_block_terms(rule, &basis, &columns, |rule, key| {
        generic_braid_tree_unchecked(rule, key.codomain_tree(), &permutation, &levels, &swaps).map(
            |terms| {
                terms
                    .into_iter()
                    .map(|(codomain_tree, coefficient)| {
                        (
                            FusionTreePairKey::pair(codomain_tree, key.domain_tree().clone()),
                            coefficient,
                        )
                    })
                    .collect::<Vec<_>>()
            },
        )
    })?;
    (basis, columns) = generic_repartition_tree_pair_block_terms(
        rule,
        basis,
        columns,
        codomain_permutation.len(),
    )?;
    Ok(order_generic_tree_pair_block(basis, columns))
}

#[doc(hidden)]
pub fn generic_permute_tree_pair_block_ordered<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
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
    let Some(first) = src_keys.first() else {
        return Ok(order_generic_tree_pair_block(
            Vec::new(),
            DenseColumns::with_capacity(0, 0),
        ));
    };
    let codomain_rank = first.codomain_tree().uncoupled().len();
    let domain_rank = first.domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    generic_braid_tree_pair_block_ordered(
        rule,
        src_keys,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

#[doc(hidden)]
pub fn generic_transpose_tree_pair_block_ordered<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if src_keys.is_empty() {
        return Ok(order_generic_tree_pair_block(
            Vec::new(),
            DenseColumns::with_capacity(0, 0),
        ));
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
    for source in src_keys {
        if source.codomain_tree().uncoupled().len() != codomain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: codomain_rank,
                actual: source.codomain_tree().uncoupled().len(),
            });
        }
        if source.domain_tree().uncoupled().len() != domain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: domain_rank,
                actual: source.domain_tree().uncoupled().len(),
            });
        }
    }
    let mut position = match permutation.iter().position(|&axis| axis == 0) {
        Some(position) => position,
        None => {
            let (basis, columns) = seed_generic_tree_pair_block(rule, src_keys)?;
            return Ok(order_generic_tree_pair_block(basis, columns));
        }
    };
    let total_rank = codomain_rank + domain_rank;
    let (mut basis, mut columns) =
        seed_generic_tree_pair_block(rule, src_keys)?;
    (basis, columns) = generic_repartition_tree_pair_block_terms(
        rule,
        basis,
        columns,
        codomain_permutation.len(),
    )?;
    if total_rank == 0 || position == 0 {
        return Ok(order_generic_tree_pair_block(basis, columns));
    }

    let half_rank = total_rank >> 1;
    while position > 0 && position < half_rank {
        (basis, columns) = compose_generic_block_terms(rule, &basis, &columns, |rule, key| {
            generic_cycle_anticlockwise_tree_pair_unchecked(rule, key)
        })?;
        position -= 1;
    }
    while position >= half_rank && position > 0 {
        (basis, columns) = compose_generic_block_terms(rule, &basis, &columns, |rule, key| {
            generic_cycle_clockwise_tree_pair_unchecked(rule, key)
        })?;
        position = (position + 1) % total_rank;
    }

    Ok(order_generic_tree_pair_block(basis, columns))
}

fn apply_first_compact_block_terms<R, K, F, I>(
    rule: &R,
    basis: &[K],
    mut transform: F,
) -> Result<(Vec<K>, DenseColumns<R::Scalar>), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar>,
    K: Eq + Hash,
    F: FnMut(&R, &K) -> Result<I, CoreError>,
    I: IntoIterator<Item = (K, R::Scalar)>,
{
    let mut index: FxHashMap<K, usize> = FxHashMap::default();
    let mut columns = DenseColumns::with_capacity(basis.len(), basis.len());
    for (source, source_local) in basis.iter().enumerate() {
        for (destination_local, coefficient) in transform(rule, source_local)? {
            let row = match index.get(&destination_local) {
                Some(&row) => row,
                None => {
                    let row = columns.push_empty_row();
                    index.insert(destination_local, row);
                    row
                }
            };
            let destination = &mut columns.row_mut(row)[source];
            *destination = Some(match destination.take() {
                Some(existing) => existing + coefficient,
                None => coefficient,
            });
        }
    }
    let mut slots: Vec<Option<K>> = (0..index.len()).map(|_| None).collect();
    for (local, row) in index {
        slots[row] = Some(local);
    }
    let locals = slots
        .into_iter()
        .map(|local| local.expect("dense rows 0..len are all filled"))
        .collect();
    Ok((locals, columns))
}

fn compact_artin_tree_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreeBasis,
    index: usize,
    inverse: bool,
) -> Result<
    (
        CompactMultiplicityFreeTreeBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_artin(rule, &basis.frame, index, inverse)?;
    let (locals, columns) =
        apply_first_compact_block_terms(rule, &basis.locals, |rule, local| {
            prepared.apply(rule, local)
        })?;
    Ok((
        CompactMultiplicityFreeTreeBasis {
            frame: prepared.output_frame,
            locals,
        },
        columns,
    ))
}

fn compact_artin_tree_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreeBasis,
    columns: &DenseColumns<R::Scalar>,
    index: usize,
    inverse: bool,
) -> Result<
    (
        CompactMultiplicityFreeTreeBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_artin(rule, &basis.frame, index, inverse)?;
    let (locals, next_columns) =
        compose_compact_block_terms(rule, &basis.locals, columns, |rule, local| {
            prepared.apply(rule, local)
        })?;
    Ok((
        CompactMultiplicityFreeTreeBasis {
            frame: prepared.output_frame,
            locals,
        },
        next_columns,
    ))
}

fn scatter_compact_tree_block<S>(
    basis: CompactMultiplicityFreeTreeBasis,
    columns: &DenseColumns<S>,
) -> Vec<Vec<(FusionTreeKey, S)>>
where
    S: Clone,
{
    let mut rows_per_source = vec![Vec::new(); columns.num_src];
    for (destination_row, destination_local) in basis.locals.into_iter().enumerate() {
        // Why not materialize in each source column: the full external frame is
        // identical for the row and must be rebuilt only once at the API edge.
        let destination_key = basis.frame.materialize(destination_local);
        for (source, coefficient) in columns.row(destination_row).iter().enumerate() {
            if let Some(coefficient) = coefficient {
                rows_per_source[source].push((destination_key.clone(), coefficient.clone()));
            }
        }
    }
    rows_per_source
}

fn compact_bendright_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_bendright(rule, &basis.frame)?;
    let (locals, columns) =
        apply_first_compact_block_terms(rule, &basis.locals, |rule, local| {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            let coefficient = prepared.coefficient(rule, &validated);
            Ok(std::iter::once((validated.local, coefficient)))
        })?;
    let frame = prepared.output_frame(rule)?;
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        columns,
    ))
}

fn compact_bendleft_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_bendleft(rule, &basis.frame)?;
    let (locals, columns) =
        apply_first_compact_block_terms(rule, &basis.locals, |rule, local| {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            Ok(std::iter::once(prepared.finish_local(rule, validated)))
        })?;
    let frame = prepared.output_frame(rule)?;
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        columns,
    ))
}

fn compact_foldright_block<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: Option<&DenseColumns<R::Scalar>>,
    swap_output: bool,
    conjugate_step: bool,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_foldright(rule, &basis.frame)?;
    let output_frame = if swap_output {
        MultiplicityFreeTreePairFrame {
            codomain: prepared.output_frame.domain.clone(),
            domain: prepared.output_frame.codomain.clone(),
        }
    } else {
        prepared.output_frame.clone()
    };
    let output_locals =
        collect_multiplicity_free_tree_pair_locals_for_frame(rule, &output_frame);
    let output_index = output_locals
        .iter()
        .cloned()
        .enumerate()
        .map(|(row, local)| (local, row))
        .collect::<FxHashMap<_, _>>();
    let num_src = columns
        .map(|columns| columns.num_src)
        .unwrap_or(basis.locals.len());
    let mut next_columns = DenseColumns::with_capacity(num_src, output_locals.len());
    for _ in &output_locals {
        next_columns.push_empty_row();
    }

    let mut forward_cache: FxHashMap<
        MultiplicityFreeTreeLocal,
        MultiplicityFreeTreeLocalTerms<R::Scalar>,
    > = FxHashMap::default();
    let mut inverse_cache: MultiplicityFreeFoldInverseCache<R::Scalar> = FxHashMap::default();
    let mut coefficient_cache: FxHashMap<
        (SectorId, SectorId),
        (R::Scalar, R::Scalar),
    > = FxHashMap::default();

    for (source_row, source) in basis.locals.iter().enumerate() {
        if columns.is_some_and(|columns| {
            columns
                .row(source_row)
                .iter()
                .all(Option::is_none)
        }) {
            continue;
        }
        if !forward_cache.contains_key(&source.codomain) {
            let terms = multiplicity_free_multi_fmove_local(
                rule,
                &basis.frame.codomain,
                &source.codomain,
            )?;
            forward_cache.insert(source.codomain.clone(), terms);
        }
        let forward = forward_cache
            .get(&source.codomain)
            .expect("forward fold table inserted above");
        let coupled = source.codomain.coupled;
        for (codomain, codomain_coefficient) in forward {
            let tail_coupled = codomain.coupled;
            let inverse_key = (tail_coupled, source.domain.clone());
            if !inverse_cache.contains_key(&inverse_key) {
                let terms = multiplicity_free_multi_fmove_inv_local(
                    rule,
                    tail_coupled,
                    &basis.frame.domain,
                    &source.domain,
                    &prepared.output_frame.domain,
                )?;
                inverse_cache.insert(inverse_key.clone(), terms);
            }
            let inverse = inverse_cache
                .get(&inverse_key)
                .expect("inverse fold table inserted above");
            let cache_key = (tail_coupled, coupled);
            if let Entry::Vacant(entry) = coefficient_cache.entry(cache_key) {
                entry.insert((
                    rule.sqrt_dim_scalar(coupled) * rule.inv_sqrt_dim_scalar(tail_coupled),
                    rule.a_symbol_scalar(prepared.first, tail_coupled, coupled),
                ));
            }
            let (normalization, a_symbol) = coefficient_cache
                .get(&cache_key)
                .expect("fold coefficient table inserted above");
            for (domain, domain_coefficient) in inverse {
                let mut destination = MultiplicityFreeTreePairLocal {
                    codomain: codomain.clone(),
                    domain: domain.clone(),
                };
                if swap_output {
                    destination = MultiplicityFreeTreePairLocal {
                        codomain: destination.domain,
                        domain: destination.codomain,
                    };
                }
                let destination_row = *output_index.get(&destination).ok_or(
                    CoreError::MalformedFusionTree {
                        message: "compact fold destination is outside the canonical output basis",
                    },
                )?;
                let mut coefficient = normalization.clone()
                    * rule.scalar_conj(domain_coefficient.clone())
                    * a_symbol.clone()
                    * codomain_coefficient.clone();
                if prepared.first_is_dual {
                    coefficient = coefficient * prepared.frobenius_schur_phase.clone();
                }
                if conjugate_step {
                    coefficient = rule.scalar_conj(coefficient);
                }
                if let Some(columns) = columns {
                    let source_column = columns.row(source_row);
                    let destination_column = next_columns.row_mut(destination_row);
                    for (source, source_coefficient) in source_column.iter().enumerate() {
                        let Some(source_coefficient) = source_coefficient else {
                            continue;
                        };
                        let contribution =
                            coefficient.clone() * source_coefficient.clone();
                        destination_column[source] =
                            Some(match destination_column[source].take() {
                                Some(existing) => existing + contribution,
                                None => contribution,
                            });
                    }
                } else {
                    let destination = &mut next_columns.row_mut(destination_row)[source_row];
                    *destination = Some(match destination.take() {
                        Some(existing) => existing + coefficient,
                        None => coefficient,
                    });
                }
            }
        }
    }

    Ok((
        CompactMultiplicityFreeTreePairBasis {
            frame: output_frame,
            locals: output_locals,
        },
        next_columns,
    ))
}

fn compact_foldright_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    compact_foldright_block(rule, basis, None, false, false)
}

fn compact_foldleft_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let CompactMultiplicityFreeTreePairBasis { frame, locals } = basis;
    let swapped = CompactMultiplicityFreeTreePairBasis {
        frame: MultiplicityFreeTreePairFrame {
            codomain: frame.domain,
            domain: frame.codomain,
        },
        locals: locals
            .into_iter()
            .map(|local| MultiplicityFreeTreePairLocal {
                codomain: local.domain,
                domain: local.codomain,
            })
            .collect(),
    };
    compact_foldright_block(rule, swapped, None, true, true)
}

fn compact_codomain_artin_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    index: usize,
    inverse: bool,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared =
        prepare_multiplicity_free_artin(rule, &basis.frame.codomain, index, inverse)?;
    let (locals, columns) =
        apply_first_compact_block_terms(rule, &basis.locals, |rule, local| {
            let domain = local.domain.clone();
            Ok(prepared
                .apply(rule, &local.codomain)?
                .into_iter()
                .map(move |(codomain, coefficient)| {
                    (
                        MultiplicityFreeTreePairLocal {
                            codomain,
                            domain: domain.clone(),
                        },
                        coefficient,
                    )
                }))
        })?;
    let frame = MultiplicityFreeTreePairFrame {
        codomain: prepared.output_frame,
        domain: basis.frame.domain,
    };
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        columns,
    ))
}

fn compact_bendright_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_bendright(rule, &basis.frame)?;
    let (locals, next_columns) =
        compose_compact_block_terms(rule, &basis.locals, columns, |rule, local| {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            let coefficient = prepared.coefficient(rule, &validated);
            Ok(std::iter::once((validated.local, coefficient)))
        })?;
    let frame = prepared.output_frame(rule)?;
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        next_columns,
    ))
}

fn compact_bendleft_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_bendleft(rule, &basis.frame)?;
    let (locals, next_columns) =
        compose_compact_block_terms(rule, &basis.locals, columns, |rule, local| {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            Ok(std::iter::once(prepared.finish_local(rule, validated)))
        })?;
    let frame = prepared.output_frame(rule)?;
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        next_columns,
    ))
}

fn compact_foldright_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    compact_foldright_block(rule, basis, Some(columns), false, false)
}

fn compact_foldleft_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let CompactMultiplicityFreeTreePairBasis { frame, locals } = basis;
    let swapped = CompactMultiplicityFreeTreePairBasis {
        frame: MultiplicityFreeTreePairFrame {
            codomain: frame.domain,
            domain: frame.codomain,
        },
        locals: locals
            .into_iter()
            .map(|local| MultiplicityFreeTreePairLocal {
                codomain: local.domain,
                domain: local.codomain,
            })
            .collect(),
    };
    compact_foldright_block(rule, swapped, Some(columns), true, true)
}

fn compact_codomain_artin_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
    index: usize,
    inverse: bool,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared =
        prepare_multiplicity_free_artin(rule, &basis.frame.codomain, index, inverse)?;
    let (locals, next_columns) =
        compose_compact_block_terms(rule, &basis.locals, columns, |rule, local| {
            let domain = local.domain.clone();
            Ok(prepared
                .apply(rule, &local.codomain)?
                .into_iter()
                .map(move |(codomain, coefficient)| {
                    (
                        MultiplicityFreeTreePairLocal {
                            codomain,
                            domain: domain.clone(),
                        },
                        coefficient,
                    )
                }))
        })?;
    let frame = MultiplicityFreeTreePairFrame {
        codomain: prepared.output_frame,
        domain: basis.frame.domain,
    };
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        next_columns,
    ))
}

fn compact_cycle_clockwise_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if basis.frame.codomain.uncoupled.is_empty() {
        let (basis, columns) = compact_bendleft_block_first(rule, basis)?;
        compact_foldright_block_step(rule, basis, &columns)
    } else {
        let (basis, columns) = compact_foldright_block_first(rule, basis)?;
        compact_bendleft_block_step(rule, basis, &columns)
    }
}

fn compact_cycle_clockwise_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if basis.frame.codomain.uncoupled.is_empty() {
        let (basis, columns) = compact_bendleft_block_step(rule, basis, columns)?;
        compact_foldright_block_step(rule, basis, &columns)
    } else {
        let (basis, columns) = compact_foldright_block_step(rule, basis, columns)?;
        compact_bendleft_block_step(rule, basis, &columns)
    }
}

fn compact_cycle_anticlockwise_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if basis.frame.domain.uncoupled.is_empty() {
        let (basis, columns) = compact_bendright_block_first(rule, basis)?;
        compact_foldleft_block_step(rule, basis, &columns)
    } else {
        let (basis, columns) = compact_foldleft_block_first(rule, basis)?;
        compact_bendright_block_step(rule, basis, &columns)
    }
}

fn compact_cycle_anticlockwise_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if basis.frame.domain.uncoupled.is_empty() {
        let (basis, columns) = compact_bendright_block_step(rule, basis, columns)?;
        compact_foldleft_block_step(rule, basis, &columns)
    } else {
        let (basis, columns) = compact_foldleft_block_step(rule, basis, columns)?;
        compact_bendright_block_step(rule, basis, &columns)
    }
}

fn scatter_compact_block<S: Clone>(
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: DenseColumns<S>,
) -> Vec<Vec<(FusionTreePairKey, S)>> {
    #[cfg(test)]
    COMPACT_BLOCK_DIMENSIONS.with(|dimensions| {
        dimensions.set(Some(CompactBlockDimensions {
            destination_rows: basis.locals.len(),
            source_columns: columns.num_src,
            coefficient_slots: columns.data.len(),
            coefficient_bytes: columns
                .data
                .len()
                .saturating_mul(std::mem::size_of::<Option<S>>()),
        }));
    });
    let mut rows_per_source = vec![Vec::new(); columns.num_src];
    for (destination_row, destination_local) in basis.locals.into_iter().enumerate() {
        let destination = basis.frame.materialize(destination_local);
        for (source, coefficient) in columns.row(destination_row).iter().enumerate() {
            if let Some(coefficient) = coefficient {
                rows_per_source[source].push((destination.clone(), coefficient.clone()));
            }
        }
    }
    rows_per_source
}

fn order_compact_block<S: Clone>(
    basis: CompactMultiplicityFreeTreePairBasis,
    mut columns: DenseColumns<S>,
) -> OrderedBlockLinearMap<FusionTreePairKey, S> {
    #[cfg(test)]
    COMPACT_BLOCK_DIMENSIONS.with(|dimensions| {
        dimensions.set(Some(CompactBlockDimensions {
            destination_rows: basis.locals.len(),
            source_columns: columns.num_src,
            coefficient_slots: columns.data.len(),
            coefficient_bytes: columns
                .data
                .len()
                .saturating_mul(std::mem::size_of::<Option<S>>()),
        }));
    });

    let source_count = columns.num_src;
    let basis_row_count = basis.locals.len();
    let mut ordered_basis_rows = Vec::with_capacity(basis_row_count);
    let mut ordered_row_for_basis = vec![usize::MAX; basis_row_count];
    let mut singleton_basis_rows = Vec::with_capacity(source_count);
    let mut is_singleton = true;

    for source in 0..source_count {
        let mut only_basis_row = None;
        for (basis_row, ordered_row) in ordered_row_for_basis.iter_mut().enumerate() {
            if columns.row(basis_row)[source].is_none() {
                continue;
            }
            if *ordered_row == usize::MAX {
                *ordered_row = ordered_basis_rows.len();
                ordered_basis_rows.push(basis_row);
            }
            if only_basis_row.replace(basis_row).is_some() {
                is_singleton = false;
            }
        }
        match only_basis_row {
            Some(basis_row) => singleton_basis_rows.push(basis_row),
            None => {
                is_singleton = false;
                singleton_basis_rows.push(usize::MAX);
            }
        }
    }

    let CompactMultiplicityFreeTreePairBasis { frame, locals } = basis;
    let mut local_slots = locals.into_iter().map(Some).collect::<Vec<_>>();
    let destinations = ordered_basis_rows
        .iter()
        .map(|&basis_row| {
            frame.materialize(
                local_slots[basis_row]
                    .take()
                    .expect("ordered compact rows contain each basis row once"),
            )
        })
        .collect();

    let storage = if is_singleton {
        let mut destination_rows = Vec::with_capacity(source_count);
        let mut coefficients = Vec::with_capacity(source_count);
        for (source, basis_row) in singleton_basis_rows.into_iter().enumerate() {
            destination_rows.push(ordered_row_for_basis[basis_row]);
            coefficients.push(
                columns.data[basis_row * source_count + source]
                    .take()
                    .expect("singleton source has one present coefficient"),
            );
        }
        OrderedBlockLinearStorage::SingletonColumns {
            destination_rows,
            coefficients,
        }
    } else {
        let mut coefficients =
            Vec::with_capacity(ordered_basis_rows.len().saturating_mul(source_count));
        for basis_row in ordered_basis_rows {
            let row_start = basis_row * source_count;
            coefficients.extend(
                columns.data[row_start..row_start + source_count]
                    .iter_mut()
                    .map(Option::take),
            );
        }
        OrderedBlockLinearStorage::DenseDstSrc(coefficients)
    };

    OrderedBlockLinearMap {
        destinations,
        source_count,
        storage,
    }
}

fn scatter_compact_tree_pair_block<S: Clone>(
    block: CompactMultiplicityFreeTreePairBlock<S>,
) -> CompactMultiplicityFreeTreePairRows<S> {
    if block.records_dimensions {
        scatter_compact_block(block.basis, block.columns)
    } else {
        let mut rows_per_source = vec![Vec::new(); block.columns.num_src];
        for (destination_row, destination_local) in
            block.basis.locals.into_iter().enumerate()
        {
            let destination = block.basis.frame.materialize(destination_local);
            for (source, coefficient) in
                block.columns.row(destination_row).iter().enumerate()
            {
                if let Some(coefficient) = coefficient {
                    rows_per_source[source]
                        .push((destination.clone(), coefficient.clone()));
                }
            }
        }
        rows_per_source
    }
}

fn order_compact_tree_pair_block<S: Clone>(
    block: CompactMultiplicityFreeTreePairBlock<S>,
) -> OrderedBlockLinearMap<FusionTreePairKey, S> {
    order_compact_block(block.basis, block.columns)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompactBlockDimensions {
    destination_rows: usize,
    source_columns: usize,
    coefficient_slots: usize,
    coefficient_bytes: usize,
}

#[cfg(test)]
thread_local! {
    static COMPACT_BLOCK_DIMENSIONS: std::cell::Cell<Option<CompactBlockDimensions>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn reset_compact_block_dimensions() {
    COMPACT_BLOCK_DIMENSIONS.with(|dimensions| dimensions.set(None));
}

#[cfg(test)]
fn compact_block_dimensions() -> Option<CompactBlockDimensions> {
    COMPACT_BLOCK_DIMENSIONS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn assert_compact_tree_pair_basis_matches_homspace<R>(
    rule: &R,
    basis: &CompactMultiplicityFreeTreePairBasis,
) where
    R: MultiplicityFreeFusionRule,
{
    let product_space = |frame: &MultiplicityFreeTreeFrame| {
        FusionProductSpace::new(
            frame
                .uncoupled
                .iter()
                .copied()
                .zip(frame.is_dual.iter().copied())
                .map(|(sector, is_dual)| SectorLeg::new([(sector, 1)], is_dual)),
        )
    };
    let hom_space = FusionTreeHomSpace::new(
        product_space(&basis.frame.codomain),
        product_space(&basis.frame.domain),
    );
    let expected = hom_space.fusion_tree_keys(rule);
    let actual = basis
        .locals
        .iter()
        .cloned()
        .map(|local| basis.frame.materialize(local))
        .collect::<Vec<_>>();
    assert_eq!(actual.as_slice(), expected.as_ref());
}

fn preflight_compact_repartition_source_major<R>(
    group: &ValidatedTreePairBlockGroup<'_, R>,
    current_codomain_rank: usize,
    target_codomain_rank: usize,
) -> Result<(), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let rule = group.rule;
    let (initial_frame, first_local) = MultiplicityFreeTreePairFrame::split(
        group
            .projection
            .pair_at(0)
            .ok_or(CoreError::MalformedFusionTree {
                message: "compact repartition requires at least one source",
            })?,
    );
    if current_codomain_rank > target_codomain_rank {
        let mut steps: SmallVec<[PreparedMultiplicityFreeBendRight; 8]> =
            SmallVec::new();
        let mut frame = initial_frame.clone();
        let mut local = first_local;
        let num_steps = current_codomain_rank - target_codomain_rank;
        for step in 0..num_steps {
            let prepared = prepare_multiplicity_free_bendright(rule, &frame)?;
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            local = validated.local;
            if step + 1 == num_steps {
                prepared.validate_output_frame()?;
            } else {
                frame = prepared.output_frame(rule)?;
            }
            steps.push(prepared);
        }
        for index in 1..group.source_len {
            let (source_frame, mut local) =
                MultiplicityFreeTreePairFrame::split(
                    group
                        .projection
                        .pair_at(index)
                        .expect("validated projection covers every source"),
                );
            if source_frame != initial_frame {
                return Err(CoreError::MalformedFusionTree {
                    message: TREE_PAIR_BLOCK_GROUP_ERROR,
                });
            }
            for prepared in &steps {
                local = prepared
                    .validate_local(rule, &local.codomain, &local.domain)?
                    .local;
            }
        }
    } else {
        let mut steps: SmallVec<[PreparedMultiplicityFreeBendLeft; 8]> =
            SmallVec::new();
        let mut frame = initial_frame.clone();
        let mut local = first_local;
        let num_steps = target_codomain_rank - current_codomain_rank;
        for step in 0..num_steps {
            let prepared = prepare_multiplicity_free_bendleft(rule, &frame)?;
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            local = PreparedMultiplicityFreeBendLeft::finish_local_structure(validated);
            if step + 1 == num_steps {
                prepared.validate_output_frame()?;
            } else {
                frame = prepared.output_frame(rule)?;
            }
            steps.push(prepared);
        }
        for index in 1..group.source_len {
            let (source_frame, mut local) =
                MultiplicityFreeTreePairFrame::split(
                    group
                        .projection
                        .pair_at(index)
                        .expect("validated projection covers every source"),
                );
            if source_frame != initial_frame {
                return Err(CoreError::MalformedFusionTree {
                    message: TREE_PAIR_BLOCK_GROUP_ERROR,
                });
            }
            for prepared in &steps {
                let validated =
                    prepared.validate_local(rule, &local.codomain, &local.domain)?;
                local =
                    PreparedMultiplicityFreeBendLeft::finish_local_structure(validated);
            }
        }
    }
    Ok(())
}

fn compact_repartition_tree_pair_block<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    target_codomain_rank: usize,
) -> Result<CompactMultiplicityFreeTreePairBlock<R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rule = group.rule;
    let mut current_codomain_rank = group.codomain_rank;
    if current_codomain_rank == target_codomain_rank {
        let source_len = group.source_len;
        let basis = CompactMultiplicityFreeTreePairBasis::from_group(group)?;
        let mut columns = DenseColumns::with_capacity(source_len, source_len);
        for source in 0..source_len {
            let row = columns.push_empty_row();
            columns.row_mut(row)[source] = Some(rule.scalar_one());
        }
        return Ok(CompactMultiplicityFreeTreePairBlock {
            basis,
            columns,
            records_dimensions: false,
        });
    }
    // Why not rely on the step-major compact execution for malformed inputs:
    // the legacy public API reports the first error in source-major order.
    preflight_compact_repartition_source_major(
        &group,
        current_codomain_rank,
        target_codomain_rank,
    )?;
    let (mut frame, first_local) = MultiplicityFreeTreePairFrame::split(
        group
            .projection
            .pair_at(0)
            .ok_or(CoreError::MalformedFusionTree {
                message: "compact repartition requires at least one source",
            })?,
    );
    let mut rows = Vec::with_capacity(group.source_len);
    rows.push((first_local, rule.scalar_one()));
    for index in 1..group.source_len {
        let (source_frame, source_local) =
            MultiplicityFreeTreePairFrame::split(
                group
                    .projection
                    .pair_at(index)
                    .expect("validated projection covers every source"),
            );
        if source_frame != frame {
            return Err(CoreError::MalformedFusionTree {
                message: TREE_PAIR_BLOCK_GROUP_ERROR,
            });
        }
        rows.push((source_local, rule.scalar_one()));
    }

    while current_codomain_rank > target_codomain_rank {
        let prepared = prepare_multiplicity_free_bendright(rule, &frame)?;
        for (local, coefficient) in &mut rows {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            let step_coefficient = prepared.coefficient(rule, &validated);
            *local = validated.local;
            *coefficient = coefficient.clone() * step_coefficient;
        }
        frame = prepared.output_frame(rule)?;
        current_codomain_rank -= 1;
    }
    while current_codomain_rank < target_codomain_rank {
        let prepared = prepare_multiplicity_free_bendleft(rule, &frame)?;
        for (local, coefficient) in &mut rows {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            let (next_local, step_coefficient) =
                prepared.finish_local(rule, validated);
            *local = next_local;
            *coefficient = coefficient.clone() * step_coefficient;
        }
        frame = prepared.output_frame(rule)?;
        current_codomain_rank += 1;
    }
    let source_count = rows.len();
    let mut destination_locals = Vec::with_capacity(source_count);
    let mut columns = DenseColumns::with_capacity(source_count, source_count);
    for (source, (local, coefficient)) in rows.into_iter().enumerate() {
        // Repartition is an invertible change of the tree split, so distinct
        // source basis states stay distinct. Why not hash the finished locals:
        // that would rebuild identity already proved by the reversible bend
        // sequence merely to discover the source-major row number again.
        destination_locals.push(local);
        let destination_row = columns.push_empty_row();
        debug_assert_eq!(destination_row, source);
        columns.row_mut(destination_row)[source] = Some(coefficient);
    }
    Ok(CompactMultiplicityFreeTreePairBlock {
        basis: CompactMultiplicityFreeTreePairBasis {
            frame,
            locals: destination_locals,
        },
        columns,
        records_dimensions: false,
    })
}

#[derive(Clone, Copy)]
struct ValidatedFusionTreeBlockGroup<'a, R> {
    rule: &'a R,
    src_keys: &'a [FusionTreeKey],
    rank: usize,
    projection: MultiplicityFreeTreeProjection<'a>,
}

fn validate_fusion_tree_block_group_for_rule<'a, R>(
    rule: &'a R,
    src_keys: &'a [FusionTreeKey],
) -> Result<Option<ValidatedFusionTreeBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    for source in src_keys {
        validate_fusion_tree_for_rule(rule, source)?;
    }
    validate_fusion_tree_block_group_proven(rule, src_keys)
}

fn validate_fusion_tree_block_group_proven<'a, R>(
    rule: &'a R,
    src_keys: &'a [FusionTreeKey],
) -> Result<Option<ValidatedFusionTreeBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    let projection = MultiplicityFreeTreeProjection::from_validated(rule, src_keys)?;
    let Some(reference) = src_keys.first() else {
        return Ok(None);
    };
    let same_group = |key: &FusionTreeKey| {
        key.uncoupled() == reference.uncoupled()
            && key.is_dual() == reference.is_dual()
    };
    for source in src_keys {
        // Why not compare `coupled`: distinct coupled labels are basis states
        // within one external-sector group, notably for non-Abelian SU(2)
        // blocks. Group membership is checked source-by-source so the first
        // malformed source is reported before any later group mismatch.
        if !same_group(source) {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion-tree keys must share one group",
            });
        }
    }
    Ok(Some(ValidatedFusionTreeBlockGroup {
        rule,
        src_keys,
        rank: reference.uncoupled().len(),
        projection,
    }))
}

#[allow(clippy::type_complexity)]
pub(crate) fn multiplicity_free_braid_tree_block_proven<R>(
    batch: ValidatedMultiplicityFreeTreeBatch<'_, R>,
    permutation: &[usize],
    levels: &[usize],
) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let (rule, src_keys) = batch.parts();
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let rank = first.uncoupled().len();
    if levels.len() != rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: levels.len(),
        });
    }
    let prepared = PreparedTreeBraid::new(permutation, levels, rank)?;
    let group = validate_fusion_tree_block_group_proven(rule, src_keys)?
        .expect("nonempty proven source block produces a group proof");
    multiplicity_free_braid_tree_block_validated(group, prepared)
}

/// Apply one braid to every source tree in an all-codomain block.
///
/// Sources and result rows retain source order. Floating-point summation can
/// differ from repeated scalar calls, so coefficients agree numerically rather
/// than necessarily bit-for-bit. Empty input returns an empty result. Every
/// nonempty source must share external sectors, duality, and fusion style;
/// malformed mixed groups fail before symbol evaluation.
///
/// Every source follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
#[allow(clippy::type_complexity)]
pub fn multiplicity_free_braid_tree_block<R>(
    rule: &R,
    src_keys: &[FusionTreeKey],
    permutation: &[usize],
    levels: &[usize],
) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    validate_multiplicity_free_execution_style(rule)?;
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let rank = first.uncoupled().len();
    if levels.len() != rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: levels.len(),
        });
    }
    let prepared = PreparedTreeBraid::new(permutation, levels, rank)?;
    let group = validate_fusion_tree_block_group_for_rule(rule, src_keys)?
        .expect("nonempty source block produces a validation proof");
    multiplicity_free_braid_tree_block_validated(group, prepared)
}

#[allow(clippy::type_complexity)]
fn multiplicity_free_braid_tree_block_validated<R>(
    group: ValidatedFusionTreeBlockGroup<'_, R>,
    prepared: PreparedTreeBraid,
) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rule = group.rule;
    let src_keys = group.src_keys;
    if prepared.permutation.iter().copied().eq(0..group.rank) {
        return Ok(src_keys
            .iter()
            .map(|key| vec![(key.clone(), rule.scalar_one())])
            .collect());
    }

    let mut basis = CompactMultiplicityFreeTreeBasis::from_group(group)?;
    let mut columns = None;
    for step in &prepared.artin_steps {
        let (next_basis, next_columns) = match &columns {
            Some(columns) => compact_artin_tree_block_step(
                rule,
                basis,
                columns,
                step.index,
                step.inverse,
            )?,
            None => compact_artin_tree_block_first(
                rule,
                basis,
                step.index,
                step.inverse,
            )?,
        };
        basis = next_basis;
        columns = Some(next_columns);
    }

    // A validated non-identity permutation always contains at least one swap.
    let columns = columns.expect("non-identity permutation produces an Artin step");
    Ok(scatter_compact_tree_block(basis, &columns))
}

/// Symmetric-braiding convenience wrapper for
/// [`multiplicity_free_braid_tree_block`], with the same ordering, validation,
/// and empty-input contract.
///
/// Every source follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
#[allow(clippy::type_complexity)]
pub fn multiplicity_free_permute_tree_block<R>(
    rule: &R,
    src_keys: &[FusionTreeKey],
    permutation: &[usize],
) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
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
    validate_multiplicity_free_execution_style(rule)?;
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let rank = first.uncoupled().len();
    let levels = (0..rank).collect::<SmallVec<[usize; 8]>>();
    let prepared = PreparedTreeBraid::new(permutation, &levels, rank)?;
    let group = validate_fusion_tree_block_group_for_rule(rule, src_keys)?
        .expect("nonempty source block produces a validation proof");
    multiplicity_free_braid_tree_block_validated(group, prepared)
}

#[derive(Clone, Copy)]
struct ValidatedTreePairBlockGroup<'a, R> {
    rule: &'a R,
    source_len: usize,
    codomain_rank: usize,
    domain_rank: usize,
    projection: MultiplicityFreePairProjection<'a>,
}

const TREE_PAIR_BLOCK_GROUP_ERROR: &str = "fusion-tree block keys must share one group";

fn validate_tree_pair_block_group_for_rule<'a, R>(
    rule: &'a R,
    src_keys: &'a [FusionTreePairKey],
) -> Result<Option<ValidatedTreePairBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    for source in src_keys {
        validate_fusion_tree_pair_for_rule(rule, source)?;
    }
    let projection = MultiplicityFreePairProjection::from_validated(rule, src_keys)?;
    validate_tree_pair_block_group_projection(rule, projection)
}

fn validate_tree_pair_block_group_proven<'a, R>(
    rule: &'a R,
    src_keys: &'a [FusionTreePairKey],
) -> Result<Option<ValidatedTreePairBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    let projection = MultiplicityFreePairProjection::from_validated(rule, src_keys)?;
    validate_tree_pair_block_group_projection(rule, projection)
}

fn validate_tree_pair_block_group_structure<'a, R>(
    rule: &'a R,
    structure: &'a BlockStructure,
    indices: &'a [usize],
    orientation: FusionTreePairOrientation,
) -> Result<Option<ValidatedTreePairBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    let projection =
        MultiplicityFreePairProjection::checked_structure(rule, structure, indices, orientation)?;
    validate_tree_pair_block_group_projection(rule, projection)
}

fn validate_tree_pair_block_group_projection<'a, R>(
    rule: &'a R,
    projection: MultiplicityFreePairProjection<'a>,
) -> Result<Option<ValidatedTreePairBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    let Some(reference) = projection.pair_at(0) else {
        return Ok(None);
    };
    let reference_codomain = reference.codomain().key();
    let reference_domain = reference.domain().key();
    let same_group = |pair: ValidatedMultiplicityFreeTreePair<'_>| {
        let codomain = pair.codomain().key();
        let domain = pair.domain().key();
        codomain.uncoupled() == reference_codomain.uncoupled()
            && domain.uncoupled() == reference_domain.uncoupled()
            && codomain.is_dual() == reference_codomain.is_dual()
            && domain.is_dual() == reference_domain.is_dual()
    };
    for index in 0..projection.len() {
        let source = projection
            .pair_at(index)
            .expect("validated projection covers every source");
        // Why not infer a block from matching ranks or tree shape:
        // coefficients share a basis only when every external sector,
        // orientation, and multiplicity invariant agrees. Coupled sectors are
        // basis states within one group and intentionally need not match.
        if !same_group(source) {
            return Err(CoreError::MalformedFusionTree {
                message: TREE_PAIR_BLOCK_GROUP_ERROR,
            });
        }
    }
    Ok(Some(ValidatedTreePairBlockGroup {
        rule,
        source_len: projection.len(),
        codomain_rank: reference_codomain.uncoupled().len(),
        domain_rank: reference_domain.uncoupled().len(),
        projection,
    }))
}

#[allow(clippy::type_complexity)]
pub(crate) fn multiplicity_free_braid_tree_pair_block_proven<R>(
    batch: ValidatedMultiplicityFreePairBatch<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let (rule, src_keys) = batch.parts();
    let Some(group) = validate_tree_pair_block_group_proven(rule, src_keys)? else {
        return Ok(Vec::new());
    };
    multiplicity_free_braid_tree_pair_block_validated(group, prepared)
}

pub(crate) fn multiplicity_free_braid_tree_pair_block_ordered_proven<R>(
    batch: ValidatedMultiplicityFreePairBatch<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let (rule, src_keys) = batch.parts();
    let Some(group) = validate_tree_pair_block_group_proven(rule, src_keys)? else {
        return Ok(OrderedBlockLinearMap {
            destinations: Vec::new(),
            source_count: 0,
            storage: OrderedBlockLinearStorage::DenseDstSrc(Vec::new()),
        });
    };
    multiplicity_free_braid_tree_pair_block_ordered_validated(group, prepared)
}

#[allow(clippy::type_complexity)]
pub(crate) fn multiplicity_free_transpose_tree_pair_block_proven<R>(
    batch: ValidatedMultiplicityFreePairBatch<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let (rule, src_keys) = batch.parts();
    let Some(group) = validate_tree_pair_block_group_proven(rule, src_keys)? else {
        return Ok(Vec::new());
    };
    multiplicity_free_transpose_tree_pair_block_validated(group, prepared)
}

pub(crate) fn multiplicity_free_transpose_tree_pair_block_ordered_proven<R>(
    batch: ValidatedMultiplicityFreePairBatch<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let (rule, src_keys) = batch.parts();
    let Some(group) = validate_tree_pair_block_group_proven(rule, src_keys)? else {
        return Ok(OrderedBlockLinearMap {
            destinations: Vec::new(),
            source_count: 0,
            storage: OrderedBlockLinearStorage::DenseDstSrc(Vec::new()),
        });
    };
    multiplicity_free_transpose_tree_pair_block_ordered_validated(group, prepared)
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
///
/// `src_keys` must be empty or share one external-sector, duality, and
/// multiplicity group. Empty input returns an empty transform; mixed groups
/// return [`CoreError::MalformedFusionTree`] before any symbol evaluation.
///
/// Every source follows
/// [`FusionTreePairKey::validate_for_rule`]'s provider-domain precondition.
pub fn multiplicity_free_braid_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    validate_multiplicity_free_execution_style(rule)?;
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let codomain_rank = first.codomain_tree().uncoupled().len();
    let domain_rank = first.domain_tree().uncoupled().len();
    let prepared = PreparedTreePairOperation::prepare_braid(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )?;
    let group = validate_tree_pair_block_group_for_rule(rule, src_keys)?
        .expect("nonempty source block produces a validation proof");
    multiplicity_free_braid_tree_pair_block_validated(group, &prepared)
}

#[allow(clippy::type_complexity)]
fn multiplicity_free_braid_tree_pair_block_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    prepared.validate_source_split(group.codomain_rank, group.domain_rank)?;
    if prepared.is_identity() {
        return Ok((0..group.source_len)
            .map(|index| {
                let key = group
                    .projection
                    .pair_at(index)
                    .expect("validated projection covers every source")
                    .materialize();
                vec![(key, group.rule.scalar_one())]
            })
            .collect());
    }
    multiplicity_free_braid_tree_pair_block_compact_validated(group, prepared)
        .map(scatter_compact_tree_pair_block)
}

fn multiplicity_free_braid_tree_pair_block_ordered_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    prepared.validate_source_split(group.codomain_rank, group.domain_rank)?;
    multiplicity_free_braid_tree_pair_block_compact_validated(group, prepared)
        .map(order_compact_tree_pair_block)
}

fn multiplicity_free_braid_tree_pair_block_compact_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<CompactMultiplicityFreeTreePairBlock<R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rule = group.rule;
    let codomain_rank = group.codomain_rank;
    let domain_rank = group.domain_rank;
    match &prepared.plan {
        PreparedTreePairPlan::Identity => {
            return compact_repartition_tree_pair_block(group, codomain_rank);
        }
        PreparedTreePairPlan::Repartition => {
            return compact_repartition_tree_pair_block(
                group,
                prepared.target_codomain_rank,
            );
        }
        PreparedTreePairPlan::Braid(_) | PreparedTreePairPlan::UniqueBraid(_) => {}
        PreparedTreePairPlan::Transpose { .. } => {
            unreachable!("braid preparation cannot create a transpose plan")
        }
    }
    let all_rank = codomain_rank + domain_rank;

    // The first compact operator writes source columns directly; later
    // operators compose through the resulting dense coefficient matrix.
    let mut basis = CompactMultiplicityFreeTreePairBasis::from_group(group)?;
    let mut columns = None;

    // Step A: repartition everything into the codomain (bendleft chain).
    let mut current_codomain_rank = codomain_rank;
    while current_codomain_rank < all_rank {
        let (next_basis, next_columns) = match columns.take() {
            Some(columns) => compact_bendleft_block_step(rule, basis, &columns)?,
            None => compact_bendleft_block_first(rule, basis)?,
        };
        basis = next_basis;
        columns = Some(next_columns);
        current_codomain_rank += 1;
    }

    // Step B: braid the (now all-codomain) tree ONE adjacent swap at a time,
    // each swap batched across the whole block. This replaces the per-source
    // inner braid (`multiplicity_free_braid_tree`, whose `FusionTermAccumulator`
    // and elementary-swap term lists ran once per source tree) with the shared
    // block matrix walk — the TensorKit 0.17 `artin_braid`-on-a-block scheme.
    for step in prepared
        .plan
        .artin_steps()
        .expect("braid preparation has Artin steps")
    {
        let (next_basis, next_columns) = match columns.take() {
            Some(columns) => compact_codomain_artin_block_step(
                rule,
                basis,
                &columns,
                step.index,
                step.inverse,
            )?,
            None => compact_codomain_artin_block_first(
                rule,
                basis,
                step.index,
                step.inverse,
            )?,
        };
        basis = next_basis;
        columns = Some(next_columns);
    }

    // Step C: repartition back to the requested codomain rank.
    let target_codomain_rank = prepared.target_codomain_rank;
    while current_codomain_rank > target_codomain_rank {
        let (next_basis, next_columns) = match columns.take() {
            Some(columns) => compact_bendright_block_step(rule, basis, &columns)?,
            None => compact_bendright_block_first(rule, basis)?,
        };
        basis = next_basis;
        columns = Some(next_columns);
        current_codomain_rank -= 1;
    }
    while current_codomain_rank < target_codomain_rank {
        let (next_basis, next_columns) = match columns.take() {
            Some(columns) => compact_bendleft_block_step(rule, basis, &columns)?,
            None => compact_bendleft_block_first(rule, basis)?,
        };
        basis = next_basis;
        columns = Some(next_columns);
        current_codomain_rank += 1;
    }

    // Why not materialize after each braid: both block runners keep the
    // external frame immutable and reconstruct full keys only at the API edge.
    Ok(CompactMultiplicityFreeTreePairBlock {
        basis,
        columns: columns.expect("nonidentity braid executes at least one compact operator"),
        records_dimensions: true,
    })
}

/// Batched [`multiplicity_free_permute_tree_pair`] over a block: symmetric
/// braiding with the trivial level ordering.
///
/// The group contract is identical to
/// [`multiplicity_free_braid_tree_pair_block`]. Symmetric braiding remains a
/// required capability even for an empty source block.
///
/// Every source follows
/// [`FusionTreePairKey::validate_for_rule`]'s provider-domain precondition.
pub fn multiplicity_free_permute_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
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
    validate_multiplicity_free_execution_style(rule)?;
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let codomain_rank = first.codomain_tree().uncoupled().len();
    let domain_rank = first.domain_tree().uncoupled().len();
    let prepared = PreparedTreePairOperation::prepare_permute(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?;
    let group = validate_tree_pair_block_group_for_rule(rule, src_keys)?
        .expect("nonempty source block produces a validation proof");
    multiplicity_free_braid_tree_pair_block_validated(group, &prepared)
}

#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn multiplicity_free_permute_tree_pair_block_indexed<R>(
    rule: &R,
    structure: &BlockStructure,
    src_indices: &[usize],
    orientation: FusionTreePairOrientation,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
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
    validate_multiplicity_free_execution_style(rule)?;
    let Some(&first_index) = src_indices.first() else {
        return Ok(Vec::new());
    };
    let first = structure.block(first_index)?;
    let BlockKey::FusionTree(first) = first.key() else {
        return Err(CoreError::ExpectedFusionTreePairKey {
            actual: first.key().kind(),
        });
    };
    let (codomain_rank, domain_rank) = match orientation {
        FusionTreePairOrientation::Direct => (
            first.codomain_tree().uncoupled().len(),
            first.domain_tree().uncoupled().len(),
        ),
        FusionTreePairOrientation::Adjoint => (
            first.domain_tree().uncoupled().len(),
            first.codomain_tree().uncoupled().len(),
        ),
    };
    let prepared = PreparedTreePairOperation::prepare_permute(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?;
    let group =
        validate_tree_pair_block_group_structure(rule, structure, src_indices, orientation)?
            .expect("nonempty source block produces a validation proof");
    multiplicity_free_braid_tree_pair_block_validated(group, &prepared)
}

#[doc(hidden)]
pub fn multiplicity_free_braid_tree_pair_block_ordered_indexed<R>(
    rule: &R,
    structure: &BlockStructure,
    src_indices: &[usize],
    orientation: FusionTreePairOrientation,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    validate_multiplicity_free_execution_style(rule)?;
    prepared.validate_block_preflight(rule, PreparedTreePairFamily::BraidLike)?;
    let Some(group) =
        validate_tree_pair_block_group_structure(rule, structure, src_indices, orientation)?
    else {
        return Ok(OrderedBlockLinearMap {
            destinations: Vec::new(),
            source_count: 0,
            storage: OrderedBlockLinearStorage::DenseDstSrc(Vec::new()),
        });
    };
    multiplicity_free_braid_tree_pair_block_ordered_validated(group, prepared)
}

#[doc(hidden)]
pub fn multiplicity_free_transpose_tree_pair_block_ordered_indexed<R>(
    rule: &R,
    structure: &BlockStructure,
    src_indices: &[usize],
    orientation: FusionTreePairOrientation,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    validate_multiplicity_free_execution_style(rule)?;
    prepared.validate_block_preflight(rule, PreparedTreePairFamily::Transpose)?;
    let Some(group) =
        validate_tree_pair_block_group_structure(rule, structure, src_indices, orientation)?
    else {
        return Ok(OrderedBlockLinearMap {
            destinations: Vec::new(),
            source_count: 0,
            storage: OrderedBlockLinearStorage::DenseDstSrc(Vec::new()),
        });
    };
    multiplicity_free_transpose_tree_pair_block_ordered_validated(group, prepared)
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
///
/// The empty/group contract is identical to
/// [`multiplicity_free_braid_tree_pair_block`].
///
/// Every source follows
/// [`FusionTreePairKey::validate_for_rule`]'s provider-domain precondition.
pub fn multiplicity_free_transpose_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    validate_multiplicity_free_execution_style(rule)?;
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let codomain_rank = first.codomain_tree().uncoupled().len();
    let domain_rank = first.domain_tree().uncoupled().len();
    let prepared = PreparedTreePairOperation::prepare_transpose(
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?;
    let group = validate_tree_pair_block_group_for_rule(rule, src_keys)?
        .expect("nonempty source block produces a validation proof");
    multiplicity_free_transpose_tree_pair_block_validated(group, &prepared)
}

#[allow(clippy::type_complexity)]
fn multiplicity_free_transpose_tree_pair_block_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    prepared.validate_source_split(group.codomain_rank, group.domain_rank)?;
    if prepared.is_identity() {
        return Ok((0..group.source_len)
            .map(|index| {
                let key = group
                    .projection
                    .pair_at(index)
                    .expect("validated projection covers every source")
                    .materialize();
                vec![(key, group.rule.scalar_one())]
            })
            .collect());
    }
    multiplicity_free_transpose_tree_pair_block_compact_validated(group, prepared)
        .map(scatter_compact_tree_pair_block)
}

fn multiplicity_free_transpose_tree_pair_block_ordered_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    prepared.validate_source_split(group.codomain_rank, group.domain_rank)?;
    multiplicity_free_transpose_tree_pair_block_compact_validated(group, prepared)
        .map(order_compact_tree_pair_block)
}

fn multiplicity_free_transpose_tree_pair_block_compact_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<CompactMultiplicityFreeTreePairBlock<R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rule = group.rule;
    let codomain_rank = group.codomain_rank;
    let cycle = match &prepared.plan {
        PreparedTreePairPlan::Identity => {
            return compact_repartition_tree_pair_block(group, codomain_rank);
        }
        PreparedTreePairPlan::Repartition => {
            return compact_repartition_tree_pair_block(
                group,
                prepared.target_codomain_rank,
            );
        }
        PreparedTreePairPlan::Transpose { direction, count } => {
            (*direction, *count)
        }
        PreparedTreePairPlan::Braid(_) | PreparedTreePairPlan::UniqueBraid(_) => {
            unreachable!("transpose preparation cannot create a braid plan")
        }
    };

    let mut basis = CompactMultiplicityFreeTreePairBasis::from_group(group)?;
    let mut columns = None;

    let target_codomain_rank = prepared.target_codomain_rank;
    let mut current_codomain_rank = codomain_rank;
    while current_codomain_rank < target_codomain_rank {
        let (next_basis, next_columns) = match columns.take() {
            Some(columns) => compact_bendleft_block_step(rule, basis, &columns)?,
            None => compact_bendleft_block_first(rule, basis)?,
        };
        basis = next_basis;
        columns = Some(next_columns);
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        let (next_basis, next_columns) = match columns.take() {
            Some(columns) => compact_bendright_block_step(rule, basis, &columns)?,
            None => compact_bendright_block_first(rule, basis)?,
        };
        basis = next_basis;
        columns = Some(next_columns);
        current_codomain_rank -= 1;
    }

    for _ in 0..cycle.1 {
        let (next_basis, next_columns) = match (cycle.0, columns.take()) {
            (PreparedCycleDirection::Clockwise, Some(columns)) => {
                compact_cycle_clockwise_block_step(rule, basis, &columns)?
            }
            (PreparedCycleDirection::Clockwise, None) => {
                compact_cycle_clockwise_block_first(rule, basis)?
            }
            (PreparedCycleDirection::Anticlockwise, Some(columns)) => {
                compact_cycle_anticlockwise_block_step(rule, basis, &columns)?
            }
            (PreparedCycleDirection::Anticlockwise, None) => {
                compact_cycle_anticlockwise_block_first(rule, basis)?
            }
        };
        basis = next_basis;
        columns = Some(next_columns);
    }

    // Why not materialize after each fold: the external frame is identical for
    // every local row and remains immutable until this ordered API boundary.
    #[cfg(test)]
    assert_compact_tree_pair_basis_matches_homspace(rule, &basis);
    Ok(CompactMultiplicityFreeTreePairBlock {
        basis,
        columns: columns.expect("nonidentity transpose executes at least one compact operator"),
        records_dimensions: true,
    })
}

#[cfg(test)]
type MultiplicityFreeTreePairBlockRows<S> = Vec<Vec<(FusionTreePairKey, S)>>;

#[cfg(test)]
fn multiplicity_free_foldright_tree_pair_legacy_oracle<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let codomain = tree_pair.codomain_tree();
    if codomain.uncoupled().is_empty() {
        return Err(CoreError::MalformedFusionTree {
            message: "foldright requires at least one codomain leg",
        });
    }
    let a = codomain.uncoupled()[0];
    let is_dual_a =
        codomain
            .is_dual()
            .first()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "codomain tree is missing the first duality flag",
            })?;
    let kappa = rule.frobenius_schur_phase_scalar(a);
    let c = codomain.coupled();

    let mut terms = FusionTermAccumulator::new();
    for (codomain_prime, coeff1) in
        multiplicity_free_multi_fmove_tree_legacy_oracle(rule, codomain)?
    {
        let b = codomain_prime.coupled();
        let a_symbol = rule.a_symbol_scalar(a, b, c);
        let coeff0 = rule.sqrt_dim_scalar(c) * rule.inv_sqrt_dim_scalar(b);
        for (domain_prime, coeff2) in multiplicity_free_multi_fmove_inv_tree_legacy_oracle(
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
                FusionTreePairKey::pair(codomain_prime.clone(), domain_prime),
                coefficient,
            );
        }
    }
    Ok(terms.into_vec())
}

#[cfg(test)]
fn multiplicity_free_foldleft_tree_pair_legacy_oracle<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let swapped = FusionTreePairKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(
        multiplicity_free_foldright_tree_pair_legacy_oracle(rule, &swapped)?
            .into_iter()
            .map(|(folded, coefficient)| {
                (
                    FusionTreePairKey::pair(
                        folded.domain_tree().clone(),
                        folded.codomain_tree().clone(),
                    ),
                    rule.scalar_conj(coefficient),
                )
            })
            .collect(),
    )
}

#[cfg(test)]
fn multiplicity_free_cycle_clockwise_tree_pair_legacy_oracle<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let first: Vec<_> = if tree_pair.codomain_tree().uncoupled().is_empty() {
        multiplicity_free_bendleft_tree_pair(rule, tree_pair)?
            .into_iter()
            .collect()
    } else {
        multiplicity_free_foldright_tree_pair_legacy_oracle(rule, tree_pair)?
    };
    if tree_pair.codomain_tree().uncoupled().is_empty() {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_foldright_tree_pair_legacy_oracle(rule, key)
        })
    } else {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_bendleft_tree_pair(rule, key)
        })
    }
}

#[cfg(test)]
fn multiplicity_free_cycle_anticlockwise_tree_pair_legacy_oracle<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let first: Vec<_> = if tree_pair.domain_tree().uncoupled().is_empty() {
        multiplicity_free_bendright_tree_pair(rule, tree_pair)?
            .into_iter()
            .collect()
    } else {
        multiplicity_free_foldleft_tree_pair_legacy_oracle(rule, tree_pair)?
    };
    if tree_pair.domain_tree().uncoupled().is_empty() {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_foldleft_tree_pair_legacy_oracle(rule, key)
        })
    } else {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_bendright_tree_pair(rule, key)
        })
    }
}

#[cfg(test)]
fn multiplicity_free_transpose_tree_pair_block_full_key_oracle<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<MultiplicityFreeTreePairBlockRows<R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let Some(group) = validate_tree_pair_block_group_for_rule(rule, src_keys)? else {
        return Ok(Vec::new());
    };
    let prepared = PreparedTreePairOperation::prepare_transpose(
        group.codomain_rank,
        group.domain_rank,
        codomain_permutation,
        domain_permutation,
    )?;
    let cycle = match &prepared.plan {
        PreparedTreePairPlan::Identity => {
            return Ok(src_keys
                .iter()
                .map(|key| vec![(key.clone(), rule.scalar_one())])
                .collect());
        }
        PreparedTreePairPlan::Repartition => None,
        PreparedTreePairPlan::Transpose { direction, count } => Some((*direction, *count)),
        PreparedTreePairPlan::Braid(_) | PreparedTreePairPlan::UniqueBraid(_) => {
            unreachable!("transpose preparation cannot create a braid plan")
        }
    };

    let num_src = src_keys.len();
    let mut basis = src_keys.to_vec();
    let mut columns = DenseColumns::with_capacity(num_src, num_src);
    for source in 0..num_src {
        let row = columns.push_empty_row();
        columns.row_mut(row)[source] = Some(rule.scalar_one());
    }

    let target_codomain_rank = prepared.target_codomain_rank;
    let mut current_codomain_rank = group.codomain_rank;
    while current_codomain_rank < target_codomain_rank {
        (basis, columns) = compose_block_terms(rule, &basis, &columns, |rule, key| {
            multiplicity_free_bendleft_tree_pair(rule, key)
        })?;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        (basis, columns) = compose_block_terms(rule, &basis, &columns, |rule, key| {
            multiplicity_free_bendright_tree_pair(rule, key)
        })?;
        current_codomain_rank -= 1;
    }

    if let Some((direction, count)) = cycle {
        for _ in 0..count {
            (basis, columns) = match direction {
                PreparedCycleDirection::Clockwise => {
                    compose_block_terms(rule, &basis, &columns, |rule, key| {
                        multiplicity_free_cycle_clockwise_tree_pair_legacy_oracle(rule, key)
                    })?
                }
                PreparedCycleDirection::Anticlockwise => {
                    compose_block_terms(rule, &basis, &columns, |rule, key| {
                        multiplicity_free_cycle_anticlockwise_tree_pair_legacy_oracle(rule, key)
                    })?
                }
            };
        }
    }

    let mut rows_per_source = vec![Vec::new(); num_src];
    for (destination_row, destination) in basis.iter().enumerate() {
        for (source, coefficient) in columns.row(destination_row).iter().enumerate() {
            if let Some(coefficient) = coefficient {
                rows_per_source[source].push((destination.clone(), coefficient.clone()));
            }
        }
    }
    Ok(rows_per_source)
}

fn multiplicity_free_repartition_terms<R>(
    rule: &R,
    terms: Vec<(FusionTreePairKey, R::Scalar)>,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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

/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn unique_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
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
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: rule.fusion_style(),
        });
    }
    PreparedTreePairOperation::prepare_braid(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )?
    .execute_unique_pivotal(rule, tree_pair)
}

/// Exact rigid tree-pair braid lowering for a unique fusion rule.
///
/// This is an implementation hook for the tensor-plan compiler. Public callers
/// should use the multiplicity-free operation APIs.
#[doc(hidden)]
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn unique_rigid_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
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
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: rule.fusion_style(),
        });
    }
    PreparedTreePairOperation::prepare_braid(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )?
    .execute_unique_rigid(rule, tree_pair)
}

/// Exact rigid tree-pair permutation lowering for a unique fusion rule.
///
/// This is an implementation hook for the tensor-plan compiler. Public callers
/// should use the multiplicity-free operation APIs.
#[doc(hidden)]
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn unique_rigid_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: rule.fusion_style(),
        });
    }
    PreparedTreePairOperation::prepare_permute(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?
    .execute_unique_rigid(rule, tree_pair)
}

/// Exact rigid cyclic-transpose lowering for a unique fusion rule.
///
/// This is an implementation hook for the tensor-plan compiler. Public callers
/// should use the multiplicity-free operation APIs.
#[doc(hidden)]
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn unique_rigid_transpose_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    PreparedTreePairOperation::prepare_transpose(
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?
    .execute_unique_rigid(rule, tree_pair)
}

/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn unique_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
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
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: rule.fusion_style(),
        });
    }
    PreparedTreePairOperation::prepare_permute(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?
    .execute_unique_pivotal(rule, tree_pair)
}

/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn unique_transpose_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    PreparedTreePairOperation::prepare_transpose(
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?
    .execute_unique_pivotal(rule, tree_pair)
}

/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn unique_repartition_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
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
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    unique_repartition_tree_pair_validated(validated, target_codomain_rank)
}

fn unique_repartition_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    target_codomain_rank: usize,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rule = tree_pair.rule;
    unique_repartition_tree_pair_unchecked(rule, tree_pair.key, target_codomain_rank)
}

fn unique_repartition_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
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

fn validate_tree_pair_axis_map_inline(
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_rank: usize,
    domain_rank: usize,
) -> Result<(), CoreError> {
    let total_rank = codomain_rank + domain_rank;
    if codomain_permutation.len() + domain_permutation.len() != total_rank {
        return Err(invalid_tree_pair_axis_map(
            codomain_permutation,
            domain_permutation,
            total_rank,
        ));
    }
    let mut seen = SmallVec::<[u64; 2]>::new();
    seen.resize(total_rank.div_ceil(u64::BITS as usize), 0);
    for position in 0..total_rank {
        let axis = raw_tree_pair_axis_at(codomain_permutation, domain_permutation, position);
        if axis >= total_rank {
            return Err(invalid_tree_pair_axis_map(
                codomain_permutation,
                domain_permutation,
                total_rank,
            ));
        }
        let word = axis / u64::BITS as usize;
        let bit = 1u64 << (axis % u64::BITS as usize);
        if seen[word] & bit != 0 {
            return Err(invalid_tree_pair_axis_map(
                codomain_permutation,
                domain_permutation,
                total_rank,
            ));
        }
        seen[word] |= bit;
    }
    Ok(())
}

fn validate_tree_pair_axis_map_without_scratch(
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_rank: usize,
    domain_rank: usize,
) -> Result<(), CoreError> {
    let total_rank = codomain_rank + domain_rank;
    if codomain_permutation.len() + domain_permutation.len() != total_rank {
        return Err(invalid_tree_pair_axis_map(
            codomain_permutation,
            domain_permutation,
            total_rank,
        ));
    }
    // Why not reuse the bitmap validator: a borrowed Unique plan cannot
    // acquire rank-dependent scratch during preparation. Simple compilation
    // keeps the linear bitmap path because it owns and reuses its lowered work.
    for position in 0..total_rank {
        let axis = raw_tree_pair_axis_at(codomain_permutation, domain_permutation, position);
        if axis >= total_rank
            || (0..position).any(|earlier| {
                raw_tree_pair_axis_at(codomain_permutation, domain_permutation, earlier) == axis
            })
        {
            return Err(invalid_tree_pair_axis_map(
                codomain_permutation,
                domain_permutation,
                total_rank,
            ));
        }
    }
    Ok(())
}

fn validate_cyclic_tree_pair_axis_map_inline(
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_rank: usize,
    domain_rank: usize,
) -> Result<(), CoreError> {
    validate_tree_pair_axis_map_inline(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    let total_rank = codomain_rank + domain_rank;
    if total_rank == 0 {
        return Ok(());
    }
    for index in 0..total_rank {
        let current = linearized_tree_pair_axis_at(
            codomain_permutation,
            domain_permutation,
            codomain_rank,
            domain_rank,
            index,
        );
        let next = linearized_tree_pair_axis_at(
            codomain_permutation,
            domain_permutation,
            codomain_rank,
            domain_rank,
            (index + 1) % total_rank,
        );
        if next != (current + 1) % total_rank {
            return Err(CoreError::InvalidPermutation {
                permutation: materialize_linearized_tree_pair_permutation(
                    codomain_permutation,
                    domain_permutation,
                    codomain_rank,
                    domain_rank,
                )
                .into_vec(),
                rank: total_rank,
            });
        }
    }
    Ok(())
}

fn raw_tree_pair_axis_at(
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    position: usize,
) -> usize {
    if position < codomain_permutation.len() {
        codomain_permutation[position]
    } else {
        domain_permutation[position - codomain_permutation.len()]
    }
}

fn linearized_tree_pair_axis_at(
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_rank: usize,
    domain_rank: usize,
    position: usize,
) -> usize {
    let axis = if position < codomain_permutation.len() {
        codomain_permutation[position]
    } else {
        domain_permutation[domain_permutation.len() - 1 - (position - codomain_permutation.len())]
    };
    linearize_tree_pair_axis(axis, codomain_rank, domain_rank)
}

fn materialize_linearized_tree_pair_permutation(
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_rank: usize,
    domain_rank: usize,
) -> SmallVec<[usize; 8]> {
    let total_rank = codomain_rank + domain_rank;
    (0..total_rank)
        .map(|position| {
            linearized_tree_pair_axis_at(
                codomain_permutation,
                domain_permutation,
                codomain_rank,
                domain_rank,
                position,
            )
        })
        .collect()
}

fn invalid_tree_pair_axis_map(
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    rank: usize,
) -> CoreError {
    CoreError::InvalidPermutation {
        permutation: codomain_permutation
            .iter()
            .chain(domain_permutation)
            .copied()
            .collect(),
        rank,
    }
}

fn tree_pair_axis_map_is_identity(
    codomain_axes: &[usize],
    domain_axes: &[usize],
    codomain_rank: usize,
    domain_rank: usize,
) -> bool {
    codomain_axes.iter().copied().eq(0..codomain_rank)
        && domain_axes
            .iter()
            .copied()
            .eq(codomain_rank..codomain_rank + domain_rank)
}

fn execute_unique_tree_braid<R>(
    rule: &R,
    tree: &FusionTreeKey,
    prepared: &PreparedTreeBraid,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rank = tree.uncoupled().len();
    if prepared.permutation.len() != rank {
        return Err(CoreError::InvalidPermutation {
            permutation: prepared.permutation.to_vec(),
            rank,
        });
    }
    // Why not rebuild every Unique key directly: TeNeT's public checked
    // constructor still accepts noncanonical innerline data, whose legacy
    // Artin behavior must not be silently normalized into a different key.
    if rule.braiding_style().is_symmetric()
        && rule.has_trivial_associator_gauge()
        && is_unique_direct_braid_source(rule, tree)
    {
        let mut coefficient = rule.scalar_one();
        for right_position in 0..rank {
            for left_position in 0..right_position {
                let left_axis = prepared.permutation[left_position];
                let right_axis = prepared.permutation[right_position];
                if left_axis > right_axis {
                    let left = tree.uncoupled()[left_axis];
                    let right = tree.uncoupled()[right_axis];
                    // TensorKit treats a unit crossing as structural identity.
                    // Why not ask the provider for R(unit, a): providers are
                    // permitted to omit identity symbols and the Artin path
                    // already skips them.
                    if left == rule.vacuum() || right == rule.vacuum() {
                        continue;
                    }
                    let coupled = only_fusion_channel(rule, left, right)?;
                    coefficient =
                        coefficient * rule.r_symbol_scalar(left, right, coupled);
                }
            }
        }
        let uncoupled = prepared
            .permutation
            .iter()
            .map(|&axis| tree.uncoupled()[axis])
            .collect::<SmallVec<[SectorId; 8]>>();
        let is_dual = prepared
            .permutation
            .iter()
            .map(|&axis| tree.is_dual()[axis])
            .collect::<SmallVec<[bool; 8]>>();
        let coupled = tree.coupled();
        let destination =
            rebuild_unique_standard_fusion_tree(rule, &uncoupled, coupled, &is_dual)?;
        return Ok((destination, coefficient));
    }

    let mut current = tree.clone();
    let mut coefficient = rule.scalar_one();
    for step in &prepared.artin_steps {
        let (next, step_coefficient) = unique_artin_braid_at_with_inverse(
            rule,
            &current,
            step.index,
            step.inverse,
        )?;
        coefficient = coefficient * step_coefficient;
        current = next;
    }
    Ok((current, coefficient))
}

fn execute_unique_tree_braid_borrowed<R>(
    rule: &R,
    tree: &FusionTreeKey,
    braid: &UniqueBorrowedTreePairBraid<'_>,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rank = tree.uncoupled().len();
    if braid.source_codomain_rank + braid.source_domain_rank != rank {
        return Err(CoreError::InvalidPermutation {
            permutation: (0..braid.source_codomain_rank + braid.source_domain_rank)
                .map(|position| braid.permutation_at(position))
                .collect(),
            rank,
        });
    }
    if rule.braiding_style().is_symmetric()
        && rule.has_trivial_associator_gauge()
        && is_unique_direct_braid_source(rule, tree)
    {
        let mut coefficient = rule.scalar_one();
        for right_position in 0..rank {
            for left_position in 0..right_position {
                let left_axis = braid.permutation_at(left_position);
                let right_axis = braid.permutation_at(right_position);
                if left_axis > right_axis {
                    let left = tree.uncoupled()[left_axis];
                    let right = tree.uncoupled()[right_axis];
                    if left == rule.vacuum() || right == rule.vacuum() {
                        continue;
                    }
                    let coupled = only_fusion_channel(rule, left, right)?;
                    coefficient = coefficient * rule.r_symbol_scalar(left, right, coupled);
                }
            }
        }
        let uncoupled = (0..rank)
            .map(|position| tree.uncoupled()[braid.permutation_at(position)])
            .collect::<SmallVec<[SectorId; 8]>>();
        let is_dual = (0..rank)
            .map(|position| tree.is_dual()[braid.permutation_at(position)])
            .collect::<SmallVec<[bool; 8]>>();
        let destination = rebuild_unique_standard_fusion_tree(rule, &uncoupled, tree.coupled(), &is_dual)?;
        return Ok((destination, coefficient));
    }

    let mut current = tree.clone();
    let mut coefficient = rule.scalar_one();
    for step in braid.artin_steps() {
        let (next, step_coefficient) =
            unique_artin_braid_at_with_inverse(rule, &current, step.index, step.inverse)?;
        coefficient = coefficient * step_coefficient;
        current = next;
    }
    Ok((current, coefficient))
}

fn is_unique_direct_braid_source<R>(rule: &R, tree: &FusionTreeKey) -> bool
where
    R: MultiplicityFreeFusionRule,
{
    let rank = tree.uncoupled().len();
    if rank < 2
        || validate_fusion_tree_key_shape(tree).is_err()
        || tree
            .vertices()
            .iter()
            .any(|vertex| *vertex != MultiplicityIndex::ONE)
    {
        return false;
    }
    let coupled = tree.coupled();

    let mut running = tree.uncoupled()[0];
    for (offset, &right) in tree.uncoupled()[1..].iter().enumerate() {
        let Ok(next) = only_fusion_channel(rule, running, right) else {
            return false;
        };
        let is_last = offset + 2 == rank;
        if is_last {
            if next != coupled {
                return false;
            }
        } else if tree.innerlines().get(offset).copied() != Some(next) {
            return false;
        }
        running = next;
    }
    true
}

fn rebuild_unique_standard_fusion_tree<R>(
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
    if uncoupled.len() < 2 {
        return Err(CoreError::MalformedFusionTree {
            message: "direct unique braid rebuild requires at least two sectors",
        });
    }

    let mut innerlines = SmallVec::<[SectorId; 8]>::new();
    let mut running = uncoupled[0];
    for (offset, &right) in uncoupled[1..].iter().enumerate() {
        let next = only_fusion_channel(rule, running, right)?;
        let is_last = offset + 2 == uncoupled.len();
        if is_last {
            if next != coupled {
                return Err(CoreError::FusionChannelCount {
                    left: coupled,
                    right: coupled,
                    count: 0,
                });
            }
        } else {
            innerlines.push(next);
        }
        running = next;
    }

    Ok(FusionTreeKey::new(
        uncoupled.iter().copied(),
        coupled,
        is_dual.iter().copied(),
        innerlines,
        std::iter::repeat_n(MultiplicityIndex::ONE, uncoupled.len().saturating_sub(1)),
    ))
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
            tree.coupled()
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
    uncoupled: SectorVec,
    is_dual: DualVec,
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
    fn from_tree(tree: &FusionTreeKey) -> Self {
        Self {
            uncoupled: tree.uncoupled().iter().copied().collect(),
            is_dual: tree.is_dual().iter().copied().collect(),
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
        self.uncoupled.as_slice() == tree.uncoupled()
            && self.is_dual.as_slice() == tree.is_dual()
    }

    fn materialize(&self, local: MultiplicityFreeTreeLocal) -> FusionTreeKey {
        let vertex_count = self.uncoupled.len().saturating_sub(1);
        FusionTreeKey::new(
            self.uncoupled.iter().copied(),
            local.coupled,
            self.is_dual.iter().copied(),
            local.innerlines,
            std::iter::repeat_n(MultiplicityIndex::ONE, vertex_count),
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
    // Collect into the inline `SmallVec` types (stack-resident for ≤8 legs)
    // rather than heap `Vec`s: this is on the per-swap braid hot path.
    let mut uncoupled = frame.uncoupled.clone();
    uncoupled.swap(index, index + 1);
    let mut is_dual = frame.is_dual.clone();
    is_dual.swap(index, index + 1);
    let frame = MultiplicityFreeTreeFrame {
        uncoupled,
        is_dual,
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
                rule.scalar_one(),
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
                rule.scalar_conj(rule.r_symbol_scalar(right, left, coupled))
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

/// Elementary Artin braid of neighbouring uncoupled legs `index` and `index+1`
/// for an outer-multiplicity (`FusionStyleKind::Generic`) rule — the verbatim
/// mirror of TensorKit's `GenericFusion` branches of
/// `artin_braid(src::FusionTreeBlock, i; inv)`
/// (`fusiontrees/braiding_manipulations.jl:81-198`).
///
/// Where the multiplicity-free sibling
/// [`multiplicity_free_artin_braid_at_with_inverse`] returns a scalar per
/// output tree, here every vertex carries a one-based [`MultiplicityIndex`],
/// and one input tree can braid into several
/// output trees that differ *only* in their vertex labels. Each output's scalar
/// coefficient is the `R · F̄ · R̄` inner-index contraction TensorKit writes at
/// `braiding_manipulations.jl:181-182`.
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
            FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices),
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
            tree.coupled()
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
            let mut vertices: MultiplicityVec = tree.vertices().iter().copied().collect();
            *vertices
                .get_mut(0)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "first braid of a Generic tree requires a vertex",
                })? = MultiplicityIndex::new(nu0 + 1)
                .expect("enumerated Generic multiplicity labels are one-based");
            out.push((
                FusionTreeKey::new(
                    uncoupled.clone(),
                    tree.coupled(),
                    is_dual.clone(),
                    tree.innerlines().iter().copied(),
                    vertices,
                ),
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
                let mut vertices: MultiplicityVec = tree.vertices().iter().copied().collect();
                if vertices.len() <= index {
                    return Err(CoreError::MalformedFusionTree {
                        message: "non-first Generic braid requires adjacent vertices",
                    });
                }
                vertices[index - 1] = MultiplicityIndex::new(sigma0 + 1)
                    .expect("enumerated Generic multiplicity labels are one-based");
                vertices[index] = MultiplicityIndex::new(lambda0 + 1)
                    .expect("enumerated Generic multiplicity labels are one-based");
                out.push((
                    FusionTreeKey::new(
                        uncoupled.clone(),
                        tree.coupled(),
                        is_dual.clone(),
                        innerlines,
                        vertices,
                    ),
                    coeff,
                ));
            }
        }
    }
    Ok(out)
}

/// Read the 0-based outer-multiplicity matrix index of the vertex at position
/// `vertex_index`. [`MultiplicityIndex`] stores the one-based categorical
/// label, and TensorKit's `Rmat[μ, ν]` / `Fmat[κ, λ, μ, ρ]` are
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
        .get();
    Ok(label - 1)
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
/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
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
    let validated = validate_fusion_tree_for_rule(rule, tree)?;
    generic_braid_tree_validated(validated, permutation, levels, &swaps)
}

fn generic_braid_tree_validated<R>(
    tree: ValidatedFusionTree<'_, R>,
    permutation: &[usize],
    levels: &[usize],
    swaps: &[usize],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rule = tree.rule;
    generic_braid_tree_unchecked(rule, tree.key, permutation, levels, swaps)
}

fn generic_braid_tree_unchecked<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    levels: &[usize],
    swaps: &[usize],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rank = tree.uncoupled().len();
    if permutation.iter().copied().eq(0..rank) {
        return Ok(vec![(tree.clone(), R::Scalar::braid_one())]);
    }
    let mut current = vec![(tree.clone(), R::Scalar::braid_one())];
    let mut current_levels = levels.to_vec();
    for &swap in swaps {
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

struct PreparedMultiplicityFreeBendRight {
    codomain_rank: usize,
    domain_rank: usize,
    codomain_first: SectorId,
    domain_nonempty: bool,
    bent_sector: SectorId,
    bent_is_dual: Option<bool>,
    output_codomain_uncoupled: SectorVec,
    output_codomain_is_dual: DualVec,
    output_domain_uncoupled_prefix: SectorVec,
    output_domain_is_dual_prefix: DualVec,
}

struct ValidatedMultiplicityFreeBendRightLocal {
    local: MultiplicityFreeTreePairLocal,
    coupled: SectorId,
    left_coupled: SectorId,
    bent_is_dual: bool,
}

impl PreparedMultiplicityFreeBendRight {
    fn validate_output_frame(&self) -> Result<(), CoreError> {
        self.bent_is_dual
            .ok_or(CoreError::MalformedFusionTree {
                message: "codomain tree is missing a duality flag",
            })
            .map(|_| ())
    }

    fn output_frame<R>(&self, rule: &R) -> Result<MultiplicityFreeTreePairFrame, CoreError>
    where
        R: FusionRule,
    {
        let bent_is_dual = self.bent_is_dual.ok_or(CoreError::MalformedFusionTree {
            message: "codomain tree is missing a duality flag",
        })?;
        let mut domain_uncoupled = self.output_domain_uncoupled_prefix.clone();
        domain_uncoupled.push(rule.dual(self.bent_sector));
        let mut domain_is_dual = self.output_domain_is_dual_prefix.clone();
        domain_is_dual.push(!bent_is_dual);
        Ok(MultiplicityFreeTreePairFrame {
            codomain: MultiplicityFreeTreeFrame {
                uncoupled: self.output_codomain_uncoupled.clone(),
                is_dual: self.output_codomain_is_dual.clone(),
            },
            domain: MultiplicityFreeTreeFrame {
                uncoupled: domain_uncoupled,
                is_dual: domain_is_dual,
            },
        })
    }

    fn validate_local<R, C, D>(
        &self,
        rule: &R,
        codomain: &C,
        domain: &D,
    ) -> Result<ValidatedMultiplicityFreeBendRightLocal, CoreError>
    where
        R: FusionRule,
        C: MultiplicityFreeTreeLocalData + ?Sized,
        D: MultiplicityFreeTreeLocalData + ?Sized,
    {
        let coupled = codomain.coupled();
        if self.domain_nonempty {
            let domain_coupled = domain.coupled();
            if domain_coupled != coupled {
                return Err(CoreError::MalformedFusionTree {
                    message: "fusion tree pair requires matching coupled sectors",
                });
            }
        }

        let left_coupled = match self.codomain_rank {
            1 => rule.vacuum(),
            2 => self.codomain_first,
            _ => codomain.innerlines().last().copied().ok_or(
                CoreError::MalformedFusionTree {
                    message: "bendright requires the last codomain innerline",
                },
            )?,
        };
        let bent_is_dual = self.bent_is_dual.ok_or(CoreError::MalformedFusionTree {
            message: "codomain tree is missing a duality flag",
        })?;

        let cod_inner = codomain.innerlines();
        let new_codomain_innerlines: &[SectorId] = if self.codomain_rank > 2 {
            &cod_inner[..cod_inner.len() - 1]
        } else {
            &[]
        };
        Ok(ValidatedMultiplicityFreeBendRightLocal {
            local: MultiplicityFreeTreePairLocal {
                codomain: MultiplicityFreeTreeLocal {
                    coupled: left_coupled,
                    innerlines: new_codomain_innerlines.iter().copied().collect(),
                },
                domain: MultiplicityFreeTreeLocal {
                    coupled: left_coupled,
                    innerlines: domain
                        .innerlines()
                        .iter()
                        .copied()
                        .chain((self.domain_rank > 1).then_some(coupled))
                        .collect(),
                },
            },
            coupled,
            left_coupled,
            bent_is_dual,
        })
    }

    fn coefficient<R>(
        &self,
        rule: &R,
        local: &ValidatedMultiplicityFreeBendRightLocal,
    ) -> R::Scalar
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        let mut coefficient = rule.sqrt_dim_scalar(local.coupled)
            * rule.inv_sqrt_dim_scalar(local.left_coupled)
            * rule.b_symbol_scalar(local.left_coupled, self.bent_sector, local.coupled);
        if local.bent_is_dual {
            coefficient = coefficient
                * rule.scalar_conj(
                    rule.frobenius_schur_phase_scalar(rule.dual(self.bent_sector)),
                );
        }
        coefficient
    }
}

fn prepare_multiplicity_free_bendright<R>(
    _rule: &R,
    frame: &MultiplicityFreeTreePairFrame,
) -> Result<PreparedMultiplicityFreeBendRight, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let codomain = &frame.codomain;
    let domain = &frame.domain;
    let codomain_rank = codomain.uncoupled.len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "bendright requires at least one codomain leg",
        });
    }

    let bent_sector = codomain.uncoupled[codomain_rank - 1];
    let bent_is_dual = codomain.is_dual.get(codomain_rank - 1).copied();
    let output_codomain_uncoupled = codomain.uncoupled[..codomain_rank - 1]
        .iter()
        .copied()
        .collect();
    let output_codomain_is_dual = codomain
        .is_dual
        .iter()
        .copied()
        .take(codomain_rank - 1)
        .collect();
    let domain_rank = domain.uncoupled.len();
    Ok(PreparedMultiplicityFreeBendRight {
        codomain_rank,
        domain_rank,
        codomain_first: codomain.uncoupled[0],
        domain_nonempty: domain_rank != 0,
        bent_sector,
        bent_is_dual,
        output_codomain_uncoupled,
        output_codomain_is_dual,
        output_domain_uncoupled_prefix: domain.uncoupled.clone(),
        output_domain_is_dual_prefix: domain.is_dual.clone(),
    })
}

struct PreparedMultiplicityFreeBendLeft {
    right: PreparedMultiplicityFreeBendRight,
}

impl PreparedMultiplicityFreeBendLeft {
    fn validate_output_frame(&self) -> Result<(), CoreError> {
        self.right.validate_output_frame()
    }

    fn output_frame<R>(&self, rule: &R) -> Result<MultiplicityFreeTreePairFrame, CoreError>
    where
        R: FusionRule,
    {
        let frame = self.right.output_frame(rule)?;
        Ok(MultiplicityFreeTreePairFrame {
            codomain: frame.domain,
            domain: frame.codomain,
        })
    }

    fn validate_local<R, C, D>(
        &self,
        rule: &R,
        codomain: &C,
        domain: &D,
    ) -> Result<ValidatedMultiplicityFreeBendRightLocal, CoreError>
    where
        R: FusionRule,
        C: MultiplicityFreeTreeLocalData + ?Sized,
        D: MultiplicityFreeTreeLocalData + ?Sized,
    {
        self.right.validate_local(rule, domain, codomain)
    }

    fn finish_local<R>(
        &self,
        rule: &R,
        validated: ValidatedMultiplicityFreeBendRightLocal,
    ) -> (MultiplicityFreeTreePairLocal, R::Scalar)
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        let coefficient = rule.scalar_conj(self.right.coefficient(rule, &validated));
        (Self::finish_local_structure(validated), coefficient)
    }

    fn finish_local_structure(
        validated: ValidatedMultiplicityFreeBendRightLocal,
    ) -> MultiplicityFreeTreePairLocal {
        MultiplicityFreeTreePairLocal {
            codomain: validated.local.domain,
            domain: validated.local.codomain,
        }
    }
}

fn prepare_multiplicity_free_bendleft<R>(
    rule: &R,
    frame: &MultiplicityFreeTreePairFrame,
) -> Result<PreparedMultiplicityFreeBendLeft, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let swapped = MultiplicityFreeTreePairFrame {
        codomain: frame.domain.clone(),
        domain: frame.codomain.clone(),
    };
    Ok(PreparedMultiplicityFreeBendLeft {
        right: prepare_multiplicity_free_bendright(rule, &swapped)?,
    })
}

fn multiplicity_free_bendright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<SmallVec<[(FusionTreePairKey, R::Scalar); 1]>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    // Why not duplicate bend surgery in the future block runner: duality and
    // Frobenius-Schur phases must stay identical to the per-source operation.
    let (frame, local) = project_multiplicity_free_tree_pair(rule, tree_pair)?;
    let prepared = prepare_multiplicity_free_bendright(rule, &frame)?;
    let validated = prepared.validate_local(rule, &local.codomain, &local.domain)?;
    let frame = prepared.output_frame(rule)?;
    let coefficient = prepared.coefficient(rule, &validated);
    let mut terms = SmallVec::new();
    terms.push((frame.materialize(validated.local), coefficient));
    Ok(terms)
}

fn multiplicity_free_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<SmallVec<[(FusionTreePairKey, R::Scalar); 1]>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let (frame, local) = project_multiplicity_free_tree_pair(rule, tree_pair)?;
    let prepared = prepare_multiplicity_free_bendleft(rule, &frame)?;
    let validated = prepared.validate_local(rule, &local.codomain, &local.domain)?;
    let frame = prepared.output_frame(rule)?;
    let (local, coefficient) = prepared.finish_local(rule, validated);
    let mut terms = SmallVec::new();
    terms.push((frame.materialize(local), coefficient));
    Ok(terms)
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
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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

    let coupled = codomain.coupled();
    if !domain.uncoupled().is_empty() {
        let domain_coupled = domain.coupled();
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

    // New codomain tree: drop the last leg (TK `_bendright_treepair` :41-45).
    let cod_inner = codomain.innerlines();
    let new_codomain_innerlines: &[SectorId] = if codomain_rank > 2 {
        &cod_inner[..cod_inner.len() - 1]
    } else {
        &[]
    };
    let cod_vertices = codomain.vertices();
    let new_codomain_vertices: &[MultiplicityIndex] = if codomain_rank > 1 {
        &cod_vertices[..cod_vertices.len() - 1]
    } else {
        &[]
    };
    let new_codomain = FusionTreeKey::new(
        codomain.uncoupled()[..codomain_rank - 1].iter().copied(),
        left_coupled,
        codomain.is_dual()[..codomain_rank - 1].iter().copied(),
        new_codomain_innerlines.iter().copied(),
        new_codomain_vertices.iter().copied(),
    );

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
    let mut out: Vec<(FusionTreePairKey, R::Scalar)> = Vec::new();
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
            left_coupled,
            domain_is_dual.iter().copied(),
            domain_innerlines.iter().copied(),
            domain
                .vertices()
                .iter()
                .copied()
                .chain(
                    (domain_rank > 0).then(|| {
                        MultiplicityIndex::new(nu0 + 1)
                            .expect("enumerated Generic multiplicity labels are one-based")
                    }),
                ),
        );
        let key = FusionTreePairKey::pair(new_codomain.clone(), new_domain);
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
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let swapped = FusionTreePairKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(generic_bendright_tree_pair(rule, &swapped)?
        .into_iter()
        .map(|(bent, coefficient)| {
            (
                FusionTreePairKey::pair(bent.domain_tree().clone(), bent.codomain_tree().clone()),
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
    terms: Vec<(FusionTreePairKey, R::Scalar)>,
    mut transform: F,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
    F: FnMut(&R, &FusionTreePairKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreePairKey, R::Scalar)>,
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
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn generic_repartition_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_repartition_tree_pair_validated(validated, target_codomain_rank)
}

fn generic_repartition_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rule = tree_pair.rule;
    generic_repartition_tree_pair_unchecked(rule, tree_pair.key, target_codomain_rank)
}

fn generic_repartition_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
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
/// [`collect_fusion_trees_for_coupled`] (which uses
/// [`MultiplicityIndex::ONE`] for every vertex and is bounded on
/// `MultiplicityFreeFusionRule`): here each
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
                    coupled,
                    is_dual.iter().copied(),
                    inner_rev.iter().rev().copied(),
                    vtx_rev.iter().rev().map(|&label| {
                        MultiplicityIndex::new(label)
                            .expect("enumerated Generic multiplicity labels are one-based")
                    }),
                )
            );
        },
    );
    out
}

/// Recursive walker for [`collect_generic_fusion_trees_for_coupled`]. Mirrors
/// [`visit_fusion_trees`] (peels the LAST leg, recursing inward), but at every
/// vertex it iterates `1..=Nsymbol(...)` and records the 1-based label. Vertex
/// labels are stored in [`MultiplicityIndex`] using the same one-based
/// convention that [`mu_index`] decodes.
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

#[inline]
fn multi_associator_new_cross_channel_is_admissible<R>(
    rule: &R,
    leading: SectorId,
    short_right: SectorId,
    long_right: SectorId,
) -> bool
where
    R: FusionRule,
{
    // Why not recheck all four F vertices: validated/generated trees prove the
    // two stored vertices. At k=2 the left cross is long's stored first vertex;
    // at k>2 it is the previous stage's accepted new cross. Only this right
    // cross is newly introduced at the current stage.
    rule.nsymbol(leading, short_right, long_right) != 0
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
        let c = long.coupled();
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
        if !multi_associator_new_cross_channel_is_admissible(
            rule,
            first,
            short_right,
            middle_right,
        ) {
            return Ok(None);
        }
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
                rule.vacuum(),
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<MultiplicityIndex>::new(),
            ),
            vec![R::Scalar::braid_one()],
        )]);
    }
    if rank == 2 {
        // TK `:221-232`: single tail `(b,) → b`, coeff = unit vector `e_μ` over
        // the (unchanged) topmost vertex `a ⊗ b → c`, μ = tree.vertices[0].
        let a = tree.uncoupled()[0];
        let b = tree.uncoupled()[1];
        let c = tree.coupled();
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
            b,
            vec![tree.is_dual()[1]],
            Vec::<SectorId>::new(),
            Vec::<MultiplicityIndex>::new(),
        );
        return Ok(vec![(tail, coeff)]);
    }

    let first = tree.uncoupled()[0];
    let coupled = tree.coupled();
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
    let tree_coupled = tree.coupled();
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
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn generic_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_foldright_tree_pair_validated(validated)
}

fn generic_foldright_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rule = tree_pair.rule;
    generic_foldright_tree_pair_unchecked(rule, tree_pair.key)
}

fn generic_foldright_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let codomain = tree_pair.codomain_tree();
    if codomain.uncoupled().is_empty() {
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
    let c = codomain.coupled();

    let mut terms = FusionTermAccumulator::new();
    for (codomain_prime, coeff1) in generic_multi_fmove_tree(rule, codomain)? {
        let b = codomain_prime.coupled();
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
                FusionTreePairKey::pair(codomain_prime.clone(), domain_prime),
                coefficient,
            );
        }
    }
    Ok(terms.into_vec())
}

/// Generic-fusion `foldleft` = swap + conjugate of `foldright`, verbatim mirror
/// of TensorKit `foldleft((f₁,f₂))` (`duality_manipulations.jl:315-319`).
/// Structural twin of [`multiplicity_free_foldleft_tree_pair`].
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn generic_foldleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if tree_pair.domain_tree().uncoupled().is_empty() {
        return Err(CoreError::MalformedFusionTree {
            message: "foldleft requires at least one domain leg",
        });
    }
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_foldleft_tree_pair_validated(validated)
}

fn generic_foldleft_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rule = tree_pair.rule;
    generic_foldleft_tree_pair_unchecked(rule, tree_pair.key)
}

fn generic_foldleft_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let swapped = FusionTreePairKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(generic_foldright_tree_pair_unchecked(rule, &swapped)?
        .into_iter()
        .map(|(folded, coefficient)| {
            (
                FusionTreePairKey::pair(
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
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn generic_cycle_clockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_cycle_clockwise_tree_pair_validated(validated)
}

fn generic_cycle_clockwise_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rule = tree_pair.rule;
    generic_cycle_clockwise_tree_pair_unchecked(rule, tree_pair.key)
}

fn generic_cycle_clockwise_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if tree_pair.codomain_tree().uncoupled().is_empty() {
        let first = generic_bendleft_tree_pair(rule, tree_pair)?;
        compose_generic_tree_pair_terms(rule, first, |rule, key| {
            generic_foldright_tree_pair_unchecked(rule, key)
        })
    } else {
        let first = generic_foldright_tree_pair_unchecked(rule, tree_pair)?;
        compose_generic_tree_pair_terms(rule, first, |rule, key| {
            generic_bendleft_tree_pair(rule, key)
        })
    }
}

/// Generic-fusion `cycleanticlockwise` = foldleft ∘ bendright (or the reverse
/// order when the domain is empty). Verbatim mirror of TensorKit
/// `cycleanticlockwise` (`duality_manipulations.jl:431-440`) and structural
/// twin of [`multiplicity_free_cycle_anticlockwise_tree_pair`].
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn generic_cycle_anticlockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_cycle_anticlockwise_tree_pair_validated(validated)
}

fn generic_cycle_anticlockwise_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rule = tree_pair.rule;
    generic_cycle_anticlockwise_tree_pair_unchecked(rule, tree_pair.key)
}

fn generic_cycle_anticlockwise_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if tree_pair.domain_tree().uncoupled().is_empty() {
        let first = generic_bendright_tree_pair(rule, tree_pair)?;
        compose_generic_tree_pair_terms(rule, first, |rule, key| {
            generic_foldleft_tree_pair_unchecked(rule, key)
        })
    } else {
        let first = generic_foldleft_tree_pair_unchecked(rule, tree_pair)?;
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
    terms: Vec<(FusionTreePairKey, R::Scalar)>,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn generic_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_braid_tree_pair_proven(
        validated,
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )
}

pub(crate) fn generic_braid_tree_pair_proven<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if tree_pair.rule.fusion_style() != FusionStyleKind::Generic {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: tree_pair.rule.fusion_style(),
        });
    }
    let codomain_rank = tree_pair.key.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.key.domain_tree().uncoupled().len();
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
    let swaps = permutation_to_adjacent_swaps(&permutation, codomain_rank + domain_rank)?;
    let identity = tree_pair_axis_map_is_identity(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    );

    let mut levels = Vec::with_capacity(codomain_rank + domain_rank);
    levels.extend_from_slice(codomain_levels);
    levels.extend(domain_levels.iter().rev().copied());
    generic_braid_tree_pair_validated(
        tree_pair,
        codomain_permutation.len(),
        &permutation,
        &levels,
        &swaps,
        identity,
    )
}

fn generic_braid_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    target_codomain_rank: usize,
    permutation: &[usize],
    levels: &[usize],
    swaps: &[usize],
    identity: bool,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rule = tree_pair.rule;
    if identity {
        return Ok(vec![(tree_pair.key.clone(), R::Scalar::braid_one())]);
    }
    let all_rank = permutation.len();
    let mut current =
        generic_repartition_tree_pair_unchecked(rule, tree_pair.key, all_rank)?;
    current = compose_generic_tree_pair_terms(rule, current, |rule, key| {
        generic_braid_tree_unchecked(
            rule,
            key.codomain_tree(),
            permutation,
            levels,
            swaps,
        )
        .map(|terms| {
                terms
                    .into_iter()
                    .map(|(codomain_tree, coefficient)| {
                        (
                            FusionTreePairKey::pair(codomain_tree, key.domain_tree().clone()),
                            coefficient,
                        )
                    })
                    .collect::<Vec<_>>()
            })
    })?;
    generic_repartition_terms(rule, current, target_codomain_rank)
}

/// Generic-fusion `permute` = [`generic_braid_tree_pair`] with the identity
/// level order (symmetric braiding only). Structural twin of
/// [`multiplicity_free_permute_tree_pair`] (:886).
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn generic_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_permute_tree_pair_proven(validated, codomain_permutation, domain_permutation)
}

pub(crate) fn generic_permute_tree_pair_proven<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if !tree_pair.rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: tree_pair.rule.braiding_style(),
        });
    }
    if tree_pair.rule.fusion_style() != FusionStyleKind::Generic {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: tree_pair.rule.fusion_style(),
        });
    }
    let codomain_rank = tree_pair.key.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.key.domain_tree().uncoupled().len();
    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    let swaps = permutation_to_adjacent_swaps(&permutation, codomain_rank + domain_rank)?;
    let identity = tree_pair_axis_map_is_identity(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    );
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    let mut levels = Vec::with_capacity(codomain_rank + domain_rank);
    levels.extend_from_slice(&codomain_levels);
    levels.extend(domain_levels.iter().rev().copied());
    generic_braid_tree_pair_validated(
        tree_pair,
        codomain_permutation.len(),
        &permutation,
        &levels,
        &swaps,
        identity,
    )
}

/// Generic-fusion `transpose` (planar cyclic permutation): bend into the target
/// partition, then cycle the coupled tree into place via fold/bend. Structural
/// twin of [`multiplicity_free_transpose_tree_pair`] (:916); braid-free, so it
/// runs on planar (non-symmetric) Generic rules too.
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn generic_transpose_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_transpose_tree_pair_proven(validated, codomain_permutation, domain_permutation)
}

pub(crate) fn generic_transpose_tree_pair_proven<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    if tree_pair.rule.fusion_style() != FusionStyleKind::Generic {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: tree_pair.rule.fusion_style(),
        });
    }
    let codomain_rank = tree_pair.key.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.key.domain_tree().uncoupled().len();
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
    let position = permutation.iter().position(|&axis| axis == 0);
    generic_transpose_tree_pair_validated(
        tree_pair,
        codomain_permutation.len(),
        codomain_rank + domain_rank,
        position,
    )
}

fn generic_transpose_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    target_codomain_rank: usize,
    total_rank: usize,
    position: Option<usize>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rule = tree_pair.rule;
    let mut position = match position {
        Some(position) => position,
        None => return Ok(vec![(tree_pair.key.clone(), R::Scalar::braid_one())]),
    };
    let mut current = generic_repartition_tree_pair_unchecked(
        rule,
        tree_pair.key,
        target_codomain_rank,
    )?;
    if total_rank == 0 || position == 0 {
        return Ok(current);
    }

    let half_rank = total_rank >> 1;
    while position > 0 && position < half_rank {
        current = compose_generic_tree_pair_terms(rule, current, |rule, key| {
            generic_cycle_anticlockwise_tree_pair_unchecked(rule, key)
        })?;
        position -= 1;
    }
    while position >= half_rank && position > 0 {
        current = compose_generic_tree_pair_terms(rule, current, |rule, key| {
            generic_cycle_clockwise_tree_pair_unchecked(rule, key)
        })?;
        position = (position + 1) % total_rank;
    }

    Ok(current)
}

struct PreparedMultiplicityFreeFoldRight<S> {
    first: SectorId,
    first_is_dual: bool,
    frobenius_schur_phase: S,
    output_frame: MultiplicityFreeTreePairFrame,
}

fn prepare_multiplicity_free_foldright<R>(
    rule: &R,
    frame: &MultiplicityFreeTreePairFrame,
) -> Result<PreparedMultiplicityFreeFoldRight<R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let first = frame.codomain.uncoupled.first().copied().ok_or(
        CoreError::MalformedFusionTree {
            message: "foldright requires at least one codomain leg",
        },
    )?;
    let first_is_dual =
        frame
            .codomain
            .is_dual
            .first()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "codomain tree is missing the first duality flag",
            })?;
    let output_frame = MultiplicityFreeTreePairFrame {
        codomain: MultiplicityFreeTreeFrame {
            uncoupled: frame.codomain.uncoupled[1..].iter().copied().collect(),
            is_dual: frame.codomain.is_dual[1..].iter().copied().collect(),
        },
        domain: MultiplicityFreeTreeFrame {
            uncoupled: std::iter::once(rule.dual(first))
                .chain(frame.domain.uncoupled.iter().copied())
                .collect(),
            is_dual: std::iter::once(!first_is_dual)
                .chain(frame.domain.is_dual.iter().copied())
                .collect(),
        },
    };
    Ok(PreparedMultiplicityFreeFoldRight {
        first,
        first_is_dual,
        frobenius_schur_phase: rule.frobenius_schur_phase_scalar(first),
        output_frame,
    })
}

fn multiplicity_free_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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
    let c = codomain.coupled();

    let mut terms = FusionTermAccumulator::new();
    for (codomain_prime, coeff1) in multiplicity_free_multi_fmove_tree(rule, codomain)? {
        let b = codomain_prime.coupled();
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
                FusionTreePairKey::pair(codomain_prime.clone(), domain_prime),
                coefficient,
            );
        }
    }
    Ok(terms.into_vec())
}

fn multiplicity_free_foldleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let swapped = FusionTreePairKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(multiplicity_free_foldright_tree_pair(rule, &swapped)?
        .into_iter()
        .map(|(folded, coefficient)| {
            (
                FusionTreePairKey::pair(
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
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
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

fn collect_multiplicity_free_tree_locals_for_coupled<R>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
) -> Vec<MultiplicityFreeTreeLocal>
where
    R: MultiplicityFreeFusionRule,
{
    collect_multiplicity_free_tree_locals_for_coupled_where(
        rule,
        effective,
        coupled,
        |_, _| true,
    )
}

fn collect_multiplicity_free_tree_locals_for_coupled_where<R, P>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
    prefix_allowed: P,
) -> Vec<MultiplicityFreeTreeLocal>
where
    R: MultiplicityFreeFusionRule,
    P: Fn(usize, SectorId) -> bool,
{
    let mut locals = Vec::new();
    let mut inner_rev = Vec::new();
    visit_fusion_trees_where(
        rule,
        effective,
        coupled,
        &mut inner_rev,
        &prefix_allowed,
        &mut |inner_rev| {
            locals.push(MultiplicityFreeTreeLocal {
                coupled,
                innerlines: inner_rev.iter().rev().copied().collect(),
            });
        },
    );
    locals
}

fn collect_multiplicity_free_tree_pair_locals_for_frame<R>(
    rule: &R,
    frame: &MultiplicityFreeTreePairFrame,
) -> Vec<MultiplicityFreeTreePairLocal>
where
    R: MultiplicityFreeFusionRule,
{
    let codomain_coupled = reachable_coupled_sectors(rule, &frame.codomain.uncoupled);
    let domain_coupled = reachable_coupled_sectors(rule, &frame.domain.uncoupled);
    let mut codomain_index = 0usize;
    let mut domain_index = 0usize;
    let mut locals = Vec::new();
    while codomain_index < codomain_coupled.len() && domain_index < domain_coupled.len() {
        let codomain_sector = codomain_coupled[codomain_index];
        let domain_sector = domain_coupled[domain_index];
        match codomain_sector.cmp(&domain_sector) {
            std::cmp::Ordering::Less => codomain_index += 1,
            std::cmp::Ordering::Greater => domain_index += 1,
            std::cmp::Ordering::Equal => {
                let codomain = collect_multiplicity_free_tree_locals_for_coupled(
                    rule,
                    &frame.codomain.uncoupled,
                    codomain_sector,
                );
                let domain = collect_multiplicity_free_tree_locals_for_coupled(
                    rule,
                    &frame.domain.uncoupled,
                    domain_sector,
                );
                for domain_local in &domain {
                    for codomain_local in &codomain {
                        locals.push(MultiplicityFreeTreePairLocal {
                            codomain: codomain_local.clone(),
                            domain: domain_local.clone(),
                        });
                    }
                }
                codomain_index += 1;
                domain_index += 1;
            }
        }
    }
    locals
}

fn multiplicity_free_forward_prefix_targets<T>(
    long_uncoupled: &[SectorId],
    long: &T,
) -> Result<SectorVec, CoreError>
where
    T: MultiplicityFreeTreeLocalData + ?Sized,
{
    let mut targets = SectorVec::with_capacity(long_uncoupled.len().saturating_sub(2));
    for leg_index in 2..long_uncoupled.len() {
        let (_, right) =
            fusion_tree_vertex_neighbors_from_parts(long_uncoupled, long, leg_index)?;
        targets.push(right);
    }
    Ok(targets)
}

fn multiplicity_free_inverse_prefix_targets<T>(
    short_uncoupled: &[SectorId],
    short: &T,
) -> Result<SectorVec, CoreError>
where
    T: MultiplicityFreeTreeLocalData + ?Sized,
{
    let mut targets = SectorVec::with_capacity(short_uncoupled.len());
    for leg_index in 1..short_uncoupled.len() {
        let (left, right) =
            fusion_tree_vertex_neighbors_from_parts(short_uncoupled, short, leg_index)?;
        if leg_index == 1 {
            targets.push(left);
        }
        targets.push(right);
    }
    Ok(targets)
}

#[inline]
fn multiplicity_free_forward_prefix_allowed<R>(
    rule: &R,
    leading: SectorId,
    targets: &[SectorId],
    tail_prefix_len: usize,
    tail_prefix_coupled: SectorId,
) -> bool
where
    R: MultiplicityFreeFusionRule,
{
    tail_prefix_len < 2
        || rule.nsymbol(
            leading,
            tail_prefix_coupled,
            targets[tail_prefix_len - 2],
        ) != 0
}

#[inline]
fn multiplicity_free_inverse_prefix_allowed<R>(
    rule: &R,
    leading: SectorId,
    targets: &[SectorId],
    candidate_prefix_len: usize,
    candidate_rank: usize,
    candidate_prefix_coupled: SectorId,
) -> bool
where
    R: MultiplicityFreeFusionRule,
{
    candidate_prefix_len <= 2
        || candidate_prefix_len >= candidate_rank
        || rule.nsymbol(
            leading,
            targets[candidate_prefix_len - 2],
            candidate_prefix_coupled,
        ) != 0
}

fn multiplicity_free_multi_fmove_local<R>(
    rule: &R,
    frame: &MultiplicityFreeTreeFrame,
    local: &MultiplicityFreeTreeLocal,
) -> Result<Vec<(MultiplicityFreeTreeLocal, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let rank = frame.uncoupled.len();
    if rank != frame.is_dual.len() {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree sectors and duality flags must have matching length",
        });
    }
    if rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "multi_Fmove requires at least one uncoupled sector",
        });
    }
    if rank == 1 {
        return Ok(vec![(
            MultiplicityFreeTreeLocal {
                coupled: rule.vacuum(),
                innerlines: SectorVec::new(),
            },
            rule.scalar_one(),
        )]);
    }
    if rank == 2 {
        return Ok(vec![(
            MultiplicityFreeTreeLocal {
                coupled: frame.uncoupled[1],
                innerlines: SectorVec::new(),
            },
            rule.scalar_one(),
        )]);
    }

    let first = frame.uncoupled[0];
    let coupled = local.coupled;
    let tail_uncoupled = &frame.uncoupled[1..];
    let tail_is_dual = &frame.is_dual[1..];
    let prefix_targets =
        multiplicity_free_forward_prefix_targets(&frame.uncoupled, local)?;
    let mut candidates = Vec::new();
    for tail_coupled in rule.fusion_channels(rule.dual(first), coupled) {
        candidates.extend(collect_multiplicity_free_tree_locals_for_coupled_where(
            rule,
            tail_uncoupled,
            tail_coupled,
            |prefix_len, prefix_coupled| {
                multiplicity_free_forward_prefix_allowed(
                    rule,
                    first,
                    &prefix_targets,
                    prefix_len,
                    prefix_coupled,
                )
            },
        ));
    }
    let coefficients = multiplicity_free_multi_associator_grouped_fixed_long(
        rule,
        &frame.uncoupled,
        &frame.is_dual,
        local,
        tail_uncoupled,
        tail_is_dual,
        &candidates,
    )?;
    Ok(candidates
        .into_iter()
        .zip(coefficients)
        .filter_map(|(candidate, coefficient)| {
            coefficient.map(|coefficient| (candidate, coefficient))
        })
        .collect())
}

fn multiplicity_free_multi_fmove_inv_local<R>(
    rule: &R,
    coupled: SectorId,
    source_frame: &MultiplicityFreeTreeFrame,
    source: &MultiplicityFreeTreeLocal,
    output_frame: &MultiplicityFreeTreeFrame,
) -> Result<Vec<(MultiplicityFreeTreeLocal, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    if source_frame.uncoupled.len() != source_frame.is_dual.len()
        || output_frame.uncoupled.len() != output_frame.is_dual.len()
        || output_frame.uncoupled.len() != source_frame.uncoupled.len() + 1
        || output_frame.uncoupled[1..] != *source_frame.uncoupled.as_slice()
        || output_frame.is_dual[1..] != *source_frame.is_dual.as_slice()
    {
        return Err(CoreError::MalformedFusionTree {
            message: "multi_Fmove inverse requires one leading external sector",
        });
    }
    let leading_sector = output_frame.uncoupled[0];
    if rule.nsymbol(leading_sector, source.coupled, coupled) == 0 {
        return Err(CoreError::SectorMismatch {
            expected: coupled,
            actual: source.coupled,
        });
    }

    let prefix_targets =
        multiplicity_free_inverse_prefix_targets(&source_frame.uncoupled, source)?;
    let candidates = collect_multiplicity_free_tree_locals_for_coupled_where(
        rule,
        &output_frame.uncoupled,
        coupled,
        |prefix_len, prefix_coupled| {
            multiplicity_free_inverse_prefix_allowed(
                rule,
                leading_sector,
                &prefix_targets,
                prefix_len,
                output_frame.uncoupled.len(),
                prefix_coupled,
            )
        },
    );
    let coefficients = multiplicity_free_multi_associator_grouped_fixed_short(
        rule,
        &output_frame.uncoupled,
        &output_frame.is_dual,
        &candidates,
        &source_frame.uncoupled,
        &source_frame.is_dual,
        source,
    )?;
    Ok(candidates
        .into_iter()
        .zip(coefficients)
        .filter_map(|(candidate, coefficient)| {
            coefficient.map(|coefficient| (candidate, rule.scalar_conj(coefficient)))
        })
        .collect())
}

fn multiplicity_free_multi_fmove_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let (frame, local) = project_multiplicity_free_tree(rule, tree)?;
    let terms = multiplicity_free_multi_fmove_local(rule, &frame, &local)?;
    let output_frame = MultiplicityFreeTreeFrame {
        uncoupled: frame.uncoupled[1..].iter().copied().collect(),
        is_dual: frame.is_dual[1..].iter().copied().collect(),
    };
    Ok(terms
        .into_iter()
        .map(|(local, coefficient)| (output_frame.materialize(local), coefficient))
        .collect())
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
    let (source_frame, source) = project_multiplicity_free_tree(rule, tree)?;
    let mut uncoupled = Vec::with_capacity(tree.uncoupled().len() + 1);
    uncoupled.push(leading_sector);
    uncoupled.extend_from_slice(tree.uncoupled());
    let mut is_dual = Vec::with_capacity(tree.is_dual().len() + 1);
    is_dual.push(leading_is_dual);
    is_dual.extend_from_slice(tree.is_dual());
    let output_frame = MultiplicityFreeTreeFrame {
        uncoupled: uncoupled.into_iter().collect(),
        is_dual: is_dual.into_iter().collect(),
    };
    Ok(multiplicity_free_multi_fmove_inv_local(
        rule,
        coupled,
        &source_frame,
        &source,
        &output_frame,
    )?
    .into_iter()
    .map(|(local, coefficient)| (output_frame.materialize(local), coefficient))
    .collect())
}

#[cfg(test)]
fn fusion_tree_vertex_neighbors_legacy_oracle(
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
        tree.coupled()
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

#[cfg(test)]
fn multiplicity_free_multi_associator_scalar_legacy_oracle<R>(
    rule: &R,
    long: &FusionTreeKey,
    short: &FusionTreeKey,
) -> Result<Option<R::Scalar>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rank = long.uncoupled().len();
    if short.uncoupled().len() + 1 != rank
        || long.uncoupled()[1..] != *short.uncoupled()
        || long.is_dual()[1..] != *short.is_dual()
    {
        return Ok(None);
    }

    let mut coefficient = rule.scalar_one();
    let first = long.uncoupled()[0];
    for tensor_kit_k in 2..rank {
        let right = long.uncoupled()[tensor_kit_k];
        let (left_coupled, coupled) =
            fusion_tree_vertex_neighbors_legacy_oracle(long, tensor_kit_k)?;
        let (middle, right_coupled) =
            fusion_tree_vertex_neighbors_legacy_oracle(short, tensor_kit_k - 1)?;
        // Why not reuse the production admissibility helper: this oracle must
        // retain the pre-grouping decision path to expose a shared-helper bug.
        if rule.nsymbol(first, right_coupled, coupled) == 0 {
            return Ok(None);
        }
        coefficient = coefficient
            * rule.f_symbol_scalar(
                first,
                middle,
                right,
                coupled,
                left_coupled,
                right_coupled,
            );
    }
    Ok(Some(coefficient))
}

#[cfg(test)]
fn multiplicity_free_multi_fmove_tree_legacy_oracle<R>(
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
                rule.vacuum(),
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<MultiplicityIndex>::new(),
            ),
            rule.scalar_one(),
        )]);
    }
    if rank == 2 {
        return Ok(vec![(
            FusionTreeKey::new(
                vec![tree.uncoupled()[1]],
                tree.uncoupled()[1],
                vec![tree.is_dual()[1]],
                Vec::<SectorId>::new(),
                Vec::<MultiplicityIndex>::new(),
            ),
            rule.scalar_one(),
        )]);
    }

    let first = tree.uncoupled()[0];
    let coupled = tree.coupled();
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
                multiplicity_free_multi_associator_scalar_legacy_oracle(rule, tree, &tail_tree)?
            {
                terms.push((tail_tree, coefficient));
            }
        }
    }
    Ok(terms)
}

#[cfg(test)]
fn multiplicity_free_multi_fmove_inv_tree_legacy_oracle<R>(
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
    let tree_coupled = tree.coupled();
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

    let mut terms = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if let Some(coefficient) =
            multiplicity_free_multi_associator_scalar_legacy_oracle(rule, &candidate, tree)?
        {
            terms.push((candidate, rule.scalar_conj(coefficient)));
        }
    }
    Ok(terms)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MultiplicityFreeFsymbolArguments {
    left: SectorId,
    middle: SectorId,
    right: SectorId,
    coupled: SectorId,
    left_coupled: SectorId,
    right_coupled: SectorId,
}

impl MultiplicityFreeFsymbolArguments {
    #[inline]
    fn is_admissible<R>(self, rule: &R) -> bool
    where
        R: MultiplicityFreeFusionSymbols,
    {
        multi_associator_new_cross_channel_is_admissible(
            rule,
            self.left,
            self.right_coupled,
            self.coupled,
        )
    }

    #[inline]
    fn evaluate<R>(self, rule: &R) -> R::Scalar
    where
        R: MultiplicityFreeFusionSymbols,
    {
        rule.f_symbol_scalar(
            self.left,
            self.middle,
            self.right,
            self.coupled,
            self.left_coupled,
            self.right_coupled,
        )
    }
}

fn multiplicity_free_multi_associator_arguments_from_parts<L, S>(
    long_uncoupled: &[SectorId],
    long: &L,
    short_uncoupled: &[SectorId],
    short: &S,
    tensor_kit_k: usize,
) -> Result<MultiplicityFreeFsymbolArguments, CoreError>
where
    L: MultiplicityFreeTreeLocalData + ?Sized,
    S: MultiplicityFreeTreeLocalData + ?Sized,
{
    let right = long_uncoupled[tensor_kit_k];
    let (left_coupled, coupled) =
        fusion_tree_vertex_neighbors_from_parts(long_uncoupled, long, tensor_kit_k)?;
    let (middle, right_coupled) =
        fusion_tree_vertex_neighbors_from_parts(short_uncoupled, short, tensor_kit_k - 1)?;
    Ok(MultiplicityFreeFsymbolArguments {
        left: long_uncoupled[0],
        middle,
        right,
        coupled,
        left_coupled,
        right_coupled,
    })
}

fn multiplicity_free_multi_associator_grouped<R, F>(
    rule: &R,
    rank: usize,
    candidate_count: usize,
    arguments_for: F,
) -> Result<Vec<Option<R::Scalar>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
    F: Fn(usize, usize) -> Result<MultiplicityFreeFsymbolArguments, CoreError>,
{
    enum Coefficient<S> {
        Pending,
        Active(S),
        Rejected,
    }

    let mut coefficients = (0..candidate_count)
        .map(|_| Coefficient::Pending)
        .collect::<Vec<_>>();
    // Why not retain this table between stages or invocations: TensorKit's
    // grouping is stage-local, and a longer-lived table would create an
    // implicit provider cache for a pure eager reduction.
    let mut stage_symbols: FxHashMap<MultiplicityFreeFsymbolArguments, R::Scalar> =
        FxHashMap::default();
    stage_symbols.reserve(candidate_count);
    for tensor_kit_k in 2..rank {
        stage_symbols.clear();
        for (candidate, coefficient) in coefficients.iter_mut().enumerate() {
            if matches!(coefficient, Coefficient::Rejected) {
                continue;
            }
            let arguments = arguments_for(candidate, tensor_kit_k)?;
            if !arguments.is_admissible(rule) {
                *coefficient = Coefficient::Rejected;
                continue;
            }
            let symbol = match stage_symbols.entry(arguments) {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => {
                    let symbol = arguments.evaluate(rule);
                    entry.insert(symbol.clone());
                    symbol
                }
            };
            *coefficient = match std::mem::replace(coefficient, Coefficient::Rejected) {
                Coefficient::Pending => Coefficient::Active(symbol),
                Coefficient::Active(prefix) => Coefficient::Active(prefix * symbol),
                Coefficient::Rejected => unreachable!("rejected coefficients are skipped"),
            };
        }
    }
    Ok(coefficients
        .into_iter()
        .map(|coefficient| match coefficient {
            Coefficient::Pending => Some(rule.scalar_one()),
            Coefficient::Active(value) => Some(value),
            Coefficient::Rejected => None,
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn multiplicity_free_multi_associator_grouped_fixed_long<R, L, S>(
    rule: &R,
    long_uncoupled: &[SectorId],
    long_is_dual: &[bool],
    long: &L,
    short_uncoupled: &[SectorId],
    short_is_dual: &[bool],
    shorts: &[S],
) -> Result<Vec<Option<R::Scalar>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
    L: MultiplicityFreeTreeLocalData + ?Sized,
    S: MultiplicityFreeTreeLocalData,
{
    let rank = long_uncoupled.len();
    if short_uncoupled.len() + 1 != rank
        || long_uncoupled[1..] != *short_uncoupled
        || long_is_dual[1..] != *short_is_dual
    {
        return Ok((0..shorts.len()).map(|_| None).collect());
    }
    multiplicity_free_multi_associator_grouped(
        rule,
        rank,
        shorts.len(),
        |candidate, tensor_kit_k| {
            multiplicity_free_multi_associator_arguments_from_parts(
                long_uncoupled,
                long,
                short_uncoupled,
                &shorts[candidate],
                tensor_kit_k,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn multiplicity_free_multi_associator_grouped_fixed_short<R, L, S>(
    rule: &R,
    long_uncoupled: &[SectorId],
    long_is_dual: &[bool],
    longs: &[L],
    short_uncoupled: &[SectorId],
    short_is_dual: &[bool],
    short: &S,
) -> Result<Vec<Option<R::Scalar>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
    L: MultiplicityFreeTreeLocalData,
    S: MultiplicityFreeTreeLocalData + ?Sized,
{
    let rank = long_uncoupled.len();
    if short_uncoupled.len() + 1 != rank
        || long_uncoupled[1..] != *short_uncoupled
        || long_is_dual[1..] != *short_is_dual
    {
        return Ok((0..longs.len()).map(|_| None).collect());
    }
    multiplicity_free_multi_associator_grouped(
        rule,
        rank,
        longs.len(),
        |candidate, tensor_kit_k| {
            multiplicity_free_multi_associator_arguments_from_parts(
                long_uncoupled,
                &longs[candidate],
                short_uncoupled,
                short,
                tensor_kit_k,
            )
        },
    )
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
    multiplicity_free_multi_associator_from_parts(
        rule,
        long.uncoupled(),
        long.is_dual(),
        long,
        short.uncoupled(),
        short.is_dual(),
        short,
    )
}

#[allow(clippy::too_many_arguments)]
fn multiplicity_free_multi_associator_from_parts<R, L, S>(
    rule: &R,
    long_uncoupled: &[SectorId],
    long_is_dual: &[bool],
    long: &L,
    short_uncoupled: &[SectorId],
    short_is_dual: &[bool],
    short: &S,
) -> Result<Option<R::Scalar>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
    L: MultiplicityFreeTreeLocalData + ?Sized,
    S: MultiplicityFreeTreeLocalData + ?Sized,
{
    let rank = long_uncoupled.len();
    if short_uncoupled.len() + 1 != rank {
        return Ok(None);
    }
    if long_uncoupled[1..] != *short_uncoupled || long_is_dual[1..] != *short_is_dual {
        return Ok(None);
    }

    let mut coefficient = rule.scalar_one();
    let first = long_uncoupled[0];
    for tensor_kit_k in 2..rank {
        let right_sector = long_uncoupled[tensor_kit_k];
        let (middle_left, middle_right) =
            fusion_tree_vertex_neighbors_from_parts(long_uncoupled, long, tensor_kit_k)?;
        let (short_left, short_right) =
            fusion_tree_vertex_neighbors_from_parts(short_uncoupled, short, tensor_kit_k - 1)?;
        if !multi_associator_new_cross_channel_is_admissible(
            rule,
            first,
            short_right,
            middle_right,
        ) {
            return Ok(None);
        }
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
    fusion_tree_vertex_neighbors_from_parts(tree.uncoupled(), tree, leg_index)
}

fn fusion_tree_vertex_neighbors_from_parts<T>(
    uncoupled: &[SectorId],
    tree: &T,
    leg_index: usize,
) -> Result<(SectorId, SectorId), CoreError>
where
    T: MultiplicityFreeTreeLocalData + ?Sized,
{
    if leg_index == 0 || leg_index >= uncoupled.len() {
        return Err(CoreError::MalformedFusionTree {
            message: "vertex_info requires a non-first uncoupled leg",
        });
    }
    let left = if leg_index == 1 {
        uncoupled[0]
    } else {
        tree.innerlines()
            .get(leg_index - 2)
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "fusion tree is missing a left innerline",
            })?
    };
    let right = if leg_index + 1 == uncoupled.len() {
        tree.coupled()
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

fn inner_extended_sector<T>(tree: &T, index: usize) -> Result<SectorId, CoreError>
where
    T: MultiplicityFreeTreeData + ?Sized,
{
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
        return Ok(tree.coupled());
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

fn unique_rigid_bendright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let (frame, local) = project_multiplicity_free_tree_pair(rule, tree_pair)?;
    let prepared = prepare_multiplicity_free_bendright(rule, &frame)?;
    let validated = prepared.validate_local(rule, &local.codomain, &local.domain)?;
    let output_frame = prepared.output_frame(rule)?;
    let coefficient = prepared.coefficient(rule, &validated);
    Ok((output_frame.materialize(validated.local), coefficient))
}

fn unique_rigid_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let (frame, local) = project_multiplicity_free_tree_pair(rule, tree_pair)?;
    let prepared = prepare_multiplicity_free_bendleft(rule, &frame)?;
    let validated = prepared.validate_local(rule, &local.codomain, &local.domain)?;
    let output_frame = prepared.output_frame(rule)?;
    let (output_local, coefficient) = prepared.finish_local(rule, validated);
    Ok((output_frame.materialize(output_local), coefficient))
}

fn unique_rigid_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let codomain = tree_pair.codomain_tree();
    if codomain.uncoupled().is_empty() {
        return Err(CoreError::MalformedFusionTree {
            message: "foldright requires at least one codomain leg",
        });
    }
    let a = codomain.uncoupled()[0];
    let is_dual_a =
        codomain
            .is_dual()
            .first()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "codomain tree is missing the first duality flag",
            })?;
    let kappa = rule.frobenius_schur_phase_scalar(a);
    let c = codomain.coupled();

    let (codomain_prime, coeff1) = unique_rigid_multi_fmove_tree(rule, codomain)?;
    let b = codomain_prime.coupled();
    let a_symbol = rule.a_symbol_scalar(a, b, c);
    let coeff0 = rule.sqrt_dim_scalar(c) * rule.inv_sqrt_dim_scalar(b);
    let (domain_prime, coeff2) = unique_rigid_multi_fmove_inv_tree(
        rule,
        rule.dual(a),
        b,
        tree_pair.domain_tree(),
        !is_dual_a,
    )?;
    let mut coefficient =
        coeff0 * rule.scalar_conj(coeff2) * a_symbol * coeff1;
    if is_dual_a {
        coefficient = coefficient * kappa;
    }
    Ok((
        FusionTreePairKey::pair(codomain_prime, domain_prime),
        coefficient,
    ))
}

fn unique_rigid_foldleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let swapped = FusionTreePairKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    let (folded, coefficient) = unique_rigid_foldright_tree_pair(rule, &swapped)?;
    Ok((
        FusionTreePairKey::pair(folded.domain_tree().clone(), folded.codomain_tree().clone()),
        rule.scalar_conj(coefficient),
    ))
}

fn unique_rigid_cycle_clockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let (intermediate, first_coefficient) =
        if tree_pair.codomain_tree().uncoupled().is_empty() {
            unique_rigid_bendleft_tree_pair(rule, tree_pair)?
        } else {
            unique_rigid_foldright_tree_pair(rule, tree_pair)?
        };
    let (destination, second_coefficient) =
        if tree_pair.codomain_tree().uncoupled().is_empty() {
            unique_rigid_foldright_tree_pair(rule, &intermediate)?
        } else {
            unique_rigid_bendleft_tree_pair(rule, &intermediate)?
        };
    Ok((destination, first_coefficient * second_coefficient))
}

fn unique_rigid_cycle_anticlockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let (intermediate, first_coefficient) =
        if tree_pair.domain_tree().uncoupled().is_empty() {
            unique_rigid_bendright_tree_pair(rule, tree_pair)?
        } else {
            unique_rigid_foldleft_tree_pair(rule, tree_pair)?
        };
    let (destination, second_coefficient) =
        if tree_pair.domain_tree().uncoupled().is_empty() {
            unique_rigid_foldleft_tree_pair(rule, &intermediate)?
        } else {
            unique_rigid_bendright_tree_pair(rule, &intermediate)?
        };
    Ok((destination, first_coefficient * second_coefficient))
}

fn unique_rigid_repartition_tree_pair_validated<R>(
    validated: ValidatedFusionTreePair<'_, R>,
    target_codomain_rank: usize,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    unique_rigid_repartition_tree_pair_unchecked(
        validated.rule,
        validated.key,
        target_codomain_rank,
    )
}

fn unique_rigid_repartition_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
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
        let (next, step_coefficient) = unique_rigid_bendleft_tree_pair(rule, &current)?;
        coefficient = coefficient * step_coefficient;
        current = next;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        let (next, step_coefficient) = unique_rigid_bendright_tree_pair(rule, &current)?;
        coefficient = coefficient * step_coefficient;
        current = next;
        current_codomain_rank -= 1;
    }
    Ok((current, coefficient))
}

fn unique_rigid_multi_fmove_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let destination = unique_multi_fmove_tree(rule, tree)?;
    let coefficient = multiplicity_free_multi_associator_scalar(rule, tree, &destination)?
        .ok_or(CoreError::MalformedFusionTree {
            message: "unique multi_Fmove destination does not match the source tail",
        })?;
    Ok((destination, coefficient))
}

fn unique_rigid_multi_fmove_inv_tree<R>(
    rule: &R,
    leading_sector: SectorId,
    coupled: SectorId,
    tree: &FusionTreeKey,
    leading_is_dual: bool,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let destination =
        unique_multi_fmove_inv_tree(rule, leading_sector, coupled, tree, leading_is_dual)?;
    let coefficient = multiplicity_free_multi_associator_scalar(rule, &destination, tree)?
        .ok_or(CoreError::MalformedFusionTree {
            message: "unique inverse multi_Fmove destination does not match the source tail",
        })?;
    Ok((destination, rule.scalar_conj(coefficient)))
}

fn unique_bendright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
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

    let coupled = codomain.coupled();
    if !domain.uncoupled().is_empty() && domain.coupled() != coupled {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        });
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
        left_coupled,
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
        new_domain_vertices.push(MultiplicityIndex::ONE);
    }
    let new_domain = FusionTreeKey::new(
        new_domain_uncoupled,
        left_coupled,
        new_domain_is_dual,
        new_domain_innerlines,
        new_domain_vertices,
    );

    let coefficient = rule.bendright_scalar(left_coupled, bent_sector, coupled, bent_is_dual);
    Ok((
        FusionTreePairKey::pair(new_codomain, new_domain),
        coefficient,
    ))
}

fn unique_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
{
    let swapped = FusionTreePairKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    let (bent, coefficient) = unique_bendright_tree_pair(rule, &swapped)?;
    Ok((
        FusionTreePairKey::pair(bent.domain_tree().clone(), bent.codomain_tree().clone()),
        rule.scalar_conj(coefficient),
    ))
}

fn unique_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
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
    let recoupled = codomain_prime.coupled();
    let domain_prime = unique_multi_fmove_inv_tree(
        rule,
        rule.dual(first),
        recoupled,
        tree_pair.domain_tree(),
        !first_is_dual,
    )?;
    let destination = FusionTreePairKey::pair(codomain_prime, domain_prime);
    let coefficient = rule.foldright_scalar(tree_pair, &destination);
    Ok((destination, coefficient))
}

fn unique_foldleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
{
    let swapped = FusionTreePairKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    let (folded, coefficient) = unique_foldright_tree_pair(rule, &swapped)?;
    Ok((
        FusionTreePairKey::pair(folded.domain_tree().clone(), folded.codomain_tree().clone()),
        rule.scalar_conj(coefficient),
    ))
}

fn unique_cycle_clockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
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
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
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
    let coupled = tree.coupled();
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
