use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CoupledFusionTrees {
    pub(crate) coupled: SectorId,
    pub(crate) trees: Vec<FusionTreeKey>,
}

pub(crate) fn validate_multiplicity_free_execution_style<R>(rule: &R) -> Result<(), CoreError>
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

pub(crate) fn try_visit_selected_leg_tuples<E, F>(
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

pub(crate) fn fusion_trees_by_coupled_for_space<R>(
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
            let frozen_uncoupled: Arc<[SectorId]> = uncoupled.clone().into();
            let frozen_is_dual: Arc<[bool]> = is_dual.clone().into();
            let frozen_vertices: Arc<[MultiplicityIndex]> =
                std::iter::repeat_n(MultiplicityIndex::ONE, uncoupled.len().saturating_sub(1))
                    .collect::<Vec<_>>()
                    .into();
            for coupled in reachable_coupled_sectors(rule, &effective) {
                let trees = collect_fusion_trees_for_coupled_frozen(
                    rule,
                    &frozen_uncoupled,
                    &frozen_is_dual,
                    &frozen_vertices,
                    &effective,
                    coupled,
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

/// Checked sibling of [`fusion_trees_by_coupled_for_space`] for external
/// multiplicity-free providers.
///
/// It enumerates the identical [`CoupledFusionTrees`] set and canonical order
/// using only the fallible [`CheckedFusionAlgebra`] primitives, returning the
/// exact [`FusionAlgebraError`] instead of relying on the infallible hot-path
/// methods, which may panic or overflow on an unrepresentable sector produced
/// by an external provider.
///
/// Why a sibling and not a shared enumeration core: the built-in lowered fast
/// path decodes each sector to an associated type and the infallible path calls
/// `fusion_channels`/`nsymbol` directly; threading either through a shared
/// generic core would perturb their output ordering or dispatch. This walk is
/// structurally identical to the infallible enumerator, so its keys and order
/// match byte-for-byte on any provider whose checked methods agree with their
/// infallible counterparts.
pub(crate) fn try_fusion_trees_by_coupled_for_space_checked<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> Result<Vec<CoupledFusionTrees>, FusionAlgebraError>
where
    R: CheckedFusionAlgebra,
{
    let mut grouped = Vec::<CoupledFusionTrees>::new();
    let mut index: FxHashMap<SectorId, usize> = FxHashMap::default();
    let mut uncoupled = Vec::with_capacity(space.len());
    let mut is_dual = Vec::with_capacity(space.len());
    let mut effective = Vec::with_capacity(space.len());
    space.try_visit_selected_leg_tuples(&mut |tuple| {
        uncoupled.clear();
        is_dual.clear();
        effective.clear();
        for leg in tuple {
            uncoupled.push(leg.sector());
            is_dual.push(leg.is_dual());
            effective.push(leg.sector());
        }
        let frozen_uncoupled: Arc<[SectorId]> = uncoupled.clone().into();
        let frozen_is_dual: Arc<[bool]> = is_dual.clone().into();
        let frozen_vertices: Arc<[MultiplicityIndex]> =
            std::iter::repeat_n(MultiplicityIndex::ONE, uncoupled.len().saturating_sub(1))
                .collect::<Vec<_>>()
                .into();
        for coupled in try_reachable_coupled_sectors_checked(rule, &effective)? {
            let trees = try_collect_fusion_trees_for_coupled_frozen_checked(
                rule,
                &frozen_uncoupled,
                &frozen_is_dual,
                &frozen_vertices,
                &effective,
                coupled,
            )?;
            match index.get(&coupled) {
                Some(&i) => grouped[i].trees.extend(trees),
                None => {
                    index.insert(coupled, grouped.len());
                    grouped.push(CoupledFusionTrees { coupled, trees });
                }
            }
        }
        Ok::<(), FusionAlgebraError>(())
    })?;
    grouped.sort_by_key(|group| group.coupled);
    Ok(grouped)
}

fn try_reachable_coupled_sectors_checked<R>(
    rule: &R,
    effective: &[SectorId],
) -> Result<Vec<SectorId>, FusionAlgebraError>
where
    R: CheckedFusionAlgebra,
{
    let mut acc: Vec<SectorId> = match effective.first() {
        None => vec![rule.vacuum()],
        Some(&first) => vec![first],
    };
    for &last in effective.iter().skip(1) {
        let mut next = Vec::new();
        for &front in &acc {
            next.extend(rule.try_fusion_channels(front, last)?);
        }
        next.sort_unstable();
        next.dedup();
        acc = next;
    }
    acc.sort_unstable();
    acc.dedup();
    Ok(acc)
}

fn try_collect_fusion_trees_for_coupled_frozen_checked<R>(
    rule: &R,
    uncoupled: &Arc<[SectorId]>,
    is_dual: &Arc<[bool]>,
    vertices: &Arc<[MultiplicityIndex]>,
    effective: &[SectorId],
    coupled: SectorId,
) -> Result<Vec<FusionTreeKey>, FusionAlgebraError>
where
    R: CheckedFusionAlgebra,
{
    let mut out = Vec::new();
    // `inner_rev` accumulates the inner lines outermost-first as the walk
    // descends; the stored key wants innermost-first, so emit reverses it.
    let mut inner_rev: Vec<SectorId> = Vec::new();
    try_visit_fusion_trees_checked(rule, effective, coupled, &mut inner_rev, &mut |inner_rev| {
        out.push(FusionTreeKey::from_frozen(
            Arc::clone(uncoupled),
            coupled,
            Arc::clone(is_dual),
            inner_rev.iter().rev().copied().collect::<Vec<_>>().into(),
            Arc::clone(vertices),
        ));
        Ok(())
    })?;
    Ok(out)
}

fn try_visit_fusion_trees_checked<R, F>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
    inner_rev: &mut Vec<SectorId>,
    emit: &mut F,
) -> Result<(), FusionAlgebraError>
where
    R: CheckedFusionAlgebra,
    F: FnMut(&[SectorId]) -> Result<(), FusionAlgebraError>,
{
    match effective.len() {
        0 => {
            if coupled == rule.vacuum() {
                emit(inner_rev)?;
            }
        }
        1 => {
            if effective[0] == coupled {
                emit(inner_rev)?;
            }
        }
        2 => {
            if rule.try_nsymbol(effective[0], effective[1], coupled)? != 0 {
                emit(inner_rev)?;
            }
        }
        _ => {
            let last = effective[effective.len() - 1];
            let front_effective = &effective[..effective.len() - 1];
            let dual_last = rule.try_dual_sector(last)?;
            // Inner line `a` ranges over `coupled ⊗ dual(last)`; `Nsymbol(a,
            // last, coupled)` is the last vertex. Identical to the infallible
            // `visit_fusion_trees` walk so the emitted key order matches.
            for front_coupled in rule.try_fusion_channels(coupled, dual_last)? {
                if rule.try_nsymbol(front_coupled, last, coupled)? == 0 {
                    continue;
                }
                inner_rev.push(front_coupled);
                let result = try_visit_fusion_trees_checked(
                    rule,
                    front_effective,
                    front_coupled,
                    inner_rev,
                    emit,
                );
                inner_rev.pop();
                result?;
            }
        }
    }
    Ok(())
}

pub(crate) fn fusion_trees_by_coupled_for_selected_space<R>(
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
    let frozen_uncoupled: Arc<[SectorId]> = uncoupled.into();
    let frozen_is_dual: Arc<[bool]> = is_dual.into();
    let frozen_vertices: Arc<[MultiplicityIndex]> = std::iter::repeat_n(
        MultiplicityIndex::ONE,
        frozen_uncoupled.len().saturating_sub(1),
    )
    .collect::<Vec<_>>()
    .into();
    let mut grouped = Vec::new();
    for coupled in reachable_coupled_sectors(rule, &effective) {
        let trees = collect_fusion_trees_for_coupled_frozen(
            rule,
            &frozen_uncoupled,
            &frozen_is_dual,
            &frozen_vertices,
            &effective,
            coupled,
        );
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
/// `MultiplicityFreeFusionRule`) so outer-multiplicity rules can drive the
/// Space layer.
///
/// Escape semantics (Option A, refute/b3b-verify): the coupled candidates of
/// each leg tuple are classified by [`FusionRule::coupled_sector_fold`]. Trees
/// are enumerated for CLEAN sectors only (their tree set is exactly the
/// provider's complete set); tainted / escaped / poisoned candidates are reported in the
/// returned aggregate so the caller can refuse construction with an `Err` —
/// block dimensions are either exactly right or an error, never silently
/// truncated. A sector clean in one tuple but tainted in another is tainted
/// overall (its block would mix complete and incomplete tree sets).
pub(crate) fn fusion_trees_by_coupled_for_space_generic<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> (Vec<CoupledFusionTrees>, CoupledSectorFold)
where
    R: FusionRule,
{
    let checked = InfallibleGeneric::new(rule);
    match fusion_trees_by_coupled_for_space_generic_checked(&checked, space) {
        Ok(result) => result,
        Err(CheckedGenericStructureError::Provider(never)) => match never {},
        Err(CheckedGenericStructureError::Core(error)) => {
            // The legacy API only reports the bounded-table failure after the
            // aggregate is built. All other core failures are impossible for
            // a structurally valid legacy FusionRule walk.
            panic!("legacy Generic structural walk failed unexpectedly: {error}")
        }
    }
}

pub(crate) fn fusion_trees_by_coupled_for_space_generic_checked<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> Result<(Vec<CoupledFusionTrees>, CoupledSectorFold), CheckedGenericStructureError<R::Error>>
where
    R: CheckedGenericFusion,
{
    let mut grouped = Vec::<CoupledFusionTrees>::new();
    let mut index: FxHashMap<SectorId, usize> = FxHashMap::default();
    let mut aggregate = CoupledSectorFoldBuilder::default();
    let mut uncoupled = Vec::with_capacity(space.len());
    let mut is_dual = Vec::with_capacity(space.len());
    let mut effective = Vec::with_capacity(space.len());
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
        let frozen_uncoupled: Arc<[SectorId]> = uncoupled.clone().into();
        let frozen_is_dual: Arc<[bool]> = is_dual.clone().into();
        let fold = rule
            .try_coupled_sector_fold(&effective)
            .map_err(CheckedGenericStructureError::Provider)?;
        for &coupled in fold.clean() {
            let trees = collect_generic_fusion_trees_for_coupled_frozen_checked(
                rule,
                &frozen_uncoupled,
                &frozen_is_dual,
                &effective,
                coupled,
            )?;
            match index.get(&coupled) {
                Some(&i) => grouped[i].trees.extend(trees),
                None => {
                    index.insert(coupled, grouped.len());
                    grouped.push(CoupledFusionTrees { coupled, trees });
                }
            }
        }
        aggregate.absorb(fold);
        Ok::<(), CheckedGenericStructureError<R::Error>>(())
    })?;
    // Tainted-anywhere beats clean-somewhere, and an unknown split anywhere
    // demotes every candidate; both live in the builder's seal.
    let aggregate = aggregate.seal();
    // Drop tree groups of sectors that lost their clean status across tuples.
    grouped.retain(|group| aggregate.clean().contains(&group.coupled));
    grouped.sort_by_key(|group| group.coupled);
    Ok((grouped, aggregate))
}

/// Shared codomain×domain merge on equal coupled sectors (the generic sibling
/// of the loop in `fusion_tree_keys_uncached`).
#[cfg(test)]
pub(crate) fn merge_generic_tree_groups(
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
pub(crate) fn fusion_fold_error_message(side: &str, fold: &CoupledSectorFold) -> String {
    let mut parts = Vec::new();
    if !fold.out_of_table().is_empty() {
        parts.push(format!(
            "out-of-table coupled candidates on the {side} side: {}",
            fold.out_of_table().join(", ")
        ));
    }
    if !fold.tainted().is_empty() {
        parts.push(format!(
            "sectors requiring out-of-table intermediates on the {side} side: {:?}",
            fold.tainted()
        ));
    }
    if fold.is_unknown() {
        parts.push(format!(
            "the {side}-side fold left the one-hop frontier shell (conservative)"
        ));
    }
    format!(
        "bounded Generic provider cannot represent this space exactly ({}); block \
         dimensions are either exact or an error, never truncated. Use \
         fusion_tree_keys_generic_for_coupled for provably-clean sectors, or \
         extend the provider catalog.",
        parts.join("; ")
    )
}

/// Coupled sectors reachable by fusing all legs — TensorKit's `blocksectors`.
/// Computed once per leg tuple (not per enumeration node): the forward fold
/// `⊗` over the legs with dedup. Used only to drive the per-coupled grouping;
/// the tree enumeration itself does not consult it (see below).
pub(crate) fn reachable_coupled_sectors<R>(rule: &R, effective: &[SectorId]) -> Vec<SectorId>
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
pub(crate) fn collect_fusion_trees_for_coupled<R>(
    rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
) -> Vec<FusionTreeKey>
where
    R: MultiplicityFreeFusionRule,
{
    let frozen_uncoupled = Arc::from(uncoupled);
    let frozen_is_dual = Arc::from(is_dual);
    let frozen_vertices =
        std::iter::repeat_n(MultiplicityIndex::ONE, uncoupled.len().saturating_sub(1))
            .collect::<Vec<_>>()
            .into();
    collect_fusion_trees_for_coupled_frozen(
        rule,
        &frozen_uncoupled,
        &frozen_is_dual,
        &frozen_vertices,
        effective,
        coupled,
    )
}

fn collect_fusion_trees_for_coupled_frozen<R>(
    rule: &R,
    uncoupled: &Arc<[SectorId]>,
    is_dual: &Arc<[bool]>,
    vertices: &Arc<[MultiplicityIndex]>,
    effective: &[SectorId],
    coupled: SectorId,
) -> Vec<FusionTreeKey>
where
    R: MultiplicityFreeFusionRule,
{
    let mut out = Vec::new();
    // `inner_rev` accumulates the inner lines outermost-first as the walk
    // descends; the stored key wants innermost-first, so emit reverses it.
    let mut inner_rev: Vec<SectorId> = Vec::new();
    visit_fusion_trees(rule, effective, coupled, &mut inner_rev, &mut |inner_rev| {
        out.push(FusionTreeKey::from_frozen(
            Arc::clone(uncoupled),
            coupled,
            Arc::clone(is_dual),
            inner_rev.iter().rev().copied().collect::<Vec<_>>().into(),
            Arc::clone(vertices),
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
    visit_fusion_trees_where(rule, effective, coupled, inner_rev, &|_, _| true, emit);
}

pub(super) fn visit_fusion_trees_where<R, P, F>(
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
