use super::*;

/// Test-only oracle helper; production swaps read lines through `ArtinSite`.
#[cfg(test)]
pub(super) fn inner_extended_sector<T>(tree: &T, index: usize) -> Result<SectorId, CoreError>
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

pub(super) fn only_fusion_channel<R>(
    rule: &R,
    left: SectorId,
    right: SectorId,
) -> Result<SectorId, CoreError>
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
{
    unique_rigid_bend_tree_pair(rule, tree_pair, Bend::Right)
}

fn unique_rigid_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    unique_rigid_bend_tree_pair(rule, tree_pair, Bend::Left)
}

fn unique_rigid_bend_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    bend: Bend,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let (frame, local) = project_multiplicity_free_tree_pair(rule, tree_pair)?;
    let (state, coefficient) =
        unique_rigid_bend_state(rule, UniqueRigidTreePairState { frame, local }, bend)?;
    Ok((state.frame.materialize(state.local), coefficient))
}

struct UniqueRigidTreePairState {
    frame: MultiplicityFreeTreePairFrame,
    local: MultiplicityFreeTreePairLocal,
}

fn unique_rigid_bend_state<R>(
    rule: &R,
    state: UniqueRigidTreePairState,
    bend: Bend,
) -> Result<(UniqueRigidTreePairState, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let prepared = MultiplicityFreeBend::prepare(&state.frame, bend)?;
    let (local, coefficient) = prepared.apply(&UniqueK(rule), &state.local)?;
    let frame = prepared.output_frame(rule, &state.frame)?;
    Ok((UniqueRigidTreePairState { frame, local }, coefficient))
}

fn unique_rigid_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    // Why not `fold_surgery`: a unique fold has exactly one forward and one
    // inverse move, and threading that single term through the surgery's
    // move iterators measurably slows cold unique transposes (#1852 B2c).
    // These are the surgery's steps, unrolled, with the multiplicity-free
    // coefficient kernel.
    let kernel = SimpleK(rule);
    let codomain = tree_pair.codomain_tree();
    let site = FoldSite::new(codomain.uncoupled(), codomain.is_dual())?;
    let (fold, dual_first) = kernel.fold_begin(&site)?;
    let (codomain_prime, coeff1) = unique_rigid_multi_fmove_tree(rule, codomain)?;
    let tail_coupled = codomain_prime.coupled();
    let factors = kernel.fold_factors(&fold, tail_coupled, codomain.coupled())?;
    let (domain_prime, coeff2) = unique_rigid_multi_fmove_inv_tree(
        rule,
        dual_first,
        tail_coupled,
        tree_pair.domain_tree(),
        !site.first_is_dual,
    )?;
    let coefficient = kernel.fold_coefficient(&fold, &factors, &coeff1, &coeff2)?;
    Ok((
        FusionTreePairKey::pair(codomain_prime, domain_prime),
        coefficient,
    ))
}

pub(crate) fn unique_rigid_cycle_clockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    cycle_clockwise(
        tree_pair,
        |key| unique_rigid_bendleft_tree_pair(rule, key),
        |key| unique_rigid_foldright_tree_pair(rule, key),
    )
}

pub(crate) fn unique_rigid_cycle_anticlockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    cycle_anticlockwise(
        tree_pair,
        |key| unique_rigid_bendright_tree_pair(rule, key),
        |key| {
            left_move_by_swap(key, |swapped| {
                unique_rigid_foldright_tree_pair(rule, swapped)
            })
        },
    )
}

pub(super) fn unique_rigid_repartition_tree_pair_validated<R>(
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

pub(crate) fn unique_rigid_repartition_tree_pair_unchecked<R>(
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

    let (frame, local) = project_multiplicity_free_tree_pair(rule, tree_pair)?;
    let codomain_rank = frame.codomain.uncoupled.len();
    let (current, coefficient) = repartition_loop(
        (UniqueRigidTreePairState { frame, local }, R::Scalar::one()),
        codomain_rank,
        target_codomain_rank,
        |(state, coefficient), bend| {
            let (next, step_coefficient) = unique_rigid_bend_state(rule, state, bend)?;
            Ok::<_, CoreError>((next, coefficient * step_coefficient))
        },
    )?;
    Ok((current.frame.materialize(current.local), coefficient))
}

pub(crate) fn unique_rigid_multi_fmove_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let destination = unique_multi_fmove_tree(rule, tree)?;
    let coefficient = multiplicity_free_multi_associator_scalar(rule, tree, &destination)?.ok_or(
        CoreError::MalformedFusionTree {
            message: "unique multi_Fmove destination does not match the source tail",
        },
    )?;
    Ok((destination, coefficient))
}

pub(crate) fn unique_rigid_multi_fmove_inv_tree<R>(
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
    let coefficient = multiplicity_free_multi_associator_scalar(rule, &destination, tree)?.ok_or(
        CoreError::MalformedFusionTree {
            message: "unique inverse multi_Fmove destination does not match the source tail",
        },
    )?;
    Ok((destination, (coefficient).conj()))
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
