use super::*;

/// Exact single-tree permutation lowering for a unique fusion rule.
///
/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn unique_permute_tree<R>(
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
        return Ok((tree.clone(), R::Scalar::one()));
    }
    execute_unique_tree_braid(rule, tree, &prepared.permutation, &prepared.artin_steps)
}

/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
#[cfg(test)]
pub(crate) fn multiplicity_free_braid_tree<R>(
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

pub(crate) fn execute_multiplicity_free_tree_braid_proven<R>(
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
        return Ok(vec![(tree.clone(), R::Scalar::one())]);
    }
    execute_multiplicity_free_tree_braid(rule, tree, &prepared.permutation, &prepared.artin_steps)
}

pub(crate) fn execute_multiplicity_free_tree_braid<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    artin_steps: &[PreparedArtinStep],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if rule.fusion_style() == FusionStyleKind::Unique {
        let (destination, coefficient) =
            execute_unique_tree_braid(rule, tree, permutation, artin_steps)?;
        return Ok(vec![(destination, coefficient)]);
    }

    execute_multiplicity_free_tree_braid_steps(rule, tree, artin_steps.iter().copied())
}

pub(super) fn execute_multiplicity_free_tree_braid_steps<R, I>(
    rule: &R,
    tree: &FusionTreeKey,
    steps: I,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    I: IntoIterator<Item = PreparedArtinStep>,
{
    if rule.fusion_style() == FusionStyleKind::Unique {
        let (destination, coefficient) = execute_unique_tree_braid_steps(rule, tree, steps)?;
        return Ok(vec![(destination, coefficient)]);
    }

    let mut current = vec![(tree.clone(), R::Scalar::one())];
    for step in steps {
        let mut next_terms = FusionTermAccumulator::new();
        for (tree, coefficient) in current {
            for (next_tree, step_coefficient) in multiplicity_free_artin_braid_at_with_inverse(
                rule,
                &tree,
                step.index,
                step.inverse,
            )? {
                next_terms.push(next_tree, coefficient.clone() * step_coefficient);
            }
        }
        current = next_terms.into_vec();
    }
    Ok(current)
}

/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
#[cfg(test)]
pub(crate) fn multiplicity_free_permute_tree<R>(
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

pub(super) fn multiplicity_free_repartition_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rule = tree_pair.rule;
    let tree_pair = tree_pair.key;
    let mut current = vec![(tree_pair.clone(), R::Scalar::one())];
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
#[cfg(any(test, feature = "testing"))]
pub(crate) fn multiplicity_free_transpose_tree_pair<R>(
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
