use super::*;

pub(crate) fn execute_unique_tree_braid<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    artin_steps: &[PreparedArtinStep],
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rank = tree.uncoupled().len();
    if permutation.len() != rank {
        return Err(CoreError::InvalidPermutation {
            permutation: permutation.to_vec(),
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
        let mut coefficient = R::Scalar::one();
        for right_position in 0..rank {
            for left_position in 0..right_position {
                let left_axis = permutation[left_position];
                let right_axis = permutation[right_position];
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
                    coefficient = coefficient * rule.r_symbol_scalar(left, right, coupled);
                }
            }
        }
        let uncoupled = permutation
            .iter()
            .map(|&axis| tree.uncoupled()[axis])
            .collect::<SmallVec<[SectorId; 8]>>();
        let is_dual = permutation
            .iter()
            .map(|&axis| tree.is_dual()[axis])
            .collect::<SmallVec<[bool; 8]>>();
        let coupled = tree.coupled();
        let destination = rebuild_unique_standard_fusion_tree(rule, &uncoupled, coupled, &is_dual)?;
        return Ok((destination, coefficient));
    }

    execute_unique_tree_braid_steps(rule, tree, artin_steps.iter().copied())
}

pub(crate) fn execute_unique_tree_braid_borrowed<R>(
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
        let mut coefficient = R::Scalar::one();
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
        let destination =
            rebuild_unique_standard_fusion_tree(rule, &uncoupled, tree.coupled(), &is_dual)?;
        return Ok((destination, coefficient));
    }

    execute_unique_tree_braid_steps(rule, tree, braid.artin_steps())
}

pub(crate) fn execute_unique_tree_braid_steps<R, I>(
    rule: &R,
    tree: &FusionTreeKey,
    steps: I,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
    I: IntoIterator<Item = PreparedArtinStep>,
{
    // Why not freeze a key after every step: this state is private to one
    // execution, while only the final categorical identity can escape.
    let mut current = UnhashedFusionTree::from(tree.clone());
    let mut coefficient = R::Scalar::one();
    for step in steps {
        let step_coefficient =
            apply_unique_artin_braid_at_with_inverse(rule, &mut current, step.index, step.inverse)?;
        coefficient = coefficient * step_coefficient;
    }
    Ok((current.freeze(), coefficient))
}

pub(crate) fn is_unique_direct_braid_source<R>(rule: &R, tree: &FusionTreeKey) -> bool
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

#[cfg(test)]
pub(crate) fn immutable_unique_artin_braid_at_with_inverse_oracle<R>(
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
        return Ok((braided, R::Scalar::one()));
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
            (rule.r_symbol_scalar(right, left, coupled)).conj()
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
        (left * f_symbol).conj() * right
    } else {
        let left = rule.r_symbol_scalar(c, d, e);
        let right = rule.r_symbol_scalar(a, d, c_prime);
        left * (f_symbol * right).conj()
    };
    Ok((braided, coefficient))
}
