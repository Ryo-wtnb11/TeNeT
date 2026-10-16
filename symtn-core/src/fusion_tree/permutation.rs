use super::*;

pub(crate) fn linearize_tree_pair_permutation(
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
    validate_permutation_inline(&original_permutation, total_rank)?;

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
    validate_permutation_inline(&linearized, total_rank)?;
    Ok(linearized)
}

pub(super) fn validate_tree_pair_axis_map_inline(
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
    let mut seen = crate::axes::AxisMask::new(total_rank);
    for position in 0..total_rank {
        let axis = raw_tree_pair_axis_at(codomain_permutation, domain_permutation, position);
        if seen.insert(axis).is_err() {
            return Err(invalid_tree_pair_axis_map(
                codomain_permutation,
                domain_permutation,
                total_rank,
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_tree_pair_axis_map_without_scratch(
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

pub(super) fn validate_cyclic_tree_pair_axis_map_inline(
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

pub(super) fn raw_tree_pair_axis_at(
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

pub(super) fn linearized_tree_pair_axis_at(
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

pub(super) fn materialize_linearized_tree_pair_permutation(
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

pub(super) fn tree_pair_axis_map_is_identity(
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

fn linearize_tree_pair_axis(axis: usize, codomain_rank: usize, domain_rank: usize) -> usize {
    if axis < codomain_rank {
        axis
    } else {
        domain_rank + 2 * codomain_rank - 1 - axis
    }
}

pub(super) fn is_cyclic_permutation(permutation: &[usize]) -> bool {
    let rank = permutation.len();
    for index in 0..rank {
        if permutation[(index + 1) % rank] != (permutation[index] + 1) % rank {
            return false;
        }
    }
    true
}
