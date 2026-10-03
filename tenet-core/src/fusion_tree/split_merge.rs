use super::*;

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
    split_tree_structural(rule.vacuum(), tree, front_rank)
}

/// Split a Generic fusion tree without requiring the infallible `FusionRule`
/// contract.
///
/// This is structural only: no F/R or braid data is queried.  The input and
/// both output trees are checked through the provider-owned Generic admission
/// path before publication.
pub fn split_fusion_tree_generic_checked<C>(
    rule: &C,
    tree: &FusionTreeKey,
    front_rank: usize,
) -> Result<(FusionTreeKey, FusionTreeKey), CheckedGenericStructureError<C::Error>>
where
    C: CheckedGenericFusion,
{
    validate_generic_fusion_tree_pair_checked(
        rule,
        &FusionTreePairKey::pair(tree.clone(), tree.clone()),
    )?;
    let rank = tree.uncoupled().len();
    if front_rank > rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: front_rank,
        }
        .into());
    }

    let (front_tree, tail_tree) = split_tree_structural(rule.vacuum(), tree, front_rank)?;

    for output in [&front_tree, &tail_tree] {
        validate_generic_fusion_tree_pair_checked(
            rule,
            &FusionTreePairKey::pair(output.clone(), output.clone()),
        )?;
    }
    Ok((front_tree, tail_tree))
}

/// Merge two standard fusion trees without exchanging their external legs.
///
/// This is TensorKit's `merge(f₁, f₂, c, 1)` for a multiplicity-free rule:
/// the two coupled sectors are fused to `coupled`, then only inverse
/// associators are used to restore the left-associated standard tree. In
/// particular this operation never evaluates an R symbol and is valid for
/// [`BraidingStyleKind::NoBraiding`] providers.
///
/// Both inputs and the requested fusion channel are validated through the
/// checked algebra before any recoupling symbols are evaluated.
pub fn merge_fusion_trees_multiplicity_free<R>(
    rule: &R,
    lhs: &FusionTreeKey,
    rhs: &FusionTreeKey,
    coupled: SectorId,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CheckedFusionSpaceError>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + CanonicalUnitFusionRule,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    lhs.validate_for_rule_checked(rule)?;
    rhs.validate_for_rule_checked(rule)?;
    let multiplicity = rule.try_nsymbol(lhs.coupled(), rhs.coupled(), coupled)?;
    if multiplicity != 1 {
        return Err(CoreError::FusionChannelCount {
            left: lhs.coupled(),
            right: rhs.coupled(),
            count: multiplicity,
        }
        .into());
    }

    // The CanonicalUnitFusionRule bound is the proof that these rank-zero
    // branches have unit coefficient. Handling them structurally also avoids
    // manufacturing and then deleting a synthetic vacuum leaf.
    if lhs.uncoupled().is_empty() {
        return Ok(vec![(rhs.clone(), R::Scalar::one())]);
    }
    if rhs.uncoupled().is_empty() {
        return Ok(vec![(lhs.clone(), R::Scalar::one())]);
    }

    multiplicity_free_multi_fmove_inv_tree(rule, lhs.coupled(), coupled, rhs, false)?
        .into_iter()
        .map(|(tail, coefficient)| {
            join_fusion_tree_front_checked(rule, lhs, &tail).map(|merged| (merged, coefficient))
        })
        .collect()
}

/// Checked Generic-fusion counterpart of [`merge_fusion_trees_multiplicity_free`].
///
/// `root_vertex` is the one shared outer-multiplicity label for both halves of
/// tensor-map `otimes`.  This is F-only: merging does not exchange legs.
pub fn merge_fusion_trees_generic_checked<C>(
    rule: &C,
    lhs: &FusionTreeKey,
    rhs: &FusionTreeKey,
    coupled: SectorId,
    root_vertex: MultiplicityIndex,
) -> Result<GenericTreeTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(CheckedGenericSymbolError::Core(
            CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual: rule.fusion_style(),
            },
        ));
    }
    validate_generic_fusion_tree_pair_checked(
        rule,
        &FusionTreePairKey::pair(lhs.clone(), lhs.clone()),
    )
    .map_err(map_checked_generic_structure_error)?;
    validate_generic_fusion_tree_pair_checked(
        rule,
        &FusionTreePairKey::pair(rhs.clone(), rhs.clone()),
    )
    .map_err(map_checked_generic_structure_error)?;
    let n = rule
        .try_nsymbol(lhs.coupled(), rhs.coupled(), coupled)
        .map_err(CheckedGenericSymbolError::Provider)?;
    if root_vertex.get() == 0 || root_vertex.get() > n {
        return Err(CheckedGenericSymbolError::Core(
            CoreError::MalformedFusionTree {
                message: "tensor-product root vertex exceeds Nsymbol",
            },
        ));
    }
    if lhs.uncoupled().is_empty() {
        return Ok(vec![(rhs.clone(), C::Scalar::one())]);
    }
    if rhs.uncoupled().is_empty() {
        return Ok(vec![(lhs.clone(), C::Scalar::one())]);
    }
    let terms = generic_multi_fmove_inv_tree_checked(rule, lhs.coupled(), coupled, rhs, false)?;
    terms
        .into_iter()
        .map(|(tail, coefficients)| {
            let coefficient = coefficients.get(root_vertex.get() - 1).cloned().ok_or(
                CheckedGenericSymbolError::Core(CoreError::MalformedFusionTree {
                    message: "tensor-product root vertex coefficient is absent",
                }),
            )?;
            if coefficient.is_zero() {
                return Ok(None);
            }
            join_fusion_tree_front_generic_checked(rule, lhs, &tail)
                .map(|merged| Some((merged, coefficient)))
        })
        .filter_map(Result::transpose)
        .collect()
}

fn join_fusion_tree_front_generic_checked<C>(
    rule: &C,
    front: &FusionTreeKey,
    tail: &FusionTreeKey,
) -> Result<FusionTreeKey, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericFusion,
{
    let boundary = *tail
        .uncoupled()
        .first()
        .ok_or(CheckedGenericSymbolError::Core(
            CoreError::MalformedFusionTree {
                message: "fusion-tree front join requires a non-empty tail",
            },
        ))?;
    if boundary != front.coupled() || tail.is_dual()[0] {
        return Err(CheckedGenericSymbolError::Core(
            CoreError::MalformedFusionTree {
                message: "Generic fusion-tree front join has an invalid boundary",
            },
        ));
    }
    let merged = join_tree_structural(front, tail);
    validate_generic_fusion_tree_pair_checked(
        rule,
        &FusionTreePairKey::pair(merged.clone(), merged.clone()),
    )
    .map_err(map_checked_generic_structure_error)?;
    Ok(merged)
}

/// Replace the first, non-dual leaf of `tail` by `front`.
///
/// This is the checked inverse of [`split_fusion_tree`] at the corresponding
/// front rank. It is intentionally private: tensor product is the only caller
/// that needs this narrower structural operation.
fn join_fusion_tree_front_checked<R>(
    rule: &R,
    front: &FusionTreeKey,
    tail: &FusionTreeKey,
) -> Result<FusionTreeKey, CheckedFusionSpaceError>
where
    R: CheckedFusionAlgebra,
{
    front.validate_for_rule_checked(rule)?;
    tail.validate_for_rule_checked(rule)?;
    let Some(&boundary) = tail.uncoupled().first() else {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion-tree front join requires a non-empty tail",
        }
        .into());
    };
    if boundary != front.coupled() {
        return Err(CoreError::SectorMismatch {
            expected: front.coupled(),
            actual: boundary,
        }
        .into());
    }
    if tail.is_dual()[0] {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion-tree front join requires a non-dual boundary leaf",
        }
        .into());
    }

    if front.uncoupled().is_empty() {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion-tree front join requires a non-empty front",
        }
        .into());
    }
    let merged = join_tree_structural(front, tail);
    merged.validate_for_rule_checked(rule)?;
    Ok(merged)
}

/// TensorKit `split(f, M)` tree surgery (`basic_manipulations.jl:30`), shared by
/// the multiplicity-free and Generic entry points; each caller owns its own
/// validation order. `front_rank <= rank` is checked by the caller.
fn split_tree_structural(
    unit: SectorId,
    tree: &FusionTreeKey,
    front_rank: usize,
) -> Result<(FusionTreeKey, FusionTreeKey), CoreError> {
    let rank = tree.uncoupled().len();
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

/// Replace the first leaf of `tail` by `front` (TensorKit `join(f₁, f₂)`,
/// `basic_manipulations.jl:78`): the inverse tree surgery of
/// [`split_tree_structural`], shared by the multiplicity-free and Generic
/// merges. The caller has checked the boundary leaf.
fn join_tree_structural(front: &FusionTreeKey, tail: &FusionTreeKey) -> FusionTreeKey {
    match (front.uncoupled().len(), tail.uncoupled().len()) {
        (1, _) => {
            let mut is_dual = tail.is_dual().to_vec();
            is_dual[0] = front.is_dual()[0];
            FusionTreeKey::new(
                tail.uncoupled().to_vec(),
                tail.coupled(),
                is_dual,
                tail.innerlines().to_vec(),
                tail.vertices().to_vec(),
            )
        }
        (_, 1) => front.clone(),
        (_, _) => {
            let mut uncoupled =
                Vec::with_capacity(front.uncoupled().len() + tail.uncoupled().len() - 1);
            uncoupled.extend_from_slice(front.uncoupled());
            uncoupled.extend_from_slice(&tail.uncoupled()[1..]);
            let mut is_dual = Vec::with_capacity(front.is_dual().len() + tail.is_dual().len() - 1);
            is_dual.extend_from_slice(front.is_dual());
            is_dual.extend_from_slice(&tail.is_dual()[1..]);
            let mut innerlines =
                Vec::with_capacity(front.innerlines().len() + 1 + tail.innerlines().len());
            innerlines.extend_from_slice(front.innerlines());
            innerlines.push(front.coupled());
            innerlines.extend_from_slice(tail.innerlines());
            let mut vertices = Vec::with_capacity(front.vertices().len() + tail.vertices().len());
            vertices.extend_from_slice(front.vertices());
            vertices.extend_from_slice(tail.vertices());
            FusionTreeKey::new(uncoupled, tail.coupled(), is_dual, innerlines, vertices)
        }
    }
}
