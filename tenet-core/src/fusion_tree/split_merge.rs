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
    let mut rows = TreeScratch::default();
    let (front, back) = split_fusion_tree_into(rule, tree.as_view(), front_rank, &mut rows)?;
    Ok(rows.into_keys(front, back))
}

/// [`split_fusion_tree`] of a borrowed tree, with both halves appended to
/// `rows`; returns their rows. Nothing is allocated once `rows` has held a
/// split this wide.
#[doc(hidden)]
pub fn split_fusion_tree_into<R>(
    rule: &R,
    tree: FusionTreeRef<'_>,
    front_rank: usize,
    rows: &mut TreeScratch,
) -> Result<(usize, usize), CoreError>
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
    validate_fusion_tree_view_for_rule(rule, tree)?;
    push_split_rows(rule.vacuum(), tree, front_rank, rows.builder())
}

/// Split a Generic fusion tree without requiring the infallible `FusionRule`
/// contract.
///
/// This is structural only: no F/R or braid data is queried. The input tree is
/// admitted through the provider-owned Generic admission path; both outputs
/// are structural sub-trees of that admitted tree (every leaf, innerline and
/// vertex is copied from it, plus a unit leaf or a rank-one tree on a sector
/// the input already admitted), so they are published without re-admission,
/// as TensorKit's `split` builds them with its unchecked inner constructor
/// (`basic_manipulations.jl:30-52`).
pub fn split_fusion_tree_generic_checked<C>(
    rule: &C,
    tree: &FusionTreeKey,
    front_rank: usize,
) -> Result<(FusionTreeKey, FusionTreeKey), CheckedGenericStructureError<C::Error>>
where
    C: CheckedGenericFusion,
{
    let mut rows = TreeScratch::default();
    let (front, back) =
        split_fusion_tree_generic_checked_into(rule, tree.as_view(), front_rank, &mut rows)?;
    Ok(rows.into_keys(front, back))
}

/// [`split_fusion_tree_generic_checked`] into `rows`, as
/// [`split_fusion_tree_into`].
#[doc(hidden)]
pub fn split_fusion_tree_generic_checked_into<C>(
    rule: &C,
    tree: FusionTreeRef<'_>,
    front_rank: usize,
    rows: &mut TreeScratch,
) -> Result<(usize, usize), CheckedGenericStructureError<C::Error>>
where
    C: CheckedGenericFusion,
{
    validate_generic_fusion_tree_checked(rule, tree)?;
    let rank = tree.uncoupled().len();
    if front_rank > rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: front_rank,
        }
        .into());
    }
    Ok(push_split_rows(
        rule.vacuum(),
        tree,
        front_rank,
        rows.builder(),
    )?)
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

    SimpleK(rule)
        .multi_fmove_inv(lhs.coupled(), coupled, rhs, false)?
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
    let terms = multi_fmove_inv_surgery(&GenericK(rule), &(lhs.coupled(), false), coupled, rhs)?;
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
/// the multiplicity-free and Generic entry points, appending both halves to
/// `rows`; each caller owns its own validation order. `front_rank <= rank` is
/// checked by the caller.
fn push_split_rows(
    unit: SectorId,
    tree: FusionTreeRef<'_>,
    front_rank: usize,
    builder: &mut TreeArenaBuilder,
) -> Result<(usize, usize), CoreError> {
    let rank = tree.uncoupled().len();
    if front_rank == rank {
        let coupled = tree.coupled();
        let front = builder.push_view(tree);
        return Ok((
            front,
            builder.push_parts([coupled], coupled, [false], [], []),
        ));
    }
    let (uncoupled, is_dual) = (tree.uncoupled(), tree.is_dual());
    let (innerlines, vertices) = (tree.innerlines(), tree.vertices());

    if front_rank == 1 {
        let first = uncoupled[0];
        let front = builder.push_parts([first], first, [is_dual[0]], [], []);
        let back = builder.push_parts(
            uncoupled.iter().copied(),
            tree.coupled(),
            std::iter::once(false).chain(is_dual.iter().copied().skip(1)),
            innerlines.iter().copied(),
            vertices.iter().copied(),
        );
        return Ok((front, back));
    }

    if front_rank == 0 {
        if rank == 0 {
            return Err(CoreError::MalformedFusionTree {
                message: "split at zero requires a non-empty source fusion tree",
            });
        }
        let tail_innerlines = if rank >= 2 { innerlines.len() + 1 } else { 0 };
        let front = builder.push_parts([], unit, [], [], []);
        let back = builder.push_parts(
            std::iter::once(unit).chain(uncoupled.iter().copied()),
            tree.coupled(),
            std::iter::once(false).chain(is_dual.iter().copied()),
            (rank >= 2)
                .then_some(uncoupled[0])
                .into_iter()
                .chain(innerlines.iter().copied())
                .take(tail_innerlines),
            std::iter::once(MultiplicityIndex::ONE).chain(vertices.iter().copied()),
        );
        return Ok((front, back));
    }

    let intermediate = *innerlines
        .get(front_rank - 2)
        .ok_or(CoreError::MalformedFusionTree {
            message: "split requires the intermediate innerline",
        })?;
    let front = builder.push_parts(
        uncoupled[..front_rank].iter().copied(),
        intermediate,
        is_dual[..front_rank].iter().copied(),
        innerlines[..front_rank.saturating_sub(2)].iter().copied(),
        vertices[..front_rank - 1].iter().copied(),
    );
    let back = builder.push_parts(
        std::iter::once(intermediate).chain(uncoupled[front_rank..].iter().copied()),
        tree.coupled(),
        std::iter::once(false).chain(is_dual[front_rank..].iter().copied()),
        innerlines[front_rank - 1..].iter().copied(),
        vertices[front_rank - 1..].iter().copied(),
    );
    Ok((front, back))
}

/// Replace the first leaf of `tail` by `front` (TensorKit `join(f₁, f₂)`,
/// `basic_manipulations.jl:78`): the inverse tree surgery of
/// [`push_split_rows`], shared by the multiplicity-free and Generic
/// merges. The caller has checked the boundary leaf.
fn join_tree_structural(front: &FusionTreeKey, tail: &FusionTreeKey) -> FusionTreeKey {
    match (front.uncoupled().len(), tail.uncoupled().len()) {
        (1, _) => FusionTreeKey::new(
            tail.uncoupled().iter().copied(),
            tail.coupled(),
            std::iter::once(front.is_dual()[0]).chain(tail.is_dual().iter().copied().skip(1)),
            tail.innerlines().iter().copied(),
            tail.vertices().iter().copied(),
        ),
        (_, 1) => front.clone(),
        (_, _) => FusionTreeKey::new(
            front
                .uncoupled()
                .iter()
                .chain(&tail.uncoupled()[1..])
                .copied(),
            tail.coupled(),
            front.is_dual().iter().chain(&tail.is_dual()[1..]).copied(),
            front
                .innerlines()
                .iter()
                .copied()
                .chain(std::iter::once(front.coupled()))
                .chain(tail.innerlines().iter().copied()),
            front.vertices().iter().chain(tail.vertices()).copied(),
        ),
    }
}
