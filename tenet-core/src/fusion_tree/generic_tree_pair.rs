use super::*;

/// Generic-fusion `foldright`: bend the first codomain vertex `a ⊗ b ← c` to a
/// domain vertex `b ← dual(a) ⊗ c`. Verbatim mirror of TensorKit `foldright`
/// GenericFusion branch (`duality_manipulations.jl:238-289`), especially the
/// coefficient-vector × A × coefficient-vector contraction at `:277-284`:
///   `coeff₀ · (coeff₂' · (transpose(A) · coeff₁))`.
/// Structural twin of `multiplicity_free_foldright_tree_pair`, with the scalar
/// `coeff₁ · A · conj(coeff₂)` promoted to the vector–matrix–vector contraction
/// through the A-move matrix (which connects the two topmost `λ` vertices).
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(test)]
pub(crate) fn generic_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
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

#[cfg(test)]
fn generic_foldright_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
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
    R::Scalar: CategoricalScalar,
{
    let access = InfallibleGeneric::new(rule);
    generic_foldright_tree_pair_with(
        &access,
        tree_pair,
        |tree| generic_multi_fmove_tree(rule, tree).map_err(CheckedGenericSymbolError::Core),
        |leading, coupled, tree, leading_is_dual| {
            generic_multi_fmove_inv_tree(rule, leading, coupled, tree, leading_is_dual)
                .map_err(CheckedGenericSymbolError::Core)
        },
    )
    .map_err(map_infallible_generic_symbol_error)
}

/// Generic-fusion `foldleft` = swap + conjugate of `foldright`, verbatim mirror
/// of TensorKit `foldleft((f₁,f₂))` (`duality_manipulations.jl:315-319`).
/// Structural twin of `multiplicity_free_foldleft_tree_pair`.
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(test)]
pub(crate) fn generic_foldleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
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

#[cfg(test)]
fn generic_foldleft_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
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
    R::Scalar: CategoricalScalar,
{
    generic_foldleft_tree_pair_with(tree_pair, |key| {
        generic_foldright_tree_pair_unchecked(rule, key)
    })
}

/// Generic-fusion `cycleclockwise` = foldright ∘ bendleft (or the reverse order
/// when the codomain is empty), composing coefficient matrices. Verbatim mirror
/// of TensorKit `cycleclockwise` (`duality_manipulations.jl:401-410`) and
/// structural twin of `multiplicity_free_cycle_clockwise_tree_pair`.
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(test)]
pub(crate) fn generic_cycle_clockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
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

#[cfg(test)]
fn generic_cycle_clockwise_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree_pair.rule;
    generic_cycle_clockwise_tree_pair_unchecked(rule, tree_pair.key)
}

pub(super) fn generic_cycle_clockwise_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    generic_cycle_clockwise_tree_pair_with(
        tree_pair,
        |key| generic_bendleft_tree_pair(rule, key),
        |key| generic_foldright_tree_pair_unchecked(rule, key),
    )
}

/// Generic-fusion `cycleanticlockwise` = foldleft ∘ bendright (or the reverse
/// order when the domain is empty). Verbatim mirror of TensorKit
/// `cycleanticlockwise` (`duality_manipulations.jl:431-440`) and structural
/// twin of `multiplicity_free_cycle_anticlockwise_tree_pair`.
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(test)]
pub(crate) fn generic_cycle_anticlockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
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

#[cfg(test)]
fn generic_cycle_anticlockwise_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree_pair.rule;
    generic_cycle_anticlockwise_tree_pair_unchecked(rule, tree_pair.key)
}

pub(super) fn generic_cycle_anticlockwise_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    generic_cycle_anticlockwise_tree_pair_with(
        tree_pair,
        |key| generic_bendright_tree_pair(rule, key),
        |key| generic_foldleft_tree_pair_unchecked(rule, key),
    )
}

fn generic_foldright_tree_pair_result<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    generic_foldright_tree_pair_with(
        rule,
        tree_pair,
        |tree| generic_multi_fmove_tree_checked(rule, tree),
        |leading, coupled, tree, leading_is_dual| {
            generic_multi_fmove_inv_tree_checked(rule, leading, coupled, tree, leading_is_dual)
        },
    )
}

fn generic_foldright_tree_pair_with<C, F, I>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    mut forward: F,
    mut inverse: I,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
    F: FnMut(
        &FusionTreeKey,
    ) -> Result<GenericFmoveTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>,
    I: FnMut(
        SectorId,
        SectorId,
        &FusionTreeKey,
        bool,
    ) -> Result<GenericFmoveTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>,
{
    let codomain = tree_pair.codomain_tree();
    let Some(&a) = codomain.uncoupled().first() else {
        return Err(CoreError::MalformedFusionTree {
            message: "foldright requires at least one codomain leg",
        }
        .into());
    };
    let is_dual_a = codomain
        .is_dual()
        .first()
        .copied()
        .ok_or(CoreError::MalformedFusionTree {
            message: "codomain tree is missing the first duality flag",
        })?;
    let kappa = rule
        .try_frobenius_schur_phase_scalar(a)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let coupled = codomain.coupled();
    let dual_a = rule
        .try_dual(a)
        .map_err(CheckedGenericSymbolError::Provider)?;

    let mut terms = FusionTermAccumulator::new();
    for (codomain_prime, coeff1) in forward(codomain)? {
        let b = codomain_prime.coupled();
        let a_matrix = <C as GenericRigidAccess>::try_a_symbol_generic(rule, a, b, coupled)?;
        let (rows, cols) = a_matrix.shape();
        let coeff0 = rule
            .try_sqrt_dim_scalar(coupled)
            .map_err(CheckedGenericSymbolError::Provider)?
            * rule
                .try_inv_sqrt_dim_scalar(b)
                .map_err(CheckedGenericSymbolError::Provider)?;
        for (domain_prime, coeff2) in inverse(dual_a, b, tree_pair.domain_tree(), !is_dual_a)? {
            if coeff1.len() != rows || coeff2.len() != cols {
                return Err(CoreError::MalformedFusionTree {
                    message: "foldright: coefficient-vector length disagrees with A-matrix shape",
                }
                .into());
            }
            let mut inner = C::Scalar::zero();
            for (j, coeff2_j) in coeff2.iter().enumerate() {
                let mut column = C::Scalar::zero();
                for (i, coeff1_i) in coeff1.iter().enumerate() {
                    column = column + a_matrix.get(i, j).clone() * coeff1_i.clone();
                }
                inner = inner + coeff2_j.conj() * column;
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

fn generic_foldleft_tree_pair_result<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    generic_foldleft_tree_pair_with(tree_pair, |key| {
        generic_foldright_tree_pair_result(rule, key)
    })
}

fn generic_foldleft_tree_pair_with<S, E, F>(
    tree_pair: &FusionTreePairKey,
    mut foldright: F,
) -> Result<GenericTreePairTerms<S>, E>
where
    S: CategoricalScalar,
    F: FnMut(&FusionTreePairKey) -> Result<GenericTreePairTerms<S>, E>,
{
    let swapped = FusionTreePairKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(foldright(&swapped)?
        .into_iter()
        .map(|(folded, coefficient)| {
            (
                FusionTreePairKey::pair(
                    folded.domain_tree().clone(),
                    folded.codomain_tree().clone(),
                ),
                coefficient.conj(),
            )
        })
        .collect())
}

fn generic_cycle_clockwise_tree_pair_result<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    generic_cycle_clockwise_tree_pair_with(
        tree_pair,
        |key| generic_bendleft_tree_pair_result(rule, key),
        |key| generic_foldright_tree_pair_result(rule, key),
    )
}

fn generic_cycle_clockwise_tree_pair_with<S, E, B, F>(
    tree_pair: &FusionTreePairKey,
    mut bendleft: B,
    mut foldright: F,
) -> Result<GenericTreePairTerms<S>, E>
where
    S: CategoricalScalar,
    B: FnMut(&FusionTreePairKey) -> Result<GenericTreePairTerms<S>, E>,
    F: FnMut(&FusionTreePairKey) -> Result<GenericTreePairTerms<S>, E>,
{
    if tree_pair.codomain_tree().uncoupled().is_empty() {
        let first = bendleft(tree_pair)?;
        compose_generic_tree_pair_terms_result(first, foldright)
    } else {
        let first = foldright(tree_pair)?;
        compose_generic_tree_pair_terms_result(first, bendleft)
    }
}

fn generic_cycle_anticlockwise_tree_pair_result<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    generic_cycle_anticlockwise_tree_pair_with(
        tree_pair,
        |key| generic_bendright_tree_pair_result(rule, key),
        |key| generic_foldleft_tree_pair_result(rule, key),
    )
}

fn generic_cycle_anticlockwise_tree_pair_with<S, E, B, F>(
    tree_pair: &FusionTreePairKey,
    mut bendright: B,
    mut foldleft: F,
) -> Result<GenericTreePairTerms<S>, E>
where
    S: CategoricalScalar,
    B: FnMut(&FusionTreePairKey) -> Result<GenericTreePairTerms<S>, E>,
    F: FnMut(&FusionTreePairKey) -> Result<GenericTreePairTerms<S>, E>,
{
    if tree_pair.domain_tree().uncoupled().is_empty() {
        let first = bendright(tree_pair)?;
        compose_generic_tree_pair_terms_result(first, foldleft)
    } else {
        let first = foldleft(tree_pair)?;
        compose_generic_tree_pair_terms_result(first, bendright)
    }
}

/// Generic-fusion `braid` on a full tree pair: bend everything into the codomain,
/// braid there, bend back — the TensorKit `braid`/`fsbraid` decomposition.
/// Structural twin of [`multiplicity_free_braid_tree_pair`] (:829): the only
/// difference is the primitive family (`generic_repartition_tree_pair` /
/// `generic_braid_tree`) and the `one` seed;
/// no new recoupling formula is introduced.
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn generic_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
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
    R::Scalar: CategoricalScalar,
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
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGeneric::new(tree_pair.rule);
    generic_braid_tree_pair_result(
        &checked,
        tree_pair.key,
        target_codomain_rank,
        permutation,
        levels,
        swaps,
        identity,
    )
    .map_err(map_infallible_generic_symbol_error)
}

fn generic_braid_tree_pair_result<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
    permutation: &[usize],
    levels: &[usize],
    swaps: &[usize],
    identity: bool,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    if identity {
        return Ok(vec![(tree_pair.clone(), C::Scalar::one())]);
    }
    let all_rank = permutation.len();
    let mut current = generic_repartition_tree_pair_result(rule, tree_pair, all_rank)?;
    current = compose_generic_tree_pair_terms_result(current, |key| {
        generic_braid_tree_result(rule, key.codomain_tree(), permutation, levels, swaps).map(
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
    compose_generic_tree_pair_terms_result(current, |key| {
        generic_repartition_tree_pair_result(rule, key, target_codomain_rank)
    })
}

/// Checked Generic-fusion braid on a full tree pair.
#[expect(
    clippy::type_complexity,
    reason = "the public checked API exposes destination tree-pair coefficient rows directly"
)]
pub fn generic_braid_tree_pair_checked<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreePairKey, C::Scalar)>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    if codomain_levels.len() != codomain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: codomain_rank,
            actual: codomain_levels.len(),
        }
        .into());
    }
    if domain_levels.len() != domain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: domain_rank,
            actual: domain_levels.len(),
        }
        .into());
    }
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        }
        .into());
    }
    validate_generic_fusion_tree_pair_checked(rule, tree_pair)
        .map_err(map_checked_generic_structure_error)?;
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
    generic_braid_tree_pair_result(
        rule,
        tree_pair,
        codomain_permutation.len(),
        &permutation,
        &levels,
        &swaps,
        identity,
    )
}

/// Generic-fusion `permute` = [`generic_braid_tree_pair`] with the identity
/// level order (symmetric braiding only). Structural twin of
/// [`multiplicity_free_permute_tree_pair`] (:886).
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn generic_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
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

/// Checked Generic-fusion permutation on a full tree pair.
#[expect(
    clippy::type_complexity,
    reason = "the public checked API exposes destination tree-pair coefficient rows directly"
)]
pub fn generic_permute_tree_pair_checked<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, C::Scalar)>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        }
        .into());
    }
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    generic_braid_tree_pair_checked(
        rule,
        tree_pair,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

pub(crate) fn generic_permute_tree_pair_proven<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
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
/// twin of `multiplicity_free_transpose_tree_pair` (:916); braid-free, so it
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
    R::Scalar: CategoricalScalar,
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
    R::Scalar: CategoricalScalar,
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
    R::Scalar: CategoricalScalar,
{
    let rule = tree_pair.rule;
    generic_transpose_tree_pair_with(
        tree_pair.key,
        target_codomain_rank,
        total_rank,
        position,
        |key, rank| generic_repartition_tree_pair_unchecked(rule, key, rank),
        |key| generic_cycle_anticlockwise_tree_pair_unchecked(rule, key),
        |key| generic_cycle_clockwise_tree_pair_unchecked(rule, key),
    )
}

/// Checked Generic-fusion planar cyclic transpose.
///
/// Axis syntax and the complete source tree are validated before rigidity or
/// F-symbol queries; provider and symbol-shape failures remain typed.
pub fn generic_transpose_tree_pair_checked<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
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
        }
        .into());
    }
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        }
        .into());
    }
    validate_generic_fusion_tree_pair_checked(rule, tree_pair)
        .map_err(map_checked_generic_structure_error)?;

    generic_transpose_tree_pair_with(
        tree_pair,
        codomain_permutation.len(),
        codomain_rank + domain_rank,
        permutation.iter().position(|&axis| axis == 0),
        |key, rank| generic_repartition_tree_pair_result(rule, key, rank),
        |key| generic_cycle_anticlockwise_tree_pair_result(rule, key),
        |key| generic_cycle_clockwise_tree_pair_result(rule, key),
    )
}

fn generic_transpose_tree_pair_with<S, E, P, A, C>(
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
    total_rank: usize,
    position: Option<usize>,
    mut repartition: P,
    mut anticlockwise: A,
    mut clockwise: C,
) -> Result<Vec<(FusionTreePairKey, S)>, E>
where
    S: CategoricalScalar,
    P: FnMut(&FusionTreePairKey, usize) -> Result<Vec<(FusionTreePairKey, S)>, E>,
    A: FnMut(&FusionTreePairKey) -> Result<Vec<(FusionTreePairKey, S)>, E>,
    C: FnMut(&FusionTreePairKey) -> Result<Vec<(FusionTreePairKey, S)>, E>,
{
    let Some(mut position) = position else {
        return Ok(vec![(tree_pair.clone(), S::one())]);
    };
    let mut current = repartition(tree_pair, target_codomain_rank)?;
    if total_rank == 0 || position == 0 {
        return Ok(current);
    }
    let half_rank = total_rank >> 1;
    while position > 0 && position < half_rank {
        current = compose_generic_tree_pair_terms_result(current, &mut anticlockwise)?;
        position -= 1;
    }
    while position >= half_rank && position > 0 {
        current = compose_generic_tree_pair_terms_result(current, &mut clockwise)?;
        position = (position + 1) % total_rank;
    }
    Ok(current)
}
