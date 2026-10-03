use super::*;

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
#[cfg(test)]
pub(crate) fn generic_artin_braid_at_with_inverse<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGenericFR(rule);
    generic_artin_braid_at_with_inverse_checked(&checked, tree, index, inverse)
        .map_err(map_infallible_generic_symbol_error)
}

pub(crate) fn generic_artin_braid_at_with_inverse_checked<C>(
    rule: &C,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Result<GenericTreeTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericFRAccess,
{
    let kernel = GenericK(rule);
    let site = ArtinSite::new(&kernel, tree.uncoupled(), index, inverse)?;
    let mut uncoupled = tree.uncoupled().to_vec();
    uncoupled.swap(index, index + 1);
    let mut is_dual = tree.is_dual().to_vec();
    is_dual.swap(index, index + 1);
    let mut out = KeyArtinTerms {
        tree,
        index,
        uncoupled: uncoupled.into(),
        is_dual: is_dual.into(),
        terms: Vec::new(),
    };
    artin_surgery(&kernel, &site, tree, &mut out)?;
    Ok(out.terms)
}

/// Generic Artin outputs as full keys carrying their vertex labels.
struct KeyArtinTerms<'t, S> {
    tree: &'t FusionTreeKey,
    index: usize,
    uncoupled: Arc<[SectorId]>,
    is_dual: Arc<[bool]>,
    terms: Vec<(FusionTreeKey, S)>,
}

impl<S, E> ArtinWriter<S, CheckedGenericSymbolError<E>> for KeyArtinTerms<'_, S> {
    type Slot = FusionTreeKey;

    fn reserve(&mut self, additional: usize) {
        self.terms.reserve_exact(additional);
    }

    fn begin(
        &mut self,
        innerline: Option<(usize, SectorId)>,
        vertices: ArtinVertices,
    ) -> Result<Self::Slot, CheckedGenericSymbolError<E>> {
        let tree = self.tree;
        let index = self.index;
        let innerlines: Arc<[SectorId]> = match innerline {
            None => Arc::clone(&tree.innerlines),
            Some((position, sector)) => {
                let mut innerlines: SectorVec = tree.innerlines().iter().copied().collect();
                *innerlines
                    .get_mut(position)
                    .ok_or(CoreError::MalformedFusionTree {
                        message: artin_innerline_message(vertices),
                    })? = sector;
                innerlines.into_vec().into()
            }
        };
        let label = |value: usize| {
            MultiplicityIndex::new(value + 1)
                .expect("enumerated Generic multiplicity labels are one-based")
        };
        let vertex_labels: Arc<[MultiplicityIndex]> = match vertices {
            ArtinVertices::Keep => Arc::clone(&tree.vertices),
            ArtinVertices::SwapUnit => {
                if tree.vertices().len() <= index {
                    return Err(CoreError::MalformedFusionTree {
                        message:
                            "unit braid past the first adjacent pair requires adjacent vertices",
                    }
                    .into());
                }
                let mut labels = tree.vertices().to_vec();
                labels.swap(index - 1, index);
                labels.into()
            }
            ArtinVertices::First { nu } => {
                let mut labels: MultiplicityVec = tree.vertices().iter().copied().collect();
                *labels.get_mut(0).ok_or(CoreError::MalformedFusionTree {
                    message: "first braid of a Generic tree requires a vertex",
                })? = label(nu);
                labels.into_vec().into()
            }
            ArtinVertices::Pair { sigma, lambda } => {
                let mut labels: MultiplicityVec = tree.vertices().iter().copied().collect();
                if labels.len() <= index {
                    return Err(CoreError::MalformedFusionTree {
                        message: "non-first Generic braid requires adjacent vertices",
                    }
                    .into());
                }
                labels[index - 1] = label(sigma);
                labels[index] = label(lambda);
                labels.into_vec().into()
            }
        };
        Ok(FusionTreeKey::from_frozen(
            Arc::clone(&self.uncoupled),
            tree.coupled(),
            Arc::clone(&self.is_dual),
            innerlines,
            vertex_labels,
        ))
    }

    fn finish(
        &mut self,
        key: Self::Slot,
        coefficient: S,
    ) -> Result<(), CheckedGenericSymbolError<E>> {
        self.terms.push((key, coefficient));
        Ok(())
    }
}

/// Braid the uncoupled legs of a Generic-fusion tree by `permutation` under the
/// given `levels`, the outer-multiplicity mirror of
/// [`multiplicity_free_braid_tree`] and of TensorKit's `braid(f, p, levels)`
/// swap-decomposition loop (`braiding_manipulations.jl:235-248`,
/// non-`SymmetricBraiding` branch). The permutation is decomposed into
/// neighbouring swaps; each swap is an `generic_artin_braid_at_with_inverse`
/// with `inverse = levels[s] > levels[s+1]` (:239), and the running level
/// tuple is swapped after each step (:243-244).
///
/// Because one input tree can fan out to several vertex-labelled outputs, the
/// coefficients are threaded through a `FusionTermAccumulator` (summing paths
/// that reconverge on the same output tree), exactly as the multiplicity-free
/// braid does.
// `pub` to mirror the mult-free split (`multiplicity_free_braid_tree` is `pub`,
// its per-swap artin helper private). Being a public root also keeps
// `generic_artin_braid_at_with_inverse` / `mu_index` reachable, so they emit no
// dead-code warning before Stage B2's recouple wrapper consumes them.
/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
#[cfg(test)]
pub(crate) fn generic_braid_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    levels: &[usize],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
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
    let steps = PreparedTreeBraid::new(permutation, levels, rank)?.artin_steps;
    let validated = validate_fusion_tree_for_rule(rule, tree)?;
    generic_braid_tree_validated(validated, permutation, &steps)
}

#[cfg(test)]
fn generic_braid_tree_validated<R>(
    tree: ValidatedFusionTree<'_, R>,
    permutation: &[usize],
    steps: &[PreparedArtinStep],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree.rule;
    generic_braid_tree_unchecked(rule, tree.key, permutation, steps)
}

pub(super) fn generic_braid_tree_unchecked<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    steps: &[PreparedArtinStep],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGenericFR(rule);
    generic_braid_tree_result(&checked, tree, permutation, steps)
        .map_err(map_infallible_generic_symbol_error)
}

pub(super) fn generic_braid_tree_result<C>(
    rule: &C,
    tree: &FusionTreeKey,
    permutation: &[usize],
    steps: &[PreparedArtinStep],
) -> Result<GenericTreeTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericFRAccess,
{
    let rank = tree.uncoupled().len();
    if permutation.iter().copied().eq(0..rank) {
        return Ok(vec![(tree.clone(), C::Scalar::one())]);
    }
    braid_tree_steps(tree, steps.iter().copied(), |tree, step| {
        generic_artin_braid_at_with_inverse_checked(rule, tree, step.index, step.inverse)
    })
}
