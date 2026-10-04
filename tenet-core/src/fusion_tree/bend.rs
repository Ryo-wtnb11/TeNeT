use super::*;

pub(crate) fn multiplicity_free_repartition_terms<R>(
    rule: &R,
    terms: Vec<(FusionTreePairKey, R::Scalar)>,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let current = terms;
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
    let codomain_rank = first_key.codomain_tree().uncoupled().len();
    repartition_loop(
        current,
        codomain_rank,
        target_codomain_rank,
        |terms, bend| multiplicity_free_bend_terms(rule, terms, bend),
    )
}

/// One repartition bend over a multiplicity-free term list.
pub(super) fn multiplicity_free_bend_terms<R>(
    rule: &R,
    terms: Vec<(FusionTreePairKey, R::Scalar)>,
    bend: Bend,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    compose_terms(terms, |key| match bend {
        Bend::Left => multiplicity_free_bendleft_tree_pair(rule, key),
        Bend::Right => multiplicity_free_bendright_tree_pair(rule, key),
    })
}

/// One multiplicity-free bend at a fixed frame. `bendleft` is `bendright`
/// through the swapped pair with a conjugated coefficient
/// (`duality_manipulations.jl:140-148`).
pub(super) struct MultiplicityFreeBend {
    site: BendSite,
    left: bool,
}

impl MultiplicityFreeBend {
    pub(super) fn prepare(
        frame: &MultiplicityFreeTreePairFrame,
        bend: Bend,
    ) -> Result<Self, CoreError> {
        let left = matches!(bend, Bend::Left);
        let (codomain, domain) = orient(left, &frame.codomain, &frame.domain);
        Ok(Self {
            site: BendSite::new(
                &codomain.uncoupled,
                &codomain.is_dual,
                domain.uncoupled.len(),
            )?,
            left,
        })
    }

    pub(super) fn validate_output_frame(&self) -> Result<(), CoreError> {
        self.site.bent_is_dual().map(drop)
    }

    /// The externals after the bend (`duality_manipulations.jl:40-51`).
    pub(super) fn output_frame<R: FusionRule>(
        &self,
        rule: &R,
        frame: &MultiplicityFreeTreePairFrame,
    ) -> Result<MultiplicityFreeTreePairFrame, CoreError> {
        let bent_is_dual = self.site.bent_is_dual()?;
        let (codomain, domain) = orient(self.left, &frame.codomain, &frame.domain);
        let kept = self.site.codomain_rank - 1;
        let shrunk = MultiplicityFreeTreeFrame::from_frozen_externals(
            codomain.uncoupled[..kept].into(),
            codomain.is_dual[..kept].into(),
        );
        let grown = MultiplicityFreeTreeFrame::from_frozen_externals(
            domain
                .uncoupled
                .iter()
                .copied()
                .chain(std::iter::once(rule.dual(self.site.bent_sector)))
                .collect(),
            domain
                .is_dual
                .iter()
                .copied()
                .chain(std::iter::once(!bent_is_dual))
                .collect(),
        );
        let (codomain, domain) = orient(self.left, shrunk, grown);
        Ok(MultiplicityFreeTreePairFrame { codomain, domain })
    }

    #[inline(always)]
    pub(super) fn apply<K>(
        &self,
        kernel: &K,
        local: &MultiplicityFreeTreePairLocal,
    ) -> Result<(MultiplicityFreeTreePairLocal, K::S), CoreError>
    where
        K: BendKernel<E = CoreError, Row = <K as StyleKernel>::S>,
    {
        let (codomain, domain) = orient(self.left, &local.codomain, &local.domain);
        let (lines, coefficient) = bend_surgery(kernel, &self.site, codomain, domain)?;
        let local = self.local(&lines, codomain, domain);
        Ok(if self.left {
            (local, coefficient.conj())
        } else {
            (local, coefficient)
        })
    }

    /// The bend's structure alone, for the source-major preflight.
    pub(super) fn next_local(
        &self,
        vacuum: SectorId,
        local: &MultiplicityFreeTreePairLocal,
    ) -> Result<MultiplicityFreeTreePairLocal, CoreError> {
        let (codomain, domain) = orient(self.left, &local.codomain, &local.domain);
        let lines = self.site.lines(vacuum, codomain, domain)?;
        Ok(self.local(&lines, codomain, domain))
    }

    /// Drop the last codomain innerline; the domain gains `c` as an innerline
    /// and both trees couple to `a` (`duality_manipulations.jl:40-51`).
    #[inline(always)]
    fn local(
        &self,
        lines: &BendLines,
        codomain: &MultiplicityFreeTreeLocal,
        domain: &MultiplicityFreeTreeLocal,
    ) -> MultiplicityFreeTreePairLocal {
        let kept: &[SectorId] = if self.site.codomain_rank > 2 {
            &codomain.innerlines[..codomain.innerlines.len() - 1]
        } else {
            &[]
        };
        let shrunk = MultiplicityFreeTreeLocal {
            coupled: lines.left_coupled,
            innerlines: kept.iter().copied().collect(),
        };
        let grown = MultiplicityFreeTreeLocal {
            coupled: lines.left_coupled,
            innerlines: domain
                .innerlines
                .iter()
                .copied()
                .chain((self.site.domain_rank > 1).then_some(lines.coupled))
                .collect(),
        };
        let (codomain, domain) = orient(self.left, shrunk, grown);
        MultiplicityFreeTreePairLocal { codomain, domain }
    }
}

#[inline(always)]
fn orient<T>(swap: bool, codomain: T, domain: T) -> (T, T) {
    if swap {
        (domain, codomain)
    } else {
        (codomain, domain)
    }
}

#[expect(
    clippy::type_complexity,
    reason = "the SmallVec inline capacity is part of this local bend allocation contract"
)]
pub(super) fn multiplicity_free_bend_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    bend: Bend,
) -> Result<SmallVec<[(FusionTreePairKey, R::Scalar); 1]>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let (frame, local) = project_multiplicity_free_tree_pair(rule, tree_pair)?;
    let prepared = MultiplicityFreeBend::prepare(&frame, bend)?;
    let (local, coefficient) = prepared.apply(&SimpleK(rule), &local)?;
    let frame = prepared.output_frame(rule, &frame)?;
    let mut terms = SmallVec::new();
    terms.push((frame.materialize(local), coefficient));
    Ok(terms)
}

#[expect(
    clippy::type_complexity,
    reason = "the SmallVec inline capacity is part of this local bend allocation contract"
)]
pub(crate) fn multiplicity_free_bendright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<SmallVec<[(FusionTreePairKey, R::Scalar); 1]>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    multiplicity_free_bend_tree_pair(rule, tree_pair, Bend::Right)
}

#[expect(
    clippy::type_complexity,
    reason = "the SmallVec inline capacity is part of this local bend allocation contract"
)]
pub(crate) fn multiplicity_free_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<SmallVec<[(FusionTreePairKey, R::Scalar); 1]>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    multiplicity_free_bend_tree_pair(rule, tree_pair, Bend::Left)
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
#[cfg(any(test, feature = "testing"))]
pub(crate) fn generic_bendright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGeneric::new(rule);
    generic_bendright_tree_pair_result(&checked, tree_pair)
        .map_err(map_infallible_generic_symbol_error)
}

#[cfg(test)]
pub(crate) fn generic_bendright_tree_pair_checked<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    validate_generic_fusion_tree_pair_checked(rule, tree_pair)
        .map_err(map_checked_generic_structure_error)?;
    generic_bendright_tree_pair_result(rule, tree_pair)
}

pub(super) fn generic_bendright_tree_pair_result<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    let codomain = tree_pair.codomain_tree();
    let domain = tree_pair.domain_tree();
    let site = BendSite::new(
        codomain.uncoupled(),
        codomain.is_dual(),
        domain.uncoupled().len(),
    )?;
    let (lines, row) = bend_surgery(&GenericK(rule), &site, codomain, domain)?;
    let codomain_rank = site.codomain_rank;
    let domain_rank = site.domain_rank;

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
        lines.left_coupled,
        codomain.is_dual()[..codomain_rank - 1].iter().copied(),
        new_codomain_innerlines.iter().copied(),
        new_codomain_vertices.iter().copied(),
    );

    // Base domain data shared by every ν; only the appended vertex label varies
    // (TK :100-103, `uncoupled₂/coupled₂/isdual₂/inner₂` hoisted out of the loop).
    let domain_uncoupled: Arc<[SectorId]> = domain
        .uncoupled()
        .iter()
        .copied()
        .chain(std::iter::once(row.bent_dual))
        .collect::<Vec<_>>()
        .into();
    let domain_is_dual: Arc<[bool]> = domain
        .is_dual()
        .iter()
        .copied()
        .chain(std::iter::once(!lines.bent_is_dual))
        .collect::<Vec<_>>()
        .into();
    let domain_innerlines: Arc<[SectorId]> = domain
        .innerlines()
        .iter()
        .copied()
        .chain((domain_rank > 1).then_some(lines.coupled))
        .collect::<Vec<_>>()
        .into();

    let mut out: Vec<(FusionTreePairKey, C::Scalar)> = Vec::new();
    for (nu0, coeff) in row.terms() {
        // vertices₂ = N₂>0 ? (f₂.vertices..., ν) : ()  (TK :107). ν is the
        // 1-based output vertex label (mu_index inverts this on the way back).
        let new_domain = FusionTreeKey::from_frozen(
            Arc::clone(&domain_uncoupled),
            lines.left_coupled,
            Arc::clone(&domain_is_dual),
            Arc::clone(&domain_innerlines),
            domain
                .vertices()
                .iter()
                .copied()
                .chain((domain_rank > 0).then(|| {
                    MultiplicityIndex::new(nu0 + 1)
                        .expect("enumerated Generic multiplicity labels are one-based")
                }))
                .collect::<Vec<_>>()
                .into(),
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
#[cfg(any(test, feature = "testing"))]
pub(crate) fn generic_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGeneric::new(rule);
    generic_bendleft_tree_pair_result(&checked, tree_pair)
        .map_err(map_infallible_generic_symbol_error)
}

#[cfg(test)]
pub(crate) fn generic_bendleft_tree_pair_checked<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    validate_generic_fusion_tree_pair_checked(rule, tree_pair)
        .map_err(map_checked_generic_structure_error)?;
    generic_bendleft_tree_pair_result(rule, tree_pair)
}

pub(super) fn generic_bendleft_tree_pair_result<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    left_move_by_swap(tree_pair, |swapped| {
        generic_bendright_tree_pair_result(rule, swapped)
    })
}

#[cfg(test)]
/// Generic-fusion `repartition`: bend legs between codomain and domain until the
/// codomain has `target_codomain_rank` legs. Verbatim mirror of TensorKit
/// `repartition` / `_repartition_body` (`duality_manipulations.jl:460-505`): the
/// generated function unrolls `|N|` `bendleft`/`bendright` steps and composes
/// their coefficient matrices (`U = Utmp * U`), which is exactly this
/// accumulate-and-compose loop. Structural twin of
/// [`multiplicity_free_repartition_tree_pair`] :794-827.
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(test)]
pub(crate) fn generic_repartition_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
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

#[cfg(test)]
fn generic_repartition_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree_pair.rule;
    let checked = InfallibleGeneric::new(rule);
    generic_repartition_tree_pair_result(&checked, tree_pair.key, target_codomain_rank)
        .map_err(map_infallible_generic_symbol_error)
}

#[cfg(any(test, feature = "testing"))]
pub(super) fn generic_repartition_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGeneric::new(rule);
    generic_repartition_tree_pair_result(&checked, tree_pair, target_codomain_rank)
        .map_err(map_infallible_generic_symbol_error)
}

/// Checked Generic-fusion repartition for a mutable fallible provider.
///
/// Structural validation completes before the first F, dimension, or pivotal
/// query, and provider failures retain their typed source.
#[expect(
    clippy::type_complexity,
    reason = "the public checked API exposes destination tree-pair coefficient rows directly"
)]
#[cfg(test)]
pub(crate) fn generic_repartition_tree_pair_checked<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, C::Scalar)>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    let total_rank =
        tree_pair.codomain_tree().uncoupled().len() + tree_pair.domain_tree().uncoupled().len();
    if target_codomain_rank > total_rank {
        return Err(CoreError::DimensionMismatch {
            expected: total_rank,
            actual: target_codomain_rank,
        }
        .into());
    }
    validate_generic_fusion_tree_pair_checked(rule, tree_pair)
        .map_err(map_checked_generic_structure_error)?;
    generic_repartition_tree_pair_result(rule, tree_pair, target_codomain_rank)
}

pub(super) fn generic_repartition_tree_pair_result<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    repartition_loop(
        vec![(tree_pair.clone(), C::Scalar::one())],
        tree_pair.codomain_tree().uncoupled().len(),
        target_codomain_rank,
        |terms, bend| {
            compose_terms(terms, |key| match bend {
                Bend::Left => generic_bendleft_tree_pair_result(rule, key),
                Bend::Right => generic_bendright_tree_pair_result(rule, key),
            })
        },
    )
}
