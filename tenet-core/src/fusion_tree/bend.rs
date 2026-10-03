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
    let mut current = terms;
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
    let mut current_codomain_rank = first_key.codomain_tree().uncoupled().len();
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

pub(super) struct PreparedMultiplicityFreeBendRight {
    codomain_rank: usize,
    domain_rank: usize,
    codomain_first: SectorId,
    domain_nonempty: bool,
    bent_sector: SectorId,
    bent_is_dual: Option<bool>,
    output_codomain_uncoupled: SectorVec,
    output_codomain_is_dual: DualVec,
    output_domain_uncoupled_prefix: SectorVec,
    output_domain_is_dual_prefix: DualVec,
}

pub(super) struct ValidatedMultiplicityFreeBendRightLocal {
    pub(super) local: MultiplicityFreeTreePairLocal,
    coupled: SectorId,
    left_coupled: SectorId,
    bent_is_dual: bool,
}

impl PreparedMultiplicityFreeBendRight {
    pub(super) fn validate_output_frame(&self) -> Result<(), CoreError> {
        self.bent_is_dual
            .ok_or(CoreError::MalformedFusionTree {
                message: "codomain tree is missing a duality flag",
            })
            .map(|_| ())
    }

    pub(super) fn output_frame<R>(&self, rule: &R) -> Result<MultiplicityFreeTreePairFrame, CoreError>
    where
        R: FusionRule,
    {
        let bent_is_dual = self.bent_is_dual.ok_or(CoreError::MalformedFusionTree {
            message: "codomain tree is missing a duality flag",
        })?;
        let mut domain_uncoupled = self.output_domain_uncoupled_prefix.clone();
        domain_uncoupled.push(rule.dual(self.bent_sector));
        let mut domain_is_dual = self.output_domain_is_dual_prefix.clone();
        domain_is_dual.push(!bent_is_dual);
        Ok(MultiplicityFreeTreePairFrame {
            codomain: MultiplicityFreeTreeFrame::from_frozen_externals(
                self.output_codomain_uncoupled.clone().into_vec().into(),
                self.output_codomain_is_dual.clone().into_vec().into(),
            ),
            domain: MultiplicityFreeTreeFrame::from_frozen_externals(
                domain_uncoupled.into_vec().into(),
                domain_is_dual.into_vec().into(),
            ),
        })
    }

    pub(super) fn validate_local<R, C, D>(
        &self,
        rule: &R,
        codomain: &C,
        domain: &D,
    ) -> Result<ValidatedMultiplicityFreeBendRightLocal, CoreError>
    where
        R: FusionRule,
        C: MultiplicityFreeTreeLocalData + ?Sized,
        D: MultiplicityFreeTreeLocalData + ?Sized,
    {
        let coupled = codomain.coupled();
        if self.domain_nonempty {
            let domain_coupled = domain.coupled();
            if domain_coupled != coupled {
                return Err(CoreError::MalformedFusionTree {
                    message: "fusion tree pair requires matching coupled sectors",
                });
            }
        }

        let left_coupled = match self.codomain_rank {
            1 => rule.vacuum(),
            2 => self.codomain_first,
            _ => codomain.innerlines().last().copied().ok_or(
                CoreError::MalformedFusionTree {
                    message: "bendright requires the last codomain innerline",
                },
            )?,
        };
        let bent_is_dual = self.bent_is_dual.ok_or(CoreError::MalformedFusionTree {
            message: "codomain tree is missing a duality flag",
        })?;

        let cod_inner = codomain.innerlines();
        let new_codomain_innerlines: &[SectorId] = if self.codomain_rank > 2 {
            &cod_inner[..cod_inner.len() - 1]
        } else {
            &[]
        };
        Ok(ValidatedMultiplicityFreeBendRightLocal {
            local: MultiplicityFreeTreePairLocal {
                codomain: MultiplicityFreeTreeLocal {
                    coupled: left_coupled,
                    innerlines: new_codomain_innerlines.iter().copied().collect(),
                },
                domain: MultiplicityFreeTreeLocal {
                    coupled: left_coupled,
                    innerlines: domain
                        .innerlines()
                        .iter()
                        .copied()
                        .chain((self.domain_rank > 1).then_some(coupled))
                        .collect(),
                },
            },
            coupled,
            left_coupled,
            bent_is_dual,
        })
    }

    pub(super) fn coefficient<R>(
        &self,
        rule: &R,
        local: &ValidatedMultiplicityFreeBendRightLocal,
    ) -> R::Scalar
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        let mut coefficient = rule.sqrt_dim_scalar(local.coupled)
            * rule.inv_sqrt_dim_scalar(local.left_coupled)
            * rule.b_symbol_scalar(local.left_coupled, self.bent_sector, local.coupled);
        if local.bent_is_dual {
            coefficient = coefficient
                * rule
                    .frobenius_schur_phase_scalar(rule.dual(self.bent_sector))
                    .conj();
        }
        coefficient
    }
}

pub(super) fn prepare_multiplicity_free_bendright<R>(
    _rule: &R,
    frame: &MultiplicityFreeTreePairFrame,
) -> Result<PreparedMultiplicityFreeBendRight, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let codomain = &frame.codomain;
    let domain = &frame.domain;
    let codomain_rank = codomain.uncoupled.len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "bendright requires at least one codomain leg",
        });
    }

    let bent_sector = codomain.uncoupled[codomain_rank - 1];
    let bent_is_dual = codomain.is_dual.get(codomain_rank - 1).copied();
    let output_codomain_uncoupled = codomain.uncoupled[..codomain_rank - 1]
        .iter()
        .copied()
        .collect();
    let output_codomain_is_dual = codomain
        .is_dual
        .iter()
        .copied()
        .take(codomain_rank - 1)
        .collect();
    let domain_rank = domain.uncoupled.len();
    Ok(PreparedMultiplicityFreeBendRight {
        codomain_rank,
        domain_rank,
        codomain_first: codomain.uncoupled[0],
        domain_nonempty: domain_rank != 0,
        bent_sector,
        bent_is_dual,
        output_codomain_uncoupled,
        output_codomain_is_dual,
        output_domain_uncoupled_prefix: domain.uncoupled.iter().copied().collect(),
        output_domain_is_dual_prefix: domain.is_dual.iter().copied().collect(),
    })
}

pub(super) struct PreparedMultiplicityFreeBendLeft {
    right: PreparedMultiplicityFreeBendRight,
}

impl PreparedMultiplicityFreeBendLeft {
    pub(super) fn validate_output_frame(&self) -> Result<(), CoreError> {
        self.right.validate_output_frame()
    }

    pub(super) fn output_frame<R>(&self, rule: &R) -> Result<MultiplicityFreeTreePairFrame, CoreError>
    where
        R: FusionRule,
    {
        let frame = self.right.output_frame(rule)?;
        Ok(MultiplicityFreeTreePairFrame {
            codomain: frame.domain,
            domain: frame.codomain,
        })
    }

    pub(super) fn validate_local<R, C, D>(
        &self,
        rule: &R,
        codomain: &C,
        domain: &D,
    ) -> Result<ValidatedMultiplicityFreeBendRightLocal, CoreError>
    where
        R: FusionRule,
        C: MultiplicityFreeTreeLocalData + ?Sized,
        D: MultiplicityFreeTreeLocalData + ?Sized,
    {
        self.right.validate_local(rule, domain, codomain)
    }

    pub(super) fn finish_local<R>(
        &self,
        rule: &R,
        validated: ValidatedMultiplicityFreeBendRightLocal,
    ) -> (MultiplicityFreeTreePairLocal, R::Scalar)
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        let coefficient = (self.right.coefficient(rule, &validated)).conj();
        (Self::finish_local_structure(validated), coefficient)
    }

    pub(super) fn finish_local_structure(
        validated: ValidatedMultiplicityFreeBendRightLocal,
    ) -> MultiplicityFreeTreePairLocal {
        MultiplicityFreeTreePairLocal {
            codomain: validated.local.domain,
            domain: validated.local.codomain,
        }
    }
}

pub(super) fn prepare_multiplicity_free_bendleft<R>(
    rule: &R,
    frame: &MultiplicityFreeTreePairFrame,
) -> Result<PreparedMultiplicityFreeBendLeft, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let swapped = MultiplicityFreeTreePairFrame {
        codomain: frame.domain.clone(),
        domain: frame.codomain.clone(),
    };
    Ok(PreparedMultiplicityFreeBendLeft {
        right: prepare_multiplicity_free_bendright(rule, &swapped)?,
    })
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
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    // Why not duplicate bend surgery in the future block runner: duality and
    // Frobenius-Schur phases must stay identical to the per-source operation.
    let (frame, local) = project_multiplicity_free_tree_pair(rule, tree_pair)?;
    let prepared = prepare_multiplicity_free_bendright(rule, &frame)?;
    let validated = prepared.validate_local(rule, &local.codomain, &local.domain)?;
    let frame = prepared.output_frame(rule)?;
    let coefficient = prepared.coefficient(rule, &validated);
    let mut terms = SmallVec::new();
    terms.push((frame.materialize(validated.local), coefficient));
    Ok(terms)
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
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let (frame, local) = project_multiplicity_free_tree_pair(rule, tree_pair)?;
    let prepared = prepare_multiplicity_free_bendleft(rule, &frame)?;
    let validated = prepared.validate_local(rule, &local.codomain, &local.domain)?;
    let frame = prepared.output_frame(rule)?;
    let (local, coefficient) = prepared.finish_local(rule, validated);
    let mut terms = SmallVec::new();
    terms.push((frame.materialize(local), coefficient));
    Ok(terms)
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
pub(crate) fn generic_bendright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGenericRigid(rule);
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
    let codomain_rank = codomain.uncoupled().len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "bendright requires at least one codomain leg",
        }
        .into());
    }

    let coupled = codomain.coupled();
    if !domain.uncoupled().is_empty() {
        let domain_coupled = domain.coupled();
        if domain_coupled != coupled {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion tree pair requires matching coupled sectors",
            }
            .into());
        }
    }

    // a = N₁==1 ? unit : N₁==2 ? uncoupled[1] : innerlines[end]  (TK :37).
    let left_coupled = match codomain_rank {
        1 => rule.vacuum(),
        2 => codomain.uncoupled()[0],
        _ => codomain
            .innerlines()
            .last()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "bendright requires the last codomain innerline",
            })?,
    };
    // b = uncoupled[N₁]  (TK :38).
    let bent_sector = codomain.uncoupled()[codomain_rank - 1];
    let bent_is_dual = codomain.is_dual().get(codomain_rank - 1).copied().ok_or(
        CoreError::MalformedFusionTree {
            message: "codomain tree is missing a duality flag",
        },
    )?;
    let domain_bent_sector = rule
        .try_dual(bent_sector)
        .map_err(CheckedGenericSymbolError::Provider)?;

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
        left_coupled,
        codomain.is_dual()[..codomain_rank - 1].iter().copied(),
        new_codomain_innerlines.iter().copied(),
        new_codomain_vertices.iter().copied(),
    );

    let domain_rank = domain.uncoupled().len();
    // Base domain data shared by every ν; only the appended vertex label varies
    // (TK :100-103, `uncoupled₂/coupled₂/isdual₂/inner₂` hoisted out of the loop).
    let domain_uncoupled: Arc<[SectorId]> = domain
        .uncoupled()
        .iter()
        .copied()
        .chain(std::iter::once(domain_bent_sector))
        .collect::<Vec<_>>()
        .into();
    let domain_is_dual: Arc<[bool]> = domain
        .is_dual()
        .iter()
        .copied()
        .chain(std::iter::once(!bent_is_dual))
        .collect::<Vec<_>>()
        .into();
    let domain_innerlines: Arc<[SectorId]> = domain
        .innerlines()
        .iter()
        .copied()
        .chain((domain_rank > 1).then_some(coupled))
        .collect::<Vec<_>>()
        .into();

    // coeff₀ = √dim(c)·(1/√dim(a)); ·conj(κ_{dual(b)}) if the bent leg is dual
    // (TK :89-92, same placement as the mult-free bend :2424-2429).
    let mut coeff0 = rule
        .try_sqrt_dim_scalar(coupled)
        .map_err(CheckedGenericSymbolError::Provider)?
        * rule
            .try_inv_sqrt_dim_scalar(left_coupled)
            .map_err(CheckedGenericSymbolError::Provider)?;
    if bent_is_dual {
        let dual_bent_sector = rule
            .try_dual(bent_sector)
            .map_err(CheckedGenericSymbolError::Provider)?;
        coeff0 = coeff0
            * rule
                .try_frobenius_schur_phase_scalar(dual_bent_sector)
                .map_err(CheckedGenericSymbolError::Provider)?
                .conj();
    }

    // Bmat = Bsymbol(a, b, c)  (TK :98); μ = N₁>1 ? vertices[end] : 1  (TK :99).
    let bmat = rule.try_b_symbol_generic(left_coupled, bent_sector, coupled)?;
    let mu0 = if codomain_rank > 1 {
        mu_index(codomain, codomain_rank - 2)?
    } else {
        0
    };

    let (_, cols) = bmat.shape();
    if mu0 >= bmat.shape().0 {
        return Err(CheckedGenericSymbolError::Shape {
            symbol: "B",
            expected: vec![mu0 + 1, cols],
            actual: vec![bmat.shape().0, cols],
        });
    }
    let mut out: Vec<(FusionTreePairKey, C::Scalar)> = Vec::new();
    for nu0 in 0..cols {
        // coeff = coeff₀ · Bmat[μ, ν]  (TK :105); iszero → skip  (TK :106).
        let coeff = coeff0.clone() * bmat.get(mu0, nu0).clone();
        if coeff.is_zero() {
            continue;
        }
        // vertices₂ = N₂>0 ? (f₂.vertices..., ν) : ()  (TK :107). ν is the
        // 1-based output vertex label (mu_index inverts this on the way back).
        let new_domain = FusionTreeKey::from_frozen(
            Arc::clone(&domain_uncoupled),
            left_coupled,
            Arc::clone(&domain_is_dual),
            Arc::clone(&domain_innerlines),
            domain
                .vertices()
                .iter()
                .copied()
                .chain(
                    (domain_rank > 0).then(|| {
                        MultiplicityIndex::new(nu0 + 1)
                            .expect("enumerated Generic multiplicity labels are one-based")
                    }),
                )
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
pub(crate) fn generic_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGenericRigid(rule);
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
    let swapped = FusionTreePairKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(generic_bendright_tree_pair_result(rule, &swapped)?
        .into_iter()
        .map(|(bent, coefficient)| {
            (
                FusionTreePairKey::pair(bent.domain_tree().clone(), bent.codomain_tree().clone()),
                coefficient.conj(),
            )
        })
        .collect())
}

#[cfg(test)]
pub(crate) fn compose_generic_tree_pair_terms<R, F, I>(
    rule: &R,
    terms: Vec<(FusionTreePairKey, R::Scalar)>,
    mut transform: F,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
    F: FnMut(&R, &FusionTreePairKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreePairKey, R::Scalar)>,
{
    compose_generic_tree_pair_terms_result(terms, |key| transform(rule, key))
}

pub(super) fn compose_generic_tree_pair_terms_result<S, E, F, I>(
    terms: Vec<(FusionTreePairKey, S)>,
    mut transform: F,
) -> Result<Vec<(FusionTreePairKey, S)>, E>
where
    S: CategoricalScalar,
    F: FnMut(&FusionTreePairKey) -> Result<I, E>,
    I: IntoIterator<Item = (FusionTreePairKey, S)>,
{
    let mut output = FusionTermAccumulator::new();
    for (key, coefficient) in terms {
        for (next_key, next_coefficient) in transform(&key)? {
            output.push(next_key, coefficient.clone() * next_coefficient);
        }
    }
    Ok(output.into_vec())
}

/// Generic-fusion `repartition`: bend legs between codomain and domain until the
/// codomain has `target_codomain_rank` legs. Verbatim mirror of TensorKit
/// `repartition` / `_repartition_body` (`duality_manipulations.jl:460-505`): the
/// generated function unrolls `|N|` `bendleft`/`bendright` steps and composes
/// their coefficient matrices (`U = Utmp * U`), which is exactly this
/// accumulate-and-compose loop. Structural twin of
/// [`multiplicity_free_repartition_tree_pair`] :794-827.
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
pub fn generic_repartition_tree_pair<R>(
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

fn generic_repartition_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree_pair.rule;
    let checked = InfallibleGenericRigid(rule);
    generic_repartition_tree_pair_result(&checked, tree_pair.key, target_codomain_rank)
        .map_err(map_infallible_generic_symbol_error)
}

pub(super) fn generic_repartition_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGenericRigid(rule);
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
pub fn generic_repartition_tree_pair_checked<C>(
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
    let mut current = vec![(tree_pair.clone(), C::Scalar::one())];
    let mut current_codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    // N = numout - target > 0 ⇒ bendright; < 0 ⇒ bendleft (TK :492).
    while current_codomain_rank < target_codomain_rank {
        current = compose_generic_tree_pair_terms_result(current, |key| {
            generic_bendleft_tree_pair_result(rule, key)
        })?;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        current = compose_generic_tree_pair_terms_result(current, |key| {
            generic_bendright_tree_pair_result(rule, key)
        })?;
        current_codomain_rank -= 1;
    }
    Ok(current)
}
