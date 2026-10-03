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
    // Entry gate: Generic-fusion only. `has_multiplicity()` is exactly the
    // `FusionStyle(I) isa GenericFusion` predicate TensorKit branches on
    // (braiding_manipulations.jl:137,170). Mult-free rules must use the
    // scalar-coefficient path instead.
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        }
        .into());
    }

    let rank = tree.uncoupled().len();
    if index + 1 >= rank {
        return Err(CoreError::InvalidBraidIndex { index, rank }.into());
    }

    // a, b = uncoupled[i], uncoupled[i+1]; swap them into uncoupled′, isdual′
    // (braiding_manipulations.jl:86-93). `left`/`right` keep the a/b naming for
    // the i == 1 special case; the i > 1 case renames below, matching TK's
    // "other naming convention" comment at :151.
    let left = tree.uncoupled()[index];
    let right = tree.uncoupled()[index + 1];
    let mut uncoupled = tree.uncoupled().to_vec();
    uncoupled.swap(index, index + 1);
    let mut is_dual = tree.is_dual().to_vec();
    is_dual.swap(index, index + 1);
    let uncoupled: Arc<[SectorId]> = uncoupled.into();
    let is_dual: Arc<[bool]> = is_dual.into();

    // Braiding with the trivial sector: simple and always possible, coefficient
    // 1, no F/R needed (braiding_manipulations.jl:101-120). Identical bookkeeping
    // to the mult-free branch; the vertices just carry OM labels now.
    if left == rule.vacuum() || right == rule.vacuum() {
        let (innerlines, vertices) = if index > 0 {
            let mut innerlines = tree.innerlines().to_vec();
            let mut vertices = tree.vertices().to_vec();
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
                }
                .into());
            }
            vertices.swap(index - 1, index);
            (Arc::from(innerlines), Arc::from(vertices))
        } else {
            (Arc::clone(&tree.innerlines), Arc::clone(&tree.vertices))
        };
        return Ok(vec![(
            FusionTreeKey::from_frozen(
                uncoupled,
                tree.coupled(),
                is_dual,
                innerlines,
                vertices,
            ),
            C::Scalar::one(),
        )]);
    }

    // NoBraiding rules cannot braid non-trivial sectors
    // (braiding_manipulations.jl:122-123).
    if !rule.braiding_style().has_braiding() {
        return Err(CoreError::UnsupportedSectorBraid {
            left,
            right,
            style: rule.braiding_style(),
        }
        .into());
    }

    if index == 0 {
        // c = N > 2 ? inner[1] : coupled′  (braiding_manipulations.jl:131)
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
        // GenericFusion i == 1 branch (braiding_manipulations.jl:137-148).
        // μ = vertices[1] (the single input vertex), 1-based label -> 0-based
        // matrix row (:138).
        let mu0 = mu_index(tree, 0)?;
        // Rmat = inv ? Rsymbol(b,a,c)' : Rsymbol(a,b,c)  (:139). We fetch the
        // *un-adjointed* base and take the adjoint at the element read below.
        let rmat = if inverse {
            checked_generic_r_symbol(rule, right, left, coupled)?
        } else {
            checked_generic_r_symbol(rule, left, right, coupled)?
        };
        // ν ranges over the output vertex = the columns of Rmat (:140). The
        // adjoint flips the shape back, so that count is N(b,a,c) either way.
        let n_nu = rule
            .try_nsymbol(right, left, coupled)
            .map_err(CheckedGenericSymbolError::Provider)?;
        let mut out = Vec::with_capacity(n_nu);
        for nu0 in 0..n_nu {
            // R = Rmat[μ, ν]  (:141). For the adjoint, Rmat[μ,ν] = conj(base[ν,μ]).
            let r = if inverse {
                rmat.get(nu0, mu0).conj()
            } else {
                rmat.get(mu0, nu0).clone()
            };
            if r.is_zero() {
                continue; // iszero(R) && continue  (:142)
            }
            // vertices′ = setindex(vertices, ν, 1)  (:143)
            let mut vertices: MultiplicityVec = tree.vertices().iter().copied().collect();
            *vertices
                .get_mut(0)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "first braid of a Generic tree requires a vertex",
                })? = MultiplicityIndex::new(nu0 + 1)
                .expect("enumerated Generic multiplicity labels are one-based");
            out.push((
                FusionTreeKey::from_frozen(
                    Arc::clone(&uncoupled),
                    tree.coupled(),
                    Arc::clone(&is_dual),
                    Arc::clone(&tree.innerlines),
                    vertices.into_vec().into(),
                ),
                r,
            ));
        }
        return Ok(out);
    }

    // case i > 1: other naming convention (braiding_manipulations.jl:151-156).
    // b = uncoupled[i]; d = uncoupled[i+1]; a = inner_ext[i-1]; c = inner_ext[i];
    // e = inner_ext[i+1].
    let a = inner_extended_sector(tree, index - 1)?;
    let b = left;
    let c = inner_extended_sector(tree, index)?;
    let d = right;
    let e = inner_extended_sector(tree, index + 1)?;
    // μ = vertices[i-1]; ν = vertices[i]  (:175-176), 1-based label -> 0-based.
    let mu0 = mu_index(tree, index - 1)?;
    let nu0 = mu_index(tree, index)?;

    let mut out = Vec::new();
    // for c′ in intersect(a ⊗ d, e ⊗ conj(b))  (:171). `c' ∈ a⊗d` filtered by
    // N(c',b,e) > 0 is exactly that intersection (N(c',b,e) > 0 ⟺ c' ∈ e⊗conj(b)),
    // the same rewrite the mult-free branch uses.
    //
    // `fusion_channels_in_table` (not `fusion_channels`): for a bounded table
    // bounded rule the pair (a, d) can escape even on a legal tree (e.g.
    // a=27, d=8). Skipped frontier c′ are provably dead: transforms only run
    // on structures whose sectors the coupled fold admitted as clean, and a
    // nonzero frontier-c′ term would be a complete tree through an
    // out-of-table inner line — contradicting cleanness.
    for c_prime in rule
        .try_fusion_channels_in_table(a, d)
        .map_err(CheckedGenericSymbolError::Provider)?
    {
        if rule
            .try_nsymbol(c_prime, b, e)
            .map_err(CheckedGenericSymbolError::Provider)?
            == 0
        {
            continue;
        }
        // Rmat1 = inv ? Rsymbol(d,c,e)' : Rsymbol(c,d,e)   (:172)
        // Rmat2 = inv ? Rsymbol(d,a,c')' : Rsymbol(a,d,c')  (:173)
        // Fmat = Fsymbol(d,a,b,e,c',c)                      (:174)
        let rmat1 = if inverse {
            checked_generic_r_symbol(rule, d, c, e)?
        } else {
            checked_generic_r_symbol(rule, c, d, e)?
        };
        let rmat2 = if inverse {
            checked_generic_r_symbol(rule, d, a, c_prime)?
        } else {
            checked_generic_r_symbol(rule, a, d, c_prime)?
        };
        let fmat = checked_generic_f_symbol(rule, d, a, b, e, c_prime, c)?;
        // Output vertex ranges σ ∈ 1:N(a,d,c'), λ ∈ 1:N(c',b,e)  (:177-178);
        // inner-sum ranges ρ ∈ 1:N(d,c,e), κ ∈ 1:N(d,a,c')  (:180).
        let n_sigma = rule.try_nsymbol(a, d, c_prime).map_err(CheckedGenericSymbolError::Provider)?;
        let n_lambda = rule.try_nsymbol(c_prime, b, e).map_err(CheckedGenericSymbolError::Provider)?;
        let n_rho = rule.try_nsymbol(d, c, e).map_err(CheckedGenericSymbolError::Provider)?;
        let n_kappa = rule.try_nsymbol(d, a, c_prime).map_err(CheckedGenericSymbolError::Provider)?;
        for sigma0 in 0..n_sigma {
            for lambda0 in 0..n_lambda {
                // coeff = zero(oneT)  (:179)
                let mut coeff = C::Scalar::zero();
                for rho0 in 0..n_rho {
                    for kappa0 in 0..n_kappa {
                        // coeff += Rmat1[ν,ρ] * conj(Fmat[κ,λ,μ,ρ]) * conj(Rmat2[σ,κ])
                        // (:181-182). Adjoint element reads (see the trait doc):
                        //   Rmat1[ν,ρ]      : base[ν,ρ]      | conj(base[ρ,ν])   (inv)
                        //   conj(Rmat2[σ,κ]): conj(base[σ,κ])| base[κ,σ]         (inv, double-conj cancels)
                        let r1 = if inverse {
                            rmat1.get(rho0, nu0).conj()
                        } else {
                            rmat1.get(nu0, rho0).clone()
                        };
                        let f_conj = fmat.get(kappa0, lambda0, mu0, rho0).conj();
                        let r2_conj = if inverse {
                            rmat2.get(kappa0, sigma0).clone()
                        } else {
                            rmat2.get(sigma0, kappa0).conj()
                        };
                        coeff = coeff + r1 * f_conj * r2_conj;
                    }
                }
                if coeff.is_zero() {
                    continue; // iszero(coeff) && continue  (:184)
                }
                // vertices′ = setindex(setindex(vertices, σ, i-1), λ, i)  (:185-186)
                // inner′ = setindex(inner, c′, i-1)  (:187)
                let mut innerlines: SectorVec = tree.innerlines().iter().copied().collect();
                *innerlines
                    .get_mut(index - 1)
                    .ok_or(CoreError::MalformedFusionTree {
                        message: "non-first braid requires an innerline to update",
                    })? = c_prime;
                let mut vertices: MultiplicityVec = tree.vertices().iter().copied().collect();
                if vertices.len() <= index {
                    return Err(CoreError::MalformedFusionTree {
                        message: "non-first Generic braid requires adjacent vertices",
                    }
                    .into());
                }
                vertices[index - 1] = MultiplicityIndex::new(sigma0 + 1)
                    .expect("enumerated Generic multiplicity labels are one-based");
                vertices[index] = MultiplicityIndex::new(lambda0 + 1)
                    .expect("enumerated Generic multiplicity labels are one-based");
                out.push((
                    FusionTreeKey::from_frozen(
                        Arc::clone(&uncoupled),
                        tree.coupled(),
                        Arc::clone(&is_dual),
                        innerlines.into_vec().into(),
                        vertices.into_vec().into(),
                    ),
                    coeff,
                ));
            }
        }
    }
    Ok(out)
}

/// Read the 0-based outer-multiplicity matrix index of the vertex at position
/// `vertex_index`. [`MultiplicityIndex`] stores the one-based categorical
/// label, and TensorKit's `Rmat[μ, ν]` / `Fmat[κ, λ, μ, ρ]` are
/// 1-based Julia indices, so the stored label maps to the 0-based Rust index by
/// subtracting one.
pub(super) fn mu_index(tree: &FusionTreeKey, vertex_index: usize) -> Result<usize, CoreError> {
    let label = tree
        .vertices()
        .get(vertex_index)
        .copied()
        .ok_or(CoreError::MalformedFusionTree {
            message: "Generic braid requires a vertex label at the braided position",
        })?
        .get();
    Ok(label - 1)
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
pub fn generic_braid_tree<R>(
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
    let swaps = permutation_to_adjacent_swaps(permutation, rank)?;
    let validated = validate_fusion_tree_for_rule(rule, tree)?;
    generic_braid_tree_validated(validated, permutation, levels, &swaps)
}

fn generic_braid_tree_validated<R>(
    tree: ValidatedFusionTree<'_, R>,
    permutation: &[usize],
    levels: &[usize],
    swaps: &[usize],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree.rule;
    generic_braid_tree_unchecked(rule, tree.key, permutation, levels, swaps)
}

pub(super) fn generic_braid_tree_unchecked<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    levels: &[usize],
    swaps: &[usize],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGenericFR(rule);
    generic_braid_tree_result(&checked, tree, permutation, levels, swaps)
        .map_err(map_infallible_generic_symbol_error)
}

pub(super) fn generic_braid_tree_result<C>(
    rule: &C,
    tree: &FusionTreeKey,
    permutation: &[usize],
    levels: &[usize],
    swaps: &[usize],
) -> Result<GenericTreeTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericFRAccess,
{
    let rank = tree.uncoupled().len();
    if permutation.iter().copied().eq(0..rank) {
        return Ok(vec![(tree.clone(), C::Scalar::one())]);
    }
    let mut current = vec![(tree.clone(), C::Scalar::one())];
    let mut current_levels = levels.to_vec();
    for &swap in swaps {
        let inverse = current_levels[swap] > current_levels[swap + 1];
        let mut next_terms = FusionTermAccumulator::new();
        for (tree, coefficient) in current {
            for (next_tree, step_coefficient) in
                generic_artin_braid_at_with_inverse_checked(rule, &tree, swap, inverse)?
            {
                next_terms.push(next_tree, coefficient.clone() * step_coefficient);
            }
        }
        current_levels.swap(swap, swap + 1);
        current = next_terms.into_vec();
    }
    Ok(current)
}
