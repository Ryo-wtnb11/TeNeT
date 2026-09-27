fn apply_first_compact_block_terms<R, K, F, I>(
    rule: &R,
    basis: &[K],
    mut transform: F,
) -> Result<(Vec<K>, DenseColumns<R::Scalar>), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar>,
    K: Eq + Hash,
    F: FnMut(&R, &K) -> Result<I, CoreError>,
    I: IntoIterator<Item = (K, R::Scalar)>,
{
    let mut index: FxHashMap<K, usize> = FxHashMap::default();
    let mut columns = DenseColumns::with_capacity(basis.len(), basis.len());
    for (source, source_local) in basis.iter().enumerate() {
        for (destination_local, coefficient) in transform(rule, source_local)? {
            let row = match index.get(&destination_local) {
                Some(&row) => row,
                None => {
                    let row = columns.push_empty_row();
                    index.insert(destination_local, row);
                    row
                }
            };
            let destination = &mut columns.row_mut(row)[source];
            *destination = Some(match destination.take() {
                Some(existing) => existing + coefficient,
                None => coefficient,
            });
        }
    }
    let mut slots: Vec<Option<K>> = (0..index.len()).map(|_| None).collect();
    for (local, row) in index {
        slots[row] = Some(local);
    }
    let locals = slots
        .into_iter()
        .map(|local| local.expect("dense rows 0..len are all filled"))
        .collect();
    Ok((locals, columns))
}

fn compact_artin_tree_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreeBasis,
    index: usize,
    inverse: bool,
) -> Result<
    (
        CompactMultiplicityFreeTreeBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_artin(rule, &basis.frame, index, inverse)?;
    let (locals, columns) =
        apply_first_compact_block_terms(rule, &basis.locals, |rule, local| {
            prepared.apply(rule, local)
        })?;
    Ok((
        CompactMultiplicityFreeTreeBasis {
            frame: prepared.output_frame,
            locals,
        },
        columns,
    ))
}

fn compact_artin_tree_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreeBasis,
    columns: &DenseColumns<R::Scalar>,
    index: usize,
    inverse: bool,
) -> Result<
    (
        CompactMultiplicityFreeTreeBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_artin(rule, &basis.frame, index, inverse)?;
    let (locals, next_columns) =
        compose_compact_block_terms(rule, &basis.locals, columns, |rule, local| {
            prepared.apply(rule, local)
        })?;
    Ok((
        CompactMultiplicityFreeTreeBasis {
            frame: prepared.output_frame,
            locals,
        },
        next_columns,
    ))
}

fn scatter_compact_tree_block<S>(
    basis: CompactMultiplicityFreeTreeBasis,
    columns: &DenseColumns<S>,
) -> Vec<Vec<(FusionTreeKey, S)>>
where
    S: Clone,
{
    let mut rows_per_source = vec![Vec::new(); columns.num_src];
    for (destination_row, destination_local) in basis.locals.into_iter().enumerate() {
        // Why not materialize in each source column: the full external frame is
        // identical for the row and must be rebuilt only once at the API edge.
        let destination_key = basis.frame.materialize(destination_local);
        for (source, coefficient) in columns.row(destination_row).iter().enumerate() {
            if let Some(coefficient) = coefficient {
                rows_per_source[source].push((destination_key.clone(), coefficient.clone()));
            }
        }
    }
    rows_per_source
}

fn compact_bendright_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_bendright(rule, &basis.frame)?;
    let (locals, columns) =
        apply_first_compact_block_terms(rule, &basis.locals, |rule, local| {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            let coefficient = prepared.coefficient(rule, &validated);
            Ok(std::iter::once((validated.local, coefficient)))
        })?;
    let frame = prepared.output_frame(rule)?;
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        columns,
    ))
}

fn compact_bendleft_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_bendleft(rule, &basis.frame)?;
    let (locals, columns) =
        apply_first_compact_block_terms(rule, &basis.locals, |rule, local| {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            Ok(std::iter::once(prepared.finish_local(rule, validated)))
        })?;
    let frame = prepared.output_frame(rule)?;
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        columns,
    ))
}

fn compact_foldright_block<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: Option<&DenseColumns<R::Scalar>>,
    swap_output: bool,
    conjugate_step: bool,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_foldright(rule, &basis.frame)?;
    let output_frame = if swap_output {
        MultiplicityFreeTreePairFrame {
            codomain: prepared.output_frame.domain.clone(),
            domain: prepared.output_frame.codomain.clone(),
        }
    } else {
        prepared.output_frame.clone()
    };
    let output_locals =
        collect_multiplicity_free_tree_pair_locals_for_frame(rule, &output_frame);
    let output_index = output_locals
        .iter()
        .cloned()
        .enumerate()
        .map(|(row, local)| (local, row))
        .collect::<FxHashMap<_, _>>();
    let num_src = columns
        .map(|columns| columns.num_src)
        .unwrap_or(basis.locals.len());
    let mut next_columns = DenseColumns::with_capacity(num_src, output_locals.len());
    for _ in &output_locals {
        next_columns.push_empty_row();
    }

    let mut forward_cache: FxHashMap<
        MultiplicityFreeTreeLocal,
        MultiplicityFreeTreeLocalTerms<R::Scalar>,
    > = FxHashMap::default();
    let mut inverse_cache: MultiplicityFreeFoldInverseCache<R::Scalar> = FxHashMap::default();
    let mut coefficient_cache: FxHashMap<
        (SectorId, SectorId),
        (R::Scalar, R::Scalar),
    > = FxHashMap::default();

    for (source_row, source) in basis.locals.iter().enumerate() {
        if columns.is_some_and(|columns| {
            columns
                .row(source_row)
                .iter()
                .all(Option::is_none)
        }) {
            continue;
        }
        if !forward_cache.contains_key(&source.codomain) {
            let terms = multiplicity_free_multi_fmove_local(
                rule,
                &basis.frame.codomain,
                &source.codomain,
            )?;
            forward_cache.insert(source.codomain.clone(), terms);
        }
        let forward = forward_cache
            .get(&source.codomain)
            .expect("forward fold table inserted above");
        let coupled = source.codomain.coupled;
        for (codomain, codomain_coefficient) in forward {
            let tail_coupled = codomain.coupled;
            let inverse_key = (tail_coupled, source.domain.clone());
            if !inverse_cache.contains_key(&inverse_key) {
                let terms = multiplicity_free_multi_fmove_inv_local(
                    rule,
                    tail_coupled,
                    &basis.frame.domain,
                    &source.domain,
                    &prepared.output_frame.domain,
                )?;
                inverse_cache.insert(inverse_key.clone(), terms);
            }
            let inverse = inverse_cache
                .get(&inverse_key)
                .expect("inverse fold table inserted above");
            let cache_key = (tail_coupled, coupled);
            if let Entry::Vacant(entry) = coefficient_cache.entry(cache_key) {
                entry.insert((
                    rule.sqrt_dim_scalar(coupled) * rule.inv_sqrt_dim_scalar(tail_coupled),
                    rule.a_symbol_scalar(prepared.first, tail_coupled, coupled),
                ));
            }
            let (normalization, a_symbol) = coefficient_cache
                .get(&cache_key)
                .expect("fold coefficient table inserted above");
            for (domain, domain_coefficient) in inverse {
                let mut destination = MultiplicityFreeTreePairLocal {
                    codomain: codomain.clone(),
                    domain: domain.clone(),
                };
                if swap_output {
                    destination = MultiplicityFreeTreePairLocal {
                        codomain: destination.domain,
                        domain: destination.codomain,
                    };
                }
                let destination_row = *output_index.get(&destination).ok_or(
                    CoreError::MalformedFusionTree {
                        message: "compact fold destination is outside the canonical output basis",
                    },
                )?;
                let mut coefficient = normalization.clone()
                    * (domain_coefficient.clone()).conj()
                    * a_symbol.clone()
                    * codomain_coefficient.clone();
                if prepared.first_is_dual {
                    coefficient = coefficient * prepared.frobenius_schur_phase.clone();
                }
                if conjugate_step {
                    coefficient = (coefficient).conj();
                }
                if let Some(columns) = columns {
                    let source_column = columns.row(source_row);
                    let destination_column = next_columns.row_mut(destination_row);
                    for (source, source_coefficient) in source_column.iter().enumerate() {
                        let Some(source_coefficient) = source_coefficient else {
                            continue;
                        };
                        let contribution =
                            coefficient.clone() * source_coefficient.clone();
                        destination_column[source] =
                            Some(match destination_column[source].take() {
                                Some(existing) => existing + contribution,
                                None => contribution,
                            });
                    }
                } else {
                    let destination = &mut next_columns.row_mut(destination_row)[source_row];
                    *destination = Some(match destination.take() {
                        Some(existing) => existing + coefficient,
                        None => coefficient,
                    });
                }
            }
        }
    }

    Ok((
        CompactMultiplicityFreeTreePairBasis {
            frame: output_frame,
            locals: output_locals,
        },
        next_columns,
    ))
}

fn compact_foldright_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    compact_foldright_block(rule, basis, None, false, false)
}

fn compact_foldleft_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let CompactMultiplicityFreeTreePairBasis { frame, locals } = basis;
    let swapped = CompactMultiplicityFreeTreePairBasis {
        frame: MultiplicityFreeTreePairFrame {
            codomain: frame.domain,
            domain: frame.codomain,
        },
        locals: locals
            .into_iter()
            .map(|local| MultiplicityFreeTreePairLocal {
                codomain: local.domain,
                domain: local.codomain,
            })
            .collect(),
    };
    compact_foldright_block(rule, swapped, None, true, true)
}

fn compact_codomain_artin_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    index: usize,
    inverse: bool,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared =
        prepare_multiplicity_free_artin(rule, &basis.frame.codomain, index, inverse)?;
    let (locals, columns) =
        apply_first_compact_block_terms(rule, &basis.locals, |rule, local| {
            let domain = local.domain.clone();
            Ok(prepared
                .apply(rule, &local.codomain)?
                .into_iter()
                .map(move |(codomain, coefficient)| {
                    (
                        MultiplicityFreeTreePairLocal {
                            codomain,
                            domain: domain.clone(),
                        },
                        coefficient,
                    )
                }))
        })?;
    let frame = MultiplicityFreeTreePairFrame {
        codomain: prepared.output_frame,
        domain: basis.frame.domain,
    };
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        columns,
    ))
}

fn compact_bendright_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_bendright(rule, &basis.frame)?;
    let (locals, next_columns) =
        compose_compact_block_terms(rule, &basis.locals, columns, |rule, local| {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            let coefficient = prepared.coefficient(rule, &validated);
            Ok(std::iter::once((validated.local, coefficient)))
        })?;
    let frame = prepared.output_frame(rule)?;
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        next_columns,
    ))
}

fn compact_bendleft_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_bendleft(rule, &basis.frame)?;
    let (locals, next_columns) =
        compose_compact_block_terms(rule, &basis.locals, columns, |rule, local| {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            Ok(std::iter::once(prepared.finish_local(rule, validated)))
        })?;
    let frame = prepared.output_frame(rule)?;
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        next_columns,
    ))
}

fn compact_foldright_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    compact_foldright_block(rule, basis, Some(columns), false, false)
}

fn compact_foldleft_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let CompactMultiplicityFreeTreePairBasis { frame, locals } = basis;
    let swapped = CompactMultiplicityFreeTreePairBasis {
        frame: MultiplicityFreeTreePairFrame {
            codomain: frame.domain,
            domain: frame.codomain,
        },
        locals: locals
            .into_iter()
            .map(|local| MultiplicityFreeTreePairLocal {
                codomain: local.domain,
                domain: local.codomain,
            })
            .collect(),
    };
    compact_foldright_block(rule, swapped, Some(columns), true, true)
}

fn compact_codomain_artin_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
    index: usize,
    inverse: bool,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared =
        prepare_multiplicity_free_artin(rule, &basis.frame.codomain, index, inverse)?;
    let (locals, next_columns) =
        compose_compact_block_terms(rule, &basis.locals, columns, |rule, local| {
            let domain = local.domain.clone();
            Ok(prepared
                .apply(rule, &local.codomain)?
                .into_iter()
                .map(move |(codomain, coefficient)| {
                    (
                        MultiplicityFreeTreePairLocal {
                            codomain,
                            domain: domain.clone(),
                        },
                        coefficient,
                    )
                }))
        })?;
    let frame = MultiplicityFreeTreePairFrame {
        codomain: prepared.output_frame,
        domain: basis.frame.domain,
    };
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        next_columns,
    ))
}

fn compact_cycle_clockwise_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if basis.frame.codomain.uncoupled.is_empty() {
        let (basis, columns) = compact_bendleft_block_first(rule, basis)?;
        compact_foldright_block_step(rule, basis, &columns)
    } else {
        let (basis, columns) = compact_foldright_block_first(rule, basis)?;
        compact_bendleft_block_step(rule, basis, &columns)
    }
}

fn compact_cycle_clockwise_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if basis.frame.codomain.uncoupled.is_empty() {
        let (basis, columns) = compact_bendleft_block_step(rule, basis, columns)?;
        compact_foldright_block_step(rule, basis, &columns)
    } else {
        let (basis, columns) = compact_foldright_block_step(rule, basis, columns)?;
        compact_bendleft_block_step(rule, basis, &columns)
    }
}

fn compact_cycle_anticlockwise_block_first<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if basis.frame.domain.uncoupled.is_empty() {
        let (basis, columns) = compact_bendright_block_first(rule, basis)?;
        compact_foldleft_block_step(rule, basis, &columns)
    } else {
        let (basis, columns) = compact_foldleft_block_first(rule, basis)?;
        compact_bendright_block_step(rule, basis, &columns)
    }
}

fn compact_cycle_anticlockwise_block_step<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: &DenseColumns<R::Scalar>,
) -> Result<
    (
        CompactMultiplicityFreeTreePairBasis,
        DenseColumns<R::Scalar>,
    ),
    CoreError,
>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if basis.frame.domain.uncoupled.is_empty() {
        let (basis, columns) = compact_bendright_block_step(rule, basis, columns)?;
        compact_foldleft_block_step(rule, basis, &columns)
    } else {
        let (basis, columns) = compact_foldleft_block_step(rule, basis, columns)?;
        compact_bendright_block_step(rule, basis, &columns)
    }
}

fn scatter_compact_block<S: Clone>(
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: DenseColumns<S>,
) -> Vec<Vec<(FusionTreePairKey, S)>> {
    #[cfg(test)]
    COMPACT_BLOCK_DIMENSIONS.with(|dimensions| {
        dimensions.set(Some(CompactBlockDimensions {
            destination_rows: basis.locals.len(),
            source_columns: columns.num_src,
            coefficient_slots: columns.data.len(),
            coefficient_bytes: columns
                .data
                .len()
                .saturating_mul(std::mem::size_of::<Option<S>>()),
        }));
    });
    let mut rows_per_source = vec![Vec::new(); columns.num_src];
    for (destination_row, destination_local) in basis.locals.into_iter().enumerate() {
        let destination = basis.frame.materialize(destination_local);
        for (source, coefficient) in columns.row(destination_row).iter().enumerate() {
            if let Some(coefficient) = coefficient {
                rows_per_source[source].push((destination.clone(), coefficient.clone()));
            }
        }
    }
    rows_per_source
}

fn order_compact_block<S: Clone>(
    basis: CompactMultiplicityFreeTreePairBasis,
    mut columns: DenseColumns<S>,
) -> OrderedBlockLinearMap<FusionTreePairKey, S> {
    #[cfg(test)]
    COMPACT_BLOCK_DIMENSIONS.with(|dimensions| {
        dimensions.set(Some(CompactBlockDimensions {
            destination_rows: basis.locals.len(),
            source_columns: columns.num_src,
            coefficient_slots: columns.data.len(),
            coefficient_bytes: columns
                .data
                .len()
                .saturating_mul(std::mem::size_of::<Option<S>>()),
        }));
    });

    let source_count = columns.num_src;
    let basis_row_count = basis.locals.len();
    let mut ordered_basis_rows = Vec::with_capacity(basis_row_count);
    let mut ordered_row_for_basis = vec![usize::MAX; basis_row_count];
    let mut singleton_basis_rows = Vec::with_capacity(source_count);
    let mut is_singleton = true;

    for source in 0..source_count {
        let mut only_basis_row = None;
        for (basis_row, ordered_row) in ordered_row_for_basis.iter_mut().enumerate() {
            if columns.row(basis_row)[source].is_none() {
                continue;
            }
            if *ordered_row == usize::MAX {
                *ordered_row = ordered_basis_rows.len();
                ordered_basis_rows.push(basis_row);
            }
            if only_basis_row.replace(basis_row).is_some() {
                is_singleton = false;
            }
        }
        match only_basis_row {
            Some(basis_row) => singleton_basis_rows.push(basis_row),
            None => {
                is_singleton = false;
                singleton_basis_rows.push(usize::MAX);
            }
        }
    }

    let CompactMultiplicityFreeTreePairBasis { frame, locals } = basis;
    let mut local_slots = locals.into_iter().map(Some).collect::<Vec<_>>();
    let destinations = ordered_basis_rows
        .iter()
        .map(|&basis_row| {
            frame.materialize(
                local_slots[basis_row]
                    .take()
                    .expect("ordered compact rows contain each basis row once"),
            )
        })
        .collect();

    let storage = if is_singleton {
        let mut destination_rows = Vec::with_capacity(source_count);
        let mut coefficients = Vec::with_capacity(source_count);
        for (source, basis_row) in singleton_basis_rows.into_iter().enumerate() {
            destination_rows.push(ordered_row_for_basis[basis_row]);
            coefficients.push(
                columns.data[basis_row * source_count + source]
                    .take()
                    .expect("singleton source has one present coefficient"),
            );
        }
        OrderedBlockLinearStorage::SingletonColumns {
            destination_rows,
            coefficients,
        }
    } else {
        let mut coefficients =
            Vec::with_capacity(ordered_basis_rows.len().saturating_mul(source_count));
        for basis_row in ordered_basis_rows {
            let row_start = basis_row * source_count;
            coefficients.extend(
                columns.data[row_start..row_start + source_count]
                    .iter_mut()
                    .map(Option::take),
            );
        }
        OrderedBlockLinearStorage::DenseDstSrc(coefficients)
    };

    OrderedBlockLinearMap {
        destinations,
        source_count,
        storage,
    }
}

fn scatter_compact_tree_pair_block<S: Clone>(
    block: CompactMultiplicityFreeTreePairBlock<S>,
) -> CompactMultiplicityFreeTreePairRows<S> {
    if block.records_dimensions {
        scatter_compact_block(block.basis, block.columns)
    } else {
        let mut rows_per_source = vec![Vec::new(); block.columns.num_src];
        for (destination_row, destination_local) in
            block.basis.locals.into_iter().enumerate()
        {
            let destination = block.basis.frame.materialize(destination_local);
            for (source, coefficient) in
                block.columns.row(destination_row).iter().enumerate()
            {
                if let Some(coefficient) = coefficient {
                    rows_per_source[source]
                        .push((destination.clone(), coefficient.clone()));
                }
            }
        }
        rows_per_source
    }
}

fn order_compact_tree_pair_block<S: Clone>(
    block: CompactMultiplicityFreeTreePairBlock<S>,
) -> OrderedBlockLinearMap<FusionTreePairKey, S> {
    order_compact_block(block.basis, block.columns)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompactBlockDimensions {
    destination_rows: usize,
    source_columns: usize,
    coefficient_slots: usize,
    coefficient_bytes: usize,
}

#[cfg(test)]
thread_local! {
    static COMPACT_BLOCK_DIMENSIONS: std::cell::Cell<Option<CompactBlockDimensions>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn reset_compact_block_dimensions() {
    COMPACT_BLOCK_DIMENSIONS.with(|dimensions| dimensions.set(None));
}

#[cfg(test)]
fn compact_block_dimensions() -> Option<CompactBlockDimensions> {
    COMPACT_BLOCK_DIMENSIONS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn assert_compact_tree_pair_basis_matches_homspace<R>(
    rule: &R,
    basis: &CompactMultiplicityFreeTreePairBasis,
) where
    R: MultiplicityFreeFusionRule,
{
    let product_space = |frame: &MultiplicityFreeTreeFrame| {
        FusionProductSpace::new(
            frame
                .uncoupled
                .iter()
                .copied()
                .zip(frame.is_dual.iter().copied())
                .map(|(sector, is_dual)| SectorLeg::new([(sector, 1)], is_dual)),
        )
    };
    let hom_space = FusionTreeHomSpace::new(
        product_space(&basis.frame.codomain),
        product_space(&basis.frame.domain),
    );
    let expected = hom_space.fusion_tree_keys(rule);
    let actual = basis
        .locals
        .iter()
        .cloned()
        .map(|local| basis.frame.materialize(local))
        .collect::<Vec<_>>();
    assert_eq!(actual.as_slice(), expected.as_ref());
}

fn preflight_compact_repartition_source_major<R>(
    group: &ValidatedTreePairBlockGroup<'_, R>,
    current_codomain_rank: usize,
    target_codomain_rank: usize,
) -> Result<(), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let rule = group.rule;
    let (initial_frame, first_local) = MultiplicityFreeTreePairFrame::split(
        group
            .projection
            .pair_at(0)
            .ok_or(CoreError::MalformedFusionTree {
                message: "compact repartition requires at least one source",
            })?,
    );
    if current_codomain_rank > target_codomain_rank {
        let mut steps: SmallVec<[PreparedMultiplicityFreeBendRight; 8]> =
            SmallVec::new();
        let mut frame = initial_frame.clone();
        let mut local = first_local;
        let num_steps = current_codomain_rank - target_codomain_rank;
        for step in 0..num_steps {
            let prepared = prepare_multiplicity_free_bendright(rule, &frame)?;
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            local = validated.local;
            if step + 1 == num_steps {
                prepared.validate_output_frame()?;
            } else {
                frame = prepared.output_frame(rule)?;
            }
            steps.push(prepared);
        }
        for index in 1..group.source_len {
            let (source_frame, mut local) =
                MultiplicityFreeTreePairFrame::split(
                    group
                        .projection
                        .pair_at(index)
                        .expect("validated projection covers every source"),
                );
            if source_frame != initial_frame {
                return Err(CoreError::MalformedFusionTree {
                    message: TREE_PAIR_BLOCK_GROUP_ERROR,
                });
            }
            for prepared in &steps {
                local = prepared
                    .validate_local(rule, &local.codomain, &local.domain)?
                    .local;
            }
        }
    } else {
        let mut steps: SmallVec<[PreparedMultiplicityFreeBendLeft; 8]> =
            SmallVec::new();
        let mut frame = initial_frame.clone();
        let mut local = first_local;
        let num_steps = target_codomain_rank - current_codomain_rank;
        for step in 0..num_steps {
            let prepared = prepare_multiplicity_free_bendleft(rule, &frame)?;
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            local = PreparedMultiplicityFreeBendLeft::finish_local_structure(validated);
            if step + 1 == num_steps {
                prepared.validate_output_frame()?;
            } else {
                frame = prepared.output_frame(rule)?;
            }
            steps.push(prepared);
        }
        for index in 1..group.source_len {
            let (source_frame, mut local) =
                MultiplicityFreeTreePairFrame::split(
                    group
                        .projection
                        .pair_at(index)
                        .expect("validated projection covers every source"),
                );
            if source_frame != initial_frame {
                return Err(CoreError::MalformedFusionTree {
                    message: TREE_PAIR_BLOCK_GROUP_ERROR,
                });
            }
            for prepared in &steps {
                let validated =
                    prepared.validate_local(rule, &local.codomain, &local.domain)?;
                local =
                    PreparedMultiplicityFreeBendLeft::finish_local_structure(validated);
            }
        }
    }
    Ok(())
}

fn compact_repartition_tree_pair_block<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    target_codomain_rank: usize,
) -> Result<CompactMultiplicityFreeTreePairBlock<R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rule = group.rule;
    let mut current_codomain_rank = group.codomain_rank;
    if current_codomain_rank == target_codomain_rank {
        let source_len = group.source_len;
        let basis = CompactMultiplicityFreeTreePairBasis::from_group(group)?;
        let mut columns = DenseColumns::with_capacity(source_len, source_len);
        for source in 0..source_len {
            let row = columns.push_empty_row();
            columns.row_mut(row)[source] = Some(R::Scalar::one());
        }
        return Ok(CompactMultiplicityFreeTreePairBlock {
            basis,
            columns,
            records_dimensions: false,
        });
    }
    // Why not rely on the step-major compact execution for malformed inputs:
    // the legacy public API reports the first error in source-major order.
    preflight_compact_repartition_source_major(
        &group,
        current_codomain_rank,
        target_codomain_rank,
    )?;
    let (mut frame, first_local) = MultiplicityFreeTreePairFrame::split(
        group
            .projection
            .pair_at(0)
            .ok_or(CoreError::MalformedFusionTree {
                message: "compact repartition requires at least one source",
            })?,
    );
    let mut rows = Vec::with_capacity(group.source_len);
    rows.push((first_local, R::Scalar::one()));
    for index in 1..group.source_len {
        let (source_frame, source_local) =
            MultiplicityFreeTreePairFrame::split(
                group
                    .projection
                    .pair_at(index)
                    .expect("validated projection covers every source"),
            );
        if source_frame != frame {
            return Err(CoreError::MalformedFusionTree {
                message: TREE_PAIR_BLOCK_GROUP_ERROR,
            });
        }
        rows.push((source_local, R::Scalar::one()));
    }

    while current_codomain_rank > target_codomain_rank {
        let prepared = prepare_multiplicity_free_bendright(rule, &frame)?;
        for (local, coefficient) in &mut rows {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            let step_coefficient = prepared.coefficient(rule, &validated);
            *local = validated.local;
            *coefficient = coefficient.clone() * step_coefficient;
        }
        frame = prepared.output_frame(rule)?;
        current_codomain_rank -= 1;
    }
    while current_codomain_rank < target_codomain_rank {
        let prepared = prepare_multiplicity_free_bendleft(rule, &frame)?;
        for (local, coefficient) in &mut rows {
            let validated =
                prepared.validate_local(rule, &local.codomain, &local.domain)?;
            let (next_local, step_coefficient) =
                prepared.finish_local(rule, validated);
            *local = next_local;
            *coefficient = coefficient.clone() * step_coefficient;
        }
        frame = prepared.output_frame(rule)?;
        current_codomain_rank += 1;
    }
    let source_count = rows.len();
    let mut destination_locals = Vec::with_capacity(source_count);
    let mut columns = DenseColumns::with_capacity(source_count, source_count);
    for (source, (local, coefficient)) in rows.into_iter().enumerate() {
        // Repartition is an invertible change of the tree split, so distinct
        // source basis states stay distinct. Why not hash the finished locals:
        // that would rebuild identity already proved by the reversible bend
        // sequence merely to discover the source-major row number again.
        destination_locals.push(local);
        let destination_row = columns.push_empty_row();
        debug_assert_eq!(destination_row, source);
        columns.row_mut(destination_row)[source] = Some(coefficient);
    }
    Ok(CompactMultiplicityFreeTreePairBlock {
        basis: CompactMultiplicityFreeTreePairBasis {
            frame,
            locals: destination_locals,
        },
        columns,
        records_dimensions: false,
    })
}
