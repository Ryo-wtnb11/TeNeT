use super::*;

fn apply_first_compact_block_terms<S, K, F, I>(
    basis: &[K],
    mut transform: F,
) -> Result<(Vec<K>, DenseColumns<S>), CoreError>
where
    S: Clone + Add<Output = S>,
    K: Eq + Hash,
    F: FnMut(&K) -> Result<I, CoreError>,
    I: IntoIterator<Item = (K, S)>,
{
    let mut index: FxHashMap<K, usize> = FxHashMap::default();
    let mut columns = DenseColumns::with_capacity(basis.len(), basis.len());
    for (source, source_local) in basis.iter().enumerate() {
        for (destination_local, coefficient) in transform(source_local)? {
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

/// Run one compact move over `basis`: the first move of a chain writes the
/// source columns directly (`columns == None`); later moves compose through
/// the previous coefficient matrix. The move itself is `transform`.
fn apply_compact_block_terms<S, K, F, I>(
    basis: &[K],
    columns: Option<&DenseColumns<S>>,
    transform: F,
) -> Result<(Vec<K>, DenseColumns<S>), CoreError>
where
    S: Clone + Add<Output = S> + Mul<Output = S>,
    K: Eq + Hash,
    F: FnMut(&K) -> Result<I, CoreError>,
    I: IntoIterator<Item = (K, S)>,
{
    match columns {
        None => apply_first_compact_block_terms(basis, transform),
        Some(columns) => compose_block_terms(basis, columns, transform),
    }
}

pub(super) fn compact_artin_tree_block<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreeBasis,
    columns: Option<&DenseColumns<R::Scalar>>,
    index: usize,
    inverse: bool,
) -> Result<(CompactMultiplicityFreeTreeBasis, DenseColumns<R::Scalar>), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let prepared = prepare_multiplicity_free_artin(rule, &basis.frame, index, inverse)?;
    let (locals, next_columns) =
        apply_compact_block_terms(&basis.locals, columns, |local| prepared.apply(rule, local))?;
    Ok((
        CompactMultiplicityFreeTreeBasis {
            frame: prepared.output_frame,
            locals,
        },
        next_columns,
    ))
}

pub(super) fn scatter_compact_tree_block<S>(
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

pub(crate) fn compact_bendright_block<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: Option<&DenseColumns<R::Scalar>>,
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
    let (locals, next_columns) = apply_compact_block_terms(&basis.locals, columns, |local| {
        let validated = prepared.validate_local(rule, &local.codomain, &local.domain)?;
        let coefficient = prepared.coefficient(rule, &validated);
        Ok(std::iter::once((validated.local, coefficient)))
    })?;
    let frame = prepared.output_frame(rule)?;
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        next_columns,
    ))
}

pub(crate) fn compact_bendleft_block<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: Option<&DenseColumns<R::Scalar>>,
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
    let (locals, next_columns) = apply_compact_block_terms(&basis.locals, columns, |local| {
        let validated = prepared.validate_local(rule, &local.codomain, &local.domain)?;
        Ok(std::iter::once(prepared.finish_local(rule, validated)))
    })?;
    let frame = prepared.output_frame(rule)?;
    Ok((
        CompactMultiplicityFreeTreePairBasis { frame, locals },
        next_columns,
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
    let output_locals = collect_multiplicity_free_tree_pair_locals_for_frame(rule, &output_frame);
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
    let mut coefficient_cache: FxHashMap<(SectorId, SectorId), (R::Scalar, R::Scalar)> =
        FxHashMap::default();

    for (source_row, source) in basis.locals.iter().enumerate() {
        if columns.is_some_and(|columns| columns.row(source_row).iter().all(Option::is_none)) {
            continue;
        }
        if !forward_cache.contains_key(&source.codomain) {
            let terms =
                multiplicity_free_multi_fmove_local(rule, &basis.frame.codomain, &source.codomain)?;
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
                entry.insert(
                    prepared
                        .coefficient
                        .sector_factors(rule, tail_coupled, coupled),
                );
            }
            let factors = coefficient_cache
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
                let destination_row =
                    *output_index
                        .get(&destination)
                        .ok_or(CoreError::MalformedFusionTree {
                            message:
                                "compact fold destination is outside the canonical output basis",
                        })?;
                let mut coefficient = prepared.coefficient.coefficient(
                    factors,
                    codomain_coefficient,
                    domain_coefficient,
                );
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
                        let contribution = coefficient.clone() * source_coefficient.clone();
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

fn compact_foldleft_block<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: Option<&DenseColumns<R::Scalar>>,
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
    compact_foldright_block(rule, swapped, columns, true, true)
}

pub(crate) fn compact_codomain_artin_block<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: Option<&DenseColumns<R::Scalar>>,
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
    let prepared = prepare_multiplicity_free_artin(rule, &basis.frame.codomain, index, inverse)?;
    let (locals, next_columns) = apply_compact_block_terms(&basis.locals, columns, |local| {
        let domain = local.domain.clone();
        Ok(prepared.apply(rule, &local.codomain)?.into_iter().map(
            move |(codomain, coefficient)| {
                (
                    MultiplicityFreeTreePairLocal {
                        codomain,
                        domain: domain.clone(),
                    },
                    coefficient,
                )
            },
        ))
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

pub(super) fn compact_cycle_clockwise_block<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: Option<&DenseColumns<R::Scalar>>,
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
        let (basis, columns) = compact_bendleft_block(rule, basis, columns)?;
        compact_foldright_block(rule, basis, Some(&columns), false, false)
    } else {
        let (basis, columns) = compact_foldright_block(rule, basis, columns, false, false)?;
        compact_bendleft_block(rule, basis, Some(&columns))
    }
}

pub(super) fn compact_cycle_anticlockwise_block<R>(
    rule: &R,
    basis: CompactMultiplicityFreeTreePairBasis,
    columns: Option<&DenseColumns<R::Scalar>>,
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
        let (basis, columns) = compact_bendright_block(rule, basis, columns)?;
        compact_foldleft_block(rule, basis, Some(&columns))
    } else {
        let (basis, columns) = compact_foldleft_block(rule, basis, columns)?;
        compact_bendright_block(rule, basis, Some(&columns))
    }
}

pub(crate) fn scatter_compact_block<S: Clone>(
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

pub(crate) fn order_compact_block<S: Clone>(
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

pub(super) fn scatter_compact_tree_pair_block<S: Clone>(
    block: CompactMultiplicityFreeTreePairBlock<S>,
) -> CompactMultiplicityFreeTreePairRows<S> {
    if block.records_dimensions {
        scatter_compact_block(block.basis, block.columns)
    } else {
        let mut rows_per_source = vec![Vec::new(); block.columns.num_src];
        for (destination_row, destination_local) in block.basis.locals.into_iter().enumerate() {
            let destination = block.basis.frame.materialize(destination_local);
            for (source, coefficient) in block.columns.row(destination_row).iter().enumerate() {
                if let Some(coefficient) = coefficient {
                    rows_per_source[source].push((destination.clone(), coefficient.clone()));
                }
            }
        }
        rows_per_source
    }
}

pub(super) fn order_compact_tree_pair_block<S: Clone>(
    block: CompactMultiplicityFreeTreePairBlock<S>,
) -> OrderedBlockLinearMap<FusionTreePairKey, S> {
    order_compact_block(block.basis, block.columns)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompactBlockDimensions {
    pub(crate) destination_rows: usize,
    pub(crate) source_columns: usize,
    pub(crate) coefficient_slots: usize,
    pub(crate) coefficient_bytes: usize,
}

#[cfg(test)]
thread_local! {
    static COMPACT_BLOCK_DIMENSIONS: std::cell::Cell<Option<CompactBlockDimensions>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn reset_compact_block_dimensions() {
    COMPACT_BLOCK_DIMENSIONS.with(|dimensions| dimensions.set(None));
}

#[cfg(test)]
pub(crate) fn compact_block_dimensions() -> Option<CompactBlockDimensions> {
    COMPACT_BLOCK_DIMENSIONS.with(std::cell::Cell::get)
}

/// Test-only invariant of a compact transpose output basis: its keys are
/// distinct and all lie in the destination frame's HomSpace.
///
/// Why not compare with the HomSpace key order: bend steps keep the
/// first-appearance order of the previous basis while changing the coupled
/// sector, and caller-selected cohorts may be partial, so no row order or
/// completeness is promised. Consumers resolve destinations by key.
#[cfg(test)]
pub(super) fn assert_compact_tree_pair_basis_in_homspace<R>(
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
    let homspace_keys = hom_space.fusion_tree_keys(rule);
    let homspace_keys = homspace_keys
        .iter()
        .collect::<std::collections::HashSet<_>>();
    let mut seen = std::collections::HashSet::new();
    for local in &basis.locals {
        let key = basis.frame.materialize(local.clone());
        assert!(
            homspace_keys.contains(&key),
            "compact basis key {key:?} is outside the destination HomSpace"
        );
        assert!(seen.insert(key), "compact basis repeats a destination key");
    }
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
    let (initial_frame, first_local) =
        MultiplicityFreeTreePairFrame::split(group.projection.pair_at(0).ok_or(
            CoreError::MalformedFusionTree {
                message: "compact repartition requires at least one source",
            },
        )?);
    let (bend, num_steps) = if current_codomain_rank > target_codomain_rank {
        (Bend::Right, current_codomain_rank - target_codomain_rank)
    } else {
        (Bend::Left, target_codomain_rank - current_codomain_rank)
    };
    let mut steps: SmallVec<[PreparedStructuralBend; 8]> = SmallVec::new();
    let mut frame = initial_frame.clone();
    let mut local = first_local;
    for step in 0..num_steps {
        let prepared = PreparedStructuralBend::prepare(rule, &frame, bend)?;
        local = prepared.next_local(rule, &local)?;
        if step + 1 == num_steps {
            prepared.validate_output_frame()?;
        } else {
            frame = prepared.output_frame(rule)?;
        }
        steps.push(prepared);
    }
    for index in 1..group.source_len {
        let (source_frame, mut local) = MultiplicityFreeTreePairFrame::split(
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
            local = prepared.next_local(rule, &local)?;
        }
    }
    Ok(())
}

/// The structural part of one repartition bend (no coefficient), for the
/// source-major preflight.
enum PreparedStructuralBend {
    Right(PreparedMultiplicityFreeBendRight),
    Left(PreparedMultiplicityFreeBendLeft),
}

impl PreparedStructuralBend {
    fn prepare<R>(
        rule: &R,
        frame: &MultiplicityFreeTreePairFrame,
        bend: Bend,
    ) -> Result<Self, CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        Ok(match bend {
            Bend::Right => Self::Right(prepare_multiplicity_free_bendright(rule, frame)?),
            Bend::Left => Self::Left(prepare_multiplicity_free_bendleft(rule, frame)?),
        })
    }

    fn next_local<R>(
        &self,
        rule: &R,
        local: &MultiplicityFreeTreePairLocal,
    ) -> Result<MultiplicityFreeTreePairLocal, CoreError>
    where
        R: FusionRule,
    {
        Ok(match self {
            Self::Right(prepared) => {
                prepared
                    .validate_local(rule, &local.codomain, &local.domain)?
                    .local
            }
            Self::Left(prepared) => PreparedMultiplicityFreeBendLeft::finish_local_structure(
                prepared.validate_local(rule, &local.codomain, &local.domain)?,
            ),
        })
    }

    fn validate_output_frame(&self) -> Result<(), CoreError> {
        match self {
            Self::Right(prepared) => prepared.validate_output_frame(),
            Self::Left(prepared) => prepared.validate_output_frame(),
        }
    }

    fn output_frame<R: FusionRule>(
        &self,
        rule: &R,
    ) -> Result<MultiplicityFreeTreePairFrame, CoreError> {
        match self {
            Self::Right(prepared) => prepared.output_frame(rule),
            Self::Left(prepared) => prepared.output_frame(rule),
        }
    }
}

pub(super) fn compact_repartition_tree_pair_block<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    target_codomain_rank: usize,
) -> Result<CompactMultiplicityFreeTreePairBlock<R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rule = group.rule;
    let current_codomain_rank = group.codomain_rank;
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
    let (frame, first_local) =
        MultiplicityFreeTreePairFrame::split(group.projection.pair_at(0).ok_or(
            CoreError::MalformedFusionTree {
                message: "compact repartition requires at least one source",
            },
        )?);
    let mut rows = Vec::with_capacity(group.source_len);
    rows.push((first_local, R::Scalar::one()));
    for index in 1..group.source_len {
        let (source_frame, source_local) = MultiplicityFreeTreePairFrame::split(
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

    let (frame, rows) = repartition_loop(
        (frame, rows),
        current_codomain_rank,
        target_codomain_rank,
        |(frame, mut rows), bend| {
            let frame = match bend {
                Bend::Right => {
                    let prepared = prepare_multiplicity_free_bendright(rule, &frame)?;
                    for (local, coefficient) in &mut rows {
                        let validated =
                            prepared.validate_local(rule, &local.codomain, &local.domain)?;
                        let step_coefficient = prepared.coefficient(rule, &validated);
                        *local = validated.local;
                        *coefficient = coefficient.clone() * step_coefficient;
                    }
                    prepared.output_frame(rule)?
                }
                Bend::Left => {
                    let prepared = prepare_multiplicity_free_bendleft(rule, &frame)?;
                    for (local, coefficient) in &mut rows {
                        let validated =
                            prepared.validate_local(rule, &local.codomain, &local.domain)?;
                        let (next_local, step_coefficient) = prepared.finish_local(rule, validated);
                        *local = next_local;
                        *coefficient = coefficient.clone() * step_coefficient;
                    }
                    prepared.output_frame(rule)?
                }
            };
            Ok::<_, CoreError>((frame, rows))
        },
    )?;
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
