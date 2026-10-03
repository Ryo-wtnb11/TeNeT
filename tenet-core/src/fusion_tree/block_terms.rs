use super::*;

pub(crate) enum FusionTermAccumulator<K, S> {
    Empty,
    Singleton(K, S),
    Map {
        order: Vec<K>,
        coefficients: FxHashMap<K, S>,
    },
}

impl<K, S> FusionTermAccumulator<K, S>
where
    K: Clone + Eq + Hash,
    S: Clone + Add<Output = S>,
{
    pub(crate) fn new() -> Self {
        Self::Empty
    }

    pub(crate) fn push(&mut self, key: K, coefficient: S) {
        match self {
            Self::Empty => {
                *self = Self::Singleton(key, coefficient);
            }
            Self::Singleton(existing_key, existing) if existing_key == &key => {
                *existing = existing.clone() + coefficient;
            }
            Self::Singleton(_, _) => {
                let previous = std::mem::replace(self, Self::Empty);
                let Self::Singleton(existing_key, existing_coefficient) = previous else {
                    unreachable!("matched singleton state");
                };
                let mut order = Vec::with_capacity(2);
                let mut coefficients = FxHashMap::default();
                Self::push_map_term(
                    &mut order,
                    &mut coefficients,
                    existing_key,
                    existing_coefficient,
                );
                Self::push_map_term(&mut order, &mut coefficients, key, coefficient);
                *self = Self::Map {
                    order,
                    coefficients,
                };
            }
            Self::Map {
                order,
                coefficients,
            } => {
                Self::push_map_term(order, coefficients, key, coefficient);
            }
        }
    }

    fn push_map_term(
        order: &mut Vec<K>,
        coefficients: &mut FxHashMap<K, S>,
        key: K,
        coefficient: S,
    ) {
        match coefficients.entry(key) {
            Entry::Occupied(mut entry) => {
                let existing = entry.get_mut();
                *existing = existing.clone() + coefficient;
            }
            Entry::Vacant(entry) => {
                order.push(entry.key().clone());
                entry.insert(coefficient);
            }
        }
    }

    pub(crate) fn into_vec(self) -> Vec<(K, S)> {
        match self {
            Self::Empty => Vec::new(),
            Self::Singleton(key, coefficient) => vec![(key, coefficient)],
            Self::Map {
                order,
                mut coefficients,
            } => {
                let mut terms = Vec::with_capacity(order.len());
                for key in order {
                    let coefficient = coefficients
                        .remove(&key)
                        .expect("accumulator order only contains inserted keys");
                    terms.push((key, coefficient));
                }
                terms
            }
        }
    }
}

pub(crate) fn compose_tree_pair_terms<R, F, I>(
    rule: &R,
    terms: Vec<(FusionTreePairKey, R::Scalar)>,
    mut transform: F,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    F: FnMut(&R, &FusionTreePairKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreePairKey, R::Scalar)>,
{
    let mut output = FusionTermAccumulator::new();
    for (key, coefficient) in terms {
        for (next_key, next_coefficient) in transform(rule, &key)? {
            output.push(next_key, coefficient.clone() * next_coefficient);
        }
    }
    Ok(output.into_vec())
}

/// Batched analog of [`compose_tree_pair_terms`]: apply `transform` to every
/// tree-pair of a whole block at once, threading a coefficient *matrix* (a
/// sparse column per original source) instead of re-running the per-source
/// term list. `columns[i]` maps `src index -> coefficient` for `basis[i]`.
///
/// This is the TensorKit 0.17 `artin_braid`/`fsbraid` batching: the elementary
/// step (bend / Artin braid) is walked ONCE for the block and its coefficients
/// are spread across all source columns, so intermediate allocation is
/// O(steps) rather than O(steps × sources) — the term-list style TeNeT used
/// (equivalent to TensorKit ≤0.16's per-tree `FusionTreeDict`) allocated a
/// fresh accumulator and cloned keys per source per step.
/// Dense coefficient matrix (TK's `Matrix{E}`): rows are destination basis
/// trees, columns the original sources. Stored row-major in ONE flat
/// allocation that grows amortized as rows are added, instead of a
/// `Vec<Vec<_>>` that heap-allocs a fresh column per destination tree (the
/// batched braid over a whole block adds hundreds of thousands of rows across
/// its bend/braid steps, so the per-row allocation dominated the cold
/// recoupling build).
pub(crate) struct DenseColumns<S> {
    pub(super) data: Vec<Option<S>>,
    pub(crate) num_src: usize,
    pub(crate) num_rows: usize,
}

/// Ordered block-linear result produced by one categorical block transform.
///
/// This is an internal crate-boundary vocabulary for lowering reduced blocks.
/// Destination keys are unique and ordered by source-major first appearance;
/// columns retain the caller's source order. Structurally absent entries remain
/// distinct from present zero coefficients, and no destination row is wholly
/// absent.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct OrderedBlockLinearMap<K, S> {
    pub(super) destinations: Vec<K>,
    pub(super) source_count: usize,
    pub(super) storage: OrderedBlockLinearStorage<S>,
}

/// Structural storage for an [`OrderedBlockLinearMap`].
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub enum OrderedBlockLinearStorage<S> {
    /// Exactly one structurally present destination for every source column.
    SingletonColumns {
        destination_rows: Vec<usize>,
        coefficients: Vec<S>,
    },
    /// Row-major destination-by-source coefficients.
    DenseDstSrc(Vec<Option<S>>),
}

#[doc(hidden)]
impl<K, S> OrderedBlockLinearMap<K, S> {
    #[inline]
    pub fn destinations(&self) -> &[K] {
        &self.destinations
    }

    #[inline]
    pub fn source_count(&self) -> usize {
        self.source_count
    }

    #[inline]
    pub fn storage(&self) -> &OrderedBlockLinearStorage<S> {
        &self.storage
    }

    #[inline]
    pub fn into_parts(self) -> (Vec<K>, usize, OrderedBlockLinearStorage<S>) {
        (self.destinations, self.source_count, self.storage)
    }
}

impl<S: Clone> DenseColumns<S> {
    pub(crate) fn with_capacity(num_src: usize, rows_hint: usize) -> Self {
        Self {
            data: Vec::with_capacity(rows_hint.saturating_mul(num_src)),
            num_src,
            num_rows: 0,
        }
    }

    /// Append a new all-empty row, returning its index.
    pub(crate) fn push_empty_row(&mut self) -> usize {
        let row = self.num_rows;
        self.data
            .resize_with(self.data.len() + self.num_src, || None);
        self.num_rows += 1;
        row
    }

    #[inline]
    pub(crate) fn row(&self, row: usize) -> &[Option<S>] {
        let start = row * self.num_src;
        &self.data[start..start + self.num_src]
    }

    #[inline]
    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [Option<S>] {
        let start = row * self.num_src;
        &mut self.data[start..start + self.num_src]
    }
}

#[cfg(test)]
pub(super) fn compose_block_terms<R, F, I>(
    rule: &R,
    basis: &[FusionTreePairKey],
    columns: &DenseColumns<R::Scalar>,
    mut transform: F,
) -> Result<(Vec<FusionTreePairKey>, DenseColumns<R::Scalar>), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    F: FnMut(&R, &FusionTreePairKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreePairKey, R::Scalar)>,
{
    let num_src = columns.num_src;
    // Dedup destination tree-pairs to dense rows. The key is *moved* into the
    // map (no per-destination clone — this dedup is the hottest FusionTreeKey
    // clone/eq/hash site on the cold recoupling path); `next_basis` is rebuilt
    // from the map by row index afterwards. Rows are assigned in first-
    // appearance order, so the rebuilt `next_basis` order — and therefore every
    // coefficient — is bit-for-bit identical to pushing the key eagerly.
    let mut index: FxHashMap<FusionTreePairKey, usize> = FxHashMap::default();
    let mut next_columns: DenseColumns<R::Scalar> =
        DenseColumns::with_capacity(num_src, basis.len());
    for (source_row, source_key) in basis.iter().enumerate() {
        for (dst_key, step_coefficient) in transform(rule, source_key)? {
            let row = match index.get(&dst_key) {
                Some(&row) => row,
                None => {
                    let row = next_columns.push_empty_row();
                    index.insert(dst_key, row);
                    row
                }
            };
            // dst_column[src] += step_coefficient * source_column[src] for each
            // source that reaches this basis tree. Source and destination live
            // in different matrices, so the borrows don't overlap.
            let source_column = columns.row(source_row);
            let dst_column = next_columns.row_mut(row);
            for (src, source_coefficient) in source_column.iter().enumerate() {
                let Some(source_coefficient) = source_coefficient else {
                    continue;
                };
                let contribution = step_coefficient.clone() * source_coefficient.clone();
                dst_column[src] = Some(match dst_column[src].take() {
                    Some(existing) => existing + contribution,
                    None => contribution,
                });
            }
        }
    }
    // Rebuild the basis in row order (= first-appearance order). Rows are dense
    // `0..index.len()`, so place each moved key at its row index.
    let mut slots: Vec<Option<FusionTreePairKey>> = (0..index.len()).map(|_| None).collect();
    for (key, row) in index {
        slots[row] = Some(key);
    }
    let next_basis: Vec<FusionTreePairKey> = slots
        .into_iter()
        .map(|key| key.expect("dense rows 0..len are all filled"))
        .collect();
    Ok((next_basis, next_columns))
}

pub(super) struct CompactMultiplicityFreeTreeBasis {
    pub(super) frame: MultiplicityFreeTreeFrame,
    pub(super) locals: Vec<MultiplicityFreeTreeLocal>,
}

impl CompactMultiplicityFreeTreeBasis {
    pub(super) fn from_group<R>(
        group: ValidatedFusionTreeBlockGroup<'_, R>,
    ) -> Result<Self, CoreError> {
        let src_keys = group.src_keys;
        let (frame, first_local) =
            MultiplicityFreeTreeFrame::split(group.projection.tree_at(0).ok_or(
                CoreError::MalformedFusionTree {
                    message: "compact block basis requires at least one source",
                },
            )?);
        let mut locals = Vec::with_capacity(src_keys.len());
        locals.push(first_local);
        for (index, source) in src_keys.iter().enumerate().skip(1) {
            if !frame.matches_tree(source) {
                return Err(CoreError::MalformedFusionTree {
                    message: "fusion-tree keys must share one group",
                });
            }
            locals.push(MultiplicityFreeTreeLocal::from_proven(
                group
                    .projection
                    .tree_at(index)
                    .expect("validated projection covers every source"),
            ));
        }
        Ok(Self { frame, locals })
    }
}

pub(crate) struct CompactMultiplicityFreeTreePairBasis {
    pub(super) frame: MultiplicityFreeTreePairFrame,
    pub(super) locals: Vec<MultiplicityFreeTreePairLocal>,
}

pub(super) struct CompactMultiplicityFreeTreePairBlock<S> {
    pub(super) basis: CompactMultiplicityFreeTreePairBasis,
    pub(super) columns: DenseColumns<S>,
    pub(super) records_dimensions: bool,
}

pub(super) type CompactMultiplicityFreeTreePairRows<S> = Vec<Vec<(FusionTreePairKey, S)>>;

impl CompactMultiplicityFreeTreePairBasis {
    pub(crate) fn from_group<R>(
        group: ValidatedTreePairBlockGroup<'_, R>,
    ) -> Result<Self, CoreError> {
        let (frame, first_local) =
            MultiplicityFreeTreePairFrame::split(group.projection.pair_at(0).ok_or(
                CoreError::MalformedFusionTree {
                    message: "compact block basis requires at least one source",
                },
            )?);
        let mut locals = Vec::with_capacity(group.source_len);
        locals.push(first_local);
        for index in 1..group.source_len {
            let source = group
                .projection
                .pair_at(index)
                .expect("validated projection covers every source");
            if !frame.matches_tree_pair_ref(source) {
                return Err(CoreError::MalformedFusionTree {
                    message: TREE_PAIR_BLOCK_GROUP_ERROR,
                });
            }
            locals.push(MultiplicityFreeTreePairLocal::from_proven(source));
        }
        Ok(Self { frame, locals })
    }
}

pub(super) fn compose_compact_block_terms<R, K, F, I>(
    rule: &R,
    basis: &[K],
    columns: &DenseColumns<R::Scalar>,
    mut transform: F,
) -> Result<(Vec<K>, DenseColumns<R::Scalar>), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    K: Eq + Hash,
    F: FnMut(&R, &K) -> Result<I, CoreError>,
    I: IntoIterator<Item = (K, R::Scalar)>,
{
    let num_src = columns.num_src;
    let mut index: FxHashMap<K, usize> = FxHashMap::default();
    let mut next_columns = DenseColumns::with_capacity(num_src, basis.len());
    for (source_row, source_local) in basis.iter().enumerate() {
        for (destination_local, step_coefficient) in transform(rule, source_local)? {
            let row = match index.get(&destination_local) {
                Some(&row) => row,
                None => {
                    let row = next_columns.push_empty_row();
                    index.insert(destination_local, row);
                    row
                }
            };
            let source_column = columns.row(source_row);
            let destination_column = next_columns.row_mut(row);
            for (src, source_coefficient) in source_column.iter().enumerate() {
                let Some(source_coefficient) = source_coefficient else {
                    continue;
                };
                let contribution = step_coefficient.clone() * source_coefficient.clone();
                destination_column[src] = Some(match destination_column[src].take() {
                    Some(existing) => existing + contribution,
                    None => contribution,
                });
            }
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
    Ok((locals, next_columns))
}

pub(crate) fn compose_generic_block_terms<R, F, I>(
    rule: &R,
    basis: &[FusionTreePairKey],
    columns: &DenseColumns<R::Scalar>,
    mut transform: F,
) -> Result<(Vec<FusionTreePairKey>, DenseColumns<R::Scalar>), CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
    F: FnMut(&R, &FusionTreePairKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreePairKey, R::Scalar)>,
{
    let num_src = columns.num_src;
    let mut index: FxHashMap<FusionTreePairKey, usize> = FxHashMap::default();
    let mut next_columns = DenseColumns::with_capacity(num_src, basis.len());
    for (source_row, source_key) in basis.iter().enumerate() {
        for (destination_key, step_coefficient) in transform(rule, source_key)? {
            let row = match index.get(&destination_key) {
                Some(&row) => row,
                None => {
                    let row = next_columns.push_empty_row();
                    index.insert(destination_key, row);
                    row
                }
            };
            let source_column = columns.row(source_row);
            let destination_column = next_columns.row_mut(row);
            for (source, source_coefficient) in source_column.iter().enumerate() {
                let Some(source_coefficient) = source_coefficient else {
                    continue;
                };
                let contribution = step_coefficient.clone() * source_coefficient.clone();
                destination_column[source] = Some(match destination_column[source].take() {
                    Some(existing) => existing + contribution,
                    None => contribution,
                });
            }
        }
    }
    let mut slots: Vec<Option<FusionTreePairKey>> = (0..index.len()).map(|_| None).collect();
    for (key, row) in index {
        slots[row] = Some(key);
    }
    let basis = slots
        .into_iter()
        .map(|key| key.expect("dense rows 0..len are all filled"))
        .collect();
    Ok((basis, next_columns))
}

fn seed_generic_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
) -> Result<(Vec<FusionTreePairKey>, DenseColumns<R::Scalar>), CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    if rule.fusion_style() != FusionStyleKind::Generic {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let mut basis = Vec::with_capacity(src_keys.len());
    for source in src_keys {
        validate_fusion_tree_pair_for_rule(rule, source)?;
        basis.push(source.clone());
    }
    let mut columns = DenseColumns::with_capacity(src_keys.len(), src_keys.len());
    for source in 0..src_keys.len() {
        let row = columns.push_empty_row();
        columns.row_mut(row)[source] = Some(R::Scalar::one());
    }
    Ok((basis, columns))
}

fn order_generic_tree_pair_block<S: Clone>(
    basis: Vec<FusionTreePairKey>,
    columns: DenseColumns<S>,
) -> OrderedBlockLinearMap<FusionTreePairKey, S> {
    let source_count = columns.num_src;
    let mut ordered_basis_rows = Vec::with_capacity(basis.len());
    let mut ordered_row_for_basis = vec![usize::MAX; basis.len()];
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

    let destinations = ordered_basis_rows
        .iter()
        .map(|&basis_row| basis[basis_row].clone())
        .collect::<Vec<_>>();
    let storage = if is_singleton {
        let mut destination_rows = Vec::with_capacity(source_count);
        let mut coefficients = Vec::with_capacity(source_count);
        for (source, basis_row) in singleton_basis_rows.into_iter().enumerate() {
            destination_rows.push(ordered_row_for_basis[basis_row]);
            coefficients.push(
                columns.data[basis_row * source_count + source]
                    .clone()
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
                    .iter()
                    .cloned(),
            );
        }
        OrderedBlockLinearStorage::DenseDstSrc(coefficients)
    };

    OrderedBlockLinearMap {
        destinations,
        source_count: columns.num_src,
        storage,
    }
}

fn generic_repartition_tree_pair_block_terms<R>(
    rule: &R,
    mut basis: Vec<FusionTreePairKey>,
    mut columns: DenseColumns<R::Scalar>,
    target_codomain_rank: usize,
) -> Result<(Vec<FusionTreePairKey>, DenseColumns<R::Scalar>), CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let Some(first) = basis.first() else {
        return Ok((basis, columns));
    };
    let total_rank =
        first.codomain_tree().uncoupled().len() + first.domain_tree().uncoupled().len();
    if target_codomain_rank > total_rank {
        return Err(CoreError::DimensionMismatch {
            expected: total_rank,
            actual: target_codomain_rank,
        });
    }
    let mut current_codomain_rank = first.codomain_tree().uncoupled().len();
    while current_codomain_rank < target_codomain_rank {
        (basis, columns) = compose_generic_block_terms(rule, &basis, &columns, |rule, key| {
            generic_bendleft_tree_pair(rule, key)
        })?;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        (basis, columns) = compose_generic_block_terms(rule, &basis, &columns, |rule, key| {
            generic_bendright_tree_pair(rule, key)
        })?;
        current_codomain_rank -= 1;
    }
    Ok((basis, columns))
}

#[doc(hidden)]
pub fn generic_braid_tree_pair_block_ordered<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    if src_keys.is_empty() {
        return Ok(order_generic_tree_pair_block(
            Vec::new(),
            DenseColumns::with_capacity(0, 0),
        ));
    }
    let codomain_rank = src_keys[0].codomain_tree().uncoupled().len();
    let domain_rank = src_keys[0].domain_tree().uncoupled().len();
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
    for source in src_keys {
        if source.codomain_tree().uncoupled().len() != codomain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: codomain_rank,
                actual: source.codomain_tree().uncoupled().len(),
            });
        }
        if source.domain_tree().uncoupled().len() != domain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: domain_rank,
                actual: source.domain_tree().uncoupled().len(),
            });
        }
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

    let (mut basis, mut columns) = seed_generic_tree_pair_block(rule, src_keys)?;
    if identity {
        return Ok(order_generic_tree_pair_block(basis, columns));
    }
    let all_rank = permutation.len();
    (basis, columns) = generic_repartition_tree_pair_block_terms(rule, basis, columns, all_rank)?;
    (basis, columns) = compose_generic_block_terms(rule, &basis, &columns, |rule, key| {
        generic_braid_tree_unchecked(rule, key.codomain_tree(), &permutation, &levels, &swaps).map(
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
    (basis, columns) = generic_repartition_tree_pair_block_terms(
        rule,
        basis,
        columns,
        codomain_permutation.len(),
    )?;
    Ok(order_generic_tree_pair_block(basis, columns))
}

#[doc(hidden)]
pub fn generic_permute_tree_pair_block_ordered<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
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
    let Some(first) = src_keys.first() else {
        return Ok(order_generic_tree_pair_block(
            Vec::new(),
            DenseColumns::with_capacity(0, 0),
        ));
    };
    let codomain_rank = first.codomain_tree().uncoupled().len();
    let domain_rank = first.domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    generic_braid_tree_pair_block_ordered(
        rule,
        src_keys,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

#[doc(hidden)]
pub fn generic_transpose_tree_pair_block_ordered<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    if src_keys.is_empty() {
        return Ok(order_generic_tree_pair_block(
            Vec::new(),
            DenseColumns::with_capacity(0, 0),
        ));
    }
    let codomain_rank = src_keys[0].codomain_tree().uncoupled().len();
    let domain_rank = src_keys[0].domain_tree().uncoupled().len();
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
    for source in src_keys {
        if source.codomain_tree().uncoupled().len() != codomain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: codomain_rank,
                actual: source.codomain_tree().uncoupled().len(),
            });
        }
        if source.domain_tree().uncoupled().len() != domain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: domain_rank,
                actual: source.domain_tree().uncoupled().len(),
            });
        }
    }
    let mut position = match permutation.iter().position(|&axis| axis == 0) {
        Some(position) => position,
        None => {
            let (basis, columns) = seed_generic_tree_pair_block(rule, src_keys)?;
            return Ok(order_generic_tree_pair_block(basis, columns));
        }
    };
    let total_rank = codomain_rank + domain_rank;
    let (mut basis, mut columns) = seed_generic_tree_pair_block(rule, src_keys)?;
    (basis, columns) = generic_repartition_tree_pair_block_terms(
        rule,
        basis,
        columns,
        codomain_permutation.len(),
    )?;
    if total_rank == 0 || position == 0 {
        return Ok(order_generic_tree_pair_block(basis, columns));
    }

    let half_rank = total_rank >> 1;
    while position > 0 && position < half_rank {
        (basis, columns) = compose_generic_block_terms(rule, &basis, &columns, |rule, key| {
            generic_cycle_anticlockwise_tree_pair_unchecked(rule, key)
        })?;
        position -= 1;
    }
    while position >= half_rank && position > 0 {
        (basis, columns) = compose_generic_block_terms(rule, &basis, &columns, |rule, key| {
            generic_cycle_clockwise_tree_pair_unchecked(rule, key)
        })?;
        position = (position + 1) % total_rank;
    }

    Ok(order_generic_tree_pair_block(basis, columns))
}
