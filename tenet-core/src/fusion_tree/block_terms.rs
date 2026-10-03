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

/// Batched analog of [`compose_terms`]: apply `transform` to every
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
/// columns retain the caller's source order. The order of the destinations a
/// single source first reaches is deterministic for identical inputs but
/// otherwise unspecified (it is neither HomSpace order nor stable across
/// algorithm changes), so consumers must resolve destinations by key, never by
/// position. Structurally absent entries remain distinct from present zero
/// coefficients, and no destination row is wholly absent.
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

impl<S> DenseColumns<S> {
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

/// Compose a column-batched block with one move: every basis row's move
/// terms are spread across all source columns (`dst += step · source`). The
/// one block composer for multiplicity-free keys, compact locals and Generic
/// keys.
pub(crate) fn compose_block_terms<K, S, E, F, I>(
    basis: &[K],
    columns: &DenseColumns<S>,
    mut transform: F,
) -> Result<(Vec<K>, DenseColumns<S>), E>
where
    K: Eq + Hash,
    S: Clone + Add<Output = S> + Mul<Output = S>,
    F: FnMut(&K) -> Result<I, E>,
    I: IntoIterator<Item = (K, S)>,
{
    let num_src = columns.num_src;
    // Dedup destination tree-pairs to dense rows. The key is *moved* into the
    // map (no per-destination clone — this dedup is the hottest FusionTreeKey
    // clone/eq/hash site on the cold recoupling path); `next_basis` is rebuilt
    // from the map by row index afterwards. Rows are assigned in first-
    // appearance order, so the rebuilt `next_basis` order — and therefore every
    // coefficient — is bit-for-bit identical to pushing the key eagerly.
    let mut index: FxHashMap<K, usize> = FxHashMap::default();
    let mut next_columns: DenseColumns<S> = DenseColumns::with_capacity(num_src, basis.len());
    for (source_row, source_key) in basis.iter().enumerate() {
        for (dst_key, step_coefficient) in transform(source_key)? {
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
    let mut slots: Vec<Option<K>> = (0..index.len()).map(|_| None).collect();
    for (key, row) in index {
        slots[row] = Some(key);
    }
    let next_basis: Vec<K> = slots
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

fn order_generic_tree_pair_block<S>(
    basis: Vec<FusionTreePairKey>,
    columns: DenseColumns<S>,
) -> OrderedBlockLinearMap<FusionTreePairKey, S> {
    let mut slots = basis.into_iter().map(Some).collect::<Vec<_>>();
    order_block_columns(slots.len(), columns, |basis_row| {
        slots[basis_row]
            .take()
            .expect("ordered block rows contain each basis row once")
    })
}

/// The Generic keyed-block driver of the shared block schedule.
struct GenericTreePairBlockDriver<'a, R> {
    rule: &'a R,
}

impl<R> BlockDriver for GenericTreePairBlockDriver<'_, R>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    type State = (Vec<FusionTreePairKey>, DenseColumns<R::Scalar>);
    type Error = CoreError;
    type BraidSchedule<'s> = (&'s [usize], &'s [PreparedArtinStep]);

    fn bend(
        &mut self,
        (basis, columns): Self::State,
        bend: Bend,
    ) -> Result<Self::State, CoreError> {
        let rule = self.rule;
        compose_block_terms(&basis, &columns, |key| match bend {
            Bend::Left => generic_bendleft_tree_pair(rule, key),
            Bend::Right => generic_bendright_tree_pair(rule, key),
        })
    }

    fn braid_codomain(
        &mut self,
        (basis, columns): Self::State,
        (permutation, steps): (&[usize], &[PreparedArtinStep]),
    ) -> Result<Self::State, CoreError> {
        let rule = self.rule;
        compose_block_terms(&basis, &columns, |key| {
            generic_braid_tree_unchecked(rule, key.codomain_tree(), permutation, steps)
                .map(|terms| with_domain(key, terms))
        })
    }

    fn cycle(
        &mut self,
        (basis, columns): Self::State,
        direction: PreparedCycleDirection,
    ) -> Result<Self::State, CoreError> {
        let rule = self.rule;
        compose_block_terms(&basis, &columns, |key| match direction {
            PreparedCycleDirection::Clockwise => {
                generic_cycle_clockwise_tree_pair_unchecked(rule, key)
            }
            PreparedCycleDirection::Anticlockwise => {
                generic_cycle_anticlockwise_tree_pair_unchecked(rule, key)
            }
        })
    }
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
    let GenericBraidSchedule {
        permutation,
        identity,
        steps,
    } = generic_braid_schedule(
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )?;

    let (mut basis, mut columns) = seed_generic_tree_pair_block(rule, src_keys)?;
    if identity {
        return Ok(order_generic_tree_pair_block(basis, columns));
    }
    (basis, columns) = block_braid(
        &mut GenericTreePairBlockDriver { rule },
        (basis, columns),
        codomain_rank,
        permutation.len(),
        codomain_permutation.len(),
        (&permutation, &steps),
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
    let Some(position) = permutation.iter().position(|&axis| axis == 0) else {
        let (basis, columns) = seed_generic_tree_pair_block(rule, src_keys)?;
        return Ok(order_generic_tree_pair_block(basis, columns));
    };
    let total_rank = codomain_rank + domain_rank;
    let (basis, columns) = seed_generic_tree_pair_block(rule, src_keys)?;
    let (basis, columns) = block_transpose(
        &mut GenericTreePairBlockDriver { rule },
        (basis, columns),
        codomain_rank,
        codomain_permutation.len(),
        transpose_cycles(position, total_rank),
    )?;
    Ok(order_generic_tree_pair_block(basis, columns))
}
