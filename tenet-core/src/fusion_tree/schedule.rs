//! The one schedule layer for fusion-tree moves.
//!
//! Every tree-pair operation is a fixed sequence of local moves: bends for a
//! repartition, fold-and-bend for a cyclic transpose, and adjacent Artin swaps
//! inside an all-codomain tree for a braid. This module owns those sequences
//! once. Multiplicity-free, unique-fusion and Generic callers supply only the
//! move itself; the representation it acts on (a term list, a single term, or
//! a column-batched block) is the caller's.
//!
//! TensorKit composes the same sequences (`duality_manipulations.jl`
//! `repartition` :460-505, `cycleclockwise` :401, `cycleanticlockwise` :431,
//! `transpose` :518; `braiding_manipulations.jl` `fsbraid` :302-331), with the
//! fusion-style branch inside each move's coefficient.

use super::*;

/// Direction of one repartition bend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Bend {
    /// Move the last domain leg to the codomain (`bendleft`).
    Left,
    /// Move the last codomain leg to the domain (`bendright`).
    Right,
}

/// Compose a term list with one move: TensorKit's `U = U_tmp * U` on a sparse
/// list. Destinations reached twice are summed in first-appearance order.
pub(crate) fn compose_terms<S, E, F, I>(
    terms: Vec<(FusionTreePairKey, S)>,
    mut transform: F,
) -> Result<Vec<(FusionTreePairKey, S)>, E>
where
    S: Clone + Add<Output = S> + Mul<Output = S>,
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

/// The terms one move emits for one tree pair.
pub(super) trait StepTerms<S> {
    type Terms: Iterator<Item = (FusionTreePairKey, S)>;
    fn into_terms(self) -> Self::Terms;
}

impl<S> StepTerms<S> for Vec<(FusionTreePairKey, S)> {
    type Terms = std::vec::IntoIter<(FusionTreePairKey, S)>;
    fn into_terms(self) -> Self::Terms {
        self.into_iter()
    }
}

impl<S, A> StepTerms<S> for SmallVec<A>
where
    A: smallvec::Array<Item = (FusionTreePairKey, S)>,
{
    type Terms = smallvec::IntoIter<A>;
    fn into_terms(self) -> Self::Terms {
        self.into_iter()
    }
}

impl<S> StepTerms<S> for (FusionTreePairKey, S) {
    type Terms = std::iter::Once<(FusionTreePairKey, S)>;
    fn into_terms(self) -> Self::Terms {
        std::iter::once(self)
    }
}

/// A running tree-pair transform: a term list, or one term for unique fusion,
/// where every move is a bijection and no list is ever allocated.
pub(super) trait TreePairTerms<S>: Sized {
    /// The terms of the first move of a sequence (no `one` seed is multiplied).
    fn first_step<I: StepTerms<S>>(step: I) -> Result<Self, CoreError>;
    /// Apply the next move to every term and multiply `coefficient · step`.
    fn then<E, F, I>(self, step: F) -> Result<Self, E>
    where
        E: From<CoreError>,
        F: FnMut(&FusionTreePairKey) -> Result<I, E>,
        I: StepTerms<S>;
    /// Exchange codomain and domain of every term and conjugate it, the
    /// TensorKit `foldleft`/`bendleft` "through (f₂, f₁) => conj(coeff)" rule.
    fn swap_sides_conj(self) -> Self;
}

fn swap_sides(key: FusionTreePairKey) -> FusionTreePairKey {
    FusionTreePairKey::pair(key.domain_tree().clone(), key.codomain_tree().clone())
}

impl<S: CategoricalScalar> TreePairTerms<S> for Vec<(FusionTreePairKey, S)> {
    fn first_step<I: StepTerms<S>>(step: I) -> Result<Self, CoreError> {
        Ok(step.into_terms().collect())
    }

    fn then<E, F, I>(self, mut step: F) -> Result<Self, E>
    where
        E: From<CoreError>,
        F: FnMut(&FusionTreePairKey) -> Result<I, E>,
        I: StepTerms<S>,
    {
        compose_terms(self, |key| step(key).map(StepTerms::into_terms))
    }

    fn swap_sides_conj(self) -> Self {
        self.into_iter()
            .map(|(key, coefficient)| (swap_sides(key), coefficient.conj()))
            .collect()
    }
}

impl<S: CategoricalScalar> TreePairTerms<S> for (FusionTreePairKey, S) {
    fn first_step<I: StepTerms<S>>(step: I) -> Result<Self, CoreError> {
        step.into_terms()
            .next()
            .ok_or(CoreError::MalformedFusionTree {
                message: "a unique-fusion move emits exactly one tree pair",
            })
    }

    fn then<E, F, I>(self, mut step: F) -> Result<Self, E>
    where
        E: From<CoreError>,
        F: FnMut(&FusionTreePairKey) -> Result<I, E>,
        I: StepTerms<S>,
    {
        let (key, coefficient) = self;
        let (next, step_coefficient) = Self::first_step(step(&key)?)?;
        Ok((next, coefficient * step_coefficient))
    }

    fn swap_sides_conj(self) -> Self {
        let (key, coefficient) = self;
        (swap_sides(key), coefficient.conj())
    }
}

/// Bend legs between codomain and domain until the codomain has `target`
/// legs (TensorKit `repartition`, `duality_manipulations.jl:460-505`).
pub(super) fn repartition_loop<T, E>(
    mut current: T,
    mut codomain_rank: usize,
    target: usize,
    mut bend: impl FnMut(T, Bend) -> Result<T, E>,
) -> Result<T, E> {
    while codomain_rank < target {
        current = bend(current, Bend::Left)?;
        codomain_rank += 1;
    }
    while codomain_rank > target {
        current = bend(current, Bend::Right)?;
        codomain_rank -= 1;
    }
    Ok(current)
}

/// A left move (`foldleft`, or Generic `bendleft`) as its right move through
/// the swapped pair, conjugated (`duality_manipulations.jl:140-147, 315`).
pub(super) fn left_move_by_swap<T, S, E>(
    tree_pair: &FusionTreePairKey,
    right_move: impl FnOnce(&FusionTreePairKey) -> Result<T, E>,
) -> Result<T, E>
where
    T: TreePairTerms<S>,
{
    let swapped = FusionTreePairKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(right_move(&swapped)?.swap_sides_conj())
}

/// `cycleclockwise` (`duality_manipulations.jl:401`): foldright then bendleft,
/// or bendleft then foldright when the codomain is empty. `cycleanticlockwise`
/// (`:431`): foldleft then bendright, or bendright then foldleft when the
/// domain is empty; foldleft is foldright through the swapped pair.
pub(super) fn cycle<T, S, E, B, F, IB, IF>(
    tree_pair: &FusionTreePairKey,
    direction: PreparedCycleDirection,
    mut bend: B,
    mut foldright: F,
) -> Result<T, E>
where
    T: TreePairTerms<S>,
    E: From<CoreError>,
    B: FnMut(&FusionTreePairKey, Bend) -> Result<IB, E>,
    F: FnMut(&FusionTreePairKey) -> Result<IF, E>,
    IB: StepTerms<S>,
    IF: StepTerms<S> + TreePairTerms<S>,
{
    match direction {
        PreparedCycleDirection::Clockwise => {
            if tree_pair.codomain_tree().uncoupled().is_empty() {
                T::first_step(bend(tree_pair, Bend::Left)?)?.then(foldright)
            } else {
                T::first_step(foldright(tree_pair)?)?.then(|key| bend(key, Bend::Left))
            }
        }
        PreparedCycleDirection::Anticlockwise => {
            let foldleft = |key: &FusionTreePairKey| left_move_by_swap(key, &mut foldright);
            if tree_pair.domain_tree().uncoupled().is_empty() {
                T::first_step(bend(tree_pair, Bend::Right)?)?.then(foldleft)
            } else {
                let mut foldleft = foldleft;
                T::first_step(foldleft(tree_pair)?)?.then(|key| bend(key, Bend::Right))
            }
        }
    }
}

/// The cyclic rotation a planar transpose applies after repartitioning, for
/// the output position of source axis 0 (`duality_manipulations.jl:518`):
/// rotate the shorter way round, or not at all.
pub(super) fn transpose_cycles(
    position: usize,
    total_rank: usize,
) -> Option<(PreparedCycleDirection, usize)> {
    if position == 0 {
        None
    } else if position < total_rank >> 1 {
        Some((PreparedCycleDirection::Anticlockwise, position))
    } else {
        Some((PreparedCycleDirection::Clockwise, total_rank - position))
    }
}

pub(super) fn run_cycles<T, E>(
    mut current: T,
    cycles: Option<(PreparedCycleDirection, usize)>,
    mut cycle: impl FnMut(T, PreparedCycleDirection) -> Result<T, E>,
) -> Result<T, E> {
    if let Some((direction, count)) = cycles {
        for _ in 0..count {
            current = cycle(current, direction)?;
        }
    }
    Ok(current)
}

/// Apply a prepared Artin schedule one adjacent swap at a time
/// (`braiding_manipulations.jl` `braid`: `for s in permutation2swaps(p)`),
/// for term lists, single unique-fusion states and the compact tree block.
/// The compact pair-block braid runs its own plain loop in
/// `CompactTreePairDriver::braid_codomain` (a step closure measured slower on
/// the cold braid plan build).
#[inline]
pub(super) fn run_artin_steps<T, E>(
    mut state: T,
    steps: impl IntoIterator<Item = PreparedArtinStep>,
    mut step: impl FnMut(T, PreparedArtinStep) -> Result<T, E>,
) -> Result<T, E> {
    for artin in steps {
        state = step(state, artin)?;
    }
    Ok(state)
}

/// The adjacent-swap braid of one tree as a term list, with `one` as the seed.
pub(super) fn braid_tree_steps<S, E, I, F, J>(
    tree: &FusionTreeKey,
    steps: I,
    mut artin: F,
) -> Result<Vec<(FusionTreeKey, S)>, E>
where
    S: CategoricalScalar,
    I: IntoIterator<Item = PreparedArtinStep>,
    F: FnMut(&FusionTreeKey, PreparedArtinStep) -> Result<J, E>,
    J: IntoIterator<Item = (FusionTreeKey, S)>,
{
    run_artin_steps(vec![(tree.clone(), S::one())], steps, |current, step| {
        let mut next_terms = FusionTermAccumulator::new();
        for (tree, coefficient) in current {
            for (next_tree, step_coefficient) in artin(&tree, step)? {
                next_terms.push(next_tree, coefficient.clone() * step_coefficient);
            }
        }
        Ok(next_terms.into_vec())
    })
}

/// Rotate a running transform `count` times in `direction`, applying the
/// clockwise or anticlockwise cycle move to every term.
pub(super) fn run_cycle_terms<T, S, E, C, I>(
    current: T,
    cycles: Option<(PreparedCycleDirection, usize)>,
    mut cycle: C,
) -> Result<T, E>
where
    T: TreePairTerms<S>,
    E: From<CoreError>,
    C: FnMut(&FusionTreePairKey, PreparedCycleDirection) -> Result<I, E>,
    I: StepTerms<S>,
{
    run_cycles(current, cycles, |terms, direction| {
        terms.then(|key| cycle(key, direction))
    })
}

/// The middle and last steps of `fsbraid` (`braiding_manipulations.jl:302-309`)
/// on an all-codomain transform: braid every term's codomain tree, then
/// repartition back with `back`.
pub(super) fn braid_via_codomain<T, S, E, B, I>(
    all_codomain: T,
    braid: B,
    back: impl FnOnce(T) -> Result<T, E>,
) -> Result<T, E>
where
    T: TreePairTerms<S>,
    E: From<CoreError>,
    B: FnMut(&FusionTreePairKey) -> Result<I, E>,
    I: StepTerms<S>,
{
    back(all_codomain.then(braid)?)
}

/// Attach each braided codomain tree to the term's unchanged domain tree.
pub(super) fn with_domain<S>(
    key: &FusionTreePairKey,
    codomain_terms: Vec<(FusionTreeKey, S)>,
) -> Vec<(FusionTreePairKey, S)> {
    codomain_terms
        .into_iter()
        .map(|(codomain_tree, coefficient)| {
            (
                FusionTreePairKey::pair(codomain_tree, key.domain_tree().clone()),
                coefficient,
            )
        })
        .collect()
}

/// Braid a tree-pair term list the way `fsbraid` does
/// (`braiding_manipulations.jl:302-309`): repartition every leg into the
/// codomain, braid that tree, and repartition to the target split.
pub(super) fn braid_terms_via_codomain<S, E, B, C>(
    terms: Vec<(FusionTreePairKey, S)>,
    codomain_rank: usize,
    all_rank: usize,
    target_codomain_rank: usize,
    mut bend: B,
    mut braid_codomain: C,
) -> Result<Vec<(FusionTreePairKey, S)>, E>
where
    S: CategoricalScalar,
    E: From<CoreError>,
    B: FnMut(Vec<(FusionTreePairKey, S)>, Bend) -> Result<Vec<(FusionTreePairKey, S)>, E>,
    C: FnMut(&FusionTreeKey) -> Result<Vec<(FusionTreeKey, S)>, E>,
{
    let all_codomain = repartition_loop(terms, codomain_rank, all_rank, &mut bend)?;
    braid_via_codomain(
        all_codomain,
        |key| braid_codomain(key.codomain_tree()).map(|terms| with_domain(key, terms)),
        |braided| repartition_loop(braided, all_rank, target_codomain_rank, bend),
    )
}

/// One step of a column-batched block schedule. The multiplicity-free compact
/// driver and the Generic keyed-block driver implement it; the block braid and
/// transpose sequences below are shared.
pub(super) trait BlockDriver {
    type State;
    type Error;
    /// What one block braid needs besides the state: the prepared Artin
    /// schedule.
    type BraidSchedule<'s>;
    fn bend(&mut self, state: Self::State, bend: Bend) -> Result<Self::State, Self::Error>;
    fn braid_codomain(
        &mut self,
        state: Self::State,
        schedule: Self::BraidSchedule<'_>,
    ) -> Result<Self::State, Self::Error>;
    fn cycle(
        &mut self,
        state: Self::State,
        direction: PreparedCycleDirection,
    ) -> Result<Self::State, Self::Error>;
}

/// Block `fsbraid` (`braiding_manipulations.jl:317-331`): repartition into
/// the codomain, braid it with the driver's schedule, repartition to the
/// target split.
pub(super) fn block_braid<D: BlockDriver>(
    driver: &mut D,
    state: D::State,
    codomain_rank: usize,
    all_rank: usize,
    target_codomain_rank: usize,
    schedule: D::BraidSchedule<'_>,
) -> Result<D::State, D::Error> {
    let state = repartition_loop(state, codomain_rank, all_rank, |state, bend| {
        driver.bend(state, bend)
    })?;
    let state = driver.braid_codomain(state, schedule)?;
    repartition_loop(state, all_rank, target_codomain_rank, |state, bend| {
        driver.bend(state, bend)
    })
}

/// Block planar transpose: repartition, then rotate.
pub(super) fn block_transpose<D: BlockDriver>(
    driver: &mut D,
    state: D::State,
    codomain_rank: usize,
    target_codomain_rank: usize,
    cycles: Option<(PreparedCycleDirection, usize)>,
) -> Result<D::State, D::Error> {
    let state = repartition_loop(state, codomain_rank, target_codomain_rank, |state, bend| {
        driver.bend(state, bend)
    })?;
    run_cycles(state, cycles, |state, direction| {
        driver.cycle(state, direction)
    })
}
