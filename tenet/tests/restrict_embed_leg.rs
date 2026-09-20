//! Semantics of `TensorMap::restrict_leg` / `embed_leg` (#1285).
//!
//! Two oracles, both independent of the restriction kernel under test:
//!
//! * a **dense gather**: `to_physical_dense` of the result must equal a
//!   per-sector gather of `to_physical_dense` of the source, with the dense
//!   offsets of the restricted leg recomputed (TeNeT's physical order is
//!   sector, then degeneracy, then carrier index);
//! * an **inclusion isometry**: `ι_σ` hand-filled with `from_block_fn`, padded
//!   with `TensorMap::id` through `otimes`, and applied with `compose`.
//!   `contract` is deliberately not used: it applies the fermionic supertrace
//!   twist to every dual contracted right-hand axis, so it would test a
//!   different map on odd sectors.

use std::ops::Range;
use std::sync::Arc;

use num_complex::Complex64;
use tenet::core::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep, ZNFusionRule,
};
use tenet::prelude::{GradedSpace, LegSelection, Runtime, TensorMap};

/// `(degeneracy, carrier dimension)` per sector, in the leg's stored order.
type AxisLayout = Vec<(usize, usize)>;

/// Destination dense index -> source dense index for one axis.
///
/// `selection[i]` is the kept degeneracy range of the `i`-th stored sector,
/// `None` when that sector is dropped entirely.
fn axis_gather(layout: &AxisLayout, selection: &[Option<Range<usize>>]) -> Vec<usize> {
    let mut map = Vec::new();
    let mut offset = 0;
    for (position, &(degeneracy, carrier)) in layout.iter().enumerate() {
        if let Some(range) = &selection[position] {
            for index in range.clone() {
                for basis in 0..carrier {
                    map.push(offset + index * carrier + basis);
                }
            }
        }
        offset += degeneracy * carrier;
    }
    map
}

fn identity_gather(layout: &AxisLayout) -> Vec<usize> {
    (0..layout
        .iter()
        .map(|&(degeneracy, carrier)| degeneracy * carrier)
        .sum())
        .collect()
}

/// Column-major gather of `data` along every axis.
fn gather<D: Copy>(shape: &[usize], data: &[D], maps: &[Vec<usize>]) -> (Vec<usize>, Vec<D>) {
    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    }
    let out_shape: Vec<usize> = maps.iter().map(Vec::len).collect();
    let total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..total {
        let source: usize = index
            .iter()
            .enumerate()
            .map(|(axis, &position)| maps[axis][position] * strides[axis])
            .sum();
        out.push(data[source]);
        for axis in 0..index.len() {
            index[axis] += 1;
            if index[axis] < out_shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    (out_shape, out)
}

/// `data` with every entry outside the kept index sets replaced by zero.
fn masked(shape: &[usize], data: &[f64], maps: &[Vec<usize>]) -> Vec<f64> {
    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    }
    let mut out = vec![0.0; data.len()];
    let total: usize = maps.iter().map(Vec::len).product();
    let out_shape: Vec<usize> = maps.iter().map(Vec::len).collect();
    let mut index = vec![0usize; shape.len()];
    for _ in 0..total {
        let position: usize = index
            .iter()
            .enumerate()
            .map(|(axis, &offset)| maps[axis][offset] * strides[axis])
            .sum();
        out[position] = data[position];
        for axis in 0..index.len() {
            index[axis] += 1;
            if index[axis] < out_shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    out
}

/// `source` placed into a zero buffer of `shape` at the gathered indices: the
/// dense form of an embedding.
fn scattered<D: Copy>(shape: &[usize], source: &[D], maps: &[Vec<usize>], zero: D) -> Vec<D> {
    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    }
    let out_shape: Vec<usize> = maps.iter().map(Vec::len).collect();
    let mut out = vec![zero; shape.iter().product()];
    let total: usize = out_shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    for &value in source.iter().take(total) {
        let destination: usize = index
            .iter()
            .enumerate()
            .map(|(axis, &offset)| maps[axis][offset] * strides[axis])
            .sum();
        out[destination] = value;
        for axis in 0..index.len() {
            index[axis] += 1;
            if index[axis] < out_shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    out
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length");
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (left - right).abs() <= 1e-12 * right.abs().max(1.0),
            "entry {index}: {left} != {right}"
        );
    }
}

fn assert_close_complex(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len(), "length");
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (left - right).norm() <= 1e-12 * right.norm().max(1.0),
            "entry {index}: {left} != {right}"
        );
    }
}

fn u1(provider: &Arc<U1FusionRule>, pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn su2(provider: &Arc<SU2FusionRule>, pairs: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(twice_spin, degeneracy)| (SU2Irrep::from_twice_spin(twice_spin), degeneracy)),
    )
    .unwrap()
}

fn fz2(
    provider: &Arc<FermionParityFusionRule>,
    pairs: &[(bool, usize)],
) -> GradedSpace<FermionParityFusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(odd, degeneracy)| (if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN }, degeneracy)),
    )
    .unwrap()
}

/// The stored-order `(degeneracy, carrier)` layout and the per-position
/// selection ranges the dense oracle needs.
fn dense_plan<S: PartialEq>(
    sectors: &[S],
    degeneracies: &[usize],
    carrier: impl Fn(&S) -> usize,
    selected: &[(S, Range<usize>)],
) -> (AxisLayout, Vec<Option<Range<usize>>>) {
    let layout: AxisLayout = sectors
        .iter()
        .zip(degeneracies)
        .map(|(sector, &degeneracy)| (degeneracy, carrier(sector)))
        .collect();
    let selection = sectors
        .iter()
        .map(|sector| {
            selected
                .iter()
                .find(|(candidate, _)| candidate == sector)
                .map(|(_, range)| range.clone())
        })
        .collect();
    (layout, selection)
}

#[test]
fn restrict_matches_the_dense_gather_on_a_dual_u1_codomain_leg() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    // Degeneracies asymmetric under c -> -c, so a label/dual mix-up cannot
    // survive by coincidence.
    let leg = u1(&provider, &[(-1, 2), (0, 3), (1, 4)])
        .try_dual()
        .unwrap();
    let other = u1(&provider, &[(-1, 1), (0, 2), (1, 3)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &other], [&other], 7).unwrap();

    // Selection in the dual leg's own labels: the caller never dualises.
    let selected = [(U1Irrep::new(-1), 1..2), (U1Irrep::new(1), 0..2)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let restricted = source.restrict_leg(0, &selection).unwrap();

    assert_eq!(restricted.codomain()[0], *selection.subspace());
    assert_eq!(restricted.codomain()[1], other);
    assert_eq!(restricted.domain()[0], other);
    assert!(restricted.codomain()[0].is_dual());

    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        |_| 1,
        &selected,
    );
    let (other_layout, _) =
        dense_plan::<U1Irrep>(&other.sectors().unwrap(), other.degeneracies(), |_| 1, &[]);
    let dense = source.to_physical_dense().unwrap();
    let maps = vec![
        axis_gather(&layout, &ranges),
        identity_gather(&other_layout),
        identity_gather(&other_layout),
    ];
    let (shape, expected) = gather(&dense.shape, &dense.data, &maps);
    let actual = restricted.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    assert_close(&actual.data, &expected);
}

#[test]
fn restrict_keeps_su2_multiplets_intact_for_a_middle_sector_with_offset() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    // Three sectors so that dropping the middle one shifts the dense offsets
    // of the last, and j = 1/2, 1 carry nontrivial multiplet dimensions.
    let leg = su2(&provider, &[(0, 2), (1, 3), (2, 2)]);
    let other = su2(&provider, &[(0, 1), (1, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&other, &other], 11).unwrap();

    let selected = [
        (SU2Irrep::from_twice_spin(0), 1..2),
        (SU2Irrep::from_twice_spin(2), 0..1),
    ];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let restricted = source.restrict_leg(0, &selection).unwrap();
    assert_eq!(restricted.codomain()[0].degeneracies(), &[1, 1]);

    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        |sector: &SU2Irrep| sector.twice_spin() + 1,
        &selected,
    );
    let (other_layout, _) = dense_plan::<SU2Irrep>(
        &other.sectors().unwrap(),
        other.degeneracies(),
        |sector| sector.twice_spin() + 1,
        &[],
    );
    let dense = source.to_physical_dense().unwrap();
    let maps = vec![
        axis_gather(&layout, &ranges),
        identity_gather(&other_layout),
        identity_gather(&other_layout),
    ];
    let (shape, expected) = gather(&dense.shape, &dense.data, &maps);
    let actual = restricted.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    assert_close(&actual.data, &expected);
}

#[test]
fn restrict_equals_composition_with_the_inclusion_isometry_on_both_sides() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2), (0, 3), (1, 2)]);
    let spectator = u1(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&spectator, &leg], [&leg, &spectator], 3).unwrap();

    let selected = [(U1Irrep::new(-1), 0..1), (U1Irrep::new(0), 1..3)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let start = |sector: &U1Irrep| {
        selected
            .iter()
            .find(|(candidate, _)| candidate == sector)
            .map_or(0, |(_, range)| range.start)
    };
    // ι_σ : V <- W, a one on every selected degeneracy coordinate.
    let iota = TensorMap::<_, f64>::from_block_fn(
        &runtime,
        [&leg],
        [selection.subspace()],
        |trees, indices| {
            if indices[0] == start(&trees.domain_uncoupled()[0]) + indices[1] {
                1.0
            } else {
                0.0
            }
        },
    )
    .unwrap();
    // The adjoint isometry, hand-filled rather than taken from `adjoint()`.
    let iota_dagger = TensorMap::<_, f64>::from_block_fn(
        &runtime,
        [selection.subspace()],
        [&leg],
        |trees, indices| {
            if indices[1] == start(&trees.codomain_uncoupled()[0]) + indices[0] {
                1.0
            } else {
                0.0
            }
        },
    )
    .unwrap();
    let spectator_id = TensorMap::<_, f64>::id(&runtime, [&spectator]).unwrap();

    // Codomain leg 1: (id ⊗ ι†) ∘ t.
    let projector = spectator_id.otimes(&iota_dagger).unwrap();
    let expected = projector.compose(&source).unwrap();
    let restricted = source.restrict_leg(1, &selection).unwrap();
    assert_eq!(restricted.codomain(), expected.codomain());
    assert_eq!(restricted.domain(), expected.domain());
    assert_close(restricted.data(), expected.data());

    // Domain leg 2: t ∘ (ι ⊗ id).
    let inserter = iota.otimes(&spectator_id).unwrap();
    let expected = source.compose(&inserter).unwrap();
    let restricted = source.restrict_leg(2, &selection).unwrap();
    assert_eq!(restricted.codomain(), expected.codomain());
    assert_eq!(restricted.domain(), expected.domain());
    assert_close(restricted.data(), expected.data());
}

#[test]
fn restrict_of_a_fermionic_odd_dual_leg_equals_the_isometry_composition() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let leg = fz2(&provider, &[(false, 2), (true, 3)]).try_dual().unwrap();
    let spectator = fz2(&provider, &[(false, 2), (true, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &spectator], [&spectator], 5).unwrap();

    let selected = [(Z2Irrep::ODD, 1..3)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let start = |sector: &Z2Irrep| if *sector == Z2Irrep::ODD { 1 } else { 0 };
    let iota_dagger = TensorMap::<_, f64>::from_block_fn(
        &runtime,
        [selection.subspace()],
        [&leg],
        |trees, indices| {
            if indices[1] == start(&trees.codomain_uncoupled()[0]) + indices[0] {
                1.0
            } else {
                0.0
            }
        },
    )
    .unwrap();
    let spectator_id = TensorMap::<_, f64>::id(&runtime, [&spectator]).unwrap();
    let expected = iota_dagger
        .otimes(&spectator_id)
        .unwrap()
        .compose(&source)
        .unwrap();
    let restricted = source.restrict_leg(0, &selection).unwrap();
    assert_eq!(restricted.codomain(), expected.codomain());
    assert_close(restricted.data(), expected.data());
}

#[test]
fn restrict_reads_a_complex_lazy_adjoint_domain_leg_with_a_multi_sector_selection() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let rows = u1(&provider, &[(0, 2), (1, 3)]);
    let columns = u1(&provider, &[(0, 3), (1, 2)]);
    let parent: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&rows], [&columns], 13).unwrap();
    let lazy = parent.adjoint().unwrap();

    // The lazy adjoint's domain leg 1 is the parent's codomain leg.
    let leg = lazy.domain()[0].clone();
    let selected = [(U1Irrep::new(0), 1..2), (U1Irrep::new(1), 0..2)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let restricted = lazy.restrict_leg(1, &selection).unwrap();

    // Oracle: the dense gather of the adjoint's own physical expansion, which
    // shares no code with the restriction kernel's storage remap and
    // conjugation.
    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        |_| 1,
        &selected,
    );
    let (row_layout, _) = dense_plan::<U1Irrep>(
        &columns.sectors().unwrap(),
        columns.degeneracies(),
        |_| 1,
        &[],
    );
    let dense = lazy.to_physical_dense().unwrap();
    let maps = vec![identity_gather(&row_layout), axis_gather(&layout, &ranges)];
    let (shape, gathered) = gather(&dense.shape, &dense.data, &maps);
    let actual = restricted.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    assert_close_complex(&actual.data, &gathered);
}

#[test]
fn restrict_commutes_with_adjoint_under_the_codomain_domain_axis_map() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let rows = u1(&provider, &[(0, 2), (1, 3)]);
    let columns = u1(&provider, &[(0, 3), (1, 2)]);
    let source: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&rows], [&columns], 17).unwrap();
    let selection = LegSelection::try_new(&columns, [(U1Irrep::new(1), 0..1)]).unwrap();

    // Domain leg 1 of `t` is codomain leg 0 of `t†`.
    let left = source
        .restrict_leg(1, &selection)
        .unwrap()
        .adjoint()
        .unwrap();
    let right = source
        .adjoint()
        .unwrap()
        .restrict_leg(0, &selection)
        .unwrap();
    assert_eq!(left.codomain(), right.codomain());
    assert_eq!(left.domain(), right.domain());
    assert_close_complex(left.data(), right.data());
}

#[test]
fn full_selection_is_the_identity_and_round_trips_are_exact() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = su2(&provider, &[(0, 2), (1, 2), (2, 1)]);
    let other = su2(&provider, &[(0, 1), (1, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &other], [&other], 23).unwrap();

    let full = LegSelection::try_new(
        &leg,
        leg.sectors()
            .unwrap()
            .into_iter()
            .zip(leg.degeneracies())
            .map(|(sector, &degeneracy)| (sector, 0..degeneracy)),
    )
    .unwrap();
    assert_eq!(full.subspace(), &leg);
    let restricted = source.restrict_leg(0, &full).unwrap();
    assert_eq!(restricted.codomain(), source.codomain());
    assert_eq!(restricted.data(), source.data());

    // restrict ∘ embed = id, bitwise.
    let partial = LegSelection::try_new(&leg, [(SU2Irrep::from_twice_spin(1), 1..2)]).unwrap();
    let small = source.restrict_leg(0, &partial).unwrap();
    let embedded = small.embed_leg(0, &partial).unwrap();
    assert_eq!(embedded.codomain()[0], leg);
    assert_eq!(
        embedded.restrict_leg(0, &partial).unwrap().data(),
        small.data()
    );
}

#[test]
fn embed_after_restrict_is_the_orthogonal_projector() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2), (0, 3), (1, 2)]);
    let other = u1(&provider, &[(0, 2), (1, 1)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&other, &other], 29).unwrap();

    let selected = [(U1Irrep::new(0), 1..3), (U1Irrep::new(1), 0..1)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let projected = source
        .restrict_leg(0, &selection)
        .unwrap()
        .embed_leg(0, &selection)
        .unwrap();
    assert_eq!(projected.codomain()[0], leg);

    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        |_| 1,
        &selected,
    );
    let (other_layout, _) =
        dense_plan::<U1Irrep>(&other.sectors().unwrap(), other.degeneracies(), |_| 1, &[]);
    let dense = source.to_physical_dense().unwrap();
    let maps = vec![
        axis_gather(&layout, &ranges),
        identity_gather(&other_layout),
        identity_gather(&other_layout),
    ];
    let expected = masked(&dense.shape, &dense.data, &maps);
    let actual = projected.to_physical_dense().unwrap();
    assert_eq!(actual.shape, dense.shape);
    assert_close(&actual.data, &expected);

    // Idempotent: projecting twice changes nothing.
    let twice = projected
        .restrict_leg(0, &selection)
        .unwrap()
        .embed_leg(0, &selection)
        .unwrap();
    assert_eq!(twice.data(), projected.data());
}

#[test]
fn a_valid_selection_with_no_admissible_block_returns_an_empty_tensor() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    // A rank-(1,0) map with a nonzero charge has no admissible fusion tree.
    let charged = u1(&provider, &[(1, 3)]);
    let source = TensorMap::<_, f64>::zeros(&runtime, [&charged], []).unwrap();
    assert_eq!(source.block_count(), 0);

    let selection = LegSelection::try_new(&charged, [(U1Irrep::new(1), 1..2)]).unwrap();
    let restricted = source.restrict_leg(0, &selection).unwrap();
    assert_eq!(restricted.block_count(), 0);
    assert!(restricted.data().is_empty());
    assert_eq!(restricted.codomain()[0].degeneracies(), &[1]);

    let embedded = restricted.embed_leg(0, &selection).unwrap();
    assert_eq!(embedded.codomain()[0], charged);
    assert!(embedded.data().is_empty());

    // Leg-level validation still rejects malformed ranges there, because no
    // block would ever reach the kernel.
    assert!(LegSelection::try_new(&charged, [(U1Irrep::new(1), 2..4)]).is_err());
}

#[test]
fn selections_are_validated_against_their_parent_leg_before_anything_is_built() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(0, 2), (1, 3)]);
    let other = u1(&provider, &[(0, 2)]);

    assert!(LegSelection::try_new(&leg, std::iter::empty()).is_err());
    assert!(LegSelection::try_new(&leg, [(U1Irrep::new(0), 1..1)]).is_err());
    assert!(LegSelection::try_new(&leg, [(U1Irrep::new(0), Range { start: 2, end: 1 })]).is_err());
    assert!(LegSelection::try_new(&leg, [(U1Irrep::new(0), 0..3)]).is_err());
    assert!(LegSelection::try_new(&leg, [(U1Irrep::new(5), 0..1)]).is_err());
    assert!(LegSelection::try_new(
        &leg,
        [(U1Irrep::new(0), 0..1), (U1Irrep::new(0), 1..2)].into_iter()
    )
    .is_err());
    // A dual leg's own labels are the dualised ones; the non-dual label is
    // simply absent.
    let dual = leg.try_dual().unwrap();
    assert!(LegSelection::try_new(&dual, [(U1Irrep::new(1), 0..1)]).is_err());
    assert!(LegSelection::try_new(&dual, [(U1Irrep::new(-1), 0..1)]).is_ok());

    let selection = LegSelection::try_new(&leg, [(U1Irrep::new(0), 0..1)]).unwrap();
    let source = TensorMap::<_, f64>::zeros(&runtime, [&leg], [&other]).unwrap();
    assert!(source.restrict_leg(2, &selection).is_err());
    assert!(source.restrict_leg(1, &selection).is_err());
    // `embed_leg` wants the subspace, not the parent.
    assert!(source.embed_leg(0, &selection).is_err());
    let small = source.restrict_leg(0, &selection).unwrap();
    assert!(small.restrict_leg(0, &selection).is_err());
    assert!(small.embed_leg(0, &selection).is_ok());
}

#[test]
fn restricting_a_compact_diagonal_payload_is_rejected() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(0, 3), (1, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 31).unwrap();
    let diagonal = source.svd_compact().unwrap().1;
    let bond = diagonal.domain()[0].clone();
    let selection = LegSelection::try_new(&bond, [(U1Irrep::new(0), 0..1)]).unwrap();
    assert!(diagonal.restrict_leg(1, &selection).is_err());
    assert!(diagonal.embed_leg(1, &selection).is_err());
}

#[test]
fn embed_scatters_a_complex_lazy_adjoint_dual_domain_leg_per_sector() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2), (0, 3), (1, 4)])
        .try_dual()
        .unwrap();
    let selected = [(U1Irrep::new(1), 0..1), (U1Irrep::new(-1), 2..4)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let spectator = u1(&provider, &[(0, 2), (1, 1)]);
    let columns = u1(&provider, &[(-1, 1), (0, 2)]);

    // The parent already lives on the subspace, so the lazy adjoint itself is
    // the embed source: a dual leg, in the domain, complex, read through the
    // adjoint axis map, with a multi-sector table.
    let parent: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [selection.subspace(), &spectator], [&columns], 53)
            .unwrap();
    let lazy = parent.adjoint().unwrap();
    assert_eq!(lazy.domain()[0], *selection.subspace());

    let embedded = lazy.embed_leg(1, &selection).unwrap();
    assert_eq!(embedded.domain()[0], leg);
    assert_eq!(embedded.domain()[1], spectator);
    assert_eq!(embedded.codomain()[0], columns);

    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        |_| 1,
        &selected,
    );
    let (column_layout, _) = dense_plan::<U1Irrep>(
        &columns.sectors().unwrap(),
        columns.degeneracies(),
        |_| 1,
        &[],
    );
    let (spectator_layout, _) = dense_plan::<U1Irrep>(
        &spectator.sectors().unwrap(),
        spectator.degeneracies(),
        |_| 1,
        &[],
    );
    let maps = vec![
        identity_gather(&column_layout),
        axis_gather(&layout, &ranges),
        identity_gather(&spectator_layout),
    ];
    let source = lazy.to_physical_dense().unwrap();
    let actual = embedded.to_physical_dense().unwrap();
    let expected = scattered(&actual.shape, &source.data, &maps, Complex64::new(0.0, 0.0));
    assert_close_complex(&actual.data, &expected);

    // And the restriction of that embedding is the original, bitwise.
    assert_eq!(
        embedded.restrict_leg(1, &selection).unwrap().data(),
        lazy.data()
    );
}

#[test]
fn restrict_and_embed_on_a_product_fz2_u1_leg_match_the_isometry_composition() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let label = |odd: bool, charge: i32| {
        product_sector(
            if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
            U1Irrep::new(charge),
        )
    };
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (label(false, 0), 2),
            (label(true, 1), 3),
            (label(false, 2), 2),
        ],
    )
    .unwrap();
    let other = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [(label(false, 0), 2), (label(true, 1), 1)],
    )
    .unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &other], [&other], 59).unwrap();

    let selected = [(label(true, 1), 1..3), (label(false, 2), 0..1)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let start = |sector: &_| {
        selected
            .iter()
            .find(|(candidate, _)| candidate == sector)
            .map_or(0, |(_, range)| range.start)
    };
    let restricted = source.restrict_leg(0, &selection).unwrap();

    // The product rule has no physical-basis expansion, so the oracle here is
    // the inclusion isometry, applied with `compose` only.
    let iota_dagger = TensorMap::<_, Complex64>::from_block_fn(
        &runtime,
        [selection.subspace()],
        [&leg],
        |trees, indices| {
            if indices[1] == start(&trees.codomain_uncoupled()[0]) + indices[0] {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        },
    )
    .unwrap();
    let iota = TensorMap::<_, Complex64>::from_block_fn(
        &runtime,
        [&leg],
        [selection.subspace()],
        |trees, indices| {
            if indices[0] == start(&trees.domain_uncoupled()[0]) + indices[1] {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        },
    )
    .unwrap();
    let spectator_id = TensorMap::<_, Complex64>::id(&runtime, [&other]).unwrap();

    let expected = iota_dagger
        .otimes(&spectator_id)
        .unwrap()
        .compose(&source)
        .unwrap();
    assert_eq!(restricted.codomain(), expected.codomain());
    assert_close_complex(restricted.data(), expected.data());

    let embedded = restricted.embed_leg(0, &selection).unwrap();
    let expected = iota
        .otimes(&spectator_id)
        .unwrap()
        .compose(&restricted)
        .unwrap();
    assert_eq!(embedded.codomain(), expected.codomain());
    assert_close_complex(embedded.data(), expected.data());
}

#[test]
fn a_rank_five_restriction_still_gathers_one_axis_only() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2), (0, 2), (1, 3)]);
    let small = u1(&provider, &[(0, 1), (1, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&small, &leg, &small], [&small, &small], 61).unwrap();
    assert_eq!(source.rank(), 5);

    let selected = [(U1Irrep::new(1), 1..3)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let restricted = source.restrict_leg(1, &selection).unwrap();

    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        |_| 1,
        &selected,
    );
    let (small_layout, _) =
        dense_plan::<U1Irrep>(&small.sectors().unwrap(), small.degeneracies(), |_| 1, &[]);
    let dense = source.to_physical_dense().unwrap();
    let maps = vec![
        identity_gather(&small_layout),
        axis_gather(&layout, &ranges),
        identity_gather(&small_layout),
        identity_gather(&small_layout),
        identity_gather(&small_layout),
    ];
    let (shape, expected) = gather(&dense.shape, &dense.data, &maps);
    let actual = restricted.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    assert_close(&actual.data, &expected);
}

#[test]
fn a_selection_built_on_another_rule_is_a_rule_mismatch() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let two = Arc::new(ZNFusionRule::new(2).unwrap());
    let three = Arc::new(ZNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new_with_arc(Arc::clone(&two), [(two.irrep(0), 2), (two.irrep(1), 2)])
            .unwrap();
    let foreign = GradedSpace::try_new_with_arc(
        Arc::clone(&three),
        [(three.irrep(0), 2), (three.irrep(1), 2)],
    )
    .unwrap();
    let source = TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg]).unwrap();
    let selection = LegSelection::try_new(&foreign, [(three.irrep(0), 0..1)]).unwrap();
    let error = source.restrict_leg(0, &selection).unwrap_err();
    assert!(
        format!("{error}").contains("rule"),
        "expected a rule mismatch, got {error}"
    );
    assert!(source.embed_leg(0, &selection).is_err());
}

#[test]
fn restriction_commutes_with_permute_and_with_a_contraction_over_an_untouched_leg() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let leg = fz2(&provider, &[(false, 2), (true, 3)]);
    let spectator = fz2(&provider, &[(false, 2), (true, 2)]);
    let bond = fz2(&provider, &[(false, 1), (true, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &spectator], [&bond], 67).unwrap();
    let partner: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&bond], [&spectator], 71).unwrap();
    let selection = LegSelection::try_new(&leg, [(Z2Irrep::ODD, 1..3)]).unwrap();

    // Swapping two codomain legs and restricting are independent.
    let restrict_then_permute = source
        .restrict_leg(0, &selection)
        .unwrap()
        .permute(&[1, 0], &[2])
        .unwrap();
    let permute_then_restrict = source
        .permute(&[1, 0], &[2])
        .unwrap()
        .restrict_leg(1, &selection)
        .unwrap();
    assert_eq!(
        restrict_then_permute.codomain(),
        permute_then_restrict.codomain()
    );
    assert_close(restrict_then_permute.data(), permute_then_restrict.data());

    // Contracting the untouched bond commutes with the restriction. Both
    // sides run the same `contract`, so the fermionic supertrace twist on the
    // dual contracted axis appears identically on each.
    let restrict_then_contract = source
        .restrict_leg(0, &selection)
        .unwrap()
        .contract(&partner, &[2], &[0], &[0, 1, 2])
        .unwrap();
    let contract_then_restrict = source
        .contract(&partner, &[2], &[0], &[0, 1, 2])
        .unwrap()
        .restrict_leg(0, &selection)
        .unwrap();
    assert_eq!(
        restrict_then_contract.codomain(),
        contract_then_restrict.codomain()
    );
    assert_close(restrict_then_contract.data(), contract_then_restrict.data());
}
