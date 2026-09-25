//! Allocation contract of `restrict_leg` and of the re-routed internal
//! network restriction (#1285).
//!
//! The contract is *not* a total call count: each call rebuilds the small
//! structural objects of the destination hom-space. What is pinned is that
//! the output payload is never zero-filled — its blocks tile it, so it is
//! allocated uninitialized and overwritten once (#1290) — and that everything
//! else is independent of the degeneracy dimensions — two tensors with the
//! same sector structure and different degeneracies must allocate the same
//! number of non-payload blocks.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::{Arc, Mutex};

use tenet::core::{TypedSectorAdmission, U1FusionRule, U1Irrep};
use tenet::prelude::{GradedSpace, LegSelection, Runtime, TensorMap};
use tenet::typed::NetworkDegeneracyRestriction;

const ZEROED_LOG_CAPACITY: usize = 64;

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
    static ZEROED_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static ZEROED_SIZES: [Cell<usize>; ZEROED_LOG_CAPACITY] =
        const { [const { Cell::new(0) }; ZEROED_LOG_CAPACITY] };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            ALLOCATED_BYTES.set(ALLOCATED_BYTES.get() + layout.size());
        }
        pointer
    }

    // Overridden so a scalar zero fill cannot masquerade as a calloc.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            ALLOCATED_BYTES.set(ALLOCATED_BYTES.get() + layout.size());
            let index = ZEROED_ALLOCATIONS.get();
            ZEROED_ALLOCATIONS.set(index + 1);
            if index < ZEROED_LOG_CAPACITY {
                ZEROED_SIZES.with(|sizes| sizes[index].set(layout.size()));
            }
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            // A realloc hands back a buffer of `new_size`, so charge all of
            // it: a payload-sized temporary grown this way must show up.
            ALLOCATED_BYTES.set(ALLOCATED_BYTES.get() + new_size);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

struct Measurement {
    allocations: usize,
    bytes: usize,
    zeroed_sizes: Vec<usize>,
}

fn measure(operation: impl FnOnce()) -> Measurement {
    ALLOCATIONS.set(0);
    ALLOCATED_BYTES.set(0);
    ZEROED_ALLOCATIONS.set(0);
    COUNTING.set(true);
    operation();
    COUNTING.set(false);
    Measurement {
        allocations: ALLOCATIONS.get(),
        bytes: ALLOCATED_BYTES.get(),
        zeroed_sizes: ZEROED_SIZES.with(|sizes| {
            sizes[..ZEROED_ALLOCATIONS.get().min(ZEROED_LOG_CAPACITY)]
                .iter()
                .map(Cell::get)
                .collect()
        }),
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

/// One `restrict_leg` on a leg whose degeneracies are `scale` times a base
/// shape: the measurement and the payload byte count for that scale.
fn restrict_measurement(scale: usize) -> (Measurement, usize) {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(
        &provider,
        &[(-1, 2 * scale), (0, 3 * scale), (1, 2 * scale)],
    );
    let other = u1(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&other, &other], 41).unwrap();
    let selection = LegSelection::try_new(
        &leg,
        [
            (U1Irrep::new(-1), 0..scale),
            (U1Irrep::new(0), scale..3 * scale),
        ],
    )
    .unwrap();

    // Warm every structural cache the call can hit before measuring.
    let warm = source.restrict_leg(0, &selection).unwrap();
    let payload_bytes = std::mem::size_of_val(warm.data());

    let mut output = None;
    let measurement = measure(|| {
        output = Some(black_box(source.restrict_leg(0, &selection).unwrap()));
    });
    assert_eq!(output.unwrap().data(), warm.data());
    (measurement, payload_bytes)
}

#[test]
fn restrict_leg_allocates_one_unfilled_payload_and_degeneracy_independent_scratch() {
    let _guard = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (small, small_bytes) = restrict_measurement(1);
    let (large, large_bytes) = restrict_measurement(8);
    assert!(large_bytes > small_bytes);

    for (measurement, bytes) in [(&small, small_bytes), (&large, large_bytes)] {
        assert!(
            !measurement.zeroed_sizes.contains(&bytes),
            "no payload-sized zeroed allocation, got {:?} for {bytes} bytes",
            measurement.zeroed_sizes
        );
    }
    // The block copy's scratch is inline (#1362) and the warm layout lookup
    // builds no key (#1367), so every call is the result's structural objects
    // or its payload.
    assert_eq!(small.allocations, 9);
    // Structural work does not grow with the degeneracy dimensions. The byte
    // budget pins the payload to one buffer and rules out a second
    // payload-sized buffer taken through plain `alloc`/`realloc`.
    assert_eq!(small.allocations, large.allocations);
    assert_eq!(
        small.bytes - small_bytes,
        large.bytes - large_bytes,
        "non-payload bytes must not depend on the degeneracy dimensions: \
         {small:?} vs {large:?} bytes for payloads {small_bytes}/{large_bytes}",
        small = small.bytes,
        large = large.bytes
    );
}

#[test]
fn the_network_restriction_of_several_axes_fills_no_payload() {
    let _guard = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let rows = u1(&provider, &[(0, 6)]);
    let columns = u1(&provider, &[(0, 8)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&rows], [&columns], 43).unwrap();
    let zero_id = TypedSectorAdmission::try_encode_label(provider.as_ref(), &zero).unwrap();
    let restrictions = [
        NetworkDegeneracyRestriction {
            effective_axis: 0,
            authority_sector: zero_id,
            range: 1..4,
            partner: false,
        },
        NetworkDegeneracyRestriction {
            effective_axis: 1,
            authority_sector: zero_id,
            range: 2..6,
            partner: false,
        },
    ];

    let warm = source
        .network_restrict_degeneracies(false, &restrictions)
        .unwrap();
    let payload_bytes = std::mem::size_of_val(warm.data());
    assert_eq!(warm.data().len(), 3 * 4);

    let mut output = None;
    let measurement = measure(|| {
        output = Some(black_box(
            source
                .network_restrict_degeneracies(false, &restrictions)
                .unwrap(),
        ));
    });
    assert_eq!(output.unwrap().data(), warm.data());
    assert!(
        !measurement.zeroed_sizes.contains(&payload_bytes),
        "two restricted axes write one unfilled payload, got {:?}",
        measurement.zeroed_sizes
    );
    assert!(measurement.bytes >= payload_bytes);
}

/// One compact `restrict_diagonal` on an `s : bond <- bond` whose degeneracies
/// are `scale` times a base shape, keeping a fixed prefix of each sector.
fn restrict_diagonal_measurement(scale: usize) -> Measurement {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2 * scale), (0, 3 * scale)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 47).unwrap();
    let (_, s, _) = source.svd_compact().unwrap();
    let bond = s.domain()[0].clone();
    let selection =
        LegSelection::try_new(&bond, [(U1Irrep::new(-1), 0..2), (U1Irrep::new(0), 0..3)]).unwrap();

    let warm = s.restrict_diagonal(&selection).unwrap();
    assert!(
        warm.diagonal_spectrum().unwrap().is_some(),
        "a compact receiver must stay compact"
    );

    let mut output = None;
    let measurement = measure(|| {
        output = Some(black_box(s.restrict_diagonal(&selection).unwrap()));
    });
    assert_eq!(
        output.unwrap().diagonal_spectrum().unwrap(),
        warm.diagonal_spectrum().unwrap()
    );
    measurement
}

#[test]
fn compact_restrict_diagonal_costs_the_kept_values_and_nothing_per_discarded_one() {
    let _guard = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // The kept prefix is the same in both; only the discarded tail grows. A
    // compact restriction must therefore cost exactly the same, which is the
    // `O(sum_c k'_c)` contract: no dense block is ever materialized.
    let small = restrict_diagonal_measurement(1);
    let large = restrict_diagonal_measurement(16);
    assert_eq!(small.allocations, large.allocations);
    assert_eq!(small.bytes, large.bytes);
}

#[test]
fn lazy_adjoint_add_and_materialization_fill_no_payload() {
    let _guard = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // What: the owned outputs of a lazy-adjoint `add` and of the first
    // lazy-adjoint `data()` tile their storage, so neither zero-fills its
    // payload before overwriting every block (#1290). For `f64` the base
    // fill was `vec![0.0; len]`, an allocator-zeroed allocation.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2), (0, 3), (1, 2)]);
    let other = u1(&provider, &[(-1, 1), (0, 2), (1, 3)]);
    let square: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &other], [&leg, &other], 53).unwrap();
    let partner: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &other], [&leg, &other], 59).unwrap();
    let lazy = square.adjoint().unwrap();
    let payload_bytes = std::mem::size_of_val(square.data());
    assert!(square.block_count() > 1);

    let mut sum = None;
    let add = measure(|| {
        sum = Some(black_box(lazy.add(&partner, 0.5, -2.0).unwrap()));
    });
    assert!(
        !add.zeroed_sizes.contains(&payload_bytes),
        "lazy add zero-filled its payload: {:?}",
        add.zeroed_sizes
    );

    let materialize = measure(|| {
        black_box(lazy.data().len());
    });
    assert!(
        !materialize.zeroed_sizes.contains(&payload_bytes),
        "lazy materialization zero-filled its payload: {:?}",
        materialize.zeroed_sizes
    );
    // Oracle: the elementwise sum over the materialized adjoint payload,
    // which shares the partner's logical layout.
    let expected: Vec<u64> = lazy
        .data()
        .iter()
        .zip(partner.data())
        .map(|(&x, &y)| (0.5 * x + -2.0 * y).to_bits())
        .collect();
    let actual: Vec<u64> = sum.unwrap().data().iter().map(|x| x.to_bits()).collect();
    assert_eq!(actual, expected);
}

/// Steady-state `restrict_leg` on `[leg, other] <- [leg, other]`: every
/// coupled sector has several row trees, so the destination blocks are
/// strided sub-matrices. Returns the measurement and payload bytes of one
/// warm call whose previous outputs were dropped.
fn steady_multi_row_restrict(scale: usize) -> (Measurement, usize) {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(
        &provider,
        &[(-1, 2 * scale), (0, 3 * scale), (1, 2 * scale)],
    );
    let other = u1(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &other], [&leg, &other], 61).unwrap();
    let selection = LegSelection::try_new(
        &leg,
        [
            (U1Irrep::new(-1), 0..scale),
            (U1Irrep::new(0), scale..3 * scale),
        ],
    )
    .unwrap();
    let warm = source.restrict_leg(0, &selection).unwrap();
    let expected = warm.data().to_vec();
    let payload_bytes = std::mem::size_of_val(warm.data());
    drop(warm);
    drop(black_box(source.restrict_leg(0, &selection).unwrap()));
    let mut output = None;
    let measurement = measure(|| {
        output = Some(black_box(source.restrict_leg(0, &selection).unwrap()));
    });
    assert_eq!(output.unwrap().data(), expected.as_slice());
    (measurement, payload_bytes)
}

#[test]
fn steady_state_restrict_leg_with_several_row_trees_pays_no_layout_proof() {
    let _guard = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (small, small_bytes) = steady_multi_row_restrict(1);
    let (large, large_bytes) = steady_multi_row_restrict(4);
    for (measurement, bytes) in [(&small, small_bytes), (&large, large_bytes)] {
        assert!(
            !measurement.zeroed_sizes.contains(&bytes),
            "no payload-sized zeroed allocation, got {:?} for {bytes} bytes",
            measurement.zeroed_sizes
        );
    }
    // origin/main (zero-filled output) measures 11 here. Proving the tiling
    // by compiling `coupled_sector_regions` on the per-call destination
    // wrapper measured 50: the proof must be an O(1) property recorded when
    // the canonical layout is built (#1290).
    assert!(
        small.allocations <= 11,
        "steady-state restrict_leg allocated {} times",
        small.allocations
    );
    assert_eq!(small.allocations, large.allocations);
}
