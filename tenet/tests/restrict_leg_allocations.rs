//! Allocation contract of `restrict_leg` and of the re-routed internal
//! network restriction (#1285).
//!
//! The contract is *not* a total call count: each call rebuilds the small
//! structural objects of the destination hom-space. What is pinned is that
//! exactly one allocator-zeroed allocation has the output payload's size, and
//! that everything else is independent of the degeneracy dimensions — two
//! tensors with the same sector structure and different degeneracies must
//! allocate the same number of non-payload blocks.

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
fn restrict_leg_allocates_one_zeroed_payload_and_degeneracy_independent_scratch() {
    let _guard = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (small, small_bytes) = restrict_measurement(1);
    let (large, large_bytes) = restrict_measurement(8);
    assert!(large_bytes > small_bytes);

    for (measurement, bytes) in [(&small, small_bytes), (&large, large_bytes)] {
        assert_eq!(
            measurement
                .zeroed_sizes
                .iter()
                .filter(|&&size| size == bytes)
                .count(),
            1,
            "exactly one payload-sized zeroed allocation, got {:?} for {bytes} bytes",
            measurement.zeroed_sizes
        );
    }
    // Structural work does not grow with the degeneracy dimensions. The byte
    // budget is what rules out a second payload-sized buffer taken through
    // plain `alloc`/`realloc`, which the zeroed-size log cannot see.
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
fn the_network_restriction_still_takes_one_allocation_for_several_axes() {
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
    assert_eq!(
        measurement
            .zeroed_sizes
            .iter()
            .filter(|&&size| size == payload_bytes)
            .count(),
        1,
        "two restricted axes must still share one payload allocation, got {:?}",
        measurement.zeroed_sizes
    );
}
