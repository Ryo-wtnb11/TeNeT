//! Allocation contract of the owned eager contraction output (#1212): the
//! payload is one allocator-zeroed allocation of `P * size_of::<D>()` bytes
//! for both dtypes, and nothing else on the warmed path allocates zeroed
//! memory or re-touches the inactive blocks.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::{Arc, Mutex};

use num_complex::{Complex32, Complex64};
use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, TensorMap, TensorScalar};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static ZEROED_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    // Sizes of the zeroed allocations in order; the allocator cannot grow a
    // `Vec` while it is being called, so the log is a fixed ring.
    static ZEROED_SIZES: [Cell<usize>; ZEROED_LOG_CAPACITY] =
        const { [const { Cell::new(0) }; ZEROED_LOG_CAPACITY] };
}

const ZEROED_LOG_CAPACITY: usize = 64;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }

    // Why override: the default `alloc_zeroed` is `alloc` + `write_bytes`,
    // which would hide a TeNeT-side zero fill behind the same counter.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
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
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

struct Measurement {
    allocations: usize,
    zeroed_sizes: Vec<usize>,
}

impl Measurement {
    fn zeroed_allocations_of(&self, bytes: usize) -> usize {
        self.zeroed_sizes
            .iter()
            .filter(|&&size| size == bytes)
            .count()
    }
}

fn measure(f: impl FnOnce()) -> Measurement {
    ALLOCATIONS.set(0);
    ZEROED_ALLOCATIONS.set(0);
    COUNTING.set(true);
    f();
    COUNTING.set(false);
    Measurement {
        allocations: ALLOCATIONS.get(),
        zeroed_sizes: ZEROED_SIZES.with(|sizes| {
            sizes[..ZEROED_ALLOCATIONS.get().min(ZEROED_LOG_CAPACITY)]
                .iter()
                .map(Cell::get)
                .collect()
        }),
    }
}

fn u1_space(provider: &Arc<U1FusionRule>, sectors: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::clone(provider),
        sectors
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn su2_space(
    provider: &Arc<SU2FusionRule>,
    sectors: &[(usize, usize)],
) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::clone(provider),
        sectors
            .iter()
            .map(|&(twice_spin, degeneracy)| (SU2Irrep::from_twice_spin(twice_spin), degeneracy)),
    )
    .unwrap()
}

/// The contracted bond carries only charge 0, so the destination `[a] <- [c]`
/// has coupled charges -1 and 1 that no GEMM writes (inactive blocks).
fn u1_contract_measurement<D: TensorScalar + std::fmt::Debug>(max_allocations: usize, seed: u64) {
    let _measurement = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let open = u1_space(&provider, &[(-1, 2), (0, 3), (1, 2)]);
    let bond = u1_space(&provider, &[(0, 2)]);
    let lhs: TensorMap<_, D> = TensorMap::rand_with_seed(&runtime, [&open], [&bond], seed).unwrap();
    let rhs: TensorMap<_, D> =
        TensorMap::rand_with_seed(&runtime, [&bond], [&open], seed + 1).unwrap();
    let warm = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    let payload_len = warm.data().len();
    assert_eq!(payload_len, 2 * 2 + 3 * 3 + 2 * 2);
    assert_eq!(warm.block_count(), 3);

    let mut output = None;
    let measurement = measure(|| {
        output = Some(black_box(lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap()));
    });
    let output = output.unwrap();
    assert_eq!(output.data(), warm.data());

    let payload_bytes = std::mem::size_of_val(warm.data());
    assert_eq!(
        measurement.zeroed_allocations_of(payload_bytes),
        1,
        "the owned output must be exactly one allocator-zeroed payload of {payload_bytes} bytes; zeroed sizes {:?}",
        measurement.zeroed_sizes
    );
    assert!(
        measurement.allocations <= max_allocations,
        "owned U(1) contraction allocated {} times (limit {max_allocations})",
        measurement.allocations
    );
}

// Total allocations of one warmed owned U(1) contraction measured on the
// pre-change path (origin/main 9467383f, dense_threads(1), debug build):
// f64 38 calls / 2380 bytes with one 136-byte zeroed payload; Complex64 38
// calls / 2516 bytes with no zeroed payload (TeNeT-side fill). After #1212:
// f64 38 / 2380, Complex64 38 / 2516, each with exactly one zeroed payload.
const TOTAL_ALLOCATIONS_U1: usize = 38;

#[test]
fn owned_f64_contraction_output_is_one_allocator_zeroed_payload() {
    u1_contract_measurement::<f64>(TOTAL_ALLOCATIONS_U1, 1_212_001);
}

#[test]
fn owned_c64_contraction_output_is_one_allocator_zeroed_payload() {
    u1_contract_measurement::<Complex64>(TOTAL_ALLOCATIONS_U1, 1_212_002);
}

/// SU(2) compose: non-abelian recoupling, so this is the fixture where the
/// coefficient scratch is converted per structure identity.
fn su2_compose_measurement<D: TensorScalar>(seed: u64) -> usize {
    let _measurement = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    // Destination `[a, c] <- [d, e]` couples to spins 1/2 and 3/2; the bond
    // `b` carries only spin 1/2, so the spin-3/2 block is inactive.
    let a = su2_space(&provider, &[(0, 2), (2, 3)]);
    let c = su2_space(&provider, &[(1, 2)]);
    let b = su2_space(&provider, &[(1, 3)]);
    let lhs: TensorMap<_, D> = TensorMap::rand_with_seed(&runtime, [&a, &c], [&b], seed).unwrap();
    let rhs: TensorMap<_, D> =
        TensorMap::rand_with_seed(&runtime, [&b], [&a, &c], seed + 1).unwrap();
    let warm = lhs.compose(&rhs).unwrap();
    let measurement = measure(|| {
        black_box(lhs.compose(&rhs).unwrap());
    });
    let payload_bytes = std::mem::size_of_val(warm.data());
    assert_eq!(
        measurement.zeroed_allocations_of(payload_bytes),
        1,
        "the owned output must be exactly one allocator-zeroed payload of \
         {payload_bytes} bytes; zeroed sizes {:?}",
        measurement.zeroed_sizes
    );
    assert!(
        measurement.allocations <= 36,
        "owned SU(2) compose allocated {} times",
        measurement.allocations
    );
    measurement.allocations
}

#[test]
fn owned_su2_contraction_with_inactive_sectors_is_one_allocator_zeroed_payload() {
    // Pre-change: 36 calls / 3468 bytes, one 1088-byte zeroed payload;
    // unchanged after #1212.
    su2_compose_measurement::<f64>(1_212_003);
}

/// #1315: a single-precision payload must run the same algorithm on the
/// recoupling path, not a different one. Same fixture, same call count; only
/// the zeroed payload is half the size, which
/// `zeroed_allocations_of(size_of_val(data))` checks per dtype.
#[test]
fn owned_su2_single_precision_contraction_allocates_like_double() {
    let double = su2_compose_measurement::<f64>(1_315_003);
    let single = su2_compose_measurement::<f32>(1_315_003);
    assert_eq!(
        single, double,
        "f32 SU(2) compose allocated {single} times against {double} for f64"
    );
    let complex_double = su2_compose_measurement::<Complex64>(1_315_005);
    let complex_single = su2_compose_measurement::<Complex32>(1_315_005);
    assert_eq!(complex_single, complex_double);
}
