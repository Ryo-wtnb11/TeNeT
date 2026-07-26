//! Allocation probes for the typed facade's compact diagonal storage (#570).
//!
//! The typed facade publishes no compact accessor — [`TensorMap::data`] always
//! reports the dense buffer — so the `Σ_c k_c` storage claim cannot be asserted
//! through the API. It is asserted here instead, the way the erased sibling's
//! `tests/compact_diagonal_allocations.rs` asserts its own: by counting bytes
//! through a global allocator while one operation runs.
//!
//! Every measurement is warmed first. The engine's layout and fusion-tree
//! caches allocate on first use, and those allocations belong to the cache, not
//! to the operation under test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use tenet::core::{Z2FusionRule, Z2Irrep};
use tenet::prelude::{Complex64, Runtime};
use tenet::typed::{GradedSpace, TensorMap};

struct CountingAllocator;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicU64 = AtomicU64::new(0);
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if ENABLED.load(Ordering::Relaxed) && !pointer.is_null() {
            ALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if ENABLED.load(Ordering::Relaxed) && !pointer.is_null() {
            ALLOCATED.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// One coupled sector of degeneracy `DEGENERACY`, so the dense payload of a
/// bond factor is exactly `DEGENERACY²` scalars and the compact one
/// `DEGENERACY`. A single sector keeps the arithmetic in the assertions
/// readable; the multi-sector case is covered by the value oracles in
/// `typed_facade.rs`.
const DEGENERACY: usize = 128;

/// `DEGENERACY² * size_of::<f64>()`: one dense f64 bond payload. Every ceiling
/// below is stated against this, because it is the allocation compact storage
/// exists to avoid.
fn dense_payload_bytes() -> u64 {
    (DEGENERACY * DEGENERACY * std::mem::size_of::<f64>()) as u64
}

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::builder().dense_threads(1).build().unwrap())
}

fn leg() -> GradedSpace<Z2FusionRule> {
    GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, DEGENERACY)], false).unwrap()
}

fn pseudo_random(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

fn source(seed: u64) -> TensorMap<Z2FusionRule, f64> {
    let leg = leg();
    let mut state = seed;
    TensorMap::from_block_fn(runtime(), [&leg], [&leg], move |_, _| {
        pseudo_random(&mut state)
    })
    .unwrap()
}

fn complex_source(seed: u64) -> TensorMap<Z2FusionRule, Complex64> {
    let leg = leg();
    let mut state = seed;
    TensorMap::from_block_fn(runtime(), [&leg], [&leg], move |_, _| {
        Complex64::new(pseudo_random(&mut state), pseudo_random(&mut state))
    })
    .unwrap()
}

fn measured_bytes<T>(operation: impl FnOnce() -> T) -> u64 {
    ALLOCATED.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Release);
    let output = black_box(operation());
    ENABLED.store(false, Ordering::Release);
    black_box(output);
    ALLOCATED.load(Ordering::Relaxed)
}

#[test]
fn svd_compacts_s_is_built_compact_and_materializes_only_on_demand() {
    // What: `svd_compact` stores `s` as `Σ_c k_c` values. The proof is that the
    // dense buffer is still missing afterwards — the first `data()` on a fresh
    // `s` has to allocate it, and it could not if construction had already
    // built one. The second `data()` allocates nothing, which is the body-level
    // cache being shared rather than rebuilt.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let tensor = source(0x5eed_0001);

    // Warm the factorization path itself; `s` is discarded, only caches persist.
    black_box(tensor.svd_compact().unwrap());

    let s = tensor.svd_compact().unwrap().1;
    let first = measured_bytes(|| s.data().len());
    assert!(
        first >= dense_payload_bytes(),
        "the dense s payload was already built at construction: first data() \
         allocated only {first} bytes"
    );
    assert_eq!(
        measured_bytes(|| s.data().len()),
        0,
        "the materialization cache is not shared between reads"
    );

    // And the same buffer, not a fresh one: a clone shares the body.
    let clone = s.clone();
    assert_eq!(
        measured_bytes(|| clone.data().len()),
        0,
        "a clone rebuilt the materialization instead of sharing it"
    );
}

#[test]
fn svd_truncs_s_is_built_compact_too() {
    // What: `svd_trunc` takes the same `_factors_` seam as `svd_compact`, so its
    // `s` carries the same storage. Same proof shape.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let tensor = source(0x5eed_0002);
    let truncation = tenet::typed::Truncation::Full;
    black_box(tensor.svd_trunc(&truncation).unwrap());

    let s = tensor.svd_trunc(&truncation).unwrap().s;
    let first = measured_bytes(|| s.data().len());
    assert!(
        first >= dense_payload_bytes(),
        "svd_trunc built a dense s at construction: first data() allocated only \
         {first} bytes"
    );
}

#[test]
fn a_complex_payloads_s_is_compact_as_well() {
    // What: the compact arm is dtype-generic — `D` is a type parameter, so a
    // c64 spectrum takes exactly the same route with no widening variant.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let tensor = complex_source(0x5eed_0003);
    black_box(tensor.svd_compact().unwrap());

    let s = tensor.svd_compact().unwrap().1;
    let first = measured_bytes(|| s.data().len());
    assert!(
        first >= 2 * dense_payload_bytes(),
        "the dense c64 s payload was already built at construction: first \
         data() allocated only {first} bytes"
    );
}

/// Runs `operation` once to warm every process-global cache it touches, then
/// measures a second, identical run.
fn warmed_bytes<T>(operation: impl Fn() -> T) -> u64 {
    black_box(operation());
    measured_bytes(&operation)
}

fn spectrum(seed: u64) -> TensorMap<Z2FusionRule, f64> {
    source(seed).svd_compact().unwrap().1
}

#[test]
fn storage_local_compact_operations_never_build_a_dense_payload() {
    // What: scale, adjoint, add(diagonal, diagonal) and compose(D, D) all stay
    // in O(Σ_c k_c). Each allocates its own compact result and nothing else, so
    // the ceiling is far below one dense payload.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let d = spectrum(0x5eed_0011);
    let ceiling = dense_payload_bytes();

    for (name, bytes) in [
        ("scale", warmed_bytes(|| d.scale(0.5))),
        ("adjoint", warmed_bytes(|| d.adjoint().unwrap())),
        ("add", warmed_bytes(|| d.add(&d, 0.75, -0.5).unwrap())),
        ("compose", warmed_bytes(|| d.compose(&d).unwrap())),
    ] {
        assert!(
            bytes < ceiling,
            "compact {name} allocated at least one dense payload: {bytes} bytes"
        );
    }

    // The reductions allocate nothing at all: they read the stored spectrum
    // rather than its materialization, so there is no destination to own.
    for (name, bytes) in [
        ("norm", warmed_bytes(|| d.norm().unwrap())),
        ("norm_inf", warmed_bytes(|| d.norm_inf().unwrap())),
        ("tr", warmed_bytes(|| d.tr().unwrap())),
        ("inner", warmed_bytes(|| d.inner(&d).unwrap())),
    ] {
        assert_eq!(bytes, 0, "compact {name} allocated temporary storage");
    }

    // And none of the above materialized the spectrum as a side effect: the
    // first read still has to build the dense buffer.
    assert!(
        measured_bytes(|| d.data().len()) >= ceiling,
        "one of the compact operations materialized the spectrum behind our back"
    );
}

#[test]
fn a_mixed_add_allocates_only_its_own_dense_result() {
    // What: adding a spectrum to a dense tensor on the same bond space scatters
    // straight into the owned result. Materializing the spectrum first would
    // double this, which is what the ceiling rejects.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let d = spectrum(0x5eed_0012);
    let dense = TensorMap::id(runtime(), &d.domain()).unwrap();
    // Reading `dense` must not be what pays for the diagonal: warm nothing on
    // `d` beyond what the operation itself needs.
    let bytes = warmed_bytes(|| d.add(&dense, 0.75, -0.5).unwrap());

    assert!(
        bytes < dense_payload_bytes() * 3 / 2,
        "diagonal + dense allocated more than one dense payload: {bytes} bytes"
    );
    // The byte ceiling alone cannot see a materialization that the warm-up run
    // already paid for and cached, so assert the absence directly: `d` must
    // still owe its dense buffer. This is what dies if the mixed arm reaches
    // for `dense_data()` instead of scattering the spectrum.
    assert!(
        measured_bytes(|| d.data().len()) >= dense_payload_bytes(),
        "the mixed add materialized the diagonal operand"
    );
    // Same on the mirrored arm.
    let e = spectrum(0x5eed_0014);
    black_box(dense.add(&e, 0.75, -0.5).unwrap());
    assert!(
        measured_bytes(|| e.data().len()) >= dense_payload_bytes(),
        "the mirrored mixed add materialized the diagonal operand"
    );
}

#[test]
fn absorbing_a_spectrum_through_compose_scales_instead_of_densifying() {
    // What: `u * s` and `s * vh` take the bond-scaling arms. Each allocates its
    // own dense result — `u` and `vh` are dense — but not a second dense buffer
    // for `s`, which is what the ceiling here rejects. The dense GEMM route
    // would need `s` materialized as well.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let tensor = source(0x5eed_0013);
    let (u, s, vh) = tensor.svd_compact().unwrap();
    let ceiling = dense_payload_bytes() * 3 / 2;

    assert!(
        warmed_bytes(|| u.compose(&s).unwrap()) < ceiling,
        "u * s densified the spectrum"
    );
    assert!(
        warmed_bytes(|| s.compose(&vh).unwrap()) < ceiling,
        "s * vh densified the spectrum"
    );
    // Still compact afterwards, for the same reason as above.
    assert!(
        measured_bytes(|| s.data().len()) >= dense_payload_bytes(),
        "compose materialized the spectrum behind our back"
    );
}
