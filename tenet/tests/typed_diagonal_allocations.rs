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
        // p = 3 is the general arm; p = 2 and p = Inf only delegate to the two
        // above, so they would prove nothing here.
        ("norm_p(3)", warmed_bytes(|| d.norm_p(3.0).unwrap())),
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

#[test]
fn the_matrix_functions_have_o_rank_diagonal_arms() {
    // What: `exp`, `inv`, `pinv` and `sqrt` on a spectrum factor are elementwise
    // on the `Σ_c k_c` stored values, not block work on the `Σ_c k_c²`
    // materialization. The ceiling catches a densified route; the "still owes"
    // assertion afterwards catches one that densifies into the shared cache,
    // which the warm-up run would otherwise have paid for silently.
    //
    // `exp` is the one that is new in the typed facade: the erased sibling
    // materializes a diagonal payload and eigendecomposes it, so this probe is
    // what pins the parity fix rather than just guarding it.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let d = spectrum(0x5eed_0021);
    let ceiling = dense_payload_bytes();

    for (name, bytes) in [
        ("exp", warmed_bytes(|| d.exp().unwrap())),
        ("inv", warmed_bytes(|| d.inv().unwrap())),
        ("pinv", warmed_bytes(|| d.pinv(1e-12).unwrap())),
        ("sqrt", warmed_bytes(|| d.sqrt().unwrap())),
    ] {
        assert!(
            bytes < ceiling,
            "compact {name} allocated at least one dense payload: {bytes} bytes"
        );
    }

    assert!(
        measured_bytes(|| d.data().len()) >= ceiling,
        "one of the matrix functions materialized the spectrum behind our back"
    );
}

#[test]
fn a_complex_spectrums_matrix_functions_stay_o_rank_too() {
    // What: the arms are dtype-generic, so a c64 spectrum takes the same route
    // — at twice the byte size, which is what the ceiling here accounts for.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let d = complex_source(0x5eed_0022).svd_compact().unwrap().1;
    let ceiling = 2 * dense_payload_bytes();

    for (name, bytes) in [
        ("exp", warmed_bytes(|| d.exp().unwrap())),
        ("inv", warmed_bytes(|| d.inv().unwrap())),
        ("pinv", warmed_bytes(|| d.pinv(1e-12).unwrap())),
        ("sqrt", warmed_bytes(|| d.sqrt().unwrap())),
    ] {
        assert!(
            bytes < ceiling,
            "compact c64 {name} allocated at least one dense payload: {bytes} bytes"
        );
    }

    assert!(
        measured_bytes(|| d.data().len()) >= ceiling,
        "one of the c64 matrix functions materialized the spectrum"
    );
}

#[test]
fn contracting_a_spectrum_scales_instead_of_densifying() {
    // What: `contract` against a compact operand takes the same scaling route
    // `compose` does (issue #584) — TensorKit's `lmul!`/`rmul!` on a
    // `DiagonalTensorMap` — rather than materializing the `Σ_c k_c²` block
    // diagonal and running a GEMM.
    //
    // The byte ceiling alone cannot prove it: the scaling route allocates a
    // scaled copy plus the `permute` destination, which is about what the dense
    // route spends on the materialization plus the GEMM result. What proves it
    // is that the spectrum is *still* compact afterwards — the first `data()`
    // on it must still pay for the dense buffer. That is what dies if the arm
    // is removed and `contract` reaches for `dense_data()` instead.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let d = spectrum(0x5eed_0031);
    let dense = source(0x5eed_0032);

    // `t · s`: the spectrum scales `t`'s contracted domain leg (`rmul!`).
    black_box(dense.contract(&d, &[1], &[0], &[0, 1]).unwrap());
    assert!(
        measured_bytes(|| d.data().len()) >= dense_payload_bytes(),
        "t * s materialized the spectrum"
    );

    // `s · t`: the mirror image, scaling `t`'s leading codomain leg (`lmul!`).
    let e = spectrum(0x5eed_0033);
    black_box(e.contract(&dense, &[1], &[0], &[0, 1]).unwrap());
    assert!(
        measured_bytes(|| e.data().len()) >= dense_payload_bytes(),
        "s * t materialized the spectrum"
    );

    // `s · s` stays compact end to end, so here the ceiling *is* decisive: the
    // whole contraction must cost less than a single dense payload.
    let f = spectrum(0x5eed_0034);
    let bytes = warmed_bytes(|| f.contract(&f, &[1], &[0], &[0, 1]).unwrap());
    assert!(
        bytes < dense_payload_bytes(),
        "s * s allocated at least one dense payload: {bytes} bytes"
    );
    assert!(
        measured_bytes(|| f.data().len()) >= dense_payload_bytes(),
        "s * s materialized its operand"
    );
}

#[test]
fn the_rank_one_swap_keeps_its_source_and_its_result_compact() {
    // What: re-ordering the two legs of a spectrum factor is a per-sector
    // rescaling of the stored values (#585), so it neither reads nor writes a
    // `Σ_c k_c²` payload. Three things are measured, because each catches a
    // different way of getting it wrong: the swap's own byte cost (a densified
    // route would pay for two dense buffers), that the *source* still owes its
    // materialization afterwards (a route through `dense_data()` would have
    // filled the shared cache and hidden itself from the byte ceiling), and
    // that the *result* owes one too (a compact-in, dense-out route would pass
    // both of the first two).
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let d = spectrum(0x5eed_0041);
    let ceiling = dense_payload_bytes();

    for (name, bytes) in [
        ("permute", warmed_bytes(|| d.permute(&[1], &[0]).unwrap())),
        ("transpose", warmed_bytes(|| d.transpose().unwrap())),
        (
            "transpose_axes",
            warmed_bytes(|| d.transpose_axes(&[1], &[0]).unwrap()),
        ),
    ] {
        assert!(
            bytes < ceiling,
            "compact {name} allocated at least one dense payload: {bytes} bytes"
        );
    }

    assert!(
        measured_bytes(|| d.data().len()) >= ceiling,
        "the rank-one swap materialized its source"
    );
    let swapped = d.transpose().unwrap();
    assert!(
        measured_bytes(|| swapped.data().len()) >= ceiling,
        "the rank-one swap built a dense result"
    );
}

#[test]
fn the_geometries_outside_the_proved_swap_keep_the_dense_route() {
    // What: the compact arm is deliberately narrow. `repartition(1)` is the
    // identity partition of a bond space and `braid` is an explicit braid;
    // neither is the proved swap, so both must still go through the dense tree
    // transform. Recorded as a probe rather than a comment because the whole
    // `contract` value-oracle sweep in `typed_facade.rs` uses `repartition(1)`
    // as its *dense twin* — widening the arm to cover it would silently turn
    // that oracle into a comparison of the fast path against itself.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let d = spectrum(0x5eed_0042);

    let repartitioned = d.repartition(1).unwrap();
    assert_eq!(
        measured_bytes(|| repartitioned.data().len()),
        0,
        "repartition(1) returned compact storage instead of the dense route"
    );
    let braided = d.braid(&[1], &[0], &[0, 1]).unwrap();
    assert_eq!(
        measured_bytes(|| braided.data().len()),
        0,
        "an explicit braid returned compact storage instead of the dense route"
    );
}

#[test]
fn compact_is_posdef_never_builds_or_caches_a_dense_payload() {
    // What: `is_posdef` on a spectrum factor is a comparison over the stored
    // values (#585). It is not allocation-*free* — the Hermiticity gate it
    // opens with is `t - t†`, which owns two compact `Σ_c k_c` results, and
    // that is the route TensorKit takes too — but it must stay far below the
    // `Σ_c k_c²` materialization the eigensolver route needed, and it must not
    // populate the shared dense cache, which the byte ceiling alone cannot see
    // once the warm-up run has paid for it.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let ceiling = dense_payload_bytes();

    let d = spectrum(0x5eed_0051);
    let bytes = warmed_bytes(|| d.is_posdef(0.0).unwrap());
    assert!(
        bytes < ceiling,
        "compact is_posdef allocated at least one dense payload: {bytes} bytes"
    );
    assert!(
        measured_bytes(|| d.data().len()) >= ceiling,
        "compact is_posdef materialized the spectrum"
    );

    let e = complex_source(0x5eed_0052).svd_compact().unwrap().1;
    let bytes = warmed_bytes(|| e.is_posdef(1e-12).unwrap());
    assert!(
        bytes < 2 * ceiling,
        "compact c64 is_posdef allocated at least one dense payload: {bytes} bytes"
    );
    assert!(
        measured_bytes(|| e.data().len()) >= 2 * ceiling,
        "compact c64 is_posdef materialized the spectrum"
    );
}
