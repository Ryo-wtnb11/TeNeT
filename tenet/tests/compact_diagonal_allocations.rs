use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use tenet::prelude::*;

struct CountingAllocator;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicU64 = AtomicU64::new(0);
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());
static F64_64: OnceLock<Fixture> = OnceLock::new();
static F64_128: OnceLock<Fixture> = OnceLock::new();
static C64_128: OnceLock<Fixture> = OnceLock::new();
static C64_EIG_16: OnceLock<Tensor> = OnceLock::new();

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

struct Fixture {
    diagonal: Tensor,
    dense: Tensor,
}

fn prepare_fixture(degeneracy: usize, dtype: Dtype, seed: u64) -> Fixture {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::u1([(0, degeneracy)]);
    let source = Tensor::rand_with_seed(&runtime, dtype, [&space], [&space], seed).unwrap();
    let diagonal = source.svd_compact().unwrap().1;
    let dense = Tensor::rand_with_seed(&runtime, dtype, [&space], [&space], seed + 1).unwrap();
    Fixture { diagonal, dense }
}

fn f64_fixture(degeneracy: usize) -> &'static Fixture {
    match degeneracy {
        64 => F64_64.get_or_init(|| prepare_fixture(64, Dtype::F64, 801)),
        128 => F64_128.get_or_init(|| prepare_fixture(128, Dtype::F64, 803)),
        _ => panic!("unsupported allocation fixture size {degeneracy}"),
    }
}

fn c64_fixture() -> &'static Fixture {
    C64_128.get_or_init(|| prepare_fixture(128, Dtype::C64, 805))
}

fn c64_eig_fixture() -> &'static Tensor {
    C64_EIG_16.get_or_init(|| {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(0, 16)]);
        let source = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 807).unwrap();
        source.eig_full().unwrap().0
    })
}

fn measured_bytes<T>(operation: impl FnOnce() -> T) -> u64 {
    ALLOCATED.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Release);
    let output = black_box(operation());
    ENABLED.store(false, Ordering::Release);
    black_box(output);
    ALLOCATED.load(Ordering::Relaxed)
}

fn measured_product_bytes(diagonal: &Tensor) -> u64 {
    black_box(diagonal.compose(diagonal).unwrap());
    measured_bytes(|| diagonal.compose(diagonal).unwrap())
}

fn measured_unary_bytes<T>(diagonal: &Tensor, operation: impl Fn(&Tensor) -> T) -> u64 {
    black_box(operation(diagonal));
    measured_bytes(|| operation(diagonal))
}

fn measured_dense_add_bytes(fixture: &Fixture) -> u64 {
    black_box(fixture.diagonal.add(&fixture.dense, 0.75, -0.5).unwrap());
    measured_bytes(|| fixture.diagonal.add(&fixture.dense, 0.75, -0.5).unwrap())
}

/// A compact diagonal product stores one value per bond basis state. Comparing
/// two sizes makes the gate insensitive to fixed cache/metadata allocations
/// while rejecting the old dense d-by-d materialization.
#[test]
fn diagonal_product_allocation_bytes_scale_linearly() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let small = measured_product_bytes(&f64_fixture(64).diagonal);
    let large = measured_product_bytes(&f64_fixture(128).diagonal);
    assert!(
        large <= small * 4,
        "allocation growth is not O(d): d=64 used {small} bytes, d=128 used {large} bytes"
    );
    assert!(
        large < (128 * 128 * std::mem::size_of::<f64>()) as u64,
        "compact product allocated at least one dense payload: {large} bytes"
    );

    // What: both stored-block and compact-spectrum trace are reductions with no
    // destination or scratch allocation after warmup.
    let fixture = f64_fixture(128);
    assert_eq!(
        measured_unary_bytes(&fixture.dense, |tensor| tensor.tr().unwrap()),
        0
    );
    assert_eq!(
        measured_unary_bytes(&fixture.diagonal, |tensor| tensor.tr().unwrap()),
        0
    );
    assert_eq!(
        measured_unary_bytes(c64_eig_fixture(), |tensor| tensor.tr().unwrap()),
        0
    );
}

#[test]
fn storage_local_diagonal_operations_do_not_allocate_dense_payloads() {
    // What: compact operations allocate at most their O(r) owned result, while
    // reductions and metadata-only transforms allocate no temporary storage.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    for (dtype, fixture) in [("f64", f64_fixture(128)), ("c64", c64_fixture())] {
        for (name, bytes) in [
            (
                "adjoint",
                measured_unary_bytes(&fixture.diagonal, |d| d.adjoint().unwrap()),
            ),
            (
                "twist",
                measured_unary_bytes(&fixture.diagonal, |d| d.twist(&[0]).unwrap()),
            ),
            (
                "norm",
                measured_unary_bytes(&fixture.diagonal, |d| d.norm().unwrap()),
            ),
            // `norm_p` at a general exponent takes the compact accumulator
            // rather than delegating to `norm` / `norm_inf`, so it needs its
            // own probe: p = 3 is the arm that would have to materialize if
            // the `Σ_c dim(c) Σ_i |λ_i|^p` sum were written over the dense
            // block diagonal instead of the stored spectra.
            (
                "norm_p(3)",
                measured_unary_bytes(&fixture.diagonal, |d| d.norm_p(3.0).unwrap()),
            ),
        ] {
            assert_eq!(bytes, 0, "{dtype} {name} allocated temporary storage");
        }
        for (name, bytes) in [
            (
                "scale",
                measured_unary_bytes(&fixture.diagonal, |d| d.scale(0.5).unwrap()),
            ),
            (
                "add",
                measured_unary_bytes(&fixture.diagonal, |d| d.add(d, 0.75, -0.5).unwrap()),
            ),
        ] {
            assert!(
                bytes < (128 * 128 * std::mem::size_of::<f64>()) as u64,
                "{dtype} {name} allocated at least one dense diagonal payload: {bytes} bytes"
            );
        }
    }

    // The byte ceilings above cannot see a route that materializes: the
    // warm-up run inside `measured_unary_bytes` would pay for it and cache it
    // on the very tensor the measurement then reads, so the measured run is
    // free either way. Assert the absence directly instead, on a *fresh*
    // spectrum no warm-up has touched: after every reduction above has run on
    // it, its first `data()` must still have to build the dense buffer.
    //
    // A fresh tensor rather than the shared `OnceLock` fixture because the
    // assertion consumes the one un-materialized read a tensor has.
    let fresh = prepare_fixture(128, Dtype::F64, 809).diagonal;
    black_box(fresh.adjoint().unwrap());
    black_box(fresh.twist(&[0]).unwrap());
    black_box(fresh.norm().unwrap());
    black_box(fresh.norm_p(3.0).unwrap());
    black_box(fresh.tr().unwrap());
    assert!(
        measured_bytes(|| fresh.data().len()) >= (128 * 128 * std::mem::size_of::<f64>()) as u64,
        "one of the compact reductions materialized the spectrum behind our back"
    );

    let f64_diagonal = &f64_fixture(128).diagonal;
    let to_c64 = measured_unary_bytes(f64_diagonal, Tensor::to_c64);
    assert!(
        to_c64 < (128 * 128 * std::mem::size_of::<f64>()) as u64,
        "to_c64 allocated at least one dense diagonal payload: {to_c64} bytes"
    );
}

#[test]
fn diagonal_dense_add_allocates_only_the_dense_result_payload() {
    // What: adding a compact diagonal to a dense tensor scatters into the
    // owned result without allocating a second dense diagonal input.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let degeneracy = 128usize;
    let bytes = measured_dense_add_bytes(f64_fixture(degeneracy));
    let dense_payload = degeneracy * degeneracy * std::mem::size_of::<f64>();
    assert!(
        bytes < (dense_payload * 3 / 2) as u64,
        "diagonal+dense add allocated more than one dense payload: {bytes} bytes"
    );
}

/// Bytes one full-pair trace costs on a *fresh* compact tensor.
///
/// Why not [`measured_unary_bytes`]: its warm-up run is on the very tensor it
/// then measures, and a route that materialized would have filled that tensor's
/// shared materialization cache during the warm-up — so the measured run would
/// read the cache and look compact. The warm-up here runs on a throwaway twin,
/// which leaves the process-global layout and fusion-tree caches hot (what the
/// warm-up is for) while the measured tensor still owes its dense buffer.
fn measured_fresh_trace_bytes(degeneracy: usize, dtype: Dtype, seed: u64) -> u64 {
    let warmup = prepare_fixture(degeneracy, dtype, seed);
    black_box(warmup.diagonal.trace_pairs(&[(0, 1)]).unwrap());
    let fixture = prepare_fixture(degeneracy, dtype, seed);
    measured_bytes(|| fixture.diagonal.trace_pairs(&[(0, 1)]).unwrap())
}

#[test]
fn full_rank_one_trace_pairs_allocation_bytes_scale_linearly() {
    // What: `trace_pairs` over the only pair of a rank-(1,1) compact tensor
    // reduces the stored spectrum instead of materializing the `d`-by-`d`
    // payload the partial-trace engine used to need (#585). Two sizes make the
    // gate insensitive to the fixed rank-0 destination and its layout metadata
    // while rejecting the old materialization.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let small = measured_fresh_trace_bytes(64, Dtype::F64, 811);
    let large = measured_fresh_trace_bytes(128, Dtype::F64, 813);
    assert!(
        large <= small * 4,
        "allocation growth is not O(d): d=64 used {small} bytes, d=128 used {large} bytes"
    );
    assert!(
        large < (128 * 128 * std::mem::size_of::<f64>()) as u64,
        "compact trace_pairs allocated at least one dense payload: {large} bytes"
    );
    let complex = measured_fresh_trace_bytes(128, Dtype::C64, 815);
    assert!(
        complex < (2 * 128 * 128 * std::mem::size_of::<f64>()) as u64,
        "compact c64 trace_pairs allocated at least one dense payload: {complex} bytes"
    );
}

/// Bytes one compact `exp` costs on a *fresh* diagonal, plus the same follow-up
/// the `trace_pairs` gate uses: the warm-up runs on a throwaway twin so the
/// process-global caches are hot while the measured tensor still owes its dense
/// buffer (the old `measured_unary_bytes` warm-up could not see a route that
/// materialized, because it materialized the very tensor it then measured).
fn measured_fresh_exp_bytes(degeneracy: usize, dtype: Dtype, seed: u64) -> u64 {
    let warmup = prepare_fixture(degeneracy, dtype, seed);
    black_box(warmup.diagonal.exp().unwrap());
    let fixture = prepare_fixture(degeneracy, dtype, seed);
    measured_bytes(|| fixture.diagonal.exp().unwrap())
}

#[test]
fn diagonal_exp_allocation_bytes_scale_linearly() {
    // What: issue #578. `exp` on compact storage is elementwise over the
    // `Σ_c k_c` stored values, so it allocates its O(r) result and nothing
    // else — no dense payload, no eigendecomposition factors.
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let small = measured_fresh_exp_bytes(64, Dtype::F64, 817);
    let large = measured_fresh_exp_bytes(128, Dtype::F64, 819);
    assert!(
        large <= small * 4,
        "allocation growth is not O(d): d=64 used {small} bytes, d=128 used {large} bytes"
    );
    assert!(
        large < (128 * 128 * std::mem::size_of::<f64>()) as u64,
        "compact exp allocated at least one dense payload: {large} bytes"
    );
    let complex = measured_fresh_exp_bytes(128, Dtype::C64, 821);
    assert!(
        complex < (2 * 128 * 128 * std::mem::size_of::<f64>()) as u64,
        "compact c64 exp allocated at least one dense payload: {complex} bytes"
    );

    // Neither the source nor the image was materialized on the way: each one's
    // *first* `data()` must still have to build the dense buffer. On the old
    // dense route the source paid for it inside `exp` and cached it, and the
    // image came back dense already, so both reads would be free.
    let dense_payload = (128 * 128 * std::mem::size_of::<f64>()) as u64;
    let source = prepare_fixture(128, Dtype::F64, 823).diagonal;
    let image = source.exp().unwrap();
    assert!(
        measured_bytes(|| source.data().len()) >= dense_payload,
        "exp materialized its source behind our back"
    );
    assert!(
        measured_bytes(|| image.data().len()) >= dense_payload,
        "exp returned a materialized image"
    );

    let complex_source = prepare_fixture(128, Dtype::C64, 825).diagonal;
    let complex_image = complex_source.exp().unwrap();
    assert!(
        measured_bytes(|| complex_image.data_c64().len()) >= 2 * dense_payload,
        "c64 exp returned a materialized image"
    );
}
