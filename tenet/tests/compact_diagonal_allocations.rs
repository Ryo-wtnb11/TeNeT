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
