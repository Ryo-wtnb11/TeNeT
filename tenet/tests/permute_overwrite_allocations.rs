use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::{Arc, Mutex};

use tenet::core::{
    product_sector, ProductFusionRuleExt, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, TensorMap};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            BYTES.set(BYTES.get() + layout.size());
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
            BYTES.set(BYTES.get() + new_size);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

fn measure(f: impl FnOnce()) -> (usize, usize) {
    ALLOCATIONS.set(0);
    BYTES.set(0);
    COUNTING.set(true);
    f();
    COUNTING.set(false);
    (ALLOCATIONS.get(), BYTES.get())
}

#[test]
fn cached_permute_overwrite_does_not_allocate_on_the_caller_thread() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: a warmed multiplicity-free non-Abelian permutation reuses its
    // compiled plan and replay workspace without allocating on the caller.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule.product(SU2FusionRule));
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(0)),
                8,
            ),
            (
                product_sector(U1Irrep::new(1), SU2Irrep::from_twice_spin(1)),
                6,
            ),
            (
                product_sector(U1Irrep::new(-1), SU2Irrep::from_twice_spin(1)),
                6,
            ),
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(2)),
                4,
            ),
        ],
        false,
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 197).unwrap();
    let mut destination = source.permute(&[1], &[2, 0]).unwrap();

    source
        .permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0)
        .unwrap();
    let destination_data = destination.data().as_ptr();

    ALLOCATIONS.set(0);
    BYTES.set(0);
    COUNTING.set(true);
    source
        .permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0)
        .unwrap();
    COUNTING.set(false);
    black_box(destination.data());

    assert_eq!((ALLOCATIONS.get(), BYTES.get()), (0, 0));
    assert_eq!(destination.data().as_ptr(), destination_data);
    assert!(std::ptr::eq(destination.provider(), provider.as_ref()));
}

#[test]
fn cached_u1_permute_overwrite_does_not_allocate_on_the_caller_thread() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: a warmed UniqueFusion permutation reuses its completed transformer
    // and replay workspace without allocating on the caller.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (U1Irrep::new(-1), 4),
            (U1Irrep::new(0), 8),
            (U1Irrep::new(1), 4),
        ],
        false,
    )
    .unwrap();
    let source: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 418).unwrap();
    let mut destination = source.permute(&[1], &[2, 0]).unwrap();

    source
        .permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0)
        .unwrap();
    let destination_data = destination.data().as_ptr();

    ALLOCATIONS.set(0);
    BYTES.set(0);
    COUNTING.set(true);
    source
        .permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0)
        .unwrap();
    COUNTING.set(false);
    black_box(destination.data());

    assert_eq!((ALLOCATIONS.get(), BYTES.get()), (0, 0));
    assert_eq!(destination.data().as_ptr(), destination_data);
    assert!(std::ptr::eq(destination.provider(), provider.as_ref()));
}

#[test]
fn cached_planar_overwrites_do_not_allocate_on_the_caller_thread() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: the shared typed destination seam also reuses admitted full,
    // explicit, and repartition transpose operations without caller allocation.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
        false,
    )
    .unwrap();
    let source: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 779).unwrap();

    let mut full = source.transpose().unwrap();
    source.transpose_overwrite_into(&mut full, 1.0).unwrap();
    assert_eq!(
        measure(|| source.transpose_overwrite_into(&mut full, 1.0).unwrap()),
        (0, 0)
    );

    let mut explicit = source.transpose_axes(&[1, 2], &[0]).unwrap();
    source
        .transpose_axes_overwrite_into(&mut explicit, &[1, 2], &[0], 1.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .transpose_axes_overwrite_into(&mut explicit, &[1, 2], &[0], 1.0)
                .unwrap()
        }),
        (0, 0)
    );

    let mut repartitioned = source.repartition(1).unwrap();
    source
        .repartition_overwrite_into(&mut repartitioned, 1.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .repartition_overwrite_into(&mut repartitioned, 1.0)
                .unwrap()
        }),
        (0, 0)
    );
    assert!(std::ptr::eq(full.provider(), provider.as_ref()));
    assert!(std::ptr::eq(explicit.provider(), provider.as_ref()));
    assert!(std::ptr::eq(repartitioned.provider(), provider.as_ref()));
}
