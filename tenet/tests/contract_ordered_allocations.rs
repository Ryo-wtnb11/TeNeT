use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use tenet::prelude::*;

struct CountingAllocator;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicU64 = AtomicU64::new(0);

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

fn measure_allocated_bytes(f: impl FnOnce()) -> u64 {
    ALLOCATED.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Release);
    f();
    ENABLED.store(false, Ordering::Release);
    ALLOCATED.load(Ordering::Relaxed)
}

#[test]
fn ordered_contract_does_not_allocate_the_default_order_owned_payload() {
    // What: after warming both routes, the ordered API avoids the complete
    // default-order owned tensor retained by explicit contract-then-permute.
    let runtime = Runtime::builder()
        .dense_threads(1)
        .recoupling_threads(1)
        .build()
        .unwrap();
    let space = Space::su2([(0, 4), (1, 6), (2, 4)]);
    let lhs = Tensor::rand_with_seed(
        &runtime,
        Dtype::F64,
        [&space, &space],
        [&space, &space],
        2241,
    )
    .unwrap();
    let rhs = Tensor::rand_with_seed(
        &runtime,
        Dtype::F64,
        [&space, &space],
        [&space, &space],
        2242,
    )
    .unwrap();
    let lhs_axes = [3, 2];
    let rhs_axes = [0, 1];
    let output_axes = [1, 0, 2, 3];

    let sequential = || {
        let default = lhs.contract(&rhs, &lhs_axes, &rhs_axes).unwrap();
        black_box(default.permute(&output_axes[..2], &output_axes[2..]).unwrap())
    };
    let fused = || {
        black_box(
            lhs.contract_ordered(&rhs, &lhs_axes, &rhs_axes, &output_axes)
                .unwrap(),
        )
    };

    black_box(sequential());
    black_box(fused());
    let sequential_bytes = measure_allocated_bytes(|| {
        black_box(sequential());
    });
    let fused_bytes = measure_allocated_bytes(|| {
        black_box(fused());
    });
    let owned_payload_bytes = lhs
        .contract(&rhs, &lhs_axes, &rhs_axes)
        .unwrap()
        .data()
        .len() as u64
        * std::mem::size_of::<f64>() as u64;

    assert!(
        fused_bytes + owned_payload_bytes <= sequential_bytes,
        "ordered={fused_bytes} B, sequential={sequential_bytes} B, \
         default owned payload={owned_payload_bytes} B"
    );
}
