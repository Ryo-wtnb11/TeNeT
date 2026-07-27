//! Allocation gate for the general (Padé) arm of `exp` (issue #577).
//!
//! The design's storage claim is `O(Σ_c n_c²)` for the result and
//! `O(max_c n_c²)` for scratch — one workspace sized to the largest coupled
//! sector, reused by every sector, with nothing allocated inside the loop.
//!
//! The gate is differential rather than absolute: adding coupled sectors of the
//! same order must cost about one output block each. A workspace allocated per
//! sector — the obvious way to write this kernel wrong — would instead cost ten
//! blocks each, which is what the bound below rejects.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use tenet::prelude::*;

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

const ORDER: usize = 48;

/// A U(1) endomorphism with `sectors` non-Hermitian blocks of order [`ORDER`].
fn fixture(sectors: i32) -> Tensor {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::u1((0..sectors).map(|charge| (charge, ORDER)));
    // Nonsymmetric in the two degeneracy indices, so every block takes the
    // general arm; scaled small so no block needs squaring and the measurement
    // is not a function of the fill's magnitude.
    Tensor::from_block_fn(&runtime, [&space], [&space], |_, indices| {
        0.01 * (1.0 + indices[0] as f64 - 0.5 * indices[1] as f64)
    })
    .unwrap()
}

fn measured_exp_bytes(tensor: &Tensor) -> u64 {
    // Warm the plan and layout caches on a fresh twin, then on the tensor
    // itself, so the measurement sees only the per-call work.
    black_box(fixture(1).exp().unwrap());
    black_box(tensor.exp().unwrap());
    ALLOCATED.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Release);
    let output = black_box(tensor.exp().unwrap());
    ENABLED.store(false, Ordering::Release);
    black_box(output);
    ALLOCATED.load(Ordering::Relaxed)
}

#[test]
fn general_exp_scratch_does_not_grow_with_the_sector_count() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let block_bytes = (ORDER * ORDER * std::mem::size_of::<f64>()) as u64;

    let two = measured_exp_bytes(&fixture(2));
    let eight = measured_exp_bytes(&fixture(8));

    // One workspace of ten blocks exists at all, sized to the largest sector.
    assert!(
        two >= 10 * block_bytes,
        "the Padé workspace looks absent: two sectors allocated only {two} bytes"
    );
    // Six extra sectors may cost their six output blocks (plus route and region
    // metadata, which is O(sector count) and tiny); a per-sector workspace would
    // cost sixty.
    let growth = eight - two;
    assert!(
        growth <= 12 * block_bytes,
        "allocation grew by {growth} bytes over six added sectors, more than the \
         {} bytes an O(max_c n_c²) workspace allows — scratch is per sector",
        12 * block_bytes
    );
}
