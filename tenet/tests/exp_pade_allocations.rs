//! Allocation gate for the general (Padé) arm of `exp` (issue #577).
//!
//! The design's storage claim is an `O(max_c n_c²)` workspace — sized to the
//! *largest* coupled sector, reused by every sector, with nothing allocated
//! inside the sector loop. On the canonical direct-region layout the fixture
//! below builds, that workspace is the whole of the scratch; the fallback for
//! a payload whose sectors are not contiguous regions matricizes all of them
//! first and costs `O(Σ_c n_c²)` besides, which is not what this gate
//! measures.
//!
//! Measuring that directly is not possible from outside: a call to `exp` also
//! pays for its own result and for the layout and backend work every route
//! pays, and the dense backend allocates per GEMM. What *is* separable is the
//! shape of the total. Allocation at a fixed block order is affine in the
//! sector count,
//!
//! ```text
//! bytes(sectors) = intercept + sectors * slope
//! ```
//!
//! and the workspace is the whole of the intercept: charged once, so it sits
//! outside the sector term. Moving it inside the loop — the obvious way to
//! write this kernel wrong — moves ten blocks from the intercept into the
//! slope, and sizing it to `Σ_c n_c²` instead of `max_c n_c²` does the same.
//! The gate therefore reads the intercept off two sector counts and checks that
//! it is exactly one workspace, at two block orders so that "one workspace" is
//! also tested against the block order it must scale with.

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

/// Buffers in the Padé workspace, each `max_c n_c²`.
const WORKSPACE_BLOCKS: f64 = 10.0;

/// Blocks a single added coupled sector may cost: its own output block plus the
/// dense backend's per-call scratch, and nothing of the workspace.
const PER_SECTOR_BLOCK_BUDGET: f64 = 26.0;

/// A U(1) endomorphism with `sectors` non-Hermitian blocks of order `order`.
fn fixture(sectors: i32, order: usize) -> Tensor {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::u1((0..sectors).map(|charge| (charge, order)));
    // Nonsymmetric in the two degeneracy indices, so every block takes the
    // general arm; scaled small so no block needs squaring and the measurement
    // does not depend on the fill's magnitude.
    Tensor::from_block_fn(&runtime, [&space], [&space], |_, indices| {
        0.01 * (1.0 + indices[0] as f64 - 0.5 * indices[1] as f64)
    })
    .unwrap()
}

fn measured_exp_bytes(sectors: i32, order: usize) -> u64 {
    let tensor = fixture(sectors, order);
    // Warm the plan and layout caches on a fresh twin of the same shape, then
    // on the tensor itself, so the measurement sees only the per-call work.
    black_box(fixture(sectors, order).exp().unwrap());
    black_box(tensor.exp().unwrap());
    ALLOCATED.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Release);
    let output = black_box(tensor.exp().unwrap());
    ENABLED.store(false, Ordering::Release);
    black_box(output);
    ALLOCATED.load(Ordering::Relaxed)
}

#[test]
fn general_exp_scratch_is_one_workspace_sized_to_the_largest_sector() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();

    for order in [48usize, 96] {
        let block_bytes = (order * order * std::mem::size_of::<f64>()) as f64;
        let two = measured_exp_bytes(2, order) as f64;
        let four = measured_exp_bytes(4, order) as f64;
        // bytes(s) = intercept + s * slope, read off two points.
        let intercept = 2.0 * two - four;
        let blocks = intercept / block_bytes;
        assert!(
            (WORKSPACE_BLOCKS - 1.0..=WORKSPACE_BLOCKS + 1.0).contains(&blocks),
            "order {order}: the sector-count-independent allocation is {blocks:.2} blocks, \
             not the {WORKSPACE_BLOCKS} of a single max-sector Padé workspace \
             (two sectors: {two} bytes, four: {four})"
        );
        // And the sector term itself, which a *second* workspace allocated
        // inside the loop would inflate by ten blocks while leaving the
        // intercept alone. The per-sector cost is mostly the dense backend's
        // own scratch for six GEMMs and a solve, so the bound is loose: it
        // measures 17.5 blocks at order 48 and 21.0 at order 96 on the shipped
        // executor, against the 27.5 / 31.0 a per-sector workspace would cost.
        let slope = (four - two) / 2.0 / block_bytes;
        assert!(
            slope <= PER_SECTOR_BLOCK_BUDGET,
            "order {order}: each added sector allocates {slope:.2} blocks, over the \
             {PER_SECTOR_BLOCK_BUDGET} budget — scratch is being taken per sector"
        );
    }
}
