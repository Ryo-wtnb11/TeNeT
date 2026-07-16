//! #41 metric gate: the non-baked strided copy/scale/axpy path (delegated to
//! strided-rs #140 `copy_scale_raw` / `axpy_raw`) must stay allocation-free for
//! rank <= 8, matching the hand-rolled `fused_pair` it replaced. The primary
//! #41 requirement is "the paths stay allocation-free"; this pins it through the
//! public `StridedHostKernelAdapter` (baked = None, so it exercises the raw
//! delegation) so a regression to an allocating strided entry point fails here.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tenet_operations::{HostKernelAdapter, StridedHostKernelAdapter};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(ptr, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn non_baked_rank4_copy_and_axpy_do_not_allocate() {
    let mut adapter = StridedHostKernelAdapter::default();
    let shape = [4usize, 4, 4, 4];
    let src_strides = [1isize, 4, 16, 64]; // column-major source
    let dst_strides = [64isize, 16, 4, 1]; // transposed destination
    let src: Vec<f64> = (0..256).map(|i| i as f64 * 0.5 - 3.0).collect();
    let mut dst = vec![0.0f64; 256];
    let mut zero_strides: Vec<isize> = Vec::new();

    // Warm up any one-time strided-rs setup before counting.
    adapter
        .copy_scale_strided(
            &mut dst,
            &src,
            &shape,
            &dst_strides,
            &src_strides,
            0,
            0,
            false,
            1.0,
        )
        .unwrap();

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    for _ in 0..64 {
        // pack (assign, baked = None -> copy_scale_raw)
        adapter
            .copy_scale_strided(
                &mut dst,
                &src,
                &shape,
                &dst_strides,
                &src_strides,
                0,
                0,
                false,
                1.0,
            )
            .unwrap();
        // tensoradd accumulate (beta = 1, baked = None -> axpy_raw)
        adapter
            .add_strided(
                &mut zero_strides,
                &mut dst,
                &src,
                &shape,
                &dst_strides,
                &src_strides,
                0,
                0,
                false,
                1.0,
                1.0,
            )
            .unwrap();
    }
    COUNTING.set(false);

    // What: the rank<=8 raw-kernel delegation borrows already-validated layout
    // metadata (RawStrided{Ref,Mut}) and fuses on the stack, so no call heap-
    // allocates. Fails if a delegated path routes through an allocating strided
    // view kernel.
    assert_eq!(
        ALLOCATIONS.get(),
        0,
        "delegated non-baked rank-4 copy/axpy must be allocation-free"
    );
}
