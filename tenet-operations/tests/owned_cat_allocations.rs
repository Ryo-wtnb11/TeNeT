//! Allocation gate for the owned concatenation writer: a strided-row
//! (transposed-source) copy executes through strided's compiled copy plan
//! into the single output payload with no further allocation.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenet_operations::{try_cat_owned_raw, OwnedCatCopy, OwnedCatSide};

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

#[test]
fn strided_row_cat_allocates_only_the_output_payload() {
    // What: a domain cat of a transposed 48x40 slab (source row stride 40)
    // and a contiguous 48x24 slab allocates exactly the output vector. The
    // 1920-element strided copy stays below strided's parallel threshold, so
    // this pins the serial replay.
    let (rows, lhs_cols, rhs_cols) = (48, 40, 24);
    let lhs: Vec<f64> = (0..rows * lhs_cols).map(|value| value as f64).collect();
    let rhs: Vec<f64> = (0..rows * rhs_cols).map(|value| -(value as f64)).collect();
    let len = rows * (lhs_cols + rhs_cols);
    let copies = [
        OwnedCatCopy::new(
            0,
            0,
            0,
            [rows, lhs_cols],
            [lhs_cols, 1],
            rows,
            0..len,
            false,
        ),
        OwnedCatCopy::new(
            1,
            0,
            rows * lhs_cols,
            [rows, rhs_cols],
            [1, rows],
            rows,
            0..len,
            false,
        ),
    ];
    let run = || try_cat_owned_raw(len, OwnedCatSide::Domain, &copies, [&lhs, &rhs]).unwrap();
    let warm = run();

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let output = black_box(run());
    COUNTING.set(false);

    assert_eq!(ALLOCATIONS.get(), 1);
    assert_eq!(output, warm);
    for column in 0..lhs_cols {
        for row in 0..rows {
            assert_eq!(output[column * rows + row], lhs[row * lhs_cols + column]);
        }
    }
    assert_eq!(output[rows * lhs_cols..], rhs[..]);
}
