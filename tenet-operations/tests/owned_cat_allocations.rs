//! Allocation gate for the owned concatenation writer: a strided-row
//! (transposed-source) copy executes through strided's compiled copy plan
//! into the single output payload with no further allocation.

use std::hint::black_box;

use tenet_operations::{try_cat_owned_raw, OwnedCatCopy, OwnedCatSide};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

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
    counting_alloc::start();
    let output = black_box(run());
    let allocs = counting_alloc::stop();

    assert_eq!(allocs.calls, 1);
    assert_eq!(output, warm);
    for column in 0..lhs_cols {
        for row in 0..rows {
            assert_eq!(output[column * rows + row], lhs[row * lhs_cols + column]);
        }
    }
    assert_eq!(output[rows * lhs_cols..], rhs[..]);
}
