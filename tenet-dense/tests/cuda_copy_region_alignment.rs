//! Device gates for `cuda_copy_region_into`'s destination-offset routing
//! (#1320).
//!
//! Tenferro 0.5.0's `copy_read_into` advertised the *allocation's* 256-byte
//! alignment to cuTENSOR even for a destination view that starts mid-buffer
//! (`tenferro-gpu-0.5.0/src/cubecl/permutation.rs:723`), so cuTENSOR could pick
//! a vectorized kernel the shifted pointer cannot satisfy and the launch failed
//! with `cudaErrorMisalignedAddress`. Tenferro 0.6.0 reports the true
//! alignment (`permutation.rs:771`); the guard is kept pending leaf M2, and
//! these gates stay its regression fixtures. The adapter routes a
//! destination whose byte offset is not a multiple of 256 through the region
//! primitive, whose contraction descriptors report the truthful per-element
//! view alignment.
//!
//! The oracle here is a host scatter over the same index space; nothing is
//! derived from the adapter's own metadata. The routing expectation is
//! recomputed from the byte offset rather than read from the predicate under
//! test.
//!
//! Run with `cargo test -p tenet-dense --features cuda,cpu-faer --test \
//! cuda_copy_region_alignment -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::fmt::Debug;
use std::sync::Mutex;

use num_complex::{Complex32, Complex64};
use tenet_dense::{
    cuda_copy_region_into, cuda_transfer_stats, reset_cuda_transfer_stats, CudaDenseContext,
    CudaDenseStorage, CudaScalar,
};

/// The boundary counters are process-wide, so the routing assertions must not
/// overlap.
static COUNTER_TESTS: Mutex<()> = Mutex::new(());

/// The alignment Tenferro's permutation descriptor claims for both operands.
const CUTENSOR_DESCRIPTOR_ALIGNMENT: usize = 256;

trait CopyScalar: CudaScalar + Copy + Debug + PartialEq {
    const NAME: &'static str;
    fn sample(index: usize) -> Self;
    fn sentinel(index: usize) -> Self;
}

impl CopyScalar for f32 {
    const NAME: &'static str = "f32";

    fn sample(index: usize) -> Self {
        (1.0 + index as f64 * 0.25) as Self
    }

    fn sentinel(index: usize) -> Self {
        (-1000.0 - index as f64) as Self
    }
}

impl CopyScalar for Complex32 {
    const NAME: &'static str = "Complex32";

    fn sample(index: usize) -> Self {
        Complex32::new(
            (1.0 + index as f64 * 0.25) as f32,
            (-0.5 - index as f64 * 0.125) as f32,
        )
    }

    fn sentinel(index: usize) -> Self {
        Complex32::new((-1000.0 - index as f64) as f32, (7.0 + index as f64) as f32)
    }
}

impl CopyScalar for f64 {
    const NAME: &'static str = "f64";

    fn sample(index: usize) -> Self {
        1.0 + index as f64 * 0.25
    }

    fn sentinel(index: usize) -> Self {
        -1000.0 - index as f64
    }
}

impl CopyScalar for Complex64 {
    const NAME: &'static str = "Complex64";

    fn sample(index: usize) -> Self {
        Complex64::new(1.0 + index as f64 * 0.25, -0.5 - index as f64 * 0.125)
    }

    fn sentinel(index: usize) -> Self {
        Complex64::new(-1000.0 - index as f64, 7.0 + index as f64)
    }
}

fn context() -> CudaDenseContext {
    CudaDenseContext::new(0).expect("CUDA device 0 must be available for the device suite")
}

/// One copy of a compact `rows x cols` source into `dst[offset..]` with
/// leading dimension `ld`, checked bitwise against a host scatter and checked
/// for which of the two routes ran.
fn copy_case<D: CopyScalar>(
    ctx: &mut CudaDenseContext,
    offset: usize,
    rows: usize,
    cols: usize,
    ld: usize,
) {
    const LEN: usize = 512;
    assert!(offset + rows + (cols - 1) * ld <= LEN);

    let source: Vec<D> = (0..rows * cols).map(D::sample).collect();
    let initial: Vec<D> = (0..LEN).map(D::sentinel).collect();
    let mut expected = initial.clone();
    for col in 0..cols {
        for row in 0..rows {
            expected[offset + row + col * ld] = source[row + col * rows];
        }
    }

    let src = CudaDenseStorage::upload::<D>(ctx, &source).expect("upload source");
    let mut dst = CudaDenseStorage::upload::<D>(ctx, &initial).expect("upload destination");

    let before = cuda_transfer_stats();
    cuda_copy_region_into::<D>(ctx, &mut dst, offset, ld, &src, rows, cols).unwrap_or_else(|err| {
        panic!(
            "{} copy at offset {offset} ({rows}x{cols}, ld {ld}) failed: {err}",
            D::NAME
        )
    });
    let actual = dst.download::<D>(ctx).expect("download");
    let after = cuda_transfer_stats();

    for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            got,
            want,
            "{}: element {index} after a {rows}x{cols} copy at offset {offset} (ld {ld})",
            D::NAME
        );
    }

    let byte_offset = offset * std::mem::size_of::<D>();
    let takes_region_route = !byte_offset.is_multiple_of(CUTENSOR_DESCRIPTOR_ALIGNMENT);
    assert_eq!(
        after.copy_calls - before.copy_calls,
        1,
        "{}: every route is one copy call (offset {offset})",
        D::NAME
    );
    assert_eq!(
        after.gemm_calls - before.gemm_calls,
        u64::from(takes_region_route),
        "{}: offset {offset} ({byte_offset} bytes) took the wrong route",
        D::NAME
    );
}

/// The offsets a multi-sector tensor actually produces, including the 9 and 25
/// of #1320's `[(0,3),(1,2)]` fixture, against square and rectangular blocks
/// and against a leading dimension wider than the block.
fn alignment_sweep<D: CopyScalar>(ctx: &mut CudaDenseContext) {
    let aligned = CUTENSOR_DESCRIPTOR_ALIGNMENT / std::mem::size_of::<D>();
    for &offset in &[
        0,
        1,
        2,
        3,
        9,
        25,
        aligned - 1,
        aligned,
        aligned + 1,
        2 * aligned,
    ] {
        for &(rows, cols) in &[(1, 1), (2, 2), (3, 3), (2, 3), (3, 2), (4, 1), (1, 4)] {
            copy_case::<D>(ctx, offset, rows, cols, rows);
            copy_case::<D>(ctx, offset, rows, cols, rows + 2);
        }
    }
}

/// #1320: a destination at an odd element offset with an even extent is the
/// exact shape that faulted; every offset in the sweep must now both produce
/// the right bytes and take the route its byte offset admits.
///
/// All four payload dtypes, because the route is decided by the *byte* offset:
/// 4-byte and 8-byte elements send the same element offset down different
/// routes (offset 32 is the permute route for `f64`/[`Complex32`] and the
/// region route for `f32`), so neither element size proves the other.
#[test]
#[ignore = "requires a real CUDA device"]
fn copy_region_is_correct_at_every_destination_offset() {
    let _guard = COUNTER_TESTS.lock().unwrap_or_else(|err| err.into_inner());
    reset_cuda_transfer_stats();
    let mut ctx = context();
    alignment_sweep::<f32>(&mut ctx);
    alignment_sweep::<f64>(&mut ctx);
    alignment_sweep::<Complex32>(&mut ctx);
    alignment_sweep::<Complex64>(&mut ctx);
}
