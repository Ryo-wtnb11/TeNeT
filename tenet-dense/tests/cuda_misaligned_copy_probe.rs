//! Tenferro-only probe for the `copy_read_into` destination-alignment defect
//! behind #1320.
//!
//! It calls `tenferro_gpu` / `tenferro_tensor` directly, with no TeNeT
//! production surface involved, so what it records is evidence about the
//! pinned Tenferro release rather than about the adapter (0.5.0 faulted;
//! 0.6.0 carries the fix).
//!
//! A `cudaErrorMisalignedAddress` is a *sticky* CUDA error: once a launch
//! fails, the context is unusable and every later device call in the same
//! process fails too. This probe therefore lives in its own test binary and
//! holds exactly one test, so a fault here cannot poison the rest of the
//! device suite.
//!
//! It records, it does not gate: the destination offsets it walks are exactly
//! the ones the adapter now refuses to send down this route, and Tenferro
//! 0.6.0 carries the fix (tensor4all/tenferro-rs#1836), so a
//! pass here after a dependency bump is the expected outcome, not a
//! regression. Failures are printed per offset and the summary is asserted
//! only to be non-empty.
//!
//! Run with `cargo test -p tenet-dense --features cuda,cpu-faer --test \
//! cuda_misaligned_copy_probe -- --ignored --nocapture` on a CUDA host.

#![cfg(feature = "cuda")]

use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_tensor::{Tensor, TensorRead, TensorScalar, TensorStructural, TensorWrite};

use tenet_dense::CudaScalar;

/// One `copy_read_into` of a compact `rows x cols` source into a destination
/// view at `dst_offset` elements, returning the error text if the submission
/// or the following synchronizing download fails.
fn probe_copy(
    backend: &mut CudaBackend,
    dst_offset: usize,
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    const LEN: usize = 256;
    let source: Vec<f64> = (0..rows * cols).map(|index| 1.0 + index as f64).collect();
    let src_host =
        Tensor::from_vec_col_major(vec![rows * cols], source).map_err(|e| e.to_string())?;
    let dst_host =
        Tensor::from_vec_col_major(vec![LEN], vec![0.0_f64; LEN]).map_err(|e| e.to_string())?;
    let src = upload_tensor(backend.runtime(), &src_host).map_err(|e| e.to_string())?;
    let mut dst = upload_tensor(backend.runtime(), &dst_host).map_err(|e| e.to_string())?;

    {
        let src_view = f64::typed(&src)
            .ok_or("source dtype")?
            .backend_region_view(vec![rows, cols], vec![1, rows as isize], 0)
            .map_err(|e| e.to_string())?;
        let dst_view = f64::typed_mut(&mut dst)
            .ok_or("destination dtype")?
            .backend_region_view_mut(
                vec![rows, cols],
                vec![1, rows as isize],
                dst_offset as isize,
            )
            .map_err(|e| e.to_string())?;
        backend
            .copy_read_into(
                TensorRead::from_view(f64::tensor_view(src_view)),
                TensorWrite::from_view(f64::tensor_view_mut(dst_view)),
            )
            .map_err(|e| format!("copy_read_into: {e}"))?;
    }
    // The copy is asynchronous; a misaligned launch surfaces at the next
    // synchronizing call, which in the original report was the download.
    download_tensor(backend.runtime(), &dst).map_err(|e| format!("download: {e}"))?;
    Ok(())
}

#[test]
#[ignore = "requires a real CUDA device"]
fn records_copy_read_into_destination_offset_behaviour() {
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0))
        .expect("CUDA device 0 must be available for the device probe");

    // f64: the descriptor claims 256-byte alignment, which an element offset
    // satisfies only at multiples of 32. Offsets 9 and 25 are the ones
    // #1320's `[(0,3),(1,2)]` and `[(0,5),(1,2)]` fixtures produce.
    let mut outcomes = Vec::new();
    for &(offset, rows, cols) in &[
        (0usize, 2usize, 2usize),
        (4, 2, 2),
        (9, 2, 2),
        (9, 3, 3),
        (16, 2, 2),
        (25, 2, 2),
        (32, 2, 2),
    ] {
        let outcome = probe_copy(&mut backend, offset, rows, cols);
        println!(
            "copy_read_into f64 dst_offset {offset} ({} bytes) {rows}x{cols}: {}",
            offset * 8,
            match &outcome {
                Ok(()) => "ok".to_string(),
                Err(message) => format!("FAILED: {message}"),
            }
        );
        let failed = outcome.is_err();
        outcomes.push((offset, rows, cols, failed));
        if failed {
            // Sticky: the context is gone, so nothing after this is meaningful.
            println!("context poisoned after the first failure; stopping the sweep");
            break;
        }
    }
    assert!(
        !outcomes.is_empty(),
        "the probe must record at least one case"
    );
}
