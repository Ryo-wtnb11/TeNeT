//! Host-allocation contract of the CUDA transfer boundary.
//!
//! The probe counts caller-thread allocations whose size is exactly the
//! payload length in bytes, which is the only allocation class a redundant
//! payload copy can belong to; every other allocation Tenferro, CubeCL and the
//! CUDA driver make is a different size and is ignored. The device is required
//! because `upload`/`download` are the units under test, so these are
//! `#[ignore]` like the rest of the device suite.

#![cfg(feature = "cuda")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;

use num_complex::Complex64;
use tenet_dense::{cuda_transfer_stats, CudaDenseContext, CudaDenseStorage, CudaScalar};

/// Payload-sized host allocations a download makes on the caller thread: the
/// one vector Tenferro decodes the readback bytes into, handed straight back.
/// Measured on an A100 with Tenferro 0.5.0: 2 before this change (TeNeT copied
/// that vector once more), 1 after, for both f64 and `Complex64`.
///
/// The upload side is asserted as a difference instead of an absolute count:
/// how many payload-sized buffers CubeCL stages through is its own affair (2
/// for f64, 3 for `Complex64` on the same host), while the copy TeNeT adds for
/// a borrowed slice is exactly one, on every dtype and every backend version.
const DOWNLOAD_PAYLOAD_ALLOCATIONS: usize = 1;

struct PayloadProbe;

thread_local! {
    static TARGET_BYTES: Cell<usize> = const { Cell::new(0) };
    static HITS: Cell<usize> = const { Cell::new(0) };
}

fn record(size: usize) {
    if TARGET_BYTES.get() == size {
        HITS.set(HITS.get() + 1);
    }
}

unsafe impl GlobalAlloc for PayloadProbe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() {
            record(new_size);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: PayloadProbe = PayloadProbe;

static DEVICE: Mutex<()> = Mutex::new(());

/// Runs `f` while counting caller-thread allocations of exactly
/// `payload_bytes`. A zero target would match nothing, so it is rejected.
fn measure<T>(payload_bytes: usize, f: impl FnOnce() -> T) -> (T, usize) {
    assert_ne!(payload_bytes, 0, "the probe needs a nonzero payload size");
    HITS.set(0);
    TARGET_BYTES.set(payload_bytes);
    let output = f();
    TARGET_BYTES.set(0);
    (output, HITS.get())
}

/// Element count large enough to leave Tenferro's pinned small-payload
/// readback path (`PINNED_SCALAR_BYTES`) and odd enough that no incidental
/// allocation happens to share the payload size.
const ELEMENTS: usize = 4099;

fn assert_transfer_copies<D>(ctx: &CudaDenseContext, data: Vec<D>)
where
    D: CudaScalar + PartialEq + std::fmt::Debug,
{
    let payload_bytes = std::mem::size_of_val(data.as_slice());

    let borrowed_before = cuda_transfer_stats();
    let (borrowed, borrowed_hits) = measure(payload_bytes, || {
        CudaDenseStorage::upload::<D>(ctx, &data).unwrap()
    });
    let borrowed_upload = stats_delta(borrowed_before, cuda_transfer_stats());

    let owned_source = data.clone();
    let owned_before = cuda_transfer_stats();
    let (owned, owned_hits) = measure(payload_bytes, || {
        CudaDenseStorage::upload_owned::<D>(ctx, owned_source).unwrap()
    });
    let owned_upload = stats_delta(owned_before, cuda_transfer_stats());

    assert_eq!(
        borrowed_hits,
        owned_hits + 1,
        "a borrowed upload of {payload_bytes} bytes costs exactly one host copy \
         more than an owned one"
    );
    assert_eq!(
        owned_upload, borrowed_upload,
        "device traffic must be identical for owned and borrowed uploads"
    );

    let download_before = cuda_transfer_stats();
    let (values, download_hits) = measure(payload_bytes, || owned.download::<D>(ctx).unwrap());
    let download = stats_delta(download_before, cuda_transfer_stats());

    assert_eq!(values, data, "download must return the uploaded payload");
    assert_eq!(
        download_hits, DOWNLOAD_PAYLOAD_ALLOCATIONS,
        "download must hand back Tenferro's vector instead of copying it"
    );
    assert_eq!(download.d2h_calls, 1);
    assert_eq!(download.d2h_bytes, payload_bytes as u64);

    let (round_trip, _) = measure(payload_bytes, || borrowed.download::<D>(ctx).unwrap());
    assert_eq!(round_trip, data);
}

fn stats_delta(
    before: tenet_dense::CudaTransferStats,
    after: tenet_dense::CudaTransferStats,
) -> tenet_dense::CudaTransferStats {
    tenet_dense::CudaTransferStats {
        h2d_calls: after.h2d_calls - before.h2d_calls,
        h2d_bytes: after.h2d_bytes - before.h2d_bytes,
        d2h_calls: after.d2h_calls - before.d2h_calls,
        d2h_bytes: after.d2h_bytes - before.d2h_bytes,
        device_allocs: after.device_allocs - before.device_allocs,
        gemm_calls: after.gemm_calls - before.gemm_calls,
        solver_calls: after.solver_calls - before.solver_calls,
        copy_calls: after.copy_calls - before.copy_calls,
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn owned_f64_transfers_cost_no_redundant_host_copy() {
    let _serialized = DEVICE.lock().unwrap_or_else(|err| err.into_inner());
    let ctx = CudaDenseContext::new(0).unwrap();
    let data: Vec<f64> = (0..ELEMENTS).map(|index| index as f64).collect();
    assert_transfer_copies(&ctx, data);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn owned_complex64_transfers_cost_no_redundant_host_copy() {
    let _serialized = DEVICE.lock().unwrap_or_else(|err| err.into_inner());
    let ctx = CudaDenseContext::new(0).unwrap();
    let data: Vec<Complex64> = (0..ELEMENTS)
        .map(|index| Complex64::new(index as f64, -(index as f64) - 0.5))
        .collect();
    assert_transfer_copies(&ctx, data);
}
