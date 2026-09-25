//! Device compact SVD downloads its singular values once per call, whatever
//! the number of coupled sectors (#1484). Its only download is the spectrum,
//! so the whole call makes exactly one D2H.
//!
//! This file holds a single test because it reads the process-wide
//! [`cuda_transfer_stats`] counters.
//!
//! Run with `cargo test -p tenet --features cuda,cpu-faer --test \
//! typed_cuda_spectrum_download -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::dense::cuda_transfer_stats;
use tenet::typed::{GradedSpace, Runtime, TensorMap};

#[test]
#[ignore = "requires a real CUDA device"]
fn compact_svd_downloads_its_spectrum_once_whatever_the_sector_count() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    for charges in [1_usize, 9] {
        let half = charges as i32 / 2;
        let leg = GradedSpace::try_new_with_arc(
            Arc::new(U1FusionRule),
            (-half..=half).map(|charge| (U1Irrep::new(charge), 3)),
        )
        .unwrap();
        let host = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            1.0 + ((index[0] * 3 + index[1] * 5) % 7) as f64
        })
        .unwrap();
        let device = host.to_cuda().unwrap();

        let before = cuda_transfer_stats().d2h_calls;
        let (_, s, _) = device.svd_compact().unwrap();
        let downloads = cuda_transfer_stats().d2h_calls - before;
        assert_eq!(downloads, 1, "{charges} sectors: one spectrum download");

        let (_, expected, _) = host.svd_compact().unwrap();
        let s = s.to_host().unwrap();
        for (device, host) in s.data().iter().zip(expected.data()) {
            assert!((device - host).abs() <= 1e-12 * host.abs().max(1.0));
        }
    }
}
