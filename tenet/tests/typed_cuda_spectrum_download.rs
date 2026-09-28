//! Host/device traffic of the device diagonal factorizations, whatever the
//! number of coupled sectors.
//!
//! - Compact SVD writes each sector's singular values into `s` on the device
//!   (#1536): it downloads nothing, and uploads only the three zero-initialized
//!   factors (the only device allocation path until #740), so `s` costs no
//!   upload beyond its zero initialization.
//! - Full EIGH must read its eigenvalues on the host (non-finite check, `|λ|`
//!   order, factor-space plan), all sectors' with one download (#1484), and
//!   uploads `d` once, already filled, together with `v`'s zeros and one
//!   selector (its dataflow is unchanged by #1536).
//!
//! This file holds a single test because it reads the process-wide
//! [`cuda_transfer_stats`] counters.
//!
//! Run with `cargo test -p tenet-rs --features cuda,cpu-faer --test \
//! typed_cuda_spectrum_download -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use tenet::expert::{cuda_transfer_stats, CudaTransferStats};
use tenet::sector::{U1FusionRule, U1Irrep};
use tenet::typed::{Eigh, GradedSpace, Runtime, Svd, TensorMap};

/// The receiver's own split as leg roles: `rows = 0..nout`.
fn codomain_axes<R, D, S>(t: &tenet::typed::TensorMap<R, D, S>) -> Vec<usize> {
    (0..t.codomain_rank()).collect()
}

/// The receiver's own split as leg roles: `cols = nout..rank`.
fn domain_axes<R, D, S>(t: &tenet::typed::TensorMap<R, D, S>) -> Vec<usize> {
    (t.codomain_rank()..t.rank()).collect()
}

fn delta(before: CudaTransferStats) -> CudaTransferStats {
    let after = cuda_transfer_stats();
    CudaTransferStats {
        h2d_calls: after.h2d_calls - before.h2d_calls,
        h2d_bytes: after.h2d_bytes - before.h2d_bytes,
        d2h_calls: after.d2h_calls - before.d2h_calls,
        d2h_bytes: after.d2h_bytes - before.d2h_bytes,
        gauge_ops: after.gauge_ops - before.gauge_ops,
        ..CudaTransferStats::default()
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_diagonal_factors_transfer_only_what_the_host_decides_on() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let bytes = |len: usize| (len * std::mem::size_of::<f64>()) as u64;
    for charges in [1_usize, 9] {
        let half = charges as i32 / 2;
        let leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            (-half..=half).map(|charge| (U1Irrep::new(charge), 3)),
        )
        .unwrap();
        let host = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            1.0 + ((index[0] * 3 + index[1] * 5) % 7) as f64
        })
        .unwrap();
        let device = host.to_cuda().unwrap();

        let before = cuda_transfer_stats();
        let Svd { u, s, vh } = device
            .svd_compact(&codomain_axes(&device), &domain_axes(&device))
            .unwrap();
        let svd = delta(before);
        let (u, s, vh) = (
            u.to_host().unwrap(),
            s.to_host().unwrap(),
            vh.to_host().unwrap(),
        );
        assert_eq!((svd.d2h_calls, svd.d2h_bytes), (0, 0), "{charges} sectors");
        // Exactly the three zero uploads plus one upload of the gauge's
        // row weights (every route has 3 rows), so `s` adds nothing beyond
        // its zero initialization.
        assert_eq!(svd.h2d_calls, 4, "{charges} sectors");
        assert_eq!(
            svd.h2d_bytes,
            bytes(
                u.materialize().unwrap().dense_data().unwrap().len()
                    + s.materialize().unwrap().dense_data().unwrap().len()
                    + vh.materialize().unwrap().dense_data().unwrap().len()
                    + 3
            ),
            "{charges} sectors"
        );
        // Per aligned real route: 13 phase ops, then a broadcast and a mul on
        // each side; the gauge never downloads (checked above).
        assert_eq!(svd.gauge_ops, 17 * charges as u64, "{charges} sectors");
        let Svd { s: expected, .. } = host
            .svd_compact(&codomain_axes(&host), &domain_axes(&host))
            .unwrap();
        for (device, host) in s
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.materialize().unwrap().dense_data().unwrap())
        {
            assert!((device - host).abs() <= 1e-12 * host.abs().max(1.0));
        }

        let hermitian = host.axpby(1.0, &host.adjoint().unwrap(), 1.0).unwrap();
        let device = hermitian.to_cuda().unwrap();
        let before = cuda_transfer_stats();
        let Eigh { d, v } = device
            .eigh_full(&codomain_axes(&device), &domain_axes(&device))
            .unwrap();
        let eigh = delta(before);
        let (d, v) = (d.to_host().unwrap(), v.to_host().unwrap());
        let eigenvalues = 3 * charges;
        // The eigenvalues must reach the host; the rest of the download is
        // the O(1)-per-sector Hermiticity verdicts.
        assert!(eigh.d2h_bytes >= bytes(eigenvalues), "{charges}: {eigh:?}");
        // `d` arrives filled in its one dense upload, beside `v`'s zeros and
        // the `n_c x n_c` selector; the remainder is per-sector scalars.
        assert!(
            eigh.h2d_bytes
                >= bytes(
                    2 * d.materialize().unwrap().dense_data().unwrap().len()
                        + v.materialize().unwrap().dense_data().unwrap().len()
                ),
            "{charges}: {eigh:?}"
        );
        let Eigh { d: expected, .. } = hermitian
            .eigh_full(&codomain_axes(&hermitian), &domain_axes(&hermitian))
            .unwrap();
        for (device, host) in d
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.materialize().unwrap().dense_data().unwrap())
        {
            assert!((device - host).abs() <= 1e-10 * host.abs().max(1.0));
        }
    }
}
