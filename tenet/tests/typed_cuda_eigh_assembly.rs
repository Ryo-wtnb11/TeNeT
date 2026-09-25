//! Device `eigh_full` assembles each layout-aligned coupled sector with one
//! GEMM and uploads one selector per call, whatever the number of fusion trees
//! (#1485).
//!
//! This file holds a single test because it reads the process-wide
//! [`cuda_transfer_stats`] counters.
//!
//! Run with `cargo test -p tenet-rs --features cuda,cpu-faer --test \
//! typed_cuda_eigh_assembly -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::dense::cuda_transfer_stats;
use tenet::typed::{GradedSpace, Runtime, TensorMap};

fn leg(charges: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        charges
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn close(got: &TensorMap<U1FusionRule, f64>, want: &TensorMap<U1FusionRule, f64>) {
    assert_eq!(got.data().len(), want.data().len());
    for (got, want) in got.data().iter().zip(want.data()) {
        assert!((got - want).abs() <= 1e-10, "{got} vs {want}");
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn eigh_assembly_is_one_gemm_per_sector_and_one_selector_upload_per_call() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let small = leg(&[(-1, 2), (0, 1), (1, 2)]);
    // The same five coupled sectors with the same matrix sizes (4, 4, 9, 4,
    // 4), reached through one codomain tree each (`fused`) or through nine
    // trees in total (`small x small`).
    let fused = leg(&[(-2, 4), (-1, 4), (0, 9), (1, 4), (2, 4)]);
    let one_tree = TensorMap::rand_with_seed(&runtime, [&fused], [&fused], 7).unwrap();
    let many_trees =
        TensorMap::rand_with_seed(&runtime, [&small, &small], [&small, &small], 7).unwrap();

    let mut counts = Vec::new();
    for source in [one_tree, many_trees] {
        // Exactly Hermitian, so device admission skips its residual stage and
        // uploads one normalizer per block (#1486).
        let source = source.add(&source.adjoint().unwrap(), 1.0, 1.0).unwrap();
        let device = source.to_cuda().unwrap();

        let before = cuda_transfer_stats();
        let (d, v) = device.eigh_full().unwrap();
        let after = cuda_transfer_stats();
        let gemms = after.gemm_calls - before.gemm_calls;
        let uploads = after.h2d_calls - before.h2d_calls;
        // One GEMM per coupled sector; five normalizers, the diagonal, the
        // zero eigenvector buffer and one selector.
        assert_eq!((gemms, uploads), (5, 5 + 3));
        counts.push(after.h2d_bytes - before.h2d_bytes);

        // Device vs host: the same descending-|λ| spectrum, and the device
        // eigenvectors (raw cuSOLVER gauge) satisfy the eigen equation.
        let (host_d, _) = source.eigh_full().unwrap();
        let d = d.to_host().unwrap();
        let v = v.to_host().unwrap();
        close(&d, &host_d);
        close(&source.compose(&v).unwrap(), &v.compose(&d).unwrap());
    }
    assert_eq!(
        counts[0], counts[1],
        "upload bytes do not depend on the trees"
    );
}
