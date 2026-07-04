//! User-layer CUDA runtime tests.
//!
//! The `#[ignore]` tests need a CUDA device; run them with
//! `cargo test -p tenet --features cuda,cpu-faer -- --ignored`.

#![cfg(feature = "cuda")]

use tenet::core::Placement;
use tenet::prelude::*;

/// Elementwise comparison with a tight relative tolerance: the device GEMM
/// replays the same per-sector matrices in the same order as the host, but
/// cuTENSOR may fuse multiply-adds differently.
fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    let scale = expected.iter().fold(1.0f64, |acc, &x| acc.max(x.abs()));
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-12 * scale,
            "element {index}: device {a} vs host {e} (scale {scale})"
        );
    }
}

fn u1_space() -> Space {
    Space::u1([(-1, 2), (0, 3), (1, 2)])
}

fn su2_space() -> Space {
    Space::su2([(0, 2), (1, 1)])
}

#[test]
fn to_cuda_requires_cuda_runtime() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let t = Tensor::rand(&rt, [&v], [&v]).unwrap();
    let err = t.to_cuda().unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
}

#[test]
fn c64_to_cuda_is_explicit_error() {
    let rt = Runtime::builder().build().unwrap();
    let v = u1_space();
    let t = Tensor::rand_c64(&rt, [&v], [&v]).unwrap();
    let err = t.to_cuda().unwrap_err();
    assert!(matches!(err, Error::UnsupportedOnDevice(_)), "got {err:?}");
}

#[test]
#[ignore]
fn u1_contract_on_cuda_matches_host() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let a = Tensor::rand(&rt, [&v, &v], [&v]).unwrap();
    let b = Tensor::rand(&rt, [&v], [&v, &v]).unwrap();

    let host = a.compose(&b).unwrap();

    let a_dev = a.to_cuda().unwrap();
    let b_dev = b.to_cuda().unwrap();
    assert_eq!(a_dev.placement(), Placement::Cuda(0));
    let c_dev = a_dev.compose(&b_dev).unwrap();
    assert_eq!(c_dev.placement(), Placement::Cuda(0));
    let c = c_dev.to_host().unwrap();

    assert_eq!(c.placement(), Placement::Host);
    assert_eq!(c.rank(), host.rank());
    assert_close(c.data(), host.data());
}

#[test]
#[ignore]
fn su2_rank5_peps_contract_on_cuda_matches_host() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let phys = Space::su2([(0, 1), (1, 1)]);
    let bond = su2_space();

    // PEPS-shaped rank-5 tensors: (phys, left, up) <- (right, down),
    // contracted over the two bond legs (canonical core form).
    let a = Tensor::rand(&rt, [&phys, &bond, &bond], [&bond, &bond]).unwrap();
    let b = Tensor::rand(&rt, [&bond, &bond], [&phys, &bond, &bond]).unwrap();

    let host = a.compose(&b).unwrap();

    let c = a
        .to_cuda()
        .unwrap()
        .compose(&b.to_cuda().unwrap())
        .unwrap()
        .to_host()
        .unwrap();

    assert_eq!(c.rank(), host.rank());
    assert_close(c.data(), host.data());
}

#[test]
#[ignore]
fn contract_with_explicit_axes_on_cuda_matches_host() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let a = Tensor::rand(&rt, [&v, &v], [&v, &v]).unwrap();
    let b = Tensor::rand(&rt, [&v, &v], [&v]).unwrap();

    // Core form written out explicitly: a's domain against b's codomain.
    let host = a.contract(&b, &[2, 3], &[0, 1]).unwrap();
    let c = a
        .to_cuda()
        .unwrap()
        .contract(&b.to_cuda().unwrap(), &[2, 3], &[0, 1])
        .unwrap()
        .to_host()
        .unwrap();

    assert_close(c.data(), host.data());
}

#[test]
#[ignore]
fn non_canonical_contract_on_cuda_is_explicit_error() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let a = Tensor::rand(&rt, [&v], [&v, &v]).unwrap();
    let b = Tensor::rand(&rt, [&v, &v], [&v]).unwrap();

    // Reversed pairing of the contracted legs is not the canonical core
    // form; the host handles it via tree transforms, the device must
    // refuse instead of falling back.
    let a_dev = a.to_cuda().unwrap();
    let b_dev = b.to_cuda().unwrap();
    assert!(a.contract(&b, &[2, 1], &[0, 1]).is_ok());
    let err = a_dev.contract(&b_dev, &[2, 1], &[0, 1]).unwrap_err();
    assert!(matches!(err, Error::Operation(_)), "got {err:?}");
    assert!(
        err.to_string().contains("fully-direct"),
        "unexpected message: {err}"
    );
}

#[test]
#[ignore]
fn unsupported_ops_on_device_are_explicit_errors() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let t = Tensor::rand(&rt, [&v, &v], [&v])
        .unwrap()
        .to_cuda()
        .unwrap();

    for (name, err) in [
        ("permute", t.permute(&[1, 0], &[2]).unwrap_err()),
        ("transpose", t.transpose().map(|_| ()).unwrap_err()),
        (
            "braid",
            t.braid(&[1, 0], &[2], &[0, 1, 2]).map(|_| ()).unwrap_err(),
        ),
        ("adjoint", t.adjoint().unwrap_err()),
        ("svd_vals", t.svd_vals().map(|_| ()).unwrap_err()),
        ("lq", t.lq_compact().map(|_| ()).unwrap_err()),
    ] {
        assert!(
            matches!(err, Error::UnsupportedOnDevice(_)),
            "{name}: got {err:?}"
        );
    }
}

#[test]
#[ignore]
fn mixed_placement_contract_is_placement_error() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let a = Tensor::rand(&rt, [&v], [&v]).unwrap();
    let b = Tensor::rand(&rt, [&v], [&v]).unwrap();
    let b_dev = b.to_cuda().unwrap();

    let err = a.compose(&b_dev).unwrap_err();
    assert!(matches!(err, Error::PlacementMismatch), "got {err:?}");
    let err = b_dev.compose(&a).unwrap_err();
    assert!(matches!(err, Error::PlacementMismatch), "got {err:?}");
}

#[test]
#[ignore]
fn cross_runtime_device_contract_is_runtime_error() {
    let rt1 = Runtime::builder().cuda(0).build().unwrap();
    let rt2 = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let a = Tensor::rand(&rt1, [&v], [&v]).unwrap().to_cuda().unwrap();
    let b = Tensor::rand(&rt2, [&v], [&v]).unwrap().to_cuda().unwrap();

    let err = a.compose(&b).unwrap_err();
    assert!(matches!(err, Error::RuntimeMismatch), "got {err:?}");
}

#[test]
#[ignore]
fn to_cuda_to_host_round_trip_is_identity() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let t = Tensor::rand(&rt, [&v, &v], [&v]).unwrap();
    let back = t.to_cuda().unwrap().to_host().unwrap();
    assert_eq!(back.data(), t.data());
}

// ---------------------------------------------------------------------------
// Phase 2, slice 1: device norm / inner / add / scale + decompositions.
// ---------------------------------------------------------------------------

fn assert_close_tol(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    let scale = expected.iter().fold(1.0f64, |acc, &x| acc.max(x.abs()));
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * scale,
            "element {index}: device {a} vs host {e} (scale {scale})"
        );
    }
}

fn assert_spectra_close(
    actual: &[tenet::matrixalgebra::SectorSpectrum],
    expected: &[tenet::matrixalgebra::SectorSpectrum],
    tol: f64,
) {
    assert_eq!(actual.len(), expected.len(), "sector count differs");
    for (a, e) in actual.iter().zip(expected) {
        assert_eq!(a.sector, e.sector, "sector order differs");
        assert_close_tol(&a.values, &e.values, tol);
    }
}

#[test]
#[ignore]
fn norm_inner_add_scale_on_cuda_match_host_u1() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let a = Tensor::rand(&rt, [&v, &v], [&v]).unwrap();
    let b = Tensor::rand(&rt, [&v, &v], [&v]).unwrap();
    let a_dev = a.to_cuda().unwrap();
    let b_dev = b.to_cuda().unwrap();

    let norm = a.norm().unwrap();
    let norm_dev = a_dev.norm().unwrap();
    assert!((norm_dev - norm).abs() <= 1e-12 * (1.0 + norm));

    let inner = a.inner(&b).unwrap();
    let inner_dev = a_dev.inner(&b_dev).unwrap();
    assert!((inner_dev.re - inner.re).abs() <= 1e-12 * (1.0 + inner.re.abs()));
    assert_eq!(inner_dev.im, 0.0);

    let sum = a.add(&b, 1.5, -0.5).unwrap();
    let sum_dev = a_dev.add(&b_dev, 1.5, -0.5).unwrap();
    assert_eq!(sum_dev.placement(), Placement::Cuda(0));
    assert_close(sum_dev.to_host().unwrap().data(), sum.data());

    let scaled = a.scale(-2.5).unwrap();
    let scaled_dev = a_dev.scale(-2.5).unwrap();
    assert_eq!(scaled_dev.placement(), Placement::Cuda(0));
    assert_close(scaled_dev.to_host().unwrap().data(), scaled.data());
}

#[test]
#[ignore]
fn norm_and_inner_on_cuda_apply_su2_quantum_dimension_weights() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let bond = su2_space();
    let phys = Space::su2([(0, 1), (1, 1)]);
    // The gold check (permute-invariance of the weighted norm) needs a
    // device permute, which slice 1 does not have; instead the host norm of
    // the *same* tensor is the reference — it already carries the per-sector
    // dim(c) weights that a raw unweighted device reduction would miss.
    let a = Tensor::rand(&rt, [&phys, &bond], [&bond]).unwrap();
    let b = Tensor::rand(&rt, [&phys, &bond], [&bond]).unwrap();
    let a_dev = a.to_cuda().unwrap();
    let b_dev = b.to_cuda().unwrap();

    let norm = a.norm().unwrap();
    let norm_dev = a_dev.norm().unwrap();
    assert!(
        (norm_dev - norm).abs() <= 1e-12 * (1.0 + norm),
        "device {norm_dev} vs host {norm}"
    );

    let inner = a.inner(&b).unwrap();
    let inner_dev = a_dev.inner(&b_dev).unwrap();
    assert!((inner_dev.re - inner.re).abs() <= 1e-12 * (1.0 + inner.re.abs()));

    // Consistency: <a, a> == norm(a)^2 on device.
    let self_inner = a_dev.inner(&a_dev).unwrap().re;
    assert!((self_inner - norm_dev * norm_dev).abs() <= 1e-12 * (1.0 + self_inner));
}

#[test]
#[ignore]
fn svd_compact_on_cuda_matches_host() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let bond = su2_space();
    for t in [
        Tensor::rand(&rt, [&v, &v], [&v]).unwrap(),
        Tensor::rand(&rt, [&bond, &bond], [&bond]).unwrap(),
    ] {
        let (hu, hs, hvh) = t.svd_compact().unwrap();
        let (du, ds, dvh) = t.to_cuda().unwrap().svd_compact().unwrap();
        assert_eq!(du.placement(), Placement::Cuda(0));

        // Singular values: compare through the diagonal S tensors (exact
        // layout parity with the host factor).
        assert_close_tol(ds.to_host().unwrap().data(), hs.data(), 1e-10);

        // Factors are unique only up to per-degenerate-subspace phases;
        // compare the reconstruction on host after download.
        let u = du.to_host().unwrap();
        let s = ds.to_host().unwrap();
        let vh = dvh.to_host().unwrap();
        let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert_close_tol(rebuilt.data(), t.data(), 1e-10);
        assert_eq!(rebuilt.rank(), t.rank());
        drop((hu, hvh));
    }
}

#[test]
#[ignore]
fn svd_trunc_on_cuda_matches_host_spectrum_and_error() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let truncation = Truncation::Rank(3);
    let t = Tensor::rand(&rt, [&v, &v], [&v]).unwrap();

    let host = t.svd_trunc(&truncation).unwrap();
    let dev = t.to_cuda().unwrap().svd_trunc(&truncation).unwrap();

    assert_spectra_close(&dev.singular_values, &host.singular_values, 1e-10);
    assert!(
        (dev.error - host.error).abs() <= 1e-10 * (1.0 + host.error),
        "device error {} vs host {}",
        dev.error,
        host.error
    );

    // Truncated reconstruction matches the host truncated reconstruction.
    let host_rebuilt = host.u.compose(&host.s).unwrap().compose(&host.vh).unwrap();
    let dev_rebuilt = dev
        .u
        .to_host()
        .unwrap()
        .compose(&dev.s.to_host().unwrap())
        .unwrap()
        .compose(&dev.vh.to_host().unwrap())
        .unwrap();
    assert_close_tol(dev_rebuilt.data(), host_rebuilt.data(), 1e-10);
}

#[test]
#[ignore]
fn qr_compact_on_cuda_matches_host_factors() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let bond = su2_space();
    for t in [
        Tensor::rand(&rt, [&v, &v], [&v]).unwrap(),
        Tensor::rand(&rt, [&bond, &bond], [&bond]).unwrap(),
    ] {
        let (hq, hr) = t.qr_compact().unwrap();
        let (dq, dr) = t.to_cuda().unwrap().qr_compact().unwrap();
        let q = dq.to_host().unwrap();
        let r = dr.to_host().unwrap();

        // The positive-diagonal gauge makes the compact QR unique for
        // generic full-rank blocks: factors must match the host elementwise.
        assert_close_tol(q.data(), hq.data(), 1e-10);
        assert_close_tol(r.data(), hr.data(), 1e-10);

        // And reconstruct the input.
        let rebuilt = q.compose(&r).unwrap();
        assert_close_tol(rebuilt.data(), t.data(), 1e-10);
    }
}

#[test]
#[ignore]
fn eigh_on_cuda_matches_host() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let a = Tensor::rand(&rt, [&v], [&v]).unwrap();
    // Hermitian endomorphism: t = a + a^dagger.
    let t = a.add(&a.adjoint().unwrap(), 1.0, 1.0).unwrap();

    let (hd, hv) = t.eigh_full().unwrap();
    let (dd, dv) = t.to_cuda().unwrap().eigh_full().unwrap();

    // Eigenvalues in the host order (descending by magnitude per sector).
    assert_close_tol(dd.to_host().unwrap().data(), hd.data(), 1e-10);

    // Eigenvectors up to phase: compare v * d * v^dagger reconstructions.
    let v_host = dv.to_host().unwrap();
    let d_host = dd.to_host().unwrap();
    let rebuilt = v_host
        .compose(&d_host)
        .unwrap()
        .compose(&v_host.adjoint().unwrap())
        .unwrap();
    assert_close_tol(rebuilt.data(), t.data(), 1e-10);
    drop((hd, hv));
}

#[test]
#[ignore]
fn eigh_trunc_on_cuda_matches_host_spectrum_and_error() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_space();
    let a = Tensor::rand(&rt, [&v], [&v]).unwrap();
    let t = a.add(&a.adjoint().unwrap(), 1.0, 1.0).unwrap();
    let truncation = Truncation::Rank(4);

    let host = t.eigh_trunc(&truncation).unwrap();
    let dev = t.to_cuda().unwrap().eigh_trunc(&truncation).unwrap();

    assert_spectra_close(&dev.eigenvalues, &host.eigenvalues, 1e-10);
    assert!((dev.error - host.error).abs() <= 1e-10 * (1.0 + host.error));

    // Eigenvector phases are arbitrary (faer vs cuSOLVER column signs), so
    // compare the phase-free truncated reconstruction v * d * v^dagger.
    let host_rebuilt = host
        .v
        .compose(&host.d)
        .unwrap()
        .compose(&host.v.adjoint().unwrap())
        .unwrap();
    let dev_v = dev.v.to_host().unwrap();
    let dev_rebuilt = dev_v
        .compose(&dev.d.to_host().unwrap())
        .unwrap()
        .compose(&dev_v.adjoint().unwrap())
        .unwrap();
    assert_close_tol(dev_rebuilt.data(), host_rebuilt.data(), 1e-10);
}

#[test]
#[ignore]
fn device_pipeline_contract_svd_trunc_matches_host_pipeline() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    let phys = Space::su2([(0, 1), (1, 1)]);
    let bond = su2_space();
    let truncation = Truncation::Rank(4);

    let a = Tensor::rand(&rt, [&phys, &bond], [&bond]).unwrap();
    let b = Tensor::rand(&rt, [&bond], [&phys, &bond]).unwrap();

    // Host pipeline.
    let host_c = a.compose(&b).unwrap();
    let host = host_c.svd_trunc(&truncation).unwrap();

    // Device pipeline: upload -> contract -> svd_trunc, all on device.
    let dev_c = a.to_cuda().unwrap().compose(&b.to_cuda().unwrap()).unwrap();
    assert_eq!(dev_c.placement(), Placement::Cuda(0));
    let dev = dev_c.svd_trunc(&truncation).unwrap();
    assert_eq!(dev.u.placement(), Placement::Cuda(0));

    assert_spectra_close(&dev.singular_values, &host.singular_values, 1e-10);
    assert!((dev.error - host.error).abs() <= 1e-10 * (1.0 + host.error));

    // Norm consistency without leaving the device until the final check.
    let host_norm = host_c.norm().unwrap();
    let dev_norm = dev_c.norm().unwrap();
    assert!((dev_norm - host_norm).abs() <= 1e-12 * (1.0 + host_norm));

    let host_rebuilt = host.u.compose(&host.s).unwrap().compose(&host.vh).unwrap();
    let dev_rebuilt = dev
        .u
        .to_host()
        .unwrap()
        .compose(&dev.s.to_host().unwrap())
        .unwrap()
        .compose(&dev.vh.to_host().unwrap())
        .unwrap();
    assert_close_tol(dev_rebuilt.data(), host_rebuilt.data(), 1e-10);
}
