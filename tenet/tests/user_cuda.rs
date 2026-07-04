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
        ("adjoint", t.adjoint().unwrap_err()),
        ("norm", t.norm().unwrap_err()),
        ("svd", t.svd_compact().map(|_| ()).unwrap_err()),
        ("scale", t.scale(2.0).map(|_| ()).unwrap_err()),
        ("add", t.add(&t, 1.0, 1.0).map(|_| ()).unwrap_err()),
        ("inner", t.inner(&t).map(|_| ()).unwrap_err()),
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
