//! Selectable CPU linear-algebra backend (#64): `RuntimeBuilder::linalg_backend`
//! chooses the dense SVD/QR/eigh provider. Backend choice is performance-only —
//! results stay identical across providers to numerical precision.

use tenet::prelude::*;

fn u1_space() -> Space {
    Space::u1([(-1, 2), (0, 3), (1, 2)])
}

fn spectra(rt: &Runtime) -> Vec<f64> {
    let v = u1_space();
    let t = Tensor::rand_with_seed(rt, Dtype::F64, [&v, &v], [&v, &v], 101).unwrap();
    let mut all: Vec<f64> = t
        .svd_vals()
        .unwrap()
        .into_iter()
        .flat_map(|s| s.values)
        .collect();
    all.sort_by(|a, b| b.partial_cmp(a).unwrap());
    all
}

#[test]
fn faer_backend_builds_and_computes() {
    // The default provider, requested explicitly, must build and produce a
    // sane spectrum (descending, all non-negative singular values).
    let rt = Runtime::builder()
        .linalg_backend(LinalgBackend::Faer)
        .build()
        .unwrap();
    let s = spectra(&rt);
    assert!(!s.is_empty());
    assert!(s.iter().all(|&x| x >= 0.0));
    for pair in s.windows(2) {
        assert!(pair[0] >= pair[1] - 1e-12);
    }
}

#[test]
fn linalg_backend_composes_with_dense_threads() {
    // Selecting a provider together with an explicit thread count routes through
    // the threads-and-kind constructor.
    let rt = Runtime::builder()
        .linalg_backend(LinalgBackend::Faer)
        .dense_threads(1)
        .build()
        .unwrap();
    assert!(!spectra(&rt).is_empty());
}

#[test]
fn faer_and_blas_agree_when_blas_is_available() {
    // BLAS is only linked under a `blas-*` cargo feature; when absent, building
    // the Blas provider fails cleanly and there is nothing to compare (CI runs
    // the default faer-only feature set, so this path exercises the skip).
    let faer = Runtime::builder()
        .linalg_backend(LinalgBackend::Faer)
        .build()
        .unwrap();
    let blas = match Runtime::builder()
        .linalg_backend(LinalgBackend::Blas)
        .build()
    {
        Ok(rt) => rt,
        Err(_) => return,
    };
    let faer_s = spectra(&faer);
    let blas_s = spectra(&blas);
    assert_eq!(faer_s.len(), blas_s.len());
    for (a, b) in faer_s.iter().zip(&blas_s) {
        assert!((a - b).abs() <= 1e-10, "faer {a} vs blas {b}");
    }
}
