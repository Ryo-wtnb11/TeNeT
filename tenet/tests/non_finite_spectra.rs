//! A NaN reaching a truncation or pseudo-inverse cutoff is a typed error, never
//! a silent finite decision (#1295).
//!
//! Neither reference defines this: MatrixAlgebraKit `findtruncated` /
//! `_truncerr_impl` compare against a NaN-poisoned norm and keep nothing, and
//! TensorKit `pinv(::DiagonalTensorMap)` maps a NaN entry to zero through
//! `pinv(x::Number)`. TeNeT rejects the spectrum instead. Each fixture puts the
//! NaN in the *second* sector so a fold that ignores NaN would still see a
//! finite maximum from the first.

use std::sync::Arc;

use num_complex::Complex64;
use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::operations::OperationError;
use tenet::prelude::{Error, Runtime, TensorMap, Truncation};
use tenet::typed::{GradedSpace, SectorSpectrum};

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

fn policies() -> Vec<Truncation> {
    vec![
        Truncation::Full,
        Truncation::rank(2),
        Truncation::relative_cutoff(0.1).unwrap(),
        Truncation::relative_inf_cutoff(0.1).unwrap(),
        Truncation::relative_error(0.1).unwrap(),
        Truncation::rank(3).and(Truncation::relative_inf_cutoff(0.1).unwrap()),
    ]
}

macro_rules! assert_nan_spectrum_is_rejected {
    ($leg:expr, $sectors:expr, $scalar:expr) => {{
        let leg = &$leg;
        let [first, second] = $sectors;
        let spectra = vec![
            SectorSpectrum {
                sector: first,
                values: vec![$scalar(4.0), $scalar(1.0)],
            },
            SectorSpectrum {
                sector: second,
                values: vec![$scalar(f64::NAN), $scalar(0.5)],
            },
        ];
        for policy in policies() {
            assert!(
                leg.find_truncated(&spectra, &policy).is_err(),
                "find_truncated accepted a NaN spectrum under {policy:?}"
            );
        }
        let diagonal = TensorMap::diagonal(&runtime(), leg, spectra).unwrap();
        for rcond in [0.0, 0.1] {
            assert!(
                matches!(diagonal.pinv(rcond), Err(Error::InvalidArgument(_))),
                "compact pinv hid a NaN entry at rcond {rcond}"
            );
        }
        // The dense routes: a NaN payload never yields a finite factor either.
        // The truncated compositions fail at the factorization or at the
        // decision; neither publishes a factor.
        for policy in policies() {
            let svd = diagonal
                .svd_compact()
                .and_then(|tenet::typed::Svd { s, .. }| {
                    s.domain()[0].find_truncated(&s.diagview()?, &policy)
                });
            assert!(svd.is_err(), "svd composition {policy:?}");
            let eigh = diagonal
                .eigh_full()
                .and_then(|tenet::typed::Eigh { d, .. }| {
                    d.domain()[0].find_truncated(&d.diagview()?, &policy)
                });
            assert!(eigh.is_err(), "eigh composition {policy:?}");
        }
    }};
}

fn real(value: f64) -> f64 {
    value
}

fn complex(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}

fn u1() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    )
    .unwrap()
}

fn su2() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap()
}

#[test]
fn u1_real_nan_spectrum_is_rejected() {
    assert_nan_spectrum_is_rejected!(u1(), [U1Irrep::new(0), U1Irrep::new(1)], real);
}

#[test]
fn u1_complex_nan_spectrum_is_rejected() {
    assert_nan_spectrum_is_rejected!(u1(), [U1Irrep::new(0), U1Irrep::new(1)], complex);
}

#[test]
fn su2_real_nan_spectrum_is_rejected() {
    assert_nan_spectrum_is_rejected!(
        su2(),
        [SU2Irrep::from_twice_spin(0), SU2Irrep::from_twice_spin(1)],
        real
    );
}

#[test]
fn su2_complex_nan_spectrum_is_rejected() {
    assert_nan_spectrum_is_rejected!(
        su2(),
        [SU2Irrep::from_twice_spin(0), SU2Irrep::from_twice_spin(1)],
        complex
    );
}

#[test]
fn compact_pinv_of_a_finite_diagonal_is_unchanged() {
    // Hand-computed: sigma_max = 4, rcond 0.2 -> cutoff 0.8; 0.5 is cut.
    let diagonal: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime(),
        &u1(),
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![4.0, 1.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0, 0.5],
            },
        ],
    )
    .unwrap();
    let image = diagonal
        .pinv(0.2)
        .unwrap()
        .diagonal_spectrum()
        .unwrap()
        .unwrap();
    assert_eq!(image[0].values, vec![0.25, 1.0]);
    assert_eq!(image[1].values, vec![0.5, 0.0]);
}

/// A NaN tensor through the dense multiplicity-free pinv routes (owned and
/// lazy adjoint). Today's CPU backend refuses the NaN SVD before any cutoff
/// runs, so that typed backend error is what is pinned here; a backend that
/// returned NaN singular values instead would reach `pinv_cutoff` and answer
/// `Error::InvalidArgument`. Either way the result is never `Ok`.
#[test]
fn dense_pinv_of_a_nan_tensor_is_a_typed_backend_error() {
    let runtime = runtime();
    let leg = u1();
    let real: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, index: &[usize]| {
            if *trees.coupled() == U1Irrep::new(1) && index == [0, 0] {
                f64::NAN
            } else if index[0] == index[1] {
                2.0
            } else {
                0.5
            }
        })
        .unwrap();
    let complex = real.to_c64();
    for (case, result) in [
        ("f64 owned", real.pinv(0.1).map(|_| ())),
        ("f64 adjoint", real.adjoint().unwrap().pinv(0.1).map(|_| ())),
        ("c64 owned", complex.pinv(0.1).map(|_| ())),
        (
            "c64 adjoint",
            complex.adjoint().unwrap().pinv(0.1).map(|_| ()),
        ),
    ] {
        match result {
            Err(Error::Operation(error)) => {
                assert!(
                    matches!(*error, OperationError::Dense(_)),
                    "{case}: {error:?}"
                )
            }
            other => panic!("{case}: expected the backend SVD rejection, got {other:?}"),
        }
    }
}
