use std::sync::Arc;

use num_complex::Complex64;
use tenet::core::{Z2FusionRule, Z2Irrep};
use tenet::prelude::{Error, Runtime};
use tenet::typed::{GradedSpace, SectorSpectrum, TensorMap};

fn typed_bond(provider: &Arc<Z2FusionRule>, degeneracy: usize) -> GradedSpace<Z2FusionRule> {
    GradedSpace::try_new_with_shared_provider(
        Arc::clone(provider),
        [(Z2Irrep::EVEN, degeneracy), (Z2Irrep::ODD, degeneracy)],
    )
    .unwrap()
}

#[test]
fn powi_f64_matches_exact_spectra_and_stays_compact() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(Z2FusionRule);
    let bond = typed_bond(&provider, 1);
    let typed = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Z2Irrep::EVEN,
                values: vec![2.0],
            },
            SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: vec![3.0],
            },
        ],
    )
    .unwrap();

    for exponent in [0, 1, 5, -3] {
        let typed_power = typed.powi(exponent).unwrap();
        assert!(typed_power.diagonal_spectrum().unwrap().is_some());
    }
    assert_eq!(
        typed.powi(0).unwrap().diagonal_spectrum().unwrap(),
        Some(vec![
            SectorSpectrum {
                sector: Z2Irrep::EVEN,
                values: vec![1.0],
            },
            SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: vec![1.0],
            },
        ])
    );
    assert_eq!(
        typed.powi(5).unwrap().diagonal_spectrum().unwrap(),
        Some(vec![
            SectorSpectrum {
                sector: Z2Irrep::EVEN,
                values: vec![32.0],
            },
            SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: vec![243.0],
            },
        ])
    );
    let negative = typed
        .powi(-3)
        .unwrap()
        .diagonal_spectrum()
        .unwrap()
        .unwrap();
    for (actual, expected) in negative
        .iter()
        .flat_map(|spectrum| &spectrum.values)
        .zip([0.125, 1.0 / 27.0])
    {
        assert!((actual - expected).abs() < 1e-15);
    }
}

#[test]
fn powi_c64_handles_i32_min_identity() {
    let runtime = Runtime::builder().build().unwrap();
    let one = Complex64::new(1.0, 0.0);
    let values = [Complex64::new(2.0, 1.0), Complex64::new(3.0, -0.5)];
    let provider = Arc::new(Z2FusionRule);
    let bond = typed_bond(&provider, 1);
    let typed = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Z2Irrep::EVEN,
                values: vec![values[0]],
            },
            SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: vec![values[1]],
            },
        ],
    )
    .unwrap();

    for exponent in [0, 1, 4, -2] {
        let power = typed
            .powi(exponent)
            .unwrap()
            .diagonal_spectrum()
            .unwrap()
            .unwrap();
        for (actual, expected) in power
            .iter()
            .flat_map(|spectrum| &spectrum.values)
            .zip(values.map(|value| value.powi(exponent)))
        {
            assert_eq!(*actual, expected);
        }
    }

    let typed_roots = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Z2Irrep::EVEN,
                values: vec![Complex64::new(-1.0, 0.0)],
            },
            SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: vec![Complex64::new(0.0, 1.0)],
            },
        ],
    )
    .unwrap();
    let typed_min = typed_roots.powi(i32::MIN).unwrap();
    assert_eq!(
        typed_min.diagonal_spectrum().unwrap(),
        Some(vec![
            SectorSpectrum {
                sector: Z2Irrep::EVEN,
                values: vec![one],
            },
            SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: vec![one],
            },
        ])
    );
}

#[test]
fn powi_rejects_non_endomorphisms_and_singular_negative_powers() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(Z2FusionRule);
    let typed_narrow = typed_bond(&provider, 1);
    let typed_wide = typed_bond(&provider, 2);
    let typed =
        TensorMap::from_block_fn(&runtime, [&typed_wide], [&typed_narrow], |_, _| 1.0).unwrap();
    let expected =
        Error::InvalidArgument("powi() requires an endomorphism (domain == codomain)".into());
    for exponent in [0, 1, -1] {
        assert_eq!(typed.powi(exponent).unwrap_err(), expected);
    }

    let typed_singular = TensorMap::diagonal(
        &runtime,
        &typed_narrow,
        [
            SectorSpectrum {
                sector: Z2Irrep::EVEN,
                values: vec![0.0],
            },
            SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: vec![1.0],
            },
        ],
    )
    .unwrap();
    assert!(matches!(
        typed_singular.powi(-1),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn dense_powi_matches_hand_computed_matrix_powers() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(Z2FusionRule);
    let bond = GradedSpace::try_new_with_shared_provider(provider, [(Z2Irrep::EVEN, 2)]).unwrap();
    let matrix = [[2.0, 1.0], [0.0, 3.0]];
    let typed =
        TensorMap::from_block_fn(&runtime, [&bond], [&bond], |_, i| matrix[i[0]][i[1]]).unwrap();
    let square = [[4.0, 5.0], [0.0, 9.0]];
    let expected_square =
        TensorMap::from_block_fn(&runtime, [&bond], [&bond], |_, i| square[i[0]][i[1]]).unwrap();
    let inverse = [[0.5, -1.0 / 6.0], [0.0, 1.0 / 3.0]];
    let expected_inverse =
        TensorMap::from_block_fn(&runtime, [&bond], [&bond], |_, i| inverse[i[0]][i[1]]).unwrap();
    let identity: TensorMap<Z2FusionRule, f64> = TensorMap::id(&runtime, [&bond]).unwrap();

    let typed_zero = typed.powi(0).unwrap();
    assert_ne!(typed_zero.data(), typed.data());
    assert_eq!(typed_zero.data(), identity.data());
    assert_eq!(typed.powi(2).unwrap().data(), expected_square.data());

    for (actual, expected) in typed
        .powi(-1)
        .unwrap()
        .data()
        .iter()
        .zip(expected_inverse.data())
    {
        assert!((actual - expected).abs() < 1e-14);
    }
}
