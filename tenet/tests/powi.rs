use std::sync::Arc;

use num_complex::Complex64;
use tenet::core::{Z2FusionRule, Z2Irrep};
use tenet::prelude::{Dtype, Error, Runtime, Scalar, Space, Tensor};
use tenet::typed::{GradedSpace, SectorSpectrum, TensorMap};

fn typed_bond(provider: &Arc<Z2FusionRule>, degeneracy: usize) -> GradedSpace<Z2FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        [(Z2Irrep::EVEN, degeneracy), (Z2Irrep::ODD, degeneracy)],
        false,
    )
    .unwrap()
}

#[test]
fn powi_typed_and_erased_f64_agree_and_stay_compact() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::z2([(0, 1), (1, 1)]);
    let erased = Tensor::diagonal(
        &runtime,
        Dtype::F64,
        &space,
        [vec![Scalar::F64(2.0)], vec![Scalar::F64(3.0)]],
    )
    .unwrap();
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
        let erased_power = erased.powi(exponent).unwrap();
        let typed_power = typed.powi(exponent).unwrap();
        assert_eq!(typed_power.data(), erased_power.data());
        assert!(erased_power.diagonal_spectrum().unwrap().is_some());
        assert!(typed_power.diagonal_spectrum().unwrap().is_some());
    }
    assert_eq!(
        erased.powi(0).unwrap().diagonal_spectrum().unwrap(),
        Some(vec![vec![Scalar::F64(1.0)], vec![Scalar::F64(1.0)]])
    );
    assert_eq!(
        erased.powi(5).unwrap().diagonal_spectrum().unwrap(),
        Some(vec![vec![Scalar::F64(32.0)], vec![Scalar::F64(243.0)]])
    );
    let negative = erased
        .powi(-3)
        .unwrap()
        .diagonal_spectrum()
        .unwrap()
        .unwrap();
    for (actual, expected) in negative.iter().flatten().zip([0.125, 1.0 / 27.0]) {
        assert!((actual.try_f64().unwrap() - expected).abs() < 1e-15);
    }
}

#[test]
fn powi_typed_and_erased_c64_agree_including_i32_min_identity() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::z2([(0, 1), (1, 1)]);
    let one = Complex64::new(1.0, 0.0);
    let values = [Complex64::new(2.0, 1.0), Complex64::new(3.0, -0.5)];
    let erased = Tensor::diagonal(
        &runtime,
        Dtype::C64,
        &space,
        [vec![Scalar::C64(values[0])], vec![Scalar::C64(values[1])]],
    )
    .unwrap();
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
        assert_eq!(
            typed.powi(exponent).unwrap().data(),
            erased.powi(exponent).unwrap().data_c64()
        );
    }

    let erased_roots = Tensor::diagonal(
        &runtime,
        Dtype::C64,
        &space,
        [
            vec![Scalar::C64(Complex64::new(-1.0, 0.0))],
            vec![Scalar::C64(Complex64::new(0.0, 1.0))],
        ],
    )
    .unwrap();
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
    let erased_min = erased_roots.powi(i32::MIN).unwrap();
    let typed_min = typed_roots.powi(i32::MIN).unwrap();
    assert_eq!(
        erased_min.diagonal_spectrum().unwrap(),
        Some(vec![vec![Scalar::C64(one)], vec![Scalar::C64(one)]])
    );
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
    assert_eq!(typed_min.data(), erased_min.data_c64());
}

#[test]
fn powi_rejects_non_endomorphisms_and_singular_negative_powers() {
    let runtime = Runtime::builder().build().unwrap();
    let narrow = Space::z2([(0, 1), (1, 1)]);
    let wide = Space::z2([(0, 2), (1, 2)]);
    let erased = Tensor::from_block_fn(&runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();

    let provider = Arc::new(Z2FusionRule);
    let typed_narrow = typed_bond(&provider, 1);
    let typed_wide = typed_bond(&provider, 2);
    let typed =
        TensorMap::from_block_fn(&runtime, [&typed_wide], [&typed_narrow], |_, _| 1.0).unwrap();
    let expected =
        Error::InvalidArgument("powi() requires an endomorphism (domain == codomain)".into());
    for exponent in [0, 1, -1] {
        assert_eq!(erased.powi(exponent).unwrap_err(), expected);
        assert_eq!(typed.powi(exponent).unwrap_err(), expected);
    }

    let erased_singular = Tensor::diagonal(
        &runtime,
        Dtype::F64,
        &narrow,
        [vec![Scalar::F64(0.0)], vec![Scalar::F64(1.0)]],
    )
    .unwrap();
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
        erased_singular.powi(-1),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        typed_singular.powi(-1),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn dense_powi_matches_hand_computed_matrix_powers() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::z2([(0, 2)]);
    let provider = Arc::new(Z2FusionRule);
    let bond = GradedSpace::try_new(provider, [(Z2Irrep::EVEN, 2)], false).unwrap();
    let matrix = [[2.0, 1.0], [0.0, 3.0]];
    let erased =
        Tensor::from_block_fn(&runtime, [&space], [&space], |_, i| matrix[i[0]][i[1]]).unwrap();
    let typed =
        TensorMap::from_block_fn(&runtime, [&bond], [&bond], |_, i| matrix[i[0]][i[1]]).unwrap();
    let square = [[4.0, 5.0], [0.0, 9.0]];
    let expected_square =
        Tensor::from_block_fn(&runtime, [&space], [&space], |_, i| square[i[0]][i[1]]).unwrap();
    let inverse = [[0.5, -1.0 / 6.0], [0.0, 1.0 / 3.0]];
    let expected_inverse =
        Tensor::from_block_fn(&runtime, [&space], [&space], |_, i| inverse[i[0]][i[1]]).unwrap();
    let identity = Tensor::id(&runtime, Dtype::F64, [&space]).unwrap();

    let erased_zero = erased.powi(0).unwrap();
    let typed_zero = typed.powi(0).unwrap();
    assert_ne!(erased_zero.data(), erased.data());
    assert_ne!(typed_zero.data(), typed.data());
    assert_eq!(erased_zero.data(), identity.data());
    assert_eq!(typed_zero.data(), identity.data());
    assert_eq!(erased.powi(2).unwrap().data(), expected_square.data());
    assert_eq!(typed.powi(2).unwrap().data(), expected_square.data());

    for actual in [
        erased.powi(-1).unwrap().data(),
        typed.powi(-1).unwrap().data(),
    ] {
        for (actual, expected) in actual.iter().zip(expected_inverse.data()) {
            assert!((actual - expected).abs() < 1e-14);
        }
    }
}
