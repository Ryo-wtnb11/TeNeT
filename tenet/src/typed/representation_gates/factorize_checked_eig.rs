use super::*;

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eigh_vals_reads_stored_real_spectrum() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eigh_vals_spy(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    Complex64::new(2.0, 0.0),
                    Complex64::new(-2.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(-4.0, 0.0), Complex64::new(0.0, 0.0)],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let got = input.eigh_vals(&[0], &[1]).unwrap();
    assert_eq!(
        got,
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![-4.0, 0.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![-2.0, 2.0, 1.0],
            },
        ]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.total(), 0);

    let swapped = input.eigh_vals(&[1], &[0]).unwrap();
    assert_eq!(calls.total(), 2);
    let swapped_dense = input.permute(&[1], &[0]).unwrap().materialize().unwrap();
    assert_eq!(swapped, swapped_dense.eigh_vals(&[0], &[1]).unwrap());
    let dense = input.materialize().unwrap();
    assert_eq!(got, dense.eigh_vals(&[0], &[1]).unwrap());

    let real: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![2.0, -2.0, 1.0],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![-4.0, 0.0],
            },
        ],
    )
    .unwrap();
    let dense_real = real.materialize().unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        real.eigh_vals(&[0], &[1]).unwrap(),
        dense_real.eigh_vals(&[0], &[1]).unwrap()
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let narrow: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    num_complex::Complex32::new(2.0, 0.0),
                    num_complex::Complex32::new(-2.0, 0.0),
                    num_complex::Complex32::new(1.0, 0.0),
                ],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![
                    num_complex::Complex32::new(-4.0, 0.0),
                    num_complex::Complex32::new(0.0, 0.0),
                ],
            },
        ],
    )
    .unwrap();
    let dense_narrow = narrow.materialize().unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        narrow.eigh_vals(&[0], &[1]).unwrap(),
        dense_narrow.eigh_vals(&[0], &[1]).unwrap()
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    calls.reset();

    let dual_leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual_sector = dual_leg.sectors().unwrap().remove(0);
    let dual: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &dual_leg,
        [SectorSpectrum {
            sector: dual_sector.clone(),
            values: vec![-2.0, 2.0],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        dual.eigh_vals(&[0], &[1]).unwrap(),
        vec![SectorSpectrum {
            sector: dual_sector,
            values: vec![-2.0, 2.0],
        }]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.total(), 0);
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eigh_vals_rejects_inconsistent_spectrum_admission() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![1.0, -1.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![2.0],
            },
        ],
    )
    .unwrap();
    let spectrum = input.spectrum().unwrap();
    // A spectrum that does not cover the bond is misuse: a typed error.
    let admit = |entries: &[tenet_matrixalgebra::SectorSpectrum<f64>]| {
        tenet_matrixalgebra::seam::eigh_vals_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(
            &mut tenet_dense::DefaultDenseExecutor::new(),
            tenet_matrixalgebra::seam::FactorSource::Diagonal {
                space: input.logical_space(),
                spectrum: entries,
            },
        )
    };
    assert!(admit(spectrum).is_ok());
    let mut reversed = spectrum.to_vec();
    reversed.reverse();
    assert!(admit(&reversed).is_ok());
    assert!(admit(&spectrum[..1]).is_err());
    let mut duplicate = spectrum.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(admit(&duplicate).is_err());
    let mut wrong_size = spectrum.to_vec();
    wrong_size[0].values.push(0.0);
    assert!(admit(&wrong_size).is_err());
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eigh_vals_applies_the_dense_hermiticity_check_directly() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eigh_vals_spy(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 1)]).unwrap();
    let near: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![0, 0],
            values: vec![Complex64::new(1.0, 1e-15)],
        }],
    )
    .unwrap();
    // The dense route's relative Hermiticity check, applied to the diagonal
    // directly: within tolerance the eigenvalue is the real part.
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(near.eigh_vals(&[0], &[1]).unwrap()[0].values, [1.0]);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.total(), 0);

    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![0, 0],
            values: vec![f64::NAN],
        }],
    )
    .unwrap();
    // NaN is refused first by the shared finite-input stage (#1986).
    DIAGONAL_MATERIALIZATIONS.set(0);
    let compact_error = nonfinite.eigh_vals(&[0], &[1]).unwrap_err();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert!(compact_error
        .to_string()
        .contains("eigh input components must be finite"));
}

#[cfg(feature = "racah-generated")]
fn assert_same_complex_multisets<S: PartialEq + std::fmt::Debug>(
    actual: &[SectorSpectrum<S, Complex64>],
    expected: &[SectorSpectrum<S, Complex64>],
    tol: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.sector, expected.sector);
        assert!(actual
            .values
            .windows(2)
            .all(|pair| pair[0].norm() >= pair[1].norm()));
        let sorted = |values: &[Complex64]| {
            let mut values = values.to_vec();
            values.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
            values
        };
        let (left, right) = (sorted(&actual.values), sorted(&expected.values));
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(&right) {
            assert!((left - right).norm() <= tol, "{left} != {right}");
        }
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eig_vals_reads_stored_complex_spectrum() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eig_vals_spy(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    Complex64::new(1.0, 1.0),
                    Complex64::new(-3.0, 4.0),
                    Complex64::new(0.0, 0.5),
                ],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(-0.25, 0.0), Complex64::new(2.0, -2.0)],
            },
        ],
    )
    .unwrap();
    let saved = input.diagview().unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let got = input.eig_vals(&[0], &[1]).unwrap();
    assert_eq!(
        got,
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(2.0, -2.0), Complex64::new(-0.25, 0.0)],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    Complex64::new(-3.0, 4.0),
                    Complex64::new(1.0, 1.0),
                    Complex64::new(0.0, 0.5),
                ],
            },
        ]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.total(), 0);
    assert_eq!(input.diagview().unwrap(), saved);

    let dense = input.materialize().unwrap();
    assert_same_complex_multisets(&got, &dense.eig_vals(&[0], &[1]).unwrap(), 1e-12);
    calls.reset();

    let swapped = input.eig_vals(&[1], &[0]).unwrap();
    assert_eq!(calls.total(), 2);
    let swapped_dense = input.permute(&[1], &[0]).unwrap().materialize().unwrap();
    assert_same_complex_multisets(
        &swapped,
        &swapped_dense.eig_vals(&[0], &[1]).unwrap(),
        1e-12,
    );

    calls.reset();
    let lazy = dense.adjoint().unwrap();
    assert!(matches!(lazy.repr, TypedTensorRepr::Adjoint(_)));
    assert!(matches!(
        lazy.eig_vals(&[0], &[1]),
        Err(GenericTensorError::Facade(Error::InvalidArgument(message)))
            if message.contains("lazy adjoints")
    ));
    assert_eq!(calls.total(), 0);
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eig_vals_widens_after_payload_rounding() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eig_vals_spy(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 2)]).unwrap();
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let r32 = |value: f32| f64::from(value);

    let real: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![0.1, -0.7],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![1.3, 0.2],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let got = real.eig_vals(&[0], &[1]).unwrap();
    assert_eq!(
        got,
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![c(r32(-0.7), 0.0), c(r32(0.1), 0.0)],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![c(r32(1.3), 0.0), c(r32(0.2), 0.0)],
            },
        ]
    );
    assert_ne!(got[0].values[1].re, 0.1);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    let dense = real.materialize().unwrap().eig_vals(&[0], &[1]).unwrap();
    assert_same_complex_multisets(&got, &dense, 1e-6);

    let real64: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![0.1, -0.7],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![1.3, 0.2],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let got = real64.eig_vals(&[0], &[1]).unwrap();
    assert_eq!(got[0].values, vec![c(-0.7, 0.0), c(0.1, 0.0)]);
    assert_eq!(got[1].values, vec![c(1.3, 0.0), c(0.2, 0.0)]);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    let dense = real64.materialize().unwrap().eig_vals(&[0], &[1]).unwrap();
    assert_same_complex_multisets(&got, &dense, 1e-12);

    let narrow: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![
                    num_complex::Complex32::new(0.1, -0.3),
                    num_complex::Complex32::new(-0.7, 0.0),
                ],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    num_complex::Complex32::new(0.0, 1.3),
                    num_complex::Complex32::new(0.2, 0.2),
                ],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let got = narrow.eig_vals(&[0], &[1]).unwrap();
    assert_eq!(
        got,
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![c(r32(-0.7), 0.0), c(r32(0.1), r32(-0.3))],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![c(0.0, r32(1.3)), c(r32(0.2), r32(0.2))],
            },
        ]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    let dense = narrow.materialize().unwrap().eig_vals(&[0], &[1]).unwrap();
    assert_same_complex_multisets(&got, &dense, 1e-6);
    assert_eq!(calls.total(), 6);
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eig_vals_keeps_ties_and_dual_bond() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eig_vals_spy(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 4)]).unwrap();
    let ties: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![1, 1],
            values: vec![c(3.0, 0.0), c(0.0, -3.0), c(-3.0, 0.0), c(1.0, 0.0)],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let got = ties.eig_vals(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.total(), 0);
    assert_same_complex_multisets(
        &got,
        &[SectorSpectrum {
            sector: vec![1, 1],
            values: vec![c(-3.0, 0.0), c(0.0, -3.0), c(3.0, 0.0), c(1.0, 0.0)],
        }],
        0.0,
    );
    assert_same_complex_multisets(
        &got,
        &ties.materialize().unwrap().eig_vals(&[0], &[1]).unwrap(),
        1e-12,
    );
    calls.reset();

    let dual_leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual_sector = dual_leg.sectors().unwrap().remove(0);
    let dual: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &dual_leg,
        [SectorSpectrum {
            sector: dual_sector.clone(),
            values: vec![-2.0, 5.0],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let got = dual.eig_vals(&[0], &[1]).unwrap();
    assert_eq!(
        got,
        vec![SectorSpectrum {
            sector: dual_sector,
            values: vec![c(5.0, 0.0), c(-2.0, 0.0)],
        }]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.total(), 0);
    assert_same_complex_multisets(
        &got,
        &dual.materialize().unwrap().eig_vals(&[0], &[1]).unwrap(),
        1e-12,
    );
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eig_vals_rejects_inconsistent_spectrum_admission() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(1.0, 2.0), Complex64::new(-1.0, 0.0)],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![Complex64::new(0.0, 2.0)],
            },
        ],
    )
    .unwrap();
    let spectrum = input.spectrum().unwrap();
    // A spectrum that does not cover the bond is misuse: a typed error.
    let admit = |entries: &[tenet_matrixalgebra::SectorSpectrum<Complex64>]| {
        tenet_matrixalgebra::seam::eig_vals_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(
            &mut tenet_dense::DefaultDenseExecutor::new(),
            tenet_matrixalgebra::seam::FactorSource::Diagonal {
                space: input.logical_space(),
                spectrum: entries,
            },
        )
    };
    assert!(admit(spectrum).is_ok());
    let mut reversed = spectrum.to_vec();
    reversed.reverse();
    assert!(admit(&reversed).is_ok());
    assert!(admit(&spectrum[..1]).is_err());
    let mut duplicate = spectrum.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(admit(&duplicate).is_err());
    let mut wrong_size = spectrum.to_vec();
    wrong_size[0].values.push(Complex64::new(0.0, 0.0));
    assert!(admit(&wrong_size).is_err());
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eig_vals_refuses_nonfinite_values() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eig_vals_spy(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 1), (vec![1, 0], 1)]).unwrap();
    for bad in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(0.0, f64::INFINITY),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: vec![0, 0],
                    values: vec![Complex64::new(1.0, 0.0)],
                },
                SectorSpectrum {
                    sector: vec![1, 0],
                    values: vec![bad],
                },
            ],
        )
        .unwrap();
        // A nonfinite value is refused by the shared finite-input stage, and
        // the overflowing `MAX + MAX i` by the dense route's eigenvalue check;
        // no dense solver either way.
        calls.reset();
        DIAGONAL_MATERIALIZATIONS.set(0);
        let error = input.eig_vals(&[0], &[1]).unwrap_err().to_string();
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        assert_eq!(calls.total(), 0);
        let expected = if bad.re.is_finite() && bad.im.is_finite() {
            "eigenvalues must be finite"
        } else {
            "eig input components must be finite"
        };
        assert!(error.contains(expected), "{error}");
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eigh_full_avoids_input_materialization_and_solver() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eigh_full_spy(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    macro_rules! check {
        ($dtype:ty, $from:expr) => {{
            let from: fn(f64) -> $dtype = $from;
            let input: TensorMap<_, $dtype> = TensorMap::diagonal(
                &runtime,
                &leg,
                [
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![from(2.0), from(-2.0), from(1.0)],
                    },
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![from(0.0), from(-4.0)],
                    },
                ],
            )
            .unwrap();
            DIAGONAL_MATERIALIZATIONS.set(0);
            calls.reset();
            let Eigh { d, v } = input.eigh_full(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert_eq!(calls.total(), 0);
            assert!(std::ptr::eq(d.provider(), provider.as_ref()));
            assert!(std::ptr::eq(v.provider(), provider.as_ref()));
            assert!(d.dense_data().is_err());
            assert_eq!(
                d.diagview().unwrap(),
                vec![
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![from(-4.0), from(0.0)],
                    },
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![from(-2.0), from(2.0), from(1.0)],
                    },
                ]
            );
            assert_eq!(v.codomain(), input.codomain());
            assert!(!v.domain()[0].is_dual());
            // Column j of each sector block is the unit vector of the stored
            // index that sorts into position j.
            let expected: TensorMap<_, $dtype> = TensorMap::from_subblock_fn(
                &runtime,
                v.codomain().iter(),
                v.domain().iter(),
                |trees, ij| {
                    let order: &[usize] = if trees.coupled() == &vec![0, 0] {
                        &[1, 0]
                    } else {
                        &[1, 0, 2]
                    };
                    from(f64::from(ij[0] == order[ij[1]]))
                },
            )
            .unwrap();
            assert_eq!(v.dense_data().unwrap(), expected.dense_data().unwrap());
        }};
    }
    check!(f64, |x| x);
    check!(Complex64, |x| Complex64::new(x, 0.0));
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eigh_full_matches_independent_dense_oracle() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let dual_leg = leg.try_dual().unwrap();
    let dual_sectors = dual_leg.sectors().unwrap();
    macro_rules! check {
        ($dtype:ty, $from:expr, $tol:expr) => {{
            let from: fn(f64) -> $dtype = $from;
            let tol: f64 = $tol;
            let distance = |a: &TensorMap<_, $dtype>, b: &TensorMap<_, $dtype>| {
                a.axpby(from(1.0), b, from(-1.0))
                    .unwrap()
                    .norm(2.0)
                    .unwrap()
            };
            let build = |space: &GradedSpace<_>, sectors: [Vec<i64>; 2], values: [Vec<f64>; 2]| {
                let [first, second] = values;
                let [s0, s1] = sectors;
                TensorMap::<_, $dtype>::diagonal(
                    &runtime,
                    space,
                    [
                        SectorSpectrum {
                            sector: s0,
                            values: first.into_iter().map(from).collect(),
                        },
                        SectorSpectrum {
                            sector: s1,
                            values: second.into_iter().map(from).collect(),
                        },
                    ],
                )
                .unwrap()
            };
            let distinct = [vec![-2.0, 5.0], vec![3.0, -1.0, 0.5]];
            for (space, sectors) in [
                (&leg, [vec![0, 0], vec![1, 0]]),
                (
                    &dual_leg,
                    [dual_sectors[0].clone(), dual_sectors[1].clone()],
                ),
            ] {
                let input = build(space, sectors, distinct.clone());
                // Oracle: an independently materialized dense payload through
                // the dense checked EIGH route.
                let dense = input.materialize().unwrap();
                let expected = dense.eigh_full(&[0], &[1]).unwrap();
                DIAGONAL_MATERIALIZATIONS.set(0);
                let got = input.eigh_full(&[0], &[1]).unwrap();
                assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
                assert_eq!(got.v.codomain(), expected.v.codomain());
                assert_eq!(got.v.domain(), expected.v.domain());
                assert_eq!(got.d.codomain(), expected.d.codomain());
                for (actual, oracle) in got
                    .d
                    .diagview()
                    .unwrap()
                    .iter()
                    .zip(expected.d.diagview().unwrap())
                {
                    assert_eq!(actual.sector, oracle.sector);
                    for (&a, &b) in actual.values.iter().zip(&oracle.values) {
                        assert!(ScalarOps::abs_value(a - b) <= tol);
                    }
                }
                assert!(distance(&got.v, &expected.v) <= tol);

                // Changed leg roles go through an explicit permute.
                let swapped = input.eigh_full(&[1], &[0]).unwrap();
                let swapped_oracle = input
                    .permute(&[1], &[0])
                    .unwrap()
                    .materialize()
                    .unwrap()
                    .eigh_full(&[0], &[1])
                    .unwrap();
                assert_eq!(
                    swapped.d.diagview().unwrap().len(),
                    swapped_oracle.d.diagview().unwrap().len()
                );
                for (actual, oracle) in swapped
                    .d
                    .diagview()
                    .unwrap()
                    .iter()
                    .zip(swapped_oracle.d.diagview().unwrap())
                {
                    assert_eq!(actual.sector, oracle.sector);
                    for (&a, &b) in actual.values.iter().zip(&oracle.values) {
                        assert!(ScalarOps::abs_value(a - b) <= tol);
                    }
                }
                assert!(distance(&swapped.v, &swapped_oracle.v) <= tol);
            }

            // Ties: exact equal values and ±x. Degenerate bases are not a
            // public guarantee, so check the factorization itself.
            let input = build(
                &leg,
                [vec![0, 0], vec![1, 0]],
                [vec![1.0, 1.0], vec![2.0, -2.0, 2.0]],
            );
            let dense = input.materialize().unwrap();
            let oracle = dense.eigh_full(&[0], &[1]).unwrap();
            DIAGONAL_MATERIALIZATIONS.set(0);
            let Eigh { d, v } = input.eigh_full(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert_eq!(
                d.diagview().unwrap(),
                vec![
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![from(1.0), from(1.0)],
                    },
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![from(-2.0), from(2.0), from(2.0)],
                    },
                ]
            );
            let multiset = |spectrum: Vec<SectorSpectrum<Vec<i64>, $dtype>>| {
                spectrum
                    .into_iter()
                    .map(|entry| {
                        let mut values: Vec<f64> = entry
                            .values
                            .iter()
                            .map(|value| FactorScalar::widen_complex(*value).re)
                            .collect();
                        values.sort_by(f64::total_cmp);
                        (entry.sector, values)
                    })
                    .collect::<Vec<_>>()
            };
            let got_multiset = multiset(d.diagview().unwrap());
            for ((sector, a), (oracle_sector, b)) in got_multiset
                .iter()
                .zip(multiset(oracle.d.diagview().unwrap()))
            {
                assert_eq!(*sector, oracle_sector);
                for (a, b) in a.iter().zip(&b) {
                    assert!((a - b).abs() <= tol);
                }
            }
            let rebuilt = v
                .compose(&d)
                .unwrap()
                .compose(&v.adjoint().unwrap().materialize().unwrap())
                .unwrap();
            assert!(distance(&rebuilt, &dense) <= tol);
            let gram = v
                .adjoint()
                .unwrap()
                .materialize()
                .unwrap()
                .compose(&v)
                .unwrap();
            let identity: TensorMap<_, $dtype> = TensorMap::from_subblock_fn(
                &runtime,
                gram.codomain().iter(),
                gram.domain().iter(),
                |_, ij| from(f64::from(ij[0] == ij[1])),
            )
            .unwrap();
            assert!(distance(&gram, &identity) <= tol);
        }};
    }
    check!(f32, |x| x as f32, 1e-5);
    check!(f64, |x| x, 1e-12);
    check!(
        num_complex::Complex32,
        |x| num_complex::Complex32::new(x as f32, 0.0),
        1e-5
    );
    check!(Complex64, |x| Complex64::new(x, 0.0), 1e-12);
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eig_full_avoids_input_materialization_and_solver() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let svd_vals_calls = Arc::clone(&calls);
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eig_full_spy(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let c = |re: f64, im: f64| Complex64::new(re, im);
    macro_rules! check {
        ($dtype:ty, $from:expr, [$($a:expr),*], [$($b:expr),*], [$($da:expr),*], [$($db:expr),*]) => {{
            let from: fn(Complex64) -> $dtype = $from;
            let input: TensorMap<_, $dtype> = TensorMap::diagonal(
                &runtime,
                &leg,
                [
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![$(from($b)),*],
                    },
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![$(from($a)),*],
                    },
                ],
            )
            .unwrap();
            DIAGONAL_MATERIALIZATIONS.set(0);
            calls.reset();
            svd_vals_calls.reset();
            let Eig { d, v } = input.eig_full(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert_eq!(calls.of(EIG_FULL), 0);
            assert_eq!(svd_vals_calls.get(Kernel::SvdVals), 0);
            assert!(std::ptr::eq(d.provider(), provider.as_ref()));
            assert!(std::ptr::eq(v.provider(), provider.as_ref()));
            assert!(d.dense_data().is_err());
            assert_eq!(
                d.diagview().unwrap(),
                vec![
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![$($da),*],
                    },
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![$($db),*],
                    },
                ]
            );
            assert_eq!(v.codomain(), input.codomain());
            assert!(!v.domain()[0].is_dual());
            // Column j of each sector block is the unit vector of the stored
            // index that sorts into position j.
            let expected: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
                &runtime,
                v.codomain().iter(),
                v.domain().iter(),
                |trees, ij| {
                    let order: &[usize] = if trees.coupled() == &vec![0, 0] {
                        &[0, 1]
                    } else {
                        &[1, 2, 0]
                    };
                    c(f64::from(ij[0] == order[ij[1]]), 0.0)
                },
            )
            .unwrap();
            assert_eq!(v.dense_data().unwrap(), expected.dense_data().unwrap());

            // The counters observe the dense route: changed leg roles
            // materialize and run one EIG and one rank gate per sector.
            DIAGONAL_MATERIALIZATIONS.set(0);
            input.eig_full(&[1], &[0]).unwrap();
            assert_eq!(calls.of(EIG_FULL), 2);
            assert_eq!(svd_vals_calls.get(Kernel::SvdVals), 2);
        }};
    }
    // Magnitudes: sector (0,0) 4 > 0.25; sector (1,0) 5 > |-2+0.5i| > |1+i|.
    check!(
        Complex64,
        |z| z,
        [c(-4.0, 0.0), c(0.0, 0.25)],
        [c(1.0, 1.0), c(0.0, -5.0), c(-2.0, 0.5)],
        [c(-4.0, 0.0), c(0.0, 0.25)],
        [c(0.0, -5.0), c(-2.0, 0.5), c(1.0, 1.0)]
    );
    check!(
        f64,
        |z| z.re,
        [c(-4.0, 0.0), c(0.25, 0.0)],
        [c(1.0, 0.0), c(-5.0, 0.0), c(-2.0, 0.0)],
        [c(-4.0, 0.0), c(0.25, 0.0)],
        [c(-5.0, 0.0), c(-2.0, 0.0), c(1.0, 0.0)]
    );
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eig_full_matches_independent_dense_oracle() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let dual_leg = leg.try_dual().unwrap();
    let dual_sectors = dual_leg.sectors().unwrap();
    macro_rules! check {
        ($dtype:ty, $eig:ty, $from:expr, $complex:expr, $tol:expr) => {{
            // Payload values are rounded to `$dtype` first; `$eig` widening
            // is exact afterwards.
            let from: fn(f64, f64) -> $dtype = $from;
            let eig = |value: $dtype| -> $eig {
                <$eig as FactorScalar>::from_complex64(FactorScalar::widen_complex(value))
            };
            let tol: f64 = $tol;
            let im = if $complex { 1.0 } else { 0.0 };
            let distance = |a: &TensorMap<_, $eig>, b: &TensorMap<_, $eig>| {
                a.axpby(<$eig>::new(1.0, 0.0), b, <$eig>::new(-1.0, 0.0))
                    .unwrap()
                    .norm(2.0)
                    .unwrap()
            };
            let build = |space: &GradedSpace<_>,
                         sectors: [Vec<i64>; 2],
                         values: [Vec<(f64, f64)>; 2]| {
                let [first, second] = values;
                let [s0, s1] = sectors;
                TensorMap::<_, $dtype>::diagonal(
                    &runtime,
                    space,
                    [
                        SectorSpectrum {
                            sector: s0,
                            values: first.into_iter().map(|(re, i)| from(re, i * im)).collect(),
                        },
                        SectorSpectrum {
                            sector: s1,
                            values: second.into_iter().map(|(re, i)| from(re, i * im)).collect(),
                        },
                    ],
                )
                .unwrap()
            };
            let assert_d = |got: &TensorMap<_, $eig>, oracle: &TensorMap<_, $eig>| {
                let (got, oracle) = (got.diagview().unwrap(), oracle.diagview().unwrap());
                assert_eq!(got.len(), oracle.len());
                for (actual, oracle) in got.iter().zip(&oracle) {
                    assert_eq!(actual.sector, oracle.sector);
                    assert_eq!(actual.values.len(), oracle.values.len());
                    for (&a, &b) in actual.values.iter().zip(&oracle.values) {
                        assert!(ScalarOps::abs_value(a - b) <= tol, "{a} != {b}");
                    }
                }
            };
            // Distinct magnitudes: |0.1-0.3i|<0.7 and 0.2(+0.2i)<1.3(i)<2.9.
            let distinct = [
                vec![(0.1, -0.3), (-0.7, 0.0)],
                vec![(0.0, 1.3), (0.2, 0.2), (-2.9, 0.1)],
            ];
            let distinct = if $complex {
                distinct
            } else {
                [
                    vec![(0.1, 0.0), (-0.7, 0.0)],
                    vec![(1.3, 0.0), (0.2, 0.0), (-2.9, 0.0)],
                ]
            };
            for (space, sectors, dual) in [
                (&leg, [vec![0, 0], vec![1, 0]], false),
                (
                    &dual_leg,
                    [dual_sectors[0].clone(), dual_sectors[1].clone()],
                    true,
                ),
            ] {
                let input = build(space, sectors, distinct.clone());
                // Oracle: an independently materialized dense payload through
                // the dense checked EIG route.
                let dense = input.materialize().unwrap();
                let expected = dense.eig_full(&[0], &[1]).unwrap();
                DIAGONAL_MATERIALIZATIONS.set(0);
                let got = input.eig_full(&[0], &[1]).unwrap();
                assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
                assert_eq!(got.v.codomain(), expected.v.codomain());
                assert_eq!(got.v.domain(), expected.v.domain());
                assert_eq!(got.d.codomain(), expected.d.codomain());
                assert_d(&got.d, &expected.d);
                assert!(distance(&got.v, &expected.v) <= tol);
                if !dual {
                    // Hand values: the rounded payload, widened exactly.
                    let stored = input.diagview().unwrap();
                    let by = |sector: &[i64]| {
                        stored
                            .iter()
                            .find(|entry| entry.sector == sector)
                            .unwrap()
                            .values
                            .clone()
                    };
                    let (a, b) = (by(&[0, 0]), by(&[1, 0]));
                    assert_eq!(
                        got.d.diagview().unwrap(),
                        vec![
                            SectorSpectrum {
                                sector: vec![0, 0],
                                values: vec![eig(a[1]), eig(a[0])],
                            },
                            SectorSpectrum {
                                sector: vec![1, 0],
                                values: vec![eig(b[2]), eig(b[0]), eig(b[1])],
                            },
                        ]
                    );
                }

                // Changed leg roles go through an explicit permute.
                let swapped = input.eig_full(&[1], &[0]).unwrap();
                let swapped_oracle = input
                    .permute(&[1], &[0])
                    .unwrap()
                    .materialize()
                    .unwrap()
                    .eig_full(&[0], &[1])
                    .unwrap();
                assert_d(&swapped.d, &swapped_oracle.d);
                assert!(distance(&swapped.v, &swapped_oracle.v) <= tol);
            }

            // Equal-magnitude ties keep stored order. Degenerate bases are not
            // a public guarantee, so check the factorization itself.
            let ties = if $complex {
                [
                    vec![(2.0, 0.0), (0.0, -2.0)],
                    vec![(-3.0, 0.0), (1.0, 0.0), (0.0, 3.0)],
                ]
            } else {
                [
                    vec![(2.0, 0.0), (-2.0, 0.0)],
                    vec![(-3.0, 0.0), (1.0, 0.0), (3.0, 0.0)],
                ]
            };
            let input = build(&leg, [vec![0, 0], vec![1, 0]], ties);
            let stored = input.diagview().unwrap();
            let by = |sector: &[i64]| {
                stored
                    .iter()
                    .find(|entry| entry.sector == sector)
                    .unwrap()
                    .values
                    .clone()
            };
            let dense = input.materialize().unwrap();
            let oracle = dense.eig_full(&[0], &[1]).unwrap();
            DIAGONAL_MATERIALIZATIONS.set(0);
            let Eig { d, v } = input.eig_full(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            let (a, b) = (by(&[0, 0]), by(&[1, 0]));
            assert_eq!(
                d.diagview().unwrap(),
                vec![
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![eig(a[0]), eig(a[1])],
                    },
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![eig(b[0]), eig(b[2]), eig(b[1])],
                    },
                ]
            );
            let widen = |spectrum: Vec<SectorSpectrum<Vec<i64>, $eig>>| {
                spectrum
                    .into_iter()
                    .map(|entry| SectorSpectrum {
                        sector: entry.sector,
                        values: entry
                            .values
                            .into_iter()
                            .map(FactorScalar::widen_complex)
                            .collect(),
                    })
                    .collect::<Vec<_>>()
            };
            assert_same_complex_multisets(
                &widen(d.diagview().unwrap()),
                &widen(oracle.d.diagview().unwrap()),
                tol,
            );
            let rebuilt = v
                .compose(&d)
                .unwrap()
                .compose(&v.inv(&[0], &[1]).unwrap())
                .unwrap();
            let widened_input = TensorMap::<_, $eig>::diagonal(
                &runtime,
                &leg,
                stored.iter().map(|entry| SectorSpectrum {
                    sector: entry.sector.clone(),
                    values: entry.values.iter().map(|&value| eig(value)).collect(),
                }),
            )
            .unwrap()
            .materialize()
            .unwrap();
            assert!(distance(&rebuilt, &widened_input) <= tol);
        }};
    }
    check!(f32, num_complex::Complex32, |re, _| re as f32, false, 1e-5);
    check!(f64, Complex64, |re, _| re, false, 1e-12);
    check!(
        num_complex::Complex32,
        num_complex::Complex32,
        |re, im| num_complex::Complex32::new(re as f32, im as f32),
        true,
        1e-5
    );
    check!(Complex64, Complex64, Complex64::new, true, 1e-12);
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eig_full_refuses_nonfinite_values() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eig_full_spy(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 1), (vec![1, 0], 2)]).unwrap();
    for bad in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(0.0, f64::INFINITY),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: vec![0, 0],
                    values: vec![Complex64::new(1.0, 0.0)],
                },
                SectorSpectrum {
                    sector: vec![1, 0],
                    values: vec![bad, Complex64::new(3.0, 0.0)],
                },
            ],
        )
        .unwrap();
        // The shared finite-input stage (the dense route's own checked
        // pre-check, same error), or the dense eigenvalue check for the
        // overflowing `MAX + MAX i`; no dense solver either way.
        let dense_error = input
            .materialize()
            .unwrap()
            .eig_full(&[0], &[1])
            .map(drop)
            .unwrap_err();
        calls.reset();
        DIAGONAL_MATERIALIZATIONS.set(0);
        let compact_error = input.eig_full(&[0], &[1]).map(drop).unwrap_err();
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        assert_eq!(calls.total(), 0);
        assert_eq!(format!("{compact_error:?}"), format!("{dense_error:?}"));
    }

    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![1.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![f64::NAN, 2.0],
            },
        ],
    )
    .unwrap();
    let dense_error = nonfinite
        .materialize()
        .unwrap()
        .eig_full(&[0], &[1])
        .map(drop)
        .unwrap_err();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let compact_error = nonfinite.eig_full(&[0], &[1]).map(drop).unwrap_err();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(format!("{compact_error:?}"), format!("{dense_error:?}"));
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eig_full_rejects_inconsistent_spectrum_admission() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(1.0, 2.0), Complex64::new(-1.0, 0.0)],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![Complex64::new(0.0, 2.0)],
            },
        ],
    )
    .unwrap();
    let spectrum = input.spectrum().unwrap();
    // A spectrum that does not cover the bond is misuse, refused with a typed
    // error; no dense executor is used.
    let admit = |entries: &[tenet_matrixalgebra::SectorSpectrum<Complex64>]| {
        tenet_matrixalgebra::seam::eig_full_checked_generic(
            &mut tenet_dense::DefaultDenseExecutor::new(),
            tenet_matrixalgebra::seam::FactorSource::Diagonal {
                space: input.logical_space(),
                spectrum: entries,
            },
        )
        .is_ok()
    };
    assert!(admit(spectrum));
    let mut reversed = spectrum.to_vec();
    reversed.reverse();
    assert!(admit(&reversed));
    assert!(!admit(&spectrum[..1]));
    let mut duplicate = spectrum.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(!admit(&duplicate));
    let mut wrong_size = spectrum.to_vec();
    wrong_size[0].values.push(Complex64::new(0.0, 0.0));
    assert!(!admit(&wrong_size));
    for bad in [
        Complex64::new(f64::INFINITY, 0.0),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        // A nonfinite value is refused by the shared finite-input stage, and
        // an overflowing magnitude by the dense route's eigenvalue check.
        let mut nonfinite = spectrum.to_vec();
        nonfinite[0].values[0] = bad;
        assert!(!admit(&nonfinite));
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eigh_full_applies_the_dense_hermiticity_check_directly() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eigh_full_spy(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 1), (vec![1, 0], 2)]).unwrap();
    let complex = |values: [Complex64; 2]| {
        TensorMap::<_, Complex64>::diagonal(
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: vec![0, 0],
                    values: vec![values[0]],
                },
                SectorSpectrum {
                    sector: vec![1, 0],
                    values: vec![values[1], Complex64::new(3.0, 0.0)],
                },
            ],
        )
        .unwrap()
    };
    // A nonzero imaginary part meets the dense route's Hermiticity check,
    // applied to the diagonal directly.
    let near = complex([Complex64::new(1.0, 1e-15), Complex64::new(-1.0, 0.0)]);
    DIAGONAL_MATERIALIZATIONS.set(0);
    let Eigh { d, .. } = near.eigh_full(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.total(), 0);
    assert_eq!(d.diagview().unwrap()[0].values, [Complex64::new(1.0, 0.0)]);

    let non_hermitian = complex([Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.5)]);
    let dense_error = non_hermitian
        .materialize()
        .unwrap()
        .eigh_full(&[0], &[1])
        .unwrap_err();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let compact_error = non_hermitian.eigh_full(&[0], &[1]).unwrap_err();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(compact_error.to_string(), dense_error.to_string());

    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![1.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![f64::NAN, 2.0],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let compact_error = nonfinite.eigh_full(&[0], &[1]).map(drop).unwrap_err();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert!(compact_error
        .to_string()
        .contains("eigh input components must be finite"));
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eigh_full_rejects_inconsistent_spectrum_admission() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![1.0, -1.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![2.0],
            },
        ],
    )
    .unwrap();
    let spectrum = input.spectrum().unwrap();
    // A spectrum that does not cover the bond is misuse, refused with a typed
    // error; no dense executor is used.
    let admit = |entries: &[tenet_matrixalgebra::SectorSpectrum<f64>]| {
        tenet_matrixalgebra::seam::eigh_full_checked_generic(
            &mut tenet_dense::DefaultDenseExecutor::new(),
            tenet_matrixalgebra::seam::FactorSource::Diagonal {
                space: input.logical_space(),
                spectrum: entries,
            },
        )
        .is_ok()
    };
    assert!(admit(spectrum));
    let mut reversed = spectrum.to_vec();
    reversed.reverse();
    assert!(admit(&reversed));
    assert!(!admit(&spectrum[..1]));
    let mut duplicate = spectrum.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(!admit(&duplicate));
    let mut wrong_size = spectrum.to_vec();
    wrong_size[0].values.push(0.0);
    assert!(!admit(&wrong_size));
    // A nonfinite value is refused by the shared finite-input stage.
    let mut complex = spectrum.to_vec();
    complex[0].values[0] = f64::INFINITY;
    assert!(!admit(&complex));
}
