use super::*;

#[test]
fn compact_diagonal_svd_full_preserves_spaces_without_materialization_or_solver() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(polar_spy(&calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(2), 3)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(-2),
            values: vec![0.0, -4.0, 2.0],
        }],
    )
    .unwrap();
    let w = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(-2), 3)]).unwrap();
    let dense = input.materialize().unwrap();
    let old = dense.svd_full(&[0], &[1]).unwrap();
    assert!(calls.of(POLAR_SVD) > 0);
    calls.reset();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let out = input.svd_full(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.of(POLAR_SVD), 0);
    assert_eq!(out.u.codomain(), input.codomain());
    assert_eq!(out.vh.domain(), input.domain());
    assert_eq!(out.u.domain(), vec![w.clone()]);
    assert_eq!(out.s.codomain(), vec![w.clone()]);
    assert_eq!(out.s.domain(), vec![w.clone()]);
    assert_eq!(out.vh.codomain(), vec![w]);
    for (actual, expected) in [(&out.u, &old.u), (&out.s, &old.s), (&out.vh, &old.vh)] {
        assert_eq!(actual.codomain(), expected.codomain());
        assert_eq!(actual.domain(), expected.domain());
    }
    assert!(out.u.dense_data().is_ok());
    assert!(out.vh.dense_data().is_ok());
    assert!(out.s.dense_data().is_err());
    assert_eq!(out.s.diagview().unwrap()[0].values, vec![4.0, 2.0, 0.0]);
    assert_eq!(
        out.s.diagonal_spectrum().unwrap().unwrap(),
        out.s.diagview().unwrap()
    );
    let rebuilt = out.u.compose(&out.s).unwrap().compose(&out.vh).unwrap();
    assert_eq!(rebuilt.dense_data().unwrap(), dense.dense_data().unwrap());
    for (gram, identity) in [
        (
            out.u.adjoint().unwrap().compose(&out.u).unwrap(),
            TensorMap::isomorphism(&runtime, &out.u.domain(), &out.u.domain()).unwrap(),
        ),
        (
            out.vh.compose(&out.vh.adjoint().unwrap()).unwrap(),
            TensorMap::isomorphism(&runtime, &out.vh.codomain(), &out.vh.codomain()).unwrap(),
        ),
    ] {
        assert!(gram.axpby(1.0, &identity, -1.0).unwrap().norm(2.0).unwrap() < 1e-12);
    }
    assert_eq!(
        out.s.materialize().unwrap().dense_data().unwrap(),
        &[4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn full_svd_compact_bond_requires_complete_exact_nondual_sector_legs() {
    let a = tenet_core::SectorId::new(1);
    let b = tenet_core::SectorId::new(2);
    let row = tenet_core::SectorLeg::new([(a, 2), (b, 3)], false);
    let exact = vec![
        tenet_matrixalgebra::SectorSpectrum {
            sector: a,
            values: vec![4.0, 1.0],
        },
        tenet_matrixalgebra::SectorSpectrum {
            sector: b,
            values: vec![3.0, 0.0, 0.0],
        },
    ];
    assert!(full_svd_spectrum_matches_bonds(&row, &row, &exact));
    let same_total = tenet_core::SectorLeg::new([(a, 3), (b, 2)], false);
    assert!(!full_svd_spectrum_matches_bonds(&row, &same_total, &exact));
    assert!(!full_svd_spectrum_matches_bonds(&row, &row, &exact[..1]));
    let dual = tenet_core::SectorLeg::new([(a, 2), (b, 3)], true);
    assert!(!full_svd_spectrum_matches_bonds(&dual, &dual, &exact));
    let duplicate = [exact[0].clone(), exact[0].clone()];
    assert!(!full_svd_spectrum_matches_bonds(&row, &row, &duplicate));
}

#[test]
fn dense_multisector_full_svd_publishes_compact_s_for_real_and_complex() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let input: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 2) as f64
            } else {
                0.25
            }
        })
        .unwrap();
    let Svd { u, s, vh } = input.svd_full(&[0], &[1]).unwrap();
    assert!(s.dense_data().is_err());
    let values = s.diagview().unwrap();
    assert_eq!(values.len(), 2);
    assert_eq!(values[0].sector, U1Irrep::new(0));
    assert_eq!(values[1].sector, U1Irrep::new(1));
    assert_eq!(values[0].values.len(), 2);
    assert_eq!(values[1].values.len(), 3);
    assert!(values
        .iter()
        .all(|entry| entry.values.windows(2).all(|pair| pair[0] >= pair[1])));
    assert_eq!(
        s.materialize().unwrap().diagview().unwrap(),
        s.diagview().unwrap()
    );
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
    assert!(input.svd_full(&[1], &[0]).unwrap().s.dense_data().is_err());

    let complex = input.convert::<Complex64>().scale(Complex64::new(1.0, 0.5));
    let Svd { u, s, vh } = complex.svd_full(&[0], &[1]).unwrap();
    assert!(s.dense_data().is_err());
    assert_eq!(s.diagview().unwrap().len(), 2);
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-12));
}

#[test]
fn compact_diagonal_null_uses_coordinate_kernel_without_dense_solver() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(polar_spy(&calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![3.0, 0.0, -2.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![0.0, 0.0],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let left = input.left_null(&[0], &[1]).unwrap();
    let right = input.right_null(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.of(POLAR_SVD), 0);
    assert_eq!(left.codomain(), input.codomain());
    assert_eq!(right.domain(), input.domain());
    assert_eq!(left.domain()[0].sectors().unwrap().len(), 2);
    assert_eq!(right.codomain()[0].sectors().unwrap().len(), 2);
    assert_eq!(left.domain()[0].degeneracy(&U1Irrep::new(0)).unwrap(), 1);
    assert_eq!(left.domain()[0].degeneracy(&U1Irrep::new(1)).unwrap(), 2);
    assert_eq!(right.codomain()[0].degeneracy(&U1Irrep::new(0)).unwrap(), 1);
    assert_eq!(right.codomain()[0].degeneracy(&U1Irrep::new(1)).unwrap(), 2);
    let coordinates = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    assert_eq!(left.dense_data().unwrap(), coordinates);
    assert_eq!(right.dense_data().unwrap(), coordinates);
    assert!(
        left.adjoint()
            .unwrap()
            .compose(&input)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    assert!(
        input
            .compose(&right.adjoint().unwrap())
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
}

#[test]
fn compact_diagonal_null_su2_sectors_use_reduced_coordinate_basis() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(polar_spy(&calls)))
        .build()
        .unwrap();
    let spin0 = SU2Irrep::from_twice_spin(0);
    let half = SU2Irrep::from_twice_spin(1);
    let leg = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 3), (half, 2)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: spin0,
                values: vec![3.0, 0.0, -2.0],
            },
            SectorSpectrum {
                sector: half,
                values: vec![0.0, 4.0],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let left = input.left_null(&[0], &[1]).unwrap();
    let right = input.right_null(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.of(POLAR_SVD), 0);
    assert_eq!(left.codomain(), input.codomain());
    assert_eq!(right.domain(), input.domain());
    for bond in [&left.domain()[0], &right.codomain()[0]] {
        assert!(!bond.is_dual());
        assert_eq!(bond.sectors().unwrap().len(), 2);
        assert_eq!(bond.degeneracy(&spin0).unwrap(), 1);
        assert_eq!(bond.degeneracy(&half).unwrap(), 1);
    }
    let coordinates = [0.0, 1.0, 0.0, 1.0, 0.0];
    assert_eq!(left.dense_data().unwrap(), coordinates);
    assert_eq!(right.dense_data().unwrap(), coordinates);
    assert!(
        left.adjoint()
            .unwrap()
            .compose(&input)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    assert!(
        input
            .compose(&right.adjoint().unwrap())
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    let left_gram = left.adjoint().unwrap().compose(&left).unwrap();
    let right_gram = right.compose(&right.adjoint().unwrap()).unwrap();
    let left_eye = TensorMap::isomorphism(&runtime, &left.domain(), &left.domain()).unwrap();
    let right_eye = TensorMap::isomorphism(&runtime, &right.codomain(), &right.codomain()).unwrap();
    assert!(
        left_gram
            .axpby(1.0, &left_eye, -1.0)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    assert!(
        right_gram
            .axpby(1.0, &right_eye, -1.0)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
}

#[test]
fn compact_diagonal_null_and_cutoff_cover_all_scalars() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(polar_spy(&calls)))
        .build()
        .unwrap();
    let near_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let sector = U1Irrep::new(0);
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(sector, 3)]).unwrap();
    macro_rules! check_scalar {
        ($scalar:ty, $convert:expr, $eps:expr) => {{
            let input: TensorMap<_, $scalar> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector,
                    values: vec![($convert)(0.0), ($convert)(3.0), ($convert)(0.0)],
                }],
            )
            .unwrap();
            calls.reset();
            DIAGONAL_MATERIALIZATIONS.set(0);
            let left = input.left_null(&[0], &[1]).unwrap();
            let right = input.right_null(&[0], &[1]).unwrap();
            assert_eq!(calls.of(POLAR_SVD), 0);
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            let coordinates = [
                ($convert)(1.0),
                ($convert)(0.0),
                ($convert)(0.0),
                ($convert)(0.0),
                ($convert)(0.0),
                ($convert)(1.0),
            ];
            assert_eq!(left.dense_data().unwrap(), coordinates);
            assert_eq!(right.dense_data().unwrap(), coordinates);
            assert_eq!(left.domain()[0].degeneracy(&sector).unwrap(), 2);
            assert_eq!(right.codomain()[0].degeneracy(&sector).unwrap(), 2);

            let full: TensorMap<_, $scalar> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector,
                    values: vec![($convert)(1.0), ($convert)(2.0), ($convert)(3.0)],
                }],
            )
            .unwrap();
            calls.reset();
            DIAGONAL_MATERIALIZATIONS.set(0);
            let full_left = full.left_null(&[0], &[1]).unwrap();
            let full_right = full.right_null(&[0], &[1]).unwrap();
            assert_eq!(calls.of(POLAR_SVD), 0);
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert!(full_left.dense_data().unwrap().is_empty());
            assert!(full_right.dense_data().unwrap().is_empty());
            assert!(full_left.domain()[0].sectors().unwrap().is_empty());
            assert!(full_right.codomain()[0].sectors().unwrap().is_empty());

            // The dense route's rank cutoff `eps * max(rows, cols) * sigma_max`
            // is applied to `|a_i|` directly: at or below it is null.
            let cutoff = ($eps as f64) * 2.0;
            let small_leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(sector, 2)]).unwrap();
            for (small, expected_len) in [(0.5 * cutoff, 2), (cutoff, 2), (2.0 * cutoff, 0)] {
                let near: TensorMap<_, $scalar> = TensorMap::diagonal(
                    &near_runtime,
                    &small_leg,
                    [SectorSpectrum {
                        sector,
                        values: vec![($convert)(1.0), ($convert)(small)],
                    }],
                )
                .unwrap();
                DIAGONAL_MATERIALIZATIONS.set(0);
                let left = near.left_null(&[0], &[1]).unwrap();
                let right = near.right_null(&[0], &[1]).unwrap();
                assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
                assert_eq!(left.dense_data().unwrap().len(), expected_len);
                assert_eq!(right.dense_data().unwrap().len(), expected_len);
                if expected_len != 0 {
                    // The null direction is the unit vector of the small value.
                    let unit = [($convert)(0.0), ($convert)(1.0)];
                    assert_eq!(left.dense_data().unwrap(), unit);
                    assert_eq!(right.dense_data().unwrap(), unit);
                }
            }
        }};
    }
    check_scalar!(f32, |x: f64| x as f32, f32::EPSILON);
    check_scalar!(f64, |x: f64| x, f64::EPSILON);
    check_scalar!(
        num_complex::Complex32,
        |x: f64| num_complex::Complex32::new(x as f32, 0.0),
        f32::EPSILON
    );
    check_scalar!(
        num_complex::Complex64,
        |x: f64| num_complex::Complex64::new(x, 0.0),
        f64::EPSILON
    );

    let tiny: TensorMap<_, f32> = TensorMap::diagonal(
        &near_runtime,
        &GradedSpace::try_new(Arc::new(U1FusionRule), [(sector, 1)]).unwrap(),
        [SectorSpectrum {
            sector,
            values: vec![f32::from_bits(1)],
        }],
    )
    .unwrap();
    // A subnormal is above its own `eps * sigma_max` cutoff, so the sector has
    // full rank; read directly, with no materialization.
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert!(tiny
        .left_null(&[0], &[1])
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let unrepresentable: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &near_runtime,
        &GradedSpace::try_new(Arc::new(U1FusionRule), [(sector, 1)]).unwrap(),
        [SectorSpectrum {
            sector,
            values: vec![num_complex::Complex32::new(f32::MAX, f32::MAX)],
        }],
    )
    .unwrap();
    // `|MAX (1 + i)|` overflows f32, so the sector is ranked in scaled form:
    // a nonzero 1 x 1 sector has full rank and an empty null space.
    DIAGONAL_MATERIALIZATIONS.set(0);
    let null = unrepresentable.left_null(&[0], &[1]).unwrap();
    assert!(null.dense_data().unwrap().is_empty());
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    // Hand-computed, scaled by `m = MAX`: `|a| / m = (sqrt 2, 1 / MAX)` and the
    // cutoff `eps * 2 * sqrt 2`, so the `1` direction is null and the overflow
    // direction is not.
    let mixed: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &near_runtime,
        &GradedSpace::try_new(Arc::new(U1FusionRule), [(sector, 2)]).unwrap(),
        [SectorSpectrum {
            sector,
            values: vec![
                num_complex::Complex32::new(f32::MAX, f32::MAX),
                num_complex::Complex32::new(1.0, 0.0),
            ],
        }],
    )
    .unwrap();
    let null = mixed.left_null(&[0], &[1]).unwrap();
    assert_eq!(
        null.dense_data().unwrap(),
        [
            num_complex::Complex32::new(0.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0)
        ]
    );
}

#[test]
fn compact_diagonal_null_preserves_dual_space_and_lazy_adjoint_semantics() {
    use num_complex::Complex64;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(1), 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(-1),
            values: vec![Complex64::new(0.0, 0.0), Complex64::new(2.0, 3.0)],
        }],
    )
    .unwrap();
    let dense = input.materialize().unwrap();
    let lazy = dense.adjoint().unwrap();
    let left = input.left_null(&[0], &[1]).unwrap();
    let right = input.right_null(&[0], &[1]).unwrap();
    assert_eq!(left.codomain(), input.codomain());
    assert_eq!(right.domain(), input.domain());
    assert_eq!(left.domain()[0].degeneracy(&U1Irrep::new(-1)).unwrap(), 1);
    assert!(!left.domain()[0].is_dual());
    assert!(!right.codomain()[0].is_dual());
    assert_eq!(
        left.dense_data().unwrap(),
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    );
    assert_eq!(
        right.dense_data().unwrap(),
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    );
    let lazy_left = lazy.left_null(&[0], &[1]).unwrap();
    let lazy_right = lazy.right_null(&[0], &[1]).unwrap();
    assert_eq!(lazy_left.codomain(), lazy.codomain());
    assert_eq!(lazy_right.domain(), lazy.domain());
    assert_eq!(
        lazy_left.domain()[0].degeneracy(&U1Irrep::new(-1)).unwrap(),
        1
    );
    assert_eq!(
        lazy_right.codomain()[0]
            .degeneracy(&U1Irrep::new(-1))
            .unwrap(),
        1
    );
    assert!(
        lazy_left
            .adjoint()
            .unwrap()
            .compose(&lazy)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    assert!(
        lazy.compose(&lazy_right.adjoint().unwrap())
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
}

#[test]
fn compact_diagonal_null_product_sectors_roles_and_nonfinite_refusal() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(even, 2), (odd, 2)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: even,
                values: vec![0.0, -2.0],
            },
            SectorSpectrum {
                sector: odd,
                values: vec![3.0, 0.0],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let left = input.left_null(&[0], &[1]).unwrap();
    let right = input.right_null(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(left.domain()[0].degeneracy(&even).unwrap(), 1);
    assert_eq!(left.domain()[0].degeneracy(&odd).unwrap(), 1);
    assert_eq!(right.codomain()[0].degeneracy(&even).unwrap(), 1);
    assert_eq!(right.codomain()[0].degeneracy(&odd).unwrap(), 1);
    assert_eq!(left.dense_data().unwrap(), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(right.dense_data().unwrap(), [1.0, 0.0, 0.0, 1.0]);
    let permuted = input.permute(&[1], &[0]).unwrap();
    let changed_left = input.left_null(&[1], &[0]).unwrap();
    let changed_right = input.right_null(&[1], &[0]).unwrap();
    assert_eq!(changed_left.codomain(), permuted.codomain());
    assert_eq!(changed_right.domain(), permuted.domain());
    assert!(
        changed_left
            .adjoint()
            .unwrap()
            .compose(&permuted)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    assert!(
        permuted
            .compose(&changed_right.adjoint().unwrap())
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );

    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: even,
                values: vec![f64::NAN, 0.0],
            },
            SectorSpectrum {
                sector: odd,
                values: vec![3.0, 0.0],
            },
        ],
    )
    .unwrap();
    // A NaN is refused by the shared finite-input stage, with no
    // materialization.
    DIAGONAL_MATERIALIZATIONS.set(0);
    let error = nonfinite.left_null(&[0], &[1]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("null input components must be finite"),
        "{error}"
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn compact_diagonal_svd_full_sorted_hand_oracle_all_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap();
    macro_rules! check {
        ($ty:ty, $second:expr, $phase:expr, $tol:expr) => {{
            let input: TensorMap<_, $ty> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![
                        <$ty as FactorScalar>::from_real(0.0),
                        $second,
                        <$ty as FactorScalar>::from_real(2.0),
                        <$ty as FactorScalar>::from_real(-4.0),
                    ],
                }],
            )
            .unwrap();
            let out = input.svd_full(&[0], &[1]).unwrap();
            assert_eq!(
                out.s.diagview().unwrap()[0].values,
                [4.0, 2.0, 2.0, 0.0].map(<$ty as FactorScalar>::from_real)
            );
            // Hand order: magnitude 4, the two 2s in source order, then zero.
            let permutation = [3, 1, 2, 0];
            let phases = [
                Complex64::new(-1.0, 0.0),
                $phase,
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
            ];
            for col in 0..4 {
                for row in 0..4 {
                    let u = Complex64::new(if row == permutation[col] { 1.0 } else { 0.0 }, 0.0);
                    let vh = if col == permutation[row] {
                        phases[row]
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    assert!(
                        (out.u.dense_data().unwrap()[row + 4 * col].widen_complex() - u).norm()
                            < $tol
                    );
                    assert!(
                        (out.vh.dense_data().unwrap()[row + 4 * col].widen_complex() - vh).norm()
                            < $tol
                    );
                }
            }
        }};
    }
    check!(f32, -2.0, Complex64::new(-1.0, 0.0), 1e-6);
    check!(f64, -2.0, Complex64::new(-1.0, 0.0), 1e-12);
    check!(
        num_complex::Complex32,
        num_complex::Complex32::new(0.0, -2.0),
        Complex64::new(0.0, -1.0),
        1e-6
    );
    check!(
        Complex64,
        Complex64::new(0.0, -2.0),
        Complex64::new(0.0, -1.0),
        1e-12
    );
}

#[test]
fn compact_diagonal_svd_full_subnormal_phase_and_nonfinite_refusal() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    macro_rules! check {
        ($ty:ty, $value:expr, $phase:expr, $tol:expr, $bad:expr) => {{
            let make = |value| {
                TensorMap::<_, $ty>::diagonal(
                    &runtime,
                    &leg,
                    [SectorSpectrum {
                        sector: U1Irrep::new(0),
                        values: vec![value],
                    }],
                )
                .unwrap()
            };
            let input = make($value);
            DIAGONAL_MATERIALIZATIONS.set(0);
            let out = input.svd_full(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert!(out.s.diagview().unwrap()[0].values[0].widen_complex().re > 0.0);
            assert!(
                (out.u.dense_data().unwrap()[0].widen_complex() - Complex64::new(1.0, 0.0)).norm()
                    < $tol
            );
            assert!((out.vh.dense_data().unwrap()[0].widen_complex() - $phase).norm() < $tol);
            assert_eq!(
                out.u
                    .compose(&out.s)
                    .unwrap()
                    .compose(&out.vh)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                &[$value]
            );
            // A nonfinite value is refused by the shared finite-input stage;
            // a finite overflowing one is MAK `svd_full!(::DiagonalAlgorithm)`,
            // `S = abs(a) = Inf`. No materialization either way.
            for value in $bad {
                let input = make(value);
                DIAGONAL_MATERIALIZATIONS.set(0);
                let result = input.svd_full(&[0], &[1]);
                assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
                let widened = value.widen_complex();
                if widened.re.is_finite() && widened.im.is_finite() {
                    let out = result.unwrap();
                    let singular = out.s.diagview().unwrap()[0].values[0].widen_complex().re;
                    assert_eq!(singular, f64::INFINITY);
                } else {
                    let error = result.map(drop).unwrap_err();
                    assert!(
                        error
                            .to_string()
                            .contains("svd input components must be finite"),
                        "{error}"
                    );
                }
            }
        }};
    }
    check!(
        f32,
        -f32::from_bits(1),
        Complex64::new(-1.0, 0.0),
        1e-6,
        [f32::NAN, f32::INFINITY]
    );
    check!(
        f64,
        -f64::from_bits(1),
        Complex64::new(-1.0, 0.0),
        1e-12,
        [f64::NAN, f64::INFINITY]
    );
    let phase = Complex64::new(0.5_f64.sqrt(), 0.5_f64.sqrt());
    check!(
        num_complex::Complex32,
        num_complex::Complex32::new(f32::from_bits(1), f32::from_bits(1)),
        phase,
        1e-6,
        [
            num_complex::Complex32::new(f32::NAN, 0.0),
            num_complex::Complex32::new(f32::INFINITY, 0.0),
            num_complex::Complex32::new(f32::MAX, f32::MAX)
        ]
    );
    check!(
        Complex64,
        Complex64::new(f64::from_bits(1), f64::from_bits(1)),
        phase,
        1e-12,
        [
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(f64::MAX, f64::MAX)
        ]
    );
}

#[test]
fn compact_diagonal_svd_uses_spectrum_without_dense_input() {
    for full in [false, true] {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
        )
        .unwrap();
        let source: TensorMap<_, f64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: U1Irrep::new(1),
                    values: vec![3.0, -2.0],
                },
                SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![1.0, -4.0, 0.0],
                },
            ],
        )
        .unwrap();
        let input = source.materialize().unwrap().dense_data().unwrap().to_vec();
        DIAGONAL_MATERIALIZATIONS.set(0);
        let Svd { u, s, vh } = (if full {
            source.svd_full(&[0], &[1])
        } else {
            source.svd_compact(&[0], &[1])
        })
        .unwrap();
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        assert_eq!(
            s.diagview()
                .unwrap()
                .iter()
                .map(|e| e.values.clone())
                .collect::<Vec<_>>(),
            vec![vec![4.0, 1.0, 0.0], vec![3.0, 2.0]]
        );
        let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert!(rebuilt
            .dense_data()
            .unwrap()
            .iter()
            .zip(input)
            .all(|(a, b)| (a - b).abs() < 1e-12));
        assert_eq!(source.diagview().unwrap()[0].values, vec![1.0, -4.0, 0.0]);
    }
}

#[test]
fn compact_diagonal_svd_vals_uses_only_the_stored_spectrum() {
    let solver_calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(fail_second_svd(&solver_calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![-2.0, 3.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![-4.0, 0.0, 1.0],
            },
        ],
    )
    .unwrap();
    let before = input.diagview().unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        input.svd_vals(&[0], &[1]).unwrap(),
        vec![
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![4.0, 1.0, 0.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![3.0, 2.0]
            },
        ]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(input.diagview().unwrap(), before);

    let dual_bond = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(1), 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &dual_bond,
        [SectorSpectrum {
            sector: U1Irrep::new(-1),
            values: vec![-2.0, 1.0],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(dual.svd_vals(&[0], &[1]).unwrap()[0].values, [2.0, 1.0]);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(solver_calls.total(), 0);
}

#[test]
fn compact_diagonal_svd_vals_rounds_at_payload_precision() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let real: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![-1.000_000_1, 0.0],
        }],
    )
    .unwrap();
    assert_eq!(
        real.svd_vals(&[0], &[1]).unwrap()[0].values,
        [1.000_000_1_f32 as f64, 0.0]
    );

    let complex: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                num_complex::Complex32::new(1.0, 1.0),
                num_complex::Complex32::new(-3.0, 4.0),
            ],
        }],
    )
    .unwrap();
    assert_eq!(
        complex.svd_vals(&[0], &[1]).unwrap()[0].values,
        [5.0, (2.0_f64.sqrt() as f32) as f64]
    );
}

#[test]
fn compact_diagonal_svd_vals_refuses_nonfinite_and_overflows_at_range_edges() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    for value in [
        num_complex::Complex32::new(f32::NAN, 0.0),
        num_complex::Complex32::new(f32::INFINITY, 0.0),
        num_complex::Complex32::new(f32::MAX * 0.75, f32::MAX * 0.75),
    ] {
        let input: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![value],
            }],
        )
        .unwrap();
        // A nonfinite value is refused by the shared finite-input stage; a
        // finite one is MAK `svd_vals!(::DiagonalAlgorithm)`, `abs(a)`:
        // `|0.75 MAX (1 + i)| = 1.06 MAX` overflows f32 to Inf.
        DIAGONAL_MATERIALIZATIONS.set(0);
        let result = input.svd_vals(&[0], &[1]);
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        if value.re.is_finite() {
            assert_eq!(result.unwrap()[0].values[0], f64::INFINITY);
        } else {
            let error = result.unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("svd input components must be finite"),
                "{error}"
            );
        }
    }
}

#[test]
fn compact_diagonal_svd_complex_permutation_and_phase() {
    for full in [false, true] {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
        )
        .unwrap();
        let z = Complex64::new;
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: U1Irrep::new(1),
                    values: vec![z(-2.0, 0.0), z(4.0, 0.0)],
                },
                SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![z(0.0, 0.0), z(0.0, -3.0), z(1.0, 1.0)],
                },
            ],
        )
        .unwrap();
        DIAGONAL_MATERIALIZATIONS.set(0);
        let Svd { u, s, vh } = (if full {
            input.svd_full(&[0], &[1])
        } else {
            input.svd_compact(&[0], &[1])
        })
        .unwrap();
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        let root2 = 2.0_f64.sqrt();
        assert_eq!(u.dense_data().unwrap().len(), 13);
        assert_eq!(vh.dense_data().unwrap().len(), 13);
        for (actual, expected) in u.dense_data().unwrap().iter().zip([
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
        ]) {
            assert!((*actual - expected).norm() < 1e-12);
        }
        for (actual, expected) in vh.dense_data().unwrap().iter().zip([
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(0.0, -1.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0 / root2, 1.0 / root2),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(-1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
        ]) {
            assert!((*actual - expected).norm() < 1e-12);
        }
        assert_eq!(
            s.diagview()
                .unwrap()
                .iter()
                .map(|e| e.values.clone())
                .collect::<Vec<_>>(),
            vec![
                vec![z(3.0, 0.0), z(root2, 0.0), z(0.0, 0.0)],
                vec![z(4.0, 0.0), z(2.0, 0.0)]
            ]
        );
        let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert!(rebuilt
            .dense_data()
            .unwrap()
            .iter()
            .zip(input.materialize().unwrap().dense_data().unwrap())
            .all(|(a, b)| (*a - *b).norm() < 1e-12));
    }
}

#[test]
fn compact_diagonal_svd_other_host_scalars_and_sectors() {
    for full in [false, true] {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        macro_rules! check {
            ($rule:expr, $legs:expr, $spectra:expr, $dtype:ty, $tol:expr) => {{
                let leg = GradedSpace::try_new(Arc::new($rule), $legs).unwrap();
                let input: TensorMap<_, $dtype> =
                    TensorMap::diagonal(&runtime, &leg, $spectra).unwrap();
                let before = input.materialize().unwrap().dense_data().unwrap().to_vec();
                let saved = input.diagview().unwrap();
                let dense_values = input.materialize().unwrap().svd_vals(&[0], &[1]).unwrap();
                DIAGONAL_MATERIALIZATIONS.set(0);
                let compact_values = input.svd_vals(&[0], &[1]).unwrap();
                assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
                assert_eq!(compact_values.len(), dense_values.len());
                for (actual, expected) in compact_values.iter().zip(&dense_values) {
                    assert_eq!(actual.sector, expected.sector);
                    assert_eq!(actual.values.len(), expected.values.len());
                    for (&a, &b) in actual.values.iter().zip(&expected.values) {
                        assert!((a - b).abs() <= $tol * b.max(1.0));
                    }
                }
                DIAGONAL_MATERIALIZATIONS.set(0);
                let Svd { u, s, vh } = (if full {
                    input.svd_full(&[0], &[1])
                } else {
                    input.svd_compact(&[0], &[1])
                })
                .unwrap();
                assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
                let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
                assert!(rebuilt
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(before)
                    .all(|(a, b)| FactorScalar::widen_complex(*a - b).norm() <= $tol));
                assert_eq!(input.diagview().unwrap(), saved);
            }};
        }
        check!(
            U1FusionRule,
            [(U1Irrep::new(0), 3)],
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![-0.0_f32, -3.0, 1.0]
            }],
            f32,
            1e-5
        );
        check!(
            U1FusionRule,
            [(U1Irrep::new(0), 2)],
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    num_complex::Complex32::new(1.0, 2.0),
                    num_complex::Complex32::new(-3.0, 1.0)
                ]
            }],
            num_complex::Complex32,
            1e-5
        );
        let spin0 = SU2Irrep::from_twice_spin(0);
        let spin_half = SU2Irrep::from_twice_spin(1);
        check!(
            SU2FusionRule,
            [(spin0, 2), (spin_half, 1)],
            [
                SectorSpectrum {
                    sector: spin_half,
                    values: vec![-4.0_f64]
                },
                SectorSpectrum {
                    sector: spin0,
                    values: vec![2.0, -3.0]
                },
            ],
            f64,
            1e-12
        );
        let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
        let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
        check!(
            U1FusionRule.product(FermionParityFusionRule),
            [(even, 2), (odd, 1)],
            [
                SectorSpectrum {
                    sector: odd,
                    values: vec![-4.0_f64]
                },
                SectorSpectrum {
                    sector: even,
                    values: vec![2.0, -3.0]
                },
            ],
            f64,
            1e-12
        );
        check!(
            U1FusionRule,
            [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
            [SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0_f64, -3.0]
            }],
            f64,
            1e-12
        );
        let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
        let input: TensorMap<_, f64> =
            TensorMap::diagonal(&runtime, &empty, Vec::<SectorSpectrum<_, f64>>::new()).unwrap();
        DIAGONAL_MATERIALIZATIONS.set(0);
        assert!(input.svd_vals(&[0], &[1]).unwrap().is_empty());
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        let Svd { u, s, vh } = (if full {
            input.svd_full(&[0], &[1])
        } else {
            input.svd_compact(&[0], &[1])
        })
        .unwrap();
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        assert_eq!(
            u.dense_data().unwrap().len() + vh.dense_data().unwrap().len(),
            0
        );
        assert!(s.diagview().unwrap().is_empty());
    }
}

#[test]
fn compact_diagonal_svd_at_c32_range_edge_overflows_like_mak() {
    for full in [false, true] {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
        let large = f32::MAX * 0.75;
        let input: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![num_complex::Complex32::new(large, large)],
            }],
        )
        .unwrap();
        // MAK `svd_full!(::DiagonalAlgorithm)`: `S = abs(a)` overflows f32
        // to Inf (`1.06 MAX`), and `Vh` keeps the finite unit phase.
        let Svd { s, vh, .. } = if full {
            input.svd_full(&[0], &[1])
        } else {
            input.svd_compact(&[0], &[1])
        }
        .unwrap();
        assert_eq!(
            s.diagview().unwrap()[0].values,
            [num_complex::Complex32::new(f32::INFINITY, 0.0)]
        );
        let phase = vh.dense_data().unwrap()[0];
        let half = std::f32::consts::FRAC_1_SQRT_2;
        assert!((phase - num_complex::Complex32::new(half, half)).norm() <= 4.0 * f32::EPSILON);
    }
}

#[test]
fn compact_diagonal_svd_subnormal_complex_phase_is_unit() {
    for full in [false, true] {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
        let tiny = f64::from_bits(1);
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![Complex64::new(tiny, tiny)],
            }],
        )
        .unwrap();
        let Svd { u, s, vh } = (if full {
            input.svd_full(&[0], &[1])
        } else {
            input.svd_compact(&[0], &[1])
        })
        .unwrap();
        assert!((vh.dense_data().unwrap()[0].norm() - 1.0).abs() < 1e-12);
        assert!((u.dense_data().unwrap()[0].norm() - 1.0).abs() < 1e-12);
        let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert_eq!(rebuilt.dense_data().unwrap()[0], Complex64::new(tiny, tiny));
    }
}
