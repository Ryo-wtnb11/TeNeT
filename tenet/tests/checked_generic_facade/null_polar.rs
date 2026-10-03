use super::*;

fn multiply_2x2<D: tenet::typed::TensorScalar>(
    left: [[D; 2]; 2],
    right: [[D; 2]; 2],
) -> [[D; 2]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|col| left[row][0] * right[0][col] + left[row][1] * right[1][col])
    })
}

#[test]
fn checked_generic_polar_matches_independent_real_and_complex_qh_oracles() {
    // What: both factor orders retain the source authority and match Q/H,
    // including conjugate-transpose arithmetic for a genuinely complex Q.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();

    let h = [[2.0, 0.5], [0.5, 3.0]];
    let q = [[0.0, -1.0], [1.0, 0.0]];
    let left_data = multiply_2x2(q, h);
    let right_data = multiply_2x2(h, q);
    let q_tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| q[ij[0]][ij[1]]).unwrap();
    let h_tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| h[ij[0]][ij[1]]).unwrap();
    let left: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| left_data[ij[0]][ij[1]])
            .unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| right_data[ij[0]][ij[1]])
            .unwrap();
    let LeftPolar {
        w: actual_q,
        p: actual_h,
    } = left.left_polar(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(actual_q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(actual_h.provider(), provider.as_ref()));
    assert!(tenet::typed::__network::runtime_identity(actual_q.runtime()).matches(left.runtime()));
    assert_same_checked_generic_layout_and_close(&actual_q, &q_tensor, |a, b| (a - b).abs());
    assert_same_checked_generic_layout_and_close(&actual_h, &h_tensor, |a, b| (a - b).abs());
    let RightPolar {
        p: actual_h,
        wh: actual_q,
    } = right.right_polar(&[0], &[1]).unwrap();
    assert_same_checked_generic_layout_and_close(&actual_q, &q_tensor, |a, b| (a - b).abs());
    assert_same_checked_generic_layout_and_close(&actual_h, &h_tensor, |a, b| (a - b).abs());

    let s = std::f64::consts::FRAC_1_SQRT_2;
    let cq = [
        [Complex64::new(s, 0.0), Complex64::new(0.0, s)],
        [Complex64::new(0.0, s), Complex64::new(s, 0.0)],
    ];
    let ch = [
        [Complex64::new(2.0, 0.0), Complex64::new(0.25, 0.5)],
        [Complex64::new(0.25, -0.5), Complex64::new(3.0, 0.0)],
    ];
    let complex_q: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| cq[ij[0]][ij[1]]).unwrap();
    let complex_h: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| ch[ij[0]][ij[1]]).unwrap();
    for (source, left) in [(multiply_2x2(cq, ch), true), (multiply_2x2(ch, cq), false)] {
        let source: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| source[ij[0]][ij[1]])
                .unwrap();
        let (actual_q, actual_h) = if left {
            let LeftPolar { w, p } = source.left_polar(&[0], &[1]).unwrap();
            (w, p)
        } else {
            let RightPolar { p, wh } = source.right_polar(&[0], &[1]).unwrap();
            (wh, p)
        };
        assert_same_checked_generic_layout_and_close(&actual_q, &complex_q, |a, b| (a - b).norm());
        assert_same_checked_generic_layout_and_close(&actual_h, &complex_h, |a, b| (a - b).norm());
    }
}

#[test]
fn checked_generic_polar_direction_covers_rectangular_side_only_and_empty_inputs() {
    // What: complete dimensions, not populated blocks, select the requested
    // tall/wide contract before any dense factorization.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let tall = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&tall], [&wide], |_, ij| {
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]][ij[0]][ij[1]]
        })
        .unwrap();
    assert!(source.left_polar(&[0], &[1]).is_ok());
    assert!(matches!(
        source.right_polar(&[0], &[1]),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Operation(_)
        ))
    ));
    let codomain_only =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)]).unwrap();
    let domain_only = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let side_only: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&codomain_only], [&domain_only], |_, _| 0.0)
            .unwrap();
    assert!(side_only.left_polar(&[0], &[1]).is_ok());
    assert!(side_only.right_polar(&[0], &[1]).is_err());
    let empty = GradedSpace::try_new(Arc::clone(&provider), []).unwrap();
    let empty_map: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&empty], [&empty]).unwrap();
    assert!(empty_map
        .left_polar(&[0], &[1])
        .unwrap()
        .w
        .dense_data()
        .unwrap()
        .is_empty());
    assert!(empty_map
        .right_polar(&[0], &[1])
        .unwrap()
        .p
        .dense_data()
        .unwrap()
        .is_empty());
}

#[test]
fn checked_generic_polar_lazy_redirects_to_the_opposite_parent_operation() {
    // What: lazy adjoints reuse the parent's owned P, return owned W-adjoints,
    // and report the operation requested on the receiver.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
            [
                [Complex64::new(2.0, 0.0), Complex64::new(0.5, 0.75)],
                [Complex64::new(-0.25, 0.5), Complex64::new(3.0, -0.25)],
            ][ij[0]][ij[1]]
        })
        .unwrap();
    let lazy = source.adjoint().unwrap();
    let RightPolar {
        p: parent_p,
        wh: parent_w,
    } = source.right_polar(&[0], &[1]).unwrap();
    let LeftPolar {
        w: actual_w,
        p: actual_p,
    } = lazy.left_polar(&[0], &[1]).unwrap();
    assert_same_checked_generic_layout_and_close(
        &actual_w,
        &parent_w.adjoint().unwrap(),
        |a, b| (a - b).norm(),
    );
    assert_same_checked_generic_layout_and_close(&actual_p, &parent_p, |a, b| (a - b).norm());
    let LeftPolar {
        w: parent_w,
        p: parent_p,
    } = source.left_polar(&[0], &[1]).unwrap();
    let RightPolar {
        p: actual_p,
        wh: actual_w,
    } = lazy.right_polar(&[0], &[1]).unwrap();
    assert_same_checked_generic_layout_and_close(&actual_p, &parent_p, |a, b| (a - b).norm());
    assert_same_checked_generic_layout_and_close(
        &actual_w,
        &parent_w.adjoint().unwrap(),
        |a, b| (a - b).norm(),
    );
    assert!(std::ptr::eq(actual_w.provider(), provider.as_ref()));
    assert!(
        tenet::typed::__network::runtime_identity(actual_w.runtime()).matches(source.runtime())
    );

    let tall = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let tall_source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&tall], [&narrow], |_, _| 1.0).unwrap();
    let error = tall_source
        .adjoint()
        .unwrap()
        .left_polar(&[0], &[1])
        .unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Operation(error))
            if matches!(
                error,
                tenet::typed::OperationError::InvalidArgument { message }
                    if message == "left_polar requires rows >= columns in every coupled-sector matrix"
            )
    ));
}

#[test]
fn checked_generic_polar_completes_rank_deficient_and_zero_sectors() {
    // What: compact-full SVD completion returns a full isometry/coisometry;
    // P remains Hermitian PSD for rank-deficient and zero matrices.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    for zero in [false, true] {
        let source: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
                if zero {
                    Complex64::new(0.0, 0.0)
                } else {
                    let row = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -0.5)];
                    let col = [Complex64::new(0.5, -1.0), Complex64::new(-1.5, 2.0)];
                    row[ij[0]] * col[ij[1]].conj()
                }
            })
            .unwrap();
        for left in [true, false] {
            let (w, p) = if left {
                let LeftPolar { w, p } = source.left_polar(&[0], &[1]).unwrap();
                (w, p)
            } else {
                let RightPolar { p, wh } = source.right_polar(&[0], &[1]).unwrap();
                (wh, p)
            };
            let rebuilt = if left {
                w.compose(&p).unwrap()
            } else {
                p.compose(&w).unwrap()
            };
            for (&actual, &expected) in rebuilt
                .dense_data()
                .unwrap()
                .iter()
                .zip(source.dense_data().unwrap())
            {
                assert!((actual - expected).norm() < 1e-9);
            }
            for col in 0..2 {
                for row in 0..2 {
                    let gram = (0..2).fold(Complex64::new(0.0, 0.0), |sum, inner| {
                        if left {
                            sum + w.dense_data().unwrap()[inner + 2 * row].conj()
                                * w.dense_data().unwrap()[inner + 2 * col]
                        } else {
                            sum + w.dense_data().unwrap()[row + 2 * inner]
                                * w.dense_data().unwrap()[col + 2 * inner].conj()
                        }
                    });
                    assert!((gram - Complex64::new(f64::from(row == col), 0.0)).norm() < 1e-10);
                    assert!(
                        (p.dense_data().unwrap()[row + 2 * col]
                            - p.dense_data().unwrap()[col + 2 * row].conj())
                        .norm()
                            < 1e-10
                    );
                }
            }
            assert!(p
                .eigh_vals(&[0], &[1])
                .unwrap()
                .iter()
                .flat_map(|entry| &entry.values)
                .all(|&value| value >= -1e-10));
        }
    }
}

#[test]
fn checked_generic_polar_stages_svd_and_both_gemms_without_publication() {
    // What: every SVD completes before W/P GEMMs, and either GEMM failure
    // returns no factor while preserving the source and provider authority.
    for (fail_svd, fail_gemm, expected_svd, expected_gemm) in [
        (None, None, 2, 4),
        (Some(1), None, 1, 0),
        (Some(2), None, 2, 0),
        (None, Some(1), 2, 1),
        (None, Some(2), 2, 2),
        (None, Some(4), 2, 4),
    ] {
        let svd_calls = Arc::new(SpyCounts::default());
        let gemm_calls = Arc::clone(&svd_calls);
        let runtime = Runtime::builder()
            .dense_threads(1)
            .with_dense_executor(Box::new(pinv_spy(&svd_calls, fail_svd, fail_gemm)))
            .build()
            .unwrap();
        let provider = Arc::new(CheckedOnlyToy::new(0));
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)])
            .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, _| {
                if trees.coupled() == &Label::Vacuum {
                    2.0
                } else {
                    3.0
                }
            })
            .unwrap();
        let before = source.dense_data().unwrap().to_vec();
        let result = source.left_polar(&[0], &[1]);
        if fail_svd.is_some() || fail_gemm.is_some() {
            assert!(matches!(
                result,
                Err(GenericTensorError::Plan(
                    tenet::typed::CheckedGenericPlanError::Operation(_)
                ))
            ));
        } else {
            let LeftPolar { w, p } = result.unwrap();
            assert!(std::ptr::eq(w.provider(), provider.as_ref()));
            assert!(std::ptr::eq(p.provider(), provider.as_ref()));
        }
        assert_eq!(svd_calls.of(PINV_SVD), expected_svd);
        assert_eq!(gemm_calls.of(Kernel::GEMM), expected_gemm);
        assert_eq!(svd_calls.total(), expected_svd + expected_gemm);
        assert_eq!(source.dense_data().unwrap(), before.as_slice());
    }
}

#[test]
fn checked_only_compact_diagonal_polar_is_direct_and_keeps_dense_fallback() {
    let svd_calls = Arc::new(SpyCounts::default());
    let gemm_calls = Arc::clone(&svd_calls);
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(pinv_spy(&svd_calls, None, None)))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 1)]).unwrap();
    let source: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::X,
                values: vec![Complex64::new(0.0, -2.0)],
            },
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
            },
        ],
    )
    .unwrap();
    for left in [true, false] {
        let (w, p) = if left {
            let LeftPolar { w, p } = source.left_polar(&[0], &[1]).unwrap();
            (w, p)
        } else {
            let RightPolar { p, wh } = source.right_polar(&[0], &[1]).unwrap();
            (wh, p)
        };
        assert_eq!(svd_calls.of(PINV_SVD), 0);
        assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
        assert_eq!(gemm_calls.total(), 0);
        assert!(std::ptr::eq(w.provider(), provider.as_ref()));
        assert!(std::ptr::eq(p.provider(), provider.as_ref()));
        assert_eq!(w.codomain(), source.codomain());
        assert_eq!(w.domain(), source.domain());
        assert!(w.dense_data().is_err());
        assert!(p.dense_data().is_err());
        let rebuilt = if left { w.compose(&p) } else { p.compose(&w) }.unwrap();
        assert_same_checked_generic_layout_and_close(&rebuilt, &source, |a, b| (a - b).norm());
    }

    let check_real = |source: &TensorMap<CheckedOnlyToy, f64>,
                      expected_phase: &[SectorSpectrum<Label, f64>],
                      expected_magnitude: &[SectorSpectrum<Label, f64>]| {
        for left in [true, false] {
            let (w, p) = if left {
                let LeftPolar { w, p } = source.left_polar(&[0], &[1]).unwrap();
                (w, p)
            } else {
                let RightPolar { p, wh } = source.right_polar(&[0], &[1]).unwrap();
                (wh, p)
            };
            assert!(std::ptr::eq(w.provider(), provider.as_ref()));
            assert!(std::ptr::eq(p.provider(), provider.as_ref()));
            assert_eq!(w.codomain(), source.codomain());
            assert_eq!(w.domain(), source.domain());
            assert_eq!(p.codomain(), source.domain());
            assert_eq!(p.domain(), source.domain());
            assert!(w.dense_data().is_err());
            assert!(p.dense_data().is_err());
            assert_eq!(w.diagview().unwrap(), expected_phase);
            assert_eq!(p.diagview().unwrap(), expected_magnitude);
            let rebuilt = if left { w.compose(&p) } else { p.compose(&w) }.unwrap();
            assert_same_checked_generic_layout_and_close(&rebuilt, source, |a, b| (a - b).abs());
        }
    };
    let real: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::X,
                values: vec![3.0],
            },
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![-2.0, 0.0],
            },
        ],
    )
    .unwrap();
    check_real(
        &real,
        &[
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![-1.0, 1.0],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![1.0],
            },
        ],
        &[
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![2.0, 0.0],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![3.0],
            },
        ],
    );
    let dual_bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual_sector = dual_bond.sectors().unwrap().remove(0);
    let dual: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &dual_bond,
        [SectorSpectrum {
            sector: dual_sector,
            values: vec![-4.0, 0.0],
        }],
    )
    .unwrap();
    check_real(
        &dual,
        &[SectorSpectrum {
            sector: dual_sector,
            values: vec![-1.0, 1.0],
        }],
        &[SectorSpectrum {
            sector: dual_sector,
            values: vec![4.0, 0.0],
        }],
    );
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    assert_eq!(gemm_calls.total(), 0);

    for bad_value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        let bad: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &bond,
            [
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![bad_value, Complex64::new(1.0, 0.0)],
                },
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![Complex64::new(2.0, 0.0)],
                },
            ],
        )
        .unwrap();
        svd_calls.reset();
        reset_provider_queries(&provider);
        let compact_error = bad.left_polar(&[0], &[1]).unwrap_err();
        let compact_queries = provider.queries_since_reset.load(Ordering::Relaxed);
        assert!(svd_calls.of(PINV_SVD) > 0);
        let dense = bad.materialize().unwrap();
        reset_provider_queries(&provider);
        let dense_error = dense.left_polar(&[0], &[1]).unwrap_err();
        assert_eq!(
            provider.queries_since_reset.load(Ordering::Relaxed),
            compact_queries
        );
        assert_eq!(compact_error.to_string(), dense_error.to_string());
    }
}

#[test]
fn checked_generic_lazy_polar_second_svd_failure_keeps_parent_unchanged() {
    for left in [true, false] {
        let svd_calls = Arc::new(SpyCounts::default());
        let gemm_calls = Arc::clone(&svd_calls);
        let runtime = Runtime::builder()
            .dense_threads(1)
            .with_dense_executor(Box::new(pinv_spy(&svd_calls, Some(2), None)))
            .build()
            .unwrap();
        let provider = Arc::new(CheckedOnlyToy::new(0));
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)])
            .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, _| {
                if trees.coupled() == &Label::Vacuum {
                    2.0
                } else {
                    3.0
                }
            })
            .unwrap();
        let before = source.dense_data().unwrap().to_vec();
        let lazy = source.adjoint().unwrap();
        let result = if left {
            lazy.left_polar(&[0], &[1]).map(drop)
        } else {
            lazy.right_polar(&[0], &[1]).map(drop)
        };

        assert!(matches!(
            result,
            Err(GenericTensorError::Plan(
                tenet::typed::CheckedGenericPlanError::Operation(_)
            ))
        ));
        assert_eq!(svd_calls.of(PINV_SVD), 2);
        assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
        assert_eq!(gemm_calls.total(), 2);
        assert_eq!(source.dense_data().unwrap(), before.as_slice());
    }
}

#[test]
fn checked_generic_polar_provider_error_precedes_dense_work() {
    // What: complete codomain dimensions are queried before domain,
    // direction, admission, and dense work.
    let svd_calls = Arc::new(SpyCounts::default());
    let gemm_calls = Arc::clone(&svd_calls);
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(pinv_spy(&svd_calls, None, None)))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    let refused_compact: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![f64::NAN, 1.0],
        }],
    )
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(matches!(
        source.left_polar(&[0], &[1]),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    assert_eq!(gemm_calls.total(), 0);
    assert_eq!(source.dense_data().unwrap(), before);
    assert!(matches!(
        refused_compact.left_polar(&[0], &[1]),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    assert_eq!(gemm_calls.total(), 0);
}

#[test]
fn checked_generic_null_spaces_cover_rank_cutoff_zero_disjoint_and_side_only_sectors() {
    // What: Generic null spaces use the documented numerical rank, keep full
    // zero/side-only directions, drop full-rank sectors, and retain authority.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let tolerance = f64::EPSILON * 2.0;
    for (small, nullity) in [(0.5 * tolerance, 1), (2.0 * tolerance, 0)] {
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&x], [&x], |_, index| match index {
                [0, 0] => 1.0,
                [1, 1] => small,
                _ => 0.0,
            })
            .unwrap();
        for null in [
            source.left_null(&[0], &[1]).unwrap(),
            source.right_null(&[0], &[1]).unwrap(),
        ] {
            assert!(std::ptr::eq(null.provider(), provider.as_ref()));
            if nullity == 0 {
                assert!(null.dense_data().unwrap().is_empty());
            } else {
                assert_eq!(null.dense_data().unwrap().len(), 2);
            }
        }
    }

    let zero: TensorMap<_, Complex64> = TensorMap::zeros(&runtime, [&x], [&x]).unwrap();
    assert_eq!(
        zero.left_null(&[0], &[1])
            .unwrap()
            .dense_data()
            .unwrap()
            .len(),
        4
    );
    assert_eq!(
        zero.right_null(&[0], &[1])
            .unwrap()
            .dense_data()
            .unwrap()
            .len(),
        4
    );

    let vacuum = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 3)]).unwrap();
    let disjoint: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&x], [&vacuum]).unwrap();
    let left = disjoint.left_null(&[0], &[1]).unwrap();
    let right = disjoint.right_null(&[0], &[1]).unwrap();
    assert_eq!(left.dense_data().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(right.dense_data().unwrap().len(), 9);
    for column in 0..3 {
        for row in 0..3 {
            assert_eq!(
                right.dense_data().unwrap()[row + 3 * column],
                f64::from(row == column)
            );
        }
    }

    let shared =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();
    let unit = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let codomain_side: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&shared], [&unit], |trees, index| {
            f64::from(trees.coupled() == &Label::Vacuum && index[0] == index[1])
        })
        .unwrap();
    assert_eq!(
        codomain_side
            .left_null(&[0], &[1])
            .unwrap()
            .dense_data()
            .unwrap()
            .len(),
        4
    );
    assert!(codomain_side
        .right_null(&[0], &[1])
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
}

#[test]
fn checked_generic_null_dense_failure_is_typed_and_nonpublishing() {
    // What: a later sector SVD failure crosses the public Generic facade as a
    // typed plan error without changing the source or returning a partial null.
    let svd_calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(pinv_spy(&svd_calls, Some(2), None)))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    assert!(matches!(
        source.left_null(&[0], &[1]),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Operation(_)
        ))
    ));
    assert_eq!(svd_calls.of(PINV_SVD), 2);
    assert_eq!(source.dense_data().unwrap(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[test]
fn checked_compact_null_fallback_reuses_the_admission_dimension_query() {
    let svd_calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(pinv_spy(&svd_calls, None, None)))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![1.0, 4.0 * f64::EPSILON],
        }],
    )
    .unwrap();
    let dense = source.materialize().unwrap();

    reset_provider_queries(&provider);
    source.left_null(&[0], &[1]).unwrap();
    let compact_queries = provider.queries_since_reset.load(Ordering::Relaxed);
    assert_eq!(svd_calls.of(PINV_SVD), 1);

    reset_provider_queries(&provider);
    svd_calls.reset();
    dense.left_null(&[0], &[1]).unwrap();
    assert_eq!(
        provider.queries_since_reset.load(Ordering::Relaxed),
        compact_queries
    );
    assert_eq!(svd_calls.of(PINV_SVD), 1);
}

// Why not fewer args: each parameter is an independent fixture input for one
// assertion helper shared by several SU(N) null-projector tests; bundling
// them would add a struct with a single call-site shape.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_null_projectors<D>(
    n: usize,
    label: Vec<i64>,
    u: [D; 2],
    v: [D; 2],
    u_norm_squared: f64,
    v_norm_squared: f64,
    adjoint: impl Fn(D) -> D,
    close: impl Fn(D, D) -> f64,
) where
    D: tenet::typed::FactorizationScalar + fmt::Debug + PartialEq + numerics::Numeric,
{
    use tenet::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label, 2)]).unwrap();
    let source: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row == column {
                u[trees.codomain_vertices()[0].get() - 1]
                    * adjoint(v[trees.domain_vertices()[0].get() - 1])
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap();
    assert!((0..source.subblock_count()).any(|index| {
        let trees = source.subblock_fusion_trees(index).unwrap();
        trees.codomain_vertices()[0].get() == 2
            && trees.domain_vertices()[0].get() == 1
            && source.dense_data().unwrap()[source.subblock(index).unwrap().offset()]
                != D::from_real(0.0)
    }));
    let outer_multiplicity_sectors = (0..source.subblock_count())
        .filter_map(|index| {
            let trees = source.subblock_fusion_trees(index).unwrap();
            trees
                .codomain_vertices()
                .iter()
                .chain(trees.domain_vertices())
                .any(|vertex| vertex.get() > 1)
                .then(|| trees.coupled().clone())
        })
        .collect::<Vec<_>>();

    let left = source.left_null(&[0, 1], &[2, 3]).unwrap();
    let right = source.right_null(&[0, 1], &[2, 3]).unwrap();
    assert!(std::ptr::eq(left.provider(), provider.as_ref()));
    assert!(std::ptr::eq(right.provider(), provider.as_ref()));
    let left_adjoint = left.adjoint().unwrap();
    let left_adjoint = left_adjoint
        .axpby(D::from_real(1.0), &left_adjoint, D::from_real(0.0))
        .unwrap();
    let right_adjoint = right.adjoint().unwrap();
    let right_adjoint = right_adjoint
        .axpby(D::from_real(1.0), &right_adjoint, D::from_real(0.0))
        .unwrap();
    let left_projector = left.compose(&left_adjoint).unwrap();
    let right_projector = right_adjoint.compose(&right).unwrap();
    let expected_left: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row != column || !outer_multiplicity_sectors.contains(trees.coupled()) {
                return D::from_real(0.0);
            }
            let i = trees.codomain_vertices()[0].get() - 1;
            let j = trees.domain_vertices()[0].get() - 1;
            D::from_real(f64::from(i == j))
                + D::from_real(-1.0 / u_norm_squared) * u[i] * adjoint(u[j])
        })
        .unwrap();
    let expected_right: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row != column || !outer_multiplicity_sectors.contains(trees.coupled()) {
                return D::from_real(0.0);
            }
            let i = trees.codomain_vertices()[0].get() - 1;
            let j = trees.domain_vertices()[0].get() - 1;
            D::from_real(f64::from(i == j))
                + D::from_real(-1.0 / v_norm_squared) * v[i] * adjoint(v[j])
        })
        .unwrap();
    for (name, actual, expected) in [
        ("left", &left_projector, &expected_left),
        ("right", &right_projector, &expected_right),
    ] {
        assert_eq!(actual.subblock_count(), expected.subblock_count());
        for index in 0..actual.subblock_count() {
            let actual = actual.subblock(index).unwrap();
            let expected = expected.subblock(index).unwrap();
            assert_eq!(actual.key(), expected.key());
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.strides(), expected.strides());
        }
        for (index, (&actual, &expected)) in actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .enumerate()
        {
            assert!(
                close(actual, expected) < 1e-9,
                "{name} projector mismatch at raw {index}: {actual:?} != {expected:?}"
            );
        }
    }

    for value in left_adjoint.compose(&source).unwrap().dense_data().unwrap() {
        assert!(close(*value, D::from_real(0.0)) < 1e-9);
    }
    for value in source
        .compose(&right_adjoint)
        .unwrap()
        .dense_data()
        .unwrap()
    {
        assert!(close(*value, D::from_real(0.0)) < 1e-9);
    }
    for gram in [
        left_adjoint.compose(&left).unwrap(),
        right.compose(&right_adjoint).unwrap(),
    ] {
        for block_index in 0..gram.subblock_count() {
            let block = gram.subblock(block_index).unwrap();
            for column in 0..block.shape()[1] {
                for row in 0..block.shape()[0] {
                    let expected = D::from_real(f64::from(row == column));
                    let actual = gram.dense_data().unwrap()
                        [block.offset() + row * block.strides()[0] + column * block.strides()[1]];
                    assert!(close(actual, expected) < 1e-9);
                }
            }
        }
    }

    let lazy = source.adjoint().unwrap();
    let lazy_left = lazy.left_null(&[0, 1], &[2, 3]).unwrap();
    let expected_lazy_left = right_adjoint;
    assert!(std::ptr::eq(lazy_left.provider(), provider.as_ref()));
    // Path agreement is the contract: the lazy route is defined as the
    // adjoint of the parent's right null space, gauge included.
    numerics::assert_slices_close(
        "left_null of the lazy adjoint",
        lazy_left.dense_data().unwrap(),
        expected_lazy_left.dense_data().unwrap(),
        endomorphism_terms(source.dense_data().unwrap().len()),
    );
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_null_projectors_resolve_cross_mu_for_both_dtypes() {
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_null_projectors::<f64>(
            n,
            label.clone(),
            [1.0, 2.0],
            [1.0, -1.0],
            5.0,
            2.0,
            |value| value,
            |actual, expected| (actual - expected).abs(),
        );
        assert_sun_checked_generic_null_projectors::<Complex64>(
            n,
            label,
            [Complex64::new(1.0, 1.0), Complex64::new(2.0, -0.5)],
            [Complex64::new(0.5, -1.0), Complex64::new(-1.5, 2.0)],
            6.25,
            7.5,
            |value| value.conj(),
            |actual, expected| (actual - expected).norm(),
        );
    }
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_polar_qh<D>(
    n: usize,
    label: Vec<i64>,
    q: [[D; 2]; 2],
    h: [[D; 2]; 2],
    close: impl Fn(D, D) -> f64 + Copy,
) where
    D: tenet::typed::FactorizationScalar + fmt::Debug + PartialEq,
{
    use tenet::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label.clone(), 2)]).unwrap();
    let build = |matrix: [[D; 2]; 2], fallback: D| {
        let cross_sector = label.clone();
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], move |trees, index| {
            let row = index[0] + 2 * index[1];
            let col = index[2] + 2 * index[3];
            if row != col {
                return D::from_real(0.0);
            }
            if trees.coupled() == &cross_sector {
                matrix[trees.codomain_vertices()[0].get() - 1][trees.domain_vertices()[0].get() - 1]
            } else if trees.codomain_vertices() == trees.domain_vertices() {
                fallback
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap()
    };
    let expected_q = build(q, D::from_real(1.0));
    let expected_h = build(h, D::from_real(2.0));
    let mut saw_cross_mu = false;
    for (source_matrix, left) in [(multiply_2x2(q, h), true), (multiply_2x2(h, q), false)] {
        let source = build(source_matrix, D::from_real(2.0));
        saw_cross_mu |= (0..source.subblock_count()).any(|index| {
            let trees = source.subblock_fusion_trees(index).unwrap();
            trees.coupled() == &label
                && trees.codomain_vertices()[0].get() == 2
                && trees.domain_vertices()[0].get() == 1
        });
        let (actual_q, actual_h) = if left {
            let LeftPolar { w, p } = source.left_polar(&[0, 1], &[2, 3]).unwrap();
            (w, p)
        } else {
            let RightPolar { p, wh } = source.right_polar(&[0, 1], &[2, 3]).unwrap();
            (wh, p)
        };
        assert!(std::ptr::eq(actual_q.provider(), provider.as_ref()));
        assert!(std::ptr::eq(actual_h.provider(), provider.as_ref()));
        assert_eq!(actual_q.codomain(), source.codomain());
        assert_eq!(actual_q.domain(), source.domain());
        assert_same_checked_generic_layout_and_close(&actual_q, &expected_q, close);
        assert_same_checked_generic_layout_and_close(&actual_h, &expected_h, close);
        let rebuilt = if left {
            actual_q.compose(&actual_h).unwrap()
        } else {
            actual_h.compose(&actual_q).unwrap()
        };
        assert_same_checked_generic_layout_and_close(&rebuilt, &source, close);
    }
    assert!(
        saw_cross_mu,
        "SU(N) polar fixture must carry a cross-mu full key"
    );
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_polar_cross_mu_qh_oracles_for_both_dtypes() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_polar_qh::<f64>(
            n,
            label.clone(),
            [[0.0, -1.0], [1.0, 0.0]],
            [[2.0, 0.5], [0.5, 3.0]],
            |actual, expected| (actual - expected).abs(),
        );
        assert_sun_checked_generic_polar_qh::<Complex64>(
            n,
            label,
            [
                [Complex64::new(s, 0.0), Complex64::new(0.0, s)],
                [Complex64::new(0.0, s), Complex64::new(s, 0.0)],
            ],
            [
                [Complex64::new(2.0, 0.0), Complex64::new(0.25, 0.5)],
                [Complex64::new(0.25, -0.5), Complex64::new(3.0, 0.0)],
            ],
            |actual, expected| (actual - expected).norm(),
        );
    }
}
