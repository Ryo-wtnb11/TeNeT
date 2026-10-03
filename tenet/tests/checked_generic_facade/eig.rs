use super::*;

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_eigh<D>(
    n: usize,
    label: Vec<i64>,
    off_diagonal: D,
    adjoint: impl Fn(D) -> D + Copy,
    real: impl Fn(D) -> f64 + Copy,
    norm_squared: f64,
    close: impl Fn(D, D) -> f64 + Copy,
) where
    D: tenet::typed::FactorizationScalar + fmt::Debug,
{
    use std::cell::Cell;

    use tenet::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label.clone(), 2)]).unwrap();
    let cross_sector = label.clone();
    let source: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row != column {
                return D::from_real(0.0);
            }
            if trees.coupled() == &cross_sector {
                let mu_row = trees.codomain_vertices()[0].get();
                let mu_column = trees.domain_vertices()[0].get();
                match (mu_row, mu_column) {
                    (1, 1) => D::from_real(4.0 + 10.0 * row as f64),
                    (2, 2) => D::from_real(-1.0 + 10.0 * row as f64),
                    (1, 2) => off_diagonal,
                    (2, 1) => adjoint(off_diagonal),
                    _ => D::from_real(0.0),
                }
            } else if trees.codomain_vertices() == trees.domain_vertices() {
                D::from_real(50.0 + 10.0 * row as f64)
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap();
    assert!((0..source.subblock_count()).any(|index| {
        let trees = source.subblock_fusion_trees(index).unwrap();
        trees.coupled() == &label
            && trees.codomain_vertices()[0].get() == 1
            && trees.domain_vertices()[0].get() == 2
            && source.subblock(index).unwrap().shape() == [2, 2, 2, 2]
            && source.dense_data().unwrap()[source.subblock(index).unwrap().offset()]
                == off_diagonal
    }));

    let Eigh { d, v } = source.eigh_full(&[0, 1], &[2, 3]).unwrap();
    assert!(std::ptr::eq(d.provider(), provider.as_ref()));
    assert!(std::ptr::eq(v.provider(), provider.as_ref()));
    let dense_len = d.materialize().unwrap().dense_data().unwrap().len();
    assert!(!format!("{d:?}").contains(&format!("elements: {dense_len}")));
    assert_eq!(v.codomain(), source.codomain());
    assert_eq!(v.domain(), d.codomain());
    for (actual, expected) in source
        .compose(&v)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(v.compose(&d).unwrap().dense_data().unwrap())
    {
        assert!(close(*actual, *expected) < 1e-8);
    }

    let lazy_vh = v.adjoint().unwrap();
    let logical_vh = lazy_vh
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    let position = Cell::new(0usize);
    let codomain = v.domain();
    let domain = v.codomain();
    let vh: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, codomain.iter(), domain.iter(), |_, _| {
            let index = position.get();
            position.set(index + 1);
            logical_vh[index]
        })
        .unwrap();
    assert_same_checked_generic_layout_and_close(
        &v.compose(&d).unwrap().compose(&vh).unwrap(),
        &source,
        close,
    );

    let root = (6.25 + norm_squared).sqrt();
    let lambda_plus = 1.5 + root;
    let lambda_minus = 1.5 - root;
    let identity_codomain = d.codomain();
    let identity_domain = d.domain();
    let identity: TensorMap<_, D> = TensorMap::from_subblock_fn(
        &runtime,
        identity_codomain.iter(),
        identity_domain.iter(),
        |_, index| D::from_real(f64::from(index[0] == index[1])),
    )
    .unwrap();
    let mut selector = identity.clone();
    let mut skipped_target = false;
    for index in 0..d.subblock_count() {
        let block = d.subblock(index).unwrap();
        for diagonal in 0..block.shape()[0] {
            let value = d.materialize().unwrap().dense_data().unwrap()
                [block.offset() + diagonal * block.strides()[0] + diagonal * block.strides()[1]];
            let scalar = real(value);
            if (scalar - lambda_plus).abs() < 1e-8 {
                assert!(!skipped_target, "target eigenvalue must be nondegenerate");
                skipped_target = true;
                continue;
            }
            let shifted = d
                .axpby(D::from_real(1.0), &identity, D::from_real(-scalar))
                .unwrap()
                .scale(D::from_real(1.0 / (lambda_plus - scalar)));
            selector = selector.compose(&shifted).unwrap();
        }
    }
    assert!(skipped_target);
    let projector = v.compose(&selector).unwrap().compose(&vh).unwrap();
    let expected: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.coupled() != &label || row != 0 || column != 0 {
                return D::from_real(0.0);
            }
            let mu_row = trees.codomain_vertices()[0].get();
            let mu_column = trees.domain_vertices()[0].get();
            let inverse_denominator = D::from_real(1.0 / (lambda_plus - lambda_minus));
            match (mu_row, mu_column) {
                (1, 1) => D::from_real(4.0 - lambda_minus) * inverse_denominator,
                (2, 2) => D::from_real(-1.0 - lambda_minus) * inverse_denominator,
                (1, 2) => off_diagonal * inverse_denominator,
                (2, 1) => adjoint(off_diagonal) * inverse_denominator,
                _ => D::from_real(0.0),
            }
        })
        .unwrap();
    assert_same_checked_generic_layout_and_close(&projector, &expected, close);
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_eigh_cross_mu_projectors_for_both_dtypes() {
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_eigh(
            n,
            label.clone(),
            1.0,
            |value| value,
            |value| value,
            1.0,
            |actual, expected| (actual - expected).abs(),
        );
        assert_sun_checked_generic_eigh(
            n,
            label,
            Complex64::new(1.0, 1.0),
            |value| value.conj(),
            |value| value.re,
            2.0,
            |actual, expected| (actual - expected).norm(),
        );
    }
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_eig<D>(n: usize, label: Vec<i64>)
where
    D: SunEigInput,
{
    use tenet::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label.clone(), 2)]).unwrap();
    let source: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row != column {
                return D::from_real(0.0);
            }
            if trees.coupled() == &label {
                let mu_row = trees.codomain_vertices()[0].get();
                let mu_column = trees.domain_vertices()[0].get();
                let shift = 10.0 * row as f64;
                D::from_real(match (mu_row, mu_column) {
                    (1, 1) | (2, 2) => 1.0 + shift,
                    (1, 2) => -3.0,
                    (2, 1) => 1.0,
                    _ => 0.0,
                })
            } else if trees.codomain_vertices() == trees.domain_vertices() {
                D::from_real(50.0 + 10.0 * row as f64)
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap();
    let Eig { d, v } = source.eig_full(&[0, 1], &[2, 3]).unwrap();
    assert!(std::ptr::eq(d.provider(), provider.as_ref()));
    assert!(std::ptr::eq(v.provider(), provider.as_ref()));
    let dense_len = d.materialize().unwrap().dense_data().unwrap().len();
    assert!(!format!("{d:?}").contains(&format!("elements: {dense_len}")));
    let complex_source = D::to_complex(&source);
    let av = complex_source.compose(&v).unwrap();
    let vd = v.compose(&d).unwrap();
    assert!(av
        .dense_data()
        .unwrap()
        .iter()
        .zip(vd.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-8));
    let rebuilt = v
        .compose(&d)
        .unwrap()
        .compose(&v.inv(&[0, 1], &[2]).unwrap())
        .unwrap();
    assert_same_checked_generic_layout_and_close(&rebuilt, &complex_source, |actual, expected| {
        (actual - expected).norm()
    });

    let lambda_plus = Complex64::new(1.0, 3.0_f64.sqrt());
    let identity: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        d.codomain().iter(),
        d.domain().iter(),
        |_, index| Complex64::new(f64::from(index[0] == index[1]), 0.0),
    )
    .unwrap();
    let mut selector = identity.clone();
    let mut found_target = false;
    for index in 0..d.subblock_count() {
        let block = d.subblock(index).unwrap();
        for diagonal in 0..block.shape()[0] {
            let value = d.materialize().unwrap().dense_data().unwrap()
                [block.offset() + diagonal * block.strides()[0] + diagonal * block.strides()[1]];
            if (value - lambda_plus).norm() < 1.0e-8 {
                assert!(!found_target);
                found_target = true;
            } else {
                selector = selector
                    .compose(
                        &d.axpby(Complex64::new(1.0, 0.0), &identity, -value)
                            .unwrap()
                            .scale(Complex64::new(1.0, 0.0) / (lambda_plus - value)),
                    )
                    .unwrap();
            }
        }
    }
    assert!(found_target);
    let projector = v
        .compose(&selector)
        .unwrap()
        .compose(&v.inv(&[0, 1], &[2]).unwrap())
        .unwrap();
    let root = 3.0_f64.sqrt();
    let expected: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.coupled() != &label || row != 0 || column != 0 {
                return Complex64::new(0.0, 0.0);
            }
            match (
                trees.codomain_vertices()[0].get(),
                trees.domain_vertices()[0].get(),
            ) {
                (1, 1) | (2, 2) => Complex64::new(0.5, 0.0),
                (1, 2) => Complex64::new(0.0, root / 2.0),
                (2, 1) => Complex64::new(0.0, -1.0 / (2.0 * root)),
                _ => Complex64::new(0.0, 0.0),
            }
        })
        .unwrap();
    assert_same_checked_generic_layout_and_close(&projector, &expected, |actual, expected| {
        (actual - expected).norm()
    });
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_eig_nonnormal_cross_mu_projectors_for_both_dtypes() {
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_eig::<f64>(n, label.clone());
        assert_sun_checked_generic_eig::<Complex64>(n, label);
    }
}

#[test]
fn checked_generic_eigh_vals_preserves_spectrum_and_dtype() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 2.5).unwrap();

    let spectra = source.eigh_vals(&[0], &[1]).unwrap();
    assert_eq!(spectra.len(), 1);
    assert_eq!(spectra[0].sector, Label::X);
    assert_eq!(spectra[0].values, vec![2.5]);

    let complex = source.convert::<Complex64>();
    assert_eq!(complex.eigh_vals(&[0], &[1]).unwrap(), spectra);

    let compact: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![2.5],
        }],
    )
    .unwrap();
    assert_eq!(compact.eigh_vals(&[0], &[1]).unwrap(), spectra);
    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        compact.eigh_vals(&[0], &[1]),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);
}

fn assert_checked_generic_eigh_factors<D>(
    source: &TensorMap<CheckedOnlyToy, D>,
    close: impl Fn(D, D) -> f64 + Copy,
    adjoint: impl Fn(D) -> D + Copy,
) where
    D: tenet::typed::FactorizationScalar + tenet::typed::SpectrumMagnitude + fmt::Debug,
{
    let Eigh { d, v } = source.eigh_full(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(d.provider(), source.provider()));
    assert!(std::ptr::eq(v.provider(), source.provider()));
    assert!(d.runtime().shares_state_with(source.runtime()));
    assert!(v.runtime().shares_state_with(source.runtime()));
    assert_eq!(v.codomain(), source.codomain());
    assert_eq!(d.codomain(), d.domain());
    assert_eq!(v.domain(), d.codomain());
    assert!(format!("{d:?}").contains("elements: 3"));
    assert_eq!(d.materialize().unwrap().dense_data().unwrap().len(), 9);

    for (actual, expected) in source
        .compose(&v)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(v.compose(&d).unwrap().dense_data().unwrap())
    {
        assert!(close(*actual, *expected) < 1e-10);
    }
    let vectors = v.dense_data().unwrap();
    let diagonal_materialized = d.materialize().unwrap();
    let diagonal = diagonal_materialized.dense_data().unwrap();
    for column in 0..3 {
        for row in 0..3 {
            let gram = (0..3).fold(D::from_real(0.0), |sum, inner| {
                sum + adjoint(vectors[inner + row * 3]) * vectors[inner + column * 3]
            });
            assert!(close(gram, D::from_real(f64::from(row == column))) < 1e-10);
            let rebuilt = (0..3).fold(D::from_real(0.0), |sum, inner| {
                sum + vectors[row + inner * 3]
                    * diagonal[inner + inner * 3]
                    * adjoint(vectors[column + inner * 3])
            });
            assert!(close(rebuilt, source.dense_data().unwrap()[row + column * 3]) < 1e-10);
        }
    }

    // The truncated eigendecomposition is a composition (#1534).
    let found = d.domain()[0]
        .find_truncated(&d.diagview().unwrap(), &Truncation::rank(5))
        .unwrap();
    let truncated_d = d
        .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    let truncated_v = v
        .restrict_leg(&[(v.codomain_rank(), &found.selection)])
        .unwrap();
    assert!(std::ptr::eq(truncated_d.provider(), source.provider()));
    assert!(std::ptr::eq(truncated_v.provider(), source.provider()));
    // The kept values carry the same values in the same order, exactly — no
    // rounding, and no reordering of the values inside a sector.
    let compact = truncated_d.diagview().unwrap();
    assert_eq!(compact.len(), 1);
    assert_eq!(compact[0].sector, Label::X);
    for (actual, expected) in compact[0]
        .values
        .iter()
        .zip([D::from_real(-3.0), D::from_real(2.0)])
    {
        assert_eq!(close(*actual, expected), 0.0);
    }
    let expected_error = (1.0 + 2.0_f64.sqrt()).sqrt();
    assert!((found.error - expected_error).abs() < 1e-12);
}

#[test]
fn checked_generic_eigh_full_and_trunc_preserve_contract_for_both_dtypes() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [-3.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    assert_checked_generic_eigh_factors(
        &source,
        |actual, expected| (actual - expected).abs(),
        |value| value,
    );
    let complex = source.convert::<Complex64>();
    assert_checked_generic_eigh_factors(
        &complex,
        |actual, expected| (actual - expected).norm(),
        |value| value.conj(),
    );
    let Eigh { d: complex_d, .. } = complex.eigh_full(&[0], &[1]).unwrap();
    assert!(complex_d
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.im == 0.0));
}

#[test]
fn checked_generic_eigh_lazy_success_and_failure_leave_the_view_lazy() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let hermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[2.0, 1.0], [1.0, -1.0]][index[0]][index[1]]
        })
        .unwrap();
    let lazy = hermitian.adjoint().unwrap();
    assert!(lazy.network_reuse_class(false) == tenet::typed::NetworkReuseClass::LazyAdjoint);
    assert!(lazy.eigh_full(&[0], &[1]).is_ok());
    assert!(lazy.network_reuse_class(false) == tenet::typed::NetworkReuseClass::LazyAdjoint);

    let nonhermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[0.0, 1.0], [0.0, 0.0]][index[0]][index[1]]
        })
        .unwrap();
    let lazy = nonhermitian.adjoint().unwrap();
    assert!(lazy.eigh_full(&[0], &[1]).is_err());
    assert!(lazy.network_reuse_class(false) == tenet::typed::NetworkReuseClass::LazyAdjoint);
}

#[test]
fn checked_generic_eigh_rejects_invalid_inputs_before_publication() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let nonendomorphism: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
    assert!(nonendomorphism.eigh_full(&[0], &[1]).is_err());

    let nonhermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, index| {
            [[0.0, 1.0], [0.0, 0.0]][index[0]][index[1]]
        })
        .unwrap();
    assert!(nonhermitian.eigh_full(&[0], &[1]).is_err());
    let nonfinite: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, index| {
            if index[0] == index[1] {
                f64::NAN
            } else {
                0.0
            }
        })
        .unwrap();
    assert!(nonfinite.eigh_full(&[0], &[1]).is_err());
}

struct EighFaultExecutor {
    inner: DefaultDenseExecutor,
    calls: Arc<AtomicUsize>,
    fail_at: Option<usize>,
}

#[derive(Clone, Copy)]
enum EigFault {
    Eig,
    Svd,
}

struct EigFaultExecutor {
    inner: DefaultDenseExecutor,
    fault: EigFault,
}

impl DenseExecutor for EigFaultExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(input)
    }

    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        if matches!(self.fault, EigFault::Svd) {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "svd_vals",
                message: "injected checked Generic EIG rank-check failure".to_string(),
            });
        }
        self.inner.svd_vals(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        if matches!(self.fault, EigFault::Eig) {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "eig",
                message: "injected checked Generic EIG failure".to_string(),
            });
        }
        self.inner.eig(input)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

impl DenseExecutor for EighFaultExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if self.fail_at == Some(call) {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "eigh",
                message: "injected checked Generic EIGH failure".to_string(),
            });
        }
        self.inner.eigh(input)
    }

    fn eigh_into(
        &mut self,
        _: DenseRead<'_>,
        _: DenseWrite<'_>,
        _: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eigh_into",
            message: "checked Generic EIGH must use owned output".to_string(),
        })
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

#[test]
fn checked_generic_eigh_preflights_all_sectors_and_runs_once_per_sector() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(EighFaultExecutor {
            inner: DefaultDenseExecutor::new(),
            calls: Arc::clone(&calls),
            fail_at: None,
        }))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();
    let hermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, index| {
            if index[0] == index[1] {
                if trees.coupled() == &Label::Vacuum {
                    1.0
                } else {
                    2.0
                }
            } else {
                0.0
            }
        })
        .unwrap();
    hermitian.eigh_full(&[0], &[1]).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 2);

    calls.store(0, Ordering::Relaxed);
    let nonhermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, index| {
            if trees.coupled() == &Label::X && index == [0, 1] {
                1.0
            } else {
                0.0
            }
        })
        .unwrap();
    assert!(nonhermitian.eigh_full(&[0], &[1]).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_eigh_dense_failure_preserves_the_source() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(EighFaultExecutor {
            inner: DefaultDenseExecutor::new(),
            calls: Arc::clone(&calls),
            fail_at: Some(1),
        }))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[2.0, 1.0], [1.0, -1.0]][index[0]][index[1]]
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    assert!(source.eigh_full(&[0], &[1]).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(source.dense_data().unwrap(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[test]
fn checked_generic_eig_dense_and_rank_check_failures_preserve_the_source() {
    for fault in [EigFault::Eig, EigFault::Svd] {
        let runtime = Runtime::builder()
            .dense_threads(1)
            .with_dense_executor(Box::new(EigFaultExecutor {
                inner: DefaultDenseExecutor::new(),
                fault,
            }))
            .build()
            .unwrap();
        let provider = Arc::new(CheckedOnlyToy::new(0));
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
                [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
            })
            .unwrap();
        let before = source.dense_data().unwrap().to_vec();
        assert!(source.eig_full(&[0], &[1]).is_err());
        assert_eq!(source.dense_data().unwrap(), before);
        assert!(std::ptr::eq(source.provider(), provider.as_ref()));
    }
}

#[test]
fn checked_generic_eigh_qdim_and_decode_failures_publish_no_pair() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [2.0, -1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();

    // The truncation composition decodes labels in `diagview` and reads the
    // quantum dimension in `find_truncated`; both surface the provider error.
    let Eigh { d, .. } = source.eigh_full(&[0], &[1]).unwrap();
    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        d.diagview(),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);

    let spectra = d.diagview().unwrap();
    provider.fail_dim.store(true, Ordering::Relaxed);
    assert!(matches!(
        d.domain()[0].find_truncated(&spectra, &Truncation::rank(1)),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    provider.fail_dim.store(false, Ordering::Relaxed);

    assert_eq!(source.dense_data().unwrap(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[test]
fn checked_generic_eig_qdim_and_decode_failures_publish_no_pair() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();

    // The truncation composition decodes labels in `diagview` and reads the
    // quantum dimension in `find_truncated`; both surface the provider error.
    let Eig { d, .. } = source.eig_full(&[0], &[1]).unwrap();
    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        d.diagview(),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);

    let spectra = d.diagview().unwrap();
    provider.fail_dim.store(true, Ordering::Relaxed);
    assert!(matches!(
        d.domain()[0].find_truncated(&spectra, &Truncation::rank(1)),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    provider.fail_dim.store(false, Ordering::Relaxed);

    assert_eq!(source.dense_data().unwrap(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[test]
fn checked_generic_eigh_signed_ties_are_stable_and_degenerate_projectors_are_invariant() {
    use std::cell::Cell;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let tied: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [-2.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let Eigh { d, .. } = tied.eigh_full(&[0], &[1]).unwrap();
    // The order of the tied magnitudes is the contract; the values carry the
    // eigensolver's rounding (`terms` = the block size 3).
    numerics::assert_slices_close(
        "eigh_full tied spectrum",
        &[
            d.materialize().unwrap().dense_data().unwrap()[0],
            d.materialize().unwrap().dense_data().unwrap()[4],
            d.materialize().unwrap().dense_data().unwrap()[8],
        ],
        &[-2.0, 2.0, 1.0],
        3,
    );

    let degenerate: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [2.0, 2.0, -1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let Eigh { d, v } = degenerate.eigh_full(&[0], &[1]).unwrap();
    let selector_codomain = d.codomain();
    let selector_domain = d.domain();
    let selector: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        &runtime,
        selector_codomain.iter(),
        selector_domain.iter(),
        |_, index| {
            f64::from(
                index[0] == index[1]
                    && d.materialize().unwrap().dense_data().unwrap()[index[0] * 4] == 2.0,
            )
        },
    )
    .unwrap();
    let lazy_vh = v.adjoint().unwrap();
    let data = lazy_vh
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    let position = Cell::new(0usize);
    let codomain = v.domain();
    let domain = v.codomain();
    let vh: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, codomain.iter(), domain.iter(), |_, _| {
            let index = position.get();
            position.set(index + 1);
            data[index]
        })
        .unwrap();
    let projector = v.compose(&selector).unwrap().compose(&vh).unwrap();
    numerics::assert_slices_close(
        "eigh degenerate projector",
        projector.dense_data().unwrap(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        3,
    );
}

#[test]
fn checked_generic_eig_vals_preserves_spectrum_and_dtype() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 2) as f64
            } else {
                0.0
            }
        })
        .unwrap();

    let spectra = source.eig_vals(&[0], &[1]).unwrap();
    assert_eq!(spectra.len(), 1);
    assert_eq!(spectra[0].sector, Label::X);
    assert_eq!(
        spectra[0].values,
        vec![Complex64::new(3.0, 0.0), Complex64::new(2.0, 0.0)]
    );

    let complex = source.convert::<Complex64>();
    assert_eq!(complex.eig_vals(&[0], &[1]).unwrap(), spectra);

    let compact: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![Complex64::new(1.0, -1.0), Complex64::new(0.0, 3.0)],
        }],
    )
    .unwrap();
    assert_eq!(
        compact.eig_vals(&[0], &[1]).unwrap(),
        vec![SectorSpectrum {
            sector: Label::X,
            values: vec![Complex64::new(0.0, 3.0), Complex64::new(1.0, -1.0)],
        }]
    );
    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        compact.eig_vals(&[0], &[1]),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);
}

fn assert_checked_generic_eig_reconstruction(
    source: &TensorMap<CheckedOnlyToy, Complex64>,
    d: &TensorMap<CheckedOnlyToy, Complex64>,
    v: &TensorMap<CheckedOnlyToy, Complex64>,
) {
    assert!(std::ptr::eq(d.provider(), source.provider()));
    assert!(std::ptr::eq(v.provider(), source.provider()));
    assert!(d.runtime().shares_state_with(source.runtime()));
    assert!(v.runtime().shares_state_with(source.runtime()));
    assert_eq!(v.codomain(), source.codomain());
    assert_eq!(v.domain(), d.codomain());
    let av = source.compose(v).unwrap();
    let vd = v.compose(d).unwrap();
    let scale = source
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    assert!(av
        .dense_data()
        .unwrap()
        .iter()
        .zip(vd.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() <= 1.0e-11 * scale));
    let rebuilt = v
        .compose(d)
        .unwrap()
        .compose(&v.inv(&[0], &[1]).unwrap())
        .unwrap();
    assert_same_checked_generic_layout_and_close(&rebuilt, source, |actual, expected| {
        (actual - expected).norm()
    });
}

#[test]
fn checked_generic_eig_full_is_complex_and_reconstructs_nonnormal_inputs() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let Eig {
        d: real_d,
        v: real_v,
    } = real.eig_full(&[0], &[1]).unwrap();
    assert!(real_d
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im.abs() > 1.0));
    for column in 0..2 {
        let pivot = (0..2)
            .max_by(|&left, &right| {
                real_v.dense_data().unwrap()[left + column * 2]
                    .norm()
                    .total_cmp(&real_v.dense_data().unwrap()[right + column * 2].norm())
            })
            .unwrap();
        let pivot = real_v.dense_data().unwrap()[pivot + column * 2];
        assert!(pivot.im.abs() < 1.0e-12);
        assert!(pivot.re >= 0.0);
    }
    assert_checked_generic_eig_reconstruction(&real.convert::<Complex64>(), &real_d, &real_v);

    let complex = real.convert::<Complex64>().scale(Complex64::new(1.0, 0.25));
    let Eig {
        d: complex_d,
        v: complex_v,
    } = complex.eig_full(&[0], &[1]).unwrap();
    assert!(complex_d
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im.abs() > 1.0));
    assert_checked_generic_eig_reconstruction(&complex, &complex_d, &complex_v);

    let compact: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, -3.0)],
        }],
    )
    .unwrap();
    let Eig { d, v } = compact.eig_full(&[0], &[1]).unwrap();
    assert_eq!(
        d.diagview().unwrap()[0].values,
        [Complex64::new(0.0, -3.0), Complex64::new(1.0, 1.0)]
    );
    let dense = compact.materialize().unwrap();
    assert_checked_generic_eig_reconstruction(&dense, &d, &v);
    let fault_runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(EigFaultExecutor {
            inner: DefaultDenseExecutor::new(),
            fault: EigFault::Eig,
        }))
        .build()
        .unwrap();
    let fault_provider = Arc::new(CheckedOnlyToy::new(1));
    let fault_leg = GradedSpace::try_new(Arc::clone(&fault_provider), [(Label::X, 2)]).unwrap();
    let fault_compact: TensorMap<_, Complex64> = TensorMap::diagonal(
        &fault_runtime,
        &fault_leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, -3.0)],
        }],
    )
    .unwrap();
    // The eligible compact path reaches checked V publication; falling back
    // to dense EIG would return the injected dense error instead.
    fault_provider.invalid_style.store(true, Ordering::Relaxed);
    let compact_error = fault_compact.eig_full(&[0], &[1]).unwrap_err();
    assert!(
        matches!(
            compact_error,
            GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Operation(
                tenet::typed::OperationError::Core(
                    tenet::typed::CoreError::UnsupportedFusionStyle { .. }
                )
            ))
        ),
        "{compact_error:?}"
    );
}

#[test]
fn checked_generic_eig_ties_are_stable_and_degenerate_projectors_are_invariant() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let tied: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [-2.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let Eig { d, .. } = tied.eig_full(&[0], &[1]).unwrap();
    numerics::assert_slices_close(
        "eig_full tied spectrum",
        &[
            d.materialize().unwrap().dense_data().unwrap()[0],
            d.materialize().unwrap().dense_data().unwrap()[4],
            d.materialize().unwrap().dense_data().unwrap()[8],
        ],
        &[
            Complex64::new(-2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        3,
    );

    let degenerate: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [2.0, 2.0, -1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let Eig { d, v } = degenerate.eig_full(&[0], &[1]).unwrap();
    let selector: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        d.codomain().iter(),
        d.domain().iter(),
        |_, index| {
            Complex64::new(
                f64::from(
                    index[0] == index[1]
                        && (d.materialize().unwrap().dense_data().unwrap()[index[0] * 4]
                            - Complex64::new(2.0, 0.0))
                        .norm()
                            < 1.0e-12,
                ),
                0.0,
            )
        },
    )
    .unwrap();
    let projector = v
        .compose(&selector)
        .unwrap()
        .compose(&v.inv(&[0], &[1]).unwrap())
        .unwrap();
    numerics::assert_slices_close(
        "eig degenerate projector",
        projector.dense_data().unwrap(),
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        3,
    );
}

#[test]
fn checked_generic_eig_truncation_reports_discarded_spectrum_norm_only() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [-3.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    // The truncated eigendecomposition is a composition (#1534).
    let Eig { d, v } = source.eig_full(&[0], &[1]).unwrap();
    let found = d.domain()[0]
        .find_truncated(&d.diagview().unwrap(), &Truncation::rank(5))
        .unwrap();
    let truncated_d = d
        .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    let truncated_v = v
        .restrict_leg(&[(v.codomain_rank(), &found.selection)])
        .unwrap();
    assert!(std::ptr::eq(truncated_d.provider(), provider.as_ref()));
    assert!(std::ptr::eq(truncated_v.provider(), provider.as_ref()));
    let kept = truncated_d.diagview().unwrap();
    assert_eq!(kept[0].sector, Label::X);
    assert_eq!(
        kept[0].values,
        vec![Complex64::new(-3.0, 0.0), Complex64::new(2.0, 0.0)]
    );
    assert!((found.error - (1.0 + 2.0_f64.sqrt()).sqrt()).abs() < 1.0e-12);
}

#[test]
fn checked_generic_eig_rejects_jordan_and_nonfinite_inputs() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let jordan: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[1.0, 1.0], [0.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let error = jordan.eig_full(&[0], &[1]).unwrap_err();
    assert!(format!("{error:?}")
        .contains("eig requires a numerically diagonalizable coupled-sector matrix"));

    let nonfinite: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            if index == [0, 0] {
                f64::NAN
            } else {
                0.0
            }
        })
        .unwrap();
    let error = nonfinite.eig_full(&[0], &[1]).unwrap_err();
    assert!(format!("{error:?}").contains("eig input components must be finite"));
}

#[test]
fn checked_generic_eig_lazy_calls_leave_the_source_view_lazy() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let lazy = source.adjoint().unwrap();
    assert!(lazy.network_reuse_class(false) == tenet::typed::NetworkReuseClass::LazyAdjoint);
    assert!(lazy.eig_full(&[0], &[1]).is_ok());
    assert!(lazy.network_reuse_class(false) == tenet::typed::NetworkReuseClass::LazyAdjoint);
}

#[test]
fn checked_tr_matches_the_literal_weighted_sum_and_keeps_error_precedence() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();
    // 2+2 endomorphism: diagonal blocks exist per (coupled, vertices) pair and
    // off-diagonal multiplicity blocks must not contribute.
    let complex =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], lazy_oracle_value)
            .unwrap();
    assert!((0..complex.subblock_count()).any(|index| {
        let trees = complex.subblock_fusion_trees(index).unwrap();
        trees.codomain_vertices() != trees.domain_vertices()
    }));
    let dim = |label: &Label| match label {
        Label::Vacuum => 1.0,
        Label::X => 1.0 + 2.0_f64.sqrt(),
        _ => unreachable!(),
    };

    let mut expected = Complex64::new(0.0, 0.0);
    common::literal_weighted_trace(&snapshot!(complex), dim, |value, weight| {
        expected += value * weight
    });
    assert!(expected.im.abs() > 1e-6);
    let lazy = complex.adjoint().unwrap();
    let sectors = (0..complex.subblock_count())
        .map(|index| *complex.subblock_fusion_trees(index).unwrap().coupled())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    provider.coefficient_queries.store(0, Ordering::Relaxed);
    assert!(lazy_close_c64(complex.tr().unwrap(), expected));
    assert_eq!(
        provider.coefficient_queries.swap(0, Ordering::Relaxed),
        sectors
    );
    assert!(lazy_close_c64(lazy.tr().unwrap(), expected.conj()));
    assert_eq!(
        provider.coefficient_queries.swap(0, Ordering::Relaxed),
        sectors
    );

    let real = complex.re();
    let mut expected_real = 0.0;
    common::literal_weighted_trace(&snapshot!(real), dim, |value, weight| {
        expected_real += value * weight
    });
    let real_lazy = real.adjoint().unwrap();
    provider.coefficient_queries.store(0, Ordering::Relaxed);
    assert!(lazy_close_f64(real.tr().unwrap(), expected_real));
    assert_eq!(
        provider.coefficient_queries.swap(0, Ordering::Relaxed),
        sectors
    );
    assert!(lazy_close_f64(real_lazy.tr().unwrap(), expected_real));
    assert_eq!(
        provider.coefficient_queries.swap(0, Ordering::Relaxed),
        sectors
    );
    let mut unweighted = 0.0;
    common::literal_weighted_trace(
        &snapshot!(real),
        |_| 1.0,
        |value, weight| unweighted += value * weight,
    );
    assert!((unweighted - expected_real).abs() > 1e-6);

    // Error precedence: endomorphism check before any provider query, then the
    // provider's dim failure, for owned and lazy inputs alike.
    let non_endomorphism =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], lazy_oracle_value).unwrap();
    provider.fail_dim.store(true, Ordering::Relaxed);
    provider.coefficient_queries.store(0, Ordering::Relaxed);
    for tensor in [
        non_endomorphism.clone(),
        non_endomorphism.adjoint().unwrap(),
    ] {
        assert!(matches!(
            tensor.tr().unwrap_err(),
            GenericTensorError::Facade(tenet::typed::Error::InvalidArgument(message))
                if message == "tr() requires an endomorphism (domain == codomain)"
        ));
    }
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
    for tensor in [complex.clone(), lazy.clone()] {
        assert!(matches!(
            tensor.tr().unwrap_err(),
            GenericTensorError::Structure(CheckedGenericStructureError::Provider(
                ToyError::Algebra
            ))
        ));
    }
    provider.fail_dim.store(false, Ordering::Relaxed);
    assert!(lazy_close_c64(complex.tr().unwrap(), expected));
}

/// TensorKit `inner` (`vectorinterface.jl`): `sum_c dim(c) * <a_c, b_c>` over
/// every stored block with the first argument conjugated, walked with an
/// explicit odometer over public block geometry so the expected value never
/// passes through the coupled-region or oriented owners under test.
fn literal_weighted_inner<D: Copy>(
    lhs: &common::Snapshot<Label, D>,
    rhs: &common::Snapshot<Label, D>,
    dim: impl Fn(&Label) -> f64,
    mut accumulate: impl FnMut(D, D, f64),
) {
    assert_eq!(lhs.blocks.len(), rhs.blocks.len());
    for ((trees, geometry), (other_trees, other_geometry)) in lhs.blocks.iter().zip(&rhs.blocks) {
        assert_eq!(trees, other_trees);
        assert_eq!(geometry.shape, other_geometry.shape);
        let weight = dim(trees.coupled());
        common::for_each_index(&geometry.shape, |index| {
            accumulate(
                lhs.data[common::linear(geometry, index)],
                rhs.data[common::linear(other_geometry, index)],
                weight,
            );
        });
    }
}

#[test]
fn checked_inner_and_norm_take_one_weight_per_sector_and_keep_error_precedence() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();
    // 2+2 with outer multiplicity: every coupled sector owns several blocks
    // (B > G), so a per-block weight lookup would be observable as extra
    // provider queries below.
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], lazy_oracle_value)
        .unwrap();
    let rhs =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            lazy_oracle_value(trees, indices) * Complex64::new(0.5, -1.5) + Complex64::new(1.0, 2.0)
        })
        .unwrap();
    let sectors = (0..lhs.subblock_count())
        .map(|index| *lhs.subblock_fusion_trees(index).unwrap().coupled())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    assert_eq!(sectors, 2);
    assert!(lhs.subblock_count() > sectors);
    let dim = |label: &Label| match label {
        Label::Vacuum => 1.0,
        Label::X => 1.0 + 2.0_f64.sqrt(),
        _ => unreachable!(),
    };
    let literal_c64 = |a: &TensorMap<CheckedOnlyToy, Complex64>,
                       b: &TensorMap<CheckedOnlyToy, Complex64>| {
        let mut expected = Complex64::new(0.0, 0.0);
        literal_weighted_inner(&snapshot!(a), &snapshot!(b), dim, |x, y, weight| {
            expected += x.conj() * y * weight
        });
        expected
    };
    let literal_f64 = |a: &TensorMap<CheckedOnlyToy, f64>, b: &TensorMap<CheckedOnlyToy, f64>| {
        let mut expected = 0.0;
        literal_weighted_inner(&snapshot!(a), &snapshot!(b), dim, |x, y, weight| {
            expected += x * y * weight
        });
        expected
    };

    // Owned, complex and real.
    let expected = literal_c64(&lhs, &rhs);
    assert!(expected.im.abs() > 1e-6);
    assert!(lazy_close_c64(lhs.inner(&rhs).unwrap(), expected));
    assert!(lazy_close_c64(rhs.inner(&lhs).unwrap(), expected.conj()));
    assert!(lazy_close_f64(
        lhs.norm(2.0).unwrap(),
        literal_c64(&lhs, &lhs).re.sqrt()
    ));
    let real_lhs = lhs.re();
    let real_rhs = rhs.re();
    assert!(lazy_close_f64(
        real_lhs.inner(&real_rhs).unwrap(),
        literal_f64(&real_lhs, &real_rhs)
    ));
    assert!(lazy_close_f64(
        real_lhs.norm(2.0).unwrap(),
        literal_f64(&real_lhs, &real_lhs).sqrt()
    ));
    let mut unweighted = Complex64::new(0.0, 0.0);
    literal_weighted_inner(
        &snapshot!(lhs),
        &snapshot!(rhs),
        |_| 1.0,
        |x, y, w| unweighted += x.conj() * y * w,
    );
    assert!((unweighted - expected).norm() > 1e-6);

    // Lazy adjoints in every orientation: the literal oracle on the
    // materialized lazy payload, and <a†, b†> = conj(<a, b>) against owned.
    let lazy_lhs = lhs.adjoint().unwrap();
    let lazy_rhs = rhs.adjoint().unwrap();
    let expected_lazy = literal_c64(&lazy_lhs, &lazy_rhs);
    assert!(lazy_close_c64(expected_lazy, expected.conj()));
    assert!(lazy_close_c64(
        lazy_lhs.inner(&lazy_rhs).unwrap(),
        expected_lazy
    ));
    let expected_mixed = literal_c64(&lazy_lhs, &rhs);
    assert!(lazy_close_c64(
        lazy_lhs.inner(&rhs).unwrap(),
        expected_mixed
    ));
    assert!(lazy_close_c64(
        rhs.inner(&lazy_lhs).unwrap(),
        expected_mixed.conj()
    ));
    assert!(lazy_close_f64(
        lazy_lhs.norm(2.0).unwrap(),
        lhs.norm(2.0).unwrap()
    ));
    let real_lazy = real_lhs.adjoint().unwrap();
    assert!(lazy_close_f64(
        real_lazy.inner(&real_rhs).unwrap(),
        literal_f64(&real_lazy, &real_rhs)
    ));

    // Exactly G `dim` queries per call on the dense and the oriented owners.
    // The former per-call map hashed 3G (dense) or 2G + B (lazy) times; hashes
    // are not provider queries, so only the query count is pinned here.
    for (row, call) in [
        (
            "owned inner",
            Box::new(|| lhs.inner(&rhs).map(|_| ())) as Box<dyn Fn() -> _>,
        ),
        ("owned norm", Box::new(|| lhs.norm(2.0).map(|_| ()))),
        (
            "lazy-lazy inner",
            Box::new(|| lazy_lhs.inner(&lazy_rhs).map(|_| ())),
        ),
        (
            "lazy-owned inner",
            Box::new(|| lazy_lhs.inner(&rhs).map(|_| ())),
        ),
        (
            "owned-lazy inner",
            Box::new(|| rhs.inner(&lazy_lhs).map(|_| ())),
        ),
        ("lazy norm", Box::new(|| lazy_lhs.norm(2.0).map(|_| ()))),
    ] {
        provider.coefficient_queries.store(0, Ordering::Relaxed);
        call().unwrap();
        assert_eq!(
            provider.coefficient_queries.load(Ordering::Relaxed),
            sectors,
            "{row}"
        );
    }

    // Error precedence: space mismatch and diagonal-payload rejection come
    // before any weight query even while `dim` is failing; then the provider
    // failure surfaces for owned and lazy inputs on both reductions.
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let mismatched =
        TensorMap::from_subblock_fn(&runtime, [&wide, &wide], [&wide, &wide], lazy_oracle_value)
            .unwrap();
    let diagonal = TensorMap::<_, Complex64>::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![Complex64::new(3.0, 0.0)],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            },
        ],
    )
    .unwrap();
    provider.fail_dim.store(true, Ordering::Relaxed);
    provider.coefficient_queries.store(0, Ordering::Relaxed);
    for (a, b) in [
        (&lhs, &mismatched),
        (&mismatched, &lhs),
        (&lazy_lhs, &mismatched),
        (&mismatched, &lazy_lhs),
    ] {
        assert!(matches!(
            a.inner(b).unwrap_err(),
            GenericTensorError::Facade(tenet::typed::Error::InvalidArgument(message))
                if message == "tensors live on different spaces or block layouts"
        ));
    }
    assert!(matches!(
        diagonal.inner(&diagonal).unwrap_err(),
        GenericTensorError::Facade(tenet::typed::Error::InvalidArgument(message))
            if message == "checked Generic reductions require dense payloads"
    ));
    assert!(matches!(
        diagonal.norm(2.0).unwrap_err(),
        GenericTensorError::Facade(tenet::typed::Error::InvalidArgument(message))
            if message == "checked Generic reductions require dense payloads"
    ));
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
    for tensor in [&lhs, &lazy_lhs] {
        assert!(matches!(
            tensor.inner(&rhs).unwrap_err(),
            GenericTensorError::Structure(CheckedGenericStructureError::Provider(
                ToyError::Algebra
            ))
        ));
        assert!(matches!(
            tensor.norm(2.0).unwrap_err(),
            GenericTensorError::Structure(CheckedGenericStructureError::Provider(
                ToyError::Algebra
            ))
        ));
    }
    provider.fail_dim.store(false, Ordering::Relaxed);
    assert!(lazy_close_c64(lhs.inner(&rhs).unwrap(), expected));
    assert!(lazy_close_c64(
        lazy_lhs.inner(&lazy_rhs).unwrap(),
        expected_lazy
    ));
}
