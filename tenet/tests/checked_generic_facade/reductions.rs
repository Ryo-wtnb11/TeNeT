use super::*;

#[test]
fn checked_generic_reductions_cover_real_complex_dense_payloads() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
    let inner = source.inner(&source).unwrap();
    assert!(inner.is_finite());
    assert!((source.norm(2.0).unwrap() * source.norm(2.0).unwrap() - inner).abs() < 1e-12);
    assert!(source.tr().unwrap().is_finite());
    let complex = source.convert::<Complex64>();
    assert!(complex.inner(&complex).unwrap().re.is_finite());
    assert!(complex.norm(2.0).unwrap().is_finite());
    assert!(complex.tr().unwrap().re.is_finite());
    assert!(provider.coefficient_queries.load(Ordering::Relaxed) > 0);
    // Checked Generic has only the Frobenius reduction: every other exponent
    // is a typed rejection, never a silently different norm.
    for p in [1.0, 3.0, f64::INFINITY, 0.0, f64::NAN] {
        assert!(
            matches!(
                source.norm(p),
                Err(GenericTensorError::Facade(
                    tenet::typed::Error::InvalidArgument(_)
                ))
            ),
            "checked Generic norm({p})"
        );
    }
}

#[test]
fn checked_generic_reductions_do_not_requery_the_admitted_fusion_style() {
    // Admission fixed the Generic style; a provider whose answer changes
    // afterwards is neither asked again nor able to drop the `dim` weights.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
    let inner = source.inner(&source).unwrap();
    let norm = source.norm(2.0).unwrap();
    provider.style_queries.store(0, Ordering::Relaxed);
    provider.invalid_style.store(true, Ordering::Relaxed);
    assert_eq!(source.inner(&source).unwrap().to_bits(), inner.to_bits());
    assert_eq!(source.norm(2.0).unwrap().to_bits(), norm.to_bits());
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_host_add_scale_cover_real_and_complex_payloads() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
    let added = source.axpby(2.0, &source, -1.0).unwrap();
    assert_eq!(added.dense_data().unwrap(), source.dense_data().unwrap());
    let scaled = source.scale(3.0);
    assert!(scaled
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(a, b)| (*a - 3.0 * *b).abs() < 1e-12));

    let complex = source.convert::<Complex64>();
    let added = complex
        .axpby(
            Complex64::new(2.0, 0.0),
            &complex,
            Complex64::new(-1.0, 0.0),
        )
        .unwrap();
    assert_eq!(added.dense_data().unwrap(), complex.dense_data().unwrap());
    let scaled = complex.scale(Complex64::new(0.5, -1.0));
    assert!(scaled
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(a, b)| (*a - *b * Complex64::new(0.5, -1.0)).norm() < 1e-12));
}

#[test]
fn checked_generic_add_rejects_runtime_before_layout_without_queries() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let left: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&foreign_runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    reset_provider_queries(&provider);
    let before = left.dense_data().unwrap().to_vec();
    let error = left.axpby(1.0, &right, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::typed::Error::RuntimeMismatch)
    ));
    assert_eq!(left.dense_data().unwrap(), before.as_slice());
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_add_rejects_layout_mismatch_without_queries() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let left: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    reset_provider_queries(&provider);
    let before = left.dense_data().unwrap().to_vec();
    let error = left.axpby(1.0, &right, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::typed::Error::InvalidArgument(_))
    ));
    assert_eq!(left.dense_data().unwrap(), before.as_slice());
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_add_assign_rejects_runtime_before_layout_and_preserves_receiver() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let mut left: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&foreign_runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    let before_data = left.dense_data().unwrap().to_vec();
    let before_trees = (0..left.subblock_count())
        .map(|index| left.subblock_fusion_trees(index).unwrap())
        .collect::<Vec<_>>();
    reset_provider_queries(&provider);
    let error = right.axpby_into(&mut left, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::typed::Error::RuntimeMismatch)
    ));
    assert_eq!(left.dense_data().unwrap(), before_data.as_slice());
    assert_eq!(
        (0..left.subblock_count())
            .map(|index| left.subblock_fusion_trees(index).unwrap())
            .collect::<Vec<_>>(),
        before_trees
    );
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_add_assign_rejects_layout_mismatch_and_preserves_receiver() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let mut left: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    let before_data = left.dense_data().unwrap().to_vec();
    let before_trees = (0..left.subblock_count())
        .map(|index| left.subblock_fusion_trees(index).unwrap())
        .collect::<Vec<_>>();
    reset_provider_queries(&provider);
    let error = right.axpby_into(&mut left, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::typed::Error::InvalidArgument(_))
    ));
    assert_eq!(left.dense_data().unwrap(), before_data.as_slice());
    assert_eq!(
        (0..left.subblock_count())
            .map(|index| left.subblock_fusion_trees(index).unwrap())
            .collect::<Vec<_>>(),
        before_trees
    );
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_lazy_adjoint_preserves_provider_and_reductions() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();

    let adjoint = source.adjoint().unwrap();
    assert!(std::ptr::eq(adjoint.provider(), provider.as_ref()));
    assert!((adjoint.norm(2.0).unwrap() - source.norm(2.0).unwrap()).abs() < 1.0e-12);
    assert!((adjoint.tr().unwrap() - source.tr().unwrap()).abs() < 1.0e-12);

    let complex = source.convert::<Complex64>();
    let complex_adjoint = complex.adjoint().unwrap();
    assert!(std::ptr::eq(complex_adjoint.provider(), provider.as_ref()));
    assert!((complex_adjoint.tr().unwrap() - complex.tr().unwrap().conj()).norm() < 1.0e-12);
}

#[test]
fn checked_generic_reduction_dimension_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.norm(2.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(ToyError::Algebra))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_adjoint_and_reductions_preserve_provider_and_errors() {
    use tenet::sector::SUNFusionRule;
    use tenet::sector::SUNFusionRuleError;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, _| {
                trees.coupled().iter().sum::<i64>() as f64 + 1.0
            })
            .unwrap();

        let dagger = source.adjoint().unwrap();
        assert!(std::ptr::eq(dagger.provider(), provider.as_ref()));
        assert!((dagger.norm(2.0).unwrap() - source.norm(2.0).unwrap()).abs() < 1.0e-12);
        assert!((dagger.inner(&dagger).unwrap() - source.inner(&source).unwrap()).abs() < 1.0e-12);
        assert!((dagger.tr().unwrap() - source.tr().unwrap()).abs() < 1.0e-12);

        let complex = source.convert::<Complex64>();
        let complex_dagger = complex.adjoint().unwrap();
        assert!(std::ptr::eq(complex_dagger.provider(), provider.as_ref()));
        assert!((complex_dagger.tr().unwrap() - complex.tr().unwrap().conj()).norm() < 1.0e-12);
    }

    let provider = SUNFusionRule::new(3).unwrap();
    let three = provider.encode_dynkin(&[1, 0]).unwrap();
    let eight = provider.encode_dynkin(&[1, 1]).unwrap();
    assert!(matches!(
        provider.try_r_symbol_generic(three, three, eight),
        Err(SUNFusionRuleError::Racah(_))
    ));
}

/// TensorKit's zero-scale rule on the checked-generic `add`/`scale`
/// (VectorInterface `scale(x, α) = (iszero(α) ? zero(x) : x) * α`, observed on
/// TensorKit 0.17.1 as in `typed_zero_scale.rs`): a zero coefficient drops its
/// operand, NaN and `Inf` included (#1442).
#[test]
fn checked_generic_add_and_scale_drop_zero_scaled_operands_as_tensorkit() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let x: TensorMap<_, f64> = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
        [[f64::INFINITY, f64::NAN], [1.5, -2.0]][ij[0]][ij[1]]
    })
    .unwrap();
    let y: TensorMap<_, f64> = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
        [[3.0, 4.0], [f64::NAN, 0.5]][ij[0]][ij[1]]
    })
    .unwrap();
    let scale = |value: f64, factor: f64| if factor == 0.0 { 0.0 } else { value * factor };
    let same = |got: &[f64], want: &[f64]| {
        assert_eq!(got.len(), want.len());
        for (&got, &want) in got.iter().zip(want) {
            assert!(
                (got.is_nan() && want.is_nan()) || got == want,
                "{got} != {want}"
            );
        }
    };
    for (alpha, beta) in [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0), (2.0, -1.0)] {
        let want: Vec<f64> = x
            .dense_data()
            .unwrap()
            .iter()
            .zip(y.dense_data().unwrap())
            .map(|(&a, &b)| scale(a, alpha) + scale(b, beta))
            .collect();
        same(
            x.axpby(alpha, &y, beta).unwrap().dense_data().unwrap(),
            &want,
        );
        let mut assigned = x.materialize().unwrap();
        y.axpby_into(&mut assigned, beta, alpha).unwrap();
        same(assigned.dense_data().unwrap(), &want);
    }
    for factor in [0.0, 2.0] {
        let want: Vec<f64> = x
            .dense_data()
            .unwrap()
            .iter()
            .map(|&a| scale(a, factor))
            .collect();
        same(x.scale(factor).dense_data().unwrap(), &want);
        let mut assigned = x.materialize().unwrap();
        assigned.scale_assign(factor).unwrap();
        same(assigned.dense_data().unwrap(), &want);
    }
}
