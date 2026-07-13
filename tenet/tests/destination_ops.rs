use tenet::prelude::*;

fn assert_close(lhs: &[f64], rhs: &[f64], tolerance: f64) {
    assert_eq!(lhs.len(), rhs.len());
    for (a, b) in lhs.iter().zip(rhs) {
        assert!((a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs())));
    }
}

fn assert_close_c64(lhs: &[Complex64], rhs: &[Complex64], tolerance: f64) {
    assert_eq!(lhs.len(), rhs.len());
    for (a, b) in lhs.iter().zip(rhs) {
        assert!((*a - *b).norm() <= tolerance * (1.0 + a.norm().max(b.norm())));
    }
}

#[test]
fn contract_overwrite_into_matches_owned_f64() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::u1([(-1, 2), (0, 3), (1, 2)]);
    let lhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 30101).unwrap();
    let rhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 30102).unwrap();
    let contracted = lhs.contract(&rhs, &[1], &[0]).unwrap();
    let mut destination = contracted.scale(f64::NAN).unwrap();
    let mut context = TensorExecutionContext::default();

    context
        .contract_overwrite_into(&mut destination, &lhs, &rhs, &[1], &[0], Scalar::F64(1.5))
        .unwrap();
    let expected = contracted.scale(1.5).unwrap();
    assert_close(destination.data(), expected.data(), 1e-12);
}

#[test]
fn permute_overwrite_into_matches_owned_c64() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::su2([(0, 2), (1, 2), (2, 1)]);
    let source =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space, &space], [&space], 30111).unwrap();
    let permuted = source.permute(&[1], &[2, 0]).unwrap();
    let mut destination = permuted
        .scale_c64(Complex64::new(f64::NAN, f64::NAN))
        .unwrap();
    let mut context = TensorExecutionContext::default();
    let alpha = Complex64::new(0.75, -0.25);

    context
        .permute_overwrite_into(&mut destination, &source, &[1], &[2, 0], Scalar::C64(alpha))
        .unwrap();
    let expected = permuted.scale_c64(alpha).unwrap();
    assert_close_c64(destination.data_c64(), expected.data_c64(), 1e-12);
}

#[test]
fn destination_validation_precedes_mutation() {
    let runtime = Runtime::builder().build().unwrap();
    let other_runtime = Runtime::builder().build().unwrap();
    let space = Space::u1([(0, 3)]);
    let lhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 30121).unwrap();
    let rhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 30122).unwrap();
    let foreign =
        Tensor::rand_with_seed(&other_runtime, Dtype::F64, [&space], [&space], 30123).unwrap();
    let mut wrong_layout =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space, &space], [], 30124).unwrap();
    let before = wrong_layout.data().to_vec();
    let mut context = TensorExecutionContext::default();

    assert!(context
        .contract_overwrite_into(&mut wrong_layout, &lhs, &rhs, &[1], &[0], Scalar::F64(1.0),)
        .is_err());
    assert_eq!(wrong_layout.data(), before);

    let mut destination = lhs.contract(&rhs, &[1], &[0]).unwrap();
    assert!(context
        .contract_overwrite_into(
            &mut destination,
            &lhs,
            &foreign,
            &[1],
            &[0],
            Scalar::F64(1.0),
        )
        .is_err());

    let mut aliased = lhs.clone();
    assert!(context
        .contract_overwrite_into(&mut aliased, &lhs, &rhs, &[1], &[0], Scalar::F64(1.0),)
        .is_err());

    let mut bound = TensorExecutionContext::for_runtime(&runtime).unwrap();
    let mut foreign_destination = foreign.contract(&foreign, &[1], &[0]).unwrap();
    let foreign_before = foreign_destination.data().to_vec();
    assert!(bound
        .contract_overwrite_into(
            &mut foreign_destination,
            &foreign,
            &foreign,
            &[1],
            &[0],
            Scalar::F64(1.0),
        )
        .is_err());
    assert_eq!(foreign_destination.data(), foreign_before);

    let lhs_c = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 30125).unwrap();
    let rhs_c = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 30126).unwrap();
    let mut wrong_dtype = lhs_c.contract(&rhs_c, &[1], &[0]).unwrap();
    let dtype_before = wrong_dtype.data_c64().to_vec();
    assert!(context
        .contract_overwrite_into(&mut wrong_dtype, &lhs, &rhs, &[1], &[0], Scalar::F64(1.0),)
        .is_err());
    assert_eq!(wrong_dtype.data_c64(), dtype_before);

    let z2 = Space::z2([(0, 3), (1, 1)]);
    let z2_lhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&z2], [&z2], 30127).unwrap();
    let z2_rhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&z2], [&z2], 30128).unwrap();
    let mut wrong_rule = z2_lhs.contract(&z2_rhs, &[1], &[0]).unwrap();
    let rule_before = wrong_rule.data().to_vec();
    assert!(context
        .contract_overwrite_into(&mut wrong_rule, &lhs, &rhs, &[1], &[0], Scalar::F64(1.0),)
        .is_err());
    assert_eq!(wrong_rule.data(), rule_before);
}

#[test]
fn su3_contract_overwrite_clears_structural_zero_output() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::su3([((1, 0), 2), ((0, 1), 1)]).unwrap();
    let lhs =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space, &space], 30131).unwrap();
    let rhs =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space, &space], [&space], 30132).unwrap();
    let shape = lhs.contract(&rhs, &[1, 2], &[0, 1]).unwrap();
    let mut destination = shape.scale(f64::NAN).unwrap();
    let mut context = TensorExecutionContext::for_runtime(&runtime).unwrap();

    context
        .contract_overwrite_into(
            &mut destination,
            &lhs,
            &rhs,
            &[1, 2],
            &[0, 1],
            Scalar::F64(0.0),
        )
        .unwrap();

    assert!(destination.data().iter().all(|value| !value.is_nan()));
    assert!(
        destination.data().iter().all(|value| *value == 0.0),
        "alpha=0 must clear every SU(3) output block, including blocks without a contributing GEMM"
    );
}

#[test]
fn destination_dispatch_matches_owned_for_every_rule() {
    let runtime = Runtime::builder().build().unwrap();
    let spaces = vec![
        Space::u1([(-1, 1), (0, 2), (1, 1)]),
        Space::z2([(0, 2), (1, 1)]),
        Space::fz2([(0, 2), (1, 1)]),
        Space::su2([(0, 2), (1, 1)]),
        Space::product([((0, 0), 2), ((1, 1), 1)]).unwrap(),
        Space::fz2_u1_su2([((0, 0, 0), 2), ((1, 1, 1), 1)]).unwrap(),
        Space::su3([((1, 0), 2), ((0, 1), 1)]).unwrap(),
    ];

    for (index, space) in spaces.iter().enumerate() {
        let lhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [space],
            [space],
            30_200 + index as u64 * 2,
        )
        .unwrap();
        let rhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [space],
            [space],
            30_201 + index as u64 * 2,
        )
        .unwrap();
        let contracted = lhs.contract(&rhs, &[1], &[0]).unwrap();
        let mut contract_destination = contracted.scale(f64::NAN).unwrap();
        let permuted = lhs.permute(&[1], &[0]).unwrap();
        let mut permute_destination = permuted.scale(f64::NAN).unwrap();
        let mut context = TensorExecutionContext::for_runtime(&runtime).unwrap();
        let mut contract_cache = ContractOverwriteCache::default();
        let mut permute_cache = PermuteOverwriteCache::default();

        for _ in 0..2 {
            assert_eq!(
                context
                    .try_contract_overwrite_into(
                        &mut contract_cache,
                        &mut contract_destination,
                        &lhs,
                        &rhs,
                        &[1],
                        &[0],
                        Scalar::F64(1.0),
                    )
                    .unwrap(),
                OverwriteOutcome::Written
            );
            assert_eq!(
                context
                    .try_permute_overwrite_into(
                        &mut permute_cache,
                        &mut permute_destination,
                        &lhs,
                        &[1],
                        &[0],
                        Scalar::F64(1.0),
                    )
                    .unwrap(),
                OverwriteOutcome::Written
            );
        }

        assert_eq!(contract_cache.preparations(), 1);
        assert_eq!(permute_cache.preparations(), 1);
        assert_close(contract_destination.data(), contracted.data(), 1e-12);
        assert_close(permute_destination.data(), permuted.data(), 1e-12);
    }
}

#[test]
fn checked_contract_cache_prepares_once_and_reports_incompatible_without_writing() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::u1([(-1, 2), (0, 3), (1, 2)]);
    let lhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 30_301).unwrap();
    let rhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 30_302).unwrap();
    let expected = lhs.contract(&rhs, &[1], &[0]).unwrap();
    let mut destination = expected.scale(f64::NAN).unwrap();
    let mut context = TensorExecutionContext::for_runtime(&runtime).unwrap();
    let mut cache = ContractOverwriteCache::default();

    assert_eq!(
        context
            .try_contract_overwrite_into(
                &mut cache,
                &mut destination,
                &lhs,
                &rhs,
                &[1],
                &[0],
                Scalar::F64(1.0),
            )
            .unwrap(),
        OverwriteOutcome::Written
    );
    assert_eq!(cache.preparations(), 1);
    assert_close(destination.data(), expected.data(), 1e-12);

    destination = expected.scale(f64::NAN).unwrap();
    assert_eq!(
        context
            .try_contract_overwrite_into(
                &mut cache,
                &mut destination,
                &lhs,
                &rhs,
                &[1],
                &[0],
                Scalar::F64(1.0),
            )
            .unwrap(),
        OverwriteOutcome::Written
    );
    assert_eq!(cache.preparations(), 1);

    let mut nonunique = expected.scale(f64::NAN).unwrap();
    let held = nonunique.clone();
    let before = nonunique
        .data()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(
        context
            .try_contract_overwrite_into(
                &mut cache,
                &mut nonunique,
                &lhs,
                &rhs,
                &[1],
                &[0],
                Scalar::F64(1.0),
            )
            .unwrap(),
        OverwriteOutcome::Incompatible
    );
    assert_eq!(
        nonunique
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        before
    );
    assert_eq!(cache.preparations(), 1);
    drop(held);

    let mut wrong_layout = Tensor::zeros(&runtime, Dtype::F64, [&space, &space], []).unwrap();
    let before = wrong_layout.data().to_vec();
    assert_eq!(
        context
            .try_contract_overwrite_into(
                &mut cache,
                &mut wrong_layout,
                &lhs,
                &rhs,
                &[1],
                &[0],
                Scalar::F64(1.0),
            )
            .unwrap(),
        OverwriteOutcome::Incompatible
    );
    assert_eq!(wrong_layout.data(), before);
    let fallback = lhs.contract(&rhs, &[1], &[0]).unwrap();
    assert_close(fallback.data(), expected.data(), 1e-12);

    let lhs_same_space =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 30_303).unwrap();
    let rhs_same_space =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 30_304).unwrap();
    let same_space_expected = lhs_same_space
        .contract(&rhs_same_space, &[1], &[0])
        .unwrap();
    let mut same_space_destination = same_space_expected.scale(f64::NAN).unwrap();
    let comparisons_before = cache.structural_comparisons();
    assert_eq!(
        context
            .try_contract_overwrite_into(
                &mut cache,
                &mut same_space_destination,
                &lhs_same_space,
                &rhs_same_space,
                &[1],
                &[0],
                Scalar::F64(1.0),
            )
            .unwrap(),
        OverwriteOutcome::Written
    );
    assert_eq!(cache.preparations(), 1);
    assert!(cache.structural_comparisons() > comparisons_before);
    assert_close(
        same_space_destination.data(),
        same_space_expected.data(),
        1e-12,
    );
    let comparisons_after = cache.structural_comparisons();
    assert_eq!(
        context
            .try_contract_overwrite_into(
                &mut cache,
                &mut same_space_destination,
                &lhs_same_space,
                &rhs_same_space,
                &[1],
                &[0],
                Scalar::F64(1.0),
            )
            .unwrap(),
        OverwriteOutcome::Written
    );
    assert_eq!(cache.structural_comparisons(), comparisons_after);
}

#[test]
fn checked_contract_rejections_preserve_destination_bits() {
    let runtime = Runtime::builder().build().unwrap();
    let foreign_runtime = Runtime::builder().build().unwrap();
    let space = Space::u1([(-1, 2), (0, 3), (1, 2)]);
    let lhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 30_305).unwrap();
    let rhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 30_306).unwrap();
    let expected = lhs.contract(&rhs, &[1], &[0]).unwrap();
    let mut context = TensorExecutionContext::for_runtime(&runtime).unwrap();
    let mut cache = ContractOverwriteCache::default();

    let mut wrong_alpha = expected.scale(f64::NAN).unwrap();
    let before = wrong_alpha
        .data()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert!(matches!(
        context.try_contract_overwrite_into(
            &mut cache,
            &mut wrong_alpha,
            &lhs,
            &rhs,
            &[1],
            &[0],
            Scalar::C64(Complex64::new(1.0, 0.0)),
        ),
        Err(Error::DtypeMismatch)
    ));
    assert_eq!(
        wrong_alpha
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        before
    );

    let lhs_c = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 30_308).unwrap();
    let rhs_c = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 30_309).unwrap();
    let mut wrong_dtype = lhs_c.contract(&rhs_c, &[1], &[0]).unwrap();
    let before = wrong_dtype
        .data_c64()
        .iter()
        .map(|value| (value.re.to_bits(), value.im.to_bits()))
        .collect::<Vec<_>>();
    assert_eq!(
        context
            .try_contract_overwrite_into(
                &mut cache,
                &mut wrong_dtype,
                &lhs,
                &rhs,
                &[1],
                &[0],
                Scalar::F64(1.0),
            )
            .unwrap(),
        OverwriteOutcome::Incompatible
    );
    assert_eq!(
        wrong_dtype
            .data_c64()
            .iter()
            .map(|value| (value.re.to_bits(), value.im.to_bits()))
            .collect::<Vec<_>>(),
        before
    );

    let foreign =
        Tensor::rand_with_seed(&foreign_runtime, Dtype::F64, [&space], [&space], 30_307).unwrap();
    let diagonal = lhs.svd_trunc(&Truncation::Full).unwrap().s;
    let lazy_adjoint = lhs.adjoint().unwrap();
    let cases = [(&foreign, &rhs), (&diagonal, &rhs), (&lazy_adjoint, &rhs)];
    for (case_lhs, case_rhs) in cases {
        let mut destination = expected.scale(f64::NAN).unwrap();
        let before = destination
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            context
                .try_contract_overwrite_into(
                    &mut cache,
                    &mut destination,
                    case_lhs,
                    case_rhs,
                    &[1],
                    &[0],
                    Scalar::F64(1.0),
                )
                .unwrap(),
            OverwriteOutcome::Incompatible
        );
        assert_eq!(
            destination
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before
        );
    }

    let mut aliased = lhs.clone();
    let before = aliased.data().to_vec();
    assert_eq!(
        context
            .try_contract_overwrite_into(
                &mut cache,
                &mut aliased,
                &lhs,
                &rhs,
                &[1],
                &[0],
                Scalar::F64(1.0),
            )
            .unwrap(),
        OverwriteOutcome::Incompatible
    );
    assert_eq!(aliased.data(), before);
}

#[test]
fn checked_permutation_cache_prepares_once_for_su3_c64() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::su3([((1, 0), 2), ((0, 1), 1)]).unwrap();
    let source =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space, &space], [&space], 30_311).unwrap();
    let expected = source.permute(&[1], &[2, 0]).unwrap();
    let mut destination = expected
        .scale_c64(Complex64::new(f64::NAN, f64::NAN))
        .unwrap();
    let mut context = TensorExecutionContext::for_runtime(&runtime).unwrap();
    let mut cache = PermuteOverwriteCache::default();

    for _ in 0..2 {
        assert_eq!(
            context
                .try_permute_overwrite_into(
                    &mut cache,
                    &mut destination,
                    &source,
                    &[1],
                    &[2, 0],
                    Scalar::C64(Complex64::new(1.0, 0.0)),
                )
                .unwrap(),
            OverwriteOutcome::Written
        );
        assert_close_c64(destination.data_c64(), expected.data_c64(), 1e-12);
    }
    assert_eq!(cache.preparations(), 1);

    let held = destination.clone();
    let nonunique_before = destination
        .data_c64()
        .iter()
        .map(|value| (value.re.to_bits(), value.im.to_bits()))
        .collect::<Vec<_>>();
    assert_eq!(
        context
            .try_permute_overwrite_into(
                &mut cache,
                &mut destination,
                &source,
                &[1],
                &[2, 0],
                Scalar::C64(Complex64::new(1.0, 0.0)),
            )
            .unwrap(),
        OverwriteOutcome::Incompatible
    );
    assert_eq!(
        destination
            .data_c64()
            .iter()
            .map(|value| (value.re.to_bits(), value.im.to_bits()))
            .collect::<Vec<_>>(),
        nonunique_before
    );
    drop(held);

    let before = destination.data_c64().to_vec();
    assert!(context
        .try_permute_overwrite_into(
            &mut cache,
            &mut destination,
            &source,
            &[4],
            &[2, 0],
            Scalar::C64(Complex64::new(1.0, 0.0)),
        )
        .is_err());
    assert_eq!(destination.data_c64(), before);
}

#[test]
fn checked_contract_cache_rebuilds_for_changed_spaces() {
    let runtime = Runtime::builder().build().unwrap();
    let first_space = Space::u1([(-1, 2), (0, 3), (1, 2)]);
    let second_space = Space::u1([(-1, 3), (0, 4), (1, 3)]);
    let mut context = TensorExecutionContext::for_runtime(&runtime).unwrap();
    let mut cache = ContractOverwriteCache::default();

    for (index, space) in [&first_space, &second_space].into_iter().enumerate() {
        let lhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [space],
            [space],
            30_321 + index as u64 * 2,
        )
        .unwrap();
        let rhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [space],
            [space],
            30_322 + index as u64 * 2,
        )
        .unwrap();
        let expected = lhs.contract(&rhs, &[1], &[0]).unwrap();
        let mut destination = expected.scale(f64::NAN).unwrap();

        assert_eq!(
            context
                .try_contract_overwrite_into(
                    &mut cache,
                    &mut destination,
                    &lhs,
                    &rhs,
                    &[1],
                    &[0],
                    Scalar::F64(1.0),
                )
                .unwrap(),
            OverwriteOutcome::Written
        );
        assert_eq!(cache.preparations(), index as u64 + 1);
        assert_close(destination.data(), expected.data(), 1e-12);
    }
}
