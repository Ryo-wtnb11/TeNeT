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
