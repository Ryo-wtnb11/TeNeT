use tenet::prelude::*;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-11 * (1.0 + actual.abs().max(expected.abs())),
            "element {index}: actual={actual}, expected={expected}"
        );
    }
}

fn assert_matches_sequential(runtime: &Runtime, space: &Space, seed: u64) {
    let lhs =
        Tensor::rand_with_seed(runtime, Dtype::F64, [space, space], [space, space], seed).unwrap();
    let rhs = Tensor::rand_with_seed(
        runtime,
        Dtype::F64,
        [space, space],
        [space, space],
        seed + 1,
    )
    .unwrap();
    let lhs_axes = [3, 2];
    let rhs_axes = [0, 1];
    let output_axes = [2, 0, 3, 1];

    let default = lhs.contract(&rhs, &lhs_axes, &rhs_axes).unwrap();
    let expected = default
        .permute(&output_axes[..2], &output_axes[2..])
        .unwrap();
    let actual = lhs
        .contract_ordered(&rhs, &lhs_axes, &rhs_axes, &output_axes)
        .unwrap();
    let memoized = lhs
        .contract_ordered(&rhs, &lhs_axes, &rhs_axes, &output_axes)
        .unwrap();

    // What: crossed pAB uses the same oriented fusion-tree basis and reduced
    // block data as the literal TensorKit-style contract then permute sequence.
    assert_close(actual.data(), expected.data());
    assert_close(memoized.data(), expected.data());
    assert_eq!(actual.codomain_rank(), expected.codomain_rank());
    assert_eq!(actual.domain_rank(), expected.domain_rank());
    for axis in 0..actual.rank() {
        assert_eq!(actual.space(axis), expected.space(axis));
    }
}

#[test]
fn crossed_output_order_matches_sequential_for_multiplicity_free_rules() {
    let runtime = Runtime::builder()
        .dense_threads(1)
        .recoupling_threads(1)
        .build()
        .unwrap();
    let spaces = [
        Space::u1([(-2, 1), (0, 3), (1, 2)]),
        Space::su2([(0, 2), (1, 3), (2, 1)]).unwrap(),
        Space::fz2([(0, 2), (1, 3)]).unwrap(),
        Space::fz2_u1_su2([
            ((0, 0, 0), 2),
            ((1, -1, 1), 2),
            ((1, 1, 1), 1),
            ((0, 2, 2), 1),
        ])
        .unwrap(),
    ];
    for (index, space) in spaces.iter().enumerate() {
        assert_matches_sequential(&runtime, space, 224_100 + index as u64 * 10);
    }
}

#[test]
fn crossed_output_order_matches_sequential_for_lazy_complex_adjoint() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::u1([(-2, 1), (-1, 2), (1, 3)]);
    let parent =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space, &space], 224_201).unwrap();
    let lhs = parent.adjoint().unwrap();
    let rhs =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space, &space], 224_202).unwrap();
    let output_axes = [2, 0, 3, 1];

    let default = lhs.contract(&rhs, &[2], &[0]).unwrap();
    let expected = default
        .permute(&output_axes[..2], &output_axes[2..])
        .unwrap();
    let actual = lhs
        .contract_ordered(&rhs, &[2], &[0], &output_axes)
        .unwrap();

    // What: pAB does not lose the adjoint conjugation or asymmetric U(1)
    // dual-sector relabel while folding both into the prepared contraction.
    assert_eq!(actual.data_c64().len(), expected.data_c64().len());
    for (&actual, &expected) in actual.data_c64().iter().zip(expected.data_c64()) {
        assert!((actual - expected).norm() < 1.0e-11);
    }
}

#[test]
fn crossed_output_order_matches_sequential_for_rhs_and_both_lazy_adjoints() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::u1([(-2, 1), (-1, 2), (1, 3)]);
    let lhs_parent =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space, &space], 224_211).unwrap();
    let rhs_parent =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space, &space], 224_212).unwrap();
    let plain_lhs =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space, &space], [&space], 224_213).unwrap();
    let output_axes = [2, 0, 3, 1];

    for (lhs, rhs, lhs_axes, rhs_axes) in [
        (
            plain_lhs.clone(),
            rhs_parent.adjoint().unwrap(),
            vec![2],
            vec![0],
        ),
        (
            lhs_parent.adjoint().unwrap(),
            rhs_parent.adjoint().unwrap(),
            vec![2],
            vec![0],
        ),
    ] {
        let default = lhs.contract(&rhs, &lhs_axes, &rhs_axes).unwrap();
        let expected = default
            .permute(&output_axes[..2], &output_axes[2..])
            .unwrap();
        let actual = lhs
            .contract_ordered(&rhs, &lhs_axes, &rhs_axes, &output_axes)
            .unwrap();
        // What: crossed pAB keeps either operand's lazy conjugation folded into
        // the contraction seam without changing the oriented reduced blocks.
        assert_eq!(actual.data_c64().len(), expected.data_c64().len());
        for (&actual, &expected) in actual.data_c64().iter().zip(expected.data_c64()) {
            assert!((actual - expected).norm() < 1.0e-11);
        }
    }
}

#[test]
fn ordered_contract_preserves_output_validation_and_zero_splits() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::fz2([(0, 1), (1, 2)]).unwrap();
    let lhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 224_301).unwrap();
    let rhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 224_302).unwrap();

    let duplicate = lhs.contract_ordered(&rhs, &[1], &[0], &[0, 0]).unwrap_err();
    let out_of_range = lhs.contract_ordered(&rhs, &[1], &[0], &[0, 2]).unwrap_err();
    let expected_duplicate = lhs
        .contract(&rhs, &[1], &[0])
        .unwrap()
        .permute(&[0], &[0])
        .unwrap_err();
    let expected_out_of_range = lhs
        .contract(&rhs, &[1], &[0])
        .unwrap()
        .permute(&[0], &[2])
        .unwrap_err();
    assert_eq!(duplicate, expected_duplicate);
    assert_eq!(out_of_range, expected_out_of_range);

    let bad_len = lhs.contract_ordered(&rhs, &[1], &[0], &[0]).unwrap_err();
    assert_eq!(
        bad_len.to_string(),
        "invalid argument: output axis list length 1 does not match open rank 2"
    );

    let scalar_lhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [], [&space], 224_303).unwrap();
    let two_domain_rhs =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space, &space], 224_304).unwrap();
    let zero_lhs_split = scalar_lhs
        .contract_ordered(&two_domain_rhs, &[0], &[0], &[1, 0])
        .unwrap();
    let zero_lhs_expected = scalar_lhs
        .contract(&two_domain_rhs, &[0], &[0])
        .unwrap()
        .permute(&[], &[1, 0])
        .unwrap();
    assert_close(zero_lhs_split.data(), zero_lhs_expected.data());
}

#[test]
fn ordered_contract_preserves_contraction_error_precedence() {
    let runtime = Runtime::builder().build().unwrap();
    let lhs_space = Space::u1([(-1, 1), (0, 2), (1, 1)]);
    let rhs_space = Space::u1([(-1, 1), (0, 3), (1, 1)]);
    let lhs =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&lhs_space], [&lhs_space], 224_321).unwrap();
    let rhs =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&rhs_space], [&rhs_space], 224_322).unwrap();

    let expected = lhs.contract(&rhs, &[1], &[0]).unwrap_err();
    let actual = lhs.contract_ordered(&rhs, &[1], &[0], &[0]).unwrap_err();
    // What: pAB validation remains after contraction compatibility, matching
    // the historical contract-then-permute API when both inputs are invalid.
    assert_eq!(actual, expected);
}

#[test]
fn ordered_contract_preserves_dtype_precedence_over_bad_pab_and_space() {
    let runtime = Runtime::builder().build().unwrap();
    let lhs_space = Space::u1([(-1, 1), (0, 2), (1, 1)]);
    let incompatible_rhs_space = Space::u1([(-1, 1), (0, 3), (1, 1)]);
    let lhs =
        Tensor::rand_with_seed(&runtime, Dtype::F64, [&lhs_space], [&lhs_space], 224_331).unwrap();
    let compatible_rhs =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&lhs_space], [&lhs_space], 224_332).unwrap();
    let incompatible_rhs = Tensor::rand_with_seed(
        &runtime,
        Dtype::C64,
        [&incompatible_rhs_space],
        [&incompatible_rhs_space],
        224_333,
    )
    .unwrap();

    // What: outer host-dense preconditions stay authoritative. Neither a bad
    // pAB nor a contracted-leg degeneracy mismatch may hide DtypeMismatch.
    assert_eq!(
        lhs.contract_ordered(&compatible_rhs, &[1], &[0], &[0])
            .unwrap_err(),
        Error::DtypeMismatch
    );
    assert_eq!(
        lhs.contract_ordered(&incompatible_rhs, &[1], &[0], &[0])
            .unwrap_err(),
        Error::DtypeMismatch
    );
}

#[test]
fn ordered_contract_parallel_replay_matches_serial() {
    let serial = Runtime::builder().recoupling_threads(1).build().unwrap();
    let parallel = Runtime::builder().recoupling_threads(2).build().unwrap();
    let serial_space = Space::su2([(0, 2), (1, 3), (2, 2)]).unwrap();
    let parallel_space = Space::su2([(0, 2), (1, 3), (2, 2)]).unwrap();
    let run = |runtime: &Runtime, space: &Space| {
        let lhs =
            Tensor::rand_with_seed(runtime, Dtype::F64, [space, space], [space, space], 224_401)
                .unwrap();
        let rhs =
            Tensor::rand_with_seed(runtime, Dtype::F64, [space, space], [space, space], 224_402)
                .unwrap();
        lhs.contract_ordered(&rhs, &[3, 2], &[0, 1], &[2, 0, 3, 1])
            .unwrap()
    };

    // What: compiled output replay is deterministic across serial and
    // parallel recoupling executors.
    let serial_output = run(&serial, &serial_space);
    let parallel_output = run(&parallel, &parallel_space);
    assert_close(serial_output.data(), parallel_output.data());
}
