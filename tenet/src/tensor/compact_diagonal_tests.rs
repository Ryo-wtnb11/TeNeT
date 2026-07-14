use super::*;

fn product_space() -> Space {
    Space::product([((-1, 1), 2), ((0, 0), 3), ((1, 1), 2)]).unwrap()
}

fn real_diagonal(rt: &Runtime, space: &Space, seed: u64) -> Tensor {
    let source = Tensor::rand_with_seed(rt, Dtype::F64, [space], [space], seed).unwrap();
    source.svd_compact().unwrap().1
}

fn complex_diagonal(rt: &Runtime, space: &Space, seed: u64) -> Tensor {
    let source = Tensor::rand_with_seed(rt, Dtype::C64, [space], [space], seed).unwrap();
    let diagonal = source.svd_compact().unwrap().1;
    let real = match diagonal.data.as_ref() {
        Data::Diagonal(DiagonalData::RealC64(spectrum)) => spectrum,
        storage => panic!("expected compact real-c64 spectrum, got {storage:?}"),
    };
    let spectrum = real
        .iter()
        .map(|entry| SectorSpectrum {
            sector: entry.sector,
            values: entry
                .values
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    if index == 0 {
                        Complex64::new(0.0, 0.0)
                    } else {
                        Complex64::new(value, -(index as f64 + 1.0))
                    }
                })
                .collect(),
        })
        .collect();
    diagonal.from_diagonal_complex_spectrum(spectrum).unwrap()
}

fn real_c64_diagonal(rt: &Runtime, space: &Space, seed: u64) -> Tensor {
    let source = Tensor::rand_with_seed(rt, Dtype::C64, [space], [space], seed).unwrap();
    source.svd_compact().unwrap().1
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.space, expected.space);
    assert_eq!(actual.dtype(), expected.dtype());
    match (actual.coupled_data(), expected.coupled_data()) {
        (Data::F64(actual), Data::F64(expected)) => {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-11);
            }
        }
        (Data::C64(actual), Data::C64(expected)) => {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((*actual - *expected).norm() < 1e-11);
            }
        }
        pair => panic!("dtype mismatch: {pair:?}"),
    }
}

#[test]
fn one_axis_diagonal_products_stay_compact() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    for space in [
        Space::u1([(-1, 2), (0, 3), (1, 2)]),
        Space::su2([(0, 2), (1, 2), (2, 1)]),
        product_space(),
    ] {
        for (lhs, rhs) in [
            (
                real_diagonal(&rt, &space, 701),
                real_diagonal(&rt, &space, 702),
            ),
            (
                complex_diagonal(&rt, &space, 703),
                complex_diagonal(&rt, &space, 704),
            ),
            (
                real_c64_diagonal(&rt, &space, 705),
                complex_diagonal(&rt, &space, 706),
            ),
        ] {
            let actual = lhs.contract(&rhs, &[1], &[0]).unwrap();
            assert!(
                matches!(actual.data.as_ref(), Data::Diagonal(_)),
                "one-axis diagonal product must preserve compact storage"
            );
            assert!(lhs.materialized.get().is_none());
            assert!(rhs.materialized.get().is_none());

            let composed = lhs.compose(&rhs).unwrap();
            assert!(matches!(composed.data.as_ref(), Data::Diagonal(_)));
            assert!(lhs.materialized.get().is_none());
            assert!(rhs.materialized.get().is_none());

            let expected = lhs
                .densified_if_diagonal()
                .contract(&rhs.densified_if_diagonal(), &[1], &[0])
                .unwrap();
            assert_tensor_close(&actual, &expected);
            assert_tensor_close(&composed, &expected);
        }
    }
}

#[test]
fn complex_diagonal_scales_dense_and_lazy_operands() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    for space in [
        Space::u1([(-1, 2), (0, 3), (1, 2)]),
        Space::su2([(0, 2), (1, 2), (2, 1)]),
        product_space(),
    ] {
        let diagonal = complex_diagonal(&rt, &space, 711);
        let dense =
            Tensor::rand_with_seed(&rt, Dtype::C64, [&space, &space], [&space], 712).unwrap();
        let lazy = dense.adjoint().unwrap();
        let mut successes = 0;

        for (operand, diagonal_axes, operand_axes) in
            [(&dense, [1], [0]), (&lazy, [1], [0]), (&lazy, [0], [2])]
        {
            let expected =
                diagonal
                    .densified_if_diagonal()
                    .contract(operand, &diagonal_axes, &operand_axes);
            let actual = diagonal.contract(operand, &diagonal_axes, &operand_axes);
            match (actual, expected) {
                (Ok(actual), Ok(expected)) => {
                    assert_tensor_close(&actual, &expected);
                    successes += 1;
                }
                (Err(actual), Err(expected)) => {
                    assert_eq!(actual.to_string(), expected.to_string())
                }
                pair => panic!("compact/dense route disagreement: {pair:?}"),
            }
        }
        assert!(
            successes > 0,
            "fixture must exercise a successful compact route"
        );
    }
}

#[test]
fn ambiguous_diagonal_contractions_keep_dense_fallback() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::u1([(0, 4)]);
    let lhs = real_diagonal(&rt, &space, 721);
    let rhs = real_diagonal(&rt, &space, 722);

    let outer = lhs.contract(&rhs, &[], &[]).unwrap();
    assert!(!matches!(outer.data.as_ref(), Data::Diagonal(_)));
}

#[test]
fn diagonal_fast_paths_preserve_world_checks_and_dense_routes() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_rt = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::u1([(0, 3)]);
    let diagonal = real_diagonal(&rt, &space, 731);
    let foreign = real_diagonal(&foreign_rt, &space, 732);
    assert!(matches!(
        diagonal.compose(&foreign),
        Err(Error::RuntimeMismatch)
    ));

    let lhs = Tensor::rand_with_seed(&rt, Dtype::F64, [&space], [&space], 733).unwrap();
    let rhs = Tensor::rand_with_seed(&rt, Dtype::F64, [&space], [&space], 734).unwrap();
    let expected = lhs
        .densified_if_diagonal()
        .contract(&rhs.densified_if_diagonal(), &[1], &[0]);
    let actual = lhs.contract(&rhs, &[1], &[0]);
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => assert_tensor_close(&actual, &expected),
        pair => panic!("ordinary dense route changed: {pair:?}"),
    }
}
