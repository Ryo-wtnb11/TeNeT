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

fn fixed_diagonal(
    rt: &Runtime,
    space: &Space,
    plus: SectorId,
    plus_value: f64,
    minus_value: f64,
) -> Tensor {
    Tensor::from_block_fn(rt, [space], [space], |key, _| match key {
        BlockKey::FusionTree(key) if key.codomain_uncoupled()[0] == plus => plus_value,
        _ => minus_value,
    })
    .unwrap()
    .svd_compact()
    .unwrap()
    .1
}

fn scalar_block_by_codomain_sector(tensor: &Tensor, sector: SectorId) -> f64 {
    let Data::F64(data) = tensor.coupled_data() else {
        panic!("expected f64 tensor")
    };
    let structure = tensor.space.structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        if key.codomain_uncoupled()[0] == sector {
            assert_eq!(block.shape().iter().product::<usize>(), 1);
            return data[block.offset()];
        }
    }
    panic!("missing codomain sector {sector:?}")
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
        let dense =
            Tensor::rand_with_seed(&rt, Dtype::C64, [&space, &space], [&space], 712).unwrap();
        let lazy = dense.adjoint().unwrap();
        let mut successes = 0;

        for (operand, diagonal_axes, operand_axes, stays_compact) in [
            (&dense, [1], [0], true),
            (&lazy, [1], [0], true),
            (&lazy, [0], [2], false),
        ] {
            let diagonal = complex_diagonal(&rt, &space, 711);
            let actual = diagonal.contract(operand, &diagonal_axes, &operand_axes);
            assert_eq!(diagonal.materialized.get().is_none(), stays_compact);
            let expected =
                diagonal
                    .densified_if_diagonal()
                    .contract(operand, &diagonal_axes, &operand_axes);
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

#[test]
fn nonselfdual_odd_product_routes_match_dense_oracles_and_storage() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::product([((1, 1), 1), ((-1, 1), 1)]).unwrap();
    let plus = Space::product([((1, 1), 1)]).unwrap().sectors[0].0;
    let minus = Space::product([((-1, 1), 1)]).unwrap().sectors[0].0;
    let probe = fixed_diagonal(&rt, &space, plus, 2.0, 3.0);
    let homspace = probe.space.homspace();
    assert!(!homspace.codomain().legs()[0].is_dual());
    assert!(!homspace.domain().legs()[0].is_dual());
    assert_eq!(homspace.external_axis_is_dual(0), Some(false));
    assert_eq!(homspace.external_axis_is_dual(1), Some(true));

    let mut successful_noncanonical = 0;
    for lhs_axis in 0..2 {
        for rhs_axis in 0..2 {
            let lhs = fixed_diagonal(&rt, &space, plus, 2.0, 3.0);
            let rhs = fixed_diagonal(&rt, &space, plus, 5.0, 7.0);
            let actual = lhs.contract(&rhs, &[lhs_axis], &[rhs_axis]);
            let oracle_lhs = fixed_diagonal(&rt, &space, plus, 2.0, 3.0).densified_if_diagonal();
            let oracle_rhs = fixed_diagonal(&rt, &space, plus, 5.0, 7.0).densified_if_diagonal();
            let expected = oracle_lhs.contract(&oracle_rhs, &[lhs_axis], &[rhs_axis]);
            match (actual, expected) {
                (Ok(actual), Ok(expected)) => {
                    assert!(expected.norm().unwrap() > 0.0);
                    assert_tensor_close(&actual, &expected);
                    assert_eq!(
                        matches!(actual.data.as_ref(), Data::Diagonal(_)),
                        lhs_axis == 1 && rhs_axis == 0,
                        "only canonical diagonal composition may stay compact: lhs={lhs_axis}, rhs={rhs_axis}"
                    );
                    if lhs_axis == 1 && rhs_axis == 0 {
                        assert_eq!(scalar_block_by_codomain_sector(&expected, minus), 21.0);
                        assert_eq!(scalar_block_by_codomain_sector(&expected, plus), 10.0);
                    } else {
                        successful_noncanonical += 1;
                    }
                }
                (Err(actual), Err(expected)) => {
                    assert_eq!(actual.to_string(), expected.to_string())
                }
                pair => panic!(
                    "compact/dense route disagreement: lhs={lhs_axis}, rhs={rhs_axis}: {pair:?}"
                ),
            }
        }
    }
    assert!(
        successful_noncanonical > 0,
        "fixture must exercise a valid noncanonical dense fallback"
    );

    let dual = space.dual();
    let dense_dual = Tensor::from_block_fn(&rt, [&dual], [&dual], |key, _| match key {
        BlockKey::FusionTree(key) if key.codomain_uncoupled()[0] == plus => 5.0,
        _ => 7.0,
    })
    .unwrap();
    let diagonal = fixed_diagonal(&rt, &space, plus, 2.0, 3.0);
    let actual = dense_dual.contract(&diagonal, &[1], &[1]).unwrap();
    assert!(!matches!(actual.data.as_ref(), Data::Diagonal(_)));
    assert!(diagonal.materialized.get().is_some());
    let expected = dense_dual
        .contract(&diagonal.densified_if_diagonal(), &[1], &[1])
        .unwrap();
    assert_eq!(scalar_block_by_codomain_sector(&expected, minus), 14.0);
    assert_eq!(scalar_block_by_codomain_sector(&expected, plus), 15.0);
    assert_tensor_close(&actual, &expected);

    let diagonal = fixed_diagonal(&rt, &space, plus, 2.0, 3.0);
    let actual = diagonal.contract(&dense_dual, &[1], &[1]).unwrap();
    assert!(!matches!(actual.data.as_ref(), Data::Diagonal(_)));
    assert!(diagonal.materialized.get().is_some());
    let expected = diagonal
        .densified_if_diagonal()
        .contract(&dense_dual, &[1], &[1])
        .unwrap();
    assert!(expected.norm().unwrap() > 0.0);
    assert_tensor_close(&actual, &expected);

    let dense = Tensor::from_block_fn(&rt, [&space], [&space], |key, _| match key {
        BlockKey::FusionTree(key) if key.codomain_uncoupled()[0] == plus => 5.0,
        _ => 7.0,
    })
    .unwrap();

    let diagonal = fixed_diagonal(&rt, &space, plus, 2.0, 3.0);
    let actual = diagonal.compose(&dense).unwrap();
    let expected = fixed_diagonal(&rt, &space, plus, 2.0, 3.0)
        .densified_if_diagonal()
        .compose(&dense)
        .unwrap();
    assert!(expected.norm().unwrap() > 0.0);
    assert_tensor_close(&actual, &expected);

    let diagonal = fixed_diagonal(&rt, &space, plus, 2.0, 3.0);
    let actual = dense.compose(&diagonal).unwrap();
    let expected = dense
        .compose(&fixed_diagonal(&rt, &space, plus, 2.0, 3.0).densified_if_diagonal())
        .unwrap();
    assert!(expected.norm().unwrap() > 0.0);
    assert_tensor_close(&actual, &expected);
}
