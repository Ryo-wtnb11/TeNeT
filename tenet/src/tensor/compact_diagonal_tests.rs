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

fn dense_oracle(diagonal: &Tensor) -> Tensor {
    diagonal.clone().densified_if_diagonal()
}

fn assert_compact_unmaterialized(tensor: &Tensor) {
    assert!(matches!(tensor.data.as_ref(), Data::Diagonal(_)));
    assert!(tensor.materialized.get().is_none());
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
fn compact_storage_local_operations_match_dense_oracles() {
    // What: every storage-local unary/reduction/binary operation preserves the
    // compact spectrum and matches the former dense path across supported rules.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    for space in [
        Space::u1([(-2, 1), (1, 3)]),
        Space::fz2([(0, 2), (1, 3)]),
        Space::su2([(0, 2), (1, 3), (2, 1)]),
        product_space(),
    ] {
        let lhs = real_diagonal(&rt, &space, 801);
        let rhs = real_diagonal(&rt, &space, 802);
        let lhs_dense = dense_oracle(&lhs);
        let rhs_dense = dense_oracle(&rhs);

        let adjoint = lhs.adjoint().unwrap();
        assert!(Arc::ptr_eq(&adjoint.data, &lhs.data));
        let scaled = lhs.scale(-1.25).unwrap();
        let added = lhs.add(&rhs, 0.75, -0.5).unwrap();
        let widened = lhs.to_c64();
        for tensor in [&lhs, &rhs, &adjoint, &scaled, &added, &widened] {
            assert_compact_unmaterialized(tensor);
        }
        assert_tensor_close(&adjoint, &lhs_dense.adjoint().unwrap());
        assert_tensor_close(&scaled, &lhs_dense.scale(-1.25).unwrap());
        assert_tensor_close(&added, &lhs_dense.add(&rhs_dense, 0.75, -0.5).unwrap());
        assert_tensor_close(&widened, &lhs_dense.to_c64());
        assert!((lhs.norm().unwrap() - lhs_dense.norm().unwrap()).abs() < 1e-11);
        assert!((lhs.norm_inf().unwrap() - lhs_dense.norm_inf().unwrap()).abs() < 1e-11);
        assert!(
            (lhs.inner(&rhs).unwrap().to_c64() - lhs_dense.inner(&rhs_dense).unwrap().to_c64())
                .norm()
                < 1e-11
        );
        assert!(lhs.materialized.get().is_none());
        assert!(rhs.materialized.get().is_none());

        for legs in [&[0][..], &[1][..], &[0, 1][..]] {
            let twisted = lhs.twist(legs).unwrap();
            assert_compact_unmaterialized(&twisted);
            assert_tensor_close(&twisted, &lhs_dense.twist(legs).unwrap());
            assert!(lhs.materialized.get().is_none());
        }
    }
}

#[test]
fn compact_c64_operations_match_dense_oracles() {
    // What: real-valued and genuinely complex c64 spectra follow dense dtype,
    // conjugation, axpby, and inner-product semantics without materialization.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    for space in [
        Space::u1([(-2, 1), (1, 3)]),
        Space::fz2([(0, 2), (1, 3)]),
        Space::su2([(0, 2), (1, 3), (2, 1)]),
        product_space(),
    ] {
        for (lhs, rhs) in [
            (
                real_c64_diagonal(&rt, &space, 811),
                real_c64_diagonal(&rt, &space, 812),
            ),
            (
                real_c64_diagonal(&rt, &space, 813),
                complex_diagonal(&rt, &space, 814),
            ),
            (
                complex_diagonal(&rt, &space, 815),
                real_c64_diagonal(&rt, &space, 816),
            ),
            (
                complex_diagonal(&rt, &space, 817),
                complex_diagonal(&rt, &space, 818),
            ),
        ] {
            let lhs_dense = dense_oracle(&lhs);
            let rhs_dense = dense_oracle(&rhs);
            let factor = Complex64::new(-0.5, 0.75);
            let alpha = Complex64::new(0.25, -0.5);
            let beta = Complex64::new(-0.75, 0.125);

            let adjoint = lhs.adjoint().unwrap();
            assert_eq!(
                Arc::ptr_eq(&adjoint.data, &lhs.data),
                matches!(lhs.data.as_ref(), Data::Diagonal(DiagonalData::RealC64(_)))
            );
            let scaled = lhs.scale_c64(factor).unwrap();
            let added_real = lhs.add(&rhs, 0.75, -0.5).unwrap();
            let added = lhs.add_c64(&rhs, alpha, beta).unwrap();
            let widened = lhs.to_c64();
            for tensor in [&lhs, &rhs, &adjoint, &scaled, &added_real, &added, &widened] {
                assert_compact_unmaterialized(tensor);
            }
            assert_tensor_close(&adjoint, &lhs_dense.adjoint().unwrap());
            assert_tensor_close(&scaled, &lhs_dense.scale_c64(factor).unwrap());
            assert_tensor_close(&added_real, &lhs_dense.add(&rhs_dense, 0.75, -0.5).unwrap());
            assert_tensor_close(&added, &lhs_dense.add_c64(&rhs_dense, alpha, beta).unwrap());
            assert_tensor_close(&widened, &lhs_dense.to_c64());
            assert!((lhs.norm().unwrap() - lhs_dense.norm().unwrap()).abs() < 1e-11);
            assert!((lhs.norm_inf().unwrap() - lhs_dense.norm_inf().unwrap()).abs() < 1e-11);
            assert!(
                (lhs.inner(&rhs).unwrap().to_c64() - lhs_dense.inner(&rhs_dense).unwrap().to_c64())
                    .norm()
                    < 1e-11
            );
            assert!(lhs.materialized.get().is_none());
            assert!(rhs.materialized.get().is_none());

            for legs in [&[0][..], &[1][..], &[0, 1][..]] {
                let twisted = lhs.twist(legs).unwrap();
                assert_compact_unmaterialized(&twisted);
                assert_tensor_close(&twisted, &lhs_dense.twist(legs).unwrap());
                assert!(lhs.materialized.get().is_none());
            }
        }
    }
}

#[test]
fn compact_dense_binary_operations_scatter_without_materializing_source() {
    // What: diagonal+dense and dense+diagonal add/inner consume only the
    // diagonal entries, including a lazy-adjoint dense operand.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    for space in [
        Space::u1([(-2, 1), (1, 3)]),
        Space::fz2([(0, 2), (1, 3)]),
        Space::su2([(0, 2), (1, 3), (2, 1)]),
        product_space(),
    ] {
        let diagonal = real_diagonal(&rt, &space, 821);
        let diagonal_dense = dense_oracle(&diagonal);
        let dense = Tensor::rand_with_seed(&rt, Dtype::F64, [&space], [&space], 822).unwrap();
        let lazy = dense.adjoint().unwrap();
        for operand in [&dense, &lazy] {
            let operand_dense = Tensor {
                rt: operand.rt.clone(),
                space: Arc::clone(&operand.space),
                data: Arc::new(operand.coupled_data().clone()),
                adjoint_source: None,
                materialized: OnceLock::new(),
            };
            assert_tensor_close(
                &diagonal.add(operand, 0.75, -0.5).unwrap(),
                &diagonal_dense.add(&operand_dense, 0.75, -0.5).unwrap(),
            );
            assert_tensor_close(
                &operand.add(&diagonal, 0.75, -0.5).unwrap(),
                &operand_dense.add(&diagonal_dense, 0.75, -0.5).unwrap(),
            );
            assert!(
                (diagonal.inner(operand).unwrap().to_c64()
                    - diagonal_dense.inner(&operand_dense).unwrap().to_c64())
                .norm()
                    < 1e-11
            );
            assert!(
                (operand.inner(&diagonal).unwrap().to_c64()
                    - operand_dense.inner(&diagonal_dense).unwrap().to_c64())
                .norm()
                    < 1e-11
            );
            assert!(diagonal.materialized.get().is_none());
        }

        let dense = Tensor::rand_with_seed(&rt, Dtype::C64, [&space], [&space], 824).unwrap();
        let alpha = Complex64::new(0.25, -0.5);
        let beta = Complex64::new(-0.75, 0.125);
        for diagonal in [
            real_c64_diagonal(&rt, &space, 823),
            complex_diagonal(&rt, &space, 825),
        ] {
            let diagonal_dense = dense_oracle(&diagonal);
            assert_tensor_close(
                &diagonal.add(&dense, 0.75, -0.5).unwrap(),
                &diagonal_dense.add(&dense, 0.75, -0.5).unwrap(),
            );
            assert_tensor_close(
                &dense.add(&diagonal, 0.75, -0.5).unwrap(),
                &dense.add(&diagonal_dense, 0.75, -0.5).unwrap(),
            );
            assert_tensor_close(
                &diagonal.add_c64(&dense, alpha, beta).unwrap(),
                &diagonal_dense.add_c64(&dense, alpha, beta).unwrap(),
            );
            assert_tensor_close(
                &dense.add_c64(&diagonal, alpha, beta).unwrap(),
                &dense.add_c64(&diagonal_dense, alpha, beta).unwrap(),
            );
            assert!(
                (diagonal.inner(&dense).unwrap().to_c64()
                    - diagonal_dense.inner(&dense).unwrap().to_c64())
                .norm()
                    < 1e-11
            );
            assert!(
                (dense.inner(&diagonal).unwrap().to_c64()
                    - dense.inner(&diagonal_dense).unwrap().to_c64())
                .norm()
                    < 1e-11
            );
            assert!(diagonal.materialized.get().is_none());
        }
    }
}

#[test]
fn compact_storage_local_operations_preserve_validation_precedence() {
    // What: storage dispatch does not bypass runtime, space, or dtype checks.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_rt = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::u1([(0, 2), (1, 1)]);
    let other_space = Space::u1([(0, 3), (1, 1)]);
    let diagonal = real_diagonal(&rt, &space, 831);
    let foreign = real_diagonal(&foreign_rt, &space, 832);
    let mismatched = real_diagonal(&rt, &other_space, 833);
    let complex = real_c64_diagonal(&rt, &space, 834);

    assert!(matches!(
        diagonal.add(&foreign, 1.0, 1.0),
        Err(Error::RuntimeMismatch)
    ));
    assert!(matches!(
        diagonal.add(&mismatched, 1.0, 1.0),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        diagonal.add(&complex, 1.0, 1.0),
        Err(Error::DtypeMismatch)
    ));
    assert!(matches!(
        diagonal.scale_c64(Complex64::new(1.0, 1.0)),
        Err(Error::DtypeMismatch)
    ));
}

#[test]
fn identity_compact_twist_shares_storage() {
    // What: identity twists on bosonic/all-even compact spectra are O(1),
    // while an odd fermionic sector still takes the compact O(r) value path.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    for space in [
        Space::u1([(-1, 2), (2, 1)]),
        Space::su2([(0, 2), (1, 1)]),
        Space::fz2([(0, 3)]),
    ] {
        let diagonal = real_diagonal(&rt, &space, 841);
        let twisted = diagonal.twist(&[0]).unwrap();
        assert!(Arc::ptr_eq(&twisted.data, &diagonal.data));
        assert_compact_unmaterialized(&diagonal);
        assert_compact_unmaterialized(&twisted);
    }

    let odd = Space::fz2([(1, 3)]);
    let diagonal = real_diagonal(&rt, &odd, 842);
    let twisted = diagonal.twist(&[0]).unwrap();
    assert!(!Arc::ptr_eq(&twisted.data, &diagonal.data));
    assert_compact_unmaterialized(&diagonal);
    assert_compact_unmaterialized(&twisted);
}

#[test]
fn su3_compact_storage_ops_and_fallback_boundaries_are_explicit() {
    // What: storage-local SU(3) operations and ordinary trace remain compact;
    // norm still uses the generic block-semantic materialization boundary.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::su3([((1, 0), 2), ((0, 1), 1)]).unwrap();
    for dtype in [Dtype::F64, Dtype::C64] {
        let source = Tensor::rand_with_seed(&rt, dtype, [&space], [&space], 851).unwrap();
        let diagonal = source.svd_compact().unwrap().1;
        let adjoint = diagonal.adjoint().unwrap();
        let scaled = diagonal.scale(-0.5).unwrap();
        let added = diagonal.add(&diagonal, 0.75, -0.5).unwrap();
        for tensor in [&diagonal, &adjoint, &scaled, &added] {
            assert_compact_unmaterialized(tensor);
        }
        assert!(Arc::ptr_eq(&adjoint.data, &diagonal.data));

        let norm_input = diagonal.clone();
        assert!(norm_input.materialized.get().is_none());
        assert!(norm_input.norm().unwrap().is_finite());
        assert!(norm_input.materialized.get().is_some());

        let trace_input = diagonal.clone();
        assert!(trace_input.materialized.get().is_none());
        assert!(trace_input.tr().unwrap().to_c64().norm().is_finite());
        assert!(trace_input.materialized.get().is_none());
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
