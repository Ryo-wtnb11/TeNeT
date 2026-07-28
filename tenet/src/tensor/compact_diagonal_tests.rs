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
    let real = match diagonal.stored_data() {
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

fn su3_diagonal(rt: &Runtime, dtype: Dtype, space: &Space, seed: u64) -> Tensor {
    if dtype == Dtype::F64 {
        real_diagonal(rt, space, seed)
    } else {
        real_c64_diagonal(rt, space, seed)
    }
}

fn compact_su3_inner_oracle(lhs: &Tensor, rhs: &Tensor) -> Complex64 {
    fn reduce<VL, VR>(
        rule: &Su3FusionRule,
        lhs: &[SectorSpectrum<VL>],
        rhs: &[SectorSpectrum<VR>],
        map_lhs: impl Fn(VL) -> Complex64,
        map_rhs: impl Fn(VR) -> Complex64,
    ) -> Complex64
    where
        VL: Copy,
        VR: Copy,
    {
        assert_eq!(lhs.len(), rhs.len());
        let mut total = Complex64::new(0.0, 0.0);
        for (lhs, rhs) in lhs.iter().zip(rhs) {
            assert_eq!(lhs.sector, rhs.sector);
            assert_eq!(lhs.values.len(), rhs.values.len());
            let sqrt = rule.sqrt_dim_scalar(lhs.sector);
            let weight = sqrt * sqrt;
            let mut partial = Complex64::new(0.0, 0.0);
            for (&lhs, &rhs) in lhs.values.iter().zip(&rhs.values) {
                partial += map_lhs(lhs).conj() * map_rhs(rhs);
            }
            total += weight * partial;
        }
        total
    }

    let rule = lhs.su3_rule();
    match (lhs.stored_data(), rhs.stored_data()) {
        (
            Data::Diagonal(DiagonalData::RealF64(lhs)),
            Data::Diagonal(DiagonalData::RealF64(rhs)),
        ) => reduce(
            rule,
            lhs,
            rhs,
            |value| Complex64::new(value, 0.0),
            |value| Complex64::new(value, 0.0),
        ),
        (
            Data::Diagonal(DiagonalData::RealC64(lhs)),
            Data::Diagonal(DiagonalData::RealC64(rhs)),
        ) => reduce(
            rule,
            lhs,
            rhs,
            |value| Complex64::new(value, 0.0),
            |value| Complex64::new(value, 0.0),
        ),
        pair => panic!("expected matching compact SU(3) diagonal storage, got {pair:?}"),
    }
}

fn assert_svd_trunc_builds_one_compact_diagonal_layout(space: Space, dtype: Dtype, seed: u64) {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let tensor = Tensor::rand_with_seed(&rt, dtype, [&space], [&space], seed).unwrap();
    DIAGONAL_RESULT_LAYOUT_BUILDS.with(|builds| builds.set(Some(0)));
    let result = tensor.svd_trunc(&Truncation::rank(2)).unwrap();
    assert_eq!(
        DIAGONAL_RESULT_LAYOUT_BUILDS.with(|builds| builds.replace(None)),
        Some(1)
    );
    assert!(matches!(result.s.stored_data(), Data::Diagonal(_)));
}

#[test]
fn public_svd_trunc_builds_one_compact_diagonal_layout_for_both_rule_paths() {
    // What: multiplicity-free and Generic truncation each assemble one compact
    // diagonal result after their factor-only matrixalgebra call.
    assert_svd_trunc_builds_one_compact_diagonal_layout(
        Space::u1([(-1, 2), (0, 3), (1, 2)]),
        Dtype::F64,
        428_001,
    );
    assert_svd_trunc_builds_one_compact_diagonal_layout(
        Space::su3([((1, 0), 2), ((0, 1), 2)]).unwrap(),
        Dtype::C64,
        428_002,
    );
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor) {
    let actual = actual.materialized_tensor().unwrap();
    let expected = expected.materialized_tensor().unwrap();
    assert_eq!(actual.ordinary_body().space, expected.ordinary_body().space);
    assert_eq!(actual.dtype(), expected.dtype());
    match (
        actual.coupled_data().unwrap(),
        expected.coupled_data().unwrap(),
    ) {
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
    assert!(matches!(tensor.stored_data(), Data::Diagonal(_)));
    assert!(!tensor.has_cached_materialization());
}

#[test]
fn asymmetric_odd_product_swap_preserves_compact_c64_storage_and_dense_values() {
    // What: ordinary permutation and planar transpose keep a non-self-dual,
    // fermionic diagonal compact while matching the dense route's stored
    // diagonal values and numerical structural zeros.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::product([((-2, 1), 2), ((1, 1), 3)]).unwrap();

    let compact_spectrum_bits = |tensor: &Tensor| {
        let Data::Diagonal(DiagonalData::C64(spectrum)) = tensor.stored_data() else {
            panic!("expected compact c64 spectrum")
        };
        spectrum
            .iter()
            .map(|entry| {
                (
                    entry.sector,
                    entry
                        .values
                        .iter()
                        .map(|value| (value.re.to_bits(), value.im.to_bits()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    let dense_diagonal_values = |tensor: &Tensor| {
        let data = tensor.try_data_c64().unwrap();
        let structure = tensor.ordinary_body().space.structure();
        (0..structure.block_count())
            .map(|index| {
                let block = structure.block(index).unwrap();
                let BlockKey::FusionTree(key) = block.key() else {
                    panic!("expected fusion-tree block")
                };
                let [rows, columns] = block.shape() else {
                    panic!("expected matrix block")
                };
                let [row_stride, column_stride] = block.strides() else {
                    panic!("expected matrix strides")
                };
                for column in 0..*columns {
                    for row in 0..*rows {
                        if row != column {
                            assert_eq!(
                                data[block.offset() + row * row_stride + column * column_stride],
                                Complex64::new(0.0, 0.0)
                            );
                        }
                    }
                }
                (
                    key.codomain_tree().coupled(),
                    (0..(*rows).min(*columns))
                        .map(|diagonal| {
                            data[block.offset() + diagonal * (row_stride + column_stride)]
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };

    let assert_matches_dense = |source: &Tensor, actual: &Tensor, expected: &Tensor| {
        assert_compact_unmaterialized(actual);
        assert_eq!(actual.ordinary_body().space, expected.ordinary_body().space);
        assert_eq!(
            actual.ordinary_body().space.structure(),
            expected.ordinary_body().space.structure()
        );
        let Data::Diagonal(DiagonalData::C64(actual_values)) = actual.stored_data() else {
            panic!("expected compact c64 spectrum")
        };
        let expected_values = dense_diagonal_values(expected);
        assert_eq!(actual_values.len(), expected_values.len());
        for (actual, (expected_sector, expected)) in actual_values.iter().zip(&expected_values) {
            assert_eq!(actual.sector, *expected_sector);
            assert_eq!(actual.values.len(), expected.len());
            for (actual, expected) in actual.values.iter().zip(expected) {
                // Why not compare signed-zero bits: they are not part of compact
                // operation equivalence, and dense coefficient multiplication
                // can produce a different zero sign during replay.
                if *actual == Complex64::new(0.0, 0.0) && *expected == Complex64::new(0.0, 0.0) {
                    continue;
                }
                assert_eq!(actual.re.to_bits(), expected.re.to_bits());
                assert_eq!(actual.im.to_bits(), expected.im.to_bits());
            }
        }
        assert_compact_unmaterialized(source);
    };

    let permute_source = complex_diagonal(&runtime, &space, 484_001);
    let permute_oracle = complex_diagonal(&runtime, &space, 484_001)
        .densified_if_diagonal()
        .permute(&[1], &[0])
        .unwrap();
    let permuted = permute_source.permute(&[1], &[0]).unwrap();
    assert_matches_dense(&permute_source, &permuted, &permute_oracle);
    let restored = permuted.permute(&[1], &[0]).unwrap();
    assert!(matches!(restored.stored_data(), Data::Diagonal(_)));
    assert_eq!(
        restored.ordinary_body().space,
        permute_source.ordinary_body().space
    );
    assert_eq!(
        compact_spectrum_bits(&restored),
        compact_spectrum_bits(&permute_source)
    );

    let transpose_source = complex_diagonal(&runtime, &space, 484_002);
    let transpose_oracle = complex_diagonal(&runtime, &space, 484_002)
        .densified_if_diagonal()
        .transpose()
        .unwrap();
    let transposed = transpose_source.transpose().unwrap();
    assert_matches_dense(&transpose_source, &transposed, &transpose_oracle);
    let restored = transposed.transpose().unwrap();
    assert!(matches!(restored.stored_data(), Data::Diagonal(_)));
    assert_eq!(
        restored.ordinary_body().space,
        transpose_source.ordinary_body().space
    );
    assert_eq!(
        compact_spectrum_bits(&restored),
        compact_spectrum_bits(&transpose_source)
    );
}

#[test]
fn nonempty_trace_pairs_reduces_the_compact_spectrum_without_densifying() {
    // What: the full pair of a rank-(1,1) compact tensor is now reduced from
    // the stored spectrum (#585) and still agrees with the explicitly densified
    // route. The previous revision of this test pinned the opposite — that the
    // trace materialized — which was the live compact regression the #585
    // design audit found.
    //
    // There is no wider geometry to guard on this side: compact diagonal
    // storage only ever holds a rank-(1,1) bond, and an empty pair list returns
    // the tensor unchanged before any of this, so the full pair is the whole
    // reachable surface of the arm.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = product_space();
    let diagonal = complex_diagonal(&runtime, &space, 261_408);
    let oracle = diagonal.clone().densified_if_diagonal();

    let actual = diagonal.trace_pairs(&[(0, 1)]).unwrap();
    let expected = oracle.trace_pairs(&[(0, 1)]).unwrap();

    assert_tensor_close(&actual, &expected);
    assert_compact_unmaterialized(&diagonal);
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
    let Data::F64(data) = tensor.coupled_data().unwrap() else {
        panic!("expected f64 tensor")
    };
    let structure = tensor.ordinary_body().space.structure();
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
        Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap(),
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
                matches!(actual.stored_data(), Data::Diagonal(_)),
                "one-axis diagonal product must preserve compact storage"
            );
            assert!(!lhs.has_cached_materialization());
            assert!(!rhs.has_cached_materialization());

            let composed = lhs.compose(&rhs).unwrap();
            assert!(matches!(composed.stored_data(), Data::Diagonal(_)));
            assert!(!lhs.has_cached_materialization());
            assert!(!rhs.has_cached_materialization());

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
fn fermionic_diagonal_compose_multiplies_compact_values_without_supertrace_twist() {
    // What: TensorKit map composition of two compact fermionic diagonals stays
    // compact and equals the same coefficient-free coupled-block product.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    for space in [
        Space::fz2([(0, 2), (1, 2)]).unwrap().dual(),
        Space::fz2_u1_su2([((0, 0, 0), 2), ((1, 0, 1), 2)])
            .unwrap()
            .dual(),
    ] {
        let lhs = complex_diagonal(&rt, &space, 353_701);
        let rhs = complex_diagonal(&rt, &space, 353_702);
        let actual = lhs.compose(&rhs).unwrap();
        assert!(matches!(actual.stored_data(), Data::Diagonal(_)));
        assert!(!lhs.has_cached_materialization());
        assert!(!rhs.has_cached_materialization());
        let oracle = lhs
            .densified_if_diagonal()
            .compose(&rhs.densified_if_diagonal())
            .unwrap();

        assert_tensor_close(&actual, &oracle);
    }
}

#[test]
fn complex_diagonal_scales_dense_and_lazy_operands() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    for space in [
        Space::u1([(-1, 2), (0, 3), (1, 2)]),
        Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap(),
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
            assert_eq!(!diagonal.has_cached_materialization(), stays_compact);
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
fn diagonal_compose_materializes_an_accepted_lazy_operand_once() {
    // What: both canonical composition orders decide eligibility from lazy
    // metadata, then publish one owned adjoint body; an invalid bond leaves the
    // rejected lazy operand untouched.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::u1([(-1, 2), (0, 3), (2, 1)]);
    let diagonal = complex_diagonal(&rt, &space, 715);

    for diagonal_on_left in [false, true] {
        let parent = Tensor::rand_with_seed(&rt, Dtype::C64, [&space], [&space], 716).unwrap();
        let lazy = parent.adjoint().unwrap();
        let eager = Tensor::rand_with_seed(&rt, Dtype::C64, [&space], [&space], 716)
            .unwrap()
            .adjoint()
            .unwrap()
            .materialized_tensor()
            .unwrap();
        let (actual, expected) = if diagonal_on_left {
            (
                diagonal.compose(&lazy).unwrap(),
                dense_oracle(&diagonal).compose(&eager).unwrap(),
            )
        } else {
            (
                lazy.compose(&diagonal).unwrap(),
                eager.compose(&dense_oracle(&diagonal)).unwrap(),
            )
        };

        assert_tensor_close(&actual, &expected);
        assert_eq!(lazy.adjoint_body_builds(), 1);
    }

    let incompatible = Space::u1([(-1, 2), (0, 4), (2, 1)]);
    let parent = Tensor::zeros(&rt, Dtype::C64, [&incompatible], [&incompatible]).unwrap();
    let lazy = parent.adjoint().unwrap();
    let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();
    assert_eq!(
        diagonal.compose(&lazy).unwrap_err(),
        dense_oracle(&diagonal).compose(&eager).unwrap_err()
    );
    assert_eq!(lazy.adjoint_body_builds(), 0);
}

#[test]
fn ambiguous_diagonal_contractions_keep_dense_fallback() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::u1([(0, 4)]);
    let lhs = real_diagonal(&rt, &space, 721);
    let rhs = real_diagonal(&rt, &space, 722);

    let outer = lhs.contract(&rhs, &[], &[]).unwrap();
    assert!(!matches!(outer.stored_data(), Data::Diagonal(_)));
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
        Space::fz2([(0, 2), (1, 3)]).unwrap(),
        Space::su2([(0, 2), (1, 3), (2, 1)]).unwrap(),
        product_space(),
    ] {
        let lhs = real_diagonal(&rt, &space, 801);
        let rhs = real_diagonal(&rt, &space, 802);
        let lhs_dense = dense_oracle(&lhs);
        let rhs_dense = dense_oracle(&rhs);

        let adjoint = lhs.adjoint().unwrap();
        assert!(Arc::ptr_eq(
            &adjoint.ordinary_body().data,
            &lhs.ordinary_body().data
        ));
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
        assert!(!lhs.has_cached_materialization());
        assert!(!rhs.has_cached_materialization());

        for legs in [&[0][..], &[1][..], &[0, 1][..]] {
            let twisted = lhs.twist(legs).unwrap();
            assert_compact_unmaterialized(&twisted);
            assert_tensor_close(&twisted, &lhs_dense.twist(legs).unwrap());
            assert!(!lhs.has_cached_materialization());
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
        Space::fz2([(0, 2), (1, 3)]).unwrap(),
        Space::su2([(0, 2), (1, 3), (2, 1)]).unwrap(),
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
                Arc::ptr_eq(&adjoint.ordinary_body().data, &lhs.ordinary_body().data),
                matches!(lhs.stored_data(), Data::Diagonal(DiagonalData::RealC64(_)))
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
            assert!(!lhs.has_cached_materialization());
            assert!(!rhs.has_cached_materialization());

            for legs in [&[0][..], &[1][..], &[0, 1][..]] {
                let twisted = lhs.twist(legs).unwrap();
                assert_compact_unmaterialized(&twisted);
                assert_tensor_close(&twisted, &lhs_dense.twist(legs).unwrap());
                assert!(!lhs.has_cached_materialization());
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
        Space::fz2([(0, 2), (1, 3)]).unwrap(),
        Space::su2([(0, 2), (1, 3), (2, 1)]).unwrap(),
        product_space(),
    ] {
        let diagonal = real_diagonal(&rt, &space, 821);
        let diagonal_dense = dense_oracle(&diagonal);
        let dense = Tensor::rand_with_seed(&rt, Dtype::F64, [&space], [&space], 822).unwrap();
        let lazy = dense.adjoint().unwrap();
        for operand in [&dense, &lazy] {
            let operand_dense = Tensor::owned(
                operand.rt.clone(),
                Arc::clone(&operand.materialized_body().unwrap().space),
                Arc::new(operand.coupled_data().unwrap().clone()),
            );
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
            assert!(!diagonal.has_cached_materialization());
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
            assert!(!diagonal.has_cached_materialization());
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
        Space::su2([(0, 2), (1, 1)]).unwrap(),
        Space::fz2([(0, 3)]).unwrap(),
    ] {
        let diagonal = real_diagonal(&rt, &space, 841);
        let twisted = diagonal.twist(&[0]).unwrap();
        assert!(Arc::ptr_eq(
            &twisted.ordinary_body().data,
            &diagonal.ordinary_body().data
        ));
        assert_compact_unmaterialized(&diagonal);
        assert_compact_unmaterialized(&twisted);
    }

    let odd = Space::fz2([(1, 3)]).unwrap();
    let diagonal = real_diagonal(&rt, &odd, 842);
    let twisted = diagonal.twist(&[0]).unwrap();
    assert!(!Arc::ptr_eq(
        &twisted.ordinary_body().data,
        &diagonal.ordinary_body().data
    ));
    assert_compact_unmaterialized(&diagonal);
    assert_compact_unmaterialized(&twisted);
}

#[test]
fn su3_compact_storage_ops_and_fallback_boundaries_are_explicit() {
    // What: storage-local SU(3) operations and ordinary trace remain compact;
    // norm uses the same storage-local quantum-dimension reduction.
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
        assert!(Arc::ptr_eq(
            &adjoint.ordinary_body().data,
            &diagonal.ordinary_body().data
        ));

        let norm_input = diagonal.clone();
        assert!(!norm_input.has_cached_materialization());
        assert!(norm_input.norm().unwrap().is_finite());
        assert!(!norm_input.has_cached_materialization());

        let trace_input = diagonal.clone();
        assert!(!trace_input.has_cached_materialization());
        assert!(trace_input.tr().unwrap().to_c64().norm().is_finite());
        assert!(!trace_input.has_cached_materialization());
    }
}

#[test]
fn su3_compact_norm_inner_and_dot_match_dense_oracles_without_materialization() {
    // What: Generic compact diagonal norm/inner/dot reduce stored spectra with
    // the same quantum-dimension weighting as TensorKit, without densifying the
    // compact operand.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::su3([((1, 0), 2), ((0, 1), 2), ((1, 1), 2)]).unwrap();

    for dtype in [Dtype::F64, Dtype::C64] {
        let lhs = su3_diagonal(&rt, dtype, &space, 317_001);
        let rhs = su3_diagonal(&rt, dtype, &space, 317_002);
        let has_dim8_multiplicity = match lhs.stored_data() {
            Data::Diagonal(DiagonalData::RealF64(spectrum)) => spectrum.iter().any(|entry| {
                let sqrt = lhs.su3_rule().sqrt_dim_scalar(entry.sector);
                (sqrt * sqrt - 8.0).abs() < 1e-12 && entry.values.len() >= 2
            }),
            Data::Diagonal(DiagonalData::RealC64(spectrum)) => spectrum.iter().any(|entry| {
                let sqrt = lhs.su3_rule().sqrt_dim_scalar(entry.sector);
                (sqrt * sqrt - 8.0).abs() < 1e-12 && entry.values.len() >= 2
            }),
            _ => false,
        };
        assert!(has_dim8_multiplicity);

        let expected_norm = su3_diagonal(&rt, dtype, &space, 317_001)
            .densified_if_diagonal()
            .norm()
            .unwrap();
        let actual_norm = lhs.norm().unwrap();
        assert!((actual_norm - expected_norm).abs() < 1e-11);
        let direct_norm = compact_su3_inner_oracle(&lhs, &lhs).re.sqrt();
        assert!((actual_norm - direct_norm).abs() < 1e-11);
        assert!(!lhs.has_cached_materialization());

        let expected_inner = {
            let lhs_dense = su3_diagonal(&rt, dtype, &space, 317_001).densified_if_diagonal();
            let rhs_dense = su3_diagonal(&rt, dtype, &space, 317_002).densified_if_diagonal();
            lhs_dense.inner(&rhs_dense).unwrap()
        };
        let actual_inner = lhs.inner(&rhs).unwrap();
        assert!((actual_inner.to_c64() - expected_inner.to_c64()).norm() < 1e-11);
        let direct_inner = compact_su3_inner_oracle(&lhs, &rhs);
        assert!((actual_inner.to_c64() - direct_inner).norm() < 1e-11);
        assert_eq!(lhs.dot(&rhs).unwrap().to_c64(), actual_inner.to_c64());
        assert!(!lhs.has_cached_materialization());
        assert!(!rhs.has_cached_materialization());

        let dense_rhs = Tensor::rand_with_seed(&rt, dtype, [&space], [&space], 317_003).unwrap();
        let expected_left = {
            let lhs_dense = su3_diagonal(&rt, dtype, &space, 317_001).densified_if_diagonal();
            lhs_dense.inner(&dense_rhs).unwrap()
        };
        let actual_left = lhs.inner(&dense_rhs).unwrap();
        assert!((actual_left.to_c64() - expected_left.to_c64()).norm() < 1e-11);
        assert!(!lhs.has_cached_materialization());

        let expected_right = {
            let rhs_dense = su3_diagonal(&rt, dtype, &space, 317_002).densified_if_diagonal();
            dense_rhs.inner(&rhs_dense).unwrap()
        };
        let actual_right = dense_rhs.inner(&rhs).unwrap();
        assert!((actual_right.to_c64() - expected_right.to_c64()).norm() < 1e-11);
        if dtype == Dtype::C64 {
            assert!(
                (actual_left.to_c64() - actual_right.to_c64()).im.abs() > 1e-12,
                "fixture must distinguish C64 conjugation order"
            );
        }
        assert!(!rhs.has_cached_materialization());
    }
}

#[test]
fn nonselfdual_odd_product_routes_match_dense_oracles_and_storage() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::product([((1, 1), 1), ((-1, 1), 1)]).unwrap();
    let plus = Space::product([((1, 1), 1)]).unwrap().sectors[0].0;
    let minus = Space::product([((-1, 1), 1)]).unwrap().sectors[0].0;
    let probe = fixed_diagonal(&rt, &space, plus, 2.0, 3.0);
    let homspace = probe.ordinary_body().space.homspace();
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
                        matches!(actual.stored_data(), Data::Diagonal(_)),
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
    assert!(!matches!(actual.stored_data(), Data::Diagonal(_)));
    assert!(diagonal.has_cached_materialization());
    let expected = dense_dual
        .contract(&diagonal.densified_if_diagonal(), &[1], &[1])
        .unwrap();
    assert_eq!(scalar_block_by_codomain_sector(&expected, minus), 14.0);
    assert_eq!(scalar_block_by_codomain_sector(&expected, plus), 15.0);
    assert_tensor_close(&actual, &expected);

    let diagonal = fixed_diagonal(&rt, &space, plus, 2.0, 3.0);
    let actual = diagonal.contract(&dense_dual, &[1], &[1]).unwrap();
    assert!(!matches!(actual.stored_data(), Data::Diagonal(_)));
    assert!(diagonal.has_cached_materialization());
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

#[test]
fn fermionic_adjoint_diagonal_contractions_match_dense_oracles() {
    // What: lhs dagger, rhs dagger, and both preserve the odd-sector contraction sign.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let spaces = [
        Space::fz2([(0, 2), (1, 2)]).unwrap(),
        Space::product([((0, 0), 2), ((1, 1), 2)]).unwrap(),
    ];

    for space in spaces {
        for (lhs_adjoint, rhs_adjoint) in [(true, false), (false, true), (true, true)] {
            let lhs = complex_diagonal(&rt, &space, 261_201);
            let rhs = complex_diagonal(&rt, &space, 261_202);
            let lhs = if lhs_adjoint {
                lhs.adjoint().unwrap()
            } else {
                lhs
            };
            let rhs = if rhs_adjoint {
                rhs.adjoint().unwrap()
            } else {
                rhs
            };

            let actual = lhs.contract(&rhs, &[1], &[0]).unwrap();
            let expected = dense_oracle(&lhs)
                .contract(&dense_oracle(&rhs), &[1], &[0])
                .unwrap();
            assert_tensor_close(&actual, &expected);
        }
    }
}

#[test]
fn fermionic_odd_diagonal_contraction_matches_materialized_adjoint() {
    // What: compact contraction makes the same fZ2 odd-sector twist decision
    // for a lazy adjoint and its materialized form.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let odd = Space::fz2([(1, 2)]).unwrap();
    let diagonal = real_diagonal(&rt, &odd, 477_101);
    let parent = Tensor::rand_with_seed(&rt, Dtype::F64, [&odd], [&odd], 477_102).unwrap();
    let lazy = parent.adjoint().unwrap();

    let actual = diagonal.contract(&lazy, &[1], &[0]).unwrap();
    let expected = dense_oracle(&diagonal)
        .contract(&lazy.materialized_tensor().unwrap(), &[1], &[0])
        .unwrap();

    assert_tensor_close(&actual, &expected);
}

#[test]
fn absorb_densifies_a_compact_destination_before_prefix_overwrite() {
    // What: absorb returns ordinary dense storage because overwriting a block
    // prefix need not preserve the destination's compact diagonal invariant.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = product_space();
    let diagonal = real_diagonal(&runtime, &space, 395_010);
    let codomain = diagonal.codomain_spaces();
    let domain = diagonal.domain_spaces();
    let source = Tensor::from_block_fn(&runtime, codomain.iter(), domain.iter(), |_, indices| {
        (1 + indices.iter().sum::<usize>()) as f64
    })
    .unwrap();

    assert_compact_unmaterialized(&diagonal);
    let actual = diagonal.absorb(&source).unwrap();

    assert!(!matches!(actual.stored_data(), Data::Diagonal(_)));
    assert!(diagonal.has_cached_materialization());
    assert_eq!(actual.data(), source.data());
}

/// The single value of a rank-0 result, widened so f64 and c64 fixtures share
/// one assertion.
fn rank_zero_value(tensor: &Tensor) -> Complex64 {
    match tensor.scalar().unwrap() {
        Scalar::F64(value) => Complex64::new(value, 0.0),
        Scalar::C64(value) => value,
    }
}

fn assert_scalar_close(actual: Complex64, expected: Complex64, label: &str) {
    assert!(
        (actual - expected).norm() <= 1e-12 * expected.norm().max(1.0),
        "{label}: {actual:?} vs {expected:?}"
    );
}

#[test]
fn full_rank_one_trace_pairs_matches_the_dense_route_on_every_rule() {
    // What: the categorical trace of a rank-(1,1) tensor over its only pair is
    // the same number whichever storage it is read from — for a self-dual
    // bosonic rule, for one with a non-unit quantum dimension, for a fermionic
    // one whose twist makes it a supertrace, and for a dual leg, which is what
    // decides the orientation the coefficients are taken in.
    //
    // Both pair orders are swept: `(0, 1)` and `(1, 0)` name the same pair of
    // legs and must trace to the same number.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let spaces = [
        ("u1", Space::u1([(-1, 2), (0, 3), (1, 2)])),
        (
            "cu1",
            Space::cu1([((0, 0), 2), ((0, 1), 3), ((1, 2), 2)]).unwrap(),
        ),
        ("su2", Space::su2([(0, 2), (1, 3)]).unwrap()),
        ("fz2", Space::fz2([(0, 2), (1, 3)]).unwrap()),
        ("product", product_space()),
    ];
    for (name, space) in spaces {
        for dual in [false, true] {
            let space = if dual { space.dual() } else { space.clone() };
            for (dtype, diagonal) in [
                ("f64", real_diagonal(&runtime, &space, 585_001)),
                ("c64", complex_diagonal(&runtime, &space, 585_002)),
                ("real-c64", real_c64_diagonal(&runtime, &space, 585_003)),
                // The bond space a compact swap produces, so the arm is also
                // read on legs it bent itself rather than only on freshly
                // factorized ones.
                (
                    "transposed c64",
                    complex_diagonal(&runtime, &space, 585_004)
                        .transpose()
                        .unwrap(),
                ),
                (
                    "permuted c64",
                    complex_diagonal(&runtime, &space, 585_005)
                        .permute(&[1], &[0])
                        .unwrap(),
                ),
                // Not the excluded adjoint-*view* path — `adjoint` of compact
                // storage short-circuits to an owned conjugated spectrum, so
                // this is the compact arm again, on conjugated values.
                (
                    "adjoint c64",
                    complex_diagonal(&runtime, &space, 585_006)
                        .adjoint()
                        .unwrap(),
                ),
            ] {
                let oracle = dense_oracle(&diagonal);
                for pairs in [[(0usize, 1usize)], [(1, 0)]] {
                    let actual = diagonal.trace_pairs(&pairs).unwrap();
                    let expected = oracle.trace_pairs(&pairs).unwrap();
                    assert_eq!(actual.dtype(), expected.dtype());
                    assert_scalar_close(
                        rank_zero_value(&actual),
                        rank_zero_value(&expected),
                        &format!("{name} dual={dual} {dtype} {pairs:?}"),
                    );
                }
                assert_compact_unmaterialized(&diagonal);
            }
        }
    }
}

#[test]
fn full_rank_one_trace_pairs_is_the_supertrace_for_a_fermionic_rule() {
    // What: `trace_pairs` is the categorical trace with the fermionic twist —
    // unlike `tr`, which is TensorKit's positive matrix trace. On a single
    // fermion-parity sector the two differ by exactly the twist, so an odd
    // space flips the sign and an even one does not. The bosonic Z2 twin of the
    // odd space pins that the sign is the *twist* and not the parity label.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (name, space, sign) in [
        ("fz2 even", Space::fz2([(0, 3)]).unwrap(), 1.0),
        ("fz2 odd", Space::fz2([(1, 3)]).unwrap(), -1.0),
        ("z2 odd", Space::z2([(1, 3)]), 1.0),
    ] {
        let diagonal = complex_diagonal(&runtime, &space, 585_011);
        let Scalar::C64(trace) = diagonal.tr().unwrap() else {
            panic!("a c64 spectrum traces to a c64 scalar");
        };
        assert_scalar_close(
            rank_zero_value(&diagonal.trace_pairs(&[(0, 1)]).unwrap()),
            trace * sign,
            name,
        );

        // And the orientation is load-bearing, not incidental: a planar
        // transpose duals both bond legs, and `tensortrace` twists a traced
        // channel only where its leg is *not* dual, so the same tensor traces
        // without the fermionic sign once transposed. A coefficient that read
        // the sector alone would return the supertrace here too.
        let transposed = diagonal.transpose().unwrap();
        let Scalar::C64(transposed_trace) = transposed.tr().unwrap() else {
            panic!("a c64 spectrum traces to a c64 scalar");
        };
        assert_scalar_close(
            rank_zero_value(&transposed.trace_pairs(&[(0, 1)]).unwrap()),
            transposed_trace,
            &format!("{name} transposed"),
        );
    }
}

#[test]
fn compact_is_posdef_matches_the_dense_route_and_never_materializes() {
    // What: reading the stored spectrum answers exactly what eigendecomposing
    // the materialization answers — on positive, positive-*semi*definite,
    // negative and indefinite spectra, at every tolerance, for both the real
    // and the real-valued-c64 storage. The strictness at zero is the one place
    // a `>=` would pass a semidefinite spectrum the dense route rejects.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = product_space();

    // A constant block is rank one, so all but one singular value per sector is
    // zero: the positive-semidefinite fixture.
    let semidefinite = Tensor::from_block_fn(&runtime, [&space], [&space], |_, _| 1.0)
        .unwrap()
        .svd_compact()
        .unwrap()
        .1;
    for (name, diagonal) in [
        ("positive f64", real_diagonal(&runtime, &space, 585_021)),
        ("positive c64", real_c64_diagonal(&runtime, &space, 585_022)),
        ("semidefinite", semidefinite.clone()),
        (
            "negative",
            real_diagonal(&runtime, &space, 585_023)
                .scale(-1.0)
                .unwrap(),
        ),
        (
            "indefinite",
            real_diagonal(&runtime, &space, 585_024)
                .add(&semidefinite, 1.0, -3.0)
                .unwrap(),
        ),
    ] {
        let oracle = dense_oracle(&diagonal);
        for tol in [0.0, 1e-14, 1e-8, 1e-3, 0.5] {
            assert_eq!(
                diagonal.is_posdef(tol).unwrap(),
                oracle.is_posdef(tol).unwrap(),
                "{name} at tol {tol}"
            );
        }
        assert_compact_unmaterialized(&diagonal);
    }

    assert!(real_diagonal(&runtime, &space, 585_021)
        .is_posdef(0.0)
        .unwrap());
    assert!(!semidefinite.is_posdef(0.0).unwrap());

    // The Hermiticity gate is load-bearing and is checked separately, because
    // every fixture above would answer the same with or without it. A spectrum
    // whose values all have a positive real part *and* a non-negligible
    // imaginary one is positive definite to a predicate that reads only the
    // stored real parts, and is not Hermitian at all — so the gate is the only
    // thing standing between the compact arm and a wrong `true`.
    let skewed = real_c64_diagonal(&runtime, &space, 585_025)
        .scale_c64(Complex64::new(1.0, 1.0))
        .unwrap();
    assert!(!skewed.is_hermitian(1e-10).unwrap());
    let oracle = dense_oracle(&skewed);
    for tol in [0.0, 1e-14, 1e-8, 1e-3] {
        assert!(!skewed.is_posdef(tol).unwrap(), "skewed at tol {tol}");
        assert_eq!(
            skewed.is_posdef(tol).unwrap(),
            oracle.is_posdef(tol).unwrap(),
            "skewed at tol {tol}"
        );
    }
    assert_compact_unmaterialized(&skewed);
}

/// Asserts `image` is the elementwise `exp` of the compact `source`, variant
/// for variant. The variant pairing is half the assertion: `RealC64 -> C64`
/// would double the stored payload of a real spectrum inside a c64 tensor, and
/// a dense materialization comparison could not see it.
fn assert_elementwise_exp(source: &Tensor, image: &Tensor) {
    fn real(source: &[SectorSpectrum<f64>], image: &[SectorSpectrum<f64>]) {
        assert_eq!(source.len(), image.len());
        for (source, image) in source.iter().zip(image) {
            assert_eq!(source.sector, image.sector);
            assert_eq!(source.values.len(), image.values.len());
            for (&source, &image) in source.values.iter().zip(&image.values) {
                assert!(
                    (image - source.exp()).abs() < 1e-11,
                    "{image} is not exp({source})"
                );
            }
        }
    }
    match (source.stored_data(), image.stored_data()) {
        (
            Data::Diagonal(DiagonalData::RealF64(source)),
            Data::Diagonal(DiagonalData::RealF64(image)),
        )
        | (
            Data::Diagonal(DiagonalData::RealC64(source)),
            Data::Diagonal(DiagonalData::RealC64(image)),
        ) => real(source, image),
        (Data::Diagonal(DiagonalData::C64(source)), Data::Diagonal(DiagonalData::C64(image))) => {
            assert_eq!(source.len(), image.len());
            for (source, image) in source.iter().zip(image) {
                assert_eq!(source.sector, image.sector);
                assert_eq!(source.values.len(), image.values.len());
                for (&source, &image) in source.values.iter().zip(&image.values) {
                    assert!(
                        (image - source.exp()).norm() < 1e-11,
                        "{image} is not exp({source})"
                    );
                }
            }
        }
        pair => panic!("exp did not preserve the compact variant: {pair:?}"),
    }
}

#[test]
fn compact_exp_is_elementwise_and_matches_the_dense_route() {
    // What: issue #578. TensorKit's `exp(::DiagonalTensorMap)`
    // (`tensors/diagonal.jl:383-390`) is elementwise on the stored values and
    // stays diagonal; the erased facade used to densify and run the Hermitian
    // eigendecomposition on the block-diagonal buffer. The values must not
    // move — the densified twin is the oracle — while the storage does.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    for space in [
        Space::u1([(-2, 1), (1, 3)]),
        Space::fz2([(0, 2), (1, 3)]).unwrap(),
        Space::su2([(0, 2), (1, 3), (2, 1)]).unwrap(),
        product_space(),
    ] {
        // An f64 SVD spectrum and its c64 sibling (`RealC64`: a real spectrum
        // inside a complex tensor). Both are Hermitian as matrices, so the
        // dense route accepts them and can serve as the value oracle.
        for source in [
            real_diagonal(&rt, &space, 578_001),
            real_c64_diagonal(&rt, &space, 578_002),
        ] {
            let dense = dense_oracle(&source);
            let image = source.exp().unwrap();
            assert_compact_unmaterialized(&image);
            assert_tensor_close(&image, &dense.exp().unwrap());
            assert_elementwise_exp(&source, &image);
            // The source is untouched: same values, still compact, and never
            // materialized behind our back.
            assert_compact_unmaterialized(&source);
            assert_tensor_close(&source, &dense);

            // A later compact contraction still folds to a bond scaling
            // instead of a GEMM, and `exp(s) * exp(s) = exp(2s)` there.
            let squared = image.compose(&image).unwrap();
            assert_compact_unmaterialized(&squared);
            assert_tensor_close(
                &squared,
                &dense.exp().unwrap().compose(&dense.exp().unwrap()).unwrap(),
            );
        }

        // A genuinely complex spectrum. This used to be where storage decided
        // what `exp` *accepts* — the dense route was the Hermitian spectral
        // function and refused the matrix. Issue #577 gave it the general Padé
        // arm, so storage now decides only how `exp` is computed, and the two
        // must agree: Padé of a diagonal matrix is the elementwise exponential.
        let source = complex_diagonal(&rt, &space, 578_003);
        let image = source.exp().unwrap();
        assert_tensor_close(&image, &dense_oracle(&source).exp().unwrap());
        assert_compact_unmaterialized(&image);
        assert_elementwise_exp(&source, &image);
        assert_compact_unmaterialized(&source);
    }
}

#[test]
fn compact_su3_exp_stays_compact_before_the_unwired_su3_rejection() {
    // What: the compact arm sits *before* `reject_unwired_su3` in `exp`, and
    // must: a Generic SU(3) spectrum has no dense route at all (the firewall
    // refuses it, see `su3_panic_firewall.rs`), so ordering the arm after the
    // rejection would turn an operation that is pure elementwise arithmetic on
    // stored values — needing no fusion wiring whatsoever — into an error.
    // Same placement as the `inv`/`pinv` compact arms.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let space = Space::su3([((1, 0), 2), ((0, 1), 2), ((1, 1), 2)]).unwrap();
    for dtype in [Dtype::F64, Dtype::C64] {
        let source = su3_diagonal(&rt, dtype, &space, 578_031);
        let image = source.exp().unwrap();
        assert_compact_unmaterialized(&image);
        assert_eq!(image.dtype(), source.dtype());
        assert_elementwise_exp(&source, &image);
        assert_compact_unmaterialized(&source);
    }
}

#[test]
fn exp_of_an_adjoint_conjugates_the_spectrum_and_never_reaches_an_adjoint_view() {
    // What: the `is_adjoint_view` materialization guard at the top of `exp`.
    // Both of `exp`'s arms below it read `ordinary_body()`, which panics on a
    // view, so the guard is what makes `exp` defined on an adjoint at all.
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let space = product_space();

    // Compact side: `adjoint` of compact storage short-circuits to an owned
    // conjugated spectrum, so `exp` sees an ordinary compact tensor and is
    // elementwise on the conjugated values.
    let source = complex_diagonal(&rt, &space, 578_041);
    let adjoint = source.adjoint().unwrap();
    assert!(!adjoint.is_adjoint_view());
    let image = adjoint.exp().unwrap();
    assert_compact_unmaterialized(&image);
    assert_elementwise_exp(&adjoint, &image);
    let (
        Data::Diagonal(DiagonalData::C64(source_values)),
        Data::Diagonal(DiagonalData::C64(image_values)),
    ) = (source.stored_data(), image.stored_data())
    else {
        panic!("expected compact c64 spectra")
    };
    assert_eq!(source_values.len(), image_values.len());
    for (source, image) in source_values.iter().zip(image_values) {
        assert_eq!(source.sector, image.sector);
        assert_eq!(source.values.len(), image.values.len());
        for (&source, &image) in source.values.iter().zip(&image.values) {
            assert!(
                (image - source.conj().exp()).norm() < 1e-11,
                "{image} is not exp(conj({source}))"
            );
        }
    }
    assert_compact_unmaterialized(&source);

    // Dense side: a genuine adjoint view. Without the guard this reaches
    // `exp_impl` and panics instead of exponentiating the adjoint.
    let random = Tensor::rand_with_seed(&rt, Dtype::C64, [&space], [&space], 578_042).unwrap();
    let hermitian = random.adjoint().unwrap().compose(&random).unwrap();
    let view = hermitian.adjoint().unwrap();
    assert!(view.is_adjoint_view());
    assert_tensor_close(
        &view.exp().unwrap(),
        &view.materialized_tensor().unwrap().exp().unwrap(),
    );
}
