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
    assert_eq!(
        actual.logical_space().unwrap(),
        expected.logical_space().unwrap()
    );
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
fn nonempty_trace_pairs_keeps_the_existing_diagonal_densification_boundary() {
    // What: partial trace of compact storage remains the established dense
    // fallback and agrees with an explicitly densified tensor.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = product_space();
    let diagonal = complex_diagonal(&runtime, &space, 261_408);
    let oracle = diagonal.clone().densified_if_diagonal();

    let actual = diagonal.trace_pairs(&[(0, 1)]).unwrap();
    let expected = oracle.trace_pairs(&[(0, 1)]).unwrap();

    assert_tensor_close(&actual, &expected);
    assert!(diagonal.has_cached_materialization());
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
